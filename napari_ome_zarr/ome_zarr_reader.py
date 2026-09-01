# zarr v3

from abc import ABC
from typing import Any, Callable, Dict, Iterable, List, Tuple
from xml.etree import ElementTree as ET

import dask.array as da
import numpy as np
import transformnd as tnd
import zarr
from napari.utils.colormaps import AVAILABLE_COLORMAPS, Colormap
from ome_zarr import OMEZarrLabels, OMEZarrMultiscale, OMEZarrScene
from zarr import Group
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import SyncMixin

from .plate import get_first_field_path, get_first_well, get_pyramid_lazy

# StrDict = Dict[str, Any]
# LayerData = Union[Tuple[Any], Tuple[Any, StrDict], Tuple[Any, StrDict, str]]
LayerData = Tuple[List[da.core.Array], Dict[str, Any], str]

AXES_TYPES = {"x": "space", "y": "space", "z": "space", "c": "channel", "t": "time"}
AXES_5D = [
    {"name": "t", "type": "time"},
    {"name": "c", "type": "channel"},
    {"name": "z", "type": "space"},
    {"name": "y", "type": "space"},
    {"name": "x", "type": "space"},
]


def _match_colors_to_available_colormap(custom_cmap: Colormap) -> Colormap:
    """Helper function to match Colormap to an existing napari Colormap.
    If the colormap matches, return the specific napari Colormap, otherwise return the
    the original Colormap.
    """
    for available_cmap in AVAILABLE_COLORMAPS.values():
        if (
            np.array_equal(available_cmap.controls, custom_cmap.controls)
            and np.array_equal(available_cmap.colors, custom_cmap.colors)
            and available_cmap.interpolation == custom_cmap.interpolation
        ):
            custom_cmap = available_cmap
            break

    return custom_cmap


def _ome_zarr_ms_to_layer_props(
    multiscales: OMEZarrMultiscale | OMEZarrLabels,
    channel_index: int | None,
    inserted_defaults: list[dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    Helper function to extract properties from an OME-Zarr
    multiscale or label images that would apply to all channels
    (if present) alike.
    """

    # get scale (same for all channels)
    s = list(multiscales.images[0].scale.values())
    scale = (
        s[:channel_index] + s[channel_index + 1 :] if channel_index is not None else s
    )
    props: Dict[str, Any] = {}
    if multiscales.images[0].axes_units:
        props["units"] = tuple(multiscales.images[0].axes_units.values())

    props["axis_labels"] = tuple([ax for ax in multiscales.images[0].axes if ax != "c"])
    props["scale"] = scale
    props["name"] = multiscales.name

    if inserted_defaults is not None:
        for item in inserted_defaults:
            if "scale" in item:
                scale.insert(item["index"], item["scale"])
            if "axis_labels" in item:
                axis_labels = list(props["axis_labels"])
                axis_labels.insert(item["index"], item["axis_labels"])
                props["axis_labels"] = tuple(axis_labels)
            if "units" in item:
                units = list(props["units"])
                units.insert(item["index"], item["units"])
                props["units"] = tuple(units)

    return props


def _strip_channel_from_affine(
    affine: np.ndarray,
    input_ch_idx: int | None,
    output_ch_idx: int | None,
) -> np.ndarray:
    """Remove channel row/col from affine matrix."""
    if input_ch_idx is not None:
        affine = np.delete(affine, input_ch_idx, axis=1)
    if output_ch_idx is not None:
        affine = np.delete(affine, output_ch_idx, axis=0)
    return affine


def _expand_affine_for_projection(
    seq: tnd.TransformSequence,
) -> np.ndarray:
    """
    Handle affine expansion when ProjectAxis adds dimensions.

    Transforms before ProjectAxis need extra columns inserted so matrix
    multiplication works after the axis projection.
    """
    transform_sequence_flat = seq.flatten()

    # Find ProjectAxis transform
    project_idx = next(
        (
            i
            for i, tf in enumerate(transform_sequence_flat)
            if isinstance(tf, tnd.transforms.ProjectAxis)
        ),
        None,
    )

    if project_idx is None or project_idx == 0:
        seq = tnd.TransformSequence(transform_sequence_flat.transforms[1:])
        return seq.simplify().to_affine().matrix

    created_output_idxs = transform_sequence_flat[project_idx].created
    updated_transforms = []

    # Expand transforms before ProjectAxis
    for tf in transform_sequence_flat[:project_idx]:
        single_affine = tf.to_affine().matrix
        for output_idx in created_output_idxs:
            single_affine = np.insert(
                single_affine, output_idx, np.eye(single_affine.shape[0])[:, 0], axis=1
            )
        updated_transforms.append(tnd.transforms.Affine(single_affine))

    # Keep transforms after ProjectAxis as-is
    updated_transforms.extend(transform_sequence_flat[project_idx + 1 :])

    return tnd.TransformSequence(updated_transforms).simplify().to_affine().matrix


def _extract_channel_props(
    multiscales: OMEZarrMultiscale | OMEZarrLabels,
) -> Dict[str, Any] | None:
    """
    Helper function to extract channel properties from an OME-Zarr
    multiscale or label images that would apply per channel
    (if present).
    """

    props: Dict[str, Any] | None = None
    if hasattr(multiscales, "omero") and multiscales.omero is not None:
        omero = multiscales.omero.model_dump()
        colormaps = []
        ch_names = []
        visibles = []
        contrast_limits: list = []
        model = omero.get("rdefs", {}).get("model", "unset")
        greyscale = model == "greyscale"

        for index, ch in enumerate(omero["channels"]):
            color = ch.get("color", None)
            if color is not None:
                rgb = [(int(color[i : i + 2], 16) / 255) for i in range(0, 6, 2)]
                if greyscale:
                    rgb = [1, 1, 1]
                # colormap is range: black -> rgb color
                cm = Colormap([[0, 0, 0], rgb])
                # Try to match colormap to an existing napari colormap
                cm = _match_colors_to_available_colormap(cm)
                colormaps.append(cm)
            ch_name = ch.get("label", f"channel_{index}")
            ch_names.append(
                multiscales.name and f"{multiscales.name}: {ch_name}" or ch_name
            )
            visibles.append(ch.get("active", True))

            window = ch.get("window", None)
            if window is not None:
                start = window.get("start", None)
                end = window.get("end", None)
                if start is not None and end is not None:
                    # skip if None. Otherwise check no previous skip
                    if len(contrast_limits) == index:
                        contrast_limits.append([start, end])

        if len(colormaps) == 1:
            colormaps = colormaps[0]  # type: ignore
        if len(visibles) == 1:
            visibles = visibles[0]  # type: ignore
        if len(contrast_limits) == 1:
            contrast_limits = contrast_limits[0]  # type: ignore
        if len(ch_names) == 1:
            ch_names = ch_names[0]  # type: ignore
        props = {
            "colormap": colormaps,
            "name": ch_names,
            "visible": visibles,
            "contrast_limits": contrast_limits,
        }

    return props


class Spec(ABC):
    def __init__(self, group: Group) -> None:
        self.group = group
        self.parent_transforms: List[Dict[str, Any]] = []

    @staticmethod
    def matches(group: Group) -> bool:
        return False

    def data(self) -> List[da.core.Array]:
        return []

    def metadata(self) -> Dict[str, Any]:
        # napari layer metadata
        return {}

    def to_layer_data(self) -> List[LayerData]:
        return []

    def children(self) -> list["Spec"]:
        return []

    def iter_nodes(self) -> Iterable["Spec"]:
        yield self
        for child in self.children():
            yield from child.iter_nodes()

    def iter_data(self) -> Iterable[da.core.Array]:
        for node in self.iter_nodes():
            data = node.data()
            if data:
                yield data

    @staticmethod
    def get_attrs(group: Group) -> dict:
        if "ome" in group.attrs:
            return group.attrs["ome"]
        return group.attrs


class Multiscales(Spec):
    @staticmethod
    def matches(group: Group) -> bool:
        return "multiscales" in Spec.get_attrs(group)

    def to_layer_data(self) -> List[LayerData]:
        ms = OMEZarrMultiscale.from_ome_zarr(self.group)

        data = [img.data for img in ms.images]

        has_channel = "c" in ms.images[0].axes
        channel_index = ms.images[0].axes.index("c") if has_channel else None

        # Get image-specific properties
        # (channel axis removed from scale/units/axis_labels)
        props = _ome_zarr_ms_to_layer_props(ms, channel_index)

        # Tell napari where the channel axis is
        if has_channel:
            props["channel_axis"] = channel_index

        # Merge channel-specific properties (colormaps, names, visible, contrast_limits)
        channel_props = _extract_channel_props(ms)
        if channel_props is not None:
            props |= channel_props

        layers: List[LayerData] = [(data, props, "image")]

        if ms.labels is not None:
            for label_key in ms.labels.keys():
                label_spec = Label(self.group[f"labels/{label_key}"])
                layers.extend(label_spec.to_layer_data())

        return layers

    def _splits_channels(self) -> bool:
        """Whether a channel axis is turned into separate napari layers.

        Images split into one layer per channel via ``channel_axis``, so the
        channel axis is dropped from the per-axis metadata (axis_labels, units,
        scale, translate) to match each split layer's reduced ndim. Labels keep
        every axis in a single layer and so must keep the channel axis (see
        ``Label._splits_channels``).
        """
        return True


class Bioformats2raw(Spec):
    @staticmethod
    def matches(group: Group) -> bool:
        attrs = Spec.get_attrs(group)
        # Don't consider "plate" as a Bioformats2raw layout
        return "bioformats2raw.layout" in attrs and "plate" not in attrs

    def children(self) -> list[Spec]:
        # lookup children from series of OME/METADATA.xml
        xml_data = SyncMixin()._sync(
            self.group.store.get(
                "OME/METADATA.ome.xml", prototype=default_buffer_prototype()
            )
        )
        root = ET.fromstring(xml_data.to_bytes())
        rv: list[Spec] = []
        for child in root:
            # {http://www.openmicroscopy.org/Schemas/OME/2016-06}Image
            node_id = child.attrib.get("ID", "")
            if child.tag.endswith("Image") and node_id.startswith("Image:"):
                image_path = node_id.replace("Image:", "")
                g = self.group[image_path]
                if Multiscales.matches(g):
                    rv.append(Multiscales(g))
        return rv

    # override to NOT yield self since node has no data
    def iter_nodes(self) -> Iterable[Spec]:
        for child in self.children():
            yield from child.iter_nodes()


def cs_path_name(in_out: dict) -> str:
    # helper to get [path/]name from 'input' or 'output' dict
    name = in_out["name"]
    if "path" in in_out:
        name = in_out["path"] + "/" + name
    return name


class Scene(Spec):
    @staticmethod
    def matches(group: Group) -> bool:
        attrs = Spec.get_attrs(group)
        return "scene" in attrs

    def to_layer_data(
        self, target_coordinate_system: tuple[str, str] | None = None
    ) -> List[LayerData]:
        layers: List[LayerData] = []
        scene = OMEZarrScene.from_ome_zarr(self.group)
        all_cs = scene.get_coordinate_system()

        if all_cs:
            # Get first coordinate system (sorted for determinism)
            first_cs_key = next(iter(sorted(all_cs.keys())))
            target_coordinate_system = first_cs_key

        for key in scene.images.keys():

            _layers = Multiscales(self.group[key]).to_layer_data()
            ms = scene.images[key]

            # traverse graph into target coordinate system
            input_coordinate_system = (
                key,
                scene.images[key].metadata.intrinsic_coordinate_system.name,
            )
            if target_coordinate_system is None:
                raise ValueError("No target_coordinate_system was provided.")
            seq = scene._graph.get_sequence(
                input_coordinate_system, target_coordinate_system, full=True
            )
            affine = seq.simplify().to_affine().matrix

            # Get axes of input and output coordinate systems
            input_cs = scene.get_coordinate_system(*input_coordinate_system)
            output_cs = scene.get_coordinate_system(*target_coordinate_system)

            input_cs_obj = input_cs[input_coordinate_system]
            output_cs_obj = output_cs[target_coordinate_system]

            # Expand data if output has more spatial dims than input
            input_spatial = [
                ax.name for ax in input_cs_obj.axes if ax.type != "channel"
            ]
            output_spatial = [
                ax.name for ax in output_cs_obj.axes if ax.type != "channel"
            ]
            output_cs_ax_types = [ax.type for ax in output_cs_obj.axes]
            input_cs_ax_types = [ax.type for ax in input_cs_obj.axes]
            output_ch_idx = (
                output_cs_ax_types.index("channel")
                if "channel" in output_cs_ax_types
                else None
            )
            input_ch_index = (
                input_cs_ax_types.index("channel")
                if "channel" in input_cs_ax_types
                else None
            )

            n_extra = len(output_spatial) - len(input_spatial)
            if n_extra > 0:
                affine = _expand_affine_for_projection(seq)

                # Get created axis indices from ProjectAxis, adjusted for channel
                transforms = seq.flatten()
                project_tf = next(
                    (
                        tf
                        for tf in transforms
                        if isinstance(tf, tnd.transforms.ProjectAxis)
                    ),
                    None,
                )
                created_output_idxs = project_tf.created if project_tf else []

                # need to consider channel dim which we remove from affine if present
                if output_ch_idx is not None:
                    created_output_idxs = [
                        idx - 1 for idx in created_output_idxs if idx > output_ch_idx
                    ]

                # Insert singleton dimensions in layer data and update props
                for idx, lyr in enumerate(_layers):
                    layer_data = lyr[0]
                    layer_props = lyr[1]
                    updated_props = _ome_zarr_ms_to_layer_props(
                        ms,
                        input_ch_index,
                        inserted_defaults=[
                            {
                                "index": i,
                                "scale": 1.0,
                                "axis_labels": "Unknown",
                                "units": None,
                            }
                            for i in created_output_idxs
                        ],
                    )

                    # update all properties except name (keep original name)
                    layer_props |= {
                        k: v for k, v in updated_props.items() if k != "name"
                    }

                    # insert singleton dimensions in layer data
                    for out_idx in created_output_idxs:
                        layer_data = [
                            da.expand_dims(d, axis=out_idx) for d in layer_data
                        ]

                    # update layer data tuple
                    _layers[idx] = (layer_data, layer_props, lyr[2])

            # Strip channel row/col from affine if present
            affine = _strip_channel_from_affine(affine, input_ch_index, output_ch_idx)

            for lyr in _layers:
                lyr[1]["affine"] = affine
            layers.extend(_layers)

        return layers


class Plate(Spec):
    @staticmethod
    def matches(group: Group) -> bool:
        return "plate" in Spec.get_attrs(group)

    def data(self) -> list[da.core.Array]:
        # we want to return a dask pyramid...
        return get_pyramid_lazy(self.group)

    def metadata(self) -> dict:
        well_group = get_first_well(self.group)
        first_field_path = get_first_field_path(well_group)
        image_group = well_group[first_field_path]
        return Multiscales(image_group).to_layer_data()[0][1]

    def to_layer_data(self) -> List[LayerData]:
        return [(self.data(), self.metadata(), "image")]

    def children(self) -> list[Spec]:
        # Plate has children If it has labels - check one Well...
        # Child is PlateLabels
        well_group = get_first_well(self.group)
        first_field_path = get_first_field_path(well_group)
        image_group = well_group[first_field_path]
        labels_group = image_group.get("labels", None)
        if labels_group is not None:
            labels_attrs = Spec.get_attrs(labels_group)
            if "labels" in labels_attrs:
                ch: list[Spec] = []
                for labels_path in labels_attrs["labels"]:
                    ch.append(PlateLabels(self.group, labels_path=labels_path))
                return ch
        return []


class PlateLabels(Plate):
    def __init__(self, group: Group, labels_path: str):
        super().__init__(group)
        self.labels_path = labels_path

    def data(self) -> list[da.core.Array]:
        # return a dask pyramid...
        return get_pyramid_lazy(self.group, self.labels_path)

    def children(self) -> list[Spec]:
        # Need to override Plate.children()
        return []

    def to_layer_data(self) -> List[LayerData]:
        return [(self.data(), self.metadata(), "labels")]

    def metadata(self) -> dict:
        # override Plate metadata (no channel-axis etc)
        well_group = get_first_well(self.group)
        first_field_path = get_first_field_path(well_group)
        image_group = well_group[first_field_path]
        labelimage_group = image_group["labels"][self.labels_path]
        m = Label(labelimage_group).metadata()
        rv: dict[str, Any] = {"scale": m.get("scale", None)}
        if "axis_labels" in m:
            rv["axis_labels"] = m["axis_labels"]
        if "units" in m:
            rv["units"] = m["units"]
        return rv


class Labels(Spec):
    @staticmethod
    def matches(group: Group) -> bool:
        return "labels" in Spec.get_attrs(group)

    # override to NOT yield self since node has no data
    def iter_nodes(self) -> Iterable[Spec]:
        attrs = Spec.get_attrs(self.group)
        for name in attrs["labels"]:
            g = self.group[name]
            if Label.matches(g):
                yield Label(g)


class Label(Multiscales):
    @staticmethod
    def matches(group: Group) -> bool:
        # label must also be Multiscales
        if not Multiscales.matches(group):
            return False
        return "image-label" in Spec.get_attrs(group)

    def to_layer_data(self) -> List[LayerData]:
        import pandas as pd
        from ome_zarr import OMEZarrLabels

        ms = OMEZarrLabels.from_ome_zarr(self.group)

        has_channel = "c" in ms.images[0].axes
        channel_index = ms.images[0].axes.index("c") if has_channel else None
        n_channels = ms.images[0].data.shape[channel_index] if has_channel else 1

        labels_layers: List[LayerData] = []
        for ch_idx in range(n_channels):
            data = (
                [da.take(img.data, ch_idx, axis=channel_index) for img in ms.images]
                if has_channel
                else [img.data for img in ms.images]
            )

            props = _ome_zarr_ms_to_layer_props(ms, channel_index)
            props["name"] = ms.name

            # Get color settings if present
            if (
                hasattr(ms, "image_label")
                and hasattr(ms.image_label, "colors")
                and ms.image_label.colors is not None
            ):
                colors = []
                values = []
                for idx in range(len(ms.image_label.colors)):
                    val = ms.image_label.colors[idx].label_value
                    rgba = ms.image_label.colors[idx].rgba
                    colors.append([x / 255 for x in rgba])
                    values.append(val)

                if 0 not in values:
                    # add default color for background (0)
                    colors.insert(0, [0, 0, 0, 0])
                if len(colors) > 0:
                    props["colormap"] = colors

            if (
                hasattr(ms, "image_label")
                and hasattr(ms.image_label, "properties")
                and ms.image_label.properties is not None
            ):
                features = pd.DataFrame(
                    [f.model_dump() for f in ms.image_label.properties]
                )

                if "label_value" in features.columns:
                    features.sort_values(by="label_value", inplace=True)
                props["features"] = features
                props["visible"] = False

            labels_layers.append((data, props, "labels"))

        return labels_layers

    def _splits_channels(self) -> bool:
        # A label is loaded as a single layer keeping all axes (no per-channel
        # split), so the channel axis must be retained in the per-axis metadata
        # to match the layer ndim.
        return False


def read_ome_zarr(root_group: Group) -> Callable:
    def f(*args: Any, **kwargs: Any) -> List[LayerData]:
        layers: List[LayerData] = list()

        print("Root group", root_group.attrs.asdict())

        spec: Spec | None = None

        if Labels.matches(root_group):
            # Try starting at parent Image
            parent_path = root_group.store.root.parent
            parent_group = zarr.open_group(parent_path)
            if Multiscales.matches(parent_group):
                spec = Multiscales(parent_group)
            else:
                # not sure how to handle this?
                spec = Labels(root_group)
        elif Label.matches(root_group):
            # Try starting at parent Image - up 2 dirs
            parent_path = root_group.store.root.parent.parent
            parent_group = zarr.open_group(parent_path)
            if Multiscales.matches(parent_group):
                spec = Multiscales(parent_group)
            else:
                # not sure how to handle this?
                spec = Label(root_group)
        elif Bioformats2raw.matches(root_group):
            spec = Bioformats2raw(root_group)
        elif Multiscales.matches(root_group):
            spec = Multiscales(root_group)
        elif Plate.matches(root_group):
            spec = Plate(root_group)
        elif Scene.matches(root_group):
            spec = Scene(root_group)
        else:
            print("No matching spec", root_group)

        if spec:
            layers.extend(spec.to_layer_data())

        return layers

    return f
