# zarr v3

from abc import ABC
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple
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

DEFAULT_SCALE = 1.0
DEFAULT_UNIT = "pixel"
DEFAULT_AXIS_LABEL = "Unknown"


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


def _ome_zarr_multiscales_to_layer_props(
    multiscales: OMEZarrMultiscale | OMEZarrLabels,
    channel_index: int | None,
) -> Dict[str, Any]:
    """
    Helper function to extract properties from an OME-Zarr
    multiscale that can be forwarded to a napari layer,
    irrespective of whether this layer is a image or labels layer.

    The channel dimension is omitted because it is split into
    different layers.
    """

    # get scale (same for all channels)
    s = list(multiscales.images[0].scale.values())
    scale = (
        s[:channel_index] + s[channel_index + 1 :] if channel_index is not None else s
    )
    props: Dict[str, Any] = {}
    if multiscales.images[0].axes_units:
        units = [
            multiscales.images[0].axes_units.get(ax, "pixel")
            for ax in multiscales.images[0].axes
            if ax != "c"
        ]
        props["units"] = tuple(units)

    props["axis_labels"] = tuple([ax for ax in multiscales.images[0].axes if ax != "c"])
    props["scale"] = scale
    props["name"] = multiscales.name

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
    multiplication works after the axis projection. If there are none (the
    sequence starts with ProjectAxis), pad a synthesized Identity instead -
    padding is needed whenever axes are created, regardless of position.
    """
    transform_sequence_flat = seq.flatten()

    # Find ProjectAxis transform
    project_axis_idx, project_tf = next(
        (
            (i, tf)
            for i, tf in enumerate(transform_sequence_flat)
            if isinstance(tf, tnd.transforms.ProjectAxis)
        ),
        (None, None),
    )
    if project_tf is None:
        return seq.simplify().to_affine().matrix

    created_output_idxs = project_tf.created
    pre_transforms = list(transform_sequence_flat.transforms[:project_axis_idx]) or [
        tnd.transforms.Identity(ndim=project_tf.ndims.source)
    ]
    updated_transforms = []

    # Expand transforms before ProjectAxis
    for tf in pre_transforms:
        single_affine = tf.to_affine().matrix
        for output_idx in created_output_idxs:
            # Insert a passthrough dim (new output == new input, identity) as
            # both a column and a row - a column alone leaves cols > rows,
            # since this transform doesn't yet know about the new axis
            # ProjectAxis is about to create.
            single_affine = np.insert(single_affine, output_idx, 0.0, axis=1)
            row = np.zeros(single_affine.shape[1])
            row[output_idx] = 1.0
            single_affine = np.insert(single_affine, output_idx, row, axis=0)
        updated_transforms.append(tnd.transforms.Affine(single_affine))

    # Keep transforms after ProjectAxis as-is
    if not isinstance(project_axis_idx, int):
        raise ValueError("ProjectAxis transform not found in the sequence.")
    updated_transforms.extend(
        transform_sequence_flat.transforms[project_axis_idx + 1 :]
    )

    return tnd.TransformSequence(updated_transforms).simplify().to_affine().matrix


def _extract_channel_props(
    multiscales: OMEZarrMultiscale | OMEZarrLabels,
) -> List[Dict[str, Any]] | None:
    """
    Helper function to extract per-channel properties (colormap, name,
    visible, contrast_limits) from an OME-Zarr multiscale or label image's
    omero metadata. Returns one dict per channel, or None if there's no
    omero metadata.
    """

    if not (hasattr(multiscales, "omero") and multiscales.omero is not None):
        return None

    omero = multiscales.omero.model_dump()
    model = omero.get("rdefs", {}).get("model", "unset")
    greyscale = model == "greyscale"

    channels: List[Dict[str, Any]] = []
    for index, ch in enumerate(omero["channels"]):
        props: Dict[str, Any] = {}

        color = ch.get("color", None)
        if color is not None:
            rgb = [(int(color[i : i + 2], 16) / 255) for i in range(0, 6, 2)]
            if greyscale:
                rgb = [1, 1, 1]
            # colormap is range: black -> rgb color
            cm = Colormap([[0, 0, 0], rgb])
            # Try to match colormap to an existing napari colormap
            props["colormap"] = _match_colors_to_available_colormap(cm)

        ch_name = ch.get("label", f"channel_{index}")
        props["name"] = multiscales.name and f"{multiscales.name}: {ch_name}" or ch_name
        props["visible"] = ch.get("active", True)

        window = ch.get("window", None)
        if window is not None:
            start = window.get("start", None)
            end = window.get("end", None)
            if start is not None and end is not None:
                props["contrast_limits"] = [start, end]

        props["visible"] = ch.get("active", True)

        channels.append(props)

    return channels


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

    def children(self) -> Sequence["Spec"]:
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

        axes_types = tuple(ms.images[0].axes_types.values())
        if "channel" in axes_types:
            channel_index = axes_types.index("channel")
            n_channels = int(ms.images[0].data.shape[channel_index])
        else:
            channel_index = None
            n_channels = 1

        channel_properties = _extract_channel_props(ms)

        layers: List[LayerData] = []
        for ch_idx in range(n_channels):
            data = (
                [da.take(img.data, ch_idx, axis=channel_index) for img in ms.images]
                if channel_index is not None
                else [img.data for img in ms.images]
            )

            props = _ome_zarr_multiscales_to_layer_props(ms, channel_index)
            props["name"] = ms.name
            props["blending"] = "additive"
            if channel_properties is not None:
                props.update(channel_properties[ch_idx])

            layers.extend([(data, props, "image")])

        if hasattr(ms, "labels") and ms.labels is not None:
            for label_key in ms.labels.keys():
                label_spec = Label(self.group[f"labels/{label_key}"])
                layers.extend(label_spec.to_layer_data())

        return layers


class Bioformats2raw(Spec):
    @staticmethod
    def matches(group: Group) -> bool:
        attrs = Spec.get_attrs(group)
        # Don't consider "plate" as a Bioformats2raw layout
        return "bioformats2raw.layout" in attrs and "plate" not in attrs

    def to_layer_data(self) -> List[LayerData]:
        layers: List[LayerData] = []
        for child in self.children():
            layers.extend(child.to_layer_data())
        return layers

    def children(self) -> list[Multiscales]:
        # lookup children from series of OME/METADATA.xml
        xml_data = SyncMixin()._sync(
            self.group.store.get(
                "OME/METADATA.ome.xml", prototype=default_buffer_prototype()
            )
        )
        root = ET.fromstring(xml_data.to_bytes())
        rv: list[Multiscales] = []
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

        if all_cs and target_coordinate_system is None:
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

            # Get axes of input and output coordinate systems
            input_cs = scene.get_coordinate_system(*input_coordinate_system)
            output_cs = scene.get_coordinate_system(*target_coordinate_system)

            input_cs_obj = input_cs[input_coordinate_system]
            output_cs_obj = output_cs[target_coordinate_system]

            if input_coordinate_system != target_coordinate_system:
                seq = scene._graph.get_sequence(
                    input_coordinate_system, target_coordinate_system, full=True
                )
            else:
                # Identity affine if no transformation is needed
                seq = tnd.TransformSequence(
                    transforms=[tnd.transforms.Identity(ndim=len(input_cs_obj.axes))]
                )

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

            # Check whether  the transformation can be represented
            # as an affine matrix
            affine_obj = seq.simplify().to_affine()

            if affine_obj is None:
                raise ValueError(
                    "Affine transformation could not be computed."
                    f"for transform sequence {seq}"
                )

            # ProjectAxis is the only transform that changes dimensionality;
            # If an affine matrix is non-square, a projectAxis transform
            # must exist in the transform sequence. The projectAxis transform is handled
            # here by broadcasting the array to match the output dimensionality.
            # Hence, we need to identify the ProjectAxis transform and expand the
            # affine matrices in the sequence accordingly.
            project_tf = None
            if affine_obj.matrix.shape[0] == affine_obj.matrix.shape[1]:
                affine = affine_obj.matrix
            else:
                project_tf = next(
                    (
                        tf
                        for tf in seq.flatten()
                        if isinstance(tf, tnd.transforms.ProjectAxis)
                    ),
                    None,
                )
                affine = _expand_affine_for_projection(seq)

            n_extra = len(output_spatial) - len(input_spatial)
            if n_extra > 0:
                # indices are in the full output CS space (may include a
                # channel axis, created or pre-existing); layer data/props
                # never have channel, so drop created channel entries and
                # shift every space index past the channel position down by 1.
                created_output_idxs = [
                    i - 1 if output_ch_idx is not None and output_ch_idx < i else i
                    for i in (project_tf.created if project_tf else [])
                    if output_cs_obj.axes[i].type == "space"
                ]

                # Insert singleton dimensions in layer data and update props.
                # TODO: Currently, we assume that labels share the image's spatial
                # scale/units. This MAY not hold in the future.
                for idx, lyr in enumerate(_layers):
                    layer_data = lyr[0]
                    layer_props = lyr[1]
                    updated_properties = _ome_zarr_multiscales_to_layer_props(
                        ms, input_ch_index
                    )
                    # created dims have no real scale/axis label/unit - fill in
                    # placeholders at their position in the output CS
                    for i in created_output_idxs:
                        if "scale" in updated_properties:
                            updated_properties["scale"].insert(i, DEFAULT_SCALE)
                        if "axis_labels" in updated_properties:
                            axis_labels = list(updated_properties["axis_labels"])
                            axis_labels.insert(i, DEFAULT_AXIS_LABEL)
                            updated_properties["axis_labels"] = tuple(axis_labels)
                        if "units" in updated_properties:
                            units = list(updated_properties["units"])
                            units.insert(i, DEFAULT_UNIT)
                            updated_properties["units"] = tuple(units)

                    # update all properties except name (keep original name)
                    layer_props |= {
                        k: v for k, v in updated_properties.items() if k != "name"
                    }

                    for out_idx in created_output_idxs:
                        layer_data = [
                            da.expand_dims(d, axis=out_idx) for d in layer_data
                        ]

                    # update layer data tuple
                    _layers[idx] = (layer_data, layer_props, lyr[2])

            # _expand_affine_for_projection always pads a pseudo input column for
            # every created output axis (synthesizing an Identity pre-transform if
            # none existed). If channel was one of those created axes, that pseudo
            # input column lands at output_ch_idx, not input_ch_index (None, since
            # the true input has no channel).
            if project_tf is not None and output_ch_idx in project_tf.created:
                input_ch_index = output_ch_idx

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

    def to_layer_data(self) -> List[LayerData]:
        data = self.data()

        # get metadata information of a single well
        well_group = get_first_well(self.group)
        first_field_path = get_first_field_path(well_group)
        img = OMEZarrMultiscale.from_ome_zarr(well_group[first_field_path])

        axes_types = tuple(img.images[0].axes_types.values())
        has_channel = "channel" in axes_types
        channel_index = axes_types.index("channel") if has_channel else None
        n_channels = (
            int(img.images[0].data.shape[channel_index]) if has_channel else 1
        )

        base_props = _ome_zarr_multiscales_to_layer_props(
            img, channel_index=channel_index
        )
        channel_properties = _extract_channel_props(img)

        layers: List[LayerData] = []
        for ch_idx in range(n_channels):
            ch_data = (
                [da.take(d, ch_idx, axis=channel_index) for d in data]
                if channel_index is not None
                else data
            )

            props = dict(base_props)
            props["blending"] = "additive"
            if channel_properties is not None:
                props.update(channel_properties[ch_idx])

            layers.append((ch_data, props, "image"))

        for child in self.children():
            layers.extend(child.to_layer_data())
        return layers

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

    def to_layer_data(self) -> List[LayerData]:
        layers: List[LayerData] = []
        for node in self.iter_nodes():
            layers.extend(node.to_layer_data())
        return layers

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
        axes_types = tuple(ms.images[0].axes_types.values())

        if "channel" in axes_types:
            channel_index = axes_types.index("channel")
            n_channels = ms.images[0].data.shape[channel_index]
        else:
            channel_index = None
            n_channels = 1

        labels_layers: List[LayerData] = []
        for ch_idx in range(n_channels):
            data = (
                [da.take(img.data, ch_idx, axis=channel_index) for img in ms.images]
                if channel_index is not None
                else [img.data for img in ms.images]
            )

            props = _ome_zarr_multiscales_to_layer_props(ms, channel_index)
            props["name"] = ms.name
            props["blending"] = "additive"
            props["visible"] = False

            # Get color settings if present
            if (
                hasattr(ms, "image_label")
                and hasattr(ms.image_label, "colors")
                and ms.image_label.colors is not None
            ):
                colors = {
                    c.label_value: [x / 255 for x in c.rgba]
                    for c in ms.image_label.colors
                }
                # default color for background (0)
                colors.setdefault(0, [0, 0, 0, 0])
                if colors:
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


def read_ome_zarr(root_group: Group) -> Callable:
    def f(*args: Any, **kwargs: Any) -> List[LayerData]:
        layers: List[LayerData] = list()

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
