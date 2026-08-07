# zarr v3

import warnings
from abc import ABC
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Tuple
from xml.etree import ElementTree as ET

import dask.array as da
import numpy as np
import zarr
from napari.utils.colormaps import AVAILABLE_COLORMAPS, Colormap
from napari.utils.transforms import Affine
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
    multiscales: OMEZarrMultiscale | OMEZarrLabels, channel_index: int | None
) -> Dict[str, Any]:

    # get scale (same for all channels)
    s = list(multiscales.images[0].scale.values())
    scale = (
        s[:channel_index] + s[channel_index + 1 :] if channel_index is not None else s
    )

    props: Dict[str, Any] = {}
    if multiscales.images[0].axes_units:
        props["units"] = list(multiscales.images[0].axes_units.values())
    props["axis_labels"] = [ax for ax in multiscales.images[0].axes if ax != "c"]
    props["scale"] = scale
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
        n_channels = ms.images[0].data.shape[channel_index] if has_channel else 1

        layers: List[LayerData] = []
        for ch_idx in range(n_channels):
            data = (
                [da.take(img.data, ch_idx, axis=channel_index) for img in ms.images]
                if has_channel
                else [img.data for img in ms.images]
            )

            props = _ome_zarr_ms_to_layer_props(ms, channel_index)
            props["name"] = ms.name

            layers.extend([(data, props, "image")])

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
                    rv.extend(Multiscales(g).to_layer_data())
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

    def to_layer_data(self) -> List[LayerData]:
        layers: List[LayerData] = []
        scene = OMEZarrScene.from_ome_zarr(self.group)
        all_cs = scene.get_coordinate_system()

        target_coordinate_system: tuple[str, str] | None = None
        if "" in all_cs:
            target_coordinate_system = ("", all_cs[""][0].name)
        else:
            # Pick image with highest dimensionality to avoid projecting down
            key_with_max_axes = max(all_cs.keys(), key=lambda k: len(all_cs[k][0].axes))
            target_coordinate_system = (key_with_max_axes, all_cs[key_with_max_axes][0].name)

        for key in scene.images.keys():

            _layers = Multiscales(self.group[key]).to_layer_data()
            ms = OMEZarrMultiscale.from_ome_zarr(self.group[key])

            # traverse graph into target coordinate system
            input_cs = (key, scene.images[key].metadata.intrinsic_coordinate_system.name)
            seq = scene._graph.get_sequence(input_cs, target_coordinate_system)
            affine = seq.simplify().to_affine().matrix

            # Get axes of input and output coordinate systems to find channel axes
            input_cs_obj = scene.get_coordinate_system(input_cs[0], input_cs[1])
            input_cs_axes = input_cs_obj[input_cs[0]][0].axes
            output_cs_obj = scene.get_coordinate_system(
                target_coordinate_system[0], target_coordinate_system[1]
            )
            output_cs_axes = output_cs_obj[target_coordinate_system[0]][0].axes

            # Find channel axis indices in input/output coordinate systems
            input_ch_idx = next(
                (i for i, ax in enumerate(input_cs_axes) if ax.type == "channel"), None
            )
            output_ch_idx = next(
                (i for i, ax in enumerate(output_cs_axes) if ax.type == "channel"), None
            )

            # Common case: both have channel at same index, matrix is square
            if (
                input_ch_idx is not None
                and affine.shape[0] == affine.shape[1]
            ):
                affine = np.delete(affine, input_ch_idx, axis=0)
                affine = np.delete(affine, input_ch_idx, axis=1)
                for lyr in _layers:
                    lyr[1]["affine"] = affine
                layers.extend(_layers)
                continue

            # Remove channel row/column independently (handles dimension changes)
            # Row = output axis, Column = input axis
            if output_ch_idx is not None:
                affine = np.delete(affine, output_ch_idx, axis=0)
            if input_ch_idx is not None:
                affine = np.delete(affine, input_ch_idx, axis=1)

            # Check for spatial dimension mismatch after channel removal
            n_out = affine.shape[0] - 1  # excl homogeneous row
            n_in = affine.shape[1] - 1   # excl homogeneous col
            input_axes = ms.images[0].axes
            n_data_spatial = len([a for a in input_axes if a != "c"])

            if n_out > n_in:
                # ponytail: transform adds spatial dim(s). Broadcast data with
                # singleton at front. Revisit ordering if real data needs differ.

                # Build square affine sized for output CS
                new_affine = np.eye(n_out + 1)
                new_affine[:, -n_out:] = affine
                affine = new_affine

                n_extra = n_out - n_in
                for idx, lyr in enumerate(_layers):
                    data_expanded = [da.expand_dims(arr, axis=tuple(range(n_extra))) for arr in lyr[0]]
                    props_expanded = lyr[1].copy()

                    # we need to update the layer properties to match the new data shape.
                    props_expanded["scale"] = props_expanded["scale"][-1] + props_expanded["scale"]
                    props_expanded["axis_labels"] = [ax.name for ax in output_cs_axes if ax.type != "channel"]
                    props_expanded["units"] = [props_expanded["units"][-1]] + props_expanded["units"] 
                    _layers[idx] = (data_expanded, props_expanded, lyr[2])
                
            elif n_out < n_in:
                # ponytail: transform removes spatial dims - unusual, warn and
                # use identity. Add projection support when a real case appears.
                warnings.warn(
                    f"Transform reduces dims ({n_in} -> {n_out}), using identity"
                )
                affine = np.eye(n_data_spatial + 1)
                # also need to prepend the name of the added dim to 

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
        return Multiscales(image_group).metadata()

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

            # get color settings if present
            if hasattr(ms, "image_label") and hasattr(ms.image_label, "colors"):

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

            labels_layers.extend([(data, props, "labels")])

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
