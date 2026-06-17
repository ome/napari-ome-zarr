# zarr v3

from abc import ABC
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Tuple
from xml.etree import ElementTree as ET

import dask.array as da
import numpy as np
import zarr
from napari.utils.colormaps import AVAILABLE_COLORMAPS, Colormap
from napari.utils.transforms import Affine
from zarr import Group
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import SyncMixin

from ome_zarr import OMEZarrMultiscale, OMEZarrScene

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


def remove_axis_from_transform(transform: Dict[str, Any], axis: int) -> Dict[str, Any]:
    """Remove a specific axis from an OME-Zarr transform dict."""
    new_transform = transform.copy()
    if transform["type"] == "scale":
        new_scale = transform["scale"][:]
        del new_scale[axis]
        new_transform["scale"] = new_scale
    if transform["type"] == "translation":
        new_translation = transform["translation"][:]
        del new_translation[axis]
        new_transform["translation"] = new_translation
    if transform["type"] == "rotation":
        matrix = np.array(transform["rotation"])
        matrix = np.delete(matrix, axis, 0)  # remove row
        matrix = np.delete(matrix, axis, 1)  # remove column
        new_transform["rotation"] = matrix.tolist()
    if transform["type"] == "affine":
        matrix = np.array(transform["affine"])
        matrix = np.delete(matrix, axis, 0)  # remove row
        matrix = np.delete(matrix, axis, 1)  # remove column
        new_transform["affine"] = matrix.tolist()
    if transform["type"] == "sequence":
        new_transforms = []
        for sub_transform in transform["transformations"]:
            new_sub_transform = remove_axis_from_transform(sub_transform, axis)
            new_transforms.append(new_sub_transform)
        new_transform["transformations"] = new_transforms
    return new_transform


def transform_to_affine(transform: Dict[str, Any]) -> Affine:
    """Convert a single OME-Zarr transform dict to an Affine object."""
    aff: Affine = None
    if transform["type"] == "scale":
        aff = Affine(scale=transform["scale"])
    if transform["type"] == "translation":
        aff = Affine(translate=transform["translation"])
    if transform["type"] == "rotation":
        matrix = np.array(transform["rotation"])
        # Spec says that rotation matrix is 1 row and column smaller than affine
        matrix = np.pad(
            matrix,
            pad_width=((0, 1), (0, 1)),
            mode="constant",
            constant_values=0,
        )
        matrix[-1, -1] = 1
        aff = Affine(affine_matrix=matrix)
    if transform["type"] == "affine":
        matrix = np.array(transform["affine"])
        # Spec says that rotation matrix is 1 row smaller than affine
        matrix = np.pad(
            matrix,
            pad_width=((0, 1), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        matrix[-1, -1] = 1
        aff = Affine(affine_matrix=matrix)
    return aff


def transforms_to_affine(
    transforms: List[Dict[str, Any]], channel_axis: int | None
) -> Affine:
    # first unwrap and flatten any 'sequence' transforms...
    # NB: if any 'sequence' contains another 'sequence' this is ignored.
    flat_transforms: List[Dict[str, Any]] = []
    for transf in transforms:
        if transf["type"] == "sequence":
            flat_transforms.extend(transf["transformations"])
        else:
            flat_transforms.append(transf)

    # Don't create Affine until we know dimensions...
    aff: Affine = None
    for transf in flat_transforms:
        # print("transforms_to_affine..........ch,transf", channel_axis, transf)
        trans_aff = transform_to_affine(transf)
        if aff is None:
            aff = trans_aff
        elif trans_aff is not None:
            aff = trans_aff.compose(aff)
    # finally, remove channel axis from 2D matrix
    if channel_axis is not None:
        matrix = aff.affine_matrix
        for dim in (0, 1):
            matrix = np.delete(matrix, channel_axis, dim)
        aff = Affine(affine_matrix=matrix)
    return aff


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

    def children(self) -> list[Spec]:
        ch: list[Spec] = []
        # test for child "labels"
        try:
            grp = self.group["labels"]
            attrs = Spec.get_attrs(grp)
            if "labels" in attrs:
                for name in attrs["labels"]:
                    g = grp[name]
                    if Label.matches(g):
                        label_image = Label(g)
                        # Label inherits parent transforms
                        ch_axis = self.metadata().get("channel_axis", None)
                        for transf in self.parent_transforms:
                            label_image.add_parent_transform(transf, ch_axis)
                        ch.append(label_image)
        except KeyError:
            pass
        return ch

    def data(self) -> list[da.core.Array]:
        ms_img = OMEZarrMultiscale.from_ome_zarr(self.group)
        return [img.data for img in ms_img.images]

    def metadata(self) -> Dict[str, Any]:
        rsp: dict = {}

        ms_img = OMEZarrMultiscale.from_ome_zarr(self.group)
        axes = ms_img.images[0].axes
        scale = list(ms_img.images[0].scale.values())
        if "c" in axes:
            rsp["channel_axis"] = axes.index("c")
            scale.pop(rsp["channel_axis"])
        
        if ms_img.images[0].axes_units is not None:
            rsp["units"] = list(ms_img.images[0].axes_units.values())
        rsp["scale"] = scale
        rsp["name"] = ms_img.name

        if ms_img.omero is not None:
            attrs = ms_img.omero.model_dump()
            colormaps = []
            ch_names = []
            visibles = []
            contrast_limits: list[list[int]] = []
            model = attrs.get("rdefs", {}).get("model", "unset")
            greyscale = model == "greyscale"

            for index, ch in enumerate(attrs.get("channels", [])):
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
                ch_names.append(ch.get("label", f"channel_{index}"))
                visibles.append(ch.get("active", True))

                window = ch.get("window", None)
                if window is not None:
                    start = window.get("start", None)
                    end = window.get("end", None)
                    if start is not None and end is not None:
                        # skip if None. Otherwise check no previous skip
                        if len(contrast_limits) == index:
                            contrast_limits.append([start, end])

            if rsp.get("channel_axis") is not None:
                rsp["colormap"] = colormaps
                rsp["name"] = [f"{ms_img.name}:{ch}" for ch in ch_names]
                if len(contrast_limits) > 0:
                    rsp["contrast_limits"] = contrast_limits
                rsp["visible"] = visibles
            else:
                rsp["colormap"] = colormaps[0]
                rsp["name"] = f"{ms_img.name}:{ch_names[0]}"
                if len(contrast_limits) > 0:
                    rsp["contrast_limits"] = contrast_limits[0]
                rsp["visible"] = visibles[0]

        if self.parent_transforms is not None:
            rsp["affine"] = self.parent_transforms

        return rsp


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


def path_name(in_out: dict | str) -> str:
    # helper to get path from 'input' or 'output' dict
    if isinstance(in_out, str):
        return in_out
    paths = []
    if "path" in in_out:
        paths.append(in_out["path"])
    if "name" in in_out:
        paths.append(in_out["name"])
    return "/".join(paths)


class Scene(Spec):
    @staticmethod
    def matches(group: Group) -> bool:
        attrs = Spec.get_attrs(group)
        print("Scene.matches", attrs)
        return "scene" in attrs

    def children(self) -> list[Spec]:
        scene = OMEZarrScene.from_ome_zarr(self.group)
        # lookup children from coordinateTransformations 'input' paths
        rv: list[Spec] = []
        scene_attrs = Spec.get_attrs(self.group).get("scene", {})
        transfs = scene_attrs.get("coordinateTransformations", [])
        output = None
        for transf in transfs:
            # TODO: input may NOT have "path"
            # TODO: input "name" is ignored!
            image_path = transf.get("input").get("path", None)
            out = transf.get("output")
            if output is None:
                output = out
            elif path_name(output) != path_name(out):
                print(
                    f"WARNING: '{path_name(out)}' output different"
                    f"from previous '{path_name(output)}'"
                )
                continue
            g = self.group[image_path]
            print("scene child", image_path, g)
            if Multiscales.matches(g):
                ms_image = Multiscales(g)
                # child image gets the parent transform
                # TODO: use the input 'name' to specify which coordinateSystem to use
                ms_image.parent_transforms.append(transf)
                rv.append(ms_image)

        # check if 'output' is path to image...
        try:
            if output is not None:
                g = self.group[output.get("path", "")]
                if Multiscales.matches(g):
                    # Add to start (layer behind other tiles)
                    rv.insert(0, Multiscales(g))
        except KeyError:
            pass
        return rv

    # override to NOT yield self since node has no data
    def iter_nodes(self) -> Iterable[Spec]:
        from napari import current_viewer
        from magicgui.widgets import create_widget, ComboBox

        ngff_scene = OMEZarrScene.from_ome_zarr(self.group)
        
        # # browse images for possible coordinate systems
        # for tf in ngff_scene.coordinate_transformations:
        #     input_path = tf.input.path if tf.input.path is not None else ""
        #     output_path = tf.output.path if tf.output.path is not None else ""
        #     input = f"{input_path}:{tf.input.name}"
        #     output = f"{output_path}:{tf.output.name}"
        #     possible_coordinate_systems.update([input, output])

        # create napari widget for user to select coordinate system to use as reference for transforms
        target_cs = ":world"
        cs = ngff_scene.get_coordinate_system(target_cs[1:])

        # viewer = current_viewer()

        # widget = ComboBox(
        #     value = list(possible_coordinate_systems)[0],
        #     choices=list(possible_coordinate_systems),
        # )
        # viewer.window.add_dock_widget(widget, area="right")            


        # Load all image subgroups
        for img_name in ngff_scene.images.keys():
            img_group = self.group[img_name]
            # Assume images have their own to_ome_zarr-like interface
            # You may need to adapt based on your OMEZarrMultiscale.from_ome_zarr implementation
            ngff_ms = OMEZarrMultiscale.from_ome_zarr(img_group)
            try:
                composed_affine = ngff_scene._graph.get_sequence(
                    f"{img_name}:{ngff_ms.metadata.intrinsic_coordinate_system.name}",
                    target_cs).simplify().to_affine().matrix
                
                if "c" in [ax.name for ax in cs.axes]:
                    composed_affine = composed_affine[1:, 1:]
            except ValueError:
                print(f"WARNING: No transform found from {img_name} to {target_cs}")
                composed_affine = None
                

            ms = Multiscales(img_group)
            ms.parent_transforms = composed_affine
            yield ms


        # ngff_scene = OMEZarrScene.from_ome_zarr(self.group)

        # # transforms key is each transform output "path.zarr/name"
        # # (where name is name of coordinateSystem)
        # transforms = defaultdict(list)  # type: Dict[str, List[Dict[str, Any]]]
        # # track unique coordinateSystems by "path.zarr/name"
        # # systems = set()

        # # FIRST, go through all transforms in this scene,
        # # AND any child transforms we find at 'input' or 'output' paths...
        # scene_attrs = Spec.get_attrs(self.group).get("scene", {})
        # image_attrs = {}
        # for transf in scene_attrs.get("coordinateTransformations", []):
        #     output = transf["output"]
        #     transf["input_full_path"] = path_name(transf["input"])
        #     transforms[path_name(output)].append(transf)
        #     # traverse to input/output coordinateSystem paths...
        #     for io in ("input", "output"):
        #         image_path = transf[io].get("path", None)
        #         if image_path is not None and image_path not in image_attrs:
        #             image_attrs[image_path] = Spec.get_attrs(self.group[image_path])
        #             # need to add child transforms to our graph
        #             for ms in image_attrs[image_path].get("multiscales", []):
        #                 for child_transf in ms.get("coordinateTransformations", []):
        #                     # TODO: assert output doesn't have 'path'?
        #                     out_name = path_name(child_transf["output"])
        #                     out_path_name = image_path + "/" + out_name
        #                     input_path_name = (
        #                         image_path + "/" + path_name(child_transf["input"])
        #                     )
        #                     child_transf["input_full_path"] = input_path_name
        #                     transforms[out_path_name].append(child_transf)
                        # and the datasets... - find 'output' (just use first one)
                        # for ds in ms.get("datasets", [])[:1]:
                        #     # only expect single transform...
                        #     for ds_transf in ds.get("coordinateTransformations", []):
                        #         out_name = path_name(ds_transf["output"])
                        #         out_path_name = image_path + "/" + out_name
                        #         # don't expect this to be used. e.g. image.zarr/s0
                        #         input_path_name = (
                        #             image_path + "/" + path_name(ds_transf["input"])
                        #         )
                        #         ds_transf["input_full_path"] = input_path_name
                        #         transforms[out_path_name].append(ds_transf)
                        #         # Use this to create the Multiscales object below...
                        #         ds_transf["multiscale_path"] = image_path

        # print("Scene.iter_nodes...transforms")
        # for key, transfs in transforms.items():
        #     print(f"  {key}: ", [t["input_full_path"] for t in transfs])
        #   translated_x_and_y:  ['4995115_full.zarr/physical', 'translated_x50']
        #   4995115_full.zarr/physical:  ['4995115_full.zarr/s0']
        #   translated_x50:  ['rot10.zarr/rotated', 'rot45.zarr/rotated']
        #   rot10.zarr/rotated:  ['rot10.zarr/physical']
        #   rot10.zarr/physical:  ['rot10.zarr/s0']
        #   rot45.zarr/rotated:  ['rot45.zarr/physical']
        #   rot45.zarr/physical:  ['rot45.zarr/s0']

        # find the unique coordinateSystems (outputs)
        # outputs = set(transforms.keys())

        # # remove any output that is also an input (i.e. intermediate node in graph)
        # for transf_list in transforms.values():
        #     outputs -= {t["input_full_path"] for t in transf_list}

        # print("Scene.iter_nodes...final outputs ----->", outputs)

        # assert len(outputs) == 1, "Multiple output coordinateSystems not supported"

        # # now we can iterate through the graph starting at the unique output
        # # coordinateSystem, and yield any nodes we find along the way...
        # def iter_graph(
        #     path_name: str, parent_trans: list[Dict[str, Any]]
        # ) -> Iterable[list[Dict[str, Any]]]:
        #     # find transforms that output to this node
        #     node_transfs = transforms.get(path_name)
        #     # print("iter_graph", path_name, parent_trans)

        #     if node_transfs is None:
        #         yield parent_trans
        #     # find the group for this node
        #     else:
        #         for transf in node_transfs:
        #             parents_copy = parent_trans[:]
        #             parents_copy.append(transf)
        #             yield from iter_graph(transf["input_full_path"], parents_copy)

        # output = outputs.pop()
        # inputs = list(iter_graph(output, []))

        # for img in ngff_scene.images:
        #     yield img


        # print("IMAGES")

        # # Ignore any transform lists that don't lead to a multiscale image...
        # multiscale_inputs = [inp for inp in inputs if "multiscale_path" in inp[-1]]

        # for trans_list in multiscale_inputs:
        #     print("---------------------")
        #     print(trans_list[-1]["multiscale_path"])
        #     # the last transform should have "multiscale_path" key...
        #     ms_image = Multiscales(self.group[trans_list[-1]["multiscale_path"]])

        #     # reverse, so we START at datasets -> coordinateSystem -> scene
        #     trans_list.reverse()
        #     ms_image.parent_transforms = trans_list
        #     yield ms_image

        # for child in self.children():
        #     yield from child.iter_nodes()


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
        return {"scale": m.get("scale", None)}


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

    def add_parent_transform(
        self, transform: Dict[str, Any], parent_channel_axis: int | None
    ) -> None:
        # Add the parent transform to the current transform. If
        # parent_channel_axis is not in Label, we need to remove that axis
        # from the transform.
        label_channel_axis = self.metadata().get("channel_axis", None)
        if (
            parent_channel_axis is not None
            and parent_channel_axis != label_channel_axis
        ):
            print("remove axis from transform", parent_channel_axis)
            transform = remove_axis_from_transform(transform, parent_channel_axis)
        self.parent_transforms.append(transform)

    def metadata(self) -> Dict[str, Any]:
        # override Multiscales metadata
        # call super
        ms_data = super().metadata()
        if ms_data is None:
            ms_data = {}
        if "channel_axis" in ms_data:
            ms_data.pop("channel_axis")

        attrs = Spec.get_attrs(self.group)
        image_label = attrs.get("image-label", {})
        colors: dict[int | bool, list[float]] = {}
        color_list = image_label.get("colors", [])
        if color_list:
            for color in color_list:
                try:
                    label_value = color["label-value"]
                    rgba = color.get("rgba", None)
                    if rgba:
                        rgba = [x / 255 for x in rgba]

                    if isinstance(label_value, (bool, int)):
                        colors[label_value] = rgba
                    else:
                        raise Exception("not bool or int")

                except Exception:
                    pass
                    # LOGGER.exception("invalid color - %s", color)

        props_list = image_label.get("properties", [])
        if props_list:
            props_by_labelid: dict[int, dict[str, str]] = {}
            for props in props_list:
                label_val = props["label-value"]
                props_by_labelid[label_val] = dict(props)
                del props_by_labelid[label_val]["label-value"]

            properties: Dict[str, List] = {}
            # First, create lists for all existing keys...
            for label_id, props_dict in props_by_labelid.items():
                for key in props_dict.keys():
                    properties[key] = []

            keys = list(properties.keys())

            properties["index"] = []
            for label_id, props_dict in props_by_labelid.items():
                properties["index"].append(label_id)
                # ...in case some objects don't have all the keys
                for key in keys:
                    properties[key].append(props_dict.get(key, None))
            ms_data["properties"] = properties

        rsp = {
            "name": f"labels{self.group.name}",
            "visible": False,  # labels not visible initially
            **ms_data,
        }
        # in case no colors, don't set colormap (no labels will be shown)
        if len(colors) > 0:
            rsp["colormap"] = colors

        return rsp


def read_ome_zarr(root_group: Group) -> Callable:
    def f(*args: Any, **kwargs: Any) -> List[LayerData]:
        results: List[LayerData] = list()

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
            print("Scene")
            spec = Scene(root_group)
        else:
            print("No matching spec", root_group)

        if spec:
            nodes = list(spec.iter_nodes())
            for node in nodes:
                #print("node", node, node.group)
                node_data = node.data()
                metadata = node.metadata()
                print("Node:", node.group.name, "metadata:", metadata)
                layer_type = "image"
                if Label.matches(node.group) or isinstance(node, PlateLabels):
                    layer_type = "labels"
                rv: LayerData = (node_data, metadata, layer_type)
                results.append(rv)

        return results

    return f
