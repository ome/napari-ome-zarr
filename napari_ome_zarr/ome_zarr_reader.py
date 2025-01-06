

# zarr v3

import zarr
from zarr import Group
from zarr.core.sync import SyncMixin
from zarr.core.buffer import default_buffer_prototype

import dask.array as da
from typing import List
from vispy.color import Colormap
from xml.etree import ElementTree as ET

from typing import Any, Dict, List, Tuple, Union

LayerData = Union[Tuple[Any], Tuple[Any, Dict], Tuple[Any, Dict, str]]


class Spec():

    def __init__(self, group: Group):
        self.group = group

    @staticmethod
    def matches(group: Group) -> bool:
        return False
    
    def data(self) -> List[da.core.Array] | None:
        return None

    def metadata(self) -> Dict[str, Any] | None:
        # napari layer metadata
        return {}

    def children(self):
        return []

    def iter_nodes(self):
        yield self
        for child in self.children():
            for ch in child.iter_nodes():
                yield ch

    def iter_data(self):
        for node in self.iter_nodes():
            data = node.data()
            if data:
                yield data

    @staticmethod
    def get_attrs(group: Group):
        if "ome" in group.attrs:
            return group.attrs["ome"]
        return group.attrs


class Multiscales(Spec):

    @staticmethod
    def matches(group: Group) -> bool:
        return "multiscales" in Spec.get_attrs(group)

    def children(self):
        ch = []
        # test for child "labels"
        try:
            grp = self.group["labels"]
            attrs = Spec.get_attrs(grp)
            if "labels" in attrs:
                for name in attrs["labels"]:
                    g = grp[name]
                    if Label.matches(g):
                        ch.append(Label(g))
        except KeyError:
            pass
        return ch

    def data(self):
        attrs = Spec.get_attrs(self.group)
        paths = [ds["path"] for ds in attrs["multiscales"][0]["datasets"]]
        return [da.from_zarr(self.group[path]) for path in paths]

    def metadata(self):
        rsp = {}
        attrs = Spec.get_attrs(self.group)
        axes = attrs["multiscales"][0]["axes"]
        atypes = [axis["type"] for axis in axes]
        if "channel" in atypes:
            channel_axis = atypes.index("channel")
            rsp["channel_axis"] = channel_axis
        if "omero" in attrs:
            colormaps = []
            ch_names = []
            visibles = []
            contrast_limits = []

            for index, ch in enumerate(attrs["omero"]["channels"]):
                color = ch.get("color", None)
                if color is not None:
                    rgb = [(int(color[i : i + 2], 16) / 255) for i in range(0, 6, 2)]
                    # colormap is range: black -> rgb color
                    colormaps.append(Colormap([[0, 0, 0], rgb]))
                ch_names.append(ch.get("label", str(index)))
                visibles.append(ch.get("active", True))

                window = ch.get("window", None)
                if window is not None:
                    start = window.get("start", None)
                    end = window.get("end", None)
                    if start is None or end is None:
                        # Disable contrast limits settings if one is missing
                        contrast_limits = None
                    elif contrast_limits is not None:
                        contrast_limits.append([start, end])

            if rsp.get("channel_axis") is not None:
                rsp["colormap"] = colormaps
                rsp["name"] = ch_names
                rsp["contrast_limits"] = contrast_limits
                rsp["visible"] = visibles
            else:
                rsp["colormap"] = colormaps[0]
                rsp["name"] = ch_names[0]
                rsp["contrast_limits"] = contrast_limits[0]
                rsp["visible"] = visibles[0]

        return rsp

class Bioformats2raw(Spec):

    @staticmethod
    def matches(group: Group) -> bool:
        attrs = Spec.get_attrs(group)
        # Don't consider "plate" as a Bioformats2raw layout
        return "bioformats2raw.layout" in attrs and "plate" not in attrs

    def children(self):
        # lookup children from series of OME/METADATA.xml
        xml_data = SyncMixin()._sync(self.group.store.get("OME/METADATA.ome.xml", prototype=default_buffer_prototype()))
        # print("xml_data", xml_data.to_bytes())
        root = ET.fromstring(xml_data.to_bytes())
        rv = []
        for child in root:
            # {http://www.openmicroscopy.org/Schemas/OME/2016-06}Image
            print(child.tag)
            node_id = child.attrib.get("ID", "")
            if child.tag.endswith("Image") and node_id.startswith("Image:"):
                print("Image ID", node_id)
                image_path = node_id.replace("Image:", "")
                g = self.group[image_path]
                if Multiscales.matches(g):
                    rv.append(Multiscales(g))
        return rv

    # override to NOT yield self since node has no data
    def iter_nodes(self):
        for child in self.children():
            for ch in child.iter_nodes():
                yield ch
    

class Plate(Spec):

    @staticmethod
    def matches(group: Group) -> bool:
        return "plate" in Spec.get_attrs(group)


class Label(Multiscales):

    @staticmethod
    def matches(group: Group) -> bool:
        # label must also be Multiscales
        if not Multiscales.matches(group):
            return False
        return "image-label" in Spec.get_attrs(group)

    def metadata(self) -> Dict[str, Any] | None:
        # override Multiscales metadata
        return {}


def read_ome_zarr(url):

    def f(*args: Any, **kwargs: Any) -> List[LayerData]:

        results: List[LayerData] = list()

        # TODO: handle missing file 
        root_group = zarr.open(url)

        print("Root group", root_group.attrs.asdict())

        if Bioformats2raw.matches(root_group):
            spec = Bioformats2raw(root_group)
        elif Multiscales.matches(root_group):
            spec = Multiscales(root_group)
        elif Plate.matches(root_group):
            spec = Plate(root_group)

        if spec:
            print("spec", spec)
            nodes = list(spec.iter_nodes())
            print("Nodes", nodes)
            for node in nodes:
                node_data = node.data()
                metadata = node.metadata()
                # print(Spec.get_attrs(node.group))
                if Label.matches(node.group):
                    rv: LayerData = (node_data, metadata, "labels")
                else:
                    rv: LayerData = (node_data, metadata)
                results.append(rv)

        return results
    
    return f
