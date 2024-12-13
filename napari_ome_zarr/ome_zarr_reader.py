# zarr v3

from typing import Any, Callable, Dict, List, Tuple, Union

import dask.array as da
import numpy as np
import zarr
from vispy.color import Colormap
from zarr import Group

LayerData = Union[Tuple[Any], Tuple[Any, Dict], Tuple[Any, Dict, str]]


class Spec:
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
            yield from child.iter_nodes()

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
            for ch in attrs["omero"]["channels"]:
                color = ch.get("color", None)
                if color is not None:
                    rgb = [(int(color[i : i + 2], 16) / 255) for i in range(0, 6, 2)]
                    # colormap is range: black -> rgb color
                    colormaps.append(Colormap([[0, 0, 0], rgb]))
            rsp["colormap"] = colormaps
        return rsp


class Bioformats2raw(Spec):
    @staticmethod
    def matches(group: Group) -> bool:
        attrs = Spec.get_attrs(group)
        # Don't consider "plate" as a Bioformats2raw layout
        return "bioformats2raw.layout" in attrs and "plate" not in attrs

    def children(self):
        # TDOO: lookup children from series of OME/METADATA.xml
        childnames = ["0"]
        rv = []
        for name in childnames:
            g = self.group[name]
            if Multiscales.matches(g):
                rv.append(Multiscales(g))
        return rv


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
