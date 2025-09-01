"""This module is a napari plugin.

It implements the ``napari_get_reader`` hook specification, (to create a reader plugin).
"""

import logging
import warnings
from importlib.metadata import version
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
from napari.utils.colormaps import AVAILABLE_COLORMAPS, Colormap
from ome_zarr.io import parse_url
from ome_zarr.reader import Label, Node, Reader
from ome_zarr.types import LayerData, PathLike, ReaderFunction

LOGGER = logging.getLogger("napari_ome_zarr.reader")

METADATA_KEYS = ("name", "visible", "contrast_limits", "colormap", "metadata")

# major and minor versions as int
napari_version = tuple(map(int, list(version("napari").split(".")[:2])))


def napari_get_reader(path: PathLike) -> Optional[ReaderFunction]:
    """Returns a reader for supported paths that include IDR ID.

    - URL of the form: https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/ID.zarr/
    """
    if isinstance(path, list):
        if len(path) > 1:
            warnings.warn("more than one path is not currently supported")
        path = path[0]
    zarr = parse_url(path)
    if zarr:
        reader = Reader(zarr)
        return transform(reader())
    # Ignoring this path
    return None


def transform_properties(
    props: Optional[Dict[str, Dict]] = None,
) -> Optional[Dict[str, List]]:
    """
    Transform properties

    Transform a dict of {label_id : {key: value, key2: value2}}
    with a key for every LABEL
    into a dict of a key for every VALUE, with a list of values for each
    .. code::

        {
            "index": [1381342, 1381343...]
            "omero:roiId": [1381342, 1381343...],
            "omero:shapeId": [1682567, 1682567...]
        }

    """
    if props is None:
        return None

    properties: Dict[str, List] = {}

    # First, create lists for all existing keys...
    for label_id, props_dict in props.items():
        for key in props_dict.keys():
            properties[key] = []

    keys = list(properties.keys())

    properties["index"] = []
    for label_id, props_dict in props.items():
        properties["index"].append(label_id)
        # ...in case some objects don't have all the keys
        for key in keys:
            properties[key].append(props_dict.get(key, None))
    return properties


def transform_scale(
    node_metadata: Dict, metadata: Dict, channel_axis: Optional[int]
) -> None:
    """
    e.g. transformation is {"scale": [0.2, 0.06, 0.06]}
    Get a list of these for each level in data. Just use first?
    """
    if "coordinateTransformations" in node_metadata:
        level_0_transforms = node_metadata["coordinateTransformations"][0]
        for transf in level_0_transforms:
            if "scale" in transf:
                scale = transf["scale"]
                if channel_axis is not None:
                    scale.pop(channel_axis)
                metadata["scale"] = tuple(scale)
            if "translation" in transf:
                translate = transf["translation"]
                if channel_axis is not None:
                    translate.pop(channel_axis)
                metadata["translate"] = tuple(translate)


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


def transform(nodes: Iterator[Node]) -> Optional[ReaderFunction]:
    def f(*args: Any, **kwargs: Any) -> List[LayerData]:
        results: List[LayerData] = list()

        for node in nodes:
            data: List[Any] = node.data
            metadata: Dict[str, Any] = {}
            if data is None or len(data) < 1:
                LOGGER.debug("skipping non-data %s", node)
            else:
                LOGGER.debug("transforming %s", node)
                LOGGER.debug("node.metadata: %s", node.metadata)

                layer_type: str = "image"
                channel_axis = None
                try:
                    ch_types = [axis["type"] for axis in node.metadata["axes"]]
                    if "channel" in ch_types:
                        channel_axis = ch_types.index("channel")
                except Exception:
                    LOGGER.error("Error reading axes: Please update ome-zarr")
                    raise

                transform_scale(node.metadata, metadata, channel_axis)

                if node.load(Label):
                    layer_type = "labels"
                    for x in METADATA_KEYS:
                        if x in node.metadata:
                            metadata[x] = node.metadata[x]
                        elif x == "colormap" and node.metadata["color"]:
                            # key changed 'color' -> 'colormap' in napari 0.5
                            if napari_version >= (0, 5):
                                metadata["colormap"] = node.metadata["color"]
                            else:
                                metadata["color"] = node.metadata["color"]
                    if channel_axis is not None:
                        data = [
                            np.squeeze(level, axis=channel_axis) for level in node.data
                        ]
                else:
                    cms = node.metadata.get("colormap", [])
                    for idx, cm in enumerate(cms):
                        if not isinstance(cm, Colormap):
                            cms[idx] = Colormap(cm)
                        # Try to match colormap to an existing napari colormap
                        cms[idx] = _match_colors_to_available_colormap(cms[idx])

                    if channel_axis is not None:
                        # multi-channel; Copy known metadata values
                        metadata["channel_axis"] = channel_axis
                        for x in METADATA_KEYS:
                            if x in node.metadata:
                                metadata[x] = node.metadata[x]
                        # overwrite 'name' if we have 'channel_names'
                        if "channel_names" in node.metadata:
                            metadata["name"] = node.metadata["channel_names"]
                    else:
                        # single channel image, so metadata just needs
                        # single items (not lists)
                        for x in METADATA_KEYS:
                            if x in node.metadata:
                                try:
                                    metadata[x] = node.metadata[x][0]
                                except Exception:
                                    pass
                        # overwrite 'name' if we have 'channel_names'
                        if "channel_names" in node.metadata:
                            if len(node.metadata["channel_names"]) > 0:
                                metadata["name"] = node.metadata["channel_names"][0]

                properties = transform_properties(node.metadata.get("properties"))
                if properties is not None:
                    metadata["properties"] = properties

                rv: LayerData = (data, metadata, layer_type)
                LOGGER.debug("Transformed: %s", rv)
                results.append(rv)

        return results

    return f
