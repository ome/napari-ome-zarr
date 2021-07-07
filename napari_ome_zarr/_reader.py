"""This module is a napari plugin.

It implements the ``napari_get_reader`` hook specification, (to create a reader plugin).
"""


import logging
import warnings
from typing import Any, Callable, Dict, Iterator, List, Optional

from ome_zarr.data import CHANNEL_DIMENSION
from ome_zarr.io import parse_url
from ome_zarr.reader import Label, Node, Reader
from ome_zarr.types import LayerData, PathLike, ReaderFunction

try:
    from napari_plugin_engine import napari_hook_implementation
except ImportError:

    def napari_hook_implementation(
        func: Callable, *args: Any, **kwargs: Any
    ) -> Callable:
        return func


LOGGER = logging.getLogger("napari_ome_zarr.reader")


@napari_hook_implementation
def napari_get_reader(path: PathLike) -> Optional[ReaderFunction]:
    """Returns a reader for supported paths that include IDR ID.

    - URL of the form: https://s3.embassy.ebi.ac.uk/idr/zarr/v0.1/ID.zarr/
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


def transform(nodes: Iterator[Node]) -> Optional[ReaderFunction]:
    def f(*args: Any, **kwargs: Any) -> List[LayerData]:
        results: List[LayerData] = list()

        for node in nodes:
            data: List[Any] = node.data

            # check if this node has a multiscale spec and try to load the axes
            # NOTE this was the best way I found for getting the axes metadata, which is still very cumbersome
            axes = None
            for spec in node.specs:
                axes = spec.lookup('multiscales', [{}])[0].get('axes', None)
                if axes is not None:
                    break

            # NOTE the metadata seems to be always empty, why do we even have it?
            metadata: Dict[str, Any] = node.metadata

            if data is None or len(data) < 1:
                LOGGER.debug(f"skipping non-data {node}")
            else:
                LOGGER.debug(f"transforming {node}")
                shape = data[0].shape

                layer_type: str = "image"
                if node.load(Label):
                    layer_type = "labels"
                    if "colormap" in metadata:
                        del metadata["colormap"]

                # multiscale spec >= 0.3 has the axes list that can be used to determine
                # if there is a channel axis, and which axis it is
                elif axes is not None and "c" in axes:
                    channel_dimension = axes.index("c")
                    if shape[channel_dimension] > 1:
                        metadata["channel_axis"] = channel_dimension

                # multiscale spec < 0.3 is 5d and has a hardcoded chanel axis
                elif axes is None and shape[CHANNEL_DIMENSION] > 1:
                    metadata["channel_axis"] = CHANNEL_DIMENSION

                else:
                    for x in ("name", "visible", "contrast_limits", "colormap"):
                        if x in metadata:
                            try:
                                metadata[x] = metadata[x][0]
                            except Exception:
                                del metadata[x]

                rv: LayerData = (data, metadata, layer_type)
                LOGGER.debug(f"Transformed: {rv}")
                results.append(rv)

        return results

    return f
