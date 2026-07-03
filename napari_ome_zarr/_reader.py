"""This module is a napari plugin.

It implements the ``napari_get_reader`` hook specification, (to create a reader plugin).
"""

import warnings
from typing import Any, Callable

import numpy as np
import zarr

from .ome_zarr_reader import LayerData, read_ome_zarr


def napari_get_reader(path: str | list) -> Callable | None:
    """Returns a reader for supported paths that include IDR ID.

    - URL of the form: https://livingobjects.ebi.ac.uk/idr/zarr/v0.1/ID.zarr/
    """
    if isinstance(path, list):
        if len(path) > 1:
            warnings.warn("more than one path is not currently supported")
        path = path[0]

    group = None
    try:
        group = zarr.open_group(path, mode="r")
    except Exception as e:
        warnings.warn(f"Failed to open Zarr group: {e}")
        return None

    if group is not None:
        return read_ome_zarr(group)
    return None


def napari_get_eager_reader(path: str | list) -> Callable | None:
    read_ome_zarr_lazy = napari_get_reader(path)
    if read_ome_zarr_lazy is not None:

        def read_ome_zarr_eager(*args: Any, **kwargs: Any) -> list[LayerData]:
            lazy_result = read_ome_zarr_lazy(*args, **kwargs)
            eager_result = []
            for data, metadata, layer_type in lazy_result:
                if isinstance(data, list):  # multiscales
                    eager_data = list(map(np.asarray, data))
                else:  # single scale
                    eager_data = np.asarray(data)
                eager_result.append((eager_data, metadata, layer_type))
            return eager_result

        return read_ome_zarr_eager
