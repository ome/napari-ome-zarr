"""This module is a napari plugin.

It implements the ``napari_get_reader`` hook specification, (to create a reader plugin).
"""

import warnings
from typing import Callable

import numpy as np
import zarr
from napari.utils.colormaps import AVAILABLE_COLORMAPS, Colormap

from .ome_zarr_reader import read_ome_zarr


def napari_get_reader(path: str | list) -> Callable | None:
    """Returns a reader for supported paths that include IDR ID.

    - URL of the form: https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/ID.zarr/
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
