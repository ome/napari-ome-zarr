"""This module is a napari plugin.

It implements the ``napari_get_reader`` hook specification, (to create a reader plugin).
"""

import warnings
from typing import Callable

import zarr

from .ome_zarr_reader import Scene, read_ome_zarr


def _open_group(path: str | list) -> zarr.Group | None:
    if isinstance(path, list):
        if len(path) > 1:
            warnings.warn("more than one path is not currently supported")
        path = path[0]

    try:
        return zarr.open_group(path, mode="r")
    except Exception as e:
        warnings.warn(f"Failed to open Zarr group: {e}")
        return None


def napari_get_reader(path: str | list) -> Callable | None:
    """Returns a reader for supported paths that include IDR ID.

    - URL of the form: https://livingobjects.ebi.ac.uk/idr/zarr/v0.1/ID.zarr/
    """
    group = _open_group(path)
    if group is not None:
        return read_ome_zarr(group)
    return None


def napari_get_fused_reader(path: str | list) -> Callable | None:
    """Returns a reader for OME-Zarr scenes, with the scene images fused
    into a single layer per channel and label image (see FusedScene).
    """
    group = _open_group(path)
    if group is not None and Scene.matches(group):
        return read_ome_zarr(group, fused=True)
    return None
