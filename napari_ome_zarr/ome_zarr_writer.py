import time
from typing import Any

import numpy as np
from napari.qt import create_worker
from napari.qt.threading import thread_worker
from napari.types import LayerData
from napari.utils import progress
from ome_zarr.classes import OMEZarrImage, OMEZarrLabels, OMEZarrMultiscale


def write_05_ome_zarr(path: str, layer_data, attributes: dict) -> list[str]:
    """
    Function to write a single layer to an OME-Zarr file using the 0.5 specification.
    """
    ms = _napari_layer_to_ome_zarr(path, layer_data, attributes)

    # set up jobs to write data concurrently
    jobs = ms.to_ome_zarr(path, overwrite=True, compute=False, version="0.5")
    _save_data(jobs)

    # return path to any file(s) that were successfully written
    return [path]


def write_06_ome_zarr(path: str, layer_data, attributes: dict) -> list[str]:
    """
    Function to write a single layer to an OME-Zarr file using the 0.6 specification.
    """
    ms = _napari_layer_to_ome_zarr(path, layer_data, attributes)

    # set up jobs to write data concurrently
    jobs = ms.to_ome_zarr(path, overwrite=True, compute=False, version="0.6")
    _save_data(jobs)

    # return path to any file(s) that were successfully written
    return [path]


def _save_data(jobs: list["dask.delayed.Delayed"]):
    """
    Save the given list of dask delayed jobs concurrently using a thread worker.
    Each job corresponds to writing a pyramid layer of the OME-Zarr image.
    """

    @thread_worker(progress={"total": len(jobs)})
    def save_pyramid_layers(jobs):
        for job in jobs:
            job.compute()
            yield

    save_pyramid_layers(jobs).start()


def _stack_layers_by_resolution(
    image_layers: list[LayerData], base_axes: list[str]
) -> tuple:
    """Stack all image layers into a channel dimension at each resolution level."""
    # Normalize each layer to multiscale format (list of scales)
    normalized = []
    for data, attrs in image_layers:
        normalized.append(data if attrs.get("multiscale", False) else [data])

    # Stack by resolution level
    num_levels = max(len(scales) for scales in normalized)
    stacked_scales = [
        np.stack([scales[i] for scales in normalized], axis=0)
        for i in range(num_levels)
    ]

    return stacked_scales, ["c"] + base_axes


def multiple_layers_to_ome_zarr(
    path: str, layers_data: list[LayerData]
) -> OMEZarrMultiscale:
    """
    Writer for multiple layers that assumes that the layers can be
    combined into multiple channels of the same image OR multiple label images.

    The channels are then stacked into a single multiscale image,
    the labels are treated as separate multiscale images.
    """
    axes = layers_data[0][1]["axis_labels"]

    image_layers: list[LayerData] = [
        layer for layer in layers_data if layer[1].get("type") == "image"
    ]
    label_layers: list[LayerData] = [
        layer for layer in layers_data if layer[1].get("type") == "labels"
    ]

    # stack image layers into a single multiscale image
    # Check whether all image layers are multiscale
    label_imgs = []
    for layer in label_layers:
        label_imgs.append(_napari_labels_to_ome_zarr(path, layer[0], layer[1]))

    # if only a single image layer exists, we can fall back to our existing
    # single image layer handling routine
    if len(image_layers) == 1:
        img_ms = _napari_image_to_ome_zarr(path, image_layers[0][0], image_layers[0][1])

    # if multiple image layers exist, we need to stack them along a new channel axis
    elif len(image_layers) > 1:
        stacked_data, new_axes = _stack_layers_by_resolution(image_layers, axes)
        attrs = image_layers[0][1].copy()
        attrs["axis_labels"] = new_axes
        img_ms = _napari_layer_to_ome_zarr(path, stacked_data, attrs)

    # add labels to the multiscale image if any label layers exist
    if len(label_layers) > 0:
        img_ms.labels = label_imgs

    return img_ms


def _napari_layer_to_ome_zarr(
    path: str, layer_data, attributes: dict
) -> OMEZarrMultiscale | OMEZarrLabels:
    """
    Convert the given layer data and attributes into an OME-Zarr multiscale image.
    If the image is not already a multiscale, a single-scale image is converted into a multiscale.
    If the image is already a multiscale, the provided resolution levels are used.
    """

    if layer_data[2] == "image":
        return _napari_image_to_ome_zarr(path, layer_data, attributes)
    elif layer_data[2] == "labels":
        return _napari_labels_to_ome_zarr(path, layer_data, attributes)
    else:
        raise ValueError(f"Unsupported layer type: {layer_data[2]}")


def _napari_image_to_ome_zarr(
    path: str, layer_data, attributes: dict
) -> OMEZarrMultiscale:
    """
    Convert a napari image layer to an OME-Zarr multiscale image class.
    """
    axes = attributes["axis_labels"]

    # check if first color is black - this indicates a non-inverted colormap
    first_color = attributes["colormap"]["colors"][0]
    if all([i == 0 for i in first_color[:-1]]):
        rgba = np.asarray(attributes["colormap"]["colors"][-1])
    else:
        rgba = np.asarray(first_color)

    # If the image is not already a multiscale, compute resolution levels
    if not attributes.get("multiscale", False):
        img = OMEZarrImage(
            data=layer_data,
            axes=axes,
            scale={d: float(attributes["scale"][i]) for i, d in enumerate(axes)},
            name=attributes.get("name", "image"),
        )
        img_ms = OMEZarrMultiscale.from_singlescale(
            image=img,
            contrast_limits=[attributes.get("contrast_limits", None)],
            channel_colors=[[int(c * 255) for c in rgba]],
        )
        return img_ms

    # If the image is already a multiscale, use the provided resolution levels
    resolution_levels: list = []
    for i in range(len(layer_data)):
        img = OMEZarrImage(
            data=layer_data[i],
            axes=axes,
            scale={d: float(attributes["scale"][i]) for i, d in enumerate(axes)},
            name=attributes.get("name", "image"),
        )
        resolution_levels.append(img)

    img_ms = OMEZarrMultiscale.from_pyramid(
        images=resolution_levels,
        contrast_limits=[attributes.get("contrast_limits", None)],
        channel_colors=[[int(c * 255) for c in rgba]],
    )
    return img_ms


def _napari_labels_to_ome_zarr(
    path: str, layer_data, attributes: dict
) -> OMEZarrLabels:
    """
    Convert a napari labels layer to an OME-Zarr labels class.
    """
    axes = attributes["axis_labels"]

    # If the labels are not already a multiscale, compute resolution levels
    if not attributes.get("multiscale", False):
        img = OMEZarrImage(
            data=layer_data,
            axes=axes,
            scale={d: float(attributes["scale"][i]) for i, d in enumerate(axes)},
            name=attributes.get("name", "labels"),
        )
        img_ms = OMEZarrLabels.from_singlescale(image=img)
        return img_ms

    # If the labels are already a multiscale, use the provided resolution levels
    resolution_levels: list = []
    for i in range(len(layer_data)):
        img = OMEZarrImage(
            data=layer_data[i],
            axes=axes,
            scale={d: float(attributes["scale"][i]) for i, d in enumerate(axes)},
            name=attributes.get("name", "labels"),
        )
        resolution_levels.append(img)

    img_ms = OMEZarrLabels.from_pyramid(images=resolution_levels)
    return img_ms
