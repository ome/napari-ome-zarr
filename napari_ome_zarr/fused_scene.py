"""
Fuse the layers of an OME-Zarr scene (as made by Scene.to_layer_data) into a single
layer per channel and per label image, each a lazy Dask canvas per pyramid level.

Each layer is placed by its napari transform (affine @ scale), so the fused layer shows
the same as the separate layers, but napari only needs to render one layer instead of
one per image. Layers only shifted relative to the canvas pixels are copied (rounded to
whole pixels), others (rotated, scaled) are resampled. Overlapping pixels show the
maximum value for images, and the last non-zero value for labels.
"""

import itertools
import warnings
from typing import Any, Dict, List, Tuple

import dask.array as da
import numpy as np
from ome_zarr import OMEZarrScene
from scipy import ndimage

from .utils import count_layers_in_image

LayerData = Tuple[List[da.core.Array], Dict[str, Any], str]

# chunk size of the last two canvas axes;
# 1024 or 2048 are usually the sweet spots for GPU texture uploads
CHUNK_SIZE = 1024


def _box_corners(lo, hi) -> np.ndarray:
    """All corners of the box [lo, hi], as rows of homogeneous coordinates."""
    corners = np.array(list(itertools.product(*zip(lo, hi))), dtype=float)
    return np.hstack([corners, np.ones((len(corners), 1))])


def _layer_matrices(layer: LayerData) -> List[np.ndarray]:
    """
    Per pyramid level: homogeneous matrix from array index to world,
    as napari displays the layer.
    """
    data, props, _ = layer
    ndim = data[0].ndim
    affine = props.get("affine")
    affine = (
        np.eye(ndim + 1)
        if affine is None
        else np.asarray(getattr(affine, "affine_matrix", affine), dtype=float)
    )
    data_to_world = np.eye(ndim + 1)
    data_to_world[:-1, :-1] = np.diag(props.get("scale", np.ones(ndim)))
    data_to_world[:-1, -1] = props.get("translate", np.zeros(ndim))
    data_to_world = affine @ data_to_world
    # napari places pixel i of a multiscale level at level 0 pixel i * downsample factor
    level0_shape = np.array(data[0].shape)
    return [
        data_to_world @ np.diag(list(level0_shape / np.array(level.shape)) + [1])
        for level in data
    ]


class _Placement:
    """
    A layer pyramid level, placed in a fused canvas level
    by the matrix from layer to canvas index.
    """

    def __init__(self, array: da.core.Array, matrix: np.ndarray, order: int):
        ndim = len(matrix) - 1
        self.array = array
        self.order = order
        self.shape = np.array(array.shape)
        self.inverse = np.linalg.inv(matrix)
        if np.allclose(matrix[:-1, :-1], np.eye(ndim), atol=1e-6):
            # only shifted: copy, rounded to whole pixels
            self.offset = np.round(matrix[:-1, -1]).astype(int)
            self.lo, self.hi = self.offset, self.offset + self.shape
        else:
            # rotated / scaled: resample
            self.offset = None
            corners = _box_corners(-0.5 * np.ones(ndim), self.shape - 0.5) @ matrix.T
            self.lo = np.floor(corners[:, :-1].min(axis=0)).astype(int)
            self.hi = np.ceil(corners[:, :-1].max(axis=0)).astype(int) + 1

    def read_region(self, lo: np.ndarray, hi: np.ndarray):
        """
        Lazy layer data needed for canvas pixels [lo, hi),
        with the layer index of its first pixel.
        """
        if self.offset is not None:
            return (
                self.array[
                    tuple(
                        slice(a, b) for a, b in zip(lo - self.offset, hi - self.offset)
                    )
                ],
                None,
            )
        corners = _box_corners(lo, hi - 1) @ self.inverse.T
        region_lo = np.clip(
            np.floor(corners[:, :-1].min(axis=0)).astype(int) - 1, 0, self.shape
        )
        region_hi = np.clip(
            np.ceil(corners[:, :-1].max(axis=0)).astype(int) + 2, 0, self.shape
        )
        if np.any(region_lo >= region_hi):
            return None, None
        return (
            self.array[tuple(slice(a, b) for a, b in zip(region_lo, region_hi))],
            region_lo,
        )

    def resample(
        self, data: np.ndarray, region_lo: np.ndarray, lo: np.ndarray, hi: np.ndarray
    ) -> np.ndarray:
        """Layer data at canvas pixels [lo, hi) as float, NaN outside the layer."""
        linear = self.inverse[:-1, :-1]
        offset = linear @ lo + self.inverse[:-1, -1] - region_lo
        return ndimage.affine_transform(
            # float64 keeps label values exact
            data.astype(np.float64 if self.order == 0 else np.float32),
            linear,
            offset,
            output_shape=tuple(hi - lo),
            order=self.order,
            mode="constant",
            cval=np.nan,
        )


def fuse_layers(layers: List[LayerData], name: str, visible: bool = True) -> LayerData:
    """
    Fuse image or labels layers with the same number of dimensions
    into a single multiscale layer.
    """
    layer_type = layers[0][2]
    is_labels = layer_type == "labels"
    ndim = layers[0][0][0].ndim
    dtype = layers[0][0][0].dtype
    tiles = []
    for layer in layers:
        matrices = _layer_matrices(layer)
        # per world axis: the pixel size of each level
        pixel_sizes = [np.linalg.norm(matrix[:-1, :-1], axis=1) for matrix in matrices]
        tiles.append(
            {"pyramid": layer[0], "matrices": matrices, "pixel_sizes": pixel_sizes}
        )

    # canvas: the finest pixel size, covering all layers;
    # canvas pixel 0 is centered at origin
    pixel_size = np.min([tile["pixel_sizes"][0] for tile in tiles], axis=0)
    corners = np.vstack(
        [
            _box_corners(-0.5 * np.ones(ndim), np.array(tile["pyramid"][0].shape) - 0.5)
            @ tile["matrices"][0].T
            for tile in tiles
        ]
    )[:, :-1]
    bbox_min, bbox_max = corners.min(axis=0), corners.max(axis=0)
    origin = bbox_min + pixel_size / 2
    level0_shape = np.maximum(
        np.ceil((bbox_max - bbox_min) / pixel_size - 1e-6).astype(int), 1
    )

    # downsampling factors per level and axis, from the layer with the most levels
    reference = max(tiles, key=lambda tile: len(tile["pyramid"]))
    level_factors = [
        np.maximum(np.round(sizes / reference["pixel_sizes"][0]), 1)
        for sizes in reference["pixel_sizes"]
    ]

    pyramid = []
    for factor in level_factors:
        canvas_matrix = np.eye(ndim + 1)
        canvas_matrix[:-1, :-1] = np.diag(pixel_size * factor)
        canvas_matrix[:-1, -1] = origin
        canvas_inverse = np.linalg.inv(canvas_matrix)

        placements = []
        for tile in tiles:
            # the coarsest layer level that isn't coarser than this canvas level
            tile_level = 0
            for index, sizes in enumerate(tile["pixel_sizes"]):
                if np.all(sizes <= pixel_size * factor * (1 + 1e-3)):
                    tile_level = index
            placements.append(
                _Placement(
                    tile["pyramid"][tile_level],
                    canvas_inverse @ tile["matrices"][tile_level],
                    order=0 if is_labels else 1,
                )
            )

        shape = tuple(np.maximum(np.ceil(level0_shape / factor).astype(int), 1))
        chunks = (1,) * (ndim - 2) + (CHUNK_SIZE,) * min(ndim, 2)
        canvas_template = da.zeros(shape, chunks=chunks, dtype=dtype)

        def render_chunk(block, block_info=None, _placements=placements):
            if block_info is None or len(block_info) == 0:
                return block
            loc = block_info[None]["array-location"]
            chunk_lo = np.array([start for start, _ in loc])
            chunk_hi = np.array([end for _, end in loc])
            chunk_data = np.zeros(block.shape, dtype=block.dtype)

            # 1. collect the lazy slices of the overlapping layers
            tasks = []
            for placement in _placements:
                lo = np.maximum(chunk_lo, placement.lo)
                hi = np.minimum(chunk_hi, placement.hi)
                if np.any(lo >= hi):
                    continue
                region, region_lo = placement.read_region(lo, hi)
                if region is not None:
                    tasks.append((placement, region, region_lo, lo, hi))

            # 2. fetch all overlapping data in parallel
            fetched_data = da.compute(*[task[1] for task in tasks])

            # 3. apply the fetched data to the chunk synchronously
            # to avoid race conditions
            for (placement, _, region_lo, lo, hi), values in zip(tasks, fetched_data):
                target = chunk_data[
                    tuple(slice(a, b) for a, b in zip(lo - chunk_lo, hi - chunk_lo))
                ]
                if region_lo is None:
                    valid = (
                        values != 0 if is_labels else np.ones(values.shape, dtype=bool)
                    )
                else:
                    values = placement.resample(values, region_lo, lo, hi)
                    valid = ~np.isnan(values)
                    if np.issubdtype(block.dtype, np.integer):
                        values = np.round(values)
                    if is_labels:
                        valid &= values != 0
                    values = np.where(valid, values, 0).astype(block.dtype)
                if is_labels:
                    target[valid] = values[valid]
                else:
                    target[valid] = np.maximum(target[valid], values[valid])
            return chunk_data

        pyramid.append(canvas_template.map_blocks(render_chunk, dtype=dtype))

    props = {
        key: value
        for key, value in layers[0][1].items()
        if key not in ("affine", "features")
    }
    props.update(name=name, scale=tuple(pixel_size), translate=tuple(origin))
    if not visible:
        props["visible"] = False
    return pyramid, props, layer_type


def fuse_scene_layers(
    layers: List[LayerData],
    scene: OMEZarrScene,
    coordinate_system: str,
    visible: bool = True,
) -> List[LayerData]:
    """
    Fuse the layers of a scene (Scene.to_layer_data) per channel and label image:
    the i-th channel layer, or j-th label layer, of each scene image.
    Layers are only fused if they have the same dtype, and image layers the same
    channel name; otherwise they are kept as separate layers.
    """
    groups: Dict[Tuple[str, int, int], List[LayerData]] = {}
    names: Dict[Tuple[str, int, int], List[str]] = {}
    start = 0
    # Scene.to_layer_data gives the channel layers and then the label layers
    # of each scene image
    for ms_image in scene.images.values():
        counts = count_layers_in_image(ms_image)
        image_layers = layers[start : start + counts["image_layers"]]
        label_layers = layers[
            start
            + counts["image_layers"] : start
            + counts["image_layers"]
            + counts["label_layers"]
        ]
        label_keys = (
            list(ms_image.labels.keys()) if getattr(ms_image, "labels", None) else []
        )
        start += counts["image_layers"] + counts["label_layers"]
        for index, layer in enumerate(image_layers):
            key = ("image", layer[0][0].ndim, index)
            # image layer names are '<image>: <channel>'
            # (or '<image>' without channel names)
            name = layer[1].get("name", "")
            names.setdefault(key, []).append(
                name.split(": ", 1)[1] if ": " in name else ""
            )
            groups.setdefault(key, []).append(layer)
        for index, (layer, label_key) in enumerate(zip(label_layers, label_keys)):
            key = ("labels", layer[0][0].ndim, index)
            names.setdefault(key, []).append(label_key)
            groups.setdefault(key, []).append(layer)

    fused = []
    for key, group in groups.items():
        group_names = set(names[key])
        dtypes = {layer[0][0].dtype for layer in group}
        if len(dtypes) > 1 or (key[0] == "image" and len(group_names) > 1):
            if len(dtypes) > 1:
                reason = f"different dtypes {sorted(map(str, dtypes))}"
            else:
                reason = f"different channel names {sorted(group_names)}"
            layer_names = [layer[1].get("name") for layer in group]
            warnings.warn(f"Not fusing {key[0]} layers {layer_names}: {reason}")
            for data, props, layer_type in group:
                if not visible:
                    props = {**props, "visible": False}
                fused.append((data, props, layer_type))
            continue
        images = "images" if len(group) > 1 else "image"
        fused_name = f"{coordinate_system} ({len(group)} {images})"
        name = group_names.pop() if len(group_names) == 1 else f"{key[0]} {key[2]}"
        if name:
            fused_name = f"{fused_name}: {name}"
        fused.append(fuse_layers(group, fused_name, visible=visible))
    return fused
