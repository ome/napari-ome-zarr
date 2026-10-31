import numpy as np
import pytest
import zarr
from skimage import data

from napari_ome_zarr._reader import napari_get_reader
from napari_ome_zarr._tests.conftest import (
    create_overlap_tiles_scene,
    create_YX_to_CZYX_scene,
    create_YXto_ZYX_scene,
)
from napari_ome_zarr.ome_zarr_reader import FusedScene

PIXEL_SIZE = 0.5
AXES = [
    {"name": "c", "type": "channel"},
    {"name": "y", "type": "space", "unit": "micrometer"},
    {"name": "x", "type": "space", "unit": "micrometer"},
]


def fused_layers(path):
    return FusedScene(zarr.open_group(str(path), mode="r")).to_layer_data()


def level0(layer) -> np.ndarray:
    return np.asarray(layer[0][0].compute())


def test_overlap_tiles(tmp_path):
    image_layer, labels_layer = fused_layers(create_overlap_tiles_scene(tmp_path))
    assert image_layer[1]["name"] == "world (4 images): Brightfield"
    assert labels_layer[2] == "labels"
    assert labels_layer[1]["name"].startswith("world (4 images)")
    # the tiles are crops of the same image at pixel (y, x), translated by (y, x) micrometers
    image = data.human_mitosis()
    expected = np.zeros((768, 768), dtype=image.dtype)
    for y in (0, 256):
        for x in (0, 256):
            tile = image[y : y + 306, x : x + 306]
            expected[2 * y : 2 * y + tile.shape[0], 2 * x : 2 * x + tile.shape[1]] = tile
    np.testing.assert_array_equal(level0(image_layer), expected)
    assert tuple(image_layer[1]["scale"]) == (PIXEL_SIZE, PIXEL_SIZE)
    assert image_layer[1]["units"] == ("micrometer", "micrometer")
    assert [level.shape for level in image_layer[0]][:2] == [(768, 768), (384, 384)]
    # labels: thresholded tiles
    tile = image[:306, :306]
    np.testing.assert_array_equal(level0(labels_layer)[:306, :306], tile > tile.mean())


@pytest.mark.parametrize("create_scene", [create_YXto_ZYX_scene, create_YX_to_CZYX_scene])
def test_project_axis(tmp_path, create_scene):
    # a z=30 slice of cells3d, placed in the 3D cells3d by a projectAxis
    layers = fused_layers(create_scene(tmp_path))
    channel0 = layers[0]
    assert channel0[1]["name"] == "scene (2 images): Channel 0"
    np.testing.assert_array_equal(level0(channel0), data.cells3d()[:, 0])


def write_tile(root, path, data, extra_transforms=None, axes=AXES):
    """Write a 2-level c/y/x multiscale image, with OME-Zarr 0.6 'sequence' dataset transformations."""
    group = root.create_group(path)
    datasets = []
    for level in range(2):
        factor = 2**level
        level_data = data[:, ::factor, ::factor]
        group.create_array(f"s{level}", data=level_data, chunks=level_data.shape)
        offset = (factor - 1) / 2 * PIXEL_SIZE
        datasets.append({
            "path": f"s{level}",
            "coordinateTransformations": [{
                "type": "sequence",
                "input": {"path": f"s{level}"},
                "output": {"name": "physical"},
                "transformations": [
                    {"type": "scale", "scale": [1.0, PIXEL_SIZE * factor, PIXEL_SIZE * factor]},
                    {"type": "translation", "translation": [0.0, offset, offset]},
                ],
            }],
        })
    coordinate_systems = [{"name": "physical", "axes": axes}]
    if extra_transforms:
        coordinate_systems.append({"name": "rotated", "axes": axes})
    multiscale = {"name": path, "coordinateSystems": coordinate_systems, "datasets": datasets}
    if extra_transforms:
        multiscale["coordinateTransformations"] = extra_transforms
    group.attrs["ome"] = {
        "version": "0.6",
        "multiscales": [multiscale],
        "omero": {"channels": [{"color": "FFFFFF", "label": "ch0", "window": {"min": 0, "max": 255, "start": 0, "end": 255}}]},
    }


def translation(path, output, yx, name="physical"):
    return {
        "type": "translation",
        "translation": [0.0, *yx],
        "input": {"path": path, "name": name},
        "output": {"name": output},
    }


def write_scene(path, transforms, coordinate_systems=("stitched",)):
    root = zarr.open_group(path, mode="r+")
    root.attrs["ome"] = {
        "version": "0.6",
        "scene": {
            "coordinateSystems": [{"name": name, "axes": AXES} for name in coordinate_systems],
            "coordinateTransformations": transforms,
        },
    }


@pytest.fixture
def tiles_path(tmp_path):
    """Two tiles with a marker pixel: a (value 100), b (value 200)."""
    path = tmp_path / "scene.ome.zarr"
    root = zarr.open_group(path, mode="w")
    tile = np.zeros((1, 8, 8), dtype=np.uint8)
    tile[0, 2, 3] = 100
    write_tile(root, "a", tile)
    write_tile(root, "b", tile * 2)
    return path


def test_coordinate_systems(tiles_path):
    write_scene(
        tiles_path,
        [
            # stitched: b 10 pixels below and 3 pixels right of a
            translation("a", "stitched", [1.0, 2.0]),
            translation("b", "stitched", [6.0, 3.5]),
            # grid: b right of a
            translation("a", "grid", [0.0, 0.0]),
            translation("b", "grid", [0.0, 4.0]),
        ],
        coordinate_systems=("stitched", "grid"),
    )
    layers = napari_get_reader(str(tiles_path))(str(tiles_path))
    assert [(layer[1]["name"], layer[1]["visible"]) for layer in layers] == [
        ("stitched (2 images): ch0", True),
        ("grid (2 images): ch0", False),
    ]
    stitched, grid = layers
    assert tuple(stitched[1]["translate"]) == (1.0, 2.0)
    stitched_data = level0(stitched)
    assert stitched_data.shape == (18, 11)
    assert stitched_data[2, 3] == 100
    assert stitched_data[12, 6] == 200
    assert np.count_nonzero(stitched_data) == 2
    grid_data = level0(grid)
    assert grid_data.shape == (8, 16)
    assert grid_data[2, 3] == 100
    assert grid_data[2, 11] == 200
    assert [level.shape for level in grid[0]] == [(8, 16), (4, 8)]


def test_map_axis(tiles_path):
    root = zarr.open_group(tiles_path, mode="r+")
    # c stored as c/x/y
    tile = np.zeros((1, 8, 8), dtype=np.uint8)
    tile[0, 2, 3] = 200
    write_tile(root, "c", tile.transpose(0, 2, 1), axes=[AXES[0], AXES[2], AXES[1]])
    write_scene(
        tiles_path,
        [
            translation("a", "stitched", [1.0, 2.0]),
            {
                "type": "sequence",
                "input": {"path": "c", "name": "physical"},
                "output": {"name": "stitched"},
                "transformations": [
                    {"type": "mapAxis", "mapAxis": [0, 2, 1]},
                    {"type": "translation", "translation": [0.0, 6.0, 3.5]},
                ],
            },
        ],
    )
    (layer,) = fused_layers(tiles_path)
    fused = level0(layer)
    assert tuple(layer[1]["translate"]) == (1.0, 2.0)
    assert fused.shape == (18, 11)
    assert fused[2, 3] == 100
    assert fused[12, 6] == 200
    assert np.count_nonzero(fused) == 2


@pytest.mark.xfail(reason="transformnd ByDimension has no affine (to_affine)", raises=ValueError, strict=True)
def test_by_dimension(tiles_path):
    write_scene(
        tiles_path,
        [
            {
                "type": "byDimension",
                "input": {"path": "a", "name": "physical"},
                "output": {"name": "stitched"},
                "transformations": [
                    {
                        "transformation": {"type": "translation", "translation": [1.0, 2.0]},
                        "inputAxes": [1, 2],
                        "outputAxes": [1, 2],
                    },
                    {"transformation": {"type": "identity"}, "inputAxes": [0], "outputAxes": [0]},
                ],
            },
        ],
    )
    (layer,) = fused_layers(tiles_path)
    assert tuple(layer[1]["translate"]) == (1.0, 2.0)
    assert level0(layer)[2, 3] == 100


def test_rotated_tile_is_resampled(tmp_path):
    path = tmp_path / "rotated.ome.zarr"
    root = zarr.open_group(path, mode="w")
    tile = np.zeros((1, 8, 8), dtype=np.uint8)
    tile[0, 2, 3] = 100
    # 90 degree rotation of y/x in the image's own 'rotated' coordinate system: (y, x) -> (x, -y)
    rotation = {
        "type": "rotation",
        "rotation": [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
        "input": {"name": "physical"},
        "output": {"name": "rotated"},
    }
    write_tile(root, "a", tile, extra_transforms=[rotation])
    write_scene(path, [translation("a", "world", [0.0, 0.0], name="rotated")], coordinate_systems=("world",))
    (layer,) = fused_layers(path)
    fused = level0(layer)
    assert fused.shape == (8, 8)
    # pixel (y=2, x=3) -> physical (1, 1.5) -> rotated (1.5, -1); canvas origin x = -3.5
    assert tuple(layer[1]["translate"]) == (0.0, -3.5)
    assert fused[3, 5] == 100
    assert np.count_nonzero(fused) == 1
