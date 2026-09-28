import numpy as np
import pytest
import zarr
from ome_zarr import OMEZarrImage, OMEZarrLabels, OMEZarrMultiscale, OMEZarrScene
from ome_zarr.data import astronaut, create_zarr
from ome_zarr.writer import write_image, write_plate_metadata, write_well_metadata
from ome_zarr_models.v06.coordinate_transforms import (
    Axis,
    CoordinateSystem,
    CoordinateSystemIdentifier,
    Sequence,
    Translation,
)
from skimage import data


def create_overlap_tiles_scene(tmp_path):
    """
    Write a scene with four overlapping tiles of a 2D image to `path`.
    """
    image = data.human_mitosis()

    world_cs = CoordinateSystem(
        name="world",
        axes=(
            Axis(name="y", type="space", unit="micrometer"),
            Axis(name="x", type="space", unit="micrometer"),
        ),
    )

    # split into four overlapping tiles
    tiles = []
    transforms = []
    overlap = 50  # pixels
    height, width = image.shape[:2]
    for y in range(0, height, height // 2):
        for x in range(0, width, width // 2):
            y_end = min(y + height // 2 + overlap, height)
            x_end = min(x + width // 2 + overlap, width)
            tile = image[y:y_end, x:x_end]

            binary = tile > tile.mean()

            oz_binary = OMEZarrImage(
                data=binary,
                axes="yx",
                scale={"y": 0.5, "x": 0.5},
                axes_units={"y": "micrometer", "x": "micrometer"},
                name=f"binary_tile_{y}_{x}",
            )

            oz_binary_ms = OMEZarrLabels(
                oz_binary,
            )

            oz_image = OMEZarrImage(
                data=tile,
                axes="yx",
                scale={"y": 0.5, "x": 0.5},
                axes_units={"y": "micrometer", "x": "micrometer"},
                name=f"tile_{y}_{x}",
            )
            oz_ms = OMEZarrMultiscale(
                oz_image,
                channel_names=["Brightfield"],
                channel_colors=["FFFFFF"],
                labels=oz_binary_ms,
            )

            translation = Translation(
                translation=(y, x),
                input=CoordinateSystemIdentifier(name="physical", path=f"tile_{y}_{x}"),
                output=CoordinateSystemIdentifier(name="world"),
            )
            transforms.append(translation)
            tiles.append(oz_ms)

    scene = OMEZarrScene(
        images=tiles,
        coordinate_systems=(world_cs,),
        coordinate_transformations=transforms,
    )

    path = tmp_path / "scene_tiles_overlap.ome.zarr"
    scene.to_ome_zarr(str(path), overwrite=True)
    return path


def create_YX_to_CZYX_scene(tmp_path):
    """
    Write a scene with a 2D image embedded in a 3D coordinate system
    that also has a channel dimension to `path`.
    """

    img = data.cells3d().transpose((1, 0, 2, 3))
    some_slice = img[0, 30, :, :]

    oz_img = OMEZarrImage(
        data=img,
        axes=["c", "z", "y", "x"],
        scale={"c": 1, "z": 1, "y": 1, "x": 1},
        name="cells3d",
    )

    oz_ms = OMEZarrMultiscale(
        image=oz_img,
    )

    slice_img = OMEZarrImage(
        data=some_slice, axes=["y", "x"], scale={"y": 1, "x": 1}, name="cells3d_slice"
    )

    slice_ms = OMEZarrMultiscale(
        image=slice_img,
    )

    transform_to_3d = Sequence.model_validate(
        {
            "type": "sequence",
            "input": {"path": "cells3d_slice", "name": "physical"},
            "output": {"path": "cells3d", "name": "physical"},
            "transformations": [
                {"type": "projectAxis", "createdOutputs": [0, 1]},
                {"type": "translation", "translation": [0, 30, 0, 0]},
            ],
        }
    )

    scene = OMEZarrScene(
        images=[oz_ms, slice_ms], coordinate_transformations=[transform_to_3d]
    )
    path = tmp_path / "scene_YX_to_CZYX.ome.zarr"
    scene.to_ome_zarr(str(path), overwrite=True)
    return path


def create_YXto_ZYX_scene(tmp_path):
    """
    Write a scene with a 2D image embedded in a 3D coordinate system to `path`.
    """

    img = data.cells3d().transpose((1, 0, 2, 3))
    some_slice = img[0, 30, :, :]

    oz_img = OMEZarrImage(
        data=img[0],
        axes=["z", "y", "x"],
        scale={"z": 1, "y": 1, "x": 1},
        name="cells3d",
    )

    oz_ms = OMEZarrMultiscale(
        image=oz_img,
    )

    slice_img = OMEZarrImage(
        data=some_slice, axes=["y", "x"], scale={"y": 1, "x": 1}, name="cells3d_slice"
    )

    slice_ms = OMEZarrMultiscale(
        image=slice_img,
    )

    transform_to_3d = Sequence.model_validate(
        {
            "type": "sequence",
            "input": {"path": "cells3d_slice", "name": "physical"},
            "output": {"path": "cells3d", "name": "physical"},
            "transformations": [
                {"type": "projectAxis", "createdOutputs": [0]},
                {"type": "translation", "translation": [30, 0, 0]},
            ],
        }
    )

    scene = OMEZarrScene(
        images=[oz_ms, slice_ms], coordinate_transformations=[transform_to_3d]
    )
    path = tmp_path / "scene_YX_to_ZYX.ome.zarr"
    scene.to_ome_zarr(str(path), overwrite=True)
    return path


def _plain_image(path):
    create_zarr(str(path))
    return path


def _image_with_labels(path):
    create_zarr(str(path), method=astronaut, label_name="astronaut")
    return path


def bioformats2raw():
    """
    Return the path to a bioformats2raw converted OME-Zarr dataset.
    """
    return "https://livingobjects.ebi.ac.uk/idr/zarr/v0.4/idr0079A/idr0079_images.zarr"


@pytest.fixture(
    params=[create_overlap_tiles_scene, create_YX_to_CZYX_scene, create_YXto_ZYX_scene]
)
def scene_path(request, tmp_path):
    """
    Path to a locally-built OME-Zarr scene.
    """
    return request.param(tmp_path)


@pytest.fixture(params=[_plain_image, _image_with_labels])
def image_path(request, tmp_path):
    """Path to a locally-built OME-Zarr image, with and without labels."""
    return request.param(tmp_path / "image.zarr")


@pytest.fixture
def plate_path(tmp_path):
    """Path to a locally-built single-well OME-Zarr plate."""
    path = tmp_path / "plate.zarr"
    root = zarr.open_group(str(path))
    write_plate_metadata(root, ["A"], ["1"], ["A/1"])
    well_group = root.require_group("A").require_group("1")
    write_well_metadata(well_group, ["0"])
    image_group = well_group.require_group("0")
    write_image(
        image=np.ones((1, 1, 8, 8), dtype=np.uint8),
        group=image_group,
        axes="czyx",
        scale_factors=[],  # 8x8 too small for default (2,4,8,16) pyramid
    )
    return path
