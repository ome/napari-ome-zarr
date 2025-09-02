import math
from pathlib import Path

import numpy as np
import pytest
import zarr
from napari.utils.colormaps import AVAILABLE_COLORMAPS, Colormap
from ome_zarr.data import astronaut, create_zarr
from ome_zarr.writer import write_image, write_plate_metadata, write_well_metadata

from napari_ome_zarr._reader import (
    _match_colors_to_available_colormap,
    napari_get_reader,
)


class TestNapari:
    @pytest.fixture(autouse=True)
    def initdir(self, tmp_path: Path):
        """
        Write some temporary test data.

        create_zarr() creates an image pyramid and labels zarr directories.
        """
        self.path_3d = tmp_path / "data_3d"
        self.path_3d.mkdir()
        create_zarr(str(self.path_3d), method=astronaut, label_name="astronaut")

        self.path_2d = tmp_path / "data_2d"
        self.path_2d.mkdir()
        create_zarr(str(self.path_2d))

    def test_get_reader_hit(self):
        reader = napari_get_reader(str(self.path_3d))
        assert reader is not None
        assert callable(reader)

    @pytest.mark.parametrize("path", ["path_3d", "path_2d"])
    def test_reader(self, path):
        path_str = str(getattr(self, path))
        reader = napari_get_reader(path_str)
        results = reader(path_str)
        assert len(results) == 2
        image, label = results
        assert isinstance(image[0], list)
        assert isinstance(image[1], dict)
        if path == "path_3d":
            assert image[1]["channel_axis"] == 0
            assert image[1]["name"] == ["Red", "Green", "Blue"]
        else:
            assert "channel_axis" not in image[1]
            assert image[1]["name"] == "channel_0"

    @pytest.mark.parametrize("path", ["path_3d", "path_2d"])
    def test_get_reader_with_list(self, path):
        # a better test here would use real data
        reader = napari_get_reader([str(getattr(self, path))])
        assert reader is not None
        assert callable(reader)

    def test_get_reader_pass(self):
        reader = napari_get_reader("fake.file")
        assert reader is None

    def assert_layers(self, layers, visible_1, visible_2, path="path_3d"):
        # TODO: check name

        assert len(layers) == 2
        image, label = layers

        data, metadata, layer_type = self.assert_layer(image)
        if path == "path_3d":
            assert 0 == metadata["channel_axis"]
            assert ["Red", "Green", "Blue"] == metadata["name"]
            assert [
                AVAILABLE_COLORMAPS["red"],
                AVAILABLE_COLORMAPS["green"],
                AVAILABLE_COLORMAPS["blue"],
            ] == metadata["colormap"]
            assert [[0, 255]] * 3 == metadata["contrast_limits"]
            assert [visible_1] * 3 == metadata["visible"]
        else:
            assert "channel_axis" not in metadata
            assert metadata["name"] == "channel_0"
            assert metadata["colormap"] == AVAILABLE_COLORMAPS["gray"]
            assert metadata["contrast_limits"] == [0, 255]
            assert metadata["visible"] == visible_1

        data, metadata, layer_type = self.assert_layer(label)
        assert visible_2 == metadata["visible"]

    def assert_layer(self, layer_data):
        data, metadata, layer_type = layer_data
        if not data or not metadata:
            assert False, f"unknown layer: {layer_data}"
        assert layer_type in ("image", "labels")
        return data, metadata, layer_type

    @pytest.mark.parametrize("path", ["path_3d", "path_2d"])
    def test_image(self, path):
        layers = napari_get_reader(str(getattr(self, path)))()
        self.assert_layers(layers, True, False, path)

    def test_labels(self):
        filename = str(self.path_3d / "labels")
        layers = napari_get_reader(filename)()
        self.assert_layers(layers, False, True)

    def test_label(self):
        filename = str(self.path_3d / "labels" / "astronaut")
        layers = napari_get_reader(filename)()
        self.assert_layers(layers, False, True)


@pytest.mark.parametrize(
    "colors, expected_name",
    [
        ([[0, 0, 0], [0.0, 0.0, 1.0]], "blue"),  # Existing napari colormap
        ([[0, 0, 0], [0.0, 0.0, 0.9]], "custom"),  # Custom colormap
    ],
)
def test_match_colors_to_available_colormap(colors, expected_name):
    colormap = Colormap(colors)
    colormap = _match_colors_to_available_colormap(colormap)
    assert colormap.name == expected_name


class TestPlates:
    @pytest.fixture(autouse=True)
    def initdir(self, tmp_path: Path):
        """
        Write some temporary test data.

        create_zarr() creates an image pyramid and labels zarr directories.
        """
        self.plate_path = tmp_path / "plate.zarr"

        self.row_names = ["A", "B"]
        self.col_names = ["1", "2", "3"]
        self.well_paths = ["A/1", "A/2", "B/1", "B/3"]
        self.field_paths = ["0", "1", "2"]
        self.sizex = 1000
        self.sizey = 500
        self.sizez = 10
        self.sizec = 3

        def generate_data(well_idx, field_idx):
            return np.ones(
                (self.sizec, self.sizez, self.sizey, self.sizex), dtype=np.uint8
            ) * (well_idx * 10 + field_idx * 5)

        # write the plate of images and corresponding metadata
        root = zarr.open_group(self.plate_path)
        write_plate_metadata(root, self.row_names, self.col_names, self.well_paths)
        for wi, wp in enumerate(self.well_paths):
            row, col = wp.split("/")
            row_group = root.require_group(row)
            well_group = row_group.require_group(col)
            write_well_metadata(well_group, self.field_paths)
            for fi, field in enumerate(self.field_paths):
                image_group = well_group.require_group(str(field))
                write_image(image=generate_data(wi, fi), group=image_group, axes="czyx")

    def test_read_plate(self):
        layers = napari_get_reader(str(self.plate_path))()
        assert len(layers) == 1
        plate = layers[0]
        data, metadata, layer_type = plate
        assert data[0].shape == (
            self.sizec,
            self.sizez,
            self.sizey * len(self.row_names),
            self.sizex * len(self.col_names),
        )

        # check plate compared with an Image
        well_path = self.plate_path / self.well_paths[0]
        img_layers = napari_get_reader(str(well_path))()
        assert len(img_layers) == 1
        img_layer = img_layers[0]
        img_data, img_metadata, img_layer_type = img_layer

        # plate pyramid should have same number of resolutions as images
        assert len(img_data) == len(data)

        tilex = self.sizex
        tiley = self.sizey
        for data_n in data:
            for col_idx, col in enumerate(self.col_names):
                for row_idx, row in enumerate(self.row_names):
                    well_path = f"{row}/{col}"
                    expected_pixel_val = 0
                    if well_path in self.well_paths:
                        well_idx = self.well_paths.index(well_path)
                        # field is 0
                        expected_pixel_val = well_idx * 10
                    # check pixel at top-left of each Well
                    well_coord_y = tiley * row_idx
                    well_coord_x = tilex * col_idx
                    assert (
                        data_n[0, 0, well_coord_y, well_coord_x] == expected_pixel_val
                    )
                    # check pixel in centre of each Well - same value
                    well_coord_y = tiley * row_idx + tiley // 2
                    well_coord_x = tilex * col_idx + tilex // 2
                    assert (
                        data_n[0, 0, well_coord_y, well_coord_x] == expected_pixel_val
                    )

            tilex = math.ceil(tilex / 2)
            tiley = math.ceil(tiley / 2)
