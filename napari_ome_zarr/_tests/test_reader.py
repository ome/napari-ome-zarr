import math
from pathlib import Path

import numpy as np
import pytest
import zarr
from napari.utils.colormaps import AVAILABLE_COLORMAPS, Colormap
from ome_zarr import OMEZarrMultiscale
from ome_zarr.data import astronaut, create_zarr
from ome_zarr.writer import (
    write_image,
    write_labels,
    write_plate_metadata,
    write_well_metadata,
)

from napari_ome_zarr._reader import napari_get_reader
from napari_ome_zarr.ome_zarr_reader import _match_colors_to_available_colormap
from napari_ome_zarr.utils import count_layers_in_image


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
    def test_reader(self, path, make_napari_viewer):

        viewer = make_napari_viewer()

        path_str = str(getattr(self, path))
        viewer.open(path=path_str, plugin="napari-ome-zarr")

        image = OMEZarrMultiscale.from_ome_zarr(path_str)

        # Check that we get the correct amount of layers in the viewer
        n_layers = count_layers_in_image(image)
        assert len(viewer.layers) == n_layers["image_layers"] + n_layers["label_layers"]

        # check that we have the correct properties in the viewer
        for layer in viewer.layers:
            assert layer.axis_labels == ("y", "x")
            assert layer.units[0] == "pixel"
            assert layer.units[1] == "pixel"

        # check that all channel properties have been passed through
        if hasattr(image, "omero") and image.omero is not None:
            for ch in image.omero.channels:
                ch_name = f"{image.name}: {ch.label}"
                assert ch_name in viewer.layers

                # assert channel display settings
                contrast_limits = [ch.window.start, ch.window.end]
                assert viewer.layers[ch_name].contrast_limits == contrast_limits
                assert viewer.layers[ch_name].visible == ch.active

        # check the same for labels
        if hasattr(image, "labels") and image.labels is not None:
            for label_key in image.labels.keys():
                assert label_key in viewer.layers

                # we default labels to not visible here
                assert viewer.layers[label_key].visible is False

        # Check that colormaps are recognized correctly
        if image.name == "astronaut":
            assert (
                viewer.layers["astronaut: Red"].colormap == AVAILABLE_COLORMAPS["red"]
            )
            assert (
                viewer.layers["astronaut: Green"].colormap
                == AVAILABLE_COLORMAPS["green"]
            )
            assert (
                viewer.layers["astronaut: Blue"].colormap == AVAILABLE_COLORMAPS["blue"]
            )

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

        # data, metadata, layer_type = self.assert_layer(image)
        if path == "path_3d":
            assert len(layers) == 4
            image_c1, image_c2, image_c3, label = layers
            # TODO: Update name check once OMEZarrScene merged
            # with https://github.com/ome/ome-zarr-py/pull/622
            # assert ["Red", "Green", "Blue"] == metadata["name"]
            # assert [
            #     AVAILABLE_COLORMAPS["red"],
            #     AVAILABLE_COLORMAPS["green"],
            #     AVAILABLE_COLORMAPS["blue"],
            # ] == metadata["colormap"]
            # assert [[0, 255]] * 3 == metadata["contrast_limits"]
            # assert [visible_1] * 3 == metadata["visible"]
        else:
            assert len(layers) == 2
            image, label = layers
            data, metadata, layer_type = self.assert_layer(image)
            # TODO: Update name check once OMEZarrScene merged
            # with https://github.com/ome/ome-zarr-py/pull/622
            # assert metadata["name"] == "channel_0"
            # assert metadata["colormap"] == AVAILABLE_COLORMAPS["gray"]
            # assert metadata["contrast_limits"] == [0, 255]
            # assert metadata["visible"] == visible_1

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
        path_to_image = str(getattr(self, path))
        print(f"test_image {path_to_image}")
        layers = napari_get_reader(path_to_image)()
        self.assert_layers(layers, True, False, path)

    def test_labels(self):
        filename = str(self.path_3d / "labels")
        print(f"test_labels {filename}")
        layers = napari_get_reader(filename)()
        self.assert_layers(layers, True, False)

    def test_label(self):
        filename = str(self.path_3d / "labels" / "astronaut_labels")
        layers = napari_get_reader(filename)()
        self.assert_layers(layers, True, False)


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


SPATIAL_UNITS = {"z": "micrometer", "y": "micrometer", "x": "micrometer"}


def test_units_forwarded(tmp_path: Path):
    """NGFF v0.4+ axes carry per-axis ``unit``; forward into napari ``units``."""
    path = tmp_path / "with_units.zarr"
    grp = zarr.open_group(str(path), mode="w")
    image = np.zeros((1, 4, 8, 8), dtype=np.uint8)
    axes = [
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space"},
        {"name": "y", "type": "space"},
        {"name": "x", "type": "space"},
    ]
    # units are supplied via the ``axes_units`` argument: write_image ignores a
    # ``unit`` key embedded in the axes dicts (ome-zarr >=0.18).
    write_image(image=image, group=grp, axes=axes, axes_units=SPATIAL_UNITS)

    layers = napari_get_reader(str(path))()
    assert len(layers) == 1
    _, metadata, _ = layers[0]
    assert metadata["axis_labels"] == ("z", "y", "x")
    assert metadata["units"] == ("micrometer", "micrometer", "micrometer")


def test_label_with_channel_axis_keeps_all_axes(tmp_path: Path):
    """A label is one un-split layer, so it must keep the channel axis in its
    axis_labels (regression: previously the channel axis was dropped from
    axis_labels but not from the data, so axis_labels length != layer ndim and
    napari raised ``axis_labels must have length ndim``)."""
    path = tmp_path / "img_with_label.zarr"
    root = zarr.open_group(str(path), mode="w")
    axes = [
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space"},
        {"name": "y", "type": "space"},
        {"name": "x", "type": "space"},
    ]
    write_image(
        image=np.zeros((2, 4, 8, 8), dtype=np.uint8),
        group=root,
        axes=axes,
        axes_units=SPATIAL_UNITS,
    )
    write_labels(
        labels=np.zeros((1, 4, 8, 8), dtype=np.int8),
        group=root,
        name="lbl",
        axes=axes,
        axes_units=SPATIAL_UNITS,
    )

    layers = napari_get_reader(str(path))()
    image = next(layer for layer in layers if layer[2] == "image")
    label = next(layer for layer in layers if layer[2] == "labels")

    # image: napari splits on the channel axis, so it drops to spatial axes only
    assert image[1]["axis_labels"] == ("z", "y", "x")
    assert image[1]["units"] == ("micrometer", "micrometer", "micrometer")

    # label: not split, so the channel axis is retained and axis_labels length
    # must equal the (4D) layer ndim
    # assert "channel_axis" not in label[1]
    assert label[1]["axis_labels"] == ("z", "y", "x")
    assert len(label[1]["axis_labels"]) == label[0][0].ndim
    # units are forwarded per-axis: the unit-less channel axis stays None so the
    # spatial units still reach napari (otherwise the whole units tuple would be
    # dropped and the label would be unit-inconsistent with the split images,
    # suppressing the scale bar)
    assert label[1]["units"] == ("micrometer", "micrometer", "micrometer")
    assert len(label[1]["units"]) == label[0][0].ndim


class TestPlates:
    @pytest.fixture(autouse=True)
    def initdir(self, tmp_path: Path):
        """
        Write some temporary test data.

        create_zarr() creates an image pyramid and labels zarr directories.
        """
        self.plate_path = tmp_path / "plate.zarr"
        print(f"Creating test plate at {self.plate_path}")

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
        assert metadata["axis_labels"] == ("field", "z", "y", "x")

        # check plate compared with an Image
        well_path = self.plate_path / self.well_paths[0] / "0"
        img_layers = napari_get_reader(str(well_path))()
        assert len(img_layers) == 3
        img_layer = img_layers[0]
        img_data, img_metadata, img_layer_type = img_layer
        assert img_metadata["axis_labels"] == ("z", "y", "x")

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
                    print(
                        "well_path", well_path, "expected_pixel_val", expected_pixel_val
                    )
                    # check pixel at top-left of each Well
                    well_coord_y = tiley * row_idx
                    well_coord_x = tilex * col_idx
                    assert (
                        data_n[0, 0, well_coord_y, well_coord_x].compute()
                        == expected_pixel_val
                    )
                    # check pixel in centre of each Well - same value
                    well_coord_y = tiley * row_idx + tiley // 2
                    well_coord_x = tilex * col_idx + tilex // 2
                    assert (
                        data_n[0, 0, well_coord_y, well_coord_x].compute()
                        == expected_pixel_val
                    )

            tilex = math.ceil(tilex / 2)
            tiley = math.ceil(tiley / 2)


def test_missing_units(tmp_path, make_napari_viewer):
    """
    This test checks whether an OME-Zarr image where not
    all axis units are specified is loaded correctly.
    """
    from ome_zarr import OMEZarrImage, OMEZarrMultiscale

    viewer = make_napari_viewer()

    data = np.random.random((3, 128, 128))

    oz_img = OMEZarrImage(
        data=data,
        axes="tyx",
        axes_units={"y": "meter", "x": "meter"},
        scale={"t": 1, "y": 1, "x": 1},
    )
    oz_ms = OMEZarrMultiscale(image=oz_img)
    oz_ms.to_ome_zarr(str(tmp_path / "test_missing_units.ome.zarr"), overwrite=True)

    layer = viewer.open(
        str(tmp_path / "test_missing_units.ome.zarr"), plugin="napari-ome-zarr"
    )[0]

    assert layer.units == ("pixel", "meter", "meter")
