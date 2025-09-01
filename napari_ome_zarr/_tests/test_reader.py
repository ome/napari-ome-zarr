from pathlib import Path

import numpy as np
import pytest
from ome_zarr.data import astronaut, create_zarr

from napari_ome_zarr._reader import napari_get_reader


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
            assert [[0, 255]] * 3 == metadata["contrast_limits"]
            assert [visible_1] * 3 == metadata["visible"]
        else:
            assert "channel_axis" not in metadata
            assert metadata["name"] == "channel_0"
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
        pth_to_image = str(getattr(self, path))
        print(f"test_image {pth_to_image}")
        layers = napari_get_reader(pth_to_image)()
        self.assert_layers(layers, True, False, path)

    def test_labels(self):
        filename = str(self.path_3d / "labels")
        print(f"test_labels {filename}")
        layers = napari_get_reader(filename)()
        self.assert_layers(layers, True, False)

    def test_label(self):
        filename = str(self.path_3d / "labels" / "astronaut")
        layers = napari_get_reader(filename)()
        self.assert_layers(layers, True, False)

    @pytest.mark.skip(
        reason="Qt builds are failing on all platforms",
    )
    def test_viewer(self, make_napari_viewer):
        """example of testing the viewer."""
        viewer = make_napari_viewer()

        shapes = [(4000, 3000), (2000, 1500), (1000, 750), (500, 375)]
        np.random.seed(0)
        data = [np.random.random(s) for s in shapes]
        _ = viewer.add_image(data, multiscale=True, contrast_limits=[0, 1])
        layer = viewer.layers[0]

        # Set canvas size to target amount
        viewer.window.qt_viewer.view.canvas.size = (800, 600)
        # FutureWarning: Public access to Window.qt_viewer is deprecated
        # and will be removed in v0.6.0
        try:
            viewer.window.qt_viewer.on_draw(None)

            # Check that current level is first large enough to fill the canvas with
            # a greater than one pixel depth
            assert layer.data_level == 2
        except AttributeError:
            pass
