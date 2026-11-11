def test_plates(plate_path, make_napari_viewer):
    viewer = make_napari_viewer()

    viewer.open(str(plate_path), plugin="napari-ome-zarr")

    assert len(viewer.layers) > 0


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
