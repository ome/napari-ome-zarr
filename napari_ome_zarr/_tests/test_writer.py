import napari
from skimage import data


def test_write_05_ome_zarr():
    viewer = napari.Viewer()
    image = data.human_mitosis()
    viewer.add_image(image)

    viewer.layers[0].save("test", plugin="napari-ome-zarr")

    viewer.open("test.zarr", plugin="napari-ome-zarr")

    assert len(viewer.layers) == 2
