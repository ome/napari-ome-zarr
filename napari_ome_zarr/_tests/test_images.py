from napari_ome_zarr._tests.conftest import bioformats2raw, create_overlap_tiles_scene
from napari_ome_zarr.utils import count_layers_in_scene


def test_images(image_path, make_napari_viewer):
    """
    Test opening some standard images
    """
    viewer = make_napari_viewer()

    viewer.open(str(image_path), plugin="napari-ome-zarr")

    assert len(viewer.layers) > 0


def test_bioformats2raw(make_napari_viewer):
    """
    Test opening a bioformats2raw converted OME-Zarr dataset.
    """
    viewer = make_napari_viewer()

    viewer.open(str(bioformats2raw()), plugin="napari-ome-zarr")

    assert len(viewer.layers) > 0


def test_scenes(tmp_path, make_napari_viewer):
    """
    Test opening some locally built OME-Zarr scenes.
    """
    from napari.layers import Image, Labels
    from ome_zarr import OMEZarrScene

    path = create_overlap_tiles_scene(tmp_path)

    # open in viewer
    viewer = make_napari_viewer()
    viewer.open(str(path), plugin="napari-ome-zarr")
    assert len(viewer.layers) > 0

    # count layers in the scene
    # and compare against napari
    scene = OMEZarrScene.from_ome_zarr(str(path))
    counted = count_layers_in_scene(scene)
    n_images = len([layer for layer in viewer.layers if isinstance(layer, Image)])
    n_labels = len([layer for layer in viewer.layers if isinstance(layer, Labels)])

    assert counted["image_layers"] == n_images
    assert counted["label_layers"] == n_labels
