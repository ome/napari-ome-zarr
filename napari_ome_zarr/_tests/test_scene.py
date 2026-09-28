from napari_ome_zarr._tests.conftest import create_overlap_tiles_scene
from napari_ome_zarr.utils import count_layers_in_scene

from ome_zarr import OMEZarrScene
import numpy as np

def test_scene_in_napari(scene_path, make_napari_viewer):
    from napari.layers import Image, Labels

    viewer = make_napari_viewer()
    viewer.open(path=str(scene_path), plugin="napari-ome-zarr")

    # Check every channel and layer is accounted for
    scene = OMEZarrScene.from_ome_zarr(str(scene_path))
    n_layers = count_layers_in_scene(scene)
    assert len(viewer.layers) == n_layers["image_layers"] + n_layers["label_layers"]

    # Make sure we have the correct amount of labels and image layers
    n_labels_layer_viewer = 0
    n_image_layers_viewer = 0
    for layer in viewer.layers:
        if isinstance(layer, Image):
            n_image_layers_viewer += 1
        elif isinstance(layer, Labels):
            n_labels_layer_viewer += 1

    assert n_image_layers_viewer == n_layers["image_layers"]
    assert n_labels_layer_viewer == n_layers["label_layers"]


def test_properties_forwarding(tmp_path, make_napari_viewer):
    """
    This checks whether the layer properties units, labels, etc
    are correctly populated by the napari-ome-zarr plugin.

    We do this on a specific scene created by the create_overlap_tiles_scene fixture.
    """
    path = create_overlap_tiles_scene(tmp_path)

    viewer = make_napari_viewer()
    viewer.open(path=str(path), plugin="napari-ome-zarr")

    # Make sure we populated the properties correctly
    for layer in viewer.layers:
        assert layer.units[0] == "micrometer"
        assert layer.units[1] == "micrometer"
        assert layer.axis_labels == ("y", "x")

    # check that all layers are named appropriately
    scene = OMEZarrScene.from_ome_zarr(str(path))
    for _, ms_image in scene.images.items():
        if hasattr(ms_image, "omero") and ms_image.omero is not None:
            for ch in ms_image.omero.channels:
                ch_name = f"{ms_image.name}: {ch.label}"
                assert ch_name in viewer.layers
        else:
            assert ms_image.name in viewer.layers

        if hasattr(ms_image, "labels") and ms_image.labels is not None:
            for label_name in ms_image.labels.keys():
                assert label_name in viewer.layers

    # check that scale values have been correctly forwarded
    # to image AND labels layers
    for _, ms_image in scene.images.items():
        layers = [
            layer for layer in viewer.layers if layer.name.startswith(ms_image.name)
        ]
        for layer in layers:
            assert np.array_equal(
                layer.scale, np.asarray(list(ms_image.images[0].scale.values()))
            )

        if hasattr(ms_image, "labels") and ms_image.labels is not None:
            for label_name, label_img in ms_image.labels.items():
                layer = viewer.layers[label_name]
                assert np.array_equal(
                    layer.scale, np.asarray(list(label_img.images[0].scale.values()))
                )

    # Check that the affine matrix has been properly set in napari layers
    for _, ms_image in scene.images.items():
        transform = scene._graph.get_sequence(
            (f"{ms_image.name}", "physical"), ("", "world")
        )
        affine = transform.simplify().to_affine().matrix

        layers = [
            layer for layer in viewer.layers if layer.name.startswith(ms_image.name)
        ]
        for layer in layers:
            assert np.array_equal(layer.affine.affine_matrix, affine)

        # If no tranform between image and labels space is specified
        # the affine should propagate to the labels layers as well
        if hasattr(ms_image, "labels") and ms_image.labels is not None:
            for label_name, label_img in ms_image.labels.items():
                layer = viewer.layers[label_name]
                assert np.array_equal(layer.affine.affine_matrix, affine)
