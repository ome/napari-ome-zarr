import pytest

from napari_ome_zarr._tests.conftest import _image_with_labels, _plain_image


@pytest.mark.parametrize(
    "build, n_image_layers, n_label_layers",
    [
        (_plain_image, 1, 1),  # coins: single-channel image + 1 label
        (_image_with_labels, 3, 1),  # astronaut: 3-channel image + 1 label
    ],
)
def test_counting_image_layers(build, n_image_layers, n_label_layers, tmp_path):
    from ome_zarr import OMEZarrMultiscale

    from napari_ome_zarr.utils import count_layers_in_image

    path = build(tmp_path / "image.zarr")

    image = OMEZarrMultiscale.from_ome_zarr(str(path))
    n_layers = count_layers_in_image(image)
    assert n_layers == {
        "image_layers": n_image_layers,
        "label_layers": n_label_layers,
    }
