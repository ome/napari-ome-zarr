from ome_zarr import OMEZarrMultiscale, OMEZarrScene


def count_layers_in_scene(scene: OMEZarrScene) -> dict:
    """
    Browse the images in a scene and sum up how many layers
    should be created. One channel corresponds to one layer.
    One label image corresponds to one layer.
    """
    n_image_layers: int = 0
    n_label_layers: int = 0
    for _, ms_image in scene.images.items():
        layers = count_layers_in_image(ms_image)
        n_image_layers += layers["image_layers"]
        n_label_layers += layers["label_layers"]

    return {"image_layers": n_image_layers, "label_layers": n_label_layers}


def count_layers_in_image(image: OMEZarrMultiscale) -> dict:
    """
    Count the number of image and label layers in a single OMEZarrMultiscale image.
    One channel corresponds to one image layer.
    One label image corresponds to one label layer.
    """
    n_image_layers: int = 0
    n_label_layers: int = 0
    img = image.images[0]

    if "c" in img.axes:
        ch_axis = "".join(img.axes).find("c")
        n_channels = int(img.data.shape[ch_axis])
        n_image_layers += n_channels
    else:
        n_image_layers += 1

    if hasattr(image, "labels") and image.labels is not None:
        n_label_layers += len(image.labels.items())

    return {"image_layers": n_image_layers, "label_layers": n_label_layers}
