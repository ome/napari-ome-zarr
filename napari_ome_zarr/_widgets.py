import magicgui


@magicgui.magic_factory(
    call_button="Import",
    ome_zarr_url={"label": "OME-ZARR URL"},
)
def import_from_url(ome_zarr_url: str) -> None:
    """Import an OME-Zarr from a URL."""
    from napari import current_viewer
    from napari.utils.notifications import show_warning

    from napari_ome_zarr._reader import napari_get_reader

    viewer = current_viewer()

    if viewer is None:
        return

    reader = napari_get_reader(ome_zarr_url)
    if reader is None:
        show_warning(f"No reader found for {ome_zarr_url}")
        return None

    layer_data = reader(ome_zarr_url)
    for layer in layer_data:
        if layer[2] == "image":
            viewer.add_image(layer[0], **layer[1])
        elif layer[2] == "labels":
            viewer.add_labels(layer[0], **layer[1])
        else:
            pass
