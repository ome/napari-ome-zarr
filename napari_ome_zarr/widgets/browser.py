"""OME-Zarr navigation widget for napari."""

from __future__ import annotations

from typing import Any

import napari
import zarr
from magicgui import magicgui
from magicgui.widgets import ComboBox, Container, Label
from napari.utils.notifications import show_error
from ome_zarr import OMEZarrMultiscale

from ..ome_zarr_reader import Multiscales, Plate


class OMEZarrBrowser(Container):
    """
    Container widget for OME-Zarr navigation.

    Shows URL entry, detects Spec type, and provides type-specific controls.
    Multiscales: coordinate system dropdown with live affine updates.
    """

    def __init__(self):
        super().__init__()
        self._viewer = napari.current_viewer()
        self._group: zarr.Group | None = None
        # ponytail: track layers we added, so we can update affines without reloading
        self._managed_layers: list[Any] = []

        # URL entry widget
        @magicgui(call_button="Open", url={"label": "URL/Path"})
        def open_store(url: str = "") -> None:
            self._open(url)

        self._open_widget = open_store
        self.append(self._open_widget)

        # Coordinate system controls (hidden until a multi-CS Multiscales is detected)
        self._cs_label = Label(value="Coordinate System:")
        self._cs_dropdown = ComboBox(choices=[], label="")
        self._cs_dropdown.changed.connect(self._on_cs_changed)
        self._cs_container = Container(widgets=[self._cs_label, self._cs_dropdown])
        self._cs_container.visible = False
        self.append(self._cs_container)

        # Status label
        self._status = Label(value="")
        self.append(self._status)

    def _open(self, url: str) -> None:
        """Open zarr store, detect type, show appropriate controls."""
        if not url.strip():
            return

        # Clear previous state
        self._group = None
        self._managed_layers.clear()
        self._cs_container.visible = False
        self._status.value = "Opening..."

        try:
            self._group = zarr.open_group(url, mode="r")
        except Exception as e:
            show_error(f"Failed to open: {e}")
            self._status.value = "Failed"
            return

        # Detect Spec type
        if Plate.matches(self._group):
            # ponytail: Plate nav needs Qt grid widget, stub for now
            self._status.value = "Plate detected (navigation not yet implemented)"
            self._load_simple(Plate)
        elif Multiscales.matches(self._group):
            self._setup_multiscales()
        else:
            self._status.value = "Unknown OME-Zarr type"

    def _setup_multiscales(self) -> None:
        """Load a Multiscales image and, if it has multiple coordinate systems, show the dropdown."""
        assert self._group is not None
        ms = OMEZarrMultiscale.from_ome_zarr(self._group)
        cs_names = [cs.name for cs in ms.metadata.coordinateSystems]

        self._load_simple(Multiscales)

        if len(cs_names) > 1:
            default_cs = ms.metadata.intrinsic_coordinate_system.name
            self._cs_dropdown.choices = cs_names
            self._cs_dropdown.value = default_cs
            self._cs_container.visible = True

        self._status.value = f"Multiscales loaded ({len(cs_names)} coordinate systems)"

    def _on_cs_changed(self, new_cs: str) -> None:
        """Update layer affines when coordinate system changes."""
        # ponytail: zarr.Group.__len__ is nmembers(), which is 0 for stores that can't
        # list (e.g. plain HTTP) - `not self._group` is truthy there even when open.
        if self._group is None or not self._managed_layers:
            return

        new_layer_data = Multiscales(self._group).to_layer_data(
            target_coordinate_system=new_cs
        )

        # Match by name and update affines
        # ponytail: O(n²) name matching, fine for typical layer counts (<100 layers)
        new_by_name = {props.get("name"): props for _, props, _ in new_layer_data}
        for layer in self._managed_layers:
            if layer.name in new_by_name and "affine" in new_by_name[layer.name]:
                layer.affine = new_by_name[layer.name]["affine"]

    def _load_simple(self, spec_cls) -> None:
        """Load a Multiscales or Plate spec and add its layers to the viewer."""
        assert self._group is not None
        spec = spec_cls(self._group)
        for data, props, layer_type in spec.to_layer_data():
            add_method = getattr(self._viewer, f"add_{layer_type}")
            layer = add_method(data, **props)
            self._managed_layers.append(layer)
