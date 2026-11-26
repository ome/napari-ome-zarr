"""Node-specific controls for loading OME-Zarr multiscales."""

from __future__ import annotations

from typing import Any

import zarr
from magicgui.widgets import ComboBox, Container, PushButton
from ome_zarr import OMEZarrMultiscale, OMEZarrScene
from psygnal import Signal

from ..ome_zarr_reader import Multiscales, Scene


class MultiscalesWidget(Container):
    status_changed = Signal(str)

    def __init__(self, viewer: Any, group: zarr.Group):
        super().__init__()
        self.native.setObjectName("multiscalesWidget")
        self.native.setStyleSheet(
            "#multiscalesWidget { border: 1px solid gray; border-radius: 4px; }"
        )
        self._viewer = viewer
        self._group = group
        self._managed_layers: list[Any] = []

        ms = OMEZarrMultiscale.from_ome_zarr(group)
        cs_names = [cs.name for cs in ms.metadata.coordinateSystems]

        self._load_button = PushButton(text="Load Multiscale image")
        self._load_button.clicked.connect(self.load)
        self.append(self._load_button)

        self._cs_dropdown = ComboBox(
            choices=cs_names,
            value=ms.metadata.intrinsic_coordinate_system.name,
            label="Coordinate System",
        )

        # we show the coordinate system dropdown only if there 
        # are multiple coordinate systems available
        self._cs_dropdown.visible = len(cs_names) > 1
        self._cs_dropdown.changed.connect(self._on_cs_changed)
        self.append(self._cs_dropdown)

    def load(self) -> None:
        self._load_button.enabled = False
        self._managed_layers.clear()
        self.status_changed.emit("Loading...")
        try:
            for data, props, layer_type in Multiscales(self._group).to_layer_data(
                target_coordinate_system=self._cs_dropdown.value
            ):
                layer = getattr(self._viewer, f"add_{layer_type}")(
                    data, **props
                )
                self._managed_layers.append(layer)
            self.status_changed.emit(
                f"Loaded {len(self._managed_layers)} layer(s)"
            )
        except Exception as error:
            self.status_changed.emit(
                f"Failed to load: {type(error).__name__}: {error}"
            )
        finally:
            self._load_button.enabled = True

    def _on_cs_changed(self, new_cs: str) -> None:
        if not self._managed_layers:
            return
        new_by_name = {
            props.get("name"): props
            for _, props, _ in Multiscales(self._group).to_layer_data(
                target_coordinate_system=new_cs
            )
        }
        for layer in self._managed_layers:
            props = new_by_name.get(layer.name)
            if props is not None and "affine" in props:
                layer.affine = props["affine"]

