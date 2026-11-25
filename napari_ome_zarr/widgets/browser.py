"""OME-Zarr navigation widget for napari."""

from __future__ import annotations

import napari
import zarr
from magicgui.widgets import Container, Label

from ..ome_zarr_reader import Multiscales
from .connect import ConnectionForm
from .multiscales import MultiscalesWidget
from .tree import ZarrTree


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
        self._node_widget: Container | None = None

        self._connection_form = ConnectionForm()
        self._connection_form.opened.connect(self._on_opened)
        self._connection_form.status_changed.connect(self._on_status)
        self.append(self._connection_form)

        self._tree = ZarrTree()
        self._tree.selected_node.connect(self._on_selected_node)
        self.native.layout().addWidget(self._tree)

        # Status label
        self._status = Label(value="")
        self._status.native.setWordWrap(True)
        self.append(self._status)

    def _on_status(self, message: str) -> None:
        """
        Triggered if any status message is emitted by child widgets.
        """
        self._status.value = message

    def _on_opened(self, group: zarr.Group) -> None:
        """Detect Spec type for a newly-opened store and load its layers."""
        self._group = group
        self._tree.set_group(group)
        self._show_node_widget(None)
        self._on_status("Store opened")

    def _show_node_widget(self, widget: Container | None) -> None:
        """
        Utility function to show a new node-specific widget, replacing any existing one.
        """
        if self._node_widget is not None:
            old_native = self._node_widget.native
            self.native.layout().removeWidget(old_native)
            old_native.setParent(None)
            old_native.deleteLater()
        self._node_widget = widget
        if widget is not None:
            status_index = self.native.layout().indexOf(self._status.native)
            self.native.layout().insertWidget(status_index, widget.native)
            widget.status_changed.connect(self._on_status)

    def _on_selected_node(self, node: zarr.Group | zarr.Array | None) -> None:
        """
        Callback for when a node in the Zarr tree is selected.
        """
        if isinstance(node, zarr.Group) and Multiscales.matches(node):
            self._show_node_widget(MultiscalesWidget(self._viewer, node))
        elif isinstance(node, zarr.Array):
            self._show_node_widget(None)
        else:
            self._show_node_widget(None)
