"""Lazy-loading tree widget for browsing a zarr store's hierarchy."""

from __future__ import annotations

import zarr
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QTreeWidget, QTreeWidgetItem
from napari_ome_zarr.ome_zarr_reader import Multiscales

_LOADING = "..."  # placeholder child, swapped for real children on first expand


class ZarrTree(QTreeWidget):
    """
    QTreeWidget mirroring a zarr group's hierarchy.

    Sub-groups are populated on first expand rather than eagerly, since
    remote stores can be slow (or deep, e.g. multiscale pyramids) to list.
    """
    selected_node = Signal(object)

    def __init__(self):
        super().__init__()
        self.setHeaderHidden(True)
        self.itemExpanded.connect(self._on_expanded)
        self.itemSelectionChanged.connect(self._on_selection_changed)

    def set_group(self, group: zarr.Group) -> None:
        """Replace the tree's contents with the hierarchy rooted at `group`."""
        self.clear()
        self.addTopLevelItem(self._make_item(group.basename or "/", group))

    def _make_item(self, name: str, node: zarr.Group | zarr.Array) -> QTreeWidgetItem:
        icon = ""
        node_type = "Group"
        if isinstance(node, zarr.Group):
            if Multiscales.matches(node):
                icon = "🖼️"
                node_type = "Multiscales"
            item = QTreeWidgetItem([f"{icon} {name} ({node_type})"])
            item.setData(0, Qt.ItemDataRole.UserRole, node)
            item.addChild(QTreeWidgetItem([_LOADING]))  # shows expand arrow
        else:
            if isinstance(node, zarr.Array):
                icon = "🧊"
                item = QTreeWidgetItem([f"{icon} {name} ({node.shape}, {node.dtype})"])
                item.setData(0, Qt.ItemDataRole.UserRole, node)
            else:
                item = QTreeWidgetItem([f"{icon} {name}"])
        return item

    def _on_expanded(self, item: QTreeWidgetItem) -> None:
        if item.childCount() != 1 or item.child(0).text(0) != _LOADING:
            return  # already populated
        group = item.data(0, Qt.ItemDataRole.UserRole)
        item.takeChildren()
        for name, node in sorted(group.members(), key=lambda member: member[0].casefold()):
            item.addChild(self._make_item(name, node))

    def _on_selection_changed(self) -> None:
        items = self.selectedItems()
        item = items[0] if items else None
        node = (
            item.data(0, Qt.ItemDataRole.UserRole)
            if item is not None
            else None
        )
        self.selected_node.emit(node)
