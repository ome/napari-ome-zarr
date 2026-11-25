"""Store connection form: URL/credentials entry, test, and caching."""

from __future__ import annotations

import zarr
from magicgui.widgets import Container, LineEdit, Password, PushButton
from napari.utils.notifications import show_error
from psygnal import Signal
from urllib.parse import urlsplit

from ..ome_zarr_reader import Multiscales, Plate


class ConnectionForm(Container):
    """Widget for entering a store URL and optional S3 credentials.

    Emits `opened` with the `zarr.Group` when "Open" succeeds. A successful
    connection is cached in memory and reused until the URL or keys change.
    """

    opened = Signal(object)  # zarr.Group
    status_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self._tested_group: zarr.Group | None = None
        self._tested_connection: tuple[str, str, str] | None = None

        self.url = LineEdit(label="URL/Path")
        self.access_key = LineEdit(label="Access key (optional)")
        self.secret_key = Password(label="Secret key (optional)")
        for field in (self.url, self.access_key, self.secret_key):
            field.changed.connect(self._clear_connection_cache)

        test_button = PushButton(text="Test connection")
        open_button = PushButton(text="Connect")
        test_button.clicked.connect(self._on_test_clicked)
        open_button.clicked.connect(self._on_open_clicked)

        self.append(self.url)
        self.append(self.access_key)
        self.append(self.secret_key)
        self.append(Container(widgets=[test_button, open_button], layout="horizontal"))

    def _clear_connection_cache(self) -> None:
        """Drop the in-memory connection after credentials or URL change."""
        self._tested_group = None
        self._tested_connection = None

    def _connection_values(self) -> tuple[str, str, str]:
        url = self.url.value.strip()
        access_key = self.access_key.value.strip()
        secret_key = self.secret_key.value
        if not url:
            raise ValueError("Enter a URL or path.")
        if bool(access_key) != bool(secret_key):
            raise ValueError("Provide both access and secret keys, or leave both empty.")
        return url, access_key, secret_key

    def connect(self) -> zarr.Group:
        """
        Return a tested group, or open and cache the current form values.
        """
        connection = self._connection_values()
        if self._tested_group is not None and connection == self._tested_connection:
            return self._tested_group

        # Parse the URL and extract the S3 bucket and key path.
        # S3 stores need to be passed as s3:// to zarr.open_group,
        # otherwise zarr will not recognize it as an S3 store.
        url, access_key, secret_key = connection
        u = urlsplit(url)
        bucket, _, key_path = u.path.strip("/").partition("/")
        if access_key:
            storage_options = {
                "key": access_key,
                "secret": secret_key,
                "client_kwargs": {"endpoint_url": f"{u.scheme}://{u.netloc}"},
            }
        else:
            # ponytail: no keys given -> assume a public bucket, add a login
            # prompt on AccessDenied if private-but-keyless turns out common
            storage_options = {
                "anon": True,
                "client_kwargs": {"endpoint_url": f"{u.scheme}://{u.netloc}"},
            }
        url = f"s3://{bucket}/{key_path}"
        self.status_changed.emit("Connecting...")
        group = zarr.open_group(
            url,
            mode="r",
            storage_options=storage_options,
            )
        self._tested_group = group
        self._tested_connection = connection
        return group

    def _on_test_clicked(self) -> None:
        """Test the entered store without loading its layers."""
        try:
            _ = self.connect()
            msg = f"Connection test succeeded for {self.url.value.strip()}"
        except Exception as e:
            msg = f"Connection test for {self.url.value.strip()} failed with {type(e).__name__}: {e}"
        self.status_changed.emit(msg)

    def _on_open_clicked(self) -> None:
        try:
            group = self.connect()
        except Exception as e:
            msg = f"Failed to open with {type(e).__name__}: {e}"
            self.status_changed.emit(msg)
            return
        self.opened.emit(group)
