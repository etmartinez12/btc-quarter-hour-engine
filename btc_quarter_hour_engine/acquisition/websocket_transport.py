from __future__ import annotations

from typing import Any

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import ClientConnection, connect as _sync_connect


class WebsocketsTransport:
    """Concrete :class:`WebSocketTransport` backed by the ``websockets`` sync client.

    This is the only transport implementation that talks to a real network
    socket; every other consumer in this codebase depends on the
    ``WebSocketTransport`` protocol so tests can inject fakes instead. Kept
    deliberately thin: connection lifecycle, byte/text decoding and timeout
    translation only, no protocol semantics.
    """

    def __init__(self) -> None:
        self._connection: ClientConnection | None = None

    @property
    def connected(self) -> bool:
        return self._connection is not None

    def connect(self, *, url: str, connect_timeout: float) -> None:
        self._connection = _sync_connect(url, open_timeout=connect_timeout)

    def send(self, payload: str) -> None:
        if self._connection is None:
            raise RuntimeError("Transport is not connected")
        self._connection.send(payload)

    def recv(self, timeout: float | None = None) -> str:
        if self._connection is None:
            raise RuntimeError("Transport is not connected")
        try:
            message = self._connection.recv(timeout=timeout)
        except TimeoutError as exc:
            raise TimeoutError(f"No websocket message received within {timeout}s") from exc
        except ConnectionClosed as exc:
            raise ConnectionError("WebSocket connection closed") from exc
        if isinstance(message, bytes):
            return message.decode("utf-8")
        return message

    def close(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            finally:
                self._connection = None


__all__ = ["WebsocketsTransport"]
