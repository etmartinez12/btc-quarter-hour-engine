from __future__ import annotations

import threading

import pytest
from websockets.sync.server import serve

from btc_quarter_hour_engine.acquisition.websocket_transport import WebsocketsTransport


def _echo_handler(connection):
    for message in connection:
        if message == "__close__":
            return
        connection.send(f"echo:{message}")


@pytest.fixture()
def loopback_server():
    server = serve(_echo_handler, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.socket.getsockname()[:2]
    try:
        yield f"ws://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_transport_connects_sends_and_receives_over_loopback(loopback_server):
    transport = WebsocketsTransport()
    transport.connect(url=loopback_server, connect_timeout=5.0)
    try:
        assert transport.connected
        transport.send("hello")
        assert transport.recv(timeout=5.0) == "echo:hello"
    finally:
        transport.close()
    assert not transport.connected


def test_transport_recv_raises_timeout_error_when_no_message(loopback_server):
    transport = WebsocketsTransport()
    transport.connect(url=loopback_server, connect_timeout=5.0)
    try:
        with pytest.raises(TimeoutError):
            transport.recv(timeout=0.2)
    finally:
        transport.close()


def test_transport_recv_raises_connection_error_after_server_closes(loopback_server):
    transport = WebsocketsTransport()
    transport.connect(url=loopback_server, connect_timeout=5.0)
    transport.send("__close__")
    with pytest.raises(ConnectionError):
        transport.recv(timeout=5.0)
    transport.close()


def test_transport_send_and_recv_before_connect_raise_runtime_error():
    transport = WebsocketsTransport()
    with pytest.raises(RuntimeError, match="not connected"):
        transport.send("x")
    with pytest.raises(RuntimeError, match="not connected"):
        transport.recv(timeout=0.1)


def test_transport_close_before_connect_is_a_noop():
    transport = WebsocketsTransport()
    transport.close()  # must not raise
    assert not transport.connected
