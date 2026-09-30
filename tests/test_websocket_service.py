from __future__ import annotations

import json

import pytest

from btc_quarter_hour_engine.acquisition.coinbase_websocket import CoinbaseWebSocketClient
from btc_quarter_hour_engine.acquisition.config import CoinbaseWebSocketConfig
from btc_quarter_hour_engine.acquisition.websocket_service import CoinbaseWebSocketService


class FakeTransport:
    def __init__(self, messages=None, *, fail_connect_times=0):
        self.messages = list(messages or [])
        self.sent = []
        self.connected = False
        self.close_count = 0
        self.connect_count = 0
        self._fail_connect_times = fail_connect_times

    def connect(self, *, url, connect_timeout):
        self.connect_count += 1
        if self.connect_count <= self._fail_connect_times:
            raise ConnectionError("simulated connect failure")
        self.connected = True

    def send(self, payload):
        self.sent.append(payload)

    def recv(self, timeout=None):
        if not self.messages:
            raise TimeoutError("no more messages")
        return self.messages.pop(0)

    def close(self):
        self.connected = False
        self.close_count += 1


def _msg(message_type, **kw):
    payload = {"type": message_type}
    payload.update(kw)
    return json.dumps(payload)


def _snapshot(seq, time, bid=100.0, ask=101.0):
    return _msg(
        "snapshot",
        product_id="BTC-USD",
        sequence_num=seq,
        time=time,
        bids=[{"price": bid, "quantity": 1.0}],
        asks=[{"price": ask, "quantity": 1.0}],
    )


def _update(seq, time, side, price, quantity):
    return _msg(
        "l2_data",
        product_id="BTC-USD",
        sequence_num=seq,
        time=time,
        changes=[{"side": side, "price": price, "new_quantity": quantity}],
    )


def _service(messages, **config_kwargs):
    config = CoinbaseWebSocketConfig(receive_timeout_seconds=0.01, **config_kwargs)
    transport = FakeTransport(messages)
    client = CoinbaseWebSocketClient(config=config, transport=transport)
    service = CoinbaseWebSocketService(config=config, client=client, sleep_fn=lambda *_: None, monotonic_fn=lambda: 0.0)
    return service, transport


def test_connect_and_subscribe_sends_level2_and_heartbeats_subscriptions():
    service, transport = _service([])
    service.connect_and_subscribe()
    assert transport.connected
    payloads = [json.loads(p) for p in transport.sent]
    assert payloads[0] == {"type": "subscribe", "product_ids": ["BTC-USD"], "channel": "level2"}
    assert payloads[1] == {"type": "subscribe", "channel": "heartbeats"}


def test_handle_message_snapshot_then_update_produces_synced_book():
    service, _ = _service([])
    result = service.handle_message(_snapshot(1, "2024-01-01T00:00:00Z"))
    assert result["status"] == "snapshot_synced"
    assert service.order_book.is_synced()
    result = service.handle_message(_update(2, "2024-01-01T00:00:01Z", "bid", 100.5, 2.0))
    assert result["status"] == "l2_update_applied"
    assert service.order_book.best_bid == 100.5


def test_handle_message_raises_and_invalidates_on_sequence_gap():
    service, _ = _service([])
    service.connect_and_subscribe()
    service.handle_message(_snapshot(1, "2024-01-01T00:00:00Z"))
    with pytest.raises(ValueError):
        service.handle_message(_update(5, "2024-01-01T00:00:01Z", "bid", 100.5, 2.0))
    assert not service.order_book.is_synced()
    assert service.connection.sequence_gap_count == 1
    assert len(service.diagnostics) == 1
    assert service.diagnostics[0]["kind"] == "sequence_error"


def test_handle_message_heartbeat_updates_health_tracking():
    service, _ = _service([], heartbeat_timeout_seconds=1000)
    assert service.heartbeat_is_healthy()
    result = service.handle_message(_msg("heartbeat", sequence=3, time="2024-01-01T00:00:00Z"))
    assert result == {"status": "heartbeat", "heartbeat_counter": 3}
    assert service.connection is None  # heartbeat handling does not require an active connection record
    assert service.heartbeat_is_healthy()


def test_heartbeat_is_healthy_reports_false_after_timeout():
    service, _ = _service([], heartbeat_timeout_seconds=5)
    tick = {"value": 0.0}
    service.monotonic_fn = lambda: tick["value"]
    service.last_heartbeat_monotonic = service.monotonic_fn()
    tick["value"] = 10.0
    assert not service.heartbeat_is_healthy()


def test_derived_observations_accumulate_and_drain():
    service, _ = _service([])
    service.handle_message(_snapshot(1, "2024-01-01T00:14:50Z"))
    service.handle_message(_msg("heartbeat", sequence=1, time="2024-01-01T00:14:55Z"))
    service.handle_message(_update(2, "2024-01-01T00:15:01Z", "bid", 100.5, 2.0))
    assert len(service.observations) == 1
    drained = service.drain_observations()
    assert len(drained) == 1
    assert service.observations == []


def test_reconnect_restarts_sequence_epoch_and_resyncs():
    messages = [
        _snapshot(1, "2024-01-01T00:00:00Z"),
        # A brand-new connection's first snapshot uses a fresh, low sequence
        # number; this must not look like a gap relative to the old epoch.
        _snapshot(1, "2024-01-01T00:00:05Z"),
    ]
    service, transport = _service(messages, max_reconnect_attempts=3, initial_reconnect_backoff_seconds=0)
    service.connect_and_subscribe()
    service.handle_message(messages[0])
    assert service.order_book.is_synced()

    connection = service.reconnect()
    assert connection is not None
    assert service.order_book.is_synced()
    assert transport.close_count >= 1
    assert transport.connect_count >= 2


def test_reconnect_raises_after_exhausting_attempts():
    service, _ = _service([], max_reconnect_attempts=2, initial_reconnect_backoff_seconds=0)
    with pytest.raises(RuntimeError, match="exhausted"):
        service.reconnect()
