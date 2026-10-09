from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from btc_quarter_hour_engine.acquisition.coinbase_websocket import CoinbaseWebSocketClient
from btc_quarter_hour_engine.acquisition.config import CoinbaseWebSocketConfig
from btc_quarter_hour_engine.acquisition.websocket_service import (
    CoinbaseWebSocketService,
    ReconnectExhaustedError,
    _bbo_state_row,
)
from btc_quarter_hour_engine.market_data.order_book import Level2OrderBook, OrderBookState


class FakeTransport:
    def __init__(self, messages=None, *, fail_connect_times=0, fail_connect_calls=()):
        self.messages = list(messages or [])
        self.sent = []
        self.connected = False
        self.close_count = 0
        self.connect_count = 0
        self._fail_connect_times = fail_connect_times
        self._fail_connect_calls = set(fail_connect_calls)

    def connect(self, *, url, connect_timeout):
        self.connect_count += 1
        if (
            self.connect_count <= self._fail_connect_times
            or self.connect_count in self._fail_connect_calls
        ):
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


def _l2_message(event_type, seq, timestamp, updates):
    return json.dumps(
        {
            "channel": "l2_data",
            "client_id": "",
            "timestamp": timestamp,
            "sequence_num": seq,
            "events": [{"type": event_type, "product_id": "BTC-USD", "updates": updates}],
        }
    )


def _snapshot(seq, time, bid=100.0, ask=101.0):
    return _l2_message(
        "snapshot",
        seq,
        time,
        [
            {"side": "bid", "price_level": str(bid), "new_quantity": "1.0", "event_time": time},
            {"side": "offer", "price_level": str(ask), "new_quantity": "1.0", "event_time": time},
        ],
    )


def _update(seq, time, side, price, quantity):
    return _l2_message(
        "update", seq, time, [{"side": side, "price_level": str(price), "new_quantity": str(quantity), "event_time": time}]
    )


def _heartbeat(seq, time, *, counter=None):
    return json.dumps(
        {
            "channel": "heartbeats",
            "client_id": "",
            "timestamp": time,
            "sequence_num": seq,
            "events": [{"current_time": time, "heartbeat_counter": str(seq if counter is None else counter)}],
        }
    )


def _subscriptions(seq, time="2026-10-02T17:20:32.027067134Z"):
    return json.dumps(
        {
            "channel": "subscriptions",
            "timestamp": time,
            "sequence_num": seq,
            "events": [{"subscriptions": {"level2": ["BTC-USD"]}}],
        }
    )


def _fake_monotonic(start: float = 0.0, step: float = 1.0):
    """An incrementing fake monotonic clock: every call advances by ``step``.

    Using a *constant* clock (e.g. ``lambda: 0.0``) would make any bounded
    wait loop keyed off ``monotonic_fn`` (such as
    :meth:`CoinbaseWebSocketService._wait_for_snapshot`) never observe its
    deadline elapsing, spinning forever when no matching message ever
    arrives. Every test in this module that can reach such a loop must use
    this incrementing clock instead.
    """
    counter = {"value": start}

    def _tick() -> float:
        counter["value"] += step
        return counter["value"]

    return _tick


def _service(messages, *, fail_connect_times=0, fail_connect_calls=(), **config_kwargs):
    config = CoinbaseWebSocketConfig(receive_timeout_seconds=0.01, **config_kwargs)
    transport = FakeTransport(
        messages,
        fail_connect_times=fail_connect_times,
        fail_connect_calls=fail_connect_calls,
    )
    client = CoinbaseWebSocketClient(config=config, transport=transport)
    service = CoinbaseWebSocketService(
        config=config, client=client, sleep_fn=lambda *_: None, monotonic_fn=_fake_monotonic()
    )
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
    assert {"bids", "asks"} <= result["book"].keys()
    assert service.order_book.is_synced()
    result = service.handle_message(_update(2, "2024-01-01T00:00:01Z", "bid", 100.5, 2.0))
    assert result["status"] == "l2_update_applied"
    assert {"bids", "asks"} <= result["book"].keys()
    assert service.order_book.best_bid == 100.5


def test_handle_message_fast_mode_skips_snapshots_for_snapshot_update_and_stale(monkeypatch):
    service, _ = _service([])
    service.connect_and_subscribe()

    def fail_snapshot():
        raise AssertionError("full book snapshot must not be materialized")

    monkeypatch.setattr(service.order_book, "snapshot", fail_snapshot)
    result = service.handle_message(_snapshot(1, "2024-01-01T00:00:00Z"), include_book_snapshot=False)
    assert result == {"status": "snapshot_synced"}
    result = service.handle_message(
        _update(2, "2024-01-01T00:00:01Z", "bid", 100.5, 2.0),
        include_book_snapshot=False,
    )
    assert result == {"status": "l2_update_applied"}
    result = service.handle_message(
        _update(2, "2024-01-01T00:00:02Z", "bid", 999.0, 2.0),
        include_book_snapshot=False,
    )
    assert result == {"status": "stale_sequence_ignored"}


def test_bbo_state_row_uses_equivalent_top_of_book_values():
    book = Level2OrderBook(product_id="BTC-USD")
    book.apply_snapshot(
        product_id="BTC-USD",
        levels={
            "bid": [{"price": 100.0, "quantity": 1.0}, {"price": 102.0, "quantity": 3.0}],
            "ask": [{"price": 105.0, "quantity": 4.0}, {"price": 103.0, "quantity": 5.0}],
        },
    )
    book.last_sequence_num = 42
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)

    row = _bbo_state_row(
        book=book,
        connection_id="connection-1",
        source_time_utc=timestamp,
        session_id="session-1",
        frame_index=7,
        ingest_time_utc=timestamp,
    )

    assert row["best_bid"] == 102.0
    assert row["best_bid_size"] == 3.0
    assert row["best_ask"] == 103.0
    assert row["best_ask_size"] == 5.0
    assert row["spread"] == 1.0
    assert row["midpoint"] == 102.5
    assert row["book_synced"] is True
    assert row["state"] == "SYNCED"
    assert row["sequence_num"] == 42


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


def test_handle_message_stale_or_duplicate_sequence_is_non_fatal():
    """A redelivered/duplicate sequence number must be dropped without
    raising, without invalidating the book, and without counting as a gap --
    this is the key stale-vs-gap disposition distinction."""
    service, _ = _service([])
    service.connect_and_subscribe()
    service.handle_message(_snapshot(1, "2024-01-01T00:00:00Z"))
    result = service.handle_message(_update(1, "2024-01-01T00:00:01Z", "bid", 999.0, 2.0))
    assert result["status"] == "stale_sequence_ignored"
    assert service.order_book.is_synced()
    assert service.order_book.best_bid == 100.0  # the stale update's payload must not have been applied
    assert service.connection.stale_sequence_count == 1
    assert service.connection.sequence_gap_count == 0
    assert service.diagnostics == []


def test_interleaved_envelopes_advance_connection_sequence_without_mutating_book():
    service, _ = _service([])
    service.connect_and_subscribe()
    service.handle_message(_snapshot(0, "2026-10-02T17:20:30Z"))
    service.handle_message(_update(1, "2026-10-02T17:20:31Z", "bid", 100.5, 2.0))
    before_subscription = service.order_book.snapshot()
    service.handle_message(_subscriptions(2, "2026-10-02T17:20:32Z"))
    assert service.order_book.snapshot() == before_subscription
    service.handle_message(_update(3, "2026-10-02T17:20:33Z", "offer", 101.5, 3.0))
    service.handle_message(_heartbeat(4, "2026-10-02T17:20:34Z"))
    service.handle_message(_update(5, "2026-10-02T17:20:35Z", "bid", 100.75, 4.0))
    service.handle_message(_subscriptions(6, "2026-10-02T17:20:36Z"))

    assert service.connection.sequence_gap_count == 0
    assert service.connection.stale_sequence_count == 0
    assert service.connection.last_envelope_sequence_num == 6
    assert service.connection.last_sequence_num == 5
    assert service.connection.level2_messages_received == 4
    assert service.connection.heartbeat_messages_received == 1
    assert service.order_book.is_synced()
    assert service.order_book.best_bid == 100.75
    assert service.order_book.best_ask == 101.0
    assert 101.5 in service.order_book.asks
    assert len(service.level2_update_rows) == 5
    assert len(service.bbo_state_rows) == 4


def test_true_gap_across_channels_invalidates_book():
    service, _ = _service([])
    service.connect_and_subscribe()
    service.handle_message(_snapshot(10, "2026-10-02T17:20:30Z"))
    service.handle_message(_heartbeat(11, "2026-10-02T17:20:31Z"))
    with pytest.raises(ValueError, match="expected 12, received 13"):
        service.handle_message(_update(13, "2026-10-02T17:20:32Z", "bid", 100.5, 2.0))
    assert service.connection.sequence_gap_count == 1
    assert not service.order_book.is_synced()


def test_stale_control_envelope_is_ignored_without_invalidating_book():
    service, _ = _service([])
    service.connect_and_subscribe()
    service.handle_message(_snapshot(10, "2026-10-02T17:20:30Z"))
    service.handle_message(_heartbeat(11, "2026-10-02T17:20:31Z"))
    before = service.order_book.snapshot()
    result = service.handle_message(_subscriptions(10, "2026-10-02T17:20:32Z"))
    assert result["status"] == "stale_sequence_ignored"
    assert service.connection.stale_sequence_count == 1
    assert service.connection.sequence_gap_count == 0
    assert service.order_book.snapshot() == before
    service.handle_message(_update(12, "2026-10-02T17:20:33Z", "bid", 100.5, 2.0))
    assert service.order_book.is_synced()
    assert service.connection.last_envelope_sequence_num == 12


def test_unknown_valid_channel_advances_connection_sequence_only():
    service, _ = _service([])
    service.connect_and_subscribe()
    service.handle_message(_snapshot(10, "2026-10-02T17:20:30Z"))
    before = service.order_book.snapshot()
    result = service.handle_message(json.dumps({
        "channel": "future_control_channel",
        "timestamp": "2026-10-02T17:20:31Z",
        "sequence_num": 11,
        "events": [{"status": "ok"}],
    }))
    assert result["status"] == "ignored"
    assert service.order_book.snapshot() == before
    service.handle_message(_update(12, "2026-10-02T17:20:32Z", "bid", 100.5, 2.0))
    assert service.connection.sequence_gap_count == 0
    assert service.connection.last_envelope_sequence_num == 12


def test_connection_sequence_baseline_resets_on_reconnect():
    service, _ = _service([])
    first = service.new_connection()
    for seq in (100, 101, 102):
        service.handle_message(_subscriptions(seq, f"2026-10-02T17:20:{seq - 70:02d}Z"))
    assert first.last_envelope_sequence_num == 102

    second = service.new_connection()
    for seq in (0, 1, 2):
        service.handle_message(_subscriptions(seq, f"2026-10-02T17:21:{seq:02d}Z"))
    assert second.last_envelope_sequence_num == 2
    assert second.sequence_gap_count == 0
    assert service.connection_history == [first]


def test_handle_message_heartbeat_updates_health_tracking():
    service, _ = _service([], heartbeat_timeout_seconds=1000)
    assert service.heartbeat_is_healthy()
    result = service.handle_message(_heartbeat(3, "2024-01-01T00:00:00Z"))
    assert result == {"status": "heartbeat", "heartbeat_counter": 3}
    assert service.connection is None  # heartbeat handling does not require an active connection record
    assert service.heartbeat_is_healthy()


def test_heartbeat_counter_discontinuity_is_recorded_without_invalidating_book():
    service, _ = _service([])
    service.connect_and_subscribe()
    service.handle_message(_snapshot(1, "2024-01-01T00:00:00Z"))
    service.handle_message(_heartbeat(2, "2024-01-01T00:00:01Z", counter=1))
    service.handle_message(_heartbeat(3, "2024-01-01T00:00:02Z", counter=3))
    assert service.connection.heartbeat_discontinuity_count == 1
    assert service.order_book.is_synced()


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
    service.handle_message(_heartbeat(1, "2024-01-01T00:14:55Z"))
    service.handle_message(_update(2, "2024-01-01T00:15:01Z", "bid", 100.5, 2.0))
    assert len(service.observations) == 1
    drained = service.drain_observations()
    assert len(drained) == 1
    assert service.observations == []


def test_level2_update_rows_and_bbo_state_rows_accumulate_and_drain():
    service, _ = _service([])
    service.handle_message(_snapshot(1, "2024-01-01T00:00:00Z"))
    service.handle_message(_update(2, "2024-01-01T00:00:01Z", "bid", 100.5, 2.0))
    # Snapshot contributes 2 rows (bid + ask levels), the update contributes 1.
    assert len(service.level2_update_rows) == 3
    assert {row["side"] for row in service.level2_update_rows} == {"bid", "ask"}
    assert len(service.bbo_state_rows) == 2  # one per successfully applied message
    assert service.bbo_state_rows[-1]["best_bid"] == 100.5

    drained_updates = service.drain_level2_update_rows()
    drained_state = service.drain_bbo_state_rows()
    assert len(drained_updates) == 3
    assert len(drained_state) == 2
    assert service.level2_update_rows == []
    assert service.bbo_state_rows == []

def test_multi_update_l2_envelope_is_applied_atomically_and_records_one_bbo():
    service, _ = _service([])
    service.connect_and_subscribe()
    service.handle_message(_snapshot(0, "2024-01-01T00:00:00Z"))
    initial_update_count = len(service.level2_update_rows)
    initial_bbo_count = len(service.bbo_state_rows)
    message = _l2_message(
        "update",
        1,
        "2024-01-01T00:00:01Z",
        [
            {"side": "bid", "price_level": "102.0", "new_quantity": "1.0"},
            {"side": "offer", "price_level": "101.0", "new_quantity": "0"},
            {"side": "offer", "price_level": "103.0", "new_quantity": "1.0"},
        ],
    )

    result = service.handle_message(message)

    assert result["status"] == "l2_update_applied"
    assert service.order_book.is_synced()
    assert service.order_book.best_bid == 102.0
    assert service.order_book.best_ask == 103.0
    assert service.connection.crossed_book_count == 0
    assert service.connection.malformed_level2_count == 0
    assert service.connection.last_sequence_num == 1
    assert len(service.level2_update_rows) - initial_update_count == 3
    assert len(service.bbo_state_rows) - initial_bbo_count == 1
    assert service.bbo_state_rows[-1]["best_bid"] == 102.0
    assert service.bbo_state_rows[-1]["best_ask"] == 103.0


def test_live_service_accepts_same_connection_clear_heartbeat_and_refill():
    service, transport = _service([])
    service.connect_and_subscribe()
    service.handle_message(_snapshot(1, "2024-01-01T00:14:50Z"))
    service.handle_message(_heartbeat(2, "2024-01-01T00:14:55Z"))
    clear = _l2_message(
        "update",
        3,
        "2024-01-01T00:14:59Z",
        [
            {"side": "bid", "price_level": "100.0", "new_quantity": "0"},
            {"side": "offer", "price_level": "101.0", "new_quantity": "0"},
        ],
    )

    result = service.handle_message(clear)

    assert result["status"] == "l2_rebuild_in_progress"
    assert service.order_book.state == OrderBookState.REBUILDING
    assert not service.order_book.is_synced()
    assert service.connection.invalidated_at_utc is None
    assert service.connection.malformed_level2_count == 0
    assert service.connection.crossed_book_count == 0
    assert service.connection.book_rebuild_count == 1
    assert service.connection.last_sequence_num == 3
    assert service.bbo_state_rows[-1]["state"] == "REBUILDING"
    assert service.bbo_state_rows[-1]["book_synced"] is False
    assert service.bbo_state_rows[-1]["best_bid"] is None
    assert service.bbo_state_rows[-1]["best_ask"] is None

    before_heartbeat = dict(service.order_book.bids), dict(service.order_book.asks)
    service.handle_message(_heartbeat(4, "2024-01-01T00:15:00Z"))
    assert (service.order_book.bids, service.order_book.asks) == before_heartbeat
    assert service.order_book.state == OrderBookState.REBUILDING
    assert service.connection.last_envelope_sequence_num == 4

    result = service.handle_message(_l2_message(
        "update",
        5,
        "2024-01-01T00:15:01Z",
        [
            {"side": "bid", "price_level": "99.0", "new_quantity": "2.0"},
            {"side": "offer", "price_level": "100.0", "new_quantity": "3.0"},
        ],
    ))

    assert result["status"] == "l2_rebuild_complete"
    assert service.order_book.state == OrderBookState.SYNCED
    assert (service.order_book.best_bid, service.order_book.best_ask) == (99.0, 100.0)
    assert service.connection.invalidated_at_utc is None
    assert service.connection.last_sequence_num == 5
    assert service.connection.last_envelope_sequence_num == 5
    assert service.connection.book_rebuild_count == 1
    assert service.connection.book_recovery_count == 1
    assert transport.connect_count == 1
    assert [event["from_state"] for event in service.diagnostics] == ["SYNCED", "REBUILDING"]
    assert [event["to_state"] for event in service.diagnostics] == ["REBUILDING", "SYNCED"]
    assert len(service.level2_update_rows) == 6
    assert [row["sequence_num"] for row in service.level2_update_rows[-4:]] == [3, 3, 5, 5]
    assert service.bbo_state_rows[-1]["state"] == "SYNCED"
    assert service.bbo_state_rows[-1]["sequence_num"] == 5


def test_stale_duplicate_during_live_rebuild_is_ignored_without_losing_recovery():
    service, _ = _service([])
    service.connect_and_subscribe()
    service.handle_message(_snapshot(1, "2024-01-01T00:00:00Z"))
    service.handle_message(_l2_message("update", 2, "2024-01-01T00:00:01Z", [
        {"side": "offer", "price_level": "101.0", "new_quantity": "0"},
    ]))

    stale = service.handle_message(_l2_message("update", 2, "2024-01-01T00:00:02Z", [
        {"side": "offer", "price_level": "101.0", "new_quantity": "1.0"},
    ]))
    assert stale["status"] == "stale_sequence_ignored"
    assert service.order_book.state == OrderBookState.REBUILDING
    assert service.connection.book_rebuild_count == 1
    assert service.connection.stale_sequence_count == 1

    recovered = service.handle_message(_l2_message("update", 3, "2024-01-01T00:00:03Z", [
        {"side": "offer", "price_level": "100.5", "new_quantity": "1.0"},
    ]))
    assert recovered["status"] == "l2_rebuild_complete"
    assert service.order_book.is_synced()
    assert service.connection.book_recovery_count == 1


def test_live_service_sequence_gap_during_rebuild_invalidates_the_connection():
    service, _ = _service([])
    service.connect_and_subscribe()
    service.handle_message(_snapshot(1, "2024-01-01T00:14:50Z"))
    service.handle_message(_heartbeat(2, "2024-01-01T00:14:55Z"))
    service.handle_message(_l2_message(
        "update",
        3,
        "2024-01-01T00:14:59Z",
        [{"side": "offer", "price_level": "101.0", "new_quantity": "0"}],
    ))
    service.handle_message(_heartbeat(4, "2024-01-01T00:14:59.500Z"))

    with pytest.raises(ValueError, match="expected 5, received 6"):
        service.handle_message(_l2_message(
            "update",
            6,
            "2024-01-01T00:15:01Z",
            [{"side": "offer", "price_level": "100.0", "new_quantity": "1"}],
        ))

    assert service.order_book.state == OrderBookState.INVALID
    assert service.connection.invalidated_at_utc is not None
    assert service.connection.sequence_gap_count == 1
    assert service.connection.book_rebuild_count == 1
    assert service.connection.book_recovery_count == 0
    assert service.order_book.last_sequence_num == 3


def test_level2_update_rows_and_bbo_state_rows_not_recorded_for_stale_or_gap():
    service, _ = _service([])
    service.connect_and_subscribe()
    service.handle_message(_snapshot(1, "2024-01-01T00:00:00Z"))
    service.drain_level2_update_rows()
    service.drain_bbo_state_rows()

    # Stale: dropped, no new rows.
    service.handle_message(_update(1, "2024-01-01T00:00:01Z", "bid", 999.0, 2.0))
    assert service.level2_update_rows == []
    assert service.bbo_state_rows == []

    # Gap: invalidates, no new rows either.
    with pytest.raises(ValueError):
        service.handle_message(_update(5, "2024-01-01T00:00:02Z", "bid", 999.0, 2.0))
    assert service.level2_update_rows == []
    assert service.bbo_state_rows == []


def test_injectable_now_fn_controls_recorded_wall_clock_timestamps():
    fixed_time = datetime(2030, 1, 1, tzinfo=timezone.utc)
    service, _ = _service([])
    service.now_fn = lambda: fixed_time
    connection = service.connect_and_subscribe()
    assert connection.connected_at_utc == fixed_time
    assert connection.subscribed_at_utc == fixed_time


def test_new_connection_appends_prior_connection_to_history():
    service, _ = _service([])
    first = service.connect_and_subscribe()
    assert service.connection_history == []
    second = service.new_connection()
    assert service.connection_history == [first]
    assert service.connection is second
    assert first is not second


def test_reconnect_appends_every_superseded_connection_to_history():
    messages = [_snapshot(1, "2024-01-01T00:00:00Z"), _snapshot(1, "2024-01-01T00:00:05Z")]
    service, transport = _service(messages, max_reconnect_attempts=3, initial_reconnect_backoff_seconds=0)
    first = service.connect_and_subscribe()
    service.handle_message(messages[0])
    second = service.reconnect()
    assert first in service.connection_history
    assert second not in service.connection_history
    assert service.connection is second


def test_reconnect_restarts_sequence_epoch_and_resyncs():
    messages = [
        _snapshot(1, "2024-01-01T00:00:00Z"),
        # A brand-new connection's first snapshot uses a fresh, low sequence
        # number; this must not look like a gap relative to the old epoch.
        _snapshot(3, "2024-01-01T00:00:05Z"),
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


def test_reconnect_retries_connection_establishment_failures_then_syncs():
    service, transport = _service(
        [_snapshot(1, "2024-01-01T00:00:05Z")],
        fail_connect_times=2,
        max_reconnect_attempts=3,
        initial_reconnect_backoff_seconds=0,
    )

    connection = service.reconnect()

    assert transport.connect_count == 3
    assert connection is service.connection
    assert connection.disconnect_reason is None
    assert service.order_book.is_synced()
    assert service.order_book.last_sequence_num == 1
    assert [
        item.disconnect_reason
        for item in service.connection_history[-2:]
    ] == ["connection_establishment_failed", "connection_establishment_failed"]


def test_reconnect_exhausts_exactly_after_connection_establishment_attempt_limit():
    service, transport = _service(
        [],
        fail_connect_times=5,
        max_reconnect_attempts=3,
        initial_reconnect_backoff_seconds=0,
    )

    with pytest.raises(ReconnectExhaustedError):
        service.reconnect()
    assert transport.connect_count == 3
    assert service.connection.disconnect_reason == "connection_establishment_failed"


def test_reconnect_stop_callback_interrupts_connect_failure_retries():
    from btc_quarter_hour_engine.acquisition.websocket_service import ReconnectStopRequested

    service, transport = _service(
        [],
        fail_connect_times=10,
        max_reconnect_attempts=None,
        initial_reconnect_backoff_seconds=0,
    )
    checks = {"count": 0}

    def stop_after_repeated_failures():
        checks["count"] += 1
        return transport.connect_count >= 3 and checks["count"] >= 5

    with pytest.raises(ReconnectStopRequested):
        service.reconnect(stop_fn=stop_after_repeated_failures)

    assert transport.connect_count == 3


def test_reconnect_wait_for_snapshot_drains_non_snapshot_messages_first():
    """A real reconnect does not necessarily receive a snapshot as its very
    first message: heartbeats or other messages may arrive first. The wait
    loop must keep receiving until a snapshot actually lands.

    The initial sync is fed directly via ``handle_message`` (independent of
    the transport's message queue) so the queue below -- which the
    reconnect's ``_wait_for_snapshot`` drains from -- only ever contains the
    post-reconnect heartbeats and the eventual resync snapshot.
    """
    initial_sync_message = _snapshot(1, "2024-01-01T00:00:00Z")
    queued_messages = [
        _heartbeat(1, "2024-01-01T00:00:01Z"),
        _heartbeat(2, "2024-01-01T00:00:02Z"),
        _snapshot(3, "2024-01-01T00:00:05Z"),
    ]
    service, transport = _service(
        queued_messages, max_reconnect_attempts=3, initial_reconnect_backoff_seconds=0, snapshot_wait_timeout_seconds=1000.0
    )
    service.connect_and_subscribe()
    service.handle_message(initial_sync_message)
    assert service.order_book.is_synced()

    connection = service.reconnect()
    assert connection is not None
    assert service.order_book.is_synced()
    # Every buffered message must have been drained (2 heartbeats + the resync snapshot).
    assert transport.messages == []


def test_reconnect_snapshot_wait_does_not_materialize_full_book(monkeypatch):
    resync_snapshot = _snapshot(3, "2024-01-01T00:00:05Z")
    service, _ = _service(
        [resync_snapshot], max_reconnect_attempts=2, initial_reconnect_backoff_seconds=0,
        snapshot_wait_timeout_seconds=1000.0,
    )
    service.connect_and_subscribe()
    service.handle_message(_snapshot(1, "2024-01-01T00:00:00Z"))

    def fail_snapshot():
        raise AssertionError("reconnect snapshot wait must not materialize full book")

    monkeypatch.setattr(service.order_book, "snapshot", fail_snapshot)
    connection = service.reconnect()

    assert connection is not None
    assert service.order_book.is_synced()


def test_reconnect_raises_after_exhausting_attempts():
    # `snapshot_wait_timeout_seconds=0` keeps `_wait_for_snapshot` a no-op so
    # this stays fast/deterministic even though the fake clock advances.
    service, _ = _service(
        [], max_reconnect_attempts=2, initial_reconnect_backoff_seconds=0, snapshot_wait_timeout_seconds=0.0
    )
    with pytest.raises(RuntimeError, match="exhausted"):
        service.reconnect()


def test_reconnect_gives_up_waiting_once_deadline_elapses_without_a_snapshot():
    """Even with a bounded wait deadline, `_wait_for_snapshot` must not spin
    forever receiving non-snapshot messages -- it must give up once its
    monotonic deadline elapses and let the outer reconnect loop retry (and
    eventually raise once attempts are exhausted)."""
    messages = [_heartbeat(1, "2024-01-01T00:00:01Z"), _heartbeat(2, "2024-01-01T00:00:02Z")]
    service, _ = _service(
        messages, max_reconnect_attempts=1, initial_reconnect_backoff_seconds=0, snapshot_wait_timeout_seconds=1.5
    )
    with pytest.raises(RuntimeError, match="exhausted"):
        service.reconnect()


def test_reconnect_stop_callback_interrupts_snapshot_wait():
    from btc_quarter_hour_engine.acquisition.websocket_service import ReconnectStopRequested

    service, _ = _service(
        [], max_reconnect_attempts=None, initial_reconnect_backoff_seconds=0,
        snapshot_wait_timeout_seconds=1000.0,
    )
    checks = {"count": 0}

    def stop_after_connect():
        checks["count"] += 1
        return checks["count"] >= 3

    with pytest.raises(ReconnectStopRequested):
        service.reconnect(stop_fn=stop_after_connect)


def test_mark_invalid_records_reason_and_invalidates_book():
    service, _ = _service([])
    service.connect_and_subscribe()
    service.handle_message(_snapshot(1, "2024-01-01T00:00:00Z"))
    service.mark_invalid("simulated failure")
    assert service.order_book.state == OrderBookState.INVALID
    assert service.connection.disconnect_reason == "simulated failure"
    assert service.connection.invalidated_at_utc is not None
