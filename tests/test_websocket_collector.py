from __future__ import annotations

import json

import pandas as pd
import pytest

from btc_quarter_hour_engine.acquisition.coinbase_websocket import CoinbaseWebSocketClient, parse_coinbase_ws_message
from btc_quarter_hour_engine.acquisition.collector import WebSocketCollector
from btc_quarter_hour_engine.acquisition.config import CoinbaseWebSocketConfig
from btc_quarter_hour_engine.acquisition.websocket_service import CoinbaseWebSocketService
from btc_quarter_hour_engine.market_data.replay import replay_events
from btc_quarter_hour_engine.storage.forward_parquet import ForwardParquetStore
from btc_quarter_hour_engine.storage.websocket_raw import RawSegmentWriter


class FakeTransport:
    def __init__(self, messages, *, connection_errors_after=None):
        self.messages = list(messages)
        self.sent = []
        self.connect_count = 0
        self.close_count = 0
        self._connection_errors_after = connection_errors_after
        self._recv_count = 0

    def connect(self, *, url, connect_timeout):
        self.connect_count += 1

    def send(self, payload):
        self.sent.append(payload)

    def recv(self, timeout=None):
        self._recv_count += 1
        if self._connection_errors_after is not None and self._recv_count == self._connection_errors_after:
            raise ConnectionError("simulated drop")
        if not self.messages:
            raise TimeoutError("no more messages")
        return self.messages.pop(0)

    def close(self):
        self.close_count += 1


def _msg(message_type, **kw):
    payload = {"type": message_type}
    payload.update(kw)
    return json.dumps(payload)


def _snapshot(seq, time, bid=100.0, ask=101.0):
    return _msg(
        "snapshot", product_id="BTC-USD", sequence_num=seq, time=time,
        bids=[{"price": bid, "quantity": 1.0}], asks=[{"price": ask, "quantity": 1.0}],
    )


def _update(seq, time, side, price, quantity):
    return _msg(
        "l2_data", product_id="BTC-USD", sequence_num=seq, time=time,
        changes=[{"side": side, "price": price, "new_quantity": quantity}],
    )


def _heartbeat(seq, time):
    return _msg("heartbeat", sequence=seq, time=time)


def _build_collector(tmp_path, messages, *, heartbeat_timeout_seconds=30.0, config_kwargs=None, transport_kwargs=None):
    config = CoinbaseWebSocketConfig(
        receive_timeout_seconds=0.01,
        heartbeat_timeout_seconds=heartbeat_timeout_seconds,
        **(config_kwargs or {}),
    )
    transport = FakeTransport(messages, **(transport_kwargs or {}))
    client = CoinbaseWebSocketClient(config=config, transport=transport)
    service = CoinbaseWebSocketService(config=config, client=client, sleep_fn=lambda *_: None, monotonic_fn=lambda: 0.0)
    raw_writer = RawSegmentWriter(tmp_path, product_id="BTC-USD", max_frames=100)
    forward_store = ForwardParquetStore(tmp_path)
    collector = WebSocketCollector(
        service=service, raw_segment_writer=raw_writer, forward_store=forward_store, output_root=str(tmp_path)
    )
    return collector, service, transport, raw_writer


def test_collector_produces_eligible_observation_and_complete_manifest(tmp_path):
    messages = [
        _snapshot(1, "2024-01-01T00:14:50Z"),
        _heartbeat(1, "2024-01-01T00:14:55Z"),
        _update(2, "2024-01-01T00:15:01Z", "bid", 100.5, 2.0),
    ]
    collector, service, transport, raw_writer = _build_collector(tmp_path, messages)
    result = collector.run(max_messages=len(messages))

    assert result.observation_count == 1
    assert result.eligible_observation_count == 1
    assert result.connection_count == 1
    assert result.reconnect_count == 0

    # Manifest completeness.
    manifest = result.manifest
    assert manifest["dataset_id"]
    assert manifest["source"] == "coinbase_advanced"
    assert manifest["product_id"] == "BTC-USD"
    assert manifest["data_kind"] == "quarter_hour_bbo"
    assert manifest["canonical_target_eligible"] is True
    assert manifest["canonical_target_ineligibility_reason"] is None
    assert manifest["raw_artifacts"] == result.raw_segments
    assert len(manifest["raw_artifacts"]) == 1
    assert manifest["raw_artifacts"][0]["frame_count"] == len(messages)
    assert manifest["normalized_artifacts"] == result.normalized_artifacts
    assert manifest["coverage"]["observation_count"] == 1
    assert manifest["coverage"]["eligible_observation_count"] == 1
    assert manifest["acquisition_started_at_utc"] is not None
    assert manifest["acquisition_completed_at_utc"] is not None
    assert result.manifest_path.exists()
    assert json.loads(result.manifest_path.read_text())["dataset_id"] == manifest["dataset_id"]

    # Normalized parquet output.
    assert len(result.normalized_artifacts) == 1
    frame = pd.read_parquet(result.normalized_artifacts[0]["path"])
    assert list(frame["eligible"]) == [True]
    assert frame["best_bid"].iloc[0] == 100.0
    assert frame["best_ask"].iloc[0] == 101.0
    assert frame["product_id"].iloc[0] == "BTC-USD"

    # Raw segment: exact bytes are sealed and independently readable/verifiable.
    segment = result.raw_segments[0]
    assert raw_writer.raw_store.verify_digest(segment["path"], segment["sha256"])
    frames = raw_writer.read_segment_frames(segment["path"])
    assert [f.decode("utf-8") for f in frames] == messages


def test_collector_with_no_eligible_observations_reports_ineligibility_reason(tmp_path):
    messages = [_heartbeat(1, "2024-01-01T00:00:00Z")]
    collector, *_ = _build_collector(tmp_path, messages)
    result = collector.run(max_messages=len(messages))
    assert result.observation_count == 0
    assert result.manifest["canonical_target_eligible"] is False
    assert "no eligible" in result.manifest["canonical_target_ineligibility_reason"]
    assert result.normalized_artifacts == []


def test_collector_reconnects_on_connection_error_and_continues(tmp_path):
    messages_after_reconnect = [_snapshot(1, "2024-01-01T00:00:00Z")]
    # A generous heartbeat timeout keeps the post-resync heartbeat check
    # healthy for the duration of this test, so the *wall-clock* bound below
    # is the only thing that ends the run (not a second, message-starved
    # reconnect attempt, which would spin forever since
    # `max_reconnect_attempts=None` here).
    config = CoinbaseWebSocketConfig(
        receive_timeout_seconds=0.01,
        max_reconnect_attempts=None,
        initial_reconnect_backoff_seconds=0,
        heartbeat_timeout_seconds=1_000.0,
    )
    transport = FakeTransport(messages_after_reconnect, connection_errors_after=1)
    client = CoinbaseWebSocketClient(config=config, transport=transport)
    counter = {"value": 0.0}

    def fake_monotonic():
        counter["value"] += 1.0
        return counter["value"]

    service = CoinbaseWebSocketService(config=config, client=client, sleep_fn=lambda *_: None, monotonic_fn=fake_monotonic)
    raw_writer = RawSegmentWriter(tmp_path, product_id="BTC-USD", max_frames=100)
    forward_store = ForwardParquetStore(tmp_path)
    collector = WebSocketCollector(
        service=service, raw_segment_writer=raw_writer, forward_store=forward_store, output_root=str(tmp_path)
    )
    # Bounded by wall-clock (fake monotonic advances every tick) so the loop
    # cannot spin forever once the post-reconnect message queue is drained.
    result = collector.run(max_duration_seconds=5.0)
    assert result.reconnect_count == 1
    assert result.connection_count == 2
    assert transport.close_count >= 1


def test_collector_stops_on_max_duration(tmp_path):
    ticks = iter([0.0, 0.0, 100.0, 100.0])

    def fake_monotonic():
        try:
            return next(ticks)
        except StopIteration:
            return 100.0

    config = CoinbaseWebSocketConfig(receive_timeout_seconds=0.01)
    transport = FakeTransport([])
    client = CoinbaseWebSocketClient(config=config, transport=transport)
    service = CoinbaseWebSocketService(config=config, client=client, sleep_fn=lambda *_: None, monotonic_fn=fake_monotonic)
    raw_writer = RawSegmentWriter(tmp_path, product_id="BTC-USD", max_frames=100)
    forward_store = ForwardParquetStore(tmp_path)
    collector = WebSocketCollector(service=service, raw_segment_writer=raw_writer, forward_store=forward_store, output_root=str(tmp_path))
    result = collector.run(max_duration_seconds=1.0)
    assert result.observation_count == 0


def test_collector_raw_segments_replay_to_same_observations_as_live(tmp_path):
    messages = [
        _snapshot(1, "2024-01-01T00:14:50Z"),
        _heartbeat(1, "2024-01-01T00:14:55Z"),
        _update(2, "2024-01-01T00:15:01Z", "bid", 100.5, 2.0),
        _heartbeat(2, "2024-01-01T00:15:03Z"),
        _update(3, "2024-01-01T00:30:02Z", "ask", 101.5, 3.0),
    ]
    collector, service, transport, raw_writer = _build_collector(tmp_path, messages)
    result = collector.run(max_messages=len(messages))

    segment = result.raw_segments[0]
    frames = raw_writer.read_segment_frames(segment["path"])
    events = [parse_coinbase_ws_message(frame) for frame in frames]
    replayed = replay_events(events, product_id="BTC-USD", heartbeat_timeout_seconds=30.0)

    assert len(replayed) == result.observation_count
    # Replay must be internally deterministic (sorted by event time)...
    assert [(o.timestamp_utc, o.eligible, o.best_bid, o.best_ask) for o in replayed] == [
        (o.timestamp_utc, o.eligible, o.best_bid, o.best_ask)
        for o in sorted(replayed, key=lambda o: o.timestamp_utc)
    ]
    # ...and must exactly match what the live collector actually persisted,
    # proving live processing and offline replay of the sealed raw segment
    # are byte-for-byte equivalent in their derived output.
    live_frame = pd.read_parquet(result.normalized_artifacts[0]["path"]).sort_values("source_time_utc")
    replayed_rows = [
        (o.timestamp_utc.isoformat(), o.best_bid, o.best_ask, o.eligible)
        for o in sorted(replayed, key=lambda o: o.timestamp_utc)
    ]
    live_rows = [
        (pd.Timestamp(ts).isoformat(), bid, ask, bool(eligible))
        for ts, bid, ask, eligible in zip(
            live_frame["source_time_utc"], live_frame["best_bid"], live_frame["best_ask"], live_frame["eligible"]
        )
    ]
    assert replayed_rows == live_rows
