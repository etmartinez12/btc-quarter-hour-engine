from __future__ import annotations

import json
from datetime import datetime, timezone

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


def _heartbeat(seq, time):
    return json.dumps(
        {
            "channel": "heartbeats",
            "client_id": "",
            "timestamp": time,
            "sequence_num": seq,
            "events": [{"current_time": time, "heartbeat_counter": str(seq)}],
        }
    )


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
        _heartbeat(2, "2024-01-01T00:14:55Z"),
        _update(3, "2024-01-01T00:15:01Z", "bid", 100.5, 2.0),
    ]
    collector, service, transport, raw_writer = _build_collector(tmp_path, messages)
    result = collector.run(max_messages=len(messages))

    assert result.observation_count == 1
    assert result.eligible_observation_count == 1
    assert result.connection_count == 1
    assert result.reconnect_count == 0
    assert result.heartbeat_message_count == 1

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
    assert manifest["coverage"]["level2_update_row_count"] == result.level2_update_row_count
    assert manifest["coverage"]["bbo_state_row_count"] == result.bbo_state_row_count
    assert manifest["acquisition_started_at_utc"] is not None
    assert manifest["acquisition_completed_at_utc"] is not None
    assert result.manifest_path.exists()
    assert json.loads(result.manifest_path.read_text())["dataset_id"] == manifest["dataset_id"]

    # Normalized parquet output: quarter-hour BBO, plus normalized
    # level2_updates (one row per individual book mutation across the
    # snapshot + update messages) and bbo_state (one row per successfully
    # applied mutation) artifacts.
    assert len(result.bbo_normalized_artifacts) == 1
    assert len(result.level2_update_artifacts) == 1
    assert len(result.bbo_state_artifacts) == 1
    assert len(result.normalized_artifacts) == 3
    frame = pd.read_parquet(result.bbo_normalized_artifacts[0]["path"])
    assert list(frame["canonical_target_eligible"]) == [True]
    assert list(frame["eligibility_reason"]) == ["eligible"]
    assert frame["boundary_time_utc"].iloc[0] == pd.Timestamp("2024-01-01T00:15:00Z")
    assert frame["best_bid"].iloc[0] == 100.0
    assert frame["best_ask"].iloc[0] == 101.0
    assert frame["product_id"].iloc[0] == "BTC-USD"

    level2_frame = pd.read_parquet(result.level2_update_artifacts[0]["path"])
    assert len(level2_frame) == 3  # 2 snapshot levels + 1 incremental update
    assert set(level2_frame["side"]) == {"bid", "ask"}

    bbo_state_frame = pd.read_parquet(result.bbo_state_artifacts[0]["path"])
    assert len(bbo_state_frame) == 2  # one per successfully applied message (snapshot, update)
    assert bbo_state_frame["state"].iloc[-1] == "SYNCED"

    # Raw segment: exact bytes are sealed and independently readable/verifiable.
    segment = result.raw_segments[0]
    assert raw_writer.raw_store.verify_digest(segment["path"], segment["sha256"])
    frames = raw_writer.read_segment_frames(segment["path"])
    assert [f.decode("utf-8") for f in frames] == messages


def test_collector_keeps_transient_cross_multi_update_envelope_synced(tmp_path):
    update_time = "2024-01-01T00:14:59.900Z"
    messages = [
        _snapshot(1, "2024-01-01T00:14:50Z"),
        _heartbeat(2, "2024-01-01T00:14:55Z"),
        _l2_message(
            "update",
            3,
            update_time,
            [
                {"side": "bid", "price_level": "102.0", "new_quantity": "1.0", "event_time": update_time},
                {"side": "offer", "price_level": "101.0", "new_quantity": "0", "event_time": update_time},
                {"side": "offer", "price_level": "103.0", "new_quantity": "1.0", "event_time": update_time},
            ],
        ),
        _heartbeat(4, "2024-01-01T00:15:01Z"),
    ]
    collector, service, _, _ = _build_collector(tmp_path, messages)
    result = collector.run(max_messages=len(messages))

    assert result.connection_count == 1
    assert result.reconnect_count == 0
    assert result.manifest["integrity"]["sequence_gap_count"] == 0
    assert service.connection.crossed_book_count == 0
    assert service.connection.malformed_level2_count == 0
    assert service.order_book.is_synced()
    assert (service.order_book.best_bid, service.order_book.best_ask) == (102.0, 103.0)
    assert result.level2_update_row_count == 5
    assert result.bbo_state_row_count == 2

    state_frame = pd.read_parquet(result.bbo_state_artifacts[0]["path"])
    assert len(state_frame) == 2
    assert state_frame.iloc[-1]["best_bid"] == 102.0
    assert state_frame.iloc[-1]["best_ask"] == 103.0


def test_collector_with_no_eligible_observations_reports_ineligibility_reason(tmp_path):
    messages = [_heartbeat(1, "2024-01-01T00:00:00Z")]
    collector, *_ = _build_collector(tmp_path, messages)
    result = collector.run(max_messages=len(messages))
    assert result.observation_count == 1
    assert result.manifest["quarter_hour_summary"]["ineligible_boundaries"] == 1
    assert result.manifest["canonical_target_eligible"] is False
    assert result.manifest["canonical_target_eligible"] is False
    assert len(result.bbo_normalized_artifacts) == 1
    assert pd.read_parquet(result.bbo_normalized_artifacts[0]["path"])["eligibility_reason"].iloc[0] == "no_synced_snapshot"


def test_collector_counts_real_heartbeat_and_keeps_counters_distinct(tmp_path):
    heartbeat = json.dumps({
        "channel": "heartbeats",
        "timestamp": "2026-10-02T17:20:32.414830619Z",
        "sequence_num": 1,
        "events": [{
            "current_time": "2026-10-02 17:20:32.414052991 +0000 UTC m=+181244.321549317",
            "heartbeat_counter": 181244,
        }],
    })
    collector, _, _, _ = _build_collector(
        tmp_path,
        [_snapshot(0, "2026-10-02T17:20:30Z"), heartbeat],
    )
    result = collector.run(max_messages=2)
    assert result.heartbeat_message_count == 1
    assert result.manifest["connections"][0]["heartbeat_messages_received"] == 1
    assert result.manifest["connections"][0]["last_envelope_sequence_num"] == 1
    assert result.manifest["connections"][0]["last_sequence_num"] == 0


def test_collector_reconnects_after_a_true_envelope_sequence_gap(tmp_path):
    messages = [
        _snapshot(10, "2024-01-01T00:14:50Z"),
        _update(12, "2024-01-01T00:14:55Z", "bid", 100.5, 2.0),
        _snapshot(0, "2024-01-01T00:15:05Z"),
    ]
    collector, _, _, _ = _build_collector(
        tmp_path,
        messages,
        config_kwargs={"initial_reconnect_backoff_seconds": 0},
    )
    result = collector.run(max_messages=2)
    assert result.reconnect_count == 1
    assert result.connection_count == 2
    assert result.raw_frame_count == 3
    assert result.manifest["integrity"]["sequence_gap_count"] == 1


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


def test_reconnect_exhaustion_finalizes_controlled_session(tmp_path):
    clock = {"now": datetime(2024, 1, 1, 0, 14, 50, tzinfo=timezone.utc)}
    ack = json.dumps({
        "channel": "subscriptions",
        "client_id": "",
        "timestamp": "2024-01-01T00:14:56Z",
        "sequence_num": 1,
        "events": [],
    })

    class AttemptTransport(FakeTransport):
        def __init__(self):
            super().__init__([])
            self.per_connection = {
                1: [_snapshot(1, "2024-01-01T00:14:50Z"), _heartbeat(1, "2024-01-01T00:14:55Z")],
                2: [ack, _heartbeat(1, "2024-01-01T00:14:57Z")],
                3: [ack, _heartbeat(1, "2024-01-01T00:14:58Z")],
            }
            self.drop_sent = False

        def recv(self, timeout=None):
            current = self.connect_count
            if self.per_connection[current]:
                return self.per_connection[current].pop(0)
            if current == 1 and not self.drop_sent:
                self.drop_sent = True
                raise ConnectionError("simulated initial connection drop")
            if current == 3:
                clock["now"] = datetime(2024, 1, 1, 0, 15, 5, tzinfo=timezone.utc)
            raise TimeoutError("snapshot did not arrive")

    ticks = {"value": 0.0}

    def monotonic():
        ticks["value"] += 0.1
        return ticks["value"]

    config = CoinbaseWebSocketConfig(
        receive_timeout_seconds=0.01,
        snapshot_wait_timeout_seconds=2.0,
        max_reconnect_attempts=2,
        initial_reconnect_backoff_seconds=0,
        heartbeat_timeout_seconds=30,
    )
    transport = AttemptTransport()
    client = CoinbaseWebSocketClient(config=config, transport=transport, now_fn=lambda: clock["now"])
    service = CoinbaseWebSocketService(
        config=config,
        client=client,
        now_fn=lambda: clock["now"],
        monotonic_fn=monotonic,
        sleep_fn=lambda *_: None,
    )
    collector = WebSocketCollector(
        service=service,
        raw_segment_writer=RawSegmentWriter(tmp_path, product_id="BTC-USD", max_frames=100),
        forward_store=ForwardParquetStore(tmp_path),
        output_root=str(tmp_path),
    )

    result = collector.run()

    assert result.termination_reason == "reconnect_exhausted"
    assert result.connection_count == 3
    assert result.reconnect_count == 2
    assert transport.connect_count == 3
    assert transport.close_count >= 3
    assert result.raw_segments
    assert result.raw_frame_count == sum(segment["frame_count"] for segment in result.raw_segments)
    assert result.manifest_path.exists()
    persisted = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert persisted["termination_reason"] == "reconnect_exhausted"
    assert persisted["connection_count"] == 3
    assert persisted["reconnect_count"] == 2
    boundary_artifact = result.bbo_normalized_artifacts[0]
    boundary_frame = pd.read_parquet(boundary_artifact["path"])
    boundary = boundary_frame.loc[
        boundary_frame["boundary_time_utc"] == pd.Timestamp("2024-01-01T00:15:00Z")
    ].iloc[0]
    assert not bool(boundary["canonical_target_eligible"])
    assert boundary["eligibility_reason"] in {"connection_unhealthy", "heartbeat_stale", "no_synced_snapshot"}


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
        _heartbeat(2, "2024-01-01T00:14:55Z"),
        _update(3, "2024-01-01T00:15:01Z", "bid", 100.5, 2.0),
        _heartbeat(4, "2024-01-01T00:15:03Z"),
        _update(5, "2024-01-01T00:30:02Z", "ask", 101.5, 3.0),
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
    live_frame = pd.read_parquet(result.normalized_artifacts[0]["path"]).sort_values("boundary_time_utc")
    replayed_rows = [
        (o.timestamp_utc.isoformat(), o.best_bid, o.best_ask, o.eligible)
        for o in sorted(replayed, key=lambda o: o.timestamp_utc)
    ]
    live_rows = [
        (pd.Timestamp(ts).isoformat(), bid, ask, bool(eligible))
        for ts, bid, ask, eligible in zip(
            live_frame["boundary_time_utc"], live_frame["best_bid"], live_frame["best_ask"], live_frame["canonical_target_eligible"]
        )
    ]
    assert replayed_rows == live_rows


def test_final_canonical_replay_includes_late_arriving_pre_boundary_l2(tmp_path):
    messages = [
        _snapshot(1, "2024-01-01T13:14:50Z"),
        _heartbeat(2, "2024-01-01T13:14:55Z"),
        _update(3, "2024-01-01T13:15:00.100Z", "offer", 101.5, 1),
        _update(4, "2024-01-01T13:14:59.950Z", "bid", 100.5, 1),
    ]
    collector, service, _, _ = _build_collector(tmp_path, messages)
    result = collector.run(max_messages=len(messages))
    row = pd.read_parquet(result.bbo_normalized_artifacts[0]["path"]).iloc[0]
    assert row["best_bid"] == 100.5
    assert row["best_ask"] == 101.0
    assert row["midpoint"] == 100.75
    assert bool(row["canonical_target_eligible"])
    assert result.manifest["quarter_hour_summary"]["eligible_boundaries"] == 1
    assert result.manifest["session_id"] == service.session_id


def test_final_replay_uses_each_update_time_in_one_envelope(tmp_path):
    before, after = "2024-01-01T13:14:59.900Z", "2024-01-01T13:15:00.100Z"
    messages = [
        _snapshot(199, "2024-01-01T13:14:50Z"),
        _heartbeat(200, "2024-01-01T13:14:55Z"),
        _l2_message("update", 201, after, [
            {"side": "bid", "price_level": "100.5", "new_quantity": "1", "event_time": before},
            {"side": "offer", "price_level": "101.5", "new_quantity": "1", "event_time": after},
        ]),
    ]
    collector, _, _, _ = _build_collector(tmp_path, messages)
    result = collector.run(max_messages=len(messages))
    row = pd.read_parquet(result.bbo_normalized_artifacts[0]["path"]).iloc[0]
    assert (row["best_bid"], row["best_ask"]) == (100.5, 101)
    updates = pd.read_parquet(result.level2_update_artifacts[0]["path"])
    assert set(updates["event_time_utc"]) >= {pd.Timestamp(before), pd.Timestamp(after)}


def test_binary_transport_bytes_survive_final_raw_segment(tmp_path):
    payload = _snapshot(1, "2024-01-01T13:14:50Z").encode("utf-8")
    collector, _, _, writer = _build_collector(tmp_path, [payload])
    result = collector.run(max_messages=1)
    assert writer.read_segment_frames(result.raw_segments[0]["path"]) == [payload]
    assert result.raw_segments[0]["frames"][0]["raw_frame_sha256"]


def test_unexpected_failure_leaves_durable_partial_without_success_manifest(tmp_path):
    messages = [_snapshot(1, "2024-01-01T13:14:50Z")] * 3
    collector, _, transport, writer = _build_collector(tmp_path, messages)
    original_recv = transport.recv
    count = 0

    def fail_after_three(timeout=None):
        nonlocal count
        count += 1
        if count > 3:
            raise OSError("disk or transport failure")
        return original_recv(timeout)

    transport.recv = fail_after_three
    with pytest.raises(OSError, match="failure"):
        collector.run()
    assert writer.active_partial_path.exists()
    assert writer.active_partial_path.stat().st_size > 3 * 8
    assert writer.sealed_segments == []
    assert not list((tmp_path / "manifests").rglob("*.json")) if (tmp_path / "manifests").exists() else True


def test_keyboard_interrupt_finalizes_and_closes(tmp_path):
    collector, _, transport, _ = _build_collector(tmp_path, [_snapshot(1, "2024-01-01T13:14:50Z")])
    original_recv = transport.recv
    count = 0

    def interrupt(timeout=None):
        nonlocal count
        count += 1
        if count == 2:
            raise KeyboardInterrupt()
        return original_recv(timeout)

    transport.recv = interrupt
    result = collector.run()
    assert result.termination_reason == "keyboard_interrupt"
    assert result.manifest_path.exists()
    assert transport.close_count == 1


def test_reconnect_frames_are_recorded_with_new_epoch_and_row_provenance(tmp_path):
    messages = [
        _snapshot(10, "2024-01-01T13:14:50Z"),
        _heartbeat(11, "2024-01-01T13:14:55Z"),
        _snapshot(1, "2024-01-01T13:15:05Z"),
        _heartbeat(2, "2024-01-01T13:29:59Z"),
        _update(3, "2024-01-01T13:30:01Z", "bid", 100.5, 1),
    ]
    collector, service, transport, _ = _build_collector(
        tmp_path, messages,
        config_kwargs={"initial_reconnect_backoff_seconds": 0},
        transport_kwargs={"connection_errors_after": 3},
    )
    result = collector.run(stop_fn=lambda: transport.connect_count == 2 and not transport.messages)
    assert result.connection_count == 2
    assert result.raw_frame_count == 5
    assert len(result.raw_segments) == 2
    assert result.raw_segments[0]["connection_id"] != result.raw_segments[1]["connection_id"]
    assert result.raw_segments[1]["frames"][0]["frame_index"] == 2
    updates = pd.concat([pd.read_parquet(item["path"]) for item in result.level2_update_artifacts])
    assert updates["frame_index"].notna().all()
    assert set(updates["connection_id"]) == {c.connection_id for c in [*service.connection_history, service.connection]}
    boundaries = pd.concat([pd.read_parquet(item["path"]) for item in result.bbo_normalized_artifacts])
    assert boundaries.loc[boundaries["boundary_time_utc"] == pd.Timestamp("2024-01-01T13:15:00Z"), "eligibility_reason"].iloc[0] == "connection_unhealthy"
    assert bool(boundaries.loc[boundaries["boundary_time_utc"] == pd.Timestamp("2024-01-01T13:30:00Z"), "canonical_target_eligible"].iloc[0])


def test_malformed_frame_invalidates_canonical_epoch_and_counts_integrity(tmp_path):
    invalid = json.dumps({
        "channel": "l2_data", "timestamp": "2024-01-01T13:14:56Z", "sequence_num": 2,
        "events": [{"type": "update", "product_id": "BTC-USD", "updates": [
            {"side": "bid", "price_level": "inf", "new_quantity": "1", "event_time": "2024-01-01T13:14:56Z"}
        ]}],
    })
    messages = [_snapshot(1, "2024-01-01T13:14:50Z"), _heartbeat(2, "2024-01-01T13:14:55Z"), invalid,
                _update(2, "2024-01-01T13:15:01Z", "bid", 100.5, 1)]
    collector, _, _, _ = _build_collector(tmp_path, messages)
    result = collector.run(max_messages=len(messages))
    row = pd.read_parquet(result.bbo_normalized_artifacts[0]["path"]).iloc[0]
    assert row["eligibility_reason"] == "malformed_source_state"
    assert result.manifest["integrity"]["malformed_frame_count"] == 1
    assert result.manifest["integrity"]["malformed_level2_count"] == 1


def test_quiet_collection_retains_every_boundary_through_session_end(tmp_path):
    config = CoinbaseWebSocketConfig(heartbeat_timeout_seconds=30)
    clock = {"now": datetime(2024, 1, 1, 13, 14, 49, tzinfo=timezone.utc)}
    transport = FakeTransport([_snapshot(1, "2024-01-01T13:14:50Z")])
    client = CoinbaseWebSocketClient(config=config, transport=transport, now_fn=lambda: clock["now"])
    service = CoinbaseWebSocketService(config=config, client=client, now_fn=lambda: clock["now"],
                                       monotonic_fn=lambda: 0.0)
    collector = WebSocketCollector(
        service=service, raw_segment_writer=RawSegmentWriter(tmp_path),
        forward_store=ForwardParquetStore(tmp_path), output_root=str(tmp_path),
    )

    def stop_after_silence():
        if transport.messages:
            return False
        clock["now"] = datetime(2024, 1, 1, 13, 45, 1, tzinfo=timezone.utc)
        return True

    result = collector.run(stop_fn=stop_after_silence)
    frame = pd.read_parquet(result.bbo_normalized_artifacts[0]["path"])
    assert list(frame["boundary_time_utc"]) == [
        pd.Timestamp("2024-01-01T13:15:00Z"),
        pd.Timestamp("2024-01-01T13:30:00Z"),
        pd.Timestamp("2024-01-01T13:45:00Z"),
    ]
    assert set(frame["eligibility_reason"]) == {"heartbeat_stale"}
