from __future__ import annotations

import json
import math
from copy import deepcopy
from datetime import datetime, timezone

import pytest

from btc_quarter_hour_engine.acquisition.coinbase_websocket import parse_coinbase_ws_message
from btc_quarter_hour_engine.market_data.order_book import Level2OrderBook, OrderBookState
from btc_quarter_hour_engine.market_data.replay import CanonicalReplayAccumulator
from btc_quarter_hour_engine.research.features import (
    _BucketStats,
    _IngestFeatureReplay,
)


def _utc(timestamp: str) -> datetime:
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def _frame(channel: str, timestamp: str, sequence: int, *, updates=None, event_type="update"):
    if channel == "l2_data":
        events = [{
            "type": event_type,
            "product_id": "BTC-USD",
            "updates": [
                {
                    "side": side,
                    "event_time": event_time,
                    "price_level": str(price),
                    "new_quantity": str(quantity),
                }
                for side, event_time, price, quantity in updates
            ],
        }]
    else:
        events = [{"current_time": timestamp, "heartbeat_counter": "1"}]
    return json.dumps({
        "channel": channel,
        "timestamp": timestamp,
        "sequence_num": sequence,
        "events": events,
    }).encode()


def _snapshot(timestamp="2026-01-01T00:14:50Z", sequence=1):
    return _frame(
        "l2_data",
        timestamp,
        sequence,
        event_type="snapshot",
        updates=[
            ("bid", timestamp, 100.0, 1.0),
            ("offer", timestamp, 101.0, 2.0),
            ("bid", timestamp, 99.5, 3.0),
            ("offer", timestamp, 101.5, 4.0),
        ],
    )


def _update(sequence, timestamp, side, price, quantity, wire_side=None):
    return _frame(
        "l2_data",
        timestamp,
        sequence,
        updates=[(wire_side or side, timestamp, price, quantity)],
    )


def _stream():
    return [
        _snapshot(),
        _frame("heartbeats", "2026-01-01T00:14:55Z", 2),
        _update(3, "2026-01-01T00:14:59Z", "bid", 100.0, 1.5),
        _frame("heartbeats", "2026-01-01T00:15:01Z", 4),
        _update(5, "2026-01-01T00:15:05Z", "offer", 101.0, 2.5),
        _frame("heartbeats", "2026-01-01T00:29:59Z", 6),
        _update(7, "2026-01-01T00:30:01Z", "bid", 100.5, 2.0),
        _frame("heartbeats", "2026-01-01T00:44:59Z", 8),
        _update(9, "2026-01-01T00:45:01Z", "offer", 101.0, 0.0),
    ]


def _feature_replay(frames):
    replay = _IngestFeatureReplay("BTC-USD")
    for index, raw in enumerate(frames):
        event = parse_coinbase_ws_message(raw)
        ingest = event.get("time_utc") or event["event_time_utc"]
        replay.consume(
            raw=raw,
            connection_id="conn-1",
            frame_index=index,
            ingest_time_utc=ingest,
        )
    return replay


def _book_checkpoint_round_trip(book):
    encoded = json.dumps(book.to_checkpoint(), allow_nan=False)
    return Level2OrderBook.from_checkpoint(json.loads(encoded))


def test_order_book_checkpoint_round_trip_and_continuation():
    original = Level2OrderBook(product_id="BTC-USD")
    original.apply_snapshot(
        product_id="BTC-USD",
        levels={
            "bid": [
                {"price": 100.0, "quantity": 1.0},
                {"price": 99.0, "quantity": 3.0},
            ],
            "ask": [
                {"price": 101.0, "quantity": 2.0},
                {"price": 102.0, "quantity": 4.0},
            ],
        },
    )
    original.last_sequence_num = 12
    restored = _book_checkpoint_round_trip(original)

    def observe(book):
        return (
            dict(book.bids), dict(book.asks), book.best_bid, book.best_ask,
            book.best_bid_size, book.best_ask_size, book.midpoint, book.spread,
            book.state, book.last_sequence_num, book.last_error,
        )

    assert observe(restored) == observe(original)
    original.apply_update(side="bid", price=100.5, quantity=5.0)
    restored.apply_update(side="bid", price=100.5, quantity=5.0)
    original.apply_update(side="ask", price=101.0, quantity=0.0)
    restored.apply_update(side="ask", price=101.0, quantity=0.0)
    assert observe(restored) == observe(original)


def test_feature_replay_checkpoint_continuation_is_exact():
    frames = _stream()
    uninterrupted = _feature_replay(frames)
    interrupted = _feature_replay(frames[:5])
    payload = json.dumps(interrupted.to_checkpoint(), allow_nan=False)
    resumed = _IngestFeatureReplay.from_checkpoint(json.loads(payload))
    for index, raw in enumerate(frames[5:], start=5):
        event = parse_coinbase_ws_message(raw)
        resumed.consume(
            raw=raw,
            connection_id="conn-1",
            frame_index=index,
            ingest_time_utc=event.get("time_utc") or event["event_time_utc"],
        )
    assert resumed.to_checkpoint() == uninterrupted.to_checkpoint()
    assert resumed.book.snapshot() == uninterrupted.book.snapshot()
    assert resumed.buckets.keys() == uninterrupted.buckets.keys()
    for timestamp in resumed.buckets:
        left, right = resumed.buckets[timestamp], uninterrupted.buckets[timestamp]
        for name in _BucketStats.__dataclass_fields__:
            a, b = getattr(left, name), getattr(right, name)
            assert (math.isnan(a) and math.isnan(b)) or a == b


def test_feature_nonfinite_bucket_values_are_json_safe_and_round_trip():
    replay = _IngestFeatureReplay("BTC-USD")
    replay.consume(
        raw=b"not-json",
        connection_id="conn-1",
        frame_index=0,
        ingest_time_utc=_utc("2026-01-01T00:01:00Z"),
    )
    checkpoint = replay.to_checkpoint()
    encoded = json.dumps(checkpoint, allow_nan=False)
    decoded = json.loads(encoded)
    restored = _IngestFeatureReplay.from_checkpoint(decoded)
    bucket = next(iter(restored.buckets.values()))
    assert math.isnan(bucket.asof_midpoint)
    assert bucket.spread_bps_max == -math.inf


def _canonical_replay(frames, connection_ids):
    replay = CanonicalReplayAccumulator(
        product_id="BTC-USD",
        session_id="session-1",
        heartbeat_timeout_seconds=30.0,
    )
    for index, (raw, connection_id) in enumerate(zip(frames, connection_ids)):
        event = parse_coinbase_ws_message(raw)
        ingest = event.get("time_utc") or event["event_time_utc"]
        replay.consume(raw, connection_id, index, ingest)
    return replay


def test_canonical_checkpoint_continuation_matches_observations_exactly():
    frames = _stream()
    connection_ids = ["conn-1"] * 5 + ["conn-2"] * 4
    # A new connection epoch has its own sequence and snapshot.
    frames[5:] = [
        _snapshot("2026-01-01T00:29:59Z", 1),
        _frame("heartbeats", "2026-01-01T00:30:00Z", 2),
        _update(3, "2026-01-01T00:30:01Z", "bid", 100.5, 2.0),
        _frame("heartbeats", "2026-01-01T00:44:59Z", 4),
        _update(5, "2026-01-01T00:45:01Z", "offer", 101.0, 0.0),
    ]
    connection_ids = ["conn-1"] * 5 + ["conn-2"] * 5
    uninterrupted = _canonical_replay(frames, connection_ids)
    interrupted = _canonical_replay(frames[:6], connection_ids[:6])
    checkpoint = interrupted.to_checkpoint()
    encoded = json.dumps(checkpoint, allow_nan=False)
    resumed = CanonicalReplayAccumulator.from_checkpoint(json.loads(encoded))
    for index, (raw, connection_id) in enumerate(
        zip(frames[6:], connection_ids[6:]), start=6,
    ):
        event = parse_coinbase_ws_message(raw)
        resumed.consume(
            raw,
            connection_id,
            index,
            event.get("time_utc") or event["event_time_utc"],
        )
    start = _utc("2026-01-01T00:00:00Z")
    end = _utc("2026-01-01T01:00:00Z")
    derived = _utc("2026-01-01T02:00:00Z")
    original_rows = uninterrupted.finalize(
        session_started_at_utc=start,
        session_completed_at_utc=end,
        derived_at_utc=derived,
    )
    resumed_rows = resumed.finalize(
        session_started_at_utc=start,
        session_completed_at_utc=end,
        derived_at_utc=derived,
    )
    assert resumed_rows == original_rows
    assert resumed.to_checkpoint() == uninterrupted.to_checkpoint()
    assert resumed._connections["conn-1"].book is None
    assert resumed._connections["conn-1"].terminal_state is not None
    assert resumed._connections["conn-2"].heartbeat_buckets
    assert resumed._connections["conn-2"].boundary_candidates


@pytest.mark.parametrize(
    ("restore", "checkpoint"),
    [
        (
            Level2OrderBook.from_checkpoint,
            {"version": 99, "product_id": "BTC-USD", "bids": [], "asks": [],
             "state": "UNINITIALIZED", "last_sequence_num": None, "last_error": None},
        ),
        (
            _IngestFeatureReplay.from_checkpoint,
            {"version": 99},
        ),
        (
            CanonicalReplayAccumulator.from_checkpoint,
            {"version": 99},
        ),
    ],
)
def test_checkpoint_rejects_unsupported_versions(restore, checkpoint):
    with pytest.raises(ValueError, match="Unsupported checkpoint schema version"):
        restore(checkpoint)


def test_checkpoint_rejects_missing_keys_and_malformed_order_book_values():
    book_checkpoint = Level2OrderBook(product_id="BTC-USD").to_checkpoint()
    missing = deepcopy(book_checkpoint)
    del missing["asks"]
    with pytest.raises(ValueError, match="Malformed checkpoint keys"):
        Level2OrderBook.from_checkpoint(missing)

    invalid_state = deepcopy(book_checkpoint)
    invalid_state["state"] = "BROKEN"
    with pytest.raises(ValueError, match="Invalid checkpoint order-book state"):
        Level2OrderBook.from_checkpoint(invalid_state)

    malformed_level = deepcopy(book_checkpoint)
    malformed_level["bids"] = [{"price": 100.0}]
    with pytest.raises(ValueError, match="Malformed checkpoint bid level"):
        Level2OrderBook.from_checkpoint(malformed_level)

    invalid_price = deepcopy(book_checkpoint)
    invalid_price["bids"] = [{"price": 0, "quantity": 1}]
    with pytest.raises(ValueError, match="Invalid checkpoint bid price"):
        Level2OrderBook.from_checkpoint(invalid_price)


def test_feature_checkpoint_rejects_bad_timestamps_and_nested_product_mismatch():
    replay = _feature_replay(_stream()[:1])
    checkpoint = replay.to_checkpoint()
    checkpoint["buckets"][0]["bucket_time_utc"] = "not-a-time"
    with pytest.raises(ValueError, match="Malformed checkpoint timestamp"):
        _IngestFeatureReplay.from_checkpoint(checkpoint)

    checkpoint = replay.to_checkpoint()
    checkpoint["buckets"][0]["bucket_time_utc"] = "2026-01-01T00:15:00"
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        _IngestFeatureReplay.from_checkpoint(checkpoint)

    checkpoint = replay.to_checkpoint()
    checkpoint["book"]["product_id"] = "ETH-USD"
    with pytest.raises(ValueError, match="product_id does not match"):
        _IngestFeatureReplay.from_checkpoint(checkpoint)

    checkpoint = replay.to_checkpoint()
    del checkpoint["frames_seen"]
    with pytest.raises(ValueError, match="Malformed checkpoint keys"):
        _IngestFeatureReplay.from_checkpoint(checkpoint)


def test_canonical_checkpoint_rejects_malformed_nested_state():
    replay = _canonical_replay(_stream()[:2], ["conn-1", "conn-1"])
    checkpoint = replay.to_checkpoint()
    checkpoint["connections"][0]["boundary_candidates"] = [{
        "boundary_time_utc": "2026-01-01T00:15:00",
        "state": {},
    }]
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        CanonicalReplayAccumulator.from_checkpoint(checkpoint)

    checkpoint = replay.to_checkpoint()
    checkpoint["connections"][0]["book"] = {
        **checkpoint["connections"][0]["book"],
        "version": 999,
    }
    with pytest.raises(ValueError, match="Unsupported checkpoint schema version"):
        CanonicalReplayAccumulator.from_checkpoint(checkpoint)

    checkpoint = replay.to_checkpoint()
    del checkpoint["has_frames"]
    with pytest.raises(ValueError, match="Malformed checkpoint keys"):
        CanonicalReplayAccumulator.from_checkpoint(checkpoint)

    replay = _canonical_replay(_stream(), ["conn-1"] * len(_stream()))
    checkpoint = replay.to_checkpoint()
    checkpoint["connections"][0]["candidate_cursor_utc"] = "2026-01-01T00:15:01Z"
    with pytest.raises(ValueError, match="Candidate cursor must use"):
        CanonicalReplayAccumulator.from_checkpoint(checkpoint)


def test_order_book_checkpoint_preserves_invalid_error_and_rejects_product_mismatch():
    book = Level2OrderBook(product_id="BTC-USD")
    book.invalidate("wire error")
    restored = _book_checkpoint_round_trip(book)
    assert restored.state == OrderBookState.INVALID
    assert restored.last_error == "wire error"
