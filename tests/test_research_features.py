from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from btc_quarter_hour_engine.research.features import (
    _BucketStats,
    _IngestFeatureReplay,
    _ceil_quarter_hour,
    _window_features,
)


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _l2_message(event_type, seq, timestamp, updates):
    return json.dumps(
        {
            "channel": "l2_data",
            "client_id": "",
            "timestamp": timestamp,
            "sequence_num": seq,
            "events": [
                {
                    "type": event_type,
                    "product_id": "BTC-USD",
                    "updates": updates,
                }
            ],
        }
    ).encode()


def _snapshot(seq, timestamp):
    return _l2_message(
        "snapshot",
        seq,
        timestamp,
        [
            {
                "side": "bid",
                "price_level": "100.0",
                "new_quantity": "1.0",
                "event_time": timestamp,
            },
            {
                "side": "offer",
                "price_level": "101.0",
                "new_quantity": "1.0",
                "event_time": timestamp,
            },
        ],
    )


def _update(seq, timestamp, side="bid", price="100.5", quantity="2.0"):
    return _l2_message(
        "update",
        seq,
        timestamp,
        [
            {
                "side": side,
                "price_level": price,
                "new_quantity": quantity,
                "event_time": timestamp,
            }
        ],
    )


def _bucket_frame(replay):
    rows = []

    for bucket_end, bucket in sorted(replay.buckets.items()):
        rows.append(
            {
                "bucket_end": pd.Timestamp(bucket_end),
                **{
                    name: getattr(bucket, name)
                    for name in _BucketStats.__dataclass_fields__
                },
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("bucket_end")
        .set_index("bucket_end")
    )


def test_quarter_hour_bucket_is_right_closed():
    assert _ceil_quarter_hour(
        _utc("2026-01-01T00:14:59.999999Z")
    ) == _utc("2026-01-01T00:15:00Z")

    assert _ceil_quarter_hour(
        _utc("2026-01-01T00:15:00Z")
    ) == _utc("2026-01-01T00:15:00Z")

    assert _ceil_quarter_hour(
        _utc("2026-01-01T00:15:00.000001Z")
    ) == _utc("2026-01-01T00:30:00Z")


def test_feature_availability_uses_ingest_time_not_source_time():
    replay = _IngestFeatureReplay("BTC-USD")

    replay.consume(
        raw=_snapshot(0, "2026-01-01T00:14:50Z"),
        connection_id="conn-1",
        frame_index=0,
        ingest_time_utc=_utc("2026-01-01T00:14:59Z"),
    )

    # Source event is before 00:15, but the system does not receive it
    # until after 00:15. It must not affect the 00:15 prediction.
    replay.consume(
        raw=_update(1, "2026-01-01T00:14:55Z"),
        connection_id="conn-1",
        frame_index=1,
        ingest_time_utc=_utc("2026-01-01T00:15:01Z"),
    )

    buckets = _bucket_frame(replay)

    at_0015 = _window_features(
        buckets,
        timestamp=pd.Timestamp("2026-01-01T00:15:00Z"),
        window_minutes=15,
        current_midpoint=100.5,
    )

    at_0030 = _window_features(
        buckets,
        timestamp=pd.Timestamp("2026-01-01T00:30:00Z"),
        window_minutes=15,
        current_midpoint=100.5,
    )

    assert at_0015["l2_update_count_15m"] == 0
    assert at_0030["l2_update_count_15m"] == 1


def test_future_bucket_cannot_change_earlier_window():
    replay = _IngestFeatureReplay("BTC-USD")

    replay.consume(
        raw=_snapshot(0, "2026-01-01T00:14:50Z"),
        connection_id="conn-1",
        frame_index=0,
        ingest_time_utc=_utc("2026-01-01T00:14:59Z"),
    )

    before = _window_features(
        _bucket_frame(replay),
        timestamp=pd.Timestamp("2026-01-01T00:15:00Z"),
        window_minutes=15,
        current_midpoint=100.5,
    )

    replay.consume(
        raw=_update(1, "2026-01-01T00:15:01Z"),
        connection_id="conn-1",
        frame_index=1,
        ingest_time_utc=_utc("2026-01-01T00:15:01Z"),
    )

    after = _window_features(
        _bucket_frame(replay),
        timestamp=pd.Timestamp("2026-01-01T00:15:00Z"),
        window_minutes=15,
        current_midpoint=100.5,
    )

    assert before == after


def test_connection_change_resets_sequence_baseline():
    replay = _IngestFeatureReplay("BTC-USD")

    replay.consume(
        raw=_snapshot(100, "2026-01-01T00:00:00Z"),
        connection_id="conn-a",
        frame_index=0,
        ingest_time_utc=_utc("2026-01-01T00:00:00Z"),
    )

    replay.consume(
        raw=_snapshot(0, "2026-01-01T00:00:01Z"),
        connection_id="conn-b",
        frame_index=1,
        ingest_time_utc=_utc("2026-01-01T00:00:01Z"),
    )

    assert replay.sequence_gaps == 0
    assert replay.applied_l2_envelopes == 2


def test_asof_price_excludes_late_arriving_preboundary_event():
    replay = _IngestFeatureReplay("BTC-USD")

    # Snapshot is available before 00:15.
    replay.consume(
        raw=_snapshot(0, "2026-01-01T00:14:50Z"),
        connection_id="conn-1",
        frame_index=0,
        ingest_time_utc=_utc("2026-01-01T00:14:59Z"),
    )

    # Coinbase timestamps this update before 00:15, but our collector
    # receives it after 00:15. It belongs to the 00:30 feature bucket.
    replay.consume(
        raw=_update(
            1,
            "2026-01-01T00:14:55Z",
            side="bid",
            price="100.5",
            quantity="2.0",
        ),
        connection_id="conn-1",
        frame_index=1,
        ingest_time_utc=_utc("2026-01-01T00:15:01Z"),
    )

    buckets = _bucket_frame(replay)

    at_0015 = buckets.loc[
        pd.Timestamp("2026-01-01T00:15:00Z")
    ]

    at_0030 = buckets.loc[
        pd.Timestamp("2026-01-01T00:30:00Z")
    ]

    # 00:15 can only know the original 100 / 101 snapshot.
    assert at_0015["asof_midpoint"] == 100.5
    assert at_0015["asof_best_bid_size"] == 1.0

    # The late-arriving update becomes available only after 00:15.
    assert at_0030["asof_midpoint"] == 100.75
    assert at_0030["asof_best_bid_size"] == 2.0


def test_future_frames_do_not_mutate_previous_asof_bucket():
    replay = _IngestFeatureReplay("BTC-USD")

    replay.consume(
        raw=_snapshot(0, "2026-01-01T00:14:50Z"),
        connection_id="conn-1",
        frame_index=0,
        ingest_time_utc=_utc("2026-01-01T00:14:59Z"),
    )

    before = _bucket_frame(replay).loc[
        pd.Timestamp("2026-01-01T00:15:00Z")
    ].copy()

    replay.consume(
        raw=_update(
            1,
            "2026-01-01T00:15:01Z",
            side="bid",
            price="100.75",
            quantity="9.0",
        ),
        connection_id="conn-1",
        frame_index=1,
        ingest_time_utc=_utc("2026-01-01T00:15:01Z"),
    )

    after = _bucket_frame(replay).loc[
        pd.Timestamp("2026-01-01T00:15:00Z")
    ]

    pd.testing.assert_series_equal(
        before,
        after,
        check_names=False,
    )
