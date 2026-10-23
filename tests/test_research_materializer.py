from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from btc_quarter_hour_engine.acquisition.config import CoinbaseWebSocketConfig
from btc_quarter_hour_engine.research import materializer
from btc_quarter_hour_engine.research.features import (
    PREDICTOR_CONTRACT_VERSION,
    PREDICTOR_COLUMNS,
    _IngestFeatureReplay,
    build_raw_research_feature_frame,
    replay_raw_feature_buckets,
)
from btc_quarter_hour_engine.research.forward_dataset import (
    build_research_dataset,
    discover_sealed_segments,
)
from btc_quarter_hour_engine.storage.raw_segments import RawSegmentWriter


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _frame(channel: str, timestamp: str, sequence: int, *, kind: str = "") -> bytes:
    if channel == "l2_data":
        updates = [
            {
                "side": side,
                "event_time": timestamp,
                "price_level": price,
                "new_quantity": quantity,
            }
            for side, price, quantity in (
                ("bid", "100", "1.5"),
                ("offer", "101", "2.5"),
            )
        ]
        events = [{"type": kind, "product_id": "BTC-USD", "updates": updates}]
    else:
        events = [{"current_time": timestamp, "heartbeat_counter": str(sequence)}]
    return json.dumps(
        {
            "channel": channel,
            "timestamp": timestamp,
            "sequence_num": sequence,
            "events": events,
        },
        separators=(",", ":"),
    ).encode()


def _stream(*, future_frame: bool = False) -> list[bytes]:
    records = [
        ("l2_data", "2026-01-01T00:14:50Z", "snapshot"),
        ("heartbeats", "2026-01-01T00:14:55Z", ""),
        ("l2_data", "2026-01-01T00:14:59Z", "update"),
        ("heartbeats", "2026-01-01T00:15:01Z", ""),
        ("l2_data", "2026-01-01T00:15:05Z", "update"),
        ("heartbeats", "2026-01-01T00:29:59Z", ""),
        ("l2_data", "2026-01-01T00:30:01Z", "update"),
        ("heartbeats", "2026-01-01T00:44:59Z", ""),
        ("l2_data", "2026-01-01T00:45:01Z", "update"),
    ]
    if future_frame:
        records.extend(
            [
                ("heartbeats", "2026-01-01T00:45:02Z", ""),
                ("heartbeats", "2026-01-01T01:00:00Z", ""),
            ]
        )
    return [
        _frame(channel, timestamp, index + 1, kind=kind)
        for index, (channel, timestamp, kind) in enumerate(records)
    ]


def _long_stream() -> list[bytes]:
    start = _time("2026-01-01T00:14:50Z")
    records = [
        ("l2_data", start, "snapshot"),
        ("heartbeats", start + timedelta(seconds=5), ""),
        ("l2_data", start + timedelta(seconds=9), "update"),
    ]
    for quarter in range(1, 21):
        quarter_start = start + timedelta(minutes=15 * quarter)
        records.extend([
            ("heartbeats", quarter_start + timedelta(seconds=1), ""),
            ("l2_data", quarter_start + timedelta(seconds=5), "update"),
        ])
    return [
        _frame(channel, timestamp.isoformat().replace("+00:00", "Z"), index + 1, kind=kind)
        for index, (channel, timestamp, kind) in enumerate(records)
    ]


def _shift_stream(frames: list[bytes], offset: timedelta) -> list[bytes]:
    shifted = []
    for raw in frames:
        payload = json.loads(raw)

        def shift_times(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in {"timestamp", "event_time", "current_time"} and isinstance(item, str):
                        value[key] = (
                            _time(item) + offset
                        ).isoformat().replace("+00:00", "Z")
                    else:
                        shift_times(item)
            elif isinstance(value, list):
                for item in value:
                    shift_times(item)

        shift_times(payload)
        shifted.append(json.dumps(payload, separators=(",", ":")).encode())
    return shifted


def _shift_prices(frames: list[bytes], offset: float) -> list[bytes]:
    shifted = []
    for raw in frames:
        payload = json.loads(raw)

        def shift_levels(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == "price_level":
                        value[key] = str(float(item) + offset)
                    else:
                        shift_levels(item)
            elif isinstance(value, list):
                for item in value:
                    shift_levels(item)

        shift_levels(payload)
        shifted.append(json.dumps(payload, separators=(",", ":")).encode())
    return shifted


def _write_input(
    root: Path,
    *,
    future_frame: bool = False,
    session_id: str = "session-materializer",
    time_offset: timedelta = timedelta(0),
    price_offset: float = 0.0,
    long: bool = False,
):
    writer = RawSegmentWriter(
        root,
        product_id="BTC-USD",
        session_id=session_id,
        max_frames=7 if long else 2,
    )
    frames = _long_stream() if long else _stream(future_frame=future_frame)
    if time_offset:
        frames = _shift_stream(frames, time_offset)
    if price_offset:
        frames = _shift_prices(frames, price_offset)
    for raw in frames:
        payload = json.loads(raw)
        timestamp = payload["timestamp"]
        writer.add_frame(
            raw=raw,
            message_type="heartbeat" if payload["channel"] == "heartbeats" else payload["events"][0]["type"],
            connection_id="connection-1",
            ingest_time_utc=_time(timestamp),
        )
    writer.seal()
    return writer, frames


def _materialize(
    input_root: Path, output_root: Path, *, heartbeat_timeout_seconds=5.0, **kwargs
):
    return materializer.materialize_research(
        input_root=input_root,
        output_root=output_root,
        websocket_config=CoinbaseWebSocketConfig(
            product_id="BTC-USD",
            heartbeat_timeout_seconds=heartbeat_timeout_seconds,
        ),
        **kwargs,
    )


def _frames(output_root: Path) -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_parquet(output_root / name)
        for name in (
            "canonical_boundaries.parquet",
            "direction_targets.parquet",
            "feature_buckets.parquet",
            "research_features.parquet",
        )
    }


def test_one_shot_and_resumed_materialization_are_identical(tmp_path):
    input_root = tmp_path / "raw"
    _write_input(input_root)
    one_shot_root = tmp_path / "one-shot"
    resumed_root = tmp_path / "resumed"

    _materialize(input_root, one_shot_root)
    first = _materialize(
        input_root,
        resumed_root,
        max_new_segments=2,
        checkpoint_every_segments=2,
    )
    second = _materialize(
        input_root,
        resumed_root,
        max_new_segments=20,
        checkpoint_every_segments=2,
    )

    one_checkpoint = json.loads((one_shot_root / "checkpoint.json").read_text())
    resumed_checkpoint = json.loads((resumed_root / "checkpoint.json").read_text())
    assert resumed_checkpoint == one_checkpoint
    assert first.diagnostics["new_segments_processed_this_run"] == 2
    assert second.diagnostics["new_segments_processed_this_run"] == 3
    assert second.diagnostics["expected_next_frame_index"] == len(_stream())
    for name, one_frame in _frames(one_shot_root).items():
        pd.testing.assert_frame_equal(one_frame, _frames(resumed_root)[name], check_exact=True)


def test_materializer_matches_full_replay_oracle_value_for_value(tmp_path):
    input_root = tmp_path / "raw"
    _write_input(input_root)
    discovered = discover_sealed_segments(input_root)
    dataset = build_research_dataset(
        input_root=input_root,
        output_root=tmp_path / "full-replay",
        discovered_segments=discovered,
    )
    oracle_buckets, _ = replay_raw_feature_buckets(
        dataset.manifest_path,
        input_root,
    )
    oracle_features, _ = build_raw_research_feature_frame(
        dataset.manifest_path,
        input_root,
    )
    _materialize(input_root, tmp_path / "incremental")
    materialized = _frames(tmp_path / "incremental")

    expected_boundaries = pd.read_parquet(dataset.boundary_artifact)
    expected_targets = pd.read_parquet(dataset.target_artifact)
    expected_features = oracle_features
    expected_buckets = oracle_buckets.reset_index()
    pd.testing.assert_frame_equal(
        materialized["canonical_boundaries.parquet"], expected_boundaries, check_exact=True
    )
    pd.testing.assert_frame_equal(
        materialized["direction_targets.parquet"], expected_targets, check_exact=True
    )
    pd.testing.assert_frame_equal(
        materialized["feature_buckets.parquet"], expected_buckets, check_exact=True
    )
    pd.testing.assert_frame_equal(
        materialized["research_features.parquet"], expected_features, check_exact=True
    )
    assert PREDICTOR_CONTRACT_VERSION == "bbo-causal-v1"
    assert len(PREDICTOR_COLUMNS) == 114


def test_noop_does_not_replay_processed_prefix_and_preserves_checkpoint(tmp_path, monkeypatch):
    input_root = tmp_path / "raw"
    _write_input(input_root)
    output_root = tmp_path / "research"
    _materialize(input_root, output_root)
    checkpoint_before = (output_root / "checkpoint.json").read_bytes()

    def unexpected_replay(*args, **kwargs):
        raise AssertionError("no-op invocation replayed a committed raw segment")

    monkeypatch.setattr(materializer, "_iter_segment_frames", unexpected_replay)
    result = _materialize(input_root, output_root)

    assert result.diagnostics["new_segments_processed_this_run"] == 0
    assert result.diagnostics["new_frames_processed_this_run"] == 0
    assert (output_root / "checkpoint.json").read_bytes() == checkpoint_before
    checkpoint = json.loads(checkpoint_before)
    assert checkpoint["version"] == 2
    assert checkpoint["heartbeat_timeout_seconds"] == 5.0
    assert (
        checkpoint["processed_sessions"][-1]["canonical_replay"][
            "heartbeat_timeout_seconds"
        ]
        == 5.0
    )
    assert result.diagnostics["heartbeat_timeout_seconds"] == 5.0
    metadata = json.loads(result.materialization_path.read_text())
    assert metadata["heartbeat_timeout_seconds"] == 5.0


def test_resume_rejects_changed_heartbeat_timeout_before_suffix_replay(tmp_path, monkeypatch):
    input_root = tmp_path / "raw"
    _write_input(input_root)
    output_root = tmp_path / "research"
    _materialize(input_root, output_root, max_new_segments=2)
    checkpoint_before = (output_root / "checkpoint.json").read_bytes()

    def unexpected_replay(*args, **kwargs):
        raise AssertionError("configuration validation must precede suffix replay")

    monkeypatch.setattr(materializer, "_iter_segment_frames", unexpected_replay)
    with pytest.raises(ValueError, match="heartbeat timeout does not match requested configuration"):
        _materialize(input_root, output_root, heartbeat_timeout_seconds=30.0)
    assert (output_root / "checkpoint.json").read_bytes() == checkpoint_before


@pytest.mark.parametrize("nested", [False, True], ids=["top-level", "nested-canonical"])
def test_restore_rejects_inconsistent_checkpoint_heartbeat_timeout(tmp_path, monkeypatch, nested):
    input_root = tmp_path / "raw"
    _write_input(input_root)
    output_root = tmp_path / "research"
    _materialize(input_root, output_root, max_new_segments=2)
    checkpoint_path = output_root / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    target = (
        checkpoint["processed_sessions"][-1]["canonical_replay"]
        if nested
        else checkpoint
    )
    target["heartbeat_timeout_seconds"] = 30.0
    checkpoint_path.write_text(json.dumps(checkpoint, allow_nan=False))

    def unexpected_replay(*args, **kwargs):
        raise AssertionError("checkpoint validation must precede suffix replay")

    monkeypatch.setattr(materializer, "_iter_segment_frames", unexpected_replay)
    with pytest.raises(ValueError, match="heartbeat timeout does not match"):
        _materialize(input_root, output_root)


@pytest.mark.parametrize("timeout", [True, None, "5.0", -1, float("nan"), float("inf")])
def test_checkpoint_rejects_invalid_top_level_heartbeat_timeout(tmp_path, timeout):
    input_root = tmp_path / "raw"
    _write_input(input_root)
    output_root = tmp_path / "research"
    _materialize(input_root, output_root, max_new_segments=2)
    checkpoint_path = output_root / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["heartbeat_timeout_seconds"] = timeout
    checkpoint_path.write_text(json.dumps(checkpoint))

    with pytest.raises(ValueError):
        _materialize(input_root, output_root)


def test_resume_accepts_equivalent_integer_heartbeat_timeout(tmp_path):
    input_root = tmp_path / "raw"
    _write_input(input_root)
    output_root = tmp_path / "research"
    _materialize(input_root, output_root)
    checkpoint_before = (output_root / "checkpoint.json").read_bytes()

    result = _materialize(input_root, output_root, heartbeat_timeout_seconds=5)

    assert result.diagnostics["new_frames_processed_this_run"] == 0
    assert (output_root / "checkpoint.json").read_bytes() == checkpoint_before


def test_prefix_metadata_mutation_fails_before_processing_any_segment(tmp_path, monkeypatch):
    input_root = tmp_path / "raw"
    writer, _ = _write_input(input_root)
    output_root = tmp_path / "research"
    _materialize(input_root, output_root, max_new_segments=1)

    sidecar = Path(writer.sealed_segments[0]["metadata_path"])
    metadata = json.loads(sidecar.read_text())
    metadata["identity_mutation"] = True
    sidecar.write_text(json.dumps(metadata))

    def unexpected_replay(*args, **kwargs):
        raise AssertionError("prefix verification should precede segment replay")

    monkeypatch.setattr(materializer, "_iter_segment_frames", unexpected_replay)
    with pytest.raises(ValueError, match="prefix changed"):
        _materialize(input_root, output_root)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda checkpoint: checkpoint.update(expected_next_frame_index=99),
        lambda checkpoint: checkpoint["processed_segments"].reverse(),
        lambda checkpoint: checkpoint["processed_segments"].pop(),
        lambda checkpoint: checkpoint.update(source="wrong-source"),
        lambda checkpoint: checkpoint.update(product_id="ETH-USD"),
        lambda checkpoint: checkpoint.update(predictor_contract_version="changed"),
        lambda checkpoint: checkpoint["processed_sessions"][-1][
            "canonical_replay"
        ].update(version=999),
    ],
)
def test_corrupt_checkpoint_fails_closed(tmp_path, mutation):
    input_root = tmp_path / "raw"
    _write_input(input_root)
    output_root = tmp_path / "research"
    _materialize(input_root, output_root)
    checkpoint_path = output_root / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    mutation(checkpoint)
    checkpoint_path.write_text(json.dumps(checkpoint))

    with pytest.raises(ValueError):
        _materialize(input_root, output_root)


def test_single_session_v1_checkpoint_is_intentionally_migrated_on_resume(tmp_path):
    input_root = tmp_path / "raw"
    _write_input(input_root)
    output_root = tmp_path / "research"
    _materialize(input_root, output_root, max_new_segments=2)
    checkpoint_path = output_root / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    session = checkpoint["processed_sessions"][0]
    checkpoint_v1 = {
        "version": 1,
        "source": checkpoint["source"],
        "product_id": checkpoint["product_id"],
        "predictor_contract_version": checkpoint["predictor_contract_version"],
        "heartbeat_timeout_seconds": checkpoint["heartbeat_timeout_seconds"],
        "session_id": session["session_id"],
        "expected_next_frame_index": session["expected_next_frame_index"],
        "first_ingest_time_utc": checkpoint["first_ingest_time_utc"],
        "last_ingest_time_utc": checkpoint["last_ingest_time_utc"],
        "processed_segments": checkpoint["processed_segments"],
        "feature_replay": session["feature_replay"],
        "canonical_replay": session["canonical_replay"],
    }
    checkpoint_path.write_text(json.dumps(checkpoint_v1))

    _materialize(input_root, output_root)
    migrated = json.loads(checkpoint_path.read_text())
    assert migrated["version"] == 2
    assert len(migrated["processed_sessions"]) == 1
    assert migrated["session_id"] == "session-materializer"


def test_active_partial_is_ignored_and_counted(tmp_path):
    input_root = tmp_path / "raw"
    writer, _ = _write_input(input_root)
    writer.add_frame(
        raw=_frame("heartbeats", "2026-01-01T01:01:00Z", 10),
        message_type="heartbeat",
        connection_id="connection-1",
        ingest_time_utc=_time("2026-01-01T01:01:00Z"),
    )

    result = _materialize(input_root, tmp_path / "research")
    assert result.diagnostics["active_partial_count_ignored"] == 1
    assert result.diagnostics["processed_segment_count"] == len(writer.sealed_segments)
    assert result.diagnostics["total_processed_frame_count"] == len(_stream())
    assert writer.active_partial_path.exists()


def test_atomic_checkpoint_failure_before_replace_preserves_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.json"
    original = b'{"version":1}'
    path.write_bytes(original)

    def fail_replace(*args, **kwargs):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(materializer.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        materializer._atomic_write_json(path, {"version": 2}, prefix=".checkpoint.")

    assert path.read_bytes() == original
    assert list(tmp_path.glob(".checkpoint.*.tmp")) == []


def test_full_and_resumed_materialization_across_session_rollover_are_identical(tmp_path):
    input_root = tmp_path / "raw"
    _write_input(input_root, session_id="z-session", long=True)
    resumed_root = tmp_path / "resumed"
    _materialize(input_root, resumed_root)
    _write_input(
        input_root,
        session_id="a-session",
        time_offset=timedelta(hours=5, seconds=6),
        long=True,
    )

    full_root = tmp_path / "full"
    _materialize(input_root, full_root)
    resumed = _materialize(input_root, resumed_root)
    assert resumed.diagnostics["processed_session_count"] == 2
    assert resumed.diagnostics["ordered_processed_session_ids"] == [
        "z-session",
        "a-session",
    ]
    assert resumed.diagnostics["latest_processed_session_id"] == "a-session"
    assert json.loads((full_root / "checkpoint.json").read_text()) == json.loads(
        (resumed_root / "checkpoint.json").read_text()
    )
    for name, full_frame in _frames(full_root).items():
        pd.testing.assert_frame_equal(
            full_frame,
            _frames(resumed_root)[name],
            check_exact=True,
        )

    dataset = build_research_dataset(
        input_root=input_root,
        output_root=tmp_path / "oracle",
        discovered_segments=discover_sealed_segments(input_root),
    )
    oracle_buckets, _ = replay_raw_feature_buckets(
        dataset.manifest_path, input_root
    )
    oracle_features, _ = build_raw_research_feature_frame(
        dataset.manifest_path, input_root
    )
    pd.testing.assert_frame_equal(
        _frames(full_root)["canonical_boundaries.parquet"],
        pd.read_parquet(dataset.boundary_artifact),
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        _frames(full_root)["direction_targets.parquet"],
        pd.read_parquet(dataset.target_artifact),
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        _frames(full_root)["feature_buckets.parquet"],
        oracle_buckets.reset_index(),
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        _frames(full_root)["research_features.parquet"],
        oracle_features,
        check_exact=True,
    )

    boundaries = pd.read_parquet(full_root / "canonical_boundaries.parquet")
    features = pd.read_parquet(full_root / "research_features.parquet")
    session_b_boundaries = boundaries.loc[
        boundaries["session_id"] == "a-session", "boundary_time_utc"
    ]
    session_b_features = features.loc[
        features["timestamp"].between(
            session_b_boundaries.min(), session_b_boundaries.max()
        )
    ]
    assert session_b_features["feature_ready"].any()
    first_ready = session_b_features.loc[
        session_b_features["feature_ready"], "timestamp"
    ].min()
    assert first_ready >= session_b_boundaries.min() + pd.Timedelta(minutes=240)
    buckets = pd.read_parquet(full_root / "feature_buckets.parquet")
    assert set(buckets["session_id"]) == {"z-session", "a-session"}

    checkpoint_before_noop = (resumed_root / "checkpoint.json").read_bytes()
    no_op = _materialize(input_root, resumed_root)
    assert no_op.diagnostics["new_segments_processed_this_run"] == 0
    assert no_op.diagnostics["new_frames_processed_this_run"] == 0
    assert (resumed_root / "checkpoint.json").read_bytes() == checkpoint_before_noop
    for name, frame in _frames(full_root).items():
        pd.testing.assert_frame_equal(
            frame, _frames(resumed_root)[name], check_exact=True
        )


def test_overlapping_sessions_fail_closed(tmp_path):
    input_root = tmp_path / "raw"
    _write_input(input_root, session_id="session-a")
    _write_input(
        input_root,
        session_id="session-b",
        time_offset=timedelta(minutes=1),
    )

    with pytest.raises(ValueError, match="overlap or cannot be uniquely ordered"):
        _materialize(input_root, tmp_path / "research")


def test_malformed_first_segment_of_next_session_fails_closed(tmp_path):
    input_root = tmp_path / "raw"
    _write_input(input_root, session_id="session-a")
    writer, _ = _write_input(
        input_root,
        session_id="session-b",
        time_offset=timedelta(hours=2),
    )
    first = writer.sealed_segments[0]
    Path(first["path"]).unlink()
    Path(first["metadata_path"]).unlink()

    with pytest.raises(ValueError, match="Noncontiguous sealed segment indexes"):
        _materialize(input_root, tmp_path / "research")


def test_interleaved_sessions_and_return_to_an_earlier_session_fail_closed(tmp_path):
    input_root = tmp_path / "raw"
    first = RawSegmentWriter(
        input_root, product_id="BTC-USD", session_id="session-a", max_frames=1
    )
    second = RawSegmentWriter(
        input_root, product_id="BTC-USD", session_id="session-b", max_frames=1
    )
    for writer, stamp, sequence in (
        (first, "2026-01-01T00:00:00Z", 1),
        (second, "2026-01-01T00:01:00Z", 1),
    ):
        writer.add_frame(
            raw=_frame("heartbeats", stamp, sequence),
            message_type="heartbeat",
            connection_id=f"{writer.session_id}-connection",
            ingest_time_utc=_time(stamp),
        )
    first.add_frame(
        raw=_frame("heartbeats", "2026-01-01T00:02:00Z", 2),
        message_type="heartbeat",
        connection_id="session-a-connection",
        ingest_time_utc=_time("2026-01-01T00:02:00Z"),
    )

    with pytest.raises(ValueError, match="overlap or cannot be uniquely ordered"):
        _materialize(input_root, tmp_path / "research")


def test_regressing_ingest_times_within_session_fail_closed(tmp_path):
    input_root = tmp_path / "raw"
    writer = RawSegmentWriter(
        input_root, product_id="BTC-USD", session_id="session-a", max_frames=1
    )
    for sequence, stamp in enumerate(
        ("2026-01-01T00:02:00Z", "2026-01-01T00:01:00Z"),
        start=1,
    ):
        writer.add_frame(
            raw=_frame("heartbeats", stamp, sequence),
            message_type="heartbeat",
            connection_id="session-a-connection",
            ingest_time_utc=_time(stamp),
        )

    with pytest.raises(ValueError, match="Ingest time regressed within session"):
        _materialize(input_root, tmp_path / "research")


def test_new_session_requires_its_own_order_book_snapshot(tmp_path):
    input_root = tmp_path / "raw"
    _write_input(input_root, session_id="session-a")
    writer = RawSegmentWriter(
        input_root, product_id="BTC-USD", session_id="session-b", max_frames=1
    )
    stamp = "2026-01-01T02:00:00Z"
    writer.add_frame(
        raw=_frame("heartbeats", stamp, 1),
        message_type="heartbeat",
        connection_id="session-b-connection",
        ingest_time_utc=_time(stamp),
    )

    output_root = tmp_path / "research"
    _materialize(input_root, output_root)
    boundaries = pd.read_parquet(output_root / "canonical_boundaries.parquet")
    session_b = boundaries.loc[boundaries["session_id"] == "session-b"]
    assert not session_b.empty
    assert not session_b["book_synced"].any()
    assert not session_b["canonical_target_eligible"].any()


def test_endpoint_valid_direction_target_is_preserved_across_session_transition(tmp_path):
    input_root = tmp_path / "raw"
    _write_input(input_root, session_id="session-a")
    _write_input(
        input_root,
        session_id="session-b",
        time_offset=timedelta(minutes=45),
        price_offset=1.0,
    )
    output_root = tmp_path / "research"
    _materialize(input_root, output_root)
    targets = pd.read_parquet(output_root / "direction_targets.parquet")

    crossing = targets.loc[
        (targets["session_id_t"] == "session-a")
        & (targets["session_id_t_plus_15m"] == "session-b")
    ]
    assert len(crossing) == 1
    assert bool(crossing.iloc[0]["target_eligible"])


def test_corrupted_checkpoint_session_identity_fails_closed(tmp_path):
    input_root = tmp_path / "raw"
    _write_input(input_root, session_id="session-a")
    _write_input(
        input_root,
        session_id="session-b",
        time_offset=timedelta(hours=2),
    )
    output_root = tmp_path / "research"
    _materialize(input_root, output_root)
    checkpoint_path = output_root / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["processed_sessions"][0]["canonical_replay"]["session_id"] = "session-b"
    checkpoint_path.write_text(json.dumps(checkpoint))

    with pytest.raises(ValueError, match="canonical replay checkpoint session_id"):
        _materialize(input_root, output_root)


def test_output_root_inside_raw_input_is_rejected(tmp_path):
    input_root = tmp_path / "raw"
    _write_input(input_root)
    with pytest.raises(ValueError, match="outside input_root"):
        _materialize(input_root, input_root / "derived")


def test_future_partial_bucket_is_checkpointed_then_exported_after_boundary_advances(tmp_path):
    input_root = tmp_path / "raw"
    writer, frames = _write_input(input_root, future_frame=True)
    output_root = tmp_path / "research"

    _materialize(input_root, output_root, max_new_segments=5)
    checkpoint = json.loads((output_root / "checkpoint.json").read_text())
    restored_features = _IngestFeatureReplay.from_checkpoint(
        checkpoint["processed_sessions"][-1]["feature_replay"]
    )
    future_bucket = _time("2026-01-01T01:00:00Z")
    assert future_bucket in restored_features.buckets
    first_export = pd.read_parquet(output_root / "feature_buckets.parquet")
    assert pd.to_datetime(first_export["bucket_end"], utc=True).max() < pd.Timestamp(future_bucket)

    result = _materialize(input_root, output_root)
    second_export = pd.read_parquet(output_root / "feature_buckets.parquet")
    assert result.diagnostics["new_segments_processed_this_run"] == 1
    assert pd.Timestamp(future_bucket) in pd.to_datetime(second_export["bucket_end"], utc=True).tolist()
    discovered = discover_sealed_segments(input_root)
    dataset = build_research_dataset(
        input_root=input_root,
        output_root=tmp_path / "oracle",
        discovered_segments=discovered,
    )
    oracle_buckets, _ = replay_raw_feature_buckets(dataset.manifest_path, input_root)
    pd.testing.assert_frame_equal(
        second_export,
        oracle_buckets.reset_index(),
        check_exact=True,
    )
    assert len(frames) == 11
    assert writer.sealed_segments


def test_discovery_is_snapshotted_once_per_invocation(tmp_path, monkeypatch):
    input_root = tmp_path / "raw"
    _write_input(input_root)
    original = materializer.discover_sealed_segments
    calls = 0

    def counted_discovery(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(materializer, "discover_sealed_segments", counted_discovery)
    _materialize(input_root, tmp_path / "research")
    assert calls == 1


def test_checkpoint_json_uses_deterministic_strict_encoding(tmp_path):
    input_root = tmp_path / "raw"
    _write_input(input_root)
    output_root = tmp_path / "research"
    _materialize(input_root, output_root)
    payload = (output_root / "checkpoint.json").read_bytes()
    checkpoint = json.loads(payload)
    assert json.dumps(
        checkpoint,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode() == payload
    assert hashlib.sha256(payload).hexdigest() == materializer._sha256_file(
        output_root / "checkpoint.json"
    )
