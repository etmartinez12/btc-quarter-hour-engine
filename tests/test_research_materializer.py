from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
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


def _write_input(root: Path, *, future_frame: bool = False):
    writer = RawSegmentWriter(
        root,
        product_id="BTC-USD",
        session_id="session-materializer",
        max_frames=2,
    )
    frames = _stream(future_frame=future_frame)
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
    assert checkpoint["version"] == 1
    assert checkpoint["heartbeat_timeout_seconds"] == 5.0
    assert checkpoint["canonical_replay"]["heartbeat_timeout_seconds"] == 5.0
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
    target = checkpoint["canonical_replay"] if nested else checkpoint
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
        lambda checkpoint: checkpoint["canonical_replay"].update(version=999),
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


def test_multiple_sessions_fail_closed(tmp_path):
    input_root = tmp_path / "raw"
    _write_input(input_root)
    other = RawSegmentWriter(
        input_root,
        product_id="BTC-USD",
        session_id="second-session",
        max_frames=1,
    )
    raw = _frame("heartbeats", "2026-01-02T00:00:00Z", 1)
    other.add_frame(
        raw=raw,
        message_type="heartbeat",
        connection_id="second-connection",
        ingest_time_utc=_time("2026-01-02T00:00:00Z"),
    )

    with pytest.raises(ValueError, match="exactly one source session"):
        _materialize(input_root, tmp_path / "research")


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
    restored_features = _IngestFeatureReplay.from_checkpoint(checkpoint["feature_replay"])
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
