from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from btc_quarter_hour_engine.storage.forward_manifest import (
    build_forward_dataset_id,
    build_forward_manifest,
    write_forward_manifest,
)
from btc_quarter_hour_engine.storage.manifest import build_dataset_id, build_manifest, write_manifest


def test_manifest_stability_and_target_elegibility():
    manifest_a = build_manifest(
        source="coinbase_advanced",
        product_id="BTC-USD",
        data_kind="candles",
        requested_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        requested_end=datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
        granularity="ONE_MINUTE",
        raw_artifacts=[{"sha256": "abc"}],
        normalized_artifacts=[{"path": "x", "sha256": "def", "row_count": 1}],
        coverage={"expected_bucket_count": 60, "observed_bucket_count": 58, "missing_bucket_count": 2},
        canonical_target_eligible=False,
        canonical_target_ineligibility_reason="Historical OHLCV candles do not reproduce the exact best-bid/ask midpoint used by the canonical quarter-hour target.",
        request_count=1,
    )
    manifest_b = build_manifest(
        source="coinbase_advanced",
        product_id="BTC-USD",
        data_kind="candles",
        requested_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        requested_end=datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
        granularity="ONE_MINUTE",
        raw_artifacts=[{"sha256": "abc"}],
        normalized_artifacts=[{"path": "x", "sha256": "def", "row_count": 1}],
        coverage={"expected_bucket_count": 60, "observed_bucket_count": 58, "missing_bucket_count": 2},
        canonical_target_eligible=False,
        canonical_target_ineligibility_reason="Historical OHLCV candles do not reproduce the exact best-bid/ask midpoint used by the canonical quarter-hour target.",
        request_count=1,
    )
    assert manifest_a["dataset_id"] == manifest_b["dataset_id"]
    assert manifest_a["canonical_target_eligible"] is False
    assert "canonical" in manifest_a["canonical_target_ineligibility_reason"].lower()

    dataset_id = build_dataset_id(
        source="coinbase_advanced",
        product_id="BTC-USD",
        data_kind="candles",
        requested_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        requested_end=datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
        granularity="ONE_MINUTE",
        raw_artifact_hashes=["abc", "def"],
    )
    assert dataset_id


def test_dataset_id_depends_on_order_range_product_granularity():
    base = dict(source="coinbase_advanced", product_id="BTC-USD", data_kind="candles",
                requested_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                requested_end=datetime(2024, 1, 2, tzinfo=timezone.utc),
                granularity="ONE_MINUTE", raw_artifact_hashes=["one", "two"])
    original = build_dataset_id(**base)
    assert original == build_dataset_id(**base)
    assert build_dataset_id(**base, data_schema_version="2") != original
    for change in (
        {"raw_artifact_hashes": ["two", "one"]}, {"raw_artifact_hashes": ["one", "other"]},
        {"requested_end": datetime(2024, 1, 3, tzinfo=timezone.utc)},
        {"product_id": "ETH-USD"}, {"granularity": "FIVE_MINUTE"},
    ):
        assert build_dataset_id(**(base | change)) != original


def test_manifest_atomic_idempotent_and_conflicts_fail(tmp_path):
    manifest = build_manifest(
        source="coinbase_advanced", product_id="BTC-USD", data_kind="candles",
        requested_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        requested_end=datetime(2024, 1, 2, tzinfo=timezone.utc),
        granularity="ONE_MINUTE", raw_artifacts=[{"sha256": "abc"}],
        normalized_artifacts=[], coverage={"expected_bucket_count": 1440},
        canonical_target_eligible=False, canonical_target_ineligibility_reason="context only",
    )
    path = write_manifest(manifest, tmp_path)
    original = path.read_bytes()
    assert write_manifest(manifest, tmp_path) == path
    changed = {**manifest, "coverage": {"expected_bucket_count": 0}}
    with pytest.raises(ValueError, match="conflict"):
        write_manifest(changed, tmp_path)
    assert path.read_bytes() == original
    assert not list((tmp_path / "manifests").glob(".*"))


@pytest.mark.parametrize("change", [
    {"coverage": {"expected_bucket_count": 0}},
    {"canonical_target_eligible": True},
    {"schema_version": "2"},
])
def test_manifest_repeat_ignores_run_times_but_rejects_stable_conflicts(tmp_path, change):
    manifest = build_manifest(
        source="coinbase_advanced", product_id="BTC-USD", data_kind="candles",
        requested_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        requested_end=datetime(2024, 1, 1, 0, 2, tzinfo=timezone.utc),
        granularity="ONE_MINUTE", raw_artifacts=[{
            "sha256": "abc", "retrieved_at_utc": "2024-01-01T00:00:00Z",
        }], normalized_artifacts=[{"path": "part-a.parquet", "sha256": "def", "row_count": 2}],
        coverage={"expected_bucket_count": 2, "observed_bucket_count": 2},
        canonical_target_eligible=False, canonical_target_ineligibility_reason="context only",
        acquisition_started_at_utc=datetime(2024, 1, 1, tzinfo=timezone.utc),
        acquisition_completed_at_utc=datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc),
    )
    path = write_manifest(manifest, tmp_path)
    persisted_bytes = path.read_bytes()
    repeat = {
        **manifest,
        "acquisition_started_at_utc": "2024-01-03T00:00:00Z",
        "acquisition_completed_at_utc": "2024-01-03T00:01:00Z",
        "raw_artifacts": [{"sha256": "abc", "retrieved_at_utc": "2024-01-03T00:00:00Z"}],
    }
    assert write_manifest(repeat, tmp_path) == path
    assert path.read_bytes() == persisted_bytes
    conflicting = {**repeat, **change}
    with pytest.raises(ValueError, match="conflict"):
        write_manifest(conflicting, tmp_path)
    assert path.read_bytes() == persisted_bytes


def test_forward_manifest_is_session_shaped_and_content_addressed(tmp_path):
    raw_segments = [
        {"sha256": "a" * 64, "session_id": "session-1", "segment_index": 0},
        {"sha256": "b" * 64, "session_id": "session-1", "segment_index": 1},
    ]
    manifest = build_forward_manifest(
        source="coinbase_advanced",
        product_id="BTC-USD",
        session_id="session-1",
        websocket_url="wss://advanced-trade-ws.coinbase.com",
        channels=["heartbeats", "level2"],
        session_started_at_utc=datetime(2024, 1, 1, tzinfo=timezone.utc),
        session_completed_at_utc=datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
        termination_reason="duration_reached",
        connection_count=2,
        reconnect_count=1,
        connections=[{"connection_id": "connection-1"}],
        raw_segments=raw_segments,
        normalized_artifacts={
            "quarter_hour_bbo": [{"sha256": "c" * 64}],
            "level2_updates": [{"sha256": "d" * 64}],
            "bbo_state": [{"sha256": "e" * 64}],
        },
        quarter_hour_summary={
            "boundaries_seen": 3,
            "eligible_boundaries": 2,
            "ineligible_boundaries": 1,
        },
        integrity={"heartbeat_discontinuity_count": 1, "sequence_gap_count": 2},
    )

    assert manifest["forward_manifest_schema_version"] == "1"
    assert manifest["session_id"] == "session-1"
    assert manifest["dataset_id"] == build_forward_manifest(
        source="coinbase_advanced",
        product_id="BTC-USD",
        session_id="session-2",
        session_started_at_utc=datetime(2025, 1, 1, tzinfo=timezone.utc),
        raw_segments=raw_segments,
    )["dataset_id"]
    assert set(manifest["normalized_artifacts"]) == {
        "quarter_hour_bbo",
        "level2_updates",
        "bbo_state",
    }
    assert manifest["quarter_hour_summary"]["ineligible_boundaries"] == 1
    assert manifest["integrity"]["heartbeat_discontinuity_count"] == 1
    assert manifest["integrity"]["malformed_frame_count"] == 0
    assert manifest["canonical_source"] == {
        "price_definition": "best_bid_ask_midpoint",
        "boundary_schedule": ":00/:15/:30/:45 UTC",
    }

    path = write_forward_manifest(manifest, tmp_path)
    assert path.name == "session-1.json"
    assert write_forward_manifest(manifest, tmp_path) == path
    assert json.loads(path.read_text())["dataset_id"] == manifest["dataset_id"]
    assert len(list(path.parent.glob(".*"))) == 0


def test_forward_dataset_id_tracks_order_and_schema_versions():
    args = {
        "source": "coinbase_advanced",
        "product_id": "BTC-USD",
        "raw_segments": [{"sha256": "a"}, {"sha256": "b"}],
    }
    original = build_forward_manifest(**args)["dataset_id"]
    assert build_forward_manifest(**args)["dataset_id"] == original
    assert build_forward_manifest(
        **(args | {"raw_segments": [{"sha256": "b"}, {"sha256": "a"}]})
    )["dataset_id"] != original
    assert build_forward_dataset_id(
        **args,
        schema_versions={
            "raw_segment": "2",
            "level2_updates": "1",
            "bbo_state": "1",
            "quarter_hour_bbo": "1",
            "forward_manifest": "1",
        },
    ) != original
