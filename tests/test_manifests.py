from __future__ import annotations

from datetime import datetime, timezone

import pytest

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
