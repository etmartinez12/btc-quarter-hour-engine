from __future__ import annotations

from datetime import datetime, timezone

from btc_quarter_hour_engine.storage.manifest import build_dataset_id, build_manifest


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
