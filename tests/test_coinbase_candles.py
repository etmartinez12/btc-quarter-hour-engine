from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from btc_quarter_hour_engine.acquisition.coinbase_rest import (
    coverage_diagnostics_for_candles,
    parse_coinbase_candle_records,
    validate_candle_records,
)


def test_parse_and_validate_candle_records():
    payload = {
        "candles": [
            {"time": "2024-01-01T12:00:00Z", "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5, "volume": 10.0},
            {"time": "2024-01-01T12:01:00Z", "open": 100.5, "high": 101.5, "low": 100.0, "close": 101.0, "volume": 12.5},
        ]
    }
    records = parse_coinbase_candle_records(payload)
    validated = validate_candle_records(records)
    assert len(validated) == 2
    assert validated[0]["time"].endswith("Z")
    assert pd.to_datetime(validated[0]["time"], utc=True).tzinfo is not None


def test_missing_bucket_diagnostics_count_missing_ranges():
    start = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 12, 4, tzinfo=timezone.utc)
    records = [
        {"time": "2024-01-01T12:00:00Z", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0},
        {"time": "2024-01-01T12:01:00Z", "open": 101.0, "high": 102.0, "low": 100.0, "close": 101.0, "volume": 2.0},
        {"time": "2024-01-01T12:03:00Z", "open": 103.0, "high": 104.0, "low": 102.0, "close": 103.0, "volume": 3.0},
        {"time": "2024-01-01T12:04:00Z", "open": 104.0, "high": 105.0, "low": 103.0, "close": 104.0, "volume": 4.0},
    ]
    coverage = coverage_diagnostics_for_candles(records=records, start=start, end=end, granularity="ONE_MINUTE")
    assert coverage["expected_bucket_count"] == 5
    assert coverage["observed_bucket_count"] == 4
    assert coverage["missing_bucket_count"] == 1
    assert coverage["coverage_fraction"] == pytest.approx(0.8)
    assert "2024-01-01T12:02:00+00:00" in coverage["missing_timestamps"][0]


def test_validate_rejects_inconsistent_ohlc():
    with pytest.raises(ValueError):
        validate_candle_records([
            {"time": "2024-01-01T12:00:00Z", "open": 100.0, "high": 99.0, "low": 98.0, "close": 99.5, "volume": 1.0}
        ])
