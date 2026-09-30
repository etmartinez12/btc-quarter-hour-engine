from datetime import datetime, timezone

import numpy as np
import pytest

from btc_quarter_hour_engine.acquisition.coinbase_rest import (
    coverage_diagnostics_for_candles, parse_coinbase_candle_records, validate_candle_records,
)

START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def candle(start="1704067200", **overrides):
    return {"start": start, "low": "99", "high": "102", "open": "100", "close": "101", "volume": "0.5", **overrides}


def test_current_candle_shape_sorted_and_utc():
    parsed = parse_coinbase_candle_records({"candles": [candle("1704067260"), candle()]})
    validated = validate_candle_records(parsed)
    assert validated[0]["bucket_start"] == START
    assert validated[0]["bucket_start"].tzinfo is timezone.utc
    assert validated[0]["open"] == 100.0
    assert validated[1]["bucket_start"] > validated[0]["bucket_start"]
    assert set(validated[0]) == {"bucket_start", "open", "high", "low", "close", "volume"}


@pytest.mark.parametrize("overrides", [
    {"open": 0}, {"high": 0}, {"low": 0}, {"close": 0},
    {"open": -1}, {"volume": -1}, {"open": np.nan},
    {"high": np.inf}, {"low": -np.inf}, {"high": 98},
    {"high": 99}, {"high": 100}, {"low": 101}, {"low": 102},
])
def test_invalid_ohlcv_rejected(overrides):
    with pytest.raises(ValueError):
        validate_candle_records([{**parse_coinbase_candle_records({"candles": [candle()]})[0], **overrides}])


def test_invalid_and_duplicate_timestamp_rejected():
    with pytest.raises(ValueError, match="Invalid candle"):
        parse_coinbase_candle_records({"candles": [candle("not-a-timestamp")]})
    record = parse_coinbase_candle_records({"candles": [candle()]})[0]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_candle_records([record, record])


def test_half_open_coverage_and_missing_bucket():
    records = validate_candle_records(parse_coinbase_candle_records({
        "candles": [candle(), candle("1704067320")],
    }))
    coverage = coverage_diagnostics_for_candles(
        records=records, start=START,
        end=datetime(2024, 1, 1, 0, 3, tzinfo=timezone.utc), granularity="ONE_MINUTE",
    )
    assert coverage["expected_bucket_count"] == 3
    assert coverage["observed_bucket_count"] == 2
    assert coverage["missing_bucket_count"] == 1
    assert coverage["coverage_fraction"] == pytest.approx(2 / 3)
    assert "00:01:00" in coverage["missing_timestamps"][0]
    with pytest.raises(ValueError, match="outside"):
        coverage_diagnostics_for_candles(
            records=[*records, {"bucket_start": datetime(2024, 1, 1, 0, 3, tzinfo=timezone.utc)}],
            start=START, end=datetime(2024, 1, 1, 0, 3, tzinfo=timezone.utc), granularity="ONE_MINUTE",
        )
