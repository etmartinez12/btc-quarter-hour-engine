import numpy as np
import pandas as pd
import pytest

from btc_quarter_hour_engine.acquisition import load_sample_market_data, normalize_market_frame
from btc_quarter_hour_engine.features import build_feature_frame
from btc_quarter_hour_engine.run_baseline import build_training_dataset


def _fully_warmed_feature_row(features: pd.DataFrame) -> pd.Series:
    feature_columns = [column for column in features.columns if column != "timestamp"]
    fully_warmed = features.dropna(subset=feature_columns)
    assert not fully_warmed.empty
    return fully_warmed.iloc[0]


def test_market_data_loader_rejects_duplicate_timestamps():
    raw = load_sample_market_data()
    duplicate = pd.concat([raw, raw.iloc[[100]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate timestamps"):
        normalize_market_frame(duplicate)


def test_build_training_dataset_rejects_missing_minute():
    raw = load_sample_market_data().iloc[:1500].copy()
    raw = raw.drop(raw.index[500]).reset_index(drop=True)

    with pytest.raises(ValueError, match="one-minute cadence"):
        build_training_dataset(market_data=raw)


def test_market_data_loader_rejects_non_minute_cadence():
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:01:00Z",
                    "2026-01-01T00:03:00Z",
                ],
                utc=True,
            ),
            "bid": [100.0, 101.0, 102.0],
            "ask": [101.0, 102.0, 103.0],
            "last_trade": [100.5, 101.5, 102.5],
            "volume": [10.0, 11.0, 12.0],
        }
    )

    with pytest.raises(ValueError, match="one-minute cadence"):
        normalize_market_frame(data)


def test_build_feature_frame_adds_phase_two_feature_families_on_boundary_rows():
    raw = load_sample_market_data()

    features = build_feature_frame(raw)

    expected_columns = {
        "price_acceleration_1m",
        "return_2m",
        "return_10m",
        "realized_vol_60m",
        "volume_60m",
        "range_15m",
        "price_vs_volume_weighted_midpoint_15m",
        "volume_zscore_60m",
        "price_vs_ema_15m",
        "price_vs_ema_60m",
        "ema_spread_15m_60m",
        "regime_trend_240m",
        "regime_vol_1440m",
        "momentum_5m_minus_30m",
        "volume_5m_to_30m_ratio",
        "order_book_imbalance",
        "depth_imbalance_5",
        "spread_change_1m",
        "trade_count_15m",
        "buy_volume_15m",
        "sell_volume_15m",
        "trade_flow_imbalance_15m",
        "average_trade_size_15m",
    }

    assert expected_columns.issubset(features.columns)
    assert features["timestamp"].dt.minute.mod(15).eq(0).all()
    assert features["timestamp"].dt.second.eq(0).all()


def test_build_feature_frame_is_invariant_to_future_changes():
    raw = load_sample_market_data()
    features = build_feature_frame(raw)
    target_row = _fully_warmed_feature_row(features)
    target_timestamp = target_row["timestamp"]
    before = target_row.copy()

    mutated = raw.copy()
    future_mask = mutated["timestamp"] > target_timestamp
    mutated.loc[future_mask, "bid"] *= 100.0
    mutated.loc[future_mask, "ask"] *= 100.0
    mutated.loc[future_mask, "last_trade"] *= 100.0
    mutated.loc[future_mask, "volume"] *= 1000.0
    if "bid_size" in mutated.columns:
        mutated.loc[future_mask, "bid_size"] = 0.001
        mutated.loc[future_mask, "ask_size"] = 9_999_999.0
        mutated.loc[future_mask, "depth_bid_5"] = 0.001
        mutated.loc[future_mask, "depth_ask_5"] = 9_999_999.0
        mutated.loc[future_mask, "trade_count"] = 1.0
        mutated.loc[future_mask, "buy_volume"] = 0.001
        mutated.loc[future_mask, "sell_volume"] = 9_999_999.0

    mutated_features = build_feature_frame(mutated)
    after = mutated_features.loc[mutated_features["timestamp"].eq(target_timestamp)].iloc[0]

    columns = [column for column in before.index if column != "timestamp"]
    pd.testing.assert_series_equal(before[columns], after[columns], check_names=False)


def test_build_feature_frame_changes_when_current_data_changes():
    raw = load_sample_market_data()
    features = build_feature_frame(raw)
    target_row = _fully_warmed_feature_row(features)
    target_timestamp = target_row["timestamp"]
    before = target_row.copy()

    mutated = raw.copy()
    recent_mask = (mutated["timestamp"] <= target_timestamp) & (mutated["timestamp"] >= target_timestamp - pd.Timedelta(minutes=60))
    mutated.loc[recent_mask, "bid"] *= 1.5
    mutated.loc[recent_mask, "ask"] *= 1.6
    mutated.loc[recent_mask, "last_trade"] *= 1.7
    mutated.loc[recent_mask, "volume"] *= 2.5
    if "bid_size" in mutated.columns:
        mutated.loc[recent_mask, "bid_size"] *= 1.25
        mutated.loc[recent_mask, "ask_size"] *= 1.25
        mutated.loc[recent_mask, "depth_bid_5"] *= 1.5
        mutated.loc[recent_mask, "depth_ask_5"] *= 1.5
        mutated.loc[recent_mask, "trade_count"] *= 1.5
        mutated.loc[recent_mask, "buy_volume"] *= 2.0
        mutated.loc[recent_mask, "sell_volume"] *= 2.0

    mutated_features = build_feature_frame(mutated)
    after = mutated_features.loc[mutated_features["timestamp"].eq(target_timestamp)].iloc[0]

    columns = [column for column in before.index if column != "timestamp"]
    assert not before[columns].equals(after[columns])


def test_build_feature_frame_rejects_missing_minute():
    raw = load_sample_market_data().iloc[:1500].copy()
    raw = raw.drop(raw.index[500]).reset_index(drop=True)

    with pytest.raises(ValueError, match="one-minute cadence"):
        build_feature_frame(raw)


def test_build_feature_frame_rejects_duplicate_timestamp():
    raw = load_sample_market_data()
    duplicate = pd.concat([raw, raw.iloc[[100]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate timestamps"):
        build_feature_frame(duplicate)


def test_build_training_dataset_rejects_duplicate_timestamp():
    raw = load_sample_market_data()
    duplicate = pd.concat([raw, raw.iloc[[100]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate timestamps"):
        build_training_dataset(market_data=duplicate)


def test_normalize_market_frame_rejects_invalid_timestamps():
    raw = load_sample_market_data().copy()

    for invalid_value in ["not-a-date", "2026-99-99T00:00:00Z", None]:
        mutated = raw.copy()
        mutated.loc[0, "timestamp"] = invalid_value
        with pytest.raises(ValueError, match="invalid timestamps"):
            normalize_market_frame(mutated)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bid", "bad"),
        ("bid", None),
        ("bid", np.nan),
        ("bid", np.inf),
        ("bid", -np.inf),
        ("ask", "bad"),
        ("ask", None),
        ("ask", np.nan),
        ("ask", np.inf),
        ("ask", -np.inf),
        ("last_trade", "bad"),
        ("last_trade", None),
        ("last_trade", np.nan),
        ("last_trade", np.inf),
        ("last_trade", -np.inf),
        ("volume", "bad"),
        ("volume", None),
        ("volume", np.nan),
        ("volume", np.inf),
        ("volume", -np.inf),
    ],
)
def test_normalize_market_frame_rejects_invalid_required_numeric_values(field, value):
    raw = load_sample_market_data().copy()
    mutated = raw.copy()
    mutated.loc[0, field] = value

    with pytest.raises(ValueError):
        normalize_market_frame(mutated)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bid", 0),
        ("bid", -1),
        ("ask", 0),
        ("ask", -1),
        ("last_trade", 0),
        ("last_trade", -1),
        ("volume", -1),
    ],
)
def test_normalize_market_frame_rejects_invalid_market_invariants(field, value):
    raw = load_sample_market_data().copy()
    mutated = raw.copy()
    mutated.loc[0, field] = value

    with pytest.raises(ValueError):
        normalize_market_frame(mutated)


def test_normalize_market_frame_accepts_zero_volume():
    raw = load_sample_market_data().copy()
    mutated = raw.copy()
    mutated.loc[0, "volume"] = 0.0

    normalized = normalize_market_frame(mutated)

    assert (normalized["volume"] >= 0).all()


def test_normalize_market_frame_rejects_crossed_quote():
    raw = load_sample_market_data().copy()
    mutated = raw.copy()
    mutated.loc[0, "bid"] = 101.0
    mutated.loc[0, "ask"] = 100.0

    with pytest.raises(ValueError, match="crossed"):
        normalize_market_frame(mutated)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bid_size", -1.0),
        ("bid_size", np.inf),
        ("bid_size", -np.inf),
        ("bid_size", np.nan),
        ("bid_size", "bad"),
        ("ask_size", -1.0),
        ("ask_size", np.inf),
        ("ask_size", -np.inf),
        ("ask_size", np.nan),
        ("ask_size", "bad"),
        ("depth_bid_5", -1.0),
        ("depth_bid_5", np.inf),
        ("depth_bid_5", -np.inf),
        ("depth_bid_5", np.nan),
        ("depth_bid_5", "bad"),
        ("depth_ask_5", -1.0),
        ("depth_ask_5", np.inf),
        ("depth_ask_5", -np.inf),
        ("depth_ask_5", np.nan),
        ("depth_ask_5", "bad"),
        ("trade_count", -1.0),
        ("trade_count", np.inf),
        ("trade_count", -np.inf),
        ("trade_count", np.nan),
        ("trade_count", "bad"),
        ("buy_volume", -1.0),
        ("buy_volume", np.inf),
        ("buy_volume", -np.inf),
        ("buy_volume", np.nan),
        ("buy_volume", "bad"),
        ("sell_volume", -1.0),
        ("sell_volume", np.inf),
        ("sell_volume", -np.inf),
        ("sell_volume", np.nan),
        ("sell_volume", "bad"),
    ],
)
def test_normalize_market_frame_rejects_invalid_optional_microstructure_values(field, value):
    raw = load_sample_market_data().copy()
    mutated = raw.copy()
    mutated.loc[0, field] = value

    with pytest.raises(ValueError):
        normalize_market_frame(mutated)


def test_normalize_market_frame_accepts_valid_out_of_order_input():
    raw = load_sample_market_data().copy()
    subset = raw.iloc[[10, 0, 7, 2, 5, 1, 8, 3, 9, 4, 6]].copy().reset_index(drop=True)

    normalized = normalize_market_frame(subset)

    expected = subset.sort_values("timestamp").reset_index(drop=True)
    pd.testing.assert_frame_equal(normalized, expected)
    assert normalized["timestamp"].is_monotonic_increasing
    assert len(normalized) == len(subset)
    assert normalized["timestamp"].nunique() == len(subset)


def test_sample_market_data_includes_microstructure_columns():
    raw = load_sample_market_data()

    expected_columns = {
        "bid_size",
        "ask_size",
        "depth_bid_5",
        "depth_ask_5",
        "trade_count",
        "buy_volume",
        "sell_volume",
    }

    assert expected_columns.issubset(raw.columns)
