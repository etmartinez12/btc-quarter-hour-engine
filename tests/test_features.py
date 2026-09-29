import pandas as pd
import pytest

from btc_quarter_hour_engine.acquisition import load_sample_market_data, normalize_market_frame
from btc_quarter_hour_engine.features import build_feature_frame
from btc_quarter_hour_engine.run_baseline import build_training_dataset


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
    target_timestamp = features["timestamp"].iloc[5]
    before = features.loc[features["timestamp"].eq(target_timestamp)].iloc[0]

    mutated = raw.copy()
    future_mask = mutated["timestamp"] > target_timestamp
    mutated.loc[future_mask, "bid"] *= 10.0
    mutated.loc[future_mask, "ask"] *= 10.0
    mutated.loc[future_mask, "last_trade"] *= 10.0
    mutated.loc[future_mask, "volume"] *= 1000.0
    if "bid_size" in mutated.columns:
        mutated.loc[future_mask, "bid_size"] = 0.001
        mutated.loc[future_mask, "ask_size"] = 999999.0
        mutated.loc[future_mask, "buy_volume"] *= 1000.0
        mutated.loc[future_mask, "sell_volume"] *= 0.001

    mutated_features = build_feature_frame(mutated)
    after = mutated_features.loc[mutated_features["timestamp"].eq(target_timestamp)].iloc[0]

    columns = [column for column in before.index if column != "timestamp"]
    pd.testing.assert_series_equal(before[columns], after[columns], check_names=False)


def test_build_feature_frame_changes_when_current_data_changes():
    raw = load_sample_market_data()
    features = build_feature_frame(raw)
    target_timestamp = features["timestamp"].iloc[5]
    before = features.loc[features["timestamp"].eq(target_timestamp)].iloc[0]

    mutated = raw.copy()
    mutated.loc[mutated["timestamp"] <= target_timestamp, "bid"] *= 2.0
    mutated.loc[mutated["timestamp"] <= target_timestamp, "ask"] *= 2.0
    mutated.loc[mutated["timestamp"] <= target_timestamp, "volume"] *= 3.0

    mutated_features = build_feature_frame(mutated)
    after = mutated_features.loc[mutated_features["timestamp"].eq(target_timestamp)].iloc[0]

    columns = [column for column in before.index if column != "timestamp"]
    assert not before[columns].equals(after[columns])


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
