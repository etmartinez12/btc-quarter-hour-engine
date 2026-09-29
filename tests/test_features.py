import pandas as pd

from btc_quarter_hour_engine.acquisition import load_sample_market_data
from btc_quarter_hour_engine.features import build_feature_frame


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

    try:
        from btc_quarter_hour_engine.acquisition.loader import _normalize_market_frame

        _normalize_market_frame(data)
        raise AssertionError("Expected ValueError for non-1-minute cadence")
    except ValueError as exc:
        assert "strict 1-minute cadence" in str(exc)


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
        "price_vs_vwap_15m",
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
