from btc_quarter_hour_engine.acquisition import load_sample_market_data
from btc_quarter_hour_engine.features import build_feature_frame


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
    }

    assert expected_columns.issubset(features.columns)
    assert features["timestamp"].dt.minute.mod(15).eq(0).all()
    assert features["timestamp"].dt.second.eq(0).all()
