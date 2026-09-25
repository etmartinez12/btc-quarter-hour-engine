from __future__ import annotations

import numpy as np
import pandas as pd

from .boundaries import compute_midpoint, is_quarter_hour_boundary
from .config import FeatureConfig


MICROSTRUCTURE_COLUMNS = {
    "bid_size",
    "ask_size",
    "depth_bid_5",
    "depth_ask_5",
    "trade_count",
    "buy_volume",
    "sell_volume",
}


def build_feature_frame(frame: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Create leakage-safe quarter-hour features.

    Every feature is computed from data available at or before boundary timestamp t.
    The returned rows are aligned only to exact quarter-hour boundary timestamps.
    """

    cfg = config or FeatureConfig()
    market = frame.copy().sort_values("timestamp").reset_index(drop=True)
    market["midpoint"] = compute_midpoint(market)
    valid_midpoint = market["midpoint"].gt(0)
    market["log_midpoint"] = np.nan
    market.loc[valid_midpoint, "log_midpoint"] = np.log(market.loc[valid_midpoint, "midpoint"])
    market["spread_bps"] = np.nan
    market.loc[valid_midpoint, "spread_bps"] = (
        (market.loc[valid_midpoint, "ask"] - market.loc[valid_midpoint, "bid"])
        / market.loc[valid_midpoint, "midpoint"]
    ) * 10_000.0
    market["return_1m"] = market["log_midpoint"].diff()
    market["price_acceleration_1m"] = market["return_1m"].diff()

    for window in cfg.momentum_windows_minutes:
        market[f"return_{window}m"] = market["log_midpoint"].diff(window)

    for window in cfg.volatility_windows_minutes:
        market[f"realized_vol_{window}m"] = market["return_1m"].rolling(window=window, min_periods=window).std()

    for window in cfg.volume_windows_minutes:
        market[f"volume_{window}m"] = market["volume"].rolling(window=window, min_periods=window).sum()

    for window in cfg.range_windows_minutes:
        rolling_high = market["midpoint"].rolling(window=window, min_periods=window).max()
        rolling_low = market["midpoint"].rolling(window=window, min_periods=window).min()
        market[f"range_{window}m"] = rolling_high / rolling_low - 1.0
        market[f"distance_from_high_{window}m"] = market["midpoint"] / rolling_high - 1.0
        market[f"distance_from_low_{window}m"] = market["midpoint"] / rolling_low - 1.0

    for window in cfg.vwap_windows_minutes:
        rolling_notional = (market["midpoint"] * market["volume"]).rolling(window=window, min_periods=window).sum()
        rolling_volume = market["volume"].rolling(window=window, min_periods=window).sum()
        vwap = rolling_notional / rolling_volume
        market[f"price_vs_vwap_{window}m"] = market["midpoint"] / vwap - 1.0

    volume_zscore_window = cfg.volume_zscore_window_minutes
    volume_mean = market["volume"].rolling(window=volume_zscore_window, min_periods=volume_zscore_window).mean()
    volume_std = market["volume"].rolling(window=volume_zscore_window, min_periods=volume_zscore_window).std()
    market[f"volume_zscore_{volume_zscore_window}m"] = (market["volume"] - volume_mean) / volume_std

    ema_fast = market["midpoint"].ewm(span=cfg.ema_fast_window_minutes, adjust=False, min_periods=cfg.ema_fast_window_minutes).mean()
    ema_slow = market["midpoint"].ewm(span=cfg.ema_slow_window_minutes, adjust=False, min_periods=cfg.ema_slow_window_minutes).mean()
    market[f"price_vs_ema_{cfg.ema_fast_window_minutes}m"] = market["midpoint"] / ema_fast - 1.0
    market[f"price_vs_ema_{cfg.ema_slow_window_minutes}m"] = market["midpoint"] / ema_slow - 1.0
    market[f"ema_spread_{cfg.ema_fast_window_minutes}m_{cfg.ema_slow_window_minutes}m"] = ema_fast / ema_slow - 1.0

    extrema_window = cfg.rolling_extrema_window_minutes
    rolling_high = market["midpoint"].rolling(window=extrema_window, min_periods=extrema_window).max()
    rolling_low = market["midpoint"].rolling(window=extrema_window, min_periods=extrema_window).min()
    market[f"rolling_extrema_distance_from_high_{extrema_window}m"] = market["midpoint"] / rolling_high - 1.0
    market[f"rolling_extrema_distance_from_low_{extrema_window}m"] = market["midpoint"] / rolling_low - 1.0

    for window in cfg.regime_windows_minutes:
        if f"return_{window}m" not in market.columns:
            market[f"return_{window}m"] = market["log_midpoint"].diff(window)
        realized_vol = market["return_1m"].rolling(window=window, min_periods=window).std()
        market[f"regime_trend_{window}m"] = market[f"return_{window}m"]
        market[f"regime_vol_{window}m"] = realized_vol

    market["momentum_5m_minus_30m"] = market["return_5m"] - market["return_30m"]
    market["momentum_15m_minus_60m"] = market["return_15m"] - market["return_60m"]
    market["volume_5m_to_30m_ratio"] = market["volume_5m"] / market["volume_30m"]
    market["vol_regime_ratio_15m_to_60m"] = market["realized_vol_15m"] / market["realized_vol_60m"]

    if MICROSTRUCTURE_COLUMNS.issubset(market.columns):
        book_size_total = market["bid_size"] + market["ask_size"]
        depth_total = market["depth_bid_5"] + market["depth_ask_5"]
        flow_total = market["buy_volume"] + market["sell_volume"]

        market["spread_change_1m"] = market["spread_bps"].diff()
        market["order_book_imbalance"] = (market["bid_size"] - market["ask_size"]) / book_size_total
        market["depth_imbalance_5"] = (market["depth_bid_5"] - market["depth_ask_5"]) / depth_total
        market["quote_pressure"] = market["bid_size"] / market["ask_size"] - 1.0
        market["buy_sell_volume_imbalance_1m"] = (market["buy_volume"] - market["sell_volume"]) / flow_total
        market["aggressive_buy_share_1m"] = market["buy_volume"] / flow_total
        market["aggressive_sell_share_1m"] = market["sell_volume"] / flow_total
        market["average_trade_size_1m"] = flow_total / market["trade_count"]

        for window in cfg.microstructure_windows_minutes:
            market[f"trade_count_{window}m"] = market["trade_count"].rolling(window=window, min_periods=window).sum()
            market[f"buy_volume_{window}m"] = market["buy_volume"].rolling(window=window, min_periods=window).sum()
            market[f"sell_volume_{window}m"] = market["sell_volume"].rolling(window=window, min_periods=window).sum()
            rolling_flow_total = market[f"buy_volume_{window}m"] + market[f"sell_volume_{window}m"]
            market[f"trade_flow_imbalance_{window}m"] = (
                market[f"buy_volume_{window}m"] - market[f"sell_volume_{window}m"]
            ) / rolling_flow_total
            market[f"average_trade_size_{window}m"] = rolling_flow_total / market[f"trade_count_{window}m"]
            market[f"order_book_imbalance_mean_{window}m"] = market["order_book_imbalance"].rolling(
                window=window, min_periods=window
            ).mean()
            market[f"depth_imbalance_mean_{window}m"] = market["depth_imbalance_5"].rolling(
                window=window, min_periods=window
            ).mean()

    market = market.replace([np.inf, -np.inf], np.nan)

    seconds_of_day = (
        market["timestamp"].dt.hour * 3600
        + market["timestamp"].dt.minute * 60
        + market["timestamp"].dt.second
    )
    market["tod_sin"] = np.sin(2.0 * np.pi * seconds_of_day / (24 * 60 * 60))
    market["tod_cos"] = np.cos(2.0 * np.pi * seconds_of_day / (24 * 60 * 60))
    market["dow_sin"] = np.sin(2.0 * np.pi * market["timestamp"].dt.dayofweek / 7.0)
    market["dow_cos"] = np.cos(2.0 * np.pi * market["timestamp"].dt.dayofweek / 7.0)
    market["is_weekend"] = market["timestamp"].dt.dayofweek.ge(5).astype(int)

    feature_columns = [
        column
        for column in market.columns
        if column
        not in {"bid", "ask", "last_trade", "volume", "log_midpoint"}
    ]
    features = market.loc[is_quarter_hour_boundary(market["timestamp"]), feature_columns].copy()
    return features.reset_index(drop=True)
