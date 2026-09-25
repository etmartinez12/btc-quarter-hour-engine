from __future__ import annotations

import numpy as np
import pandas as pd

from .boundaries import compute_midpoint, is_quarter_hour_boundary
from .config import FeatureConfig


def build_feature_frame(frame: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Create leakage-safe quarter-hour features.

    Every feature is computed from data available at or before boundary timestamp t.
    The returned rows are aligned only to exact quarter-hour boundary timestamps.
    """

    cfg = config or FeatureConfig()
    market = frame.copy().sort_values("timestamp").reset_index(drop=True)
    market["midpoint"] = compute_midpoint(market)
    market["log_midpoint"] = np.log(market["midpoint"])
    market["spread_bps"] = ((market["ask"] - market["bid"]) / market["midpoint"]) * 10_000.0
    market["return_1m"] = market["log_midpoint"].diff()

    for window in cfg.momentum_windows_minutes:
        market[f"return_{window}m"] = market["log_midpoint"].diff(window)

    for window in cfg.volatility_windows_minutes:
        market[f"realized_vol_{window}m"] = market["return_1m"].rolling(window=window, min_periods=window).std()

    for window in cfg.volume_windows_minutes:
        market[f"volume_{window}m"] = market["volume"].rolling(window=window, min_periods=window).sum()

    extrema_window = cfg.rolling_extrema_window_minutes
    rolling_high = market["midpoint"].rolling(window=extrema_window, min_periods=extrema_window).max()
    rolling_low = market["midpoint"].rolling(window=extrema_window, min_periods=extrema_window).min()
    market[f"distance_from_high_{extrema_window}m"] = market["midpoint"] / rolling_high - 1.0
    market[f"distance_from_low_{extrema_window}m"] = market["midpoint"] / rolling_low - 1.0

    minutes_of_day = market["timestamp"].dt.hour * 60 + market["timestamp"].dt.minute
    market["tod_sin"] = np.sin(2.0 * np.pi * minutes_of_day / (24 * 60))
    market["tod_cos"] = np.cos(2.0 * np.pi * minutes_of_day / (24 * 60))
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
