from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("timestamp", "bid", "ask", "last_trade", "volume")
REQUIRED_NUMERIC_COLUMNS = ("bid", "ask", "last_trade", "volume")
OPTIONAL_MICROSTRUCTURE_COLUMNS = (
    "bid_size",
    "ask_size",
    "depth_bid_5",
    "depth_ask_5",
    "trade_count",
    "buy_volume",
    "sell_volume",
)
NONNEGATIVE_OPTIONAL_COLUMNS = (
    "bid_size",
    "ask_size",
    "depth_bid_5",
    "depth_ask_5",
    "trade_count",
    "buy_volume",
    "sell_volume",
)


def _validate_minute_cadence(frame: pd.DataFrame) -> pd.DataFrame:
    """Enforce the project contract for minute-bar market data.

    Every minute-based aggregate is trailing and end-stamped: the row at 13:00 represents
    the interval (12:59:00, 13:00:00]. Therefore the raw input must be a strict 1-minute grid.
    """

    normalized = frame.copy().sort_values("timestamp").reset_index(drop=True)
    if normalized.empty:
        return normalized

    timestamps = pd.to_datetime(normalized["timestamp"], utc=True)
    expected = pd.Timedelta(minutes=1)
    deltas = timestamps.diff().dropna()
    if not deltas.eq(expected).all():
        first_bad = deltas[~deltas.eq(expected)].iloc[0]
        index = deltas[~deltas.eq(expected)].index[0]
        previous_ts = timestamps.iloc[index - 1]
        current_ts = timestamps.iloc[index]
        raise ValueError(
            "Expected one-minute cadence but found:\n"
            f"previous: {previous_ts}\n"
            f"current:  {current_ts}\n"
            f"delta:    {current_ts - previous_ts}\n"
            f"expected: {expected}"
        )
    return normalized


def normalize_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"market data missing required columns: {missing}")

    normalized = frame.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True, errors="coerce")
    if normalized["timestamp"].isna().any():
        raise ValueError("market data contains invalid timestamps")

    duplicate_mask = normalized["timestamp"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_timestamps = normalized.loc[duplicate_mask, "timestamp"].drop_duplicates().tolist()
        raise ValueError(f"market data contains duplicate timestamps: {duplicate_timestamps[:10]}")

    normalized = normalized.sort_values("timestamp").reset_index(drop=True)

    numeric_columns = [
        *REQUIRED_NUMERIC_COLUMNS,
        *[column for column in OPTIONAL_MICROSTRUCTURE_COLUMNS if column in normalized.columns],
    ]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    required_missing = normalized[list(REQUIRED_NUMERIC_COLUMNS)].isna().any(axis=None)
    if required_missing:
        raise ValueError("market data contains missing or non-numeric required values")

    if normalized["bid"].le(0).any():
        raise ValueError("bid must be positive")
    if normalized["ask"].le(0).any():
        raise ValueError("ask must be positive")
    if normalized["last_trade"].le(0).any():
        raise ValueError("last_trade must be positive")
    if normalized["volume"].lt(0).any():
        raise ValueError("volume must be non-negative")
    if normalized["bid"].gt(normalized["ask"]).any():
        raise ValueError("market data contains crossed bid/ask quotes")

    required_numeric_values = normalized[list(REQUIRED_NUMERIC_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(required_numeric_values).all():
        raise ValueError("market data contains non-finite required values")

    optional_columns = [column for column in NONNEGATIVE_OPTIONAL_COLUMNS if column in normalized.columns]
    if optional_columns:
        optional_values = normalized[optional_columns].to_numpy(dtype=float)
        if not np.isfinite(optional_values).all():
            raise ValueError("market data contains non-finite optional microstructure values")
        if normalized[optional_columns].lt(0).any().any():
            raise ValueError("market data contains negative optional microstructure values")

    normalized = _validate_minute_cadence(normalized)
    return normalized


def _normalize_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return normalize_market_frame(frame)


def load_market_data_csv(path: str | Path) -> pd.DataFrame:
    return normalize_market_frame(pd.read_csv(path))


def load_sample_market_data() -> pd.DataFrame:
    sample_path = files("btc_quarter_hour_engine.data").joinpath("coinbase_btc_usd_1m_sample.csv")
    return normalize_market_frame(pd.read_csv(sample_path))
