from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ("timestamp", "bid", "ask", "last_trade", "volume")


def _normalize_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"market data missing required columns: {missing}")

    normalized = frame.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True)
    normalized = normalized.sort_values("timestamp").drop_duplicates("timestamp")
    return normalized.reset_index(drop=True)


def load_market_data_csv(path: str | Path) -> pd.DataFrame:
    return _normalize_market_frame(pd.read_csv(path))


def load_sample_market_data() -> pd.DataFrame:
    sample_path = files("btc_quarter_hour_engine.data").joinpath("coinbase_btc_usd_1m_sample.csv")
    return _normalize_market_frame(pd.read_csv(sample_path))
