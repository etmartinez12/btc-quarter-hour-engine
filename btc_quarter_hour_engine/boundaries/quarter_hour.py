from __future__ import annotations

import pandas as pd


BOUNDARY_FREQ = "15min"


def compute_midpoint(frame: pd.DataFrame, bid_column: str = "bid", ask_column: str = "ask") -> pd.Series:
    return (frame[bid_column] + frame[ask_column]) / 2.0


def is_quarter_hour_boundary(timestamp: pd.Series) -> pd.Series:
    return (
        timestamp.dt.minute.mod(15).eq(0)
        & timestamp.dt.second.eq(0)
        & timestamp.dt.microsecond.eq(0)
    )


def build_boundary_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Extract exact quarter-hour boundary prices.

    Canonical price definition:
    P_t = (best_bid_t + best_ask_t) / 2 at the exact quarter-hour timestamp.
    """

    market = frame.copy()
    market["timestamp"] = pd.to_datetime(market["timestamp"], utc=True)
    market["midpoint"] = compute_midpoint(market)
    mask = is_quarter_hour_boundary(market["timestamp"])
    boundaries = market.loc[mask, ["timestamp", "bid", "ask", "midpoint", "volume"]].copy()
    boundaries = boundaries.sort_values("timestamp").reset_index(drop=True)
    return boundaries
