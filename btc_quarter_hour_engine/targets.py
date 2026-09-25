from __future__ import annotations

import numpy as np
import pandas as pd


def build_direction_target(boundary_prices: pd.DataFrame, price_column: str = "midpoint") -> pd.DataFrame:
    """Build quarter-hour labels from boundary t to boundary t+15m.

    The label uses only the canonical price observed at boundary t and the next exact
    quarter-hour canonical price. Flat moves are unlabeled because the research target
    is defined only for strictly up or strictly down moves.
    """

    dataset = boundary_prices[["timestamp", price_column]].copy()
    dataset = dataset.sort_values("timestamp").reset_index(drop=True)
    dataset = dataset.rename(columns={price_column: "price_t"})

    timestamps = pd.to_datetime(dataset["timestamp"], utc=True)
    deltas_seconds = timestamps.diff().dt.total_seconds()
    invalid_deltas = deltas_seconds.dropna()
    if not invalid_deltas.empty and not invalid_deltas.eq(15 * 60).all():
        bad_rows = invalid_deltas.index[~invalid_deltas.eq(15 * 60)].tolist()
        raise ValueError(
            "Boundary timestamps must be exactly 15 minutes apart; found non-15-minute gaps at rows "
            f"{bad_rows}."
        )

    dataset["price_t_plus_15m"] = dataset["price_t"].shift(-1)
    dataset["price_change"] = dataset["price_t_plus_15m"] - dataset["price_t"]
    valid_prices = dataset["price_t"].gt(0) & dataset["price_t_plus_15m"].gt(0)
    dataset["log_return_15m"] = np.nan
    dataset.loc[valid_prices, "log_return_15m"] = np.log(
        dataset.loc[valid_prices, "price_t_plus_15m"] / dataset.loc[valid_prices, "price_t"]
    )
    dataset["target"] = pd.Series(pd.NA, index=dataset.index, dtype="Int64")
    dataset.loc[valid_prices & dataset["price_change"].gt(0), "target"] = 1
    dataset.loc[valid_prices & dataset["price_change"].lt(0), "target"] = 0
    return dataset
