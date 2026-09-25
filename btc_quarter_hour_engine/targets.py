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
    dataset = dataset.rename(columns={price_column: "price_t"})
    dataset["price_t_plus_15m"] = dataset["price_t"].shift(-1)
    dataset["price_change"] = dataset["price_t_plus_15m"] - dataset["price_t"]
    valid_prices = dataset["price_t"].gt(0) & dataset["price_t_plus_15m"].gt(0)
    dataset["log_return_15m"] = np.nan
    dataset.loc[valid_prices, "log_return_15m"] = np.log(
        dataset.loc[valid_prices, "price_t_plus_15m"] / dataset.loc[valid_prices, "price_t"]
    )
    dataset["target"] = pd.Series(pd.NA, index=dataset.index, dtype="Int64")
    dataset.loc[dataset["price_change"] > 0, "target"] = 1
    dataset.loc[dataset["price_change"] < 0, "target"] = 0
    return dataset
