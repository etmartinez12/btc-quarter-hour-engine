from __future__ import annotations

import numpy as np
import pandas as pd

from ..boundaries import is_quarter_hour_boundary
from ..config import FeatureConfig
from ..features import build_feature_frame


class LivePredictor:
    """Minimal live inference adapter for future quarter-hour scheduling."""

    def __init__(self, model, feature_config: FeatureConfig | None = None) -> None:
        self.model = model
        self.feature_config = feature_config or FeatureConfig()

    def predict_latest_probability(self, market_frame: pd.DataFrame) -> float:
        if market_frame.empty:
            raise ValueError("market_frame must contain at least one row")

        latest_timestamp = pd.to_datetime(market_frame["timestamp"], utc=True).iloc[-1]
        boundary_check = pd.Series([latest_timestamp])
        if not is_quarter_hour_boundary(boundary_check).iloc[0]:
            raise ValueError(
                "latest market timestamp must be an exact quarter-hour boundary to avoid stale predictions"
            )

        features = build_feature_frame(market_frame, self.feature_config)
        latest_row = features.loc[features["timestamp"].eq(latest_timestamp)].drop(
            columns=["timestamp"], errors="ignore"
        )
        if latest_row.empty or latest_row.isna().any(axis=None):
            raise ValueError("no usable feature row is available for the latest quarter-hour boundary")

        probabilities = np.asarray(self.model.predict_proba(latest_row))
        if probabilities.ndim != 2 or probabilities.shape[0] == 0 or probabilities.shape[1] < 2:
            raise ValueError("predict_proba must return a 2D array-like with class probabilities")

        probability_up = probabilities[0][1]
        return float(probability_up)
