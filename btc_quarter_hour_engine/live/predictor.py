from __future__ import annotations

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
        latest_row = features.drop(columns=["timestamp"], errors="ignore").tail(1)
        probability_up = self.model.predict_proba(latest_row)[0, 1]
        return float(probability_up)
