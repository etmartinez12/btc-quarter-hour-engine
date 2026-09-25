from __future__ import annotations

import pandas as pd

from ..config import FeatureConfig
from ..features import build_feature_frame


class LivePredictor:
    """Minimal live inference adapter for future quarter-hour scheduling."""

    def __init__(self, model, feature_config: FeatureConfig | None = None) -> None:
        self.model = model
        self.feature_config = feature_config or FeatureConfig()

    def predict_latest_probability(self, market_frame: pd.DataFrame) -> float:
        features = build_feature_frame(market_frame, self.feature_config)
        latest_row = features.drop(columns=["timestamp"], errors="ignore").tail(1)
        probability_up = self.model.predict_proba(latest_row)[0, 1]
        return float(probability_up)
