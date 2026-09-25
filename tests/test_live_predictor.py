import pytest

from btc_quarter_hour_engine.acquisition import load_sample_market_data
from btc_quarter_hour_engine.live import LivePredictor


class DummyModel:
    def predict_proba(self, X):
        return [[0.4, 0.6] for _ in range(len(X))]


def test_live_predictor_rejects_non_boundary_latest_timestamp():
    raw = load_sample_market_data().iloc[:20].copy()
    predictor = LivePredictor(DummyModel())

    with pytest.raises(ValueError, match="exact quarter-hour boundary"):
        predictor.predict_latest_probability(raw)
