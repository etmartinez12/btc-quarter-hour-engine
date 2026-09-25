import pandas as pd
import pytest

from btc_quarter_hour_engine.acquisition import load_sample_market_data
from btc_quarter_hour_engine.live import LivePredictor


class DummyModel:
    def predict_proba(self, X):
        return [[0.4, 0.6] for _ in range(len(X))]


class BadShapeModel:
    def predict_proba(self, X):
        return [0.6 for _ in range(len(X))]


def test_live_predictor_rejects_non_boundary_latest_timestamp():
    raw = load_sample_market_data().iloc[:20].copy()
    predictor = LivePredictor(DummyModel())

    with pytest.raises(ValueError, match="exact quarter-hour boundary"):
        predictor.predict_latest_probability(raw)


def test_live_predictor_returns_probability_for_boundary_aligned_input():
    raw = load_sample_market_data().iloc[:1441].copy()
    predictor = LivePredictor(DummyModel())

    actual = predictor.predict_latest_probability(raw)

    assert actual == 0.6


def test_live_predictor_rejects_boundary_when_latest_features_are_not_usable():
    raw = load_sample_market_data().iloc[:16].copy()
    predictor = LivePredictor(DummyModel())

    with pytest.raises(ValueError, match="no usable feature row"):
        predictor.predict_latest_probability(raw)


def test_live_predictor_rejects_invalid_probability_shape():
    raw = load_sample_market_data().iloc[:1441].copy()
    predictor = LivePredictor(BadShapeModel())

    with pytest.raises(ValueError, match="2D array-like"):
        predictor.predict_latest_probability(raw)


def test_live_predictor_rejects_duplicate_latest_boundary_rows():
    raw = load_sample_market_data().iloc[:1441].copy()
    raw = pd.concat([raw, raw.iloc[[-1]]], ignore_index=True)
    predictor = LivePredictor(DummyModel())

    with pytest.raises(ValueError, match="multiple feature rows"):
        predictor.predict_latest_probability(raw)
