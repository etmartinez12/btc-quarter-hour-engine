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

    with pytest.raises(ValueError, match="duplicate timestamps"):
        predictor.predict_latest_probability(raw)


def test_live_predictor_rejects_missing_minute():
    raw = load_sample_market_data().iloc[:1441].copy()
    raw = raw.drop(raw.index[500]).reset_index(drop=True)
    assert raw["timestamp"].iloc[-1].minute % 15 == 0
    predictor = LivePredictor(DummyModel())

    with pytest.raises(ValueError, match="one-minute cadence"):
        predictor.predict_latest_probability(raw)


def test_live_predictor_normalizes_out_of_order_rows_before_finding_latest_timestamp():
    raw = load_sample_market_data().iloc[:1441].copy()
    reordered = raw.copy()
    tail = reordered.iloc[-20:].copy().reset_index(drop=True)
    reordered = pd.concat([reordered.iloc[:-20], tail.iloc[[3, 1, 2, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]], ignore_index=True)
    predictor = LivePredictor(DummyModel())

    actual = predictor.predict_latest_probability(reordered)

    assert actual == 0.6
