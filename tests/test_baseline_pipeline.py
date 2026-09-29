import json

import pandas as pd
import pytest

from btc_quarter_hour_engine.config import BaselineConfig, ValidationConfig
from btc_quarter_hour_engine.run_baseline import _to_builtin, run_walk_forward_benchmark
from btc_quarter_hour_engine.validation.metrics import classification_metrics
from btc_quarter_hour_engine.validation.naive_baselines import previous_quarter_direction


def test_run_walk_forward_benchmark_raises_when_no_splits_are_possible():
    config = BaselineConfig(
        validation=ValidationConfig(initial_train_size=10_000, test_size=16, step_size=16)
    )

    with pytest.raises(ValueError, match="zero folds"):
        run_walk_forward_benchmark(config)


def test_walk_forward_benchmark_summary_is_json_serializable():
    summary = run_walk_forward_benchmark()

    serialized = json.dumps(_to_builtin(summary))

    assert "\"models\"" in serialized
    assert "\"ensembles\"" in serialized


def test_classification_metrics_handles_single_class_test_fold():
    metrics = classification_metrics([1, 1], [1, 1], [0.8, 0.9])

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["brier_score"] >= 0.0
    assert metrics["log_loss"] != metrics["log_loss"]
    assert metrics["roc_auc"] != metrics["roc_auc"]


def test_previous_quarter_direction_uses_exact_timestamp_history():
    frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01 00:00:00Z"), pd.Timestamp("2024-01-01 00:15:00Z")],
            "midpoint": [100.0, 110.0],
            "price_t_minus_15m": [95.0, 100.0],
        }
    )

    predictions = previous_quarter_direction(frame)

    assert predictions.tolist() == [1, 1]

    missing_history = frame.drop(columns=["price_t_minus_15m"])
    with pytest.raises(KeyError, match="exact price_t_minus_15m"):
        previous_quarter_direction(missing_history)


def test_walk_forward_benchmark_includes_phase_two_diagnostics():
    summary = run_walk_forward_benchmark()
    model_summary = summary["models"]["logistic_regression"]
    weighted_average = summary["ensembles"]["validation_weighted_average"]

    assert "feature_family_counts" in summary
    assert summary["feature_family_counts"]["regime"] > 0
    assert summary["feature_family_counts"]["microstructure"] > 0
    assert {"logistic_regression", "lightgbm", "extra_trees", "xgboost"} <= set(summary["models"])
    assert len(model_summary["confidence_accuracy"]) >= 2
    assert len(model_summary["accuracy_by_move_size"]) == 6
    assert len(model_summary["accuracy_by_boundary_slot"]) == 4
    assert len(summary["model_comparison"]["pairwise_prediction_agreement"]) == 6
    assert {"simple_average", "validation_weighted_average"} <= set(summary["ensembles"])
    assert len(weighted_average["weight_history"]) > 0
    assert len(summary["model_comparison"]["ensemble_consensus"]["accuracy_by_votes_up"]) == 5


def test_benchmark_comparison_uses_common_oof_population():
    summary = run_walk_forward_benchmark()

    common_n = summary["n_common_oof_predictions"]
    assert all(row["n_predictions"] == common_n for row in summary["benchmark_comparison"])
