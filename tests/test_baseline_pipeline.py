import json

import pytest

from btc_quarter_hour_engine.config import BaselineConfig, ValidationConfig
from btc_quarter_hour_engine.run_baseline import _to_builtin, run_walk_forward_benchmark
from btc_quarter_hour_engine.validation.metrics import classification_metrics


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


def test_classification_metrics_handles_single_class_test_fold():
    metrics = classification_metrics([1, 1], [1, 1], [0.8, 0.9])

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["brier_score"] >= 0.0
    assert metrics["log_loss"] != metrics["log_loss"]
    assert metrics["roc_auc"] != metrics["roc_auc"]


def test_walk_forward_benchmark_includes_phase_two_diagnostics():
    summary = run_walk_forward_benchmark()
    model_summary = summary["models"]["logistic_regression"]

    assert "feature_family_counts" in summary
    assert summary["feature_family_counts"]["regime"] > 0
    assert summary["feature_family_counts"]["microstructure"] > 0
    assert {"logistic_regression", "lightgbm", "extra_trees", "xgboost"} <= set(summary["models"])
    assert len(model_summary["confidence_accuracy"]) >= 2
    assert len(model_summary["accuracy_by_move_size"]) == 6
    assert len(model_summary["accuracy_by_boundary_slot"]) == 4
    assert len(summary["model_comparison"]["pairwise_prediction_agreement"]) == 6
