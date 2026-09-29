import json

import pandas as pd
import pytest

from btc_quarter_hour_engine.acquisition import load_sample_market_data, normalize_market_frame
from btc_quarter_hour_engine.boundaries import build_boundary_price_frame
from btc_quarter_hour_engine.config import BaselineConfig, ValidationConfig
from btc_quarter_hour_engine.run_baseline import _to_builtin, build_training_dataset, run_walk_forward_benchmark
from btc_quarter_hour_engine.validation.metrics import classification_metrics, pairwise_prediction_agreement
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


def test_dataset_has_label_availability_and_excludes_metadata_columns():
    dataset, feature_columns = build_training_dataset()

    pd.testing.assert_series_equal(
        dataset["label_available_time"],
        dataset["timestamp"] + pd.Timedelta(minutes=15),
        check_names=False,
    )
    assert "label_available_time" not in feature_columns
    assert "price_t_minus_15m" not in feature_columns


def test_dataset_uses_exact_previous_boundary_price_for_price_t_minus_15m():
    dataset, _ = build_training_dataset()
    raw = normalize_market_frame(load_sample_market_data())
    boundaries = build_boundary_price_frame(raw)

    for row in dataset.head(5).itertuples(index=False):
        expected_timestamp = row.timestamp - pd.Timedelta(minutes=15)
        expected = boundaries.loc[boundaries["timestamp"].eq(expected_timestamp), "midpoint"]
        assert not expected.empty
        assert row.price_t_minus_15m == pytest.approx(float(expected.iloc[0]))


def test_fold_level_label_availability_is_strictly_respected():
    summary = run_walk_forward_benchmark()

    for fold in summary["folds"]:
        if fold["train_end"] is None:
            continue
        train_end = pd.Timestamp(fold["train_end"])
        test_start = pd.Timestamp(fold["test_start"])
        assert train_end + pd.Timedelta(minutes=15) <= test_start


def test_run_walk_forward_benchmark_accepts_caller_supplied_market_data():
    market_data = load_sample_market_data()
    summary = run_walk_forward_benchmark(market_data=market_data)

    for key in ["models", "ensembles", "naive_baselines", "benchmark_comparison"]:
        assert key in summary


def test_summary_reports_demo_validation_protocol_metadata():
    summary = run_walk_forward_benchmark()

    assert summary["validation_protocol"]["mode"] == "demo_row_based"
    assert summary["validation_protocol"]["label_availability_enforced"] is True
    assert summary["validation_protocol"]["test_overlap_allowed"] is False
    assert "research_validation_spec" in summary
    for key in ["initial_train_period", "test_period", "step_period", "expanding_training", "test_overlap_allowed"]:
        assert key in summary["research_validation_spec"]


def test_pairwise_prediction_agreement_uses_composite_oof_keys():
    left = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "2026-01-01T00:15:00Z", "2026-01-01T00:15:00Z"], utc=True),
            "fold_number": [0, 1, 0, 1],
            "y_pred": [1, 0, 0, 1],
            "y_proba": [0.9, 0.2, 0.3, 0.8],
        }
    )
    right = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "2026-01-01T00:15:00Z", "2026-01-01T00:15:00Z"], utc=True),
            "fold_number": [0, 1, 0, 1],
            "y_pred": [1, 1, 0, 0],
            "y_proba": [0.8, 0.7, 0.1, 0.9],
        }
    )

    agreement = pairwise_prediction_agreement({"model_a": left, "model_b": right})[0]
    timestamp_only_count = len(left.merge(right, on=["timestamp"], how="inner"))

    assert agreement["n_common_predictions"] == 4
    assert timestamp_only_count != agreement["n_common_predictions"]
