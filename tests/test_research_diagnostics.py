import json

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from btc_quarter_hour_engine.models import HistGradientBoostingBaselineModel
from btc_quarter_hour_engine.research.diagnostics import (
    calibration_diagnostics,
    feature_drift_diagnostics,
)
from btc_quarter_hour_engine.research.diagnostics_cli import run_diagnostics
from btc_quarter_hour_engine.research.features import (
    PREDICTOR_COLUMNS,
    PREDICTOR_CONTRACT_VERSION,
)
from btc_quarter_hour_engine.research.walk_forward import (
    ExploratoryResearchConfig,
    prepare_research_frame,
    run_exploratory_walk_forward,
)


def _synthetic_research_frame(periods=500):
    timestamps = pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=periods,
        freq="15min",
    )
    data = {}
    for idx, column in enumerate(PREDICTOR_COLUMNS):
        values = np.arange(periods, dtype=float) + idx + 1.0
        if column == "asof_book_synced":
            values = np.ones(periods, dtype=float)
        if column.startswith("momentum_log_return_"):
            values = np.sin(np.arange(periods, dtype=float) / 7.0)
        data[column] = values

    frame = pd.DataFrame(data)
    frame["timestamp"] = timestamps
    frame["label_available_time"] = timestamps + pd.Timedelta(minutes=15)
    frame["target"] = (np.arange(periods) % 2).astype(int)
    frame["target_log_return_15m"] = np.where(
        frame["target"].eq(1), 0.001, -0.001
    )
    frame["trainable"] = True
    frame["feature_ready"] = True
    frame["target_eligible"] = True
    frame["canonical_eligibility_reason"] = "eligible"
    frame["target_eligibility_reason"] = "eligible"
    return frame


def _config():
    return ExploratoryResearchConfig(
        initial_train_period="72h",
        test_period="12h",
        step_period="12h",
        extra_trees_n_estimators=10,
    )


def test_hist_gradient_boosting_wrapper_emits_valid_probabilities():
    frame = _synthetic_research_frame(100)
    _, X, y = prepare_research_frame(frame)
    model = HistGradientBoostingBaselineModel().fit(X, y)

    probabilities = model.predict_proba(X.iloc[:10])

    assert model.model.get_params()["early_stopping"] is False
    assert probabilities.shape == (10, 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
    assert len(model.predict(X.iloc[:10])) == 10
    assert list(model.model.feature_names_in_) == list(PREDICTOR_COLUMNS)


def test_walk_forward_preserves_contract_and_common_probabilistic_oof():
    features = _synthetic_research_frame()
    summary, predictions = run_exploratory_walk_forward(features, _config())

    assert summary["predictor_contract_version"] == "bbo-causal-v1"
    assert summary["predictor_count"] == 114
    assert {
        key: summary["validation"][key]
        for key in (
            "hist_gradient_boosting_max_iter",
            "hist_gradient_boosting_learning_rate",
            "hist_gradient_boosting_max_leaf_nodes",
            "hist_gradient_boosting_min_samples_leaf",
            "hist_gradient_boosting_l2_regularization",
        )
    } == {
        "hist_gradient_boosting_max_iter": 100,
        "hist_gradient_boosting_learning_rate": 0.1,
        "hist_gradient_boosting_max_leaf_nodes": 15,
        "hist_gradient_boosting_min_samples_leaf": 20,
        "hist_gradient_boosting_l2_regularization": 1.0,
    }
    assert PREDICTOR_CONTRACT_VERSION == "bbo-causal-v1"
    expected_benchmarks = {
        "always_up",
        "always_down",
        "train_majority",
        "causal_momentum_15m",
        "logistic_regression",
        "extra_trees",
        "hist_gradient_boosting",
    }
    assert expected_benchmarks.issubset(summary["benchmarks"])

    probabilistic = predictions.loc[
        predictions["benchmark_type"].eq("model")
    ]
    timestamps = {
        name: set(group["timestamp"])
        for name, group in probabilistic.groupby("benchmark")
    }
    assert set(timestamps) == {
        "logistic_regression",
        "extra_trees",
        "hist_gradient_boosting",
    }
    assert len({frozenset(values) for values in timestamps.values()}) == 1

    _, X, _ = prepare_research_frame(features)
    assert list(X.columns) == list(PREDICTOR_COLUMNS)
    assert not {"target", "target_log_return_15m", "label_available_time"}.intersection(
        X.columns
    )


def test_calibration_bins_are_deterministic_and_cover_all_predictions():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    probabilities = np.array([0.0, 0.1, 0.5, 0.9, 1.0, 0.55])

    first = calibration_diagnostics(y_true, probabilities)
    second = calibration_diagnostics(y_true, probabilities)

    assert first == second
    assert sum(row["n_predictions"] for row in first["bins"]) == len(y_true)
    assert len(first["bins"]) == 10
    assert all(0.0 <= row["lower_probability_bound"] <= 1.0 for row in first["bins"])
    assert all(0.0 <= row["upper_probability_bound"] <= 1.0 for row in first["bins"])
    assert all(
        row["mean_predicted_probability"] is None
        or 0.0 <= row["mean_predicted_probability"] <= 1.0
        for row in first["bins"]
    )
    assert first["probability_min"] == 0.0
    assert first["probability_max"] == 1.0
    assert set(first["confidence_fractions"]) == {
        ">=0.55",
        ">=0.60",
        ">=0.65",
        ">=0.70",
        ">=0.75",
        ">=0.80",
    }


def test_feature_drift_is_chronological_complete_and_finite_for_constants():
    features = _synthetic_research_frame(20)
    metadata, drift = feature_drift_diagnostics(features)

    assert metadata["split_rule"].startswith("Sort trainable rows by timestamp")
    assert metadata["early_end"] < metadata["late_start"]
    assert len(drift) == 114
    assert set(drift["feature_name"]) == set(PREDICTOR_COLUMNS)
    assert drift.loc[
        drift["feature_name"].eq("asof_book_synced"),
        "absolute_drift_score",
    ].iloc[0] == 0.0
    numeric = drift.select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy(dtype=float)).all()


def test_per_fold_metrics_match_oof_rows_and_agreement_uses_common_population():
    summary, predictions = run_exploratory_walk_forward(
        _synthetic_research_frame(), _config()
    )
    record = next(
        item
        for item in summary["fold_metrics"]
        if item["benchmark"] == "hist_gradient_boosting"
    )
    matching = predictions.loc[
        predictions["benchmark"].eq(record["benchmark"])
        & predictions["fold_number"].eq(record["fold_number"])
    ]

    assert record["n_test"] == len(matching)
    assert record["test_up_rate"] == matching["y_true"].mean()
    assert record["accuracy"] == (
        matching["y_true"].to_numpy() == matching["y_pred"].to_numpy()
    ).mean()
    assert record["strict_label_embargo_ok"]
    assert len(summary["model_agreement"]) == 3
    assert {
        row["n_common_predictions"] for row in summary["model_agreement"]
    } == {summary["n_oof_predictions_per_benchmark"]}
    assert {row["n_common_folds"] for row in summary["model_agreement"]} == {
        summary["n_folds"]
    }


def test_diagnostics_cli_writes_stable_artifact_schemas(tmp_path):
    features_path = tmp_path / "research_features.parquet"
    _synthetic_research_frame().to_parquet(features_path, index=False)
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    first_paths = run_diagnostics(features_path, first_root, _config())
    second_paths = run_diagnostics(features_path, second_root, _config())

    assert set(first_paths) == {
        "diagnostics",
        "oof_predictions",
        "feature_drift",
        "fold_metrics",
    }
    assert all(path.exists() for path in first_paths.values())
    assert first_paths["diagnostics"].read_bytes() == second_paths[
        "diagnostics"
    ].read_bytes()
    for artifact in ("oof_predictions", "feature_drift", "fold_metrics"):
        first = pd.read_parquet(first_paths[artifact])
        second = pd.read_parquet(second_paths[artifact])
        assert_frame_equal(first, second)

    diagnostics = json.loads(first_paths["diagnostics"].read_text())
    assert diagnostics["predictor_contract_version"] == "bbo-causal-v1"
    assert diagnostics["predictor_count"] == 114
    assert diagnostics["scientific_claim_status"] == (
        "development_exploratory_analysis_not_untouched_prospective_validation"
    )
    assert len(pd.read_parquet(first_paths["feature_drift"])) == 114
    assert {
        "roc_auc",
        "brier_score",
        "log_loss",
        "test_start",
        "test_end",
    }.issubset(pd.read_parquet(first_paths["fold_metrics"]).columns)
