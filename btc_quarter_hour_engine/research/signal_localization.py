from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from btc_quarter_hour_engine.research.diagnostics import _finite_or_none
from btc_quarter_hour_engine.research.feature_families_v2 import (
    FAMILY_CONTRACT_VERSION,
    V2_FEATURE_FAMILIES,
    localization_variants,
)
from btc_quarter_hour_engine.research.features_v2 import (
    V2_PREDICTOR_CONTRACT_VERSION,
)
from btc_quarter_hour_engine.research.walk_forward import (
    ExploratoryResearchConfig,
    _baseline_predictions,
    _fold_record,
    _model_factories,
    _model_fold_predictions,
    build_exploratory_folds,
    prepare_research_frame,
)
from btc_quarter_hour_engine.validation.metrics import classification_metrics


IDENTITY_COLUMNS = ["fold_number", "timestamp"]
POPULATION_COLUMNS = IDENTITY_COLUMNS + ["y_true", "target_log_return_15m"]
METRICS = ("accuracy", "roc_auc", "brier_score", "log_loss")
EXPERIMENT_COLUMNS = ["mode", "family", "benchmark"]


def localization_config() -> ExploratoryResearchConfig:
    return ExploratoryResearchConfig(
        predictor_contract_version=V2_PREDICTOR_CONTRACT_VERSION
    )


def _validate_input(features: pd.DataFrame) -> None:
    if features.columns.duplicated().any():
        raise ValueError("Duplicate input columns")
    required = {
        "timestamp", "label_available_time", "target", "target_log_return_15m",
        "trainable", "feature_ready", "target_eligible",
    }
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Missing localization input columns: {sorted(missing)}")
    for column in ("trainable", "feature_ready", "target_eligible"):
        if (
            not pd.api.types.is_bool_dtype(features[column])
            or features[column].isna().any()
        ):
            raise ValueError(f"{column} must contain non-null booleans")
    trainable = features.loc[features["trainable"]]
    if not trainable[["feature_ready", "target_eligible"]].all().all():
        raise ValueError("Trainable rows must be feature-ready and target-eligible")
    if not trainable["target"].isin([0, 1]).all():
        raise ValueError("Trainable targets must be exactly binary 0 or 1")
    for column in ("timestamp", "label_available_time"):
        values = pd.to_datetime(trainable[column], utc=True, errors="raise")
        if values.isna().any():
            raise ValueError(f"Trainable {column} contains missing timestamps")
    returns = pd.to_numeric(trainable["target_log_return_15m"], errors="raise")
    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise ValueError("Trainable target returns must be finite")


def validate_localization_predictions(
    predictions: pd.DataFrame,
    expected_population: pd.DataFrame,
) -> None:
    variants = localization_variants()
    models = set(_model_factories(localization_config()))
    baselines = {
        "always_up", "always_down", "train_majority", "causal_momentum_15m",
    }
    expected_experiments = {
        (mode, family, model)
        for mode, family in variants for model in models
    } | {("reference", "all", baseline) for baseline in baselines}
    required = {
        *EXPERIMENT_COLUMNS, *POPULATION_COLUMNS, "benchmark_type",
        "y_pred", "y_proba",
    }
    if not required.issubset(predictions.columns):
        raise ValueError("Missing OOF prediction columns")
    if predictions[EXPERIMENT_COLUMNS + POPULATION_COLUMNS].isna().any().any():
        raise ValueError("Missing OOF identifiers or targets")
    if expected_population.duplicated(IDENTITY_COLUMNS).any():
        raise ValueError("Duplicate expected OOF keys")
    if predictions.duplicated(EXPERIMENT_COLUMNS + IDENTITY_COLUMNS).any():
        raise ValueError("Duplicate OOF prediction keys")
    actual_experiments = set(
        predictions[EXPERIMENT_COLUMNS].itertuples(index=False, name=None)
    )
    if actual_experiments != expected_experiments:
        raise ValueError("Missing or unexpected OOF experiments/models")
    expected = expected_population[POPULATION_COLUMNS].sort_values(
        IDENTITY_COLUMNS
    ).reset_index(drop=True)
    for (_, _, benchmark), group in predictions.groupby(EXPERIMENT_COLUMNS):
        population = group[POPULATION_COLUMNS].sort_values(
            IDENTITY_COLUMNS
        ).reset_index(drop=True)
        if not population.equals(expected):
            raise ValueError(
                f"{benchmark} OOF fold population or targets differ from reference"
            )
        is_model = benchmark in models
        expected_type = "model" if is_model else "baseline"
        if not group["benchmark_type"].eq(expected_type).all():
            raise ValueError("Inconsistent OOF benchmark type")
        if not group["y_pred"].isin([0, 1]).all():
            raise ValueError("Missing or invalid OOF class predictions")
        if is_model:
            probabilities = group["y_proba"].to_numpy(dtype=float)
            if (
                not np.isfinite(probabilities).all()
                or ((probabilities < 0) | (probabilities > 1)).any()
            ):
                raise ValueError("OOF probabilities must be finite and within [0, 1]")
            if not np.array_equal(group["y_pred"], (probabilities >= 0.5).astype(int)):
                raise ValueError("OOF classes disagree with fixed 0.5 decision rule")
        elif not group["y_proba"].isna().all():
            raise ValueError("Hard-label baselines must not claim probabilities")


def _metrics(group: pd.DataFrame) -> dict[str, Any]:
    is_model = group["benchmark_type"].iloc[0] == "model"
    values = classification_metrics(
        group["y_true"], group["y_pred"],
        group["y_proba"] if is_model else None,
    )
    record: dict[str, Any] = {}
    for name in METRICS:
        record[name] = _finite_or_none(values.get(name))
        if record[name] is None:
            record[f"{name}_undefined_reason"] = (
                "hard_label_baseline_has_no_probability"
                if not is_model else "single_target_class_in_population"
            )
        else:
            record[f"{name}_undefined_reason"] = None
    return record


def localization_metrics(
    predictions: pd.DataFrame,
    expected_population: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_localization_predictions(predictions, expected_population)
    rows = []
    fold_rows = []
    for (mode, family, benchmark), group in predictions.groupby(
        EXPERIMENT_COLUMNS, sort=True
    ):
        identity = dict(zip(EXPERIMENT_COLUMNS, (mode, family, benchmark)))
        fold_accuracies = []
        for fold_number, fold in group.groupby("fold_number", sort=True):
            metrics = _metrics(fold)
            fold_accuracies.append(metrics["accuracy"])
            fold_rows.append({
                **identity, "fold_number": int(fold_number),
                "n_predictions": len(fold), **metrics,
            })
        accuracies = np.asarray(fold_accuracies, dtype=float)
        rows.append({
            **identity,
            "benchmark_type": group["benchmark_type"].iloc[0],
            "n_predictions": len(group),
            **_metrics(group),
            "n_folds": len(accuracies),
            "fold_accuracy_mean": float(accuracies.mean()),
            "fold_accuracy_std": float(accuracies.std(ddof=0)),
            "fold_accuracy_min": float(accuracies.min()),
            "fold_accuracy_max": float(accuracies.max()),
        })
    summary = pd.DataFrame(rows)
    per_fold = pd.DataFrame(fold_rows)
    for table, keys, names in (
        (summary, ["benchmark"], [
            *METRICS, "fold_accuracy_mean", "fold_accuracy_std",
            "fold_accuracy_min", "fold_accuracy_max",
        ]),
        (per_fold, ["benchmark", "fold_number"], list(METRICS)),
    ):
        reference = table.loc[table["mode"].eq("reference")].set_index(keys)
        for name in names:
            reference_values = table[keys].merge(
                reference[[name]].rename(columns={name: "reference_value"}),
                left_on=keys, right_index=True, how="left", validate="many_to_one",
            )["reference_value"]
            table[f"delta_{name}_vs_full_v2"] = (
                table[name].to_numpy(dtype=float)
                - reference_values.to_numpy(dtype=float)
            )
    return summary, per_fold


def run_signal_localization(
    features: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _validate_input(features)
    config = localization_config()
    frame, X, y = prepare_research_frame(
        features, V2_PREDICTOR_CONTRACT_VERSION
    )
    variants = localization_variants()
    folds = build_exploratory_folds(frame, config)
    population_parts = []
    fold_records = []
    parts = []
    for fold in folds:
        number = fold["fold_number"]
        train_idx, test_idx = fold["train_idx"], fold["test_idx"]
        train, test = frame.iloc[train_idx], frame.iloc[test_idx]
        fold_records.append(_fold_record(number, train, test))
        population_parts.append(pd.DataFrame({
            "fold_number": number,
            "timestamp": test["timestamp"].to_numpy(),
            "y_true": y.iloc[test_idx].to_numpy(dtype=int),
            "target_log_return_15m": test["target_log_return_15m"].to_numpy(),
        }))
        for (mode, family), columns in variants.items():
            subset = X.loc[:, list(columns)]
            for prediction in _model_fold_predictions(
                subset.iloc[train_idx], y.iloc[train_idx],
                subset.iloc[test_idx], y.iloc[test_idx],
                test, number, config,
            ):
                parts.append(prediction.assign(mode=mode, family=family))
        for name, labels in _baseline_predictions(train, test).items():
            parts.append(population_parts[-1].assign(
                benchmark=name, benchmark_type="baseline", y_pred=labels,
                y_proba=np.nan, mode="reference", family="all",
            ))
    population = pd.concat(population_parts, ignore_index=True)
    predictions = pd.concat(parts, ignore_index=True).sort_values(
        EXPERIMENT_COLUMNS + IDENTITY_COLUMNS, kind="stable"
    ).reset_index(drop=True)
    metrics, fold_metrics = localization_metrics(predictions, population)
    metadata = {
        "mode": "v2_signal_localization",
        "scientific_claim_status": (
            "development_exploratory_analysis_not_untouched_prospective_validation"
        ),
        "predictor_contract_version": V2_PREDICTOR_CONTRACT_VERSION,
        "predictor_count": X.shape[1],
        "family_contract_version": FAMILY_CONTRACT_VERSION,
        "feature_families": V2_FEATURE_FAMILIES,
        "variants": [
            {"mode": mode, "family": family, "predictors": columns,
             "predictor_count": len(columns)}
            for (mode, family), columns in variants.items()
        ],
        "validation": {
            **asdict(config), "expanding_training": True,
            "test_overlap_allowed": False,
            "strict_label_cutoff": "label_available_time < test_start",
            "fold_accuracy_std_ddof": 0,
            "delta_definition": "variant_minus_full_v2_on_identical_oof_rows",
        },
        "n_trainable_rows": len(frame),
        "n_oof_predictions_per_benchmark": len(population),
        "n_folds": len(folds),
        "folds": fold_records,
        "metrics": metrics.astype(object).where(metrics.notna(), None).to_dict(
            orient="records"
        ),
        "fold_metrics": fold_metrics.astype(object).where(
            fold_metrics.notna(), None
        ).to_dict(orient="records"),
    }
    return metadata, predictions, metrics, fold_metrics
