from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from btc_quarter_hour_engine.research.features import (
    PREDICTOR_COLUMNS,
    PREDICTOR_CONTRACT_VERSION,
    predictor_columns_for_contract,
    research_features_for_contract,
)
from btc_quarter_hour_engine.validation.metrics import (
    classification_metrics,
    pairwise_prediction_agreement,
)


CALIBRATION_BIN_EDGES = tuple(np.linspace(0.0, 1.0, 11))
CONFIDENCE_THRESHOLDS = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
TEMPORAL_GROUP_MIN_ROWS = 8


def calibration_diagnostics(
    y_true,
    y_proba,
) -> dict[str, Any]:
    actual = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(y_proba, dtype=float)

    if actual.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("Calibration inputs must be one-dimensional")
    if len(actual) != len(probabilities):
        raise ValueError("Calibration inputs must have equal lengths")
    if not np.isfinite(probabilities).all():
        raise ValueError("Calibration probabilities must be finite")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("Calibration probabilities must be within [0, 1]")

    confidence = np.maximum(probabilities, 1.0 - probabilities)
    bins: list[dict[str, Any]] = []
    for index, (lower, upper) in enumerate(
        zip(CALIBRATION_BIN_EDGES[:-1], CALIBRATION_BIN_EDGES[1:])
    ):
        if index == len(CALIBRATION_BIN_EDGES) - 2:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)

        count = int(mask.sum())
        mean_probability = (
            float(probabilities[mask].mean()) if count else None
        )
        observed_rate = float(actual[mask].mean()) if count else None
        bins.append(
            {
                "lower_probability_bound": float(lower),
                "upper_probability_bound": float(upper),
                "n_predictions": count,
                "mean_predicted_probability": mean_probability,
                "observed_up_rate": observed_rate,
                "calibration_error": (
                    observed_rate - mean_probability if count else None
                ),
            }
        )

    count = int(len(probabilities))
    return {
        "n_predictions": count,
        "probability_mean": float(probabilities.mean()) if count else None,
        "probability_std": (
            float(probabilities.std(ddof=0)) if count else None
        ),
        "probability_min": float(probabilities.min()) if count else None,
        "probability_max": float(probabilities.max()) if count else None,
        "confidence_fractions": {
            f">={threshold:.2f}": (
                float((confidence >= threshold).mean()) if count else None
            )
            for threshold in CONFIDENCE_THRESHOLDS
        },
        "bins": bins,
    }


def _finite_or_none(value: Any) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _group_performance(
    frame: pd.DataFrame,
    group_column: str,
    *,
    minimum_rows: int = 1,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_column, sort=True, dropna=False):
        if len(group) < minimum_rows:
            continue
        rows.append(
            {
                group_column: str(key),
                "n_predictions": int(len(group)),
                "up_rate": float(group["y_true"].mean()),
                "accuracy": float(
                    (group["y_true"] == group["y_pred"]).mean()
                ),
            }
        )
    return rows


def temporal_performance_diagnostics(
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    frame = predictions.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["utc_hour"] = frame["timestamp"].dt.hour
    frame["utc_date"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
    frame["boundary_slot"] = frame["timestamp"].dt.minute.map(
        {0: ":00-:15", 15: ":15-:30", 30: ":30-:45", 45: ":45-:00"}
    )
    magnitude = np.abs(
        np.exp(frame["target_log_return_15m"].to_numpy(dtype=float)) - 1.0
    )
    frame["move_size_bucket"] = pd.cut(
        magnitude,
        bins=(0.0, 0.0001, 0.0005, 0.001, 0.0025, 0.005, np.inf),
        labels=(
            "<0.01%",
            "0.01-0.05%",
            "0.05-0.10%",
            "0.10-0.25%",
            "0.25-0.50%",
            ">0.50%",
        ),
        include_lowest=True,
        right=False,
    ).astype("string")

    return {
        "minimum_rows_per_utc_date": TEMPORAL_GROUP_MIN_ROWS,
        "by_boundary_slot": _group_performance(frame, "boundary_slot"),
        "by_utc_hour": _group_performance(frame, "utc_hour"),
        "by_utc_date": _group_performance(
            frame,
            "utc_date",
            minimum_rows=TEMPORAL_GROUP_MIN_ROWS,
        ),
        "by_move_size": _group_performance(
            frame,
            "move_size_bucket",
        ),
    }


def fold_metrics_and_model_diagnostics(
    predictions: pd.DataFrame,
    fold_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    fold_metadata = {
        int(record["fold_number"]): record for record in fold_records
    }
    fold_rows: list[dict[str, Any]] = []
    benchmark_diagnostics: dict[str, Any] = {}
    model_predictions: dict[str, pd.DataFrame] = {}

    for benchmark, group in predictions.groupby("benchmark", sort=True):
        benchmark_type = str(group["benchmark_type"].iloc[0])
        group = group.sort_values(
            ["fold_number", "timestamp"], kind="stable"
        ).reset_index(drop=True)

        if benchmark_type == "model":
            model_predictions[str(benchmark)] = group
            calibration = calibration_diagnostics(
                group["y_true"], group["y_proba"]
            )
        else:
            calibration = None

        fold_accuracies: list[float] = []
        for fold_number, fold_group in group.groupby(
            "fold_number", sort=True
        ):
            fold_number_int = int(fold_number)
            metadata = fold_metadata[fold_number_int]
            y_proba = (
                fold_group["y_proba"]
                if benchmark_type == "model"
                else None
            )
            metrics = classification_metrics(
                fold_group["y_true"],
                fold_group["y_pred"],
                y_proba,
            )
            fold_accuracies.append(float(metrics["accuracy"]))
            record = {
                "benchmark": str(benchmark),
                "benchmark_type": benchmark_type,
                "fold_number": fold_number_int,
                "train_start": metadata["train_start"],
                "train_end": metadata["train_end"],
                "test_start": metadata["test_start"],
                "test_end": metadata["test_end"],
                "n_train": int(metadata["n_train"]),
                "n_test": int(len(fold_group)),
                "train_up_rate": float(metadata["train_up_rate"]),
                "test_up_rate": float(fold_group["y_true"].mean()),
                "strict_label_embargo_ok": bool(
                    metadata["strict_label_embargo_ok"]
                ),
            }
            record.update(
                {name: _finite_or_none(value) for name, value in metrics.items()}
            )
            fold_rows.append(record)

        accuracy_values = np.asarray(fold_accuracies, dtype=float)
        benchmark_diagnostics[str(benchmark)] = {
            "calibration": calibration,
            "temporal_performance": temporal_performance_diagnostics(group),
            "fold_stability": {
                "n_folds": int(len(accuracy_values)),
                "accuracy_mean": (
                    float(accuracy_values.mean())
                    if len(accuracy_values)
                    else None
                ),
                "accuracy_std": (
                    float(accuracy_values.std(ddof=0))
                    if len(accuracy_values)
                    else None
                ),
                "accuracy_min": (
                    float(accuracy_values.min())
                    if len(accuracy_values)
                    else None
                ),
                "accuracy_max": (
                    float(accuracy_values.max())
                    if len(accuracy_values)
                    else None
                ),
            },
        }

    agreement_rows = pairwise_prediction_agreement(model_predictions)
    for row in agreement_rows:
        row["n_common_folds"] = int(
            predictions.loc[
                predictions["benchmark"].isin(
                    [row["left_model"], row["right_model"]]
                ),
                "fold_number",
            ].nunique()
        )

    return fold_rows, benchmark_diagnostics, agreement_rows


def _finite_values(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return numeric[np.isfinite(numeric)]


def _distribution(values: pd.Series) -> dict[str, Any]:
    finite = _finite_values(values)
    if not len(finite):
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "iqr": None,
            "min": None,
            "max": None,
        }
    q25, median, q75 = np.quantile(finite, [0.25, 0.5, 0.75])
    return {
        "count": int(len(finite)),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
        "median": float(median),
        "iqr": float(q75 - q25),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def feature_drift_diagnostics(
    features: pd.DataFrame,
    *,
    predictor_contract_version: str = PREDICTOR_CONTRACT_VERSION,
) -> tuple[dict[str, Any], pd.DataFrame]:
    features = research_features_for_contract(
        features,
        predictor_contract_version,
    )
    predictor_columns = predictor_columns_for_contract(
        predictor_contract_version
    )
    missing = sorted(set(predictor_columns).difference(features.columns))
    if missing:
        raise ValueError(
            "Feature drift frame is missing predictors: "
            f"{missing}"
        )
    if "timestamp" not in features.columns or "trainable" not in features.columns:
        raise ValueError(
            "Feature drift frame requires timestamp and trainable columns"
        )

    frame = features.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    trainable = frame.loc[frame["trainable"].astype(bool)].sort_values(
        "timestamp", kind="stable"
    )
    split_index = max(1, len(trainable) // 2) if len(trainable) else 0
    early = trainable.iloc[:split_index]
    late = trainable.iloc[split_index:]
    records: list[dict[str, Any]] = []

    for feature_name in predictor_columns:
        early_values = _finite_values(early[feature_name])
        late_values = _finite_values(late[feature_name])
        early_stats = _distribution(early[feature_name])
        late_stats = _distribution(late[feature_name])
        early_mean = early_stats["mean"]
        late_mean = late_stats["mean"]

        if early_mean is None or late_mean is None:
            standardized_shift = None
        else:
            delta = late_mean - early_mean
            early_variance = (
                float(np.var(early_values, ddof=0)) if len(early_values) else 0.0
            )
            late_variance = (
                float(np.var(late_values, ddof=0)) if len(late_values) else 0.0
            )
            pooled_scale = float(
                np.sqrt((early_variance + late_variance) / 2.0)
            )
            if pooled_scale == 0.0:
                standardized_shift = 0.0 if delta == 0.0 else float(np.sign(delta))
            else:
                standardized_shift = float(delta / pooled_scale)

        all_values = _finite_values(frame[feature_name])
        total_rows = int(len(frame))
        records.append(
            {
                "feature_name": feature_name,
                "early_period_count": int(len(early_values)),
                "late_period_count": int(len(late_values)),
                "early_mean": early_stats["mean"],
                "early_std": early_stats["std"],
                "late_mean": late_stats["mean"],
                "late_std": late_stats["std"],
                "early_median": early_stats["median"],
                "late_median": late_stats["median"],
                "early_iqr": early_stats["iqr"],
                "late_iqr": late_stats["iqr"],
                "standardized_mean_difference": standardized_shift,
                "absolute_drift_score": (
                    abs(standardized_shift)
                    if standardized_shift is not None
                    else None
                ),
                "finite_rate": (
                    float(len(all_values) / total_rows) if total_rows else None
                ),
                "missing_rate": (
                    float(1.0 - len(all_values) / total_rows)
                    if total_rows
                    else None
                ),
            }
        )

    drift = pd.DataFrame.from_records(records).sort_values(
        ["absolute_drift_score", "feature_name"],
        ascending=[False, True],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)

    metadata = {
        "split_rule": (
            "Sort trainable rows by timestamp ascending; first max(1, floor(n/2)) "
            "rows are early and the remainder are late."
        ),
        "n_trainable_rows": int(len(trainable)),
        "early_start": (
            early["timestamp"].min() if len(early) else None
        ),
        "early_end": early["timestamp"].max() if len(early) else None,
        "late_start": late["timestamp"].min() if len(late) else None,
        "late_end": late["timestamp"].max() if len(late) else None,
        "predictor_count": int(len(predictor_columns)),
        "features_ranked_by": "absolute_drift_score descending, feature_name ascending",
        "constant_scale_rule": (
            "When both partitions have zero pooled standard deviation, shift is "
            "0 for equal means and sign(mean difference) otherwise."
        ),
    }
    return metadata, drift


def _reason_counts(frame: pd.DataFrame, column: str) -> dict[str, int] | None:
    if column not in frame.columns:
        return None
    values = frame[column].dropna().astype(str).value_counts()
    return {str(key): int(values[key]) for key in sorted(values.index)}


def dataset_quality_diagnostics(
    features: pd.DataFrame,
    *,
    predictor_columns: tuple[str, ...] = PREDICTOR_COLUMNS,
) -> dict[str, Any]:
    frame = features.copy()
    total_rows = int(len(frame))
    if "timestamp" in frame.columns and total_rows:
        timestamps = pd.to_datetime(frame["timestamp"], utc=True)
        first_timestamp = timestamps.min()
        last_timestamp = timestamps.max()
    else:
        first_timestamp = None
        last_timestamp = None

    if "feature_ready" in frame.columns:
        feature_ready = frame["feature_ready"].fillna(False).astype(bool)
    else:
        predictor_values = frame.loc[:, list(predictor_columns)].apply(
            pd.to_numeric, errors="coerce"
        )
        feature_ready = pd.Series(
            np.isfinite(predictor_values.to_numpy(dtype=float)).all(axis=1),
            index=frame.index,
        )

    target_eligible = (
        frame["target_eligible"].fillna(False).astype(bool)
        if "target_eligible" in frame.columns
        else frame.get("target", pd.Series(index=frame.index)).notna()
    )
    trainable = (
        frame["trainable"].fillna(False).astype(bool)
        if "trainable" in frame.columns
        else feature_ready & target_eligible
    )
    trainable_targets = pd.to_numeric(
        frame.loc[trainable, "target"], errors="coerce"
    ) if "target" in frame.columns else pd.Series(dtype=float)
    target_counts = trainable_targets.value_counts().to_dict()
    n_up = int(target_counts.get(1, 0))
    n_down = int(target_counts.get(0, 0))
    n_trainable = int(trainable.sum())

    magnitude_summary = None
    if "target_log_return_15m" in frame.columns:
        returns = pd.to_numeric(
            frame.loc[trainable, "target_log_return_15m"], errors="coerce"
        )
        magnitudes = np.abs(np.expm1(returns.to_numpy(dtype=float)))
        magnitudes = magnitudes[np.isfinite(magnitudes)]
        magnitude_summary = (
            {
                "count": int(len(magnitudes)),
                "mean": float(magnitudes.mean()),
                "std": float(magnitudes.std(ddof=0)),
                "median": float(np.median(magnitudes)),
                "p90": float(np.quantile(magnitudes, 0.90)),
                "max": float(magnitudes.max()),
            }
            if len(magnitudes)
            else {"count": 0}
        )

    return {
        "total_feature_rows": total_rows,
        "feature_ready_rows": int(feature_ready.sum()),
        "target_eligible_rows": int(target_eligible.sum()),
        "trainable_rows": n_trainable,
        "trainable_up_count": n_up,
        "trainable_down_count": n_down,
        "trainable_up_rate": (
            float(n_up / n_trainable) if n_trainable else None
        ),
        "trainable_down_rate": (
            float(n_down / n_trainable) if n_trainable else None
        ),
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "predictor_count": int(len(predictor_columns)),
        "rows_excluded_for_feature_readiness": int(
            total_rows - int(feature_ready.sum())
        ),
        "rows_excluded_for_target_eligibility": int(
            total_rows - int(target_eligible.sum())
        ),
        "canonical_eligibility_reason_counts": _reason_counts(
            frame, "canonical_eligibility_reason"
        ),
        "target_eligibility_reason_counts": _reason_counts(
            frame, "target_eligibility_reason"
        ),
        "target_return_magnitude": magnitude_summary,
    }


def top_drift_features(
    drift: pd.DataFrame,
    limit: int = 20,
) -> list[dict[str, Any]]:
    return drift.head(limit).where(pd.notna(drift), None).to_dict("records")
