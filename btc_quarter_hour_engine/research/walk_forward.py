from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from btc_quarter_hour_engine.models import (
    ExtraTreesBaselineModel,
    HistGradientBoostingBaselineModel,
    LogisticBaselineModel,
)
from btc_quarter_hour_engine.research.diagnostics import (
    dataset_quality_diagnostics,
    feature_drift_diagnostics,
    fold_metrics_and_model_diagnostics,
    top_drift_features,
)
from btc_quarter_hour_engine.research.features import (
    PREDICTOR_CONTRACT_VERSION,
    predictor_columns_for_contract,
    research_features_for_contract,
    research_model_matrix,
)
from btc_quarter_hour_engine.validation.metrics import (
    accuracy_by_boundary_slot,
    accuracy_by_move_size,
    classification_metrics,
    confidence_accuracy_table,
)
from btc_quarter_hour_engine.validation.walk_forward import (
    ExpandingTimeWindowSplit,
)


@dataclass(slots=True)
class ExploratoryResearchConfig:
    initial_train_period: str = "72h"
    test_period: str = "12h"
    step_period: str = "12h"
    random_state: int = 42
    logistic_max_iter: int = 1000
    extra_trees_n_estimators: int = 300
    extra_trees_max_depth: int | None = 6
    hist_gradient_boosting_max_iter: int = 100
    hist_gradient_boosting_learning_rate: float = 0.1
    hist_gradient_boosting_max_leaf_nodes: int = 15
    hist_gradient_boosting_min_samples_leaf: int = 20
    hist_gradient_boosting_l2_regularization: float = 1.0
    predictor_contract_version: str = PREDICTOR_CONTRACT_VERSION


def prepare_research_frame(
    features: pd.DataFrame,
    predictor_contract_version: str = PREDICTOR_CONTRACT_VERSION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    required = {
        "timestamp",
        "label_available_time",
        "target",
        "target_log_return_15m",
        "trainable",
    }

    missing = required.difference(features.columns)
    if missing:
        raise ValueError(
            f"Research feature frame missing columns: {sorted(missing)}"
        )

    frame = research_features_for_contract(
        features,
        predictor_contract_version,
    ).copy()

    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        utc=True,
        errors="raise",
    )
    frame["label_available_time"] = pd.to_datetime(
        frame["label_available_time"],
        utc=True,
        errors="raise",
    )

    frame = (
        frame.loc[frame["trainable"]]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if frame.empty:
        raise ValueError("Research feature frame has zero trainable rows")

    if frame["timestamp"].duplicated().any():
        raise ValueError("Research timestamps must be unique")

    X, y = research_model_matrix(
        frame,
        predictor_contract_version=predictor_contract_version,
    )

    return frame, X, y


def build_exploratory_folds(
    frame: pd.DataFrame,
    config: ExploratoryResearchConfig | None = None,
) -> list[dict[str, Any]]:
    cfg = config or ExploratoryResearchConfig()

    splitter = ExpandingTimeWindowSplit(
        initial_train_period=cfg.initial_train_period,
        test_period=cfg.test_period,
        step_period=cfg.step_period,
    )

    folds: list[dict[str, Any]] = []
    all_test_indices: list[int] = []

    for fold_number, (raw_train_idx, test_idx) in enumerate(
        splitter.split(frame["timestamp"])
    ):
        test_start = frame.iloc[test_idx]["timestamp"].min()

        raw_train = frame.iloc[raw_train_idx]

        eligible_mask = (
            raw_train["label_available_time"] < test_start
        )

        train_idx = raw_train_idx[
            eligible_mask.to_numpy()
        ]

        if len(train_idx) == 0:
            raise ValueError(
                f"Fold {fold_number} has zero embargo-safe training rows"
            )

        train = frame.iloc[train_idx]
        test = frame.iloc[test_idx]

        if train["label_available_time"].max() >= test_start:
            raise ValueError(
                f"Fold {fold_number} contains unavailable training labels"
            )

        if train["target"].nunique() < 2:
            raise ValueError(
                f"Fold {fold_number} training set has fewer than two classes"
            )

        if len(set(train_idx).intersection(test_idx)):
            raise ValueError(
                f"Fold {fold_number} train/test indices overlap"
            )

        all_test_indices.extend(test_idx.tolist())

        folds.append(
            {
                "fold_number": fold_number,
                "train_idx": train_idx,
                "test_idx": test_idx,
                "test_start": test_start,
            }
        )

    if not folds:
        raise ValueError("Exploratory walk-forward produced zero folds")

    if len(all_test_indices) != len(set(all_test_indices)):
        raise ValueError("OOF test windows overlap")

    return folds


def _model_factories(
    config: ExploratoryResearchConfig,
):
    return {
        "logistic_regression": lambda: LogisticBaselineModel(
            max_iter=config.logistic_max_iter,
            random_state=config.random_state,
        ),
        "extra_trees": lambda: ExtraTreesBaselineModel(
            n_estimators=config.extra_trees_n_estimators,
            max_depth=config.extra_trees_max_depth,
            random_state=config.random_state,
        ),
        "hist_gradient_boosting": lambda: HistGradientBoostingBaselineModel(
            max_iter=config.hist_gradient_boosting_max_iter,
            learning_rate=config.hist_gradient_boosting_learning_rate,
            max_leaf_nodes=config.hist_gradient_boosting_max_leaf_nodes,
            min_samples_leaf=config.hist_gradient_boosting_min_samples_leaf,
            l2_regularization=(
                config.hist_gradient_boosting_l2_regularization
            ),
            random_state=config.random_state,
        ),
    }


def _baseline_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, np.ndarray]:
    majority = int(train["target"].mean() > 0.5)

    momentum = (
        test["momentum_log_return_15m"]
        .gt(0)
        .astype(int)
        .to_numpy()
    )

    return {
        "always_up": np.ones(len(test), dtype=int),
        "always_down": np.zeros(len(test), dtype=int),
        "train_majority": np.full(
            len(test),
            majority,
            dtype=int,
        ),
        "causal_momentum_15m": momentum,
    }


def _model_fold_predictions(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test: pd.DataFrame,
    fold_number: int,
    config: ExploratoryResearchConfig,
) -> list[pd.DataFrame]:
    parts = []
    for model_name, factory in _model_factories(config).items():
        model = factory()
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        parts.append(
            pd.DataFrame(
                {
                    "benchmark": model_name,
                    "benchmark_type": "model",
                    "fold_number": fold_number,
                    "timestamp": test["timestamp"].to_numpy(),
                    "y_true": y_test.to_numpy(dtype=int),
                    "y_pred": (probabilities >= 0.5).astype(int),
                    "y_proba": probabilities,
                    "target_log_return_15m": (
                        test["target_log_return_15m"].to_numpy()
                    ),
                }
            )
        )
    return parts


def _fold_record(
    fold_number: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, Any]:
    test_start = test["timestamp"].min()

    return {
        "fold_number": int(fold_number),
        "train_start": train["timestamp"].min(),
        "train_end": train["timestamp"].max(),
        "max_train_label_available_time": (
            train["label_available_time"].max()
        ),
        "test_start": test_start,
        "test_end": test["timestamp"].max(),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train_up_rate": float(train["target"].mean()),
        "test_up_rate": float(test["target"].mean()),
        "strict_label_embargo_ok": bool(
            train["label_available_time"].max() < test_start
        ),
    }


def run_exploratory_walk_forward(
    features: pd.DataFrame,
    config: ExploratoryResearchConfig | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    cfg = config or ExploratoryResearchConfig()

    frame, X, y = prepare_research_frame(
        features,
        cfg.predictor_contract_version,
    )
    folds = build_exploratory_folds(frame, cfg)

    model_parts: list[pd.DataFrame] = []
    baseline_parts: list[pd.DataFrame] = []
    fold_records: list[dict[str, Any]] = []

    for fold in folds:
        fold_number = fold["fold_number"]
        train_idx = fold["train_idx"]
        test_idx = fold["test_idx"]

        train = frame.iloc[train_idx]
        test = frame.iloc[test_idx]

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        fold_records.append(
            _fold_record(
                fold_number,
                train,
                test,
            )
        )

        model_parts.extend(
            _model_fold_predictions(
                X_train, y_train, X_test, y_test, test, fold_number, cfg
            )
        )

        for baseline_name, predictions in _baseline_predictions(
            train,
            test,
        ).items():
            baseline_parts.append(
                pd.DataFrame(
                    {
                        "benchmark": baseline_name,
                        "benchmark_type": "baseline",
                        "fold_number": fold_number,
                        "timestamp": test["timestamp"].to_numpy(),
                        "y_true": y_test.to_numpy(dtype=int),
                        "y_pred": predictions,
                        "y_proba": np.nan,
                        "target_log_return_15m": (
                            test["target_log_return_15m"].to_numpy()
                        ),
                    }
                )
            )

    predictions = pd.concat(
        model_parts + baseline_parts,
        ignore_index=True,
    ).sort_values(
        ["benchmark", "fold_number", "timestamp"]
    ).reset_index(drop=True)


    benchmark_summary: dict[str, Any] = {}

    for benchmark, group in predictions.groupby(
        "benchmark",
        sort=True,
    ):
        benchmark_type = group["benchmark_type"].iloc[0]

        y_proba = (
            group["y_proba"]
            if benchmark_type == "model"
            else None
        )

        metrics = classification_metrics(
            group["y_true"],
            group["y_pred"],
            y_proba,
        )

        record: dict[str, Any] = {
            "benchmark_type": benchmark_type,
            "n_predictions": int(len(group)),
            "metrics": metrics,
            "accuracy_by_move_size": accuracy_by_move_size(
                group["y_true"],
                group["y_pred"],
                group["target_log_return_15m"],
            ),
            "accuracy_by_boundary_slot": accuracy_by_boundary_slot(
                group["timestamp"],
                group["y_true"],
                group["y_pred"],
            ),
        }

        if benchmark_type == "model":
            record["confidence_accuracy"] = confidence_accuracy_table(
                group["timestamp"],
                group["y_true"],
                group["y_proba"],
            )
        else:
            record["confidence_accuracy"] = None

        benchmark_summary[benchmark] = record

    expected_oof = sum(
        len(fold["test_idx"])
        for fold in folds
    )

    for benchmark, record in benchmark_summary.items():
        if record["n_predictions"] != expected_oof:
            raise ValueError(
                f"{benchmark} does not cover the common OOF population"
            )

    fold_metrics, model_diagnostics, model_agreement = (
        fold_metrics_and_model_diagnostics(predictions, fold_records)
    )
    for benchmark, diagnostics in model_diagnostics.items():
        benchmark_summary[benchmark].update(diagnostics)

    predictor_columns = predictor_columns_for_contract(
        cfg.predictor_contract_version
    )
    drift_metadata, feature_drift = feature_drift_diagnostics(
        features,
        predictor_contract_version=cfg.predictor_contract_version,
    )
    quality_features = research_features_for_contract(
        features,
        cfg.predictor_contract_version,
    )
    quality = dataset_quality_diagnostics(
        quality_features,
        predictor_columns=predictor_columns,
    )

    summary: dict[str, Any] = {
        "mode": "exploratory_short_history",
        "scientific_claim_status": (
            "development_exploratory_analysis_not_untouched_prospective_validation"
        ),
        "predictor_contract_version": cfg.predictor_contract_version,
        "predictor_count": len(predictor_columns),
        "n_trainable_rows": int(len(frame)),
        "dataset_summary": quality,
        "feature_drift": {
            **drift_metadata,
            "top_drift_features": top_drift_features(feature_drift),
        },
        "n_folds": int(len(folds)),
        "n_oof_predictions_per_benchmark": int(expected_oof),
        "validation": {
            **asdict(cfg),
            "expanding_training": True,
            "test_overlap_allowed": False,
            "strict_label_cutoff": (
                "label_available_time < test_start"
            ),
        },
        "folds": fold_records,
        "fold_metrics": fold_metrics,
        "benchmarks": benchmark_summary,
        "model_agreement": model_agreement,
    }

    return summary, predictions
