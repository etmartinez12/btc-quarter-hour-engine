from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd

from .acquisition import load_sample_market_data, normalize_market_frame
from .boundaries import build_boundary_price_frame
from .config import BaselineConfig, ResearchValidationConfig
from .ensemble import build_consensus_diagnostics, build_simple_average_ensemble, build_validation_weighted_ensemble
from .features import build_feature_frame
from .models import ExtraTreesBaselineModel, LightGBMBaselineModel, LogisticBaselineModel, XGBoostBaselineModel
from .targets import build_direction_target
from .validation import (
    ExpandingWindowSplit,
    accuracy_by_boundary_slot,
    accuracy_by_move_size,
    build_common_oof_evaluation_frame,
    classification_metrics,
    confidence_accuracy_table,
    pairwise_prediction_agreement,
    score_naive_baselines,
)


def _to_builtin(value):
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            return value
    return value


def _feature_family_summary(feature_columns: list[str]) -> dict[str, int]:
    family_rules = {
        "momentum": ("return_", "momentum_", "price_acceleration"),
        "volatility": ("realized_vol_", "vol_regime_ratio_", "regime_vol_"),
        "volume": ("volume_",),
        "microstructure": (
            "bid_size",
            "ask_size",
            "depth_bid_5",
            "depth_ask_5",
            "trade_count",
            "buy_volume",
            "sell_volume",
            "order_book_imbalance",
            "depth_imbalance_",
            "quote_pressure",
            "buy_sell_volume_imbalance_",
            "aggressive_buy_share_",
            "aggressive_sell_share_",
            "average_trade_size_",
            "trade_count_",
            "buy_volume_",
            "sell_volume_",
            "trade_flow_imbalance_",
            "spread_change_",
        ),
        "technical": (
            "spread_bps",
            "range_",
            "distance_from_",
            "rolling_extrema_distance_",
            "price_vs_ema_",
            "ema_spread_",
            "price_vs_volume_weighted_midpoint_",
        ),
        "regime": ("regime_trend_",),
        "time": ("tod_", "dow_", "is_weekend"),
        "price_level": ("midpoint",),
    }
    summary: dict[str, int] = {}
    for family, prefixes in family_rules.items():
        summary[family] = sum(
            1
            for column in feature_columns
            if any(column == prefix or column.startswith(prefix) for prefix in prefixes)
        )
    return summary


def _align_oof_predictions(model_oof_predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    from .validation.oof import align_oof_predictions

    return align_oof_predictions(model_oof_predictions)


def build_training_dataset(
    market_data: pd.DataFrame | None = None,
    config: BaselineConfig | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    cfg = config or BaselineConfig()
    source = market_data if market_data is not None else load_sample_market_data()
    raw = normalize_market_frame(source)
    boundaries = build_boundary_price_frame(raw)
    features = build_feature_frame(raw, cfg.feature)
    targets = build_direction_target(boundaries)

    previous_prices = boundaries[["timestamp", "midpoint"]].copy()
    previous_prices["timestamp"] = previous_prices["timestamp"] + pd.Timedelta(minutes=15)
    previous_prices = previous_prices.rename(columns={"midpoint": "price_t_minus_15m"})

    dataset = features.merge(targets, on="timestamp", how="inner")
    dataset = dataset.merge(previous_prices, on="timestamp", how="left")
    dataset = dataset.dropna(subset=["target"]).copy()
    dataset["label_available_time"] = dataset["timestamp"] + pd.Timedelta(minutes=15)
    dataset["target"] = dataset["target"].astype(int)

    feature_columns = [
        column
        for column in dataset.columns
        if column
        not in {
            "timestamp",
            "target",
            "price_t",
            "price_t_plus_15m",
            "price_change",
            "log_return_15m",
            "label_available_time",
            "price_t_minus_15m",
        }
    ]
    dataset = dataset.dropna(subset=feature_columns).reset_index(drop=True)
    return dataset, feature_columns


def run_walk_forward_benchmark(
    config: BaselineConfig | None = None,
    market_data: pd.DataFrame | None = None,
) -> dict[str, object]:
    cfg = config or BaselineConfig()
    dataset, feature_columns = build_training_dataset(market_data=market_data, config=cfg)
    X = dataset[feature_columns]
    y = dataset["target"]

    splitter = ExpandingWindowSplit(
        initial_train_size=cfg.validation.initial_train_size,
        test_size=cfg.validation.test_size,
        step_size=cfg.validation.step_size,
    )
    n_splits = splitter.get_n_splits(X)
    if n_splits == 0:
        raise ValueError(
            "walk-forward validation produced zero folds; reduce initial_train_size/test_size "
            "or provide more quarter-hour observations"
        )

    fold_summaries: list[dict[str, object]] = []
    for fold_number, (train_idx, test_idx) in enumerate(splitter.split(X)):
        train_df = dataset.iloc[train_idx].copy()
        test_df = dataset.iloc[test_idx].copy()
        test_start_time = test_df["timestamp"].min()
        train_df = train_df.loc[train_df["label_available_time"] <= test_start_time].copy()
        if not train_df.empty and train_df["label_available_time"].max() > test_start_time:
            raise ValueError("training data for a fold includes labels that are not yet available at the test start time")
        fold_summaries.append(
            {
                "fold_number": int(fold_number),
                "train_start": str(train_df["timestamp"].min()) if not train_df.empty else None,
                "train_end": str(train_df["timestamp"].max()) if not train_df.empty else None,
                "test_start": str(test_df["timestamp"].min()),
                "test_end": str(test_df["timestamp"].max()),
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "train_up_rate": float(train_df["target"].mean()) if not train_df.empty else None,
                "test_up_rate": float(test_df["target"].mean()) if not test_df.empty else None,
            }
        )

    model_factories = {
        "logistic_regression": lambda: LogisticBaselineModel(
            max_iter=cfg.logistic_max_iter,
            random_state=cfg.random_state,
        ),
        "extra_trees": lambda: ExtraTreesBaselineModel(
            n_estimators=cfg.extra_trees_n_estimators,
            max_depth=cfg.extra_trees_max_depth,
            random_state=cfg.random_state,
        ),
        "lightgbm": lambda: LightGBMBaselineModel(
            n_estimators=cfg.lightgbm_n_estimators,
            learning_rate=cfg.lightgbm_learning_rate,
            num_leaves=cfg.lightgbm_num_leaves,
            random_state=cfg.random_state,
        ),
        "xgboost": lambda: XGBoostBaselineModel(
            n_estimators=cfg.xgboost_n_estimators,
            learning_rate=cfg.xgboost_learning_rate,
            max_depth=cfg.xgboost_max_depth,
            random_state=cfg.random_state,
        ),
    }

    summary: dict[str, object] = {
        "n_rows": len(dataset),
        "n_features": len(feature_columns),
        "n_common_oof_predictions": 0,
        "feature_columns": feature_columns,
        "feature_family_counts": _feature_family_summary(feature_columns),
        "base_models": list(model_factories),
        "models": {},
        "validation": asdict(cfg.validation),
        "validation_protocol": {
            "mode": "demo_row_based",
            "initial_train_size": cfg.validation.initial_train_size,
            "test_size": cfg.validation.test_size,
            "step_size": cfg.validation.step_size,
            "test_overlap_allowed": False,
            "label_availability_enforced": True,
            "canonical_horizon_minutes": 15,
            "n_common_oof_predictions": 0,
        },
        "folds": fold_summaries,
    }
    model_oof_predictions: dict[str, pd.DataFrame] = {}

    for model_name, factory in model_factories.items():
        fold_metrics: list[dict[str, float]] = []
        skipped_folds = 0
        oof_parts: list[pd.DataFrame] = []

        for fold_number, (train_idx, test_idx) in enumerate(splitter.split(X)):
            model = factory()
            train_df = dataset.iloc[train_idx].copy()
            test_df = dataset.iloc[test_idx].copy()
            test_start_time = test_df["timestamp"].min()
            train_df = train_df.loc[train_df["label_available_time"] <= test_start_time].copy()
            if not train_df.empty and train_df["label_available_time"].max() > test_start_time:
                raise ValueError("training data for a fold includes labels that are not yet available at the test start time")
            X_train = X.iloc[train_df.index]
            y_train = y.iloc[train_df.index]
            X_test = X.iloc[test_df.index]
            y_test = y.iloc[test_df.index]

            if y_train.nunique() < 2:
                skipped_folds += 1
                continue

            model.fit(X_train, y_train)
            probabilities = model.predict_proba(X_test)[:, 1]
            predictions = (probabilities >= 0.5).astype(int)
            fold_metrics.append(classification_metrics(y_test, predictions, probabilities))
            oof_parts.append(
                pd.DataFrame(
                    {
                        "fold_number": [fold_number] * len(X_test),
                        "timestamp": test_df["timestamp"].to_numpy(),
                        "y_true": y_test.to_numpy(),
                        "y_pred": predictions,
                        "y_proba": probabilities,
                        "log_return_15m": test_df["log_return_15m"].to_numpy(),
                        "midpoint": test_df["midpoint"].to_numpy(),
                        "price_t_minus_15m": test_df["price_t_minus_15m"].to_numpy(),
                        "return_1m": test_df["return_1m"].to_numpy(),
                        "return_5m": test_df["return_5m"].to_numpy(),
                    }
                )
            )

        if not fold_metrics:
            raise ValueError(
                f"walk-forward validation produced no trainable folds for {model_name}; "
                "at least one training split must contain both classes"
            )

        metrics_frame = pd.DataFrame(fold_metrics)
        oof_frame = pd.concat(oof_parts, ignore_index=True).sort_values(["fold_number", "timestamp"]).reset_index(drop=True)
        summary["models"][model_name] = {
            "folds": len(fold_metrics),
            "skipped_folds": skipped_folds,
            "mean_fold_metrics": metrics_frame.mean(numeric_only=True).to_dict(),
            "pooled_oof_metrics": classification_metrics(oof_frame["y_true"], oof_frame["y_pred"], oof_frame["y_proba"]),
            "confidence_accuracy": confidence_accuracy_table(
                oof_frame["timestamp"],
                oof_frame["y_true"],
                oof_frame["y_proba"],
            ),
            "accuracy_by_move_size": accuracy_by_move_size(
                oof_frame["y_true"],
                oof_frame["y_pred"],
                oof_frame["log_return_15m"],
            ),
            "accuracy_by_boundary_slot": accuracy_by_boundary_slot(
                oof_frame["timestamp"],
                oof_frame["y_true"],
                oof_frame["y_pred"],
            ),
        }
        model_oof_predictions[model_name] = oof_frame

    aligned_oof = _align_oof_predictions(model_oof_predictions)
    common_oof = build_common_oof_evaluation_frame(model_oof_predictions, random_state=cfg.random_state)
    summary["n_common_oof_predictions"] = int(len(common_oof))
    summary["common_oof_population"] = {
        "n_observations": int(len(common_oof)),
        "n_folds": int(common_oof["fold_number"].nunique()) if len(common_oof) else 0,
        "timestamp_min": str(common_oof["timestamp"].min()) if len(common_oof) else None,
        "timestamp_max": str(common_oof["timestamp"].max()) if len(common_oof) else None,
    }
    research_cfg = ResearchValidationConfig()
    summary["research_validation_spec"] = {
        **asdict(research_cfg),
        "expanding_training": True,
        "test_overlap_allowed": False,
    }
    summary["validation_protocol"]["n_common_oof_predictions"] = int(len(common_oof))
    y_true = common_oof["y_true"]
    summary["oof_class_balance"] = {
        "n_predictions": int(len(common_oof)),
        "n_up": int((y_true == 1).sum()),
        "n_down": int((y_true == 0).sum()),
        "up_rate": float((y_true == 1).mean()) if len(common_oof) else 0.0,
        "down_rate": float((y_true == 0).mean()) if len(common_oof) else 0.0,
    }
    if summary["oof_class_balance"]["n_up"] + summary["oof_class_balance"]["n_down"] != summary["oof_class_balance"]["n_predictions"]:
        raise ValueError("Oof class balance does not sum to the common OOF population count")
    summary["naive_baselines"] = score_naive_baselines(common_oof)

    ensemble_model_names = list(model_oof_predictions)
    simple_average_oof = build_simple_average_ensemble(aligned_oof, ensemble_model_names)
    weighted_average_oof, weight_history = build_validation_weighted_ensemble(
        aligned_oof,
        ensemble_model_names,
        metric=cfg.ensemble.weighted_metric,
        min_weight=cfg.ensemble.min_weight,
    )

    summary["ensembles"] = {}
    for ensemble_name, ensemble_frame in {
        "simple_average": simple_average_oof,
        "validation_weighted_average": weighted_average_oof,
    }.items():
        summary["ensembles"][ensemble_name] = {
            "pooled_oof_metrics": classification_metrics(
                ensemble_frame["y_true"],
                ensemble_frame["y_pred"],
                ensemble_frame["y_proba"],
            ),
            "confidence_accuracy": confidence_accuracy_table(
                ensemble_frame["timestamp"],
                ensemble_frame["y_true"],
                ensemble_frame["y_proba"],
            ),
            "accuracy_by_move_size": accuracy_by_move_size(
                ensemble_frame["y_true"],
                ensemble_frame["y_pred"],
                ensemble_frame["log_return_15m"],
            ),
            "accuracy_by_boundary_slot": accuracy_by_boundary_slot(
                ensemble_frame["timestamp"],
                ensemble_frame["y_true"],
                ensemble_frame["y_pred"],
            ),
        }

    summary["ensembles"]["validation_weighted_average"]["weight_history"] = weight_history
    summary["model_comparison"] = {
        "pairwise_prediction_agreement": pairwise_prediction_agreement(model_oof_predictions),
        "ensemble_consensus": build_consensus_diagnostics(aligned_oof, ensemble_model_names),
    }

    benchmark_rows: list[dict[str, object]] = []
    common_prediction_count = summary["n_common_oof_predictions"]
    for baseline_name, baseline_summary in summary["naive_baselines"].items():
        metrics = baseline_summary["metrics"]
        benchmark_rows.append(
            {
                "name": baseline_name,
                "benchmark_type": "naive",
                "n_predictions": common_prediction_count,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "brier_score": None,
                "log_loss": None,
                "roc_auc": None,
            }
        )

    for model_name in summary["models"]:
        model_columns = common_oof[["y_true", f"{model_name}_pred", f"{model_name}_proba"]].copy()
        metrics = classification_metrics(
            model_columns["y_true"],
            model_columns[f"{model_name}_pred"],
            model_columns[f"{model_name}_proba"],
        )
        benchmark_rows.append(
            {
                "name": model_name,
                "benchmark_type": "model",
                "n_predictions": common_prediction_count,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "brier_score": metrics.get("brier_score"),
                "log_loss": metrics.get("log_loss"),
                "roc_auc": metrics.get("roc_auc"),
            }
        )

    simple_average_common = simple_average_oof[["timestamp", "fold_number", "y_true", "y_pred", "y_proba"]].merge(
        common_oof[["timestamp", "fold_number"]],
        on=["timestamp", "fold_number"],
        how="inner",
    )
    if len(simple_average_common) != common_prediction_count:
        raise ValueError("simple-average ensemble does not align to the common OOF population")
    weighted_average_common = weighted_average_oof[["timestamp", "fold_number", "y_true", "y_pred", "y_proba"]].merge(
        common_oof[["timestamp", "fold_number"]],
        on=["timestamp", "fold_number"],
        how="inner",
    )
    if len(weighted_average_common) != common_prediction_count:
        raise ValueError("validation-weighted ensemble does not align to the common OOF population")
    ensemble_frames = {
        "simple_average": simple_average_common,
        "validation_weighted_average": weighted_average_common,
    }
    for ensemble_name, ensemble_frame in ensemble_frames.items():
        metrics = classification_metrics(
            ensemble_frame["y_true"],
            ensemble_frame["y_pred"],
            ensemble_frame["y_proba"],
        )
        benchmark_rows.append(
            {
                "name": ensemble_name,
                "benchmark_type": "ensemble",
                "n_predictions": common_prediction_count,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "brier_score": metrics.get("brier_score"),
                "log_loss": metrics.get("log_loss"),
                "roc_auc": metrics.get("roc_auc"),
            }
        )

    for row in benchmark_rows:
        if row["n_predictions"] != common_prediction_count:
            raise ValueError("benchmark_comparison rows must all use the common OOF population size")

    summary["benchmark_comparison"] = benchmark_rows
    return summary


def main() -> None:
    summary = run_walk_forward_benchmark()
    print(json.dumps(_to_builtin(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
