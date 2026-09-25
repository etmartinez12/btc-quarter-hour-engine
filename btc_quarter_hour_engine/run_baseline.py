from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd

from .acquisition import load_sample_market_data
from .boundaries import build_boundary_price_frame
from .config import BaselineConfig
from .features import build_feature_frame
from .models import LightGBMBaselineModel, LogisticBaselineModel
from .targets import build_direction_target
from .validation import (
    ExpandingWindowSplit,
    accuracy_by_boundary_slot,
    accuracy_by_move_size,
    classification_metrics,
    confidence_accuracy_table,
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
        "technical": (
            "spread_bps",
            "range_",
            "distance_from_",
            "rolling_extrema_distance_",
            "price_vs_ema_",
            "ema_spread_",
            "price_vs_vwap_",
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


def build_training_dataset(config: BaselineConfig | None = None) -> tuple[pd.DataFrame, list[str]]:
    cfg = config or BaselineConfig()
    raw = load_sample_market_data()
    boundaries = build_boundary_price_frame(raw)
    features = build_feature_frame(raw, cfg.feature)
    targets = build_direction_target(boundaries)
    dataset = features.merge(targets, on="timestamp", how="inner")
    dataset = dataset.dropna(subset=["target"]).copy()
    dataset["target"] = dataset["target"].astype(int)

    feature_columns = [
        column
        for column in dataset.columns
        if column not in {"timestamp", "target", "price_t", "price_t_plus_15m", "price_change", "log_return_15m"}
    ]
    dataset = dataset.dropna(subset=feature_columns).reset_index(drop=True)
    return dataset, feature_columns


def run_walk_forward_benchmark(config: BaselineConfig | None = None) -> dict[str, object]:
    cfg = config or BaselineConfig()
    dataset, feature_columns = build_training_dataset(cfg)
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

    model_factories = {
        "logistic_regression": lambda: LogisticBaselineModel(
            max_iter=cfg.logistic_max_iter,
            random_state=cfg.random_state,
        ),
        "lightgbm": lambda: LightGBMBaselineModel(
            n_estimators=cfg.lightgbm_n_estimators,
            learning_rate=cfg.lightgbm_learning_rate,
            num_leaves=cfg.lightgbm_num_leaves,
            random_state=cfg.random_state,
        ),
    }

    summary: dict[str, object] = {
        "n_rows": len(dataset),
        "n_features": len(feature_columns),
        "feature_columns": feature_columns,
        "feature_family_counts": _feature_family_summary(feature_columns),
        "models": {},
        "validation": asdict(cfg.validation),
    }

    for model_name, factory in model_factories.items():
        fold_metrics: list[dict[str, float]] = []
        skipped_folds = 0
        oof_parts: list[pd.DataFrame] = []

        for train_idx, test_idx in splitter.split(X):
            model = factory()
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

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
                        "timestamp": dataset.iloc[test_idx]["timestamp"].to_numpy(),
                        "y_true": y_test.to_numpy(),
                        "y_pred": predictions,
                        "y_proba": probabilities,
                        "log_return_15m": dataset.iloc[test_idx]["log_return_15m"].to_numpy(),
                    }
                )
            )

        if not fold_metrics:
            raise ValueError(
                f"walk-forward validation produced no trainable folds for {model_name}; "
                "at least one training split must contain both classes"
            )

        metrics_frame = pd.DataFrame(fold_metrics)
        oof_frame = pd.concat(oof_parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        summary["models"][model_name] = {
            "folds": len(fold_metrics),
            "skipped_folds": skipped_folds,
            "mean_metrics": metrics_frame.mean(numeric_only=True).to_dict(),
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

    return summary


def main() -> None:
    summary = run_walk_forward_benchmark()
    print(json.dumps(_to_builtin(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
