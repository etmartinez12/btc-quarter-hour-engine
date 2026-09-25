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
from .validation import ExpandingWindowSplit, classification_metrics


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
        "models": {},
        "validation": asdict(cfg.validation),
    }

    for model_name, factory in model_factories.items():
        fold_metrics: list[dict[str, float]] = []
        skipped_folds = 0

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

        if not fold_metrics:
            raise ValueError(
                f"walk-forward validation produced no trainable folds for {model_name}; "
                "at least one training split must contain both classes"
            )

        metrics_frame = pd.DataFrame(fold_metrics)
        summary["models"][model_name] = {
            "folds": len(fold_metrics),
            "skipped_folds": skipped_folds,
            "mean_metrics": metrics_frame.mean(numeric_only=True).to_dict(),
        }

    return summary


def main() -> None:
    summary = run_walk_forward_benchmark()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
