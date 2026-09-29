from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import accuracy_by_boundary_slot, accuracy_by_move_size, classification_metrics
from .oof import align_oof_predictions

DEFAULT_RANDOM_SEED = 42
NAIVE_BASELINE_ORDER = (
    "always_up",
    "always_down",
    "random_50",
    "previous_quarter_direction",
    "momentum_1m",
    "momentum_5m",
)


def _sorted_evaluation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    sort_columns = []
    if {"fold_number", "timestamp"}.issubset(frame.columns):
        sort_columns = ["fold_number", "timestamp"]
    elif "timestamp" in frame.columns:
        sort_columns = ["timestamp"]
    if sort_columns:
        return frame.sort_values(sort_columns).reset_index(drop=True)
    return frame.copy().reset_index(drop=True)


def _prediction_series(values, *, index=None, name: str = "y_pred") -> pd.Series:
    series = pd.Series(values, index=index, name=name)
    return series.astype(int)


def always_up(frame: pd.DataFrame) -> pd.Series:
    ordered = _sorted_evaluation_frame(frame)
    return _prediction_series(np.ones(len(ordered), dtype=int), index=ordered.index)


def always_down(frame: pd.DataFrame) -> pd.Series:
    ordered = _sorted_evaluation_frame(frame)
    return _prediction_series(np.zeros(len(ordered), dtype=int), index=ordered.index)


def random_50(frame: pd.DataFrame, random_state: int | None = None) -> pd.Series:
    """Create deterministic 50/50 baseline labels using the project random seed."""
    ordered = _sorted_evaluation_frame(frame)
    seed = DEFAULT_RANDOM_SEED if random_state is None else int(random_state)
    rng = np.random.RandomState(seed)
    random_draws = rng.randint(0, 2, size=len(ordered))
    return _prediction_series(random_draws, index=ordered.index)


def previous_quarter_direction(frame: pd.DataFrame) -> pd.Series:
    """Predict the previous quarter-hour move direction using the exact canonical historical price."""
    ordered = _sorted_evaluation_frame(frame)
    if "midpoint" not in ordered.columns:
        raise KeyError("previous_quarter_direction requires exact midpoint and price_t_minus_15m")
    if "price_t_minus_15m" not in ordered.columns:
        raise KeyError("previous_quarter_direction requires exact price_t_minus_15m")

    current_price = ordered["midpoint"].to_numpy(dtype=float)
    previous_price = ordered["price_t_minus_15m"].to_numpy(dtype=float)
    predictions = (current_price > previous_price).astype(int)
    return _prediction_series(predictions, index=ordered.index)


def momentum_1m(frame: pd.DataFrame) -> pd.Series:
    ordered = _sorted_evaluation_frame(frame)
    if "return_1m" not in ordered.columns:
        raise KeyError("momentum_1m requires a return_1m column")
    return _prediction_series(ordered["return_1m"].gt(0).astype(int).to_numpy(), index=ordered.index)


def momentum_5m(frame: pd.DataFrame) -> pd.Series:
    ordered = _sorted_evaluation_frame(frame)
    if "return_5m" not in ordered.columns:
        raise KeyError("momentum_5m requires a return_5m column")
    return _prediction_series(ordered["return_5m"].gt(0).astype(int).to_numpy(), index=ordered.index)


def build_naive_baseline_predictions(frame: pd.DataFrame, random_state: int | None = None) -> dict[str, pd.Series]:
    return {
        "always_up": always_up(frame),
        "always_down": always_down(frame),
        "random_50": random_50(frame, random_state=random_state),
        "previous_quarter_direction": previous_quarter_direction(frame),
        "momentum_1m": momentum_1m(frame),
        "momentum_5m": momentum_5m(frame),
    }


def build_common_oof_evaluation_frame(
    model_oof_predictions: dict[str, pd.DataFrame],
    random_state: int | None = None,
) -> pd.DataFrame:
    aligned = align_oof_predictions(model_oof_predictions)
    naive_predictions = build_naive_baseline_predictions(aligned, random_state=random_state)
    for baseline_name, baseline_predictions in naive_predictions.items():
        aligned[f"{baseline_name}_pred"] = baseline_predictions.to_numpy(dtype=int)
    return aligned.sort_values(["fold_number", "timestamp"]).reset_index(drop=True)


def score_naive_baselines(common_oof: pd.DataFrame) -> dict[str, dict[str, object]]:
    scored: dict[str, dict[str, object]] = {}
    for baseline_name in NAIVE_BASELINE_ORDER:
        pred_column = f"{baseline_name}_pred"
        if pred_column not in common_oof.columns:
            baseline_predictions = build_naive_baseline_predictions(common_oof)
            common_oof[pred_column] = baseline_predictions[baseline_name].to_numpy(dtype=int)

        frame = common_oof[["timestamp", "fold_number", "y_true", "log_return_15m", pred_column]].copy()
        frame["y_pred"] = frame[pred_column].astype(int)
        scored[baseline_name] = {
            "n_predictions": int(len(frame)),
            "metrics": classification_metrics(frame["y_true"], frame["y_pred"], y_proba=None),
            "confidence_accuracy": None,
            "accuracy_by_move_size": accuracy_by_move_size(
                frame["y_true"],
                frame["y_pred"],
                frame["log_return_15m"],
            ),
            "accuracy_by_boundary_slot": accuracy_by_boundary_slot(
                frame["timestamp"],
                frame["y_true"],
                frame["y_pred"],
            ),
        }
    return scored


__all__ = [
    "DEFAULT_RANDOM_SEED",
    "NAIVE_BASELINE_ORDER",
    "always_up",
    "always_down",
    "random_50",
    "previous_quarter_direction",
    "momentum_1m",
    "momentum_5m",
    "build_naive_baseline_predictions",
    "build_common_oof_evaluation_frame",
    "score_naive_baselines",
]
