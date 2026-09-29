from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import classification_metrics, confidence_accuracy_table, accuracy_by_move_size, accuracy_by_boundary_slot

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
    """Predict the previous quarter-hour move direction.

    The historical move is computed as P[t] versus P[t-15m]. When the move is exactly
    flat, the baseline uses the documented deterministic tie policy and predicts LOWER (0).
    """
    ordered = _sorted_evaluation_frame(frame)
    current_price = ordered["price_t"] if "price_t" in ordered.columns else ordered["midpoint"]
    previous_price = (
        ordered["price_t_minus_15m"]
        if "price_t_minus_15m" in ordered.columns
        else ordered["previous_price_t"]
        if "previous_price_t" in ordered.columns
        else ordered["midpoint"].shift(1)
    )
    if previous_price is None:
        raise KeyError("previous_quarter_direction requires price_t, midpoint, or last-quarter historical price columns")
    predictions = (current_price.to_numpy(dtype=float) > previous_price.to_numpy(dtype=float)).astype(int)
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


def _align_oof_predictions(model_oof_predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    model_names = list(model_oof_predictions)
    if not model_names:
        raise ValueError("model_oof_predictions must contain at least one model")
    first_name = model_names[0]
    keep_columns = [
        "timestamp",
        "fold_number",
        "y_true",
        "log_return_15m",
        "midpoint",
        "price_t_minus_15m",
        "return_1m",
        "return_5m",
    ]
    aligned = model_oof_predictions[first_name][[column for column in keep_columns if column in model_oof_predictions[first_name].columns]].copy()
    for model_name in model_names:
        model_frame = model_oof_predictions[model_name][["timestamp", "fold_number", "y_pred", "y_proba"]].rename(
            columns={
                "y_pred": f"{model_name}_pred",
                "y_proba": f"{model_name}_proba",
            }
        )
        aligned = aligned.merge(model_frame, on=["timestamp", "fold_number"], how="inner")
    return aligned.sort_values(["fold_number", "timestamp"]).reset_index(drop=True)


def build_common_oof_evaluation_frame(
    model_oof_predictions: dict[str, pd.DataFrame],
    random_state: int | None = None,
) -> pd.DataFrame:
    aligned = _align_oof_predictions(model_oof_predictions)
    naive_predictions = build_naive_baseline_predictions(aligned, random_state=random_state)
    for baseline_name, baseline_predictions in naive_predictions.items():
        aligned[f"{baseline_name}_pred"] = baseline_predictions.to_numpy(dtype=int)
    return aligned.sort_values(["fold_number", "timestamp"]).reset_index(drop=True)


def score_naive_baselines(common_oof: pd.DataFrame) -> dict[str, dict[str, object]]:
    scored: dict[str, dict[str, object]] = {}
    for baseline_name in NAIVE_BASELINE_ORDER:
        pred_column = f"{baseline_name}_pred"
        if pred_column not in common_oof.columns:
            pred_column = pred_column
            baseline_predictions = build_naive_baseline_predictions(common_oof)
            common_oof[f"{baseline_name}_pred"] = baseline_predictions[baseline_name].to_numpy(dtype=int)
        frame = common_oof[["timestamp", "fold_number", "y_true", "log_return_15m", pred_column]].copy()
        frame["y_pred"] = frame[pred_column].astype(int)
        frame["y_proba"] = frame["y_pred"].astype(float)
        scored[baseline_name] = {
            "n_predictions": int(len(frame)),
            "mean_metrics": classification_metrics(frame["y_true"], frame["y_pred"], frame["y_proba"]),
            "confidence_accuracy": confidence_accuracy_table(
                frame["timestamp"],
                frame["y_true"],
                frame["y_proba"],
            ),
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
