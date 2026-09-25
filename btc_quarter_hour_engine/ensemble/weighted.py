from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss


def _normalized_weights(raw_weights: dict[str, float], min_weight: float) -> dict[str, float]:
    clipped = {name: max(float(weight), min_weight) for name, weight in raw_weights.items()}
    total = sum(clipped.values())
    return {name: weight / total for name, weight in clipped.items()}


def build_simple_average_ensemble(frame: pd.DataFrame, model_names: Sequence[str]) -> pd.DataFrame:
    probability_columns = [f"{model_name}_proba" for model_name in model_names]
    ensemble = frame.copy()
    ensemble["ensemble_name"] = "simple_average"
    ensemble["y_proba"] = ensemble[probability_columns].mean(axis=1)
    ensemble["y_pred"] = (ensemble["y_proba"] >= 0.5).astype(int)
    return ensemble


def build_validation_weighted_ensemble(
    frame: pd.DataFrame,
    model_names: Sequence[str],
    metric: str = "accuracy",
    min_weight: float = 1e-6,
) -> tuple[pd.DataFrame, list[dict[str, float | int]]]:
    if metric not in {"accuracy", "brier"}:
        raise ValueError("metric must be 'accuracy' or 'brier'")

    probability_columns = [f"{model_name}_proba" for model_name in model_names]
    prediction_columns = [f"{model_name}_pred" for model_name in model_names]
    weighted = frame.copy().sort_values(["fold_number", "timestamp"]).reset_index(drop=True)
    weighted["ensemble_name"] = "validation_weighted_average"
    weighted["y_proba"] = np.nan

    fold_weight_history: list[dict[str, float | int]] = []
    unique_folds = sorted(weighted["fold_number"].unique())

    for fold_number in unique_folds:
        history = weighted.loc[weighted["fold_number"] < fold_number]
        if history.empty:
            weights = {model_name: 1.0 for model_name in model_names}
        else:
            if metric == "accuracy":
                weights = {
                    model_name: accuracy_score(history["y_true"], history[f"{model_name}_pred"])
                    for model_name in model_names
                }
            else:
                weights = {
                    model_name: 1.0 - brier_score_loss(history["y_true"], history[f"{model_name}_proba"])
                    for model_name in model_names
                }
        normalized_weights = _normalized_weights(weights, min_weight=min_weight)
        fold_weight_history.append(
            {"fold_number": int(fold_number), **{f"{model_name}_weight": normalized_weights[model_name] for model_name in model_names}}
        )

        fold_mask = weighted["fold_number"].eq(fold_number)
        weighted.loc[fold_mask, "y_proba"] = sum(
            weighted.loc[fold_mask, column] * normalized_weights[model_name]
            for model_name, column in zip(model_names, probability_columns, strict=False)
        )

    weighted["y_pred"] = (weighted["y_proba"] >= 0.5).astype(int)
    return weighted, fold_weight_history


def build_consensus_diagnostics(frame: pd.DataFrame, model_names: Sequence[str]) -> dict[str, object]:
    probability_columns = [f"{model_name}_proba" for model_name in model_names]
    prediction_columns = [f"{model_name}_pred" for model_name in model_names]

    diagnostics = frame.copy()
    diagnostics["probability_mean"] = diagnostics[probability_columns].mean(axis=1)
    diagnostics["probability_std"] = diagnostics[probability_columns].std(axis=1, ddof=0)
    diagnostics["max_probability_spread"] = diagnostics[probability_columns].max(axis=1) - diagnostics[probability_columns].min(axis=1)
    diagnostics["votes_up"] = diagnostics[prediction_columns].sum(axis=1)
    diagnostics["votes_down"] = len(model_names) - diagnostics["votes_up"]
    diagnostics["vote_margin"] = (diagnostics["votes_up"] - diagnostics["votes_down"]).abs()
    diagnostics["consensus_pred"] = (diagnostics["votes_up"] >= (len(model_names) / 2.0)).astype(int)

    consensus_rows: list[dict[str, float | int]] = []
    for votes_up in range(len(model_names) + 1):
        subset = diagnostics.loc[diagnostics["votes_up"] == votes_up]
        consensus_rows.append(
            {
                "votes_up": int(votes_up),
                "n_predictions": int(len(subset)),
                "accuracy": float(accuracy_score(subset["y_true"], subset["consensus_pred"])) if len(subset) else float("nan"),
                "mean_probability_std": float(subset["probability_std"].mean()) if len(subset) else float("nan"),
            }
        )

    return {
        "summary": {
            "mean_probability_std": float(diagnostics["probability_std"].mean()),
            "mean_vote_margin": float(diagnostics["vote_margin"].mean()),
            "mean_max_probability_spread": float(diagnostics["max_probability_spread"].mean()),
        },
        "accuracy_by_votes_up": consensus_rows,
    }
