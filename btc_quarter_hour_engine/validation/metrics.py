from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, precision_score, recall_score, roc_auc_score


def classification_metrics(y_true, y_pred, y_proba) -> dict[str, Any]:
    unique_classes = np.unique(y_true)
    metrics: dict[str, Any] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics["brier_score"] = brier_score_loss(y_true, y_proba)
    except ValueError:
        metrics["brier_score"] = float("nan")
    if len(unique_classes) < 2:
        metrics["log_loss"] = float("nan")
        metrics["roc_auc"] = float("nan")
        return metrics

    metrics["log_loss"] = log_loss(y_true, np.column_stack([1.0 - y_proba, y_proba]), labels=[0, 1])
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics
