from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
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


def confidence_accuracy_table(
    timestamps,
    y_true,
    y_proba,
    thresholds: tuple[float, ...] = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80),
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, utc=True),
            "y_true": y_true,
            "y_proba": y_proba,
        }
    )
    frame["confidence"] = np.maximum(frame["y_proba"], 1.0 - frame["y_proba"])
    frame["y_pred"] = (frame["y_proba"] >= 0.5).astype(int)

    rows: list[dict[str, Any]] = [
        {
            "bucket": "all",
            "n_predictions": int(len(frame)),
            "accuracy": float(accuracy_score(frame["y_true"], frame["y_pred"])) if len(frame) else float("nan"),
        }
    ]
    for threshold in thresholds:
        subset = frame.loc[frame["confidence"] >= threshold]
        rows.append(
            {
                "bucket": f">={threshold:.0%}",
                "n_predictions": int(len(subset)),
                "accuracy": float(accuracy_score(subset["y_true"], subset["y_pred"])) if len(subset) else float("nan"),
            }
        )
    return rows


def accuracy_by_move_size(
    y_true,
    y_pred,
    log_return_15m,
    bins: tuple[float, ...] = (0.0, 0.0001, 0.0005, 0.001, 0.0025, 0.005, float("inf")),
    labels: tuple[str, ...] = ("<0.01%", "0.01-0.05%", "0.05-0.10%", "0.10-0.25%", "0.25-0.50%", ">0.50%"),
) -> list[dict[str, Any]]:
    magnitude = np.abs(np.exp(np.asarray(log_return_15m)) - 1.0)
    buckets = pd.cut(magnitude, bins=bins, labels=labels, include_lowest=True, right=False)
    frame = pd.DataFrame({"bucket": buckets, "y_true": y_true, "y_pred": y_pred})

    rows: list[dict[str, Any]] = []
    for bucket in labels:
        subset = frame.loc[frame["bucket"] == bucket]
        rows.append(
            {
                "bucket": bucket,
                "n_predictions": int(len(subset)),
                "accuracy": float(accuracy_score(subset["y_true"], subset["y_pred"])) if len(subset) else float("nan"),
            }
        )
    return rows


def accuracy_by_boundary_slot(timestamps, y_true, y_pred) -> list[dict[str, Any]]:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, utc=True),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    slot_labels = {
        0: ":00→:15",
        15: ":15→:30",
        30: ":30→:45",
        45: ":45→:00",
    }
    frame["slot"] = frame["timestamp"].dt.minute.map(slot_labels)

    rows: list[dict[str, Any]] = []
    for minute, slot in slot_labels.items():
        subset = frame.loc[frame["timestamp"].dt.minute.eq(minute)]
        rows.append(
            {
                "slot": slot,
                "n_predictions": int(len(subset)),
                "accuracy": float(accuracy_score(subset["y_true"], subset["y_pred"])) if len(subset) else float("nan"),
            }
        )
    return rows
