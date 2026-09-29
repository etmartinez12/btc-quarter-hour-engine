from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def validate_oof_predictions(model_oof_predictions: Mapping[str, pd.DataFrame]) -> None:
    """Validate alignment and uniqueness assumptions for OOF prediction tables."""
    if not model_oof_predictions:
        raise ValueError("model_oof_predictions must not be empty")

    for model_name, frame in model_oof_predictions.items():
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{model_name} OOF predictions must be a pandas.DataFrame")
        if "timestamp" not in frame.columns:
            raise ValueError(f"{model_name} OOF predictions are missing the timestamp column")
        if {"timestamp", "fold_number"}.issubset(frame.columns):
            duplicate_keys = frame.duplicated(subset=["timestamp", "fold_number"], keep=False)
            if duplicate_keys.any():
                dup_rows = frame.loc[duplicate_keys, ["timestamp", "fold_number"]].drop_duplicates().to_dict("records")
                raise ValueError(f"{model_name} OOF predictions contain duplicate (timestamp, fold_number) rows: {dup_rows}")
            if frame["timestamp"].duplicated(keep=False).any():
                duplicate_timestamps = frame.loc[frame["timestamp"].duplicated(keep=False), "timestamp"].drop_duplicates().tolist()
                raise ValueError(f"{model_name} OOF predictions contain duplicate timestamps across folds: {duplicate_timestamps}")


def align_oof_predictions(model_oof_predictions: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Align model OOF predictions on the common (timestamp, fold_number) key."""
    if not model_oof_predictions:
        raise ValueError("model_oof_predictions must not be empty")

    validate_oof_predictions(model_oof_predictions)
    model_names = list(model_oof_predictions)
    first_name = model_names[0]
    first_frame = model_oof_predictions[first_name].copy()
    required = ["timestamp", "fold_number", "y_true", "y_pred", "y_proba"]
    missing = [column for column in required if column not in first_frame.columns]
    if missing:
        raise ValueError(f"{first_name} OOF frame is missing required columns: {missing}")

    aligned = first_frame[["timestamp", "fold_number", "y_true", "y_pred", "y_proba"]].copy().rename(
        columns={"y_pred": f"{first_name}_pred", "y_proba": f"{first_name}_proba"}
    )

    for model_name in model_names[1:]:
        frame = model_oof_predictions[model_name].copy()
        missing = [column for column in ["timestamp", "fold_number", "y_true", "y_pred", "y_proba"] if column not in frame.columns]
        if missing:
            raise ValueError(f"{model_name} OOF frame is missing required columns: {missing}")

        if not frame[["timestamp", "fold_number"]].drop_duplicates().shape[0] == len(frame):
            raise ValueError(f"{model_name} OOF predictions contain duplicate (timestamp, fold_number) rows")

        merged = frame[["timestamp", "fold_number", "y_true", "y_pred", "y_proba"]].rename(
            columns={"y_pred": f"{model_name}_pred", "y_proba": f"{model_name}_proba"}
        )
        aligned = aligned.merge(merged, on=["timestamp", "fold_number"], how="inner")

        if not aligned["y_true_x"].equals(aligned["y_true_y"]) if "y_true_y" in aligned.columns else False:
            pass

    if "y_true_x" in aligned.columns:
        aligned = aligned.rename(columns={"y_true_x": "y_true"})
        aligned = aligned.drop(columns=[column for column in aligned.columns if column == "y_true_y"])
    else:
        aligned = aligned.rename(columns={"y_true": "y_true"})

    if aligned.duplicated(subset=["timestamp", "fold_number"]).any():
        raise ValueError("Aligned OOF predictions still contain duplicate (timestamp, fold_number) rows")

    return aligned.sort_values(["fold_number", "timestamp"]).reset_index(drop=True)
