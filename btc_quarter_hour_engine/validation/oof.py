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
        missing = [column for column in ["timestamp", "fold_number"] if column not in frame.columns]
        if missing:
            raise ValueError(f"{model_name} OOF predictions are missing required columns: {missing}")

        duplicate_keys = frame.duplicated(subset=["timestamp", "fold_number"], keep=False)
        if duplicate_keys.any():
            dup_rows = frame.loc[duplicate_keys, ["timestamp", "fold_number"]].drop_duplicates().to_dict("records")
            raise ValueError(f"{model_name} OOF predictions contain duplicate (timestamp, fold_number) rows: {dup_rows}")

        timestamp_fold_counts = frame.groupby("timestamp")["fold_number"].nunique()
        duplicate_timestamps = timestamp_fold_counts[timestamp_fold_counts > 1].index.tolist()
        if duplicate_timestamps:
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

    metadata_columns = [
        column
        for column in [
            "timestamp",
            "fold_number",
            "y_true",
            "log_return_15m",
            "midpoint",
            "price_t_minus_15m",
            "return_1m",
            "return_5m",
        ]
        if column in first_frame.columns
    ]
    aligned = first_frame[metadata_columns + ["y_pred", "y_proba"]].copy().rename(
        columns={"y_pred": f"{first_name}_pred", "y_proba": f"{first_name}_proba"}
    )

    for model_name in model_names[1:]:
        frame = model_oof_predictions[model_name].copy()
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"{model_name} OOF frame is missing required columns: {missing}")

        compare = aligned[["timestamp", "fold_number", "y_true"]].merge(
            frame[["timestamp", "fold_number", "y_true"]].rename(columns={"y_true": f"{model_name}_y_true"}),
            on=["timestamp", "fold_number"],
            how="inner",
        )
        mismatched = compare.loc[compare["y_true"] != compare[f"{model_name}_y_true"], ["timestamp", "fold_number"]]
        if not mismatched.empty:
            mismatch_records = mismatched.drop_duplicates().to_dict("records")
            raise ValueError(f"{model_name} OOF targets disagree with the canonical frame at: {mismatch_records}")

        merged = frame[["timestamp", "fold_number", "y_pred", "y_proba"]].rename(
            columns={"y_pred": f"{model_name}_pred", "y_proba": f"{model_name}_proba"}
        )
        aligned = aligned.merge(merged, on=["timestamp", "fold_number"], how="inner")

    if "y_true" not in aligned.columns:
        raise ValueError("Aligned OOF predictions lost the canonical y_true column")

    if aligned.duplicated(subset=["timestamp", "fold_number"]).any():
        raise ValueError("Aligned OOF predictions still contain duplicate (timestamp, fold_number) rows")

    return aligned.sort_values(["fold_number", "timestamp"]).reset_index(drop=True)
