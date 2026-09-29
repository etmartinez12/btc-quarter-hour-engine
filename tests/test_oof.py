import pandas as pd
import pytest

from btc_quarter_hour_engine.validation.oof import align_oof_predictions, validate_oof_predictions


def test_validate_oof_predictions_rejects_missing_and_duplicate_keys():
    with pytest.raises(ValueError, match="must not be empty"):
        validate_oof_predictions({})

    bad_missing_timestamp = {"model": pd.DataFrame({"fold_number": [0], "y_true": [1], "y_pred": [1], "y_proba": [0.9]})}
    with pytest.raises(ValueError, match="missing required columns"):
        validate_oof_predictions(bad_missing_timestamp)

    bad_missing_fold = {"model": pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01T00:00:00Z")], "y_true": [1], "y_pred": [1], "y_proba": [0.9]})}
    with pytest.raises(ValueError, match="missing required columns"):
        validate_oof_predictions(bad_missing_fold)

    duplicate_keys = {"model": pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")], "fold_number": [0, 0], "y_true": [1, 1], "y_pred": [1, 1], "y_proba": [0.9, 0.9]})}
    with pytest.raises(ValueError, match="duplicate \(timestamp, fold_number\)"):
        validate_oof_predictions(duplicate_keys)

    cross_fold_duplicate = {"model": pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")], "fold_number": [0, 1], "y_true": [1, 0], "y_pred": [1, 0], "y_proba": [0.8, 0.2]})}
    with pytest.raises(ValueError, match="duplicate timestamps across folds"):
        validate_oof_predictions(cross_fold_duplicate)


def test_align_oof_predictions_rejects_target_mismatch_and_succeeds_on_clean_data():
    canonical = {
        "model_a": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:15:00Z")],
                "fold_number": [0, 0],
                "y_true": [1, 0],
                "y_pred": [1, 0],
                "y_proba": [0.9, 0.2],
                "log_return_15m": [0.01, -0.02],
                "midpoint": [100.0, 99.0],
                "price_t_minus_15m": [99.0, 100.0],
                "return_1m": [0.001, -0.002],
                "return_5m": [0.005, -0.006],
            }
        )
    }

    with pytest.raises(ValueError, match="disagree with the canonical frame"):
        align_oof_predictions(
            {
                **canonical,
                "model_b": pd.DataFrame(
                    {
                        "timestamp": canonical["model_a"]["timestamp"].tolist(),
                        "fold_number": canonical["model_a"]["fold_number"].tolist(),
                        "y_true": [0, 1],
                        "y_pred": [0, 1],
                        "y_proba": [0.2, 0.8],
                    }
                ),
            }
        )

    aligned = align_oof_predictions(
        {
            "model_a": canonical["model_a"],
            "model_b": pd.DataFrame(
                {
                    "timestamp": canonical["model_a"]["timestamp"].tolist(),
                    "fold_number": canonical["model_a"]["fold_number"].tolist(),
                    "y_true": [1, 0],
                    "y_pred": [1, 0],
                    "y_proba": [0.8, 0.1],
                }
            ),
        }
    )

    assert aligned.columns.tolist() == [
        "timestamp",
        "fold_number",
        "y_true",
        "log_return_15m",
        "midpoint",
        "price_t_minus_15m",
        "return_1m",
        "return_5m",
        "model_a_pred",
        "model_a_proba",
        "model_b_pred",
        "model_b_proba",
    ]
    assert "y_true_x" not in aligned.columns
    assert "y_true_y" not in aligned.columns
    assert not aligned.duplicated(subset=["timestamp", "fold_number"]).any()


def test_align_oof_predictions_keeps_common_intersection_and_preserves_metadata():
    model_a = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:15:00Z")],
            "fold_number": [0, 0],
            "y_true": [1, 0],
            "y_pred": [1, 0],
            "y_proba": [0.9, 0.1],
            "log_return_15m": [0.01, -0.02],
            "midpoint": [100.0, 99.0],
            "price_t_minus_15m": [99.0, 100.0],
            "return_1m": [0.001, -0.002],
            "return_5m": [0.005, -0.006],
        }
    )
    model_b = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01T00:15:00Z")],
            "fold_number": [0],
            "y_true": [0],
            "y_pred": [0],
            "y_proba": [0.1],
        }
    )

    aligned = align_oof_predictions({"model_a": model_a, "model_b": model_b})

    assert len(aligned) == 1
    assert list(aligned["timestamp"]) == [pd.Timestamp("2024-01-01T00:15:00Z")]
    assert len(aligned.columns) == 12
    assert aligned["model_b_pred"].tolist() == [0]
    assert aligned["model_b_proba"].tolist() == [0.1]
