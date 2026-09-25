import pandas as pd

from btc_quarter_hour_engine.ensemble import build_validation_weighted_ensemble


def test_validation_weighted_ensemble_uses_only_prior_fold_performance():
    aligned = pd.DataFrame(
        {
            "fold_number": [0, 0, 1, 1],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:15:00Z",
                    "2026-01-01T00:30:00Z",
                    "2026-01-01T00:45:00Z",
                ],
                utc=True,
            ),
            "y_true": [1, 0, 1, 0],
            "model_a_pred": [1, 0, 1, 1],
            "model_a_proba": [0.9, 0.1, 0.8, 0.7],
            "model_b_pred": [0, 1, 0, 0],
            "model_b_proba": [0.4, 0.6, 0.3, 0.2],
            "log_return_15m": [0.01, -0.01, 0.02, -0.02],
        }
    )

    ensemble, weight_history = build_validation_weighted_ensemble(
        aligned,
        ["model_a", "model_b"],
        metric="accuracy",
    )

    assert weight_history[0]["model_a_weight"] == 0.5
    assert weight_history[0]["model_b_weight"] == 0.5
    assert weight_history[1]["model_a_weight"] > weight_history[1]["model_b_weight"]
    fold_one_rows = ensemble.loc[ensemble["fold_number"] == 1]
    assert (fold_one_rows["y_proba"] > aligned.loc[aligned["fold_number"] == 1, "model_b_proba"]).all()
