import pandas as pd

from btc_quarter_hour_engine.ensemble import build_consensus_diagnostics, build_validation_weighted_ensemble


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


def test_build_consensus_diagnostics_handles_ties_and_non_ties():
    base = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:15:00Z",
                    "2026-01-01T00:30:00Z",
                    "2026-01-01T00:45:00Z",
                    "2026-01-01T01:00:00Z",
                ],
                utc=True,
            ),
            "fold_number": [0, 0, 0, 0, 0],
            "y_true": [1, 1, 0, 0, 1],
            "model_a_pred": [1, 1, 0, 0, 1],
            "model_a_proba": [0.9, 0.7, 0.2, 0.1, 0.8],
            "model_b_pred": [1, 1, 0, 0, 1],
            "model_b_proba": [0.8, 0.8, 0.3, 0.2, 0.7],
            "model_c_pred": [1, 1, 0, 0, 0],
            "model_c_proba": [0.7, 0.9, 0.1, 0.3, 0.4],
            "model_d_pred": [1, 0, 0, 1, 0],
            "model_d_proba": [0.6, 0.4, 0.2, 0.9, 0.3],
        }
    )

    vote_patterns = {
        "4-0": base.loc[[0]].copy(),
        "3-1": base.loc[[1]].copy(),
        "2-2": base.loc[[2]].copy(),
        "1-3": base.loc[[3]].copy(),
        "0-4": base.loc[[4]].copy(),
    }
    vote_patterns["3-1"]["model_d_pred"] = 0
    vote_patterns["3-1"]["model_d_proba"] = 0.4
    vote_patterns["2-2"]["model_a_pred"] = 1
    vote_patterns["2-2"]["model_b_pred"] = 1
    vote_patterns["2-2"]["model_c_pred"] = 0
    vote_patterns["2-2"]["model_d_pred"] = 0
    vote_patterns["2-2"]["model_a_proba"] = 0.6
    vote_patterns["2-2"]["model_b_proba"] = 0.65
    vote_patterns["2-2"]["model_c_proba"] = 0.45
    vote_patterns["2-2"]["model_d_proba"] = 0.35
    vote_patterns["1-3"]["model_a_pred"] = 0
    vote_patterns["1-3"]["model_b_pred"] = 0
    vote_patterns["1-3"]["model_c_pred"] = 0
    vote_patterns["1-3"]["model_d_pred"] = 1
    vote_patterns["1-3"]["model_a_proba"] = 0.2
    vote_patterns["1-3"]["model_b_proba"] = 0.3
    vote_patterns["1-3"]["model_c_proba"] = 0.1
    vote_patterns["1-3"]["model_d_proba"] = 0.7
    vote_patterns["0-4"]["model_a_pred"] = 0
    vote_patterns["0-4"]["model_b_pred"] = 0
    vote_patterns["0-4"]["model_c_pred"] = 0
    vote_patterns["0-4"]["model_d_pred"] = 0
    vote_patterns["0-4"]["model_a_proba"] = 0.2
    vote_patterns["0-4"]["model_b_proba"] = 0.1
    vote_patterns["0-4"]["model_c_proba"] = 0.3
    vote_patterns["0-4"]["model_d_proba"] = 0.25

    for votes_up, frame in {
        4: vote_patterns["4-0"],
        3: vote_patterns["3-1"],
        2: vote_patterns["2-2"],
        1: vote_patterns["1-3"],
        0: vote_patterns["0-4"],
    }.items():
        metrics = build_consensus_diagnostics(frame, ["model_a", "model_b", "model_c", "model_d"])
        accuracy_row = next(row for row in metrics["accuracy_by_votes_up"] if row["votes_up"] == votes_up)
        if votes_up in {4, 3}:
            assert accuracy_row["is_tie"] is False
            assert accuracy_row["accuracy"] is not None
        elif votes_up == 2:
            assert accuracy_row["is_tie"] is True
            assert accuracy_row["accuracy"] is None
        elif votes_up in {1, 0}:
            assert accuracy_row["is_tie"] is False
            assert accuracy_row["accuracy"] is not None
