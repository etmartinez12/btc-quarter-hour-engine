import pandas as pd
import pytest

from btc_quarter_hour_engine.validation.naive_baselines import (
    always_down,
    always_up,
    momentum_1m,
    momentum_5m,
    previous_quarter_direction,
    random_50,
    score_naive_baselines,
)


@pytest.fixture
def common_oof_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T00:15:00Z",
                    "2024-01-01T00:30:00Z",
                    "2024-01-01T00:45:00Z",
                ],
                utc=True,
            ),
            "fold_number": [0, 0, 0, 0],
            "y_true": [1, 0, 1, 0],
            "log_return_15m": [0.01, -0.02, 0.03, -0.04],
            "midpoint": [100.0, 99.0, 101.0, 100.5],
            "price_t_minus_15m": [95.0, 100.0, 98.0, 101.0],
            "return_1m": [0.001, -0.002, 0.003, -0.004],
            "return_5m": [0.005, -0.006, 0.007, -0.008],
        }
    )


def test_always_up_predictions_are_all_one(common_oof_frame):
    preds = always_up(common_oof_frame)
    assert preds.tolist() == [1, 1, 1, 1]


def test_always_down_predictions_are_all_zero(common_oof_frame):
    preds = always_down(common_oof_frame)
    assert preds.tolist() == [0, 0, 0, 0]


def test_random_50_is_deterministic_after_timestamp_sorting(common_oof_frame):
    shuffled = common_oof_frame.iloc[[2, 0, 3, 1]].copy().reset_index(drop=True)

    first = random_50(common_oof_frame, random_state=42)
    second = random_50(common_oof_frame, random_state=42)
    shuffled_out = random_50(shuffled, random_state=42)

    assert first.tolist() == second.tolist()
    assert shuffled_out.tolist() == [0, 1, 0, 0]


def test_previous_quarter_direction_exact_histories_and_missing_guard(common_oof_frame):
    preds = previous_quarter_direction(common_oof_frame)
    assert preds.tolist() == [1, 0, 1, 0]

    frame_with_equal_history = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01T00:00:00Z")],
            "midpoint": [100.0],
            "price_t_minus_15m": [100.0],
        }
    )
    assert previous_quarter_direction(frame_with_equal_history).tolist() == [0]

    with pytest.raises(KeyError, match="exact price_t_minus_15m"):
        previous_quarter_direction(common_oof_frame.drop(columns=["price_t_minus_15m"]))


def test_momentum_rules_match_positive_zero_and_negative_cases():
    frame = pd.DataFrame(
        {"return_1m": [0.01, 0.0, -0.01], "return_5m": [0.05, 0.0, -0.05]}
    )

    assert momentum_1m(frame).tolist() == [1, 0, 0]
    assert momentum_5m(frame).tolist() == [1, 0, 0]


def test_score_naive_baselines_has_hard_rules_and_no_probability_metrics(common_oof_frame):
    scored = score_naive_baselines(common_oof_frame)

    assert set(scored) == {
        "always_up",
        "always_down",
        "random_50",
        "previous_quarter_direction",
        "momentum_1m",
        "momentum_5m",
    }
    for name, summary in scored.items():
        assert "metrics" in summary
        assert summary["n_predictions"] == len(common_oof_frame)
        assert set(summary["metrics"]) == {"accuracy", "precision", "recall", "f1"}
        assert summary["confidence_accuracy"] is None
        assert len(summary["accuracy_by_move_size"]) == 6
        assert len(summary["accuracy_by_boundary_slot"]) == 4
