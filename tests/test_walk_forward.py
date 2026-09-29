import numpy as np
import pandas as pd
import pytest

from btc_quarter_hour_engine.config import ResearchValidationConfig
from btc_quarter_hour_engine.validation.walk_forward import ExpandingTimeWindowSplit, ExpandingWindowSplit


def test_expanding_window_split_preserves_temporal_order_without_overlap():
    splitter = ExpandingWindowSplit(initial_train_size=4, test_size=2, step_size=2)

    splits = list(splitter.split(np.arange(10)))

    assert len(splits) == 3
    assert splits[0][0].tolist() == [0, 1, 2, 3]
    assert splits[0][1].tolist() == [4, 5]
    assert splits[1][0].tolist() == [0, 1, 2, 3, 4, 5]
    assert splits[1][1].tolist() == [6, 7]
    assert splits[2][0].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert splits[2][1].tolist() == [8, 9]

    for train_idx, test_idx in splits:
        assert max(train_idx) < min(test_idx)
        assert set(train_idx).isdisjoint(set(test_idx))


def test_expanding_window_split_rejects_non_positive_step_size():
    splitter = ExpandingWindowSplit(initial_train_size=4, test_size=2, step_size=0)

    try:
        list(splitter.split(np.arange(10)))
    except ValueError as exc:
        assert "step_size must be positive" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-positive step_size")


def test_expanding_window_split_rejects_overlapping_oof_windows():
    splitter = ExpandingWindowSplit(initial_train_size=8, test_size=4, step_size=2)

    with pytest.raises(ValueError, match="step_size must be greater than or equal to test_size"):
        list(splitter.split(np.arange(20)))


def test_expanding_window_split_accepts_equal_and_larger_steps():
    step_equal = ExpandingWindowSplit(initial_train_size=8, test_size=4, step_size=4)
    splits_equal = list(step_equal.split(np.arange(20)))
    assert len(splits_equal) == 3

    step_larger = ExpandingWindowSplit(initial_train_size=8, test_size=4, step_size=6)
    splits_larger = list(step_larger.split(np.arange(20)))
    assert len(splits_larger) == 2
    for train_idx, test_idx in splits_larger:
        assert max(train_idx) < min(test_idx)


def test_expanding_window_split_test_indices_are_unique_across_folds():
    splitter = ExpandingWindowSplit(initial_train_size=8, test_size=4, step_size=4)
    splits = list(splitter.split(np.arange(20)))

    all_test_indices = np.concatenate([test_idx for _, test_idx in splits])
    assert len(all_test_indices) == len(np.unique(all_test_indices))


def test_expanding_time_window_split_enforces_temporal_order_and_unique_oof_windows():
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=288, freq="15min")
    splitter = ExpandingTimeWindowSplit(initial_train_period="1D", test_period="6H", step_period="6H")

    splits = list(splitter.split(timestamps))
    assert len(splits) > 0

    previous_train_len = None
    all_test_indices = []
    for train_idx, test_idx in splits:
        assert timestamps[train_idx].max() < timestamps[test_idx].min()
        assert len(np.unique(np.concatenate([train_idx, test_idx]))) == len(train_idx) + len(test_idx)
        assert len(np.concatenate([test_idx])) == len(np.unique(np.concatenate([test_idx])))
        all_test_indices.extend(test_idx.tolist())
        train_len = len(train_idx)
        if previous_train_len is not None:
            assert train_len >= previous_train_len
        previous_train_len = train_len

    assert len(all_test_indices) == len(np.unique(all_test_indices))


def test_expanding_time_window_split_rejects_overlapping_research_windows():
    with pytest.raises(ValueError, match="step_period must be greater than or equal to test_period"):
        list(
            ExpandingTimeWindowSplit(initial_train_period="1D", test_period="6H", step_period="3H").split(
                pd.date_range("2026-01-01T00:00:00Z", periods=96, freq="15min")
            )
        )


def test_expanding_time_window_split_rejects_zero_and_negative_periods():
    with pytest.raises(ValueError):
        list(
            ExpandingTimeWindowSplit(initial_train_period="1D", test_period="0H", step_period="6H").split(
                pd.date_range("2026-01-01T00:00:00Z", periods=96, freq="15min")
            )
        )

    with pytest.raises(ValueError):
        list(
            ExpandingTimeWindowSplit(initial_train_period="1D", test_period="6H", step_period="0H").split(
                pd.date_range("2026-01-01T00:00:00Z", periods=96, freq="15min")
            )
        )

    with pytest.raises(ValueError):
        list(
            ExpandingTimeWindowSplit(initial_train_period="-1D", test_period="6H", step_period="6H").split(
                pd.date_range("2026-01-01T00:00:00Z", periods=96, freq="15min")
            )
        )


def test_expanding_time_window_split_rejects_unsorted_or_invalid_timestamps():
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=20, freq="15min")
    shuffled = timestamps[::-1].copy()
    with pytest.raises(ValueError, match="timestamps must be sorted chronologically"):
        list(ExpandingTimeWindowSplit(initial_train_period="1D", test_period="6H", step_period="6H").split(shuffled))

    with pytest.raises((ValueError, TypeError)):
        list(
            ExpandingTimeWindowSplit(initial_train_period="1D", test_period="6H", step_period="6H").split(
                ["2026-01-01T00:00:00Z", "bad-value"]
            )
        )


def test_expanding_time_window_split_empty_input_yields_zero_folds():
    assert list(ExpandingTimeWindowSplit(initial_train_period="1D", test_period="6H", step_period="6H").split([])) == []


def test_research_validation_config_constructs_time_window_split():
    config = ResearchValidationConfig(initial_train_period="1D", test_period="6H", step_period="6H")
    splitter = ExpandingTimeWindowSplit.from_config(config)

    assert splitter.initial_train_period == config.initial_train_period
    assert splitter.test_period == config.test_period
    assert splitter.step_period == config.step_period
