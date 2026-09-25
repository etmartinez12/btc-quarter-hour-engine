import numpy as np

from btc_quarter_hour_engine.validation.walk_forward import ExpandingWindowSplit


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
