import numpy as np
import pandas as pd

from btc_quarter_hour_engine.research.features import (
    PREDICTOR_COLUMNS,
)
from btc_quarter_hour_engine.research.walk_forward import (
    ExploratoryResearchConfig,
    build_exploratory_folds,
    prepare_research_frame,
    run_exploratory_walk_forward,
)


def _synthetic_research_frame(periods=500):
    timestamps = pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=periods,
        freq="15min",
    )

    data = {}

    for idx, column in enumerate(PREDICTOR_COLUMNS):
        values = (
            np.arange(periods, dtype=float)
            + 1.0
            + idx
        )

        if column == "asof_book_synced":
            values = np.ones(periods, dtype=float)

        if column.startswith("momentum_log_return_"):
            values = np.sin(
                np.arange(periods) / 7.0
            )

        data[column] = values

    frame = pd.DataFrame(data)

    frame["timestamp"] = timestamps
    frame["label_available_time"] = (
        timestamps + pd.Timedelta(minutes=15)
    )
    frame["target"] = (
        np.arange(periods) % 2
    ).astype(int)
    frame["target_log_return_15m"] = np.where(
        frame["target"].eq(1),
        0.001,
        -0.001,
    )
    frame["trainable"] = True

    return frame


def test_research_folds_enforce_strict_label_embargo():
    frame, _, _ = prepare_research_frame(
        _synthetic_research_frame()
    )

    config = ExploratoryResearchConfig(
        initial_train_period="72h",
        test_period="12h",
        step_period="12h",
    )

    folds = build_exploratory_folds(
        frame,
        config,
    )

    assert folds

    all_test = []

    for fold in folds:
        train = frame.iloc[fold["train_idx"]]
        test = frame.iloc[fold["test_idx"]]

        test_start = test["timestamp"].min()

        assert (
            train["label_available_time"].max()
            < test_start
        )

        assert set(fold["train_idx"]).isdisjoint(
            fold["test_idx"]
        )

        all_test.extend(
            fold["test_idx"].tolist()
        )

    assert len(all_test) == len(set(all_test))


def test_strict_embargo_excludes_label_available_at_test_start():
    frame, _, _ = prepare_research_frame(
        _synthetic_research_frame()
    )

    config = ExploratoryResearchConfig(
        initial_train_period="72h",
        test_period="12h",
        step_period="12h",
    )

    first = build_exploratory_folds(
        frame,
        config,
    )[0]

    train = frame.iloc[first["train_idx"]]
    test = frame.iloc[first["test_idx"]]

    test_start = test["timestamp"].min()

    assert not (
        train["label_available_time"] == test_start
    ).any()

    candidate = frame.loc[
        frame["label_available_time"].eq(test_start)
    ]

    assert len(candidate) == 1
    assert candidate.index[0] not in first["train_idx"]


def test_research_walk_forward_produces_common_oof_population():
    frame = _synthetic_research_frame()

    config = ExploratoryResearchConfig(
        initial_train_period="72h",
        test_period="12h",
        step_period="12h",
        extra_trees_n_estimators=10,
    )

    summary, predictions = run_exploratory_walk_forward(
        frame,
        config,
    )

    assert summary["predictor_count"] == 114
    assert summary["n_folds"] > 0

    expected = summary[
        "n_oof_predictions_per_benchmark"
    ]

    counts = (
        predictions.groupby("benchmark")
        .size()
        .to_dict()
    )

    assert counts
    assert set(counts.values()) == {expected}

    for _, group in predictions.groupby("benchmark"):
        assert not group["timestamp"].duplicated().any()


def test_causal_momentum_baseline_uses_only_causal_feature():
    frame = _synthetic_research_frame()

    config = ExploratoryResearchConfig(
        initial_train_period="72h",
        test_period="12h",
        step_period="12h",
        extra_trees_n_estimators=10,
    )

    _, predictions = run_exploratory_walk_forward(
        frame,
        config,
    )

    momentum = predictions.loc[
        predictions["benchmark"].eq(
            "causal_momentum_15m"
        )
    ]

    expected = (
        frame.set_index("timestamp")
        .loc[
            pd.to_datetime(momentum["timestamp"], utc=True),
            "momentum_log_return_15m",
        ]
        .gt(0)
        .astype(int)
        .to_numpy()
    )

    assert np.array_equal(
        momentum["y_pred"].to_numpy(),
        expected,
    )
