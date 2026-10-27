from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from btc_quarter_hour_engine.research.diagnostics_cli import (
    main as diagnostics_main,
)
from btc_quarter_hour_engine.research.features import (
    PREDICTOR_COLUMNS,
    PREDICTOR_CONTRACT_VERSION,
    research_model_matrix,
)
from btc_quarter_hour_engine.research.features_v2 import (
    V2_PREDICTOR_COLUMNS,
    V2_PREDICTOR_CONTRACT_VERSION,
    build_v2_research_feature_frame,
)
from btc_quarter_hour_engine.research.walk_forward import (
    ExploratoryResearchConfig,
    build_exploratory_folds,
    prepare_research_frame,
    run_exploratory_walk_forward,
)


def _v1_features(periods: int = 4) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-01-03T06:15:00Z",
        periods=periods,
        freq="15min",
    )
    data = {
        column: np.arange(periods, dtype=float) + index + 1.0
        for index, column in enumerate(PREDICTOR_COLUMNS)
    }

    for column in PREDICTOR_COLUMNS:
        if column.startswith("momentum_log_return_"):
            data[column] = np.linspace(-0.02, 0.03, periods)
    data["asof_book_synced"] = np.ones(periods, dtype=float)
    data["asof_midpoint"] = np.full(periods, 50_000.0)
    data["asof_spread_bps"] = np.full(periods, 8.0)
    data["asof_best_bid_size"] = np.full(periods, 2.0)
    data["asof_best_ask_size"] = np.full(periods, 3.0)
    data["asof_top_book_imbalance"] = np.full(periods, -0.2)

    for minutes in (15, 30, 60, 240):
        minimum = np.full(periods, 100.0)
        if minutes == 15:
            maximum = np.full(periods, 101.0)
        elif minutes == 30:
            maximum = np.full(periods, 102.0)
        elif minutes == 60:
            maximum = np.full(periods, 104.0)
        else:
            maximum = np.full(periods, 108.0)
        data[f"bbo_midpoint_min_{minutes}m"] = minimum
        data[f"bbo_midpoint_max_{minutes}m"] = maximum

        data[f"bbo_spread_mean_bps_{minutes}m"] = np.full(
            periods,
            minutes / 10.0,
        )
        data[f"bbo_spread_std_bps_{minutes}m"] = np.full(
            periods,
            minutes / 20.0,
        )
        data[f"bbo_spread_max_bps_{minutes}m"] = np.full(
            periods,
            minutes / 5.0,
        )
        data[f"bbo_imbalance_mean_{minutes}m"] = np.full(
            periods,
            {15: 0.5, 30: 0.4, 60: 0.2, 240: 0.1}[minutes],
        )
        data[f"bbo_imbalance_std_{minutes}m"] = np.full(
            periods,
            minutes / 1000.0,
        )
        data[f"distance_from_high_{minutes}m"] = np.full(
            periods,
            -minutes / 100.0,
        )
        data[f"distance_from_low_{minutes}m"] = np.full(
            periods,
            minutes / 100.0,
        )
        data[f"l2_update_imbalance_{minutes}m"] = np.full(
            periods,
            minutes / 200.0,
        )
        data[f"l2_delete_imbalance_{minutes}m"] = np.full(
            periods,
            -minutes / 200.0,
        )

        rates = {
            "bbo_state_count": 2.0 if minutes < 240 else 1.0,
            "midpoint_change_count": 4.0 if minutes < 240 else 1.0,
            "l2_envelope_count": 3.0,
            "l2_update_count": 2.0 if minutes < 240 else 1.0,
            "l2_delete_count": 1.0 if minutes < 240 else 0.5,
        }
        for count_name, rate in rates.items():
            data[f"{count_name}_{minutes}m"] = np.full(
                periods,
                minutes * rate,
            )

    frame = pd.DataFrame(data)
    frame["timestamp"] = timestamps
    frame["feature_ready"] = True
    frame["target_eligible"] = True
    frame["trainable"] = True
    frame["target"] = (np.arange(periods) % 2).astype(int)
    frame["target_log_return_15m"] = np.where(
        frame["target"].eq(1),
        0.001,
        -0.001,
    )
    frame["target_time"] = timestamps + pd.Timedelta(minutes=15)
    frame["label_available_time"] = timestamps + pd.Timedelta(minutes=15)
    frame["canonical_midpoint"] = np.full(periods, 999_999.0)
    frame["price_change"] = np.full(periods, 123.0)
    return frame


def test_v1_contract_remains_the_default_114_predictors():
    frame = _v1_features()
    X, _ = research_model_matrix(frame)

    assert PREDICTOR_CONTRACT_VERSION == "bbo-causal-v1"
    assert len(PREDICTOR_COLUMNS) == 114
    assert tuple(X.columns) == PREDICTOR_COLUMNS


def test_v2_contract_has_exact_columns_and_excludes_forbidden_predictors():
    features = build_v2_research_feature_frame(_v1_features())

    assert V2_PREDICTOR_CONTRACT_VERSION == "bbo-causal-v2"
    assert len(V2_PREDICTOR_COLUMNS) == 91
    assert len(set(V2_PREDICTOR_COLUMNS)) == 91
    assert tuple(features.loc[:, list(V2_PREDICTOR_COLUMNS)].columns) == (
        V2_PREDICTOR_COLUMNS
    )
    assert not {
        "asof_midpoint",
        "asof_best_bid_size",
        "asof_best_ask_size",
        "asof_book_synced",
        "hour_utc",
        "quarter_hour_slot",
        "day_of_week",
        "target",
        "target_time",
        "label_available_time",
        "target_log_return_15m",
        "canonical_midpoint",
        "price_change",
    }.intersection(V2_PREDICTOR_COLUMNS)
    assert not any(
        name.startswith("bbo_midpoint_min_")
        or name.startswith("bbo_midpoint_max_")
        for name in V2_PREDICTOR_COLUMNS
    )
    assert not any("count" in name for name in V2_PREDICTOR_COLUMNS)
    assert not any(
        "canonical" in name
        or "coverage" in name
        or "valid_fraction" in name
        or "synced_fraction" in name
        for name in V2_PREDICTOR_COLUMNS
    )

    X, y = research_model_matrix(
        _v1_features(),
        predictor_contract_version=V2_PREDICTOR_CONTRACT_VERSION,
    )
    assert tuple(X.columns) == V2_PREDICTOR_COLUMNS
    assert len(X.columns) == 91
    assert y.tolist() == [0, 1, 0, 1]


def test_v2_normalizations_and_cross_window_deltas_are_correct():
    frame = build_v2_research_feature_frame(_v1_features())
    row = frame.iloc[0]

    assert np.isclose(
        row["midpoint_range_bps_15m"],
        np.log(101.0 / 100.0) * 10_000.0,
    )
    assert row["bbo_state_rate_per_minute_15m"] == 2.0
    assert row["l2_update_rate_per_minute_240m"] == 1.0
    assert np.isclose(
        row["log_bbo_state_rate_ratio_15m_240m"],
        np.log1p(2.0) - np.log1p(1.0),
    )
    assert row["imbalance_mean_delta_15m_60m"] == 0.3
    assert row["imbalance_mean_delta_15m_240m"] == 0.4
    assert row["imbalance_mean_delta_60m_240m"] == 0.1
    assert row["spread_mean_delta_bps_15m_60m"] == -4.5
    assert row["spread_mean_delta_bps_15m_240m"] == -22.5
    assert row["spread_std_delta_bps_15m_240m"] == -11.25
    assert np.isclose(
        row["momentum_rate_delta_15m_240m"],
        row["momentum_log_return_15m"] / 15
        - row["momentum_log_return_240m"] / 240,
    )
    assert np.isclose(
        row["midpoint_range_fraction_15m_240m"],
        row["midpoint_range_bps_15m"] / row["midpoint_range_bps_240m"],
    )
    assert np.isfinite(
        frame.loc[:, list(V2_PREDICTOR_COLUMNS)].to_numpy(dtype=float)
    ).all()


def test_v2_zero_range_ratio_is_deterministic_and_invalid_ranges_are_rejected():
    source = _v1_features()
    for minutes in (15, 30, 60, 240):
        source[f"bbo_midpoint_max_{minutes}m"] = source[
            f"bbo_midpoint_min_{minutes}m"
        ]
    frame = build_v2_research_feature_frame(source)

    assert frame["midpoint_range_fraction_15m_240m"].tolist() == [0.0] * len(
        frame
    )
    assert frame["midpoint_range_fraction_60m_240m"].tolist() == [0.0] * len(
        frame
    )

    source.loc[0, "bbo_midpoint_min_15m"] = 0.0
    try:
        build_v2_research_feature_frame(source)
    except ValueError as exc:
        assert "V2 trainable predictors" in str(exc)
    else:
        raise AssertionError("Invalid trainable midpoint range did not fail")

    source = _v1_features()
    source.loc[0, "bbo_state_count_15m"] = -1
    try:
        build_v2_research_feature_frame(source)
    except ValueError as exc:
        assert "counts must be nonnegative" in str(exc)
    else:
        raise AssertionError("Negative trainable event count did not fail")


def test_v2_calendar_encodings_use_utc_fractional_hour_and_weekend():
    source = _v1_features()
    source["timestamp"] = pd.to_datetime(
        [
            "2026-01-03T06:15:00Z",
            "2026-01-03T06:30:00Z",
            "2026-01-03T06:45:00Z",
            "2026-01-03T07:00:00Z",
        ],
        utc=True,
    )
    row = build_v2_research_feature_frame(source).iloc[0]

    assert np.isclose(row["hour_sin"], np.sin(2.0 * np.pi * 6.25 / 24.0))
    assert np.isclose(row["hour_cos"], np.cos(2.0 * np.pi * 6.25 / 24.0))
    assert np.isclose(
        row["quarter_hour_sin"],
        np.sin(2.0 * np.pi * 25.0 / 96.0),
    )
    assert np.isclose(
        row["quarter_hour_cos"],
        np.cos(2.0 * np.pi * 25.0 / 96.0),
    )
    assert np.isclose(row["day_of_week_sin"], np.sin(2.0 * np.pi * 5 / 7))
    assert np.isclose(row["day_of_week_cos"], np.cos(2.0 * np.pi * 5 / 7))
    assert row["is_weekend"] == 1.0


def test_v2_transformation_is_deterministic_nonmutating_and_preserves_audits():
    source = _v1_features()
    original = source.copy(deep=True)

    first = build_v2_research_feature_frame(source)
    second = build_v2_research_feature_frame(source)

    assert_frame_equal(source, original)
    assert_frame_equal(first, second)
    assert_series_equal(first["timestamp"], source["timestamp"])
    assert_series_equal(first["target"], source["target"])
    assert_series_equal(first["feature_ready"], source["feature_ready"])
    assert_series_equal(first["target_eligible"], source["target_eligible"])
    assert_series_equal(first["trainable"], source["trainable"])
    assert_series_equal(
        first["target_log_return_15m"],
        source["target_log_return_15m"],
    )
    assert_series_equal(first["canonical_midpoint"], source["canonical_midpoint"])


def test_v1_v2_walk_forward_comparability_is_identical():
    source = _v1_features(periods=500)
    source["target"] = (np.arange(len(source)) % 2).astype(int)
    source["target_log_return_15m"] = np.where(
        source["target"].eq(1),
        0.001,
        -0.001,
    )
    source["label_available_time"] = source["timestamp"] + pd.Timedelta(
        minutes=15
    )
    config = ExploratoryResearchConfig(
        initial_train_period="72h",
        test_period="12h",
        step_period="12h",
        extra_trees_n_estimators=10,
    )

    v1_frame, _, v1_target = prepare_research_frame(source, "bbo-causal-v1")
    v2_frame, _, v2_target = prepare_research_frame(source, "bbo-causal-v2")
    v1_folds = build_exploratory_folds(v1_frame, config)
    v2_folds = build_exploratory_folds(v2_frame, config)
    assert_series_equal(v1_target, v2_target)
    assert_series_equal(v1_frame["timestamp"], v2_frame["timestamp"])
    assert len(v1_folds) == len(v2_folds)
    for v1_fold, v2_fold in zip(v1_folds, v2_folds):
        assert np.array_equal(v1_fold["train_idx"], v2_fold["train_idx"])
        assert np.array_equal(v1_fold["test_idx"], v2_fold["test_idx"])
        v1_test_start = v1_frame.iloc[v1_fold["test_idx"]]["timestamp"].min()
        v2_test_start = v2_frame.iloc[v2_fold["test_idx"]]["timestamp"].min()
        assert v1_test_start == v2_test_start
        assert (
            v1_frame.iloc[v1_fold["train_idx"]]["label_available_time"].max()
            < v1_test_start
        )
        assert (
            v2_frame.iloc[v2_fold["train_idx"]]["label_available_time"].max()
            < v2_test_start
        )

    v1_summary, v1_predictions = run_exploratory_walk_forward(source, config)
    v2_config = replace(config, predictor_contract_version="bbo-causal-v2")
    v2_summary, v2_predictions = run_exploratory_walk_forward(source, v2_config)
    assert v1_summary["n_oof_predictions_per_benchmark"] == (
        v2_summary["n_oof_predictions_per_benchmark"]
    )
    v1_test_timestamps = v1_predictions.loc[
        v1_predictions["benchmark"].eq("logistic_regression"), "timestamp"
    ].tolist()
    v2_test_timestamps = v2_predictions.loc[
        v2_predictions["benchmark"].eq("logistic_regression"), "timestamp"
    ].tolist()
    assert v1_test_timestamps == v2_test_timestamps
    assert v1_summary["n_folds"] == v2_summary["n_folds"]


def test_v2_cli_records_requested_contract_and_91_feature_drift(tmp_path):
    features_path = tmp_path / "features.parquet"
    _v1_features(periods=500).to_parquet(features_path, index=False)
    output = tmp_path / "v2"
    diagnostics_main(
        [
            "--features",
            str(features_path),
            "--output-root",
            str(output),
            "--predictor-contract",
            "bbo-causal-v2",
        ]
    )

    summary = json.loads((output / "diagnostics.json").read_text())
    drift = pd.read_parquet(output / "feature_drift.parquet")
    assert summary["predictor_contract_version"] == "bbo-causal-v2"
    assert summary["predictor_count"] == 91
    assert summary["feature_drift"]["predictor_count"] == 91
    assert len(drift) == 91
    assert set(drift["feature_name"]) == set(V2_PREDICTOR_COLUMNS)


def test_v2_cli_defaults_to_v1_contract(monkeypatch):
    captured = {}

    def fake_run_diagnostics(features_path, output_root, config):
        captured["version"] = config.predictor_contract_version
        return {}

    monkeypatch.setattr(
        "btc_quarter_hour_engine.research.diagnostics_cli.run_diagnostics",
        fake_run_diagnostics,
    )
    diagnostics_main(["--features", "features.parquet", "--output-root", "out"])

    assert captured["version"] == "bbo-causal-v1"
