import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from btc_quarter_hour_engine.research.feature_families_v2 import (
    V2_FEATURE_FAMILIES,
    localization_variants,
    validate_feature_families,
)
from btc_quarter_hour_engine.research.features import (
    PREDICTOR_COLUMNS,
    PREDICTOR_CONTRACT_VERSION,
)
from btc_quarter_hour_engine.research.features_v2 import (
    V2_CROSS_WINDOW_COLUMNS,
    V2_PREDICTOR_COLUMNS,
)
from btc_quarter_hour_engine.research.signal_localization import (
    EXPERIMENT_COLUMNS,
    IDENTITY_COLUMNS,
    METRICS,
    POPULATION_COLUMNS,
    localization_config,
    localization_metrics,
    run_signal_localization,
    validate_localization_predictions,
)
from btc_quarter_hour_engine.research.signal_localization_cli import (
    main,
    run_localization,
)
from btc_quarter_hour_engine.research.walk_forward import (
    ExploratoryResearchConfig,
    build_exploratory_folds,
    prepare_research_frame,
    run_exploratory_walk_forward,
)
from btc_quarter_hour_engine.research import signal_localization


def _causal_features(periods=385):
    rng = np.random.default_rng(2026)
    history = np.zeros(periods + 17)
    for index in range(1, len(history)):
        history[index] = 0.6 * history[index - 1] + rng.normal()
    current = history[16:-1]
    data = {
        column: rng.uniform(1, 2, periods)
        for column in PREDICTOR_COLUMNS
    }
    data["asof_book_synced"] = np.ones(periods)
    data["asof_midpoint"] = 50_000 + current
    data["asof_top_book_imbalance"] = np.tanh(current)
    for minutes in (15, 30, 60, 240):
        data[f"momentum_log_return_{minutes}m"] = (
            pd.Series(history).rolling(minutes // 15).sum().to_numpy()[16:-1]
            / 1000
        )
        data[f"bbo_midpoint_min_{minutes}m"] = np.full(periods, 50_000 - minutes)
        data[f"bbo_midpoint_max_{minutes}m"] = np.full(periods, 50_000 + minutes)
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="15min", tz="UTC")
    frame = pd.DataFrame(data)
    frame["timestamp"] = timestamps
    frame["label_available_time"] = timestamps + pd.Timedelta(minutes=15)
    frame["target_log_return_15m"] = history[17:] / 1000
    frame["target"] = (history[17:] > 0).astype(int)
    frame["trainable"] = True
    frame["feature_ready"] = True
    frame["target_eligible"] = True
    return frame


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    features = _causal_features()
    original = features.copy(deep=True)
    frame, X, y = prepare_research_frame(features, "bbo-causal-v2")
    folds = build_exploratory_folds(frame, localization_config())
    variants = localization_variants()
    calls = []
    results = []
    original_predict = signal_localization._model_fold_predictions

    def checked_predict(X_train, y_train, X_test, y_test, test, number, config):
        fold = folds[number]
        columns = list(variants.values())[len(calls) % len(variants)]
        assert_frame_equal(X_train, X.iloc[fold["train_idx"]].loc[:, list(columns)])
        assert_frame_equal(X_test, X.iloc[fold["test_idx"]].loc[:, list(columns)])
        assert y_train.equals(y.iloc[fold["train_idx"]])
        assert y_test.equals(y.iloc[fold["test_idx"]])
        assert_frame_equal(test, frame.iloc[fold["test_idx"]])
        assert config == localization_config()
        calls.append(number)
        return original_predict(X_train, y_train, X_test, y_test, test, number, config)

    def capture_run(source):
        result = run_signal_localization(source)
        results.append(result)
        return result

    root = tmp_path_factory.mktemp("synthetic_localization")
    source = root / "synthetic.parquet"
    features.to_parquet(source, index=False)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(signal_localization, "_model_fold_predictions", checked_predict)
        patch.setattr(
            "btc_quarter_hour_engine.research.signal_localization_cli.run_signal_localization",
            capture_run,
        )
        main([
            "--features", str(source), "--output-root", str(root / "results"),
            "--predictor-contract", "bbo-causal-v2",
        ])
    assert len(calls) == len(folds) * 15
    result = results[0]
    assert_frame_equal(features, original)
    return features, result


def _population(predictions):
    return predictions.loc[
        predictions["mode"].eq("reference")
        & predictions["benchmark"].eq("logistic_regression"),
        POPULATION_COLUMNS,
    ].reset_index(drop=True)


def test_family_partition_exact_and_cross_window_exclusive():
    validate_feature_families()
    assert {name: len(columns) for name, columns in V2_FEATURE_FAMILIES.items()} == {
        "momentum": 4, "spread": 13, "order_book_imbalance": 17,
        "activity_intensity": 20, "range_volatility": 12,
        "cross_window_regimes": 18, "calendar": 7,
    }
    flattened = [column for columns in V2_FEATURE_FAMILIES.values() for column in columns]
    assert len(flattened) == len(set(flattened)) == 91
    assert set(flattened) == set(V2_PREDICTOR_COLUMNS)
    assert V2_FEATURE_FAMILIES["cross_window_regimes"] == V2_CROSS_WINDOW_COLUMNS
    assert PREDICTOR_CONTRACT_VERSION == "bbo-causal-v1"
    assert len(PREDICTOR_COLUMNS) == 114
    variants = localization_variants()
    assert len(variants) == 15
    assert variants["reference", "all"] == V2_PREDICTOR_COLUMNS
    for family, members in V2_FEATURE_FAMILIES.items():
        group = variants["group_only", family]
        omitted = variants["leave_one_group_out", family]
        assert set(group) == set(members)
        assert set(group).isdisjoint(omitted)
        assert len(group) + len(omitted) == 91
        assert group == tuple(name for name in V2_PREDICTOR_COLUMNS if name in members)


@pytest.mark.parametrize("defect", ["duplicate", "missing", "forbidden", "unknown", "swapped"])
def test_invalid_family_contract_rejected(defect):
    families = dict(V2_FEATURE_FAMILIES)
    if defect == "duplicate":
        families["calendar"] += families["momentum"][:1]
    elif defect == "missing":
        families["momentum"] = families["momentum"][1:]
    elif defect == "forbidden":
        families["calendar"] += ("target",)
    elif defect == "unknown":
        families["momentum"] = ("unapproved",) + families["momentum"][1:]
    else:
        families["momentum"], families["calendar"] = (
            families["calendar"], families["momentum"]
        )
    with pytest.raises(ValueError):
        validate_feature_families(families)


def test_chronological_folds_and_strict_embargo_unchanged():
    source = _causal_features(500)
    source.loc[12, ["trainable", "feature_ready", "target_eligible"]] = False
    v1, _, v1_y = prepare_research_frame(source)
    v2, X, v2_y = prepare_research_frame(source, "bbo-causal-v2")
    assert_frame_equal(v1[["timestamp", "target"]], v2[["timestamp", "target"]])
    assert v1_y.equals(v2_y)
    assert tuple(X.columns) == V2_PREDICTOR_COLUMNS
    first = build_exploratory_folds(v1, ExploratoryResearchConfig())
    second = build_exploratory_folds(v2, localization_config())
    assert len(first) == len(second) > 1
    for left, right in zip(first, second):
        for key in ("train_idx", "test_idx"):
            np.testing.assert_array_equal(left[key], right[key])
        train = v2.iloc[right["train_idx"]]
        test = v2.iloc[right["test_idx"]]
        assert train["timestamp"].max() < test["timestamp"].min()
        assert train["label_available_time"].max() < right["test_start"]
        at_cutoff = v2.index[v2["label_available_time"].eq(right["test_start"])]
        assert len(at_cutoff) == 1
        assert set(at_cutoff).isdisjoint(right["train_idx"])


def test_label_available_before_target_boundary_rejected():
    source = _causal_features()
    source.loc[0, "label_available_time"] = (
        source.loc[0, "timestamp"] + pd.Timedelta(minutes=15)
        - pd.Timedelta(nanoseconds=1)
    )
    with pytest.raises(ValueError, match="label_available_time.*timestamp \\+ 15 minutes"):
        signal_localization._validate_input(source)


def test_delayed_label_allowed_and_strict_training_embargo_preserved():
    source = _causal_features()
    first_test_start = source["timestamp"].min() + pd.Timedelta(hours=72)
    source.loc[0, "label_available_time"] = first_test_start
    signal_localization._validate_input(source)
    frame, _, _ = prepare_research_frame(source, "bbo-causal-v2")
    folds = build_exploratory_folds(frame, localization_config())
    assert 0 not in folds[0]["train_idx"]
    assert 0 in folds[1]["train_idx"]
    for fold in folds:
        assert (
            frame.iloc[fold["train_idx"]]["label_available_time"].max()
            < fold["test_start"]
        )


@pytest.mark.parametrize("both_excluded", [False, True])
def test_duplicate_source_timestamp_with_excluded_row_rejected(both_excluded):
    source = _causal_features()
    source.loc[1, "timestamp"] = source.loc[0, "timestamp"]
    source.loc[1, ["trainable", "feature_ready", "target_eligible"]] = False
    source.loc[1, "asof_spread_bps"] = np.inf
    if both_excluded:
        source.loc[0, ["trainable", "feature_ready", "target_eligible"]] = False
    with pytest.raises(ValueError, match="timestamps must be unique.*complete input"):
        run_signal_localization(source)


def test_integration_full_reference_is_original_benchmark(experiment):
    source, (metadata, predictions, metrics, per_fold) = experiment
    reference_metadata, original = run_exploratory_walk_forward(
        source, localization_config()
    )
    reference = predictions.loc[predictions["mode"].eq("reference")]
    original = original.sort_values(["benchmark", *IDENTITY_COLUMNS]).reset_index(drop=True)
    reference = reference.sort_values(["benchmark", *IDENTITY_COLUMNS]).reset_index(drop=True)
    assert_frame_equal(reference[original.columns], original)
    assert metadata["validation"]["random_state"] == 42
    assert metadata["validation"]["extra_trees_n_estimators"] == 300
    assert metadata["validation"]["initial_train_period"] == "72h"
    assert metadata["validation"]["test_period"] == "12h"
    assert metadata["validation"]["step_period"] == "12h"
    assert metadata["folds"] == reference_metadata["folds"]
    assert metadata["n_folds"] > 1
    assert metadata["n_oof_predictions_per_benchmark"] == len(_population(predictions))
    assert len(metrics) == 45 + 4
    assert len(per_fold) == len(metrics) * metadata["n_folds"]
    validate_localization_predictions(predictions, _population(predictions))
    for _, row in metrics.iterrows():
        mask = np.logical_and.reduce([
            predictions[key].eq(row[key]) for key in EXPERIMENT_COLUMNS
        ])
        group = predictions.loc[mask]
        assert row["accuracy"] == group["y_true"].eq(group["y_pred"]).mean()
        full = metrics.loc[
            metrics["mode"].eq("reference") & metrics["benchmark"].eq(row["benchmark"])
        ].iloc[0]
        for metric in METRICS:
            if pd.notna(row[metric]):
                assert row[f"delta_{metric}_vs_full_v2"] == pytest.approx(
                    row[metric] - full[metric]
                )
        fold_accuracy = group.groupby("fold_number").apply(
            lambda part: part["y_true"].eq(part["y_pred"]).mean(),
            include_groups=False,
        )
        for statistic, value in (
            ("mean", fold_accuracy.mean()), ("std", fold_accuracy.std(ddof=0)),
            ("min", fold_accuracy.min()), ("max", fold_accuracy.max()),
        ):
            assert row[f"fold_accuracy_{statistic}"] == pytest.approx(value)
    for _, row in per_fold.iterrows():
        mask = np.logical_and.reduce([
            predictions[key].eq(row[key])
            for key in EXPERIMENT_COLUMNS + ["fold_number"]
        ])
        group = predictions.loc[mask]
        assert row["accuracy"] == group["y_true"].eq(group["y_pred"]).mean()


@pytest.mark.parametrize(
    "defect", [
        "duplicate", "missing_row", "missing_model", "fold", "timestamp",
        "target", "return", "nan_probability", "bad_probability", "bad_class",
        "benchmark_type", "missing_column",
    ],
)
def test_oof_fail_closed(experiment, defect):
    predictions = experiment[1][1].copy()
    expected = _population(predictions)
    if defect == "duplicate":
        predictions = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    elif defect == "missing_row":
        predictions = predictions.iloc[1:]
    elif defect == "missing_model":
        predictions = predictions.loc[predictions["benchmark"].ne("extra_trees")]
    elif defect == "missing_column":
        predictions = predictions.drop(columns="y_proba")
    else:
        column, value = {
            "fold": ("fold_number", 999),
            "timestamp": ("timestamp", pd.Timestamp("2020-01-01", tz="UTC")),
            "target": ("y_true", 1 - predictions.loc[0, "y_true"]),
            "return": ("target_log_return_15m", 999.0),
            "nan_probability": ("y_proba", np.nan),
            "bad_probability": ("y_proba", 1.1),
            "bad_class": ("y_pred", 1 - predictions.loc[0, "y_pred"]),
            "benchmark_type": ("benchmark_type", "baseline"),
        }[defect]
        predictions.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_localization_predictions(predictions, expected)


def test_undefined_metrics_explicit_not_zero(experiment):
    predictions = experiment[1][1].copy()
    predictions["y_true"] = 1
    metrics, per_fold = localization_metrics(predictions, _population(predictions))
    for table in (metrics, per_fold):
        models = table.loc[table["benchmark"].eq("logistic_regression")]
        for metric in ("roc_auc", "log_loss"):
            assert models[metric].isna().all()
            assert models[f"{metric}_undefined_reason"].eq(
                "single_target_class_in_population"
            ).all()
            assert models[f"delta_{metric}_vs_full_v2"].isna().all()
        assert models["brier_score"].notna().all()
        baselines = table.loc[table["benchmark"].eq("always_up")]
        assert baselines["roc_auc_undefined_reason"].eq(
            "hard_label_baseline_has_no_probability"
        ).all()


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("column", ["asof_spread_bps", "l2_update_count_15m"])
def test_nonfinite_trainable_predictors_rejected(value, column):
    source = _causal_features()
    source.loc[0, column] = value
    with pytest.raises(ValueError, match="trainable.*(missing|non-finite)"):
        run_signal_localization(source)


@pytest.mark.parametrize(
    "defect", [
        "duplicate_column", "duplicate_timestamp", "missing_source", "fractional_target",
        "missing_timestamp", "missing_label_time", "bad_flags", "ineligible",
        "no_trainable", "no_folds", "one_training_class",
    ],
)
def test_input_failures(defect):
    source = _causal_features()
    if defect == "duplicate_column":
        source = pd.concat([source, source[["asof_spread_bps"]]], axis=1)
    elif defect == "duplicate_timestamp":
        source.loc[1, "timestamp"] = source.loc[0, "timestamp"]
    elif defect == "missing_source":
        source = source.drop(columns="bbo_state_count_15m")
    elif defect == "fractional_target":
        source["target"] = source["target"].astype(float)
        source.loc[0, "target"] = 0.5
    elif defect == "missing_timestamp":
        source.loc[0, "timestamp"] = pd.NaT
    elif defect == "missing_label_time":
        source.loc[0, "label_available_time"] = pd.NaT
    elif defect == "bad_flags":
        source["trainable"] = "True"
    elif defect == "ineligible":
        source.loc[0, "target_eligible"] = False
    elif defect == "no_trainable":
        source["trainable"] = False
    elif defect == "no_folds":
        source = source.iloc[:100]
    elif defect == "one_training_class":
        source["target"] = 1
    with pytest.raises(ValueError):
        run_signal_localization(source)


def test_deterministic_rerun_with_shuffled_input_and_ineligible_nonfinite(experiment):
    source, first = experiment
    source = source.copy()
    ineligible = source.iloc[[0]].copy()
    ineligible["timestamp"] -= pd.Timedelta(minutes=15)
    ineligible[["trainable", "feature_ready", "target_eligible"]] = False
    ineligible["asof_spread_bps"] = np.inf
    combined = pd.concat([source, ineligible], ignore_index=True).sample(
        frac=1, random_state=8
    )
    second = run_signal_localization(combined)
    assert first[0] == second[0]
    for left, right in zip(first[1:], second[1:]):
        assert_frame_equal(left, right)


def test_cli_deterministic_artifacts_and_manifest(experiment, tmp_path, monkeypatch):
    source = tmp_path / "synthetic.parquet"
    experiment[0].to_parquet(source, index=False)
    calls = []

    def cached_run(features):
        assert_frame_equal(features, experiment[0])
        calls.append(True)
        return experiment[1]

    monkeypatch.setattr(
        "btc_quarter_hour_engine.research.signal_localization_cli.run_signal_localization",
        cached_run,
    )
    for root in ("first", "second"):
        main([
            "--features", str(source), "--output-root", str(tmp_path / root),
            "--predictor-contract", "bbo-causal-v2",
        ])
    assert len(calls) == 2
    first = tmp_path / "first"
    second = tmp_path / "second"
    assert len(list(first.iterdir())) == 8
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["snapshot_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert len(manifest["git_sha"]) == 40
    assert manifest["predictor_contract_version"] == "bbo-causal-v2"
    assert manifest["validation"]["strict_label_cutoff"] == "label_available_time < test_start"
    assert sum(map(len, manifest["feature_families"].values())) == 91
    for name, digest in manifest["output_sha256"].items():
        assert digest == hashlib.sha256((first / name).read_bytes()).hexdigest()
    summary = json.loads((first / "signal_localization.json").read_text())
    baseline = next(row for row in summary["metrics"] if row["benchmark"] == "always_up")
    assert baseline["roc_auc"] is None
    assert baseline["delta_roc_auc_vs_full_v2"] is None
    assert "NaN" not in (first / "signal_localization.json").read_text()
    actual = pd.read_parquet(first / "oof_predictions.parquet")
    assert_frame_equal(actual, experiment[1][1])
    with pytest.raises(ValueError, match="never overwritten"):
        run_localization(source, first, predictor_contract="bbo-causal-v2")


def test_cli_requires_explicit_v2_contract(tmp_path):
    base = ["--features", "missing.parquet", "--output-root", str(tmp_path)]
    for args in (base, base + ["--predictor-contract", "bbo-causal-v1"]):
        with pytest.raises(SystemExit) as exc:
            main(args)
        assert exc.value.code == 2
    with pytest.raises(ValueError, match="explicit bbo-causal-v2"):
        run_localization("missing.parquet", tmp_path, predictor_contract="bbo-causal-v1")


def test_cli_rejects_snapshot_changed_during_run(experiment, tmp_path, monkeypatch):
    source = tmp_path / "synthetic.parquet"
    experiment[0].to_parquet(source, index=False)
    output = tmp_path / "results"

    def changed_snapshot(features):
        source.write_bytes(b"changed input")
        return experiment[1]

    monkeypatch.setattr(
        "btc_quarter_hour_engine.research.signal_localization_cli.run_signal_localization",
        changed_snapshot,
    )
    with pytest.raises(ValueError, match="snapshot changed"):
        run_localization(source, output, predictor_contract="bbo-causal-v2")
    assert not output.exists()
