import hashlib
import io
import json
import pickle
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from btc_quarter_hour_engine.research import model_freeze as freeze
from btc_quarter_hour_engine.research.features import PREDICTOR_COLUMNS
from btc_quarter_hour_engine.research.features_v2 import V2_PREDICTOR_COLUMNS
from btc_quarter_hour_engine.research.feature_families_v2 import localization_variants
from btc_quarter_hour_engine.research.model_freeze_cli import main


def _features(periods=24):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({
        column: rng.uniform(1, 2, periods) for column in PREDICTOR_COLUMNS
    })
    for minutes in (15, 30, 60, 240):
        frame[f"bbo_midpoint_min_{minutes}m"] = 50_000.0 - minutes
        frame[f"bbo_midpoint_max_{minutes}m"] = 50_000.0 + minutes
    frame["timestamp"] = pd.date_range(
        "2026-10-06T00:00:00Z", periods=periods, freq="15min"
    )
    frame["label_available_time"] = frame["timestamp"] + pd.Timedelta(minutes=15)
    frame["target_time"] = frame["label_available_time"]
    frame["canonical_target_eligible"] = True
    frame["target"] = np.arange(periods) % 2
    frame["target_log_return_15m"] = np.where(frame["target"].eq(1), 0.001, -0.001)
    for column in ("trainable", "feature_ready", "target_eligible"):
        frame[column] = True
    return frame


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("freeze")
    source = root / "synthetic.parquet"
    features = _features()
    features.to_parquet(source, index=False)
    with pytest.MonkeyPatch.context() as patch:
        # Test-only substitution; neither the API nor CLI permits snapshot overrides.
        patch.setattr(freeze, "SNAPSHOT_SHA256", freeze._sha256_file(source))
        patch.setattr(freeze, "TRAINING_OBSERVATIONS", len(features))
        receipt = io.StringIO()
        with redirect_stdout(receipt):
            main([
                "--training-input", str(source),
                "--output-directory", str(root / "development"),
                "--development-dry-run",
            ])
        result = json.loads(receipt.getvalue())
        models, manifest = freeze.load_frozen_models(
            result["output_directory"],
            trusted_manifest_sha256=result["trusted_manifest_sha256"],
            require_official=False,
        )
        yield root, source, features, result, models, manifest


def _load(result, **kwargs):
    return freeze.load_frozen_models(
        result["output_directory"],
        trusted_manifest_sha256=result["trusted_manifest_sha256"],
        **kwargs,
    )


def test_contract_parameters_serialization_and_determinism(bundle):
    _, _, features, result, models, manifest = bundle
    variants = localization_variants()
    columns = freeze.frozen_predictors()
    assert columns == {
        "range_volatility": variants["group_only", "range_volatility"],
        "full_v2": variants["reference", "all"],
    }
    assert tuple(map(len, columns.values())) == (12, 91)
    assert columns["full_v2"] == V2_PREDICTOR_COLUMNS
    frame, X, y = freeze.validate_training_input(features)
    for name, names in columns.items():
        pipeline = models[name]
        assert tuple(pipeline.feature_names_in_) == names
        model = pipeline.named_steps["model"]
        assert model.n_estimators == 300
        assert len(model.estimators_) == 300
        assert model.max_depth == 6
        assert model.random_state == 42
        assert pipeline.named_steps["imputer"].strategy == "median"
        assert np.isfinite(pipeline.named_steps["imputer"].statistics_).all()
        matrix = X.loc[:, list(names)]
        fitted = freeze._new_pipeline().fit(matrix, y)
        expected = freeze.verified_probabilities(fitted, matrix, names)
        actual = freeze.verified_probabilities(pipeline, matrix, names)
        np.testing.assert_array_equal(expected, actual)
        np.testing.assert_array_equal(pipeline.predict(matrix), fitted.predict(matrix))
        np.testing.assert_array_equal(
            actual, freeze.verified_probabilities(pipeline, matrix, names)
        )
        assert manifest["models"][name]["predictors"] == list(names)
        assert manifest["models"][name]["configuration"] == freeze._configuration(fitted)
    assert manifest["status"] == freeze.DEVELOPMENT_STATUS
    assert manifest["training"]["observation_count"] == len(features)
    assert manifest["training"]["timestamp_min"] == frame["timestamp"].min().isoformat()
    assert manifest["training"]["timestamp_max"] == frame["timestamp"].max().isoformat()
    assert manifest["training"]["max_label_available_time"] == (
        frame["label_available_time"].max().isoformat()
    )
    assert pd.Timestamp(manifest["training"]["started_at"]) <= pd.Timestamp(
        manifest["training"]["completed_at"]
    ) <= pd.Timestamp(manifest["freeze_completed_at"])
    assert set(manifest["environment"]) == {
        "python", "numpy", "pandas", "scikit-learn", "scipy", "pyarrow"
    }
    assert len(manifest["source"]["commit_sha"]) == 40
    assert isinstance(manifest["source"]["working_tree_state"], str)
    assert manifest["source"]["dirty"] == bool(manifest["source"]["working_tree_state"])
    assert manifest["predictor_contract_version"] == "bbo-causal-v2"
    assert manifest["family_contract_version"] == "bbo-causal-v2-families-v1"
    assert result["trusted_manifest_sha256"] == hashlib.sha256(
        (Path(result["output_directory"]) / "manifest.json").read_bytes()
    ).hexdigest()


@pytest.mark.parametrize("defect", [
    "duplicate", "unsorted", "nat", "off_boundary", "outside_october",
    "missing_predictor", "nan_predictor", "infinite_predictor", "invalid_count",
    "missing_label", "early_label", "late_label", "nonbinary", "single_class",
    "wrong_direction", "not_ready", "ineligible", "hidden_trainable", "null_flag",
    "string_flag", "count", "duplicate_column", "nonfinite_return",
    "flat_return", "target_time", "canonical_ineligible",
])
def test_invalid_training_input_rejected(bundle, defect):
    frame = bundle[2].copy()
    if defect == "duplicate":
        frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    elif defect == "unsorted":
        frame = frame.iloc[::-1].reset_index(drop=True)
    elif defect == "nat":
        frame.loc[0, "timestamp"] = pd.NaT
    elif defect == "off_boundary":
        frame.loc[0, "timestamp"] += pd.Timedelta(seconds=1)
    elif defect == "outside_october":
        frame["timestamp"] -= pd.Timedelta(days=30)
    elif defect == "missing_predictor":
        frame = frame.drop(columns=["asof_spread_bps"])
    elif defect == "nan_predictor":
        frame.loc[0, "distance_from_high_15m"] = np.nan
    elif defect == "infinite_predictor":
        frame.loc[0, "asof_spread_bps"] = np.inf
    elif defect == "invalid_count":
        frame.loc[0, "bbo_state_count_15m"] = -1
    elif defect == "missing_label":
        frame.loc[0, "label_available_time"] = pd.NaT
    elif defect == "early_label":
        frame.loc[0, "label_available_time"] = frame.loc[0, "timestamp"]
    elif defect == "late_label":
        frame.loc[0, "label_available_time"] = pd.Timestamp(freeze.WINDOW_START)
    elif defect == "nonbinary":
        frame.loc[0, "target"] = 2
    elif defect == "single_class":
        frame["target"] = 0
    elif defect == "wrong_direction":
        frame.loc[0, "target_log_return_15m"] = 0.001
    elif defect in ("not_ready", "ineligible"):
        frame.loc[0, "feature_ready" if defect == "not_ready" else "target_eligible"] = False
    elif defect == "hidden_trainable":
        frame.loc[0, "trainable"] = False
    elif defect == "null_flag":
        frame["trainable"] = frame["trainable"].astype("boolean")
        frame.loc[0, "trainable"] = pd.NA
    elif defect == "string_flag":
        frame["trainable"] = "True"
    elif defect == "count":
        frame = frame.iloc[:-1]
    elif defect == "duplicate_column":
        frame = pd.concat([frame, frame[["target"]]], axis=1)
    elif defect == "nonfinite_return":
        frame.loc[0, "target_log_return_15m"] = np.inf
    elif defect == "flat_return":
        frame.loc[0, "target_log_return_15m"] = 0.0
    elif defect == "target_time":
        frame.loc[0, "target_time"] += pd.Timedelta(minutes=15)
    elif defect == "canonical_ineligible":
        frame.loc[0, "canonical_target_eligible"] = False
    with pytest.raises(ValueError):
        freeze.validate_training_input(frame)


def test_nontrainable_rows_are_not_repaired_or_promoted(bundle):
    frame = bundle[2].copy()
    extra = frame.iloc[[-1]].copy()
    extra["timestamp"] += pd.Timedelta(minutes=15)
    extra["trainable"] = False
    extra["feature_ready"] = False
    extra["target_eligible"] = False
    extra["target"] = np.nan
    extra["label_available_time"] = pd.NaT
    combined = pd.concat([frame, extra], ignore_index=True)
    prepared, _, _ = freeze.validate_training_input(combined)
    assert len(prepared) == len(frame)
    assert len(combined) == len(frame) + 1


def test_hash_mismatch_rejected_before_training(bundle, tmp_path, monkeypatch):
    wrong = tmp_path / "wrong.parquet"
    wrong.write_bytes(bundle[1].read_bytes() + b"modified")
    monkeypatch.setattr(freeze, "_new_pipeline", lambda: pytest.fail("Must not fit"))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        freeze.freeze_models(wrong, tmp_path / "output", development=True)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("existing", ["empty", "nonempty", "file"])
def test_existing_output_never_overwritten(bundle, tmp_path, existing):
    output = tmp_path / "existing"
    if existing == "file":
        output.write_text("preserve")
    else:
        output.mkdir()
        if existing == "nonempty":
            (output / "evidence").write_text("preserve")
    with pytest.raises(FileExistsError):
        freeze.freeze_models(bundle[1], output, development=True)
    assert output.exists()
    if existing == "file":
        assert output.read_text() == "preserve"


@pytest.mark.parametrize("artifact", ["range_volatility.pkl", "full_v2.pkl", "manifest.json"])
def test_modified_artifact_rejected_before_any_deserialization(
    bundle, tmp_path, monkeypatch, artifact
):
    original = Path(bundle[3]["output_directory"])
    for path in original.iterdir():
        (tmp_path / path.name).write_bytes(path.read_bytes())
    with (tmp_path / artifact).open("ab") as stream:
        stream.write(b"modified")
    monkeypatch.setattr(pickle, "loads", lambda _: pytest.fail("Unverified deserialization"))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        freeze.load_frozen_models(
            tmp_path, trusted_manifest_sha256=bundle[3]["trusted_manifest_sha256"],
            require_official=False,
        )


def test_manifest_order_rejected_before_deserialization(bundle, tmp_path, monkeypatch):
    original = Path(bundle[3]["output_directory"])
    for path in original.iterdir():
        (tmp_path / path.name).write_bytes(path.read_bytes())
    manifest = json.loads((tmp_path / "manifest.json").read_bytes())
    manifest["models"]["full_v2"]["predictors"].reverse()
    data = json.dumps(manifest).encode()
    (tmp_path / "manifest.json").write_bytes(data)
    monkeypatch.setattr(pickle, "loads", lambda _: pytest.fail("Unverified deserialization"))
    with pytest.raises(ValueError, match="predictor ordering"):
        freeze.load_frozen_models(
            tmp_path, trusted_manifest_sha256=hashlib.sha256(data).hexdigest(),
            require_official=False,
        )


def test_serialized_pipeline_order_mismatch_rejected(bundle, tmp_path):
    original = Path(bundle[3]["output_directory"])
    for path in original.iterdir():
        (tmp_path / path.name).write_bytes(path.read_bytes())
    manifest = json.loads((tmp_path / "manifest.json").read_bytes())
    model = pickle.loads((tmp_path / "full_v2.pkl").read_bytes())
    model.named_steps["imputer"].feature_names_in_ = model.feature_names_in_[::-1].copy()
    data = pickle.dumps(model, protocol=5)
    (tmp_path / "full_v2.pkl").write_bytes(data)
    manifest["models"]["full_v2"]["sha256"] = hashlib.sha256(data).hexdigest()
    manifest_data = json.dumps(manifest).encode()
    (tmp_path / "manifest.json").write_bytes(manifest_data)
    with pytest.raises(ValueError, match="predictor order"):
        freeze.load_frozen_models(
            tmp_path, trusted_manifest_sha256=hashlib.sha256(manifest_data).hexdigest(),
            require_official=False,
        )


def test_predictor_order_and_nonfinite_inference_rejected(bundle):
    names = freeze.frozen_predictors()["range_volatility"]
    _, X, _ = freeze.validate_training_input(bundle[2])
    model = bundle[4]["range_volatility"]
    with pytest.raises(ValueError, match="Predictor order"):
        freeze.verified_probabilities(model, X.loc[:, list(reversed(names))], names)
    X.loc[0, names[0]] = np.nan
    with pytest.raises(ValueError, match="finite"):
        freeze.verified_probabilities(model, X.loc[:, list(names)], names)


@pytest.mark.parametrize("defect", ["nan", "bounds", "sum", "shape"])
def test_invalid_probability_outputs_rejected(bundle, monkeypatch, defect):
    names = freeze.frozen_predictors()["range_volatility"]
    _, X, _ = freeze.validate_training_input(bundle[2])
    model = bundle[4]["range_volatility"]
    values = np.full((len(X), 2), 0.5)
    if defect == "nan":
        values[0, 0] = np.nan
    elif defect == "bounds":
        values[0] = [-0.1, 1.1]
    elif defect == "sum":
        values[0] = [0.2, 0.2]
    else:
        values = values[:, :1]
    monkeypatch.setattr(model, "predict_proba", lambda _: values)
    with pytest.raises(ValueError, match="probability"):
        freeze.verified_probabilities(model, X.loc[:, list(names)], names)


def test_development_cannot_be_loaded_as_official(bundle):
    with pytest.raises(ValueError, match="Development-only"):
        _load(bundle[3])


def test_changing_only_status_cannot_promote_development(bundle, tmp_path, monkeypatch):
    manifest = json.loads(json.dumps(bundle[5]))
    manifest["status"] = freeze.OFFICIAL_STATUS
    data = json.dumps(manifest).encode()
    (tmp_path / "manifest.json").write_bytes(data)
    monkeypatch.setattr(pickle, "loads", lambda _: pytest.fail("Must not deserialize"))
    with pytest.raises(ValueError, match="Official freeze approval"):
        freeze.load_frozen_models(
            tmp_path, trusted_manifest_sha256=hashlib.sha256(data).hexdigest(),
        )


def test_cli_requires_explicit_mode_and_development_disallows_approval(bundle, tmp_path):
    args = ["--training-input", str(bundle[1]), "--output-directory", str(tmp_path / "out")]
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(args + ["--development-dry-run", "--approved-implementation-sha", "a" * 40])
    assert exc.value.code == 2
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("defect", ["dirty", "missing", "wrong", "protocol", "deadline"])
def test_official_approval_gate(bundle, monkeypatch, defect):
    monkeypatch.setattr(freeze, "_utc_now", lambda: "2026-10-28T20:00:00Z")
    state = freeze._source_state()
    state["dirty"] = False
    state["working_tree_state"] = ""
    implementation = state["commit_sha"]
    protocol = implementation
    if defect == "dirty":
        state["dirty"] = True
    elif defect == "missing":
        implementation = None
    elif defect == "wrong":
        implementation = "0" * 40
    elif defect == "protocol":
        state["protocol_git_blob"] = "0" * 40
    elif defect == "deadline":
        monkeypatch.setattr(freeze, "_utc_now", lambda: "2026-11-15T00:00:00Z")
    with pytest.raises(ValueError):
        freeze._approval(
            state, development=False, approved_implementation_sha=implementation,
            approved_protocol_sha=protocol,
        )


def test_clean_approved_revision_passes_only_the_approval_gate(bundle, monkeypatch):
    monkeypatch.setattr(freeze, "_utc_now", lambda: "2026-10-28T20:00:00Z")
    state = freeze._source_state()
    state.update(dirty=False, working_tree_state="")
    freeze._approval(
        state, development=False, approved_implementation_sha=state["commit_sha"],
        approved_protocol_sha=state["commit_sha"],
    )


def test_exact_production_observation_count_without_fitting(monkeypatch):
    monkeypatch.setattr(freeze, "TRAINING_OBSERVATIONS", 1935)
    frame, X, y = freeze.validate_training_input(_features(1935))
    assert len(frame) == len(X) == len(y) == 1935
    with pytest.raises(ValueError, match="1935"):
        freeze.validate_training_input(_features(1934))


def test_atomic_publication_cannot_replace_a_racing_empty_directory(tmp_path):
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    (staging / "model").write_text("new")
    output.mkdir()
    with pytest.raises(OSError):
        freeze._publish_directory(staging, output)
    assert not list(output.iterdir())
    assert (staging / "model").read_text() == "new"


@pytest.mark.parametrize("failure", ["snapshot", "serialize", "roundtrip", "source", "publish"])
def test_training_or_staging_failure_leaves_no_output(bundle, tmp_path, monkeypatch, failure):
    source = bundle[1]
    original_hash = freeze._sha256_file
    original_source = freeze._source_state()
    if failure == "snapshot":
        monkeypatch.setattr(
            freeze, "_sha256_file",
            lambda path: "0" * 64 if Path(path) == source else original_hash(path),
        )
    elif failure == "serialize":
        def broken_dump(*args, **kwargs):
            raise OSError("serialization failed")
        monkeypatch.setattr(pickle, "dumps", broken_dump)
    elif failure == "roundtrip":
        monkeypatch.setattr(
            freeze, "_load_pipeline", lambda *args: freeze._new_pipeline()
        )
    elif failure == "source":
        calls = iter([original_source, {**original_source, "commit_sha": "0" * 40}])
        monkeypatch.setattr(freeze, "_source_state", lambda: next(calls))
    else:
        def broken_publish(*args):
            raise OSError("publication failed")
        monkeypatch.setattr(freeze, "_publish_directory", broken_publish)
    with pytest.raises((ValueError, OSError, AttributeError)):
        freeze.freeze_models(source, tmp_path / "output", development=True)
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.iterdir())


def test_integrity_gates_survive_python_optimization(bundle, tmp_path):
    script = """
import sys
from pathlib import Path
from btc_quarter_hour_engine.research.model_freeze import freeze_models
try:
    freeze_models(sys.argv[1], sys.argv[2], development=True)
except ValueError as exc:
    if "SHA256 mismatch" not in str(exc):
        raise
else:
    raise RuntimeError("Integrity check was disabled under python -O")
if Path(sys.argv[2]).exists():
    raise RuntimeError("Invalid input produced an output")
"""
    result = subprocess.run(
        [sys.executable, "-O", "-c", script, str(bundle[1]), str(tmp_path / "output")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_hard_coded_production_population_and_snapshot():
    # Separate interpreter prevents test-only fixture substitutions from leaking here.
    result = subprocess.run([
        sys.executable, "-c",
        "from btc_quarter_hour_engine.research.model_freeze import "
        "SNAPSHOT_SHA256, TRAINING_OBSERVATIONS; "
        "print(SNAPSHOT_SHA256, TRAINING_OBSERVATIONS)",
    ], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == (
        "ae0417ab38de3a0204baf8202d741a541e38f79597c47d3691da177ac32aa2fc 1935"
    )
