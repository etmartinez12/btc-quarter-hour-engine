"""Offline prospective model freeze. Pickle files require independently trusted hashes."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import io
import json
import os
import pickle
import platform
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from btc_quarter_hour_engine.research.feature_families_v2 import (
    FAMILY_CONTRACT_VERSION,
    localization_variants,
)
from btc_quarter_hour_engine.research.features_v2 import V2_PREDICTOR_CONTRACT_VERSION
from btc_quarter_hour_engine.research.forward_dataset import _sha256_file
from btc_quarter_hour_engine.research.signal_localization import (
    _validate_input,
    localization_config,
)
from btc_quarter_hour_engine.research.walk_forward import (
    _model_factories,
    prepare_research_frame,
)


SNAPSHOT_SHA256 = "ae0417ab38de3a0204baf8202d741a541e38f79597c47d3691da177ac32aa2fc"
TRAINING_OBSERVATIONS = 1935
WINDOW_START = "2026-11-15T00:00:00Z"
WINDOW_END = "2026-12-15T00:00:00Z"
FREEZE_DEADLINE = "2026-11-14T23:00:00Z"
PROTOCOL_PATH = "docs/research/v2-prospective-validation-protocol-2026-10-28.md"
DEVELOPMENT_STATUS = "development_only_nonofficial"
OFFICIAL_STATUS = "official_prospective_freeze"
REPOSITORY = Path(__file__).resolve().parents[2]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "--no-optional-locks", "-C", str(REPOSITORY), *arguments],
        text=True,
    ).strip()


def _source_state() -> dict[str, Any]:
    state = _git("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "commit_sha": _git("rev-parse", "HEAD"),
        "working_tree_state": state,
        "dirty": bool(state),
        "protocol_sha256": _sha256_file(REPOSITORY / PROTOCOL_PATH),
        "protocol_commit_sha": _git("log", "-1", "--format=%H", "--", PROTOCOL_PATH),
        "protocol_git_blob": _git(
            "hash-object", f"--path={PROTOCOL_PATH}", str(REPOSITORY / PROTOCOL_PATH),
        ),
    }


def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        **{
            package: importlib.metadata.version(package)
            for package in ("numpy", "pandas", "scikit-learn", "scipy", "pyarrow")
        },
    }


def frozen_predictors() -> dict[str, tuple[str, ...]]:
    variants = localization_variants()
    predictors = {
        "range_volatility": variants["group_only", "range_volatility"],
        "full_v2": variants["reference", "all"],
    }
    if tuple(map(len, predictors.values())) != (12, 91):
        raise ValueError("Frozen predictor counts must be exactly 12 and 91")
    return predictors


def validate_training_input(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Validate before the shared preparer can sort or select the population."""
    _validate_input(features)
    timestamps = pd.to_datetime(features["timestamp"], utc=True, errors="raise")
    if timestamps.isna().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("Source timestamps must be non-null and chronological")
    if (timestamps != timestamps.dt.floor("15min")).any():
        raise ValueError("Source timestamps must be exact quarter-hour boundaries")
    expected_trainable = features["feature_ready"] & features["target_eligible"]
    if not features["trainable"].equals(expected_trainable):
        raise ValueError("trainable must equal feature_ready & target_eligible")
    selected = features.loc[features["trainable"]]
    if len(selected) != TRAINING_OBSERVATIONS:
        raise ValueError(f"Expected exactly {TRAINING_OBSERVATIONS} trainable observations")
    if set(selected["target"]) != {0, 1}:
        raise ValueError("Training targets must contain both binary classes")
    labels = pd.to_datetime(selected["label_available_time"], utc=True, errors="raise")
    if (labels >= pd.Timestamp(WINDOW_START)).any():
        raise ValueError("Training labels must be available before the prospective window")
    selected_times = timestamps.loc[features["trainable"]]
    if "target_time" in selected:
        target_times = pd.to_datetime(selected["target_time"], utc=True, errors="raise")
        if not target_times.equals(selected_times + pd.Timedelta(minutes=15)):
            raise ValueError("Target times must equal the next quarter-hour boundary")
    if "canonical_target_eligible" in selected:
        eligibility = selected["canonical_target_eligible"]
        if (
            not pd.api.types.is_bool_dtype(eligibility)
            or eligibility.isna().any()
            or not eligibility.all()
        ):
            raise ValueError("Trainable source boundaries must be canonically eligible")
    if (
        (selected_times < pd.Timestamp("2026-10-06T00:00:00Z")).any()
        or (selected_times >= pd.Timestamp("2026-10-28T00:00:00Z")).any()
    ):
        raise ValueError("Training observations must belong to the October development period")
    returns = selected["target_log_return_15m"].to_numpy(dtype=float)
    if (returns == 0).any() or not np.array_equal(
        selected["target"].to_numpy(), (returns > 0).astype(int)
    ):
        raise ValueError("Binary targets disagree with the direction of their target returns")
    frame, X, y = prepare_research_frame(features, V2_PREDICTOR_CONTRACT_VERSION)
    if (
        len(frame) != len(selected)
        or not frame["timestamp"].reset_index(drop=True).equals(
            selected_times.reset_index(drop=True)
        )
    ):
        raise ValueError("Research preparation changed the approved training population")
    if tuple(X.columns) != frozen_predictors()["full_v2"]:
        raise ValueError("Prepared predictor ordering differs from the frozen contract")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("All training predictors must be finite; imputation is not a repair")
    return frame, X, y


def _new_pipeline() -> Pipeline:
    return _model_factories(localization_config())["extra_trees"]().pipeline


def _configuration(pipeline: Pipeline) -> dict[str, Any]:
    imputer = pipeline.named_steps["imputer"].get_params(deep=False)
    if isinstance(imputer["missing_values"], float) and np.isnan(imputer["missing_values"]):
        imputer["missing_values"] = {"nonfinite_float": "NaN"}
    return {
        "pipeline": ["imputer", "model"],
        "imputer": imputer,
        "model_type": "sklearn.ensemble.ExtraTreesClassifier",
        "model": pipeline.named_steps["model"].get_params(deep=False),
    }


def verified_probabilities(
    pipeline: Pipeline, X: pd.DataFrame, predictors: tuple[str, ...]
) -> np.ndarray:
    if tuple(X.columns) != predictors:
        raise ValueError("Predictor order does not match the frozen model")
    if tuple(pipeline.feature_names_in_) != predictors:
        raise ValueError("Fitted pipeline predictor order does not match the contract")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Prediction inputs must be finite")
    if not np.array_equal(pipeline.classes_, [0, 1]):
        raise ValueError("Model classes must be ordered DOWN, UP")
    probabilities = pipeline.predict_proba(X)
    if (
        probabilities.shape != (len(X), 2)
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    ):
        raise ValueError("Invalid model probability outputs")
    return probabilities


def _checked_bytes(path: Path, expected_sha256: str) -> bytes:
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ):
        raise ValueError("A trusted SHA256 is required")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError(f"SHA256 mismatch: {path.name}")
    return data


def _load_pipeline(
    data: bytes, predictors: tuple[str, ...], configuration: dict[str, Any]
) -> Pipeline:
    # Only callers holding hash-checked bytes from a trusted manifest may enter here.
    pipeline = pickle.loads(data)
    expected = _new_pipeline()
    if (
        type(pipeline) is not Pipeline
        or list(pipeline.named_steps) != ["imputer", "model"]
        or any(
            type(pipeline.named_steps[name]) is not type(expected.named_steps[name])
            for name in ("imputer", "model")
        )
        or _configuration(pipeline) != configuration
        or configuration != _configuration(expected)
        or tuple(pipeline.feature_names_in_) != predictors
    ):
        raise ValueError("Serialized model configuration or predictor order mismatch")
    return pipeline


def load_frozen_models(
    directory: str | Path,
    *,
    trusted_manifest_sha256: str,
    require_official: bool = True,
) -> tuple[dict[str, Pipeline], dict[str, Any]]:
    """Load only evidence whose manifest hash was independently trusted.

    Hashes are integrity checks, not authentication. Never accept a manifest/hash
    pair from an untrusted source: pickle can execute code.
    """
    root = Path(directory)
    manifest = json.loads(_checked_bytes(root / "manifest.json", trusted_manifest_sha256))
    if manifest["manifest_version"] != 1:
        raise ValueError("Unsupported freeze manifest version")
    if manifest["status"] not in (OFFICIAL_STATUS, DEVELOPMENT_STATUS):
        raise ValueError("Unrecognized freeze status")
    if require_official and manifest["status"] != OFFICIAL_STATUS:
        raise ValueError("Development-only artifacts are not an official prospective freeze")
    if (
        manifest["snapshot_sha256"] != SNAPSHOT_SHA256
        or manifest["training"]["observation_count"] != TRAINING_OBSERVATIONS
        or manifest["predictor_contract_version"] != V2_PREDICTOR_CONTRACT_VERSION
        or manifest["family_contract_version"] != FAMILY_CONTRACT_VERSION
        or manifest["environment"] != _environment()
        or manifest["schedule"] != {
            "start": WINDOW_START, "end_exclusive": WINDOW_END,
            "freeze_deadline": FREEZE_DEADLINE,
            "prediction_deadline_seconds": 60, "probability_threshold": 0.5,
        }
    ):
        raise ValueError("Frozen snapshot, contracts, population, or environment mismatch")
    training = manifest["training"]
    if not (
        pd.Timestamp("2026-10-06T00:00:00Z")
        <= pd.Timestamp(training["timestamp_min"])
        <= pd.Timestamp(training["timestamp_max"])
        < pd.Timestamp("2026-10-28T00:00:00Z")
        and pd.Timestamp(training["timestamp_max"]) + pd.Timedelta(minutes=15)
        <= pd.Timestamp(training["max_label_available_time"])
        < pd.Timestamp(WINDOW_START)
        and pd.Timestamp(training["max_label_available_time"])
        < pd.Timestamp(training["started_at"])
        <= pd.Timestamp(training["completed_at"])
        <= pd.Timestamp(manifest["freeze_completed_at"])
    ):
        raise ValueError("Frozen training population or label-availability timing mismatch")
    if manifest["status"] == OFFICIAL_STATUS:
        if (
            manifest["source"]["dirty"]
            or manifest["source"]["working_tree_state"]
            or manifest["approved_implementation_sha"] != manifest["source"]["commit_sha"]
            or not re.fullmatch(r"[0-9a-f]{40}", manifest["approved_implementation_sha"] or "")
            or not re.fullmatch(r"[0-9a-f]{40}", manifest["approved_protocol_sha"] or "")
            or pd.Timestamp(manifest["freeze_completed_at"]) > pd.Timestamp(FREEZE_DEADLINE)
        ):
            raise ValueError("Official freeze approval or deadline evidence is invalid")
    predictors = frozen_predictors()
    if set(manifest["models"]) != set(predictors):
        raise ValueError("Freeze must contain exactly the candidate and control")
    verified = {}
    for name, columns in predictors.items():
        record = manifest["models"][name]
        if record["predictors"] != list(columns) or record["filename"] != f"{name}.pkl":
            raise ValueError("Manifest predictor ordering or artifact filename mismatch")
        verified[name] = _checked_bytes(root / record["filename"], record["sha256"])
    # Check every artifact and metadata record before deserializing either model.
    expected_configuration = _configuration(_new_pipeline())
    if any(
        record["configuration"] != expected_configuration
        for record in manifest["models"].values()
    ):
        raise ValueError("Frozen model configuration differs from the research factory")
    models = {
        name: _load_pipeline(
            verified[name], columns, manifest["models"][name]["configuration"]
        )
        for name, columns in predictors.items()
    }
    return models, manifest


def _approval(
    state: dict[str, Any],
    *,
    development: bool,
    approved_implementation_sha: str | None,
    approved_protocol_sha: str | None,
) -> None:
    if development:
        if approved_implementation_sha or approved_protocol_sha:
            raise ValueError("Development runs cannot claim official revision approval")
        return
    if state["dirty"]:
        raise ValueError("Official freezing requires a clean working tree")
    for sha in (approved_implementation_sha, approved_protocol_sha):
        if not sha or not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError("Official freezing requires explicit full approved Git SHAs")
    if approved_implementation_sha != state["commit_sha"]:
        raise ValueError("Approved implementation SHA must match the running revision")
    _git("merge-base", "--is-ancestor", str(approved_protocol_sha), state["commit_sha"])
    committed_protocol = _git("rev-parse", f"{approved_protocol_sha}:{PROTOCOL_PATH}")
    if committed_protocol != state["protocol_git_blob"]:
        raise ValueError("Approved protocol does not match the current protocol")
    if pd.Timestamp(_utc_now()) > pd.Timestamp(FREEZE_DEADLINE):
        raise ValueError("Official freeze deadline has passed; a prospective amendment is required")


def _write_durable(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _publish_directory(staging: Path, output: Path) -> None:
    """Atomic no-replace rename, including protection from concurrent creators."""
    if sys.platform == "win32":
        os.rename(staging, output)
    elif sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        rename = getattr(libc, "renameat2", None)
        if rename is None:
            raise OSError("Atomic no-replace directory publication is unavailable")
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                           ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(staging), -100, os.fsencode(output), 1) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(output))
    else:
        raise OSError("Atomic no-replace publication is supported only on Windows and Linux")


def freeze_models(
    training_input: str | Path,
    output_directory: str | Path,
    *,
    development: bool,
    approved_implementation_sha: str | None = None,
    approved_protocol_sha: str | None = None,
) -> dict[str, Any]:
    source = Path(training_input).resolve(strict=True)
    output = Path(os.path.abspath(output_directory))
    if os.path.lexists(output):
        raise FileExistsError("Freeze output must be a new, previously unused destination")
    if not output.parent.is_dir():
        raise ValueError("Output parent directory must already exist")
    if output.resolve().is_relative_to(REPOSITORY):
        raise ValueError("Freeze artifacts must be stored outside the implementation checkout")
    state = _source_state()
    _approval(
        state, development=development,
        approved_implementation_sha=approved_implementation_sha,
        approved_protocol_sha=approved_protocol_sha,
    )
    snapshot = _checked_bytes(source, SNAPSHOT_SHA256)
    training_started = _utc_now()
    frame, X, y = validate_training_input(pd.read_parquet(io.BytesIO(snapshot)))
    if pd.Timestamp(frame["label_available_time"].max()) >= pd.Timestamp(training_started):
        raise ValueError("All training labels must be available before training begins")
    predictors = frozen_predictors()
    models = {}
    for name, columns in predictors.items():
        models[name] = _new_pipeline().fit(X.loc[:, list(columns)], y)
    training_completed = _utc_now()
    if _sha256_file(source) != SNAPSHOT_SHA256:
        raise ValueError("Training snapshot changed during training")
    manifest = {
        "manifest_version": 1,
        "status": DEVELOPMENT_STATUS if development else OFFICIAL_STATUS,
        "approved_implementation_sha": approved_implementation_sha,
        "approved_protocol_sha": approved_protocol_sha,
        "source": state,
        "snapshot_sha256": SNAPSHOT_SHA256,
        "predictor_contract_version": V2_PREDICTOR_CONTRACT_VERSION,
        "family_contract_version": FAMILY_CONTRACT_VERSION,
        "environment": _environment(),
        "training": {
            "started_at": training_started,
            "completed_at": training_completed,
            "observation_count": len(frame),
            "timestamp_min": frame["timestamp"].min().isoformat(),
            "timestamp_max": frame["timestamp"].max().isoformat(),
            "max_label_available_time": frame["label_available_time"].max().isoformat(),
        },
        "schedule": {
            "start": WINDOW_START, "end_exclusive": WINDOW_END,
            "freeze_deadline": FREEZE_DEADLINE,
            "prediction_deadline_seconds": 60, "probability_threshold": 0.5,
        },
        "freeze_started_at": training_started,
        "models": {},
    }
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.staging-", dir=output.parent
    ) as temporary:
        staging = Path(temporary)
        for name, columns in predictors.items():
            pipeline = models[name]
            matrix = X.loc[:, list(columns)]
            expected = verified_probabilities(pipeline, matrix, columns)
            if not np.array_equal(expected, verified_probabilities(pipeline, matrix, columns)):
                raise ValueError("In-memory inference is not deterministic")
            data = pickle.dumps(pipeline, protocol=5)
            digest = hashlib.sha256(data).hexdigest()
            filename = f"{name}.pkl"
            _write_durable(staging / filename, data)
            configuration = _configuration(pipeline)
            loaded = _load_pipeline(
                _checked_bytes(staging / filename, digest), columns, configuration
            )
            actual = verified_probabilities(loaded, matrix, columns)
            if (
                not np.array_equal(expected, actual)
                or not np.array_equal(pipeline.predict(matrix), loaded.predict(matrix))
            ):
                raise ValueError("Serialized model predictions differ from the fitted pipeline")
            manifest["models"][name] = {
                "filename": filename, "sha256": digest,
                "predictors": list(columns), "configuration": configuration,
            }
        manifest["freeze_completed_at"] = _utc_now()
        if _source_state() != state:
            raise ValueError("Source revision or working-tree state changed during freezing")
        _approval(
            state, development=development,
            approved_implementation_sha=approved_implementation_sha,
            approved_protocol_sha=approved_protocol_sha,
        )
        manifest_data = (
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        _write_durable(staging / "manifest.json", manifest_data)
        manifest_sha256 = hashlib.sha256(manifest_data).hexdigest()
        load_frozen_models(
            staging, trusted_manifest_sha256=manifest_sha256, require_official=not development,
        )
        if _sha256_file(source) != SNAPSHOT_SHA256:
            raise ValueError("Training snapshot changed before publication")
        if _source_state() != state:
            raise ValueError("Source revision or working-tree state changed before publication")
        _approval(
            state, development=development,
            approved_implementation_sha=approved_implementation_sha,
            approved_protocol_sha=approved_protocol_sha,
        )
        _publish_directory(staging, output)
    return {
        "output_directory": str(output), "status": manifest["status"],
        "trusted_manifest_sha256": manifest_sha256,
    }
