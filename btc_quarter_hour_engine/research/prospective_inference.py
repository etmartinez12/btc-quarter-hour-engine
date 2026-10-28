from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from btc_quarter_hour_engine.research.model_freeze import (
    OFFICIAL_STATUS,
    frozen_predictors,
    load_frozen_models,
    verified_probabilities,
)
from btc_quarter_hour_engine.research.prospective_contract import (
    BASELINE_PREDICTION,
    FULL_V2_MODEL_SHA256,
    PROBABILITY_THRESHOLD,
    RANGE_MODEL_SHA256,
    TRUSTED_MANIFEST_SHA256,
)
from btc_quarter_hour_engine.research.prospective_source import CausalFeatureVector
from btc_quarter_hour_engine.research.prospective_source import (
    CausalInputError,
    RawFrameSnapshot,
    build_causal_feature_vector,
    compute_feature_fingerprint,
)
from btc_quarter_hour_engine.research.prospective_ledger import (
    ClockHealthError,
    ProspectiveLedger,
)


_EXPECTED_MODEL_HASHES = {
    "range_volatility": RANGE_MODEL_SHA256,
    "full_v2": FULL_V2_MODEL_SHA256,
}


def _utc(value: datetime, *, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _require_git_sha(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("Implementation revision must be a full lowercase Git SHA")
    return value


@dataclass(frozen=True, slots=True)
class FrozenModelBundle:
    models: dict[str, Any]
    model_sha256: dict[str, str]
    manifest_sha256: str
    manifest: dict[str, Any]

    def __post_init__(self) -> None:
        if self.manifest_sha256 != TRUSTED_MANIFEST_SHA256:
            raise ValueError("Frozen manifest digest does not match the independently trusted digest")
        if self.manifest.get("status") != OFFICIAL_STATUS:
            raise ValueError("Development-only models cannot be used for prospective inference")
        if self.model_sha256 != _EXPECTED_MODEL_HASHES:
            raise ValueError("Frozen model digests do not match the prospective protocol")
        if set(self.models) != set(_EXPECTED_MODEL_HASHES):
            raise ValueError("Exactly the frozen candidate and control models are required")


def load_official_models(directory: str | Path) -> FrozenModelBundle:
    """Load only the independently pinned official artifact set."""
    models, manifest = load_frozen_models(
        directory,
        trusted_manifest_sha256=TRUSTED_MANIFEST_SHA256,
        require_official=True,
    )
    if manifest.get("status") != OFFICIAL_STATUS:
        raise ValueError("Official prospective model freeze is required")
    model_hashes = {
        name: manifest["models"][name]["sha256"]
        for name in ("range_volatility", "full_v2")
    }
    if model_hashes != _EXPECTED_MODEL_HASHES:
        raise ValueError("Official model SHA256 values differ from the frozen protocol")
    return FrozenModelBundle(
        models=models,
        model_sha256=model_hashes,
        manifest_sha256=TRUSTED_MANIFEST_SHA256,
        manifest=manifest,
    )


def infer_causal_prediction(
    features: CausalFeatureVector,
    model_bundle: FrozenModelBundle,
    *,
    implementation_git_sha: str,
    generated_at_utc: datetime | None = None,
    generation_clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Create immutable prediction evidence without reading any target or label."""
    implementation_sha = _require_git_sha(implementation_git_sha)
    if generated_at_utc is not None and generation_clock is not None:
        raise ValueError("Provide either an explicit dry-run timestamp or a generation clock")
    boundary = _utc(features.boundary_utc, field="prediction boundary")
    target = _utc(features.target_utc, field="target boundary")
    if target != boundary + pd.Timedelta(minutes=15):
        raise ValueError("Prediction target must be exactly T+15 minutes")
    if not features.data_ready or not features.feature_ready:
        raise ValueError("Causal data and feature readiness are required")
    if not features.clock_health_verified:
        raise ValueError("Source clock health has not been verified")
    if features.feature_contract != "bbo-causal-v2":
        raise ValueError("Inference requires the frozen V2 feature contract")
    if features.feature_family_contract != "bbo-causal-v2-families-v1":
        raise ValueError("Inference requires the frozen V2 feature-family contract")

    predictors = frozen_predictors()
    if features.candidate_predictors != predictors["range_volatility"]:
        raise ValueError("Candidate predictor ordering differs from the frozen model")
    if features.control_predictors != predictors["full_v2"]:
        raise ValueError("Control predictor ordering differs from the frozen model")
    if tuple(features.values) != predictors["full_v2"] or len(features.values) != 91:
        raise ValueError("Feature vector ordering/count differs from the frozen V2 contract")
    normalized_values: dict[str, float] = {}
    for name in predictors["full_v2"]:
        value = features.values[name]
        if isinstance(value, bool):
            raise ValueError(f"Feature {name} must be finite numeric data")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Feature {name} must be finite numeric data") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"Feature {name} must be finite numeric data")
        normalized_values[name] = numeric
    if features.feature_fingerprint != compute_feature_fingerprint(
        normalized_values,
        candidate_predictors=predictors["range_volatility"],
        control_predictors=predictors["full_v2"],
        feature_contract=features.feature_contract,
        feature_family_contract=features.feature_family_contract,
    ):
        raise ValueError("Feature vector does not match its fingerprint")
    vector = pd.DataFrame(
        [[normalized_values[name] for name in predictors["full_v2"]]],
        columns=predictors["full_v2"],
    )
    candidate_matrix = vector.loc[:, list(predictors["range_volatility"])]
    control_matrix = vector.loc[:, list(predictors["full_v2"])]
    candidate_probabilities = verified_probabilities(
        model_bundle.models["range_volatility"],
        candidate_matrix,
        predictors["range_volatility"],
    )
    control_probabilities = verified_probabilities(
        model_bundle.models["full_v2"],
        control_matrix,
        predictors["full_v2"],
    )
    candidate_probability = float(candidate_probabilities[0, 1])
    control_probability = float(control_probabilities[0, 1])
    if generation_clock is not None:
        generated_at_utc = generation_clock()
    elif generated_at_utc is None:
        generated_at_utc = datetime.now(timezone.utc)
    generated_at = _utc(generated_at_utc, field="prediction generation timestamp")
    if not boundary <= generated_at < target:
        raise ValueError("Inference cannot be generated before T or at/after the target boundary")
    candidate_prediction = int(candidate_probability >= PROBABILITY_THRESHOLD)
    control_prediction = int(control_probability >= PROBABILITY_THRESHOLD)
    return {
        "boundary_utc": boundary,
        "target_utc": target,
        "data_ready": features.data_ready,
        "feature_ready": features.feature_ready,
        "clock_health_verified": features.clock_health_verified,
        "source_session_id": features.source_session_id,
        "source_watermark_utc": features.source_watermark_utc,
        "source_provenance_sha256": features.source_provenance_sha256,
        "source_frame_count": features.source_frame_count,
        "excluded_ingest_future_count": features.excluded_ingest_future_count,
        "excluded_event_future_count": features.excluded_event_future_count,
        "feature_contract": features.feature_contract,
        "feature_family_contract": features.feature_family_contract,
        "candidate_predictors": list(features.candidate_predictors),
        "control_predictors": list(features.control_predictors),
        "feature_fingerprint": features.feature_fingerprint,
        "feature_vector": normalized_values,
        "manifest_sha256": model_bundle.manifest_sha256,
        "candidate_model_sha256": model_bundle.model_sha256["range_volatility"],
        "control_model_sha256": model_bundle.model_sha256["full_v2"],
        "implementation_git_sha": implementation_sha,
        "generated_at_utc": generated_at,
        "persisted_at_utc": None,
        "candidate_probability": candidate_probability,
        "candidate_prediction": candidate_prediction,
        "control_probability": control_probability,
        "control_prediction": control_prediction,
        "baseline_prediction": BASELINE_PREDICTION,
        "timing_eligible": False,
        "failure_reasons": [],
    }


def process_boundary(
    snapshot: RawFrameSnapshot,
    boundary_utc: datetime,
    model_bundle: FrozenModelBundle,
    *,
    implementation_git_sha: str,
    ledger: ProspectiveLedger,
) -> dict[str, Any]:
    """Run one boundary and append its outcome or an explicit failure event."""
    boundary = _utc(boundary_utc, field="prediction boundary")
    try:
        features = build_causal_feature_vector(snapshot, boundary)
    except CausalInputError as exc:
        recorded_at = ledger.clock_health.now()
        ledger.append_attempt(
            boundary,
            event_type="missing_data",
            recorded_at_utc=recorded_at,
            reason=exc.reason,
            details={
                "data_ready": exc.data_ready,
                "feature_ready": exc.feature_ready,
                "source_session_id": exc.source_session_id,
            },
        )
        raise
    except Exception as exc:
        recorded_at = ledger.clock_health.now()
        ledger.append_attempt(
            boundary,
            event_type="inference_failed",
            recorded_at_utc=recorded_at,
            reason=f"feature_generation_failed:{type(exc).__name__}:{exc}",
            details={"source_session_id": snapshot.session_id},
        )
        raise
    try:
        evidence = infer_causal_prediction(
            features,
            model_bundle,
            implementation_git_sha=implementation_git_sha,
            generation_clock=ledger.clock_health.now,
        )
    except ClockHealthError as exc:
        observed = ledger.clock_health.observed_wall_utc
        if observed is not None:
            ledger.append_attempt(
                boundary,
                event_type="clock_skew",
                recorded_at_utc=observed,
                reason=str(exc),
                details={"source_session_id": features.source_session_id},
            )
        raise
    except Exception as exc:
        recorded_at = ledger.clock_health.now()
        event_type = (
            "stale_attempt"
            if recorded_at >= boundary + timedelta(minutes=15)
            else "late_attempt"
            if recorded_at > boundary + timedelta(seconds=60)
            else "invalid_attempt"
            if recorded_at < boundary
            else "inference_failed"
        )
        ledger.append_attempt(
            boundary,
            event_type=event_type,
            recorded_at_utc=recorded_at,
            reason=str(exc) or "inference_failed",
            details={
                "data_ready": features.data_ready,
                "feature_ready": features.feature_ready,
                "source_session_id": features.source_session_id,
            },
        )
        raise
    return ledger.append_prediction(
        evidence,
    )
