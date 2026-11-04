from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from btc_quarter_hour_engine.acquisition.coinbase_websocket import (
    parse_coinbase_ws_message,
)
from btc_quarter_hour_engine.research.features import (
    _IngestFeatureReplay,
    _feature_buckets_frame,
    _window_features,
)
from btc_quarter_hour_engine.research.feature_families_v2 import FAMILY_CONTRACT_VERSION
from btc_quarter_hour_engine.research.features_v2 import (
    V2_PREDICTOR_COLUMNS,
    V2_PREDICTOR_CONTRACT_VERSION,
    build_v2_research_feature_frame,
)
from btc_quarter_hour_engine.research.model_freeze import frozen_predictors
from btc_quarter_hour_engine.research.prospective_source import (
    CausalFeatureVector,
    CausalInputError,
    RawFrameRecord,
    RawFrameSnapshot,
    _frame_event_times,
    _iso,
    _utc,
    _is_boundary,
    compute_feature_fingerprint,
)


CHECKPOINT_SCHEMA_VERSION = 1
MAX_CHECKPOINT_BOOTSTRAP_FRAMES = 100_000
MAX_INCREMENTAL_TAIL_FRAMES = 100_000
_CHAIN_DOMAIN = b"btc-qh-prospective-raw-prefix-v1\0"
_CHECKPOINT_DOMAIN = b"btc-qh-feature-replay-checkpoint-v1\0"
_BUCKET_HISTORY = timedelta(minutes=240)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def initial_prefix_commitment(session_id: str, started_at_utc: datetime) -> str:
    """Return the domain-separated hash-chain seed for an immutable source session."""
    return hashlib.sha256(
        _CHAIN_DOMAIN
        + _canonical_json({
            "session_id": session_id,
            "session_started_at_utc": _iso(started_at_utc),
        })
    ).hexdigest()


def advance_prefix_commitment(previous: str, frame: RawFrameRecord) -> str:
    """Commit frame identity and provenance in source delivery order."""
    record = {
        "session_id": frame.session_id,
        "frame_index": frame.frame_index,
        "connection_id": frame.connection_id,
        "ingest_time_utc": _iso(frame.ingest_time_utc),
        "raw_frame_sha256": hashlib.sha256(frame.raw_bytes).hexdigest(),
    }
    return hashlib.sha256(
        bytes.fromhex(previous) + _CHAIN_DOMAIN + _canonical_json(record)
    ).hexdigest()


def _checkpoint_digest(envelope: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in envelope.items() if key != "integrity_sha256"}
    return hashlib.sha256(_CHECKPOINT_DOMAIN + _canonical_json(unsigned)).hexdigest()


@dataclass(frozen=True, slots=True)
class FeatureReplayCheckpoint:
    """Versioned immutable serialized replay state and its verified source anchor."""

    serialized: str

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        session_started_at_utc: datetime,
        cutoff_utc: datetime,
        last_frame_index: int,
        last_ingest_time_utc: datetime,
        last_event_time_utc: datetime,
        prefix_commitment_sha256: str,
        replay: _IngestFeatureReplay,
    ) -> "FeatureReplayCheckpoint":
        if replay.connection_id is None or replay.last_envelope_sequence_num is None:
            raise ValueError("A checkpoint requires an established connection and sequence")
        payload: dict[str, Any] = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "session_id": session_id,
            "session_started_at_utc": _iso(session_started_at_utc),
            "connection_id": replay.connection_id,
            "last_frame_index": last_frame_index,
            "last_accepted_sequence_num": replay.last_envelope_sequence_num,
            "cutoff_utc": _iso(cutoff_utc),
            "last_ingest_time_utc": _iso(last_ingest_time_utc),
            "last_event_time_utc": _iso(last_event_time_utc),
            "prefix_commitment_sha256": prefix_commitment_sha256,
            "product_id": replay.product_id,
            "feature_contract": V2_PREDICTOR_CONTRACT_VERSION,
            "feature_family_contract": FAMILY_CONTRACT_VERSION,
            "replay_state": replay.to_checkpoint(),
        }
        payload["integrity_sha256"] = _checkpoint_digest(payload)
        return cls(_canonical_json(payload).decode("utf-8"))

    @classmethod
    def from_dict(cls, value: Any) -> "FeatureReplayCheckpoint":
        if not isinstance(value, dict):
            raise ValueError("Checkpoint envelope must be an object")
        expected = {
            "schema_version",
            "session_id",
            "session_started_at_utc",
            "connection_id",
            "last_frame_index",
            "last_accepted_sequence_num",
            "cutoff_utc",
            "last_ingest_time_utc",
            "last_event_time_utc",
            "prefix_commitment_sha256",
            "product_id",
            "feature_contract",
            "feature_family_contract",
            "replay_state",
            "integrity_sha256",
        }
        if set(value) != expected:
            raise ValueError("Malformed checkpoint envelope keys")
        if type(value["schema_version"]) is not int or value["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported checkpoint schema version: {value['schema_version']!r}")
        if value["integrity_sha256"] != _checkpoint_digest(value):
            raise ValueError("Checkpoint integrity digest mismatch")
        for field in ("session_id", "connection_id", "product_id"):
            if not isinstance(value[field], str) or not value[field]:
                raise ValueError(f"Checkpoint {field} must be a nonempty string")
        if value["feature_contract"] != V2_PREDICTOR_CONTRACT_VERSION:
            raise ValueError("Checkpoint feature contract mismatch")
        if value["feature_family_contract"] != FAMILY_CONTRACT_VERSION:
            raise ValueError("Checkpoint feature-family contract mismatch")
        for field in ("last_frame_index", "last_accepted_sequence_num"):
            if type(value[field]) is not int or value[field] < 0:
                raise ValueError(f"Checkpoint {field} must be a nonnegative integer")
        for field in (
            "prefix_commitment_sha256",
            "integrity_sha256",
        ):
            digest = value[field]
            if not isinstance(digest, str) or len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"Checkpoint {field} must be lowercase SHA256")
        parsed_times: dict[str, datetime] = {}
        for field in (
            "session_started_at_utc",
            "cutoff_utc",
            "last_ingest_time_utc",
            "last_event_time_utc",
        ):
            if not isinstance(value[field], str):
                raise ValueError(f"Checkpoint {field} must be an ISO-8601 string")
            try:
                parsed_times[field] = _utc(
                    datetime.fromisoformat(value[field].replace("Z", "+00:00")),
                    field=field,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Checkpoint {field} is malformed") from exc
        if not _is_boundary(parsed_times["cutoff_utc"]):
            raise ValueError("Checkpoint cutoff is not an exact quarter-hour")
        if parsed_times["session_started_at_utc"] > parsed_times["cutoff_utc"] - _BUCKET_HISTORY:
            raise ValueError("Checkpoint session does not cover 240-minute history")
        if (
            parsed_times["last_ingest_time_utc"] > parsed_times["cutoff_utc"]
            or parsed_times["last_event_time_utc"] > parsed_times["cutoff_utc"]
        ):
            raise ValueError("Checkpoint source cutoffs exceed checkpoint boundary")
        # Reuse the repository's strict serializer parser for every nested order-book,
        # counter, and 240-minute bucket field before accepting a persisted checkpoint.
        replay = _IngestFeatureReplay.from_checkpoint(value["replay_state"])
        if replay.product_id != value["product_id"] or replay.product_id != "BTC-USD":
            raise ValueError("Checkpoint product identity mismatch")
        if (
            replay.connection_id != value["connection_id"]
            or replay.last_envelope_sequence_num != value["last_accepted_sequence_num"]
        ):
            raise ValueError("Checkpoint replay sequence or connection identity mismatch")
        if replay.malformed_frames or replay.sequence_gaps or replay.stale_frames:
            raise ValueError("Checkpoint contains failed replay-integrity counters")
        if replay.frames_seen != value["last_frame_index"] + 1:
            raise ValueError("Checkpoint frame index and replay count disagree")
        if replay.parsed_frames > replay.frames_seen or replay.applied_l2_envelopes > replay.parsed_frames:
            raise ValueError("Checkpoint replay counters are inconsistent")
        expected_buckets = {
            timestamp.to_pydatetime()
            for timestamp in pd.date_range(
                end=pd.Timestamp(parsed_times["cutoff_utc"]),
                periods=17,
                freq="15min",
                tz="UTC",
            )
        }
        if set(replay.buckets) != expected_buckets:
            raise ValueError("Checkpoint does not contain exactly 240-minute bucket history")
        _validate_checkpoint_book(replay)
        return cls(_canonical_json(value).decode("utf-8"))

    @property
    def data(self) -> dict[str, Any]:
        # Do not expose mutable state owned by the checkpoint instance.
        return json.loads(self.serialized)

    @property
    def session_id(self) -> str:
        return self.data["session_id"]

    @property
    def last_frame_index(self) -> int:
        return self.data["last_frame_index"]

    @property
    def cutoff_utc(self) -> datetime:
        return _utc(
            datetime.fromisoformat(self.data["cutoff_utc"].replace("Z", "+00:00")),
            field="checkpoint cutoff",
        )

    @property
    def prefix_commitment_sha256(self) -> str:
        return self.data["prefix_commitment_sha256"]

    @classmethod
    def load(cls, path: str | Path) -> "FeatureReplayCheckpoint":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(value)

    def write_once(self, path: str | Path) -> None:
        """Publish atomically without replacing an existing checkpoint."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(self.serialized.encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
            # Hard-link creation is atomic and fails rather than overwriting.
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class IncrementalRawFrameSnapshot:
    """Verified source tail beginning immediately after a committed frame prefix."""

    session_id: str
    session_started_at_utc: datetime
    watermark_utc: datetime
    clock_health_verified: bool
    prefix_frame_index: int
    prefix_commitment_sha256: str
    frames: tuple[RawFrameRecord, ...]
    frame_commitments: tuple[tuple[int, str], ...] = ()


def _validate_checkpoint_book(replay: _IngestFeatureReplay) -> None:
    if not replay.book.is_synced():
        raise ValueError("Checkpoint order book is not synchronized")
    bid = replay.book.best_bid
    ask = replay.book.best_ask
    if (
        bid is None
        or ask is None
        or not math.isfinite(float(bid))
        or not math.isfinite(float(ask))
        or float(bid) <= 0
        or float(ask) <= float(bid)
    ):
        raise ValueError("Checkpoint order book is missing or crossed")


def _replay_frames(
    frames: tuple[RawFrameRecord, ...],
    *,
    session_id: str,
    expected_index: int,
    replay: _IngestFeatureReplay,
    cutoff: datetime,
    previous_ingest: datetime | None,
    prefix_commitment: str,
) -> tuple[int, datetime | None, datetime | None, str, int, int]:
    latest_event: datetime | None = None
    included_count = 0
    excluded_event_count = 0
    for frame in frames:
        if frame.session_id != session_id:
            raise CausalInputError("raw_frame_session_id_mismatch", source_session_id=session_id)
        ingest = _utc(frame.ingest_time_utc, field="frame ingest time")
        if ingest > cutoff:
            break
        if type(frame.frame_index) is not int or frame.frame_index != expected_index:
            raise CausalInputError("noncontiguous_raw_frame_index", source_session_id=session_id)
        if not isinstance(frame.connection_id, str) or not frame.connection_id:
            raise CausalInputError("raw_frame_missing_connection_id", source_session_id=session_id)
        if previous_ingest is not None and ingest < previous_ingest:
            raise CausalInputError("raw_frame_ingest_order_regressed", source_session_id=session_id)
        if not isinstance(frame.raw_bytes, bytes):
            raise CausalInputError("raw_frame_bytes_missing", source_session_id=session_id)
        raw_digest = hashlib.sha256(frame.raw_bytes).hexdigest()
        if raw_digest != frame.raw_frame_sha256:
            raise CausalInputError("raw_frame_hash_mismatch", source_session_id=session_id)
        prefix_commitment = advance_prefix_commitment(prefix_commitment, frame)
        try:
            event = parse_coinbase_ws_message(frame.raw_bytes)
            event_times = _frame_event_times(event)
        except (KeyError, TypeError, UnicodeError, ValueError) as exc:
            raise CausalInputError(
                "malformed_raw_frame_at_or_before_boundary", source_session_id=session_id
            ) from exc
        if any(event_time > cutoff for event_time in event_times):
            # A frame delivered by T but carrying a future event cannot be
            # checkpointed and then replayed later without retroactive repair.
            excluded_event_count += 1
            expected_index += 1
            previous_ingest = ingest
            continue
        if event_times:
            latest_event = max(latest_event, max(event_times)) if latest_event else max(event_times)
        replay.consume(
            raw=frame.raw_bytes,
            connection_id=frame.connection_id,
            frame_index=frame.frame_index,
            ingest_time_utc=ingest,
        )
        if replay.malformed_frames or replay.sequence_gaps or replay.stale_frames:
            raise CausalInputError("raw_frame_replay_integrity_failure", source_session_id=session_id)
        expected_index += 1
        previous_ingest = ingest
        included_count += 1
    return (
        expected_index,
        previous_ingest,
        latest_event,
        prefix_commitment,
        included_count,
        excluded_event_count,
    )


def create_feature_replay_checkpoint(
    snapshot: RawFrameSnapshot,
    cutoff_utc: datetime,
) -> FeatureReplayCheckpoint:
    """Build the first bounded replay checkpoint from one verified session prefix."""
    cutoff = _utc(cutoff_utc, field="checkpoint cutoff")
    if not _is_boundary(cutoff):
        raise ValueError("Checkpoint cutoff must be an exact UTC quarter-hour")
    started = _utc(snapshot.session_started_at_utc, field="session start")
    watermark = _utc(snapshot.watermark_utc, field="source watermark")
    if snapshot.clock_health_verified is not True or watermark < cutoff:
        raise CausalInputError("checkpoint_source_completeness_unverified", source_session_id=snapshot.session_id)
    if started > cutoff - _BUCKET_HISTORY:
        raise CausalInputError("insufficient_240_minute_session_history", source_session_id=snapshot.session_id)
    if not isinstance(snapshot.frames, tuple):
        raise CausalInputError("raw_frame_snapshot_must_be_immutable_tuple", source_session_id=snapshot.session_id)
    if len(snapshot.frames) > MAX_CHECKPOINT_BOOTSTRAP_FRAMES:
        raise CausalInputError(
            "checkpoint_bootstrap_exceeds_bounded_frame_limit",
            source_session_id=snapshot.session_id,
        )
    replay = _IngestFeatureReplay("BTC-USD")
    seed = initial_prefix_commitment(snapshot.session_id, started)
    expected, previous_ingest, latest_event, commitment, count, excluded_event = _replay_frames(
        snapshot.frames,
        session_id=snapshot.session_id,
        expected_index=0,
        replay=replay,
        cutoff=cutoff,
        previous_ingest=None,
        prefix_commitment=seed,
    )
    if excluded_event:
        raise CausalInputError(
            "future_event_in_checkpoint_prefix",
            source_session_id=snapshot.session_id,
        )
    last_index = expected - 1
    if count == 0 or last_index < 0 or previous_ingest is None or latest_event is None:
        raise CausalInputError("no_causally_eligible_raw_frames", source_session_id=snapshot.session_id)
    if replay.frames_seen != count or replay.book.last_sequence_num is None:
        raise CausalInputError("checkpoint_prefix_not_fully_replayed", source_session_id=snapshot.session_id)
    _validate_checkpoint_book(replay)
    buckets = _feature_buckets_frame(replay, latest_boundary=pd.Timestamp(cutoff))
    expected_buckets = pd.date_range(end=pd.Timestamp(cutoff), periods=17, freq="15min", tz="UTC")
    if any(timestamp not in buckets.index for timestamp in expected_buckets):
        raise CausalInputError("checkpoint_missing_240_minute_history", source_session_id=snapshot.session_id)
    replay.buckets = {
        timestamp.to_pydatetime(): replay.buckets[timestamp.to_pydatetime()]
        for timestamp in expected_buckets
    }
    return FeatureReplayCheckpoint.create(
        session_id=snapshot.session_id,
        session_started_at_utc=started,
        cutoff_utc=cutoff,
        last_frame_index=last_index,
        last_ingest_time_utc=previous_ingest,
        last_event_time_utc=latest_event,
        prefix_commitment_sha256=commitment,
        replay=replay,
    )


def _make_vector(
    replay: _IngestFeatureReplay,
    *,
    session_id: str,
    session_started: datetime,
    watermark: datetime,
    boundary: datetime,
    source_provenance_sha256: str,
    source_frame_count: int,
    excluded_ingest_future_count: int,
    excluded_event_future_count: int,
) -> CausalFeatureVector:
    buckets = _feature_buckets_frame(replay, latest_boundary=pd.Timestamp(boundary))
    if buckets.empty or pd.Timestamp(boundary) not in buckets.index:
        raise CausalInputError("missing_asof_feature_bucket", data_ready=False, source_session_id=session_id)
    asof = buckets.loc[pd.Timestamp(boundary)]
    current_midpoint = float(asof["asof_midpoint"])
    data_ready = bool(asof["asof_book_synced"]) and np.isfinite(current_midpoint)
    if not data_ready:
        raise CausalInputError("order_book_not_synchronized_at_boundary", source_session_id=session_id)
    row: dict[str, Any] = {
        "timestamp": pd.Timestamp(boundary),
        "feature_ready": False,
        "target_eligible": False,
        "trainable": False,
        "target": np.nan,
        "asof_spread_bps": float(asof["asof_spread_bps"]),
        "asof_top_book_imbalance": float(asof["asof_top_book_imbalance"]),
    }
    for minutes in (15, 30, 60, 240):
        lag = pd.Timestamp(boundary) - pd.Timedelta(minutes=minutes)
        lag_midpoint = (
            float(buckets.at[lag, "asof_midpoint"])
            if lag in buckets.index and pd.notna(buckets.at[lag, "asof_midpoint"])
            else np.nan
        )
        row[f"momentum_log_return_{minutes}m"] = (
            float(np.log(current_midpoint / lag_midpoint))
            if np.isfinite(lag_midpoint) and lag_midpoint > 0 else np.nan
        )
        row.update(_window_features(
            buckets,
            timestamp=pd.Timestamp(boundary),
            window_minutes=minutes,
            current_midpoint=current_midpoint,
        ))
    row["feature_ready"] = bool(
        pd.Timestamp(boundary) >= pd.Timestamp(session_started) + pd.Timedelta(minutes=240)
        and asof["asof_book_synced"]
        and row["bbo_bucket_coverage_240m"] == 16
        and np.isfinite(row["momentum_log_return_240m"])
    )
    v2_frame = build_v2_research_feature_frame(pd.DataFrame([row]))
    v2_values = v2_frame.loc[:, list(V2_PREDICTOR_COLUMNS)].iloc[0]
    if not row["feature_ready"]:
        raise CausalInputError("v2_feature_readiness_contract_not_satisfied", data_ready=True, source_session_id=session_id)
    if v2_values.isna().any() or not np.isfinite(v2_values.to_numpy(dtype=float)).all():
        missing = [name for name, value in v2_values.items() if pd.isna(value) or not np.isfinite(float(value))]
        raise CausalInputError("missing_or_nonfinite_v2_predictors:" + ",".join(missing), data_ready=True, source_session_id=session_id)
    predictors = frozen_predictors()
    if len(predictors["range_volatility"]) != 12 or len(predictors["full_v2"]) != 91:
        raise CausalInputError("frozen_predictor_count_mismatch", data_ready=True, source_session_id=session_id)
    values = {name: float(v2_values[name]) for name in predictors["full_v2"]}
    fingerprint = compute_feature_fingerprint(
        values,
        candidate_predictors=predictors["range_volatility"],
        control_predictors=predictors["full_v2"],
        feature_contract=V2_PREDICTOR_CONTRACT_VERSION,
        feature_family_contract=FAMILY_CONTRACT_VERSION,
    )
    return CausalFeatureVector(
        boundary_utc=boundary,
        target_utc=boundary + timedelta(minutes=15),
        source_session_id=session_id,
        source_watermark_utc=watermark,
        source_provenance_sha256=source_provenance_sha256,
        source_frame_count=source_frame_count,
        excluded_ingest_future_count=excluded_ingest_future_count,
        excluded_event_future_count=excluded_event_future_count,
        feature_contract=V2_PREDICTOR_CONTRACT_VERSION,
        feature_family_contract=FAMILY_CONTRACT_VERSION,
        candidate_predictors=predictors["range_volatility"],
        control_predictors=predictors["full_v2"],
        values=values,
        feature_fingerprint=fingerprint,
        data_ready=True,
        feature_ready=True,
        clock_health_verified=True,
    )


def build_incremental_causal_feature_vector(
    tail: IncrementalRawFrameSnapshot,
    boundary_utc: datetime,
    checkpoint: FeatureReplayCheckpoint,
    *,
    timings: dict[str, float] | None = None,
) -> CausalFeatureVector:
    """Restore a verified checkpoint and process only source records after it."""
    checkpoint_started = time.perf_counter() if timings is not None else 0.0
    boundary = _utc(boundary_utc, field="prediction boundary")
    if not _is_boundary(boundary):
        raise CausalInputError("prediction_boundary_not_exact_utc_quarter_hour", source_session_id=tail.session_id)
    data = checkpoint.data
    if tail.clock_health_verified is not True:
        raise CausalInputError("source_clock_health_unverified", source_session_id=tail.session_id)
    started = _utc(tail.session_started_at_utc, field="session start")
    watermark = _utc(tail.watermark_utc, field="source watermark")
    if watermark < boundary:
        raise CausalInputError("source_watermark_before_boundary", source_session_id=tail.session_id)
    if tail.session_id != data["session_id"] or started != _utc(
        datetime.fromisoformat(data["session_started_at_utc"].replace("Z", "+00:00")),
        field="checkpoint session start",
    ):
        raise CausalInputError("checkpoint_source_session_mismatch", source_session_id=tail.session_id)
    if tail.prefix_frame_index != data["last_frame_index"] or tail.prefix_commitment_sha256 != data["prefix_commitment_sha256"]:
        raise CausalInputError("checkpoint_prefix_commitment_mismatch", source_session_id=tail.session_id)
    if boundary <= checkpoint.cutoff_utc:
        raise CausalInputError("checkpoint_not_strictly_before_prediction_boundary", source_session_id=tail.session_id)
    if not isinstance(tail.frames, tuple):
        raise CausalInputError("raw_frame_tail_must_be_immutable_tuple", source_session_id=tail.session_id)
    if len(tail.frames) > MAX_INCREMENTAL_TAIL_FRAMES:
        raise CausalInputError("raw_frame_tail_exceeds_bounded_replay_limit", source_session_id=tail.session_id)
    if not isinstance(tail.frame_commitments, tuple) or len(tail.frame_commitments) != len(tail.frames):
        raise CausalInputError("raw_frame_tail_commitments_missing", source_session_id=tail.session_id)
    commitment_by_index = dict(tail.frame_commitments)
    if len(commitment_by_index) != len(tail.frame_commitments):
        raise CausalInputError("raw_frame_tail_commitments_duplicated", source_session_id=tail.session_id)
    replay = _IngestFeatureReplay.from_checkpoint(data["replay_state"])
    if timings is not None:
        timings["checkpoint_restore_ms"] = (time.perf_counter() - checkpoint_started) * 1000
    _validate_checkpoint_book(replay)
    expected_index = data["last_frame_index"] + 1
    previous_ingest = _utc(
        datetime.fromisoformat(data["last_ingest_time_utc"].replace("Z", "+00:00")),
        field="checkpoint ingest cutoff",
    )
    commitment = data["prefix_commitment_sha256"]
    replay_started = time.perf_counter() if timings is not None else 0.0
    for frame in tail.frames:
        ingest = _utc(frame.ingest_time_utc, field="frame ingest time")
        if ingest > watermark:
            raise CausalInputError("frame_ingest_time_exceeds_source_watermark", source_session_id=tail.session_id)
        if ingest > boundary:
            break
        if type(frame.frame_index) is not int or frame.frame_index != expected_index:
            raise CausalInputError("noncontiguous_raw_frame_index", source_session_id=tail.session_id)
        if ingest < previous_ingest:
            raise CausalInputError("raw_frame_ingest_order_regressed", source_session_id=tail.session_id)
        if frame.session_id != tail.session_id or not isinstance(frame.connection_id, str) or not frame.connection_id:
            raise CausalInputError("raw_frame_identity_mismatch", source_session_id=tail.session_id)
        if not isinstance(frame.raw_bytes, bytes) or hashlib.sha256(frame.raw_bytes).hexdigest() != frame.raw_frame_sha256:
            raise CausalInputError("raw_frame_hash_mismatch", source_session_id=tail.session_id)
        try:
            event = parse_coinbase_ws_message(frame.raw_bytes)
            event_times = _frame_event_times(event)
        except (KeyError, TypeError, UnicodeError, ValueError) as exc:
            raise CausalInputError("malformed_raw_frame_at_or_before_boundary", source_session_id=tail.session_id) from exc
        commitment = advance_prefix_commitment(commitment, frame)
        if commitment_by_index.get(frame.frame_index) != commitment:
            raise CausalInputError("raw_frame_tail_commitment_mismatch", source_session_id=tail.session_id)
        expected_index += 1
        previous_ingest = ingest
        if any(event_time > boundary for event_time in event_times):
            raise CausalInputError(
                "future_event_prevents_incremental_continuity",
                source_session_id=tail.session_id,
            )
        replay.consume(raw=frame.raw_bytes, connection_id=frame.connection_id, frame_index=frame.frame_index, ingest_time_utc=ingest)
        if replay.malformed_frames or replay.sequence_gaps or replay.stale_frames:
            raise CausalInputError("raw_frame_replay_integrity_failure", source_session_id=tail.session_id)
    if timings is not None:
        timings["incremental_tail_replay_ms"] = (time.perf_counter() - replay_started) * 1000
    if replay.book.state.name != "SYNCED":
        raise CausalInputError("order_book_not_synchronized_at_boundary", source_session_id=tail.session_id)
    _validate_checkpoint_book(replay)
    # Bind this vector to the checkpoint commitment and only the causally
    # consumed tail. Future frames do not enter its provenance or fingerprint.
    provenance = hashlib.sha256(
        _CHECKPOINT_DOMAIN
        + _canonical_json({
            "session_id": tail.session_id,
            "checkpoint_integrity_sha256": data["integrity_sha256"],
            "boundary_utc": _iso(boundary),
            "last_consumed_frame_index": expected_index - 1,
            "tail_prefix_commitment_sha256": commitment,
        })
    ).hexdigest()
    feature_started = time.perf_counter() if timings is not None else 0.0
    vector = _make_vector(
        replay,
        session_id=tail.session_id,
        session_started=started,
        watermark=watermark,
        boundary=boundary,
        source_provenance_sha256=provenance,
        source_frame_count=replay.frames_seen,
        excluded_ingest_future_count=0,
        excluded_event_future_count=0,
    )
    if timings is not None:
        timings["feature_generation_ms"] = (time.perf_counter() - feature_started) * 1000
    return vector


def advance_feature_replay_checkpoint(
    tail: IncrementalRawFrameSnapshot,
    cutoff_utc: datetime,
    checkpoint: FeatureReplayCheckpoint,
) -> FeatureReplayCheckpoint:
    """Advance checkpoint state over a bounded contiguous tail, never rewriting it."""
    cutoff = _utc(cutoff_utc, field="checkpoint cutoff")
    if not _is_boundary(cutoff):
        raise ValueError("Checkpoint cutoff must be an exact UTC quarter-hour")
    data = checkpoint.data
    started = _utc(tail.session_started_at_utc, field="session start")
    watermark = _utc(tail.watermark_utc, field="source watermark")
    if watermark < cutoff or tail.clock_health_verified is not True:
        raise CausalInputError("checkpoint_source_completeness_unverified", source_session_id=tail.session_id)
    if tail.session_id != data["session_id"] or started != _utc(
        datetime.fromisoformat(data["session_started_at_utc"].replace("Z", "+00:00")),
        field="checkpoint session start",
    ):
        raise CausalInputError("checkpoint_source_session_mismatch", source_session_id=tail.session_id)
    if tail.prefix_frame_index != data["last_frame_index"] or tail.prefix_commitment_sha256 != data["prefix_commitment_sha256"]:
        raise CausalInputError("checkpoint_prefix_commitment_mismatch", source_session_id=tail.session_id)
    if cutoff <= checkpoint.cutoff_utc:
        raise CausalInputError("checkpoint_cutoff_not_strictly_advanced", source_session_id=tail.session_id)
    replay = _IngestFeatureReplay.from_checkpoint(data["replay_state"])
    if not isinstance(tail.frames, tuple):
        raise CausalInputError("raw_frame_tail_must_be_immutable_tuple", source_session_id=tail.session_id)
    if len(tail.frames) > MAX_INCREMENTAL_TAIL_FRAMES:
        raise CausalInputError("raw_frame_tail_exceeds_bounded_replay_limit", source_session_id=tail.session_id)
    expected_index = data["last_frame_index"] + 1
    previous_ingest = _utc(
        datetime.fromisoformat(data["last_ingest_time_utc"].replace("Z", "+00:00")),
        field="checkpoint ingest cutoff",
    )
    latest_event = _utc(
        datetime.fromisoformat(data["last_event_time_utc"].replace("Z", "+00:00")),
        field="checkpoint event cutoff",
    )
    commitment = data["prefix_commitment_sha256"]
    if not isinstance(tail.frame_commitments, tuple) or len(tail.frame_commitments) != len(tail.frames):
        raise CausalInputError("raw_frame_tail_commitments_missing", source_session_id=tail.session_id)
    commitment_by_index = dict(tail.frame_commitments)
    if len(commitment_by_index) != len(tail.frame_commitments):
        raise CausalInputError("raw_frame_tail_commitments_duplicated", source_session_id=tail.session_id)
    for frame in tail.frames:
        ingest = _utc(frame.ingest_time_utc, field="frame ingest time")
        if ingest > watermark:
            raise CausalInputError("frame_ingest_time_exceeds_source_watermark", source_session_id=tail.session_id)
        if ingest > cutoff:
            break
        if type(frame.frame_index) is not int or frame.frame_index != expected_index:
            raise CausalInputError("noncontiguous_raw_frame_index", source_session_id=tail.session_id)
        if ingest < previous_ingest:
            raise CausalInputError("raw_frame_ingest_order_regressed", source_session_id=tail.session_id)
        if frame.session_id != tail.session_id or not isinstance(frame.connection_id, str) or not frame.connection_id:
            raise CausalInputError("raw_frame_identity_mismatch", source_session_id=tail.session_id)
        if not isinstance(frame.raw_bytes, bytes) or hashlib.sha256(frame.raw_bytes).hexdigest() != frame.raw_frame_sha256:
            raise CausalInputError("raw_frame_hash_mismatch", source_session_id=tail.session_id)
        try:
            event = parse_coinbase_ws_message(frame.raw_bytes)
            event_times = _frame_event_times(event)
        except (KeyError, TypeError, UnicodeError, ValueError) as exc:
            raise CausalInputError("malformed_raw_frame_at_or_before_boundary", source_session_id=tail.session_id) from exc
        if any(event_time > cutoff for event_time in event_times):
            raise CausalInputError("future_event_in_checkpoint_prefix", source_session_id=tail.session_id)
        if event_times:
            latest_event = max(latest_event, max(event_times))
        commitment = advance_prefix_commitment(commitment, frame)
        if commitment_by_index.get(frame.frame_index) != commitment:
            raise CausalInputError("raw_frame_tail_commitment_mismatch", source_session_id=tail.session_id)
        replay.consume(raw=frame.raw_bytes, connection_id=frame.connection_id, frame_index=frame.frame_index, ingest_time_utc=ingest)
        if replay.malformed_frames or replay.sequence_gaps or replay.stale_frames:
            raise CausalInputError("raw_frame_replay_integrity_failure", source_session_id=tail.session_id)
        expected_index += 1
        previous_ingest = ingest
    last_index = expected_index - 1
    if last_index <= data["last_frame_index"] or previous_ingest is None:
        raise CausalInputError("checkpoint_tail_has_no_new_frames", source_session_id=tail.session_id)
    _validate_checkpoint_book(replay)
    buckets = _feature_buckets_frame(replay, latest_boundary=pd.Timestamp(cutoff))
    expected_buckets = pd.date_range(end=pd.Timestamp(cutoff), periods=17, freq="15min", tz="UTC")
    if any(timestamp not in buckets.index for timestamp in expected_buckets):
        raise CausalInputError("checkpoint_missing_240_minute_history", source_session_id=tail.session_id)
    replay.buckets = {
        timestamp.to_pydatetime(): replay.buckets[timestamp.to_pydatetime()]
        for timestamp in expected_buckets
    }
    return FeatureReplayCheckpoint.create(
        session_id=tail.session_id,
        session_started_at_utc=started,
        cutoff_utc=cutoff,
        last_frame_index=last_index,
        last_ingest_time_utc=previous_ingest,
        last_event_time_utc=latest_event,
        prefix_commitment_sha256=commitment,
        replay=replay,
    )


def tail_from_snapshot(
    snapshot: RawFrameSnapshot,
    checkpoint: FeatureReplayCheckpoint,
) -> IncrementalRawFrameSnapshot:
    """Offline verifier for tests/benchmarks; production should use journal.tail_snapshot()."""
    data = checkpoint.data
    if snapshot.session_id != data["session_id"]:
        raise CausalInputError("checkpoint_source_session_mismatch", source_session_id=snapshot.session_id)
    seed = initial_prefix_commitment(snapshot.session_id, snapshot.session_started_at_utc)
    commitment = seed
    expected = 0
    for frame in snapshot.frames:
        if frame.frame_index > data["last_frame_index"]:
            break
        if frame.frame_index != expected or hashlib.sha256(frame.raw_bytes).hexdigest() != frame.raw_frame_sha256:
            raise CausalInputError("checkpoint_prefix_source_integrity_failure", source_session_id=snapshot.session_id)
        commitment = advance_prefix_commitment(commitment, frame)
        expected += 1
    if expected != data["last_frame_index"] + 1 or commitment != data["prefix_commitment_sha256"]:
        raise CausalInputError("checkpoint_prefix_commitment_mismatch", source_session_id=snapshot.session_id)
    tail_frames = tuple(
        frame for frame in snapshot.frames
        if frame.frame_index > data["last_frame_index"]
    )
    if len(tail_frames) > MAX_INCREMENTAL_TAIL_FRAMES:
        raise CausalInputError("raw_frame_tail_exceeds_bounded_replay_limit", source_session_id=snapshot.session_id)
    return IncrementalRawFrameSnapshot(
        session_id=snapshot.session_id,
        session_started_at_utc=snapshot.session_started_at_utc,
        watermark_utc=snapshot.watermark_utc,
        clock_health_verified=snapshot.clock_health_verified,
        prefix_frame_index=data["last_frame_index"],
        prefix_commitment_sha256=commitment,
        frames=tail_frames,
        frame_commitments=tuple(
            (frame.frame_index, chain)
            for frame, chain in _tail_commitments(
                tail_frames,
                commitment,
                data["last_frame_index"],
            )
        ),
    )


def _tail_commitments(
    frames: tuple[RawFrameRecord, ...],
    seed: str,
    after_frame_index: int,
) -> tuple[tuple[RawFrameRecord, str], ...]:
    chain = seed
    expected_index = after_frame_index
    commitments = []
    for frame in frames:
        if frame.frame_index != expected_index + 1:
            raise CausalInputError("noncontiguous_raw_frame_index", source_session_id=frame.session_id)
        if hashlib.sha256(frame.raw_bytes).hexdigest() != frame.raw_frame_sha256:
            raise CausalInputError("raw_frame_hash_mismatch", source_session_id=frame.session_id)
        chain = advance_prefix_commitment(chain, frame)
        commitments.append((frame, chain))
        expected_index = frame.frame_index
    return tuple(commitments)
