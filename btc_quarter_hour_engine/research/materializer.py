from __future__ import annotations

import errno
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping

import pandas as pd

from btc_quarter_hour_engine.acquisition.config import CoinbaseWebSocketConfig
from btc_quarter_hour_engine.market_data.replay import CanonicalReplayAccumulator
from btc_quarter_hour_engine.research.features import (
    PREDICTOR_CONTRACT_VERSION,
    _IngestFeatureReplay,
    _feature_buckets_frame,
    assemble_research_feature_frame_from_frames,
)
from btc_quarter_hour_engine.research.forward_dataset import (
    DiscoveredRawSegment,
    DiscoveredSegmentSet,
    _boundary_frame,
    _check_separate_output_root,
    _group_and_validate_segments,
    _iter_segment_frames,
    _validate_source_paths,
    discover_sealed_segments,
)
from btc_quarter_hour_engine.storage.forward_schema import FORWARD_SOURCE
from btc_quarter_hour_engine.targets import build_canonical_direction_targets


RESEARCH_MATERIALIZER_CHECKPOINT_VERSION = 2

_SEGMENT_FIELDS = {
    "session_id",
    "segment_index",
    "sha256",
    "metadata_sha256",
    "provenance_sha256",
    "raw_segment_schema_version",
    "frame_count",
    "first_ingest_time_utc",
    "last_ingest_time_utc",
}
_CHECKPOINT_FIELDS_V1 = {
    "version",
    "source",
    "product_id",
    "predictor_contract_version",
    "heartbeat_timeout_seconds",
    "session_id",
    "expected_next_frame_index",
    "first_ingest_time_utc",
    "last_ingest_time_utc",
    "processed_segments",
    "feature_replay",
    "canonical_replay",
}
_CHECKPOINT_FIELDS = {
    "version",
    "source",
    "product_id",
    "predictor_contract_version",
    "heartbeat_timeout_seconds",
    "session_id",
    "expected_next_frame_index",
    "total_processed_frame_count",
    "first_ingest_time_utc",
    "last_ingest_time_utc",
    "processed_segments",
    "processed_sessions",
}
_SESSION_CHECKPOINT_FIELDS = {
    "session_id",
    "expected_next_frame_index",
    "first_ingest_time_utc",
    "last_ingest_time_utc",
    "feature_replay",
    "canonical_replay",
}


@dataclass(slots=True)
class _SessionReplayState:
    session_id: str
    expected_next_frame_index: int
    first_ingest_time_utc: datetime
    last_ingest_time_utc: datetime
    feature_replay: _IngestFeatureReplay
    canonical_replay: CanonicalReplayAccumulator


@dataclass(frozen=True, slots=True)
class MaterializationResult:
    checkpoint_path: Path
    checkpoint_sha256: str
    materialization_path: Path
    diagnostics: dict[str, Any]


def _heartbeat_timeout(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Materializer heartbeat timeout must be a finite nonnegative number")
    try:
        timeout = float(value)
    except OverflowError as exc:
        raise ValueError(
            "Materializer heartbeat timeout must be a finite nonnegative number"
        ) from exc
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("Materializer heartbeat timeout must be a finite nonnegative number")
    return timeout


def _iso_utc(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Materializer timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _parse_checkpoint_time(value: Any, *, field: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Invalid materializer checkpoint timestamp: {field}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid materializer checkpoint timestamp: {field}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"Materializer checkpoint {field} must be timezone-aware UTC")
    utc = parsed.astimezone(timezone.utc)
    if _iso_utc(utc) != value:
        raise ValueError(f"Materializer checkpoint {field} must use canonical UTC ISO-8601")
    return utc


def _segment_record(segment: DiscoveredRawSegment) -> dict[str, Any]:
    return {
        "session_id": segment.session_id,
        "segment_index": segment.segment_index,
        "sha256": segment.sha256,
        "metadata_sha256": segment.metadata_sha256,
        "provenance_sha256": segment.provenance_sha256,
        "raw_segment_schema_version": segment.raw_segment_schema_version,
        "frame_count": segment.frame_count,
        "first_ingest_time_utc": _iso_utc(segment.first_ingest_time_utc),
        "last_ingest_time_utc": _iso_utc(segment.last_ingest_time_utc),
    }


def _validate_ledger(
    ledger: Any,
) -> tuple[dict[str, tuple[int, datetime, datetime]], int, datetime, datetime]:
    if not isinstance(ledger, list):
        raise ValueError("Materializer checkpoint processed_segments must be a list")
    total_frames = 0
    first_time: datetime | None = None
    last_time: datetime | None = None
    session_counts: dict[str, int] = {}
    session_bounds: dict[str, tuple[datetime, datetime]] = {}
    closed_sessions: set[str] = set()
    previous_session: str | None = None
    previous_segment_index = -1
    for index, record in enumerate(ledger):
        if not isinstance(record, dict) or set(record) != _SEGMENT_FIELDS:
            raise ValueError("Malformed materializer checkpoint processed segment")
        session_id = record["session_id"]
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("Invalid materializer checkpoint segment session_id")
        segment_index = record["segment_index"]
        if isinstance(segment_index, bool) or not isinstance(segment_index, int):
            raise ValueError("Materializer checkpoint segment indexes are malformed")
        new_session = session_id != previous_session
        if new_session:
            if session_id in closed_sessions or segment_index != 0:
                raise ValueError("Materializer checkpoint session segment indexes are noncontiguous")
            if previous_session is not None:
                closed_sessions.add(previous_session)
                previous_bounds = session_bounds[previous_session]
                next_first = _parse_checkpoint_time(
                    record["first_ingest_time_utc"], field="first_ingest_time_utc"
                )
                if next_first is None or next_first <= previous_bounds[1]:
                    raise ValueError("Materializer checkpoint sessions are not strictly chronological")
            previous_session = session_id
            previous_segment_index = -1
        if segment_index != previous_segment_index + 1:
            raise ValueError("Materializer checkpoint segment indexes are noncontiguous")
        previous_segment_index = segment_index
        for key in ("sha256", "metadata_sha256", "provenance_sha256"):
            digest = record[key]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"Invalid materializer checkpoint {key}")
        if not isinstance(record["raw_segment_schema_version"], str):
            raise ValueError("Invalid materializer checkpoint raw segment schema version")
        frame_count = record["frame_count"]
        if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
            raise ValueError("Invalid materializer checkpoint frame_count")
        first = _parse_checkpoint_time(record["first_ingest_time_utc"], field="first_ingest_time_utc")
        last = _parse_checkpoint_time(record["last_ingest_time_utc"], field="last_ingest_time_utc")
        if first is None or last is None or last < first:
            raise ValueError("Invalid materializer checkpoint ingest-time bounds")
        if index == 0:
            first_time = first
        if session_id not in session_bounds:
            session_bounds[session_id] = (first, last)
        else:
            session_first, session_last = session_bounds[session_id]
            if first < session_last:
                raise ValueError("Materializer checkpoint session ingest times regress")
            session_bounds[session_id] = (session_first, last)
        session_counts[session_id] = session_counts.get(session_id, 0) + frame_count
        last_time = last
        total_frames += frame_count
    if first_time is None or last_time is None:
        raise ValueError("Materializer checkpoint must contain a nonempty processed prefix")
    sessions = {
        session_id: (session_counts[session_id], *session_bounds[session_id])
        for session_id in session_counts
    }
    return sessions, total_frames, first_time, last_time


def _checkpoint_payload(
    *,
    source: str,
    product_id: str,
    sessions: list[_SessionReplayState],
    first_ingest_time_utc: datetime,
    last_ingest_time_utc: datetime,
    processed_segments: list[dict[str, Any]],
) -> dict[str, Any]:
    if not sessions:
        raise ValueError("Materializer checkpoint requires at least one processed session")
    latest = sessions[-1]
    return {
        "version": RESEARCH_MATERIALIZER_CHECKPOINT_VERSION,
        "source": source,
        "product_id": product_id,
        "predictor_contract_version": PREDICTOR_CONTRACT_VERSION,
        "heartbeat_timeout_seconds": latest.canonical_replay.heartbeat_timeout_seconds,
        "session_id": latest.session_id,
        "expected_next_frame_index": latest.expected_next_frame_index,
        "total_processed_frame_count": sum(
            session.expected_next_frame_index for session in sessions
        ),
        "first_ingest_time_utc": _iso_utc(first_ingest_time_utc),
        "last_ingest_time_utc": _iso_utc(last_ingest_time_utc),
        "processed_segments": processed_segments,
        "processed_sessions": [
            {
                "session_id": session.session_id,
                "expected_next_frame_index": session.expected_next_frame_index,
                "first_ingest_time_utc": _iso_utc(session.first_ingest_time_utc),
                "last_ingest_time_utc": _iso_utc(session.last_ingest_time_utc),
                "feature_replay": session.feature_replay.to_checkpoint(),
                "canonical_replay": session.canonical_replay.to_checkpoint(),
            }
            for session in sessions
        ],
    }


def _checkpoint_state(
    checkpoint: Mapping[str, Any],
    *,
    source: str,
    product_id: str,
    heartbeat_timeout_seconds: float,
) -> tuple[list[_SessionReplayState], datetime, datetime, list[dict[str, Any]]]:
    if "version" not in checkpoint:
        raise ValueError("Malformed materializer checkpoint keys")
    version = checkpoint["version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("Unsupported materializer checkpoint schema version")
    if version == 1:
        if set(checkpoint) != _CHECKPOINT_FIELDS_V1:
            raise ValueError("Malformed materializer checkpoint keys")
        checkpoint = {
            "version": RESEARCH_MATERIALIZER_CHECKPOINT_VERSION,
            "source": checkpoint["source"],
            "product_id": checkpoint["product_id"],
            "predictor_contract_version": checkpoint["predictor_contract_version"],
            "heartbeat_timeout_seconds": checkpoint["heartbeat_timeout_seconds"],
            "session_id": checkpoint["session_id"],
            "expected_next_frame_index": checkpoint["expected_next_frame_index"],
            "total_processed_frame_count": checkpoint["expected_next_frame_index"],
            "first_ingest_time_utc": checkpoint["first_ingest_time_utc"],
            "last_ingest_time_utc": checkpoint["last_ingest_time_utc"],
            "processed_segments": checkpoint["processed_segments"],
            "processed_sessions": [{
                "session_id": checkpoint["session_id"],
                "expected_next_frame_index": checkpoint["expected_next_frame_index"],
                "first_ingest_time_utc": checkpoint["first_ingest_time_utc"],
                "last_ingest_time_utc": checkpoint["last_ingest_time_utc"],
                "feature_replay": checkpoint["feature_replay"],
                "canonical_replay": checkpoint["canonical_replay"],
            }],
        }
    elif version != RESEARCH_MATERIALIZER_CHECKPOINT_VERSION:
        raise ValueError("Unsupported materializer checkpoint schema version")
    if set(checkpoint) != _CHECKPOINT_FIELDS:
        raise ValueError("Malformed materializer checkpoint keys")
    if checkpoint["source"] != source:
        raise ValueError("Materializer checkpoint source does not match requested source")
    if checkpoint["product_id"] != product_id:
        raise ValueError("Materializer checkpoint product_id does not match requested product")
    if checkpoint["predictor_contract_version"] != PREDICTOR_CONTRACT_VERSION:
        raise ValueError("Materializer checkpoint predictor contract does not match")
    timeout = _heartbeat_timeout(checkpoint["heartbeat_timeout_seconds"])
    if timeout != _heartbeat_timeout(heartbeat_timeout_seconds):
        raise ValueError(
            "Materializer checkpoint heartbeat timeout does not match requested configuration"
        )
    expected = checkpoint["expected_next_frame_index"]
    if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
        raise ValueError("Invalid materializer checkpoint expected_next_frame_index")

    processed = checkpoint["processed_segments"]
    ledger_sessions, frame_total, ledger_first, ledger_last = _validate_ledger(processed)
    if (
        isinstance(checkpoint["total_processed_frame_count"], bool)
        or not isinstance(checkpoint["total_processed_frame_count"], int)
        or checkpoint["total_processed_frame_count"] != frame_total
    ):
        raise ValueError("Materializer checkpoint total frame count does not match processed segments")
    sessions_data = checkpoint["processed_sessions"]
    if not isinstance(sessions_data, list) or not sessions_data:
        raise ValueError("Materializer checkpoint processed_sessions must be a nonempty list")
    sessions: list[_SessionReplayState] = []
    for record in sessions_data:
        if not isinstance(record, dict) or set(record) != _SESSION_CHECKPOINT_FIELDS:
            raise ValueError("Malformed materializer checkpoint session state")
        session_id = record["session_id"]
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("Invalid materializer checkpoint session_id")
        session_expected = record["expected_next_frame_index"]
        if (
            isinstance(session_expected, bool)
            or not isinstance(session_expected, int)
            or session_expected < 0
        ):
            raise ValueError("Invalid materializer checkpoint expected_next_frame_index")
        session_first = _parse_checkpoint_time(
            record["first_ingest_time_utc"], field="first_ingest_time_utc"
        )
        session_last = _parse_checkpoint_time(
            record["last_ingest_time_utc"], field="last_ingest_time_utc"
        )
        ledger_state = ledger_sessions.get(session_id)
        if (
            ledger_state is None
            or session_first is None
            or session_last is None
            or session_expected != ledger_state[0]
            or session_first != ledger_state[1]
            or session_last != ledger_state[2]
        ):
            raise ValueError("Materializer checkpoint session state does not match processed ledger")
        feature = _IngestFeatureReplay.from_checkpoint(record["feature_replay"])
        canonical = CanonicalReplayAccumulator.from_checkpoint(record["canonical_replay"])
        if feature.product_id != product_id or canonical.product_id != product_id:
            raise ValueError("Nested replay checkpoint product_id does not match")
        if canonical.session_id != session_id:
            raise ValueError("Nested canonical replay checkpoint session_id does not match")
        if feature.frames_seen != session_expected or canonical.frame_count != session_expected:
            raise ValueError("Nested replay checkpoint frame counts do not match materializer")
        if timeout != float(canonical.heartbeat_timeout_seconds):
            raise ValueError(
                "Materializer checkpoint heartbeat timeout does not match nested canonical replay"
            )
        sessions.append(_SessionReplayState(
            session_id=session_id,
            expected_next_frame_index=session_expected,
            first_ingest_time_utc=session_first,
            last_ingest_time_utc=session_last,
            feature_replay=feature,
            canonical_replay=canonical,
        ))
    if [session.session_id for session in sessions] != list(ledger_sessions):
        raise ValueError("Materializer checkpoint session order does not match processed ledger")
    latest_session = sessions[-1]
    if checkpoint["session_id"] != latest_session.session_id:
        raise ValueError("Materializer checkpoint latest session does not match processed sessions")
    if expected != latest_session.expected_next_frame_index:
        raise ValueError(
            "Materializer checkpoint expected_next_frame_index does not match latest session"
        )
    first = _parse_checkpoint_time(
        checkpoint["first_ingest_time_utc"], field="first_ingest_time_utc"
    )
    last = _parse_checkpoint_time(
        checkpoint["last_ingest_time_utc"], field="last_ingest_time_utc"
    )
    if first != ledger_first or last != ledger_last:
        raise ValueError("Materializer checkpoint ingest-time bounds do not match processed segments")
    if first is None or last is None:
        raise ValueError("Materializer checkpoint must contain a nonempty processed prefix")
    return sessions, first, last, processed


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError as exc:
        if exc.errno in {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS}:
            return
        raise
    try:
        os.fsync(descriptor)
    except OSError as exc:
        if exc.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS}:
            raise
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, payload: bytes, *, prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=path.parent,
            prefix=prefix,
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Mapping[str, Any], *, prefix: str) -> None:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    _atomic_write_bytes(path, serialized, prefix=prefix)


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        frame.to_parquet(temporary_path, index=False)
        with temporary_path.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_checkpoint(
    path: Path, *, source: str, product_id: str, heartbeat_timeout_seconds: float
):
    try:
        checkpoint = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Cannot read valid materializer checkpoint: {path}") from exc
    if not isinstance(checkpoint, dict):
        raise ValueError("Materializer checkpoint must be a JSON object")
    return _checkpoint_state(
        checkpoint,
        source=source,
        product_id=product_id,
        heartbeat_timeout_seconds=heartbeat_timeout_seconds,
    )


def _reject_json_constant(value: str):
    raise ValueError(f"Non-standard JSON constant: {value}")


def materialize_research(
    *,
    input_root: str | Path,
    output_root: str | Path,
    product_id: str = "BTC-USD",
    source: str = FORWARD_SOURCE,
    max_new_segments: int | None = None,
    checkpoint_every_segments: int = 1,
    websocket_config: CoinbaseWebSocketConfig | None = None,
) -> MaterializationResult:
    input_path = Path(input_root)
    output_path = Path(output_root)
    if not input_path.is_dir():
        raise ValueError(f"input_root must be an existing directory: {input_path}")
    _check_separate_output_root(input_path, output_path)
    if (
        isinstance(checkpoint_every_segments, bool)
        or not isinstance(checkpoint_every_segments, int)
        or checkpoint_every_segments <= 0
    ):
        raise ValueError("checkpoint_every_segments must be a positive integer")
    if max_new_segments is not None and (
        isinstance(max_new_segments, bool)
        or not isinstance(max_new_segments, int)
        or max_new_segments < 0
    ):
        raise ValueError("max_new_segments must be a nonnegative integer")

    config = websocket_config or CoinbaseWebSocketConfig(product_id=product_id)
    if config.product_id != product_id:
        raise ValueError("WebSocket configuration product_id does not match requested product")
    timeout = _heartbeat_timeout(config.heartbeat_timeout_seconds)

    # This is the sole discovery call; all processing below uses this fixed snapshot.
    discovered = discover_sealed_segments(input_path, product_id=product_id, source=source)
    if not isinstance(discovered, DiscoveredSegmentSet):
        raise ValueError("Sealed segment discovery returned an invalid snapshot")
    segments = discovered.segments
    _validate_source_paths(input_path, segments)
    if any(segment.source != source for segment in segments):
        raise ValueError("Discovered segment source does not match requested source")
    if any(segment.product_id != product_id for segment in segments):
        raise ValueError("Discovered segment product does not match requested product")
    grouped = _group_and_validate_segments(segments)
    segments = tuple(
        segment
        for session_segments in grouped.values()
        for segment in session_segments
    )

    checkpoint_path = output_path / "checkpoint.json"
    if checkpoint_path.exists():
        (
            session_states,
            first_ingest_time,
            last_ingest_time,
            processed_segments,
        ) = _read_checkpoint(
            checkpoint_path,
            source=source,
            product_id=product_id,
            heartbeat_timeout_seconds=timeout,
        )
        current_records = [_segment_record(segment) for segment in segments]
        if len(processed_segments) > len(current_records):
            raise ValueError("Previously processed sealed segment is missing from discovery")
        for index, previous in enumerate(processed_segments):
            if previous != current_records[index]:
                raise ValueError(
                    f"Previously processed segment prefix changed at position {index}"
                )
        new_session_ids = {
            record["session_id"] for record in current_records[len(processed_segments):]
        }
        processed_session_ids = {session.session_id for session in session_states}
        newly_seen_sessions = new_session_ids.difference(processed_session_ids)
        if len(newly_seen_sessions) > 1:
            raise ValueError(
                "A resumed checkpoint may append segments from only one next collector session"
            )
    else:
        session_states = []
        processed_segments = []
        first_ingest_time = None
        last_ingest_time = None

    new_segments = list(segments[len(processed_segments):])
    if max_new_segments is not None:
        new_segments = new_segments[:max_new_segments]
    previous_count = len(processed_segments)
    new_frames_processed = 0
    uncommitted_segment_count = 0

    for segment in new_segments:
        if not session_states or session_states[-1].session_id != segment.session_id:
            if any(state.session_id == segment.session_id for state in session_states):
                raise ValueError("Materializer cannot return to an earlier collector session")
            session_states.append(_SessionReplayState(
                session_id=segment.session_id,
                expected_next_frame_index=0,
                first_ingest_time_utc=segment.first_ingest_time_utc,
                last_ingest_time_utc=segment.first_ingest_time_utc,
                feature_replay=_IngestFeatureReplay(product_id),
                canonical_replay=CanonicalReplayAccumulator(
                    product_id=product_id,
                    session_id=segment.session_id,
                    heartbeat_timeout_seconds=timeout,
                ),
            ))
        session_state = session_states[-1]
        if segment.segment_index != sum(
            1
            for record in processed_segments
            if record["session_id"] == segment.session_id
        ):
            raise ValueError(
                f"Unexpected segment progression for collector session {segment.session_id}"
            )
        for raw, connection_id, frame_index, ingest_time in _iter_segment_frames(
            segment,
            expected_frame_index=session_state.expected_next_frame_index,
        ):
            if ingest_time < session_state.last_ingest_time_utc:
                raise ValueError(
                    f"Ingest time regressed within collector session {segment.session_id}"
                )
            if last_ingest_time is not None and ingest_time < last_ingest_time:
                raise ValueError("Ingest time regressed across the processed session sequence")
            session_state.canonical_replay.consume(
                raw, connection_id, frame_index, ingest_time
            )
            session_state.feature_replay.consume(
                raw=raw,
                connection_id=connection_id,
                frame_index=frame_index,
                ingest_time_utc=ingest_time,
            )
            session_state.expected_next_frame_index = frame_index + 1
            first_ingest_time = first_ingest_time or ingest_time
            last_ingest_time = ingest_time
            session_state.last_ingest_time_utc = ingest_time
            new_frames_processed += 1

        processed_segments.append(_segment_record(segment))
        uncommitted_segment_count += 1
        if uncommitted_segment_count >= checkpoint_every_segments:
            if first_ingest_time is None or last_ingest_time is None:
                raise ValueError("Processed sealed prefix has no ingest-time bounds")
            checkpoint = _checkpoint_payload(
                source=source,
                product_id=product_id,
                sessions=session_states,
                first_ingest_time_utc=first_ingest_time,
                last_ingest_time_utc=last_ingest_time,
                processed_segments=processed_segments,
            )
            _atomic_write_json(checkpoint_path, checkpoint, prefix=".checkpoint.")
            uncommitted_segment_count = 0

    if uncommitted_segment_count:
        if first_ingest_time is None or last_ingest_time is None:
            raise ValueError("Processed sealed prefix has no ingest-time bounds")
        checkpoint = _checkpoint_payload(
            source=source,
            product_id=product_id,
            sessions=session_states,
            first_ingest_time_utc=first_ingest_time,
            last_ingest_time_utc=last_ingest_time,
            processed_segments=processed_segments,
        )
        _atomic_write_json(checkpoint_path, checkpoint, prefix=".checkpoint.")
    elif not checkpoint_path.exists():
        raise ValueError("No sealed segments were processed; cannot create a materializer checkpoint")

    if first_ingest_time is None or last_ingest_time is None:
        raise ValueError("Materializer checkpoint has no ingest-time bounds")
    observations = []
    for state in session_states:
        observations.extend(state.canonical_replay.finalize(
            session_started_at_utc=state.first_ingest_time_utc,
            session_completed_at_utc=state.last_ingest_time_utc,
            derived_at_utc=state.last_ingest_time_utc,
        ))
    boundaries = _boundary_frame(observations)
    if not boundaries.empty:
        boundaries = boundaries.sort_values(
            ["boundary_time_utc", "session_id"], kind="stable"
        ).reset_index(drop=True)
        if boundaries["boundary_time_utc"].duplicated().any():
            raise ValueError("Duplicate canonical boundary timestamps across collector sessions")
    targets = build_canonical_direction_targets(boundaries)
    latest_boundary = (
        boundaries["boundary_time_utc"].max()
        if not boundaries.empty
        else None
    )
    boundary_groups = {
        session_id: frame.reset_index(drop=True)
        for session_id, frame in boundaries.groupby("session_id", sort=False)
    } if not boundaries.empty else {}
    bucket_frames: list[pd.DataFrame] = []
    feature_frames: list[pd.DataFrame] = []
    for state in session_states:
        session_boundaries = boundary_groups.get(state.session_id)
        session_latest = (
            session_boundaries["boundary_time_utc"].max()
            if session_boundaries is not None and not session_boundaries.empty
            else None
        )
        session_buckets = _feature_buckets_frame(
            state.feature_replay,
            latest_boundary=session_latest,
        )
        if len(session_states) > 1:
            bucket_frames.append(
                session_buckets.reset_index().assign(session_id=state.session_id)
            )
        else:
            bucket_frames.append(session_buckets.reset_index())
        if session_boundaries is not None and not session_boundaries.empty:
            session_targets = targets.loc[
                targets["timestamp"].isin(session_boundaries["boundary_time_utc"])
            ]
            feature_frames.append(
                assemble_research_feature_frame_from_frames(
                    session_boundaries,
                    session_targets,
                    session_buckets,
                )
            )
    buckets_output = (
        pd.concat(bucket_frames, ignore_index=True)
        if bucket_frames
        else pd.DataFrame(columns=["bucket_end"])
    )
    features = (
        pd.concat(feature_frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        if feature_frames
        else assemble_research_feature_frame_from_frames(
            boundaries, targets, pd.DataFrame()
        )
    )

    output_path.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "canonical_boundaries.parquet": boundaries,
        "direction_targets.parquet": targets,
        "feature_buckets.parquet": buckets_output,
        "research_features.parquet": features,
    }
    artifact_hashes: dict[str, str] = {}
    for name, frame in artifacts.items():
        artifact_path = output_path / name
        _atomic_write_parquet(frame, artifact_path)
        artifact_hashes[name] = _sha256_file(artifact_path)

    checkpoint_sha256 = _sha256_file(checkpoint_path)
    latest_session = session_states[-1]
    total_processed_frames = sum(
        state.expected_next_frame_index for state in session_states
    )
    checkpoint_schema_version = json.loads(
        checkpoint_path.read_text(encoding="utf-8")
    )["version"]
    eligible_boundaries = int(boundaries["canonical_target_eligible"].sum())
    eligible_targets = int(targets["target_eligible"].sum())
    diagnostics = {
        "status": "success",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "discovered_sealed_segments": len(segments),
        "previously_processed_segments": previous_count,
        "source": source,
        "product_id": product_id,
        "predictor_contract_version": PREDICTOR_CONTRACT_VERSION,
        "heartbeat_timeout_seconds": latest_session.canonical_replay.heartbeat_timeout_seconds,
        "session_id": latest_session.session_id,
        "processed_session_count": len(session_states),
        "ordered_processed_session_ids": [
            state.session_id for state in session_states
        ],
        "latest_processed_session_id": latest_session.session_id,
        "processed_segment_count": len(processed_segments),
        "total_processed_frame_count": total_processed_frames,
        "expected_next_frame_index": latest_session.expected_next_frame_index,
        "first_ingest_time_utc": _iso_utc(first_ingest_time),
        "last_ingest_time_utc": _iso_utc(last_ingest_time),
        "active_partial_count_ignored": discovered.active_partial_count_ignored,
        "new_segments_processed_this_run": len(processed_segments) - previous_count,
        "new_frames_processed_this_run": new_frames_processed,
        "canonical_boundary_row_count": len(boundaries),
        "eligible_boundary_count": eligible_boundaries,
        "target_row_count": len(targets),
        "eligible_target_count": eligible_targets,
        "feature_bucket_count": len(buckets_output),
        "research_feature_row_count": len(features),
        "feature_ready_count": int(features["feature_ready"].sum()),
        "trainable_count": int(features["trainable"].sum()),
        "latest_canonical_boundary_timestamp": (
            latest_boundary.isoformat().replace("+00:00", "Z")
            if latest_boundary is not None
            else None
        ),
        "artifact_sha256": artifact_hashes,
    }
    _atomic_write_json(
        output_path / "materialization.json",
        {
            "materializer_checkpoint_schema_version": checkpoint_schema_version,
            "checkpoint_sha256": checkpoint_sha256,
            **{
                key: value
                for key, value in diagnostics.items()
                if key not in {"status", "checkpoint_path", "artifact_sha256"}
            },
            "artifact_sha256": artifact_hashes,
        },
        prefix=".materialization.",
    )
    return MaterializationResult(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        materialization_path=output_path / "materialization.json",
        diagnostics=diagnostics,
    )


__all__ = [
    "MaterializationResult",
    "RESEARCH_MATERIALIZER_CHECKPOINT_VERSION",
    "materialize_research",
]
