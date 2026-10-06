from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Iterator, Sequence

import pandas as pd

from btc_quarter_hour_engine.acquisition.config import CoinbaseWebSocketConfig
from btc_quarter_hour_engine.market_data.boundary_observations import QuarterHourObservation
from btc_quarter_hour_engine.market_data.replay import CanonicalReplayAccumulator
from btc_quarter_hour_engine.storage.forward_schema import (
    COINBASE_WS_RAW_SEGMENT_SCHEMA_VERSION,
    DATA_KIND_WEBSOCKET_SEGMENTS,
    FORWARD_SOURCE,
)
from btc_quarter_hour_engine.storage.manifest import _git_sha
from btc_quarter_hour_engine.storage.research_schema import (
    DATA_KIND_RESEARCH_BOUNDARIES,
    DATA_KIND_RESEARCH_TARGETS,
    RESEARCH_BOUNDARY_SCHEMA_VERSION,
    RESEARCH_DATASET_SCHEMA_VERSION,
    RESEARCH_INPUT_MODE,
    RESEARCH_MANIFEST_SCHEMA_VERSION,
    RESEARCH_TARGET_SCHEMA_VERSION,
)
from btc_quarter_hour_engine.targets import build_canonical_direction_targets


@dataclass(frozen=True, slots=True)
class DiscoveredRawSegment:
    source: str
    product_id: str
    session_id: str
    segment_index: int
    path: Path
    metadata_path: Path
    sha256: str
    byte_count: int
    frame_count: int
    first_ingest_time_utc: datetime
    last_ingest_time_utc: datetime
    raw_segment_schema_version: str
    metadata_sha256: str
    provenance_sha256: str


@dataclass(frozen=True, slots=True)
class DiscoveredSegmentSet:
    segments: tuple[DiscoveredRawSegment, ...]
    active_partial_count_ignored: int


@dataclass(frozen=True, slots=True)
class ResearchDatasetResult:
    dataset_id: str
    dataset_dir: Path
    manifest: dict[str, Any]
    manifest_path: Path
    boundary_artifact: Path
    target_artifact: Path
    boundary_count: int
    eligible_boundary_count: int
    target_candidate_count: int
    eligible_target_count: int
    sealed_segment_count: int
    raw_frame_count: int


def _utc_datetime(value: Any, *, label: str, path: Path) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"Invalid {label} in {path}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid {label} in {path}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"Naive {label} in {path}")
    return parsed.astimezone(timezone.utc)


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _segment_from_metadata(metadata_path: Path, *, source: str, product_id: str) -> DiscoveredRawSegment:
    try:
        metadata_bytes = metadata_path.read_bytes()
        metadata = json.loads(metadata_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read sealed segment metadata: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid sealed segment metadata: {metadata_path}")
    if metadata.get("source") != source:
        raise ValueError(f"Raw segment source mismatch in {metadata_path}")
    if metadata.get("data_kind") != DATA_KIND_WEBSOCKET_SEGMENTS:
        raise ValueError(f"Raw segment data kind mismatch in {metadata_path}")
    if metadata.get("product_id") != product_id:
        raise ValueError(
            f"Raw segment product mismatch in {metadata_path}: "
            f"expected {product_id}, found {metadata.get('product_id')!r}"
        )

    request = metadata.get("request_metadata")
    if not isinstance(request, dict):
        raise ValueError(f"Missing raw segment request metadata: {metadata_path}")
    if request.get("sealed") is not True:
        raise ValueError(f"Raw segment metadata is not sealed: {metadata_path}")

    schema_version = request.get("raw_segment_schema_version")
    if schema_version != COINBASE_WS_RAW_SEGMENT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported raw segment schema version {schema_version!r} "
            f"in {metadata_path}"
        )
    session_id = request.get("session_id")
    segment_index = request.get("segment_index")
    frame_count = request.get("frame_count")
    frames = request.get("frames")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError(f"Invalid session_id in {metadata_path}")
    if not _is_nonnegative_int(segment_index):
        raise ValueError(f"Invalid segment_index in {metadata_path}")
    if not _is_positive_int(frame_count):
        raise ValueError(f"Invalid frame_count in {metadata_path}")
    if not isinstance(frames, list) or len(frames) != frame_count:
        raise ValueError(f"Frame provenance count mismatch in {metadata_path}")

    digest = metadata.get("sha256")
    byte_count = metadata.get("byte_count")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"Invalid raw segment sha256 in {metadata_path}")
    if not _is_nonnegative_int(byte_count):
        raise ValueError(f"Invalid raw segment byte_count in {metadata_path}")

    # Metadata sidecars are content-addressed siblings of their compressed raw
    # payload; no metadata-provided path is trusted to redirect source reads.
    artifact_stem = metadata_path.name.removesuffix(".meta.json")
    if artifact_stem != digest:
        raise ValueError(f"Raw segment filename/hash mismatch in {metadata_path}")
    raw_path = metadata_path.with_name(f"{digest}.json.gz")
    if not raw_path.is_file():
        raise ValueError(f"Sealed raw segment artifact is missing: {raw_path}")

    first_ingest = _utc_datetime(
        request.get("first_ingest_time_utc"), label="first_ingest_time_utc", path=metadata_path
    )
    last_ingest = _utc_datetime(
        request.get("last_ingest_time_utc"), label="last_ingest_time_utc", path=metadata_path
    )
    metadata_digest = hashlib.sha256(metadata_bytes).hexdigest()
    provenance_payload = {
        "session_id": session_id,
        "segment_index": segment_index,
        "raw_segment_schema_version": schema_version,
        "frame_count": frame_count,
        "frames": frames,
    }
    provenance_digest = hashlib.sha256(
        json.dumps(provenance_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return DiscoveredRawSegment(
        source=source,
        product_id=product_id,
        session_id=session_id,
        segment_index=segment_index,
        path=raw_path,
        metadata_path=metadata_path,
        sha256=digest,
        byte_count=byte_count,
        frame_count=frame_count,
        first_ingest_time_utc=first_ingest,
        last_ingest_time_utc=last_ingest,
        raw_segment_schema_version=schema_version,
        metadata_sha256=metadata_digest,
        provenance_sha256=provenance_digest,
    )


def discover_sealed_segments(
    input_root: str | Path,
    *,
    product_id: str = "BTC-USD",
    source: str = FORWARD_SOURCE,
) -> DiscoveredSegmentSet:
    """Snapshot sealed metadata names once; active partial files are never opened."""
    segment_dir = (
        Path(input_root)
        / "raw"
        / source
        / DATA_KIND_WEBSOCKET_SEGMENTS
        / product_id
    )
    if not segment_dir.is_dir():
        raise ValueError(f"No sealed raw segment directory exists: {segment_dir}")

    metadata_paths = tuple(sorted(segment_dir.glob("*.meta.json")))
    partial_count = sum(
        1
        for candidate in segment_dir.iterdir()
        if candidate.name.endswith(".partial") and candidate.is_file()
    )
    discovered = tuple(
        sorted(
            (
                _segment_from_metadata(path, source=source, product_id=product_id)
                for path in metadata_paths
            ),
            key=lambda segment: (segment.session_id, segment.segment_index),
        )
    )
    if not discovered:
        raise ValueError(f"No sealed raw segments found for {source} {product_id} in {segment_dir}")
    return DiscoveredSegmentSet(
        segments=discovered,
        active_partial_count_ignored=partial_count,
    )


def _validate_metadata_snapshot(segment: DiscoveredRawSegment) -> list[dict[str, Any]]:
    try:
        raw = segment.metadata_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Cannot reread source metadata: {segment.metadata_path}") from exc
    if hashlib.sha256(raw).hexdigest() != segment.metadata_sha256:
        raise ValueError(f"Source metadata changed after discovery: {segment.metadata_path}")
    try:
        metadata = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid source metadata: {segment.metadata_path}") from exc
    request = metadata.get("request_metadata") if isinstance(metadata, dict) else None
    if not isinstance(request, dict) or not isinstance(request.get("frames"), list):
        raise ValueError(f"Missing frame provenance: {segment.metadata_path}")
    if (
        metadata.get("source") != segment.source
        or metadata.get("product_id") != segment.product_id
        or request.get("session_id") != segment.session_id
        or request.get("segment_index") != segment.segment_index
        or request.get("frame_count") != segment.frame_count
        or request.get("raw_segment_schema_version") != segment.raw_segment_schema_version
        or metadata.get("sha256") != segment.sha256
        or metadata.get("byte_count") != segment.byte_count
        or request.get("sealed") is not True
    ):
        raise ValueError(f"Source metadata no longer matches discovery: {segment.metadata_path}")
    frames = request["frames"]
    if len(frames) != segment.frame_count:
        raise ValueError(f"Frame provenance count mismatch in {segment.metadata_path}")
    return frames


def _iter_framed_bytes(
    segment: DiscoveredRawSegment,
) -> Iterator[tuple[bytes, int]]:
    digest = hashlib.sha256()
    byte_count = 0
    frame_count = 0
    offset = 0
    try:
        with gzip.open(segment.path, "rb") as handle:
            while True:
                header = handle.read(8)
                if not header:
                    break
                if len(header) != 8:
                    raise ValueError(
                        f"Corrupt sealed segment framing at offset {offset}: {segment.path}"
                    )
                digest.update(header)
                byte_count += len(header)
                offset += len(header)
                frame_length = int.from_bytes(header, "big", signed=False)
                frame = handle.read(frame_length)
                if len(frame) != frame_length:
                    raise ValueError(
                        f"Corrupt sealed segment framing at offset {offset}: {segment.path}"
                    )
                digest.update(frame)
                byte_count += len(frame)
                offset += len(frame)
                frame_count += 1
                yield frame, frame_count - 1
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        raise ValueError(f"Cannot read sealed raw segment: {segment.path}") from exc
    if frame_count != segment.frame_count:
        raise ValueError(
            f"Raw segment frame count mismatch in {segment.path}: "
            f"expected {segment.frame_count}, read {frame_count}"
        )
    if byte_count != segment.byte_count:
        raise ValueError(
            f"Raw segment byte_count mismatch in {segment.path}: "
            f"expected {segment.byte_count}, read {byte_count}"
        )
    if digest.hexdigest() != segment.sha256:
        raise ValueError(f"Raw segment digest mismatch: {segment.path}")


def _iter_segment_frames(
    segment: DiscoveredRawSegment,
    *,
    expected_frame_index: int,
) -> Iterator[tuple[bytes, str | None, int, datetime]]:
    provenance = _validate_metadata_snapshot(segment)
    expected_first: datetime | None = None
    expected_last: datetime | None = None
    record_count = 0
    for (raw_bytes, _local_index), frame in zip(
        _iter_framed_bytes(segment), provenance, strict=True
    ):
        if not isinstance(frame, dict):
            raise ValueError(f"Invalid frame provenance record in {segment.metadata_path}")
        if "connection_id" not in frame:
            raise ValueError(f"Missing connection_id in {segment.metadata_path}")
        frame_index = frame.get("frame_index")
        if not _is_nonnegative_int(frame_index) or frame_index != expected_frame_index:
            raise ValueError(
                f"Noncontiguous frame_index in {segment.metadata_path}: "
                f"expected {expected_frame_index}, found {frame_index!r}"
            )
        connection_id = frame.get("connection_id")
        if connection_id is not None and not isinstance(connection_id, str):
            raise ValueError(f"Invalid connection_id in {segment.metadata_path}")
        ingest_time = _utc_datetime(
            frame.get("ingest_time_utc"),
            label="frame ingest_time_utc",
            path=segment.metadata_path,
        )
        frame_hash = frame.get("raw_frame_sha256")
        if (
            not isinstance(frame_hash, str)
            or len(frame_hash) != 64
            or any(character not in "0123456789abcdef" for character in frame_hash)
        ):
            raise ValueError(f"Invalid raw_frame_sha256 in {segment.metadata_path}")
        if hashlib.sha256(raw_bytes).hexdigest() != frame_hash:
            raise ValueError(
                f"Raw frame digest mismatch in {segment.metadata_path} at frame {frame_index}"
            )
        expected_first = ingest_time if expected_first is None else expected_first
        expected_last = ingest_time
        record_count += 1
        yield raw_bytes, connection_id, frame_index, ingest_time
        expected_frame_index += 1
    if record_count != segment.frame_count:
        raise ValueError(f"Frame provenance count mismatch in {segment.metadata_path}")
    if expected_first != segment.first_ingest_time_utc or expected_last != segment.last_ingest_time_utc:
        raise ValueError(f"Segment ingest-time bounds mismatch in {segment.metadata_path}")


def _group_and_validate_segments(
    segments: Sequence[DiscoveredRawSegment],
) -> dict[str, tuple[DiscoveredRawSegment, ...]]:
    if not segments:
        raise ValueError("No sealed raw segments were discovered")
    grouped: dict[str, list[DiscoveredRawSegment]] = defaultdict(list)
    for segment in segments:
        if segment.source != FORWARD_SOURCE:
            raise ValueError(f"Unsupported source in discovered segment: {segment.source}")
        if segment.raw_segment_schema_version != COINBASE_WS_RAW_SEGMENT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported raw segment schema version: {segment.raw_segment_schema_version}"
            )
        grouped[segment.session_id].append(segment)
    result: dict[str, tuple[DiscoveredRawSegment, ...]] = {}
    for session_id, session_segments in grouped.items():
        ordered = tuple(sorted(session_segments, key=lambda segment: segment.segment_index))
        indexes = [segment.segment_index for segment in ordered]
        if indexes != list(range(len(indexes))):
            raise ValueError(
                f"Noncontiguous sealed segment indexes for session {session_id}: {indexes}"
            )
        if any(segment.session_id != session_id for segment in ordered):
            raise ValueError(f"Session ID mismatch in segment group {session_id}")
        result[session_id] = ordered
    return result


def _observation_row(observation: QuarterHourObservation) -> dict[str, Any]:
    return {
        "source": observation.source,
        "product_id": observation.product_id,
        "boundary_time_utc": observation.boundary_time_utc,
        "session_id": observation.session_id,
        "connection_id": observation.connection_id,
        "source_state_time_utc": observation.source_state_time_utc,
        "derived_at_utc": observation.derived_at_utc,
        "source_sequence_num": observation.source_sequence_num,
        "best_bid": observation.best_bid,
        "best_bid_size": observation.best_bid_size,
        "best_ask": observation.best_ask,
        "best_ask_size": observation.best_ask_size,
        "spread": observation.spread,
        "midpoint": observation.midpoint,
        "book_synced": observation.book_synced,
        "canonical_target_eligible": observation.canonical_target_eligible,
        "eligibility_reason": observation.eligibility_reason,
        "boundary_schema_version": observation.boundary_schema_version,
    }


def _boundary_frame(observations: Iterable[QuarterHourObservation]) -> pd.DataFrame:
    columns = [
        "source", "product_id", "boundary_time_utc", "session_id", "connection_id",
        "source_state_time_utc", "derived_at_utc", "source_sequence_num", "best_bid",
        "best_bid_size", "best_ask", "best_ask_size", "spread", "midpoint", "book_synced",
        "canonical_target_eligible", "eligibility_reason", "boundary_schema_version",
    ]
    frame = pd.DataFrame([_observation_row(item) for item in observations], columns=columns)
    for column in ("boundary_time_utc", "source_state_time_utc", "derived_at_utc"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def _iso(value: datetime | pd.Timestamp | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        raise ValueError("Research manifest timestamps must be timezone-aware")
    return parsed.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(_filesystem_path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _filesystem_path(path: Path) -> str:
    absolute = os.path.abspath(os.fspath(path))
    if os.name != "nt" or absolute.startswith("\\\\?\\"):
        return absolute
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute[2:]
    return "\\\\?\\" + absolute


def _write_content_addressed_parquet(
    frame: pd.DataFrame,
    *,
    dataset_dir: Path,
    artifact_prefix: str,
) -> tuple[Path, dict[str, Any]]:
    with NamedTemporaryFile(dir=dataset_dir, prefix=f".{artifact_prefix}.", suffix=".tmp", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        frame.to_parquet(temporary_path, index=False, engine="pyarrow")
        with temporary_path.open("rb+") as handle:
            os.fsync(handle.fileno())
        digest = _sha256_file(temporary_path)
        target = dataset_dir / f"{artifact_prefix}-{digest}.parquet"
        try:
            if os.name == "nt":
                os.rename(_filesystem_path(temporary_path), _filesystem_path(target))
            else:
                os.link(temporary_path, target)
        except FileExistsError:
            if _sha256_file(target) != digest:
                raise ValueError(f"Immutable research artifact conflict: {target}")
        record = {
            "data_kind": (
                DATA_KIND_RESEARCH_BOUNDARIES
                if artifact_prefix == DATA_KIND_RESEARCH_BOUNDARIES
                else DATA_KIND_RESEARCH_TARGETS
            ),
            "path": target.name,
            "sha256": digest,
            "row_count": len(frame),
            "schema_version": (
                RESEARCH_BOUNDARY_SCHEMA_VERSION
                if artifact_prefix == DATA_KIND_RESEARCH_BOUNDARIES
                else RESEARCH_TARGET_SCHEMA_VERSION
            ),
            "first_timestamp": _iso(frame.iloc[0]["boundary_time_utc"])
            if artifact_prefix == DATA_KIND_RESEARCH_BOUNDARIES and len(frame)
            else _iso(frame.iloc[0]["timestamp"])
            if len(frame)
            else None,
            "last_timestamp": _iso(frame.iloc[-1]["boundary_time_utc"])
            if artifact_prefix == DATA_KIND_RESEARCH_BOUNDARIES and len(frame)
            else _iso(frame.iloc[-1]["timestamp"])
            if len(frame)
            else None,
        }
        result_path = Path(_filesystem_path(target)) if os.name == "nt" else target
        return result_path, record
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_immutable_manifest(path: Path, manifest: dict[str, Any]) -> None:
    payload = json.dumps(manifest, indent=2, sort_keys=True, separators=(",", ": ")).encode("utf-8")
    with NamedTemporaryFile(dir=path.parent, prefix=".manifest.", suffix=".tmp", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
    try:
        try:
            if os.name == "nt":
                os.rename(_filesystem_path(temporary_path), _filesystem_path(path))
            else:
                os.link(temporary_path, path)
        except FileExistsError:
            with open(_filesystem_path(path), "rb") as existing:
                existing_payload = existing.read()
            if existing_payload != payload:
                raise ValueError(f"Existing research manifest conflicts with rebuilt dataset: {path}")
    finally:
        temporary_path.unlink(missing_ok=True)


def _check_separate_output_root(input_root: Path, output_root: Path) -> None:
    input_resolved = input_root.resolve()
    output_resolved = output_root.resolve()
    if output_resolved == input_resolved or input_resolved in output_resolved.parents:
        raise ValueError("output_root must be separate from and outside input_root")


def _dataset_identity(
    *,
    source: str,
    product_id: str,
    segments: Sequence[DiscoveredRawSegment],
    heartbeat_timeout_seconds: float,
    git_sha: str,
) -> tuple[str, dict[str, Any]]:
    identity = {
        "dataset_schema_version": RESEARCH_DATASET_SCHEMA_VERSION,
        "research_manifest_schema_version": RESEARCH_MANIFEST_SCHEMA_VERSION,
        "source": source,
        "product_id": product_id,
        "ordered_source_segments": [
            {
                "session_id": segment.session_id,
                "segment_index": segment.segment_index,
                "sha256": segment.sha256,
                "raw_segment_schema_version": segment.raw_segment_schema_version,
                "provenance_sha256": segment.provenance_sha256,
            }
            for segment in segments
        ],
        "boundary_schema_version": RESEARCH_BOUNDARY_SCHEMA_VERSION,
        "target_schema_version": RESEARCH_TARGET_SCHEMA_VERSION,
        "heartbeat_timeout_seconds": heartbeat_timeout_seconds,
        "horizon_minutes": 15,
        "price_definition": "best_bid_ask_midpoint",
        "flat_move_policy": "unlabeled",
        "input_mode": RESEARCH_INPUT_MODE,
        "software_git_sha": git_sha,
    }
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest(), identity


def _build_manifest(
    *,
    dataset_id: str,
    source: str,
    product_id: str,
    segments: Sequence[DiscoveredRawSegment],
    observations: pd.DataFrame,
    targets: pd.DataFrame,
    active_partial_count_ignored: int,
    heartbeat_timeout_seconds: float,
    git_sha: str,
    boundary_artifact: dict[str, Any],
    target_artifact: dict[str, Any],
) -> dict[str, Any]:
    session_groups: dict[str, list[DiscoveredRawSegment]] = defaultdict(list)
    for segment in segments:
        session_groups[segment.session_id].append(segment)
    source_sessions = []
    for session_id, session_segments in sorted(session_groups.items()):
        ordered = sorted(session_segments, key=lambda item: item.segment_index)
        source_sessions.append(
            {
                "session_id": session_id,
                "first_segment_index": ordered[0].segment_index,
                "last_segment_index": ordered[-1].segment_index,
                "segment_count": len(ordered),
                "frame_count": sum(item.frame_count for item in ordered),
                "first_ingest_time_utc": _iso(ordered[0].first_ingest_time_utc),
                "last_ingest_time_utc": _iso(ordered[-1].last_ingest_time_utc),
            }
        )
    all_ingest_times = [
        timestamp
        for segment in segments
        for timestamp in (segment.first_ingest_time_utc, segment.last_ingest_time_utc)
    ]
    boundary_reason_counts = {
        str(key): int(value)
        for key, value in sorted(observations["eligibility_reason"].value_counts().items())
    }
    target_reason_counts = {
        str(key): int(value)
        for key, value in sorted(targets["target_eligibility_reason"].value_counts().items())
    }
    eligible_boundaries = int(observations["canonical_target_eligible"].sum())
    ineligible_boundaries = len(observations) - eligible_boundaries
    eligible_targets = targets.loc[targets["target_eligible"]]
    up_count = int((eligible_targets["target"] == 1).sum())
    down_count = int((eligible_targets["target"] == 0).sum())
    flat_count = int(targets["target_eligibility_reason"].eq("flat_move").sum())
    if eligible_targets.shape[0] != up_count + down_count:
        raise ValueError("Target count consistency failure")
    if len(targets) != sum(target_reason_counts.values()):
        raise ValueError("Target candidate count consistency failure")
    if len(observations) != eligible_boundaries + ineligible_boundaries:
        raise ValueError("Boundary count consistency failure")

    try:
        package_version = version("btc-quarter-hour-engine")
    except PackageNotFoundError:
        package_version = "0.1.0"

    return {
        "research_manifest_schema_version": RESEARCH_MANIFEST_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "input_mode": RESEARCH_INPUT_MODE,
        "source": source,
        "product_id": product_id,
        "heartbeat_timeout_seconds": heartbeat_timeout_seconds,
        "canonical_horizon_minutes": 15,
        "price_definition": "best_bid_ask_midpoint",
        "flat_move_policy": "unlabeled",
        "software": {"package_version": package_version, "git_sha": git_sha},
        "source_summary": {
            "session_count": len(source_sessions),
            "sealed_segment_count": len(segments),
            "frame_count": sum(segment.frame_count for segment in segments),
            "first_ingest_time_utc": _iso(min(all_ingest_times)),
            "last_ingest_time_utc": _iso(max(all_ingest_times)),
            "active_partial_count_ignored": active_partial_count_ignored,
        },
        "source_sessions": source_sessions,
        "source_segments": [
            {
                "session_id": segment.session_id,
                "segment_index": segment.segment_index,
                "sha256": segment.sha256,
                "frame_count": segment.frame_count,
                "first_ingest_time_utc": _iso(segment.first_ingest_time_utc),
                "last_ingest_time_utc": _iso(segment.last_ingest_time_utc),
                "path": str(segment.path),
                "metadata_path": str(segment.metadata_path),
                "byte_count": segment.byte_count,
                "raw_segment_schema_version": segment.raw_segment_schema_version,
                "provenance_sha256": segment.provenance_sha256,
                "metadata_sha256": segment.metadata_sha256,
            }
            for segment in segments
        ],
        "boundary_summary": {
            "total": len(observations),
            "eligible": eligible_boundaries,
            "ineligible": ineligible_boundaries,
            "counts_by_eligibility_reason": boundary_reason_counts,
        },
        "target_summary": {
            "candidate_rows": len(targets),
            "eligible_labels": len(eligible_targets),
            "up_labels": up_count,
            "down_labels": down_count,
            "flat_moves": flat_count,
            "latest_labeled_timestamp_utc": _iso(eligible_targets["timestamp"].max())
            if not eligible_targets.empty
            else None,
            "counts_by_target_eligibility_reason": target_reason_counts,
        },
        "artifacts": [boundary_artifact, target_artifact],
        "snapshot_semantics": {
            "input_is_complete_session": False,
            "input_is_sealed_prefix": True,
            "active_partial_files_ignored": True,
            "future_segments_may_supersede_snapshot": True,
            "canonical_relative_to_included_prefix": True,
            "description": (
                "Results are canonical with respect to the included sealed raw prefix. "
                "Active and future raw frames are excluded; a later extraction with more "
                "sealed segments may supersede this immutable, reproducible snapshot."
            ),
        },
    }


def build_research_dataset(
    *,
    input_root: str | Path,
    output_root: str | Path,
    discovered_segments: DiscoveredSegmentSet | Sequence[DiscoveredRawSegment],
    product_id: str = "BTC-USD",
    source: str = FORWARD_SOURCE,
    websocket_config: CoinbaseWebSocketConfig | None = None,
    active_partial_count_ignored: int = 0,
    git_sha: str | None = None,
) -> ResearchDatasetResult:
    """Build an immutable dataset from only the previously discovered segments."""
    input_path = Path(input_root)
    output_path = Path(output_root)
    if not input_path.is_dir():
        raise ValueError(f"input_root must be an existing directory: {input_path}")
    _check_separate_output_root(input_path, output_path)

    if isinstance(discovered_segments, DiscoveredSegmentSet):
        segments = discovered_segments.segments
        active_partial_count_ignored = discovered_segments.active_partial_count_ignored
    else:
        segments = tuple(discovered_segments)
    if not segments:
        raise ValueError("No sealed raw segments were discovered")
    if any(segment.product_id != product_id for segment in segments):
        raise ValueError(f"Discovered segment product does not match requested product {product_id}")
    grouped = _group_and_validate_segments(segments)
    config = websocket_config or CoinbaseWebSocketConfig(product_id=product_id)
    if config.product_id != product_id:
        raise ValueError("WebSocket configuration product_id does not match requested product")
    timeout = config.heartbeat_timeout_seconds
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("Canonical heartbeat timeout must be finite and nonnegative")
    resolved_git_sha = git_sha or _git_sha() or "unknown"

    all_observations: list[QuarterHourObservation] = []
    session_summaries: dict[str, tuple[datetime, datetime]] = {}
    total_frames = 0
    for session_id, session_segments in sorted(grouped.items()):
        accumulator = CanonicalReplayAccumulator(
            product_id=product_id,
            session_id=session_id,
            heartbeat_timeout_seconds=timeout,
        )
        expected_frame_index = 0
        first_ingest_time: datetime | None = None
        last_ingest_time: datetime | None = None
        for segment in session_segments:
            for raw_bytes, connection_id, frame_index, ingest_time in _iter_segment_frames(
                segment,
                expected_frame_index=expected_frame_index,
            ):
                accumulator.consume(raw_bytes, connection_id, frame_index, ingest_time)
                expected_frame_index += 1
                first_ingest_time = first_ingest_time or ingest_time
                last_ingest_time = ingest_time
                total_frames += 1
        if first_ingest_time is None or last_ingest_time is None:
            raise ValueError(f"Session {session_id} has no frames in its sealed prefix")
        session_summaries[session_id] = (first_ingest_time, last_ingest_time)
        all_observations.extend(
            accumulator.finalize(
                session_started_at_utc=first_ingest_time,
                session_completed_at_utc=last_ingest_time,
                derived_at_utc=last_ingest_time,
            )
        )

    boundary_frame = _boundary_frame(all_observations)
    if not boundary_frame.empty:
        boundary_frame = boundary_frame.sort_values(
            ["boundary_time_utc", "session_id"], kind="stable"
        ).reset_index(drop=True)
        duplicated = boundary_frame["boundary_time_utc"].duplicated(keep=False)
        if duplicated.any():
            evidence = boundary_frame.loc[
                duplicated,
                ["boundary_time_utc", "session_id", "eligibility_reason"],
            ].to_dict(orient="records")
            raise ValueError(f"Duplicate global boundary timestamps: {evidence}")
    targets = build_canonical_direction_targets(boundary_frame)

    ordered_segments = tuple(
        sorted(segments, key=lambda segment: (segment.session_id, segment.segment_index))
    )
    dataset_id, _identity = _dataset_identity(
        source=source,
        product_id=product_id,
        segments=ordered_segments,
        heartbeat_timeout_seconds=timeout,
        git_sha=resolved_git_sha,
    )

    dataset_dir = output_path / "datasets" / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    boundary_path, boundary_artifact_record = _write_content_addressed_parquet(
        boundary_frame,
        dataset_dir=dataset_dir,
        artifact_prefix=DATA_KIND_RESEARCH_BOUNDARIES,
    )
    target_path, target_artifact_record = _write_content_addressed_parquet(
        targets,
        dataset_dir=dataset_dir,
        artifact_prefix=DATA_KIND_RESEARCH_TARGETS,
    )
    manifest = _build_manifest(
        dataset_id=dataset_id,
        source=source,
        product_id=product_id,
        segments=ordered_segments,
        observations=boundary_frame,
        targets=targets,
        active_partial_count_ignored=active_partial_count_ignored,
        heartbeat_timeout_seconds=timeout,
        git_sha=resolved_git_sha,
        boundary_artifact=boundary_artifact_record,
        target_artifact=target_artifact_record,
    )
    manifest_path = dataset_dir / "manifest.json"
    _write_immutable_manifest(manifest_path, manifest)
    return ResearchDatasetResult(
        dataset_id=dataset_id,
        dataset_dir=dataset_dir,
        manifest=manifest,
        manifest_path=manifest_path,
        boundary_artifact=boundary_path,
        target_artifact=target_path,
        boundary_count=len(boundary_frame),
        eligible_boundary_count=int(boundary_frame["canonical_target_eligible"].sum()),
        target_candidate_count=len(targets),
        eligible_target_count=int(targets["target_eligible"].sum()),
        sealed_segment_count=len(ordered_segments),
        raw_frame_count=total_frames,
    )


__all__ = [
    "DiscoveredRawSegment",
    "DiscoveredSegmentSet",
    "ResearchDatasetResult",
    "build_research_dataset",
    "discover_sealed_segments",
]
