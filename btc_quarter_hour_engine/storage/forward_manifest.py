from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import uuid4

from btc_quarter_hour_engine.storage.forward_schema import (
    COINBASE_BBO_STATE_SCHEMA_VERSION,
    COINBASE_BOUNDARY_BBO_SCHEMA_VERSION,
    COINBASE_L2_UPDATE_SCHEMA_VERSION,
    COINBASE_WS_RAW_SEGMENT_SCHEMA_VERSION,
    COINBASE_WS_SESSION_MANIFEST_SCHEMA_VERSION,
    DATA_KIND_BBO_STATE,
    DATA_KIND_LEVEL2_UPDATES,
    DATA_KIND_QUARTER_HOUR_BBO,
    FORWARD_SOURCE,
)
from btc_quarter_hour_engine.storage.manifest import _git_sha


def _utc_isotime(value: datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        value = parsed
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Forward manifest timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _schema_versions() -> dict[str, str]:
    return {
        "raw_segment": COINBASE_WS_RAW_SEGMENT_SCHEMA_VERSION,
        "level2_updates": COINBASE_L2_UPDATE_SCHEMA_VERSION,
        "bbo_state": COINBASE_BBO_STATE_SCHEMA_VERSION,
        "quarter_hour_bbo": COINBASE_BOUNDARY_BBO_SCHEMA_VERSION,
        "forward_manifest": COINBASE_WS_SESSION_MANIFEST_SCHEMA_VERSION,
    }


def build_forward_dataset_id(
    *,
    source: str,
    product_id: str,
    raw_segments: list[dict[str, Any]],
    schema_versions: dict[str, str] | None = None,
) -> str:
    """Return a stable ID for forward content, independent of session timing."""
    raw_hashes = []
    for segment in raw_segments:
        digest = segment.get("sha256")
        if not isinstance(digest, str) or not digest:
            raise ValueError("Every sealed raw segment requires a sha256")
        raw_hashes.append(digest)
    canonical = {
        "source": source,
        "product_id": product_id,
        "ordered_raw_segment_hashes": raw_hashes,
        "schema_versions": schema_versions or _schema_versions(),
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _raw_segment_summaries(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "path",
        "metadata_path",
        "sha256",
        "byte_count",
        "session_id",
        "connection_id",
        "segment_index",
        "raw_segment_schema_version",
        "frame_count",
        "first_sequence_num",
        "last_sequence_num",
        "first_ingest_time_utc",
        "last_ingest_time_utc",
        "first_received_at_utc",
        "last_received_at_utc",
    )
    return [{key: segment[key] for key in fields if key in segment} for segment in segments]


def _group_normalized_artifacts(
    artifacts: list[dict[str, Any]] | dict[str, Any] | None,
) -> dict[str, list[dict[str, Any]]]:
    groups = {
        DATA_KIND_QUARTER_HOUR_BBO: [],
        DATA_KIND_LEVEL2_UPDATES: [],
        DATA_KIND_BBO_STATE: [],
    }
    if isinstance(artifacts, dict):
        for kind in groups:
            value = artifacts.get(kind, [])
            if value is None:
                continue
            if not isinstance(value, list):
                raise TypeError(f"normalized_artifacts[{kind!r}] must be a list")
            groups[kind] = value
        return groups
    for artifact in artifacts or []:
        kind = artifact.get("data_kind")
        if kind in groups:
            groups[kind].append(artifact)
    return groups


def build_forward_manifest(
    *,
    source: str = FORWARD_SOURCE,
    product_id: str,
    session_id: str | None = None,
    websocket_url: str | None = None,
    channels: list[str] | None = None,
    session_started_at_utc: datetime | str | None = None,
    session_completed_at_utc: datetime | str | None = None,
    termination_reason: str | None = None,
    connection_count: int = 0,
    reconnect_count: int = 0,
    connections: list[dict[str, Any]] | None = None,
    raw_segments: list[dict[str, Any]] | None = None,
    normalized_artifacts: list[dict[str, Any]] | dict[str, Any] | None = None,
    quarter_hour_summary: dict[str, Any] | None = None,
    integrity: dict[str, Any] | None = None,
    schema_versions: dict[str, str] | None = None,
    software: dict[str, Any] | None = None,
    # Legacy collector arguments remain accepted while acquisition wiring is
    # updated to the session-oriented fields above.
    data_kind: str | None = None,
    raw_artifacts: list[dict[str, Any]] | None = None,
    coverage: dict[str, Any] | None = None,
    canonical_target_eligible: bool | None = None,
    canonical_target_ineligibility_reason: str | None = None,
    acquisition_started_at_utc: datetime | str | None = None,
    acquisition_completed_at_utc: datetime | str | None = None,
    request_count: int | None = None,
    schema_version: str | None = None,
    purpose: str | None = None,
) -> dict[str, Any]:
    segments = _raw_segment_summaries(raw_segments if raw_segments is not None else (raw_artifacts or []))
    resolved_session_id = (
        session_id
        if session_id is not None
        else next(
            (str(segment["session_id"]) for segment in segments if segment.get("session_id")),
            str(uuid4()),
        )
    )
    if not resolved_session_id:
        raise ValueError("session_id must not be empty")
    normalized = (
        _group_normalized_artifacts(normalized_artifacts)
        if isinstance(normalized_artifacts, dict)
        else list(normalized_artifacts or [])
    )
    coverage = coverage or {}
    summary = {
        "boundaries_seen": coverage.get("boundaries_seen", coverage.get("observation_count", 0)),
        "eligible_boundaries": coverage.get(
            "eligible_boundaries", coverage.get("eligible_observation_count", 0)
        ),
        "ineligible_boundaries": coverage.get(
            "ineligible_boundaries",
            max(
                0,
                coverage.get("observation_count", 0)
                - coverage.get("eligible_observation_count", 0),
            ),
        ),
    }
    if quarter_hour_summary:
        summary.update(quarter_hour_summary)
    counter_names = (
        "sequence_gap_count",
        "stale_sequence_count",
        "heartbeat_timeout_count",
        "heartbeat_discontinuity_count",
        "malformed_frame_count",
        "malformed_level2_count",
        "crossed_book_count",
        "book_rebuild_count",
        "book_recovery_count",
    )
    resolved_integrity = {name: 0 for name in counter_names}
    if integrity:
        resolved_integrity.update(integrity)
    versions = _schema_versions()
    if schema_versions:
        versions.update(schema_versions)
    resolved_software = dict(software or {})
    if "package_version" not in resolved_software:
        try:
            resolved_software["package_version"] = version("btc-quarter-hour-engine")
        except PackageNotFoundError:
            resolved_software["package_version"] = "unknown"
    resolved_software.setdefault("git_sha", _git_sha())

    started = session_started_at_utc or acquisition_started_at_utc
    completed = session_completed_at_utc or acquisition_completed_at_utc
    eligible = canonical_target_eligible
    if eligible is None:
        eligible = summary["eligible_boundaries"] > 0
    ineligibility_reason = canonical_target_ineligibility_reason
    resolved_schema_version = schema_version or COINBASE_BOUNDARY_BBO_SCHEMA_VERSION

    manifest = {
        "forward_manifest_schema_version": COINBASE_WS_SESSION_MANIFEST_SCHEMA_VERSION,
        "dataset_id": build_forward_dataset_id(
            source=source,
            product_id=product_id,
            raw_segments=segments,
            schema_versions=versions,
        ),
        "session_id": resolved_session_id,
        "source": source,
        "product_id": product_id,
        "websocket_url": websocket_url,
        "channels": list(channels or []),
        "session_started_at_utc": _utc_isotime(started),
        "session_completed_at_utc": _utc_isotime(completed),
        "termination_reason": termination_reason,
        "connection_count": connection_count,
        "reconnect_count": reconnect_count,
        "connections": list(connections or []),
        "raw_segments": segments,
        "normalized_artifacts": normalized,
        "quarter_hour_summary": summary,
        "integrity": resolved_integrity,
        "canonical_source": {
            "price_definition": "best_bid_ask_midpoint",
            "boundary_schedule": ":00/:15/:30/:45 UTC",
        },
        "software": resolved_software,
        "schema_versions": versions,
        # Compatibility aliases for callers that still consume the old
        # collection result shape; the canonical session fields are above.
        "data_kind": data_kind or DATA_KIND_QUARTER_HOUR_BBO,
        "raw_artifacts": segments,
        "coverage": {
            **coverage,
            "observation_count": summary["boundaries_seen"],
            "eligible_observation_count": summary["eligible_boundaries"],
        },
        "canonical_target_eligible": eligible,
        "canonical_target_ineligibility_reason": ineligibility_reason,
        "acquisition_started_at_utc": _utc_isotime(started),
        "acquisition_completed_at_utc": _utc_isotime(completed),
        "request_count": request_count if request_count is not None else len(segments),
        "schema_version": resolved_schema_version,
        "purpose": purpose,
    }
    return manifest


def write_forward_manifest(manifest: dict[str, Any], output_root: str | Path) -> Path:
    """Atomically persist one immutable session manifest per session ID."""
    session_id = str(manifest.get("session_id", ""))
    if not re.fullmatch(r"[A-Za-z0-9._-]+", session_id):
        raise ValueError("Invalid session_id")
    dataset_id = str(manifest.get("dataset_id", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", dataset_id):
        raise ValueError("Invalid dataset_id")
    manifest_dir = Path(output_root) / "manifests" / "forward_sessions"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    target = manifest_dir / f"{session_id}.json"
    payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    with NamedTemporaryFile(dir=manifest_dir, prefix=f".{session_id}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() != payload:
                raise ValueError(f"Forward manifest conflict for session ID {session_id}")
    finally:
        temporary.unlink(missing_ok=True)
    return target


__all__ = ["build_forward_dataset_id", "build_forward_manifest", "write_forward_manifest"]
