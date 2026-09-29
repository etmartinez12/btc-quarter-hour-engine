from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

COINBASE_CANDLE_SCHEMA_VERSION = "1"
COINBASE_BOOK_SNAPSHOT_SCHEMA_VERSION = "1"
DATASET_MANIFEST_SCHEMA_VERSION = "1"


def _utc_isotime(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.isoformat()
    return value.astimezone().isoformat()


def _git_sha() -> str | None:
    env_sha = os.environ.get("BTC_QH_GIT_SHA")
    if env_sha:
        return env_sha
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip() or None
    except Exception:
        return None


def build_dataset_id(
    *,
    source: str,
    product_id: str,
    data_kind: str,
    requested_start: str | datetime | None,
    requested_end: str | datetime | None,
    granularity: str | None,
    raw_artifact_hashes: list[str],
    manifest_schema_version: str = DATASET_MANIFEST_SCHEMA_VERSION,
) -> str:
    canonical = {
        "manifest_schema_version": manifest_schema_version,
        "source": source,
        "product_id": product_id,
        "data_kind": data_kind,
        "requested_start": _utc_isotime(requested_start) if isinstance(requested_start, datetime) else requested_start,
        "requested_end": _utc_isotime(requested_end) if isinstance(requested_end, datetime) else requested_end,
        "granularity": granularity,
        "raw_artifact_hashes": sorted(raw_artifact_hashes),
    }
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def build_manifest(
    *,
    source: str,
    product_id: str,
    data_kind: str,
    requested_start: str | datetime | None,
    requested_end: str | datetime | None,
    granularity: str | None,
    raw_artifacts: list[dict[str, Any]],
    normalized_artifacts: list[dict[str, Any]],
    coverage: dict[str, Any],
    canonical_target_eligible: bool,
    canonical_target_ineligibility_reason: str | None,
    acquisition_started_at_utc: datetime | None = None,
    acquisition_completed_at_utc: datetime | None = None,
    request_count: int = 0,
    schema_version: str = COINBASE_CANDLE_SCHEMA_VERSION,
    dataset_id: str | None = None,
) -> dict[str, Any]:
    raw_hashes = [artifact.get("sha256") for artifact in raw_artifacts if artifact.get("sha256")]
    resolved_id = dataset_id or build_dataset_id(
        source=source,
        product_id=product_id,
        data_kind=data_kind,
        requested_start=requested_start,
        requested_end=requested_end,
        granularity=granularity,
        raw_artifact_hashes=raw_hashes,
    )
    try:
        package_version = version("btc-quarter-hour-engine")
    except PackageNotFoundError:
        package_version = "unknown"

    manifest = {
        "manifest_schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
        "dataset_id": resolved_id,
        "source": source,
        "source_api_family": "coinbase_advanced_trade",
        "data_kind": data_kind,
        "product_id": product_id,
        "granularity": granularity,
        "requested_start": _utc_isotime(requested_start) if isinstance(requested_start, datetime) else requested_start,
        "requested_end": _utc_isotime(requested_end) if isinstance(requested_end, datetime) else requested_end,
        "acquisition_started_at_utc": _utc_isotime(acquisition_started_at_utc),
        "acquisition_completed_at_utc": _utc_isotime(acquisition_completed_at_utc),
        "request_count": request_count,
        "raw_artifacts": raw_artifacts,
        "normalized_artifacts": normalized_artifacts,
        "coverage": coverage,
        "canonical_target_eligible": canonical_target_eligible,
        "canonical_target_ineligibility_reason": canonical_target_ineligibility_reason,
        "schema_version": schema_version,
        "software": {
            "package_version": package_version,
            "git_sha": _git_sha(),
        },
    }
    return manifest


def write_manifest(manifest: dict[str, Any], output_root: str | Path) -> Path:
    root = Path(output_root)
    manifest_dir = root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    dataset_id = str(manifest.get("dataset_id", "unknown"))
    target_path = manifest_dir / f"{dataset_id}.json"
    target_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return target_path
