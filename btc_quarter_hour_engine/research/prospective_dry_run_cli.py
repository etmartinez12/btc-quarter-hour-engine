from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from btc_quarter_hour_engine.research.prospective_inference import (
    infer_causal_prediction,
    load_official_models,
)
from btc_quarter_hour_engine.research.prospective_source import (
    RawFrameRecord,
    RawFrameSnapshot,
    build_causal_feature_vector,
)


def _parse_utc(value: str) -> datetime:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("Timestamps must include an explicit UTC offset")
    return timestamp


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_snapshot(path: Path) -> RawFrameSnapshot:
    document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    expected_snapshot_keys = {
        "session_id",
        "session_started_at_utc",
        "watermark_utc",
        "clock_health_verified",
        "frames",
    }
    if not isinstance(document, dict) or set(document) != expected_snapshot_keys:
        raise ValueError("Source snapshot has unknown or missing fields")
    if not isinstance(document["frames"], list):
        raise ValueError("Source snapshot frames must be a JSON array")
    frames = []
    expected_frame_keys = {
        "raw_hex",
        "session_id",
        "connection_id",
        "frame_index",
        "ingest_time_utc",
        "raw_frame_sha256",
    }
    for item in document["frames"]:
        if not isinstance(item, dict) or set(item) != expected_frame_keys:
            raise ValueError("Raw frame has unknown or missing provenance fields")
        try:
            raw_bytes = bytes.fromhex(item["raw_hex"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Raw frame raw_hex must contain valid hexadecimal bytes") from exc
        frames.append(
            RawFrameRecord(
                raw_bytes=raw_bytes,
                session_id=item["session_id"],
                connection_id=item["connection_id"],
                frame_index=item["frame_index"],
                ingest_time_utc=_parse_utc(item["ingest_time_utc"]),
                raw_frame_sha256=item["raw_frame_sha256"],
            )
        )
    return RawFrameSnapshot(
        session_id=document["session_id"],
        session_started_at_utc=_parse_utc(document["session_started_at_utc"]),
        watermark_utc=_parse_utc(document["watermark_utc"]),
        clock_health_verified=document["clock_health_verified"],
        frames=tuple(frames),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline-only prospective inference dry run; never persists a prediction."
    )
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--boundary", required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--implementation-git-sha", required=True)
    parser.add_argument("--generated-at", required=True)
    arguments = parser.parse_args(argv)

    snapshot = _load_snapshot(arguments.source_snapshot)
    features = build_causal_feature_vector(
        snapshot, _parse_utc(arguments.boundary)
    )
    models = load_official_models(arguments.model_directory)
    evidence = infer_causal_prediction(
        features,
        models,
        implementation_git_sha=arguments.implementation_git_sha,
        generated_at_utc=_parse_utc(arguments.generated_at),
    )
    evidence["boundary_utc"] = evidence["boundary_utc"].isoformat().replace("+00:00", "Z")
    evidence["target_utc"] = evidence["target_utc"].isoformat().replace("+00:00", "Z")
    evidence["source_watermark_utc"] = (
        evidence["source_watermark_utc"].isoformat().replace("+00:00", "Z")
    )
    evidence["generated_at_utc"] = (
        evidence["generated_at_utc"].isoformat().replace("+00:00", "Z")
    )
    evidence["dry_run"] = True
    evidence["persisted_at_utc"] = None
    evidence["timing_eligible"] = False
    print(json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
