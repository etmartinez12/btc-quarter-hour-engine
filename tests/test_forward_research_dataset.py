from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from btc_quarter_hour_engine.market_data.replay import replay_recorded_frames
from btc_quarter_hour_engine.research.forward_dataset import (
    build_research_dataset,
    discover_sealed_segments,
)
from btc_quarter_hour_engine.storage.websocket_raw import RawSegmentWriter


def _wire_frame(channel: str, timestamp: datetime, sequence: int, *, kind: str = "") -> bytes:
    iso = timestamp.isoformat().replace("+00:00", "Z")
    if channel == "l2_data":
        events = [
            {
                "type": kind,
                "product_id": "BTC-USD",
                "updates": [
                    {
                        "side": side,
                        "event_time": iso,
                        "price_level": str(price),
                        "new_quantity": "1",
                    }
                    for side, price in (
                        ("bid", 99 if kind == "snapshot" else 100),
                        ("offer", 101),
                    )
                ],
            }
        ]
    else:
        events = [{"current_time": iso, "heartbeat_counter": str(sequence)}]
    return json.dumps(
        {
            "channel": channel,
            "timestamp": iso,
            "sequence_num": sequence,
            "events": events,
        },
        separators=(",", ":"),
    ).encode()


def _session_frames(start: datetime) -> list[bytes]:
    return [
        _wire_frame("l2_data", start, 1, kind="snapshot"),
        _wire_frame("heartbeats", start, 2),
        _wire_frame("l2_data", start + timedelta(minutes=15), 3, kind="update"),
        _wire_frame("heartbeats", start + timedelta(minutes=15), 4),
    ]


def _write_session(
    root: Path,
    *,
    session_id: str = "session-1",
    start: datetime | None = None,
    max_frames: int = 100,
) -> tuple[RawSegmentWriter, list[bytes]]:
    start = start or datetime(2026, 1, 1, tzinfo=timezone.utc)
    writer = RawSegmentWriter(
        root,
        product_id="BTC-USD",
        session_id=session_id,
        max_frames=max_frames,
    )
    frames = _session_frames(start)
    if session_id != "session-1":
        whitespace = 1 + ord(session_id[-1]) % 3
        frames = [frame + b" " * whitespace for frame in frames]
    for frame in frames:
        writer.add_frame(
            raw=frame,
            message_type="snapshot" if b'"snapshot"' in frame else (
                "heartbeat" if b'"heartbeats"' in frame else "l2_data"
            ),
            connection_id=f"{session_id}-connection",
            ingest_time_utc=start + timedelta(seconds=0),
        )
    writer.seal()
    return writer, frames


def _build(input_root: Path, output_root: Path, discovered=None, **kwargs):
    selected = discovered or discover_sealed_segments(input_root)
    return build_research_dataset(
        input_root=input_root,
        output_root=output_root,
        discovered_segments=selected,
        git_sha="test-git-sha",
        **kwargs,
    )


def _tree_snapshot(root: Path) -> dict[str, tuple[int, str]]:
    return {
        str(path.relative_to(root)): (
            path.stat().st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def test_build_uses_sealed_prefix_and_leaves_active_partial_unchanged(tmp_path):
    input_root = tmp_path / "production"
    output_root = tmp_path / "research"
    writer, _ = _write_session(input_root)
    active_frame = _wire_frame(
        "heartbeats", datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc), 5
    )
    writer.add_frame(
        raw=active_frame,
        message_type="heartbeat",
        connection_id="session-1-connection",
        ingest_time_utc=datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc),
    )
    partial_path = writer.active_partial_path
    before = _tree_snapshot(input_root)

    result = _build(input_root, output_root)

    assert result.sealed_segment_count == 1
    assert result.raw_frame_count == 4
    assert result.manifest["source_summary"]["active_partial_count_ignored"] == 1
    assert result.boundary_count == 2
    assert result.manifest["input_mode"] == "sealed_prefix_snapshot"
    assert result.manifest["snapshot_semantics"]["input_is_complete_session"] is False
    assert _tree_snapshot(input_root) == before
    assert partial_path.exists()
    boundaries = pd.read_parquet(result.boundary_artifact)
    assert list(boundaries["boundary_time_utc"]) == list(
        pd.to_datetime(
            ["2026-01-01T00:00:00Z", "2026-01-01T00:15:00Z"], utc=True
        )
    )


def test_short_valid_sealed_prefix_is_published_with_no_eligible_labels(tmp_path):
    input_root = tmp_path / "production"
    writer = RawSegmentWriter(
        input_root, product_id="BTC-USD", session_id="short-session"
    )
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for frame in _session_frames(start)[:2]:
        writer.add_frame(
            raw=frame,
            message_type="snapshot" if b'"snapshot"' in frame else "heartbeat",
            connection_id="short-connection",
            ingest_time_utc=start,
        )
    writer.seal()

    result = _build(input_root, tmp_path / "research")

    assert result.boundary_count == 1
    assert result.eligible_boundary_count == 1
    assert result.target_candidate_count == 1
    assert result.eligible_target_count == 0
    assert result.manifest_path.exists()


def test_discovery_snapshot_excludes_later_sealed_segment_and_fresh_snapshot_changes_id(tmp_path):
    input_root = tmp_path / "production"
    output_root = tmp_path / "research"
    writer = RawSegmentWriter(
        input_root, product_id="BTC-USD", session_id="snapshot-session", max_frames=2
    )
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    frames = _session_frames(start)
    for frame in frames[:2]:
        writer.add_frame(
            raw=frame,
            message_type="snapshot" if b'"snapshot"' in frame else "heartbeat",
            connection_id="snapshot-connection",
            ingest_time_utc=start,
        )
    initial_snapshot = discover_sealed_segments(input_root)
    assert len(initial_snapshot.segments) == 1
    for frame in frames[2:]:
        writer.add_frame(
            raw=frame,
            message_type="l2_data" if b'"l2_data"' in frame else "heartbeat",
            connection_id="snapshot-connection",
            ingest_time_utc=start,
        )
    writer.seal()

    first = _build(input_root, output_root, initial_snapshot)
    fresh_snapshot = discover_sealed_segments(input_root)
    second = _build(input_root, output_root, fresh_snapshot)

    assert first.raw_frame_count == 2
    assert second.raw_frame_count == 4
    assert first.dataset_id != second.dataset_id
    assert first.dataset_dir != second.dataset_dir


def test_raw_digest_corruption_fails_without_publishing_manifest(tmp_path):
    input_root = tmp_path / "production"
    output_root = tmp_path / "research"
    _write_session(input_root)
    discovered = discover_sealed_segments(input_root)
    raw_path = discovered.segments[0].path
    raw_path.write_bytes(raw_path.read_bytes() + b"corruption")

    with pytest.raises(ValueError, match="Cannot read sealed raw segment|digest mismatch"):
        _build(input_root, output_root, discovered)
    assert not list(output_root.rglob("manifest.json"))


def test_corrupt_per_frame_digest_is_rejected(tmp_path):
    input_root = tmp_path / "production"
    output_root = tmp_path / "research"
    _write_session(input_root)
    discovered = discover_sealed_segments(input_root)
    segment = discovered.segments[0]
    metadata = json.loads(segment.metadata_path.read_text())
    metadata["request_metadata"]["frames"][0]["raw_frame_sha256"] = "0" * 64
    segment.metadata_path.write_text(json.dumps(metadata))
    # Discovery fixes the metadata snapshot at extraction start.
    discovered = discover_sealed_segments(input_root)

    with pytest.raises(ValueError, match="Raw frame digest mismatch"):
        _build(input_root, output_root, discovered)
    assert not list(output_root.rglob("manifest.json"))


def test_metadata_frame_count_mismatch_is_rejected_at_discovery(tmp_path):
    input_root = tmp_path / "production"
    writer, _ = _write_session(input_root)
    metadata_path = Path(writer.sealed_segments[0]["metadata_path"])
    metadata = json.loads(metadata_path.read_text())
    metadata["request_metadata"]["frame_count"] += 1
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="Frame provenance count mismatch"):
        discover_sealed_segments(input_root)


def test_missing_segment_index_fails_instead_of_skipping_prefix(tmp_path):
    input_root = tmp_path / "production"
    writer = RawSegmentWriter(
        input_root, product_id="BTC-USD", session_id="session-1", max_frames=1
    )
    frames = _session_frames(datetime(2026, 1, 1, tzinfo=timezone.utc))
    for index, frame in enumerate(frames):
        writer.add_frame(
            raw=frame,
            message_type="snapshot" if index == 0 else (
                "l2_data" if index == 2 else "heartbeat"
            ),
            connection_id="session-1-connection",
            ingest_time_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
    second_segment = writer.sealed_segments[1]
    Path(second_segment["path"]).unlink()
    Path(second_segment["metadata_path"]).unlink()

    discovered = discover_sealed_segments(input_root)
    with pytest.raises(ValueError, match="Noncontiguous sealed segment indexes"):
        _build(input_root, tmp_path / "research", discovered)


def test_noncontiguous_frame_index_fails(tmp_path):
    input_root = tmp_path / "production"
    writer, _ = _write_session(input_root)
    metadata_path = Path(writer.sealed_segments[0]["metadata_path"])
    metadata = json.loads(metadata_path.read_text())
    metadata["request_metadata"]["frames"][2]["frame_index"] = 4
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="Noncontiguous frame_index"):
        _build(input_root, tmp_path / "research")


def test_multiple_sessions_keep_provenance_and_boundary_order(tmp_path):
    input_root = tmp_path / "production"
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _write_session(input_root, session_id="later", start=start + timedelta(hours=1))
    _write_session(input_root, session_id="earlier", start=start)

    result = _build(input_root, tmp_path / "research")
    boundaries = pd.read_parquet(result.boundary_artifact)
    assert boundaries["boundary_time_utc"].is_monotonic_increasing
    assert set(boundaries["session_id"]) == {"earlier", "later"}
    assert {item["session_id"] for item in result.manifest["source_sessions"]} == {
        "earlier",
        "later",
    }


def test_duplicate_boundary_across_sessions_fails_loudly(tmp_path):
    input_root = tmp_path / "production"
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _write_session(input_root, session_id="session-a", start=start)
    _write_session(input_root, session_id="session-b", start=start)

    with pytest.raises(ValueError, match="Duplicate global boundary timestamps"):
        _build(input_root, tmp_path / "research")
    assert not list((tmp_path / "research").rglob("manifest.json"))


def test_research_boundaries_match_independent_replay_oracle(tmp_path):
    input_root = tmp_path / "production"
    writer, raw_frames = _write_session(input_root)
    result = _build(input_root, tmp_path / "research")
    metadata = json.loads(Path(writer.sealed_segments[0]["metadata_path"]).read_text())
    provenance = metadata["request_metadata"]["frames"]
    replayed = replay_recorded_frames(
        [
            (raw, record["connection_id"], record["frame_index"])
            for raw, record in zip(raw_frames, provenance, strict=True)
        ],
        product_id="BTC-USD",
        session_id="session-1",
        heartbeat_timeout_seconds=5,
        session_started_at_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        session_completed_at_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        derived_at_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    research_rows = pd.read_parquet(result.boundary_artifact)
    for expected, actual in zip(replayed, research_rows.to_dict(orient="records"), strict=True):
        assert actual["boundary_time_utc"].to_pydatetime() == expected.boundary_time_utc
        for name in (
            "source",
            "product_id",
            "session_id",
            "connection_id",
            "source_sequence_num",
            "source_state_time_utc",
            "derived_at_utc",
            "best_bid",
            "best_bid_size",
            "best_ask",
            "best_ask_size",
            "spread",
            "midpoint",
            "book_synced",
            "canonical_target_eligible",
            "eligibility_reason",
            "boundary_schema_version",
        ):
            assert actual[name] == getattr(expected, name)


def test_build_fails_for_product_mismatch_and_output_under_input(tmp_path):
    input_root = tmp_path / "production"
    _write_session(input_root)
    discovered = discover_sealed_segments(input_root)
    with pytest.raises(ValueError, match="product"):
        build_research_dataset(
            input_root=input_root,
            output_root=tmp_path / "research",
            discovered_segments=discovered,
            product_id="ETH-USD",
        )
    for output_root in (input_root, input_root / "research"):
        with pytest.raises(ValueError, match="outside input_root"):
            _build(input_root, output_root, discovered)


def test_idempotent_build_and_source_hash_change_create_immutable_snapshots(tmp_path):
    input_root = tmp_path / "production"
    output_root = tmp_path / "research"
    writer, _ = _write_session(input_root)
    discovered = discover_sealed_segments(input_root)
    first = _build(input_root, output_root, discovered)
    first_manifest = first.manifest_path.read_bytes()
    first_files = sorted(path.name for path in first.dataset_dir.iterdir())
    rebuilt = _build(input_root, output_root, discovered)

    assert rebuilt.dataset_id == first.dataset_id
    assert rebuilt.manifest_path == first.manifest_path
    assert rebuilt.manifest_path.read_bytes() == first_manifest
    assert sorted(path.name for path in rebuilt.dataset_dir.iterdir()) == first_files

    # A later sealed segment changes the exact included source prefix.
    next_time = datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc)
    writer.add_frame(
        raw=_wire_frame("heartbeats", next_time, 5),
        message_type="heartbeat",
        connection_id="session-1-connection",
        ingest_time_utc=next_time,
    )
    writer.seal()
    newer = _build(input_root, output_root)
    assert newer.dataset_id != first.dataset_id
    assert first.manifest_path.read_bytes() == first_manifest
    assert first.dataset_dir != newer.dataset_dir
