from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from btc_quarter_hour_engine.research.forward_dataset import (
    build_research_dataset,
    discover_sealed_segments,
)
from btc_quarter_hour_engine.storage.websocket_raw import RawSegmentWriter


def _one_frame_input(root):
    writer = RawSegmentWriter(
        root, product_id="BTC-USD", session_id="manifest-session"
    )
    writer.add_frame(
        raw=b'{"channel":"subscriptions","timestamp":"2026-01-01T00:01:00Z","sequence_num":1,"events":[]}',
        message_type="subscriptions",
        connection_id="manifest-connection",
        ingest_time_utc=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
    )
    writer.seal()
    return discover_sealed_segments(root)


def test_manifest_records_reproducibility_and_sealed_prefix_semantics(tmp_path):
    input_root = tmp_path / "production"
    discovered = _one_frame_input(input_root)
    first = build_research_dataset(
        input_root=input_root,
        output_root=tmp_path / "research-a",
        discovered_segments=discovered,
        git_sha="fixed-git-sha",
    )
    second = build_research_dataset(
        input_root=input_root,
        output_root=tmp_path / "research-b",
        discovered_segments=discovered,
        git_sha="fixed-git-sha",
    )
    manifest = first.manifest

    assert first.dataset_id == second.dataset_id
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert manifest["input_mode"] == "sealed_prefix_snapshot"
    assert manifest["snapshot_semantics"] == {
        "input_is_complete_session": False,
        "input_is_sealed_prefix": True,
        "active_partial_files_ignored": True,
        "future_segments_may_supersede_snapshot": True,
        "canonical_relative_to_included_prefix": True,
        "description": manifest["snapshot_semantics"]["description"],
    }
    assert manifest["software"]["git_sha"] == "fixed-git-sha"
    assert manifest["source_summary"]["frame_count"] == 1
    assert manifest["boundary_summary"]["total"] == 0
    assert manifest["boundary_summary"]["eligible"] == 0
    assert manifest["target_summary"]["candidate_rows"] == 0
    assert manifest["target_summary"]["eligible_labels"] == 0
    assert len(pd.read_parquet(first.boundary_artifact)) == 0
    assert len(pd.read_parquet(first.target_artifact)) == 0
    assert json.loads(first.manifest_path.read_text()) == manifest
    for artifact in manifest["artifacts"]:
        assert artifact["sha256"]
        assert artifact["row_count"] == 0


def test_source_product_and_schema_version_are_rejected(tmp_path):
    input_root = tmp_path / "production"
    discovered = _one_frame_input(input_root)
    metadata_path = discovered.segments[0].metadata_path
    import json

    metadata = json.loads(metadata_path.read_text())
    metadata["product_id"] = "ETH-USD"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="product mismatch"):
        discover_sealed_segments(input_root)

    # Restore the fixture and verify unknown raw schemas fail closed.
    metadata["product_id"] = "BTC-USD"
    metadata["request_metadata"]["raw_segment_schema_version"] = "999"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Unsupported raw segment schema version"):
        discover_sealed_segments(input_root)
