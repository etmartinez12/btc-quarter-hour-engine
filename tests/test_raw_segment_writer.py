from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from btc_quarter_hour_engine.storage.websocket_raw import RawSegmentWriter


def test_add_frame_buffers_until_sealed_explicitly(tmp_path):
    writer = RawSegmentWriter(tmp_path, product_id="BTC-USD", max_frames=10)
    now = datetime.now(timezone.utc)
    assert writer.add_frame(raw='{"type":"heartbeat"}', message_type="heartbeat", received_at_utc=now) is None
    assert writer.pending_frame_count == 1
    assert writer.sealed_segments == []
    sealed = writer.seal()
    assert sealed is not None
    assert sealed["frame_count"] == 1
    assert writer.pending_frame_count == 0
    assert writer.sealed_segments == [sealed]


def test_auto_seals_at_max_frames(tmp_path):
    writer = RawSegmentWriter(tmp_path, product_id="BTC-USD", max_frames=3)
    now = datetime.now(timezone.utc)
    for i in range(2):
        result = writer.add_frame(raw=f'{{"i":{i}}}', message_type="l2_data", sequence_num=i, received_at_utc=now)
        assert result is None
    sealed = writer.add_frame(raw='{"i":2}', message_type="l2_data", sequence_num=2, received_at_utc=now)
    assert sealed is not None
    assert sealed["frame_count"] == 3
    assert writer.pending_frame_count == 0


def test_seal_is_noop_when_nothing_buffered(tmp_path):
    writer = RawSegmentWriter(tmp_path, product_id="BTC-USD")
    assert writer.seal() is None


def test_sealed_segment_round_trips_exact_frame_bytes(tmp_path):
    writer = RawSegmentWriter(tmp_path, product_id="BTC-USD", max_frames=10)
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    frames = [
        '{"type":"snapshot","sequence_num":1}',
        '{"type":"l2_data","sequence_num":2, "note": "contains \\n escaped newline, not a real one"}',
        "",  # degenerate empty frame must still round-trip
    ]
    for i, frame in enumerate(frames):
        writer.add_frame(
            raw=frame,
            message_type="snapshot" if i == 0 else "l2_data",
            sequence_num=i + 1,
            received_at_utc=base + timedelta(seconds=i),
        )
    sealed = writer.seal()
    recovered = writer.read_segment_frames(sealed["path"])
    assert [f.decode("utf-8") for f in recovered] == frames


def test_sealed_segment_is_content_addressed_and_verifiable(tmp_path):
    writer = RawSegmentWriter(tmp_path, product_id="BTC-USD", max_frames=10)
    writer.add_frame(raw='{"type":"heartbeat"}', message_type="heartbeat", received_at_utc=datetime.now(timezone.utc))
    sealed = writer.seal()
    assert writer.raw_store.verify_digest(sealed["path"], sealed["sha256"])
    payload = writer.raw_store.read_response(sealed["path"])
    assert hashlib.sha256(payload).hexdigest() == sealed["sha256"]


def test_sealed_segment_metadata_tracks_provenance(tmp_path):
    writer = RawSegmentWriter(tmp_path, product_id="BTC-USD", max_frames=10)
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    writer.add_frame(raw="a", message_type="snapshot", connection_id="conn-1", sequence_num=1, received_at_utc=base)
    writer.add_frame(raw="b", message_type="l2_data", connection_id="conn-1", sequence_num=2, received_at_utc=base + timedelta(seconds=1))
    sealed = writer.seal()
    assert sealed["connection_id"] == "conn-1"
    assert sealed["first_sequence_num"] == 1
    assert sealed["last_sequence_num"] == 2
    assert sealed["segment_index"] == 0

    writer.add_frame(raw="c", message_type="heartbeat", connection_id="conn-1", received_at_utc=base + timedelta(seconds=2))
    second = writer.seal()
    assert second["segment_index"] == 1
    assert second["first_sequence_num"] is None


def test_frames_are_durably_appended_and_sealed_with_session_provenance(tmp_path):
    writer = RawSegmentWriter(
        tmp_path,
        product_id="BTC-USD",
        session_id="session-1",
        max_frames=10,
    )
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    frame = b'{ "sequence_num": 7, "raw":true } \n'
    writer.add_frame(
        raw=frame,
        message_type="l2_data",
        connection_id="connection-1",
        ingest_time_utc=base,
    )

    partial = writer.active_partial_path
    assert partial.suffix == ".partial"
    assert partial.exists()
    assert partial.read_bytes() == len(frame).to_bytes(8, "big") + frame
    assert writer.sealed_segments == []

    sealed = writer.seal()
    assert sealed is not None
    assert not partial.exists()
    assert sealed["session_id"] == "session-1"
    assert sealed["connection_id"] == "connection-1"
    assert sealed["raw_segment_schema_version"] == "1"
    assert "frames" not in sealed
    assert sealed["metadata_path"].endswith(".meta.json")
    assert sealed["frame_count"] == 1
    assert writer.read_segment_frames(sealed["path"]) == [frame]
    metadata = json.loads(Path(sealed["metadata_path"]).read_text())
    assert metadata["request_metadata"]["session_id"] == "session-1"
    assert metadata["request_metadata"]["frames"] == [{
        "frame_index": 0,
        "connection_id": "connection-1",
        "ingest_time_utc": "2024-01-01T00:00:00Z",
        "sequence_num": 7,
        "raw_frame_sha256": hashlib.sha256(frame).hexdigest(),
        "message_type": "l2_data",
    }]
    assert list(writer.iter_sealed_frames())[0].raw_bytes == frame


def test_lazy_sealed_frame_iteration_preserves_global_order_and_provenance(tmp_path):
    writer = RawSegmentWriter(tmp_path, product_id="BTC-USD", session_id="session-1", max_frames=2)
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    source = [
        (b"first", "connection-1"),
        (b"second", "connection-1"),
        (b"third", "connection-2"),
    ]
    for index, (payload, connection_id) in enumerate(source):
        writer.add_frame(
            raw=payload,
            message_type="heartbeat",
            connection_id=connection_id,
            ingest_time_utc=base + timedelta(seconds=index),
        )
    writer.seal()

    frames = list(writer.iter_sealed_frames())
    assert [(frame.raw_bytes, frame.connection_id, frame.frame_index) for frame in frames] == [
        (b"first", "connection-1", 0),
        (b"second", "connection-1", 1),
        (b"third", "connection-2", 2),
    ]


def test_unsealed_partial_remains_incomplete_after_writer_restart(tmp_path):
    first = RawSegmentWriter(tmp_path, product_id="BTC-USD", session_id="crashed-session")
    first.add_frame(raw=b'{"type":"heartbeat"}', message_type="heartbeat")
    partial_path = first.active_partial_path
    assert partial_path.exists()

    restarted = RawSegmentWriter(tmp_path, product_id="BTC-USD", session_id="new-session")
    assert restarted.sealed_segments == []
    assert partial_path.exists()
    assert not list(partial_path.parent.glob("*.json.gz"))


def test_add_frame_rejects_naive_timestamp(tmp_path):
    writer = RawSegmentWriter(tmp_path, product_id="BTC-USD")
    with pytest.raises(ValueError, match="timezone-aware"):
        writer.add_frame(raw="x", message_type="heartbeat", received_at_utc=datetime(2024, 1, 1))


def test_read_segment_frames_rejects_corrupt_framing(tmp_path):
    writer = RawSegmentWriter(tmp_path, product_id="BTC-USD")
    writer.add_frame(raw="x", message_type="heartbeat", received_at_utc=datetime.now(timezone.utc))
    sealed = writer.seal()
    corrupt_path = tmp_path / "corrupt.json.gz"
    import gzip

    with gzip.open(sealed["path"], "rb") as handle:
        payload = handle.read()
    with gzip.open(corrupt_path, "wb") as handle:
        handle.write(payload[:-1])  # truncate mid-frame
    with pytest.raises(ValueError, match="Corrupt sealed segment"):
        writer.read_segment_frames(corrupt_path)
