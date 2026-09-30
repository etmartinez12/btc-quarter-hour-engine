from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

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
