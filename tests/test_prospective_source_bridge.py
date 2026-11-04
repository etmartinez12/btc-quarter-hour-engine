from __future__ import annotations

import hashlib
import json
import socket
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from btc_quarter_hour_engine.acquisition.prospective_source_bridge import (
    MAX_RAW_FRAME_BYTES,
    ClockHealthEvidence,
    NonBlockingSourcePublisher,
    ProspectiveSourceReceiver,
    ProspectiveSourceSocketServer,
    _encode_event,
    _read_event,
    parse_chrony_tracking,
)
from btc_quarter_hour_engine.research.prospective_source import (
    ProspectiveRawFrameJournal,
)


UTC = timezone.utc
START = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
BOUNDARY = START + timedelta(minutes=15)


def _clock(*, verified: bool = True) -> ClockHealthEvidence:
    return ClockHealthEvidence(
        observed_at_utc=BOUNDARY + timedelta(seconds=2),
        synchronized=verified,
        verified=verified,
        offset_seconds=0.001 if verified else None,
        reference_time_utc=BOUNDARY + timedelta(seconds=1) if verified else None,
        leap_status="Normal" if verified else None,
        source="fixture",
        reason=None if verified else "unknown synchronization",
    )


class JournalTransport:
    def __init__(self):
        self.events = []

    def send_and_wait_for_ack(self, header, raw_bytes):
        ack = {
            "durable": True,
            "session_id": header["session_id"],
            "through_frame_index": (
                header["frame_index"]
                if header["kind"] == "frame"
                else header["through_frame_index"]
            ),
        }
        self.events.append((header, raw_bytes))
        return ack


def _header(*, kind="frame", index=0, timestamp=None, connection="connection-0"):
    raw = b"fixture-frame"
    return {
        "version": 1,
        "kind": kind,
        "session_id": "session-1",
        "session_started_at_utc": START.isoformat().replace("+00:00", "Z"),
        "connection_id": connection,
        "reconnect_epoch": 0,
        "connection_first_frame_index": 0,
        "frame_index": index,
        "ingest_time_utc": (timestamp or START).isoformat().replace("+00:00", "Z"),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_length": len(raw),
        "producer_health": {"publisher_valid": True},
    }


def test_publisher_receiver_preserve_raw_bytes_and_fence_same_timestamp_ties(tmp_path):
    journal = ProspectiveRawFrameJournal(tmp_path / "source.sqlite")
    receiver = ProspectiveSourceReceiver(journal, clock_health_provider=_clock)
    transport = JournalTransport()
    publisher = NonBlockingSourcePublisher(
        tmp_path / "unused.sock",
        session_id="session-1",
        session_started_at_utc=START,
        transport=transport,
        max_queue_frames=8,
    )
    tied = BOUNDARY
    raws = (b"\x00one", b"two\xff", b"witness")
    times = (tied, tied, tied + timedelta(milliseconds=1))
    for index, (raw, received) in enumerate(zip(raws, times)):
        assert publisher.publish_frame(
            connection_id="connection-0",
            reconnect_epoch=0,
            frame_index=index,
            raw_bytes=raw,
            received_at_utc=received,
            producer_health={"publisher_valid": True},
        )
    publisher._queue.join()
    for header, raw_bytes in transport.events:
        receiver.process(header, raw_bytes)
    assert publisher.valid, publisher.failure_reason
    snapshot = journal.snapshot("session-1")
    assert [frame.raw_bytes for frame in snapshot.frames] == list(raws)
    assert [frame.frame_index for frame in snapshot.frames] == [0, 1, 2]
    assert [frame.ingest_time_utc for frame in snapshot.frames] == list(times)
    assert snapshot.watermark_utc == BOUNDARY
    assert snapshot.clock_health_verified is True
    persisted_health = journal._connection.execute(
        "SELECT frame_index, connection_id, reconnect_epoch, producer_health_json, "
        "producer_health_sha256 FROM source_frame_health ORDER BY frame_index"
    ).fetchall()
    assert [row["frame_index"] for row in persisted_health] == [0, 1, 2]
    assert all(row["connection_id"] == "connection-0" for row in persisted_health)
    assert all(row["reconnect_epoch"] == 0 for row in persisted_health)
    assert all(
        hashlib.sha256(row["producer_health_json"].encode()).hexdigest()
        == row["producer_health_sha256"]
        for row in persisted_health
    )
    persisted_fence = journal._connection.execute(
        "SELECT producer_health_json, producer_health_sha256 FROM source_fences"
    ).fetchone()
    assert hashlib.sha256(
        persisted_fence["producer_health_json"].encode()
    ).hexdigest() == persisted_fence["producer_health_sha256"]
    assert transport.events[-1][0]["kind"] == "fence"
    assert transport.events[-1][0]["through_frame_index"] == 2
    publisher.close()
    journal.close()


def test_fence_before_durable_prefix_and_missing_indices_fail_closed(tmp_path):
    journal = ProspectiveRawFrameJournal(tmp_path / "source.sqlite")
    receiver = ProspectiveSourceReceiver(journal, clock_health_provider=_clock)
    frame_header = _header(
        timestamp=BOUNDARY + timedelta(seconds=1),
    )
    fence_header = {
        "version": 1,
        "kind": "fence",
        "session_id": "session-1",
        "watermark_utc": BOUNDARY.isoformat().replace("+00:00", "Z"),
        "through_frame_index": 0,
        "connection_id": "connection-0",
        "reconnect_epoch": 0,
        "producer_health": {"publisher_valid": True},
        "raw_length": 0,
    }
    with pytest.raises(ValueError, match="durable frame prefix"):
        receiver.process(fence_header, b"")
    assert journal.session_started_at("session-1") is None

    gap_header = dict(frame_header, frame_index=1, connection_first_frame_index=1)
    with pytest.raises(ValueError, match="next source connection"):
        receiver.process(gap_header, b"fixture-frame")
    assert journal._connection.execute(
        "SELECT COUNT(*) FROM source_frames WHERE session_id = 'session-1'"
    ).fetchone()[0] == 0

    corrupt_header = dict(frame_header, raw_length=len(b"corrupted"))
    with pytest.raises(ValueError, match="digest mismatch"):
        receiver.process(corrupt_header, b"corrupted")
    journal.close()


def test_publisher_queue_saturation_is_nonblocking_and_sticky(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    class SlowTransport:
        def send_and_wait_for_ack(self, header, raw_bytes):
            entered.set()
            release.wait(timeout=2)
            return {
                "durable": True,
                "session_id": header["session_id"],
                "through_frame_index": header["frame_index"],
            }

    publisher = NonBlockingSourcePublisher(
        tmp_path / "unused.sock",
        session_id="session-1",
        session_started_at_utc=START,
        transport=SlowTransport(),
        max_queue_frames=1,
        max_queue_bytes=64,
    )
    assert publisher.publish_frame(
        connection_id="connection-0",
        reconnect_epoch=0,
        frame_index=0,
        raw_bytes=b"small",
        received_at_utc=START,
        producer_health={"publisher_valid": True},
    )
    assert entered.wait(timeout=1)
    for index in range(1, 2):
        assert publisher.publish_frame(
            connection_id="connection-0",
            reconnect_epoch=0,
            frame_index=index,
            raw_bytes=b"small",
            received_at_utc=START + timedelta(seconds=index),
            producer_health={"publisher_valid": True},
        )
    before = time.perf_counter()
    assert not publisher.publish_frame(
        connection_id="connection-0",
        reconnect_epoch=0,
        frame_index=2,
        raw_bytes=b"small",
        received_at_utc=START + timedelta(seconds=2),
        producer_health={"publisher_valid": True},
    )
    assert time.perf_counter() - before < 0.1
    assert not publisher.valid
    assert publisher.metrics()["loss_count"] == 1
    release.set()
    publisher.close(timeout_seconds=0.5)


def test_shutdown_timeout_invalidates_unacknowledged_publisher(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    class SlowTransport:
        def send_and_wait_for_ack(self, header, raw_bytes):
            entered.set()
            release.wait(timeout=2)
            return {
                "durable": True,
                "session_id": header["session_id"],
                "through_frame_index": header["frame_index"],
            }

    publisher = NonBlockingSourcePublisher(
        tmp_path / "unused.sock",
        session_id="session-1",
        session_started_at_utc=START,
        transport=SlowTransport(),
    )
    assert publisher.publish_frame(
        connection_id="connection-0",
        reconnect_epoch=0,
        frame_index=0,
        raw_bytes=b"small",
        received_at_utc=START,
        producer_health={"publisher_valid": True},
    )
    assert entered.wait(timeout=1)
    publisher.close(timeout_seconds=0)
    assert not publisher.valid
    assert publisher.failure_reason == "publisher_shutdown_timeout"
    release.set()
    publisher._worker.join(timeout=1)
    assert not publisher._worker.is_alive()


def test_length_prefixed_stream_handles_maximum_websocket_frame():
    left, right = socket.socketpair()
    raw = b"x" * MAX_RAW_FRAME_BYTES
    header = {
        "version": 1,
        "kind": "frame",
        "raw_length": len(raw),
        "session_id": "large-session",
    }
    received = []

    def read():
        received.append(_read_event(right, max_frame_bytes=MAX_RAW_FRAME_BYTES))

    thread = threading.Thread(target=read)
    thread.start()
    left.sendall(_encode_event(header, raw))
    thread.join(timeout=10)
    left.close()
    right.close()
    assert not thread.is_alive()
    assert received == [(header, raw)]


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix-domain transport requires AF_UNIX")
def test_unix_receiver_acknowledges_only_committed_frames(tmp_path):
    socket_path = tmp_path / "source.sock"
    journal_path = tmp_path / "source.sqlite"
    ready = threading.Event()
    errors = []

    def serve():
        journal = ProspectiveRawFrameJournal(journal_path)
        receiver = ProspectiveSourceReceiver(journal, clock_health_provider=_clock)
        server = ProspectiveSourceSocketServer(socket_path, receiver)
        errors.append((receiver, server))
        ready.set()
        try:
            server.serve_forever()
        except Exception as exc:
            errors.append(exc)
        finally:
            journal.close()

    thread = threading.Thread(target=serve)
    thread.start()
    assert ready.wait(timeout=5)
    deadline = time.monotonic() + 5
    while not socket_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert socket_path.exists()
    publisher = NonBlockingSourcePublisher(
        socket_path,
        session_id="session-1",
        session_started_at_utc=START,
    )
    assert publisher.publish_frame(
        connection_id="connection-0",
        reconnect_epoch=0,
        frame_index=0,
        raw_bytes=b"socket-frame",
        received_at_utc=BOUNDARY + timedelta(milliseconds=1),
        producer_health={"publisher_valid": True},
    )
    publisher._queue.join()
    assert publisher.valid, publisher.failure_reason
    publisher.close()
    _, server = errors[0]
    server.stop()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert not any(isinstance(error, Exception) for error in errors[1:])

    journal = ProspectiveRawFrameJournal(journal_path)
    snapshot = journal.snapshot("session-1")
    assert snapshot.frames[0].raw_bytes == b"socket-frame"
    assert snapshot.watermark_utc == BOUNDARY
    journal.close()


def test_partial_and_oversized_stream_messages_are_rejected():
    class PartialConnection:
        def __init__(self, chunks):
            self.chunks = list(chunks)

        def recv(self, _):
            return self.chunks.pop(0) if self.chunks else b""

    with pytest.raises(ConnectionError, match="mid-message"):
        _read_event(
            PartialConnection([(10).to_bytes(8, "big"), b"ab"]),
            max_frame_bytes=32,
        )

    with pytest.raises(ValueError, match="oversized"):
        _read_event(
            PartialConnection([(20_000).to_bytes(8, "big")]),
            max_frame_bytes=32,
        )


def test_chrony_clock_health_requires_fresh_sync_and_offset_tolerance():
    now = datetime(2026, 1, 1, 0, 15, 2, tzinfo=UTC)
    good = """\
Reference time (UTC) : Thu Jan  1 00:15:01 2026
System time     : 0.001 seconds slow of NTP time
Leap status     : Normal
"""
    stale = good.replace("00:15:01", "00:10:00")
    unhealthy = good.replace("0.001", "0.5")
    assert parse_chrony_tracking(good, observed_at_utc=now).verified
    assert not parse_chrony_tracking(stale, observed_at_utc=now).verified
    assert not parse_chrony_tracking(unhealthy, observed_at_utc=now).verified
    assert not parse_chrony_tracking(
        "Leap status : Not synchronised", observed_at_utc=now
    ).verified


def test_fenced_watermark_prevents_late_at_or_before_cutoff_frames(tmp_path):
    journal = ProspectiveRawFrameJournal(tmp_path / "source.sqlite")
    receiver = ProspectiveSourceReceiver(journal, clock_health_provider=_clock)
    header = _header(timestamp=BOUNDARY + timedelta(seconds=1))
    receiver.process(header, b"fixture-frame")
    fence = {
        "version": 1,
        "kind": "fence",
        "session_id": "session-1",
        "watermark_utc": BOUNDARY.isoformat().replace("+00:00", "Z"),
        "through_frame_index": 0,
        "connection_id": "connection-0",
        "reconnect_epoch": 0,
        "producer_health": {"publisher_valid": True},
        "raw_length": 0,
    }
    assert receiver.process(fence, b"")["durable"]
    late = dict(header)
    late.update(
        frame_index=1,
        connection_first_frame_index=0,
        ingest_time_utc=BOUNDARY.isoformat().replace("+00:00", "Z"),
    )
    late["raw_sha256"] = hashlib.sha256(b"fixture-frame").hexdigest()
    with pytest.raises(ValueError, match="regressed|at or before"):
        receiver.process(late, b"fixture-frame")
    journal.close()


def test_receiver_restart_and_reconnect_preserve_global_frame_order(tmp_path):
    path = tmp_path / "source.sqlite"
    journal = ProspectiveRawFrameJournal(path)
    receiver = ProspectiveSourceReceiver(
        journal, clock_health_provider=lambda: _clock(verified=False)
    )
    first = _header(timestamp=BOUNDARY - timedelta(seconds=1))
    receiver.process(first, b"fixture-frame")
    journal.close()

    journal = ProspectiveRawFrameJournal(path)
    receiver = ProspectiveSourceReceiver(
        journal, clock_health_provider=lambda: _clock(verified=False)
    )
    second = _header(index=1, timestamp=BOUNDARY, connection="connection-1")
    second.update(reconnect_epoch=1, connection_first_frame_index=1)
    receiver.process(second, b"fixture-frame")
    third = _header(index=2, timestamp=BOUNDARY + timedelta(milliseconds=1), connection="connection-1")
    third.update(reconnect_epoch=1, connection_first_frame_index=1)
    receiver.process(third, b"fixture-frame")
    fence = {
        "version": 1,
        "kind": "fence",
        "session_id": "session-1",
        "watermark_utc": BOUNDARY.isoformat().replace("+00:00", "Z"),
        "through_frame_index": 2,
        "connection_id": "connection-1",
        "reconnect_epoch": 1,
        "producer_health": {"publisher_valid": True},
        "raw_length": 0,
    }
    receiver.process(fence, b"")
    snapshot = journal.snapshot("session-1")
    assert [frame.frame_index for frame in snapshot.frames] == [0, 1, 2]
    assert [frame.connection_id for frame in snapshot.frames] == [
        "connection-0",
        "connection-1",
        "connection-1",
    ]
    assert snapshot.watermark_utc == BOUNDARY
    assert snapshot.clock_health_verified is False
    journal.close()


def test_sqlite_write_failure_cannot_certify_watermark(tmp_path, monkeypatch):
    journal = ProspectiveRawFrameJournal(tmp_path / "source.sqlite")
    receiver = ProspectiveSourceReceiver(journal, clock_health_provider=_clock)
    monkeypatch.setattr(
        journal,
        "append_frame",
        lambda **_: (_ for _ in ()).throw(sqlite3.OperationalError("disk is full")),
    )
    with pytest.raises(sqlite3.OperationalError, match="disk is full"):
        receiver.process(
            _header(timestamp=BOUNDARY + timedelta(seconds=1)), b"fixture-frame"
        )
    assert journal._connection.execute(
        "SELECT COUNT(*) FROM source_watermarks"
    ).fetchone()[0] == 0
    journal.close()
