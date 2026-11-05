from __future__ import annotations

import errno
import hashlib
import json
import multiprocessing
import os
import socket
import sqlite3
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from btc_quarter_hour_engine.acquisition import prospective_source_bridge as bridge_module
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
    _validate_bridge_health,
    _require_bridge_prediction_health,
)
from btc_quarter_hour_engine.market_data.replay import CanonicalReplayAccumulator


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


def _producer_health(connection_id="connection-0", *, canonical_boundary=None):
    return {
        "publisher_valid": True,
        "post_frame_health": {
            "heartbeat_seen": False,
            "heartbeat_last_at_utc": None,
            "heartbeat_healthy": False,
            "order_book_synchronized": False,
            "connection_id": connection_id,
            "connection_last_sequence_num": None,
            "last_l2_sequence_num": None,
            "sequence_gap_count": 0,
            "stale_sequence_count": 0,
            "heartbeat_timeout_count": 0,
            "heartbeat_discontinuity_count": 0,
            "malformed_frame_count": 0,
            "malformed_level2_count": 0,
            "crossed_book_count": 0,
            "heartbeat_timeout_seconds": 5.0,
        },
        "canonical_boundary_finalized": canonical_boundary is not None,
        "canonical_boundary": canonical_boundary,
    }


def _boundary_health(
    connection_id,
    boundary,
    *,
    eligible,
    reason="eligible",
    synced=True,
    heartbeat_at=None,
    sequence=1,
):
    heartbeat = heartbeat_at or boundary - timedelta(seconds=1)
    canonical = {
        "boundary_utc": boundary.isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        ),
        "canonical_target_eligible": eligible,
        "eligibility_reason": reason,
        "book_synced": synced,
        "last_heartbeat_at_utc": heartbeat.isoformat().replace("+00:00", "Z"),
        "heartbeat_timeout_seconds": 5.0,
        "connection_id": connection_id,
        "source_sequence_num": sequence if synced else None,
    }
    health = _producer_health(connection_id, canonical_boundary=canonical)
    health["post_frame_health"].update(
        heartbeat_seen=True,
        heartbeat_last_at_utc=canonical["last_heartbeat_at_utc"],
        heartbeat_healthy=eligible,
        order_book_synchronized=synced,
        connection_last_sequence_num=sequence,
        last_l2_sequence_num=sequence if synced else None,
        sequence_gap_count=0 if synced else 1,
    )
    return health


def _wire_frame(sequence, timestamp, *, snapshot=False, heartbeat=False):
    event_time = timestamp.isoformat().replace("+00:00", "Z")
    if heartbeat:
        channel = "heartbeats"
        events = [{"current_time": event_time, "heartbeat_counter": str(sequence)}]
    else:
        channel = "l2_data"
        updates = (
            [
                {"side": "bid", "price_level": "100", "new_quantity": "1", "event_time": event_time},
                {"side": "offer", "price_level": "101", "new_quantity": "2", "event_time": event_time},
            ]
            if snapshot
            else [
                {"side": "bid", "price_level": "100", "new_quantity": "1.5", "event_time": event_time},
            ]
        )
        events = [{
            "type": "snapshot" if snapshot else "update",
            "product_id": "BTC-USD",
            "updates": updates,
        }]
    return json.dumps(
        {
            "channel": channel,
            "timestamp": event_time,
            "sequence_num": sequence,
            "events": events,
        },
        separators=(",", ":"),
    ).encode()


def _frame_header(raw, *, index, received, health, session_id="session-1"):
    return {
        "version": 1,
        "kind": "frame",
        "session_id": session_id,
        "session_started_at_utc": START.isoformat().replace("+00:00", "Z"),
        "connection_id": "connection-0",
        "reconnect_epoch": 0,
        "connection_first_frame_index": 0,
        "frame_index": index,
        "ingest_time_utc": received.isoformat().replace("+00:00", "Z"),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_length": len(raw),
        "producer_health": health,
    }


def _canonical_boundary_health(raw_frames, ingest_times):
    replay = CanonicalReplayAccumulator(
        product_id="BTC-USD", session_id="session-1", heartbeat_timeout_seconds=5
    )
    for index, (raw, ingest) in enumerate(zip(raw_frames, ingest_times)):
        replay.consume(raw, "connection-0", index, ingest)
    observation = replay.boundary_observation(
        BOUNDARY,
        derived_at_utc=BOUNDARY + timedelta(minutes=15),
        connections=[{"connection_id": "connection-0"}],
    )
    health = _boundary_health(
        "connection-0",
        BOUNDARY,
        eligible=observation.canonical_target_eligible,
        reason=observation.eligibility_reason,
        synced=observation.book_synced,
        heartbeat_at=observation.last_heartbeat_at_utc,
        sequence=observation.source_sequence_num or 0,
    )
    health["canonical_boundary"]["last_heartbeat_at_utc"] = (
        observation.last_heartbeat_at_utc.isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        )
        if observation.last_heartbeat_at_utc is not None
        else None
    )
    health["post_frame_health"].update(
        heartbeat_seen=True,
        heartbeat_last_at_utc=(
            observation.last_heartbeat_at_utc.isoformat().replace("+00:00", "Z")
            if observation.last_heartbeat_at_utc is not None
            else None
        ),
        heartbeat_healthy=observation.canonical_target_eligible,
        order_book_synchronized=True,
        connection_last_sequence_num=3,
        last_l2_sequence_num=3,
    )
    return replay, health


def _clock_at(boundary, *, verified=True):
    return ClockHealthEvidence(
        observed_at_utc=boundary + timedelta(seconds=2),
        synchronized=verified,
        verified=verified,
        offset_seconds=0.001 if verified else None,
        reference_time_utc=boundary + timedelta(seconds=1) if verified else None,
        leap_status="Normal" if verified else None,
        source="fixture",
        reason=None if verified else "unknown synchronization",
    )


def _run_socket_server_process(socket_path, status_pipe):
    journal = ProspectiveRawFrameJournal(socket_path.with_suffix(".sqlite"))
    receiver = ProspectiveSourceReceiver(journal, clock_health_provider=_clock)
    server = ProspectiveSourceSocketServer(socket_path, receiver)
    try:
        server.serve_forever()
    except Exception as exc:
        status_pipe.send((type(exc).__name__, str(exc)))
    finally:
        journal.close()
        status_pipe.close()


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
        "producer_health": _producer_health(connection),
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
            producer_health=_producer_health(),
        )
    publisher._queue.join()
    for header, raw_bytes in transport.events:
        receiver.process(header, raw_bytes)
    assert publisher.valid, publisher.failure_reason
    persisted_frames = journal._connection.execute(
        "SELECT frame_index, raw_bytes, ingest_time_utc FROM source_frames "
        "ORDER BY frame_index"
    ).fetchall()
    assert [bytes(row["raw_bytes"]) for row in persisted_frames] == list(raws)
    assert [row["frame_index"] for row in persisted_frames] == [0, 1, 2]
    assert [datetime.fromisoformat(row["ingest_time_utc"].replace("Z", "+00:00"))
            for row in persisted_frames] == list(times)
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
    assert journal._connection.execute("SELECT COUNT(*) FROM source_fences").fetchone()[0] == 0
    assert all(header["kind"] == "frame" for header, _ in transport.events)
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
        "producer_health": _producer_health(),
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
        producer_health=_producer_health(),
    )
    assert entered.wait(timeout=1)
    for index in range(1, 2):
        assert publisher.publish_frame(
            connection_id="connection-0",
            reconnect_epoch=0,
            frame_index=index,
            raw_bytes=b"small",
            received_at_utc=START + timedelta(seconds=index),
            producer_health=_producer_health(),
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
        producer_health=_producer_health(),
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
    assert journal._connection.execute(
        "SELECT raw_bytes FROM source_frames WHERE session_id = 'session-1'"
    ).fetchone()["raw_bytes"] == b"socket-frame"
    assert journal._connection.execute(
        "SELECT COUNT(*) FROM source_prefix_commitments WHERE session_id = 'session-1'"
    ).fetchone()[0] == 1
    with pytest.raises(ValueError, match="no committed producer fence"):
        journal.snapshot("session-1")
    journal.close()


def test_ordered_verified_fence_certifies_durable_snapshot(tmp_path):
    journal = ProspectiveRawFrameJournal(tmp_path / "source.sqlite")

    clock = _clock_at(BOUNDARY)
    receiver = ProspectiveSourceReceiver(
        journal, clock_health_provider=lambda: clock
    )
    ingest_times = (
        BOUNDARY - timedelta(seconds=2),
        BOUNDARY - timedelta(seconds=1),
        BOUNDARY + timedelta(seconds=1),
    )
    raw_frames = (
        _wire_frame(1, ingest_times[0], snapshot=True),
        _wire_frame(2, ingest_times[1], heartbeat=True),
        _wire_frame(3, ingest_times[2]),
    )
    _replay, final_health = _canonical_boundary_health(raw_frames, ingest_times)
    publisher = NonBlockingSourcePublisher(
        tmp_path / "unused.sock",
        session_id="session-1",
        session_started_at_utc=START,
        transport=JournalTransport(),
    )
    for index, (raw, received) in enumerate(zip(raw_frames, ingest_times)):
        assert publisher.publish_frame(
            connection_id="connection-0",
            reconnect_epoch=0,
            frame_index=index,
            raw_bytes=raw,
            received_at_utc=received,
            producer_health=(
                final_health if index == len(raw_frames) - 1 else _producer_health()
            ),
        )
    publisher._queue.join()
    assert publisher.valid, publisher.failure_reason
    publisher.close()
    for header, raw_bytes in publisher.transport.events:
        receiver.process(header, raw_bytes)

    snapshot = journal.snapshot("session-1")
    assert snapshot.source_certified is True
    assert snapshot.watermark_utc == BOUNDARY
    assert [frame.raw_bytes for frame in snapshot.frames] == list(raw_frames)
    assert snapshot.producer_health["canonical_boundary"]["boundary_utc"] == (
        BOUNDARY.isoformat(timespec="microseconds").replace("+00:00", "Z")
    )
    assert journal._connection.execute(
        "SELECT COUNT(*) FROM source_fences WHERE session_id = 'session-1'"
    ).fetchone()[0] == 1
    journal.close()

    journal = ProspectiveRawFrameJournal(tmp_path / "source.sqlite")
    restarted = ProspectiveSourceReceiver(
        journal, clock_health_provider=lambda: clock
    )
    after_fence = _wire_frame(
        4, BOUNDARY + timedelta(seconds=2)
    )
    restarted.process(
        _frame_header(
            after_fence,
            index=3,
            received=BOUNDARY + timedelta(seconds=2),
            health=_producer_health(),
        ),
        after_fence,
    )
    checkpoint = journal._connection.execute(
        "SELECT frame_index FROM source_canonical_checkpoints "
        "WHERE session_id = 'session-1'"
    ).fetchone()
    assert checkpoint["frame_index"] == 3
    assert journal._connection.execute(
        "SELECT COUNT(*) FROM source_fences WHERE session_id = 'session-1'"
    ).fetchone()[0] == 1
    assert journal.snapshot("session-1").source_certified
    journal._connection.execute(
        "UPDATE source_canonical_checkpoints SET checkpoint_json = '{}' "
        "WHERE session_id = 'session-1'"
    )
    with pytest.raises(ValueError, match="checkpoint"):
        journal.snapshot("session-1")
    journal.close()


@pytest.mark.skipif(
    not (sys.platform.startswith("linux") and hasattr(socket, "AF_UNIX")),
    reason="Process-level stale Unix socket recovery is Linux-only",
)
def test_unix_server_recovery_after_forced_process_termination(tmp_path):
    context = multiprocessing.get_context("fork")
    socket_path = tmp_path / "restart.sock"

    def start_server():
        parent_pipe, child_pipe = context.Pipe(duplex=False)
        process = context.Process(
            target=_run_socket_server_process,
            args=(socket_path, child_pipe),
        )
        process.start()
        child_pipe.close()
        return process, parent_pipe

    def wait_for_listening(path, process, pipe):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if pipe.poll():
                pytest.fail(f"Receiver failed to start: {pipe.recv()}")
            if not process.is_alive():
                pytest.fail(f"Receiver exited with code {process.exitcode}")
            if path.exists():
                probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    probe.settimeout(0.1)
                    probe.connect(str(path))
                    return
                except OSError:
                    pass
                finally:
                    probe.close()
            time.sleep(0.01)
        pytest.fail("Receiver did not become available")

    active, active_pipe = start_server()
    contender, contender_pipe = None, None
    recovered, recovered_pipe = None, None
    try:
        wait_for_listening(socket_path, active, active_pipe)
        original_inode = socket_path.lstat().st_ino

        contender, contender_pipe = start_server()
        assert contender_pipe.poll(5)
        error_type, error_message = contender_pipe.recv()
        contender.join(timeout=5)
        assert error_type == "RuntimeError"
        assert "owns the socket lock" in error_message
        assert socket_path.lstat().st_ino == original_inode
        assert active.is_alive()

        active.terminate()
        active.join(timeout=5)
        assert not active.is_alive()
        assert socket_path.exists()

        recovered, recovered_pipe = start_server()
        wait_for_listening(socket_path, recovered, recovered_pipe)
        assert recovered.is_alive()
    finally:
        for process in (active, contender, recovered):
            if process is not None and process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for pipe in (active_pipe, contender_pipe, recovered_pipe):
            if pipe is not None:
                pipe.close()


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Requires Unix-domain sockets")
def test_socket_recovery_fails_closed_for_non_socket_foreign_or_ambiguous_paths(
    tmp_path, monkeypatch
):
    non_socket_path = tmp_path / "ordinary-file"
    non_socket_path.write_text("not a socket")
    non_socket_journal = ProspectiveRawFrameJournal(tmp_path / "non-socket.sqlite")
    server = ProspectiveSourceSocketServer(
        non_socket_path, ProspectiveSourceReceiver(non_socket_journal)
    )
    with pytest.raises(FileExistsError, match="non-socket"):
        server._remove_stale_socket_if_safe()
    assert non_socket_path.read_text() == "not a socket"
    non_socket_journal.close()

    foreign_path = tmp_path / "foreign.sock"
    owner_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    owner_socket.bind(str(foreign_path))
    owner_socket.listen(1)
    foreign_journal = ProspectiveRawFrameJournal(tmp_path / "foreign.sqlite")
    foreign_server = ProspectiveSourceSocketServer(
        foreign_path, ProspectiveSourceReceiver(foreign_journal)
    )
    real_geteuid = os.geteuid
    monkeypatch.setattr(bridge_module.os, "geteuid", lambda: real_geteuid() + 1)
    with pytest.raises(PermissionError, match="another user"):
        foreign_server._remove_stale_socket_if_safe()
    monkeypatch.setattr(bridge_module.os, "geteuid", real_geteuid)
    owner_socket.close()
    foreign_path.unlink()
    foreign_journal.close()

    ambiguous_path = tmp_path / "ambiguous.sock"
    ambiguous_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    ambiguous_socket.bind(str(ambiguous_path))
    ambiguous_socket.listen(1)
    ambiguous_journal = ProspectiveRawFrameJournal(tmp_path / "ambiguous.sqlite")
    ambiguous_server = ProspectiveSourceSocketServer(
        ambiguous_path, ProspectiveSourceReceiver(ambiguous_journal)
    )

    class AmbiguousProbe:
        def settimeout(self, _):
            pass

        def connect(self, _):
            raise PermissionError(errno.EACCES, "probe denied")

        def close(self):
            pass

    real_socket_factory = bridge_module.socket.socket
    monkeypatch.setattr(
        bridge_module.socket,
        "socket",
        lambda *args, **kwargs: AmbiguousProbe(),
    )
    with pytest.raises(RuntimeError, match="whether.*stale"):
        ambiguous_server._remove_stale_socket_if_safe()
    monkeypatch.setattr(bridge_module.socket, "socket", real_socket_factory)
    ambiguous_socket.close()
    ambiguous_path.unlink()
    ambiguous_journal.close()


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


def test_unfinalized_boundary_cannot_be_fenced_or_certified(tmp_path):
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
        "producer_health": _producer_health(),
        "raw_length": 0,
    }
    with pytest.raises(ValueError, match="finalized canonical boundary"):
        receiver.process(fence, b"")
    assert journal._connection.execute("SELECT COUNT(*) FROM source_fences").fetchone()[0] == 0
    late = dict(header)
    late.update(
        frame_index=1,
        connection_first_frame_index=0,
        ingest_time_utc=BOUNDARY.isoformat().replace("+00:00", "Z"),
    )
    late["raw_sha256"] = hashlib.sha256(b"fixture-frame").hexdigest()
    with pytest.raises(ValueError, match="regressed"):
        receiver.process(late, b"fixture-frame")
    journal.close()


def test_receiver_restart_and_reconnect_preserve_global_frame_order_without_fence(tmp_path):
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
    persisted = journal._connection.execute(
        "SELECT frame_index, connection_id FROM source_frames ORDER BY frame_index"
    ).fetchall()
    assert [(row["frame_index"], row["connection_id"]) for row in persisted] == [
        (0, "connection-0"),
        (1, "connection-1"),
        (2, "connection-1"),
    ]
    assert journal._connection.execute("SELECT COUNT(*) FROM source_fences").fetchone()[0] == 0
    assert journal.session_mode("session-1") == "bridge"
    checkpoint = journal._connection.execute(
        "SELECT frame_index, prefix_commitment_sha256 FROM source_canonical_checkpoints "
        "WHERE session_id = 'session-1'"
    ).fetchone()
    assert checkpoint["frame_index"] == 2
    assert checkpoint["prefix_commitment_sha256"]
    journal.close()


def test_bridge_mode_rejects_legacy_and_forged_clock_certification(tmp_path):
    path = tmp_path / "source.sqlite"
    journal = ProspectiveRawFrameJournal(path)
    receiver = ProspectiveSourceReceiver(journal, clock_health_provider=_clock)
    header = _header(timestamp=BOUNDARY + timedelta(seconds=1))
    receiver.process(header, b"fixture-frame")
    with pytest.raises(ValueError, match="committed producer fence"):
        journal.advance_watermark(
            "session-1", BOUNDARY, clock_health_verified=True
        )

    forged_clock = ClockHealthEvidence(
        observed_at_utc=BOUNDARY + timedelta(seconds=2),
        synchronized=True,
        verified=True,
        offset_seconds=0.5,
        reference_time_utc=BOUNDARY + timedelta(seconds=1),
        leap_status="Normal",
        source="forged fixture",
    )
    forged_receiver = ProspectiveSourceReceiver(
        journal, clock_health_provider=lambda: forged_clock
    )
    with pytest.raises(ValueError, match="finalized canonical boundary"):
        forged_receiver.process(
            {
                "version": 1,
                "kind": "fence",
                "session_id": "session-1",
                "watermark_utc": BOUNDARY.isoformat().replace("+00:00", "Z"),
                "through_frame_index": 0,
                "connection_id": "connection-0",
                "reconnect_epoch": 0,
                "producer_health": header["producer_health"],
                "raw_length": 0,
            },
            b"",
        )
    assert journal._connection.execute(
        "SELECT COUNT(*) FROM source_watermarks"
    ).fetchone()[0] == 0
    journal.close()


def test_later_healthy_reconnection_does_not_rewrite_unhealthy_frame_evidence(tmp_path):
    journal = ProspectiveRawFrameJournal(tmp_path / "source.sqlite")
    boundaries = (BOUNDARY, BOUNDARY + timedelta(minutes=15))
    receiver = ProspectiveSourceReceiver(journal, clock_health_provider=_clock)

    first_health = _boundary_health(
        "connection-0",
        boundaries[0],
        eligible=False,
        reason="sequence_gap",
        synced=False,
        sequence=1,
    )
    first = _header(timestamp=boundaries[0] + timedelta(seconds=1))
    first["producer_health"] = first_health
    receiver.process(first, b"fixture-frame")
    recovered_health = _boundary_health(
        "connection-1", boundaries[1], eligible=True, sequence=2
    )
    recovered = _header(
        index=1,
        timestamp=boundaries[1] + timedelta(seconds=1),
        connection="connection-1",
    )
    recovered.update(reconnect_epoch=1, connection_first_frame_index=1)
    recovered["producer_health"] = recovered_health
    receiver.process(recovered, b"fixture-frame")
    stored = journal._connection.execute(
        "SELECT frame_index, producer_health_json FROM source_frame_health "
        "ORDER BY frame_index"
    ).fetchall()
    assert json.loads(stored[0]["producer_health_json"])["canonical_boundary"][
        "eligibility_reason"
    ] == "sequence_gap"
    assert json.loads(stored[1]["producer_health_json"])["canonical_boundary"][
        "canonical_target_eligible"
    ] is True
    assert journal._connection.execute("SELECT COUNT(*) FROM source_fences").fetchone()[0] == 0
    journal.close()


@pytest.mark.parametrize("corruption", ["missing", "altered", "discontinuous"])
def test_receiver_fails_closed_on_invalid_canonical_recovery_checkpoint(
    tmp_path, corruption
):
    path = tmp_path / "source.sqlite"
    journal = ProspectiveRawFrameJournal(path)
    receiver = ProspectiveSourceReceiver(journal, clock_health_provider=_clock)
    receiver.process(
        _header(timestamp=BOUNDARY - timedelta(seconds=1)), b"fixture-frame"
    )
    checkpoint = journal._connection.execute(
        "SELECT frame_index, prefix_commitment_sha256, checkpoint_json, "
        "checkpoint_sha256 FROM source_canonical_checkpoints "
        "WHERE session_id = 'session-1'"
    ).fetchone()
    assert checkpoint["frame_index"] == 0
    if corruption == "missing":
        journal._connection.execute(
            "DROP TRIGGER source_canonical_checkpoints_no_delete"
        )
        journal._connection.execute(
            "DELETE FROM source_canonical_checkpoints WHERE session_id = 'session-1'"
        )
    elif corruption == "altered":
        journal._connection.execute(
            "UPDATE source_canonical_checkpoints SET checkpoint_json = '{}' "
            "WHERE session_id = 'session-1'"
        )
    else:
        payload = json.loads(checkpoint["checkpoint_json"])
        payload["state"]["frame_count"] += 1
        checkpoint_json, checkpoint_sha256 = (
            journal._canonical_checkpoint_payload(
                session_id="session-1",
                frame_index=checkpoint["frame_index"],
                prefix_commitment_sha256=checkpoint["prefix_commitment_sha256"],
                state=payload["state"],
            )
        )
        journal._connection.execute(
            "UPDATE source_canonical_checkpoints SET checkpoint_json = ?, "
            "checkpoint_sha256 = ? WHERE session_id = 'session-1'",
            (checkpoint_json, checkpoint_sha256),
        )
    journal.close()

    journal = ProspectiveRawFrameJournal(path)
    restarted = ProspectiveSourceReceiver(journal, clock_health_provider=_clock)
    second = _header(index=1, timestamp=BOUNDARY, connection="connection-0")
    with pytest.raises(ValueError, match="(?i)checkpoint"):
        restarted.process(second, b"fixture-frame")
    assert journal._connection.execute(
        "SELECT COUNT(*) FROM source_frames WHERE session_id = 'session-1'"
    ).fetchone()[0] == 1
    journal.close()


@pytest.mark.parametrize(
    "health, message",
    [
        ({"publisher_valid": True}, "incomplete"),
        (
            {
                **_boundary_health(
                    "connection-0", BOUNDARY, eligible=True, sequence=1
                ),
                "canonical_boundary": {
                    **_boundary_health(
                        "connection-0", BOUNDARY, eligible=True, sequence=1
                    )["canonical_boundary"],
                    "last_heartbeat_at_utc": (
                        BOUNDARY + timedelta(seconds=1)
                    ).isoformat().replace("+00:00", "Z"),
                },
            },
            "contradicts its evidence",
        ),
    ],
)
def test_bridge_rejects_missing_or_inconsistent_health_evidence(
    tmp_path, health, message
):
    journal = ProspectiveRawFrameJournal(tmp_path / "source.sqlite")
    receiver = ProspectiveSourceReceiver(journal, clock_health_provider=_clock)
    header = _header(timestamp=BOUNDARY + timedelta(seconds=1))
    header["producer_health"] = health
    with pytest.raises(ValueError, match=message):
        receiver.process(header, b"fixture-frame")
    assert journal.session_started_at("session-1") is None
    journal.close()


def test_boundary_connection_may_differ_from_post_cutoff_witness_connection():
    health = _boundary_health(
        "connection-before-boundary", BOUNDARY, eligible=True, sequence=1
    )
    health["post_frame_health"]["connection_id"] = "connection-after-boundary"
    validated = _validate_bridge_health(
        health,
        boundary=BOUNDARY,
        connection_id="connection-after-boundary",
    )
    assert (
        validated["canonical_boundary"]["connection_id"]
        == "connection-before-boundary"
    )


@pytest.mark.parametrize(
    "health",
    [
        _boundary_health(
            "connection-0",
            BOUNDARY,
            eligible=False,
            reason="heartbeat_stale",
            heartbeat_at=BOUNDARY - timedelta(seconds=10),
        ),
        _boundary_health(
            "connection-0",
            BOUNDARY,
            eligible=False,
            reason="no_synced_snapshot",
            synced=False,
        ),
        _boundary_health(
            "connection-0",
            BOUNDARY,
            eligible=False,
            reason="sequence_gap",
            synced=False,
        ),
    ],
)
def test_bridge_prediction_requires_canonical_heartbeat_and_book_eligibility(health):
    with pytest.raises(ValueError, match="canonical_target_ineligible_at_boundary"):
        _require_bridge_prediction_health(
            source_mode="bridge",
            source_certified=True,
            producer_health=health,
            boundary=BOUNDARY,
            session_id="session-1",
        )


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
