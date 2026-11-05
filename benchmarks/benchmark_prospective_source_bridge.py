from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import threading
import time
import tracemalloc
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btc_quarter_hour_engine.acquisition.prospective_source_bridge import (
    MAX_RAW_FRAME_BYTES,
    ClockHealthEvidence,
    NonBlockingSourcePublisher,
    ProspectiveSourceReceiver,
    ProspectiveSourceSocketServer,
)
from btc_quarter_hour_engine.research.prospective_source import (
    ProspectiveRawFrameJournal,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _frame_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("frame sizes must be comma-separated integers") from exc
    if not sizes or any(size < 1 or size > MAX_RAW_FRAME_BYTES for size in sizes):
        raise argparse.ArgumentTypeError(
            f"frame sizes must be between 1 and {MAX_RAW_FRAME_BYTES} bytes"
        )
    return sizes


def _synthetic_snapshot(target_size: int) -> bytes:
    prefix = (
        b'{"channel":"l2_data","client_id":"","timestamp":"2026-01-01T00:00:00Z",'
        b'"sequence_num":1,"events":[{"type":"snapshot","product_id":"BTC-USD",'
        b'"updates":[{"side":"bid","price_level":"100.00","new_quantity":"1.0",'
        b'"event_time":"2026-01-01T00:00:00Z"},{"side":"offer","price_level":"101.00",'
        b'"new_quantity":"1.0","event_time":"2026-01-01T00:00:00Z"}]}],'
        b'"benchmark_padding":"'
    )
    suffix = b'"}'
    padding_size = target_size - len(prefix) - len(suffix)
    if padding_size < 0:
        if target_size >= 2:
            return b"{}" + b" " * (target_size - 2)
        return b"x" * target_size
    return prefix + b"x" * padding_size + suffix


def _benchmark_health(connection_id: str) -> dict[str, object]:
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
        "canonical_boundary_finalized": False,
        "canonical_boundary": None,
    }


def _run_checkpoint_recovery_benchmark(history_frame_count: int) -> dict[str, object]:
    """Measure checkpoint recovery without replaying the historical raw journal."""
    with TemporaryDirectory(prefix="prospective-checkpoint-benchmark-") as directory:
        journal_path = Path(directory) / "source.sqlite"
        session_id = "long-session-benchmark"
        connection_id = "benchmark-connection"
        started = datetime(2026, 1, 1, tzinfo=timezone.utc)
        raw = b"synthetic malformed frame for bounded-state recovery"
        health = _benchmark_health(connection_id)
        journal = ProspectiveRawFrameJournal(journal_path)
        receiver = ProspectiveSourceReceiver(
            journal,
            clock_health_provider=lambda: ClockHealthEvidence(
                observed_at_utc=started,
                synchronized=False,
                verified=False,
                offset_seconds=None,
                reference_time_utc=None,
                leap_status=None,
                source="synthetic recovery benchmark",
                reason="no live clock evidence",
            ),
        )
        tracemalloc.start()
        history_started = time.perf_counter()
        for index in range(history_frame_count):
            received = started + timedelta(seconds=index + 1)
            receiver.process(
                {
                    "version": 1,
                    "kind": "frame",
                    "session_id": session_id,
                    "session_started_at_utc": started.isoformat().replace("+00:00", "Z"),
                    "connection_id": connection_id,
                    "reconnect_epoch": 0,
                    "connection_first_frame_index": 0,
                    "frame_index": index,
                    "ingest_time_utc": received.isoformat().replace("+00:00", "Z"),
                    "raw_sha256": hashlib.sha256(raw).hexdigest(),
                    "raw_length": len(raw),
                    "producer_health": health,
                },
                raw,
            )
        history_seconds = time.perf_counter() - history_started
        _, history_peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        checkpoint_size = journal._connection.execute(
            "SELECT length(checkpoint_json) FROM source_canonical_checkpoints "
            "WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
        journal.close()

        tracemalloc.start()
        journal = ProspectiveRawFrameJournal(journal_path)
        restarted_receiver = ProspectiveSourceReceiver(
            journal,
            clock_health_provider=lambda: ClockHealthEvidence(
                observed_at_utc=started,
                synchronized=False,
                verified=False,
                offset_seconds=None,
                reference_time_utc=None,
                leap_status=None,
                source="synthetic recovery benchmark",
                reason="no live clock evidence",
            ),
        )

        def append(index: int) -> None:
            received = started + timedelta(seconds=index + 1)
            restarted_receiver.process(
                {
                    "version": 1,
                    "kind": "frame",
                    "session_id": session_id,
                    "session_started_at_utc": started.isoformat().replace("+00:00", "Z"),
                    "connection_id": connection_id,
                    "reconnect_epoch": 0,
                    "connection_first_frame_index": 0,
                    "frame_index": index,
                    "ingest_time_utc": received.isoformat().replace("+00:00", "Z"),
                    "raw_sha256": hashlib.sha256(raw).hexdigest(),
                    "raw_length": len(raw),
                    "producer_health": health,
                },
                raw,
            )

        cold_started = time.perf_counter()
        append(history_frame_count)
        cold_recovery_seconds = time.perf_counter() - cold_started
        _, cold_peak_bytes = tracemalloc.get_traced_memory()
        cold_metrics = restarted_receiver.metrics()
        warm_started = time.perf_counter()
        append(history_frame_count + 1)
        warm_continuation_seconds = time.perf_counter() - warm_started
        _, warm_peak_bytes = tracemalloc.get_traced_memory()
        journal.close()
        tracemalloc.stop()
        return {
            "historical_frames": history_frame_count,
            "history_append_seconds": history_seconds,
            "history_python_peak_bytes": history_peak_bytes,
            "checkpoint_bytes": checkpoint_size,
            "cold_receiver_recovery_seconds": cold_recovery_seconds,
            "cold_receiver_recovery_python_peak_bytes": cold_peak_bytes,
            "cold_checkpoint_load_latency_max_ms": cold_metrics[
                "canonical_checkpoint_load_latency_max_ms"
            ],
            "warm_continuation_seconds": warm_continuation_seconds,
            "warm_continuation_python_peak_bytes": warm_peak_bytes,
            "recovery_mode_note": (
                "new receiver and reopened SQLite connection; OS page cache is not purged"
            ),
        }


def run_benchmark(
    *,
    frame_count_per_size: int = 3,
    frame_sizes: tuple[int, ...] = (1024, 65_536, MAX_RAW_FRAME_BYTES),
    history_frame_count: int = 10_000,
) -> dict[str, object]:
    with TemporaryDirectory(prefix="prospective-source-benchmark-") as directory:
        root = Path(directory)
        socket_path = root / "source.sock"
        journal_path = root / "source.sqlite"
        now = datetime.now(timezone.utc)
        base = now.replace(
            minute=(now.minute // 15) * 15,
            second=0,
            microsecond=0,
        )
        started = base - timedelta(hours=4)
        clock = ClockHealthEvidence(
            observed_at_utc=base + timedelta(seconds=1),
            synchronized=False,
            verified=False,
            offset_seconds=None,
            reference_time_utc=None,
            leap_status=None,
            source="synthetic benchmark",
            reason="benchmark clock evidence is intentionally unverified",
        )
        ready = threading.Event()
        errors: list[str] = []
        receiver_holder: list[ProspectiveSourceReceiver] = []
        server_holder: list[ProspectiveSourceSocketServer] = []
        server_thread: threading.Thread | None = None
        benchmark_transport: object | None = None
        tracemalloc.start()
        payload_by_size = {size: _synthetic_snapshot(size) for size in frame_sizes}
        max_payload_size = max(len(payload) for payload in payload_by_size.values())

        def receive() -> None:
            journal = ProspectiveRawFrameJournal(journal_path)
            receiver = ProspectiveSourceReceiver(
                journal, clock_health_provider=lambda: clock
            )
            server = ProspectiveSourceSocketServer(socket_path, receiver)
            receiver_holder.append(receiver)
            server_holder.append(server)
            ready.set()
            try:
                server.serve_forever()
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
            finally:
                journal.close()

        if hasattr(socket, "AF_UNIX"):
            server_thread = threading.Thread(
                target=receive, name="source-benchmark-receiver"
            )
            server_thread.start()
            deadline = time.monotonic() + 10
            while not socket_path.exists() and time.monotonic() < deadline:
                if not server_thread.is_alive():
                    break
                time.sleep(0.01)
            if not ready.wait(timeout=10) or not socket_path.exists():
                if errors:
                    raise RuntimeError(errors[0])
                raise RuntimeError("Source receiver did not create its local socket")
            receiver = receiver_holder[0]
            transport_name = "Unix-domain stream and SQLite"
        else:
            journal = ProspectiveRawFrameJournal(journal_path)
            receiver = ProspectiveSourceReceiver(
                journal, clock_health_provider=lambda: clock
            )

            class InMemoryTransport:
                def __init__(self) -> None:
                    self.events: list[tuple[dict[str, object], bytes]] = []

                def send_and_wait_for_ack(self, header, raw_bytes):
                    self.events.append((header, raw_bytes))
                    return {
                        "durable": True,
                        "session_id": header["session_id"],
                        "through_frame_index": (
                            header["frame_index"]
                            if header["kind"] == "frame"
                            else header["through_frame_index"]
                        ),
                    }

            benchmark_transport = InMemoryTransport()
            transport_name = "in-memory fake transport and SQLite (platform fallback)"

        publisher = NonBlockingSourcePublisher(
            socket_path,
            session_id="synthetic-benchmark-session",
            session_started_at_utc=started,
            max_queue_frames=max(64, frame_count_per_size * len(frame_sizes) * 2 + 8),
            max_queue_bytes=max(64 * 1024 * 1024, max_payload_size * 2),
            max_frame_bytes=max_payload_size,
            transport=benchmark_transport,
        )
        cpu_started = time.process_time()
        wall_started = time.perf_counter()
        next_index = 0
        for size in frame_sizes:
            for burst_index in range(frame_count_per_size):
                if next_index == 0:
                    received = base - timedelta(milliseconds=1)
                elif next_index == 1:
                    received = base
                elif next_index == 2:
                    received = base + timedelta(milliseconds=1)
                else:
                    received = base + timedelta(milliseconds=1 + next_index)
                if not publisher.publish_frame(
                    connection_id="synthetic-connection",
                    reconnect_epoch=0,
                    frame_index=next_index,
                    raw_bytes=payload_by_size[size],
                    received_at_utc=received,
                    producer_health={
                        **_benchmark_health("synthetic-connection"),
                        "publisher_valid": publisher.valid,
                    },
                ):
                    break
                next_index += 1
        publisher._queue.join()
        publisher.close(timeout_seconds=2.0)
        wall_seconds = time.perf_counter() - wall_started
        cpu_seconds = time.process_time() - cpu_started
        _, peak_python_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        publisher_metrics = publisher.metrics()
        if server_thread is not None:
            server_holder[0].stop()
            server_thread.join(timeout=10)
            if server_thread.is_alive():
                raise RuntimeError("Source receiver failed to stop")
        else:
            for header, raw_bytes in benchmark_transport.events:
                receiver.process(header, raw_bytes)
            journal.close()
        if errors:
            raise RuntimeError(errors[0])

        recovery_started = time.perf_counter()
        recovered = ProspectiveRawFrameJournal(journal_path)
        recovery_seconds = time.perf_counter() - recovery_started
        frame_total = recovered._connection.execute(
            "SELECT COUNT(*) FROM source_frames WHERE session_id = ?",
            ("synthetic-benchmark-session",),
        ).fetchone()[0]
        fence_total = recovered._connection.execute(
            "SELECT COUNT(*) FROM source_fences WHERE session_id = ?",
            ("synthetic-benchmark-session",),
        ).fetchone()[0]
        recovered.close()
        receiver_metrics = receiver.metrics()
        return {
            "mode": f"synthetic {transport_name}; not production/T+60 evidence",
            "frames_requested_per_size": frame_count_per_size,
            "requested_frame_sizes_bytes": list(frame_sizes),
            "frame_sizes_bytes": sorted(
                {len(payload) for payload in payload_by_size.values()}
            ),
            "events_enqueued": publisher_metrics["enqueue_count"],
            "frames_enqueued": publisher_metrics["frame_enqueue_count"],
            "fences_enqueued": publisher_metrics["fence_enqueue_count"],
            "frames_committed": receiver_metrics["frames_committed"],
            "fences_committed": fence_total,
            "journal_rows_after_restart": frame_total,
            "wall_seconds": wall_seconds,
            "cpu_seconds": cpu_seconds,
            "python_peak_bytes": peak_python_bytes,
            "publisher": publisher_metrics,
            "receiver": receiver_metrics,
            "journal_reopen_seconds": recovery_seconds,
            "continuity_failures": receiver_metrics["source_gaps"]
            + receiver_metrics["frames_rejected"],
            "publisher_valid": publisher_metrics["valid"],
            "checkpoint_recovery": _run_checkpoint_recovery_benchmark(
                history_frame_count
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure the local prospective source publisher and journal path."
    )
    parser.add_argument("--frame-count-per-size", type=_positive_int, default=3)
    parser.add_argument(
        "--history-frames",
        type=_positive_int,
        default=10_000,
        help="Historical source rows to exercise before receiver restart recovery.",
    )
    parser.add_argument(
        "--frame-sizes",
        type=_frame_sizes,
        default=(1024, 65_536, MAX_RAW_FRAME_BYTES),
        help="Comma-separated raw frame sizes; default includes the 16 MiB WS maximum.",
    )
    args = parser.parse_args()
    print(json.dumps(
        run_benchmark(
            frame_count_per_size=args.frame_count_per_size,
            frame_sizes=args.frame_sizes,
            history_frame_count=args.history_frames,
        ),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
