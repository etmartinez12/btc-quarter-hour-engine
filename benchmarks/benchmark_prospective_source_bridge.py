from __future__ import annotations

import argparse
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


def run_benchmark(
    *,
    frame_count_per_size: int = 3,
    frame_sizes: tuple[int, ...] = (1024, 65_536, MAX_RAW_FRAME_BYTES),
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
                        "publisher_valid": publisher.valid,
                        "heartbeat_healthy": True,
                        "order_book_synchronized": True,
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
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure the local prospective source publisher and journal path."
    )
    parser.add_argument("--frame-count-per-size", type=_positive_int, default=3)
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
        ),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
