from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing
import os
import platform
import socket
import sqlite3
import statistics
import struct
import subprocess
import sys
import threading
import time
import tracemalloc
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btc_quarter_hour_engine.acquisition.prospective_source_bridge import (
    MAX_RAW_FRAME_BYTES,
    ClockHealthEvidence,
    NonBlockingSourcePublisher,
    ProspectiveSourceReceiver,
    ProspectiveSourceSocketServer,
    UnixSocketFrameTransport,
    _encode_event,
    _read_event,
    _recv_exact,
    _send_ack,
)
from btc_quarter_hour_engine.research.prospective_source import (
    ProspectiveRawFrameJournal,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _latencies(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "samples": len(ordered),
        "median_ms": statistics.median(ordered) if ordered else None,
        "p95_ms": percentile(0.95) if ordered else None,
        "p99_ms": percentile(0.99) if ordered else None,
        "max_ms": max(ordered) if ordered else None,
    }


def _rss_bytes() -> int:
    if sys.platform.startswith("linux"):
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD,
        ]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.WorkingSetSize
    raise RuntimeError("RSS measurement requires Linux or Windows")


class _ResourceSampler:
    def __init__(self) -> None:
        self.baseline = _rss_bytes()
        self.peak = self.baseline
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def _sample(self) -> None:
        while not self.stop.wait(0.01):
            self.peak = max(self.peak, _rss_bytes())

    def finish(self, cpu_seconds: float, wall_seconds: float) -> dict[str, float | int]:
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, _rss_bytes())
        return {
            "cpu_seconds": cpu_seconds,
            "cpu_percent_one_core": 100 * cpu_seconds / wall_seconds if wall_seconds else 0,
            "rss_baseline_bytes": self.baseline,
            "rss_peak_bytes": self.peak,
            "rss_growth_bytes": self.peak - self.baseline,
        }


def _book_frames(levels: int, updates: int, updates_per_frame: int, rate: int):
    """Deterministic Advanced Trade snapshot, healthy crossing, and depth mutations."""
    if levels < 2 or levels % 2:
        raise ValueError("Populated level count must be even and at least two")
    if updates < 1 or updates_per_frame < 1 or rate < 1:
        raise ValueError("Update counts and arrival rate must be positive")
    boundary = datetime(2026, 1, 1, 0, 15, tzinfo=timezone.utc)

    def iso(timestamp):
        return timestamp.isoformat(timespec="microseconds").replace("+00:00", "Z")

    def mutation(index, timestamp, quantity):
        side = "bid" if index % 2 == 0 else "offer"
        depth = index // 2
        cents = 6_000_000 - depth if side == "bid" else 6_000_001 + depth
        return {
            "side": side,
            "price_level": f"{cents // 100}.{cents % 100:02d}",
            "new_quantity": quantity,
            "event_time": iso(timestamp),
        }

    def l2(sequence, timestamp, kind, changes):
        return json.dumps({
            "channel": "l2_data", "client_id": "", "timestamp": iso(timestamp),
            "sequence_num": sequence,
            "events": [{"type": kind, "product_id": "BTC-USD", "updates": changes}],
        }, separators=(",", ":")).encode()

    started = boundary - timedelta(seconds=2)
    yield started, l2(1, started, "snapshot", [
        mutation(index, started, "0.010000") for index in range(levels)
    ])
    heartbeat = boundary - timedelta(seconds=1)
    yield heartbeat, json.dumps({
        "channel": "heartbeats", "client_id": "", "timestamp": iso(heartbeat),
        "sequence_num": 2,
        "events": [{"current_time": iso(heartbeat), "heartbeat_counter": "1"}],
    }, separators=(",", ":")).encode()
    sequence = 3
    counter = 1
    for index in range(updates):
        timestamp = boundary + timedelta(seconds=1 + index / rate)
        if index and index % rate == 0:
            counter += 1
            yield timestamp, json.dumps({
                "channel": "heartbeats", "client_id": "", "timestamp": iso(timestamp),
                "sequence_num": sequence,
                "events": [{"current_time": iso(timestamp), "heartbeat_counter": str(counter)}],
            }, separators=(",", ":")).encode()
            sequence += 1
        changes = [
            mutation(
                ((index * updates_per_frame + offset) * 7919) % levels,
                timestamp, f"0.{10000 + (index + offset) % 90000:06d}",
            )
            for offset in range(updates_per_frame)
        ]
        yield timestamp, l2(sequence, timestamp, "update", changes)
        sequence += 1


def _book_receiver(pipe, root_string: str, checkpoint_interval: int | None, recover: bool = False):
    root = Path(root_string)
    serialization: list[float] = []
    transactions: list[float] = []
    process_times: list[float] = []
    sampler = _ResourceSampler()
    cpu_start, wall_start = time.process_time(), time.perf_counter()

    class MeasuredJournal(ProspectiveRawFrameJournal):
        def _canonical_checkpoint_payload(self, **kwargs):
            begin = time.perf_counter()
            value = super()._canonical_checkpoint_payload(**kwargs)
            serialization.append((time.perf_counter() - begin) * 1000)
            return value

        @contextmanager
        def _transaction(self):
            begin = time.perf_counter()
            with super()._transaction():
                yield
            transactions.append((time.perf_counter() - begin) * 1000)

    class MeasuredReceiver(ProspectiveSourceReceiver):
        def process(self, header, raw):
            begin = time.perf_counter()
            value = super().process(header, raw)
            if header["kind"] == "frame" and header["frame_index"] >= 2:
                process_times.append((time.perf_counter() - begin) * 1000)
            return value

    journal = None
    try:
        journal = MeasuredJournal(root / "source.sqlite")
        kwargs = {} if checkpoint_interval is None else {"checkpoint_interval_frames": checkpoint_interval}
        receiver = MeasuredReceiver(
            journal,
            clock_health_provider=lambda: ClockHealthEvidence(
                observed_at_utc=datetime(2026, 1, 1, 0, 15, 1, tzinfo=timezone.utc),
                synchronized=False, verified=False, offset_seconds=None,
                reference_time_utc=None, leap_status=None, source="synthetic populated-book benchmark",
                reason="benchmark does not assert independent UTC synchronization",
            ),
            **kwargs,
        )
        if recover:
            recovery_samples = []
            for _ in range(5):
                replay = None
                gc.collect()
                receiver = MeasuredReceiver(journal, **kwargs)
                begin = time.perf_counter()
                replay = receiver._canonical_replay("populated-book-benchmark", 5.0)
                recovery_samples.append((time.perf_counter() - begin) * 1000)
            pipe.send({
                "first_process_cold_recovery_ms": recovery_samples[0],
                "fresh_receiver_recovery": _latencies(recovery_samples),
                "state_sha256": hashlib.sha256(json.dumps(replay.to_checkpoint(), sort_keys=True).encode()).hexdigest(),
                "resources": sampler.finish(time.process_time() - cpu_start, time.perf_counter() - wall_start),
                "receiver": receiver.metrics(),
            })
            return

        if os.name == "posix":
            server = ProspectiveSourceSocketServer(root / "source.sock", receiver)
            pipe.send({"transport": "production Unix-domain socket server", "endpoint": str(root / "source.sock")})

            def control():
                pipe.recv()
                server.stop()

            thread = threading.Thread(target=control, daemon=True)
            thread.start()
            server.serve_forever()
            thread.join(timeout=5)
        else:
            # Windows cannot start the fail-closed production Unix socket server.
            # Exercise identical framing, receiver, FULL commits and ACKs over TCP.
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                listener.listen()
                listener.settimeout(0.1)
                pipe.send({"transport": "Windows loopback TCP framing; NOT a Unix socket gate", "endpoint": listener.getsockname()})
                while not pipe.poll():
                    try:
                        connection, _ = listener.accept()
                    except socket.timeout:
                        continue
                    with connection:
                        connection.settimeout(5)
                        header, raw = _read_event(connection, max_frame_bytes=MAX_RAW_FRAME_BYTES)
                        _send_ack(connection, receiver.process(header, raw))
                pipe.recv()
        run_wall = time.perf_counter() - wall_start
        resources = sampler.finish(time.process_time() - cpu_start, run_wall)
        replay = receiver._canonical_replays.get("populated-book-benchmark")
        digest = hashlib.sha256(json.dumps(replay.to_checkpoint(), sort_keys=True).encode()).hexdigest() if replay else None
        row = journal._connection.execute(
            "SELECT length(checkpoint_json), frame_index FROM source_canonical_checkpoints"
        ).fetchone()
        warm_start = time.perf_counter()
        warm = ProspectiveSourceReceiver(journal)._canonical_replay("populated-book-benchmark", 5.0) if replay else None
        warm_ms = (time.perf_counter() - warm_start) * 1000
        fence = journal._connection.execute(
            "SELECT producer_health_json FROM source_fences ORDER BY watermark_utc DESC LIMIT 1"
        ).fetchone()
        pipe.send({
            "receiver": receiver.metrics(),
            "receiver_process": _latencies(process_times),
            "checkpoint_json_serialization": _latencies(serialization),
            "sqlite_full_transaction": _latencies(transactions),
            "sqlite_synchronous": journal._connection.execute("PRAGMA synchronous").fetchone()[0],
            "checkpoint_bytes": row[0] if row else 0,
            "checkpoint_frame_index": row[1] if row else None,
            "journal_bytes_before_close": sum(p.stat().st_size for p in root.glob("source.sqlite*")),
            "resources": resources,
            "state_sha256": digest,
            "warm_recovery_ms": warm_ms,
            "warm_recovery_identical": bool(warm and hashlib.sha256(json.dumps(warm.to_checkpoint(), sort_keys=True).encode()).hexdigest() == digest),
            "canonical_boundary_eligible": bool(fence and json.loads(fence[0])["canonical_boundary"]["canonical_target_eligible"]),
        })
    except Exception as exc:
        pipe.send({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        sampler.stop.set()
        sampler.thread.join()
        if journal is not None:
            journal.close()
        pipe.close()


def run_populated_book_benchmark(
    *, levels: int, updates: int = 1024, updates_per_frame: int = 10,
    rate: int = 100, burst_size: int = 1, enabled: bool = True,
    checkpoint_interval: int | None = None,
) -> dict[str, object]:
    from btc_quarter_hour_engine.acquisition.coinbase_websocket import CoinbaseWebSocketClient
    from btc_quarter_hour_engine.acquisition.collector import WebSocketCollector
    from btc_quarter_hour_engine.acquisition.config import CoinbaseWebSocketConfig
    from btc_quarter_hour_engine.acquisition.websocket_service import CoinbaseWebSocketService
    from btc_quarter_hour_engine.storage.forward_parquet import ForwardParquetStore
    from btc_quarter_hour_engine.storage.websocket_raw import RawSegmentWriter

    if burst_size < 1:
        raise ValueError("Burst size must be positive")
    with TemporaryDirectory(prefix="book-gate-") as directory:
        root = Path(directory)
        parent = child = receiver_process = None
        checkpoint_report = {}
        transport_note = "integration disabled"
        if enabled:
            context = multiprocessing.get_context("spawn")
            parent, child = context.Pipe()
            receiver_process = context.Process(target=_book_receiver, args=(child, directory, checkpoint_interval))
            receiver_process.start()
            child.close()
            if not parent.poll(30):
                receiver_process.terminate()
                receiver_process.join()
                raise RuntimeError("Benchmark receiver startup timed out")
            ready = parent.recv()
            if "error" in ready:
                receiver_process.join()
                raise RuntimeError(ready["error"])
            transport_note = ready["transport"]
            endpoint = ready["endpoint"]
            if os.name == "posix":
                deadline = time.monotonic() + 10
                while not Path(endpoint).exists() and time.monotonic() < deadline:
                    time.sleep(0.01)

        collector_times: list[float] = []
        arrival_lag: list[float] = []
        enqueue_times: list[float] = []
        ack_times: list[float] = []
        delivery_times: list[float] = []
        queued_at: dict[tuple[str, int], float] = {}
        publisher = None

        class Feed:
            def __init__(self):
                self.frames = iter(_book_frames(levels, updates, updates_per_frame, rate))
                self.index = -1
                self.now = datetime(2026, 1, 1, 0, 14, 58, tzinfo=timezone.utc)
                self.begin = None
                self.processing_started = 0.0

            def connect(self, **_kwargs):
                pass

            def send(self, _payload):
                pass

            def close(self):
                pass

            def recv(self, timeout=None):
                timestamp, raw = next(self.frames)
                self.index += 1
                if self.index == 2:
                    if publisher is not None:
                        publisher._queue.join()
                    self.begin = time.perf_counter()
                if self.index >= 2:
                    due = self.begin + ((self.index - 2) // burst_size) * burst_size / rate
                    delay = due - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    arrival_lag.append(max(0, (time.perf_counter() - due) * 1000))
                self.now = timestamp
                self.processing_started = time.perf_counter()
                return raw

        feed = Feed()

        class AckTransport:
            def send_and_wait_for_ack(self, header, raw_bytes):
                begin = time.perf_counter()
                if os.name == "posix":
                    ack = UnixSocketFrameTransport(endpoint).send_and_wait_for_ack(header, raw_bytes)
                else:
                    with socket.create_connection(endpoint, timeout=5) as connection:
                        connection.sendall(_encode_event(header, raw_bytes))
                        size = struct.unpack("!I", _recv_exact(connection, 4))[0]
                        ack = json.loads(_recv_exact(connection, size))
                if ack.get("durable") is not True:
                    raise RuntimeError("Benchmark received a non-durable ACK")
                index = header.get("frame_index", header.get("through_frame_index"))
                if header["kind"] == "frame" and index >= 2:
                    end = time.perf_counter()
                    ack_times.append((end - begin) * 1000)
                    delivery_times.append((end - queued_at[("frame", index)]) * 1000)
                return ack

        class MeasuredPublisher(NonBlockingSourcePublisher):
            def _enqueue(self, header, raw_bytes=b""):
                begin = time.perf_counter()
                index = header.get("frame_index", header.get("through_frame_index"))
                queued_at[(header["kind"], index)] = begin
                result = super()._enqueue(header, raw_bytes)
                if header["kind"] == "frame" and index >= 2:
                    enqueue_times.append((time.perf_counter() - begin) * 1000)
                return result

        class MeasuredCollector(WebSocketCollector):
            def _publish_source_frame(self, frame, frame_index, observations):
                super()._publish_source_frame(frame, frame_index, observations)
                if frame_index >= 2:
                    collector_times.append((time.perf_counter() - feed.processing_started) * 1000)

        config = CoinbaseWebSocketConfig(heartbeat_timeout_seconds=5.0)
        client = CoinbaseWebSocketClient(config=config, transport=feed, now_fn=lambda: feed.now)
        service = CoinbaseWebSocketService(
            config=config, client=client, session_id="populated-book-benchmark", now_fn=lambda: feed.now,
        )
        if enabled:
            publisher = MeasuredPublisher(
                root / "source.sock", session_id=service.session_id,
                session_started_at_utc=feed.now, transport=AckTransport(),
            )
        sampler = _ResourceSampler()
        cpu_start, wall_start = time.process_time(), time.perf_counter()
        collector = MeasuredCollector(
            service=service,
            raw_segment_writer=RawSegmentWriter(root / "collector", product_id="BTC-USD", max_frames=4096),
            forward_store=ForwardParquetStore(root / "collector"), output_root=str(root / "collector"),
            source_publisher=publisher,
        )
        frame_count = updates + 2 + (updates - 1) // rate
        try:
            result = collector.run(max_messages=frame_count)
            if publisher is not None:
                publisher._queue.join()
            wall_seconds = time.perf_counter() - wall_start
            resources = sampler.finish(time.process_time() - cpu_start, wall_seconds)
            publisher_metrics = publisher.metrics() if publisher is not None else None
            if enabled:
                parent.send("stop")
                if not parent.poll(30):
                    raise RuntimeError("Benchmark receiver stop timed out")
                checkpoint_report = parent.recv()
                receiver_process.join(timeout=10)
                if "error" in checkpoint_report:
                    raise RuntimeError(checkpoint_report["error"])
                journal_bytes = sum(p.stat().st_size for p in root.glob("source.sqlite*"))
                recovery_pipe, recovery_child = context.Pipe()
                recovered_process = context.Process(target=_book_receiver, args=(recovery_child, directory, checkpoint_interval, True))
                recovery_start = time.perf_counter()
                recovered_process.start()
                recovery_child.close()
                if not recovery_pipe.poll(30):
                    recovered_process.terminate()
                    recovered_process.join()
                    raise RuntimeError("Benchmark checkpoint recovery timed out")
                recovery = recovery_pipe.recv()
                recovered_process.join(timeout=10)
                recovery_pipe.close()
                if "error" in recovery:
                    raise RuntimeError(recovery["error"])
                recovery["process_start_and_recovery_ms"] = (time.perf_counter() - recovery_start) * 1000
                recovery["identical_state"] = recovery["state_sha256"] == checkpoint_report["state_sha256"]
                checkpoint_report["cold_recovery"] = recovery
                checkpoint_report["journal_bytes_after_close"] = journal_bytes
                checkpoint_report["journal_bytes_per_committed_frame"] = journal_bytes / max(1, checkpoint_report["receiver"]["frames_committed"])
            return {
                "levels": levels, "incremental_update_frames": updates,
                "updates_per_frame": updates_per_frame, "offered_frames_per_second": rate,
                "burst_size": burst_size, "integration_enabled": enabled,
                "checkpoint_interval": checkpoint_interval, "transport": transport_note,
                "arrival_profile": "burst" if burst_size > 1 else "normal",
                "wall_seconds": wall_seconds, "collector_process": _latencies(collector_times),
                "collector_arrival_lag": _latencies(arrival_lag), "source_enqueue": _latencies(enqueue_times),
                "durable_frame_ack": _latencies(ack_times), "source_queue_to_ack": _latencies(delivery_times),
                "collector_resources": resources, "publisher": publisher_metrics,
                "collector_termination": result.termination_reason,
                "collector_source_metrics": result.source_metrics,
                "source_queue_capacity_frames": publisher._queue.maxsize if publisher else None,
                "source_queue_capacity_bytes": publisher.max_queue_bytes if publisher else None,
                "collector_populated_levels": len(service.order_book.bids) + len(service.order_book.asks),
                "checkpoint": checkpoint_report,
                "measurement_note": "Synthetic valid Coinbase wire data; snapshot/heartbeat warmup drains before open-loop paced/burst arrivals. Percentiles exclude warmup for frames; transactions/serialization include bootstrap and fences. CPU is process time; RSS sampled every 10 ms. No OS page-cache purge or live UTC evidence.",
            }
        finally:
            sampler.stop.set()
            sampler.thread.join()
            if publisher is not None:
                publisher.close(timeout_seconds=5)
            if receiver_process is not None:
                if receiver_process.is_alive():
                    receiver_process.terminate()
                    receiver_process.join()
                parent.close()


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
    parser.add_argument("--populated-books", action="store_true", help="Run isolated 10k/25k/40k level scenarios instead of padded/malformed history.")
    parser.add_argument("--book-updates", type=_positive_int, default=1024)
    parser.add_argument("--book-burst-rate", type=_positive_int, default=100, help="Average offered rate in 32-frame bursts; use 1000 for overload stress.")
    parser.add_argument("--book-levels", default="10000,25000,40000")
    parser.add_argument("--book-intervals", default="default", help="Comma-separated checkpoint intervals, e.g. 1,default; disabled collector is always included.")
    parser.add_argument("--require-book-capacity", action="store_true", help="Fail if the default bounded path loses frames, fails canonical parity/recovery, or interrupts collection.")
    parser.add_argument("--book-worker", help=argparse.SUPPRESS)
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
    if args.book_worker:
        print(json.dumps(run_populated_book_benchmark(**json.loads(args.book_worker))))
        return
    if args.populated_books:
        scenarios = []
        for levels in (int(value) for value in args.book_levels.split(",")):
            for rate, burst in ((100, 1), (args.book_burst_rate, 32)):
                for interval in ("disabled", *args.book_intervals.split(",")):
                    parameters = {
                        "levels": levels, "updates": args.book_updates, "rate": rate, "burst_size": burst,
                        "enabled": interval != "disabled",
                        "checkpoint_interval": None if interval in {"default", "disabled"} else int(interval),
                    }
                    completed = subprocess.run(
                        [sys.executable, str(Path(__file__).resolve()), "--book-worker", json.dumps(parameters)],
                        capture_output=True, text=True,
                    )
                    if completed.returncode:
                        raise RuntimeError(f"Populated-book worker failed for {parameters}: {completed.stderr}")
                    scenarios.append(json.loads(completed.stdout))
        baselines = {
            (item["levels"], item["arrival_profile"]): item
            for item in scenarios if not item["integration_enabled"]
        }
        for item in scenarios:
            if item["integration_enabled"]:
                disabled = baselines[(item["levels"], item["arrival_profile"])]
                item["enabled_vs_disabled_collector_median_ratio"] = (
                    item["collector_process"]["median_ms"] / disabled["collector_process"]["median_ms"]
                )
        print(json.dumps({
            "environment": {
                "platform": platform.platform(), "python": platform.python_version(),
                "sqlite": sqlite3.sqlite_version, "logical_cpus": os.cpu_count(),
                "checkpoint_policy": "default: at most 128 tail frames / 8 MiB; fence checkpoints remain atomic",
            },
            "populated_book_scenarios": scenarios,
        }, indent=2, sort_keys=True))
        if args.require_book_capacity:
            failed = [
                (item["levels"], item["arrival_profile"])
                for item in scenarios
                if item["integration_enabled"] and item["checkpoint_interval"] is None
                and (
                    not item["publisher"]["valid"]
                    or item["checkpoint"]["receiver"]["frames_committed"] != item["publisher"]["frame_enqueue_count"]
                    or not item["checkpoint"]["cold_recovery"]["identical_state"]
                    or not item["checkpoint"]["warm_recovery_identical"]
                    or not item["checkpoint"]["canonical_boundary_eligible"]
                    or item["collector_termination"] != "max_messages"
                )
            ]
            if failed:
                raise RuntimeError(f"Populated-book ingestion capacity gate failed: {failed}")
        return
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
