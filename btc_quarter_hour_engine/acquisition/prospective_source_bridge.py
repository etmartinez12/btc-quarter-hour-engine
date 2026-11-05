from __future__ import annotations

import errno
import copy
import hashlib
import json
import math
import os
import queue
import re
import socket
import stat
import struct
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from btc_quarter_hour_engine.research.prospective_source import (
    MAX_CANONICAL_CHECKPOINT_TAIL_BYTES,
    MAX_CANONICAL_CHECKPOINT_TAIL_FRAMES,
    SOURCE_CERTIFICATION_WINDOW,
    ProspectiveRawFrameJournal,
    _validate_bridge_health,
)
from btc_quarter_hour_engine.market_data.replay import CanonicalReplayAccumulator

try:
    import fcntl
except ImportError:  # pragma: no cover - unavailable on Windows
    fcntl = None


PROTOCOL_VERSION = 1
MAX_HEADER_BYTES = 16 * 1024
MAX_RAW_FRAME_BYTES = 16 * 1024 * 1024


def _utc(value: datetime, *, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return _utc(value, field="timestamp").isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _parse_utc(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO-8601 string")
    try:
        return _utc(datetime.fromisoformat(value.replace("Z", "+00:00")), field=field)
    except ValueError as exc:
        raise ValueError(f"{field} is malformed") from exc


@dataclass(frozen=True, slots=True)
class ClockHealthEvidence:
    observed_at_utc: datetime
    synchronized: bool
    verified: bool
    offset_seconds: float | None
    reference_time_utc: datetime | None
    leap_status: str | None
    source: str
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["observed_at_utc"] = _iso(self.observed_at_utc)
        value["reference_time_utc"] = (
            _iso(self.reference_time_utc) if self.reference_time_utc is not None else None
        )
        return value


def parse_chrony_tracking(
    output: str,
    *,
    observed_at_utc: datetime,
    max_offset_seconds: float = 0.050,
    max_reference_age_seconds: float = 120.0,
) -> ClockHealthEvidence:
    """Evaluate read-only `chronyc tracking` evidence; unknown values fail closed."""
    observed = _utc(observed_at_utc, field="clock observation time")
    if (
        not math.isfinite(max_offset_seconds)
        or max_offset_seconds < 0
        or not math.isfinite(max_reference_age_seconds)
        or max_reference_age_seconds <= 0
    ):
        raise ValueError("Clock-health tolerances must be finite and positive")
    fields = {
        key.strip().lower(): value.strip()
        for key, value in (
            line.split(":", 1) for line in output.splitlines() if ":" in line
        )
    }
    offset: float | None = None
    system_time = fields.get("system time")
    if system_time:
        match = re.match(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s+seconds\b", system_time)
        if match:
            try:
                parsed_offset = float(match.group(1))
                if math.isfinite(parsed_offset):
                    offset = parsed_offset
            except ValueError:
                pass
    reference: datetime | None = None
    reference_value = fields.get("reference time (utc)")
    if reference_value:
        for pattern in ("%a %b %d %H:%M:%S %Y",):
            try:
                reference = datetime.strptime(reference_value, pattern).replace(
                    tzinfo=timezone.utc
                )
                break
            except ValueError:
                continue
        if reference is None:
            try:
                reference = _parse_utc(reference_value, field="chrony reference time")
            except ValueError:
                pass
    leap = fields.get("leap status")
    age = (observed - reference).total_seconds() if reference is not None else None
    synchronized = bool(
        leap is not None
        and leap.casefold() == "normal"
        and offset is not None
        and abs(offset) <= max_offset_seconds
        and age is not None
        and 0 <= age <= max_reference_age_seconds
    )
    reasons: list[str] = []
    if leap is None or leap.casefold() != "normal":
        reasons.append("chrony leap status is not Normal")
    if offset is None:
        reasons.append("chrony system offset is unavailable")
    elif abs(offset) > max_offset_seconds:
        reasons.append("chrony system offset exceeds tolerance")
    if reference is None:
        reasons.append("chrony reference time is unavailable")
    elif age is not None and (age < 0 or age > max_reference_age_seconds):
        reasons.append("chrony reference evidence is stale")
    return ClockHealthEvidence(
        observed_at_utc=observed,
        synchronized=synchronized,
        verified=synchronized,
        offset_seconds=offset,
        reference_time_utc=reference,
        leap_status=leap,
        source="chronyc tracking",
        reason="; ".join(reasons) if reasons else None,
    )


def read_chrony_clock_health(
    *,
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    runner: Callable[..., Any] = subprocess.run,
    max_offset_seconds: float = 0.050,
    max_reference_age_seconds: float = 120.0,
    timeout_seconds: float = 2.0,
) -> ClockHealthEvidence:
    """Read Ubuntu chrony status without changing host time or service settings."""
    observed = _utc(now_fn(), field="clock observation time")
    try:
        result = runner(
            ["chronyc", "tracking"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return ClockHealthEvidence(
            observed, False, False, None, None, None, "chronyc tracking",
            f"chronyc tracking unavailable: {type(exc).__name__}",
        )
    if result.returncode != 0:
        return ClockHealthEvidence(
            observed, False, False, None, None, None, "chronyc tracking",
            f"chronyc tracking exited with status {result.returncode}",
        )
    return parse_chrony_tracking(
        result.stdout,
        observed_at_utc=observed,
        max_offset_seconds=max_offset_seconds,
        max_reference_age_seconds=max_reference_age_seconds,
    )


def _encode_event(header: dict[str, Any], raw_bytes: bytes = b"") -> bytes:
    encoded_header = json.dumps(
        header, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(encoded_header) > MAX_HEADER_BYTES:
        raise ValueError("Source bridge message header exceeds its size limit")
    body = struct.pack(">I", len(encoded_header)) + encoded_header + raw_bytes
    return struct.pack(">Q", len(body)) + body


def _recv_exact(connection: socket.socket, length: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        part = connection.recv(length - len(chunks))
        if not part:
            raise ConnectionError("Source bridge connection ended mid-message")
        chunks.extend(part)
    return bytes(chunks)


def _read_event(connection: socket.socket, *, max_frame_bytes: int) -> tuple[dict[str, Any], bytes]:
    body_size = struct.unpack(">Q", _recv_exact(connection, 8))[0]
    if body_size < 4 or body_size > max_frame_bytes + MAX_HEADER_BYTES + 4:
        raise ValueError("Source bridge message length is invalid or oversized")
    body = _recv_exact(connection, body_size)
    header_size = struct.unpack(">I", body[:4])[0]
    if header_size > MAX_HEADER_BYTES or header_size > len(body) - 4:
        raise ValueError("Source bridge header length is invalid")
    try:
        header = json.loads(body[4 : 4 + header_size])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Source bridge header is malformed JSON") from exc
    if not isinstance(header, dict):
        raise ValueError("Source bridge header must be a JSON object")
    raw_bytes = body[4 + header_size :]
    if header.get("raw_length") != len(raw_bytes):
        raise ValueError("Source bridge raw-frame length mismatch")
    return header, raw_bytes


def _send_ack(connection: socket.socket, value: dict[str, Any]) -> None:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    connection.sendall(struct.pack(">I", len(payload)) + payload)


class UnixSocketFrameTransport:
    """Length-prefixed local stream transport; supports frames up to the WS cap."""

    def __init__(self, path: str | Path, *, timeout_seconds: float = 5.0) -> None:
        self.path = str(path)
        self.timeout_seconds = timeout_seconds

    def send_and_wait_for_ack(self, header: dict[str, Any], raw_bytes: bytes) -> dict[str, Any]:
        encoded = _encode_event(header, raw_bytes)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout_seconds)
            connection.connect(self.path)
            connection.sendall(encoded)
            ack_size = struct.unpack(">I", _recv_exact(connection, 4))[0]
            if ack_size > MAX_HEADER_BYTES:
                raise ValueError("Source bridge acknowledgement is oversized")
            try:
                ack = json.loads(_recv_exact(connection, ack_size))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("Source bridge acknowledgement is malformed") from exc
        if not isinstance(ack, dict) or ack.get("durable") is not True:
            raise ConnectionError("Source bridge did not acknowledge a durable commit")
        return ack


class NonBlockingSourcePublisher:
    """Bounded in-memory tee; collector calls only enqueue, never perform IPC."""

    def __init__(
        self,
        socket_path: str | Path,
        *,
        session_id: str,
        session_started_at_utc: datetime,
        transport: Any | None = None,
        max_queue_frames: int = 64,
        max_queue_bytes: int = 64 * 1024 * 1024,
        max_frame_bytes: int = MAX_RAW_FRAME_BYTES,
        ack_timeout_seconds: float = 5.0,
    ) -> None:
        if not session_id:
            raise ValueError("Source session ID must be nonempty")
        if max_queue_frames < 1 or max_queue_bytes < 1:
            raise ValueError("Source publisher queue limits must be positive")
        if type(max_frame_bytes) is not int or max_frame_bytes < 1:
            raise ValueError("Source frame size limit must be a positive integer")
        self.session_id = session_id
        self.session_started_at_utc = _utc(
            session_started_at_utc, field="source session start"
        )
        self.max_queue_bytes = max_queue_bytes
        self.max_frame_bytes = max_frame_bytes
        self.transport = transport or UnixSocketFrameTransport(
            socket_path, timeout_seconds=ack_timeout_seconds
        )
        self._queue: queue.Queue[tuple[dict[str, Any], bytes, int]] = queue.Queue(
            maxsize=max_queue_frames
        )
        self._lock = threading.Lock()
        self._queued_bytes = 0
        self._high_water_frames = 0
        self._high_water_bytes = 0
        self._failed_reason: str | None = None
        self._closed = False
        self._last_frame_index = -1
        self._last_ingest_time: datetime | None = None
        self._last_connection_id: str | None = None
        self._connection_epoch = -1
        self._connection_first_frame_index = 0
        self._last_fence_utc: datetime | None = None
        self._enqueue_count = 0
        self._frame_enqueue_count = 0
        self._fence_enqueue_count = 0
        self._loss_count = 0
        self._enqueue_latency_total_ms = 0.0
        self._enqueue_latency_max_ms = 0.0
        self._delivery_latency_total_ms = 0.0
        self._delivery_latency_max_ms = 0.0
        self._ack_latency_total_ms = 0.0
        self._ack_latency_max_ms = 0.0
        self._worker = threading.Thread(
            target=self._deliver_loop, name="prospective-source-publisher", daemon=True
        )
        self._worker.start()

    @property
    def valid(self) -> bool:
        with self._lock:
            return self._failed_reason is None

    @property
    def failure_reason(self) -> str | None:
        with self._lock:
            return self._failed_reason

    def _fail_locked(self, reason: str) -> None:
        if self._failed_reason is None:
            self._failed_reason = reason
            self._loss_count += 1

    def _enqueue(self, header: dict[str, Any], raw_bytes: bytes = b"") -> bool:
        started = time.perf_counter_ns()
        size = len(raw_bytes)
        with self._lock:
            if self._closed or self._failed_reason is not None:
                return False
            if size > self.max_frame_bytes:
                self._fail_locked("frame_exceeds_maximum_size")
                return False
            if self._queued_bytes + size > self.max_queue_bytes:
                self._fail_locked("publisher_queue_byte_limit")
                return False
            try:
                self._queue.put_nowait((header, raw_bytes, time.perf_counter_ns()))
            except queue.Full:
                self._fail_locked("publisher_queue_frame_limit")
                return False
            self._queued_bytes += size
            self._high_water_frames = max(self._high_water_frames, self._queue.qsize())
            self._high_water_bytes = max(self._high_water_bytes, self._queued_bytes)
            latency = (time.perf_counter_ns() - started) / 1_000_000
            self._enqueue_count += 1
            self._enqueue_latency_total_ms += latency
            self._enqueue_latency_max_ms = max(self._enqueue_latency_max_ms, latency)
            return True

    def publish_frame(
        self,
        *,
        connection_id: str,
        reconnect_epoch: int,
        frame_index: int,
        raw_bytes: bytes,
        received_at_utc: datetime,
        producer_health: dict[str, Any],
    ) -> bool:
        received = _utc(received_at_utc, field="collector receive timestamp")
        if not isinstance(raw_bytes, bytes):
            raise TypeError("Published raw frame must be immutable bytes")
        if type(frame_index) is not int or frame_index != self._last_frame_index + 1:
            with self._lock:
                self._fail_locked("producer_frame_index_discontinuity")
            return False
        if not connection_id or type(reconnect_epoch) is not int or reconnect_epoch < 0:
            with self._lock:
                self._fail_locked("producer_connection_identity_invalid")
            return False
        if self._last_ingest_time is not None and received < self._last_ingest_time:
            with self._lock:
                self._fail_locked("producer_receive_clock_regressed")
            return False
        if self._last_connection_id != connection_id:
            expected_epoch = self._connection_epoch + 1
            if reconnect_epoch != expected_epoch:
                with self._lock:
                    self._fail_locked("producer_reconnect_epoch_discontinuity")
                return False
            self._connection_epoch = reconnect_epoch
            self._last_connection_id = connection_id
            self._connection_first_frame_index = frame_index
        elif reconnect_epoch != self._connection_epoch:
            with self._lock:
                self._fail_locked("connection_identity_changed_within_epoch")
            return False
        if received < self.session_started_at_utc:
            with self._lock:
                self._fail_locked("frame_precedes_source_session")
            return False
        header = {
            "version": PROTOCOL_VERSION,
            "kind": "frame",
            "session_id": self.session_id,
            "session_started_at_utc": _iso(self.session_started_at_utc),
            "connection_id": connection_id,
            "reconnect_epoch": reconnect_epoch,
            "connection_first_frame_index": self._connection_first_frame_index,
            "frame_index": frame_index,
            "ingest_time_utc": _iso(received),
            "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "raw_length": len(raw_bytes),
            "producer_health": producer_health,
        }
        enqueued = self._enqueue(header, raw_bytes)
        if enqueued:
            self._last_frame_index = frame_index
            self._last_ingest_time = received
            with self._lock:
                self._frame_enqueue_count += 1
            cutoff = received.replace(
                minute=received.minute - received.minute % 15, second=0, microsecond=0
            )
            if cutoff > self.session_started_at_utc and received > cutoff:
                if self._last_fence_utc is None or cutoff > self._last_fence_utc:
                    finalized = (
                        producer_health.get("canonical_boundary_finalized") is True
                        and isinstance(producer_health.get("canonical_boundary"), dict)
                    )
                    if (
                        finalized
                        and received <= cutoff + SOURCE_CERTIFICATION_WINDOW
                    ):
                        self.publish_fence(cutoff, producer_health=producer_health)
        return enqueued

    def publish_fence(
        self, watermark_utc: datetime, *, producer_health: dict[str, Any]
    ) -> bool:
        watermark = _utc(watermark_utc, field="producer fence cutoff")
        if (
            self._last_frame_index < 0
            or self._last_ingest_time is None
            or self._last_ingest_time <= watermark
            or self._last_connection_id is None
        ):
            return False
        if self._last_fence_utc is not None and watermark <= self._last_fence_utc:
            return False
        header = {
            "version": PROTOCOL_VERSION,
            "kind": "fence",
            "session_id": self.session_id,
            "watermark_utc": _iso(watermark),
            "through_frame_index": self._last_frame_index,
            "connection_id": self._last_connection_id,
            "reconnect_epoch": self._connection_epoch,
            "producer_health": producer_health,
            "raw_length": 0,
        }
        if self._enqueue(header):
            self._last_fence_utc = watermark
            with self._lock:
                self._fence_enqueue_count += 1
            return True
        return False

    def _deliver_loop(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                with self._lock:
                    if self._closed:
                        return
                continue
            header, raw_bytes, enqueued_at = item
            try:
                with self._lock:
                    if self._failed_reason is not None:
                        continue
                ack_started = time.perf_counter_ns()
                ack = self.transport.send_and_wait_for_ack(header, raw_bytes)
                expected = (
                    header["frame_index"]
                    if header["kind"] == "frame"
                    else header["through_frame_index"]
                )
                if ack.get("session_id") != self.session_id or ack.get(
                    "through_frame_index"
                ) != expected:
                    raise ValueError("Durable acknowledgement does not match sent source progress")
                ack_latency = (time.perf_counter_ns() - ack_started) / 1_000_000
                delivery_latency = (time.perf_counter_ns() - enqueued_at) / 1_000_000
                with self._lock:
                    self._ack_latency_total_ms += ack_latency
                    self._ack_latency_max_ms = max(self._ack_latency_max_ms, ack_latency)
                    self._delivery_latency_total_ms += delivery_latency
                    self._delivery_latency_max_ms = max(
                        self._delivery_latency_max_ms, delivery_latency
                    )
            except Exception as exc:
                with self._lock:
                    self._fail_locked(f"transport_or_ack_failure:{type(exc).__name__}")
            finally:
                with self._lock:
                    self._queued_bytes = max(0, self._queued_bytes - len(raw_bytes))
                self._queue.task_done()

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            count = self._enqueue_count
            return {
                "valid": self._failed_reason is None,
                "failure_reason": self._failed_reason,
                "enqueue_count": count,
                "frame_enqueue_count": self._frame_enqueue_count,
                "fence_enqueue_count": self._fence_enqueue_count,
                "loss_count": self._loss_count,
                "queue_depth": self._queue.qsize(),
                "queue_bytes": self._queued_bytes,
                "queue_high_water_frames": self._high_water_frames,
                "queue_high_water_bytes": self._high_water_bytes,
                "enqueue_latency_mean_ms": (
                    self._enqueue_latency_total_ms / count if count else 0.0
                ),
                "enqueue_latency_max_ms": self._enqueue_latency_max_ms,
                "delivery_latency_mean_ms": (
                    self._delivery_latency_total_ms / count if count else 0.0
                ),
                "delivery_latency_max_ms": self._delivery_latency_max_ms,
                "durable_ack_latency_mean_ms": (
                    self._ack_latency_total_ms / count if count else 0.0
                ),
                "durable_ack_latency_max_ms": self._ack_latency_max_ms,
            }

    def close(self, *, timeout_seconds: float = 1.0) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._worker.join(timeout=max(0.0, timeout_seconds))
        if self._worker.is_alive():
            with self._lock:
                self._fail_locked("publisher_shutdown_timeout")


class ProspectiveSourceReceiver:
    """Durably receive a single local publisher stream into the isolated journal."""

    def __init__(
        self,
        journal: ProspectiveRawFrameJournal,
        *,
        clock_health_provider: Callable[[], ClockHealthEvidence] = read_chrony_clock_health,
        max_frame_bytes: int = MAX_RAW_FRAME_BYTES,
        checkpoint_interval_frames: int = MAX_CANONICAL_CHECKPOINT_TAIL_FRAMES,
    ) -> None:
        if type(max_frame_bytes) is not int or max_frame_bytes < 1:
            raise ValueError("Source frame size limit must be a positive integer")
        if (
            type(checkpoint_interval_frames) is not int
            or not 1 <= checkpoint_interval_frames <= MAX_CANONICAL_CHECKPOINT_TAIL_FRAMES
        ):
            raise ValueError("Canonical checkpoint interval exceeds the bounded tail limit")
        self.journal = journal
        self.clock_health_provider = clock_health_provider
        self.max_frame_bytes = max_frame_bytes
        self.checkpoint_interval_frames = checkpoint_interval_frames
        self.frames_committed = 0
        self.frames_rejected = 0
        self.gaps = 0
        self.fences_committed = 0
        self.frame_write_latency_max_ms = 0.0
        self.durable_ack_latency_max_ms = 0.0
        self.watermark_latency_max_ms = 0.0
        self.canonical_checkpoint_load_latency_max_ms = 0.0
        self.canonical_checkpoint_bytes_max = 0
        self.canonical_checkpoints_committed = 0
        self.canonical_tail_frames_recovered = 0
        self.canonical_tail_bytes_recovered = 0
        self._session_started: dict[str, datetime] = {}
        self._canonical_replays: dict[str, CanonicalReplayAccumulator] = {}
        self._canonical_timeouts: dict[str, float] = {}
        self._canonical_checkpoint_indices: dict[str, int] = {}
        self._canonical_tail_bytes: dict[str, int] = {}

    def _canonical_replay(
        self, session_id: str, heartbeat_timeout_seconds: float
    ) -> CanonicalReplayAccumulator:
        timeout = float(heartbeat_timeout_seconds)
        replay = self._canonical_replays.get(session_id)
        if replay is not None:
            if self._canonical_timeouts[session_id] != timeout:
                raise ValueError("Source canonical replay heartbeat tolerance changed")
            return replay

        started_at = time.perf_counter_ns()
        started = self.journal.session_started_at(session_id)
        if started is None:
            raise ValueError("Canonical source replay session is not registered")
        latest = self.journal._connection.execute(
            "SELECT f.frame_index, p.commitment_sha256 "
            "FROM source_frames f LEFT JOIN source_prefix_commitments p "
            "ON p.session_id = f.session_id AND p.frame_index = f.frame_index "
            "WHERE f.session_id = ? ORDER BY f.frame_index DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        if latest is None:
            replay = CanonicalReplayAccumulator(
                product_id="BTC-USD",
                session_id=session_id,
                heartbeat_timeout_seconds=timeout,
            )
            self._canonical_checkpoint_indices[session_id] = -1
            self._canonical_tail_bytes[session_id] = 0
        else:
            replay = self.journal._validate_latest_canonical_checkpoint(session_id)
            if (
                replay.heartbeat_timeout_seconds != timeout
            ):
                raise ValueError("Source canonical replay heartbeat tolerance changed")
            anchor, tail_bytes = self.journal._canonical_tail_extent(session_id, latest["frame_index"])
            self._canonical_checkpoint_indices[session_id] = anchor
            self._canonical_tail_bytes[session_id] = tail_bytes
            self.canonical_tail_frames_recovered += latest["frame_index"] - anchor
            self.canonical_tail_bytes_recovered += tail_bytes
            checkpoint_size = self.journal._connection.execute(
                "SELECT length(checkpoint_json) FROM source_canonical_checkpoints WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]
            self.canonical_checkpoint_bytes_max = max(
                self.canonical_checkpoint_bytes_max, checkpoint_size,
            )
        self._canonical_replays[session_id] = replay
        self._canonical_timeouts[session_id] = timeout
        self.canonical_checkpoint_load_latency_max_ms = max(
            self.canonical_checkpoint_load_latency_max_ms,
            (time.perf_counter_ns() - started_at) / 1_000_000,
        )
        return replay

    def _verify_canonical_boundary(
        self,
        *,
        session_id: str,
        watermark: datetime,
        frame_health: dict[str, Any],
    ) -> None:
        canonical = frame_health["canonical_boundary"]
        replay = self._canonical_replay(
            session_id,
            frame_health["post_frame_health"]["heartbeat_timeout_seconds"],
        )
        if not replay.boundary_is_finalized(watermark):
            raise ValueError("Canonical boundary was not finalized by committed raw frames")
        if replay.frame_count < 1:
            raise ValueError("Canonical source replay has no committed frame prefix")
        observations = [
            replay.boundary_observation(
                watermark,
                derived_at_utc=watermark + timedelta(minutes=15),
                connections=[
                    {"connection_id": connection_id}
                    for connection_id in replay._connections
                ],
            )
        ]
        observation = next(
            (item for item in observations if item.boundary_time_utc == watermark),
            None,
        )
        if observation is None:
            raise ValueError("Canonical replay cannot establish the fenced boundary")
        expected = {
            "boundary_utc": _iso(observation.boundary_time_utc),
            "canonical_target_eligible": observation.canonical_target_eligible,
            "eligibility_reason": observation.eligibility_reason,
            "book_synced": observation.book_synced,
            "last_heartbeat_at_utc": (
                _iso(observation.last_heartbeat_at_utc)
                if observation.last_heartbeat_at_utc is not None
                else None
            ),
            "heartbeat_timeout_seconds": float(
                observation.heartbeat_timeout_seconds
            ),
            "connection_id": observation.connection_id,
            "source_sequence_num": observation.source_sequence_num,
        }
        if canonical != expected:
            raise ValueError("Producer canonical boundary disagrees with raw-frame replay")

    def process(self, header: dict[str, Any], raw_bytes: bytes) -> dict[str, Any]:
        started = time.perf_counter_ns()
        try:
            ack = self._process(header, raw_bytes)
        except Exception as exc:
            self.frames_rejected += 1
            if any(
                marker in str(exc).casefold()
                for marker in (
                    "frame index",
                    "frame prefix",
                    "reconnect epoch",
                    "sequence",
                    "durable frame prefix",
                )
            ):
                self.gaps += 1
            raise
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        self.durable_ack_latency_max_ms = max(self.durable_ack_latency_max_ms, elapsed)
        return ack

    def _process(self, header: dict[str, Any], raw_bytes: bytes) -> dict[str, Any]:
        if header.get("version") != PROTOCOL_VERSION:
            raise ValueError("Unsupported source bridge protocol version")
        session_id = header.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Source bridge session identity is missing")
        if header.get("raw_length") != len(raw_bytes):
            raise ValueError("Source bridge raw length mismatch")
        kind = header.get("kind")
        if kind == "frame":
            return self._process_frame(header, raw_bytes)
        if kind == "fence" and not raw_bytes:
            return self._process_fence(header)
        raise ValueError("Unsupported source bridge message kind")

    def _process_frame(self, header: dict[str, Any], raw_bytes: bytes) -> dict[str, Any]:
        if len(raw_bytes) > self.max_frame_bytes:
            raise ValueError("Source frame exceeds configured maximum")
        session_id = header["session_id"]
        connection_id = header.get("connection_id")
        reconnect_epoch = header.get("reconnect_epoch")
        frame_index = header.get("frame_index")
        first_frame = header.get("connection_first_frame_index")
        producer_health = header.get("producer_health")
        if not isinstance(connection_id, str) or not connection_id:
            raise ValueError("Source frame connection identity is missing")
        if (
            type(reconnect_epoch) is not int
            or reconnect_epoch < 0
            or type(frame_index) is not int
            or frame_index < 0
            or type(first_frame) is not int
            or first_frame < 0
            or not isinstance(producer_health, dict)
            or producer_health.get("publisher_valid") is not True
        ):
            raise ValueError("Source frame sequence metadata is invalid")
        ingest = _parse_utc(header.get("ingest_time_utc"), field="frame receive timestamp")
        canonical_boundary = producer_health.get("canonical_boundary")
        health_boundary = (
            _parse_utc(
                canonical_boundary["boundary_utc"],
                field="canonical producer-health boundary",
            )
            if isinstance(canonical_boundary, dict)
            and isinstance(canonical_boundary.get("boundary_utc"), str)
            else ingest.replace(
                minute=ingest.minute - ingest.minute % 15, second=0, microsecond=0
            )
        )
        _validate_bridge_health(
            producer_health,
            boundary=health_boundary,
            connection_id=connection_id,
        )
        started = _parse_utc(
            header.get("session_started_at_utc"), field="source session start"
        )
        digest = hashlib.sha256(raw_bytes).hexdigest()
        if header.get("raw_sha256") != digest:
            raise ValueError("Source frame digest mismatch")
        if session_id not in self._session_started:
            persisted_start = self.journal.session_started_at(session_id)
            if persisted_start is None:
                self.journal.start_session(
                    session_id, started, source_mode="bridge"
                )
                persisted_start = started
            if persisted_start != started:
                raise ValueError("Source session start identity changed")
            if self.journal.session_mode(session_id) != "bridge":
                raise ValueError("Source session was not registered as bridge-backed")
            self._session_started[session_id] = persisted_start
        elif (
            self._session_started[session_id] != started
            or self.journal.session_mode(session_id) != "bridge"
        ):
            raise ValueError("Source session identity or mode changed")
        self.journal.register_connection_epoch(
            session_id=session_id,
            reconnect_epoch=reconnect_epoch,
            connection_id=connection_id,
            first_frame_index=first_frame,
        )
        replay = self._canonical_replay(
            session_id,
            producer_health["post_frame_health"]["heartbeat_timeout_seconds"],
        )
        if replay.frame_count != frame_index:
            raise ValueError("Canonical replay checkpoint frame index is discontinuous")
        try:
            replay.consume(
                raw_bytes,
                connection_id,
                frame_index,
                ingest,
                enforce_live_limits=True,
            )
            replay._require_bounded_live_state()
            checkpoint_due = (
                self._canonical_checkpoint_indices[session_id] < 0
                or frame_index - self._canonical_checkpoint_indices[session_id] >= self.checkpoint_interval_frames
                or self._canonical_tail_bytes[session_id] + len(raw_bytes) > MAX_CANONICAL_CHECKPOINT_TAIL_BYTES
            )
            canonical_checkpoint = replay.to_checkpoint() if checkpoint_due else None
        except Exception:
            self._canonical_replays.pop(session_id, None)
            self._canonical_timeouts.pop(session_id, None)
            raise
        write_started = time.perf_counter_ns()
        try:
            self.journal.append_frame(
                session_id=session_id,
                connection_id=connection_id,
                frame_index=frame_index,
                ingest_time_utc=ingest,
                raw_bytes=raw_bytes,
                reconnect_epoch=reconnect_epoch,
                producer_health=producer_health,
                canonical_replay_checkpoint=canonical_checkpoint,
                canonical_checkpoint_tail=not checkpoint_due,
            )
        except Exception:
            self._canonical_replays.pop(session_id, None)
            self._canonical_timeouts.pop(session_id, None)
            raise
        if checkpoint_due:
            self._canonical_checkpoint_indices[session_id] = frame_index
            self._canonical_tail_bytes[session_id] = 0
            self.canonical_checkpoints_committed += 1
        else:
            self._canonical_tail_bytes[session_id] += len(raw_bytes)
        write_latency = (time.perf_counter_ns() - write_started) / 1_000_000
        self.frame_write_latency_max_ms = max(
            self.frame_write_latency_max_ms, write_latency
        )
        self.frames_committed += 1
        return {
            "durable": True,
            "kind": "frame",
            "session_id": session_id,
            "through_frame_index": frame_index,
            "raw_sha256": digest,
        }

    def _process_fence(self, header: dict[str, Any]) -> dict[str, Any]:
        session_id = header["session_id"]
        watermark = _parse_utc(header.get("watermark_utc"), field="producer fence cutoff")
        frame_index = header.get("through_frame_index")
        reconnect_epoch = header.get("reconnect_epoch")
        connection_id = header.get("connection_id")
        health = header.get("producer_health")
        if (
            type(frame_index) is not int
            or frame_index < 0
            or type(reconnect_epoch) is not int
            or reconnect_epoch < 0
            or not isinstance(connection_id, str)
            or not isinstance(health, dict)
            or health.get("publisher_valid") is not True
        ):
            raise ValueError("Producer fence metadata is invalid or unhealthy")
        validated_health = _validate_bridge_health(
            health,
            boundary=watermark,
            connection_id=connection_id,
        )
        latest = self.journal._connection.execute(
            "SELECT frame_index FROM source_frames WHERE session_id = ? "
            "ORDER BY frame_index DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        if latest is None or latest["frame_index"] != frame_index:
            self.gaps += 1
            raise ValueError("Producer fence arrived before its complete durable frame prefix")
        if (
            validated_health["canonical_boundary_finalized"] is not True
            or validated_health["canonical_boundary"] is None
        ):
            raise ValueError("Producer fence lacks finalized canonical boundary evidence")
        self._verify_canonical_boundary(
            session_id=session_id,
            watermark=watermark,
            frame_health=validated_health,
        )
        replay = self._canonical_replays[session_id]
        # Pruning replaces only bounded metadata. Share the read-only live book
        # until the fence commits, rather than copying every populated level.
        pruned_replay = copy.copy(replay)
        pruned_replay._connections = {
            key: copy.copy(state) for key, state in replay._connections.items()
        }
        pruned_replay.prune_for_next_boundary(watermark)
        canonical_checkpoint = pruned_replay.to_checkpoint()
        clock = self.clock_health_provider()
        if not isinstance(clock, ClockHealthEvidence):
            raise TypeError("Clock-health provider returned invalid evidence")
        fence_started = time.perf_counter_ns()
        self.journal.advance_fenced_watermark(
            session_id,
            watermark,
            through_frame_index=frame_index,
            connection_id=connection_id,
            reconnect_epoch=reconnect_epoch,
            clock_health_verified=clock.verified,
            clock_evidence=clock.as_dict(),
            producer_health=health,
            canonical_replay_checkpoint=canonical_checkpoint,
        )
        self._canonical_replays[session_id] = pruned_replay
        self._canonical_checkpoint_indices[session_id] = frame_index
        self._canonical_tail_bytes[session_id] = 0
        self.canonical_checkpoints_committed += 1
        fence_latency = (time.perf_counter_ns() - fence_started) / 1_000_000
        self.watermark_latency_max_ms = max(
            self.watermark_latency_max_ms, fence_latency
        )
        self.fences_committed += 1
        return {
            "durable": True,
            "kind": "fence",
            "session_id": session_id,
            "through_frame_index": frame_index,
            "watermark_utc": _iso(watermark),
            "clock_health_verified": clock.verified,
        }

    def metrics(self) -> dict[str, Any]:
        return {
            "frames_committed": self.frames_committed,
            "frames_rejected": self.frames_rejected,
            "source_gaps": self.gaps,
            "fences_committed": self.fences_committed,
            "frame_write_latency_max_ms": self.frame_write_latency_max_ms,
            "durable_ack_latency_max_ms": self.durable_ack_latency_max_ms,
            "watermark_certification_latency_max_ms": self.watermark_latency_max_ms,
            "canonical_checkpoint_load_latency_max_ms": self.canonical_checkpoint_load_latency_max_ms,
            "canonical_checkpoint_bytes_max": self.canonical_checkpoint_bytes_max,
            "canonical_checkpoints_committed": self.canonical_checkpoints_committed,
            "canonical_tail_frames_recovered": self.canonical_tail_frames_recovered,
            "canonical_tail_bytes_recovered": self.canonical_tail_bytes_recovered,
        }

    def note_rejection(self, *, gap: bool = False) -> None:
        self.frames_rejected += 1
        if gap:
            self.gaps += 1


class ProspectiveSourceSocketServer:
    """Owner-local Unix-domain receiver; socket permissions are restricted to 0600."""

    def __init__(
        self,
        socket_path: str | Path,
        receiver: ProspectiveSourceReceiver,
        *,
        backlog: int = 8,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.receiver = receiver
        self.backlog = backlog
        self._listener: socket.socket | None = None
        self._stop = threading.Event()
        self._lock_fd: int | None = None
        self._bound_socket_identity: tuple[int, int] | None = None

    def _acquire_startup_lock(self) -> None:
        if os.name != "posix" or fcntl is None or not hasattr(os, "O_NOFOLLOW"):
            raise OSError("Safe Unix socket recovery is unsupported on this platform")
        lock_path = self.socket_path.with_name(self.socket_path.name + ".lock")
        flags = os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW
        try:
            fd = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            raise OSError("Unable to safely open receiver startup lock") from exc
        try:
            metadata = os.fstat(fd)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
            ):
                raise PermissionError("Receiver startup lock has an untrusted owner or type")
            os.fchmod(fd, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(
                    "Another receiver startup or active receiver owns the socket lock"
                ) from exc
            self._lock_fd = fd
        except BaseException:
            os.close(fd)
            raise

    def _remove_stale_socket_if_safe(self) -> None:
        try:
            before = self.socket_path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(before.st_mode):
            raise FileExistsError(
                f"Refusing to replace non-socket receiver path: {self.socket_path}"
            )
        if before.st_uid != os.geteuid():
            raise PermissionError(
                f"Refusing to replace socket owned by another user: {self.socket_path}"
            )
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.settimeout(1.0)
            probe.connect(str(self.socket_path))
        except OSError as exc:
            if exc.errno != errno.ECONNREFUSED:
                raise RuntimeError(
                    "Unable to establish whether the existing receiver socket is stale"
                ) from exc
        else:
            raise RuntimeError(
                f"Refusing to replace active receiver socket: {self.socket_path}"
            )
        finally:
            probe.close()
        try:
            after = self.socket_path.lstat()
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Receiver socket changed during stale-path verification"
            ) from exc
        if (
            after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or not stat.S_ISSOCK(after.st_mode)
            or after.st_uid != os.geteuid()
        ):
            raise RuntimeError(
                "Receiver socket changed during stale-path verification"
            )
        self.socket_path.unlink()

    def serve_forever(self) -> None:
        if self._listener is not None:
            raise RuntimeError("Source bridge server is already running")
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self._acquire_startup_lock()
        listener: socket.socket | None = None
        try:
            self._remove_stale_socket_if_safe()
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(self.socket_path))
            os.chmod(self.socket_path, 0o600)
            metadata = self.socket_path.lstat()
            if (
                not stat.S_ISSOCK(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
            ):
                raise PermissionError("Bound receiver socket has unexpected owner or permissions")
            self._bound_socket_identity = (metadata.st_dev, metadata.st_ino)
            self._listener = listener
            listener.listen(self.backlog)
            listener.settimeout(0.5)
            while not self._stop.is_set():
                try:
                    connection, _ = listener.accept()
                except TimeoutError:
                    continue
                except OSError:
                    if self._stop.is_set():
                        break
                    raise
                with connection:
                    connection.settimeout(10.0)
                    try:
                        header, raw_bytes = _read_event(
                            connection, max_frame_bytes=self.receiver.max_frame_bytes
                        )
                        ack = self.receiver.process(header, raw_bytes)
                        _send_ack(connection, ack)
                    except (ConnectionError, OSError, ValueError, TypeError):
                        self.receiver.note_rejection()
        finally:
            if listener is not None:
                listener.close()
            self._listener = None
            identity = self._bound_socket_identity
            self._bound_socket_identity = None
            try:
                metadata = self.socket_path.lstat()
            except FileNotFoundError:
                metadata = None
            if (
                identity is not None
                and metadata is not None
                and (metadata.st_dev, metadata.st_ino) == identity
                and stat.S_ISSOCK(metadata.st_mode)
                and metadata.st_uid == os.geteuid()
            ):
                self.socket_path.unlink()
            if self._lock_fd is not None:
                os.close(self._lock_fd)
                self._lock_fd = None

    def stop(self) -> None:
        self._stop.set()
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass


__all__ = [
    "ClockHealthEvidence",
    "MAX_RAW_FRAME_BYTES",
    "NonBlockingSourcePublisher",
    "PROTOCOL_VERSION",
    "ProspectiveSourceReceiver",
    "ProspectiveSourceSocketServer",
    "UnixSocketFrameTransport",
    "parse_chrony_tracking",
    "read_chrony_clock_health",
]
