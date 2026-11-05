"""Deadline-aware orchestration for the frozen prospective inference pipeline.

The module is intentionally inert on import.  Creating a runner does not start
a worker, modify a source journal, or open a ledger.  Official execution is an
explicit caller action and requires the official model loader, a configured
authorization identifier, a bridge-backed journal, and independently checked
chrony evidence.
"""
from __future__ import annotations

import math
import os
import sqlite3
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, TypeVar

from btc_quarter_hour_engine.acquisition.prospective_source_bridge import (
    ClockHealthEvidence,
    read_chrony_clock_health,
)
from btc_quarter_hour_engine.research.prospective_inference import (
    FrozenModelBundle,
    is_verified_official_bundle,
    load_official_models,
    process_incremental_boundary,
)
from btc_quarter_hour_engine.research.prospective_ledger import (
    ClockHealth,
    ClockHealthError,
    DuplicatePredictionError,
    LatePredictionError,
    PREDICTION_DEADLINE_SECONDS,
    ProspectiveLedger,
    WINDOW_END_EXCLUSIVE,
    WINDOW_START,
)
from btc_quarter_hour_engine.research.prospective_replay import (
    FeatureReplayCheckpoint,
    FeatureReplayWarmup,
    IncrementalRawFrameSnapshot,
    accumulate_feature_replay_warmup,
    advance_feature_replay_checkpoint,
    finish_feature_replay_warmup,
    initial_prefix_commitment,
)
from btc_quarter_hour_engine.research.prospective_source import (
    ProspectiveRawFrameJournal,
    _require_bridge_prediction_health,
)

_ReadResult = TypeVar("_ReadResult")

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

try:  # Python's Windows locking API is required when the package runs on Windows.
    import msvcrt
except ImportError:  # pragma: no cover - platform-specific
    msvcrt = None


class Clock(Protocol):
    def now(self) -> datetime: ...


class Waiter(Protocol):
    def wait(self, seconds: float) -> None: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class SystemWaiter:
    def wait(self, seconds: float) -> None:
        time.sleep(seconds)


@dataclass(frozen=True)
class RunnerConfig:
    source_journal_path: Path
    checkpoint_directory: Path
    prediction_ledger_path: Path
    official_model_directory: Path
    implementation_git_sha: str
    expected_source_session_id: str
    operational_authorization_id: str
    source_poll_seconds: float = 0.25
    max_tail_frames: int = 100_000
    warmup_chunk_frames: int = 10_000

    def validate(self, *, require_existing: bool = True) -> None:
        if any(
            not Path(value).is_absolute()
            for value in (
                self.source_journal_path,
                self.checkpoint_directory,
                self.prediction_ledger_path,
                self.official_model_directory,
            )
        ):
            raise ValueError("Source, checkpoint, ledger, and model paths must be absolute")
        if (
            Path(self.source_journal_path).is_symlink()
            or Path(self.checkpoint_directory).is_symlink()
            or Path(self.prediction_ledger_path).is_symlink()
            or Path(self.official_model_directory).is_symlink()
        ):
            raise ValueError("Configured source, checkpoint, ledger, and model paths must not be symlinks")
        paths = {
            "source": Path(self.source_journal_path).resolve(),
            "checkpoints": Path(self.checkpoint_directory).resolve(),
            "ledger": Path(self.prediction_ledger_path).resolve(),
            "models": Path(self.official_model_directory).resolve(),
        }
        if len(set(paths.values())) != len(paths):
            raise ValueError("Source, checkpoint, ledger, and model paths must be distinct")
        for directory_name in ("checkpoints", "models"):
            directory = paths[directory_name]
            for file_name in ("source", "ledger"):
                if paths[file_name].is_relative_to(directory):
                    raise ValueError(f"{file_name} path cannot be inside {directory_name} directory")
        if paths["models"].is_relative_to(paths["checkpoints"]) or paths["checkpoints"].is_relative_to(paths["models"]):
            raise ValueError("Checkpoint and model directories cannot contain one another")
        if not isinstance(self.implementation_git_sha, str) or (
            len(self.implementation_git_sha) != 40
            or any(character not in "0123456789abcdef" for character in self.implementation_git_sha)
        ):
            raise ValueError("Implementation revision must be a full lowercase Git SHA")
        if not isinstance(self.expected_source_session_id, str) or not self.expected_source_session_id:
            raise ValueError("An explicit expected bridge source-session ID is required")
        if not isinstance(self.operational_authorization_id, str) or not self.operational_authorization_id.strip():
            raise ValueError("Explicit operational authorization is required for official mode")
        if (
            isinstance(self.source_poll_seconds, bool)
            or not isinstance(self.source_poll_seconds, (int, float))
            or not math.isfinite(self.source_poll_seconds)
            or not 0.01 <= self.source_poll_seconds <= 5
        ):
            raise ValueError("Source polling interval must be between 0.01 and 5 seconds")
        if type(self.max_tail_frames) is not int or not 1 <= self.max_tail_frames <= 100_000:
            raise ValueError("Maximum raw tail must be within the replay hard limit")
        if type(self.warmup_chunk_frames) is not int or not 1 <= self.warmup_chunk_frames <= 100_000:
            raise ValueError("Warm-up chunk size must be within the replay hard limit")
        if require_existing:
            if not paths["source"].is_file():
                raise FileNotFoundError(paths["source"])
            if not paths["checkpoints"].is_dir():
                raise FileNotFoundError(paths["checkpoints"])
            if not paths["models"].is_dir():
                raise FileNotFoundError(paths["models"])
            if paths["source"] == paths["ledger"]:
                raise ValueError("Source journal and prediction ledger cannot share a file")


@dataclass
class BoundaryResult:
    boundary_utc: datetime
    status: str
    reason: str | None = None
    evidence: dict[str, Any] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        safe_evidence = None
        if self.evidence is not None:
            safe_evidence = {
                key: self.evidence[key]
                for key in (
                    "boundary_utc",
                    "target_utc",
                    "source_session_id",
                    "source_frame_count",
                    "feature_fingerprint",
                    "generated_at_utc",
                    "persisted_at_utc",
                    "timing_eligible",
                )
                if key in self.evidence
            }
        return {
            "boundary_utc": _format_value(self.boundary_utc),
            "status": self.status,
            "reason": self.reason,
            "evidence_summary": _format_value(safe_evidence),
            "metrics": _format_value(self.metrics),
        }


class BoundaryScheduler:
    """UTC quarter-hour scheduler with injectable time and blocking wait."""

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        waiter: Waiter | None = None,
        poll_seconds: float = 0.25,
    ) -> None:
        if not math.isfinite(poll_seconds) or poll_seconds <= 0:
            raise ValueError("Scheduler poll interval must be positive and finite")
        self.clock = clock or SystemClock()
        self.waiter = waiter or SystemWaiter()
        self.poll_seconds = poll_seconds

    @staticmethod
    def most_recent_boundary(now: datetime) -> datetime:
        instant = _utc(now, "scheduler time")
        return instant.replace(
            minute=instant.minute - instant.minute % 15,
            second=0,
            microsecond=0,
        )

    @staticmethod
    def next_boundary(now: datetime) -> datetime:
        instant = _utc(now, "scheduler time")
        minute_floor = BoundaryScheduler.most_recent_boundary(instant)
        return (
            minute_floor
            if instant == minute_floor
            else minute_floor + timedelta(minutes=15)
        )

    @staticmethod
    def is_open(boundary: datetime, now: datetime) -> bool:
        scheduled = _utc(boundary, "scheduled boundary")
        instant = _utc(now, "scheduler time")
        return scheduled <= instant <= scheduled + timedelta(seconds=PREDICTION_DEADLINE_SECONDS)

    def wait_for_boundary(
        self, boundary: datetime, *, should_stop: Callable[[], bool] | None = None
    ) -> bool:
        """Block without spinning; return False if its deadline has already passed."""
        scheduled = _utc(boundary, "scheduled boundary")
        while True:
            if should_stop is not None and should_stop():
                return False
            now = _utc(self.clock.now(), "scheduler clock")
            if now >= scheduled:
                return self.is_open(scheduled, now)
            self.waiter.wait(min(self.poll_seconds, (scheduled - now).total_seconds()))


class SingleWriterGuard:
    """Nonblocking process lock scoped to one official prediction ledger."""

    def __init__(self, ledger_path: Path) -> None:
        resolved = Path(ledger_path).resolve()
        self.path = resolved.with_suffix(resolved.suffix + ".runner.lock")
        self._fd: int | None = None

    def __enter__(self) -> "SingleWriterGuard":
        self._fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            if os.name == "nt":  # pragma: no cover - Windows CI
                assert msvcrt is not None
                if os.fstat(self._fd).st_size == 0:
                    os.write(self._fd, b"\0")
                    os.lseek(self._fd, 0, os.SEEK_SET)
                msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
            else:
                assert fcntl is not None
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            os.close(self._fd)
            self._fd = None
            raise RuntimeError("Another prospective runner owns this ledger") from exc
        return self

    def __exit__(self, *_: object) -> None:
        if self._fd is None:
            return
        try:
            if os.name == "nt":  # pragma: no cover - Windows CI
                assert msvcrt is not None
                os.lseek(self._fd, 0, os.SEEK_SET)
                msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
            else:
                assert fcntl is not None
                fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None


def _utc(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _filename(kind: str, checkpoint: FeatureReplayCheckpoint) -> str:
    digest = checkpoint.data["integrity_sha256"]
    cutoff = checkpoint.cutoff_utc.strftime("%Y%m%dT%H%M%SZ")
    session_key = hashlib.sha256(checkpoint.session_id.encode("utf-8")).hexdigest()[:16]
    return (
        f"{kind}_{session_key}_{cutoff}_{checkpoint.last_frame_index:012d}_"
        f"{digest}.json"
    )


def _save_checkpoint(directory: Path, checkpoint: FeatureReplayCheckpoint, *, warmup: bool = False) -> Path:
    destination = directory / _filename("warmup" if warmup else "checkpoint", checkpoint)
    cls = FeatureReplayWarmup if warmup else FeatureReplayCheckpoint
    if not destination.exists():
        try:
            checkpoint.write_once(destination)
        except OSError:
            # A post-link failure may leave the exact immutable destination.
            # Re-attest that file and parent rather than replacing evidence.
            if not destination.exists():
                raise
    # Read-after-publish (or recovery) validates file contents and directory entry.
    durable = cls.load_durable(destination)
    if durable.serialized != checkpoint.serialized:
        raise ValueError("Published checkpoint differs from requested immutable evidence")
    return destination


class ProspectiveInferenceRunner:
    """One-boundary runner; callers own the explicit service loop and shutdown."""

    def __init__(
        self,
        config: RunnerConfig,
        *,
        ledger: ProspectiveLedger | None,
        journal: ProspectiveRawFrameJournal,
        model_bundle: FrozenModelBundle | None = None,
        clock: Clock | None = None,
        waiter: Waiter | None = None,
        clock_evidence_provider: Callable[[], ClockHealthEvidence] = read_chrony_clock_health,
        official_mode: bool = True,
    ) -> None:
        config.validate()
        if official_mode and clock is not None and type(clock) is not SystemClock:
            raise ValueError("Injected clocks are unavailable in official operation")
        if official_mode and waiter is not None and type(waiter) is not SystemWaiter:
            raise ValueError("Injected waiters are unavailable in official operation")
        if official_mode and clock_evidence_provider is not read_chrony_clock_health:
            raise ValueError("Injected UTC clock evidence is unavailable in official operation")
        self.config = config
        self.ledger = ledger
        self._preparation_clock_health = ledger.clock_health if ledger is not None else ClockHealth()
        self.journal = journal
        self.model_bundle = model_bundle
        self.clock_evidence_provider = clock_evidence_provider
        self.official_mode = official_mode
        self.scheduler = BoundaryScheduler(
            clock=clock,
            waiter=waiter,
            poll_seconds=config.source_poll_seconds,
        )
        self._stopping = False
        self._writer_lock_held = False
        self.startup_validation_ms = 0.0
        self._last_utc_evidence: ClockHealthEvidence | None = None
        self._last_utc_evidence_check_monotonic = float("-inf")

    @classmethod
    def from_config(
        cls,
        config: RunnerConfig,
        *,
        ledger: ProspectiveLedger,
        official_mode: bool = True,
    ) -> "ProspectiveInferenceRunner":
        config.validate()
        startup_started = time.perf_counter()
        journal = ProspectiveRawFrameJournal.open_read_only(config.source_journal_path)
        try:
            bundle = load_official_models(config.official_model_directory)
            runner = cls(
                config,
                ledger=ledger,
                journal=journal,
                model_bundle=bundle,
                official_mode=official_mode,
            )
            runner.startup_validation_ms = (
                time.perf_counter() - startup_started
            ) * 1000
            return runner
        except BaseException:
            journal.close()
            raise

    def close(self) -> None:
        """Idempotent local shutdown; no source receiver or system service is stopped."""
        self._stopping = True
        self.journal.close()

    def stop(self) -> None:
        self._stopping = True

    def validate_runtime(self) -> ClockHealthEvidence:
        if not self.official_mode:
            raise ValueError("Runtime validation is only defined for explicitly authorized official mode")
        if self.ledger is None:
            raise ValueError("Prediction operation requires an immutable ledger")
        if not self.config.operational_authorization_id.strip():
            raise ValueError("Official inference lacks explicit operational authorization")
        if os.environ.get("BTC_QH_PROSPECTIVE_AUTHORIZATION_ID") != self.config.operational_authorization_id:
            raise ValueError("Independent deployment authorization is absent or does not match")
        if os.environ.get("BTC_QH_IMPLEMENTATION_GIT_SHA") != self.config.implementation_git_sha:
            raise ValueError("Running implementation SHA is absent or does not match the reviewed configuration")
        self.ledger.clock_health.now()
        if (
            self.ledger.window_start != WINDOW_START
            or self.ledger.window_end_exclusive != WINDOW_END_EXCLUSIVE
            or self.ledger.scheduled_count != 2880
        ):
            raise ValueError("Prediction ledger does not match the frozen 2,880-boundary schedule")
        if (
            type(self.ledger.clock_health) is not ClockHealth
            or not self.ledger.clock_health.uses_system_clock
        ):
            raise ValueError("Injected ledger clocks are unavailable in official operation")
        if self.model_bundle is None or not is_verified_official_bundle(self.model_bundle):
            raise ValueError("Official models must be loaded and verified by load_official_models()")
        if self.journal.session_mode(self.config.expected_source_session_id) != "bridge":
            raise ValueError("Official inference requires a bridge-backed source session")
        if self.journal.session_started_at(self.config.expected_source_session_id) is None:
            raise ValueError("Configured bridge source session is not registered")
        return self._require_fresh_utc_evidence(force=True)

    def _require_fresh_utc_evidence(self, *, force: bool = False) -> ClockHealthEvidence:
        monotonic = time.monotonic()
        if (
            not force
            and self._last_utc_evidence is not None
            and monotonic - self._last_utc_evidence_check_monotonic < 10.0
        ):
            evidence = self._last_utc_evidence
        else:
            evidence = self.clock_evidence_provider()
            self._last_utc_evidence = evidence
            self._last_utc_evidence_check_monotonic = monotonic
        now = _utc(self.scheduler.clock.now(), "current UTC time")
        if (
            not isinstance(evidence, ClockHealthEvidence)
            or evidence.verified is not True
            or evidence.synchronized is not True
            or evidence.reference_time_utc is None
            or evidence.leap_status != "Normal"
            or evidence.offset_seconds is None
            or not math.isfinite(evidence.offset_seconds)
            or abs(evidence.offset_seconds) > 0.050
            or not 0 <= (now - _utc(evidence.observed_at_utc, "chrony observation")).total_seconds() <= 120
            or not 0 <= (now - _utc(evidence.reference_time_utc, "chrony reference")).total_seconds() <= 120
        ):
            raise ClockHealthError("independent_utc_clock_evidence_unavailable_or_stale")
        return evidence

    def _load_latest_checkpoint(
        self,
        *,
        session_id: str,
        kind: str,
        cutoff_before: datetime | None = None,
        cutoff_exact: datetime | None = None,
        inclusive: bool = False,
    ) -> FeatureReplayCheckpoint | None:
        if kind not in {"checkpoint", "warmup"}:
            raise ValueError("Unsupported checkpoint kind")
        session_key = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]
        candidates: list[tuple[datetime, int, Path]] = []
        for path in Path(self.config.checkpoint_directory).glob(
            f"{kind}_{session_key}_*.json"
        ):
            parts = path.stem.split("_")
            if len(parts) != 5 or parts[0] != kind or parts[1] != session_key:
                raise ValueError("Malformed immutable checkpoint filename")
            try:
                cutoff = datetime.strptime(parts[2], "%Y%m%dT%H%M%SZ").replace(
                    tzinfo=timezone.utc
                )
                frame_index = int(parts[3])
            except ValueError as exc:
                raise ValueError("Malformed immutable checkpoint filename") from exc
            if frame_index < 0 or len(parts[4]) != 64:
                raise ValueError("Malformed immutable checkpoint filename")
            if cutoff_exact is not None and cutoff != cutoff_exact:
                continue
            if cutoff_before is not None and (
                cutoff > cutoff_before or (cutoff == cutoff_before and not inclusive)
            ):
                continue
            candidates.append((cutoff, frame_index, path))
        if not candidates:
            return None
        latest_identity = max((cutoff, index) for cutoff, index, _ in candidates)
        latest_paths = sorted(
            path
            for cutoff, index, path in candidates
            if (cutoff, index) == latest_identity
        )
        if len(latest_paths) != 1:
            raise ValueError("Conflicting immutable checkpoints share the same source position")
        path = latest_paths[0]
        checkpoint_type = FeatureReplayWarmup if kind == "warmup" else FeatureReplayCheckpoint
        try:
            checkpoint = checkpoint_type.load_durable(path)
        except (ValueError, OSError) as exc:
            raise ValueError(f"Checkpoint validation failed for {path.name}: {exc}") from exc
        if (
            checkpoint.session_id != session_id
            or checkpoint.cutoff_utc != latest_identity[0]
            or checkpoint.last_frame_index != latest_identity[1]
            or path.name != _filename(kind, checkpoint)
        ):
            raise ValueError("Checkpoint filename and immutable checkpoint provenance differ")
        return checkpoint

    def _latest_checkpoint_before(
        self,
        boundary: datetime,
    ) -> FeatureReplayCheckpoint | None:
        return self._load_latest_checkpoint(
            session_id=self.config.expected_source_session_id,
            kind="checkpoint",
            cutoff_before=boundary,
        )

    def _ensure_checkpoint(self, boundary: datetime, metrics: dict[str, Any]) -> FeatureReplayCheckpoint:
        started = time.perf_counter()
        checkpoint = self._latest_checkpoint_before(boundary)
        metrics["checkpoint_load_ms"] = (time.perf_counter() - started) * 1000
        if checkpoint is not None:
            return checkpoint
        raise ValueError("missing_ready_checkpoint: run explicit preparation before the boundary")

    def _read_with_retry(
        self,
        operation: Callable[[], _ReadResult],
        *,
        deadline: datetime,
        metrics: dict[str, Any],
    ) -> _ReadResult:
        """Retry only read contention, never inference, commits or attestations."""
        for attempt in range(3):
            if self._stopping:
                raise RuntimeError("runner_stopping_during_read")
            if self.scheduler.clock.now() > deadline:
                raise LatePredictionError("read_deadline_elapsed")
            try:
                return operation()
            except sqlite3.OperationalError as exc:
                code = getattr(exc, "sqlite_errorcode", None)
                transient = (
                    code is not None and (code & 0xFF) in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
                ) or str(exc) in {"database is locked", "database table is locked"}
                remaining = (deadline - self.scheduler.clock.now()).total_seconds()
                if not transient or attempt == 2 or remaining <= 0:
                    raise
                metrics["read_retries"] = metrics.get("read_retries", 0) + 1
                self._preparation_clock_health.now()
                self._require_fresh_utc_evidence()
                self.scheduler.waiter.wait(min(self.config.source_poll_seconds, remaining))
        raise AssertionError("Unreachable read retry state")

    def _validate_source_tail(
        self, tail: IncrementalRawFrameSnapshot, boundary: datetime
    ) -> None:
        if (
            tail.source_mode != "bridge"
            or tail.source_certified is not True
            or tail.clock_health_verified is not True
            or tail.watermark_utc != boundary
            or tail.through_boundary_utc != boundary
            or tail.session_id != self.config.expected_source_session_id
        ):
            raise ValueError("invalid_or_substitute_source_fence")
        _require_bridge_prediction_health(
            source_mode=tail.source_mode,
            source_certified=tail.source_certified,
            producer_health=tail.producer_health,
            boundary=boundary,
            session_id=tail.session_id,
        )

    def prepare_checkpoint(self, boundary_utc: datetime) -> dict[str, Any]:
        """Explicit pre-window maintenance. Never opens or writes a prediction ledger."""
        boundary = _utc(boundary_utc, "preparation boundary")
        if (
            boundary.minute % 15 or boundary.second or boundary.microsecond
            or not WINDOW_START <= boundary < WINDOW_END_EXCLUSIVE
        ):
            raise ValueError("Preparation requires an exact registered prediction boundary")
        metrics: dict[str, Any] = {}
        started = time.perf_counter()
        with SingleWriterGuard(self.config.prediction_ledger_path):
            checkpoint = self._prepare_checkpoint_locked(boundary, metrics)
        return {
            "mode": "checkpoint_preparation",
            "ready": True,
            "boundary_utc": _format_value(boundary),
            "cutoff_utc": _format_value(checkpoint.cutoff_utc),
            "source_session_id": checkpoint.session_id,
            "checkpoint": str(Path(self.config.checkpoint_directory) / _filename("checkpoint", checkpoint)),
            "integrity_sha256": checkpoint.data["integrity_sha256"],
            "official_records_written": False,
            "preparation_ms": (time.perf_counter() - started) * 1000,
            **metrics,
        }

    def _check_preparation_time(self, boundary: datetime) -> None:
        if self._stopping:
            raise RuntimeError("runner_stopping_during_warmup")
        self._preparation_clock_health.now()
        self._require_fresh_utc_evidence()
        if self.scheduler.clock.now() >= boundary:
            raise LatePredictionError("checkpoint_preparation_must_finish_before_boundary")

    def _prepare_checkpoint_locked(
        self, boundary: datetime, metrics: dict[str, Any]
    ) -> FeatureReplayCheckpoint:
        self._check_preparation_time(boundary)
        session_id = self.config.expected_source_session_id
        if self.journal.session_mode(session_id) != "bridge":
            raise ValueError("Checkpoint preparation requires a bridge source session")
        source_started = self.journal.session_started_at(session_id)
        if source_started is None:
            raise ValueError("Configured bridge source session is not registered")
        cutoff = boundary - timedelta(minutes=15)
        if source_started > cutoff - timedelta(hours=4):
            raise ValueError("incomplete_four_hour_warmup")
        ready = self._load_latest_checkpoint(
            session_id=session_id, kind="checkpoint", cutoff_exact=cutoff,
        )
        if ready is not None:
            witness = self._read_with_retry(
                lambda: self.journal.tail_snapshot(
                    session_id, after_frame_index=ready.last_frame_index,
                    expected_prefix_commitment_sha256=ready.prefix_commitment_sha256,
                    through_boundary_utc=cutoff, max_tail_frames=1,
                ),
                deadline=boundary, metrics=metrics,
            )
            self._validate_source_tail(witness, cutoff)
            if witness.frames or witness.has_more:
                raise ValueError("Ready checkpoint source prefix is incomplete")
            self._check_preparation_time(boundary)
            metrics["recovered_ready_checkpoint"] = True
            return ready
        warmup = self._load_latest_checkpoint(
            session_id=session_id,
            kind="warmup",
            cutoff_exact=cutoff,
        )
        if warmup is not None and not isinstance(warmup, FeatureReplayWarmup):
            warmup = FeatureReplayWarmup.from_dict(warmup.data)
        metrics["resumed_warmup"] = warmup is not None
        metrics["warmup_chunks"] = 0
        metrics["warmup_frames"] = 0
        if warmup is None:
            prefix_index = -1
            prefix_digest = initial_prefix_commitment(session_id, source_started)
        else:
            prefix_index = warmup.last_frame_index
            prefix_digest = warmup.prefix_commitment_sha256

        while not self._stopping:
            self._check_preparation_time(boundary)
            chunk = self._read_with_retry(
                lambda: self.journal.warmup_chunk(
                    session_id,
                    after_frame_index=prefix_index,
                    expected_prefix_commitment_sha256=prefix_digest,
                    through_boundary_utc=cutoff,
                    max_chunk_frames=self.config.warmup_chunk_frames,
                ),
                deadline=boundary, metrics=metrics,
            )
            self._validate_source_tail(chunk, cutoff)
            if chunk.frames:
                warmup = accumulate_feature_replay_warmup(chunk, cutoff, warmup)
                _save_checkpoint(Path(self.config.checkpoint_directory), warmup, warmup=True)
                metrics["warmup_chunks"] += 1
                metrics["warmup_frames"] += len(chunk.frames)
                prefix_index = warmup.last_frame_index
                prefix_digest = warmup.prefix_commitment_sha256
            if not chunk.has_more:
                if warmup is None:
                    raise ValueError("incomplete_four_hour_warmup")
                # Read an empty, fresh completion witness so the last chunk is
                # not replayed twice by finish_feature_replay_warmup().
                exhausted = self._read_with_retry(
                    lambda: self.journal.warmup_chunk(
                        session_id,
                        after_frame_index=prefix_index,
                        expected_prefix_commitment_sha256=prefix_digest,
                        through_boundary_utc=cutoff,
                        max_chunk_frames=self.config.warmup_chunk_frames,
                    ),
                    deadline=boundary, metrics=metrics,
                )
                self._validate_source_tail(exhausted, cutoff)
                if exhausted.frames or exhausted.has_more:
                    continue
                checkpoint = finish_feature_replay_warmup(exhausted, warmup)
                self._check_preparation_time(boundary)
                _save_checkpoint(Path(self.config.checkpoint_directory), checkpoint)
                self._check_preparation_time(boundary)
                return checkpoint
        raise RuntimeError("runner_stopping_during_warmup")

    def _read_exact_fence(
        self,
        boundary: datetime,
        checkpoint: FeatureReplayCheckpoint,
        metrics: dict[str, Any],
    ):
        assert self.ledger is not None
        session_id = self.config.expected_source_session_id
        started = time.perf_counter()
        fence_started = started
        while not self._stopping:
            now = _utc(self.scheduler.clock.now(), "runner UTC time")
            if now < boundary:
                self.scheduler.waiter.wait(
                    min(self.config.source_poll_seconds, (boundary - now).total_seconds())
                )
                continue
            deadline = boundary + timedelta(seconds=PREDICTION_DEADLINE_SECONDS)
            if now > deadline:
                raise LatePredictionError("missed_deadline_waiting_for_exact_source_fence")
            self.ledger.clock_health.now()
            self._require_fresh_utc_evidence()
            if not self._read_with_retry(
                lambda: self.journal.has_fence_record(session_id, boundary),
                deadline=deadline, metrics=metrics,
            ):
                now = self.scheduler.clock.now()
                if now >= deadline:
                    raise LatePredictionError("missed_deadline_waiting_for_exact_source_fence")
                self.scheduler.waiter.wait(
                    min(
                        self.config.source_poll_seconds,
                        max(0.0, (deadline - now).total_seconds()),
                    )
                )
                continue
            tail_started = time.perf_counter()
            tail = self._read_with_retry(
                lambda: self.journal.tail_snapshot(
                    session_id,
                    after_frame_index=checkpoint.last_frame_index,
                    expected_prefix_commitment_sha256=checkpoint.prefix_commitment_sha256,
                    through_boundary_utc=boundary,
                    max_tail_frames=self.config.max_tail_frames,
                ),
                deadline=deadline, metrics=metrics,
            )
            metrics["source_fence_availability_ms"] = (time.perf_counter() - fence_started) * 1000
            metrics["source_tail_read_and_verification_ms"] = (time.perf_counter() - tail_started) * 1000
            metrics["source_wait_total_ms"] = (time.perf_counter() - started) * 1000
            self._validate_source_tail(tail, boundary)
            return tail
        raise RuntimeError("runner_stopped_waiting_for_source_fence")

    def _record_failure(self, boundary: datetime, status: str, reason: str) -> None:
        assert self.ledger is not None
        lifecycle = self.ledger.lifecycle(boundary)
        prediction_committed = lifecycle["prediction"] is not None
        if prediction_committed and status not in {
            "checkpoint_recovery_required",
            "clock_sync_failure",
        }:
            # A committed record is immutable; only report recovery trouble.
            return
        last_event = lifecycle["events"][-1] if lifecycle["events"] else None
        if last_event is not None and last_event["reason"] == reason:
            return
        event_type = (
            "clock_skew" if status == "clock_sync_failure"
            else "late_attempt" if status in {"late_prediction", "missed_deadline"}
            else "duplicate_attempt" if status == "duplicate_prediction"
            else "missing_data" if status in {"waiting_for_source_fence", "missing_source_data", "invalid_source_health"}
            else "inference_failed"
        )
        if any(event["event_type"] == event_type for event in lifecycle["events"]):
            return
        try:
            recorded = self.ledger.clock_health.now()
        except ClockHealthError:
            recorded = self.ledger.clock_health.observed_wall_utc
            if recorded is None:
                return
            event_type = "clock_skew"
        self.ledger.append_attempt(
            boundary,
            event_type=event_type,
            recorded_at_utc=recorded,
            reason=reason,
            details={"runner_status": status},
        )

    def _report_failure(
        self, boundary: datetime, status: str, reason: str, metrics: dict[str, Any]
    ) -> None:
        try:
            self._record_failure(boundary, status, reason)
        except sqlite3.Error as exc:
            metrics["failure_record_error"] = f"{type(exc).__name__}:{exc}"

    def recover_committed_checkpoint(self, boundary_utc: datetime) -> FeatureReplayCheckpoint:
        """Recover state after a committed prediction without rerunning or re-attesting it.

        This maintenance operation is deliberately separate from the boundary
        scheduler so an old checkpoint repair can never delay another deadline.
        It requires the exact original bridge fence, a trusted prior checkpoint,
        and a ledger prediction; it does not create prediction evidence.
        """
        if self.ledger is None:
            raise ValueError("Checkpoint recovery requires an existing immutable ledger")
        if self._writer_lock_held:
            try:
                return self._recover_committed_checkpoint_locked(boundary_utc)
            except Exception as exc:
                self._record_failure(
                    _utc(boundary_utc, "checkpoint recovery boundary"),
                    "checkpoint_recovery_required",
                    f"{type(exc).__name__}:{exc}",
                )
                raise
        with SingleWriterGuard(self.config.prediction_ledger_path):
            try:
                return self._recover_committed_checkpoint_locked(boundary_utc)
            except Exception as exc:
                self._record_failure(
                    _utc(boundary_utc, "checkpoint recovery boundary"),
                    "checkpoint_recovery_required",
                    f"{type(exc).__name__}:{exc}",
                )
                raise

    def _recover_committed_checkpoint_locked(
        self, boundary_utc: datetime
    ) -> FeatureReplayCheckpoint:
        assert self.ledger is not None
        boundary = _utc(boundary_utc, "checkpoint recovery boundary")
        lifecycle = self.ledger.lifecycle(boundary)
        if lifecycle["prediction"] is None:
            raise ValueError("checkpoint recovery requires an existing immutable prediction")
        checkpoint = self._load_latest_checkpoint(
            session_id=self.config.expected_source_session_id,
            kind="checkpoint",
            cutoff_before=boundary,
            inclusive=True,
        )
        if checkpoint is None:
            checkpoint = self._ensure_checkpoint(boundary, {})
        if checkpoint.cutoff_utc == boundary:
            return checkpoint
        tail = self.journal.tail_snapshot(
            self.config.expected_source_session_id,
            after_frame_index=checkpoint.last_frame_index,
            expected_prefix_commitment_sha256=checkpoint.prefix_commitment_sha256,
            through_boundary_utc=boundary,
            max_tail_frames=self.config.max_tail_frames,
        )
        if tail.source_mode != "bridge" or tail.source_certified is not True:
            raise ValueError("checkpoint recovery requires the exact certified bridge fence")
        advanced = advance_feature_replay_checkpoint(tail, boundary, checkpoint)
        _save_checkpoint(Path(self.config.checkpoint_directory), advanced)
        return advanced

    def process_boundary(self, boundary_utc: datetime) -> BoundaryResult:
        if not self.official_mode or self.ledger is None:
            raise ValueError("Prediction processing is unavailable in diagnostic-only mode")
        if self._writer_lock_held:
            return self._process_boundary_locked(boundary_utc)
        with SingleWriterGuard(self.config.prediction_ledger_path):
            return self._process_boundary_locked(boundary_utc)

    def _process_boundary_locked(self, boundary_utc: datetime) -> BoundaryResult:
        assert self.ledger is not None
        boundary = _utc(boundary_utc, "scheduled boundary")
        boundary_started = time.perf_counter()
        if boundary.minute % 15 or boundary.second or boundary.microsecond:
            raise ValueError("Runner accepts only exact UTC quarter-hour boundaries")
        if not WINDOW_START <= boundary < WINDOW_END_EXCLUSIVE:
            raise ValueError("Boundary is outside the frozen registered schedule")
        now = _utc(self.scheduler.clock.now(), "runner UTC time")
        if now < boundary:
            return BoundaryResult(boundary, "not_due", "scheduled boundary has not arrived")
        result_metrics: dict[str, Any] = {
            "startup_validation_ms": self.startup_validation_ms
        }
        try:
            # Even unconfirmed/late commits own this identity forever. A later
            # diagnostic event must not hide an earlier permanent missed event.
            deadline = boundary + timedelta(seconds=PREDICTION_DEADLINE_SECONDS)
            before = (
                self.ledger.lifecycle(boundary)
                if now > deadline
                else self._read_with_retry(
                    lambda: self.ledger.lifecycle(boundary),
                    deadline=deadline, metrics=result_metrics,
                )
            )
            if before["prediction"] is not None:
                return BoundaryResult(
                    boundary, "duplicate_prediction",
                    f"immutable ledger lifecycle: {before['status']}", metrics=result_metrics,
                )
            if before["label"] is not None or any(
                event["event_type"] == "missed" for event in before["events"]
            ):
                return BoundaryResult(boundary, "missed_deadline", "boundary is labeled or permanently missed")
            if self.scheduler.clock.now() > deadline:
                self.ledger.mark_missed_boundaries()
                return BoundaryResult(boundary, "missed_deadline", "historical boundary is never backfilled")
            if self.model_bundle is None or not is_verified_official_bundle(self.model_bundle):
                self._record_failure(boundary, "model_verification_failure", "official model bundle is unverified")
                return BoundaryResult(boundary, "model_verification_failure", "official model bundle is unverified")
            runtime_started = time.perf_counter()
            self._read_with_retry(
                self.validate_runtime, deadline=deadline, metrics=result_metrics,
            )
            result_metrics["runtime_validation_ms"] = (
                time.perf_counter() - runtime_started
            ) * 1000
            if not self.scheduler.is_open(boundary, self.scheduler.clock.now()):
                self.ledger.mark_missed_boundaries()
                return BoundaryResult(boundary, "missed_deadline", "deadline elapsed during runtime validation")
            if self.journal.session_mode(self.config.expected_source_session_id) != "bridge":
                raise ValueError("invalid_source_health: official inference requires bridge mode")
            checkpoint = self._ensure_checkpoint(boundary, result_metrics)
            tail = self._read_exact_fence(boundary, checkpoint, result_metrics)
            self._require_fresh_utc_evidence()
            operation_started = time.perf_counter()
            stage_timings: dict[str, float] = {}
            evidence = process_incremental_boundary(
                tail,
                boundary,
                checkpoint,
                self.model_bundle,
                implementation_git_sha=self.config.implementation_git_sha,
                ledger=self.ledger,
                timings=stage_timings,
                pre_persist_check=lambda: self._require_fresh_utc_evidence(force=True),
                pre_attestation_check=lambda: self._require_fresh_utc_evidence(force=True),
            )
            result_metrics.update(stage_timings)
            result_metrics["inference_and_ledger_total_ms"] = (
                time.perf_counter() - operation_started
            ) * 1000
            result_metrics["deadline_margin_seconds"] = (
                boundary + timedelta(seconds=PREDICTION_DEADLINE_SECONDS)
                - evidence["persisted_at_utc"]
            ).total_seconds()
            result_metrics.update(_process_metrics(self.ledger, boundary))
            clock_alert = None
            try:
                self._require_fresh_utc_evidence(force=True)
            except ClockHealthError as exc:
                # This is subsequent health, not a revision of the successful
                # durability decision. Keep evidence and still advance state.
                clock_alert = str(exc)
                result_metrics["subsequent_clock_alert"] = clock_alert
            advance_started = time.perf_counter()
            try:
                advanced = advance_feature_replay_checkpoint(tail, boundary, checkpoint)
                _save_checkpoint(Path(self.config.checkpoint_directory), advanced)
                result_metrics["checkpoint_advance_publish_ms"] = (
                    time.perf_counter() - advance_started
                ) * 1000
            except Exception as exc:
                result_metrics["checkpoint_recovery_required"] = True
                reason = f"prediction_committed_checkpoint_publication_failed:{type(exc).__name__}:{exc}"
                self._report_failure(boundary, "checkpoint_recovery_required", reason, result_metrics)
                result_metrics["total_boundary_completion_ms"] = (
                    time.perf_counter() - boundary_started
                ) * 1000
                return BoundaryResult(boundary, "prediction_committed_checkpoint_recovery_required", reason, evidence, result_metrics)
            result_metrics["total_boundary_completion_ms"] = (
                time.perf_counter() - boundary_started
            ) * 1000
            if clock_alert is not None:
                self._report_failure(
                    boundary, "clock_sync_failure",
                    f"post_attestation_alert:{clock_alert}", result_metrics,
                )
            return BoundaryResult(
                boundary, "predicted_clock_alert" if clock_alert else "predicted",
                clock_alert, evidence, result_metrics,
            )
        except LatePredictionError as exc:
            reason = str(exc)
            status = (
                "missing_source_data"
                if "source_fence" in reason or "certified_source_fence" in reason
                else "missed_deadline"
            )
        except DuplicatePredictionError as exc:
            status, reason = "duplicate_prediction", str(exc)
        except ClockHealthError as exc:
            status, reason = "clock_sync_failure", str(exc)
        except sqlite3.Error as exc:
            status, reason = "ledger_or_source_io_failure", f"{type(exc).__name__}:{exc}"
        except (FileNotFoundError, OSError) as exc:
            status, reason = "checkpoint_or_io_failure", f"{type(exc).__name__}:{exc}"
        except Exception as exc:
            message = str(exc)
            if "fence" in message.lower() and "lacks" in message.lower():
                status = "waiting_for_source_fence"
            elif "clock" in message.lower() or "chrony" in message.lower():
                status = "clock_sync_failure"
            elif "checkpoint" in message.lower() or "warmup" in message.lower() or "240_minute" in message:
                status = "checkpoint_failure"
            elif "duplicate" in message.lower():
                status = "duplicate_prediction"
            elif "source" in message.lower() or "bridge" in message.lower() or "boundary" in message.lower():
                status = "invalid_source_health"
            else:
                status = "inference_or_persistence_failure"
            reason = f"{type(exc).__name__}:{message}"
        self._report_failure(boundary, status, reason, result_metrics)
        if _utc(self.scheduler.clock.now(), "runner UTC time") > boundary + timedelta(seconds=PREDICTION_DEADLINE_SECONDS):
            try:
                self.ledger.mark_missed_boundaries()
            except (sqlite3.Error, ClockHealthError) as exc:
                result_metrics["missed_reconciliation_error"] = f"{type(exc).__name__}:{exc}"
            if status == "waiting_for_source_fence":
                status = "missing_source_data"
        result_metrics.update(_process_metrics(self.ledger, boundary))
        result_metrics["total_boundary_completion_ms"] = (
            time.perf_counter() - boundary_started
        ) * 1000
        return BoundaryResult(boundary, status, reason, None, result_metrics)

    def run_forever(
        self,
        *,
        diagnostic_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Explicit opt-in loop. No caller invokes this merely by importing the module."""
        if not self.official_mode or self.ledger is None:
            raise ValueError("Diagnostic mode cannot start a scheduler loop")
        startup_metrics: dict[str, Any] = {}
        startup_evidence = self._read_with_retry(
            self.validate_runtime,
            deadline=self.scheduler.clock.now() + timedelta(seconds=PREDICTION_DEADLINE_SECONDS),
            metrics=startup_metrics,
        )
        with SingleWriterGuard(self.config.prediction_ledger_path):
            self._writer_lock_held = True
            try:
                if diagnostic_sink is not None:
                    diagnostic_sink({
                        "event": "runner_started",
                        "source_session_id": self.config.expected_source_session_id,
                        "clock_evidence_source": startup_evidence.source,
                        "startup_validation_ms": self.startup_validation_ms,
                        "startup_read_retries": startup_metrics.get("read_retries", 0),
                        "scheduled_boundaries": self.ledger.scheduled_count,
                        "window_start_utc": WINDOW_START.isoformat().replace("+00:00", "Z"),
                        "window_end_exclusive_utc": WINDOW_END_EXCLUSIVE.isoformat().replace("+00:00", "Z"),
                    })
                boundary = max(
                    WINDOW_START,
                    BoundaryScheduler.most_recent_boundary(self.scheduler.clock.now()),
                )
                while not self._stopping:
                    # Reconciliation is idempotent; no prediction or attestation
                    # write is retried when source/ledger reads are contended.
                    try:
                        self.ledger.mark_missed_boundaries()
                    except (sqlite3.Error, ClockHealthError) as exc:
                        if diagnostic_sink is None:
                            raise
                        diagnostic_sink({
                            "event": "missed_reconciliation_failed",
                            "reason": f"{type(exc).__name__}:{exc}",
                        })
                    if boundary >= WINDOW_END_EXCLUSIVE:
                        return
                    self.scheduler.wait_for_boundary(
                        boundary, should_stop=lambda: self._stopping,
                    )
                    if self._stopping:
                        return
                    result = self.process_boundary(boundary)
                    if diagnostic_sink is not None:
                        diagnostic_sink(result.as_dict())
                    boundary = max(
                        boundary + timedelta(minutes=15),
                        BoundaryScheduler.most_recent_boundary(self.scheduler.clock.now()),
                    )
            finally:
                self._writer_lock_held = False


def _process_metrics(ledger: ProspectiveLedger, boundary: datetime) -> dict[str, Any]:
    result: dict[str, Any] = {
        "cpu_time_seconds": time.process_time(),
        "scheduled_boundaries": ledger.scheduled_count,
    }
    try:
        with open(f"/proc/{os.getpid()}/statm", "r", encoding="ascii") as stream:
            resident_pages = int(stream.read().split()[1])
        result["rss_bytes"] = resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError, AttributeError):
        result["rss_bytes"] = None
    try:
        result["lifecycle_status"] = ledger.lifecycle(boundary)["status"]
        row = ledger._connection.execute(
            "SELECT COUNT(*) FROM scheduled_boundaries s LEFT JOIN predictions p "
            "ON p.boundary_utc=s.boundary_utc WHERE p.boundary_utc IS NULL"
        ).fetchone()
        result["prediction_backlog_boundaries"] = int(row[0])
    except Exception:
        result["lifecycle_status"] = "diagnostic_unavailable"
        result["prediction_backlog_boundaries"] = None
    return result


def _format_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return _utc(value, "diagnostic timestamp").isoformat().replace("+00:00", "Z")
    if isinstance(value, dict):
        return {str(key): _format_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_format_value(item) for item in value]
    return value


def summarize_latency_samples(samples_ms: list[float]) -> dict[str, float | int | None]:
    """Deterministic offline summary; never represents verified production SLOs."""
    values = sorted(float(value) for value in samples_ms if math.isfinite(float(value)))
    if not values:
        return {"count": 0, "median_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None}

    def percentile(fraction: float) -> float:
        index = max(0, math.ceil(fraction * len(values)) - 1)
        return values[index]

    middle = len(values) // 2
    median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
    return {
        "count": len(values),
        "median_ms": median,
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "max_ms": values[-1],
    }
