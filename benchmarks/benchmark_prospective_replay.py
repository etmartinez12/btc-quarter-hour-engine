#!/usr/bin/env python3
"""Offline bounded causal replay benchmark; no model, market feed, or live ledger writes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import statistics
import sys
import tempfile
import time
import tracemalloc
from dataclasses import replace
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btc_quarter_hour_engine.research.prospective_replay import (
    MAX_CHECKPOINT_BOOTSTRAP_FRAMES,
    FeatureReplayCheckpoint,
    FeatureReplayWarmup,
    advance_feature_replay_checkpoint,
    build_incremental_causal_feature_vector,
    accumulate_feature_replay_warmup,
    finish_feature_replay_warmup,
    initial_prefix_commitment,
)
from btc_quarter_hour_engine.research.prospective_source import (
    CausalFeatureVector,
    ProspectiveRawFrameJournal,
    RawFrameRecord,
    RawFrameSnapshot,
    build_causal_feature_vector,
)


UTC = timezone.utc
BENCHMARK_BOUNDARY = datetime(2026, 11, 15, 0, 0, tzinfo=UTC)
STAGES = (
    "checkpoint_file_read_ms",
    "source_tail_read_ms",
    "checkpoint_restore_ms",
    "incremental_tail_replay_ms",
    "feature_generation_ms",
    "model_inference_ms",
    "ledger_commit_attestation_ms",
    "checkpoint_advance_and_write_ms",
)


def _wire_frame(sequence: int, timestamp: datetime, *, snapshot: bool) -> bytes:
    event_time = timestamp.isoformat().replace("+00:00", "Z")
    updates = (
        [
            {"side": "bid", "price_level": "100", "new_quantity": "1", "event_time": event_time},
            {"side": "offer", "price_level": "101", "new_quantity": "2", "event_time": event_time},
        ]
        if snapshot
        else [
            {"side": "bid", "price_level": "100", "new_quantity": "1.5", "event_time": event_time},
            {"side": "bid", "price_level": "99", "new_quantity": "0", "event_time": event_time},
        ]
    )
    return json.dumps({
        "channel": "l2_data",
        "client_id": "",
        "timestamp": event_time,
        "sequence_num": sequence,
        "events": [{
            "type": "snapshot" if snapshot else "update",
            "product_id": "BTC-USD",
            "updates": updates,
        }],
    }).encode()


def _make_record(index: int, sequence: int, timestamp: datetime, *, snapshot: bool) -> RawFrameRecord:
    raw = _wire_frame(sequence, timestamp, snapshot=snapshot)
    return RawFrameRecord(
        raw_bytes=raw,
        session_id="offline-benchmark",
        connection_id="offline-connection-1",
        frame_index=index,
        ingest_time_utc=timestamp,
        raw_frame_sha256=hashlib.sha256(raw).hexdigest(),
    )


def synthetic_snapshot(frame_count: int, boundary_count: int) -> RawFrameSnapshot:
    return RawFrameSnapshot(
        session_id="offline-benchmark",
        session_started_at_utc=BENCHMARK_BOUNDARY - timedelta(minutes=240),
        watermark_utc=BENCHMARK_BOUNDARY + timedelta(minutes=15 * boundary_count),
        clock_health_verified=True,
        frames=tuple(synthetic_frames(frame_count, boundary_count)),
    )


def synthetic_frames(frame_count: int, boundary_count: int, history_hours: int = 4):
    """Stream arbitrary source history without retaining raw frames in memory."""
    if frame_count < 17:
        raise ValueError("frame_count must be >= 17 for 240-minute history")
    if boundary_count < 1:
        raise ValueError("boundary_count must be >= 1")
    if history_hours < 4:
        raise ValueError("history_hours must be >= 4")
    start = BENCHMARK_BOUNDARY - timedelta(hours=history_hours)
    span = BENCHMARK_BOUNDARY - start
    for index in range(frame_count):
        yield _make_record(
            index,
            index,
            start + span * index / (frame_count - 1),
            snapshot=index == 0,
        )
    next_sequence = frame_count
    for boundary_index in range(1, boundary_count + 1):
        timestamp = BENCHMARK_BOUNDARY + timedelta(minutes=15 * boundary_index)
        yield _make_record(next_sequence, next_sequence, timestamp, snapshot=False)
        yield _make_record(next_sequence + 1, next_sequence + 1, timestamp, snapshot=False)
        next_sequence += 2


def synthetic_inference(features: CausalFeatureVector) -> tuple[float, float]:
    """Replace with an inference callback to measure a local frozen-model load."""
    del features
    return 0.51, 0.49


class SyntheticDurableLedger:
    """SQLite FULL-sync test sink, not the prospective evidence ledger."""

    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path, isolation_level=None)
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = FULL")
        self.connection.executescript(
            """
            CREATE TABLE predictions (
                boundary_utc TEXT PRIMARY KEY,
                feature_fingerprint TEXT NOT NULL,
                candidate_probability REAL NOT NULL,
                control_probability REAL NOT NULL
            );
            CREATE TABLE attestations (
                boundary_utc TEXT PRIMARY KEY REFERENCES predictions(boundary_utc),
                attested_at_utc TEXT NOT NULL
            );
            """
        )

    def persist_and_attest(
        self,
        features: CausalFeatureVector,
        probabilities: tuple[float, float],
    ) -> None:
        boundary = features.boundary_utc.isoformat()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self.connection.execute(
                "INSERT INTO predictions VALUES (?, ?, ?, ?)",
                (boundary, features.feature_fingerprint, *probabilities),
            )
            self.connection.execute("COMMIT")
        except BaseException:
            if self.connection.in_transaction:
                self.connection.execute("ROLLBACK")
            raise
        attested = datetime.now(UTC).isoformat()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self.connection.execute(
                "INSERT INTO attestations VALUES (?, ?)", (boundary, attested)
            )
            self.connection.execute("COMMIT")
        except BaseException:
            if self.connection.in_transaction:
                self.connection.execute("ROLLBACK")
            raise

    def close(self) -> None:
        self.connection.close()


def _summary(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"median": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(samples)
    return {
        "median": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max": max(ordered),
    }


def run_benchmark(
    *,
    frame_counts: tuple[int, ...] = (256, 1024, 4096),
    boundary_count: int = 12,
    inference: Callable[[CausalFeatureVector], tuple[float, float]] = synthetic_inference,
    ledger_callback: Callable[[CausalFeatureVector, tuple[float, float]], None] | None = None,
    warmup_chunk_frames: int = 10_000,
    history_hours: int = 4,
    simulate_checkpoint_publication: bool = False,
    verify_full_replay: bool = False,
    fast_synthetic_fixture: bool = False,
) -> dict[str, object]:
    if type(warmup_chunk_frames) is not int or not 1 <= warmup_chunk_frames <= MAX_CHECKPOINT_BOOTSTRAP_FRAMES:
        raise ValueError("Warm-up chunk size must be within the existing hard frame limit")

    def publish(checkpoint, path):
        if simulate_checkpoint_publication:
            # Explicit benchmark-only I/O simulation, never used by production.
            with path.open("x", encoding="utf-8") as stream:
                stream.write(checkpoint.serialized)
        else:
            checkpoint.write_once(path)

    results = []
    with tempfile.TemporaryDirectory(prefix="prospective-replay-benchmark-") as directory, ExitStack() as cleanup:
        root = Path(directory)
        for frame_count in frame_counts:
            source = RawFrameSnapshot(
                session_id="offline-benchmark",
                session_started_at_utc=BENCHMARK_BOUNDARY - timedelta(hours=history_hours),
                watermark_utc=BENCHMARK_BOUNDARY + timedelta(minutes=15 * boundary_count),
                clock_health_verified=True, frames=(),
            )
            case_dir = root / str(frame_count)
            case_dir.mkdir()
            checkpoint_path = case_dir / "checkpoint-000.json"
            journal = ProspectiveRawFrameJournal(case_dir / "source.sqlite")
            cleanup.callback(journal.close)
            journal.start_session(source.session_id, source.session_started_at_utc)
            if fast_synthetic_fixture:
                # Synthetic fixture generation only, outside measured stages.
                # Never apply this to an actual source journal.
                journal._connection.execute("PRAGMA synchronous = OFF")
            for frame in synthetic_frames(frame_count, boundary_count, history_hours):
                journal.append_frame(
                    session_id=frame.session_id,
                    connection_id=frame.connection_id,
                    frame_index=frame.frame_index,
                    ingest_time_utc=frame.ingest_time_utc,
                    raw_bytes=frame.raw_bytes,
                )
            if fast_synthetic_fixture:
                journal._connection.execute("PRAGMA synchronous = FULL")
            journal.advance_watermark(
                source.session_id,
                source.watermark_utc,
                clock_health_verified=True,
            )
            # Normal restart checks only the committed schema marker; source
            # history is neither counted nor replayed here.
            journal.close()
            restart_started = time.perf_counter()
            journal = ProspectiveRawFrameJournal(case_dir / "source.sqlite")
            cleanup.callback(journal.close)
            journal_restart_ms = (time.perf_counter() - restart_started) * 1000
            setup_started = time.perf_counter()
            warmup = None
            chunk_count = 0
            max_chunk_size = 0
            max_warmup_size = 0
            while True:
                chunk = journal.warmup_chunk(
                    source.session_id,
                    after_frame_index=-1 if warmup is None else warmup.last_frame_index,
                    expected_prefix_commitment_sha256=(
                        initial_prefix_commitment(source.session_id, source.session_started_at_utc)
                        if warmup is None else warmup.prefix_commitment_sha256
                    ),
                    through_boundary_utc=BENCHMARK_BOUNDARY,
                    max_chunk_frames=warmup_chunk_frames,
                )
                warmup = accumulate_feature_replay_warmup(chunk, BENCHMARK_BOUNDARY, warmup)
                chunk_count += 1
                max_chunk_size = max(max_chunk_size, len(chunk.frames))
                max_warmup_size = max(max_warmup_size, len(warmup.serialized.encode("utf-8")))
                warmup_path = case_dir / f"warmup-{chunk_count:05d}.json"
                publish(warmup, warmup_path)
                warmup = FeatureReplayWarmup.load(warmup_path)
                if not chunk.has_more:
                    exhausted = replace(
                        chunk, frames=(), frame_commitments=(),
                        prefix_frame_index=warmup.last_frame_index,
                        prefix_commitment_sha256=warmup.prefix_commitment_sha256,
                    )
                    checkpoint = finish_feature_replay_warmup(exhausted, warmup)
                    break
            publish(checkpoint, checkpoint_path)
            checkpoint_setup_ms = (time.perf_counter() - setup_started) * 1000
            # Separate allocation-only pass over one maximum-size source chunk;
            # do not contaminate the timed resumable bootstrap with tracing.
            tracemalloc.start()
            try:
                memory_chunk = journal.warmup_chunk(
                    source.session_id, after_frame_index=-1,
                    expected_prefix_commitment_sha256=initial_prefix_commitment(
                        source.session_id, source.session_started_at_utc,
                    ),
                    through_boundary_utc=BENCHMARK_BOUNDARY,
                    max_chunk_frames=warmup_chunk_frames,
                )
                memory_warmup = accumulate_feature_replay_warmup(memory_chunk, BENCHMARK_BOUNDARY)
                _, warmup_peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            del memory_chunk, memory_warmup
            sink = SyntheticDurableLedger(case_dir / "synthetic-ledger.sqlite")
            cleanup.callback(sink.close)
            samples = {stage: [] for stage in STAGES}
            failures = 0
            peak_memory_bytes = 0
            tail_work = []
            current_checkpoint = checkpoint
            current_path = checkpoint_path
            memory_sampled = False
            for boundary_index in range(1, boundary_count + 1):
                boundary = BENCHMARK_BOUNDARY + timedelta(minutes=15 * boundary_index)
                try:
                    stage_start = time.perf_counter()
                    current_checkpoint = FeatureReplayCheckpoint.load(current_path)
                    samples["checkpoint_file_read_ms"].append((time.perf_counter() - stage_start) * 1000)
                    stage_start = time.perf_counter()
                    tail = journal.tail_snapshot(
                        source.session_id,
                        after_frame_index=current_checkpoint.last_frame_index,
                        expected_prefix_commitment_sha256=current_checkpoint.prefix_commitment_sha256,
                        through_boundary_utc=boundary,
                    )
                    samples["source_tail_read_ms"].append((time.perf_counter() - stage_start) * 1000)
                    timings: dict[str, float] = {}
                    features = build_incremental_causal_feature_vector(
                        tail, boundary, current_checkpoint, timings=timings
                    )
                    for stage in (
                        "checkpoint_restore_ms",
                        "incremental_tail_replay_ms",
                        "feature_generation_ms",
                    ):
                        samples[stage].append(timings[stage])
                    tail_work.append(len(tail.frames))
                    stage_start = time.perf_counter()
                    probabilities = inference(features)
                    samples["model_inference_ms"].append((time.perf_counter() - stage_start) * 1000)
                    stage_start = time.perf_counter()
                    if ledger_callback is None:
                        sink.persist_and_attest(features, probabilities)
                    else:
                        ledger_callback(features, probabilities)
                    samples["ledger_commit_attestation_ms"].append((time.perf_counter() - stage_start) * 1000)
                    stage_start = time.perf_counter()
                    current_checkpoint = advance_feature_replay_checkpoint(
                        tail, boundary, current_checkpoint
                    )
                    current_path = case_dir / f"checkpoint-{boundary_index:03d}.json"
                    publish(current_checkpoint, current_path)
                    samples["checkpoint_advance_and_write_ms"].append((time.perf_counter() - stage_start) * 1000)
                    if not memory_sampled:
                        # Measure allocations in a separate pass so tracemalloc
                        # overhead does not inflate stage latency results.
                        tracemalloc.start()
                        tracemalloc.reset_peak()
                        memory_tail = journal.tail_snapshot(
                            source.session_id,
                            after_frame_index=checkpoint.last_frame_index,
                            expected_prefix_commitment_sha256=checkpoint.prefix_commitment_sha256,
                            through_boundary_utc=boundary,
                        )
                        build_incremental_causal_feature_vector(
                            memory_tail, boundary, checkpoint
                        )
                        _, peak = tracemalloc.get_traced_memory()
                        peak_memory_bytes = max(peak_memory_bytes, peak)
                        tracemalloc.stop()
                        memory_sampled = True
                except Exception:
                    failures += 1
                    if tracemalloc.is_tracing():
                        tracemalloc.stop()
                    break
            sink.close()
            journal.close()
            parity = None
            full_replay_ms = None
            if verify_full_replay and failures == 0:
                # Offline reference only; intentionally materializes complete
                # history, outside bounded-path timings/allocation measurements.
                full_source = replace(
                    source, frames=tuple(synthetic_frames(frame_count, boundary_count, history_hours)),
                )
                reference_started = time.perf_counter()
                reference = build_causal_feature_vector(full_source, source.watermark_utc)
                full_replay_ms = (time.perf_counter() - reference_started) * 1000
                parity = (
                    features.candidate_predictors == reference.candidate_predictors
                    and features.control_predictors == reference.control_predictors
                    and features.values == reference.values
                    and features.feature_fingerprint == reference.feature_fingerprint
                )
                if not parity:
                    failures += 1
                del full_source, reference
            results.append({
                "session_frame_count": frame_count + 2 * boundary_count,
                "history_hours": history_hours,
                "warmup_chunks": chunk_count,
                "max_warmup_chunk_frames": max_chunk_size,
                "max_warmup_serialized_bytes": max_warmup_size,
                "peak_traced_warmup_memory_bytes": warmup_peak,
                "journal_restart_ms": journal_restart_ms,
                "full_replay_parity": parity,
                "full_replay_ms": full_replay_ms,
                "initial_historical_frame_count": frame_count,
                "checkpoint_serialized_bytes": len(checkpoint.serialized.encode("utf-8")),
                "boundaries_completed": len(tail_work),
                "initial_checkpoint_build_ms": checkpoint_setup_ms,
                "tail_frames_per_boundary": {
                    "median": statistics.median(tail_work) if tail_work else 0,
                    "max": max(tail_work, default=0),
                },
                "peak_traced_python_memory_bytes": peak_memory_bytes,
                "failure_count": failures,
                "stage_ms": {stage: _summary(values) for stage, values in samples.items()},
            })
    return {
        "benchmark": "bounded prospective causal replay",
        "mode": "synthetic offline; no Coinbase feed, official models, or prospective ledger",
        "boundary_count_requested": boundary_count,
        "checkpoint_publication": "SIMULATED (no durability claim)" if simulate_checkpoint_publication else "POSIX file and directory fsync",
        "source_fixture_sync": "UNSYNCED synthetic generation only; FULL restored before watermark" if fast_synthetic_fixture else "FULL",
        "session_sizes": results,
        "limitations": [
            "Synthetic measurements do not establish production T+60-second performance.",
            "The default inference and ledger callbacks are placeholders; inject local reviewed callbacks for those stages.",
            "Python allocation peak excludes native allocator and operating-system page-cache memory.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-counts", default="256,1024,4096")
    parser.add_argument("--boundaries", type=int, default=12)
    parser.add_argument("--warmup-chunk-frames", type=int, default=10_000)
    parser.add_argument("--history-hours", type=int, default=4)
    parser.add_argument("--simulate-checkpoint-publication", action="store_true",
                        help="Benchmark-only file I/O simulation; no durability claim")
    parser.add_argument("--verify-full-replay", action="store_true",
                        help="Also materialize the full offline reference outside bounded-path measurements")
    parser.add_argument("--fast-synthetic-fixture", action="store_true",
                        help="Skip commit fsync only while generating synthetic source fixtures, outside measured stages")
    args = parser.parse_args()
    counts = tuple(int(value) for value in args.frame_counts.split(","))
    print(json.dumps(run_benchmark(
        frame_counts=counts, boundary_count=args.boundaries,
        warmup_chunk_frames=args.warmup_chunk_frames, history_hours=args.history_hours,
        simulate_checkpoint_publication=args.simulate_checkpoint_publication,
        verify_full_replay=args.verify_full_replay,
        fast_synthetic_fixture=args.fast_synthetic_fixture,
    ), indent=2))


if __name__ == "__main__":
    main()
