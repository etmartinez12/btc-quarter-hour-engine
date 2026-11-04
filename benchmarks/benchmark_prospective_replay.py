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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from btc_quarter_hour_engine.research.prospective_replay import (
    FeatureReplayCheckpoint,
    advance_feature_replay_checkpoint,
    build_incremental_causal_feature_vector,
    create_feature_replay_checkpoint,
)
from btc_quarter_hour_engine.research.prospective_source import (
    CausalFeatureVector,
    ProspectiveRawFrameJournal,
    RawFrameRecord,
    RawFrameSnapshot,
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
    if frame_count < 17:
        raise ValueError("frame_count must be >= 17 for 240-minute history")
    if boundary_count < 1:
        raise ValueError("boundary_count must be >= 1")
    final_boundary = BENCHMARK_BOUNDARY + timedelta(minutes=15 * boundary_count)
    start = BENCHMARK_BOUNDARY - timedelta(minutes=240)
    span = BENCHMARK_BOUNDARY - start
    records = [
        _make_record(
            index,
            index,
            start + span * index / (frame_count - 1),
            snapshot=index == 0,
        )
        for index in range(frame_count)
    ]
    next_sequence = frame_count
    for boundary_index in range(1, boundary_count + 1):
        timestamp = BENCHMARK_BOUNDARY + timedelta(minutes=15 * boundary_index)
        records.extend((
            _make_record(len(records), next_sequence, timestamp, snapshot=False),
            _make_record(len(records) + 1, next_sequence + 1, timestamp, snapshot=False),
        ))
        next_sequence += 2
    return RawFrameSnapshot(
        session_id="offline-benchmark",
        session_started_at_utc=start,
        watermark_utc=final_boundary,
        clock_health_verified=True,
        frames=tuple(records),
    )


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
) -> dict[str, object]:
    results = []
    with tempfile.TemporaryDirectory(prefix="prospective-replay-benchmark-") as directory:
        root = Path(directory)
        for frame_count in frame_counts:
            source = synthetic_snapshot(frame_count, boundary_count)
            setup_started = time.perf_counter()
            checkpoint = create_feature_replay_checkpoint(source, BENCHMARK_BOUNDARY)
            checkpoint_setup_ms = (time.perf_counter() - setup_started) * 1000
            case_dir = root / str(frame_count)
            case_dir.mkdir()
            checkpoint_path = case_dir / "checkpoint-000.json"
            checkpoint.write_once(checkpoint_path)
            journal = ProspectiveRawFrameJournal(case_dir / "source.sqlite")
            journal.start_session(source.session_id, source.session_started_at_utc)
            for frame in source.frames:
                journal.append_frame(
                    session_id=frame.session_id,
                    connection_id=frame.connection_id,
                    frame_index=frame.frame_index,
                    ingest_time_utc=frame.ingest_time_utc,
                    raw_bytes=frame.raw_bytes,
                )
            journal.advance_watermark(
                source.session_id,
                source.watermark_utc,
                clock_health_verified=True,
            )
            sink = SyntheticDurableLedger(case_dir / "synthetic-ledger.sqlite")
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
                    current_checkpoint.write_once(current_path)
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
            results.append({
                "session_frame_count": len(source.frames),
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
    args = parser.parse_args()
    counts = tuple(int(value) for value in args.frame_counts.split(","))
    print(json.dumps(run_benchmark(frame_counts=counts, boundary_count=args.boundaries), indent=2))


if __name__ == "__main__":
    main()
