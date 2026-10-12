"""Deterministic order-book and canonical-replay benchmarks.

Workload construction is excluded from timed sections. The order-book mode
compares the production implementation with the independent copy-on-write
oracle in ``tests/reference_order_book.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.reference_order_book import CopyOnWriteReferenceBook

RUNTIME_ROOT = Path(os.environ.get("BTC_QH_RUNTIME_ROOT", ROOT)).resolve()
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from btc_quarter_hour_engine.market_data.order_book import Level2OrderBook
from btc_quarter_hour_engine.market_data.replay import CanonicalReplayAccumulator
from btc_quarter_hour_engine.market_data import replay as replay_module


def _runtime_metadata() -> dict[str, str]:
    git_sha = subprocess.run(
        ["git", "-C", str(RUNTIME_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "runtime_root": str(RUNTIME_ROOT),
        "runtime_git_sha": git_sha,
        "order_book_module": str(Path(inspect.getfile(Level2OrderBook)).resolve()),
        "replay_module": str(Path(inspect.getfile(replay_module)).resolve()),
    }


def _levels(depth: int) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    bids = [{"price": float(50_000 - index), "quantity": 1.0} for index in range(depth)]
    asks = [{"price": float(100_000 + index), "quantity": 1.0} for index in range(depth)]
    return bids, asks


def _book_snapshot(book: Any, depth: int) -> None:
    bids, asks = _levels(depth)
    book.apply_snapshot(product_id="BTC-USD", levels={"bid": bids, "ask": asks})


def _make_workload(
    workload: str,
    *,
    depth: int,
    envelopes: int,
    updates_per_envelope: int,
    seed: int,
) -> list[list[dict[str, Any]]]:
    rng = random.Random(seed)
    bid_prices = [float(50_000 - index) for index in range(depth)]
    ask_prices = [float(100_000 + index) for index in range(depth)]
    result: list[list[dict[str, Any]]] = []
    if workload == "top-churn":
        next_bid = 50_000.01
        next_ask = 100_000.0 - 0.01
        current_bid = 50_000.0
        current_ask = 100_000.0
        for _ in range(envelopes):
            result.append([
                {"side": "bid", "price": current_bid, "quantity": 0.0},
                {"side": "bid", "price": next_bid, "quantity": 1.0},
                {"side": "ask", "price": current_ask, "quantity": 0.0},
                {"side": "ask", "price": next_ask, "quantity": 1.0},
            ])
            current_bid = next_bid
            current_ask = next_ask
            next_bid += 0.01
            next_ask -= 0.01
    elif workload == "deep-churn":
        for index in range(envelopes):
            side = "bid" if index % 2 == 0 else "ask"
            price = float(1 + index if side == "bid" else 140_000 + index)
            result.append([
                {"side": side, "price": price, "quantity": 1.0},
                {"side": side, "price": price, "quantity": 0.0},
            ])
    elif workload == "rebuild":
        result.append([
            *({"side": "bid", "price": row["price"], "quantity": 0.0} for row in _levels(depth)[0]),
            *({"side": "ask", "price": row["price"], "quantity": 0.0} for row in _levels(depth)[1]),
        ])
        result.append([
            *({"side": "bid", "price": row["price"], "quantity": 1.0} for row in _levels(depth)[0]),
            *({"side": "ask", "price": row["price"], "quantity": 1.0} for row in _levels(depth)[1]),
        ])
        while len(result) < envelopes:
            price = rng.choice(bid_prices)
            result.append([{"side": "bid", "price": price, "quantity": rng.random() + 0.01}])
    else:
        for _ in range(envelopes):
            updates = []
            for _ in range(rng.randint(max(1, updates_per_envelope - 1), updates_per_envelope + 1)):
                side = rng.choice(("bid", "ask"))
                price = rng.choice(bid_prices if side == "bid" else ask_prices)
                updates.append({"side": side, "price": price, "quantity": rng.random() + 0.01})
            result.append(updates)
    return result


def _run_book(book: Any, workload: list[list[dict[str, Any]]]) -> tuple[float, int]:
    started = time.perf_counter_ns()
    for updates in workload:
        book.apply_updates(updates)
    elapsed_ns = time.perf_counter_ns() - started
    return elapsed_ns / 1_000_000_000, sum(len(updates) for updates in workload)


def _max_heap_live_ratios(depth: int, workload: list[list[dict[str, Any]]]) -> dict[str, float]:
    book = Level2OrderBook(product_id="BTC-USD")
    _book_snapshot(book, depth)
    ratios = {"bid": 1.0, "ask": 1.0}
    for updates in workload:
        book.apply_updates(updates)
        ratios["bid"] = max(ratios["bid"], len(book._bid_heap) / max(1, len(book.bids)))
        ratios["ask"] = max(ratios["ask"], len(book._ask_heap) / max(1, len(book.asks)))
        if (
            len(book._bid_heap) > 2 * len(book.bids) + 1024
            or len(book._ask_heap) > 2 * len(book.asks) + 1024
        ):
            raise RuntimeError("Heap compaction exceeded its documented bound")
    return ratios


def _book_result(workload: str, *, depth: int, envelopes: int, updates_per_envelope: int,
                 seed: int, repeats: int) -> dict[str, Any]:
    updates = _make_workload(
        workload, depth=depth, envelopes=envelopes,
        updates_per_envelope=updates_per_envelope, seed=seed,
    )
    results: dict[str, list[float]] = {"copy_on_write_reference": [], "optimized": []}
    final_state: dict[str, Any] | None = None
    update_count = sum(len(envelope) for envelope in updates)
    for _ in range(repeats):
        reference = CopyOnWriteReferenceBook(product_id="BTC-USD")
        optimized = Level2OrderBook(product_id="BTC-USD")
        _book_snapshot(reference, depth)
        _book_snapshot(optimized, depth)
        for label, book in (("copy_on_write_reference", reference), ("optimized", optimized)):
            elapsed, _ = _run_book(book, updates)
            results[label].append(elapsed)
        if (
            reference.bids != optimized.bids
            or reference.asks != optimized.asks
            or reference.state.value != optimized.state.value
            or reference.best_bid != optimized.best_bid
            or reference.best_ask != optimized.best_ask
        ):
            raise RuntimeError(f"Final semantic mismatch in {workload!r} workload")
        final_state = {
            "state": optimized.state.value,
            "live_bids": len(optimized.bids),
            "live_asks": len(optimized.asks),
            "bid_heap": len(getattr(optimized, "_bid_heap", ())),
            "ask_heap": len(getattr(optimized, "_ask_heap", ())),
        }
    reference_median = statistics.median(results["copy_on_write_reference"])
    optimized_median = statistics.median(results["optimized"])
    max_heap_live_ratios = _max_heap_live_ratios(depth, updates)
    return {
        "workload": workload,
        "depth_per_side": depth,
        "envelope_count": len(updates),
        "update_count": update_count,
        "repeats": repeats,
        "elapsed_seconds": {
            "copy_on_write_reference": results["copy_on_write_reference"],
            "optimized": results["optimized"],
        },
        "median_seconds": {
            "copy_on_write_reference": reference_median,
            "optimized": optimized_median,
        },
        "median_envelopes_per_second": {
            "copy_on_write_reference": len(updates) / reference_median,
            "optimized": len(updates) / optimized_median,
        },
        "median_updates_per_second": {
            "copy_on_write_reference": update_count / reference_median,
            "optimized": update_count / optimized_median,
        },
        "speedup": reference_median / optimized_median,
        "max_heap_live_ratios": max_heap_live_ratios,
        "final": final_state,
    }


def _wire_frame(channel: str, timestamp: datetime, sequence: int, *,
                updates: list[dict[str, Any]] | None = None, event_type: str = "update") -> bytes:
    stamp = timestamp.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if channel == "l2_data":
        events = [{
            "type": event_type,
            "product_id": "BTC-USD",
            "updates": [
                {
                    "side": update["side"],
                    "event_time": stamp,
                    "price_level": str(update["price"]),
                    "new_quantity": str(update["quantity"]),
                }
                for update in (updates or [])
            ],
        }]
    else:
        events = [{"current_time": stamp, "heartbeat_counter": str(sequence)}]
    return json.dumps({
        "channel": channel,
        "timestamp": stamp,
        "sequence_num": sequence,
        "events": events,
    }, separators=(",", ":")).encode("utf-8")


def _canonical_frames(*, depth: int, frame_count: int) -> tuple[list[bytes], int, int]:
    base = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    snapshot_bids, snapshot_asks = _levels(depth)
    frames = [_wire_frame("l2_data", base, 1, updates=[
        *({"side": "bid", "price": row["price"], "quantity": row["quantity"]} for row in snapshot_bids),
        *({"side": "offer", "price": row["price"], "quantity": row["quantity"]} for row in snapshot_asks),
    ], event_type="snapshot")]
    clear_frame_index = max(1, int(frame_count * 0.4))
    refill_frame_index = clear_frame_index + 1
    sequence = 1
    for index in range(1, frame_count):
        sequence += 1
        timestamp = base + timedelta(microseconds=index * 1000)
        if index == clear_frame_index:
            updates = [
                {"side": "bid", "price": row["price"], "quantity": 0.0}
                for row in snapshot_bids
            ]
            frames.append(_wire_frame("l2_data", timestamp, sequence, updates=updates))
        elif index == refill_frame_index:
            frames.append(_wire_frame("l2_data", timestamp, sequence, updates=[
                {"side": "bid", "price": 49_999.0, "quantity": 1.0},
            ]))
        elif index % 10 == 0:
            frames.append(_wire_frame("heartbeats", timestamp, sequence))
        else:
            side = "bid" if index % 2 else "offer"
            price = 49_999.0 if side == "bid" else 100_000.0
            frames.append(_wire_frame("l2_data", timestamp, sequence, updates=[
                {"side": side, "price": price, "quantity": 1.0 + (index % 7) / 10.0},
            ]))
    return frames, clear_frame_index, refill_frame_index


def _canonical_result(
    *,
    depth: int,
    frame_count: int,
    repeats: int,
    implementation: str,
    frames_file: Path | None = None,
    frames_output: Path | None = None,
) -> dict[str, Any]:
    from btc_quarter_hour_engine.market_data import boundary_observations, replay

    original_replay_book = replay.Level2OrderBook
    original_observation_book = boundary_observations.Level2OrderBook
    if implementation == "copy-on-write-reference":
        book_type = CopyOnWriteReferenceBook
    else:
        book_type = Level2OrderBook
    rebuild_transitions: list[tuple[str, str]] = []

    class TrackingBook(book_type):
        def apply_updates(self, updates: Any) -> None:
            previous_state = self.state.value
            super().apply_updates(updates)
            current_state = self.state.value
            if previous_state != current_state:
                rebuild_transitions.append((previous_state, current_state))

    replay.Level2OrderBook = TrackingBook
    boundary_observations.Level2OrderBook = TrackingBook
    base = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    frames, clear_frame_index, refill_frame_index = (
        ([line for line in frames_file.read_bytes().splitlines() if line], max(1, int(frame_count * 0.4)),
         max(1, int(frame_count * 0.4)) + 1)
        if frames_file is not None
        else _canonical_frames(depth=depth, frame_count=frame_count)
    )
    if len(frames) != frame_count:
        raise ValueError(f"Expected {frame_count} frames in workload, found {len(frames)}")
    if frames_output is not None:
        frames_output.write_bytes(b"\n".join(frames) + b"\n")

    try:
        consume_results: list[float] = []
        finalize_results: list[float] = []
        observations: list[Any] = []
        final_book: Any = None
        for _ in range(repeats):
            accumulator = CanonicalReplayAccumulator(
                product_id="BTC-USD", session_id="benchmark", heartbeat_timeout_seconds=30,
            )
            consume_started = time.perf_counter_ns()
            for index, raw in enumerate(frames):
                accumulator.consume(raw, "benchmark-connection", index, base)
            consume_results.append((time.perf_counter_ns() - consume_started) / 1_000_000_000)
            finalize_started = time.perf_counter_ns()
            observations = accumulator.finalize(
                session_started_at_utc=base,
                session_completed_at_utc=base + timedelta(seconds=frame_count),
                derived_at_utc=base + timedelta(seconds=frame_count),
            )
            finalize_results.append((time.perf_counter_ns() - finalize_started) / 1_000_000_000)
            final_book = accumulator._connections["benchmark-connection"].book
    finally:
        replay.Level2OrderBook = original_replay_book
        boundary_observations.Level2OrderBook = original_observation_book

    expected_transitions = [("SYNCED", "REBUILDING"), ("REBUILDING", "SYNCED")]
    if not all(transition in rebuild_transitions for transition in expected_transitions):
        raise RuntimeError(f"Canonical replay did not complete the rebuild episode: {rebuild_transitions!r}")
    consume_median = statistics.median(consume_results)
    finalize_median = statistics.median(finalize_results)
    return {
        "implementation": implementation,
        **_runtime_metadata(),
        "frame_count": len(frames),
        "depth_per_side": depth,
        "clear_frame_index": clear_frame_index,
        "refill_frame_index": refill_frame_index,
        "clear_update_count": depth,
        "refill_update_count": 1,
        "rebuild_transitions": rebuild_transitions,
        "repeats": repeats,
        "consume_seconds": consume_results,
        "consume_median_seconds": consume_median,
        "consume_frames_per_second": len(frames) / consume_median,
        "finalize_seconds": finalize_results,
        "finalize_median_seconds": finalize_median,
        "final_boundary_candidate_count": len(observations),
        "observation_sha256": hashlib.sha256(repr(observations).encode("utf-8")).hexdigest(),
        "final_live_bids": len(final_book.bids) if final_book is not None else 0,
        "final_live_asks": len(final_book.asks) if final_book is not None else 0,
        "final_bid_heap": len(getattr(final_book, "_bid_heap", ())),
        "final_ask_heap": len(getattr(final_book, "_ask_heap", ())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    book = subparsers.add_parser("order-book")
    book.add_argument("--depth-per-side", type=int, default=20_000)
    book.add_argument("--envelopes", type=int, default=25_000)
    book.add_argument("--updates-per-envelope", type=int, default=2)
    book.add_argument("--seed", type=int, default=2026)
    book.add_argument("--repeats", type=int, default=3)
    book.add_argument("--workload", choices=("small-updates", "top-churn", "deep-churn", "rebuild"),
                      default="small-updates")
    book.add_argument("--json-output", type=Path)
    canonical = subparsers.add_parser("canonical-replay")
    canonical.add_argument("--depth-per-side", type=int, default=10_000)
    canonical.add_argument("--frames", type=int, default=20_000)
    canonical.add_argument("--repeats", type=int, default=3)
    canonical.add_argument("--implementation",
                           choices=("optimized", "copy-on-write-reference", "both"),
                           default="optimized")
    canonical.add_argument("--frames-file", type=Path)
    canonical.add_argument("--frames-output", type=Path)
    canonical.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.mode == "order-book":
        result = _book_result(
            args.workload, depth=args.depth_per_side, envelopes=args.envelopes,
            updates_per_envelope=args.updates_per_envelope, seed=args.seed, repeats=args.repeats,
        )
    else:
        implementations = (
            ("copy-on-write-reference", "optimized")
            if args.implementation == "both" else (args.implementation,)
        )
        results = [
            _canonical_result(
                depth=args.depth_per_side, frame_count=args.frames, repeats=args.repeats,
                implementation=implementation,
                frames_file=args.frames_file,
                frames_output=args.frames_output,
            )
            for implementation in implementations
        ]
        if len(results) == 2:
            if (
                results[0]["final_live_bids"] != results[1]["final_live_bids"]
                or results[0]["final_live_asks"] != results[1]["final_live_asks"]
                or results[0]["final_boundary_candidate_count"] != results[1]["final_boundary_candidate_count"]
                or results[0]["observation_sha256"] != results[1]["observation_sha256"]
            ):
                raise RuntimeError("Canonical replay benchmark produced different scientific output")
            results[1]["speedup_vs_reference"] = (
                results[0]["consume_median_seconds"] / results[1]["consume_median_seconds"]
            )
        result = results[0] if len(results) == 1 else results
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output is not None:
        args.json_output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
