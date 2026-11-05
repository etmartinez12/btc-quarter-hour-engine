from __future__ import annotations

import hashlib
import sys
from datetime import timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from benchmark_prospective_source_bridge import (
    _book_frames,
    _latencies,
    run_populated_book_benchmark,
)
from btc_quarter_hour_engine.acquisition.prospective_source_bridge import ProspectiveSourceReceiver
from btc_quarter_hour_engine.market_data.replay import CanonicalReplayAccumulator
from btc_quarter_hour_engine.research.prospective_source import ProspectiveRawFrameJournal
from tests.test_prospective_source_bridge import _frame_header, _producer_health


@pytest.mark.parametrize("levels", [10_000, 25_000, 40_000])
@pytest.mark.parametrize("updates", [125, 130])
def test_populated_checkpoint_plus_tail_has_exact_replay_parity(tmp_path, levels, updates):
    frames = list(_book_frames(levels, updates, 10, 100))
    journal = ProspectiveRawFrameJournal(tmp_path / "book.sqlite")
    receiver = ProspectiveSourceReceiver(journal)
    reference = CanonicalReplayAccumulator(product_id="BTC-USD", session_id="session-1", heartbeat_timeout_seconds=5.0)
    checkpoint_calls = 0
    original = journal._store_canonical_checkpoint

    def count_checkpoint(**kwargs):
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        original(**kwargs)

    journal._store_canonical_checkpoint = count_checkpoint
    for index, (received, raw) in enumerate(frames):
        receiver.process(_frame_header(raw, index=index, received=received, health=_producer_health()), raw)
        reference.consume(raw, "connection-0", index, received, enforce_live_limits=True)
    assert checkpoint_calls == 1 + (len(frames) - 1) // 128
    assert len(reference._connections["connection-0"].book.bids) + len(reference._connections["connection-0"].book.asks) == levels
    expected = reference.to_checkpoint()
    journal.close()
    journal = ProspectiveRawFrameJournal(tmp_path / "book.sqlite")
    restarted = ProspectiveSourceReceiver(journal)
    recovered = restarted._canonical_replay("session-1", 5.0)
    assert recovered.to_checkpoint() == expected
    assert restarted.metrics()["canonical_tail_frames_recovered"] == (len(frames) - 1) % 128
    boundary = frames[0][0] + timedelta(seconds=2)
    assert recovered.boundary_observation(boundary, derived_at_utc=frames[-1][0]) == reference.boundary_observation(
        boundary, derived_at_utc=frames[-1][0],
    )
    assert recovered.boundary_observation(boundary, derived_at_utc=frames[-1][0]).canonical_target_eligible
    journal.close()


@pytest.mark.parametrize("corruption", [
    "raw", "raw_and_digest", "timestamp", "connection", "prefix", "health",
    "missing_frame", "missing_health", "anchor_identity", "missing_anchor_predecessor",
])
def test_corrupt_checkpoint_tail_cannot_ack_or_heal_source(tmp_path, corruption):
    journal = ProspectiveRawFrameJournal(tmp_path / "book.sqlite")
    receiver = ProspectiveSourceReceiver(journal, checkpoint_interval_frames=3)
    frames = list(_book_frames(100, 6, 2, 100))
    for index, (received, raw) in enumerate(frames[:5]):
        receiver.process(_frame_header(raw, index=index, received=received, health=_producer_health()), raw)
    expected_count = 5
    if corruption in {"raw", "raw_and_digest", "timestamp", "connection", "missing_frame", "anchor_identity"}:
        journal._connection.execute("DROP TRIGGER source_frames_immutable_update")
        journal._connection.execute("DROP TRIGGER source_frames_immutable_delete")
        if corruption == "raw":
            journal._connection.execute("UPDATE source_frames SET raw_bytes = X'00' WHERE frame_index = 4")
        elif corruption == "raw_and_digest":
            journal._connection.execute("UPDATE source_frames SET raw_bytes = ?, raw_frame_sha256 = ? WHERE frame_index = 4", (b"changed", hashlib.sha256(b"changed").hexdigest()))
        elif corruption in {"connection", "anchor_identity"}:
            journal._connection.execute("UPDATE source_frames SET connection_id = 'forged' WHERE frame_index = ?", (3 if corruption == "anchor_identity" else 4,))
        elif corruption == "timestamp":
            journal._connection.execute("UPDATE source_frames SET ingest_time_utc = '2026-01-01T00:15:02.000000Z' WHERE frame_index = 4")
        else:
            journal._connection.execute("PRAGMA foreign_keys = OFF")
            journal._connection.execute("DELETE FROM source_frames WHERE frame_index = 4")
            expected_count -= 1
    elif corruption in {"prefix", "missing_anchor_predecessor"}:
        journal._connection.execute("DROP TRIGGER source_prefix_commitments_immutable_update")
        journal._connection.execute("DROP TRIGGER source_prefix_commitments_immutable_delete")
        if corruption == "prefix":
            journal._connection.execute("UPDATE source_prefix_commitments SET commitment_sha256 = ? WHERE frame_index = 4", ("0" * 64,))
        else:
            journal._connection.execute("DELETE FROM source_prefix_commitments WHERE frame_index = 2")
    else:
        journal._connection.execute("DROP TRIGGER source_frame_health_immutable_update")
        journal._connection.execute("DROP TRIGGER source_frame_health_immutable_delete")
        if corruption == "health":
            journal._connection.execute("UPDATE source_frame_health SET producer_health_json = '{}' WHERE frame_index = 4")
        else:
            journal._connection.execute("DELETE FROM source_frame_health WHERE frame_index = 4")
    journal.close()
    journal = ProspectiveRawFrameJournal(tmp_path / "book.sqlite")
    restarted = ProspectiveSourceReceiver(journal, checkpoint_interval_frames=3)
    received, raw = frames[5]
    with pytest.raises(ValueError, match="checkpoint"):
        restarted.process(_frame_header(raw, index=5, received=received, health=_producer_health()), raw)
    assert journal._connection.execute("SELECT COUNT(*) FROM source_frames").fetchone()[0] == expected_count
    assert journal._connection.execute("SELECT frame_index FROM source_canonical_checkpoints").fetchone()[0] == 3
    journal.close()


def test_checkpoint_refresh_failure_recovers_exact_last_acknowledged_tail(tmp_path, monkeypatch):
    journal = ProspectiveRawFrameJournal(tmp_path / "book.sqlite")
    receiver = ProspectiveSourceReceiver(journal, checkpoint_interval_frames=3)
    frames = list(_book_frames(1000, 4, 10, 100))
    for index, (received, raw) in enumerate(frames[:3]):
        receiver.process(_frame_header(raw, index=index, received=received, health=_producer_health()), raw)
    expected = receiver._canonical_replays["session-1"].to_checkpoint()
    original = journal._store_canonical_checkpoint

    def fail_after_store(**kwargs):
        original(**kwargs)
        raise OSError("interrupted FULL checkpoint transaction")

    monkeypatch.setattr(journal, "_store_canonical_checkpoint", fail_after_store)
    received, raw = frames[3]
    with pytest.raises(OSError, match="interrupted"):
        receiver.process(_frame_header(raw, index=3, received=received, health=_producer_health()), raw)
    monkeypatch.setattr(journal, "_store_canonical_checkpoint", original)
    assert receiver._canonical_replay("session-1", 5).to_checkpoint() == expected
    assert journal._connection.execute("SELECT COUNT(*) FROM source_frames").fetchone()[0] == 3
    assert journal._connection.execute("SELECT frame_index FROM source_canonical_checkpoints").fetchone()[0] == 0
    assert receiver.process(_frame_header(raw, index=3, received=received, health=_producer_health()), raw)["durable"]
    assert journal._connection.execute("SELECT frame_index FROM source_canonical_checkpoints").fetchone()[0] == 3
    journal.close()


def test_checkpoint_tail_byte_limit_forces_atomic_refresh(tmp_path, monkeypatch):
    import btc_quarter_hour_engine.acquisition.prospective_source_bridge as bridge
    import btc_quarter_hour_engine.research.prospective_source as source

    monkeypatch.setattr(bridge, "MAX_CANONICAL_CHECKPOINT_TAIL_BYTES", 1000)
    monkeypatch.setattr(source, "MAX_CANONICAL_CHECKPOINT_TAIL_BYTES", 1000)
    journal = ProspectiveRawFrameJournal(tmp_path / "book.sqlite")
    receiver = ProspectiveSourceReceiver(journal)
    frames = list(_book_frames(100, 3, 10, 100))
    for index, (received, raw) in enumerate(frames):
        receiver.process(_frame_header(raw, index=index, received=received, health=_producer_health()), raw)
        _, size = journal._canonical_tail_extent("session-1", index)
        assert size <= 1000
    assert journal._connection.execute("SELECT frame_index FROM source_canonical_checkpoints").fetchone()[0] == len(frames) - 1
    journal.close()


@pytest.mark.parametrize("kind", ["frames", "bytes"])
def test_out_of_bound_tail_is_rejected_before_replay_allocation(tmp_path, monkeypatch, kind):
    import btc_quarter_hour_engine.research.prospective_source as source

    journal = ProspectiveRawFrameJournal(tmp_path / "book.sqlite")
    receiver = ProspectiveSourceReceiver(journal)
    received, raw = next(_book_frames(100, 1, 2, 100))
    receiver.process(_frame_header(raw, index=0, received=received, health=_producer_health()), raw)
    limit = 130 if kind == "frames" else 2
    for index in range(1, limit):
        journal.append_frame(
            session_id="session-1", connection_id="connection-0", frame_index=index,
            ingest_time_utc=received + timedelta(milliseconds=index),
            raw_bytes=b"x" if kind == "frames" else b"x" * (source.MAX_CANONICAL_CHECKPOINT_TAIL_BYTES + 1),
            reconnect_epoch=0, producer_health=_producer_health(),
        )

    def forbid_state_load(_state):
        raise AssertionError("Out-of-bound recovery allocated a replay state")

    monkeypatch.setattr(CanonicalReplayAccumulator, "from_checkpoint", forbid_state_load)
    with pytest.raises(ValueError, match="tail .*limit"):
        ProspectiveSourceReceiver(journal)._canonical_replay("session-1", 5.0)
    journal.close()


@pytest.mark.parametrize("interval", [0, 129, True])
def test_checkpoint_interval_cannot_expand_recovery_bound(tmp_path, interval):
    journal = ProspectiveRawFrameJournal(tmp_path / "book.sqlite")
    with pytest.raises(ValueError, match="bounded tail"):
        ProspectiveSourceReceiver(journal, checkpoint_interval_frames=interval)
    journal.close()


def test_real_collector_publisher_receiver_populated_book_metrics_and_recovery():
    result = run_populated_book_benchmark(levels=1000, updates=8, rate=10, burst_size=2)
    assert result["collector_termination"] == "max_messages"
    assert result["collector_populated_levels"] == 1000
    assert result["collector_process"]["samples"] == 8
    assert result["publisher"]["valid"], result["publisher"]["failure_reason"]
    assert result["checkpoint"]["receiver"]["frames_committed"] == 10
    assert result["checkpoint"]["receiver"]["fences_committed"] == 1
    assert result["checkpoint"]["sqlite_synchronous"] == 2
    assert result["checkpoint"]["warm_recovery_identical"]
    assert result["checkpoint"]["cold_recovery"]["identical_state"]
    assert result["durable_frame_ack"]["samples"] == 8
    assert result["checkpoint"]["checkpoint_bytes"] > 10_000
    assert result["collector_resources"]["rss_peak_bytes"] >= result["collector_resources"]["rss_baseline_bytes"]


def test_latency_percentiles_and_absent_samples_are_explicit():
    assert _latencies([1, 2, 3, 4, 5]) == {
        "samples": 5, "median_ms": 3, "p95_ms": 4.8, "p99_ms": 4.96, "max_ms": 5,
    }
    assert _latencies([])["median_ms"] is None
