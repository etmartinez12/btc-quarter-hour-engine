from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from btc_quarter_hour_engine.research.prospective_source import (
    CausalInputError,
    ProspectiveRawFrameJournal,
    RawFrameRecord,
    RawFrameSnapshot,
    build_causal_feature_vector,
)
from btc_quarter_hour_engine.acquisition.prospective_source_bridge import (
    ClockHealthEvidence,
    ProspectiveSourceReceiver,
)
from btc_quarter_hour_engine.market_data.replay import CanonicalReplayAccumulator
from btc_quarter_hour_engine.research.model_freeze import OFFICIAL_STATUS, frozen_predictors
from btc_quarter_hour_engine.research.prospective_contract import (
    FULL_V2_MODEL_SHA256,
    RANGE_MODEL_SHA256,
    TRUSTED_MANIFEST_SHA256,
)
from btc_quarter_hour_engine.research.prospective_inference import (
    FrozenModelBundle,
    process_incremental_boundary,
)
from btc_quarter_hour_engine.research import prospective_inference as inference
from btc_quarter_hour_engine.research import prospective_replay as replay_module
from btc_quarter_hour_engine.research.prospective_replay import (
    FeatureReplayCheckpoint,
    FeatureReplayWarmup,
    _checkpoint_digest,
    _tail_commitments,
    advance_feature_replay_checkpoint,
    build_incremental_causal_feature_vector,
    create_feature_replay_checkpoint,
    tail_from_snapshot,
    accumulate_feature_replay_warmup,
    finish_feature_replay_warmup,
    initial_prefix_commitment,
)


UTC = timezone.utc
BOUNDARY = datetime(2026, 1, 1, 4, 0, tzinfo=UTC)


def _wire_frame(sequence: int, timestamp: datetime, *, snapshot: bool = False) -> bytes:
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


def _record(index: int, sequence: int, timestamp: datetime, *, connection="connection-1", snapshot=False, raw=None):
    payload = raw if raw is not None else _wire_frame(sequence, timestamp, snapshot=snapshot)
    return RawFrameRecord(
        raw_bytes=payload,
        session_id="replay-session",
        connection_id=connection,
        frame_index=index,
        ingest_time_utc=timestamp,
        raw_frame_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _snapshot(
    *, end: datetime = BOUNDARY + timedelta(minutes=45),
    start: datetime = BOUNDARY - timedelta(hours=4),
) -> RawFrameSnapshot:
    frames = [_record(0, 0, start, snapshot=True)]
    index = 1
    sequence = 1
    timestamp = start + timedelta(minutes=15)
    while timestamp <= end:
        frames.append(_record(index, sequence, timestamp))
        index += 1
        sequence += 1
        frames.append(_record(index, sequence, timestamp))
        index += 1
        sequence += 1
        timestamp += timedelta(minutes=15)
    return RawFrameSnapshot(
        session_id="replay-session",
        session_started_at_utc=start,
        watermark_utc=end,
        clock_health_verified=True,
        frames=tuple(frames),
    )


def _with_boundary_witnesses(snapshot, boundaries):
    boundary_set = set(boundaries)
    frames = []
    sequence = 0
    for position, original in enumerate(snapshot.frames):
        payload = json.loads(original.raw_bytes)
        payload["sequence_num"] = sequence
        raw_bytes = json.dumps(payload).encode()
        frames.append(
            RawFrameRecord(
                raw_bytes=raw_bytes,
                session_id=original.session_id,
                connection_id=original.connection_id,
                frame_index=len(frames),
                ingest_time_utc=original.ingest_time_utc,
                raw_frame_sha256=hashlib.sha256(raw_bytes).hexdigest(),
            )
        )
        sequence += 1
        next_time = (
            snapshot.frames[position + 1].ingest_time_utc
            if position + 1 < len(snapshot.frames)
            else None
        )
        if original.ingest_time_utc in boundary_set and next_time != original.ingest_time_utc:
            heartbeat_time = original.ingest_time_utc - timedelta(seconds=1)
            heartbeat = json.dumps(
                {
                    "channel": "heartbeats",
                    "client_id": "",
                    "timestamp": heartbeat_time.isoformat().replace("+00:00", "Z"),
                    "sequence_num": sequence,
                    "events": [
                        {
                            "current_time": heartbeat_time.isoformat().replace(
                                "+00:00", "Z"
                            ),
                            "heartbeat_counter": str(sequence),
                        }
                    ],
                }
            ).encode()
            frames.append(
                _record(
                    len(frames),
                    sequence,
                    original.ingest_time_utc,
                    raw=heartbeat,
                )
            )
            sequence += 1
            witness_time = original.ingest_time_utc + timedelta(seconds=1)
            frames.append(_record(len(frames), sequence, witness_time))
            sequence += 1
    return replace(
        snapshot,
        watermark_utc=max(boundaries) + timedelta(seconds=1),
        frames=tuple(frames),
    )


def _checkpoint(snapshot: RawFrameSnapshot, boundary=BOUNDARY):
    return create_feature_replay_checkpoint(snapshot, boundary)


def _populate_journal(path, source):
    journal = ProspectiveRawFrameJournal(path)
    journal.start_session(source.session_id, source.session_started_at_utc)
    for frame in source.frames:
        journal.append_frame(
            session_id=frame.session_id, connection_id=frame.connection_id,
            frame_index=frame.frame_index, ingest_time_utc=frame.ingest_time_utc,
            raw_bytes=frame.raw_bytes,
        )
    journal.advance_watermark(source.session_id, source.watermark_utc, clock_health_verified=True)
    return journal


@pytest.mark.parametrize("api", [build_incremental_causal_feature_vector, advance_feature_replay_checkpoint])
@pytest.mark.parametrize("construction", ["direct", "modified"])
def test_public_apis_reject_valid_statistics_with_stale_digest_and_matching_prefix(api, construction):
    source = _snapshot()
    checkpoint = _checkpoint(source)
    tail = tail_from_snapshot(source, checkpoint)
    data = checkpoint.data
    data["replay_state"]["buckets"][-1]["stats"]["spread_bps_sum"] += 1.0
    serialized = json.dumps(data)
    if construction == "direct":
        forged = FeatureReplayCheckpoint(serialized)
    else:
        forged = FeatureReplayCheckpoint(checkpoint.serialized)
        object.__setattr__(forged, "serialized", serialized)
    assert tail.prefix_commitment_sha256 == forged.prefix_commitment_sha256
    with pytest.raises(ValueError, match="integrity digest"):
        api(tail, BOUNDARY + timedelta(minutes=15), forged)


@pytest.mark.parametrize("api", [build_incremental_causal_feature_vector, advance_feature_replay_checkpoint])
def test_public_apis_validate_structure_even_with_recomputed_digest(api):
    source = _snapshot()
    checkpoint = _checkpoint(source)
    tail = tail_from_snapshot(source, checkpoint)
    data = checkpoint.data
    data["replay_state"]["frames_seen"] += 1
    data["integrity_sha256"] = _checkpoint_digest(data)
    with pytest.raises(ValueError, match="frame index and replay count"):
        api(tail, BOUNDARY + timedelta(minutes=15), FeatureReplayCheckpoint(json.dumps(data)))
    with pytest.raises(ValueError, match="Incomplete source tail"):
        api(replace(tail, has_more=True), BOUNDARY + timedelta(minutes=15), checkpoint)


def test_chunked_warmup_is_restartable_bounded_and_matches_full_replay(tmp_path, publication_platform):
    # Eight hours of history, duplicate timestamps, and boundaries split across
    # chunks exercise retention independently of raw frame count.
    start = BOUNDARY - timedelta(hours=8)
    frames = [_record(0, 0, start, snapshot=True)]
    for minute in range(15, 8 * 60 + 16, 15):
        for _ in range(2):
            frames.append(_record(len(frames), len(frames), start + timedelta(minutes=minute)))
    source = RawFrameSnapshot(
        session_id="replay-session", session_started_at_utc=start,
        watermark_utc=BOUNDARY + timedelta(minutes=15),
        clock_health_verified=True, frames=tuple(frames),
    )
    path = tmp_path / "source.sqlite"
    journal = _populate_journal(path, source)
    warmup = None
    chunks = 0
    while True:
        chunk = journal.warmup_chunk(
            source.session_id,
            after_frame_index=-1 if warmup is None else warmup.last_frame_index,
            expected_prefix_commitment_sha256=(
                initial_prefix_commitment(source.session_id, start)
                if warmup is None else warmup.prefix_commitment_sha256
            ),
            through_boundary_utc=BOUNDARY, max_chunk_frames=3,
        )
        previous = warmup
        warmup = accumulate_feature_replay_warmup(chunk, BOUNDARY, previous)
        assert len(warmup.data["replay_state"]["buckets"]) <= 17
        for api in (build_incremental_causal_feature_vector, advance_feature_replay_checkpoint):
            with pytest.raises(ValueError, match="schema version"):
                api(chunk, BOUNDARY + timedelta(minutes=15), warmup)
        if chunks == 0:
            with pytest.raises(ValueError, match="incomplete"):
                finish_feature_replay_warmup(chunk, warmup)
            with pytest.raises(ValueError, match="prefix commitment"):
                accumulate_feature_replay_warmup(replace(chunk, prefix_commitment_sha256="0" * 64), BOUNDARY)
            corrupt = warmup.data
            corrupt["replay_state"]["buckets"][0]["stats"]["spread_bps_sum"] += 1
            with pytest.raises(ValueError, match="integrity digest"):
                accumulate_feature_replay_warmup(
                    chunk, BOUNDARY, FeatureReplayWarmup(json.dumps(corrupt)),
                )
        state_path = tmp_path / f"warmup-{chunks}.json"
        warmup.write_once(state_path)
        warmup = FeatureReplayWarmup.load(state_path)
        journal.close()
        journal = ProspectiveRawFrameJournal(path)
        chunks += 1
        if not chunk.has_more:
            exhausted = journal.warmup_chunk(
                source.session_id, after_frame_index=warmup.last_frame_index,
                expected_prefix_commitment_sha256=warmup.prefix_commitment_sha256,
                through_boundary_utc=BOUNDARY, max_chunk_frames=3,
            )
            assert exhausted.frames == () and exhausted.has_more is False
            with pytest.raises(ValueError, match="cutoff mismatch"):
                finish_feature_replay_warmup(
                    replace(exhausted, through_boundary_utc=BOUNDARY - timedelta(minutes=15)), warmup,
                )
            checkpoint = finish_feature_replay_warmup(exhausted, warmup)
            break
    assert chunks > 10
    assert checkpoint.serialized == create_feature_replay_checkpoint(source, BOUNDARY).serialized
    tail = journal.tail_snapshot(
        source.session_id, after_frame_index=checkpoint.last_frame_index,
        expected_prefix_commitment_sha256=checkpoint.prefix_commitment_sha256,
        through_boundary_utc=source.watermark_utc,
    )
    actual = build_incremental_causal_feature_vector(tail, source.watermark_utc, checkpoint)
    expected = build_causal_feature_vector(source, source.watermark_utc)
    assert actual.values == expected.values
    assert actual.feature_fingerprint == expected.feature_fingerprint


def _populate_receiver_journal(path, source, boundaries, *, clock_verified=True):
    journal = ProspectiveRawFrameJournal(path)
    clock_values = iter(
        (
            ClockHealthEvidence(
                observed_at_utc=boundary + timedelta(seconds=2),
                synchronized=clock_verified,
                verified=clock_verified,
                offset_seconds=0.001 if clock_verified else None,
                reference_time_utc=boundary + timedelta(seconds=1) if clock_verified else None,
                leap_status="Normal" if clock_verified else None,
                source="synthetic test evidence",
            )
            for boundary in boundaries
        )
    )
    receiver = ProspectiveSourceReceiver(
        journal, clock_health_provider=lambda: next(clock_values)
    )
    canonical_replay = CanonicalReplayAccumulator(
        product_id="BTC-USD",
        session_id=source.session_id,
        heartbeat_timeout_seconds=10.0,
    )
    for frame in source.frames:
        canonical_replay.consume(
            frame.raw_bytes,
            frame.connection_id,
            frame.frame_index,
            frame.ingest_time_utc,
        )
        current_boundary = frame.ingest_time_utc.replace(
            minute=frame.ingest_time_utc.minute
            - frame.ingest_time_utc.minute % 15,
            second=0,
            microsecond=0,
        )
        canonical_boundary = None
        finalized = canonical_replay.boundary_is_finalized(current_boundary)
        observation = None
        if finalized:
            observation = next(
                (
                    candidate
                    for candidate in canonical_replay.finalize(
                        session_started_at_utc=source.session_started_at_utc,
                        session_completed_at_utc=frame.ingest_time_utc,
                        derived_at_utc=frame.ingest_time_utc,
                        connections=[{"connection_id": frame.connection_id}],
                    )
                    if candidate.boundary_time_utc == current_boundary
                ),
                None,
            )
        if observation is not None:
            canonical_boundary = {
                "boundary_utc": observation.boundary_time_utc.isoformat(
                    timespec="microseconds"
                ).replace("+00:00", "Z"),
                "canonical_target_eligible": observation.canonical_target_eligible,
                "eligibility_reason": observation.eligibility_reason,
                "book_synced": observation.book_synced,
                "last_heartbeat_at_utc": (
                    observation.last_heartbeat_at_utc.isoformat(
                        timespec="microseconds"
                    ).replace("+00:00", "Z")
                    if observation.last_heartbeat_at_utc is not None
                    else None
                ),
                "heartbeat_timeout_seconds": 10.0,
                "connection_id": observation.connection_id,
                "source_sequence_num": observation.source_sequence_num,
            }
        source_sequence = json.loads(frame.raw_bytes).get("sequence_num")
        health = {
            "publisher_valid": True,
            "post_frame_health": {
                "heartbeat_seen": False,
                "heartbeat_last_at_utc": None,
                "heartbeat_healthy": False,
                "order_book_synchronized": False,
                "connection_id": frame.connection_id,
                "connection_last_sequence_num": source_sequence,
                "last_l2_sequence_num": source_sequence,
                "sequence_gap_count": 0,
                "stale_sequence_count": 0,
                "heartbeat_timeout_count": 0,
                "heartbeat_discontinuity_count": 0,
                "malformed_frame_count": 0,
                "malformed_level2_count": 0,
                "crossed_book_count": 0,
                "heartbeat_timeout_seconds": 10.0,
            },
            "canonical_boundary_finalized": observation is not None,
            "canonical_boundary": canonical_boundary,
        }
        receiver.process(
            {
                "version": 1,
                "kind": "frame",
                "session_id": source.session_id,
                "session_started_at_utc": source.session_started_at_utc.isoformat().replace(
                    "+00:00", "Z"
                ),
                "connection_id": frame.connection_id,
                "reconnect_epoch": 0,
                "connection_first_frame_index": 0,
                "frame_index": frame.frame_index,
                "ingest_time_utc": frame.ingest_time_utc.isoformat().replace("+00:00", "Z"),
                "raw_sha256": frame.raw_frame_sha256,
                "raw_length": len(frame.raw_bytes),
                "producer_health": health,
            },
            frame.raw_bytes,
        )
        if canonical_boundary is not None and frame.ingest_time_utc in {
            boundary + timedelta(seconds=1) for boundary in boundaries
        }:
            boundary = current_boundary
            receiver.process(
                {
                    "version": 1,
                    "kind": "fence",
                    "session_id": source.session_id,
                    "watermark_utc": boundary.isoformat().replace("+00:00", "Z"),
                    "through_frame_index": frame.frame_index,
                    "connection_id": frame.connection_id,
                    "reconnect_epoch": 0,
                    "producer_health": health,
                    "raw_length": 0,
                },
                b"",
            )
    return journal


def test_receiver_journal_warmup_and_incremental_vector_match_full_replay(tmp_path):
    target_boundary = BOUNDARY + timedelta(minutes=15)
    source = _with_boundary_witnesses(
        _snapshot(end=target_boundary), (BOUNDARY, target_boundary)
    )
    journal = _populate_receiver_journal(
        tmp_path / "source.sqlite", source, (BOUNDARY, target_boundary)
    )
    warmup_chunk = journal.warmup_chunk(
        source.session_id,
        after_frame_index=-1,
        expected_prefix_commitment_sha256=initial_prefix_commitment(
            source.session_id, source.session_started_at_utc
        ),
        through_boundary_utc=BOUNDARY,
    )
    warmup = accumulate_feature_replay_warmup(warmup_chunk, BOUNDARY)
    exhausted = journal.warmup_chunk(
        source.session_id,
        after_frame_index=warmup.last_frame_index,
        expected_prefix_commitment_sha256=warmup.prefix_commitment_sha256,
        through_boundary_utc=BOUNDARY,
    )
    finish_feature_replay_warmup(exhausted, warmup)

    checkpoint = create_feature_replay_checkpoint(
        journal.snapshot_through(source.session_id, boundary_utc=BOUNDARY), BOUNDARY
    )
    tail = journal.tail_snapshot(
        source.session_id,
        after_frame_index=checkpoint.last_frame_index,
        expected_prefix_commitment_sha256=checkpoint.prefix_commitment_sha256,
        through_boundary_utc=target_boundary,
    )
    incremental = build_incremental_causal_feature_vector(tail, target_boundary, checkpoint)
    full_snapshot = journal.snapshot_through(source.session_id, boundary_utc=target_boundary)
    full = build_causal_feature_vector(full_snapshot, target_boundary)

    assert incremental.candidate_predictors == full.candidate_predictors
    assert incremental.control_predictors == full.control_predictors
    assert len(incremental.candidate_predictors) == 12
    assert len(incremental.control_predictors) == 91
    assert incremental.values == full.values
    assert incremental.feature_fingerprint == full.feature_fingerprint
    journal.close()


def test_warmup_cannot_promote_missing_exact_240_minute_history(tmp_path):
    source = _snapshot()
    # Registered session is old enough, but first bucket is absent.
    frames = tuple(replace(frame, frame_index=i) for i, frame in enumerate(source.frames[1:]))
    frames = (replace(frames[0], raw_bytes=_wire_frame(1, frames[0].ingest_time_utc, snapshot=True),
                      raw_frame_sha256=hashlib.sha256(_wire_frame(1, frames[0].ingest_time_utc, snapshot=True)).hexdigest()), *frames[1:])
    source = replace(source, frames=frames)
    journal = _populate_journal(tmp_path / "missing.sqlite", source)
    chunk = journal.warmup_chunk(
        source.session_id, after_frame_index=-1,
        expected_prefix_commitment_sha256=initial_prefix_commitment(source.session_id, source.session_started_at_utc),
        through_boundary_utc=BOUNDARY,
    )
    warmup = accumulate_feature_replay_warmup(chunk, BOUNDARY)
    exhausted = replace(chunk, frames=(), frame_commitments=(),
                        prefix_frame_index=warmup.last_frame_index,
                        prefix_commitment_sha256=warmup.prefix_commitment_sha256)
    with pytest.raises(ValueError, match="exactly 240-minute"):
        finish_feature_replay_warmup(exhausted, warmup)
    journal.close()


def test_journal_migration_is_explicit_atomic_and_not_repeated(tmp_path, monkeypatch):
    path = tmp_path / "legacy.sqlite"
    journal = _populate_journal(path, _snapshot())
    journal.close()
    connection = sqlite3.connect(path)
    connection.execute("DROP TABLE source_prefix_schema")
    connection.execute("DROP TABLE source_prefix_commitments")
    connection.close()
    with pytest.raises(ValueError, match="Legacy prefix migration required"):
        ProspectiveRawFrameJournal(path)
    journal = ProspectiveRawFrameJournal(path, migrate_legacy_prefixes=True)
    checkpoint = _checkpoint(_snapshot())
    assert journal.tail_snapshot(
        checkpoint.session_id, after_frame_index=checkpoint.last_frame_index,
        expected_prefix_commitment_sha256=checkpoint.prefix_commitment_sha256,
        through_boundary_utc=BOUNDARY + timedelta(minutes=15),
    ).frames
    journal.close()
    queries = []
    real_connect = sqlite3.connect

    def traced_connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        connection.set_trace_callback(queries.append)
        return connection

    monkeypatch.setattr(sqlite3, "connect", traced_connect)
    for _ in range(3):
        journal = ProspectiveRawFrameJournal(path)
        journal.close()
    reads = [query.lower() for query in queries if query.lstrip().lower().startswith("select")]
    assert reads
    assert not any("source_frames" in query or "source_prefix_commitments" in query for query in reads)


@pytest.mark.parametrize("fault", ["file_sync", "link", "directory_sync"])
def test_checkpoint_publication_faults_fail_closed_and_recover(tmp_path, monkeypatch, publication_platform, fault):
    checkpoint = _checkpoint(_snapshot())
    destination = tmp_path / "durable.json"
    real_fsync, real_link = os.fsync, os.link
    events = []
    linked = False

    def sync(fd):
        events.append("directory_sync" if linked else "file_sync")
        if fault == ("directory_sync" if linked else "file_sync"):
            raise OSError("injected sync failure")
        # Directory preflight on POSIX is separately tested below. fsync calls
        # here are fault simulations so read-only Windows recovery is portable.

    def link(source, target):
        nonlocal linked
        events.append("link")
        if fault == "link":
            raise OSError("injected link failure")
        real_link(source, target)
        linked = True

    # Avoid injecting the failure into the preflight probe.
    if os.name == "posix":
        monkeypatch.setattr(replay_module, "_open_durable_directory",
                            lambda path: os.open(path, os.O_RDONLY | os.O_DIRECTORY))
    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(os, "link", link)
    with pytest.raises(OSError, match="injected"):
        checkpoint.write_once(destination)
    assert not list(tmp_path.glob(".durable.json.*.tmp"))
    if fault != "directory_sync":
        assert not destination.exists()
    else:
        assert events == ["file_sync", "link", "directory_sync"]
        assert FeatureReplayCheckpoint.load(destination) == checkpoint
        monkeypatch.setattr(os, "fsync", lambda fd: None)
        with pytest.raises(FileExistsError):
            checkpoint.write_once(destination)
        assert FeatureReplayCheckpoint.load_durable(destination) == checkpoint
    monkeypatch.setattr(os, "fsync", real_fsync)
    monkeypatch.setattr(os, "link", real_link)


def test_partial_or_orphaned_checkpoint_files_are_never_recovered_as_ready(tmp_path, publication_platform):
    checkpoint = _checkpoint(_snapshot())
    destination = tmp_path / "interrupted.json"
    orphan = tmp_path / ".interrupted.json.abandoned.tmp"
    orphan.write_text(checkpoint.serialized[:100], encoding="utf-8")
    assert not destination.exists()
    checkpoint.write_once(destination)
    assert FeatureReplayCheckpoint.load(destination) == checkpoint
    with pytest.raises(ValueError):
        FeatureReplayCheckpoint.load(orphan)
    with pytest.raises(FileExistsError):
        checkpoint.write_once(destination)
    assert orphan.exists()  # Never delete evidence owned by another writer.


def test_publication_requires_supported_directory_sync(tmp_path, monkeypatch):
    checkpoint = _checkpoint(_snapshot())
    destination = tmp_path / "unsupported.json"
    if os.name != "posix":
        with pytest.raises(OSError, match="requires POSIX"):
            checkpoint.write_once(destination)
    else:
        monkeypatch.setattr(os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("unsupported sync")))
        with pytest.raises(OSError, match="unsupported sync"):
            checkpoint.write_once(destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.skipif(os.name != "posix", reason="Real directory fsync requires POSIX; Windows simulations are not durability evidence")
def test_real_posix_checkpoint_publication_and_recovery(tmp_path):
    checkpoint = _checkpoint(_snapshot())
    destination = tmp_path / "posix.json"
    checkpoint.write_once(destination)
    assert FeatureReplayCheckpoint.load_durable(destination) == checkpoint


def test_interrupted_write_never_publishes_partial_checkpoint(tmp_path, monkeypatch, publication_platform):
    checkpoint = _checkpoint(_snapshot())
    destination = tmp_path / "short-write.json"
    original = os.fdopen

    class InterruptedStream:
        def __init__(self, fd, mode):
            self.stream = original(fd, mode)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def write(self, data):
            self.stream.write(data[:64])
            self.stream.flush()
            raise InterruptedError("interrupted checkpoint write")

    with monkeypatch.context() as patch:
        patch.setattr(os, "fdopen", InterruptedStream)
        with pytest.raises(InterruptedError, match="interrupted checkpoint"):
            checkpoint.write_once(destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".short-write.json.*.tmp"))
    checkpoint.write_once(destination)
    assert FeatureReplayCheckpoint.load(destination) == checkpoint


def test_concurrent_checkpoint_publishers_cannot_overwrite_winner(tmp_path, monkeypatch, publication_platform):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    first = _checkpoint(_snapshot())
    second = _checkpoint(_snapshot(), BOUNDARY + timedelta(minutes=15))
    destination = tmp_path / "race.json"
    barrier = Barrier(2)
    original_link = os.link

    def competing_link(source, target):
        barrier.wait(timeout=10)
        original_link(source, target)

    monkeypatch.setattr(os, "link", competing_link)

    def publish(checkpoint):
        try:
            checkpoint.write_once(destination)
            return checkpoint
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(publish, (first, second)))
    winners = [outcome for outcome in outcomes if outcome is not None]
    assert len(winners) == 1
    assert FeatureReplayCheckpoint.load(destination) == winners[0]
    assert not list(tmp_path.glob(".race.json.*.tmp"))


def test_legacy_migration_failure_rolls_back_progress_and_marker(tmp_path):
    path = tmp_path / "failed-migration.sqlite"
    source = _snapshot()
    _populate_journal(path, source).close()
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE source_prefix_schema")
        connection.execute("DROP TABLE source_prefix_commitments")
        connection.execute("DROP TRIGGER source_frames_immutable_update")
        connection.execute("UPDATE source_frames SET raw_frame_sha256 = ? WHERE frame_index = 5", ("0" * 64,))
    with pytest.raises(ValueError, match="prefix integrity validation"):
        ProspectiveRawFrameJournal(path, migrate_legacy_prefixes=True)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM source_prefix_schema").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM source_prefix_commitments").fetchone()[0] == 0
        connection.execute("DROP TRIGGER source_frames_immutable_update")
        connection.execute("UPDATE source_frames SET raw_frame_sha256 = ? WHERE frame_index = 5",
                           (source.frames[5].raw_frame_sha256,))
    ProspectiveRawFrameJournal(path, migrate_legacy_prefixes=True).close()
    ProspectiveRawFrameJournal(path).close()


def test_warmup_can_restart_before_initial_book_snapshot(tmp_path, publication_platform):
    source = _snapshot()
    start = source.session_started_at_utc
    subscription = json.dumps({
        "channel": "subscriptions", "timestamp": start.isoformat(),
        "sequence_num": 0, "events": [],
    }).encode()
    frames = [_record(0, 0, start, raw=subscription)]
    for frame in source.frames:
        index = frame.frame_index + 1
        frames.append(_record(index, index, frame.ingest_time_utc, snapshot=index == 1))
    source = replace(source, frames=tuple(frames))
    journal = _populate_journal(tmp_path / "presnapshot.sqlite", source)
    chunk = journal.warmup_chunk(
        source.session_id, after_frame_index=-1,
        expected_prefix_commitment_sha256=initial_prefix_commitment(source.session_id, start),
        through_boundary_utc=BOUNDARY, max_chunk_frames=1,
    )
    warmup = accumulate_feature_replay_warmup(chunk, BOUNDARY)
    state_path = tmp_path / "not-ready.json"
    warmup.write_once(state_path)
    warmup = FeatureReplayWarmup.load(state_path)
    with pytest.raises(ValueError, match="schema version"):
        FeatureReplayCheckpoint.load(state_path)
    tail = journal.warmup_chunk(
        source.session_id, after_frame_index=warmup.last_frame_index,
        expected_prefix_commitment_sha256=warmup.prefix_commitment_sha256,
        through_boundary_utc=BOUNDARY,
    )
    checkpoint = finish_feature_replay_warmup(tail, warmup)
    assert checkpoint.serialized == create_feature_replay_checkpoint(source, BOUNDARY).serialized
    journal.close()


def test_source_chunk_seek_bounds_work_across_large_timestamp_ties(tmp_path):
    source = _snapshot()
    start = source.session_started_at_utc
    source = replace(source, frames=tuple(
        _record(index, index, start, snapshot=index == 0) for index in range(1001)
    ))
    journal = _populate_journal(tmp_path / "ties.sqlite", source)
    costs = []
    for after in (0, 995):
        prefix = journal._connection.execute(
            "SELECT commitment_sha256 FROM source_prefix_commitments WHERE frame_index = ?",
            (after,),
        ).fetchone()[0]
        steps = 0

        def progress():
            nonlocal steps
            steps += 1
            return 0

        journal._connection.set_progress_handler(progress, 1)
        try:
            chunk = journal.warmup_chunk(
                source.session_id, after_frame_index=after,
                expected_prefix_commitment_sha256=prefix,
                through_boundary_utc=BOUNDARY, max_chunk_frames=2,
            )
        finally:
            journal._connection.set_progress_handler(None, 0)
        assert len(chunk.frames) == 2 and chunk.has_more is True
        assert chunk.frames[0].frame_index == after + 1
        costs.append(steps)
    assert max(costs) < 1000
    assert costs[1] <= costs[0] + 100
    journal.close()


@pytest.fixture
def publication_platform(monkeypatch, tmp_path):
    """Windows simulations test ordering/faults, NOT filesystem crash durability."""
    if os.name != "posix":
        sync_target = tmp_path / "directory-fsync-simulation"
        sync_target.touch()
        monkeypatch.setattr(
            replay_module, "_open_durable_directory",
            lambda path: os.open(sync_target, os.O_RDWR),
        )


def test_incremental_replay_matches_full_replay_across_boundaries_and_restart(tmp_path, publication_platform):
    source = _snapshot()
    checkpoint = _checkpoint(source)
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint.write_once(checkpoint_path)
    restored = FeatureReplayCheckpoint.load(checkpoint_path)
    assert restored.serialized == checkpoint.serialized
    with pytest.raises(FileExistsError):
        checkpoint.write_once(checkpoint_path)

    current = restored
    for boundary in (
        BOUNDARY + timedelta(minutes=15),
        BOUNDARY + timedelta(minutes=30),
        BOUNDARY + timedelta(minutes=45),
    ):
        tail = tail_from_snapshot(source, current)
        actual = build_incremental_causal_feature_vector(tail, boundary, current)
        expected = build_causal_feature_vector(source, boundary)
        assert actual.candidate_predictors == expected.candidate_predictors
        assert actual.control_predictors == expected.control_predictors
        assert tuple(actual.values) == tuple(expected.values)
        assert actual.values == expected.values
        assert actual.feature_fingerprint == expected.feature_fingerprint
        assert len(actual.candidate_predictors) == 12
        assert len(actual.control_predictors) == 91
        if boundary > current.cutoff_utc:
            current = advance_feature_replay_checkpoint(tail, boundary, current)
            # Simulate process restart by loading the exact serialized value.
            current = FeatureReplayCheckpoint.from_dict(current.data)


def test_journal_returns_only_a_chain_verified_incremental_tail(tmp_path):
    source = _snapshot()
    checkpoint = _checkpoint(source)
    journal = ProspectiveRawFrameJournal(tmp_path / "source.sqlite")
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
    journal_path = journal.path
    journal.close()
    journal = ProspectiveRawFrameJournal(journal_path)
    bootstrap = journal.snapshot_through(
        source.session_id,
        boundary_utc=BOUNDARY,
    )
    assert create_feature_replay_checkpoint(bootstrap, BOUNDARY).serialized == checkpoint.serialized
    with pytest.raises(ValueError, match="bounded replay limit"):
        journal.snapshot_through(
            source.session_id,
            boundary_utc=BOUNDARY,
            max_frames=1,
        )
    tail = journal.tail_snapshot(
        source.session_id,
        after_frame_index=checkpoint.last_frame_index,
        expected_prefix_commitment_sha256=checkpoint.prefix_commitment_sha256,
        through_boundary_utc=BOUNDARY + timedelta(minutes=15),
    )
    assert len(tail.frames) == 2
    with pytest.raises(ValueError, match="bounded replay limit"):
        journal.tail_snapshot(
            source.session_id,
            after_frame_index=checkpoint.last_frame_index,
            expected_prefix_commitment_sha256=checkpoint.prefix_commitment_sha256,
            through_boundary_utc=BOUNDARY + timedelta(minutes=15),
            max_tail_frames=1,
        )
    assert all(frame.ingest_time_utc <= BOUNDARY + timedelta(minutes=15) for frame in tail.frames)
    before_future = build_incremental_causal_feature_vector(
        tail, BOUNDARY + timedelta(minutes=15), checkpoint
    )
    assert before_future.feature_fingerprint == build_causal_feature_vector(
        source, BOUNDARY + timedelta(minutes=15)
    ).feature_fingerprint
    future_ingest = source.watermark_utc + timedelta(seconds=1)
    future_raw = b"future-only-invalid-record"
    journal.append_frame(
        session_id=source.session_id,
        connection_id="connection-1",
        frame_index=len(source.frames),
        ingest_time_utc=future_ingest,
        raw_bytes=future_raw,
    )
    journal.advance_watermark(
        source.session_id,
        future_ingest,
        clock_health_verified=True,
    )
    tail_after_future = journal.tail_snapshot(
        source.session_id,
        after_frame_index=checkpoint.last_frame_index,
        expected_prefix_commitment_sha256=checkpoint.prefix_commitment_sha256,
        through_boundary_utc=BOUNDARY + timedelta(minutes=15),
    )
    after_future = build_incremental_causal_feature_vector(
        tail_after_future, BOUNDARY + timedelta(minutes=15), checkpoint
    )
    assert after_future == before_future
    journal.close()


def test_checkpoint_tampering_versions_missing_prefix_and_wrong_session_fail_closed():
    source = _snapshot()
    checkpoint = _checkpoint(source)
    payload = checkpoint.data
    payload["session_id"] = "other-session"
    with pytest.raises(ValueError, match="integrity digest"):
        FeatureReplayCheckpoint.from_dict(payload)

    payload = checkpoint.data
    payload["schema_version"] = 99
    with pytest.raises(ValueError, match="Unsupported checkpoint"):
        FeatureReplayCheckpoint.from_dict(payload)

    payload = checkpoint.data
    payload["connection_id"] = "wrong-connection"
    payload["integrity_sha256"] = _checkpoint_digest(payload)
    with pytest.raises(ValueError, match="connection identity"):
        FeatureReplayCheckpoint.from_dict(payload)

    payload = checkpoint.data
    payload["replay_state"]["buckets"].pop()
    payload["integrity_sha256"] = _checkpoint_digest(payload)
    with pytest.raises(ValueError, match="240-minute bucket history"):
        FeatureReplayCheckpoint.from_dict(payload)

    tail = tail_from_snapshot(source, checkpoint)
    with pytest.raises(CausalInputError, match="strictly_before"):
        build_incremental_causal_feature_vector(tail, BOUNDARY, checkpoint)
    with pytest.raises(CausalInputError, match="prefix_commitment"):
        build_incremental_causal_feature_vector(
            replace(tail, prefix_frame_index=tail.prefix_frame_index - 1),
            BOUNDARY + timedelta(minutes=15),
            checkpoint,
        )
    with pytest.raises(CausalInputError, match="session_mismatch"):
        build_incremental_causal_feature_vector(
            replace(tail, session_id="other-session"),
            BOUNDARY + timedelta(minutes=15),
            checkpoint,
        )


def test_altered_raw_prefix_tail_hash_and_sequence_gaps_are_rejected():
    source = _snapshot()
    checkpoint = _checkpoint(source)
    # Offline prefix recovery verifies every raw hash and the commitment chain.
    changed = list(source.frames)
    changed[0] = replace(changed[0], raw_bytes=changed[0].raw_bytes + b"x")
    with pytest.raises(CausalInputError, match="prefix_source_integrity"):
        tail_from_snapshot(replace(source, frames=tuple(changed)), checkpoint)
    missing_prefix = source.frames[:1] + source.frames[2:]
    with pytest.raises(CausalInputError, match="prefix_source_integrity"):
        tail_from_snapshot(replace(source, frames=missing_prefix), checkpoint)
    reordered_prefix = list(source.frames)
    reordered_prefix[1], reordered_prefix[2] = reordered_prefix[2], reordered_prefix[1]
    with pytest.raises(CausalInputError, match="prefix_source_integrity"):
        tail_from_snapshot(replace(source, frames=tuple(reordered_prefix)), checkpoint)

    tail = tail_from_snapshot(source, checkpoint)
    with pytest.raises(CausalInputError, match="noncontiguous_raw_frame_index"):
        build_incremental_causal_feature_vector(
            replace(tail, frames=tail.frames[1:], frame_commitments=tail.frame_commitments[1:]),
            BOUNDARY + timedelta(minutes=15),
            checkpoint,
        )
    corrupt = list(tail.frames)
    corrupt[0] = replace(corrupt[0], raw_frame_sha256="0" * 64)
    with pytest.raises(CausalInputError, match="raw_frame_hash"):
        build_incremental_causal_feature_vector(
            replace(tail, frames=tuple(corrupt)),
            BOUNDARY + timedelta(minutes=15),
            checkpoint,
        )

    gap_frame = _record(
        tail.frames[0].frame_index,
        10000,
        tail.frames[0].ingest_time_utc,
    )
    gap_tail = replace(tail, frames=(gap_frame, *tail.frames[1:]))
    gap_tail = replace(
        gap_tail,
        frame_commitments=tuple(
            (frame.frame_index, chain)
            for frame, chain in _tail_commitments(
                gap_tail.frames,
                checkpoint.prefix_commitment_sha256,
                checkpoint.last_frame_index,
            )
        ),
    )
    with pytest.raises(CausalInputError, match="replay_integrity"):
        build_incremental_causal_feature_vector(
            gap_tail,
            BOUNDARY + timedelta(minutes=15),
            checkpoint,
        )

    stale = _record(
        tail.frames[0].frame_index,
        checkpoint.data["last_accepted_sequence_num"],
        tail.frames[0].ingest_time_utc,
    )
    stale_tail = replace(tail, frames=(stale, *tail.frames[1:]))
    stale_tail = replace(
        stale_tail,
        frame_commitments=tuple(
            (frame.frame_index, chain)
            for frame, chain in _tail_commitments(
                stale_tail.frames,
                checkpoint.prefix_commitment_sha256,
                checkpoint.last_frame_index,
            )
        ),
    )
    with pytest.raises(CausalInputError, match="replay_integrity"):
        build_incremental_causal_feature_vector(
            stale_tail,
            BOUNDARY + timedelta(minutes=15),
            checkpoint,
        )

    wrong_connection = replace(tail.frames[0], connection_id="unrecognized-epoch")
    connection_tail = replace(tail, frames=(wrong_connection, *tail.frames[1:]))
    connection_tail = replace(
        connection_tail,
        frame_commitments=tuple(
            (frame.frame_index, chain)
            for frame, chain in _tail_commitments(
                connection_tail.frames,
                checkpoint.prefix_commitment_sha256,
                checkpoint.last_frame_index,
            )
        ),
    )
    with pytest.raises(CausalInputError, match="synchronized_at_boundary"):
        build_incremental_causal_feature_vector(
            connection_tail,
            BOUNDARY + timedelta(minutes=15),
            checkpoint,
        )


def test_reconnect_requires_a_fresh_snapshot_and_future_frames_do_not_change_vector():
    source = _snapshot(end=BOUNDARY + timedelta(minutes=15))
    checkpoint = _checkpoint(source)
    restart_time = BOUNDARY + timedelta(minutes=30)
    first_index = len(source.frames)
    reconnect_snapshot = _record(
        first_index,
        0,
        restart_time,
        connection="connection-2",
        snapshot=True,
    )
    reconnect_update = _record(
        first_index + 1,
        1,
        restart_time,
        connection="connection-2",
    )
    extended = replace(
        source,
        watermark_utc=restart_time,
        frames=source.frames + (reconnect_snapshot, reconnect_update),
    )
    tail = tail_from_snapshot(extended, checkpoint)
    actual = build_incremental_causal_feature_vector(tail, restart_time, checkpoint)
    expected = build_causal_feature_vector(extended, restart_time)
    assert actual.feature_fingerprint == expected.feature_fingerprint

    future = _record(
        first_index + 2,
        2,
        restart_time + timedelta(seconds=1),
        raw=b"malformed future bytes",
    )
    with_future = replace(extended, watermark_utc=future.ingest_time_utc, frames=extended.frames + (future,))
    actual_with_future = build_incremental_causal_feature_vector(
        tail_from_snapshot(with_future, checkpoint), restart_time, checkpoint
    )
    assert actual_with_future.values == actual.values
    assert actual_with_future.feature_fingerprint == actual.feature_fingerprint


def test_future_event_at_or_before_t_is_rejected_not_replayed_or_repaired():
    source = _snapshot(end=BOUNDARY + timedelta(minutes=15))
    checkpoint = _checkpoint(source)
    future_event_time = BOUNDARY + timedelta(minutes=16)
    raw = _wire_frame(len(source.frames), future_event_time, snapshot=False)
    # Receive before the prediction boundary but carry an exchange event after it.
    future_event = replace(
        _record(
            len(source.frames),
            len(source.frames),
            BOUNDARY + timedelta(minutes=15),
            raw=raw,
        ),
    )
    extended = replace(
        source,
        watermark_utc=BOUNDARY + timedelta(minutes=15),
        frames=source.frames + (future_event,),
    )
    with pytest.raises(CausalInputError, match="future_event_prevents_incremental"):
        build_incremental_causal_feature_vector(
            tail_from_snapshot(extended, checkpoint),
            BOUNDARY + timedelta(minutes=15),
            checkpoint,
        )


def test_missing_four_hour_history_and_incomplete_source_are_rejected():
    source = _snapshot()
    with pytest.raises(CausalInputError, match="insufficient_240_minute_session_history"):
        create_feature_replay_checkpoint(
            replace(
                source,
                session_started_at_utc=source.session_started_at_utc + timedelta(minutes=15),
            ),
            BOUNDARY,
        )
    with pytest.raises(CausalInputError, match="completeness_unverified"):
        create_feature_replay_checkpoint(
            replace(source, clock_health_verified=False), BOUNDARY
        )


def test_offline_benchmark_reports_stage_percentiles_work_memory_and_failures(publication_platform):
    import runpy

    run_benchmark = runpy.run_path(
        "benchmarks/benchmark_prospective_replay.py"
    )["run_benchmark"]

    report = run_benchmark(frame_counts=(32, 64), boundary_count=2, warmup_chunk_frames=7,
                           verify_full_replay=True)
    results = report["session_sizes"]
    for result in results:
        assert result["failure_count"] == 0
        assert result["boundaries_completed"] == 2
        assert result["tail_frames_per_boundary"]["max"] == 2
        assert result["peak_traced_python_memory_bytes"] > 0
        assert result["checkpoint_serialized_bytes"] > 0
        assert result["full_replay_parity"] is True
        assert result["max_warmup_chunk_frames"] <= 7
        assert result["warmup_chunks"] > 1
        assert result["peak_traced_warmup_memory_bytes"] > 0
    assert results[1]["session_frame_count"] > results[0]["session_frame_count"]
    assert max(r["peak_traced_python_memory_bytes"] for r in results) < (
        2 * min(r["peak_traced_python_memory_bytes"] for r in results)
    )
    assert max(r["checkpoint_serialized_bytes"] for r in results) < (
        2 * min(r["checkpoint_serialized_bytes"] for r in results)
    )
    result = results[0]
    for stage in (
        "checkpoint_file_read_ms",
        "source_tail_read_ms",
        "checkpoint_restore_ms",
        "incremental_tail_replay_ms",
        "feature_generation_ms",
        "model_inference_ms",
        "ledger_commit_attestation_ms",
        "checkpoint_advance_and_write_ms",
    ):
        assert set(result["stage_ms"][stage]) == {"median", "p95", "p99", "max"}


class _DeterministicPipeline:
    def __init__(self, names, probability):
        import numpy as np

        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.classes_ = np.asarray([0, 1])
        self.probability = probability

    def predict_proba(self, matrix):
        import numpy as np

        return np.tile(
            np.asarray([1.0 - self.probability, self.probability]),
            (len(matrix), 1),
        )


def test_incremental_inference_adapter_uses_existing_ledger_persistence(monkeypatch):
    class Clock:
        def now(self):
            return BOUNDARY + timedelta(minutes=15, seconds=1)

    class Ledger:
        clock_health = Clock()

        def __init__(self):
            self.evidence = None

        def append_prediction(self, evidence, *, model_bundle):
            self.evidence = evidence
            assert model_bundle is bundle
            return evidence

        def append_attempt(self, *_args, **_kwargs):
            raise AssertionError("A valid bounded replay must not record a failure")

    predictors = frozen_predictors()
    bundle = FrozenModelBundle(
        models={
            "range_volatility": _DeterministicPipeline(
                predictors["range_volatility"], 0.7
            ),
            "full_v2": _DeterministicPipeline(predictors["full_v2"], 0.4),
        },
        model_sha256={
            "range_volatility": RANGE_MODEL_SHA256,
            "full_v2": FULL_V2_MODEL_SHA256,
        },
        manifest_sha256=TRUSTED_MANIFEST_SHA256,
        manifest={"status": OFFICIAL_STATUS},
    )
    monkeypatch.setattr(inference, "is_verified_official_bundle", lambda _bundle: True)
    source = _snapshot()
    checkpoint = _checkpoint(source)
    tail = tail_from_snapshot(source, checkpoint)
    ledger = Ledger()
    result = process_incremental_boundary(
        tail,
        BOUNDARY + timedelta(minutes=15),
        checkpoint,
        bundle,
        implementation_git_sha="1" * 40,
        ledger=ledger,
    )
    assert result["feature_fingerprint"] == build_causal_feature_vector(
        source, BOUNDARY + timedelta(minutes=15)
    ).feature_fingerprint
    assert result["candidate_prediction"] == 1
    assert ledger.evidence is result
