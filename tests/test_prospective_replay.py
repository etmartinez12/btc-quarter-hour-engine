from __future__ import annotations

import hashlib
import json
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
from btc_quarter_hour_engine.research.prospective_replay import (
    FeatureReplayCheckpoint,
    _checkpoint_digest,
    _tail_commitments,
    advance_feature_replay_checkpoint,
    build_incremental_causal_feature_vector,
    create_feature_replay_checkpoint,
    tail_from_snapshot,
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


def _snapshot(*, end: datetime = BOUNDARY + timedelta(minutes=45)) -> RawFrameSnapshot:
    start = BOUNDARY - timedelta(hours=4)
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


def _checkpoint(snapshot: RawFrameSnapshot, boundary=BOUNDARY):
    return create_feature_replay_checkpoint(snapshot, boundary)


def test_incremental_replay_matches_full_replay_across_boundaries_and_restart(tmp_path):
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


def test_offline_benchmark_reports_stage_percentiles_work_memory_and_failures():
    import runpy

    run_benchmark = runpy.run_path(
        "benchmarks/benchmark_prospective_replay.py"
    )["run_benchmark"]

    report = run_benchmark(frame_counts=(32, 64), boundary_count=2)
    results = report["session_sizes"]
    for result in results:
        assert result["failure_count"] == 0
        assert result["boundaries_completed"] == 2
        assert result["tail_frames_per_boundary"]["max"] == 2
        assert result["peak_traced_python_memory_bytes"] > 0
        assert result["checkpoint_serialized_bytes"] > 0
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
        assert set(result["stage_ms"][stage]) == {"median", "p95", "max"}


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
