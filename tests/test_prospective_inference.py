from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from btc_quarter_hour_engine.research.model_freeze import OFFICIAL_STATUS
from btc_quarter_hour_engine.research.prospective_contract import (
    FULL_V2_MODEL_SHA256,
    RANGE_MODEL_SHA256,
    TRUSTED_MANIFEST_SHA256,
)
from btc_quarter_hour_engine.research.prospective_inference import (
    FrozenModelBundle,
    infer_causal_prediction,
    load_official_models,
    process_boundary,
)
from btc_quarter_hour_engine.research.prospective_ledger import (
    ClockHealth,
    ProspectiveLedger,
)
from btc_quarter_hour_engine.research.prospective_source import (
    CausalInputError,
    ProspectiveRawFrameJournal,
    RawFrameRecord,
    RawFrameSnapshot,
    build_causal_feature_vector,
)
from btc_quarter_hour_engine.research import prospective_inference as inference


UTC = timezone.utc
BOUNDARY = datetime(2026, 1, 1, 4, 0, tzinfo=UTC)
IMPLEMENTATION_SHA = "1" * 40


def _wire_frame(sequence: int, timestamp: datetime, *, snapshot: bool = False) -> bytes:
    event_type = "snapshot" if snapshot else "update"
    updates = (
        [
            {
                "side": "bid",
                "price_level": "100",
                "new_quantity": "1",
                "event_time": timestamp.isoformat().replace("+00:00", "Z"),
            },
            {
                "side": "offer",
                "price_level": "101",
                "new_quantity": "2",
                "event_time": timestamp.isoformat().replace("+00:00", "Z"),
            },
        ]
        if snapshot
        else [
            {
                "side": "bid",
                "price_level": "100",
                "new_quantity": "1.5",
                "event_time": timestamp.isoformat().replace("+00:00", "Z"),
            },
            {
                "side": "bid",
                "price_level": "99",
                "new_quantity": "0",
                "event_time": timestamp.isoformat().replace("+00:00", "Z"),
            },
        ]
    )
    return json.dumps({
        "channel": "l2_data",
        "client_id": "",
        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
        "sequence_num": sequence,
        "events": [{
            "type": event_type,
            "product_id": "BTC-USD",
            "updates": updates,
        }],
    }).encode()


def _record(index: int, sequence: int, timestamp: datetime, *, snapshot=False, raw=None):
    payload = raw if raw is not None else _wire_frame(sequence, timestamp, snapshot=snapshot)
    return RawFrameRecord(
        raw_bytes=payload,
        session_id="session-1",
        connection_id="connection-1",
        frame_index=index,
        ingest_time_utc=timestamp,
        raw_frame_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _snapshot(
    boundary: datetime = BOUNDARY, *, add_future_frames: bool = False
) -> RawFrameSnapshot:
    start = boundary - timedelta(hours=4)
    frames = [_record(0, 0, start, snapshot=True)]
    index = 1
    sequence = 1
    timestamp = start + timedelta(minutes=15)
    while timestamp <= boundary:
        frames.append(_record(index, sequence, timestamp))
        index += 1
        sequence += 1
        frames.append(_record(index, sequence, timestamp))
        index += 1
        sequence += 1
        timestamp += timedelta(minutes=15)
    watermark = boundary
    if add_future_frames:
        after = boundary + timedelta(seconds=1)
        # A future exchange event can be received by T and is excluded by
        # event time, not by ingest time.
        future_event = _record(index, sequence, boundary, raw=_wire_frame(
            sequence, after
        ))
        frames.append(future_event)
        index += 1
        sequence += 1
        frames.append(_record(index, sequence, after))
        index += 1
        sequence += 1
        # Malformed frames that arrive after T must not affect T's evidence.
        frames.append(_record(index, sequence, after + timedelta(seconds=1), raw=b"not a Coinbase frame"))
        watermark = after + timedelta(seconds=1)
    return RawFrameSnapshot(
        session_id="session-1",
        session_started_at_utc=start,
        watermark_utc=watermark,
        clock_health_verified=True,
        frames=tuple(frames),
    )


class _DeterministicPipeline:
    def __init__(self, names, probability):
        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.classes_ = np.asarray([0, 1])
        self.probability = probability

    def predict_proba(self, matrix):
        return np.tile(
            np.asarray([1.0 - self.probability, self.probability]),
            (len(matrix), 1),
        )


def _bundle(candidate_probability=0.7, control_probability=0.4):
    from btc_quarter_hour_engine.research.model_freeze import frozen_predictors

    predictors = frozen_predictors()
    return FrozenModelBundle(
        models={
            "range_volatility": _DeterministicPipeline(
                predictors["range_volatility"], candidate_probability
            ),
            "full_v2": _DeterministicPipeline(
                predictors["full_v2"], control_probability
            ),
        },
        model_sha256={
            "range_volatility": RANGE_MODEL_SHA256,
            "full_v2": FULL_V2_MODEL_SHA256,
        },
        manifest_sha256=TRUSTED_MANIFEST_SHA256,
        manifest={"status": OFFICIAL_STATUS},
    )


def _verified_test_bundle(monkeypatch, candidate_probability=0.7, control_probability=0.4):
    bundle = _bundle(candidate_probability, control_probability)

    def verified_loader(_directory, **kwargs):
        assert kwargs == {
            "trusted_manifest_sha256": TRUSTED_MANIFEST_SHA256,
            "require_official": True,
        }
        return bundle.models, {
            "status": OFFICIAL_STATUS,
            "models": {
                "range_volatility": {"sha256": RANGE_MODEL_SHA256},
                "full_v2": {"sha256": FULL_V2_MODEL_SHA256},
            },
        }

    monkeypatch.setattr(inference, "load_frozen_models", verified_loader)
    return load_official_models("synthetic-official-fixture")


def test_v2_features_and_inference_are_strictly_causal_and_ordered():
    vector = build_causal_feature_vector(_snapshot(), BOUNDARY)
    assert vector.data_ready and vector.feature_ready
    assert vector.target_utc == BOUNDARY + timedelta(minutes=15)
    assert len(vector.candidate_predictors) == 12
    assert len(vector.control_predictors) == 91
    assert tuple(vector.values) == vector.control_predictors

    evidence = infer_causal_prediction(
        vector,
        _bundle(),
        implementation_git_sha=IMPLEMENTATION_SHA,
        generated_at_utc=BOUNDARY + timedelta(seconds=1),
    )
    assert evidence["candidate_probability"] == pytest.approx(0.7)
    assert evidence["candidate_prediction"] == 1
    assert evidence["control_probability"] == pytest.approx(0.4)
    assert evidence["control_prediction"] == 0
    assert evidence["baseline_prediction"] == 0
    assert evidence["persisted_at_utc"] is None


def test_future_ingest_and_exchange_event_frames_cannot_change_completed_features():
    baseline = build_causal_feature_vector(_snapshot(), BOUNDARY)
    with_future = build_causal_feature_vector(
        _snapshot(add_future_frames=True), BOUNDARY
    )
    assert baseline.values == with_future.values
    assert baseline.feature_fingerprint == with_future.feature_fingerprint
    assert baseline.source_provenance_sha256 == with_future.source_provenance_sha256
    assert with_future.excluded_ingest_future_count == 2
    assert with_future.excluded_event_future_count == 1

    baseline_prediction = infer_causal_prediction(
        baseline,
        _bundle(),
        implementation_git_sha=IMPLEMENTATION_SHA,
        generated_at_utc=BOUNDARY + timedelta(seconds=1),
    )
    future_prediction = infer_causal_prediction(
        with_future,
        _bundle(),
        implementation_git_sha=IMPLEMENTATION_SHA,
        generated_at_utc=BOUNDARY + timedelta(seconds=1),
    )
    assert (
        baseline_prediction["candidate_probability"],
        baseline_prediction["control_probability"],
        baseline_prediction["candidate_prediction"],
        baseline_prediction["control_prediction"],
    ) == (
        future_prediction["candidate_probability"],
        future_prediction["control_probability"],
        future_prediction["candidate_prediction"],
        future_prediction["control_prediction"],
    )


def test_isolated_raw_frame_journal_is_durable_and_append_only(tmp_path):
    source = _snapshot(add_future_frames=True)
    path = tmp_path / "source.sqlite"
    journal = ProspectiveRawFrameJournal(path)
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
    expected = build_causal_feature_vector(source, BOUNDARY)
    journal.close()

    restarted = ProspectiveRawFrameJournal(path)
    restored = restarted.snapshot(source.session_id)
    actual = build_causal_feature_vector(restored, BOUNDARY)
    assert actual.values == expected.values
    assert actual.feature_fingerprint == expected.feature_fingerprint
    with pytest.raises(ValueError, match="watermark precedes"):
        restarted.advance_watermark(
            source.session_id,
            BOUNDARY,
            clock_health_verified=True,
        )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        restarted._connection.execute(
            "UPDATE source_frames SET raw_bytes = raw_bytes "
            "WHERE session_id = ? AND frame_index = 0",
            (source.session_id,),
        )
    restarted.close()


def test_source_watermarks_are_irreversible_across_restarts(tmp_path):
    source = _snapshot()
    path = tmp_path / "irreversible-source.sqlite"
    journal = ProspectiveRawFrameJournal(path)
    journal.start_session(source.session_id, source.session_started_at_utc)
    first = source.frames[0]
    journal.append_frame(
        session_id=first.session_id,
        connection_id=first.connection_id,
        frame_index=0,
        ingest_time_utc=first.ingest_time_utc,
        raw_bytes=first.raw_bytes,
    )
    watermark = first.ingest_time_utc + timedelta(seconds=2)
    journal.advance_watermark(
        source.session_id,
        watermark,
        clock_health_verified=True,
    )
    for rejected_time in (watermark - timedelta(seconds=1), watermark):
        with pytest.raises(ValueError, match="at or before"):
            journal.append_frame(
                session_id=source.session_id,
                connection_id="connection-1",
                frame_index=1,
                ingest_time_utc=rejected_time,
                raw_bytes=source.frames[1].raw_bytes,
            )
    with pytest.raises(ValueError, match="repeated"):
        journal.advance_watermark(
            source.session_id,
            watermark,
            clock_health_verified=True,
        )
    with pytest.raises(ValueError, match="regressed"):
        journal.advance_watermark(
            source.session_id,
            watermark - timedelta(seconds=1),
            clock_health_verified=True,
        )
    earlier_snapshot = journal.snapshot(source.session_id)
    journal.close()

    restarted = ProspectiveRawFrameJournal(path)
    with pytest.raises(ValueError, match="at or before"):
        restarted.append_frame(
            session_id=source.session_id,
            connection_id="connection-1",
            frame_index=1,
            ingest_time_utc=watermark,
            raw_bytes=source.frames[1].raw_bytes,
        )
    accepted_time = watermark + timedelta(seconds=1)
    restarted.append_frame(
        session_id=source.session_id,
        connection_id="connection-1",
        frame_index=1,
        ingest_time_utc=accepted_time,
        raw_bytes=source.frames[1].raw_bytes,
    )
    assert len(earlier_snapshot.frames) == 1
    assert len(restarted.snapshot(source.session_id).frames) == 2
    with pytest.raises(ValueError, match="regressed"):
        restarted.append_frame(
            session_id=source.session_id,
            connection_id="connection-1",
            frame_index=2,
            ingest_time_utc=watermark + timedelta(milliseconds=500),
            raw_bytes=source.frames[2].raw_bytes,
        )
    restarted.close()


def test_target_labels_are_not_an_input_to_feature_generation():
    assert "target" not in RawFrameSnapshot.__dataclass_fields__
    assert "label" not in RawFrameSnapshot.__dataclass_fields__
    assert "target_eligible" not in RawFrameSnapshot.__dataclass_fields__
    first = build_causal_feature_vector(_snapshot(), BOUNDARY)
    second = build_causal_feature_vector(_snapshot(), BOUNDARY)
    assert first.values == second.values
    assert first.feature_fingerprint == second.feature_fingerprint


def test_post_boundary_watermark_is_required_and_source_clock_must_be_verified():
    original = _snapshot()
    with pytest.raises(CausalInputError, match="source_watermark_before_boundary"):
        build_causal_feature_vector(
            replace(original, watermark_utc=BOUNDARY - timedelta(microseconds=1)),
            BOUNDARY,
        )
    with pytest.raises(CausalInputError, match="source_clock_health_unverified"):
        build_causal_feature_vector(
            replace(original, clock_health_verified=False), BOUNDARY
        )


def test_malformed_raw_frame_and_unsynchronized_book_fail_closed():
    original = _snapshot()
    malformed = replace(
        original,
        frames=original.frames[:1] + (
            _record(1, 1, original.frames[0].ingest_time_utc, raw=b"invalid"),
        ),
    )
    with pytest.raises(CausalInputError, match="malformed_raw_frame"):
        build_causal_feature_vector(malformed, BOUNDARY)

    start = BOUNDARY - timedelta(hours=4)
    update = _record(0, 0, BOUNDARY)
    unsynced = RawFrameSnapshot(
        session_id="session-1",
        session_started_at_utc=start,
        watermark_utc=BOUNDARY,
        clock_health_verified=True,
        frames=(update,),
    )
    with pytest.raises(CausalInputError, match="order_book_not_synchronized"):
        build_causal_feature_vector(unsynced, BOUNDARY)


def test_missing_or_nonfinite_predictors_are_rejected():
    vector = build_causal_feature_vector(_snapshot(), BOUNDARY)
    values = dict(vector.values)
    values["momentum_log_return_15m"] = float("nan")
    invalid = replace(vector, values=values)
    with pytest.raises(ValueError, match="finite"):
        infer_causal_prediction(
            invalid,
            _bundle(),
            implementation_git_sha=IMPLEMENTATION_SHA,
            generated_at_utc=BOUNDARY + timedelta(seconds=1),
        )

    changed = dict(vector.values)
    changed["asof_spread_bps"] += 1.0
    with pytest.raises(ValueError, match="fingerprint"):
        infer_causal_prediction(
            replace(vector, values=changed),
            _bundle(),
            implementation_git_sha=IMPLEMENTATION_SHA,
            generated_at_utc=BOUNDARY + timedelta(seconds=1),
        )


def test_predictor_order_model_hash_and_probability_outputs_are_validated():
    vector = build_causal_feature_vector(_snapshot(), BOUNDARY)
    reordered = replace(
        vector,
        candidate_predictors=tuple(reversed(vector.candidate_predictors)),
    )
    with pytest.raises(ValueError, match="Candidate predictor ordering"):
        infer_causal_prediction(
            reordered,
            _bundle(),
            implementation_git_sha=IMPLEMENTATION_SHA,
            generated_at_utc=BOUNDARY + timedelta(seconds=1),
        )

    bad_hashes = {
        "range_volatility": "0" * 64,
        "full_v2": FULL_V2_MODEL_SHA256,
    }
    with pytest.raises(ValueError, match="model digests"):
        FrozenModelBundle(
            models=_bundle().models,
            model_sha256=bad_hashes,
            manifest_sha256=TRUSTED_MANIFEST_SHA256,
            manifest={"status": OFFICIAL_STATUS},
        )

    bad_bundle = FrozenModelBundle(
        models={
            "range_volatility": _DeterministicPipeline(
                vector.candidate_predictors, float("nan")
            ),
            "full_v2": _DeterministicPipeline(vector.control_predictors, 0.4),
        },
        model_sha256={
            "range_volatility": RANGE_MODEL_SHA256,
            "full_v2": FULL_V2_MODEL_SHA256,
        },
        manifest_sha256=TRUSTED_MANIFEST_SHA256,
        manifest={"status": OFFICIAL_STATUS},
    )
    with pytest.raises(ValueError, match="probability"):
        infer_causal_prediction(
            vector,
            bad_bundle,
            implementation_git_sha=IMPLEMENTATION_SHA,
            generated_at_utc=BOUNDARY + timedelta(seconds=1),
        )


def test_official_loader_pins_trusted_manifest_and_rejects_development_status(monkeypatch):
    calls = []

    def fake_loader(directory, **kwargs):
        calls.append((directory, kwargs))
        return _bundle().models, {
            "status": "development_only_nonofficial",
            "models": {
                "range_volatility": {"sha256": RANGE_MODEL_SHA256},
                "full_v2": {"sha256": FULL_V2_MODEL_SHA256},
            },
        }

    monkeypatch.setattr(inference, "load_frozen_models", fake_loader)
    with pytest.raises(ValueError, match="Official prospective"):
        load_official_models("unused-local-fixture")
    assert calls == [(
        "unused-local-fixture",
        {
            "trusted_manifest_sha256": TRUSTED_MANIFEST_SHA256,
            "require_official": True,
        },
    )]

    with pytest.raises(ValueError, match="Development-only"):
        FrozenModelBundle(
            models=_bundle().models,
            model_sha256={
                "range_volatility": RANGE_MODEL_SHA256,
                "full_v2": FULL_V2_MODEL_SHA256,
            },
            manifest_sha256=TRUSTED_MANIFEST_SHA256,
            manifest={"status": "development_only_nonofficial"},
        )


def test_directly_constructed_bundle_cannot_become_official():
    forged = _bundle()
    vector = build_causal_feature_vector(_snapshot(), BOUNDARY)
    assert not inference.is_verified_official_bundle(forged)
    assert not hasattr(inference, "register_official_bundle")
    with pytest.raises(ValueError, match="load_official_models"):
        infer_causal_prediction(
            vector,
            forged,
            implementation_git_sha=IMPLEMENTATION_SHA,
            generated_at_utc=BOUNDARY + timedelta(seconds=1),
            require_official_bundle=True,
        )


def test_verified_model_mutation_invalidates_official_bundle(monkeypatch):
    bundle = _verified_test_bundle(monkeypatch)
    vector = build_causal_feature_vector(_snapshot(), BOUNDARY)
    assert inference.is_verified_official_bundle(bundle)
    bundle.models["range_volatility"].probability = 0.99
    assert not inference.is_verified_official_bundle(bundle)
    with pytest.raises(ValueError, match="load_official_models"):
        infer_causal_prediction(
            vector,
            bundle,
            implementation_git_sha=IMPLEMENTATION_SHA,
            generated_at_utc=BOUNDARY + timedelta(seconds=1),
            require_official_bundle=True,
        )


def test_inference_cannot_be_generated_at_or_after_target_time():
    vector = build_causal_feature_vector(_snapshot(), BOUNDARY)
    with pytest.raises(ValueError, match="at/after the target"):
        infer_causal_prediction(
            vector,
            _bundle(),
            implementation_git_sha=IMPLEMENTATION_SHA,
            generated_at_utc=BOUNDARY + timedelta(minutes=15),
        )


def test_boundary_runner_persists_success_and_records_missing_data(tmp_path, monkeypatch):
    boundary = datetime(2026, 11, 15, 0, 0, tzinfo=UTC)
    class ManualClock:
        wall = boundary + timedelta(seconds=2)
        monotonic = 0.0

    clock = ManualClock()
    model_bundle = _verified_test_bundle(monkeypatch)
    ledger = ProspectiveLedger(
        tmp_path / "runner.sqlite",
        clock_health=ClockHealth(
            max_drift_seconds=2,
            wall_clock=lambda: clock.wall,
            monotonic_clock=lambda: clock.monotonic,
        ),
    )
    result = process_boundary(
        _snapshot(boundary),
        boundary,
        model_bundle,
        implementation_git_sha=IMPLEMENTATION_SHA,
        ledger=ledger,
    )
    assert result["timing_eligible"] is True
    assert result["generated_at_utc"] == boundary + timedelta(seconds=2)
    assert ledger.lifecycle(boundary)["status"] == "predicted"

    next_boundary = boundary + timedelta(minutes=15)
    clock.wall = next_boundary + timedelta(seconds=2)
    clock.monotonic = 900
    with pytest.raises(CausalInputError, match="source_clock_health_unverified"):
        process_boundary(
            replace(
                _snapshot(next_boundary),
                clock_health_verified=False,
            ),
            next_boundary,
            model_bundle,
            implementation_git_sha=IMPLEMENTATION_SHA,
            ledger=ledger,
        )
    state = ledger.lifecycle(next_boundary)
    assert state["prediction"] is None
    assert state["status"] == "missing_data"
    assert state["events"][0]["reason"] == "source_clock_health_unverified"
    ledger.close()


def test_unexpected_inference_failure_is_recorded_and_propagated(tmp_path, monkeypatch):
    boundary = datetime(2026, 11, 15, 0, 0, tzinfo=UTC)
    model_bundle = _verified_test_bundle(monkeypatch)
    ledger = ProspectiveLedger(
        tmp_path / "inference-failure.sqlite",
        clock_health=ClockHealth(
            wall_clock=lambda: boundary + timedelta(seconds=2),
            monotonic_clock=lambda: 0.0,
        ),
    )

    def fail_inference(*args, **kwargs):
        raise RuntimeError("synthetic model runtime failure")

    monkeypatch.setattr(inference, "infer_causal_prediction", fail_inference)
    with pytest.raises(RuntimeError, match="synthetic model runtime failure"):
        process_boundary(
            _snapshot(boundary),
            boundary,
            model_bundle,
            implementation_git_sha=IMPLEMENTATION_SHA,
            ledger=ledger,
        )
    state = ledger.lifecycle(boundary)
    assert state["prediction"] is None
    assert state["status"] == "inference_failed"
    assert "synthetic model runtime failure" in state["events"][0]["reason"]
    ledger.close()


def test_process_boundary_rejects_forged_hash_only_bundle(tmp_path):
    boundary = datetime(2026, 11, 15, 0, 0, tzinfo=UTC)
    ledger = ProspectiveLedger(tmp_path / "forged-bundle.sqlite")
    with pytest.raises(ValueError, match="loaded by load_official_models"):
        process_boundary(
            _snapshot(boundary),
            boundary,
            _bundle(),
            implementation_git_sha=IMPLEMENTATION_SHA,
            ledger=ledger,
        )
    assert ledger.lifecycle(boundary)["prediction"] is None
    assert ledger.lifecycle(boundary)["events"] == []
    ledger.close()


def test_process_boundary_rejects_mutated_loaded_bundle(tmp_path, monkeypatch):
    boundary = datetime(2026, 11, 15, 0, 0, tzinfo=UTC)
    model_bundle = _verified_test_bundle(monkeypatch)
    model_bundle.model_sha256["range_volatility"] = "0" * 64
    ledger = ProspectiveLedger(tmp_path / "mutated-bundle.sqlite")
    with pytest.raises(ValueError, match="loaded by load_official_models"):
        process_boundary(
            _snapshot(boundary),
            boundary,
            model_bundle,
            implementation_git_sha=IMPLEMENTATION_SHA,
            ledger=ledger,
        )
    assert ledger.lifecycle(boundary)["prediction"] is None
    ledger.close()
