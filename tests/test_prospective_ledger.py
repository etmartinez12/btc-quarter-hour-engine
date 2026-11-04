from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from btc_quarter_hour_engine.research.feature_families_v2 import localization_variants
from btc_quarter_hour_engine.research.model_freeze import OFFICIAL_STATUS, frozen_predictors
from btc_quarter_hour_engine.research.prospective_contract import (
    FULL_V2_MODEL_SHA256,
    RANGE_MODEL_SHA256,
    TRUSTED_MANIFEST_SHA256,
)
from btc_quarter_hour_engine.research.prospective_ledger import (
    ClockHealth,
    ClockHealthError,
    DuplicateLabelError,
    DuplicatePredictionError,
    LatePredictionError,
    ProspectiveLedger,
)
from btc_quarter_hour_engine.research.prospective_source import (
    compute_feature_fingerprint,
)


UTC = timezone.utc
START = datetime(2026, 11, 15, 0, 0, tzinfo=UTC)
IMPLEMENTATION_SHA = "1" * 40


class _DeterministicPipeline:
    def __init__(self, names, probability):
        self.feature_names_in_ = list(names)
        self.classes_ = [0, 1]
        self.probability = probability

    def predict_proba(self, matrix):
        return np.tile(
            np.asarray([1.0 - self.probability, self.probability]),
            (len(matrix), 1),
        )


@pytest.fixture
def official_model_bundle(monkeypatch):
    from btc_quarter_hour_engine.research import prospective_inference as inference

    predictors = frozen_predictors()
    models = {
        "range_volatility": _DeterministicPipeline(
            predictors["range_volatility"], 0.7
        ),
        "full_v2": _DeterministicPipeline(predictors["full_v2"], 0.4),
    }

    def fake_loader(_directory, **kwargs):
        assert kwargs["trusted_manifest_sha256"] == TRUSTED_MANIFEST_SHA256
        assert kwargs["require_official"] is True
        return models, {
            "status": OFFICIAL_STATUS,
            "models": {
                "range_volatility": {"sha256": RANGE_MODEL_SHA256},
                "full_v2": {"sha256": FULL_V2_MODEL_SHA256},
            },
        }

    monkeypatch.setattr(inference, "load_frozen_models", fake_loader)
    return inference.load_official_models("deterministic-test-fixture")


def _ledger(path, *, count=1, max_drift=20_000.0):
    class ManualClock:
        wall = START
        monotonic = 0.0

        def wall_now(self):
            return self.wall

        def monotonic_now(self):
            return self.monotonic

        def advance(self, seconds):
            self.wall += timedelta(seconds=seconds)
            self.monotonic += seconds

        def set(self, wall, monotonic):
            self.wall = wall
            self.monotonic = monotonic

    manual = ManualClock()
    return ProspectiveLedger(
        path,
        window_start=START,
        window_end_exclusive=START + timedelta(minutes=15 * count),
        clock_health=ClockHealth(
            max_drift_seconds=max_drift,
            wall_clock=manual.wall_now,
            monotonic_clock=manual.monotonic_now,
        ),
    ), manual


def _evidence(boundary=START):
    predictors = localization_variants()
    candidate = tuple(predictors["group_only", "range_volatility"])
    control = tuple(predictors["reference", "all"])
    return {
        "boundary_utc": boundary,
        "target_utc": boundary + timedelta(minutes=15),
        "data_ready": True,
        "feature_ready": True,
        "clock_health_verified": True,
        "source_session_id": "session-1",
        "source_watermark_utc": boundary,
        "source_provenance_sha256": "a" * 64,
        "source_frame_count": 10,
        "excluded_ingest_future_count": 1,
        "excluded_event_future_count": 2,
        "feature_contract": "bbo-causal-v2",
        "feature_family_contract": "bbo-causal-v2-families-v1",
        "candidate_predictors": list(candidate),
        "control_predictors": list(control),
        "feature_fingerprint": compute_feature_fingerprint(
            {name: 0.25 for name in control},
            candidate_predictors=candidate,
            control_predictors=control,
            feature_contract="bbo-causal-v2",
            feature_family_contract="bbo-causal-v2-families-v1",
        ),
        "feature_vector": {name: 0.25 for name in control},
        "manifest_sha256": TRUSTED_MANIFEST_SHA256,
        "candidate_model_sha256": RANGE_MODEL_SHA256,
        "control_model_sha256": FULL_V2_MODEL_SHA256,
        "implementation_git_sha": IMPLEMENTATION_SHA,
        "generated_at_utc": boundary + timedelta(seconds=1),
        "candidate_probability": 0.7,
        "candidate_prediction": 1,
        "control_probability": 0.4,
        "control_prediction": 0,
        "baseline_prediction": 0,
        "failure_reasons": [],
    }


def test_registered_schedule_is_complete_and_immutable(tmp_path):
    ledger = ProspectiveLedger(tmp_path / "full.sqlite")
    schedule = ledger.schedule()
    assert ledger.scheduled_count == 2880
    assert len(schedule) == 2880
    assert schedule[0] == {
        "boundary_utc": "2026-11-15T00:00:00.000000Z",
        "target_utc": "2026-11-15T00:15:00.000000Z",
    }
    assert schedule[-1]["boundary_utc"] == "2026-12-14T23:45:00.000000Z"
    with pytest.raises(sqlite3.IntegrityError):
        ledger._connection.execute(
            "INSERT INTO scheduled_boundaries VALUES (?, ?)",
            (schedule[0]["boundary_utc"], schedule[0]["target_utc"]),
        )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        ledger._connection.execute(
            "UPDATE scheduled_boundaries SET target_utc = target_utc "
            "WHERE boundary_utc = ?",
            (schedule[0]["boundary_utc"],),
        )
    ledger.close()


def test_legacy_ledger_without_attestation_is_preserved_and_rejected(tmp_path):
    path = tmp_path / "legacy.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA user_version = 1")
    connection.execute("CREATE TABLE legacy_evidence (value TEXT NOT NULL)")
    connection.execute("INSERT INTO legacy_evidence VALUES ('keep')")
    connection.commit()
    connection.close()

    with pytest.raises(ValueError, match="lacks post-commit durability"):
        ProspectiveLedger(path)

    preserved = sqlite3.connect(path)
    assert preserved.execute("SELECT value FROM legacy_evidence").fetchone() == ("keep",)
    assert preserved.execute("PRAGMA user_version").fetchone() == (1,)
    preserved.close()


def test_prediction_persists_once_with_all_frozen_evidence(tmp_path, official_model_bundle):
    ledger, clock = _ledger(tmp_path / "ledger.sqlite")
    clock.set(START + timedelta(seconds=5), 100)
    result = ledger.append_prediction(
        _evidence(),
        model_bundle=official_model_bundle,
    )
    assert result["timing_eligible"] is True
    assert ledger.lifecycle(START)["status"] == "predicted"
    assert result["persisted_at_utc"] == START + timedelta(seconds=5)
    state = ledger.lifecycle(START)
    assert state["evaluation_eligible"] is True
    assert state["durability_attestation"]["deadline_met"] == 1
    assert state["prediction"]["candidate_model_sha256"] == RANGE_MODEL_SHA256
    assert state["prediction"]["control_model_sha256"] == FULL_V2_MODEL_SHA256
    assert state["prediction"]["manifest_sha256"] == TRUSTED_MANIFEST_SHA256
    assert state["prediction"]["candidate_probability"] == 0.7
    assert state["prediction"]["candidate_prediction"] == 1
    assert state["prediction"]["baseline_prediction"] == 0
    assert state["prediction"]["write_started_at_utc"] == "2026-11-15T00:00:05.000000Z"
    assert state["prediction"]["clock_health_verified"] == 1
    assert len(state["prediction"]["feature_vector_json"].encode()) > 0
    with pytest.raises(DuplicatePredictionError):
        clock.advance(1)
        ledger.append_prediction(
            _evidence(),
            model_bundle=official_model_bundle,
        )
    assert [event["event_type"] for event in ledger.lifecycle(START)["events"]] == [
        "prediction_recorded",
        "durability_attested",
        "duplicate_attempt",
    ]
    ledger.close()


def test_late_write_is_an_event_not_a_prediction_and_is_never_backfilled(
    tmp_path, official_model_bundle
):
    ledger, clock = _ledger(tmp_path / "late.sqlite")
    clock.set(START + timedelta(seconds=61), 0)
    with pytest.raises(LatePredictionError):
        ledger.append_prediction(
            _evidence(),
            model_bundle=official_model_bundle,
        )
    state = ledger.lifecycle(START)
    assert state["prediction"] is None
    assert state["events"][0]["event_type"] == "late_attempt"
    assert state["status"] == "late_attempt"
    clock.advance(19)
    assert ledger.mark_missed_boundaries() == 1
    state = ledger.lifecycle(START)
    assert state["prediction"] is None
    assert state["status"] == "missed"
    assert [event["event_type"] for event in state["events"]] == [
        "late_attempt",
        "missed",
    ]
    clock.advance(10)
    assert ledger.mark_missed_boundaries() == 0
    ledger.close()


def test_commit_crossing_deadline_is_persisted_but_not_evaluation_eligible(
    tmp_path, monkeypatch, official_model_bundle
):
    ledger, clock = _ledger(tmp_path / "commit-crossing.sqlite")
    clock.set(START + timedelta(seconds=59), 59)
    original_transaction = ledger._transaction
    transaction_count = 0

    @contextmanager
    def commit_then_advance_clock():
        nonlocal transaction_count
        with original_transaction():
            yield
        transaction_count += 1
        if transaction_count == 1:
            clock.advance(2)

    monkeypatch.setattr(ledger, "_transaction", commit_then_advance_clock)
    with pytest.raises(LatePredictionError, match="committed after"):
        ledger.append_prediction(
            _evidence(), model_bundle=official_model_bundle
        )

    state = ledger.lifecycle(START)
    assert state["prediction"] is not None
    assert state["durability_attestation"]["deadline_met"] == 0
    assert state["evaluation_eligible"] is False
    assert state["status"] == "durability_late"
    ledger.close()


def test_interrupted_post_commit_confirmation_stays_ineligible_after_restart(
    tmp_path, monkeypatch, official_model_bundle
):
    path = tmp_path / "confirmation-interrupted.sqlite"
    ledger, clock = _ledger(path)
    clock.set(START + timedelta(seconds=5), 5)
    original_insert_event = ledger._insert_event

    def interrupt_confirmation(boundary, event_type, *args, **kwargs):
        if event_type == "durability_attested":
            raise RuntimeError("simulated crash during durability confirmation")
        return original_insert_event(boundary, event_type, *args, **kwargs)

    monkeypatch.setattr(ledger, "_insert_event", interrupt_confirmation)
    with pytest.raises(RuntimeError, match="during durability confirmation"):
        ledger.append_prediction(
            _evidence(), model_bundle=official_model_bundle
        )
    state = ledger.lifecycle(START)
    assert state["prediction"] is not None
    assert state["durability_attestation"] is None
    assert state["evaluation_eligible"] is False
    assert state["status"] == "durability_unconfirmed"
    ledger.close()

    restarted, restart_clock = _ledger(path)
    restart_clock.set(START + timedelta(seconds=10), 10)
    recovered = restarted.lifecycle(START)
    assert recovered["prediction"] is not None
    assert recovered["durability_attestation"] is None
    assert recovered["evaluation_eligible"] is False
    with pytest.raises(DuplicatePredictionError):
        restarted.append_prediction(
            _evidence(), model_bundle=official_model_bundle
        )
    assert restarted.lifecycle(START)["durability_attestation"] is None
    restarted.close()


def test_public_persistence_rejects_unverified_model_bundle(tmp_path):
    ledger, _ = _ledger(tmp_path / "unverified-model.sqlite")
    with pytest.raises(ValueError, match="load_official_models"):
        ledger.append_prediction(_evidence(), model_bundle=object())
    assert ledger.lifecycle(START)["prediction"] is None
    ledger.close()


def test_ledger_rejects_fabricated_probabilities_with_verified_bundle(
    tmp_path, official_model_bundle
):
    ledger, clock = _ledger(tmp_path / "fabricated-output.sqlite")
    clock.set(START + timedelta(seconds=2), 0)
    evidence = _evidence()
    evidence["candidate_probability"] = 0.99
    with pytest.raises(ValueError, match="does not match verified model inference"):
        ledger.append_prediction(evidence, model_bundle=official_model_bundle)
    assert ledger.lifecycle(START)["prediction"] is None
    ledger.close()


def test_clock_skew_fails_closed_and_is_recorded(tmp_path, official_model_bundle):
    ledger, clock = _ledger(
        tmp_path / "clock.sqlite", count=2, max_drift=1.0
    )
    clock.set(START + timedelta(seconds=1), 10)
    ledger.append_prediction(
        _evidence(START),
        model_bundle=official_model_bundle,
    )
    second = START + timedelta(minutes=15)
    clock.set(second + timedelta(seconds=1), 11)
    with pytest.raises(ClockHealthError, match="wall_monotonic_clock_skew"):
        ledger.append_prediction(
            _evidence(second),
            model_bundle=official_model_bundle,
        )
    assert ledger.lifecycle(second)["prediction"] is None
    assert ledger.lifecycle(second)["events"][0]["event_type"] == "clock_skew"
    ledger.close()


def test_interrupted_transaction_rolls_back_prediction_and_event(
    tmp_path, monkeypatch, official_model_bundle
):
    ledger, clock = _ledger(tmp_path / "interrupted.sqlite")
    clock.set(START + timedelta(seconds=2), 0)
    original = ledger._insert_event

    def interrupt_after_event(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("simulated process interruption before commit")

    monkeypatch.setattr(ledger, "_insert_event", interrupt_after_event)
    with pytest.raises(RuntimeError, match="interruption"):
        ledger.append_prediction(
            _evidence(),
            model_bundle=official_model_bundle,
        )
    assert ledger.lifecycle(START)["prediction"] is None
    assert ledger.lifecycle(START)["events"] == []

    monkeypatch.setattr(ledger, "_insert_event", original)
    clock.advance(1)
    ledger.append_prediction(
        _evidence(),
        model_bundle=official_model_bundle,
    )
    ledger.close()


def test_restart_preserves_prediction_and_duplicate_identity(
    tmp_path, official_model_bundle
):
    path = tmp_path / "restart.sqlite"
    ledger, clock = _ledger(path)
    clock.set(START + timedelta(seconds=2), 0)
    ledger.append_prediction(
        _evidence(),
        model_bundle=official_model_bundle,
    )
    ledger.close()

    restarted, restart_clock = _ledger(path)
    restart_clock.set(START + timedelta(seconds=3), 1)
    assert restarted.scheduled_count == 1
    assert restarted.lifecycle(START)["prediction"] is not None
    with pytest.raises(DuplicatePredictionError):
        restarted.append_prediction(
            _evidence(),
            model_bundle=official_model_bundle,
        )
    restarted.close()


def test_label_is_separate_append_only_evidence_and_cannot_rewrite_prediction(
    tmp_path, official_model_bundle
):
    ledger, clock = _ledger(tmp_path / "labels.sqlite")
    clock.set(START + timedelta(seconds=2), 0)
    ledger.append_prediction(
        _evidence(),
        model_bundle=official_model_bundle,
    )
    before = dict(ledger.lifecycle(START)["prediction"])
    target_time = START + timedelta(minutes=15)
    available = target_time + timedelta(seconds=10)
    clock.set(available + timedelta(seconds=2), 912)
    ledger.attach_label(
        START,
        target=1,
        target_utc=target_time,
        available_at_utc=available,
        target_eligible=True,
        source_provenance_sha256="c" * 64,
        evidence={"canonical_price_source": "synthetic"},
    )
    after = ledger.lifecycle(START)
    assert dict(after["prediction"]) == before
    assert after["label"]["target"] == 1
    assert after["status"] == "labeled"
    assert [event["event_type"] for event in after["events"]] == [
        "prediction_recorded",
        "durability_attested",
        "label_attached",
    ]
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        ledger._connection.execute(
            "UPDATE predictions SET candidate_prediction = 0 WHERE boundary_utc = ?",
            ("2026-11-15T00:00:00.000000Z",),
        )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        ledger._connection.execute(
            "UPDATE labels SET target = 0 WHERE boundary_utc = ?",
            ("2026-11-15T00:00:00.000000Z",),
        )
    with pytest.raises(DuplicateLabelError):
        clock.advance(1)
        ledger.attach_label(
            START,
            target=0,
            target_utc=target_time,
            available_at_utc=available,
            target_eligible=True,
            source_provenance_sha256="d" * 64,
        )
    ledger.close()


def test_ledger_rejects_wrong_predictor_order_model_hash_and_nonfinite_data(
    tmp_path, official_model_bundle
):
    ledger, clock = _ledger(tmp_path / "invalid.sqlite")
    clock.set(START + timedelta(seconds=2), 0)
    bad_order = _evidence()
    bad_order["control_predictors"].reverse()
    with pytest.raises(ValueError, match="feature order"):
        ledger.append_prediction(
            bad_order,
            model_bundle=official_model_bundle,
        )

    bad_model = _evidence()
    bad_model["candidate_model_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="model provenance"):
        ledger.append_prediction(
            bad_model,
            model_bundle=official_model_bundle,
        )

    bad_values = _evidence()
    bad_values["feature_vector"]["asof_spread_bps"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        ledger.append_prediction(
            bad_values,
            model_bundle=official_model_bundle,
        )

    tampered_values = _evidence()
    tampered_values["feature_vector"]["asof_spread_bps"] = 0.5
    with pytest.raises(ValueError, match="fingerprint"):
        ledger.append_prediction(
            tampered_values, model_bundle=official_model_bundle
        )
    ledger.close()
