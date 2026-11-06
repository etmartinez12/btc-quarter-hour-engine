"""Isolated fitted models and clocks; real receiver, replay, runner and SQLite.

The loader's disk boundary is substituted only here. No production trust gate
or CLI injection option is added. Windows directory fsync is simulated using
the existing publication fixture; Ubuntu CI exercises real POSIX publication.
"""
from __future__ import annotations

import json
import os
import sqlite3
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesClassifier

from btc_quarter_hour_engine.research.prospective_clock import (
    CHRONY_CLOCK_POLICY, ClockHealthEvidence, read_chrony_clock_health,
)
from btc_quarter_hour_engine.research import prospective_inference as inference
from btc_quarter_hour_engine.research import prospective_provenance as provenance
from btc_quarter_hour_engine.research import prospective_runner as runner_module
from btc_quarter_hour_engine.research.model_freeze import OFFICIAL_STATUS, frozen_predictors
from btc_quarter_hour_engine.research.prospective_contract import (
    FULL_V2_MODEL_SHA256, RANGE_MODEL_SHA256, TRUSTED_MANIFEST_SHA256,
)
from btc_quarter_hour_engine.research.prospective_ledger import (
    ClockHealth, ClockHealthError, LatePredictionError, ProspectiveLedger,
    WINDOW_END_EXCLUSIVE,
)
from btc_quarter_hour_engine.research.prospective_runner import (
    ProspectiveInferenceRunner, RunnerConfig, SingleWriterGuard,
)
from btc_quarter_hour_engine.research.prospective_source import (
    ProspectiveRawFrameJournal, build_causal_feature_vector,
)
from tests.test_prospective_replay import (
    _populate_receiver_journal, _snapshot, _with_boundary_witnesses,
    publication_platform,
)
from tests.chrony_fixtures import chrony_evidence
from tests.test_prospective_runner import START, _config


CHRONY_FAILURES = {
    "unsynchronised": {"verified": False},
    "stale_reference": {"reference_age": 1100.0, "last_rx": 10},
    "source_lost": {"reach": "376"},
    "excessive_offset": {"system_time": "0.050100000 seconds fast"},
    "excessive_error": {"root_delay": "0.080000000", "root_dispersion": "0.020000000"},
    "future_reference": {"reference_age": -5.0, "last_rx": 0},
    "clock_jump": {"divergence": 0.2},
    "unavailable": None,
}
CHRONY_FAILURE_REASONS = {
    "unsynchronised": "leap status is not Normal",
    "stale_reference": "chrony reference evidence is stale",
    "source_lost": "selected source missed its most recent poll",
    "excessive_offset": "system offset exceeds tolerance",
    "excessive_error": "estimated error exceeds tolerance",
    "future_reference": "reference time is in the future",
    "clock_jump": "wall clock jumped while chrony evidence was observed",
    "unavailable": "chronyc -n tracking unavailable: FileNotFoundError",
}


class TestClock:
    __test__ = False

    def __init__(self, value):
        self.value = value
        self.elapsed = 0.0
        self.healthy = True
        self.failure = None
        self.reference_age = 1.0
        self.reference_time = None
        self.observations = 0

    def now(self):
        return self.value

    def advance(self, seconds):
        self.value += timedelta(seconds=seconds)
        self.elapsed += seconds

    def move_to(self, value):
        self.advance((value - self.value).total_seconds())

    def evidence(self) -> ClockHealthEvidence:
        # Real chronyc text through the real parser and shared policy.
        self.observations += 1
        failure = self.failure or (None if self.healthy else "unsynchronised")
        if failure == "unavailable":
            def missing(command, **kwargs):
                raise FileNotFoundError(command[0])

            return read_chrony_clock_health(
                now_fn=self.now, monotonic_fn=lambda: self.elapsed, runner=missing,
            )
        age = (
            self.reference_age
            if self.reference_time is None
            else (self.value - self.reference_time).total_seconds()
        )
        return chrony_evidence(
            self.value, **{"reference_age": age, **(CHRONY_FAILURES[failure] if failure else {})},
        )


class TestWaiter:
    __test__ = False

    def __init__(self, clock):
        self.clock = clock
        self.waits = []

    def wait(self, seconds):
        assert seconds > 0, "Scheduler must never busy-wait"
        self.waits.append(seconds)
        self.clock.advance(seconds)


@dataclass
class Engine:
    config: RunnerConfig
    clock: TestClock
    bundle: inference.FrozenModelBundle
    writer: ProspectiveRawFrameJournal
    ledger: ProspectiveLedger
    runner: ProspectiveInferenceRunner = field(init=False)
    preparation_report: dict | None = None

    def reopen(self):
        self.runner.close()
        self.ledger.close()
        self.ledger = ProspectiveLedger(
            self.config.prediction_ledger_path,
            clock_health=ClockHealth(
                wall_clock=self.clock.now, monotonic_clock=lambda: self.clock.elapsed,
            ),
        )
        self.runner = self.new_runner(self.ledger)
        return self.runner

    def new_runner(self, ledger):
        journal = runner_module.ProspectiveRawFrameJournal.open_read_only(self.config.source_journal_path)
        return ProspectiveInferenceRunner(
            self.config, ledger=ledger, journal=journal, model_bundle=self.bundle,
            clock=self.clock, waiter=TestWaiter(self.clock),
            clock_evidence_provider=runner_module.read_chrony_clock_health,
        )

    def run(self, count=1):
        reports = []

        def sink(report):
            if "boundary_utc" in report:
                reports.append(report)
                if len(reports) == count:
                    self.runner.stop()

        self.runner.run_forever(diagnostic_sink=sink)
        return reports


@pytest.fixture
def engine_factory(tmp_path, monkeypatch, publication_platform):
    # Test-only substitution of time sources. validate_runtime(), all official
    # bundle checks and persistence checks still execute, unchanged.
    monkeypatch.setattr(runner_module, "SystemClock", TestClock)
    monkeypatch.setattr(runner_module, "SystemWaiter", TestWaiter)
    monkeypatch.setattr(ClockHealth, "uses_system_clock", property(lambda self: True))
    if os.name != "posix":
        # Windows cannot attest directory or read-only descriptor durability.
        # This is a test simulation only; Linux CI uses unmodified fsync.
        monkeypatch.setattr(os, "fsync", lambda fd: os.fstat(fd))
    models = {}
    for name, names in frozen_predictors().items():
        labels = [0, 1, 1, 1] if name == "range_volatility" else [0, 0, 0, 1]
        models[name] = ExtraTreesClassifier(n_estimators=2, random_state=35).fit(
            pd.DataFrame(np.zeros((4, len(names))), columns=names), labels,
        )

    def disk_loader(directory, **kwargs):
        assert kwargs == {"trusted_manifest_sha256": TRUSTED_MANIFEST_SHA256, "require_official": True}
        return models, {
            "status": OFFICIAL_STATUS,
            "models": {
                "range_volatility": {"sha256": RANGE_MODEL_SHA256},
                "full_v2": {"sha256": FULL_V2_MODEL_SHA256},
            },
        }

    monkeypatch.setattr(inference, "load_frozen_models", disk_loader)
    # Test-only implementation identity: the real Git/attestation verification
    # is exercised in test_prospective_provenance. The runner's per-boundary
    # recheck of the executing package still runs unmodified here.
    package_root = provenance.imported_package_root()
    package_files = provenance.snapshot_package_files(package_root)

    def verified_test_checkout(expected, *, attestation_path=None, attestation_sha256=None, modules=None):
        return provenance.VerifiedImplementation(
            mode=provenance.MODE_GIT_CHECKOUT,
            git_sha=provenance.require_git_sha(expected),
            package_root=package_root,
            files=package_files,
            package_content_sha256=provenance.package_content_digest(package_files),
            source_root=package_root.parent,
        )

    monkeypatch.setattr(provenance, "verify_runtime_implementation", verified_test_checkout)
    engines = []

    def create(*, prepare=True, clock_verified=True, incomplete=False, source_clock_reference_age=1.0):
        root = tmp_path / f"engine-{len(engines)}"
        root.mkdir()
        config = replace(
            _config(root), expected_source_session_id="replay-session", warmup_chunk_frames=8,
        )
        cutoff = START - timedelta(minutes=15)
        boundaries = (cutoff, START, START + timedelta(minutes=15))
        source = _with_boundary_witnesses(
            _snapshot(
                start=cutoff - timedelta(hours=4) + (timedelta(minutes=15) if incomplete else timedelta()),
                end=boundaries[-1],
            ),
            boundaries,
        )
        writer = _populate_receiver_journal(
            config.source_journal_path, source, boundaries, clock_verified=clock_verified,
            clock_reference_age=source_clock_reference_age,
        )
        clock = TestClock(START - timedelta(seconds=5))
        monkeypatch.setattr(runner_module, "read_chrony_clock_health", clock.evidence)
        monkeypatch.setenv("BTC_QH_PROSPECTIVE_AUTHORIZATION_ID", config.operational_authorization_id)
        monkeypatch.setenv("BTC_QH_IMPLEMENTATION_GIT_SHA", config.implementation_git_sha)
        bundle = inference.load_official_models(root / "models")
        assert inference.is_verified_official_bundle(bundle)
        ledger = ProspectiveLedger(
            config.prediction_ledger_path,
            clock_health=ClockHealth(wall_clock=clock.now, monotonic_clock=lambda: clock.elapsed),
        )
        engine = Engine(config, clock, bundle, writer, ledger)
        engine.runner = engine.new_runner(ledger)
        engines.append(engine)
        if prepare:
            report = engine.runner.prepare_checkpoint(START)
            engine.preparation_report = report
            assert report["ready"] is True
            assert report["warmup_chunks"] >= 4
            assert report["official_records_written"] is False
            assert engine.ledger.lifecycle(START)["events"] == []
        return engine

    yield create
    for engine in engines:
        engine.runner.close()
        engine.ledger.close()
        engine.writer.close()


def test_consecutive_boundaries_have_exact_full_replay_and_prediction_parity(engine_factory):
    engine = engine_factory()
    engine.clock.move_to(START + timedelta(seconds=2))
    reports = engine.run(2)
    assert [report["status"] for report in reports] == ["predicted", "predicted"]
    for boundary in (START, START + timedelta(minutes=15)):
        state = engine.ledger.lifecycle(boundary)
        full = build_causal_feature_vector(
            engine.writer.snapshot_through(engine.config.expected_source_session_id, boundary_utc=boundary),
            boundary,
        )
        row = state["prediction"]
        assert row["feature_fingerprint"] == full.feature_fingerprint
        assert json.loads(row["feature_vector_json"]) == [
            [name, full.values[name]] for name in full.control_predictors
        ]
        assert json.loads(row["candidate_predictors_json"]) == list(full.candidate_predictors)
        assert json.loads(row["control_predictors_json"]) == list(full.control_predictors)
        assert len(full.candidate_predictors) == 12 and len(full.control_predictors) == 91
        assert row["candidate_probability"] == 0.75 and row["candidate_prediction"] == 1
        assert row["control_probability"] == 0.25 and row["control_prediction"] == 0
        assert row["baseline_prediction"] == 0
        assert row["manifest_sha256"] == TRUSTED_MANIFEST_SHA256
        assert row["candidate_model_sha256"] == RANGE_MODEL_SHA256
        assert row["control_model_sha256"] == FULL_V2_MODEL_SHA256
        assert state["evaluation_eligible"] is True
        assert state["label"] is None
        assert not set(full.control_predictors) & {"target", "label", "target_price", "target_utc"}
        assert row["target_utc"] == (boundary + timedelta(minutes=15)).isoformat(timespec="microseconds").replace("+00:00", "Z")
        assert all(value >= 0 for key, value in reports[0]["metrics"].items() if key.endswith("_ms"))
    with pytest.raises(sqlite3.IntegrityError):
        engine.ledger._connection.execute("DELETE FROM predictions")


@pytest.mark.parametrize("offset", [-1, 0, 1, 10, 30, 59, 60, 61])
def test_scheduler_startup_and_restart_open_windows(engine_factory, offset):
    engine = engine_factory()
    engine.reopen()
    engine.clock.move_to(START + timedelta(seconds=offset))
    reports = engine.run(2)
    assert [report["boundary_utc"] for report in reports] == [
        START.isoformat().replace("+00:00", "Z"),
        (START + timedelta(minutes=15)).isoformat().replace("+00:00", "Z"),
    ]
    assert reports[0]["status"] == ("missed_deadline" if offset == 61 else "predicted")
    assert reports[1]["status"] == "predicted"
    assert engine.ledger.lifecycle(START)["evaluation_eligible"] is (offset <= 60)
    assert engine.clock.now() >= START + timedelta(minutes=15)


def test_all_registered_boundaries_reconcile_without_rounding_forward():
    for index in range(2880):
        boundary = START + timedelta(minutes=15 * index)
        assert runner_module.BoundaryScheduler.most_recent_boundary(boundary + timedelta(seconds=30)) == boundary
        assert boundary < WINDOW_END_EXCLUSIVE
    assert boundary + timedelta(minutes=15) == WINDOW_END_EXCLUSIVE


def test_restart_after_complete_prediction_never_repeats_or_reattests(engine_factory):
    engine = engine_factory()
    engine.clock.move_to(START + timedelta(seconds=1))
    assert engine.runner.process_boundary(START).status == "predicted"
    before = engine.ledger.lifecycle(START)
    engine.reopen()
    reports = engine.run(2)
    assert [report["status"] for report in reports] == ["duplicate_prediction", "predicted"]
    assert engine.ledger.lifecycle(START) == before


def test_preparation_is_restartable_and_never_opens_prediction_ledger(engine_factory, monkeypatch):
    engine = engine_factory(prepare=False)
    engine.ledger.close()
    engine.config.prediction_ledger_path.unlink()
    runner = engine.new_runner(None)
    # This pre-warm-only object uses the same isolated clock health as the fixture.
    runner._preparation_clock_health = ClockHealth(
        wall_clock=engine.clock.now, monotonic_clock=lambda: engine.clock.elapsed,
    )
    original = runner_module._save_checkpoint
    publications = []

    def interrupt(directory, checkpoint, *, warmup=False):
        path = original(directory, checkpoint, warmup=warmup)
        publications.append(path)
        if warmup and len(publications) == 2:
            raise OSError("simulated crash after continuation publication")
        return path

    with monkeypatch.context() as patch:
        patch.setattr(runner_module, "_save_checkpoint", interrupt)
        with pytest.raises(OSError, match="simulated crash"):
            runner.prepare_checkpoint(START)
    saved = {path: path.read_bytes() for path in publications}
    runner.close()
    runner = engine.new_runner(None)
    runner._preparation_clock_health = ClockHealth(
        wall_clock=engine.clock.now, monotonic_clock=lambda: engine.clock.elapsed,
    )
    try:
        report = runner.prepare_checkpoint(START)
        assert report["resumed_warmup"] is True
        assert report["cutoff_utc"] == (START - timedelta(minutes=15)).isoformat().replace("+00:00", "Z")
        assert report["preparation_ms"] >= 0
        assert not engine.config.prediction_ledger_path.exists()
        assert all(path.read_bytes() == raw for path, raw in saved.items())
        again = runner.prepare_checkpoint(START)
        assert again["recovered_ready_checkpoint"] is True
        assert again["integrity_sha256"] == report["integrity_sha256"]
    finally:
        runner.close()


@pytest.mark.parametrize("fault", ["missing", "corrupt", "wrong_session", "incomplete"])
def test_checkpoints_fail_closed_without_in_deadline_cold_replay(engine_factory, monkeypatch, fault):
    engine = engine_factory(prepare=fault not in {"missing", "incomplete"}, incomplete=fault == "incomplete")
    if fault == "incomplete":
        with pytest.raises(ValueError, match="four_hour"):
            engine.runner.prepare_checkpoint(START)
    if fault in {"corrupt", "wrong_session"}:
        path = next(engine.config.checkpoint_directory.glob("checkpoint_*.json"))
        if fault == "corrupt":
            path.write_text('{"broken":', encoding="utf-8")
        else:
            # A well-formed checkpoint from another session under the expected
            # name must not be trusted even when its digest is recomputed.
            from btc_quarter_hour_engine.research.prospective_replay import _checkpoint_digest
            data = json.loads(path.read_text())
            data["session_id"] = "unrelated-session"
            data["integrity_sha256"] = _checkpoint_digest(data)
            path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(
        engine.runner.journal, "warmup_chunk",
        lambda *args, **kwargs: pytest.fail("Boundary processing must never perform cold warm-up"),
    )
    engine.clock.move_to(START + timedelta(seconds=10))
    result = engine.runner.process_boundary(START)
    assert result.status != "predicted"
    assert engine.ledger.lifecycle(START)["prediction"] is None


def test_preparation_stops_at_window_and_preserves_bounded_continuation(engine_factory, monkeypatch):
    engine = engine_factory(prepare=False)
    save = runner_module._save_checkpoint

    def delayed_save(directory, checkpoint, **kwargs):
        path = save(directory, checkpoint, **kwargs)
        engine.clock.advance(6)
        return path

    monkeypatch.setattr(runner_module, "_save_checkpoint", delayed_save)
    with pytest.raises(LatePredictionError, match="finish_before_boundary"):
        engine.runner.prepare_checkpoint(START)
    assert len(list(engine.config.checkpoint_directory.glob("warmup_*.json"))) == 1
    assert not list(engine.config.checkpoint_directory.glob("checkpoint_*.json"))
    assert engine.ledger.lifecycle(START)["events"] == []


@pytest.mark.parametrize("stage", ["before_write", "during_write", "before_attestation", "attestation_lock", "after_attestation"])
def test_independent_utc_health_at_each_durability_stage(engine_factory, monkeypatch, stage):
    engine = engine_factory()
    engine.clock.move_to(START + timedelta(seconds=2))
    original_append = engine.ledger.append_prediction
    original_infer = inference.infer_causal_prediction
    original_insert = engine.ledger._insert_event
    original_transaction = engine.ledger._transaction
    transaction_count = 0

    def infer(*args, **kwargs):
        evidence = original_infer(*args, **kwargs)
        if stage == "before_write":
            engine.clock.healthy = False
        return evidence

    def insert(boundary, event_type, *args, **kwargs):
        original_insert(boundary, event_type, *args, **kwargs)
        if stage == "during_write" and event_type == "prediction_recorded":
            engine.clock.healthy = False

    @contextmanager
    def transaction():
        nonlocal transaction_count
        transaction_count += 1
        with original_transaction():
            if stage == "attestation_lock" and transaction_count == 2:
                engine.clock.healthy = False
            yield
        if stage == "before_attestation":
            engine.clock.healthy = False

    def append(*args, **kwargs):
        evidence = original_append(*args, **kwargs)
        if stage == "after_attestation":
            engine.clock.healthy = False
        return evidence

    monkeypatch.setattr(inference, "infer_causal_prediction", infer)
    monkeypatch.setattr(engine.ledger, "_insert_event", insert)
    monkeypatch.setattr(engine.ledger, "_transaction", transaction)
    monkeypatch.setattr(engine.ledger, "append_prediction", append)
    result = engine.runner.process_boundary(START)
    state = engine.ledger.lifecycle(START)
    assert (state["prediction"] is None) is (stage == "before_write")
    assert state["evaluation_eligible"] is (stage == "after_attestation")
    if stage == "after_attestation":
        assert result.status == "predicted_clock_alert"
        assert result.evidence["timing_eligible"] is True
        assert engine.runner._load_latest_checkpoint(
            session_id=engine.config.expected_source_session_id, kind="checkpoint", cutoff_exact=START,
        ) is not None
    else:
        assert result.status == "clock_sync_failure"
        assert state["durability_attestation"] is None


@pytest.mark.parametrize("failure", sorted(CHRONY_FAILURES))
@pytest.mark.parametrize("stage", ["before_write", "before_attestation", "after_attestation"])
def test_shared_chrony_policy_governs_each_durability_decision(engine_factory, monkeypatch, stage, failure):
    engine = engine_factory()
    engine.clock.move_to(START + timedelta(seconds=2))
    original_infer = inference.infer_causal_prediction
    original_insert = engine.ledger._insert_event
    original_append = engine.ledger.append_prediction

    def infer(*args, **kwargs):
        evidence = original_infer(*args, **kwargs)
        if stage == "before_write":
            engine.clock.failure = failure
        return evidence

    def insert(boundary, event_type, *args, **kwargs):
        original_insert(boundary, event_type, *args, **kwargs)
        if stage == "before_attestation" and event_type == "prediction_recorded":
            engine.clock.failure = failure

    def append(*args, **kwargs):
        evidence = original_append(*args, **kwargs)
        if stage == "after_attestation":
            engine.clock.failure = failure
        return evidence

    monkeypatch.setattr(inference, "infer_causal_prediction", infer)
    monkeypatch.setattr(engine.ledger, "_insert_event", insert)
    monkeypatch.setattr(engine.ledger, "append_prediction", append)
    result = engine.runner.process_boundary(START)
    state = engine.ledger.lifecycle(START)
    assert CHRONY_FAILURE_REASONS[failure] in result.reason
    if stage == "after_attestation":
        # A later failure is an operational alert; it cannot revoke evidence
        # that was already certified and attested in time.
        assert result.status == "predicted_clock_alert"
        assert state["evaluation_eligible"] is True
        assert state["durability_attestation"] is not None
    else:
        assert result.status == "clock_sync_failure"
        assert state["evaluation_eligible"] is False
        assert state["durability_attestation"] is None
        assert (state["prediction"] is None) is (stage == "before_write")


def test_mid_poll_chrony_reference_age_remains_certified_and_eligible(engine_factory):
    # Ubuntu/GCE acceptance: ~1,033 s between chrony updates at poll 10.
    engine = engine_factory(source_clock_reference_age=1033.0)
    engine.clock.reference_age = 1033.0
    engine.clock.move_to(START + timedelta(seconds=2))
    result = engine.runner.process_boundary(START)
    state = engine.ledger.lifecycle(START)
    assert result.status == "predicted"
    assert state["evaluation_eligible"] is True
    assert state["prediction"]["clock_health_verified"] == 1
    assert state["durability_attestation"]["clock_health_verified"] == 1


def test_reference_crossing_the_freshness_horizon_during_write_blocks_attestation(
    engine_factory, monkeypatch,
):
    horizon = CHRONY_CLOCK_POLICY.reference_age_horizon(10)
    engine = engine_factory()
    engine.clock.move_to(START + timedelta(seconds=2))
    # Fresh (1,059 s < 1,060.48 s) at pre-persist, stale (1,062 s) at attestation.
    engine.clock.reference_time = engine.clock.value - timedelta(seconds=1059)
    original_insert = engine.ledger._insert_event

    def insert(boundary, event_type, *args, **kwargs):
        original_insert(boundary, event_type, *args, **kwargs)
        if event_type == "prediction_recorded":
            engine.clock.advance(3)

    monkeypatch.setattr(engine.ledger, "_insert_event", insert)
    observations = engine.clock.observations
    result = engine.runner.process_boundary(START)
    state = engine.ledger.lifecycle(START)
    assert 1059 < horizon < 1062
    assert engine.clock.observations - observations >= 2
    assert result.status == "clock_sync_failure"
    assert CHRONY_FAILURE_REASONS["stale_reference"] in result.reason
    assert state["prediction"] is not None
    assert state["durability_attestation"] is None
    assert state["evaluation_eligible"] is False


def test_clock_freshness_evidence_does_not_change_features_or_predictions(engine_factory):
    fresh = engine_factory()
    mid_poll = engine_factory(source_clock_reference_age=1033.0)
    mid_poll.clock.reference_age = 1000.0
    rows = []
    for engine in (fresh, mid_poll):
        engine.clock.move_to(START + timedelta(seconds=2))
        assert engine.runner.process_boundary(START).status == "predicted"
        rows.append(engine.ledger.lifecycle(START)["prediction"])
    fields = (
        "target_utc", "source_watermark_utc", "source_frame_count",
        "excluded_ingest_future_count", "excluded_event_future_count",
        "feature_contract", "feature_family_contract", "feature_fingerprint",
        "candidate_predictors_json", "control_predictors_json", "feature_vector_json",
        "candidate_model_sha256", "control_model_sha256", "candidate_probability",
        "candidate_prediction", "control_probability", "control_prediction",
        "baseline_prediction",
    )
    assert {name: rows[0][name] for name in fields} == {name: rows[1][name] for name in fields}
    assert rows[0]["source_watermark_utc"] == rows[0]["boundary_utc"]
    assert rows[0]["baseline_prediction"] == 0
    assert "target" not in json.dumps(json.loads(rows[0]["feature_vector_json"])).lower()


def test_commit_crossing_deadline_is_durable_but_ineligible(engine_factory, monkeypatch):
    engine = engine_factory()
    engine.clock.move_to(START + timedelta(seconds=59))
    original = engine.ledger._transaction

    @contextmanager
    def transaction():
        with original():
            yield
        engine.clock.advance(2)

    monkeypatch.setattr(engine.ledger, "_transaction", transaction)
    result = engine.runner.process_boundary(START)
    state = engine.ledger.lifecycle(START)
    assert result.status == "missed_deadline"
    assert state["prediction"] is not None
    assert state["status"] == "durability_late"
    assert state["evaluation_eligible"] is False


def test_prediction_survives_failed_checkpoint_publication_and_restart(engine_factory, monkeypatch):
    engine = engine_factory()
    engine.clock.move_to(START + timedelta(seconds=2))
    with monkeypatch.context() as patch:
        def fail(*args, **kwargs):
            raise OSError("checkpoint disk unavailable")
        patch.setattr(runner_module, "_save_checkpoint", fail)
        result = engine.runner.process_boundary(START)
    assert result.status == "prediction_committed_checkpoint_recovery_required"
    before = engine.ledger.lifecycle(START)
    engine.reopen()
    assert engine.runner.process_boundary(START).status == "duplicate_prediction"
    assert engine.ledger.lifecycle(START) == before
    # Future state advances from the last trustworthy prefix, without repairing
    # or repeating an old prediction on the scheduler's critical path.
    engine.clock.move_to(START + timedelta(minutes=15, seconds=2))
    assert engine.runner.process_boundary(START + timedelta(minutes=15)).status == "predicted"


def test_crash_after_commit_never_reattests_on_restart(engine_factory, monkeypatch):
    engine = engine_factory()
    engine.clock.move_to(START + timedelta(seconds=2))
    with monkeypatch.context() as patch:
        def crash(*args, **kwargs):
            raise KeyboardInterrupt("simulated process termination")
        patch.setattr(engine.ledger, "_attest_durable_commit", crash)
        with pytest.raises(KeyboardInterrupt):
            engine.runner.process_boundary(START)
    before = engine.ledger.lifecycle(START)
    assert before["status"] == "durability_unconfirmed"
    engine.reopen()
    assert engine.run()[0]["status"] == "duplicate_prediction"
    assert engine.ledger.lifecycle(START) == before
    assert before["evaluation_eligible"] is False


def test_competing_runner_cannot_acquire_scheduler_lock(engine_factory):
    engine = engine_factory()
    engine.clock.move_to(START)
    with SingleWriterGuard(engine.config.prediction_ledger_path):
        with pytest.raises(RuntimeError, match="Another prospective runner"):
            engine.runner.run_forever()
    assert engine.ledger.lifecycle(START)["events"] == []
    assert engine.run()[0]["status"] == "predicted"


@pytest.mark.parametrize("read", ["lifecycle", "fence", "tail"])
def test_transient_sqlite_read_contention_retries_without_duplicate_writes(engine_factory, monkeypatch, read):
    engine = engine_factory()
    engine.clock.move_to(START + timedelta(seconds=30))
    owner, name = {
        "lifecycle": (engine.ledger, "lifecycle"),
        "fence": (engine.runner.journal, "has_fence_record"),
        "tail": (engine.runner.journal, "tail_snapshot"),
    }[read]
    original = getattr(owner, name)
    calls = 0

    def contend(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise sqlite3.OperationalError("database is locked")
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, contend)
    result = engine.runner.process_boundary(START)
    assert result.status == "predicted"
    assert result.metrics["read_retries"] == 2
    assert engine.runner.scheduler.waiter.waits == [0.25, 0.25]
    assert sum(event["event_type"] == "prediction_recorded" for event in engine.ledger.lifecycle(START)["events"]) == 1


def test_permanent_read_contention_is_bounded_and_reports_recording_failure(engine_factory, monkeypatch):
    engine = engine_factory()
    engine.clock.move_to(START + timedelta(seconds=30))
    calls = 0

    def contend(*args):
        nonlocal calls
        calls += 1
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(engine.ledger, "lifecycle", contend)
    result = engine.runner.process_boundary(START)
    assert result.status == "ledger_or_source_io_failure"
    assert result.metrics["read_retries"] == 2
    assert "failure_record_error" in result.metrics
    assert calls <= 5


@pytest.mark.parametrize("arrival", [2, 59, 61, None])
def test_exact_fence_arrival_deadline_without_poll_event_flood(engine_factory, monkeypatch, arrival):
    engine = engine_factory()
    engine.clock.move_to(START)
    original = engine.runner.journal.has_fence_record

    def available(session_id, boundary):
        return arrival is not None and engine.clock.now() >= boundary + timedelta(seconds=arrival) and original(session_id, boundary)

    monkeypatch.setattr(engine.runner.journal, "has_fence_record", available)
    result = engine.runner.process_boundary(START)
    state = engine.ledger.lifecycle(START)
    assert state["evaluation_eligible"] is (arrival is not None and arrival <= 60)
    assert result.status == ("predicted" if state["evaluation_eligible"] else "missing_source_data")
    assert len(state["events"]) <= 2
    assert engine.clock.now() <= START + timedelta(seconds=60)


def test_unhealthy_source_fence_cannot_prepare_or_predict(engine_factory):
    engine = engine_factory(prepare=False, clock_verified=False)
    with pytest.raises(ValueError, match="source_fence"):
        engine.runner.prepare_checkpoint(START)
    assert not list(engine.config.checkpoint_directory.glob("checkpoint_*.json"))
    assert engine.ledger.lifecycle(START)["events"] == []


def test_wall_clock_jump_blocks_inference(engine_factory):
    engine = engine_factory()
    engine.runner.validate_runtime()
    engine.clock.value += timedelta(seconds=10)
    result = engine.runner.process_boundary(START)
    assert result.status == "clock_sync_failure"
    assert engine.ledger.lifecycle(START)["evaluation_eligible"] is False


def test_stop_during_wait_and_restart_do_not_write_early(engine_factory, monkeypatch):
    engine = engine_factory()
    waiter = engine.runner.scheduler.waiter
    original = waiter.wait

    def stop(seconds):
        original(seconds)
        engine.runner.stop()

    monkeypatch.setattr(waiter, "wait", stop)
    engine.runner.run_forever()
    assert engine.ledger.lifecycle(START)["events"] == []
    engine.reopen()
    assert engine.run()[0]["status"] == "predicted"


def test_real_sqlite_exclusive_read_contention_is_released_by_fake_waiter(engine_factory, monkeypatch):
    engine = engine_factory()
    engine.clock.move_to(START + timedelta(seconds=30))
    engine.runner.journal.close()
    # WAL normally isolates the consumer from receiver writes. DELETE mode
    # exercises a real SQLITE_BUSY failure deterministically, without threads.
    engine.writer._connection.execute("PRAGMA journal_mode=DELETE")
    engine.runner.journal = runner_module.ProspectiveRawFrameJournal.open_read_only(engine.config.source_journal_path)
    engine.runner.journal._connection.execute("PRAGMA busy_timeout=0")
    engine.writer._connection.execute("BEGIN EXCLUSIVE")
    original = engine.runner.scheduler.waiter.wait

    def release(seconds):
        engine.writer._connection.execute("ROLLBACK")
        original(seconds)

    monkeypatch.setattr(engine.runner.scheduler.waiter, "wait", release)
    try:
        result = engine.runner.process_boundary(START)
        assert result.status == "predicted"
        assert result.metrics["read_retries"] == 1
    finally:
        if engine.writer._connection.in_transaction:
            engine.writer._connection.execute("ROLLBACK")


def test_read_retry_cannot_extend_deadline(engine_factory, monkeypatch):
    engine = engine_factory()
    engine.clock.move_to(START + timedelta(seconds=59.9))
    calls = 0

    def busy(*args):
        nonlocal calls
        calls += 1
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(engine.runner.journal, "has_fence_record", busy)
    result = engine.runner.process_boundary(START)
    assert result.status == "ledger_or_source_io_failure"
    assert calls == 2
    assert engine.clock.now() == START + timedelta(seconds=60)
    assert engine.ledger.lifecycle(START)["prediction"] is None


@pytest.mark.parametrize("terminal", ["missed", "labeled"])
def test_terminal_lifecycle_is_not_hidden_by_a_later_diagnostic(engine_factory, terminal):
    engine = engine_factory()
    engine.clock.move_to(START + timedelta(seconds=1))
    if terminal == "labeled":
        # Simulate a ledger from a prior process whose UTC was ahead. The runner
        # must honor the immutable label rather than backfill in its local window.
        future = START + timedelta(minutes=15)
        engine.ledger.clock_health = ClockHealth(
            wall_clock=lambda: future, monotonic_clock=lambda: 0,
        )
        engine.ledger.attach_label(
            START, target=0, target_utc=future, available_at_utc=future,
            target_eligible=True, source_provenance_sha256="b" * 64,
        )
    else:
        engine.ledger.append_attempt(
            START, event_type="missed", recorded_at_utc=START + timedelta(seconds=61),
            reason="prior_process_missed_deadline",
        )
    engine.ledger.append_attempt(
        START, event_type="inference_failed", recorded_at_utc=START + timedelta(seconds=62),
        reason="subsequent_diagnostic",
    )
    result = engine.runner.process_boundary(START)
    assert result.status == "missed_deadline"
    assert engine.ledger.lifecycle(START)["prediction"] is None


def test_preparation_revalidates_ready_checkpoint_source_prefix(engine_factory, monkeypatch):
    engine = engine_factory()
    original = engine.runner.journal.tail_snapshot

    def wrong_prefix(session_id, **kwargs):
        return original(session_id, **{
            **kwargs, "expected_prefix_commitment_sha256": "f" * 64,
        })

    monkeypatch.setattr(engine.runner.journal, "tail_snapshot", wrong_prefix)
    with pytest.raises(ValueError, match="prefix commitment"):
        engine.runner.prepare_checkpoint(START)
    assert engine.ledger.lifecycle(START)["events"] == []


def test_missing_preparation_fence_cannot_use_a_later_fence(engine_factory, monkeypatch):
    engine = engine_factory(prepare=False)
    original = engine.runner.journal.warmup_chunk

    def substitute(session_id, **kwargs):
        return original(session_id, **{**kwargs, "through_boundary_utc": START})

    monkeypatch.setattr(engine.runner.journal, "warmup_chunk", substitute)
    with pytest.raises(ValueError, match="substitute_source_fence"):
        engine.runner.prepare_checkpoint(START)
    assert engine.ledger.lifecycle(START)["events"] == []


def test_configuration_diagnostic_and_preparation_cli_never_open_ledger(engine_factory, monkeypatch, capsys):
    from btc_quarter_hour_engine.research import prospective_runner_cli as cli

    engine = engine_factory(prepare=False)
    config_path = engine.config.checkpoint_directory.parent / "runner.json"
    config_path.write_text(json.dumps({
        key: str(value) if isinstance(value, Path) else value
        for key, value in engine.config.__dict__.items()
    }), encoding="utf-8")
    engine.ledger.close()
    engine.config.prediction_ledger_path.unlink()
    real_runner = ProspectiveInferenceRunner

    def test_runner(config, *, ledger, journal):
        assert ledger is None
        runner = real_runner(
            config, ledger=None, journal=journal, clock=engine.clock,
            waiter=TestWaiter(engine.clock),
            clock_evidence_provider=runner_module.read_chrony_clock_health,
        )
        runner._preparation_clock_health = ClockHealth(
            wall_clock=engine.clock.now, monotonic_clock=lambda: engine.clock.elapsed,
        )
        return runner

    monkeypatch.setattr(cli, "ProspectiveInferenceRunner", test_runner)
    for mode in ("validate", "dry-run", "prepare", "simulate"):
        args = [mode]
        if mode != "simulate":
            args += ["--config", str(config_path)]
        if mode in {"dry-run", "prepare"}:
            args += ["--boundary", "2026-11-15T00:00:00Z"]
        assert cli.main(args) == 0
        output = json.loads(capsys.readouterr().out)
        assert output.get("official_records_written", False) is False
        assert not engine.config.prediction_ledger_path.exists()


def test_offline_preparation_and_restart_measurements(engine_factory, record_property):
    tracemalloc.start()
    try:
        engine = engine_factory()
        engine.reopen()
        recovered = engine.runner.prepare_checkpoint(START)
        engine.clock.move_to(START + timedelta(seconds=2))
        reports = engine.run(2)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert all(report["status"] == "predicted" for report in reports)
    assert recovered["recovered_ready_checkpoint"] is True
    record_property("synthetic_preparation_ms", engine.preparation_report["preparation_ms"])
    record_property("ready_checkpoint_reload_ms", recovered["preparation_ms"])
    record_property("warmup_frames", engine.preparation_report["warmup_frames"])
    record_property("warmup_chunks", engine.preparation_report["warmup_chunks"])
    record_property("traced_peak_bytes", peak)
    record_property("boundary_latency", json.dumps(runner_module.summarize_latency_samples([
        report["metrics"]["total_boundary_completion_ms"] for report in reports
    ])))
    record_property("stage_metrics", json.dumps([report["metrics"] for report in reports]))
    record_property("checkpoint_fsync", "POSIX" if os.name == "posix" else "simulated-Windows")


def test_scheduler_retries_transient_startup_reads(engine_factory, monkeypatch):
    engine = engine_factory()
    engine.clock.move_to(START + timedelta(seconds=30))
    original = engine.runner.journal.session_mode
    calls = 0

    def busy_once(session_id):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise sqlite3.OperationalError("database is locked")
        return original(session_id)

    monkeypatch.setattr(engine.runner.journal, "session_mode", busy_once)
    assert engine.run()[0]["status"] == "predicted"
    assert engine.runner.scheduler.waiter.waits == [0.25]


def test_historical_completed_prediction_is_not_reported_as_missing(engine_factory):
    engine = engine_factory()
    engine.clock.move_to(START + timedelta(seconds=2))
    assert engine.runner.process_boundary(START).status == "predicted"
    before = engine.ledger.lifecycle(START)
    engine.reopen()
    engine.clock.move_to(START + timedelta(seconds=61))
    assert engine.run()[0]["status"] == "duplicate_prediction"
    assert engine.ledger.lifecycle(START) == before


def test_clock_jump_past_deadline_reports_unconfirmed_commit(engine_factory, monkeypatch):
    engine = engine_factory()
    engine.clock.move_to(START + timedelta(seconds=2))
    original = engine.ledger._transaction

    @contextmanager
    def transaction():
        with original():
            yield
        engine.clock.value += timedelta(seconds=70)

    monkeypatch.setattr(engine.ledger, "_transaction", transaction)
    result = engine.runner.process_boundary(START)
    state = engine.ledger.lifecycle(START)
    assert result.status == "clock_sync_failure"
    assert "missed_reconciliation_error" in result.metrics
    assert state["status"] == "durability_unconfirmed"
    assert state["evaluation_eligible"] is False


def test_failed_subsequent_alert_logging_retains_successful_evidence(engine_factory, monkeypatch):
    engine = engine_factory()
    engine.clock.move_to(START + timedelta(seconds=2))
    original = engine.ledger.append_prediction

    def append(*args, **kwargs):
        result = original(*args, **kwargs)
        engine.clock.healthy = False
        return result

    def busy(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(engine.ledger, "append_prediction", append)
    monkeypatch.setattr(engine.ledger, "append_attempt", busy)
    result = engine.runner.process_boundary(START)
    assert result.status == "predicted_clock_alert"
    assert result.evidence["timing_eligible"] is True
    assert "failure_record_error" in result.metrics
    assert engine.ledger.lifecycle(START)["evaluation_eligible"] is True


_PROVENANCE_FAILURES = {
    "unexpected_commit": "executing checkout HEAD " + "c" * 40 + " does not match expected implementation",
    "modified_source": "Git working tree is not clean:  M btc_quarter_hour_engine/research/prospective_runner.py",
    "mismatched_installed_package": "installed package files differ from attestation",
    "unverifiable": "executing package is not in a Git checkout and no implementation attestation is "
    "configured; implementation identity is unverifiable",
}


def _substitute_verifier(kind):
    def verifier(expected, *, attestation_path=None, attestation_sha256=None, modules=None):
        if kind == "mismatched_identity":
            root = provenance.imported_package_root()
            files = provenance.snapshot_package_files(root)
            return provenance.VerifiedImplementation(
                mode=provenance.MODE_GIT_CHECKOUT, git_sha="b" * 40, package_root=root,
                files=files, package_content_sha256=provenance.package_content_digest(files),
            )
        raise provenance.ImplementationProvenanceError(_PROVENANCE_FAILURES[kind])

    return verifier


def _write_cli_config(engine):
    path = engine.config.checkpoint_directory.parent / "runner.json"
    path.write_text(json.dumps({
        key: str(value) if isinstance(value, Path) else value
        for key, value in engine.config.__dict__.items()
    }), encoding="utf-8")
    return path


def test_ledger_binds_the_independently_verified_implementation(engine_factory):
    engine = engine_factory()
    identity = engine.runner.verified_implementation
    assert identity is not None and identity.mode == provenance.MODE_GIT_CHECKOUT
    assert engine.preparation_report["implementation_git_sha"] == identity.git_sha
    assert engine.preparation_report["implementation_provenance_mode"] == identity.mode
    assert engine.preparation_report["implementation_package_sha256"] == identity.package_content_sha256
    engine.clock.move_to(START + timedelta(seconds=2))
    reports = engine.run(2)
    assert [report["status"] for report in reports] == ["predicted", "predicted"]
    for report, boundary in zip(reports, (START, START + timedelta(minutes=15))):
        assert report["metrics"]["implementation_git_sha"] == identity.git_sha
        assert report["metrics"]["implementation_provenance_mode"] == identity.mode
        row = engine.ledger.lifecycle(boundary)["prediction"]
        assert row["implementation_git_sha"] == identity.git_sha == engine.config.implementation_git_sha


@pytest.mark.parametrize(
    "kind", [*sorted(_PROVENANCE_FAILURES), "mismatched_identity"],
)
def test_unverified_implementation_fails_closed_without_prediction(engine_factory, monkeypatch, kind):
    engine = engine_factory(prepare=False)
    # The operator acknowledgement still matches; it is not proof of identity.
    assert os.environ["BTC_QH_IMPLEMENTATION_GIT_SHA"] == engine.config.implementation_git_sha
    monkeypatch.setattr(provenance, "verify_runtime_implementation", _substitute_verifier(kind))
    with pytest.raises(provenance.ImplementationProvenanceError):
        engine.runner.prepare_checkpoint(START)
    assert not any(engine.config.checkpoint_directory.glob("*.json"))
    with pytest.raises(provenance.ImplementationProvenanceError):
        engine.runner.validate_runtime()
    engine.clock.move_to(START + timedelta(seconds=2))
    result = engine.runner.process_boundary(START)
    assert result.status == "implementation_provenance_failure"
    assert result.reason.startswith("ImplementationProvenanceError:")
    assert result.evidence is None
    state = engine.ledger.lifecycle(START)
    assert state["prediction"] is None
    assert [event["event_type"] for event in state["events"]] == ["inference_failed"]
    with pytest.raises(provenance.ImplementationProvenanceError):
        engine.runner.run_forever()
    assert engine.runner.verified_implementation is None


@pytest.mark.parametrize("drift", ["modified_package_file", "foreign_package_module"])
def test_implementation_drift_after_startup_blocks_later_boundaries(engine_factory, monkeypatch, tmp_path, drift):
    import importlib.util
    import sys

    engine = engine_factory()
    engine.clock.move_to(START + timedelta(seconds=2))
    assert engine.runner.process_boundary(START).status == "predicted"
    first = engine.ledger.lifecycle(START)
    if drift == "modified_package_file":
        original = provenance.snapshot_package_files

        def drifted(root):
            files = list(original(root))
            path, _, size = files[-1]
            files[-1] = (path, "0" * 64, size)
            return tuple(files)

        monkeypatch.setattr(provenance, "snapshot_package_files", drifted)
        expected = "package files changed after verification"
    else:
        rogue = tmp_path / "rogue.py"
        rogue.write_text("VALUE = 1\n", encoding="utf-8")
        name = "btc_quarter_hour_engine.research._unverified_rogue"
        spec = importlib.util.spec_from_file_location(name, rogue)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setitem(sys.modules, name, module)
        expected = "originates outside the verified package"
    second = START + timedelta(minutes=15)
    engine.clock.move_to(second + timedelta(seconds=2))
    result = engine.runner.process_boundary(second)
    assert result.status == "implementation_provenance_failure"
    assert expected in result.reason
    assert engine.ledger.lifecycle(second)["prediction"] is None
    assert engine.ledger.lifecycle(START) == first


def test_official_and_validate_cli_refuse_unverified_implementation(engine_factory, monkeypatch, capsys):
    from btc_quarter_hour_engine.research import prospective_runner_cli as cli

    engine = engine_factory(prepare=False)
    config_path = _write_cli_config(engine)
    engine.ledger.close()
    engine.config.prediction_ledger_path.unlink()
    monkeypatch.setattr(provenance, "verify_runtime_implementation", _substitute_verifier("unexpected_commit"))
    assert cli.main([
        "official", "--config", str(config_path),
        "--authorization-id", engine.config.operational_authorization_id,
        "--confirm-authorized-deployment",
    ]) == 2
    assert "ImplementationProvenanceError" in capsys.readouterr().err
    assert not engine.config.prediction_ledger_path.exists()
    assert cli.main(["validate", "--config", str(config_path)]) == 2
    assert "does not match expected implementation" in capsys.readouterr().err
    assert not engine.config.prediction_ledger_path.exists()


def test_verify_implementation_cli_reports_identity_without_side_effects(engine_factory, capsys):
    from btc_quarter_hour_engine.research import prospective_runner_cli as cli

    assert cli.main(["verify-implementation", "--implementation-git-sha", "a" * 40]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["verified"] is True
    assert output["implementation"]["git_sha"] == "a" * 40
    assert output["implementation"]["package_file_count"] > 0
    assert output["official_records_written"] is False and output["ledger_opened"] is False
    assert cli.main([
        "verify-implementation", "--implementation-git-sha", "a" * 40, "--attestation", "x.json",
    ]) == 2
    assert "must be supplied together" in capsys.readouterr().err
