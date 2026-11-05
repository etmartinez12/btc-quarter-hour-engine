from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from btc_quarter_hour_engine.research.prospective_ledger import ClockHealth, ProspectiveLedger
from btc_quarter_hour_engine.research.prospective_runner import (
    BoundaryScheduler,
    ProspectiveInferenceRunner,
    RunnerConfig,
    SingleWriterGuard,
    summarize_latency_samples,
)
from btc_quarter_hour_engine.research.prospective_source import ProspectiveRawFrameJournal


UTC = timezone.utc
START = datetime(2026, 11, 15, 0, 0, tzinfo=UTC)


class ManualClock:
    def __init__(self, value: datetime):
        self.value = value

    def now(self) -> datetime:
        return self.value


class AdvancingWaiter:
    def __init__(self, clock: ManualClock):
        self.clock = clock
        self.waits: list[float] = []

    def wait(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.clock.value += timedelta(seconds=seconds)


def _config(tmp_path: Path, *, source: Path | None = None) -> RunnerConfig:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    source_path = source or tmp_path / "source.sqlite"
    return RunnerConfig(
        source_journal_path=source_path,
        checkpoint_directory=checkpoint_dir,
        prediction_ledger_path=tmp_path / "ledger.sqlite",
        official_model_directory=model_dir,
        implementation_git_sha="a" * 40,
        expected_source_session_id="expected-bridge-session",
        operational_authorization_id="deployment-approval-35",
    )


def test_scheduler_uses_exact_utc_quarter_hours_and_never_fires_early():
    before = datetime(2026, 11, 15, 0, 14, 59, 999999, tzinfo=UTC)
    exact = datetime(2026, 11, 15, 0, 15, tzinfo=UTC)
    assert BoundaryScheduler.next_boundary(before) == exact
    assert BoundaryScheduler.next_boundary(exact) == exact
    assert not BoundaryScheduler.is_open(exact, exact - timedelta(microseconds=1))
    assert BoundaryScheduler.is_open(exact, exact + timedelta(seconds=60))
    assert not BoundaryScheduler.is_open(exact, exact + timedelta(seconds=60, microseconds=1))

    clock = ManualClock(exact - timedelta(seconds=2))
    waiter = AdvancingWaiter(clock)
    scheduler = BoundaryScheduler(clock=clock, waiter=waiter, poll_seconds=0.5)
    assert scheduler.wait_for_boundary(exact)
    assert clock.value == exact
    assert waiter.waits == [0.5, 0.5, 0.5, 0.5]


def test_scheduler_refuses_historical_boundary_without_backfilling():
    clock = ManualClock(START + timedelta(seconds=61))
    scheduler = BoundaryScheduler(clock=clock, waiter=AdvancingWaiter(clock))
    assert not scheduler.wait_for_boundary(START)


def test_runner_configuration_rejects_shared_paths_and_bad_limits(tmp_path):
    config = _config(tmp_path)
    config.validate(require_existing=False)
    with pytest.raises(ValueError, match="distinct"):
        RunnerConfig(
            **{
                **config.__dict__,
                "prediction_ledger_path": config.source_journal_path,
            }
        ).validate(require_existing=False)
    with pytest.raises(ValueError, match="tail"):
        RunnerConfig(**{**config.__dict__, "max_tail_frames": 100_001}).validate(
            require_existing=False
        )


def test_source_consumer_opens_query_only_and_does_not_change_journal(tmp_path):
    path = tmp_path / "source.sqlite"
    journal = ProspectiveRawFrameJournal(path)
    journal.start_session(
        "read-only-session", datetime(2026, 11, 15, tzinfo=UTC), source_mode="bridge"
    )
    journal.close()

    consumer = ProspectiveRawFrameJournal.open_read_only(path)
    assert consumer.session_mode("read-only-session") == "bridge"
    assert not consumer.has_fence_record("read-only-session", START)
    assert consumer._connection.execute("PRAGMA query_only").fetchone()[0] == 1
    with pytest.raises(Exception):
        consumer.start_session(
            "must-not-write", datetime(2026, 11, 15, tzinfo=UTC), source_mode="bridge"
        )
    assert consumer.session_started_at("must-not-write") is None
    consumer.close()


def test_single_writer_lock_excludes_a_competing_runner(tmp_path):
    ledger_path = tmp_path / "ledger.sqlite"
    with SingleWriterGuard(ledger_path):
        with pytest.raises(RuntimeError, match="Another prospective runner"):
            with SingleWriterGuard(ledger_path):
                pass
    with SingleWriterGuard(ledger_path):
        pass


def test_latency_summaries_include_percentiles_and_empty_case():
    assert summarize_latency_samples([]) == {
        "count": 0,
        "median_ms": None,
        "p95_ms": None,
        "p99_ms": None,
        "max_ms": None,
    }
    summary = summarize_latency_samples([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert summary == {
        "count": 10,
        "median_ms": 5.5,
        "p95_ms": 10.0,
        "p99_ms": 10.0,
        "max_ms": 10.0,
    }


def test_official_clock_health_can_be_distinguished_from_test_clock():
    assert ClockHealth().uses_system_clock is True
    assert ClockHealth(wall_clock=lambda: START).uses_system_clock is False


def test_runner_missed_boundary_is_reported_without_prediction_write(tmp_path):
    config = _config(tmp_path)
    journal_writer = ProspectiveRawFrameJournal(config.source_journal_path)
    journal_writer.start_session(
        "expected-bridge-session",
        START - timedelta(hours=5),
        source_mode="bridge",
    )
    journal_writer.close()
    journal = ProspectiveRawFrameJournal.open_read_only(config.source_journal_path)
    ledger = ProspectiveLedger(
        config.prediction_ledger_path,
        window_start=START,
        window_end_exclusive=START + timedelta(minutes=15),
    )
    clock = ManualClock(START + timedelta(seconds=61))
    runner = ProspectiveInferenceRunner(
        config,
        ledger=ledger,
        journal=journal,
        clock=clock,
        official_mode=False,
    )
    result = runner.process_boundary(START)
    assert result.status == "missed_deadline"
    assert "never backfilled" in result.reason
    assert ledger.lifecycle(START)["prediction"] is None
    runner.close()
    ledger.close()


def test_synthetic_cli_does_not_open_a_ledger_or_start_inference(capsys):
    from btc_quarter_hour_engine.research.prospective_runner_cli import main

    assert main(["simulate", "--count", "2880"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["scheduled_boundaries"] == 2880
    assert output["official_records_written"] is False
    assert output["official_models_loaded"] is False


def test_read_only_cli_diagnostic_does_not_create_prediction_ledger(tmp_path, capsys):
    from btc_quarter_hour_engine.research.prospective_runner_cli import main

    config = _config(tmp_path)
    writer = ProspectiveRawFrameJournal(config.source_journal_path)
    writer.start_session(
        config.expected_source_session_id,
        START - timedelta(hours=5),
        source_mode="bridge",
    )
    writer.close()
    document = {
        "source_journal_path": str(config.source_journal_path),
        "checkpoint_directory": str(config.checkpoint_directory),
        "prediction_ledger_path": str(config.prediction_ledger_path),
        "official_model_directory": str(config.official_model_directory),
        "implementation_git_sha": config.implementation_git_sha,
        "expected_source_session_id": config.expected_source_session_id,
        "operational_authorization_id": config.operational_authorization_id,
        "source_poll_seconds": config.source_poll_seconds,
        "max_tail_frames": config.max_tail_frames,
        "warmup_chunk_frames": config.warmup_chunk_frames,
    }
    config_path = tmp_path / "runner.json"
    config_path.write_text(json.dumps(document), encoding="utf-8")
    assert main([
        "dry-run",
        "--config",
        str(config_path),
        "--boundary",
        "2026-11-15T00:00:00Z",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["exact_boundary_fence_present"] is False
    assert result["official_records_written"] is False
    assert result["models_loaded"] is False
    assert not config.prediction_ledger_path.exists()
