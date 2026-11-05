from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from btc_quarter_hour_engine.research.prospective_inference import load_official_models
from btc_quarter_hour_engine.research.prospective_ledger import ProspectiveLedger
from btc_quarter_hour_engine.research.prospective_runner import (
    BoundaryScheduler,
    ProspectiveInferenceRunner,
    RunnerConfig,
)
from btc_quarter_hour_engine.research.prospective_source import (
    ProspectiveRawFrameJournal,
)


_CONFIG_KEYS = {
    "source_journal_path",
    "checkpoint_directory",
    "prediction_ledger_path",
    "official_model_directory",
    "implementation_git_sha",
    "expected_source_session_id",
    "operational_authorization_id",
    "source_poll_seconds",
    "max_tail_frames",
    "warmup_chunk_frames",
}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key}")
        result[key] = value
    return result


def _read_config(path: Path) -> RunnerConfig:
    config = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
    )
    if not isinstance(config, dict) or set(config) != _CONFIG_KEYS:
        raise ValueError(f"Configuration must contain exactly: {', '.join(sorted(_CONFIG_KEYS))}")
    return RunnerConfig(
        source_journal_path=Path(config["source_journal_path"]),
        checkpoint_directory=Path(config["checkpoint_directory"]),
        prediction_ledger_path=Path(config["prediction_ledger_path"]),
        official_model_directory=Path(config["official_model_directory"]),
        implementation_git_sha=config["implementation_git_sha"],
        expected_source_session_id=config["expected_source_session_id"],
        operational_authorization_id=config["operational_authorization_id"],
        source_poll_seconds=config["source_poll_seconds"],
        max_tail_frames=config["max_tail_frames"],
        warmup_chunk_frames=config["warmup_chunk_frames"],
    )


def _utc(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("Boundary must contain an explicit UTC offset")
    return result.astimezone(timezone.utc)


def _emit_diagnostic(value: dict[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Controlled prospective inference runner; import never starts inference."
    )
    commands = parser.add_subparsers(dest="mode", required=True)
    validate = commands.add_parser("validate", help="Validate configuration and pinned official models")
    validate.add_argument("--config", type=Path, required=True)
    dry_run = commands.add_parser(
        "dry-run", help="Read-only diagnostics; never loads models or writes the ledger"
    )
    dry_run.add_argument("--config", type=Path, required=True)
    dry_run.add_argument("--boundary", required=True)
    prepare = commands.add_parser(
        "prepare", help="Explicit source-verified pre-window warm-up; never opens a prediction ledger"
    )
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--boundary", required=True)
    simulate = commands.add_parser(
        "simulate", help="Synthetic registered-schedule and deadline smoke test"
    )
    simulate.add_argument("--count", type=int, default=2880)
    official = commands.add_parser(
        "official", help="Start explicitly authorized official inference (deployment only)"
    )
    official.add_argument("--config", type=Path, required=True)
    official.add_argument("--authorization-id", required=True)
    official.add_argument(
        "--confirm-authorized-deployment",
        action="store_true",
        help="Required second explicit acknowledgement before starting the runner.",
    )
    args = parser.parse_args(argv)

    try:
        if args.mode == "simulate":
            if type(args.count) is not int or not 1 <= args.count <= 2880:
                raise ValueError("--count must be between 1 and 2880")
            from btc_quarter_hour_engine.research.prospective_ledger import (
                WINDOW_END_EXCLUSIVE,
                WINDOW_START,
            )

            first = BoundaryScheduler.next_boundary(WINDOW_START)
            last = WINDOW_START + timedelta(minutes=15 * (args.count - 1))
            print(json.dumps({
                "mode": "synthetic_simulation",
                "scheduled_boundaries": args.count,
                "first_boundary_utc": first.isoformat().replace("+00:00", "Z"),
                "last_boundary_utc": last.isoformat().replace("+00:00", "Z"),
                "registered_window_end_exclusive_utc": WINDOW_END_EXCLUSIVE.isoformat().replace("+00:00", "Z"),
                "official_records_written": False,
                "official_models_loaded": False,
            }, sort_keys=True))
            return 0

        config = _read_config(args.config)
        config.validate()
        if args.mode == "validate":
            # This verifies the actual official loader gate but does not infer or persist.
            load_official_models(config.official_model_directory)
            print(json.dumps({
                "mode": "configuration_validation",
                "configuration": "valid",
                "official_model_manifest": "verified",
                "official_inference_started": False,
                "ledger_opened": False,
            }, sort_keys=True))
            return 0

        if args.mode in {"dry-run", "prepare"}:
            boundary = _utc(args.boundary)
            journal = ProspectiveRawFrameJournal.open_read_only(config.source_journal_path)
            try:
                if args.mode == "prepare":
                    runner = ProspectiveInferenceRunner(config, ledger=None, journal=journal)
                    _emit_diagnostic(runner.prepare_checkpoint(boundary))
                    return 0
                session_id = config.expected_source_session_id
                mode = journal.session_mode(session_id)
                started = journal.session_started_at(session_id)
                exists = journal.has_fence_record(session_id, boundary)
                details = {
                    "mode": "read_only_diagnostic",
                    "source_session_id": session_id,
                    "source_mode": mode,
                    "source_session_registered": started is not None,
                    "exact_boundary_fence_present": exists,
                    "official_records_written": False,
                    "models_loaded": False,
                    "ledger_opened": False,
                }
                print(json.dumps(details, sort_keys=True))
                return 0 if started is not None and mode == "bridge" else 2
            finally:
                journal.close()

        if args.authorization_id != config.operational_authorization_id:
            raise ValueError("Authorization ID does not match the reviewed runner configuration")
        if not args.confirm_authorized_deployment:
            raise ValueError("Official mode requires --confirm-authorized-deployment")
        if os.environ.get("BTC_QH_PROSPECTIVE_AUTHORIZATION_ID") != config.operational_authorization_id:
            raise ValueError("Independent deployment authorization is absent or does not match")
        if os.environ.get("BTC_QH_IMPLEMENTATION_GIT_SHA") != config.implementation_git_sha:
            raise ValueError("Running implementation SHA is absent or does not match")
        ledger = ProspectiveLedger(config.prediction_ledger_path)
        runner: ProspectiveInferenceRunner | None = None
        try:
            runner = ProspectiveInferenceRunner.from_config(config, ledger=ledger)
            runner.run_forever(diagnostic_sink=_emit_diagnostic)
        finally:
            if runner is not None:
                runner.close()
            ledger.close()
        return 0
    except Exception as exc:
        print(f"prospective runner {args.mode} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
