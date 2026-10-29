from __future__ import annotations

import json
import math
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from btc_quarter_hour_engine.research.feature_families_v2 import (
    FAMILY_CONTRACT_VERSION,
    localization_variants,
)
from btc_quarter_hour_engine.research.prospective_contract import (
    FULL_V2_MODEL_SHA256,
    RANGE_MODEL_SHA256,
    TRUSTED_MANIFEST_SHA256,
)
from btc_quarter_hour_engine.research.prospective_source import (
    compute_feature_fingerprint,
)
WINDOW_START = datetime(2026, 11, 15, tzinfo=timezone.utc)
WINDOW_END_EXCLUSIVE = datetime(2026, 12, 15, tzinfo=timezone.utc)
PREDICTION_DEADLINE_SECONDS = 60
_EVENT_TYPES = {
    "prediction_recorded",
    "durability_attested",
    "durability_late",
    "missing_data",
    "inference_failed",
    "late_attempt",
    "missed",
    "stale_attempt",
    "duplicate_attempt",
    "invalid_attempt",
    "clock_skew",
    "label_attached",
    "duplicate_label",
    "dry_run",
}


def _utc(value: datetime, *, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return _utc(value, field="timestamp").isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _is_boundary(value: datetime) -> bool:
    return value.minute % 15 == 0 and value.second == 0 and value.microsecond == 0


def _require_digest(value: Any, *, length: int, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase hexadecimal digest")
    return value


class ClockHealthError(RuntimeError):
    pass


class ClockHealth:
    """Detect wall-clock regressions and wall/monotonic clock jumps."""

    def __init__(
        self,
        *,
        max_drift_seconds: float = 1.0,
        wall_clock: Callable[[], datetime] | None = None,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> None:
        if not math.isfinite(max_drift_seconds) or max_drift_seconds < 0:
            raise ValueError("max_drift_seconds must be finite and nonnegative")
        self.max_drift_seconds = float(max_drift_seconds)
        self._wall_clock = wall_clock or (lambda: datetime.now(timezone.utc))
        self._monotonic_clock = monotonic_clock or time.monotonic
        self._previous_wall: datetime | None = None
        self._previous_monotonic: float | None = None
        self._observed_wall: datetime | None = None

    @property
    def observed_wall_utc(self) -> datetime | None:
        return self._observed_wall

    def now(self) -> datetime:
        return self.check(self._wall_clock(), self._monotonic_clock())

    def check(
        self,
        wall_utc: datetime,
        monotonic_seconds: float | None = None,
    ) -> datetime:
        wall = _utc(wall_utc, field="clock wall time")
        monotonic = time.monotonic() if monotonic_seconds is None else monotonic_seconds
        self._observed_wall = wall
        if not isinstance(monotonic, (int, float)) or not math.isfinite(monotonic):
            raise ClockHealthError("monotonic_clock_invalid")
        monotonic = float(monotonic)
        if self._previous_wall is not None and self._previous_monotonic is not None:
            wall_delta = (wall - self._previous_wall).total_seconds()
            monotonic_delta = monotonic - self._previous_monotonic
            if (
                wall_delta < 0
                or monotonic_delta < 0
                or abs(wall_delta - monotonic_delta) > self.max_drift_seconds
            ):
                raise ClockHealthError("wall_monotonic_clock_skew")
        self._previous_wall = wall
        self._previous_monotonic = monotonic
        return wall


class DuplicatePredictionError(ValueError):
    pass


class LatePredictionError(ValueError):
    pass


class DuplicateLabelError(ValueError):
    pass


class ProspectiveLedger:
    """SQLite-backed immutable schedule, prediction, event, and label ledger."""

    def __init__(
        self,
        path: str | Path,
        *,
        window_start: datetime = WINDOW_START,
        window_end_exclusive: datetime = WINDOW_END_EXCLUSIVE,
        clock_health: ClockHealth | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.window_start = _utc(window_start, field="window start")
        self.window_end_exclusive = _utc(
            window_end_exclusive, field="window end"
        )
        if (
            not _is_boundary(self.window_start)
            or not _is_boundary(self.window_end_exclusive)
            or self.window_end_exclusive <= self.window_start
            or (self.window_end_exclusive - self.window_start).total_seconds() % 900
        ):
            raise ValueError("Ledger schedule must be a positive UTC quarter-hour interval")
        self.clock_health = clock_health or ClockHealth()
        self._connection = sqlite3.connect(
            self.path,
            timeout=30,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        schema_version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
        if schema_version not in (0, 2):
            self._connection.close()
            raise ValueError(
                "Existing ledger lacks post-commit durability attestations; "
                "preserve it read-only and initialize a new ledger"
            )
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 30000")
        self._connection.execute("PRAGMA synchronous = FULL")
        self._connection.execute("PRAGMA journal_mode = WAL")
        try:
            self._initialize()
        except BaseException:
            self._connection.close()
            raise

    @property
    def scheduled_count(self) -> int:
        return int(
            (self.window_end_exclusive - self.window_start).total_seconds() // 900
        )

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise
        else:
            try:
                self._connection.execute("COMMIT")
            except BaseException:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise

    def _initialize(self) -> None:
        schema_version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
        if schema_version not in (0, 2):
            raise ValueError(
                "Existing ledger lacks post-commit durability attestations; "
                "preserve it read-only and initialize a new ledger"
            )
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS scheduled_boundaries (
                boundary_utc TEXT PRIMARY KEY,
                target_utc TEXT NOT NULL UNIQUE,
                CHECK (length(boundary_utc) = 27),
                CHECK (length(target_utc) = 27)
            );
            CREATE TABLE IF NOT EXISTS attempt_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                boundary_utc TEXT NOT NULL,
                event_type TEXT NOT NULL,
                recorded_at_utc TEXT NOT NULL,
                reason TEXT NOT NULL,
                details_json TEXT NOT NULL,
                CHECK (event_type IN (
                    'prediction_recorded', 'missing_data', 'inference_failed',
                    'durability_attested', 'durability_late',
                    'late_attempt', 'missed', 'stale_attempt',
                    'duplicate_attempt', 'invalid_attempt', 'clock_skew',
                    'label_attached', 'duplicate_label', 'dry_run'
                ))
            );
            CREATE TABLE IF NOT EXISTS predictions (
                boundary_utc TEXT PRIMARY KEY
                    REFERENCES scheduled_boundaries(boundary_utc),
                target_utc TEXT NOT NULL,
                data_ready INTEGER NOT NULL CHECK (data_ready = 1),
                feature_ready INTEGER NOT NULL CHECK (feature_ready = 1),
                source_session_id TEXT NOT NULL,
                source_watermark_utc TEXT NOT NULL,
                source_provenance_sha256 TEXT NOT NULL
                    CHECK (length(source_provenance_sha256) = 64),
                source_frame_count INTEGER NOT NULL CHECK (source_frame_count > 0),
                excluded_ingest_future_count INTEGER NOT NULL
                    CHECK (excluded_ingest_future_count >= 0),
                excluded_event_future_count INTEGER NOT NULL
                    CHECK (excluded_event_future_count >= 0),
                clock_health_verified INTEGER NOT NULL CHECK (clock_health_verified = 1),
                feature_contract TEXT NOT NULL,
                feature_family_contract TEXT NOT NULL,
                feature_fingerprint TEXT NOT NULL
                    CHECK (length(feature_fingerprint) = 64),
                candidate_predictors_json TEXT NOT NULL,
                control_predictors_json TEXT NOT NULL,
                feature_vector_json TEXT NOT NULL,
                manifest_sha256 TEXT NOT NULL CHECK (length(manifest_sha256) = 64),
                candidate_model_sha256 TEXT NOT NULL
                    CHECK (length(candidate_model_sha256) = 64),
                control_model_sha256 TEXT NOT NULL
                    CHECK (length(control_model_sha256) = 64),
                implementation_git_sha TEXT NOT NULL
                    CHECK (length(implementation_git_sha) = 40),
                generated_at_utc TEXT NOT NULL,
                write_started_at_utc TEXT NOT NULL,
                candidate_probability REAL NOT NULL
                    CHECK (candidate_probability >= 0 AND candidate_probability <= 1),
                candidate_prediction INTEGER NOT NULL
                    CHECK (candidate_prediction IN (0, 1)),
                control_probability REAL NOT NULL
                    CHECK (control_probability >= 0 AND control_probability <= 1),
                control_prediction INTEGER NOT NULL
                    CHECK (control_prediction IN (0, 1)),
                baseline_prediction INTEGER NOT NULL CHECK (baseline_prediction = 0),
                failure_reasons_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS durability_attestations (
                boundary_utc TEXT PRIMARY KEY
                    REFERENCES predictions(boundary_utc),
                attested_at_utc TEXT NOT NULL,
                deadline_met INTEGER NOT NULL CHECK (deadline_met IN (0, 1)),
                clock_health_verified INTEGER NOT NULL CHECK (clock_health_verified = 1)
            );
            CREATE TABLE IF NOT EXISTS labels (
                boundary_utc TEXT PRIMARY KEY
                    REFERENCES scheduled_boundaries(boundary_utc),
                target_utc TEXT NOT NULL,
                target INTEGER NOT NULL CHECK (target IN (0, 1)),
                available_at_utc TEXT NOT NULL,
                target_eligible INTEGER NOT NULL CHECK (target_eligible IN (0, 1)),
                source_provenance_sha256 TEXT NOT NULL
                    CHECK (length(source_provenance_sha256) = 64),
                evidence_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS attempt_events_boundary_idx
                ON attempt_events(boundary_utc, event_id);
            """
        )
        for table in (
            "scheduled_boundaries",
            "attempt_events",
            "predictions",
            "durability_attestations",
            "labels",
        ):
            for operation in ("UPDATE", "DELETE"):
                trigger = f"{table}_immutable_{operation.lower()}"
                self._connection.execute(
                    f"""
                    CREATE TRIGGER IF NOT EXISTS {trigger}
                    BEFORE {operation} ON {table}
                    BEGIN
                        SELECT RAISE(ABORT, 'append-only ledger table');
                    END
                    """
                )

        with self._transaction():
            rows = self._connection.execute(
                "SELECT boundary_utc, target_utc FROM scheduled_boundaries "
                "ORDER BY boundary_utc"
            ).fetchall()
            expected: list[tuple[str, str]] = []
            boundary = self.window_start
            while boundary < self.window_end_exclusive:
                target = boundary + timedelta(minutes=15)
                expected.append((_iso(boundary), _iso(target)))
                boundary += timedelta(minutes=15)
            if rows:
                actual = [(row["boundary_utc"], row["target_utc"]) for row in rows]
                if actual != expected:
                    raise ValueError("Existing ledger schedule differs from the registered window")
            else:
                self._connection.executemany(
                    "INSERT INTO scheduled_boundaries (boundary_utc, target_utc) "
                    "VALUES (?, ?)",
                    expected,
                )
        self._connection.execute("PRAGMA user_version = 2")

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> ProspectiveLedger:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def schedule(self) -> list[dict[str, str]]:
        return [
            dict(row)
            for row in self._connection.execute(
                "SELECT boundary_utc, target_utc FROM scheduled_boundaries "
                "ORDER BY boundary_utc"
            ).fetchall()
        ]

    def _scheduled_target(self, boundary: datetime) -> datetime:
        timestamp = _iso(boundary)
        row = self._connection.execute(
            "SELECT target_utc FROM scheduled_boundaries WHERE boundary_utc = ?",
            (timestamp,),
        ).fetchone()
        if row is None:
            raise ValueError("Boundary is outside the registered schedule")
        return _parse_utc(row["target_utc"])

    def _insert_event(
        self,
        boundary: datetime,
        event_type: str,
        recorded_at: datetime,
        reason: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if event_type not in _EVENT_TYPES:
            raise ValueError(f"Unsupported ledger event type: {event_type}")
        if not isinstance(reason, str) or not reason:
            raise ValueError("Ledger event reason must be nonempty")
        payload = json.dumps(
            dict(details or {}), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        self._connection.execute(
            "INSERT INTO attempt_events "
            "(boundary_utc, event_type, recorded_at_utc, reason, details_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (_iso(boundary), event_type, _iso(recorded_at), reason, payload),
        )

    def append_attempt(
        self,
        boundary_utc: datetime,
        *,
        event_type: str,
        recorded_at_utc: datetime,
        reason: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        boundary = _utc(boundary_utc, field="attempt boundary")
        recorded_at = _utc(recorded_at_utc, field="event timestamp")
        with self._transaction():
            self._insert_event(boundary, event_type, recorded_at, reason, details)

    def append_prediction(
        self,
        evidence: Mapping[str, Any],
        *,
        model_bundle: object | None = None,
    ) -> dict[str, Any]:
        return self._append_prediction_evidence(evidence, model_bundle=model_bundle)

    def _append_prediction_evidence(
        self,
        evidence: Mapping[str, Any],
        *,
        model_bundle: object | None,
    ) -> dict[str, Any]:
        from btc_quarter_hour_engine.research.prospective_inference import (
            _verify_recorded_model_outputs,
            is_verified_official_bundle,
        )

        if not is_verified_official_bundle(model_bundle):
            raise ValueError(
                "Prediction persistence requires a bundle loaded by load_official_models()"
            )
        _verify_recorded_model_outputs(dict(evidence), model_bundle)
        boundary = _utc(evidence["boundary_utc"], field="prediction boundary")
        target = self._scheduled_target(boundary)
        if not _is_boundary(boundary):
            raise ValueError("Prediction boundary is not an exact UTC quarter-hour")
        if evidence.get("target_utc") != target:
            raise ValueError("Prediction target must be exactly T+15 minutes")
        candidate_sha = _require_digest(
            evidence.get("candidate_model_sha256"), length=64, field="candidate model SHA256"
        )
        control_sha = _require_digest(
            evidence.get("control_model_sha256"), length=64, field="control model SHA256"
        )
        source_sha = _require_digest(
            evidence.get("source_provenance_sha256"), length=64, field="source provenance SHA256"
        )
        fingerprint = _require_digest(
            evidence.get("feature_fingerprint"), length=64, field="feature fingerprint"
        )
        implementation_sha = _require_digest(
            evidence.get("implementation_git_sha"), length=40, field="implementation Git SHA"
        )
        manifest_sha = _require_digest(
            evidence.get("manifest_sha256"), length=64, field="manifest SHA256"
        )
        if (
            candidate_sha != RANGE_MODEL_SHA256
            or control_sha != FULL_V2_MODEL_SHA256
            or manifest_sha != TRUSTED_MANIFEST_SHA256
        ):
            raise ValueError("Prediction model or manifest hash differs from the frozen protocol")
        if evidence.get("data_ready") is not True or evidence.get("feature_ready") is not True:
            raise ValueError("Only data-ready, feature-ready predictions may be persisted")
        if evidence.get("clock_health_verified") is not True:
            raise ValueError("Clock health must be verified before prediction persistence")
        if evidence.get("feature_contract") != "bbo-causal-v2":
            raise ValueError("Prediction feature contract does not match the frozen V2 contract")
        if evidence.get("feature_family_contract") != FAMILY_CONTRACT_VERSION:
            raise ValueError("Prediction feature-family contract differs from the frozen contract")
        if not isinstance(evidence.get("source_session_id"), str) or not evidence["source_session_id"]:
            raise ValueError("Prediction source session must be nonempty")
        watermark = _utc(evidence["source_watermark_utc"], field="source watermark")
        if watermark < boundary:
            raise ValueError("Source watermark precedes the prediction boundary")
        generated_at = _utc(evidence["generated_at_utc"], field="generation timestamp")
        candidate_probability = self._probability(
            evidence.get("candidate_probability"), field="candidate probability"
        )
        control_probability = self._probability(
            evidence.get("control_probability"), field="control probability"
        )
        candidate_prediction = evidence.get("candidate_prediction")
        control_prediction = evidence.get("control_prediction")
        if (
            type(candidate_prediction) is not int
            or candidate_prediction != int(candidate_probability >= 0.5)
        ):
            raise ValueError("Candidate prediction disagrees with the frozen 0.5 threshold")
        if (
            type(control_prediction) is not int
            or control_prediction != int(control_probability >= 0.5)
        ):
            raise ValueError("Control prediction disagrees with the frozen 0.5 threshold")
        if type(evidence.get("baseline_prediction")) is not int or evidence["baseline_prediction"] != 0:
            raise ValueError("Always-DOWN baseline prediction must be class 0")
        vector = evidence.get("feature_vector")
        if not isinstance(vector, Mapping) or len(vector) != 91:
            raise ValueError("Prediction evidence must contain exactly 91 ordered V2 values")
        variants = localization_variants()
        expected_candidate = tuple(variants["group_only", "range_volatility"])
        expected_control = tuple(variants["reference", "all"])
        if (
            tuple(vector) != expected_control
            or tuple(evidence.get("candidate_predictors", ())) != expected_candidate
            or tuple(evidence.get("control_predictors", ())) != expected_control
        ):
            raise ValueError("Prediction feature order differs from frozen predictor contracts")
        if type(evidence.get("source_frame_count")) is not int or evidence["source_frame_count"] < 1:
            raise ValueError("Prediction source frame count must be positive")
        if (
            type(evidence.get("excluded_ingest_future_count")) is not int
            or evidence["excluded_ingest_future_count"] < 0
            or type(evidence.get("excluded_event_future_count")) is not int
            or evidence["excluded_event_future_count"] < 0
        ):
            raise ValueError("Excluded future-frame counts must be nonnegative integers")
        normalized_vector = {
            name: self._finite_number(value, field=f"feature {name}")
            for name, value in vector.items()
        }
        vector_json = json.dumps(
            [[name, value] for name, value in normalized_vector.items()],
            separators=(",", ":"),
            allow_nan=False,
        )
        if fingerprint != compute_feature_fingerprint(
            normalized_vector,
            candidate_predictors=expected_candidate,
            control_predictors=expected_control,
            feature_contract=evidence["feature_contract"],
            feature_family_contract=evidence["feature_family_contract"],
        ):
            raise ValueError("Prediction feature vector does not match its fingerprint")
        failure_reasons = evidence.get("failure_reasons")
        if (
            not isinstance(failure_reasons, list)
            or any(not isinstance(reason, str) or not reason for reason in failure_reasons)
        ):
            raise ValueError("Prediction failure reasons must be an explicit string list")
        reasons_json = json.dumps(
            failure_reasons,
            separators=(",", ":"),
            allow_nan=False,
        )
        duplicate = False
        late = False
        clock_error: ClockHealthError | None = None
        invalid_timestamp = False
        write_started_at: datetime | None = None
        with self._transaction():
            try:
                write_started_at = self.clock_health.now()
            except ClockHealthError as exc:
                clock_error = exc
                observed = self.clock_health.observed_wall_utc
                if observed is not None:
                    self._insert_event(boundary, "clock_skew", observed, str(exc))
            if clock_error is None and generated_at > write_started_at:
                self._insert_event(
                    boundary,
                    "invalid_attempt",
                    write_started_at,
                    "generation_timestamp_after_persistence_clock",
                )
                invalid_timestamp = True
            elif clock_error is None and generated_at < boundary:
                self._insert_event(
                    boundary,
                    "invalid_attempt",
                    write_started_at,
                    "generation_timestamp_before_boundary",
                )
                invalid_timestamp = True
            elif clock_error is None and (
                write_started_at > boundary + timedelta(seconds=PREDICTION_DEADLINE_SECONDS)
                or write_started_at >= target
                or generated_at >= target
            ):
                self._insert_event(
                    boundary,
                    "late_attempt",
                    write_started_at,
                    "generation_or_persistence_after_deadline",
                )
                late = True
            elif clock_error is None and self._connection.execute(
                "SELECT 1 FROM predictions WHERE boundary_utc = ?", (_iso(boundary),)
            ).fetchone():
                self._insert_event(
                    boundary,
                    "duplicate_attempt",
                    write_started_at,
                    "prediction_already_recorded",
                )
                duplicate = True
            elif clock_error is None and self._connection.execute(
                "SELECT 1 FROM labels WHERE boundary_utc = ?", (_iso(boundary),)
            ).fetchone():
                self._insert_event(
                    boundary,
                    "stale_attempt",
                    write_started_at,
                    "label_already_attached",
                )
                duplicate = True
            elif clock_error is None:
                assert write_started_at is not None
                self._connection.execute(
                    """
                    INSERT INTO predictions (
                        boundary_utc, target_utc, data_ready, feature_ready,
                        source_session_id, source_watermark_utc,
                        source_provenance_sha256, source_frame_count,
                        excluded_ingest_future_count, excluded_event_future_count,
                        clock_health_verified, feature_contract,
                        feature_family_contract,
                        feature_fingerprint, candidate_predictors_json,
                        control_predictors_json, feature_vector_json,
                        manifest_sha256,
                        candidate_model_sha256, control_model_sha256,
                        implementation_git_sha, generated_at_utc,
                        write_started_at_utc, candidate_probability,
                        candidate_prediction, control_probability,
                        control_prediction, baseline_prediction,
                        failure_reasons_json
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        _iso(boundary),
                        _iso(target),
                        1,
                        1,
                        evidence["source_session_id"],
                        _iso(watermark),
                        source_sha,
                        evidence["source_frame_count"],
                        evidence["excluded_ingest_future_count"],
                        evidence["excluded_event_future_count"],
                        1,
                        evidence["feature_contract"],
                        evidence["feature_family_contract"],
                        fingerprint,
                        json.dumps(
                            list(expected_candidate), separators=(",", ":"), allow_nan=False
                        ),
                        json.dumps(
                            list(expected_control), separators=(",", ":"), allow_nan=False
                        ),
                        vector_json,
                        manifest_sha,
                        candidate_sha,
                        control_sha,
                        implementation_sha,
                        _iso(generated_at),
                        _iso(write_started_at),
                        candidate_probability,
                        candidate_prediction,
                        control_probability,
                        control_prediction,
                        evidence["baseline_prediction"],
                        reasons_json,
                    ),
                )
                self._insert_event(
                    boundary,
                    "prediction_recorded",
                    write_started_at,
                    "prediction_transaction_committed_pending_durability_attestation",
                    {
                        "feature_fingerprint": fingerprint,
                        "source_provenance_sha256": source_sha,
                    },
                )
        if clock_error is not None:
            raise clock_error
        if invalid_timestamp:
            raise ValueError("Prediction generation timestamp is invalid")
        if late:
            raise LatePredictionError("Prediction missed its T+60-second persistence deadline")
        if duplicate:
            raise DuplicatePredictionError(
                "A prediction or label already exists for this boundary"
            )
        assert write_started_at is not None
        attested_at = self._attest_durable_commit(boundary)
        deadline_met = (
            attested_at <= boundary + timedelta(seconds=PREDICTION_DEADLINE_SECONDS)
        )
        if not deadline_met:
            raise LatePredictionError(
                "Prediction transaction committed after its T+60-second deadline"
            )
        return {
            **dict(evidence),
            "boundary_utc": boundary,
            "target_utc": target,
            "generated_at_utc": generated_at,
            "write_started_at_utc": write_started_at,
            "persisted_at_utc": attested_at,
            "timing_eligible": deadline_met,
            "failure_reasons": [],
        }

    def _attest_durable_commit(self, boundary: datetime) -> datetime:
        try:
            attested_at = self.clock_health.now()
        except ClockHealthError as exc:
            observed = self.clock_health.observed_wall_utc
            if observed is not None:
                with self._transaction():
                    self._insert_event(boundary, "clock_skew", observed, str(exc))
            raise

        deadline_met = (
            attested_at <= boundary + timedelta(seconds=PREDICTION_DEADLINE_SECONDS)
        )
        with self._transaction():
            exists = self._connection.execute(
                "SELECT 1 FROM predictions WHERE boundary_utc = ?",
                (_iso(boundary),),
            ).fetchone()
            if exists is None:
                raise RuntimeError("Cannot attest a prediction transaction that is not durable")
            self._connection.execute(
                """
                INSERT INTO durability_attestations (
                    boundary_utc, attested_at_utc, deadline_met,
                    clock_health_verified
                ) VALUES (?, ?, ?, 1)
                """,
                (_iso(boundary), _iso(attested_at), int(deadline_met)),
            )
            self._insert_event(
                boundary,
                "durability_attested" if deadline_met else "durability_late",
                attested_at,
                (
                    "prediction_commit_confirmed_before_deadline"
                    if deadline_met
                    else "prediction_commit_confirmed_after_deadline"
                ),
            )
        return attested_at

    @staticmethod
    def _finite_number(value: Any, *, field: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{field} must be finite numeric data")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{field} must be finite numeric data") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"{field} must be finite numeric data")
        return numeric

    @classmethod
    def _probability(cls, value: Any, *, field: str) -> float:
        numeric = cls._finite_number(value, field=field)
        if not 0 <= numeric <= 1:
            raise ValueError(f"{field} must be between zero and one")
        return numeric

    def mark_missed_boundaries(self) -> int:
        try:
            now = self.clock_health.now()
        except ClockHealthError as exc:
            observed = self.clock_health.observed_wall_utc
            if observed is not None:
                with self._transaction():
                    self._insert_event(
                        self.window_start,
                        "clock_skew",
                        observed,
                        str(exc),
                    )
            raise
        inserted = 0
        with self._transaction():
            rows = self._connection.execute(
                """
                SELECT s.boundary_utc
                FROM scheduled_boundaries s
                LEFT JOIN predictions p ON p.boundary_utc = s.boundary_utc
                WHERE p.boundary_utc IS NULL
                ORDER BY s.boundary_utc
                """
            ).fetchall()
            for row in rows:
                boundary = _parse_utc(row["boundary_utc"])
                if now <= boundary + timedelta(seconds=PREDICTION_DEADLINE_SECONDS):
                    continue
                exists = self._connection.execute(
                    "SELECT 1 FROM attempt_events "
                    "WHERE boundary_utc = ? AND event_type = 'missed'",
                    (row["boundary_utc"],),
                ).fetchone()
                if exists is None:
                    self._insert_event(
                        boundary,
                        "missed",
                        now,
                        "no_prediction_persisted_by_deadline",
                    )
                    inserted += 1
        return inserted

    def attach_label(
        self,
        boundary_utc: datetime,
        *,
        target: int,
        target_utc: datetime,
        available_at_utc: datetime,
        target_eligible: bool,
        source_provenance_sha256: str,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        boundary = _utc(boundary_utc, field="label boundary")
        target_time = _utc(target_utc, field="label target timestamp")
        available_at = _utc(available_at_utc, field="label availability timestamp")
        try:
            recorded_at = self.clock_health.now()
        except ClockHealthError as exc:
            observed = self.clock_health.observed_wall_utc
            if observed is not None:
                with self._transaction():
                    self._insert_event(boundary, "clock_skew", observed, str(exc))
            raise
        expected_target = self._scheduled_target(boundary)
        if target_time != expected_target:
            raise ValueError("Label target must be exactly T+15 minutes")
        if target not in (0, 1) or isinstance(target, bool):
            raise ValueError("Target label must be binary class 0 or 1")
        if not isinstance(target_eligible, bool):
            raise ValueError("Target eligibility must be an explicit boolean")
        if available_at < target_time or available_at > recorded_at:
            raise ValueError("Label availability must be at or after T+15 and no later than persistence")
        digest = _require_digest(
            source_provenance_sha256, length=64, field="label source provenance SHA256"
        )
        details = json.dumps(
            dict(evidence or {}), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        duplicate = False
        with self._transaction():
            if self._connection.execute(
                "SELECT 1 FROM labels WHERE boundary_utc = ?", (_iso(boundary),)
            ).fetchone():
                self._insert_event(
                    boundary, "duplicate_label", recorded_at, "label_already_attached"
                )
                duplicate = True
            else:
                self._connection.execute(
                    """
                    INSERT INTO labels (
                        boundary_utc, target_utc, target, available_at_utc,
                        target_eligible, source_provenance_sha256, evidence_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _iso(boundary),
                        _iso(target_time),
                        target,
                        _iso(available_at),
                        int(target_eligible),
                        digest,
                        details,
                    ),
                )
                self._insert_event(
                    boundary,
                    "label_attached",
                    recorded_at,
                    "label_appended_as_separate_evidence",
                )
        if duplicate:
            raise DuplicateLabelError("A label already exists for this boundary")

    def lifecycle(self, boundary_utc: datetime) -> dict[str, Any]:
        boundary = _iso(_utc(boundary_utc, field="lifecycle boundary"))
        schedule = self._connection.execute(
            "SELECT boundary_utc, target_utc FROM scheduled_boundaries "
            "WHERE boundary_utc = ?",
            (boundary,),
        ).fetchone()
        if schedule is None:
            raise ValueError("Boundary is outside the registered schedule")
        prediction = self._connection.execute(
            "SELECT * FROM predictions WHERE boundary_utc = ?", (boundary,)
        ).fetchone()
        attestation = self._connection.execute(
            "SELECT * FROM durability_attestations WHERE boundary_utc = ?",
            (boundary,),
        ).fetchone()
        label = self._connection.execute(
            "SELECT * FROM labels WHERE boundary_utc = ?", (boundary,)
        ).fetchone()
        events = self._connection.execute(
            "SELECT * FROM attempt_events WHERE boundary_utc = ? ORDER BY event_id",
            (boundary,),
        ).fetchall()
        evaluation_eligible = bool(
            attestation is not None
            and attestation["deadline_met"] == 1
            and attestation["clock_health_verified"] == 1
            and _parse_utc(attestation["attested_at_utc"])
            <= _parse_utc(schedule["boundary_utc"])
            + timedelta(seconds=PREDICTION_DEADLINE_SECONDS)
        )
        if label is not None:
            status = "labeled"
        elif prediction is not None:
            status = (
                "predicted"
                if evaluation_eligible
                else "durability_late"
                if attestation is not None
                else "durability_unconfirmed"
            )
        elif events:
            status = events[-1]["event_type"]
        else:
            status = "pending"
        return {
            "schedule": dict(schedule),
            "prediction": dict(prediction) if prediction is not None else None,
            "durability_attestation": (
                dict(attestation) if attestation is not None else None
            ),
            "evaluation_eligible": evaluation_eligible,
            "label": dict(label) if label is not None else None,
            "events": [dict(row) for row in events],
            "status": status,
        }
