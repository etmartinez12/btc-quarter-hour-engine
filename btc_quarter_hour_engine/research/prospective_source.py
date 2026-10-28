from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from btc_quarter_hour_engine.acquisition.coinbase_websocket import (
    parse_coinbase_ws_message,
)
from btc_quarter_hour_engine.research.features import (
    _BucketStats,
    _IngestFeatureReplay,
    _feature_buckets_frame,
    _window_features,
)
from btc_quarter_hour_engine.research.features_v2 import (
    V2_PREDICTOR_COLUMNS,
    V2_PREDICTOR_CONTRACT_VERSION,
    build_v2_research_feature_frame,
)
from btc_quarter_hour_engine.research.feature_families_v2 import FAMILY_CONTRACT_VERSION
from btc_quarter_hour_engine.research.model_freeze import frozen_predictors


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


def _sha256(value: str, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA256 digest")
    return value


def compute_feature_fingerprint(
    values: dict[str, float],
    *,
    candidate_predictors: tuple[str, ...],
    control_predictors: tuple[str, ...],
    feature_contract: str,
    feature_family_contract: str,
) -> str:
    payload = {
        "contract": feature_contract,
        "family_contract": feature_family_contract,
        "candidate_predictors": list(candidate_predictors),
        "control_predictors": list(control_predictors),
        "values": [[name, float(values[name])] for name in control_predictors],
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class RawFrameRecord:
    raw_bytes: bytes
    session_id: str
    connection_id: str
    frame_index: int
    ingest_time_utc: datetime
    raw_frame_sha256: str


@dataclass(frozen=True, slots=True)
class RawFrameSnapshot:
    """A complete, immutable source snapshot through its declared watermark."""

    session_id: str
    session_started_at_utc: datetime
    watermark_utc: datetime
    clock_health_verified: bool
    frames: tuple[RawFrameRecord, ...]


@dataclass(frozen=True, slots=True)
class CausalFeatureVector:
    boundary_utc: datetime
    target_utc: datetime
    source_session_id: str
    source_watermark_utc: datetime
    source_provenance_sha256: str
    source_frame_count: int
    excluded_ingest_future_count: int
    excluded_event_future_count: int
    feature_contract: str
    feature_family_contract: str
    candidate_predictors: tuple[str, ...]
    control_predictors: tuple[str, ...]
    values: dict[str, float]
    feature_fingerprint: str
    data_ready: bool
    feature_ready: bool
    clock_health_verified: bool


class CausalInputError(ValueError):
    def __init__(
        self,
        reason: str,
        *,
        data_ready: bool = False,
        source_session_id: str | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.data_ready = data_ready
        self.feature_ready = False
        self.source_session_id = source_session_id


class ProspectiveRawFrameJournal:
    """Isolated durable append-only feed adapter; not connected to the collector."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self.path, timeout=30, isolation_level=None
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 30000")
        self._connection.execute("PRAGMA synchronous = FULL")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS source_sessions (
                session_id TEXT PRIMARY KEY,
                session_started_at_utc TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS source_frames (
                session_id TEXT NOT NULL REFERENCES source_sessions(session_id),
                frame_index INTEGER NOT NULL CHECK (frame_index >= 0),
                connection_id TEXT NOT NULL,
                ingest_time_utc TEXT NOT NULL,
                raw_frame_sha256 TEXT NOT NULL,
                raw_bytes BLOB NOT NULL,
                PRIMARY KEY (session_id, frame_index)
            );
            CREATE TABLE IF NOT EXISTS source_watermarks (
                watermark_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES source_sessions(session_id),
                watermark_utc TEXT NOT NULL,
                clock_health_verified INTEGER NOT NULL CHECK (clock_health_verified IN (0, 1)),
                UNIQUE (session_id, watermark_utc)
            );
            CREATE INDEX IF NOT EXISTS source_frames_ingest_idx
                ON source_frames(session_id, ingest_time_utc, frame_index);
            """
        )
        for table in ("source_sessions", "source_frames", "source_watermarks"):
            for operation in ("UPDATE", "DELETE"):
                self._connection.execute(
                    f"""
                    CREATE TRIGGER IF NOT EXISTS {table}_immutable_{operation.lower()}
                    BEFORE {operation} ON {table}
                    BEGIN
                        SELECT RAISE(ABORT, 'append-only source journal');
                    END
                    """
                )

    @contextmanager
    def _transaction(self):
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

    @contextmanager
    def _read_transaction(self):
        self._connection.execute("BEGIN")
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

    def close(self) -> None:
        self._connection.close()

    def start_session(self, session_id: str, started_at_utc: datetime) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Source session ID must be nonempty")
        started = _utc(started_at_utc, field="source session start")
        with self._transaction():
            self._connection.execute(
                "INSERT INTO source_sessions(session_id, session_started_at_utc) "
                "VALUES (?, ?)",
                (session_id, _iso(started)),
            )

    def append_frame(
        self,
        *,
        session_id: str,
        connection_id: str,
        frame_index: int,
        ingest_time_utc: datetime,
        raw_bytes: bytes,
    ) -> str:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Source session ID must be nonempty")
        if not isinstance(connection_id, str) or not connection_id:
            raise ValueError("Connection ID must be nonempty")
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError("Frame index must be a nonnegative integer")
        if not isinstance(raw_bytes, bytes):
            raise TypeError("Raw frame must be immutable bytes")
        ingest = _utc(ingest_time_utc, field="frame ingest time")
        digest = hashlib.sha256(raw_bytes).hexdigest()
        with self._transaction():
            session = self._connection.execute(
                "SELECT session_started_at_utc FROM source_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise ValueError("Source session has not been registered")
            previous = self._connection.execute(
                "SELECT frame_index, ingest_time_utc FROM source_frames "
                "WHERE session_id = ? ORDER BY frame_index DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            expected_index = 0 if previous is None else previous["frame_index"] + 1
            if frame_index != expected_index:
                raise ValueError("Source frame index is not the next immutable identity")
            if previous is not None and ingest < _parse_utc(previous["ingest_time_utc"]):
                raise ValueError("Source ingest timestamp regressed")
            if ingest < _parse_utc(session["session_started_at_utc"]):
                raise ValueError("Source frame precedes its registered session start")
            latest_watermark = self._connection.execute(
                "SELECT MAX(watermark_utc) AS latest FROM source_watermarks "
                "WHERE session_id = ?",
                (session_id,),
            ).fetchone()["latest"]
            if latest_watermark is not None and ingest <= _parse_utc(latest_watermark):
                raise ValueError(
                    "Source frame ingest time is at or before the committed completeness watermark"
                )
            self._connection.execute(
                """
                INSERT INTO source_frames (
                    session_id, frame_index, connection_id, ingest_time_utc,
                    raw_frame_sha256, raw_bytes
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, frame_index, connection_id, _iso(ingest), digest, raw_bytes),
            )
        return digest

    def advance_watermark(
        self,
        session_id: str,
        watermark_utc: datetime,
        *,
        clock_health_verified: bool,
    ) -> None:
        if not isinstance(clock_health_verified, bool):
            raise ValueError("Clock-health status must be an explicit boolean")
        watermark = _utc(watermark_utc, field="source watermark")
        with self._transaction():
            session = self._connection.execute(
                "SELECT session_started_at_utc FROM source_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise ValueError("Source session has not been registered")
            if watermark < _parse_utc(session["session_started_at_utc"]):
                raise ValueError("Source watermark precedes the session start")
            latest_frame = self._connection.execute(
                "SELECT MAX(ingest_time_utc) AS latest FROM source_frames "
                "WHERE session_id = ?",
                (session_id,),
            ).fetchone()["latest"]
            if latest_frame is not None and watermark < _parse_utc(latest_frame):
                raise ValueError("Source watermark precedes an already appended frame")
            latest_watermark = self._connection.execute(
                "SELECT MAX(watermark_utc) AS latest FROM source_watermarks "
                "WHERE session_id = ?",
                (session_id,),
            ).fetchone()["latest"]
            if latest_watermark is not None:
                previous_watermark = _parse_utc(latest_watermark)
                if watermark == previous_watermark:
                    raise ValueError("Source watermark is repeated")
                if watermark < previous_watermark:
                    raise ValueError("Source watermark regressed")
            self._connection.execute(
                """
                INSERT INTO source_watermarks (
                    session_id, watermark_utc, clock_health_verified
                ) VALUES (?, ?, ?)
                """,
                (session_id, _iso(watermark), int(clock_health_verified)),
            )

    def snapshot(self, session_id: str) -> RawFrameSnapshot:
        with self._read_transaction():
            session = self._connection.execute(
                "SELECT session_started_at_utc FROM source_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise ValueError("Source session has not been registered")
            watermark = self._connection.execute(
                "SELECT watermark_utc, clock_health_verified FROM source_watermarks "
                "WHERE session_id = ? ORDER BY watermark_id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            if watermark is None:
                raise ValueError("Source session has no completeness watermark")
            frames = tuple(
                RawFrameRecord(
                    raw_bytes=bytes(row["raw_bytes"]),
                    session_id=row["session_id"],
                    connection_id=row["connection_id"],
                    frame_index=row["frame_index"],
                    ingest_time_utc=_parse_utc(row["ingest_time_utc"]),
                    raw_frame_sha256=row["raw_frame_sha256"],
                )
                for row in self._connection.execute(
                    "SELECT * FROM source_frames WHERE session_id = ? ORDER BY frame_index",
                    (session_id,),
                ).fetchall()
            )
            return RawFrameSnapshot(
                session_id=session_id,
                session_started_at_utc=_parse_utc(session["session_started_at_utc"]),
                watermark_utc=_parse_utc(watermark["watermark_utc"]),
                clock_health_verified=bool(watermark["clock_health_verified"]),
                frames=frames,
            )


def _frame_event_times(event: dict[str, Any]) -> tuple[datetime, ...]:
    envelope = event.get("envelope")
    if envelope is None:
        raise ValueError("Raw Coinbase frame has no parsed envelope")

    times = [envelope.timestamp_utc]
    if event.get("type") == "heartbeat":
        times.append(event["time_utc"])
    elif event.get("type") in {"snapshot", "l2_data"}:
        times.append(event["event_time_utc"])
        times.extend(
            update["event_time_utc"]
            for update in event["updates"]
            if update.get("event_time_utc") is not None
        )
    return tuple(_utc(timestamp, field="exchange event time") for timestamp in times)


def build_causal_feature_vector(
    snapshot: RawFrameSnapshot,
    boundary_utc: datetime,
) -> CausalFeatureVector:
    """Replay only verified raw frames available and exchange-timestamped by T.

    The snapshot contract deliberately carries no canonical target, label, or
    target-eligibility fields.
    """
    boundary = _utc(boundary_utc, field="prediction boundary")
    if not _is_boundary(boundary):
        raise CausalInputError(
            "prediction_boundary_not_exact_utc_quarter_hour",
            source_session_id=snapshot.session_id,
        )
    if not isinstance(snapshot.session_id, str) or not snapshot.session_id:
        raise CausalInputError("missing_source_session_id")
    if snapshot.clock_health_verified is not True:
        raise CausalInputError(
            "source_clock_health_unverified",
            source_session_id=snapshot.session_id,
        )

    try:
        session_started = _utc(
            snapshot.session_started_at_utc, field="session start"
        )
        watermark = _utc(snapshot.watermark_utc, field="source watermark")
    except ValueError as exc:
        raise CausalInputError(
            "invalid_source_timestamp",
            source_session_id=snapshot.session_id,
        ) from exc
    if watermark < boundary:
        raise CausalInputError(
            "source_watermark_before_boundary",
            source_session_id=snapshot.session_id,
        )
    if session_started > boundary - timedelta(minutes=240):
        raise CausalInputError(
            "insufficient_240_minute_session_history",
            source_session_id=snapshot.session_id,
        )
    if not isinstance(snapshot.frames, tuple):
        raise CausalInputError(
            "raw_frame_snapshot_must_be_immutable_tuple",
            source_session_id=snapshot.session_id,
        )

    replay = _IngestFeatureReplay("BTC-USD")
    included: list[dict[str, Any]] = []
    expected_index = 0
    previous_ingest: datetime | None = None
    excluded_ingest_future_count = 0
    excluded_event_future_count = 0

    for frame in snapshot.frames:
        if frame.session_id != snapshot.session_id:
            raise CausalInputError(
                "raw_frame_session_id_mismatch",
                source_session_id=snapshot.session_id,
            )
        try:
            ingest_time = _utc(frame.ingest_time_utc, field="frame ingest time")
        except ValueError as exc:
            raise CausalInputError(
                "invalid_frame_ingest_timestamp",
                source_session_id=snapshot.session_id,
            ) from exc
        if ingest_time > watermark:
            raise CausalInputError(
                "frame_ingest_time_exceeds_source_watermark",
                source_session_id=snapshot.session_id,
            )
        if ingest_time > boundary:
            excluded_ingest_future_count += 1
            continue
        if (
            type(frame.frame_index) is not int
            or frame.frame_index != expected_index
        ):
            raise CausalInputError(
                "noncontiguous_raw_frame_index",
                data_ready=False,
                source_session_id=snapshot.session_id,
            )
        if not isinstance(frame.connection_id, str) or not frame.connection_id:
            raise CausalInputError(
                "raw_frame_missing_connection_id",
                source_session_id=snapshot.session_id,
            )
        if previous_ingest is not None and ingest_time < previous_ingest:
            raise CausalInputError(
                "raw_frame_ingest_order_regressed",
                source_session_id=snapshot.session_id,
            )
        previous_ingest = ingest_time
        expected_index += 1
        if not isinstance(frame.raw_bytes, bytes):
            raise CausalInputError(
                "raw_frame_bytes_missing",
                source_session_id=snapshot.session_id,
            )
        try:
            frame_hash = _sha256(frame.raw_frame_sha256, field="raw frame hash")
        except ValueError as exc:
            raise CausalInputError(
                "invalid_raw_frame_hash",
                source_session_id=snapshot.session_id,
            ) from exc
        if hashlib.sha256(frame.raw_bytes).hexdigest() != frame_hash:
            raise CausalInputError(
                "raw_frame_hash_mismatch",
                source_session_id=snapshot.session_id,
            )
        try:
            event = parse_coinbase_ws_message(frame.raw_bytes)
            event_times = _frame_event_times(event)
        except (KeyError, TypeError, UnicodeError, ValueError) as exc:
            raise CausalInputError(
                "malformed_raw_frame_at_or_before_boundary",
                source_session_id=snapshot.session_id,
            ) from exc
        if any(event_time > boundary for event_time in event_times):
            excluded_event_future_count += 1
            continue

        replay.consume(
            raw=frame.raw_bytes,
            connection_id=frame.connection_id,
            frame_index=frame.frame_index,
            ingest_time_utc=ingest_time,
        )
        included.append({
            "connection_id": frame.connection_id,
            "frame_index": frame.frame_index,
            "ingest_time_utc": _iso(ingest_time),
            "raw_frame_sha256": frame_hash,
        })

    if not included:
        raise CausalInputError(
            "no_causally_eligible_raw_frames",
            source_session_id=snapshot.session_id,
        )
    if replay.malformed_frames or replay.sequence_gaps or replay.stale_frames:
        raise CausalInputError(
            "raw_frame_replay_integrity_failure",
            source_session_id=snapshot.session_id,
        )

    buckets = _feature_buckets_frame(
        replay, latest_boundary=pd.Timestamp(boundary)
    )
    if buckets.empty or pd.Timestamp(boundary) not in buckets.index:
        raise CausalInputError(
            "missing_asof_feature_bucket",
            data_ready=False,
            source_session_id=snapshot.session_id,
        )

    asof = buckets.loc[pd.Timestamp(boundary)]
    current_midpoint = float(asof["asof_midpoint"])
    data_ready = bool(asof["asof_book_synced"]) and np.isfinite(current_midpoint)
    if not data_ready:
        raise CausalInputError(
            "order_book_not_synchronized_at_boundary",
            source_session_id=snapshot.session_id,
        )

    row: dict[str, Any] = {
        "timestamp": pd.Timestamp(boundary),
        "feature_ready": False,
        "target_eligible": False,
        "trainable": False,
        "target": np.nan,
        "asof_spread_bps": float(asof["asof_spread_bps"]),
        "asof_top_book_imbalance": float(asof["asof_top_book_imbalance"]),
    }
    for minutes in (15, 30, 60, 240):
        lag = pd.Timestamp(boundary) - pd.Timedelta(minutes=minutes)
        lag_midpoint = (
            float(buckets.at[lag, "asof_midpoint"])
            if lag in buckets.index and pd.notna(buckets.at[lag, "asof_midpoint"])
            else np.nan
        )
        row[f"momentum_log_return_{minutes}m"] = (
            float(np.log(current_midpoint / lag_midpoint))
            if np.isfinite(lag_midpoint) and lag_midpoint > 0
            else np.nan
        )
        row.update(
            _window_features(
                buckets,
                timestamp=pd.Timestamp(boundary),
                window_minutes=minutes,
                current_midpoint=current_midpoint,
            )
        )

    row["feature_ready"] = bool(
        pd.Timestamp(boundary) >= pd.Timestamp(session_started)
        + pd.Timedelta(minutes=240)
        and asof["asof_book_synced"]
        and row["bbo_bucket_coverage_240m"] == 16
        and np.isfinite(row["momentum_log_return_240m"])
    )
    v2_frame = build_v2_research_feature_frame(pd.DataFrame([row]))
    v2_values = v2_frame.loc[:, list(V2_PREDICTOR_COLUMNS)].iloc[0]
    if not row["feature_ready"]:
        raise CausalInputError(
            "v2_feature_readiness_contract_not_satisfied",
            data_ready=True,
            source_session_id=snapshot.session_id,
        )
    if v2_values.isna().any() or not np.isfinite(v2_values.to_numpy(dtype=float)).all():
        missing = [
            name for name, value in v2_values.items()
            if pd.isna(value) or not np.isfinite(float(value))
        ]
        raise CausalInputError(
            "missing_or_nonfinite_v2_predictors:" + ",".join(missing),
            data_ready=True,
            source_session_id=snapshot.session_id,
        )

    predictors = frozen_predictors()
    if len(predictors["range_volatility"]) != 12 or len(predictors["full_v2"]) != 91:
        raise CausalInputError(
            "frozen_predictor_count_mismatch",
            data_ready=True,
            source_session_id=snapshot.session_id,
        )
    values = {
        name: float(v2_values[name])
        for name in predictors["full_v2"]
    }
    provenance_payload = {
        "session_id": snapshot.session_id,
        "session_started_at_utc": _iso(session_started),
        "boundary_utc": _iso(boundary),
        "frames": included,
    }
    provenance_sha256 = hashlib.sha256(
        json.dumps(
            provenance_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    fingerprint = compute_feature_fingerprint(
        values,
        candidate_predictors=predictors["range_volatility"],
        control_predictors=predictors["full_v2"],
        feature_contract=V2_PREDICTOR_CONTRACT_VERSION,
        feature_family_contract=FAMILY_CONTRACT_VERSION,
    )
    return CausalFeatureVector(
        boundary_utc=boundary,
        target_utc=boundary + timedelta(minutes=15),
        source_session_id=snapshot.session_id,
        source_watermark_utc=watermark,
        source_provenance_sha256=provenance_sha256,
        source_frame_count=len(included),
        excluded_ingest_future_count=excluded_ingest_future_count,
        excluded_event_future_count=excluded_event_future_count,
        feature_contract=V2_PREDICTOR_CONTRACT_VERSION,
        feature_family_contract=FAMILY_CONTRACT_VERSION,
        candidate_predictors=predictors["range_volatility"],
        control_predictors=predictors["full_v2"],
        values=values,
        feature_fingerprint=fingerprint,
        data_ready=True,
        feature_ready=True,
        clock_health_verified=snapshot.clock_health_verified,
    )
