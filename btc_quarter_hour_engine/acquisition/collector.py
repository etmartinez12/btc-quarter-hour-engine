from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from btc_quarter_hour_engine.market_data.replay import replay_recorded_frames
from .coinbase_websocket import parse_coinbase_ws_message
from btc_quarter_hour_engine.storage.forward_manifest import build_forward_manifest, write_forward_manifest
from btc_quarter_hour_engine.storage.forward_parquet import ForwardParquetStore
from btc_quarter_hour_engine.storage.forward_schema import (
    DATA_KIND_BBO_STATE,
    DATA_KIND_LEVEL2_UPDATES,
    DATA_KIND_QUARTER_HOUR_BBO,
    FORWARD_BBO_SCHEMA_VERSION,
    FORWARD_BBO_STATE_SCHEMA_VERSION,
    FORWARD_LEVEL2_UPDATES_SCHEMA_VERSION,
    FORWARD_SOURCE,
)
from btc_quarter_hour_engine.storage.websocket_raw import RawSegmentWriter

from .websocket_service import CoinbaseWebSocketService


@dataclass(slots=True)
class CollectorResult:
    manifest: dict[str, Any]
    manifest_path: Path
    observation_count: int
    eligible_observation_count: int
    raw_segments: list[dict[str, Any]]
    # Union of every normalized artifact written this run (quarter-hour BBO,
    # level2 updates, and BBO book-state), in the order they were written.
    # This is what is embedded verbatim in the forward manifest.
    normalized_artifacts: list[dict[str, Any]]
    # Same artifacts, split out per `data_kind` for convenient direct access.
    bbo_normalized_artifacts: list[dict[str, Any]]
    level2_update_artifacts: list[dict[str, Any]]
    bbo_state_artifacts: list[dict[str, Any]]
    level2_update_row_count: int
    bbo_state_row_count: int
    connection_count: int
    reconnect_count: int
    termination_reason: str
    raw_frame_count: int
    level2_message_count: int
    heartbeat_message_count: int


def _observation_row(observation: Any) -> dict[str, Any]:
    return asdict(observation)


@dataclass(slots=True)
class WebSocketCollector:
    """Drive one :class:`CoinbaseWebSocketService` through a receive loop,
    sealing exact raw frame bytes into immutable segments, tracking
    reconnects, and persisting derived quarter-hour BBO observations plus a
    complete forward manifest on completion.

    In addition to the coarse quarter-hour BBO observations, every
    successfully applied level2 book mutation is persisted as a normalized
    ``level2_updates`` row (one row per individual price-level change, with
    its own per-update event time) and a ``bbo_state`` row (the resulting
    best-bid/ask snapshot), giving full forward-collection fidelity beyond
    just the boundary-eligible observations.

    Stop conditions (``max_messages`` / ``max_duration_seconds`` /
    ``stop_fn``) make the loop finite and therefore directly testable without
    any network dependency; a production CLI invocation simply omits them
    (or passes a signal-driven ``stop_fn``) to run indefinitely.
    """

    service: CoinbaseWebSocketService
    raw_segment_writer: RawSegmentWriter
    forward_store: ForwardParquetStore
    output_root: str
    source: str = FORWARD_SOURCE
    data_kind: str = DATA_KIND_QUARTER_HOUR_BBO
    sleep_fn: Any = field(default=None)
    connection_count: int = field(default=0, init=False)
    reconnect_count: int = field(default=0, init=False)
    _frame_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.sleep_fn is None:
            self.sleep_fn = time.sleep
        self.raw_segment_writer.bind_session(self.service.session_id)
        self.service.on_reconnect_frame = self._record_frame

    def _record_frame(self, frame: Any) -> int:
        connection_id = self.service.connection.connection_id if self.service.connection else None
        message_type = frame.message.get("type", "unknown") if frame.message is not None else "malformed"
        sequence_num = frame.message.get("sequence_num") if frame.message is not None else None
        frame_index = self._frame_count
        self.raw_segment_writer.add_frame(
            raw=frame.raw_bytes, message_type=message_type, connection_id=connection_id,
            sequence_num=sequence_num, ingest_time_utc=frame.received_at_utc,
        )
        self._frame_count += 1
        return frame_index

    def run(
        self,
        *,
        max_messages: int | None = None,
        max_duration_seconds: float | None = None,
        stop_fn: Any = None,
    ) -> CollectorResult:
        started_at = self.service._utcnow()
        start_monotonic = self.service.monotonic_fn()
        message_count = 0
        termination_reason = "requested_stop"
        try:
            self.service.connect_and_subscribe()
            self.connection_count += 1
            while True:
                if max_messages is not None and message_count >= max_messages:
                    termination_reason = "max_messages"
                    break
                if max_duration_seconds is not None and (self.service.monotonic_fn() - start_monotonic) >= max_duration_seconds:
                    termination_reason = "duration_complete"
                    break
                if stop_fn is not None and stop_fn():
                    termination_reason = "requested_stop"
                    break
                try:
                    frame = self.service.client.receive_message()
                except TimeoutError:
                    if not self.service.heartbeat_is_healthy():
                        if self.service.connection is not None:
                            self.service.connection.heartbeat_timeout_count += 1
                        self._reconnect()
                    continue
                except ConnectionError:
                    self._reconnect()
                    continue
                frame_index = self._record_frame(frame)
                try:
                    self.service.handle_message(
                        frame.raw_bytes, frame_index=frame_index, ingest_time_utc=frame.received_at_utc,
                    )
                except ValueError:
                    if frame.parse_error:
                        self.service.record_malformed_frame(frame.raw_bytes)
                message_count += 1
        except KeyboardInterrupt:
            termination_reason = "keyboard_interrupt"
        except BaseException:
            if self.service.connection is not None:
                self.service.connection.disconnected_at_utc = self.service._utcnow()
                self.service.connection.disconnect_reason = "unexpected_failure"
            self.service.client.close()
            raise
        try:
            if self.service.connection is not None:
                self.service.connection.disconnected_at_utc = self.service._utcnow()
                self.service.connection.disconnect_reason = termination_reason
            self.service.client.close()
            self.raw_segment_writer.seal()
            completed_at = self.service._utcnow()
            return self._finalize(started_at=started_at, completed_at=completed_at, termination_reason=termination_reason)
        finally:
            self.service.on_reconnect_frame = None

    def _reconnect(self) -> None:
        self.raw_segment_writer.seal()
        try:
            self.service.reconnect()
        except RuntimeError as exc:
            raise RuntimeError("reconnect_exhausted") from exc
        self.connection_count += 1
        self.reconnect_count += 1

    def _write_artifacts(self, *, rows: list[dict[str, Any]], data_kind: str, schema_version: str, timestamp_field: str) -> list[dict[str, Any]]:
        if not rows:
            return []
        return self.forward_store.write_rows(
            rows=rows,
            source=self.source,
            product_id=self.service.config.product_id,
            data_kind=data_kind,
            schema_version=schema_version,
            timestamp_field=timestamp_field,
        )

    def _finalize(self, *, started_at: datetime, completed_at: datetime, termination_reason: str) -> CollectorResult:
        self.service.drain_observations()
        raw_frames = [
            (raw, frame["connection_id"], frame["frame_index"])
            for segment in self.raw_segment_writer.sealed_segments
            for raw, frame in zip(
                self.raw_segment_writer.read_segment_frames(segment["path"]), segment["frames"], strict=True,
            )
        ]
        source_samples: list[tuple[datetime, datetime]] = []
        for segment in self.raw_segment_writer.sealed_segments:
            for raw, metadata in zip(
                self.raw_segment_writer.read_segment_frames(segment["path"]), segment["frames"], strict=True,
            ):
                try:
                    event = parse_coinbase_ws_message(raw)
                except (ValueError, UnicodeError):
                    continue
                kind = event.get("type")
                if kind not in {"snapshot", "l2_data", "heartbeat"}:
                    continue
                source_time = (
                    event.get("time_utc") if kind == "heartbeat" else
                    event.get("envelope_time_utc") if kind == "snapshot" else
                    event.get("event_time_utc")
                )
                if source_time is not None:
                    source_samples.append((
                        source_time,
                        datetime.fromisoformat(metadata["ingest_time_utc"].replace("Z", "+00:00")),
                    ))
        if source_samples:
            first_source, first_ingest = source_samples[0]
            last_source, last_ingest = source_samples[-1]
            source_start = first_source - max(timedelta(0), first_ingest - started_at)
            source_end = last_source + max(timedelta(0), completed_at - last_ingest)
        else:
            source_start, source_end = started_at, completed_at
        connections = [*self.service.connection_history]
        if self.service.connection is not None:
            connections.append(self.service.connection)
        self.connection_count = len(connections)
        self.reconnect_count = max(0, self.connection_count - 1)
        observations = replay_recorded_frames(
            raw_frames, product_id=self.service.config.product_id, session_id=self.service.session_id,
            heartbeat_timeout_seconds=self.service.config.heartbeat_timeout_seconds,
            connections=connections,
            session_started_at_utc=source_start, session_completed_at_utc=source_end,
            derived_at_utc=completed_at,
        )
        bbo_rows = [_observation_row(observation) for observation in observations]
        bbo_artifacts = self._write_artifacts(
            rows=bbo_rows, data_kind=self.data_kind, schema_version=FORWARD_BBO_SCHEMA_VERSION,
            timestamp_field="boundary_time_utc",
        )

        level2_update_rows = self.service.drain_level2_update_rows()
        level2_update_artifacts = self._write_artifacts(
            rows=level2_update_rows,
            data_kind=DATA_KIND_LEVEL2_UPDATES,
            schema_version=FORWARD_LEVEL2_UPDATES_SCHEMA_VERSION,
            timestamp_field="event_time_utc",
        )

        bbo_state_rows = self.service.drain_bbo_state_rows()
        bbo_state_artifacts = self._write_artifacts(
            rows=bbo_state_rows, data_kind=DATA_KIND_BBO_STATE, schema_version=FORWARD_BBO_STATE_SCHEMA_VERSION,
            timestamp_field="state_time_utc",
        )

        normalized_artifacts = [*bbo_artifacts, *level2_update_artifacts, *bbo_state_artifacts]
        eligible_count = sum(1 for observation in observations if observation.canonical_target_eligible)
        coverage = {
            "observation_count": len(observations),
            "eligible_observation_count": eligible_count,
            "level2_update_row_count": len(level2_update_rows),
            "bbo_state_row_count": len(bbo_state_rows),
            "connection_count": self.connection_count,
            "reconnect_count": self.reconnect_count,
            "first_observation_utc": (
                observations[0].timestamp_utc.isoformat().replace("+00:00", "Z") if observations else None
            ),
            "last_observation_utc": (
                observations[-1].timestamp_utc.isoformat().replace("+00:00", "Z") if observations else None
            ),
        }
        from dataclasses import fields

        def connection_record(connection: Any) -> dict[str, Any]:
            return {
                field_.name: (
                    value.isoformat().replace("+00:00", "Z") if isinstance(value := getattr(connection, field_.name), datetime)
                    else value
                )
                for field_ in fields(connection)
            }

        counters = (
            "sequence_gap_count", "stale_sequence_count", "heartbeat_timeout_count",
            "heartbeat_discontinuity_count", "malformed_frame_count",
            "malformed_level2_count", "crossed_book_count",
        )
        integrity = {name: sum(getattr(connection, name) for connection in connections) for name in counters}
        manifest = build_forward_manifest(
            source=self.source,
            product_id=self.service.config.product_id,
            session_id=self.service.session_id,
            websocket_url=self.service.config.url,
            channels=["level2", "heartbeats"],
            session_started_at_utc=started_at,
            session_completed_at_utc=completed_at,
            termination_reason=termination_reason,
            connection_count=self.connection_count,
            reconnect_count=self.reconnect_count,
            connections=[connection_record(c) for c in connections],
            raw_segments=self.raw_segment_writer.sealed_segments,
            normalized_artifacts=normalized_artifacts,
            coverage=coverage,
            quarter_hour_summary={
                "boundaries_seen": len(observations),
                "eligible_boundaries": eligible_count,
                "ineligible_boundaries": len(observations) - eligible_count,
            },
            integrity=integrity,
        )
        manifest_path = write_forward_manifest(manifest, self.output_root)
        return CollectorResult(
            manifest=manifest,
            manifest_path=manifest_path,
            observation_count=len(observations),
            eligible_observation_count=eligible_count,
            raw_segments=self.raw_segment_writer.sealed_segments,
            normalized_artifacts=normalized_artifacts,
            bbo_normalized_artifacts=bbo_artifacts,
            level2_update_artifacts=level2_update_artifacts,
            bbo_state_artifacts=bbo_state_artifacts,
            level2_update_row_count=len(level2_update_rows),
            bbo_state_row_count=len(bbo_state_rows),
            connection_count=self.connection_count,
            reconnect_count=self.reconnect_count,
            termination_reason=termination_reason,
            raw_frame_count=self._frame_count,
            level2_message_count=sum(c.level2_messages_received for c in connections),
            heartbeat_message_count=sum(c.heartbeat_messages_received for c in connections),
        )


__all__ = ["WebSocketCollector", "CollectorResult"]
