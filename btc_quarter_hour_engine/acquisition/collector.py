from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any

from btc_quarter_hour_engine.market_data.replay import CanonicalReplayAccumulator
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

from .websocket_service import (
    CoinbaseWebSocketService,
    EnvelopeSequenceGapError,
    ReconnectExhaustedError,
    ReconnectStopRequested,
)


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
    source_metrics: dict[str, Any] | None = None


def _observation_row(observation: Any) -> dict[str, Any]:
    row = asdict(observation)
    row.pop("last_heartbeat_at_utc", None)
    row.pop("heartbeat_timeout_seconds", None)
    return row


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
    flush_row_threshold: int = 100_000
    flush_interval_seconds: float = 300.0
    source_publisher: Any = None
    source_publisher_factory: Any = None
    connection_count: int = field(default=0, init=False)
    reconnect_count: int = field(default=0, init=False)
    _frame_count: int = field(default=0, init=False)
    _checkpoint_started_monotonic: float = field(default=0.0, init=False)
    _level2_update_row_count: int = field(default=0, init=False)
    _bbo_state_row_count: int = field(default=0, init=False)
    _level2_update_artifacts: list[dict[str, Any]] = field(default_factory=list, init=False)
    _bbo_state_artifacts: list[dict[str, Any]] = field(default_factory=list, init=False)
    _canonical_accumulator: CanonicalReplayAccumulator = field(init=False)
    _stop_fn: Any = field(default=None, init=False)
    _source_connection_id: str | None = field(default=None, init=False)
    _source_reconnect_epoch: int = field(default=-1, init=False)

    def __post_init__(self) -> None:
        if self.sleep_fn is None:
            self.sleep_fn = time.sleep
        if self.flush_row_threshold < 1:
            raise ValueError("flush_row_threshold must be >= 1")
        if not math.isfinite(self.flush_interval_seconds) or self.flush_interval_seconds <= 0:
            raise ValueError("flush_interval_seconds must be > 0")
        self.raw_segment_writer.bind_session(self.service.session_id)
        self.service.on_reconnect_frame = self._record_frame
        self._canonical_accumulator = CanonicalReplayAccumulator(
            product_id=self.service.config.product_id,
            session_id=self.service.session_id,
            heartbeat_timeout_seconds=self.service.config.heartbeat_timeout_seconds,
        )

    def _connections(self) -> list[Any]:
        connections = [*self.service.connection_history]
        if self.service.connection is not None:
            connections.append(self.service.connection)
        return connections

    def _record_frame(self, frame: Any) -> int:
        connection_id = self.service.connection.connection_id if self.service.connection else None
        message_type = frame.message.get("type", "unknown") if frame.message is not None else "malformed"
        sequence_num = frame.message.get("sequence_num") if frame.message is not None else None
        frame_index = self._frame_count
        self.raw_segment_writer.add_frame(
            raw=frame.raw_bytes, message_type=message_type, connection_id=connection_id,
            sequence_num=sequence_num, ingest_time_utc=frame.received_at_utc,
        )
        self._canonical_accumulator.consume(
            frame.raw_bytes,
            connection_id,
            frame_index,
            frame.received_at_utc,
        )
        self._frame_count += 1
        return frame_index

    @staticmethod
    def _health_timestamp(value: datetime | None) -> str | None:
        if value is None:
            return None
        return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        )

    def _publish_source_frame(
        self,
        frame: Any,
        frame_index: int | None,
        observations: list[Any],
    ) -> None:
        if self.source_publisher is None or frame_index is None:
            return
        connection = self.service.connection
        connection_id = connection.connection_id if connection is not None else None
        if connection_id != self._source_connection_id:
            self._source_reconnect_epoch += 1
            self._source_connection_id = connection_id
        received = frame.received_at_utc.astimezone(timezone.utc)
        cutoff = received.replace(
            minute=received.minute - received.minute % 15, second=0, microsecond=0
        )
        boundary_observation = next(
            (
                item
                for item in reversed(observations)
                if item.boundary_time_utc == cutoff
            ),
            None,
        )
        canonical_boundary = None
        if boundary_observation is not None:
            canonical_boundary = {
                "boundary_utc": self._health_timestamp(
                    boundary_observation.boundary_time_utc
                ),
                "canonical_target_eligible": bool(
                    boundary_observation.canonical_target_eligible
                ),
                "eligibility_reason": boundary_observation.eligibility_reason,
                "book_synced": bool(boundary_observation.book_synced),
                "last_heartbeat_at_utc": self._health_timestamp(
                    boundary_observation.last_heartbeat_at_utc
                ),
                "heartbeat_timeout_seconds": float(
                    boundary_observation.heartbeat_timeout_seconds
                ),
                "connection_id": boundary_observation.connection_id,
                "source_sequence_num": boundary_observation.source_sequence_num,
            }
        diagnostics = connection
        health = {
            "publisher_valid": bool(getattr(self.source_publisher, "valid", True)),
            "heartbeat_seen": self.service.last_heartbeat_at is not None,
            "heartbeat_last_at_utc": self._health_timestamp(
                self.service.last_heartbeat_at
            ),
            "heartbeat_healthy": bool(self.service.heartbeat_is_healthy()),
            "order_book_synchronized": self.service.order_book.is_synced(),
            "connection_id": connection_id,
            "connection_last_sequence_num": (
                diagnostics.last_envelope_sequence_num if diagnostics is not None else None
            ),
            "last_l2_sequence_num": (
                diagnostics.last_sequence_num if diagnostics is not None else None
            ),
            "sequence_gap_count": (
                diagnostics.sequence_gap_count if diagnostics is not None else 0
            ),
            "stale_sequence_count": (
                diagnostics.stale_sequence_count if diagnostics is not None else 0
            ),
            "heartbeat_timeout_count": (
                diagnostics.heartbeat_timeout_count if diagnostics is not None else 0
            ),
            "heartbeat_discontinuity_count": (
                diagnostics.heartbeat_discontinuity_count if diagnostics is not None else 0
            ),
            "malformed_frame_count": (
                diagnostics.malformed_frame_count if diagnostics is not None else 0
            ),
            "malformed_level2_count": (
                diagnostics.malformed_level2_count if diagnostics is not None else 0
            ),
            "crossed_book_count": (
                diagnostics.crossed_book_count if diagnostics is not None else 0
            ),
            "canonical_boundary": canonical_boundary,
        }
        try:
            published = self.source_publisher.publish_frame(
                connection_id=connection_id,
                reconnect_epoch=self._source_reconnect_epoch,
                frame_index=frame_index,
                raw_bytes=bytes(frame.raw_bytes),
                received_at_utc=frame.received_at_utc,
                producer_health=health,
            )
            if not published:
                self.service._record_diagnostic(
                    kind="prospective_source_progress_invalid",
                    reason=getattr(self.source_publisher, "failure_reason", None),
                    frame_index=frame_index,
                )
        except Exception as exc:
            self.service._record_diagnostic(
                kind="prospective_source_publish_failed",
                reason=type(exc).__name__,
                frame_index=frame_index,
            )

    def run(
        self,
        *,
        max_messages: int | None = None,
        max_duration_seconds: float | None = None,
        stop_fn: Any = None,
    ) -> CollectorResult:
        started_at = self.service._utcnow()
        if self.source_publisher_factory is not None:
            if self.source_publisher is not None:
                raise ValueError("Provide a source publisher or factory, not both")
            try:
                self.source_publisher = self.source_publisher_factory(
                    self.service.session_id, started_at
                )
            except Exception as exc:
                self.service._record_diagnostic(
                    kind="prospective_source_init_failed",
                    reason=type(exc).__name__,
                )
        start_monotonic = self.service.monotonic_fn()
        self._checkpoint_started_monotonic = start_monotonic
        message_count = 0
        termination_reason = "requested_stop"
        self._stop_fn = stop_fn
        self.service.on_reconnect_frame_processed = self._publish_source_frame
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
                self._maybe_flush_normalized_checkpoint()
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
                observation_start = len(self.service.observations)
                try:
                    self.service.handle_message(
                        frame.raw_bytes, frame_index=frame_index, ingest_time_utc=frame.received_at_utc,
                        include_book_snapshot=False,
                    )
                except EnvelopeSequenceGapError:
                    self._publish_source_frame(
                        frame,
                        frame_index,
                        self.service.observations[observation_start:],
                    )
                    self._reconnect()
                except ValueError:
                    if frame.parse_error:
                        self.service.record_malformed_frame(frame.raw_bytes)
                    self._publish_source_frame(
                        frame,
                        frame_index,
                        self.service.observations[observation_start:],
                    )
                else:
                    self._publish_source_frame(
                        frame,
                        frame_index,
                        self.service.observations[observation_start:],
                    )
                message_count += 1
                self._maybe_flush_normalized_checkpoint()
        except KeyboardInterrupt:
            termination_reason = "keyboard_interrupt"
        except ReconnectStopRequested:
            termination_reason = "requested_stop"
        except ReconnectExhaustedError:
            termination_reason = "reconnect_exhausted"
        except BaseException:
            if self.service.connection is not None:
                self.service.connection.disconnected_at_utc = self.service._utcnow()
                self.service.connection.disconnect_reason = "unexpected_failure"
            self.service.client.close()
            if self.source_publisher is not None:
                self.source_publisher.close()
            raise
        try:
            if self.service.connection is not None:
                self.service.connection.disconnected_at_utc = self.service._utcnow()
                self.service.connection.disconnect_reason = termination_reason
            self.service.client.close()
            self.raw_segment_writer.seal()
            completed_at = self.service._utcnow()
            result = self._finalize(
                started_at=started_at,
                completed_at=completed_at,
                termination_reason=termination_reason,
            )
        finally:
            self.service.on_reconnect_frame = None
            self.service.on_reconnect_frame_processed = None
            self._stop_fn = None
            if self.source_publisher is not None:
                self.source_publisher.close()
        if self.source_publisher is not None:
            result.source_metrics = self.source_publisher.metrics()
        return result

    def _reconnect(self) -> None:
        self._flush_normalized_checkpoint()
        self.raw_segment_writer.seal()
        self.service.reconnect(stop_fn=self._stop_fn)
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

    def _maybe_flush_normalized_checkpoint(self) -> None:
        pending_rows = len(self.service.level2_update_rows) + len(self.service.bbo_state_rows)
        elapsed = self.service.monotonic_fn() - self._checkpoint_started_monotonic
        if pending_rows >= self.flush_row_threshold or elapsed >= self.flush_interval_seconds:
            self._flush_normalized_checkpoint()

    def _flush_normalized_checkpoint(self) -> None:
        level2_rows = self.service.drain_level2_update_rows()
        bbo_rows = self.service.drain_bbo_state_rows()
        self.service.drain_observations()
        try:
            level2_artifacts = self._write_artifacts(
                rows=level2_rows,
                data_kind=DATA_KIND_LEVEL2_UPDATES,
                schema_version=FORWARD_LEVEL2_UPDATES_SCHEMA_VERSION,
                timestamp_field="event_time_utc",
            )
            bbo_artifacts = self._write_artifacts(
                rows=bbo_rows,
                data_kind=DATA_KIND_BBO_STATE,
                schema_version=FORWARD_BBO_STATE_SCHEMA_VERSION,
                timestamp_field="state_time_utc",
            )
        except BaseException:
            self.service.level2_update_rows = level2_rows + self.service.level2_update_rows
            self.service.bbo_state_rows = bbo_rows + self.service.bbo_state_rows
            raise
        self._level2_update_row_count += len(level2_rows)
        self._bbo_state_row_count += len(bbo_rows)
        self._level2_update_artifacts.extend(level2_artifacts)
        self._bbo_state_artifacts.extend(bbo_artifacts)
        self._checkpoint_started_monotonic = self.service.monotonic_fn()

    def _finalize(self, *, started_at: datetime, completed_at: datetime, termination_reason: str) -> CollectorResult:
        self.service.drain_observations()
        connections = self._connections()
        self.connection_count = len(connections)
        self.reconnect_count = max(0, self.connection_count - 1)
        observations = self._canonical_accumulator.finalize(
            session_started_at_utc=started_at,
            session_completed_at_utc=completed_at,
            derived_at_utc=completed_at,
            connections=connections,
        )
        bbo_rows = [_observation_row(observation) for observation in observations]
        bbo_artifacts = self._write_artifacts(
            rows=bbo_rows, data_kind=self.data_kind, schema_version=FORWARD_BBO_SCHEMA_VERSION,
            timestamp_field="boundary_time_utc",
        )

        self._flush_normalized_checkpoint()
        level2_update_artifacts = self._level2_update_artifacts
        bbo_state_artifacts = self._bbo_state_artifacts

        normalized_artifacts = [*bbo_artifacts, *level2_update_artifacts, *bbo_state_artifacts]
        eligible_count = sum(1 for observation in observations if observation.canonical_target_eligible)
        level2_message_count = sum(c.level2_messages_received for c in connections)
        heartbeat_message_count = sum(c.heartbeat_messages_received for c in connections)
        coverage = {
            "observation_count": len(observations),
            "eligible_observation_count": eligible_count,
            "level2_update_row_count": self._level2_update_row_count,
            "normalized_l2_rows": self._level2_update_row_count,
            "bbo_state_row_count": self._bbo_state_row_count,
            "bbo_state_rows": self._bbo_state_row_count,
            "boundary_rows": len(observations),
            "raw_frame_count": self._frame_count,
            "level2_message_count": level2_message_count,
            "heartbeat_message_count": heartbeat_message_count,
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
            "malformed_level2_count", "crossed_book_count", "book_rebuild_count",
            "book_recovery_count",
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
            level2_update_row_count=self._level2_update_row_count,
            bbo_state_row_count=self._bbo_state_row_count,
            connection_count=self.connection_count,
            reconnect_count=self.reconnect_count,
            termination_reason=termination_reason,
            raw_frame_count=self._frame_count,
            level2_message_count=level2_message_count,
            heartbeat_message_count=heartbeat_message_count,
            source_metrics=(
                self.source_publisher.metrics()
                if self.source_publisher is not None
                else None
            ),
        )


__all__ = ["WebSocketCollector", "CollectorResult"]
