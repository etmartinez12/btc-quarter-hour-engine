from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from btc_quarter_hour_engine.storage.forward_manifest import build_forward_manifest
from btc_quarter_hour_engine.storage.forward_parquet import ForwardParquetStore
from btc_quarter_hour_engine.storage.manifest import write_manifest
from btc_quarter_hour_engine.storage.websocket_raw import RawSegmentWriter

from .websocket_service import CoinbaseWebSocketService

FORWARD_BBO_SCHEMA_VERSION = "1"


@dataclass(slots=True)
class CollectorResult:
    manifest: dict[str, Any]
    manifest_path: Path
    observation_count: int
    eligible_observation_count: int
    raw_segments: list[dict[str, Any]]
    normalized_artifacts: list[dict[str, Any]]
    connection_count: int
    reconnect_count: int


def _observation_row(observation: Any) -> dict[str, Any]:
    return {
        "source_time_utc": observation.timestamp_utc,
        "product_id": observation.product_id,
        "best_bid": observation.best_bid,
        "best_ask": observation.best_ask,
        "best_bid_size": observation.best_bid_size,
        "best_ask_size": observation.best_ask_size,
        "midpoint": observation.midpoint,
        "eligible": observation.eligible,
    }


@dataclass(slots=True)
class WebSocketCollector:
    """Drive one :class:`CoinbaseWebSocketService` through a receive loop,
    sealing exact raw frame bytes into immutable segments, tracking
    reconnects, and persisting derived quarter-hour BBO observations plus a
    complete forward manifest on completion.

    Stop conditions (``max_messages`` / ``max_duration_seconds`` /
    ``stop_fn``) make the loop finite and therefore directly testable without
    any network dependency; a production CLI invocation simply omits them
    (or passes a signal-driven ``stop_fn``) to run indefinitely.
    """

    service: CoinbaseWebSocketService
    raw_segment_writer: RawSegmentWriter
    forward_store: ForwardParquetStore
    output_root: str
    source: str = "coinbase_advanced"
    data_kind: str = "quarter_hour_bbo"
    sleep_fn: Any = field(default=None)
    connection_count: int = field(default=0, init=False)
    reconnect_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.sleep_fn is None:
            self.sleep_fn = time.sleep

    def run(
        self,
        *,
        max_messages: int | None = None,
        max_duration_seconds: float | None = None,
        stop_fn: Any = None,
    ) -> CollectorResult:
        started_at = datetime.now(timezone.utc)
        start_monotonic = self.service.monotonic_fn()
        self.service.connect_and_subscribe()
        self.connection_count += 1
        message_count = 0
        try:
            while True:
                if max_messages is not None and message_count >= max_messages:
                    break
                if max_duration_seconds is not None and (self.service.monotonic_fn() - start_monotonic) >= max_duration_seconds:
                    break
                if stop_fn is not None and stop_fn():
                    break
                try:
                    frame = self.service.client.receive_message()
                except TimeoutError:
                    if not self.service.heartbeat_is_healthy():
                        self._reconnect()
                    continue
                except ConnectionError:
                    self._reconnect()
                    continue

                connection_id = self.service.connection.connection_id if self.service.connection else None
                self.raw_segment_writer.add_frame(
                    raw=frame.raw,
                    message_type=frame.message.get("type", "unknown"),
                    connection_id=connection_id,
                    sequence_num=frame.message.get("sequence_num"),
                    received_at_utc=frame.received_at_utc,
                )
                try:
                    # Re-parse from the exact original raw payload (not the
                    # already-normalized `frame.message`) so the service's
                    # own parsing/validation path is exercised identically
                    # to how a recorded segment would be replayed later.
                    self.service.handle_message(frame.raw)
                except ValueError:
                    # Malformed payload or sequence gap: already recorded in
                    # service diagnostics/order-book invalidation; the raw
                    # frame is still sealed above (exact bytes preserved),
                    # continue receiving so a subsequent snapshot can resync.
                    pass
                message_count += 1
        finally:
            self.raw_segment_writer.seal()
        completed_at = datetime.now(timezone.utc)
        return self._finalize(started_at=started_at, completed_at=completed_at)

    def _reconnect(self) -> None:
        self.raw_segment_writer.seal()
        self.service.reconnect()
        self.connection_count += 1
        self.reconnect_count += 1

    def _finalize(self, *, started_at: datetime, completed_at: datetime) -> CollectorResult:
        observations = self.service.drain_observations()
        rows = [_observation_row(observation) for observation in observations]
        normalized_artifacts: list[dict[str, Any]] = []
        if rows:
            normalized_artifacts = self.forward_store.write_rows(
                rows=rows,
                source=self.source,
                product_id=self.service.config.product_id,
                data_kind=self.data_kind,
                schema_version=FORWARD_BBO_SCHEMA_VERSION,
            )
        eligible_count = sum(1 for observation in observations if observation.eligible)
        coverage = {
            "observation_count": len(observations),
            "eligible_observation_count": eligible_count,
            "connection_count": self.connection_count,
            "reconnect_count": self.reconnect_count,
            "first_observation_utc": (
                observations[0].timestamp_utc.isoformat().replace("+00:00", "Z") if observations else None
            ),
            "last_observation_utc": (
                observations[-1].timestamp_utc.isoformat().replace("+00:00", "Z") if observations else None
            ),
        }
        manifest = build_forward_manifest(
            source=self.source,
            product_id=self.service.config.product_id,
            data_kind=self.data_kind,
            raw_artifacts=self.raw_segment_writer.sealed_segments,
            normalized_artifacts=normalized_artifacts,
            coverage=coverage,
            canonical_target_eligible=eligible_count > 0,
            canonical_target_ineligibility_reason=(
                None if eligible_count > 0 else "no eligible quarter-hour boundary observations were captured"
            ),
            acquisition_started_at_utc=started_at,
            acquisition_completed_at_utc=completed_at,
            request_count=len(self.raw_segment_writer.sealed_segments),
            schema_version=FORWARD_BBO_SCHEMA_VERSION,
            purpose="forward_bbo_collection",
        )
        manifest_path = write_manifest(manifest, self.output_root)
        return CollectorResult(
            manifest=manifest,
            manifest_path=manifest_path,
            observation_count=len(observations),
            eligible_observation_count=eligible_count,
            raw_segments=self.raw_segment_writer.sealed_segments,
            normalized_artifacts=normalized_artifacts,
            connection_count=self.connection_count,
            reconnect_count=self.reconnect_count,
        )


__all__ = ["WebSocketCollector", "CollectorResult"]
