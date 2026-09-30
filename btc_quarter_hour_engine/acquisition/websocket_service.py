from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from btc_quarter_hour_engine.market_data.order_book import Level2OrderBook
from btc_quarter_hour_engine.market_data.replay import BoundaryEventProcessor, process_parsed_event

from .coinbase_websocket import CoinbaseWebSocketClient, parse_coinbase_ws_message
from .config import CoinbaseWebSocketConfig


@dataclass(slots=True)
class ConnectionDiagnostics:
    connection_id: str
    connected_at_utc: datetime | None = None
    subscribed_at_utc: datetime | None = None
    snapshot_synced_at_utc: datetime | None = None
    invalidated_at_utc: datetime | None = None
    disconnected_at_utc: datetime | None = None
    disconnect_reason: str | None = None
    last_sequence_num: int | None = None
    messages_received: int = 0
    level2_messages_received: int = 0
    heartbeat_messages_received: int = 0
    sequence_gap_count: int = 0


@dataclass(slots=True)
class CoinbaseWebSocketService:
    """Stateful wrapper coordinating connection lifecycle, sequence/heartbeat
    correctness, order-book state, and quarter-hour boundary observation
    derivation for one logical Coinbase Advanced Trade websocket session.

    ``observations`` accumulates every derived quarter-hour observation since
    construction, across reconnects; callers (the collector) drain it with
    :meth:`drain_observations`.
    """

    config: CoinbaseWebSocketConfig = field(default_factory=CoinbaseWebSocketConfig)
    client: CoinbaseWebSocketClient | None = None
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    connection: ConnectionDiagnostics | None = None
    order_book: Level2OrderBook = field(default_factory=Level2OrderBook)
    boundary_processor: BoundaryEventProcessor | None = None
    last_heartbeat_at: datetime | None = None
    last_heartbeat_monotonic: float | None = None
    heartbeat_counter: int | None = None
    sleep_fn: Any = None
    monotonic_fn: Any = None
    observations: list[Any] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = CoinbaseWebSocketClient(config=self.config)
        self.order_book.product_id = self.config.product_id
        if self.boundary_processor is None:
            self.boundary_processor = BoundaryEventProcessor(
                product_id=self.config.product_id,
                heartbeat_timeout_seconds=self.config.heartbeat_timeout_seconds,
            )
        if self.monotonic_fn is None:
            self.monotonic_fn = time.monotonic
        if self.sleep_fn is None:
            self.sleep_fn = time.sleep
        if self.last_heartbeat_monotonic is None:
            self.last_heartbeat_monotonic = self.monotonic_fn()

    def _utcnow(self) -> datetime:
        return datetime.now(timezone.utc)

    def new_connection(self) -> ConnectionDiagnostics:
        self.order_book.reset_for_new_connection()
        self.order_book.invalidate("new connection epoch")
        self.boundary_processor.note_gap()
        connection = ConnectionDiagnostics(connection_id=str(uuid.uuid4()), connected_at_utc=self._utcnow())
        self.connection = connection
        return connection

    def connect_and_subscribe(self) -> ConnectionDiagnostics:
        connection = self.new_connection()
        self.client.connect()
        connection.connected_at_utc = self._utcnow()
        self.client.subscribe(product_id=self.config.product_id)
        connection.subscribed_at_utc = self._utcnow()
        self.connection = connection
        return connection

    def _record_diagnostic(self, *, kind: str, **payload: Any) -> None:
        self.diagnostics.append({"kind": kind, **payload})

    def handle_message(self, message: Any) -> dict[str, Any]:
        payload = parse_coinbase_ws_message(message)
        message_type = payload.get("type")
        if self.connection is not None:
            self.connection.messages_received += 1

        if message_type in {"snapshot", "l2_data"}:
            if self.connection is not None:
                self.connection.level2_messages_received += 1
            sequence_num = payload.get("sequence_num")
            was_synced_before = self.order_book.is_synced()
            new_observations = process_parsed_event(event=payload, book=self.order_book, processor=self.boundary_processor)
            self.observations.extend(new_observations)
            if self.connection is not None:
                self.connection.last_sequence_num = sequence_num
            if self.boundary_processor.gap_since_sync:
                if self.connection is not None:
                    self.connection.sequence_gap_count += 1
                    self.connection.invalidated_at_utc = self._utcnow()
                self._record_diagnostic(
                    kind="sequence_error",
                    expected_sequence=(
                        self.order_book.last_sequence_num + 1 if self.order_book.last_sequence_num is not None else None
                    ),
                    received_sequence=sequence_num,
                    gap_detected_at=self._utcnow().isoformat(),
                    connection_id=self.connection.connection_id if self.connection else None,
                )
                if was_synced_before:
                    raise ValueError(f"Order book desynchronized at sequence {sequence_num!r}")
                raise ValueError(f"Order book failed to synchronize at sequence {sequence_num!r}")
            if message_type == "snapshot":
                if self.connection is not None:
                    self.connection.snapshot_synced_at_utc = self._utcnow()
                return {"status": "snapshot_synced", "book": self.order_book.snapshot()}
            return {"status": "l2_update_applied", "book": self.order_book.snapshot()}

        if message_type == "heartbeat":
            if self.connection is not None:
                self.connection.heartbeat_messages_received += 1
            heartbeat_time = payload.get("time_utc") or self._utcnow()
            new_observations = self.boundary_processor.observe(book=self.order_book, event_time_utc=heartbeat_time)
            self.observations.extend(new_observations)
            self.boundary_processor.observe_heartbeat(time_utc=heartbeat_time)
            self.last_heartbeat_at = heartbeat_time
            self.last_heartbeat_monotonic = self.monotonic_fn()
            if payload.get("sequence") is not None:
                self.heartbeat_counter = int(payload["sequence"])
            return {"status": "heartbeat", "heartbeat_counter": self.heartbeat_counter}

        return {"status": "ignored", "message": payload}

    def drain_observations(self) -> list[Any]:
        drained = self.observations
        self.observations = []
        return drained

    def heartbeat_is_healthy(self) -> bool:
        if self.last_heartbeat_monotonic is None:
            return True
        return (self.monotonic_fn() - self.last_heartbeat_monotonic) <= self.config.heartbeat_timeout_seconds

    def mark_invalid(self, reason: str) -> None:
        self.order_book.invalidate(reason)
        self.boundary_processor.note_gap()
        if self.connection is not None:
            self.connection.invalidated_at_utc = self._utcnow()
            self.connection.disconnect_reason = reason

    def reconnect(self) -> ConnectionDiagnostics:
        if self.client is not None:
            self.client.close()
        self.mark_invalid("reconnect")
        self.order_book.reset()
        self.connection = None
        delay = self.config.initial_reconnect_backoff_seconds
        attempt = 1
        while True:
            self.sleep_fn(delay)
            connection = self.connect_and_subscribe()
            try:
                frame = self.client.receive_message()
                self.handle_message(frame.raw)
            except (TimeoutError, ConnectionError, ValueError):
                pass
            if self.order_book.is_synced():
                return connection
            if self.config.max_reconnect_attempts is not None and attempt >= self.config.max_reconnect_attempts:
                raise RuntimeError("WebSocket reconnect attempts exhausted")
            attempt += 1
            delay = min(self.config.max_reconnect_backoff_seconds, delay * 2)


__all__ = ["CoinbaseWebSocketService", "ConnectionDiagnostics"]
