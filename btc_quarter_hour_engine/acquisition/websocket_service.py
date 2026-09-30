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


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


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
    stale_sequence_count: int = 0


def _level2_update_rows(
    *,
    event: dict[str, Any],
    product_id: str,
    connection_id: str | None,
    fallback_time_utc: datetime,
) -> list[dict[str, Any]]:
    """Build one normalized ``level2_updates`` row per individual book mutation.

    Each row preserves the update's own per-entry ``event_time_utc`` (full
    wire fidelity, distinct from the message envelope time and from the
    coarser quarter-hour boundary observations) alongside the sequence
    number and connection it arrived on.
    """
    sequence_num = event.get("sequence_num")
    rows: list[dict[str, Any]] = []
    for update in event.get("updates", []):
        event_time_utc = update.get("event_time_utc") or fallback_time_utc
        rows.append(
            {
                "source_time_utc": event_time_utc,
                "product_id": product_id,
                "connection_id": connection_id,
                "sequence_num": sequence_num,
                "message_type": event.get("type"),
                "side": update.get("side"),
                "price": update.get("price"),
                "quantity": update.get("quantity"),
            }
        )
    return rows


def _bbo_state_row(
    *,
    book: Level2OrderBook,
    connection_id: str | None,
    source_time_utc: datetime,
) -> dict[str, Any]:
    return {
        "source_time_utc": source_time_utc,
        "product_id": book.product_id,
        "connection_id": connection_id,
        "state": book.state.value,
        "best_bid": book.best_bid,
        "best_ask": book.best_ask,
        "best_bid_size": book.best_bid_size,
        "best_ask_size": book.best_ask_size,
        "midpoint": book.midpoint,
    }


@dataclass(slots=True)
class CoinbaseWebSocketService:
    """Stateful wrapper coordinating connection lifecycle, sequence/heartbeat
    correctness, order-book state, and quarter-hour boundary observation
    derivation for one logical Coinbase Advanced Trade websocket session.

    ``observations`` accumulates every derived quarter-hour observation since
    construction, across reconnects; ``level2_update_rows``/``bbo_state_rows``
    accumulate normalized per-message-update and per-mutation book-state rows
    respectively (finer-grained than the boundary observations, for full
    forward-collection fidelity). Callers (the collector) drain all three
    with :meth:`drain_observations`/:meth:`drain_level2_update_rows`/
    :meth:`drain_bbo_state_rows`.

    ``now_fn`` and ``monotonic_fn`` are injectable clocks (defaulting to the
    real wall clock and ``time.monotonic``) so tests can fully control both
    wall-clock timestamps recorded on diagnostics/rows and the monotonic
    budget used by heartbeat-health checks and reconnect backoff/wait-for-
    snapshot timing, without any real sleeping.
    """

    config: CoinbaseWebSocketConfig = field(default_factory=CoinbaseWebSocketConfig)
    client: CoinbaseWebSocketClient | None = None
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    connection: ConnectionDiagnostics | None = None
    # Every prior connection for this service instance, oldest first, for
    # audit/diagnostics (e.g. counting reconnects, inspecting why a past
    # connection was invalidated). The current connection is appended here
    # only once superseded by a new one.
    connection_history: list[ConnectionDiagnostics] = field(default_factory=list)
    order_book: Level2OrderBook = field(default_factory=Level2OrderBook)
    boundary_processor: BoundaryEventProcessor | None = None
    last_heartbeat_at: datetime | None = None
    last_heartbeat_monotonic: float | None = None
    heartbeat_counter: int | None = None
    sleep_fn: Any = None
    monotonic_fn: Any = None
    now_fn: Any = None
    observations: list[Any] = field(default_factory=list)
    level2_update_rows: list[dict[str, Any]] = field(default_factory=list)
    bbo_state_rows: list[dict[str, Any]] = field(default_factory=list)
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
        if self.now_fn is None:
            self.now_fn = _utcnow
        if self.last_heartbeat_monotonic is None:
            self.last_heartbeat_monotonic = self.monotonic_fn()

    def _utcnow(self) -> datetime:
        return self.now_fn()

    def new_connection(self) -> ConnectionDiagnostics:
        if self.connection is not None:
            self.connection_history.append(self.connection)
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
            connection_id = self.connection.connection_id if self.connection else None
            sequence_num = payload.get("sequence_num")
            was_synced_before = self.order_book.is_synced()
            is_stale = self.order_book.classify_sequence(sequence_num).value == "stale"
            new_observations = process_parsed_event(event=payload, book=self.order_book, processor=self.boundary_processor)
            self.observations.extend(new_observations)
            if is_stale:
                if self.connection is not None:
                    self.connection.stale_sequence_count += 1
                return {"status": "stale_sequence_ignored", "book": self.order_book.snapshot()}
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
                    connection_id=connection_id,
                )
                if was_synced_before:
                    raise ValueError(f"Order book desynchronized at sequence {sequence_num!r}")
                raise ValueError(f"Order book failed to synchronize at sequence {sequence_num!r}")
            envelope_time_utc = payload.get("envelope_time_utc") or self._utcnow()
            self.level2_update_rows.extend(
                _level2_update_rows(
                    event=payload, product_id=self.config.product_id, connection_id=connection_id,
                    fallback_time_utc=envelope_time_utc,
                )
            )
            if self.order_book.is_synced():
                self.bbo_state_rows.append(
                    _bbo_state_row(book=self.order_book, connection_id=connection_id, source_time_utc=envelope_time_utc)
                )
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

    def drain_level2_update_rows(self) -> list[dict[str, Any]]:
        drained = self.level2_update_rows
        self.level2_update_rows = []
        return drained

    def drain_bbo_state_rows(self) -> list[dict[str, Any]]:
        drained = self.bbo_state_rows
        self.bbo_state_rows = []
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

    def _wait_for_snapshot(self, *, deadline_seconds: float) -> None:
        """Keep receiving messages on the current connection until the order
        book becomes synced (a snapshot has been applied) or ``deadline_seconds``
        of monotonic budget elapses, whichever comes first.

        A real reconnect does not necessarily receive a snapshot as its very
        first message (heartbeats or stale/duplicate frames from the prior
        epoch may arrive first), so this actively drains messages rather than
        giving up after a single receive attempt.
        """
        start = self.monotonic_fn()
        while (self.monotonic_fn() - start) < deadline_seconds:
            try:
                frame = self.client.receive_message()
            except TimeoutError:
                continue
            try:
                self.handle_message(frame.raw)
            except ValueError:
                pass
            if self.order_book.is_synced():
                return

    def reconnect(self) -> ConnectionDiagnostics:
        if self.client is not None:
            self.client.close()
        self.mark_invalid("reconnect")
        if self.connection is not None:
            self.connection.disconnected_at_utc = self._utcnow()
        self.order_book.reset_for_new_connection()
        delay = self.config.initial_reconnect_backoff_seconds
        attempt = 1
        while True:
            self.sleep_fn(delay)
            connection = self.connect_and_subscribe()
            try:
                self._wait_for_snapshot(deadline_seconds=self.config.snapshot_wait_timeout_seconds)
            except ConnectionError:
                pass
            if self.order_book.is_synced():
                return connection
            if self.config.max_reconnect_attempts is not None and attempt >= self.config.max_reconnect_attempts:
                raise RuntimeError("WebSocket reconnect attempts exhausted")
            attempt += 1
            delay = min(self.config.max_reconnect_backoff_seconds, delay * 2)


__all__ = ["CoinbaseWebSocketService", "ConnectionDiagnostics"]
