from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .order_book import Level2OrderBook, OrderBookState
from btc_quarter_hour_engine.storage.forward_schema import COINBASE_BOUNDARY_BBO_SCHEMA_VERSION, FORWARD_SOURCE


def require_utc(timestamp: datetime) -> datetime:
    if not isinstance(timestamp, datetime) or timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("Canonical timestamps must be timezone-aware")
    return timestamp.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class QuarterHourObservation:
    source: str
    product_id: str
    boundary_time_utc: datetime
    session_id: str | None
    connection_id: str | None
    source_state_time_utc: datetime | None
    derived_at_utc: datetime
    source_sequence_num: int | None
    best_bid: float | None
    best_bid_size: float | None
    best_ask: float | None
    best_ask_size: float | None
    spread: float | None
    midpoint: float | None
    book_synced: bool
    canonical_target_eligible: bool
    eligibility_reason: str
    boundary_schema_version: str = COINBASE_BOUNDARY_BBO_SCHEMA_VERSION

    @property
    def timestamp_utc(self) -> datetime:
        return self.boundary_time_utc

    @property
    def eligible(self) -> bool:
        return self.canonical_target_eligible


def is_exact_quarter_hour_boundary(timestamp: datetime) -> bool:
    utc = require_utc(timestamp)
    return utc.minute % 15 == 0 and utc.second == 0 and utc.microsecond == 0


def derive_quarter_hour_observation(
    *,
    book: Level2OrderBook,
    timestamp_utc: datetime,
    product_id: str | None = None,
    session_id: str | None = None,
    connection_id: str | None = None,
    source_state_time_utc: datetime | None = None,
    derived_at_utc: datetime | None = None,
    source_sequence_num: int | None = None,
    last_heartbeat_at: datetime | None = None,
    heartbeat_timeout_seconds: float = 5.0,
    integrity_reason: str | None = None,
) -> QuarterHourObservation:
    boundary = require_utc(timestamp_utc)
    if not is_exact_quarter_hour_boundary(boundary):
        raise ValueError("Timestamp is not an exact quarter-hour boundary")
    if heartbeat_timeout_seconds < 0:
        raise ValueError("Heartbeat timeout must be nonnegative")
    state_time = require_utc(source_state_time_utc) if source_state_time_utc is not None else None
    derived = require_utc(derived_at_utc) if derived_at_utc is not None else boundary
    heartbeat = require_utc(last_heartbeat_at) if last_heartbeat_at is not None else None
    bid, ask = book.best_bid, book.best_ask
    bid_size, ask_size = book.best_bid_size, book.best_ask_size
    synced = book.is_synced()
    if integrity_reason:
        reason = integrity_reason
    elif not synced:
        if book.state == OrderBookState.REBUILDING:
            reason = "book_rebuilding"
        elif book.last_error and "crossed" in book.last_error.lower():
            reason = "crossed_book"
        elif book.last_error and "sequence gap" in book.last_error.lower():
            reason = "sequence_gap"
        elif book.last_error and "malformed" in book.last_error.lower():
            reason = "malformed_source_state"
        elif not book.bids and not book.asks:
            reason = "no_synced_snapshot"
        elif not book.bids:
            reason = "missing_bid"
        elif not book.asks:
            reason = "missing_ask"
        else:
            reason = "no_synced_snapshot"
    elif bid is None:
        reason = "missing_bid"
    elif ask is None:
        reason = "missing_ask"
    elif heartbeat is None or heartbeat > boundary or (boundary - heartbeat).total_seconds() > heartbeat_timeout_seconds:
        reason = "heartbeat_stale"
    else:
        reason = "eligible"
    valid_bbo = synced and bid is not None and ask is not None
    return QuarterHourObservation(
        source=FORWARD_SOURCE,
        product_id=product_id or book.product_id or "",
        boundary_time_utc=boundary,
        session_id=session_id,
        connection_id=connection_id,
        source_state_time_utc=state_time,
        derived_at_utc=derived,
        source_sequence_num=source_sequence_num,
        best_bid=float(bid) if valid_bbo else None,
        best_bid_size=float(bid_size) if valid_bbo and bid_size is not None else None,
        best_ask=float(ask) if valid_bbo else None,
        best_ask_size=float(ask_size) if valid_bbo and ask_size is not None else None,
        spread=float(ask - bid) if valid_bbo else None,
        midpoint=float((bid + ask) / 2) if valid_bbo else None,
        book_synced=synced,
        canonical_target_eligible=reason == "eligible",
        eligibility_reason=reason,
    )


__all__ = ["QuarterHourObservation", "derive_quarter_hour_observation", "is_exact_quarter_hour_boundary"]
