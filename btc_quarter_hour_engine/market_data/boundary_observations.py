from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .order_book import Level2OrderBook


@dataclass(frozen=True, slots=True)
class QuarterHourObservation:
    timestamp_utc: datetime
    best_bid: float
    best_ask: float
    best_bid_size: float
    best_ask_size: float
    midpoint: float
    eligible: bool = False
    product_id: str | None = None


def is_exact_quarter_hour_boundary(timestamp: datetime) -> bool:
    utc = timestamp.astimezone(timezone.utc)
    return utc.minute % 15 == 0 and utc.second == 0 and utc.microsecond == 0


def derive_quarter_hour_observation(
    *,
    book: Level2OrderBook,
    timestamp_utc: datetime,
    product_id: str | None = None,
    eligible: bool = False,
) -> QuarterHourObservation:
    if not book.is_synced():
        raise ValueError("Book is not synchronized")
    if not is_exact_quarter_hour_boundary(timestamp_utc):
        raise ValueError("Timestamp is not an exact quarter-hour boundary")
    bid = book.best_bid
    ask = book.best_ask
    bid_size = book.best_bid_size
    ask_size = book.best_ask_size
    if bid is None or ask is None or bid_size is None or ask_size is None:
        raise ValueError("Book lacks a valid BBO")
    midpoint = (bid + ask) / 2.0
    return QuarterHourObservation(
        timestamp_utc=timestamp_utc.astimezone(timezone.utc),
        best_bid=float(bid),
        best_ask=float(ask),
        best_bid_size=float(bid_size),
        best_ask_size=float(ask_size),
        midpoint=float(midpoint),
        eligible=bool(eligible),
        product_id=product_id or book.product_id,
    )


__all__ = ["QuarterHourObservation", "derive_quarter_hour_observation", "is_exact_quarter_hour_boundary"]
