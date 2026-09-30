from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from .boundary_observations import QuarterHourObservation, derive_quarter_hour_observation
from .order_book import Level2OrderBook, SequenceDisposition, SequenceGapError


def floor_to_quarter_hour(timestamp: datetime) -> datetime:
    """Return the exact quarter-hour boundary at or before ``timestamp`` (UTC)."""
    utc = timestamp.astimezone(timezone.utc)
    minute = (utc.minute // 15) * 15
    return utc.replace(minute=minute, second=0, microsecond=0)


def _event_time(event: Mapping[str, Any]) -> datetime | None:
    return event.get("event_time_utc") or event.get("time_utc") or event.get("envelope_time_utc")


@dataclass(slots=True)
class BoundaryEventProcessor:
    """Derive quarter-hour BBO observations deterministically as events arrive, in event-time order.

    This is the single source of eligibility logic shared by both the live
    collector and offline deterministic replay of previously recorded
    (sealed) raw segments, so a backfill replay of a recorded session
    produces byte-for-byte the same observations and eligibility flags that
    would have been produced live.

    An observation is derived for every exact quarter-hour boundary crossed
    by event time (not wall-clock receive time), using the order book state
    *as of* that boundary. A boundary is marked ``eligible`` only when, at
    that instant:
      - the book is synchronized (has a valid two-sided BBO), and
      - no sequence gap/invalidation has occurred since the last successful
        snapshot sync, and
      - a heartbeat has been observed within ``heartbeat_timeout_seconds``
        of the boundary (proving the connection was alive, not silently
        stalled).
    """

    product_id: str
    heartbeat_timeout_seconds: float = 5.0
    last_heartbeat_at: datetime | None = None
    gap_since_sync: bool = True
    _last_boundary_checked: datetime | None = field(default=None, repr=False)

    def observe_heartbeat(self, *, time_utc: datetime) -> None:
        self.last_heartbeat_at = time_utc.astimezone(timezone.utc)

    def note_gap(self) -> None:
        self.gap_since_sync = True

    def note_synced(self) -> None:
        self.gap_since_sync = False

    def observe(self, *, book: Level2OrderBook, event_time_utc: datetime | None) -> list[QuarterHourObservation]:
        """Emit observations for every quarter-hour boundary newly crossed as of ``event_time_utc``."""
        if event_time_utc is None:
            return []
        event_time_utc = event_time_utc.astimezone(timezone.utc)
        floor = floor_to_quarter_hour(event_time_utc)
        if self._last_boundary_checked is None:
            # Bootstrap: allow the very first quarter-hour containing this event to be
            # eligible for emission without walking arbitrarily far into the past.
            self._last_boundary_checked = floor - timedelta(minutes=15)
        boundaries: list[datetime] = []
        cursor = self._last_boundary_checked + timedelta(minutes=15)
        while cursor <= floor:
            boundaries.append(cursor)
            cursor += timedelta(minutes=15)
        observations: list[QuarterHourObservation] = []
        for boundary in boundaries:
            eligible = (
                book.is_synced()
                and not self.gap_since_sync
                and self.last_heartbeat_at is not None
                and (boundary - self.last_heartbeat_at).total_seconds() <= self.heartbeat_timeout_seconds
            )
            if book.is_synced():
                observations.append(
                    derive_quarter_hour_observation(
                        book=book,
                        timestamp_utc=boundary,
                        product_id=self.product_id,
                        eligible=eligible,
                    )
                )
            self._last_boundary_checked = boundary
        return observations


def process_parsed_event(
    *,
    event: Mapping[str, Any],
    book: Level2OrderBook,
    processor: BoundaryEventProcessor,
) -> list[QuarterHourObservation]:
    """Apply one already-parsed Coinbase event to ``book``/``processor`` and return new observations.

    Shared by the live collector (fed messages as they arrive) and
    :func:`replay_events` (fed a deterministically time-sorted recorded
    stream), guaranteeing identical semantics in both modes.
    """
    event_type = event.get("type")
    observations = processor.observe(book=book, event_time_utc=_event_time(event))
    if event_type == "heartbeat":
        time_utc = event.get("time_utc")
        processor.observe_heartbeat(time_utc=time_utc or datetime.now(timezone.utc))
        return observations
    if event_type not in {"snapshot", "l2_data"}:
        return observations
    sequence_num = event.get("sequence_num")
    try:
        disposition = book.validate_sequence(sequence_num)
    except SequenceGapError:
        book.invalidate("sequence gap")
        processor.note_gap()
        return observations
    if disposition is SequenceDisposition.STALE:
        # A redelivered/duplicate message for this connection epoch: its
        # effect is already reflected in the book, so it is safely dropped
        # without invalidating the book or marking a gap.
        return observations
    try:
        if event_type == "snapshot":
            book.reset()
            updates = event.get("updates", [])
            book.apply_snapshot(
                product_id=book.product_id,
                levels={
                    "bid": [u for u in updates if u["side"] == "bid"],
                    "ask": [u for u in updates if u["side"] == "ask"],
                },
            )
        else:
            for change in event.get("updates", []):
                book.apply_update(side=change["side"], price=change["price"], quantity=change["quantity"])
    except ValueError:
        processor.note_gap()
        return observations
    processor.note_synced()
    return observations


def replay_events(
    events: Iterable[Mapping[str, Any]],
    *,
    product_id: str,
    heartbeat_timeout_seconds: float = 5.0,
) -> list[QuarterHourObservation]:
    """Deterministically replay a recorded (possibly out-of-order) parsed event stream.

    Events are sorted by event time before replay so that the result depends
    only on event-time ordering, never on the order frames happened to be
    received or stored in -- the same recorded segment always replays to the
    same observations.
    """
    ordered = sorted(
        (event for event in events if _event_time(event) is not None or event.get("type") == "heartbeat"),
        key=lambda event: _event_time(event) or datetime.max.replace(tzinfo=timezone.utc),
    )
    book = Level2OrderBook(product_id=product_id)
    processor = BoundaryEventProcessor(product_id=product_id, heartbeat_timeout_seconds=heartbeat_timeout_seconds)
    observations: list[QuarterHourObservation] = []
    for event in ordered:
        observations.extend(process_parsed_event(event=event, book=book, processor=processor))
    return observations


__all__ = [
    "BoundaryEventProcessor",
    "floor_to_quarter_hour",
    "process_parsed_event",
    "replay_events",
]
