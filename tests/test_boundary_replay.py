from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from btc_quarter_hour_engine.market_data.order_book import Level2OrderBook
from btc_quarter_hour_engine.market_data.replay import (
    BoundaryEventProcessor,
    floor_to_quarter_hour,
    process_parsed_event,
    replay_events,
)


def test_floor_to_quarter_hour():
    assert floor_to_quarter_hour(datetime(2024, 1, 1, 0, 14, 59, tzinfo=timezone.utc)) == datetime(
        2024, 1, 1, 0, 0, tzinfo=timezone.utc
    )
    assert floor_to_quarter_hour(datetime(2024, 1, 1, 0, 15, 0, tzinfo=timezone.utc)) == datetime(
        2024, 1, 1, 0, 15, tzinfo=timezone.utc
    )
    assert floor_to_quarter_hour(datetime(2024, 1, 1, 0, 44, tzinfo=timezone.utc)) == datetime(
        2024, 1, 1, 0, 30, tzinfo=timezone.utc
    )


def _snapshot_event(sequence_num, time, bid=100.0, ask=101.0):
    return {
        "type": "snapshot",
        "product_id": "BTC-USD",
        "sequence_num": sequence_num,
        "event_time_utc": time,
        "updates": [
            {"side": "bid", "price": bid, "quantity": 1.0},
            {"side": "ask", "price": ask, "quantity": 1.0},
        ],
    }


def _update_event(sequence_num, time, side, price, quantity):
    return {
        "type": "l2_data",
        "product_id": "BTC-USD",
        "sequence_num": sequence_num,
        "event_time_utc": time,
        "updates": [{"side": side, "price": price, "quantity": quantity}],
    }


def _heartbeat_event(time):
    return {"type": "heartbeat", "time_utc": time}


def test_no_observation_emitted_before_book_is_synced():
    book = Level2OrderBook(product_id="BTC-USD")
    processor = BoundaryEventProcessor(product_id="BTC-USD", heartbeat_timeout_seconds=30)
    # First event's own quarter-hour floor should not retroactively emit before any sync.
    event = _snapshot_event(1, datetime(2024, 1, 1, 0, 15, 0, tzinfo=timezone.utc))
    observations = process_parsed_event(event=event, book=book, processor=processor)
    assert observations == []
    assert book.is_synced()


def test_boundary_crossed_with_healthy_heartbeat_is_eligible():
    book = Level2OrderBook(product_id="BTC-USD")
    processor = BoundaryEventProcessor(product_id="BTC-USD", heartbeat_timeout_seconds=30)
    process_parsed_event(event=_snapshot_event(1, datetime(2024, 1, 1, 0, 14, 50, tzinfo=timezone.utc)), book=book, processor=processor)
    process_parsed_event(event=_heartbeat_event(datetime(2024, 1, 1, 0, 14, 55, tzinfo=timezone.utc)), book=book, processor=processor)
    observations = process_parsed_event(
        event=_update_event(2, datetime(2024, 1, 1, 0, 15, 1, tzinfo=timezone.utc), "bid", 100.5, 1.0),
        book=book,
        processor=processor,
    )
    assert len(observations) == 1
    obs = observations[0]
    assert obs.timestamp_utc == datetime(2024, 1, 1, 0, 15, tzinfo=timezone.utc)
    assert obs.eligible is True
    assert obs.best_bid == 100.0 and obs.best_ask == 101.0


def test_boundary_crossed_with_stale_heartbeat_is_ineligible():
    book = Level2OrderBook(product_id="BTC-USD")
    processor = BoundaryEventProcessor(product_id="BTC-USD", heartbeat_timeout_seconds=5)
    process_parsed_event(event=_snapshot_event(1, datetime(2024, 1, 1, 0, 14, 50, tzinfo=timezone.utc)), book=book, processor=processor)
    process_parsed_event(event=_heartbeat_event(datetime(2024, 1, 1, 0, 14, 50, tzinfo=timezone.utc)), book=book, processor=processor)
    observations = process_parsed_event(
        event=_update_event(2, datetime(2024, 1, 1, 0, 15, 1, tzinfo=timezone.utc), "bid", 100.5, 1.0),
        book=book,
        processor=processor,
    )
    assert len(observations) == 1
    assert observations[0].eligible is False


def test_sequence_gap_makes_subsequent_boundary_ineligible_until_resync():
    book = Level2OrderBook(product_id="BTC-USD")
    processor = BoundaryEventProcessor(product_id="BTC-USD", heartbeat_timeout_seconds=30)
    process_parsed_event(event=_snapshot_event(1, datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)), book=book, processor=processor)
    process_parsed_event(event=_heartbeat_event(datetime(2024, 1, 1, 0, 0, 1, tzinfo=timezone.utc)), book=book, processor=processor)
    # Simulate a dropped message: sequence jumps from 1 to 5.
    gap_observations = process_parsed_event(
        event=_update_event(5, datetime(2024, 1, 1, 0, 5, 0, tzinfo=timezone.utc), "bid", 100.5, 1.0),
        book=book,
        processor=processor,
    )
    assert gap_observations == []
    assert not book.is_synced()
    assert processor.gap_since_sync is True

    # Real Coinbase sequence recovery only happens via a new connection epoch
    # (reconnect + fresh subscribe), which restarts the per-connection
    # sequence counter -- an in-band snapshot cannot silently "skip ahead".
    book.reset_for_new_connection()
    process_parsed_event(event=_snapshot_event(1, datetime(2024, 1, 1, 0, 10, 0, tzinfo=timezone.utc)), book=book, processor=processor)
    process_parsed_event(event=_heartbeat_event(datetime(2024, 1, 1, 0, 14, 59, tzinfo=timezone.utc)), book=book, processor=processor)
    observations = process_parsed_event(
        event=_update_event(2, datetime(2024, 1, 1, 0, 15, 0, tzinfo=timezone.utc), "bid", 100.5, 1.0),
        book=book,
        processor=processor,
    )
    assert len(observations) == 1
    assert observations[0].eligible is True


def test_replay_events_is_deterministic_regardless_of_input_order():
    events = [
        _snapshot_event(1, datetime(2024, 1, 1, 0, 14, 50, tzinfo=timezone.utc)),
        _heartbeat_event(datetime(2024, 1, 1, 0, 14, 55, tzinfo=timezone.utc)),
        _update_event(2, datetime(2024, 1, 1, 0, 15, 1, tzinfo=timezone.utc), "bid", 100.5, 1.0),
        _heartbeat_event(datetime(2024, 1, 1, 0, 15, 3, tzinfo=timezone.utc)),
        _update_event(3, datetime(2024, 1, 1, 0, 29, 59, tzinfo=timezone.utc), "ask", 101.5, 2.0),
        _heartbeat_event(datetime(2024, 1, 1, 0, 29, 59, tzinfo=timezone.utc)),
    ]
    baseline = replay_events(events, product_id="BTC-USD", heartbeat_timeout_seconds=30)

    shuffled = list(events)
    rng = random.Random(1234)
    for _ in range(5):
        rng.shuffle(shuffled)
        result = replay_events(shuffled, product_id="BTC-USD", heartbeat_timeout_seconds=30)
        assert result == baseline


def test_replay_events_matches_incremental_live_processing():
    live_book = Level2OrderBook(product_id="BTC-USD")
    live_processor = BoundaryEventProcessor(product_id="BTC-USD", heartbeat_timeout_seconds=30)
    events = [
        _snapshot_event(1, datetime(2024, 1, 1, 0, 14, 50, tzinfo=timezone.utc)),
        _heartbeat_event(datetime(2024, 1, 1, 0, 14, 55, tzinfo=timezone.utc)),
        _update_event(2, datetime(2024, 1, 1, 0, 15, 1, tzinfo=timezone.utc), "bid", 100.5, 1.0),
        _heartbeat_event(datetime(2024, 1, 1, 0, 15, 3, tzinfo=timezone.utc)),
        _update_event(3, datetime(2024, 1, 1, 0, 30, 2, tzinfo=timezone.utc), "ask", 101.5, 3.0),
    ]
    live_observations = []
    for event in events:
        live_observations.extend(process_parsed_event(event=event, book=live_book, processor=live_processor))

    replayed = replay_events(events, product_id="BTC-USD", heartbeat_timeout_seconds=30)
    assert replayed == live_observations


def test_replay_events_retains_every_intermediate_boundary_on_a_multi_boundary_jump():
    """If the event stream jumps straight from one quarter-hour to a point
    several quarter-hours later (e.g. a quiet period with no updates), every
    boundary crossed in between must still get its own observation -- none
    may be silently skipped."""
    events = [
        _snapshot_event(1, datetime(2024, 1, 1, 0, 0, 5, tzinfo=timezone.utc)),
        _heartbeat_event(datetime(2024, 1, 1, 0, 0, 10, tzinfo=timezone.utc)),
        # A full quiet hour later: 4 quarter-hour boundaries (00:15, 00:30,
        # 00:45, 01:00) are crossed by this single event's event time.
        _update_event(2, datetime(2024, 1, 1, 1, 0, 1, tzinfo=timezone.utc), "bid", 100.5, 1.0),
    ]
    replayed = replay_events(events, product_id="BTC-USD", heartbeat_timeout_seconds=3600)
    boundaries = [obs.timestamp_utc for obs in replayed]
    assert boundaries == [
        datetime(2024, 1, 1, 0, 15, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 0, 30, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 0, 45, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
    ]
    # Every one of those boundaries is eligible: synced, no gap, and the
    # single early heartbeat is still within the (generous) timeout of each.
    assert all(obs.eligible for obs in replayed)


def test_replay_events_handles_a_late_arriving_event_recorded_after_a_later_one():
    """An event recorded to the raw log *after* a chronologically later one
    (e.g. redelivered/out-of-order on the wire) must still be placed at its
    own true event time during replay, contributing its own boundary
    crossing rather than being dropped or misordered."""
    early = _snapshot_event(1, datetime(2024, 1, 1, 0, 14, 50, tzinfo=timezone.utc))
    late_arrival = _heartbeat_event(datetime(2024, 1, 1, 0, 14, 52, tzinfo=timezone.utc))
    later = _update_event(2, datetime(2024, 1, 1, 0, 15, 5, tzinfo=timezone.utc), "bid", 100.5, 1.0)
    # `late_arrival`'s event time precedes `later`'s, but it is appended to
    # the recorded stream *after* `later` -- simulating network reordering.
    recorded_order = [early, later, late_arrival]

    replayed = replay_events(recorded_order, product_id="BTC-USD", heartbeat_timeout_seconds=30)
    chronological_order = [early, late_arrival, later]
    replayed_in_true_order = replay_events(chronological_order, product_id="BTC-USD", heartbeat_timeout_seconds=30)

    assert replayed == replayed_in_true_order
    assert len(replayed) == 1
    assert replayed[0].eligible is True  # the late-arriving heartbeat still counts toward the 00:15 boundary's health


def test_replay_events_treats_stale_duplicate_sequence_as_non_fatal():
    """A duplicate/redelivered sequence number recorded in a replayed segment
    must not be treated as a gap: the book stays synced and later events
    keep producing eligible observations."""
    events = [
        _snapshot_event(1, datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        _heartbeat_event(datetime(2024, 1, 1, 0, 0, 1, tzinfo=timezone.utc)),
        _update_event(2, datetime(2024, 1, 1, 0, 0, 2, tzinfo=timezone.utc), "bid", 100.5, 1.0),
        # A redelivered duplicate of the already-applied update above.
        _update_event(2, datetime(2024, 1, 1, 0, 0, 3, tzinfo=timezone.utc), "bid", 999.0, 1.0),
        _heartbeat_event(datetime(2024, 1, 1, 0, 15, 1, tzinfo=timezone.utc)),
    ]
    replayed = replay_events(events, product_id="BTC-USD", heartbeat_timeout_seconds=3600)
    assert len(replayed) == 1
    assert replayed[0].eligible is True
    # The duplicate's bogus price must never have been applied.
    assert replayed[0].best_bid == 100.5
    """An event recorded to the raw log *after* a chronologically later one
    (e.g. redelivered/out-of-order on the wire) must still be placed at its
    own true event time during replay, contributing its own boundary
    crossing rather than being dropped or misordered."""
    early = _snapshot_event(1, datetime(2024, 1, 1, 0, 14, 50, tzinfo=timezone.utc))
    late_arrival = _heartbeat_event(datetime(2024, 1, 1, 0, 14, 52, tzinfo=timezone.utc))
    later = _update_event(2, datetime(2024, 1, 1, 0, 15, 5, tzinfo=timezone.utc), "bid", 100.5, 1.0)
    # `late_arrival`'s event time precedes `later`'s, but it is appended to
    # the recorded stream *after* `later` -- simulating network reordering.
    recorded_order = [early, later, late_arrival]

    replayed = replay_events(recorded_order, product_id="BTC-USD", heartbeat_timeout_seconds=30)
    chronological_order = [early, late_arrival, later]
    replayed_in_true_order = replay_events(chronological_order, product_id="BTC-USD", heartbeat_timeout_seconds=30)

    assert replayed == replayed_in_true_order
    assert len(replayed) == 1
    assert replayed[0].eligible is True  # the late-arriving heartbeat still counts toward the 00:15 boundary's health


def test_replay_events_treats_stale_duplicate_sequence_as_non_fatal():
    """A duplicate/redelivered sequence number recorded in a replayed segment
    must not be treated as a gap: the book stays synced and later events
    keep producing eligible observations."""
    events = [
        _snapshot_event(1, datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        _heartbeat_event(datetime(2024, 1, 1, 0, 0, 1, tzinfo=timezone.utc)),
        _update_event(2, datetime(2024, 1, 1, 0, 0, 2, tzinfo=timezone.utc), "bid", 100.5, 1.0),
        # A redelivered duplicate of the already-applied update above.
        _update_event(2, datetime(2024, 1, 1, 0, 0, 3, tzinfo=timezone.utc), "bid", 999.0, 1.0),
        _heartbeat_event(datetime(2024, 1, 1, 0, 15, 1, tzinfo=timezone.utc)),
    ]
    replayed = replay_events(events, product_id="BTC-USD", heartbeat_timeout_seconds=3600)
