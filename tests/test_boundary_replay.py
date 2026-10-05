from __future__ import annotations

import random
import json
from datetime import datetime, timedelta, timezone

import pytest

from btc_quarter_hour_engine.market_data.boundary_observations import (
    derive_quarter_hour_observation,
    is_exact_quarter_hour_boundary,
)
from btc_quarter_hour_engine.market_data.order_book import Level2OrderBook
from btc_quarter_hour_engine.market_data.replay import (
    BoundaryEventProcessor,
    CanonicalReplayAccumulator,
    floor_to_quarter_hour,
    process_parsed_event,
    replay_recorded_frames,
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


def test_l2_mutation_path_does_not_infer_gaps_from_l2_only_sequences():
    book = Level2OrderBook(product_id="BTC-USD")
    processor = BoundaryEventProcessor(product_id="BTC-USD", heartbeat_timeout_seconds=30)
    process_parsed_event(event=_snapshot_event(1, datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)), book=book, processor=processor)
    process_parsed_event(
        event=_update_event(5, datetime(2024, 1, 1, 0, 5, 0, tzinfo=timezone.utc), "bid", 100.5, 1.0),
        book=book,
        processor=processor,
    )
    assert book.is_synced()
    assert book.last_sequence_num == 5
    assert processor.gap_since_sync is False

def test_failed_l2_batch_does_not_advance_book_sequence_provenance():
    book = Level2OrderBook(product_id="BTC-USD")
    processor = BoundaryEventProcessor(product_id="BTC-USD", heartbeat_timeout_seconds=30)
    process_parsed_event(
        event=_snapshot_event(1, datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
        book=book,
        processor=processor,
    )

    process_parsed_event(
        event=_update_event(2, datetime(2024, 1, 1, 0, 0, 1, tzinfo=timezone.utc), "bid", 102.0, 1.0),
        book=book,
        processor=processor,
    )

    assert book.state.value == "INVALID"
    assert book.last_sequence_num == 1


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
    assert replayed[0].best_bid == 100.5


def _at(second: str) -> datetime:
    return datetime.fromisoformat(f"2024-01-01T13:{second}+00:00")


def _utc(timestamp: str) -> datetime:
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def _frame(channel, timestamp, sequence, *, updates=None, event_type="update", counter=1):
    if channel == "l2_data":
        events = [{"type": event_type, "product_id": "BTC-USD", "updates": [
            {"side": side, "event_time": at, "price_level": str(price), "new_quantity": str(size)}
            for side, at, price, size in updates
        ]}]
    else:
        events = [{"current_time": timestamp, "heartbeat_counter": str(counter)}]
    return json.dumps({"channel": channel, "timestamp": timestamp, "sequence_num": sequence, "events": events}).encode()


def _subscription(sequence, timestamp="2024-01-01T13:14:56Z"):
    return json.dumps({
        "channel": "subscriptions",
        "timestamp": timestamp,
        "sequence_num": sequence,
        "events": [{"subscriptions": {"level2": ["BTC-USD"]}}],
    }).encode()


def _snapshot(timestamp="2024-01-01T13:14:50Z"):
    return _frame("l2_data", timestamp, 1, event_type="snapshot", updates=[
        ("bid", "1970-01-01T00:00:00Z", 100, 1),
        ("offer", "1970-01-01T00:00:00Z", 101, 2),
    ])


def _update(sequence, *updates):
    return _frame("l2_data", "2024-01-01T13:15:01Z", sequence, updates=updates)


def _heartbeat(time="2024-01-01T13:14:55Z", sequence=2, counter=1):
    return _frame("heartbeats", time, sequence, counter=counter)


def _replay(raw_frames, **kwargs):
    return replay_recorded_frames(
        ((raw, "conn-1", i) for i, raw in enumerate(raw_frames)),
        product_id="BTC-USD", session_id="session-1", heartbeat_timeout_seconds=30, **kwargs,
    )


def test_future_envelope_blocks_later_sequence_even_if_later_sequence_has_preboundary_event_time():
    ledger = _replay([
        _snapshot(), _heartbeat(),
        _update(3, ("offer", "2024-01-01T13:15:00.100Z", 101.5, 1)),
        _update(4, ("bid", "2024-01-01T13:14:59.950Z", 100.5, 1)),
    ])
    assert len(ledger) == 1
    row = ledger[0]
    assert row.boundary_time_utc == _at("15:00")
    assert (row.best_bid, row.best_ask, row.midpoint) == (100, 101, 100.5)
    assert (row.source_sequence_num, row.source_state_time_utc) == (1, _at("14:50"))
    assert (row.connection_id, row.session_id) == ("conn-1", "session-1")
    assert row.canonical_target_eligible and row.eligibility_reason == "eligible"
    assert row.derived_at_utc == _at("15:00.100000")


def test_straddling_envelope_is_excluded_atomically_from_boundary():
    ledger = _replay([
        _snapshot(), _heartbeat(),
        _update(3,
            ("bid", "2024-01-01T13:14:59.900Z", 100.5, 1),
            ("offer", "2024-01-01T13:15:00.100Z", 101.5, 1)),
    ])
    row = ledger[0]
    assert row.eligible and row.book_synced
    assert (row.best_bid, row.best_ask, row.midpoint) == (100, 101, 100.5)
    assert (row.source_sequence_num, row.source_state_time_utc) == (1, _at("14:50"))

def test_same_timestamp_envelope_mutations_replay_as_one_atomic_batch():
    ledger = _replay([
        _snapshot(), _heartbeat(),
        _update(3,
            ("bid", "2024-01-01T13:14:59.900Z", 102, 1),
            ("offer", "2024-01-01T13:14:59.900Z", 101, 0),
            ("offer", "2024-01-01T13:14:59.900Z", 103, 1)),
    ], session_completed_at_utc=_at("15:00"))
    row = ledger[0]
    assert row.eligible
    assert (row.best_bid, row.best_ask, row.midpoint) == (102.0, 103.0, 102.5)
    assert row.eligibility_reason == "eligible"


def test_straddling_envelope_is_excluded_atomically_and_applied_at_later_boundary():
    ledger = _replay([
        _snapshot(), _heartbeat(),
        _update(3,
            ("bid", "2024-01-01T13:14:59.900Z", 102, 1),
            ("offer", "2024-01-01T13:15:00.100Z", 101, 0),
            ("offer", "2024-01-01T13:15:00.100Z", 103, 1)),
        _heartbeat("2024-01-01T13:29:59Z", sequence=4),
    ], session_completed_at_utc=_at("30:00"))
    before, after = ledger
    assert before.eligible and before.book_synced
    assert (before.best_bid, before.best_ask) == (100, 101)
    assert (before.source_sequence_num, before.source_state_time_utc) == (1, _at("14:50"))
    assert after.eligible and after.book_synced
    assert (after.best_bid, after.best_ask, after.midpoint) == (102, 103, 102.5)
    assert (after.source_sequence_num, after.source_state_time_utc) == (3, _at("15:00.100000"))


def test_cross_envelope_timestamp_inversion_preserves_delivery_order():
    ledger = _replay([
        _snapshot(), _heartbeat(),
        _update(3,
            ("offer", "2024-01-01T13:14:59.900Z", 101, 0),
            ("offer", "2024-01-01T13:14:59.900Z", 102, 1)),
        _update(4,
            ("offer", "2024-01-01T13:14:59.850Z", 102, 0),
            ("offer", "2024-01-01T13:14:59.850Z", 103, 1)),
    ], session_completed_at_utc=_at("15:00"))
    row = ledger[0]
    assert row.eligible
    assert (row.best_bid, row.best_ask, row.midpoint) == (100, 103, 101.5)
    assert (row.source_sequence_num, row.source_state_time_utc) == (4, _at("14:59.850000"))


def test_future_envelope_blocks_later_preboundary_envelope_until_later_boundary():
    ledger = _replay([
        _snapshot(), _heartbeat(),
        _update(3,
            ("offer", "2024-01-01T13:15:00.100Z", 101, 0),
            ("offer", "2024-01-01T13:15:00.100Z", 102, 1)),
        _update(4,
            ("offer", "2024-01-01T13:14:59.950Z", 102, 0),
            ("offer", "2024-01-01T13:14:59.950Z", 103, 1)),
        _heartbeat("2024-01-01T13:29:59Z", sequence=5),
    ], session_completed_at_utc=_at("30:00"))
    at_1500, at_1515 = ledger
    assert at_1500.eligible
    assert (at_1500.best_bid, at_1500.best_ask) == (100, 101)
    assert (at_1500.source_sequence_num, at_1500.source_state_time_utc) == (1, _at("14:50"))
    assert at_1515.eligible
    assert (at_1515.best_bid, at_1515.best_ask) == (100, 103)
    assert (at_1515.source_sequence_num, at_1515.source_state_time_utc) == (4, _at("14:59.950000"))


def test_preboundary_multi_update_envelope_is_applied_as_one_atomic_batch():
    ledger = _replay([
        _snapshot(), _heartbeat(),
        _update(3,
            ("bid", "2024-01-01T13:14:58.900Z", 102, 1),
            ("offer", "2024-01-01T13:14:59.100Z", 101, 0),
            ("offer", "2024-01-01T13:14:59.200Z", 103, 1)),
    ], session_completed_at_utc=_at("15:00"))
    row = ledger[0]
    assert row.eligible and row.book_synced
    assert (row.best_bid, row.best_ask, row.midpoint) == (102, 103, 102.5)
    assert (row.source_sequence_num, row.source_state_time_utc) == (3, _at("14:59.200000"))


def test_heartbeat_remains_independent_of_future_l2_prefix_blocker():
    ledger = _replay([
        _snapshot(), _heartbeat("2024-01-01T13:14:00Z"),
        _update(3, ("bid", "2024-01-01T13:15:00.100Z", 100.5, 1)),
        _heartbeat("2024-01-01T13:14:59.500Z", sequence=4),
    ], session_completed_at_utc=_at("15:00"))
    row = ledger[0]
    assert row.eligible and row.book_synced
    assert (row.best_bid, row.best_ask) == (100, 101)
    assert row.source_sequence_num == 1


def test_real_shaped_1715_boundary_uses_last_committed_complete_envelope():
    ledger = _replay([
        _frame("l2_data", "2026-10-05T17:14:50Z", 2534, event_type="snapshot", updates=[
            ("bid", "1970-01-01T00:00:00Z", 85255.47, 1),
            ("offer", "1970-01-01T00:00:00Z", 85255.49, 1),
        ]),
        _heartbeat("2026-10-05T17:14:55Z", sequence=2535),
        _frame("l2_data", "2026-10-05T17:14:59.977369Z", 2536, updates=[
            ("bid", "2026-10-05T17:14:59.977369Z", 85255.48, 1),
        ]),
        _frame("l2_data", "2026-10-05T17:15:00.045315Z", 2537, updates=[
            ("offer", "2026-10-05T17:15:00.045315Z", 85255.50, 1),
        ]),
    ], session_completed_at_utc=_utc("2026-10-05T17:15:00Z"))
    row = ledger[0]
    assert row.eligible and row.book_synced
    assert (row.best_bid, row.best_ask) == (85255.48, 85255.49)
    assert row.midpoint == pytest.approx(85255.485)
    assert row.source_sequence_num == 2536
    assert row.source_state_time_utc == _utc("2026-10-05T17:14:59.977369Z")


def test_real_shaped_1730_boundary_uses_last_committed_complete_envelope():
    ledger = _replay([
        _frame("l2_data", "2026-10-05T17:29:50Z", 19647, event_type="snapshot", updates=[
            ("bid", "1970-01-01T00:00:00Z", 85375.15, 1),
            ("offer", "1970-01-01T00:00:00Z", 85375.17, 1),
        ]),
        _heartbeat("2026-10-05T17:29:55Z", sequence=19648),
        _frame("l2_data", "2026-10-05T17:29:59.966539Z", 19649, updates=[
            ("bid", "2026-10-05T17:29:59.966539Z", 85375.16, 1),
        ]),
        _frame("l2_data", "2026-10-05T17:30:00.028495Z", 19650, updates=[
            ("offer", "2026-10-05T17:30:00.028495Z", 85375.18, 1),
        ]),
    ], session_completed_at_utc=_utc("2026-10-05T17:30:00Z"))
    row = ledger[0]
    assert row.eligible and row.book_synced
    assert (row.best_bid, row.best_ask) == (85375.16, 85375.17)
    assert row.midpoint == pytest.approx(85375.165)
    assert row.source_sequence_num == 19649
    assert row.source_state_time_utc == _utc("2026-10-05T17:29:59.966539Z")


def test_exact_boundary_mutation_inclusive_and_one_microsecond_future_exclusive():
    ledger = _replay([
        _snapshot(), _heartbeat(),
        _update(3, ("bid", "2024-01-01T13:15:00Z", 100.5, 1)),
        _update(4, ("offer", "2024-01-01T13:15:00.000001Z", 101.5, 1)),
    ])
    assert (ledger[0].best_bid, ledger[0].best_ask) == (100.5, 101)


def test_valid_envelope_transient_cross_resolves_before_boundary_with_provenance():
    ledger = _replay([
        _frame("l2_data", "2024-01-01T13:14:50Z", 1, event_type="snapshot", updates=[
            ("bid", "1970-01-01T00:00:00Z", 85241.36, 1),
            ("offer", "1970-01-01T00:00:00Z", 85241.78, 1),
        ]),
        _heartbeat("2024-01-01T13:14:55Z", sequence=2),
        _update(3,
            ("bid", "2024-01-01T13:14:58.900Z", 85242.00, 1),
            ("offer", "2024-01-01T13:14:59.100Z", 85241.78, 0),
            ("bid", "2024-01-01T13:14:59.200Z", 85242.00, 0),
            ("offer", "2024-01-01T13:14:59.300Z", 85241.78, 1)),
    ], session_completed_at_utc=_at("15:00"))
    row = ledger[0]
    assert row.eligible and row.book_synced
    assert row.eligibility_reason == "eligible"
    assert (row.best_bid, row.best_ask, row.midpoint) == (85241.36, 85241.78, 85241.57)
    assert row.source_state_time_utc == _at("14:59.300000")
    assert row.source_sequence_num == 3


def test_later_boundary_applies_previously_straddling_envelope_atomically():
    ledger = _replay([
        _snapshot(), _heartbeat(),
        _update(3,
            ("bid", "2024-01-01T13:14:59.900Z", 102, 1),
            ("offer", "2024-01-01T13:15:00.100Z", 101, 0),
            ("offer", "2024-01-01T13:15:00.100Z", 103, 1)),
        _heartbeat("2024-01-01T13:29:59Z", sequence=4),
    ], session_completed_at_utc=_at("30:00"))
    assert len(ledger) == 2
    prior, recovered = ledger
    assert prior.eligible and prior.book_synced
    assert (prior.best_bid, prior.best_ask) == (100, 101)
    assert recovered.eligible and recovered.book_synced
    assert (recovered.best_bid, recovered.best_ask, recovered.midpoint) == (102, 103, 102.5)
    assert recovered.source_state_time_utc == _at("15:00.100000")
    assert recovered.source_sequence_num == 3


def test_recovered_book_still_requires_fresh_heartbeat():
    ledger = _replay([
        _snapshot(), _heartbeat(),
        _update(3,
            ("bid", "2024-01-01T13:14:59.900Z", 102, 1),
            ("offer", "2024-01-01T13:15:00.100Z", 101, 0),
            ("offer", "2024-01-01T13:15:00.100Z", 103, 1)),
    ], session_completed_at_utc=_at("30:00"))
    assert len(ledger) == 2
    assert ledger[0].eligible and ledger[0].book_synced
    assert ledger[1].book_synced
    assert (ledger[1].best_bid, ledger[1].best_ask) == (102, 103)
    assert not ledger[1].eligible
    assert ledger[1].eligibility_reason == "heartbeat_stale"


def test_truly_crossed_complete_source_envelope_permanently_invalidates_epoch():
    ledger = _replay([
        _snapshot(), _heartbeat(),
        _update(3, ("bid", "2024-01-01T13:14:59Z", 102, 1)),
        _update(4, ("offer", "2024-01-01T13:15:01Z", 103, 1)),
        _heartbeat("2024-01-01T13:29:59Z", sequence=5),
    ], session_completed_at_utc=_at("30:00"))
    assert len(ledger) == 2
    assert ledger[0].eligibility_reason == "crossed_book"
    assert ledger[1].eligibility_reason == "crossed_book"
    assert not ledger[1].book_synced and not ledger[1].eligible


def test_future_complete_envelope_invalidity_does_not_poison_prior_boundary():
    ledger = _replay([
        _snapshot(), _heartbeat(),
        _update(3,
            ("bid", "2024-01-01T13:14:59.900Z", 100.5, 1),
            ("bid", "2024-01-01T13:15:00.100Z", 102, 1)),
    ], session_completed_at_utc=_at("15:00"))
    row = ledger[0]
    assert row.eligible and row.book_synced
    assert (row.best_bid, row.best_ask) == (100, 101)
    assert row.source_state_time_utc == _at("14:50")
    assert row.source_sequence_num == 1


def test_ineligible_ledger_reasons_and_no_price_freshness():
    start, end = _at("15:00"), _at("30:00")
    assert [r.eligibility_reason for r in _replay([], session_started_at_utc=start, session_completed_at_utc=end)] == [
        "no_synced_snapshot", "no_synced_snapshot",
    ]
    gap = _replay([
        _snapshot(), _heartbeat(),
        _update(5, ("bid", "2024-01-01T13:14:59Z", 100.5, 1)),
    ], session_completed_at_utc=start)
    assert gap[0].eligibility_reason == "sequence_gap"
    assert gap[0].best_bid is None
    crossed = _replay([
        _snapshot(), _heartbeat(),
        _update(3, ("bid", "2024-01-01T13:14:59Z", 102, 1)),
    ], session_completed_at_utc=start)
    assert crossed[0].eligibility_reason == "crossed_book"
    assert not crossed[0].book_synced
    missing_ask = _replay([
        _frame("l2_data", "2024-01-01T13:14:50Z", 1, event_type="snapshot",
               updates=[("bid", "1970-01-01T00:00:00Z", 100, 1)]),
        _heartbeat(),
    ], session_completed_at_utc=start)
    assert missing_ask[0].eligibility_reason == "missing_ask"
    stale = _replay([_snapshot(), _heartbeat("2024-01-01T13:14:00Z")], session_completed_at_utc=start)
    assert stale[0].eligibility_reason == "heartbeat_stale"
    healthy = _replay(
        [_snapshot(), _heartbeat("2024-01-01T13:14:59Z")],
        session_completed_at_utc=start,
    )
    assert healthy[0].eligible
    disconnected = _replay(
        [_snapshot(), _heartbeat()],
        connections=[{"connection_id": "conn-1", "connected_at_utc": _at("14:50"),
                      "source_disconnected_at_utc": _at("14:59")}],
        session_completed_at_utc=start,
    )
    assert disconnected[0].eligibility_reason == "connection_unhealthy"


def test_future_heartbeat_cannot_validate_prior_boundary():
    ledger = _replay(
        [_snapshot(), _heartbeat("2024-01-01T13:15:01Z")],
        session_completed_at_utc=_at("15:01"),
    )
    assert ledger[0].eligibility_reason == "heartbeat_stale"


def test_old_price_remains_valid_with_fresh_heartbeat_and_stale_sequence_is_ignored():
    ledger = _replay([
        _snapshot("2024-01-01T13:00:01Z"),
        _update(2, ("bid", "2024-01-01T13:00:02Z", 100.5, 1)),
        _update(2, ("bid", "2024-01-01T13:14:59Z", 999, 1)),
        _heartbeat("2024-01-01T13:14:59Z", sequence=3),
    ], session_completed_at_utc=_at("15:00"))
    assert ledger[0].eligible
    assert ledger[0].best_bid == 100.5
    assert ledger[0].source_state_time_utc == _at("00:02")


def test_sequence_integrity_uses_arrival_not_sorted_update_time():
    ledger = _replay([
        _snapshot(), _heartbeat(),
        _update(3, ("offer", "2024-01-01T13:15:00.100Z", 101.5, 1)),
        _update(5, ("bid", "2024-01-01T13:14:59.950Z", 100.5, 1)),
    ])
    assert ledger[0].eligibility_reason == "sequence_gap"
    assert ledger[0].best_bid is None


def test_final_replay_tracks_subscriptions_and_heartbeats_in_envelope_sequence():
    frames = [
        _frame("l2_data", "2024-01-01T13:14:50Z", 0, event_type="snapshot", updates=[
            ("bid", "1970-01-01T00:00:00Z", 100, 1),
            ("offer", "1970-01-01T00:00:00Z", 101, 2),
        ]),
        _update(1, ("bid", "2024-01-01T13:14:54Z", 100.5, 1)),
        _subscription(2),
        _frame("heartbeats", "2024-01-01T13:14:59Z", 3, counter=1),
        _update(4, ("offer", "2024-01-01T13:14:59Z", 101.5, 1)),
    ]
    ledger = _replay(frames, session_completed_at_utc=_at("15:00"))
    assert len(ledger) == 1
    assert ledger[0].eligible
    assert ledger[0].best_bid == 100.5


def test_final_replay_detects_gap_across_non_l2_envelopes():
    frames = [
        _frame("l2_data", "2024-01-01T13:14:50Z", 0, event_type="snapshot", updates=[
            ("bid", "1970-01-01T00:00:00Z", 100, 1),
            ("offer", "1970-01-01T00:00:00Z", 101, 2),
        ]),
        _frame("heartbeats", "2024-01-01T13:14:55Z", 1, counter=1),
        _update(3, ("bid", "2024-01-01T13:14:59Z", 100.5, 1)),
    ]
    ledger = _replay(frames, session_completed_at_utc=_at("15:00"))
    assert ledger[0].eligibility_reason == "sequence_gap"
    assert not ledger[0].eligible
    assert ledger[0].best_bid is None


def test_delivery_order_wins_over_nonmonotonic_frame_indices():
    frames = [(_snapshot(), "conn-1", 30), (_heartbeat(), "conn-1", 20),
              (_update(3, ("bid", "2024-01-01T13:15:00Z", 100.5, 1)), "conn-1", 10),
              (_update(4, ("offer", "2024-01-01T13:15:00.100Z", 101.5, 1)), "conn-1", 0)]
    row = replay_recorded_frames(
        frames, product_id="BTC-USD", session_id="session-1", heartbeat_timeout_seconds=30,
    )[0]
    assert row.eligible and row.best_bid == 100.5 and row.best_ask == 101


def test_reconnect_does_not_inherit_previous_book_or_heartbeat():
    frames = [(_snapshot(), "conn-1", 0), (_heartbeat(), "conn-1", 1),
              (_heartbeat("2024-01-01T13:15:00Z"), "conn-2", 0)]
    ledger = replay_recorded_frames(
        frames, product_id="BTC-USD", session_id="session-1", heartbeat_timeout_seconds=30,
        connections=[
            {"connection_id": "conn-1", "connected_at_utc": _at("14:50"), "disconnected_at_utc": _at("14:59")},
            {"connection_id": "conn-2", "connected_at_utc": _at("15:00")},
        ],
    )
    assert ledger[0].connection_id == "conn-2"
    assert ledger[0].eligibility_reason == "no_synced_snapshot"


def test_canonical_timestamps_reject_naive_datetimes():
    naive = datetime(2024, 1, 1, 13, 15)
    for helper in (is_exact_quarter_hour_boundary, floor_to_quarter_hour):
        with pytest.raises(ValueError, match="timezone-aware"):
            helper(naive)
    with pytest.raises(ValueError, match="timezone-aware"):
        derive_quarter_hour_observation(book=Level2OrderBook(product_id="BTC-USD"), timestamp_utc=naive)
    with pytest.raises(ValueError, match="timezone-aware"):
        _replay([], session_started_at_utc=naive)


def test_bounded_streaming_accumulator_matches_offline_replay_for_scientific_edge_cases():
    base = [
        _snapshot(),
        _heartbeat(),
        _update(3, ("offer", "2024-01-01T13:15:00.100Z", 101.5, 1)),
        _update(4, ("bid", "2024-01-01T13:14:59.950Z", 100.5, 1)),
    ]
    straddling = [
        _snapshot(),
        _heartbeat(),
        _update(
            3,
            ("bid", "2024-01-01T13:14:59.900Z", 100.5, 1),
            ("offer", "2024-01-01T13:15:00.100Z", 101.5, 1),
        ),
    ]
    exact_boundary = [
        _snapshot(),
        _heartbeat(),
        _update(3, ("bid", "2024-01-01T13:15:00Z", 100.5, 1)),
        _update(4, ("offer", "2024-01-01T13:15:00.000001Z", 101.5, 1)),
    ]
    integrity_cases = [
        [_snapshot(), _heartbeat(), _update(4, ("bid", "2024-01-01T13:14:59Z", 100.5, 1))],
        [_snapshot(), _heartbeat(), b"malformed source frame"],
        [_snapshot(), _heartbeat("2024-01-01T13:14:00Z"), _heartbeat("2024-01-01T13:15:01Z", sequence=3)],
    ]
    reconnect = [
        (_snapshot(), "conn-1"),
        (_heartbeat(), "conn-1"),
        (_snapshot("2024-01-01T13:15:05Z"), "conn-2"),
        (_heartbeat("2024-01-01T13:15:10Z", sequence=2), "conn-2"),
    ]
    scenarios = [
        [(raw, "conn-1") for raw in base],
        [(raw, "conn-1") for raw in straddling],
        [(raw, "conn-1") for raw in exact_boundary],
        *[[(raw, "conn-1") for raw in case] for case in integrity_cases],
        reconnect,
    ]

    for scenario in scenarios:
        recorded = [
            (
                raw if isinstance(raw, bytes) else raw.encode(),
                connection_id,
                frame_index,
                _utc("2024-01-01T13:14:50Z") + timedelta(seconds=frame_index),
            )
            for frame_index, (raw, connection_id) in enumerate(scenario)
        ]
        connections = [{"connection_id": "conn-1"}, {"connection_id": "conn-2"}]
        accumulator = CanonicalReplayAccumulator(
            frames_factory=lambda rows=recorded: iter(rows),
            product_id="BTC-USD",
            session_id="session-1",
            heartbeat_timeout_seconds=30,
            connections_factory=lambda records=connections: records,
        )
        for raw, connection_id, frame_index, _ingest_time in recorded:
            accumulator.consume(raw, connection_id, frame_index)
        streamed = accumulator.finalize(
            session_started_at_utc=None,
            session_completed_at_utc=None,
            derived_at_utc=_utc("2024-01-01T14:00:00Z"),
        )
        offline = replay_recorded_frames(
            [(raw, connection_id, frame_index) for raw, connection_id, frame_index, _ in recorded],
            product_id="BTC-USD",
            session_id="session-1",
            heartbeat_timeout_seconds=30,
            connections=connections,
            derived_at_utc=_utc("2024-01-01T14:00:00Z"),
        )
        assert streamed == offline
