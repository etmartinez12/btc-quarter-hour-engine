from __future__ import annotations

import pytest

from btc_quarter_hour_engine.market_data.order_book import (
    Level2OrderBook,
    OrderBookState,
)
from tests.reference_order_book import CopyOnWriteReferenceBook


def _synced_book() -> Level2OrderBook:
    book = Level2OrderBook(product_id="BTC-USD")
    book.apply_snapshot(
        product_id="BTC-USD",
        levels={
            "bid": [{"price": 100.0, "quantity": 1.0}],
            "ask": [{"price": 101.0, "quantity": 2.0}],
        },
    )
    return book


def test_snapshot_sync_and_bbo_derivation():
    book = _synced_book()
    assert book.state == OrderBookState.SYNCED
    assert book.best_bid == 100.0
    assert book.best_ask == 101.0
    assert book.best_bid_size == 1.0
    assert book.best_ask_size == 2.0
    assert book.spread == 1.0
    assert book.midpoint == 100.5
    assert book.is_synced()


def test_top_of_book_matches_properties_for_multiple_levels():
    book = Level2OrderBook(product_id="BTC-USD")
    book.apply_snapshot(
        product_id="BTC-USD",
        levels={
            "bid": [
                {"price": 100.0, "quantity": 1.0},
                {"price": 102.0, "quantity": 3.0},
                {"price": 101.0, "quantity": 2.0},
            ],
            "ask": [
                {"price": 105.0, "quantity": 4.0},
                {"price": 103.0, "quantity": 5.0},
                {"price": 104.0, "quantity": 6.0},
            ],
        },
    )

    top = book.top_of_book()

    assert top["product_id"] == book.product_id
    assert top["state"] == book.state.value
    assert top["best_bid"] == book.best_bid
    assert top["best_bid_size"] == book.best_bid_size
    assert top["best_ask"] == book.best_ask
    assert top["best_ask_size"] == book.best_ask_size
    assert top["spread"] == book.spread
    assert top["midpoint"] == book.midpoint
    assert top["book_synced"] == book.is_synced()


def test_top_of_book_handles_empty_and_invalid_books():
    book = Level2OrderBook(product_id="BTC-USD")
    top = book.top_of_book()
    assert top["state"] == OrderBookState.UNINITIALIZED.value
    assert top["best_bid"] is None
    assert top["best_bid_size"] is None
    assert top["best_ask"] is None
    assert top["best_ask_size"] is None
    assert top["spread"] is None
    assert top["midpoint"] is None
    assert top["book_synced"] is False

    book.invalidate("test")
    top = book.top_of_book()
    assert top["state"] == OrderBookState.INVALID.value
    assert top["best_bid"] is None
    assert top["best_bid_size"] is None
    assert top["best_ask"] is None
    assert top["best_ask_size"] is None
    assert top["spread"] is None
    assert top["midpoint"] is None
    assert top["book_synced"] is False


def test_is_synced_tracks_validated_state_and_reset():
    book = Level2OrderBook(product_id="BTC-USD")
    assert not book.is_synced()
    book.apply_snapshot(
        product_id="BTC-USD",
        levels={"bid": [{"price": 100.0, "quantity": 1.0}], "ask": [{"price": 101.0, "quantity": 1.0}]},
    )
    assert book.is_synced()
    book.invalidate("test")
    assert not book.is_synced()
    book.reset()
    assert not book.is_synced()


def test_top_of_book_stays_small_for_large_depth_while_snapshot_keeps_depth():
    book = Level2OrderBook(product_id="BTC-USD")
    book._replace_levels_for_state(
        bids={float(price): 1.0 for price in range(1, 5001)},
        asks={float(price): 1.0 for price in range(5001, 10001)},
        state=OrderBookState.SYNCED,
    )

    top = book.top_of_book()
    full = book.snapshot()

    assert set(top) == {
        "product_id", "state", "best_bid", "best_bid_size", "best_ask", "best_ask_size",
        "spread", "midpoint", "book_synced",
    }
    assert len(full["bids"]) == 5000
    assert len(full["asks"]) == 5000


def test_apply_update_requires_prior_sync():
    book = Level2OrderBook(product_id="BTC-USD")
    with pytest.raises(ValueError, match="not synchronized"):
        book.apply_update(side="bid", price=1.0, quantity=1.0)


def test_apply_update_zero_quantity_removes_level():
    book = _synced_book()
    book.apply_update(side="bid", price=99.0, quantity=1.0)
    assert book.bids[99.0] == 1.0
    book.apply_update(side="bid", price=99.0, quantity=0.0)
    assert 99.0 not in book.bids
    assert book.best_bid == 100.0


def test_apply_update_rejecting_crossed_book_invalidates_state():
    book = _synced_book()
    with pytest.raises(ValueError, match="Crossed book"):
        book.apply_update(side="bid", price=105.0, quantity=1.0)
    assert book.state == OrderBookState.INVALID
    assert not book.is_synced()


def test_apply_update_rejects_invalid_price_and_quantity():
    book = _synced_book()
    with pytest.raises(ValueError):
        book.apply_update(side="bid", price=-1.0, quantity=1.0)
    with pytest.raises(ValueError):
        book.apply_update(side="bid", price=100.0, quantity=-1.0)
    with pytest.raises(ValueError):
        book.apply_update(side="bid", price=float("nan"), quantity=1.0)


def test_apply_update_rejects_unsupported_side():
    book = _synced_book()
    with pytest.raises(ValueError, match="Unsupported side"):
        book.apply_update(side="middle", price=100.0, quantity=1.0)

def test_apply_updates_allows_transient_crossed_gce_shaped_batch():
    book = Level2OrderBook(product_id="BTC-USD")
    book.apply_snapshot(
        product_id="BTC-USD",
        levels={
            "bid": [{"price": 83984.02, "quantity": 1.0}],
            "ask": [{"price": 83984.76, "quantity": 1.0}],
        },
    )

    book.apply_updates([
        {"side": "bid", "price": 83992.99, "quantity": 0.04710111},
        {"side": "ask", "price": 83984.76, "quantity": 0},
        {"side": "ask", "price": 83993.50, "quantity": 1.0},
    ])

    assert book.state == OrderBookState.SYNCED
    assert book.best_bid == 83992.99
    assert book.best_ask == 83993.50
    assert book.last_error is None


def test_apply_updates_commits_one_sided_candidate_as_rebuilding_then_recovers():
    book = _synced_book()
    book.apply_updates([{"side": "ask", "price": 101.0, "quantity": 0}])

    assert book.state == OrderBookState.REBUILDING
    assert not book.is_synced()
    assert book.best_bid == 100.0
    assert book.best_ask is None
    assert book.midpoint is None
    assert book.last_error is None

    book.apply_updates([{"side": "ask", "price": 102.0, "quantity": 1.0}])

    assert book.state == OrderBookState.SYNCED
    assert book.is_synced()
    assert book.best_ask == 102.0


def test_apply_updates_allows_a_complete_empty_book_clear_and_refill():
    book = _synced_book()
    book.apply_updates([
        {"side": "bid", "price": 100.0, "quantity": 0},
        {"side": "ask", "price": 101.0, "quantity": 0},
    ])

    assert book.state == OrderBookState.REBUILDING
    assert book.bids == {}
    assert book.asks == {}
    assert not book.is_synced()
    assert book.top_of_book()["midpoint"] is None

    book.apply_updates([
        {"side": "bid", "price": 99.0, "quantity": 2.0},
        {"side": "ask", "price": 100.0, "quantity": 3.0},
    ])

    assert book.state == OrderBookState.SYNCED
    assert book.is_synced()
    assert (book.best_bid, book.best_ask) == (99.0, 100.0)


def test_crossed_candidate_during_rebuild_is_fatal_and_not_committed():
    book = _synced_book()
    book.apply_updates([{"side": "ask", "price": 101.0, "quantity": 0}])
    before_bids = dict(book.bids)
    before_asks = dict(book.asks)

    with pytest.raises(ValueError, match="Crossed book"):
        book.apply_updates([
            {"side": "bid", "price": 102.0, "quantity": 1.0},
            {"side": "ask", "price": 101.0, "quantity": 1.0},
        ])

    assert book.state == OrderBookState.INVALID
    assert book.bids == before_bids
    assert book.asks == before_asks
    with pytest.raises(ValueError, match="not synchronized"):
        book.apply_updates([{"side": "ask", "price": 103.0, "quantity": 1.0}])


def test_malformed_envelope_during_rebuild_is_fatal_and_atomic():
    book = _synced_book()
    book.apply_updates([{"side": "ask", "price": 101.0, "quantity": 0}])
    before_bids = dict(book.bids)
    before_asks = dict(book.asks)

    with pytest.raises(ValueError, match="Invalid ask quantity"):
        book.apply_updates([
            {"side": "ask", "price": 102.0, "quantity": 1.0},
            {"side": "ask", "price": 103.0, "quantity": -1.0},
        ])

    assert book.state == OrderBookState.INVALID
    assert book.bids == before_bids
    assert book.asks == before_asks


def test_apply_updates_rejects_final_cross_without_committing_any_levels():
    book = _synced_book()
    before_bids = dict(book.bids)
    before_asks = dict(book.asks)

    with pytest.raises(ValueError, match="Crossed book"):
        book.apply_updates([{"side": "bid", "price": 102.0, "quantity": 1.0}])

    assert book.state == OrderBookState.INVALID
    assert book.bids == before_bids
    assert book.asks == before_asks
    assert book.last_error is not None and "Crossed book" in book.last_error


@pytest.mark.parametrize(
    "bad_update",
    [
        {"side": "bid", "price": 0, "quantity": 1.0},
        {"side": "bid", "price": float("inf"), "quantity": 1.0},
        {"side": "bid", "price": float("nan"), "quantity": 1.0},
        {"side": "bid", "price": 99.0, "quantity": -1.0},
        {"side": "bid", "price": 99.0, "quantity": float("inf")},
        {"side": "middle", "price": 99.0, "quantity": 1.0},
    ],
)
def test_apply_updates_malformed_batch_is_atomic(bad_update):
    book = _synced_book()
    before_bids = dict(book.bids)
    before_asks = dict(book.asks)

    with pytest.raises(ValueError):
        book.apply_updates([
            {"side": "bid", "price": 99.0, "quantity": 1.0},
            bad_update,
            {"side": "ask", "price": 102.0, "quantity": 1.0},
        ])

    assert book.state == OrderBookState.INVALID
    assert book.bids == before_bids
    assert book.asks == before_asks
    assert book.last_error


def test_apply_updates_preserves_wire_order_for_repeated_price_levels():
    book = _synced_book()
    book.apply_updates([
        {"side": "bid", "price": 99.0, "quantity": 2.0},
        {"side": "bid", "price": 99.0, "quantity": 3.0},
        {"side": "bid", "price": 99.0, "quantity": 0},
    ])
    assert 99.0 not in book.bids
    assert book.best_bid == 100.0


def test_snapshot_requires_two_sided_book():
    book = Level2OrderBook(product_id="BTC-USD")
    with pytest.raises(ValueError, match="both bid and ask"):
        book.apply_snapshot(product_id="BTC-USD", levels={"bid": [{"price": 1.0, "quantity": 1.0}], "ask": []})
    assert book.state == OrderBookState.INVALID


@pytest.mark.parametrize(
    "levels",
    [
        {
            "bid": [{"price": 99.0, "quantity": 2.0}, None],
            "ask": [{"price": 102.0, "quantity": 3.0}],
        },
        {
            "bid": [{"price": 99.0, "quantity": 2.0}, {"price": 98.0, "quantity": -1.0}],
            "ask": [{"price": 102.0, "quantity": 3.0}],
        },
    ],
)
def test_rejected_snapshot_is_atomic_and_fatal(levels):
    book = _synced_book()
    before_bids = dict(book.bids)
    before_asks = dict(book.asks)

    with pytest.raises(ValueError):
        book.apply_snapshot(product_id="BTC-USD", levels=levels)

    assert book.state == OrderBookState.INVALID
    assert not book.is_synced()
    assert book.bids == before_bids
    assert book.asks == before_asks
    assert book.last_error


def test_crossed_snapshot_is_atomic_and_fatal():
    book = _synced_book()
    before_bids = dict(book.bids)
    before_asks = dict(book.asks)

    with pytest.raises(ValueError, match="Crossed book"):
        book.apply_snapshot(
            product_id="BTC-USD",
            levels={
                "bid": [{"price": 103.0, "quantity": 2.0}],
                "ask": [{"price": 102.0, "quantity": 3.0}],
            },
        )

    assert book.state == OrderBookState.INVALID
    assert not book.is_synced()
    assert book.bids == before_bids
    assert book.asks == before_asks
    assert "Crossed book" in book.last_error


def test_reset_clears_book_but_retains_l2_sequence_provenance():
    book = _synced_book()
    book.last_sequence_num = 42
    book.reset()
    assert book.state == OrderBookState.UNINITIALIZED
    assert book.best_bid is None
    assert book.best_ask is None
    # A plain reset() is used for an in-band resnapshot on the *same*
    # connection; sequence numbering must stay continuous across it.
    assert book.last_sequence_num == 42


def test_reset_for_new_connection_clears_book_and_l2_sequence_provenance():
    book = _synced_book()
    book.last_sequence_num = 42
    book.reset_for_new_connection()
    assert book.state == OrderBookState.UNINITIALIZED
    assert book.best_bid is None
    assert book.best_ask is None
    assert book.last_sequence_num is None


def test_invalidate_marks_state_and_records_reason():
    book = _synced_book()
    book.invalidate("manual invalidation")
    assert book.state == OrderBookState.INVALID
    assert book.last_error == "manual invalidation"
    assert not book.is_synced()


def test_snapshot_returns_sorted_levels_and_state():
    book = _synced_book()
    book.apply_update(side="bid", price=99.0, quantity=1.0)
    snap = book.snapshot()
    assert snap["product_id"] == "BTC-USD"
    assert snap["state"] == "SYNCED"
    assert list(snap["bids"].keys()) == sorted(snap["bids"].keys())
    assert snap["best_bid"] == 100.0
    assert snap["best_ask"] == 101.0


def test_snapshot_builds_exact_heaps_and_reset_clears_them():
    book = Level2OrderBook(product_id="BTC-USD")
    book.apply_snapshot(
        product_id="BTC-USD",
        levels={
            "bid": [{"price": 100.0, "quantity": 1.0}, {"price": 99.0, "quantity": 2.0}],
            "ask": [{"price": 101.0, "quantity": 3.0}, {"price": 102.0, "quantity": 4.0}],
        },
    )
    assert sorted(-price for price in book._bid_heap) == sorted(book.bids)
    assert sorted(book._ask_heap) == sorted(book.asks)
    assert len(book._bid_heap) == len(book.bids)
    assert len(book._ask_heap) == len(book.asks)

    book.last_sequence_num = 42
    book.reset()
    assert book._bid_heap == book._ask_heap == []
    assert book.last_sequence_num == 42
    book.apply_snapshot(
        product_id="BTC-USD",
        levels={"bid": [{"price": 100.0, "quantity": 1.0}],
                "ask": [{"price": 101.0, "quantity": 2.0}]},
    )
    book.reset_for_new_connection()
    assert book._bid_heap == book._ask_heap == []
    assert book.last_sequence_num is None


def test_failed_snapshot_preserves_old_heaps_and_dictionary_levels():
    book = _synced_book()
    before_bids, before_asks = dict(book.bids), dict(book.asks)
    before_bid_heap, before_ask_heap = list(book._bid_heap), list(book._ask_heap)

    with pytest.raises(ValueError, match="Crossed book"):
        book.apply_snapshot(
            product_id="BTC-USD",
            levels={"bid": [{"price": 103.0, "quantity": 1.0}],
                    "ask": [{"price": 102.0, "quantity": 1.0}]},
        )

    assert book.bids == before_bids and book.asks == before_asks
    assert book._bid_heap == before_bid_heap
    assert book._ask_heap == before_ask_heap
    assert book.state == OrderBookState.INVALID
    assert (book.best_bid, book.best_ask) == (100.0, 101.0)


def test_small_successful_update_keeps_dictionary_objects_in_place():
    book = _synced_book()
    bids_identity, asks_identity = id(book.bids), id(book.asks)
    book.apply_update(side="bid", price=99.0, quantity=2.0)
    assert id(book.bids) == bids_identity
    assert id(book.asks) == asks_identity


def test_deleting_last_side_is_lazily_pruned_by_heap_lookup():
    book = Level2OrderBook(product_id="BTC-USD")
    book.apply_snapshot(
        product_id="BTC-USD",
        levels={"bid": [{"price": 100.0, "quantity": 1.0}],
                "ask": [{"price": 101.0, "quantity": 1.0}]},
    )
    book.apply_updates([{"side": "bid", "price": 100.0, "quantity": 0.0}])
    assert book.state == OrderBookState.REBUILDING
    assert book._bid_heap
    assert book.best_bid is None
    assert book._bid_heap == []


def test_failed_cross_rollback_restores_pruned_best_bid_heap_entry():
    book = Level2OrderBook(product_id="BTC-USD")
    book.apply_snapshot(
        product_id="BTC-USD",
        levels={
            "bid": [{"price": 100.0, "quantity": 1.0}, {"price": 99.0, "quantity": 2.0}],
            "ask": [{"price": 101.0, "quantity": 3.0}],
        },
    )
    before_bids, before_asks = dict(book.bids), dict(book.asks)
    with pytest.raises(ValueError, match="Crossed book"):
        book.apply_updates([
            {"side": "bid", "price": 100.0, "quantity": 0.0},
            {"side": "ask", "price": 98.0, "quantity": 1.0},
        ])
    assert book.bids == before_bids and book.asks == before_asks
    assert book._peek_best_bid() == 100.0
    assert book.state == OrderBookState.INVALID


def test_failed_cross_rollback_restores_pruned_best_ask_heap_entry():
    book = Level2OrderBook(product_id="BTC-USD")
    book.apply_snapshot(
        product_id="BTC-USD",
        levels={
            "bid": [{"price": 100.0, "quantity": 1.0}],
            "ask": [{"price": 101.0, "quantity": 2.0}, {"price": 102.0, "quantity": 3.0}],
        },
    )
    before_bids, before_asks = dict(book.bids), dict(book.asks)
    with pytest.raises(ValueError, match="Crossed book"):
        book.apply_updates([
            {"side": "ask", "price": 101.0, "quantity": 0.0},
            {"side": "ask", "price": 102.0, "quantity": 1.0},
            {"side": "bid", "price": 103.0, "quantity": 1.0},
        ])
    assert book.bids == before_bids and book.asks == before_asks
    assert book._peek_best_ask() == 101.0
    assert book.state == OrderBookState.INVALID


def test_rollback_journals_original_value_across_repeated_same_price_mutations():
    book = _synced_book()
    before_bids, before_asks = dict(book.bids), dict(book.asks)
    with pytest.raises(ValueError, match="Crossed book"):
        book.apply_updates([
            {"side": "bid", "price": 100.0, "quantity": 2.0},
            {"side": "bid", "price": 100.0, "quantity": 0.0},
            {"side": "bid", "price": 100.0, "quantity": 5.0},
            {"side": "bid", "price": 102.0, "quantity": 1.0},
        ])
    assert book.bids == before_bids
    assert book.asks == before_asks
    assert book.bids[100.0] == 1.0
    assert book.state == OrderBookState.INVALID


def test_heap_compaction_bounds_churn_and_collapses_full_clear():
    book = Level2OrderBook(product_id="BTC-USD")
    depth = 3000
    book._replace_levels_for_state(
        bids={float(50_000 - price): 1.0 for price in range(depth)},
        asks={float(100_000 + price): 1.0 for price in range(depth)},
        state=OrderBookState.SYNCED,
    )
    for index in range(6000):
        side = "bid" if index % 2 == 0 else "ask"
        price = float(1 + index if side == "bid" else 110_000 + index)
        book.apply_updates([
            {"side": side, "price": price, "quantity": 1.0},
            {"side": side, "price": price, "quantity": 0.0},
        ])
        if index % 500 == 0:
            assert len(book._bid_heap) <= 2 * len(book.bids) + 1024
            assert len(book._ask_heap) <= 2 * len(book.asks) + 1024
            assert book.state == OrderBookState.SYNCED
            assert book.best_bid == 50_000.0
            assert book.best_ask == 100_000.0
    assert len(book._bid_heap) <= 2 * len(book.bids) + 1024
    assert len(book._ask_heap) <= 2 * len(book.asks) + 1024

    book.apply_updates([
        *({"side": "bid", "price": price, "quantity": 0.0} for price in tuple(book.bids)),
        *({"side": "ask", "price": price, "quantity": 0.0} for price in tuple(book.asks)),
    ])
    assert book.state == OrderBookState.REBUILDING
    assert book.bids == book.asks == {}
    assert book._bid_heap == book._ask_heap == []


def _assert_matches_reference(actual, expected):
    assert actual.state.value == expected.state.value
    assert actual.bids == expected.bids
    assert actual.asks == expected.asks
    assert actual.best_bid == expected.best_bid
    assert actual.best_bid_size == expected.best_bid_size
    assert actual.best_ask == expected.best_ask
    assert actual.best_ask_size == expected.best_ask_size
    assert actual.spread == expected.spread
    assert actual.midpoint == expected.midpoint
    assert actual.is_synced() == expected.is_synced()
    assert actual.last_error == expected.last_error


@pytest.mark.parametrize("seed", [1, 7, 42, 2026, 19019])
def test_deterministic_randomized_differential_against_copy_on_write_reference(seed):
    import random

    rng = random.Random(seed)
    actual = Level2OrderBook(product_id="BTC-USD")
    expected = CopyOnWriteReferenceBook(product_id="BTC-USD")
    initial = {
        "bid": [{"price": float(100 - i), "quantity": float(i + 1)} for i in range(8)],
        "ask": [{"price": float(101 + i), "quantity": float(i + 1)} for i in range(8)],
    }
    actual.apply_snapshot(product_id="BTC-USD", levels=initial)
    expected.apply_snapshot(product_id="BTC-USD", levels=initial)
    serial = 0
    envelope_count = 2100
    for index in range(envelope_count):
        if index % 211 == 0:
            updates = [
                *({"side": "bid", "price": price, "quantity": 0.0} for price in tuple(actual.bids)),
                *({"side": "ask", "price": price, "quantity": 0.0} for price in tuple(actual.asks)),
            ]
            actual.apply_updates(updates)
            expected.apply_updates(updates)
        elif actual.state == OrderBookState.REBUILDING:
            updates = [
                {"side": "bid", "price": 99.0, "quantity": 1.0},
                {"side": "ask", "price": 102.0, "quantity": 1.0},
            ]
            actual.apply_updates(updates)
            expected.apply_updates(updates)
        elif index % 257 == 0:
            before_bids, before_asks = dict(actual.bids), dict(actual.asks)
            updates = [{
                "side": "bid",
                "price": actual.best_ask + 1.0,
                "quantity": 1.0,
            }]
            with pytest.raises(ValueError) as actual_error:
                actual.apply_updates(updates)
            with pytest.raises(ValueError) as expected_error:
                expected.apply_updates(updates)
            assert "Crossed book" in str(actual_error.value)
            assert "Crossed book" in str(expected_error.value)
            assert actual.bids == before_bids == expected.bids
            assert actual.asks == before_asks == expected.asks
            assert actual.state == OrderBookState.INVALID
            assert expected.state.value == OrderBookState.INVALID.value
            assert actual.best_bid == expected.best_bid
            assert actual.best_ask == expected.best_ask
            actual.reset()
            expected.reset()
            actual.apply_snapshot(product_id="BTC-USD", levels=initial)
            expected.apply_snapshot(product_id="BTC-USD", levels=initial)
        else:
            side = rng.choice(("bid", "ask"))
            levels = actual.bids if side == "bid" else actual.asks
            action = rng.randrange(4)
            if action == 0:
                price = rng.choice(tuple(levels))
                quantity = rng.random() * 5 + 0.01
            elif action == 1:
                serial += 1
                price = 1.0 + serial / 10_000 if side == "bid" else 1_000.0 + serial
                quantity = rng.random() * 5 + 0.01
            elif action == 2 and len(levels) > 1:
                best = max(levels) if side == "bid" else min(levels)
                price = rng.choice(tuple(price for price in levels if price != best))
                quantity = 0.0
            else:
                price = max(levels) if side == "bid" else min(levels)
                quantity = 0.0
            updates = [{"side": side, "price": price, "quantity": quantity}]
            actual.apply_updates(updates)
            expected.apply_updates(updates)
        _assert_matches_reference(actual, expected)
    assert envelope_count == 2100


@pytest.mark.parametrize(
    "update",
    [
        {"side": "middle", "price": 99.0, "quantity": 1.0},
        {"side": "bid", "price": float("nan"), "quantity": 1.0},
        {"side": "ask", "price": 102.0, "quantity": -1.0},
    ],
)
def test_differential_malformed_envelopes_are_atomic(update):
    actual, expected = _synced_book(), CopyOnWriteReferenceBook(product_id="BTC-USD")
    expected.apply_snapshot(
        product_id="BTC-USD",
        levels={"bid": [{"price": 100.0, "quantity": 1.0}],
                "ask": [{"price": 101.0, "quantity": 2.0}]},
    )
    before_bids, before_asks = dict(actual.bids), dict(actual.asks)
    updates = [{"side": "bid", "price": 99.0, "quantity": 1.0}, update]
    with pytest.raises(ValueError):
        actual.apply_updates(updates)
    with pytest.raises(ValueError):
        expected.apply_updates(updates)
    assert actual.bids == before_bids == expected.bids
    assert actual.asks == before_asks == expected.asks
    _assert_matches_reference(actual, expected)
