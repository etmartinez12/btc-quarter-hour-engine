from __future__ import annotations

import pytest

from btc_quarter_hour_engine.market_data.order_book import (
    Level2OrderBook,
    OrderBookState,
)


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


def test_apply_updates_allows_temporarily_one_sided_book():
    book = _synced_book()
    book.apply_updates([
        {"side": "ask", "price": 101.0, "quantity": 0},
        {"side": "ask", "price": 102.0, "quantity": 1.0},
    ])
    assert book.is_synced()
    assert book.best_ask == 102.0


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
