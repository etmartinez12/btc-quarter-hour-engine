from __future__ import annotations

import heapq
import math
from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Any

HEAP_COMPACTION_FACTOR = 2
HEAP_COMPACTION_SLACK = 1024
_MISSING = object()


class OrderBookState(str, Enum):
    UNINITIALIZED = "UNINITIALIZED"
    SYNCED = "SYNCED"
    REBUILDING = "REBUILDING"
    INVALID = "INVALID"


class Level2OrderBook:
    """Deterministic Coinbase L2 book with recoverable and fatal states.

    A complete update envelope may commit a temporarily one-sided or empty
    candidate as ``REBUILDING``. Sequence-contiguous envelopes can restore
    ``SYNCED``; malformed or crossed candidates transition to ``INVALID``.
    """

    def __init__(self, *, product_id: str | None = None) -> None:
        self.product_id = product_id
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self._bid_heap: list[float] = []
        self._ask_heap: list[float] = []
        self.state = OrderBookState.UNINITIALIZED
        self.last_sequence_num: int | None = None
        self.last_error: str | None = None

    def _rebuild_heaps(self) -> None:
        self._bid_heap = [-price for price in self.bids]
        self._ask_heap = list(self.asks)
        heapq.heapify(self._bid_heap)
        heapq.heapify(self._ask_heap)

    def _replace_levels_for_state(
        self,
        *,
        bids: Mapping[float, float],
        asks: Mapping[float, float],
        state: OrderBookState,
    ) -> None:
        """Reconstruct a trusted compact replay state and its runtime indexes."""
        self.bids = dict(bids)
        self.asks = dict(asks)
        self._rebuild_heaps()
        self.state = state
        self.last_error = None

    def _peek_best_bid(self) -> float | None:
        while self._bid_heap:
            price = -self._bid_heap[0]
            if price in self.bids:
                return price
            heapq.heappop(self._bid_heap)
        return None

    def _peek_best_ask(self) -> float | None:
        while self._ask_heap:
            price = self._ask_heap[0]
            if price in self.asks:
                return price
            heapq.heappop(self._ask_heap)
        return None

    def _compact_heaps_if_needed(self) -> None:
        if len(self._bid_heap) > HEAP_COMPACTION_FACTOR * len(self.bids) + HEAP_COMPACTION_SLACK:
            self._bid_heap = [-price for price in self.bids]
            heapq.heapify(self._bid_heap)
        if len(self._ask_heap) > HEAP_COMPACTION_FACTOR * len(self.asks) + HEAP_COMPACTION_SLACK:
            self._ask_heap = list(self.asks)
            heapq.heapify(self._ask_heap)

    @property
    def best_bid(self) -> float | None:
        return self._peek_best_bid()

    @property
    def best_ask(self) -> float | None:
        return self._peek_best_ask()

    @property
    def best_bid_size(self) -> float | None:
        bid = self._peek_best_bid()
        return self.bids[bid] if bid is not None else None

    @property
    def best_ask_size(self) -> float | None:
        ask = self._peek_best_ask()
        return self.asks[ask] if ask is not None else None

    @property
    def spread(self) -> float | None:
        bid = self._peek_best_bid()
        ask = self._peek_best_ask()
        if bid is None or ask is None:
            return None
        return ask - bid

    @property
    def midpoint(self) -> float | None:
        bid = self._peek_best_bid()
        ask = self._peek_best_ask()
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def invalidate(self, reason: str | None = None) -> None:
        self.last_error = reason
        self.state = OrderBookState.INVALID

    def reset(self) -> None:
        self.bids.clear()
        self.asks.clear()
        self._bid_heap.clear()
        self._ask_heap.clear()
        self.state = OrderBookState.UNINITIALIZED
        self.last_error = None

    def reset_for_new_connection(self) -> None:
        """Full reset used when starting a brand new connection epoch.

        Unlike :meth:`reset` (used for an in-band resnapshot on the *same*
        connection), this also clears ``last_sequence_num``, which retains
        the source sequence number of the most recently applied L2 mutation.
        """
        self.reset()
        self.last_sequence_num = None

    def _require_valid_price(self, price: Any, *, field_name: str = "price") -> float:
        try:
            numeric = float(price)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed {field_name}: {price!r}") from exc
        if not math.isfinite(numeric) or numeric <= 0:
            raise ValueError(f"Invalid {field_name}: {price!r}")
        return numeric

    def _require_valid_quantity(self, quantity: Any, *, field_name: str = "quantity") -> float:
        try:
            numeric = float(quantity)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed {field_name}: {quantity!r}") from exc
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError(f"Invalid {field_name}: {quantity!r}")
        return numeric

    def _normalize_side(self, side: Any) -> str:
        normalized = str(side).lower()
        if normalized not in {"bid", "ask"}:
            raise ValueError(f"Unsupported side: {side!r}")
        return normalized

    def _validate_candidate_book(self, bids: Mapping[float, float], asks: Mapping[float, float]) -> None:
        if not bids or not asks:
            raise ValueError("Order book requires both bid and ask sides")
        best_bid = max(bids)
        best_ask = min(asks)
        bid_size = bids[best_bid]
        ask_size = asks[best_ask]
        values = [best_bid, best_ask, bid_size, ask_size]
        if any(value is None for value in values):
            raise ValueError("Best bid/ask values are incomplete")
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Best bid/ask values must be finite")
        if best_bid <= 0 or best_ask <= 0:
            raise ValueError("Best bid and ask prices must be positive")
        if best_bid > best_ask:
            raise ValueError("Crossed book: best_bid exceeds best_ask")
        if bid_size < 0 or ask_size < 0:
            raise ValueError("Best bid/ask sizes cannot be negative")

    def _validate_live_candidate(self) -> None:
        bid = self._peek_best_bid()
        ask = self._peek_best_ask()
        if bid is None or ask is None:
            raise ValueError("Order book requires both bid and ask sides")
        bid_size = self.bids[bid]
        ask_size = self.asks[ask]
        values = [bid, ask, bid_size, ask_size]
        if any(value is None for value in values):
            raise ValueError("Best bid/ask values are incomplete")
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Best bid/ask values must be finite")
        if bid <= 0 or ask <= 0:
            raise ValueError("Best bid and ask prices must be positive")
        if bid > ask:
            raise ValueError("Crossed book: best_bid exceeds best_ask")
        if bid_size < 0 or ask_size < 0:
            raise ValueError("Best bid/ask sizes cannot be negative")

    def _validate_book_integrity(self) -> None:
        self._validate_candidate_book(self.bids, self.asks)

    def _apply_level(self, *, side: str, price: float, quantity: float) -> None:
        if quantity == 0:
            self.bids.pop(price, None) if side == "bid" else self.asks.pop(price, None)
            return
        if side == "bid":
            self.bids[price] = quantity
        else:
            self.asks[price] = quantity

    @staticmethod
    def _apply_level_to(
        bids: dict[float, float],
        asks: dict[float, float],
        *,
        side: str,
        price: float,
        quantity: float,
    ) -> None:
        levels = bids if side == "bid" else asks
        if quantity == 0:
            levels.pop(price, None)
        else:
            levels[price] = quantity

    def apply_snapshot(self, *, product_id: str | None, levels: Mapping[str, list[Mapping[str, Any]]]) -> None:
        if product_id is not None:
            self.product_id = product_id
        staged_bids: dict[float, float] = {}
        staged_asks: dict[float, float] = {}
        try:
            if not isinstance(levels, Mapping):
                raise ValueError("Malformed order book snapshot")
            for side in ("bid", "ask"):
                for level in list(levels.get(side, []) or levels.get(f"{side}s", []) or []):
                    if not isinstance(level, Mapping):
                        raise ValueError(f"Malformed {side} level")
                    price = self._require_valid_price(level.get("price"), field_name=f"{side} price")
                    qty = self._require_valid_quantity(level.get("quantity"), field_name=f"{side} quantity")
                    self._apply_level_to(
                        staged_bids,
                        staged_asks,
                        side=side,
                        price=price,
                        quantity=qty,
                    )
            self._validate_candidate_book(staged_bids, staged_asks)
        except ValueError as exc:
            self.invalidate(str(exc))
            raise
        except TypeError as exc:
            self.invalidate(str(exc))
            raise ValueError(str(exc)) from exc
        self.bids = staged_bids
        self.asks = staged_asks
        self._rebuild_heaps()
        self.state = OrderBookState.SYNCED
        self.last_error = None

    def apply_update(self, *, side: str, price: Any, quantity: Any) -> None:
        self.apply_updates([{"side": side, "price": price, "quantity": quantity}])

    def apply_updates(self, updates: Iterable[Mapping[str, Any]]) -> None:
        if self.state not in {OrderBookState.SYNCED, OrderBookState.REBUILDING}:
            raise ValueError("Order book is not synchronized")

        normalized_updates: list[tuple[str, float, float]] = []
        try:
            for update in updates:
                if not isinstance(update, Mapping):
                    raise ValueError("Malformed order book update")
                side = self._normalize_side(update.get("side"))
                price = self._require_valid_price(update.get("price"), field_name=f"{side} price")
                quantity = self._require_valid_quantity(update.get("quantity"), field_name=f"{side} quantity")
                normalized_updates.append((side, price, quantity))
        except (TypeError, ValueError) as exc:
            self.invalidate(str(exc))
            raise ValueError(str(exc)) from exc

        journal: dict[tuple[str, float], object] = {}
        for side, price, quantity in normalized_updates:
            levels = self.bids if side == "bid" else self.asks
            key = (side, price)
            if key not in journal:
                journal[key] = levels.get(price, _MISSING)
            if quantity == 0:
                levels.pop(price, None)
            else:
                was_absent = price not in levels
                levels[price] = quantity
                if was_absent:
                    heap = self._bid_heap if side == "bid" else self._ask_heap
                    heapq.heappush(heap, -price if side == "bid" else price)
        if not self.bids or not self.asks:
            # Coinbase can clear a complete book in-band and refill it in
            # later sequence-contiguous envelopes. Commit each full envelope,
            # but never present this intermediate state as a canonical BBO.
            self.state = OrderBookState.REBUILDING
            self.last_error = None
            self._compact_heaps_if_needed()
            return
        try:
            self._validate_live_candidate()
        except ValueError as exc:
            for (side, price), original_quantity in journal.items():
                levels = self.bids if side == "bid" else self.asks
                if original_quantity is _MISSING:
                    levels.pop(price, None)
                else:
                    levels[price] = original_quantity
                    heap = self._bid_heap if side == "bid" else self._ask_heap
                    heapq.heappush(heap, -price if side == "bid" else price)
            self.invalidate(str(exc))
            self._compact_heaps_if_needed()
            raise

        self.state = OrderBookState.SYNCED
        self.last_error = None
        self._compact_heaps_if_needed()

    def snapshot(self) -> dict[str, Any]:
        return {
            "product_id": self.product_id,
            "state": self.state.value,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "best_bid_size": self.best_bid_size,
            "best_ask_size": self.best_ask_size,
            "spread": self.spread,
            "midpoint": self.midpoint,
            "bids": dict(sorted(self.bids.items())),
            "asks": dict(sorted(self.asks.items())),
        }

    def top_of_book(self) -> dict[str, Any]:
        best_bid = self._peek_best_bid()
        best_bid_size = self.bids[best_bid] if best_bid is not None else None
        best_ask = self._peek_best_ask()
        best_ask_size = self.asks[best_ask] if best_ask is not None else None
        spread = best_ask - best_bid if best_bid is not None and best_ask is not None else None
        midpoint = (best_bid + best_ask) / 2.0 if best_bid is not None and best_ask is not None else None
        return {
            "product_id": self.product_id,
            "state": self.state.value,
            "best_bid": best_bid,
            "best_bid_size": best_bid_size,
            "best_ask": best_ask,
            "best_ask_size": best_ask_size,
            "spread": spread,
            "midpoint": midpoint,
            "book_synced": self.is_synced(),
        }

    def is_synced(self) -> bool:
        return self.state == OrderBookState.SYNCED and bool(self.bids) and bool(self.asks)


__all__ = ["Level2OrderBook", "OrderBookState"]
