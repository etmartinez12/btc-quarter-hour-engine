"""Copy-on-write Level2 book used as an independent semantic oracle."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Any


class ReferenceBookState(str, Enum):
    UNINITIALIZED = "UNINITIALIZED"
    SYNCED = "SYNCED"
    REBUILDING = "REBUILDING"
    INVALID = "INVALID"


class CopyOnWriteReferenceBook:
    """Straightforward copy-on-write implementation of the PR #18 contract."""

    def __init__(self, *, product_id: str | None = None) -> None:
        self.product_id = product_id
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.state = ReferenceBookState.UNINITIALIZED
        self.last_sequence_num: int | None = None
        self.last_error: str | None = None

    @property
    def best_bid(self) -> float | None:
        return max(self.bids) if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return min(self.asks) if self.asks else None

    @property
    def best_bid_size(self) -> float | None:
        return self.bids[self.best_bid] if self.best_bid is not None else None

    @property
    def best_ask_size(self) -> float | None:
        return self.asks[self.best_ask] if self.best_ask is not None else None

    @property
    def spread(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    @property
    def midpoint(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2.0

    def top_of_book(self) -> dict[str, Any]:
        bid, ask = self.best_bid, self.best_ask
        bid_size = self.bids[bid] if bid is not None else None
        ask_size = self.asks[ask] if ask is not None else None
        spread = ask - bid if bid is not None and ask is not None else None
        midpoint = (bid + ask) / 2.0 if bid is not None and ask is not None else None
        return {
            "product_id": self.product_id,
            "state": self.state.value,
            "best_bid": bid,
            "best_bid_size": bid_size,
            "best_ask": ask,
            "best_ask_size": ask_size,
            "spread": spread,
            "midpoint": midpoint,
            "book_synced": self.is_synced(),
        }

    def is_synced(self) -> bool:
        return self.state == ReferenceBookState.SYNCED and bool(self.bids) and bool(self.asks)

    def invalidate(self, reason: str | None = None) -> None:
        self.last_error = reason
        self.state = ReferenceBookState.INVALID

    def reset(self) -> None:
        self.bids.clear()
        self.asks.clear()
        self.state = ReferenceBookState.UNINITIALIZED
        self.last_error = None

    def reset_for_new_connection(self) -> None:
        self.reset()
        self.last_sequence_num = None

    @staticmethod
    def _price(value: Any, field: str) -> float:
        try:
            result = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed {field}: {value!r}") from exc
        if not math.isfinite(result) or result <= 0:
            raise ValueError(f"Invalid {field}: {value!r}")
        return result

    @staticmethod
    def _quantity(value: Any, field: str) -> float:
        try:
            result = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed {field}: {value!r}") from exc
        if not math.isfinite(result) or result < 0:
            raise ValueError(f"Invalid {field}: {value!r}")
        return result

    @staticmethod
    def _side(value: Any) -> str:
        side = str(value).lower()
        if side not in {"bid", "ask"}:
            raise ValueError(f"Unsupported side: {value!r}")
        return side

    @classmethod
    def _normalize(cls, updates: Iterable[Mapping[str, Any]]) -> list[tuple[str, float, float]]:
        result = []
        for update in updates:
            if not isinstance(update, Mapping):
                raise ValueError("Malformed order book update")
            side = cls._side(update.get("side"))
            price = cls._price(update.get("price"), f"{side} price")
            quantity = cls._quantity(update.get("quantity"), f"{side} quantity")
            result.append((side, price, quantity))
        return result

    @staticmethod
    def _apply(
        bids: dict[float, float],
        asks: dict[float, float],
        updates: Iterable[tuple[str, float, float]],
    ) -> None:
        for side, price, quantity in updates:
            levels = bids if side == "bid" else asks
            if quantity == 0:
                levels.pop(price, None)
            else:
                levels[price] = quantity

    @staticmethod
    def _validate(bids: Mapping[float, float], asks: Mapping[float, float]) -> None:
        if not bids or not asks:
            raise ValueError("Order book requires both bid and ask sides")
        bid, ask = max(bids), min(asks)
        bid_size, ask_size = bids[bid], asks[ask]
        values = (bid, ask, bid_size, ask_size)
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

    def apply_snapshot(
        self,
        *,
        product_id: str | None,
        levels: Mapping[str, list[Mapping[str, Any]]],
    ) -> None:
        if product_id is not None:
            self.product_id = product_id
        bids: dict[float, float] = {}
        asks: dict[float, float] = {}
        try:
            if not isinstance(levels, Mapping):
                raise ValueError("Malformed order book snapshot")
            for side in ("bid", "ask"):
                for level in list(levels.get(side, []) or levels.get(f"{side}s", []) or []):
                    if not isinstance(level, Mapping):
                        raise ValueError(f"Malformed {side} level")
                    price = self._price(level.get("price"), f"{side} price")
                    quantity = self._quantity(level.get("quantity"), f"{side} quantity")
                    self._apply(bids, asks, [(side, price, quantity)])
            self._validate(bids, asks)
        except ValueError as exc:
            self.invalidate(str(exc))
            raise
        except TypeError as exc:
            self.invalidate(str(exc))
            raise ValueError(str(exc)) from exc
        self.bids, self.asks = bids, asks
        self.state = ReferenceBookState.SYNCED
        self.last_error = None

    def apply_updates(self, updates: Iterable[Mapping[str, Any]]) -> None:
        if self.state not in {ReferenceBookState.SYNCED, ReferenceBookState.REBUILDING}:
            raise ValueError("Order book is not synchronized")
        try:
            normalized = self._normalize(updates)
        except (TypeError, ValueError) as exc:
            self.invalidate(str(exc))
            raise ValueError(str(exc)) from exc

        bids, asks = dict(self.bids), dict(self.asks)
        self._apply(bids, asks, normalized)
        if not bids or not asks:
            self.bids, self.asks = bids, asks
            self.state = ReferenceBookState.REBUILDING
            self.last_error = None
            return
        try:
            self._validate(bids, asks)
        except ValueError as exc:
            self.invalidate(str(exc))
            raise
        self.bids, self.asks = bids, asks
        self.state = ReferenceBookState.SYNCED
        self.last_error = None
