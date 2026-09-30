from __future__ import annotations

import math
from collections.abc import Mapping
from enum import Enum
from typing import Any


class OrderBookState(str, Enum):
    UNINITIALIZED = "UNINITIALIZED"
    SYNCED = "SYNCED"
    INVALID = "INVALID"


class Level2OrderBook:
    """Minimal, deterministic Coinbase L2 book state machine."""

    def __init__(self, *, product_id: str | None = None) -> None:
        self.product_id = product_id
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.state = OrderBookState.UNINITIALIZED
        self.last_sequence_num: int | None = None
        self.last_error: str | None = None

    @property
    def best_bid(self) -> float | None:
        return max(self.bids.items(), key=lambda item: item[0])[0] if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return min(self.asks.items(), key=lambda item: item[0])[0] if self.asks else None

    @property
    def best_bid_size(self) -> float | None:
        if not self.bids:
            return None
        return self.bids[self.best_bid]

    @property
    def best_ask_size(self) -> float | None:
        if not self.asks:
            return None
        return self.asks[self.best_ask]

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

    def invalidate(self, reason: str | None = None) -> None:
        self.last_error = reason
        self.state = OrderBookState.INVALID

    def reset(self) -> None:
        self.bids.clear()
        self.asks.clear()
        self.state = OrderBookState.UNINITIALIZED
        self.last_error = None

    def _require_valid_price(self, price: Any, *, field_name: str = "price") -> float:
        try:
            numeric = float(price)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Malformed {field_name}: {price!r}") from exc
        if not math.isfinite(numeric) or numeric <= 0:
            raise ValueError(f"Invalid {field_name}: {price!r}")
        return numeric

    def _require_valid_quantity(self, quantity: Any, *, field_name: str = "quantity") -> float:
        try:
            numeric = float(quantity)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Malformed {field_name}: {quantity!r}") from exc
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError(f"Invalid {field_name}: {quantity!r}")
        return numeric

    def _normalize_side(self, side: Any) -> str:
        normalized = str(side).lower()
        if normalized not in {"bid", "ask"}:
            raise ValueError(f"Unsupported side: {side!r}")
        return normalized

    def _validate_book_integrity(self) -> None:
        if not self.bids or not self.asks:
            raise ValueError("Order book requires both bid and ask sides")
        best_bid = self.best_bid
        best_ask = self.best_ask
        bid_size = self.best_bid_size
        ask_size = self.best_ask_size
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

    def _apply_level(self, *, side: str, price: float, quantity: float) -> None:
        if quantity == 0:
            self.bids.pop(price, None) if side == "bid" else self.asks.pop(price, None)
            return
        if side == "bid":
            self.bids[price] = quantity
        else:
            self.asks[price] = quantity

    def apply_snapshot(self, *, product_id: str | None, levels: Mapping[str, list[Mapping[str, Any]]]) -> None:
        if product_id is not None:
            self.product_id = product_id
        self.bids.clear()
        self.asks.clear()
        for side in ("bid", "ask"):
            for level in list(levels.get(side, []) or levels.get(f"{side}s", []) or []):
                if not isinstance(level, Mapping):
                    raise ValueError(f"Malformed {side} level")
                price = self._require_valid_price(level.get("price"), field_name=f"{side} price")
                qty = self._require_valid_quantity(level.get("quantity"), field_name=f"{side} quantity")
                self._apply_level(side=side, price=price, quantity=qty)
        try:
            self._validate_book_integrity()
        except ValueError as exc:
            self.invalidate(str(exc))
            raise
        self.state = OrderBookState.SYNCED

    def apply_update(self, *, side: str, price: Any, quantity: Any) -> None:
        normalized_side = self._normalize_side(side)
        price_value = self._require_valid_price(price, field_name=f"{normalized_side} price")
        quantity_value = self._require_valid_quantity(quantity, field_name=f"{normalized_side} quantity")
        if self.state != OrderBookState.SYNCED:
            raise ValueError("Order book is not synchronized")
        self._apply_level(side=normalized_side, price=price_value, quantity=quantity_value)
        try:
            self._validate_book_integrity()
        except ValueError as exc:
            self.invalidate(str(exc))
            raise

    def validate_sequence(self, sequence_num: int | None) -> None:
        if sequence_num is None:
            return
        if self.last_sequence_num is None:
            self.last_sequence_num = sequence_num
            return
        if sequence_num <= self.last_sequence_num:
            raise ValueError(f"Sequence out of order: expected > {self.last_sequence_num}, received {sequence_num}")
        if sequence_num > self.last_sequence_num + 1:
            raise ValueError(
                f"Sequence gap detected: expected {self.last_sequence_num + 1}, received {sequence_num}"
            )
        self.last_sequence_num = sequence_num

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

    def is_synced(self) -> bool:
        return self.state == OrderBookState.SYNCED and self.best_bid is not None and self.best_ask is not None


__all__ = ["Level2OrderBook", "OrderBookState"]
