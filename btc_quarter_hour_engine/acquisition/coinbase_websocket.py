from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from .config import CoinbaseWebSocketConfig


class WebSocketTransport(Protocol):
    def send(self, payload: str) -> None: ...
    def recv(self, timeout: float | None = None) -> str: ...
    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class CoinbaseWebSocketFrame:
    raw: str
    message: dict[str, Any]
    received_at_utc: datetime
    connection_id: str | None = None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _normalize_side(value: Any) -> str:
    normalized = str(value).lower()
    mapping = {"bid": "bid", "ask": "ask", "buy": "bid", "sell": "ask"}
    if normalized not in mapping:
        raise ValueError(f"Unsupported side: {value!r}")
    return mapping[normalized]


def parse_level2_event(message: Mapping[str, Any]) -> dict[str, Any]:
    if not _is_mapping(message):
        raise ValueError("Level 2 message must be a mapping")
    event_type = message.get("type")
    if event_type not in {"snapshot", "l2_data"}:
        raise ValueError(f"Unsupported Level 2 event type: {event_type!r}")
    product_id = message.get("product_id")
    if not isinstance(product_id, str) or not product_id:
        raise ValueError("Level 2 event missing product_id")
    sequence_num = message.get("sequence_num")
    if sequence_num is None:
        raise ValueError("Level 2 event missing sequence_num")
    try:
        sequence_num = int(sequence_num)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Malformed sequence_num: {sequence_num!r}") from exc
    if sequence_num < 0:
        raise ValueError(f"Negative sequence_num: {sequence_num!r}")
    envelope_time = message.get("time") or message.get("timestamp")
    if envelope_time is None:
        raise ValueError("Level 2 event missing envelope timestamp")
    try:
        parsed_time = datetime.fromisoformat(str(envelope_time).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Malformed Level 2 timestamp: {envelope_time!r}") from exc
    if parsed_time.tzinfo is None:
        raise ValueError(f"Timezone-naive Level 2 timestamp: {envelope_time!r}")
    event_time_utc = None
    raw_event_time = message.get("event_time") or message.get("event_time_utc") or message.get("time")
    if raw_event_time is not None:
        try:
            event_time_utc = datetime.fromisoformat(str(raw_event_time).replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError as exc:
            raise ValueError(f"Malformed Level 2 event time: {raw_event_time!r}") from exc
    updates = []
    if event_type == "snapshot":
        for side in ("bids", "asks"):
            entries = message.get(side, [])
            if not isinstance(entries, list):
                raise ValueError(f"Snapshot {side} entries must be a list")
            for entry in entries:
                if not _is_mapping(entry):
                    raise ValueError(f"Snapshot {side} item must be an object")
                price = entry.get("price")
                qty = entry.get("quantity", entry.get("size"))
                if price is None or qty is None:
                    raise ValueError(f"Snapshot {side} item missing price/quantity")
                price_value = float(price)
                qty_value = float(qty)
                if not (price_value > 0 and qty_value >= 0 and price_value == price_value):
                    raise ValueError(f"Invalid snapshot level for {side}: {entry!r}")
                updates.append({"side": "bid" if side == "bids" else "ask", "price": price_value, "quantity": qty_value})
    else:
        changes = message.get("changes") or message.get("updates") or []
        if not isinstance(changes, list):
            raise ValueError("Level 2 update missing changes list")
        for change in changes:
            if not _is_mapping(change):
                raise ValueError("Level 2 update change must be an object")
            price = change.get("price")
            size = change.get("new_quantity", change.get("quantity", change.get("size")))
            if price is None or size is None:
                raise ValueError(f"Malformed Level 2 update: {change!r}")
            side = _normalize_side(change.get("side"))
            price_value = float(price)
            quantity_value = float(size)
            if not (price_value > 0 and quantity_value >= 0 and price_value == price_value):
                raise ValueError(f"Invalid Level 2 update: {change!r}")
            updates.append({"side": side, "price": price_value, "quantity": quantity_value})
    return {
        "type": event_type,
        "product_id": product_id,
        "sequence_num": sequence_num,
        "envelope_time_utc": parsed_time.astimezone(timezone.utc),
        "event_time_utc": event_time_utc,
        "updates": updates,
    }


def parse_heartbeat_message(message: Mapping[str, Any]) -> dict[str, Any]:
    if not _is_mapping(message):
        raise ValueError("Heartbeat message must be a mapping")
    if message.get("type") != "heartbeat":
        raise ValueError("Not a heartbeat message")
    heartbeat_counter = message.get("sequence")
    if heartbeat_counter is not None:
        try:
            heartbeat_counter = int(heartbeat_counter)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Malformed heartbeat sequence: {heartbeat_counter!r}") from exc
    server_time = message.get("time") or message.get("timestamp")
    if server_time is not None:
        try:
            server_time = datetime.fromisoformat(str(server_time).replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError as exc:
            raise ValueError(f"Malformed heartbeat timestamp: {server_time!r}") from exc
    return {
        "type": "heartbeat",
        "sequence": heartbeat_counter,
        "time_utc": server_time,
    }


def parse_coinbase_ws_message(raw: str | bytes | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(raw, (bytes, bytearray)):
        data = raw.decode("utf-8")
    elif isinstance(raw, Mapping):
        payload = dict(raw)
    else:
        data = str(raw)
    if isinstance(raw, (str, bytes, bytearray)):
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed websocket JSON payload: {raw!r}") from exc
    if not _is_mapping(payload):
        raise ValueError("WebSocket message payload is not a JSON object")
    message_type = payload.get("type")
    if message_type == "heartbeat":
        return parse_heartbeat_message(payload)
    if message_type in {"snapshot", "l2_data"}:
        return parse_level2_event(payload)
    return {"type": message_type, "payload": payload}


class CoinbaseWebSocketClient:
    """Small adapter around the public Coinbase websocket with dependency injection."""

    def __init__(
        self,
        *,
        config: CoinbaseWebSocketConfig | None = None,
        transport: WebSocketTransport | None = None,
        sleep_fn: Any | None = None,
        monotonic_fn: Any | None = None,
    ) -> None:
        self.config = config or CoinbaseWebSocketConfig()
        self.transport = transport
        self.sleep_fn = sleep_fn or time.sleep
        self.monotonic_fn = monotonic_fn or time.monotonic
        self._connected = False
        self._last_message_at = self.monotonic_fn()

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        if self.transport is None:
            raise RuntimeError("No websocket transport configured")
        if hasattr(self.transport, "connect"):
            self.transport.connect(url=self.config.url, connect_timeout=self.config.connect_timeout_seconds)
        self._connected = True
        self._last_message_at = self.monotonic_fn()

    def close(self) -> None:
        if self.transport is not None and hasattr(self.transport, "close"):
            self.transport.close()
        self._connected = False

    def subscribe(self, *, product_id: str | None = None) -> None:
        if self.transport is None:
            raise RuntimeError("No websocket transport configured")
        payload = {
            "type": "subscribe",
            "product_ids": [product_id or self.config.product_id],
            "channel": "level2",
        }
        self.transport.send(json.dumps(payload))
        self.transport.send(json.dumps({"type": "subscribe", "channel": "heartbeats"}))

    def receive_message(self, *, timeout: float | None = None) -> CoinbaseWebSocketFrame:
        if self.transport is None:
            raise RuntimeError("No websocket transport configured")
        raw = self.transport.recv(timeout=timeout if timeout is not None else self.config.receive_timeout_seconds)
        payload = parse_coinbase_ws_message(raw)
        self._last_message_at = self.monotonic_fn()
        return CoinbaseWebSocketFrame(raw=str(raw), message=payload, received_at_utc=_utcnow())


__all__ = [
    "CoinbaseWebSocketClient",
    "CoinbaseWebSocketConfig",
    "CoinbaseWebSocketFrame",
    "parse_coinbase_ws_message",
    "parse_heartbeat_message",
    "parse_level2_event",
]
