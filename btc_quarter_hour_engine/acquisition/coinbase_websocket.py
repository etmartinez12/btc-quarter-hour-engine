from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from .config import CoinbaseWebSocketConfig


class WebSocketTransport(Protocol):
    def send(self, payload: str) -> None: ...
    def recv(self, timeout: float | None = None) -> str | bytes: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Real Coinbase Advanced Trade websocket envelope
#
# Every server -> client message shares one outer envelope: a `channel`
# identifying the subscription ("l2_data", "heartbeats", "subscriptions",
# ...), a connection-scoped monotonically increasing `sequence_num`, a
# server-assigned `timestamp`, and a list of channel-specific `events`. See:
# https://docs.cdp.coinbase.com/api-reference/advanced-trade-api/websocket/level2
#
#   {
#     "channel": "l2_data",
#     "client_id": "",
#     "timestamp": "2023-02-09T20:32:50.714964855Z",
#     "sequence_num": 0,
#     "events": [
#       {
#         "type": "snapshot",
#         "product_id": "BTC-USD",
#         "updates": [
#           {"side": "bid", "event_time": "1970-01-01T00:00:00Z",
#            "price_level": "21921.73", "new_quantity": "0.30000000"}
#         ]
#       }
#     ]
#   }
#
# Both "snapshot" and "update" events share the exact same `updates` shape
# (there is no separate `bids`/`asks`/`changes` list as in the legacy
# Coinbase Exchange feed); `side` is "bid" or "offer" on the wire (not
# "ask"). Heartbeats also carry a separate `heartbeat_counter` per event;
# this counter is distinct from the shared envelope sequence.
# ---------------------------------------------------------------------------

CHANNEL_LEVEL2 = "l2_data"
CHANNEL_HEARTBEATS = "heartbeats"
CHANNEL_SUBSCRIPTIONS = "subscriptions"

LEVEL2_EVENT_TYPE_SNAPSHOT = "snapshot"
LEVEL2_EVENT_TYPE_UPDATE = "update"

# Internal normalized event-type tags used by market_data.order_book /
# market_data.replay / acquisition.websocket_service. These are decoupled
# from Coinbase's own wire-level event `type` (see `_flatten_level2_envelope`)
# so downstream book/replay logic never has to know about envelope framing.
NORMALIZED_TYPE_SNAPSHOT = "snapshot"
NORMALIZED_TYPE_LEVEL2_UPDATE = "l2_data"
NORMALIZED_TYPE_HEARTBEAT = "heartbeat"

# Coinbase's wire-level `side` uses "offer" for the ask side; "buy"/"sell"
# aliases are also accepted defensively even though the documented schema
# only emits "bid"/"offer".
_SIDE_ALIASES = {"bid": "bid", "buy": "bid", "offer": "ask", "ask": "ask", "sell": "ask"}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_side(value: Any) -> str:
    normalized = str(value).lower()
    if normalized not in _SIDE_ALIASES:
        raise ValueError(f"Unsupported side: {value!r}")
    return _SIDE_ALIASES[normalized]


def _require_utc_datetime(value: Any, *, field_name: str) -> datetime:
    if value is None:
        raise ValueError(f"Missing {field_name}")
    text = str(value).strip()
    go_timestamp = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.(\d{1,9}) ([+-]\d{4}) UTC m=[+-]\d+(?:\.\d+)?",
        text,
    )
    if go_timestamp:
        wall_time, fraction, offset = go_timestamp.groups()
        microseconds = fraction[:6].ljust(6, "0")
        offset = f"{offset[:3]}:{offset[3:]}"
        try:
            parsed = datetime.fromisoformat(f"{wall_time.replace(' ', 'T')}.{microseconds}{offset}")
        except ValueError as exc:
            raise ValueError(f"Malformed {field_name}: {value!r}") from exc
        return parsed.astimezone(timezone.utc)
    normalized = text.replace("Z", "+00:00")
    if " " in normalized and "T" not in normalized:
        # Coinbase heartbeats' `current_time` uses a space-separated
        # "YYYY-MM-DD HH:MM:SS.ffffff" form rather than ISO-8601 with "T".
        normalized = normalized.replace(" ", "T", 1)
    iso_fraction = re.fullmatch(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.(\d{7,})(.*)", normalized)
    if iso_fraction:
        wall_time, fraction, suffix = iso_fraction.groups()
        normalized = f"{wall_time}.{fraction[:6]}{suffix}"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"Malformed {field_name}: {value!r}") from exc
    if parsed.tzinfo is None:
        # Heartbeat `current_time` is documented without an explicit offset;
        # Coinbase always reports it in UTC.
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _optional_utc_datetime(value: Any, *, field_name: str) -> datetime | None:
    if value is None:
        return None
    return _require_utc_datetime(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class Level2UpdateEntry:
    """One book mutation within an `l2_data` event's `updates` array."""

    side: str
    price_level: float
    new_quantity: float
    event_time_utc: datetime | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "price": self.price_level,
            "quantity": self.new_quantity,
            "event_time_utc": self.event_time_utc,
        }


@dataclass(frozen=True, slots=True)
class Level2Event:
    """One `events[]` entry of a `channel: "l2_data"` message."""

    type: str  # "snapshot" | "update"
    product_id: str
    updates: tuple[Level2UpdateEntry, ...]


@dataclass(frozen=True, slots=True)
class HeartbeatEvent:
    """One `events[]` entry of a `channel: "heartbeats"` message."""

    current_time_utc: datetime | None
    heartbeat_counter: int | None


@dataclass(frozen=True, slots=True)
class CoinbaseMessageEnvelope:
    """The outer envelope shared by every Coinbase Advanced Trade websocket message."""

    channel: str
    client_id: str | None
    timestamp_utc: datetime
    sequence_num: int
    events: tuple[Any, ...]


def _parse_level2_update_entry(entry: Any) -> Level2UpdateEntry:
    if not isinstance(entry, Mapping):
        raise ValueError(f"Level 2 update entry must be an object: {entry!r}")
    price = entry.get("price_level")
    quantity = entry.get("new_quantity")
    if price is None or quantity is None:
        raise ValueError(f"Level 2 update entry missing price_level/new_quantity: {entry!r}")
    try:
        price_value = float(price)
        quantity_value = float(quantity)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Malformed Level 2 update entry: {entry!r}") from exc
    if not (math.isfinite(price_value) and math.isfinite(quantity_value) and price_value > 0 and quantity_value >= 0):
        raise ValueError(f"Invalid Level 2 update entry: {entry!r}")
    side = _normalize_side(entry.get("side"))
    event_time = _optional_utc_datetime(entry.get("event_time"), field_name="update event_time")
    return Level2UpdateEntry(side=side, price_level=price_value, new_quantity=quantity_value, event_time_utc=event_time)


def parse_level2_event(event: Any) -> Level2Event:
    """Parse one `events[]` entry of a `channel: "l2_data"` message."""
    if not isinstance(event, Mapping):
        raise ValueError(f"Level 2 event must be a mapping: {event!r}")
    event_type = event.get("type")
    if event_type not in {LEVEL2_EVENT_TYPE_SNAPSHOT, LEVEL2_EVENT_TYPE_UPDATE}:
        raise ValueError(f"Unsupported Level 2 event type: {event_type!r}")
    product_id = event.get("product_id")
    if not isinstance(product_id, str) or not product_id:
        raise ValueError("Level 2 event missing product_id")
    updates_payload = event.get("updates")
    if not isinstance(updates_payload, list) or not updates_payload:
        raise ValueError("Level 2 event missing non-empty updates list")
    updates = tuple(_parse_level2_update_entry(entry) for entry in updates_payload)
    return Level2Event(type=event_type, product_id=product_id, updates=updates)


def parse_heartbeat_event(event: Any) -> HeartbeatEvent:
    """Parse one `events[]` entry of a `channel: "heartbeats"` message."""
    if not isinstance(event, Mapping):
        raise ValueError(f"Heartbeat event must be a mapping: {event!r}")
    counter = event.get("heartbeat_counter")
    if counter is not None:
        try:
            counter = int(counter)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Malformed heartbeat_counter: {counter!r}") from exc
    current_time = _optional_utc_datetime(event.get("current_time"), field_name="heartbeat current_time")
    return HeartbeatEvent(current_time_utc=current_time, heartbeat_counter=counter)


def parse_coinbase_envelope(payload: Mapping[str, Any]) -> CoinbaseMessageEnvelope:
    """Parse one already-JSON-decoded message into the real Advanced Trade envelope."""
    if not isinstance(payload, Mapping):
        raise ValueError("WebSocket message payload is not a JSON object")
    channel = payload.get("channel")
    if not isinstance(channel, str) or not channel:
        raise ValueError(f"WebSocket message missing channel: {payload!r}")
    sequence_num = payload.get("sequence_num")
    try:
        sequence_num = int(sequence_num)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Malformed sequence_num: {sequence_num!r}") from exc
    if sequence_num < 0:
        raise ValueError(f"Negative sequence_num: {sequence_num!r}")
    timestamp_utc = _require_utc_datetime(payload.get("timestamp"), field_name="envelope timestamp")
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise ValueError("WebSocket message missing events list")
    events: tuple[Any, ...]
    if channel == CHANNEL_LEVEL2:
        events = tuple(parse_level2_event(event) for event in raw_events)
    elif channel == CHANNEL_HEARTBEATS:
        events = tuple(parse_heartbeat_event(event) for event in raw_events)
    else:
        events = tuple(raw_events)
    return CoinbaseMessageEnvelope(
        channel=channel,
        client_id=payload.get("client_id"),
        timestamp_utc=timestamp_utc,
        sequence_num=sequence_num,
        events=events,
    )


def _flatten_level2_envelope(envelope: CoinbaseMessageEnvelope) -> dict[str, Any]:
    if not envelope.events:
        raise ValueError("l2_data message contains no events")
    first: Level2Event = envelope.events[0]
    event_type = first.type
    product_id = first.product_id
    updates: list[dict[str, Any]] = []
    event_times: list[datetime] = []
    for event in envelope.events:
        if event.type != event_type:
            raise ValueError("l2_data message mixes snapshot and update event types")
        if event.product_id != product_id:
            raise ValueError("l2_data message mixes multiple product_ids")
        for update in event.updates:
            updates.append(update.as_dict())
            if update.event_time_utc is not None:
                event_times.append(update.event_time_utc)
    normalized_type = NORMALIZED_TYPE_SNAPSHOT if event_type == LEVEL2_EVENT_TYPE_SNAPSHOT else NORMALIZED_TYPE_LEVEL2_UPDATE
    # Coinbase's own per-update `event_time` is a known-unreliable placeholder
    # for snapshot events (historically pinned to the Unix epoch), so
    # snapshot event time always uses the envelope `timestamp` instead. For
    # incremental updates we prefer the latest per-update `event_time` when
    # present, falling back to the envelope timestamp otherwise.
    if normalized_type == NORMALIZED_TYPE_SNAPSHOT or not event_times:
        event_time_utc = envelope.timestamp_utc
    else:
        event_time_utc = max(event_times)
    return {
        "type": normalized_type,
        "channel": envelope.channel,
        "product_id": product_id,
        "sequence_num": envelope.sequence_num,
        "envelope_time_utc": envelope.timestamp_utc,
        "event_time_utc": event_time_utc,
        "updates": updates,
        "envelope": envelope,
    }


def _flatten_heartbeat_envelope(envelope: CoinbaseMessageEnvelope) -> dict[str, Any]:
    heartbeat_counter: int | None = None
    current_time_utc: datetime | None = None
    if envelope.events:
        first: HeartbeatEvent = envelope.events[0]
        heartbeat_counter = first.heartbeat_counter
        current_time_utc = first.current_time_utc
    return {
        "type": NORMALIZED_TYPE_HEARTBEAT,
        "channel": envelope.channel,
        "sequence": heartbeat_counter,
        "time_utc": current_time_utc or envelope.timestamp_utc,
        "envelope": envelope,
    }


def parse_coinbase_ws_message(raw: str | bytes | Mapping[str, Any]) -> dict[str, Any]:
    """Parse one raw websocket frame into the normalized internal event dict.

    Accepts the exact bytes/text received over the wire (or an already
    JSON-decoded mapping), validates it against the real Coinbase Advanced
    Trade envelope (see module docstring), and flattens it into the internal
    shape consumed by ``market_data.order_book``/``market_data.replay``:
    ``{"type", "product_id", "sequence_num", "envelope_time_utc",
    "event_time_utc", "updates"}`` for level2 events, or
    ``{"type": "heartbeat", "sequence", "time_utc"}`` for heartbeats. The
    parsed ``CoinbaseMessageEnvelope`` (with full per-update fidelity,
    including individual ``event_time``s) is always attached under the
    ``"envelope"`` key for callers that need full wire fidelity (e.g.
    normalized ``level2_updates`` storage).
    """
    if isinstance(raw, (bytes, bytearray)):
        data = raw.decode("utf-8")
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed websocket JSON payload: {raw!r}") from exc
    elif isinstance(raw, Mapping):
        payload = dict(raw)
    else:
        data = str(raw)
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed websocket JSON payload: {raw!r}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("WebSocket message payload is not a JSON object")
    envelope = parse_coinbase_envelope(payload)
    if envelope.channel == CHANNEL_LEVEL2:
        return _flatten_level2_envelope(envelope)
    if envelope.channel == CHANNEL_HEARTBEATS:
        return _flatten_heartbeat_envelope(envelope)
    return {"type": envelope.channel, "channel": envelope.channel, "envelope": envelope, "payload": dict(payload)}


@dataclass(frozen=True, slots=True)
class CoinbaseWebSocketFrame:
    raw_bytes: bytes
    message: dict[str, Any] | None
    received_at_utc: datetime
    connection_id: str | None = None
    parse_error: str | None = None

    @property
    def raw(self) -> bytes:
        return self.raw_bytes


class CoinbaseWebSocketClient:
    """Small adapter around the public Coinbase websocket with dependency injection."""

    def __init__(
        self,
        *,
        config: CoinbaseWebSocketConfig | None = None,
        transport: WebSocketTransport | None = None,
        sleep_fn: Any | None = None,
        monotonic_fn: Any | None = None,
        now_fn: Any | None = None,
    ) -> None:
        self.config = config or CoinbaseWebSocketConfig()
        self.transport = transport
        self.sleep_fn = sleep_fn or time.sleep
        self.monotonic_fn = monotonic_fn or time.monotonic
        self.now_fn = now_fn or _utcnow
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
        """Receive exactly one raw frame and attempt to parse it.

        The raw bytes/text are always preserved on the returned frame,
        *independent of whether parsing succeeds* -- a malformed frame still
        comes back as a frame (with ``message=None`` and ``parse_error``
        set) rather than raising, so callers can seal the exact original
        bytes into an immutable raw segment before (and regardless of)
        attempting to interpret them.
        """
        if self.transport is None:
            raise RuntimeError("No websocket transport configured")
        raw = self.transport.recv(timeout=timeout if timeout is not None else self.config.receive_timeout_seconds)
        raw_bytes = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
        self._last_message_at = self.monotonic_fn()
        received_at_utc = self.now_fn()
        try:
            payload = parse_coinbase_ws_message(raw_bytes)
        except (ValueError, UnicodeError) as exc:
            return CoinbaseWebSocketFrame(raw_bytes=raw_bytes, message=None, received_at_utc=received_at_utc, parse_error=str(exc))
        return CoinbaseWebSocketFrame(raw_bytes=raw_bytes, message=payload, received_at_utc=received_at_utc)


__all__ = [
    "CHANNEL_HEARTBEATS",
    "CHANNEL_LEVEL2",
    "CHANNEL_SUBSCRIPTIONS",
    "CoinbaseMessageEnvelope",
    "CoinbaseWebSocketClient",
    "CoinbaseWebSocketConfig",
    "CoinbaseWebSocketFrame",
    "HeartbeatEvent",
    "LEVEL2_EVENT_TYPE_SNAPSHOT",
    "LEVEL2_EVENT_TYPE_UPDATE",
    "Level2Event",
    "Level2UpdateEntry",
    "NORMALIZED_TYPE_HEARTBEAT",
    "NORMALIZED_TYPE_LEVEL2_UPDATE",
    "NORMALIZED_TYPE_SNAPSHOT",
    "parse_coinbase_envelope",
    "parse_coinbase_ws_message",
    "parse_heartbeat_event",
    "parse_level2_event",
]
