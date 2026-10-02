from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CoinbaseRESTConfig:
    base_url: str = "https://api.coinbase.com"
    product_id: str = "BTC-USD"
    candle_granularity: str = "ONE_MINUTE"
    timeout_seconds: float = 30.0
    max_retries: int = 5
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 30.0
    user_agent: str = "btc-quarter-hour-engine/0.1.0"


@dataclass(slots=True)
class CoinbaseWebSocketConfig:
    url: str = "wss://advanced-trade-ws.coinbase.com"
    product_id: str = "BTC-USD"
    max_message_size_bytes: int = 16 * 1024 * 1024

    heartbeat_timeout_seconds: float = 5.0

    connect_timeout_seconds: float = 15.0
    receive_timeout_seconds: float = 5.0

    # Bounds how long reconnect() actively receives messages while waiting
    # for a fresh snapshot to arrive on a newly (re)subscribed connection,
    # before giving up and backing off for another connect attempt.
    snapshot_wait_timeout_seconds: float = 10.0

    initial_reconnect_backoff_seconds: float = 1.0
    max_reconnect_backoff_seconds: float = 30.0
    max_reconnect_attempts: int | None = None

    raw_segment_max_frames: int = 10000

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_message_size_bytes, bool)
            or not isinstance(self.max_message_size_bytes, int)
            or self.max_message_size_bytes <= 0
        ):
            raise ValueError("max_message_size_bytes must be a positive integer")
