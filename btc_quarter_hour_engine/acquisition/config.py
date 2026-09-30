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
