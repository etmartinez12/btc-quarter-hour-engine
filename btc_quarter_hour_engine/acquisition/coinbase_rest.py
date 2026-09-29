from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Callable

import httpx
import pandas as pd

from .config import CoinbaseRESTConfig

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
PERMANENT_ERROR_CODES = {400, 401, 403, 404}


def _as_utc_iso(value: datetime) -> str:
    utc_value = value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return utc_value.isoformat().replace("+00:00", "Z")


class CoinbasePublicRESTClient:
    """Minimal Coinbase Advanced Trade public REST client with injectable HTTP transport."""

    def __init__(
        self,
        *,
        config: CoinbaseRESTConfig | None = None,
        client: httpx.Client | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self.config = config or CoinbaseRESTConfig()
        self.sleep_fn = sleep_fn or time.sleep
        if client is None:
            self.client = httpx.Client(
                base_url=self.config.base_url,
                timeout=self.config.timeout_seconds,
                follow_redirects=True,
                headers={"User-Agent": self.config.user_agent, "Accept": "application/json"},
            )
        else:
            self.client = client

    def _request_json(self, path: str, *, params: dict[str, Any] | None = None, method: str = "GET") -> Any:
        max_attempts = max(1, self.config.max_retries + 1)
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.client.request(
                    method,
                    path,
                    params=params,
                    headers={"User-Agent": self.config.user_agent, "Accept": "application/json"},
                )
            except httpx.HTTPError as exc:
                if attempt >= max_attempts or not self._is_retryable_transport_error(exc):
                    raise
                self._sleep_before_retry(attempt)
                continue

            if response.status_code in RETRYABLE_STATUS_CODES:
                if attempt >= max_attempts:
                    raise httpx.HTTPStatusError(
                        f"Retry budget exhausted for {path}: HTTP {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                self._sleep_before_retry(attempt, response=response)
                continue

            if response.status_code >= 400:
                if response.status_code in PERMANENT_ERROR_CODES:
                    raise httpx.HTTPStatusError(
                        f"Permanent HTTP error for {path}: {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                raise httpx.HTTPStatusError(
                    f"HTTP error for {path}: {response.status_code}",
                    request=response.request,
                    response=response,
                )

            try:
                payload = response.json()
            except ValueError as exc:  # pragma: no cover - defensive path
                raise ValueError(f"Malformed JSON response from {path}") from exc
            return payload

        raise RuntimeError(f"Request to {path} failed without producing a response")

    def _sleep_before_retry(self, attempt: int, *, response: httpx.Response | None = None) -> None:
        retry_after = None
        if response is not None:
            retry_after_header = response.headers.get("Retry-After")
            if retry_after_header:
                try:
                    retry_after = float(retry_after_header)
                except ValueError:
                    retry_after = None
        delay = retry_after if retry_after is not None else self.config.initial_backoff_seconds * (2 ** (attempt - 1))
        if self.config.max_backoff_seconds is not None:
            delay = min(delay, self.config.max_backoff_seconds)
        self.sleep_fn(delay)

    @staticmethod
    def _is_retryable_transport_error(exc: Exception) -> bool:
        return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError, httpx.NetworkError))

    def get_server_time(self) -> dict[str, Any]:
        payload = self._request_json("/v2/time")
        data = payload.get("data") if isinstance(payload, dict) else payload
        if isinstance(data, dict):
            return data
        raise ValueError("server-time payload is not a dictionary")

    def get_product(self, product_id: str) -> dict[str, Any]:
        payload = self._request_json(f"/products/{product_id}")
        if isinstance(payload, dict):
            return payload
        raise ValueError(f"Unexpected product response for {product_id!r}: {type(payload).__name__}")

    def get_candles(
        self,
        product_id: str,
        start: datetime,
        end: datetime,
        granularity: str = "ONE_MINUTE",
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "start": _as_utc_iso(start),
            "end": _as_utc_iso(end),
            "granularity": granularity,
        }
        if limit is not None:
            params["limit"] = int(limit)
        payload = self._request_json(f"/products/{product_id}/candles", params=params)
        candles = self.parse_candle_payload(payload)
        return candles

    def get_product_book(self, product_id: str, limit: int | None = None) -> dict[str, Any]:
        params = {"limit": int(limit)} if limit is not None else None
        payload = self._request_json(f"/products/{product_id}/book", params=params)
        return self.parse_product_book_response(payload)

    def get_market_trades(self, product_id: str, limit: int | None = None, after: int | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = int(limit)
        if after is not None:
            params["after"] = int(after)
        payload = self._request_json(f"/products/{product_id}/trades", params=params or None)
        return self.parse_trade_payload(payload)

    @staticmethod
    def parse_candle_payload(payload: Any) -> list[dict[str, Any]]:
        return parse_coinbase_candle_records(payload)

    @staticmethod
    def parse_product_book_response(payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            bids = payload.get("bids") or payload.get("best_bid_levels") or []
            asks = payload.get("asks") or payload.get("best_ask_levels") or []
            if not bids or not asks:
                candidates = payload.get("data")
                if isinstance(candidates, dict):
                    bids = candidates.get("bids") or []
                    asks = candidates.get("asks") or []
            if not bids or not asks:
                raise ValueError("Product book payload missing bid/ask levels")

            best_bid = float(bids[0][0]) if isinstance(bids[0], (list, tuple)) else float(bids[0]["price"])
            best_bid_size = float(bids[0][1]) if isinstance(bids[0], (list, tuple)) else float(bids[0].get("size", 0.0))
            best_ask = float(asks[0][0]) if isinstance(asks[0], (list, tuple)) else float(asks[0]["price"])
            best_ask_size = float(asks[0][1]) if isinstance(asks[0], (list, tuple)) else float(asks[0].get("size", 0.0))
            return {
                "best_bid": best_bid,
                "best_bid_size": best_bid_size,
                "best_ask": best_ask,
                "best_ask_size": best_ask_size,
                "spread": best_ask - best_bid,
                "midpoint": (best_bid + best_ask) / 2.0,
            }
        raise ValueError(f"Could not parse product book payload from {type(payload).__name__}")

    @staticmethod
    def parse_trade_payload(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            trades = payload.get("trades") or payload.get("data")
            if isinstance(trades, list):
                return trades
        if isinstance(payload, list):
            return payload
        raise ValueError(f"Could not parse market-trade payload from {type(payload).__name__}")

    def close(self) -> None:
        self.client.close()


def _extract_granularity_seconds(granularity: str) -> int:
    mapping = {
        "ONE_MINUTE": 60,
        "FIVE_MINUTE": 300,
        "FIFTEEN_MINUTE": 900,
        "THIRTY_MINUTE": 1800,
        "ONE_HOUR": 3600,
        "SIX_HOUR": 21600,
        "ONE_DAY": 86400,
    }
    normalized = str(granularity).upper()
    if normalized not in mapping:
        raise ValueError(f"Unsupported candle granularity: {granularity!r}")
    return mapping[normalized]


def parse_coinbase_candle_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        normalized = []
        for item in payload:
            if isinstance(item, (list, tuple)) and len(item) >= 6:
                timestamp, low, high, open_, close, volume = item[:6]
                normalized.append(
                    {
                        "time": timestamp,
                        "low": float(low),
                        "high": float(high),
                        "open": float(open_),
                        "close": float(close),
                        "volume": float(volume),
                    }
                )
        return normalized

    data = payload.get("data") if isinstance(payload, dict) else None
    if data is None and isinstance(payload, dict):
        data = payload.get("candles")
    if isinstance(data, list):
        records: list[dict[str, Any]] = []
        for item in data:
            if isinstance(item, dict):
                if "time" not in item and "bucket_start" in item:
                    item = {**item, "time": item["bucket_start"]}
                timestamp = item.get("time")
                record = {
                    "time": timestamp,
                    "low": float(item.get("low", 0.0)),
                    "high": float(item.get("high", 0.0)),
                    "open": float(item.get("open", 0.0)),
                    "close": float(item.get("close", 0.0)),
                    "volume": float(item.get("volume", 0.0)),
                }
                records.append(record)
        return records
    raise ValueError(f"Could not parse candle payload from response of type {type(payload).__name__}")


def validate_candle_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        raise TypeError("records must be a list of candle dictionaries")

    validated: list[dict[str, Any]] = []
    seen: set[datetime] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Candle record #{index} is not a dictionary")
        required = ["time", "open", "high", "low", "close", "volume"]
        missing = [name for name in required if name not in record]
        if missing:
            raise ValueError(f"Candle record #{index} missing required keys: {missing}")

        bucket = pd.to_datetime(record["time"], utc=True, errors="coerce")
        if pd.isna(bucket):
            raise ValueError(f"Candle record #{index} has invalid timestamp: {record['time']!r}")
        if bucket in seen:
            raise ValueError(f"Duplicate bucket timestamp in normalized candles: {bucket.isoformat()}")
        seen.add(bucket)

        values = {"open": float(record["open"]), "high": float(record["high"]), "low": float(record["low"]), "close": float(record["close"]), "volume": float(record["volume"])}
        if not all(pd.notna(pd.Series(list(values.values())))):
            raise ValueError(f"Candle record #{index} contains non-finite or NaN values")
        if values["open"] <= 0 or values["high"] <= 0 or values["low"] <= 0 or values["close"] <= 0:
            raise ValueError(f"Candle record #{index} contains nonpositive OHLC values")
        if values["volume"] < 0:
            raise ValueError(f"Candle record #{index} has negative volume")
        if values["high"] < values["low"]:
            raise ValueError(f"Candle record #{index} has high < low")
        if values["high"] < max(values["open"], values["close"]):
            raise ValueError(f"Candle record #{index} has inconsistent high relative to open/close")
        if values["low"] > min(values["open"], values["close"]):
            raise ValueError(f"Candle record #{index} has inconsistent low relative to open/close")

        validated.append({
            "time": bucket.isoformat().replace("+00:00", "Z"),
            "open": values["open"],
            "high": values["high"],
            "low": values["low"],
            "close": values["close"],
            "volume": values["volume"],
        })

    return sorted(validated, key=lambda x: x["time"])


def coverage_diagnostics_for_candles(
    *,
    records: list[dict[str, Any]],
    start: datetime,
    end: datetime,
    granularity: str,
) -> dict[str, Any]:
    start_utc = start.astimezone(timezone.utc) if start.tzinfo else start.replace(tzinfo=timezone.utc)
    end_utc = end.astimezone(timezone.utc) if end.tzinfo else end.replace(tzinfo=timezone.utc)
    step_seconds = _extract_granularity_seconds(granularity)
    expected_timestamps = pd.date_range(start=start_utc, end=end_utc, freq=pd.to_timedelta(step_seconds, unit="s"), tz="UTC")
    observed_timestamps = pd.DatetimeIndex(
        pd.to_datetime([record["time"] if isinstance(record.get("time"), str) else record["time"] for record in records], utc=True)
    )
    expected_keys = set(expected_timestamps)
    observed_keys = set(observed_timestamps)
    missing = sorted(expected_keys - observed_keys)
    duplicate_count = len(records) - len(observed_timestamps.drop_duplicates()) if len(records) else 0
    observed_bucket_count = len(observed_keys)
    coverage_fraction = (observed_bucket_count / len(expected_keys)) if expected_keys else 0.0
    return {
        "expected_bucket_count": len(expected_keys),
        "observed_bucket_count": observed_bucket_count,
        "missing_bucket_count": len(missing),
        "duplicate_bucket_count": duplicate_count,
        "first_bucket": expected_timestamps[0].isoformat() if len(expected_timestamps) else None,
        "last_bucket": expected_timestamps[-1].isoformat() if len(expected_timestamps) else None,
        "coverage_fraction": coverage_fraction,
        "missing_timestamps": [value.isoformat() for value in missing[:50]],
    }


def main() -> None:  # pragma: no cover
    print("Coinbase public REST client is available for acquisition utilities.")


__all__ = [
    "CoinbasePublicRESTClient",
    "CoinbaseRESTConfig",
    "RETRYABLE_STATUS_CODES",
    "PERMANENT_ERROR_CODES",
    "parse_coinbase_candle_records",
    "validate_candle_records",
    "coverage_diagnostics_for_candles",
    "main",
]
