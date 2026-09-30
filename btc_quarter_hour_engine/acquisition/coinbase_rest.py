from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable
from urllib.parse import quote

import httpx
import pandas as pd

from .chunking import _SUPPORTED_GRANULARITIES, _coerce_utc, _normalize_granularity
from .config import CoinbaseRESTConfig

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
TIME_PATH = "/api/v3/brokerage/time"
PRODUCT_PATH = "/api/v3/brokerage/market/products/{product_id}"
CANDLES_PATH = PRODUCT_PATH + "/candles"
BOOK_PATH = "/api/v3/brokerage/market/product_book"
TICKER_PATH = PRODUCT_PATH + "/ticker"


def validate_product_id(product_id: str) -> str:
    if not isinstance(product_id, str) or not re.fullmatch(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+", product_id):
        raise ValueError("product_id must be a safe Coinbase product identifier (for example BTC-USD)")
    return product_id


def _product_path(template: str, product_id: str) -> str:
    validate_product_id(product_id)
    return template.format(product_id=quote(product_id, safe="-"))


def _unix_seconds(value: datetime) -> str:
    utc = _coerce_utc(value)
    if utc.microsecond:
        raise ValueError("Unix timestamp must have whole-second precision")
    return str(int(utc.timestamp()))


@dataclass(frozen=True, slots=True)
class CoinbaseHTTPResult:
    response_bytes: bytes
    json_payload: Any
    status_code: int
    content_type: str | None
    retrieved_at_utc: datetime
    request_path: str
    request_params: dict[str, Any]


class CoinbasePublicRESTClient:
    """Public Advanced Trade transport with exact response-byte provenance."""

    def __init__(
        self,
        *,
        config: CoinbaseRESTConfig | None = None,
        client: httpx.Client | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self.config = config or CoinbaseRESTConfig()
        self.sleep_fn = sleep_fn or time.sleep
        self.client = client or httpx.Client(
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
            headers={"User-Agent": self.config.user_agent, "Accept": "application/json"},
        )

    def request(self, path: str, *, params: dict[str, Any] | None = None) -> CoinbaseHTTPResult:
        attempts = self.config.max_retries + 1
        if attempts < 1:
            raise ValueError("max_retries must be nonnegative")
        for attempt in range(1, attempts + 1):
            try:
                response = self.client.get(
                    path,
                    params=params,
                    headers={"User-Agent": self.config.user_agent, "Accept": "application/json"},
                )
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt == attempts:
                    raise
                self._sleep_before_retry(attempt)
                continue
            if response.status_code in RETRYABLE_STATUS_CODES and attempt < attempts:
                self._sleep_before_retry(attempt, response=response)
                continue
            response.raise_for_status()
            try:
                payload = response.json()
            except ValueError as exc:
                raise ValueError(f"Malformed JSON response from {path}") from exc
            return CoinbaseHTTPResult(
                response_bytes=response.content,
                json_payload=payload,
                status_code=response.status_code,
                content_type=response.headers.get("Content-Type"),
                retrieved_at_utc=datetime.now(timezone.utc),
                request_path=path,
                request_params=dict(params or {}),
            )
        raise RuntimeError(f"Request to {path} exhausted retries")

    def _sleep_before_retry(self, attempt: int, *, response: httpx.Response | None = None) -> None:
        delay = self.config.initial_backoff_seconds * 2 ** (attempt - 1)
        if response is not None and response.headers.get("Retry-After"):
            header = response.headers["Retry-After"]
            try:
                delay = max(0.0, float(header))
            except ValueError:
                try:
                    delay = max(0.0, (parsedate_to_datetime(header).astimezone(timezone.utc) -
                                      datetime.now(timezone.utc)).total_seconds())
                except (TypeError, OverflowError, ValueError):
                    pass
        if self.config.max_backoff_seconds is not None:
            delay = min(delay, self.config.max_backoff_seconds)
        self.sleep_fn(delay)

    def fetch_server_time(self) -> CoinbaseHTTPResult:
        return self.request(TIME_PATH)

    def get_server_time(self) -> dict[str, Any]:
        payload = self.fetch_server_time().json_payload
        if not isinstance(payload, dict):
            raise ValueError("server-time payload is not an object")
        return payload

    def fetch_product(self, product_id: str) -> CoinbaseHTTPResult:
        return self.request(_product_path(PRODUCT_PATH, product_id))

    def get_product(self, product_id: str) -> dict[str, Any]:
        payload = self.fetch_product(product_id).json_payload
        if not isinstance(payload, dict):
            raise ValueError("product payload is not an object")
        return payload

    def fetch_candles(
        self, product_id: str, start: datetime, end: datetime,
        granularity: str = "ONE_MINUTE", limit: int | None = None,
    ) -> CoinbaseHTTPResult:
        if _coerce_utc(end) <= _coerce_utc(start):
            raise ValueError("Candle end must be after start")
        params: dict[str, Any] = {
            "start": _unix_seconds(start),
            # Coinbase may include an end-boundary bucket; use the last second in [start, end).
            "end": str(int(_coerce_utc(end).timestamp()) - 1),
            "granularity": _normalize_granularity(granularity),
        }
        if limit is not None:
            params["limit"] = int(limit)
        return self.request(_product_path(CANDLES_PATH, product_id), params=params)

    def get_candles(
        self, product_id: str, start: datetime, end: datetime,
        granularity: str = "ONE_MINUTE", limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return parse_coinbase_candle_records(
            self.fetch_candles(product_id, start, end, granularity, limit).json_payload
        )

    def fetch_product_book(self, product_id: str, limit: int | None = None) -> CoinbaseHTTPResult:
        params: dict[str, Any] = {"product_id": validate_product_id(product_id)}
        if limit is not None:
            params["limit"] = int(limit)
        return self.request(BOOK_PATH, params=params)

    def get_product_book(self, product_id: str, limit: int | None = None) -> dict[str, Any]:
        return parse_product_book_response(self.fetch_product_book(product_id, limit).json_payload)

    def fetch_market_trades(self, product_id: str, limit: int | None = None) -> CoinbaseHTTPResult:
        params = {"limit": int(limit)} if limit is not None else None
        return self.request(_product_path(TICKER_PATH, product_id), params=params)

    def get_market_trades(self, product_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        payload = self.fetch_market_trades(product_id, limit).json_payload
        if not isinstance(payload, dict) or not isinstance(payload.get("trades"), list):
            raise ValueError("Ticker payload missing trades array")
        return payload["trades"]

    parse_candle_payload = staticmethod(lambda payload: parse_coinbase_candle_records(payload))
    parse_product_book_response = staticmethod(lambda payload: parse_product_book_response(payload))

    def close(self) -> None:
        self.client.close()


def _bucket_timestamp(value: Any) -> datetime:
    if isinstance(value, str) and value.isdecimal() or isinstance(value, int):
        try:
            return datetime.fromtimestamp(int(value), timezone.utc)
        except (OverflowError, OSError, ValueError) as exc:
            raise ValueError(f"Invalid candle timestamp: {value!r}") from exc
    if isinstance(value, datetime):
        return _coerce_utc(value)
    if isinstance(value, str):
        try:
            return _coerce_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError as exc:
            raise ValueError(f"Invalid candle timestamp: {value!r}") from exc
    raise ValueError(f"Invalid candle timestamp: {value!r}")


def parse_coinbase_candle_records(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("candles"), list):
        raise ValueError("Candle response missing candles array")
    records = []
    for index, item in enumerate(payload["candles"]):
        if not isinstance(item, dict):
            raise ValueError(f"Candle #{index} is not an object")
        try:
            records.append({
                "bucket_start": _bucket_timestamp(item["start"]),
                **{name: float(item[name]) for name in ("open", "high", "low", "close", "volume")},
            })
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid candle #{index}: {exc}") from exc
    return sorted(records, key=lambda record: record["bucket_start"])


def validate_candle_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        raise TypeError("records must be a list")
    validated: list[dict[str, Any]] = []
    seen: set[datetime] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Candle #{index} is not an object")
        try:
            bucket = _bucket_timestamp(record["bucket_start"])
            values = {name: float(record[name]) for name in ("open", "high", "low", "close", "volume")}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid candle #{index}: {exc}") from exc
        if bucket in seen:
            raise ValueError(f"Duplicate bucket timestamp: {bucket.isoformat()}")
        seen.add(bucket)
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"Candle #{index} has non-finite OHLCV")
        if any(values[name] <= 0 for name in ("open", "high", "low", "close")):
            raise ValueError(f"Candle #{index} has nonpositive OHLC")
        if values["volume"] < 0:
            raise ValueError(f"Candle #{index} has negative volume")
        if values["high"] < values["low"] or values["high"] < max(values["open"], values["close"]) or \
                values["low"] > min(values["open"], values["close"]):
            raise ValueError(f"Candle #{index} has inconsistent OHLC")
        validated.append({"bucket_start": bucket, **values})
    return sorted(validated, key=lambda record: record["bucket_start"])


def parse_product_book_response(payload: Any) -> dict[str, Any]:
    book = payload.get("pricebook") if isinstance(payload, dict) else None
    if not isinstance(book, dict):
        raise ValueError("Product book response missing pricebook")
    if not isinstance(book.get("bids"), list) or not book["bids"]:
        raise ValueError("Product book missing bids")
    if not isinstance(book.get("asks"), list) or not book["asks"]:
        raise ValueError("Product book missing asks")
    try:
        levels = {}
        for side in ("bids", "asks"):
            levels[side] = []
            for level in book[side]:
                price, size = float(level["price"]), float(level["size"])
                if not math.isfinite(price) or not math.isfinite(size) or price <= 0 or size < 0:
                    raise ValueError(f"Invalid {side} level price or size")
                levels[side].append((price, size))
        best_bid, bid_size = max(levels["bids"], key=lambda level: level[0])
        best_ask, ask_size = min(levels["asks"], key=lambda level: level[0])
        source_time = _bucket_timestamp(book["time"])
        product_id = book["product_id"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid product book: {exc}") from exc
    if not isinstance(product_id, str) or not product_id:
        raise ValueError("Invalid product book product_id")
    if best_bid > best_ask:
        raise ValueError("Product book has invalid or crossed quotes")
    return {
        "product_id": product_id, "source_time_utc": source_time,
        "best_bid": best_bid, "best_bid_size": bid_size,
        "best_ask": best_ask, "best_ask_size": ask_size,
        "spread": best_ask - best_bid, "midpoint": (best_bid + best_ask) / 2,
    }


def coverage_diagnostics_for_candles(
    *, records: list[dict[str, Any]], start: datetime, end: datetime, granularity: str,
) -> dict[str, Any]:
    start_utc, end_utc = _coerce_utc(start), _coerce_utc(end)
    step = _SUPPORTED_GRANULARITIES[_normalize_granularity(granularity)]
    if end_utc <= start_utc or start_utc.timestamp() % step or end_utc.timestamp() % step:
        raise ValueError("Coverage interval must be aligned and nonempty")
    expected = pd.date_range(start_utc, end_utc, freq=f"{step}s", inclusive="left")
    observed = [pd.Timestamp(_bucket_timestamp(record["bucket_start"])) for record in records]
    expected_set, observed_set = set(expected), set(observed)
    if observed_set - expected_set:
        raise ValueError("Returned candle timestamp outside requested interval or granularity")
    missing = sorted(expected_set - observed_set)
    return {
        "expected_bucket_count": len(expected),
        "observed_bucket_count": len(observed_set),
        "missing_bucket_count": len(missing),
        "duplicate_bucket_count": len(observed) - len(observed_set),
        "first_bucket": min(observed).isoformat() if observed else None,
        "last_bucket": max(observed).isoformat() if observed else None,
        "coverage_fraction": len(observed_set) / len(expected),
        "missing_timestamps": [value.isoformat() for value in missing[:50]],
    }
