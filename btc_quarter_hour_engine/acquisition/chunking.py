from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable


_SUPPORTED_GRANULARITIES = {
    "ONE_MINUTE": 60,
    "FIVE_MINUTE": 300,
    "FIFTEEN_MINUTE": 900,
    "THIRTY_MINUTE": 1800,
    "ONE_HOUR": 3600,
    "SIX_HOUR": 21600,
    "ONE_DAY": 86400,
}


@dataclass(frozen=True, slots=True)
class CandleChunk:
    chunk_number: int
    requested_start: datetime
    requested_end: datetime


def _coerce_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        raise ValueError("Timezone-aware datetime required; pass UTC or an offset-aware timestamp.")
    return dt.astimezone(timezone.utc)


def _normalize_granularity(granularity: str) -> str:
    normalized = str(granularity).upper().strip()
    if normalized not in _SUPPORTED_GRANULARITIES:
        raise ValueError(f"Unsupported candle granularity: {granularity!r}")
    return normalized


def build_candle_chunk_plan(
    *,
    start: datetime,
    end: datetime,
    granularity: str,
    max_candles_per_request: int = 300,
) -> list[CandleChunk]:
    """Plan deterministic, non-overlapping candle chunks for a requested date range."""
    start_utc = _coerce_utc(start)
    end_utc = _coerce_utc(end)
    if end_utc <= start_utc:
        raise ValueError("Requested candle range must satisfy end > start.")

    granularity_name = _normalize_granularity(granularity)
    chunk_seconds = _SUPPORTED_GRANULARITIES[granularity_name]
    if max_candles_per_request <= 0:
        raise ValueError("max_candles_per_request must be positive.")

    max_window_seconds = max_candles_per_request * chunk_seconds
    if start_utc.timestamp() % chunk_seconds or end_utc.timestamp() % chunk_seconds:
        raise ValueError("Candle range boundaries must align with the requested granularity.")
    chunks: list[CandleChunk] = []
    cursor = start_utc
    chunk_number = 1
    while cursor < end_utc:
        chunk_end = min(cursor + timedelta(seconds=max_window_seconds), end_utc)
        chunks.append(
            CandleChunk(
                chunk_number=chunk_number,
                requested_start=cursor,
                requested_end=chunk_end,
            )
        )
        cursor = chunk_end
        chunk_number += 1

    return chunks


def iter_candle_chunks(*, start: datetime, end: datetime, granularity: str, max_candles_per_request: int = 300) -> Iterable[CandleChunk]:
    return iter(build_candle_chunk_plan(
        start=start,
        end=end,
        granularity=granularity,
        max_candles_per_request=max_candles_per_request,
    ))
