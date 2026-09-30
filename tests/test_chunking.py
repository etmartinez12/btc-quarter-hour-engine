from datetime import datetime, timedelta, timezone

import pytest

from btc_quarter_hour_engine.acquisition.chunking import build_candle_chunk_plan

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def plan(minutes, limit=2):
    return build_candle_chunk_plan(
        start=START, end=START + timedelta(minutes=minutes), granularity="ONE_MINUTE",
        max_candles_per_request=limit,
    )


def test_one_chunk_and_multi_chunk_contiguous_deterministic():
    assert len(plan(1)) == 1
    chunks = plan(5)
    assert chunks == plan(5)
    assert [c.chunk_number for c in chunks] == [1, 2, 3]
    assert chunks[0].requested_start == START
    assert chunks[-1].requested_end == START + timedelta(minutes=5)
    assert all(left.requested_end == right.requested_start for left, right in zip(chunks, chunks[1:]))
    assert [int((c.requested_end - c.requested_start).total_seconds() / 60) for c in chunks] == [2, 2, 1]
    assert all(not hasattr(chunk, "endpoint") for chunk in chunks)


@pytest.mark.parametrize("overrides", [
    {"start": datetime(2026, 1, 1)},
    {"end": datetime(2026, 1, 1, 0, 5)},
    {"end": START}, {"end": START - timedelta(minutes=1)},
    {"granularity": "TEN_MINUTE"},
    {"max_candles_per_request": 0}, {"max_candles_per_request": -1},
    {"start": START + timedelta(seconds=1)},
])
def test_invalid_plan_rejected(overrides):
    kwargs = dict(start=START, end=START + timedelta(minutes=5),
                  granularity="ONE_MINUTE", max_candles_per_request=2)
    with pytest.raises(ValueError):
        build_candle_chunk_plan(**(kwargs | overrides))
