from datetime import datetime, timezone

import httpx
import pytest

from btc_quarter_hour_engine.acquisition.coinbase_rest import (
    BOOK_PATH, CANDLES_PATH, PRODUCT_PATH, TICKER_PATH, TIME_PATH,
    CoinbasePublicRESTClient, _unix_seconds, parse_product_book_response,
)
from btc_quarter_hour_engine.acquisition.config import CoinbaseRESTConfig


def _client(handler, *, retries=2, sleep=None):
    transport = httpx.MockTransport(handler)
    return CoinbasePublicRESTClient(
        config=CoinbaseRESTConfig(max_retries=retries, initial_backoff_seconds=0.01),
        client=httpx.Client(transport=transport, base_url="https://api.coinbase.com"),
        sleep_fn=sleep or (lambda _: None),
    )


def test_current_public_paths_headers_and_unix_candle_params():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"candles": [], "trades": []})

    client = _client(handler)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 0, 2, tzinfo=timezone.utc)
    client.fetch_server_time()
    client.fetch_product("BTC-USD")
    result = client.fetch_candles("BTC-USD", start, end)
    client.fetch_product_book("BTC-USD", limit=5)
    client.fetch_market_trades("BTC-USD", limit=7)
    assert [request.url.path for request in requests] == [
        TIME_PATH, PRODUCT_PATH.format(product_id="BTC-USD"),
        CANDLES_PATH.format(product_id="BTC-USD"), BOOK_PATH,
        TICKER_PATH.format(product_id="BTC-USD"),
    ]
    assert result.request_params == {
        "start": str(int(start.timestamp())), "end": str(int(end.timestamp()) - 1),
        "granularity": "ONE_MINUTE",
    }
    assert requests[3].url.params["product_id"] == "BTC-USD"
    assert requests[3].url.params["limit"] == "5"
    assert requests[4].url.params["limit"] == "7"
    assert requests[0].headers["accept"] == "application/json"
    assert "btc-quarter-hour-engine" in requests[0].headers["user-agent"]
    client.close()


def test_unix_seconds_requires_aware_datetime_and_normalizes_offset():
    with pytest.raises(ValueError, match="Timezone-aware"):
        _unix_seconds(datetime(2024, 1, 1))
    assert _unix_seconds(datetime(2024, 1, 1, tzinfo=timezone.utc)) == "1704067200"


@pytest.mark.parametrize("status", [429, 503])
def test_retryable_http_status(status):
    calls = []
    delays = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, headers={"Retry-After": "2"}) if len(calls) == 1 else httpx.Response(200, json={})

    client = _client(handler, sleep=delays.append)
    client.get_product("BTC-USD")
    assert len(calls) == 2
    assert delays == [2]


@pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ReadTimeout])
def test_transient_transport_error_retried(error):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise error("transient", request=request)
        return httpx.Response(200, json={})

    _client(handler).get_product("BTC-USD")
    assert len(calls) == 2


def test_exhaustion_and_permanent_error():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, json={})

    with pytest.raises(httpx.HTTPStatusError):
        _client(handler, retries=1).get_product("BTC-USD")
    assert len(calls) == 2
    calls.clear()
    with pytest.raises(httpx.HTTPStatusError):
        _client(lambda request: handler_400(calls, request)).get_product("BTC-USD")
    assert len(calls) == 1


def handler_400(calls, request):
    calls.append(request)
    return httpx.Response(400, json={})


def test_exact_response_bytes_and_malformed_json():
    raw = b'{ "candles" : [ { "start":"1", "open":"1.00" } ] }'
    client = _client(lambda request: httpx.Response(200, content=raw, headers={"Content-Type": "application/json"}))
    result = client.fetch_candles("BTC-USD", datetime(2024, 1, 1, tzinfo=timezone.utc),
                                  datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc))
    assert result.response_bytes == raw
    assert result.content_type == "application/json"
    assert result.status_code == 200
    assert result.retrieved_at_utc.tzinfo is timezone.utc
    bad = _client(lambda request: httpx.Response(200, text="not-json"))
    with pytest.raises(ValueError, match="Malformed JSON"):
        bad.get_product("BTC-USD")


BOOK = {
    "pricebook": {
        "product_id": "BTC-USD",
        "bids": [{"price": "100.0", "size": "1.2"}],
        "asks": [{"price": "101.0", "size": "1.5"}],
        "time": "2026-01-01T00:00:00Z",
    }
}


def test_current_book_shape():
    book = parse_product_book_response(BOOK)
    assert book["product_id"] == "BTC-USD"
    assert book["source_time_utc"] == datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert (book["best_bid"], book["best_bid_size"], book["best_ask"], book["best_ask_size"]) == (100, 1.2, 101, 1.5)
    assert book["spread"] == 1
    assert book["midpoint"] == 100.5


@pytest.mark.parametrize("side,field,value", [
    ("bids", "price", "102"), ("bids", "price", "0"), ("asks", "price", "-1"),
    ("bids", "size", "-1"), ("asks", "size", "nan"),
    ("bids", "price", "not-a-number"), ("asks", "size", "not-a-number"),
])
def test_invalid_book_levels_rejected(side, field, value):
    from copy import deepcopy
    payload = deepcopy(BOOK)
    payload["pricebook"][side][0][field] = value
    with pytest.raises(ValueError):
        parse_product_book_response(payload)


@pytest.mark.parametrize("side", ["bids", "asks"])
def test_missing_book_side_rejected(side):
    from copy import deepcopy
    payload = deepcopy(BOOK)
    payload["pricebook"][side] = []
    with pytest.raises(ValueError):
        parse_product_book_response(payload)


def test_invalid_secondary_book_level_and_unsafe_product_rejected():
    from copy import deepcopy
    payload = deepcopy(BOOK)
    payload["pricebook"]["bids"].append({"price": "99", "size": "inf"})
    with pytest.raises(ValueError):
        parse_product_book_response(payload)
    with pytest.raises(ValueError, match="safe Coinbase"):
        _client(lambda request: httpx.Response(200, json={})).fetch_product("../BTC-USD")
