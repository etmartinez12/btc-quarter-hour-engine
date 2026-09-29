from __future__ import annotations

import json

import httpx
import pytest

from btc_quarter_hour_engine.acquisition.coinbase_rest import CoinbasePublicRESTClient
from btc_quarter_hour_engine.acquisition.config import CoinbaseRESTConfig


def _transport_factory(responses):
    def handler(request):
        if not responses:
            raise AssertionError("unexpected request")
        response = responses.pop(0)
        return httpx.Response(response[0], json=response[1]) if isinstance(response[1], dict) else httpx.Response(response[0], text=response[1])
    return handler


def test_client_retries_on_503_then_succeeds():
    sleep_calls = []
    responses = [
        (503, {"error": "temp"}),
        (503, {"error": "temp"}),
        (200, {"data": {"iso": "2024-01-01T00:00:00Z"}}),
    ]
    transport = httpx.MockTransport(_transport_factory(responses))
    client = CoinbasePublicRESTClient(
        config=CoinbaseRESTConfig(max_retries=3, initial_backoff_seconds=0.01, max_backoff_seconds=0.05),
        client=httpx.Client(transport=transport, base_url="https://api.coinbase.com"),
        sleep_fn=lambda delay: sleep_calls.append(delay),
    )

    result = client.get_server_time()

    assert result == {"iso": "2024-01-01T00:00:00Z"}
    assert len(sleep_calls) == 2


def test_client_raises_immediately_on_permanent_400():
    transport = httpx.MockTransport(lambda request: httpx.Response(400, json={"error": "bad request"}))
    client = CoinbasePublicRESTClient(
        config=CoinbaseRESTConfig(max_retries=3),
        client=httpx.Client(transport=transport, base_url="https://api.coinbase.com"),
    )

    with pytest.raises(httpx.HTTPStatusError):
        client.get_product("BTC-USD")


def test_book_and_candle_parsing_from_fixture_payloads():
    payload = {
        "bids": [[100.0, 1.2], [99.5, 3.0]],
        "asks": [[101.0, 1.5], [101.5, 2.0]],
    }
    parsed_book = CoinbasePublicRESTClient.parse_product_book_response(payload)
    assert parsed_book["best_bid"] == 100.0
    assert parsed_book["best_ask"] == 101.0
    assert parsed_book["midpoint"] == pytest.approx((100.0 + 101.0) / 2.0)

    candle_payload = [
        [1704067200, 100.0, 102.0, 101.0, 101.5, 0.5],
        [1704067260, 101.0, 103.0, 101.5, 102.0, 0.7],
    ]
    parsed = CoinbasePublicRESTClient.parse_candle_payload(candle_payload)
    assert len(parsed) == 2
    assert parsed[0]["open"] == 101.0
