from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from btc_quarter_hour_engine.acquisition.coinbase_websocket import (
    parse_coinbase_ws_message,
    parse_heartbeat_message,
    parse_level2_event,
)


def _snapshot(**overrides):
    payload = {
        "type": "snapshot",
        "product_id": "BTC-USD",
        "sequence_num": 10,
        "time": "2024-01-01T00:00:00Z",
        "bids": [{"price": "100.0", "quantity": "1.5"}],
        "asks": [{"price": "101.0", "quantity": "2.5"}],
    }
    payload.update(overrides)
    return payload


def test_parse_snapshot_event_normalizes_levels_and_utc_time():
    event = parse_level2_event(_snapshot())
    assert event["type"] == "snapshot"
    assert event["product_id"] == "BTC-USD"
    assert event["sequence_num"] == 10
    assert event["envelope_time_utc"] == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert event["event_time_utc"] == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert {"side": "bid", "price": 100.0, "quantity": 1.5} in event["updates"]
    assert {"side": "ask", "price": 101.0, "quantity": 2.5} in event["updates"]


def test_parse_l2_update_normalizes_buy_sell_aliases():
    payload = {
        "type": "l2_data",
        "product_id": "BTC-USD",
        "sequence_num": 11,
        "time": "2024-01-01T00:00:01Z",
        "changes": [
            {"side": "buy", "price": "100.5", "new_quantity": "3.0"},
            {"side": "sell", "price": "101.5", "size": "0.0"},
        ],
    }
    event = parse_level2_event(payload)
    assert event["updates"] == [
        {"side": "bid", "price": 100.5, "quantity": 3.0},
        {"side": "ask", "price": 101.5, "quantity": 0.0},
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {"type": "unknown"},
        {"sequence_num": None},
        {"sequence_num": -1},
        {"sequence_num": "not-a-number"},
        {"time": None},
        {"time": "not-a-timestamp"},
        {"time": "2024-01-01T00:00:00"},  # timezone-naive
        {"product_id": None},
        {"product_id": ""},
    ],
)
def test_parse_level2_event_rejects_malformed_envelopes(overrides):
    with pytest.raises(ValueError):
        parse_level2_event(_snapshot(**overrides))


def test_parse_snapshot_rejects_invalid_price_and_quantity():
    with pytest.raises(ValueError):
        parse_level2_event(_snapshot(bids=[{"price": "-1.0", "quantity": "1.0"}]))
    with pytest.raises(ValueError):
        parse_level2_event(_snapshot(bids=[{"price": "1.0", "quantity": "-1.0"}]))
    with pytest.raises(ValueError):
        parse_level2_event(_snapshot(bids=[{"price": "nan", "quantity": "1.0"}]))


def test_parse_update_rejects_unsupported_side():
    payload = {
        "type": "l2_data",
        "product_id": "BTC-USD",
        "sequence_num": 11,
        "time": "2024-01-01T00:00:01Z",
        "changes": [{"side": "sideways", "price": "1.0", "quantity": "1.0"}],
    }
    with pytest.raises(ValueError, match="Unsupported side"):
        parse_level2_event(payload)


def test_parse_level2_event_requires_mapping():
    with pytest.raises(ValueError, match="mapping"):
        parse_level2_event("not-a-mapping")  # type: ignore[arg-type]


def test_parse_heartbeat_message_normalizes_sequence_and_time():
    event = parse_heartbeat_message({"type": "heartbeat", "sequence": "7", "time": "2024-01-01T00:00:05Z"})
    assert event == {"type": "heartbeat", "sequence": 7, "time_utc": datetime(2024, 1, 1, 0, 0, 5, tzinfo=timezone.utc)}


def test_parse_heartbeat_rejects_non_heartbeat_type():
    with pytest.raises(ValueError, match="Not a heartbeat"):
        parse_heartbeat_message({"type": "snapshot"})


def test_parse_coinbase_ws_message_dispatches_by_type():
    snapshot_raw = json.dumps(_snapshot())
    parsed = parse_coinbase_ws_message(snapshot_raw)
    assert parsed["type"] == "snapshot"

    heartbeat_raw = json.dumps({"type": "heartbeat", "sequence": 1, "time": "2024-01-01T00:00:00Z"})
    parsed_heartbeat = parse_coinbase_ws_message(heartbeat_raw)
    assert parsed_heartbeat["type"] == "heartbeat"

    subscriptions_raw = json.dumps({"type": "subscriptions", "channels": []})
    parsed_other = parse_coinbase_ws_message(subscriptions_raw)
    assert parsed_other == {"type": "subscriptions", "payload": {"type": "subscriptions", "channels": []}}


def test_parse_coinbase_ws_message_rejects_malformed_json():
    with pytest.raises(ValueError, match="Malformed websocket JSON"):
        parse_coinbase_ws_message("{not json")


def test_parse_coinbase_ws_message_rejects_non_object_json():
    with pytest.raises(ValueError, match="not a JSON object"):
        parse_coinbase_ws_message("[1, 2, 3]")


def test_parse_coinbase_ws_message_accepts_bytes():
    raw_bytes = json.dumps({"type": "heartbeat", "sequence": 2, "time": "2024-01-01T00:00:02Z"}).encode("utf-8")
    parsed = parse_coinbase_ws_message(raw_bytes)
    assert parsed["type"] == "heartbeat"
    assert parsed["sequence"] == 2
