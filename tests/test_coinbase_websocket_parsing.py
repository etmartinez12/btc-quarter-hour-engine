from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from btc_quarter_hour_engine.acquisition.coinbase_websocket import (
    CoinbaseMessageEnvelope,
    HeartbeatEvent,
    Level2Event,
    Level2UpdateEntry,
    parse_coinbase_envelope,
    parse_coinbase_ws_message,
    parse_heartbeat_event,
    parse_level2_event,
)


def _level2_update(**overrides):
    update = {"side": "bid", "price_level": "100.0", "new_quantity": "1.5", "event_time": "1970-01-01T00:00:00Z"}
    update.update(overrides)
    return update


def _level2_event(**overrides):
    event = {
        "type": "snapshot",
        "product_id": "BTC-USD",
        "updates": [
            _level2_update(side="bid", price_level="100.0", new_quantity="1.5"),
            _level2_update(side="offer", price_level="101.0", new_quantity="2.5"),
        ],
    }
    event.update(overrides)
    return event


def _level2_message(**overrides):
    payload = {
        "channel": "l2_data",
        "client_id": "",
        "timestamp": "2024-01-01T00:00:00Z",
        "sequence_num": 10,
        "events": [_level2_event()],
    }
    payload.update(overrides)
    return payload


def _heartbeat_message(**overrides):
    payload = {
        "channel": "heartbeats",
        "client_id": "",
        "timestamp": "2024-01-01T00:00:05Z",
        "sequence_num": 1,
        "events": [{"current_time": "2024-01-01 00:00:05.000000", "heartbeat_counter": "7"}],
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# parse_level2_event: one events[] entry of a channel: "l2_data" message
# ---------------------------------------------------------------------------


def test_parse_level2_event_snapshot_normalizes_sides_and_levels():
    event = parse_level2_event(_level2_event())
    assert isinstance(event, Level2Event)
    assert event.type == "snapshot"
    assert event.product_id == "BTC-USD"
    assert event.updates == (
        Level2UpdateEntry(
            side="bid", price_level=100.0, new_quantity=1.5, event_time_utc=datetime(1970, 1, 1, tzinfo=timezone.utc)
        ),
        Level2UpdateEntry(
            side="ask", price_level=101.0, new_quantity=2.5, event_time_utc=datetime(1970, 1, 1, tzinfo=timezone.utc)
        ),
    )


def test_parse_level2_event_update_normalizes_side_aliases():
    event = parse_level2_event(
        _level2_event(
            type="update",
            updates=[
                _level2_update(side="buy", price_level="100.5", new_quantity="3.0", event_time="2024-01-01T00:00:01Z"),
                _level2_update(side="sell", price_level="101.5", new_quantity="0.0", event_time="2024-01-01T00:00:01Z"),
            ],
        )
    )
    assert [update.side for update in event.updates] == ["bid", "ask"]
    assert [update.as_dict()["price"] for update in event.updates] == [100.5, 101.5]
    assert [update.as_dict()["quantity"] for update in event.updates] == [3.0, 0.0]


def test_parse_level2_event_update_allows_missing_event_time():
    event = parse_level2_event(_level2_event(type="update", updates=[_level2_update(event_time=None)]))
    assert event.updates[0].event_time_utc is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"type": "unknown"},
        {"product_id": None},
        {"product_id": ""},
        {"updates": []},
        {"updates": "not-a-list"},
    ],
)
def test_parse_level2_event_rejects_malformed_envelopes(overrides):
    with pytest.raises(ValueError):
        parse_level2_event(_level2_event(**overrides))


def test_parse_level2_event_rejects_invalid_price_and_quantity():
    with pytest.raises(ValueError):
        parse_level2_event(_level2_event(updates=[_level2_update(price_level="-1.0")]))
    with pytest.raises(ValueError):
        parse_level2_event(_level2_event(updates=[_level2_update(new_quantity="-1.0")]))
    with pytest.raises(ValueError):
        parse_level2_event(_level2_event(updates=[_level2_update(price_level="nan")]))


def test_parse_level2_event_rejects_unsupported_side():
    with pytest.raises(ValueError, match="Unsupported side"):
        parse_level2_event(_level2_event(updates=[_level2_update(side="sideways")]))


def test_parse_level2_event_requires_mapping():
    with pytest.raises(ValueError, match="mapping"):
        parse_level2_event("not-a-mapping")  # type: ignore[arg-type]


def test_parse_level2_event_update_entry_requires_mapping():
    with pytest.raises(ValueError, match="object"):
        parse_level2_event(_level2_event(updates=["not-a-mapping"]))


# ---------------------------------------------------------------------------
# parse_heartbeat_event: one events[] entry of a channel: "heartbeats" message
# ---------------------------------------------------------------------------


def test_parse_heartbeat_event_normalizes_counter_and_space_separated_time():
    event = parse_heartbeat_event({"current_time": "2024-01-01 00:00:05.000000", "heartbeat_counter": "7"})
    assert isinstance(event, HeartbeatEvent)
    assert event.heartbeat_counter == 7
    assert event.current_time_utc == datetime(2024, 1, 1, 0, 0, 5, tzinfo=timezone.utc)


def test_parse_heartbeat_event_allows_missing_fields():
    event = parse_heartbeat_event({})
    assert event.heartbeat_counter is None
    assert event.current_time_utc is None


def test_parse_heartbeat_event_rejects_malformed_counter():
    with pytest.raises(ValueError, match="heartbeat_counter"):
        parse_heartbeat_event({"heartbeat_counter": "not-a-number"})


def test_parse_heartbeat_event_requires_mapping():
    with pytest.raises(ValueError, match="mapping"):
        parse_heartbeat_event("not-a-mapping")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# parse_coinbase_envelope: the outer envelope shared by every message
# ---------------------------------------------------------------------------


def test_parse_coinbase_envelope_parses_l2_data_events():
    envelope = parse_coinbase_envelope(_level2_message())
    assert isinstance(envelope, CoinbaseMessageEnvelope)
    assert envelope.channel == "l2_data"
    assert envelope.sequence_num == 10
    assert envelope.timestamp_utc == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert len(envelope.events) == 1
    assert isinstance(envelope.events[0], Level2Event)


def test_parse_coinbase_envelope_parses_heartbeat_events():
    envelope = parse_coinbase_envelope(_heartbeat_message())
    assert envelope.channel == "heartbeats"
    assert isinstance(envelope.events[0], HeartbeatEvent)


def test_parse_coinbase_envelope_passes_through_unknown_channel_events():
    envelope = parse_coinbase_envelope(
        {"channel": "subscriptions", "sequence_num": 0, "timestamp": "2024-01-01T00:00:00Z", "events": [{"a": 1}]}
    )
    assert envelope.channel == "subscriptions"
    assert envelope.events == ({"a": 1},)


@pytest.mark.parametrize(
    "overrides",
    [
        {"channel": None},
        {"channel": ""},
        {"sequence_num": None},
        {"sequence_num": -1},
        {"sequence_num": "not-a-number"},
        {"timestamp": None},
        {"timestamp": "not-a-timestamp"},
        {"events": "not-a-list"},
    ],
)
def test_parse_coinbase_envelope_rejects_malformed_payloads(overrides):
    with pytest.raises(ValueError):
        parse_coinbase_envelope(_level2_message(**overrides))


def test_parse_coinbase_envelope_requires_mapping():
    with pytest.raises(ValueError, match="JSON object"):
        parse_coinbase_envelope("not-a-mapping")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# parse_coinbase_ws_message: full raw-frame -> normalized internal dict
# ---------------------------------------------------------------------------


def test_parse_coinbase_ws_message_flattens_snapshot():
    parsed = parse_coinbase_ws_message(json.dumps(_level2_message()))
    assert parsed["type"] == "snapshot"
    assert parsed["product_id"] == "BTC-USD"
    assert parsed["sequence_num"] == 10
    # Snapshot event time always uses the envelope timestamp, never the
    # (unreliable, historically epoch-pinned) per-update event_time.
    assert parsed["event_time_utc"] == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert {"side": "bid", "price": 100.0, "quantity": 1.5, "event_time_utc": datetime(1970, 1, 1, tzinfo=timezone.utc)} in parsed["updates"]
    assert isinstance(parsed["envelope"], CoinbaseMessageEnvelope)


def test_parse_coinbase_ws_message_flattens_update_using_latest_update_event_time():
    payload = _level2_message(
        events=[
            _level2_event(
                type="update",
                updates=[
                    _level2_update(event_time="2024-01-01T00:00:01Z"),
                    _level2_update(event_time="2024-01-01T00:00:05Z"),
                ],
            )
        ]
    )
    parsed = parse_coinbase_ws_message(json.dumps(payload))
    assert parsed["type"] == "l2_data"
    assert parsed["event_time_utc"] == datetime(2024, 1, 1, 0, 0, 5, tzinfo=timezone.utc)


def test_parse_coinbase_ws_message_rejects_mixed_event_types_or_products():
    payload = _level2_message(events=[_level2_event(type="snapshot"), _level2_event(type="update")])
    with pytest.raises(ValueError, match="mixes snapshot and update"):
        parse_coinbase_ws_message(json.dumps(payload))

    payload = _level2_message(events=[_level2_event(product_id="BTC-USD"), _level2_event(product_id="ETH-USD")])
    with pytest.raises(ValueError, match="multiple product_ids"):
        parse_coinbase_ws_message(json.dumps(payload))


def test_parse_coinbase_ws_message_flattens_heartbeat():
    parsed = parse_coinbase_ws_message(json.dumps(_heartbeat_message()))
    assert parsed["type"] == "heartbeat"
    assert parsed["sequence"] == 7
    assert parsed["time_utc"] == datetime(2024, 1, 1, 0, 0, 5, tzinfo=timezone.utc)


def test_parse_coinbase_ws_message_falls_back_to_envelope_timestamp_when_heartbeat_missing_current_time():
    payload = _heartbeat_message(events=[{"heartbeat_counter": "3"}])
    parsed = parse_coinbase_ws_message(json.dumps(payload))
    assert parsed["time_utc"] == datetime(2024, 1, 1, 0, 0, 5, tzinfo=timezone.utc)


def test_parse_coinbase_ws_message_passes_through_other_channels():
    payload = {"channel": "subscriptions", "sequence_num": 0, "timestamp": "2024-01-01T00:00:00Z", "events": []}
    parsed = parse_coinbase_ws_message(json.dumps(payload))
    assert parsed["type"] == "subscriptions"
    assert parsed["payload"] == payload


def test_parse_coinbase_ws_message_rejects_malformed_json():
    with pytest.raises(ValueError, match="Malformed websocket JSON"):
        parse_coinbase_ws_message("{not json")


def test_parse_coinbase_ws_message_rejects_non_object_json():
    with pytest.raises(ValueError, match="not a JSON object"):
        parse_coinbase_ws_message("[1, 2, 3]")


def test_parse_coinbase_ws_message_accepts_bytes():
    raw_bytes = json.dumps(_heartbeat_message()).encode("utf-8")
    parsed = parse_coinbase_ws_message(raw_bytes)
    assert parsed["type"] == "heartbeat"
    assert parsed["sequence"] == 7


def test_parse_coinbase_ws_message_accepts_already_decoded_mapping():
    parsed = parse_coinbase_ws_message(_heartbeat_message())
    assert parsed["type"] == "heartbeat"
