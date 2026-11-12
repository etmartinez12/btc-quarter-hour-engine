from __future__ import annotations

import json
import re
from contextlib import contextmanager
from datetime import datetime, timezone

import pytest

from btc_quarter_hour_engine.acquisition import coinbase_websocket as parser_module
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


@pytest.mark.parametrize("field", ["price_level", "new_quantity"])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_parse_level2_event_rejects_nonfinite_numbers(field, value):
    with pytest.raises(ValueError, match="Invalid Level 2"):
        parse_level2_event(_level2_event(updates=[_level2_update(**{field: value})]))


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


def test_parse_real_coinbase_heartbeat_time_and_preserve_envelope_sequence():
    payload = {
        "channel": "heartbeats",
        "timestamp": "2026-10-02T17:20:32.414830619Z",
        "sequence_num": 16,
        "events": [
            {
                "current_time": "2026-10-02 17:20:32.414052991 +0000 UTC m=+181244.321549317",
                "heartbeat_counter": 181244,
            }
        ],
    }
    parsed = parse_coinbase_ws_message(payload)
    assert parsed["type"] == "heartbeat"
    assert parsed["sequence"] == 181244
    assert parsed["time_utc"] == datetime(2026, 10, 2, 17, 20, 32, 414052, tzinfo=timezone.utc)
    assert parsed["envelope"].sequence_num == 16


def test_parse_coinbase_ws_message_falls_back_to_envelope_timestamp_when_heartbeat_missing_current_time():
    payload = _heartbeat_message(events=[{"heartbeat_counter": "3"}])
    parsed = parse_coinbase_ws_message(json.dumps(payload))
    assert parsed["time_utc"] == datetime(2024, 1, 1, 0, 0, 5, tzinfo=timezone.utc)


def test_parse_coinbase_ws_message_passes_through_other_channels():
    payload = {"channel": "subscriptions", "sequence_num": 0, "timestamp": "2024-01-01T00:00:00Z", "events": []}
    parsed = parse_coinbase_ws_message(json.dumps(payload))
    assert parsed["type"] == "subscriptions"
    assert parsed["payload"] == payload


def test_parse_real_coinbase_subscription_envelope():
    payload = {
        "channel": "subscriptions",
        "timestamp": "2026-10-02T17:20:32.027067134Z",
        "sequence_num": 5,
        "events": [{"subscriptions": {"level2": ["BTC-USD"]}}],
    }
    parsed = parse_coinbase_ws_message(payload)
    assert parsed["type"] == "subscriptions"
    assert parsed["envelope"].sequence_num == 5


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


def _phase1_timestamp(value, *, field_name):
    """Frozen timestamp oracle from 4efcf7efe5e77ef231c0f5cd93b996ab2768619d."""
    if value is None:
        raise ValueError(f"Missing {field_name}")
    text = str(value).strip()
    go_timestamp = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.(\d{1,9}) ([+-]\d{4}) UTC m=[+-]\d+(?:\.\d+)?",
        text,
    )
    if go_timestamp:
        wall_time, fraction, offset = go_timestamp.groups()
        microseconds = fraction[:6].ljust(6, "0")
        offset = f"{offset[:3]}:{offset[3:]}"
        try:
            parsed = datetime.fromisoformat(f"{wall_time.replace(' ', 'T')}.{microseconds}{offset}")
        except ValueError as exc:
            raise ValueError(f"Malformed {field_name}: {value!r}") from exc
        return parsed.astimezone(timezone.utc)
    normalized = text.replace("Z", "+00:00")
    if " " in normalized and "T" not in normalized:
        normalized = normalized.replace(" ", "T", 1)
    iso_fraction = re.fullmatch(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.(\d{7,})(.*)", normalized)
    if iso_fraction:
        wall_time, fraction, suffix = iso_fraction.groups()
        normalized = f"{wall_time}.{fraction[:6]}{suffix}"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"Malformed {field_name}: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _outcome(action):
    try:
        return ("value", action())
    except (ValueError, TypeError, OverflowError) as exc:
        return ("error", type(exc), str(exc), type(exc.__cause__), str(exc.__cause__))


_TIMESTAMP_CASES = [
    None, "", "not-a-timestamp", 123, True, [], {},
    "2026-10-06T18:30:00Z",
    "2026-10-06T18:30:00.1Z",
    "2026-10-06T18:30:00.123456Z",
    "2026-10-06T18:30:00.123456789Z",
    "2026-10-06T18:30:00.123456789123Z",
    "2026-10-06T18:30:00,123456789+00:00",
    "2026-10-06T18:30:00.123456789+05:30",
    "2026-10-06T18:30:00.123456789-04:00",
    "2026-10-06T18:30:00.123456789+00:00:30.5",
    "2026-10-06T18:30:00.123456789+0000",
    "2026-10-06T18:30:00",
    "2026-10-06 18:30:00.123456789",
    " 2026-10-06T18:30:00.123456789Z \n",
    "2026-10-06 18:30:00.123456789 +0000 UTC m=+123.456789",
    "2026-10-06 18:30:00.1 -0530 UTC m=-123",
    "2026-10-06 18:30:00.1234567890 +0000 UTC m=+123",
    "2026-10-06 18:30:00.123456789 +2400 UTC m=+123",
    "2026-10-06 18:30:00.123456789 +0000 UTC m=bad",
    "2026-02-30T18:30:00.123456789Z",
    "2026-10-06T25:30:00.123456789Z",
    "2026-10-06T18:30:60Z",
    "2026-10-06T18:30:00.123456789+24:00",
    "2026-10-06T18:30:00.123456789Zjunk",
    "2026-10-06T18:30:00.123456789Z\njunk",
    "20261006T183000+00:00",
    "2026-W41-2T18:30:00Z",
    "2026-10-06",
    "2026-10-06T18:30:00.123456\u0667Z",
]


@pytest.mark.parametrize("value", _TIMESTAMP_CASES)
def test_timestamp_normalization_matches_phase1(value):
    expected = _outcome(lambda: _phase1_timestamp(value, field_name="test timestamp"))
    actual = _outcome(
        lambda: parser_module._require_utc_datetime(value, field_name="test timestamp")
    )
    assert actual == expected


@contextmanager
def _phase1_parser(monkeypatch):
    def uncached_level2_event(event):
        if not isinstance(event, parser_module.Mapping):
            raise ValueError(f"Level 2 event must be a mapping: {event!r}")
        event_type = event.get("type")
        if event_type not in {"snapshot", "update"}:
            raise ValueError(f"Unsupported Level 2 event type: {event_type!r}")
        product_id = event.get("product_id")
        if not isinstance(product_id, str) or not product_id:
            raise ValueError("Level 2 event missing product_id")
        updates_payload = event.get("updates")
        if not isinstance(updates_payload, list) or not updates_payload:
            raise ValueError("Level 2 event missing non-empty updates list")
        updates = tuple(parser_module._parse_level2_update_entry(entry) for entry in updates_payload)
        return Level2Event(type=event_type, product_id=product_id, updates=updates)

    with monkeypatch.context() as patch:
        patch.setattr(parser_module, "_require_utc_datetime", _phase1_timestamp)
        patch.setattr(parser_module, "parse_level2_event", uncached_level2_event)
        yield


def _phase1_message(payload, monkeypatch):
    with _phase1_parser(monkeypatch):
        return _outcome(lambda: parse_coinbase_ws_message(payload))


@pytest.mark.parametrize("kind", ["snapshot", "update", "heartbeat"])
@pytest.mark.parametrize("value", _TIMESTAMP_CASES)
def test_timestamp_messages_match_phase1(kind, value, monkeypatch):
    if kind == "heartbeat":
        payload = _heartbeat_message(events=[{"current_time": value, "heartbeat_counter": 7}])
    else:
        payload = _level2_message(events=[_level2_event(
            type=kind, updates=[
                _level2_update(event_time=value),
                _level2_update(side="offer", event_time=value),
                _level2_update(event_time="2026-10-06T18:30:01.000000001Z"),
                _level2_update(event_time=value),
            ],
        )])
    assert _outcome(lambda: parse_coinbase_ws_message(payload)) == _phase1_message(payload, monkeypatch)


@pytest.mark.parametrize("field,value", [
    ("price_level", "nan"), ("new_quantity", "-1"), ("side", "unknown"),
    ("event_time", "malformed"),
])
def test_repeated_timestamp_does_not_skip_update_validation(field, value, monkeypatch):
    payload = _level2_message(events=[_level2_event(
        type="update",
        updates=[_level2_update(), _level2_update(**{field: value})],
    )])
    expected = _phase1_message(payload, monkeypatch)
    assert expected[0] == "error"
    assert _outcome(lambda: parse_coinbase_ws_message(payload)) == expected


def test_timestamp_reuse_is_event_local_and_does_not_cache_coercible_objects(monkeypatch):
    class ChangingTimestamp:
        def __init__(self):
            self.calls = 0

        def __str__(self):
            self.calls += 1
            return f"2026-10-06T18:30:0{self.calls}Z"

    def payload():
        value = ChangingTimestamp()
        return _level2_message(events=[_level2_event(updates=[
            _level2_update(event_time=value), _level2_update(event_time=value),
        ])])

    expected = _phase1_message(payload(), monkeypatch)
    assert _outcome(lambda: parse_coinbase_ws_message(payload())) == expected

    value = "2026-10-06T18:30:00.123456789Z"
    first = parse_coinbase_ws_message(_level2_message(events=[_level2_event(
        updates=[_level2_update(event_time=value)] * 3,
    )]))
    assert first["envelope"].events[0].updates[0].event_time_utc.microsecond == 123456
    with pytest.raises(ValueError, match="Malformed update event_time"):
        parse_coinbase_ws_message(_level2_message(events=[_level2_event(
            updates=[_level2_update(event_time=value), _level2_update(event_time="bad")],
        )]))


@pytest.mark.parametrize("kind", ["snapshot", "update", "heartbeat"])
@pytest.mark.parametrize("event_time", [
    "2026-10-06T18:30:00.000000999Z",
    "2026-10-06T18:30:00.000001000Z",
    "2026-10-06T14:30:00.000001000-04:00",
    "malformed",
])
def test_timestamp_optimization_preserves_causal_gate_and_replay(kind, event_time, monkeypatch):
    import hashlib

    from btc_quarter_hour_engine.research import prospective_replay as replay_module
    from btc_quarter_hour_engine.research.prospective_source import RawFrameRecord

    boundary = datetime(2026, 10, 6, 18, 30, tzinfo=timezone.utc)
    if kind == "heartbeat":
        payload = _heartbeat_message(
            timestamp=boundary.isoformat(), sequence_num=0,
            events=[{"current_time": event_time, "heartbeat_counter": 1}],
        )
    else:
        payload = _level2_message(
            timestamp=boundary.isoformat(), sequence_num=0,
            events=[_level2_event(type=kind, updates=[
                _level2_update(event_time=event_time),
                _level2_update(side="offer", event_time=event_time),
            ])],
        )
    raw = json.dumps(payload).encode()
    frame = RawFrameRecord(
        raw_bytes=raw, session_id="timestamp-parity", connection_id="connection-1",
        frame_index=0, ingest_time_utc=boundary,
        raw_frame_sha256=hashlib.sha256(raw).hexdigest(),
    )

    def run():
        replay = replay_module._CausalFeatureReplay("BTC-USD")
        outcome = _outcome(lambda: replay_module._replay_frames(
            (frame,), session_id=frame.session_id, expected_index=0, replay=replay,
            cutoff=boundary, previous_ingest=None,
            prefix_commitment=replay_module.initial_prefix_commitment(frame.session_id, boundary),
        ))
        return outcome, replay.to_checkpoint()

    actual = run()
    with _phase1_parser(monkeypatch):
        expected = run()
    assert actual == expected


@pytest.mark.parametrize("kind", ["snapshot", "update"])
def test_multiple_level2_events_keep_timestamp_caches_event_local(kind, monkeypatch):
    value = "2026-10-06T18:30:00.123456789+05:30"
    payload = _level2_message(events=[
        _level2_event(type=kind, updates=[
            _level2_update(event_time=value),
            _level2_update(side="offer", event_time=value),
        ]),
        _level2_event(type=kind, updates=[
            _level2_update(price_level="99", event_time=value),
            _level2_update(side="offer", price_level="102", event_time=value),
        ]),
    ])
    calls = []
    original = parser_module._require_utc_datetime

    def counted(timestamp, *, field_name):
        calls.append((timestamp, field_name))
        return original(timestamp, field_name=field_name)

    monkeypatch.setattr(parser_module, "_require_utc_datetime", counted)
    actual = parse_coinbase_ws_message(payload)
    assert calls.count((value, "update event_time")) == 2
    assert ("value", actual) == _phase1_message(payload, monkeypatch)
    assert len(actual["envelope"].events) == 2
    assert len(actual["updates"]) == 4


@pytest.mark.parametrize("kind", ["snapshot", "update"])
def test_second_level2_event_cannot_reuse_first_events_timestamp_for_invalid_input(kind, monkeypatch):
    payload = _level2_message(events=[
        _level2_event(type=kind),
        _level2_event(type=kind, updates=[
            _level2_update(),
            _level2_update(event_time="malformed"),
        ]),
    ])
    expected = _phase1_message(payload, monkeypatch)
    assert expected[0] == "error"
    assert _outcome(lambda: parse_coinbase_ws_message(payload)) == expected
