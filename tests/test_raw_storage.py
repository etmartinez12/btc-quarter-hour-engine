from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest

from btc_quarter_hour_engine.storage.raw import ImmutableRawStore


def test_raw_store_round_trip_and_idempotency(tmp_path):
    store = ImmutableRawStore(tmp_path / "data")
    payload = b'{"hello": "world", "numbers": [1, 2, 3]}'
    artifact = store.write_response(
        source="coinbase_advanced",
        data_kind="candles",
        product_id="BTC-USD",
        request_metadata={"granularity": "ONE_MINUTE"},
        response_bytes=payload,
        retrieved_at=datetime.now(timezone.utc),
    )

    assert artifact.sha256 == hashlib.sha256(payload).hexdigest()
    assert store.read_response(artifact.path) == payload
    assert artifact.path.exists()

    same_artifact = store.write_response(
        source="coinbase_advanced",
        data_kind="candles",
        product_id="BTC-USD",
        request_metadata={"granularity": "ONE_MINUTE"},
        response_bytes=payload,
        retrieved_at=datetime.now(timezone.utc),
    )
    assert same_artifact.sha256 == artifact.sha256
    assert same_artifact.path == artifact.path

    with gzip.open(artifact.path, "rb") as handle:
        assert handle.read() == payload


def test_raw_store_rejects_content_collision(tmp_path):
    store = ImmutableRawStore(tmp_path / "data")
    payload = b'{"a": 2}'
    digest = hashlib.sha256(payload).hexdigest()
    final_path = tmp_path / "data" / "raw" / "coinbase_advanced" / "candles" / "BTC-USD" / f"{digest}.json.gz"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_bytes(gzip.compress(b'{"a": 1}', mtime=0))

    try:
        store.write_response(
            source="coinbase_advanced",
            data_kind="candles",
            product_id="BTC-USD",
            request_metadata={"x": 1},
            response_bytes=payload,
            retrieved_at=datetime.now(timezone.utc),
        )
    except ValueError:
        return
    raise AssertionError("collision should have raised ValueError")


def test_raw_metadata_utc_and_http_provenance(tmp_path):
    store = ImmutableRawStore(tmp_path)
    retrieved = datetime(2026, 1, 1, 12, tzinfo=timezone(timedelta(hours=5)))
    artifact = store.write_response(
        source="coinbase_advanced", data_kind="candles", product_id="BTC-USD",
        request_metadata={
            "request_path": "/api/v3/brokerage/market/products/BTC-USD/candles",
            "request_params": {"start": "1"}, "http_status_code": 200,
            "content_type": "application/json", "requested_start": "2026-01-01T00:00:00Z",
            "requested_end": "2026-01-01T00:01:00Z", "granularity": "ONE_MINUTE",
        },
        response_bytes=b'{ "candles" : [] }', retrieved_at=retrieved,
    )
    metadata = json.loads(artifact.metadata_path.read_text())
    assert metadata["retrieved_at_utc"] == "2026-01-01T07:00:00Z"
    assert artifact.retrieved_at.tzinfo == timezone.utc
    assert metadata["http_status_code"] == 200
    assert metadata["request_params"] == {"start": "1"}
    assert metadata["byte_count"] == len(b'{ "candles" : [] }')
    assert metadata["sha256"] == hashlib.sha256(b'{ "candles" : [] }').hexdigest()
    assert gzip.compress(b'{ "candles" : [] }', compresslevel=9, mtime=0) == artifact.path.read_bytes()
    artifact.metadata_path.unlink()
    with pytest.raises(ValueError, match="metadata missing"):
        store.write_response(
            source="coinbase_advanced", data_kind="candles", product_id="BTC-USD",
            request_metadata={}, response_bytes=b'{ "candles" : [] }', retrieved_at=retrieved,
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        store.write_response(
            source="coinbase_advanced", data_kind="candles", product_id="BTC-USD",
            request_metadata={}, response_bytes=b"{}", retrieved_at=datetime(2026, 1, 1),
        )
