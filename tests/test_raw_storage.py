from __future__ import annotations

import gzip
import hashlib
from datetime import datetime, timezone

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
