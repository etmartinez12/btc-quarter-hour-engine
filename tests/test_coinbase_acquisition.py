import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd
import pyarrow.parquet as pq
import pytest

from btc_quarter_hour_engine.acquisition.coinbase_rest import CoinbasePublicRESTClient
from btc_quarter_hour_engine.acquisition.config import CoinbaseRESTConfig
from btc_quarter_hour_engine.acquisition.service import acquire_coinbase_book_snapshot, acquire_coinbase_candles
from btc_quarter_hour_engine.storage.parquet import NormalizedParquetStore
from btc_quarter_hour_engine.storage.raw import ImmutableRawStore

START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def candle(epoch):
    return {"start": str(epoch), "low": "99", "high": "102", "open": "100", "close": "101", "volume": "0.5"}


def make_client(handler):
    return CoinbasePublicRESTClient(
        config=CoinbaseRESTConfig(max_retries=0),
        client=httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.coinbase.com"),
    )


def acquire(tmp_path, client, **overrides):
    args = dict(
        client=client, raw_store=ImmutableRawStore(tmp_path),
        normalized_store=NormalizedParquetStore(tmp_path), output_root=tmp_path,
        product_id="BTC-USD", start=START, end=START + timedelta(minutes=3),
        granularity="ONE_MINUTE", max_candles_per_request=2,
    )
    return acquire_coinbase_candles(**(args | overrides))


def test_acquisition_preserves_exact_bytes_real_coverage_and_schema(tmp_path):
    payloads = [
        b'{ "candles" : [ { "start":"1704067200", "low":"99", "high":"102", "open":"100.00", "close":"101", "volume":"0.5" } ] }',
        json.dumps({"candles": [candle(1704067320)]}).encode(),
    ]
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=payloads[len(calls) - 1], headers={"Content-Type": "application/json"})

    result = acquire(tmp_path, make_client(handler))
    manifest = result.manifest
    assert len(calls) == 2
    assert manifest["acquisition_started_at_utc"].endswith("Z")
    assert manifest["acquisition_completed_at_utc"].endswith("Z")
    assert manifest["coverage"]["expected_bucket_count"] == 3
    assert manifest["coverage"]["observed_bucket_count"] == 2
    assert manifest["coverage"]["missing_bucket_count"] == 1
    assert manifest["coverage"]["coverage_fraction"] == pytest.approx(2 / 3)
    assert manifest["canonical_target_eligible"] is False
    assert manifest["purpose"] == "historical_context_only"
    assert manifest["software"]["package_version"] != "unknown"
    for index, artifact in enumerate(manifest["raw_artifacts"]):
        assert artifact["chunk_number"] == index + 1
        assert artifact["records_received"] == 1
        assert artifact["first_returned_timestamp"] == artifact["last_returned_timestamp"]
        assert artifact["sha256"] == hashlib.sha256(payloads[index]).hexdigest()
        assert ImmutableRawStore(tmp_path).read_response(artifact["path"]) == payloads[index]
        metadata = json.loads(Path(artifact["path"]).with_name(
            Path(artifact["path"]).name.replace(".json.gz", ".meta.json")
        ).read_text())
        assert metadata["request_path"].endswith("/candles")
        assert metadata["request_params"]["start"] == calls[index].url.params["start"]
    assert result.manifest_path.exists()
    artifact = manifest["normalized_artifacts"][0]
    data = pq.ParquetFile(artifact["path"]).read().to_pandas()
    assert len(data) == 2
    assert set(data.columns) == {
        "source", "product_id", "granularity", "bucket_start",
        "open", "high", "low", "close", "volume", "retrieved_at_utc",
    }
    assert not {"bid", "ask", "midpoint", "price_t", "price_t_plus_15m"} & set(data.columns)
    assert data["bucket_start"].dt.tz is not None


def test_partial_chunk_failure_retains_raw_without_success_artifacts(tmp_path):
    responses = [json.dumps({"candles": [candle(1704067200)]}).encode(),
                 json.dumps({"candles": [candle(1704067320)]}).encode()]
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 3:
            return httpx.Response(400, json={"error": "bad"})
        return httpx.Response(200, content=responses[len(calls) - 1])

    with pytest.raises(RuntimeError, match=r"chunk 3 failed for \[.*\):") as exc:
        acquire(tmp_path, make_client(handler), end=START + timedelta(minutes=5))
    assert isinstance(exc.value.__cause__, httpx.HTTPStatusError)
    assert len(list((tmp_path / "raw").rglob("*.json.gz"))) == 2
    assert not (tmp_path / "manifests").exists()
    assert not (tmp_path / "normalized").exists()


def test_empty_acquisition_not_success(tmp_path):
    client = make_client(lambda request: httpx.Response(200, json={"candles": []}))
    with pytest.raises(ValueError, match="coverage_fraction=0.0"):
        acquire(tmp_path, client)
    assert not (tmp_path / "manifests").exists()


def test_duplicate_across_chunks_fails_before_normalized_output(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"candles": [candle(1704067200)]})

    with pytest.raises(RuntimeError, match="outside aligned chunk"):
        acquire(tmp_path, make_client(handler))
    assert not (tmp_path / "normalized").exists()
    assert not (tmp_path / "manifests").exists()


def test_book_snapshot_keeps_source_and_retrieval_time_separate(tmp_path):
    payload = {
        "pricebook": {
            "product_id": "BTC-USD", "bids": [{"price": "100", "size": "1.2"}],
            "asks": [{"price": "101", "size": "1.5"}],
            "time": "2026-01-01T00:00:00Z",
        }
    }
    result = acquire_coinbase_book_snapshot(
        client=make_client(lambda request: httpx.Response(200, json=payload)),
        raw_store=ImmutableRawStore(tmp_path), normalized_store=NormalizedParquetStore(tmp_path),
        output_root=tmp_path, product_id="BTC-USD",
    )
    data = pq.ParquetFile(result.manifest["normalized_artifacts"][0]["path"]).read().to_pandas()
    assert data["source_time_utc"].iloc[0] == pd.Timestamp("2026-01-01T00:00:00Z")
    assert data["retrieved_at_utc"].iloc[0] != data["source_time_utc"].iloc[0]
    assert data["best_bid"].iloc[0] == 100
    assert data["midpoint"].iloc[0] == 100.5
    assert data["canonical_target_eligible"].iloc[0] == False
    assert data["purpose"].iloc[0] == "connectivity_and_schema_validation"
    assert result.manifest["canonical_target_eligible"] is False
    assert result.manifest["purpose"] == "connectivity_and_schema_validation"
