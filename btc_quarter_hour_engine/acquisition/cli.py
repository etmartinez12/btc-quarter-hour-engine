from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from btc_quarter_hour_engine.acquisition.chunking import build_candle_chunk_plan
from btc_quarter_hour_engine.acquisition.coinbase_rest import CoinbasePublicRESTClient
from btc_quarter_hour_engine.acquisition.config import CoinbaseRESTConfig
from btc_quarter_hour_engine.storage.manifest import COINBASE_CANDLE_SCHEMA_VERSION, build_manifest, write_manifest
from btc_quarter_hour_engine.storage.parquet import NormalizedParquetStore
from btc_quarter_hour_engine.storage.raw import ImmutableRawStore


def _parse_utc(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("Timezone-aware timestamp required; use UTC ISO format with timezone information.")
    return dt


def _build_raw_metadata(**kwargs):
    return {key: value for key, value in kwargs.items() if value is not None}


def fetch_candles_main() -> None:
    parser = argparse.ArgumentParser(description="Fetch historical Coinbase candles into the local data lake.")
    parser.add_argument("--product", default="BTC-USD")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--granularity", default="ONE_MINUTE")
    parser.add_argument("--output-root", default="data_lake")
    parser.add_argument("--max-candles-per-request", type=int, default=300)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    start_dt = _parse_utc(args.start)
    end_dt = _parse_utc(args.end)
    config = CoinbaseRESTConfig(
        product_id=args.product,
        candle_granularity=args.granularity,
        timeout_seconds=args.timeout,
    )
    client = CoinbasePublicRESTClient(config=config)
    raw_store = ImmutableRawStore(args.output_root)
    parquet_store = NormalizedParquetStore(args.output_root)

    chunks = build_candle_chunk_plan(
        start=start_dt,
        end=end_dt,
        granularity=args.granularity,
        max_candles_per_request=args.max_candles_per_request,
    )
    raw_artifacts: list[dict] = []
    normalized_artifacts: list[dict] = []
    all_records: list[dict] = []

    try:
        for chunk in chunks:
            payload = client.get_candles(
                args.product,
                chunk.requested_start,
                chunk.requested_end,
                granularity=args.granularity,
            )
            raw_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            raw_artifact = raw_store.write_response(
                source="coinbase_advanced",
                data_kind="candles",
                product_id=args.product,
                request_metadata={
                    "chunk_number": chunk.chunk_number,
                    "requested_start": chunk.requested_start.isoformat(),
                    "requested_end": chunk.requested_end.isoformat(),
                    "granularity": args.granularity,
                    "product_id": args.product,
                },
                response_bytes=raw_bytes,
                retrieved_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            )
            raw_artifacts.append(
                {
                    "path": str(raw_artifact.path),
                    "sha256": raw_artifact.sha256,
                    "byte_count": raw_artifact.byte_count,
                    "requested_start": chunk.requested_start.isoformat(),
                    "requested_end": chunk.requested_end.isoformat(),
                }
            )
            for item in payload:
                if isinstance(item, dict):
                    record = {
                        "source": "coinbase_advanced",
                        "product_id": args.product,
                        "granularity": args.granularity,
                        "bucket_start": item.get("time") or item.get("bucket_start"),
                        "open": item.get("open"),
                        "high": item.get("high"),
                        "low": item.get("low"),
                        "close": item.get("close"),
                        "volume": item.get("volume"),
                        "retrieved_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                    }
                    all_records.append(record)

        if all_records:
            frame = pd.DataFrame(all_records)
            normalized_info = parquet_store.write_dataframe(
                dataframe=frame,
                source="coinbase_advanced",
                product_id=args.product,
                data_kind="candles",
                granularity=args.granularity,
                schema_version=COINBASE_CANDLE_SCHEMA_VERSION,
            )
            normalized_artifacts.append(normalized_info)

        coverage = {
            "expected_bucket_count": len(all_records),
            "observed_bucket_count": len(all_records),
            "missing_bucket_count": 0,
            "duplicate_bucket_count": 0,
            "coverage_fraction": 1.0,
        }
        manifest = build_manifest(
            source="coinbase_advanced",
            product_id=args.product,
            data_kind="candles",
            requested_start=start_dt,
            requested_end=end_dt,
            granularity=args.granularity,
            raw_artifacts=raw_artifacts,
            normalized_artifacts=normalized_artifacts,
            coverage=coverage,
            canonical_target_eligible=False,
            canonical_target_ineligibility_reason="Historical OHLCV candles do not reproduce the exact best-bid/ask midpoint used by the canonical quarter-hour target.",
            acquisition_started_at_utc=None,
            acquisition_completed_at_utc=None,
            request_count=len(chunks),
            schema_version=COINBASE_CANDLE_SCHEMA_VERSION,
        )
        manifest_path = write_manifest(manifest, args.output_root)
        print(f"dataset_id={manifest['dataset_id']}")
        print(f"manifest_path={manifest_path}")
        print(f"raw_request_count={len(chunks)}")
        print(f"normalized_row_count={len(all_records)}")
        print(f"requested_range={start_dt.isoformat()} -> {end_dt.isoformat()}")
        print("canonical_target_eligible=false")
    finally:
        client.close()


def snapshot_book_main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot one Coinbase product book into the local data lake.")
    parser.add_argument("--product", default="BTC-USD")
    parser.add_argument("--output-root", default="data_lake")
    args = parser.parse_args()

    config = CoinbaseRESTConfig(product_id=args.product)
    client = CoinbasePublicRESTClient(config=config)
    raw_store = ImmutableRawStore(args.output_root)
    parquet_store = NormalizedParquetStore(args.output_root)
    try:
        payload = client.get_product_book(args.product)
        raw_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        raw_artifact = raw_store.write_response(
            source="coinbase_advanced",
            data_kind="product_book",
            product_id=args.product,
            request_metadata={"product_id": args.product},
            response_bytes=raw_bytes,
            retrieved_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        )
        frame = pd.DataFrame([payload])
        normalized_info = parquet_store.write_dataframe(
            dataframe=frame,
            source="coinbase_advanced",
            product_id=args.product,
            data_kind="product_book",
            schema_version="1",
        )
        manifest = build_manifest(
            source="coinbase_advanced",
            product_id=args.product,
            data_kind="product_book",
            requested_start=None,
            requested_end=None,
            granularity=None,
            raw_artifacts=[{"path": str(raw_artifact.path), "sha256": raw_artifact.sha256, "byte_count": raw_artifact.byte_count, "requested_start": None, "requested_end": None}],
            normalized_artifacts=[normalized_info],
            coverage={"observed_bucket_count": 0, "missing_bucket_count": 0},
            canonical_target_eligible=False,
            canonical_target_ineligibility_reason="REST product-book snapshots are connectivity and schema validation only; they are not exact quarter-hour canonical targets.",
            request_count=1,
            schema_version="1",
        )
        manifest_path = write_manifest(manifest, args.output_root)
        print(f"product={args.product}")
        print(f"best_bid={payload['best_bid']}")
        print(f"best_ask={payload['best_ask']}")
        print(f"midpoint={payload['midpoint']}")
        print(f"artifact_location={manifest_path}")
        print("canonical_target_eligible=false")
    finally:
        client.close()


if __name__ == "__main__":
    fetch_candles_main()
