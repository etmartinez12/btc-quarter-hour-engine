from __future__ import annotations

import argparse
from datetime import datetime

from btc_quarter_hour_engine.storage.parquet import NormalizedParquetStore
from btc_quarter_hour_engine.storage.raw import ImmutableRawStore

from .coinbase_rest import CoinbasePublicRESTClient
from .config import CoinbaseRESTConfig
from .service import acquire_coinbase_book_snapshot, acquire_coinbase_candles


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timezone-aware timestamp required")
    return parsed


def fetch_candles_main() -> None:
    parser = argparse.ArgumentParser(description="Fetch historical Coinbase candles.")
    parser.add_argument("--product", default="BTC-USD")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--granularity", default="ONE_MINUTE")
    parser.add_argument("--output-root", default="data_lake")
    parser.add_argument("--max-candles-per-request", type=int, default=300)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    client = CoinbasePublicRESTClient(config=CoinbaseRESTConfig(timeout_seconds=args.timeout))
    try:
        result = acquire_coinbase_candles(
            client=client, raw_store=ImmutableRawStore(args.output_root),
            normalized_store=NormalizedParquetStore(args.output_root), output_root=args.output_root,
            product_id=args.product, start=_parse_utc(args.start), end=_parse_utc(args.end),
            granularity=args.granularity, max_candles_per_request=args.max_candles_per_request,
        )
        print(f"dataset_id={result.manifest['dataset_id']}")
        print(f"manifest_path={result.manifest_path}")
        print(f"coverage={result.manifest['coverage']}")
        print("canonical_target_eligible=false")
    finally:
        client.close()


def snapshot_book_main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot one Coinbase public product book.")
    parser.add_argument("--product", default="BTC-USD")
    parser.add_argument("--output-root", default="data_lake")
    args = parser.parse_args()
    client = CoinbasePublicRESTClient()
    try:
        result = acquire_coinbase_book_snapshot(
            client=client, raw_store=ImmutableRawStore(args.output_root),
            normalized_store=NormalizedParquetStore(args.output_root), output_root=args.output_root,
            product_id=args.product,
        )
        print(f"dataset_id={result.manifest['dataset_id']}")
        print(f"manifest_path={result.manifest_path}")
        print("canonical_target_eligible=false")
    finally:
        client.close()


if __name__ == "__main__":
    fetch_candles_main()
