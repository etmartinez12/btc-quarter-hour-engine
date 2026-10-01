from __future__ import annotations

import argparse
from datetime import datetime

from btc_quarter_hour_engine.storage.parquet import NormalizedParquetStore
from btc_quarter_hour_engine.storage.raw import ImmutableRawStore
from btc_quarter_hour_engine.storage.forward_parquet import ForwardParquetStore
from btc_quarter_hour_engine.storage.websocket_raw import RawSegmentWriter

from .coinbase_rest import CoinbasePublicRESTClient
from .coinbase_websocket import CoinbaseWebSocketClient
from .collector import WebSocketCollector
from .config import CoinbaseRESTConfig, CoinbaseWebSocketConfig
from .service import acquire_coinbase_book_snapshot, acquire_coinbase_candles
from .websocket_service import CoinbaseWebSocketService
from .websocket_transport import WebsocketsTransport


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
        coverage = result.manifest["coverage"]
        print(f"raw_request_count={result.manifest['request_count']}")
        print(f"normalized_row_count={sum(item['row_count'] for item in result.manifest['normalized_artifacts'])}")
        print(f"requested_start={result.manifest['requested_start']}")
        print(f"requested_end={result.manifest['requested_end']}")
        print(f"observed_start={coverage['first_bucket']}")
        print(f"observed_end={coverage['last_bucket']}")
        print(f"missing_bucket_count={coverage['missing_bucket_count']}")
        print(f"coverage_fraction={coverage['coverage_fraction']}")
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


def collect_coinbase_bbo_main() -> None:
    parser = argparse.ArgumentParser(description="Run the forward Coinbase WebSocket L2/BBO collector.")
    parser.add_argument("--product", default="BTC-USD")
    parser.add_argument("--output-root", default="data_lake")
    parser.add_argument("--max-messages", type=int, default=None, help="Stop after this many websocket messages (omit to run indefinitely).")
    parser.add_argument("--max-duration-seconds", type=float, default=None, help="Stop after this many wall-clock seconds (omit to run indefinitely).")
    args = parser.parse_args()

    config = CoinbaseWebSocketConfig(product_id=args.product)
    client = CoinbaseWebSocketClient(config=config, transport=WebsocketsTransport())
    service = CoinbaseWebSocketService(config=config, client=client)
    raw_segment_writer = RawSegmentWriter(
        args.output_root, product_id=args.product, session_id=service.session_id,
        max_frames=config.raw_segment_max_frames
    )
    forward_store = ForwardParquetStore(args.output_root)
    collector = WebSocketCollector(
        service=service,
        raw_segment_writer=raw_segment_writer,
        forward_store=forward_store,
        output_root=args.output_root,
    )
    result = collector.run(max_messages=args.max_messages, max_duration_seconds=args.max_duration_seconds)
    print(f"session_id={result.manifest['session_id']}")
    print(f"dataset_id={result.manifest['dataset_id']}")
    print(f"manifest_path={result.manifest_path}")
    print(f"product_id={result.manifest['product_id']}")
    print(f"session_start={result.manifest['session_started_at_utc']}")
    print(f"session_end={result.manifest['session_completed_at_utc']}")
    print(f"termination_reason={result.manifest['termination_reason']}")
    print(f"raw_frame_count={sum(segment['frame_count'] for segment in result.raw_segments)}")
    print(f"level2_message_count={sum(c['level2_messages_received'] for c in result.manifest['connections'])}")
    print(f"heartbeat_message_count={sum(c['heartbeat_messages_received'] for c in result.manifest['connections'])}")
    for counter in ("sequence_gap_count", "stale_sequence_count", "heartbeat_timeout_count"):
        print(f"{counter}={result.manifest['integrity'][counter]}")
    for name, key in (
        ("quarter_hour_boundaries_seen", "boundaries_seen"),
        ("quarter_hour_boundaries_eligible", "eligible_boundaries"),
        ("quarter_hour_boundaries_ineligible", "ineligible_boundaries"),
    ):
        print(f"{name}={result.manifest['quarter_hour_summary'][key]}")
    print(f"raw_segment_count={len(result.raw_segments)}")
    print(f"normalized_l2_rows={result.level2_update_row_count}")
    print(f"bbo_state_rows={result.bbo_state_row_count}")
    print(f"boundary_rows={result.observation_count}")
    print(f"observation_count={result.observation_count}")
    print(f"eligible_observation_count={result.eligible_observation_count}")
    print(f"level2_update_row_count={result.level2_update_row_count}")
    print(f"bbo_state_row_count={result.bbo_state_row_count}")
    print(f"connection_count={result.connection_count}")
    print(f"reconnect_count={result.reconnect_count}")
    print(f"canonical_target_eligible={str(result.manifest['canonical_target_eligible']).lower()}")


# Deprecated alias retained for backward compatibility with earlier installs
# of this CLI entry point; prefer `btc-qh-collect-coinbase-bbo`
# (`collect_coinbase_bbo_main`) for new usage.
run_websocket_collector_main = collect_coinbase_bbo_main
