from __future__ import annotations

import argparse

from btc_quarter_hour_engine.research.materializer import materialize_research


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Incrementally materialize research artifacts from sealed Coinbase WebSocket segments."
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--product", default="BTC-USD")
    parser.add_argument("--max-new-segments", type=int)
    parser.add_argument("--checkpoint-every-segments", type=int, default=1)
    args = parser.parse_args()

    try:
        result = materialize_research(
            input_root=args.input_root,
            output_root=args.output_root,
            product_id=args.product,
            max_new_segments=args.max_new_segments,
            checkpoint_every_segments=args.checkpoint_every_segments,
        )
    except ValueError as exc:
        parser.error(str(exc))

    diagnostics = result.diagnostics
    print(f"status={diagnostics['status']}")
    print(f"checkpoint_path={diagnostics['checkpoint_path']}")
    print(f"checkpoint_sha256={diagnostics['checkpoint_sha256']}")
    print(f"heartbeat_timeout_seconds={diagnostics['heartbeat_timeout_seconds']}")
    print(f"discovered_sealed_segments={diagnostics['discovered_sealed_segments']}")
    print(f"previously_processed_segments={diagnostics['previously_processed_segments']}")
    print(f"new_segments_processed={diagnostics['new_segments_processed_this_run']}")
    print(f"total_processed_segments={diagnostics['processed_segment_count']}")
    print(f"new_frames_processed={diagnostics['new_frames_processed_this_run']}")
    print(f"total_processed_frames={diagnostics['total_processed_frame_count']}")
    print(f"expected_next_frame_index={diagnostics['expected_next_frame_index']}")
    print(f"active_partial_count_ignored={diagnostics['active_partial_count_ignored']}")
    print(f"canonical_boundary_count={diagnostics['canonical_boundary_row_count']}")
    print(f"eligible_boundary_count={diagnostics['eligible_boundary_count']}")
    print(f"target_count={diagnostics['target_row_count']}")
    print(f"eligible_target_count={diagnostics['eligible_target_count']}")
    print(f"feature_bucket_count={diagnostics['feature_bucket_count']}")
    print(f"research_feature_count={diagnostics['research_feature_row_count']}")
    print(f"feature_ready_count={diagnostics['feature_ready_count']}")
    print(f"trainable_count={diagnostics['trainable_count']}")
    print(f"latest_boundary_utc={diagnostics['latest_canonical_boundary_timestamp']}")
    print(f"predictor_contract_version={diagnostics['predictor_contract_version']}")


if __name__ == "__main__":
    main()
