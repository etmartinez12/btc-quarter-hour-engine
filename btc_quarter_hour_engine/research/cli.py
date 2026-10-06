from __future__ import annotations

import argparse

from btc_quarter_hour_engine.research.forward_dataset import (
    build_research_dataset,
    discover_sealed_segments,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an immutable forward research dataset from sealed Coinbase WebSocket segments."
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--product", default="BTC-USD")
    args = parser.parse_args()

    try:
        discovered = discover_sealed_segments(
            args.input_root,
            product_id=args.product,
        )
        result = build_research_dataset(
            input_root=args.input_root,
            output_root=args.output_root,
            discovered_segments=discovered,
            product_id=args.product,
        )
    except ValueError as exc:
        parser.error(str(exc))

    boundary_summary = result.manifest["boundary_summary"]
    target_summary = result.manifest["target_summary"]
    print(f"dataset_id={result.dataset_id}")
    print(f"manifest_path={result.manifest_path}")
    print(f"source_session_count={result.manifest['source_summary']['session_count']}")
    print(f"sealed_segment_count={result.sealed_segment_count}")
    print(f"raw_frame_count={result.raw_frame_count}")
    print(
        "active_partial_count_ignored="
        f"{result.manifest['source_summary']['active_partial_count_ignored']}"
    )
    print(f"boundary_count={result.boundary_count}")
    print(f"eligible_boundary_count={result.eligible_boundary_count}")
    print(f"ineligible_boundary_count={boundary_summary['ineligible']}")
    print(f"target_candidate_count={result.target_candidate_count}")
    print(f"eligible_target_count={result.eligible_target_count}")
    print(f"up_target_count={target_summary['up_labels']}")
    print(f"down_target_count={target_summary['down_labels']}")
    print(f"flat_move_count={target_summary['flat_moves']}")
    print(f"first_boundary_utc={result.manifest['artifacts'][0]['first_timestamp']}")
    print(f"last_boundary_utc={result.manifest['artifacts'][0]['last_timestamp']}")
    print(
        "latest_labeled_timestamp_utc="
        f"{target_summary['latest_labeled_timestamp_utc']}"
    )
    print(f"input_mode={result.manifest['input_mode']}")


if __name__ == "__main__":
    main()
