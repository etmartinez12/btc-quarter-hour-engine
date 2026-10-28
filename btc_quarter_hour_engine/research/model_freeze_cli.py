from __future__ import annotations

import argparse
import json
import subprocess

from btc_quarter_hour_engine.research.model_freeze import freeze_models


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Offline fixed V2 ExtraTrees model freeze.")
    parser.add_argument("--training-input", required=True)
    parser.add_argument("--output-directory", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--development-dry-run", action="store_true",
        help="Fit and verify NONOFFICIAL development-only artifacts.",
    )
    mode.add_argument(
        "--official", action="store_true",
        help="Explicitly attest external approval of the supplied protocol and implementation SHAs.",
    )
    parser.add_argument("--approved-implementation-sha")
    parser.add_argument("--approved-protocol-sha")
    args = parser.parse_args(argv)
    try:
        result = freeze_models(
            args.training_input, args.output_directory,
            development=args.development_dry_run,
            approved_implementation_sha=args.approved_implementation_sha,
            approved_protocol_sha=args.approved_protocol_sha,
        )
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
