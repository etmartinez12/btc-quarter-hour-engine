from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from btc_quarter_hour_engine.research.diagnostics_cli import _json_safe
from btc_quarter_hour_engine.research.features_v2 import V2_PREDICTOR_CONTRACT_VERSION
from btc_quarter_hour_engine.research.forward_dataset import _sha256_file
from btc_quarter_hour_engine.research.signal_localization import run_signal_localization


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(value), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n",
    )


def run_localization(
    features_path: str | Path,
    output_root: str | Path,
    *,
    predictor_contract: str,
) -> dict[str, Path]:
    if predictor_contract != V2_PREDICTOR_CONTRACT_VERSION:
        raise ValueError("Signal localization requires explicit bbo-causal-v2 contract")
    source = Path(features_path).resolve(strict=True)
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; evidence is never overwritten")
    repository = Path(__file__).resolve().parents[2]
    git_sha = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    git_dirty = bool(subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain",
         "--untracked-files=normal"], text=True
    ).strip())
    snapshot_sha256 = _sha256_file(source)
    metadata, predictions, metrics, fold_metrics = run_signal_localization(
        pd.read_parquet(source)
    )
    if _sha256_file(source) != snapshot_sha256:
        raise ValueError("Input snapshot changed while running signal localization")
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, table in (
        ("oof_predictions", predictions),
        ("metrics", metrics),
        ("fold_metrics", fold_metrics),
    ):
        for extension in ("csv", "parquet"):
            path = output / f"{name}.{extension}"
            if extension == "csv":
                table.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")
            else:
                table.to_parquet(path, index=False)
            artifacts[f"{name}_{extension}"] = path
    summary_path = output / "signal_localization.json"
    _write_json(summary_path, metadata)
    artifacts["summary"] = summary_path
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, {
        "manifest_version": 1,
        "snapshot_sha256": snapshot_sha256,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "environment": {
            "python": platform.python_version(),
            **{
                package: importlib.metadata.version(package)
                for package in ("numpy", "pandas", "scikit-learn", "pyarrow")
            },
        },
        "predictor_contract_version": predictor_contract,
        "family_contract_version": metadata["family_contract_version"],
        "feature_families": metadata["feature_families"],
        "variants": metadata["variants"],
        "validation": metadata["validation"],
        "folds": metadata["folds"],
        "n_trainable_rows": metadata["n_trainable_rows"],
        "n_oof_predictions_per_benchmark": metadata["n_oof_predictions_per_benchmark"],
        "scientific_claim_status": metadata["scientific_claim_status"],
        "output_sha256": {
            path.name: _sha256_file(path) for path in artifacts.values()
        },
    })
    artifacts["manifest"] = manifest_path
    return artifacts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Descriptive fixed-model V2 feature-family signal localization."
    )
    parser.add_argument("--features", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--predictor-contract", required=True, choices=(V2_PREDICTOR_CONTRACT_VERSION,)
    )
    args = parser.parse_args(argv)
    try:
        artifacts = run_localization(
            args.features, args.output_root, predictor_contract=args.predictor_contract
        )
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    for name, path in artifacts.items():
        print(f"{name}_path={path}")


if __name__ == "__main__":
    main()
