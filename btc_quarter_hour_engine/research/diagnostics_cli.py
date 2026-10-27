from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from btc_quarter_hour_engine.research.walk_forward import (
    ExploratoryResearchConfig,
    run_exploratory_walk_forward,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run_diagnostics(
    features_path: str | Path,
    output_root: str | Path,
    config: ExploratoryResearchConfig | None = None,
) -> dict[str, Any]:
    features = pd.read_parquet(features_path)
    cfg = config or ExploratoryResearchConfig()
    summary, predictions = run_exploratory_walk_forward(
        features,
        cfg,
    )
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)

    predictions = predictions.sort_values(
        ["benchmark", "fold_number", "timestamp"],
        kind="stable",
    ).reset_index(drop=True)
    fold_metrics = pd.DataFrame.from_records(summary["fold_metrics"])
    if not fold_metrics.empty:
        fold_metrics = fold_metrics.sort_values(
            ["benchmark", "fold_number"],
            kind="stable",
        ).reset_index(drop=True)

    predictions.to_parquet(output / "oof_predictions.parquet", index=False)
    fold_metrics.to_parquet(output / "fold_metrics.parquet", index=False)
    from btc_quarter_hour_engine.research.diagnostics import (
        feature_drift_diagnostics,
    )

    _, feature_drift = feature_drift_diagnostics(
        features,
        predictor_contract_version=cfg.predictor_contract_version,
    )
    feature_drift.to_parquet(output / "feature_drift.parquet", index=False)

    diagnostics_path = output / "diagnostics.json"
    with diagnostics_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(
            _json_safe(summary),
            file,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")

    return {
        "diagnostics": diagnostics_path,
        "oof_predictions": output / "oof_predictions.parquet",
        "feature_drift": output / "feature_drift.parquet",
        "fold_metrics": output / "fold_metrics.parquet",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run leakage-safe exploratory research diagnostics against a "
            "materialized research feature Parquet."
        )
    )
    parser.add_argument("--features", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--predictor-contract",
        choices=("bbo-causal-v1", "bbo-causal-v2"),
        default="bbo-causal-v1",
    )
    args = parser.parse_args(argv)

    try:
        artifacts = run_diagnostics(
            args.features,
            args.output_root,
            ExploratoryResearchConfig(
                predictor_contract_version=args.predictor_contract
            ),
        )
    except ValueError as exc:
        parser.error(str(exc))

    for name, path in artifacts.items():
        print(f"{name}_path={path}")


if __name__ == "__main__":
    main()
