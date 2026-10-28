from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

EXPECTED_SHA = (
    "ae0417ab38de3a0204baf8202d741a541e38f79597c47d3691da177ac32aa2fc"
)

repo = Path(__file__).resolve().parents[2]

snapshot = (
    Path.home()
    / "btc-qh-v1-diagnostics-input"
    / "research_features.parquet"
)

comparison_root = (
    Path.home()
    / "btc-qh-v1-v2-comparison"
)

artifact_root = (
    repo
    / "research_artifacts"
    / "2026-10-27-v1-v2-development-comparison"
)

doc_path = (
    repo
    / "docs"
    / "research"
    / "v1-v2-development-comparison-2026-10-27.md"
)

artifact_root.mkdir(parents=True, exist_ok=True)
doc_path.parent.mkdir(parents=True, exist_ok=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


actual_sha = sha256_file(snapshot)

if actual_sha != EXPECTED_SHA:
    raise SystemExit(
        f"Snapshot SHA mismatch: {actual_sha}"
    )

print("snapshot_sha256 =", actual_sha)


def load_contract(version):
    root = comparison_root / version

    with (root / "diagnostics.json").open() as handle:
        diagnostics = json.load(handle)

    folds = pd.read_parquet(root / "fold_metrics.parquet")
    drift = pd.read_parquet(root / "feature_drift.parquet")
    predictions = pd.read_parquet(
        root / "oof_predictions.parquet"
    )

    return diagnostics, folds, drift, predictions


v1, v1_folds, v1_drift, v1_preds = load_contract("v1")
v2, v2_folds, v2_drift, v2_preds = load_contract("v2")

assert v1["predictor_contract_version"] == "bbo-causal-v1"
assert v2["predictor_contract_version"] == "bbo-causal-v2"

assert v1["predictor_count"] == 114
assert v2["predictor_count"] == 91

assert v1["n_trainable_rows"] == v2["n_trainable_rows"] == 1935
assert v1["n_folds"] == v2["n_folds"] == 35

assert (
    v1["n_oof_predictions_per_benchmark"]
    == v2["n_oof_predictions_per_benchmark"]
    == 1622
)

print("V1/V2 dataset and fold counts verified")

benchmarks = tuple(sorted(v1["benchmarks"]))

assert set(benchmarks) == set(v2["benchmarks"])

for name in benchmarks:
    columns = ["timestamp", "fold_number"]

    left = v1_preds.loc[
        v1_preds["benchmark"].eq(name), columns
    ].sort_values(columns).reset_index(drop=True)

    right = v2_preds.loc[
        v2_preds["benchmark"].eq(name), columns
    ].sort_values(columns).reset_index(drop=True)

    if len(left) != 1622 or not left.equals(right):
        raise RuntimeError(
            f"OOF population mismatch: {name}"
        )

print("All benchmark OOF populations match")

baselines = (
    "always_down",
    "always_up",
    "train_majority",
    "causal_momentum_15m",
)

for name in baselines:
    a = v1["benchmarks"][name]["metrics"]["accuracy"]
    b = v2["benchmarks"][name]["metrics"]["accuracy"]
    assert a == b

print("All baseline accuracies match")

rows = []

for label, diagnostics in (("v1", v1), ("v2", v2)):
    for name, record in diagnostics["benchmarks"].items():
        metrics = record["metrics"]

        row = {
            "contract": label,
            "benchmark": name,
            "type": record["benchmark_type"],
            "n": record["n_predictions"],
        }

        for key in (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "brier_score",
            "log_loss",
        ):
            row[key] = metrics.get(key)

        rows.append(row)

output = artifact_root / "model_metrics.csv"
pd.DataFrame(rows).to_csv(output, index=False)

print("Created:", output.name)
print("Rows:", len(rows))

for name, left, right in (
    ("fold_metrics", v1_folds, v2_folds),
    ("feature_drift", v1_drift, v2_drift),
):
    combined = pd.concat(
        [
            left.assign(contract="v1"),
            right.assign(contract="v2"),
        ],
        ignore_index=True,
    )

    output = artifact_root / f"{name}.csv"
    combined.to_csv(output, index=False)

    print("Created:", output.name)
    print("Rows:", len(combined))

assert len(v1_drift) == 114
assert len(v2_drift) == 91

print("V1/V2 feature drift counts verified")

models = (
    "logistic_regression",
    "extra_trees",
    "hist_gradient_boosting",
)

confidence_rows = []
agreement_rows = []

for label, diagnostics in (("v1", v1), ("v2", v2)):
    for model in models:
        records = diagnostics["benchmarks"][model][
            "confidence_accuracy"
        ]
        for record in records:
            confidence_rows.append({
                "contract": label,
                "model": model,
                **record,
            })

    for record in diagnostics["model_agreement"]:
        agreement_rows.append({
            "contract": label,
            **record,
        })

pd.DataFrame(confidence_rows).to_csv(
    artifact_root / "confidence_accuracy.csv",
    index=False,
)

pd.DataFrame(agreement_rows).to_csv(
    artifact_root / "model_agreement.csv",
    index=False,
)

print("Confidence rows:", len(confidence_rows))
print("Model agreement rows:", len(agreement_rows))


def drift_summary(frame):
    values = frame["absolute_drift_score"]
    return {
        "feature_count": len(values),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p90": float(values.quantile(0.90)),
        "maximum": float(values.max()),
        "count_ge_0_5": int((values >= 0.5).sum()),
        "count_ge_1_0": int((values >= 1.0).sum()),
    }


summary = {
    "status": "exploratory_development_evidence",
    "snapshot_sha256": actual_sha,
    "trainable_rows": 1935,
    "folds": 35,
    "oof_rows_per_benchmark": 1622,
    "population_comparison": {
        "matching_counts": True,
        "matching_oof_keys": True,
        "matching_baseline_accuracies": True,
    },
    "predictor_counts": {
        "bbo-causal-v1": 114,
        "bbo-causal-v2": 91,
    },
    "drift": {
        "v1": drift_summary(v1_drift),
        "v2": drift_summary(v2_drift),
    },
    "model_metrics": {},
}

for label, diagnostics in (("v1", v1), ("v2", v2)):
    summary["model_metrics"][label] = {}
    for name, record in diagnostics["benchmarks"].items():
        summary["model_metrics"][label][name] = record["metrics"]

output = artifact_root / "comparison_summary.json"

with output.open("w") as handle:
    json.dump(summary, handle, indent=2, allow_nan=False)
    handle.write("\n")

print("Created:", output.name)
print("V1 mean drift:", summary["drift"]["v1"]["mean"])
print("V2 mean drift:", summary["drift"]["v2"]["mean"])

def metric(d, name, key):
    return d["benchmarks"][name]["metrics"][key]


report = [
    "# V1 vs V2 Development Comparison — 2026-10-27",
    "",
    "## Experimental provenance",
    "",
    f"- Frozen snapshot SHA256: `{actual_sha}`",
    "- Development data: October 6–27, 2026",
    "- Trainable observations: 1,935",
    "- Chronological walk-forward folds: 35",
    "- OOF observations per benchmark: 1,622",
    "- V1 predictors: 114",
    "- V2 predictors: 91",
    "- Identical OOF keys and baseline accuracies verified.",
    "- Strict label-availability embargo retained.",
    "",
    "## Model comparison",
    "",
    "| Model | V1 accuracy | V2 accuracy | V1 AUC | V2 AUC |",
    "|---|---:|---:|---:|---:|",
]

labels = {
    "logistic_regression": "Logistic Regression",
    "extra_trees": "ExtraTrees",
    "hist_gradient_boosting": "HistGradientBoosting",
}

for name in models:
    report.append(
        f"| {labels[name]} "
        f"| {metric(v1, name, 'accuracy'):.6f} "
        f"| {metric(v2, name, 'accuracy'):.6f} "
        f"| {metric(v1, name, 'roc_auc'):.6f} "
        f"| {metric(v2, name, 'roc_auc'):.6f} |"
    )

report += [
    "",
    f"Always-DOWN baseline: {metric(v1, 'always_down', 'accuracy'):.6f}.",
    "",
    "## Drift comparison",
    "",
    "| Statistic | V1 | V2 |",
    "|---|---:|---:|",
]

for key in ("mean", "median", "p90", "maximum"):
    report.append(
        f"| {key} | {summary['drift']['v1'][key]:.6f} "
        f"| {summary['drift']['v2'][key]:.6f} |"
    )

doc_path.write_text("\n".join(report) + "\n", encoding="utf-8")
print("Created:", doc_path.name)

report += [
    "",
    "## Scientific interpretation",
    "",
    "V2 reduced mean predictor drift by approximately 39.4%.",
    "All nine V1 predictors with drift >= 1.0 were removed.",
    "This supports using relative, normalized features instead",
    "of raw absolute BTC price levels.",
    "",
    "However, no V2 model demonstrated a directional edge.",
    "All three V2 ROC AUC values remained near 0.50.",
    "Every V2 classifier remained below the always-DOWN baseline.",
    "",
    "## Confidence and validation limitations",
    "",
    "V2 HGB achieved 55.16% accuracy among 252 predictions",
    "with confidence >= 80%. This is exploratory only.",
    "No confidence threshold was validated or selected.",
    "",
    "V2 was designed using this development period.",
    "These results are not untouched prospective validation.",
    "No trading profitability or deployment claim is justified.",
    "",
    "## PR #29 research objectives",
    "",
    "Evaluate existing V2 feature families independently:",
    "",
    "- Momentum",
    "- Spread",
    "- Order-book imbalance",
    "- Activity and update intensity",
    "- Range and volatility",
    "- Cross-window regimes",
    "- Calendar",
    "",
    "Run group-only and leave-one-group-out diagnostics.",
    "Preserve fixed models, identical chronological folds,",
    "strict label embargo, and existing predictor contracts.",
    "",
    "Do not tune models, select trading thresholds,",
    "or add V3 predictors in PR #29.",
    "",
    "The ensemble remains in the project scope but is",
    "not being optimized in this diagnostic phase.",
    "",
    "## Evidence",
    "",
    "Machine-readable CSV/JSON artifacts are stored under",
    "`research_artifacts/2026-10-27-v1-v2-development-comparison/`.",
    "Original diagnostic Parquet and JSON outputs remain",
    "on the research VM.",
]

doc_path.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8",
)

print("Research interpretation added")

source_files = (
    "diagnostics.json",
    "fold_metrics.parquet",
    "feature_drift.parquet",
    "oof_predictions.parquet",
)

manifest = {
    "schema_version": 1,
    "status": "development_evidence_not_prospective",
    "source_snapshot_sha256": actual_sha,
    "source_git_commit": (
        "51cd1552a193f9bf0a80fb261cdd0b81db0f022f"
    ),
    "source_diagnostics": {},
}

for label in ("v1", "v2"):
    manifest["source_diagnostics"][label] = {
        name: sha256_file(comparison_root / label / name)
        for name in source_files
    }

manifest_path = artifact_root / "manifest.json"

manifest_path.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)

print("Created:", manifest_path.name)

files = sorted(
    p for p in artifact_root.iterdir()
    if p.is_file() and p.name != "SHA256SUMS"
)

lines = [
    f"{sha256_file(p)}  {p.name}"
    for p in files
]

(artifact_root / "SHA256SUMS").write_text(
    "\n".join(lines) + "\n"
)

print("Created: SHA256SUMS")
print("Checksummed artifacts:", len(files))
