from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

repo = Path(__file__).resolve().parents[2]
root = repo / "research_artifacts" / "2026-10-28-v2-signal-localization"

manifest = json.loads((root / "manifest.json").read_text())

assert manifest["git_sha"] == (
    "273d732c0181303f6d7b667b67ca01436e91b659"
)
assert manifest["snapshot_sha256"] == (
    "ae0417ab38de3a0204baf8202d741a541e38f79597c47d3691da177ac32aa2fc"
)
assert manifest["git_dirty"] is False

for name, expected in manifest["output_sha256"].items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Artifact checksum mismatch: {name}")

metrics = pd.read_parquet(root / "metrics.parquet")
predictions = pd.read_parquet(root / "oof_predictions.parquet")

assert len(metrics) == 49
assert len(predictions) == 49 * 1622

print("Frozen evidence: VERIFIED")
print("Model/baseline configurations:", len(metrics))
print("Total OOF prediction records:", len(predictions))
print("Original experiment files: UNCHANGED")

# Paired comparison: range-only ExtraTrees versus always-DOWN.
keys = ["fold_number", "timestamp"]
columns = keys + ["y_true", "y_pred", "target_log_return_15m"]

range_only = predictions.loc[
    predictions["mode"].eq("group_only")
    & predictions["family"].eq("range_volatility")
    & predictions["benchmark"].eq("extra_trees"),
    columns,
]

always_down = predictions.loc[
    predictions["mode"].eq("reference")
    & predictions["family"].eq("all")
    & predictions["benchmark"].eq("always_down"),
    columns,
]

assert len(range_only) == len(always_down) == 1622

paired = range_only.merge(
    always_down,
    on=keys,
    suffixes=("_range", "_down"),
    validate="one_to_one",
)

assert len(paired) == 1622
assert (paired["y_true_range"] == paired["y_true_down"]).all()
assert (
    paired["target_log_return_15m_range"]
    == paired["target_log_return_15m_down"]
).all()

paired["range_correct"] = (
    paired["y_pred_range"] == paired["y_true_range"]
)
paired["down_correct"] = (
    paired["y_pred_down"] == paired["y_true_down"]
)

paired_results = {
    "both_correct": int(
        (paired["range_correct"] & paired["down_correct"]).sum()
    ),
    "range_only_correct": int(
        (paired["range_correct"] & ~paired["down_correct"]).sum()
    ),
    "down_only_correct": int(
        (~paired["range_correct"] & paired["down_correct"]).sum()
    ),
    "both_wrong": int(
        (~paired["range_correct"] & ~paired["down_correct"]).sum()
    ),
}

assert sum(paired_results.values()) == 1622

print("\n=== PAIRED PREDICTIONS ===")
for name, count in paired_results.items():
    print(f"{name}: {count}")

print(
    "Net additional correct:",
    paired_results["range_only_correct"]
    - paired_results["down_only_correct"],
)

# Temporal consistency using the original chronological fold order.
fold_numbers = sorted(paired["fold_number"].unique())
assert len(fold_numbers) == 35

temporal_results = {}

for period, selected_folds in (
    ("early", fold_numbers[:17]),
    ("late", fold_numbers[17:]),
):
    subset = paired.loc[
        paired["fold_number"].isin(selected_folds)
    ]

    range_accuracy = float(subset["range_correct"].mean())
    down_accuracy = float(subset["down_correct"].mean())

    temporal_results[period] = {
        "folds": len(selected_folds),
        "observations": len(subset),
        "range_accuracy": range_accuracy,
        "down_accuracy": down_accuracy,
        "difference_pp": 100 * (range_accuracy - down_accuracy),
    }

print("\n=== TEMPORAL CONSISTENCY ===")

for period, result in temporal_results.items():
    print(
        f"{period.upper()}: "
        f"folds={result['folds']}, "
        f"n={result['observations']}, "
        f"range_accuracy={result['range_accuracy']:.5f}, "
        f"down_accuracy={result['down_accuracy']:.5f}, "
        f"difference_pp={result['difference_pp']:+.3f}"
    )

print("\n=== DIRECTIONAL DISTRIBUTION ===")
print(
    "Range predicted UP:",
    f"{paired['y_pred_range'].mean():.5f}"
)
print(
    "Actual UP:",
    f"{paired['y_true_range'].mean():.5f}"
)

assert (
    temporal_results["early"]["observations"]
    + temporal_results["late"]["observations"]
    == 1622
)

def metric_row(mode, family, benchmark):
    rows = metrics.loc[
        metrics["mode"].eq(mode)
        & metrics["family"].eq(family)
        & metrics["benchmark"].eq(benchmark)
    ]
    assert len(rows) == 1
    return rows.iloc[0]


range_model = metric_row(
    "group_only", "range_volatility", "extra_trees"
)
full_model = metric_row(
    "reference", "all", "extra_trees"
)
baseline = metric_row(
    "reference", "all", "always_down"
)

assert int(range_model["n_predictions"]) == 1622
assert int(full_model["n_predictions"]) == 1622
assert int(baseline["n_predictions"]) == 1622

report = [
    "# V2 Signal-Localization Scientific Findings",
    "",
    "## Evidence provenance",
    "",
    "- Experiment date: 2026-10-28",
    f"- Input SHA256: `{manifest['snapshot_sha256']}`",
    f"- Research Git SHA: `{manifest['git_sha']}`",
    "- Predictor contract: `bbo-causal-v2`",
    "- Predictors: 91 across seven frozen families",
    "- Evaluation: 35 chronological walk-forward folds",
    "- OOF observations per configuration: 1,622",
    "- Evaluations: 45 learned configurations and four baselines",
    "- All original artifact checksums verified.",
    "- Git working tree at experiment time: clean.",
    "",
    "## Primary observed result",
    "",
    f"- Range-only ExtraTrees accuracy: {range_model['accuracy']:.6f}",
    f"- Range-only ExtraTrees AUC: {range_model['roc_auc']:.6f}",
    f"- Full V2 ExtraTrees accuracy: {full_model['accuracy']:.6f}",
    f"- Always-DOWN accuracy: {baseline['accuracy']:.6f}",
    "",
    "## Paired prediction outcomes",
    "",
]

for key, count in paired_results.items():
    report.append(f"- {key}: {count}")

report += [
    "",
    "## Temporal consistency",
    "",
    "| Period | Folds | N | Range accuracy | DOWN accuracy | Difference (pp) |",
    "|---|---:|---:|---:|---:|---:|",
]

for period, result in temporal_results.items():
    report.append(
        f"| {period.title()} | {result['folds']} "
        f"| {result['observations']} "
        f"| {result['range_accuracy']:.5f} "
        f"| {result['down_accuracy']:.5f} "
        f"| {result['difference_pp']:+.3f} |"
    )

report_path = (
    repo / "docs/research"
    / "v2-signal-localization-findings-2026-10-28.md"
)

report_path.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8",
)

print("\nCreated:", report_path.name)

# Preserve results for every evaluated configuration.
def display_metric(value):
    if pd.isna(value):
        return "—"
    return f"{float(value):.5f}"


ordered = metrics.copy()
mode_order = {
    "reference": 0,
    "group_only": 1,
    "leave_one_group_out": 2,
}
ordered["_order"] = ordered["mode"].map(mode_order)
ordered = ordered.sort_values(
    ["_order", "family", "benchmark"]
)

assert len(ordered) == 49
assert ordered[["mode", "family", "benchmark"]].duplicated().sum() == 0

report += [
    "",
    "## Complete experimental results",
    "",
    "All 49 configurations are reported below.",
    "The 45 learned-model comparisons were evaluated on the",
    "same 1,622 OOF observations.",
    "",
    "Accuracy differences are relative to each model's",
    "full V2 reference, in percentage points.",
    "Undefined metrics are shown as an em dash.",
    "",
    "| Mode | Family | Model | Accuracy | AUC | Brier | Log loss | Fold std | Delta acc (pp) |",
    "|---|---|---|---:|---:|---:|---:|---:|---:|",
]

for _, row in ordered.iterrows():
    delta = row["delta_accuracy_vs_full_v2"]

    report.append(
        f"| {row['mode']} | {row['family']} "
        f"| {row['benchmark']} "
        f"| {display_metric(row['accuracy'])} "
        f"| {display_metric(row['roc_auc'])} "
        f"| {display_metric(row['brier_score'])} "
        f"| {display_metric(row['log_loss'])} "
        f"| {display_metric(row['fold_accuracy_std'])} "
        f"| {display_metric(delta * 100)} |"
    )

report_path.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8",
)

print("Complete results table added:", len(ordered), "rows")

fold_metrics = pd.read_parquet(root / "fold_metrics.parquet")

def fold_accuracy(mode, family, benchmark):
    rows = fold_metrics.loc[
        fold_metrics["mode"].eq(mode)
        & fold_metrics["family"].eq(family)
        & fold_metrics["benchmark"].eq(benchmark)
    ]
    result = rows.set_index("fold_number")["accuracy"].sort_index()
    assert len(result) == 35
    assert result.index.is_unique
    return result

selected = fold_accuracy("group_only", "range_volatility", "extra_trees")
full = fold_accuracy("reference", "all", "extra_trees")
down = fold_accuracy("reference", "all", "always_down")

assert selected.index.equals(full.index)
assert selected.index.equals(down.index)

def fold_comparison(other):
    difference = selected - other
    return (
        int((difference > 0).sum()),
        int((difference == 0).sum()),
        int((difference < 0).sum()),
    )

vs_full = fold_comparison(full)
vs_down = fold_comparison(down)

print("\n=== FOLD CONSISTENCY ===")
print("Versus full V2:", vs_full)
print("Versus always-DOWN:", vs_down)

assert vs_full == (22, 6, 7)
assert vs_down == (25, 3, 7)

net_gain = (
    paired_results["range_only_correct"]
    - paired_results["down_only_correct"]
)

report += [
    "",
    "## Fold consistency",
    "",
    f"- Range-only ExtraTrees versus full V2: "
    f"{vs_full[0]} better, {vs_full[1]} tied, "
    f"{vs_full[2]} worse.",
    f"- Range-only ExtraTrees versus always-DOWN: "
    f"{vs_down[0]} better, {vs_down[1]} tied, "
    f"{vs_down[2]} worse.",
    "",
    "## Scientific interpretation",
    "",
    f"Range-only ExtraTrees achieved "
    f"{range_model['accuracy']:.2%} accuracy and "
    f"{range_model['roc_auc']:.4f} ROC AUC.",
    f"It produced {net_gain} additional correct predictions "
    "relative to always-DOWN across 1,622 OOF observations.",
    "",
    "The advantage was not consistent across time:",
    f"- Early folds: {temporal_results['early']['difference_pp']:+.3f} pp.",
    f"- Late folds: {temporal_results['late']['difference_pp']:+.3f} pp.",
    "",
    f"The model predicted UP on "
    f"{paired['y_pred_range'].mean():.2%} of observations,",
    f"compared with an actual UP frequency of "
    f"{paired['y_true_range'].mean():.2%}.",
    "",
    "## Scientific limitations",
    "",
    "- Results come from development data used to design V2.",
    "- The candidate was identified after 45 learned-model comparisons.",
    "- The early/late split is a post-hoc diagnostic, not a holdout.",
    "- Chronological folds are not statistically independent.",
    "- Class imbalance and changing regimes affect baseline accuracy.",
    "- Ablation results do not establish causal feature importance.",
    "- No statistical significance or generalizable edge is established.",
    "- No transaction costs, execution, or profitability were evaluated.",
    "",
    "## Research disposition",
    "",
    "Retain the 12-feature range/volatility ExtraTrees configuration",
    "as a development-selected hypothesis for prospective evaluation.",
    "Do not treat its historical performance as validated.",
    "",
    "Freeze the candidate, training cutoff, comparator definitions,",
    "evaluation chronology, and assessment criteria before examining",
    "prospective outcomes. No retrospective threshold optimization,",
    "ensemble selection, or V3 feature changes are authorized here.",
]

report_path.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8",
)

print("Scientific interpretation added")
print("Report lines:", len(report))
