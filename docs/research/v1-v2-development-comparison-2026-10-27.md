# V1 vs V2 Development Comparison — 2026-10-27

## Experimental provenance

- Frozen snapshot SHA256: `ae0417ab38de3a0204baf8202d741a541e38f79597c47d3691da177ac32aa2fc`
- Development data: October 6–27, 2026
- Trainable observations: 1,935
- Chronological walk-forward folds: 35
- OOF observations per benchmark: 1,622
- V1 predictors: 114
- V2 predictors: 91
- Identical OOF keys and baseline accuracies verified.
- Strict label-availability embargo retained.

## Model comparison

| Model | V1 accuracy | V2 accuracy | V1 AUC | V2 AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.484587 | 0.503699 | 0.496215 | 0.498939 |
| ExtraTrees | 0.498150 | 0.501850 | 0.500583 | 0.498818 |
| HistGradientBoosting | 0.503699 | 0.492602 | 0.508773 | 0.502600 |

Always-DOWN baseline: 0.508631.

## Drift comparison

| Statistic | V1 | V2 |
|---|---:|---:|
| mean | 0.290974 | 0.176473 |
| median | 0.138661 | 0.115503 |
| p90 | 0.555333 | 0.461535 |
| maximum | 1.350366 | 0.812818 |

## Scientific interpretation

V2 reduced mean predictor drift by approximately 39.4%.
All nine V1 predictors with drift >= 1.0 were removed.
This supports using relative, normalized features instead
of raw absolute BTC price levels.

However, no V2 model demonstrated a directional edge.
All three V2 ROC AUC values remained near 0.50.
Every V2 classifier remained below the always-DOWN baseline.

## Confidence and validation limitations

V2 HGB achieved 55.16% accuracy among 252 predictions
with confidence >= 80%. This is exploratory only.
No confidence threshold was validated or selected.

V2 was designed using this development period.
These results are not untouched prospective validation.
No trading profitability or deployment claim is justified.

## PR #29 research objectives

Evaluate existing V2 feature families independently:

- Momentum
- Spread
- Order-book imbalance
- Activity and update intensity
- Range and volatility
- Cross-window regimes
- Calendar

Run group-only and leave-one-group-out diagnostics.
Preserve fixed models, identical chronological folds,
strict label embargo, and existing predictor contracts.

Do not tune models, select trading thresholds,
or add V3 predictors in PR #29.

The ensemble remains in the project scope but is
not being optimized in this diagnostic phase.

## Evidence

Machine-readable CSV/JSON artifacts are stored under
`research_artifacts/2026-10-27-v1-v2-development-comparison/`.
Original diagnostic Parquet and JSON outputs remain
on the research VM.
