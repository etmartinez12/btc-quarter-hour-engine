# V1 development diagnostics — 2026-10-27

## Snapshot and protocol

This note records the frozen `bbo-causal-v1` development diagnostic snapshot
used to motivate the V2 feature contract.

| Item | Value |
|---|---|
| Feature snapshot SHA256 | `ae0417ab38de3a0204baf8202d741a541e38f79597c47d3691da177ac32aa2fc` |
| Feature period | 2026-10-06 14:30:00 UTC through 2026-10-27 14:00:00 UTC |
| Total feature rows | 2,015 |
| Trainable rows | 1,935 |
| Trainable UP / DOWN | 952 / 983 |
| UP rate | 0.4919896641 |
| Predictor contract | `bbo-causal-v1` |
| Predictor count | 114 |
| Walk-forward folds | 35 |
| Common OOF predictions per benchmark | 1,622 |

The evaluation used the existing expanding chronological walk-forward
protocol: 72 hours of initial training, 12-hour test windows, 12-hour steps,
and strict label embargo (`label_available_time < test_start`). Model
configurations were fixed. There was no random cross-validation or
hyperparameter search.

Reproduce the PR #27 V1 diagnostics from a materialized feature Parquet:

```powershell
python -m btc_quarter_hour_engine.research.diagnostics_cli `
  --features path\to\research_features.parquet `
  --output-root path\to\diagnostics
```

The CLI defaults to `bbo-causal-v1`; the explicit
`--predictor-contract bbo-causal-v1` option is equivalent.

## Aggregate benchmark results

| Benchmark | Accuracy | ROC AUC |
|---|---:|---:|
| Always down | 0.508631 | — |
| HistGradientBoosting | 0.503699 | 0.508773 |
| ExtraTrees | 0.498150 | 0.500583 |
| Logistic Regression | 0.484587 | 0.496215 |
| Causal 15-minute momentum | 0.494451 | — |

The classifiers were near chance and did not demonstrate a directional
advantage over the always-down baseline. Confidence accuracy was not
monotonic, so the results do not justify a confidence threshold.

The chronological fold metrics showed instability rather than a consistent
advantage across periods. Model-agreement diagnostics are descriptive of
shared predictions, not evidence that those predictions are correct; they do
not change the no-demonstrated-edge conclusion. The snapshot summary used for
this note did not include numerical fold-dispersion or pairwise-agreement
values, so none are asserted here.

## Drift finding and V2 rationale

The strongest chronological drift was concentrated in absolute midpoint
price-level predictors, including `bbo_midpoint_max_*`,
`bbo_midpoint_min_*`, and `asof_midpoint`. Order-book imbalance means also
showed meaningful regime drift. These findings motivate replacing raw level
dependence with relative, scale-normalized, rate-normalized, and
cross-window-regime features while retaining only information available at
the same causal boundary.

## Scientific guardrails

This Oct 6–27 dataset is development evidence, including evidence used to
design V2. Any V2 result on this same snapshot remains development evidence
and cannot be described as prospective validation. Do not optimize confidence
thresholds, search hyperparameters, alter model parameters, make trading/PnL
claims, or claim V2 is superior before it is measured.

A new untouched prospective holdout begins only after the V2 feature and
model specification is frozen. The V1 production/materializer contract
remains the default; V2 is an explicit opt-in research contract.
