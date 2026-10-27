# V2 feature-family signal localization

This is the second phase of PR #29: a descriptive study of whether existing
causal V2 predictor families contain detectable BTC-USD next-quarter-hour
directional information. It adds no predictors, changes no labels or eligibility
rules, and does not change the default `bbo-causal-v1` contract.
The previously committed development evidence and its reproduction script are
unchanged.

## Frozen family contract

`bbo-causal-v2-families-v1` partitions all 91 approved V2 predictors:

| Family | Count | Membership |
|---|---:|---|
| Momentum | 4 | Log returns at 15, 30, 60 and 240 minutes |
| Spread | 13 | As-of spread; window spread mean, std and max |
| Order-book imbalance | 17 | As-of top-book imbalance; window BBO imbalance mean/std and L2 update/delete imbalances |
| Activity/intensity | 20 | Window BBO-state, midpoint-change, L2-envelope, update and delete rates per minute |
| Range/volatility | 12 | Window distance from high/low and midpoint log-range in basis points |
| Cross-window regimes | 18 | All V2 cross-window imbalance/spread differences, activity log-rate ratios, range fractions and momentum-rate differences |
| Calendar | 7 | UTC hour, quarter-hour and day-of-week sine/cosine pairs; weekend indicator |

Window summaries use the existing 15/30/60/240-minute windows. Spread std stays
in Spread; imbalance std stays in Order-book imbalance. Cross-window differences
and ratios belong **only** to Cross-window regimes, not also to their input
families. Holding out Spread, for example, deliberately retains spread-derived
cross-window regimes; this is a partition ablation, not removal of all
mathematical descendants of spread.

The explicit membership is in
`btc_quarter_hour_engine/research/feature_families_v2.py` and is also embedded in
every output manifest. Runtime checks reject duplicates, missing/extra or
forbidden predictors and changed family membership.

## Reproduction

Use the frozen **materialized research feature Parquet**, including V1 source
summaries and existing audit/target columns, not an OOF Parquet. V2 is derived by
the unchanged V2 transformation. Verify the input checksum against the previously
committed snapshot manifest before running. Run from a clean implementation
commit in a recorded Python/dependency environment:

```powershell
python -m btc_quarter_hour_engine.research.signal_localization_cli --features "C:\research\frozen\research_features.parquet" --output-root "C:\research\results\v2-localization" --predictor-contract bbo-causal-v2
```

The installed equivalent is `btc-qh-localize-v2-signal` with the same arguments.
The V2 contract must be explicit. There are no CLI options for model settings,
seeds, train/test windows, thresholds or family selection. The destination must
be absent or empty; existing evidence is never overwritten. An input whose hash
changes during the run is rejected before outputs are written.

**Do not run or tune on the real October 6-27 frozen development dataset locally.**
The command above documents reproduction for the approved research environment;
implementation validation uses synthetic causal data only.

## Controlled experiment

All three existing models (Logistic Regression, ExtraTrees and
HistGradientBoosting) use the shared benchmark fitting/prediction path, original
wrappers and preprocessing, and unchanged `ExploratoryResearchConfig` defaults.
This includes seed 42, Logistic max iterations 1000, ExtraTrees 300 estimators
with max depth 6, and HGB 100 iterations, learning rate 0.1, 15 leaf nodes,
20 minimum samples per leaf, L2 regularization 1.0 and disabled early stopping.

There are 45 learned-model experiment combinations:

- Full V2 reference: 91 predictors, three models.
- Group-only: one family at a time, seven families times three models.
- Leave-one-group-out: all predictors except one family, seven families times
  three models.

The four original hard-label baselines are retained once, under the reference.
They are not refitted for each family and do not receive invented probabilities.

Feature eligibility and targets are preserved. Trainable rows are sorted by
timestamp and prepared once under the full V2 contract; **all** trainable V2
predictors must be finite even if a particular ablation would omit them.
Missing columns, duplicate input columns/timestamps, nonbinary trainable targets,
missing trainable times, invalid eligibility flags and nonfinite trainable
predictors fail closed rather than changing the comparison population.

The existing chronological expanding splitter creates folds once: initial
training 72 hours, test 12 hours, step 12 hours. It retains its existing handling
of incomplete trailing windows. Every variant receives exactly those train/test
indices. The embargo is strictly `label_available_time < test_start`; labels
available exactly at the test start are excluded. There is no shuffled CV,
resampling, tuning or fold-dependent feature selection.

Original OOF identifiers `timestamp`, `fold_number`, `y_true` (target) and
`benchmark` (model/baseline identifier) are preserved, with `mode` and `family`
added. Every experiment is checked against the splitter's exact OOF population,
including target labels and target returns. Duplicate keys, missing models/rows,
changed folds/targets and nonfinite/out-of-range model probabilities are rejected.
The original fixed 0.5 classification rule is reused; no threshold is selected.

## Outputs and interpretation

| File | Contents |
|---|---|
| `signal_localization.json` | Contract, exact family membership and variant predictors, fold metadata, pooled and fold metrics |
| `oof_predictions.csv`, `oof_predictions.parquet` | Original OOF fields plus mode/family for exact row-wise comparison |
| `metrics.csv`, `metrics.parquet` | Pooled accuracy, ROC AUC, Brier score, log loss; fold accuracy mean/std/min/max; reference differences |
| `fold_metrics.csv`, `fold_metrics.parquet` | The same four classification metrics and reference differences within each fold |
| `manifest.json` | Input SHA256, Git SHA and dirty flag, Python/dependency versions, family contract, fixed configuration, folds and SHA256 hashes of every other output |

Reference differences are **variant minus full V2**, matched by model and,
for fold metrics, fold number, only after exact population checks. Positive
accuracy/AUC deltas favor the variant; negative Brier/log-loss deltas favor it.
Fold standard deviation uses `ddof=0`; fold summaries weight folds equally,
whereas pooled accuracy weights OOF rows equally.

Following the existing benchmark metric convention, ROC AUC and log loss are
undefined for single-class evaluation populations. Undefined metrics/deltas are
JSON `null` and blank CSV/nullable Parquet values, never zero; metric-specific
reason columns distinguish single-class populations from hard-label baselines
without probabilities. Reference baselines have no probabilistic deltas.

Results and JSON serialization have fixed ordering, no run clock or
destination-dependent paths, and no fabricated NaN/Infinity JSON values. Exact
file-byte reproducibility requires the same clean Git revision, input bytes and
dependency/runtime environment; the manifest records those identifiers.
The manifest does not hash itself. A dirty flag marks a run that cannot be
reproduced from its Git SHA alone.

## Scientific limitations

These are exploratory **development** diagnostics, not untouched prospective
validation, statistical significance tests or evidence of profitability.
The October development period already informed V2 design. Forty-five learned
comparisons create substantial multiple-comparison and selection risk. Small
OOF populations, serial dependence, changing market regimes, class imbalance
and correlated predictors can produce unstable apparent differences.

Group-only results measure one family's performance under fixed models, not
intrinsic information content. Leave-one-group-out deltas reflect redundancy
and interactions as well as that family's contribution; they are not causal
feature attribution. Fold dispersion is descriptive, not a confidence interval.
A weak result does not prove that no directional information exists, and a
favorable result does not establish a generalizable edge.

There are no V3 features, ensemble optimization/deployment, confidence or trading
threshold selection, or execution/cost/profit estimates in this phase. Any
future performance claim requires a separately specified, untouched prospective
evaluation, not relabeling these diagnostics as a holdout.
