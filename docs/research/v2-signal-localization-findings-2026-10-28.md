# V2 Signal-Localization Scientific Findings

## Evidence provenance

- Experiment date: 2026-10-28
- Input SHA256: `ae0417ab38de3a0204baf8202d741a541e38f79597c47d3691da177ac32aa2fc`
- Research Git SHA: `273d732c0181303f6d7b667b67ca01436e91b659`
- Predictor contract: `bbo-causal-v2`
- Predictors: 91 across seven frozen families
- Evaluation: 35 chronological walk-forward folds
- OOF observations per configuration: 1,622
- Evaluations: 45 learned configurations and four baselines
- All original artifact checksums verified.
- Git working tree at experiment time: clean.

## Primary observed result

- Range-only ExtraTrees accuracy: 0.527127
- Range-only ExtraTrees AUC: 0.524398
- Full V2 ExtraTrees accuracy: 0.501850
- Always-DOWN accuracy: 0.508631

## Paired prediction outcomes

- both_correct: 520
- range_only_correct: 335
- down_only_correct: 305
- both_wrong: 462

## Temporal consistency

| Period | Folds | N | Range accuracy | DOWN accuracy | Difference (pp) |
|---|---:|---:|---:|---:|---:|
| Early | 17 | 804 | 0.52985 | 0.49254 | +3.731 |
| Late | 18 | 818 | 0.52445 | 0.52445 | +0.000 |

## Complete experimental results

All 49 configurations are reported below.
The 45 learned-model comparisons were evaluated on the
same 1,622 OOF observations.

Accuracy differences are relative to each model's
full V2 reference, in percentage points.
Undefined metrics are shown as an em dash.

| Mode | Family | Model | Accuracy | AUC | Brier | Log loss | Fold std | Delta acc (pp) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| reference | all | always_down | 0.50863 | — | — | — | 0.06808 | 0.00000 |
| reference | all | always_up | 0.49137 | — | — | — | 0.06808 | 0.00000 |
| reference | all | causal_momentum_15m | 0.49445 | — | — | — | 0.05388 | 0.00000 |
| reference | all | extra_trees | 0.50185 | 0.49882 | 0.25151 | 0.69621 | 0.05495 | 0.00000 |
| reference | all | hist_gradient_boosting | 0.49260 | 0.50260 | 0.28997 | 0.79668 | 0.07405 | 0.00000 |
| reference | all | logistic_regression | 0.50370 | 0.49894 | 0.27207 | 0.74879 | 0.06506 | 0.00000 |
| reference | all | train_majority | 0.49383 | — | — | — | 0.06850 | 0.00000 |
| group_only | activity_intensity | extra_trees | 0.49260 | 0.48003 | 0.25298 | 0.69931 | 0.06573 | -0.92478 |
| group_only | activity_intensity | hist_gradient_boosting | 0.50308 | 0.49846 | 0.28156 | 0.77138 | 0.07335 | 1.04809 |
| group_only | activity_intensity | logistic_regression | 0.49075 | 0.49335 | 0.25289 | 0.69930 | 0.06384 | -1.29470 |
| group_only | calendar | extra_trees | 0.50555 | 0.49876 | 0.25499 | 0.70349 | 0.06404 | 0.36991 |
| group_only | calendar | hist_gradient_boosting | 0.50801 | 0.51544 | 0.27409 | 0.75113 | 0.08161 | 1.54131 |
| group_only | calendar | logistic_regression | 0.49692 | 0.49103 | 0.25226 | 0.69774 | 0.06884 | -0.67818 |
| group_only | cross_window_regimes | extra_trees | 0.50617 | 0.49904 | 0.25087 | 0.69484 | 0.07522 | 0.43157 |
| group_only | cross_window_regimes | hist_gradient_boosting | 0.51726 | 0.51956 | 0.28127 | 0.77703 | 0.05849 | 2.46609 |
| group_only | cross_window_regimes | logistic_regression | 0.50000 | 0.49734 | 0.25383 | 0.70114 | 0.06234 | -0.36991 |
| group_only | momentum | extra_trees | 0.51911 | 0.51866 | 0.25152 | 0.69703 | 0.05828 | 1.72626 |
| group_only | momentum | hist_gradient_boosting | 0.50432 | 0.50867 | 0.28330 | 0.77888 | 0.05068 | 1.17139 |
| group_only | momentum | logistic_regression | 0.48644 | 0.47938 | 0.25146 | 0.69609 | 0.06393 | -1.72626 |
| group_only | order_book_imbalance | extra_trees | 0.49630 | 0.50667 | 0.25070 | 0.69456 | 0.08131 | -0.55487 |
| group_only | order_book_imbalance | hist_gradient_boosting | 0.48150 | 0.49518 | 0.28950 | 0.79416 | 0.09475 | -1.10974 |
| group_only | order_book_imbalance | logistic_regression | 0.50308 | 0.51377 | 0.25412 | 0.70204 | 0.07455 | -0.06165 |
| group_only | range_volatility | extra_trees | 0.52713 | 0.52440 | 0.25049 | 0.69461 | 0.07433 | 2.52774 |
| group_only | range_volatility | hist_gradient_boosting | 0.50185 | 0.50325 | 0.28717 | 0.79027 | 0.07663 | 0.92478 |
| group_only | range_volatility | logistic_regression | 0.50247 | 0.49172 | 0.25307 | 0.69972 | 0.06694 | -0.12330 |
| group_only | spread | extra_trees | 0.50801 | 0.49641 | 0.25213 | 0.69783 | 0.05221 | 0.61652 |
| group_only | spread | hist_gradient_boosting | 0.49199 | 0.49544 | 0.28418 | 0.77847 | 0.08500 | -0.06165 |
| group_only | spread | logistic_regression | 0.50432 | 0.49471 | 0.25388 | 0.70173 | 0.07085 | 0.06165 |
| leave_one_group_out | activity_intensity | extra_trees | 0.50432 | 0.50261 | 0.25120 | 0.69560 | 0.08201 | 0.24661 |
| leave_one_group_out | activity_intensity | hist_gradient_boosting | 0.50617 | 0.50613 | 0.28938 | 0.79642 | 0.07779 | 1.35635 |
| leave_one_group_out | activity_intensity | logistic_regression | 0.49383 | 0.49977 | 0.26834 | 0.73845 | 0.06775 | -0.98644 |
| leave_one_group_out | calendar | extra_trees | 0.49938 | 0.49581 | 0.25117 | 0.69553 | 0.08322 | -0.24661 |
| leave_one_group_out | calendar | hist_gradient_boosting | 0.50925 | 0.50505 | 0.29083 | 0.80109 | 0.06924 | 1.66461 |
| leave_one_group_out | calendar | logistic_regression | 0.50678 | 0.50426 | 0.26638 | 0.73266 | 0.07027 | 0.30826 |
| leave_one_group_out | cross_window_regimes | extra_trees | 0.50617 | 0.50317 | 0.25100 | 0.69519 | 0.06863 | 0.43157 |
| leave_one_group_out | cross_window_regimes | hist_gradient_boosting | 0.50678 | 0.51080 | 0.28725 | 0.79329 | 0.07892 | 1.41800 |
| leave_one_group_out | cross_window_regimes | logistic_regression | 0.49630 | 0.49759 | 0.27145 | 0.74615 | 0.07078 | -0.73983 |
| leave_one_group_out | momentum | extra_trees | 0.50370 | 0.50220 | 0.25092 | 0.69503 | 0.07885 | 0.18496 |
| leave_one_group_out | momentum | hist_gradient_boosting | 0.49815 | 0.50186 | 0.29008 | 0.79804 | 0.06748 | 0.55487 |
| leave_one_group_out | momentum | logistic_regression | 0.50123 | 0.50157 | 0.27082 | 0.74587 | 0.06902 | -0.24661 |
| leave_one_group_out | order_book_imbalance | extra_trees | 0.49568 | 0.49316 | 0.25183 | 0.69689 | 0.06172 | -0.61652 |
| leave_one_group_out | order_book_imbalance | hist_gradient_boosting | 0.49507 | 0.49844 | 0.29167 | 0.80324 | 0.06302 | 0.24661 |
| leave_one_group_out | order_book_imbalance | logistic_regression | 0.50493 | 0.49903 | 0.26502 | 0.73008 | 0.08501 | 0.12330 |
| leave_one_group_out | range_volatility | extra_trees | 0.49568 | 0.48808 | 0.25174 | 0.69666 | 0.07276 | -0.61652 |
| leave_one_group_out | range_volatility | hist_gradient_boosting | 0.51295 | 0.51158 | 0.28988 | 0.80284 | 0.07917 | 2.03453 |
| leave_one_group_out | range_volatility | logistic_regression | 0.50185 | 0.49754 | 0.27098 | 0.74518 | 0.07340 | -0.18496 |
| leave_one_group_out | spread | extra_trees | 0.51048 | 0.50702 | 0.25091 | 0.69504 | 0.09207 | 0.86313 |
| leave_one_group_out | spread | hist_gradient_boosting | 0.50801 | 0.51128 | 0.28878 | 0.79839 | 0.08263 | 1.54131 |
| leave_one_group_out | spread | logistic_regression | 0.49445 | 0.50419 | 0.26862 | 0.73923 | 0.08120 | -0.92478 |

## Fold consistency

- Range-only ExtraTrees versus full V2: 22 better, 6 tied, 7 worse.
- Range-only ExtraTrees versus always-DOWN: 25 better, 3 tied, 7 worse.

## Scientific interpretation

Range-only ExtraTrees achieved 52.71% accuracy and 0.5244 ROC AUC.
It produced 30 additional correct predictions relative to always-DOWN across 1,622 OOF observations.

The advantage was not consistent across time:
- Early folds: +3.731 pp.
- Late folds: +0.000 pp.

The model predicted UP on 39.46% of observations,
compared with an actual UP frequency of 49.14%.

## Scientific limitations

- Results come from development data used to design V2.
- The candidate was identified after 45 learned-model comparisons.
- The early/late split is a post-hoc diagnostic, not a holdout.
- Chronological folds are not statistically independent.
- Class imbalance and changing regimes affect baseline accuracy.
- Ablation results do not establish causal feature importance.
- No statistical significance or generalizable edge is established.
- No transaction costs, execution, or profitability were evaluated.

## Research disposition

Retain the 12-feature range/volatility ExtraTrees configuration
as a development-selected hypothesis for prospective evaluation.
Do not treat its historical performance as validated.

Freeze the candidate, training cutoff, comparator definitions,
evaluation chronology, and assessment criteria before examining
prospective outcomes. No retrospective threshold optimization,
ensemble selection, or V3 feature changes are authorized here.
