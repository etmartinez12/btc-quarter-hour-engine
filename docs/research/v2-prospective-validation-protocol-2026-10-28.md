# V2 Prospective Validation Protocol

Status: Proposed. Not yet authorized for execution.
Protocol date: 2026-10-28.

## Research objective

Determine whether the development-selected range/volatility
ExtraTrees model exhibits reproducible next-quarter-hour
BTC-USD directional information on previously unseen data.

The October 6-27 development observations are not a holdout.
The selected candidate is exploratory, not validated.

## Frozen training population

Training input:
btc-qh-v1-diagnostics-input/research_features.parquet

SHA256:
ae0417ab38de3a0204baf8202d741a541e38f79597c47d3691da177ac32aa2fc

Use the 1,935 existing trainable observations.
Do not append additional observations to training.

Training labels must satisfy the existing causal target
and label-availability requirements.

## Fixed candidate and controls

Candidate:
ExtraTrees, the 12 V2 range/volatility predictors.

Model control:
ExtraTrees, all 91 existing V2 predictors.

Baseline:
Always predict DOWN.

Both ExtraTrees models use existing fixed research
hyperparameters and random seed 42.

Fit each model once on the frozen training population.
Serialize and hash both fitted models before evaluation.
No updating, refitting, calibration, or threshold tuning
during the evaluation window.

The fixed probability decision threshold is 0.5.

## Prospective evaluation window

Start: 2026-11-15T00:00:00Z
End exclusive: 2026-12-15T00:00:00Z

Duration: 30 complete UTC days.
Maximum: 2,880 quarter-hour observations.

The evaluation window must not begin until this protocol,
implementation, and fitted models are frozen and recorded.

If freezing is incomplete before the scheduled start,
the study must be rescheduled by a documented amendment
made before inspecting prospective outcomes.

Never backdate the protocol or retrospectively designate
previously examined observations as prospective.

## Prediction target and timing

The prediction unit is one exact BTC-USD quarter-hour
boundary at timestamp T.

Predict whether the exact boundary price at T+15 minutes
is higher than the exact boundary price at T.

Use the existing canonical target construction,
price source, and eligibility rules without modification.

Generate predictions using only information available
at or before boundary T. Source market events with
event times after T must never enter its feature vector.

Record predictions no later than T+60 seconds.
Predictions must be recorded before the target boundary
and before the corresponding label becomes available.

A prediction generated after the deadline cannot be
included in the primary prospective evaluation.

## Required prediction evidence

For every scheduled boundary, preserve:

- Boundary timestamp and next target timestamp.
- Data and feature-readiness status.
- Source dataset/session identifiers.
- Feature contract and feature-vector fingerprint.
- Fitted model SHA256 and implementation Git SHA.
- Prediction generation and persistence timestamps.
- Candidate and control probabilities and class labels.
- Baseline always-DOWN prediction.
- Prediction deadline compliance and error reason.
- Subsequent label, eligibility, and availability time.

Predictions must be persisted before outcomes are known.
Store them in an append-only, auditable record.

Never replace missing or late predictions with
retrospectively generated predictions.

Never overwrite an existing prediction after its
label becomes available.

## Eligibility and missing data

Apply the existing canonical boundary eligibility
and causal feature-readiness requirements.

A scored observation must have:

- A valid source boundary and target boundary.
- Complete finite predictors for both frozen models.
- Predictions persisted before the deadline.
- A subsequently available, eligible target label.
- Matching source, model, and feature-contract hashes.

All models and always-DOWN must be evaluated on
the identical set of eligible observations.

Record missing, late, invalid, and unscorable
observations separately with explicit reasons.

Do not silently drop unfavorable predictions or
change eligibility criteria based on outcomes.

## Coverage requirements

The window contains 2,880 scheduled boundaries.

Require at least 90% timely, eligible, scored coverage:
2,592 observations out of 2,880.

Additionally, require at least 24 of the 30 UTC days
to have 80% or greater timely scored coverage.

Report coverage for every day and for the full period.

If either minimum is missed, report the study as
inconclusive under this protocol.

Do not extend the evaluation window to compensate
for missing observations.

## Operational prerequisite

The existing hourly research materializer is not
sufficient by itself for live quarter-hour predictions.

A separate causal inference and prediction-recording
workflow must be implemented and validated before
this protocol can be activated.

It must not interfere with production collection,
canonical data preservation, or materialization.

## Primary evaluation endpoint

The primary endpoint is the paired accuracy difference:

Range-only ExtraTrees accuracy minus always-DOWN accuracy.

Calculate both accuracies over the identical eligible,
timely scored observations in the fixed evaluation window.

Express the difference in percentage points.

Do not select a different primary model, baseline,
endpoint, or evaluation period after observing results.

## Statistical uncertainty

Estimate uncertainty using a UTC-day block bootstrap.

- Resampling unit: complete UTC calendar day.
- Resample the 30 evaluation days with replacement.
- Bootstrap repetitions: 10,000.
- Fixed bootstrap random seed: 20261028.
- Preserve all scored observations within each sampled day.
- Calculate pooled paired accuracy differences for each sample.
- Report the two-sided 95% percentile interval.

Do not independently resample individual quarter-hour
observations or treat overlapping market periods as
independent experimental replications.

The interval is descriptive under day-level resampling
assumptions. Market regime dependence across days may
make the interval optimistic.

## Secondary evaluation endpoints

Report, without model selection:

- Candidate and full V2 ExtraTrees accuracy.
- Candidate and full V2 ROC AUC.
- Candidate and full V2 Brier score and log loss.
- Candidate accuracy minus full V2 accuracy.
- Candidate UP prediction frequency.
- Actual UP frequency and class balance.
- Confusion matrices for all three strategies.
- Daily accuracy and coverage.
- Accuracy by UTC quarter-hour boundary slot.
- First-half versus second-half performance.

Secondary endpoints are descriptive and cannot
replace a failed primary endpoint.

No additional confidence thresholds, trading rules,
feature selections, or ensemble configurations
may be evaluated as part of this protocol.

## Predeclared decision criteria

First verify all coverage, eligibility, provenance,
prediction-timing, and model-integrity requirements.

If any required integrity check fails, the study
cannot support a positive primary conclusion.

If coverage requirements fail, classify the study
as inconclusive, regardless of observed accuracy.

If integrity and coverage checks pass, classify
the primary result as supporting further validation
only when BOTH conditions hold:

1. Candidate accuracy exceeds always-DOWN accuracy
   by at least 1.00 percentage point.

2. The lower bound of the specified two-sided 95%
   day-block bootstrap interval is greater than zero.

Otherwise, report that the predefined evidence
threshold was not met.

The 1.00 percentage-point threshold is a research
screening criterion, not an estimate of trading value.

Meeting both criteria does not establish profitability,
guarantee future performance, or authorize deployment.

Any subsequent confirmation requires a new,
independently registered evaluation period.

## Analysis restrictions

Perform one primary analysis after the scheduled
evaluation window closes and all eligible labels
have become available.

Do not stop early for favorable results.
Do not extend the window after unfavorable results.
Do not revise eligibility based on model outcomes.

Operational monitoring may check coverage, missing
predictions, collection health, and data integrity,
but must not disclose comparative performance
before the scheduled analysis.

Any protocol deviation must be documented with
its timing, reason, and effect on interpretability.

## Model and implementation freeze

The protocol must be approved and committed before
examining any prospective evaluation outcomes.

The implementation, fitted models, and inference
configuration must be frozen no later than
2026-11-14T23:00:00Z.

Freeze the following evidence:

- Protocol Git commit SHA.
- Inference implementation Git commit SHA.
- Frozen training-input SHA256.
- Exact ordered candidate predictor names (12).
- Exact ordered full V2 predictor names (91).
- Predictor and feature-family contract versions.
- Model types, hyperparameters, and random seeds.
- Fitted model and preprocessing artifact SHA256 values.
- Python and dependency versions.
- Training start/completion timestamps.
- Training observation count and timestamp range.
- Evaluation schedule and prediction deadline.

Fit each model once using the existing research
model factories and preprocessing behavior.

The selected 12-feature candidate must use the
existing range/volatility family without modification.

The full V2 comparator must use all 91 approved
predictors in their existing contract order.

No training observation may be added beyond the
frozen October development snapshot.

Verify model deserialization, predictor ordering,
finite inputs, probability bounds, and deterministic
inference before activation.

Preserve the frozen model artifacts and manifest
independently from the running inference service.

Never replace a frozen model during the evaluation.

If the freeze deadline cannot be met, do not start
the scheduled evaluation. Document a prospective
protocol amendment before inspecting outcomes.

## Activation requirements

This protocol does not authorize live execution
until a separate implementation review is complete.

Before activation, verify:

- The Coinbase collector remains operational.
- Canonical source data is unchanged.
- Features are computed strictly as of boundary T.
- Events after T cannot enter the prediction vector.
- Model hashes match the frozen manifest.
- Predictions persist before their deadline.
- Prediction writes are durable and append-only.
- Duplicate boundary predictions are rejected.
- Missing and late predictions are explicitly recorded.
- Labels cannot overwrite prediction evidence.
- UTC clock synchronization is monitored.
- Service restart preserves prior prediction records.
- Synthetic and pre-window dry-run tests pass.
- Production collector and materializer remain isolated.

The workflow must fail closed when a required
invariant cannot be established.

An outage must never cause historical predictions
to be reconstructed as if recorded live.

Model training, inference, and outcome scoring
must remain logically separated.

## Evaluation completion

The fixed evaluation window ends at
2026-12-15T00:00:00Z.

Wait until the final target label and its eligibility
evidence have become available before scoring.

Verify all prediction records, model hashes,
input provenance, and coverage requirements.

Preserve the complete scheduled-boundary ledger,
including missing predictions and exclusion reasons.

Perform the predefined primary analysis once.
Report the uncertainty interval, all secondary
metrics, coverage, and protocol deviations.

Publish negative and inconclusive findings.

No trading or model deployment is authorized
solely by this experiment.

## Research scope preservation

V3 microstructure feature development remains
within the Bitcoin Price Prediction Engine scope.

The original multi-model and ensemble objectives
also remain in scope.

Neither V3 nor ensemble development may alter
this prospective experiment's frozen models,
feature contracts, evaluation data, or scoring rules.

Future experiments require separate documented
protocols and independent validation.
