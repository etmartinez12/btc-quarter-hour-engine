# Prospective Evaluation Window Amendment
Date: 2026-11-16
Status: Proposed, pending reviewed implementation and deployment authorization

## Reason

The original 2026-11-15T00:00:00Z launch was not activated.
The first registered boundary has passed. No historical prediction
may be reclassified as timely prospective evidence.

The experiment is rescheduled before any new official prediction
collection is authorized.

## Revised registration

Start: 2026-11-18T00:00:00Z
End exclusive: 2026-12-18T00:00:00Z
Last boundary: 2026-12-17T23:45:00Z
Duration: 30 complete UTC days
Scheduled boundaries: 2880
Frequency: 15 minutes

First checkpoint cutoff: 2026-11-17T23:45:00Z
Latest source-session start: 2026-11-17T19:45:00Z
Operational configuration freeze target: 2026-11-17T23:00:00Z

## Unchanged experimental contract

Product: Coinbase BTC-USD
Prediction target: Direction of price at T+15m versus T
Prediction timing: Exact UTC quarter-hour T
Durable confirmation deadline: T+60 seconds
Probability threshold: 0.5
Candidate: frozen range_volatility model, 12 predictors
Control: frozen full_v2 model, 91 predictors
Baseline: always DOWN
Retraining, calibration, tuning, and backfilling: prohibited

The separately approved independent Coinbase bridge supplies
prospective inference data. Canonical outcome labels remain subject
to the existing target-source and eligibility protocol.

## Frozen evidence preservation

The original 2026-10-28 protocol and model-freeze records are
historical evidence and must remain unchanged.

The trusted official manifest and both model binary hashes remain
unchanged. Their historical registration dates are not rewritten.
The new runtime schedule must be independently reviewed and bound
to a new verified implementation Git SHA.

Existing rehearsal sessions, checkpoints, and measurements are
excluded from official prospective evaluation.

## Activation gates

- Reviewed, regression-tested schedule amendment
- Verified implementation provenance and frozen model loader
- Independently certified continuous source and clock health
- Durable four-hour pre-window checkpoint
- Live T+60 prediction and attestation acceptance
- Fail-closed storage mount and service configuration
- Separately documented operational authorization

No official inference is enabled by this document.
Failure of an activation gate requires postponement, not backfill.
