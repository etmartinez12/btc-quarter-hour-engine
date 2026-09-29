# btc-quarter-hour-engine

Bitcoin quarter-hour prediction baseline focused on one reproducible research problem:

- **Instrument:** BTC-USD
- **Primary exchange:** Coinbase
- **Canonical price at boundary `t`:** best bid/ask midpoint at the exact quarter-hour timestamp
- **Prediction target:** whether `P[t+15m] > P[t]`
- **Leakage rule:** every feature for timestamp `t` uses only information available at or before `t`

This repository now covers the Phase 1 baseline, the Phase 2 feature-engineering expansion, the Phase 3 microstructure layer, the first Phase 4 model-diversity step, and the first Phase 5 ensemble layer:

1. historical raw BTC market data ingestion skeleton
2. exact quarter-hour boundary extraction
3. leakage-safe feature engineering with momentum, volatility, volume, technical, regime, and microstructure families
4. baseline model set spanning logistic regression, ExtraTrees, LightGBM, and XGBoost
5. walk-forward validation
6. minimal evaluation pipeline on bundled sample data, including confidence, move-size, boundary-slot, cross-model agreement, and first-stage ensemble diagnostics
7. microstructure-ready sample schema with order-book and trade-flow style inputs
8. leakage-safe simple-average and validation-weighted ensemble baselines

## Canonical price definition

Version 1 uses Coinbase BTC-USD midpoint snapshots at exact quarter-hour boundaries:

`P_t = (best_bid_t + best_ask_t) / 2`

The boundary timestamps are:

- `HH:00:00`
- `HH:15:00`
- `HH:30:00`
- `HH:45:00`

The code treats this midpoint as the single source of truth for both labels and benchmark evaluation. Future variants can add alternate price definitions, but they should remain separate experiments.

Minute-bar inputs must be sampled on a strict 1-minute cadence with no gaps. The project defines all minute-level aggregates as trailing intervals ending at the row's timestamp, so a value at `13:00` represents the interval `(12:59:00, 13:00:00]` rather than a forward-looking window.

## Market Data Contract

The project enforces a strict minute-level data contract before any feature generation or training step.

- Timestamps are normalized to UTC before validation.
- Minute input represents exact one-minute observations, not forward-filled intervals.
- Exactly one row must exist for each minute in the current baseline pipeline.
- Missing minutes are rejected before model training or inference.
- Duplicate timestamps are rejected before model training or inference.
- Valid out-of-order rows are sorted chronologically and accepted.
- Minute aggregates are trailing and end-stamped; a row labeled `13:00` represents `(12:59:00, 13:00:00]`.
- `bid > 0`
- `ask > 0`
- `last_trade > 0`
- `volume >= 0`
- `bid <= ask`
- Optional microstructure fields may be absent, but if present they must be finite and nonnegative.
- Malformed source data fails fast.
- The baseline does not silently forward-fill missing minutes.
- The baseline does not silently remove duplicate timestamps.
- All feature construction must use information available at or before prediction timestamp `t`.
- Changing future observations after `t` must not alter features at `t`.

This requirement is enforced through the shared `normalize_market_frame()` contract used by file loading, training-data construction, feature generation, and live inference.

> The bundled Coinbase-style CSV is synthetic/mock data used to exercise the research pipeline. Accuracy from this sample must not be interpreted as evidence of real Bitcoin predictive performance.

## Architecture

```text
btc_quarter_hour_engine/
  acquisition/
    loader.py        raw/sample data loading
    coinbase_rest.py Coinbase Advanced Trade public REST client
    config.py        acquisition configuration
    chunking.py      historical candle chunk planning
  storage/
    raw.py           immutable content-addressed raw payload store
    parquet.py       normalized Parquet output
    manifest.py      acquisition manifests and dataset IDs
  boundaries/         exact quarter-hour boundary extraction
  features.py         leakage-safe feature engineering
  targets.py          quarter-hour labels and returns
  models/             logistic regression, ExtraTrees, LightGBM, and XGBoost baselines
  ensemble/           simple-average and weighted ensemble utilities
  validation/         OOF integrity, naïve benchmarks, row-based demo validation, and time-based research splitting
  live/               minimal live prediction interface
  config.py           shared configuration dataclasses
  run_baseline.py

data_lake/            local-only acquisition output, gitignored
```

## Real Coinbase Data Acquisition

The project now includes a public Coinbase Advanced Trade REST acquisition layer for research context and backfill data. Historical `ONE_MINUTE` candles are acquired as `historical_context_only` observations and remain separate from the canonical exact-boundary target definition. Source payloads are preserved immutably and content-addressed via SHA-256, while normalized analytical outputs are stored as Parquet datasets with manifest provenance for reproducibility.

## Canonical Price Eligibility

The canonical project price remains the exact midpoint of Coinbase best bid and best ask at quarter-hour boundaries:

`P_t = (best_bid_t + best_ask_t) / 2`

Historical OHLCV candle closes are not substituted for this exact canonical midpoint. Therefore:

```text
historical candle data:
    canonical_target_eligible = false

REST one-shot book snapshot:
    canonical_target_eligible = false
    purpose = connectivity/schema validation

future WebSocket BBO observations:
    intended canonical target source
```

This distinction is preserved in manifest metadata and in the normalized data model so the scientific benchmark remains unchanged.

## Local Data Lake

Real acquisition output is kept under a local `data_lake/` directory instead of the packaged synthetic fixture area.

```text
data_lake/
  raw/
    coinbase_advanced/
      candles/
      product_book/
      market_trades/
      server_time/
  normalized/
    coinbase_advanced/
      candles/
      product_book/
  manifests/
```

`raw` is the immutable source-of-truth record, `normalized` stores derived analytical datasets, and `manifests` keeps acquisition provenance and reproducibility metadata.

## Benchmark and OOF Evaluation

The project compares ML models against a small set of credible, legally available baselines rather than treating raw model accuracy as the full story. The current benchmark includes:

- always UP
- always DOWN
- deterministic seeded 50/50 random
- previous exact quarter-hour direction
- 1-minute momentum
- 5-minute momentum

Beating 50% accuracy alone is not sufficient. A model must outperform simple rules using only information that was actually available at prediction time.

All headline benchmark comparisons use the exact same common OOF population. This keeps the comparison fair by ensuring every model and baseline is evaluated on identical unseen timestamps and fold coverage.

### `mean_fold_metrics`

Metrics are calculated independently for each fold and then averaged across folds.

### `pooled_oof_metrics`

All unseen OOF predictions are concatenated and metrics are calculated once across the full pooled evaluation set.

The pooled/common OOF metrics are the primary direct benchmark comparison because they preserve the same legal population across all baselines and models.

Hard naïve rules report:

- accuracy
- precision
- recall
- F1

But they do not receive fabricated Brier score, log loss, or ROC-AUC values because they do not emit calibrated probabilities.

## Validation Protocols

### Demo validation

The bundled row-based validation defaults are:

```text
initial_train_size = 64
test_size = 16
step_size = 16
```

This is used for synthetic bundled data, CI, and smoke testing. It is not evidence of real-world predictive performance.

### Research validation

The intended future research protocol uses timestamp windows rather than fixed row counts:

```text
initial training period = 365 days
test period = 30 days
step period = 30 days
expanding training history = yes
overlapping test windows = no
```

For this research protocol, the prediction timestamp is `t`, and `label_available_time = t + 15 minutes`. A training observation can only be used once `label_available_time <= current prediction/test start`, which prevents the model from using an outcome before it would have been known in live operation.

> The bundled Coinbase-style CSV is synthetic/mock data used to exercise the research pipeline. Accuracy from this sample must not be interpreted as evidence of real Bitcoin predictive performance.

## Installation

```bash
pip install -e .[dev]
```

## Run the baseline pipeline

```bash
btc-qh-baseline
```

Or:

```bash
python -m btc_quarter_hour_engine.run_baseline
```

The sample pipeline will:

- load bundled mock Coinbase-style 1-minute market data from `btc_quarter_hour_engine/data/coinbase_btc_usd_1m_sample.csv`
- include mock best-size, depth, trade-count, and buy/sell flow columns for Phase 3 microstructure experiments
- compute the midpoint price series
- extract exact quarter-hour boundary rows
- build leakage-safe features aligned to boundary `t`
- construct targets from `t` to `t+15m`
- run expanding walk-forward validation
- print aggregate metrics for logistic regression, ExtraTrees, LightGBM, and XGBoost
- report accuracy by confidence threshold, move-size bucket, and quarter-hour slot
- report pairwise prediction agreement and probability correlation across model families
- report simple-average and validation-weighted ensemble benchmark outputs plus consensus diagnostics

## Testing

```bash
pytest
```

The unit tests cover:

- quarter-hour target construction behavior
- walk-forward split ordering and non-overlap
- naive baseline correctness
- OOF uniqueness and alignment checks
- overlapping-fold rejection
- research time splitter behavior
- label availability enforcement
- consensus tie behavior
- live predictor boundary freshness checks
- Phase 2/3 feature-family generation on boundary rows
- Phase 4 multi-model benchmark coverage and comparison outputs
- Phase 3 microstructure sample-data coverage
- Phase 5 ensemble benchmark outputs and leakage-safe weighting behavior

## Design notes

- Phase 2 expands the feature space while keeping every feature computable from information available at or before boundary `t`.
- Phase 3 adds microstructure-style features such as order-book imbalance, trade-flow imbalance, and average trade size using only contemporaneous or historical values up to boundary `t`.
- Phase 4 begins the model-diversity stage from the original roadmap by comparing linear, bagged-tree, boosted-tree, and gradient-boosted baselines under the same walk-forward protocol.
- Phase 5 starts the ensemble stage with simple averaging and validation-weighted averaging computed from chronologically prior fold performance only.
- Features are computed from historical windows ending at `t`; no feature reaches into `t+15m`.
- Flat moves where `P[t+15m] == P[t]` are left unlabeled and excluded from supervised training.
- The package layout is designed for later expansion into XGBoost, sequence models, microstructure models, and ensembles without changing the target definition.
