# btc-quarter-hour-engine

Bitcoin quarter-hour prediction baseline focused on one reproducible research problem:

- **Instrument:** BTC-USD
- **Primary exchange:** Coinbase
- **Canonical price at boundary `t`:** best bid/ask midpoint at the exact quarter-hour timestamp
- **Prediction target:** whether `P[t+15m] > P[t]`
- **Leakage rule:** every feature for timestamp `t` uses only information available at or before `t`

This repository now covers the Phase 1 baseline, the Phase 2 feature-engineering expansion, the Phase 3 microstructure layer, and the first Phase 4 model-diversity step:

1. historical raw BTC market data ingestion skeleton
2. exact quarter-hour boundary extraction
3. leakage-safe feature engineering with momentum, volatility, volume, technical, regime, and microstructure families
4. baseline model set spanning logistic regression, ExtraTrees, LightGBM, and XGBoost
5. walk-forward validation
6. minimal evaluation pipeline on bundled sample data, including confidence, move-size, boundary-slot, and cross-model agreement diagnostics
7. microstructure-ready sample schema with order-book and trade-flow style inputs

## Canonical price definition

Version 1 uses Coinbase BTC-USD midpoint snapshots at exact quarter-hour boundaries:

`P_t = (best_bid_t + best_ask_t) / 2`

The boundary timestamps are:

- `HH:00:00`
- `HH:15:00`
- `HH:30:00`
- `HH:45:00`

The code treats this midpoint as the single source of truth for both labels and benchmark evaluation. Future variants can add alternate price definitions, but they should remain separate experiments.

## Architecture

```text
btc_quarter_hour_engine/
  acquisition/   raw/sample data loading
  boundaries/    exact quarter-hour boundary extraction
  features.py    leakage-safe feature engineering
  targets.py     quarter-hour labels and returns
  models/        logistic regression and LightGBM baselines
  validation/    walk-forward splits and metrics
  live/          minimal live prediction interface
  config.py      shared configuration dataclasses
  run_baseline.py
```

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

## Testing

```bash
pytest
```

The unit tests cover:

- quarter-hour target construction behavior
- walk-forward split ordering and non-overlap
- zero-fold walk-forward benchmark rejection
- live predictor boundary freshness checks
- Phase 2/3 feature-family generation on boundary rows
- Phase 4 multi-model benchmark coverage and comparison outputs
- Phase 3 microstructure sample-data coverage

## Design notes

- Phase 2 expands the feature space while keeping every feature computable from information available at or before boundary `t`.
- Phase 3 adds microstructure-style features such as order-book imbalance, trade-flow imbalance, and average trade size using only contemporaneous or historical values up to boundary `t`.
- Phase 4 begins the model-diversity stage from the original roadmap by comparing linear, bagged-tree, boosted-tree, and gradient-boosted baselines under the same walk-forward protocol.
- Features are computed from historical windows ending at `t`; no feature reaches into `t+15m`.
- Flat moves where `P[t+15m] == P[t]` are left unlabeled and excluded from supervised training.
- The package layout is designed for later expansion into XGBoost, sequence models, microstructure models, and ensembles without changing the target definition.
