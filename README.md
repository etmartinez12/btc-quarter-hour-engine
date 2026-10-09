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
    coinbase_websocket.py  Coinbase Advanced Trade websocket envelope/event parsing
    websocket_transport.py concrete websockets-based transport
    websocket_service.py   connection lifecycle, sequence/heartbeat, boundary observation derivation
    collector.py     runnable websocket receive loop, reconnect handling, manifest/parquet finalization
    config.py        acquisition configuration
    chunking.py      historical candle chunk planning
    service.py       validated candle/book acquisition orchestration
    cli.py           console-script entry points (REST + websocket collector)
  storage/
    raw.py           immutable content-addressed raw payload store
    parquet.py       normalized Parquet output
    manifest.py      acquisition manifests and dataset IDs
    websocket_raw.py exact-byte sealed raw websocket segment writer
    forward_parquet.py  normalized forward (websocket-derived) Parquet output
    forward_manifest.py forward acquisition manifest builder
  research/
    forward_dataset.py sealed-prefix canonical boundary and target extraction
    cli.py           offline research dataset builder
  market_data/
    order_book.py    sequence-validated L2 order book
    replay.py        provisional live boundaries and canonical sealed-source replay
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

The public Coinbase Advanced Trade v3 market-data client acquires historical candles for research context. Requested candle intervals use aligned UTC half-open `[start, end)` bounds; the final request second is `end - 1 second` to exclude the next bucket. Returned gaps are reported in manifest coverage, never filled. Historical candles are `historical_context_only` and remain separate from the canonical exact-boundary target definition. Exact HTTP response bytes are preserved immutably under SHA-256 and gzip; validated normalized records are partitioned by source UTC date in Parquet, with acquisition and schema provenance in manifests.

## Canonical Price Eligibility

The canonical project price remains the exact midpoint of Coinbase best bid and best ask at quarter-hour boundaries:

`P_t = (best_bid_t + best_ask_t) / 2`

Historical OHLCV candle closes are not substituted for this exact canonical midpoint. Therefore:

```text
historical candle data:
    canonical_target_eligible = false

REST one-shot book snapshot:
    canonical_target_eligible = false
    purpose = connectivity_and_schema_validation

future WebSocket BBO observations:
    intended canonical target source
```

This distinction is preserved in manifest metadata and in the normalized data model so the scientific benchmark remains unchanged.

## Forward Coinbase WebSocket L2/BBO Collection

The canonical target source is now implemented: a real-time Coinbase Advanced Trade `level2` + `heartbeats` WebSocket collector that derives best-bid/best-ask midpoint observations at exact quarter-hour boundaries, going forward from whenever it is run (never backfilled from history).

```text
future WebSocket BBO observations:
    canonical_target_eligible = true, when a boundary is reached with a
    synced, gap-free order book and a recent heartbeat
```

### Components

- `acquisition/coinbase_websocket.py` — explicit dataclasses (`CoinbaseMessageEnvelope`, `Level2Event`, `Level2UpdateEntry`, `HeartbeatEvent`) modeling the *real* Coinbase Advanced Trade websocket wire shape: one outer envelope (`channel`, `sequence_num`, `timestamp`, `events`) shared by every message, with `channel: "l2_data"` events carrying `type: "snapshot"|"update"` plus an `updates` array (`side: "bid"|"offer"`, `price_level`, `new_quantity`, `event_time`), and `channel: "heartbeats"` events carrying `current_time`/`heartbeat_counter`. Malformed frames never raise inside the receive path: `CoinbaseWebSocketClient.receive_message()` always returns a frame with the exact raw bytes, setting `message=None`/`parse_error=...` on a parse failure instead, so raw capture happens strictly *before* (and independent of) parsing.
- `acquisition/websocket_transport.py` — concrete `WebsocketsTransport`, a thin wrapper over `websockets.sync.client.connect` (blocking, no asyncio required).
- `market_data/order_book.py` — `Level2OrderBook`, a sequence-validated L2 book with distinct `UNINITIALIZED`, `SYNCED`, recoverable `REBUILDING`, and fatal `INVALID` states. Each update envelope remains atomic: Coinbase production traffic has demonstrated same-connection clear/refill rebuilds in ordinary update envelopes, briefly leaving a one-sided or empty book before later sequence-contiguous updates restore it. The intermediate state is committed but never exposed as a canonical BBO; crossed books and malformed updates remain fatal. `reset()` (in-band resnapshot, same connection, sequence continuity preserved) is distinct from `reset_for_new_connection()` (genuine reconnect; Coinbase sequence numbers are scoped per-connection, so the old connection's last sequence number must not be compared against the new one).
- `market_data/replay.py` — live boundary processing is operational/provisional only. Normal collector finalization uses the incremental `CanonicalReplayAccumulator`; `replay_recorded_frames()` remains the independent offline audit oracle. Both preserve sequence validation in delivery order, reconnect isolation, and causal treatment of source-time updates.
- `acquisition/websocket_service.py` — `CoinbaseWebSocketService`: connection lifecycle, stale-vs-gap sequence handling, heartbeat-health tracking, and reconnect-with-wait-for-snapshot, all routed through the shared boundary processor above. `now_fn`/`monotonic_fn` are both injectable for fully deterministic tests. Every superseded connection's diagnostics are retained in `connection_history` (not overwritten) so past reconnects remain auditable. In addition to quarter-hour BBO observations, every applied book mutation is accumulated as a normalized `level2_update_rows` entry (one row per individual price-level change, full per-update event time) and a `bbo_state_rows` entry (the resulting book state, including truthful partial or empty `REBUILDING` states), drained via `drain_level2_update_rows()`/`drain_bbo_state_rows()`.
- `storage/websocket_raw.py` — `RawSegmentWriter`: durably appends exact text or binary frame bytes to a `.partial` artifact (no re-serialization), then seals immutable, content-addressed, gzip-compressed segments with per-frame provenance. Unexpected failure retains the incomplete partial; it is not a sealed artifact or completed manifest.
- `storage/forward_schema.py` — the single source of truth for every forward-collection `data_kind`/schema-version/source constant (`quarter_hour_bbo`, `level2_updates`, `bbo_state`, `websocket_segments`, `websocket_frames`), imported everywhere those values are needed instead of being hard-coded.
- `storage/forward_parquet.py` — `ForwardParquetStore`: writes derived quarter-hour BBO, `level2_updates`, and `bbo_state` rows to dated, normalized Parquet partitions.
- `storage/forward_manifest.py` — a dedicated forward-session manifest records connection diagnostics, integrity counters, boundary summary, raw and normalized artifacts, and a dataset ID derived from ordered sealed hashes and schema versions rather than completion time.
- `acquisition/collector.py` — `WebSocketCollector` records every received raw frame before interpretation. Normal completion or Ctrl+C seals raw data, persists normalized L2/BBO rows, generates the **final canonical** quarter-hour ledger with the incremental accumulator, and writes a session manifest. Unexpected failure closes the transport without claiming a completed session.

The `bbo_state` and quarter-hour boundary row shapes are unchanged: their string `state`/`eligibility_reason` fields now carry `REBUILDING`/`book_rebuilding`, so their schema versions remain `1`. The forward session manifest advances to schema `3` to add deterministic per-connection rebuild and recovery counters to connection diagnostics and integrity summaries.

### Eligibility rule

A quarter-hour boundary observation is `eligible = True` only if, at the moment the boundary is crossed:

1. the order book is synced (an in-sequence snapshot has been applied since the last reconnect), **and**
2. there has been no sequence gap since that sync, **and**
3. a heartbeat has been observed within `heartbeat_timeout_seconds` of the boundary timestamp.

Every encountered quarter-hour boundary is retained, including boundaries without a synced snapshot. Each row carries `canonical_target_eligible` and an explicit `eligibility_reason` such as `no_synced_snapshot`, `book_rebuilding`, `sequence_gap`, `connection_unhealthy`, `crossed_book`, or `heartbeat_stale`. A boundary whose causal book is rebuilding has no eligible price; a later sequence-contiguous refill restores eligibility without requiring a reconnect. Three independent production occurrences showed full or partial-side clears followed by same-connection refills, with no malformed frames, sequence gaps, or intervening reconnects. A heartbeat after the boundary cannot establish health at that earlier boundary. Price updates themselves do not require freshness while a synchronized, healthy book remains valid.

A sequence number that duplicates or precedes the last-applied one (`STALE`) is dropped without penalty — a synced book stays synced and a rebuilding book remains recoverable; no reconnect is triggered. Only a true forward `GAP` invalidates the connection epoch and forces a reconnect.

### Reconnect and wait-for-snapshot

On a connection error or a genuine sequence gap, the service reconnects and then blocks — draining and discarding any non-snapshot messages that arrive in the meantime — until a fresh `l2_data` snapshot is received or `snapshot_wait_timeout_seconds` (`CoinbaseWebSocketConfig`) elapses, at which point it gives up and surfaces the failure rather than running with a desynced book. `now_fn`/`monotonic_fn` are injectable on both `CoinbaseWebSocketClient` and `CoinbaseWebSocketService`, so this deadline logic is fully deterministic under test. Every reconnect's outcome is appended to `connection_history` rather than overwriting prior state, giving a complete audit trail of every connection attempt across a run.

### Running the collector

```bash
btc-qh-collect-coinbase-bbo \
  --product BTC-USD \
  --output-root data_lake \
  --max-duration-seconds 3600
```

(`btc-qh-run-coinbase-websocket-collector` remains available as a deprecated backward-compatible alias for the same entry point.)

Omit `--max-messages`/`--max-duration-seconds` to run indefinitely (e.g. under a supervisor process); the collector reconnects automatically on connection errors, a genuine sequence gap, or a stale heartbeat, up to `CoinbaseWebSocketConfig.max_reconnect_attempts` (default: unlimited).

On duration/explicit stop or Ctrl+C, the collector seals raw frames, writes normalized updates and operational BBO state, finalizes the canonical boundary ledger through the incremental accumulator, and writes a completed session manifest under `data_lake/manifests/forward_sessions/`. The offline `replay_recorded_frames()` path remains an audit oracle, not normal production finalization. On unexpected failure, a durable `.partial` file remains and no completed-success manifest is written.

### Raw segment layout and replay

Sealed raw segments live under `data_lake/raw/coinbase_advanced/websocket_segments/BTC-USD/<sha256>.json.gz`, with 8-byte-length-prefixed exact original frame bytes, including binary and malformed frames. Segment metadata retains session, connection, frame index, sequence, ingest time and frame hash. `RawSegmentWriter.read_segment_frames(path)` recovers the byte strings. Normal collector finalization uses the incremental accumulator. `replay_recorded_frames()` is retained as the independent offline audit oracle; both preserve delivery-order sequence validation and causal event-time mutation semantics.

## Forward Research Dataset Extraction

Build an offline research snapshot from immutable sealed raw segments while the collector continues running:

```bash
btc-qh-build-forward-research-dataset \
  --input-root /var/lib/btc-qh/data_lake_prod \
  --output-root /var/lib/btc-qh/research \
  --product BTC-USD
```

The production source is read-only. Discovery snapshots sealed segment metadata once; active `.partial` files are ignored and never opened. Each immutable dataset ID binds the exact raw and metadata sidecar hashes, provenance, discovery-time partial count, configuration, and required software Git SHA. Its manifest uses logical source paths relative to `--input-root`, so equivalent snapshots remain portable across mount points. The input is a `sealed_prefix_snapshot`, not necessarily a completed collector session: active and future frames are excluded, and a later snapshot may supersede it without changing the prior dataset.

Canonical labels use only exact `t` and `t+15m` boundary rows, and require both boundaries to be eligible with finite positive midpoint prices. Missing boundaries are not bridged, flat moves remain unlabeled, and no candles, interpolation, nearest-time matching, or forward filling are used.

Normalized output lives under `data_lake/normalized/coinbase_advanced/BTC-USD/<data_kind>/date=YYYY-MM-DD/part-<sha256>.parquet` for three `data_kind` values:

- `quarter_hour_bbo` — final canonical ledger with `boundary_time_utc`, source/product/session/connection, source state time and sequence, derivation time, bid/ask prices and sizes, spread, midpoint, book sync state, eligibility and reason, and boundary schema version.
- `level2_updates` — individual updates with `event_time_utc`, `message_time_utc`, `ingest_time_utc`, source/product/session/connection/frame/sequence, event type, side, price, new quantity and L2 schema version.
- `bbo_state` — operational state with `state_time_utc`, `ingest_time_utc`, source/product/session/connection/frame/sequence, bid/ask prices and sizes, spread, midpoint, book sync state and book schema version.

## Local Data Lake

Real acquisition output is kept under a local `data_lake/` directory instead of the packaged synthetic fixture area.

```text
data_lake/
  raw/coinbase_advanced/{candles,product_book}/BTC-USD/<sha256>.json.gz
  raw/coinbase_advanced/{candles,product_book}/BTC-USD/<sha256>.meta.json
  normalized/coinbase_advanced/BTC-USD/<data_kind>/<granularity-if-candles>/date=YYYY-MM-DD/part-<sha256>.parquet
  manifests/<dataset_id>.json
```

`raw` is the immutable source-of-truth record, `normalized` stores derived analytical datasets, and `manifests` keeps acquisition provenance and reproducibility metadata.

### Optional manual Coinbase network smoke test

These commands require outbound access to Coinbase and are **not run in CI**:

```bash
btc-qh-fetch-coinbase-candles \
  --product BTC-USD \
  --start 2026-09-29T18:00:00Z \
  --end 2026-09-29T18:10:00Z \
  --granularity ONE_MINUTE \
  --output-root data_lake

btc-qh-snapshot-coinbase-book \
  --product BTC-USD \
  --output-root data_lake
```

Expect exact response bytes and HTTP metadata in `raw/`, dated validated Parquet under `normalized/`, and a dataset manifest under `manifests/`. Partial acquisitions retain completed raw responses but do not publish a success manifest or normalized output. Neither historical candles nor one-shot REST book snapshots qualify as canonical exact-boundary BBO targets.

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
- Coinbase WebSocket real envelope parsing (`channel`/`events` for `l2_data` snapshot/update and `heartbeats`, malformed frames/envelopes)
- L2 order-book stale-vs-gap sequence disposition (non-fatal duplicate/redelivered sequence numbers vs. fatal true gaps), crossed-book invalidation, and reset-vs-reconnect semantics
- deterministic final quarter-hour replay (late-arriving L2 mutations, per-update event times, inclusive boundary, heartbeat health at boundary, connection epochs, and explicit ineligibility reasons)
- WebSocket service subscribe/parse/heartbeat/reconnect-with-wait-for-snapshot behavior, injected clocks, and `connection_history` accumulation
- normalized `level2_updates`/`bbo_state` row capture and draining alongside quarter-hour BBO observations
- exact-byte raw segment sealing, content-addressing, and corrupt-framing detection, including malformed frames
- concrete websocket transport round-trip behavior against a loopback echo server (no internet access)
- end-to-end collector manifest completeness, normalized Parquet output for all three data kinds, reconnect-on-error handling, and sealed-source canonical replay distinct from provisional live state
- collector CLI operational summary output for both the preferred (`btc-qh-collect-coinbase-bbo`) and deprecated alias entry points

All WebSocket/collector tests are network-independent: the transport tests use a `websockets` server bound to `127.0.0.1:0`, and all other tests use an in-memory fake transport.

## Design notes

- Phase 2 expands the feature space while keeping every feature computable from information available at or before boundary `t`.
- Phase 3 adds microstructure-style features such as order-book imbalance, trade-flow imbalance, and average trade size using only contemporaneous or historical values up to boundary `t`.
- Phase 4 begins the model-diversity stage from the original roadmap by comparing linear, bagged-tree, boosted-tree, and gradient-boosted baselines under the same walk-forward protocol.
- Phase 5 starts the ensemble stage with simple averaging and validation-weighted averaging computed from chronologically prior fold performance only.
- Features are computed from historical windows ending at `t`; no feature reaches into `t+15m`.
- Flat moves where `P[t+15m] == P[t]` are left unlabeled and excluded from supervised training.
- The package layout is designed for later expansion into XGBoost, sequence models, microstructure models, and ensembles without changing the target definition.
