# Prospective inference and ledger implementation

This code implements offline causal feature replay, frozen-model inference, and
an append-only SQLite ledger for the unchanged V2 prospective protocol. It does
not start a service, connect to Coinbase, alter the collector or materializer,
or authorize prospective execution.

## Frozen model gate

`load_official_models()` calls `load_frozen_models()` with official-only
enforcement and the independently trusted manifest SHA256
`a52589a103a36a976adda49d65de38bee2c0ab656c2f4f13caa4c41067556d13`. It then
checks the exact range-only ExtraTrees SHA256
`db90d2eeb5b15a2a4b730cabcf8f9f2557e5816bcc02ed117a3acafc7a1f86e9` and full
V2 ExtraTrees SHA256
`c8d2bf2289faf567896d1f514a3daf5d8c17d7a8b6cacddc1a700cb517d47fed`.
Development-only freezes and any changed manifest or model artifact are
rejected before inference. No model is fit, selected, or modified here.

## Causal source contract and unresolved integration

`build_causal_feature_vector()` accepts only an immutable
`RawFrameSnapshot` containing the session start, a source watermark, an explicit
clock-health assertion, and raw Coinbase frames with per-frame ingest time,
connection ID, frame index, and SHA256. The watermark must cover T. Every frame
with ingest time after T is excluded before parsing. For included frames, the
Coinbase envelope time and every normalized exchange event/update time must be
at or before T; a frame with a future event time is excluded in full. Malformed
included frames, sequence gaps, stale frames, an unsynchronized book, incomplete
four-hour history, missing/nonfinite values, or any mismatch fail closed.

Features use the existing ingest-order `_IngestFeatureReplay`,
`_window_features()`, and `build_v2_research_feature_frame()` implementations.
The label, target price, target eligibility, and target outcome are absent from
the source API. The V2 contract produces the exact 91 controls; the frozen
range/volatility family selects its exact 12 ordered candidate predictors.
Predictions use only the two frozen models, class order `[0, 1]`, and the fixed
0.5 threshold. The baseline is always class 0. Feature fingerprints are
recomputed by both inference and ledger validation so changed predictor values
cannot be recorded under an earlier fingerprint.

`ProspectiveRawFrameJournal` is an isolated append-only SQLite adapter for a
future source integration. It durably records raw bytes, per-frame ingest
metadata, session identity, and explicit completeness watermarks; it is not
connected to or tested against the production collector. A future producer
must supply trustworthy receive times and prove the watermark means all frames
through that time have arrived. Neither property can be established by the
local journal alone.

The current raw-segment writer makes frame provenance durable when segments are
sealed, but its per-frame ingest-time metadata is not published as a separate
low-latency feed while a segment is active. This implementation therefore
provides a strict snapshot adapter and offline replay interface, not a connected
60-second production source. The snapshot's watermark and
`clock_health_verified` value are external assertions; the local adapter cannot
prove that a separate source has delivered every frame through T or that the
host clock is synchronized to UTC. Do not mark those guarantees verified until
a separately isolated VM source has been reviewed and tested. If either is
unknown, do not produce or persist an eligible prediction.

The SQLite `ClockHealth` monitor detects wall-clock regressions and wall versus
monotonic-clock jumps after its baseline check. It does not independently verify
NTP/chrony synchronization or absolute UTC accuracy. Production operation must
add and validate that external host-time health signal before activation.

## Offline dry run

The CLI accepts an explicitly offline, target-free JSON snapshot and is
intentionally dry-run-only. It prints evidence to stdout and never persists a
prediction or contacts a market-data source:

```powershell
python -m btc_quarter_hour_engine.research.prospective_dry_run_cli `
  --dry-run `
  --source-snapshot "C:\research\synthetic-source-snapshot.json" `
  --boundary "2026-11-15T00:00:00Z" `
  --model-directory "C:\research\official-model-freeze" `
  --implementation-git-sha "<full-reviewed-commit-sha>" `
  --generated-at "2026-11-15T00:00:01Z"
```

The snapshot JSON has exactly these top-level keys:
`session_id`, `session_started_at_utc`, `watermark_utc`,
`clock_health_verified`, and `frames`. Each frame has exactly
`raw_hex`, `session_id`, `connection_id`, `frame_index`, `ingest_time_utc`, and
`raw_frame_sha256`. Unknown keys (including target or label fields), duplicate
JSON keys, and invalid provenance are rejected.

## Ledger operation

Constructing `ProspectiveLedger(path)` registers all 2,880 boundaries in
`2026-11-15T00:00:00Z` through `2026-12-15T00:00:00Z`. An existing database
with a different schedule is rejected. Schedule, events, predictions, and
labels are protected by SQLite update/delete triggers; prediction and label
boundary identities are unique. SQLite uses `WAL`, `synchronous=FULL`, foreign
keys, and immediate transactions. Prediction data and the corresponding
success event commit together or neither is published.

Only complete, ready predictions with exact frozen hashes and feature ordering
can be written. The ledger derives timing eligibility from the generation and
persistence timestamps. The inference runner takes generation time from the
ledger's clock-health source immediately after model execution; persistence
time is independently taken inside the durable write transaction. Late writes
create a `late_attempt` event and never a prediction row.
`mark_missed_boundaries()` appends missed events after deadline without
generating or backfilling predictions. Labels are appended to a separate
immutable table after their availability time; they cannot update prediction
evidence. Duplicate and stale attempts are retained as events.

## Activation boundary

There is no enabled service, scheduler, collector hook, or automatic inference
process in this change. Before any activation, implementation review, explicit
approval, VM integration testing, proof of the complete low-latency source
watermark, independent clock synchronization monitoring, and operational
recovery testing remain required by the protocol.
