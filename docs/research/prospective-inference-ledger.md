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
Official verification status is assigned only inside this loader path; there is
no public bundle-registration API. Before official inference and persistence,
the in-process verified status and a fingerprint of the bundle declarations
and serialized fitted-model state are rechecked. The ledger also reruns both
verified models against the recorded ordered 91-feature vector and rejects
probabilities or classes that differ from that inference. This provenance
check assumes the Python process and its loaded code are trusted: it is not a
sandbox against hostile code with arbitrary introspection or memory access.

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

`ProspectiveRawFrameJournal` is an isolated append-only SQLite adapter. PR #34
adds an optional, disabled-by-default publisher at the existing collector's
raw-frame boundary and a separate local receiver. The synthetic integration
preserves raw bytes, collector receive times, session/frame/connection identity,
producer health diagnostics, and commits them to the source journal. It has not
been enabled on or validated against a production collector. The receiver
certifies watermarks only after an ordered post-cutoff producer fence follows
the durable contiguous frame prefix; the journal alone cannot establish
completeness without this producer/receiver protocol.
Once a journal watermark is committed, no frame with ingest time at or before
that watermark may be appended, and watermarks must advance strictly. This
preserves already-issued snapshots against backdated additions. The legacy
`advance_watermark()` API remains an offline adapter that accepts an explicit
caller assertion. The bridge instead uses the fenced-watermark API and records
the fence, producer-health evidence, and separately evaluated chrony status.
Those local checks are not a production host or filesystem durability
attestation; independent VM validation is still an activation prerequisite.

The current raw-segment writer remains authoritative for production collection.
The optional publisher is a separate read-only tee and does not change raw
segment or canonical target semantics. Its local queue is bounded and
nonblocking to the collector; overflow, transport failure, or ambiguous ack
invalidates subsequent progress rather than reconstructing missing data. The
receiver's default clock adapter reads chrony status, but its host evidence,
socket permissions, and storage durability still require VM validation. The
bridge does not implement a live scheduler or a connected 60-second prediction
service.

The original `build_causal_feature_vector()` / `process_boundary()` APIs still
support the PR #32 complete-snapshot offline path. The bounded alternative is
implemented in `research/prospective_replay.py` and
`process_incremental_boundary()`: versioned `_IngestFeatureReplay` checkpoints
reuse its existing `to_checkpoint()` / `from_checkpoint()` format and retain
the complete book, connection epoch and accepted sequence, prior midpoint,
counters, and exactly the required 240-minute bucket history. Each checkpoint
binds the session, last raw frame, ingest/event cutoffs, frozen V2 contract
versions, and raw-prefix commitment. Its integrity digest is verified before
replay at both public checkpoint APIs, even for caller-constructed objects.
Checkpoint files use write-once atomic linking, file fsync before publication,
and parent-directory fsync afterward on supported POSIX filesystems. Unsupported
platforms fail closed. The parent must already be durably provisioned.

`ProspectiveRawFrameJournal.tail_snapshot()` verifies the checkpoint's prefix
commitment and reads only contiguous, hash-verified records after that frame
through T. The incremental adapter restores state, rejects gaps, duplicate or
stale sequences, malformed records, at-or-before-T future exchange events,
wrong session identity, invalid checkpoint state, missing history, and
unsynchronized or crossed books. It uses the unchanged existing bucket
summaries and V2 feature builder, so parity is judged by exact ordered
predictors and canonical feature fingerprints. The adapter does not replace
the full-snapshot API used by existing callers and is not connected to a
live production collector. Checkpoint provenance uses a hash chain rather than the
full-snapshot frame-list digest; both commit to raw provenance, but this
serialization distinction must be preserved and explicitly reviewed in any
later ledger integration. The ledger schema and immutable evidence rules have
not been changed.

Small bootstrap snapshots via `snapshot_through()` retain their 100,000-frame
hard limit. Arbitrary initial history uses `warmup_chunk()` and
`accumulate_feature_replay_warmup()` with a separately serialized schema-2
`FeatureReplayWarmup`. Each chunk is source-chain verified, retains at most
17 buckets, and is persisted under a unique name before the next chunk.
Restart resumes after the verified prefix; at most an uncommitted chunk is
repeated. `finish_feature_replay_warmup()` alone promotes an exhausted,
cutoff-bound source prefix with the unchanged exact 240-minute readiness.
Neither public prediction API accepts warm-up or incomplete tails.

Normal journal initialization reads an immutable prefix-schema marker, not
historical row counts. Marker-less old journals fail closed unless explicitly
opened offline with `migrate_legacy_prefixes=True`; that one-time historical
scan validates and atomically commits the migration. It is not normal recovery.
After a publication failure that left a destination, `load_durable()` validates
and resynchronizes the file and parent without overwriting evidence; ordinary
`load()` makes no durability attestation. Orphan `.tmp` files are not selected.

Normal prediction recovery requires a checkpoint strictly before T; if one is
missing or a bootstrap/tail cap is exceeded, fail closed rather than replay an
unbounded archive or reconstruct a missed timely prediction.

The offline benchmark at `benchmarks/benchmark_prospective_replay.py`
separates checkpoint file reads, bounded source-tail reads, checkpoint restore,
incremental tail replay, feature generation, injectable model inference,
injectable persistence/durability attestation, and checkpoint advancement.
It reports per-stage median/p95/max, tail work, failures, and traced Python
memory across increasing session sizes. Its defaults use synthetic frames,
placeholder inference, and a synthetic FULL-sync SQLite sink. Neither the
synthetic runs nor a local VM demonstrate production T+60-second performance.
Architecture alternatives, VM acceptance checks, and offline commands are in
`docs/research/prospective-bounded-replay-adr.md`.

Do not mark completeness or clock guarantees verified until a separately
isolated VM source has been reviewed and tested. If either is unknown, do not
produce or persist an eligible prediction.

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
success event commit together or neither is published. The recorded `write_started_at_utc` is not persistence evidence.
After that transaction commits, a separate immutable durability attestation
records a clock-health-checked upper bound on when commit completion was
confirmed. Primary evaluation eligibility requires this attestation to be at
or before T+60 seconds. A late attestation is retained but ineligible; a crash
or clock failure before confirmation leaves prediction evidence
`durability_unconfirmed`, which remains ineligible after restart and is never
retroactively attested. Earlier prediction rows are never updated to add
confirmation data. Existing schema-version-1 ledgers have no commit
attestations; they are rejected for writing and must be preserved read-only.

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
