# ADR: Bounded prospective replay and source integration

Status: Offline implementation complete; live source integration not authorized.
Date: 2026-11-04.

## Decision

Use immutable, versioned `_IngestFeatureReplay` checkpoints and replay only a
contiguous raw tail through the prediction boundary. A checkpoint binds the
session start and ID, active connection, last frame index and accepted
Coinbase sequence, ingest/event cutoffs, complete synchronized order book,
prior midpoint and replay counters, retained 240-minute bucket history,
feature/feature-family contract versions, and a hash-chain commitment to its
raw prefix. Its canonical JSON envelope has a SHA256 integrity digest; a
checkpoint file is published with exclusive create and cannot overwrite a
previous checkpoint. `ProspectiveRawFrameJournal.tail_snapshot()` reads only
frames after the checkpoint through the requested boundary and verifies the
stored raw hashes and hash chain. It rejects missing indices, changed or
reordered records, stale/gapped sequences, mismatched source identity,
unverified clock/completeness, invalid replay state, and unsynchronized or
crossed books. Source tails have a finite hard cap of 100,000 frames per
boundary (the journal adapter can apply a lower cap); exceeding it fails
closed and requires more frequent checkpointing, not a larger unreviewed tail.
The single-snapshot bootstrap retains its 100,000-frame hard limit. Longer
initial histories use resumable accumulation, not a larger cap: journal
`warmup_chunk()` seeks after the last committed frame and reads at most the
configured chunk size plus one lookahead row (default 10,000, hard maximum
100,000). `accumulate_feature_replay_warmup()` verifies source commitments,
continuity, hashes and causal timestamps, then retains at most 17 buckets.
Persist each schema-2 `FeatureReplayWarmup` under a unique name before reading
the next chunk. Reloading it validates the complete envelope and nested state;
restart repeats at most the uncommitted chunk, never the processed archive.
Accumulation work is O(total initial frames), with bounded work per call and
memory proportional to one chunk's bytes, the active book, and 17 bucket
summaries, not the length of historical raw delivery.

Warm-up state is **not** a prediction checkpoint, even if it happens to contain
17 buckets. `finish_feature_replay_warmup()` requires a source-verified exhausted
tail through the exact target cutoff, including ties in ingest timestamps, and
then enforces the unchanged exact 240-minute history and synchronized-book
requirements. Both public prediction/advancement APIs independently validate
the ready envelope digest and full structure, including for directly
constructed or mutated dataclasses; both reject schema-2 state and partial
tails. A missing ready checkpoint suppresses prediction. Never reconstruct a
missed timely prediction from an archive.

### Publication and recovery durability

Provision the checkpoint directory durably before calling `write_once()`.
Publication probes directory synchronization, writes a same-directory
temporary file, flushes and fsyncs its content, atomically hard-links it to the
destination without replacement, and fsyncs the parent directory before
success. Unsupported directory sync, failed writes, failed hard links, and
sync errors fail closed. There is deliberately no Windows fallback: this
implementation requires POSIX directory-fsync semantics on a supporting
filesystem. Filesystems/storage that do not honor these semantics are not
validated by a successful local simulation.

A failure after linking can leave a complete but unacknowledged destination.
Do not delete or overwrite it. `load_durable()` validates it and syncs its file
and parent again before returning; malformed/partial destinations are rejected.
Ordinary `load()` validates content only, not a durability attestation. Duplicate
names still fail, including competing writers. Unlinked/incomplete `.tmp`
files are not checkpoints and are never auto-selected or automatically removed
on restart. A crash may leave an orphan temporary link; it cannot substitute
for an acknowledged destination.

### Journal initialization

Normal recovery reads an immutable, transactionally committed prefix-schema
marker. It does not enumerate sessions, count raw frames or commitments, or
rehash the source archive. Tail reads seek by the indexed receive-time/frame
identity tuple with a boundary upper bound, including when many frames share
a timestamp or a large future suffix exists. They do not select future raw
bytes. SQLite WAL recovery and filesystem I/O remain platform concerns to
exercise on the VM; this does not assert constant wall-clock recovery.

Existing journals without the marker require an explicit **one-time offline**
`ProspectiveRawFrameJournal(path, migrate_legacy_prefixes=True)` migration.
This intentionally scans historical sessions/frames, validates raw hashes,
indices, ingest order and any existing chain, and commits commitments plus the
marker atomically. Interrupted/failed migration rolls back and may require
repeating the migration; it is not normal prediction recovery. Default startup
fails closed until migration succeeds. Do not remove the marker or immutable
journal evidence to force recovery.

The adapter restores the existing order-book replay state and reuses the
existing feature bucket and V2 feature construction code. It does not change
the 12 candidate predictors, 91 control predictors, formulas, ordering,
readiness requirements, thresholds, or target. Completed vectors remain
causal: records ingested after T are not read by the bounded source adapter;
at-or-before-T frames with future exchange timestamps cannot be advanced into
a checkpoint. A missing or late boundary is not reconstructed as timely.
Checkpoint integrity is not a completeness attestation: the source watermark
and clock-health assertion remain independent prerequisites.

## Source alternatives for a later integration PR

| Option | Canonical target and timestamps | Completeness / identity | Isolation and latency | Storage / restart |
|---|---|---|---|---|
| **A. Read-only tee of the existing collector stream (recommended)** | Retains the existing Coinbase BTC-USD feed and collector receive timestamp, connection ID, and frame/sequence identity. The canonical target and source-time boundary-price method remain unchanged. | Best opportunity to share the collector's authoritative frame order, reconnect epochs, sequence gaps, and snapshot/rebuild transitions. Completeness still requires an explicit source watermark; a clock-health Boolean alone is insufficient. | A bounded, nonblocking queue avoids coupling inference latency to collection. Queue overflow or consumer failure must invalidate completeness and suppress prediction rather than block the collector or silently drop frames. Queue behavior and collector health need VM verification. | Persist raw frames/provenance and watermarks in an isolated inference journal; persist checkpoints atomically. Restart recovers only from verified prefix commitments and must not backfill timely predictions. |
| **B. Independent Coinbase WebSocket sidecar** | A second Coinbase connection observes the same product but is not guaranteed to have identical delivery time, sequence epoch, snapshot timing, or transient book path. Keep the existing canonical target price source; never blend the feeds. | Requires independent connection/session IDs, sequence/gap checks, snapshot recovery, raw capture, clock verification, and completeness watermarks. Two feeds do not prove each other's completeness. | Process isolation limits collector interference, but startup, network scheduling, snapshot rebuild, and extra subscription add latency and operational failure modes. A sidecar may be unavailable while collection continues. | Separate journal and checkpoint lifecycle; restart begins a new connection epoch and must resynchronize from a snapshot. |
| **C. Consume committed raw-segment files** | Uses the same collector's raw bytes and target definition. | Segment metadata provides durable per-frame provenance after sealing, but the currently active partial segment does not expose a proven complete low-latency frame stream. Its watermark and receive-time completeness are not suitable evidence for T+60. | No live collector hook, but waiting for segment sealing adds uncertain latency. Reading a partial file does not establish that all frames through T were delivered or durably recorded. | Simple offline recovery from sealed immutable segments; not a defensible live prediction source without a separately designed, verified live journal. |

Recommendation: **Option A**, contingent on a separately reviewed bounded
read-only tee from the existing collector and a fail-closed queue/watermark
contract. It reuses the canonical stream and preserves collector ownership.
No collector hook, second feed, canonical-source change, scheduler, or live
inference is implemented here. Option C remains appropriate for offline
rebuilds only; current raw segments are not established as low latency because
their complete per-frame ingest metadata is unavailable until sealing.

## Boundaries and operational validation

This implementation is offline-only. `ProspectiveRawFrameJournal` is an
isolated test/research adapter; it is not connected to `WebSocketCollector`,
the materializer, or a service. Existing protocol timing, source price, model
artifacts, ledger evidence semantics, and prediction schedule are unchanged.
No synthetic measurement is a production T+60-second claim.

Before any later integration or activation, an operator/reviewer must verify
on the intended VM and under the production service identity:

1. The selected source is complete through each T, and receive timestamps,
   UTC clock synchronization, session identity, frame indices, Coinbase
   sequence continuity, snapshot/rebuild and reconnect transitions are
   independently observable. Do not equate sealed-segment timestamps or a
   producer Boolean with proof of completeness.
2. The read-only queue cannot block or degrade the collector; overflow,
   process failure, disk-full, clock skew, and unclean restart fail closed and
   produce no eligible prediction. Confirm unchanged collector/materializer
   health and canonical raw storage.
3. Checkpoint files and raw-prefix commitments survive process/host restart;
   corrupted, missing, reordered, duplicated, stale, wrong-session, wrong-
   connection, or discontinuous data is rejected. Verify that recovery never
   changes an existing prediction or marks a reconstructed one timely.
4. On actual VM hardware, benchmark source reads, checkpoint restore, bounded
   tail work, feature generation, frozen-model inference, ledger transaction,
   post-commit durability attestation, and Python plus OS-level peak memory.
   Exercise increasing session sizes, cold and warm restarts, load, and
   failure paths. A synthetic benchmark does not establish this prerequisite.
5. Re-run the unchanged prospective protocol's model-hash, feature-order,
   causality, ledger immutability, deadline, restart, and duplicate-boundary
   gates. Obtain a separate implementation review and explicit activation
   approval before enabling anything live.

Offline checks:

```powershell
python -m pytest tests/test_prospective_replay.py tests/test_prospective_inference.py
python -m pytest
python benchmarks/benchmark_prospective_replay.py --frame-counts 256,1024,4096 --boundaries 12
python benchmarks/benchmark_prospective_replay.py --frame-counts 100001,200001 --history-hours 24 --warmup-chunk-frames 10000 --boundaries 3 --verify-full-replay
git diff --check
```

On Windows, the benchmark requires the explicit
`--simulate-checkpoint-publication` option. Its output labels publication as
simulated, and it provides **no filesystem crash-durability evidence**. Unit
fault/race simulations also do not replace the POSIX publication/recovery test
or actual VM power-loss/restart testing.

For long histories, repeatedly request `warmup_chunk()` using the persisted
warm-up's `last_frame_index` and `prefix_commitment_sha256` (initially `-1` and
`initial_prefix_commitment(session_id, started_at)`). Accumulate and durably
publish/reload each result. When `has_more` is false, either pass the final
chunk and the *previous* warm-up to `finish_feature_replay_warmup()`, or request
an empty exhausted chunk after the final persisted state and finish with that
state. Source chunks bind their exhaustion assertion to
`through_boundary_utc`; a later watermark alone is not exhaustion evidence.
Select the known last acknowledged state explicitly; never choose arbitrary
temporary files or the highest filename as proof of durability.

`snapshot_through()` / `create_feature_replay_checkpoint()` remain small offline
bootstrap conveniences, not arbitrary-history recovery. The source must retain
a ready checkpoint strictly before the current boundary for normal operation.

The benchmark accepts injected model-inference and ledger callbacks through
`run_benchmark()` for local measurement. Its default inference is deterministic
and its default sink is a synthetic FULL-sync SQLite transaction plus
attestation; neither represents the official models or the prospective
evidence ledger. Report measured percentiles, maximum, failures, tail frames
per boundary, and traced Python memory together with the environment. Do not
state production deadline compliance from these local synthetic results.
Large-history runs stream synthetic source creation, report chunk sizes,
warm-up serialized sizes, warm-up allocation peaks and journal restart timing,
and optionally compare all predictors, values and fingerprints to full replay.
The optional full reference intentionally materializes complete synthetic
history **outside** bounded-path allocation measurements and timings.
`--fast-synthetic-fixture` optionally skips per-commit fsync only while creating
the synthetic source fixture, outside all reported stages; FULL synchronization
is restored before the source watermark. Output labels this mode. It is not
source crash-durability evidence and must never be applied to a real journal.

## Correction verification (local, 2026-11-04)

Environment: Windows, Python 3.10.19, SQLite 3.50.2. No VM, Coinbase source,
frozen dataset/model, deployment, merge, or live-prediction activation was used.

- Full `python -m pytest -q`: **725 passed, 1 skipped**, 22 existing warnings.
- Focused replay/inference tests: **45 passed, 1 skipped**.
- The skip is the real POSIX directory-fsync publication/recovery test.
  Windows runs labeled publication fault/race simulations and explicitly tests
  that real unsupported publication fails closed.
- `python -m compileall -q btc_quarter_hour_engine tests benchmarks` and
  `git diff --check`: passed.
- Adversarial coverage includes directly constructed/mutated checkpoints with
  valid statistics, stale integrity digests and matching source commitments;
  recomputed digests over invalid structure; partial/schema-2 prediction
  rejection; exact-history readiness; pre-snapshot warm-up; eight-hour
  checkpoint/full-replay parity across persisted chunks and journal restarts;
  migration rollback and scan-free normal startup; bounded indexed seeking
  across large timestamp ties; interrupted writes; file/link/directory-sync
  failures; competing destination names; orphan files; and publication recovery.

Final synthetic command:

```powershell
python benchmarks/benchmark_prospective_replay.py --frame-counts 100001,200001 --boundaries 3 --history-hours 24 --warmup-chunk-frames 10000 --simulate-checkpoint-publication --fast-synthetic-fixture --verify-full-replay
```

| Metric | 100,001 historical frames | 200,001 historical frames |
|---|---:|---:|
| Historical duration | 24 hours | 24 hours |
| Resumable warm-up chunks | 11 | 21 |
| Maximum frames per chunk | 10,000 | 10,000 |
| Maximum serialized warm-up bytes | 15,410 | 15,401 |
| Traced warm-up peak bytes | 12,801,881 | 12,740,205 |
| Initial checkpoint accumulation + simulated persistence | 32.07 s | 66.77 s |
| Journal reopen (one measurement per case) | 3.36 ms | 5.32 ms |
| Boundaries completed / failures | 3 / 0 | 3 / 0 |
| Tail frames per boundary (median / max) | 2 / 2 | 2 / 2 |
| Feature generation median / p95 / max | 147.14 / 177.16 / 177.16 ms | 106.43 / 119.50 / 119.50 ms |
| Steady-path traced Python peak bytes | 566,963 | 557,443 |
| Full-replay ordered predictors, values and fingerprint | Exact match | Exact match |

These are only three boundaries per case, not production percentile estimates.
Source fixture generation skipped per-commit fsync outside measured stages;
FULL synchronization was restored before watermarking. Checkpoint publication
was **simulated**, not crash-durable Windows publication. Inference was a
placeholder and the ledger sink was synthetic. Full-reference replay
materialized complete synthetic history outside bounded-path measurements.
Traced allocations exclude native/OS memory. The initial accumulated work
increases with total history; restart and each accumulation call do not replay
that processed history.

Remaining separately authorized VM gates include the real POSIX test and
filesystem power-loss/restart behavior, provisioned-directory durability,
source and clock completeness, cold/WAL recovery, queue/collector isolation,
OS-level memory and load, actual frozen-model/immutable-ledger timing, and all
unchanged prospective deadline/duplicate/causality gates. These local results
do not establish T+60 compliance or authorize deployment/activation.
