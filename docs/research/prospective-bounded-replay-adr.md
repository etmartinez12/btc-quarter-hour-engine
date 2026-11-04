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
Initial bootstrap reads only through its boundary and has the same 100,000
frame hard limit. An oversized or absent checkpoint must fail closed; never
replay an unbounded archived session to reconstruct a missed prediction.

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
git diff --check
```

The first checkpoint can be bootstrapped with
`ProspectiveRawFrameJournal.snapshot_through()`; older full-session snapshots
remain supported only up to the bootstrap cap. The source must retain a
checkpoint before the current boundary for normal operation.

The benchmark accepts injected model-inference and ledger callbacks through
`run_benchmark()` for local measurement. Its default inference is deterministic
and its default sink is a synthetic FULL-sync SQLite transaction plus
attestation; neither represents the official models or the prospective
evidence ledger. Report measured percentiles, maximum, failures, tail frames
per boundary, and traced Python memory together with the environment. Do not
state production deadline compliance from these local synthetic results.
