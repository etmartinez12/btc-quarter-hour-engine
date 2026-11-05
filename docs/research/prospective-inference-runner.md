# Prospective inference runner (PR #35)

## Status and safety boundary

This is a separately invoked consumer of the existing PR #34 bridge journal,
the bounded PR #33 feature replay, the official V2 model loader, and the
append-only prospective ledger. Importing the module, installing the command,
or validating a configuration does not start inference. This change does not
enable a collector publisher, source receiver, service, timer, cron job, or
systemd unit. No research materializer, target construction, or model-training
path is called by the runner.

The protocol remains exactly the registered 2,880 UTC quarter-hours from
`2026-11-15T00:00:00Z` through (exclusive) `2026-12-15T00:00:00Z`, with target
`T+15m`, the fixed 0.5 threshold, and the durable T+60s deadline. Only
`load_official_models()` can supply an accepted official model bundle.

## Processing order

For one exact scheduled boundary `T`, the explicit runner:

1. Confirms `T` is in the immutable ledger schedule, that it is not early, and
   that the current wall time has not passed `T+60s`. Historical boundaries
   are never recreated as timely predictions.
2. Takes the nonblocking single-writer process lock for the ledger.
3. Checks independent chrony/UTC evidence and the ledger wall/monotonic clock
   health. Rechecks them while awaiting the source and immediately before
   inference.
4. Requires the configured source-session ID to exist and be in `bridge` mode.
   The runner opens the source SQLite file with `mode=ro` and `query_only`;
   it does not initialize, migrate, or append to that journal.
5. Loads the latest valid, durable, immutable ready checkpoint strictly before
   `T`. On a cold start it resumes or performs the separately persisted
   four-hour warm-up to `T-15m`, one bounded chunk at a time. Every continuation
   is published before reading the next chunk.
6. Waits with blocking sleeps (not a spin loop) for the *exact* bridge fence at
   `T`. Missing-fence polling is bounded by the deadline; the runner never
   substitutes an offline watermark, a later fence, or a stale session.
7. Reads the contiguous hash-verified raw tail through `T`, calls
   `process_incremental_boundary()`, and relies on its existing causal feature,
   official-model, ledger, and durability-attestation checks.
8. Only after the prediction transaction and post-commit attestation succeed,
   advances replay state through `T` and publishes a new write-once checkpoint.
   A checkpoint publication failure does not alter or repeat the committed
   prediction; it returns `prediction_committed_checkpoint_recovery_required`.
9. Returns structured per-boundary status and stage measurements. Failures are
   events, never fabricated prediction rows.

Ledger lifecycle is checked before work. Existing predictions, late or
unconfirmed commits, labels, and missed boundaries are not overwritten or
re-attested. `recover_committed_checkpoint()` is not implicitly run against an
old boundary; checkpoint recovery must be scheduled independently so that it
cannot hold up a later boundary.

## Trust boundaries and recovery

* **Source:** the configured bridge session and exact fence are mandatory.
  Existing fence validation checks producer-health and canonical boundary
  eligibility, synchronized order book, independent chrony evidence, witness
  frame, and source-prefix commitments. `tail_snapshot()` checks the source
  prefix, frame hashes, order and continuity. The runner is not a source
  receiver and never manufactures a watermark.
* **Checkpoint:** checkpoint envelopes are digest-validated and are
  write-once. `load_durable()` synchronizes the file and containing directory;
  unsupported filesystems/platforms fail closed. Every checkpoint is matched
  to the configured source session and later reconciled against the journal
  prefix when the tail is requested. Checkpoint files for another session are
  never inherited.
* **Prediction:** the ledger remains authoritative. Its immutable prediction
  row, transaction, clock health, and separate durable attestation determine
  whether a result is eligible. A crash after a commit but before the
  attestation remains unconfirmed and is never retrospectively attested.
* **Concurrency:** a process-level lock serializes runner ownership of one
  ledger. SQLite's existing immediate transactions remain authoritative for
  writes.
* **Clock:** chrony evidence is checked for synchronization, normal leap
  status, offset within 50 ms, and a fresh reference/observation. `ClockHealth`
  separately detects wall/monotonic jumps. A local `ClockHealth` Boolean or
  CLI option cannot substitute for chrony evidence.

Restart uses the latest valid durable checkpoint and immutable ledger state.
The normal scheduler selects only the current boundary. If the checkpoint,
source prefix, session, four-hour history, frame cap, clock evidence, or model
integrity cannot be verified, it fails closed. Missing source fences are
polled only until the T+60 deadline; terminal status is recorded without
repetitive polling events. Collector interruption or source reconnection that
breaks continuity makes the tail ineligible.

## Modes and exact commands

Install the package in the reviewed environment first. Example JSON
configuration (all four paths must be separate; pre-create the checkpoint
directory and do not point at production files for tests):

```json
{
  "source_journal_path": "/srv/btc-qh/prospective-source.sqlite",
  "checkpoint_directory": "/srv/btc-qh/prospective-checkpoints",
  "prediction_ledger_path": "/srv/btc-qh/prospective-ledger.sqlite",
  "official_model_directory": "/srv/btc-qh/frozen-models",
  "implementation_git_sha": "<40 lowercase hex characters>",
  "expected_source_session_id": "<reviewed bridge session ID>",
  "operational_authorization_id": "<separately approved deployment ID>",
  "source_poll_seconds": 0.25,
  "max_tail_frames": 100000,
  "warmup_chunk_frames": 10000
}
```

**Configuration and model validation only** (loads/verifies artifacts, creates
no ledger and does not infer):

```powershell
btc-qh-prospective-inference-runner validate --config .\runner.json
```

**Read-only journal diagnostic** (does not load models, open a prediction
ledger, infer, or write prediction evidence):

```powershell
btc-qh-prospective-inference-runner dry-run `
  --config .\runner.json --boundary 2026-11-15T00:00:00Z
```

**Synthetic schedule/deadline smoke check** (no source, model, or ledger):

```powershell
btc-qh-prospective-inference-runner simulate --count 2880
```

**Official mode** is a distinct command and requires both an explicit
authorization ID matching the reviewed configuration and a second deployment
acknowledgement. The host environment must independently provide the same
authorization via `BTC_QH_PROSPECTIVE_AUTHORIZATION_ID`; it is not a clock
health assertion. It must also set `BTC_QH_IMPLEMENTATION_GIT_SHA` to the
reviewed implementation SHA. Example syntax only—**do not run until all
activation gates are separately approved**:

```powershell
$env:BTC_QH_PROSPECTIVE_AUTHORIZATION_ID = "<approved deployment ID>"
$env:BTC_QH_IMPLEMENTATION_GIT_SHA = "<reviewed implementation SHA>"
btc-qh-prospective-inference-runner official --config .\runner.json `
  --authorization-id "<approved deployment ID>" `
  --confirm-authorized-deployment
```

Official mode is not used by tests or by this implementation task. No CLI
smoke test invokes it.

## Failure classification and diagnostics

The one-boundary result distinguishes `not_due`, `predicted`,
`prediction_committed_checkpoint_recovery_required`, `waiting_for_source_fence`,
`missing_source_data`, `invalid_source_health`, `checkpoint_failure`,
`clock_sync_failure`, `model_verification_failure`,
`inference_or_persistence_failure`, `duplicate_prediction`, `late_prediction`,
and `missed_deadline`. Existing ledger event types are reused; the ledger
remains append-only.

Runner metrics include source-fence wait/availability, tail read and
verification, checkpoint load, checkpoint restore, bounded tail replay,
feature generation, candidate/control inference, SQLite transaction,
post-commit attestation, checkpoint publication, total completion, deadline
margin, independent UTC checks immediately before and after the ledger
operation, startup model/source validation, CPU time, resident memory where the
OS exposes it, and unpredicted boundary backlog. Checkpoint-load timing is the
restart-recovery measurement. The offline replay benchmark reports median, p95, p99, and
maximum stage latency and traced Python allocation peak. Its synthetic
inference and SQLite sink are clearly separate from the verified official
models and production durability; neither is evidence of T+60 compliance.
The official CLI emits structured JSON-line operational summaries. Those
summaries omit candidate/control probabilities, predicted classes, and raw
feature values so monitoring does not disclose comparative results before the
registered analysis.

## Activation and VM acceptance (not performed)

Before any official execution, separately review and authorize the protocol,
implementation commit, frozen manifest/models, implementation SHA, paths,
session binding, and operational authorization. On an isolated VM, with no
production collector/materializer impact:

1. Validate official artifact SHA256s and dependency/Python versions against
   the freeze manifest.
2. Verify bridge-only operation, exact certified fences, canonical
   eligibility, independent chrony freshness, and source receiver continuity
   across reconnects and interruptions.
3. Run synthetic and pre-window read-only diagnostics; prove post-T frames
   cannot affect features and target/label data is absent from inference.
4. Exercise every crash boundary: warm-up continuation, checkpoint link/fsync,
   prediction transaction, attestation, restart reconciliation, and
   checkpoint-advance failure after a successful prediction.
5. Run representative official-model inference and storage durability
   benchmarks on the target VM. Record stage median/p95/p99/max, RSS, backlog,
   clock evidence, disk/filesystem semantics, and measured T+60 margin. Repeat
   under realistic source load and collection interruption. Synthetic
   timings do not qualify.
6. Confirm the collector and canonical materializer remain unchanged, test
   graceful shutdown/restart, and verify no duplicate or retroactively timely
   predictions can be written.
7. Approve deployment separately and create any service configuration only
   through the normal production change process. This PR creates no enabled
   service, timer, cron entry, or collector hook.

This implementation has not received independent VM or production acceptance.
The source bridge's host/socket/disk properties, production chrony, genuine
frozen model load/inference latency, filesystem durability and deployment
authorization remain prerequisites. In particular, the existing immutable
checkpoint publisher currently requires POSIX directory `fsync`; Windows
local development cannot attest checkpoint publication and must fail closed.
