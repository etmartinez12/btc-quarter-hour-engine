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

1. Takes the nonblocking single-writer process lock for the ledger.
2. Confirms `T` is in the registered schedule, rejects early work, and
   reconciles the immutable ledger lifecycle before attempting inference.
   Expired boundaries are never recreated as timely predictions.
3. Checks independent chrony/UTC evidence and the ledger wall/monotonic clock
   health. Rechecks them while awaiting the source and immediately before
   inference.
4. Requires the configured source-session ID to exist and be in `bridge` mode.
   The runner opens the source SQLite file with `mode=ro` and `query_only`;
   it does not initialize, migrate, or append to that journal.
5. Loads the latest valid, durable, immutable ready checkpoint strictly before
   `T`. Missing or corrupt state fails closed. The deadline path never runs
   initial warm-up: explicit `prepare` must have completed it beforehand.
6. Waits with blocking sleeps (not a spin loop) for the *exact* bridge fence at
   `T`. Missing-fence polling is bounded by the deadline; the runner never
   substitutes an offline watermark, a later fence, or a stale session.
7. Reads the contiguous hash-verified raw tail through `T`, calls
   `process_incremental_boundary()`, and relies on its existing causal feature,
   official-model, ledger, and durability-attestation checks.
8. Forces a new independent UTC-health check before the prediction write and
   again after its COMMIT, inside the separate attestation transaction, after
   acquiring its write lock and before taking the attestation timestamp.
   Only after the prediction transaction and post-commit attestation succeed,
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
On startup, the scheduler floors UTC to the most recent exact quarter-hour,
clamped to the registered start. At T+10 it reconciles T, not T+15m. Ledger
prediction identity (including late/unconfirmed commits), labels, and any
permanent `missed` event prevent another attempt, even if a later diagnostic
is the most recent event. The T+60 timestamp is inclusive, as in the ledger;
any actual generation/commit/confirmation delay past it is ineligible.
After one result the cursor advances at least one full quarter-hour. Forward
clock jumps skip expired work via missed-boundary reconciliation, never
through historical inference or checkpoint repair. Shutdown interrupts a
scheduled wait at its next bounded wait step.

Only SQLite BUSY/LOCKED **reads** are retried, at most three attempts, with
blocking, deadline-capped waits. No prediction transaction or attestation is
retried. Read/diagnostic persistence failures are surfaced in structured
results rather than silently treated as success. If the checkpoint,
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

For an installed-wheel deployment, also set
`"implementation_attestation_path"` (absolute path) and
`"implementation_attestation_sha256"` (the reviewed pin); see
[Runtime implementation provenance](#runtime-implementation-provenance).
Both are omitted (or `null`) for Git-checkout execution.

**Configuration and model validation only** (verifies the executing
implementation's provenance, loads/verifies artifacts, creates no ledger and
does not infer):

```powershell
btc-qh-prospective-inference-runner validate --config .\runner.json
```

**Read-only journal diagnostic** (does not load models, open a prediction
ledger, infer, or write prediction evidence):

```powershell
btc-qh-prospective-inference-runner dry-run `
  --config .\runner.json --boundary 2026-11-15T00:00:00Z
```

**Explicit checkpoint preparation** (source reads and immutable checkpoint
writes only; no model loading, prediction-ledger opening, prediction records,
or deployment authorization):

```powershell
btc-qh-prospective-inference-runner prepare `
  --config .\runner.json --boundary 2026-11-15T00:00:00Z
```

For the first registered prediction, the normal cutoff is
`2026-11-14T23:45:00Z`. The separately authorized source must contain the exact
240 minutes of history through that cutoff: its session must begin no later
than `2026-11-14T19:45:00Z` (prefer earlier for operational margin). This is a
source-readiness requirement, **not authorization to enable the bridge**.
Begin `prepare` as soon as the certified 23:45 fence exists, allowing the
whole pre-window interval for replay and durable publication. Do not defer it
until midnight. Benchmark that budget on the isolated VM beforehand.

Preparation verifies the source mode/session, exact cutoff fence, independent
UTC evidence, canonical eligibility, causal raw prefix and every bounded
chunk. Each continuation is durably published before the next chunk is read.
Interruptions resume from the latest trustworthy continuation for that exact
cutoff. The replay API still enforces all 17 exact quarter-hour buckets and
the 240-minute history; a continuation is not a ready checkpoint. Preparation
checks time between chunks and around ready publication, and refuses work
at/after the prediction boundary. Partial/corrupt state is never promoted.

Before midnight, require a successful JSON result with `ready: true`,
`cutoff_utc: "2026-11-14T23:45:00Z"`, the expected source session, immutable
checkpoint path and integrity SHA256. Rerunning `prepare` reopens and
revalidates the published checkpoint against its exact certified source
prefix without replaying the four hours or touching the ledger; the result
then includes `recovered_ready_checkpoint: true`. Record the successful
readiness result before T. A failed or incomplete preparation is not approval
to cold-start inference inside T+60. Later prediction-state advancement
remains permitted from an older trustworthy checkpoint if the verified tail
fits the existing replay limit.

**Synthetic schedule/deadline smoke check** (no source, model, or ledger):

```powershell
btc-qh-prospective-inference-runner simulate --count 2880
```

**Official mode** is a distinct command and requires both an explicit
authorization ID matching the reviewed configuration and a second deployment
acknowledgement. The host environment must independently provide the same
authorization via `BTC_QH_PROSPECTIVE_AUTHORIZATION_ID`; it is not a clock
health assertion. It must also set `BTC_QH_IMPLEMENTATION_GIT_SHA` to the
reviewed implementation SHA. That variable is only an operator
acknowledgement: before opening the ledger, official mode independently
verifies the executing code (see
[Runtime implementation provenance](#runtime-implementation-provenance)) and
refuses to start if it cannot. Example syntax only—**do not run until all
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

## Runtime implementation provenance

The ledger's `implementation_git_sha` must describe the code that is actually
executing. `RunnerConfig.implementation_git_sha` is the reviewed
*expectation*; the identity written to the ledger comes only from
`prospective_provenance.verify_runtime_implementation()`, which inspects the
imported `btc_quarter_hour_engine` package locally (no GitHub or other network
access). Matching environment variables are never treated as proof. There
are exactly two accepted modes; anything else fails closed.

**Git-checkout mode** (no attestation configured). The imported package
directory must be the top-level package of a Git work tree, and:

1. `git rev-parse --show-toplevel` must be that package's parent directory.
   Inherited `GIT_*` environment variables (for example `GIT_DIR`) are
   removed so they cannot redirect the check.
2. `HEAD` must be exactly the expected 40-hex commit.
3. `git status --porcelain=v1 --untracked-files=all --ignore-submodules=none`
   must be empty for the **whole** checkout (modified, staged, deleted,
   renamed and untracked files all fail). Keep runtime data such as ledgers,
   journals, checkpoints and logs outside the checkout.
4. No index entry may be flagged assume-unchanged or skip-worktree.
5. Every regular file in the package must hash, in Python, to the blob
   recorded for it in `HEAD` (LF/CRLF checkout normalization is accepted), and
   the package file set must equal the tracked set. This independently
   detects edits that `git status` was made to overlook. Links, special
   files, and anything except `.pyc` files inside `__pycache__` are refused.

**Installed-wheel mode** (attestation configured). Git metadata is not
assumed. The required deployment artifact is an **implementation
attestation** JSON (`schema: "btc-qh-implementation-attestation/v1"`) produced
at build time by `attest-build` from a clean, verified checkout. It records
the distribution, version, package, Git SHA, wheel filename, wheel SHA256,
every package file's SHA256 and size, and a package content digest. The
reviewed configuration pins the attestation's own SHA256. At runtime:

1. The attestation file's SHA256 must equal the pin before it is parsed;
   parsing then rejects duplicate keys, non-finite numbers, unknown or missing
   fields, a different package, or an inconsistent digest.
2. The attested Git SHA must equal the expected implementation SHA.
3. The installed package files must equal the attested inventory exactly (no
   extra, missing, or changed files).
4. Exactly one installed distribution must own the package, with a matching
   version and a `RECORD` whose package entries agree with the attestation.
   An editable install is refused (use Git-checkout mode), and a recorded
   `direct_url.json` archive hash must equal the attested wheel SHA256.

Without a configured attestation, a package outside a Git checkout is
**unverifiable** and refused. A partial attestation configuration (path
without pin, or pin without path) is refused. Until the attestation artifact
exists and its SHA256 is pinned in the reviewed configuration, wheel
execution fails closed.

**Loaded modules (both modes).** Every already-imported
`btc_quarter_hour_engine` module, including the package itself and a package
`__main__`, must have been imported by the standard `SourceFileLoader` from a
file in the verified set, with consistent `__file__`/origin/cache locations and
an unaltered package search path. Custom loader subclasses, modules from
another location, extension or sourceless loaders are refused.

**Bytecode caches (both modes).** Source hashes alone do not authenticate
`.pyc` payloads: a timestamp/size header or even a checked source-hash header
can remain valid while the cached instructions are replaced. For every
verified Python source file, including not-yet-imported modules, the verifier
derives the cache path for the current interpreter and optimization level
(including an external `PYTHONPYCACHEPREFIX`). If a cache exists, it requires
a valid header and code payload, and compares the complete code object,
including nested code/constants, stack sizes and line/exception tables,
against a fresh compilation of the hash-verified source with matching
interpreter optimization. No cached code
is executed by this comparison. Missing caches are allowed; malformed,
unchecked-hash, mismatched, or unverifiable active caches fail closed.
Legitimate installer staging filenames are supported, as CPython adjusts
`co_filename` when loading a relocated source cache. Caches for other Python
versions or optimization levels are not selected by this interpreter.

**Binding and rechecks.** The first verification occurs in `validate`, at
`official` startup before the ledger is opened, in `prepare`, and in
committed-checkpoint recovery. The runner keeps the verified identity and
rechecks it once per boundary before inference (package location, every file
hash/size, active bytecode payloads, and the loaded-module set). The ledger's `implementation_git_sha`
for each prediction is the verified identity's SHA. Checkpoint preparation
reports also include the mode and package content digest. Any failure
returns `implementation_provenance_failure`, writes an `inference_failed`
diagnostic event, and creates no prediction. It is not retried, and drift
after startup blocks every later boundary.

Commands (they verify and print JSON only: no model load, source read, ledger
or prediction write):

```powershell
# Git checkout: verify the executing checkout.
btc-qh-prospective-inference-runner verify-implementation `
  --implementation-git-sha "<reviewed SHA>"

# Build host, from a fresh clean checkout of the reviewed commit:
Remove-Item -Recurse -Force .\build -ErrorAction SilentlyContinue
python -m pip wheel . --no-deps --no-build-isolation -w <dist directory>
btc-qh-prospective-inference-runner attest-build `
  --wheel <dist directory>\btc_quarter_hour_engine-0.1.0-py3-none-any.whl `
  --output <artifact directory>\implementation-attestation.json `
  --implementation-git-sha "<reviewed SHA>"

# Target VM, after installing that exact wheel:
btc-qh-prospective-inference-runner verify-implementation `
  --implementation-git-sha "<reviewed SHA>" `
  --attestation <absolute path>\implementation-attestation.json `
  --attestation-sha256 "<attestation_sha256 from attest-build>"
```

`attest-build` runs from the verified Git checkout, requires the wheel's
package contents to match that checkout byte for byte, and writes the
attestation exclusively (an existing output is never overwritten). Its
`attestation_sha256` output is the value to review and pin. Remove stale
`build/` output before building, because setuptools can otherwise copy files
that are no longer in the checkout (the content check then fails). The
package content digest depends on line endings, so a CRLF checkout and an LF
build differ; only the Git SHA is written to prediction evidence.

Limitations: this is self-verification by the running process. It detects
an unexpected commit, modified/extra/missing source, a mismatched installed
package and an unverifiable identity. It does not protect against someone
able to modify the interpreter, this verifier, the attestation pin in the
reviewed configuration, or third-party dependencies. The environment must
also prevent in-memory module changes and artifact replacement between
loading and verification; these are point-in-time checks, not an atomic
secure loader. Host access control and the frozen environment remain
responsible for those. Git
`safe.directory`/ownership refusals fail closed.

## Failure classification and diagnostics

The one-boundary result distinguishes `not_due`, `predicted`,
`predicted_clock_alert`,
`prediction_committed_checkpoint_recovery_required`, `waiting_for_source_fence`,
`missing_source_data`, `invalid_source_health`, `checkpoint_failure`,
`clock_sync_failure`, `model_verification_failure`,
`implementation_provenance_failure`,
`inference_or_persistence_failure`, `ledger_or_source_io_failure`,
`duplicate_prediction`, `late_prediction`,
and `missed_deadline`. Existing ledger event types are reused; the ledger
remains append-only.

Runner metrics include source-fence wait/availability, tail read and
verification, checkpoint load, checkpoint restore, bounded tail replay,
feature generation, candidate/control inference, SQLite transaction,
post-commit attestation, checkpoint publication, total completion, deadline
margin, independent UTC checks before persistence and at the durability
decision, startup model/source validation, CPU time, resident memory where the
OS exposes it, and unpredicted boundary backlog. Checkpoint-load timing is the
restart-recovery measurement. The offline replay benchmark reports median, p95, p99, and
maximum stage latency and traced Python allocation peak. Its synthetic
inference and SQLite sink are clearly separate from the verified official
models and production durability; neither is evidence of T+60 compliance.
The official CLI emits structured JSON-line operational summaries. Those
summaries omit candidate/control probabilities, predicted classes, and raw
feature values so monitoring does not disclose comparative results before the
registered analysis.

### UTC eligibility versus subsequent alerts

| Failure detected | Durable evidence and eligibility |
| --- | --- |
| Before prediction write | No prediction is written. |
| During prediction write, or before/while acquiring the attestation transaction | The fresh independent check must pass after the prediction COMMIT and attestation lock acquisition. Failure retains any committed prediction but rolls back the confirmation; it is unconfirmed and ineligible. |
| Attestation timestamp after T+60 | Late confirmation is retained but ineligible. |
| Crash after prediction COMMIT, before confirmation | Prediction remains unconfirmed and ineligible across restart; never retrospectively re-attested. |
| Clock failure observed after successful attestation | A subsequent operational alert (`predicted_clock_alert`), not a rewrite of the earlier durability decision. The prediction and its eligibility remain immutable, and checkpoint advancement is still attempted. |

The gate samples independent UTC health at the durability decision; it is not
a claim of continuous synchronization before or after that sample. A failure
to record a subsequent alert is also surfaced in `failure_record_error`; it
does not erase an already committed prediction. The inference API's optional
`pre_attestation_check` is forwarded into the ledger; omitted callbacks retain
the existing public flow. `post_persist_check` remains available to existing
callers but is not used as the runner's eligibility gate.

### Deterministic integration coverage and measurements

`tests/test_prospective_runner_integration.py` feeds local raw-frame fixtures
through the real source receiver and certified read-only journal, prepares
and recovers chunked checkpoints, and executes the real runner, causal replay,
official-loader verification gate and immutable SQLite ledger across
consecutive boundaries. Only test clocks and disk-loaded model artifacts are
substituted; the test models are deterministically fitted ExtraTrees. The
production loader and CLI gain no test bypass.

Coverage includes restart offsets -1/0/1/10/30/59/60/61 seconds, complete and
unconfirmed commits, post-commit checkpoint failure, transient real SQLite
read contention, three-attempt retry limits, exact fences, clock/chrony
failure and no-write diagnostic/preparation modes. Full replay and incremental
feature vectors, ordering, fingerprints and candidate/control/always-DOWN
outputs must agree exactly. The registered 2,880-boundary schedule is checked.

The Python 3.10/3.11/3.12 Actions jobs preserve JUnit artifacts containing
warm-up readiness/latency, checkpoint reload latency, two-boundary stage
timings and traced Python peak allocation. These are small **synthetic**
fixtures, not production load or T+60 acceptance measurements. Ubuntu uses
real POSIX file/directory fsync; Windows test publication is explicitly
simulated because the existing publisher does not support that platform.

## Activation and VM acceptance (not performed)

Before any official execution, separately review and authorize the protocol,
implementation commit, frozen manifest/models, implementation SHA, paths,
session binding, and operational authorization. On an isolated VM, with no
production collector/materializer impact:

1. Validate official artifact SHA256s and dependency/Python versions against
   the freeze manifest.
2. Establish implementation provenance on the VM. For wheel execution, build
   the wheel from a fresh clean checkout of the reviewed commit, run
   `attest-build`, review and pin `implementation_attestation_path`/
   `implementation_attestation_sha256`, install exactly that wheel
   (not editable), and confirm `verify-implementation` reports the reviewed
   SHA in `installed_artifact` mode. For checkout execution, confirm a clean
   checkout with runtime data outside it reports `git_checkout` mode. Also
   confirm that a modified installed file, a wrong pin, and a wrong expected
   SHA are each refused.
3. Verify bridge-only operation, exact certified fences, canonical
   eligibility, independent chrony freshness, and source receiver continuity
   across reconnects and interruptions.
4. Run synthetic and pre-window read-only diagnostics; prove post-T frames
   cannot affect features and target/label data is absent from inference.
5. Exercise every crash boundary: warm-up continuation, checkpoint link/fsync,
   prediction transaction, attestation, restart reconciliation, and
   checkpoint-advance failure after a successful prediction.
6. Run representative official-model inference and storage durability
   benchmarks on the target VM. Record stage median/p95/p99/max, RSS, backlog,
   clock evidence, disk/filesystem semantics, and measured T+60 margin. Repeat
   under realistic source load and collection interruption. Synthetic
   timings do not qualify.
7. Confirm the collector and canonical materializer remain unchanged, test
   graceful shutdown/restart, and verify no duplicate or retroactively timely
   predictions can be written.
8. Approve deployment separately and create any service configuration only
   through the normal production change process. This PR creates no enabled
   service, timer, cron entry, or collector hook.

This implementation has not received independent VM or production acceptance.
The source bridge's host/socket/disk properties, production chrony, genuine
frozen model load/inference latency, filesystem durability, the reviewed
implementation attestation (for wheel execution) and deployment
authorization remain prerequisites. In particular, the existing immutable
checkpoint publisher currently requires POSIX directory `fsync`; Windows
local development cannot attest checkpoint publication and must fail closed.
