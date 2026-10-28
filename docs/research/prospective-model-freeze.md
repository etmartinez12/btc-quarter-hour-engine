# Offline prospective model freeze

This implements only the model-training and artifact-freeze phase of the
[prospective protocol](v2-prospective-validation-protocol-2026-10-28.md).
It does not activate that protocol or implement live inference, collection,
prediction recording, evaluation, services, or trading.

## Development-only run

Use the frozen October snapshot, not an updated materializer output. The CLI
requires explicit input, a new output directory, and an explicit mode:

```powershell
python -m btc_quarter_hour_engine.research.model_freeze_cli `
  --training-input "C:\research\frozen\research_features.parquet" `
  --output-directory "C:\research\freezes\development-001" `
  --development-dry-run
```

The output parent must already exist and must be outside the implementation
checkout. Even an existing empty destination is rejected. No input-hash or
population-count override is available, including in development mode.

The actual snapshot is currently on the research VM, not the Windows development
machine. After pulling this branch into the **development** checkout, run this
single-line command in the VM shell (not in the production checkout):

```bash
/home/martinez_elias12/.venv/bin/python -m btc_quarter_hour_engine.research.model_freeze_cli --training-input /home/martinez_elias12/btc-qh-v1-diagnostics-input/research_features.parquet --output-directory /home/martinez_elias12/prospective-freeze-development-001 --development-dry-run
```

Run from `/home/martinez_elias12/btc-quarter-hour-engine-dev`, with the intended
branch checked out. Choose a new output name for every attempt. Do not restart,
reconfigure, or interact with the collector or hourly research materializer.
A dry-run really fits and verifies both pipelines, but always records
`development_only_nonofficial`. It is not an official prospective freeze.

## Validation and artifacts

The input must match SHA256
`ae0417ab38de3a0204baf8202d741a541e38f79597c47d3691da177ac32aa2fc`
and contain exactly 1,935 trainable observations. The same hash-checked bytes
are passed to the Parquet reader, and the on-disk hash is checked after fitting
and again before publication.

Validation rejects duplicate, missing, unsorted, or off-boundary timestamps;
inconsistent trainability/eligibility; incomplete or nonfinite predictors;
invalid source counts; nonbinary or single-class targets; inconsistent return
directions; and unavailable labels. Nontrainable source rows are retained as
source evidence but are not promoted into the training population. Invalid
trainable rows are never dropped, sorted into compliance, or repaired.

`prepare_research_frame()` derives the existing `bbo-causal-v2` contract.
`localization_variants()` supplies the ordered 12 range/volatility and 91 full-V2
predictors. Both complete pipelines come from the existing research ExtraTrees
factory: median imputation, 300 estimators, maximum depth 6, seed 42. There is
no search, calibration, threshold tuning, or configuration override.

Each successful destination contains:

| File | Content |
|---|---|
| `range_volatility.pkl` | Fitted candidate pipeline, including imputer state |
| `full_v2.pkl` | Fitted comparator pipeline, including imputer state |
| `manifest.json` | Status, ordered predictors, contract versions, complete estimator/preprocessor configuration, artifact hashes, snapshot hash, implementation/protocol provenance, working-tree state, dependency versions, training population/times, and unchanged prospective schedule |

The CLI prints a receipt containing the manifest SHA256. Preserve that receipt
independently with the evidence; it is not a self-authenticating signature.
The imputer's NaN missing-value parameter is represented in JSON as
`{"nonfinite_float": "NaN"}` rather than a nonstandard JSON NaN.

Before publication, each hash-checked local model is deserialized and its
probabilities and class predictions are compared exactly with the in-memory
pipeline over the entire training population. Probabilities must be finite,
bounded, sum to one, and use class order `[0, 1]`. Repeated inference must agree.
This is serialization verification, not prospective performance evaluation.

Files are written and verified in a temporary sibling directory. A complete
set is published with an atomic **no-replace** directory rename on Windows or
Linux (`renameat2` with `RENAME_NOREPLACE` on Linux). Unsupported systems or
filesystems fail closed; there is no non-atomic fallback. Competing destination
creation is also rejected, including an empty directory. Failed attempts clean
up their staging directory without publishing a partial freeze.

## Trusted verification and loading

Pickle deserialization can execute code. A hash alone is not authentication.
Only load an artifact set whose manifest SHA256 was independently obtained
from a trusted freeze operation, never an arbitrary downloaded manifest/hash
pair. Do not trust a replacement digest supplied alongside changed artifacts.

```python
from btc_quarter_hour_engine.research.model_freeze import load_frozen_models

models, manifest = load_frozen_models(
    trusted_output_directory,
    trusted_manifest_sha256=independently_preserved_manifest_sha256,
    require_official=False,  # Explicit opt-in for development-only inspection.
)
```

The loader checks the manifest hash first, then metadata, exact predictor
ordering, both artifact hashes, and factory configurations **before any model
deserialization**. It deserializes the exact bytes that passed hashing, avoiding
a separate unverified file reopen. The recorded Python/dependency versions must
match the loading environment. After deserialization, it verifies pipeline
types, configuration, and fitted feature order. It defaults to requiring an
official freeze, so development evidence cannot silently be used as official.
`verified_probabilities()` additionally checks matrix order, finiteness, class
ordering, and probability validity for offline verification.

## Official freeze gate (not authorized by this implementation)

Do **not** run an official freeze yet. First obtain independent review and
approval of the protocol and implementation. The future operator must explicitly
invoke `--official` and supply `--approved-implementation-sha` and
`--approved-protocol-sha`, each a full 40-character Git commit SHA.

Supplying these arguments is the operator's attestation of **external approval**;
the tool cannot grant approval or infer it from a branch name, PR, clean checkout,
or a commit's existence. The implementation SHA must equal the running checkout's
HEAD, and its tracked and untracked working tree must be clean. The approved
protocol commit must be an ancestor of HEAD and contain the identical
Git-normalized protocol blob (allowing platform-specific checkout line endings).
The manifest records both that blob ID and the checkout file's SHA256.
These gates are rechecked before publication. Official publication must
complete by `2026-11-14T23:00:00Z`; a missed deadline requires a documented
prospective amendment, not a bypass or backdated timestamp.

Official approval arguments are forbidden in development mode. Editing a
development manifest does not create valid official approval evidence and
invalidates the independently preserved manifest hash. An official freeze
requires a separate invocation, fresh training, and a new destination.

No artifact here authorizes service activation. Live inference, durable
append-only prediction evidence, operational validation, and prospective scoring
remain separate work under the unchanged protocol.
