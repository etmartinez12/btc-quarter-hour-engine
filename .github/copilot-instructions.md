# Bitcoin Quarter-Hour Engine: Copilot Development Contract

## Operating environment

This repository is developed on the same Google Cloud VM that hosts production BTC data collection and research services.

Development and production are intentionally isolated.

Work only inside the current Copilot development session/worktree.

Never access, inspect, modify, delete, overwrite, or write to:

- /home/martinez_elias12/btc-quarter-hour-engine
- /home/martinez_elias12/.venv
- /var/lib/btc-qh
- /etc/systemd/system
- /home/martinez_elias12/.ssh
- /home/martinez_elias12/.config/gh
- /home/martinez_elias12/.config/chrome-remote-desktop

Never use sudo.

Never stop, restart, reload, enable, disable, or modify systemd services.

Never alter VM-level configuration, firewall configuration, AppArmor policy, mounts, swap, or operating-system packages.

Do not attempt to bypass the Copilot sandbox.

Assume agent shell execution has no outbound network or local-network access.

If work genuinely requires production data, external network access, credentials, package installation, service operations, or a path outside the sandbox, stop and explain exactly what is required.

## Development Python environment

Use the existing shared development Python environment:

/home/martinez_elias12/btc-quarter-hour-engine-dev/.venv

Do not create another virtual environment inside a Copilot worktree.

Do not install or upgrade Python packages unless explicitly instructed.

The current Copilot worktree must remain the source of code under test.

## Testing workflow

During implementation, run the smallest relevant tests first.

Use:

./scripts/copilot-test <pytest target>

Examples:

./scripts/copilot-test tests/test_prospective_replay.py -q
./scripts/copilot-test tests/test_prospective_inference.py -q

Do not repeatedly run the entire suite after every small edit.

Before declaring a change ready for human review, run:

./scripts/copilot-verify

Do not hard-code or assume an expected number of passing tests. Report the actual pass, skip, warning, and failure counts from the current repository.

Large synthetic benchmarks, long-history replay benchmarks, production-like load tests, and other high-resource workloads require explicit user approval before execution.

## Git workflow

Work only on the current Copilot session branch/worktree.

Never modify main directly.

Never force-push.

Do not push branches, open pull requests, merge pull requests, delete remote branches, or deploy unless explicitly instructed.

Authenticated Git and GitHub CLI access are intentionally unavailable to normal agent shell execution.

Leave completed changes locally for review.

Before modifying code, inspect:

git status --short
git branch --show-current

After modifying code:

1. Run relevant targeted tests.
2. Inspect the final diff.
3. Run ./scripts/copilot-verify before calling the implementation ready for review.
4. Report files changed, tests run, verification results, and unresolved risks.

## Core scientific contract

This project predicts whether the exact canonical BTC-USD boundary price at T+15 minutes is greater than the exact canonical price at T.

Canonical boundaries occur at exact UTC quarter hours.

The canonical price is derived from the synchronized Coinbase BTC-USD best-bid/best-ask midpoint according to the repository's eligibility rules.

Causality is a correctness requirement.

Features for boundary T may use only information legitimately available at or before T.

Events, labels, observations, or state that become available after T must never influence the feature vector or prediction for T.

Preserve:

- strict temporal ordering
- label-availability embargoes
- exact quarter-hour alignment
- source provenance
- reconnect isolation
- sequence integrity
- boundary eligibility rules
- fail-closed behavior
- immutable evidence semantics

Never weaken validation, causality, integrity checks, provenance checks, error handling, or data-quality requirements merely to make tests pass.

Prefer explicit failure over silently generating incomplete, ambiguous, or scientifically invalid output.

## Production isolation

The live collector is a production-critical workload.

The production research materializer is a separate lower-priority workload.

Development code must not alter either service or their operational state.

Do not read production data directly from /var/lib/btc-qh during ordinary development.

Use repository fixtures, synthetic fixtures, or an explicitly staged immutable research dataset approved for the task.

Do not infer that access to a path means permission to use it.

## Prospective V2 experiment

The repository contains a predeclared V2 prospective validation protocol.

Treat the committed protocol and prospective freeze documentation as authoritative.

The scheduled prospective evaluation is:

Start: 2026-11-15T00:00:00Z
End exclusive: 2026-12-15T00:00:00Z

The implementation/model freeze deadline is:

2026-11-14T23:00:00Z

The fixed candidate is the development-selected range/volatility ExtraTrees model using the approved 12-predictor V2 family.

The fixed control is ExtraTrees using the approved full 91-predictor V2 contract.

The decision threshold is fixed at 0.5.

The frozen training population, model configuration, predictor ordering, artifact hashes, feature contracts, evaluation dates, coverage requirements, prediction deadline, eligibility rules, and scoring rules must not be altered casually.

Do not modify the prospective protocol, candidate selection, comparator, training population, predictor sets, threshold, evaluation window, coverage criteria, model artifacts, model hashes, feature contract, prediction timing, ledger semantics, or eligibility rules unless the user explicitly requests a protocol-level change.

A change to those elements is not ordinary implementation work and must be treated as a prospective-protocol change.

Never reconstruct a missing historical prediction and represent it as timely prospective evidence.

Never overwrite immutable prediction evidence after the corresponding outcome becomes available.

Prospective prediction generation, label availability, and outcome scoring must remain logically separated.

## Prospective source bridge

The prospective source bridge and bounded causal replay path are designed to fail closed.

Do not equate a producer heartbeat, wall-clock time, sealed-segment existence, or later recovery with proof that a boundary was complete and healthy at T.

Preserve frame continuity, source watermark semantics, clock-health evidence, checkpoint integrity, reconnect epochs, durable acknowledgment semantics, and canonical as-of-boundary evidence.

The live-source bridge remains subject to separate VM validation and explicit activation authorization.

Do not enable it on the production collector merely because implementation tests pass.

Synthetic benchmarks do not establish production T+60 compliance.

## Research scope

V3 microstructure feature development remains in scope.

The original multi-model and ensemble objectives remain in scope.

Those research tracks must remain separate from the frozen V2 prospective experiment.

Future research must not mutate or contaminate the V2 prospective models, evaluation data, protocol, feature contracts, or scoring rules.

## Performance work

This VM has limited memory and simultaneously hosts production workloads.

Avoid unnecessary full-suite repetitions and high-resource benchmarks.

Use targeted tests while iterating.

For performance claims, distinguish:

- synthetic benchmark evidence
- development VM measurements
- production contention measurements
- actual prospective timing evidence

Do not describe synthetic or local benchmark performance as demonstrated production deadline compliance.
