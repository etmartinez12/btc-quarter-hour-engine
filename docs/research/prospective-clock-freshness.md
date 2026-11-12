# Prospective UTC clock-health freshness policy (`chrony-freshness-v1`)

The source bridge, the source journal, and the inference runner certify UTC
through one shared implementation:
`btc_quarter_hour_engine/research/prospective_clock.py`. No other module has
its own clock age or offset limit.

## Why this policy exists

During isolated Ubuntu VM acceptance, application clock certification failed
even though chrony reported a microsecond-scale offset, a synchronized Google
metadata NTP source, and `Normal` leap status. There were two defects:

1. The parser required the label `Reference time (UTC)`. Real `chronyc tracking`
   output uses `Ref time (UTC)`. The unit fixture used the wrong label, so the
   defect was not caught.
2. A fixed 120-second maximum reference age conflicted with chrony's adaptive
   polling. On the VM, chrony updated about every 1,033 seconds. The bridge
   and the runner each enforced their own copy of the 120-second limit.

## Evidence collection (read-only)

```
chronyc -n tracking
chronyc -n -c sources
```

Both commands run with a 2-second timeout. Nothing changes the system clock,
chrony configuration, the production collector, or the canonical materializer.
The reader brackets both commands with wall-clock and monotonic-clock samples.
If wall time and monotonic time diverge by more than 50 ms during the read,
the clock jumped and the evidence fails. A read that takes longer than 2
seconds also fails.

### Parsing

* `Ref time (UTC)` (chrony 4.x) and `Reference time (UTC)` are both accepted.
  A duplicate or conflicting field is rejected.
* Reference times use chrony's `%a %b %d %H:%M:%S %Y` layout. Single-digit,
  zero-padded, and space-padded days are all accepted. Existing
  timezone-aware ISO 8601 values are also still accepted.
* `System time` must state `fast` or `slow`. `slow` maps to a negative
  offset. A value with no direction is malformed.
* Every required field must be present exactly once and must be finite and
  well-formed. A missing, duplicate, malformed, or non-finite field makes the
  evidence unverified. It is never partially accepted.
* The CSV sources output must contain exactly 10 fields per row, with valid
  mode, state, address, stratum, poll, octal reach, and `last_rx` values. A
  `last_rx` of `4294967295` means chrony has no sample.

## Certification conditions

The evidence is certified only when every condition below holds. Each
condition is recomputed from the stored fields, and the stored `verified` and
`synchronized` flags are never trusted.

| Signal | Requirement |
| --- | --- |
| Leap status | Exactly `Normal` |
| Tracking reference | Valid stratum 1-15. Reference ID is not `00000000` (unsynchronized) and not `7F7F01xx` (local fallback) |
| System offset | `abs(offset) <= 50 ms`, the unchanged protocol threshold |
| Estimated error | `root_delay / 2 + root_dispersion <= 50 ms` |
| Selected source | Exactly one `*` row. It is an NTP server (`^`) or peer (`=`) whose address matches the tracking reference ID, with stratum 1-15 |
| Reachability | Reach register is nonzero, and the most recent poll was answered (`reach & 1`) |
| Poll | Selected-source poll `<= 2^10 = 1,024 s` |
| Last sample | `0 <= last_rx <= horizon` |
| Reference age | `0 <= observed - reference <= horizon`. A negative age is a future timestamp |
| Decision-time age | When a decision time is supplied, the age at the decision time must also be `<= horizon` |
| Observation | The read took `<= 2 s`, and wall time diverged from monotonic time by `<= 50 ms` |

The reference-age horizon is set by the poll interval of the selected source:

```
horizon = min(1,080 s, 2**poll * 1.02 + 16 s)
```

The `1.02` factor covers chrony's transmit randomization. The 16 seconds of
grace covers response and processing time.

| Poll | Interval | Horizon |
| --- | --- | --- |
| 6 | 64 s | 81.28 s |
| 8 | 256 s | 277.12 s |
| 10 | 1,024 s | 1,060.48 s |

The absolute ceiling of 1,080 seconds is a hard upper limit on staleness. A
healthy offset or `Normal` leap status alone never certifies older evidence.
The policy validates that the ceiling covers one maximum polling cycle and
stays below two polling intervals. chrony's update interval is recorded for
audit and must be finite and nonnegative, but it never widens the horizon.

## Where the policy is enforced

* **Source fence.** At each fence, the receiver reads chrony fresh, with no
  caching. The journal stores the full evidence. It accepts the evidence only
  if the observation time falls between the fence witness's ingest time and
  T+60. It then re-evaluates the evidence under this policy and requires the
  recorded flags to match exactly. Forged `verified` evidence is rejected, and
  no fence is written. Uncertified evidence is recorded with
  `clock_health_verified = 0`, and `build_causal_feature_vector()` refuses that
  snapshot with `source_clock_health_unverified`.
* **Runner.** Two kinds of check apply:
  * *Forced checks* always take a fresh observation, which must be at most
    5 seconds old at the decision. They run at startup validation, before the
    prediction write, and in the attestation transaction (after its write lock
    is acquired and before the attestation timestamp). They also run after
    attestation.
  * *Unforced checks*, made while waiting for the source and before inference,
    may reuse an observation that is at most 10 seconds old.

  Every check runs the evaluator with the decision time. Any failure raises
  `independent_utc_clock_evidence_unavailable_or_stale: <reasons>`, including
  reader failures such as `chronyc -n tracking unavailable: FileNotFoundError`.

Clock evidence is metadata about the host clock. It is never a model input.
Changing clock evidence or its age changes no feature, fingerprint, or
candidate, control, or always-DOWN output. The integration test
`test_clock_freshness_evidence_does_not_change_features_or_predictions`
asserts this.

## Failures that remove eligibility versus operational alerts

| When the clock check fails | Effect |
| --- | --- |
| At the source fence | The fence is recorded as uncertified. No prediction is made for that boundary (`clock_sync_failure`). |
| Before the prediction write | No prediction row is written (`clock_sync_failure`). |
| During the write, or at the attestation decision | The committed prediction is kept but never attested, so it is not eligible for evaluation (`clock_sync_failure`). It is never re-attested retrospectively. |
| After a successful attestation | Operational alert only (`predicted_clock_alert`, `subsequent_clock_alert`). The earlier certified decision stays immutable. |

All of these fail closed: chrony is unavailable, it times out, or it exits
nonzero; the output is malformed or conflicting; the source is
unsynchronized, a local fallback, unreachable, or missed its last poll; it has
no samples; the offset or estimated error is too large; the reference or last
sample is stale; the reference is in the future; the observation is slow; or
the clock jumps during the read.

## Operational risk: lost polls at `maxpoll 10`

At poll 10, if one NTP response is lost, the reference age reaches about
2,090 seconds before the next update. Under this policy, the reach bit or the
1,060.48-second horizon then fails the evidence. Certification fails closed
for up to one polling interval, about 17 minutes, so those boundaries are
correctly reported as `clock_sync_failure`. Lowering `maxpoll` for the
selected source would reduce this exposure. That is a chrony configuration
change that must be separately authorized on the VM. This PR does not make it.

## Decisions to review

* A separate 50 ms limit on estimated error (`root_delay / 2 +
  root_dispersion`). Combining it with the offset into one error bound would
  require a protocol amendment.
* The most recent poll must be answered (`reach & 1`). Reference clocks (`#`)
  and local fallback references are rejected.
* The horizon depends only on the poll, with an absolute ceiling of 1,080
  seconds.
* An IPv6 source matches its reference ID through chrony's numeric
  (`-n`) name.
* A `System time` value with no direction is malformed.
* Legacy evidence (the 8 original fields) cannot be verified, and legacy
  `verified = true` evidence is rejected as contradictory. No production
  journals exist yet.

## Test fixture caveat

`tests/chrony_fixtures.py::UBUNTU_GCE_TRACKING` reconstructs the VM output from
chrony 4.x format strings. It uses the reported values: the Google metadata
source, `Ref time (UTC)`, an update interval of about 1,033 seconds, a
microsecond offset, and `Normal` leap status. It is not a verbatim capture.
Record the literal `chronyc -n tracking` and `chronyc -n -c sources` output on
the VM, add it as a fixture, and rerun VM acceptance before activation.
