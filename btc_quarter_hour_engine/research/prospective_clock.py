"""Shared chrony clock-health evidence and the bounded freshness policy.

The source bridge (source-fence certification), the source journal (stored
fence validation), and the inference runner (prediction persistence and
durability attestation) all certify UTC through this module, so one validated
policy governs every evaluation-eligibility decision.

Evidence is read-only: ``chronyc -n tracking`` and ``chronyc -n -c sources``
are executed, and nothing changes host time or chrony configuration.

Freshness policy (``chrony-freshness-v1``)
-----------------------------------------
chrony polls its selected source adaptively (``2**poll`` seconds, at most
``maxpoll``; the Ubuntu/GCE default reaches 1,024 s). The reference timestamp
therefore legitimately ages up to one polling interval, plus chrony's
transmission randomisation (up to +2%), between updates. Evidence is
certified only when every condition holds:

* tracking leap status is exactly ``Normal`` with a valid stratum (1-15) and a
  reference ID that is neither unsynchronised (``00000000``) nor a local
  fallback (``7F7F01xx``);
* ``|system time offset| <= 50 ms`` (unchanged protocol threshold);
* estimated error ``root_delay / 2 + root_dispersion <= 50 ms``;
* exactly one selected (``*``) NTP server or peer source whose address matches
  the tracking reference ID, with stratum 1-15, poll ``<= 1,024 s``, reach
  register non-zero with the most recent poll answered (``reach & 1``), and a
  last sample no older than the reference-age horizon;
* ``0 <= reference age <= min(1,080 s, 2**poll * 1.02 + 16 s)`` measured at the
  observation and, when supplied, at the decision time (a negative age is a
  future timestamp and fails closed);
* the observation took at most 2 s and wall time advanced within 50 ms of
  monotonic time while it was taken (no clock jump during the read).

The update interval is recorded for audit and must be finite and nonnegative
but never extends the horizon, so a past gap cannot loosen the current cycle.
The absolute 1,080 s ceiling bounds staleness regardless of poll, offset, or
leap status.
"""
from __future__ import annotations

import ipaddress
import math
import re
import subprocess
import time
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

CHRONY_POLICY_VERSION = "chrony-freshness-v1"
CHRONY_TRACKING_COMMAND = ("chronyc", "-n", "tracking")
CHRONY_SOURCES_COMMAND = ("chronyc", "-n", "-c", "sources")
CHRONY_EVIDENCE_SOURCE = "chronyc -n tracking; chronyc -n -c sources"
CHRONY_NO_SAMPLE_LAST_RX = 0xFFFFFFFF
_MIN_POLL_LOG2 = -7
_DERIVED_TOLERANCE = 1e-9


class ChronyEvidenceError(ValueError):
    """Raised when chrony output is malformed, ambiguous, or conflicting."""


@dataclass(frozen=True, slots=True)
class ChronyFreshnessPolicy:
    version: str = CHRONY_POLICY_VERSION
    max_offset_seconds: float = 0.050
    max_estimated_error_seconds: float = 0.050
    max_poll_interval_seconds: float = 1024.0
    poll_randomness: float = 0.02
    reference_age_grace_seconds: float = 16.0
    max_reference_age_seconds: float = 1080.0
    max_observation_duration_seconds: float = 2.0
    max_clock_jump_seconds: float = 0.050
    max_decision_observation_age_seconds: float = 5.0
    max_cached_observation_age_seconds: float = 10.0

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("Clock policy version must be nonempty")
        values = [
            getattr(self, item.name) for item in fields(self) if item.name != "version"
        ]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in values
        ):
            raise ValueError("Clock policy tolerances must be finite and nonnegative")
        if (
            self.max_offset_seconds <= 0
            or self.max_estimated_error_seconds <= 0
            or self.max_observation_duration_seconds <= 0
            or self.max_clock_jump_seconds <= 0
            or self.max_decision_observation_age_seconds <= 0
            or self.max_cached_observation_age_seconds
            < self.max_decision_observation_age_seconds
        ):
            raise ValueError("Clock policy tolerances must be positive and ordered")
        if (
            self.max_poll_interval_seconds < 1
            or not math.log2(self.max_poll_interval_seconds).is_integer()
        ):
            raise ValueError("Maximum poll interval must be a power of two seconds")
        cycle = (
            self.max_poll_interval_seconds * (1 + self.poll_randomness)
            + self.reference_age_grace_seconds
        )
        if not cycle <= self.max_reference_age_seconds <= 2 * self.max_poll_interval_seconds:
            raise ValueError(
                "Reference-age ceiling must cover one maximum polling cycle "
                "and stay below two polling intervals"
            )

    @property
    def max_poll_log2(self) -> int:
        return int(math.log2(self.max_poll_interval_seconds))

    def reference_age_horizon(self, poll_log2: int) -> float:
        """Permitted reference age for the selected source's current poll."""
        if type(poll_log2) is not int or not _MIN_POLL_LOG2 <= poll_log2 <= self.max_poll_log2:
            raise ValueError("Selected-source poll is outside the permitted policy")
        return min(
            self.max_reference_age_seconds,
            (2.0**poll_log2) * (1 + self.poll_randomness) + self.reference_age_grace_seconds,
        )


CHRONY_CLOCK_POLICY = ChronyFreshnessPolicy()


def _utc(value: datetime, *, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return _utc(value, field="timestamp").isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _parse_iso(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} is malformed") from exc
    return _utc(parsed, field=field)


_STR_FIELDS = frozenset(
    {
        "leap_status",
        "reason",
        "policy_version",
        "reference_id",
        "reference_name",
        "selected_source_mode",
        "selected_source_state",
        "selected_source_address",
    }
)
_FLOAT_FIELDS = frozenset(
    {
        "offset_seconds",
        "update_interval_seconds",
        "root_delay_seconds",
        "root_dispersion_seconds",
        "estimated_error_seconds",
        "maximum_error_bound_seconds",
        "reference_age_seconds",
        "reference_age_horizon_seconds",
        "observation_duration_seconds",
        "wall_monotonic_divergence_seconds",
    }
)
_INT_FIELDS = frozenset(
    {
        "stratum",
        "selected_source_stratum",
        "selected_source_poll_log2",
        "selected_source_reach",
        "selected_source_last_rx_seconds",
    }
)
_LEGACY_REQUIRED = (
    "observed_at_utc",
    "synchronized",
    "verified",
    "offset_seconds",
    "reference_time_utc",
    "leap_status",
    "source",
    "reason",
)


@dataclass(frozen=True, slots=True)
class ClockHealthEvidence:
    observed_at_utc: datetime
    synchronized: bool
    verified: bool
    offset_seconds: float | None
    reference_time_utc: datetime | None
    leap_status: str | None
    source: str
    reason: str | None = None
    policy_version: str | None = None
    reference_id: str | None = None
    reference_name: str | None = None
    stratum: int | None = None
    update_interval_seconds: float | None = None
    root_delay_seconds: float | None = None
    root_dispersion_seconds: float | None = None
    estimated_error_seconds: float | None = None
    maximum_error_bound_seconds: float | None = None
    reference_age_seconds: float | None = None
    reference_age_horizon_seconds: float | None = None
    selected_source_mode: str | None = None
    selected_source_state: str | None = None
    selected_source_address: str | None = None
    selected_source_stratum: int | None = None
    selected_source_poll_log2: int | None = None
    selected_source_reach: int | None = None
    selected_source_last_rx_seconds: int | None = None
    observation_duration_seconds: float | None = None
    wall_monotonic_divergence_seconds: float | None = None

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["observed_at_utc"] = _iso(self.observed_at_utc)
        value["reference_time_utc"] = (
            _iso(self.reference_time_utc) if self.reference_time_utc is not None else None
        )
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ClockHealthEvidence":
        """Strictly decode stored evidence; unknown keys or wrong types raise."""
        if not isinstance(value, Mapping):
            raise ValueError("Clock evidence must be a mapping")
        names = {item.name for item in fields(cls)}
        unknown = set(value) - names
        if unknown:
            raise ValueError(f"Clock evidence has unknown fields: {sorted(unknown)}")
        missing = [name for name in _LEGACY_REQUIRED if name not in value]
        if missing:
            raise ValueError(f"Clock evidence is missing fields: {missing}")
        decoded: dict[str, Any] = {}
        for name in names:
            raw = value.get(name)
            if name == "observed_at_utc":
                decoded[name] = _parse_iso(raw, field=name)
            elif name == "reference_time_utc":
                decoded[name] = None if raw is None else _parse_iso(raw, field=name)
            elif name in {"synchronized", "verified"}:
                if type(raw) is not bool:
                    raise TypeError(f"Clock evidence {name} must be boolean")
                decoded[name] = raw
            elif name == "source":
                if not isinstance(raw, str) or not raw:
                    raise TypeError("Clock evidence source must be a nonempty string")
                decoded[name] = raw
            elif name in _STR_FIELDS:
                if raw is not None and not isinstance(raw, str):
                    raise TypeError(f"Clock evidence {name} must be a string")
                decoded[name] = raw
            elif name in _FLOAT_FIELDS:
                if raw is not None and (
                    isinstance(raw, bool) or not isinstance(raw, (int, float))
                ):
                    raise TypeError(f"Clock evidence {name} must be numeric")
                decoded[name] = None if raw is None else float(raw)
            elif name in _INT_FIELDS:
                if raw is not None and type(raw) is not int:
                    raise TypeError(f"Clock evidence {name} must be an integer")
                decoded[name] = raw
            else:  # pragma: no cover - guarded by the field tables
                raise ValueError(f"Unhandled clock evidence field {name}")
        return cls(**decoded)


@dataclass(frozen=True, slots=True)
class ClockAssessment:
    verified: bool
    reasons: tuple[str, ...]
    estimated_error_seconds: float | None
    maximum_error_bound_seconds: float | None
    reference_age_seconds: float | None
    reference_age_horizon_seconds: float | None


@dataclass(frozen=True, slots=True)
class _Tracking:
    reference_id: str
    reference_name: str
    stratum: int
    reference_time_utc: datetime
    offset_seconds: float
    root_delay_seconds: float
    root_dispersion_seconds: float
    update_interval_seconds: float
    leap_status: str


@dataclass(frozen=True, slots=True)
class _Source:
    mode: str
    state: str
    address: str
    stratum: int
    poll_log2: int
    reach: int
    last_rx_seconds: int | None


_TRACKING_ALIASES = {
    "reference id": "reference_id",
    "stratum": "stratum",
    "ref time (utc)": "reference_time",
    "reference time (utc)": "reference_time",
    "system time": "system_time",
    "root delay": "root_delay",
    "root dispersion": "root_dispersion",
    "update interval": "update_interval",
    "leap status": "leap_status",
}
_REFERENCE_ID = re.compile(r"([0-9A-Fa-f]{8}) \((.*)\)")
_UNSIGNED = r"(?:\d+(?:\.\d+)?)"
_SYSTEM_TIME = re.compile(rf"({_UNSIGNED}) seconds (fast|slow) of NTP time")
_SECONDS = re.compile(rf"({_UNSIGNED}) seconds")
_INTEGER = re.compile(r"\d+")
_SIGNED_INTEGER = re.compile(r"[+-]?\d+")
_SIGNED_DECIMAL = re.compile(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?")
_DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTHS = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)
# chronyc prints a zero-padded day; the earlier strptime parser also accepted
# asctime-style space padding ("Thu Jan  1"), which remains supported.
_CHRONY_TIME = re.compile(
    rf"({'|'.join(_DAYS)}) +({'|'.join(_MONTHS)}) +(\d{{1,2}}) +"
    r"(\d{2}):(\d{2}):(\d{2}) +(\d{4})"
)
_LEAP_VALUES = frozenset(
    {"Normal", "Insert second", "Delete second", "Not synchronised", "Invalid"}
)


def _finite_number(text: str, *, field: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ChronyEvidenceError(f"{field} is not finite")
    return value


def _parse_reference_time(value: str) -> datetime:
    match = _CHRONY_TIME.fullmatch(value)
    if match is not None:
        day, month, mday, hour, minute, second, year = match.groups()
        try:
            parsed = datetime(
                int(year), _MONTHS.index(month) + 1, int(mday),
                int(hour), int(minute), int(second), tzinfo=timezone.utc,
            )
        except ValueError as exc:
            raise ChronyEvidenceError("Ref time (UTC) is not a valid date") from exc
        if _DAYS[parsed.weekday()] != day:
            raise ChronyEvidenceError("Ref time (UTC) weekday contradicts its date")
        return parsed
    # Legacy compatibility: earlier evidence accepted an aware ISO-8601 value.
    try:
        parsed_iso = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ChronyEvidenceError("Ref time (UTC) is malformed") from exc
    if parsed_iso.tzinfo is None or parsed_iso.utcoffset() is None:
        raise ChronyEvidenceError("Ref time (UTC) lacks a UTC offset")
    return parsed_iso.astimezone(timezone.utc)


def parse_chrony_tracking_fields(output: str) -> _Tracking:
    """Strictly parse ``chronyc tracking`` (both ``Ref time`` spellings)."""
    if not isinstance(output, str):
        raise ChronyEvidenceError("chrony tracking output must be text")
    raw: dict[str, str] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        if ":" not in line:
            raise ChronyEvidenceError("chrony tracking line lacks a field separator")
        key, value = line.split(":", 1)
        normalized = " ".join(key.split()).casefold()
        canonical = _TRACKING_ALIASES.get(normalized)
        if canonical is None:
            continue
        if canonical in raw:
            raise ChronyEvidenceError(f"chrony tracking repeats or conflicts on {normalized}")
        raw[canonical] = value.strip()
    missing = sorted(set(_TRACKING_ALIASES.values()) - set(raw))
    if missing:
        raise ChronyEvidenceError(f"chrony tracking lacks fields: {missing}")

    reference = _REFERENCE_ID.fullmatch(raw["reference_id"])
    if reference is None:
        raise ChronyEvidenceError("Reference ID is malformed")
    if _INTEGER.fullmatch(raw["stratum"]) is None:
        raise ChronyEvidenceError("Stratum is malformed")
    system_time = _SYSTEM_TIME.fullmatch(raw["system_time"])
    if system_time is None:
        raise ChronyEvidenceError("System time lacks an absolute fast/slow offset")
    magnitude = _finite_number(system_time.group(1), field="System time")
    # chrony prints |correction|; "fast" means the clock is ahead of NTP time.
    offset = magnitude if system_time.group(2) == "fast" else -magnitude
    measured: dict[str, float] = {}
    for name, label in (
        ("root_delay", "Root delay"),
        ("root_dispersion", "Root dispersion"),
        ("update_interval", "Update interval"),
    ):
        match = _SECONDS.fullmatch(raw[name])
        if match is None:
            raise ChronyEvidenceError(f"{label} is malformed")
        measured[name] = _finite_number(match.group(1), field=label)
    if raw["leap_status"] not in _LEAP_VALUES:
        raise ChronyEvidenceError("Leap status is not a recognised chrony value")
    return _Tracking(
        reference_id=reference.group(1).upper(),
        reference_name=reference.group(2),
        stratum=int(raw["stratum"]),
        reference_time_utc=_parse_reference_time(raw["reference_time"]),
        offset_seconds=offset,
        root_delay_seconds=measured["root_delay"],
        root_dispersion_seconds=measured["root_dispersion"],
        update_interval_seconds=measured["update_interval"],
        leap_status=raw["leap_status"],
    )


def parse_chrony_sources(output: str) -> tuple[_Source, ...]:
    """Strictly parse ``chronyc -c sources`` CSV (exactly ten fields per row)."""
    if not isinstance(output, str):
        raise ChronyEvidenceError("chrony sources output must be text")
    rows: list[_Source] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.strip().split(",")
        if len(parts) != 10:
            raise ChronyEvidenceError("chrony sources row does not have ten CSV fields")
        mode, state, address, stratum, poll, reach, last_rx, *offsets = parts
        if mode not in {"^", "=", "#"} or state not in {"*", "+", "-", "?", "x", "~"}:
            raise ChronyEvidenceError("chrony sources mode or state is malformed")
        if not address:
            raise ChronyEvidenceError("chrony sources address is empty")
        if _INTEGER.fullmatch(stratum) is None or _SIGNED_INTEGER.fullmatch(poll) is None:
            raise ChronyEvidenceError("chrony sources stratum or poll is malformed")
        if re.fullmatch(r"[0-7]{1,3}", reach) is None or int(reach, 8) > 0o377:
            raise ChronyEvidenceError("chrony sources reach is malformed")
        if _INTEGER.fullmatch(last_rx) is None or int(last_rx) > CHRONY_NO_SAMPLE_LAST_RX:
            raise ChronyEvidenceError("chrony sources LastRx is malformed")
        for offset in offsets:
            if _SIGNED_DECIMAL.fullmatch(offset) is None:
                raise ChronyEvidenceError("chrony sources measurement is malformed")
            _finite_number(offset, field="chrony sources measurement")
        last = int(last_rx)
        rows.append(
            _Source(
                mode=mode,
                state=state,
                address=address,
                stratum=int(stratum),
                poll_log2=int(poll),
                reach=int(reach, 8),
                last_rx_seconds=None if last == CHRONY_NO_SAMPLE_LAST_RX else last,
            )
        )
    return tuple(rows)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _aware(value: Any) -> datetime | None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        return None
    return value.astimezone(timezone.utc)


def _derive(
    evidence: ClockHealthEvidence, policy: ChronyFreshnessPolicy
) -> tuple[float | None, float | None, float | None, float | None]:
    delay = _finite(evidence.root_delay_seconds)
    dispersion = _finite(evidence.root_dispersion_seconds)
    estimated = (
        delay / 2 + dispersion
        if delay is not None and dispersion is not None and delay >= 0 and dispersion >= 0
        else None
    )
    offset = _finite(evidence.offset_seconds)
    bound = abs(offset) + estimated if offset is not None and estimated is not None else None
    observed = _aware(evidence.observed_at_utc)
    reference = _aware(evidence.reference_time_utc)
    age = (observed - reference).total_seconds() if observed and reference else None
    try:
        horizon = policy.reference_age_horizon(evidence.selected_source_poll_log2)  # type: ignore[arg-type]
    except ValueError:
        horizon = None
    return estimated, bound, age, horizon


def _reference_matches_source(evidence: ClockHealthEvidence) -> bool:
    address = evidence.selected_source_address
    if not isinstance(address, str):
        return False
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    if ip.version == 4:
        return f"{int(ip):08X}" == evidence.reference_id
    name = evidence.reference_name
    try:
        return isinstance(name, str) and ipaddress.ip_address(name) == ip
    except ValueError:
        return False


def evaluate_clock_evidence(
    evidence: Any,
    *,
    decision_utc: datetime | None = None,
    policy: ChronyFreshnessPolicy = CHRONY_CLOCK_POLICY,
) -> ClockAssessment:
    """Recompute certification from stored fields; never trusts ``verified``."""
    if not isinstance(evidence, ClockHealthEvidence):
        return ClockAssessment(False, ("clock evidence has an invalid type",), None, None, None, None)
    estimated, bound, age, horizon = _derive(evidence, policy)
    reasons: list[str] = []
    if evidence.policy_version != policy.version:
        reasons.append("clock evidence was not produced under the current freshness policy")
    observed = _aware(evidence.observed_at_utc)
    if observed is None:
        reasons.append("clock observation time is unavailable")
    if evidence.leap_status != "Normal":
        reasons.append("chrony leap status is not Normal")
    reference_id = evidence.reference_id
    if not isinstance(reference_id, str) or re.fullmatch(r"[0-9A-F]{8}", reference_id) is None:
        reasons.append("chrony reference ID is unavailable")
    elif reference_id == "00000000" or reference_id.startswith("7F7F01"):
        reasons.append("chrony is not synchronised to an external reference")
    if type(evidence.stratum) is not int or not 1 <= evidence.stratum <= 15:
        reasons.append("chrony stratum is outside 1-15")
    offset = _finite(evidence.offset_seconds)
    if offset is None:
        reasons.append("chrony system offset is unavailable")
    elif abs(offset) > policy.max_offset_seconds:
        reasons.append("chrony system offset exceeds tolerance")
    if estimated is None:
        reasons.append("chrony root delay or dispersion is unavailable")
    elif estimated > policy.max_estimated_error_seconds:
        reasons.append("chrony estimated error exceeds tolerance")
    interval = _finite(evidence.update_interval_seconds)
    if interval is None or interval < 0:
        reasons.append("chrony update interval is unavailable")

    if evidence.selected_source_mode not in {"^", "="}:
        reasons.append("chrony selected source is not an NTP server or peer")
    if evidence.selected_source_state != "*":
        reasons.append("chrony has no selected synchronization source")
    source_stratum = evidence.selected_source_stratum
    if type(source_stratum) is not int or not 1 <= source_stratum <= 15:
        reasons.append("chrony selected-source stratum is outside 1-15")
    if horizon is None:
        reasons.append("chrony selected-source poll is outside the permitted policy")
    reach = evidence.selected_source_reach
    if type(reach) is not int or not 1 <= reach <= 0o377:
        reasons.append("chrony selected source is unreachable")
    elif not reach & 1:
        reasons.append("chrony selected source missed its most recent poll")
    last_rx = evidence.selected_source_last_rx_seconds
    if type(last_rx) is not int or last_rx < 0:
        reasons.append("chrony selected source has no samples")
    elif horizon is not None and last_rx > horizon:
        reasons.append("chrony selected-source sample is stale")
    if not _reference_matches_source(evidence):
        reasons.append("chrony tracking reference does not match the selected source")

    if age is None:
        reasons.append("chrony reference time is unavailable")
    elif age < 0:
        reasons.append("chrony reference time is in the future")
    elif horizon is not None and age > horizon:
        reasons.append("chrony reference evidence is stale")
    if decision_utc is not None:
        decision = _aware(decision_utc)
        reference = _aware(evidence.reference_time_utc)
        if decision is None or observed is None or reference is None:
            reasons.append("clock decision time is unavailable")
        elif decision < observed:
            reasons.append("clock evidence was observed after the decision time")
        elif horizon is not None and (decision - reference).total_seconds() > horizon:
            reasons.append("chrony reference evidence is stale at the decision time")

    duration = _finite(evidence.observation_duration_seconds)
    if duration is None or not 0 <= duration <= policy.max_observation_duration_seconds:
        reasons.append("chrony observation duration is unavailable or excessive")
    divergence = _finite(evidence.wall_monotonic_divergence_seconds)
    if divergence is None or abs(divergence) > policy.max_clock_jump_seconds:
        reasons.append("wall clock jumped while chrony evidence was observed")

    for stored, derived, label in (
        (evidence.estimated_error_seconds, estimated, "estimated error"),
        (evidence.maximum_error_bound_seconds, bound, "maximum error bound"),
        (evidence.reference_age_seconds, age, "reference age"),
        (evidence.reference_age_horizon_seconds, horizon, "reference-age horizon"),
    ):
        stored_value = _finite(stored)
        if derived is None and stored is None:
            continue
        if (
            derived is None
            or stored_value is None
            or abs(stored_value - derived) > _DERIVED_TOLERANCE
        ):
            reasons.append(f"stored {label} contradicts the recorded chrony evidence")
    return ClockAssessment(
        verified=not reasons,
        reasons=tuple(reasons),
        estimated_error_seconds=estimated,
        maximum_error_bound_seconds=bound,
        reference_age_seconds=age,
        reference_age_horizon_seconds=horizon,
    )


def parse_chrony_tracking(
    output: str,
    *,
    observed_at_utc: datetime,
    sources_output: str | None = None,
    observation_duration_seconds: float | None = None,
    wall_monotonic_divergence_seconds: float | None = None,
    source: str = CHRONY_EVIDENCE_SOURCE,
    policy: ChronyFreshnessPolicy = CHRONY_CLOCK_POLICY,
) -> ClockHealthEvidence:
    """Build policy-evaluated evidence; any malformed input yields unverified evidence."""
    observed = _utc(observed_at_utc, field="clock observation time")
    problems: list[str] = []
    tracking: _Tracking | None = None
    selected: _Source | None = None
    try:
        tracking = parse_chrony_tracking_fields(output)
    except ChronyEvidenceError as exc:
        problems.append(f"chrony tracking output is malformed: {exc}")
    if sources_output is None:
        problems.append("chrony selected-source evidence is unavailable")
    else:
        try:
            rows = parse_chrony_sources(sources_output)
        except ChronyEvidenceError as exc:
            problems.append(f"chrony sources output is malformed: {exc}")
        else:
            chosen = [row for row in rows if row.state == "*"]
            if len(chosen) > 1:
                problems.append("chrony sources output is malformed: multiple selected sources")
            elif chosen:
                selected = chosen[0]
    draft = ClockHealthEvidence(
        observed_at_utc=observed,
        synchronized=False,
        verified=False,
        offset_seconds=tracking.offset_seconds if tracking else None,
        reference_time_utc=tracking.reference_time_utc if tracking else None,
        leap_status=tracking.leap_status if tracking else None,
        source=source,
        policy_version=policy.version,
        reference_id=tracking.reference_id if tracking else None,
        reference_name=tracking.reference_name if tracking else None,
        stratum=tracking.stratum if tracking else None,
        update_interval_seconds=tracking.update_interval_seconds if tracking else None,
        root_delay_seconds=tracking.root_delay_seconds if tracking else None,
        root_dispersion_seconds=tracking.root_dispersion_seconds if tracking else None,
        selected_source_mode=selected.mode if selected else None,
        selected_source_state=selected.state if selected else None,
        selected_source_address=selected.address if selected else None,
        selected_source_stratum=selected.stratum if selected else None,
        selected_source_poll_log2=selected.poll_log2 if selected else None,
        selected_source_reach=selected.reach if selected else None,
        selected_source_last_rx_seconds=selected.last_rx_seconds if selected else None,
        observation_duration_seconds=observation_duration_seconds,
        wall_monotonic_divergence_seconds=wall_monotonic_divergence_seconds,
    )
    estimated, bound, age, horizon = _derive(draft, policy)
    draft = replace(
        draft,
        estimated_error_seconds=estimated,
        maximum_error_bound_seconds=bound,
        reference_age_seconds=age,
        reference_age_horizon_seconds=horizon,
    )
    assessment = evaluate_clock_evidence(draft, policy=policy)
    reasons = list(dict.fromkeys([*problems, *assessment.reasons]))
    verified = assessment.verified and not problems
    return replace(
        draft,
        synchronized=verified,
        verified=verified,
        reason="; ".join(reasons) if reasons else None,
    )


def _system_utc_now() -> datetime:
    return datetime.now(timezone.utc)


def read_chrony_clock_health(
    *,
    now_fn: Callable[[], datetime] = _system_utc_now,
    monotonic_fn: Callable[[], float] = time.monotonic,
    runner: Callable[..., Any] = subprocess.run,
    timeout_seconds: float = 2.0,
    policy: ChronyFreshnessPolicy = CHRONY_CLOCK_POLICY,
) -> ClockHealthEvidence:
    """Freshly read chrony status without changing host time or service settings."""
    wall_start = _utc(now_fn(), field="clock observation time")
    monotonic_start = monotonic_fn()
    outputs: list[str] = []
    failure: str | None = None
    for command in (CHRONY_TRACKING_COMMAND, CHRONY_SOURCES_COMMAND):
        try:
            result = runner(
                list(command),
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            failure = f"{' '.join(command)} unavailable: {type(exc).__name__}"
            break
        if result.returncode != 0:
            failure = f"{' '.join(command)} exited with status {result.returncode}"
            break
        outputs.append(result.stdout)
    monotonic_end = monotonic_fn()
    wall_end = _utc(now_fn(), field="clock observation time")
    duration = monotonic_end - monotonic_start
    divergence = (wall_end - wall_start).total_seconds() - duration
    if failure is not None:
        return ClockHealthEvidence(
            observed_at_utc=wall_end,
            synchronized=False,
            verified=False,
            offset_seconds=None,
            reference_time_utc=None,
            leap_status=None,
            source=CHRONY_EVIDENCE_SOURCE,
            reason=failure,
            policy_version=policy.version,
            observation_duration_seconds=duration,
            wall_monotonic_divergence_seconds=divergence,
        )
    return parse_chrony_tracking(
        outputs[0],
        observed_at_utc=wall_end,
        sources_output=outputs[1],
        observation_duration_seconds=duration,
        wall_monotonic_divergence_seconds=divergence,
        policy=policy,
    )


__all__ = [
    "CHRONY_CLOCK_POLICY",
    "CHRONY_EVIDENCE_SOURCE",
    "CHRONY_NO_SAMPLE_LAST_RX",
    "CHRONY_POLICY_VERSION",
    "CHRONY_SOURCES_COMMAND",
    "CHRONY_TRACKING_COMMAND",
    "ChronyEvidenceError",
    "ChronyFreshnessPolicy",
    "ClockAssessment",
    "ClockHealthEvidence",
    "evaluate_clock_evidence",
    "parse_chrony_sources",
    "parse_chrony_tracking",
    "parse_chrony_tracking_fields",
    "read_chrony_clock_health",
]
