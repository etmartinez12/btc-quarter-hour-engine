"""Deterministic chrony fixtures shared by clock, bridge, replay, and runner tests.

``UBUNTU_GCE_TRACKING`` reproduces the layout of ``chronyc tracking`` from the
upstream chrony 4.x format strings with the values reported by the isolated
Ubuntu/GCE acceptance VM: the Google metadata NTP source, ``Ref time (UTC)``,
a ~1,033 s update interval, a microsecond-scale offset, and ``Normal`` leap
status. It is a reconstruction, not a verbatim capture; replace it with the
literal VM capture when one is recorded.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from btc_quarter_hour_engine.research.prospective_clock import (
    CHRONY_NO_SAMPLE_LAST_RX,
    ClockHealthEvidence,
    evaluate_clock_evidence,
    parse_chrony_tracking,
)

_DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

GCE_METADATA_ADDRESS = "169.254.169.254"
GCE_METADATA_REFERENCE = "A9FEA9FE (metadata.google.internal)"

UBUNTU_GCE_TRACKING = """\
Reference ID    : A9FEA9FE (metadata.google.internal)
Stratum         : 3
Ref time (UTC)  : {ref_time}
System time     : 0.000012345 seconds fast of NTP time
Last offset     : -0.000008612 seconds
RMS offset      : 0.000021840 seconds
Frequency       : 11.204 ppm slow
Residual freq   : -0.000 ppm
Skew            : 0.012 ppm
Root delay      : 0.000391626 seconds
Root dispersion : 0.000387918 seconds
Update interval : 1033.4 seconds
Leap status     : Normal
"""

UNSYNCHRONISED_TRACKING = """\
Reference ID    : 00000000 ()
Stratum         : 0
Ref time (UTC)  : Thu Jan 01 00:00:00 1970
System time     : 0.000000000 seconds fast of NTP time
Last offset     : +0.000000000 seconds
RMS offset      : 0.000000000 seconds
Frequency       : 0.000 ppm slow
Residual freq   : +0.000 ppm
Skew            : 0.000 ppm
Root delay      : 1.000000000 seconds
Root dispersion : 1.000000000 seconds
Update interval : 0.0 seconds
Leap status     : Not synchronised
"""


def format_ref_time(value: datetime) -> str:
    """Locale-independent ``%a %b %d %H:%M:%S %Y`` as printed by chronyc."""
    value = value.astimezone(timezone.utc)
    return (
        f"{_DAYS[value.weekday()]} {_MONTHS[value.month - 1]} {value.day:02d} "
        f"{value.hour:02d}:{value.minute:02d}:{value.second:02d} {value.year:04d}"
    )


def tracking_output(
    reference_time: datetime,
    *,
    system_time: str = "0.000012345 seconds fast",
    root_delay: str = "0.000391626",
    root_dispersion: str = "0.000387918",
    update_interval: str = "1033.4",
    leap_status: str = "Normal",
    reference: str = GCE_METADATA_REFERENCE,
    stratum: int = 3,
    reference_label: str = "Ref time (UTC)  ",
) -> str:
    return (
        UBUNTU_GCE_TRACKING.replace("{ref_time}", format_ref_time(reference_time))
        .replace("0.000012345 seconds fast", system_time)
        .replace("Root delay      : 0.000391626", f"Root delay      : {root_delay}")
        .replace("Root dispersion : 0.000387918", f"Root dispersion : {root_dispersion}")
        .replace("Update interval : 1033.4", f"Update interval : {update_interval}")
        .replace("Leap status     : Normal", f"Leap status     : {leap_status}")
        .replace(GCE_METADATA_REFERENCE, reference)
        .replace("Stratum         : 3", f"Stratum         : {stratum}")
        .replace("Ref time (UTC)  ", reference_label)
    )


def sources_row(
    *,
    mode: str = "^",
    state: str = "*",
    address: str = GCE_METADATA_ADDRESS,
    stratum: int = 2,
    poll: int = 10,
    reach: str = "377",
    last_rx: int = 1,
) -> str:
    return (
        f"{mode},{state},{address},{stratum},{poll},{reach},{last_rx},"
        "-0.000004211,-0.000008612,0.000194834"
    )


def unsynchronised_sources() -> str:
    return sources_row(state="?", stratum=0, poll=6, reach="0", last_rx=CHRONY_NO_SAMPLE_LAST_RX) + "\n"


def chrony_evidence(
    observed: datetime,
    *,
    reference_age: float = 1.0,
    verified: bool = True,
    poll: int = 10,
    last_rx: int | None = None,
    reach: str = "377",
    source: str | None = None,
    observation_duration: float = 0.01,
    divergence: float = 0.0,
    **tracking: object,
) -> ClockHealthEvidence:
    """Build evidence through the real parser and the shared freshness policy."""
    observed = observed.astimezone(timezone.utc)
    if verified:
        reference_time = (observed - timedelta(seconds=reference_age)).replace(microsecond=0)
        tracking_text = tracking_output(reference_time, **tracking)  # type: ignore[arg-type]
        sources_text = sources_row(
            poll=poll,
            reach=reach,
            last_rx=int(reference_age) if last_rx is None else last_rx,
        ) + "\n"
    else:
        tracking_text = UNSYNCHRONISED_TRACKING
        sources_text = unsynchronised_sources()
    evidence = parse_chrony_tracking(
        tracking_text,
        observed_at_utc=observed,
        sources_output=sources_text,
        observation_duration_seconds=observation_duration,
        wall_monotonic_divergence_seconds=divergence,
    )
    if source is not None:
        evidence = replace(evidence, source=source)
    assert evaluate_clock_evidence(evidence).verified is evidence.verified
    return evidence


__all__ = [
    "GCE_METADATA_ADDRESS",
    "GCE_METADATA_REFERENCE",
    "UBUNTU_GCE_TRACKING",
    "UNSYNCHRONISED_TRACKING",
    "chrony_evidence",
    "format_ref_time",
    "sources_row",
    "tracking_output",
    "unsynchronised_sources",
]
