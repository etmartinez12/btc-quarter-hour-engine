"""Shared chrony parsing and the bounded freshness policy (chrony-freshness-v1)."""
from __future__ import annotations

import inspect
import subprocess
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from btc_quarter_hour_engine.research import prospective_clock as clock_module
from btc_quarter_hour_engine.research.prospective_clock import (
    CHRONY_CLOCK_POLICY,
    CHRONY_NO_SAMPLE_LAST_RX,
    CHRONY_POLICY_VERSION,
    CHRONY_SOURCES_COMMAND,
    CHRONY_TRACKING_COMMAND,
    ChronyEvidenceError,
    ChronyFreshnessPolicy,
    ClockHealthEvidence,
    evaluate_clock_evidence,
    parse_chrony_sources,
    parse_chrony_tracking,
    parse_chrony_tracking_fields,
    read_chrony_clock_health,
)
from btc_quarter_hour_engine.research.prospective_source import _validate_bridge_clock_evidence
from tests.chrony_fixtures import (
    UNSYNCHRONISED_TRACKING,
    chrony_evidence,
    sources_row,
    tracking_output,
    unsynchronised_sources,
)

UTC = timezone.utc
REFERENCE = datetime(2026, 5, 12, 14, 15, 1, tzinfo=UTC)
OBSERVED = REFERENCE + timedelta(seconds=1)
TIMING = {"observation_duration_seconds": 0.01, "wall_monotonic_divergence_seconds": 0.0}

# Layout of ``chronyc -n tracking`` / ``chronyc -n -c sources`` on the Ubuntu/GCE
# acceptance VM (Google metadata NTP source, adaptive 1,024 s polling).
UBUNTU_TRACKING = """\
Reference ID    : A9FEA9FE (metadata.google.internal)
Stratum         : 3
Ref time (UTC)  : Tue May 12 14:15:01 2026
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
UBUNTU_SOURCES = "^,*,169.254.169.254,2,10,377,1,-0.000004211,-0.000008612,0.000194834\n"


def _parse(tracking=UBUNTU_TRACKING, sources=UBUNTU_SOURCES, *, observed=OBSERVED, **timing):
    return parse_chrony_tracking(
        tracking, observed_at_utc=observed, sources_output=sources, **{**TIMING, **timing}
    )


def _at_age(age_seconds, *, poll=10, last_rx=None, **tracking):
    return parse_chrony_tracking(
        tracking_output(REFERENCE, **tracking),
        observed_at_utc=REFERENCE + timedelta(seconds=age_seconds),
        sources_output=sources_row(
            poll=poll, last_rx=int(min(age_seconds, 60)) if last_rx is None else last_rx
        ),
        **TIMING,
    )


def test_real_ubuntu_chronyc_tracking_output_is_certified():
    assert REFERENCE.strftime("%a") == "Tue"
    evidence = _parse()
    assert evidence.verified is True and evidence.synchronized is True
    assert evidence.reason is None
    assert evidence.policy_version == CHRONY_POLICY_VERSION
    assert evidence.reference_time_utc == REFERENCE
    assert evidence.reference_id == "A9FEA9FE"
    assert evidence.reference_name == "metadata.google.internal"
    assert evidence.stratum == 3
    assert evidence.leap_status == "Normal"
    assert evidence.offset_seconds == pytest.approx(0.000012345, abs=0)
    assert evidence.update_interval_seconds == 1033.4
    assert evidence.estimated_error_seconds == pytest.approx(0.000391626 / 2 + 0.000387918)
    assert evidence.maximum_error_bound_seconds == pytest.approx(
        0.000012345 + 0.000391626 / 2 + 0.000387918
    )
    assert evidence.reference_age_seconds == 1.0
    assert evidence.reference_age_horizon_seconds == pytest.approx(1024 * 1.02 + 16)
    assert (
        evidence.selected_source_mode,
        evidence.selected_source_state,
        evidence.selected_source_address,
        evidence.selected_source_stratum,
        evidence.selected_source_poll_log2,
        evidence.selected_source_reach,
        evidence.selected_source_last_rx_seconds,
    ) == ("^", "*", "169.254.169.254", 2, 10, 0o377, 1)
    assert evaluate_clock_evidence(evidence).verified


def test_ref_time_label_is_the_previously_hidden_defect():
    # The former parser only read "Reference time (UTC)" and so never found
    # the reference timestamp in genuine chronyc output.
    fields = parse_chrony_tracking_fields(UBUNTU_TRACKING)
    assert fields.reference_time_utc == REFERENCE
    assert "Reference time (UTC)" not in UBUNTU_TRACKING


def test_slow_system_time_is_a_negative_signed_offset():
    evidence = _parse(UBUNTU_TRACKING.replace("seconds fast of", "seconds slow of"))
    assert evidence.verified
    assert evidence.offset_seconds == pytest.approx(-0.000012345, abs=0)


@pytest.mark.parametrize(
    "label, value",
    [
        ("Reference time (UTC)", "Tue May 12 14:15:01 2026"),
        ("Ref time (UTC)", "Tue May 12 14:15:01 2026"),
        ("Ref time (UTC)", "2026-05-12T14:15:01Z"),
        ("Reference time (UTC)", "2026-05-12T16:15:01+02:00"),
    ],
)
def test_previously_supported_reference_formats_remain_compatible(label, value):
    tracking = UBUNTU_TRACKING.replace(
        "Ref time (UTC)  : Tue May 12 14:15:01 2026", f"{label} : {value}"
    )
    evidence = _parse(tracking)
    assert evidence.verified, evidence.reason
    assert evidence.reference_time_utc == REFERENCE


def test_space_padded_day_from_the_legacy_format_is_accepted():
    reference = datetime(2026, 1, 1, 0, 15, 1, tzinfo=UTC)
    tracking = tracking_output(reference).replace("Thu Jan 01", "Thu Jan  1")
    assert "Thu Jan  1" in tracking
    fields = parse_chrony_tracking_fields(tracking)
    assert fields.reference_time_utc == reference


@pytest.mark.parametrize(
    "extra",
    [
        "Reference time (UTC) : Tue May 12 14:15:01 2026\n",
        "Reference time (UTC) : Tue May 12 14:10:01 2026\n",
        "Ref time (UTC)  : Tue May 12 14:15:01 2026\n",
        "Leap status     : Normal\n",
        "System time     : 0.000012345 seconds fast of NTP time\n",
        "reference id    : A9FEA9FE (metadata.google.internal)\n",
    ],
)
def test_duplicate_or_conflicting_fields_fail_closed(extra):
    tracking = UBUNTU_TRACKING + extra
    with pytest.raises(ChronyEvidenceError, match="repeats or conflicts"):
        parse_chrony_tracking_fields(tracking)
    evidence = _parse(tracking)
    assert evidence.verified is False
    assert "malformed" in evidence.reason


@pytest.mark.parametrize(
    "old, new",
    [
        ("A9FEA9FE (metadata.google.internal)", "A9FEA9F (metadata.google.internal)"),
        ("A9FEA9FE (metadata.google.internal)", "A9FEA9FE metadata.google.internal"),
        ("Stratum         : 3", "Stratum         : three"),
        ("Stratum         : 3", "Stratum         : -3"),
        ("0.000012345 seconds fast of NTP time", "0.000012345 seconds"),
        ("0.000012345 seconds fast of NTP time", "-0.000012345 seconds fast of NTP time"),
        ("0.000012345 seconds fast of NTP time", "nan seconds fast of NTP time"),
        ("0.000012345 seconds fast of NTP time", "1e-5 seconds fast of NTP time"),
        ("0.000012345 seconds fast of NTP time", "0.000012345 seconds ahead of NTP time"),
        ("Tue May 12 14:15:01 2026", "Wed May 12 14:15:01 2026"),
        ("Tue May 12 14:15:01 2026", "Tue May 32 14:15:01 2026"),
        ("Tue May 12 14:15:01 2026", "Tue Mai 12 14:15:01 2026"),
        ("Tue May 12 14:15:01 2026", "2026-05-12T14:15:01"),
        ("Tue May 12 14:15:01 2026", "yesterday"),
        ("Root delay      : 0.000391626", "Root delay      : abc"),
        ("Root delay      : 0.000391626", "Root delay      : " + "9" * 400),
        ("Root dispersion : 0.000387918", "Root dispersion : -0.000387918"),
        ("Update interval : 1033.4", "Update interval : inf"),
        ("Leap status     : Normal", "Leap status     : normal"),
        ("Leap status     : Normal", "Leap status     : Unknown"),
        ("Leap status     : Normal\n", "Leap status     : Normal\nunexpected line without separator\n"),
        ("Leap status     : Normal\n", ""),
        ("Root dispersion : 0.000387918 seconds\n", ""),
        ("Ref time (UTC)  : Tue May 12 14:15:01 2026\n", ""),
    ],
)
def test_malformed_or_incomplete_tracking_fails_closed(old, new):
    assert UBUNTU_TRACKING.count(old) == 1
    tracking = UBUNTU_TRACKING.replace(old, new)
    with pytest.raises(ChronyEvidenceError):
        parse_chrony_tracking_fields(tracking)
    evidence = _parse(tracking)
    assert evidence.verified is False and evidence.synchronized is False
    assert evidence.reason.startswith("chrony tracking output is malformed")


@pytest.mark.parametrize("value", [None, b"Leap status : Normal", 7])
def test_non_text_tracking_fails_closed(value):
    with pytest.raises(ChronyEvidenceError):
        parse_chrony_tracking_fields(value)
    assert _parse(value).verified is False


@pytest.mark.parametrize(
    "row",
    [
        "^,*,169.254.169.254,2,10,377,1,-0.000004211,-0.000008612",
        "^,*,169.254.169.254,2,10,377,1,-0.000004211,-0.000008612,0.000194834,9",
        "%,*,169.254.169.254,2,10,377,1,-0.000004211,-0.000008612,0.000194834",
        "^,!,169.254.169.254,2,10,377,1,-0.000004211,-0.000008612,0.000194834",
        "^,*,,2,10,377,1,-0.000004211,-0.000008612,0.000194834",
        "^,*,169.254.169.254,x,10,377,1,-0.000004211,-0.000008612,0.000194834",
        "^,*,169.254.169.254,2,1.5,377,1,-0.000004211,-0.000008612,0.000194834",
        "^,*,169.254.169.254,2,10,400,1,-0.000004211,-0.000008612,0.000194834",
        "^,*,169.254.169.254,2,10,8,1,-0.000004211,-0.000008612,0.000194834",
        "^,*,169.254.169.254,2,10,377,-1,-0.000004211,-0.000008612,0.000194834",
        "^,*,169.254.169.254,2,10,377,4294967296,-0.000004211,-0.000008612,0.000194834",
        "^,*,169.254.169.254,2,10,377,1,nan,-0.000008612,0.000194834",
        "^,*,169.254.169.254,2,10,377,1,-0.000004211,inf,0.000194834",
        "^,*,169.254.169.254,2,10,377,1,-0.000004211,-0.000008612,1e999",
    ],
)
def test_malformed_sources_csv_fails_closed(row):
    with pytest.raises(ChronyEvidenceError):
        parse_chrony_sources(row + "\n")
    evidence = _parse(sources=row + "\n")
    assert evidence.verified is False
    assert "chrony sources output is malformed" in evidence.reason


def test_sources_parser_accepts_peers_and_multiple_candidates():
    rows = parse_chrony_sources(
        "^,+,169.254.169.253,2,10,377,40,0.000001,0.000001,0.0001\n"
        + UBUNTU_SOURCES
        + f"=,-,10.0.0.9,3,6,0,{CHRONY_NO_SAMPLE_LAST_RX},0.0,0.0,0.0\n"
    )
    assert [row.state for row in rows] == ["+", "*", "-"]
    assert rows[2].last_rx_seconds is None
    evidence = _parse(sources="".join(
        (
            "^,+,169.254.169.253,2,10,377,40,0.000001,0.000001,0.0001\n",
            UBUNTU_SOURCES,
            f"=,-,10.0.0.9,3,6,0,{CHRONY_NO_SAMPLE_LAST_RX},0.0,0.0,0.0\n",
        )
    ))
    assert evidence.verified, evidence.reason


def test_reference_age_is_accepted_throughout_a_normal_1024_second_poll():
    horizon = CHRONY_CLOCK_POLICY.reference_age_horizon(10)
    assert horizon == pytest.approx(1060.48)
    for age in range(0, 1061):
        evidence = _at_age(age)
        assert evidence.verified, (age, evidence.reason)
        assert evidence.reference_age_seconds == age
    # The VM's observed update interval (~1,033 s) is inside one poll cycle.
    assert _at_age(1033.4).verified
    assert _at_age(1060.48).verified
    for age in (1060.49, 1061, 1080, 1100, 2090, 86_400):
        evidence = _at_age(age)
        assert not evidence.verified
        assert "chrony reference evidence is stale" in evidence.reason


def test_adaptive_poll_tightens_the_reference_age_horizon():
    assert CHRONY_CLOCK_POLICY.reference_age_horizon(6) == pytest.approx(81.28)
    assert _at_age(81, poll=6).verified
    assert not _at_age(82, poll=6).verified
    assert CHRONY_CLOCK_POLICY.reference_age_horizon(-7) == pytest.approx(2**-7 * 1.02 + 16)
    assert _at_age(16, poll=-7).verified
    assert not _at_age(17, poll=-7).verified


@pytest.mark.parametrize("poll", [11, 12, 17, -8])
def test_poll_outside_policy_fails_closed(poll):
    with pytest.raises(ValueError, match="poll"):
        CHRONY_CLOCK_POLICY.reference_age_horizon(poll)
    evidence = _at_age(5, poll=poll)
    assert not evidence.verified
    assert "poll is outside the permitted policy" in evidence.reason


def test_hard_reference_age_ceiling_bounds_every_poll_and_ignores_update_interval():
    for poll in range(-7, 11):
        assert CHRONY_CLOCK_POLICY.reference_age_horizon(poll) <= 1080.0
    assert CHRONY_CLOCK_POLICY.max_reference_age_seconds == 1080.0
    long_gap = _at_age(1100, update_interval="100000.0")
    assert not long_gap.verified
    assert long_gap.reference_age_horizon_seconds == pytest.approx(1060.48)
    assert "update interval" not in long_gap.reason
    for poll in (True, 10.0, "10", None):
        with pytest.raises(ValueError):
            CHRONY_CLOCK_POLICY.reference_age_horizon(poll)


def test_default_policy_preserves_protocol_thresholds():
    policy = ChronyFreshnessPolicy()
    assert policy == CHRONY_CLOCK_POLICY
    assert policy.version == "chrony-freshness-v1"
    assert policy.max_offset_seconds == 0.050
    assert policy.max_estimated_error_seconds == 0.050
    assert policy.max_poll_interval_seconds == 1024.0
    assert policy.max_poll_log2 == 10
    assert policy.max_reference_age_seconds == 1080.0
    assert policy.max_decision_observation_age_seconds == 5.0
    assert policy.max_cached_observation_age_seconds == 10.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"version": ""},
        {"max_offset_seconds": 0.0},
        {"max_offset_seconds": -0.05},
        {"max_offset_seconds": float("nan")},
        {"max_estimated_error_seconds": float("inf")},
        {"max_offset_seconds": True},
        {"max_poll_interval_seconds": 1000.0},
        {"max_poll_interval_seconds": 0.5},
        {"reference_age_grace_seconds": 60.0},
        {"max_reference_age_seconds": 1000.0},
        {"max_reference_age_seconds": 2049.0},
        {"max_observation_duration_seconds": 0.0},
        {"max_clock_jump_seconds": 0.0},
        {"max_cached_observation_age_seconds": 4.0},
    ],
)
def test_policy_rejects_invalid_or_unbounded_configuration(overrides):
    with pytest.raises(ValueError):
        ChronyFreshnessPolicy(**overrides)


def test_future_reference_and_decision_before_observation_fail_closed():
    future = parse_chrony_tracking(
        tracking_output(REFERENCE),
        observed_at_utc=REFERENCE - timedelta(seconds=1),
        sources_output=sources_row(last_rx=0),
        **TIMING,
    )
    assert not future.verified
    assert "chrony reference time is in the future" in future.reason
    evidence = _at_age(5)
    early = evaluate_clock_evidence(
        evidence, decision_utc=evidence.observed_at_utc - timedelta(microseconds=1)
    )
    assert not early.verified
    assert "clock evidence was observed after the decision time" in early.reasons


def test_reference_staleness_is_rechecked_at_the_decision_time():
    evidence = _at_age(1000)
    assert evaluate_clock_evidence(
        evidence, decision_utc=REFERENCE + timedelta(seconds=1060)
    ).verified
    stale = evaluate_clock_evidence(evidence, decision_utc=REFERENCE + timedelta(seconds=1061))
    assert not stale.verified
    assert stale.reasons == ("chrony reference evidence is stale at the decision time",)
    naive = evaluate_clock_evidence(evidence, decision_utc=datetime(2026, 5, 12, 14, 20))
    assert "clock decision time is unavailable" in naive.reasons


@pytest.mark.parametrize(
    "system_time, verified",
    [
        ("0.050000000 seconds slow", True),
        ("0.050000000 seconds fast", True),
        ("0.050000001 seconds slow", False),
        ("0.051000000 seconds fast", False),
        ("0.500000000 seconds slow", False),
    ],
)
def test_offset_threshold_is_unchanged_at_50_ms(system_time, verified):
    evidence = _at_age(5, system_time=system_time)
    assert evidence.verified is verified
    if not verified:
        assert "chrony system offset exceeds tolerance" in evidence.reason


@pytest.mark.parametrize(
    "delay, dispersion, verified",
    [
        ("0.000000000", "0.050000000", True),
        ("0.000000000", "0.050000001", False),
        ("0.100000000", "0.000001000", False),
        ("2.000000000", "0.000000000", False),
    ],
)
def test_estimated_error_is_bounded(delay, dispersion, verified):
    evidence = _at_age(5, root_delay=delay, root_dispersion=dispersion)
    assert evidence.verified is verified
    if not verified:
        assert "chrony estimated error exceeds tolerance" in evidence.reason


@pytest.mark.parametrize(
    "row, reason",
    [
        (sources_row(last_rx=CHRONY_NO_SAMPLE_LAST_RX), "chrony selected source has no samples"),
        (sources_row(reach="0"), "chrony selected source is unreachable"),
        (sources_row(reach="376"), "chrony selected source missed its most recent poll"),
        (sources_row(last_rx=1061), "chrony selected-source sample is stale"),
        (sources_row(state="+"), "chrony has no selected synchronization source"),
        (sources_row(state="x"), "chrony has no selected synchronization source"),
        (sources_row(stratum=0), "chrony selected-source stratum is outside 1-15"),
        (sources_row(stratum=16), "chrony selected-source stratum is outside 1-15"),
        (sources_row(address="169.254.169.253"), "chrony tracking reference does not match"),
        (sources_row(mode="#", address="GPS"), "chrony selected source is not an NTP server or peer"),
        ("", "chrony has no selected synchronization source"),
        (sources_row() + "\n" + sources_row(address="10.0.0.1"), "multiple selected sources"),
    ],
)
def test_selected_source_reachability_and_identity_fail_closed(row, reason):
    evidence = _parse(sources=row + "\n")
    assert not evidence.verified
    assert reason in evidence.reason


def test_peer_and_ipv6_sources_match_the_tracking_reference():
    assert _parse(sources=sources_row(mode="=") + "\n").verified
    ipv6 = UBUNTU_TRACKING.replace(
        "A9FEA9FE (metadata.google.internal)", "5D2B8C17 (2001:db8::1)"
    )
    assert _parse(ipv6, sources=sources_row(address="2001:db8::1") + "\n").verified
    mismatch = _parse(ipv6, sources=sources_row(address="2001:db8::2") + "\n")
    assert not mismatch.verified
    assert "does not match the selected source" in mismatch.reason


@pytest.mark.parametrize(
    "tracking, reason",
    [
        (UNSYNCHRONISED_TRACKING, "chrony leap status is not Normal"),
        (
            UBUNTU_TRACKING.replace(
                "A9FEA9FE (metadata.google.internal)", "7F7F0101 ()"
            ).replace("Stratum         : 3", "Stratum         : 10"),
            "chrony is not synchronised to an external reference",
        ),
        (
            UBUNTU_TRACKING.replace("A9FEA9FE (metadata.google.internal)", "00000000 ()"),
            "chrony is not synchronised to an external reference",
        ),
        (UBUNTU_TRACKING.replace("Leap status     : Normal", "Leap status     : Insert second"), "not Normal"),
        (UBUNTU_TRACKING.replace("Stratum         : 3", "Stratum         : 16"), "chrony stratum is outside 1-15"),
    ],
)
def test_unsynchronised_local_and_leap_states_fail_closed(tracking, reason):
    evidence = _parse(tracking)
    assert not evidence.verified
    assert reason in evidence.reason


def test_unsynchronised_output_never_certifies():
    evidence = _parse(UNSYNCHRONISED_TRACKING, unsynchronised_sources())
    assert not evidence.verified and not evidence.synchronized
    assert evidence.reference_id == "00000000"
    assert not chrony_evidence(OBSERVED, verified=False).verified


@pytest.mark.parametrize(
    "duration, divergence, reason",
    [
        (2.1, 0.0, "observation duration is unavailable or excessive"),
        (-0.001, 0.0, "observation duration is unavailable or excessive"),
        (None, 0.0, "observation duration is unavailable or excessive"),
        (0.01, 0.051, "wall clock jumped"),
        (0.01, -0.051, "wall clock jumped"),
        (0.01, None, "wall clock jumped"),
        (0.01, float("nan"), "wall clock jumped"),
    ],
)
def test_observation_duration_and_clock_jumps_fail_closed(duration, divergence, reason):
    evidence = _parse(
        observation_duration_seconds=duration, wall_monotonic_divergence_seconds=divergence
    )
    assert not evidence.verified
    assert reason in evidence.reason
    assert _parse(observation_duration_seconds=2.0, wall_monotonic_divergence_seconds=0.05).verified


def test_source_loss_and_recovery_follow_the_reach_register():
    sequence = [
        (sources_row(reach="377", last_rx=5), 5, True),
        (sources_row(reach="376", last_rx=1029), 1029, False),
        (sources_row(reach="374", last_rx=2053), 2053, False),
        (sources_row(state="?", reach="0", last_rx=CHRONY_NO_SAMPLE_LAST_RX), 4000, False),
        (sources_row(reach="1", last_rx=2), 2, True),
        (sources_row(reach="3", last_rx=3), 3, True),
    ]
    observed = REFERENCE
    outcomes = []
    for row, age, expected in sequence:
        observed += timedelta(seconds=1024)
        evidence = parse_chrony_tracking(
            tracking_output(observed - timedelta(seconds=age)),
            observed_at_utc=observed,
            sources_output=row + "\n",
            **TIMING,
        )
        outcomes.append(evidence.verified)
        assert evidence.verified is expected, (row, evidence.reason)
    assert outcomes == [True, False, False, False, True, True]


class _Completed(SimpleNamespace):
    pass


def _reader(outputs, *, walls=None, monotonic=None):
    calls = []
    walls = iter(walls or (OBSERVED - timedelta(seconds=0.02), OBSERVED))
    monotonic = iter(monotonic or (100.0, 100.02))

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        result = outputs[len(calls) - 1]
        if isinstance(result, BaseException):
            raise result
        return result

    evidence = read_chrony_clock_health(
        now_fn=lambda: next(walls), monotonic_fn=lambda: next(monotonic), runner=runner
    )
    return evidence, calls


def test_reader_runs_only_read_only_chronyc_queries_and_brackets_the_observation():
    evidence, calls = _reader(
        [
            _Completed(returncode=0, stdout=UBUNTU_TRACKING, stderr=""),
            _Completed(returncode=0, stdout=UBUNTU_SOURCES, stderr=""),
        ]
    )
    assert evidence.verified, evidence.reason
    assert [command for command, _ in calls] == [
        ["chronyc", "-n", "tracking"],
        ["chronyc", "-n", "-c", "sources"],
    ]
    assert CHRONY_TRACKING_COMMAND == ("chronyc", "-n", "tracking")
    assert CHRONY_SOURCES_COMMAND == ("chronyc", "-n", "-c", "sources")
    for _, kwargs in calls:
        assert kwargs == {"check": False, "capture_output": True, "text": True, "timeout": 2.0}
    assert evidence.observed_at_utc == OBSERVED
    assert evidence.observation_duration_seconds == pytest.approx(0.02)
    assert evidence.wall_monotonic_divergence_seconds == pytest.approx(0.0, abs=1e-9)
    source = inspect.getsource(clock_module)
    for mutating in ("makestep", "settime", "burst", "timedatectl", "hwclock", "adjtime"):
        assert mutating not in source


@pytest.mark.parametrize(
    "outputs, reason",
    [
        ([FileNotFoundError("chronyc")], "chronyc -n tracking unavailable: FileNotFoundError"),
        ([PermissionError("denied")], "chronyc -n tracking unavailable: PermissionError"),
        (
            [subprocess.TimeoutExpired(["chronyc"], 2.0)],
            "chronyc -n tracking unavailable: TimeoutExpired",
        ),
        ([_Completed(returncode=1, stdout="", stderr="506 Cannot talk to daemon")], "exited with status 1"),
        (
            [
                _Completed(returncode=0, stdout=UBUNTU_TRACKING, stderr=""),
                _Completed(returncode=1, stdout="", stderr=""),
            ],
            "chronyc -n -c sources exited with status 1",
        ),
        (
            [
                _Completed(returncode=0, stdout=UBUNTU_TRACKING, stderr=""),
                FileNotFoundError("chronyc"),
            ],
            "chronyc -n -c sources unavailable",
        ),
        (
            [
                _Completed(returncode=0, stdout="", stderr=""),
                _Completed(returncode=0, stdout=UBUNTU_SOURCES, stderr=""),
            ],
            "chrony tracking output is malformed",
        ),
    ],
)
def test_reader_fails_closed_when_chrony_is_unavailable_or_silent(outputs, reason):
    evidence, _ = _reader(outputs)
    assert evidence.verified is False and evidence.synchronized is False
    assert reason in evidence.reason
    assert evidence.policy_version == CHRONY_POLICY_VERSION
    assert not evaluate_clock_evidence(evidence).verified


def test_reader_detects_a_wall_clock_jump_during_observation():
    evidence, _ = _reader(
        [
            _Completed(returncode=0, stdout=UBUNTU_TRACKING, stderr=""),
            _Completed(returncode=0, stdout=UBUNTU_SOURCES, stderr=""),
        ],
        walls=(OBSERVED - timedelta(seconds=1.02), OBSERVED),
    )
    assert not evidence.verified
    assert "wall clock jumped while chrony evidence was observed" in evidence.reason


def test_evidence_round_trips_and_decoding_is_strict():
    evidence = _parse()
    stored = evidence.as_dict()
    assert stored["observed_at_utc"] == "2026-05-12T14:15:02.000000Z"
    assert ClockHealthEvidence.from_dict(stored) == evidence
    with pytest.raises(ValueError, match="unknown"):
        ClockHealthEvidence.from_dict({**stored, "extra": 1})
    for key in ("observed_at_utc", "verified", "reason"):
        with pytest.raises(ValueError, match="missing"):
            ClockHealthEvidence.from_dict({k: v for k, v in stored.items() if k != key})
    for key, value in (
        ("verified", 1),
        ("selected_source_poll_log2", 10.0),
        ("offset_seconds", "0.0"),
        ("offset_seconds", True),
        ("reference_id", 7),
        ("source", ""),
    ):
        with pytest.raises(TypeError):
            ClockHealthEvidence.from_dict({**stored, key: value})
    with pytest.raises(ValueError, match="malformed"):
        ClockHealthEvidence.from_dict({**stored, "reference_time_utc": "not a time"})
    with pytest.raises(ValueError):
        ClockHealthEvidence.from_dict({**stored, "observed_at_utc": "2026-05-12T14:15:02"})
    with pytest.raises(ValueError):
        ClockHealthEvidence.from_dict(["not", "a", "mapping"])


@pytest.mark.parametrize(
    "changes",
    [
        {"offset_seconds": 0.5},
        {"reference_time_utc": REFERENCE - timedelta(seconds=2000)},
        {"reference_age_seconds": 0.5},
        {"reference_age_horizon_seconds": 1080.0},
        {"estimated_error_seconds": 0.0},
        {"maximum_error_bound_seconds": 0.0},
        {"policy_version": "chrony-freshness-v0"},
        {"policy_version": None},
        {"selected_source_reach": 0},
        {"selected_source_poll_log2": 12},
        {"leap_status": "Not synchronised"},
        {"wall_monotonic_divergence_seconds": 1.0},
    ],
)
def test_forged_verified_evidence_is_recomputed_and_rejected(changes):
    forged = replace(_parse(), **changes)
    assert forged.verified is True
    assert not evaluate_clock_evidence(forged).verified
    stored = forged.as_dict()
    with pytest.raises(ValueError, match="verification flag contradicts its evidence"):
        _validate_bridge_clock_evidence(
            stored, watermark=REFERENCE, witness_time=REFERENCE
        )


def test_legacy_eight_field_verified_evidence_is_rejected_as_unpoliced():
    legacy = {
        "observed_at_utc": "2026-05-12T14:15:02.000000Z",
        "synchronized": True,
        "verified": True,
        "offset_seconds": 0.001,
        "reference_time_utc": "2026-05-12T14:15:01.000000Z",
        "leap_status": "Normal",
        "source": "chronyc tracking",
        "reason": None,
    }
    decoded = ClockHealthEvidence.from_dict(legacy)
    assert decoded.policy_version is None
    assessment = evaluate_clock_evidence(decoded)
    assert not assessment.verified
    assert "clock evidence was not produced under the current freshness policy" in assessment.reasons
    with pytest.raises(ValueError, match="contradicts"):
        _validate_bridge_clock_evidence(legacy, watermark=REFERENCE, witness_time=REFERENCE)
    unverified = {**legacy, "verified": False, "synchronized": False}
    assert _validate_bridge_clock_evidence(
        unverified, watermark=REFERENCE, witness_time=REFERENCE
    ) is False


def test_journal_accepts_only_policy_consistent_flags_within_the_window():
    good = _parse().as_dict()
    assert _validate_bridge_clock_evidence(good, watermark=REFERENCE, witness_time=REFERENCE) is True
    bad = _parse(observation_duration_seconds=3.0).as_dict()
    assert bad["verified"] is False
    assert _validate_bridge_clock_evidence(bad, watermark=REFERENCE, witness_time=REFERENCE) is False
    with pytest.raises(ValueError, match="contradicts"):
        _validate_bridge_clock_evidence(
            {**bad, "verified": True, "synchronized": True},
            watermark=REFERENCE,
            witness_time=REFERENCE,
        )
    with pytest.raises(ValueError, match="contradicts"):
        _validate_bridge_clock_evidence(
            {**good, "verified": False, "synchronized": False},
            watermark=REFERENCE,
            witness_time=REFERENCE,
        )
    with pytest.raises(ValueError, match="outside the prospective window"):
        _validate_bridge_clock_evidence(
            good, watermark=REFERENCE, witness_time=OBSERVED + timedelta(seconds=1)
        )
    with pytest.raises(ValueError, match="outside the prospective window"):
        _validate_bridge_clock_evidence(
            good, watermark=REFERENCE - timedelta(seconds=61), witness_time=REFERENCE - timedelta(seconds=62)
        )


def test_clock_module_is_dependency_free_and_cannot_reach_features_or_targets():
    source = inspect.getsource(clock_module)
    assert "btc_quarter_hour_engine" not in source.split('"""', 2)[2]
    for name in ("label", "target", "feature", "model", "pandas", "numpy"):
        assert f"import {name}" not in source
        assert f"from {name}" not in source


def test_frozen_target_models_features_and_deadline_are_unchanged():
    from btc_quarter_hour_engine.research import prospective_contract as contract
    from btc_quarter_hour_engine.research import prospective_ledger as ledger
    from btc_quarter_hour_engine.research.model_freeze import frozen_predictors

    assert contract.PREDICTION_DEADLINE_SECONDS == ledger.PREDICTION_DEADLINE_SECONDS == 60
    assert contract.WINDOW_END_EXCLUSIVE == ledger.WINDOW_END_EXCLUSIVE
    predictors = frozen_predictors()
    assert set(predictors) == {"range_volatility", "full_v2"}
    assert len(predictors["range_volatility"]) == 12
    assert len(predictors["full_v2"]) == 91
