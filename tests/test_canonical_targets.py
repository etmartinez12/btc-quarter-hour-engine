from __future__ import annotations

import math

import numpy as np
import pandas as pd

from btc_quarter_hour_engine.targets import build_canonical_direction_targets


def _boundaries(times, prices, eligible=None, sessions=None):
    eligible = eligible or [True] * len(times)
    sessions = sessions or ["session"] * len(times)
    timestamps = pd.to_datetime(times, utc=True)
    return pd.DataFrame(
        {
            "boundary_time_utc": timestamps,
            "midpoint": prices,
            "canonical_target_eligible": eligible,
            "eligibility_reason": [
                "eligible" if value else "heartbeat_stale" for value in eligible
            ],
            "session_id": sessions,
            "connection_id": [f"{session}-connection" for session in sessions],
            "source_sequence_num": list(range(1, len(times) + 1)),
            "source_state_time_utc": timestamps,
        }
    )


def test_exact_t_plus_15_join_does_not_bridge_missing_boundary():
    source = _boundaries(
        ["2026-01-01T00:00:00Z", "2026-01-01T00:30:00Z"],
        [100.0, 101.0],
    )

    actual = build_canonical_direction_targets(source)

    assert pd.isna(actual.loc[0, "target"])
    assert not actual.loc[0, "target_eligible"]
    assert actual.loc[0, "target_eligibility_reason"] == "missing_t_plus_15m_boundary"
    assert actual.loc[0, "target_time"] == pd.Timestamp("2026-01-01T00:15:00Z")
    assert actual.loc[0, "label_available_time"] == actual.loc[0, "target_time"]


def test_source_and_target_boundary_eligibility_are_required():
    source_ineligible = _boundaries(
        ["2026-01-01T00:00:00Z", "2026-01-01T00:15:00Z"],
        [100.0, 101.0],
        eligible=[False, True],
    )
    target_ineligible = _boundaries(
        ["2026-01-01T00:00:00Z", "2026-01-01T00:15:00Z"],
        [100.0, 101.0],
        eligible=[True, False],
    )

    source_result = build_canonical_direction_targets(source_ineligible)
    target_result = build_canonical_direction_targets(target_ineligible)

    assert source_result.loc[0, "target_eligibility_reason"] == "source_boundary_ineligible"
    assert pd.isna(source_result.loc[0, "target"])
    assert target_result.loc[0, "target_eligibility_reason"] == "target_boundary_ineligible"
    assert pd.isna(target_result.loc[0, "target"])


def test_up_down_and_flat_targets_use_nullable_integer_dtype():
    source = _boundaries(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:15:00Z",
            "2026-01-01T00:30:00Z",
            "2026-01-01T00:45:00Z",
        ],
        [100.0, 101.0, 99.0, 99.0],
    )

    actual = build_canonical_direction_targets(source)

    assert str(actual["target"].dtype) == "Int64"
    assert actual.loc[0, "target"] == 1
    assert actual.loc[1, "target"] == 0
    assert pd.isna(actual.loc[2, "target"])
    assert actual.loc[2, "target_eligibility_reason"] == "flat_move"
    assert actual.loc[2, "log_return_15m"] == 0.0
    assert pd.isna(actual.loc[3, "target"])


def test_cross_session_exact_boundaries_can_form_a_label():
    source = _boundaries(
        ["2026-01-01T00:00:00Z", "2026-01-01T00:15:00Z"],
        [100.0, 102.0],
        sessions=["session-a", "session-b"],
    )

    actual = build_canonical_direction_targets(source)

    assert actual.loc[0, "target"] == 1
    assert actual.loc[0, "session_id_t"] == "session-a"
    assert actual.loc[0, "session_id_t_plus_15m"] == "session-b"


def test_invalid_prices_never_produce_infinite_returns():
    source = _boundaries(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:15:00Z",
            "2026-01-01T00:30:00Z",
            "2026-01-01T00:45:00Z",
            "2026-01-01T01:00:00Z",
            "2026-01-01T01:15:00Z",
        ],
        [0.0, 100.0, np.inf, 100.0, np.nan, 99.0],
    )

    actual = build_canonical_direction_targets(source)

    assert actual.loc[0, "target_eligibility_reason"] == "invalid_price_t"
    assert actual.loc[1, "target_eligibility_reason"] == "invalid_price_t_plus_15m"
    assert actual.loc[2, "target_eligibility_reason"] == "invalid_price_t"
    assert actual.loc[3, "target_eligibility_reason"] == "invalid_price_t_plus_15m"
    assert actual.loc[4, "target_eligibility_reason"] == "invalid_price_t"
    assert all(not math.isinf(value) for value in actual["log_return_15m"].dropna())


def test_extreme_finite_prices_keep_log_return_finite():
    source = _boundaries(
        ["2026-01-01T00:00:00Z", "2026-01-01T00:15:00Z"],
        [1e-308, 1e308],
    )

    actual = build_canonical_direction_targets(source)

    assert actual.loc[0, "target"] == 1
    assert np.isfinite(actual.loc[0, "log_return_15m"])
