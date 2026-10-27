from __future__ import annotations

from collections.abc import Mapping

from btc_quarter_hour_engine.research.features_v2 import (
    V2_CALENDAR_COLUMNS,
    V2_CROSS_WINDOW_COLUMNS,
    V2_MOMENTUM_COLUMNS,
    V2_PREDICTOR_COLUMNS,
    V2_WINDOWS_MINUTES,
)


FAMILY_CONTRACT_VERSION = "bbo-causal-v2-families-v1"


def _windows(*bases: str) -> tuple[str, ...]:
    return tuple(
        f"{base}_{minutes}m"
        for minutes in V2_WINDOWS_MINUTES
        for base in bases
    )


V2_FEATURE_FAMILIES = {
    "momentum": V2_MOMENTUM_COLUMNS,
    "spread": ("asof_spread_bps",) + _windows(
        "bbo_spread_mean_bps", "bbo_spread_std_bps", "bbo_spread_max_bps"
    ),
    "order_book_imbalance": ("asof_top_book_imbalance",) + _windows(
        "bbo_imbalance_mean", "bbo_imbalance_std",
        "l2_update_imbalance", "l2_delete_imbalance",
    ),
    "activity_intensity": _windows(
        "bbo_state_rate_per_minute", "midpoint_change_rate_per_minute",
        "l2_envelope_rate_per_minute", "l2_update_rate_per_minute",
        "l2_delete_rate_per_minute",
    ),
    "range_volatility": _windows(
        "distance_from_high", "distance_from_low", "midpoint_range_bps"
    ),
    "cross_window_regimes": V2_CROSS_WINDOW_COLUMNS,
    "calendar": V2_CALENDAR_COLUMNS,
}


def validate_feature_families(
    families: Mapping[str, tuple[str, ...]] = V2_FEATURE_FAMILIES,
) -> None:
    if tuple(families) != tuple(V2_FEATURE_FAMILIES) or any(
        not columns for columns in families.values()
    ):
        raise ValueError("Expected the seven ordered, nonempty V2 feature families")
    columns = [column for group in families.values() for column in group]
    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate predictors in V2 feature-family partition")
    forbidden = {
        "asof_midpoint", "asof_best_bid_size", "asof_best_ask_size",
        "asof_book_synced", "hour_utc", "quarter_hour_slot", "day_of_week",
        "target", "target_time", "label_available_time", "target_log_return_15m",
        "timestamp", "price_change", "trainable", "feature_ready", "target_eligible",
    }
    if any(
        column in forbidden
        or any(token in column for token in (
            "canonical", "coverage", "valid_fraction", "synced_fraction", "count",
            "bbo_midpoint_min", "bbo_midpoint_max",
        ))
        for column in columns
    ):
        raise ValueError("Forbidden predictor in V2 feature-family partition")
    if (
        len(V2_PREDICTOR_COLUMNS) != 91
        or len(set(V2_PREDICTOR_COLUMNS)) != 91
        or len(columns) != 91
        or set(columns) != set(V2_PREDICTOR_COLUMNS)
    ):
        raise ValueError("Feature families must cover exactly the 91 V2 predictors")
    for family, expected in V2_FEATURE_FAMILIES.items():
        if families[family] != expected:
            raise ValueError(f"V2 family membership changed for {family}")


def localization_variants() -> dict[tuple[str, str], tuple[str, ...]]:
    validate_feature_families()
    variants = {("reference", "all"): V2_PREDICTOR_COLUMNS}
    for mode in ("group_only", "leave_one_group_out"):
        for family, columns in V2_FEATURE_FAMILIES.items():
            members = set(columns)
            variants[mode, family] = tuple(
                column for column in V2_PREDICTOR_COLUMNS
                if (column in members) == (mode == "group_only")
            )
    return variants
