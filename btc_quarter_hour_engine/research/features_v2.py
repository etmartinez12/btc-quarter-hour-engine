from __future__ import annotations

import numpy as np
import pandas as pd


V2_PREDICTOR_CONTRACT_VERSION = "bbo-causal-v2"
V2_WINDOWS_MINUTES = (15, 30, 60, 240)

V2_CURRENT_STATE_COLUMNS = (
    "asof_spread_bps",
    "asof_top_book_imbalance",
)
V2_MOMENTUM_COLUMNS = tuple(
    f"momentum_log_return_{minutes}m"
    for minutes in V2_WINDOWS_MINUTES
)
V2_WINDOW_PREDICTOR_BASES = (
    "bbo_spread_mean_bps",
    "bbo_spread_std_bps",
    "bbo_spread_max_bps",
    "bbo_imbalance_mean",
    "bbo_imbalance_std",
    "distance_from_high",
    "distance_from_low",
    "l2_update_imbalance",
    "l2_delete_imbalance",
    "midpoint_range_bps",
    "bbo_state_rate_per_minute",
    "midpoint_change_rate_per_minute",
    "l2_envelope_rate_per_minute",
    "l2_update_rate_per_minute",
    "l2_delete_rate_per_minute",
)
V2_WINDOW_PREDICTOR_COLUMNS = tuple(
    f"{base}_{minutes}m"
    for minutes in V2_WINDOWS_MINUTES
    for base in V2_WINDOW_PREDICTOR_BASES
)
V2_CROSS_WINDOW_COLUMNS = (
    "imbalance_mean_delta_15m_60m",
    "imbalance_mean_delta_15m_240m",
    "imbalance_mean_delta_60m_240m",
    "spread_mean_delta_bps_15m_60m",
    "spread_mean_delta_bps_15m_240m",
    "spread_std_delta_bps_15m_240m",
    "log_bbo_state_rate_ratio_15m_240m",
    "log_bbo_state_rate_ratio_60m_240m",
    "log_midpoint_change_rate_ratio_15m_240m",
    "log_midpoint_change_rate_ratio_60m_240m",
    "log_l2_update_rate_ratio_15m_240m",
    "log_l2_update_rate_ratio_60m_240m",
    "log_l2_delete_rate_ratio_15m_240m",
    "log_l2_delete_rate_ratio_60m_240m",
    "midpoint_range_fraction_15m_240m",
    "midpoint_range_fraction_60m_240m",
    "momentum_rate_delta_15m_240m",
    "momentum_rate_delta_60m_240m",
)
V2_CALENDAR_COLUMNS = (
    "hour_sin",
    "hour_cos",
    "quarter_hour_sin",
    "quarter_hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "is_weekend",
)
V2_PREDICTOR_COLUMNS = (
    V2_CURRENT_STATE_COLUMNS
    + V2_MOMENTUM_COLUMNS
    + V2_WINDOW_PREDICTOR_COLUMNS
    + V2_CROSS_WINDOW_COLUMNS
    + V2_CALENDAR_COLUMNS
)


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def _range_bps(frame: pd.DataFrame, minutes: int) -> pd.Series:
    minimum = _numeric_column(frame, f"bbo_midpoint_min_{minutes}m")
    maximum = _numeric_column(frame, f"bbo_midpoint_max_{minutes}m")
    valid = (
        np.isfinite(minimum.to_numpy())
        & np.isfinite(maximum.to_numpy())
        & minimum.gt(0).to_numpy()
        & maximum.gt(0).to_numpy()
        & maximum.ge(minimum).to_numpy()
    )
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    if valid.any():
        result.loc[valid] = (
            np.log(maximum.loc[valid] / minimum.loc[valid]) * 10_000.0
        )
    return result


def _per_minute_rate(
    frame: pd.DataFrame,
    count_column: str,
    minutes: int,
) -> pd.Series:
    return _numeric_column(frame, f"{count_column}_{minutes}m") / minutes


def _activity_log_ratio(
    frame: pd.DataFrame,
    count_column: str,
    short_minutes: int,
) -> pd.Series:
    short_rate = _per_minute_rate(frame, count_column, short_minutes)
    long_rate = _per_minute_rate(frame, count_column, 240)
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    valid = (
        np.isfinite(short_rate.to_numpy())
        & np.isfinite(long_rate.to_numpy())
        & short_rate.ge(0).to_numpy()
        & long_rate.ge(0).to_numpy()
    )
    if valid.any():
        result.loc[valid] = (
            np.log1p(short_rate.loc[valid])
            - np.log1p(long_rate.loc[valid])
        )
    return result


def _required_source_columns() -> tuple[str, ...]:
    required = [
        "timestamp",
        "feature_ready",
        "target_eligible",
        "trainable",
        "target",
        *V2_CURRENT_STATE_COLUMNS,
        *V2_MOMENTUM_COLUMNS,
    ]
    for minutes in V2_WINDOWS_MINUTES:
        required.extend(
            (
                f"bbo_spread_mean_bps_{minutes}m",
                f"bbo_spread_std_bps_{minutes}m",
                f"bbo_spread_max_bps_{minutes}m",
                f"bbo_imbalance_mean_{minutes}m",
                f"bbo_imbalance_std_{minutes}m",
                f"distance_from_high_{minutes}m",
                f"distance_from_low_{minutes}m",
                f"l2_update_imbalance_{minutes}m",
                f"l2_delete_imbalance_{minutes}m",
                f"bbo_midpoint_min_{minutes}m",
                f"bbo_midpoint_max_{minutes}m",
                f"bbo_state_count_{minutes}m",
                f"midpoint_change_count_{minutes}m",
                f"l2_envelope_count_{minutes}m",
                f"l2_update_count_{minutes}m",
                f"l2_delete_count_{minutes}m",
            )
        )
    return tuple(dict.fromkeys(required))


def build_v2_research_feature_frame(
    v1_features: pd.DataFrame,
) -> pd.DataFrame:
    """Derive the stationary V2 predictors from causal V1 summaries."""
    missing = sorted(set(_required_source_columns()).difference(v1_features.columns))
    if missing:
        raise ValueError(
            "V1 feature frame is missing required V2 source columns: "
            f"{missing}"
        )

    frame = v1_features.copy()
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    trainable = frame["trainable"].fillna(False).astype(bool)
    count_columns = [
        f"{count}_{minutes}m"
        for minutes in V2_WINDOWS_MINUTES
        for count in (
            "bbo_state_count",
            "midpoint_change_count",
            "l2_envelope_count",
            "l2_update_count",
            "l2_delete_count",
        )
    ]
    trainable_counts = frame.loc[trainable, count_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if trainable_counts.isna().any().any():
        bad = trainable_counts.columns[
            trainable_counts.isna().any()
        ].tolist()
        raise ValueError(
            "V2 trainable source counts contain missing or invalid values: "
            f"{bad}"
        )
    if not np.isfinite(trainable_counts.to_numpy(dtype=float)).all():
        raise ValueError("V2 trainable source counts contain non-finite values")
    if trainable_counts.lt(0).any().any():
        raise ValueError("V2 trainable source counts must be nonnegative")

    for minutes in V2_WINDOWS_MINUTES:
        frame[f"midpoint_range_bps_{minutes}m"] = _range_bps(frame, minutes)
        for predictor_base, count_base in (
            ("bbo_state_rate_per_minute", "bbo_state_count"),
            ("midpoint_change_rate_per_minute", "midpoint_change_count"),
            ("l2_envelope_rate_per_minute", "l2_envelope_count"),
            ("l2_update_rate_per_minute", "l2_update_count"),
            ("l2_delete_rate_per_minute", "l2_delete_count"),
        ):
            frame[f"{predictor_base}_{minutes}m"] = _per_minute_rate(
                frame,
                count_base,
                minutes,
            )

    for short_minutes, long_minutes in ((15, 60), (15, 240), (60, 240)):
        frame[
            f"imbalance_mean_delta_{short_minutes}m_{long_minutes}m"
        ] = (
            _numeric_column(frame, f"bbo_imbalance_mean_{short_minutes}m")
            - _numeric_column(frame, f"bbo_imbalance_mean_{long_minutes}m")
        )

    for short_minutes, long_minutes in ((15, 60), (15, 240)):
        frame[
            f"spread_mean_delta_bps_{short_minutes}m_{long_minutes}m"
        ] = (
            _numeric_column(frame, f"bbo_spread_mean_bps_{short_minutes}m")
            - _numeric_column(frame, f"bbo_spread_mean_bps_{long_minutes}m")
        )
    frame["spread_std_delta_bps_15m_240m"] = (
        _numeric_column(frame, "bbo_spread_std_bps_15m")
        - _numeric_column(frame, "bbo_spread_std_bps_240m")
    )

    for count_column, output_prefix in (
        ("bbo_state_count", "log_bbo_state_rate_ratio"),
        ("midpoint_change_count", "log_midpoint_change_rate_ratio"),
        ("l2_update_count", "log_l2_update_rate_ratio"),
        ("l2_delete_count", "log_l2_delete_rate_ratio"),
    ):
        for short_minutes in (15, 60):
            frame[
                f"{output_prefix}_{short_minutes}m_240m"
            ] = _activity_log_ratio(frame, count_column, short_minutes)

    for short_minutes in (15, 60):
        short_range = _numeric_column(
            frame,
            f"midpoint_range_bps_{short_minutes}m",
        )
        long_range = _numeric_column(frame, "midpoint_range_bps_240m")
        valid = (
            np.isfinite(short_range.to_numpy())
            & np.isfinite(long_range.to_numpy())
            & short_range.ge(0).to_numpy()
            & long_range.ge(0).to_numpy()
            & (
                long_range.gt(0).to_numpy()
                | (short_range.eq(0) & long_range.eq(0)).to_numpy()
            )
        )
        result = pd.Series(np.nan, index=frame.index, dtype=float)
        both_zero = valid & short_range.eq(0).to_numpy() & long_range.eq(0).to_numpy()
        positive_long = valid & long_range.gt(0).to_numpy()
        result.loc[both_zero] = 0.0
        result.loc[positive_long] = (
            short_range.loc[positive_long] / long_range.loc[positive_long]
        )
        frame[f"midpoint_range_fraction_{short_minutes}m_240m"] = result

    for short_minutes in (15, 60):
        frame[f"momentum_rate_delta_{short_minutes}m_240m"] = (
            _numeric_column(frame, f"momentum_log_return_{short_minutes}m")
            / short_minutes
            - _numeric_column(frame, "momentum_log_return_240m") / 240
        )

    hour = (
        timestamps.dt.hour
        + timestamps.dt.minute / 60.0
        + timestamps.dt.second / 3600.0
        + timestamps.dt.microsecond / 3_600_000_000.0
    )
    hour_angle = 2.0 * np.pi * hour / 24.0
    quarter_hour_position = (
        timestamps.dt.hour * 4.0
        + timestamps.dt.minute / 15.0
        + timestamps.dt.second / 900.0
        + timestamps.dt.microsecond / 900_000_000.0
    )
    quarter_hour_angle = 2.0 * np.pi * quarter_hour_position / 96.0
    day_of_week = timestamps.dt.dayofweek.to_numpy(dtype=float)
    day_angle = 2.0 * np.pi * day_of_week / 7.0
    frame["hour_sin"] = np.sin(hour_angle)
    frame["hour_cos"] = np.cos(hour_angle)
    frame["quarter_hour_sin"] = np.sin(quarter_hour_angle)
    frame["quarter_hour_cos"] = np.cos(quarter_hour_angle)
    frame["day_of_week_sin"] = np.sin(day_angle)
    frame["day_of_week_cos"] = np.cos(day_angle)
    frame["is_weekend"] = (timestamps.dt.dayofweek >= 5).astype(float)

    for column in V2_PREDICTOR_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    predictor_values = frame.loc[:, list(V2_PREDICTOR_COLUMNS)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    trainable_values = predictor_values.loc[trainable]
    if trainable_values.isna().any().any():
        bad = trainable_values.columns[
            trainable_values.isna().any()
        ].tolist()
        raise ValueError(
            "V2 trainable predictors contain missing or invalid values: "
            f"{bad}"
        )
    if not np.isfinite(trainable_values.to_numpy(dtype=float)).all():
        raise ValueError("V2 trainable predictors contain non-finite values")

    return frame
