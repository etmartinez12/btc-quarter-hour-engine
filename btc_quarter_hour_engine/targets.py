from __future__ import annotations

import numpy as np
import pandas as pd


CANONICAL_TARGET_ELIGIBILITY_REASONS = (
    "source_boundary_ineligible",
    "missing_t_plus_15m_boundary",
    "target_boundary_ineligible",
    "invalid_price_t",
    "invalid_price_t_plus_15m",
    "flat_move",
    "eligible",
)


def build_direction_target(boundary_prices: pd.DataFrame, price_column: str = "midpoint") -> pd.DataFrame:
    """Build quarter-hour labels from boundary t to boundary t+15m.

    The label uses only the canonical price observed at boundary t and the next exact
    quarter-hour canonical price. Flat moves are unlabeled because the research target
    is defined only for strictly up or strictly down moves.
    """

    dataset = boundary_prices[["timestamp", price_column]].copy()
    dataset = dataset.sort_values("timestamp").reset_index(drop=True)
    dataset = dataset.rename(columns={price_column: "price_t"})

    timestamps = pd.to_datetime(dataset["timestamp"], utc=True)
    deltas_seconds = timestamps.diff().dt.total_seconds()
    invalid_deltas = deltas_seconds.dropna()
    if not invalid_deltas.empty and not invalid_deltas.eq(15 * 60).all():
        bad_rows = invalid_deltas.index[~invalid_deltas.eq(15 * 60)].tolist()
        raise ValueError(
            "Boundary timestamps must be exactly 15 minutes apart; found non-15-minute gaps at rows "
            f"{bad_rows}."
        )

    dataset["price_t_plus_15m"] = dataset["price_t"].shift(-1)
    dataset["price_change"] = dataset["price_t_plus_15m"] - dataset["price_t"]
    valid_prices = dataset["price_t"].gt(0) & dataset["price_t_plus_15m"].gt(0)
    dataset["log_return_15m"] = np.nan
    dataset.loc[valid_prices, "log_return_15m"] = np.log(
        dataset.loc[valid_prices, "price_t_plus_15m"] / dataset.loc[valid_prices, "price_t"]
    )
    dataset["target"] = pd.Series(pd.NA, index=dataset.index, dtype="Int64")
    dataset.loc[valid_prices & dataset["price_change"].gt(0), "target"] = 1
    dataset.loc[valid_prices & dataset["price_change"].lt(0), "target"] = 0
    return dataset


def build_canonical_direction_targets(
    canonical_boundaries: pd.DataFrame,
    price_column: str = "midpoint",
) -> pd.DataFrame:
    """Build exact t-to-t+15m labels without bridging missing boundaries.

    Ineligibility reason precedence is source eligibility, target-boundary
    presence, target eligibility, source price validity, target price validity,
    flat move, then eligible.
    """
    required = {
        "boundary_time_utc",
        price_column,
        "canonical_target_eligible",
        "eligibility_reason",
        "session_id",
        "connection_id",
        "source_sequence_num",
        "source_state_time_utc",
    }
    missing_columns = required.difference(canonical_boundaries.columns)
    if missing_columns:
        raise ValueError(
            "Canonical boundaries are missing required columns: "
            f"{sorted(missing_columns)}"
        )

    boundaries = canonical_boundaries.copy()
    boundaries["boundary_time_utc"] = pd.to_datetime(
        boundaries["boundary_time_utc"], utc=True, errors="raise"
    )
    if boundaries["boundary_time_utc"].duplicated().any():
        duplicates = boundaries.loc[
            boundaries["boundary_time_utc"].duplicated(keep=False), "boundary_time_utc"
        ].astype(str).tolist()
        raise ValueError(f"Canonical boundary timestamps must be unique: {duplicates}")
    boundaries = boundaries.sort_values("boundary_time_utc").reset_index(drop=True)
    boundaries[price_column] = pd.to_numeric(boundaries[price_column], errors="coerce")
    by_time = boundaries.set_index("boundary_time_utc", drop=False)

    rows: list[dict[str, object]] = []
    for source in boundaries.to_dict(orient="records"):
        timestamp = source["boundary_time_utc"]
        target_time = timestamp + pd.Timedelta(minutes=15)
        target = by_time.loc[target_time] if target_time in by_time.index else None
        price_t = source[price_column]
        price_t_plus = target[price_column] if target is not None else np.nan

        finite_price_t = pd.notna(price_t) and np.isfinite(price_t)
        finite_price_t_plus = pd.notna(price_t_plus) and np.isfinite(price_t_plus)
        valid_price_t = finite_price_t and price_t > 0
        valid_price_t_plus = finite_price_t_plus and price_t_plus > 0

        price_change = (
            float(price_t_plus - price_t)
            if finite_price_t and finite_price_t_plus
            else np.nan
        )
        log_return = np.nan
        if valid_price_t and valid_price_t_plus:
            with np.errstate(over="ignore", under="ignore", divide="ignore"):
                price_ratio = price_t_plus / price_t
            log_return = (
                float(np.log(price_ratio))
                if np.isfinite(price_ratio) and price_ratio > 0
                else float(np.log(price_t_plus) - np.log(price_t))
            )

        reason = "eligible"
        if not bool(source["canonical_target_eligible"]):
            reason = "source_boundary_ineligible"
        elif target is None:
            reason = "missing_t_plus_15m_boundary"
        elif not bool(target["canonical_target_eligible"]):
            reason = "target_boundary_ineligible"
        elif not valid_price_t:
            reason = "invalid_price_t"
        elif not valid_price_t_plus:
            reason = "invalid_price_t_plus_15m"
        elif price_t == price_t_plus:
            reason = "flat_move"

        target_value: object = pd.NA
        if reason == "eligible":
            target_value = 1 if price_t_plus > price_t else 0

        row: dict[str, object] = {
            "timestamp": timestamp,
            "target_time": target_time,
            "price_t": price_t,
            "price_t_plus_15m": price_t_plus,
            "price_change": price_change,
            "log_return_15m": log_return,
            "target": target_value,
            "label_available_time": target_time,
            "target_eligible": reason == "eligible",
            "target_eligibility_reason": reason,
            "session_id_t": source["session_id"],
            "connection_id_t": source["connection_id"],
            "source_sequence_num_t": source["source_sequence_num"],
            "source_state_time_utc_t": source["source_state_time_utc"],
            "session_id_t_plus_15m": target["session_id"] if target is not None else None,
            "connection_id_t_plus_15m": target["connection_id"] if target is not None else None,
            "source_sequence_num_t_plus_15m": (
                target["source_sequence_num"] if target is not None else None
            ),
            "source_state_time_utc_t_plus_15m": (
                target["source_state_time_utc"] if target is not None else None
            ),
            "canonical_target_eligible_t": bool(source["canonical_target_eligible"]),
            "canonical_eligibility_reason_t": source["eligibility_reason"],
            "canonical_target_eligible_t_plus_15m": (
                bool(target["canonical_target_eligible"]) if target is not None else False
            ),
            "canonical_eligibility_reason_t_plus_15m": (
                target["eligibility_reason"] if target is not None else None
            ),
        }
        rows.append(row)

    columns = [
        "timestamp",
        "target_time",
        "price_t",
        "price_t_plus_15m",
        "price_change",
        "log_return_15m",
        "target",
        "label_available_time",
        "target_eligible",
        "target_eligibility_reason",
        "session_id_t",
        "connection_id_t",
        "source_sequence_num_t",
        "source_state_time_utc_t",
        "session_id_t_plus_15m",
        "connection_id_t_plus_15m",
        "source_sequence_num_t_plus_15m",
        "source_state_time_utc_t_plus_15m",
        "canonical_target_eligible_t",
        "canonical_eligibility_reason_t",
        "canonical_target_eligible_t_plus_15m",
        "canonical_eligibility_reason_t_plus_15m",
    ]
    result = pd.DataFrame(rows, columns=columns)
    result["target"] = pd.array(result["target"], dtype="Int64")
    for column in ("timestamp", "target_time", "label_available_time"):
        result[column] = pd.to_datetime(result[column], utc=True)
    for column in ("source_state_time_utc_t", "source_state_time_utc_t_plus_15m"):
        result[column] = pd.to_datetime(result[column], utc=True)
    return result
