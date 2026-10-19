from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from btc_quarter_hour_engine.acquisition.coinbase_websocket import (
    parse_coinbase_ws_message,
)
from btc_quarter_hour_engine.market_data.order_book import (
    Level2OrderBook,
    OrderBookState,
)
from btc_quarter_hour_engine.research.forward_dataset import (
    _iter_segment_frames,
    _segment_from_metadata,
)


WINDOWS_MINUTES = (15, 30, 60, 240)


def _ceil_quarter_hour(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("Feature timestamps must be timezone-aware")

    timestamp = timestamp.astimezone(timezone.utc)
    minute = (timestamp.minute // 15) * 15
    floor = timestamp.replace(minute=minute, second=0, microsecond=0)

    if timestamp == floor:
        return floor
    return floor + timedelta(minutes=15)


@dataclass(slots=True)
class _BucketStats:
    bbo_state_count: int = 0
    bbo_valid_count: int = 0
    bbo_synced_count: int = 0
    spread_bps_count: int = 0
    spread_bps_sum: float = 0.0
    spread_bps_sumsq: float = 0.0
    spread_bps_max: float = -math.inf
    imbalance_count: int = 0
    imbalance_sum: float = 0.0
    imbalance_sumsq: float = 0.0
    midpoint_min: float = math.inf
    midpoint_max: float = -math.inf
    midpoint_change_count: int = 0
    l2_envelope_count: int = 0
    incremental_envelope_count: int = 0
    snapshot_envelope_count: int = 0
    l2_update_count: int = 0
    bid_update_count: int = 0
    ask_update_count: int = 0
    delete_count: int = 0
    bid_delete_count: int = 0
    ask_delete_count: int = 0
    snapshot_level_count: int = 0

    # Causal point-in-time state after the latest frame received in this
    # ingest-time bucket. These fields are predictors, unlike canonical
    # source-time boundary prices used for labels.
    asof_book_synced: int = 0
    asof_midpoint: float = math.nan
    asof_spread_bps: float = math.nan
    asof_best_bid_size: float = math.nan
    asof_best_ask_size: float = math.nan
    asof_top_book_imbalance: float = math.nan


class _IngestFeatureReplay:
    """Reconstruct feature state in immutable recorded ingest order."""

    def __init__(self, product_id: str) -> None:
        self.product_id = product_id
        self.book = Level2OrderBook(product_id=product_id)
        self.connection_id: str | None = None
        self.last_envelope_sequence_num: int | None = None
        self.previous_midpoint: float | None = None
        self.buckets: dict[datetime, _BucketStats] = {}
        self.frames_seen = 0
        self.parsed_frames = 0
        self.malformed_frames = 0
        self.stale_frames = 0
        self.sequence_gaps = 0
        self.applied_l2_envelopes = 0

    def _bucket(self, ingest_time_utc: datetime) -> _BucketStats:
        key = _ceil_quarter_hour(ingest_time_utc)
        bucket = self.buckets.get(key)
        if bucket is None:
            bucket = _BucketStats()
            self.buckets[key] = bucket
        return bucket

    def _new_connection(self, connection_id: str | None) -> None:
        self.connection_id = connection_id
        self.last_envelope_sequence_num = None
        self.book.reset_for_new_connection()
        self.book.invalidate("new connection epoch")
        self.previous_midpoint = None

    def _record_bbo(self, *, bucket: _BucketStats) -> None:
        bucket.bbo_state_count += 1

        if self.book.is_synced():
            bucket.bbo_synced_count += 1

        bid = self.book.best_bid
        ask = self.book.best_ask
        bid_size = self.book.best_bid_size
        ask_size = self.book.best_ask_size

        valid = (
            self.book.is_synced()
            and bid is not None
            and ask is not None
            and bid_size is not None
            and ask_size is not None
            and math.isfinite(float(bid))
            and math.isfinite(float(ask))
            and math.isfinite(float(bid_size))
            and math.isfinite(float(ask_size))
            and float(bid) > 0
            and float(ask) > 0
            and float(ask) >= float(bid)
            and float(bid_size) >= 0
            and float(ask_size) >= 0
        )

        if not valid:
            self.previous_midpoint = None
            return

        bid = float(bid)
        ask = float(ask)
        bid_size = float(bid_size)
        ask_size = float(ask_size)
        midpoint = (bid + ask) / 2.0
        spread_bps = (ask - bid) / midpoint * 10_000.0

        bucket.bbo_valid_count += 1
        bucket.spread_bps_count += 1
        bucket.spread_bps_sum += spread_bps
        bucket.spread_bps_sumsq += spread_bps * spread_bps
        bucket.spread_bps_max = max(bucket.spread_bps_max, spread_bps)

        size_total = bid_size + ask_size
        if size_total > 0:
            imbalance = (bid_size - ask_size) / size_total
            bucket.imbalance_count += 1
            bucket.imbalance_sum += imbalance
            bucket.imbalance_sumsq += imbalance * imbalance

        bucket.midpoint_min = min(bucket.midpoint_min, midpoint)
        bucket.midpoint_max = max(bucket.midpoint_max, midpoint)

        if self.previous_midpoint is not None and midpoint != self.previous_midpoint:
            bucket.midpoint_change_count += 1

        self.previous_midpoint = midpoint

    def _record_asof(self, ingest_time_utc: datetime) -> None:
        bucket = self._bucket(ingest_time_utc)

        bid = self.book.best_bid
        ask = self.book.best_ask
        bid_size = self.book.best_bid_size
        ask_size = self.book.best_ask_size

        valid = (
            self.book.is_synced()
            and bid is not None
            and ask is not None
            and bid_size is not None
            and ask_size is not None
        )

        bucket.asof_book_synced = int(self.book.is_synced())

        if not valid:
            bucket.asof_midpoint = math.nan
            bucket.asof_spread_bps = math.nan
            bucket.asof_best_bid_size = math.nan
            bucket.asof_best_ask_size = math.nan
            bucket.asof_top_book_imbalance = math.nan
            return

        bid = float(bid)
        ask = float(ask)
        bid_size = float(bid_size)
        ask_size = float(ask_size)

        midpoint = (bid + ask) / 2.0
        total_size = bid_size + ask_size

        bucket.asof_midpoint = midpoint
        bucket.asof_spread_bps = (ask - bid) / midpoint * 10_000.0
        bucket.asof_best_bid_size = bid_size
        bucket.asof_best_ask_size = ask_size
        bucket.asof_top_book_imbalance = (
            (bid_size - ask_size) / total_size
            if total_size > 0
            else math.nan
        )

    def consume(
        self,
        *,
        raw: bytes,
        connection_id: str | None,
        frame_index: int,
        ingest_time_utc: datetime,
    ) -> None:
        try:
            self._consume(
                raw=raw,
                connection_id=connection_id,
                frame_index=frame_index,
                ingest_time_utc=ingest_time_utc,
            )
        finally:
            self._record_asof(ingest_time_utc)

    def _consume(
        self,
        *,
        raw: bytes,
        connection_id: str | None,
        frame_index: int,
        ingest_time_utc: datetime,
    ) -> None:
        del frame_index
        self.frames_seen += 1

        if connection_id != self.connection_id:
            self._new_connection(connection_id)

        try:
            event = parse_coinbase_ws_message(raw)
        except (ValueError, UnicodeError):
            self.malformed_frames += 1
            self.book.invalidate("malformed_source_state")
            self.previous_midpoint = None
            return

        self.parsed_frames += 1
        envelope = event.get("envelope")
        if envelope is None:
            return

        sequence_num = envelope.sequence_num
        previous = self.last_envelope_sequence_num

        if previous is not None and sequence_num <= previous:
            self.stale_frames += 1
            return

        if previous is not None and sequence_num > previous + 1:
            self.sequence_gaps += 1
            self.last_envelope_sequence_num = sequence_num
            self.book.invalidate("sequence_gap")
            self.previous_midpoint = None
            return

        self.last_envelope_sequence_num = sequence_num
        kind = event.get("type")

        if kind not in {"snapshot", "l2_data"}:
            return
        if event.get("product_id") != self.product_id:
            return

        updates = event.get("updates", [])

        try:
            if kind == "snapshot":
                self.book.apply_snapshot(
                    product_id=self.product_id,
                    levels={
                        "bid": [u for u in updates if u["side"] == "bid"],
                        "ask": [u for u in updates if u["side"] == "ask"],
                    },
                )
            else:
                if self.book.state not in {OrderBookState.SYNCED, OrderBookState.REBUILDING}:
                    return
                self.book.apply_updates(updates)
        except (ValueError, KeyError, TypeError):
            self.book.invalidate("malformed_source_state")
            self.previous_midpoint = None
            return

        self.book.last_sequence_num = sequence_num
        bucket = self._bucket(ingest_time_utc)
        bucket.l2_envelope_count += 1
        self.applied_l2_envelopes += 1

        if kind == "snapshot":
            bucket.snapshot_envelope_count += 1
            bucket.snapshot_level_count += len(updates)
        else:
            bucket.incremental_envelope_count += 1
            for update in updates:
                side = update.get("side")
                quantity = update.get("quantity")
                bucket.l2_update_count += 1

                if side == "bid":
                    bucket.bid_update_count += 1
                elif side == "ask":
                    bucket.ask_update_count += 1

                if quantity == 0:
                    bucket.delete_count += 1
                    if side == "bid":
                        bucket.bid_delete_count += 1
                    elif side == "ask":
                        bucket.ask_delete_count += 1

        self._record_bbo(bucket=bucket)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return np.nan
    return float(numerator / denominator)


def _variance(*, count: float, total: float, total_sq: float) -> float:
    if count <= 0:
        return np.nan
    mean = total / count
    return max(0.0, total_sq / count - mean * mean)


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    required = {"dataset_id", "product_id", "source", "source_segments", "artifacts"}
    missing = required.difference(manifest)
    if missing:
        raise ValueError(f"Research manifest missing required fields: {sorted(missing)}")
    return manifest


def _artifact_path(
    manifest_path: Path,
    manifest: dict[str, Any],
    data_kind: str,
) -> Path:
    for artifact in manifest["artifacts"]:
        if artifact.get("data_kind") == data_kind:
            return manifest_path.parent / artifact["path"]
    raise ValueError(f"Research manifest has no {data_kind!r} artifact")


def _manifest_segments(manifest: dict[str, Any], *, input_root: Path) -> list[Any]:
    segments = []
    for record in manifest["source_segments"]:
        metadata_path = input_root / record["metadata_path"]
        segment = _segment_from_metadata(
            metadata_path,
            source=manifest["source"],
            product_id=manifest["product_id"],
        )

        checks = {
            "session_id": segment.session_id,
            "segment_index": segment.segment_index,
            "sha256": segment.sha256,
            "frame_count": segment.frame_count,
            "metadata_sha256": segment.metadata_sha256,
            "provenance_sha256": segment.provenance_sha256,
        }
        for key, actual in checks.items():
            expected = record[key]
            if actual != expected:
                raise ValueError(
                    f"Research source segment mismatch for {key}: "
                    f"expected {expected!r}, found {actual!r}"
                )
        segments.append(segment)

    segments.sort(key=lambda item: (item.session_id, item.segment_index))
    return segments


def replay_raw_feature_buckets(
    manifest_path: str | Path,
    input_root: str | Path,
    *,
    progress_every_frames: int = 1_000_000,
    max_segments: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = Path(manifest_path)
    input_root = Path(input_root)
    manifest = _load_manifest(manifest_path)

    boundaries = pd.read_parquet(
        _artifact_path(manifest_path, manifest, "canonical_boundaries")
    )
    boundaries["boundary_time_utc"] = pd.to_datetime(
        boundaries["boundary_time_utc"], utc=True, errors="raise"
    )
    last_boundary = boundaries["boundary_time_utc"].max()

    replay = _IngestFeatureReplay(product_id=manifest["product_id"])
    segments = _manifest_segments(manifest, input_root=input_root)
    if max_segments is not None:
        segments = segments[:max_segments]

    expected_frame_index = 0

    for segment_number, segment in enumerate(segments, start=1):
        if segment.first_ingest_time_utc > last_boundary.to_pydatetime():
            break

        for raw, connection_id, frame_index, ingest_time_utc in _iter_segment_frames(
            segment,
            expected_frame_index=expected_frame_index,
        ):
            expected_frame_index = frame_index + 1

            if ingest_time_utc <= last_boundary.to_pydatetime():
                replay.consume(
                    raw=raw,
                    connection_id=connection_id,
                    frame_index=frame_index,
                    ingest_time_utc=ingest_time_utc,
                )

            if (
                progress_every_frames > 0
                and replay.frames_seen > 0
                and replay.frames_seen % progress_every_frames == 0
            ):
                print(
                    "feature_replay_progress "
                    f"frames={replay.frames_seen} "
                    f"segments={segment_number} "
                    f"buckets={len(replay.buckets)}",
                    flush=True,
                )

    rows = []
    for bucket_end, bucket in sorted(replay.buckets.items()):
        rows.append(
            {
                "bucket_end": pd.Timestamp(bucket_end),
                **{
                    field: getattr(bucket, field)
                    for field in _BucketStats.__dataclass_fields__
                },
            }
        )

    bucket_frame = pd.DataFrame(rows)
    if not bucket_frame.empty:
        bucket_frame["bucket_end"] = pd.to_datetime(bucket_frame["bucket_end"], utc=True)
        bucket_frame = bucket_frame.sort_values("bucket_end").set_index("bucket_end")

    diagnostics = {
        "research_dataset_id": manifest["dataset_id"],
        "product_id": manifest["product_id"],
        "frames_seen": replay.frames_seen,
        "parsed_frames": replay.parsed_frames,
        "malformed_frames": replay.malformed_frames,
        "stale_frames": replay.stale_frames,
        "sequence_gaps": replay.sequence_gaps,
        "applied_l2_envelopes": replay.applied_l2_envelopes,
        "bucket_count": len(replay.buckets),
        "expected_next_frame_index": expected_frame_index,
    }
    return bucket_frame, diagnostics


def _window_features(
    buckets: pd.DataFrame,
    *,
    timestamp: pd.Timestamp,
    window_minutes: int,
    current_midpoint: float,
) -> dict[str, float]:
    bucket_count = window_minutes // 15
    expected = pd.date_range(end=timestamp, periods=bucket_count, freq="15min", tz="UTC")
    window = buckets.reindex(expected)

    observed_buckets = int(window["asof_book_synced"].notna().sum())
    state_count = float(window["bbo_state_count"].sum(min_count=1))
    valid_count = float(window["bbo_valid_count"].sum(min_count=1))
    synced_count = float(window["bbo_synced_count"].sum(min_count=1))
    spread_count = float(window["spread_bps_count"].sum(min_count=1))
    spread_sum = float(window["spread_bps_sum"].sum(min_count=1))
    spread_sumsq = float(window["spread_bps_sumsq"].sum(min_count=1))
    imbalance_count = float(window["imbalance_count"].sum(min_count=1))
    imbalance_sum = float(window["imbalance_sum"].sum(min_count=1))
    imbalance_sumsq = float(window["imbalance_sumsq"].sum(min_count=1))

    spread_var = _variance(
        count=spread_count,
        total=spread_sum,
        total_sq=spread_sumsq,
    )
    imbalance_var = _variance(
        count=imbalance_count,
        total=imbalance_sum,
        total_sq=imbalance_sumsq,
    )

    midpoint_min = window["midpoint_min"].min(skipna=True)
    midpoint_max = window["midpoint_max"].max(skipna=True)
    spread_max = window["spread_bps_max"].max(skipna=True)

    l2_updates = float(window["l2_update_count"].sum(min_count=1))
    bid_updates = float(window["bid_update_count"].sum(min_count=1))
    ask_updates = float(window["ask_update_count"].sum(min_count=1))
    deletes = float(window["delete_count"].sum(min_count=1))
    bid_deletes = float(window["bid_delete_count"].sum(min_count=1))
    ask_deletes = float(window["ask_delete_count"].sum(min_count=1))

    return {
        f"bbo_bucket_coverage_{window_minutes}m": float(observed_buckets),
        f"bbo_state_count_{window_minutes}m": state_count,
        f"bbo_valid_fraction_{window_minutes}m": _safe_ratio(valid_count, state_count),
        f"bbo_synced_fraction_{window_minutes}m": _safe_ratio(synced_count, state_count),
        f"bbo_spread_mean_bps_{window_minutes}m": _safe_ratio(spread_sum, spread_count),
        f"bbo_spread_std_bps_{window_minutes}m": (
            math.sqrt(spread_var) if np.isfinite(spread_var) else np.nan
        ),
        f"bbo_spread_max_bps_{window_minutes}m": spread_max,
        f"bbo_imbalance_mean_{window_minutes}m": _safe_ratio(
            imbalance_sum, imbalance_count
        ),
        f"bbo_imbalance_std_{window_minutes}m": (
            math.sqrt(imbalance_var) if np.isfinite(imbalance_var) else np.nan
        ),
        f"bbo_midpoint_min_{window_minutes}m": midpoint_min,
        f"bbo_midpoint_max_{window_minutes}m": midpoint_max,
        f"distance_from_high_{window_minutes}m": (
            current_midpoint / midpoint_max - 1.0
            if np.isfinite(current_midpoint)
            and current_midpoint > 0
            and pd.notna(midpoint_max)
            and midpoint_max > 0
            else np.nan
        ),
        f"distance_from_low_{window_minutes}m": (
            current_midpoint / midpoint_min - 1.0
            if np.isfinite(current_midpoint)
            and current_midpoint > 0
            and pd.notna(midpoint_min)
            and midpoint_min > 0
            else np.nan
        ),
        f"midpoint_change_count_{window_minutes}m": float(
            window["midpoint_change_count"].sum(min_count=1)
        ),
        f"l2_envelope_count_{window_minutes}m": float(
            window["l2_envelope_count"].sum(min_count=1)
        ),
        f"l2_incremental_envelope_count_{window_minutes}m": float(
            window["incremental_envelope_count"].sum(min_count=1)
        ),
        f"l2_update_count_{window_minutes}m": l2_updates,
        f"l2_bid_update_count_{window_minutes}m": bid_updates,
        f"l2_ask_update_count_{window_minutes}m": ask_updates,
        f"l2_update_imbalance_{window_minutes}m": _safe_ratio(
            bid_updates - ask_updates,
            bid_updates + ask_updates,
        ),
        f"l2_delete_count_{window_minutes}m": deletes,
        f"l2_bid_delete_count_{window_minutes}m": bid_deletes,
        f"l2_ask_delete_count_{window_minutes}m": ask_deletes,
        f"l2_delete_imbalance_{window_minutes}m": _safe_ratio(
            bid_deletes - ask_deletes,
            bid_deletes + ask_deletes,
        ),
        f"snapshot_envelope_count_{window_minutes}m": float(
            window["snapshot_envelope_count"].sum(min_count=1)
        ),
    }


def assemble_research_feature_frame(
    manifest_path: str | Path,
    buckets: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble causal predictors with canonical targets.

    Predictor state comes only from information received by timestamp t.
    Canonical source-time values are retained only as audit/target provenance.
    """
    manifest_path = Path(manifest_path)
    manifest = _load_manifest(manifest_path)

    boundaries = pd.read_parquet(
        _artifact_path(manifest_path, manifest, "canonical_boundaries")
    )
    targets = pd.read_parquet(
        _artifact_path(manifest_path, manifest, "direction_targets")
    )

    boundaries["boundary_time_utc"] = pd.to_datetime(
        boundaries["boundary_time_utc"], utc=True, errors="raise"
    )
    targets["timestamp"] = pd.to_datetime(
        targets["timestamp"], utc=True, errors="raise"
    )

    boundaries = boundaries.sort_values(
        "boundary_time_utc"
    ).reset_index(drop=True)

    targets = targets.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    if boundaries["boundary_time_utc"].duplicated().any():
        raise ValueError("Canonical boundary timestamps are not unique")

    if targets["timestamp"].duplicated().any():
        raise ValueError("Target timestamps are not unique")

    buckets = buckets.copy()

    if "bucket_end" in buckets.columns:
        buckets["bucket_end"] = pd.to_datetime(
            buckets["bucket_end"], utc=True, errors="raise"
        )
        buckets = buckets.set_index("bucket_end")

    if not isinstance(buckets.index, pd.DatetimeIndex):
        raise ValueError("Feature buckets require a DatetimeIndex")

    if buckets.index.tz is None:
        buckets.index = buckets.index.tz_localize("UTC")
    else:
        buckets.index = buckets.index.tz_convert("UTC")

    buckets = buckets.sort_index()

    first_boundary = boundaries["boundary_time_utc"].min()
    full_history_start = first_boundary + pd.Timedelta(minutes=240)

    rows: list[dict[str, Any]] = []

    for boundary in boundaries.itertuples(index=False):
        timestamp = boundary.boundary_time_utc

        canonical_midpoint = (
            float(boundary.midpoint)
            if pd.notna(boundary.midpoint)
            else np.nan
        )

        canonical_spread = (
            float(boundary.spread)
            if pd.notna(boundary.spread)
            else np.nan
        )

        if timestamp in buckets.index:
            causal = buckets.loc[timestamp]

            asof_book_synced = bool(causal["asof_book_synced"])

            asof_midpoint = (
                float(causal["asof_midpoint"])
                if pd.notna(causal["asof_midpoint"])
                else np.nan
            )

            asof_spread_bps = (
                float(causal["asof_spread_bps"])
                if pd.notna(causal["asof_spread_bps"])
                else np.nan
            )

            asof_bid_size = (
                float(causal["asof_best_bid_size"])
                if pd.notna(causal["asof_best_bid_size"])
                else np.nan
            )

            asof_ask_size = (
                float(causal["asof_best_ask_size"])
                if pd.notna(causal["asof_best_ask_size"])
                else np.nan
            )

            asof_imbalance = (
                float(causal["asof_top_book_imbalance"])
                if pd.notna(causal["asof_top_book_imbalance"])
                else np.nan
            )
        else:
            asof_book_synced = False
            asof_midpoint = np.nan
            asof_spread_bps = np.nan
            asof_bid_size = np.nan
            asof_ask_size = np.nan
            asof_imbalance = np.nan

        row: dict[str, Any] = {
            "timestamp": timestamp,
            "asof_book_synced": asof_book_synced,
            "asof_midpoint": asof_midpoint,
            "asof_spread_bps": asof_spread_bps,
            "asof_best_bid_size": asof_bid_size,
            "asof_best_ask_size": asof_ask_size,
            "asof_top_book_imbalance": asof_imbalance,
            "canonical_midpoint": canonical_midpoint,
            "canonical_spread_bps": (
                canonical_spread / canonical_midpoint * 10_000.0
                if np.isfinite(canonical_spread)
                and np.isfinite(canonical_midpoint)
                and canonical_midpoint > 0
                else np.nan
            ),
            "canonical_target_eligible": bool(
                boundary.canonical_target_eligible
            ),
            "canonical_eligibility_reason": boundary.eligibility_reason,
        }

        for minutes in WINDOWS_MINUTES:
            lag_time = timestamp - pd.Timedelta(minutes=minutes)
            lag_midpoint = np.nan

            if lag_time in buckets.index:
                value = buckets.at[lag_time, "asof_midpoint"]

                if pd.notna(value) and float(value) > 0:
                    lag_midpoint = float(value)

            row[f"momentum_log_return_{minutes}m"] = (
                float(np.log(asof_midpoint / lag_midpoint))
                if np.isfinite(asof_midpoint)
                and asof_midpoint > 0
                and np.isfinite(lag_midpoint)
                and lag_midpoint > 0
                else np.nan
            )

            row.update(
                _window_features(
                    buckets,
                    timestamp=timestamp,
                    window_minutes=minutes,
                    current_midpoint=asof_midpoint,
                )
            )

        row["hour_utc"] = timestamp.hour
        row["quarter_hour_slot"] = timestamp.minute // 15
        row["day_of_week"] = timestamp.dayofweek
        row["is_weekend"] = int(timestamp.dayofweek >= 5)

        row["feature_ready"] = bool(
            timestamp >= full_history_start
            and row["asof_book_synced"]
            and row["bbo_bucket_coverage_240m"] == 16
            and np.isfinite(row["asof_midpoint"])
            and np.isfinite(row["momentum_log_return_240m"])
        )

        rows.append(row)

    features = pd.DataFrame(rows)

    target_columns = [
        "timestamp",
        "target",
        "target_time",
        "label_available_time",
        "target_eligible",
        "target_eligibility_reason",
        "price_change",
        "log_return_15m",
    ]

    target_subset = targets[target_columns].rename(
        columns={"log_return_15m": "target_log_return_15m"}
    )

    features = features.merge(
        target_subset,
        on="timestamp",
        how="left",
        validate="one_to_one",
    )

    features["trainable"] = (
        features["feature_ready"]
        & features["target_eligible"]
    )

    return features.sort_values("timestamp").reset_index(drop=True)


def build_raw_research_feature_frame(
    manifest_path: str | Path,
    input_root: str | Path,
    *,
    progress_every_frames: int = 1_000_000,
    bucket_output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    buckets, diagnostics = replay_raw_feature_buckets(
        manifest_path,
        input_root,
        progress_every_frames=progress_every_frames,
    )

    if bucket_output_path is not None:
        bucket_output_path = Path(bucket_output_path)
        bucket_output_path.parent.mkdir(parents=True, exist_ok=True)

        buckets.reset_index().to_parquet(
            bucket_output_path,
            index=False,
        )

        diagnostics["bucket_artifact_path"] = str(
            bucket_output_path
        )

    features = assemble_research_feature_frame(
        manifest_path,
        buckets,
    )

    diagnostics.update(
        {
            "feature_row_count": int(len(features)),
            "feature_ready_count": int(
                features["feature_ready"].sum()
            ),
            "trainable_count": int(
                features["trainable"].sum()
            ),
            "first_feature_timestamp": (
                features.loc[
                    features["feature_ready"],
                    "timestamp",
                ].min()
                if features["feature_ready"].any()
                else None
            ),
            "last_feature_timestamp": features["timestamp"].max(),
        }
    )

    return features, diagnostics
