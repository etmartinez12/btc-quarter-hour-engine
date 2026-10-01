from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from btc_quarter_hour_engine.storage.forward_schema import (
    COINBASE_BBO_STATE_SCHEMA_VERSION,
    COINBASE_BOUNDARY_BBO_SCHEMA_VERSION,
    COINBASE_L2_UPDATE_SCHEMA_VERSION,
    DATA_KIND_BBO_STATE,
    DATA_KIND_LEVEL2_UPDATES,
    DATA_KIND_QUARTER_HOUR_BBO,
)
from btc_quarter_hour_engine.storage.parquet import NormalizedParquetStore


class ForwardParquetStore:
    def __init__(self, root: str):
        self.root = root
        self.store = NormalizedParquetStore(root)

    def write_rows(
        self,
        *,
        rows: Iterable[dict],
        source: str,
        product_id: str,
        data_kind: str,
        schema_version: str | None = None,
        timestamp_field: str | None = None,
    ) -> list[dict]:
        records = list(rows)
        if not records:
            raise ValueError("Forward collection requires at least one row")
        frame = pd.DataFrame(records)
        resolved_timestamp_field = timestamp_field or "source_time_utc"
        if timestamp_field is None and resolved_timestamp_field not in frame.columns:
            for alias in ("event_time_utc", "timestamp_utc"):
                if alias in frame.columns:
                    frame = frame.rename(columns={alias: resolved_timestamp_field})
                    break
        if resolved_timestamp_field not in frame.columns:
            raise ValueError(
                f"Forward collection rows require a {resolved_timestamp_field!r} timestamp field"
            )
        frame[resolved_timestamp_field] = pd.to_datetime(frame[resolved_timestamp_field], utc=True)
        resolved_schema_version = schema_version or {
            DATA_KIND_QUARTER_HOUR_BBO: COINBASE_BOUNDARY_BBO_SCHEMA_VERSION,
            DATA_KIND_LEVEL2_UPDATES: COINBASE_L2_UPDATE_SCHEMA_VERSION,
            DATA_KIND_BBO_STATE: COINBASE_BBO_STATE_SCHEMA_VERSION,
        }.get(data_kind, COINBASE_BOUNDARY_BBO_SCHEMA_VERSION)
        return self.store.write_dataframe(
            dataframe=frame,
            source=source,
            product_id=product_id,
            data_kind=data_kind,
            schema_version=resolved_schema_version,
            timestamp_field=resolved_timestamp_field,
        )


__all__ = ["ForwardParquetStore"]
