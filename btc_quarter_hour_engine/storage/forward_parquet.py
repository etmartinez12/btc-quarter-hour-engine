from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from btc_quarter_hour_engine.storage.parquet import NormalizedParquetStore


class ForwardParquetStore:
    def __init__(self, root: str):
        self.root = root
        self.store = NormalizedParquetStore(root)

    def write_rows(self, *, rows: Iterable[dict], source: str, product_id: str, data_kind: str, schema_version: str = "1") -> list[dict]:
        records = list(rows)
        if not records:
            raise ValueError("Forward collection requires at least one row")
        frame = pd.DataFrame(records)
        timestamp_field = "source_time_utc"
        if timestamp_field not in frame.columns:
            for alias in ("event_time_utc", "timestamp_utc"):
                if alias in frame.columns:
                    frame = frame.rename(columns={alias: timestamp_field})
                    break
        if timestamp_field not in frame.columns:
            raise ValueError(
                "Forward collection rows require a 'source_time_utc' (or 'event_time_utc'/'timestamp_utc') field"
            )
        frame[timestamp_field] = pd.to_datetime(frame[timestamp_field], utc=True)
        return self.store.write_dataframe(
            dataframe=frame,
            source=source,
            product_id=product_id,
            data_kind=data_kind,
            schema_version=schema_version,
        )


__all__ = ["ForwardParquetStore"]
