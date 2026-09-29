from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class NormalizedParquetStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.normalized_root = self.root / "normalized"

    def write_dataframe(
        self,
        *,
        dataframe: pd.DataFrame,
        source: str,
        product_id: str,
        data_kind: str,
        granularity: str | None = None,
        schema_version: str = "1",
    ) -> dict:
        dataframe = dataframe.copy()
        row_count = len(dataframe)
        first_timestamp = None
        last_timestamp = None
        if "bucket_start" in dataframe.columns:
            timestamps = pd.to_datetime(dataframe["bucket_start"], utc=True)
            if not timestamps.empty:
                first_timestamp = timestamps.min().isoformat()
                last_timestamp = timestamps.max().isoformat()
        elif "retrieved_at_utc" in dataframe.columns:
            timestamps = pd.to_datetime(dataframe["retrieved_at_utc"], utc=True)
            if not timestamps.empty:
                first_timestamp = timestamps.min().isoformat()
                last_timestamp = timestamps.max().isoformat()

        partition_dir = self.normalized_root / source / f"product_id={product_id}" / f"data_kind={data_kind}"
        if granularity is not None:
            partition_dir = partition_dir / f"granularity={str(granularity).upper()}"

        if first_timestamp is not None:
            date_value = pd.to_datetime(first_timestamp, utc=True).strftime("%Y-%m-%d")
            partition_dir = partition_dir / f"date={date_value}"

        partition_dir.mkdir(parents=True, exist_ok=True)
        target_path = partition_dir / "part-00000.parquet"

        table = pa.Table.from_pandas(dataframe, preserve_index=False)
        pq.write_table(table, target_path)

        digest = hashlib.sha256(target_path.read_bytes()).hexdigest()
        return {
            "path": str(target_path),
            "sha256": digest,
            "row_count": row_count,
            "first_timestamp": first_timestamp,
            "last_timestamp": last_timestamp,
            "schema_version": schema_version,
            "product_id": product_id,
            "source": source,
            "data_kind": data_kind,
        }
