from __future__ import annotations

import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

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
    ) -> list[dict]:
        timestamp_field = "bucket_start" if data_kind == "candles" else "source_time_utc"
        if dataframe.empty or timestamp_field not in dataframe:
            raise ValueError(f"Nonempty normalized data requires {timestamp_field}")
        frame = dataframe.copy()
        if any(pd.Timestamp(value).tzinfo is None for value in frame[timestamp_field]):
            raise ValueError(f"{timestamp_field} must be timezone-aware")
        frame[timestamp_field] = pd.to_datetime(frame[timestamp_field], utc=True)
        frame = frame.sort_values(timestamp_field).reset_index(drop=True)
        if data_kind == "candles" and frame[timestamp_field].duplicated().any():
            raise ValueError("Duplicate normalized candle bucket")
        partition_root = self.normalized_root / source / product_id / data_kind
        if granularity is not None:
            partition_root = partition_root / granularity.upper()

        artifacts: list[dict] = []
        for day, partition in frame.groupby(frame[timestamp_field].dt.strftime("%Y-%m-%d"), sort=True):
            destination = partition_root / f"date={day}"
            destination.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=destination, prefix=".part-", suffix=".parquet", delete=False) as handle:
                temporary = Path(handle.name)
            try:
                table = pa.Table.from_pandas(partition.reset_index(drop=True), preserve_index=False)
                table = table.replace_schema_metadata({
                    **(table.schema.metadata or {}),
                    b"schema_version": schema_version.encode("ascii"),
                })
                pq.write_table(table, temporary)
                with temporary.open("rb+") as handle:
                    os.fsync(handle.fileno())
                digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
                target = destination / f"part-{digest}.parquet"
                try:
                    os.link(temporary, target)
                except FileExistsError:
                    if target.read_bytes() != temporary.read_bytes():
                        raise ValueError(f"Normalized artifact conflict: {target}")
                artifacts.append({
                    "path": str(target),
                    "sha256": digest,
                    "row_count": len(partition),
                    "first_timestamp": partition[timestamp_field].min().isoformat(),
                    "last_timestamp": partition[timestamp_field].max().isoformat(),
                    "schema_version": schema_version,
                    "product_id": product_id,
                    "source": source,
                    "data_kind": data_kind,
                })
            finally:
                temporary.unlink(missing_ok=True)
        return artifacts
