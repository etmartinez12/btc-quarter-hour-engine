from datetime import datetime, timezone

import pandas as pd
import pyarrow.parquet as pq
import pytest

from btc_quarter_hour_engine.storage.parquet import NormalizedParquetStore
from btc_quarter_hour_engine.storage.forward_parquet import ForwardParquetStore


def frame(close=101.0):
    return pd.DataFrame({
        "source": ["coinbase_advanced"] * 2, "product_id": ["BTC-USD"] * 2,
        "granularity": ["ONE_MINUTE"] * 2,
        "bucket_start": [
            datetime(2026, 1, 1, 23, 59, tzinfo=timezone.utc),
            datetime(2026, 1, 2, 0, 0, tzinfo=timezone.utc),
        ],
        "open": [100., 100.], "high": [102., 102.], "low": [99., 99.],
        "close": [close] * 2, "volume": [0.5, 0.8],
        "retrieved_at_utc": [datetime(2026, 1, 3, tzinfo=timezone.utc)] * 2,
    })


def test_daily_parquet_roundtrip_immutable_and_atomic(tmp_path):
    store = NormalizedParquetStore(tmp_path)
    artifacts = store.write_dataframe(
        dataframe=frame(), source="coinbase_advanced", product_id="BTC-USD",
        data_kind="candles", granularity="ONE_MINUTE", schema_version="1",
    )
    assert len(artifacts) == 2
    assert {part["row_count"] for part in artifacts} == {1}
    assert "date=2026-01-01" in artifacts[0]["path"]
    assert "date=2026-01-02" in artifacts[1]["path"]
    for artifact in artifacts:
        table = pq.ParquetFile(artifact["path"]).read()
        assert table.schema.metadata[b"schema_version"] == b"1"
        row = table.to_pandas()
        assert list(row.columns) == list(frame().columns)
        assert row["bucket_start"].dt.tz is not None
        assert row["close"].iloc[0] == 101
        assert artifact["first_timestamp"] == artifact["last_timestamp"]
        assert artifact["sha256"]
    assert store.write_dataframe(dataframe=frame(), source="coinbase_advanced", product_id="BTC-USD",
                                 data_kind="candles", granularity="ONE_MINUTE") == artifacts
    different = store.write_dataframe(
        dataframe=frame(101.5), source="coinbase_advanced", product_id="BTC-USD",
        data_kind="candles", granularity="ONE_MINUTE",
    )
    assert {a["path"] for a in artifacts}.isdisjoint(a["path"] for a in different)
    assert not list(tmp_path.rglob(".part-*"))


def test_naive_parquet_timestamp_rejected(tmp_path):
    data = frame()
    data["bucket_start"] = [datetime(2026, 1, 1, 23, 59), datetime(2026, 1, 2)]
    with pytest.raises(ValueError, match="timezone-aware"):
        NormalizedParquetStore(tmp_path).write_dataframe(
            dataframe=data, source="coinbase_advanced", product_id="BTC-USD", data_kind="candles",
        )


@pytest.mark.parametrize("timestamp_field", ["event_time_utc", "state_time_utc", "boundary_time_utc"])
def test_explicit_timestamp_field_is_preserved_and_partitions_by_that_field(tmp_path, timestamp_field):
    data = pd.DataFrame({
        timestamp_field: [
            datetime(2026, 1, 1, 23, 59, tzinfo=timezone.utc),
            datetime(2026, 1, 2, 0, 0, tzinfo=timezone.utc),
        ],
        "value": [1, 2],
    })
    artifacts = NormalizedParquetStore(tmp_path).write_dataframe(
        dataframe=data,
        source="coinbase_advanced",
        product_id="BTC-USD",
        data_kind="quarter_hour_bbo",
        timestamp_field=timestamp_field,
    )
    assert len(artifacts) == 2
    recovered = pd.concat([pd.read_parquet(artifact["path"]) for artifact in artifacts])
    assert timestamp_field in recovered.columns
    assert "source_time_utc" not in recovered.columns


def test_forward_store_forwards_explicit_timestamp_field(tmp_path):
    rows = [{
        "event_time_utc": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "sequence_num": 1,
    }]
    artifact = ForwardParquetStore(tmp_path).write_rows(
        rows=rows,
        source="coinbase_advanced",
        product_id="BTC-USD",
        data_kind="level2_updates",
        timestamp_field="event_time_utc",
    )[0]
    recovered = pd.read_parquet(artifact["path"])
    assert "event_time_utc" in recovered.columns
    assert "source_time_utc" not in recovered.columns
