from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from btc_quarter_hour_engine.storage.manifest import (
    COINBASE_BOOK_SNAPSHOT_SCHEMA_VERSION,
    COINBASE_CANDLE_SCHEMA_VERSION,
    build_manifest,
    write_manifest,
)
from btc_quarter_hour_engine.storage.parquet import NormalizedParquetStore
from btc_quarter_hour_engine.storage.raw import ImmutableRawStore

from .chunking import _SUPPORTED_GRANULARITIES, _coerce_utc, _normalize_granularity, build_candle_chunk_plan
from .coinbase_rest import (
    CoinbaseHTTPResult,
    CoinbasePublicRESTClient,
    coverage_diagnostics_for_candles,
    parse_coinbase_candle_records,
    parse_product_book_response,
    validate_product_id,
    validate_candle_records,
)

SOURCE = "coinbase_advanced"
CANDLE_INELIGIBILITY = (
    "Historical OHLCV candles do not reproduce the exact best-bid/ask midpoint "
    "used by the canonical quarter-hour target."
)
BOOK_INELIGIBILITY = (
    "REST product-book snapshots are connectivity and schema validation only; "
    "they are not exact quarter-hour canonical targets."
)
BOOK_PURPOSE = "connectivity_and_schema_validation"


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    manifest: dict
    manifest_path: Path
    run_metadata: dict


def _iso(value: datetime) -> str:
    return _coerce_utc(value).isoformat().replace("+00:00", "Z")


def _persist_manifest(
    manifest: dict, output_root: str | Path, *, run_metadata: dict,
) -> AcquisitionResult:
    manifest_path = write_manifest(manifest, output_root)
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    return AcquisitionResult(persisted, manifest_path, run_metadata)


def _store_raw(
    *, response: CoinbaseHTTPResult, raw_store: ImmutableRawStore,
    data_kind: str, product_id: str, extra: dict,
):
    return raw_store.write_response(
        source=SOURCE,
        data_kind=data_kind,
        product_id=product_id,
        response_bytes=response.response_bytes,
        retrieved_at=response.retrieved_at_utc,
        request_metadata={
            "request_path": response.request_path,
            "request_params": response.request_params,
            "http_status_code": response.status_code,
            "content_type": response.content_type,
            **extra,
        },
    )


def acquire_coinbase_candles(
    *,
    client: CoinbasePublicRESTClient,
    raw_store: ImmutableRawStore,
    normalized_store: NormalizedParquetStore,
    output_root: str | Path,
    product_id: str,
    start: datetime,
    end: datetime,
    granularity: str = "ONE_MINUTE",
    max_candles_per_request: int = 300,
) -> AcquisitionResult:
    validate_product_id(product_id)
    start_utc, end_utc = _coerce_utc(start), _coerce_utc(end)
    granularity = _normalize_granularity(granularity)
    if max_candles_per_request > 300:
        raise ValueError("Coinbase public candles permit at most 300 buckets per request")
    chunks = build_candle_chunk_plan(
        start=start_utc, end=end_utc, granularity=granularity,
        max_candles_per_request=max_candles_per_request,
    )
    started = datetime.now(timezone.utc)
    raw_artifacts = []
    records = []
    step = _SUPPORTED_GRANULARITIES[granularity]
    for chunk in chunks:
        try:
            response = client.fetch_candles(
                product_id, chunk.requested_start, chunk.requested_end, granularity,
            )
            artifact = _store_raw(
                response=response, raw_store=raw_store, data_kind="candles", product_id=product_id,
                extra={
                    "chunk_number": chunk.chunk_number,
                    "requested_start": _iso(chunk.requested_start),
                    "requested_end": _iso(chunk.requested_end),
                    "granularity": granularity,
                },
            )
            chunk_records = validate_candle_records(parse_coinbase_candle_records(response.json_payload))
            for record in chunk_records:
                bucket = record["bucket_start"]
                if not chunk.requested_start <= bucket < chunk.requested_end or bucket.timestamp() % step:
                    raise ValueError(f"Candle {bucket.isoformat()} outside aligned chunk interval")
                records.append({
                    "source": SOURCE, "product_id": product_id, "granularity": granularity,
                    **record, "retrieved_at_utc": artifact.retrieved_at,
                })
            raw_artifacts.append({
                "chunk_number": chunk.chunk_number,
                "requested_start": _iso(chunk.requested_start),
                "requested_end": _iso(chunk.requested_end),
                "endpoint": response.request_path,
                "request_params": response.request_params,
                "http_status_code": response.status_code,
                "content_type": response.content_type,
                "retrieved_at_utc": _iso(response.retrieved_at_utc),
                "path": str(artifact.path),
                "sha256": artifact.sha256,
                "byte_count": artifact.byte_count,
                "records_received": len(chunk_records),
                "first_returned_timestamp": _iso(chunk_records[0]["bucket_start"]) if chunk_records else None,
                "last_returned_timestamp": _iso(chunk_records[-1]["bucket_start"]) if chunk_records else None,
            })
        except Exception as exc:
            raise RuntimeError(
                f"Candle chunk {chunk.chunk_number} failed for "
                f"[{_iso(chunk.requested_start)}, {_iso(chunk.requested_end)}): {exc}"
            ) from exc

    validate_candle_records(records)
    records.sort(key=lambda record: record["bucket_start"])
    coverage = coverage_diagnostics_for_candles(
        records=records, start=start_utc, end=end_utc, granularity=granularity,
    )
    if not records:
        raise ValueError(
            f"No usable candles in [{_iso(start_utc)}, {_iso(end_utc)}); "
            f"coverage_fraction={coverage['coverage_fraction']}"
        )
    normalized_artifacts = normalized_store.write_dataframe(
        dataframe=pd.DataFrame(records, columns=[
            "source", "product_id", "granularity", "bucket_start",
            "open", "high", "low", "close", "volume", "retrieved_at_utc",
        ]),
        source=SOURCE, product_id=product_id, data_kind="candles",
        granularity=granularity, schema_version=COINBASE_CANDLE_SCHEMA_VERSION,
    )
    manifest = build_manifest(
        source=SOURCE, product_id=product_id, data_kind="candles",
        requested_start=start_utc, requested_end=end_utc, granularity=granularity,
        raw_artifacts=raw_artifacts, normalized_artifacts=normalized_artifacts,
        coverage=coverage, canonical_target_eligible=False,
        canonical_target_ineligibility_reason=CANDLE_INELIGIBILITY,
        acquisition_started_at_utc=started, acquisition_completed_at_utc=datetime.now(timezone.utc),
        request_count=len(chunks), schema_version=COINBASE_CANDLE_SCHEMA_VERSION,
        purpose="historical_context_only",
    )
    return _persist_manifest(
        manifest, output_root,
        run_metadata={
            "acquisition_started_at_utc": manifest["acquisition_started_at_utc"],
            "acquisition_completed_at_utc": manifest["acquisition_completed_at_utc"],
            "raw_responses": raw_artifacts,
        },
    )


def acquire_coinbase_book_snapshot(
    *,
    client: CoinbasePublicRESTClient,
    raw_store: ImmutableRawStore,
    normalized_store: NormalizedParquetStore,
    output_root: str | Path,
    product_id: str,
) -> AcquisitionResult:
    validate_product_id(product_id)
    started = datetime.now(timezone.utc)
    response = client.fetch_product_book(product_id)
    artifact = _store_raw(
        response=response, raw_store=raw_store, data_kind="product_book",
        product_id=product_id, extra={},
    )
    book = parse_product_book_response(response.json_payload)
    if book["product_id"] != product_id:
        raise ValueError(f"Book product {book['product_id']!r} differs from requested {product_id!r}")
    row = {
        "source": SOURCE, **book, "retrieved_at_utc": artifact.retrieved_at,
        "canonical_target_eligible": False, "purpose": BOOK_PURPOSE,
    }
    normalized_artifacts = normalized_store.write_dataframe(
        dataframe=pd.DataFrame([row]), source=SOURCE, product_id=product_id,
        data_kind="product_book", schema_version=COINBASE_BOOK_SNAPSHOT_SCHEMA_VERSION,
    )
    manifest = build_manifest(
        source=SOURCE, product_id=product_id, data_kind="product_book",
        requested_start=None, requested_end=None, granularity=None,
        raw_artifacts=[{
            "path": str(artifact.path), "sha256": artifact.sha256,
            "byte_count": artifact.byte_count, "endpoint": response.request_path,
            "request_params": response.request_params,
            "http_status_code": response.status_code, "content_type": response.content_type,
            "retrieved_at_utc": _iso(response.retrieved_at_utc),
            "source_time_utc": _iso(book["source_time_utc"]),
        }],
        normalized_artifacts=normalized_artifacts,
        coverage={"observed_snapshot_count": 1},
        canonical_target_eligible=False,
        canonical_target_ineligibility_reason=BOOK_INELIGIBILITY,
        acquisition_started_at_utc=started, acquisition_completed_at_utc=datetime.now(timezone.utc),
        request_count=1, schema_version=COINBASE_BOOK_SNAPSHOT_SCHEMA_VERSION, purpose=BOOK_PURPOSE,
    )
    return _persist_manifest(
        manifest, output_root,
        run_metadata={
            "acquisition_started_at_utc": manifest["acquisition_started_at_utc"],
            "acquisition_completed_at_utc": manifest["acquisition_completed_at_utc"],
            "raw_responses": manifest["raw_artifacts"],
        },
    )
