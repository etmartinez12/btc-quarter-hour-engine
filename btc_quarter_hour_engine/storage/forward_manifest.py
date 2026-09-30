from __future__ import annotations

from datetime import datetime
from typing import Any

from btc_quarter_hour_engine.storage.manifest import build_manifest


def build_forward_manifest(
    *,
    source: str,
    product_id: str,
    data_kind: str,
    raw_artifacts: list[dict[str, Any]],
    normalized_artifacts: list[dict[str, Any]],
    coverage: dict[str, Any],
    canonical_target_eligible: bool,
    canonical_target_ineligibility_reason: str | None,
    acquisition_started_at_utc: datetime | None,
    acquisition_completed_at_utc: datetime | None,
    request_count: int,
    schema_version: str,
    purpose: str | None = None,
) -> dict[str, Any]:
    return build_manifest(
        source=source,
        product_id=product_id,
        data_kind=data_kind,
        requested_start=None,
        requested_end=None,
        granularity=None,
        raw_artifacts=raw_artifacts,
        normalized_artifacts=normalized_artifacts,
        coverage=coverage,
        canonical_target_eligible=canonical_target_eligible,
        canonical_target_ineligibility_reason=canonical_target_ineligibility_reason,
        acquisition_started_at_utc=acquisition_started_at_utc,
        acquisition_completed_at_utc=acquisition_completed_at_utc,
        request_count=request_count,
        schema_version=schema_version,
        purpose=purpose,
    )


__all__ = ["build_forward_manifest"]
