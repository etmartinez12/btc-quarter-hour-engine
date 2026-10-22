from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any


def _utc_to_checkpoint(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None or value.utcoffset().total_seconds() != 0:
        raise ValueError("Checkpoint timestamps must be timezone-aware UTC")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_from_checkpoint(value: Any, *, nullable: bool = False) -> datetime | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str):
        raise ValueError("Checkpoint timestamp must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Malformed checkpoint timestamp: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Checkpoint timestamps must be timezone-aware UTC")
    if parsed.utcoffset().total_seconds() != 0:
        raise ValueError("Checkpoint timestamps must be UTC")
    return parsed.astimezone(timezone.utc)


def _float_to_checkpoint(value: float) -> float | dict[str, str]:
    numeric = float(value)
    if math.isnan(numeric):
        return {"__float__": "nan"}
    if math.isinf(numeric):
        return {"__float__": "inf" if numeric > 0 else "-inf"}
    return numeric


def _float_from_checkpoint(value: Any, *, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, dict):
        if set(value) != {"__float__"}:
            raise ValueError("Malformed tagged checkpoint float")
        tag = value["__float__"]
        if tag == "nan":
            return math.nan
        if tag == "inf":
            return math.inf
        if tag == "-inf":
            return -math.inf
        raise ValueError("Unsupported tagged checkpoint float")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Checkpoint value must be a JSON number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("Non-finite checkpoint numbers must use an explicit tag")
    return numeric


def _require_checkpoint(
    checkpoint: Any,
    *,
    version_key: str,
    version: int,
    required: set[str],
) -> dict[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must be a JSON object")
    checkpoint = dict(checkpoint)
    if version_key in checkpoint and (
        type(checkpoint[version_key]) is not int or checkpoint[version_key] != version
    ):
        raise ValueError(f"Unsupported checkpoint schema version: {checkpoint[version_key]!r}")
    if set(checkpoint) != required:
        missing = sorted(required - set(checkpoint))
        extra = sorted(set(checkpoint) - required)
        raise ValueError(f"Malformed checkpoint keys (missing={missing}, extra={extra})")
    return checkpoint


def _require_nonnegative_int(value: Any, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"Checkpoint {field} must be a nonnegative integer")
    return value
