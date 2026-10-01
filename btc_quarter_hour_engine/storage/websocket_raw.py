from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from btc_quarter_hour_engine.storage.forward_schema import (
    DATA_KIND_WEBSOCKET_FRAMES,
    FORWARD_SOURCE,
)
from btc_quarter_hour_engine.storage.raw import ImmutableRawStore
from btc_quarter_hour_engine.storage.raw_segments import RawSegmentWriter


class WebSocketRawStore:
    """Persist raw websocket frames and associated metadata in the same immutable raw store."""

    def __init__(self, root: str, *, source: str = FORWARD_SOURCE) -> None:
        self.root = root
        self.source = source
        self.raw_store = ImmutableRawStore(root)

    def write_frame(
        self,
        *,
        frame: str | bytes,
        product_id: str,
        message_type: str,
        connection_id: str | None = None,
        **metadata: Any,
    ) -> dict[str, Any]:
        if isinstance(frame, bytes):
            payload = frame
        else:
            payload = frame.encode("utf-8")
        recorded_at = datetime.now(timezone.utc)
        artifact = self.raw_store.write_response(
            source=self.source,
            data_kind=DATA_KIND_WEBSOCKET_FRAMES,
            product_id=product_id,
            response_bytes=payload,
            retrieved_at=recorded_at,
            request_metadata={
                "message_type": message_type,
                "connection_id": connection_id,
                "timestamp_utc": recorded_at.isoformat().replace("+00:00", "Z"),
                **metadata,
            },
        )
        return {
            "path": str(artifact.path),
            "sha256": artifact.sha256,
            "byte_count": artifact.byte_count,
            "retrieved_at_utc": artifact.retrieved_at.isoformat().replace("+00:00", "Z"),
        }

    def write_message(self, *, message: dict[str, Any], product_id: str, connection_id: str | None = None) -> dict[str, Any]:
        return self.write_frame(
            frame=json.dumps(message, separators=(",", ":"), sort_keys=True),
            product_id=product_id,
            message_type=str(message.get("type", "unknown")),
            connection_id=connection_id,
            message=message,
        )

__all__ = ["WebSocketRawStore", "RawSegmentWriter"]
