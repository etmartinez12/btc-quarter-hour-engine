from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from btc_quarter_hour_engine.storage.raw import ImmutableRawStore


class WebSocketRawStore:
    """Persist raw websocket frames and associated metadata in the same immutable raw store."""

    def __init__(self, root: str, *, source: str = "coinbase_advanced") -> None:
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
            data_kind="websocket_frames",
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


@dataclass(slots=True)
class _BufferedFrame:
    raw: bytes
    message_type: str
    connection_id: str | None
    sequence_num: int | None
    received_at_utc: datetime


class RawSegmentWriter:
    """Accumulate exact raw websocket frame bytes and seal them into immutable segments.

    Frames are appended to an in-memory buffer preserving their *exact* bytes
    (no re-serialization, no lossy text normalization). When the buffer
    reaches ``max_frames`` (or :meth:`seal` is called explicitly, e.g. on
    shutdown or reconnect), the buffered frames are concatenated using a
    simple length-prefixed framing (8-byte big-endian length + payload,
    repeated) and written once through :class:`ImmutableRawStore`, which
    content-addresses and immutably persists the sealed segment. Sealing is
    the unit of "forward manifest raw artifact": each sealed segment becomes
    one manifest raw-artifact entry with full provenance.
    """

    def __init__(
        self,
        root: str,
        *,
        source: str = "coinbase_advanced",
        product_id: str = "BTC-USD",
        max_frames: int = 10000,
    ) -> None:
        if max_frames < 1:
            raise ValueError("max_frames must be >= 1")
        self.root = root
        self.source = source
        self.product_id = product_id
        self.max_frames = max_frames
        self.raw_store = ImmutableRawStore(root)
        self.sealed_segments: list[dict[str, Any]] = []
        self._buffer: list[_BufferedFrame] = []
        self._segment_index = 0

    @property
    def pending_frame_count(self) -> int:
        return len(self._buffer)

    def add_frame(
        self,
        *,
        raw: str | bytes,
        message_type: str,
        connection_id: str | None = None,
        sequence_num: int | None = None,
        received_at_utc: datetime | None = None,
    ) -> dict[str, Any] | None:
        payload = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
        timestamp = received_at_utc or datetime.now(timezone.utc)
        if timestamp.tzinfo is None:
            raise ValueError("received_at_utc must be timezone-aware")
        self._buffer.append(
            _BufferedFrame(
                raw=payload,
                message_type=message_type,
                connection_id=connection_id,
                sequence_num=sequence_num,
                received_at_utc=timestamp.astimezone(timezone.utc),
            )
        )
        if len(self._buffer) >= self.max_frames:
            return self.seal()
        return None

    def seal(self) -> dict[str, Any] | None:
        if not self._buffer:
            return None
        frames = self._buffer
        self._buffer = []
        body = b"".join(len(frame.raw).to_bytes(8, "big", signed=False) + frame.raw for frame in frames)
        first, last = frames[0], frames[-1]
        sequence_nums = [frame.sequence_num for frame in frames if frame.sequence_num is not None]
        artifact = self.raw_store.write_response(
            source=self.source,
            data_kind="websocket_segments",
            product_id=self.product_id,
            response_bytes=body,
            retrieved_at=last.received_at_utc,
            request_metadata={
                "segment_index": self._segment_index,
                "frame_count": len(frames),
                "connection_id": first.connection_id,
                "message_types": sorted({frame.message_type for frame in frames}),
                "first_sequence_num": sequence_nums[0] if sequence_nums else None,
                "last_sequence_num": sequence_nums[-1] if sequence_nums else None,
                "first_received_at_utc": first.received_at_utc.isoformat().replace("+00:00", "Z"),
                "last_received_at_utc": last.received_at_utc.isoformat().replace("+00:00", "Z"),
                "sealed": True,
            },
        )
        record = {
            "path": str(artifact.path),
            "sha256": artifact.sha256,
            "byte_count": artifact.byte_count,
            "segment_index": self._segment_index,
            "frame_count": len(frames),
            "connection_id": first.connection_id,
            "first_sequence_num": sequence_nums[0] if sequence_nums else None,
            "last_sequence_num": sequence_nums[-1] if sequence_nums else None,
            "first_received_at_utc": first.received_at_utc.isoformat().replace("+00:00", "Z"),
            "last_received_at_utc": last.received_at_utc.isoformat().replace("+00:00", "Z"),
        }
        self.sealed_segments.append(record)
        self._segment_index += 1
        return record

    def read_segment_frames(self, path: str | Path) -> list[bytes]:
        """Deterministically split a sealed segment back into its exact original frame bytes."""
        payload = self.raw_store.read_response(path)
        frames: list[bytes] = []
        offset = 0
        total = len(payload)
        while offset < total:
            if offset + 8 > total:
                raise ValueError(f"Corrupt sealed segment framing at offset {offset}: {path}")
            length = int.from_bytes(payload[offset : offset + 8], "big", signed=False)
            offset += 8
            if offset + length > total:
                raise ValueError(f"Corrupt sealed segment framing at offset {offset}: {path}")
            frames.append(payload[offset : offset + length])
            offset += length
        return frames


__all__ = ["WebSocketRawStore", "RawSegmentWriter"]
