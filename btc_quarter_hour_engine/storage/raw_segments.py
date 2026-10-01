from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from btc_quarter_hour_engine.storage.forward_schema import (
    COINBASE_WS_RAW_SEGMENT_SCHEMA_VERSION,
    DATA_KIND_WEBSOCKET_SEGMENTS,
    FORWARD_SOURCE,
)
from btc_quarter_hour_engine.storage.raw import ImmutableRawStore


@dataclass(slots=True)
class _BufferedFrame:
    frame_index: int
    raw_frame_sha256: str
    message_type: str
    connection_id: str | None
    sequence_num: int | None
    ingest_time_utc: datetime

    def provenance(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "connection_id": self.connection_id,
            "ingest_time_utc": self.ingest_time_utc.isoformat().replace("+00:00", "Z"),
            "sequence_num": self.sequence_num,
            "raw_frame_sha256": self.raw_frame_sha256,
            "message_type": self.message_type,
        }


class RawSegmentWriter:
    """Durably append exact websocket frames and atomically seal immutable segments."""

    def __init__(
        self,
        root: str | Path,
        *,
        source: str = FORWARD_SOURCE,
        product_id: str = "BTC-USD",
        session_id: str | None = None,
        max_frames: int = 10000,
    ) -> None:
        if max_frames < 1:
            raise ValueError("max_frames must be >= 1")
        self.root = Path(root)
        self.source = source
        self.product_id = product_id
        self.session_id = session_id or str(uuid4())
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        self.max_frames = max_frames
        self.raw_store = ImmutableRawStore(root)
        self.sealed_segments: list[dict[str, Any]] = []
        self._frames: list[_BufferedFrame] = []
        self._segment_index = 0
        self._next_frame_index = 0
        session_key = hashlib.sha256(self.session_id.encode("utf-8")).hexdigest()[:24]
        self._segment_dir = self.raw_store.raw_root / source / DATA_KIND_WEBSOCKET_SEGMENTS / product_id
        self._segment_dir.mkdir(parents=True, exist_ok=True)
        self._partial_path = self._segment_dir / f".{session_key}.{self._segment_index:08d}.partial"

    @property
    def pending_frame_count(self) -> int:
        return len(self._frames)

    @property
    def active_partial_path(self) -> Path:
        return self._partial_path

    def bind_session(self, session_id: str) -> None:
        if self._frames or self.sealed_segments:
            raise ValueError("Cannot change the session after receiving frames")
        if not session_id:
            raise ValueError("session_id must not be empty")
        self.session_id = session_id
        session_key = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:24]
        self._partial_path = self._segment_dir / f".{session_key}.{self._segment_index:08d}.partial"
        if self._partial_path.exists():
            raise FileExistsError(f"An incomplete raw segment already exists: {self._partial_path}")

    def add_frame(
        self,
        *,
        raw: str | bytes,
        message_type: str,
        connection_id: str | None = None,
        sequence_num: int | None = None,
        received_at_utc: datetime | None = None,
        ingest_time_utc: datetime | None = None,
    ) -> dict[str, Any] | None:
        if received_at_utc is not None and ingest_time_utc is not None:
            if received_at_utc != ingest_time_utc:
                raise ValueError("received_at_utc and ingest_time_utc must match when both are supplied")
        timestamp = ingest_time_utc or received_at_utc or datetime.now(timezone.utc)
        if not self._frames and self._partial_path.exists():
            raise FileExistsError(f"An incomplete raw segment already exists: {self._partial_path}")
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("ingest_time_utc must be timezone-aware")
        payload = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
        if sequence_num is None:
            sequence_num = self._parse_sequence_num(payload)
        metadata = _BufferedFrame(
            frame_index=self._next_frame_index,
            raw_frame_sha256=hashlib.sha256(payload).hexdigest(),
            message_type=message_type,
            connection_id=connection_id,
            sequence_num=sequence_num,
            ingest_time_utc=timestamp.astimezone(timezone.utc),
        )
        self._append_frame(payload)
        self._frames.append(metadata)
        self._next_frame_index += 1
        if len(self._frames) >= self.max_frames:
            return self.seal()
        return None

    def _append_frame(self, payload: bytes) -> None:
        with self._partial_path.open("ab") as handle:
            handle.write(len(payload).to_bytes(8, "big", signed=False))
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _parse_sequence_num(payload: bytes) -> int | None:
        try:
            value = json.loads(payload).get("sequence_num")
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
            return None
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    def seal(self) -> dict[str, Any] | None:
        if not self._frames:
            return None
        frames = self._frames
        body = self._partial_path.read_bytes()
        first, last = frames[0], frames[-1]
        connection_ids = {frame.connection_id for frame in frames}
        segment_connection_id = next(iter(connection_ids)) if len(connection_ids) == 1 else None
        sequence_nums = [frame.sequence_num for frame in frames if frame.sequence_num is not None]
        frame_provenance = [frame.provenance() for frame in frames]
        artifact = self.raw_store.write_response(
            source=self.source,
            data_kind=DATA_KIND_WEBSOCKET_SEGMENTS,
            product_id=self.product_id,
            response_bytes=body,
            retrieved_at=last.ingest_time_utc,
            request_metadata={
                "session_id": self.session_id,
                "connection_id": segment_connection_id,
                "segment_index": self._segment_index,
                "raw_segment_schema_version": COINBASE_WS_RAW_SEGMENT_SCHEMA_VERSION,
                "frame_count": len(frames),
                "frames": frame_provenance,
                "message_types": sorted({frame.message_type for frame in frames}),
                "first_sequence_num": sequence_nums[0] if sequence_nums else None,
                "last_sequence_num": sequence_nums[-1] if sequence_nums else None,
                "first_ingest_time_utc": first.ingest_time_utc.isoformat().replace("+00:00", "Z"),
                "last_ingest_time_utc": last.ingest_time_utc.isoformat().replace("+00:00", "Z"),
                "sealed": True,
            },
        )
        record = {
            "path": str(artifact.path),
            "sha256": artifact.sha256,
            "byte_count": artifact.byte_count,
            "session_id": self.session_id,
            "connection_id": segment_connection_id,
            "segment_index": self._segment_index,
            "raw_segment_schema_version": COINBASE_WS_RAW_SEGMENT_SCHEMA_VERSION,
            "frame_count": len(frames),
            "frames": frame_provenance,
            "first_sequence_num": sequence_nums[0] if sequence_nums else None,
            "last_sequence_num": sequence_nums[-1] if sequence_nums else None,
            "first_ingest_time_utc": first.ingest_time_utc.isoformat().replace("+00:00", "Z"),
            "last_ingest_time_utc": last.ingest_time_utc.isoformat().replace("+00:00", "Z"),
            # Kept for existing storage consumers; the canonical names above
            # use ingest-time terminology.
            "first_received_at_utc": first.ingest_time_utc.isoformat().replace("+00:00", "Z"),
            "last_received_at_utc": last.ingest_time_utc.isoformat().replace("+00:00", "Z"),
        }
        self.sealed_segments.append(record)
        self._partial_path.unlink()
        self._frames = []
        self._segment_index += 1
        session_key = hashlib.sha256(self.session_id.encode("utf-8")).hexdigest()[:24]
        self._partial_path = self._segment_dir / f".{session_key}.{self._segment_index:08d}.partial"
        return record

    def read_segment_frames(self, path: str | Path) -> list[bytes]:
        """Split a sealed segment into its exact original frame bytes."""
        payload = self.raw_store.read_response(path)
        frames: list[bytes] = []
        offset = 0
        while offset < len(payload):
            if offset + 8 > len(payload):
                raise ValueError(f"Corrupt sealed segment framing at offset {offset}: {path}")
            length = int.from_bytes(payload[offset : offset + 8], "big", signed=False)
            offset += 8
            if offset + length > len(payload):
                raise ValueError(f"Corrupt sealed segment framing at offset {offset}: {path}")
            frames.append(payload[offset : offset + length])
            offset += length
        return frames


__all__ = ["RawSegmentWriter"]
