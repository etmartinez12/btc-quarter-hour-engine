from __future__ import annotations

import gzip
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


@dataclass(frozen=True, slots=True)
class RawArtifact:
    source: str
    data_kind: str
    product_id: str
    sha256: str
    byte_count: int
    path: Path
    metadata_path: Path
    retrieved_at: datetime


class ImmutableRawStore:
    """Store immutable source payloads keyed by the SHA-256 of their exact bytes."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.raw_root = self.root / "raw"

    def write_response(
        self,
        *,
        source: str,
        data_kind: str,
        product_id: str,
        request_metadata: dict[str, Any],
        response_bytes: bytes,
        retrieved_at: datetime,
    ) -> RawArtifact:
        if not isinstance(response_bytes, (bytes, bytearray)):
            raise TypeError("response_bytes must be bytes-like")
        payload = bytes(response_bytes)
        digest = hashlib.sha256(payload).hexdigest()
        target_dir = self.raw_root / source / data_kind / product_id
        target_dir.mkdir(parents=True, exist_ok=True)

        compressed_path = target_dir / f"{digest}.json.gz"
        metadata_path = target_dir / f"{digest}.meta.json"

        if compressed_path.exists():
            existing_bytes = self.read_response(compressed_path)
            if existing_bytes != payload:
                raise ValueError(f"Content hash collision for {compressed_path}: different bytes already exist.")
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("sha256") != digest:
                    raise ValueError(f"Metadata hash mismatch for {metadata_path}")
            return RawArtifact(
                source=source,
                data_kind=data_kind,
                product_id=product_id,
                sha256=digest,
                byte_count=len(payload),
                path=compressed_path,
                metadata_path=metadata_path,
                retrieved_at=retrieved_at,
            )

        compressed_payload = gzip.compress(payload, compresslevel=9, mtime=0)
        self._atomic_write_bytes(compressed_path, compressed_payload)

        metadata = {
            "source": source,
            "data_kind": data_kind,
            "product_id": product_id,
            "request_metadata": request_metadata,
            "retrieved_at_utc": retrieved_at.astimezone().isoformat() if retrieved_at.tzinfo else retrieved_at.isoformat(),
            "sha256": digest,
            "byte_count": len(payload),
            "raw_artifact_path": str(compressed_path),
        }
        self._atomic_write_bytes(metadata_path, json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))

        return RawArtifact(
            source=source,
            data_kind=data_kind,
            product_id=product_id,
            sha256=digest,
            byte_count=len(payload),
            path=compressed_path,
            metadata_path=metadata_path,
            retrieved_at=retrieved_at,
        )

    def read_response(self, path: str | Path) -> bytes:
        file_path = Path(path)
        with gzip.open(file_path, "rb") as handle:
            return handle.read()

    def _atomic_write_bytes(self, path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=str(path.parent), prefix=f".{path.name}.", delete=False) as tmp_file:
            tmp_filename = Path(tmp_file.name)
            tmp_file.write(payload)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_filename, path)

    def verify_digest(self, path: str | Path, expected_sha256: str) -> bool:
        computed = hashlib.sha256(self.read_response(path)).hexdigest()
        return computed == expected_sha256
