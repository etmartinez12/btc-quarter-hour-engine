from __future__ import annotations

"""Central schema/version constants for forward (live) Coinbase market-data
collection artifacts.

Every forward collector, storage helper, and replay/test consumer imports
these constants rather than hard-coding literal strings, so a schema or
version bump happens in exactly one place and is guaranteed to stay
consistent across raw segment metadata, normalized Parquet artifacts, and
forward manifests.
"""

FORWARD_SOURCE = "coinbase_advanced"

# Normalized Parquet `data_kind` partitions written by the forward collector.
DATA_KIND_QUARTER_HOUR_BBO = "quarter_hour_bbo"
DATA_KIND_LEVEL2_UPDATES = "level2_updates"
DATA_KIND_BBO_STATE = "bbo_state"

# Raw-segment `data_kind` values used by the immutable raw store.
DATA_KIND_WEBSOCKET_SEGMENTS = "websocket_segments"
DATA_KIND_WEBSOCKET_FRAMES = "websocket_frames"

# Independently versioned so each normalized artifact's shape can evolve on
# its own schedule without forcing a version bump of unrelated artifacts.
FORWARD_BBO_SCHEMA_VERSION = "1"
FORWARD_LEVEL2_UPDATES_SCHEMA_VERSION = "1"
FORWARD_BBO_STATE_SCHEMA_VERSION = "1"

FORWARD_MANIFEST_PURPOSE = "forward_bbo_collection"

__all__ = [
    "FORWARD_SOURCE",
    "DATA_KIND_QUARTER_HOUR_BBO",
    "DATA_KIND_LEVEL2_UPDATES",
    "DATA_KIND_BBO_STATE",
    "DATA_KIND_WEBSOCKET_SEGMENTS",
    "DATA_KIND_WEBSOCKET_FRAMES",
    "FORWARD_BBO_SCHEMA_VERSION",
    "FORWARD_LEVEL2_UPDATES_SCHEMA_VERSION",
    "FORWARD_BBO_STATE_SCHEMA_VERSION",
    "FORWARD_MANIFEST_PURPOSE",
]
