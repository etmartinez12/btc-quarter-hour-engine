from __future__ import annotations

import argparse
from pathlib import Path

from btc_quarter_hour_engine.research.prospective_source import (
    ProspectiveRawFrameJournal,
)

from .prospective_source_bridge import (
    MAX_RAW_FRAME_BYTES,
    ProspectiveSourceReceiver,
    ProspectiveSourceSocketServer,
)


def receive_prospective_source_main() -> None:
    parser = argparse.ArgumentParser(
        description="Receive the optional local collector tee into an isolated source journal."
    )
    parser.add_argument("--socket-path", required=True, help="Local Unix-domain socket path.")
    parser.add_argument(
        "--journal-path",
        required=True,
        help="Separate SQLite journal path; this does not modify production raw storage.",
    )
    parser.add_argument(
        "--max-frame-bytes",
        type=int,
        default=MAX_RAW_FRAME_BYTES,
        help="Must match the existing collector's configured WebSocket message maximum.",
    )
    args = parser.parse_args()
    journal_path = Path(args.journal_path)
    journal = ProspectiveRawFrameJournal(journal_path)
    receiver = ProspectiveSourceReceiver(journal, max_frame_bytes=args.max_frame_bytes)
    server = ProspectiveSourceSocketServer(args.socket_path, receiver)
    try:
        server.serve_forever()
    finally:
        server.stop()
        journal.close()


if __name__ == "__main__":
    receive_prospective_source_main()
