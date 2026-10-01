from pathlib import Path

from btc_quarter_hour_engine.acquisition import cli


def test_candle_cli_prints_operational_summary_from_manifest(monkeypatch, capsys, tmp_path):
    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def close(self):
            pass

    manifest = {
        "dataset_id": "dataset-hash",
        "request_count": 2,
        "normalized_artifacts": [{"row_count": 3}, {"row_count": 2}],
        "requested_start": "2026-01-01T00:00:00Z",
        "requested_end": "2026-01-01T00:05:00Z",
        "coverage": {
            "first_bucket": "2026-01-01T00:00:00+00:00",
            "last_bucket": "2026-01-01T00:04:00+00:00",
            "missing_bucket_count": 1,
            "coverage_fraction": 0.8,
        },
    }
    monkeypatch.setattr(cli, "CoinbasePublicRESTClient", FakeClient)
    monkeypatch.setattr(
        cli, "acquire_coinbase_candles",
        lambda **kwargs: type("Result", (), {"manifest": manifest, "manifest_path": Path(tmp_path) / "manifest.json"})(),
    )
    monkeypatch.setattr("sys.argv", [
        "btc-qh-fetch-coinbase-candles", "--start", "2026-01-01T00:00:00Z",
        "--end", "2026-01-01T00:05:00Z", "--output-root", str(tmp_path),
    ])

    cli.fetch_candles_main()

    output = capsys.readouterr().out
    for expected in (
        "dataset_id=dataset-hash",
        f"manifest_path={tmp_path / 'manifest.json'}",
        "raw_request_count=2",
        "normalized_row_count=5",
        "requested_start=2026-01-01T00:00:00Z",
        "requested_end=2026-01-01T00:05:00Z",
        "observed_start=2026-01-01T00:00:00+00:00",
        "observed_end=2026-01-01T00:04:00+00:00",
        "missing_bucket_count=1",
        "coverage_fraction=0.8",
        "canonical_target_eligible=false",
    ):
        assert expected in output


def test_collect_coinbase_bbo_cli_prints_operational_summary_from_manifest(monkeypatch, capsys, tmp_path):
    manifest = {
        "dataset_id": "ws-dataset-hash", "canonical_target_eligible": True,
        "session_id": "session-1", "product_id": "BTC-USD",
        "session_started_at_utc": "2024-01-01T00:00:00Z",
        "session_completed_at_utc": "2024-01-01T00:15:00Z",
        "termination_reason": "max_messages",
        "connections": [{"level2_messages_received": 7, "heartbeat_messages_received": 3}],
        "integrity": {"sequence_gap_count": 0, "stale_sequence_count": 0, "heartbeat_timeout_count": 0},
        "quarter_hour_summary": {"boundaries_seen": 4, "eligible_boundaries": 3, "ineligible_boundaries": 1},
    }
    fake_result = type(
        "Result",
        (),
        {
            "manifest": manifest,
            "manifest_path": Path(tmp_path) / "manifest.json",
            "raw_segments": [{"path": "segment-1", "frame_count": 10}],
            "observation_count": 4,
            "eligible_observation_count": 3,
            "level2_update_row_count": 7,
            "bbo_state_row_count": 5,
            "connection_count": 1,
            "reconnect_count": 0,
        },
    )()

    class FakeCollector:
        def __init__(self, **kwargs):
            pass

        def run(self, **kwargs):
            return fake_result

    monkeypatch.setattr(cli, "CoinbaseWebSocketClient", lambda **kwargs: object())
    monkeypatch.setattr(cli, "WebsocketsTransport", lambda *a, **kw: object())
    monkeypatch.setattr(cli, "CoinbaseWebSocketService", lambda **kwargs: type("Service", (), {"session_id": "session-1"})())
    monkeypatch.setattr(cli, "RawSegmentWriter", lambda *a, **kw: object())
    monkeypatch.setattr(cli, "ForwardParquetStore", lambda *a, **kw: object())
    monkeypatch.setattr(cli, "WebSocketCollector", FakeCollector)
    monkeypatch.setattr("sys.argv", ["btc-qh-collect-coinbase-bbo", "--output-root", str(tmp_path)])

    cli.collect_coinbase_bbo_main()

    output = capsys.readouterr().out
    for expected in (
        "dataset_id=ws-dataset-hash",
        "session_id=session-1",
        "product_id=BTC-USD",
        "termination_reason=max_messages",
        "quarter_hour_boundaries_seen=4",
        "boundary_rows=4",
        f"manifest_path={tmp_path / 'manifest.json'}",
        "raw_segment_count=1",
        "observation_count=4",
        "eligible_observation_count=3",
        "level2_update_row_count=7",
        "bbo_state_row_count=5",
        "connection_count=1",
        "reconnect_count=0",
        "canonical_target_eligible=true",
    ):
        assert expected in output


def test_run_websocket_collector_main_is_a_backward_compatible_alias():
    assert cli.run_websocket_collector_main is cli.collect_coinbase_bbo_main
