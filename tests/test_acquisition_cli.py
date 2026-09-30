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
