from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SYSTEMD_DIR = REPOSITORY_ROOT / "deploy" / "systemd"


def test_research_materializer_service_contract() -> None:
    service_path = SYSTEMD_DIR / "btc-qh-research-materializer.service"
    service = service_path.read_text(encoding="utf-8")

    required_settings = (
        "Type=oneshot",
        "User=martinez_elias12",
        "Group=martinez_elias12",
        "WorkingDirectory=/home/martinez_elias12/btc-quarter-hour-engine",
        "Environment=PYTHONUNBUFFERED=1",
        "Environment=PYTHONDONTWRITEBYTECODE=1",
        "ConditionPathExists=/var/lib/btc-qh/research_materializer_prod/checkpoint.json",
        "ExecStart=/usr/bin/flock --nonblock --conflict-exit-code 75 "
        "/var/lib/btc-qh/research_materializer.lock "
        "/home/martinez_elias12/.venv/bin/python -m "
        "btc_quarter_hour_engine.research.materializer_cli "
        "--input-root /var/lib/btc-qh/data_lake_prod "
        "--output-root /var/lib/btc-qh/research_materializer_prod "
        "--product BTC-USD --max-new-segments 100 "
        "--checkpoint-every-segments 25",
        "SuccessExitStatus=75",
        "TimeoutStartSec=infinity",
        "SyslogIdentifier=btc-qh-research-materializer",
        "Nice=10",
        "CPUWeight=25",
        "IOWeight=25",
        "NoNewPrivileges=true",
        "PrivateTmp=true",
        "ProtectSystem=full",
        "ReadOnlyPaths=/var/lib/btc-qh/data_lake_prod",
    )

    for setting in required_settings:
        assert setting in service

    assert "btc-qh-materialize-research" not in service
    assert "Restart=always" not in service
    assert "CPUQuota=" not in service
    assert "MemoryMax=" not in service


def test_research_materializer_timer_contract() -> None:
    timer = (SYSTEMD_DIR / "btc-qh-research-materializer.timer").read_text(
        encoding="utf-8"
    )

    assert "OnCalendar=*-*-* *:10:00 UTC" in timer
    assert "Persistent=true" in timer
    assert "Unit=btc-qh-research-materializer.service" in timer


def test_unit_files_do_not_activate_the_timer_automatically() -> None:
    unit_files = tuple(SYSTEMD_DIR.glob("*"))

    assert unit_files
    assert all(path.is_file() for path in unit_files)
    assert all(
        "systemctl enable" not in path.read_text(encoding="utf-8")
        and "systemctl start" not in path.read_text(encoding="utf-8")
        for path in unit_files
    )
