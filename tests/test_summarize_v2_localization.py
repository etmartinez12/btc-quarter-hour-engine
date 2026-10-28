import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "research" / "summarize_v2_localization.py"
ARTIFACTS = (
    REPO / "research_artifacts" / "2026-10-28-v2-signal-localization"
)
REPORT_RELATIVE_PATH = (
    Path("docs")
    / "research"
    / "v2-signal-localization-findings-2026-10-28.md"
)


def _copy_run_tree(tmp_path):
    repo = tmp_path / "repo"
    script_path = repo / "scripts" / "research" / SCRIPT.name
    script_path.parent.mkdir(parents=True)
    shutil.copy2(SCRIPT, script_path)

    artifact_path = (
        repo / "research_artifacts" / "2026-10-28-v2-signal-localization"
    )
    shutil.copytree(ARTIFACTS, artifact_path)

    report_path = repo / REPORT_RELATIVE_PATH
    report_path.parent.mkdir(parents=True)
    report_path.write_bytes(b"existing report must remain unchanged\n")
    return repo, artifact_path, report_path, script_path


def _run_optimized(repo, script_path):
    return subprocess.run(
        [sys.executable, "-O", str(script_path)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


def test_corrupt_evidence_is_rejected_with_optimized_python(tmp_path):
    repo, artifacts, report_path, script_path = _copy_run_tree(tmp_path)
    evidence_path = artifacts / "metrics.parquet"
    corrupted = bytearray(evidence_path.read_bytes())
    corrupted[-1] ^= 0x01
    evidence_path.write_bytes(corrupted)
    original_report = report_path.read_bytes()

    result = _run_optimized(repo, script_path)

    assert result.returncode != 0
    assert "Artifact checksum mismatch: metrics.parquet" in result.stderr
    assert report_path.read_bytes() == original_report


def test_research_validation_failure_preserves_existing_report(tmp_path):
    repo, artifacts, report_path, script_path = _copy_run_tree(tmp_path)
    fold_metrics_path = artifacts / "fold_metrics.parquet"
    fold_metrics = pd.read_parquet(fold_metrics_path)
    selected = (
        fold_metrics["mode"].eq("group_only")
        & fold_metrics["family"].eq("range_volatility")
        & fold_metrics["benchmark"].eq("extra_trees")
    )
    fold_metrics.loc[selected, "accuracy"] = 0.0
    fold_metrics.to_parquet(fold_metrics_path, index=False)

    manifest_path = artifacts / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_sha256"]["fold_metrics.parquet"] = hashlib.sha256(
        fold_metrics_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    original_report = report_path.read_bytes()

    result = _run_optimized(repo, script_path)

    assert result.returncode != 0
    assert "Unexpected range-only versus full V2 fold findings" in result.stderr
    assert report_path.read_bytes() == original_report
