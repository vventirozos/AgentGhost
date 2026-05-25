"""Smoke test for scripts/run_selfhood_probes.sh.

Doesn't exercise the probes themselves (they require a running agent +
upstream LLM); just verifies the wrapper:
  * is executable
  * finds the probe scripts
  * writes a summary file to $SELFHOOD_PROBES_DIR
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_selfhood_probes.sh"


def test_probe_wrapper_exists_and_is_executable():
    assert SCRIPT.exists(), f"missing wrapper: {SCRIPT}"
    assert os.access(SCRIPT, os.X_OK), "wrapper must be executable"


def test_probe_wrapper_finds_both_probes():
    # Locate the probe scripts the wrapper references via grep on the
    # wrapper source — a missing reference would be a regression.
    src = SCRIPT.read_text(encoding="utf-8")
    assert "consciousness_probe.py" in src
    assert "introspective_consistency.py" in src


def test_probe_wrapper_writes_summary(tmp_path: Path):
    # Stand up a fake repo with no-op probe scripts so the wrapper
    # exercises its IO + summary plumbing without contacting a live
    # agent. SELFHOOD_REPO points the wrapper at this stub directory;
    # GHOST_HOME isolates the output.
    fake_repo = tmp_path / "fake-repo"
    (fake_repo / "scripts").mkdir(parents=True)
    for name in ("consciousness_probe.py", "introspective_consistency.py"):
        (fake_repo / "scripts" / name).write_text(
            "import sys\nprint('stub probe ok')\nsys.exit(0)\n",
            encoding="utf-8",
        )
    env = os.environ.copy()
    env["GHOST_HOME"] = str(tmp_path)
    env["SELFHOOD_REPO"] = str(fake_repo)
    res = subprocess.run(
        ["/bin/bash", str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    summary_dir = tmp_path / "system" / "selfhood" / "probes"
    assert summary_dir.exists(), res.stdout + res.stderr
    summaries = list(summary_dir.glob("probes-*.txt"))
    assert summaries, "wrapper did not write a summary file"
    body = summaries[0].read_text(encoding="utf-8")
    assert "stub probe ok" in body
    assert res.returncode == 0
