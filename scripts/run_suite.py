"""Run the whole test suite in ONE pytest invocation and report where the time goes.

Why: the suite used to run as three sequential invocations at -n 8 on a
14-core box, so cores sat idle between and inside them. One invocation keeps
every worker busy until the end.

Usage (from the repo root):
    python scripts/run_suite.py                 # every tests/test_*.py, -n 12
    python scripts/run_suite.py -n 10           # fewer workers
    python scripts/run_suite.py --report-only   # re-read the last timing file
    python scripts/run_suite.py tests/test_x.py tests/test_y.py   # a subset

Writes the JUnit XML (per-test durations) to ``$SUITE_OUT`` (default
``/tmp/ghost_suite``) and prints the slowest files and tests. ``--dist
loadfile`` keeps each file on one worker (module fixtures and the order
assumptions some files make), so the wall time is bounded below by the
slowest FILE — that list is the one to read.
"""
from __future__ import annotations

import argparse
import collections
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _report(xml_path: Path, top: int) -> None:
    root = ET.parse(xml_path).getroot()
    per_file = collections.Counter()
    per_test = []
    for tc in root.iter("testcase"):
        t = float(tc.get("time") or 0)
        cls = tc.get("classname") or ""
        # "tests.test_x" or "tests.test_x.TestClass" → tests/test_x.py
        parts = cls.split(".")
        mod = ".".join(parts[:2]) if len(parts) >= 2 else cls
        f = mod.replace(".", "/") + ".py"
        per_file[f] += t
        per_test.append((t, f"{f}::{tc.get('name')}"))
    total = sum(per_file.values())
    print(f"\n── timing ({len(per_test)} tests, {total:.0f} s of test time summed over workers) ──")
    print(f"slowest {top} files (a file runs on ONE worker — these bound the wall time):")
    for f, t in per_file.most_common(top):
        print(f"  {t:8.1f} s  {100 * t / max(total, 1e-9):5.1f}%  {f}")
    print(f"slowest {top} tests:")
    for t, name in sorted(per_test, reverse=True)[:top]:
        print(f"  {t:8.1f} s  {name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=12, help="xdist workers (default 12 of 14 cores)")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("files", nargs="*")
    a = ap.parse_args()
    out = Path(os.environ.get("SUITE_OUT", "/tmp/ghost_suite"))
    out.mkdir(parents=True, exist_ok=True)
    xml = out / "junit.xml"
    if a.report_only:
        _report(xml, a.top)
        return 0
    files = a.files or sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tests").glob("test_*.py"))
    # a shell may export GHOST_API_KEY blank (seen live): the interface
    # refuses to import with an empty key, so a blank one is replaced
    env = dict(os.environ, HF_HUB_OFFLINE="1",
               GHOST_API_KEY=(os.environ.get("GHOST_API_KEY") or "").strip() or "x",
               PYTHONPATH="src:.")
    env.pop("FORCE_COLOR", None)
    cmd = [PY, "-m", "pytest", *files, "-n", str(a.n), "--dist", "loadfile", "-q",
           "-p", "no:cacheprovider", f"--junitxml={xml}", "-o", "junit_family=xunit1"]
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=ROOT, env=env).returncode
    print(f"\nwall time: {time.time() - t0:.0f} s, exit {rc}")
    if xml.exists():
        _report(xml, a.top)
    return rc


if __name__ == "__main__":
    sys.exit(main())
