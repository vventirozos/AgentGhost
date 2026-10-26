"""Run the whole test suite in ONE pytest invocation and report where the time goes.

Why: the suite used to run as three sequential invocations at -n 8 on a
14-core box, so cores sat idle between and inside them. One invocation keeps
every worker busy until the end.

Usage (from the repo root):
    python scripts/run_suite.py                 # every tests/test_*.py, -n 8
    python scripts/run_suite.py -n 12           # more workers (swaps the Mac)
    python scripts/run_suite.py --report-only   # re-read the last timing file
    python scripts/run_suite.py tests/test_x.py tests/test_y.py   # a subset

Writes the JUnit XML (per-test durations) to ``$SUITE_OUT`` (default
``/tmp/ghost_suite``) and prints the slowest files and tests. ``--dist
loadfile`` keeps each file on one worker (module fixtures and the order
assumptions some files make), so the wall time is bounded below by the
slowest FILE — that list is the one to read.

Files are passed slowest-first, by the per-file times of the last full run
(``TIMES``, written after every run). xdist hands ``loadfile`` scopes out in
collection order, which follows the argument order; alphabetical order put
a 190 s file late in the queue and left one worker running it alone at the
end. Longest-first keeps the tail near the slowest single file. Files with
no recorded time go first.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
TIMES = ROOT / ".suite_file_times.json"


def _per_file(xml_path: Path):
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
    return per_file, per_test


def _longest_first(files):
    try:
        times = json.loads(TIMES.read_text())
    except (OSError, ValueError):
        return files
    return sorted(files, key=lambda f: -times.get(f, float("inf")))


def _save_times(per_file) -> None:
    try:
        TIMES.write_text(json.dumps({f: round(t, 2) for f, t in per_file.items()}, indent=0))
    except OSError:
        pass


def _report(xml_path: Path, top: int) -> None:
    per_file, per_test = _per_file(xml_path)
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
    ap.add_argument("-n", type=int, default=8, help="xdist workers (default 8: 12 swapped the Mac, operator 2026-10-26)")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--no-lint", action="store_true",
                    help="skip the pylint gate that runs after the tests (~70 s alone)")
    ap.add_argument("files", nargs="*")
    a = ap.parse_args()
    out = Path(os.environ.get("SUITE_OUT", "/tmp/ghost_suite"))
    out.mkdir(parents=True, exist_ok=True)
    xml = out / "junit.xml"
    if a.report_only:
        _report(xml, a.top)
        return 0
    files = a.files or _longest_first(
        sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tests").glob("test_*.py")))
    # a shell may export GHOST_API_KEY blank (seen live): the interface
    # refuses to import with an empty key, so a blank one is replaced
    env = dict(os.environ, HF_HUB_OFFLINE="1",
               GHOST_API_KEY=(os.environ.get("GHOST_API_KEY") or "").strip() or "x",
               PYTHONPATH="src:.")
    env.pop("FORCE_COLOR", None)
    cmd = [PY, "-m", "pytest", *files, "-n", str(a.n), "--dist", "loadfile", "-q",
           "-p", "no:cacheprovider", f"--junitxml={xml}", "-o", "junit_family=xunit1"]
    t0 = time.time()
    rc = pytest_rc = subprocess.run(cmd, cwd=ROOT, env=env).returncode
    pytest_s = time.time() - t0
    print(f"\nwall time: {pytest_s:.0f} s, exit {rc}")
    # The pylint gate runs AFTER pytest, alone: inside the suite its 8 jobs
    # shared the cores with 12 workers and took 436 s (71 s alone, 2026-10-26),
    # so test_lint_gate's three gate tests skip unless GHOST_LINT_GATE=1.
    # Same runner and verdict as ci.sh. Skipped for a subset or --no-lint.
    if not a.files and not a.no_lint:
        t1 = time.time()
        lrc = subprocess.run([PY, "scripts/lint.py"], cwd=ROOT, env=env).returncode
        lint = f"lint gate: {'OK' if lrc == 0 else 'FAILED'} in {time.time() - t1:.0f} s, exit {lrc}"
        rc = rc or lrc
    else:
        lint = "lint gate NOT run: python scripts/lint.py"
    if xml.exists():
        if not a.files:     # a subset's times would drop every other file
            _save_times(_per_file(xml)[0])
        _report(xml, a.top)
    print(f"\ntests: exit {pytest_rc}, wall {pytest_s:.0f} s · {lint}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
