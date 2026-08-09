"""Long-run safety gates: progress contract + pre-flight.

These exist because three separate long runs on this project were wasted or
unreadable, and each failed a precondition that was checkable BEFORE launch:

  * an ~8h ablation whose two arms resolved to BYTE-IDENTICAL flags, so the
    comparison compared a configuration with itself;
  * an ~8h warm-seeded run whose metric counted absolute store contents, so
    all three arms reported the seed's own numbers;
  * a 90-minute bench launched with block-buffered stdout, whose progress was
    then GUESSED four different ways and wrong every time.

The gates are only worth anything if they fail loudly on those exact shapes,
so that is what these tests assert.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

from ghost_agent.eval.runprogress import (  # noqa: E402
    RunProgress,
    read_progress,
    render,
)

PREFLIGHT = REPO / "scripts" / "preflight_longrun.py"
RUNSTATUS = REPO / "scripts" / "runstatus.py"


# ── progress contract ───────────────────────────────────────────────────────

def test_position_is_readable_while_the_run_is_in_flight(tmp_path):
    """THE DEFECT: a 90-minute run whose counter never flushed."""
    f = tmp_path / "p.json"
    p = RunProgress(f, total=10, label="t")
    for _ in range(3):
        p.tick(force=True)
    b = read_progress(f)
    assert b["status"] == "running" and b["done"] == 3 and b["total"] == 10
    assert b["pct"] == 30.0


def test_writes_are_atomic_so_a_reader_never_sees_a_partial_file(tmp_path):
    """⚠ Sequential read-after-write CANNOT detect a torn file — the writer
    has already finished by the time the reader looks. The first version of
    this test did exactly that and passed with `os.replace` removed, i.e. it
    asserted nothing. Caught by revert-testing.

    Two real checks now: a concurrent reader hammering the file while it is
    rewritten, and a structural assertion that the atomic primitive is still
    the one being used (the concurrent check is probabilistic; the
    structural one is not).
    """
    import threading

    f = tmp_path / "p.json"
    p = RunProgress(f, total=500)
    errors = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                if f.exists():
                    json.loads(f.read_text())
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    try:
        for _ in range(300):
            p.tick(force=True)
    finally:
        stop.set()
        t.join(timeout=5)
    assert not errors, f"reader saw a partial file: {errors[:3]}"


def test_the_atomic_primitive_is_still_in_use():
    """Structural, because the concurrent test above is probabilistic and a
    rare tear could slip through a lucky run."""
    import inspect

    from ghost_agent.eval import runprogress
    src = inspect.getsource(runprogress.RunProgress._write)
    assert "os.replace" in src, (
        "progress writes are no longer atomic — a reader can observe a "
        "truncated file and conclude the run is unreadable")
    assert ".tmp" in src, "no temp file: the write is not staged"


def test_missing_file_is_UNKNOWN_not_zero(tmp_path):
    """A status tool that reports 0% for a run it cannot see is lying."""
    b = read_progress(tmp_path / "nope.json")
    assert b["status"] == "missing"
    assert "UNKNOWN" in b["note"] and "not zero" in b["note"]


def test_a_stalled_run_is_reported_as_stalled(tmp_path):
    """Staleness must never read as progress — a wedged run looks identical
    to a healthy one if you only read `done`."""
    import os
    f = tmp_path / "p.json"
    RunProgress(f, total=10).tick(force=True)
    old = time.time() - 3600
    os.utime(f, (old, old))
    b = read_progress(f)
    assert b["status"] == "STALLED"
    assert "do NOT read `done` as current" in b["note"]


def test_unreadable_file_says_so(tmp_path):
    f = tmp_path / "p.json"
    f.write_text("{not json")
    assert read_progress(f)["status"] == "unreadable"


def test_rate_is_measured_over_a_trailing_window(tmp_path):
    """A whole-run average is dragged down by a slow start or by a replayed
    cache prefix that completed in milliseconds; it would not predict what
    the NEXT item costs."""
    f = tmp_path / "p.json"
    p = RunProgress(f, total=100, window=5)
    for _ in range(10):
        p.tick(force=True)
        time.sleep(0.01)
    assert len(p._marks) == 5, "window is not bounded"
    assert p.rate_per_min() > 0


def test_rate_is_absent_rather_than_invented_before_evidence(tmp_path):
    """One completion cannot support a rate, and a made-up ETA is exactly
    what this whole mechanism exists to stop."""
    f = tmp_path / "p.json"
    p = RunProgress(f, total=10)
    p.tick(force=True)
    assert p.rate_per_min() is None
    assert read_progress(f)["rate_per_min"] is None
    assert read_progress(f)["eta_min"] is None


def test_unknown_total_does_not_fabricate_a_percentage(tmp_path):
    f = tmp_path / "p.json"
    p = RunProgress(f, total=None)
    p.tick(force=True)
    b = read_progress(f)
    assert b["pct"] is None and b["eta_min"] is None
    assert "total unknown" in render(f)


def test_context_manager_records_a_failure(tmp_path):
    f = tmp_path / "p.json"
    with pytest.raises(ValueError):
        with RunProgress(f, total=3) as p:
            p.tick(force=True)
            raise ValueError("boom")
    b = read_progress(f)
    assert b["extra"]["finished"] is False and "boom" in b["extra"]["error"]


def test_render_never_reports_a_bare_percentage_for_a_stalled_run(tmp_path):
    import os
    f = tmp_path / "p.json"
    RunProgress(f, total=10).tick(force=True)
    old = time.time() - 3600
    os.utime(f, (old, old))
    out = render(f)
    assert "STALLED" in out and "⚠" in out


# ── runstatus CLI ───────────────────────────────────────────────────────────

def _status(path):
    return subprocess.run([sys.executable, str(RUNSTATUS), str(path)],
                          cwd=REPO, capture_output=True, text=True,
                          timeout=120,
                          env={"PYTHONPATH": str(REPO / "src"),
                               "PATH": "/usr/bin:/bin"})


def test_runstatus_exit_codes_let_a_script_branch(tmp_path):
    f = tmp_path / "p.json"
    assert _status(f).returncode == 2                    # missing
    RunProgress(f, total=5).tick(force=True)
    assert _status(f).returncode == 0                    # running
    import os
    old = time.time() - 3600
    os.utime(f, (old, old))
    assert _status(f).returncode == 3                    # STALLED


# ── pre-flight gate ─────────────────────────────────────────────────────────

def _pre(*args):
    return subprocess.run([sys.executable, str(PREFLIGHT)] + list(args),
                          cwd=REPO, capture_output=True, text=True,
                          timeout=120,
                          env={"PYTHONPATH": str(REPO / "src"),
                               "PATH": "/usr/bin:/bin"})


def test_the_gate_blocks_tonights_actual_launch():
    """THE REGRESSION TEST FOR 2026-08-09: the verifier re-bench as it was
    really launched — resumable, but unobservable, unbounded, unvalidated
    and with no measured rate. It must not clear."""
    r = _pre("--name", "as launched", "--resumable", "response cache")
    assert r.returncode == 1
    for expected in ("OBSERVABLE", "BOUNDED", "DISCRIMINATING", "MEASURED"):
        assert f"[BLOCK] {expected}" in r.stdout


def test_a_fully_prepared_run_clears(tmp_path):
    r = _pre("--name", "prepared",
             "--observable", f"progress-file:{tmp_path}/p.json",
             "--total-from-tool", "464",
             "--resumable", "response cache --cache-mode read",
             "--smoke", "16 trials; arms differ on --frontier-selfplay",
             "--measured-rate", "3.5", "--rate-source", "90s live window")
    assert r.returncode == 0 and "CLEARED FOR LAUNCH" in r.stdout


def test_eta_is_reported_as_a_RANGE_not_a_point(tmp_path):
    """Point ETAs were quoted four times tonight and missed every time."""
    r = _pre("--name", "x", "--observable", "unbuffered",
             "--total-from-tool", "100", "--resumable", "cache",
             "--smoke", "proved", "--measured-rate", "10",
             "--rate-source", "timed slice")
    assert "RANGE" in r.stdout and "-" in r.stdout


@pytest.mark.parametrize("missing,flagname", [
    ("observable", "OBSERVABLE"),
    ("total", "BOUNDED"),
    ("resumable", "RESUMABLE"),
    ("smoke", "DISCRIMINATING"),
    ("rate", "MEASURED"),
])
def test_each_precondition_blocks_on_its_own(tmp_path, missing, flagname):
    a = ["--name", "x", "--observable", "unbuffered",
         "--total-from-tool", "10", "--resumable", "cache",
         "--smoke", "proved", "--measured-rate", "5",
         "--rate-source", "timed"]
    drop = {"observable": ("--observable", "unbuffered"),
            "total": ("--total-from-tool", "10"),
            "resumable": ("--resumable", "cache"),
            "smoke": ("--smoke", "proved"),
            "rate": ("--measured-rate", "5")}[missing]
    i = a.index(drop[0])
    del a[i:i + 2]
    r = _pre(*a)
    assert r.returncode == 1, f"{flagname} did not block on its own"
    assert f"[BLOCK] {flagname}" in r.stdout


def test_an_override_must_state_a_reason_and_is_recorded(tmp_path):
    """An unexplained override is the same as having no gate."""
    r = _pre("--name", "x", "--force", "operator accepts the risk: one-off")
    assert r.returncode == 0
    assert "OVERRIDDEN" in r.stdout and "operator accepts the risk" in r.stdout
    assert "Recorded, not hidden" in r.stdout


def test_a_derived_denominator_cannot_satisfy_BOUNDED():
    """The check exists because a denominator computed by hand (35) instead
    of read from the tool (58) made every progress report wrong."""
    r = _pre("--name", "x", "--observable", "unbuffered",
             "--resumable", "cache", "--smoke", "s",
             "--measured-rate", "1", "--rate-source", "t")
    assert "[BLOCK] BOUNDED" in r.stdout
    assert "PRINTS" in r.stdout and "35 instead of 58" in r.stdout
