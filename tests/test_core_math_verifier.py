"""The mathematical verifier must RUN, and must not be quietly gutted.

`scripts/verify_core_math.py` proves properties of the computational core
(bio-scaling arithmetic, egress classification, F8 eviction ordering, the
ablation delta metric) at EXHAUSTIVE / DIFFERENTIAL / PROPERTY rigour.

Why it needs a test of its own: this project's dominant defect class is the
*silent inoperative subsystem* — a guard or instrument that everyone believes
is running and isn't. A verification harness that nobody executes is the purest
instance of that class, and it fails safe-looking: it reports nothing, so it
never contradicts anyone.

These tests therefore assert two separate things:
  1. the harness exits 0 on the current tree, and
  2. it actually performed a substantial number of checks — so deleting
     sections, or skipping them into silence, breaks the build instead of
     quietly shrinking the proof.

Mutation-tested 2026-08-09, 15 deliberate defects injected into production code
one at a time — ALL 15 caught:

  core (7): 6to4/Teredo unblocked, scale-guard floor zeroed, _bio_cooldown
    de-synced, F8 eviction reverted to pure age, F8 credit writer disabled,
    delta metric reverted to absolute counts, resolve_egress_proxy fail-open.
  confidence sequence (8): sigma denominator n+1->n, sigma prior v0 dropped,
    radius inner sqrt dropped, radius n^2->n, alpha inverted, NaN guard
    removed, _MIN_VERDICT_N 30->2, verdict zero-straddle made strict.

Two of the first seven (the scale-guard floor and the F8 credit writer) were
MISSED by earlier drafts and the harness was hardened until they weren't — see
the comments at those checks.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "verify_core_math.py"

# The count the harness reached when this test was written, WITHOUT a live
# store (section 5 skips; it contributes 5 more against real data). Checks may
# be ADDED freely; losing them is the regression worth failing on.
MIN_CHECKS = 21


def _run(tmp_home):
    env = {
        "PYTHONPATH": str(REPO / "src"),
        "PATH": "/usr/bin:/bin:/usr/sbin",
        "HOME": str(tmp_home),
        "GHOST_HOME": str(tmp_home),
    }
    return subprocess.run([sys.executable, str(SCRIPT)], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=600)


def test_verifier_exists():
    assert SCRIPT.exists(), "the core-math verifier was deleted"


def test_verifier_passes_on_an_isolated_home(tmp_path):
    r = _run(tmp_path)
    assert r.returncode == 0, (
        f"core-math verification FAILED:\n{r.stdout[-4000:]}\n{r.stderr[-2000:]}")


def test_verifier_actually_ran_its_checks(tmp_path):
    """Guard against a harness that passes because it does nothing."""
    r = _run(tmp_path)
    passed = [ln for ln in r.stdout.splitlines() if ln.strip().startswith("PASS")]
    assert len(passed) >= MIN_CHECKS, (
        f"only {len(passed)} checks ran (expected >= {MIN_CHECKS}) — the "
        f"verifier has been gutted or is silently skipping sections:\n{r.stdout}")


def test_all_three_rigour_levels_are_represented(tmp_path):
    """EXHAUSTIVE, DIFFERENTIAL and PROPERTY each catch a different class of
    error. Losing a whole level silently narrows what can be proven."""
    r = _run(tmp_path)
    for level in ("EXHAUSTIVE", "DIFFERENTIAL", "PROPERTY"):
        assert level in r.stdout, f"no {level} checks ran at all"


def test_skips_are_announced_not_silent(tmp_path):
    """With no live store, section 5 must SKIP visibly — never vanish."""
    r = _run(tmp_path)
    assert "SKIP" in r.stdout and "no live store" in r.stdout, (
        "the live-numbers section disappeared without saying so:\n" + r.stdout)
    assert "skipped:" in r.stdout, "the summary line hides the skip"


@pytest.mark.parametrize("marker", [
    "_MIN_USABLE_WINDOW_S",   # the scale-guard floor
    "_credit_surfaced",       # F8 credit writer
    "resolve_egress_proxy",   # the egress fail-closed backstop
    "asymp_cs_radius",        # the verdict instrument
    "_MIN_VERDICT_N",         # the gate that makes it trustworthy
])
def test_verifier_still_covers_each_pinned_subsystem(marker):
    """If a subsystem is renamed, the check that covers it must be updated
    rather than left matching nothing."""
    assert marker in SCRIPT.read_text(), (
        f"{marker} is no longer referenced by the verifier — coverage lost")
