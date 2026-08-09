"""Paired bench comparison: the arithmetic, and the refusal.

Two independent things are pinned here.

THE ARITHMETIC. McNemar's exact test is what makes a small real improvement
decidable at this bench's size: the absolute balanced score carries a ±0.05
95% half-width, so treating two runs as independent proportions can only
resolve changes larger than ~0.1 — bigger than almost any real change. Paired,
only the trials that CHANGED carry information.

THE REFUSAL. The 2026-08-04 baseline was measured on the WORKER route and
production moved to CRITIC two days later; the numbers were compared anyway
because nothing checked. The tool now diffs provenance FIRST and refuses,
unless the difference is declared with --expect-differs.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import verify_bench_compare as VBC  # noqa: E402

SCRIPT = REPO / "scripts" / "verify_bench_compare.py"


# ── McNemar exact ───────────────────────────────────────────────────────────

def test_no_discordant_pairs_is_p_one():
    assert VBC.mcnemar_exact(0, 0) == 1.0


def test_symmetric_discordance_is_not_significant():
    assert VBC.mcnemar_exact(5, 5) == pytest.approx(1.0)


def test_known_values_against_the_binomial():
    """DIFFERENTIAL: against scipy's definition, computed by hand here.

    b=0,c=6 -> two-sided p = 2 * 0.5^6 = 0.03125
    b=1,c=9 -> two-sided p = 2 * (C(10,0)+C(10,1)) * 0.5^10 = 0.021484375
    """
    assert VBC.mcnemar_exact(0, 6) == pytest.approx(2 * 0.5 ** 6)
    assert VBC.mcnemar_exact(1, 9) == pytest.approx(2 * 11 * 0.5 ** 10)


def test_p_is_symmetric_in_its_arguments():
    """The test is two-sided: which side improved cannot change the p-value."""
    for b, c in ((0, 6), (2, 9), (4, 11)):
        assert VBC.mcnemar_exact(b, c) == pytest.approx(VBC.mcnemar_exact(c, b))


def test_p_never_exceeds_one():
    for b in range(6):
        for c in range(6):
            assert 0.0 <= VBC.mcnemar_exact(b, c) <= 1.0


def test_more_lopsided_evidence_gives_a_smaller_p():
    assert VBC.mcnemar_exact(0, 10) < VBC.mcnemar_exact(2, 8) < VBC.mcnemar_exact(4, 6)


def test_concordant_trials_do_not_affect_the_result():
    """The whole point of pairing: 1000 unchanged trials must not dilute the
    evidence carried by the ones that moved."""
    assert VBC.mcnemar_exact(0, 6) == VBC.mcnemar_exact(0, 6)  # b,c only inputs


# ── scoring ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fault,verdict,ok", [
    ("clean", "CONFIRMED", True),
    ("clean", "REFUTED", False),
    ("evidence_truncation", "UNCERTAIN", True),   # NOT_REFUTED is the bar
    ("evidence_truncation", "REFUTED", False),
    ("fabrication", "REFUTED", True),
    ("fabrication", "CONFIRMED", False),
    ("fact_swap", None, False),                   # unparseable is not a pass
])
def test_correctness_follows_the_fault_class(fault, verdict, ok):
    assert VBC._correct({"fault": fault, "verdict": verdict}) is ok


# ── the refusal ─────────────────────────────────────────────────────────────

def _report(tmp, name, *, leg="critic", code="aaa", trials=(), started="t0"):
    r = {
        "started_utc": started,
        "provenance": {
            "cases_sha256": "pool", "faults_sha256": "faults",
            "code": {"verifier": code, "bench": "bbb"},
            "judge": {"base_url": "http://j", "model": ""},
            "escalation": {"arm": "judge+escalation", "cheap_route": leg},
            "templates": {}, "verify_flags": {},
            "cache": {"measures": "live judge"},
        },
        "arms": {"two_stage_on": {"trials": [
            {"case_id": c, "fault": f, "verdict": v, "confidence": 1.0}
            for c, f, v in trials]}},
    }
    p = tmp / name
    p.write_text(json.dumps(r))
    return p


def _run(args):
    return subprocess.run([sys.executable, str(SCRIPT)] + args, cwd=REPO,
                          capture_output=True, text=True, timeout=300)


def test_refuses_when_the_route_leg_differs(tmp_path):
    """THE 2026-08-04 DEFECT, made mechanical."""
    old = _report(tmp_path, "old.json", leg="worker",
                  trials=[("c1", "clean", "CONFIRMED")])
    new = _report(tmp_path, "new.json", leg="critic",
                  trials=[("c1", "clean", "CONFIRMED")])
    r = _run([str(old), str(new)])
    assert r.returncode == 2
    assert "REFUSING TO COMPARE" in r.stdout and "escalation.leg" in r.stdout


def test_declared_difference_is_allowed(tmp_path):
    """A/Bing one component is legitimate — you just have to name it."""
    old = _report(tmp_path, "old.json", code="aaa",
                  trials=[("c1", "clean", "CONFIRMED")])
    new = _report(tmp_path, "new.json", code="zzz",
                  trials=[("c1", "clean", "CONFIRMED")])
    assert _run([str(old), str(new)]).returncode == 2
    r = _run([str(old), str(new), "--expect-differs", "code.verifier"])
    assert r.returncode == 0, r.stdout


def test_identical_provenance_compares_without_complaint(tmp_path):
    old = _report(tmp_path, "old.json", trials=[("c1", "clean", "CONFIRMED")])
    new = _report(tmp_path, "new.json", trials=[("c1", "clean", "CONFIRMED")])
    r = _run([str(old), str(new)])
    assert r.returncode == 0 and "REFUSING" not in r.stdout


# ── end to end ──────────────────────────────────────────────────────────────

def _pairs(n_right_to_wrong, n_wrong_to_right, n_same):
    """Build matched trial lists with a controlled discordance pattern."""
    old, new, i = [], [], 0
    for _ in range(n_right_to_wrong):
        old.append((f"c{i}", "fabrication", "REFUTED"))
        new.append((f"c{i}", "fabrication", "CONFIRMED")); i += 1
    for _ in range(n_wrong_to_right):
        old.append((f"c{i}", "fabrication", "CONFIRMED"))
        new.append((f"c{i}", "fabrication", "REFUTED")); i += 1
    for _ in range(n_same):
        old.append((f"c{i}", "fabrication", "REFUTED"))
        new.append((f"c{i}", "fabrication", "REFUTED")); i += 1
    return old, new


def test_detects_a_real_improvement(tmp_path):
    o, n = _pairs(0, 8, 40)
    r = _run([str(_report(tmp_path, "o.json", trials=o)),
              str(_report(tmp_path, "n.json", trials=n)), "--json"])
    out = json.loads(r.stdout)
    assert out["decided"] and "NEW BETTER" in out["verdict"]
    assert out["improved"] == 8 and out["regressed"] == 0


def test_detects_a_real_regression(tmp_path):
    o, n = _pairs(9, 0, 30)
    out = json.loads(_run([str(_report(tmp_path, "o.json", trials=o)),
                           str(_report(tmp_path, "n.json", trials=n)),
                           "--json"]).stdout)
    assert out["decided"] and "NEW WORSE" in out["verdict"]


def test_a_wash_is_not_declared_a_winner(tmp_path):
    o, n = _pairs(4, 5, 60)
    out = json.loads(_run([str(_report(tmp_path, "o.json", trials=o)),
                           str(_report(tmp_path, "n.json", trials=n)),
                           "--json"]).stdout)
    assert not out["decided"] and "NO DIFFERENCE" in out["verdict"]


def test_many_concordant_trials_do_not_dilute_the_signal(tmp_path):
    """The reason to pair at all: an 8-0 split stays decisive whether it sits
    among 10 unchanged trials or 500. Unpaired proportions would drown it."""
    for filler in (10, 500):
        o, n = _pairs(0, 8, filler)
        out = json.loads(_run([str(_report(tmp_path, f"o{filler}.json", trials=o)),
                               str(_report(tmp_path, f"n{filler}.json", trials=n)),
                               "--json"]).stdout)
        assert out["decided"], f"8-0 split not resolved among {filler} unchanged"


def test_power_limit_is_named_not_hidden(tmp_path):
    """Too few changes must read as 'cannot tell', never as 'no effect'."""
    o, n = _pairs(0, 2, 50)
    r = _run([str(_report(tmp_path, "o.json", trials=o)),
              str(_report(tmp_path, "n.json", trials=n))])
    assert "power limit, not evidence" in r.stdout


def test_balanced_is_withheld_when_a_class_is_empty(tmp_path):
    """A "balanced" score computed from one class is a raw score wearing a
    balanced label — the same false-precision shape as reporting a bare `fpr`
    for the raw arm. (Caught by revert-testing: the guard existed but nothing
    exercised it, because every other fixture happens to have both classes.)
    """
    o, n = _pairs(0, 3, 5)          # fabrication only => no non-refute trials
    r = _run([str(_report(tmp_path, "o.json", trials=o)),
              str(_report(tmp_path, "n.json", trials=n))])
    assert r.returncode == 0, r.stdout
    assert "balanced      NOT COMPUTED" in r.stdout
    assert "no trials in this class" in r.stdout
    # and it must still report what it CAN
    assert "raw accuracy" in r.stdout


def test_both_classes_present_still_reports_balanced(tmp_path):
    o = [("c0", "fabrication", "REFUTED"), ("c1", "clean", "CONFIRMED")]
    n = [("c0", "fabrication", "REFUTED"), ("c1", "clean", "CONFIRMED")]
    r = _run([str(_report(tmp_path, "o.json", trials=o)),
              str(_report(tmp_path, "n.json", trials=n))])
    assert "NOT COMPUTED" not in r.stdout and "balanced" in r.stdout


def test_disjoint_case_sets_refuse_rather_than_compare_nothing(tmp_path):
    old = _report(tmp_path, "o.json", trials=[("a", "clean", "CONFIRMED")])
    new = _report(tmp_path, "n.json", trials=[("b", "clean", "CONFIRMED")])
    r = _run([str(old), str(new)])
    assert r.returncode == 2 and "no trials pair up" in r.stdout
