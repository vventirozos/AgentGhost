"""Unit tests for the paired-ablation statistics.

The definitive full-vs-thin verdict rests entirely on these functions, so they
get pinned down here. Pure math — no agent boot, no network.
"""

import sys
from pathlib import Path

import pytest

# scripts/ is not a package; put it on the path so we can import the driver.
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

ap = pytest.importorskip("ablation_paired")


# --------------------------------------------------------------------------
# McNemar exact two-sided p-value
# --------------------------------------------------------------------------

def test_mcnemar_no_discordant_pairs_is_one():
    assert ap._mcnemar_exact(0, 0) == 1.0


def test_mcnemar_symmetric_is_not_significant():
    # equal discordances => no evidence of a difference
    assert ap._mcnemar_exact(5, 5) > 0.5
    assert ap._mcnemar_exact(20, 20) > 0.5


def test_mcnemar_all_one_direction_is_significant():
    # 10 pairs all flipping the same way is strong evidence
    assert ap._mcnemar_exact(10, 0) < 0.01
    assert ap._mcnemar_exact(0, 10) < 0.01


def test_mcnemar_is_symmetric_in_b_c():
    assert ap._mcnemar_exact(8, 2) == ap._mcnemar_exact(2, 8)


def test_mcnemar_small_imbalance_not_significant():
    # 6 vs 4 discordant is not enough to reject at 0.05
    assert ap._mcnemar_exact(6, 4) > 0.05


def test_mcnemar_clamped_to_one():
    assert ap._mcnemar_exact(1, 1) <= 1.0


# --------------------------------------------------------------------------
# Paired difference CI (ref - cfg), with b/c discordant counts
# --------------------------------------------------------------------------

def test_paired_diff_counts_and_sign():
    # ref better: b (ref pass / cfg fail) > c (ref fail / cfg pass)
    pairs = [(True, False)] * 8 + [(False, True)] * 2 + [(True, True)] * 20
    diff, lo, hi, b, c = ap._paired_diff_ci(pairs)
    assert b == 8 and c == 2
    assert diff == pytest.approx((8 - 2) / 30)
    assert lo < diff < hi


def test_paired_diff_empty():
    assert ap._paired_diff_ci([]) == (0.0, 0.0, 0.0, 0, 0)


def test_paired_diff_all_concordant_is_zero():
    pairs = [(True, True)] * 10 + [(False, False)] * 5
    diff, lo, hi, b, c = ap._paired_diff_ci(pairs)
    assert b == 0 and c == 0
    assert diff == 0.0 and lo == 0.0 and hi == 0.0


def test_paired_diff_negative_when_cfg_better():
    pairs = [(False, True)] * 7 + [(True, False)] * 1
    diff, lo, hi, b, c = ap._paired_diff_ci(pairs)
    assert b == 1 and c == 7
    assert diff < 0


# --------------------------------------------------------------------------
# Within-task variance (determinism detector)
# --------------------------------------------------------------------------

def test_within_task_variance_zero_when_deterministic():
    recs = [{"config": "full", "task_id": "t1", "passed": True} for _ in range(5)]
    assert ap._within_task_variance(recs, "full") == 0.0


def test_within_task_variance_positive_when_mixed():
    recs = ([{"config": "full", "task_id": "t1", "passed": True}] * 3
            + [{"config": "full", "task_id": "t1", "passed": False}] * 3)
    assert ap._within_task_variance(recs, "full") > 0.0


def test_within_task_variance_ignores_other_configs():
    recs = [{"config": "thin", "task_id": "t1", "passed": True},
            {"config": "thin", "task_id": "t1", "passed": False}]
    # no records for 'full' -> defined as 0.0
    assert ap._within_task_variance(recs, "full") == 0.0


# --------------------------------------------------------------------------
# Report builder smoke (renders without raising on a realistic record set)
# --------------------------------------------------------------------------

def test_build_report_smoke():
    recs = []
    truth = {"a": (True, True), "b": (True, False), "c": (False, False)}
    for rep in range(4):
        for tid, (pf, pt) in truth.items():
            recs.append({"config": "full", "repeat": rep, "task_id": tid,
                         "cluster": "x", "passed": pf, "duration_s": 5.0, "reason": ""})
            recs.append({"config": "thin", "repeat": rep, "task_id": tid,
                         "cluster": "x", "passed": pt, "duration_s": 4.0, "reason": ""})
    report = ap._build_report(recs, "full",
                              {"suite": "default", "n_tasks": 3, "repeats": 4, "n_pairs": 12})
    assert "Paired ablation report" in report
    assert "`thin`" in report
    assert "McNemar" in report
