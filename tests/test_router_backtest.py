"""§4I Phase 2 backtest (scripts/router_confidence_backtest.py).

The script's FLAT branch is what stops the plan, so the thing that must be
right is its willingness to say "no". A bare spread threshold said DISCRIMINATES
on perfectly flat ground truth ~88% of the time at the default bucket size.
"""
from __future__ import annotations

import importlib.util
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "router_backtest", REPO / "scripts" / "router_confidence_backtest.py")
bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt)

from ghost_agent.core.experiments import asymp_cs_radius  # noqa: E402


def _traj(conf, outcome, kind="user_request"):
    return SimpleNamespace(task_kind=kind, outcome=outcome,
                           extra={"router_confidence": conf})


def test_buckets_cover_the_unit_interval_and_split_at_the_router_threshold():
    edges = [lo for lo, _ in bt.BUCKETS]
    assert 0.30 in edges, "the router's own escalation threshold must be an edge"
    assert bt._bucket_of(0.0) is not None and bt._bucket_of(1.0) is not None
    assert bt._bucket_of(-0.1) is None and bt._bucket_of(1.5) is None


def test_unstamped_and_non_user_turns_are_excluded():
    trajs = [_traj(0.2, "failed"),
             SimpleNamespace(task_kind="user_request", outcome="failed", extra={}),
             _traj(0.2, "failed", kind="self_play")]
    stats, cov = bt.collect(trajs)
    assert cov["user_turns"] == 2      # the self_play turn is not counted
    assert cov["stamped"] == 1


def test_out_of_range_confidence_is_reported_not_hidden():
    """Counting it as 'not stamped' would hide a producer bug as a coverage
    gap."""
    stats, cov = bt.collect([_traj(1.7, "failed")])
    assert cov["stamped"] == 1
    assert cov["out_of_range"] == 1


def test_unknown_outcomes_count_as_seen_but_not_resolved():
    stats, cov = bt.collect([_traj(0.2, "unknown"), _traj(0.2, "passed")])
    key = bt._bucket_of(0.2)
    assert stats[key]["n"] == 2
    assert stats[key]["unknown"] == 1
    assert cov["resolved"] == 1


def _verdict(rows_spec, min_per_bucket=30):
    """rows_spec: list of (n, failure_rate) → the script's verdict logic."""
    rng = random.Random(0)
    alpha = bt.ALPHA / max(1, len(bt.BUCKETS))
    rows = []
    for n, rate in rows_spec:
        outs = [1.0] * int(round(n * rate)) + [0.0] * (n - int(round(n * rate)))
        rows.append({"failure_rate": sum(outs) / n,
                     "ci": asymp_cs_radius(outs, alpha=alpha),
                     "usable": n >= min_per_bucket})
    usable = [r for r in rows if r["usable"] and r["ci"] is not None]
    best = min(usable, key=lambda r: r["failure_rate"])
    worst = max(usable, key=lambda r: r["failure_rate"])
    spread = worst["failure_rate"] - best["failure_rate"]
    disjoint = ((best["failure_rate"] + best["ci"])
                < (worst["failure_rate"] - worst["ci"]))
    return spread, disjoint


def test_a_wide_spread_on_thin_data_is_not_significant():
    """The defect: two noisy point estimates 30 apart looked like a finding."""
    spread, disjoint = _verdict([(30, 0.20), (30, 0.55)])
    assert spread >= bt.DISCRIMINATION_THRESHOLD
    assert not disjoint          # ...but the intervals still overlap


def test_a_real_separation_on_ample_data_is_significant():
    spread, disjoint = _verdict([(600, 0.12), (600, 0.62)])
    assert spread >= bt.DISCRIMINATION_THRESHOLD
    assert disjoint


def test_flat_ground_truth_does_not_read_as_discriminating():
    """4000 draws of a perfectly flat world; the old spread-only rule fired on
    ~88% of them at the default bucket size."""
    rng = random.Random(11)
    alpha = bt.ALPHA / max(1, len(bt.BUCKETS))
    false_hits = 0
    for _ in range(400):
        rows = []
        for _ in bt.BUCKETS:
            outs = [1.0 if rng.random() < 0.3 else 0.0 for _ in range(30)]
            rows.append({"failure_rate": sum(outs) / 30,
                         "ci": asymp_cs_radius(outs, alpha=alpha)})
        best = min(rows, key=lambda r: r["failure_rate"])
        worst = max(rows, key=lambda r: r["failure_rate"])
        spread = worst["failure_rate"] - best["failure_rate"]
        disjoint = ((best["failure_rate"] + best["ci"])
                    < (worst["failure_rate"] - worst["ci"]))
        if spread >= bt.DISCRIMINATION_THRESHOLD and disjoint:
            false_hits += 1
    assert false_hits == 0, f"{false_hits}/400 false DISCRIMINATES on flat data"


def test_missing_corpus_exits_two_and_writes_nothing(tmp_path, monkeypatch, capsys):
    missing = tmp_path / "nope"
    monkeypatch.setattr("sys.argv",
                        ["x", "--trajectories", str(missing), "--json"])
    assert bt.main() == 2
    assert "corpus_missing" in capsys.readouterr().out
    assert not missing.exists()
