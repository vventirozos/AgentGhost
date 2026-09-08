"""§4FD pins: the prediction-powered line in the live arm report.

Each pin names the world where it fails (R4). The corpus is written to a
temp trajectory root with a corrections sidecar, and the report is produced
through the SAME entry point introspect uses (`report_from_trajectories`), so
a wiring mutant (judge_map never built, fold never called, render never
appended) is killed here and not only in the unit pins of core/ppi.py.
"""
import json
import random
from pathlib import Path

import pytest

from ghost_agent.core import experiments as ex
from ghost_agent.distill.collector import TrajectoryCollector
from ghost_agent.distill.schema import Trajectory


def _write_corpus(root: Path, rows, corrections):
    day = root / "2026-09-07"
    day.mkdir(parents=True)
    with (day / "session-20260907T000000-aaaa.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with (root / "corrections.jsonl").open("w") as f:
        for c in corrections:
            f.write(json.dumps(c) + "\n")


def _row(i, arm, outcome="unknown", exp="fs_batch"):
    t = Trajectory(id=f"t{i:04d}", session_id="s", task_kind="user_request",
                   user_request=f"request {i}", outcome=outcome, n_steps=1,
                   duration_s=1.0,
                   extra={ex.EXTRA_KEY: {exp: arm}})
    return t.to_dict()


def _build(tmp_path, *, gold_per_arm=20, judged_per_arm=200, seed=1,
           skew=False):
    """A corpus where the machine judge is a noisy copy of the truth and a
    random (or skewed) slice carries a human label."""
    rng = random.Random(seed)
    rows, corr = [], []
    i = 0
    for arm, p_fail in (("control", 0.30), ("treatment", 0.15)):
        for k in range(judged_per_arm):
            truth = 1 if rng.random() < p_fail else 0
            machine = 1 - truth if rng.random() < 0.2 else truth
            rows.append(_row(i, arm))
            corr.append({"trajectory_id": f"t{i:04d}",
                         "outcome": "failed" if machine else "passed",
                         "reason": "verifier refuted (late)" if machine else "ok",
                         "source": "verifier_late", "timestamp": "2026-09-07T00:00:00"})
            # gold: a random slice, or (skewed) only rows the judge failed
            want_gold = (k < gold_per_arm) if not skew else (machine == 1 and k < 4 * gold_per_arm)
            if want_gold:
                corr.append({"trajectory_id": f"t{i:04d}",
                             "outcome": "failed" if truth else "passed",
                             "reason": "thumbs", "source": "human_feedback:web",
                             "timestamp": "2026-09-07T00:01:00"})
            i += 1
    _write_corpus(tmp_path, rows, corr)
    return tmp_path


def test_collector_exposes_machine_and_human_outcomes_from_every_row(tmp_path):
    """World where it fails: the reader is last-write-wins, so a row with a
    verifier verdict AND a later human label loses its machine verdict."""
    _build(tmp_path, gold_per_arm=5, judged_per_arm=10)
    coll = TrajectoryCollector(root=tmp_path, session_id="reader")
    m = coll.machine_and_human_outcomes()
    assert len(m) == 20
    both = [v for v in m.values() if v[0] is not None and v[1] is not None]
    assert len(both) == 10
    only_machine = [v for v in m.values() if v[1] is None]
    assert len(only_machine) == 10 and all(v[0] in ("passed", "failed") for v in only_machine)


def test_overlay_keeps_the_native_outcome_beside_the_correction(tmp_path):
    """World where it fails: the overlay overwrites `outcome` and nothing
    keeps the write-time verdict, so the judge cannot be recovered for a
    human-relabelled row whose machine verdict was inline."""
    rows = [_row(0, "control", outcome="failed")]
    corr = [{"trajectory_id": "t0000", "outcome": "passed", "reason": "thumbs",
             "source": "human_feedback:web", "timestamp": "x"}]
    _write_corpus(tmp_path, rows, corr)
    coll = TrajectoryCollector(root=tmp_path, session_id="reader")
    [t] = list(coll.iter_trajectories())
    assert t.outcome == "passed"
    assert t.extra.get("outcome_native") == "failed"
    judge, gold = ex._judge_and_gold(t, coll.machine_and_human_outcomes())
    assert judge == 1.0 and gold == 0.0


def test_report_renders_a_ppi_line_with_a_narrower_interval(tmp_path):
    """World where it fails: judge_map is never built, `_fold` drops the
    judge/gold, or `_render_block` never appends the line — the report has
    no `failure_rate[ppi]` row. And: the PPI interval must be NARROWER than
    the GOLD-ONLY one (human_failure_rate, n=20), because 200 judged rows
    per arm back those 20 gold labels — that is the estimator's whole claim."""
    _build(tmp_path)
    out = ex.report_from_trajectories(str(tmp_path), alpha=0.05)
    line = next(l for l in out.splitlines() if "failure_rate[ppi]" in l)
    assert "gold=20/20" in line and "judged=200/200" in line
    assert "informational" in line
    assert "SKEWED" not in line
    assert "machine-as-truth CS" in line
    # one-look: PPI must be narrower than gold-only by a real margin
    pct = line.split("one-look")[1].split("(")[1].split("% width")[0]
    assert pct.startswith("-"), line
    assert float(pct) <= -10.0, line
    # anytime: the union-bound figure is printed beside the gold-only CS
    assert "anytime ±" in line and "union bound" in line


def test_ppi_line_flags_a_skewed_gold_slice(tmp_path):
    """World where it fails: the representativeness check is not surfaced,
    so a bottom-quintile labelling policy (§4ER) silently biases the line."""
    _build(tmp_path, skew=True)
    out = ex.report_from_trajectories(str(tmp_path), alpha=0.05)
    line = next(l for l in out.splitlines() if "failure_rate[ppi]" in l)
    assert "SKEWED" in line


def test_no_gold_means_no_ppi_line_not_a_zero(tmp_path):
    """World where it fails: an arm with no human labels renders a PPI
    estimate anyway (a number from nothing), or the refusal loses its
    reason."""
    _build(tmp_path, gold_per_arm=0)
    out = ex.report_from_trajectories(str(tmp_path), alpha=0.05)
    assert "failure_rate[ppi]" not in out
    # gold on ONE arm only → the line renders as a refusal naming the other
    # arm (a mutant requiring gold on both arms renders nothing at all)
    rows, corr = [], []
    for i in range(40):
        arm = "control" if i < 20 else "treatment"
        rows.append(_row(i, arm))
        corr.append({"trajectory_id": f"t{i:04d}", "outcome": "passed", "reason": "ok",
                     "source": "verifier_late", "timestamp": "x"})
        if arm == "control" and i < 6:
            corr.append({"trajectory_id": f"t{i:04d}", "outcome": "passed", "reason": "t",
                         "source": "human_feedback:web", "timestamp": "y"})
    _write_corpus(tmp_path / "half", rows, corr)
    out2 = ex.report_from_trajectories(str(tmp_path / "half"), alpha=0.05)
    line = next(l for l in out2.splitlines() if "failure_rate[ppi]" in l)
    assert "NO PPI ESTIMATE" in line and "treatment" in line and "gold" in line


def test_rows_without_a_machine_verdict_leave_the_ppi_population(tmp_path):
    """World where it fails: an `unknown` row is folded as judge=0 (a free
    pass), inflating the judged count and biasing the estimate."""
    rows = [_row(i, "control") for i in range(6)] + [_row(i, "treatment") for i in range(6, 12)]
    corr = []
    for i in range(12):
        if i % 3 == 0:
            continue  # no machine verdict at all → must not be judged
        corr.append({"trajectory_id": f"t{i:04d}", "outcome": "passed",
                     "reason": "ok", "source": "verifier_late", "timestamp": "x"})
        corr.append({"trajectory_id": f"t{i:04d}", "outcome": "passed",
                     "reason": "thumbs", "source": "human_feedback:cli", "timestamp": "y"})
    _write_corpus(tmp_path, rows, corr)
    coll = TrajectoryCollector(root=tmp_path, session_id="reader")
    all_stats, _, _ = ex.summarize_streaming(
        coll.iter_trajectories(), judge_map=coll.machine_and_human_outcomes())
    c = all_stats["fs_batch"]["control"]
    assert len(c.judge_labeled) == 4 and len(c.judge_unlabeled) == 0
    assert c.n == 6


def test_one_definition_of_human_and_retractions_clear_both(tmp_path):
    """§4FH M3/m2/m3. World where it fails: `user_correction` (a lexical
    detector) counts as gold, `operator_overlay` does not, or an operator
    retraction (`outcome=unknown`) leaves the retracted machine verdict as
    the judge."""
    rows = [_row(i, "control") for i in range(3)]
    corr = [
        {"trajectory_id": "t0000", "outcome": "failed", "reason": "no",
         "source": "user_correction", "timestamp": "x"},           # detector → machine
        {"trajectory_id": "t0001", "outcome": "passed", "reason": "op",
         "source": "operator_overlay", "timestamp": "x"},          # operator → human
        {"trajectory_id": "t0002", "outcome": "failed", "reason": "refuted",
         "source": "verifier_late", "timestamp": "x"},
        {"trajectory_id": "t0002", "outcome": "unknown", "reason": "false positive",
         "source": "operator_overlay", "timestamp": "y"},          # retraction
    ]
    _write_corpus(tmp_path, rows, corr)
    coll = TrajectoryCollector(root=tmp_path, session_id="reader")
    m = coll.machine_and_human_outcomes()
    assert m["t0000"] == ("failed", None)
    assert m["t0001"] == (None, "passed")
    assert m["t0002"] == (None, None)


def test_non_dict_extra_does_not_drop_the_row(tmp_path):
    """§4FH m6. World where it fails: `extra.get` on a non-dict raises inside
    the walk and the row leaves EVERY metric, not just the PPI series."""
    t = Trajectory(id="tx", session_id="s", task_kind="user_request", outcome="failed",
                   n_steps=1, duration_s=1.0, extra={ex.EXTRA_KEY: {"fs_batch": "control"}})
    d = t.to_dict(); d["extra"] = {ex.EXTRA_KEY: {"fs_batch": "control"}}
    j, g = ex._judge_and_gold(SimpleNamespaceLike(d), None)
    assert j == 1.0 and g is None


class SimpleNamespaceLike:
    def __init__(self, d):
        self.id = d["id"]; self.outcome = d["outcome"]; self.extra = "not-a-dict"


def test_ppi_line_compares_anytime_widths_on_the_same_rows(tmp_path):
    """§4FH M3. World where it fails: the anytime comparison is printed
    against `human_failure_rate`'s CS (a different, larger population) and
    reads as if PPI were ~2× wider than its own gold rows."""
    _build(tmp_path)
    out = ex.report_from_trajectories(str(tmp_path), alpha=0.05)
    line = next(l for l in out.splitlines() if "failure_rate[ppi]" in l)
    assert "gold-only CS on the same rows" in line
    assert "human_failure_rate CS" in line and "all human-labelled rows, n=" in line
