"""`scripts/heal_diary_outcomes.py` — the one-shot diary/corpus reconciliation.

The forward fix (queue #7) wires the late verdict and the human label into
the diary; this script repairs the backlog those legs never wrote. Its whole
safety argument is that it FOLLOWS the corpus and touches nothing else, so
that is what these pins drive:

  * it heals only rows that ALREADY EXIST in the diary and are ALREADY
    `unknown` — it never creates a row (capture is real_only-gated and a
    created row would launder a sim/bench turn into the diary) and never
    re-labels a row that already carries a verdict;
  * the outcome it writes is read through the corrections OVERLAY, so the
    collector's authority order (human label supersedes a machine verdict)
    is inherited rather than re-stated;
  * a dry run writes nothing at all.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import importlib.util
from pathlib import Path

import pytest

from ghost_agent.distill.collector import TrajectoryCollector
from ghost_agent.distill.schema import Outcome, Trajectory
from ghost_agent.selfhood import SelfModel


def _load_script():
    """Load the script by path — scripts/ is not a package."""
    path = (Path(__file__).resolve().parent.parent / "scripts"
            / "heal_diary_outcomes.py")
    spec = importlib.util.spec_from_file_location("heal_diary_outcomes", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


heal = _load_script()


@pytest.fixture
def home(tmp_path):
    """A GHOST_HOME with a corpus and a diary sharing trajectory ids."""
    system = tmp_path / "system"
    traj_root = system / "trajectories"
    col = TrajectoryCollector(root=traj_root, session_id="s", enabled=True)
    sm = SelfModel(system / "selfhood", enabled=True)

    ids = {}
    for key in ("late_pass", "late_fail", "human_wins", "still_unknown",
                "already_labelled"):
        t = Trajectory(session_id=key, user_request=f"do {key}",
                       final_response="done",
                       outcome=Outcome.UNKNOWN.value, extra={"req_id": key})
        col.append(t)
        ids[key] = t.id
        sm.capture_turn(trajectory_id=t.id, user_request=f"do {key}",
                        tool_names=["execute_command"],
                        outcome=("passed" if key == "already_labelled"
                                 else "unknown"),
                        final_response="done")

    # A diary row whose turn the corpus never captured at all.
    sm.capture_turn(trajectory_id="orphan-id", user_request="orphaned",
                    tool_names=[], outcome="unknown", final_response="x")

    col.update_outcome(ids["late_pass"], "passed", source="verifier_late")
    col.update_outcome(ids["late_fail"], "failed", reason="never ran it",
                       source="verifier_late")
    # Authority order: the machine said failed, the human then said passed.
    col.update_outcome(ids["human_wins"], "failed", reason="judge says no",
                       source="verifier_late")
    col.update_outcome(ids["human_wins"], "passed",
                       source="human_feedback:web")
    # `already_labelled` is resolved in the corpus too — the diary already
    # agrees, so it must simply not be counted as a gap.
    col.update_outcome(ids["already_labelled"], "passed",
                       source="verifier_late")

    return tmp_path, system, ids, sm


def _diary_rows(sm):
    rows = {}
    for line in sm.autobio.path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        rows[d.get("trajectory_id")] = d
    return rows


def _run(tmp_path, *argv):
    argv = ["--home", str(tmp_path)] + list(argv)
    old = sys.argv
    sys.argv = ["heal_diary_outcomes.py"] + argv
    try:
        return heal.main()
    finally:
        sys.argv = old


class TestGapDetection:
    def test_only_unknown_rows_with_a_resolved_corpus_turn_are_gaps(
            self, home):
        tmp_path, system, ids, sm = home
        resolved = heal.resolved_outcomes(system / "trajectories")
        gaps = heal.diary_gaps(sm.autobio.path, resolved)
        got = {g[0] for g in gaps}

        assert got == {ids["late_pass"], ids["late_fail"], ids["human_wins"]}
        # explicitly NOT gaps:
        assert ids["still_unknown"] not in got     # corpus has no verdict
        assert ids["already_labelled"] not in got  # diary already agrees
        assert "orphan-id" not in got              # no corpus turn at all

    def test_the_overlay_decides_the_outcome_not_the_newest_machine_verdict(
            self, home):
        """`human_wins` carries a later human `passed` over a machine
        `failed`. Reading through iter_trajectories inherits the collector's
        authority order; a script that read the raw sidecar tail-first, or
        preferred `verifier_late`, would write `failed` here."""
        tmp_path, system, ids, sm = home
        resolved = heal.resolved_outcomes(system / "trajectories")

        assert resolved[ids["human_wins"]][0] == "passed"


class TestDryRunWritesNothing:
    def test_dry_run_leaves_every_row_untouched(self, home):
        tmp_path, system, ids, sm = home
        before = sm.autobio.path.read_bytes()

        assert _run(tmp_path) == 0

        assert sm.autobio.path.read_bytes() == before
        assert not list(sm.autobio.path.parent.glob("*.pre-heal-*"))


class TestApply:
    def test_apply_heals_exactly_the_gaps(self, home):
        tmp_path, system, ids, sm = home
        assert _run(tmp_path, "--apply") == 0

        rows = _diary_rows(sm)
        assert rows[ids["late_pass"]]["outcome"] == "passed"
        assert rows[ids["late_fail"]]["outcome"] == "failed"
        assert rows[ids["human_wins"]]["outcome"] == "passed"
        # untouched
        assert rows[ids["still_unknown"]]["outcome"] == "unknown"
        assert rows[ids["already_labelled"]]["outcome"] == "passed"
        assert rows["orphan-id"]["outcome"] == "unknown"

    def test_apply_patches_the_prose_verdict_clause(self, home):
        tmp_path, system, ids, sm = home
        assert _run(tmp_path, "--apply") == 0

        rows = _diary_rows(sm)
        for key in ("late_pass", "late_fail", "human_wins"):
            summary = rows[ids[key]].get("summary") or ""
            assert "without a verdict either way" not in summary

    def test_apply_creates_no_new_rows(self, home):
        tmp_path, system, ids, sm = home
        before = len(_diary_rows(sm))

        assert _run(tmp_path, "--apply") == 0

        assert len(_diary_rows(sm)) == before

    def test_apply_backs_the_diary_up_first(self, home):
        tmp_path, system, ids, sm = home
        before = sm.autobio.path.read_bytes()

        assert _run(tmp_path, "--apply") == 0

        backups = list(sm.autobio.path.parent.glob("*.pre-heal-*"))
        assert len(backups) == 1
        assert backups[0].read_bytes() == before   # the PRE state, verbatim

    def test_apply_is_idempotent(self, home):
        tmp_path, system, ids, sm = home
        assert _run(tmp_path, "--apply") == 0
        after_first = _diary_rows(sm)

        assert _run(tmp_path, "--apply") == 0

        assert _diary_rows(sm) == after_first

    def test_limit_caps_the_rows_changed(self, home):
        tmp_path, system, ids, sm = home
        assert _run(tmp_path, "--apply", "--limit", "1") == 0

        healed = [d for d in _diary_rows(sm).values()
                  if d.get("outcome") in ("passed", "failed")]
        # one healed + the pre-existing `already_labelled`
        assert len(healed) == 2


class TestGuards:
    def test_missing_home_is_an_error_not_a_silent_noop(self, monkeypatch):
        monkeypatch.delenv("GHOST_HOME", raising=False)
        old = sys.argv
        sys.argv = ["heal_diary_outcomes.py"]
        try:
            assert heal.main() == 2
        finally:
            sys.argv = old

    def test_missing_store_is_an_error(self, tmp_path):
        assert _run(tmp_path) == 2
