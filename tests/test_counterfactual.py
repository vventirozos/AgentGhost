"""Counterfactual replay phase 1 (core/counterfactual.py, 2026-07-17).

The measurement leg of the post-mortem→lesson loop: concluded self-play
challenges are persisted with their validators; idle slots occasionally
replay them against the CURRENT lessons; a past-SUCCESS that now FAILS
quarantines (never deletes) the lessons hydrated into the failing run
and notifies the operator.
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core import counterfactual as cf
from ghost_agent.core.autonomous_activity import ActivityLog, SEVERITY_NOTIFY
from ghost_agent.memory.skills import SkillMemory


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    return tmp_path


def _persist(status="FAILURE", **kw):
    return cf.persist_challenge(
        challenge=kw.get("challenge", "count the rows"),
        setup_script=kw.get("setup", "print('setup')"),
        validation_script=kw.get("validator", "print('ok')"),
        status=status, cluster=kw.get("cluster", "data_analysis"),
    )


class TestPersistence:
    def test_decisive_challenge_persisted(self, home):
        cid = _persist("FAILURE")
        assert cid
        rows = [json.loads(l) for l in
                (home / "system" / "counterfactual" / "challenges.jsonl")
                .read_text().splitlines()]
        assert rows[0]["id"] == cid
        assert rows[0]["status"] == "FAILURE"
        assert rows[0]["validation_script"]

    def test_undecisive_or_validatorless_skipped(self, home):
        assert _persist("ERROR") is None
        assert cf.persist_challenge(challenge="x", setup_script="",
                                    validation_script="", status="SUCCESS") is None

    def test_decorated_live_statuses_persist_normalized(self, home):
        # The only live caller (dream.py) passes DECORATED status strings.
        assert _persist("SUCCESS (in 2 attempts)")
        assert _persist("FAILURE (Exhausted 3 attempts)")
        rows = [json.loads(l) for l in
                (home / "system" / "counterfactual" / "challenges.jsonl")
                .read_text().splitlines()]
        assert [r["status"] for r in rows] == ["SUCCESS", "FAILURE"]

    def test_non_agent_outcomes_skipped(self, home):
        for status in ("ABORTED_BY_SOLVER (attempt 2/3)",
                       "INFRA_ABORT (validator crashed on attempt 1 — "
                       "generator bug, agent not charged)",
                       "", "UNKNOWN"):
            assert _persist(status) is None
        assert not (home / "system" / "counterfactual").exists()

    def test_no_ghost_home_noop(self, monkeypatch):
        monkeypatch.delenv("GHOST_HOME", raising=False)
        assert _persist("FAILURE") is None


class TestSelection:
    def test_candidates_exclude_replayed(self, home):
        a = _persist("FAILURE")
        b = _persist("SUCCESS", challenge="another one")
        cf.record_result(challenge_id=a, original="FAILURE",
                         replay="SUCCESS", verdict="generalized")
        cand = cf.load_replay_candidates(10)
        assert [c["id"] for c in cand] == [b]

    def test_classify_all_verdicts(self):
        assert cf.classify("FAILURE", "SUCCESS") == "generalized"
        assert cf.classify("SUCCESS", "FAILURE") == "regression"
        assert cf.classify("SUCCESS", "SUCCESS") == "stable-pass"
        assert cf.classify("FAILURE", "FAILURE") == "still-failing"

    def test_classify_decorated_live_statuses(self):
        assert cf.classify("FAILURE (Exhausted 3 attempts)",
                           "SUCCESS (in 1 attempts)") == "generalized"
        assert cf.classify("SUCCESS (in 2 attempts)",
                           "FAILURE (Exhausted 3 attempts)") == "regression"
        assert cf.classify("SUCCESS",
                           "SUCCESS (in 1 attempts)") == "stable-pass"
        assert cf.classify("FAILURE (Aborted on attempt 1)",
                           "FAILURE (Exhausted 3 attempts)") == "still-failing"

    def test_classify_non_agent_replay_is_inconclusive(self):
        for replay in ("ABORTED_BY_SOLVER (attempt 2/3)",
                       "INFRA_ABORT (validator crashed on attempt 1 — "
                       "generator bug, agent not charged)",
                       "", "UNKNOWN"):
            assert cf.classify("SUCCESS", replay) == "inconclusive"
            assert cf.classify("FAILURE", replay) == "inconclusive"


class _FakeDreamer:
    def __init__(self, replay_status):
        self._status = replay_status
        self.injected = []
        self.last_self_play_status = ""

    async def synthetic_self_play(self, **kw):
        self.injected.append(kw.get("injected_challenge"))
        self.last_self_play_status = self._status


def _ctx(tmp_path, sm=None):
    log = ActivityLog(tmp_path / "activity.jsonl")
    return SimpleNamespace(skill_memory=sm, activity_log=log), log


class TestBatch:
    async def test_regression_quarantines_and_notifies(self, home, tmp_path):
        _persist("SUCCESS")
        (tmp_path / "mem").mkdir()
        sm = SkillMemory(tmp_path / "mem")
        sm.learn_lesson("use sets for dedup", "looped instead",
                        "track uniques in a set", trigger="use sets for dedup")
        sm.last_playbook_triggers = ["use sets for dedup"]
        ctx, log = _ctx(tmp_path, sm)
        dreamer = _FakeDreamer("FAILURE")

        summary = await cf.run_counterfactual_batch(dreamer, ctx)
        assert summary["replayed"] == 1
        assert summary["regressions"] == 1
        assert summary["quarantined"] == ["use sets for dedup"]
        # The replay rode the injection seam.
        assert dreamer.injected[0]["challenge"] == "count the rows"
        # Quarantined lesson no longer enters prompts…
        assert "use sets" not in (sm.get_playbook_context(
            "use sets for dedup") or "")
        # …but is still ON DISK with the reason (review, not deletion).
        raw = sm._load_playbook()
        assert raw and raw[0].get("quarantined") is True
        assert "counterfactual regression" in raw[0]["quarantine_reason"]
        # Operator notification is notify-severity.
        recs, _ = log.read_since(0)
        assert any(r.severity == SEVERITY_NOTIFY
                   and "regression" in r.summary for r in recs)

    async def test_generalized_is_info_and_no_quarantine(self, home, tmp_path):
        _persist("FAILURE")
        ctx, log = _ctx(tmp_path)
        dreamer = _FakeDreamer("SUCCESS")
        summary = await cf.run_counterfactual_batch(dreamer, ctx)
        assert summary["generalized"] == 1 and not summary["quarantined"]
        recs, _ = log.read_since(0)
        assert any("generalized" in r.summary for r in recs)

    async def test_results_ledger_prevents_reruns(self, home, tmp_path):
        _persist("FAILURE")
        ctx, _ = _ctx(tmp_path)
        dreamer = _FakeDreamer("SUCCESS")
        await cf.run_counterfactual_batch(dreamer, ctx)
        again = await cf.run_counterfactual_batch(dreamer, ctx)
        assert again["replayed"] == 0

    async def test_decorated_replay_success_is_not_regression(
            self, home, tmp_path):
        # "SUCCESS (in 1 attempts)" must grade as a pass, not quarantine
        # good lessons as a phantom regression.
        _persist("SUCCESS")
        (tmp_path / "mem").mkdir()
        sm = SkillMemory(tmp_path / "mem")
        sm.learn_lesson("use sets for dedup", "looped instead",
                        "track uniques in a set", trigger="use sets for dedup")
        sm.last_playbook_triggers = ["use sets for dedup"]
        ctx, log = _ctx(tmp_path, sm)
        dreamer = _FakeDreamer("SUCCESS (in 1 attempts)")
        summary = await cf.run_counterfactual_batch(dreamer, ctx)
        assert summary["stable"] == 1
        assert summary["regressions"] == 0 and not summary["quarantined"]
        raw = sm._load_playbook()
        assert raw and not raw[0].get("quarantined")


def _mem_with_lessons(tmp_path, *triggers):
    (tmp_path / "mem").mkdir(exist_ok=True)
    sm = SkillMemory(tmp_path / "mem")
    for trig in triggers:
        sm.learn_lesson(trig, "mistake", "correction", trigger=trig)
    return sm


class TestInconclusive:
    async def test_no_quarantine_no_notify_and_retry_allowed(
            self, home, tmp_path):
        cid = _persist("SUCCESS")
        sm = _mem_with_lessons(tmp_path, "use sets for dedup")
        sm.last_playbook_triggers = ["use sets for dedup"]
        ctx, log = _ctx(tmp_path, sm)
        dreamer = _FakeDreamer("ABORTED_BY_SOLVER (attempt 2/3)")
        summary = await cf.run_counterfactual_batch(dreamer, ctx)
        assert summary["inconclusive"] == 1
        assert summary["regressions"] == 0 and not summary["quarantined"]
        raw = sm._load_playbook()
        assert raw and not raw[0].get("quarantined")
        recs, _ = log.read_since(0)
        assert not any(r.severity == SEVERITY_NOTIFY for r in recs)
        # Not marked done: the challenge stays eligible for a retry…
        assert [c["id"] for c in cf.load_replay_candidates(10)] == [cid]
        # …and the state record carries the attempt count.
        rows = [json.loads(l) for l in
                (home / "system" / "counterfactual" / "results.jsonl")
                .read_text().splitlines()]
        assert rows[0]["verdict"] == "inconclusive"
        assert rows[0]["attempts"] == 1

    async def test_empty_status_replay_is_inconclusive(self, home, tmp_path):
        # An early-return/infra path leaves last_self_play_status empty,
        # which the batch reads as "UNKNOWN".
        _persist("SUCCESS")
        ctx, _ = _ctx(tmp_path)
        dreamer = _FakeDreamer("")
        summary = await cf.run_counterfactual_batch(dreamer, ctx)
        assert summary["inconclusive"] == 1 and summary["regressions"] == 0

    async def test_inconclusive_retries_are_bounded(self, home, tmp_path):
        _persist("SUCCESS")
        ctx, _ = _ctx(tmp_path)
        dreamer = _FakeDreamer(
            "INFRA_ABORT (validator crashed on attempt 1 — "
            "generator bug, agent not charged)")
        for expected in (1, 1, 1, 0):  # gives up quietly after 3 attempts
            summary = await cf.run_counterfactual_batch(dreamer, ctx)
            assert summary["replayed"] == expected
        rows = [json.loads(l) for l in
                (home / "system" / "counterfactual" / "results.jsonl")
                .read_text().splitlines()]
        assert [r["attempts"] for r in rows] == [1, 2, 3]


class TestTriggerSnapshot:
    async def test_dreamer_snapshot_preferred_over_shared_attribute(
            self, home, tmp_path):
        _persist("SUCCESS")
        sm = _mem_with_lessons(tmp_path, "sim hydrated lesson",
                               "user turn lesson")
        # A concurrent interactive turn re-stamped the shared attribute…
        sm.last_playbook_triggers = ["user turn lesson"]
        ctx, _ = _ctx(tmp_path, sm)
        dreamer = _FakeDreamer("FAILURE (Exhausted 3 attempts)")
        # …but the dreamer stamped what the SIM actually hydrated.
        dreamer.last_selfplay_hydrated_triggers = ["sim hydrated lesson"]
        summary = await cf.run_counterfactual_batch(dreamer, ctx)
        assert summary["regressions"] == 1
        assert summary["quarantined"] == ["sim hydrated lesson"]
        assert sm.get_playbook_items("user turn lesson")  # untouched

    async def test_empty_snapshot_quarantines_nothing(self, home, tmp_path):
        _persist("SUCCESS")
        sm = _mem_with_lessons(tmp_path, "user turn lesson")
        sm.last_playbook_triggers = ["user turn lesson"]
        ctx, _ = _ctx(tmp_path, sm)
        dreamer = _FakeDreamer("FAILURE (Exhausted 3 attempts)")
        dreamer.last_selfplay_hydrated_triggers = []  # sim hydrated nothing
        summary = await cf.run_counterfactual_batch(dreamer, ctx)
        assert summary["regressions"] == 1
        assert summary["quarantined"] == []
        assert sm.get_playbook_items("user turn lesson")

    async def test_none_snapshot_falls_back_to_shared_attribute(
            self, home, tmp_path):
        _persist("SUCCESS")
        sm = _mem_with_lessons(tmp_path, "user turn lesson")
        sm.last_playbook_triggers = ["user turn lesson"]
        ctx, _ = _ctx(tmp_path, sm)
        dreamer = _FakeDreamer("FAILURE (Exhausted 3 attempts)")
        dreamer.last_selfplay_hydrated_triggers = None  # back-compat
        summary = await cf.run_counterfactual_batch(dreamer, ctx)
        assert summary["quarantined"] == ["user turn lesson"]


class TestSkillsQuarantine:
    def test_filter_covers_both_retrieval_surfaces(self, tmp_path):
        sm = SkillMemory(tmp_path)
        sm.learn_lesson("parse json safely", "used eval",
                        "use json.loads", trigger="parse json safely")
        assert sm.quarantine_lesson("parse json safely", "test") == 1
        assert sm.get_playbook_items("parse json safely") == []
        out = sm.get_playbook_context("parse json safely")
        assert "parse json" not in (out or "")

    def test_hydration_side_channel_set(self, tmp_path):
        sm = SkillMemory(tmp_path)
        sm.learn_lesson("regex over split", "split on comma",
                        "use re for csv edge cases",
                        trigger="regex over split")
        sm.get_playbook_context("regex over split")
        assert sm.last_playbook_triggers == ["regex over split"]


class TestWiringPins:
    def test_dream_seam_and_conclusion_hooks(self):
        src = (Path(__file__).resolve().parents[1]
               / "src" / "ghost_agent" / "core" / "dream.py").read_text()
        assert "injected_challenge: dict = None" in src
        assert "if _tpl is None and not gen_ok:" in src
        assert "self.last_self_play_status = str(status_str)" in src
        assert "if not injected_challenge:" in src  # replays not re-persisted

    def test_idle_hook_runs_counterfactual_slot(self):
        src = (Path(__file__).resolve().parents[1]
               / "src" / "ghost_agent" / "core" / "agent.py").read_text()
        assert "run_counterfactual_batch" in src
        assert "load_replay_candidates(1)" in src
