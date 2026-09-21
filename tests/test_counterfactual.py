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


async def _confirmed_regression(dreamer, ctx):
    """§4JF: one failed replay of a past SUCCESS is a CANDIDATE (no
    quarantine, info-level); the reproducing failure confirms. Returns the
    confirming batch's summary after asserting the candidate step."""
    first = await cf.run_counterfactual_batch(dreamer, ctx)
    assert first["regressions"] == 0 and first.get("candidates") == 1
    assert not first["quarantined"]
    return await cf.run_counterfactual_batch(dreamer, ctx)


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

        summary = await _confirmed_regression(dreamer, ctx)
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
        # Recorded for the digest / introspect, NEVER pushed (2026-09-21:
        # the operator does not want a night-time page for a regression).
        recs, _ = log.read_since(0)
        assert any("regression" in r.summary and "use sets for dedup" in r.summary for r in recs)
        assert not any(r.severity == SEVERITY_NOTIFY for r in recs)

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
        summary = await _confirmed_regression(dreamer, ctx)
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
        summary = await _confirmed_regression(dreamer, ctx)
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
        summary = await _confirmed_regression(dreamer, ctx)
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


class TestReplayGate:
    """Learning-state gate (2026-07-27): a batch only runs when the
    lessons/skills state changed since the last decisive batch — 45/45
    live replays of an unchanged state returned stable-pass and taught
    nothing, while consuming ~40% of the idle self-play budget."""

    def _playbook(self, home, content="[]"):
        mem = home / "system" / "memory"
        mem.mkdir(parents=True, exist_ok=True)
        (mem / "skills_playbook.json").write_text(content)

    def test_fingerprint_tracks_playbook_content(self, home):
        self._playbook(home, "[]")
        fp1 = cf.learning_fingerprint()
        fp2 = cf.learning_fingerprint()
        assert fp1 and fp1 == fp2  # stable on unchanged state
        self._playbook(home, '[{"trigger": "new lesson"}]')
        assert cf.learning_fingerprint() != fp1

    def test_fingerprint_empty_without_home(self, monkeypatch):
        monkeypatch.delenv("GHOST_HOME", raising=False)
        assert cf.learning_fingerprint() == ""

    async def test_decisive_batch_stamps_gate_and_second_skips(
            self, home, tmp_path):
        self._playbook(home)
        _persist("SUCCESS", challenge="first")
        ctx, _ = _ctx(tmp_path)
        s1 = await cf.run_counterfactual_batch(_FakeDreamer("SUCCESS"), ctx)
        assert s1["replayed"] == 1 and "skipped" not in s1
        # Same learning state + a fresh pending challenge → gate skips.
        _persist("SUCCESS", challenge="second")
        s2 = await cf.run_counterfactual_batch(_FakeDreamer("SUCCESS"), ctx)
        assert s2["replayed"] == 0
        assert "unchanged" in s2.get("skipped", "")

    async def test_learning_change_rearms_gate(self, home, tmp_path):
        self._playbook(home)
        _persist("SUCCESS", challenge="first")
        ctx, _ = _ctx(tmp_path)
        await cf.run_counterfactual_batch(_FakeDreamer("SUCCESS"), ctx)
        _persist("SUCCESS", challenge="second")
        self._playbook(home, '[{"trigger": "fresh lesson"}]')
        s2 = await cf.run_counterfactual_batch(_FakeDreamer("SUCCESS"), ctx)
        assert s2["replayed"] == 1 and "skipped" not in s2

    async def test_kill_switch_disables_gate(self, home, tmp_path,
                                             monkeypatch):
        self._playbook(home)
        monkeypatch.setenv("GHOST_COUNTERFACTUAL_GATE", "0")
        _persist("SUCCESS", challenge="first")
        ctx, _ = _ctx(tmp_path)
        await cf.run_counterfactual_batch(_FakeDreamer("SUCCESS"), ctx)
        _persist("SUCCESS", challenge="second")
        s2 = await cf.run_counterfactual_batch(_FakeDreamer("SUCCESS"), ctx)
        assert s2["replayed"] == 1  # ungated legacy behaviour

    async def test_inconclusive_batch_does_not_stamp(self, home, tmp_path):
        self._playbook(home)
        _persist("SUCCESS", challenge="first")
        ctx, _ = _ctx(tmp_path)
        s1 = await cf.run_counterfactual_batch(
            _FakeDreamer("ABORTED_BY_SOLVER"), ctx)
        assert s1["inconclusive"] == 1
        # Nothing decisive was measured → the retry stays eligible even
        # though the learning state is unchanged.
        s2 = await cf.run_counterfactual_batch(_FakeDreamer("SUCCESS"), ctx)
        assert s2["replayed"] == 1 and "skipped" not in s2


# ── §4JF (2026-09-21): a regression must REPRODUCE; quarantine has a way back ──
#
# Measured: 7 regressions in 416 replays (~2% — the variance band of a 98%-pass
# pool), 32 lessons quarantined by them, none of the 7 challenges ever replayed
# again, no path back. World where these fail: the tree of 2026-09-20.

from ghost_agent.core.autonomous_activity import SEVERITY_INFO


class _SeqDreamer:
    """Replays answer with a scripted SEQUENCE of statuses."""

    def __init__(self, statuses):
        self._seq = list(statuses)
        self.injected = []
        self.last_self_play_status = ""

    async def synthetic_self_play(self, **kw):
        self.injected.append(kw.get("injected_challenge"))
        self.last_self_play_status = self._seq.pop(0) if self._seq else "SUCCESS"


def _rows(home):
    return cf._read_jsonl(home / "system" / "counterfactual" / "results.jsonl")


class TestReproduceBeforeQuarantine:
    async def test_first_failure_is_a_candidate_not_a_regression(self, home, tmp_path):
        _persist("SUCCESS")
        sm = _mem_with_lessons(tmp_path, "use sets for dedup")
        sm.last_playbook_triggers = ["use sets for dedup"]
        ctx, log = _ctx(tmp_path, sm)
        summary = await cf.run_counterfactual_batch(_SeqDreamer(["FAILURE"]), ctx)
        assert summary["regressions"] == 0 and summary.get("candidates") == 1
        assert not summary["quarantined"]
        assert sm._load_playbook()[0].get("quarantined") is not True
        recs, _ = log.read_since(0)
        cand = [r for r in recs if "regression-candidate" in r.summary]
        assert cand and cand[0].severity == SEVERITY_INFO
        assert "replayed again before any lesson is quarantined" in cand[0].summary
        assert _rows(home)[-1]["verdict"] == cf.VERDICT_CANDIDATE

    async def test_a_candidate_goes_first_in_the_next_batch_even_past_the_gate(self, home, tmp_path):
        """The learning-state gate would park the candidate until the next
        lesson landed; a candidate is owed its reproducing replay now."""
        a = _persist("SUCCESS", challenge="first")
        b = _persist("SUCCESS", challenge="second")
        ctx, _ = _ctx(tmp_path, _mem_with_lessons(tmp_path, "x"))
        await cf.run_counterfactual_batch(_SeqDreamer(["FAILURE"]), ctx, limit=1)   # `first` → candidate
        allowed, why = cf.should_replay()
        assert allowed and "candidate" in why
        cands = cf.load_replay_candidates(limit=5)
        assert cands[0]["challenge"] == "first" and cands[1]["challenge"] == "second"

    async def test_a_pass_clears_the_candidate_without_quarantine(self, home, tmp_path):
        _persist("SUCCESS")
        sm = _mem_with_lessons(tmp_path, "use sets for dedup")
        sm.last_playbook_triggers = ["use sets for dedup"]
        ctx, log = _ctx(tmp_path, sm)
        d = _SeqDreamer(["FAILURE", "SUCCESS"])
        await cf.run_counterfactual_batch(d, ctx)
        s2 = await cf.run_counterfactual_batch(d, ctx)
        assert s2["stable"] == 1 and s2["regressions"] == 0
        assert sm._load_playbook()[0].get("quarantined") is not True
        recs, _ = log.read_since(0)
        assert not any(r.severity == SEVERITY_NOTIFY for r in recs)
        # concluded: a third batch has nothing to replay
        assert (await cf.run_counterfactual_batch(d, ctx))["replayed"] == 0

    async def test_the_reproducing_failure_quarantines_and_notifies_once(self, home, tmp_path):
        _persist("SUCCESS")
        sm = _mem_with_lessons(tmp_path, "use sets for dedup")
        sm.last_playbook_triggers = ["use sets for dedup"]
        ctx, log = _ctx(tmp_path, sm)
        d = _SeqDreamer(["FAILURE", "FAILURE"])
        await cf.run_counterfactual_batch(d, ctx)
        s2 = await cf.run_counterfactual_batch(d, ctx)
        assert s2["regressions"] == 1 and s2["quarantined"] == ["use sets for dedup"]
        assert sm._load_playbook()[0].get("quarantined") is True
        recs, _ = log.read_since(0)
        confirmed = [r for r in recs if "reproduced on a second replay" in r.summary]
        assert len(confirmed) == 1 and confirmed[0].severity == SEVERITY_INFO
        assert not any(r.severity == SEVERITY_NOTIFY for r in recs)   # no push, ever


class TestQuarantineHasAWayBack:
    async def test_a_later_pass_lifts_the_quarantine_this_loop_imposed(self, home, tmp_path, monkeypatch):
        # In production the quarantine WRITE changes the playbook fingerprint
        # and opens the learning-state gate for the recheck; this store lives
        # outside the fingerprinted dir, so the gate (out of scope) is off.
        monkeypatch.setenv("GHOST_COUNTERFACTUAL_GATE", "0")
        _persist("SUCCESS")
        sm = _mem_with_lessons(tmp_path, "use sets for dedup", "unrelated lesson")
        sm.last_playbook_triggers = ["use sets for dedup"]
        ctx, log = _ctx(tmp_path, sm)
        d = _SeqDreamer(["FAILURE", "FAILURE", "SUCCESS"])
        await cf.run_counterfactual_batch(d, ctx)
        await cf.run_counterfactual_batch(d, ctx)          # confirmed → quarantined
        by = lambda: {r["trigger"]: r for r in sm._load_playbook()}
        assert by()["use sets for dedup"].get("quarantined") is True
        assert by()["unrelated lesson"].get("quarantined") is not True
        s3 = await cf.run_counterfactual_batch(d, ctx)     # the recheck passes
        assert s3["replayed"] == 1 and s3.get("restored") == ["use sets for dedup"]
        row = by()["use sets for dedup"]
        assert row.get("quarantined") is False and "counterfactual regression" in row["unquarantined_from"]
        assert _rows(home)[-1].get("restored") == ["use sets for dedup"]
        recs, _ = log.read_since(0)
        assert any("quarantine lifted on: use sets for dedup" in r.summary for r in recs)

    async def test_rechecks_are_bounded(self, home, tmp_path, monkeypatch):
        monkeypatch.setenv("GHOST_COUNTERFACTUAL_GATE", "0")
        _persist("SUCCESS")
        sm = _mem_with_lessons(tmp_path, "l")
        sm.last_playbook_triggers = ["l"]
        ctx, _ = _ctx(tmp_path, sm)
        d = _SeqDreamer(["FAILURE"] * 10)
        for _ in range(2 + cf.MAX_REGRESSION_RECHECKS + 3):
            await cf.run_counterfactual_batch(d, ctx)
        verdicts = [r["verdict"] for r in _rows(home)]
        # candidate, confirmed regression, then at most MAX_REGRESSION_RECHECKS rechecks (still failing)
        assert verdicts[:2] == [cf.VERDICT_CANDIDATE, "regression"]
        assert len(verdicts) == 2 + cf.MAX_REGRESSION_RECHECKS
        # nothing from this loop is pushed; failed rechecks say "quarantine stands"
        recs, _ = ctx.activity_log.read_since(0)
        assert not any(r.severity == SEVERITY_NOTIFY for r in recs)
        assert sum(1 for r in recs if "recheck still failing; the quarantine stands" in r.summary) == cf.MAX_REGRESSION_RECHECKS

    def test_unquarantine_is_scoped_to_the_reason_that_imposed_it(self, tmp_path):
        sm = _mem_with_lessons(tmp_path, "a", "b")
        sm.quarantine_lesson("a", reason="counterfactual regression on challenge X: …")
        sm.quarantine_lesson("b", reason="mirror audit: contradicts the profile")
        assert sm.unquarantine_lesson("b", reason_contains="challenge X") == 0   # not this loop's
        assert sm.unquarantine_lesson("a", reason_contains="challenge X") == 1
        rows = {r["trigger"]: r for r in sm._load_playbook()}
        assert rows["a"]["quarantined"] is False and rows["b"]["quarantined"] is True
        assert sm.unquarantine_lesson("a", reason_contains="challenge X") == 0   # idempotent
