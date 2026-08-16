"""Human outcome labels — core/feedback + /api/feedback (Track-1a, 2026-08-13).

The feedback channel turns a 👍/👎 (Slack reaction, web-UI tap) into a
corrections-sidecar record, resolving a turn's ``outcome=unknown`` with the
judgment of the human who read the reply. These tests pin:

  * request-id normalization ("chatcmpl-" wire form → bare req_id);
  * the day-partition trajectory scan (extra.req_id + session_id fallback,
    newest-day-wins, last-match-wins);
  * label semantics (positive → passed with NO reason; negative → failed
    with the note as reason; sidecar source is ``human_feedback:<source>``);
  * the in-process cache stamp (``human_labeled``) and the late-verdict
    backfill YIELDING to it — human labels outrank machine verdicts;
  * the lesson-outcome stash flush riding the human label;
  * the /api/feedback HTTP contract (auth, 400/404/200).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ghost_agent.core.feedback import (
    apply_human_label,
    find_trajectory_for_request,
    normalize_request_id,
)
from ghost_agent.distill.collector import TrajectoryCollector
from ghost_agent.distill.schema import Outcome, Trajectory


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _collector(tmp_path):
    return TrajectoryCollector(root=tmp_path, session_id="t-session")


def _write_traj(collector, req_id, *, traj_id=None, session_id="",
                outcome=Outcome.UNKNOWN.value, extra_req=True):
    t = Trajectory(
        session_id=session_id or req_id,
        user_request="do the thing",
        final_response="done",
        outcome=outcome,
        extra={"req_id": req_id} if extra_req else {},
    )
    if traj_id:
        t.id = traj_id
    assert collector.append(t) is not None
    return t


def _fake_agent(collector, cache=None):
    flush_calls = []

    def _flush(tid, success):
        flush_calls.append((tid, success))

    agent = SimpleNamespace(
        context=SimpleNamespace(
            trajectory_collector=collector,
            _recent_trajectories_for_correction=cache if cache is not None else {},
        ),
        _flush_stashed_lesson_outcome=_flush,
    )
    return agent, flush_calls


# ──────────────────────────────────────────────────────────────────────
# normalize_request_id
# ──────────────────────────────────────────────────────────────────────

class TestNormalize:
    @pytest.mark.parametrize("raw,expected", [
        ("chatcmpl-abc123", "abc123"),
        ("abc123", "abc123"),
        ("  chatcmpl-x  ", "x"),
        ("", ""),
        (None, ""),
    ])
    def test_forms(self, raw, expected):
        assert normalize_request_id(raw) == expected


# ──────────────────────────────────────────────────────────────────────
# find_trajectory_for_request
# ──────────────────────────────────────────────────────────────────────

class TestFind:
    def test_finds_by_extra_req_id(self, tmp_path):
        c = _collector(tmp_path)
        t = _write_traj(c, "r1")
        found = find_trajectory_for_request(c, "r1")
        assert found is not None and found.id == t.id

    def test_finds_by_wire_form(self, tmp_path):
        c = _collector(tmp_path)
        t = _write_traj(c, "r1")
        found = find_trajectory_for_request(c, "chatcmpl-r1")
        assert found is not None and found.id == t.id

    def test_session_id_fallback_for_legacy_records(self, tmp_path):
        c = _collector(tmp_path)
        t = _write_traj(c, "r2", extra_req=False)  # only session_id carries it
        found = find_trajectory_for_request(c, "r2")
        assert found is not None and found.id == t.id

    def test_last_match_wins(self, tmp_path):
        c = _collector(tmp_path)
        _write_traj(c, "r3", traj_id="a" * 32)
        t2 = _write_traj(c, "r3", traj_id="b" * 32)
        found = find_trajectory_for_request(c, "r3")
        assert found.id == t2.id

    def test_miss_returns_none(self, tmp_path):
        c = _collector(tmp_path)
        _write_traj(c, "r1")
        assert find_trajectory_for_request(c, "nope") is None

    def test_none_collector(self):
        assert find_trajectory_for_request(None, "r1") is None

    # ── the day-scan window itself (R1 test review G1: every fixture used
    # to land in TODAY's partition, so the whole multi-day walk was
    # unpinned — `range(1)` stayed green) ─────────────────────────────────

    @staticmethod
    def _write_in_day(root, day_offset, req_id, traj_id=None):
        import datetime as dt
        day = (dt.datetime.utcnow().date()
               - dt.timedelta(days=day_offset)).strftime("%Y-%m-%d")
        t = Trajectory(session_id=req_id, user_request="x",
                       extra={"req_id": req_id})
        if traj_id:
            t.id = traj_id
        d = root / day
        d.mkdir(parents=True, exist_ok=True)
        with (d / "session-t-session.jsonl").open("a", encoding="utf-8") as f:
            f.write(t.to_jsonl() + "\n")
        return t

    def test_finds_in_an_older_day_partition(self, tmp_path):
        # The common real case: the reaction lands after midnight UTC.
        c = _collector(tmp_path)
        t = self._write_in_day(tmp_path, 2, "rOld")
        found = find_trajectory_for_request(c, "rOld")
        assert found is not None and found.id == t.id

    def test_newest_day_wins_across_partitions(self, tmp_path):
        c = _collector(tmp_path)
        self._write_in_day(tmp_path, 2, "rDup", traj_id="a" * 32)
        newer = self._write_in_day(tmp_path, 0, "rDup", traj_id="b" * 32)
        assert find_trajectory_for_request(c, "rDup").id == newer.id

    def test_beyond_scan_window_is_a_miss(self, tmp_path):
        # Pins _SCAN_DAYS = 8: offsets 0..7 are scanned, 8 is out.
        c = _collector(tmp_path)
        self._write_in_day(tmp_path, 8, "rAncient")
        assert find_trajectory_for_request(c, "rAncient") is None

    def test_garbage_day_file_does_not_abort_the_scan(self, tmp_path):
        import datetime as dt
        c = _collector(tmp_path)
        bad_day = (dt.datetime.utcnow().date()
                   - dt.timedelta(days=1)).strftime("%Y-%m-%d")
        (tmp_path / bad_day).mkdir(parents=True, exist_ok=True)
        (tmp_path / bad_day / "session-x.jsonl").write_text(
            "{not json at all\n", encoding="utf-8")
        t = self._write_in_day(tmp_path, 2, "rDeep")
        found = find_trajectory_for_request(c, "rDeep")
        assert found is not None and found.id == t.id


# ──────────────────────────────────────────────────────────────────────
# apply_human_label
# ──────────────────────────────────────────────────────────────────────

class TestApplyLabel:
    def test_positive_labels_passed_with_no_reason(self, tmp_path):
        c = _collector(tmp_path)
        t = _write_traj(c, "r1")
        agent, _ = _fake_agent(c)
        res = apply_human_label(agent, "chatcmpl-r1", "positive",
                                note="great", source="web")
        assert res["ok"] and res["outcome"] == "passed"
        assert res["trajectory_id"] == t.id
        rows = [json.loads(l) for l in
                (tmp_path / "corrections.jsonl").read_text().splitlines()]
        assert rows[-1]["trajectory_id"] == t.id
        assert rows[-1]["outcome"] == "passed"
        assert rows[-1]["reason"] == ""          # passed NEVER carries a reason
        assert rows[-1]["source"] == "human_feedback:web"
        # The overlay serves the label to every corpus reader.
        got = [x for x in c.iter_trajectories() if x.id == t.id]
        assert got[0].outcome == "passed"

    def test_negative_labels_failed_with_note_as_reason(self, tmp_path):
        c = _collector(tmp_path)
        t = _write_traj(c, "r1")
        agent, _ = _fake_agent(c)
        res = apply_human_label(agent, "r1", "negative",
                                note="wrong answer", source="slack:owner")
        assert res["ok"] and res["outcome"] == "failed"
        got = [x for x in c.iter_trajectories() if x.id == t.id]
        assert got[0].outcome == "failed"
        assert got[0].failure_reason == "wrong answer"

    def test_negative_without_note_gets_default_reason(self, tmp_path):
        c = _collector(tmp_path)
        t = _write_traj(c, "r1")
        agent, _ = _fake_agent(c)
        assert apply_human_label(agent, "r1", "negative")["ok"]
        got = [x for x in c.iter_trajectories() if x.id == t.id]
        assert got[0].failure_reason == "human negative feedback"

    def test_invalid_signal_rejected(self, tmp_path):
        agent, _ = _fake_agent(_collector(tmp_path))
        res = apply_human_label(agent, "r1", "meh")
        assert not res["ok"] and "signal" in res["error"]

    def test_missing_request_id_rejected(self, tmp_path):
        agent, _ = _fake_agent(_collector(tmp_path))
        assert not apply_human_label(agent, "", "positive")["ok"]

    def test_unknown_request_id_is_a_miss(self, tmp_path):
        c = _collector(tmp_path)
        _write_traj(c, "r1")
        agent, _ = _fake_agent(c)
        res = apply_human_label(agent, "unknown-req", "positive")
        assert not res["ok"]
        assert res["error"].startswith("no trajectory found")

    def test_unwired_collector(self):
        agent = SimpleNamespace(context=SimpleNamespace(
            trajectory_collector=None))
        assert not apply_human_label(agent, "r1", "positive")["ok"]

    def test_cache_is_mutated_and_stamped_human_labeled(self, tmp_path):
        c = _collector(tmp_path)
        t = _write_traj(c, "r1")
        cached = Trajectory(id=t.id, session_id="r1",
                            outcome=Outcome.UNKNOWN.value,
                            extra={"req_id": "r1"})
        agent, _ = _fake_agent(c, cache={"fp1": cached})
        assert apply_human_label(agent, "r1", "negative", note="bad")["ok"]
        assert cached.outcome == "failed"
        assert cached.failure_reason == "bad"
        assert cached.extra.get("human_labeled") is True

    def test_repeated_identical_label_is_idempotent(self, tmp_path):
        # First live test (2026-08-13): three 👍 in 2s appended three sidecar
        # rows and three INFO lines. A repeat of the current label must be
        # acknowledged (ok + unchanged) WITHOUT a new row.
        c = _collector(tmp_path)
        _write_traj(c, "r1")
        agent, _ = _fake_agent(c)
        r1 = apply_human_label(agent, "r1", "positive", source="web")
        r2 = apply_human_label(agent, "r1", "positive", source="web")
        r3 = apply_human_label(agent, "r1", "positive", source="web")
        assert r1["ok"] and "unchanged" not in r1
        assert r2["ok"] and r2.get("unchanged") is True
        assert r3["ok"] and r3.get("unchanged") is True
        rows = (tmp_path / "corrections.jsonl").read_text().splitlines()
        assert len(rows) == 1

    def test_switching_signal_still_writes(self, tmp_path):
        c = _collector(tmp_path)
        t = _write_traj(c, "r1")
        agent, _ = _fake_agent(c)
        assert "unchanged" not in apply_human_label(agent, "r1", "positive",
                                                    source="web")
        r2 = apply_human_label(agent, "r1", "negative", source="web")
        assert r2["ok"] and "unchanged" not in r2
        rows = (tmp_path / "corrections.jsonl").read_text().splitlines()
        assert len(rows) == 2
        got = [x for x in c.iter_trajectories() if x.id == t.id]
        assert got[0].outcome == "failed"   # last-write-wins preserved

    def test_negative_with_new_note_still_writes(self, tmp_path):
        c = _collector(tmp_path)
        t = _write_traj(c, "r1")
        agent, _ = _fake_agent(c)
        apply_human_label(agent, "r1", "negative", note="reason A",
                          source="web")
        r2 = apply_human_label(agent, "r1", "negative", note="reason B",
                               source="web")
        assert r2["ok"] and "unchanged" not in r2
        assert c.latest_correction(t.id)["reason"] == "reason B"

    def test_latest_correction_helper(self, tmp_path):
        c = _collector(tmp_path)
        t = _write_traj(c, "r1")
        assert c.latest_correction(t.id) is None
        c.update_outcome(t.id, "failed", reason="x", source="s1")
        c.update_outcome(t.id, "passed", source="s2")
        latest = c.latest_correction(t.id)
        assert latest["outcome"] == "passed" and latest["source"] == "s2"
        assert c.latest_correction("nope") is None
        assert c.latest_correction("") is None

    def test_apply_does_not_flush_from_the_worker(self, tmp_path):
        # The flush helper spawns loop-bound background work; calling it
        # from apply_human_label's to_thread worker popped the stash and
        # then LOST the write (R1 review). The route flushes on the loop —
        # pinned in TestFeedbackEndpoint — and this layer must not.
        c = _collector(tmp_path)
        _write_traj(c, "r1")
        agent, flushes = _fake_agent(c)
        assert apply_human_label(agent, "r1", "positive")["ok"]
        assert flushes == []

    def test_second_source_still_writes(self, tmp_path):
        # Same signal from a DIFFERENT source is a real event, not a repeat
        # (R1 test review: deleting the source half of the idempotency
        # check was green).
        c = _collector(tmp_path)
        t = _write_traj(c, "r1")
        agent, _ = _fake_agent(c)
        apply_human_label(agent, "r1", "positive", source="slack:owner")
        r2 = apply_human_label(agent, "r1", "positive", source="web")
        assert r2["ok"] and "unchanged" not in r2
        rows = (tmp_path / "corrections.jsonl").read_text().splitlines()
        assert len(rows) == 2

    def test_duck_typed_collector_without_redaction_still_labels(
            self, tmp_path):
        # R2 moved the reason comparison INSIDE update_outcome (atomic
        # dedupe), so apply_human_label itself must work against a
        # duck-typed collector that lacks `.redaction` (the codebase does
        # pass fakes) — only the stream-log line degrades gracefully.
        c = _collector(tmp_path)
        _write_traj(c, "r1")

        class _NoRedaction:
            def __init__(self, real):
                self._real = real

            @property
            def redaction(self):
                raise RuntimeError("duck-typed collector without redaction")

            def update_outcome(self, *a, **k):
                return self._real.update_outcome(*a, **k)

            def latest_correction(self, tid):
                return self._real.latest_correction(tid)

            def iter_trajectories(self, **k):
                return self._real.iter_trajectories(**k)

        wrapped = _NoRedaction(c)
        agent = SimpleNamespace(context=SimpleNamespace(
            trajectory_collector=wrapped,
            _recent_trajectories_for_correction={}))
        apply_human_label(agent, "r1", "negative", note="first", source="web")
        r2 = apply_human_label(agent, "r1", "negative", note="second",
                               source="web")
        assert r2["ok"] and "unchanged" not in r2
        rows = (tmp_path / "corrections.jsonl").read_text().splitlines()
        assert len(rows) == 2

    def test_positive_reason_cleared_at_this_layer(self, tmp_path):
        # FG2 (R1 test review): the collector ALSO enforces passed-has-no-
        # reason, so the disk assertion couldn't attribute the guarantee to
        # feedback.py. Pin the kwargs this layer actually sends.
        sent = {}

        class _Rec:
            def update_outcome(self, tid, outcome, reason="", source="",
                               **kw):
                sent.update(tid=tid, outcome=outcome, reason=reason,
                            source=source)
                return True

            def latest_correction(self, tid):
                return None
        rec = _Rec()
        traj = Trajectory(id="t" * 32, session_id="r1",
                          extra={"req_id": "r1"})
        import ghost_agent.core.feedback as fb
        agent = SimpleNamespace(context=SimpleNamespace(
            trajectory_collector=rec,
            _recent_trajectories_for_correction={}))
        orig = fb.find_trajectory_for_request
        fb.find_trajectory_for_request = lambda c, r, max_days=8: traj
        try:
            res = apply_human_label(agent, "r1", "positive",
                                    note="nice work!")
        finally:
            fb.find_trajectory_for_request = orig
        assert res["ok"]
        assert sent["reason"] == ""            # note NEVER rides a positive
        assert sent["source"] == "human_feedback:api"   # default source


# ──────────────────────────────────────────────────────────────────────
# Writer-side authority: a machine correction cannot supersede a human one
# ──────────────────────────────────────────────────────────────────────

class TestWriterYieldsToHuman:
    def test_machine_write_withheld_over_a_human_record(self, tmp_path):
        # The race-proof half (R1 review): the late-verdict write is
        # DEFERRED through a background thread, so it can be checked
        # before the human label exists yet land after it — only the
        # writer, inside its lock, can close that. The DISTINCT sentinel
        # (R4) lets callers tell "human won" from a plain write failure.
        c = _collector(tmp_path)
        t = _write_traj(c, "r1")
        assert c.update_outcome(t.id, "passed", source="human_feedback:web")
        assert c.update_outcome(t.id, "failed", reason="late refute",
                                source="verifier_late",
                                yield_to_human=True) == "withheld"
        latest = c.latest_correction(t.id)
        assert latest["outcome"] == "passed"
        assert latest["source"] == "human_feedback:web"

    def test_machine_write_lands_when_no_human_record(self, tmp_path):
        c = _collector(tmp_path)
        t = _write_traj(c, "r1")
        assert c.update_outcome(t.id, "failed", reason="late refute",
                                source="verifier_late",
                                yield_to_human=True) is True

    def test_machine_write_supersedes_an_older_machine_record(self, tmp_path):
        c = _collector(tmp_path)
        t = _write_traj(c, "r1")
        assert c.update_outcome(t.id, "failed", reason="r",
                                source="verifier_late", yield_to_human=True)
        assert c.update_outcome(t.id, "passed",
                                source="verifier_late", yield_to_human=True)
        assert c.latest_correction(t.id)["outcome"] == "passed"

    def test_human_over_human_stays_last_write_wins(self, tmp_path):
        c = _collector(tmp_path)
        t = _write_traj(c, "r1")
        assert c.update_outcome(t.id, "passed", source="human_feedback:web")
        assert c.update_outcome(t.id, "failed", reason="user pushback",
                                source="user_correction")
        assert c.latest_correction(t.id)["outcome"] == "failed"


class TestWriterSkipIdentical:
    def test_identical_repeat_returns_unchanged_without_a_row(self, tmp_path):
        # R2 review: dedupe must live INSIDE the writer lock — a
        # caller-side compare-then-write races N concurrent labels.
        c = _collector(tmp_path)
        t = _write_traj(c, "r1")
        assert c.update_outcome(t.id, "passed",
                                source="human_feedback:web",
                                skip_identical=True) is True
        assert c.update_outcome(t.id, "passed",
                                source="human_feedback:web",
                                skip_identical=True) == "unchanged"
        rows = (tmp_path / "corrections.jsonl").read_text().splitlines()
        assert len(rows) == 1

    def test_different_reason_still_writes(self, tmp_path):
        c = _collector(tmp_path)
        t = _write_traj(c, "r1")
        c.update_outcome(t.id, "failed", reason="a",
                         source="human_feedback:web", skip_identical=True)
        assert c.update_outcome(t.id, "failed", reason="b",
                                source="human_feedback:web",
                                skip_identical=True) is True
        assert c.latest_correction(t.id)["reason"] == "b"

    def test_memoized_corrections_see_new_writes(self, tmp_path):
        # _load_corrections is memoized on (size, mtime_ns); every append
        # changes the size, so readers must never see a stale snapshot.
        c = _collector(tmp_path)
        t = _write_traj(c, "r1")
        assert c.latest_correction(t.id) is None
        c.update_outcome(t.id, "failed", reason="x", source="s1")
        assert c.latest_correction(t.id)["outcome"] == "failed"
        c.update_outcome(t.id, "passed", source="s2")
        assert c.latest_correction(t.id)["outcome"] == "passed"


class TestFlushRebook:
    """R2 review: the stash pops once, so the FIRST resolver consumed the
    trigger set and a later opposite-sign human label vanished — lesson
    counters kept the machine's sign for a turn the human overruled."""

    def _agent(self):
        from ghost_agent.core.agent import GhostAgent
        from collections import OrderedDict
        booked = []

        class _SM:
            def record_surfaced_outcomes(self, triggers, success):
                booked.append((tuple(triggers), success))

        fake = SimpleNamespace(
            _FLUSHED_TRIG_RETAIN_MAX=GhostAgent._FLUSHED_TRIG_RETAIN_MAX,
            context=SimpleNamespace(
                _surfaced_triggers_by_traj=OrderedDict(),
                skill_memory=_SM()))
        flush = GhostAgent._flush_stashed_lesson_outcome.__get__(fake)
        return fake, flush, booked

    def test_opposite_sign_rebooks_from_the_retained_set(self):
        fake, flush, booked = self._agent()
        fake.context._surfaced_triggers_by_traj["t1"] = ["lesson-a"]

        async def go():
            flush("t1", False)   # machine refute books the failure
            flush("t1", False)   # same sign again → no-op
            flush("t1", True)    # human 👍 → compensating success booking
            await asyncio.sleep(0.05)

        asyncio.run(go())
        assert booked == [(("lesson-a",), False), (("lesson-a",), True)]

    def test_never_flushed_stays_a_noop(self):
        fake, flush, booked = self._agent()

        async def go():
            flush("missing", True)
            await asyncio.sleep(0.02)

        asyncio.run(go())
        assert booked == []


# ──────────────────────────────────────────────────────────────────────
# The late-verdict backfill YIELDS to a human label
# ──────────────────────────────────────────────────────────────────────

class TestHumanLabelWins:
    def _fake_self(self, collector, cached):
        flushes = []
        update_calls = []

        class _Rec:
            enabled = True

            def update_outcome(self, tid, outcome, reason="", source="",
                               **kw):
                update_calls.append((tid, outcome, source))
                return True

        rec = _Rec()
        rec.iter_trajectories = collector.iter_trajectories
        fake = SimpleNamespace(
            context=SimpleNamespace(
                trajectory_collector=rec,
                _recent_trajectories_for_correction={"fp": cached},
                calibration_tracker=None,
            ),
            _flush_stashed_lesson_outcome=lambda tid, ok: flushes.append(
                (tid, ok)),
        )
        return fake, flushes, update_calls

    @staticmethod
    async def _drain(update_calls, expect):
        # Deterministic wait (R1 test review H3): the write lands via
        # spawn_bg(to_thread(...)); poll instead of racing a fixed sleep.
        for _ in range(200):
            if len(update_calls) >= expect:
                return
            await asyncio.sleep(0.01)

    def test_late_verdict_withheld_after_human_label(self, tmp_path):
        from ghost_agent.core.agent import GhostAgent
        c = _collector(tmp_path)
        t = _write_traj(c, "r1")
        cached = Trajectory(id=t.id, session_id="r1", outcome="failed",
                            failure_reason="bad",
                            extra={"req_id": "r1", "human_labeled": True})
        fake, flushes, update_calls = self._fake_self(c, cached)

        async def go():
            # NOTE: _backfill_trajectory_outcome is SYNC — asyncio.run
            # exists only to provide the loop spawn_bg needs. Do not "fix"
            # this into an await.
            GhostAgent._backfill_trajectory_outcome(fake, t.id, "passed")
            await asyncio.sleep(0.05)

        asyncio.run(go())
        assert update_calls == []      # no sidecar overwrite
        assert flushes == []           # already flushed by the label
        assert cached.outcome == "failed"  # the human verdict stands

    def test_late_refuted_cannot_overwrite_a_human_thumbs_up(self, tmp_path):
        # THE load-bearing direction (R1 test review G2): FAILED "always
        # lands" in the resolve ladder, so the guard is the SOLE protection
        # when a human 👍 meets a late machine REFUTED — the other
        # direction was already blocked by the ladder, making the original
        # test non-discriminating.
        from ghost_agent.core.agent import GhostAgent
        c = _collector(tmp_path)
        t = _write_traj(c, "r1")
        cached = Trajectory(id=t.id, session_id="r1", outcome="passed",
                            extra={"req_id": "r1", "human_labeled": True})
        fake, flushes, update_calls = self._fake_self(c, cached)

        async def go():
            GhostAgent._backfill_trajectory_outcome(fake, t.id, "failed",
                                                    "verifier says no")
            await asyncio.sleep(0.05)

        asyncio.run(go())
        assert update_calls == []
        assert cached.outcome == "passed"
        assert flushes == []

    def test_late_verdict_proceeds_without_human_label(self, tmp_path):
        from ghost_agent.core.agent import GhostAgent
        c = _collector(tmp_path)
        t = _write_traj(c, "r1")
        cached = Trajectory(id=t.id, session_id="r1",
                            outcome=Outcome.UNKNOWN.value,
                            extra={"req_id": "r1"})
        fake, flushes, update_calls = self._fake_self(c, cached)

        async def go():
            GhostAgent._backfill_trajectory_outcome(fake, t.id, "failed",
                                                    "verifier says no")
            await self._drain(update_calls, 1)

        asyncio.run(go())
        # Control: without the stamp the backfill still does its job —
        # the guard must not over-block (the §4AN over-removal lesson).
        assert flushes == [(t.id, False)]
        assert update_calls and update_calls[0][1] == "failed"

    def test_integration_label_then_backfill_one_cache(self, tmp_path):
        # FG3 (R1 test review): both halves hard-coded the "human_labeled"
        # literal on their own side — wire them together for real.
        from ghost_agent.core.agent import GhostAgent
        c = _collector(tmp_path)
        t = _write_traj(c, "r1")
        cached = Trajectory(id=t.id, session_id="r1",
                            outcome=Outcome.UNKNOWN.value,
                            extra={"req_id": "r1"})
        cache = {"fp": cached}
        agent = SimpleNamespace(context=SimpleNamespace(
            trajectory_collector=c,
            _recent_trajectories_for_correction=cache))
        assert apply_human_label(agent, "r1", "positive")["ok"]

        fake, flushes, update_calls = self._fake_self(c, cached)
        fake.context._recent_trajectories_for_correction = cache

        async def go():
            GhostAgent._backfill_trajectory_outcome(fake, t.id, "failed",
                                                    "late refute")
            await asyncio.sleep(0.05)

        asyncio.run(go())
        assert update_calls == []          # the real stamp blocked it
        latest = c.latest_correction(t.id)
        assert latest["source"].startswith("human_feedback")
        assert latest["outcome"] == "passed"


class TestWholeChainGuard:
    """_record_late_verdict must yield ENTIRELY to a human label — not just
    the sidecar write (R1 review: the stream re-render, lesson retraction,
    follow-up filing, and the next-turn banner all ran anyway)."""

    def _fake_self(self, cached, locked=True):
        calls = []
        fake = SimpleNamespace(
            context=SimpleNamespace(
                _recent_trajectories_for_correction={"fp": cached},
                skill_memory=None,
                trajectory_collector=None,
            ),
            _backfill_trajectory_outcome=lambda *a, **k: calls.append(
                ("backfill", a)),
            _emit_late_outcome_correction=lambda *a, **k: calls.append(
                ("emit", a)),
            _file_refute_followup_tasks=lambda *a, **k: calls.append(
                ("followups", a)),
            _critic_async_enabled=lambda: False,
        )
        from ghost_agent.core.agent import GhostAgent
        fake._human_label_locked = GhostAgent._human_label_locked.__get__(fake)
        return fake, calls

    def _refuted(self):
        from ghost_agent.core.verifier import VerifyVerdict
        return SimpleNamespace(verdict=VerifyVerdict.REFUTED,
                               confidence=0.9, issues=["wrong number"],
                               reasoning="not supported")

    def test_whole_chain_withheld_when_human_labeled(self):
        from ghost_agent.core.agent import GhostAgent
        cached = Trajectory(id="c" * 32, session_id="r1", outcome="passed",
                            extra={"req_id": "r1", "human_labeled": True})
        fake, calls = self._fake_self(cached)
        GhostAgent._record_late_verdict(fake, self._refuted(), "c" * 32)
        assert calls == []   # no backfill, no re-render, no follow-ups

    def test_whole_chain_proceeds_without_the_stamp(self):
        # Over-removal guard (§4AN lesson): the gate must not block the
        # normal late-verdict machinery.
        from ghost_agent.core.agent import GhostAgent
        cached = Trajectory(id="c" * 32, session_id="r1",
                            outcome=Outcome.UNKNOWN.value,
                            extra={"req_id": "r1"})
        fake, calls = self._fake_self(cached, locked=False)
        GhostAgent._record_late_verdict(fake, self._refuted(), "c" * 32)
        kinds = [k for k, _ in calls]
        assert "backfill" in kinds and "emit" in kinds

    def test_human_label_locked_helper(self):
        from ghost_agent.core.agent import GhostAgent
        stamped = Trajectory(id="a" * 32, extra={"human_labeled": True})
        plain = Trajectory(id="b" * 32, extra={})
        fake = SimpleNamespace(context=SimpleNamespace(
            _recent_trajectories_for_correction={"x": stamped, "y": plain},
            trajectory_collector=None))
        locked = GhostAgent._human_label_locked.__get__(fake)
        assert locked("a" * 32) is True
        assert locked("b" * 32) is False
        assert locked("missing") is False

    def test_human_label_locked_falls_back_to_the_sidecar(self, tmp_path):
        # R3 review: the 32-entry cache is process-wide and multi-user —
        # a label whose trajectory was EVICTED stamped nothing and the
        # whole consequence chain ran against a human-labeled turn. The
        # sidecar is the durable truth.
        from ghost_agent.core.agent import GhostAgent
        c = _collector(tmp_path)
        t = _write_traj(c, "r1")
        c.update_outcome(t.id, "passed", source="human_feedback:web")
        fake = SimpleNamespace(context=SimpleNamespace(
            _recent_trajectories_for_correction={},   # evicted
            trajectory_collector=c))
        locked = GhostAgent._human_label_locked.__get__(fake)
        assert locked(t.id) is True
        # A machine-sourced record does NOT lock.
        t2 = _write_traj(c, "r2")
        c.update_outcome(t2.id, "failed", reason="x", source="verifier_late")
        assert locked(t2.id) is False

    def test_withheld_write_revokes_the_banner(self, tmp_path):
        # R3 review: when the human label lands in the DEFERRAL window,
        # the banner was enqueued after the route's revoke ran — the
        # withheld-write callback must drop it (and never book calibration
        # for a verdict the corpus refused).
        from ghost_agent.core.agent import GhostAgent
        c = _collector(tmp_path)
        t = _write_traj(c, "r1")
        cached = Trajectory(id=t.id, session_id="r1",
                            outcome=Outcome.UNKNOWN.value,
                            extra={"req_id": "r1"})
        drops = []
        calib = []
        update_calls = []

        class _RefusingRec:
            enabled = True
            iter_trajectories = c.iter_trajectories

            def update_outcome(self, *a, **k):
                update_calls.append(a)
                return "withheld"    # a human label stands on disk

        fake = SimpleNamespace(
            context=SimpleNamespace(
                trajectory_collector=_RefusingRec(),
                _recent_trajectories_for_correction={"fp": cached},
                calibration_tracker=SimpleNamespace(
                    record_late_verdict_correction=lambda rid, v:
                        calib.append((rid, v))),
            ),
            _flush_stashed_lesson_outcome=lambda tid, ok: None,
            _drop_pending_corrections_for=lambda tid: drops.append(tid),
        )

        async def go():
            GhostAgent._backfill_trajectory_outcome(fake, t.id, "failed",
                                                    "late refute")
            await asyncio.sleep(0.05)

        asyncio.run(go())
        assert drops == [t.id]
        assert calib == []                       # no supersession booked
        assert cached.outcome == Outcome.UNKNOWN.value   # cache untouched

    def test_plain_write_failure_does_not_revoke_the_banner(self, tmp_path):
        # R4: a disk error (False) must NOT be conflated with "a human
        # label won" ("withheld") — revoking the banner on ENOSPC would
        # silently un-tell the user about a real refute.
        from ghost_agent.core.agent import GhostAgent
        c = _collector(tmp_path)
        t = _write_traj(c, "r1")
        cached = Trajectory(id=t.id, session_id="r1",
                            outcome=Outcome.UNKNOWN.value,
                            extra={"req_id": "r1"})
        drops = []

        class _BrokenRec:
            enabled = True
            iter_trajectories = c.iter_trajectories

            def update_outcome(self, *a, **k):
                return False    # genuine write failure

        fake = SimpleNamespace(
            context=SimpleNamespace(
                trajectory_collector=_BrokenRec(),
                _recent_trajectories_for_correction={"fp": cached},
                calibration_tracker=None,
            ),
            _flush_stashed_lesson_outcome=lambda tid, ok: None,
            _drop_pending_corrections_for=lambda tid: drops.append(tid),
        )

        async def go():
            GhostAgent._backfill_trajectory_outcome(fake, t.id, "failed",
                                                    "late refute")
            await asyncio.sleep(0.05)

        asyncio.run(go())
        assert drops == []                       # banner stands
        assert cached.outcome == Outcome.UNKNOWN.value


# ──────────────────────────────────────────────────────────────────────
# /api/feedback HTTP contract
# ──────────────────────────────────────────────────────────────────────

class TestFeedbackEndpoint:
    def _make_app(self, collector):
        fastapi = pytest.importorskip("fastapi")
        from fastapi import FastAPI
        from ghost_agent.api.routes import router
        app = FastAPI()
        app.include_router(router)
        agent = MagicMock()
        agent.context = MagicMock()
        agent.context.args.api_key = "test-key"
        agent.context.trajectory_collector = collector
        agent.context._recent_trajectories_for_correction = {}
        app.state.agent = agent
        return app

    def _client(self, app):
        from fastapi.testclient import TestClient
        return TestClient(app)

    HDRS = {"X-Ghost-Key": "test-key"}

    def test_requires_api_key(self, tmp_path):
        app = self._make_app(_collector(tmp_path))
        with self._client(app) as c:
            r = c.post("/api/feedback",
                       json={"request_id": "r1", "signal": "positive"})
            assert r.status_code in (401, 403)

    def test_missing_request_id_is_400(self, tmp_path):
        app = self._make_app(_collector(tmp_path))
        with self._client(app) as c:
            r = c.post("/api/feedback", headers=self.HDRS,
                       json={"signal": "positive"})
            assert r.status_code == 400

    def test_bad_signal_is_400(self, tmp_path):
        app = self._make_app(_collector(tmp_path))
        with self._client(app) as c:
            r = c.post("/api/feedback", headers=self.HDRS,
                       json={"request_id": "r1", "signal": "sideways"})
            assert r.status_code == 400

    def test_unknown_request_id_is_404(self, tmp_path):
        app = self._make_app(_collector(tmp_path))
        with self._client(app) as c:
            r = c.post("/api/feedback", headers=self.HDRS,
                       json={"request_id": "ghost", "signal": "positive"})
            assert r.status_code == 404

    def test_flush_happens_on_the_route_not_the_worker(self, tmp_path):
        # The route flushes the lesson-outcome stash ON the event loop
        # after apply_human_label returns (R1 review: the worker-thread
        # flush popped the stash then lost the loop-bound write). Called on
        # EVERY ok — including idempotent repeats (R2: the flush is
        # sign-aware/no-op-safe, and repeating heals a first attempt whose
        # post-write path failed) — and a human label revokes any queued
        # machine correction banner for the turn.
        c_ = _collector(tmp_path)
        t = _write_traj(c_, "r9")
        app = self._make_app(c_)
        agent = app.state.agent
        with self._client(app) as c:
            r = c.post("/api/feedback", headers=self.HDRS,
                       json={"request_id": "r9", "signal": "negative"})
            assert r.status_code == 200
            agent._flush_stashed_lesson_outcome.assert_called_with(
                t.id, False)
            agent._drop_pending_corrections_for.assert_called_with(t.id)
            r2 = c.post("/api/feedback", headers=self.HDRS,
                        json={"request_id": "r9", "signal": "negative"})
            assert r2.status_code == 200 and r2.json().get("unchanged")
            assert agent._flush_stashed_lesson_outcome.call_count == 2

    def test_sidecar_write_failure_is_503(self, tmp_path, monkeypatch):
        c_ = _collector(tmp_path)
        _write_traj(c_, "r9")
        monkeypatch.setattr(c_, "update_outcome", lambda *a, **k: False)
        app = self._make_app(c_)
        with self._client(app) as c:
            r = c.post("/api/feedback", headers=self.HDRS,
                       json={"request_id": "r9", "signal": "positive"})
            assert r.status_code == 503
            body = r.json()
            assert body["ok"] is False and body["code"] == "unavailable"

    def test_error_bodies_are_uniform_with_codes(self, tmp_path):
        # Status comes from the machine-readable `code`, and EVERY error
        # body carries {ok, error, code} — a client reading `error` must
        # never get None just because the status was 400 (R1 review: the
        # 400s used to ship FastAPI's {"detail": ...} shape, and the 404
        # was selected by matching the error PROSE).
        app = self._make_app(_collector(tmp_path))
        with self._client(app) as c:
            r400 = c.post("/api/feedback", headers=self.HDRS,
                          json={"signal": "positive"})
            assert r400.status_code == 400
            assert r400.json()["code"] == "bad_request"
            assert r400.json()["error"]
            r404 = c.post("/api/feedback", headers=self.HDRS,
                          json={"request_id": "ghost", "signal": "positive"})
            assert r404.status_code == 404
            assert r404.json()["code"] == "not_found"

    def test_label_lands_in_sidecar(self, tmp_path):
        c_ = _collector(tmp_path)
        t = _write_traj(c_, "r9")
        app = self._make_app(c_)
        with self._client(app) as c:
            r = c.post("/api/feedback", headers=self.HDRS,
                       json={"request_id": "chatcmpl-r9",
                             "signal": "negative", "note": "nope",
                             "source": "web"})
            assert r.status_code == 200
            body = r.json()
            assert body["ok"] and body["trajectory_id"] == t.id
        got = [x for x in c_.iter_trajectories() if x.id == t.id]
        assert got[0].outcome == "failed"
        assert got[0].failure_reason == "nope"
