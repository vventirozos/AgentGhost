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


# ── Feedback-channel fresh-eye review, 2026-08-17 ──────────────────────────
# R1 ran 51 mutations: 37 killed, 14 survived. These pin the survivors. The
# theme is the project's standing one — the guards with the most careful
# rationale comments were the ones nothing tested.

class TestProvenanceReachesTheAnalysis:
    """§4BR's decision rule accepts ONLY human feedback as a correctness
    signal and explicitly disowns the plain outcome label as circular. The
    overlay dropped `source`, so a human 👎 and a `verifier_late` machine
    verdict were byte-identical to the analysis — the pre-registered metric
    was unmeasurable by the instrument that names it, while the report
    rendered the forbidden circular one and looked right."""

    def _corpus(self, tmp_path, source):
        from ghost_agent.distill.collector import TrajectoryCollector
        from ghost_agent.distill.schema import Trajectory
        c = TrajectoryCollector(root=tmp_path / "trajectories", session_id="s")
        t = Trajectory(user_request="q", final_response="a",
                       task_kind="user_request",
                       extra={"experiments": {"verify_depth": "treatment"}})
        t.outcome = "unknown"
        c.append(t)
        assert c.update_outcome(t.id, "failed", reason="bad", source=source)
        return c

    def test_the_overlay_stamps_outcome_source(self, tmp_path):
        c = self._corpus(tmp_path, "human_feedback:slack:owner")
        traj = next(iter(c.iter_trajectories()))
        assert traj.extra.get("outcome_source") == "human_feedback:slack:owner"

    def test_a_human_label_is_DISTINGUISHABLE_from_a_machine_verdict(
            self, tmp_path):
        from ghost_agent.core.experiments import _metric_values
        human = next(iter(self._corpus(tmp_path / "h",
                                       "human_feedback:web").iter_trajectories()))
        machine = next(iter(self._corpus(tmp_path / "m",
                                         "verifier_late").iter_trajectories()))
        hm, mm = _metric_values(human), _metric_values(machine)
        # Both carry the circular metric...
        assert hm["failure_rate"] == 1.0 and mm["failure_rate"] == 1.0
        # ...and ONLY the human one carries the §4BR metric.
        assert hm["human_failure_rate"] == 1.0
        assert "human_failure_rate" not in mm
        assert hm["human_label_rate"] == 1.0
        assert mm["human_label_rate"] == 0.0

    def test_the_metric_is_registered_so_the_report_renders_it(self):
        from ghost_agent.core import experiments as ex
        names = [m for m, _ in ex._METRICS]
        assert "human_failure_rate" in names
        assert "human_label_rate" in names

    def test_the_descriptive_rate_does_not_consume_alpha(self):
        """A denominator must not widen every real interval to pay for a
        diagnostic.

        ⚠ The first version re-derived the count with the implementation's
        own expression, so it was a tautology: adding §4BR's PRIMARY metric
        to `_DESCRIPTIVE_METRICS` — exempting it from Bonferroni entirely —
        left the suite green. Literal count, plus an explicit floor on
        which metrics may be exempt.
        """
        from ghost_agent.core import experiments as ex
        assert ex._DESCRIPTIVE_METRICS == frozenset({"human_label_rate"}), (
            "only the denominator may skip the alpha split")
        assert ex._tested_metric_count() == 4, (
            "4 tested metrics: failure_rate, human_failure_rate, n_steps, "
            "duration_s")
        # The primary metric must be TESTED, never exempt.
        assert "human_failure_rate" not in ex._DESCRIPTIVE_METRICS

    def test_the_primary_metric_inherits_the_attrition_confound(self):
        """R2 MAJOR-4: `human_failure_rate` conditions on outcome-resolved
        AND human-labelled, so differential attrition applies a fortiori —
        but only `failure_rate` was annotated."""
        from ghost_agent.core import experiments as ex
        assert "human_failure_rate" in ex._ATTRITION_SENSITIVE
        assert "failure_rate" in ex._ATTRITION_SENSITIVE

    def test_the_descriptive_rate_never_announces_a_verdict(self):
        """R2 MAJOR-5: excluding it from the alpha split did not stop it
        emitting a DECIDED verdict and pushing it to the operator ledger.
        With lower_is_better=False, "more thumbs in treatment" announced as
        a quality win — and it is the first human metric to clear the
        n>=30/arm floor on live data, so it would have been the operator's
        FIRST verdict on the arm."""
        from ghost_agent.core import experiments as ex
        from ghost_agent.distill.collector import Trajectory

        def _t(arm, human):
            e = {ex.EXTRA_KEY: {"exp1": arm}}
            if human:
                e["outcome_source"] = "human_feedback:web"
            tr = Trajectory(user_request="q", final_response="a",
                            task_kind="user_request", extra=e)
            tr.outcome = "passed"
            return tr

        trajs = ([_t("control", i < 10) for i in range(60)]
                 + [_t("treatment", i < 50) for i in range(60)])
        summary = ex.summarize_trajectories(trajs)
        cmp_ = next(c for c in ex.compare_arms(summary["exp1"])
                    if c.metric == "human_label_rate")
        assert cmp_.confound, "a descriptive rate must carry a caveat"
        assert "DESCRIPTIVE" in cmp_.confound
        pend = ex.pending_announcements(
            summary, ex.summarize_trajectories(trajs, triggered_only=True))
        assert not any("human_label_rate" in str(p) for p in pend), (
            "the denominator announced as a result")

    def test_the_prefix_matches_the_writers(self):
        """Two modules spell this constant; a rename in one silently
        disarms the metric."""
        from ghost_agent.core.experiments import _HUMAN_SOURCE_PREFIX
        from ghost_agent.distill.collector import HUMAN_SOURCE_PREFIX
        from ghost_agent.core.feedback import _SOURCE_PREFIX
        assert _HUMAN_SOURCE_PREFIX == HUMAN_SOURCE_PREFIX == _SOURCE_PREFIX


class TestEverHumanNotLatestHuman:
    """C10: both the writer guard and `_human_label_locked` tested only the
    LATEST sidecar record, so any non-human row in between disarmed the
    human protection and a machine verdict overwrote a human-labelled turn."""

    def test_an_intermediate_correction_does_not_disarm_the_guard(self, tmp_path):
        from ghost_agent.distill.collector import TrajectoryCollector
        from ghost_agent.distill.schema import Trajectory
        c = TrajectoryCollector(root=tmp_path / "t", session_id="s")
        t = Trajectory(user_request="q", final_response="a",
                       task_kind="user_request")
        c.append(t)
        assert c.update_outcome(t.id, "passed",
                               source="human_feedback:slack:owner")
        # A non-human row lands between the human label and the late verdict.
        assert c.update_outcome(t.id, "failed", reason="user said no",
                               source="user_correction")
        # The machine verdict must STILL be withheld.
        assert c.update_outcome(t.id, "passed", source="verifier_late",
                               yield_to_human=True) == "withheld"

    def test_has_human_label_sees_history_not_just_the_top(self, tmp_path):
        from ghost_agent.distill.collector import TrajectoryCollector
        from ghost_agent.distill.schema import Trajectory
        c = TrajectoryCollector(root=tmp_path / "t", session_id="s")
        t = Trajectory(user_request="q", final_response="a",
                       task_kind="user_request")
        c.append(t)
        assert c.has_human_label(t.id) is False
        c.update_outcome(t.id, "passed", source="human_feedback:web")
        assert c.has_human_label(t.id) is True
        c.update_outcome(t.id, "failed", source="operator_overlay")
        assert c.has_human_label(t.id) is True, "history must survive"
        assert c.has_human_label("") is False


class TestOneHumanOverturningAnotherIsNotSILENT:
    """C5, with two live instances: in open-channel mode `slack:requester`
    is any team member, and a colleague's 👎 replaced the operator's 👍 3.5h
    later with nothing logged. Last-write-wins STANDS (operator doctrine) —
    it just must not be invisible when the turn is experiment evidence."""

    def test_a_different_human_source_flipping_the_label_warns(
            self, tmp_path, caplog):
        import logging
        from ghost_agent.distill.collector import TrajectoryCollector
        from ghost_agent.distill.schema import Trajectory
        c = TrajectoryCollector(root=tmp_path / "t", session_id="s")
        t = Trajectory(user_request="q", final_response="a",
                       task_kind="user_request")
        c.append(t)
        c.update_outcome(t.id, "passed", source="human_feedback:slack:owner")
        with caplog.at_level(logging.DEBUG):
            c.update_outcome(t.id, "failed", reason="no",
                             source="human_feedback:slack:requester")
        hits = [r for r in caplog.records if "CHANGED by a different" in r.getMessage()]
        assert hits, "the operator's label was replaced with no record"
        assert hits[0].levelno >= logging.WARNING

    def test_the_SAME_human_changing_their_mind_is_quiet(self, tmp_path, caplog):
        """Not news: one person switching their own thumb."""
        import logging
        from ghost_agent.distill.collector import TrajectoryCollector
        from ghost_agent.distill.schema import Trajectory
        c = TrajectoryCollector(root=tmp_path / "t", session_id="s")
        t = Trajectory(user_request="q", final_response="a",
                       task_kind="user_request")
        c.append(t)
        c.update_outcome(t.id, "passed", source="human_feedback:web")
        with caplog.at_level(logging.DEBUG):
            c.update_outcome(t.id, "failed", reason="no",
                             source="human_feedback:web")
        assert not [r for r in caplog.records
                    if "CHANGED by a different" in r.getMessage()]


class TestTheLossPathsAreVISIBLE:
    """C3: every refusal returned a code and logged NOTHING. The live agent
    runs at INFO (the launcher passes no --debug), so a DEBUG diagnostic is
    no diagnostic. At the scarce qualifying-label rate each lost label costs days
    of §4BR's clock."""

    def _agent(self, tmp_path, with_collector=True):
        from types import SimpleNamespace
        from ghost_agent.distill.collector import TrajectoryCollector
        coll = TrajectoryCollector(root=tmp_path / "t",
                                   session_id="s") if with_collector else None
        return SimpleNamespace(context=SimpleNamespace(
            trajectory_collector=coll))

    def test_a_missing_trajectory_logs_a_warning(self, tmp_path, caplog):
        import logging
        from ghost_agent.core.feedback import apply_human_label
        with caplog.at_level(logging.DEBUG):
            res = apply_human_label(self._agent(tmp_path), "req-nope",
                                    "positive")
        assert res["code"] == "not_found"
        hits = [r for r in caplog.records if "NOT RECORDED" in r.getMessage()]
        assert hits and hits[0].levelno >= logging.WARNING

    def test_a_bad_signal_logs_a_warning(self, tmp_path, caplog):
        import logging
        from ghost_agent.core.feedback import apply_human_label
        with caplog.at_level(logging.DEBUG):
            res = apply_human_label(self._agent(tmp_path), "req-1", "sideways")
        assert res["code"] == "bad_request"
        assert any(r.levelno >= logging.WARNING and "REJECTED" in r.getMessage()
                   for r in caplog.records)

    def test_a_missing_collector_logs_a_warning(self, tmp_path, caplog):
        import logging
        from ghost_agent.core.feedback import apply_human_label
        with caplog.at_level(logging.DEBUG):
            res = apply_human_label(self._agent(tmp_path, False), "r", "positive")
        assert res["code"] == "unavailable"
        assert any(r.levelno >= logging.WARNING and "DROPPED" in r.getMessage()
                   for r in caplog.records)


class TestTheInvariantsRoundOneFoundUnpinned:
    """C7/C8/C11/C15 — guards whose rationale comments cite live incidents
    and whose mutations all survived."""

    def test_a_PASSED_correction_clears_the_reason_at_BOTH_ends(self, tmp_path):
        """collector.py claims "Enforced at BOTH ends"; both mutations
        survived. The overlay half is the one that fixed live trajectory
        f78c8b33 (passed WITH failure_reason still stamped)."""
        from ghost_agent.distill.collector import TrajectoryCollector
        from ghost_agent.distill.schema import Trajectory
        c = TrajectoryCollector(root=tmp_path / "t", session_id="s")
        t = Trajectory(user_request="q", final_response="a",
                       task_kind="user_request")
        t.failure_reason = "structural failure"
        t.outcome = "failed"
        c.append(t)
        # Writer end: the record must not carry a reason with PASSED.
        c.update_outcome(t.id, "passed", reason="ignore me",
                         source="human_feedback:web")
        rec = c.latest_correction(t.id)
        assert rec["outcome"] == "passed"
        assert not rec.get("reason"), "writer kept a reason on PASSED"
        # Overlay end: legacy rows can still carry (passed, reason).
        out = next(iter(c.iter_trajectories()))
        assert out.outcome == "passed"
        assert out.failure_reason == "", "overlay kept a reason on PASSED"

    def test_the_sidecar_reason_is_REDACTED(self, tmp_path):
        """/api/feedback accepts a 500-char note from any authenticated
        caller, and the overlay copies it into `traj.failure_reason` — i.e.
        into the training corpus."""
        from ghost_agent.distill.collector import TrajectoryCollector
        from ghost_agent.distill.schema import Trajectory
        c = TrajectoryCollector(root=tmp_path / "t", session_id="s")
        t = Trajectory(user_request="q", final_response="a",
                       task_kind="user_request")
        c.append(t)
        c.update_outcome(t.id, "failed",
                         reason="my key is sk-ant-api03-SECRETSECRETSECRET",
                         source="human_feedback:web")
        rec = c.latest_correction(t.id)
        assert "SECRETSECRETSECRET" not in rec["reason"], rec["reason"]

    def test_latest_correction_hands_out_a_COPY(self, tmp_path):
        """Its docstring: handing out the cached dict "would let a caller
        poison the cache"."""
        from ghost_agent.distill.collector import TrajectoryCollector
        from ghost_agent.distill.schema import Trajectory
        c = TrajectoryCollector(root=tmp_path / "t", session_id="s")
        t = Trajectory(user_request="q", final_response="a",
                       task_kind="user_request")
        c.append(t)
        c.update_outcome(t.id, "failed", reason="r", source="human_feedback:web")
        got = c.latest_correction(t.id)
        got["outcome"] = "POISONED"
        assert c.latest_correction(t.id)["outcome"] == "failed"

    def test_a_disabled_collector_does_not_report_success(self, tmp_path):
        from ghost_agent.distill.collector import TrajectoryCollector
        c = TrajectoryCollector(root=tmp_path / "t", session_id="s",
                                enabled=False)
        assert not c.update_outcome("any-id", "passed",
                                    source="human_feedback:web")

    def test_one_unreadable_day_file_does_not_abort_the_scan(self, tmp_path):
        """C15/M69: a single bad day file aborting the walk would hide every
        later trajectory from `find_trajectory_for_request`."""
        from ghost_agent.distill.collector import TrajectoryCollector
        from ghost_agent.distill.schema import Trajectory
        c = TrajectoryCollector(root=tmp_path / "t", session_id="s")
        good = Trajectory(user_request="q", final_response="a",
                          task_kind="user_request")
        c.append(good)
        bad_day = tmp_path / "t" / "2026-01-01"
        bad_day.mkdir(parents=True)
        (bad_day / "session-x.jsonl").write_text("{not json\n\x00\x00\n")
        ids = [x.id for x in c.iter_trajectories()]
        assert good.id in ids


class TestRoundTwoResiduals:
    """R2's minors, each with a surviving mutation or a live-history fact."""

    def _coll(self, tmp_path):
        from ghost_agent.distill.collector import TrajectoryCollector
        return TrajectoryCollector(root=tmp_path / "t", session_id="s")

    def _traj(self, coll):
        from ghost_agent.distill.schema import Trajectory
        t = Trajectory(user_request="q", final_response="a",
                       task_kind="user_request")
        coll.append(t)
        return t

    def test_a_LATER_machine_row_does_not_drop_the_turn_from_the_metric(
            self, tmp_path):
        """MINOR-7: the guard became ever-human and the METRIC stayed
        latest-source, so an `operator_overlay` row after a human label
        silently removed the turn from §4BR's denominator."""
        from ghost_agent.core.experiments import _metric_values
        c = self._coll(tmp_path)
        t = self._traj(c)
        c.update_outcome(t.id, "failed", reason="bad",
                         source="human_feedback:slack:owner")
        c.update_outcome(t.id, "failed", reason="bad",
                         source="operator_overlay")
        out = next(x for x in c.iter_trajectories() if x.id == t.id)
        assert out.extra.get("human_labeled") is True
        m = _metric_values(out)
        assert m["human_label_rate"] == 1.0
        assert m["human_failure_rate"] == 1.0, (
            "a human-graded turn left the §4BR denominator")

    def test_the_provenance_TIMESTAMP_is_not_dead_code(self, tmp_path):
        """MINOR-8: the overlay read `corr["at"]`; the writer emits
        `timestamp`, and 376/376 live rows have no `at` key."""
        c = self._coll(tmp_path)
        t = self._traj(c)
        c.update_outcome(t.id, "passed", source="human_feedback:web")
        out = next(x for x in c.iter_trajectories() if x.id == t.id)
        # IDENTITY, not truthiness (R3 m4): a truthiness assertion passed
        # when the stamp was the TRAJECTORY's timestamp, or a constant
        # placeholder — the same property-vs-identity trap the "dead `at`
        # key" fix was correcting.
        assert out.extra.get("outcome_source_at") == c.latest_correction(
            t.id)["timestamp"], (
            "the stamp is not the CORRECTION's timestamp")

    def test_a_nondict_latest_does_not_swallow_the_write(self, tmp_path,
                                                        monkeypatch):
        """MINOR-9: a cross-process write could leave `latest` a non-dict;
        the unguarded `.get` raised into the broad except and returned
        False — a DROPPED LABEL reported as a disk error. Same shape, same
        line, as the NameError this warning introduced in round 1."""
        c = self._coll(tmp_path)
        t = self._traj(c)
        monkeypatch.setattr(c, "_load_corrections",
                            lambda: {t.id: "not-a-dict"})
        assert c.update_outcome(t.id, "passed",
                                source="human_feedback:web") is True

    def test_an_unrefreshable_cache_falls_back_to_a_SCAN(self, tmp_path,
                                                        monkeypatch):
        """MINOR-9b: an OSError inside the reload left the stale set in
        place, and a stale "no human label" is the direction that loses
        data — a machine verdict would overwrite a human one."""
        c = self._coll(tmp_path)
        t = self._traj(c)
        c.update_outcome(t.id, "passed", source="human_feedback:web")
        # Poison the memo so the signature can never match.
        monkeypatch.setattr(c, "_load_corrections", lambda: {})
        c._ever_human_cache = ((-1, -1), set())
        assert c.has_human_label(t.id) is True, (
            "fell back to a stale 'no' instead of scanning")

    def test_the_C5_warning_is_quiet_when_the_OUTCOME_agrees(self, tmp_path,
                                                            caplog):
        """MINOR-10: dropping the outcome-differs clause survived the suite,
        and on live history it would fire spuriously — two humans agreeing
        on two surfaces is not news."""
        import logging
        c = self._coll(tmp_path)
        t = self._traj(c)
        c.update_outcome(t.id, "passed", source="human_feedback:web")
        with caplog.at_level(logging.DEBUG):
            c.update_outcome(t.id, "passed",
                             source="human_feedback:slack:owner")
        assert not [r for r in caplog.records
                    if "CHANGED by a different" in r.getMessage()], (
            "two humans AGREEING was reported as a conflict")


class TestTheLateVerdictCONSEQUENCESAreGatedToo:
    """R2 MAJOR-1. The writer guard protects the OUTCOME ROW; this guard
    protects the CONSEQUENCES — lesson scrubbing, follow-up task filing, the
    "correction to my previous answer" banner, the stream re-render. Round 1
    moved the writer to ever-human and left this one latest-only, so an
    intermediate `user_correction` row protected the row and not the
    consequences: the lessons of a human-approved turn were scrubbed and
    follow-ups filed, both irrecoverable. The round-1 test docstring claimed
    both sites were covered; only one was."""

    def _agent_with(self, tmp_path):
        from ghost_agent.distill.collector import TrajectoryCollector
        from ghost_agent.distill.schema import Trajectory
        from tests.helpers import make_agent, make_context
        coll = TrajectoryCollector(root=tmp_path / "t", session_id="s")
        ctx = make_context(memory_dir=tmp_path / "memory",
                           trajectory_collector=coll)
        t = Trajectory(user_request="q", final_response="a",
                       task_kind="user_request")
        coll.append(t)
        return make_agent(ctx), coll, t

    def test_a_human_label_locks_the_chain(self, tmp_path):
        agent, coll, t = self._agent_with(tmp_path)
        assert agent._human_label_locked(t.id) is False
        coll.update_outcome(t.id, "passed", source="human_feedback:slack:owner")
        assert agent._human_label_locked(t.id) is True

    def test_an_INTERMEDIATE_correction_does_not_unlock_the_chain(self, tmp_path):
        """THE case: the row stays protected, and before the fix the
        consequences did not."""
        agent, coll, t = self._agent_with(tmp_path)
        coll.update_outcome(t.id, "passed", source="human_feedback:slack:owner")
        coll.update_outcome(t.id, "failed", reason="user said no",
                            source="user_correction")
        assert agent._human_label_locked(t.id) is True, (
            "the late-verdict consequence chain would run against a "
            "human-approved turn: lessons scrubbed, follow-ups filed")

    def test_a_machine_only_history_does_not_lock(self, tmp_path):
        """The mirror — locking on machine rows would disable the late
        verdict entirely."""
        agent, coll, t = self._agent_with(tmp_path)
        coll.update_outcome(t.id, "failed", reason="x", source="verifier_late")
        assert agent._human_label_locked(t.id) is False


def test_the_attrition_confound_is_inherited_BEHAVIOURALLY():
    """R2 MAJOR-4, pinned by behaviour not set-membership: the first version
    asserted `"human_failure_rate" in _ATTRITION_SENSITIVE`, which a
    mutation of the USE SITE (`metric == "failure_rate"`) left green."""
    from ghost_agent.core import experiments as ex
    from ghost_agent.distill.collector import Trajectory

    def _t(arm, outcome):
        tr = Trajectory(user_request="q", final_response="a",
                        task_kind="user_request",
                        extra={ex.EXTRA_KEY: {"exp1": arm},
                               "outcome_source": "human_feedback:web"})
        tr.outcome = outcome
        return tr

    # Differential UNKNOWN rate between arms → attrition confound.
    trajs = ([_t("control", "unknown" if i < 30 else "failed")
              for i in range(60)]
             + [_t("treatment", "unknown" if i < 3 else "failed")
                for i in range(60)])
    cmps = {c.metric: c for c in ex.compare_arms(
        ex.summarize_trajectories(trajs)["exp1"])}
    assert "attrition" in cmps["failure_rate"].confound
    assert "attrition" in cmps["human_failure_rate"].confound, (
        "§4BR's PRIMARY metric conditions on resolved AND labelled, so it "
        "inherits differential attrition a fortiori — and carried no caveat")


def test_the_conftest_live_path_isolation_is_COLLECTION_TIME():
    """R3 m9: the module-level isolation block had no test at all — the
    "guard that stops guarding when someone writes the next test file"
    class it was created to fix.

    Asserted structurally AND behaviourally: the block must run at import
    (a function-scoped fixture is too late for module-scoped fixtures and
    for import-time constant binding, which is how the live Slack index
    was truncated and how webpush binds its live subscription file)."""
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parent / "conftest.py").read_text()
    tree = ast.parse(src)
    # A module-level (not function-nested) assignment into os.environ for
    # each live var.
    module_level = ast.dump(ast.Module(
        body=[n for n in tree.body
              if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                    ast.ClassDef))],
        type_ignores=[]))
    for var in ("GHOST_SLACK_REPLY_INDEX", "GHOST_SLACKBOT_LOG",
                "GHOST_PUSH_SUBS_FILE", "GHOST_VAPID_FILE"):
        assert var in module_level, (
            f"{var} is not isolated at COLLECTION time — a fixture cannot "
            "cover module-scoped loads or import-time binding")
    # Behavioural: webpush must not be bound to the operator's live file.
    import importlib.util
    from pathlib import Path as _P
    spec = importlib.util.spec_from_file_location(
        "_wp_probe", _P(__file__).resolve().parents[1] / "interface"
        / "webpush_notify.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert "Data/AI/.ghost_push_subs.json" not in str(m._SUBS_FILE), (
        "webpush is bound to the LIVE subscription file under pytest")


class TestCircularityIsPerExperimentNotPerMetric:
    """R3 M3 — the sharpest finding of the review. Round 2 suppressed the
    DENOMINATOR (`human_label_rate`) on the argument that "the operator's
    first verdict would come from a quantity the rule says is not a
    hypothesis", and left `failure_rate` — which verify_depth's rule
    explicitly disowns as CIRCULAR, because the treatment changes the very
    verdict the outcome label derives from — announcing freely. It reaches a
    verdict FIRST, since the honest metric's floor may take months. The
    small hole was closed using the big hole's argument."""

    @staticmethod
    def _corpus(exp_name):
        from ghost_agent.core import experiments as ex
        from ghost_agent.distill.collector import Trajectory

        def _t(arm, outcome):
            return Trajectory(
                user_request="q", final_response="a",
                task_kind="user_request",
                extra={ex.EXTRA_KEY: {exp_name: arm},
                       "outcome_source": "human_feedback:web"})

        rows = []
        for i in range(40):
            t = _t("control", None)
            t.outcome = "failed" if i < 30 else "passed"
            rows.append(t)
            t2 = _t("treatment", None)
            t2.outcome = "failed" if i < 4 else "passed"
            rows.append(t2)
        return rows

    def test_failure_rate_is_CIRCULAR_for_a_verdict_changing_arm(self):
        from ghost_agent.core import experiments as ex
        rows = self._corpus("verify_depth")
        cmps = {c.metric: c for c in ex.compare_arms(
            ex.summarize_trajectories(rows)["verify_depth"],
            experiment="verify_depth")}
        assert "CIRCULAR" in cmps["failure_rate"].confound
        assert cmps["failure_rate"].verdict.startswith("NO VERDICT"), (
            "the disowned metric still led with a direction")

    def test_the_HUMAN_metric_still_reaches_a_verdict(self):
        """The point of the exemption is to leave the honest metric usable."""
        from ghost_agent.core import experiments as ex
        rows = self._corpus("verify_depth")
        cmps = {c.metric: c for c in ex.compare_arms(
            ex.summarize_trajectories(rows)["verify_depth"],
            experiment="verify_depth")}
        assert "TREATMENT BETTER" in cmps["human_failure_rate"].verdict

    def test_failure_rate_stays_LEGITIMATE_for_other_arms(self):
        """Circularity is per-(experiment, metric): risk_steer does not
        touch the verdict, so a blanket exclusion would have destroyed a
        valid metric for three live arms."""
        from ghost_agent.core import experiments as ex
        rows = self._corpus("risk_steer")
        cmps = {c.metric: c for c in ex.compare_arms(
            ex.summarize_trajectories(rows)["risk_steer"],
            experiment="risk_steer")}
        assert "CIRCULAR" not in cmps["failure_rate"].confound
        assert "TREATMENT" in cmps["failure_rate"].verdict

    def test_the_circular_metric_is_never_ANNOUNCED(self):
        from ghost_agent.core import experiments as ex
        rows = self._corpus("verify_depth")
        summary = ex.summarize_trajectories(rows)
        pend = ex.pending_announcements(
            summary, ex.summarize_trajectories(rows, triggered_only=True))
        assert not any("failure_rate|" in str(p) and "human" not in str(p)
                       for p in pend), (
            "a metric the decision rule disowns was pushed to the operator")

    def test_the_report_threads_the_experiment_name(self):
        """`compare_arms` cannot know the experiment unless the renderer
        tells it — an untraded name silently disables every per-experiment
        caveat."""
        from ghost_agent.core import experiments as ex
        out = ex.render_report(ex.summarize_trajectories(
            self._corpus("verify_depth")))
        assert "CIRCULAR for verify_depth" in out, out[:600]


def test_the_descriptive_metric_renders_NO_VERDICT():
    """R3 m8: suppression covered the announcement push; the report LINE the
    operator reads still opened with "TREATMENT BETTER" for a denominator
    whose lower_is_better is False."""
    from ghost_agent.core import experiments as ex
    from ghost_agent.distill.collector import Trajectory

    def _t(arm, human):
        e = {ex.EXTRA_KEY: {"exp1": arm}}
        if human:
            e["outcome_source"] = "human_feedback:web"
        tr = Trajectory(user_request="q", final_response="a",
                        task_kind="user_request", extra=e)
        tr.outcome = "passed"
        return tr

    rows = ([_t("control", i < 10) for i in range(60)]
            + [_t("treatment", i < 50) for i in range(60)])
    cmp_ = next(c for c in ex.compare_arms(
        ex.summarize_trajectories(rows)["exp1"])
        if c.metric == "human_label_rate")
    assert cmp_.verdict.startswith("NO VERDICT")
    assert "TREATMENT BETTER" not in cmp_.verdict


def test_the_human_lock_FAILS_CLOSED_on_an_uncertain_answer(tmp_path):
    """The guard's failure direction, which one round got backwards.

    It must FAIL CLOSED: an ambiguous answer withholds the machine verdict,
    because what the guard gates — lesson retraction, follow-up filing, the
    correction banner — is irrecoverable, while withholding merely defers.
    An `is True` check fails OPEN for any truthy non-bool and runs the whole
    chain against a human-approved turn (review R4 M2).

    The real defect that prompted `is True` was a TEST DOUBLE returning a
    truthy Mock; that is fixed in `test_critic_async._wire_corpus`.
    """
    from unittest.mock import MagicMock
    from tests.helpers import make_agent, make_context
    ctx = make_context(memory_dir=tmp_path / "memory")
    agent = make_agent(ctx)

    class _Truthy:
        def __bool__(self):
            return True

    coll = MagicMock()
    ctx.trajectory_collector = coll

    coll.has_human_label.return_value = _Truthy()
    assert agent._human_label_locked("t-1") is True, (
        "an ambiguous truthy answer must LOCK — the consequences this "
        "guard gates are irrecoverable")

    coll.has_human_label.return_value = True
    assert agent._human_label_locked("t-1") is True
    coll.has_human_label.return_value = False
    assert agent._human_label_locked("t-1") is False
    # A collector that does not implement it at all must not lock.
    ctx.trajectory_collector = object()
    assert agent._human_label_locked("t-1") is False


class TestTheCircularTableIsFullyPinned:
    """R4 M3/M4/M5/m6: the headline fix's `tts_bon` half, the TRIGGERED
    block's threading, the CIRCULAR-before-attrition precedence, and the
    table's key validity all survived mutation."""

    def test_the_verify_depth_entry_is_load_bearing(self):
        from ghost_agent.core import experiments as ex
        assert "failure_rate" in ex._CIRCULAR_FOR.get("verify_depth", ())
        assert "CIRCULAR" in ex.circular_note("verify_depth", "failure_rate")

    def test_tts_bon_is_NOT_circular_because_bench_outcomes_are_the_ORACLE(self):
        """A retraction pinned so it cannot be re-made (review R5 F1).

        tts_bon was listed as circular for one round on the reasoning that
        bench outcomes are verifier-derived. They are not: dream's §4BF-1c
        write-back stamps the bench trajectory outcome from the BANK ORACLE
        (`source="bench_validator"` — 79/79 tts_bon-stamped bench rows), and
        `update_outcome`'s oracle-yield leg refuses to let a late verifier
        verdict supersede it. So `failure_rate` there IS the ground truth
        the arm's pre-registered rule names.

        The suppression removed the arm's ONLY decidable metric and made its
        SHIP condition ("no confound annotation on the winning metric")
        structurally unsatisfiable — a suppression can create a
        silent-inoperative arm, so it is not the conservative default.
        """
        from ghost_agent.core import experiments as ex
        assert "tts_bon" not in ex._CIRCULAR_FOR, (
            "tts_bon's bench failure_rate is the bank oracle's grade; "
            "suppressing it leaves the arm with nothing decidable")
        assert ex.circular_note("tts_bon", "failure_rate") == ""
    def test_the_ORACLE_yield_guard_is_what_makes_it_non_circular(
            self, tmp_path):
        """BEHAVIOURAL, not lexical. The first version asserted
        `"bench_validator" in <source>` — which the explanatory COMMENT also
        satisfies, so disabling the guard left it green (review R5, my own
        lexical proxy).

        This is the guard the non-circularity rests on: a late verifier
        verdict must not be allowed to overwrite the bank oracle's label. If
        it could, `failure_rate` on the bench population WOULD become
        verifier-derived and the suppression I just retracted would be
        right after all."""
        from ghost_agent.distill.collector import TrajectoryCollector
        from ghost_agent.distill.schema import Trajectory
        c = TrajectoryCollector(root=tmp_path / "t", session_id="s")
        t = Trajectory(user_request="q", final_response="a",
                       task_kind="bench")
        c.append(t)
        assert c.update_outcome(t.id, "passed", source="bench_validator")
        # The bench-local verifier lands ~45s later and must be REFUSED.
        assert c.update_outcome(t.id, "failed", reason="llm says no",
                                source="verifier_late",
                                yield_to_human=True) == "withheld"
        assert c.latest_correction(t.id)["source"] == "bench_validator"
        assert c.latest_correction(t.id)["outcome"] == "passed"

    def test_every_table_key_is_a_REAL_experiment(self):
        """m6: a typo or rename silently voids the caveat, and tts_bon is
        not in DEFAULT_SPECS at all — so its entry is dead on a box with no
        experiments.json."""
        from ghost_agent.core import experiments as ex
        # ⚠ Hermetic. An earlier version read the operator's live
        # experiments.json by absolute path, which coupled the suite to a
        # MUTABLE file: the arm's own documented rollback ("delete the
        # tts_bon entry") turned it RED, and on any box without that file it
        # skipped and gave no coverage at all (review R5 F6). The drift
        # check against the live registry lives separately in
        # test_verify_depth_routing.py, where a skip is the intended
        # behaviour.
        known = {sp.name for sp in ex.DEFAULT_SPECS}
        unknown = set(ex._CIRCULAR_FOR) - known
        assert not unknown, (
            f"_CIRCULAR_FOR names experiments not in DEFAULT_SPECS: "
            f"{unknown}. A name only present in an on-disk registry makes "
            "the caveat dead on every other box — declare it in the code "
            "defaults or drop the entry.")

    def test_every_table_key_has_a_REACHABLE_remedy(self):
        """A suppression must name a replacement that EXISTS for that
        experiment's population.

        ⚠ The first version asserted `exp in _CIRCULAR_REMEDY` — key
        presence, not the property the name claims — so replacing a remedy
        with the exact unreachable text it was written to eliminate left the
        suite green (review R5 F2). This is the same blind spot as the
        finding it was guarding, one level up.
        """
        from ghost_agent.core import experiments as ex
        live_pop = set(ex.DEFAULT_SPECS and
                       {sp.name for sp in ex.DEFAULT_SPECS
                        if getattr(sp, "scope", "live") == "live"})
        for exp in ex._CIRCULAR_FOR:
            remedy = ex._CIRCULAR_REMEDY.get(exp, "")
            assert remedy, (
                f"{exp} suppresses a metric without naming what replaces it")
            # Human feedback arrives against a user request_id, so a
            # human-graded metric is structurally n=0 on any non-live
            # population — naming it there is an unreachable remedy.
            if exp not in live_pop:
                assert "human_" not in remedy, (
                    f"{exp} is not live-scoped, so its remedy cannot be a "
                    f"human-graded metric: {remedy!r}")
            assert ex.circular_note(exp, "failure_rate").endswith(remedy), (
                "the rendered note must carry the remedy verbatim")

    def test_the_TRIGGERED_block_carries_the_experiment_too(self):
        """M4: dropping `experiment=` from the triggered render survived —
        and that is the block the report labels "READ THIS BLOCK"."""
        from ghost_agent.core import experiments as ex
        from ghost_agent.distill.collector import Trajectory

        def _t(arm, outcome):
            tr = Trajectory(user_request="q", final_response="a",
                            task_kind="user_request",
                            extra={ex.EXTRA_KEY: {"verify_depth": arm},
                                   "verify_depth_fired": True})
            tr.outcome = outcome
            return tr

        rows = []
        for i in range(40):
            rows.append(_t("control", "failed" if i < 30 else "passed"))
            rows.append(_t("treatment", "failed" if i < 4 else "passed"))
        out = ex.render_report(
            ex.summarize_trajectories(rows),
            triggered=ex.summarize_trajectories(rows, triggered_only=True))
        head, _, trig = out.partition("triggered turns only")
        assert "CIRCULAR for verify_depth" in trig, (
            "the triggered block — the one the report says to READ — "
            "rendered failure_rate with no circularity caveat")

    def test_CIRCULARITY_outranks_the_attrition_caveat(self):
        """M5: flipping the branch order survived, and it matters —
        `verdict` gates NO VERDICT on the CIRCULAR prefix, so attrition
        winning restores "TREATMENT BETTER — ⚠ differential attrition" on a
        circular metric. Reachable on verify_depth precisely because a
        changed verdict resolves an otherwise-UNKNOWN turn."""
        from ghost_agent.core import experiments as ex
        from ghost_agent.distill.collector import Trajectory

        def _t(arm, outcome):
            tr = Trajectory(user_request="q", final_response="a",
                            task_kind="user_request",
                            extra={ex.EXTRA_KEY: {"verify_depth": arm}})
            tr.outcome = outcome
            return tr

        # Differential UNKNOWN rate -> attrition confound is also present.
        rows = ([_t("control", "unknown" if i < 30 else "failed")
                 for i in range(60)]
                + [_t("treatment", "unknown" if i < 3 else "failed")
                   for i in range(60)])
        cmp_ = next(c for c in ex.compare_arms(
            ex.summarize_trajectories(rows)["verify_depth"],
            experiment="verify_depth") if c.metric == "failure_rate")
        assert cmp_.confound.startswith("CIRCULAR"), (
            "attrition won the annotation and the metric regained a "
            "direction verdict")
        assert cmp_.verdict.startswith("NO VERDICT")
