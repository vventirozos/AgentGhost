"""The diary follows the corpus — queue #7 (2026-08-21).

Before this, the autobiographical log was verdict-backfilled from exactly
ONE place: finalize's inline ``verifier_backfill`` leg, which only fires
when the bounded in-loop critic await wins its 25s race. Measured on the
live store, 385 of 386 labelled diary records came from that race (the
386th from the user-correction hook), while the two paths that carry
essentially all production verdicts wrote the trajectory corpus and never
the diary:

  * ``_backfill_trajectory_outcome`` — the LATE async verdict (~85% of all
    verdicts on the live box);
  * ``core.feedback.apply_human_label`` — 100% of human 👍/👎 labels.

So the agent's memory of its own past was verdict-blind by architecture,
and §4CC's derived-mood streak read a diary whose newest verdict was five
days old on a box producing verdicts daily.

These tests pin the two new legs as FOLLOWERS of the corpus write — the
authority ladder (human label, bench oracle, the shape rule) stays in the
collector and in ``resolve_turn_outcome``, and the diary only ever records
what the corpus ACCEPTED. Every pin drives the real ``SelfModel`` /
``AutobiographicalMemory`` against a real file and asserts on the bytes on
disk, not on a call count.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import asyncio
import json
from types import SimpleNamespace

import pytest

from ghost_agent.core.agent import GhostAgent
from ghost_agent.core.feedback import apply_human_label
from ghost_agent.distill.collector import TrajectoryCollector
from ghost_agent.distill.schema import Outcome, Trajectory
from ghost_agent.selfhood import SelfModel


# ──────────────────────────────────────────────────────────────────────
# Helpers — a real diary on a real tmp root
# ──────────────────────────────────────────────────────────────────────

def _self_model(tmp_path, trajectory_id, *, outcome="unknown"):
    """A real SelfModel carrying one captured (unknown) turn."""
    sm = SelfModel(tmp_path / "selfhood", enabled=True)
    sm.capture_turn(
        trajectory_id=trajectory_id,
        user_request="check the disk",
        tool_names=["execute_command"],
        outcome=outcome,
        final_response="done",
    )
    return sm


def _diary_outcome(sm, trajectory_id):
    """Read the outcome back OFF DISK — not from an in-memory object."""
    path = sm.autobio.path
    hit = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("trajectory_id") == trajectory_id:
            hit = d
    return None if hit is None else hit.get("outcome")


def _diary_summary(sm, trajectory_id):
    path = sm.autobio.path
    hit = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("trajectory_id") == trajectory_id:
            hit = d
    return "" if hit is None else (hit.get("summary") or "")


def _fake_agent(collector_result, cached, self_model, *, read_only=False,
                origin_label=None, human_label_lands_late=False):
    """A GhostAgent stand-in for `_backfill_trajectory_outcome`.

    ``collector_result`` is what the corpus write returns — True (accepted),
    "withheld" (a human label stands) or False (disk failure).
    ``human_label_lands_late`` simulates a human labelling the turn in the
    gap between the corpus write returning and the diary write running.
    """
    writes = []

    class _Rec:
        enabled = True

        def update_outcome(self, tid, outcome, reason="", source="", **kw):
            writes.append((tid, outcome, reason, source))
            return collector_result

        def has_human_label(self, tid):
            # Only True AFTER the corpus write — i.e. the label landed in
            # the window, which is the only way this can be True here (an
            # earlier label would have returned "withheld" above).
            return bool(human_label_lands_late and writes)

    ctx = SimpleNamespace(
        trajectory_collector=_Rec(),
        _recent_trajectories_for_correction=({"fp": cached} if cached
                                             else {}),
        calibration_tracker=None,
        self_model=self_model,
        skill_memory=SimpleNamespace(is_read_only=read_only),
        turn_origin_label=origin_label,
    )
    fake = SimpleNamespace(
        context=ctx,
        _flush_stashed_lesson_outcome=lambda tid, ok: None,
        _drop_pending_corrections_for=lambda tid: None,
    )
    return fake, writes


def _run_late(fake, traj_id, outcome, reason="", settle=0.35):
    """Drive the SYNC `_backfill_trajectory_outcome` and let its
    spawn_bg(to_thread(...)) write land. Not an await — the method is
    synchronous and the loop exists only for the background task."""
    async def go():
        GhostAgent._backfill_trajectory_outcome(fake, traj_id, outcome,
                                                reason)
        await asyncio.sleep(settle)

    asyncio.run(go())


# ──────────────────────────────────────────────────────────────────────
# The LATE verdict path
# ──────────────────────────────────────────────────────────────────────

class TestLateVerdictReachesTheDiary:
    def test_late_passed_lands_in_the_diary(self, tmp_path):
        tid = "a" * 32
        sm = _self_model(tmp_path, tid)
        cached = Trajectory(id=tid, session_id="r1", outcome="unknown",
                            extra={"req_id": "r1"})
        fake, writes = _fake_agent(True, cached, sm)

        assert _diary_outcome(sm, tid) == "unknown"      # precondition
        _run_late(fake, tid, "passed")

        assert len(writes) == 1                          # corpus accepted
        assert _diary_outcome(sm, tid) == "passed"

    def test_late_failed_lands_with_its_reason(self, tmp_path):
        tid = "b" * 32
        sm = _self_model(tmp_path, tid)
        cached = Trajectory(id=tid, session_id="r1", outcome="unknown",
                            extra={"req_id": "r1"})
        fake, writes = _fake_agent(True, cached, sm)

        _run_late(fake, tid, "failed", "the file was never written")

        assert _diary_outcome(sm, tid) == "failed"
        # The prose verdict clause is patched too, so recall/narrative read
        # a coherent record rather than "without a verdict either way".
        summary = _diary_summary(sm, tid)
        assert "without a verdict either way" not in summary

    def test_diary_records_what_the_corpus_ACCEPTED_not_the_raw_verdict(
            self, tmp_path):
        """Identity pin, not a literal: whatever outcome reached the
        collector must be the outcome on disk in the diary. A mutant that
        writes a hardcoded/raw value diverges from the corpus argument."""
        for verdict in ("passed", "failed"):
            tid = f"{verdict}-traj".ljust(32, "0")
            sm = _self_model(tmp_path / verdict, tid)
            cached = Trajectory(id=tid, session_id="r1", outcome="unknown",
                                extra={"req_id": "r1"})
            fake, writes = _fake_agent(True, cached, sm)

            _run_late(fake, tid, verdict, "because")

            assert len(writes) == 1
            corpus_outcome = writes[0][1]
            assert _diary_outcome(sm, tid) == corpus_outcome


class TestTheOperatorStreamSaysWhichStoresTookIt:
    """§LOG doctrine: report only when it WORKS. The defect this fixes was a
    write path nothing in the live stream ever contradicted, so the success
    line must distinguish "the diary took it" from "there was no diary row"
    — otherwise the next silent regression looks identical again."""

    def test_line_says_corpus_and_diary_when_the_diary_took_it(
            self, tmp_path, capsys):
        tid = "h" * 32
        sm = _self_model(tmp_path, tid)
        cached = Trajectory(id=tid, session_id="r1", outcome="unknown",
                            extra={"req_id": "r1"})
        fake, _ = _fake_agent(True, cached, sm)

        _run_late(fake, tid, "passed")

        out = capsys.readouterr().out
        assert "corpus + diary" in out
        assert _diary_outcome(sm, tid) == "passed"

    def test_line_says_corpus_only_when_there_is_no_diary_row(
            self, tmp_path, capsys):
        """Same verdict, but the diary has no row for this trajectory (a
        sim turn, or one lost to template-rollup compaction)."""
        sm = _self_model(tmp_path, "some-other-turn")
        tid = "i" * 32
        cached = Trajectory(id=tid, session_id="r1", outcome="unknown",
                            extra={"req_id": "r1"})
        fake, _ = _fake_agent(True, cached, sm)

        _run_late(fake, tid, "passed")

        out = capsys.readouterr().out
        assert "backfilled into the corpus" in out
        assert "corpus + diary" not in out


class TestTheDiaryNeverOutrunsTheCorpus:
    def test_withheld_corpus_write_leaves_the_diary_alone(self, tmp_path):
        """A standing human label made the collector refuse. The diary has
        no source-rank of its own, so if this leg wrote anyway the machine
        verdict would silently overwrite the human's in the agent's own
        memory — the exact authority inversion `yield_to_human` exists to
        prevent, one layer down."""
        tid = "c" * 32
        sm = _self_model(tmp_path, tid)
        cached = Trajectory(id=tid, session_id="r1", outcome="unknown",
                            extra={"req_id": "r1"})
        fake, writes = _fake_agent("withheld", cached, sm)

        _run_late(fake, tid, "failed", "verifier says no")

        assert len(writes) == 1                 # the corpus was asked
        assert _diary_outcome(sm, tid) == "unknown"   # and it said no

    def test_failed_corpus_write_leaves_the_diary_alone(self, tmp_path):
        """ENOSPC / permission failure: the corpus does not carry the
        verdict, so neither may the diary."""
        tid = "d" * 32
        sm = _self_model(tmp_path, tid)
        cached = Trajectory(id=tid, session_id="r1", outcome="unknown",
                            extra={"req_id": "r1"})
        fake, writes = _fake_agent(False, cached, sm)

        _run_late(fake, tid, "failed", "verifier says no")

        assert _diary_outcome(sm, tid) == "unknown"

    def test_in_process_human_label_short_circuits_before_any_write(
            self, tmp_path):
        tid = "e" * 32
        sm = _self_model(tmp_path, tid)
        cached = Trajectory(id=tid, session_id="r1", outcome="passed",
                            extra={"req_id": "r1", "human_labeled": True})
        fake, writes = _fake_agent(True, cached, sm)

        _run_late(fake, tid, "failed", "verifier says no")

        assert writes == []
        assert _diary_outcome(sm, tid) == "unknown"

    def test_a_human_label_landing_in_the_write_WINDOW_still_wins(
            self, tmp_path):
        """The corpus write and the diary write are two separate awaits. A
        human label landing in that gap gets a "yes" from the collector
        microseconds before the human speaks — and writing anyway would
        leave the diary carrying a machine verdict the corpus no longer
        does, i.e. the human overruled in the agent's own memory. The leg
        re-asks the collector's own ever-human check immediately before
        writing."""
        tid = "w" * 32
        sm = _self_model(tmp_path, tid)
        cached = Trajectory(id=tid, session_id="r1", outcome="unknown",
                            extra={"req_id": "r1"})
        fake, writes = _fake_agent(True, cached, sm,
                                   human_label_lands_late=True)

        _run_late(fake, tid, "failed", "verifier says no")

        assert len(writes) == 1                       # corpus took it
        assert _diary_outcome(sm, tid) == "unknown"   # diary did not

    def test_a_collector_without_the_check_is_not_treated_as_a_veto(
            self, tmp_path):
        """Absence of the guard must fail OPEN, not closed — a stub or an
        older collector without `has_human_label` must not silently stop
        every diary write (the failure mode would be this whole fix going
        inert with nothing in the stream to say so)."""
        tid = "v" * 32
        sm = _self_model(tmp_path, tid)
        cached = Trajectory(id=tid, session_id="r1", outcome="unknown",
                            extra={"req_id": "r1"})
        fake, _ = _fake_agent(True, cached, sm)
        del type(fake.context.trajectory_collector).has_human_label

        _run_late(fake, tid, "passed")

        assert _diary_outcome(sm, tid) == "passed"

    def test_shape_rule_withheld_pass_never_reaches_the_diary(self, tmp_path):
        """`resolve_turn_outcome` rule 2: a SHAPE-heuristic FAILED is never
        upgraded by a late verifier PASS (only a structural-only one is,
        per the 07-31 exception), so the method returns before the corpus
        write. The diary must not be the one surface that books a success
        the ladder withheld — it is read back into recall, the competence
        prior and the mood streak."""
        tid = "f" * 32
        sm = _self_model(tmp_path, tid, outcome="failed")
        cached = Trajectory(id=tid, session_id="r1", outcome="failed",
                            failure_reason="selector thrashed 4x",
                            extra={"req_id": "r1"})
        fake, writes = _fake_agent(True, cached, sm)

        _run_late(fake, tid, "passed")

        assert writes == []
        assert _diary_outcome(sm, tid) == "failed"

    def test_structural_only_failed_IS_upgraded_on_both_surfaces(
            self, tmp_path):
        """The other side of the same ladder (07-31 async half): a
        structural-only FAILED IS liftable by a late PASS. Pinning both
        directions is what makes the test above a gate rather than a
        tautology — and pins that the diary tracks the corpus either way."""
        from ghost_agent.distill.outcome_heuristics import (
            STRUCTURAL_FAILURE_REASON)
        tid = "g" * 32
        sm = _self_model(tmp_path, tid, outcome="failed")
        cached = Trajectory(id=tid, session_id="r1", outcome="failed",
                            failure_reason=STRUCTURAL_FAILURE_REASON,
                            extra={"req_id": "r1"})
        fake, writes = _fake_agent(True, cached, sm)

        _run_late(fake, tid, "passed")

        assert len(writes) == 1
        assert writes[0][1] == "passed"
        assert _diary_outcome(sm, tid) == "passed"


class TestTheReasonIsRedactedIntoTheDiary:
    """The diary is the surface that gets read back INTO prompts, and the
    reasons written here are not the agent's own prose: they are verifier
    `issues`/`reasoning` (which quote tool output) and, on the
    user-correction path, text derived from the user's own message. The
    corpus's sibling writer has always redacted; the diary had the hole."""

    def test_a_secret_in_a_late_verdict_reason_is_scrubbed_on_disk(
            self, tmp_path):
        tid = "r" * 32
        sm = _self_model(tmp_path, tid)
        cached = Trajectory(id=tid, session_id="r1", outcome="unknown",
                            extra={"req_id": "r1"})
        fake, _ = _fake_agent(True, cached, sm)

        _run_late(fake, tid, "failed",
                  "the reply leaked ops@example.com and sk-ABCDEF0123456789abcdef")

        summary = _diary_summary(sm, tid)
        assert "ops@example.com" not in summary
        assert "sk-ABCDEF0123456789abcdef" not in summary
        assert "[REDACTED_EMAIL]" in summary

    def test_redaction_runs_before_the_120_char_clip(self):
        """Clipping first cuts the secret in half and leaves a fragment the
        pattern can no longer match — the classic ordering bug in this
        shape. The padding is sized so the address STRADDLES the 120-char
        boundary: with a shorter reason both orderings produce identical
        output and the test proves nothing (it was written that way first,
        and the ordering mutant walked straight through it)."""
        from ghost_agent.selfhood.autobiographical import _outcome_phrase
        pad = "x" * 110          # address spans chars 111..126
        phrase = _outcome_phrase("failed", f"{pad} ops@example.com")

        assert "ops@example.com" not in phrase
        # The discriminating half: clip-then-redact leaves "ops@examp",
        # which has no TLD left for the email pattern to catch.
        assert "ops@" not in phrase

    def test_capture_and_backfill_still_phrase_verdicts_identically(
            self, tmp_path):
        """The helper's whole reason for existing. Redaction must not have
        made the two paths diverge — pinned as an IDENTITY between the two
        rendered summaries, not as two literals."""
        from ghost_agent.selfhood.autobiographical import _outcome_phrase
        reason = "tool said no: admin@example.com"
        tid = "p" * 32
        sm = _self_model(tmp_path, tid)
        cached = Trajectory(id=tid, session_id="r1", outcome="unknown",
                            extra={"req_id": "r1"})
        fake, _ = _fake_agent(True, cached, sm)

        _run_late(fake, tid, "failed", reason)

        assert _outcome_phrase("failed", reason) in _diary_summary(sm, tid)


class TestRealOnlyOriginGate:
    """§4BF 1c: selfhood is a real_only admissibility row. The CAPTURE site
    writes a diary entry only for origin=user turns, so the backfill gate
    must match it exactly — otherwise a sim/bench verdict either mislabels
    a real record or (worse) teaches the gate to disagree with the capture
    site it mirrors."""

    def test_sim_origin_writes_the_corpus_but_not_the_diary(self, tmp_path):
        tid = "0" * 32
        sm = _self_model(tmp_path, tid)
        cached = Trajectory(id=tid, session_id="r1", outcome="unknown",
                            extra={"req_id": "r1"})
        fake, writes = _fake_agent(True, cached, sm, read_only=True)

        _run_late(fake, tid, "passed")

        assert len(writes) == 1                        # corpus still gets it
        assert _diary_outcome(sm, tid) == "unknown"

    def test_bench_origin_writes_the_corpus_but_not_the_diary(self, tmp_path):
        tid = "1" * 32
        sm = _self_model(tmp_path, tid)
        cached = Trajectory(id=tid, session_id="r1", outcome="unknown",
                            extra={"req_id": "r1"})
        fake, writes = _fake_agent(True, cached, sm, read_only=True,
                                   origin_label="bench")

        _run_late(fake, tid, "passed")

        assert len(writes) == 1
        assert _diary_outcome(sm, tid) == "unknown"

    def test_origin_is_snapshot_at_call_time_not_read_by_the_writer(
            self, tmp_path):
        """The corpus write is DEFERRED through a background thread and
        lands 20-60s after the turn. Reading the per-turn origin flag from
        inside that write is the §4CC R1 shape — by then the context can
        belong to the next request. Flip the flag after the call returns:
        the decision must already be made.

        Both directions are pinned (a snapshot that reads the flag once but
        at the WRONG time passes only one of them)."""
        # user at call time → sim afterwards: the write must still land.
        tid = "2" * 32
        sm = _self_model(tmp_path / "a", tid)
        cached = Trajectory(id=tid, session_id="r1", outcome="unknown",
                            extra={"req_id": "r1"})
        fake, _ = _fake_agent(True, cached, sm, read_only=False)

        async def go_user():
            GhostAgent._backfill_trajectory_outcome(fake, tid, "passed")
            fake.context.skill_memory.is_read_only = True   # next request
            await asyncio.sleep(0.35)

        asyncio.run(go_user())
        assert _diary_outcome(sm, tid) == "passed"

        # sim at call time → user afterwards: the write must still be gated.
        tid2 = "3" * 32
        sm2 = _self_model(tmp_path / "b", tid2)
        cached2 = Trajectory(id=tid2, session_id="r1", outcome="unknown",
                             extra={"req_id": "r1"})
        fake2, _ = _fake_agent(True, cached2, sm2, read_only=True)

        async def go_sim():
            GhostAgent._backfill_trajectory_outcome(fake2, tid2, "passed")
            fake2.context.skill_memory.is_read_only = False
            await asyncio.sleep(0.35)

        asyncio.run(go_sim())
        assert _diary_outcome(sm2, tid2) == "unknown"


class TestLateBackfillIsNeverFatal:
    def test_a_raising_diary_does_not_break_the_corpus_write(self, tmp_path):
        """Backfill is secondary to the turn: a broken diary must not cost
        the corpus its verdict or raise out of the background task."""
        tid = "4" * 32

        class _Exploding:
            enabled = True

            def record_outcome(self, *a, **kw):
                raise RuntimeError("diary on fire")

        cached = Trajectory(id=tid, session_id="r1", outcome="unknown",
                            extra={"req_id": "r1"})
        fake, writes = _fake_agent(True, cached, _Exploding())

        _run_late(fake, tid, "passed")

        assert len(writes) == 1
        assert cached.outcome == "passed"   # the corpus consequences stand

    def test_a_cache_MISS_still_gets_its_FAILED_into_the_diary(self, tmp_path):
        """A cache-evicted trajectory can't be inspected, so the PASSED leg
        returns early — but FAILED always lands in the corpus, and the
        diary must follow it there too. This is the path a long-running or
        heavily-multiplexed session actually takes."""
        tid = "m" * 32
        sm = _self_model(tmp_path, tid)
        fake, writes = _fake_agent(True, None, sm)     # nothing cached

        _run_late(fake, tid, "failed", "verifier says no")

        assert len(writes) == 1
        assert _diary_outcome(sm, tid) == "failed"

    def test_absent_self_model_is_a_noop(self, tmp_path):
        tid = "5" * 32
        cached = Trajectory(id=tid, session_id="r1", outcome="unknown",
                            extra={"req_id": "r1"})
        fake, writes = _fake_agent(True, cached, None)

        _run_late(fake, tid, "passed")

        assert len(writes) == 1
        assert cached.outcome == "passed"


# ──────────────────────────────────────────────────────────────────────
# The HUMAN LABEL path (core/feedback.apply_human_label)
# ──────────────────────────────────────────────────────────────────────

def _write_traj(collector, req_id, traj_id):
    t = Trajectory(
        session_id=req_id,
        user_request="do the thing",
        final_response="done",
        outcome=Outcome.UNKNOWN.value,
        extra={"req_id": req_id},
    )
    t.id = traj_id
    assert collector.append(t) is not None
    return t


def _feedback_agent(collector, self_model, *, read_only=False):
    return SimpleNamespace(
        context=SimpleNamespace(
            trajectory_collector=collector,
            _recent_trajectories_for_correction={},
            self_model=self_model,
            skill_memory=SimpleNamespace(is_read_only=read_only),
            args=None,
        ),
        _flush_stashed_lesson_outcome=lambda tid, ok: None,
    )


class TestHumanLabelReachesTheDiary:
    def test_thumbs_up_labels_the_diary(self, tmp_path):
        tid = "6" * 32
        c = TrajectoryCollector(root=tmp_path / "traj", session_id="s")
        _write_traj(c, "req-1", tid)
        sm = _self_model(tmp_path, tid)
        agent = _feedback_agent(c, sm)

        res = apply_human_label(agent, "req-1", "positive", source="web")

        assert res["ok"] is True
        assert _diary_outcome(sm, tid) == "passed"

    def test_thumbs_down_labels_the_diary_with_the_note(self, tmp_path):
        tid = "7" * 32
        c = TrajectoryCollector(root=tmp_path / "traj", session_id="s")
        _write_traj(c, "req-1", tid)
        sm = _self_model(tmp_path, tid)
        agent = _feedback_agent(c, sm)

        res = apply_human_label(agent, "req-1", "negative",
                                note="you never ran it", source="slack:owner")

        assert res["ok"] is True
        assert _diary_outcome(sm, tid) == "failed"
        assert "without a verdict either way" not in _diary_summary(sm, tid)

    def test_diary_matches_the_outcome_the_route_reports(self, tmp_path):
        """Identity pin: the diary must equal what apply_human_label says it
        wrote, for both signals — not a literal this test chose."""
        for signal in ("positive", "negative"):
            tid = f"{signal}-x".ljust(32, "0")
            c = TrajectoryCollector(root=tmp_path / signal, session_id="s")
            _write_traj(c, "req-1", tid)
            sm = _self_model(tmp_path / f"{signal}-sh", tid)
            agent = _feedback_agent(c, sm)

            res = apply_human_label(agent, "req-1", signal, source="web")

            assert _diary_outcome(sm, tid) == res["outcome"]

    def test_a_repeat_click_HEALS_a_diary_row_the_corpus_already_carries(
            self, tmp_path):
        """The corpus dedupes an identical repeat inside its lock and
        returns "unchanged" — but a diary row written before this leg
        existed (or by a write that failed) is still stale. The re-click
        doctrine (`_stamp_cache` repairs a failed stamp) applies here too,
        so this leg must run on the unchanged branch as well."""
        tid = "8" * 32
        c = TrajectoryCollector(root=tmp_path / "traj", session_id="s")
        _write_traj(c, "req-1", tid)
        sm = _self_model(tmp_path, tid)

        # First label WITHOUT a diary — the pre-fix world.
        first = apply_human_label(_feedback_agent(c, None), "req-1",
                                  "positive", source="web")
        assert first["ok"] is True
        assert _diary_outcome(sm, tid) == "unknown"

        # The re-click dedupes in the corpus...
        again = apply_human_label(_feedback_agent(c, sm), "req-1",
                                  "positive", source="web")
        assert again.get("unchanged") is True
        # ...and still heals the diary.
        assert _diary_outcome(sm, tid) == "passed"

    def test_sim_origin_label_does_not_touch_the_diary(self, tmp_path):
        tid = "9" * 32
        c = TrajectoryCollector(root=tmp_path / "traj", session_id="s")
        _write_traj(c, "req-1", tid)
        sm = _self_model(tmp_path, tid)
        agent = _feedback_agent(c, sm, read_only=True)

        res = apply_human_label(agent, "req-1", "positive", source="web")

        assert res["ok"] is True              # the corpus label still lands
        assert _diary_outcome(sm, tid) == "unknown"

    def test_a_raising_diary_does_not_fail_the_label(self, tmp_path):
        """The label is COMMITTED before this leg runs — a diary failure
        must not convert a recorded label into a 503 the client retries."""
        tid = "A" * 32
        c = TrajectoryCollector(root=tmp_path / "traj", session_id="s")
        _write_traj(c, "req-1", tid)

        class _Exploding:
            enabled = True

            def record_outcome(self, *a, **kw):
                raise RuntimeError("diary on fire")

        agent = _feedback_agent(c, _Exploding())
        res = apply_human_label(agent, "req-1", "positive", source="web")

        assert res["ok"] is True


# ──────────────────────────────────────────────────────────────────────
# END-TO-END: the REAL collector, both arrival orders
# ──────────────────────────────────────────────────────────────────────

class TestConvergenceAgainstTheRealCollector:
    """Everything above drives a STUB collector, which cannot fail the way
    the real one can: a stub ignores `yield_to_human`, so the authority
    DELEGATION that is the entire design claim was never actually
    exercised. These pins run the real `TrajectoryCollector`, the real
    `apply_human_label` and the real `_backfill_trajectory_outcome`
    against a real diary, and assert the property the design promises —
    **the diary ends up carrying exactly what the corpus resolved** — in
    both arrival orders."""

    def _world(self, tmp_path, tid):
        col = TrajectoryCollector(root=tmp_path / "traj", session_id="s",
                                  enabled=True)
        _write_traj(col, "req-1", tid)
        sm = _self_model(tmp_path / "sh", tid)
        cached = Trajectory(id=tid, session_id="req-1", outcome="unknown",
                            extra={"req_id": "req-1"})
        agent = SimpleNamespace(
            context=SimpleNamespace(
                trajectory_collector=col,
                _recent_trajectories_for_correction={"fp": cached},
                calibration_tracker=None,
                self_model=sm,
                skill_memory=SimpleNamespace(is_read_only=False),
                turn_origin_label=None,
                args=None,
            ),
            _flush_stashed_lesson_outcome=lambda t, ok: None,
            _drop_pending_corrections_for=lambda t: None,
        )
        return col, sm, agent

    @staticmethod
    def _corpus_outcome(col, tid):
        for t in col.iter_trajectories():
            if t.id == tid:
                return t.outcome
        return None

    def test_human_first_then_late_verdict(self, tmp_path):
        """The real collector must WITHHOLD the late machine verdict, and
        the diary must therefore keep the human's. A `yield_to_human` that
        stopped being passed would flip both stores here — the stub tests
        above cannot see that at all."""
        tid = "e2e-human-first".ljust(32, "0")
        col, sm, agent = self._world(tmp_path, tid)

        res = apply_human_label(agent, "req-1", "positive", source="web")
        assert res["ok"] is True
        _run_late(agent, tid, "failed", "judge disagrees")

        assert self._corpus_outcome(col, tid) == "passed"
        assert _diary_outcome(sm, tid) == "passed"
        assert _diary_outcome(sm, tid) == self._corpus_outcome(col, tid)

    def test_late_verdict_first_then_human(self, tmp_path):
        """The human supersedes in the corpus; the diary must follow it
        there rather than keeping the machine's earlier verdict."""
        tid = "e2e-late-first".ljust(32, "0")
        col, sm, agent = self._world(tmp_path, tid)

        _run_late(agent, tid, "failed", "judge disagrees")
        assert _diary_outcome(sm, tid) == "failed"      # interim state

        res = apply_human_label(agent, "req-1", "positive", source="web")
        assert res["ok"] is True

        assert self._corpus_outcome(col, tid) == "passed"
        assert _diary_outcome(sm, tid) == "passed"
        assert _diary_outcome(sm, tid) == self._corpus_outcome(col, tid)

    def test_late_verdict_alone_agrees_with_the_corpus(self, tmp_path):
        tid = "e2e-late-only".ljust(32, "0")
        col, sm, agent = self._world(tmp_path, tid)

        _run_late(agent, tid, "failed", "judge disagrees")

        assert _diary_outcome(sm, tid) == self._corpus_outcome(col, tid)
        assert _diary_outcome(sm, tid) == "failed"

    def test_human_label_wins_through_the_WRITER_when_the_cache_is_evicted(
            self, tmp_path):
        """The authority model has TWO layers and they are not
        interchangeable. The in-process `human_labeled` stamp short-circuits
        `_backfill_trajectory_outcome` before it ever reaches the collector
        — so a test whose cache carries the stamp never exercises the
        writer-side guard at all (the `yield_to_human` mutant walked
        straight through the two order tests above for exactly that
        reason). The writer-side guard exists for the case this leg
        actually rides: the late write is deferred through a background
        thread, so it can be checked before the label exists and land
        after it, and the cached trajectory can be evicted entirely.

        Here the human labels the turn with NOTHING in the cache. The
        machine verdict therefore reaches the collector, which must refuse
        it against the file — and the diary, following that refusal, must
        keep the human's verdict."""
        tid = "e2e-evicted".ljust(32, "0")
        col, sm, agent = self._world(tmp_path, tid)
        agent.context._recent_trajectories_for_correction = {}   # evicted

        res = apply_human_label(agent, "req-1", "positive", source="web")
        assert res["ok"] is True
        assert _diary_outcome(sm, tid) == "passed"

        _run_late(agent, tid, "failed", "judge disagrees")

        assert self._corpus_outcome(col, tid) == "passed"
        assert _diary_outcome(sm, tid) == "passed"
        assert _diary_outcome(sm, tid) == self._corpus_outcome(col, tid)


# ──────────────────────────────────────────────────────────────────────
# The prose must not contradict the field it sits next to
# ──────────────────────────────────────────────────────────────────────

class TestUpgradeOffACapturedFailureRewritesTheProse:
    """`update_outcome` used to patch only the "unknown" clause, so a row
    CAPTURED as failed and later upgraded read "…and it didn't land:
    <reason>." under `outcome="passed"` — the prose contradicting the field,
    on the surface the narrative layer and recall actually read. Three such
    rows exist on the live box from July. This leg makes upgrades common
    (a structural-only FAILED is liftable by a late PASS), so it also makes
    the incoherence common."""

    def test_captured_failed_then_late_pass_rewrites_the_clause(
            self, tmp_path):
        from ghost_agent.distill.outcome_heuristics import (
            STRUCTURAL_FAILURE_REASON)
        tid = "up" + "0" * 30
        sm = SelfModel(tmp_path / "sh", enabled=True)
        sm.capture_turn(trajectory_id=tid, user_request="fix the build",
                        tool_names=["execute_command"], outcome="failed",
                        final_response="done",
                        failure_reason="cannot open file.txt")
        assert "didn't land" in _diary_summary(sm, tid)      # precondition

        cached = Trajectory(id=tid, session_id="r1", outcome="failed",
                            failure_reason=STRUCTURAL_FAILURE_REASON,
                            extra={"req_id": "r1"})
        fake, _ = _fake_agent(True, cached, sm)
        _run_late(fake, tid, "passed")

        assert _diary_outcome(sm, tid) == "passed"
        summary = _diary_summary(sm, tid)
        assert "didn't land" not in summary
        assert "the answer landed" in summary

    def test_a_period_INSIDE_the_reason_does_not_splice_the_record(
            self, tmp_path):
        """The clause is last in the template, so it runs to the end of the
        string — which is what makes the swap safe. Cutting at the first
        period after the clause start would leave ".txt." dangling."""
        tid = "dot" + "0" * 29
        sm = SelfModel(tmp_path / "sh", enabled=True)
        sm.capture_turn(trajectory_id=tid, user_request="fix the build",
                        tool_names=["execute_command"], outcome="failed",
                        final_response="done",
                        failure_reason="cannot open file.txt")

        cached = Trajectory(id=tid, session_id="r1", outcome="unknown",
                            extra={"req_id": "r1"})
        fake, _ = _fake_agent(True, cached, sm)
        _run_late(fake, tid, "passed")

        summary = _diary_summary(sm, tid)
        assert ".txt" not in summary
        assert summary.endswith("landed.")

    def test_a_rollup_record_with_no_clause_is_left_alone(self, tmp_path):
        """Rollup summaries ("I worked on N of my recurring …") carry no
        verdict clause. Inventing one there would attribute a single turn's
        verdict to a merged record covering many."""
        from ghost_agent.selfhood.autobiographical import (
            _swap_verdict_clause)
        rollup = "I worked on 4 of my recurring synthetic training exercises in a row."

        assert _swap_verdict_clause(rollup, "and the answer landed") == rollup

    def test_downgrade_rewrites_the_clause_too(self, tmp_path):
        """Both directions — a captured PASSED later refuted must not keep
        reading "the answer landed"."""
        tid = "dn" + "0" * 30
        sm = SelfModel(tmp_path / "sh", enabled=True)
        sm.capture_turn(trajectory_id=tid, user_request="ship it",
                        tool_names=["file_system"], outcome="passed",
                        final_response="done")
        assert "the answer landed" in _diary_summary(sm, tid)

        cached = Trajectory(id=tid, session_id="r1", outcome="unknown",
                            extra={"req_id": "r1"})
        fake, _ = _fake_agent(True, cached, sm)
        _run_late(fake, tid, "failed", "never actually ran")

        summary = _diary_summary(sm, tid)
        assert "the answer landed" not in summary
        assert "never actually ran" in summary

    def test_a_decoy_phrase_in_the_quoted_request_cannot_truncate_the_row(
            self, tmp_path):
        """The summary QUOTES the user's request, so a request containing a
        verdict phrase plants a decoy EARLIER in the string than the real
        clause. Because the swap cuts to the end, a leftmost match does not
        merely mis-phrase — it truncates the record mid-quote and drops the
        tool phrase. Found by attacking the swap after it shipped: the
        hazard arrived WITH the fix (the code it replaced substituted in
        place and could never truncate)."""
        from ghost_agent.selfhood.autobiographical import (
            _swap_verdict_clause, summarise_turn_first_person)
        summary = summarise_turn_first_person(
            user_request="you claimed X and the answer landed but it did not",
            tool_names=["file_system"], outcome="failed",
            final_response="x", failure_reason="it crashed")

        swapped = _swap_verdict_clause(summary, "and the answer landed")

        # the quote survives intact, closing quote and all
        assert '"you claimed X and the answer landed but it did not"' in swapped
        assert "I reached for file_system" in swapped
        assert swapped.endswith("and the answer landed.")
        assert "it crashed" not in swapped

    def test_the_decoy_survives_a_real_backfill_end_to_end(self, tmp_path):
        tid = "decoy" + "0" * 27
        sm = SelfModel(tmp_path / "sh", enabled=True)
        sm.capture_turn(
            trajectory_id=tid,
            user_request="you claimed X and the answer landed but it did not",
            tool_names=["file_system"], outcome="failed",
            final_response="x", failure_reason="it crashed")

        cached = Trajectory(id=tid, session_id="r1", outcome="unknown",
                            extra={"req_id": "r1"})
        fake, _ = _fake_agent(True, cached, sm)
        _run_late(fake, tid, "passed")

        summary = _diary_summary(sm, tid)
        assert '"you claimed X and the answer landed but it did not"' in summary
        assert "I reached for file_system" in summary
        assert _diary_outcome(sm, tid) == "passed"
