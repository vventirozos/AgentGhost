"""The unacknowledged-total-failure shape rule (operator decision 2026-08-04).

Narrows the 2026-07-31 honest-failure rule, which had made a verifier PASS
outrank a structural execution failure unconditionally.

The live counter-example — req 03b96c28, trajectory f78c8b33: the user asked
for a line count of a file OUTSIDE the sandbox; `file_system`, `execute` and
`file_system` ALL failed; the reply was the single character ``0``; the cheap
judge REFUTED it correctly at conf 1.0; escalation overturned to CONFIRMED;
and the corpus label was rewritten ``failed -> passed``. A fabrication was
laundered into a positive training example by the rule built to protect
honest ones.

The rule is SHAPE-only — no model is asked anything:

    every tool call failed  AND  the reply never said so
        =>  a "passed" verdict may not lift the structural FAILED.

Both halves are load-bearing and are tested as a mutually discriminating
pair: "ALL, not ANY" keeps a recovered turn passing, and the acknowledgment
check keeps "that file does not exist" passing — without it the exact
incentive gradient that produces fabrication is recreated.
"""

import types

import pytest

from ghost_agent.distill.outcome_heuristics import (
    STRUCTURAL_FAILURE_REASON,
    is_structural_reason,
    resolve_turn_outcome,
    response_acknowledges_failure,
    tool_failure_flags,
    unacked_failure_gate_enabled,
    unacknowledged_total_failure,
    unacknowledged_total_failure_for_trajectory,
)
from ghost_agent.distill.schema import Outcome, ToolCall, Trajectory

P, F, U = Outcome.PASSED.value, Outcome.FAILED.value, Outcome.UNKNOWN.value

# The three results the live turn actually recorded, verbatim in shape.
_FS_ERR = ("Error: '/Users/x/Agent/PROJECT_JOURNAL.md' does not exist in the "
           "current project's sandbox. Files that DO exist here: a.png, b.txt")
_EXEC_ERR = ("--- EXECUTION RESULT ---\nEXIT CODE: 1\nSTDOUT/STDERR:\n"
             "wc: /Users/x/Agent/PROJECT_JOURNAL.md: No such file or directory\n")
_LIVE_REQUEST = ("Using the file system tool, count the lines in "
                 "/Users/x/Agent/PROJECT_JOURNAL.md and reply with just "
                 "the number.")


def _msgs(results, *, request=_LIVE_REQUEST, reply="0", names=None):
    """A message list in the shape `_extract_tool_calls` parses."""
    names = names or ["file_system"] * len(results)
    out = [{"role": "user", "content": request}]
    for i, (nm, res) in enumerate(zip(names, results)):
        out.append({"role": "assistant", "tool_calls": [
            {"id": f"c{i}", "function": {"name": nm, "arguments": "{}"}}]})
        out.append({"role": "tool", "tool_call_id": f"c{i}", "content": res})
    out.append({"role": "assistant", "content": reply})
    return out


# ══════════════════════════════════════════════════════════════════════
# The detector — measured against the real corpus shapes
# ══════════════════════════════════════════════════════════════════════

class TestAcknowledgmentDetector:
    """Judged on CONTENT, never on length. Every string below is taken
    from (or modelled directly on) a real trajectory in
    $GHOST_HOME/system/trajectories; the detector agrees with a hand
    labelling of all 23 all-tools-failed turns in the live corpus."""

    @pytest.mark.parametrize("reply", [
        "The `manage_projects` tool does not support `action=verify_release`.",
        "Both attempts failed with the same error — the file does not exist.",
        "Dark web search for \"gun\" returned zero results — engines are flaky.",
        "There you go — the file read error:\n```\nERROR: '/tmp/x' does not "
        "exist\n```",
        "I hit a hard limit after repeated failures and could not complete "
        "this task.",
        "Self-play complete.\n\nSynthetic challenge generation failed: ...",
        "The skill doesn't exist on file — it was likely already removed.",
        "In Python, `0/0` raises a `ZeroDivisionError`:\n```\nTraceback...\n```",
        # SHORT and honest — length must not be the discriminator.
        "not found",
        "No such file.",
    ])
    def test_honest_reports_are_acknowledgments(self, reply):
        assert response_acknowledges_failure(reply) is True

    @pytest.mark.parametrize("reply", [
        "0",
        '{"move": "g8f6", "comment": "Developing the knight to its natural '
        'square, which controls e4 and prepares castling."}',
        "Let me check the frontend code to find out why pieces aren't "
        "showing.\n\nThe file is at `static/index.html`.",
        "The answer is 9536.",
        "",
    ])
    def test_silent_or_fabricated_replies_are_not(self, reply):
        assert response_acknowledges_failure(reply) is False

    def test_instructed_literal_counts_as_acknowledgment(self):
        """The 2026-07-31 rule's own live-validation probe: the user pins
        the reply text, so the acknowledgment lives in the REQUEST. Without
        this escape that probe would regress to FAILED."""
        req = ("Try to read the file 'missing.txt' with the file_system "
               "tool. It does not exist — when the read fails, do NOT retry "
               "or search anywhere; just reply with exactly the word: NOPE")
        assert response_acknowledges_failure("NOPE", req) is True

    def test_instructed_format_does_not_license_a_fabricated_value(self):
        """"reply with just the number" names a FORMAT, not a literal — the
        live `0` must not slip through the same escape."""
        assert response_acknowledges_failure("0", _LIVE_REQUEST) is False

    def test_literal_escape_needs_the_instruction_span(self):
        """The token must sit inside an explicit exact-reply instruction,
        not merely somewhere in the request."""
        assert response_acknowledges_failure(
            "NOPE", "Is the answer NOPE or something else?") is False


class TestFailureFlagsBothShapes:
    """One sniffer for the corpus `ToolCall` shape AND the turn loop's
    plain-dict shape — a second definition is how the corpus and the
    operator's Turn Outcome line came to disagree before."""

    def test_toolcall_objects(self):
        flags = tool_failure_flags([
            ToolCall(name="file_system", result=_FS_ERR, error="e"),
            ToolCall(name="file_system", result="contents ok"),
        ])
        assert flags == [True, False]

    def test_turn_loop_dicts(self):
        flags = tool_failure_flags([
            {"name": "file_system", "content": _FS_ERR},
            {"name": "execute", "content": _EXEC_ERR},
            {"name": "file_system", "content": "line one\nline two"},
        ])
        assert flags == [True, True, False]

    def test_none_entries_are_skipped(self):
        assert tool_failure_flags([None, {"name": "x", "content": "ok"}]) == [False]


class TestShapeRule:
    def test_live_case_fires(self):
        assert unacknowledged_total_failure(
            tools=[{"name": "file_system", "content": _FS_ERR},
                   {"name": "execute", "content": _EXEC_ERR},
                   {"name": "file_system", "content": _FS_ERR}],
            final_response="0", user_request=_LIVE_REQUEST) is True

    def test_all_failed_but_acknowledged_does_not_fire(self):
        assert unacknowledged_total_failure(
            tools=[{"name": "file_system", "content": _FS_ERR}],
            final_response="That file does not exist in my sandbox.",
            user_request=_LIVE_REQUEST) is False

    def test_partial_failure_with_recovery_does_not_fire(self):
        """ALL, never ANY. A turn where one tool fails, the agent recovers
        through another, and the answer is right is a GOOD turn."""
        assert unacknowledged_total_failure(
            tools=[{"name": "file_system", "content": _FS_ERR},
                   {"name": "execute", "content": "9536"}],
            final_response="9536", user_request=_LIVE_REQUEST) is False

    def test_no_tool_calls_never_fires(self):
        assert unacknowledged_total_failure(
            tools=[], final_response="42", user_request="what is 6*7") is False

    def test_reads_a_trajectory(self):
        traj = Trajectory(
            id="t", user_request=_LIVE_REQUEST, final_response="0",
            tool_calls=[ToolCall(name="file_system", result=_FS_ERR,
                                 error="does not exist")])
        assert unacknowledged_total_failure_for_trajectory(traj) is True
        traj.final_response = "That file does not exist."
        assert unacknowledged_total_failure_for_trajectory(traj) is False

    def test_trajectory_adapter_is_none_safe(self):
        assert unacknowledged_total_failure_for_trajectory(None) is False


# ══════════════════════════════════════════════════════════════════════
# The ladder
# ══════════════════════════════════════════════════════════════════════

class TestResolveTurnOutcomeRule2b:
    def test_unacknowledged_total_failure_is_not_upgraded(self):
        assert resolve_turn_outcome(
            current=U, verifier="passed", execution_failed=True,
            unacked_total_failure=True) == F

    def test_late_backfill_shape_blocks_the_structural_upgrade(self):
        assert resolve_turn_outcome(
            current=F, current_reason=STRUCTURAL_FAILURE_REASON,
            verifier="passed", unacked_total_failure=True) == F

    def test_honest_structural_failure_still_upgrades(self):
        """The 2026-07-31 rule, on BOTH of its entry points. This is the
        discriminating twin of the two tests above: revert rule 2b and the
        pair above goes red; delete the acknowledgment check and this one
        does."""
        assert resolve_turn_outcome(
            current=U, verifier="passed", execution_failed=True) == P
        assert resolve_turn_outcome(
            current=F, current_reason=STRUCTURAL_FAILURE_REASON,
            verifier="passed") == P

    def test_rule_never_manufactures_a_failed(self):
        """Non-manufacturing: with a clean strike ledger and no structural
        FAILED to stand on, the flag changes nothing. This bounds the cost
        of the corpus text sniffer's known false positives (an
        `EXIT CODE: 1` banner nested inside an otherwise successful tool
        payload — live trajectory of 2026-07-08T11:34)."""
        assert resolve_turn_outcome(
            current=U, verifier="passed",
            unacked_total_failure=True) == P

    def test_refute_still_outranks_the_shape_rule(self):
        assert resolve_turn_outcome(
            current=U, verifier="failed", execution_failed=True,
            unacked_total_failure=True) == F

    def test_shape_heuristic_failed_stays_unupgradable(self):
        assert resolve_turn_outcome(
            current=F, current_reason="browser selector '#go' used 5×",
            verifier="passed") == F
        assert resolve_turn_outcome(
            current=F, current_reason="browser selector '#go' used 5×",
            verifier="passed", unacked_total_failure=True) == F

    def test_current_reason_defaults_preserve_legacy_callers(self):
        """Callers that don't pass `current_reason` keep the pre-2026-08-04
        "an existing FAILED is never upgraded" behaviour exactly."""
        assert resolve_turn_outcome(current=F, verifier="passed") == F
        assert resolve_turn_outcome(current=P) == P
        assert resolve_turn_outcome(current=None) == U


# ══════════════════════════════════════════════════════════════════════
# Integration — the corpus write path (shared by BOTH delivery paths)
# ══════════════════════════════════════════════════════════════════════

class _FakeCollector:
    def __init__(self):
        self.appended = []
        self.updates = []

    def append(self, traj):
        self.appended.append(traj)

    def update_outcome(self, tid, outcome, reason="", source="", **kw):
        self.updates.append((tid, outcome, reason, source))
        return True


def _backfill(agent, tid, outcome):
    """`_backfill_trajectory_outcome` fires the sidecar write through
    `spawn_bg`, which needs a running loop — drive it inside one and DRAIN
    the background tasks before returning: since §4BF R2 the cache
    mutation + success log ride the deferred write's result on a worker
    thread, so a single loop tick is no longer enough."""
    import asyncio
    from ghost_agent.utils import logging as _glog

    async def _go():
        agent._backfill_trajectory_outcome(tid, outcome)
        for _ in range(3):   # drain spawn_bg work (incl. thread joins)
            pending = [t for t in getattr(_glog, "_BG_TASKS", set())
                       if not t.done()]
            if not pending:
                break
            await asyncio.gather(*pending, return_exceptions=True)
    asyncio.run(_go())


def _agent():
    from ghost_agent.core.agent import GhostAgent
    agent = GhostAgent.__new__(GhostAgent)
    col = _FakeCollector()
    agent.context = types.SimpleNamespace(
        trajectory_collector=col, self_model=None, profile_memory=None,
        _recent_calib_for_correction=None,
        _recent_trajectories_for_correction={},
        _surfaced_triggers_by_traj=None, skill_memory=None)
    return agent, col


class TestSyncFinalizePath:
    """The NON-streamed path folds the verdict in at write time, so the
    shape rule has to fire inside `_record_turn_trajectory`."""

    def test_live_case_is_recorded_failed(self):
        agent, col = _agent()
        agent._record_turn_trajectory(
            messages=_msgs([_FS_ERR, _EXEC_ERR, _FS_ERR],
                           names=["file_system", "execute", "file_system"]),
            final_content="0", req_id="03b96c28", model="m",
            trajectory_id="f78c8b33", user_request=_LIVE_REQUEST,
            verifier="passed", execution_failed=True,
        )
        assert col.appended[0].outcome == F
        # cause-qualified since 2026-08-10 — assert the contract, not the
        # bare string (`is_structural_reason` is what production matches on)
        assert is_structural_reason(col.appended[0].failure_reason)

    def test_honest_failure_report_still_passes(self):
        agent, col = _agent()
        agent._record_turn_trajectory(
            messages=_msgs([_FS_ERR], reply="That file does not exist."),
            final_content="That file does not exist.", req_id="r", model="m",
            trajectory_id="t", user_request=_LIVE_REQUEST,
            verifier="passed", execution_failed=True,
        )
        assert col.appended[0].outcome == P

    def test_recovered_turn_still_passes(self):
        agent, col = _agent()
        agent._record_turn_trajectory(
            messages=_msgs([_FS_ERR, "9536"],
                           names=["file_system", "execute"], reply="9536"),
            final_content="9536", req_id="r", model="m", trajectory_id="t",
            user_request=_LIVE_REQUEST, verifier="passed",
            execution_failed=True,
        )
        assert col.appended[0].outcome == P


class TestStreamedFinalizePath:
    """The STREAMED path (the web UI always streams) returns the SSE
    generator before finalize runs: it records with ``verifier=None`` and
    the verdict lands later through `_backfill_trajectory_outcome`. A fix
    live on the sync path and dark here is this project's signature defect,
    so the whole sequence is replayed."""

    def _record_streamed(self, agent, *, reply, results, names, request):
        agent._record_turn_trajectory(
            messages=_msgs(results, names=names, request=request, reply=reply),
            final_content=reply, req_id="03b96c28", model="m",
            trajectory_id="f78c8b33", user_request=request,
            verifier=None, execution_failed=True,
        )

    def test_streamed_record_site_uses_the_shared_function(self):
        """Source pin: the streamed site must keep routing through
        `_record_turn_trajectory` (where the rule lives) rather than
        growing its own consolidation."""
        import inspect
        from ghost_agent.core import agent as agent_mod
        src = inspect.getsource(agent_mod)
        assert src.count("self._record_turn_trajectory(") == 2, (
            "a third trajectory record site appeared — it must route "
            "through the same consolidation")

    def test_late_confirmed_is_withheld(self, capsys):
        agent, col = _agent()
        self._record_streamed(
            agent, reply="0", results=[_FS_ERR, _EXEC_ERR, _FS_ERR],
            names=["file_system", "execute", "file_system"],
            request=_LIVE_REQUEST)
        traj = col.appended[0]
        # The reason is now CAUSE-QUALIFIED ("structural failure: <tool>: …"),
        # so match the CONTRACT rather than the bare string — that is the
        # whole point of `is_structural_reason`.
        assert traj.outcome == F and is_structural_reason(traj.failure_reason)
        agent.context._recent_trajectories_for_correction = {"k": traj}
        _backfill(agent, "f78c8b33", P)
        assert traj.outcome == F, "the late CONFIRMED laundered a fabrication"
        out = capsys.readouterr().out
        # The sidecar write is fire-and-forget, so assert on the decision
        # the operator sees rather than on a background task.
        assert "WITHHELD" in out
        assert "backfilled into the corpus" not in out

    def test_late_confirmed_still_upgrades_an_honest_report(self, capsys):
        agent, col = _agent()
        honest = "That file does not exist in my sandbox."
        self._record_streamed(
            agent, reply=honest, results=[_FS_ERR], names=["file_system"],
            request=_LIVE_REQUEST)
        traj = col.appended[0]
        assert traj.outcome == F
        agent.context._recent_trajectories_for_correction = {"k": traj}
        _backfill(agent, "f78c8b33", P)
        assert traj.outcome == P
        assert traj.failure_reason == ""
        out = capsys.readouterr().out
        assert "backfilled into the corpus" in out and "WITHHELD" not in out

    def test_late_confirmed_still_upgrades_a_recovered_turn(self):
        agent, col = _agent()
        self._record_streamed(
            agent, reply="9536", results=[_FS_ERR, "9536"],
            names=["file_system", "execute"], request=_LIVE_REQUEST)
        traj = col.appended[0]
        agent.context._recent_trajectories_for_correction = {"k": traj}
        _backfill(agent, "f78c8b33", P)
        assert traj.outcome == P

    def test_withheld_pass_does_not_tick_a_lesson_success(self):
        """The stashed-lesson drain runs before the direction guard (so a
        legitimate pass is never lost); it must book the RESOLVED outcome,
        not the raw verdict."""
        agent, col = _agent()
        self._record_streamed(
            agent, reply="0", results=[_FS_ERR], names=["file_system"],
            request=_LIVE_REQUEST)
        traj = col.appended[0]
        agent.context._recent_trajectories_for_correction = {"k": traj}
        booked = []
        agent._flush_stashed_lesson_outcome = (
            lambda tid, success: booked.append(success))
        _backfill(agent, "f78c8b33", P)
        assert booked == [False]

    def test_cache_miss_still_drains_the_stash_as_a_success(self):
        """Deliberate: the shape rule cannot be evaluated without the
        record, so a cache-evicted trajectory keeps the pre-2026-08-04
        behaviour. Ordering matters — the flush must stay AHEAD of the
        cache-miss return (2026-07-26 lost-success-tick bug); a first draft
        of this change moved the return earlier and re-opened it."""
        agent, _col = _agent()
        agent.context._recent_trajectories_for_correction = {}
        booked = []
        agent._flush_stashed_lesson_outcome = (
            lambda tid, success: booked.append(success))
        _backfill(agent, "gone", P)
        assert booked == [True]


# ══════════════════════════════════════════════════════════════════════
# The operator-facing line
# ══════════════════════════════════════════════════════════════════════

class TestTurnOutcomeLabel:
    def test_pass_is_ignored_for_an_unacknowledged_total_failure(self):
        from ghost_agent.core.agent import GhostAgent
        L = GhostAgent._turn_outcome_label
        assert L(verifier_failed=False, verifier_passed=True,
                 budget_exhausted=False, exec_terminal=True,
                 unacked_total_failure=True) == "failed"

    def test_honest_failure_still_reads_verified(self):
        from ghost_agent.core.agent import GhostAgent
        L = GhostAgent._turn_outcome_label
        assert L(verifier_failed=False, verifier_passed=True,
                 budget_exhausted=False, exec_terminal=True,
                 unacked_total_failure=False) == "verified"

    def test_late_correction_reads_the_flag_off_the_snapshot(self, capsys):
        """The tools and the reply are gone by the time a late verdict
        lands, so the flag rides the snapshot ring — recomputing it there
        would be a second derivation of the same rule."""
        agent, _col = _agent()
        agent.context._recent_turn_outcome = {"t": {
            "state": "failed", "confidence": 0.8, "tools": ["file_system"],
            "chars": 1, "exec_failures": 3, "exec_terminal": True,
            "budget_exhausted": False, "unacked_total_failure": True}}
        agent._emit_late_outcome_correction("t", "passed")
        # Label does not flip, so nothing is announced — the stream keeps
        # saying `failed`, which is what the corpus records.
        assert "CORRECTED" not in capsys.readouterr().out

    def test_late_correction_still_fires_for_an_honest_failure(self, capsys):
        agent, _col = _agent()
        agent.context._recent_turn_outcome = {"t": {
            "state": "failed", "confidence": 0.9, "tools": ["file_system"],
            "chars": 40, "exec_failures": 1, "exec_terminal": True,
            "budget_exhausted": False, "unacked_total_failure": False}}
        agent._emit_late_outcome_correction("t", "passed")
        assert "CORRECTED failed → verified" in capsys.readouterr().out


# ══════════════════════════════════════════════════════════════════════
# Calibration mirror
# ══════════════════════════════════════════════════════════════════════

_READING = types.SimpleNamespace(
    composite=0.8, entropy_component=0.5, competence_component=0.5,
    uncertainty_pressure=0.0, raw_pre_penalty_composite=0.8,
    pre_penalty_composite=0.8, entropy_observed=False,
    effort_component=0.5, effort_observed=False)


class _FakeTracker:
    def __init__(self):
        self.recorded = {}

    def record(self, **kw):
        self.recorded = kw


class TestCalibrationGrade:
    def test_unacknowledged_total_failure_is_not_graded_perfect(self):
        from ghost_agent.core.calibration import grade_turn_outcome
        assert grade_turn_outcome(
            verifier_verdict="passed", execution_failure_count=3,
            unacked_total_failure=True) < 1.0

    def _record(self, reply):
        """Drive the REAL `_record_calibration_safe`, which both delivery
        paths share, rather than only the pure grader."""
        import asyncio
        from ghost_agent.core.agent import GhostAgent
        agent = GhostAgent.__new__(GhostAgent)
        tracker = _FakeTracker()
        agent.context = types.SimpleNamespace(
            skill_memory=None, calibration_tracker=tracker,
            _calib_pending=("r", _READING), uncertainty_tracker=None,
            metacog=None, last_confidence=None)
        asyncio.run(agent._record_calibration_safe(
            req_id="r", tools_run=[{"name": "file_system", "content": _FS_ERR}],
            verifier_backfill=("passed", ""), execution_failure_count=3,
            budget_exhausted=False, final_ai_content=reply,
            user_request=_LIVE_REQUEST))
        return tracker.recorded.get("outcome")

    def test_live_case_is_not_a_perfect_calibration_sample(self):
        """Grading req 03b96c28's fabricated `0` as 1.0 teaches the
        confidence model that a confident answer over three broken tools is
        what success looks like."""
        assert self._record("0") < 1.0

    def test_honest_report_is_still_a_perfect_calibration_sample(self):
        assert self._record("That file does not exist.") == 1.0

    def test_honest_failure_report_keeps_its_one(self):
        from ghost_agent.core.calibration import grade_turn_outcome
        assert grade_turn_outcome(
            verifier_verdict="passed", execution_failure_count=3) == 1.0

    def test_refute_is_still_the_hard_zero(self):
        from ghost_agent.core.calibration import grade_turn_outcome
        assert grade_turn_outcome(
            verifier_verdict="failed", execution_failure_count=3,
            unacked_total_failure=True) == 0.0


# ══════════════════════════════════════════════════════════════════════
# The remaining finalize consumers of the same ladder
# ══════════════════════════════════════════════════════════════════════

class TestFinalizeMirrors:
    """`_finalize_and_return` feeds four consumers off one ladder: the
    corpus, calibration, the agent's own diary, and the operator's line.
    These pins guard the two that can only be reached through the full
    finalize chain."""

    @staticmethod
    def _finalize_src():
        import inspect
        from ghost_agent.core.agent import GhostAgent
        return inspect.getsource(GhostAgent._finalize_and_return)

    def test_selfhood_backfill_applies_the_same_rule(self):
        """The diary is read back into the wake-up prefix and the
        competence prior, so a withheld PASS must not enter it as a
        success — `record_outcome` would otherwise write the raw verdict."""
        src = self._finalize_src()
        assert 'if _bf_outcome == "passed" and _unacked_turn:' in src
        assert '_bf_outcome = "failed"' in src

    def test_flag_is_computed_before_the_correction_prepend(self):
        """ORDER IS LOAD-BEARING: `_take_active_correction()` prepends the
        verifier's own prose to `final_ai_content`, and that text is full
        of failure vocabulary. Computing the flag after it would let a
        deferred correction banner make a fabricated reply read as an
        acknowledgment."""
        src = self._finalize_src()
        assert src.index("_unacked_turn = _exec_terminal") < \
            src.index("self._take_active_correction()")

    def test_episode_label_applies_the_same_rule(self):
        """`_record_episode_safe`'s success label feeds the LLM that mints
        playbook lessons and gates `search_recoveries`, so a fabrication
        must not be stored as a reusable recovery."""
        import asyncio
        from ghost_agent.core.agent import GhostAgent

        def _label(reply):
            captured = {}

            class _EM:
                def record_episode(self, **kw):
                    captured.update(kw)
            agent = GhostAgent.__new__(GhostAgent)
            agent.context = types.SimpleNamespace(episodic_memory=_EM())
            asyncio.run(agent._record_episode_safe(
                _LIVE_REQUEST,
                [{"name": "file_system", "content": _FS_ERR}],
                reply, verifier_verdict="passed",
                execution_failure_count=3, req_id="r"))
            return captured.get("success")

        assert _label("0") is False
        assert _label("That file does not exist.") is True

    def test_mirrors_are_non_manufacturing_too(self):
        """A clean strike ledger keeps the 1.0 / the success even when the
        text sniffer thinks every result looks like an error — the same
        bound `resolve_turn_outcome` puts on rule 2b."""
        import asyncio
        from ghost_agent.core.agent import GhostAgent
        agent = GhostAgent.__new__(GhostAgent)
        tracker = _FakeTracker()
        agent.context = types.SimpleNamespace(
            skill_memory=None, calibration_tracker=tracker,
            _calib_pending=("r", _READING), uncertainty_tracker=None,
            metacog=None, last_confidence=None)
        asyncio.run(agent._record_calibration_safe(
            req_id="r", tools_run=[{"name": "file_system", "content": _FS_ERR}],
            verifier_backfill=("passed", ""), execution_failure_count=0,
            budget_exhausted=False, final_ai_content="0",
            user_request=_LIVE_REQUEST))
        assert tracker.recorded.get("outcome") == 1.0

    def test_calibration_is_given_the_request(self):
        """Both call sites must pass `user_request` — the instructed-literal
        escape cannot work without it, and a site that forgets it silently
        re-punishes the 2026-07-31 probe shape."""
        import inspect
        from ghost_agent.core.agent import GhostAgent
        finalize = self._finalize_src()
        stream = inspect.getsource(GhostAgent._stream_final_generation)
        assert "_record_calibration_safe(" in finalize
        assert "_record_calibration_safe(" in stream
        for src in (finalize, stream):
            assert "user_request=last_user_content" in src


# ══════════════════════════════════════════════════════════════════════
# `passed` + `structural failure` is an impossible pair
# ══════════════════════════════════════════════════════════════════════

class TestCorrectionsOverlayCoherence:
    """Live defect: trajectory f78c8b33 read back ``outcome=passed`` WITH
    ``failure_reason="structural failure"`` still stamped. The writer had
    cleared the reason in memory; the READ-side overlay re-applied the
    on-disk one. Every consumer that branches on failure_reason (the
    honest-failure upgrade guard itself, postmortem fingerprints, the
    fixture miner) saw a contradiction."""

    def _collector(self, tmp_path):
        from ghost_agent.distill.collector import TrajectoryCollector
        return TrajectoryCollector(root=tmp_path, session_id="s")

    def _write(self, col, **kw):
        col.append(Trajectory(**kw))

    def test_passed_correction_clears_the_stale_reason_on_read(self, tmp_path):
        col = self._collector(tmp_path)
        self._write(col, id="x1", outcome=F,
                    failure_reason=STRUCTURAL_FAILURE_REASON,
                    final_response="that file does not exist")
        col.update_outcome("x1", P, source="verifier_late")
        got = [t for t in col.iter_trajectories() if t.id == "x1"][0]
        assert (got.outcome, got.failure_reason) == (P, "")

    def test_legacy_sidecar_row_with_a_reason_is_still_cleared(self, tmp_path):
        col = self._collector(tmp_path)
        self._write(col, id="x2", outcome=F, failure_reason="whatever",
                    final_response="r")
        # Simulate a pre-guard sidecar row: (passed, non-empty reason).
        import json
        p = col.root / "corrections.jsonl"
        p.write_text(json.dumps({
            "trajectory_id": "x2", "outcome": P, "reason": "stale",
            "source": "legacy", "timestamp": "t"}) + "\n", encoding="utf-8")
        got = [t for t in col.iter_trajectories() if t.id == "x2"][0]
        assert (got.outcome, got.failure_reason) == (P, "")

    def test_writer_refuses_to_persist_the_pair(self, tmp_path):
        import json
        col = self._collector(tmp_path)
        col.update_outcome("x3", P, reason=STRUCTURAL_FAILURE_REASON,
                           source="hypothetical")
        rows = [json.loads(l) for l in
                (col.root / "corrections.jsonl").read_text().splitlines() if l]
        assert rows[0]["outcome"] == P and rows[0]["reason"] == ""

    def test_failed_correction_keeps_carrying_its_reason(self, tmp_path):
        col = self._collector(tmp_path)
        self._write(col, id="x4", outcome=U, final_response="r")
        col.update_outcome("x4", F, reason="verifier refuted (late): bad",
                           source="verifier_late")
        got = [t for t in col.iter_trajectories() if t.id == "x4"][0]
        assert got.outcome == F and "verifier refuted" in got.failure_reason

    def test_failed_correction_does_not_overwrite_an_existing_reason(self, tmp_path):
        col = self._collector(tmp_path)
        self._write(col, id="x5", outcome=F, failure_reason="original",
                    final_response="r")
        col.update_outcome("x5", F, reason="later", source="verifier_late")
        got = [t for t in col.iter_trajectories() if t.id == "x5"][0]
        assert got.failure_reason == "original"

    def test_unknown_overlay_may_still_explain_itself(self, tmp_path):
        """`unknown` is deliberately outside the guard — the operator
        overlay uses (unknown, reason) to de-label a record while saying
        why."""
        col = self._collector(tmp_path)
        self._write(col, id="x6", outcome=P, final_response="r")
        col.update_outcome("x6", U, reason="operator: unattributable",
                           source="operator_overlay")
        got = [t for t in col.iter_trajectories() if t.id == "x6"][0]
        assert got.outcome == U and "unattributable" in got.failure_reason

    def test_no_live_record_can_read_back_passed_with_a_reason(self, tmp_path):
        """The invariant, stated as an invariant."""
        col = self._collector(tmp_path)
        for i, (outcome, reason) in enumerate([
                (F, STRUCTURAL_FAILURE_REASON), (F, "verifier refuted"),
                (U, ""), (P, "")]):
            self._write(col, id=f"y{i}", outcome=outcome,
                        failure_reason=reason, final_response="r")
            col.update_outcome(f"y{i}", P, source="verifier_late")
        for t in col.iter_trajectories():
            assert not (t.outcome == P and t.failure_reason), (t.id, t.failure_reason)


# ══════════════════════════════════════════════════════════════════════
# Kill switch
# ══════════════════════════════════════════════════════════════════════

class TestKillSwitch:
    """`GHOST_UNACKED_FAILURE_GATE=0` restores the EXACT pre-2026-08-04
    behaviour: a verifier `passed` outranks a structural execution failure
    unconditionally. Read per call (no restart needed) and read in exactly
    ONE place, so it cannot be live on one path and dark on another."""

    def test_default_is_on(self, monkeypatch):
        monkeypatch.delenv("GHOST_UNACKED_FAILURE_GATE", raising=False)
        assert unacked_failure_gate_enabled() is True

    @pytest.mark.parametrize("val", ["0", "false", "NO", " 0 "])
    def test_off_values(self, monkeypatch, val):
        monkeypatch.setenv("GHOST_UNACKED_FAILURE_GATE", val)
        assert unacked_failure_gate_enabled() is False

    def test_off_restores_the_prior_write_path_behaviour(self, monkeypatch):
        monkeypatch.setenv("GHOST_UNACKED_FAILURE_GATE", "0")
        agent, col = _agent()
        agent._record_turn_trajectory(
            messages=_msgs([_FS_ERR, _EXEC_ERR, _FS_ERR],
                           names=["file_system", "execute", "file_system"]),
            final_content="0", req_id="r", model="m", trajectory_id="t",
            user_request=_LIVE_REQUEST, verifier="passed",
            execution_failed=True,
        )
        assert col.appended[0].outcome == P

    def test_off_restores_the_prior_late_backfill_behaviour(self, monkeypatch):
        monkeypatch.setenv("GHOST_UNACKED_FAILURE_GATE", "0")
        agent, col = _agent()
        traj = Trajectory(
            id="t", outcome=F, failure_reason=STRUCTURAL_FAILURE_REASON,
            user_request=_LIVE_REQUEST, final_response="0",
            tool_calls=[ToolCall(name="file_system", result=_FS_ERR,
                                 error="does not exist")])
        agent.context._recent_trajectories_for_correction = {"k": traj}
        _backfill(agent, "t", P)
        assert traj.outcome == P and traj.failure_reason == ""

    def test_off_restores_the_prior_operator_line(self, monkeypatch):
        monkeypatch.setenv("GHOST_UNACKED_FAILURE_GATE", "0")
        assert unacknowledged_total_failure(
            tools=[{"name": "file_system", "content": _FS_ERR}],
            final_response="0", user_request=_LIVE_REQUEST) is False

    def test_the_switch_is_read_in_exactly_one_place(self):
        """A second read site is a second thing to forget to flip."""
        import inspect
        from ghost_agent.distill import outcome_heuristics as oh
        from ghost_agent.core import agent as agent_mod
        from ghost_agent.core import calibration as calib_mod
        assert inspect.getsource(oh).count("UNACKED_FAILURE_GATE_ENV,") == 1
        for mod in (agent_mod, calib_mod):
            assert "GHOST_UNACKED_FAILURE_GATE" not in inspect.getsource(mod) \
                .replace("GHOST_UNACKED_FAILURE_GATE=0 disables", ""), (
                    f"{mod.__name__} reads the switch directly")
