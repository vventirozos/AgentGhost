"""Probe-before-hypothesis loop-breaker changes (2026-07-08).

Chess session E8: five identical re-reads of index.html across 536s while
the model theorized about a URL that one curl would have settled. Changes
pinned here:

  * the no-progress counter now trips at the SECOND identical read
    (threshold=2 at the note_action call site; was 3),
  * the hard abort fires at >=3 (was >=5),
  * every steer/abort message leads with EVIDENCE-GATHERING options —
    probe the resource, apply the change, or ask the user for the exact
    error — instead of only "trust what you have".
"""
import inspect
import re

from ghost_agent.core.strikes import StrikeLedger


class TestNoteActionMechanics:
    def test_threshold_2_trips_on_second_identical_read(self):
        s = StrikeLedger()
        _, c1, t1 = s.note_action("file_system", "index.html", "fp-abc", threshold=2)
        _, c2, t2 = s.note_action("file_system", "index.html", "fp-abc", threshold=2)
        assert (c1, t1) == (1, False)
        assert (c2, t2) == (2, True)

    def test_different_result_fingerprint_does_not_trip(self):
        # A re-read that returns DIFFERENT content is new information.
        s = StrikeLedger()
        s.note_action("file_system", "index.html", "fp-1", threshold=2)
        _, count, tripped = s.note_action("file_system", "index.html", "fp-2",
                                          threshold=2)
        assert not tripped and count == 1


class TestCallSiteWiring:
    def _src(self):
        import ghost_agent.core.agent as agent_mod
        return inspect.getsource(agent_mod)

    def test_note_action_called_with_threshold_2(self):
        src = self._src()
        idx = src.index("strikes.note_action(")
        assert "threshold=2" in src[idx:idx + 300]

    def test_hard_abort_two_tier(self):
        # 2026-07-20: the hard stop is two-tier — general tools abort at 3,
        # read/write-exempt tools keep the documented >=5 backstop
        # (READWRITE_HARD_STOP from strikes.py) so one post-steer re-read
        # can't abort a pending write.
        src = self._src()
        idx = src.index("_noprogress_trip is not None and not force_stop")
        window = src[idx:idx + 1400]
        assert "_acnt >= _hard_n" in window
        assert "READWRITE_HARD_STOP" in window
        assert "else 3" in window

    def test_steers_lead_with_evidence_gathering(self):
        src = self._src()
        idx = src.index("GATHER NEW EVIDENCE")
        window = src[idx - 2000:idx + 2500]
        # The read/write steer names all three ways forward.
        assert "GATHER NEW EVIDENCE" in window
        assert "ASK THE USER" in window
        # The finalize steer tells it to request the one settling artifact
        # instead of guessing.
        assert "devtools" in src
        # The hard-abort message asks for evidence, not hand-waving.
        assert "ATTEMPT_ABORTED_NO_PROGRESS" in src
        abort_idx = src.index("ATTEMPT_ABORTED_NO_PROGRESS")
        abort_msg = src[abort_idx:abort_idx + 900]
        assert "evidence" in abort_msg


class TestTaskUpdateJudgmentGateWiring:
    def test_done_gate_audits_and_offers_override(self):
        import ghost_agent.tools.projects as projects_mod
        src = inspect.getsource(projects_mod)
        assert "constraint_gate" in src
        assert "CONSTRAINT-OVERRIDE:" in src
        assert "judged_violations" in src
        # The refusal payload instructs fix-first, override only with
        # explicit user approval.
        assert "agent_instruction_violation" in src
