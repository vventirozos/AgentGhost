"""Regression tests for bug-hunt unit 18 (core-agent: agent.py + agent_qwen.py).

See BUGHUNT.md. Fixed bugs pinned here:

Carried-over from unit 15:
 - The per-tool metacog COMPETENCE detector classified failure via the
   "EXIT CODE: 1"/"2" substring, so exit codes 3-9 and multi-digit (127, 130)
   were recorded as competence SUCCESS. Now any non-zero exit code (regex) is a
   failure — matching the execute result-classifier. (The sibling
   "streaming outcome-penalty finalize skip" finding was DISPROVEN: handle_chat
   returns the stream generator before the finalize block is reached, so the
   guard never skips — see BUGHUNT.md deferred.)

Unit 18 (core-agent):
 - extract_json_from_text honours its "return {} on every failure mode"
   contract for non-str input, and no longer rewrites true/false/null that
   appear INSIDE a string value (AST fallback).
 - The biological watchdog HARD LOCK also defers on an in-flight user REQUEST
   (foreground_requests), not just an in-flight LLM call (foreground_tasks).
 - The three idle training phases (skills-auto extract/consolidate, PRM train,
   router train) run off the event loop via asyncio.to_thread.
 - The dream-eligibility collection.get() is guarded so a store error skips
   only the dream, not the whole tick.
 - The thinking-cap loop-breaker aborts on the SECOND event (documented design),
   not the third.
 - A deferred async correction with an EMPTY conversation fingerprint no longer
   wildcard-surfaces into an unrelated conversation; and the conversation tag is
   computed once from the un-pruned history so the record + consume sides agree
   (corrections are no longer lost in long/pruned sessions).
"""

import re
import time
import types

import pytest

from ghost_agent.core.agent import extract_json_from_text, GhostAgent
import ghost_agent.core.agent as agent_mod


# ══════════════════════════════════════════════════════════════════════
# extract_json_from_text — contract + AST-fallback corruption
# ══════════════════════════════════════════════════════════════════════

class TestExtractJson:
    def test_non_str_returns_empty_dict(self):
        # Contract: {} on EVERY failure mode. A backend that returned
        # content:null used to raise TypeError out of the pre-try re.sub.
        assert extract_json_from_text(None) == {}
        assert extract_json_from_text(123) == {}
        assert extract_json_from_text(["not", "a", "string"]) == {}

    def test_keyword_inside_string_not_corrupted(self):
        # Single-quoted → fails json.loads → AST fallback. The words
        # true/null sit INSIDE a value string and must survive verbatim.
        out = extract_json_from_text("{'answer': 'the statement is true and not null'}")
        assert out == {"answer": "the statement is true and not null"}

    def test_bare_json_keywords_still_convert(self):
        # A python-dict-ish blob with bare JSON keywords as VALUES still parses.
        out = extract_json_from_text('{"ok": true, "x": null, "y": false}')
        assert out == {"ok": True, "x": None, "y": False}

    def test_strict_json_unaffected(self):
        assert extract_json_from_text('{"a": 1, "b": "hi"}') == {"a": 1, "b": "hi"}

    def test_false_substring_in_string_survives(self):
        out = extract_json_from_text("{'note': 'this is a false alarm, truly'}")
        assert out == {"note": "this is a false alarm, truly"}


# ══════════════════════════════════════════════════════════════════════
# Biological watchdog HARD LOCK — defers on an active user request
# ══════════════════════════════════════════════════════════════════════

class TestWatchdogHardLock:
    def _agent_with_fg(self, *, tasks, requests):
        ctx = types.SimpleNamespace(
            memory_system=object(),  # truthy → passes the first gate
            llm_client=types.SimpleNamespace(
                foreground_tasks=tasks, foreground_requests=requests
            ),
        )
        agent = GhostAgent.__new__(GhostAgent)
        agent.context = ctx
        return agent

    async def test_active_request_between_llm_calls_defers_tick(self):
        # foreground_tasks==0 (no LLM call in flight) but a user REQUEST is
        # mid-turn between its LLM calls. Pre-fix the tick passed the gate and
        # began installing phase anchors; now it must return early.
        agent = self._agent_with_fg(tasks=0, requests=1)
        await agent._biological_tick()
        # The cooldown anchors are installed AFTER the gate — their absence
        # proves the tick returned at the HARD LOCK.
        assert not hasattr(agent, "_last_journal_at")

    async def test_active_llm_call_still_defers(self):
        agent = self._agent_with_fg(tasks=1, requests=0)
        await agent._biological_tick()
        assert not hasattr(agent, "_last_journal_at")


# ══════════════════════════════════════════════════════════════════════
# Deferred async-correction scoping — empty fp + stable-fp threading
# ══════════════════════════════════════════════════════════════════════

class TestPendingCorrectionScoping:
    def _agent(self, corrections):
        agent = GhostAgent.__new__(GhostAgent)
        agent._pending_corrections = corrections
        agent._active_correction = ""
        agent._correction_active_this_turn = False
        return agent

    def test_empty_conv_not_surfaced_into_unrelated_conversation(self):
        # A correction whose recorded conv fingerprint is "" must NOT wildcard
        # into whatever conversation happens to be next (fail-safe drop).
        agent = self._agent([{"note": "stale", "conv": "", "ts": time.monotonic()}])
        msgs = [{"role": "user", "content": "an unrelated question"}]
        agent._consume_pending_corrections(msgs, conv_fp="fpA")
        assert agent._correction_active_this_turn is False
        assert agent._take_active_correction() == ""

    def test_matching_conv_surfaces(self):
        agent = self._agent([{"note": "that was wrong", "conv": "fpA", "ts": time.monotonic()}])
        msgs = [{"role": "user", "content": "hello"}]
        agent._consume_pending_corrections(msgs, conv_fp="fpA")
        assert agent._correction_active_this_turn is True
        assert "that was wrong" in agent._take_active_correction()

    def test_stable_conv_fp_overrides_message_fingerprint(self):
        # The passed-in stable fp (computed once from the un-pruned history) is
        # what matches — NOT a fingerprint recomputed from these messages. This
        # is the fix for corrections lost when pruning changed the opener.
        corr = [{"note": "correction", "conv": "fpA", "ts": time.monotonic()}]
        msgs = [{"role": "user", "content": "totally different opener"}]
        # message's own fingerprint is definitely not "fpA"
        assert agent_mod.GhostAgent._conversation_fingerprint(
            GhostAgent.__new__(GhostAgent), msgs) != "fpA"
        # With the stable fp threaded in → surfaces.
        a1 = self._agent([dict(corr[0])])
        a1._consume_pending_corrections(msgs, conv_fp="fpA")
        assert a1._correction_active_this_turn is True
        # Without it (recomputing from these messages) → does NOT match.
        a2 = self._agent([dict(corr[0])])
        a2._consume_pending_corrections(msgs, conv_fp=None)
        assert a2._correction_active_this_turn is False

    def test_nonmatching_conv_held_in_queue(self):
        agent = self._agent([{"note": "other conv", "conv": "fpB", "ts": time.monotonic()}])
        msgs = [{"role": "user", "content": "hi"}]
        agent._consume_pending_corrections(msgs, conv_fp="fpA")
        assert agent._correction_active_this_turn is False
        # still queued for its own conversation
        assert any(c.get("conv") == "fpB" for c in agent._pending_corrections)


# ══════════════════════════════════════════════════════════════════════
# Source pins — fixes that live deep in the streaming/loop hot path
# ══════════════════════════════════════════════════════════════════════

class TestSourcePins:
    src = None

    @classmethod
    def setup_class(cls):
        import inspect
        cls.src = inspect.getsource(agent_mod)

    def test_competence_detector_uses_nonzero_exit_regex(self):
        # Carried finding #1: any non-zero exit code is a failure.
        assert "int(_mc_exit.group(1)) != 0" in self.src
        # The old 1/2-only substring predicate must be gone everywhere.
        assert '"EXIT CODE: 1" in str_res' not in self.src
        assert '"EXIT CODE: 2" in str_res' not in self.src

    def test_thinking_cap_aborts_on_second_event(self):
        assert "thinking_cap_events >= 2" in self.src
        assert "thinking_cap_events >= 3" not in self.src

    def test_idle_training_phases_offloaded_to_thread(self):
        assert re.search(r"to_thread\(\s*extract_candidates", self.src)
        assert re.search(r"to_thread\(\s*consolidate", self.src)
        assert re.search(r"to_thread\(\s*\n?\s*trainer\.run", self.src)

    def test_hard_lock_checks_foreground_requests(self):
        assert "getattr(_lc, 'foreground_requests', 0) > 0" in self.src

    def test_dream_eligibility_get_is_guarded(self):
        assert "dream eligibility get() failed" in self.src
