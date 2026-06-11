"""Tests for the repro-first nudge on bug-report turns and the
grounding directive after a killed thinking loop.

Regression target (req 70): "when i click 'enter the corridor' nothing
happens" triggered ~230 seconds of source re-reading and six wrong
"I found it!" theories before the first observation — which then named
the bug instantly. Bug-report-shaped requests must steer the FIRST
action to reproduce/observe, and a killed thinking loop must redirect
to a grounding tool call instead of resumed speculation.
"""
import inspect

import ghost_agent.core.agent as agent_mod
from ghost_agent.core.agent import _is_bug_report_intent


# ── intent gate ──────────────────────────────────────────────────────
def test_detects_classic_bug_reports():
    assert _is_bug_report_intent(
        'when i click "enter the corridor" nothing happens.')
    assert _is_bug_report_intent("the save button doesn't work")
    assert _is_bug_report_intent("the app crashed after the update")
    assert _is_bug_report_intent("page is blank when I open it")
    assert _is_bug_report_intent("it won't load anymore")
    assert _is_bug_report_intent("the game is stuck on the intro screen")


def test_ignores_non_bug_requests():
    assert not _is_bug_report_intent(
        "I would like YOU to create a project of your own choosing.")
    assert not _is_bug_report_intent("build me a web game about corridors")
    assert not _is_bug_report_intent("what did you learn today?")
    assert not _is_bug_report_intent("summarize this document")
    assert not _is_bug_report_intent(None)
    assert not _is_bug_report_intent("")


# ── wiring into handle_chat ──────────────────────────────────────────
def test_nudge_is_injected_before_first_llm_call():
    src = inspect.getsource(agent_mod.GhostAgent.handle_chat)
    assert "SYSTEM HINT (repro-first)" in src
    assert "_is_bug_report_intent(lc)" in src
    # the nudge must come BEFORE the turn loop's first LLM request —
    # locate it relative to the correction-promotion hook that runs
    # at the top of handle_chat
    assert (src.index("SYSTEM HINT (repro-first)")
            < src.index("user-correction promotion"))


def test_nudge_is_suppressed_in_simulations():
    src = inspect.getsource(agent_mod.GhostAgent.handle_chat)
    nudge_block = src[src.index("Repro-first nudge"):
                      src.index("SYSTEM HINT (repro-first)")]
    assert "suppress_meta_task_nudges" in nudge_block


def test_nudge_text_demands_observation_first():
    src = inspect.getsource(agent_mod.GhostAgent.handle_chat)
    start = src.index("SYSTEM HINT (repro-first)")
    block = src[start:start + 1200]
    assert "reproduce" in block
    assert "FIRST tool call" in block
    assert "file:line" in block


# ── post-abort grounding directive ───────────────────────────────────
def test_thinking_loop_alert_redirects_to_grounding():
    src = inspect.getsource(agent_mod.GhostAgent.handle_chat)
    start = src.index("self-repeating thinking loop")
    alert = src[start:start + 1500]
    assert "grounding tool call" in alert
    assert "hypothesizing" in alert
    # the pre-existing escape hatches must survive the rewrite
    assert "abort_attempt" in alert
