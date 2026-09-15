"""§4GH (2026-09-13, request e57ad0cf): a forced final that produces no
answer, and the narration-only reply as a mechanical refutation.

What happened: the no-progress breaker forced the final at turn 9; the
model emitted three more `web_search` calls (dropped), its own text was
empty, and the loop shipped the accumulated working narration of nine turns
("I have good coverage. Let me now dig into…" × 5) as the reply. The cheap
judge refuted it; the escalation to the main model overturned that to
CONFIRMED 0.85. The user got no answer after 339 s and the record said ok.

Three pins, each naming the pre-fix world:
  * `narration_only` — the shape, measured on the corpus before shipping:
    0 of 115 human-approved and 0 of 601 verifier-passed replies match.
  * the forced final: one retry with a hard answer-now directive, then an
    honest fallback built from the last evidence — driven end to end
    through `handle_chat` with the one-task latch as the forcing mechanism.
  * the verifier: a narration-only reply after tools is REFUTED
    mechanically on the tool-turn path and the judge is never consulted —
    so there is nothing for an escalation to overturn.
"""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ghost_agent.core import reply_shape_check as rsc
from ghost_agent.core import reply_smoothing as rs
from ghost_agent.core.agent import (GhostAgent, _FORCED_FINAL_ANSWER_DIRECTIVE,
                                    _no_answer_fallback_reply)
from ghost_agent.core.verifier import VerifyVerdict
from ghost_agent.tools.outcome import ToolOutcome
from tests.helpers import FakeBgTasks, make_context

# The delivered reply of request e57ad0cf, verbatim.
E57AD0CF_REPLY = (
    "I have a strong initial set of sources. Now let me dig into specifics — the country, "
    "agency, email domain, and ZachXBT's original posts — across multiple channels in parallel.\n\n"
    "I have good coverage. Let me now dig into the specifics — country/agency speculation, the "
    "exact email domain, and read the most detailed sources in parallel.\n\n"
    "I have a solid foundation. Let me now dig into specifics — the exact email domain/sender, "
    "country/agency speculation, the Telegram customer notice, and dark-web forums.\n\n"
    "The dark-web search returned mostly generic results. Let me nail down the exact email "
    "domain/sender, the customer notification details, and country/agency speculation with "
    "targeted searches.\n\n"
    "Let me read the most detailed sources in parallel to extract the exact email domain, "
    "sender address, and full customer notice text."
)


# ── the shape ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expect", [
    (E57AD0CF_REPLY, True),
    ("Let me search more specifically for terminal-related content.", True),
    ("I'll fetch the detailed status of each project.", True),
    ("Now let me fix both:", True),
    # answers — approved-shaped replies from the corpus must never match
    ("You're welcome! Let me know if you need anything. 👊", False),
    ("Sounds like a solid plan! BJJ in a couple of hours, work after. Let me know if you "
     "want me to log the session afterward.", False),
    ("The sender domain was gov.example; Revolut confirmed it on 12 September.\n\n"
     "Let me know if you want the full timeline.", False),
    ("I have good coverage. Let me now dig in.\n\n- Reuters: 12 Sept\n- TechCrunch: same day", False),
    ("Let me read it.\n\nThe page says: \"the request came from a spoofed address\".", False),
    ("Let me check https://example.org first.", False),
    ("I found 3 sources. Let me summarise: the agency was never named publicly.", True),
    # a long informative sentence beside a beat is an answer, not glue
    ("The sender used a domain that mimicked an official ministry address and the request "
     "passed the internal checks because verification relied on the display name alone. "
     "Let me dig further.", False),
    ("", False),
    ("Done.", False),
    # R3 review probes: answers that OPEN like beats but announce no work
    ("I'll be direct: the file does not exist on the server.", False),
    ("Let me be clear: that claim is false.", False),
    ("I need to ask you: which file do you mean, the config or the script?", False),
    ("I will not do that.", False),
    ("Let's go with option B.", False),
    ("I'm going to need the password before I can continue.", False),
    ("I'll take that as a yes. The task is now closed.", False),
    # "let me know" is the user's cue, even when a work verb follows
    ("Let me know when to proceed.", False),
    # a beat followed by a clarification question is an answer
    ("I'll fetch the status. Which project do you mean?", False),
])
def test_narration_only_shape(text, expect):
    assert rs.narration_only(text) is expect, text[:60]


def test_the_smoother_would_have_shipped_the_e57ad0cf_reply_untouched():
    """The fail-open rule the fix routes around: the smoother cannot rescue
    a reply that is narration all the way down — what ships is the original,
    and it still has no answer in it.

    ⚠ THE ASSERTION MOVED ONE LAYER OUT (§4GO, 2026-09-14). It used to be
    byte equality on `smooth_reply`, which held only because pass 1 could
    not see a beat that does not OPEN its paragraph. Pass 1b can, so the
    pure function now trims this reply to its last beat — and
    `is_narration_only_trim` catches that (what survived is narration) and
    the DELIVERED text is the original, which is the property this pin
    exists for. Asserting the pure function again would pin the blind spot
    rather than the behaviour."""
    assert rs.treat_reply(E57AD0CF_REPLY, n_real_tools=5) == E57AD0CF_REPLY
    assert rs.is_narration_only_trim(rs.smooth_reply(E57AD0CF_REPLY),
                                     E57AD0CF_REPLY) is True
    # …and the §4GH machinery still sees no answer in whatever survives
    assert rs.narration_only(rs.smooth_reply(E57AD0CF_REPLY)) is True


@pytest.mark.parametrize("this_turn,accumulated,expect", [
    ("", "", True),
    ("", E57AD0CF_REPLY, True),
    ("Let me search again.", "I have good coverage. Let me dig in.", True),
    ("The agency was never named; the domain was a spoofed .gov.", E57AD0CF_REPLY, False),
    ("", "The answer is 42.", False),
])
def test_forced_final_has_no_answer(this_turn, accumulated, expect):
    assert rs.forced_final_has_no_answer(this_turn, accumulated) is expect


def test_the_dropped_mutation_note_is_not_an_answer():
    """A forced final that dropped a WRITE gets the loop's own "has NOT
    been applied" note appended; the note must not count as the answer."""
    from ghost_agent.core.agent import _dropped_mutation_note, _forced_final_has_no_answer
    note = _dropped_mutation_note(["file_system"])
    assert note
    assert _forced_final_has_no_answer(note, "Let me write the file.") is True
    assert _forced_final_has_no_answer("The file is written; here is what changed." + note, "") is False


# ── the refutation ───────────────────────────────────────────────────────────

def test_refute_narration_only_is_gated_on_tools_and_not_on_image_turns():
    assert rsc.refute_narration_only(E57AD0CF_REPLY, n_real_tools=0) == []
    assert rsc.refute_narration_only(E57AD0CF_REPLY, n_real_tools=3,
                                     tool_names=["web_search", "image_generation"]) == []
    issues = rsc.refute_narration_only(E57AD0CF_REPLY, n_real_tools=3,
                                       tool_names=["web_search", "browser"])
    assert len(issues) == 1 and "narration" in issues[0]
    assert rsc.refute_narration_only("The domain was spoofed.", n_real_tools=3) == []


def test_the_no_answer_fallback_head_is_a_refuted_shape():
    """The fallback is an honest non-answer and the shape check must say
    so (the head lives in FALLBACK_HEADS for exactly this reason)."""
    reply = _no_answer_fallback_reply([
        {"role": "tool", "tool_call_id": "c", "name": "web_search",
         "content": ToolOutcome.ok("### 1. Reuters\nRevolut confirmed…", call_args={"query": "revolut"})}])
    assert reply.startswith(rsc.FALLBACK_HEADS["no_answer"])
    assert "Revolut confirmed" in reply and "web_search" in reply
    assert rsc.refute_no_answer_fallback(reply)     # refuted as a non-answer — its own arm
    assert "No tool this turn returned usable evidence" in _no_answer_fallback_reply([])


def _rows(*names):
    return [{"role": "tool", "tool_call_id": f"c{i}", "name": n,
             "content": ToolOutcome.ok("### 1. result\nbody", call_args={"query": "q"})}
            for i, n in enumerate(names)]


def test_reply_shape_refutation_names_the_narration_arm():
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = SimpleNamespace()
    v = agent._reply_shape_refutation(E57AD0CF_REPLY, "find the domain", _rows("web_search", "browser"))
    assert v is not None and v.verdict == VerifyVerdict.REFUTED
    assert v.reasoning == GhostAgent._NARRATION_ONLY_REASONING
    assert agent._reply_shape_refutation(E57AD0CF_REPLY, "find the domain", []) is None
    assert agent._reply_shape_refutation("The domain was spoofed.", "x", _rows("web_search")) is None


async def test_a_narration_only_tool_turn_is_refuted_without_consulting_the_judge():
    """Pre-fix: the judge ran, the cheap REFUTE was escalated and
    OVERTURNED to CONFIRMED 0.85. Now the judge is never called."""
    ctx = make_context()
    judge_called = []

    async def _verify(*a, **k):
        judge_called.append(1)
        raise AssertionError("the judge must not be consulted on a narration-only reply")
    ctx.verifier = SimpleNamespace(llm_client=object(), verify_claim=_verify,
                                   verify_code_output=_verify)
    agent = GhostAgent(ctx)
    rows = [{"role": "tool", "tool_call_id": "c1", "name": "browser",
             "content": ToolOutcome.ok(
                 "--- BROWSER RESULT ---\nSTATUS: OK\nOP: extract_text\nURL: https://x\n"
                 "TITLE: t\nLENGTH: 3950\n--- TEXT ---\n" + "words " * 300,
                 call_args={"operation": "extract_text", "url": "https://x"})}]
    v, lt = await agent._compute_verifier_verdict(
        tools_run_this_turn=rows, messages=[], final_ai_content=E57AD0CF_REPLY,
        last_user_content="identify the agency", lc="identify the agency",
        req_id="r1", trajectory_id="t1")
    assert judge_called == []
    assert v is not None and v.verdict == VerifyVerdict.REFUTED and v.confidence >= 0.9
    assert "reply-shape" in str(getattr(v, "override", ""))


# ── the forced final, end to end ─────────────────────────────────────────────

DONE_READBACK = json.dumps({"updated": [{"id": "t1", "status": "DONE",
                                         "result_summary": "wrote parser.py"}], "count": 1})
SEARCH_RESULT = ("### 1. Reuters — Revolut confirms breach\nRevolut said the request came "
                 "from a spoofed government domain.\n[Source: https://reuters.example/a]\n")


def _resp(content, tool_calls=None):
    return {"choices": [{"message": {"role": "assistant", "content": content,
                                     "tool_calls": tool_calls or []}}]}


def _tc(cid, name, args):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def _agent(monkeypatch, scripted):
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    monkeypatch.setenv("GHOST_CRITIC_REPAIR_BUDGET", "0")
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    ctx = make_context()
    agent = GhostAgent(ctx)
    agent.available_tools = {
        "web_search": AsyncMock(return_value=SEARCH_RESULT),
        "manage_projects": AsyncMock(return_value=DONE_READBACK),
    }
    ctx.llm_client.chat_completion = AsyncMock(side_effect=scripted)
    return agent, ctx


FIRST = [_tc("c0", "web_search", {"query": "revolut breach"}),
         _tc("c1", "manage_projects", {"action": "update", "id": "t1", "status": "DONE"})]


async def test_a_forced_final_with_no_answer_gets_one_retry_and_the_retry_is_delivered(monkeypatch):
    """Turn 1 closes the task (the latch forces the final). Turn 2 — tools
    off — emits only a dropped search and no text. Pre-fix the reply was
    "Let me close task 1." (turn 1's narration). Now: one retry with the
    hard directive, and its answer ships."""
    agent, ctx = _agent(monkeypatch, [
        _resp("Let me close task 1.", FIRST),
        _resp("", [_tc("c2", "web_search", {"query": "more"})]),     # forced final, no answer
        _resp("Task 1 is done: the spoofed .gov domain was confirmed by Reuters."),
        _resp("(unreachable)"),
    ])
    out, _, _ = await agent.handle_chat(
        {"messages": [{"role": "user", "content": "start task 1"}]}, FakeBgTasks())
    assert "Task 1 is done" in out
    assert not rs.narration_only(out)
    def _msgs(c):
        p = c.kwargs.get("messages")
        if p is None and c.args:
            p = c.args[0].get("messages") if isinstance(c.args[0], dict) else c.args[0]
        return p
    payloads = [_msgs(c) for c in ctx.llm_client.chat_completion.call_args_list]
    # the loop wraps the last user-role message in the volatile state block;
    # the directive rides inside it
    assert _FORCED_FINAL_ANSWER_DIRECTIVE in payloads[2][-1]["content"]
    assert _FORCED_FINAL_ANSWER_DIRECTIVE not in str(payloads[1][-1]["content"])
    assert agent.available_tools["web_search"].await_count == 1        # the dropped search never ran


async def test_two_misses_ship_the_honest_fallback_not_the_narration(monkeypatch):
    agent, ctx = _agent(monkeypatch, [
        _resp("Let me close task 1.", FIRST),
        _resp("", [_tc("c2", "web_search", {"query": "more"})]),
        _resp("Let me search once more.", [_tc("c3", "web_search", {"query": "again"})]),
        _resp("(unreachable)"),
    ])
    out, _, _ = await agent.handle_chat(
        {"messages": [{"role": "user", "content": "start task 1"}]}, FakeBgTasks())
    assert out.startswith(rsc.FALLBACK_HEADS["no_answer"]), out[:200]
    assert "spoofed government domain" in out                # the last evidence, verbatim
    assert "Let me close task 1." not in out                 # the narration is gone
    assert ctx.llm_client.chat_completion.await_count == 3


async def test_a_forced_final_that_answers_is_untouched(monkeypatch):
    """Control: the retry never fires when the forced final carries an answer."""
    agent, ctx = _agent(monkeypatch, [
        _resp("Let me close task 1.", FIRST),
        _resp("Task 1 is done: parser.py written and the domain confirmed."),
        _resp("(unreachable)"),
    ])
    out, _, _ = await agent.handle_chat(
        {"messages": [{"role": "user", "content": "start task 1"}]}, FakeBgTasks())
    assert "Task 1 is done" in out
    assert ctx.llm_client.chat_completion.await_count == 2


async def test_the_directive_only_fires_on_a_FORCED_final(monkeypatch):
    """The directive says "tools are OFF"; on an ordinary final (tools on)
    a narration-only reply ships as it always did and the verifier's
    shape check refutes it — the loop must not lie about the tools."""
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    ctx = make_context()
    agent = GhostAgent(ctx)
    agent.available_tools = {"web_search": AsyncMock(return_value=SEARCH_RESULT)}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("Let me search.", [_tc("c0", "web_search", {"query": "revolut"})]),
        _resp("Let me search more specifically for the sender domain."),   # not forced
        _resp("The sender domain was a spoofed .gov address."),
        _resp("(unreachable)"),
    ])
    out, _, _ = await agent.handle_chat(
        {"messages": [{"role": "user", "content": "find the sender domain"}]}, FakeBgTasks())
    # the loop's OWN narration guards may re-prompt here; what must not
    # happen is the forced-final directive (it says "tools are OFF")
    for c in ctx.llm_client.chat_completion.call_args_list:
        p = c.kwargs.get("messages") or (c.args[0].get("messages") if c.args and isinstance(c.args[0], dict) else c.args[0])
        assert all(_FORCED_FINAL_ANSWER_DIRECTIVE not in str(m.get("content")) for m in p)
    assert not rs.narration_only(out)


async def test_the_fallback_discards_the_narration_even_when_the_smoother_is_off(monkeypatch):
    """With ONE real tool the smoother never runs (SMOOTHING_MIN_TOOLS=2),
    so the only thing removing the accumulated beats is the fallback's own
    discard."""
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    monkeypatch.setenv("GHOST_CRITIC_REPAIR_BUDGET", "0")
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    ctx = make_context()
    agent = GhostAgent(ctx)
    agent.available_tools = {"manage_projects": AsyncMock(return_value=DONE_READBACK)}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("Let me close task 1.", [_tc("c1", "manage_projects", {"action": "update", "id": "t1", "status": "DONE"})]),
        _resp(""),
        _resp("Let me try once more."),
        _resp("(unreachable)"),
    ])
    out, _, _ = await agent.handle_chat(
        {"messages": [{"role": "user", "content": "start task 1"}]}, FakeBgTasks())
    assert out.startswith(rsc.FALLBACK_HEADS["no_answer"]), out[:200]
    assert "Let me close task 1." not in out and "Let me try once more." not in out


# ── R3 review of §4GH: the fallback through the verifier, the dropped-write
#    note, and the last budget turn ─────────────────────────────────────────

def test_the_fallback_is_its_own_arm_not_a_raw_dump():
    reply = _no_answer_fallback_reply(_rows("web_search"))
    assert rsc.refute_no_answer_fallback(reply)
    assert rsc.refute_raw_tool_dump(reply) == []          # not the repairable arm
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = SimpleNamespace()
    v = agent._reply_shape_refutation(reply, "x", _rows("web_search"))
    assert v.reasoning == GhostAgent._NO_ANSWER_REASONING
    assert GhostAgent._verdict_is_no_claim(v) is True


async def test_the_fallback_is_refuted_without_the_judge_and_is_not_repaired(monkeypatch):
    """Sync-critic mode (the live default), verifier ATTACHED: the second
    miss ships the fallback, the verdict is the mechanical no-claim refute,
    the judge is never called, and the in-loop repair does NOT fire on it
    (a third LLM call would have discarded the evidence the fallback
    carries and asked for 'the same answer in a better form')."""
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "0")
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    ctx = make_context()

    async def _verify(*a, **k):
        raise AssertionError("the judge must not be consulted on the fallback")
    ctx.verifier = SimpleNamespace(llm_client=object(), verify_claim=_verify,
                                   verify_code_output=_verify)
    agent = GhostAgent(ctx)
    agent.available_tools = {
        "web_search": AsyncMock(return_value=SEARCH_RESULT),
        "manage_projects": AsyncMock(return_value=DONE_READBACK),
    }
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("Let me close task 1.", FIRST),
        _resp("", [_tc("c2", "web_search", {"query": "more"})]),
        _resp("Let me search once more.", [_tc("c3", "web_search", {"query": "again"})]),
        _resp("(unreachable — a repair would land here)"),
    ])
    out, _, _ = await agent.handle_chat(
        {"messages": [{"role": "user", "content": "start task 1"}]}, FakeBgTasks())
    assert out.startswith(rsc.FALLBACK_HEADS["no_answer"]), out[:200]
    assert "spoofed government domain" in out
    assert ctx.llm_client.chat_completion.await_count == 3


async def test_a_write_dropped_on_a_forced_final_miss_is_still_declared_not_applied(monkeypatch):
    """Honesty note (2026-07-14) on the new path: the write the model kept
    trying to make never ran — both when the retry answers and when the
    fallback ships."""
    from ghost_agent.core.agent import _DROPPED_NOTE_HEAD
    for third, expect_head in (
            (_resp("Task 1 is done: the config now points at the new host."), "Task 1 is done"),
            (_resp("", [_tc("c3", "file_system", {"operation": "replace", "path": "app.cfg",
                                                  "old_text": "a", "replace_with": "b"})]),
             rsc.FALLBACK_HEADS["no_answer"])):
        monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
        monkeypatch.setenv("GHOST_CRITIC_REPAIR_BUDGET", "0")
        monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
        ctx = make_context()
        agent = GhostAgent(ctx)
        agent.available_tools = {
            "file_system": AsyncMock(return_value="SUCCESS: Wrote 3 chars to 'app.cfg'."),
            "manage_projects": AsyncMock(return_value=DONE_READBACK),
        }
        ctx.llm_client.chat_completion = AsyncMock(side_effect=[
            _resp("Let me close task 1.", [_tc("c1", "manage_projects",
                                               {"action": "update", "id": "t1", "status": "DONE"})]),
            _resp("", [_tc("c2", "file_system", {"operation": "replace", "path": "app.cfg",
                                                 "old_text": "a", "replace_with": "b"})]),
            third,
            _resp("(unreachable)"),
        ])
        out, _, _ = await agent.handle_chat(
            {"messages": [{"role": "user", "content": "start task 1"}]}, FakeBgTasks())
        assert expect_head in out, out[:200]
        assert out.count(_DROPPED_NOTE_HEAD) == 1, out
        assert agent.available_tools["file_system"].await_count == 0


async def test_no_retry_on_the_last_budget_turn_the_fallback_ships_directly(monkeypatch):
    """A `continue` on the last turn exits to the exhaustion path, which
    ships the narration this branch replaces — so the last turn goes
    straight to the fallback."""
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    monkeypatch.setenv("GHOST_CRITIC_REPAIR_BUDGET", "0")
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    ctx = make_context()
    agent = GhostAgent(ctx)
    agent.max_turns_override = 2
    agent.available_tools = {
        "web_search": AsyncMock(return_value=SEARCH_RESULT),
        "manage_projects": AsyncMock(return_value=DONE_READBACK),
    }
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("Let me close task 1.", FIRST),
        _resp("", [_tc("c2", "web_search", {"query": "more"})]),
        _resp("(unreachable)"),
    ])
    out, _, _ = await agent.handle_chat(
        {"messages": [{"role": "user", "content": "start task 1"}]}, FakeBgTasks())
    assert ctx.llm_client.chat_completion.await_count == 2
    assert out.startswith(rsc.FALLBACK_HEADS["no_answer"]), out[:200]
    assert "Let me close task 1." not in out
