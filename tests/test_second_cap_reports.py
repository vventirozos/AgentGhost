"""§4IH — a thinking loop is a reason to stop deriving, not to discard the
evidence.

THE LIVE FAILURE (probe ifs20294…, 2026-09-17, fifth IFS/Oxford re-test).
The model finally took the numpy route, ran one `web_search`, and at turn
32 saved `/workspace/oxford_grid.png` (61 KB). Then it argued with itself
about a point-count formula it had invented "known values" for, tripped
the n-gram guard at turn 29 and again at turn 34, and the second cap
shipped the bare `[ATTEMPT_ABORTED_THINKING_LOOP]` marker — 33 tool calls
and a produced plot discarded, six turns still in the budget.

Now the second cap CLOSES THE LOOP as a breaker: with a turn left, tools
off + `blocker_report_alert("thinking loop")` and the §4IF report payload
(thinking off), so the next generation is a report of the evidence in the
context; a third loop lands in the §4IF forced-final branch and ships the
evidence fallback with the marker as a trailer. On the last turn the
evidence fallback ships at once.

World where each pin fails: the second cap aborts with the bare marker
again, the report turn is not marked breaker-forced (so the §4IF payload
switch and gate guard do not apply), the last-turn arithmetic is off, or
the report's own answer is replaced by a marker.
"""
import ast
import inspect

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ghost_agent.core import agent as ag
from ghost_agent.core.agent import (FORCED_FINAL_LOOP_MARKER, GhostAgent, GhostContext,
                                    second_cap_reports)

_TREE = ast.parse(inspect.getsource(ag))


@pytest.mark.parametrize("turn,max_turns,expected", [
    (33, 40, True),      # the live case: turn 34 of 40 — six turns left
    (38, 40, True),      # turn 39: turn 40 exists for the report
    (39, 40, False),     # turn 40: nothing left
    (0, 1, False), (0, 2, True), ("x", 40, False), (3, None, False),
])
def test_second_cap_reports(turn, max_turns, expected):
    assert second_cap_reports(turn, max_turns) is expected


def test_second_cap_branch_closes_the_loop_as_a_breaker():
    """AST: the `thinking_cap_events >= 2` body branches on
    `second_cap_reports(turn, effective_max_turns)`; the report branch sets
    force_final_response, raises the breaker flag, appends
    `blocker_report_alert("thinking loop", …)` and continues; the
    fallthrough ships `forced_final_loop_fallback(...)`, stops and breaks.
    The bare-marker literal is gone from this branch."""
    caps = [n for n in ast.walk(_TREE) if isinstance(n, ast.If)
            and ast.unparse(n.test) == "thinking_cap_events >= 2"]
    assert len(caps) == 1
    cap = caps[0]
    inner = [s for s in cap.body if isinstance(s, ast.If)
             and "second_cap_reports" in ast.unparse(s.test)]
    assert len(inner) == 1
    rep = inner[0]
    assert "effective_max_turns" in ast.unparse(rep.test) and "turn" in ast.unparse(rep.test)
    body_src = "\n".join(ast.unparse(s) for s in rep.body)
    assert "force_final_response = True" in body_src
    assert "self.context._breaker_forced_final = True" in body_src
    alerts = [c for s in rep.body for c in ast.walk(s) if isinstance(c, ast.Call)
              and getattr(c.func, "id", "") == "blocker_report_alert"]
    assert len(alerts) == 1 and alerts[0].args[0].value == "thinking loop"
    assert any(isinstance(s, ast.Return) and s.value.value == "continue" for s in rep.body)
    # the fallthrough (after the inner If, still inside the cap body)
    tail = cap.body[cap.body.index(rep) + 1:]
    tail_src = "\n".join(ast.unparse(s) for s in tail)
    assert "forced_final_loop_fallback(" in tail_src and "_no_answer_fallback_reply(" in tail_src
    assert "force_stop = True" in tail_src
    assert any(isinstance(s, ast.Return) and s.value.value == "break" for s in tail)
    consts = [c.value for s in cap.body for c in ast.walk(s)
              if isinstance(c, ast.Constant) and isinstance(c.value, str)]
    assert not any("[ATTEMPT_ABORTED_THINKING_LOOP]" in c for c in consts), \
        "the bare marker literal is back in the second-cap branch"


# ── behavioural: through handle_chat, the way the escalation harness does ──

@pytest.fixture
def agent():
    context = MagicMock(spec=GhostContext)
    context.llm_client = MagicMock()
    context.llm_client.vision_clients = None
    context.sandbox_dir = "/tmp/sandbox"
    context.args = MagicMock()
    context.args.shell = "bash"
    context.args.max_context = 8000
    context.args.temperature = 0.5
    context.args.smart_memory = 0.0
    context.args.use_planning = False
    context.args.model = "qwen3.6"
    context.args.perfect_it = False
    context.args.native_tools = False
    context.profile_memory = MagicMock()
    context.profile_memory.get_context_string.return_value = ""
    context.memory_system = None
    context.skill_memory = None
    context.scratchpad = MagicMock()
    context.scratchpad.list_all.return_value = ""
    a = GhostAgent(context)
    a.max_thinking_chars_override = 500
    return a


class FakeBgTasks:
    def add_task(self, *a, **k):
        pass


RUNAWAY = ("The validator expects 0 lines but split returns "
           "a list with one empty string. ") * 20
XML = ('<tool_call>\n<function name="noop">\n'
       '<parameter name="x">1</parameter>\n</function>\n</tool_call>')
REPORT = ("## Report\n\nYou asked for the grid points over Oxford. I saved "
          "/workspace/oxford_grid.png (61 KB) at turn 32; the point-count formula "
          "is UNTESTED against a reference. What stopped me: my own derivation looped.")


@pytest.mark.asyncio
async def test_two_caps_then_a_no_think_report_turn_ships_the_report(agent):
    payloads = []

    async def capture(payload, **kwargs):
        payloads.append(payload)
        if len(payloads) <= 2:
            return {"choices": [{"message": {"content": f"<think>{RUNAWAY}</think>\n{XML}",
                                             "tool_calls": []}}]}
        return {"choices": [{"message": {"content": REPORT, "tool_calls": []}}]}

    agent.context.llm_client.chat_completion = AsyncMock(side_effect=capture)
    agent.available_tools = {"noop": AsyncMock(return_value="ok")}
    body = {"messages": [{"role": "user", "content": "second-cap report test"}]}
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[{"function": {"name": "noop"}}]):
        final, _, _ = await agent.handle_chat(body, FakeBgTasks())

    assert len(payloads) == 3, [str(p.get("messages", [])[-1].get("content", ""))[:60] for p in payloads]
    third = payloads[2]
    # the report turn: thinking off by both switches, the alert as the last user message
    assert third.get("chat_template_kwargs", {}).get("enable_thinking") is False
    last_user = third["messages"][-1]
    assert last_user["role"] == "user"
    assert "SYSTEM ALERT (thinking loop)" in last_user["content"]
    assert last_user["content"].rstrip().endswith("/no_think")
    # the model's report is the answer — no marker
    assert "## Report" in final
    assert FORCED_FINAL_LOOP_MARKER not in final and "ATTEMPT_ABORTED" not in final
    assert agent.context._breaker_forced_final is True


@pytest.mark.asyncio
async def test_two_caps_then_a_report_turn_that_goes_back_to_work_ships_the_honest_fallback(agent):
    """The realistic bad case: the report turn (thinking OFF — the fake
    honours the flag, as the model does) ignores "tools off" and emits
    narration + a tool call, twice. §4IG's dropped-call rule makes each a
    NO ANSWER: one retry directive, then the §4GH honest fallback — never
    a bare abort marker."""
    calls = {"n": 0}

    async def capture(payload, **kwargs):
        calls["n"] += 1
        if payload.get("chat_template_kwargs", {}).get("enable_thinking") is False:
            return {"choices": [{"message": {"content": f"Let me fix that.\n{XML}",
                                             "tool_calls": []}}]}
        return {"choices": [{"message": {"content": f"<think>{RUNAWAY}</think>\n{XML}",
                                         "tool_calls": []}}]}

    agent.context.llm_client.chat_completion = AsyncMock(side_effect=capture)
    agent.available_tools = {"noop": AsyncMock(return_value="ok")}
    body = {"messages": [{"role": "user", "content": "second-cap loop test"}]}
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[{"function": {"name": "noop"}}]):
        final, _, _ = await agent.handle_chat(body, FakeBgTasks())
    assert calls["n"] == 4                      # cap, cap → report turn → retry → fallback
    assert "ATTEMPT_ABORTED" not in final
    from ghost_agent.core.reply_shape_check import FALLBACK_HEADS
    assert FALLBACK_HEADS["no_answer"][:40] in final
    assert "You asked: second-cap loop test" in final        # §4II: the fallback carries the ask


@pytest.mark.asyncio
async def test_second_cap_on_the_last_turn_ships_the_fallback_at_once(agent):
    agent.max_turns_override = 2
    calls = {"n": 0}

    async def capture(payload, **kwargs):
        calls["n"] += 1
        return {"choices": [{"message": {"content": f"<think>{RUNAWAY}</think>\n{XML}",
                                         "tool_calls": []}}]}

    agent.context.llm_client.chat_completion = AsyncMock(side_effect=capture)
    agent.available_tools = {"noop": AsyncMock(return_value="ok")}
    body = {"messages": [{"role": "user", "content": "second-cap last-turn test"}]}
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[{"function": {"name": "noop"}}]):
        final, _, _ = await agent.handle_chat(body, FakeBgTasks())
    assert calls["n"] == 2
    assert FORCED_FINAL_LOOP_MARKER in final and final.index(FORCED_FINAL_LOOP_MARKER) > 0
    assert "You asked: second-cap last-turn test" in final    # §4II: the digest carries the ask


def test_every_fallback_site_passes_the_ask():
    """AST: each `_no_answer_fallback_reply(...)` call in the module carries
    an `ask=` keyword — the digest's first line is the request, and a site
    that drops it ships a digest with no subject."""
    calls = [c for c in ast.walk(_TREE) if isinstance(c, ast.Call)
             and getattr(c.func, "id", "") == "_no_answer_fallback_reply"]
    assert len(calls) >= 4
    for c in calls:
        assert any(k.arg == "ask" for k in c.keywords), ast.unparse(c)
