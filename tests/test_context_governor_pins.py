"""§ context slice R1 (2026-08-19, lens C sweep) — Driven behavioral pins for the 8 SURVIVED mutants of the CX1C sweep.
Each pin: exact input -> observable. Must PASS on pristine, FAIL on its mutant.
Lives OUTSIDE tests/ on purpose (auditor sandbox only, never shipped)."""
import asyncio
import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from ghost_agent.core.agent import GhostAgent, GhostContext, TurnState
from ghost_agent.core.strikes import StrikeLedger


def _tok_patch(monkeypatch):
    import ghost_agent.core.agent as agent_mod
    monkeypatch.setattr(agent_mod, "estimate_tokens", lambda t: len(str(t)) // 4)


def _prune_agent():
    ag = GhostAgent.__new__(GhostAgent)
    ag.context = MagicMock()
    ag.context.memory_system = None
    ag.context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "SUMMARY"}}]})
    return ag


# ── P6: recent-window must not START on an orphaned tool result ──────
@pytest.mark.asyncio
async def test_pin_P6_recent_window_pulls_in_tool_caller(monkeypatch):
    _tok_patch(monkeypatch)
    ag = _prune_agent()
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "goal " + "g" * 1200},
        {"role": "assistant", "content": "A1CALLER plain text " + "a" * 1200},
        {"role": "tool", "name": "fs", "content": "ok short"},
        {"role": "tool", "name": "fs", "content": "T2 " + "t" * 1200},
        {"role": "assistant", "content": "a2 " + "b" * 1200},
        {"role": "tool", "name": "fs", "content": "T3 " + "c" * 1200},
        {"role": "assistant", "content": "a3 " + "d" * 1200},
        {"role": "tool", "name": "fs", "content": "T4 " + "e" * 1200},
        {"role": "assistant", "content": "a4 " + "f" * 1200},
    ]
    out = await ag._prune_context(msgs, max_tokens=500, model="m")
    # pristine: pull-in walks recent_start 3->2->1, middle empty, caller kept verbatim
    assert any("A1CALLER" in str(m.get("content", "")) for m in out), \
        "orphaned-tool pull-in dead: caller was summarized away"


# ── P8: anchored findings must survive the prune ─────────────────────
@pytest.mark.asyncio
async def test_pin_P8_anchored_finding_survives(monkeypatch):
    _tok_patch(monkeypatch)
    ag = _prune_agent()
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "goal " + "g" * 1200},
        {"role": "assistant", "content": "root cause FINDMARKER frobnicator broken " + "a" * 400},
        {"role": "tool", "name": "fs", "content": "tx " + "t" * 1200},
        {"role": "assistant", "content": "r1 " + "b" * 1200},
        {"role": "assistant", "content": "r2 " + "c" * 1200},
        {"role": "assistant", "content": "r3 " + "d" * 1200},
        {"role": "assistant", "content": "r4 " + "e" * 1200},
        {"role": "assistant", "content": "r5 " + "f" * 1200},
        {"role": "assistant", "content": "r6 " + "h" * 1200},
    ]
    out = await ag._prune_context(msgs, max_tokens=500, model="m")
    marked = [m for m in out if "FINDMARKER" in str(m.get("content", ""))]
    assert marked and any("[ANCHORED]" in str(m.get("content", "")) for m in marked), \
        "anchoring dead: middle finding vanished from pruned context"


# ── C1: pinned stable block must be the PREFIX of the first user msg ─
def test_pin_C1_stable_block_is_prefix():
    msgs = [{"role": "user", "content": "the goal"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "latest"}]
    out = GhostAgent._compose_injection(msgs, "STABLEBLOCK", "DYN", pin=True)
    assert out[0]["content"].startswith("<session_context>"), \
        "stable block is not the byte-prefix -> KV cache can never reuse it"
    assert "STABLEBLOCK" in out[0]["content"].split("[USER INSTRUCTION]")[0]


# ── handle_chat harness (adapted from tests/test_bench_handle_chat_1c.py) ──
class _FakeBgTasks:
    def add_task(self, *a, **k):
        pass


def _load_bench_module():
    import importlib.util
    here = Path(__file__).resolve().parents[1] / "tests" / "test_bench_handle_chat_1c.py"
    spec = importlib.util.spec_from_file_location("bench_1c_for_pins", here)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ctx(tmp_path, max_context=8000, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.delenv("GHOST_EXPERIMENTS", raising=False)
    from ghost_agent.core import experiments as ex
    ex.reset_registry_cache()
    mod = _load_bench_module()
    context, _root = mod._bench_context(tmp_path)
    context.args.max_context = max_context
    return context



def _sse_stream_factory(responses):
    """Returns (fake_stream_fn, captured_payloads). Each call yields the next
    response text as SSE frames; repeats the last response when exhausted."""
    captured = []
    state = {"i": 0}

    def fake_stream(payload, **kw):
        captured.append(payload)
        idx = min(state["i"], len(responses) - 1)
        state["i"] += 1
        text = responses[idx]

        async def _agen():
            frame = {"choices": [{"delta": {"content": text},
                                  "finish_reason": None}]}
            yield ("data: " + json.dumps(frame) + "\n\n").encode()
            done = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
            yield ("data: " + json.dumps(done) + "\n\n").encode()
            yield b"data: [DONE]\n\n"
        return _agen()

    return fake_stream, captured


def _wire_stream(context, responses):
    fake, captured = _sse_stream_factory(responses)
    context.llm_client.stream_chat_completion = fake
    return captured


def _all_llm_texts(mock):
    texts = []
    for call in mock.call_args_list:
        payload = call.args[0] if call.args else call.kwargs.get("payload")
        if isinstance(payload, dict):
            for m in payload.get("messages", []):
                texts.append(str(m.get("content", "")))
    return "\n".join(texts)


# ── S1: first overflow must emit the context-pressure steer ──────────
@pytest.mark.asyncio
async def test_pin_S1_first_overflow_emits_steer(tmp_path, monkeypatch):
    _tok_patch(monkeypatch)
    context = _ctx(tmp_path, monkeypatch=monkeypatch)
    agent = GhostAgent(context)
    agent.thinking_budget_override = "selfplay"
    # Pressure must build DURING the request: the pre-loop rolling window
    # drops an oversized goal outright. A huge assistant preamble carrying a
    # tool call survives the L1-L3 ladder verbatim (<tool_call> exemption)
    # and pushes occupancy past the history budget for turn 2.
    huge_preamble = "analysis " + "y" * 40000 + "\n"
    tool_xml = '<tool_call>{"name": "nosuch_tool", "arguments": {}}</tool_call>'
    captured = _wire_stream(context, [huge_preamble + tool_xml, "FINAL: done."])
    body = {"messages": [{"role": "user", "content": "small goal"}]}
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[]):
        await agent.handle_chat(body, _FakeBgTasks(), request_id="pin-s1")
    texts = "\n".join(str(m.get("content", ""))
                      for pl in captured for m in pl.get("messages", []))
    assert "SYSTEM ALERT (context pressure)" in texts, \
        "steer counter dead: overflow produced no externalize-notes steer"


# ── S3: request start must CLEAR a stale lockdown (disarm path) ──────
@pytest.mark.asyncio
async def test_pin_S3_lockdown_cleared_at_request_start(tmp_path, monkeypatch):
    context = _ctx(tmp_path, monkeypatch=monkeypatch)
    context._ctx_pressure_lockdown = True          # stale from a previous request
    agent = GhostAgent(context)
    agent.thinking_budget_override = "selfplay"
    _wire_stream(context, ["FINAL: done."])
    body = {"messages": [{"role": "user", "content": "small ask"}]}
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[]):
        await agent.handle_chat(body, _FakeBgTasks(), request_id="pin-s3")
    assert context._ctx_pressure_lockdown is False, \
        "arm-without-disarm: lockdown survived into a fresh request"


# ── S4: per-batch read budget must be the window-derived constant ────
@pytest.mark.asyncio
async def test_pin_S4_read_budget_cap_value(tmp_path, monkeypatch):
    context = _ctx(tmp_path, max_context=240000, monkeypatch=monkeypatch)
    agent = GhostAgent(context)
    agent.thinking_budget_override = "selfplay"
    _wire_stream(context, [
        '<tool_call>{"name": "nosuch_tool", "arguments": {}}</tool_call>',
        "FINAL: done."])
    body = {"messages": [{"role": "user", "content": "do a thing"}]}
    # § context R1 B1: the budget is DISARMED at request end (a stale batch
    # budget used to govern every out-of-band tool call overnight), so the
    # armed limit must be captured DURING the run via a constructor spy.
    import ghost_agent.tools.file_system as _fsmod
    _armed = []
    _RealRB = _fsmod.ReadBudget

    class _SpyRB(_RealRB):
        def __init__(self, limit):
            _armed.append(limit)
            super().__init__(limit)

    monkeypatch.setattr(_fsmod, "ReadBudget", _SpyRB)
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[]):
        await agent.handle_chat(body, _FakeBgTasks(), request_id="pin-s4")
    assert 336000 in _armed, \
        f"read budget cap wrong: armed limits {_armed!r} lack 336000 (0.40 * window)"
    assert context._read_budget is None, \
        "request-end disarm missing: a stale batch budget would govern out-of-band callers"


# ── S5: lockdown must zero the batch read budget ─────────────────────
@pytest.mark.asyncio
async def test_pin_S5_lockdown_zeroes_batch_budget(tmp_path, monkeypatch):
    context = _ctx(tmp_path, monkeypatch=monkeypatch)
    context._ctx_pressure_lockdown = True
    agent = GhostAgent(context)
    ts = TurnState(
        _constraint_steer_pending=None, _proj_task_closed_this_req=False,
        _request_sys3_fired_once=False, _request_sys3_prev_justification="",
        consecutive_parse_errors=0, current_plan_json="",
        execution_failure_count=0, final_ai_content="", fname="",
        force_final_response=False, force_stop=False, forget_was_called=False,
        last_was_failure=False, preflight_blocks_this_request=0,
        request_sandbox_state=None, transient_failure_count=0,
        tool_calls=[{"id": "c1", "type": "function",
                     "function": {"name": "nosuch_tool", "arguments": "{}"}}],
        msg={"role": "assistant", "content": "", "tool_calls": []},
        ui_content="", parse_failure_reason="", model="test-model",
        last_user_content="hi", char_budget=100000,
        strikes=StrikeLedger(), task_tree=None, _user_batch_intent=None,
        _request_constraints=[], repeated_action_steered=set(),
        messages=[{"role": "user", "content": "hi"}], seen_tools=set(),
        executed_idempotent=set(), raw_tools_called=[], tool_usage={},
        tools_run_this_turn=[], request_state={}, req_id="pin-s5",
    )
    with patch("ghost_agent.core.agent.pretty_log"):
        try:
            await agent._dispatch_and_process_tool_batch(ts)
        except Exception:
            pass  # budget is constructed before the parts that may object to mocks
    rb = context._read_budget
    assert rb is not None and rb.limit == 0, \
        f"lockdown did not zero the batch read budget (limit={getattr(rb, 'limit', None)!r})"


# ── X1: the compression ladder must engage below 95% occupancy ───────
def test_pin_X1_ladder_engages_at_80pct():
    from ghost_agent.core.context_manager import ContextManager
    cm = ContextManager(max_tokens=1000)   # default len//4 estimator
    big_tool = "\n".join(f"line {i} " + "w" * 70 for i in range(40))  # ~3.1k chars
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "goal"},
        {"role": "tool", "name": "fs", "content": big_tool},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
    ]
    out = cm.compress_if_needed([dict(m) for m in msgs], max_level=3)
    assert cm.compression_level >= 1, \
        f"ladder never engaged at ~80% occupancy (level={cm.compression_level})"
    tool_out = next(m["content"] for m in out if m.get("role") == "tool")
    assert "compressed]" in tool_out and len(tool_out) < len(big_tool), \
        "old tool output was not compressed at ~80% occupancy"


# ── S2b: SECOND overflow must actually ARM the lockdown (behavioral;
#         the only existing kill is a getsource token pin) ─────────────
@pytest.mark.asyncio
async def test_pin_S2b_second_overflow_arms_lockdown(tmp_path, monkeypatch):
    _tok_patch(monkeypatch)
    context = _ctx(tmp_path, monkeypatch=monkeypatch)
    agent = GhostAgent(context)
    agent.thinking_budget_override = "selfplay"
    huge = "analysis " + "y" * 40000 + "\n"
    tool_xml = '<tool_call>{"name": "nosuch_tool", "arguments": {}}</tool_call>'
    captured = _wire_stream(context, [huge + tool_xml, huge + tool_xml,
                                      "FINAL: done."])
    body = {"messages": [{"role": "user", "content": "small goal"}]}
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[]):
        await agent.handle_chat(body, _FakeBgTasks(), request_id="pin-s2b")
    texts = "\n".join(str(m.get("content", ""))
                      for pl in captured for m in pl.get("messages", []))
    assert "SECOND overflow" in texts, "second steer never emitted"
    assert context._ctx_pressure_lockdown is True, \
        "second overflow did not arm the whole-file-read lockdown"


# ── S6b: budget-construction failure under lockdown must fail CLOSED
#         (behavioral; the only existing kill is a file-text token pin) ─
@pytest.mark.asyncio
async def test_pin_S6b_budget_failure_fails_closed_under_lockdown(
        tmp_path, monkeypatch):
    import ghost_agent.tools.file_system as fs_mod
    context = _ctx(tmp_path, monkeypatch=monkeypatch)
    context._ctx_pressure_lockdown = True
    agent = GhostAgent(context)

    def _boom(_mc):
        raise RuntimeError("injected: budget construction failed")
    monkeypatch.setattr(fs_mod, "read_byte_budget", _boom)
    ts = TurnState(
        _constraint_steer_pending=None, _proj_task_closed_this_req=False,
        _request_sys3_fired_once=False, _request_sys3_prev_justification="",
        consecutive_parse_errors=0, current_plan_json="",
        execution_failure_count=0, final_ai_content="", fname="",
        force_final_response=False, force_stop=False, forget_was_called=False,
        last_was_failure=False, preflight_blocks_this_request=0,
        request_sandbox_state=None, transient_failure_count=0,
        tool_calls=[{"id": "c1", "type": "function",
                     "function": {"name": "nosuch_tool", "arguments": "{}"}}],
        msg={"role": "assistant", "content": "", "tool_calls": []},
        ui_content="", parse_failure_reason="", model="test-model",
        last_user_content="hi", char_budget=100000,
        strikes=StrikeLedger(), task_tree=None, _user_batch_intent=None,
        _request_constraints=[], repeated_action_steered=set(),
        messages=[{"role": "user", "content": "hi"}], seen_tools=set(),
        executed_idempotent=set(), raw_tools_called=[], tool_usage={},
        tools_run_this_turn=[], request_state={}, req_id="pin-s6b",
    )
    with patch("ghost_agent.core.agent.pretty_log"):
        try:
            await agent._dispatch_and_process_tool_batch(ts)
        except Exception:
            pass
    rb = context._read_budget
    assert rb is not None and getattr(rb, "limit", None) == 0, \
        f"fail-OPEN under lockdown: read budget is {rb!r}, expected a zero budget"
