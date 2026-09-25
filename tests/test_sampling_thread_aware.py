"""The sampling profile sees the thread, not just the query — 2026-09-24.

"Seriously ?" after two image turns (slack-3e…) was classified as chit-chat
by the query-only rule and sampled at temperature 1.0 / presence 1.5 — then
made an image_generation call. `get_sampling_params` is unchanged; the
call site now ORs the query classification with the thread's tool activity.
"""
import json
from unittest.mock import AsyncMock

import pytest

from ghost_agent.core.agent import (GhostAgent, _history_shows_tool_activity,
                                    get_sampling_params, GENERAL_SAMPLING_PARAMS,
                                    CODING_SAMPLING_PARAMS)
from tests.helpers import FakeBgTasks, make_context


def _u(c): return {"role": "user", "content": c}
def _a(c, tcs=None):
    m = {"role": "assistant", "content": c}
    if tcs:
        m["tool_calls"] = tcs
    return m


@pytest.mark.parametrize("msgs,expect", [
    ([_u("hi"), _a("hello"), _u("Seriously ?")], False),
    ([_u("make an image"), _a("![generated image](/api/download/gen_1.png)"), _u("Seriously ?")], True),
    ([_u("run it"), _a("", [{"id": "c", "function": {"name": "execute", "arguments": "{}"}}]),
      {"role": "tool", "content": "EXIT CODE: 0"}, _u("and?")], True),
    ([_u("run it"), _a("done\nEXIT CODE: 0"), _u("and?")], True),
    ([_u("x"), _a("<tool_call>{}</tool_call>"), _u("y")], True),
    # multimodal history rows carry text blocks (R2 review)
    ([_u("make an image"), {"role": "assistant", "content": [{"type": "text", "text": "![generated image](/api/download/gen_1.png)"}]}, _u("Seriously ?")], True),
    # the pending user message itself never counts
    ([_u("hi"), _a("hello"), _u("![generated image](/api/download/x.png)")], False),
    # beyond the look-back window the signal is gone
    ([_u("make an image"), _a("![generated image](/api/download/gen_1.png)")] + [_u("a"), _a("b")] * 7 + [_u("z")], False),
    ([], False), (None, False), ([{"role": "assistant"}, "junk"], False),
])
def test_tool_activity_table(msgs, expect):
    assert _history_shows_tool_activity(msgs) is expect


def test_the_profile_function_itself_is_unchanged():
    assert get_sampling_params(False, query="Seriously ?") == GENERAL_SAMPLING_PARAMS
    assert get_sampling_params(True, query="Seriously ?") == CODING_SAMPLING_PARAMS


def _resp(content, tool_calls=None):
    return {"choices": [{"message": {"role": "assistant", "content": content,
                                     "tool_calls": tool_calls or []}}]}


async def _temperature_of_first_call(monkeypatch, history):
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    ctx = make_context()
    agent = GhostAgent(ctx)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[_resp("Right, sorry."), _resp("(unreachable)")])
    await agent.handle_chat({"messages": history}, FakeBgTasks())
    first = ctx.llm_client.chat_completion.call_args_list[0]
    payload = first.args[0] if first.args and isinstance(first.args[0], dict) else first.kwargs
    return payload.get("temperature")


async def test_end_to_end_a_follow_up_in_an_image_thread_is_not_sampled_as_chit_chat(monkeypatch):
    hot = await _temperature_of_first_call(monkeypatch, [
        _u("Create an image of a cat"),
        _a("![generated image](/api/download/gen_1.png)\n\nA grey cat."),
        _u("Seriously ?")])
    cold = await _temperature_of_first_call(monkeypatch, [
        _u("how are you"), _a("Fine, thanks."), _u("Seriously ?")])
    assert hot == CODING_SAMPLING_PARAMS["temperature"]
    assert cold == GENERAL_SAMPLING_PARAMS["temperature"]



async def test_slack_shaped_history_with_no_tool_trace_is_still_a_tool_thread(monkeypatch):
    """R4 review: a client may strip every tool trace from the history it
    resends (the Slack bot removes the image lines), so the history looks like
    chat. The agent's own per-conversation record decides."""
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    ctx = make_context()
    agent = GhostAgent(ctx)
    agent.available_tools = {"image_generation": AsyncMock(return_value=(
        "SUCCESS: Image generated.\n\n![generated image](/api/download/gen_1.png)"))}
    first = [_u("Create an image of a cat")]
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("", [{"id": "c0", "type": "function",
                    "function": {"name": "image_generation", "arguments": json.dumps({"prompt": "a cat"})}}]),
        _resp("Here it is."), _resp("(unreachable)")])
    await agent.handle_chat({"messages": first}, FakeBgTasks())
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[_resp("Sorry."), _resp("(unreachable)")])
    await agent.handle_chat({"messages": first + [_a("Here it is."), _u("Seriously ?")]}, FakeBgTasks())
    call = ctx.llm_client.chat_completion.call_args_list[0]
    payload = call.args[0] if call.args and isinstance(call.args[0], dict) else call.kwargs
    assert payload.get("temperature") == CODING_SAMPLING_PARAMS["temperature"]



@pytest.mark.parametrize("gap,expect", [(11, True), (12, False)])
def test_the_look_back_window_edge(gap, expect):
    """The tool row sits `gap` rows before the pending message; the window is
    12 history rows (the tool-bearing assistant row is row gap+1 from the end)."""
    msgs = [_a("![generated image](/api/download/gen_1.png)")] + [_u("x")] * gap + [_u("pending")]
    assert _history_shows_tool_activity(msgs) is expect



def test_only_a_turn_that_ran_real_tools_is_recorded_and_the_record_expires(monkeypatch):
    import time as _t
    agent = GhostAgent(make_context())
    msgs = [_u("hello")]
    agent._note_tool_conversation(msgs, [])
    agent._note_tool_conversation(msgs, [{"name": "x", "_synthetic": True}])
    assert agent._conversation_ran_tools_recently(msgs) is False
    agent._note_tool_conversation(msgs, [{"name": "web_search", "content": "r"}])
    assert agent._conversation_ran_tools_recently(msgs) is True
    real = _t.time
    monkeypatch.setattr(_t, "time", lambda: real() + agent._TOOL_CONV_TTL_S + 5)
    assert agent._conversation_ran_tools_recently(msgs) is False



async def test_the_streamed_ending_records_the_tool_conversation():
    """R6: the web UI's common ending (the streamed final) returned before
    finalize and never recorded that the conversation ran tools."""
    import dataclasses
    from unittest.mock import MagicMock
    from tests.test_finalize_stream_pins import make_stream_agent, sse, _make_stream_state
    a = make_stream_agent()
    a._record_calibration_safe = AsyncMock()

    async def final_stream(payload, use_coding=False):
        yield sse({"content": "Here."})
        yield b"data: [DONE]\n\n"
    a.context.llm_client.stream_chat_completion = final_stream
    reg = MagicMock(); reg.is_cancelled.return_value = False
    msgs = [_u("make me an image")]
    st = dataclasses.replace(_make_stream_state(reg), messages=msgs,
                             stream_tools_snapshot=[{"name": "image_generation", "content": "SUCCESS"}])
    gen, _, _ = a._stream_final_generation(st)
    async for _c in gen:
        pass
    assert a._conversation_ran_tools_recently(msgs) is True
