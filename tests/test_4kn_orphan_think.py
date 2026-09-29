"""§4KN — the orphan `</think>` (live probe-f6, 2026-09-29).

The chat template opens the think block in the prompt, so a turn's content can
be reasoning that ends in a bare `</think>`; the reply shipped
"…Let me look it up.\\n</think>\\n\\n…". The first fix lived in the shared
`_strip_think_blocks` — which also strips the PARSER's target — and dropped real
tool calls whose content held the tag (review §4KN R1, CRIT). The fix is now a
separate `_strip_orphan_think_close`, applied to the user-visible text only,
and matching only the shape every recorded leak has: the tag on its own line,
followed by a blank line or the end.

World where each pin fails: an orphan-closed reasoning prefix reaches a reply;
prose, a list item, an indented or fenced mention loses the text before it; a
tool call whose content holds the tag is dropped or mangled by the parser.
"""
import json
from unittest.mock import AsyncMock

import pytest

from ghost_agent.core import agent as A
from tests.helpers import FakeBgTasks
from tests.test_4kl_member_capability import _loop_agent
from tests.test_requester_role import _agent as _member_agent, _resp, _tc


@pytest.mark.parametrize("text,expect", [
    # the three leaks on record (3 of 3,232 final replies)
    ("Let me check my recent activity to find the details.\n\n</think>\n\nThe most recent one", "The most recent one"),
    ("I cannot directly download. Let me look it up.\n</think>\n\nI cannot download the CSV.",
     "I cannot download the CSV."),
    ("</think>\n\nHere's what I found.", "Here's what I found."),
    ("reasoning\r\n</think>\r\n\r\nAnswer", "Answer"),                          # CRLF
    ("reasoning\n</thinking>\n\nAnswer", "Answer"),                           # the `thinking` spelling
    ("I must not emit a <tool_call>.\n</think>\n\nThe answer is 42.", "The answer is 42."),   # a call MENTION
])
def test_an_orphan_close_drops_the_reasoning_before_it(text, expect):
    assert A._strip_orphan_think_close(text) == expect


@pytest.mark.parametrize("text", [
    "Qwen3 closes its reasoning with </think>",                                # end of a prose line
    "Tags:\n- open: `<think>`\n- close: </think>\n\nThat is all.",           # a list item
    "It ends reasoning with\n</think>\nand then answers.",                    # own line, no blank line after
    "Qwen closes its reasoning with this tag:\n\n</think>",                  # a reply that ENDS with the tag
    "reasoning only\n</think>",                                               # no blank line after: not the shape
    "Template:\n\n    <think>\n    reasoning\n    </think>\n\n    answer\n\nThat's it.",   # indented code
    "~~~\n<think>\nx\n</think>\n\ny\n~~~\nDone.",                                  # a ~~~ fence
    "~~~\nouter ```\ninner\n</think>\n\n``` more\n~~~\nDone.",                   # nested fences restore whole
    '{"name": "file_system", "arguments": {"content": "a\n</think>\n\nthen"}}',  # a raw-JSON call
    "```\nreasoning\n</think>\n\n```\nAnswer",                                # inside a fence
    "Ο Qwen κλείνει τη σκέψη με </think>",                                    # Greek prose
    "The model emits </think> when it stops.",
    "No tags.",
    "",
])
def test_a_mention_is_prose(text):
    assert A._strip_orphan_think_close(text) == text


def test_the_shared_stripper_does_not_touch_an_orphan_close():
    """The parser strips its target with `_strip_think_blocks`; a call's own
    content may hold the tag, so the shared stripper must leave it alone."""
    text = "reasoning\n</think>\n\n<tool_call>…"
    assert A._strip_think_blocks(text) == text


@pytest.mark.parametrize("call", [
    '<tool_call>\n<function=file_system>\n<parameter=operation>write</parameter>\n<parameter=path>p.md</parameter>\n'
    '<parameter=content>\nClose the block with\n</think>\n\nthen answer.\n</parameter>\n</function>\n</tool_call>',
    '<function_name=file_system>\n<parameter=operation>write</parameter>\n<parameter=path>p.md</parameter>\n'
    '<parameter=content>\nClose with\n</think>\n\nthen answer.\n</parameter>\n</function_name>',
    '{"name": "file_system", "arguments": {"operation": "write", "path": "p.md", '
    '"content": "Close with\n</think>\n\nthen answer."}}',
])
def test_a_call_whose_content_holds_the_tag_still_parses(call):
    """The dialects the review found dropped: XML, `<function_name=`, raw JSON
    (with a literal newline, which the parser accepts)."""
    agent = _loop_agent()
    agent.available_tools = {"file_system": AsyncMock()}
    calls, _, fail = agent._parse_assistant_tool_calls(call, {"role": "assistant", "content": call})
    assert calls and calls[0]["function"]["name"] == "file_system", (calls, fail)
    args = calls[0]["function"]["arguments"]
    args = json.loads(args) if isinstance(args, str) else args
    assert "</think>" in args.get("content", "")


@pytest.mark.parametrize("native", [True, False])
async def test_the_live_shape_never_reaches_a_members_reply(monkeypatch, tmp_path, native):
    """probe-f6 end to end, with the call native AND written in the content."""
    agent, ctx, _ = _member_agent(monkeypatch, tmp_path)
    agent.available_tools = {"web_search": AsyncMock(return_value="### 1. hw_200.csv\nheights and weights")}
    lead = "I cannot directly download files in this channel mode. Let me look it up.\n</think>\n\n"
    if native:
        turn1 = _resp(lead, [_tc("c1", "web_search", {"query": "hw_200.csv average height"})])
    else:
        turn1 = _resp(lead + '<tool_call>\n<function=web_search>\n<parameter=query>hw_200.csv average height'
                             '</parameter>\n</function>\n</tool_call>')
    replies = iter([turn1, _resp("I cannot download the CSV here. To get the average, paste the height "
                                 "column into the chat."), _resp("x"), _resp("x")])
    ctx.llm_client.chat_completion = AsyncMock(side_effect=lambda *a, **k: next(replies))
    final, _, _ = await agent.handle_chat(
        {"messages": [{"role": "user", "content": "download the csv and average the heights"}]},
        FakeBgTasks(), request_id=f"web-4kn-{native}", requester_role="member")
    assert "</think>" not in final and "Let me look it up" not in final
    assert "paste the height column" in final
    assert agent.available_tools["web_search"].await_count == 1          # the call still ran


def test_only_the_first_close_ends_the_reasoning():
    """Everything after the first qualifying close is the reply — including
    a later line that happens to be the bare tag."""
    text = "reasoning\n</think>\n\nThe answer, and the tag on its own line:\n\n</think>\n\nmore."
    assert A._strip_orphan_think_close(text) == "The answer, and the tag on its own line:\n\n</think>\n\nmore."



async def test_the_streamed_final_retry_strips_it_too():
    """The one non-streamed text the stream path appends: the no-answer
    retry. Its reasoning must not ride along (review §4KN R2, MAJOR)."""
    from tests.test_stream_forced_final_retry import _drive, _LIVE_TURN, _PREFIX, _ANSWER
    client, durable, retries = await _drive(
        [_LIVE_TURN], retry_reply="Let me write the summary now.\n</think>\n\n" + _ANSWER, prefix=_PREFIX)
    assert len(retries) == 1
    assert "Forensic Summary" in client and "</think>" not in client
    assert "Let me write the summary now" not in client and "</think>" not in durable


def test_on_a_call_turn_the_text_may_end_with_the_tag():
    text = "I cannot download it. Let me look it up.\n</think>"
    assert A._strip_orphan_think_close(text, call_turn=True) == ""
    assert A._strip_orphan_think_close(text) == text                  # an answer that ends with it is prose


async def test_a_retry_that_rewrites_its_call_is_a_call_turn():
    """The retry is told tools are off, yet often re-writes the call; once the
    markup is scrubbed the text ENDS with the tag (review §4KN R3, MAJOR)."""
    from tests.test_stream_forced_final_retry import _drive, _LIVE_TURN, _PREFIX
    reply = ("The mean is about 170 cm, but I should verify with the tool.\n</think>\n"
             "<tool_call>\n<function=web_search>\n<parameter=query>hw_200</parameter>\n</function>\n</tool_call>")
    client, durable, retries = await _drive([_LIVE_TURN], retry_reply=reply, prefix=_PREFIX)
    assert len(retries) == 1
    assert "</think>" not in client and "I should verify with the tool" not in client


async def test_a_text_only_answer_that_ends_with_the_tag_survives_end_to_end(monkeypatch, tmp_path):
    """No call this turn: the text IS the answer, and an answer about the tag
    may end with it on its own line."""
    agent, ctx, _ = _member_agent(monkeypatch, tmp_path)
    answer = "Qwen closes its reasoning block with this tag on its own line:\n\n</think>"
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[_resp(answer), _resp("x"), _resp("x")])
    final, _, _ = await agent.handle_chat(
        {"messages": [{"role": "user", "content": "which tag closes qwen's reasoning block?"}]},
        FakeBgTasks(), request_id="web-4kn-ans", requester_role="owner")
    assert "</think>" in final and "closes its reasoning block" in final


async def test_a_final_generation_that_drops_a_call_keeps_its_answer(monkeypatch, tmp_path):
    """Planner says done → final generation: calls are dropped and the text IS
    the reply, so it is not a call turn's narration (review §4KN R3)."""
    from tests.test_requester_role import _force_planning_arm
    agent, ctx, _ = _member_agent(monkeypatch, tmp_path)
    _force_planning_arm(monkeypatch)
    ctx.args.use_planning = True
    agent.available_tools = {"web_search": AsyncMock(return_value="### 1. hit")}
    answer = ("Qwen closes its reasoning block with this tag on its own line:\n\n</think>\n"
              "<tool_call>\n<function=web_search>\n<parameter=query>x</parameter>\n</function>\n</tool_call>")
    plan = {"thought": "answer now", "next_action_id": "root", "required_tool": "none",
            "tree_update": {"id": "root", "description": "A", "status": "DONE", "children": []}}

    async def chat(payload, *a, **k):
        msgs = payload.get("messages") or []
        last = str((msgs[-1] or {}).get("content") or "") if msgs else ""
        if "### AVAILABLE NATIVE TOOLS" in last:
            return _resp("```json\n" + json.dumps(plan) + "\n```")
        return _resp(answer)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=chat)
    final, _, _ = await agent.handle_chat(
        {"messages": [{"role": "user", "content": "check which tag closes qwen's reasoning block"}]},
        FakeBgTasks(), request_id="web-4kn-ff", requester_role="owner")
    assert "closes its reasoning block" in final
    assert agent.available_tools["web_search"].await_count == 0
