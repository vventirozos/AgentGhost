"""§4RF (2026-10-25, operator: "the latest request did … Model stalled after
thinking. Forcing retry. why ?"). The live WebOS turn thought for 16,252
tokens (51,214 chars, the whole app drafted in its thinking), hit the output
limit (`stop: length`) and produced no answer and no tool call. The agent
logged a STALL and told the model it had "stopped abruptly". These tests drive
the real `handle_chat` path and fail in the world where the limit is still
reported as a stall — and in the world where a real stall stops being one."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import ghost_agent.core.agent as A
from ghost_agent.core.agent import GhostAgent, GhostContext


@pytest.fixture
def agent():
    A._LENGTH_STOPS.clear()
    context = MagicMock(spec=GhostContext)
    context.args = MagicMock()
    context.args.smart_memory = 0.0
    context.args.max_context = 4000
    context.args.temperature = 0.7
    context.args.native_tools = False
    context.args.tool_turn_max_tokens = 0          # production: unset (a MagicMock int()s to 1)
    context.verifier = None
    context.self_model = None
    context.llm_client = MagicMock()
    context.sandbox_dir = None
    context.memory_system = None
    context.profile_memory = None
    context.semantic_memory = None
    context.skill_memory = None
    context.journal = None
    context.scratchpad = MagicMock()
    context.scratchpad.list_all.return_value = ""
    yield GhostAgent(context=context)
    A._LENGTH_STOPS.clear()


def _draft(n):
    return f"Drafting the app, part {n}: const APPS{n}={{dock:'files'}};"


def _two_replies(first_finish, first_content="", cuts=1):
    """Replies 1..`cuts`: thinking only (or `first_content` after it), ending
    with `first_finish`. The next reply: an answer. Records the messages and
    the max_tokens each reply was sent."""
    sent = []

    async def _stream(payload, *a, **kw):
        sent.append(json.loads(json.dumps(payload.get("messages", []), default=str)))
        limits.append(payload.get("max_tokens"))
        if len(sent) <= cuts:
            chunks = [{"choices": [{"delta": {"reasoning_content": _draft(len(sent))}}]}]
            if first_content:
                chunks.append({"choices": [{"delta": {"content": first_content}}]})
            chunks.append({"choices": [{"delta": {}, "finish_reason": first_finish}]})
        else:
            chunks = [{"choices": [{"delta": {"content": "The file is written."}}]},
                      {"choices": [{"delta": {}, "finish_reason": "stop"}]}]
        for c in chunks:
            yield f"data: {json.dumps(c)}\n\n".encode()
        yield b"data: [DONE]\n\n"
    limits = []
    _stream.limits = limits
    return _stream, sent


async def _run(agent, first_finish, first_content="", cuts=1):
    stream, sent = _two_replies(first_finish, first_content, cuts)
    agent.context.llm_client.stream_chat_completion = stream
    agent.context.llm_client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]})
    body = {"messages": [{"role": "user", "content": "build me a web OS in one html file"}],
            "stream": False}
    with patch("ghost_agent.core.agent.get_active_tool_definitions", return_value=[]), \
            patch("ghost_agent.core.agent.pretty_log") as plog:
        await agent.handle_chat(body, background_tasks=MagicMock(), request_id="req-4rf")
    logs = [" ".join(str(x) for x in c.args) for c in plog.call_args_list]
    agent._limits = stream.limits
    return sent, logs


def _alert(sent):
    assert len(sent) >= 2, "the cut reply was never retried"
    # the alert may be folded into the system-state block that follows it
    alerts = [str(m.get("content")) for m in sent[1]
              if m.get("role") == "user" and "SYSTEM ALERT: " in str(m.get("content"))]
    assert alerts, "the retry carried no alert"
    return "\n".join(alerts)


@pytest.mark.asyncio
async def test_a_cut_inside_thinking_is_named_as_the_output_limit(agent):
    """Fails where a `length` stop with only thinking was logged as a stall and
    the model was told it 'stopped abruptly' (live WebOS turn, 714 s)."""
    sent, logs = await _run(agent, "length")
    alert = _alert(sent)
    assert "OUTPUT LIMIT while you were still thinking" in alert
    assert "YOUR DRAFT" in alert and _draft(1) in alert     # the cut thinking rides along verbatim
    assert "stopped abruptly" not in alert
    assert "more_parts=true" in alert                       # the parts route is named
    assert "cut-off #1" in alert and "limited to" not in alert   # §4OA count; no lower limit (§4RF)
    assert any("Cut off at the output limit while thinking" in ln for ln in logs)
    assert not any("Model stalled after thinking" in ln for ln in logs)


@pytest.mark.asyncio
async def test_the_retry_after_a_cut_keeps_the_full_output_limit(agent):
    """§4RF step 2 (operator: "remove the step-down altogether and keep 16k
    for everything"). Fails where a cut lowers the next reply's max_tokens
    under ANY name — the 8k step-down cut 5A and 6F again inside the thinking."""
    await _run(agent, "length")
    assert len(agent._limits) >= 2
    assert agent._limits[0] == A.DEFAULT_TOOL_TURN_MAX_TOKENS
    assert agent._limits[1] == A.DEFAULT_TOOL_TURN_MAX_TOKENS


@pytest.mark.asyncio
async def test_a_cut_tool_call_is_not_called_a_cut_in_thinking(agent):
    """Fails where a reply cut inside a <tool_call> (thinking, then the call's
    arguments) is reported as cut while thinking — that is the cut-call path."""
    sent, logs = await _run(agent, "length",
                            '<tool_call>{"name": "file_system", "arguments": {"operation": "write", '
                            '"path": "a.html", "content": "<html><body>half a page')
    assert any("Output Cut Off" in ln for ln in logs)          # the cut-call path took it
    assert not any("Cut off at the output limit while thinking" in ln for ln in logs)
    assert all("OUTPUT LIMIT while you were still thinking" not in str(m.get("content"))
               for msgs in sent for m in msgs)


@pytest.mark.asyncio
async def test_a_real_stall_is_still_a_stall(agent):
    """Fails where every thinking-only reply is called a cut — a `stop` with no
    answer is the model stopping, not the limit."""
    sent, logs = await _run(agent, "stop")
    alert = _alert(sent)
    assert "stopped abruptly" in alert
    assert "OUTPUT LIMIT" not in alert
    assert any("Model stalled after thinking" in ln for ln in logs)


@pytest.mark.asyncio
async def test_every_later_reply_still_sees_every_draft(agent):
    """Option 1 (operator: "go with option 1"). Fails where the draft rode
    only on the retry and a later part (or a second cut) lost it: after two
    cuts the third reply must see both drafts, each once."""
    agent.context.args.max_context = 174000          # production (launcher --max-context)
    sent, logs = await _run(agent, "length", cuts=2)
    assert len(sent) >= 3
    seen = "\n".join(str(m.get("content")) for m in sent[2])
    assert seen.count(_draft(1)) == 1 and seen.count(_draft(2)) == 1
    assert sum("Retrying with the thinking kept as the draft" in ln for ln in logs) == 2


def test_the_message_counts_the_thinking_and_caps_nothing():
    msg = A.cut_in_thinking_message("x" * 51214, "\nThis is cut-off #1 in this request")
    assert "51,214 characters of thinking" in msg
    assert msg.index("cut-off #1") < msg.index("YOUR DRAFT (") and msg.endswith("END OF YOUR DRAFT -----")
    # §4RF: name the cause only — no instruction to think less or lower effort
    for limiting in ("think less", "shorter thinking", "reasoning effort", "do not think"):
        assert limiting not in msg.lower()
