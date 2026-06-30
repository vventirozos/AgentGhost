"""End-to-end behavioral test for the streaming-scrub + empty-output
fallback.

The scrub strips `<tool_call>` / `<function>` XML from the stream when
`is_final_generation` is true (the model hallucinated a tool call on a
text-only turn). But when the entire upstream response is pure XML (no
prose), the scrub produces an empty stream and the user would see
nothing.

The fix was a fallback SSE chunk that emits an actionable message when
the scrub consumed everything. This test drives `handle_chat` end-to-
end with a planner mock forcing `force_final_response=True` and an
upstream stream that emits pure `<tool_call>` XML — and asserts the
client receives a non-empty, informative message.

NB: the original repro used a repeated `self play` to reach this path.
That specific input is now handled deterministically BEFORE the LLM
turn (see `_explicit_terminal_command` / the turn-0 dispatch in
`agent.py`, and `tests/test_self_play_deterministic_dispatch.py`), so
this test uses a neutral user message — the scrub remains the general
safety net for any final-generation turn that hallucinates a tool call.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_empty_scrub_output_produces_fallback(monkeypatch):
    """Drive handle_chat's streaming path with:
      - A planner response that makes next_action_id = 'none' so
        force_final_response flips true and the scrub activates.
      - A stream that emits only <tool_call>...</tool_call> XML.
    The client must receive a fallback informing them the tool
    wasn't executed and telling them how to rephrase."""
    from ghost_agent.core.agent import GhostAgent

    ctx = MagicMock()
    ctx.args = MagicMock()
    ctx.args.verbose = False
    ctx.args.max_context = 4000
    ctx.args.smart_memory = 0.0
    ctx.args.temperature = 0.5
    ctx.args.use_planning = True
    ctx.args.native_tools = False
    ctx.profile_memory = MagicMock()
    ctx.profile_memory.get_context_string.return_value = ""
    ctx.scratchpad = MagicMock()
    ctx.scratchpad.list_all.return_value = ""
    ctx.memory_system = None
    ctx.skill_memory = None
    ctx.journal = None

    call_log = {"chat_completion": 0}

    async def mock_chat_completion(payload, *a, **kw):
        call_log["chat_completion"] += 1
        # Only the planner's call goes through chat_completion (it's
        # non-streaming). Return planner JSON that forces
        # force_final_response=True on the next turn.
        return {"choices": [{"message": {
            "content": '{"thought":"nothing to do","next_action_id":"none","required_tool":"all"}'
        }}]}

    async def fake_stream(*args, **kwargs):
        # The upstream stream (only called when is_final_generation +
        # stream_response) emits pure tool_call XML, no prose. This is
        # the pathological shape a final-generation turn hits when the
        # model hallucinates a lone tool call instead of answering.
        for t in [
            '<tool_call>\n',
            '<function name="self_play">\n',
            '</function>\n',
            '</tool_call>',
        ]:
            chunk = {"choices": [{"delta": {"content": t}}]}
            yield f"data: {json.dumps(chunk)}\n\n".encode('utf-8')
        yield b"data: [DONE]\n\n"

    # NOTE: the global conftest fixture `inject_global_stream_adapter`
    # clobbers `stream_chat_completion` when it's still a MagicMock/
    # AsyncMock. Attach a plain async-generator function instead so the
    # fixture leaves it alone and our pure-XML stream actually runs.
    ctx.llm_client = MagicMock()
    ctx.llm_client.chat_completion = AsyncMock(side_effect=mock_chat_completion)
    ctx.llm_client.stream_chat_completion = fake_stream

    agent = GhostAgent(ctx)
    body = {
        # Must (a) NOT be an explicit terminal command, or the turn-0
        # deterministic dispatch would intercept it, and (b) be
        # non-conversational (contain an action verb like "summarize") so
        # the planner runs and forces the final-generation streaming path
        # this test exercises. "self play" (the original repro input) hit
        # this path because "play" is an action verb — but it now routes
        # through deterministic dispatch instead.
        "messages": [{"role": "user", "content": "summarize my recent activity"}],
        "model": "test",
        "stream": True,
    }
    result, _, _ = await agent.handle_chat(body, MagicMock())

    received_content = []
    raw_chunks_in_order = []
    async for chunk in result:
        raw_chunks_in_order.append(chunk)
        s = chunk.decode("utf-8", errors="replace")
        if s.startswith("data: ") and s.strip() != "data: [DONE]":
            try:
                data = json.loads(s[6:])
                delta = data["choices"][0].get("delta", {})
                if "content" in delta and delta["content"] is not None:
                    received_content.append(delta["content"])
            except Exception:
                pass

    joined = "".join(received_content)

    # Assertion 0 (ordering): the fallback chunk must be emitted BEFORE
    # the `[DONE]` sentinel. SSE clients close on [DONE], so if [DONE]
    # arrives first the fallback never reaches the user even though the
    # bytes were yielded. Observed trace: the pretty_log "Scrub consumed
    # entire response" fired but the user's CLI rendered empty — that
    # was [DONE] arriving at index 0 and the fallback at index 1.
    done_idx = None
    fallback_idx = None
    for i, ch in enumerate(raw_chunks_in_order):
        s = ch.decode("utf-8", errors="replace")
        if s.strip() == "data: [DONE]" and done_idx is None:
            done_idx = i
        if "prepared a tool call" in s and fallback_idx is None:
            fallback_idx = i
    assert done_idx is not None, "stream never emitted [DONE]"
    assert fallback_idx is not None, "stream never emitted the fallback text"
    assert fallback_idx < done_idx, (
        f"[DONE] at index {done_idx} arrived BEFORE fallback at index "
        f"{fallback_idx}. SSE clients close on [DONE] and never see "
        f"chunks that come after it."
    )


def test_source_fallback_is_single_branch_after_direct_summary():
    """The two-branch fallback was collapsed back to one once terminal
    tools stopped taking the summary LLM path. Their result is now
    written directly to `final_ai_content` from the tool-execution
    loop — the stream-scrub empty-output fallback can no longer fire
    when a terminal tool ran, so the 'cycle already completed' branch
    became dead code and was removed.

    The remaining single branch handles only the
    planner-routed-as-text-only-but-model-emits-tool_call case and
    still says 'please rephrase'."""
    from pathlib import Path
    src = (
        Path(__file__).resolve().parent.parent
        / "src" / "ghost_agent" / "core" / "agent.py"
    ).read_text()
    # Scan code lines only — explanatory comments legitimately
    # reference the retired branch ("The `cycle already completed`
    # branch ... was retired ...").
    code = "\n".join(
        line for line in src.splitlines()
        if line.lstrip() and not line.lstrip().startswith("#")
    )

    # The dead completed-cycle wording must be gone from code.
    assert "cycle already" not in code
    assert "completed this turn" not in code
    # And the `_tools_already_run` helper was retired alongside it.
    assert "_tools_already_run" not in code

    # The single remaining branch must still say "wasn't executed"
    # — that's the planner-routed-as-text-only message.
    assert "wasn't executed" in code
