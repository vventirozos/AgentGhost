"""The forced final that produced no answer — on the STREAM path (§4HF).

§4GH built the answer-now retry for the internal path and recorded the
streamed final as out of scope ("text already on the client cannot be
retried"). Request 69fb588e (the first live treatment-arm run of the
aligned planner) then delivered 868 s of work as narration: the planner
routed turn 10 as the final answer, the model streamed "I have enough
material to finalize. Let me do a final verification batch…" plus a search
call, the scrub took the call, and the stream closed with a note. The
[DONE] sentinel is held back to the end of the generator, so a second,
tool-less generation CAN ride the same stream. These pins drive the real
generator; each names the world it fails in.
"""
import json
import os
import sys

import pytest
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ghost_agent.core import agent as agent_mod
from ghost_agent.core.agent import StreamState, _FORCED_FINAL_ANSWER_DIRECTIVE
from ghost_agent.core.reply_smoothing import UNPARSED_TOOL_CALL_NOTE

from tests.test_finalize_stream_pins import make_stream_agent, sse

TOOLS = [{"name": "web_search", "content": "Reuters: Revolut confirmed the incident on 12 Sep."},
         {"name": "browser", "content": "securityaffairs: the notice quotes the sender generically."}]
_LIVE_TURN = ("I have enough material to finalize. Let me do a final verification batch on the "
              "sender's government domain and confirm the InfoCert relationship before delivering."
              "\n\n<tool_call>\n<function=web_search>\n<parameter=query>Revolut sender domain"
              "</parameter>\n</function>\n</tool_call>")
_ANSWER = ("## Revolut September 2026 — Forensic Summary\n\nRevolut confirmed the incident on "
           "12 Sep; the sender address is not established — the visible addresses are Revolut's "
           "own inboxes. Sources: Reuters, securityaffairs.")
_PREFIX = ("Strong material on the Italy theory. Now running parallel: dark-web search, plus "
           "extracting the notification text from securityaffairs.\n\n")


def _state(reg, tools, prefix="", task_closed=False):
    return StreamState(
        created_time=123, current_trajectory_id="t1",
        execution_failure_count=0, fname="chat", force_stop=False,
        forget_was_called=False, has_coding_intent=False,
        is_final_generation=True, last_user_content="q",
        last_was_failure=False, lc="q",
        messages=[{"role": "user", "content": "q"}], model="m",
        payload={"messages": [{"role": "user", "content": "investigate"}],
                 "tools": [{"type": "function", "function": {"name": "web_search"}}]},
        req_id="rid00001",
        stream_conv_fp="fp", stream_messages_snapshot=[], stream_model="m",
        stream_prefix=prefix, stream_thought="", stream_tools_snapshot=tools,
        stream_verify_messages=[], was_complex_task=True,
        _active_turn=None, _proj_task_closed_this_req=task_closed, _turn_reg=reg)


async def _drive(deltas, *, retry_reply, prefix="", tools=TOOLS, task_closed=False):
    a = make_stream_agent()
    a.context.args.no_verifier = True
    a.context.journal = MagicMock()
    a.context.metacog = MagicMock(); a.context.metacog.enabled = False
    a._journal_append_safe = AsyncMock()
    a._record_episode_safe = AsyncMock()
    a._judge_hydration_safe = MagicMock()
    a._write_project_work_log_safe = AsyncMock()
    a._record_calibration_safe = AsyncMock()
    a._record_turn_trajectory = MagicMock()
    a._attach_late_verdict_handler = MagicMock()

    async def _fake_verdict(**kw):
        return None
    a._compute_verifier_verdict = _fake_verdict

    async def final_stream(payload, use_coding=False):
        for d in deltas:
            yield sse({"content": d})
        yield b"data: [DONE]\n\n"
    a.context.llm_client.stream_chat_completion = final_stream
    retry_calls = []

    async def retry(payload, *args, **kw):
        retry_calls.append(payload)
        return {"choices": [{"message": {"content": retry_reply}}]}
    a.context.llm_client.chat_completion = AsyncMock(side_effect=retry)

    reg = MagicMock(); reg.is_cancelled.return_value = False
    gen, _, _ = a._stream_final_generation(_state(reg, tools, prefix, task_closed))
    chunks = [c async for c in gen]
    client = "".join(
        (json.loads(c.decode()[6:]).get("choices") or [{}])[0].get("delta", {}).get("content") or ""
        for c in chunks if c.startswith(b"data: ") and c.strip() != b"data: [DONE]")
    durable = (a._record_turn_trajectory.call_args.kwargs["final_content"]
               if a._record_turn_trajectory.called else "")
    return client, durable, retry_calls


@pytest.mark.asyncio
async def test_the_live_shape_gets_its_answer_on_the_same_stream():
    """FAILS IF: the stream path has no retry — the live world: narration
    plus a tool call, then [DONE]."""
    client, durable, retries = await _drive([_LIVE_TURN], retry_reply=_ANSWER, prefix=_PREFIX)
    assert len(retries) == 1
    assert "Forensic Summary" in client and client.rstrip().endswith(UNPARSED_TOOL_CALL_NOTE)
    assert "Forensic Summary" in durable
    # §4IS (req 309f45f8): the answer first, the caveat last — the order the
    # non-stream path delivers; the live reply had opened with the note
    assert client.index("Forensic Summary") < client.index(UNPARSED_TOOL_CALL_NOTE)
    assert client.count(UNPARSED_TOOL_CALL_NOTE) == 1


@pytest.mark.asyncio
async def test_the_retry_is_told_tools_are_off_and_keeps_the_prefix():
    """FAILS IF: the retry drops `tools` (a full re-prefill of the head)
    or forgets the directive / the model's own last output."""
    _, _, retries = await _drive([_LIVE_TURN], retry_reply=_ANSWER, prefix=_PREFIX)
    p = retries[0]
    assert UNPARSED_TOOL_CALL_NOTE not in p["messages"][-2]["content"]   # the model never sees its own caveat
    assert p.get("stream") is False
    assert p.get("tools") and p.get("tool_choice") == "none"
    assert p["messages"][-1] == {"role": "user", "content": _FORCED_FINAL_ANSWER_DIRECTIVE}
    assert p["messages"][-2]["role"] == "assistant" and "final verification batch" in p["messages"][-2]["content"]


@pytest.mark.asyncio
async def test_an_answer_that_streamed_is_never_retried():
    """FAILS IF: the retry fires on every final — a real answer would be
    answered twice."""
    client, _, retries = await _drive([_ANSWER], retry_reply="SHOULD NOT APPEAR", prefix=_PREFIX)
    assert retries == []
    assert "SHOULD NOT APPEAR" not in client


@pytest.mark.asyncio
async def test_a_tool_call_after_a_real_answer_in_this_turn_is_not_retried():
    """FAILS IF: the tool-call trigger ignores this turn's own text — a
    final that answers AND then tries one more call has answered."""
    client, _, retries = await _drive([_ANSWER + "\n\n<tool_call>\n<function=web_search>\n"
                                       "<parameter=query>x</parameter>\n</function>\n</tool_call>"],
                                      retry_reply="SHOULD NOT APPEAR")
    assert retries == []


@pytest.mark.asyncio
async def test_a_lone_beat_without_a_tool_call_is_the_4gh_predicate():
    """FAILS IF: the whole-body §4GH test is dropped from the stream path —
    the e57ad0cf shape (every paragraph a beat, no tool call) must still
    retry here, the path §4GH left out."""
    client, _, retries = await _drive(["Let me dig into the remaining sources now."],
                                      retry_reply=_ANSWER,
                                      prefix="I have good coverage. Let me read the last two articles.\n\n")
    assert len(retries) == 1 and "Forensic Summary" in client


@pytest.mark.asyncio
async def test_a_second_miss_ships_the_honest_fallback_with_the_evidence():
    """FAILS IF: a retry that is narration again is appended as if it were
    the answer, or nothing is appended."""
    client, durable, retries = await _drive(
        [_LIVE_TURN], retry_reply="Let me verify the InfoCert relationship first.", prefix=_PREFIX)
    assert len(retries) == 1
    assert "Let me verify the InfoCert" not in client
    assert "securityaffairs: the notice quotes the sender generically." in client
    assert "ask me to continue" in client.lower()
    assert "securityaffairs: the notice quotes" in durable


@pytest.mark.asyncio
async def test_a_retry_that_raises_costs_nothing_but_the_answer():
    """FAILS IF: an exception in the retry breaks the stream — the client
    must still get [DONE] and the note."""
    a_client = None
    async def boom(*a, **k):
        raise RuntimeError("upstream down")
    a = make_stream_agent()
    a.context.args.no_verifier = True
    a.context.journal = MagicMock(); a.context.metacog = MagicMock(); a.context.metacog.enabled = False
    for name in ("_journal_append_safe", "_record_episode_safe", "_write_project_work_log_safe", "_record_calibration_safe"):
        setattr(a, name, AsyncMock())
    a._judge_hydration_safe = MagicMock(); a._record_turn_trajectory = MagicMock(); a._attach_late_verdict_handler = MagicMock()
    async def _fv(**kw): return None
    a._compute_verifier_verdict = _fv
    async def final_stream(payload, use_coding=False):
        yield sse({"content": _LIVE_TURN}); yield b"data: [DONE]\n\n"
    a.context.llm_client.stream_chat_completion = final_stream
    a.context.llm_client.chat_completion = AsyncMock(side_effect=boom)
    reg = MagicMock(); reg.is_cancelled.return_value = False
    gen, _, _ = a._stream_final_generation(_state(reg, TOOLS, _PREFIX))
    chunks = [c async for c in gen]
    assert chunks[-1].strip() == b"data: [DONE]"
    assert any(UNPARSED_TOOL_CALL_NOTE.encode() in c or b"could not be parsed" in c for c in chunks)


@pytest.mark.asyncio
async def test_a_pure_tool_call_final_gets_the_retry_answer_not_the_canned_sentence():
    """§4HW (2026-09-16, req 503e94c5) — REVERSES the earlier pin. The scrub
    that ate the WHOLE reply used to ship its own "I prepared a tool call but
    this turn was routed as text-only … please rephrase" sentence and the
    retry was gated off "so there would not be two replies". Live, that
    sentence WAS the whole answer to "open this page and read the price".
    FAILS IF: the retry does not fire, or the sentence still ships, or the
    durable record keeps the scrubbed markup."""
    pure = ("<tool_call>\n<function=vision_analysis>\n<parameter=path>x.png</parameter>\n"
            "</function>\n</tool_call>")
    client, durable, retries = await _drive([pure], retry_reply=_ANSWER, prefix=_PREFIX)
    assert len(retries) == 1
    assert _ANSWER in client
    assert "routed as text-only" not in client
    assert client.count("Forensic Summary") == 1          # one reply, not two
    assert durable.startswith(_PREFIX.rstrip()) and durable.endswith(_ANSWER)
    assert "<tool_call>" not in durable and "vision_analysis" not in durable


@pytest.mark.asyncio
async def test_a_full_scrub_whose_retry_also_misses_ships_the_honest_fallback():
    """FAILS IF: a second miss after a full scrub falls back to the canned
    sentence instead of the §4GH evidence fallback (which the shape check
    refutes as the non-answer it is)."""
    pure = ("<tool_call>\n<function=web_search>\n<parameter=query>Revolut</parameter>\n"
            "</function>\n</tool_call>")
    client, _, retries = await _drive([pure], retry_reply="Let me look into that further.")
    assert len(retries) == 1
    assert "ran out of this turn's budget" in client
    assert "routed as text-only" not in client


@pytest.mark.asyncio
async def test_a_task_closed_scrub_still_ships_the_task_complete_sentence_without_a_retry():
    pure = ("<tool_call>\n<function=manage_projects>\n<parameter=action>next</parameter>\n"
            "</function>\n</tool_call>")
    client, _, retries = await _drive([pure], retry_reply="SHOULD NOT APPEAR", task_closed=True)
    assert retries == []
    assert "Task complete" in client
    assert "SHOULD NOT APPEAR" not in client


@pytest.mark.asyncio
async def test_an_inline_think_block_in_the_retry_never_reaches_the_client():
    """FAILS IF: the retry's reasoning is appended as answer text — the
    model puts its plan in <think> when thinking is on (the coding-leaf
    trap), and the stream path has no separate reasoning channel here."""
    client, durable, retries = await _drive(
        [_LIVE_TURN], retry_reply="<think>plan the answer first</think>\n\n" + _ANSWER, prefix=_PREFIX)
    assert len(retries) == 1
    assert "plan the answer first" not in client and "Forensic Summary" in client
    assert "plan the answer first" not in durable


@pytest.mark.asyncio
async def test_a_full_scrub_whose_retry_raises_ships_the_sentence_as_last_resort():
    """FAILS IF: a retry that raises after a full scrub leaves the client
    with NOTHING (the deferred sentence must still go out), or the stream
    loses [DONE]."""
    async def boom(*a, **k):
        raise RuntimeError("upstream down")
    a = make_stream_agent()
    a.context.args.no_verifier = True
    a.context.journal = MagicMock(); a.context.metacog = MagicMock(); a.context.metacog.enabled = False
    for name in ("_journal_append_safe", "_record_episode_safe", "_write_project_work_log_safe", "_record_calibration_safe"):
        setattr(a, name, AsyncMock())
    a._judge_hydration_safe = MagicMock(); a._record_turn_trajectory = MagicMock(); a._attach_late_verdict_handler = MagicMock()
    async def _fv(**kw): return None
    a._compute_verifier_verdict = _fv
    pure = ("<tool_call>\n<function=vision_analysis>\n<parameter=path>x.png</parameter>\n"
            "</function>\n</tool_call>")
    async def final_stream(payload, use_coding=False):
        yield sse({"content": pure}); yield b"data: [DONE]\n\n"
    a.context.llm_client.stream_chat_completion = final_stream
    a.context.llm_client.chat_completion = AsyncMock(side_effect=boom)
    reg = MagicMock(); reg.is_cancelled.return_value = False
    gen, _, _ = a._stream_final_generation(_state(reg, TOOLS, ""))
    chunks = [c async for c in gen]
    client = "".join(
        (json.loads(c.decode()[6:]).get("choices") or [{}])[0].get("delta", {}).get("content") or ""
        for c in chunks if c.startswith(b"data: ") and c.strip() != b"data: [DONE]")
    assert chunks[-1].strip() == b"data: [DONE]"
    assert "routed as text-only" in client and "vision_analysis" in client


# ── §4IS (req 309f45f8): three scrubbed calls opened the reply with six newlines ──

_THREE_CALLS = [
    "<tool_call>\n<function=web_search>\n<parameter=query>a</parameter>\n</function>\n</tool_call>",
    "\n\n<tool_call>\n<function=web_search>\n<parameter=query>b</parameter>\n</function>\n</tool_call>",
    "\n\n<tool_call>\n<function=web_search>\n<parameter=query>c</parameter>\n</function>\n</tool_call>",
]


@pytest.mark.asyncio
async def test_seams_before_the_first_visible_character_are_not_streamed():
    """FAILS IF: the scrub streams the whitespace each removed block leaves
    while nothing visible has gone out yet — the client's reply began with a
    stack of blank lines and the note; the answer came after."""
    client, durable, retries = await _drive(_THREE_CALLS, retry_reply=_ANSWER)
    assert len(retries) == 1
    assert not client.startswith(("\n", " ")), repr(client[:40])
    assert client.lstrip().startswith("## Revolut")                      # the answer opens the reply
    # a generation that was ONLY calls is the all-consumed case: the retry's
    # answer is the reply and no note is due (nothing the user saw described
    # a step) — the dropped seams must not make it look partially delivered
    assert client.count(UNPARSED_TOOL_CALL_NOTE) == 0
    assert _ANSWER in durable


@pytest.mark.asyncio
async def test_a_partial_scrub_with_no_retry_answer_still_carries_the_note_once():
    """The note is deferred, not dropped: when the retry machinery yields no
    answer the note still goes out, exactly once."""
    partial = "Here is what I found so far.\n\n" + _THREE_CALLS[0]
    client, _, retries = await _drive([partial], retry_reply="Let me look into that further.")
    assert client.count(UNPARSED_TOOL_CALL_NOTE) == 1

