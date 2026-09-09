"""Streamed replies: the persisted copy gets the scrub and the smoother, and
a partial scrub tells the user a step did not run (§4FS, 2026-09-09).

The web UI streams and returns before `_finalize_and_return`, so the
smoother (2026-07-17) and the new markup scrub never touched the common
path. Evidence: two multi-tool replies on 2026-09-08 whose "I'll build…"
openers the smoother removes on sight when run directly, and one that kept
a `<tool_call>` block. The live stream is already delivered; the RECORD is
what learns. These pins read the stream generator's source: the generator
is ~1,400 lines of SSE plumbing; the pins below drive it for real through
the harness the stream tests already use.
"""
import inspect
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ghost_agent.core import agent as agent_mod
from ghost_agent.core.reply_smoothing import UNPARSED_TOOL_CALL_NOTE, smooth_reply


def _stream_src():
    return inspect.getsource(agent_mod.GhostAgent._stream_final_generation)


def test_the_note_is_the_shared_constant():
    assert "could not be parsed and was NOT executed" in UNPARSED_TOOL_CALL_NOTE


def test_the_2026_09_08_opener_is_what_the_record_now_loses():
    """The evidence, executed: the persisted copy of the schema-diff reply
    starts with narration the smoother removes."""
    reply = ("I'll build a schema-diff tool for two `pg_dump -s` files. The immediate step is writing the script, "
             "then testing it with sample dumps.\n\n"
             "I'll test the script with two sample schema dumps containing meaningful differences.\n\n"
             "The tool is at `schema_diff.py`; it reports 14 differences between the two dumps.")
    out = smooth_reply(reply)
    assert out.startswith("The tool is at `schema_diff.py`"), out


# --- executed: the real generator, the real harness ------------------------

from unittest.mock import AsyncMock, MagicMock

from tests.test_finalize_stream_pins import make_stream_agent, sse
from tests.test_finalize_stream_r1_fixes import _client_text
from ghost_agent.core.agent import StreamState


def _state(reg, tools):
    return StreamState(
        created_time=123, current_trajectory_id="t1",
        execution_failure_count=0, fname="chat", force_stop=False,
        forget_was_called=False, has_coding_intent=False,
        is_final_generation=True, last_user_content="q",
        last_was_failure=False, lc="q",
        messages=[{"role": "user", "content": "q"}], model="m",
        payload={"messages": []}, req_id="rid00001",
        stream_conv_fp="fp", stream_messages_snapshot=[], stream_model="m",
        stream_prefix="", stream_thought="", stream_tools_snapshot=tools,
        stream_verify_messages=[], was_complex_task=False,
        _active_turn=None, _proj_task_closed_this_req=False, _turn_reg=reg)


async def _drive_and_record(deltas, tools):
    a = make_stream_agent()
    a._record_calibration_safe = AsyncMock()
    a._record_turn_trajectory = MagicMock()

    async def final_stream(payload, use_coding=False):
        for d in deltas:
            yield sse({"content": d})
        yield b"data: [DONE]\n\n"
    a.context.llm_client.stream_chat_completion = final_stream
    reg = MagicMock(); reg.is_cancelled.return_value = False
    gen, _, _ = a._stream_final_generation(_state(reg, tools))
    chunks = [c async for c in gen]
    assert a._record_turn_trajectory.called, "the streamed turn was not recorded"
    return _client_text(chunks), a._record_turn_trajectory.call_args.kwargs["final_content"]


NARRATED = ["Let me check the file first.\n\n", "The file has 3 lines and no trailing newline."]


@pytest.mark.asyncio
async def test_executed_the_record_of_a_tool_turn_is_smoothed_and_the_stream_is_not():
    """World where it fails: the streaming smoother is a silent no-op (a
    NameError inside its try/except would look exactly like that) — the
    record keeps the opener the non-stream path would have removed."""
    client, recorded = await _drive_and_record(NARRATED, tools=[{"name": "file_system"}, {"name": "execute"}])
    assert client.startswith("Let me check the file first."), "the live stream is delivered as it was"
    assert recorded.startswith("The file has 3 lines"), recorded
    assert "Let me check" not in recorded


@pytest.mark.asyncio
async def test_executed_a_zero_tool_turn_is_recorded_as_delivered():
    """Conversational replies are never rewritten."""
    _, recorded = await _drive_and_record(NARRATED, tools=[])
    assert recorded.startswith("Let me check the file first."), recorded


@pytest.mark.asyncio
async def test_executed_a_single_tool_turn_is_recorded_as_delivered():
    """The gate is ≥2 real tools on the stream path as on the other — the
    2026-07-17 decision, kept after a one-day trial of ≥1 (review)."""
    _, recorded = await _drive_and_record(NARRATED, tools=[{"name": "file_system"}])
    assert recorded.startswith("Let me check the file first."), recorded


@pytest.mark.asyncio
async def test_executed_a_synthetic_tool_does_not_open_the_gate():
    _, recorded = await _drive_and_record(NARRATED, tools=[{"name": "x", "_synthetic": True}, {"name": "y", "_synthetic": True}])
    assert recorded.startswith("Let me check the file first."), recorded


@pytest.mark.asyncio
async def test_executed_a_partial_scrub_notes_the_stream_and_cleans_the_record():
    deltas = ["Saving it now.\n\n", "<tool_call>", "<function=file_system>", "</function>", "</tool_call>", "\n\nSaved."]
    client, recorded = await _drive_and_record(deltas, tools=[{"name": "file_system"}])
    assert client.rstrip().endswith(UNPARSED_TOOL_CALL_NOTE), client
    assert client.count(UNPARSED_TOOL_CALL_NOTE) == 1
    assert "<tool_call>" not in recorded and "<function=" not in recorded, recorded
    assert "Saved." in recorded


@pytest.mark.asyncio
async def test_executed_no_markup_no_note():
    client, recorded = await _drive_and_record(["Plain answer.", " Nothing else."], tools=[{"name": "x"}])
    assert UNPARSED_TOOL_CALL_NOTE not in client and UNPARSED_TOOL_CALL_NOTE not in recorded


@pytest.mark.asyncio
async def test_executed_an_all_consumed_scrub_gets_the_fallback_not_the_note():
    """The all-consumed case already emits its own fallback text; the note
    must not double it (the `_scrub_fallback_emitted` clause)."""
    client, _ = await _drive_and_record(["<tool_call>\n<function=execute>\n</function>\n</tool_call>"], tools=[{"name": "execute"}, {"name": "x"}])
    assert client.startswith("I prepared a tool call"), client
    assert UNPARSED_TOOL_CALL_NOTE not in client


@pytest.mark.asyncio
async def test_executed_the_watchdog_marker_gets_no_note_and_stays_in_the_record():
    """The cognitive watchdog's synthetic replan call is appended to the
    durable text on purpose (replayed by the trajectory machinery)."""
    deltas = ["Here is the analysis so far.\n\n", "<tool_call>\n<function=replan>\n<parameter=reason>\nloop\n</parameter>\n</function>\n</tool_call>", "\n\nRecovering."]
    client, recorded = await _drive_and_record(deltas, tools=[{"name": "a"}, {"name": "b"}])
    assert UNPARSED_TOOL_CALL_NOTE not in client, client
    assert "replan" in recorded, recorded


@pytest.mark.asyncio
async def test_executed_a_tool_response_echo_gets_no_note():
    deltas = ["Result:\n\n", "<tool_response>\nOK, 3 rows updated\n</tool_response>", "\n\nAll good."]
    client, recorded = await _drive_and_record(deltas, tools=[{"name": "a"}, {"name": "b"}])
    assert UNPARSED_TOOL_CALL_NOTE not in client and UNPARSED_TOOL_CALL_NOTE not in recorded


@pytest.mark.asyncio
async def test_executed_inline_code_about_the_syntax_is_kept_in_the_record():
    deltas = ["The parser expects `<tool_call>` to wrap `<function=execute>` calls.\n\n", "That is the whole format."]
    client, recorded = await _drive_and_record(deltas, tools=[{"name": "a"}, {"name": "b"}])
    assert "The parser expects `<tool_call>`" in recorded, recorded
    assert UNPARSED_TOOL_CALL_NOTE not in recorded and UNPARSED_TOOL_CALL_NOTE not in client


def test_the_streamed_verifier_judges_what_the_user_saw():
    """The live scrub removed unparsed tool XML from the stream; the verifier
    claim must be built from that scrubbed view, not the raw markup — a
    confident REFUTE on text the user never received queues a visible
    correction banner (review, 2026-09-09). Source pin: the claim builder is
    deep in the stream's late phase and no fixture reaches the verifier."""
    src = _stream_src()
    i = src.index("_sv_claim = re.sub(")
    window = src[max(0, i - 700):i + 300]
    assert "_stream_scrub_pattern.sub('', full_content)" in window, "the verifier claim is built from raw markup"
    assert "if _stream_scrub_active else full_content" in window
