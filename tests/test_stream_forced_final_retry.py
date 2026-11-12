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


def _state(reg, tools, prefix="", task_closed=False, session_id="", ff_breaker=None, ff_report_write_used=False,
           ff_breaker_reason=""):
    return StreamState(session_id=session_id,
        ff_breaker=ff_breaker, ff_report_write_used=ff_report_write_used, ff_breaker_reason=ff_breaker_reason,   # §4OG-4 (g)
        created_time=123, current_trajectory_id="t1",
        execution_failure_count=0, force_stop=False,
        forget_was_called=False, has_coding_intent=False,
        is_final_generation=True, last_user_content="q",
        last_was_failure=False, lc="q",
        messages=[{"role": "user", "content": "q"}], model="m",
        payload={"messages": [{"role": "user", "content": "investigate"}],
                 "tools": [{"type": "function", "function": {"name": "web_search"}}]},
        req_id="rid00001",
        stream_conv_fp="fp", stream_messages_snapshot=[], stream_model="m",
        stream_prefix=prefix, stream_tools_snapshot=list(tools),      # a copy: the §4OE-1b write appends to it
        stream_verify_messages=[], was_complex_task=True,
        _active_turn=None, _proj_task_closed_this_req=task_closed, _turn_reg=reg)


async def _drive(deltas, *, retry_reply, prefix="", tools=TOOLS, task_closed=False, fs=None, breaker=False,
                 session_id="", scratchpad=None, pending=None):
    a = make_stream_agent()
    if pending is not None:
        a.context._project_work_pending = pending
    if fs is not None:
        a.available_tools = {"file_system": fs}
    if breaker:
        a.context._breaker_forced_final = True                              # §4OE-1b: a breaker-forced final
        a.context._breaker_reason = "the turn budget"
        a.context._report_write_used = False
    if scratchpad is not None:
        a.context.scratchpad = scratchpad
    a.context.args.no_verifier = True
    a.context.journal = MagicMock()
    a.context.metacog = MagicMock(); a.context.metacog.enabled = False
    a._journal_append_safe = AsyncMock()
    a._record_episode_safe = AsyncMock()
    a._judge_hydration_safe = MagicMock()
    a._write_project_work_log_safe = AsyncMock()
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
        if callable(retry_reply):
            return await retry_reply(payload, *args, **kw)
        return {"choices": [{"message": {"content": retry_reply}}]}
    a.context.llm_client.chat_completion = AsyncMock(side_effect=retry)

    reg = MagicMock(); reg.is_cancelled.return_value = False
    st = _state(reg, tools, prefix, task_closed, session_id,                      # §4OG-4 (g): the snapshot production takes
                ff_breaker=getattr(a.context, "_breaker_forced_final", False) is True,
                ff_report_write_used=bool(getattr(a.context, "_report_write_used", False)),
                ff_breaker_reason=str(getattr(a.context, "_breaker_reason", "") or ""))
    _drive.last_state = st
    gen, _, _ = a._stream_final_generation(st)
    chunks = [c async for c in gen]
    client = "".join(
        (json.loads(c.decode()[6:]).get("choices") or [{}])[0].get("delta", {}).get("content") or ""
        for c in chunks if c.startswith(b"data: ") and c.strip() != b"data: [DONE]")
    durable = (a._record_turn_trajectory.call_args.kwargs["final_content"]
               if a._record_turn_trajectory.called else "")
    _drive.last_agent = a
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
    # §4KW: the builder — the constant, whose message it is, which request,
    # and (here the model tried a tool) that the call did not run
    last = p["messages"][-1]
    assert last["role"] == "user" and last["content"].startswith(_FORCED_FINAL_ANSWER_DIRECTIVE)
    assert "were NOT run" in last["content"] and "THE REQUEST YOU ARE ANSWERING" in last["content"]
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
    for name in ("_journal_append_safe", "_record_episode_safe", "_write_project_work_log_safe"):
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
    assert any(UNPARSED_TOOL_CALL_NOTE.encode() in c or b"NOT executed" in c for c in chunks)


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
async def test_a_task_closed_scrub_still_ships_the_task_complete_sentence_without_a_retry(monkeypatch):
    from ghost_agent.core import agent as _A
    monkeypatch.setattr(_A, "_CLOSE_ENDS_TURN", True)      # §4UI: dormant, pinned with the switch on
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
    for name in ("_journal_append_safe", "_record_episode_safe", "_write_project_work_log_safe"):
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



# ── §4JP (2026-09-21): the §4IG rule on the stream path ──────────────────

_DEADLINE_LIVE_TURN = ("I've found the founding years for three of the four newspapers. I still need "
                       "to search for Eleftherotypia's founding year to complete the summary."
                       "\n\n<tool_call>\n<function=web_search>\n<parameter=query>Eleftherotypia founding"
                       "</parameter>\n</function>\n</tool_call>")
_DEADLINE_REPORT = ("Report: Naftemporiki 1924, Kathimerini 1919, To Vima 1922 (found); Eleftherotypia "
                    "not searched — the client deadline stopped the work.")


@pytest.mark.asyncio
async def test_on_a_breaker_forced_final_a_tool_call_is_the_no_answer_on_the_stream_too():
    """The first live client-deadline report (probe a3b3b65c): the report
    turn streamed an assessment + "I still need to search…" + a search call;
    neither trigger saw it (the sentence is not a beat) and the client got
    the narration with a dropped-call note. FAILS IF: the stream path ignores
    `_breaker_forced_final` — the §4IG rule the internal path has had."""
    a_flag = {"set": True}
    orig = make_stream_agent

    def _agent_with_flag():
        a = orig()
        a.context._breaker_forced_final = True
        return a
    globals()["make_stream_agent"] = _agent_with_flag
    try:
        client, durable, retries = await _drive([_DEADLINE_LIVE_TURN], retry_reply=_DEADLINE_REPORT)
    finally:
        globals()["make_stream_agent"] = orig
    assert len(retries) == 1
    assert _DEADLINE_REPORT in client and _DEADLINE_REPORT in durable
    assert "I still need to search" not in client.split(_DEADLINE_REPORT)[-1]


@pytest.mark.asyncio
async def test_without_the_breaker_flag_an_answer_plus_a_call_still_ships_unretried():
    """The §4HF contract stands where no breaker closed the loop: a final that
    ANSWERS and then tries one more call has answered."""
    orig = make_stream_agent

    def _agent_unarmed():
        a = orig()
        a.context._breaker_forced_final = False
        return a
    globals()["make_stream_agent"] = _agent_unarmed
    try:
        client, _, retries = await _drive([_ANSWER + "\n\n<tool_call>\n<function=web_search>\n"
                                           "<parameter=query>x</parameter>\n</function>\n</tool_call>"],
                                          retry_reply="SHOULD NOT APPEAR")
    finally:
        globals()["make_stream_agent"] = orig
    assert retries == [] and "SHOULD NOT APPEAR" not in client


# the dialect the prompt TEACHES (`name="…"`, CDATA content) — a regex on the
# `=` dialect never fired live (§4OE-1b review r2 CRIT 1)
_FILE_REPORT_TURN = ('<tool_call>\n<function name="file_system">\n<parameter name="operation">write</parameter>\n'
                     '<parameter name="path">report.md</parameter>\n<parameter name="content"><![CDATA[# Report\n\n'
                     'four servers]]></parameter>\n</function>\n</tool_call>')
_FILE_REPORT_TURN_EQ = ("<tool_call>\n<function=file_system>\n<parameter=operation>write</parameter>\n"
                        "<parameter=path>report.md</parameter>\n<parameter=content># Report\n\nfour servers"
                        "</parameter>\n</function>\n</tool_call>")


@pytest.mark.asyncio
async def test_a_streamed_report_written_as_a_file_is_told_it_was_not_written_when_it_did_not_run():
    """§4OE-1b: a planner final (no breaker) or a write that did not land —
    the retry directive says the file was NOT written and asks for the
    report as text; a search call keeps the plain directive."""
    _, _, retries = await _drive([_FILE_REPORT_TURN], retry_reply=_ANSWER)
    assert len(retries) == 1 and "NOT written" in retries[0]["messages"][-1]["content"]
    _, _, retries = await _drive([_LIVE_TURN], retry_reply=_ANSWER, prefix=_PREFIX)
    assert "NOT written" not in retries[0]["messages"][-1]["content"]
    refused = AsyncMock(return_value="REJECTED: 'report.md' already exists (900 bytes) and this request has not read it. Nothing was changed.")
    _, _, retries = await _drive([_FILE_REPORT_TURN], retry_reply=_ANSWER, fs=refused, breaker=True)
    assert refused.await_count == 1 and "NOT written" in retries[0]["messages"][-1]["content"]
    assert "(report written)" not in retries[0]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_a_streamed_breaker_final_runs_the_report_write_and_the_retry_links_it():
    """§4OE-1b (review MAJOR 4): the owner's web client streams — the report
    written as a file on a breaker-forced streamed final RUNS through the tool,
    and the retry is told the file exists and asked for its link."""
    for turn in (_FILE_REPORT_TURN, _FILE_REPORT_TURN_EQ):                # both dialects the parser heals
        fs = AsyncMock(return_value="SUCCESS: Wrote 20 chars to 'report.md'. Script-side path (from sandbox cwd): 'report.md'.")
        client, durable, retries = await _drive([turn], retry_reply="The report is in report.md.", fs=fs, breaker=True)
        assert fs.await_count == 1 and fs.await_args.kwargs.get("path") == "report.md"
        assert "four servers" in fs.await_args.kwargs.get("content", "")
        d = retries[0]["messages"][-1]["content"]
        assert "(report written)" in d and "/api/download/report.md" in d and "NOT written" not in d
        assert "<tool_response" in retries[0]["messages"][-2]["content"]   # the receipt precedes the ask
        assert "The report is in report.md." in client
    # the write's row reached the stream's tool snapshot (the ledger, the turn
    # outcome and the verifier read it — r2 MINOR 4)
    snap = _drive.last_state.stream_tools_snapshot
    assert any(isinstance(r, dict) and r.get("name") == "file_system" and "report.md" in str(r.get("content")) for r in snap)
    assert len(TOOLS) == 2                                                    # the module fixture is not mutated
    # the STATE OF WORK persisted before the generator ran is refreshed with the report (r2 MAJOR 3)
    from ghost_agent.memory.scratchpad import Scratchpad
    from ghost_agent.core import task_state as TS
    sp = Scratchpad()
    TS.persist_state_of_work(sp, "s-stream", "The previous turn of this conversation did not finish (x).\nIt wrote or edited: `notes.md`")
    fs4 = AsyncMock(return_value="SUCCESS: Wrote 20 chars to 'report.md'. Script-side path (from sandbox cwd): 'report.md'.")
    await _drive([_FILE_REPORT_TURN], retry_reply="The report is in report.md.", fs=fs4, breaker=True, session_id="s-stream", scratchpad=sp)
    assert "`report.md`" in TS.recall_state_of_work(sp, "s-stream") and "the turn budget" in TS.recall_state_of_work(sp, "s-stream")
    # prose before the call (partial scrub): the write ran, so no "NOT executed" caveat (r3 MAJOR 1)
    fs5 = AsyncMock(return_value="SUCCESS: Wrote 20 chars to 'report.md'. Script-side path (from sandbox cwd): 'report.md'.")
    client5, durable5, _ = await _drive(["Writing the final report to a file now.\n\n" + _FILE_REPORT_TURN],
                                        retry_reply="The report is in report.md.", fs=fs5, breaker=True)
    assert fs5.await_count == 1 and UNPARSED_TOOL_CALL_NOTE not in client5 and UNPARSED_TOOL_CALL_NOTE not in durable5
    # the project work-log snapshot (taken before the generator ran) gains the file (r3 MINOR 3)
    pend = ("rid00001", {"pid": "p1", "files": set()})
    await _drive([_FILE_REPORT_TURN], retry_reply="ok report.md", fs=fs5, breaker=True, pending=pend)
    assert pend[1]["files"] == {"report.md"}
    other = ("rid-other", {"pid": "p2", "files": set()})                     # another request owns the slot now
    await _drive([_FILE_REPORT_TURN], retry_reply="ok report.md", fs=fs5, breaker=True, pending=other)
    assert other[1]["files"] == set()
    # the retry itself dying after the write landed: the digest names the file (r4 MINOR 2)
    async def _boom(payload, *a, **k):
        raise RuntimeError("upstream gone")
    fs8 = AsyncMock(return_value="SUCCESS: Wrote 20 chars to 'report.md'. Script-side path (from sandbox cwd): 'report.md'.")
    client8, _, _ = await _drive([_FILE_REPORT_TURN], retry_reply=_boom, fs=fs8, breaker=True)
    assert fs8.await_count == 1 and "report.md" in client8 and "Scrub consumed" not in client8
    # a cut JSON-dialect write is told "NOT written" too (r4 MINOR 3)
    cut_json = '<tool_call>\n{"name": "file_system", "arguments": {"operation": "write", "path": "r.md", "content": "# Rep'
    fs9 = AsyncMock(return_value="SUCCESS: Wrote 20 chars to 'r.md'.")
    _, _, retries9 = await _drive([cut_json], retry_reply=_ANSWER, fs=fs9, breaker=True)
    assert fs9.await_count == 0 and "NOT written" in retries9[0]["messages"][-1]["content"]
    # a cut write is refused by the parser: nothing is written AND the retry is told so (r3 MINOR 4)
    fs6 = AsyncMock(return_value="SUCCESS: Wrote 20 chars to 'report.md'.")
    cut = _FILE_REPORT_TURN.split("four servers")[0] + "four ser"
    _, _, retries6 = await _drive([cut], retry_reply=_ANSWER, fs=fs6, breaker=True)
    assert fs6.await_count == 0 and "NOT written" in retries6[0]["messages"][-1]["content"]
    # the §4MB egress refusal applies to the streamed write too: nothing runs, the row is
    # synthetic (not a real tool run), the retry is told the file was NOT written
    import ghost_agent.memory.egress as _eg
    _orig = _eg.content_egress_refusal
    _eg.content_egress_refusal = lambda tool, args, ctx: "REFUSED: owner identifier after outside content"
    try:
        fs7 = AsyncMock(return_value="SUCCESS: Wrote 20 chars to 'report.md'.")
        _, _, retries7 = await _drive([_FILE_REPORT_TURN], retry_reply=_ANSWER, fs=fs7, breaker=True)
    finally:
        _eg.content_egress_refusal = _orig
    assert fs7.await_count == 0 and "NOT written" in retries7[0]["messages"][-1]["content"]
    rows7 = [r for r in _drive.last_state.stream_tools_snapshot if isinstance(r, dict) and r.get("name") == "file_system"]
    assert rows7 and rows7[0].get("_synthetic") is True and "REFUSED" in str(rows7[0]["content"])
    # the retry block failing BEFORE the write decision must not cost the stream (r5 MAJOR 1)
    import ghost_agent.core.reply_smoothing as _rs
    _orig_ucm = _rs.unparsed_call_markup_present
    def _raise(*a, **k):
        raise RuntimeError("early")
    _rs.unparsed_call_markup_present = _raise
    try:
        fs10 = AsyncMock(return_value="SUCCESS: Wrote 20 chars to 'report.md'.")
        client10, _, _ = await _drive([_FILE_REPORT_TURN], retry_reply=_ANSWER, fs=fs10, breaker=True)
    finally:
        _rs.unparsed_call_markup_present = _orig_ucm
    assert fs10.await_count == 0                                              # nothing decided, nothing ran; the stream ended
    # a search beside the write is work, not a report: nothing runs
    fs2 = AsyncMock(return_value="SUCCESS: Wrote 20 chars to 'report.md'.")
    mixed = _FILE_REPORT_TURN + "\n<tool_call>\n<function=web_search>\n<parameter=query>q</parameter>\n</function>\n</tool_call>"
    await _drive([mixed], retry_reply=_ANSWER, fs=fs2, breaker=True)
    assert fs2.await_count == 0
    # a planner final (no breaker armed) never runs the write either
    fs3 = AsyncMock(return_value="SUCCESS: Wrote 20 chars to 'report.md'.")
    _, _, retries = await _drive([_FILE_REPORT_TURN], retry_reply=_ANSWER, fs=fs3, breaker=False)
    assert fs3.await_count == 0 and "NOT written" in retries[0]["messages"][-1]["content"]
