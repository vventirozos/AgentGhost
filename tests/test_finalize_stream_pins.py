"""§ finalize/stream slice R1 (2026-08-19, lens C sweep): driven behavioral pins for the SURVIVED mutants of the finalize/stream
mutation sweep. Each test passes on pristine source and fails on exactly the
mutant named in its docstring. No source-text inspection anywhere — every
assertion drives the code and checks an observable."""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core.agent import (
    FinalizeState, GhostAgent, GhostContext, StreamState)


# ---------------------------------------------------------------- helpers --

class FakeScratchpad:
    def __init__(self):
        self.d = {}
    def get(self, k, *a):
        return self.d.get(k, "")
    def set(self, k, v):
        self.d[k] = v
    def list_all(self):
        return ""


def make_fin_agent():
    ctx = MagicMock()
    ctx.llm_client.chat_completion = AsyncMock()
    ctx.args.smart_memory = 0.0
    ctx.args.perfect_it = False
    agent = GhostAgent(ctx)
    return agent


def _fs(**over):
    fields = dict(
        body={"messages": [], "stream": False},
        created_time=1234567890,
        current_plan_json={},
        current_trajectory_id="traj-1",
        execution_failure_count=0,
        final_ai_content="All done.",
        force_stop=False,
        forget_was_called=False,
        last_user_content="do the thing",
        last_was_failure=False,
        lc="do the thing",
        messages=[{"role": "user", "content": "do the thing"}],
        model="test-model",
        payload={"messages": [], "model": "test-model"},
        req_id="ab12cd34",
        thought_content="",
        tools_run_this_turn=[],
        wakeup_prefix="",
        was_complex_task=False,
        _stable_conv_fp="fp",
        _verdict_is_fresh=False,
        _verifier_verdict_cache=None,
    )
    fields.update(over)
    return FinalizeState(**fields)


def make_stream_agent():
    context = MagicMock(spec=GhostContext)
    context.args = MagicMock()
    context.args.use_planning = False
    context.args.smart_memory = 0.0
    context.args.max_context = 4000
    context.args.temperature = 0.7
    context.llm_client = MagicMock()
    context.sandbox_dir = None
    context.memory_system = None
    context.profile_memory = None
    context.semantic_memory = None
    context.skill_memory = None
    context.journal = None
    context.scratchpad = MagicMock()
    context.scratchpad.list_all.return_value = ""
    return GhostAgent(context=context)


def sse(delta):
    return f"data: {json.dumps({'choices': [{'delta': delta}]})}\n\n".encode()


class StreamScript:
    """Serves `first` on call 1 and a clean answer on later calls; records
    per-call whether the agent drained the stream to completion."""
    def __init__(self, first):
        self.first = first
        self.calls = 0
        self.drained = []
        self.events = []
    def __call__(self, *a, **k):
        self.calls += 1
        idx = self.calls
        self.drained.append(False)
        script = self
        async def gen():
            first_chunk = True
            chunks = (script.first if idx == 1
                      else [sse({"content": "CLEAN_ANSWER_AFTER_RETRY."})])
            for c in chunks:
                if first_chunk:
                    script.events.append("stream")
                    first_chunk = False
                yield c
            yield b"data: [DONE]\n\n"
            script.drained[idx - 1] = True
        return gen()


async def run_chat(agent, script, user="do the task"):
    agent.context.llm_client.stream_chat_completion = script
    body = {"messages": [{"role": "user", "content": user}], "stream": False}
    with patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[]):
        out = await agent.handle_chat(
            body, background_tasks=MagicMock(), request_id="req-driven")
    return out


# ------------------------------------------------------- finalize survivors --

async def test_F1_empty_reply_falls_back_to_tool_output():
    """F1: tools ran, model said nothing -> reply must carry the tool output,
    not the generic banner."""
    a = make_fin_agent()
    fs = _fs(final_ai_content="", tools_run_this_turn=[
        {"name": "execute", "content": "STDOUT/STDERR:\nUNIQUE_EXEC_OUTPUT_77"}])
    out, _, _ = await a._finalize_and_return(fs)
    assert "UNIQUE_EXEC_OUTPUT_77" in out


async def test_F2_no_tools_no_content_still_says_something():
    """F2: nothing ran and the model said nothing -> canned success line,
    never an empty reply."""
    out, _, _ = await make_fin_agent()._finalize_and_return(
        _fs(final_ai_content=""))
    assert "Task executed successfully." in out


async def test_F3_fallback_shows_the_action_not_the_trailing_listing():
    """F3: ACTION view first — a trailing bookkeeping listing must not shadow
    the execute output in the empty-reply fallback."""
    a = make_fin_agent()
    listing = ("PROJECT_LISTING_ROW | id=1 status=open owner=me\n" * 8)
    fs = _fs(final_ai_content="", tools_run_this_turn=[
        {"name": "execute", "content": "STDOUT/STDERR:\nREAL_ACTION_OUTPUT_ZZZ"},
        {"name": "manage_projects", "content": listing}])
    out, _, _ = await a._finalize_and_return(fs)
    assert "REAL_ACTION_OUTPUT_ZZZ" in out


async def test_F4_detached_run_is_not_reported_finished():
    """F4: a promoted (still-running) execute result must produce the STILL
    RUNNING head, never 'Process finished successfully.'"""
    from ghost_agent.sandbox.jobs import PROMOTED_RESULT_BANNER
    a = make_fin_agent()
    fs = _fs(final_ai_content="", tools_run_this_turn=[
        {"name": "execute",
         "content": PROMOTED_RESULT_BANNER + "\njob detached, no result yet"}])
    out, _, _ = await a._finalize_and_return(fs)
    assert "STILL RUNNING" in out
    assert "Process finished successfully." not in out


async def test_F5_reads_do_not_count_as_writes_for_promotion_nudge():
    """F5: three file READS in a long chat must NOT fire the promote-to-
    project footer (the >=3-writes retune)."""
    a = make_fin_agent()
    a.context.scratchpad = FakeScratchpad()
    a.context.current_project_id = None
    msgs = []
    for i in range(9):
        msgs.append({"role": "user",
                     "content": f"please analyse dataset part {i} carefully"})
        msgs.append({"role": "assistant", "content": f"done with part {i}"})
    reads = [{"name": "file_system",
              "content": f"SUCCESS: Listed directory contents of /app/part{i} "
                         f"(12 entries)"} for i in range(3)]
    fs = _fs(final_ai_content="A fine answer.", messages=msgs,
             tools_run_this_turn=reads)
    out, _, _ = await a._finalize_and_return(fs)
    assert "tracked project" not in out


async def test_F6_recovered_strikes_are_named_in_turn_outcome():
    """F6: a recovered strike must be named in the Turn Outcome line."""
    calls = []
    a = make_fin_agent()
    with patch("ghost_agent.core.agent.pretty_log",
               side_effect=lambda *ar, **kw: calls.append((ar, kw))):
        fs = _fs(final_ai_content="ok answer", execution_failure_count=1,
                 tools_run_this_turn=[
                     {"name": "execute", "content": "STDOUT/STDERR:\nfine"}])
        await a._finalize_and_return(fs)
    outcome_lines = [str(ar[1]) for ar, kw in calls
                     if ar and ar[0] == "Turn Outcome"]
    assert outcome_lines, "no Turn Outcome line at all"
    assert any(("recovered 1 strike" in t) or ("honestly reported" in t)
               for t in outcome_lines), outcome_lines


async def test_F7_consecutive_duplicate_paragraph_is_collapsed():
    """F7: 'Your name is X.\\n\\nYour name is X.' must dedupe to one."""
    out, _, _ = await make_fin_agent()._finalize_and_return(
        _fs(final_ai_content="Your name is X.\n\nYour name is X."))
    assert out.count("Your name is X.") == 1


async def test_F8_narration_only_trim_is_reverted():
    """F8: when smoothing reduces the reply to a bare narration line, the
    original (with the findings) must be delivered."""
    import ghost_agent.core.reply_smoothing as rs
    a = make_fin_agent()
    with patch.object(rs, "smooth_reply",
                      lambda s: "Let me search more specifically for it."):
        fs = _fs(
            final_ai_content=("REAL_FINDINGS_CONTENT: the port is 8080 and "
                              "the leak is in foo()."),
            tools_run_this_turn=[
                {"name": "execute", "content": "STDOUT/STDERR:\naa"},
                {"name": "execute", "content": "STDOUT/STDERR:\nbb"}])
        out, _, _ = await a._finalize_and_return(fs)
    assert "REAL_FINDINGS_CONTENT" in out


async def test_F9_start_with_constraint_hoists_the_opener():
    """F9: a 'Begin your reply with: X' constraint must hoist the X-led
    segment of the assembled reply to the head."""
    a = make_fin_agent()
    a._active_project_constraints = lambda: []
    fs = _fs(
        final_ai_content=("Some earlier analysis narration paragraph.\n\n"
                          "FINAL VERDICT: the parser is correct and the fix "
                          "holds across all twelve cases in the suite."),
        last_user_content="Begin your reply with: FINAL VERDICT")
    out, _, _ = await a._finalize_and_return(fs)
    assert out.startswith("FINAL VERDICT")


# ---------------------------------------------------- stream-loop survivors --

async def test_S1_ngram_loop_aborts_the_stream():
    """S1: a repeating reasoning stream must be aborted — its post-loop
    content must never survive into the reply."""
    a = make_stream_agent()
    para = "We must recheck the vowels count function once more. "
    chunks = [sse({"reasoning_content": para * 3}) for _ in range(12)]
    chunks.append(sse({"content": "LOOP_RUN_COMPLETED_TEXT"}))
    script = StreamScript(chunks)
    out = await run_chat(a, script)
    assert "LOOP_RUN_COMPLETED_TEXT" not in str(out)
    assert script.drained[0] is False, "looping stream was drained to the end"


async def test_S2_paragraph_loop_aborts_the_stream():
    """S2: verbatim paragraph repeats with varied filler must be aborted."""
    a = make_stream_agent()
    rep = ("I will keep the same general structure and update the "
           "session-related parts now.")
    lines = []
    for i in range(70):
        lines.append(rep)
        lines.append(f"unique filler {i} noting detail {i * 13} "
                     f"and value {i * 7} for this pass")
    buf = "\n".join(lines) + "\n"
    step = 400
    chunks = [sse({"reasoning_content": buf[i:i + step]})
              for i in range(0, len(buf), step)]
    chunks.append(sse({"content": "PARA_RUN_COMPLETED_TEXT"}))
    script = StreamScript(chunks)
    out = await run_chat(a, script)
    assert "PARA_RUN_COMPLETED_TEXT" not in str(out)
    assert script.drained[0] is False


async def test_S3_tool_call_spam_aborts_the_stream():
    """S3: dozens of unclosed <tool_call> opens must abort the stream."""
    a = make_stream_agent()
    chunks = [sse({"content": "<tool_call>" * 4}) for _ in range(15)]
    chunks.append(sse({"content": "TOOLSPAM_RUN_COMPLETED_TEXT"}))
    script = StreamScript(chunks)
    await run_chat(a, script)
    assert script.drained[0] is False, "tool-call spam stream drained fully"


async def test_S4_extended_cap_aborts_a_200k_stream():
    """S4: diverse (non-looping) thinking past the extended cap must abort."""
    a = make_stream_agent()
    chunks = []
    for i in range(300):
        seg = f"segment {i}: " + " ".join(
            f"w{i}x{j}" for j in range(120)) + ". "
        chunks.append(sse({"reasoning_content": seg}))
    chunks.append(sse({"content": "CAP_RUN_COMPLETED_TEXT"}))
    script = StreamScript(chunks)
    out = await run_chat(a, script)
    assert "CAP_RUN_COMPLETED_TEXT" not in str(out)
    assert script.drained[0] is False


async def test_S5_S9_upstream_abort_triggers_continuation():
    """S5 + S9: an upstream error frame mid-answer must trigger the
    truncation continuation, which completes the reply."""
    a = make_stream_agent()
    a.context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content":
                     " and CONTINUATION_MARKER_XYZ finishes the list."},
                     "finish_reason": "stop"}]})
    chunks = [
        sse({"content": "The partial answer lists the first three ports"}),
        b'data: ' + json.dumps({"error": "connection reset"}).encode() + b'\n\n',
    ]
    script = StreamScript(chunks)
    out = await run_chat(a, script)
    assert "CONTINUATION_MARKER_XYZ" in str(out)


def _make_stream_state(reg):
    return StreamState(
        created_time=123, current_trajectory_id="t1",
        execution_failure_count=0, fname="chat", force_stop=False,
        forget_was_called=False, has_coding_intent=False,
        is_final_generation=True, last_user_content="q",
        last_was_failure=False, lc="q",
        messages=[{"role": "user", "content": "q"}], model="m",
        payload={"messages": []}, req_id="rid00001",
        stream_conv_fp="fp", stream_messages_snapshot=[], stream_model="m",
        stream_prefix="", stream_thought="", stream_tools_snapshot=[],
        stream_verify_messages=[], was_complex_task=False,
        _active_turn=None, _proj_task_closed_this_req=False, _turn_reg=reg)


async def _drive_final_stream(a):
    async def final_stream(payload, use_coding=False):
        yield sse({"content": "A cut-off final answer"})
        yield b'data: {"error": "upstream reset"}\n\n'
        yield b"data: [DONE]\n\n"
    a.context.llm_client.stream_chat_completion = final_stream
    reg = MagicMock()
    reg.is_cancelled.return_value = False
    gen, _, _ = a._stream_final_generation(_make_stream_state(reg))
    async for _ in gen:
        pass


async def test_S7_final_stream_abort_marks_calibration_truncated():
    """S7: the final-stream abort latch must reach the calibration record."""
    a = make_stream_agent()
    a._record_calibration_safe = AsyncMock()
    await _drive_final_stream(a)
    assert a._record_calibration_safe.called
    assert a._record_calibration_safe.call_args.kwargs["truncated"] is True


async def test_S8_final_stream_abort_marks_durable_content():
    """S8: the durable content of an aborted final stream must carry the
    truncation note."""
    a = make_stream_agent()
    a._record_calibration_safe = AsyncMock()
    await _drive_final_stream(a)
    durable = a._record_calibration_safe.call_args.kwargs["final_ai_content"]
    assert "RESPONSE TRUNCATED" in durable


async def test_S10_uncertainty_tracker_reset_before_first_stream():
    """S10: turn-start must reset the uncertainty tracker BEFORE the first
    generation, so an aborted prior turn cannot leak state into this one."""
    a = make_stream_agent()
    script = StreamScript([sse({"content": "Fine short answer."})])
    ut = MagicMock()
    ut.reset.side_effect = lambda: script.events.append("reset")
    a.context.uncertainty_tracker = ut
    await run_chat(a, script)
    assert "reset" in script.events, "tracker never reset"
    assert "stream" in script.events
    assert script.events.index("reset") < script.events.index("stream")
