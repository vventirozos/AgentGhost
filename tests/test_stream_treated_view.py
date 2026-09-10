"""One treated view of a streamed reply, read by every consumer (§4FV, 2026-09-10).

`_finalize_and_return` scrubs and smooths `final_ai_content` IN PLACE, so on
the non-streaming path every consumer after it reads the delivered text by
construction. The streamed drain had no such variable: §4FS treated only the
trajectory copy and handed the rest of the drain the raw accumulator. Seven
readers on the COMMON (web-UI) path were therefore learning from, or judging,
narration the user never received and tool markup the live scrub had already
removed from the stream — the hedge scan, the smart-memory arc, the
post-mortem, the episode, the hydration judge, the project work_log and the
calibration sample. §4FT recorded six of them and fixed none.

These pins drive the real generator and read what each consumer was handed.
The negative case (a zero-tool turn) is what makes them discriminating: the
same assertions must FAIL to hold when smoothing is not supposed to run.
"""
import ast
import inspect
import os
import sys

import pytest
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ghost_agent.core import agent as agent_mod
from ghost_agent.core.agent import StreamState
from ghost_agent.core.reply_smoothing import UNPARSED_TOOL_CALL_NOTE

from tests.test_finalize_stream_pins import make_stream_agent, sse


NARRATED = ["Let me check the file first.\n\n",
            "The file has 3 lines and no trailing newline."]
BEAT = "Let me check the file first."
ANSWER = "The file has 3 lines and no trailing newline."
TWO_TOOLS = [{"name": "file_system"}, {"name": "execute"}]


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
        stream_verify_messages=[], was_complex_task=True,
        _active_turn=None, _proj_task_closed_this_req=False, _turn_reg=reg)


class Consumers(dict):
    """What each drain consumer was handed, by name."""


async def _drive(deltas, tools, *, monkeypatch=None):
    """Run the real stream generator with every drain consumer captured."""
    a = make_stream_agent()
    a.context.args.smart_memory = 0.5
    a.context.args.no_verifier = False
    a.context.journal = MagicMock()
    a.context.metacog = MagicMock()
    a.context.metacog.enabled = True
    a.context.verifier = MagicMock()
    utk = MagicMock()
    utk.scan_text_for_uncertainty = MagicMock(return_value=[])
    utk.pressure.return_value = 0.0
    a.context.uncertainty_tracker = utk

    a._journal_append_safe = AsyncMock()
    a._record_episode_safe = AsyncMock()
    a._judge_hydration_safe = MagicMock()
    a._write_project_work_log_safe = AsyncMock()
    a._record_calibration_safe = AsyncMock()
    a._record_turn_trajectory = MagicMock()
    a._attach_late_verdict_handler = MagicMock()

    verdict_claim = {}

    async def _fake_verdict(**kw):
        verdict_claim.update(kw)
        return None
    a._compute_verifier_verdict = _fake_verdict

    backstop = MagicMock(return_value=False)
    if monkeypatch is not None:
        monkeypatch.setattr(agent_mod, "_notify_promise_backstop", backstop)
        monkeypatch.setattr(agent_mod, "_user_asked_for_notification",
                            lambda *_a, **_k: True)

    async def final_stream(payload, use_coding=False):
        for d in deltas:
            yield sse({"content": d})
        yield b"data: [DONE]\n\n"
    a.context.llm_client.stream_chat_completion = final_stream

    reg = MagicMock()
    reg.is_cancelled.return_value = False
    gen, _, _ = a._stream_final_generation(_state(reg, tools))
    [c async for c in gen]
    # the verifier verdict is SPAWNED by the drain; let it start
    import asyncio
    await asyncio.sleep(0.02)

    journal = {c.args[0]: c.args[1] for c in a._journal_append_safe.call_args_list}
    seen = Consumers()
    if "smart_memory" in journal:
        seen["smart_memory"] = journal["smart_memory"]["text"]
    if "post_mortem" in journal:
        seen["post_mortem"] = journal["post_mortem"]["ai"]
    if a._record_episode_safe.called:
        seen["episode"] = a._record_episode_safe.call_args.args[2]
    if a._judge_hydration_safe.called:
        seen["hydration_judge"] = a._judge_hydration_safe.call_args.args[0]
    if a._write_project_work_log_safe.called:
        seen["work_log"] = a._write_project_work_log_safe.call_args.kwargs[
            "final_ai_content"]
    if a._record_calibration_safe.called:
        seen["calibration"] = a._record_calibration_safe.call_args.kwargs[
            "final_ai_content"]
    if a._record_turn_trajectory.called:
        seen["trajectory"] = a._record_turn_trajectory.call_args.kwargs[
            "final_content"]
    if utk.scan_text_for_uncertainty.called:
        seen["hedge_scan"] = utk.scan_text_for_uncertainty.call_args.args[0]
    if backstop.called:
        seen["notify_backstop"] = backstop.call_args.kwargs["final_content"]
    return seen, verdict_claim


# The consumers §4FT listed, plus the hedge scan it missed. Named here so a
# harness that silently stops driving one is a failure, not a green run.
EXPECTED = {"smart_memory", "post_mortem", "episode", "hydration_judge",
            "work_log", "calibration", "trajectory", "hedge_scan"}


@pytest.mark.asyncio
async def test_every_drain_consumer_is_actually_driven():
    seen, _ = await _drive(NARRATED, TWO_TOOLS)
    assert EXPECTED <= set(seen), EXPECTED - set(seen)


@pytest.mark.asyncio
async def test_every_consumer_reads_the_smoothed_reply():
    """World where this fails: any one consumer keeps `full_content` — the
    state §4FT recorded and did not fix."""
    seen, _ = await _drive(NARRATED, TWO_TOOLS)
    for name in EXPECTED:
        assert BEAT not in seen[name], f"{name} still reads the raw reply"
        assert ANSWER in seen[name], f"{name} lost the answer: {seen[name]!r}"


@pytest.mark.asyncio
async def test_a_zero_tool_turn_hands_every_consumer_the_reply_verbatim():
    """The discriminating negative: conversational replies are never
    rewritten, so the assertions above cannot be passing on a constant."""
    seen, _ = await _drive(NARRATED, [])
    for name in EXPECTED:
        assert BEAT in seen[name], f"{name} smoothed a zero-tool reply"


@pytest.mark.asyncio
async def test_a_single_tool_turn_hands_every_consumer_the_reply_verbatim():
    """The gate is ≥2 real tools (SMOOTHING_MIN_TOOLS) on this path as on
    the other — the 2026-07-17 decision, re-affirmed by §4FT."""
    seen, _ = await _drive(NARRATED, [{"name": "file_system"}])
    for name in EXPECTED:
        assert BEAT in seen[name], f"{name} smoothed a single-tool reply"


@pytest.mark.asyncio
async def test_a_synthetic_tool_does_not_open_the_gate_for_the_consumers():
    seen, _ = await _drive(
        NARRATED, [{"name": "a", "_synthetic": True},
                   {"name": "b", "_synthetic": True}])
    for name in EXPECTED:
        assert BEAT in seen[name], f"{name} counted a synthetic tool"


@pytest.mark.asyncio
async def test_every_consumer_reads_the_scrubbed_reply_and_the_note():
    """Unparsed markup: no consumer records tag soup, and each one carries
    the same note the user got — including on a single-tool turn, where the
    scrub is unconditional and the smoother does not run."""
    deltas = ["Saving it now.\n\n", "<tool_call>", "<function=file_system>",
              "</function>", "</tool_call>", "\n\nSaved."]
    seen, _ = await _drive(deltas, [{"name": "file_system"}])
    for name in EXPECTED:
        text = seen[name]
        assert "<tool_call>" not in text and "<function=" not in text, name
        assert UNPARSED_TOOL_CALL_NOTE in text, f"{name} lost the note"


@pytest.mark.asyncio
async def test_the_promise_backstop_summarises_the_delivered_answer(monkeypatch):
    """Parity with finalize, which has handed it the treated text since
    2026-07-17. The promise itself is detected from the USER's request, so
    trimming the model's narration cannot lose one."""
    seen, _ = await _drive(NARRATED, TWO_TOOLS, monkeypatch=monkeypatch)
    assert "notify_backstop" in seen
    assert seen["notify_backstop"] == ANSWER, seen["notify_backstop"]


@pytest.mark.asyncio
async def test_the_monologue_still_reads_the_raw_text():
    """The one reader that must NOT take the treated view: <think> blocks
    live inside the paragraphs the smoother drops, and the web UI's
    monologue box parses them off the log stream."""
    deltas = ["Let me check the file first.<think>counting the rows</think>\n\n",
              ANSWER]
    import logging
    got = []

    class _H(logging.Handler):
        def emit(self, record):
            msg = record.getMessage()
            if "PLANNER MONOLOGUE" in msg:
                got.append(msg)
    handler = _H()
    lg = logging.getLogger("GhostAgent")
    prev = lg.level
    lg.setLevel(logging.INFO)
    lg.addHandler(handler)
    try:
        seen, _ = await _drive(deltas, TWO_TOOLS)
    finally:
        lg.removeHandler(handler)
        lg.setLevel(prev)
    assert any("counting the rows" in m for m in got), got
    # …and the record itself dropped that paragraph, so the two really differ.
    assert "counting the rows" not in seen["trajectory"]


@pytest.mark.asyncio
async def test_the_verifier_judges_the_unsmoothed_delivered_text():
    """The deliberate exception (§4FV): the live stream showed the
    narration, so the verdict is computed on the scrubbed-but-unsmoothed
    view. A migration that pointed it at the treated view would judge text
    the user never saw."""
    _, claim = await _drive(NARRATED, TWO_TOOLS)
    assert claim, "the stream verifier gate did not run"
    assert BEAT in claim["final_ai_content"], claim["final_ai_content"]


def test_no_new_raw_reader_slips_into_the_drain():
    """The class, not the site: every `full_content` read in the drain must
    be one of the four the design keeps raw. A consumer added later that
    reaches for the accumulator fails here rather than in six months of
    quietly wrong records."""
    src = inspect.getsource(agent_mod.GhostAgent._stream_final_generation)
    tree = ast.parse("if 1:\n" + src)
    lines = ("if 1:\n" + src).split("\n")
    marker = next(i for i, l in enumerate(lines, 1)
                  if "THE TREATED VIEW OF THIS REPLY" in l)
    allowed = {
        # the treated view's own base, and its fail-open
        '"_stream_effective_content", full_content) or full_content',
        '_treated_content = full_content',
        # the monologue's <think> source (see the pin above)
        "think_matches = re.findall(r'<think>(.*?)(?:</think>|$)', full_content, flags=re.DOTALL | re.IGNORECASE)",
        # the promise headline's fallback
        '_treated_content or full_content,',
        # the verifier claim: what the user saw, unsmoothed (pin above)
        "_sv_source = (_stream_scrub_pattern.sub('', full_content)",
        'if _stream_scrub_active else full_content)',
    }
    found = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Name) and node.id == "full_content"
                and node.lineno > marker):
            found.add(lines[node.lineno - 1].strip())
    assert found <= allowed, f"new raw reader(s) in the drain: {found - allowed}"
    assert found, "the enumeration found nothing — the marker moved"


@pytest.mark.asyncio
async def test_an_all_consumed_scrub_records_the_fallback_not_tag_soup():
    """The treated view is computed from the EFFECTIVE content: when the
    scrub ate the whole reply, what the user read is the fallback sentence,
    and that is what every consumer must store (§4FS's "never tag-soup",
    now shared by all eight)."""
    deltas = ["<tool_call>\n<function=execute>\n</function>\n</tool_call>"]
    seen, _ = await _drive(deltas, TWO_TOOLS)
    for name in EXPECTED:
        assert "I prepared a tool call" in seen[name], (name, seen[name])
        assert "<function=" not in seen[name], name


def test_treat_reply_keeps_the_inverted_trim_guard():
    """2026-07-25 live: smoothing kept "Let me search more specifically…"
    and dropped the findings. `treat_reply` must revert to the untrimmed
    text when the trim leaves narration only."""
    from ghost_agent.core.reply_smoothing import treat_reply
    reply = ("Let me search more specifically for the changelog.\n\n"
             "Now let me try the archive.")
    out = treat_reply(reply, n_real_tools=4)
    assert out == reply, out


@pytest.mark.asyncio
async def test_the_treatment_says_so_on_the_operator_stream(monkeypatch):
    """A treatment that only ever runs silently cannot be told apart from
    one that never runs (§4FS's own lesson, applied to §4FV). The line
    fires only when the text actually changed."""
    lines = []
    monkeypatch.setattr(agent_mod, "pretty_log",
                        lambda title, msg, **k: lines.append((title, msg)))
    await _drive(NARRATED, TWO_TOOLS)
    assert any(t == "Reply Smoothing" and "streamed record treated" in m
               for t, m in lines), [t for t, _ in lines]
    lines.clear()
    await _drive(NARRATED, [])
    assert not any(t == "Reply Smoothing" for t, m in lines), (
        "an untouched reply announced a treatment")
