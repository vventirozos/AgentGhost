"""The checkpoint answers that never left the streamed reply (§4HG, 2026-09-16).

§4HA taught the loop to record the risk-governor checkpoint answers and drop
them at finalize. Request 095beab8 (treatment arm, streamed to the web UI)
opened with "**CONFIRMED (observed):** … **ASSUMED (not yet confirmed):** …
The smallest distinguishing check: … One targeted search, then STOP." — the
drop lives in `_finalize_and_return`, which a streamed reply never reaches,
and the record kept the text too. The second answer of the same turn ("The
distinguishing check (…) has now run twice with no new agency name surfaced
… I have enough to deliver.") was not even recognised: it echoed the steer's
OTHER words. Each pin names the world it fails in.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core import experiments as EXP
from ghost_agent.core import risk as RISK
from ghost_agent.core.agent import GhostAgent
from ghost_agent.core.reply_smoothing import is_governor_checkpoint_answer
from ghost_agent.core.risk import STEER_DIRECTIVE_TERMS

from tests.test_finalize_stream_pins import sse

_CHECKPOINT = ("**CONFIRMED (observed):**\n- TechCrunch (12 Sep 2026): Revolut confirmed the "
               "request came from a legitimate government agency domain.\n\n"
               "**ASSUMED (not yet confirmed):**\n- Country = Italy, via PEC/InfoCert.\n\n"
               "The smallest distinguishing check: whether any source names the specific "
               "Italian government agency. One targeted search, then STOP.")
_STOP_ANSWER = ("The distinguishing check (does any source name the specific Italian government "
                "agency?) has now run twice with no new agency name surfaced, and the two primary "
                "Italian sources (pasqualepillitteri.it, cybernews.com) are blocked. I have enough "
                "to deliver. Writing the final forensic report.")
_REPORT = ("## Revolut Sept 2026 — Investigation Summary\n\n**Bottom line:** the sender address "
           "is not established; the visible address is Revolut's own receiving domain.")
_REQ = {"messages": [{"role": "user", "content":
                      "Write a python script that lists the workspace files, run it, and summarise the output."}],
        "model": "Qwen-Test", "stream": True}


# ── the shape test ────────────────────────────────────────────────────────

def test_the_stop_answer_is_now_a_checkpoint_answer():
    """FAILS IF: the markers still cover only "no new information / enough
    rounds / stopped producing" — the live STOP answer scored zero."""
    assert is_governor_checkpoint_answer(_STOP_ANSWER) is True


@pytest.mark.parametrize("pair", [
    # each new marker with exactly one old one, so each alternative is load-bearing
    "Confirmed: the notice exists. The distinguishing check is whether a source names the agency.",
    "Confirmed: the notice exists. The search surfaced no new leads.",
    "Confirmed: the notice exists. That search has now run twice.",
    "Confirmed: the notice exists. I have enough to deliver.",
])
def test_each_new_marker_is_load_bearing(pair):
    """FAILS IF: one of the four §4HG markers is lost."""
    assert is_governor_checkpoint_answer(pair) is True


@pytest.mark.parametrize("prose", [
    # ordinary content that uses one of the words: never two markers
    "Revolut confirmed it blocked the sender's address the same day.",
    "I ran the extraction twice on the same page and got the capped preview.",
    # …and, paired with a real marker, the bounded contexts must NOT match
    # ("no new customers" is not "no new information"; "the distinguishing
    # feature" is not "the distinguishing check")
    "Confirmed: no new customers were affected after 12 September.",
    "Confirmed: the distinguishing feature of PEC is legal delivery weight.",
])
def test_ordinary_prose_stays_below_the_bar(prose):
    """FAILS IF: the widening lets a single echo count twice, the bounded
    contexts are widened to the bare word, or the two-marker rule goes."""
    assert is_governor_checkpoint_answer(prose) is False


def test_the_producer_pins_the_words_the_new_markers_key_on():
    """FAILS IF: "distinguish" / "no new information" leave the steer's
    vocabulary — re-wording risk.py would silently strip the smoother."""
    assert "distinguish" in STEER_DIRECTIVE_TERMS
    assert "no new information" in STEER_DIRECTIVE_TERMS


# ── the stream path ───────────────────────────────────────────────────────

@pytest.fixture
def agent(mock_context):
    return GhostAgent(mock_context)


def _wire(agent, monkeypatch):
    agent.context.args.use_planning = True
    monkeypatch.setattr(EXP, "arm_for", lambda ctx, name, req_id="": EXP.TREATMENT)
    monkeypatch.setattr(RISK, "steer_enabled", lambda: True)
    reading = RISK.RiskReading(score=0.9, depth_prior=0.6, effort_struggle=0.5,
                               failure_pressure=0.2, step=8, band="high")
    monkeypatch.setattr(RISK, "turn_risk", lambda **kw: reading)
    plans = [{"thought": "list first", "tree_update": {"id": "root", "children": [
                  {"id": "task_1", "description": "list", "status": "READY"}]},
              "next_action_id": "task_1", "required_tool": "file_system"},
             {"thought": "deliver", "tree_update": {}, "next_action_id": "none",
              "required_tool": "none"}]
    state = {"main": 0}

    async def fake(payload, *a, **kw):
        # the planner is the only non-streamed call on a streamed request
        plan = plans.pop(0) if plans else {"thought": "deliver", "tree_update": {},
                                           "next_action_id": "none", "required_tool": "none"}
        return {"choices": [{"message": {"content": json.dumps(plan)}, "finish_reason": "stop"}]}

    _call = ("\n\n<tool_call>\n<function=file_system>\n<parameter=operation>list</parameter>\n"
             "</function>\n</tool_call>")

    async def final_stream(payload, use_coding=False):
        # every main turn of a streamed request is streamed: turn 1 answers
        # the checkpoint and calls a tool (the recorded shape), turn 2 is
        # the final answer
        state["main"] += 1
        if state["main"] == 1:
            yield sse({"content": _CHECKPOINT + _call})
        else:
            yield sse({"content": _REPORT})
        yield b"data: [DONE]\n\n"

    agent.context.llm_client.chat_completion = AsyncMock(side_effect=fake)
    agent.context.llm_client.stream_chat_completion = final_stream
    agent.available_tools["file_system"] = AsyncMock(return_value="f.txt")
    agent._record_turn_trajectory = MagicMock()


async def _client_text(agent):
    res = await agent.handle_chat(_REQ, background_tasks=MagicMock())
    gen = res[0] if isinstance(res, tuple) else res
    chunks = [c async for c in gen]
    text = "".join(
        (json.loads(c.decode()[6:]).get("choices") or [{}])[0].get("delta", {}).get("content") or ""
        for c in chunks if c.startswith(b"data: ") and c.strip() != b"data: [DONE]")
    durable = (agent._record_turn_trajectory.call_args.kwargs.get("final_content", "")
               if agent._record_turn_trajectory.called else "")
    return text, durable


@pytest.mark.asyncio
async def test_a_recorded_checkpoint_answer_leaves_the_streamed_reply(agent, monkeypatch):
    """FAILS IF: the drop runs only in `_finalize_and_return` — the live
    world: the web UI opened with the CONFIRMED/ASSUMED block."""
    _wire(agent, monkeypatch)
    text, durable = await _client_text(agent)
    assert _REPORT in text, text[:300]
    assert "CONFIRMED (observed)" not in text
    assert "One targeted search, then STOP" not in text
    assert "CONFIRMED (observed)" not in durable


@pytest.mark.asyncio
async def test_the_drop_needs_a_recorded_segment(agent, monkeypatch):
    """FAILS IF: the stream prefix is filtered by resemblance instead of
    by the recorded segments — with the steer on the CONTROL arm nothing is
    recorded, and the same text is the model's own prose, kept."""
    _wire(agent, monkeypatch)
    monkeypatch.setattr(EXP, "arm_for",
                        lambda ctx, name, req_id="": EXP.CONTROL if name == RISK.EXPERIMENT else EXP.TREATMENT)
    text, _ = await _client_text(agent)
    assert _REPORT in text
    assert "CONFIRMED (observed)" in text


def test_the_stream_variant_may_empty_the_prefix():
    """FAILS IF: `keep_if_empty=False` is ignored — a prefix that was
    nothing but the checkpoint answer would ship above the answer (the
    finalize path keeps its refusal, pinned in test_governor_checkpoint_scrub)."""
    from ghost_agent.core.reply_smoothing import drop_checkpoint_segments
    assert drop_checkpoint_segments(_CHECKPOINT, [_CHECKPOINT], keep_if_empty=False) == ""
    assert drop_checkpoint_segments(_CHECKPOINT, [_CHECKPOINT]) == _CHECKPOINT


@pytest.mark.asyncio
async def test_an_interim_observation_beside_the_answer_survives_the_drop(agent, monkeypatch):
    """FAILS IF: the drop takes the whole prefix instead of the recorded
    segment — the live prefix carried a second, unrecorded paragraph too."""
    _wire(agent, monkeypatch)
    obs = "The KELA extraction returned 8414 chars and names Revolut's PEC inbox as the recipient."
    state = {"main": 0}
    _call = ("\n\n<tool_call>\n<function=file_system>\n<parameter=operation>list</parameter>\n"
             "</function>\n</tool_call>")

    async def final_stream(payload, use_coding=False):
        state["main"] += 1
        if state["main"] == 1:
            yield sse({"content": _CHECKPOINT + _call})
        elif state["main"] == 2:
            yield sse({"content": obs + _call})
        else:
            yield sse({"content": _REPORT})
        yield b"data: [DONE]\n\n"
    agent.context.llm_client.stream_chat_completion = final_stream
    plans = iter([
        {"thought": "list", "tree_update": {}, "next_action_id": "task_1", "required_tool": "file_system"},
        {"thought": "list again", "tree_update": {}, "next_action_id": "task_1", "required_tool": "file_system"},
    ])

    async def fake(payload, *a, **kw):
        plan = next(plans, {"thought": "deliver", "tree_update": {}, "next_action_id": "none", "required_tool": "none"})
        return {"choices": [{"message": {"content": json.dumps(plan)}, "finish_reason": "stop"}]}
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=fake)
    text, durable = await _client_text(agent)
    assert _REPORT in text and obs in text
    assert "CONFIRMED (observed)" not in text
    assert obs in durable and "CONFIRMED (observed)" not in durable
