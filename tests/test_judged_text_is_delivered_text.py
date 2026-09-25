"""The in-loop verifier judges the DELIVERED view — 2026-09-24.

The in-loop verdict was computed on the raw accumulated reply and
fingerprinted as such; finalisation then trimmed working narration
(`reply_smoothing`), the fingerprint no longer matched, and the verdict was
recomputed on the delivered text (slack-e6…: 26.7 s in-loop with an
escalation, then 23.3 s again). Both readers now go through
`reply_smoothing.delivery_view`, so the judged text IS the delivered text
and the in-loop verdict is reused.
"""
import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core.agent import GhostAgent
from ghost_agent.core.reply_smoothing import delivery_view, smooth_gated, treat_reply
from ghost_agent.core.verifier import VerifyResult, VerifyVerdict as VV
from tests.helpers import FakeBgTasks, make_context


def _resp(content, tool_calls=None):
    return {"choices": [{"message": {"role": "assistant", "content": content,
                                     "tool_calls": tool_calls or []}}]}


def _tc(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


NARRATED = ("Let me check the second source as well.\n\n"
            "The Tzaneio hospital was founded in 1873 by Nikitas Tzannis, per both sources.")


async def _run(monkeypatch, async_mode="0"):
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", async_mode)
    monkeypatch.setenv("GHOST_CRITIC_REPAIR_BUDGET", "5")   # async: the in-loop verdict is awaited
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    ctx = make_context()
    verifier = MagicMock()
    verifier.llm_client = MagicMock()
    judged = []

    async def judge(*a, **k):
        judged.append(k.get("claim") or (a[0] if a else ""))
        return VerifyResult(verdict=VV.CONFIRMED, confidence=0.95, reasoning="r", issues=[])
    verifier.verify_claim = judge
    verifier.verify_code_output = judge
    verifier.verify_visual = AsyncMock(return_value=None)
    ctx.verifier = verifier
    agent = GhostAgent(ctx)
    search = AsyncMock(return_value="### 1. Tzaneio\nFounded 1873 by Nikitas Tzannis.\n")
    agent.available_tools = {"web_search": search}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("", [_tc("c0", "web_search", {"query": "Tzaneio founded"}),
                   _tc("c1", "web_search", {"query": "Tzaneio founder"})]),
        _resp(NARRATED),
        _resp("(unreachable)"),
    ])
    out, _, _ = await agent.handle_chat(
        {"messages": [{"role": "user", "content": "Use web search: when was the Tzaneio hospital founded and by whom?"}]},
        FakeBgTasks())
    return out, judged


@pytest.mark.parametrize("async_mode", ["0", "1"])   # R4 pins review: production runs async
async def test_one_verdict_and_it_was_about_the_delivered_text(monkeypatch, async_mode):
    out, judged = await _run(monkeypatch, async_mode)
    assert "Let me check" not in out and "1873" in out            # narration trimmed at delivery
    assert len(judged) == 1, judged                                 # no recompute after the trim
    assert "Let me check" not in judged[0]                          # judged the delivered view


def test_the_two_readers_share_one_gate():
    text = "Let me look at the other file.\n\nThe port is 8080 and the timeout 30 s, as documented."
    two = [{"name": "file_system"}, {"name": "file_system"}]
    assert delivery_view(text, two) == treat_reply(text, n_real_tools=2) == smooth_gated(text, 2)
    assert delivery_view(text, two[:1]) == text == treat_reply(text, n_real_tools=1)
    assert "Let me look" not in delivery_view(text, two)


def test_delivery_view_never_raises_and_is_idempotent():
    assert delivery_view(None, None) is None
    text = "Let me look.\n\nThe port is 8080 and the timeout 30 s, as documented in the README."
    once = delivery_view(text, [{}, {}])
    assert delivery_view(once, [{}, {}]) == once


@pytest.mark.parametrize("answer", [
    "I have enough evidence to build a case against the contractor.",
    "I have sufficient details to complete the booking.",
    "I have enough information to do that.",
    "I have enough data to make a recommendation: go with PostgreSQL.",
])
def test_an_i_have_enough_paragraph_is_never_trimmed(answer):
    """§4KJ added an "I have enough data to <verb>…" beat; four review rounds
    found it deleting delivered answers (a lexical proxy for "is this a
    beat?") and it was REMOVED. Pinned as a non-final paragraph, where the
    beat pass would drop it."""
    from ghost_agent.core.reply_smoothing import smooth_reply
    text = answer + "\n\nThe details follow in the attached file, section 2."
    assert answer in smooth_reply(text)



async def test_finalize_smooths_only_behind_the_real_tool_gate(monkeypatch):
    """R4 pins review: the helper was pinned, not finalize's use of it. One
    REAL tool plus one REJECTED call is one real tool: the narration stays."""
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "0")
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    ctx = make_context()
    agent = GhostAgent(ctx)
    agent.available_tools = {"web_search": AsyncMock(return_value="### 1. x\nthe answer is 42\n"),
                             "learn_skill": AsyncMock(return_value="x")}
    agent.disabled_tools = {"learn_skill"}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("", [_tc("c0", "web_search", {"query": "x"}), _tc("c1", "learn_skill", {"lesson": "y"})]),
        _resp(NARRATED), _resp("(unreachable)")])
    out, _, _ = await agent.handle_chat(
        {"messages": [{"role": "user", "content": "Use web search: when was the Tzaneio hospital founded?"}]}, FakeBgTasks())
    assert "Let me check the second source" in out
