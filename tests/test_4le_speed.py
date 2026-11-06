"""§4LE (2026-10-04): per-turn speed — what holds a reply back, what breaks
the prompt cache, and what a simple turn still pays. Each test names the
world it fails in."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.test_4fm_prefix_rewarm import agent_with_llm  # noqa: F401 — the fixture


# ── the prompt cache ─────────────────────────────────────────────────────────


# ── what holds the reply ─────────────────────────────────────────────────────
@pytest.mark.parametrize("tool,blocks", [("introspect", False),
                                         ("web_search", True), ("execute", True)])
def test_a_self_report_does_not_hold_the_reply_for_its_verdict(tool, blocks):
    """Fails where "good morning ghost" waited 35 s of 60 for a verdict on
    the agent describing itself."""
    from ghost_agent.core.agent import _should_await_repair_verdict
    lt = {"name": tool, "content": "x" * 400}
    assert _should_await_repair_verdict(65.0, lt, False) is blocks


# ── simple turns ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("text,trivial", [
    ("γεια σου", True), ("Καλημέρα!", True), ("ευχαριστώ πολύ", True), ("γειά σας", True),
    ("τι κάνεις;", False), ("γεια, φτιάξε μου ένα αρχείο", False), ("πολύ καλό το άρθρο", False),
    ("ok thanks", True), ("hi, run the tests", False)])
def test_a_greek_greeting_takes_the_fast_path(text, trivial):
    """Fails where only English greetings took the fast path (5 in a week);
    every "γεια" paid the full path (~3.7 s, two LLM calls)."""
    from ghost_agent.core.agent import GhostAgent
    assert GhostAgent._is_strict_trivial_chat(text.lower()) is trivial


@pytest.mark.asyncio
@pytest.mark.parametrize("role,consumed", [("member", False), ("owner", True)])
async def test_a_members_turn_leaves_the_warmup_check_for_the_owner(monkeypatch, tmp_path, role, consumed):
    """Fails where a member's turn (whose head differs by design) raised a
    false "prefix warmup MISS" and used up the one-shot check."""
    from tests.test_requester_role import _agent, _resp
    from tests.helpers import FakeBgTasks
    agent, ctx, _ = _agent(monkeypatch, tmp_path)
    ctx._warmed_sys_hash = "boot0000"
    ctx.llm_client.chat_completion = AsyncMock(return_value=_resp("ok"))
    await agent.handle_chat({"messages": [{"role": "user", "content": "what is the capital of France?"}]},
                            FakeBgTasks(), request_id="web-1", requester_role=role)
    assert (ctx._warmed_sys_hash is None) is consumed
