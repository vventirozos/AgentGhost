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
@pytest.mark.asyncio
async def test_the_warmup_primes_both_fs_batch_heads_control_last(agent_with_llm, monkeypatch):
    """Fails where the warmup ran as "SYSTEM" (no arm) and only ever warmed
    control: 4 of 15 treatment owner turns re-prefilled ~31.6k tokens (~35 s)."""
    import ghost_agent.tools.registry as R
    agent, llm = agent_with_llm
    agent.context.args.native_tools = True
    monkeypatch.setattr(R, "fs_batch_experiment_live", lambda ctx: True)
    seen = []
    monkeypatch.setattr(agent._RequestState, "get_active_tool_defs",
                        lambda self, q: seen.append(R._fs_batch_active(agent.context)) or
                        [{"type": "function", "function": {"name": "file_system", "parameters": {}}}],
                        raising=False)
    await agent.warm_up_main_prefix(quiet=True)
    assert seen == [True, False]                     # treatment first, control LAST (newest entry)
    assert llm.chat_completion.await_count == 2


def test_a_composed_skill_description_carries_no_live_counters(tmp_path):
    """Fails where "used 3x with 33% success" changed on every use and
    re-prefilled the conversation behind the tool block."""
    from ghost_agent.tools.composed_skills import ComposedSkill, ComposedSkillRegistry, SkillStep
    reg = ComposedSkillRegistry(tmp_path)
    sk = ComposedSkill(name="briefing", trigger_description="morning briefing",
                       steps=[SkillStep(tool_name="web_search", description="news",
                                        param_template={"query": "$topic"})],
                       usage_count=3, success_count=1)
    reg.register(sk)
    d1 = json.dumps(reg.to_tool_definitions())
    sk.usage_count, sk.success_count = 9, 9
    d2 = json.dumps(reg.to_tool_definitions())
    assert d1 == d2 and "33%" not in d1 and "used 3x" not in d1


# ── what holds the reply ─────────────────────────────────────────────────────
@pytest.mark.parametrize("tool,blocks", [("introspect", False), ("self_state", False),
                                         ("web_search", True), ("execute", True)])
def test_a_self_report_does_not_hold_the_reply_for_its_verdict(tool, blocks):
    """Fails where "good morning ghost" waited 35 s of 60 for a verdict on
    the agent describing itself."""
    from ghost_agent.core.agent import _should_await_repair_verdict
    lt = {"name": tool, "content": "x" * 400}
    assert _should_await_repair_verdict(65.0, lt, False) is blocks


@pytest.mark.asyncio
async def test_the_query_expansion_gives_up_quickly():
    """Fails where a 12 s worker timeout sat before the first token."""
    from ghost_agent.core.agent import GhostAgent, PRE_REPLY_ROUTE_TIMEOUT_S
    client = MagicMock()
    client.route = AsyncMock(return_value="expanded query")
    client.worker_clients = [object()]
    me = SimpleNamespace(context=SimpleNamespace(llm_client=client, args=SimpleNamespace(model="m")))
    out = await GhostAgent._route_query_expansion(me, "previous answer", "and the other one?", "legacy")
    assert out == "expanded query" and client.route.await_count == 1
    assert client.route.await_args.kwargs.get("timeout") == PRE_REPLY_ROUTE_TIMEOUT_S <= 5


@pytest.mark.asyncio
@pytest.mark.parametrize("query,basis,decomposes", [
    ("Context: a long previous answer about postgres replication slots | User intent: and the other?",
     "and the other?", False),                                   # expansion ran: one worker call is enough
    ("what is new in postgres", "what is new in postgres", False),   # short user words
    ("compare postgres 18 async io with the io_uring work in linux 6.12 kernels",
     "compare postgres 18 async io with the io_uring work in linux 6.12 kernels", True)])
async def test_the_memory_query_is_split_only_on_the_users_own_long_words(query, basis, decomposes):
    """Fails where the 8-word gate read the EXPANDED string — every short
    follow-up paid a second worker call (median 1.8 s) before the reply."""
    from ghost_agent.core.bus import MemoryBus
    client = MagicMock()
    client.route = AsyncMock(return_value='["a", "b"]')
    await MemoryBus()._decompose_query(query, client, basis=basis)
    assert bool(client.route.await_count) is decomposes
    if decomposes:
        assert client.route.await_args.kwargs.get("timeout") == 4.0


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
async def test_hydration_passes_the_users_own_words_to_the_split():
    from ghost_agent.core.bus import MemoryBus
    client = MagicMock()
    client.route = AsyncMock(return_value='["a", "b"]')
    await MemoryBus().hydrate_context(
        "Context: a long previous answer about postgres replication slots | User intent: and the other?",
        llm_client=client, raw_user_text="and the other?")
    assert not any(c.args and c.args[0] == "DECOMPOSE_QUERY" for c in client.route.await_args_list)


@pytest.mark.asyncio
@pytest.mark.parametrize("role,consumed", [("member", False), ("owner", True)])
async def test_a_members_turn_leaves_the_warmup_check_for_the_owner(monkeypatch, tmp_path, role, consumed):
    """Fails where a member's turn (whose head differs by design) raised a
    false "prefix warmup MISS" and used up the one-shot check."""
    from tests.test_requester_role import _agent, _resp
    from tests.helpers import FakeBgTasks
    agent, ctx, _ = _agent(monkeypatch, tmp_path)
    ctx.skill_memory = MagicMock(is_read_only=False)
    ctx._warmed_sys_hash = "boot0000"
    ctx.llm_client.chat_completion = AsyncMock(return_value=_resp("ok"))
    await agent.handle_chat({"messages": [{"role": "user", "content": "what is the capital of France?"}]},
                            FakeBgTasks(), request_id="web-1", requester_role=role)
    assert (ctx._warmed_sys_hash is None) is consumed
