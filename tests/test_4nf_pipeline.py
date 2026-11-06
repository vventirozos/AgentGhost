"""§4NF (2026-10-10): the per-turn pipeline review — production behaviour.

Operator decisions: the async verifier is silent toward the owner until its
precision is proven; the announced-work worker check is off. Clear fixes: the
pre-reply worker calls have a real deadline, the user-correction judge runs
beside the turn, no post-mortem item is written (§4NM removed the producer
and its consumer), the evidence-gate scan is skipped when its experiment is
off, no planner snapshot without the planner."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core import agent as A
from ghost_agent.core.agent import GhostAgent


@pytest.fixture
def production(monkeypatch):
    """The production state: no stale switch is set (§4NM deleted it)."""
    monkeypatch.delenv("GHOST_VERIFIER_SHOW_OWNER", raising=False)


# ── the pre-reply worker calls have a real deadline ───────────────────





# ── the verifier is silent toward the owner ───────────────────────────

def test_the_owner_switch_is_gone(production, monkeypatch):
    """§4NM: the switch that showed the owner the verifier's corrections was
    deleted with its branch; setting the old env var changes nothing."""
    assert not hasattr(A, "_verifier_shows_owner")
    assert not hasattr(GhostAgent, "_notify_owner_correction")


def _refuted():
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    return VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=0.95,
                        issues=["the date is wrong"], reasoning="r")


def _uncertain_with_facts():
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    r = VerifyResult(verdict=VerifyVerdict.UNCERTAIN, confidence=0.6, issues=[], reasoning="r")
    r.unverified_facts = ["Acme Corp", "1987"]
    return r


def _late_agent(monkeypatch):
    a = GhostAgent.__new__(GhostAgent)
    a.context = SimpleNamespace(memory_system=None, trajectory_collector=None)
    for name in ("_backfill_trajectory_outcome", "_emit_late_outcome_correction",
                 "_relabel_episode", "_file_refute_followup_tasks"):
        monkeypatch.setattr(a, name, lambda *x, **k: None, raising=False)
    a._pending_corrections = []
    a._save_pending_corrections = lambda: None
    notices = []          # §4NM: the owner notice is gone; kept as an always-empty witness
    monkeypatch.setattr(a, "_critic_async_enabled", lambda: True)
    return a, notices


def test_a_late_refute_queues_no_banner_and_sends_no_notice(production, monkeypatch):
    """2 of the last 3 corrections shown to the owner were false; one was
    shown twice in one reply. The verdict still lands in the corpus."""
    a, notices = _late_agent(monkeypatch)
    a._record_late_verdict(_refuted(), "traj-1", conv_fp="conv1")
    assert a._pending_corrections == [] and notices == []


def test_the_old_switch_does_not_bring_the_refute_back(production, monkeypatch):
    monkeypatch.setenv("GHOST_VERIFIER_SHOW_OWNER", "1")
    a, notices = _late_agent(monkeypatch)
    a._record_late_verdict(_refuted(), "traj-1", conv_fp="conv1")
    assert a._pending_corrections == [] and notices == []


def test_no_source_caveat_is_queued(production, monkeypatch):
    a, _ = _late_agent(monkeypatch)
    monkeypatch.setattr(A, "_source_caveat_enabled", lambda: True)
    monkeypatch.setenv("GHOST_VERIFIER_SHOW_OWNER", "1")          # deleted switch: no effect
    assert a._queue_source_caveat(_uncertain_with_facts(), "conv1", "traj-1") is False
    assert a._pending_corrections == []


# ── the correction judge runs beside the turn ─────────────────────────

async def test_a_slow_correction_judge_does_not_hold_the_reply(monkeypatch, tmp_path):
    """It relabels the PRIOR turn only; it blocked a turn up to 20 s."""
    from tests.test_requester_role import _agent, _resp, FakeBgTasks
    agent, ctx, _ = _agent(monkeypatch, tmp_path)
    gate = asyncio.Event()
    started = []

    async def _adj(self, *a, **k):
        started.append(1)
        await gate.wait()            # never set while the turn runs
        return None
    monkeypatch.setattr(GhostAgent, "_adjudicate_correction", _adj)
    ctx.llm_client.chat_completion = AsyncMock(return_value=_resp("Sorry — it is 4."))
    out = await asyncio.wait_for(agent.handle_chat(
        {"messages": [{"role": "user", "content": "what is 2+2?"},
                      {"role": "assistant", "content": "5."},
                      {"role": "user", "content": "No, that's wrong. What is 2+2?"}]},
        FakeBgTasks(), request_id="web-owner-4nf"), timeout=30)
    assert out is not None and started            # the judge ran, the reply did not wait
    gate.set()


async def test_the_judge_then_relabels_the_prior_turn(monkeypatch):
    a = GhostAgent.__new__(GhostAgent)
    calls = []

    async def _adj(messages, last):
        return True
    a._adjudicate_correction = _adj
    a._maybe_promote_prior_turn_via_user_correction = lambda m, l, contradicts=None: calls.append(contradicts)
    await a._promote_prior_turn_after_judge([{"role": "user", "content": "x"}], "No, wrong")
    assert calls == [True]


# ── the evidence-gate scan and the planner snapshot ───────────────────

async def test_no_planner_snapshot_even_with_the_old_flag(monkeypatch, tmp_path):
    """§4NF stopped the per-iteration request copy for the planner; §4NM
    deleted the planner and the snapshot — a stale `use_planning` attribute
    on args no longer brings either back."""
    from tests.test_requester_role import _agent, _resp, FakeBgTasks
    agent, ctx, _ = _agent(monkeypatch, tmp_path)
    ctx.args.use_planning = True
    ctx._planner_prefix = None
    ctx.llm_client.chat_completion = AsyncMock(return_value=_resp("Hello there."))
    await agent.handle_chat({"messages": [{"role": "user", "content": "explain how DNS resolution works in detail"}]},
                            FakeBgTasks(), request_id="web-owner-plan")
    assert getattr(ctx, "_planner_prefix", None) is None
    assert ctx.llm_client.chat_completion.await_count >= 1
    for c in ctx.llm_client.chat_completion.await_args_list:
        assert c.kwargs.get("task_label") != "planner"


# ── the dead post-mortem item and the episode record ──────────────────

def test_no_post_mortem_item_is_queued():
    """§4NF found the item written on every complex turn while its consumer
    discarded it; §4NM removed both. No `post_mortem` journal append exists
    in agent.py."""
    import ast, inspect
    tree = ast.parse(inspect.getsource(A))
    kinds = [c.args[0].value for c in ast.walk(tree) if isinstance(c, ast.Call)
             and getattr(c.func, "attr", "") == "_journal_append_safe"
             and c.args and isinstance(c.args[0], ast.Constant)]
    assert kinds and "post_mortem" not in kinds, kinds


async def test_the_episode_record_does_not_hold_the_reply(monkeypatch, tmp_path):
    """Its readers (the episodic tier, consolidation) are off by default; it
    was awaited before the reply on every complex turn."""
    from tests.test_requester_role import _agent, _resp, _tc, FakeBgTasks
    agent, ctx, _ = _agent(monkeypatch, tmp_path)
    ctx.args.smart_memory = 0.9
    ctx.journal = MagicMock()
    gate = asyncio.Event()
    started = []

    async def _slow_episode(self, *a, **k):
        started.append(1)
        await gate.wait()
    monkeypatch.setattr(GhostAgent, "_record_episode_safe", _slow_episode)
    agent.available_tools = {"web_search": AsyncMock(return_value="result")}
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("", [_tc("c0", "web_search", {"query": "x"})]),
        _resp("", [_tc("c1", "web_search", {"query": "y"})]),
        _resp("", [_tc("c2", "web_search", {"query": "z"})]),
        _resp("It is 42."), _resp("Done."), _resp("Done.")])
    out = await asyncio.wait_for(agent.handle_chat(
        {"messages": [{"role": "user", "content": "search for x, y and z and tell me"}]},
        FakeBgTasks(), request_id="web-owner-ep"), timeout=20)
    await asyncio.sleep(0)
    assert out is not None and started            # it ran — beside the reply
    gate.set()


# ── fresh-reader round ────────────────────────────────────────────────

def test_a_correction_queued_before_the_switch_is_not_shown(production, monkeypatch):
    """Bound corrections live 24 h and survive a restart: the reader must be
    silent too, not only the writers."""
    a = GhostAgent.__new__(GhostAgent)
    a.context = SimpleNamespace()
    a._save_pending_corrections = lambda: None
    import time as _t
    a._pending_corrections = [{"note": "the date is wrong", "conv": "c1", "traj": "t", "ts": _t.monotonic(), "member": False},
                              {"note": "member note", "conv": "c2", "traj": "t2", "ts": _t.monotonic(), "member": True}]
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}, {"role": "user", "content": "q2"}]
    a._consume_pending_corrections(msgs, conv_fp="c1")
    assert a._active_correction == ""
    assert [c["note"] for c in a._pending_corrections] == ["member note"]      # a member's stays


def test_a_members_correction_still_reaches_the_member(production, monkeypatch):
    """§4LH: members are verified after the reply so a correction reaches
    their next message — the operator's silence was about the owner."""
    from ghost_agent.utils.logging import requester_role_context
    a, notices = _late_agent(monkeypatch)
    tok = requester_role_context.set("member")
    try:
        a._record_late_verdict(_refuted(), "traj-m", conv_fp="convm")
    finally:
        requester_role_context.reset(tok)
    assert a._pending_corrections and a._pending_corrections[0]["member"] is True
    assert notices == []                                   # the OWNER notice stays silent
