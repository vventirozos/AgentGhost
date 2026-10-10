"""§4MU (2026-10-09): speed, end to end. Each test names the measured cost it
fails on (lenses on the agent log joined to the llama-server log)."""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core.reply_tap import ReplyTap, reply_tap_context
from tests.test_4mp_reply_tap import ANSWER, _feed_stream, _frames, _text


# ── the re-warm never runs inside a live request ───────────────────────

async def test_the_head_rewarm_skips_while_a_request_is_live():
    """5 of 97 owner continuations: the re-warm, admitted into an image
    render's idle window, matched the owner's slot at similarity 1.000 and
    cut the live conversation back to the head (up to 12.5 s re-prefill)."""
    from ghost_agent.core.agent import GhostAgent
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = SimpleNamespace(llm_client=SimpleNamespace(foreground_requests=1))
    warmed, ticks = [], []

    async def warm(quiet=False):
        warmed.append(1)

    async def metrics():
        return 0, len(ticks)                      # idle server, something ran each tick

    async def sleep(_s):
        ticks.append(1)
        if len(ticks) == 3:
            agent.context.llm_client.foreground_requests = 0
        if len(ticks) > 4:
            raise asyncio.CancelledError()
    agent.warm_up_main_prefix = warm
    with pytest.raises(asyncio.CancelledError):
        await agent.rewarm_main_prefix_loop(180, sleep=sleep, metrics=metrics)
    assert warmed == [1, 1]                       # ticks 3 and 4 only: 1 and 2 were inside the request


# ── pre-reply worker chores: the r1 skip is REVERTED (r2 review) ───────

def _client(worker="http://nova:8088", critic="http://nova:8088"):
    from ghost_agent.core.llm import LLMClient
    c = LLMClient.__new__(LLMClient)
    c.worker_clients = [{"url": worker, "client": None, "model": "w"}]
    c.critic_clients = [{"url": critic, "client": None, "model": "c"}] if critic else []
    c.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": "rewritten"}}]})
    return c


async def test_a_parked_head_rewarm_is_never_admitted_into_a_render_window():
    """r1 review: the loop's own check races a request arriving mid-tick; a
    parked re-warm was then let in through the image-render idle window."""
    from ghost_agent.core import llm as L
    from tests.test_4ms_idle_cycle import _bare_client
    for label, admitted in (("main-prefix-rewarm", False), ("selfplay-judge", True)):
        c = _bare_client()
        c.foreground_requests = 1
        c.main_slot_idle_windows = 1
        c._idle_window_admits = 1
        real = asyncio.sleep
        n = {"i": 0}

        async def instant(_s):
            n["i"] += 1
            if n["i"] > 5:
                raise TimeoutError("still parked")
            await real(0)
        L_sleep = asyncio.sleep
        asyncio.sleep = instant
        try:
            got = "admitted"
            try:
                await c._wait_for_foreground_clear(label)
            except TimeoutError:
                got = "parked"
        finally:
            asyncio.sleep = L_sleep
        assert (got == "admitted") is admitted, label


def test_a_reply_streamed_with_its_staged_banner_commits():
    """The finalize prepends a deferred correction/caveat; streamed without
    it, the shown text no longer prefixed the final one and the reply was
    retracted (seen live; every verdict is late now)."""
    banner = "ℹ️ **On my previous answer:** the year was not in the sources.\n\n---\n\n"
    tap = ReplyTap("r1", "m", 1)
    tap.set_lead(banner)
    tap.begin_generation()
    _feed_stream(tap, ANSWER)
    shown = _text(_frames(tap))
    assert shown.startswith(banner) and shown == banner + ANSWER
    out = tap.finish(banner + ANSWER)
    assert tap.outcome == "committed" and tap.retracts == []
    # without the lead: the old behaviour
    tap2 = ReplyTap("r2", "m", 1)
    tap2.begin_generation()
    _feed_stream(tap2, ANSWER)
    _frames(tap2)
    tap2.finish(banner + ANSWER)
    assert tap2.outcome == "retracted"


def test_a_lead_is_resent_after_a_retract_and_never_twice():
    banner = "⚠️ **Correction to my previous answer:** x\n\n---\n\n"
    tap = ReplyTap("r1", "m", 1)
    tap.set_lead(banner)
    tap.begin_generation()
    _feed_stream(tap, ANSWER)
    tap.tool_call_seen()                           # a call followed: retracted
    tap.begin_generation()
    _feed_stream(tap, ANSWER)
    assert tap.sent == banner + ANSWER and tap.sent.count("Correction") == 1


def test_the_agent_leads_the_stream_only_without_a_start_with_rule():
    from ghost_agent.core.agent import GhostAgent
    agent = GhostAgent.__new__(GhostAgent)
    plain = [{"role": "user", "content": "what's new in postgres 19?"}]
    agent._active_project_constraints = lambda **k: []
    assert agent._start_with_rule_for(plain) is False
    # the SAME constraint sources the finalize reads (`_head_insert_below_start_with`)
    agent._active_project_constraints = lambda **k: ["Start your reply with: Verdict"]
    assert agent._start_with_rule_for(plain) is True

    def boom(**k):
        raise RuntimeError("x")
    agent._active_project_constraints = boom
    assert agent._start_with_rule_for(plain) is True                # unsure → no lead


def test_consuming_a_correction_sets_the_streams_lead():
    import ast
    import inspect
    import textwrap
    from ghost_agent.core import agent as A
    fn = next(n for n in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(A.GhostAgent))))
              if isinstance(n, ast.FunctionDef) and n.name == "_consume_pending_corrections")
    src = ast.unparse(fn)
    i = src.index("self._active_correction = banner + '---\\n\\n'")
    assert "_tap_c.set_lead(_owner_rule_banner_ctx.get() + self._active_correction)" in src[i:i + 700]   # §4MZ: rule line + correction
    assert "not self._start_with_rule_for(messages)" in src[i:i + 700]


# ── what waits is visible ──────────────────────────────────────────────

def test_a_wait_on_the_turn_lock_is_logged_before_it_happens():
    import ast
    import inspect
    import textwrap
    from ghost_agent.core import agent as A
    src = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(A.GhostAgent))))
    i = src.index("if self.agent_semaphore.locked():")
    assert "Turn Queue" in src[i:i + 300] and src.index("async with self.agent_semaphore:", i) > i


async def test_pre_reply_chores_run_even_while_a_verification_is_in_flight():
    """r2 review MAJOR: the skip (keyed on whole 108-146 s verdict tasks)
    fired 32 times against 2 dispatches after deploy, while 71% of the calls
    made inside a verification had succeeded. Reverted: a routing chore is
    always dispatched."""
    c = _client()
    out = await c.route("EXPAND_QUERY", {"messages": []}, fallback="FALLBACK", timeout=4.0)
    assert out == "rewritten" and c.chat_completion.await_count == 1
    assert not hasattr(c, "worker_busy_with_deferred_verify")


async def test_a_head_rewarm_that_queued_into_a_new_request_is_dropped():
    """r2 review: after `_wait_for_foreground_clear` the call still waits on
    the queue permit; a request that starts meanwhile must not get the
    re-warm dispatched into its live slot."""
    from ghost_agent.core.llm import BackgroundDeferred, LLMClient
    from tests.test_4ms_idle_cycle import _bare_client
    c = _bare_client()
    c._bg_queue_sem = asyncio.Semaphore(1)
    c.worker_clients = c.critic_clients = c.vision_clients = c.swarm_clients = c.coding_clients = []
    sent = []

    async def clear(label):
        c.foreground_requests = 1           # the owner arrives while we queued
    async def do(*a, **k):
        sent.append(1)
        return {}
    c._wait_for_foreground_clear = clear
    c._do_chat_completion = do
    with pytest.raises(BackgroundDeferred):
        await c.chat_completion({"messages": []}, is_background=True, task_label="main-prefix-rewarm")
    assert sent == []
