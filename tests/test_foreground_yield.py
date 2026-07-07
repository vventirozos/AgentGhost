"""Tests for background-LLM yielding to active foreground requests.

Regression target (post-req-70 starvation): `foreground_tasks` only
counts LLM calls in flight, but a user turn spends much of its wall-clock
BETWEEN LLM calls (tools, file I/O, browser). Background cycles grabbed
the single llama slot in those gaps and the user's next prompt queued
behind full background generations for ~12 minutes. The fix adds
`foreground_requests` — active user requests at the API layer — and
parks background callers while either signal is hot.
"""
import asyncio
import inspect

from ghost_agent.core.llm import LLMClient


def _bare_client():
    c = LLMClient.__new__(LLMClient)
    c.foreground_tasks = 0
    c.foreground_requests = 0
    c._foreground_lock = asyncio.Lock()
    return c


def _instant_sleep(monkeypatch, counter):
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds):
        counter.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)


async def test_returns_immediately_when_idle(monkeypatch):
    c = _bare_client()
    sleeps: list = []
    _instant_sleep(monkeypatch, sleeps)
    await c._wait_for_foreground_clear()
    assert sleeps == []


async def test_waits_for_active_request_then_releases(monkeypatch):
    c = _bare_client()
    c.foreground_requests = 1
    sleeps: list = []
    _instant_sleep(monkeypatch, sleeps)

    async def finish_request():
        # let the waiter spin a few ticks, then clear the request
        for _ in range(3):
            await asyncio.sleep(0)
        c.foreground_requests = 0

    await asyncio.gather(c._wait_for_foreground_clear(), finish_request())
    assert len(sleeps) >= 1  # it actually parked


async def test_active_request_outlasts_the_old_30s_cap(monkeypatch):
    """With a user request active, background must keep waiting well past
    the 30-tick budget that used to let it barge in."""
    c = _bare_client()
    c.foreground_requests = 1
    sleeps: list = []
    _instant_sleep(monkeypatch, sleeps)

    async def clear_after_120_ticks():
        while len(sleeps) < 120:
            await asyncio.sleep(0)
        c.foreground_requests = 0

    await asyncio.gather(
        c._wait_for_foreground_clear(), clear_after_120_ticks())
    assert len(sleeps) >= 120


async def test_task_only_blocking_keeps_30s_cap(monkeypatch):
    """No active request, just an in-flight LLM call: the old 30s
    proceed-anyway behavior is preserved (a stale background result
    beats none)."""
    c = _bare_client()
    c.foreground_tasks = 1
    sleeps: list = []
    _instant_sleep(monkeypatch, sleeps)
    await c._wait_for_foreground_clear()
    assert 28 <= len(sleeps) <= 35


async def test_leaked_counter_hard_ceiling(monkeypatch):
    """A leaked foreground_requests must not deadlock background work
    forever — the 600-tick hard ceiling releases it."""
    c = _bare_client()
    c.foreground_requests = 1  # never cleared
    sleeps: list = []
    _instant_sleep(monkeypatch, sleeps)
    await c._wait_for_foreground_clear()
    assert 595 <= len(sleeps) <= 605


def test_api_layer_marks_requests_on_both_paths():
    """Both chat handlers (streaming + non-streaming) must bracket
    handle_chat with the foreground_requests counter — increment before,
    guaranteed decrement after."""
    import ghost_agent.api.routes as routes
    src = inspect.getsource(routes)
    assert src.count("_mark_foreground(agent, +1)") >= 2
    assert src.count("_mark_foreground(agent, -1)") >= 2


def test_mark_foreground_counts_and_floors_at_zero():
    from types import SimpleNamespace
    from ghost_agent.api.routes import _mark_foreground

    llm = SimpleNamespace(foreground_requests=0)
    agent = SimpleNamespace(context=SimpleNamespace(llm_client=llm))
    _mark_foreground(agent, +1)
    _mark_foreground(agent, +1)
    assert llm.foreground_requests == 2
    _mark_foreground(agent, -1)
    _mark_foreground(agent, -1)
    _mark_foreground(agent, -1)  # extra decrement must floor, not go negative
    assert llm.foreground_requests == 0


def test_mark_foreground_tolerates_mocked_clients():
    """Test apps wire MagicMock contexts — instrumentation must no-op,
    never raise (the original inline version broke 11 API tests with
    `max(0, MagicMock - 1)`)."""
    from unittest.mock import MagicMock
    from ghost_agent.api.routes import _mark_foreground

    _mark_foreground(MagicMock(), +1)   # mock counter → skipped
    _mark_foreground(object(), +1)      # no context at all → skipped
    _mark_foreground(None, -1)          # no agent → skipped


def test_background_gates_use_the_shared_wait():
    src = inspect.getsource(LLMClient.chat_completion)
    assert "_wait_for_foreground_clear" in src
    src_stream = inspect.getsource(LLMClient.stream_chat_completion)
    assert "_wait_for_foreground_clear" in src_stream


# ---------------------------------------------------------------------------
# Priority-routing fixes (IMPROVEMENTS.md #2 + #19, 2026-07-07)
#
# Two symmetric misclassifications shared one root cause — the foreground
# wait protects the MAIN inference slot, but callers were labelled by "is
# this housekeeping?" instead of "does this call block a live reply / touch
# the main slot?":
#   (a) Summarizers awaited INLINE by the user's own turn (Context Shield,
#       _prune_context condenser, --perfect-it) were is_background=True and
#       parked in _wait_for_foreground_clear against their OWN request — a
#       deterministic stall up to the 600s ceiling.
#   (b) Reflection/post-mortem closures in main.py ran at FOREGROUND
#       priority, contending with the user's next turn and making
#       foreground_tasks misreport an idle reflection as a live user.
# Plus: background calls served by an off-main pool (worker/critic/vision/
# swarm) skip the wait entirely — they never contend for the main slot.
# ---------------------------------------------------------------------------


def _routable_client():
    c = _bare_client()
    c._bg_queue_sem = asyncio.Semaphore(3)
    return c


async def _skip_wait_probe(monkeypatch, client, **call_kwargs):
    """Run chat_completion with instrumented wait/do; return (waited, done)."""
    waited, done = [], []

    async def fake_wait():
        waited.append(True)

    async def fake_do(payload, *a, **kw):
        done.append(True)
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(client, "_wait_for_foreground_clear", fake_wait)
    monkeypatch.setattr(client, "_do_chat_completion", fake_do)
    await client.chat_completion({"messages": []}, **call_kwargs)
    return waited, done


async def test_background_call_to_worker_pool_skips_the_wait(monkeypatch):
    c = _routable_client()
    c.foreground_requests = 1  # a main-node wait would park here
    c.worker_clients = [{"url": "http://w", "model": "m", "client": None}]
    waited, done = await _skip_wait_probe(
        monkeypatch, c, use_worker=True, is_background=True)
    assert done and not waited


async def test_background_call_to_critic_pool_skips_the_wait(monkeypatch):
    c = _routable_client()
    c.foreground_requests = 1
    c.critic_clients = [{"url": "http://c", "model": "m", "client": None}]
    waited, done = await _skip_wait_probe(
        monkeypatch, c, use_critic=True, is_background=True)
    assert done and not waited


async def test_background_call_to_main_node_still_waits(monkeypatch):
    """No pools configured → the call lands on the main node and MUST keep
    the req-70 starvation protection (park behind an active user)."""
    c = _routable_client()
    c.worker_clients = []
    waited, done = await _skip_wait_probe(
        monkeypatch, c, use_worker=True, is_background=True)
    assert done and waited == [True]


async def test_plain_background_call_still_waits(monkeypatch):
    c = _routable_client()
    waited, done = await _skip_wait_probe(monkeypatch, c, is_background=True)
    assert done and waited == [True]


def test_inline_request_path_summarizers_are_foreground():
    """The three calls awaited inline by the user's own turn must never
    take the background path (self-stall regression guard)."""
    from ghost_agent.core.agent import GhostAgent

    prune_src = inspect.getsource(GhostAgent._prune_context)
    assert "is_background=False" in prune_src
    assert "is_background=True" not in prune_src

    chat_src = inspect.getsource(GhostAgent.handle_chat)
    assert ("chat_completion(shield_payload, use_worker=True, "
            "is_background=False)") in chat_src

    pi_src = inspect.getsource(GhostAgent._perfect_it_generate_and_learn)
    assert "is_background=not foreground" in pi_src


def test_perfect_it_inline_path_passes_foreground():
    """--perfect-it awaits the optimization on the response path; the
    deferred internal-learning task keeps the background default."""
    import re
    from ghost_agent.core.agent import GhostAgent

    src = inspect.getsource(GhostAgent.handle_chat)
    assert re.search(
        r"_perfect_it_generate_and_learn\(\s*p_payload,\s*_pp_lesson_label,"
        r"\s*current_trajectory_id,\s*foreground=True",
        src,
    )
    # the deferred wrapper must NOT force foreground
    deferred = src[src.index("_deferred_perfect_it"):]
    first_call = deferred[:deferred.index(")")]
    assert "foreground=True" not in first_call


def test_reflection_and_postmortem_llm_calls_are_background():
    """critique / plan-verify / postmortem-analyze / postmortem-patch are
    learning work: background priority, never contending with a live user."""
    import ghost_agent.main as main_mod

    src = inspect.getsource(main_mod)
    assert src.count("chat_completion(payload, is_background=True)") >= 4


async def test_long_park_emits_visibility_log(monkeypatch):
    """A background call parked 120s behind an active request logs ONE
    line so the operator's stream shows the park instead of a silent gap."""
    import ghost_agent.core.llm as llm_mod

    c = _bare_client()
    c.foreground_requests = 1  # never cleared → rides to the 600s ceiling
    sleeps: list = []
    _instant_sleep(monkeypatch, sleeps)
    logs: list = []
    monkeypatch.setattr(
        llm_mod, "pretty_log", lambda *a, **k: logs.append(a))
    await c._wait_for_foreground_clear()
    assert sum(1 for a in logs if "BG Queue Wait" in str(a)) == 1
