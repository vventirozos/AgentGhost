"""§4FM pins: the byte-stable head is re-warmed periodically so its
prompt-cache entry stays the newest on the main slot (the server saves a
~200 MiB entry on every release; ~30 fit in --cache-ram 6144; idle churn is
2–4 requests/min, so the boot warmup died ~10 minutes in and every later
user turn re-prefilled ~26k tokens — measured 28,767 tokens, 0 cached, 34 s).
Review fixes (§4FM): the resident/evicted verdict reads the reply's cached
token count (the clock also measures queueing), a tick is skipped while the
slot is busy or when nothing ran since the last re-warm, a re-warm never
re-arms the boot MISS check, the knobs are clamped, and warm-up calls are
never recorded into the corpus window.
"""
import asyncio
import ast
import inspect
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core import agent as agent_mod
from ghost_agent.core.agent import GhostAgent

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("GHOST_MAIN_PREFIX_REWARM_S", "GHOST_MAIN_PREFIX_REWARM_SLOW_S"):
        monkeypatch.delenv(k, raising=False)


def _resp(prompt=28000, cached=None):
    u = {"prompt_tokens": prompt}
    if cached is not None:
        u["prompt_tokens_details"] = {"cached_tokens": cached}
    return {"choices": [{"message": {"content": "x"}}], "usage": u}


@pytest.fixture
def agent_with_llm(mock_context):
    llm = MagicMock()
    llm.upstream_url = "http://127.0.0.1:8088"
    llm.chat_completion = AsyncMock(return_value=_resp(28000, 26000))
    mock_context.llm_client = llm
    mock_context._warmed_sys_hash = "boot0000"
    agent = GhostAgent(mock_context)
    return agent, llm


# --- knobs ---------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    (None, 180.0), ("banana", 180.0), ("0", 0.0), ("-1", 0.0),          # off spellings
    ("90", 90.0), ("5", 30.0), ("0.5", 30.0), ("18", 30.0),              # floor at 30 s
])
def test_period_knob_default_floor_and_off(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("GHOST_MAIN_PREFIX_REWARM_S", raising=False)
    else:
        monkeypatch.setenv("GHOST_MAIN_PREFIX_REWARM_S", raw)
    assert agent_mod.rewarm_period_s() == expected


def test_slow_knob_default_and_floor(monkeypatch):
    assert agent_mod.rewarm_slow_s() == 10.0
    monkeypatch.setenv("GHOST_MAIN_PREFIX_REWARM_SLOW_S", "0")
    assert agent_mod.rewarm_slow_s() == 1.0
    monkeypatch.setenv("GHOST_MAIN_PREFIX_REWARM_SLOW_S", "2.5")
    assert agent_mod.rewarm_slow_s() == 2.5


def test_verdict_reads_cached_tokens_and_falls_back_to_unknown():
    assert agent_mod.rewarm_verdict(_resp(28000, 26000), 0.5) == ("resident", 26000, 28000)
    assert agent_mod.rewarm_verdict(_resp(28000, 14000), 0.5) == ("resident", 14000, 28000)   # exactly half
    assert agent_mod.rewarm_verdict(_resp(28000, 13999), 0.5) == ("evicted", 13999, 28000)
    assert agent_mod.rewarm_verdict(_resp(28000, 0), 0.5) == ("evicted", 0, 28000)
    assert agent_mod.rewarm_verdict(_resp(28000, None), 40.0) == ("unknown", 0, 0)
    assert agent_mod.rewarm_verdict({}, 40.0) == ("unknown", 0, 0)
    assert agent_mod.rewarm_verdict(None, 40.0) == ("unknown", 0, 0)


# --- the quiet re-warm ----------------------------------------------------------

async def test_quiet_rewarm_resident_logs_nothing_and_does_not_rearm_the_miss_check(agent_with_llm):
    agent, llm = agent_with_llm
    with patch.object(agent_mod, "pretty_log") as plog:
        await agent.warm_up_main_prefix(quiet=True)
    assert llm.chat_completion.await_count == 1
    kw = llm.chat_completion.await_args.kwargs
    assert kw["task_label"] == "main-prefix-rewarm" and kw["is_background"] is True
    assert plog.call_count == 0
    assert agent.context._warmed_sys_hash == "boot0000"      # untouched: MISS fires once per boot


def _fake_clock(monkeypatch, *spans):
    """time.time() inside the module returns 1000, then 1000 + span for each
    later call — a 40 s span means "the call took 40 s" without sleeping."""
    seq = iter([1000.0] + [1000.0 + x for x in spans] + [1000.0 + spans[-1]] * 50)
    monkeypatch.setattr(agent_mod.time, "time", lambda: next(seq))


async def test_quiet_rewarm_warns_from_the_token_count_not_the_clock(agent_with_llm, monkeypatch):
    """§4FM review MAJOR-1: a re-warm that took 40 s (parked behind a user
    turn) with the head resident must stay silent; an evicted head answered
    in a moment must warn."""
    agent, llm = agent_with_llm
    _fake_clock(monkeypatch, 40.0)
    llm.chat_completion = AsyncMock(return_value=_resp(28000, 26000))
    with patch.object(agent_mod, "pretty_log") as plog:
        await agent.warm_up_main_prefix(quiet=True)
    assert plog.call_count == 0
    llm.chat_completion = AsyncMock(return_value=_resp(28000, 0))   # fast, but nothing cached
    with patch.object(agent_mod, "pretty_log") as plog:
        await agent.warm_up_main_prefix(quiet=True)
    assert plog.call_count == 1
    msg = plog.call_args.args[1]
    assert "evicted" in msg and "0 of 28000 tokens cached" in msg


async def test_quiet_rewarm_uses_the_clock_only_without_a_usage_block(agent_with_llm, monkeypatch):
    agent, llm = agent_with_llm
    no_usage = {"choices": [{"message": {"content": "x"}}]}
    llm.chat_completion = AsyncMock(return_value=no_usage)
    _fake_clock(monkeypatch, 40.0)                       # 40 s ≥ the 10 s fallback threshold
    with patch.object(agent_mod, "pretty_log") as plog:
        await agent.warm_up_main_prefix(quiet=True)
    assert plog.call_count == 1 and "0 of 0 tokens cached" in plog.call_args.args[1]
    _fake_clock(monkeypatch, 9.9)                        # just under: resident
    with patch.object(agent_mod, "pretty_log") as plog:
        await agent.warm_up_main_prefix(quiet=True)
    assert plog.call_count == 0
    monkeypatch.setenv("GHOST_MAIN_PREFIX_REWARM_SLOW_S", "5")
    _fake_clock(monkeypatch, 5.0)                        # the knob moves the threshold
    with patch.object(agent_mod, "pretty_log") as plog:
        await agent.warm_up_main_prefix(quiet=True)
    assert plog.call_count == 1


async def test_loud_warmup_is_unchanged_and_arms_the_miss_check(agent_with_llm):
    agent, llm = agent_with_llm
    with patch.object(agent_mod, "pretty_log") as plog:
        await agent.warm_up_main_prefix()
    assert plog.call_count == 2
    assert llm.chat_completion.await_args.kwargs["task_label"] == "main-prefix-warmup"
    assert agent.context._warmed_sys_hash != "boot0000"


async def test_quiet_rewarm_failure_never_escapes_but_cancellation_does(agent_with_llm):
    agent, llm = agent_with_llm
    llm.chat_completion = AsyncMock(side_effect=RuntimeError("upstream down"))
    await agent.warm_up_main_prefix(quiet=True)
    llm.chat_completion = AsyncMock(side_effect=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):          # §4FM review M1: shutdown can stop it
        await agent.warm_up_main_prefix(quiet=True)


# --- the loop -------------------------------------------------------------------

async def test_loop_skips_busy_and_idle_ticks_and_rewarms_when_the_slot_churned(mock_context):
    agent = GhostAgent(mock_context)
    calls, slept = [], []
    readings = iter([(0, 100), (0, 100),          # tick 1: re-warm (first), then refresh → 100
                     (1, 130),                    # tick 2: busy → skip
                     (0, 130), (0, 130),          # tick 3: churned since 100 → re-warm, refresh
                     (0, 130),                    # tick 4: unchanged → skip
                     (None, None), (None, None),  # tick 5: metrics down → re-warm anyway
                     ])

    state = {"busy": False}

    async def fake_metrics():
        proc, total = next(readings)
        state["busy"] = bool(proc)
        return (proc, total)

    async def fake_warm(*, quiet=False):
        calls.append(quiet)
        assert not state["busy"], "re-warmed while the main slot was busy (review MAJOR-2)"

    async def fake_sleep(s):
        slept.append(s)
        if len(slept) > 5:
            raise asyncio.CancelledError()
    agent.warm_up_main_prefix = fake_warm
    with pytest.raises(asyncio.CancelledError):
        await agent.rewarm_main_prefix_loop(180, sleep=fake_sleep, metrics=fake_metrics)
    assert slept == [180.0] * 6
    assert calls == [True, True, True]                  # ticks 1, 3 and 5 only


async def test_metrics_reader_parses_the_servers_prometheus_text(agent_with_llm, monkeypatch):
    agent, llm = agent_with_llm
    import urllib.request
    seen = {}

    class R:
        def __init__(self, body): self.body = body
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self.body

    def fake_urlopen(url, timeout=0):
        seen["url"] = url
        return R(b"# HELP x\nllamacpp:prompt_tokens_total 538870\nllamacpp:requests_processing 1\n")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert await agent._main_slot_metrics() == (1, 538870)
    assert seen["url"] == "http://127.0.0.1:8088/metrics"
    llm.upstream_url = "http://127.0.0.1:8088/v1/chat/completions"
    assert await agent._main_slot_metrics() == (1, 538870)
    assert seen["url"] == "http://127.0.0.1:8088/metrics"

    def broken(url, timeout=0):
        raise OSError("down")
    monkeypatch.setattr(urllib.request, "urlopen", broken)
    assert await agent._main_slot_metrics() == (None, None)


# --- recording + wiring ---------------------------------------------------------

def test_warmup_calls_are_never_recorded_into_the_corpus_window(monkeypatch):
    from ghost_agent.core import llm as llm_mod
    from ghost_agent.core import llm_recording
    recorded = []
    monkeypatch.setattr(llm_recording, "maybe_record", lambda *a, **k: recorded.append(k.get("task_label")))
    llm_mod.LLMClient._maybe_record_call({}, {}, task_label="main-prefix-rewarm", background=True)
    llm_mod.LLMClient._maybe_record_call({}, {}, task_label="main-prefix-warmup", background=True)
    llm_mod.LLMClient._maybe_record_call({}, {}, task_label="verifier", background=True)
    assert recorded == ["verifier"]


def test_boot_wiring_spawns_the_loop_only_when_the_period_is_positive(monkeypatch):
    from ghost_agent import main as main_mod
    spawned = []

    def fake_spawn(coro, name=""):
        coro.close()
        spawned.append(name)
        return "task"
    agent = MagicMock()
    agent.rewarm_main_prefix_loop = lambda period: _coro(period)
    assert main_mod._spawn_main_prefix_rewarm(agent, fake_spawn) == "task"
    assert spawned == ["main-prefix-rewarm"]
    for off in ("0", "-1"):
        monkeypatch.setenv("GHOST_MAIN_PREFIX_REWARM_S", off)
        assert main_mod._spawn_main_prefix_rewarm(agent, fake_spawn) is None
    assert spawned == ["main-prefix-rewarm"]


async def _coro(period):
    return period


def test_boot_calls_the_wiring_inside_the_warmup_opt_out_guard():
    """§4FM review M2: the call must sit INSIDE the `if (_warm_main and …)`
    block so GHOST_MAIN_PREFIX_WARMUP=0 (or a mocked client) disables the
    loop too. Walked on the AST, not by text adjacency."""
    from ghost_agent import main as main_mod
    tree = ast.parse(inspect.getsource(main_mod))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "_warm_main" in ast.unparse(node.test):
            body_src = "\n".join(ast.unparse(n) for n in node.body)
            if "_spawn_main_prefix_rewarm(agent, _spawn_bg_main)" in body_src:
                hits.append(node.lineno)
    assert hits, "the re-warm wiring is not inside the warmup guard"
    src = inspect.getsource(main_mod)
    assert src.count("_spawn_main_prefix_rewarm(agent, _spawn_bg_main)") == 1
