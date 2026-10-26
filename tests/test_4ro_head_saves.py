"""§4RO (2026-10-25): the request head parked on disk with llama-server's slot
save/restore. Measured on b11669: save 0.68 s / 1.73 GB, restore 0.41 s, the
next request reused 28,213 of 28,219 tokens — vs ~166 s to re-process the
head. Pins: the file only becomes the head when llama saved exactly the
warm-up prompt; a restore that does not give back that count is discarded;
restore only when the head is known to be gone (named displacement or a
fresh server), never on a periodic tick; the restore runs inside the
background lane, past its re-checks; off unless GHOST_HEAD_SAVES=1."""
from __future__ import annotations

import asyncio
import io
import json
import os
import time
import urllib.error
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core import head_saves as H
from ghost_agent.core import llm as L

pytestmark = pytest.mark.asyncio

BASE = "http://127.0.0.1:8088"


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    monkeypatch.delenv(H.ENV, raising=False)
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    H._unsupported.clear()
    yield
    H._unsupported.clear()


def _dir(tmp_path):
    d = tmp_path / "system" / "head_saves"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _store(d, key, n=28213):
    (d / f"head-{key}.bin").write_bytes(b"state")
    (d / f"head-{key}.json").write_text(json.dumps({"n_tokens": n}))


def _http_error(code, text):
    return urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(text.encode()))


# ── switches and keys ───────────────────────────────────────────────────

@pytest.mark.parametrize("raw,on", [(None, False), ("0", False), ("true", False), ("1", True), (" 1 ", True)])
def test_off_unless_exactly_one(monkeypatch, raw, on):
    if raw is not None:
        monkeypatch.setenv(H.ENV, raw)
    assert H.enabled() is on


def test_the_directory_is_ghost_home_system_head_saves(monkeypatch, tmp_path):
    assert H.head_dir() == tmp_path / "system" / "head_saves"
    monkeypatch.delenv("GHOST_HOME")
    assert H.head_dir() is None


@pytest.mark.parametrize("url", [BASE, BASE + "/", BASE + "/v1", BASE + "/v1/chat/completions"])
def test_server_base_drops_the_api_path(url):
    assert H.server_base(url) == BASE


def test_the_key_follows_the_server_and_the_rendered_head():
    k = H.head_key("m|b|t", "<|im_start|>system ...")
    assert k == H.head_key("m|b|t", "<|im_start|>system ...")
    assert k != H.head_key("m2|b|t", "<|im_start|>system ...")
    assert k != H.head_key("m|b|t", "<|im_start|>system ... low")


def test_render_sends_only_what_the_template_reads(monkeypatch):
    seen = {}

    def http(url, body=None, timeout=5.0):
        seen["url"], seen["body"] = url, body
        return {"prompt": "P"}
    monkeypatch.setattr(H, "_http", http)
    p = {"model": "x", "temperature": 0.0, "max_tokens": 1, "stream": False,
         "messages": [{"role": "user", "content": "ok"}], "tools": [1],
         "chat_template_kwargs": {"reasoning_effort": "medium"}}
    assert H.render(BASE, p) == "P"
    assert seen["url"] == BASE + "/apply-template"
    assert set(seen["body"]) == {"messages", "tools", "chat_template_kwargs"}


def test_an_empty_render_is_an_error(monkeypatch):
    monkeypatch.setattr(H, "_http", lambda *a, **k: {"prompt": ""})
    with pytest.raises(ValueError):
        H.render(BASE, {"messages": []})


def test_identity_reads_model_build_and_template(monkeypatch):
    seen = {}

    def http(url, body=None, timeout=5.0):
        return {"model_path": "/m.gguf", "build_info": "b1", "chat_template": seen.get("t", "T")}
    monkeypatch.setattr(H, "_http", http)
    a = H.server_identity(BASE)
    seen["t"] = "T2"
    assert a.startswith("/m.gguf|b1|") and H.server_identity(BASE) != a


def test_a_saved_head_needs_both_files(tmp_path):
    d = _dir(tmp_path)
    assert H.stored_tokens(d, "k") is None
    (d / "head-k.bin").write_bytes(b"x")
    assert H.stored_tokens(d, "k") is None                 # no meta: an unverified file
    (d / "head-k.json").write_text(json.dumps({"n_tokens": 5}))
    assert H.stored_tokens(d, "k") == 5
    (d / "head-k.json").write_text("garbage")
    assert H.stored_tokens(d, "k") is None


@pytest.mark.parametrize("text,fresh", [
    ("llamacpp:prompt_tokens_total 0\nllamacpp:prompt_tokens_cached_total 0\nllamacpp:tokens_predicted_total 0\n", True),
    ("llamacpp:prompt_tokens_total 0\nllamacpp:prompt_tokens_cached_total 28213\nllamacpp:tokens_predicted_total 1\n", False),
    ("llamacpp:prompt_tokens_total 0\n", False),           # a metric missing: not provably fresh
])
def test_a_fresh_server_has_served_nothing(monkeypatch, text, fresh):
    monkeypatch.setattr(H, "_http", lambda *a, **k: text)
    assert H.server_fresh(BASE) is fresh


def test_an_unreachable_server_is_not_fresh(monkeypatch):
    def boom(*a, **k):
        raise OSError("down")
    monkeypatch.setattr(H, "_http", boom)
    assert H.server_fresh(BASE) is False


# ── restore ────────────────────────────────────────────────────────────

def test_restore_asks_for_the_file_name_and_checks_the_count(monkeypatch, tmp_path):
    d = _dir(tmp_path)
    _store(d, "k")
    calls = []

    def http(url, body=None, timeout=5.0):
        calls.append((url, body))
        return {"n_restored": 28213}
    monkeypatch.setattr(H, "_http", http)
    ok, detail, _ = H.restore(BASE, d, "k")
    assert ok and "28213" in detail
    assert calls == [(BASE + "/slots/0?action=restore", {"filename": "head-k.bin"})]


def test_a_restore_with_the_wrong_count_is_discarded(monkeypatch, tmp_path):
    d = _dir(tmp_path)
    _store(d, "k")
    monkeypatch.setattr(H, "_http", lambda *a, **k: {"n_restored": 999})
    ok, detail, _ = H.restore(BASE, d, "k")
    assert not ok and "999" in detail
    assert not list(d.iterdir())


def test_a_refused_restore_keeps_the_file(monkeypatch, tmp_path):
    """r1: llama's "Unable to restore" also covers a full KV cache; the
    warm-up that follows re-prefills and overwrites a really bad file."""
    d = _dir(tmp_path)
    _store(d, "k")

    def http(*a, **k):
        raise _http_error(400, '{"error":{"message":"Unable to restore slot: No available space in KV cache or invalid slot save file"}}')
    monkeypatch.setattr(H, "_http", http)
    ok, detail, _ = H.restore(BASE, d, "k")
    assert not ok and "Unable to restore" in detail
    assert H.stored_tokens(d, "k") == 28213


def test_a_timeout_keeps_the_file(monkeypatch, tmp_path):
    d = _dir(tmp_path)
    _store(d, "k")

    def http(*a, **k):
        raise TimeoutError("slow")
    monkeypatch.setattr(H, "_http", http)
    assert H.restore(BASE, d, "k")[0] is False
    assert H.stored_tokens(d, "k") == 28213


def test_no_file_no_request(monkeypatch, tmp_path):
    monkeypatch.setattr(H, "_http", lambda *a, **k: pytest.fail("no request without a file"))
    assert H.restore(BASE, _dir(tmp_path), "k") == (False, "no saved head", 0.0)


def test_a_server_without_slot_save_path_is_latched_off(monkeypatch, tmp_path):
    d = _dir(tmp_path)
    _store(d, "k")

    def http(*a, **k):
        raise _http_error(501, '{"error":{"message":"This server does not support slots action. Start it with `--slot-save-path`"}}')
    monkeypatch.setattr(H, "_http", http)
    assert H.supported(BASE)
    H.restore(BASE, d, "k")
    assert not H.supported(BASE)
    assert H.stored_tokens(d, "k") == 28213               # not the file's fault


# ── r3: park and slot size ─────────────────────────────────────────────

def test_park_sends_one_token_that_shares_nothing(monkeypatch):
    seen = {}

    def http(url, body=None, timeout=5.0):
        seen["url"], seen["body"] = url, body
        return {"content": ""}
    monkeypatch.setattr(H, "_http", http)
    assert H.park(BASE) == (True, "parked")
    assert seen["url"] == BASE + "/completion"
    assert seen["body"]["prompt"] == H.PARK_PROMPT and seen["body"]["n_predict"] == 1
    assert seen["body"]["cache_prompt"] is True       # a stateless task is never parked
    assert not H.PARK_PROMPT.startswith("<|im_start|>")


def test_a_park_error_is_reported(monkeypatch):
    def http(*a, **k):
        raise OSError("down")
    monkeypatch.setattr(H, "_http", http)
    ok, why = H.park(BASE)
    assert not ok and "down" in why


@pytest.mark.parametrize("reply,n", [([{"id": 0, "n_prompt_tokens": 28213}], 28213),
                                     ([{"id": 0}], None), ("garbage", None)])
def test_slot_tokens_reads_slot_zero(monkeypatch, reply, n):
    monkeypatch.setattr(H, "_http", lambda *a, **k: reply)
    assert H.slot_tokens(BASE) == n


@pytest.mark.parametrize("reply,busy", [([{"id": 0, "is_processing": False}], False),
                                        ([{"id": 0, "is_processing": True}], True),
                                        ([{"id": 0}], True), (OSError("down"), True)])
def test_slot_busy_is_true_unless_llama_says_idle(monkeypatch, reply, busy):
    """r4: unknown counts as busy — no park or restore on a guess."""
    def http(*a, **k):
        if isinstance(reply, Exception):
            raise reply
        return reply
    monkeypatch.setattr(H, "_http", http)
    assert H.slot_busy(BASE) is busy


# ── save ───────────────────────────────────────────────────────────────

def _llama_saves(d, n, written=1_726_869_452):
    """llama writes the .part file it was asked for and reports n_saved."""
    def http(url, body=None, timeout=5.0):
        assert url == BASE + "/slots/0?action=save"
        (d / body["filename"]).write_bytes(b"state")
        return {"n_saved": n, "n_written": written, "timings": {"save_ms": 676.6}}
    return http


def test_a_save_of_exactly_the_head_becomes_the_head(monkeypatch, tmp_path):
    d = _dir(tmp_path)
    monkeypatch.setattr(H, "_http", _llama_saves(d, 28213))
    ok, detail = H.save(BASE, d, "k", 28213, "id")
    assert ok and "1.73 GB" in detail
    assert sorted(p.name for p in d.iterdir()) == ["head-k.bin", "head-k.json"]
    meta = json.loads((d / "head-k.json").read_text())
    assert meta["n_tokens"] == 28213 and meta["identity"] == "id"


def test_a_save_that_caught_a_conversation_is_thrown_away(monkeypatch, tmp_path):
    """The slot moved on before llama ran the save (a request arrived)."""
    d = _dir(tmp_path)
    _store(d, "k", 28213)
    monkeypatch.setattr(H, "_http", _llama_saves(d, 61000))
    ok, detail = H.save(BASE, d, "k", 28213)
    assert not ok and "61000" in detail
    assert sorted(p.name for p in d.iterdir()) == ["head-k.bin", "head-k.json"]   # old head kept
    assert (d / "head-k.bin").read_bytes() == b"state"


def test_llama_saving_elsewhere_is_named(monkeypatch, tmp_path):
    d = _dir(tmp_path)
    monkeypatch.setattr(H, "_http", lambda *a, **k: {"n_saved": 28213})
    ok, detail = H.save(BASE, d, "k", 28213)
    assert not ok and "--slot-save-path" in detail


def test_no_prompt_count_no_save(monkeypatch, tmp_path):
    monkeypatch.setattr(H, "_http", lambda *a, **k: pytest.fail("no request"))
    assert H.save(BASE, _dir(tmp_path), "k", 0)[0] is False


def test_prune_keeps_the_newest_heads_and_only_touches_its_own_files(tmp_path):
    d = _dir(tmp_path)
    now = time.time()
    for i in range(5):
        _store(d, f"k{i}")
        os.utime(d / f"head-k{i}.bin", (now - 100 * (5 - i),) * 2)
    (d / "head-old.part.bin").write_bytes(b"x")
    os.utime(d / "head-old.part.bin", (now - H.STALE_PART_S - 10,) * 2)
    (d / "head-new.part.bin").write_bytes(b"x")
    (d / "notes.txt").write_text("mine")
    H.prune(d, keep=3)
    names = sorted(p.name for p in d.iterdir())
    assert names == ["head-k2.bin", "head-k2.json", "head-k3.bin", "head-k3.json",
                     "head-k4.bin", "head-k4.json", "head-new.part.bin", "notes.txt"]


# ── the client runs the restore inside the background lane ──────────────

def _bare_client():
    c = L.LLMClient.__new__(L.LLMClient)
    c.foreground_tasks = 0
    c.foreground_requests = 0
    c.main_slot_idle_windows = 0
    c._foreground_lock = asyncio.Lock()
    c._bg_queue_sem = asyncio.Semaphore(3)
    c.upstream_url = BASE
    c.worker_clients = c.critic_clients = c.vision_clients = c.swarm_clients = c.coding_clients = []
    c._maybe_record_call = lambda *a, **k: None
    return c


async def test_before_dispatch_runs_right_before_the_call():
    c = _bare_client()
    order = []

    async def before():
        order.append("restore")

    async def do(payload, *a, **k):
        order.append("call")
        return {"_ghost_leg": {"served_by": "main"}}
    c._do_chat_completion = do
    await c.chat_completion({"messages": []}, is_background=True,
                            task_label="main-prefix-rewarm", before_dispatch=before)
    assert order == ["restore", "call"]


async def test_a_refused_warmup_never_restores():
    """A request arrived while the re-warm queued: the re-check refuses it,
    and the restore (which would replace the request's slot) never runs."""
    c = _bare_client()
    c.foreground_requests = 1
    ran = []

    async def before():
        ran.append(1)

    async def wait(label):
        c.foreground_requests = 1               # the request is live past the wait
    c._wait_for_foreground_clear = wait
    c._do_chat_completion = AsyncMock()
    with pytest.raises(L.BackgroundDeferred):
        await c.chat_completion({"messages": []}, is_background=True,
                                task_label="main-prefix-rewarm", before_dispatch=before)
    assert not ran and not c._do_chat_completion.called


async def test_a_request_that_arrives_during_the_restore_refuses_the_warmup():
    """r1: llama defers a restore while the slot is busy; a request that
    started meanwhile must not get the warm-up cutting it back to the head."""
    c = _bare_client()
    ran = []

    async def before():
        ran.append(1)
        c.foreground_requests = 1
    c._do_chat_completion = AsyncMock()
    with pytest.raises(L.BackgroundDeferred):
        await c.chat_completion({"messages": []}, is_background=True,
                                task_label="main-prefix-rewarm", before_dispatch=before)
    assert ran and not c._do_chat_completion.called


# ── the warm-up ────────────────────────────────────────────────────────

def _resp(prompt, cached):
    return {"choices": [{"message": {"content": "x"}}],
            "usage": {"prompt_tokens": prompt, "prompt_tokens_details": {"cached_tokens": cached}}}


@pytest.fixture
def warm(mock_context, monkeypatch, tmp_path):
    from ghost_agent.core import agent as A
    from ghost_agent.core.agent import GhostAgent
    monkeypatch.setenv(H.ENV, "1")
    monkeypatch.delenv("GHOST_MAIN_PREFIX_REWARM_S", raising=False)   # the loop on (180 s)
    llm = MagicMock()
    llm.upstream_url = BASE + "/v1"
    llm.foreground_requests = 0
    llm._main_streams_open = 0
    llm._head_displaced = False
    llm._head_displaced_early = False
    mock_context.llm_client = llm
    st = {"fresh": False, "restores": [], "saves": [], "lines": [], "order": [],
          "slot": 28213, "park_ok": True}
    monkeypatch.setattr(H, "server_identity", lambda base: "id")
    monkeypatch.setattr(H, "slot_tokens", lambda base: st["slot"])

    st["busy"] = False
    monkeypatch.setattr(H, "slot_busy", lambda base: st["busy"])

    def park(base):
        st["order"].append("park")
        if st.get("request_during_park"):
            llm.foreground_requests = 1
        return (True, "parked") if st["park_ok"] else (False, "HTTPError 500")
    monkeypatch.setattr(H, "park", park)
    monkeypatch.setattr(H, "park_landed", lambda base: st.get("landed", False))
    monkeypatch.setattr(H, "render", lambda base, payload: st.get("rendered", "HEAD"))
    monkeypatch.setattr(H, "server_fresh", lambda base: st["fresh"])

    def restore(base, d, key):
        st["restores"].append(key)
        st["order"].append("restore")
        return (True, "28213 tokens", 0.4)

    def save(base, d, key, n, ident=""):
        st["saves"].append(n)
        _store(d, key, n)
        return (True, f"{n} tokens, 1.73 GB")
    monkeypatch.setattr(H, "restore", restore)
    monkeypatch.setattr(H, "save", save)
    monkeypatch.setattr(A, "pretty_log", lambda title, msg=None, **kw: st["lines"].append(str(msg)))

    async def cc(payload, **kw):
        st["kw"] = kw
        st["payload"] = payload
        if "fresh_at_dispatch" in st:
            st["fresh"] = st["fresh_at_dispatch"]
        if "before_dispatch" in kw:
            await kw["before_dispatch"]()
        st["sent"] = st.get("sent", 0) + 1            # r5: what reached llama
        return st.get("resp", _resp(28213, 0))
    llm.chat_completion = cc
    return GhostAgent(mock_context), llm, st



async def _warm_with_file(agent, st, **kw):
    """One warm-up to save, then the call under test."""
    await agent.warm_up_main_prefix(quiet=True)
    st["sent"] = 0
    st["restores"].clear()
    st["saves"].clear()
    st["lines"].clear()
    st["order"].clear()
    await agent.warm_up_main_prefix(**kw)


async def test_the_first_warmup_saves_the_head(warm):
    agent, llm, st = warm
    await agent.warm_up_main_prefix(quiet=True)
    assert st["saves"] == [28213] and not st["restores"] and st["sent"] == 1


async def test_with_no_file_yet_a_kept_conversation_is_not_prefilled_over(warm):
    """§4RP review r2 MAJOR: an effort change makes a new head key with no
    file; the warm-up's prefill would replace the kept conversation."""
    agent, llm, st = warm
    llm._slot_session, llm._slot_session_used = "sess-a", time.monotonic()
    st["sent"] = 0
    await agent.warm_up_main_prefix(quiet=True)
    assert st["sent"] == 0 and st["saves"] == []
    assert agent._head_rewarm_retry is True
    llm._slot_session_used = time.monotonic() - 601        # idle: it prefills and saves
    await agent.warm_up_main_prefix(quiet=True)
    assert st["sent"] == 1 and st["saves"] == [28213]


async def test_a_slot_holding_exactly_the_head_is_left_alone(warm):
    agent, llm, st = warm
    await _warm_with_file(agent, st, quiet=True)
    assert not st["order"]


@pytest.mark.parametrize("label", ["", "greeting fast path", "member turn", "deferred-verdict"])
async def test_a_slot_holding_anything_else_is_parked_then_restored(warm, label):
    """r3 (operator): the tick after an owner request re-processed the head
    for 166 s; a member's turn, a short call — all parked, then restored."""
    agent, llm, st = warm
    st["slot"] = 41000                                # head + a finished conversation
    llm._head_displaced = bool(label)                 # what the real lane requires
    await _warm_with_file(agent, st, quiet=True, displaced_by=label)
    assert st["order"] == ["park", "restore"]
    assert any("parked in llama's RAM cache first" in m for m in st["lines"])


async def test_a_park_that_fails_restores_nothing_and_sends_nothing(warm):
    """Never a restore over a conversation llama did not park. r5: and no
    prefill either — it replaces the conversation just the same and took
    211 s live, with the owner's next turn queued behind it."""
    agent, llm, st = warm
    st["slot"] = 41000
    st["park_ok"] = False
    await _warm_with_file(agent, st, quiet=True)
    assert st["order"] == ["park"] and not st["restores"]
    assert st["sent"] == 0
    assert agent._head_rewarm_retry is True
    # r5b: a quiet tick keeps a failing park off the screen (the streak shows it)
    assert not any("deferred" in m for m in st["lines"])


async def test_a_park_whose_reply_was_lost_but_which_landed_restores(warm):
    """r5 live 20:50: the client timed out while llama copied a 43k-token
    state into its RAM cache; the slot then held only the park prompt."""
    agent, llm, st = warm
    st["slot"] = 41000
    st["park_ok"] = False
    st["landed"] = True
    await _warm_with_file(agent, st, quiet=True)
    assert st["order"] == ["park", "restore"]
    assert st["sent"] == 1
    assert any("parked in llama's RAM cache first" in m for m in st["lines"])


async def test_a_request_that_starts_during_the_park_keeps_its_slot(warm):
    """r4 MAJOR: the park waited behind another call; the owner's turn
    started meanwhile — the restore would land in its tool gap."""
    agent, llm, st = warm
    st["slot"] = 41000
    st["request_during_park"] = True
    await _warm_with_file(agent, st, quiet=True)
    assert st["order"] == ["park"] and not st["restores"]
    assert st["sent"] == 0


async def test_a_busy_slot_is_neither_parked_nor_restored(warm):
    agent, llm, st = warm
    st["slot"] = 41000
    st["busy"] = True
    await _warm_with_file(agent, st, quiet=True)
    assert not st["order"]
    assert st["sent"] == 0                           # r5: no prefill over it either
    assert agent._head_rewarm_retry is True
    assert not any("deferred" in m for m in st["lines"])   # routine: not on screen


async def test_an_open_stream_blocks_the_park(warm):
    agent, llm, st = warm
    st["slot"] = 41000
    llm._main_streams_open = 1
    await agent.warm_up_main_prefix(quiet=True)       # no file yet; no save either
    llm._main_streams_open = 0
    await agent.warm_up_main_prefix(quiet=True)       # saves
    st["order"].clear()
    llm._main_streams_open = 1
    await agent.warm_up_main_prefix(quiet=True)
    assert not st["order"]


async def test_a_fresh_server_is_still_restored_when_idle(warm):
    agent, llm, st = warm
    st["fresh"] = True
    await _warm_with_file(agent, st, quiet=True)
    assert st["order"] == ["restore"]


async def test_an_unknown_slot_restores_nothing_and_sends_nothing(warm):
    """r5 review MAJOR: /slots unanswered (llama mid-batch or copying state)
    read as "leave it" and the 28k prefill queued behind that call."""
    agent, llm, st = warm
    st["slot"] = None
    await _warm_with_file(agent, st, quiet=True)
    assert not st["order"]
    assert st["sent"] == 0 and agent._head_rewarm_retry is True


async def test_a_fresh_server_is_restored_without_a_park(warm):
    agent, llm, st = warm
    st["fresh"] = True
    st["slot"] = None
    await _warm_with_file(agent, st, quiet=True)
    assert st["order"] == ["restore"]


@pytest.mark.parametrize("at_plan,at_dispatch,restored", [(True, False, False), (False, True, True)])
async def test_freshness_is_judged_at_dispatch(warm, at_plan, at_dispatch, restored):
    """r1 MAJOR: judged before the client's wait, a request that ran
    meanwhile (its conversation now in the slot) made "fresh" stale."""
    agent, llm, st = warm
    await agent.warm_up_main_prefix(quiet=True)
    st["restores"].clear()
    st["fresh"], st["fresh_at_dispatch"] = at_plan, at_dispatch
    await agent.warm_up_main_prefix()
    assert bool(st["restores"]) is restored


async def test_a_fresh_server_restores_on_a_periodic_tick_and_at_boot(warm):
    agent, llm, st = warm
    st["fresh"] = True
    await _warm_with_file(agent, st, quiet=True)
    assert len(st["restores"]) == 1
    st["restores"].clear()
    await agent.warm_up_main_prefix()                 # the boot warm-up
    assert len(st["restores"]) == 1


async def test_a_payload_that_renders_the_same_finds_the_same_file(warm, monkeypatch):
    """Live r1: the same effort level sent explicitly changed the payload
    bytes, not the render — the restore after a llama restart missed."""
    agent, llm, st = warm
    await agent.warm_up_main_prefix(quiet=True)
    monkeypatch.setattr(agent, "_warm_up_effort", lambda p: {**p, "chat_template_kwargs": {"reasoning_effort": "medium"}})
    st["restores"].clear()
    st["fresh"] = True
    await agent.warm_up_main_prefix(quiet=True)
    assert len(st["restores"]) == 1
    assert len(list(H.head_dir().glob("head-*.bin"))) == 1


async def test_a_different_render_is_a_different_head(warm):
    agent, llm, st = warm
    await agent.warm_up_main_prefix(quiet=True)
    st["rendered"] = "HEAD at low"
    st["fresh"] = True
    st["restores"].clear()
    await agent.warm_up_main_prefix(quiet=True)
    assert not st["restores"] and len(list(H.head_dir().glob("head-*.bin"))) == 2


async def test_a_resident_head_with_a_file_is_not_saved_again(warm):
    agent, llm, st = warm
    st["resp"] = _resp(28213, 28200)
    await _warm_with_file(agent, st, quiet=True)
    assert not st["saves"]


async def test_a_prefill_overwrites_a_stale_file(warm):
    """A restore that matched only part of the head (e.g. the server's
    default effort changed): the warm-up re-processed it, so save again."""
    agent, llm, st = warm
    st["slot"] = 500
    st["resp"] = _resp(28213, 900)
    await _warm_with_file(agent, st, quiet=True, displaced_by="x")
    assert st["restores"] and st["saves"] == [28213]


async def test_no_save_while_a_request_holds_the_slot(warm):
    agent, llm, st = warm
    llm.foreground_requests = 1
    await agent.warm_up_main_prefix(quiet=True)
    assert not st["saves"]
    llm.foreground_requests = 0
    llm._main_streams_open = 1
    await agent.warm_up_main_prefix(quiet=True)
    assert not st["saves"]


async def test_a_failed_restore_says_so_and_saves_again(warm, monkeypatch):
    agent, llm, st = warm
    st["slot"] = 500
    await agent.warm_up_main_prefix(quiet=True)
    d = H.head_dir()

    def bad_restore(base, dd, key):
        H.discard(dd, key)
        return (False, "Unable to restore slot", 0.1)
    monkeypatch.setattr(H, "restore", bad_restore)
    st["saves"].clear()
    await agent.warm_up_main_prefix(quiet=True, displaced_by="x")
    assert any("head restore from disk failed (Unable to restore slot)" in m for m in st["lines"])
    assert st["saves"] == [28213] and list(d.glob("head-*.bin"))


async def test_off_means_untouched(warm, monkeypatch):
    agent, llm, st = warm
    monkeypatch.delenv(H.ENV)
    await agent.warm_up_main_prefix(quiet=True, displaced_by="x")
    assert not st["saves"] and not st["restores"] and "before_dispatch" not in st["kw"]


async def test_a_server_without_slot_saves_turns_it_off_with_one_line(warm, monkeypatch):
    agent, llm, st = warm

    def unsupported_save(base, d, key, n, ident=""):
        H._unsupported.add(base)
        return (False, "This server does not support slots action")
    monkeypatch.setattr(H, "save", unsupported_save)
    await agent.warm_up_main_prefix(quiet=True)
    await agent.warm_up_main_prefix(quiet=True)
    assert sum("started without --slot-save-path" in m for m in st["lines"]) == 1


async def test_a_plan_failure_never_breaks_the_warmup(warm, monkeypatch):
    agent, llm, st = warm

    def boom(base):
        raise OSError("props down")
    monkeypatch.setattr(H, "server_identity", boom)
    await agent.warm_up_main_prefix(quiet=True)
    assert "before_dispatch" not in st["kw"] and not st["saves"]


# ── r2: the llama launcher restores at start ───────────────────────────

def _meta(d, key, identity, at, n=28213):
    (d / f"head-{key}.bin").write_bytes(b"state")
    (d / f"head-{key}.json").write_text(json.dumps({"n_tokens": n, "identity": identity, "saved_at": at}))


def test_the_newest_head_of_this_server_is_picked(tmp_path):
    d = _dir(tmp_path)
    _meta(d, "old", "id", 100)
    _meta(d, "new", "id", 200)
    _meta(d, "other", "other-model", 300)
    (d / "head-nobin.json").write_text(json.dumps({"n_tokens": 5, "identity": "id", "saved_at": 999}))
    (d / "head-junk.json").write_text("{")
    assert H.newest_head_for(d, "id") == "new"
    assert H.newest_head_for(d, "nobody") is None


def _server(monkeypatch, *, fresh=True, ready_after=0, restored=28213, props_ok=True):
    st = {"health": 0, "restores": []}

    def http(url, body=None, timeout=5.0):
        if url.endswith("/health"):
            st["health"] += 1
            if st["health"] <= ready_after:
                raise _http_error(503, "loading")
            return {"status": "ok"}
        if url.endswith("/props"):
            if not props_ok:
                raise OSError("no props")
            return {"model_path": "/m", "build_info": "b", "chat_template": "T"}
        if url.endswith("/metrics"):
            z = "0" if fresh else "5"
            return (f"llamacpp:prompt_tokens_total {z}\nllamacpp:prompt_tokens_cached_total 0\n"
                    f"llamacpp:tokens_predicted_total {z}\n")
        if "action=restore" in url:
            st["restores"].append(body["filename"])
            return {"n_restored": restored}
        raise AssertionError(url)
    monkeypatch.setattr(H, "_http", http)
    return st


def _ident():
    import hashlib
    return "/m|b|" + hashlib.sha1(b"T").hexdigest()[:16]


def test_start_restores_the_newest_head_once_llama_listens(monkeypatch, tmp_path):
    d = _dir(tmp_path)
    _meta(d, "k", _ident(), 1)
    st = _server(monkeypatch, ready_after=3)
    line = H.restore_on_start(BASE, d, wait_s=30, sleep=lambda s: None)
    assert st["health"] == 4 and st["restores"] == ["head-k.bin"]
    assert line.startswith("restored head-k (28213 tokens)")


def test_start_never_restores_over_a_served_request(monkeypatch, tmp_path):
    d = _dir(tmp_path)
    _meta(d, "k", _ident(), 1)
    st = _server(monkeypatch, fresh=False)
    assert "already served" in H.restore_on_start(BASE, d, sleep=lambda s: None)
    assert not st["restores"]


def test_start_without_a_matching_head_does_nothing(monkeypatch, tmp_path):
    d = _dir(tmp_path)
    _meta(d, "k", "another-build", 1)
    st = _server(monkeypatch)
    assert "no saved head" in H.restore_on_start(BASE, d, sleep=lambda s: None)
    assert not st["restores"]


def test_start_gives_up_on_a_llama_that_never_loads(monkeypatch, tmp_path):
    d = _dir(tmp_path)
    _meta(d, "k", _ident(), 1)
    st = _server(monkeypatch, ready_after=10**9)
    clock = {"t": 0.0}
    monkeypatch.setattr(H.time, "monotonic", lambda: clock["t"])

    def sleep(s):
        clock["t"] += s
    assert "not ready" in H.restore_on_start(BASE, d, wait_s=5, sleep=sleep)
    assert not st["restores"]


@pytest.mark.parametrize("kw,needle", [({"props_ok": False}, "cannot read /props"),
                                       ({"restored": 7}, "restore of head-k failed")])
def test_start_failures_are_one_line(monkeypatch, tmp_path, kw, needle):
    d = _dir(tmp_path)
    _meta(d, "k", _ident(), 1)
    _server(monkeypatch, **kw)
    assert needle in H.restore_on_start(BASE, d, sleep=lambda s: None)


def test_no_core_module_shadows_a_stdlib_name_the_helper_imports():
    """r4: on Python 3.10 `-I` does not imply `-P`, so the helper's own folder
    (core/) is first on sys.path: a core/json.py would replace the stdlib one."""
    import ast
    import pathlib
    core = pathlib.Path(H.__file__).parent
    names = {p.stem for p in core.glob("*.py")} | {p.name for p in core.iterdir() if p.is_dir()}
    used = set()
    for n in ast.walk(ast.parse(open(H.__file__).read())):
        if isinstance(n, ast.Import):
            used |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            used.add(n.module.split(".")[0])
    assert not (names & used), names & used


def test_the_launcher_runs_this_file_in_isolated_mode():
    """The launcher calls the file by path with -I (no PYTHONPATH, no cwd on
    sys.path): it must import nothing outside the standard library."""
    import ast
    src = open(H.__file__).read()
    mods = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            assert n.level == 0, "no relative imports"
            mods.add((n.module or "").split(".")[0])
    import sys
    assert mods <= set(sys.stdlib_module_names) | {"__future__"}


# ── r5: a park timeout is not a park failure; a deferral is retried ─────────

@pytest.mark.parametrize("slot,expect", [
    ({"is_processing": False, "n_prompt_tokens": 1}, True),
    ({"is_processing": False, "n_prompt_tokens": 4}, True),
    ({"is_processing": False, "n_prompt_tokens": 5}, False),
    ({"is_processing": True, "n_prompt_tokens": 1}, False),
    ({"is_processing": False}, False),
    ({"n_prompt_tokens": 1}, False),
])
def test_park_landed_reads_an_idle_slot_holding_only_the_park_prompt(monkeypatch, slot, expect):
    monkeypatch.setattr(H, "_http", lambda url, body=None, timeout=0: [slot])
    assert H.park_landed(BASE) is expect


def test_park_landed_unknown_is_false(monkeypatch):
    def boom(*a, **k):
        raise OSError("down")
    monkeypatch.setattr(H, "_http", boom)
    assert H.park_landed(BASE) is False


def test_the_park_waits_long_enough_for_a_big_state():
    """28 s for 43k tokens on a box in swap; the old 30 s gave up on it."""
    assert H.PARK_TIMEOUT_S >= 120


async def test_a_deferred_restore_makes_the_next_tick_retry(mock_context):
    """The periodic tick skips when prompt_tokens_total has not moved; a
    deferral sent nothing, so without a reset it would never retry."""
    from ghost_agent.core.agent import GhostAgent
    agent = GhostAgent(mock_context)
    mock_context.llm_client = MagicMock(foreground_requests=0, _head_ref=None)
    calls = []

    async def warm(quiet=False, displaced_by=""):
        calls.append(1)
        if len(calls) == 1:
            agent._head_rewarm_retry = True          # what _head_defer sets

    async def metrics():
        return (0, 500)                              # nothing moves

    ticks = []

    async def sleep(s):
        ticks.append(s)
        if len(ticks) > 3:
            raise asyncio.CancelledError

    agent.warm_up_main_prefix = warm
    with pytest.raises(asyncio.CancelledError):
        await agent.rewarm_main_prefix_loop(period_s=180, sleep=sleep, metrics=metrics)
    assert len(calls) == 2        # tick 1 deferred → tick 2 retried → tick 3 "nothing ran"


async def test_with_the_rewarm_loop_off_a_failed_park_still_prefills(warm, monkeypatch):
    """r5 review MAJOR: no loop, no retry — deferring would leave the head
    cold for good; the warm-up prefills as before r5 (and restores nothing)."""
    agent, llm, st = warm
    monkeypatch.setenv("GHOST_MAIN_PREFIX_REWARM_S", "0")
    st["slot"] = 41000
    st["park_ok"] = False
    await _warm_with_file(agent, st, quiet=True)
    assert st["order"] == ["park"] and not st["restores"]
    assert st["sent"] == 1


async def test_with_the_rewarm_loop_off_an_unknown_slot_is_not_restored_over(warm, monkeypatch):
    agent, llm, st = warm
    monkeypatch.setenv("GHOST_MAIN_PREFIX_REWARM_S", "0")
    st["slot"] = None
    await _warm_with_file(agent, st, quiet=True)
    assert not st["restores"] and st["sent"] == 1


async def test_the_boot_warmup_says_when_it_defers(warm):
    """r5 review: the boot warm-up's deferral was logger.info only."""
    agent, llm, st = warm
    st["slot"] = 41000
    st["busy"] = True
    await _warm_with_file(agent, st, quiet=False)
    assert st["sent"] == 0
    assert any("head restore deferred (the main slot is in use" in m for m in st["lines"])


async def test_deferrals_in_a_row_show_once_on_screen(warm):
    """r5 review: a counter that never drops (a leaked open stream) kept
    every tick deferring with nothing on screen."""
    from ghost_agent.core import agent as A
    agent, llm, st = warm
    await agent.warm_up_main_prefix(quiet=True)       # saves the head
    st["slot"] = 41000
    st["busy"] = True
    for _ in range(2 * A.HEAD_DEFER_LOUD_AFTER + 1):
        st["lines"].clear()
        await agent.warm_up_main_prefix(quiet=True)
        shown = [m for m in st["lines"] if "deferred" in m]
        if agent._head_defer_streak % A.HEAD_DEFER_LOUD_AFTER == 0:   # r5b: every Nth
            assert shown and f"{agent._head_defer_streak} ticks in a row" in shown[0]
        else:
            assert not shown
    st["busy"] = False                                 # the call goes out: reset
    await agent.warm_up_main_prefix(quiet=True)
    assert agent._head_defer_streak == 0


async def test_the_boot_warmup_says_when_a_park_fails(warm):
    """r5b: quiet ticks keep it off screen; the boot warm-up still says so."""
    agent, llm, st = warm
    st["slot"] = 41000
    st["park_ok"] = False
    await _warm_with_file(agent, st, quiet=False)
    assert st["sent"] == 0
    assert any("head restore deferred (could not park the slot first" in m for m in st["lines"])


# ── §4RP: the slot kept for its conversation; the head back on a switch ──

async def test_a_conversation_inside_its_window_keeps_the_slot(warm):
    """Rule 4: no park, no restore, nothing sent — and the tick retries."""
    agent, llm, st = warm
    await agent.warm_up_main_prefix(quiet=True)       # saves the head
    st["order"].clear()
    st["sent"] = 0
    st["slot"] = 38213                                 # head + a 10k conversation
    llm._slot_session, llm._slot_session_used = "sess-a", time.monotonic()
    agent._head_defer_streak = 0
    await agent.warm_up_main_prefix(quiet=True)
    assert st["order"] == [] and st["sent"] == 0
    assert agent._head_rewarm_retry is True
    assert agent._head_defer_streak == 0               # a kept slot is not a failure
    assert not any("deferred" in m for m in st["lines"])
    assert llm._slot_session == "sess-a"


async def test_after_the_window_the_head_comes_back_and_the_slot_is_released(warm):
    agent, llm, st = warm
    await agent.warm_up_main_prefix(quiet=True)
    st["order"].clear()
    st["slot"] = 38213
    llm._slot_session, llm._slot_session_used = "sess-a", time.monotonic() - 601
    await agent.warm_up_main_prefix(quiet=True)
    assert st["order"] == ["park", "restore"]
    assert llm._slot_session is None


def _turn(st, text="new question"):
    """A conversation's call: the warmed head plus its own message."""
    p = dict(st["payload"])
    p["messages"] = list(p.get("messages") or []) + [{"role": "user", "content": text}]
    return p


def _switch_env(agent, llm, st, monkeypatch, *, slot=38213, busy=False, owner="sess-a", key="sess-b"):
    from ghost_agent.utils.logging import slot_session_context
    monkeypatch.setattr(H, "_http", lambda url, body=None, timeout=5.0:
                        [{"is_processing": busy, "n_prompt_tokens": slot}])
    llm._slot_session = owner
    slot_session_context.set(key)


async def test_another_conversations_first_call_gets_the_head_back(warm, monkeypatch):
    """Rule 3, the operator's example: 28k head + 10k conversation in the
    slot, /clear, a new conversation — it must start on the 28k head."""
    agent, llm, st = warm
    await _warm_with_file(agent, st, quiet=True)      # head saved, then plan + hook installed
    assert llm.before_main_dispatch == agent._slot_switch_before
    st["order"].clear()
    st["lines"].clear()
    _switch_env(agent, llm, st, monkeypatch)
    await agent._slot_switch_before(_turn(st))
    assert st["order"] == ["restore"]                  # no park: a resumed one meets the head anyway
    assert llm._slot_session is None
    assert any("restored the head before another conversation's first call" in m for m in st["lines"])


@pytest.mark.parametrize("case", ["same", "no_owner", "no_key", "busy", "bare_head", "unknown_slot", "off"])
async def test_the_switch_does_nothing_when_it_would_not_help(warm, monkeypatch, case):
    agent, llm, st = warm
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear()
    kw = {"same": {"key": "sess-a"}, "no_owner": {"owner": None}, "no_key": {"key": ""},
          "busy": {"busy": True}, "bare_head": {"slot": 28213}, "unknown_slot": {"slot": None},
          "off": {}}[case]
    _switch_env(agent, llm, st, monkeypatch, **kw)
    if case == "off":
        monkeypatch.delenv(H.ENV)
    await agent._slot_switch_before(_turn(st))
    assert st["order"] == []


async def test_an_unreadable_slot_skips_the_switch(warm, monkeypatch):
    agent, llm, st = warm
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear()
    _switch_env(agent, llm, st, monkeypatch)

    def boom(*a, **k):
        raise OSError("timeout")
    monkeypatch.setattr(H, "_http", boom)
    await agent._slot_switch_before(_turn(st))
    assert st["order"] == []


async def test_a_failed_switch_restore_is_said_and_keeps_the_owner(warm, monkeypatch):
    agent, llm, st = warm
    await _warm_with_file(agent, st, quiet=True)
    st["lines"].clear()
    _switch_env(agent, llm, st, monkeypatch)
    monkeypatch.setattr(H, "restore", lambda base, d, key: (False, "Unable to restore", 0.1))
    await agent._slot_switch_before(_turn(st))
    assert llm._slot_session == "sess-a"
    assert any("head restore before another conversation failed" in m for m in st["lines"])
    assert any("may still apply" in m for m in st["lines"])


async def test_a_call_that_does_not_open_with_the_head_gets_no_restore(warm, monkeypatch):
    """Review r1 MAJOR: the greeting fast path, a call with its own system
    prompt — the head would be replaced at once, and the kept conversation
    lost instead of parked by llama."""
    agent, llm, st = warm
    await _warm_with_file(agent, st, quiet=True)
    _switch_env(agent, llm, st, monkeypatch)
    await agent._slot_switch_before({"messages": [{"role": "system", "content": "Greet."},
                                                  {"role": "user", "content": "hi"}]})
    assert st["order"] == []
    assert llm._slot_session == "sess-a"


async def test_another_requests_draining_stream_skips_the_switch(warm, monkeypatch):
    agent, llm, st = warm
    await _warm_with_file(agent, st, quiet=True)
    _switch_env(agent, llm, st, monkeypatch)
    llm._main_streams_open = 1
    await agent._slot_switch_before(_turn(st))
    assert st["order"] == []
    llm._main_streams_open = 0
    llm.foreground_requests = 2                       # review r3: another request live too
    await agent._slot_switch_before(_turn(st))
    assert st["order"] == []
    llm.foreground_requests = 1                       # only this request: the switch runs
    await agent._slot_switch_before(_turn(st))
    assert st["order"] == ["restore"]


async def test_no_file_for_the_current_head_restores_nothing(warm, monkeypatch):
    """Review r1: a changed head kept the previous plan; the hook reads the
    current head's file when it runs."""
    agent, llm, st = warm
    await _warm_with_file(agent, st, quiet=True)
    hs = agent._head_plan_last
    H.discard(hs["dir"], hs["key"])
    _switch_env(agent, llm, st, monkeypatch)
    await agent._slot_switch_before(_turn(st))
    assert st["order"] == []


async def test_a_restore_thread_past_its_deadline_posts_nothing(warm, monkeypatch):
    """Review r1: the client's bound cancels the wait, not a late thread."""
    from ghost_agent.core import llm as L
    agent, llm, st = warm
    await _warm_with_file(agent, st, quiet=True)
    st["lines"].clear()
    _switch_env(agent, llm, st, monkeypatch)
    monkeypatch.setattr(L, "BEFORE_MAIN_DISPATCH_S", 1.0)
    await agent._slot_switch_before(_turn(st))
    assert st["order"] == []
    assert llm._slot_session == "sess-a"
    assert any("too late to post" in m for m in st["lines"])
    assert not any("may still apply" in m for m in st["lines"])   # r3: nothing was posted


async def test_the_restore_deadline_counts_from_the_hooks_entry(warm, monkeypatch):
    """Review r2: measured after /slots, a slow read moved the deadline past
    the moment the call goes out."""
    from ghost_agent.core import llm as L
    agent, llm, st = warm
    await _warm_with_file(agent, st, quiet=True)
    st["lines"].clear()
    _switch_env(agent, llm, st, monkeypatch)
    monkeypatch.setattr(L, "BEFORE_MAIN_DISPATCH_S", H.RESTORE_TIMEOUT_S + 2.0 + 0.3)

    def slow_slots(url, body=None, timeout=5.0):
        time.sleep(0.6)
        return [{"is_processing": False, "n_prompt_tokens": 38213}]
    monkeypatch.setattr(H, "_http", slow_slots)
    await agent._slot_switch_before(_turn(st))
    assert st["order"] == []
    assert any("too late to post" in m for m in st["lines"])


async def test_a_failed_head_plan_clears_the_switch_target(warm, monkeypatch):
    """Review r2: a plan that failed (an /apply-template timeout) left the
    previous head as the switch's target while the reference moved on."""
    agent, llm, st = warm
    await _warm_with_file(agent, st, quiet=True)
    assert agent._head_plan_last is not None

    def boom(base, payload):
        raise OSError("apply-template timed out")
    monkeypatch.setattr(H, "render", boom)
    await agent.warm_up_main_prefix(quiet=True)
    assert agent._head_plan_last is None
    _switch_env(agent, llm, st, monkeypatch)
    st["order"].clear()
    await agent._slot_switch_before(_turn(st))
    assert st["order"] == []


# ── §4RQ: an outside client (Eva's World mind) used the slot ─────────────

async def test_after_an_outside_call_the_owners_call_gets_the_head_back(warm, monkeypatch):
    """Eva's look left its own prompt in the slot. The owner's next call —
    even the SAME conversation, which the switch leaves alone — must start on
    the head (~0.5 s), not re-process 28k tokens behind a foreign prefix."""
    agent, llm, st = warm
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear()
    _switch_env(agent, llm, st, monkeypatch, slot=6100, key="sess-a")   # same conversation
    llm._outside_slot_dirty = True
    await agent._slot_switch_before(_turn(st))
    assert st["order"] == ["restore"]
    assert llm._outside_slot_dirty is False and llm._slot_session is None
    assert any("restored the head over an outside call's state" in m for m in st["lines"])


async def test_the_owners_call_waits_for_the_aborting_outside_call(warm, monkeypatch):
    agent, llm, st = warm
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear()
    _switch_env(agent, llm, st, monkeypatch, slot=6100)
    reads = iter([None, {"is_processing": True, "n_prompt_tokens": 6100},
                  {"is_processing": False, "n_prompt_tokens": 6100}])

    def http(url, body=None, timeout=5.0):
        r = next(reads)
        if r is None:
            raise OSError("llama answers /slots only between batches")
        return [r]
    monkeypatch.setattr(H, "_http", http)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    llm._outside_slot_dirty = True
    await agent._slot_switch_before(_turn(st))
    assert st["order"] == ["restore"]


async def _no_sleep(s):
    return None


async def test_an_outside_call_that_does_not_leave_in_time_is_waited_behind(warm, monkeypatch):
    from ghost_agent.core import llm as L
    agent, llm, st = warm
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear()
    _switch_env(agent, llm, st, monkeypatch, slot=6100, busy=True)
    monkeypatch.setattr(L, "BEFORE_MAIN_DISPATCH_S", H.RESTORE_TIMEOUT_S + 2.0 + 0.3)
    llm._outside_slot_dirty = True
    await agent._slot_switch_before(_turn(st))
    assert st["order"] == []
    assert any("an outside call is still on the main slot" in m for m in st["lines"])


@pytest.mark.parametrize("case", ["bare_head", "not_head_call", "other_stream"])
async def test_after_an_outside_call_nothing_is_restored_when_it_would_not_help(warm, monkeypatch, case):
    agent, llm, st = warm
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear()
    _switch_env(agent, llm, st, monkeypatch, slot=28213 if case == "bare_head" else 6100)
    llm._outside_slot_dirty = True
    if case == "other_stream":
        llm._main_streams_open = 1
    payload = ({"messages": [{"role": "system", "content": "Greet."}]} if case == "not_head_call"
               else _turn(st))
    await agent._slot_switch_before(payload)
    assert st["order"] == []
    # the head already there: the mark is spent
    assert llm._outside_slot_dirty is (case != "bare_head")


async def test_the_rewarm_restores_over_an_outside_call_without_parking_it(warm):
    """Parking Eva's state in llama's RAM cache could push out a parked
    conversation of the owner's."""
    agent, llm, st = warm
    await agent.warm_up_main_prefix(quiet=True)       # saves the head
    st["order"].clear()
    st["slot"] = 6100
    llm._outside_slot_dirty = True
    await agent.warm_up_main_prefix(quiet=True)
    assert st["order"] == ["restore"]
    assert llm._outside_slot_dirty is False


def test_the_outside_mark_is_spent_by_any_agent_call_on_the_slot():
    from types import SimpleNamespace
    from ghost_agent.core.llm import note_main_slot_prompt, note_outside_call, outside_slot_dirty
    llm = SimpleNamespace(_head_ref=None)
    note_outside_call(llm, "start", "eva_world")
    assert outside_slot_dirty(llm) and llm._outside_call_live is True
    note_outside_call(llm, "end", "eva_world")
    assert outside_slot_dirty(llm) and llm._outside_call_live is False
    note_main_slot_prompt(llm, {"messages": []}, label="a member's turn")
    assert not outside_slot_dirty(llm)


def test_ending_an_outside_call_marks_the_head_displaced_for_an_early_rewarm():
    from types import SimpleNamespace
    from ghost_agent.core.llm import note_outside_call
    llm = SimpleNamespace(_head_ref=(10, "h", 0, "t"), _slot_session="sess-a", _slot_session_used=1.0)
    note_outside_call(llm, "end", "eva_world")
    assert llm._head_displaced is True and llm._head_displaced_early is True
    assert llm._head_displaced_by == "outside call: eva_world"
    assert llm._slot_session == "sess-a"            # an outside call owns nothing


@pytest.mark.parametrize("meanwhile", ["second_request", "mark_spent", "stream_opened"])
async def test_what_happens_during_the_wait_is_read_again_before_the_restore(warm, monkeypatch, meanwhile):
    """Review r2 MAJOR: another request's call ran while this hook waited
    behind the outside call and sits in its tool gap — its conversation is in
    the slot now, not the outside call's."""
    agent, llm, st = warm
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear()
    _switch_env(agent, llm, st, monkeypatch, slot=6100)
    reads = iter([{"is_processing": True, "n_prompt_tokens": 6100},
                  {"is_processing": False, "n_prompt_tokens": 9400}])

    def http(url, body=None, timeout=5.0):
        r = next(reads)
        if r["is_processing"]:          # the other request gets its turn now
            if meanwhile == "second_request":
                llm.foreground_requests = 2
            elif meanwhile == "mark_spent":
                llm._outside_slot_dirty = False
            else:
                llm._main_streams_open = 1
        return [r]
    monkeypatch.setattr(H, "_http", http)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    llm.foreground_requests = 1
    llm._outside_slot_dirty = True
    await agent._slot_switch_before(_turn(st))
    assert st["order"] == []


async def test_a_slot_that_does_not_say_whether_it_is_busy_is_left_alone(warm, monkeypatch):
    agent, llm, st = warm
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear()
    monkeypatch.setattr(H, "_http", lambda url, body=None, timeout=5.0: [{"n_prompt_tokens": 6100}])
    llm._outside_slot_dirty = True
    await agent._slot_switch_before(_turn(st))
    assert st["order"] == []


async def test_the_rewarm_never_restores_over_a_kept_conversation_even_with_the_mark(warm):
    agent, llm, st = warm
    await agent.warm_up_main_prefix(quiet=True)
    st["order"].clear()
    st["slot"] = 38213
    llm._slot_session, llm._slot_session_used = "sess-a", time.monotonic()
    llm._outside_slot_dirty = True
    await agent.warm_up_main_prefix(quiet=True)
    assert st["order"] == []


def test_an_end_after_an_agent_call_since_the_start_sets_no_mark():
    """Review r2 MAJOR: the owner's request ran first (the outside call then
    cancelled behind it) — the slot may hold the owner's conversation."""
    from types import SimpleNamespace
    from ghost_agent.core.llm import note_main_slot_prompt, note_outside_call, outside_slot_dirty
    llm = SimpleNamespace(_head_ref=(10, "h", 0, "t"))
    note_outside_call(llm, "start", "eva_world")
    note_main_slot_prompt(llm, {"messages": []}, label="stream")     # the owner's call
    note_outside_call(llm, "end", "eva_world")
    assert not outside_slot_dirty(llm)
    # and with nothing in between, the end does mark it
    note_outside_call(llm, "start", "eva_world")
    note_outside_call(llm, "end", "eva_world")
    assert outside_slot_dirty(llm)
