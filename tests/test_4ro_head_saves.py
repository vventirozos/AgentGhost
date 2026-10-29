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
from ghost_agent.utils.logging import Icons

pytestmark = pytest.mark.asyncio

BASE = "http://127.0.0.1:8088"


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    monkeypatch.delenv(H.ENV, raising=False)
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    H._unsupported.clear()

    def _no_live_llama(url, body=None, timeout=5.0):
        # review 10-27: a test that forgot its stub posted a real slot save to
        # the live llama (a 1.7 GB .part file in the live directory)
        raise AssertionError(f"test reached the network: {url}")
    monkeypatch.setattr(H, "_http", _no_live_llama)
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


def test_a_restore_marks_the_file_used(monkeypatch, tmp_path):
    """§4RW-e (review): prune keeps by last use, and the head every boot
    restores was never marked used — the oldest save, so the first pruned."""
    import os
    d = _dir(tmp_path)
    _store(d, "k")
    os.utime(d / "head-k.json", (1000, 1000))
    monkeypatch.setattr(H, "_http", lambda *a, **k: {"n_restored": 28213})
    assert H.restore(BASE, d, "k")[0]
    assert (d / "head-k.json").stat().st_mtime > 1000


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
    # §4RU-r: the switch waits for a busy slot up to the hook's bound less the
    # restore's — 1 s here, so a slot that stays busy gives up quickly
    monkeypatch.setattr(L, "BEFORE_MAIN_DISPATCH_S", H.RESTORE_TIMEOUT_S + 3.0)
    llm = MagicMock()
    llm.upstream_url = BASE + "/v1"
    llm.foreground_requests = 0
    llm._main_streams_open = 0
    llm._head_displaced = False
    llm._head_displaced_early = False
    llm._slot_moved_on = False          # §4SA: whose state the slot holds
    mock_context.llm_client = llm
    st = {"fresh": False, "restores": [], "saves": [], "lines": [], "order": [],
          "slot": 28213, "park_ok": True}
    monkeypatch.setattr(H, "server_identity", lambda base: "id")
    monkeypatch.setattr(H, "slot_tokens", lambda base: st["slot"])

    st["busy"] = False
    monkeypatch.setattr(H, "slot_busy", lambda base: st["busy"])

    def park(base, timeout=H.PARK_TIMEOUT_S):
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
    def _log(title, msg=None, **kw):
        st["lines"].append(str(msg))
        st["logged"].append((title, str(msg), kw.get("icon")))     # §4RW: the icon per event
    st["logged"] = []
    monkeypatch.setattr(A, "pretty_log", _log)

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
    st["logged"].clear()
    st["order"].clear()
    await agent.warm_up_main_prefix(**kw)


async def test_the_first_warmup_saves_the_head(warm):
    agent, llm, st = warm
    await agent.warm_up_main_prefix(quiet=True)
    assert st["saves"] == [28213] and not st["restores"] and st["sent"] == 1
    assert [(t, i) for t, m, i in st["logged"] if "saved the head" in m] == [("Head Disk", Icons.HEAD_SAVE)]   # §4RW


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
    assert any(i == Icons.HEAD_RESTORE and t == "Head Disk" for t, m, i in st["logged"]
               if "restored the head" in m)                                    # §4RW


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
    st["slot"] = 30500                                   # the request's call reached it
    await agent.warm_up_main_prefix(quiet=True)
    assert not st["saves"]
    assert any(i == Icons.HEAD_FAIL and "head not saved" in m for t, m, i in st["logged"])


async def test_a_live_request_that_has_not_reached_the_slot_lets_the_head_be_saved(warm):
    """§4RW review: the slot still holds exactly the head — saving it is safe
    (llama's exact count refuses a call that slipped in first)."""
    agent, llm, st = warm
    llm.foreground_requests = 1
    st["slot"] = 28213
    await agent.warm_up_main_prefix(quiet=True)
    assert st["saves"] == [28213]


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
    assert any("head restore failed (Unable to restore slot)" in m for m in st["lines"])
    assert [i for t, m, i in st["logged"] if "head restore failed" in m] == [Icons.HEAD_FAIL]   # §4RW
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
    assert [i for t, m, i in st["logged"] if "--slot-save-path" in m] == [Icons.HEAD_FAIL]      # §4RW


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
    assert any("head restore deferred — the main slot is in use" in m for m in st["lines"])
    assert [i for t, m, i in st["logged"] if "deferred" in m] == [Icons.HEAD_DEFER]            # §4RW


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
    assert any("head restore deferred — could not park the slot first" in m for m in st["lines"])


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


@pytest.mark.parametrize("case", ["same", "no_key", "busy", "bare_head", "unknown_slot", "off"])
async def test_the_switch_does_nothing_when_it_would_not_help(warm, monkeypatch, case):
    agent, llm, st = warm
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear()
    kw = {"same": {"key": "sess-a"}, "no_key": {"key": ""},
          "busy": {"busy": True}, "bare_head": {"slot": 28213}, "unknown_slot": {"slot": None},
          "off": {}}[case]
    _switch_env(agent, llm, st, monkeypatch, **kw)
    if case == "off":
        monkeypatch.delenv(H.ENV)
    await agent._slot_switch_before(_turn(st))
    assert st["order"] == []


async def test_the_same_conversations_next_request_gets_the_head_back(warm, monkeypatch):
    """§4RP-b, req 5a58a6ca: a follow-up's first message is relabelled, so
    its prompt leaves the kept one just past the head — with the head's
    checkpoint pushed out llama re-processed all 30k tokens (206 s)."""
    from ghost_agent.utils.logging import request_id_context
    agent, llm, st = warm
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear()
    st["lines"].clear()
    _switch_env(agent, llm, st, monkeypatch, key="sess-a")
    llm._slot_request = "req-1"
    tok = request_id_context.set("req-1")
    try:
        await agent._slot_switch_before(_turn(st))
        assert st["order"] == []                       # the holding request's own calls keep it
        request_id_context.set("req-2")
        await agent._slot_switch_before(_turn(st))
        assert st["order"] == ["restore"]
        assert llm._slot_session is None and llm._slot_request is None
        assert any("restored the head before this conversation's next request" in m for m in st["lines"])
    finally:
        request_id_context.reset(tok)


@pytest.mark.parametrize("held_by", [None, ""])
async def test_an_unknown_holding_request_keeps_the_slot(warm, monkeypatch, held_by):
    from ghost_agent.utils.logging import request_id_context
    agent, llm, st = warm
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear()
    _switch_env(agent, llm, st, monkeypatch, key="sess-a")
    llm._slot_request = held_by
    tok = request_id_context.set("req-2")
    try:
        await agent._slot_switch_before(_turn(st))
    finally:
        request_id_context.reset(tok)
    assert st["order"] == []


async def test_a_failed_restore_for_the_next_request_names_it(warm, monkeypatch):
    from ghost_agent.utils.logging import request_id_context
    agent, llm, st = warm
    await _warm_with_file(agent, st, quiet=True)
    st["lines"].clear()
    _switch_env(agent, llm, st, monkeypatch, key="sess-a")
    llm._slot_request = "req-1"
    monkeypatch.setattr(H, "restore", lambda base, d, key: (False, "Unable to restore", 0.1))
    tok = request_id_context.set("req-2")
    try:
        await agent._slot_switch_before(_turn(st))
    finally:
        request_id_context.reset(tok)
    assert llm._slot_session == "sess-a" and llm._slot_request == "req-1"
    assert any("head restore before this conversation's next request failed" in m for m in st["lines"])


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


async def test_a_restore_does_not_spend_the_mark_while_an_outside_call_is_live(warm, monkeypatch):
    """Review 10-26: a background caller's restore is not gated by the
    /outside-call start; a look that started meanwhile replaces the head again."""
    agent, llm, st = warm
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear()
    _switch_env(agent, llm, st, monkeypatch, slot=6100)
    llm._outside_slot_dirty, llm._outside_call_live = True, True
    await agent._slot_switch_before(_turn(st))
    assert st["order"] == ["restore"] and llm._outside_slot_dirty is True


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
    assert [i for t, m, i in st["logged"] if "outside call" in m] == [Icons.HEAD_FAIL]        # §4RW


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


def test_ending_an_outside_call_marks_the_head_displaced_but_no_early_rewarm_runs():
    """2026-10-26 (operator): the early re-warm restored the head ~10 s after
    every look, and her next look re-read its whole prompt (3,300 tokens,
    12-17 s). The head stays out until a call needs it."""
    from types import SimpleNamespace
    from ghost_agent.core.llm import note_outside_call, head_rewarm_due, note_main_slot_prompt
    llm = SimpleNamespace(_head_ref=(10, "h", 0, "t"), _slot_session="sess-a", _slot_session_used=1.0,
                          foreground_requests=0, foreground_tasks=0, _main_streams_open=0)
    note_outside_call(llm, "end", "eva_world")
    assert llm._head_displaced is True and llm._head_displaced_early is True
    assert llm._head_displaced_by == "outside call: eva_world"
    assert llm._slot_session == "sess-a"            # an outside call owns nothing
    llm._last_main_call_end = 0.0
    assert head_rewarm_due(llm, 0.0) is False       # its state stays in the slot
    note_main_slot_prompt(llm, {"messages": []}, label="a member's turn", background=True)
    assert head_rewarm_due(llm, 0.0) is True        # an agent call spent the mark: the normal rules


async def test_the_rewarm_loop_leaves_an_outside_calls_state_until_the_mark_is_spent(mock_context):
    """Periodic ticks do not restore over it either; once an agent call
    takes the slot, the next tick re-warms as before."""
    from ghost_agent.core.agent import GhostAgent
    agent = GhostAgent(mock_context)
    llm = MagicMock(foreground_requests=0, _head_ref=None, _outside_slot_dirty=True)
    mock_context.llm_client = llm
    calls, ticks = [], []

    async def warm(quiet=False, displaced_by=""):
        calls.append(len(ticks))

    async def metrics():
        return (0, 500 + len(ticks))                 # something always "ran"

    async def sleep(s):
        ticks.append(s)
        if len(ticks) == 3:
            llm._outside_slot_dirty = False          # an agent call took the slot
        if len(ticks) > 4:
            raise asyncio.CancelledError

    agent.warm_up_main_prefix = warm
    with pytest.raises(asyncio.CancelledError):
        await agent.rewarm_main_prefix_loop(period_s=180, sleep=sleep, metrics=metrics)
    assert calls and calls[0] == 3                   # ticks 1-2 left her state alone


async def test_after_skipping_for_an_outside_call_a_tick_with_nothing_new_stays_quiet(mock_context):
    """The skipped tick records the token total: when the mark is then spent by
    a request's head restore alone (nothing counted), the next periodic tick
    reads "nothing ran" instead of re-warming the head it just got back."""
    from ghost_agent.core.agent import GhostAgent
    agent = GhostAgent(mock_context)
    llm = MagicMock(foreground_requests=0, _head_ref=None, _outside_slot_dirty=True)
    mock_context.llm_client = llm
    calls, ticks = [], []

    async def warm(quiet=False, displaced_by=""):
        calls.append(len(ticks))

    async def metrics():
        return (0, 700)                              # nothing counted after the look

    async def sleep(s):
        ticks.append(s)
        if len(ticks) == 2:
            llm._outside_slot_dirty = False          # the head came back by restore
        if len(ticks) > 3:
            raise asyncio.CancelledError

    agent.warm_up_main_prefix = warm
    with pytest.raises(asyncio.CancelledError):
        await agent.rewarm_main_prefix_loop(period_s=180, sleep=sleep, metrics=metrics)
    assert calls == []


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


# ── §4RU: the channel members' head, its own file ───────────────────────

def _kmeta(d, key, kind, at, identity="id", n=8000):
    (d / f"head-{key}.bin").write_bytes(b"state")
    m = {"n_tokens": n, "identity": identity, "saved_at": at}
    if kind:
        m["kind"] = kind
    (d / f"head-{key}.json").write_text(json.dumps(m))


def test_a_save_names_its_kind(monkeypatch, tmp_path):
    d = _dir(tmp_path)
    monkeypatch.setattr(H, "_http", _llama_saves(d, 8000))
    ok, _ = H.save(BASE, d, "m1", 8000, "id", H.MEMBER)
    assert ok and json.loads((d / "head-m1.json").read_text())["kind"] == "member"
    assert H.kind_of(d, "m1") == H.MEMBER and H.has_kind(d, H.MEMBER)


def test_a_head_saved_before_kinds_is_the_owners(tmp_path):
    d = _dir(tmp_path)
    _kmeta(d, "old", None, 100)
    assert H.kind_of(d, "old") == H.OWNER and not H.has_kind(d, H.MEMBER)


def test_llama_start_restores_the_owners_head_never_the_members(tmp_path):
    d = _dir(tmp_path)
    _kmeta(d, "own", None, 100)
    _kmeta(d, "mem", "member", 200)                  # newer, but the members'
    assert H.newest_head_for(d, "id") == "own"
    assert H.newest_head_for(d, "id", H.MEMBER) == "mem"


def test_prune_keeps_each_kinds_newest(tmp_path):
    d = _dir(tmp_path)
    now = time.time()
    _kmeta(d, "mem", "member", 1)
    os.utime(d / "head-mem.bin", (now - 10_000,) * 2)  # oldest file of all
    for i in range(4):
        _kmeta(d, f"o{i}", None, i)
        os.utime(d / f"head-o{i}.bin", (now - 100 * (4 - i),) * 2)
    H.prune(d, keep=3)
    names = sorted(p.name for p in d.glob("head-*.bin"))
    assert names == ["head-mem.bin", "head-o1.bin", "head-o2.bin", "head-o3.bin"]


class _RS:
    """A request state with three tools, two of them the owner's."""
    async def get_profile_str(self):
        return "OWNER PROFILE"

    def get_active_tool_defs(self, query):
        return [{"type": "function", "function": {"name": n}}
                for n in ("file_system", "web_search", "image_generation", "manage_projects")]


async def test_the_members_head_has_the_placeholder_and_only_their_tools(warm):
    from ghost_agent.core.agent import _MEMBER_ALLOWED_TOOLS, _MEMBER_PROFILE_PLACEHOLDER
    agent, llm, st = warm
    agent.context.args.native_tools = True
    agent._warm_up_effort = lambda p: {**p, "effort": "low"}
    owner, _, _ = await agent._head_payload(_RS())
    member, _, msys = await agent._head_payload(_RS(), member=True)
    assert "OWNER PROFILE" in owner["messages"][0]["content"] and owner["effort"] == "low"
    assert _MEMBER_PROFILE_PLACEHOLDER in msys and "OWNER PROFILE" not in msys
    assert [t["function"]["name"] for t in member["tools"]] == ["web_search", "image_generation"]
    assert all(t["function"]["name"] in _MEMBER_ALLOWED_TOOLS for t in member["tools"])
    assert "effort" not in member                    # a member's turn runs at the default


@pytest.fixture
def members(warm, monkeypatch):
    """The owner's head saved; renders tell the two heads apart; saves keep
    their kind; the request is a channel member's."""
    from ghost_agent.core.agent import _MEMBER_PROFILE_PLACEHOLDER
    from ghost_agent.utils.logging import requester_role_context, request_id_context
    agent, llm, st = warm
    monkeypatch.setattr(H, "render", lambda base, p: (
        "MEMBERS" if _MEMBER_PROFILE_PLACEHOLDER in p["messages"][0]["content"] else "HEAD"))
    st["kinds"] = []

    def save(base, d, key, n, ident="", kind=H.OWNER):
        st["saves"].append(n)
        st["kinds"].append(kind)
        _store(d, key, n)
        return (True, f"{n} tokens, 0.5 GB")
    monkeypatch.setattr(H, "save", save)
    # the member park needs ≥5 s of the deadline (§4RU-r)
    monkeypatch.setattr(L, "BEFORE_MAIN_DISPATCH_S", H.RESTORE_TIMEOUT_S + 2.0 + 30.0)
    tokens = [requester_role_context.set("member"), request_id_context.set("slack-1")]
    yield agent, llm, st
    request_id_context.reset(tokens[1])
    requester_role_context.reset(tokens[0])


async def _members_file(agent, llm, st):
    await agent.warm_up_main_prefix(quiet=True)          # the owner's head saved
    hs = await agent._member_head_plan(llm)
    _store(hs["dir"], hs["key"], 8000)
    st["order"].clear()
    st["restores"].clear()
    st["lines"].clear()
    return hs


def _member_turn(hs, text="hello from a member"):
    p = dict(hs["payload"])
    p["messages"] = list(p["messages"]) + [{"role": "user", "content": text}]
    return p


async def test_a_members_first_call_gets_the_members_head_back(members, monkeypatch):
    from ghost_agent.utils.logging import request_id_context
    agent, llm, st = members
    hs = await _members_file(agent, llm, st)
    _switch_env(agent, llm, st, monkeypatch, slot=28213, owner="sess-a", key="slack-thread")
    await agent._slot_switch_before(_member_turn(hs))
    assert st["restores"] == [hs["key"]]
    assert llm._slot_session == "sess-a"                 # the member never owns the slot
    assert any("restored the members' head before a channel member's request" in m for m in st["lines"])
    assert [i for t, m, i in st["logged"] if "members' head" in m] == [Icons.HEAD_RESTORE]     # §4RW
    await agent._slot_switch_before(_member_turn(hs, "its second call"))
    assert st["restores"] == [hs["key"]]                 # the request's later calls keep it
    request_id_context.set("slack-2")
    await agent._slot_switch_before(_member_turn(hs, "the next request"))
    assert st["restores"] == [hs["key"], hs["key"]]


@pytest.mark.parametrize("case", ["not_members_head", "holds_it", "busy", "off"])   # no_file: §4RW makes it
async def test_the_members_restore_does_nothing_when_it_would_not_help(members, monkeypatch, case):
    agent, llm, st = members
    hs = await _members_file(agent, llm, st)
    _switch_env(agent, llm, st, monkeypatch, slot=8000 if case == "holds_it" else 28213,
                busy=case == "busy")
    if case == "busy":
        monkeypatch.setattr(L, "BEFORE_MAIN_DISPATCH_S", H.RESTORE_TIMEOUT_S + 3.0)   # give up fast
    if case == "no_file":
        H.discard(hs["dir"], hs["key"])
    if case == "other_stream":
        llm._main_streams_open = 1
    if case == "off":
        monkeypatch.delenv(H.ENV)
    payload = _turn(st) if case == "not_members_head" else _member_turn(hs)
    await agent._slot_switch_before(payload)
    assert st["restores"] == [] and st["order"] == []


async def test_after_a_members_turn_the_owners_call_gets_the_head_back_with_no_owner(warm, monkeypatch):
    """§4RU: the head came back (no owner), then a member's turn replaced it —
    the owner's next call must not re-process the 28k head."""
    agent, llm, st = warm
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear()
    _switch_env(agent, llm, st, monkeypatch, owner=None, key="sess-a")
    llm._head_displaced = llm._slot_moved_on = True
    await agent._slot_switch_before(_turn(st))
    assert st["order"] == ["restore"]


@pytest.fixture
def quiet_agent(members):
    import datetime as _dt
    agent, llm, st = members
    agent.context.last_activity_time = _dt.datetime.now() - _dt.timedelta(seconds=3600)
    llm._slot_session = None
    llm._outside_slot_dirty = False
    return members


async def test_a_quiet_agent_makes_the_members_file_then_puts_the_owners_head_back(quiet_agent):
    from ghost_agent.utils.logging import requester_role_context
    agent, llm, st = quiet_agent
    requester_role_context.set("")                        # the re-warm loop's context
    await agent.warm_up_main_prefix(quiet=True)           # the owner's head saved
    st["saves"].clear(); st["kinds"].clear(); st["restores"].clear()
    agent._member_head_wanted = True
    st["resp"] = _resp(8000, 0)
    llm._head_displaced = True
    assert await agent._member_head_refresh() is True
    assert st["kw"]["task_label"] == "main-prefix-members"
    from ghost_agent.core.agent import _MEMBER_PROFILE_PLACEHOLDER
    assert _MEMBER_PROFILE_PLACEHOLDER in st["payload"]["messages"][0]["content"]
    assert st["kinds"] == [H.MEMBER] and st["saves"] == [8000]
    assert st["restores"] == [agent._head_plan_last["key"]]   # the owner's head, no park
    assert llm._head_displaced is False
    hs = await agent._member_head_plan(llm)
    assert hs["have"]
    assert await agent._member_head_refresh() is False    # made once


@pytest.mark.parametrize("case", ["not_wanted", "recent_request", "request_live", "kept", "no_owner_file"])
async def test_the_members_file_waits_for_a_quiet_agent(quiet_agent, case):
    import datetime as _dt
    from ghost_agent.utils.logging import requester_role_context
    agent, llm, st = quiet_agent
    requester_role_context.set("")
    await agent.warm_up_main_prefix(quiet=True)
    st["sent"] = 0
    agent._member_head_wanted = case != "not_wanted"
    if case == "recent_request":
        agent.context.last_activity_time = _dt.datetime.now()
    if case == "request_live":
        llm.foreground_requests = 1
    if case == "kept":
        llm._slot_session = "sess-a"
        llm._slot_session_used = time.monotonic()
    if case == "no_owner_file":
        H.discard(agent._head_plan_last["dir"], agent._head_plan_last["key"])
    assert await agent._member_head_refresh() is False
    assert st["sent"] == 0


async def test_a_members_file_saved_before_is_kept_fresh_after_a_restart(quiet_agent, tmp_path):
    from ghost_agent.utils.logging import requester_role_context
    agent, llm, st = quiet_agent
    requester_role_context.set("")
    await agent.warm_up_main_prefix(quiet=True)
    _kmeta(_dir(tmp_path), "oldmembers", "member", time.time() - 86400)   # a previous prompt's file
    st["resp"] = _resp(8000, 0)
    assert await agent._member_head_refresh() is True
    assert st["kinds"][-1] == H.MEMBER


async def test_a_refresh_that_a_request_beat_is_tried_again(quiet_agent):
    from ghost_agent.utils.logging import requester_role_context
    agent, llm, st = quiet_agent
    requester_role_context.set("")
    await agent.warm_up_main_prefix(quiet=True)
    agent._member_head_wanted = True

    async def deferred(payload, **kw):
        raise L.BackgroundDeferred("a request started")
    llm.chat_completion = deferred
    assert await agent._member_head_refresh() is False
    assert agent._member_head_tried is None


async def test_a_request_during_the_members_prefill_is_tried_again(quiet_agent):
    """Review: a request that came while the members' head was processed
    skipped the save — it must not count as the one try for that key."""
    from ghost_agent.utils.logging import requester_role_context
    agent, llm, st = quiet_agent
    requester_role_context.set("")
    await agent.warm_up_main_prefix(quiet=True)
    st["saves"].clear()
    agent._member_head_wanted = True
    real = llm.chat_completion

    async def a_request_arrives(payload, **kw):
        out = await real(payload, **kw)
        llm.foreground_requests = 1
        return out
    llm.chat_completion = a_request_arrives
    st["resp"] = _resp(8000, 0)
    assert await agent._member_head_refresh() is True
    assert st["saves"] == [] and agent._member_head_tried is None


async def test_a_members_call_restores_its_head_over_an_outside_calls_state(members, monkeypatch):
    agent, llm, st = members
    hs = await _members_file(agent, llm, st)
    _switch_env(agent, llm, st, monkeypatch, slot=3300, owner="sess-a", key="slack-thread")
    llm._outside_slot_dirty = True
    await agent._slot_switch_before(_member_turn(hs))
    assert st["restores"] == [hs["key"]]


async def test_a_draining_reply_or_a_queued_request_is_waited_out_not_a_reason_to_skip(warm, monkeypatch):
    """§4RU-r (review 10-27): turns hold one lock, so the previous turn's reply
    still draining and a request queued behind this one counted as "live" and
    the switch gave up — the call then re-processed from token 0. Now the
    switch waits for llama's slot to go idle and restores."""
    agent, llm, st = warm
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear()
    _switch_env(agent, llm, st, monkeypatch)
    reads = []

    def http(url, body=None, timeout=5.0):
        reads.append(url)
        return [{"is_processing": len(reads) < 3, "n_prompt_tokens": 38213}]
    monkeypatch.setattr(H, "_http", http)
    monkeypatch.setattr(L, "BEFORE_MAIN_DISPATCH_S", H.RESTORE_TIMEOUT_S + 10.0)   # room for two polls
    llm._main_streams_open = 1
    llm.foreground_requests = 3
    await agent._slot_switch_before(_turn(st))
    assert st["order"] == ["restore"] and len(reads) == 3


async def test_a_call_outside_the_running_turn_gets_no_restore(warm, monkeypatch):
    """The guard the counters stood for: a turn sits in its tool gap and a
    call that is not that turn's must not swap the slot under it."""
    from ghost_agent.utils.logging import request_id_context
    agent, llm, st = warm
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear()
    _switch_env(agent, llm, st, monkeypatch)
    tok = request_id_context.set("outsider")
    try:
        async with agent._turn_lock("turn-1"):
            await agent._slot_switch_before(_turn(st))
            assert st["order"] == []
            request_id_context.set("turn-1")          # the lock holder's own call
            await agent._slot_switch_before(_turn(st))
            assert st["order"] == ["restore"]
    finally:
        request_id_context.reset(tok)


async def test_with_no_owner_the_requests_first_call_restores_once(warm, monkeypatch):
    """Review 10-27: an unowned request (a probe, a scheduled task) ran on the
    head and claimed nothing — the owner's next request found no owner and
    re-processed from 0. Once per request: its later calls keep the slot."""
    from ghost_agent.utils.logging import request_id_context
    agent, llm, st = warm
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear()
    _switch_env(agent, llm, st, monkeypatch, owner=None, key="sess-a")
    tok = request_id_context.set("req-9")
    try:
        await agent._slot_switch_before(_turn(st))
        await agent._slot_switch_before(_turn(st, "its second call"))
    finally:
        request_id_context.reset(tok)
    assert st["order"] == ["restore"]


def _with_tools(p):
    p = dict(p)
    p["tools"] = [{"type": "function", "function": {"name": n}} for n in ("web_search", "file_system")]
    return p


async def test_the_owner_on_slack_shares_the_tools_and_gets_the_head_back_once(warm, monkeypatch):
    """Review 10-27, slack-be: the owner's Slack system text differs (public
    profile), so the call never 'opened with the head' and re-processed 28.9k
    tokens; on a restored head slack-de reused 27.7k (the tools come first)."""
    from ghost_agent.utils.logging import request_id_context
    agent, llm, st = warm
    await _warm_with_file(agent, st, quiet=True)
    head = _with_tools(st["payload"])
    L.note_head_reference(llm, head)
    st["order"].clear()
    _switch_env(agent, llm, st, monkeypatch)
    slack = dict(head)
    slack["messages"] = [{"role": "system", "content": "a public-surface system text"},
                         {"role": "user", "content": "hi from slack"}]
    tok = request_id_context.set("slack-1")
    try:
        await agent._slot_switch_before(slack)
        await agent._slot_switch_before(slack)
    finally:
        request_id_context.reset(tok)
    assert st["order"] == ["restore"]


async def test_a_call_with_another_tool_list_gets_no_restore(warm, monkeypatch):
    agent, llm, st = warm
    await _warm_with_file(agent, st, quiet=True)
    L.note_head_reference(llm, _with_tools(st["payload"]))
    st["order"].clear()
    _switch_env(agent, llm, st, monkeypatch)
    other = _turn(st)
    other["messages"][0] = {"role": "system", "content": "another prompt"}
    other["tools"] = [{"type": "function", "function": {"name": "x"}}]
    await agent._slot_switch_before(other)
    assert st["order"] == []


async def test_a_llama_now_running_another_model_gets_no_restore(warm, monkeypatch):
    """Review 10-27: the plan is a tick old; llama restarted on another model of
    the same shape would take the old file's state silently."""
    agent, llm, st = warm
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear()
    _switch_env(agent, llm, st, monkeypatch)
    monkeypatch.setattr(H, "server_identity", lambda base: "another-model")
    await agent._slot_switch_before(_turn(st))
    assert st["order"] == []


async def test_an_old_members_file_no_longer_means_members_write(quiet_agent, tmp_path):
    """Review 10-27: one old file made every prompt-changing deploy pay a
    ~50 s members' prefill forever."""
    from ghost_agent.core.agent import MEMBER_HEAD_RECENT_S
    from ghost_agent.utils.logging import requester_role_context
    agent, llm, st = quiet_agent
    requester_role_context.set("")
    await agent.warm_up_main_prefix(quiet=True)
    st["sent"] = 0
    _kmeta(_dir(tmp_path), "oldmembers", "member", time.time() - MEMBER_HEAD_RECENT_S - 60)
    old = time.time() - MEMBER_HEAD_RECENT_S - 60
    os.utime(_dir(tmp_path) / "head-oldmembers.json", (old, old))   # not used since either
    assert await agent._member_head_refresh() is False and st["sent"] == 0


async def test_a_member_restore_parks_the_slot_first_unless_it_holds_the_owners_head(members, monkeypatch):
    """Review 10-27: before §4RU llama parked whatever the slot held when a
    member's prompt arrived (a background job between calls resumed from it);
    the restore must not drop it. The owner's bare head is on disk: no park."""
    from ghost_agent.utils.logging import request_id_context
    agent, llm, st = members
    hs = await _members_file(agent, llm, st)
    _switch_env(agent, llm, st, monkeypatch, slot=40000, owner="sess-a", key="slack-thread")
    await agent._slot_switch_before(_member_turn(hs))
    assert st["order"] == ["park", "restore"]
    st["order"].clear()
    request_id_context.set("slack-2")
    _switch_env(agent, llm, st, monkeypatch, slot=28213, owner="sess-a", key="slack-thread")
    await agent._slot_switch_before(_member_turn(hs))
    assert st["order"] == ["restore"]


async def test_a_failed_park_restores_nothing_for_a_member(members, monkeypatch):
    agent, llm, st = members
    hs = await _members_file(agent, llm, st)
    _switch_env(agent, llm, st, monkeypatch, slot=40000, owner="sess-a", key="slack-thread")
    st["park_ok"] = False
    await agent._slot_switch_before(_member_turn(hs))
    assert st["order"] == ["park"]


async def test_the_loop_reaches_the_members_refresh_on_a_tick_where_nothing_ran(mock_context):
    """Review 10-27: after member-only traffic every later tick skipped as
    "nothing ran" before the refresh, so the members' file was never made."""
    from ghost_agent.core.agent import GhostAgent
    agent = GhostAgent(mock_context)
    mock_context.llm_client = MagicMock(foreground_requests=0, _head_ref=None, _outside_slot_dirty=False)
    refreshed = []

    async def warm(quiet=False, displaced_by=""):
        pass

    async def refresh():
        refreshed.append(1)
        return False

    async def metrics():
        return (0, 500)                              # nothing moves

    ticks = []

    async def sleep(s):
        ticks.append(s)
        if len(ticks) > 3:
            raise asyncio.CancelledError
    agent.warm_up_main_prefix = warm
    agent._member_head_refresh = refresh
    with pytest.raises(asyncio.CancelledError):
        await agent.rewarm_main_prefix_loop(period_s=180, sleep=sleep, metrics=metrics)
    assert len(refreshed) == 3


async def test_a_busy_slot_defers_a_fresh_restore_instead_of_prefilling(warm):
    """Review 10-27: fresh/outside modes with a busy slot returned without
    deferring, so the warm-up sent its full prefill."""
    agent, llm, st = warm
    await agent.warm_up_main_prefix(quiet=True)            # the file
    st["sent"] = 0
    st["order"].clear()
    st["fresh"] = True
    st["busy"] = True
    await agent.warm_up_main_prefix(quiet=True)
    assert st["sent"] == 0 and st["order"] == []


async def test_a_restore_reply_that_timed_out_but_landed_is_not_called_a_failure(warm, monkeypatch):
    """Review 10-27: 9 "head restore from disk failed (TimeoutError)" lines were
    restores llama applied after the client gave up; the warm-up found the
    head cached and no save happened."""
    agent, llm, st = warm
    await agent.warm_up_main_prefix(quiet=True)
    st["lines"].clear()
    st["order"].clear()
    st["fresh"] = True
    monkeypatch.setattr(H, "restore", lambda b, d, k: (st["order"].append("restore") or (False, "TimeoutError", 30.0)))
    st["resp"] = _resp(28213, 28209)
    await agent.warm_up_main_prefix(quiet=True)
    assert "restore" in st["order"]
    assert not any("failed" in m for m in st["lines"])


def test_a_save_attempt_clears_stale_part_files(monkeypatch, tmp_path):
    d = _dir(tmp_path)
    old = d / "head-m1.part.bin"
    old.write_bytes(b"x")
    os.utime(old, (time.time() - H.STALE_PART_S - 10,) * 2)

    def refused(url, body=None, timeout=5.0):
        raise _http_error(500, "Unable to save")
    monkeypatch.setattr(H, "_http", refused)
    ok, _ = H.save(BASE, d, "k", 28213, "id")
    assert not ok and not old.exists()


async def test_after_a_model_change_the_members_save_does_not_restore_the_old_owner_file(quiet_agent):
    """Review 10-27: the owner's plan is a tick old; after a llama restart on
    another model it must not be restored."""
    from ghost_agent.utils.logging import requester_role_context
    agent, llm, st = quiet_agent
    requester_role_context.set("")
    await agent.warm_up_main_prefix(quiet=True)
    agent._head_plan_last["identity"] = "the model before the restart"
    agent._member_head_wanted = True
    st["restores"].clear()
    st["kinds"].clear()
    st["resp"] = _resp(8000, 0)
    assert await agent._member_head_refresh() is True
    assert st["kinds"] == [H.MEMBER] and st["restores"] == []



async def test_a_drained_turns_late_call_cannot_restore_under_the_turn_holding_the_lock(warm, monkeypatch):
    """Round-2 review MAJOR: a streamed turn stays "running" while its reply
    drains, after it released the lock; its late call (a forced-final retry)
    must not restore the head under the next turn's tool gap."""
    from ghost_agent.utils.logging import request_id_context
    agent, llm, st = warm
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear()
    _switch_env(agent, llm, st, monkeypatch)
    async with agent._turn_lock("turn-B"):
        tok = request_id_context.set("turn-A")           # the drained turn's call
        try:
            await agent._slot_switch_before(_turn(st))
            assert st["order"] == []
            request_id_context.set("turn-B")
            await agent._slot_switch_before(_turn(st))
            assert st["order"] == ["restore"]
        finally:
            request_id_context.reset(tok)
    assert agent._turn_lock_rid is None


async def test_another_requests_call_does_not_wipe_a_requests_once_record(warm, monkeypatch):
    """Round-2 review MAJOR: the record held one request id per kind and was
    written before the call was accepted — another request's call overwrote it
    and the first request's next call restored over its own conversation."""
    from ghost_agent.utils.logging import request_id_context
    agent, llm, st = warm
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear()
    _switch_env(agent, llm, st, monkeypatch, owner=None, key="sess-a")
    tok = request_id_context.set("req-S")
    try:
        async with agent._turn_lock("req-S"):
            await agent._slot_switch_before(_turn(st))       # restores once
            request_id_context.set("req-other")              # refused: not the holder
            await agent._slot_switch_before(_turn(st))
            request_id_context.set("req-S")
            await agent._slot_switch_before(_turn(st, "its next call"))
    finally:
        request_id_context.reset(tok)
    assert st["order"] == ["restore"]


async def test_an_unreadable_slots_endpoint_gives_up_after_a_few_reads(warm, monkeypatch):
    """Round-2 review: a wedged or slot-less llama made the switch poll ~118 s
    before every request's first call."""
    from ghost_agent.core.agent import SWITCH_UNREAD_LIMIT
    agent, llm, st = warm
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear()
    _switch_env(agent, llm, st, monkeypatch)
    monkeypatch.setattr(L, "BEFORE_MAIN_DISPATCH_S", 150.0)  # the real bound
    reads = []

    def unreadable(url, body=None, timeout=5.0):
        reads.append(url)
        raise OSError("no /slots")
    monkeypatch.setattr(H, "_http", unreadable)
    t0 = time.monotonic()
    await agent._slot_switch_before(_turn(st))
    assert st["order"] == [] and len(reads) == SWITCH_UNREAD_LIMIT
    assert time.monotonic() - t0 < 5.0


async def test_the_member_park_is_bounded_by_the_deadline(members, monkeypatch):
    agent, llm, st = members
    hs = await _members_file(agent, llm, st)
    _switch_env(agent, llm, st, monkeypatch, slot=40000, owner="sess-a", key="slack-thread")
    seen = []
    monkeypatch.setattr(H, "park", lambda base, timeout=H.PARK_TIMEOUT_S: (
        seen.append(timeout) or st["order"].append("park") or (True, "parked")))
    monkeypatch.setattr(L, "BEFORE_MAIN_DISPATCH_S", H.RESTORE_TIMEOUT_S + 2.0 + 20.0)
    await agent._slot_switch_before(_member_turn(hs))
    assert seen and seen[0] <= 20.0 and st["order"] == ["park", "restore"]


@pytest.mark.parametrize("detail,resp", [("HTTPError 500: Unable to restore", _resp(28213, 28209)),
                                         ("TimeoutError: timed out", {"choices": []})])
async def test_a_failed_restore_is_still_said_unless_a_timeout_landed(warm, monkeypatch, detail, resp):
    """Round-2 review: "landed" needs a timeout AND a warm-up that found the
    head resident — not a real failure served from llama's RAM cache, nor a
    warm-up with no usage."""
    agent, llm, st = warm
    await agent.warm_up_main_prefix(quiet=True)
    st["lines"].clear()
    st["fresh"] = True
    monkeypatch.setattr(H, "restore", lambda b, d, k: (False, detail, 1.0))
    st["resp"] = resp
    await agent.warm_up_main_prefix(quiet=True)
    assert any("failed" in m for m in st["lines"])


async def test_a_members_restore_keeps_the_file_recent(members, monkeypatch):
    from ghost_agent.core.agent import MEMBER_HEAD_RECENT_S
    agent, llm, st = members
    hs = await _members_file(agent, llm, st)
    meta = hs["dir"] / f"head-{hs['key']}.json"
    m = json.loads(meta.read_text())
    m.update(kind="member", saved_at=time.time() - MEMBER_HEAD_RECENT_S - 60)
    meta.write_text(json.dumps(m))
    os.utime(meta, (time.time() - MEMBER_HEAD_RECENT_S - 60,) * 2)
    assert not H.has_kind(hs["dir"], H.MEMBER, within_s=MEMBER_HEAD_RECENT_S)
    _switch_env(agent, llm, st, monkeypatch, slot=28213, owner="sess-a", key="slack-thread")
    await agent._slot_switch_before(_member_turn(hs))
    assert H.has_kind(hs["dir"], H.MEMBER, within_s=MEMBER_HEAD_RECENT_S)


async def test_a_present_members_file_is_not_planned_again_every_tick(quiet_agent, monkeypatch):
    from ghost_agent.utils.logging import requester_role_context
    agent, llm, st = quiet_agent
    requester_role_context.set("")
    await agent.warm_up_main_prefix(quiet=True)
    hs = await agent._member_head_plan(llm)
    _store(hs["dir"], hs["key"], 8000)
    agent._member_head_wanted = True
    plans = []
    real = agent._member_head_plan

    async def counted(llm_):
        plans.append(1)
        return await real(llm_)
    monkeypatch.setattr(agent, "_member_head_plan", counted)
    assert await agent._member_head_refresh() is False
    assert await agent._member_head_refresh() is False
    assert len(plans) == 1



async def test_a_slots_read_that_times_out_is_a_long_batch_not_a_dead_llama(warm, monkeypatch):
    """Round-3 review: llama answers /slots only between batches and a 2048-token
    batch on the 27B can outlast the 3 s read — timeouts must keep the switch
    waiting (a give-up re-processes the 28k head)."""
    agent, llm, st = warm
    await _warm_with_file(agent, st, quiet=True)
    st["order"].clear()
    _switch_env(agent, llm, st, monkeypatch)
    monkeypatch.setattr(L, "BEFORE_MAIN_DISPATCH_S", H.RESTORE_TIMEOUT_S + 10.0)
    reads = []

    def slow_batches(url, body=None, timeout=5.0):
        reads.append(url)
        if len(reads) <= 4:
            raise TimeoutError("timed out")
        return [{"is_processing": False, "n_prompt_tokens": 38213}]
    monkeypatch.setattr(H, "_http", slow_batches)
    await agent._slot_switch_before(_turn(st))
    assert st["order"] == ["restore"]


async def test_a_new_owner_head_replans_the_members_file_at_once(quiet_agent, monkeypatch):
    """Round-3 review: a prompt or llama change makes new keys; the 30-min
    recheck must not hide the members' new key."""
    from ghost_agent.utils.logging import requester_role_context
    agent, llm, st = quiet_agent
    requester_role_context.set("")
    await agent.warm_up_main_prefix(quiet=True)
    agent._member_head_wanted = True
    agent._member_head_seen = ("an older owner head", time.monotonic())
    st["resp"] = _resp(8000, 0)
    assert await agent._member_head_refresh() is True


async def test_a_refused_member_call_does_not_take_the_requests_record(members, monkeypatch):
    from ghost_agent.utils.logging import request_id_context
    agent, llm, st = members
    hs = await _members_file(agent, llm, st)
    _switch_env(agent, llm, st, monkeypatch, slot=28213, owner="sess-a", key="slack-thread")
    async with agent._turn_lock("slack-1"):
        request_id_context.set("slack-late")             # not the holder: refused
        await agent._slot_switch_before(_member_turn(hs))
        assert agent.__dict__.get("_member_head_rid") != "slack-late"
        request_id_context.set("slack-1")
        await agent._slot_switch_before(_member_turn(hs))
    assert st["restores"] == [hs["key"]]


# ── §4RW: a level with no file starts from the nearest head; members save at first sight ──

def _owned_by_server(d, key, n=28213, ident="id"):
    (d / f"head-{key}.json").write_text(json.dumps({"n_tokens": n, "identity": ident,
                                                   "kind": H.OWNER, "saved_at": time.time()}))


async def test_a_level_with_no_file_starts_from_the_nearest_head_and_saves_its_own(warm):
    """Live 10-27: the owner picked xhigh (28,251 tokens, no file) and the next
    re-warm processed the whole head from token 0 (167 s)."""
    agent, llm, st = warm
    await agent.warm_up_main_prefix(quiet=True)                      # the medium head's file
    hs = agent._head_plan_last
    _owned_by_server(hs["dir"], hs["key"])
    st["rendered"], st["slot"] = "HEAD at xhigh", 9000                # nobody's state in the slot
    st["resp"] = _resp(28251, 27200)
    st["restores"].clear(); st["saves"].clear(); st["logged"].clear(); st["sent"] = 0
    await agent.warm_up_main_prefix(quiet=True)
    new = agent._head_plan_last
    assert new["key"] != hs["key"]
    assert st["restores"] == [hs["key"]] and st["sent"] == 1
    assert st["saves"] == [28251] and H.stored_tokens(new["dir"], new["key"]) == 28251
    assert any(i == Icons.HEAD_RESTORE and "nearest saved head" in m for t, m, i in st["logged"])


async def test_another_servers_head_is_not_a_start(warm):
    agent, llm, st = warm
    await agent.warm_up_main_prefix(quiet=True)
    hs = agent._head_plan_last
    _owned_by_server(hs["dir"], hs["key"], ident="another model")
    st["rendered"], st["slot"] = "HEAD at xhigh", 9000
    st["restores"].clear(); st["sent"] = 0
    await agent.warm_up_main_prefix(quiet=True)
    assert st["restores"] == [] and st["sent"] == 1                   # the prefill, as before


async def test_the_switch_starts_from_the_nearest_head_when_this_one_has_none(warm, monkeypatch):
    agent, llm, st = warm
    await agent.warm_up_main_prefix(quiet=True)
    hs = agent._head_plan_last
    _owned_by_server(hs["dir"], hs["key"])
    agent._head_plan_last = {**hs, "key": "no-file-yet", "have": False, "n_tokens": None,
                             "prompt_n": 28251}
    st["order"].clear(); st["restores"].clear()
    _switch_env(agent, llm, st, monkeypatch, slot=40000, owner="sess-a")
    st["slot"] = 40000
    await agent._slot_switch_before(_turn(st))
    assert st["restores"] == [hs["key"]]


async def test_the_switch_leaves_the_head_a_rewarm_just_made(warm, monkeypatch):
    """Review: the re-warm processed this head (no file: its save refused) —
    the near file must not go over it."""
    agent, llm, st = warm
    await agent.warm_up_main_prefix(quiet=True)
    hs = agent._head_plan_last
    _owned_by_server(hs["dir"], hs["key"])
    agent._head_plan_last = {**hs, "key": "no-file-yet", "have": False, "n_tokens": None,
                             "prompt_n": 28251}
    st["order"].clear(); st["restores"].clear()
    _switch_env(agent, llm, st, monkeypatch, slot=28251, owner="sess-a")
    st["slot"] = 28251
    await agent._slot_switch_before(_turn(st))
    assert st["restores"] == []


def _member_http(st, prompt=8000, real=None):
    real = real or H._http

    def http(url, body=None, timeout=5.0):
        if not url.endswith("/v1/chat/completions"):
            return real(url, body, timeout)
        st.setdefault("posted", []).append((url, body))
        return _resp(prompt, 0)
    return http


async def test_a_members_first_request_with_no_file_saves_it_for_the_next(members, monkeypatch):
    from ghost_agent.core import agent as A
    from ghost_agent.utils.logging import request_id_context
    agent, llm, st = members
    monkeypatch.setattr(A, "MEMBER_HEAD_MAKE_MIN_S", 0.0)          # the fixture's short deadline
    await agent.warm_up_main_prefix(quiet=True)                      # the owner's head only
    _switch_env(agent, llm, st, monkeypatch, slot=28213, owner="sess-a", key="slack-thread")
    hs = await agent._member_head_plan(llm)

    async def idle(base, late, stop=None):
        return True
    monkeypatch.setattr(agent, "_slot_idle", idle)
    monkeypatch.setattr(H, "_http", _member_http(st))
    st["saves"].clear(); st["kinds"].clear(); st["restores"].clear(); st["logged"].clear()
    await agent._slot_switch_before(_member_turn(hs))
    url, body = st["posted"][0]
    assert url == BASE + "/v1/chat/completions" and body["max_tokens"] == 1
    assert body["messages"] == hs["payload"]["messages"]             # the members' head alone
    assert st["saves"] == [8000] and st["kinds"] == [H.MEMBER] and st["restores"] == []
    assert [i for t, m, i in st["logged"] if "members' head" in m] == [Icons.HEAD_SAVE]
    request_id_context.set("slack-2")                                 # the next member request
    await agent._slot_switch_before(_member_turn(hs, "the next request"))
    assert st["restores"] == [hs["key"]] and len(st["posted"]) == 1


@pytest.mark.parametrize("case", ["busy", "no_count", "save_refused", "no_time"])
async def test_a_members_save_that_cannot_happen_leaves_the_call_as_before(members, monkeypatch, case):
    from ghost_agent.core import agent as A
    agent, llm, st = members
    monkeypatch.setattr(A, "MEMBER_HEAD_MAKE_MIN_S", 1e6 if case == "no_time" else 0.0)
    await agent.warm_up_main_prefix(quiet=True)
    _switch_env(agent, llm, st, monkeypatch, slot=28213, owner="sess-a", key="slack-thread")
    hs = await agent._member_head_plan(llm)

    async def idle(base, late, stop=None):
        return case != "busy"
    monkeypatch.setattr(agent, "_slot_idle", idle)
    monkeypatch.setattr(H, "_http", _member_http(st, prompt=0 if case == "no_count" else 8000))
    if case == "save_refused":
        monkeypatch.setattr(H, "save", lambda *a, **k: (False, "slot held 9000 tokens, expected 8000"))
    st["saves"].clear(); st["logged"].clear()
    await agent._slot_switch_before(_member_turn(hs))
    assert H.stored_tokens(hs["dir"], hs["key"]) is None and st["restores"] == []
    assert ("posted" in st) == (case not in ("busy", "no_time"))
    assert [i for t, m, i in st["logged"] if "members' head" in m] == [Icons.HEAD_FAIL]


async def test_the_members_prefill_never_replaces_an_unsaved_conversation(quiet_agent, monkeypatch):
    """§4RW-e r3: a request that started during the conversation's save left
    the record in place (or the save failed) — the members' prefill went on
    and replaced a conversation that was not on disk."""
    from ghost_agent.core import llm as L
    from ghost_agent.utils.logging import requester_role_context
    agent, llm, st = quiet_agent
    requester_role_context.set("")
    await agent.warm_up_main_prefix(quiet=True)
    st["saves"].clear()
    agent._member_head_wanted = True
    st["resp"] = _resp(8000, 0)
    llm._head_displaced = True
    llm._slot_session, llm._slot_session_used = "sess-x", time.monotonic()
    monkeypatch.setattr(L, "slot_kept", lambda llm, now=None: bool(llm._slot_session))
    monkeypatch.setattr(L, "conv_keep_enabled", lambda: True)
    shown = []

    async def displace(llm, why, late=None, stop=None):
        shown.append(why)
        return False                                       # the conversation still holds the slot
    agent._conv_displace = displace
    sent = []
    real = llm.chat_completion

    async def spy(payload, **kw):
        sent.append(kw.get("task_label"))
        return await real(payload, **kw)
    llm.chat_completion = spy
    assert await agent._member_head_refresh() is False
    assert shown and sent == [] and st["saves"] == []


# ── 10-27: the directory needs no upkeep ────────────────────────────────

def test_prune_drops_saves_of_another_server_only_when_told_which_is_running(tmp_path):
    d = _dir(tmp_path)
    _meta(d, "mine", "id", 1)
    _meta(d, "gone", "old-model", 2)
    _store(d, "legacy")                              # a meta without an identity: kept
    H.prune(d)
    assert (d / "head-gone.bin").exists()            # no identity given: nothing judged
    H.prune(d, identity="id")
    names = sorted(p.name for p in d.glob("head-*.bin"))
    assert names == ["head-legacy.bin", "head-mine.bin"]
    assert not (d / "head-gone.json").exists()


def test_prune_drops_stale_orphan_metas_and_tmp_files(tmp_path):
    d = _dir(tmp_path)
    old = time.time() - H.STALE_PART_S - 10
    _store(d, "k")
    (d / "head-nobin.json").write_text("{}")
    (d / "head-k.json.tmp").write_text("{}")
    (d / "head-fresh.json").write_text("{}")        # a save between its two writes: kept
    for n in ("head-nobin.json", "head-k.json.tmp"):
        os.utime(d / n, (old, old))
    H.prune(d)
    assert sorted(p.name for p in d.iterdir()) == ["head-fresh.json", "head-k.bin", "head-k.json"]


def test_conversations_are_held_to_a_byte_budget_but_the_newest_stays(tmp_path, monkeypatch):
    d = _dir(tmp_path)
    now = time.time()
    for i in range(3):
        (d / f"head-c{i}.bin").write_bytes(b"x" * 100)
        (d / f"head-c{i}.json").write_text(json.dumps({"n_tokens": 1, "kind": H.CONV}))
        for n in (f"head-c{i}.bin", f"head-c{i}.json"):
            os.utime(d / n, (now - 100 + i,) * 2)
    _store(d, "owner")
    monkeypatch.setattr(H, "CONV_BYTES_MAX", 250)
    H.prune(d)
    assert sorted(p.name for p in d.glob("head-*.bin")) == ["head-c1.bin", "head-c2.bin", "head-owner.bin"]
    monkeypatch.setattr(H, "CONV_BYTES_MAX", 10)     # even one is over: the newest is kept
    H.prune(d)
    assert sorted(p.name for p in d.glob("head-*.bin")) == ["head-c2.bin", "head-owner.bin"]


def test_llama_start_frees_the_saves_of_a_previous_model(monkeypatch, tmp_path):
    d = _dir(tmp_path)
    _meta(d, "k", _ident(), 1)
    _meta(d, "was", "/old|b|x", 2)
    _server(monkeypatch)
    H.restore_on_start(BASE, d, sleep=lambda s: None)
    assert (d / "head-k.bin").exists() and not (d / "head-was.bin").exists()


def test_a_save_never_judges_identity_from_its_callers_plan(monkeypatch, tmp_path):
    """Fresh-eye review 10-27: a plan read before a llama restart on another
    model passed the OLD identity to save(); pruning by it wiped the running
    server's owner head, members' head and conversations."""
    d = _dir(tmp_path)
    _meta(d, "new-owner", "NEW", 2)
    _meta(d, "new-member", "NEW", 3)
    monkeypatch.setattr(H, "_http", _llama_saves(d, 5))
    ok, detail = H.save(BASE, d, "cstale", 5, identity="OLD", kind=H.CONV, at_least=True)
    assert ok, detail
    assert (d / "head-new-owner.bin").exists() and (d / "head-new-member.bin").exists()


def test_the_byte_budget_counts_only_the_conversations_it_keeps(tmp_path, monkeypatch):
    d = _dir(tmp_path)
    now = time.time()
    for i, (key, size) in enumerate([("old", 20), ("big", 90), ("new", 20)]):
        (d / f"head-{key}.bin").write_bytes(b"x" * size)
        (d / f"head-{key}.json").write_text(json.dumps({"n_tokens": 1, "kind": H.CONV}))
        for n in (f"head-{key}.bin", f"head-{key}.json"):
            os.utime(d / n, (now - 100 + i,) * 2)
    monkeypatch.setattr(H, "CONV_BYTES_MAX", 100)
    H.prune(d)
    assert sorted(p.name for p in d.glob("head-*.bin")) == ["head-new.bin", "head-old.bin"]


def test_a_file_vanishing_mid_prune_does_not_stop_the_part_sweep(tmp_path, monkeypatch):
    d = _dir(tmp_path)
    _store(d, "k")
    (d / "head-x.part.bin").write_bytes(b"x")
    old = time.time() - H.STALE_PART_S - 10
    os.utime(d / "head-x.part.bin", (old, old))
    real_kind = H.kind_of

    def vanish(dd, key):                     # the other process deleted it meanwhile
        (dd / f"head-{key}.bin").unlink()
        (dd / f"head-{key}.json").unlink()
        return real_kind(dd, key)
    monkeypatch.setattr(H, "kind_of", vanish)
    H.prune(d)
    assert not (d / "head-x.part.bin").exists()


def test_a_state_file_without_its_meta_never_takes_a_heads_place(tmp_path):
    """Review r2: a .bin with no meta counted as an OWNER head (kind_of's
    default) and pushed a real one out; it can never be restored."""
    d = _dir(tmp_path)
    now = time.time()
    for i in range(3):
        _store(d, f"o{i}")
        for n in (f"head-o{i}.bin", f"head-o{i}.json"):
            os.utime(d / n, (now - 100 + i,) * 2)
    (d / "head-young.bin").write_bytes(b"x")          # a save between its two writes
    (d / "head-dead.bin").write_bytes(b"x")
    old = now - H.STALE_PART_S - 10
    os.utime(d / "head-dead.bin", (old, old))
    H.prune(d, keep=3)
    names = sorted(p.name for p in d.glob("head-*.bin"))
    assert names == ["head-o0.bin", "head-o1.bin", "head-o2.bin", "head-young.bin"]


def test_a_save_whose_file_was_pruned_meanwhile_reports_failure(monkeypatch, tmp_path):
    d = _dir(tmp_path)

    def http(url, body=None, timeout=5.0):
        (d / body["filename"]).write_bytes(b"state")
        return {"n_saved": 5}
    monkeypatch.setattr(H, "_http", http)
    real = os.replace

    def gone(src, dst):
        if str(src).endswith(".part.bin"):
            os.unlink(src)
        return real(src, dst)
    monkeypatch.setattr(H.os, "replace", gone)
    ok, detail = H.save(BASE, d, "k", 5, "id")
    assert not ok and "removed" in detail


def test_pruning_a_key_never_takes_the_re_save_llama_is_writing(tmp_path, monkeypatch):
    """Review r3: discard() also removed the key's .part.bin — a re-save of
    a conversation dropped by count failed and the conversation was lost."""
    d = _dir(tmp_path)
    now = time.time()
    for i in range(3):
        (d / f"head-c{i}.bin").write_bytes(b"x")
        (d / f"head-c{i}.json").write_text(json.dumps({"n_tokens": 1, "kind": H.CONV}))
        for n in (f"head-c{i}.bin", f"head-c{i}.json"):
            os.utime(d / n, (now - 100 + i,) * 2)
    (d / "head-c0.part.bin").write_bytes(b"being written")
    monkeypatch.setattr(H, "KEEP_CONV", 2)
    H.prune(d)
    assert not (d / "head-c0.bin").exists() and (d / "head-c0.part.bin").exists()


def test_a_meta_that_cannot_be_written_fails_the_save_without_raising(monkeypatch, tmp_path):
    d = _dir(tmp_path)
    _store(d, "k", n=99)                               # the previous save of this key
    monkeypatch.setattr(H, "_http", _llama_saves(d, 5))
    (d / "head-k.json.tmp").mkdir()                    # stands in for a full disk
    ok, detail = H.save(BASE, d, "k", 5, "id")
    assert not ok and "meta not written" in detail
    assert H.stored_tokens(d, "k") is None and not (d / "head-k.bin").exists()


def test_llama_start_picks_the_head_saved_last_not_the_one_used_last(tmp_path):
    """Operator 10-27: the most recently SAVED head is loaded after a llama
    restart, even when an older one was restored (touched) more recently."""
    d = _dir(tmp_path)
    _meta(d, "saved-last", "id", 200)
    _meta(d, "used-last", "id", 100)
    H.touch(d, "used-last")
    assert H.newest_head_for(d, "id") == "saved-last"


def test_a_file_re_saved_while_the_prune_runs_is_left_for_the_next_one(tmp_path, monkeypatch):
    """Review r4: the ranking was taken once; a conversation re-saved (now
    the newest, bigger) mid-prune was judged by its old rank and deleted."""
    d = _dir(tmp_path)
    now = time.time()
    for i, (key, size) in enumerate([("ck", 50), ("ca", 100)]):
        (d / f"head-{key}.bin").write_bytes(b"x" * size)
        (d / f"head-{key}.json").write_text(json.dumps({"n_tokens": 1, "kind": H.CONV}))
        for n in (f"head-{key}.bin", f"head-{key}.json"):
            os.utime(d / n, (now - 100 + i,) * 2)
    monkeypatch.setattr(H, "CONV_BYTES_MAX", 240)
    real = H.kind_of

    def resave_mid_prune(dd, key):
        if key == "ca":                              # the other thread's save of ck lands now
            (dd / "head-ck.bin").write_bytes(b"x" * 150)
            os.utime(dd / "head-ck.bin", (now + 5,) * 2)
        return real(dd, key)
    monkeypatch.setattr(H, "kind_of", resave_mid_prune)
    H.prune(d)
    assert (d / "head-ck.bin").exists() and (d / "head-ca.bin").exists()


@pytest.mark.parametrize("reply", ["not json", ["a"], {"n_saved": "x"}, {"n_saved": 5, "timings": "?"}])
def test_an_odd_llama_reply_never_raises_out_of_save(monkeypatch, tmp_path, reply):
    d = _dir(tmp_path)

    def http(url, body=None, timeout=5.0):
        (d / body["filename"]).write_bytes(b"state")
        return reply
    monkeypatch.setattr(H, "_http", http)
    ok, _ = H.save(BASE, d, "k", 5, "id")
    assert ok == (reply == {"n_saved": 5, "timings": "?"})


@pytest.mark.parametrize("reply", ["not json", ["a"]])
def test_an_odd_llama_reply_never_raises_out_of_restore_nor_costs_the_file(monkeypatch, tmp_path, reply):
    """Review r5: a non-object reply read as 0 tokens restored, and the
    mismatch path deleted a good state file."""
    d = _dir(tmp_path)
    _store(d, "k")
    monkeypatch.setattr(H, "_http", lambda *a, **k: reply)
    ok, detail, _ = H.restore(BASE, d, "k")
    assert not ok and "not the restore result" in detail
    assert H.stored_tokens(d, "k") == 28213


def test_a_non_numeric_restore_count_never_raises(monkeypatch, tmp_path):
    d = _dir(tmp_path)
    _store(d, "k")
    monkeypatch.setattr(H, "_http", lambda *a, **k: {"n_restored": "x"})
    ok, detail, _ = H.restore(BASE, d, "k")
    assert not ok and "restored 0 tokens" in detail
