"""§4GI (2026-09-13): the API never blocks the event loop on a store lock,
proxied inference is booked against the main node like a chat turn, the
API key never goes upstream, and a "must not die" handler never raises.

Pre-fix worlds (each pin fails there):
  * `memory.correct_fragment(...)` called inline in an `async def` — the
    vector store's RLock froze every stream and /api/health;
  * `catch_all` / `api_generate` posted through `http_client` with no
    `foreground_tasks`, no `_main_node_lock`, no in-flight count — a user
    turn queued behind a proxied completion was aborted as a "sole
    in-flight" upstream stall; `X-Ghost-Key` was forwarded verbatim;
  * `learning_health.py` used an undefined `logger` in the except-handler
    of the lowest-confidence block, so the one failure it existed to absorb
    raised NameError.
"""
import ast
import asyncio
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.api import routes as R

REPO = Path(__file__).resolve().parent.parent


# ── R1 enumeration: every store call in a route goes through _store_call ─────

def _is_route(fn):
    return any(isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)
               and d.func.attr in ("get", "post", "put", "delete", "api_route", "websocket")
               for d in fn.decorator_list)


def _is_context(n):
    if isinstance(n, ast.Attribute) and n.attr == "context":
        return True
    return (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "getattr"
            and len(n.args) >= 2 and isinstance(n.args[1], ast.Constant) and n.args[1].value == "context")


def _off_context(n):
    """The context attribute name when ``n`` is one attribute off the context."""
    if isinstance(n, ast.Attribute) and _is_context(n.value):
        return n.attr
    if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "getattr"
            and len(n.args) >= 2 and _is_context(n.args[0]) and isinstance(n.args[1], ast.Constant)):
        return n.args[1].value
    return None


#: Calls on a context-derived object that are DELIBERATELY left on the loop,
#: with the reason. A name list of STORES was what made the first version
#: blind (§4GJ round 3); this is the inverse — the enumeration flags
#: everything it can reach and each exemption must justify itself here. A
#: stale entry (the call no longer exists) fails its own test, so this list
#: cannot rot into a silencer.
ENUM_EXEMPTIONS = {
    ("load_workspace", "clear"):
        "scratchpad.clear() is an in-memory dict reset, no IO",
    ("load_workspace", "restore_state"):
        "scratchpad.restore_state() rebuilds an in-memory dict from the body",
    ("chat_proxy", "stream_openai"):
        "an async generator consumed with `async for` — the streaming "
        "inference path itself, not a blocking call to wrap",
    ("chat_proxy", "usage_for"):
        "an in-memory ring lookup (llm.usage_for -> dict(...)), no IO",
    ("load_workspace", "resolve"):
        "sandbox_dir is a Path, not a store; resolve() is one stat",
    ("notifications_pending", "current_offset"):
        "a single os.path.getsize stat, no read — a thread hop would cost "
        "more than the call (read_since, which does parse the file, IS wrapped)",
    ("notifications_ack", "current_offset"):
        "same single stat as notifications_pending",
}


def _context_derived_names(fn):
    """Local names in ``fn`` bound to something obtained FROM the agent
    context — either one attribute off it (`agent.context.memory_system`) or
    the result of a helper CALL that receives the context
    (`get_session_store(agent.context)`). The second form is the one the
    first version could not see: the receiver is bound off a call, so an
    attribute-only rule reported an empty list for a file with four live
    violations.

    A VALUE returned by a method ON such an object is deliberately NOT
    derived: stores come from the context, values come from stores
    (`sess = store.get(id)` then `sess.to_dict()` is an in-memory shape
    change). That line is stated here because it is the only thing keeping
    the enumeration from demanding `_store_call` around every dict access.
    """
    derived = {}
    for n in ast.walk(fn):
        if not (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)):
            continue
        name, val = n.targets[0].id, n.value
        attr = _off_context(val)
        if attr is not None:
            derived[name] = attr
            continue
        # a helper call that is handed the context (or something already
        # derived from it) hands back a context-owned object
        if isinstance(val, ast.Call):
            for arg in list(val.args) + [kw.value for kw in val.keywords]:
                if _is_context(arg) or _off_context(arg) is not None:
                    derived[name] = ast.unparse(val.func)[:40]
                    break
                if isinstance(arg, ast.Name) and arg.id in derived:
                    derived[name] = ast.unparse(val.func)[:40]
                    break
    return derived


def direct_store_calls(source: str):
    """(handler, lineno, source, call) for every NON-awaited call on an
    object this route handler obtained from the agent context.

    Fail-closed by construction: the receiver set is DERIVED from the handler
    bodies, and anything reached from the context counts until an entry in
    ENUM_EXEMPTIONS says why it may stay on the loop.
    """
    tree = ast.parse(source)
    out = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and _is_route(n)]:
        # nested `async def`/`def` bodies belong to the handler's request too
        derived = _context_derived_names(fn)
        awaited = {id(a.value) for a in ast.walk(fn)
                   if isinstance(a, ast.Await) and isinstance(a.value, ast.Call)}
        for n in ast.walk(fn):
            if not isinstance(n, ast.Call) or id(n) in awaited:
                continue
            f = n.func
            if not isinstance(f, ast.Attribute):
                continue
            origin = None
            if isinstance(f.value, ast.Name) and f.value.id in derived:
                origin = derived[f.value.id]
            elif _off_context(f.value) is not None:
                origin = _off_context(f.value)
            if origin is None:
                continue
            if (fn.name, f.attr) in ENUM_EXEMPTIONS:
                continue
            out.append((fn.name, n.lineno, origin, ast.unparse(n)[:60]))
    return out


def _routes_source():
    return (REPO / "src/ghost_agent/api/routes.py").read_text()


def test_no_route_calls_a_store_directly():
    bad = direct_store_calls(_routes_source())
    assert bad == [], f"store calls that bypass _store_call: {bad}"
    # the wrapped forms exist (the enumeration is not vacuous)
    assert _routes_source().count("await _store_call(") >= 4


def test_the_enumeration_fires_on_a_direct_call():
    src = _routes_source()
    needle = "        ok, detail = await _store_call(memory.correct_fragment, match, replacement)\n"
    assert src.count(needle) == 1
    broken = src.replace(needle, "        ok, detail = memory.correct_fragment(match, replacement)\n")
    bad = direct_store_calls(broken)
    assert len(bad) == 1 and bad[0][0] == "memory_correct", bad


def test_the_enumeration_fires_on_a_HELPER_BOUND_store_too():
    """The blind spot itself (§4GJ round 3). `get_session_store(context)`
    binds the receiver off a CALL, so the attribute-only rule reported an
    empty list while four session routes and the chat hot path called a
    file-backed, fsyncing store straight on the event loop. This fails in
    the pre-round-3 world, where `direct_store_calls` saw nothing here."""
    src = _routes_source()
    needle = "        sessions = await _store_call(store.list, limit=limit)\n"
    assert src.count(needle) == 1, "re-point this pin: sessions_list changed"
    broken = src.replace(needle, "        sessions = store.list(limit=limit)\n")
    bad = direct_store_calls(broken)
    assert [b[0] for b in bad] == ["sessions_list"], bad
    assert "get_session_store" in bad[0][2], bad


def test_every_exemption_still_names_a_real_call():
    """An exemption that outlives its call site is a silencer waiting for a
    future violation to land on the same name. Fails when a listed call is
    renamed or removed."""
    tree = ast.parse(_routes_source())
    live = set()
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, ast.AsyncFunctionDef) and _is_route(n)]:
        for n in ast.walk(fn):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                live.add((fn.name, n.func.attr))
    stale = [k for k in ENUM_EXEMPTIONS if k not in live]
    assert stale == [], f"exemptions for calls that no longer exist: {stale}"
    assert all(str(v).strip() for v in ENUM_EXEMPTIONS.values()), "every exemption needs a reason"


# ── _store_call: the loop stays free, the wait is bounded ────────────────────

async def test_store_call_leaves_the_event_loop_free_while_the_store_blocks():
    gate = threading.Event()
    ticks = []

    def blocking():
        gate.wait(2.0)
        return "done"

    async def ticker():
        while True:
            ticks.append(time.monotonic())
            await asyncio.sleep(0.02)

    t = asyncio.create_task(ticker())
    call = asyncio.create_task(R._store_call(blocking))
    await asyncio.sleep(0.3)
    n_during = len(ticks)
    gate.set()
    assert await call == "done"
    t.cancel()
    assert n_during >= 8, n_during          # ~15 ticks in 0.3 s if the loop is free


async def test_store_call_times_out_instead_of_waiting_forever(monkeypatch):
    def stuck():
        time.sleep(3.0)
    t0 = time.monotonic()
    with pytest.raises(R.StoreCallTimeout):
        await R._store_call(stuck, timeout=0.2)
    assert time.monotonic() - t0 < 1.0


# ── the handlers ─────────────────────────────────────────────────────────────

def _req(json_body, agent):
    req = MagicMock()
    req.json = AsyncMock(return_value=json_body)
    req.app.state.agent = agent
    return req


async def test_memory_correct_runs_the_store_off_the_loop_and_answers():
    gate = threading.Event()
    calls = []

    def correct_fragment(match, replacement):
        calls.append((match, replacement, threading.current_thread() is threading.main_thread()))
        gate.wait(2.0)
        return True, {"before": "x", "after": replacement}

    agent = MagicMock()
    agent.context.memory_system = SimpleNamespace(correct_fragment=correct_fragment)
    ticks = []

    async def ticker():
        while True:
            ticks.append(1)
            await asyncio.sleep(0.02)
    t = asyncio.create_task(ticker())
    with patch.object(R, "get_agent", return_value=agent):
        h = asyncio.create_task(R.memory_correct(_req({"match": "x", "replacement": "y"}, agent)))
        await asyncio.sleep(0.3)
        assert len(ticks) >= 8                  # the loop kept serving
        gate.set()
        out = await h
    t.cancel()
    assert out == {"ok": True, "before": "x", "after": "y"}
    assert calls and calls[0][2] is False       # ran in a worker thread


@pytest.mark.parametrize("route,body,attr,method", [
    ("memory_correct", {"match": "a", "replacement": "b"}, "memory_system", "correct_fragment"),
    ("memory_delete", {"match": "a"}, "memory_system", "delete_fragment"),
    ("memory_delete_skill_twin", {"triggers": ["t"]}, "memory_system", "delete_skill_twins"),
    ("lessons_quarantine", {"trigger": "t", "reason": "r"}, "skill_memory", "quarantine_lesson"),
])
async def test_a_store_that_never_returns_yields_504_within_the_bound(monkeypatch, route, body, attr, method):
    monkeypatch.setattr(R, "_STORE_CALL_TIMEOUT_S", 0.2)
    started = threading.Event()

    def stuck(*a, **k):
        started.set()
        time.sleep(2.0)
        return True, {}
    store = SimpleNamespace(**{method: stuck, "is_read_only": False})
    agent = MagicMock()
    setattr(agent.context, attr, store)
    t0 = time.monotonic()
    with patch.object(R, "get_agent", return_value=agent):
        resp = await getattr(R, route)(_req(body, agent))
    assert resp.status_code == 504, route
    assert time.monotonic() - t0 < 1.0
    assert started.is_set()


# ── proxied inference is booked like a turn ─────────────────────────────────

class _Llm:
    """The bookkeeping surface of the real client, recording every step."""

    def __init__(self, upstream="http://main:8000"):
        self.upstream_url = upstream
        self._foreground_lock = asyncio.Lock()
        self.foreground_tasks = 0
        self._main_node_lock = asyncio.Lock()
        self.events = []
        self.http_client = MagicMock()

    def _inflight_inc(self, url):
        self.events.append(("inc", url, self.foreground_tasks))

    def _inflight_dec(self, url):
        self.events.append(("dec", url, self.foreground_tasks))


def _stream_resp(chunks=(b"data: x\n\n",), ct="text/event-stream"):
    r = MagicMock()
    r.status_code = 200
    r.headers = {"content-type": ct}
    r.aclose = AsyncMock()

    async def _aiter():
        for c in chunks:
            yield c
    r.aiter_bytes = MagicMock(side_effect=_aiter)
    return r


def _proxy_req(headers, body=b'{"stream": true}'):
    req = MagicMock()
    req.method = "POST"
    req.headers = headers
    req.body = AsyncMock(return_value=body)
    return req


async def test_catch_all_books_the_stream_for_its_whole_life_and_strips_the_key():
    llm = _Llm()
    seen = {}

    async def send(req, stream=True):
        seen["fg_during_send"] = llm.foreground_tasks
        seen["events_at_send"] = list(llm.events)
        return _stream_resp()
    llm.http_client.build_request = MagicMock(side_effect=lambda m, u, headers=None, content=None: seen.setdefault("headers", headers) or MagicMock())
    llm.http_client.send = AsyncMock(side_effect=send)
    agent = MagicMock()
    agent.context.llm_client = llm
    req = _proxy_req({"content-type": "application/json", "X-Ghost-Key": "sekrit",
                      "Authorization": "Bearer t", "host": "x", "content-length": "9",
                      "user-agent": "curl"})
    with patch.object(R, "get_agent", return_value=agent):
        resp = await R.catch_all(req, "v1/chat/completions")
    # booked before the send, still booked while the body streams
    assert seen["fg_during_send"] == 1
    assert seen["events_at_send"] == [("inc", "http://main:8000", 1)]
    assert llm.foreground_tasks == 1 and ("dec", "http://main:8000", 1) not in llm.events
    fwd = {k.lower() for k in seen["headers"]}
    assert fwd == {"content-type", "user-agent"}, fwd
    # …and released when the client finishes reading
    body = b"".join([c async for c in resp.body_iterator])
    assert body == b"data: x\n\n"
    assert llm.foreground_tasks == 0
    assert llm.events[-1][0] == "dec"
    assert not llm._main_node_lock.locked()      # a stream never holds the main lock


async def test_catch_all_releases_the_booking_when_the_upstream_send_fails():
    llm = _Llm()
    llm.http_client.build_request = MagicMock(return_value=MagicMock())
    llm.http_client.send = AsyncMock(side_effect=RuntimeError("boom"))
    agent = MagicMock()
    agent.context.llm_client = llm
    with patch.object(R, "get_agent", return_value=agent):
        resp = await R.catch_all(_proxy_req({"content-type": "application/json"}), "v1/x")
    assert resp.status_code == 502
    assert llm.foreground_tasks == 0
    assert [e[0] for e in llm.events] == ["inc", "dec"]


async def test_api_generate_holds_the_main_lock_and_restores_on_failure():
    llm = _Llm()
    seen = {}

    async def post(url, json=None):
        seen["locked"] = llm._main_node_lock.locked()
        seen["fg"] = llm.foreground_tasks
        r = MagicMock()
        r.raise_for_status = MagicMock()
        r.json = MagicMock(return_value={"choices": [{"message": {"content": "hi"}}]})
        return r
    llm.http_client.post = AsyncMock(side_effect=post)
    agent = MagicMock()
    agent.context.llm_client = llm
    with patch.object(R, "get_agent", return_value=agent):
        out = await R.api_generate(_req({"prompt": "p", "model": "m"}, agent))
    assert out["response"] == "hi"
    assert seen == {"locked": True, "fg": 1}
    assert llm.foreground_tasks == 0 and not llm._main_node_lock.locked()
    # §4GJ round 4: the LOCKED path must not touch the in-flight counter —
    # `_own_inflight` already adds one for a held `_main_node_lock`, so doing
    # both made one stream plus one generate read as 3 against a truth of 2
    # and tripped the self-queued abort. This line asserted the double count.
    # See tests/test_4gj_round4_routes.py::test_a_locked_main_request_is_not_counted_twice
    assert llm.events == []

    llm.http_client.post = AsyncMock(side_effect=RuntimeError("down"))
    with patch.object(R, "get_agent", return_value=agent):
        resp = await R.api_generate(_req({"prompt": "p"}, agent))
    assert resp.status_code == 500
    assert llm.foreground_tasks == 0 and not llm._main_node_lock.locked()


async def test_booking_helper_skips_a_mock_client_without_raising():
    """The API tests hand the routes a MagicMock client: every getattr is a
    MagicMock, which is neither a lock nor an int — the helper must skip,
    not `async with` a MagicMock."""
    async with R._main_node_request(MagicMock(), hold_lock=True):
        pass


# ── learning_health: the handler that must not die ───────────────────────────

def test_the_lowest_confidence_handler_swallows_instead_of_raising(monkeypatch, tmp_path):
    from ghost_agent.core import learning_health as LH
    from ghost_agent.core import calibration as C
    from tests.test_calibration_audit import _cal
    monkeypatch.setattr(LH, "collect_learning_health", lambda md: {"calibration": _cal()})

    def boom(self, *, limit=5, days=7):
        raise RuntimeError("ranking store unreadable")
    monkeypatch.setattr(C.CalibrationTracker, "lowest_confidence_turns", boom)
    out = LH.render_learning_health(tmp_path)          # pre-fix: NameError
    assert "CALIBRATION" in out
    assert "negative class" in out
    assert "LOWEST-CONFIDENCE" not in out


# ── every flag the code reads is documented somewhere in docs/ ───────────────

def test_every_ghost_flag_read_in_src_is_documented():
    reads = set()
    rx = re.compile(r'os\.(?:environ\.get|getenv)\(\s*"(GHOST_[A-Z0-9_]+)"')
    for p in (REPO / "src").rglob("*.py"):
        reads |= set(rx.findall(p.read_text(errors="ignore")))
    assert len(reads) >= 100
    documented = set()
    for p in (REPO / "docs").rglob("*.html"):
        documented |= set(re.findall(r"GHOST_[A-Z0-9_]+", p.read_text(errors="ignore")))
    missing = sorted(reads - documented)
    assert missing == [], f"flags read in src but documented nowhere in docs/: {missing}"
    assert "GHOST_STORE_CALL_TIMEOUT" in documented


# ── §4GJ round 3: the release survives cancellation, the main POST is bounded ─

@pytest.mark.parametrize("cancels", [1, 2, 3])
async def test_the_booking_release_survives_a_contended_cancel(cancels):
    """The CRITICAL of round 3. The release lived under `async with
    fg_lock`; in a `finally` that runs on client disconnect, `acquire()` is
    an await point, so a contended lock plus a second cancellation
    (uvicorn's disconnect then shutdown) raised CancelledError and the
    decrement never ran. `foreground_tasks` stuck at 1 hard-gates the
    biological tick for the life of the process.

    Fails in the pre-fix world at cancels>=2 with the lock held (measured:
    1 cancel survives, 2 and 3 leak). The in-flight count never leaked,
    because its release is a synchronous `stack.callback` — which is the
    shape the counter now uses too.
    """
    llm = _Llm()

    async def booked():
        async with R._main_node_request(llm, hold_lock=False):
            await asyncio.sleep(10)          # "streaming to the client"

    task = asyncio.create_task(booked())
    await asyncio.sleep(0.05)
    assert llm.foreground_tasks == 1, "the booking never started"

    # a background call polling the foreground lock, as llm.py does ~1/s
    await llm._foreground_lock.acquire()
    for _ in range(cancels):
        task.cancel()
        await asyncio.sleep(0)
    llm._foreground_lock.release()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.05)

    assert llm.foreground_tasks == 0, "the main-node booking leaked"
    assert [e[0] for e in llm.events] == ["inc", "dec"]


async def test_the_booking_release_does_not_need_the_lock_at_all():
    """The mechanism, stated: the release completes while another task holds
    the foreground lock and never gives it up. Under the pre-fix code this
    hangs until the lock is released; it now returns immediately."""
    llm = _Llm()
    await llm._foreground_lock.acquire()          # never released
    async with R._main_node_request(llm, hold_lock=False):
        assert llm.foreground_tasks == 1
    assert llm.foreground_tasks == 0


async def test_api_generate_bounds_the_upstream_post_and_answers_504():
    """`/api/generate` held the process-wide `_main_node_lock` across a POST
    with no timeout, inheriting httpx's 1200s default — one wedged upstream
    parks every chat turn and every embedding for twenty minutes, and
    Starlette does not cancel the handler on disconnect. It now uses the
    budget `core/llm` already applies to a locked main POST.

    Fails in the pre-fix world by hanging (no bound) instead of returning.
    """
    llm = _Llm()
    started = asyncio.Event()

    async def hang(url, json=None):
        started.set()
        await asyncio.sleep(3600)
    llm.http_client.post = AsyncMock(side_effect=hang)
    agent = MagicMock()
    agent.context.llm_client = llm

    # the real budget, shrunk for the test — the handler must READ it, not
    # carry its own copy
    with patch.object(R, "get_agent", return_value=agent), \
            patch("ghost_agent.core.llm._MAIN_FALLBACK_TIMEOUT_S", 0.2):
        t0 = time.monotonic()
        resp = await R.api_generate(_req({"prompt": "p", "model": "m"}, agent))
        elapsed = time.monotonic() - t0

    assert started.is_set()
    assert resp.status_code == 504, getattr(resp, "body", resp)
    assert elapsed < 5.0, f"the bound did not apply ({elapsed:.1f}s)"
    # and the slot is free again for every other caller
    assert llm.foreground_tasks == 0 and not llm._main_node_lock.locked()
    # §4GJ round 4: the held lock IS the in-flight count on this path (see
    # the sibling assertion above).
    assert llm.events == []


async def test_a_healthy_upstream_is_unaffected_by_the_bound():
    """Control: the two worlds differ only when the upstream is slow."""
    llm = _Llm()

    async def ok(url, json=None):
        r = MagicMock()
        r.raise_for_status = MagicMock()
        r.json = MagicMock(return_value={"choices": [{"message": {"content": "hi"}}]})
        return r
    llm.http_client.post = AsyncMock(side_effect=ok)
    agent = MagicMock()
    agent.context.llm_client = llm
    with patch.object(R, "get_agent", return_value=agent):
        out = await R.api_generate(_req({"prompt": "p", "model": "m"}, agent))
    assert out["response"] == "hi"
    assert llm.foreground_tasks == 0
