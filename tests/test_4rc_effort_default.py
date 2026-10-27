"""§4RC (2026-10-24, operator: "make default turns record whatever the default
is. medium now, but it might change"): a Default turn (no level picked) records
the level the llama server's own --chat-template-kwargs gave it, looked up
from the server's renderer (/apply-template: text only, no slot), not assumed."""
from __future__ import annotations

import http.server
import json
import threading
import time

import pytest

from ghost_agent.core import effort as E
from ghost_agent.core import turn_facts as TF
from ghost_agent.utils import logging as L

RID = "4rc0dflt"
XHIGH = "<|im_start|>system\nS\n\n" + E.XHIGH_SENTENCE + "<|im_end|>\n"


@pytest.fixture(autouse=True)
def _fresh():
    E._server_default.update(level="", at=0.0, busy=False)
    yield
    E._server_default.update(level="", at=0.0, busy=False)


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("GHOST_REASONING_EFFORT", "1")
    return monkeypatch


class _Server:
    """A local stand-in for llama's /apply-template."""

    def __init__(self, prompt=None, status=200, levels=True):
        self.prompt, self.status, self.bodies, self.levels = prompt, status, [], levels
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                outer.bodies.append((self.path, body))
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                asked = (body.get("chat_template_kwargs") or {}).get("reasoning_effort")
                if asked:      # the template's own sentence, when it has levels
                    out = {"prompt": "S\n\nReasoning effort is set to low." if outer.levels else "S"}
                else:
                    out = {"prompt": outer.prompt} if outer.prompt is not None else {}
                self.wfile.write(json.dumps(out).encode())

            def log_message(self, *a):
                pass
        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def server():
    made = []

    def make(**kw):
        s = _Server(**kw)
        made.append(s)
        return s
    yield make
    for s in made:
        s.close()


# ── reading the level from a render ────────────────────────────────────────── #

@pytest.mark.parametrize("prompt,want", [
    (XHIGH, "xhigh"),
    ("…\n\nReasoning effort is set to low. Keep your thinking brief…", "low"),
    ("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU", "medium"),   # medium writes no sentence
    ("", "medium")])
def test_the_level_in_a_render(prompt, want):
    assert E.level_in_render(prompt) == want


def test_the_lookup_asks_the_renderer_with_no_level(server):
    s = server(prompt=XHIGH)
    assert E.fetch_server_default(s.url + "/") == "xhigh"
    assert [p for p, _b in s.bodies] == ["/apply-template", "/apply-template"]
    assert s.bodies[0][1]["chat_template_kwargs"] == {"reasoning_effort": "low"}   # has levels?
    assert "chat_template_kwargs" not in s.bodies[1][1]                            # then: no level


def test_a_template_without_levels_gives_nothing(server):
    """Review r1: the 35B's template writes no sentence either — not "medium"."""
    s = server(prompt="<|im_start|>system\nS<|im_end|>", levels=False)
    assert E.fetch_server_default(s.url) == ""
    assert len(s.bodies) == 1


@pytest.mark.parametrize("kw", [dict(prompt=None), dict(prompt=""), dict(prompt=XHIGH, status=500)])
def test_a_server_that_cannot_say_gives_nothing(server, kw):
    assert E.fetch_server_default(server(**kw).url) == ""


def test_no_server_gives_nothing():
    assert E.fetch_server_default("http://127.0.0.1:9", timeout=0.5) == ""


# ── the cache ──────────────────────────────────────────────────────────────── #

def _until(pred, s=3.0):
    end = time.monotonic() + s
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_the_cached_value_never_blocks_and_one_lookup_fills_it(server):
    s = server(prompt=XHIGH)
    gate = threading.Event()
    real = E.fetch_server_default
    E_fetch = lambda url, timeout=3.0: (gate.wait(3), real(url, timeout))[1]
    import unittest.mock as um
    with um.patch.object(E, "fetch_server_default", E_fetch):
        assert E.server_default(s.url) == ""        # first call: starts the lookup, returns at once
        assert E.server_default(s.url) == ""        # a lookup in flight is not started twice
        gate.set()
        assert _until(lambda: E.server_default(s.url) == "xhigh")
    assert len(s.bodies) == 2 and E._server_default["busy"] is False


def test_a_stale_value_is_looked_up_again_and_kept_meanwhile(server):
    s = server(prompt=XHIGH)
    E.refresh_server_default(s.url)
    s.prompt = "plain"                              # llama restarted at medium
    E._server_default["at"] -= E.SERVER_DEFAULT_TTL_S + 1
    assert E.server_default(s.url) == "xhigh"       # the old value meanwhile
    assert _until(lambda: E.server_default(s.url) == "medium")


def test_a_failed_lookup_keeps_the_last_known_value(server):
    s = server(prompt=XHIGH)
    E.refresh_server_default(s.url)
    s.status = 500
    assert E.refresh_server_default(s.url) == ""
    assert E._server_default["level"] == "xhigh"


def test_no_url_starts_nothing():
    assert E.server_default("") == "" and E._server_default["busy"] is False


# ── the turn hook records it ───────────────────────────────────────────────── #

class _Ctx:
    pass


def _agent(url="http://127.0.0.1:8088/v1"):
    from types import SimpleNamespace
    from ghost_agent.core.agent import GhostAgent
    a = GhostAgent.__new__(GhostAgent)
    a.context = _Ctx()
    a.context.llm_client = SimpleNamespace(upstream_url=url)
    return a


_PAY = {"messages": [{"role": "user", "content": "hi"}]}


def _turn(a, level="", rid=RID, payload=_PAY, member=False):
    toks = [L.effort_override_context.set(level)]
    if member:
        toks.append(("m", L.requester_role_context.set(L.REQUESTER_MEMBER)))
    try:
        return a._with_turn_effort(payload, rid)
    finally:
        for t in reversed(toks):
            if isinstance(t, tuple):
                L.requester_role_context.reset(t[1])
            else:
                L.effort_override_context.reset(t)


def _known(level):
    E._server_default.update(level=level, at=time.monotonic(), busy=False)


@pytest.mark.parametrize("level", ["medium", "xhigh", "low"])
def test_a_default_turn_records_the_servers_level_and_sends_nothing(on, level):
    _known(level)
    a = _agent()
    assert _turn(a) is _PAY
    f = TF.facts_for(a.context, RID)
    assert f["reasoning_effort"] == level and f["reasoning_effort_default"] is True


def test_a_picked_level_is_recorded_as_picked(on):
    _known("medium")
    a = _agent()
    _turn(a, "xhigh")
    f = TF.facts_for(a.context, RID)
    assert f["reasoning_effort"] == "xhigh" and "reasoning_effort_default" not in f


def test_the_lookup_uses_the_servers_root_url(on, monkeypatch):
    seen = []
    monkeypatch.setattr(E, "server_default", lambda url: seen.append(url) or "medium")
    _turn(_agent("http://127.0.0.1:8088/v1/"))
    assert seen == ["http://127.0.0.1:8088"]


@pytest.mark.parametrize("case", ["off", "unknown", "thinking_off", "member", "job", "probe", "sub"])
def test_nothing_is_recorded(on, case, monkeypatch):
    _known("medium")
    a, rid, payload, member = _agent(), RID, _PAY, False
    if case == "off":
        monkeypatch.setenv("GHOST_REASONING_EFFORT", "0")
    elif case == "unknown":
        E._server_default.update(level="", at=0.0, busy=True)    # not looked up yet
    elif case == "thinking_off":
        payload = {"messages": [], "chat_template_kwargs": {"enable_thinking": False}}
    elif case == "member":
        member = True
    else:
        rid = {"job": "job-1", "probe": "probe-x1", "sub": "sub-leaf"}[case]
    _turn(a, rid=rid, payload=payload, member=member)
    assert "reasoning_effort" not in TF.facts_for(a.context, rid)


async def test_the_boot_warmup_primes_the_lookup_and_a_rewarm_does_not(on, monkeypatch, mock_context):
    from unittest.mock import AsyncMock, MagicMock
    from ghost_agent.core.agent import GhostAgent
    calls = []
    monkeypatch.setattr(E, "refresh_server_default", lambda url: calls.append(url) or "medium")
    llm = MagicMock()
    llm.upstream_url = "http://127.0.0.1:8088/v1"
    llm.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": "x"}}]})
    mock_context.llm_client = llm
    agent = GhostAgent(mock_context)
    await agent.warm_up_main_prefix(quiet=True)
    assert calls == []
    await agent.warm_up_main_prefix()
    assert calls == ["http://127.0.0.1:8088"]


# ── review r1: the per-request lookup and the real template ────────────────── #

@pytest.mark.parametrize("case,looked", [("default", True), ("picked", False), ("member", False),
                                         ("job", False), ("off", False)])
async def test_an_owner_request_at_default_looks_the_level_up_first(on, monkeypatch, case, looked):
    seen = []
    monkeypatch.setattr(E, "refresh_server_default", lambda url, timeout=3.0: seen.append((url, timeout)) or "xhigh")
    if case == "off":
        monkeypatch.setenv("GHOST_REASONING_EFFORT", "0")
    a = _agent()
    toks = [L.effort_override_context.set("low" if case == "picked" else "")]
    if case == "member":
        toks.append(("m", L.requester_role_context.set(L.REQUESTER_MEMBER)))
    try:
        await a._effort_default_refresh("job-1" if case == "job" else RID)
    finally:
        for t in reversed(toks):
            if isinstance(t, tuple):
                L.requester_role_context.reset(t[1])
            else:
                L.effort_override_context.reset(t)
    assert seen == ([("http://127.0.0.1:8088", 1.0)] if looked else [])


def test_the_lookup_runs_at_request_start():
    import ast, pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "src/ghost_agent/core/agent.py").read_text()
    fn = [f for f in ast.walk(ast.parse(src)) if isinstance(f, ast.AsyncFunctionDef) and f.name == "handle_chat"][0]
    awaited = [n.value.func.attr for n in ast.walk(fn) if isinstance(n, ast.Await)
               and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)]
    assert "_effort_default_refresh" in awaited


TEMPLATE = "/Users/vasilis/Data/AI/Models/qwen3.8/qwen38-27b-effort-moved.jinja"


@pytest.mark.parametrize("tools", [None, [{"type": "function", "function": {"name": "f", "parameters": {}}}]])
@pytest.mark.parametrize("kw,want", [({}, "xhigh"), ({"reasoning_effort": "low"}, "low"),
                                     ({"reasoning_effort": "medium"}, "medium"),
                                     ({"reasoning_effort": "xhigh"}, "xhigh")])
def test_the_real_template_reads_back(tools, kw, want):
    """Review r1: a reworded sentence would read as medium with no test failing."""
    import os
    if not os.path.exists(TEMPLATE):
        pytest.skip("27B template not on this machine")
    jinja2 = pytest.importorskip("jinja2")
    env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda v, indent=None: json.dumps(v, ensure_ascii=False, indent=indent)

    def _raise(msg):
        raise jinja2.TemplateError(msg)
    env.globals["raise_exception"] = _raise
    out = env.from_string(open(TEMPLATE).read()).render(
        messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "U"}], tools=tools,
        add_generation_prompt=True, bos_token="", eos_token="<|im_end|>", **kw)
    assert E.level_in_render(out) == want
    assert ("Reasoning effort is set to low" in env.from_string(open(TEMPLATE).read()).render(
        messages=[{"role": "user", "content": "U"}], tools=tools, add_generation_prompt=True,
        bos_token="", eos_token="<|im_end|>", reasoning_effort="low"))


# ── §4RW: one icon per level; a Default turn shows the server's level ──────── #

def _logged(monkeypatch):
    from ghost_agent.core import agent as A
    seen = []
    monkeypatch.setattr(A, "pretty_log", lambda title, msg=None, **kw: seen.append((title, str(msg), kw.get("icon"))))
    return seen


def test_each_level_has_its_own_icon():
    icons = [L.effort_icon(x) for x in ("low", "medium", "xhigh")]
    assert icons == [L.Icons.EFFORT_LOW, L.Icons.EFFORT_MEDIUM, L.Icons.EFFORT_XHIGH]
    assert len(set(icons)) == 3


@pytest.mark.parametrize("level", ["low", "medium", "xhigh"])
def test_a_picked_level_logs_its_own_icon_once(on, monkeypatch, level):
    _known("medium")
    seen = _logged(monkeypatch)
    a = _agent()
    _turn(a, level)
    _turn(a, level)                       # the request's next turn: not again
    assert seen == [("Reasoning Effort", f"{level} — picked for this request", L.effort_icon(level))]


@pytest.mark.parametrize("level", ["low", "medium", "xhigh"])
def test_a_default_turn_logs_the_server_level_with_that_levels_icon(on, monkeypatch, level):
    _known(level)
    seen = _logged(monkeypatch)
    a = _agent()
    _turn(a)
    _turn(a)
    assert seen == [("Reasoning Effort", f"{level} — the server default", L.effort_icon(level))]


def test_a_default_turn_with_an_unknown_level_logs_nothing(on, monkeypatch):
    E._server_default.update(level="", at=0.0, busy=True)
    seen = _logged(monkeypatch)
    _turn(_agent())
    assert seen == []


def test_health_reports_the_known_default_without_a_lookup(monkeypatch):
    # §4RW: the CLI bar shows Default with the server level's icon; health is
    # polled, so it reads the cache and never starts a lookup itself
    from ghost_agent.api import routes
    from ghost_agent.core import effort as E
    started = []
    monkeypatch.setattr(E.threading, "Thread", lambda *a, **k: started.append(1))
    monkeypatch.setitem(E._server_default, "level", "")
    assert routes._effort_default_known() == ""
    monkeypatch.setitem(E._server_default, "level", "xhigh")
    assert routes._effort_default_known() == "xhigh" and started == []


def test_the_level_is_read_from_the_system_block_only():
    """§4RW-f r2: a profile quoting the sentence, or a message, never sets it."""
    from ghost_agent.core import effort as E
    sys_low = "<|im_start|>system\nI said: Reasoning effort is set to xhigh, once.\n... Reasoning effort is set to low.<|im_end|>"
    assert E.level_in_render(sys_low) == "low"
    assert E.level_in_render("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nReasoning effort is set to low<|im_end|>") == "medium"
