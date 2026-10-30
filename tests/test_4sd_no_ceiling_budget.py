"""§4SD — no ceiling on a coding request that keeps making progress.

Req d886d047 (CLI, 27B at xhigh) was cut at the flat 60 min while it was still
writing files: the CLI sent no X-Ghost-Client-Timeout, so §4RA never extended
it. Now GHOST_MAX_REQUEST_EXTENDED_S=inf, the CLI and the web page's streamed
path send "inf", and the stall window is at least twice the request's own
longest turn. Each test names the world it fails in.
"""
import math
import pathlib
import time

import pytest

from ghost_agent.utils import logging as L
from ghost_agent.core.agent import _span, client_deadline_line

ROOT = pathlib.Path(__file__).resolve().parents[1]
RID = "4sd0test"
STEP = L.EXTENSION_STEP_S


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("GHOST_MAX_REQUEST_S", "3600")
    monkeypatch.setenv("GHOST_MAX_REQUEST_EXTENDED_S", "inf")
    tok = L.client_deadline_context.set(math.inf)          # web stream / CLI in production
    yield monkeypatch
    L.client_deadline_context.reset(tok)


def _open(elapsed, progress_ago=None, longest_turn=0.0):
    now = time.monotonic()
    st = {"started": now - elapsed, "origin": None, "longest_turn_s": longest_turn}
    if progress_ago is not None:
        st["progress_t"] = now - progress_ago
        st["progress_what"] = "wrote js/apps.js"
    with L._REQ_STATE_LOCK:
        L._REQ_STATE[RID] = st


@pytest.fixture
def req():
    yield _open
    with L._REQ_STATE_LOCK:
        L._REQ_STATE.pop(RID, None)


@pytest.mark.parametrize("raw", ["inf", "INF", "none", "unlimited", " inf "])
def test_the_env_spells_no_ceiling(env, raw):
    # world: "inf" parsed as a number nobody reaches but "none" falls to 0 (no extensions)
    env.setenv("GHOST_MAX_REQUEST_EXTENDED_S", raw)
    assert math.isinf(L.extension_ceiling_s(RID))


@pytest.mark.parametrize("raw,want", [("0", 3600), ("", 3600), ("nan", 3600), ("-5", 3600), ("10800", 10800)])
def test_finite_and_bad_values_keep_their_old_meaning(env, raw, want):
    env.setenv("GHOST_MAX_REQUEST_EXTENDED_S", raw)
    assert L.extension_ceiling_s(RID) == want


def test_the_base_budget_is_unchanged_by_an_infinite_client(env):
    # world: min(inf, cap) mis-handled — every web/CLI request runs forever, coding or not
    assert L.base_deadline_s(RID) == 3600
    assert math.isinf(L.extension_ceiling_s(RID))


def test_an_infinite_client_with_no_server_cap_is_no_deadline_not_inf(env):
    # world: base = inf leaks into arithmetic (remaining inf, report floor 0.25*inf)
    env.setenv("GHOST_MAX_REQUEST_S", "0")
    assert L.base_deadline_s(RID) == 0.0


def test_d886d047_would_have_been_extended(env, req):
    # world: the CLI's request is never extended (the bug): at +3143 s, 888 s
    # after its last append, the budget stayed 3600
    _open(3143, progress_ago=888, longest_turn=930)
    assert L.request_deadline_s(RID) >= 3600 + STEP


def test_progress_keeps_extending_far_past_the_old_ceiling(env, req):
    # world: the closed-form step math still stops at some finite room, or
    # int(inf // STEP) overflows under the lock
    _open(12 * 3600 + 123, progress_ago=60)
    d = L.request_deadline_s(RID)
    assert math.isfinite(d)
    assert d - (12 * 3600 + 123) >= STEP                       # at least a step ahead
    assert (d - 3600) % STEP == 0                      # whole steps


def test_a_stall_still_ends_it(env, req):
    # world: no ceiling also means no stop — a request that stopped writing runs forever
    _open(3300, progress_ago=1801)
    assert L.request_deadline_s(RID) == 3600


def test_the_stall_window_grows_with_the_longest_turn(env, req):
    # world: a 40-min thinking turn (no 16k cap) reads as a stall at 30 min
    _open(3300, progress_ago=3000, longest_turn=2400)
    assert L.request_deadline_s(RID) > 3600
    _open(3300, progress_ago=4801, longest_turn=2400)
    assert L.request_deadline_s(RID) == 3600


def test_note_turn_seconds_keeps_the_longest(env, req):
    _open(10)
    L.note_turn_seconds(RID, 500)
    L.note_turn_seconds(RID, 200)
    L.note_turn_seconds("not-open", 9999)               # no request → ignored, no raise
    assert L._REQ_STATE[RID]["longest_turn_s"] == 500


def test_the_budget_line_says_no_limit(env):
    # world: the log reads "extendable to inf min"
    assert _span(math.inf) == "no limit"
    assert "no time limit while code is being written" in client_deadline_line(3600, math.inf)
    assert "extendable to 180 min" in client_deadline_line(3600, 10800)


def _cli():
    import importlib.machinery
    import importlib.util
    import os
    path = pathlib.Path(os.environ.get("GHOST_CLI_PATH") or ROOT.parent / "bin/ghost")
    loader = importlib.machinery.SourceFileLoader("ghost_cli_4sd", str(path))
    spec = importlib.util.spec_from_file_location("ghost_cli_4sd", path, loader=loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


class _SSE:
    status_code = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_lines(self):
        return iter([b"data: [DONE]"])


def test_the_cli_sends_inf_on_both_chat_paths(monkeypatch):
    # world: the CLI keeps sending nothing (the d886d047 cause) on either path
    cli = _cli()
    api = cli.GhostAPI("http://localhost:9", "k")
    seen = []
    monkeypatch.setattr(api.http, "post", lambda *a, **k: seen.append(k.get("headers") or {}) or _SSE())
    api.chat_stream([{"role": "user", "content": "q"}], "s1", "r1")
    cli.one_shot(api, "q", None)
    assert len(seen) == 2
    assert all(h.get("X-Ghost-Client-Timeout") == "inf" for h in seen), seen


def test_the_turn_loop_records_each_turn(monkeypatch):
    # world: note_turn_seconds is defined but never called — the stall window
    # stays 30 min however long a turn thinks
    import asyncio
    import json as _json
    from unittest.mock import AsyncMock, MagicMock, patch
    import ghost_agent.core.agent as A
    ctx = MagicMock(spec=A.GhostContext)
    ctx.args = MagicMock(smart_memory=0.0, max_context=4000, temperature=0.7,
                         native_tools=False, tool_turn_max_tokens=0)
    for k in ("verifier", "self_model", "sandbox_dir", "memory_system", "profile_memory",
              "semantic_memory", "skill_memory", "journal"):
        setattr(ctx, k, None)
    ctx.llm_client = MagicMock()
    ctx.scratchpad = MagicMock()
    ctx.scratchpad.list_all.return_value = ""

    async def _stream(payload, *a, **kw):
        for c in ({"choices": [{"delta": {"content": "done"}}]},
                  {"choices": [{"delta": {}, "finish_reason": "stop"}]}):
            yield f"data: {_json.dumps(c)}\n\n".encode()
        yield b"data: [DONE]\n\n"
    ctx.llm_client.stream_chat_completion = _stream
    ctx.llm_client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]})
    seen = []
    monkeypatch.setattr(L, "note_turn_seconds", lambda rid, sec: seen.append((rid, sec)))
    with patch("ghost_agent.core.agent.get_active_tool_definitions", return_value=[]), \
            patch("ghost_agent.core.agent.pretty_log"):
        asyncio.run(A.GhostAgent(context=ctx).handle_chat(
            {"messages": [{"role": "user", "content": "write a.py"}], "stream": False},
            background_tasks=MagicMock(), request_id="req-4sd-turn"))
    assert seen and seen[0][0] == "req-4sd-turn" and seen[0][1] >= 0


def test_the_launcher_sets_no_ceiling(tmp_path):
    # world: the launcher still exports 10800 (a 3-hour ceiling)
    import subprocess
    # the launcher's own export lines, executed by bash (the value the agent inherits)
    out = subprocess.run(["bash", "-c", 'eval "$(grep "^export GHOST_MAX_REQUEST" "$0")"; '
                          'echo "$GHOST_MAX_REQUEST_EXTENDED_S"', str(ROOT.parent / "bin/start-ghost-agent.sh")],
                         capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"}).stdout.strip()
    import os
    old = os.environ.get("GHOST_MAX_REQUEST_EXTENDED_S")
    os.environ["GHOST_MAX_REQUEST_EXTENDED_S"] = out
    try:
        assert math.isinf(L._extended_ceiling_env()), out
    finally:
        if old is None:
            os.environ.pop("GHOST_MAX_REQUEST_EXTENDED_S", None)
        else:
            os.environ["GHOST_MAX_REQUEST_EXTENDED_S"] = old


# ── review: a stall is one turn plus a step; the same write again is no progress ──

def test_the_stall_window_is_one_turn_plus_a_step(env, req):
    # world: 2x the longest turn — 64k turns (~73 min at 15 tok/s) ran ~3.6 h
    # past the last progress before the request stopped
    turn = 4400.0
    _open(10_000, progress_ago=turn + STEP - 60, longest_turn=turn)
    assert L.request_deadline_s(RID) > 10_000                 # inside the window: still extending
    _open(10_000, progress_ago=turn + STEP + 60, longest_turn=turn)
    assert L.request_deadline_s(RID) <= 10_000 + STEP          # past it: no new step


def _hook(fname, op, target, a_hash):
    from ghost_agent.core import agent as A
    from tests.test_4ra_deadline_extension import _Ctx, _hook_code
    self = _Ctx()
    self.context = _Ctx()
    self.context._script_iter = {}
    outcome = _Ctx()
    outcome.is_rejection = False
    g = {**vars(A), "self": self, "fname": fname, "ptool_op": op, "is_mutating": True, "ptarget": target,
         "a_hash": a_hash, "_res_is_error": False, "_res_unresolved": False, "_outcome": outcome}
    tok = L.request_id_context.set(RID)
    try:
        exec(_hook_code(), g)
    finally:
        L.request_id_context.reset(tok)


def test_the_same_write_again_is_not_progress(env, req):
    # world: a loop rewriting one file with identical content never stalls
    _open(100)
    same = 'file_system:{"content": "const a = 1;", "operation": "write", "path": "app.js"}'
    _hook("file_system", "write", "app.js", same)
    first = L._REQ_STATE[RID]["progress_t"]
    L._REQ_STATE[RID]["progress_t"] = first - 500               # time passes
    _hook("file_system", "write", "app.js", same)
    assert L._REQ_STATE[RID]["progress_t"] == first - 500       # the repeat did not count
    _hook("file_system", "write", "app.js", same.replace("1;", "2;"))
    assert L._REQ_STATE[RID]["progress_t"] > first - 500        # new content counts


def _write_then_run(si, content):
    from tests.test_4ra_deadline_extension import _run_hook, _x, RID as _RA_RID
    from ghost_agent.core.agent import note_run_outcome
    w = 'file_system:{"content": "%s", "operation": "write", "path": "main.py"}' % content
    _run_futility(si, "file_system", w, target="main.py")
    rec = si["main.py"]
    rec["runs"] += 1
    note_run_outcome(rec, ok=True, output=f"served on a new path {'abcdefgh'[rec['runs']]}")  # new output: progress
    assert rec["progress"] is True
    _run_hook("execute", "", True, "", _x("python3 main.py"), si, False)   # (its own request id)
    try:
        return L.request_progress_what(_RA_RID) == "ran a file it wrote"
    finally:
        with L._REQ_STATE_LOCK:
            L._REQ_STATE.pop(_RA_RID, None)


def _run_futility(si, fname, a_hash, target):
    """The futility block that counts a write on the record (the earn hook reads it)."""
    from ghost_agent.core import agent as A
    import ast
    tree = ast.parse((ROOT / "src/ghost_agent/core/agent.py").read_text(encoding="utf-8"))
    tree = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)
                and n.name == "_dispatch_and_process_tool_batch")
    blk = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and isinstance(n.test, ast.BoolOp)
               and any(isinstance(v, ast.Compare) and isinstance(v.left, ast.Name) and v.left.id == "fname"
                       and any(isinstance(c, ast.Constant) and c.value == "file_system" for c in v.comparators)
                       for v in n.test.values)
               and any(isinstance(x, ast.Name) and x.id == "_bn" for x in ast.walk(n)))
    g = {**vars(A), "_si": si, "fname": fname, "is_mutating": True, "ptarget": target,
         "_res_is_error": False, "a_hash": a_hash, "_CODE_EXTS": (".py", ".sh", ".js", ".mjs", ".ts")}
    exec(compile(ast.fix_missing_locations(ast.Module(body=blk.body, type_ignores=[])), "<futility-write>", "exec"), g)


def test_the_same_write_then_a_run_earns_once_not_every_cycle(env, req):
    # world (review): an identical (or A/B alternating) write re-armed the run's
    # earn — write, run, new output, repeat earned a step each cycle
    _open(100)
    si = {}
    assert _write_then_run(si, "v1") is True
    assert _write_then_run(si, "v1") is False                    # the same write again
    assert _write_then_run(si, "v2") is True                     # a new write earns
    assert _write_then_run(si, "v1") is False                    # back to an old one: no
    assert si["main.py"]["writes"] == 4                          # the churn count still sees every write


def test_note_code_progress_reports_whether_it_counted(env, req):
    _open(100)
    assert L.note_code_progress(RID, "wrote a.js", key="k1") is True
    assert L.note_code_progress(RID, "wrote a.js", key="k1") is False
    assert L.note_code_progress(RID, "ran a file it wrote") is True     # runs carry no key
    assert L.note_code_progress("never-opened", "x", key="k2") is False
