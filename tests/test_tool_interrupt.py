"""§4RZ step 5: the owner stops the running tool call(s); the turn goes on.

Pins: the stop object, the batch's stoppable gather (parity with gather when
off), the turn end to end (finished calls keep their results, the running one
is cancelled and the model reads that the user stopped it), the wording
(stopped only when a killer confirmed it), the three killers (sandbox job,
browser runner, yt-dlp), and the HTTP route's sender check.
"""
import asyncio
import json
import contextvars
import subprocess
import sys
import threading
import time

import pytest

#: how long a "running" test tool waits — far past any interrupt, short
#: enough that a broken stop fails in seconds, not minutes
_HANG = 8

from ghost_agent.utils import tool_stop as TS
from ghost_agent.core.turns import get_turn_registry
from tests.test_requester_role import _agent as _loop_agent, _resp, _tc
from tests.test_queued_messages import _running
from tests.helpers import FakeBgTasks
from unittest.mock import AsyncMock


# ── the stop object ─────────────────────────────────────────────────────────

def test_a_stop_is_requested_once_and_runs_its_killers():
    s = TS.ToolStop()
    ran = threading.Event()
    remove = s.on_stop(ran.set)
    assert not s.requested()
    assert s.request() is True and s.requested()
    assert ran.wait(2.0)
    assert s.request() is False                       # a second stop is not a new one
    remove()
    late = threading.Event()
    s.on_stop(late.set)                               # registered after the stop: runs at once
    assert late.wait(2)


def test_a_removed_killer_never_runs():
    s = TS.ToolStop()
    ran = []
    s.on_stop(lambda: ran.append(1))()
    s.request()
    time.sleep(0.1)
    assert ran == []


def test_the_wording_says_only_what_is_certain():
    t = TS.interrupted_text("execute", started=True, killed=True)
    assert t.startswith("INTERRUPTED BY THE USER") and "it was stopped" in t
    t = TS.interrupted_text("web_search", started=True, killed=False)
    assert "it was stopped" not in t and "may still finish in the background" in t
    t = TS.interrupted_text("vision_analysis", started=False, killed=False)
    assert t.startswith("NOT RUN") and "before this vision_analysis call started" in t
    for t in (TS.interrupted_text("x", started=a, killed=b) for a in (0, 1) for b in (0, 1)):
        assert "Do not run it again unless the user asks" in t
    # §4SB: a call that RAN may have left half-written files; one that never
    # began wrote nothing, so it is not told to check
    for killed in (True, False):
        t = TS.interrupted_text("execute", started=True, killed=killed)
        assert "Files it was writing may be incomplete" in t and t.index("incomplete") < t.index("Do not run")
        assert TS.is_user_interrupted(t)
    assert "Files it was writing" not in TS.interrupted_text("execute", started=False, killed=False)


def test_nothing_is_bound_by_default():
    assert TS.current() is None and TS.requested() is False
    TS.note_killed("execute")                         # a no-op, never raises


# ── the stoppable gather ────────────────────────────────────────────────────

async def test_with_no_stop_it_is_the_plain_gather():
    from ghost_agent.core.agent import _gather_stoppable

    async def ok():
        return 1

    async def bad():
        raise ValueError("x")
    out = await _gather_stoppable([(0, ok()), (1, bad())], None)
    assert out[0] == 1 and isinstance(out[1], ValueError)


async def test_a_stop_keeps_finished_results_and_cancels_the_rest():
    from ghost_agent.core.agent import _gather_stoppable, _stoppable_call
    stop, cancelled = TS.ToolStop(), []

    async def fast():
        return "done"

    async def slow():
        try:
            await asyncio.sleep(_HANG)
        except asyncio.CancelledError:
            cancelled.append(1)
            raise
    items = [(0, _stoppable_call(0, "a", fast(), stop)), (1, _stoppable_call(1, "b", slow(), stop))]
    asyncio.get_running_loop().call_later(0.3, stop.request)
    t0 = time.monotonic()
    out = await _gather_stoppable(items, stop)
    assert time.monotonic() - t0 < 3
    assert out[0] == "done" and isinstance(out[1], TS.Interrupted) and out[1].started
    assert cancelled == [1]


async def test_a_group_after_the_stop_never_starts():
    from ghost_agent.core.agent import _gather_stoppable
    stop = TS.ToolStop()
    stop.request()
    ran = []

    async def never():
        ran.append(1)
    out = await _gather_stoppable([(0, never())], stop)
    assert ran == [] and isinstance(out[0], TS.Interrupted) and not out[0].started


async def test_the_turn_being_cancelled_still_cancels_every_call():
    from ghost_agent.core.agent import _gather_stoppable
    cancelled = []

    async def slow():
        try:
            await asyncio.sleep(_HANG)
        except asyncio.CancelledError:
            cancelled.append(1)
            raise
    task = asyncio.ensure_future(_gather_stoppable([(0, slow()), (1, slow())], TS.ToolStop()))
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    assert cancelled == [1, 1]


async def test_the_turn_being_cancelled_runs_the_killers():
    """§4SB: a hard cancel (or the proxy's hang-up cancelling the task) while
    a batch runs must REQUEST the stop — cancelling the asyncio tasks never
    reaches a worker thread's subprocess; only the killers do. Pinned on a
    registered killer and on the real yt-dlp killer in a worker thread."""
    from ghost_agent.core.agent import _gather_stoppable, _stoppable_call
    from ghost_agent.memory import youtube_ingest as Y
    stop, registered = TS.ToolStop(), threading.Event()

    async def ytdlp_in_a_worker():
        stop.on_stop(registered.set)
        # to_thread copies the context: the worker sees the bound stop
        return await asyncio.to_thread(
            Y._run_subprocess, [sys.executable, "-c", "import time; time.sleep(30)"], 60, None)
    task = asyncio.ensure_future(
        _gather_stoppable([(0, _stoppable_call(0, "knowledge_base", ytdlp_in_a_worker(), stop))], stop))
    await asyncio.sleep(0.5)
    t0 = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stop.requested()
    assert registered.wait(TS.KILL_CONFIRM_S)
    while not stop.killed(0) and time.monotonic() - t0 < TS.KILL_CONFIRM_S:
        await asyncio.sleep(0.05)
    assert stop.killed(0), "the yt-dlp subprocess outlived the cancelled turn"
    assert time.monotonic() - t0 < TS.KILL_CONFIRM_S


# ── the turn, end to end ────────────────────────────────────────────────────

async def _turn_with_an_interrupt(monkeypatch, tmp_path, *, on=True, killer=False, vision=False,
                                  rid="web-ti1", failing=False):
    if on:
        monkeypatch.setenv("GHOST_QUEUED_MESSAGES", "1")
    else:
        monkeypatch.delenv("GHOST_QUEUED_MESSAGES", raising=False)
    agent, ctx, _ = _loop_agent(monkeypatch, tmp_path)
    reg = get_turn_registry(agent)
    seen = {"cancelled": False, "vision_ran": False}

    async def _search(**kw):
        return "### 1. a hit"

    async def _research(**kw):
        t = reg.get(rid)
        seen["tool_stop_bound"] = t.tool_stop is not None
        asyncio.get_running_loop().call_later(0.2, lambda: seen.setdefault(
            "interrupted", reg.interrupt_tools(rid)))
        try:
            await asyncio.sleep(5 if not on else _HANG)
            return "research finished"
        except asyncio.CancelledError:
            seen["cancelled"] = True
            if killer:
                TS.note_killed("deep_research")
            raise

    async def _vision(**kw):
        seen["vision_ran"] = True
        return "a picture"

    async def _weather(**kw):
        return "Error: the weather service answered 503"
    agent.available_tools = {"web_search": _search, "deep_research": _research,
                             "vision_analysis": _vision, "check_weather": _weather}
    calls = [_tc("c1", "web_search", {"query": "news"}), _tc("c2", "deep_research", {"query": "x"})]
    if vision:
        calls.append(_tc("c3", "vision_analysis", {"image_path": "a.png"}))
    if failing:
        calls.append(_tc("c4", "check_weather", {"location": "Athens"}))
    payloads = []
    replies = iter([_resp("", calls), _resp("Here is what I found."), _resp("x"), _resp("x")])

    async def _complete(*a, **k):
        payloads.append(a[0] if a and isinstance(a[0], dict) else k)
        seen.setdefault("loop_bound", []).append(TS.current())      # the loop's own context
        return next(replies)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_complete)
    body = {"session_id": "s1", "messages": [{"role": "user", "content": "research the news"}]}
    final, _, _ = await agent.handle_chat(body, FakeBgTasks(), request_id=rid, requester_role="owner")
    return final, payloads, seen, reg


def _second_prompt(payloads):
    return str(payloads[1]["messages"])


async def test_an_interrupt_stops_the_running_call_and_the_turn_goes_on(monkeypatch, tmp_path):
    final, payloads, seen, reg = await _turn_with_an_interrupt(monkeypatch, tmp_path)
    assert final == "Here is what I found."
    assert seen["tool_stop_bound"] and seen["cancelled"]
    assert seen["interrupted"] == ["deep_research"]                   # web_search had finished
    assert seen["loop_bound"] == [None, None]                         # r1 m3: never in the loop's context
    assert "STILL RUNNING" not in _second_prompt(payloads)
    text = _second_prompt(payloads)
    assert "### 1. a hit" in text                                    # the finished call keeps its result
    assert "INTERRUPTED BY THE USER: the user stopped this deep_research call" in text
    assert "may still finish in the background" in text and "and it was stopped" not in text
    assert "research finished" not in text
    assert reg.get("web-ti1") is None
    assert TS.current() is None                                       # unbound after the batch


async def test_a_confirmed_kill_is_reported_as_stopped(monkeypatch, tmp_path):
    _, payloads, *_ = await _turn_with_an_interrupt(monkeypatch, tmp_path, killer=True, rid="web-ti2")
    assert "deep_research call while it was running, and it was stopped" in _second_prompt(payloads)


async def test_a_vision_read_after_the_stop_never_runs(monkeypatch, tmp_path):
    _, payloads, seen, _ = await _turn_with_an_interrupt(monkeypatch, tmp_path, vision=True, rid="web-ti3")
    assert not seen["vision_ran"]
    assert "NOT RUN: the user interrupted the tool batch before this vision_analysis call started" \
        in _second_prompt(payloads)


async def test_off_binds_no_stop_and_nothing_interrupts(monkeypatch, tmp_path):
    from ghost_agent.utils import tool_stop as _ts
    monkeypatch.setattr(_ts, "STOP_GRACE_S", 0.1)
    final, payloads, seen, _ = await _turn_with_an_interrupt(monkeypatch, tmp_path, on=False, rid="web-ti4")
    assert seen["tool_stop_bound"] is False and seen["interrupted"] == []
    assert not seen["cancelled"] and "research finished" in _second_prompt(payloads)


def test_the_registry_interrupts_only_a_running_batch():
    from ghost_agent.core.turns import TurnRegistry
    reg = TurnRegistry()
    t = _running(reg, "r1")
    assert reg.interrupt_tools("r1") == []                 # no batch
    t.tool_stop, t.tools_now = TS.ToolStop(), ["web_search", "execute"]
    assert reg.interrupt_tools("r1") == []                 # a batch, nothing running now
    assert not t.tool_stop.requested()
    t.tool_stop.enter(1, "execute")
    assert reg.interrupt_tools("r1") == ["execute"] and t.tool_stop.requested()
    assert reg.interrupt_tools("r1") == []                 # already sent
    t.tool_stop = TS.ToolStop()
    t.tool_stop.enter(0, "execute")
    reg.cancel("r1")
    assert reg.interrupt_tools("r1") == []                 # a stopped turn
    assert reg.interrupt_tools("nope") == []


# ── the killers ─────────────────────────────────────────────────────────────

def _in_bound_context(stop, fn, *a, **k):
    """Run ``fn`` in a thread whose context carries ``stop`` — what
    `asyncio.to_thread` does for a tool's worker."""
    ctx = contextvars.copy_context()
    ctx.run(TS.bind_call, stop, 0)                   # call 0 of the batch
    box = {}
    th = threading.Thread(target=lambda: box.setdefault("r", ctx.run(fn, *a, **k)))
    th.start()
    return th, box


def test_a_sandbox_job_is_killed_on_a_stop(tmp_path):
    from tests.test_sandbox_job_promotion import FakeSandbox, _kwargs, _alive
    from ghost_agent.sandbox.jobs import SandboxJobSupervisor
    sup = SandboxJobSupervisor(FakeSandbox(tmp_path))
    stop = TS.ToolStop()
    t0 = time.monotonic()
    th, box = _in_bound_context(stop, sup.run, "sleep 30", timeout=60, exec_kwargs=_kwargs(sup))
    time.sleep(1.0)
    stop.request()
    th.join(10)
    assert not th.is_alive() and time.monotonic() - t0 < 10
    _out, code, job = box["r"]
    assert code == 130 and job is None and stop.killed(0) and not stop.killed(1)


def test_a_sandbox_job_with_no_stop_is_untouched(tmp_path):
    from tests.test_sandbox_job_promotion import FakeSandbox, _kwargs
    from ghost_agent.sandbox.jobs import SandboxJobSupervisor
    sup = SandboxJobSupervisor(FakeSandbox(tmp_path))
    out, code, job = sup.run("echo hello", timeout=30, exec_kwargs=_kwargs(sup))
    assert code == 0 and "hello" in str(out)


def test_yt_dlp_is_killed_on_a_stop():
    from ghost_agent.memory import youtube_ingest as Y
    stop = TS.ToolStop()
    t0 = time.monotonic()
    th, box = _in_bound_context(stop, Y._run_subprocess, [sys.executable, "-c", "import time; time.sleep(30)"],
                                60, None)
    time.sleep(0.5)
    stop.request()
    th.join(10)
    assert not th.is_alive() and time.monotonic() - t0 < 10
    code, _out, err = box["r"]
    assert code == 130 and "stopped by the user" in err and stop.killed(0)


def test_yt_dlp_with_no_stop_runs_and_times_out_as_before():
    from ghost_agent.memory import youtube_ingest as Y
    code, out, _ = Y._run_subprocess([sys.executable, "-c", "print('hi')"], 30, None)
    assert code == 0 and out.strip() == "hi"
    code, _, err = Y._run_subprocess([sys.executable, "-c", "import time; time.sleep(5)"], 0.5, None)
    assert code == 124 and "timed out" in err


async def test_the_browser_runner_is_killed_while_its_call_holds_the_lock():
    from ghost_agent.tools import browser_support as BS
    release = threading.Event()
    kills = []

    class Mgr:
        def execute(self, cmd, timeout=600, **kw):
            if cmd == BS._RUNNER_KILL_CMD:
                kills.append(cmd)
                release.set()
                return "k\n", 0
            release.wait(30)
            return "killed", 143
    mgr, stop = Mgr(), TS.ToolStop()
    tok = TS.bind_call(stop, 3)
    try:
        task = asyncio.ensure_future(BS._exec_holding_lock(mgr.execute, "python .browser_runner.py {}"))
        await asyncio.sleep(0.2)
        stop.request()
        out, code = await asyncio.wait_for(task, 10)
    finally:
        TS.unbind_call(tok)
    # keyed by the CALL: a sibling call of the same tool was not stopped
    assert kills and stop.killed(3) and not stop.killed(0) and code == 143
    # once the call is over its killer is gone: a later stop kills nothing
    s2 = TS.ToolStop()
    tok = TS.bind_call(s2, 4)
    try:
        release.clear()
        release.set()
        await BS._exec_holding_lock(mgr.execute, "python .browser_runner.py {}")
    finally:
        TS.unbind_call(tok)
    assert s2.settled([4])                           # disarmed once the call is over
    s2.request()
    await asyncio.sleep(0.2)
    assert len(kills) == 1


def test_the_kill_command_matches_only_the_runner(tmp_path):
    """Run the real kill script over a fake /proc: only the runner's exact
    argv is hit — never a test of a file named like it, an editor, or the
    matcher itself (s5-r8)."""
    import shlex
    from ghost_agent.tools import browser_support as BS
    runner = BS.runner_cmd(".browser_runner.py", {"url": "x"}, "navigate", 30, 1000)
    procs = {"101": shlex.split(runner),
             "102": ["python3", "-m", "pytest", "tests/test_browser_runner.py"],
             "103": ["vim", ".browser_runner.py"],
             "104": ["python3", "-u", "app/.browser_runner.py.bak", "{}"],
             "105": shlex.split(BS._RUNNER_KILL_CMD)}
    for pid, argv in procs.items():
        (tmp_path / pid).mkdir()
        (tmp_path / pid / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    script = shlex.split(BS._RUNNER_KILL_CMD)[2]
    script = script.replace("/proc/", str(tmp_path) + "/").replace("kill -TERM", "echo killed")
    out = subprocess.run(["sh", "-c", script], capture_output=True, text=True, timeout=10).stdout.split()
    assert out == ["killed", "101", "k"], out


# ── HTTP ────────────────────────────────────────────────────────────────────
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402
from tests.test_sessions_and_cancel import _make_app  # noqa: E402


class TestInterruptRoute:
    def test_off(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GHOST_QUEUED_MESSAGES", raising=False)
        app, agent = _make_app(tmp_path)
        t = _running(agent.turn_registry)
        t.tool_stop, t.tools_now = TS.ToolStop(), ["execute"]
        t.tool_stop.enter(0, "execute")
        with TestClient(app) as c:
            r = c.post("/api/turn/interrupt", json={"session_id": "s1"})
        assert r.status_code == 404 and not t.tool_stop.requested()

    def test_paths_and_sender_check(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GHOST_QUEUED_MESSAGES", "1")
        app, agent = _make_app(tmp_path)
        reg = agent.turn_registry
        t = _running(reg)
        with TestClient(app) as c:
            post = lambda sid="s1", **h: c.post("/api/turn/interrupt", json={"session_id": sid}, headers=h)
            assert post().status_code == 409                       # no tool running
            t.tool_stop, t.tools_now = TS.ToolStop(), ["execute"]
            t.tool_stop.enter(0, "execute")
            # another conversation, a member, a probe, a Slack key: never
            assert post("s2").status_code == 404
            assert post(**{"X-Ghost-Requester": "member", "X-Ghost-Member": "U1"}).status_code == 404
            assert post(**{"X-Ghost-Origin": "probe"}).status_code == 404
            assert post("slack:C1:U1").status_code == 404
            assert not t.tool_stop.requested()
            r = post()
            assert r.status_code == 200 and r.json() == {"request_id": "r1", "interrupted": ["execute"]}
            assert t.tool_stop.requested() and not reg.is_cancelled("r1")   # the turn goes on
            assert post().status_code == 409                       # already sent


async def test_a_stop_never_cuts_a_file_write_but_skips_the_next(monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_QUEUED_MESSAGES", "1")
    agent, ctx, _ = _loop_agent(monkeypatch, tmp_path)
    reg = get_turn_registry(agent)
    done = []

    async def _fs(**kw):
        if not done:
            assert reg.interrupt_tools("web-ti5") == ["file_system"]   # the one running
            await asyncio.sleep(0.3)               # the write in progress finishes
        done.append(kw.get("path"))
        return f"wrote {kw.get('path')}"
    agent.available_tools = {"file_system": _fs}
    payloads = []
    replies = iter([_resp("", [_tc("c1", "file_system", {"operation": "write", "path": "a.txt", "content": "1"}),
                               _tc("c2", "file_system", {"operation": "write", "path": "b.txt", "content": "2"})]),
                    _resp("Done."), _resp("x")])

    bound = []

    async def _complete(*a, **k):
        payloads.append(a[0] if a and isinstance(a[0], dict) else k)
        bound.append(TS.current())
        return next(replies)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_complete)
    body = {"session_id": "s1", "messages": [{"role": "user", "content": "write two files"}]}
    await agent.handle_chat(body, FakeBgTasks(), request_id="web-ti5", requester_role="owner")
    assert done == ["a.txt"]
    assert bound == [None, None]          # a file call runs in the loop's context: unbound after it
    text = str(payloads[1]["messages"])
    assert "wrote a.txt" in text and "NOT RUN" in text and "before this file_system call started" in text
    assert "The user asked to stop the tools while this file_system call was running" in text


async def test_a_lone_write_stop_reaches_the_model(monkeypatch, tmp_path):
    """s5-r9: the stop landed on a batch's only call, a write — never cut, so
    no slot was Interrupted and the next prompt held no trace of the stop."""
    monkeypatch.setenv("GHOST_QUEUED_MESSAGES", "1")
    agent, ctx, _ = _loop_agent(monkeypatch, tmp_path)
    reg = get_turn_registry(agent)

    async def _fs(**kw):
        assert reg.interrupt_tools("web-ti9") == ["file_system"]
        await asyncio.sleep(0.05)
        return f"wrote {kw.get('path')}"
    agent.available_tools = {"file_system": _fs}
    payloads = []
    replies = iter([_resp("", [_tc("c1", "file_system", {"operation": "write", "path": "a.txt", "content": "1"})]),
                    _resp("Done."), _resp("x")])

    async def _complete(*a, **k):
        payloads.append(a[0] if a and isinstance(a[0], dict) else k)
        return next(replies)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_complete)
    body = {"session_id": "s1", "messages": [{"role": "user", "content": "write a file"}]}
    await agent.handle_chat(body, FakeBgTasks(), request_id="web-ti9", requester_role="owner")
    text = str(payloads[1]["messages"])
    assert "wrote a.txt" in text and "the result above is real" in text
    assert "INTERRUPTED BY THE USER" not in text and "NOT RUN" not in text   # it finished: never "stopped"


def test_the_uncut_note_is_a_tail_not_a_stop():
    from ghost_agent.tools.outcome import ToolOutcome, OutcomeStatus, append_note
    r = append_note(ToolOutcome("wrote a.txt", status=OutcomeStatus.OK), TS.uncut_note("file_system"))
    assert r.status is OutcomeStatus.OK and str(r).startswith("wrote a.txt")
    assert not TS.is_user_interrupted(r)


async def test_a_call_waiting_in_its_ordered_chain_is_not_run(monkeypatch, tmp_path):
    """Two `execute` calls of one batch run in order; the second, still
    waiting on the first when the stop lands, never started."""
    monkeypatch.setenv("GHOST_QUEUED_MESSAGES", "1")
    agent, ctx, _ = _loop_agent(monkeypatch, tmp_path)
    reg = get_turn_registry(agent)
    ran = []

    async def _exec(**kw):
        ran.append(kw.get("command") or kw.get("code"))
        asyncio.get_running_loop().call_later(0.2, lambda: reg.interrupt_tools("web-ti6"))
        await asyncio.sleep(_HANG)
    agent.available_tools = {"execute": _exec}
    payloads = []
    replies = iter([_resp("", [_tc("c1", "execute", {"command": "sleep 60"}),
                               _tc("c2", "execute", {"command": "echo two"})]),
                    _resp("Stopped."), _resp("x")])

    async def _complete(*a, **k):
        payloads.append(a[0] if a and isinstance(a[0], dict) else k)
        return next(replies)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_complete)
    body = {"session_id": "s1", "messages": [{"role": "user", "content": "run two commands"}]}
    await agent.handle_chat(body, FakeBgTasks(), request_id="web-ti6", requester_role="owner")
    assert len(ran) == 1
    text = str(payloads[1]["messages"])
    assert "INTERRUPTED BY THE USER: the user stopped this execute call" in text
    assert "NOT RUN: the user interrupted the tool batch before this execute call started" in text


def test_an_unconfirmed_job_kill_is_not_reported_as_stopped(tmp_path, monkeypatch):
    from tests.test_sandbox_job_promotion import FakeSandbox, _kwargs
    from ghost_agent.sandbox.jobs import SandboxJobSupervisor
    sup = SandboxJobSupervisor(FakeSandbox(tmp_path))
    real_kill = sup._kill_pgroup
    monkeypatch.setattr(sup, "_kill_pgroup", lambda pid: real_kill(pid) and False)  # survived TERM+KILL
    stop = TS.ToolStop()
    th, box = _in_bound_context(stop, sup.run, "sleep 30", timeout=60, exec_kwargs=_kwargs(sup))
    time.sleep(1.0)
    stop.request()
    th.join(10)
    assert box["r"][1] == 130 and not stop.killed(0)


async def test_a_browser_kill_that_matched_nothing_is_not_reported():
    from ghost_agent.tools import browser_support as BS
    release = threading.Event()

    class Mgr:
        def execute(self, cmd, timeout=600, **kw):
            if cmd == BS._RUNNER_KILL_CMD:
                release.set()
                return "", 0                       # no runner matched
            release.wait(30)
            return "done", 0
    mgr, stop = Mgr(), TS.ToolStop()
    tok = TS.bind_call(stop, 3)
    try:
        task = asyncio.ensure_future(BS._exec_holding_lock(mgr.execute, "python .browser_runner.py {}"))
        await asyncio.sleep(0.2)
        stop.request()
        await asyncio.wait_for(task, 10)
    finally:
        TS.unbind_call(tok)
    assert not stop.killed(3)


def test_a_member_cannot_interrupt_even_their_own_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_QUEUED_MESSAGES", "1")
    app, agent = _make_app(tmp_path)
    t = _running(agent.turn_registry, "m1", kind="member")
    t.tool_stop, t.tools_now = TS.ToolStop(), ["web_search"]
    t.tool_stop.enter(0, "web_search")
    with TestClient(app) as c:
        r = c.post("/api/turn/interrupt", json={"session_id": "s1"},
                   headers={"X-Ghost-Requester": "member", "X-Ghost-Member": "U1"})
    assert r.status_code == 404 and not t.tool_stop.requested()



# ── review r1 ───────────────────────────────────────────────────────────────

def test_a_stopped_call_is_unresolved_never_a_success_or_a_failure():
    """r1 M1: a plain-string result was booked OK — memoised (the user's
    re-run answered "ALREADY READ"), recorded as applied (a re-run refused
    "already applied"), and counted as a passing run by the code ledger."""
    from ghost_agent.tools.outcome import OutcomeStatus
    from ghost_agent.core.code_claims import _exec_ok
    for started in (True, False):
        o = TS.interrupted_outcome("execute", started=started, killed=started)
        assert o.status is OutcomeStatus.UNRESOLVED and o.reason_code == TS.REASON
        assert not o.may_record_as_applied
        assert o.world_changed is started
        assert _exec_ok({"content": o}) is None


async def test_the_users_later_rerun_of_a_stopped_search_runs(monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_QUEUED_MESSAGES", "1")
    agent, ctx, _ = _loop_agent(monkeypatch, tmp_path)
    reg = get_turn_registry(agent)
    calls = []

    async def _search(**kw):
        calls.append(kw)
        if len(calls) == 1:
            asyncio.get_running_loop().call_later(0.2, lambda: reg.interrupt_tools("web-ti7"))
            await asyncio.sleep(_HANG)
        return "### 1. the real hit"
    agent.available_tools = {"web_search": _search}
    payloads = []
    same = {"query": "lisbon weather"}
    replies = iter([_resp("", [_tc("c1", "web_search", same)]), _resp("", [_tc("c2", "web_search", same)]),
                    _resp("Here it is."), _resp("x")])

    async def _complete(*a, **k):
        payloads.append(a[0] if a and isinstance(a[0], dict) else k)
        return next(replies)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_complete)
    body = {"session_id": "s1", "messages": [{"role": "user", "content": "weather in lisbon"}]}
    await agent.handle_chat(body, FakeBgTasks(), request_id="web-ti7", requester_role="owner")
    assert len(calls) == 2
    assert "the real hit" in str(payloads[2]["messages"]) and "ALREADY READ" not in str(payloads[2]["messages"])


async def test_a_job_kill_through_the_batch_is_reported_as_stopped(monkeypatch, tmp_path):
    """r1 M2: the worker thread notices the stop after the batch moved on —
    the batch waits for the armed killer, so a confirmed kill says so."""
    from tests.test_sandbox_job_promotion import FakeSandbox, _kwargs
    from ghost_agent.sandbox.jobs import SandboxJobSupervisor
    monkeypatch.setenv("GHOST_QUEUED_MESSAGES", "1")
    agent, ctx, _ = _loop_agent(monkeypatch, tmp_path)
    reg = get_turn_registry(agent)
    sup = SandboxJobSupervisor(FakeSandbox(tmp_path / "box"))

    async def _exec(**kw):
        asyncio.get_running_loop().call_later(1.0, lambda: reg.interrupt_tools("web-ti8"))
        out, code, _ = await asyncio.to_thread(sup.run, "sleep 30", timeout=60, exec_kwargs=_kwargs(sup))
        return f"EXIT CODE: {code}\n{out}"
    agent.available_tools = {"execute": _exec}
    payloads = []
    replies = iter([_resp("", [_tc("c1", "execute", {"command": "sleep 30"})]), _resp("Stopped."), _resp("x")])

    async def _complete(*a, **k):
        payloads.append(a[0] if a and isinstance(a[0], dict) else k)
        return next(replies)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_complete)
    body = {"session_id": "s1", "messages": [{"role": "user", "content": "run it"}]}
    t0 = time.monotonic()
    await agent.handle_chat(body, FakeBgTasks(), request_id="web-ti8", requester_role="owner")
    assert time.monotonic() - t0 < 15
    assert "execute call while it was running, and it was stopped" in str(payloads[1]["messages"])


async def test_a_browser_call_after_the_stop_never_starts_a_runner():
    """r1 m1: the kill ran on the event loop, before the runner existed."""
    from ghost_agent.tools import browser_support as BS
    ran = []

    class Mgr:
        def execute(self, cmd, timeout=600, **kw):
            ran.append(cmd)
            return "", 0
    stop = TS.ToolStop()
    stop.request()
    tok = TS.bind(stop)
    try:
        with pytest.raises(asyncio.CancelledError):
            await BS._exec_holding_lock(Mgr().execute, "python .browser_runner.py {}")
    finally:
        TS.unbind(tok)
    await asyncio.sleep(0.1)
    assert ran == [] and stop.settled(["browser"])


async def test_a_call_that_swallows_its_cancel_is_still_reported_stopped():
    """r1 m4: a tool that caught the cancel and returned its own error."""
    from ghost_agent.core.agent import _gather_stoppable, _stoppable_call
    stop = TS.ToolStop()

    async def stubborn():
        try:
            await asyncio.sleep(_HANG)
        except asyncio.CancelledError:
            return "Error: the search was cancelled"
    asyncio.get_running_loop().call_later(0.2, stop.request)
    out = await _gather_stoppable([(0, _stoppable_call(0, "web_search", stubborn(), stop))], stop)
    assert isinstance(out[0], TS.Interrupted) and out[0].started


def test_a_killer_registered_after_the_stop_runs_off_the_callers_thread():
    s = TS.ToolStop()
    s.request()
    where = []
    done = threading.Event()
    s.on_stop(lambda: (where.append(threading.current_thread().name), done.set()))
    assert done.wait(2) and where == ["ghost-tool-stop"]



def test_an_owner_slack_conversation_has_no_interrupt(tmp_path, monkeypatch):
    """Slack stays as it is: even the owner's own turn in a Slack thread."""
    monkeypatch.setenv("GHOST_QUEUED_MESSAGES", "1")
    app, agent = _make_app(tmp_path)
    t = _running(agent.turn_registry, "sl1", session="slack:C1:U1")
    t.tool_stop, t.tools_now = TS.ToolStop(), ["web_search"]
    t.tool_stop.enter(0, "web_search")
    with TestClient(app) as c:
        r = c.post("/api/turn/interrupt", json={"session_id": "slack:C1:U1"})
    assert r.status_code == 404 and not t.tool_stop.requested()


# ── review r2 pins ──────────────────────────────────────────────────────────

def test_a_stopped_call_is_never_listed_as_succeeded_in_the_multi_op_summary():
    from ghost_agent.tools.tool_failure import summarize_multi_op_outcomes
    out = summarize_multi_op_outcomes([
        {"tool": "file_system", "ok": True},
        {"tool": "web_search", "ok": False, "preview": "Error: 503"},
        {"tool": "execute", "ok": False, "interrupted": True},
    ])
    head, _, rest = out.partition("STOPPED BY THE USER")
    assert "1 of 2 call(s) SUCCEEDED" in out and rest and "execute" in rest
    assert "execute" not in head                       # neither SUCCEEDED nor FAILED


def test_the_stopped_outcome_is_recognised_by_text_and_by_reason():
    from ghost_agent.distill.outcome_heuristics import is_unresolved_tool_result
    for started, killed in ((True, True), (True, False), (False, False)):
        o = TS.interrupted_outcome("execute", started=started, killed=killed)
        assert TS.is_user_interrupted(o) and TS.is_user_interrupted(str(o))
        assert is_unresolved_tool_result(str(o))      # a stored row: never a success
    assert not TS.is_user_interrupted("Error: yt-dlp failed (exit 130)")
    assert not TS.is_user_interrupted("ok")


async def test_a_call_killed_inside_the_poll_gap_is_still_reported_stopped():
    """The stop reached the call; its killer made it return its own raw
    error before the batch's next poll — that is a stop, not a failure."""
    from ghost_agent.core.agent import _gather_stoppable, _stoppable_call
    stop = TS.ToolStop()

    async def killed_quickly():
        while not TS.requested():
            await asyncio.sleep(0.01)
        return "Error: yt-dlp failed (exit 130): stopped by the user"

    async def raises_cancel():
        while not TS.requested():
            await asyncio.sleep(0.01)
        raise asyncio.CancelledError()               # the browser's own raise

    items = [(0, _stoppable_call(0, "knowledge_base", killed_quickly(), stop)),
             (1, _stoppable_call(1, "browser", raises_cancel(), stop))]
    asyncio.get_running_loop().call_later(0.3, stop.request)
    out = await _gather_stoppable(items, stop)
    assert all(isinstance(r, TS.Interrupted) and r.started for r in out), out


async def test_a_call_that_finished_before_the_stop_keeps_its_result():
    from ghost_agent.core.agent import _gather_stoppable, _stoppable_call
    stop = TS.ToolStop()

    async def fast():
        return "Error: 404"

    async def slow():
        await asyncio.sleep(_HANG)
    items = [(0, _stoppable_call(0, "a", fast(), stop)), (1, _stoppable_call(1, "b", slow(), stop))]
    asyncio.get_running_loop().call_later(0.3, stop.request)
    out = await _gather_stoppable(items, stop)
    assert out[0] == "Error: 404" and isinstance(out[1], TS.Interrupted)


async def test_background_work_spawned_inside_a_call_does_not_inherit_its_stop():
    from ghost_agent.utils.logging import spawn_bg
    stop, seen = TS.ToolStop(), {}

    async def bg():
        seen["stop"], seen["call"] = TS.current(), TS.call_key()
    toks = TS.bind_call(stop, 2)
    try:
        t = spawn_bg(bg(), name="t")
        assert TS.current() is stop                   # the call itself keeps it
    finally:
        TS.unbind_call(toks)
    await t
    assert seen == {"stop": None, "call": None}


async def test_the_turn_summary_never_lists_a_stopped_call_as_succeeded(monkeypatch, tmp_path):
    final, payloads, seen, _ = await _turn_with_an_interrupt(monkeypatch, tmp_path, failing=True)
    assert seen["interrupted"] == ["deep_research"]
    text = json.dumps(payloads[1])
    assert "MULTI-STEP OUTCOME" in text, text[-3000:]
    i = text.index("MULTI-STEP OUTCOME")
    summary = text[i:text.index("STOPPED BY THE USER", i)]
    assert "deep_research" not in summary and "web_search" in summary


# ── review r3 pins ──────────────────────────────────────────────────────────

async def test_a_stopped_turn_gets_no_optimisation_suggestion(monkeypatch, tmp_path):
    """Perfect-It reads a turn that ran a heavy tool as a finished task; a
    stopped call is not one ("Task completed successfully" + OPT appended)."""
    monkeypatch.setenv("GHOST_QUEUED_MESSAGES", "1")
    agent, ctx, _ = _loop_agent(monkeypatch, tmp_path)
    ctx.args.perfect_it = True
    reg, rid, seen = get_turn_registry(agent), "web-ti-pp", []

    async def _research(**kw):
        asyncio.get_running_loop().call_later(0.2, lambda: reg.interrupt_tools(rid))
        await asyncio.sleep(_HANG)
        return "research finished"
    agent.available_tools = {"deep_research": _research}
    replies = iter([_resp("", [_tc("c2", "deep_research", {"query": "x"})]), _resp("Stopped it.")]
                   + [_resp("OPT SUGGESTION")] * 5)

    async def _complete(*a, **k):
        seen.append(str(a[0] if a else k))
        return next(replies)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_complete)
    body = {"session_id": "s1", "messages": [{"role": "user", "content": "research the news"}]}
    final, _, _ = await agent.handle_chat(body, FakeBgTasks(), request_id=rid, requester_role="owner")
    assert "OPT SUGGESTION" not in final
    assert not any("Task completed successfully" in s for s in seen)


def test_the_stopped_fallback_head_is_its_own_and_a_known_head():
    from ghost_agent.core.reply_shape_check import FALLBACK_HEADS, refute_raw_tool_dump
    h = FALLBACK_HEADS["stopped"]
    assert "stopped" in h and "success" not in h.lower()
    # a sentence the model wrote with that head (no tool framing) is not a dump
    assert refute_raw_tool_dump(h + " I can run it again if you want.") == []
    assert refute_raw_tool_dump(h + "\n\n### Final Output:\nINTERRUPTED BY THE USER") != []


async def test_a_batch_duplicate_of_a_killed_call_says_the_same(monkeypatch, tmp_path):
    """A read collapsed at dispatch shares its original's result; once the
    original was confirmed stopped, the copy must not say 'may still finish'."""
    monkeypatch.setenv("GHOST_QUEUED_MESSAGES", "1")
    agent, ctx, _ = _loop_agent(monkeypatch, tmp_path)
    reg, rid = get_turn_registry(agent), "web-ti-dup"

    async def _search(**kw):
        asyncio.get_running_loop().call_later(0.2, lambda: reg.interrupt_tools(rid))
        try:
            await asyncio.sleep(_HANG)
        except asyncio.CancelledError:
            TS.note_killed("web_search")
            raise
        return "hits"
    agent.available_tools = {"web_search": _search}
    calls = [_tc("c1", "web_search", {"query": "news"}), _tc("c2", "web_search", {"query": "news"})]
    payloads, replies = [], iter([_resp("", calls), _resp("ok"), _resp("x"), _resp("x")])

    async def _complete(*a, **k):
        payloads.append(a[0] if a and isinstance(a[0], dict) else k)
        return next(replies)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_complete)
    body = {"session_id": "s1", "messages": [{"role": "user", "content": "search"}]}
    await agent.handle_chat(body, FakeBgTasks(), request_id=rid, requester_role="owner")
    text = str(payloads[1]["messages"])
    assert "Batch Dedup" not in text                  # (sanity: the log is not the prompt)
    assert text.count("web_search call while it was running, and it was stopped") == 2, text[-2500:]
    assert "may still finish" not in text


# ── review r4 pins: every reader gives a stopped call its own branch ────────

def test_the_fallback_head_for_a_stopped_call_is_stopped_never_success():
    from ghost_agent.core.agent import _fallback_head_key
    from ghost_agent.tools.outcome import OutcomeStatus, ToolOutcome
    for started in (True, False):
        o = TS.interrupted_outcome("deep_research", started=started, killed=False)
        assert _fallback_head_key(o, False) == "stopped"
        assert _fallback_head_key(str(o), False) == "stopped"      # a stored row
    assert _fallback_head_key(ToolOutcome("job 3 running", status=OutcomeStatus.UNRESOLVED), False) == "running"
    assert _fallback_head_key("Error: boom", True) == "failed"
    assert _fallback_head_key("all good", False) == "success"


def test_a_stopped_page_load_is_not_a_passing_run_of_the_page():
    from ghost_agent.core import code_claims as cc
    from ghost_agent.tools.outcome import ToolOutcome
    w = ToolOutcome("SUCCESS: Wrote 1200 chars to '/workspace/site/index.html'.",
                    call_args={"operation": "write", "path": "/workspace/site/index.html"})
    claim = "Done — the page works and I verified it loads correctly."
    for tool, started, args in (
            ("browser", True, {"operation": "navigate", "url": "http://localhost:8000/index.html"}),
            ("vision_analysis", False, {"image": "/workspace/site/index.html"})):
        for as_text in (False, True):
            s = TS.interrupted_outcome(tool, started=started, killed=False)
            s.call_args = args
            rows = [{"role": "tool", "name": "file_system", "content": w},
                    {"role": "tool", "name": tool,
                     "content": ToolOutcome(str(s), call_args=args) if as_text else s}]
            assert all(r["run_ok"] is None for r in cc.code_evidence(rows).values())
            assert cc.unbacked_claim(claim, rows)[0], (tool, as_text)   # flagged, as with no check


def test_a_stopped_retrieval_is_not_evidence():
    from ghost_agent.core.evidence_gate import assess_turn_evidence
    from ghost_agent.core.turn_state_check import refute_turn_state
    from ghost_agent.core.claim_binding import evidence_all_failed
    reply = ("The capital of the fictional country Zorbania is Mertu, founded in 1432 by King Aldo, and it "
             "has a population of about 2.3 million people today according to the national census records "
             "published last year by the statistics office of Zorbania.")
    for tool, args in (("web_search", {"query": "x"}),
                       ("browser", {"operation": "extract_text", "url": "https://a"})):
        o = TS.interrupted_outcome(tool, started=True, killed=False)
        o.call_args = args
        row = {"role": "tool", "name": tool, "content": o, "arguments": args}
        a = assess_turn_evidence([row])
        assert a.substantive == 0, (tool, a)
        assert refute_turn_state(request="what is the capital of Zorbania", reply=reply, tools_run=[row]), tool
    assert evidence_all_failed("[web_search] " + str(TS.interrupted_outcome("web_search", started=True, killed=False)))


async def test_stopped_searches_are_not_counted_as_searches_or_observations(monkeypatch, tmp_path):
    from ghost_agent.core.strikes import StrikeLedger
    yields, actions = [], []
    real_y, real_a = StrikeLedger.note_search_yield, StrikeLedger.note_action
    monkeypatch.setattr(StrikeLedger, "note_search_yield",
                        lambda self, f: (yields.append(f), real_y(self, f))[1])
    monkeypatch.setattr(StrikeLedger, "note_action",
                        lambda self, f, *a, **k: (actions.append(f), real_a(self, f, *a, **k))[1])
    _, _, seen, _ = await _turn_with_an_interrupt(monkeypatch, tmp_path, rid="web-ti-sy")
    assert seen["interrupted"] == ["deep_research"]
    assert "web_search" in yields and "deep_research" not in yields     # the finished one still counts
    assert "deep_research" not in actions


def test_a_stored_stopped_execute_is_not_a_run():
    from ghost_agent.core import code_claims as cc
    from ghost_agent.tools.outcome import ToolOutcome
    s = TS.interrupted_outcome("execute", started=True, killed=True)
    assert cc._exec_ok({"content": s}) is None
    assert cc._exec_ok({"content": ToolOutcome(str(s))}) is None        # status lost: OK + the text


async def _two_writes_then_stop(monkeypatch, tmp_path, *, first, request, extra=(), tools=None):
    """A batch whose first write triggers a stop: the second write is NOT RUN."""
    monkeypatch.setenv("GHOST_QUEUED_MESSAGES", "1")
    agent, ctx, _ = _loop_agent(monkeypatch, tmp_path)
    reg, rid = get_turn_registry(agent), "web-ti-ch"

    async def _fs(**kw):
        if kw.get("path") == "a.py":
            reg.interrupt_tools(rid)
            await asyncio.sleep(0.3)
            return first
        return f"wrote {kw.get('path')}"
    agent.available_tools = {"file_system": _fs, **(tools or {})}
    calls = [_tc("c1", "file_system", {"operation": "write", "path": "a.py", "content": "1"}),
             _tc("c2", "file_system", {"operation": "write", "path": "b.py", "content": "2"}), *extra]
    payloads, replies = [], iter([_resp("", calls), _resp("Done."), _resp("x"), _resp("x")])

    async def _complete(*a, **k):
        payloads.append(a[0] if a and isinstance(a[0], dict) else k)
        return next(replies)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_complete)
    body = {"session_id": "s1", "messages": [{"role": "user", "content": request}]}
    await agent.handle_chat(body, FakeBgTasks(), request_id=rid, requester_role="owner")
    return payloads


async def test_a_not_run_write_is_not_an_edit_and_a_stopped_run_is_not_a_verify(monkeypatch, tmp_path):
    import ghost_agent.core.agent as AG
    churns, real = [], AG.edit_churn_decision
    monkeypatch.setattr(AG, "edit_churn_decision", lambda c, t: (churns.append((c, t)), real(c, t))[1])

    async def _exec(**kw):
        return "EXIT CODE: 0"
    payloads = await _two_writes_then_stop(
        monkeypatch, tmp_path, first="wrote a.py", request="write two files",
        extra=[_tc("c3", "execute", {"command": "python a.py"})], tools={"execute": _exec})
    assert "NOT RUN" in str(payloads[1]["messages"])
    targets = [t for _, t in churns]
    assert targets and not any("b.py" in t for t in targets), targets   # b.py was never written
    assert churns[0][0]["counts"], churns                                # a NOT RUN execute verified nothing


async def test_a_not_run_write_does_not_spend_the_constraint_check(monkeypatch, tmp_path):
    payloads = await _two_writes_then_stop(
        monkeypatch, tmp_path, first="Error: disk full",
        request="write two files. Do not use any external libraries.")
    text = str(payloads[1]["messages"])
    assert "NOT RUN" in text and "SYSTEM ALERT (constraint check)" not in text


# ── review r5 pins ──────────────────────────────────────────────────────────

async def test_a_stopped_dream_is_not_complete_and_the_turn_goes_on(monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_QUEUED_MESSAGES", "1")
    agent, ctx, _ = _loop_agent(monkeypatch, tmp_path)
    reg, rid = get_turn_registry(agent), "web-ti-dream"

    async def _dream(**kw):
        asyncio.get_running_loop().call_later(0.2, lambda: reg.interrupt_tools(rid))
        await asyncio.sleep(_HANG)
        return "dreamt"
    agent.available_tools = {"dream_mode": _dream}
    replies = iter([_resp("", [_tc("c1", "dream_mode", {})]), _resp("Okay, I stopped the dream."),
                    _resp("x"), _resp("x")])

    async def _complete(*a, **k):
        return next(replies)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_complete)
    body = {"session_id": "s1", "messages": [{"role": "user", "content":
                                              "please go to sleep and consolidate your memories now"}]}
    final, _, _ = await agent.handle_chat(body, FakeBgTasks(), request_id=rid, requester_role="owner")
    assert "Dream cycle complete" not in final and "INTERRUPTED BY THE USER" not in final
    assert "stopped the dream" in final                          # the turn went on to the model


def test_a_not_run_write_is_not_a_written_file_in_task_state():
    from ghost_agent.core import task_state as TSt
    nr = TS.interrupted_outcome("file_system", started=False, killed=False)
    fail = {"name": "execute", "content": "EXIT CODE: 1\nFAILED test", "args": {"command": "pytest"}}
    wt = {"name": "file_system", "content": "SUCCESS: wrote tests/test_a.py",
          "args": {"operation": "write", "path": "tests/test_a.py", "content": "x"}}
    src = {"name": "file_system", "content": nr,
           "args": {"operation": "write", "path": "src/a.py", "content": "x"}}
    base = TSt.tests_only_note(TSt.edit_sides([fail, wt]))
    assert base and TSt.tests_only_note(TSt.edit_sides([fail, wt, src])) == base
    page = {"name": "file_system", "content": nr,
            "args": {"operation": "write", "path": "app.html", "content": "<script>x</script>"}}
    assert TSt.untested_pages([page]) == []


# ── review r6 pin ───────────────────────────────────────────────────────────

async def test_stopped_runs_never_feed_the_futility_breaker(monkeypatch, tmp_path):
    """Seven batches of write a.py + execute, each stopped during the write:
    execute never ran, so no run is counted and no futility alert fires."""
    monkeypatch.setenv("GHOST_QUEUED_MESSAGES", "1")
    agent, ctx, _ = _loop_agent(monkeypatch, tmp_path)
    reg, rid, ran = get_turn_registry(agent), "web-ti-fut", []

    async def _fs(**kw):
        reg.interrupt_tools(rid)
        await asyncio.sleep(0.05)
        return "wrote a.py"

    async def _exec(**kw):
        ran.append(1)
        return "EXIT CODE: 0\nsame"
    agent.available_tools = {"file_system": _fs, "execute": _exec}

    def batch(n):
        return _resp("", [_tc(f"w{n}", "file_system", {"operation": "write", "path": "a.py", "content": str(n)}),
                          _tc(f"e{n}", "execute", {"command": "python a.py"})])
    payloads = []
    replies = iter([batch(n) for n in range(1, 8)] + [_resp("Done."), _resp("x"), _resp("x")])

    async def _complete(*a, **k):
        payloads.append(a[0] if a and isinstance(a[0], dict) else k)
        return next(replies)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_complete)
    body = {"session_id": "s1", "messages": [{"role": "user", "content": "write a.py and run it until it works"}]}
    await agent.handle_chat(body, FakeBgTasks(), request_id=rid, requester_role="owner")
    assert ran == []
    assert all(r.get("runs", 0) == 0 for r in (getattr(ctx, "_script_iter", None) or {}).values())
    assert not any("futility breaker" in str(p.get("messages")) for p in payloads)


# ── review r7 pins: per-request state ───────────────────────────────────────

async def test_a_stopped_read_does_not_mark_the_request_as_having_read_outside_content(monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_QUEUED_MESSAGES", "1")
    monkeypatch.setattr(TS, "STOP_GRACE_S", 0.5)
    agent, ctx, _ = _loop_agent(monkeypatch, tmp_path)
    reg, rid, got = get_turn_registry(agent), "web-ti-taint", {}

    async def _hang(**kw):
        asyncio.get_running_loop().call_later(0.2, lambda: reg.interrupt_tools(rid))
        await asyncio.sleep(_HANG)
        return "results..."

    async def _tasks(**kw):
        from ghost_agent.utils.provenance import untrusted_seen
        got["seen"] = list(untrusted_seen())
        return "SUCCESS: created"
    agent.available_tools = {"web_search": _hang, "manage_tasks": _tasks}
    replies = iter([_resp("", [_tc("a", "web_search", {"query": "q"})]),
                    _resp("", [_tc("b", "manage_tasks", {"action": "list"})]),
                    _resp("Done.")] + [_resp("x")] * 5)

    async def _complete(*a, **k):
        return next(replies)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_complete)
    body = {"session_id": "s1", "messages": [{"role": "user", "content": "search q and then list my tasks"}]}
    await agent.handle_chat(body, FakeBgTasks(), request_id=rid, requester_role="owner")
    assert got["seen"] == []


async def test_an_all_stopped_batch_is_not_a_clean_success(monkeypatch, tmp_path):
    from ghost_agent.core.strikes import StrikeLedger
    monkeypatch.setenv("GHOST_QUEUED_MESSAGES", "1")
    monkeypatch.setattr(TS, "STOP_GRACE_S", 0.5)
    cleans = []
    real = StrikeLedger.note_clean_success
    monkeypatch.setattr(StrikeLedger, "note_clean_success",
                        lambda self: (cleans.append(1), real(self))[1])
    _, _, seen, _ = await _turn_with_an_interrupt(monkeypatch, tmp_path, rid="web-ti-cs")
    assert seen["interrupted"] == ["deep_research"]
    n_mixed = len(cleans)                    # web_search finished: a real success in that batch

    agent, ctx, _ = _loop_agent(monkeypatch, tmp_path)
    reg, rid = get_turn_registry(agent), "web-ti-cs2"

    async def _hang(**kw):
        asyncio.get_running_loop().call_later(0.2, lambda: reg.interrupt_tools(rid))
        await asyncio.sleep(_HANG)
    agent.available_tools = {"deep_research": _hang}
    replies = iter([_resp("", [_tc("c", "deep_research", {"query": "x"})]), _resp("ok"), _resp("x")])

    async def _complete(*a, **k):
        return next(replies)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_complete)
    cleans.clear()
    await agent.handle_chat({"session_id": "s1", "messages": [{"role": "user", "content": "research x"}]},
                            FakeBgTasks(), request_id=rid, requester_role="owner")
    assert n_mixed >= 1 and cleans == []


def test_every_runner_command_comes_from_the_shared_builder():
    """s5-r9: the stop kills the runner by `_RUNNER_ARGV_HEAD`; a command
    spelled inline (the commit-milestone retry was) drifts from it unseen."""
    import ast
    import inspect
    from ghost_agent.tools import browser
    tree = ast.parse(inspect.getsource(browser))
    builds = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
              and getattr(n.func, "id", None) == "runner_cmd"
              and getattr((n.args or [None])[0], "id", None) == "_BROWSER_RUNNER_FILENAME"]
    assert len(builds) == 3
    for n in ast.walk(tree):   # no command line spelled inline
        if isinstance(n, ast.JoinedStr):
            parts = n.values
            for a, b in zip(parts, parts[1:]):
                assert not (isinstance(a, ast.FormattedValue)
                            and getattr(a.value, "id", None) == "_BROWSER_RUNNER_FILENAME"
                            and isinstance(b, ast.Constant) and b.value.startswith(" ")), ast.unparse(n)
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            assert ".browser_runner.py " not in n.value


async def test_a_raising_hit_file_call_still_carries_the_note(monkeypatch, tmp_path):
    """s5-r10: the stop landed on a lone write that then raised — the
    exception slot skipped the note, so the model read the error only."""
    monkeypatch.setenv("GHOST_QUEUED_MESSAGES", "1")
    agent, ctx, _ = _loop_agent(monkeypatch, tmp_path)
    reg = get_turn_registry(agent)

    async def _fs(**kw):
        assert reg.interrupt_tools("web-ti10") == ["file_system"]
        raise OSError("disk full")
    agent.available_tools = {"file_system": _fs}
    payloads = []
    replies = iter([_resp("", [_tc("c1", "file_system", {"operation": "write", "path": "a.txt", "content": "1"})]),
                    _resp("Done."), _resp("x")])

    async def _complete(*a, **k):
        payloads.append(a[0] if a and isinstance(a[0], dict) else k)
        return next(replies)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_complete)
    body = {"session_id": "s1", "messages": [{"role": "user", "content": "write a file"}]}
    await agent.handle_chat(body, FakeBgTasks(), request_id="web-ti10", requester_role="owner")
    text = str(payloads[1]["messages"])
    assert "disk full" in text and "the result above is real" in text


async def test_a_dedup_copy_of_a_hit_read_carries_the_note(monkeypatch, tmp_path):
    """s5-r10 NIT: a batch-dedup copy reads its original's hit, like an
    Interrupted copy reads its original's kill."""
    monkeypatch.setenv("GHOST_QUEUED_MESSAGES", "1")
    agent, ctx, _ = _loop_agent(monkeypatch, tmp_path)
    reg = get_turn_registry(agent)
    calls = []

    async def _fs(**kw):
        calls.append(kw.get("path"))
        assert reg.interrupt_tools("web-ti11") == ["file_system"]
        return f"content of {kw.get('path')}"
    agent.available_tools = {"file_system": _fs}
    payloads = []
    replies = iter([_resp("", [_tc("c1", "file_system", {"operation": "read", "path": "a.txt"}),
                               _tc("c2", "file_system", {"operation": "read", "path": "a.txt"})]),
                    _resp("Done."), _resp("x")])

    async def _complete(*a, **k):
        payloads.append(a[0] if a and isinstance(a[0], dict) else k)
        return next(replies)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_complete)
    body = {"session_id": "s1", "messages": [{"role": "user", "content": "read it twice"}]}
    await agent.handle_chat(body, FakeBgTasks(), request_id="web-ti11", requester_role="owner")
    assert calls == ["a.txt"]                                  # executed once, shared
    assert str(payloads[1]["messages"]).count("the result above is real") == 2


async def test_the_note_is_in_the_tool_message_only(monkeypatch, tmp_path):
    """s5-r12: a note put in the RESULT reached every copy of it — the
    failure context (r11), then the multi-step outcome preview (r12). It is
    added where the tool message is built, so no other reader sees it."""
    monkeypatch.setenv("GHOST_QUEUED_MESSAGES", "1")
    agent, ctx, _ = _loop_agent(monkeypatch, tmp_path)
    reg = get_turn_registry(agent)

    async def _fs(**kw):
        if kw.get("path") == "b.txt":
            reg.interrupt_tools("web-ti13")
            raise OSError("disk full")
        return f"wrote {kw.get('path')}"
    agent.available_tools = {"file_system": _fs}
    payloads = []
    replies = iter([_resp("", [_tc("c1", "file_system", {"operation": "write", "path": "a.txt", "content": "1"}),
                               _tc("c2", "file_system", {"operation": "write", "path": "b.txt", "content": "2"})]),
                    _resp("Done."), _resp("x")])

    async def _complete(*a, **k):
        payloads.append(a[0] if a and isinstance(a[0], dict) else k)
        return next(replies)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_complete)
    body = {"session_id": "s1", "messages": [{"role": "user", "content": "write two files"}]}
    await agent.handle_chat(body, FakeBgTasks(), request_id="web-ti13", requester_role="owner")
    msgs = payloads[1]["messages"]
    assert str(msgs).count("The user asked to stop") == 1
    text = str(msgs)
    assert "disk full" in text and "MULTI-STEP" not in text.split("The user asked to stop")[1][:400]


async def test_a_hit_call_failure_notes_only_the_tool_message(monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_QUEUED_MESSAGES", "1")
    agent, ctx, _ = _loop_agent(monkeypatch, tmp_path)
    reg = get_turn_registry(agent)

    async def _fs(**kw):
        reg.interrupt_tools("web-ti12")
        raise SystemExit(3)                    # s5-r11: a BaseException too
    agent.available_tools = {"file_system": _fs}
    payloads = []
    replies = iter([_resp("", [_tc("c1", "file_system", {"operation": "write", "path": "a.txt", "content": "1"})]),
                    _resp("Done."), _resp("x")])

    async def _complete(*a, **k):
        payloads.append(a[0] if a and isinstance(a[0], dict) else k)
        return next(replies)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_complete)
    body = {"session_id": "s1", "messages": [{"role": "user", "content": "write a file"}]}
    await agent.handle_chat(body, FakeBgTasks(), request_id="web-ti12", requester_role="owner")
    text = str(payloads[1]["messages"])
    assert text.count("the result above is real") == 1


async def test_the_note_is_in_the_models_copy_not_the_turn_record(monkeypatch, tmp_path):
    """s5-r13: the turn's record (`tools_run_this_turn`) is what the empty-
    reply fallback quotes to the USER, the verifier's evidence and the
    trajectory row — the note meant for the model must not be in it."""
    monkeypatch.setenv("GHOST_QUEUED_MESSAGES", "1")
    agent, ctx, _ = _loop_agent(monkeypatch, tmp_path)
    reg = get_turn_registry(agent)

    async def _fs(**kw):
        reg.interrupt_tools("web-ti14")
        return f"wrote {kw.get('path')}"
    agent.available_tools = {"file_system": _fs}
    payloads, records = [], []
    replies = iter([_resp("", [_tc("c1", "file_system", {"operation": "write", "path": "a.txt", "content": "1"})]),
                    _resp("Done."), _resp("x")])

    async def _complete(*a, **k):
        payloads.append(a[0] if a and isinstance(a[0], dict) else k)
        return next(replies)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_complete)
    real_fin = agent._finalize_and_return

    async def _spy(fs):
        records.append([str(t.get("content")) for t in (fs.tools_run_this_turn or [])])
        return await real_fin(fs)
    monkeypatch.setattr(agent, "_finalize_and_return", _spy)
    body = {"session_id": "s1", "messages": [{"role": "user", "content": "write a file"}]}
    await agent.handle_chat(body, FakeBgTasks(), request_id="web-ti14", requester_role="owner")
    assert "The user asked to stop" in str(payloads[1]["messages"])        # the model reads it
    assert records and any("wrote a.txt" in r for r in records[0])
    assert not any("The user asked to stop" in r for r in records[0])     # the record does not


async def _overflow_turn(monkeypatch, tmp_path, rid, script, calls, hit_paths):
    import httpx
    monkeypatch.setenv("GHOST_QUEUED_MESSAGES", "1")
    agent, ctx, _ = _loop_agent(monkeypatch, tmp_path)
    reg = get_turn_registry(agent)

    async def _fs(**kw):
        if kw.get("path") in hit_paths:
            reg.interrupt_tools(rid)
        return f"SUCCESS: wrote {kw.get('path')}"
    agent.available_tools = {"file_system": _fs}
    payloads, script, calls = [], list(script), list(calls)

    async def _complete(*a, **k):
        p = a[0] if a and isinstance(a[0], dict) else k
        if k.get("task_label") == "compaction":
            return _resp("SUMMARY")
        payloads.append([dict(m) for m in p["messages"]])
        r = script.pop(0) if script else "done."
        if r == "tool":
            cid, path = calls.pop(0)
            return _resp("", [_tc(cid, "file_system", {"operation": "write", "path": path, "content": "1"})])
        if r == "overflow":
            req = httpx.Request("POST", "http://x")
            raise httpx.HTTPStatusError("400", request=req,
                                        response=httpx.Response(400, text="context length exceeded", request=req))
        return _resp(r)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_complete)
    body = {"session_id": "s1", "messages": [{"role": "user", "content": "write files"}]}
    await agent.handle_chat(body, FakeBgTasks(), request_id=rid, requester_role="owner")
    return [" ".join(str(m["content"]) for m in p) for p in payloads]


async def test_a_second_overflow_keeps_the_stop_note(monkeypatch, tmp_path):
    """s5-r15: after the first recovery `messages` held no tool row, so a
    lookup there found nothing and the second rebuild dropped the note."""
    texts = await _overflow_turn(monkeypatch, tmp_path, "web-ti16", ["tool", "overflow", "", "overflow", "done."],
                                 [("c1", "a.txt")], {"a.txt"})
    assert "EMERGENCY TRUNCATION" in texts[4] and "The user asked to stop" in texts[4]


async def test_an_empty_call_id_never_notes_another_call(monkeypatch, tmp_path):
    """s5-r15: the native stream can leave a call id "" — matched by id, a
    later call that was never hit read the stop note."""
    texts = await _overflow_turn(monkeypatch, tmp_path, "web-ti17", ["tool", "tool", "overflow", "done."],
                                 [("", "a.txt"), ("", "b.txt")], {"a.txt"})
    rec = next(t for t in texts if "EMERGENCY TRUNCATION" in t)
    assert "wrote b.txt" in rec and "The user asked to stop" not in rec


async def test_the_overflow_recovery_keeps_the_stop_note(monkeypatch, tmp_path):
    """s5-r14: the overflow recovery rebuilds the prompt from the turn record,
    which is plain — the model never read that the user asked to stop."""
    import httpx
    monkeypatch.setenv("GHOST_QUEUED_MESSAGES", "1")
    agent, ctx, _ = _loop_agent(monkeypatch, tmp_path)
    reg = get_turn_registry(agent)

    async def _fs(**kw):
        reg.interrupt_tools("web-ti15")
        return f"SUCCESS: wrote {kw.get('path')}"
    agent.available_tools = {"file_system": _fs}
    payloads, script = [], ["tool", "overflow", "done."]

    async def _complete(*a, **k):
        p = a[0] if a and isinstance(a[0], dict) else k
        if k.get("task_label") == "compaction":
            return _resp("SUMMARY")
        payloads.append([dict(m) for m in p["messages"]])
        r = script.pop(0) if script else "done."
        if r == "tool":
            return _resp("", [_tc("c1", "file_system", {"operation": "write", "path": "a.txt", "content": "1"})])
        if r == "overflow":
            req = httpx.Request("POST", "http://x")
            raise httpx.HTTPStatusError("400", request=req,
                                        response=httpx.Response(400, text="context length exceeded", request=req))
        return _resp(r)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=_complete)
    body = {"session_id": "s1", "messages": [{"role": "user", "content": "write a file"}]}
    await agent.handle_chat(body, FakeBgTasks(), request_id="web-ti15", requester_role="owner")
    rec = " ".join(str(m["content"]) for m in payloads[2])
    assert "EMERGENCY TRUNCATION" in rec and "wrote a.txt" in rec
    assert "The user asked to stop" in rec
