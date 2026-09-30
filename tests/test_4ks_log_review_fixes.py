"""§4KS — fixes from reading one 15 h operator log (2026-09-30).

Each block names the live defect and the WORLD in which the pin fails:

* A  a service bound to the container's loopback was reported "published to
     the host" → the pin fails if a 127.0.0.1 listener on a published port
     gets the remote-access hint;
* B  the WEB-EXEC probe certified a 404 → fails if a 4xx page reads clean;
* C  "verdict deferred" was printed for turns with no verdict task → fails
     if a member turn / an empty in-loop verdict is announced as deferred;
* E  "just the number … on that line" refuted the whole reply → fails if a
     step-by-step reply ending in a bare number is refuted;
* F  a defective generated challenge forfeited the idle slot, a duplicate
     retry ran at the same temperature, escaped-newline data shipped;
* G  a lesson whose verification failed was saved anyway;
* H  frequency was bumped with no new evidence;
* K  the Turn Outcome line printed another request's confidence;
* L  a lone "<" vanished from displayed thinking;
* M  every bench request was tagged BE, every Slack request SL;
* N  streamed turns were stored with duration 0.0;
* P  a `site:` filter dropped without a word, search pages read as sources;
* Q  the text a channel member's turn is given.
Three designs were built and removed after review, and are pinned as
absent: a request view for "yes proceed", a service-root fallback in the
WEB-EXEC probe, and two replacements for the "how to" search retry.
"""
import asyncio
import ast
import json
import os
import re
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import ghost_agent.core.agent as agent_mod
from ghost_agent.core.agent import GhostAgent

SRC = Path(__file__).resolve().parents[1] / "src" / "ghost_agent"


# ══════════════════════════════════════════════════════════════════════
# A — loopback bind on a published port; restart can move the port
# ══════════════════════════════════════════════════════════════════════
from ghost_agent.sandbox.services import (
    ServiceSupervisor, loopback_only, loopback_bind_warning,
    unpublished_port_warning, REMOTE_SERVE_SCRIPT,
)
from ghost_agent.tools.sandbox_services import tool_manage_services


class _Sandbox:
    """A sandbox double: `ss` answers with whatever address the test says
    the app bound; everything else is the happy path."""

    def __init__(self, tmp_path, bind="0.0.0.0", host_netns=False,
                 published=None):
        self.host_workspace = Path(tmp_path)
        self.calls = []
        self.bind = bind
        self._host_netns = host_netns
        self._published = published

    def binds_host_netns(self):
        return self._host_netns

    def published_service_ports(self):
        if self._published is None:
            raise RuntimeError("unknown")
        return set(self._published)

    def execute(self, cmd, timeout=600, **kw):
        self.calls.append(cmd)
        if "nohup" in cmd:
            m = re.search(r'\.services/([A-Za-z0-9_-]+)\.cmd\.sh', cmd)
            svc = self.host_workspace / ".services"
            svc.mkdir(parents=True, exist_ok=True)
            (svc / f"{m.group(1)}.pid").write_text("4242")
            return ("4242\n", 0)
        m = re.search(r"ss -H -ltn 'sport = :(\d+)'", cmd)
        if m:
            if self.bind is None:
                return ("", 0)
            port = m.group(1)
            local = (f"[{self.bind}]:{port}" if ":" in self.bind
                     else f"{self.bind}:{port}")
            return (f"LISTEN 0      128    {local} 0.0.0.0:*\n", 0)
        return ("", 0)


@pytest.fixture
def _no_sleep(monkeypatch):
    import ghost_agent.sandbox.services as svc
    monkeypatch.setattr(svc.time, "sleep", lambda s: None)


@pytest.mark.parametrize("addrs,expected", [
    (["127.0.0.1"], True),
    (["127.0.0.53"], True),
    (["::1"], True),
    (["[::1]"], True),
    (["127.0.0.1", "::1"], True),
    (["::ffff:127.0.0.1"], True),          # a dual-stack runtime's loopback
    (["127.0.0.1%lo"], True),              # `ss` may append the scope
    (["::ffff:10.0.0.5"], False),
    (["0.0.0.0"], False),
    (["*"], False),
    (["::"], False),
    (["127.0.0.1", "0.0.0.0"], False),   # ONE wildcard listener is enough
    (["172.17.0.2"], False),
    (["1270.0.0.1"], False),
    ([], False),                          # nothing known → never warn
    (None, False),
])
def test_loopback_only_table(addrs, expected):
    assert loopback_only(addrs) is expected


@pytest.mark.parametrize("ss_out,code,expected", [
    ("LISTEN 0 128 127.0.0.1:8100 0.0.0.0:*\n", 0, ["127.0.0.1"]),
    ("LISTEN 0 128 0.0.0.0:8100 0.0.0.0:*\n", 0, ["0.0.0.0"]),
    ("LISTEN 0 128 *:8100 *:*\n", 0, ["*"]),
    ("LISTEN 0 128 [::]:8100 [::]:*\n", 0, ["::"]),
    ("LISTEN 0 128 [::1]:8100 [::]:*\n", 0, ["::1"]),
    ("LISTEN 0 128 127.0.0.1:8100 0.0.0.0:*\n"
     "LISTEN 0 128 [::]:8100 [::]:*\n", 0, ["127.0.0.1", "::"]),
    # a row for ANOTHER port is not this service's address
    ("LISTEN 0 128 127.0.0.1:9050 0.0.0.0:*\n", 0, None),
    ("", 0, None),
    ("LISTEN 0 128 127.0.0.1:8100 0.0.0.0:*\n", 1, None),   # probe failed
    ("garbage\n", 0, None),
])
def test_listen_addrs_reads_the_local_address(tmp_path, ss_out, code, expected):
    sb = _Sandbox(tmp_path)
    sup = ServiceSupervisor(sb)
    sup._exec = lambda cmd, timeout=30: (ss_out, code)
    assert sup._listen_addrs(8100) == expected


def test_start_on_published_port_bound_to_loopback_is_not_called_reachable(
        tmp_path, _no_sleep):
    """The live turn: Flask `app.run(port=PORT)` binds 127.0.0.1."""
    sup = ServiceSupervisor(_Sandbox(tmp_path, bind="127.0.0.1"))
    out = str(sup.start("war-sim", "python3 sim.py", port=8100))
    assert "RUNNING" in out
    assert "NOT reachable from the host" in out
    assert "127.0.0.1:8100" in out and "0.0.0.0" in out
    # the hint that called it reachable must be gone
    assert "Remote access: published to the host" not in out
    assert REMOTE_SERVE_SCRIPT not in out


@pytest.mark.parametrize("bind", ["0.0.0.0", "*", "::"])
def test_start_on_published_port_bound_to_all_interfaces_keeps_the_hint(
        tmp_path, _no_sleep, bind):
    sup = ServiceSupervisor(_Sandbox(tmp_path, bind=bind))
    out = str(sup.start("web", "python3 app.py", port=8100))
    assert "Remote access: published to the host" in out
    assert "NOT reachable from the host" not in out


def test_silent_address_probe_never_invents_the_warning(tmp_path, _no_sleep):
    sup = ServiceSupervisor(_Sandbox(tmp_path, bind=None))
    out = str(sup.start("web", "python3 app.py", port=8100))
    assert "Remote access: published to the host" in out
    assert "NOT reachable from the host" not in out


def test_host_netns_loopback_is_reachable_and_not_warned(tmp_path, _no_sleep):
    """In host network mode the container's loopback IS the host's."""
    sup = ServiceSupervisor(
        _Sandbox(tmp_path, bind="127.0.0.1", host_netns=True))
    out = str(sup.start("web", "python3 app.py", port=8100))
    assert "NOT reachable from the host" not in out


def test_unpublished_port_keeps_its_own_warning(tmp_path, _no_sleep):
    sup = ServiceSupervisor(_Sandbox(tmp_path, bind="127.0.0.1"))
    out = str(sup.start("web", "python3 app.py", port=5000))
    assert "is not published by the sandbox" in out
    assert "the app bound port" not in out          # the loopback warning's words


def test_second_instance_that_published_nothing_is_not_called_loopback(
        tmp_path, _no_sleep):
    """`published_ports=set()` (a second agent): the port is not published,
    so the loopback rule — which is about PUBLISHED ports — stays out."""
    sup = ServiceSupervisor(
        _Sandbox(tmp_path, bind="127.0.0.1", published=[]))
    assert sup._host_unreachable_bind(8100) is False


def test_status_names_the_loopback_bind(tmp_path, _no_sleep):
    sb = _Sandbox(tmp_path, bind="127.0.0.1")
    sup = ServiceSupervisor(sb)
    sup.start("war-sim", "python3 sim.py", port=8100)
    line = [l for l in sup.status().splitlines() if "war-sim" in l][0]
    assert "NOT reachable from the host" in line
    assert "remote:" not in line
    sb.bind = "0.0.0.0"
    line = [l for l in sup.status().splitlines() if "war-sim" in l][0]
    assert "remote:" in line and "NOT reachable" not in line


@pytest.mark.parametrize("text", [
    loopback_bind_warning(8100),
    unpublished_port_warning(5000, published_ports={8100}, command="python3 app.py",
                             workdir="/workspace"),
    unpublished_port_warning(5000, published_ports=set()),
])
def test_a_service_warning_is_not_an_error_line(text):
    """Found live (2026-09-30): these go into a SUCCESSFUL tool result, and
    `strikes.error_line` reads every line for failure words — "cannot" made
    the start report of an unpublished-port service an error line for the
    strike counter and the evidence digest."""
    from ghost_agent.core.strikes import error_line
    assert error_line(text) == "", error_line(text)


def test_loopback_warning_tells_the_model_not_to_announce_it():
    w = loopback_bind_warning(8100)
    assert "HOST" in w and "8100" in w and "restart" in w
    assert "loopback only" in w          # true for a `::1` listener too
    assert "Do NOT tell the user it is live" in w


def test_restart_refuses_a_different_port_before_stopping(tmp_path, _no_sleep):
    """Live: told its port was unpublished, the model called restart to
    move the service and got the same port back, silently."""
    sb = _Sandbox(tmp_path)
    sup = ServiceSupervisor(sb)
    sup.start("web", "python3 -m http.server 5000", port=5000)
    before = dict(sup._load()["web"])
    n_calls = len(sb.calls)
    out = str(sup.restart("web", port=8101))
    assert out.startswith("Error:") and "Nothing was stopped" in out
    assert "action='stop' name='web'" in out and "action='start' name='web'" in out
    assert not any("kill" in c or "nohup" in c for c in sb.calls[n_calls:])
    assert sup._load()["web"] == before


def test_the_refusal_carries_what_start_needs(tmp_path, _no_sleep):
    """R5 review: "stop, then start with its command" — but `stop` deletes
    the row holding the command and the workdir, `status` truncates the
    command at 80 characters, and a port named INSIDE the command is not
    rewritten by a start whose requested port is granted."""
    sup = ServiceSupervisor(_Sandbox(tmp_path))
    (tmp_path / "projects" / "aaa111" / "dist").mkdir(parents=True)
    cmd = ("gunicorn --workers 2 --timeout 120 --access-logfile - --error-logfile - "
           "--bind 0.0.0.0:8899 wsgi:application")
    assert len(cmd) > 80
    started = str(sup.start("web", cmd, port=8899, workdir="projects/aaa111/dist",
                            project_id="aaa111"))
    assert not started.startswith("Error:"), started
    entry = sup._load()["aaa111:web"]
    out = str(sup.restart("web", project_id="aaa111", port=8100))
    assert f"Stored command: {cmd}" in out                       # whole, not truncated
    assert f"workdir='{entry['workdir']}'" in out
    assert "action='stop' name='aaa111:web'" in out              # the key it resolved,
    assert "action='start' name='aaa111:web'" in out             # on BOTH calls
    assert "start does not rewrite" in out
    assert "change that to the new port" in out and "$PORT" in out


@pytest.mark.parametrize("port", [8080, 80, 70000, "abc", False, True, "8101", 0, "none",
                                  8100.5, [8100], float("inf")])
def test_restart_with_any_other_port_stops_nothing(tmp_path, _no_sleep, port):
    sb = _Sandbox(tmp_path)
    sup = ServiceSupervisor(sb)
    sup.start("web", "python3 app.py", port=8100)
    n_calls = len(sb.calls)
    out = str(sup.restart("web", port=port))
    assert out.startswith("Error:") and "Nothing was stopped" in out
    assert not any("kill" in c or "nohup" in c for c in sb.calls[n_calls:])


@pytest.mark.parametrize("port", [None, 8100, "8100", " 8100 ", 8100.0, "8100.0", ""])
def test_restart_with_the_stored_port_is_a_plain_restart(tmp_path, _no_sleep, port):
    sb = _Sandbox(tmp_path)
    sup = ServiceSupervisor(sb)
    sup.start("web", "python3 -m http.server 8100", port=8100)
    n_calls = len(sb.calls)
    out = str(sup.restart("web", port=port))
    assert not out.startswith("Error:"), out
    assert any("nohup" in c for c in sb.calls[n_calls:])          # it relaunched
    entry = sup._load()["web"]
    assert entry["port"] == 8100 and entry["command"] == "python3 -m http.server 8100"


@pytest.mark.parametrize("port", [0, "0", "none", "no", "off", 0.0, " None "])
def test_a_portless_service_restarts_under_its_own_spelling(tmp_path, _no_sleep, port):
    sup = ServiceSupervisor(_Sandbox(tmp_path))
    sup.start("worker", "python3 worker.py", port=0)
    assert not str(sup.restart("worker", port=port)).startswith("Error:")
    assert str(sup.restart("worker", port=8100)).startswith("Error:")


async def test_tool_restart_to_another_port_is_a_true_rejection(tmp_path, _no_sleep):
    from ghost_agent.tools.outcome import OutcomeStatus
    sb = _Sandbox(tmp_path)
    await tool_manage_services(action="start", name="web",
                               command="python3 app.py", port=8100,
                               sandbox_manager=sb)
    n_calls = len(sb.calls)
    res = await tool_manage_services(action="restart", name="web", port=8102,
                                     sandbox_manager=sb)
    assert res.status == OutcomeStatus.REJECTED
    assert not any("kill" in c or "nohup" in c for c in sb.calls[n_calls:])
    from ghost_agent.sandbox.services import get_service_supervisor
    assert get_service_supervisor(sb)._load()["web"]["port"] == 8100


def test_a_refused_relaunch_keeps_the_registration(tmp_path, _no_sleep):
    """Unchanged behaviour, pinned because this work passed through it: a
    relaunch `start` refuses is returned as `start` said it, and the row
    survives. (A "the service is DOWN" declaration was built here and
    removed — what the stop did cannot be known under a probe fault.)"""
    sup = ServiceSupervisor(_Sandbox(tmp_path))
    sup.start("web", "python3 app.py", port=8100)
    sup.start = lambda *a, **k: "Error: workdir '/workspace/app' does not exist. Nothing was launched."
    out = sup.restart("web")
    assert str(out).startswith("Error: workdir") and "was preserved" in str(out)
    assert "DOWN" not in str(out)
    assert "web" in sup._load()


def test_restart_keeps_the_stored_workdir(tmp_path, _no_sleep):
    sup = ServiceSupervisor(_Sandbox(tmp_path))
    (tmp_path / "app").mkdir()
    sup.start("web", "python3 app.py", port=8100, workdir="app")
    wd = sup._load()["web"]["workdir"]
    assert wd
    sup.restart("web")
    assert sup._load()["web"]["workdir"] == wd


def test_unpublished_warning_names_the_calls_that_move_it():
    w = unpublished_port_warning(5000, published_ports={8100, 8101})
    assert "action='stop', then action='start'" in w
    assert "same command and workdir" in w
    assert "if the command itself names 5000, change it there too" in w
    assert w.rstrip().endswith("A restart KEEPS port 5000.")
    assert "Stored command" not in w                    # nothing stored was given
    full = unpublished_port_warning(5000, published_ports={8100}, command="python3 app.py",
                                    workdir="/workspace/app")
    assert full.endswith("\nStored command: python3 app.py\nStored workdir: /workspace/app")
    assert "Stored workdir" not in unpublished_port_warning(
        5000, published_ports={8100}, command="python3 app.py")


def test_the_start_report_and_status_carry_what_the_move_needs(tmp_path, _no_sleep):
    """R6 review: the warning and the status line said "stop, then start"
    without the command or workdir — `stop` deletes the row that holds them
    and `status` cut the command at 80 characters. (A pointer to a "restart
    port=… stops nothing" call was tried and removed: R7 — it resolved to a
    different service under a bound project, and did restart one whose
    stored port was the port offered.)"""
    sb = _Sandbox(tmp_path, published={8100, 8101})
    sup = ServiceSupervisor(sb)
    (tmp_path / "projects" / "aaa111" / "backend").mkdir(parents=True)
    cmd = ("gunicorn --workers 2 --timeout 120 --access-logfile - --error-logfile - "
           "--bind 0.0.0.0:5000 wsgi:application")
    assert len(cmd) > 80
    report = str(sup.start("api", cmd, port=5000, workdir="projects/aaa111/backend",
                           project_id="aaa111"))
    wd = sup._load()["aaa111:api"]["workdir"]
    assert f"Stored command: {cmd}" in report and f"Stored workdir: {wd}" in report
    assert "stops nothing" not in report
    sup._entry_alive = lambda e: True
    sup._port_listening = lambda p: True
    line = [l for l in str(sup.status()).splitlines() if "api" in l][0]
    assert "port not published" in line and "action='stop', then action='start'" in line
    assert "a restart keeps the port" in line and "(restart on one of" not in line
    assert "on one of 8100-8104" in line
    # R8 review: the command shown names the port — followed literally it
    # binds 5000 again under a row that says 8100
    assert "if the command itself names 5000, change it there too" in line and "$PORT" in line
    assert f"cmd: {cmd}" in line and f"workdir: {wd}" in line
    assert "stops nothing" not in line
    # every other row keeps the compact line: published, loopback-bound, dead
    ok_cmd = cmd.replace("5000", "8100")
    sup.start("ok", ok_cmd, port=8100)
    ok_line = [l for l in str(sup.status()).splitlines() if l.startswith("- ok")][0]
    assert ok_cmd not in ok_line and "workdir:" not in ok_line
    sb.bind = "127.0.0.1"
    lo_line = [l for l in str(sup.status()).splitlines() if l.startswith("- ok")][0]
    assert "NOT reachable from the host" in lo_line
    assert ok_cmd not in lo_line and "workdir:" not in lo_line
    sup._entry_state = lambda e: False
    for dead in [l for l in str(sup.status()).splitlines() if l.startswith("- ")]:
        assert "DEAD" in dead and cmd not in dead and "workdir:" not in dead


def test_the_report_prints_the_command_as_stored(tmp_path, _no_sleep):
    """R8 review: the report labelled the command AS TYPED "Stored command"
    — the row holds it without its leading `cd <workdir> &&`."""
    sup = ServiceSupervisor(_Sandbox(tmp_path, published={8100}))
    (tmp_path / "app").mkdir()
    report = str(sup.start("web", "cd app && python3 app.py  ", port=5000, workdir="app"))
    stored = sup._load()["web"]["command"]
    assert stored == "python3 app.py"
    assert f"Stored command: {stored}\n" in report
    assert "Stored command: cd app" not in report


def test_a_failure_start_declared_itself_survives_the_preserved_note(tmp_path, _no_sleep):
    """A relaunch that started and then failed is a DECLARED failure; the
    "registration was preserved" note must not flatten it into a refusal."""
    sup = ServiceSupervisor(_Sandbox(tmp_path))
    sup.start("web", "python3 app.py", port=8100)
    declared = ToolOutcome.failed("Error: 'web' exited immediately (code 1).",
                                  world_changed=True, reason_code="service_exited_immediately")
    sup.start = lambda *a, **k: declared
    out = sup.restart("web")
    assert "was preserved" in out
    assert out.reason_code == "service_exited_immediately" and out.world_changed is True


def test_a_hijacked_port_is_not_answered_with_a_restart_to_another_port(tmp_path, _no_sleep):
    """The failure text used to say "restart '<name>' on another port" —
    a call `restart` now refuses."""
    sup = ServiceSupervisor(_Sandbox(tmp_path))
    sup._holder_pid = lambda port: 999
    sup._pid_ownership = lambda holder, pid: False
    out = str(sup.start("web", "python3 app.py", port=8100))
    assert "answered by a DIFFERENT process" in out, out
    assert "stop 'web' and start it on another port" in out
    assert "or restart" not in out


# ══════════════════════════════════════════════════════════════════════
# B — WEB-EXEC counts only what the browser tool declares a load
# ══════════════════════════════════════════════════════════════════════
from ghost_agent.core.agent import _web_exec_loaded
from ghost_agent.tools.browser import tool_browser

_FETCH_PAGE = "<html><script>fetch('/api/turn').then(r=>r.json())</script></html>"
_PLAIN_PAGE = "<html><body>static</body></html>"
_SVC = "http://127.0.0.1:8100"


class _Runner:
    """The sandbox as the browser tool sees it: one runner invocation that
    prints the sentinel line the real runner prints."""

    def __init__(self, out, code):
        self.out, self.code = out, code
        self.container = None

    def execute(self, cmd, timeout=600, **kw):
        return self.out, self.code

    def __getattr__(self, name):
        return lambda *a, **k: None


def _ok(status, title="Greece vs Turkey", js=None):
    def make(url):
        d = {"status": status, "url": url, "title": title, "text": "body", "length": 4}
        if js:
            d["js_errors"] = js
        return "[BROWSER_OK] " + json.dumps(d) + "\n", 0
    return make


def _refused(url):
    return f"[BROWSER_ERR] Page.goto: net::ERR_CONNECTION_REFUSED at {url}\n", 1


def _crashed(url):
    return "Traceback (most recent call last):\nplaywright crashed\n", 1


def _web_agent(tmp_path, monkeypatch, responses, page="templates/index.html",
               body=_FETCH_PAGE):
    """An agent whose `browser` is the REAL tool over a stub runner, so the
    probe reads the result shapes production emits. ``responses`` maps a URL
    (or "*") to a function url → (runner stdout, exit code)."""
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = MagicMock()
    visited = []

    async def browser(**kw):
        url = kw["url"]
        visited.append(url)
        make = responses.get(url) or responses["*"]
        out, code = make(url)
        return await tool_browser(
            operation=kw["operation"], url=url,
            wait_until=kw.get("wait_until", "domcontentloaded"),
            sandbox_dir=tmp_path, sandbox_manager=_Runner(out, code),
            allowed_local_ports={8100})

    agent.available_tools = {"browser": browser}
    monkeypatch.setattr(
        "ghost_agent.tools.file_system.project_scoped_sandbox",
        lambda ctx, stateful=False: (tmp_path, "/workspace"))
    sup = MagicMock()
    sup.list_entries.return_value = [
        {"name": "war-sim", "port": 8100, "workdir": "/workspace"}]
    sup._entry_alive.return_value = True
    monkeypatch.setattr(
        "ghost_agent.sandbox.services.get_service_supervisor", lambda sm: sup)
    target = tmp_path / page
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body)
    return agent, visited


async def _real_result(tmp_path, make, url):
    out, code = make(url)
    return await tool_browser(operation="navigate", url=url,
                              sandbox_dir=tmp_path, sandbox_manager=_Runner(out, code),
                              allowed_local_ports={8100})


async def test_loaded_reads_the_real_tools_outcomes(tmp_path):
    """The shapes the browser tool actually emits — and the ones the probe
    used to test for, which it does not."""
    u = _SVC + "/templates/index.html"
    ok = await _real_result(tmp_path, _ok(200), u)
    blocked = await _real_result(tmp_path, _ok(404, "404 Not Found"), u)
    refused = await _real_result(tmp_path, _refused, u)
    crashed = await _real_result(tmp_path, _crashed, u)
    assert _web_exec_loaded(ok) is True
    for r in (blocked, refused, crashed):
        assert _web_exec_loaded(r) is False
        # the old guard matched none of the real failures
        assert not str(r).lstrip().startswith("Error")
        assert "[BROWSER_ERR]" not in str(r)


@pytest.mark.parametrize("text,expected", [
    ("--- BROWSER RESULT ---\nSTATUS: OK\nOP: navigate\n", True),
    ("--- BROWSER RESULT ---\nSTATUS: BLOCKED (HTTP 404)\nOP: navigate\n", False),
    ("--- BROWSER RESULT ---\nSTATUS: ERROR\nboom", False),
    ("--- BROWSER RESULT ---\nSTATUS: OKAY\n", False),
    ("SUCCESS: navigated. Title: 'x'", False),      # an unknown shape is not a load
    ("STATUS: OK", False),                           # no tool header
    ("x\n--- BROWSER RESULT ---\nSTATUS: OK\n", False),   # a page QUOTING the header
    ("", False), (None, False),
])
def test_loaded_table(text, expected):
    assert _web_exec_loaded(text) is expected


def test_a_declared_failure_is_not_a_load_whatever_its_header_says():
    res = ToolOutcome.failed("--- BROWSER RESULT ---\nSTATUS: OK\nOP: navigate\n",
                             world_changed=False, reason_code="x")
    assert _web_exec_loaded(res) is False
    assert _web_exec_loaded(
        ToolOutcome.ok("--- BROWSER RESULT ---\nSTATUS: OK\nOP: navigate\n")) is True


@pytest.mark.parametrize("answer", [
    _ok(404, "404 Not Found"), _ok(500, "Internal Server Error"), _refused, _crashed])
async def test_a_page_that_did_not_load_is_not_a_clean_load(tmp_path, monkeypatch, answer):
    """The live probe: /templates/index.html → 404, reported clean. And the
    R1 review, against the real tool: a refused connection read clean."""
    agent, visited = _web_agent(tmp_path, monkeypatch, {"*": answer})
    assert await agent._execute_web_artifact(["templates/index.html"]) is None
    assert visited == [_SVC + "/templates/index.html"]


async def test_the_service_root_does_not_stand_in_for_a_page(tmp_path, monkeypatch):
    """R2 review: a 404 at the page's path was retried at `/`, and whatever
    `/` served — a JSON health route — certified the page. Nothing but the
    page's own URL is loaded."""
    agent, visited = _web_agent(tmp_path, monkeypatch, {
        _SVC + "/templates/index.html": _ok(404, "404 Not Found"),
        _SVC + "/": _ok(200, title="")})
    assert await agent._execute_web_artifact(["templates/index.html"]) is None
    assert _SVC + "/" not in visited


async def test_a_200_is_loaded_once(tmp_path, monkeypatch):
    agent, visited = _web_agent(tmp_path, monkeypatch, {"*": _ok(200)})
    assert await agent._execute_web_artifact(["templates/index.html"]) == (
        "templates/index.html", "")
    assert visited == [_SVC + "/templates/index.html"]


async def test_a_throwing_page_refutes(tmp_path, monkeypatch):
    agent, _ = _web_agent(tmp_path, monkeypatch, {
        "*": _ok(200, js=["TypeError: boom is not a function"])})
    page_rel, block = await agent._execute_web_artifact(["templates/index.html"])
    assert page_rel == "templates/index.html"
    assert block.startswith("UNCAUGHT JS EXCEPTIONS") and "boom" in block


@pytest.mark.parametrize("order", [
    ["templates/admin.html", "static/game.html"],
    ["static/game.html", "templates/admin.html"],
])
async def test_a_page_that_throws_refutes_whatever_else_failed_to_load(
        tmp_path, monkeypatch, order):
    """R2 review: the first page that did not load ended the probe, so a
    crash on a LATER page went unseen — the result depended on the order
    the files were written in."""
    agent, visited = _web_agent(tmp_path, monkeypatch, {
        _SVC + "/templates/admin.html": _ok(404, "404 Not Found"),
        _SVC + "/static/game.html": _ok(200, js=["TypeError: boom is not a function"])},
        page="templates/admin.html")
    (tmp_path / "static").mkdir()
    (tmp_path / "static" / "game.html").write_text(_FETCH_PAGE)
    page_rel, block = await agent._execute_web_artifact(order)
    assert page_rel == "static/game.html" and "boom" in block


async def test_one_unloaded_page_keeps_a_clean_neighbour_inconclusive(
        tmp_path, monkeypatch):
    agent, visited = _web_agent(tmp_path, monkeypatch, {
        _SVC + "/templates/admin.html": _ok(404, "404 Not Found"),
        _SVC + "/static/game.html": _ok(200)},
        page="templates/admin.html")
    (tmp_path / "static").mkdir()
    (tmp_path / "static" / "game.html").write_text(_FETCH_PAGE)
    for order in (["templates/admin.html", "static/game.html"],
                  ["static/game.html", "templates/admin.html"]):
        assert await agent._execute_web_artifact(order) is None
    assert len(visited) == 4                      # both pages probed, both times


@pytest.mark.parametrize("order", [["a.html", "b.html"], ["b.html", "a.html"]])
async def test_an_unserved_fetch_page_does_not_hide_a_crash_next_to_it(
        tmp_path, monkeypatch, order):
    """R3 review: a fetch-backed page no running service covers ended the
    probe before ANY page was loaded."""
    agent, visited = _web_agent(
        tmp_path, monkeypatch,
        {"*": _ok(200, js=["TypeError: boom is not a function"])},
        page="a.html", body=_FETCH_PAGE)
    (tmp_path / "b.html").write_text(_PLAIN_PAGE)
    none = MagicMock()
    none.list_entries.return_value = []
    monkeypatch.setattr(
        "ghost_agent.sandbox.services.get_service_supervisor", lambda sm: none)
    page_rel, block = await agent._execute_web_artifact(order)
    assert page_rel == "b.html" and "boom" in block
    assert len(visited) == 1 and visited[0].endswith("/b.html")     # a.html never loaded


async def test_an_unserved_fetch_page_keeps_a_clean_neighbour_inconclusive(
        tmp_path, monkeypatch):
    agent, visited = _web_agent(tmp_path, monkeypatch, {"*": _ok(200)},
                                page="a.html", body=_FETCH_PAGE)
    (tmp_path / "b.html").write_text(_PLAIN_PAGE)
    none = MagicMock()
    none.list_entries.return_value = []
    monkeypatch.setattr(
        "ghost_agent.sandbox.services.get_service_supervisor", lambda sm: none)
    assert await agent._execute_web_artifact(["a.html", "b.html"]) is None
    assert len(visited) == 1


async def test_a_page_that_cannot_be_read_is_not_loaded_by_file_url(tmp_path, monkeypatch):
    """R4 review: the host-side read failed, the fetch check was skipped by
    a bare `except`, and the page went on to a file:// load that read clean."""
    agent, visited = _web_agent(tmp_path, monkeypatch, {"*": _ok(200)},
                                page="a.html", body=_FETCH_PAGE)
    real = Path.read_text

    def read_text(self, *a, **kw):
        if self.name == "a.html":
            raise PermissionError("denied")
        return real(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", read_text)
    assert await agent._execute_web_artifact(["a.html"]) is None
    assert visited == []


@pytest.mark.parametrize("order", [["a.html", "b.html"], ["b.html", "a.html"]])
async def test_a_browser_that_raises_on_one_page_does_not_hide_the_next(
        tmp_path, monkeypatch, order):
    """R4 review: a raise on page one escaped the probe; the same two pages
    in the other order refuted."""
    agent, visited = _web_agent(
        tmp_path, monkeypatch,
        {"*": _ok(200, js=["TypeError: boom is not a function"])},
        page="a.html", body=_PLAIN_PAGE)
    (tmp_path / "b.html").write_text(_PLAIN_PAGE)
    real_browser = agent.available_tools["browser"]

    async def browser(**kw):
        if kw["url"].endswith("/a.html"):
            raise RuntimeError("sandbox manager went away")
        return await real_browser(**kw)

    agent.available_tools = {"browser": browser}
    page_rel, block = await agent._execute_web_artifact(order)
    assert page_rel == "b.html" and "boom" in block


async def test_a_browser_that_raises_is_inconclusive_not_clean(tmp_path, monkeypatch):
    agent, _ = _web_agent(tmp_path, monkeypatch, {"*": _ok(200)},
                          page="a.html", body=_PLAIN_PAGE)
    (tmp_path / "b.html").write_text(_PLAIN_PAGE)
    real_browser = agent.available_tools["browser"]

    async def browser(**kw):
        if kw["url"].endswith("/a.html"):
            raise RuntimeError("sandbox manager went away")
        return await real_browser(**kw)

    agent.available_tools = {"browser": browser}
    assert await agent._execute_web_artifact(["a.html", "b.html"]) is None


async def test_an_unreadable_page_does_not_end_the_probe(tmp_path, monkeypatch):
    """The page AFTER an unreadable one is still examined — here a
    fetch-backed page, which must still be found its running service."""
    agent, visited = _web_agent(
        tmp_path, monkeypatch,
        {"*": _ok(200, js=["TypeError: boom is not a function"])},
        page="a.html", body=_PLAIN_PAGE)
    (tmp_path / "b.html").write_text(_FETCH_PAGE)
    real = Path.read_text

    def read_text(self, *a, **kw):
        if self.name == "a.html":
            raise PermissionError("denied")
        return real(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", read_text)
    page_rel, block = await agent._execute_web_artifact(["a.html", "b.html"])
    assert page_rel == "b.html" and "boom" in block
    assert visited == [_SVC + "/b.html"]            # served, not file://


async def test_a_file_page_is_a_load(tmp_path, monkeypatch):
    agent, visited = _web_agent(tmp_path, monkeypatch, {"*": _ok(200)},
                                page="index.html", body=_PLAIN_PAGE)
    assert await agent._execute_web_artifact(["index.html"]) == ("index.html", "")
    assert visited[0].startswith("file://")


async def test_a_failed_file_load_is_inconclusive(tmp_path, monkeypatch):
    agent, _ = _web_agent(tmp_path, monkeypatch, {"*": _refused},
                          page="index.html", body=_PLAIN_PAGE)
    assert await agent._execute_web_artifact(["index.html"]) is None


# ══════════════════════════════════════════════════════════════════════
# C — "verdict deferred" only when a verdict is actually in flight
# ══════════════════════════════════════════════════════════════════════
from ghost_agent.utils.logging import requester_role_context


def _verdict_agent(async_mode=True):
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = MagicMock()
    agent._critic_async_enabled = lambda: async_mode
    agent._record_late_verdict = MagicMock()
    return agent


async def _settle():
    for _ in range(3):
        await asyncio.sleep(0)


async def test_attached_task_is_in_flight_until_it_lands():
    agent = _verdict_agent()
    fut = asyncio.get_running_loop().create_future()
    agent._attach_late_verdict_handler(fut, "T1", "fp", n_tools=1)
    assert agent._no_verdict_reason("T1")[0] == "deferred"
    fut.set_result((None, None))
    await _settle()
    assert agent._no_verdict_reason("T1")[0] == "landed"
    assert agent._record_late_verdict.call_count == 1
    assert agent._late_verdict_running() == {}


async def test_no_task_for_this_turn_is_not_deferred():
    """The live lie: async mode, nothing attached for THIS turn."""
    agent = _verdict_agent()
    fut = asyncio.get_running_loop().create_future()
    agent._attach_late_verdict_handler(fut, "OTHER", "fp", n_tools=1)
    kind, text = agent._no_verdict_reason("T1")
    assert kind == "empty"
    assert "deferred" not in text and "nothing is running late" in text
    fut.cancel()


async def test_a_cancelled_or_dead_task_did_not_land():
    """R1 review: a task that was cancelled, or died, read as "landed — see
    the LATE line above" with no LATE line ever printed."""
    agent = _verdict_agent()
    loop = asyncio.get_running_loop()
    cancelled, died = loop.create_future(), loop.create_future()
    agent._attach_late_verdict_handler(cancelled, "TC", "fp", n_tools=1)
    agent._attach_late_verdict_handler(died, "TD", "fp", n_tools=1)
    cancelled.cancel()
    died.set_exception(RuntimeError("critic node gone"))
    await _settle()
    for tid, word in (("TC", "was cancelled"), ("TD", "died")):
        kind, text = agent._no_verdict_reason(tid)
        assert kind == "lost" and word in text and "LATE line" not in text
    assert agent._record_late_verdict.call_count == 0
    assert agent._late_verdict_running() == {}


async def test_a_fault_in_the_late_side_effects_does_not_leave_the_turn_in_flight():
    """The task is marked finished BEFORE its side effects run."""
    agent = _verdict_agent()
    agent._record_late_verdict = MagicMock(side_effect=RuntimeError("store locked"))
    fut = asyncio.get_running_loop().create_future()
    agent._attach_late_verdict_handler(fut, "TF", "fp", n_tools=1)
    fut.set_result((None, None))
    await _settle()
    assert agent._late_verdict_running() == {}
    assert agent._no_verdict_reason("TF")[0] == "landed"


async def test_two_tasks_on_one_turn_are_counted():
    """The first landing must not hide the second, still running."""
    agent = _verdict_agent()
    loop = asyncio.get_running_loop()
    first, second = loop.create_future(), loop.create_future()
    agent._attach_late_verdict_handler(first, "T1", "fp", n_tools=1)
    agent._attach_late_verdict_handler(second, "T1", "fp", n_tools=1)
    first.set_result((None, None))
    await _settle()
    assert agent._no_verdict_reason("T1")[0] == "deferred"
    second.cancel()
    await _settle()
    # a sibling that was cancelled does not un-land the verdict that landed
    assert agent._no_verdict_reason("T1")[0] == "landed"


@pytest.mark.parametrize("kind,level", [
    ("member", "INFO"), ("deferred", "INFO"), ("landed", "INFO"),
    ("lost", "WARNING"), ("empty", "WARNING"), ("skipped", "WARNING"),
    ("", "WARNING"), ("something new", "WARNING"),       # unknown kinds warn
])
def test_no_verdict_level(kind, level):
    assert agent_mod._no_verdict_level(kind) == level


def test_member_turn_is_never_announced_as_deferred():
    agent = _verdict_agent()
    tok = requester_role_context.set("member")
    try:
        kind, text = agent._no_verdict_reason("T1")
    finally:
        requester_role_context.reset(tok)
    assert kind == "member"
    assert "deferred" not in text and "nothing will land late" in text


def test_sync_mode_with_no_task_says_skipped():
    agent = _verdict_agent(async_mode=False)
    assert agent._no_verdict_reason("T1")[0] == "skipped"


async def test_a_running_task_is_deferred_whatever_the_mode():
    """The sync path also hands a slow verdict to the late handler."""
    agent = _verdict_agent(async_mode=False)
    fut = asyncio.get_running_loop().create_future()
    agent._attach_late_verdict_handler(fut, "T1", "fp", n_tools=1)
    assert agent._no_verdict_reason("T1")[0] == "deferred"
    fut.cancel()


def test_without_a_trajectory_id_nothing_is_claimed_to_be_running():
    agent = _verdict_agent()
    assert agent._no_verdict_reason("")[0] == "empty"
    assert agent._no_verdict_reason(None)[0] == "empty"


async def test_ended_verdicts_are_bounded():
    agent = _verdict_agent()
    loop = asyncio.get_running_loop()
    for i in range(80):
        f = loop.create_future()
        agent._attach_late_verdict_handler(f, f"T{i}", "fp", n_tools=0)
        f.set_result((None, None))
    await _settle()
    assert len(agent._late_verdict_ended()) == 64
    assert agent._no_verdict_reason("T79")[0] == "landed"
    assert agent._no_verdict_reason("T0")[0] == "empty"
    assert agent._late_verdict_running() == {}


def _string_constants_in(func_node):
    return [n.value for n in ast.walk(func_node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def test_deferred_is_said_in_exactly_one_place():
    """Enumeration (R1): the announcement "verdict deferred — verifying
    asynchronously …" is produced by `_no_verdict_reason` and by the streamed
    gate, and by nothing else. A second emitter on the non-streamed path is
    how the line came to be printed from the mode alone."""
    tree = ast.parse((SRC / "core" / "agent.py").read_text(encoding="utf-8"))
    funcs = [n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    owners = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and "verifying asynchronously" in node.value):
            continue
        # the INNERMOST function holding the constant
        holder = max((f for f in funcs if f.lineno <= node.lineno <= f.end_lineno),
                     key=lambda f: f.lineno)
        if node.value.strip() in (ast.get_docstring(holder) or ""):
            continue                       # a docstring quoting the phrase
        owners.add(holder.name)
    # `stream_wrapper` is the streamed gate: it prints AFTER it spawns and
    # attaches, in the same block — its own, correct, deferral.
    assert owners == {"_no_verdict_reason", "stream_wrapper"}


# ══════════════════════════════════════════════════════════════════════
# E — a number-only phrase that names a line governs that line
# ══════════════════════════════════════════════════════════════════════
from ghost_agent.core.turn_state_check import (
    mechanical_constraints, refute_turn_state, NUMBER_ON_LAST_LINE)
from ghost_agent.eval.banks import normalize_gsm8k_text

_GSM = normalize_gsm8k_text(
    {"question": "Janet had 22 green pens and 10 yellow pens. Then she bought "
                 "6 bags of blue pens and 2 bags of red pens. There were 9 pens "
                 "in each bag of blue and 6 pens in each bag of red. How many "
                 "pens does Janet have now?", "answer": "x #### 98"}, 0)
_GSM_REQ = ("### SYNTHETIC TRAINING EXERCISE\nSolve this challenge efficiently. "
            "Your DELIVERABLE is your final REPLY — you may use the `execute` "
            "tool to check arithmetic, but no file you write counts as an "
            "answer. Your final reply must follow the answer format the "
            "challenge specifies (the final numeric answer on its own last "
            "line).\n\n" + _GSM["challenge"])
_STEPS = ("Janet starts with 22 + 10 = 32 pens.\nBlue: 6 × 9 = 54. "
          "Red: 2 × 6 = 12.\nTotal: 32 + 54 + 12 = 98.\n\n")


def _number_kinds(request):
    return [(c.kind, c.value) for c in mechanical_constraints(request)
            if c.kind == "number_only"]


def test_the_bench_prompt_is_read_as_a_last_line_rule():
    assert _number_kinds(_GSM_REQ) == [("number_only", NUMBER_ON_LAST_LINE)]


@pytest.mark.parametrize("reply", [
    _STEPS + "98",                 # the live first answer, refuted 3 of 3
    _STEPS + "**98**",
    _STEPS + "98\n",
    "98",
    _STEPS + "```\n98\n```",
    # R2 review: whatever passes as a whole number-only reply passes as the
    # last line — a separate, stricter pattern refuted these
    _STEPS + "3/4", _STEPS + "-3/4", _STEPS + "2:30", _STEPS + "1.5e3",
    _STEPS + "\\boxed{98}", _STEPS + "Answer: 98", _STEPS + "98 pens",
    _STEPS + "The answer is 98.",
])
def test_step_by_step_with_the_number_last_is_not_refuted(reply):
    assert refute_turn_state(request=_GSM_REQ, reply=reply) == []


@pytest.mark.parametrize("reply", [
    _STEPS + "Janet is happy with her pens.",                      # no number on it
    _STEPS + "Therefore, after buying all of those bags, Janet has 98 pens in total now.",
    _STEPS + "She counted them twice. The total is 98.",           # two sentences
    "98\n\nThat is how many pens Janet has after all of her shopping trips.",
])
def test_a_last_line_that_is_not_the_number_is_refuted(reply):
    issues = refute_turn_state(request=_GSM_REQ, reply=reply)
    assert [k for k, _ in issues] == ["number_only"]
    assert "last line" in issues[0][1]


@pytest.mark.parametrize("request_text", [
    "What is 7 times 6? Reply with just the number.",
    "Compute (17 * 3) + 5. Reply with only the number.",
    "Reply with just the number on one line.",          # the whole reply
    "Answer with only the number, on its own line.",    # still the whole reply
    "Just the number.",
])
def test_unscoped_requests_keep_the_whole_reply_rule(request_text):
    assert _number_kinds(request_text) == [("number_only", None)]
    issues = refute_turn_state(
        request=request_text,
        reply="Here is how I worked it out in full detail.\n\n42")
    assert "number_only" in [k for k, _ in issues]


@pytest.mark.parametrize("request_text,scoped", [
    ("Work it out in full. Put just the number on the last line.", True),
    ("Explain briefly. The final line must be only the number.", True),
    ("Give the total on its own line (only the number on that line).", True),
    # R2 review: the scope is stated one SENTENCE before the rule
    ("Show your work. Put your answer on the last line. Just the number, no units.", True),
    # "then" makes the clause two-part: the reader abstains altogether
    ("Show your working, then put just the number on the last line.", False),
])
def test_line_scoped_phrasings(request_text, scoped):
    assert _number_kinds(request_text) == (
        [("number_only", NUMBER_ON_LAST_LINE)] if scoped else [])
    assert refute_turn_state(
        request=request_text, reply="Two plus two.\n\n4") == []
    if scoped:
        assert [k for k, _ in refute_turn_state(
            request=request_text, reply="4\n\nI would rather not say more.")] == ["number_only"]


@pytest.mark.parametrize("req", [
    "Reply with only the number on the last line of data.csv.",
    "Read the last line in log.txt and reply with just the number.",
    "Take the final line from results.txt. Reply with just the number.",
])
def test_the_last_line_of_a_file_is_not_the_last_line_of_the_reply(req):
    """"…the last line of data.csv" names a line of a FILE."""
    assert _number_kinds(req) == [("number_only", None)]
    assert [k for k, _ in refute_turn_state(
        request=req, reply="I opened the file and read every row.\n\n42")] == ["number_only"]


def test_the_scoped_issue_is_still_a_shape_issue():
    """The refute text must stay in the `number_only:` family the repair
    directive and the task filter key on (`_REFUTE_TASK_ARTIFACT_RE`)."""
    kind, msg = refute_turn_state(
        request=_GSM_REQ, reply=_STEPS + "Janet is happy with her pens.")[0]
    assert GhostAgent._REFUTE_TASK_ARTIFACT_RE.search(f"{kind}: {msg}")


# ══════════════════════════════════════════════════════════════════════
# F — self-play: template fallback, retry temperature, malformed data
# ══════════════════════════════════════════════════════════════════════
import ghost_agent.core.dream as dream_mod
from ghost_agent.core.dream import (
    Dreamer, TEMPLATE_RETRY_REASONS, challenge_generation_temperature,
    setup_data_defect, _template_fallback_on_defective_challenge,
    DREAM_EVIDENCE_MIN_NEW_FRACTION, DATA_DEFECT_MIN_RECORDS,
    challenge_origin, verify_run_measured,
)
from ghost_agent.tools.outcome import ToolOutcome


class _Toy:
    def __init__(self, results, origin="generated", bank_empty=False):
        self.results = list(results)
        self.calls = []
        self._origin = origin
        self._bank_empty = bank_empty

    @_template_fallback_on_defective_challenge
    async def run(self, model_name="m", is_background=False,
                  injected_challenge=None, bench_meta=None, *,
                  force_template=False):
        self.calls.append({"injected": injected_challenge,
                           "force_template": force_template,
                           "is_background": is_background})
        # the real run records where its challenge came from
        # (a forced run with no template to pick falls through to generation)
        self.last_challenge_origin = ("template" if force_template and not self._bank_empty
                                      else self._origin)
        return self.results.pop(0)


def _failed(code, text="x"):
    return ToolOutcome.failed(text, world_changed=False, reason_code=code)


@pytest.mark.parametrize("code", sorted(TEMPLATE_RETRY_REASONS))
async def test_defective_generated_challenge_reruns_on_a_template(code):
    toy = _Toy([_failed(code), "SUCCESS"])
    assert await toy.run("m", is_background=True) == "SUCCESS"
    assert [c["force_template"] for c in toy.calls] == [False, True]
    assert toy.calls[1]["is_background"] is True      # the caller's args survive


async def test_a_caller_asking_for_a_template_gets_exactly_one_run():
    toy = _Toy([_failed("selfplay_setup_failed"), "SUCCESS"])
    out = await toy.run(force_template=True)
    assert getattr(out, "reason_code", "") == "selfplay_setup_failed"
    assert len(toy.calls) == 1


async def test_a_forced_run_that_had_to_generate_is_not_run_a_third_time():
    """With an empty template bank the forced run falls through to the LLM
    generator, so its challenge IS "generated" — and it is still the last."""
    toy = _Toy([_failed("selfplay_setup_failed")] * 3, bank_empty=True)
    await toy.run()
    assert [c["force_template"] for c in toy.calls] == [False, True]


async def test_the_template_run_is_never_retried_again():
    toy = _Toy([_failed("selfplay_setup_failed"), _failed("selfplay_setup_failed")])
    out = await toy.run()
    assert getattr(out, "reason_code", "") == "selfplay_setup_failed"
    assert len(toy.calls) == 2


@pytest.mark.parametrize("origin", ["template", "journal", "replay", "bench", ""])
async def test_only_a_generated_challenge_is_swapped_for_a_template(origin):
    """A replay or a bench item is the POINT of its run; a template or a
    journal challenge that fails its setup is not a generator defect
    (R1 review)."""
    toy = _Toy([_failed("selfplay_setup_failed"), "SUCCESS"], origin=origin)
    out = await toy.run("m", False, {"challenge": "c"} if origin == "replay" else None)
    assert getattr(out, "reason_code", "") == "selfplay_setup_failed"
    assert len(toy.calls) == 1


async def test_a_sandbox_fault_is_not_retried_on_a_template():
    """The sandbox failed, not the challenge: a template would fail too."""
    toy = _Toy([_failed("selfplay_setup_failed",
                        "setup failed:\n[SANDBOX INFRA ERROR — not your code] docker down"),
                "SUCCESS"])
    out = await toy.run()
    assert getattr(out, "reason_code", "") == "selfplay_setup_failed"
    assert len(toy.calls) == 1


class _SlowToy(_Toy):
    """A first run that takes 0.2 s."""

    @_template_fallback_on_defective_challenge
    async def run(self, model_name="m", is_background=False,
                  injected_challenge=None, bench_meta=None, *,
                  force_template=False):
        self.calls.append({"force_template": force_template})
        self.last_challenge_origin = "template" if force_template else "generated"
        if not force_template:
            await asyncio.sleep(0.2)
        return self.results.pop(0)


@pytest.mark.parametrize("budget,runs", [
    (0.3, 1),        # 0.2 s used of 0.3: past half — a second run would be cut off
    (600.0, 2),      # room for a second
    (None, 2),       # the idle loop sets no bound: never refused for time
])
async def test_the_retry_fits_the_budget_its_caller_states(budget, runs):
    toy = _SlowToy([_failed("selfplay_setup_failed"), "SUCCESS"])
    if budget is not None:
        toy.cycle_budget_s = budget
    await toy.run("m", True)
    assert len(toy.calls) == runs


def test_every_bounded_caller_states_its_budget():
    """Enumeration: a function that wraps `synthetic_self_play` in
    `asyncio.wait_for` and can run a GENERATED challenge sets the same
    bound as `<dreamer>.cycle_budget_s` — otherwise its retry is cut off
    mid-run. (The idle loop sets no bound and states none.)"""
    found = 0
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            stated = [ast.dump(n.value) for n in ast.walk(fn)
                      if isinstance(n, ast.Assign) and any(
                          isinstance(t, ast.Attribute) and t.attr == "cycle_budget_s"
                          for t in n.targets)]
            for node in ast.walk(fn):
                if not (isinstance(node, ast.Call)
                        and getattr(node.func, "attr", "") == "wait_for" and node.args):
                    continue
                inner = node.args[0]
                if not (isinstance(inner, ast.Call)
                        and getattr(inner.func, "attr", "") == "synthetic_self_play"):
                    continue
                if "injected_challenge" in {k.arg for k in inner.keywords}:
                    continue                 # a bench item / replay: never retried
                found += 1
                timeout = {k.arg: k.value for k in node.keywords}.get("timeout")
                assert ast.dump(timeout) in stated, f"{path.name}:{fn.name}:{inner.lineno}"
    assert found >= 2, "the walk found no bounded callers — the enumeration is blind"


async def test_the_one_shot_tool_states_its_budget(monkeypatch):
    """…and through the real tool: the dreamer it runs carries the bound."""
    from ghost_agent.tools import memory as memory_mod
    seen = {}

    class FakeDreamer:
        def __init__(self, ctx):
            pass

        async def synthetic_self_play(self, is_background=False):
            seen["budget"] = getattr(self, "cycle_budget_s", None)
            return "ok"

    monkeypatch.setattr("ghost_agent.core.dream.Dreamer", FakeDreamer)
    ctx = types.SimpleNamespace(last_user_content="please run self-play")
    await memory_mod.tool_self_play(ctx)
    assert seen.get("budget") == memory_mod.SELF_PLAY_CYCLE_TIMEOUT_S


@pytest.mark.parametrize("code", ["selfplay_quality_gate", "other", None])
async def test_other_failures_pass_through(code):
    toy = _Toy([_failed(code), "SUCCESS"])
    out = await toy.run()
    assert getattr(out, "reason_code", "x") == code and len(toy.calls) == 1


async def test_a_success_is_returned_as_is():
    toy = _Toy(["SUCCESS", "never"])
    assert await toy.run() == "SUCCESS" and len(toy.calls) == 1


@pytest.mark.parametrize("kw,expected", [
    (dict(bench_meta={"bank": "mbpp"}, injected_challenge={"c": 1}, journal_source=False, template_used=True), "bench"),
    (dict(bench_meta=None, injected_challenge={"c": 1}, journal_source=False, template_used=True), "replay"),
    (dict(bench_meta=None, injected_challenge=None, journal_source=True, template_used=False), "journal"),
    (dict(bench_meta=None, injected_challenge=None, journal_source=True, template_used=True), "journal"),
    (dict(bench_meta=None, injected_challenge=None, journal_source=False, template_used=True), "template"),
    (dict(bench_meta=None, injected_challenge=None, journal_source=False, template_used=False), "generated"),
])
def test_challenge_origin(kw, expected):
    assert challenge_origin(**kw) == expected


def test_self_play_reason_codes_are_all_classified():
    """Enumeration (R1): every `reason_code="selfplay_…"` the solve loop can
    return is either a defect the template retry covers or is listed here as
    deliberately not retried. A new failure code that forfeits the slot
    unnoticed fails this test."""
    not_retried = {"selfplay_quality_gate"}   # already falls back in-line
    tree = ast.parse((SRC / "core" / "dream.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.AsyncFunctionDef)
              and n.name == "synthetic_self_play")
    codes = set()
    for call in ast.walk(fn):
        if isinstance(call, ast.Call):
            for kw in call.keywords:
                if (kw.arg == "reason_code" and isinstance(kw.value, ast.Constant)
                        and str(kw.value.value).startswith("selfplay_")):
                    codes.add(kw.value.value)
    assert codes, "the walk found no reason codes — the enumeration is blind"
    assert codes - TEMPLATE_RETRY_REASONS - not_retried == set()
    assert TEMPLATE_RETRY_REASONS <= codes      # no stale entry either


@pytest.mark.parametrize("dups,temp", [(0, 0.3), (1, 0.6), (2, 0.9), (3, 0.9),
                                        (-1, 0.3), ("x", 0.3)])
def test_retry_temperature_rises_only_with_duplicate_rejections(dups, temp):
    assert challenge_generation_temperature(dups) == pytest.approx(temp)


_LEDGER = (b"BATCH_1|2024-01-01T00:00:00|2024-01-01T00:01:40|ERROR|95\\n"
           b"BATCH_2|2024-01-01T00:02:00|2024-01-01T00:03:00|COMPLETE|40\\n"
           b"BATCH_3|2024-01-01T00:04:00|2024-01-01T00:05:00|PENDING|7\\n"
           b"BATCH_4|2024-01-01T00:06:00|2024-01-01T00:07:00|COMPLETE|9\\n")
_ROW = b"sensor_%d,2024-01-01T00:00:0%d,%d.5"


def _joined(n, sep=b"\\n"):
    return sep.join(_ROW % (i, i, i) for i in range(n))


def test_escaped_newline_dataset_is_a_defect():
    why = setup_data_defect({"transaction_ledger.txt": _LEDGER})
    assert why and "transaction_ledger.txt" in why and "4 records" in why


@pytest.mark.parametrize("name,blob", [
    ("csv_like.txt", _joined(6)),                          # `"\\n".join(rows)`
    ("trailing_break.txt", _joined(6) + b"\n"),            # …then one real break
    ("data/nested.log", _joined(5)),
    ("minimum.txt", _joined(DATA_DEFECT_MIN_RECORDS)),
    ("spaces.log", b"\\n".join(b"2024-01-0%d LOGIN user%d" % (i, i) for i in range(1, 6))),
])
def test_record_files_joined_by_a_literal_are_defects(name, blob):
    assert setup_data_defect({name: blob})


@pytest.mark.parametrize("name,blob", [
    ("good.txt", _LEDGER.replace(b"\\n", b"\n")),                  # real breaks
    ("one_break_inside.txt", _joined(3) + b"\n" + _joined(3)),     # it HAS lines
    # real lines whose FIELDS carry an escaped newline — a log of messages
    ("messages.log", b"\n".join(b"%d|timeout\\nretrying" % i for i in range(1, 13))),
    ("three_records.txt", _joined(DATA_DEFECT_MIN_RECORDS - 1)),
    ("tiny.txt", b"a,1\\nb,2\\nc,3\\nd,4"),                        # under 40 bytes
    (".setup.py", _LEDGER),                                         # harness file
    ("bin.dat", b"\x00\x01" + _LEDGER),
    ("not_bytes.txt", "a,1\\nb,2\\nc,3\\nd,4\\ne,5\\nf,6\\ng,7\\nh,8"),
    ("doc.json", json.dumps({"t": "a,1\nb,2\nc,3\nd,4\ne,5\nf,6\ng,7\nh,8\ni,9"}).encode()),
    # the legitimate single-line shapes the first rule flagged (R1 review)
    ("paths.txt", b"C:\\nick\\notes\\new\\nightly\\build.txt is the path the nightly job uses"),
    ("patterns.txt", b"\\n+|\\s*\\n\\s*|[^\\n]+|(?:\\r\\n|\\n)+ patterns that split input into lines"),
    ("min.js", b'function f(a){return a.split("\\n").map(function(x){return x+"\\n"}).join("\\n")}'),
    ("two.jsonl", b'{"msg": "a\\nb\\nc", "k": 1}\n{"msg": "d\\ne\\nf", "k": 2}'),
    ("mail.tpl", b"Hello {name},\\nYour order {id} shipped.\\nThanks,\\nThe team -- rendered by mailer"),
    ("banner.ini", b"message=first line\\nsecond line\\nthird line\\nfourth line of the banner text"),
    ("pairs.repr", b"['alpha\\nbeta', 'gamma\\ndelta', 'eps\\nzeta', 'eta\\ntheta'] # python repr"),
    # pieces that are not RECORDS: the separator count differs
    ("ragged.txt", b"a,b,c,d\\ne,f\\ng,h,i,j,k\\nl\\nm,n,o,p,q,r,s,t,u,v,w,x,y,z"),
    ("words.txt", b"alphaalpha\\nbetabetabeta\\ngammagamma\\ndeltadelta\\nepsilonepsilon"),
])
def test_well_formed_data_is_not_a_defect(name, blob):
    assert setup_data_defect({name: blob}) is None


# Verbatim head rows of datasets the LIVE replay ledger's setup scripts
# write (each script run in an isolated, network-less container on
# 2026-09-30; challenge id, file, bytes). The last one is a challenge ABOUT
# malformed rows: 18 regular records plus the two it plants on purpose.
_LIVE_DEFECTS = [
    ('1beac70527db', 'transaction_stream.txt',
     b'txn_0|user_1|-43.747311194333264|1790771297\\ntxn_1|user_3|11.222963450869052|1790771287\\ntxn_2|user_2|134.1178035410031|1790771277\\ntxn_3|user_5|-28.26529184264596|1790771267\\ntxn_4|user_4|-42.0543301295541|1790771257\\ntxn_5|user_1|4.659493700900839|1790771247'),
    ('2e086dc98366', 'network_stream.txt',
     b'0,192.168.1.2,10.0.0.1,ICMP,627\\n1000,192.168.1.4,10.0.0.8,TCP,273\\n2000,192.168.1.9,10.0.0.3,ICMP,928\\n3000,192.168.1.1,10.0.0.1,TCP,511\\n4000,192.168.1.4,10.0.0.17,ICMP,118\\n5000,192.168.1.9,10.0.0.7,ICMP,1394'),
    ('1b11aec5ef78', 'connection_events.log',
     b'2023-10-27 10:00:03 CONNECT user_10\\n2023-10-27 10:00:06 CONNECT user_8\\n2023-10-27 10:00:09 DISCONNECT user_9\\n2023-10-27 10:00:12 DISCONNECT user_10\\n2023-10-27 10:00:15 DISCONNECT user_9\\n2023-10-27 10:00:18 CONNECT user_8'),
    ('beba1acc232f', 'session_data.txt',
     b'2:s_1_7553:1790770184:1790770385:7\\n3:s_2_9531:1790765515:1790765584:14\\n4:s_3_2575:1790768810:1790768992:7\\n5:s_4_5596:1790764933:1790764979:27\\n6:s_5_7526:1790762726:1790762810:14\\n7:s_6_8918:1790761601:1790761841:21'),
    ('a8ccd9505956', 'transaction_ledger.txt',
     b'BATCH_1|2023-10-27T09:58:17Z|2023-10-27T10:00:00Z|ERROR|95\\nBATCH_2|2023-10-27T09:44:45Z|2023-10-27T10:00:00Z|COMPLETE|18\\nBATCH_3|2023-10-27T09:13:48Z|2023-10-27T10:00:00Z|ERROR|95\\nBATCH_4|2023-10-27T09:54:03Z|2023-10-27T10:00:00Z|PENDING|76\\nBATCH_5|2023-10-27T09:57:57Z|2023-10-27T10:00:00Z|COMPLETE|12\\nBATCH_6|2023-10-27T09:25:30Z|2023-10-27T10:00:00Z|COMPLETE|78'),
    ('cc4351c3c9c1', 'connection_events.txt',
     b'1790770929,192.168.1.6,10.0.0.6,UDP,FAILED\\n1790770953,192.168.1.6,10.0.0.6,UDP,SUCCESS\\n1790770703,192.168.1.10,10.0.0.10,TCP,FAILED\\n1790770994,192.168.1.5,10.0.0.5,TCP,SUCCESS\\n1790771238,192.168.1.1,10.0.0.1,TCP,SUCCESS\\n1790771236,192.168.1.1,10.0.0.1,UDP,FAILED\\n1790770877,192.168.1.7,10.0.0.7,UDP,SUCCESS\\n1790770829,192.168.1.8,10.0.0.8,UDP,SUCCESS\\n1790771063,192.168.1.4,10.0.0.4,TCP,FAILED\\n1790770896,192.168.1.7,10.0.0.7,TCP,FAILED\\n1790771233,192.168.1.1,10.0.0.1,TCP,SUCCESS\\n1790771180,192.168.1.2,10.0.0.2,ICMP,SUCCESS\\n1790770802,192.168.1.8,10.0.0.8,UDP,FAILED\\n1790770891,192.168.1.7,10.0.0.7,UDP,SUCCESS\\n1790770802,192.168.1.8,10.0.0.8,ICMP,SUCCESS\\n1790770823,192.168.1.8,10.0.0.8,ICMP,FAILED\\n1790770930,192.168.1.6,10.0.0.6,TCP,FAILED\\n1790771133,192.168.1.3,10.0.0.3,TCP,SUCCESS\\n1672531200,1.1.1.1,2.2.2.2,TCP\\n1672531200,1.1.1.1,2.2.2.2,TCP,SUCCESS,EXTRA'),
]


@pytest.mark.parametrize("cid,name,blob", _LIVE_DEFECTS)
def test_the_rule_on_real_generator_output(cid, name, blob):
    assert blob.count(b"\n") == 0 and blob.count(b"\\n") >= 4    # the fixture is the defect
    assert setup_data_defect({name: blob}), cid


def test_a_few_planted_bad_rows_do_not_hide_the_defect():
    rows = [_ROW % (i, i, i) for i in range(20)]
    assert setup_data_defect({"x.txt": b"\\n".join(rows + [b"only,three", b"a,b,c,d,e,f"])})
    # …but a file that is mostly NOT records is not one
    assert setup_data_defect({"x.txt": b"\\n".join(rows[:6] + [b"no separator here"] * 6)}) is None


@pytest.mark.parametrize("origin,policy", [
    ("generated", "discard"), ("replay", "no_lesson"),
    ("template", ""), ("journal", ""), ("bench", ""), ("", ""), (None, ""),
])
def test_data_defect_policy(origin, policy):
    assert dream_mod.data_defect_policy(origin) == policy


def test_no_snapshot_is_not_a_defect():
    assert setup_data_defect(None) is None and setup_data_defect({}) is None


def test_the_real_entry_point_carries_the_fallback():
    """The wiring: `Dreamer.synthetic_self_play` is the wrapped function and
    accepts the flag the wrapper sets."""
    import inspect
    fn = Dreamer.synthetic_self_play
    inner = inspect.unwrap(fn)
    assert inner is not fn
    assert "force_template" in inspect.signature(inner).parameters
    assert asyncio.iscoroutinefunction(fn)


def _sp_context(tmp_path):
    ctx = MagicMock()
    ctx.memory_system = MagicMock()
    ctx.skill_memory = MagicMock()
    ctx.skill_memory.get_recent_failures.return_value = "No failures"
    ctx.llm_client = MagicMock()
    ctx.args = MagicMock()
    ctx.args.perfect_it = True
    ctx.args.smart_memory = 1.0
    ctx.sandbox_manager = MagicMock()
    ctx.sandbox_dir = str(tmp_path)
    ctx.tor_proxy = None
    ctx.scratchpad = MagicMock()
    ctx.frontier_tracker = None
    return ctx


def _xml(d):
    return "".join(f"<{k}>{v}</{k}>\n" for k, v in d.items())


_TEMPLATE = ("Print 7 to stdout from solution.py.", "",
             "import subprocess, sys\n"
             "p = subprocess.run([sys.executable, 'solution.py'], "
             "capture_output=True, text=True)\n"
             "sys.exit(0 if p.stdout.strip() == '7' else 1)\n")


def _saturated_frontier(tmp_path, monkeypatch):
    """A frontier whose target cluster is saturated and whose coin lands on
    LLM generation — the path the live 02:2x run took. WITHOUT the forced
    branch a second run would generate again, not fall back."""
    from ghost_agent.memory.frontier import FrontierTracker
    tracker = FrontierTracker(tmp_path)
    tracker.pick_seed = MagicMock(return_value={
        "mode": "frontier", "cluster_key": None, "hint": "",
        "saturated_clusters": ["python_general"]})
    tracker.most_similar_recent_challenge = MagicMock(return_value=(0.0, ""))
    monkeypatch.setattr("random.random", lambda: 0.95)     # → LLM path
    return tracker


def _solver(mock_agent_cls):
    agent = MagicMock()
    agent.handle_chat = AsyncMock(return_value=("done", None, None))
    agent._get_recent_transcript.return_value = "t" * 300
    agent.disabled_tools = set()
    agent.available_tools = {}
    mock_agent_cls.return_value = agent
    return agent


@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_setup_crash_reruns_the_idle_slot_on_a_template(
        mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch):
    """End to end through the real solve loop: run 1 generates a challenge
    whose setup script crashes in the sandbox; the slot is then spent on a
    template instead of being forfeited (live 2026-09-30 02:2x)."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    picks = []
    monkeypatch.setattr(
        "ghost_agent.core.challenge_templates.try_template",
        lambda *a, **k: None)

    def fake_pick(*a, **kw):
        picks.append(list(kw.get("exclude_clusters") or []))
        # every non-saturated cluster is "taken": only the unrestricted
        # second pick can return a template
        return None if kw.get("exclude_clusters") else _TEMPLATE

    monkeypatch.setattr(
        "ghost_agent.core.challenge_templates.pick_random_template", fake_pick)

    ctx = _sp_context(tmp_path)
    ctx.frontier_tracker = _saturated_frontier(tmp_path, monkeypatch)
    generations = []

    async def chat(payload, **kw):
        if "AI training coordinator" in payload["messages"][0]["content"]:
            generations.append(1)
        return {"choices": [{"message": {"content": _xml({
            "challenge_prompt": "Write solution.py that prints 7.",
            "setup_script": "rows = []\nprint(rows[3])",
            "validation_script": "assert True",
        })}}]}

    ctx.llm_client.chat_completion = AsyncMock(side_effect=chat)
    agent = _solver(mock_agent_cls)

    boxes = []

    def make_box(*a, **kw):
        box = MagicMock()
        first = not boxes

        def execute(cmd, *aa, **kk):
            if first and ".setup.py" in cmd and "py_compile" not in cmd:
                return ("Traceback (most recent call last):\nIndexError: "
                        "list index out of range", 1)
            return ("OK", 0)

        box.execute.side_effect = execute
        boxes.append(box)
        return box

    mock_sandbox_cls.side_effect = make_box

    out = await Dreamer(ctx).synthetic_self_play("test-model")

    assert len(generations) == 1, "run 2 generated again instead of using a template"
    assert len(boxes) == 2, "the slot was forfeited: no second run"
    # the forced pick: first without the saturated cluster, then — nothing
    # there — without restriction: any template beats a forfeited slot
    assert picks == [["python_general"], []]
    assert getattr(out, "reason_code", None) not in TEMPLATE_RETRY_REASONS
    assert "SUCCESS" in str(out)
    # the solver was only ever given the TEMPLATE challenge
    first_body = agent.handle_chat.await_args_list[0].args[0]
    assert "Print 7 to stdout" in json.dumps(first_body)
    assert "Write solution.py that prints 7" not in json.dumps(first_body)


@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_a_template_is_not_policed_for_its_data(
        mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch):
    """Templates are hand-written; the escaped-newline rule polices what
    the GENERATOR wrote."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.setattr(
        "ghost_agent.core.challenge_templates.try_template", lambda *a, **k: None)
    monkeypatch.setattr(
        "ghost_agent.core.challenge_templates.pick_random_template",
        lambda *a, **k: ("Sum transaction_ledger.txt.",
                         "open('transaction_ledger.txt','w').write('x')",
                         "assert True"))
    ctx = _sp_context(tmp_path)
    ctx.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "{}"}}]})
    agent = _solver(mock_agent_cls)

    def make_box(sandbox_dir, *a, **kw):
        box = MagicMock()

        def execute(cmd, *aa, **kk):
            if ".setup.py" in cmd and "py_compile" not in cmd:
                (Path(sandbox_dir) / "transaction_ledger.txt").write_bytes(_LEDGER)
            return ("OK", 0)

        box.execute.side_effect = execute
        return box

    mock_sandbox_cls.side_effect = make_box
    out = await Dreamer(ctx).synthetic_self_play("test-model")
    assert getattr(out, "reason_code", "") != "selfplay_setup_malformed_data"
    agent.handle_chat.assert_awaited()
    assert mock_sandbox_cls.call_count == 1


# ══════════════════════════════════════════════════════════════════════
# G — a lesson whose verification failed is discarded
# ══════════════════════════════════════════════════════════════════════
_LESSON_JSON = (
    '{"trigger": "stateful event processing with an initial state", '
    '"anti_pattern": "printing diagnostics to stdout before the JSON", '
    '"correct_pattern": "print only the final JSON object to stdout", '
    '"domains": ["algo"], "confidence": 0.95, '
    '"task": "event processing", "mistake": "stdout noise", '
    '"solution": "print only the JSON"}')

_INJECTED = {
    "challenge": "Print the JSON object to stdout from solution.py.",
    "setup_script": "",
    "validation_script": "assert True",
}
_INJECTED_WITH_DATA = {
    "challenge": "Sum the ledger in transaction_ledger.txt.",
    "setup_script": "open('transaction_ledger.txt','w').write('x')",
    "validation_script": "assert True",
}


async def _run_injected(tmp_path, monkeypatch, mock_agent_cls, mock_sandbox_cls, *,
                        validator=((1, "FAIL: wrong output"), (0, "ok"), (0, "ok")),
                        finals=("done", "done", "done", "done"),
                        verify_raises=False, injected=None, data=None,
                        tracker=None, bench_meta=None):
    """Drive the REAL solve loop on an injected challenge. ``validator`` is
    the (exit code, output) of each `.validator.py` run in order: attempt 1,
    attempt 2, …, then the lesson-verification re-run. ``data`` is what the
    setup script "wrote"."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    ctx = _sp_context(tmp_path)
    ctx.frontier_tracker = tracker
    ctx.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": _LESSON_JSON}}]})
    calls = {"n": 0}

    async def fake_handle_chat(body, **kw):
        calls["n"] += 1
        if verify_raises and calls["n"] == 3:
            raise RuntimeError("upstream died")
        body.setdefault("messages", []).extend([
            {"role": "assistant", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "ok"}])
        return (finals[min(calls["n"], len(finals)) - 1], None, None)

    agent = MagicMock()
    agent.handle_chat = AsyncMock(side_effect=fake_handle_chat)
    agent._get_recent_transcript.return_value = "t" * 300
    agent.disabled_tools = set()
    agent.available_tools = {}
    agent.max_turns_override = None
    agent.max_thinking_chars_override = None
    mock_agent_cls.return_value = agent

    runs = list(validator)

    def make_box(sandbox_dir, *a, **kw):
        box = MagicMock()

        def execute(cmd, *aa, **kk):
            if data is not None and ".setup.py" in cmd and "py_compile" not in cmd:
                (Path(sandbox_dir) / "transaction_ledger.txt").write_bytes(data)
            if ".validator.py" in cmd and "selftest" not in cmd:
                code, out = runs.pop(0) if runs else (0, "ok")
                return (out, code)
            return ("OK", 0)

        box.execute.side_effect = execute
        return box

    mock_sandbox_cls.side_effect = make_box
    dreamer = Dreamer(ctx)
    dreamer._generalization_guard = lambda *a, **k: (True, "")
    out = await dreamer.synthetic_self_play(
        "test-model", injected_challenge=dict(injected or _INJECTED),
        bench_meta=bench_meta)
    return ctx, dreamer, out, calls, agent


_sp = (patch("ghost_agent.sandbox.docker.DockerSandbox"),
       patch("ghost_agent.core.agent.GhostAgent"))


def _selfplay(fn):
    return _sp[0](_sp[1](fn))


@_selfplay
async def test_a_disproved_lesson_is_not_saved(
        mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch):
    """Struggled, then won; the re-run with the lesson injected failed."""
    ctx, dreamer, out, calls, _ = await _run_injected(
        tmp_path, monkeypatch, mock_agent_cls, mock_sandbox_cls,
        validator=((1, "FAIL"), (0, "ok"), (1, "FAIL: wrong output")))
    assert calls["n"] == 3, "the verify run did not happen — the pin is blind"
    assert dreamer.last_lesson_verify_disproved is True
    ctx.skill_memory.learn_lesson.assert_not_called()


@_selfplay
async def test_a_verified_lesson_is_saved_verified(
        mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch):
    ctx, dreamer, out, calls, _ = await _run_injected(
        tmp_path, monkeypatch, mock_agent_cls, mock_sandbox_cls,
        validator=((1, "FAIL"), (0, "ok"), (0, "ok")))
    assert calls["n"] == 3
    assert dreamer.last_lesson_verify_disproved is False
    ctx.skill_memory.learn_lesson.assert_called_once()
    assert ctx.skill_memory.learn_lesson.call_args.kwargs["verified"] is True


@_selfplay
async def test_a_verification_that_could_not_run_keeps_the_lesson_unverified(
        mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch):
    """An exception is not a measurement: the lesson is kept, unverified."""
    ctx, dreamer, out, calls, _ = await _run_injected(
        tmp_path, monkeypatch, mock_agent_cls, mock_sandbox_cls,
        validator=((1, "FAIL"), (0, "ok")), verify_raises=True)
    assert calls["n"] == 3
    assert dreamer.last_lesson_verify_disproved is False
    ctx.skill_memory.learn_lesson.assert_called_once()
    assert ctx.skill_memory.learn_lesson.call_args.kwargs["verified"] is False


@pytest.mark.parametrize("verify_final,verify_out,code", [
    # `docker.execute()` RETURNS the banner, it does not raise
    ("done", "[SANDBOX INFRA ERROR — not your code] docker daemon unreachable", 1),
    # `handle_chat` RETURNS this, it does not raise
    ("CRITICAL: The upstream LLM server is unreachable. Please retry.", "FAIL: no solution.py", 1),
])
@_selfplay
async def test_a_verify_run_that_measured_nothing_disproves_nothing(
        mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch,
        verify_final, verify_out, code):
    """R1 review: an outage in the verify run read as "the lesson did not
    help" and discarded it. The solve loop excuses exactly these."""
    ctx, dreamer, out, calls, _ = await _run_injected(
        tmp_path, monkeypatch, mock_agent_cls, mock_sandbox_cls,
        validator=((1, "FAIL"), (0, "ok"), (code, verify_out)),
        finals=("done", "done", verify_final))
    assert calls["n"] == 3
    assert dreamer.last_lesson_verify_disproved is False
    ctx.skill_memory.learn_lesson.assert_called_once()
    assert ctx.skill_memory.learn_lesson.call_args.kwargs["verified"] is False


@pytest.mark.parametrize("final,outage", [
    ("CRITICAL: Upstream error 503: no healthy upstream", True),
    ("CRITICAL: The upstream LLM server is unreachable. It may have crashed", True),
    ("\n  CRITICAL: An unexpected error occurred while communicating with the LLM: x", True),
    # R4 review: a deferred-correction banner is inserted ABOVE the outage text
    ("⚠️ **Correction to my previous answer:** the speed is 40.\n\n---\n\n"
     "CRITICAL: The upstream LLM server is unreachable. Please retry.", True),
    ("Found 3 events.\nCRITICAL: disk full (x3)", False),      # a quoted log line
    ("The log has 2 CRITICAL: lines.", False),
    ("CRITICAL: temperature above threshold in rack 4", False), # …even at the head
    ("CRITICAL: The disk is full on node 3", False),            # starts like a banner
    ("CRITICAL: Upstream latency is above the threshold", False),
    ("CRITICAL: An unexpected error occurred while parsing row 7", False),
    ("", False), (None, False),
])
def test_reply_is_upstream_outage(final, outage):
    assert dream_mod.reply_is_upstream_outage(final) is outage


def test_every_outage_reply_the_turn_loop_emits_is_a_known_banner():
    """Enumeration: each `final_ai_content = "CRITICAL: …"` in the turn loop
    starts with one of `UPSTREAM_OUTAGE_BANNERS` — a reworded or new outage
    reply would otherwise be charged to the solver."""
    tree = ast.parse((SRC / "core" / "agent.py").read_text(encoding="utf-8"))
    heads = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "final_ai_content" for t in node.targets)):
            continue
        v = node.value
        head = (v.value if isinstance(v, ast.Constant) and isinstance(v.value, str)
                else v.values[0].value if isinstance(v, ast.JoinedStr) and v.values
                and isinstance(v.values[0], ast.Constant) else "")
        if str(head).startswith("CRITICAL:"):
            heads.append((node.lineno, head))
    assert len(heads) >= 3, "the walk found no outage replies — the enumeration is blind"
    for lineno, head in heads:
        assert head.startswith(dream_mod.UPSTREAM_OUTAGE_BANNERS), f"line {lineno}: {head[:60]!r}"


@pytest.mark.parametrize("final", [
    "_(Turn cancelled: operator stop.)_ No output was produced before the cancellation.",
    "partial work\n\n_(Turn cancelled: client went away.)_",
    "I hit my context limit while gathering data for this step — the inputs were too large",
    "[ATTEMPT_ABORTED_STRIKE_CAP] I hit a hard limit after repeated failures",
    # R5 review: the turn loop APPENDS the marker after any narration, and
    # the finalizer can insert a banner above either message
    "Let me run the script and check the output.\n\n[ATTEMPT_ABORTED_STRIKE_CAP] I hit a hard limit",
    "⚠️ **Correction to my previous answer:** x\n\n---\n\nI hit my context limit while gathering data",
    "⚠️ **Correction to my previous answer:** x\n\n---\n\nCRITICAL: Upstream error 503: down",
    # every hard abort of the turn loop, not only the strike cap
    "I tried three approaches.\n\n[ATTEMPT_ABORTED_NO_PROGRESS] No progress in 4 turns",
    "[ATTEMPT_ABORTED_THINKING_LOOP] stopped a runaway reasoning loop",
])
def test_a_turn_that_never_got_to_work_disproves_nothing(final):
    """R3/R4 reviews: the other replies `handle_chat` returns without
    having done the work."""
    assert verify_run_measured(final, "wrong total", 1) is False


@pytest.mark.parametrize("final", [
    "The model's context limit is 8192 tokens, so I chunked the file. Total: 41.",
    "I hit my target of 12 rows. Total: 41.",
    "The turn was cancelled in the log at 12:03; the total is 41.",
    "ATTEMPT 2 was aborted by the script; the total is 41.",
    "The log line reads ATTEMPT_ABORTED for job 7; the total is 41.",
    "[ATTEMPT_3] retried the query; the total is 41.",
])
def test_a_reply_that_did_the_work_is_a_measurement(final):
    assert verify_run_measured(final, "wrong total", 1) is True


def _abort_notes_the_turn_loop_writes():
    """The head of every abort note `core/agent.py` puts into a reply: the
    literal passed to each `_with_abort_note(…)` call, and the two marker
    constants."""
    tree = ast.parse((SRC / "core" / "agent.py").read_text(encoding="utf-8"))
    heads = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_with_abort_note"
                and len(node.args) == 2):
            note = node.args[1]
            head = (note.value if isinstance(note, ast.Constant)
                    else note.values[0].value if isinstance(note, ast.JoinedStr)
                    and note.values and isinstance(note.values[0], ast.Constant) else None)
            heads.append((node.lineno, head))
    return heads + [(0, agent_mod.FORCED_FINAL_LOOP_MARKER), (0, agent_mod.CROSS_TURN_LOOP_MARKER)]


def test_every_abort_the_turn_loop_writes_is_read_as_not_measured():
    """Enumeration over the EMITTERS (R7 review: a marker prefix matched
    docstrings and regexes, so a reworded emitter went unnoticed and its
    reply became a "measurement" again). Each note is built into a reply
    by the real helper, after narration, and must read as not measured."""
    heads = _abort_notes_the_turn_loop_writes()
    assert len(heads) >= 4, "the walk found no abort notes — the enumeration is blind"
    for lineno, head in heads:
        assert isinstance(head, str) and head, f"line {lineno}: the note is not a literal"
        reply = agent_mod._with_abort_note("Let me check the output first.", head + " …")
        assert verify_run_measured(reply, "wrong total", 1) is False, f"line {lineno}: {head[:50]!r}"


def test_the_other_not_measured_replies_are_the_ones_the_turn_loop_writes():
    tree = ast.parse((SRC / "core" / "agent.py").read_text(encoding="utf-8"))
    literals = [n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    for marker in ("I hit my context limit while", "_(Turn cancelled:"):
        assert marker in dream_mod._VERIFY_NOT_MEASURED_MARKERS
        assert any(lit.startswith(marker) for lit in literals), marker


def test_a_reply_that_quotes_the_word_is_still_a_measurement():
    """R2 review: "CRITICAL:" anywhere in the reply read as an outage — and
    8 live challenges are ABOUT log lines carrying it."""
    assert verify_run_measured("Found 3 events.\nCRITICAL: disk full (x3)", "wrong total", 1) is True
    assert verify_run_measured("  CRITICAL: Upstream error 503: bad gateway", "", 1) is False


@pytest.mark.parametrize("final,out,code,graded,measured", [
    ("done", "FAIL: wrong output", 1, "artifact", True),
    ("done", "[SANDBOX INFRA ERROR — not your code] x", 1, "artifact", False),
    ("CRITICAL: Upstream error 502: bad gateway", "FAIL", 1, "artifact", False),
    ("the answer is 9", "answer.txt missing", 5, "final_response", False),
    ("the answer is 9", "exit five elsewhere", 5, "artifact", True),
    ("the answer is 9", "wrong answer", 1, "final_response", True),
    (None, None, 1, "artifact", True),
])
def test_verify_run_measured(final, out, code, graded, measured):
    assert verify_run_measured(final, out, code, graded) is measured


@_selfplay
async def test_a_failed_run_keeps_its_mistake_lesson(
        mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch):
    """A run that never passed records a MISTAKE; one more failure with the
    lesson injected does not disprove it. Only a struggled-then-WON lesson
    ("this gets it right first time") can be disproved."""
    ctx, dreamer, out, calls, _ = await _run_injected(
        tmp_path, monkeypatch, mock_agent_cls, mock_sandbox_cls,
        validator=((1, "FAIL"),) * 4)
    assert calls["n"] == 4, "three attempts + the verify run"
    assert dreamer.last_lesson_verify_disproved is True
    ctx.skill_memory.learn_lesson.assert_called_once()
    assert ctx.skill_memory.learn_lesson.call_args.kwargs["verified"] is False


@_selfplay
async def test_the_verify_run_gets_the_solvers_budget(
        mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch):
    """The re-run now decides whether the lesson is kept: it must not have
    10 turns where the run it is compared with had 15."""
    ctx, dreamer, out, calls, agent = await _run_injected(
        tmp_path, monkeypatch, mock_agent_cls, mock_sandbox_cls,
        validator=((1, "FAIL"), (0, "ok"), (0, "ok")))
    assert agent.max_turns_override == 15
    assert agent.max_thinking_chars_override == 12000


@_selfplay
async def test_a_stale_disproved_marker_cannot_discard_the_next_lesson(
        mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch):
    """The marker from an earlier run must not outlive it: with the helper
    replaced (as a caller/tests may), a True left behind would discard."""
    monkeypatch.setattr(Dreamer, "_verify_lesson_helpful",
                        AsyncMock(return_value=False))
    real_init = Dreamer.__init__

    def init(self, *a, **k):
        real_init(self, *a, **k)
        self.last_lesson_verify_disproved = True            # stale

    monkeypatch.setattr(Dreamer, "__init__", init)
    ctx, dreamer, out, calls, _ = await _run_injected(
        tmp_path, monkeypatch, mock_agent_cls, mock_sandbox_cls,
        validator=((1, "FAIL"), (0, "ok")))
    ctx.skill_memory.learn_lesson.assert_called_once()


# ── the malformed dataset: discard what was GENERATED, replay what is replayed ──
def _templated_seed(tmp_path):
    """A frontier whose seed names a cluster that HAS a template — the
    shape ~12% of live replays run under (R1 review): the template is looked
    up even though the injected challenge is what runs."""
    from ghost_agent.memory.frontier import FrontierTracker
    tracker = FrontierTracker(tmp_path)
    tracker.pick_seed = MagicMock(return_value={
        "mode": "frontier", "cluster_key": "data_analysis", "hint": ""})
    return tracker


@pytest.mark.parametrize("templated_seed", [False, True])
@_selfplay
async def test_a_replay_with_malformed_data_runs_but_teaches_nothing(
        mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch, templated_seed):
    """A replay is a measurement — two confirmed regressions in the live
    ledger sit on such challenges, and their recheck is the only path that
    restores the lessons they quarantined — but no lesson is drawn from a
    dataset the generator wrote wrong."""
    ctx, dreamer, out, calls, agent = await _run_injected(
        tmp_path, monkeypatch, mock_agent_cls, mock_sandbox_cls,
        validator=((1, "FAIL"), (0, "ok"), (0, "ok")),
        injected=_INJECTED_WITH_DATA, data=_LEDGER,
        tracker=_templated_seed(tmp_path) if templated_seed else None)
    assert getattr(out, "reason_code", None) is None and "SUCCESS" in str(out)
    assert str(dreamer.last_self_play_status).startswith("SUCCESS")   # a decisive replay
    assert dreamer.last_challenge_origin == "replay"
    assert calls["n"] == 2                       # solved; no lesson-verify run
    ctx.skill_memory.learn_lesson.assert_not_called()
    ctx.llm_client.chat_completion.assert_not_awaited()     # no extraction either


@_selfplay
async def test_a_replay_with_well_formed_data_still_teaches(
        mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch):
    """The control: the same replay over real line breaks mints its lesson."""
    ctx, dreamer, out, calls, _ = await _run_injected(
        tmp_path, monkeypatch, mock_agent_cls, mock_sandbox_cls,
        validator=((1, "FAIL"), (0, "ok"), (0, "ok")),
        injected=_INJECTED_WITH_DATA, data=_LEDGER.replace(b"\\n", b"\n"))
    ctx.skill_memory.learn_lesson.assert_called_once()


@_selfplay
async def test_a_bench_bank_item_is_not_policed_for_its_data(
        mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch):
    """Bench items come from an external bank with external ground truth."""
    ctx, dreamer, out, calls, _ = await _run_injected(
        tmp_path, monkeypatch, mock_agent_cls, mock_sandbox_cls,
        validator=((1, "FAIL"), (0, "ok"), (0, "ok")),
        injected=_INJECTED_WITH_DATA, data=_LEDGER,
        bench_meta={"bank": "mbpp", "item_id": "mbpp-1", "cluster": "algo"})
    assert dreamer.last_challenge_origin == "bench"
    ctx.skill_memory.learn_lesson.assert_called_once()


_GOOD_GEN = {"challenge_prompt": "Write solution.py that prints 7.",
             "validation_script": "assert True"}


async def _generate(tmp_path, monkeypatch, mock_agent_cls, mock_sandbox_cls,
                    replies, similarity):
    """Drive the generation loop: ``replies`` are the generator's outputs in
    order, ``similarity`` what the diversity gate measures for each one that
    reaches it. Returns the temperature of each generation call."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.setattr(
        "ghost_agent.core.challenge_templates.try_template", lambda *a, **k: None)
    monkeypatch.setattr(
        "ghost_agent.core.challenge_templates.pick_random_template",
        lambda *a, **k: None)
    from ghost_agent.memory.frontier import FrontierTracker
    ctx = _sp_context(tmp_path)
    tracker = FrontierTracker(tmp_path)      # the loop accepts only the real class
    tracker.most_similar_recent_challenge = MagicMock(side_effect=list(similarity))
    ctx.frontier_tracker = tracker
    temps, queue = [], list(replies)

    async def chat(payload, **kw):
        if "AI training coordinator" in payload["messages"][0]["content"]:
            temps.append(payload["temperature"])
            return {"choices": [{"message": {"content": _xml(queue.pop(0))}}]}
        return {"choices": [{"message": {"content": "{}"}}]}

    ctx.llm_client.chat_completion = AsyncMock(side_effect=chat)
    _solver(mock_agent_cls)
    box = MagicMock()
    box.execute.return_value = ("OK", 0)
    mock_sandbox_cls.return_value = box
    await Dreamer(ctx).synthetic_self_play("test-model")
    return temps


@_selfplay
async def test_a_duplicate_rejection_raises_the_retry_temperature(
        mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch):
    """Attempt 1 is rejected as a near-duplicate, so attempt 2 samples at
    0.6; a second duplicate, and attempt 3 samples at 0.9."""
    dup = (0.92, "You are given two files, event_log.txt and user_profiles.txt")
    temps = await _generate(tmp_path, monkeypatch, mock_agent_cls, mock_sandbox_cls,
                            [_GOOD_GEN] * 3, [dup, dup, (0.0, "")])
    assert temps == [pytest.approx(0.3), pytest.approx(0.6), pytest.approx(0.9)]


@_selfplay
async def test_a_rejection_of_another_kind_ends_the_streak(
        mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch):
    """R1 review: the counter never reset, so a rejection that needed a
    specific edit was retried hot. Duplicate → 0.6; then a reply with no
    validator (rejected before the diversity gate) → back to 0.3."""
    dup = (0.92, "You are given two files, event_log.txt and user_profiles.txt")
    temps = await _generate(
        tmp_path, monkeypatch, mock_agent_cls, mock_sandbox_cls,
        [_GOOD_GEN, {"challenge_prompt": "Write solution.py."}, _GOOD_GEN],
        [dup, (0.0, "")])
    assert temps == [pytest.approx(0.3), pytest.approx(0.6), pytest.approx(0.3)]


@_selfplay
async def test_escaped_newline_data_discards_the_generated_challenge(
        mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch):
    """End to end: the generated setup script 'wrote' a one-line dataset of
    literal backslash-n; the challenge never reaches the solver and the slot
    goes to a template."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.setattr(
        "ghost_agent.core.challenge_templates.try_template", lambda *a, **k: None)
    monkeypatch.setattr(
        "ghost_agent.core.challenge_templates.pick_random_template",
        lambda *a, **k: _TEMPLATE)
    ctx = _sp_context(tmp_path)
    ctx.frontier_tracker = _saturated_frontier(tmp_path, monkeypatch)
    ctx.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": _xml({
            "challenge_prompt": "Sum the ledger in transaction_ledger.txt.",
            "setup_script": "open('transaction_ledger.txt','w').write('x')",
            "validation_script": "open('transaction_ledger.txt').read()\nassert True",
            "reference_solution": "print(open('transaction_ledger.txt').read())",
        })}}]})
    agent = _solver(mock_agent_cls)
    boxes = []

    def make_box(sandbox_dir, *a, **kw):
        box = MagicMock()
        first = not boxes

        def execute(cmd, *aa, **kk):
            if first and ".setup.py" in cmd and "py_compile" not in cmd:
                (Path(sandbox_dir) / "transaction_ledger.txt").write_bytes(_LEDGER)
            return ("OK", 0)

        box.execute.side_effect = execute
        boxes.append(box)
        return box

    mock_sandbox_cls.side_effect = make_box
    dreamer = Dreamer(ctx)
    out = await dreamer.synthetic_self_play("test-model")
    assert len(boxes) == 2, "the malformed dataset reached the solver"
    assert dreamer.last_challenge_origin == "template"
    first_body = agent.handle_chat.await_args_list[0].args[0]
    assert "Print 7 to stdout" in json.dumps(first_body)
    assert "transaction_ledger" not in json.dumps(first_body)


def test_the_generator_is_told_how_to_write_a_line_break():
    from ghost_agent.core.prompts import SYNTHETIC_CHALLENGE_PROMPT as P
    rule = P[P.index("# 8. LINE BREAKS"):P.index("</setup_script>", P.index("# 8. LINE BREAKS"))]
    assert '`"\\n".join(rows)`' in rule             # ONE backslash, as rendered
    assert '`"\\\\n"`' in rule                       # the doubled form, named
    assert "\n" not in rule.strip()                  # the rule itself is one line


# ══════════════════════════════════════════════════════════════════════
# H — a frequency bump needs new evidence
# ══════════════════════════════════════════════════════════════════════
from ghost_agent.memory.skills import (
    SkillMemory, incoming_evidence, evidence_is_new, remember_evidence,
    seen_evidence, EVIDENCE_KEYS_MAX)


def _rule(sm, **kw):
    return sm.learn_lesson(
        "Parsing CSV exports", "reading rows by position",
        "Always read CSV columns by header name, never by index.",
        trigger="Parsing CSV exports", **kw)


def _freq(sm):
    return json.loads(sm.file_path.read_text())[0]["frequency"]


class _VectorStore:
    """The vector half of the lesson store, as `learn_lesson` uses it: a
    stored twin is a near-exact match for a re-learn of the same text, so
    the dedup hit comes from the VECTOR branch — the one production takes
    (every live caller passes `memory_system`; R1 review: every evidence
    test ran without one, on the JSON branch only)."""

    ADD_STORED = "stored"

    def __init__(self):
        self.rows = {}
        self.collection = self

    def add(self, text, meta=None):
        self.rows[f"id{len(self.rows)}"] = (text, dict(meta or {}))
        return self.ADD_STORED

    def query(self, query_texts, n_results=3, where=None):
        ids = list(self.rows)[:n_results]
        return {"ids": [ids],
                "documents": [[self.rows[i][0] for i in ids]],
                "metadatas": [[self.rows[i][1] for i in ids]],
                "distances": [[0.01 for _ in ids]]}

    def delete(self, ids=None, where=None):
        for i in ids or []:
            self.rows.pop(i, None)

    def update(self, ids=None, metadatas=None):
        pass


def _vrule(sm, vec, **kw):
    return sm.learn_lesson(
        "Parsing CSV exports", "reading rows by position",
        "Always read CSV columns by header name, never by index.",
        memory_system=vec, trigger="Parsing CSV exports", **kw)


def test_the_vector_branch_is_the_one_under_test(tmp_path, monkeypatch):
    """The fixture must reach the VECTOR dedup branch, or the tests below
    pin the JSON one twice."""
    sm, vec = SkillMemory(tmp_path), _VectorStore()
    seen = []
    real = sm._find_duplicate_lesson

    def spy(*a, **k):
        out = real(*a, **k)
        seen.append((out or {}).get("source"))
        return out

    monkeypatch.setattr(sm, "_find_duplicate_lesson", spy)
    assert _vrule(sm, vec, source_challenge_hash="c1") == "written"
    assert _vrule(sm, vec, source_challenge_hash="c1") == "reinforced"
    assert seen == [None, "vector"]


def test_vector_branch_same_evidence_does_not_bump(tmp_path):
    sm, vec = SkillMemory(tmp_path), _VectorStore()
    for _ in range(3):
        _vrule(sm, vec, source_challenge_hash="c1")
    assert _freq(sm) == 1


def test_vector_branch_new_evidence_bumps_once(tmp_path):
    sm, vec = SkillMemory(tmp_path), _VectorStore()
    _vrule(sm, vec, source_challenge_hash="c1")
    _vrule(sm, vec, source_challenge_hash="c2")
    _vrule(sm, vec, source_challenge_hash="c2")
    _vrule(sm, vec, source_trajectory_id="t9")
    assert _freq(sm) == 3


def test_vector_branch_no_evidence_bumps_as_before(tmp_path):
    sm, vec = SkillMemory(tmp_path), _VectorStore()
    _vrule(sm, vec)
    _vrule(sm, vec)
    assert _freq(sm) == 2


def test_vector_branch_dream_window(tmp_path):
    sm, vec = SkillMemory(tmp_path), _VectorStore()
    kw = dict(source="dream", evidence_min_new_fraction=0.5)
    window = [f"traj:{i}" for i in range(60)]
    _vrule(sm, vec, evidence_refs=window, **kw)
    _vrule(sm, vec, evidence_refs=window[3:] + ["a", "b", "c"], **kw)      # 3 of 60 fresh
    assert _freq(sm) == 1
    _vrule(sm, vec, evidence_refs=window[30:] + [f"n{i}" for i in range(30)], **kw)
    assert _freq(sm) == 2


def test_vector_branch_uncounted_evidence_stays_unseen(tmp_path):
    """A re-learn that did not count must not REMEMBER its keys: they are
    still fresh the day the window has enough of them."""
    sm, vec = SkillMemory(tmp_path), _VectorStore()
    kw = dict(source="dream", evidence_min_new_fraction=0.5)
    window = [f"traj:{i}" for i in range(60)]
    _vrule(sm, vec, evidence_refs=window, **kw)
    _vrule(sm, vec, evidence_refs=window[3:] + ["a", "b", "c"], **kw)      # 3 fresh: no
    assert _freq(sm) == 1
    _vrule(sm, vec, evidence_refs=window[30:] + ["a", "b", "c"]
           + [f"n{i}" for i in range(27)], **kw)                           # 30 fresh: yes
    assert _freq(sm) == 2


def test_same_challenge_replayed_does_not_bump(tmp_path):
    sm = SkillMemory(tmp_path)
    assert _rule(sm, source_challenge_hash="c1") == "written"
    assert _rule(sm, source_challenge_hash="c1") == "reinforced"
    assert _rule(sm, source_challenge_hash="c1") == "reinforced"
    assert _freq(sm) == 1


def test_a_new_challenge_bumps_once(tmp_path):
    sm = SkillMemory(tmp_path)
    _rule(sm, source_challenge_hash="c1")
    _rule(sm, source_challenge_hash="c2")
    _rule(sm, source_challenge_hash="c2")
    assert _freq(sm) == 2


def test_each_new_trajectory_is_new_evidence(tmp_path):
    sm = SkillMemory(tmp_path)
    for tid in ("t1", "t2", "t3", "t2"):
        _rule(sm, source_trajectory_id=tid)
    assert _freq(sm) == 3


def test_a_relearn_that_names_no_evidence_bumps_as_before(tmp_path):
    sm = SkillMemory(tmp_path)
    _rule(sm)
    _rule(sm)
    assert _freq(sm) == 2


def test_distilled_lesson_counts_cases_not_passes(tmp_path):
    """freq=84 'from 5 cases': one bump per NEW case handle."""
    sm = SkillMemory(tmp_path)
    _rule(sm, source_refs=["case:1", "case:2", "case:3"])
    _rule(sm, source_refs=["case:1", "case:2", "case:3"])           # unchanged
    _rule(sm, source_refs=["case:2", "case:3", "case:1"])           # reordered
    assert _freq(sm) == 1
    _rule(sm, source_refs=["case:1", "case:2", "case:3", "case:4"])
    assert _freq(sm) == 2
    _rule(sm, source_refs=["case:4", "case:1"])
    assert _freq(sm) == 2


def test_dream_window_needs_half_of_it_fresh(tmp_path):
    """13 → 14 → 15 → 16 in a day: a 60-fragment window sliding by 3."""
    sm = SkillMemory(tmp_path)
    window = [f"traj:{i}" for i in range(60)]
    kw = dict(source="dream",
              evidence_min_new_fraction=DREAM_EVIDENCE_MIN_NEW_FRACTION)
    _rule(sm, evidence_refs=window, **kw)
    for step in range(1, 10):                     # nine re-dreams, 3 fresh each
        window = window[3:] + [f"traj:{60 + step * 3 + j}" for j in range(3)]
        _rule(sm, evidence_refs=window, **kw)
    assert _freq(sm) == 1                         # 27 of 60 fresh: not yet half
    window = window[3:] + ["traj:n1", "traj:n2", "traj:n3"]
    _rule(sm, evidence_refs=window, **kw)         # 30 of 60 fresh
    assert _freq(sm) == 2
    _rule(sm, evidence_refs=window, **kw)         # the same window again
    assert _freq(sm) == 2


def test_evidence_refs_do_not_leak_into_source_refs(tmp_path):
    sm = SkillMemory(tmp_path)
    _rule(sm, evidence_refs=["traj:1", "traj:2"], source_refs=["ep:9"])
    row = json.loads(sm.file_path.read_text())[0]
    assert row["source_refs"] == ["ep:9"]
    assert len(row["evidence_keys"]) == 3
    assert not any("traj" in k for k in row["evidence_keys"])     # digests


def test_legacy_row_counts_its_own_provenance_as_seen():
    legacy = {"source_trajectory_id": "t1", "source_challenge_hash": "c1",
              "source_refs": ["ep:1"]}
    assert evidence_is_new(legacy, incoming_evidence(None, "t1")) is False
    assert evidence_is_new(legacy, incoming_evidence(["ep:1"], "", "c1")) is False
    assert evidence_is_new(legacy, incoming_evidence(None, "t2")) is True
    assert len(seen_evidence(legacy)) == 3


@pytest.mark.parametrize("fraction,unseen,total,expected", [
    (0.0, 1, 60, True), (0.5, 29, 60, False), (0.5, 30, 60, True),
    (0.5, 0, 60, False), (1.0, 59, 60, False), (1.0, 60, 60, True),
    (None, 1, 5, True), ("bad", 1, 5, True),
])
def test_evidence_is_new_fraction_table(fraction, unseen, total, expected):
    lesson = {}
    remember_evidence(lesson, incoming_evidence([f"old:{i}" for i in range(total - unseen)]))
    inc = incoming_evidence([f"old:{i}" for i in range(total - unseen)]
                            + [f"new:{i}" for i in range(unseen)])
    assert len(inc) == total
    assert evidence_is_new(lesson, inc, fraction) is expected


def test_remembered_evidence_is_bounded_newest_kept():
    lesson = {}
    remember_evidence(lesson, incoming_evidence([f"r:{i}" for i in range(EVIDENCE_KEYS_MAX + 50)]))
    assert len(lesson["evidence_keys"]) == EVIDENCE_KEYS_MAX
    newest = incoming_evidence([f"r:{EVIDENCE_KEYS_MAX + 49}"])[0]
    oldest = incoming_evidence(["r:0"])[0]
    assert newest in lesson["evidence_keys"] and oldest not in lesson["evidence_keys"]
    # the dream's 150-fragment window fits whole, twice over at most: a
    # window larger than the bound would count as new on every re-read
    assert 150 <= EVIDENCE_KEYS_MAX <= 300


def test_incoming_evidence_is_ordered_and_deduplicated():
    a = incoming_evidence(["x", "y", "x", "", None], "t", "c", ["y", "z"])
    assert len(a) == 5 and len(set(a)) == 5
    assert a[:2] == incoming_evidence(["x", "y"])
    assert incoming_evidence(None, "", "") == []
    assert incoming_evidence("single") == incoming_evidence(["single"])


async def test_a_redetected_tool_pattern_counts_only_when_its_support_changed():
    """The dream re-detects the same "[Pattern]" over the same lessons every
    REM cycle; its evidence is the pattern AND its support count."""
    ctx = MagicMock()
    ctx.memory_system = MagicMock()
    ctx.memory_system.collection = MagicMock()
    ctx.skill_memory = MagicMock()
    ctx.skill_memory._get_lock = lambda: threading.RLock()
    ctx.skill_memory.file_path = MagicMock()
    ctx.skill_memory.file_path.read_text.return_value = "[]"
    ctx.llm_client = MagicMock()
    ctx._last_dream_fragment_ids = None
    dreamer = Dreamer(ctx)
    dreamer.memory.collection.get.return_value = {
        "ids": [f"id{i}" for i in range(5)],
        "documents": [f"auto memory number {i}" for i in range(5)],
        "metadatas": [{"type": "auto"}] * 5, "embeddings": [[0.1]] * 5}
    ctx.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": json.dumps(
            {"consolidations": [], "heuristics": []})}}]})
    with patch.object(dream_mod, "detect_tool_patterns", return_value=[
            {"pattern_name": "strategy:execute → file_system",
             "description": "Recurring tool pattern", "frequency": 7}]):
        await dreamer.dream()
    call = [c for c in ctx.skill_memory.learn_lesson.call_args_list
            if c.kwargs.get("source") == "dream_pattern"][0]
    assert call.kwargs["evidence_refs"] == ["strategy:execute → file_system#7"]


def test_every_frequency_bump_is_evidence_gated():
    """Enumeration (R1): no `frequency … + 1` in the lesson store outside an
    `if` that asked `evidence_is_new`. The two dedup branches were bumped
    independently once already (§4L)."""
    tree = ast.parse((SRC / "memory" / "skills.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "learn_lesson")
    parents = {}
    for node in ast.walk(fn):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    bumps = []
    for node in ast.walk(fn):
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Subscript)
                        and isinstance(t.slice, ast.Constant)
                        and t.slice.value == "frequency" for t in node.targets)):
            bumps.append(node)
    assert len(bumps) >= 2, "the walk found no bumps — the enumeration is blind"
    for b in bumps:
        p, gated = parents.get(b), False
        while p is not None and p is not fn:
            if isinstance(p, ast.If) and isinstance(p.test, ast.Name) \
                    and p.test.id == "_counted":
                gated = True
                break
            p = parents.get(p)
        assert gated, f"frequency bumped without an evidence gate at line {b.lineno}"


# ══════════════════════════════════════════════════════════════════════
# K — the Turn Outcome line prints THIS request's confidence
# ══════════════════════════════════════════════════════════════════════
class _Reading:
    def __init__(self, c):
        self.composite = c


def _conf_agent():
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = MagicMock()
    agent.context.last_confidence = None
    agent.context.last_confidence_req = ""
    return agent


def test_confidence_is_only_this_requests():
    agent = _conf_agent()
    agent._set_last_confidence(_Reading(0.78), "43199788")
    assert agent._turn_confidence("43199788") == 0.78
    # the live leak: the next sim / Slack member turn computed none
    assert agent._turn_confidence("f711d7eb") is None
    assert agent._turn_confidence("slack-b4c319b1") is None
    assert agent._turn_confidence("") is None
    assert agent._turn_confidence(None) is None


def test_a_later_reading_replaces_the_stamp():
    agent = _conf_agent()
    agent._set_last_confidence(_Reading(0.78), "a")
    agent._set_last_confidence(_Reading(0.91), "b")
    assert agent._turn_confidence("a") is None
    assert agent._turn_confidence("b") == 0.91


def test_an_unidentified_reading_belongs_to_no_request():
    """A reading stamped with NO request id (a context that has none) must
    not be claimed by the next turn that also has none."""
    agent = _conf_agent()
    agent._set_last_confidence(_Reading(0.5), "")
    assert agent._turn_confidence("") is None
    assert agent._turn_confidence(None) is None


def test_a_context_without_the_stamp_reads_as_none():
    agent = _conf_agent()
    agent.context.last_confidence = _Reading(0.5)      # set by an old writer
    assert agent._turn_confidence("x") is None


def _attr_chain(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def test_last_confidence_has_one_writer_and_one_reader():
    """Enumeration (R1): `context.last_confidence` is assigned only in
    `_set_last_confidence` (and the context's own __init__) and read only in
    `_turn_confidence`. An unstamped write, or a read that skips the request
    check, is the defect."""
    tree = ast.parse((SRC / "core" / "agent.py").read_text(encoding="utf-8"))
    writers, readers = set(), set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Attribute) and node.attr == "last_confidence":
                owner = _attr_chain(node)
                if isinstance(node.ctx, ast.Store):
                    writers.add((fn.name, owner))
                else:
                    readers.add((fn.name, owner))
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "getattr" and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value == "last_confidence"):
                readers.add((fn.name, "getattr"))
    inner = {w for w in writers if w[0] not in ("__init__",)}
    assert {w[0] for w in inner} == {"_set_last_confidence"}
    assert {r[0] for r in readers} <= {"_turn_confidence", "_set_last_confidence"}
    assert ("_turn_confidence", "getattr") in readers


# ══════════════════════════════════════════════════════════════════════
# L — a lone "<" is displayed
# ══════════════════════════════════════════════════════════════════════
from ghost_agent.core.agent import ThinkTagDisplayFilter


def _shown(tokens):
    flt, acc, out = ThinkTagDisplayFilter(), "", []
    for tok in tokens:
        acc += tok
        out.append(flt.feed(tok, acc))
    out.append(flt.flush())
    return "".join(out)


@pytest.mark.parametrize("tokens,expected", [
    # the three live losses
    (["But what if x", " <", " 71", "? Then x is removed"],
     "But what if x < 71? Then x is removed"),
    (["(x", " <", " 71", ")"], "(x < 71)"),
    (["rows where value", " <", " 0"], "rows where value < 0"),
    (["a", " </", " b"], "a </ b"),
    (["a", " <", " <", " b"], "a < < b"),
    (["ends with", " <"], "ends with <"),                 # flushed at the end
    # tags are still not shown
    (["<", "think", ">", "reasoning"], "reasoning"),
    (["</", "think", ">", " tail"], " tail"),
    (["<think>", "reasoning"], "reasoning"),
    (["x", "</think>", " y"], "x y"),
    (["<", "think>", "r"], "r"),
    # a whole tag after a held opener: the opener was prose
    (["a", " <", "</think>", " b"], "a < b"),
    # the earlier regression this filter fronts
    (["Let me", " think", " about it"], "Let me think about it"),
    (["5 ", ">", " 3"], "5 > 3"),
])
def test_display_filter(tokens, expected):
    assert _shown(tokens) == expected


def test_display_filter_never_touches_the_accumulated_stream():
    flt = ThinkTagDisplayFilter()
    acc = "x"
    flt.feed(" <", acc + " <")
    assert acc == "x" and flt.flush() == " <" and flt.flush() == ""


# ══════════════════════════════════════════════════════════════════════
# M — request tags
# ══════════════════════════════════════════════════════════════════════
from ghost_agent.utils.logging import _req_tag


@pytest.mark.parametrize("req_id,tag", [
    # the id shapes the code MINTS (dream.py, jobs.py + main.py, subagent.py,
    # coding_loop.py, replay_engine.py, if_bench.py, the Slack bot) — R2
    # review: the first table pinned shapes nothing produces
    ("bench-39ab12cd34", "39"), ("bench-91c0ffee00", "91"),     # were both "BE"
    ("slack-b4c319b1", "B4"), ("slack-8f0ef540", "8F"),         # were both "SL"
    ("job-job-7f3a9b1c", "7F"), ("sub-job-7f3a9b1c", "7F"),     # a job's own id
    ("sub-leaf-3fa2b1c9-a1", "3F"), ("sub-leaf-3fa2b1c9-a2", "3F"),   # not "A1"
    ("sub-leaf-0c4d5e6f-a1", "0C"),
    ("probe-012ca9ec", "01"),
    ("replay-abc123def0-control", "AB"),
    # no hex-id part: the first two characters, as before
    ("sched-task_9e107d9d372bb6826bd81d3542a419d6", "SC"),
    ("sub-chess-178032511234", "SU"),                           # the largest prefixed kind
    ("probe-imggen-1790270331", "PR"),                          # an epoch is not an id
    ("probe-ifb-20260930T101500-if042-b-r2", "PR"),
    ("kg-live", "KG"), ("kf-live", "KF"),                       # must stay apart
    ("bench-12345678", "BE"),                                   # digits only
    ("bench-ab12c", "BE"),                                      # five characters: too short
    ("bench-", "BE"), ("bench-x", "BE"),
    # plain ids are unchanged
    ("a8a93a27", "A8"), ("eb30aa56", "EB"), ("SYSTEM", "**"),
    ("dv0123456789", "DV"),                                     # the ablation driver's tag
    ("cafe-12ab34", "CA"),                                      # a hex word is not a prefix
    ("1ac2752a-dead-4eef-9abc-0123456789ab", "1A"),             # a uuid
    ("x", "X"), ("", ""),
])
def test_req_tag(req_id, tag):
    assert _req_tag(req_id) == tag


def test_two_benches_are_told_apart():
    assert _req_tag("bench-91aa00") != _req_tag("bench-49bb00")


# ══════════════════════════════════════════════════════════════════════
# N — a streamed turn's duration
# ══════════════════════════════════════════════════════════════════════
def _traj_agent(tmp_path):
    from ghost_agent.distill.collector import TrajectoryCollector
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = MagicMock()
    agent.context.trajectory_collector = TrajectoryCollector(tmp_path)
    agent.context.trajectory_task_kind = None
    agent.context.trajectory_user_request_override = None
    agent.context.trajectory_extra_static = None
    return agent


def _record(agent, **kw):
    return agent._record_turn_trajectory(
        messages=[{"role": "system", "content": "s"},
                  {"role": "user", "content": "q"},
                  {"role": "assistant", "content": "a"}],
        final_content="a", req_id="ba2753bb", model="m",
        trajectory_id="t-1", user_request="q", **kw)


def test_streamed_turn_is_stored_with_the_elapsed_it_measured(tmp_path):
    """The request clock is closed when the drain records: without the
    caller's own measurement the row said 0.0 (147 s on the wire)."""
    row = _record(_traj_agent(tmp_path), elapsed_s=147.234)
    assert row is not None and row.duration_s == pytest.approx(147.234)


def test_without_it_a_closed_request_keeps_the_default(tmp_path):
    row = _record(_traj_agent(tmp_path))
    assert row is not None and row.duration_s == 0.0


def test_a_negative_elapsed_is_not_stored(tmp_path):
    row = _record(_traj_agent(tmp_path), elapsed_s=-3.0)
    assert row.duration_s == 0.0


# ══════════════════════════════════════════════════════════════════════
# O — the router's reason
# ══════════════════════════════════════════════════════════════════════
def test_router_reason_states_the_threshold_not_high_confidence():
    from ghost_agent.router.dispatch import ComplexityDispatcher
    clf = MagicMock()
    clf.predict.return_value = ("hard", 0.41)
    d = ComplexityDispatcher(clf, confidence_threshold=0.3)
    reason = d._cleared_reason()
    assert "0.30" in reason and "high-confidence" not in reason


# ══════════════════════════════════════════════════════════════════════
# P — search: site labels, keyword reformulation, research sources
# ══════════════════════════════════════════════════════════════════════
from ghost_agent.tools.search import (
    _sanitize_query, _reformulate_query, is_search_results_url,
    select_research_sources, removed_site_operators, site_operator_note,
    RESEARCH_MAX_SOURCES)


@pytest.mark.parametrize("query,sanitized,removed", [
    ("site:blogs.lupyd.com postgres with quic", "postgres with quic",
     ["site:blogs.lupyd.com"]),                                           # live
    ("postgres quic site:lupyd.com", "postgres quic", ["site:lupyd.com"]),
    ("python asyncio tutorial -site:pinterest.com", "python asyncio tutorial",
     ["-site:pinterest.com"]),                                            # an exclusion
    ("foo inurl:docs bar", "foo bar", ["inurl:docs"]),
    ("foo site:x.com or site:y.com", "foo", ["site:x.com", "site:y.com"]),
    # only operators and a connective: nothing to mine, the query goes out as
    # typed — so nothing was removed and the note must not claim it was
    ("site:x.com or site:y.com", "site:x.com or site:y.com", []),
    # R2 review: operators only, sent as typed — "`site:x.com` was removed"
    # was printed above a search for 'site:x.com filetype:pdf'
    ("site:x.com filetype:pdf", "site:x.com filetype:pdf", []),
    ('"site:lupyd.com" postgres', "postgres", ["site:lupyd.com"]),    # no stray quote
    ('foo site:"example.com"', "foo", ["site:example.com"]),
    # the operand IS the query: it is mined for keywords, nothing to report
    ("site:reddit.com/r/lgbtgreece/comments/1voyjgf/is_nudism_safe",
     "reddit lgbtgreece 1voyjgf nudism safe", []),
    ("report filetype:pdf", "report", []),                                # not a site
    ('"exact phrase" something', "exact phrase something", []),
    ("website: the basics", "website: the basics", []),                   # a word ending in "site"
    ("plain keywords", "plain keywords", []),
    ("", "", []), (None, None, []),
])
def test_a_site_restriction_is_removed_and_reported(query, sanitized, removed):
    assert _sanitize_query(query) == sanitized
    assert removed_site_operators(query) == removed
    note = site_operator_note(query)
    assert bool(note) is bool(removed)
    for op in removed:
        assert f"`{op}`" in note
    if removed:
        assert "NOT limited to" in note and "plain keyword" in note
        assert "ran as" not in note       # it cannot know which query produced the rows


def test_the_site_is_never_turned_into_a_keyword():
    """R1 review: the first fix appended the host's name. It leaked the
    excluded site, IP octets and dictionary-word labels into the query."""
    for q in ("x -site:pinterest.com", "x site:192.168.1.1", "x site:open.spotify.com",
              "x site:bugs.launchpad.canonical.com"):
        assert _sanitize_query(q) == "x"


def test_the_site_note_is_not_an_error_line():
    """Found live (2026-09-30): the first wording said the engines "cannot"
    restrict by site, and `strikes.error_line` read every such search
    result as a failure — the evidence digest listed the note under
    "Distinct errors hit"."""
    from ghost_agent.core.strikes import error_line
    for q in ("site:blogs.lupyd.com postgres with quic", "x -site:a.com site:b.com"):
        assert error_line(site_operator_note(q)) == "", site_operator_note(q)
        assert error_line("### 1. r\n[Source: https://a/]\n\n" + site_operator_note(q)) == ""


async def test_web_search_tells_the_model_its_site_filter_was_dropped(monkeypatch):
    from ghost_agent.tools import search as search_mod
    asked = []

    async def fake_ddgs(query, tor_proxy):
        asked.append(query)
        return "### 1. quic-go\nA QUIC implementation in Go\n[Source: https://github.com/quic-go/quic-go]"

    monkeypatch.setattr(search_mod, "tool_search_ddgs", fake_ddgs)
    out = await search_mod.tool_search("site:blogs.lupyd.com postgres with quic")
    assert "quic-go" in out
    assert "`site:blogs.lupyd.com` was removed" in out and "NOT limited to" in out
    plain = await search_mod.tool_search("postgres with quic")
    assert "[Note:" not in plain


async def test_the_note_keeps_a_failed_searchs_status(monkeypatch):
    from ghost_agent.tools import search as search_mod
    from ghost_agent.tools.outcome import OutcomeStatus

    async def failed(query, tor_proxy):
        return ToolOutcome.failed("ERROR: no results", world_changed=False,
                                  reason_code="search_empty")

    monkeypatch.setattr(search_mod, "tool_search_ddgs", failed)
    out = await search_mod.tool_search("site:lupyd.com postgres quic")
    assert out.status == OutcomeStatus.FAILED and "[Note:" in out


def test_a_keyword_query_is_still_retried_as_a_question():
    """Two mechanical alternatives were built and removed (R1/R2 reviews):
    each lost the query's subject on real queries."""
    out = _reformulate_query("lupyd postgres quic blog post")
    assert out[0] == "how to lupyd postgres quic blog post"


@pytest.mark.parametrize("url,expected", [
    ("https://www.clipzui.cc/?q=turkey+vs+greece", True),         # live
    ("https://videomon.biz/?q=turkiye+vs+greece", True),          # live
    ("https://x.com/search?q=a", True),
    ("https://x.com/Search?Q=a", True),                           # keys are case-folded
    ("https://docs.python.org/3/search.html?q=x", True),
    ("https://site.org/results.php?query=a", True),
    ("https://site.org/find/?keyword=a", True),
    ("https://www.youtube.com/results?search_query=x", True),
    ("https://site.org/?s=greek+army", True),                     # WordPress search
    ("https://site.org/?k=greek+army", True),
    ("https://www.idcommunism.com/search/label/Peletidis+Kostas", True),   # a label listing
    # real URLs from the trajectory store (R2 review): a query on a NON-root path
    ("https://www.avito.ru/moskva/zapchasti_i_aksessuary?q=shoei+x+spr+pro", True),
    ("https://sourceforge.net/directory/?q=c+++dos+decompiler", True),
    ("https://www.findamasters.com/masters-degrees/?Keywords=international+security&PG=5", True),
    ("https://www.ebay.com/shop/schuberth-sc2?_nkw=schuberth+sc2", True),
    ("https://www.shutterstock.com/search/greek+flag", True),     # a query in the PATH
    ("https://www.google.com/maps/search/Ain+Khaled,+Doha,+Qatar", True),
    ("https://site.org/search/two%20words", True),
    # documents
    ("https://www.youtube.com/watch?v=O2QPjo-RD2M", False),
    ("https://www.globalfirepower.com/countries-comparison.php", False),
    ("https://site.org/article?q=1", False),                      # a number is not a query
    ("https://site.org/?q=node/14773", False),                    # a Drupal document address
    ("https://webcache.googleusercontent.com/search?q=cache:zoopla.co.uk/for-sale/details/73576645", False),
    ("https://site.org/results?s=2026", False),                   # a season, not a search
    ("https://site.org/docs/?s=install", False),                  # `s` means search only at the root
    ("https://www.biblegateway.com/passage/?search=Genesis+17&version=KJV", False),
    # a `search` path segment alone is not a results page (real URLs)
    ("https://developers.google.com/search/docs/fundamentals/seo-starter-guide", False),
    ("https://www.postgraduatesearch.com/courses/search/postgraduate/open-university/ma-international-relations-and-security/1006681", False),
    ("https://www.greek-language.gr/greekLang/modern_greek/tools/lexica/triantafyllides/search.html?lq=νέγρος", False),
    ("https://askai.glarity.app/search/Where-can-I-find-the-Smithing-Stone", False),
    ("https://en.wikipedia.org/wiki/Search", False),
    ("https://a.org/research/paper", False),
    ("https://site.org/?q=", False),                              # an empty query
    ("https://site.org/?id=7", False),
    ("https://site.org/", False), ("", False), (None, False),
])
def test_is_search_results_url(url, expected):
    assert is_search_results_url(url) is expected


_Q = "greece vs turkey military power comparison 2026 who would win conflict analysis"


def _r(url, title=""):
    return {"href": url, "title": title, "body": ""}


_LIVE = [
    _r("https://www.youtube.com/watch?v=O2QPjo-RD2M", "Turkey vs Greece military power"),
    _r("https://www.clipzui.cc/?q=turkey+vs+greece", "turkey vs greece"),
    _r("https://sonhaberkibris.com/turkey-vs-greece-military-power", "Turkey vs Greece"),
    _r("https://dishcuss.com/explore/greece-vs-turkey", "greece vs turkey"),
    _r("https://videomon.biz/?q=turkiye+vs+greece", "turkiye vs greece"),
    _r("https://www.reddit.com/r/greece/comments/rfh0d8/x", "r/greece"),
    _r("https://www.globalfirepower.com/countries-comparison.php", "Greece and Turkiye"),
    _r("https://remp3indir.net/turkiye-vs-israil-askeri-guc", "Türkiye vs İsrail"),
]


def test_research_does_not_read_search_results_pages():
    urls = [r["href"] for r in select_research_sources(_LIVE)]
    assert not any("?q=" in u for u in urls)
    assert len(urls) == 6 and urls[0].startswith("https://www.youtube.com")


def test_research_drops_nothing_for_being_off_topic():
    """R1 review: a second rule left off-topic results unread — and dropped
    both FIA regulation PDFs for an F1 query, whose titles lacked the
    leading word. The wave already ranks; nothing else is removed."""
    batch = [_r(f"https://blog{i}.example/f1-engine-{i}", "engine talk") for i in range(4)]
    batch += [_r("https://www.fia.com/sites/default/files/2026_pu_technical_regulations.pdf",
                 "2026 Power Unit Technical Regulations"),
              _r("https://remp3indir.net/turkiye-vs-israil", "Türkiye vs İsrail")]
    assert select_research_sources(batch) == batch


def test_a_skipped_search_page_gives_its_slot_to_the_next_result():
    """The limit is applied AFTER the search pages are skipped."""
    batch = [_r(f"https://s.org/?q=greece{i}") for i in range(3)]
    batch += [_r(f"https://doc{i}.org/greece") for i in range(10)]
    got = select_research_sources(batch)
    assert len(got) == RESEARCH_MAX_SOURCES == 8
    assert all("doc" in r["href"] for r in got)


def test_a_result_keyed_by_url_is_read_the_same():
    """Result rows carry `href` (ddgs) or `url` — the fetch list reads both."""
    batch = [{"url": "https://s.org/?q=greece", "title": "search"},
             {"url": "https://doc.org/greece", "title": "doc"}]
    assert select_research_sources(batch) == batch[1:]


def test_a_batch_of_only_search_pages_is_not_emptied():
    batch = [_r("https://a.org/?q=greece"), _r("https://b.org/?q=greece")]
    assert select_research_sources(batch) == batch


def test_research_source_limit_and_junk_rows():
    many = [_r(f"https://s{i}.org/greece-{i}", "greece") for i in range(20)]
    assert select_research_sources(many) == many[:8]
    assert select_research_sources(many, limit=3) == many[:3]
    assert select_research_sources([None, "x", {}]) == [{}]
    assert select_research_sources([]) == []


# ══════════════════════════════════════════════════════════════════════
# Q — model-facing text for a channel member
# ══════════════════════════════════════════════════════════════════════
_CONV = [
    {"role": "system", "content": "s"},
    {"role": "user", "content": "if there was a war, who would win?"},
    {"role": "assistant", "content": "analysis"},
    {"role": "user", "content": "interesting if i asked you to simulate such war, how would you do it?"},
    {"role": "assistant", "content": "Here's how I'd design the simulation: phases, data, engine. Shall I build it?"},
    {"role": "user", "content": "yes proceed"},
]


def test_member_notice_and_steer_carry_the_new_rules():
    """Model-facing text: the sentences a member's turn is now given."""
    notice = agent_mod._MEMBER_CAPABILITY_NOTICE
    assert "do not mention tools or these limits" in notice
    assert "can be answered without any tool" in notice    # not "none of those tools"
    assert "you have NOT read it" in notice
    assert "pasted" in notice                   # pasted text is the exception
    assert notice in agent_mod._MEMBER_PROFILE_PLACEHOLDER   # it is what the turn receives
    steer = agent_mod.member_search_yield_steer(10)
    assert "not the content of a page the user linked" in steer
    assert "10 web searches" in steer


# ══════════════════════════════════════════════════════════════════════
# I — macro-mint skips are said once
# ══════════════════════════════════════════════════════════════════════
def test_macro_mint_skip_is_news_once():
    from ghost_agent.core.agent import macro_mint_skip_is_new
    ctx = type("C", (), {})()
    assert macro_mint_skip_is_new(ctx, "auto.generic.a", "no slots") is True
    assert macro_mint_skip_is_new(ctx, "auto.generic.a", "no slots") is False
    assert macro_mint_skip_is_new(ctx, "auto.generic.a", "meta tool") is True   # new reason
    assert macro_mint_skip_is_new(ctx, "auto.generic.b", "no slots") is True
    other = type("C", (), {})()
    assert macro_mint_skip_is_new(other, "auto.generic.a", "no slots") is True  # per process/context


def test_macro_mint_skip_memory_is_bounded():
    from ghost_agent.core.agent import macro_mint_skip_is_new
    ctx = type("C", (), {})()
    for i in range(600):
        macro_mint_skip_is_new(ctx, f"n{i}", "w")
    assert len(ctx._macro_mint_skips_seen) == 512
    assert macro_mint_skip_is_new(ctx, "n599", "w") is False
    assert macro_mint_skip_is_new(ctx, "n0", "w") is True      # evicted → news again


def test_macro_mint_skip_line_has_one_emitter(monkeypatch):
    from ghost_agent.core.agent import report_macro_mint_skip
    seen = []
    monkeypatch.setattr(agent_mod, "pretty_log",
                        lambda title, content=None, **kw: seen.append((title, content)))
    ctx = type("C", (), {})()
    assert report_macro_mint_skip(ctx, "auto.generic.a", ("file_system", "execute"), "no slots") is True
    assert report_macro_mint_skip(ctx, "auto.generic.a", ("file_system", "execute"), "no slots") is False
    assert seen == [("Macro Mint",
                     "skipped auto.generic.a (file_system → execute): no slots")]
    # Enumeration (R1): nothing else in the turn loop prints that title —
    # a second emitter is how sixteen lines a cycle came back.
    tree = ast.parse((SRC / "core" / "agent.py").read_text(encoding="utf-8"))
    owners = set()
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for call in ast.walk(fn):
                if (isinstance(call, ast.Call) and call.args
                        and isinstance(call.args[0], ast.Constant)
                        and call.args[0].value == "Macro Mint"):
                    owners.add(fn.name)
    assert owners == {"report_macro_mint_skip"}


# ══════════════════════════════════════════════════════════════════════
# Call sites — the fixes above are reached from the real paths
# ══════════════════════════════════════════════════════════════════════
import threading


async def test_dream_saves_heuristics_with_their_window_as_evidence():
    """H at the call site: the REM heuristic save names the fragment window
    and asks for half of it to be fresh."""
    ctx = MagicMock()
    ctx.memory_system = MagicMock()
    ctx.memory_system.collection = MagicMock()
    ctx.skill_memory = MagicMock()
    ctx.skill_memory._get_lock = lambda: threading.RLock()
    ctx.skill_memory.file_path = MagicMock()
    ctx.skill_memory.file_path.read_text.return_value = "[]"
    ctx.llm_client = MagicMock()
    ctx._last_dream_fragment_ids = None
    dreamer = Dreamer(ctx)
    ids = [f"id{i}" for i in range(5)]
    dreamer.memory.collection.get.return_value = {
        "ids": ids, "documents": [f"auto memory number {i}" for i in range(5)],
        "metadatas": [{"type": "auto"}] * 5, "embeddings": [[0.1]] * 5}
    ctx.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": json.dumps({
            "consolidations": [],
            "heuristics": ["Always name the concrete threat before recommending a chess move."],
        })}}]})
    ctx.skill_memory.learn_lesson = MagicMock()
    await dreamer.dream()
    call = [c for c in ctx.skill_memory.learn_lesson.call_args_list
            if c.kwargs.get("source") == "dream"][0]
    assert call.kwargs["evidence_refs"] == ids
    assert call.kwargs["evidence_min_new_fraction"] == DREAM_EVIDENCE_MIN_NEW_FRACTION == 0.5


def test_router_decision_reason_is_not_high_confidence():
    """O at the call site: a cleared decision's reason."""
    from ghost_agent.router.dispatch import ComplexityDispatcher
    clf = MagicMock()
    clf.weights_ = [1.0]
    clf.uses_embeddings_ = False
    for label, conf in (("hard", 0.41), ("easy", 0.95)):
        clf.predict.return_value = (label, conf)
        d = ComplexityDispatcher(clf, confidence_threshold=0.3)
        dec = d.route("### SYNTHETIC TRAINING EXERCISE solve it")
        assert dec.label == label and dec.escalated is False
        assert "high-confidence" not in dec.reason
        assert "0.30" in dec.reason and "threshold" in dec.reason


async def test_deep_research_fetches_only_the_selected_sources(monkeypatch):
    """P at the call site: the search-results pages are not fetched."""
    from ghost_agent.tools import search as search_mod
    monkeypatch.setattr(search_mod.importlib.util, "find_spec", lambda name: True)
    monkeypatch.setattr(search_mod, "_race_search_wave",
                        AsyncMock(return_value=list(_LIVE)))
    fetched = []

    async def fake_fetch(url, **kw):
        fetched.append(url)
        return "A page body about the Greek and Turkish armed forces."

    monkeypatch.setattr(search_mod, "helper_fetch_url_content", fake_fetch)
    out = await search_mod.tool_deep_research(
        _Q, False, None, llm_client=None, max_context=8192)
    assert fetched, "nothing was fetched — the pin is blind"
    assert not any("?q=" in u for u in fetched)
    assert sorted(set(fetched)) == sorted(
        r["href"] for r in select_research_sources(_LIVE))


async def test_the_verifier_is_not_handed_a_guessed_request(monkeypatch):
    """Q at the call site (R1 review): `_compute_verifier_verdict` judges a
    go-ahead against the user's message — no earlier user message is added
    as "the request being confirmed"."""
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = MagicMock()
    agent.available_tools = {}
    agent._is_strict_trivial_chat = lambda lc: False
    agent._active_constraint_note = lambda **kw: ""
    verifier = MagicMock()
    verifier.llm_client = MagicMock()
    verifier.verify_claim = AsyncMock(return_value=VerifyResult(
        verdict=VerifyVerdict.CONFIRMED, confidence=0.9, reasoning="ok"))
    verifier.verify_code = AsyncMock(return_value=VerifyResult(
        verdict=VerifyVerdict.CONFIRMED, confidence=0.9, reasoning="ok"))
    agent.context.verifier = verifier
    agent._execute_web_artifact = AsyncMock(return_value=None)
    tools = [{"role": "tool", "name": "browser",
              "content": "--- BROWSER RESULT ---\nSTATUS: OK\nOP: navigate\n"
                         "URL: http://127.0.0.1:8100/\nHTTP_STATUS: 200\n"
                         "TITLE: Greece vs Turkey — War Simulation\n"}]
    await agent._compute_verifier_verdict(
        tools_run_this_turn=tools, messages=list(_CONV),
        final_ai_content="The simulation is live and working.",
        last_user_content="yes proceed", lc="yes proceed")
    calls = verifier.verify_claim.await_args_list + verifier.verify_code.await_args_list
    assert calls, "no verdict call was made — the pin is blind"
    views = [str(c.kwargs.get("context") or c.kwargs.get("intent") or "")
             for c in calls]
    assert any("yes proceed" in v for v in views), views
    assert not any("simulate such war" in v or "who would win" in v for v in views), views


from tests.test_finalize_stream_pins import make_stream_agent, sse, _make_stream_state
from ghost_agent.core.agent import StreamState


async def _drain(agent, deltas, **state_overrides):
    async def final_stream(payload, use_coding=False):
        for d in deltas:
            yield sse({"content": d})
        yield b"data: [DONE]\n\n"
    agent.context.llm_client.stream_chat_completion = final_stream
    reg = MagicMock()
    reg.is_cancelled.return_value = False
    ss = _make_stream_state(reg)
    if state_overrides:
        ss = StreamState(**{**ss.__dict__, **state_overrides})
    gen, _, _ = agent._stream_final_generation(ss)
    return [c async for c in gen]


async def test_a_streamed_turn_records_its_elapsed_and_logs_its_reply(monkeypatch):
    """N at the call site: the drain passes the wall-clock it measured and
    prints the reply — the request clock is closed by then."""
    a = make_stream_agent()
    a._record_calibration_safe = AsyncMock()
    a._record_turn_trajectory = MagicMock(return_value=None)
    logged = []
    real = agent_mod.pretty_log
    monkeypatch.setattr(
        agent_mod, "pretty_log",
        lambda title, content=None, **kw: (logged.append((title, content)),
                                           real(title, content, **kw))[1])
    await _drain(a, ["The streamed ", "answer."])
    kw = a._record_turn_trajectory.call_args.kwargs
    assert isinstance(kw.get("elapsed_s"), float) and kw["elapsed_s"] >= 0.0
    assert ("Final Reply", "The streamed answer.") in logged


async def test_displayed_thinking_keeps_a_lone_less_than(monkeypatch):
    """L at the call site: through the real internal stream loop."""
    from ghost_agent.core.agent import GhostContext
    context = MagicMock(spec=GhostContext)
    context.args = MagicMock()
    context.args.use_planning = False
    context.args.smart_memory = 0.0
    context.args.max_context = 4000
    context.args.temperature = 0.7
    context.llm_client = MagicMock()
    context.sandbox_dir = None
    context.memory_system = None
    context.profile_memory = None
    context.semantic_memory = None
    context.skill_memory = None
    context.journal = None
    context.scratchpad = MagicMock()
    context.scratchpad.list_all.return_value = ""
    agent = GhostAgent(context=context)

    async def stream(*a, **k):
        for c in (["reasoning_content", "But what if x"], ["reasoning_content", " <"],
                  ["reasoning_content", " 71"], ["reasoning_content", "? Then x is removed."],
                  ["reasoning_content", "\n\n"], ["reasoning_content", "So x must be 98."],
                  ["content", "The answer is 98."]):
            yield ("data: " + json.dumps({"choices": [{"delta": {c[0]: c[1]}}]})
                   + "\n\n").encode()
        yield b"data: [DONE]\n\n"

    context.llm_client.stream_chat_completion = stream
    thoughts = []
    real = agent_mod.pretty_log

    def capture(title, content=None, **kw):
        if title == "thinking":
            thoughts.append(str(content))
        return real(title, content, **kw)

    monkeypatch.setattr(agent_mod, "pretty_log", capture)
    with patch("ghost_agent.core.agent.get_active_tool_definitions", return_value=[]):
        await agent.handle_chat(
            {"messages": [{"role": "user", "content": "What score does Brinley need?"}],
             "stream": False},
            background_tasks=MagicMock(), request_id="test-4ks-lt")
    joined = "\n".join(thoughts)
    assert "x < 71" in joined, thoughts
    assert "x 71" not in joined


# ══════════════════════════════════════════════════════════════════════
# Call sites, through a real turn (R1 review: each of these survived with
# the helper fixed and the call site broken)
# ══════════════════════════════════════════════════════════════════════
import types


class _Bg:
    def add_task(self, *a, **k):
        pass


def _chat_context(tmp_path):
    from ghost_agent.core.agent import GhostContext
    ctx = MagicMock(spec=GhostContext)
    ctx.llm_client = MagicMock()
    ctx.llm_client.vision_clients = None
    ctx.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "All done — the file is written.",
                                 "tool_calls": []}}]})
    ctx.sandbox_dir = str(tmp_path)
    ctx.args = MagicMock()
    ctx.args.shell = "bash"
    ctx.args.max_context = 8000
    ctx.args.temperature = 0.5
    ctx.args.smart_memory = 0.0
    ctx.args.use_planning = False
    ctx.args.model = "test-model"
    ctx.args.perfect_it = False
    ctx.args.no_verifier = False
    ctx.profile_memory = MagicMock()
    ctx.profile_memory.get_context_string.return_value = ""
    ctx.memory_system = None
    ctx.skill_memory = None
    ctx.scratchpad = MagicMock()
    ctx.scratchpad.list_all.return_value = ""
    ctx.memory_dir = tmp_path
    ctx.trajectory_collector = None
    ctx.last_confidence = None
    ctx.last_confidence_req = ""
    ctx.verifier = MagicMock()               # attached, with a model behind it
    ctx.verifier.llm_client = MagicMock()
    return ctx


async def _turn(ctx, *, verdict, req_id="req4ks01", role=None, env=None):
    """One real non-streamed turn; returns every (title, content) logged."""
    logged, levels = [], {}
    agent = GhostAgent(ctx)
    agent.thinking_budget_override = "selfplay"
    agent._logged_levels = levels              # content → the level it was logged at
    body = {"messages": [{"role": "user", "content": "Write the report file."}]}
    tool = {"name": "execute_python", "content": "EXIT CODE: 0\nok"}

    def _log(title, content=None, **kw):
        logged.append((title, str(content)))
        levels[str(content)] = kw.get("level", "INFO")

    with patch("ghost_agent.core.agent.pretty_log", _log), \
         patch("ghost_agent.core.agent.get_active_tool_definitions", return_value=[]), \
         patch("ghost_agent.core.agent._find_substantive_tool_for_verifier",
               return_value=tool), \
         patch.object(GhostAgent, "_compute_verifier_verdict", side_effect=verdict):
        await agent.handle_chat(body, _Bg(), request_id=req_id,
                                **({"requester_role": role} if role else {}))
        await asyncio.sleep(0.4)          # let an attached late task finish
    return agent, logged


def _verifier_lines(logged):
    return [c for t, c in logged if t == "Verifier"]


async def test_finalize_says_deferred_only_for_a_verdict_in_flight(tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    monkeypatch.setenv("GHOST_CRITIC_REPAIR_BUDGET", "0.05")

    async def slow(*a, **kw):
        await asyncio.sleep(0.25)
        return (None, {"name": "execute_python"})

    agent, logged = await _turn(_chat_context(tmp_path), verdict=slow)
    lines = _verifier_lines(logged)
    assert any("verdict deferred" in l for l in lines), lines
    assert not any("came back empty" in l for l in lines)


async def test_finalize_does_not_say_deferred_when_nothing_is_running(tmp_path, monkeypatch):
    """The live lie, at the line that printed it."""
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")

    async def empty(*a, **kw):
        return (None, {"name": "execute_python"})

    agent, logged = await _turn(_chat_context(tmp_path), verdict=empty)
    lines = _verifier_lines(logged)
    assert any("came back empty" in l and "nothing is running late" in l for l in lines), lines
    assert not any("verdict deferred" in l for l in lines)
    empty_line = [l for l in lines if "came back empty" in l][0]
    assert agent._logged_levels[empty_line] == "WARNING"     # nothing will land: say so loudly


async def test_finalize_names_a_member_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")

    async def empty(*a, **kw):
        return (None, {"name": "execute_python"})

    agent, logged = await _turn(_chat_context(tmp_path), verdict=empty, role="member")
    lines = _verifier_lines(logged)
    assert any("member turn" in l for l in lines), lines
    assert not any("verdict deferred" in l for l in lines)
    assert agent._logged_levels[[l for l in lines if "member turn" in l][0]] == "INFO"


def _with_metacog(ctx, composite):
    reading = MagicMock()
    reading.composite = composite
    reading.below_threshold = False
    ctx.metacog = MagicMock()
    ctx.metacog.confidence.score.return_value = reading
    ctx.metacog.competence.predict.return_value = (0.9, 100)
    ctx.calibration_tracker = MagicMock()
    ctx._calib_pending = None
    return ctx


async def _confirmed(*a, **kw):
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    return (VerifyResult(verdict=VerifyVerdict.CONFIRMED, confidence=0.9,
                         reasoning="ok"), {"name": "execute_python"})


async def test_the_turn_outcome_line_prints_the_confidence_this_turn_computed(tmp_path):
    """K at the call sites: the reading is stamped with THIS request and the
    line reads it back with THIS request."""
    ctx = _with_metacog(_chat_context(tmp_path), 0.42)
    agent, logged = await _turn(ctx, verdict=_confirmed, req_id="req4ks-k1")
    outcome = [c for t, c in logged if t == "Turn Outcome"]
    assert outcome and "confidence 0.42" in outcome[0], logged
    assert ctx.last_confidence_req == "req4ks-k1"


async def test_the_turn_outcome_line_omits_another_requests_confidence(tmp_path):
    """The live leak: a turn that computes no reading printed the previous
    request's."""
    ctx = _chat_context(tmp_path)
    stale = MagicMock()
    stale.composite = 0.78
    ctx.last_confidence = stale
    ctx.last_confidence_req = "43199788"
    ctx.metacog = None                        # this turn computes nothing
    agent, logged = await _turn(ctx, verdict=_confirmed, req_id="slack-b4c319b1")
    outcome = [c for t, c in logged if t == "Turn Outcome"]
    assert outcome and "confidence" not in outcome[0], outcome


async def _thinking_shown(monkeypatch, deltas):
    """Run one real non-streamed turn over the given (channel, token) deltas
    and return the displayed thinking, joined."""
    from ghost_agent.core.agent import GhostContext
    context = MagicMock(spec=GhostContext)
    context.args = MagicMock()
    context.args.use_planning = False
    context.args.smart_memory = 0.0
    context.args.max_context = 4000
    context.args.temperature = 0.7
    context.llm_client = MagicMock()
    context.sandbox_dir = None
    context.memory_system = None
    context.profile_memory = None
    context.semantic_memory = None
    context.skill_memory = None
    context.journal = None
    context.scratchpad = MagicMock()
    context.scratchpad.list_all.return_value = ""
    agent = GhostAgent(context=context)

    async def stream(*a, **k):
        for channel, token in deltas:
            yield ("data: " + json.dumps({"choices": [{"delta": {channel: token}}]})
                   + "\n\n").encode()
        yield b"data: [DONE]\n\n"

    context.llm_client.stream_chat_completion = stream
    thoughts = []
    real = agent_mod.pretty_log

    def capture(title, content=None, **kw):
        if title == "thinking":
            thoughts.append(str(content))
        return real(title, content, **kw)

    monkeypatch.setattr(agent_mod, "pretty_log", capture)
    with patch("ghost_agent.core.agent.get_active_tool_definitions", return_value=[]):
        await agent.handle_chat(
            {"messages": [{"role": "user", "content": "What score does Brinley need?"}],
             "stream": False},
            background_tasks=MagicMock(), request_id="test-4ks-think")
    return "\n".join(thoughts)


R, C_ = "reasoning_content", "content"


async def test_an_inline_think_model_keeps_its_less_than_too(monkeypatch):
    """The CONTENT display site (a model with no reasoning channel)."""
    shown = await _thinking_shown(monkeypatch, [
        (C_, "Since x"), (C_, " <"), (C_, " 5"), (C_, " the answer is 4.")])
    assert "x < 5" in shown and "x 5" not in shown


async def test_an_opener_held_when_the_reasoning_ends_is_shown(monkeypatch):
    shown = await _thinking_shown(monkeypatch, [
        (R, "The bound is"), (R, " <"), (C_, "The answer is 4.")])
    assert shown.rstrip().endswith("The bound is <")


async def test_prose_before_a_whole_close_tag_keeps_its_opener(monkeypatch):
    """" <" then "</think>": the display stops at the tag; the held "<" was
    prose and is shown, the text after the tag is not."""
    shown = await _thinking_shown(monkeypatch, [
        (R, "so a"), (R, " <"), (R, "</think>"), (R, "hidden tail"),
        (C_, "The answer is 4.")])
    assert shown.rstrip().endswith("so a <") and "hidden" not in shown


async def test_a_fragmented_close_tag_shows_no_stray_opener(monkeypatch):
    """"</" then "think>": the held opener IS the tag."""
    shown = await _thinking_shown(monkeypatch, [
        (R, "so a is small"), (R, "</"), (R, "think>"), (R, "hidden tail"),
        (C_, "The answer is 4.")])
    assert "<" not in shown and "hidden" not in shown and "so a is small" in shown


async def test_a_streamed_turn_is_stored_with_its_true_wall_clock(monkeypatch):
    """N, the value: 100 s had passed on the request clock when the stream
    began — the row must say at least that, not 0.0 and not the drain only."""
    a = make_stream_agent()
    a._record_calibration_safe = AsyncMock()
    a._record_turn_trajectory = MagicMock(return_value=None)
    monkeypatch.setattr(agent_mod._glog, "request_elapsed_s", lambda rid: 100.0)
    await _drain(a, ["The streamed ", "answer."])
    elapsed = a._record_turn_trajectory.call_args.kwargs["elapsed_s"]
    assert 100.0 <= elapsed < 105.0


# ══════════════════════════════════════════════════════════════════════
# C at the spawn sites — a member's turn gets no verdict task (R2 review)
# ══════════════════════════════════════════════════════════════════════
async def _empty_verdict(*a, **kw):
    return (None, {"name": "execute_python"})


async def test_a_member_turn_spawns_no_late_verdict(tmp_path, monkeypatch):
    """With a zero repair budget the gated call handed the (instantly
    empty) member verdict to the late handler: the finalize line said
    "nothing will land late" and a late WARNING followed it."""
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    monkeypatch.setenv("GHOST_CRITIC_REPAIR_BUDGET", "0")
    # the same turn by the owner DOES hand a task to the late handler —
    # otherwise the member assertions below would pass on any code
    owner, _ = await _turn(_chat_context(tmp_path), verdict=_empty_verdict)
    assert dict(owner._late_verdict_ended()), "no task was attached — the pin is blind"

    agent, logged = await _turn(_chat_context(tmp_path), verdict=_empty_verdict,
                                role="member", req_id="req4ks-m1")
    lines = _verifier_lines(logged)
    assert any("member turn" in l for l in lines), lines
    assert not any("LATE verdict" in l for l in lines), lines
    assert dict(agent._late_verdict_ended()) == {}
    assert agent._late_verdict_running() == {}


async def _streamed_gate_lines(monkeypatch, role):
    a = make_stream_agent()
    a._record_calibration_safe = AsyncMock()
    a.context.verifier = MagicMock()
    a.context.verifier.llm_client = MagicMock()
    a.context.args.no_verifier = False
    a._compute_verifier_verdict = AsyncMock(return_value=(None, None))
    a._attach_late_verdict_handler = MagicMock()
    logged = []
    monkeypatch.setattr(agent_mod, "pretty_log",
                        lambda title, content=None, **kw: logged.append((title, str(content))))
    # the drain sets the role itself, from the stream's captured state
    await _drain(a, ["The streamed ", "answer."], requester_role=role,
                 stream_tools_snapshot=[{"role": "tool", "name": "execute",
                                         "content": "EXIT CODE: 0\nok"}])
    await asyncio.sleep(0)
    return a, [c for t, c in logged if t == "Verifier" and "stream gate" in c]


async def test_the_stream_gate_does_not_defer_a_member_verdict(monkeypatch):
    owner, lines = await _streamed_gate_lines(monkeypatch, "owner")
    assert any("verdict deferred" in l for l in lines), lines     # the pin is not blind
    assert owner._attach_late_verdict_handler.call_count == 1

    member, lines = await _streamed_gate_lines(monkeypatch, "member")
    assert any("member turn" in l for l in lines), lines
    assert not any("verdict deferred" in l for l in lines)
    assert member._attach_late_verdict_handler.call_count == 0
    assert member._compute_verifier_verdict.call_count == 0


@pytest.mark.parametrize("budget", [None, "0"])
async def test_a_member_verdict_is_never_computed_in_a_spawned_task(
        tmp_path, monkeypatch, budget):
    """R3 review: at the DEFAULT repair budget a member's tool turn goes
    through the in-loop repair site, which spawned a verdict task for it."""
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    if budget is not None:
        monkeypatch.setenv("GHOST_CRITIC_REPAIR_BUDGET", budget)
    tasks = []

    async def verdict(*a, **kw):
        tasks.append(asyncio.current_task())
        return (None, {"name": "execute_python"})

    me = asyncio.current_task()
    await _turn(_chat_context(tmp_path), verdict=verdict)             # the owner
    assert tasks and any(t is not me for t in tasks), \
        "the owner's verdict ran inline too — the pin is blind"
    tasks.clear()
    agent, logged = await _turn(_chat_context(tmp_path), verdict=verdict,
                                role="member", req_id="req4ks-m2")
    assert all(t is me for t in tasks), "a verdict task was spawned for a member"
    assert not any("LATE verdict" in l for l in _verifier_lines(logged))
    assert agent._late_verdict_running() == {} and dict(agent._late_verdict_ended()) == {}


# ══════════════════════════════════════════════════════════════════════
# R3 — an outage banner is at the head of the reply; the restart log line
# ══════════════════════════════════════════════════════════════════════
@_selfplay
@pytest.mark.parametrize("final,is_outage", [
    ("CRITICAL: Upstream error 503: no healthy upstream", True),
    ("⚠️ **Correction to my previous answer:** x\n\n---\n\n"
     "CRITICAL: The upstream LLM server is unreachable.", True),
    ("Found 3 events.\nCRITICAL: disk full (x3)", False),
])
async def test_the_solve_loop_charges_an_outage_to_no_one(
        mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch, final, is_outage):
    """An outage aborts the attempt and is NOT the agent's failure; a solver
    reply that merely quotes a "CRITICAL:" log line is an answer and is
    graded."""
    titles = []
    real = dream_mod.pretty_log
    monkeypatch.setattr(
        dream_mod, "pretty_log",
        lambda title, content=None, **kw: (titles.append(title),
                                           real(title, content, **kw))[1])
    ctx, dreamer, out, calls, _ = await _run_injected(
        tmp_path, monkeypatch, mock_agent_cls, mock_sandbox_cls,
        validator=((0, "ok"), (0, "ok")), finals=(final,) * 4)
    status = str(dreamer.last_self_play_status or "")
    assert ("Self-Play Infra" in titles) is is_outage
    assert ("INFRA_ABORT" in status) is is_outage, status
    if not is_outage:
        assert "SUCCESS" in status, status
