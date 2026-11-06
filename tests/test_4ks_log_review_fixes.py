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
* F, G  (self-play: template fallback, retry temperature, malformed data;
     lesson verification) — removed with synthetic self-play in §4NM;
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
§4NM: the self-play blocks (F, G, R3) are gone with the subsystem.
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
from ghost_agent.tools.outcome import ToolOutcome

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
    # (§4KT: a service report is CONTENT — its words are not scanned at all;
    # the wording is kept plain regardless.)
    assert error_line(text, tool="manage_services") == "", error_line(text)


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

# §4NM: the bench banks are gone; the request shape they produced is inlined
_GSM_CHALLENGE = ("Janet had 22 green pens and 10 yellow pens. Then she bought "
                  "6 bags of blue pens and 2 bags of red pens. There were 9 pens "
                  "in each bag of blue and 6 pens in each bag of red. How many "
                  "pens does Janet have now?")
_GSM_REQ = ("### SYNTHETIC TRAINING EXERCISE\nSolve this challenge efficiently. "
            "Your DELIVERABLE is your final REPLY — you may use the `execute` "
            "tool to check arithmetic, but no file you write counts as an "
            "answer. Your final reply must follow the answer format the "
            "challenge specifies (the final numeric answer on its own last "
            "line).\n\n" + _GSM_CHALLENGE)
_STEPS = ("Janet starts with 22 + 10 = 32 pens.\nBlue: 6 × 9 = 54. "
          "Red: 2 × 6 = 12.\nTotal: 32 + 54 + 12 = 98.\n\n")


def _number_kinds(request):
    return [(c.kind, c.value) for c in mechanical_constraints(request)
            if c.kind == "number_only"]


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


# ══════════════════════════════════════════════════════════════════════
# H — a frequency bump needs new evidence
# ══════════════════════════════════════════════════════════════════════


def _rule(sm, **kw):
    kw.setdefault("source", "learn_skill")      # §4NM: the one live producer
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
    kw.setdefault("source", "learn_skill")      # §4NM: the one live producer
    return sm.learn_lesson(
        "Parsing CSV exports", "reading rows by position",
        "Always read CSV columns by header name, never by index.",
        memory_system=vec, trigger="Parsing CSV exports", **kw)


# ══════════════════════════════════════════════════════════════════════
# K — the Turn Outcome line printed THIS request's confidence (§4PJ: the
# score left the line; the omission pin below stays)
# ══════════════════════════════════════════════════════════════════════
def _attr_chain(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


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
        note = site_operator_note(q)
        assert error_line("### 1. r\n[Source: https://a/]\n\n" + note, tool="web_search") == ""


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
    # §4NJ: a video page is not read either (0 facts from 27 since 09-01)
    assert len(urls) == 5 and not any("youtube.com" in u for u in urls)


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
# Call sites — the fixes above are reached from the real paths
# ══════════════════════════════════════════════════════════════════════
import threading


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
    context.args.smart_memory = 0.0
    context.args.max_context = 4000
    context.args.temperature = 0.7
    context.llm_client = MagicMock()
    context.sandbox_dir = None
    context.memory_system = None
    context.profile_memory = None
    context.semantic_memory = None
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
    ctx.args.model = "test-model"
    ctx.args.perfect_it = False
    ctx.args.no_verifier = False
    ctx.profile_memory = MagicMock()
    ctx.profile_memory.get_context_string.return_value = ""
    ctx.memory_system = None
    ctx.scratchpad = MagicMock()
    ctx.scratchpad.list_all.return_value = ""
    ctx.memory_dir = tmp_path
    ctx.trajectory_collector = None
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


async def _confirmed(*a, **kw):
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    return (VerifyResult(verdict=VerifyVerdict.CONFIRMED, confidence=0.9,
                         reasoning="ok"), {"name": "execute_python"})


async def test_the_turn_outcome_line_omits_another_requests_confidence(tmp_path):
    """The live leak: a turn that computes no reading printed the previous
    request's. §4PJ: no turn computes one now — a stale value left on the
    context (as an old code path would leave it) must still never print."""
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
    context.args.smart_memory = 0.0
    context.args.max_context = 4000
    context.args.temperature = 0.7
    context.llm_client = MagicMock()
    context.sandbox_dir = None
    context.memory_system = None
    context.profile_memory = None
    context.semantic_memory = None
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
