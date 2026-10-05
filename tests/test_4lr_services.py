"""§4LR — sandbox services: behaviour pins for the review's fixes."""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import ghost_agent.sandbox.services as svc
from ghost_agent.sandbox.services import ServiceSupervisor, add_default_port
from ghost_agent.tools.outcome import OutcomeStatus, ToolOutcome


class FakeSandbox:
    """Records commands; `handler(cmd) -> (out, code)`; writes the pidfile
    the real cmd.sh writes, as tests/test_sandbox_services.py does."""

    def __init__(self, tmp_path, handler, host_netns=False):
        self.host_workspace = Path(tmp_path)
        self.calls = []
        self.handler = handler
        self._host_netns = host_netns
        self.container = SimpleNamespace(id="cid1", attrs={"State": {"StartedAt": "T1"}})

    def binds_host_netns(self):
        return self._host_netns

    def execute(self, cmd, timeout=600, **kw):
        self.calls.append(cmd)
        out, code = self.handler(cmd)
        if "nohup" in cmd and code == 0:
            m = re.search(r"\.services/([A-Za-z0-9_-]+)\.cmd\.sh", cmd)
            tok = out.strip().split()[-1] if out.strip() else ""
            if m and tok.isdigit():
                d = self.host_workspace / ".services"
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{m.group(1)}.pid").write_text(tok)
        return out, code


def handler(pid="321", listens=True, held_line="", addrs="0.0.0.0"):
    def h(cmd):
        if "nohup" in cmd:
            return (f"{pid}\n", 0)
        if "s.bind" in cmd:                                      # allocator bind probe: free
            return ("", 0)
        if "ss -H -ltnp" in cmd and "echo ---" in cmd:          # _ports_held_by
            return (f"{held_line}\n---\n{pid}\n", 0)
        if "ss -H -ltn " in cmd:                                 # _listen_addrs
            return (f"LISTEN 0 5 {addrs}:8100 *:*\n", 0) if listens else ("", 1)
        if "ss -H -ltnp" in cmd:
            return ("", 0)
        if "python3 -c" in cmd:                                  # connect probe
            return ("", 0 if listens else 1)
        return ("", 0)
    return h


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(svc.time, "sleep", lambda s: None)


def _script(tmp_path):
    return "".join(f.read_text() for f in (tmp_path / ".services").glob("*.cmd.sh"))


# ── M2: a server that names no port gets the lease, never the reserved 8000 ──

@pytest.mark.parametrize("cmd,want", [
    ("python3 -m http.server", "http.server 8100"),
    ("python3 -m http.server --bind 0.0.0.0", "http.server 8100 --bind 0.0.0.0"),
    ("uvicorn app:app", "uvicorn --port 8100 app:app"),
    ("flask run", "flask run --port 8100"),
    ("python3 -m http.server 8102", "http.server 8102"),          # names its own: unchanged
])
def test_a_server_without_a_port_gets_the_leased_one(cmd, want):
    assert want in add_default_port(cmd, 8100)[0]


def test_host_network_mode_binds_the_file_server_to_loopback():
    assert "--bind 127.0.0.1" in add_default_port("python3 -m http.server", 8100, loopback=True)[0]
    assert "--bind 127.0.0.1" not in add_default_port("python3 -m http.server", 8100)[0]


def test_the_started_script_carries_the_port_and_unbuffered_logs(tmp_path):
    sup = ServiceSupervisor(FakeSandbox(tmp_path, handler()))
    sup.start("web", "python3 -m http.server")
    script = _script(tmp_path)
    assert "http.server 8100" in script and "PYTHONUNBUFFERED=1" in script


def test_not_listening_names_the_port_it_really_bound_and_stops_a_reserved_one(tmp_path, monkeypatch):
    killed = []
    monkeypatch.setattr(ServiceSupervisor, "_kill_pgroup", lambda self, pid: killed.append(pid) or True)
    held = 'LISTEN 0 5 0.0.0.0:8000 0.0.0.0:* users:(("python3",pid=321,fd=3))'
    sup = ServiceSupervisor(FakeSandbox(tmp_path, handler(listens=False, held_line=held)))
    out = sup.start("web", "python3 app.py", port=8100)
    assert isinstance(out, ToolOutcome) and out.status is OutcomeStatus.FAILED
    assert "listening on port 8000 instead" in out and "STOPPED" in out and killed == [321]


def test_an_explicit_reserved_port_is_re_leased_with_a_note(tmp_path):
    sup = ServiceSupervisor(FakeSandbox(tmp_path, handler()))
    out = sup.start("web", "python3 -m http.server", port=8000)
    assert "RESERVED" in out and "8100" in out and "PORT=8000\n" not in _script(tmp_path)


def test_a_file_server_at_the_sandbox_root_is_flagged(tmp_path):
    sup = ServiceSupervisor(FakeSandbox(tmp_path, handler()))
    out = sup.start("web", "python3 -m http.server 8100")
    assert "publishes EVERYTHING" in out
    out = sup.start("web2", "python3 -m http.server 8101 --directory site")
    assert "publishes EVERYTHING" not in out


def test_host_network_wildcard_bind_is_warned(tmp_path):
    sup = ServiceSupervisor(FakeSandbox(tmp_path, handler(addrs="0.0.0.0"), host_netns=True))
    out = sup.start("web", "python3 app.py", port=8100)
    assert "reachable from the LAN" in out


# ── M4: stop-all is scoped; project rows survive ──

def _seed(tmp_path, rows):
    d = tmp_path / ".services"
    d.mkdir(parents=True, exist_ok=True)
    (d / "registry.json").write_text(json.dumps(rows))


def _rows():
    return {
        "aaa111:web": {"name": "web", "project_id": "aaa111", "pid": 401, "port": 8100, "command": "x"},
        "ccc333:jj": {"name": "jj", "project_id": "ccc333", "pid": 402, "port": 8101, "command": "y"},
        "tmp": {"name": "tmp", "pid": 403, "port": 8102, "command": "z"},
    }


def test_stop_all_from_a_project_stops_only_that_project_and_keeps_its_row(tmp_path, monkeypatch):
    stopped = []
    monkeypatch.setattr(ServiceSupervisor, "_kill_service",
                        lambda self, e, others=None: stopped.append(e["name"]) or True)
    _seed(tmp_path, _rows())
    sup = ServiceSupervisor(FakeSandbox(tmp_path, handler()))
    sup.stop_all(project_id="aaa111")
    reg = json.loads((tmp_path / ".services" / "registry.json").read_text())
    assert stopped == ["web"] and set(reg) == {"aaa111:web", "ccc333:jj", "tmp"}


def test_stop_all_with_no_project_touches_only_projectless_services(tmp_path, monkeypatch):
    stopped = []
    monkeypatch.setattr(ServiceSupervisor, "_kill_service",
                        lambda self, e, others=None: stopped.append(e["name"]) or True)
    _seed(tmp_path, _rows())
    sup = ServiceSupervisor(FakeSandbox(tmp_path, handler()))
    sup.stop_all()
    reg = json.loads((tmp_path / ".services" / "registry.json").read_text())
    assert stopped == ["tmp"] and "tmp" not in reg and "ccc333:jj" in reg
    stopped.clear()
    sup.stop_all(all_projects=True)
    reg = json.loads((tmp_path / ".services" / "registry.json").read_text())
    assert sorted(stopped) == ["jj", "web"] and set(reg) == {"aaa111:web", "ccc333:jj"}


async def test_the_vague_aliases_no_longer_mean_stop_all(tmp_path):
    from ghost_agent.tools.sandbox_services import tool_manage_services
    out = await tool_manage_services(action="cleanup", sandbox_manager=SimpleNamespace())
    assert "unknown action" in out


def test_a_service_that_survives_the_kill_is_a_declared_failure(tmp_path, monkeypatch):
    def kill(self, e, others=None):
        self._last_kill_survived = True
        return True
    monkeypatch.setattr(ServiceSupervisor, "_kill_service", kill)
    _seed(tmp_path, _rows())
    sup = ServiceSupervisor(FakeSandbox(tmp_path, handler()))
    out = sup.stop_all(all_projects=True)
    assert isinstance(out, ToolOutcome) and out.status is OutcomeStatus.FAILED


# ── M3: one service's failures never pre-flight-block another ──

def test_each_service_is_its_own_foresight_target():
    from ghost_agent.core.foresight import call_target
    a = call_target("manage_services", "start", {"action": "start", "name": "probesvc", "command": "x"})
    b = call_target("manage_services", "start", {"action": "start", "name": "sponza", "port": 8100})
    assert a and b and a != b


# ── M1: a running service is not reported stale ──

def test_a_live_service_is_not_stale_in_the_briefing(monkeypatch):
    import ghost_agent.tools.projects as PR
    rows = [{"name": "web", "port": 8100, "container_id": "cid1:T1"},
            {"name": "old", "port": 8101, "container_id": "cid0:T0"},
            {"name": "legacy", "port": 8102, "container_id": "cid1"}]
    monkeypatch.setattr(PR, "_project_service_entries", lambda ctx, pid: rows)
    ctx = SimpleNamespace(sandbox_manager=SimpleNamespace(
        container=SimpleNamespace(id="cid1", attrs={"State": {"StartedAt": "T1"}})))
    got = {r["name"]: r["stale"] for r in PR.project_services_summary(ctx, "p")}
    assert got == {"web": False, "old": True, "legacy": True}


# ── M5: the release rehearsal believes a declared failure ──

def test_a_declared_failed_restart_fails_the_rehearsal(tmp_path, monkeypatch):
    import ghost_agent.tools.projects as PR
    from ghost_agent.memory.projects import ProjectStore
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    pid = store.create_project("App")
    entry = {"name": "web", "key": f"{pid}:web", "port": 8100, "command": "x"}
    monkeypatch.setattr(PR, "_project_service_entries", lambda ctx, p: [entry])

    class Sup:
        def restart(self, addr):
            return ToolOutcome.failed("Service 'web' started (pid 9) but nothing is listening",
                                      reason_code="service_failed_to_bind")

        def list_entries(self):
            return [entry]

        def _port_listening(self, port):
            return True
    monkeypatch.setattr(svc, "get_service_supervisor", lambda sm: Sup())
    res = PR._release_rehearsal(SimpleNamespace(sandbox_manager=None), store, pid)
    assert res["ok"] is False


def test_the_rehearsal_checks_the_port_inside_the_sandbox(tmp_path, monkeypatch):
    import ghost_agent.tools.projects as PR
    from ghost_agent.memory.projects import ProjectStore
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    pid = store.create_project("App")
    entry = {"name": "web", "key": f"{pid}:web", "port": 8100, "command": "x"}
    monkeypatch.setattr(PR, "_project_service_entries", lambda ctx, p: [entry])
    monkeypatch.setattr(PR, "_probe_tcp", lambda port, **k: True)      # the host answers…

    class Sup:
        def restart(self, addr):
            return "Service 'web' RUNNING"

        def list_entries(self):
            return [entry]

        def _port_listening(self, port):
            return False                                                # …the sandbox does not
    monkeypatch.setattr(svc, "get_service_supervisor", lambda sm: Sup())
    assert PR._release_rehearsal(SimpleNamespace(sandbox_manager=None), store, pid)["ok"] is False


# ── a bulk stop is never a macro step ──

def test_a_macro_cannot_carry_stop_all_on_an_unused_slot():
    from ghost_agent.tools.composed_skills import mint_param_schema
    obs = [[{"operation": "read", "path": f"/f{i}"} for i in range(3)],
           [{"action": "stop-all", "name": f"svc{i}"} for i in range(3)]]    # a varying, UNREAD name
    _, _, why = mint_param_schema(("file_system", "manage_services"), obs)
    assert why and "stop-all" in why


def test_a_legacy_stamp_is_stale_even_when_the_start_time_is_unknown(monkeypatch):
    import ghost_agent.tools.projects as PR
    monkeypatch.setattr(PR, "_project_service_entries",
                        lambda ctx, pid: [{"name": "legacy", "port": 8102, "container_id": "cid1"}])
    ctx = SimpleNamespace(sandbox_manager=SimpleNamespace(container=SimpleNamespace(id="cid1", attrs={})))
    assert PR.project_services_summary(ctx, "p")[0]["stale"] is True


def test_another_processes_port_is_not_blamed_on_the_service(tmp_path, monkeypatch):
    monkeypatch.setattr(ServiceSupervisor, "_kill_pgroup", lambda self, pid: True)
    held = ('LISTEN 0 5 0.0.0.0:9050 0.0.0.0:* users:(("tor",pid=77,fd=3))\n'
            'LISTEN 0 5 0.0.0.0:5000 0.0.0.0:* users:(("python3",pid=321,fd=3))')
    sup = ServiceSupervisor(FakeSandbox(tmp_path, handler(listens=False, held_line=held)))
    out = sup.start("web", "python3 app.py", port=8100)
    assert "listening on port 5000 instead" in out and "9050" not in out and "STOPPED" not in out



# ── fix review: only the program that RUNS is edited ──

@pytest.mark.parametrize("cmd", [
    "python3 -m http.server $PORT", "python3 -m http.server ${PORT}",
    "pip install uvicorn && uvicorn app:app --port 8100",
    "pip install uvicorn && uvicorn app:app",
    'python3 -c "import http.server; print(1)"',
    "echo uvicorn", "python3 -m http.server | tee server.log"])
def test_commands_that_are_not_a_bare_server_are_left_alone(cmd):
    assert add_default_port(cmd, 8100) == (cmd, None)


@pytest.mark.parametrize("cmd,want", [
    ("cd site && python3 -m http.server", "cd site && python3 -m http.server 8100"),
    ("python -m uvicorn app:app", "python -m uvicorn --port 8100 app:app"),
    ("flask --app web run", "flask --app web run --port 8100"),
    ("FOO=1 python3 -m http.server --directory x", "FOO=1 python3 -m http.server 8100 --directory x")])
def test_the_running_server_gets_the_port_in_a_valid_place(cmd, want):
    assert add_default_port(cmd, 8100)[0] == want


def test_a_reserved_holder_that_survives_the_kill_is_not_reported_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(ServiceSupervisor, "_kill_pgroup", lambda self, pid: False)
    held = 'LISTEN 0 5 0.0.0.0:8000 0.0.0.0:* users:(("python3",pid=321,fd=3))'
    sup = ServiceSupervisor(FakeSandbox(tmp_path, handler(listens=False, held_line=held)))
    out = sup.start("web", "python3 app.py", port=8100)
    assert "could NOT be stopped" in out and "was STOPPED" not in out


def test_one_services_failures_never_predict_another_at_class_level():
    from ghost_agent.core.foresight import call_target, target_class
    a = target_class("manage_services", "start", call_target("manage_services", "start", {"name": "probesvc"}))
    b = target_class("manage_services", "start", call_target("manage_services", "start", {"name": "sponza", "port": 8100}))
    assert a != b and a.startswith("svc:") and b == "svc:sponza"


def test_the_rehearsal_fails_a_loopback_only_app(tmp_path, monkeypatch):
    import ghost_agent.tools.projects as PR
    from ghost_agent.memory.projects import ProjectStore
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    pid = store.create_project("App")
    entry = {"name": "web", "key": f"{pid}:web", "port": 8104, "command": "x"}
    monkeypatch.setattr(PR, "_project_service_entries", lambda ctx, p: [entry])

    class Sup:
        def restart(self, addr):
            return "Service 'web' RUNNING"

        def list_entries(self):
            return [entry]

        def _port_listening(self, port):
            return True

        def _host_unreachable_bind(self, port):
            return True
    monkeypatch.setattr(svc, "get_service_supervisor", lambda sm: Sup())
    assert PR._release_rehearsal(SimpleNamespace(sandbox_manager=None), store, pid)["ok"] is False



def test_a_repeated_stop_all_does_not_report_dead_rows_as_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(ServiceSupervisor, "_kill_service", lambda self, e, others=None: False)
    _seed(tmp_path, _rows())
    out = ServiceSupervisor(FakeSandbox(tmp_path, handler())).stop_all(project_id="aaa111")
    assert "Stopped 0 running" in out and "Already dead: aaa111:web" in out
