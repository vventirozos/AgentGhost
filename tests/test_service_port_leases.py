"""Project-aware services + port leases (2026-07-30, §4G redesign).

The solar-sim incident distilled: many projects reach for the same port,
the model was the port allocator (and has no memory of prior grants), the
project↔service link was a string grep, and daemons started outside the
supervisor held ports invisibly. This suite pins the redesign:

* ports are LEASES the supervisor grants (preference → fallback, never
  kill a holder, explain every move);
* services are PROJECT-OWNED (scoped registry keys, first-class
  project_id, lifecycle coupling by field);
* reconcile()/adopt make unregistered listeners visible and manageable —
  read-only awareness, no auto-restarts;
* the execute tool refuses to background a server (`… &`) — the leak that
  created the invisible holder in the first place.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.sandbox.services import (
    ServiceSupervisor, entry_key, split_key,
    extract_command_port, substitute_command_port,
)
from ghost_agent.tools.execute import _daemonized_server_block


# ──────────────────────────────────────────────────────────────────────
# Stateful fake sandbox: a small port/process world
# ──────────────────────────────────────────────────────────────────────

class LeaseSandbox:
    """Simulates the container's port/process reality:

    * ``busy[port] = {"pid": …, "cmdline": …, "cwd": …}`` — pre-existing
      listeners (registered or not);
    * pids in ``alive`` answer ``kill -0``;
    * launched services get fresh pids (alive, but NOT listed as
      listeners — the fake keeps launch and listener state separate so
      tests control each independently).
    """

    def __init__(self, tmp_path):
        self.host_workspace = Path(tmp_path)
        self.busy = {}
        self.alive = set()
        self.calls = []
        self._next_pid = 1000

    def occupy(self, port, pid, cmdline="", cwd=""):
        self.busy[int(port)] = {"pid": pid, "cmdline": cmdline, "cwd": cwd}
        self.alive.add(pid)

    def execute(self, cmd, timeout=600, **kw):
        self.calls.append(cmd)
        if "nohup" in cmd:
            pid = self._next_pid
            self._next_pid += 1
            self.alive.add(pid)
            m = re.search(r'\.services/([A-Za-z0-9_-]+)\.cmd\.sh', cmd)
            svc = self.host_workspace / ".services"
            svc.mkdir(parents=True, exist_ok=True)
            (svc / f"{m.group(1)}.pid").write_text(str(pid))
            # Snapshot the registry AS THE PROCESS WOULD SEE IT at launch —
            # the token-before-launch regression (2026-07-30) asserts on it.
            try:
                self.registry_at_launch = json.loads(
                    (svc / "registry.json").read_text())
            except Exception:
                self.registry_at_launch = {}
            return (f"{pid}\n", 0)
        m = re.search(r"kill -0 (\d+)", cmd)
        if m:
            return ("", 0 if int(m.group(1)) in self.alive else 1)
        if "s.bind" in cmd:
            p = int(re.search(r'0\.0\.0\.0",(\d+)', cmd).group(1))
            return ("", 1 if p in self.busy else 0)
        if "sport" in cmd:                      # _holder_pid pipeline
            p = int(re.search(r"sport = :(\d+)", cmd).group(1))
            info = self.busy.get(p)
            return ((str(info["pid"]), 0) if info else ("", 0))
        if "ss -H -ltnp" in cmd:                # _listeners scan
            lines = [
                f'LISTEN 0 5 0.0.0.0:{p} 0.0.0.0:* '
                f'users:(("python3",pid={i["pid"]},fd=3))'
                for p, i in self.busy.items()
            ]
            return ("\n".join(lines), 0)
        m = re.search(r"/proc/(\d+)/cmdline", cmd)
        if m:
            pid = int(m.group(1))
            for i in self.busy.values():
                if i["pid"] == pid:
                    return (i["cmdline"], 0)
            return ("", 1)
        m = re.search(r"readlink /proc/(\d+)/cwd", cmd)
        if m:
            pid = int(m.group(1))
            for i in self.busy.values():
                if i["pid"] == pid:
                    return (i["cwd"], 0)
            return ("", 1)
        if "python3 -c" in cmd:                 # connect probe: listening ✓
            return ("", 0)
        return ("", 0)                          # test -d, stat, kills, …


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch):
    import ghost_agent.sandbox.services as svc
    monkeypatch.setattr(svc.time, "sleep", lambda s: None)


def _sup(tmp_path):
    sb = LeaseSandbox(tmp_path)
    return ServiceSupervisor(sb), sb


# ──────────────────────────────────────────────────────────────────────
# Key/identity helpers
# ──────────────────────────────────────────────────────────────────────

class TestIdentityHelpers:
    def test_entry_key_scoping(self):
        assert entry_key("abc123", "web") == "abc123:web"
        assert entry_key(None, "web") == "web"
        assert split_key("abc123:web") == ("abc123", "web")
        assert split_key("web") == (None, "web")

    def test_extract_command_port(self):
        assert extract_command_port("python3 -m http.server 8102") == 8102
        assert extract_command_port("uvicorn app --port 8100") == 8100
        assert extract_command_port("node server.js -p 8103") == 8103
        assert extract_command_port("python3 server.py") is None

    def test_substitute_command_port(self):
        assert substitute_command_port(
            "python3 -m http.server 8102", 8102, 8103) == \
            "python3 -m http.server 8103"
        # Word boundary: 18102 must not be mangled.
        assert substitute_command_port("x 18102 8102", 8102, 9999) == \
            "x 18102 9999"


# ──────────────────────────────────────────────────────────────────────
# Port leases: preference → fallback, never kill, always explain
# ──────────────────────────────────────────────────────────────────────

class TestPortLeases:
    def test_requested_free_port_is_granted(self, tmp_path):
        sup, _ = _sup(tmp_path)
        out = sup.start("web", "python3 server.py", port=8102)
        assert "RUNNING" in out
        assert sup._load()["web"]["port"] == 8102

    def test_occupied_requested_port_falls_back_and_explains(self, tmp_path):
        # THE solar-sim fix: an unregistered holder on the requested port
        # no longer dooms the launch — the next free port is granted, the
        # report names the holder, and nothing is killed.
        sup, sb = _sup(tmp_path)
        sb.occupy(8102, 636, cmdline="python3 server.py",
                  cwd="/workspace/projects/a1b2c3d4e5f6")
        out = sup.start("solar-sim", "python3 -m http.server 8102",
                        port=8102)
        assert "RUNNING" in out
        assert sup._load()["solar-sim"]["port"] != 8102
        assert "8102 is unavailable" in out
        assert "pid 636" in out
        assert "a1b2c3d4e5f6" in out          # project attribution
        assert not any("kill" in c and "636" in c for c in sb.calls)

    def test_hardcoded_command_port_is_rewritten_on_move(self, tmp_path):
        sup, sb = _sup(tmp_path)
        sb.occupy(8102, 636)
        sup.start("web", "python3 -m http.server 8102", port=8102)
        entry = sup._load()["web"]
        assert str(entry["port"]) in entry["command"]
        assert "8102" not in entry["command"]

    def test_command_literal_is_the_preference_when_arg_omitted(self, tmp_path):
        sup, _ = _sup(tmp_path)
        sup.start("web", "python3 -m http.server 8103")
        assert sup._load()["web"]["port"] == 8103

    def test_running_claimant_makes_allocator_skip_the_port(self, tmp_path):
        sup, _ = _sup(tmp_path)
        sup.start("a", "cmd", port=8100)
        out = sup.start("b", "cmd", port=8100)
        assert "RUNNING" in out
        assert sup._load()["b"]["port"] != 8100
        assert "claimed by RUNNING service 'a'" in out

    def test_dead_services_port_is_preserved_until_range_exhausts(self, tmp_path):
        # A dead entry's port is its restart contract — the allocator
        # prefers untouched ports and only reassigns a dead claim when
        # nothing else is free (and says so).
        sup, sb = _sup(tmp_path)
        sup.start("dead1", "cmd", port=8100)
        reg = sup._load()
        sb.alive.discard(reg["dead1"]["pid"])
        sup.start("fresh", "cmd")
        assert sup._load()["fresh"]["port"] == 8101   # skipped dead1's 8100

    def test_exhausted_range_reports_port_map(self, tmp_path):
        sup, sb = _sup(tmp_path)
        for p in (8100, 8101, 8102, 8103, 8104):
            sb.occupy(p, 600 + p)
        out = sup.start("web", "cmd", port=8100)
        assert out.startswith("Error: could not grant a port")
        assert "Port map" in out

    def test_restart_moves_port_when_taken_and_updates_registry(self, tmp_path):
        sup, sb = _sup(tmp_path)
        sup.start("web", "python3 server.py", port=8100)
        sup._load()
        # Something squats 8100 while the service is being restarted.
        sb.occupy(8100, 777, cmdline="squatter")
        out = sup.restart("web")
        assert "RUNNING" in out
        assert sup._load()["web"]["port"] != 8100
        assert "8100 is unavailable" in out


# ──────────────────────────────────────────────────────────────────────
# Project ownership: scoped keys, stamping, resolution
# ──────────────────────────────────────────────────────────────────────

class TestProjectOwnership:
    def test_start_stamps_project_and_scopes_key(self, tmp_path):
        sup, _ = _sup(tmp_path)
        sup.start("web", "cmd", port=8100, project_id="aaa111")
        reg = sup._load()
        assert "aaa111:web" in reg
        assert reg["aaa111:web"]["project_id"] == "aaa111"
        assert reg["aaa111:web"]["name"] == "web"

    def test_two_projects_can_both_run_web(self, tmp_path):
        sup, _ = _sup(tmp_path)
        o1 = sup.start("web", "cmd", port=8100, project_id="aaa111")
        o2 = sup.start("web", "cmd", port=8100, project_id="bbb222")
        assert "RUNNING" in o1 and "RUNNING" in o2
        reg = sup._load()
        assert {reg["aaa111:web"]["port"], reg["bbb222:web"]["port"]} == \
            {8100, 8101}

    def test_start_never_hijacks_another_projects_service(self, tmp_path):
        # Project B's `start web` must not resolve to project A's running
        # 'web' (the global unique-match rule applies to stop/status, not
        # start).
        sup, _ = _sup(tmp_path)
        sup.start("web", "cmd", port=8100, project_id="aaa111")
        out = sup.start("web", "cmd", port=8101, project_id="bbb222")
        assert "already running" not in out
        assert "RUNNING" in out

    def test_stop_resolves_unique_display_name_across_scopes(self, tmp_path):
        sup, _ = _sup(tmp_path)
        sup.start("web", "cmd", port=8100, project_id="aaa111")
        out = sup.stop("web")
        assert "stopped" in out

    def test_stop_ambiguous_name_lists_candidates(self, tmp_path):
        sup, _ = _sup(tmp_path)
        sup.start("web", "cmd", port=8100, project_id="aaa111")
        sup.start("web", "cmd", port=8101, project_id="bbb222")
        out = sup.stop("web")
        assert "ambiguous" in out
        assert "aaa111:web" in out and "bbb222:web" in out
        # The bound project disambiguates.
        out2 = sup.stop("web", project_id="aaa111")
        assert "stopped" in out2

    def test_legacy_unscoped_entry_is_adopted_into_project(self, tmp_path):
        sup, sb = _sup(tmp_path)
        sup.start("web", "cmd", port=8100)          # legacy, no project
        reg = sup._load()
        sb.alive.discard(reg["web"]["pid"])          # dies
        sup.start("web", "cmd2", port=8100, project_id="aaa111")
        reg = sup._load()
        assert "web" not in reg
        assert reg["aaa111:web"]["project_id"] == "aaa111"

    def test_scoped_file_stems_do_not_collide(self, tmp_path):
        sup, _ = _sup(tmp_path)
        sup.start("web", "cmd", port=8100, project_id="aaa111")
        sup.start("web", "cmd", port=8101, project_id="bbb222")
        svc = tmp_path / ".services"
        assert (svc / "aaa111--web.cmd.sh").exists()
        assert (svc / "bbb222--web.cmd.sh").exists()


# ──────────────────────────────────────────────────────────────────────
# Reconcile + adopt: awareness without action
# ──────────────────────────────────────────────────────────────────────

class TestReconcileAndAdopt:
    def test_orphan_listener_is_reported_with_project_hint(self, tmp_path):
        sup, sb = _sup(tmp_path)
        sb.occupy(8102, 636, cmdline="python3 server.py",
                  cwd="/workspace/projects/a1b2c3d4e5f6")
        r = sup.reconcile()
        assert len(r["orphans"]) == 1
        o = r["orphans"][0]
        assert o["port"] == 8102 and o["pid"] == 636
        assert o["project_hint"] == "a1b2c3d4e5f6"

    def test_reconcile_is_read_only(self, tmp_path):
        sup, sb = _sup(tmp_path)
        sb.occupy(8102, 636)
        n_calls = len(sb.calls)
        sup.reconcile()
        assert not any("kill" in c.lower() and "-TERM" in c
                       for c in sb.calls[n_calls:])
        assert sup._load() == {}                 # registry untouched

    def test_registered_running_service_is_not_an_orphan(self, tmp_path):
        sup, sb = _sup(tmp_path)
        sup.start("web", "cmd", port=8100)
        pid = sup._load()["web"]["pid"]
        sb.occupy(8100, pid)                     # it is listening
        assert sup.reconcile()["orphans"] == []

    def test_reconcile_summary_none_when_clean(self, tmp_path):
        sup, _ = _sup(tmp_path)
        assert sup.reconcile_summary() is None

    def test_reconcile_summary_names_orphans(self, tmp_path):
        sup, sb = _sup(tmp_path)
        sb.occupy(8102, 636, cmdline="python3 server.py")
        s = sup.reconcile_summary()
        assert "ORPHAN listener :8102" in s and "pid 636" in s

    def test_adopt_registers_existing_listener(self, tmp_path):
        sup, sb = _sup(tmp_path)
        sb.occupy(8102, 636, cmdline="python3 server.py",
                  cwd="/workspace/projects/a1b2c3d4e5f6")
        out = sup.adopt("solar", 8102)
        assert "Adopted pid 636" in out
        reg = sup._load()
        key = "a1b2c3d4e5f6:solar"               # project inferred from cwd
        assert reg[key]["adopted"] is True
        assert reg[key]["port"] == 8102
        # Now visible to stop/lifecycle.
        assert "solar" in sup.status()

    def test_adopt_nothing_listening_errors(self, tmp_path):
        sup, _ = _sup(tmp_path)
        out = sup.adopt("ghosty", 8102)
        assert out.startswith("Error: nothing is listening")

    def test_status_port_map_shows_owners_and_orphans(self, tmp_path):
        sup, sb = _sup(tmp_path)
        sup.start("web", "cmd", port=8100, project_id="aaa111")
        sb.occupy(8102, 636, cmdline="python3 server.py")
        out = sup.status()
        assert "Port map" in out
        assert "8100: RUNNING 'web' · project aaa111" in out
        assert "8102: OCCUPIED by unregistered pid 636" in out
        assert "8101: free" in out


# ──────────────────────────────────────────────────────────────────────
# Project plumbing: association field + briefing summary
# ──────────────────────────────────────────────────────────────────────

class TestProjectPlumbing:
    def _ctx(self, sb):
        return SimpleNamespace(sandbox_manager=sb)

    def test_field_association_beats_string_match(self, tmp_path):
        from ghost_agent.tools.projects import _project_service_entries
        sup, sb = _sup(tmp_path)
        sup.start("web", "cmd", port=8100, project_id="aaa111")
        # A stamped entry NEVER heuristic-matches another project, even if
        # its command mentions that project's path.
        sup.start("evil", "cp -r /workspace/projects/bbb222 .", port=8101,
                  project_id="aaa111")
        ents_a = _project_service_entries(self._ctx(sb), "aaa111")
        ents_b = _project_service_entries(self._ctx(sb), "bbb222")
        assert {e["name"] for e in ents_a} == {"web", "evil"}
        assert ents_b == []

    def test_legacy_entry_still_string_matches(self, tmp_path):
        from ghost_agent.tools.projects import _project_service_entries
        sup, sb = _sup(tmp_path)
        sup.start("old", "cd /workspace/projects/ccc333 && node s.js",
                  port=8100)                      # unstamped legacy
        ents = _project_service_entries(self._ctx(sb), "ccc333")
        assert [e["name"] for e in ents] == ["old"]

    def test_services_summary_is_exec_free(self, tmp_path):
        from ghost_agent.tools.projects import project_services_summary
        sup, sb = _sup(tmp_path)
        sup.start("web", "cmd", port=8100, project_id="aaa111")
        n_calls = len(sb.calls)
        s = project_services_summary(self._ctx(sb), "aaa111")
        assert len(sb.calls) == n_calls          # NO container execs
        assert s[0]["name"] == "web" and s[0]["port"] == 8100

    def test_briefing_renders_services_line(self):
        from ghost_agent.core.prompts import build_project_briefing
        # Composer-level: a stub store with a minimal project record.
        class _Store:
            def get_project(self, pid):
                return {"id": pid, "title": "T", "kind": "WEB",
                        "status": "ACTIVE", "goal": "g", "metadata": {}}
            def __getattr__(self, item):
                return lambda *a, **k: []
        out = build_project_briefing(
            _Store(), "aaa111",
            services=[{"name": "web", "port": 8100,
                       "command": "python3 server.py", "stale": False}])
        assert "SERVICES" in out
        assert "web · port 8100" in out

    def test_briefing_marks_stale_services(self):
        from ghost_agent.core.prompts import build_project_briefing
        class _Store:
            def get_project(self, pid):
                return {"id": pid, "title": "T", "kind": "WEB",
                        "status": "ACTIVE", "goal": "g", "metadata": {}}
            def __getattr__(self, item):
                return lambda *a, **k: []
        out = build_project_briefing(
            _Store(), "aaa111",
            services=[{"name": "web", "port": 8100, "command": "c",
                       "stale": True}])
        assert "STALE" in out


# ──────────────────────────────────────────────────────────────────────
# 2026-07-30 review-fix regressions
# ──────────────────────────────────────────────────────────────────────

class TestReviewFixes:
    def test_portless_optout_survives_restart(self, tmp_path):
        # D1: port=0 must persist — restart used to force-grant a lease
        # (and hard-fail when the range was full).
        sup, sb = _sup(tmp_path)
        sup.start("worker", "python3 worker.py", port=0)
        for p in (8100, 8101, 8102, 8103, 8104):
            sb.occupy(p, 600 + p)                 # range fully occupied
        out = sup.restart("worker")
        assert "RUNNING" in out                   # restarts fine portless
        entry = sup._load()["worker"]
        assert entry["port"] is None and entry["portless"] is True
        script = (tmp_path / ".services" / "worker.cmd.sh").read_text()
        assert "PORT=" not in script

    def test_bound_project_stop_prefers_its_own_web(self, tmp_path):
        # D2: plain-exact-first resolution let stop('web', project=A) kill
        # a legacy project-less 'web' while A's own scoped entry ran on.
        sup, _ = _sup(tmp_path)
        sup.start("web", "cmd", port=8100)                     # legacy
        sup.start("web", "cmd", port=8101, project_id="aaa111")
        out = sup.stop("web", project_id="aaa111")
        assert "stopped" in out
        reg = sup._load()
        assert "aaa111:web" not in reg            # OUR service stopped
        assert "web" in reg                       # legacy untouched

    def test_explicit_grant_over_dead_reservation_is_noted(self, tmp_path):
        # D5: an explicit request landing on a dead entry's reserved port
        # is granted but must SAY SO, and the port map must name the
        # RUNNING owner, not the dead reservation.
        sup, sb = _sup(tmp_path)
        sup.start("dead1", "cmd", port=8100)
        sb.alive.discard(sup._load()["dead1"]["pid"])
        out = sup.start("fresh", "cmd", port=8100)
        assert "RUNNING" in out
        assert "reserved by dead service 'dead1'" in out
        assert "8100: RUNNING 'fresh'" in sup.status()

    def test_ambiguity_reported_for_restart_status_logs(self, tmp_path):
        # D6: only stop() listed candidates; the other verbs answered
        # "no service named" — actively false for an ambiguous name.
        sup, _ = _sup(tmp_path)
        sup.start("web", "cmd", port=8100, project_id="aaa111")
        sup.start("web", "cmd", port=8101, project_id="bbb222")
        for verb in (sup.restart, sup.status, sup.logs):
            out = verb("web")
            assert "ambiguous" in out, verb.__name__

    def test_explicit_key_project_wins_over_ambient(self, tmp_path):
        # Minor: start('bbb222:web') under bound project aaa111 must not
        # silently re-scope — the ambiguity error says "use the full key".
        sup, sb = _sup(tmp_path)
        sup.start("web", "cmd", port=8100, project_id="bbb222")
        sb.alive.discard(sup._load()["bbb222:web"]["pid"])
        sup.start("bbb222:web", "cmd2", port=8100, project_id="aaa111")
        reg = sup._load()
        assert "bbb222:web" in reg and "aaa111:web" not in reg

    def test_reserved_command_literal_is_rewritten(self, tmp_path):
        # Minor: a command hardcoding a RESERVED port (8080) gets a free
        # lease AND the literal rewritten, instead of silently binding the
        # reserved port.
        sup, _ = _sup(tmp_path)
        out = sup.start("web", "python3 -m http.server 8080")
        assert "RESERVED port 8080" in out
        entry = sup._load()["web"]
        assert entry["port"] == 8100
        assert "8080" not in entry["command"]
        assert "8100" in entry["command"]


# ──────────────────────────────────────────────────────────────────────
# Service tokens (2026-07-30): the participant credential for
# sandbox-hosted apps — minted at start, exported as $GHOST_SERVICE_TOKEN,
# revoked when the registry entry goes.
# ──────────────────────────────────────────────────────────────────────

class TestServiceTokens:
    def test_token_minted_stored_and_exported(self, tmp_path):
        from ghost_agent.sandbox.services import valid_service_token
        sup, sb = _sup(tmp_path)
        sup.start("web", "cmd", port=8100, project_id="aaa111")
        entry = sup._load()["aaa111:web"]
        tok = entry["token"]
        assert tok and len(tok) == 32
        script = (tmp_path / ".services" / "aaa111--web.cmd.sh").read_text()
        assert f"export GHOST_SERVICE_TOKEN={tok}" in script
        assert valid_service_token(sb, tok) is True

    def test_stop_revokes_token(self, tmp_path):
        from ghost_agent.sandbox.services import valid_service_token
        sup, sb = _sup(tmp_path)
        sup.start("web", "cmd", port=8100)
        tok = sup._load()["web"]["token"]
        sup.stop("web")
        assert valid_service_token(sb, tok) is False

    def test_wrong_or_empty_token_invalid(self, tmp_path):
        from ghost_agent.sandbox.services import valid_service_token
        sup, sb = _sup(tmp_path)
        sup.start("web", "cmd", port=8100)
        assert valid_service_token(sb, "nope") is False
        assert valid_service_token(sb, "") is False
        assert valid_service_token(None, "anything") is False

    def test_restart_mints_fresh_token(self, tmp_path):
        from ghost_agent.sandbox.services import valid_service_token
        sup, sb = _sup(tmp_path)
        sup.start("web", "cmd", port=8100)
        old = sup._load()["web"]["token"]
        sup.restart("web")
        new = sup._load()["web"]["token"]
        assert new != old
        assert valid_service_token(sb, new) is True
        assert valid_service_token(sb, old) is False

    def test_token_is_in_registry_BEFORE_launch(self, tmp_path):
        # The race that broke chess-coach-v2 live (2026-07-30): the entry
        # used to be saved only AFTER launch + liveness + port probes, so a
        # service probing the agent with $GHOST_SERVICE_TOKEN in its first
        # seconds was 403'd — and one-shot resolvers then latched a wrong
        # base URL forever. The credential must be persisted pre-launch.
        sup, sb = _sup(tmp_path)
        sup.start("web", "cmd", port=8100, project_id="aaa111")
        at_launch = getattr(sb, "registry_at_launch", {})
        assert "aaa111:web" in at_launch
        tok_at_launch = at_launch["aaa111:web"].get("token")
        assert tok_at_launch
        assert sup._load()["aaa111:web"]["token"] == tok_at_launch

    def test_failed_launch_pops_provisional_entry(self, tmp_path):
        # The pre-launch save must not leave ghost registrations behind
        # when the launch dies (exited-immediately path).
        sup, sb = _sup(tmp_path)
        real_exec = sb.execute

        def dying_exec(cmd, timeout=600, **kw):
            out, code = real_exec(cmd, timeout=timeout, **kw)
            if "nohup" in cmd:
                # the launched pid dies instantly
                sb.alive.discard(sb._next_pid - 1)
            return out, code
        sb.execute = dying_exec
        out = sup.start("web", "cmd", port=8100)
        assert "exited immediately" in out
        assert "web" not in sup._load()

    def test_briefing_summary_never_leaks_token(self, tmp_path):
        from ghost_agent.tools.projects import project_services_summary
        sup, sb = _sup(tmp_path)
        sup.start("web", "cmd", port=8100, project_id="aaa111")
        ctx = SimpleNamespace(sandbox_manager=sb)
        s = project_services_summary(ctx, "aaa111")
        assert "token" not in json.dumps(s)


# ──────────────────────────────────────────────────────────────────────
# Execute daemon guard: the orphan source, closed
# ──────────────────────────────────────────────────────────────────────

class TestDaemonizedServerGuard:
    def test_solar_sim_leak_shape_is_blocked(self):
        out = _daemonized_server_block(
            "python3 server.py &>/dev/null & sleep 2 && curl -s "
            "http://localhost:8102/")
        assert out is not None
        assert "manage_services" in out

    @pytest.mark.parametrize("cmd", [
        "python3 -m http.server 8102 &",
        "setsid python3 -m http.server 8102",
        "nohup uvicorn app:app --port 8100 &",
        "node app/server.js &",
    ])
    def test_detached_servers_blocked(self, cmd):
        assert _daemonized_server_block(cmd) is not None

    @pytest.mark.parametrize("cmd", [
        "python3 -m http.server 8102",        # foreground dies with exec
        "python3 server.py",                   # serverish but foreground
        "sleep 30 & echo bg",                  # background but not a server
        "make build && ls 2>&1",
        "grep -rn server.py src/",             # mentions, doesn't run
        "",
        # 2026-07-30 review: data is not a command —
        "cat > run.sh <<'EOF'\nnohup python3 server.py &\nEOF",
        'git commit -m "fix server.py & restart logic"',
        # …and a job-killed smoke test does not outlive the exec.
        "uvicorn app --port 8100 & curl -s localhost:8100; kill %1",
    ])
    def test_legit_commands_pass(self, cmd):
        assert _daemonized_server_block(cmd) is None

    def test_nested_shell_payload_still_blocked(self):
        # Quote-stripping must not blind the guard to `bash -c '… &'`.
        assert _daemonized_server_block(
            "bash -c 'python3 -m http.server 8102 &'") is not None


# ──────────────────────────────────────────────────────────────────────
# Tool wrapper: adopt action + project plumb-through
# ──────────────────────────────────────────────────────────────────────

class TestToolWrapper:
    @pytest.mark.asyncio
    async def test_adopt_action_via_wrapper(self, tmp_path):
        from ghost_agent.tools.sandbox_services import tool_manage_services
        sup, sb = _sup(tmp_path)
        sb.occupy(8102, 636, cmdline="python3 server.py")
        out = await tool_manage_services(
            action="adopt", name="found", port=8102, sandbox_manager=sb)
        assert "Adopted pid 636" in out

    @pytest.mark.asyncio
    async def test_start_receives_bound_project(self, tmp_path):
        from ghost_agent.tools.sandbox_services import tool_manage_services
        sup, sb = _sup(tmp_path)
        out = await tool_manage_services(
            action="start", name="web", command="cmd", port=8100,
            sandbox_manager=sb, project_id="aaa111")
        assert "project aaa111" in out
        assert "aaa111:web" in ServiceSupervisor(sb)._load()

    def test_tool_definition_documents_the_lease(self):
        from ghost_agent.tools.sandbox_services import (
            MANAGE_SERVICES_TOOL_DEFINITION as D)
        desc = D["function"]["description"]
        assert "PORTS ARE ASSIGNED" in desc
        actions = D["function"]["parameters"]["properties"]["action"]["enum"]
        assert "adopt" in actions
