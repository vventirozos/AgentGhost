"""Regression tests for the 2026-07-22 services.py review fixes.

Covers the four fixes from the sandbox-services code review:

1. Port-reclaim ownership — stop('dead-twin') must NOT kill a DIFFERENT
   registered service that legitimately holds the port now, and must not
   kill a positively-foreign unregistered holder either. The historical
   mis-tracked-orphan reclaim (holder in the recorded pid's process group,
   or ownership unknowable) still works.
2. Container-generation stamp — a recycled pid in a NEW container
   generation reads DEAD, not RUNNING; stop() of such an entry never
   signals the recycled pid.
3. restart() preserves the registration when the relaunch fails.
4. `_reap_dead` (defined-but-never-called dead code) is gone.

Harness reused from tests/test_sandbox_services.py (FakeSandbox +
happy_handler)."""

import json
import time

import pytest

from ghost_agent.sandbox.services import ServiceSupervisor

from tests.test_sandbox_services import FakeSandbox, happy_handler


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch):
    import ghost_agent.sandbox.services as svc
    monkeypatch.setattr(svc.time, "sleep", lambda s: None)


def _write_registry(tmp_path, reg):
    svc_dir = tmp_path / ".services"
    svc_dir.mkdir(parents=True, exist_ok=True)
    (svc_dir / "registry.json").write_text(json.dumps(reg))


def _entry(name, pid, port=None, container_id=None):
    e = {"name": name, "command": f"python3 {name}.py", "pid": pid,
         "port": port, "workdir": "/workspace", "started_at": time.time()}
    if container_id is not None:
        e["container_id"] = container_id
    return e


class _Container:
    def __init__(self, cid):
        self.id = cid


# ──────────────────────────────────────────────────────────────────────
# Fix 1 — port-reclaim ownership
# ──────────────────────────────────────────────────────────────────────

class TestPortReclaimOwnership:
    def test_stop_dead_twin_does_not_kill_live_port_owner(self, tmp_path):
        """THE bug: 'chess' (dead, port 8100) is stopped while 'webapp'
        (alive) now owns 8100 — the reclaim used to kill webapp."""
        kills = []

        def handler(cmd):
            if "kill -TERM" in cmd or "kill -KILL" in cmd:
                kills.append(cmd)
                return ("", 0)
            if "kill -0" in cmd:
                return ("", 0 if "222" in cmd else 1)  # webapp alive only
            if "python3 -c" in cmd:
                return ("", 0)                          # 8100 answers (webapp)
            if "ss -" in cmd:
                return ("222\n", 0)                     # holder IS webapp
            return ("", 0)

        _write_registry(tmp_path, {
            "chess": _entry("chess", 111, port=8100),
            "webapp": _entry("webapp", 222, port=8100),
        })
        sup = ServiceSupervisor(FakeSandbox(tmp_path, handler))
        out = sup.stop("chess")
        assert "chess" in out
        assert not kills, f"a process was killed: {kills}"
        reg = sup._load()
        assert "webapp" in reg and "chess" not in reg

    def test_reclaim_skips_positively_foreign_holder(self, tmp_path):
        """Unregistered holder whose pgid/sid provably is not ours →
        left alone."""
        kills = []

        def handler(cmd):
            if "kill -TERM" in cmd or "kill -KILL" in cmd:
                kills.append(cmd)
                return ("", 0)
            if "kill -0" in cmd:
                return ("", 1)                          # recorded pid dead
            if "python3 -c" in cmd:
                return ("", 0)                          # port still answers
            if "ss -" in cmd:
                return ("999\n", 0)                     # a stranger holds it
            if "/proc/999/stat" in cmd:
                return ("888 888\n", 0)                 # pgid/sid ≠ 111
            return ("", 0)

        _write_registry(tmp_path, {"chess": _entry("chess", 111, port=8100)})
        sup = ServiceSupervisor(FakeSandbox(tmp_path, handler))
        out = sup.stop("chess")
        assert "already dead" in out
        assert not kills, f"foreign holder was killed: {kills}"

    def test_reclaim_still_kills_own_process_group_orphan(self, tmp_path):
        """The safety net the reclaim exists for: the mis-tracked real
        process sits in the recorded pid's process group → killed."""
        kills = []

        def handler(cmd):
            if "kill -TERM" in cmd or "kill -KILL" in cmd:
                kills.append(cmd)
                return ("", 0)
            if "kill -0" in cmd:
                return ("", 1)                          # recorded pid dead
            if "python3 -c" in cmd:
                return ("", 0)                          # orphan still listening
            if "ss -" in cmd:
                return ("625\n", 0)
            if "/proc/625/stat" in cmd:
                return ("111 111\n", 0)                 # pgid == recorded pid
            return ("", 0)

        _write_registry(tmp_path, {"web": _entry("web", 111, port=8102)})
        sup = ServiceSupervisor(FakeSandbox(tmp_path, handler))
        sup.stop("web")
        assert any("625" in c for c in kills), kills

    def test_start_refuses_port_claimed_by_live_service(self, tmp_path):
        def handler(cmd):
            if "kill -0" in cmd:
                return ("", 0 if "111" in cmd else 1)   # chess alive
            return ("", 0)

        _write_registry(tmp_path, {"chess": _entry("chess", 111, port=8100)})
        sup = ServiceSupervisor(FakeSandbox(tmp_path, handler))
        out = sup.start("other", "python3 x.py", port=8100)
        assert out.startswith("Error:")
        assert "already claimed" in out and "chess" in out
        assert "other" not in sup._load()               # nothing registered
        # A different port is fine.
        sb2 = FakeSandbox(tmp_path, happy_handler())
        assert "RUNNING" in ServiceSupervisor(sb2).start(
            "other", "python3 x.py", port=8101)

    def test_start_flags_port_answered_by_foreign_process(self, tmp_path):
        """listening ✓ must not be claimed when the identified holder is
        provably NOT the just-started service (failed bind, stray orphan)."""
        def handler(cmd):
            if "nohup" in cmd:
                return ("300\n", 0)
            if "kill -0" in cmd:
                return ("", 0)                          # our pid alive
            if "python3 -c" in cmd:
                return ("", 0)                          # port answers…
            if "ss -" in cmd:
                return ("999\n", 0)                     # …but not by us
            if "/proc/999/stat" in cmd:
                return ("888 888\n", 0)                 # foreign pgid/sid
            return ("", 0)

        sup = ServiceSupervisor(FakeSandbox(tmp_path, handler))
        out = sup.start("web", "python3 app.py", port=8103)
        assert "DIFFERENT process" in out
        assert "listening ✓" not in out
        assert "web" in sup._load()      # process is alive → still registered


# ──────────────────────────────────────────────────────────────────────
# Fix 2 — container-generation stamp kills the recycled-pid phantom
# ──────────────────────────────────────────────────────────────────────

class TestContainerGenerationStamp:
    def test_start_stamps_container_id(self, tmp_path):
        sb = FakeSandbox(tmp_path, happy_handler())
        sb.container = _Container("gen-A")
        sup = ServiceSupervisor(sb)
        assert "RUNNING" in sup.start("web", "cmd", port=8100)
        assert sup._load()["web"]["container_id"] == "gen-A"

    def test_recycled_pid_in_new_generation_reads_dead(self, tmp_path):
        sb = FakeSandbox(tmp_path, happy_handler())     # kill -0 says ALIVE
        sb.container = _Container("gen-A")
        sup = ServiceSupervisor(sb)
        sup.start("web", "cmd", port=8100)
        assert "web: RUNNING" in sup.status()
        sb.container = _Container("gen-B")              # container recreated
        out = sup.status()
        assert "web: DEAD" in out                       # despite kill -0 == 0

    def test_stop_in_new_generation_never_signals_recycled_pid(self, tmp_path):
        kills = []

        def handler(cmd):
            if "kill -TERM" in cmd or "kill -KILL" in cmd:
                kills.append(cmd)
                return ("", 0)
            if "nohup" in cmd:
                return ("100\n", 0)
            if "kill -0" in cmd:
                return ("", 0)          # the NUMBER is alive (recycled pid)
            if "python3 -c" in cmd:
                return ("", 1)          # nothing listening in new container
            return ("", 0)

        sb = FakeSandbox(tmp_path, handler)
        sb.container = _Container("gen-A")
        sup = ServiceSupervisor(sb)
        sup.start("web", "cmd")
        sb.container = _Container("gen-B")
        out = sup.stop("web")
        assert "already dead" in out
        assert not kills, f"recycled pid was signalled: {kills}"

    def test_same_generation_still_reads_running(self, tmp_path):
        sb = FakeSandbox(tmp_path, happy_handler())
        sb.container = _Container("gen-A")
        sup = ServiceSupervisor(sb)
        sup.start("web", "cmd", port=8100)
        assert "web: RUNNING" in sup.status()           # no false negative

    def test_legacy_entry_without_stamp_falls_back_to_pid(self, tmp_path):
        # Pre-fix registries have no container_id — pid check alone applies.
        sb = FakeSandbox(tmp_path, happy_handler())
        sb.container = _Container("gen-B")
        _write_registry(tmp_path, {"old": _entry("old", 555, port=8101)})
        sup = ServiceSupervisor(sb)
        assert "old: RUNNING" in sup.status()

    def test_start_refuses_duplicate_only_for_same_generation(self, tmp_path):
        # An entry from an OLD generation is dead: same-name start proceeds
        # instead of erroring "already running".
        sb = FakeSandbox(tmp_path, happy_handler())
        sb.container = _Container("gen-B")
        _write_registry(tmp_path, {
            "web": _entry("web", 555, port=8100, container_id="gen-A")})
        sup = ServiceSupervisor(sb)
        out = sup.start("web", "cmd", port=8100)
        assert "already running" not in out
        assert "RUNNING" in out
        assert sup._load()["web"]["container_id"] == "gen-B"


# ──────────────────────────────────────────────────────────────────────
# Fix 3 — restart() preserves the registration on relaunch failure
# ──────────────────────────────────────────────────────────────────────

class TestRestartPreservesRegistration:
    def test_failed_relaunch_restores_entry(self, tmp_path):
        sb = FakeSandbox(tmp_path, happy_handler())
        sup = ServiceSupervisor(sb)
        assert "RUNNING" in sup.start(
            "web", "python3 app.py", port=8100, workdir="projects/x")

        def broken(cmd):                # workdir vanished; everything dead
            if "test -d" in cmd:
                return ("", 1)
            if "kill -0" in cmd:
                return ("", 1)
            if "python3 -c" in cmd:
                return ("", 1)
            return ("", 0)

        sb.handler = broken
        out = sup.restart("web")
        assert out.startswith("Error:")
        assert "preserved" in out
        reg = sup._load()
        assert "web" in reg, "registration was lost on failed restart"
        assert reg["web"]["command"] == "python3 app.py"
        assert reg["web"]["port"] == 8100
        assert reg["web"]["workdir"] == "/workspace/projects/x"

    def test_restart_recovers_after_transient_failure(self, tmp_path):
        sb = FakeSandbox(tmp_path, happy_handler())
        sup = ServiceSupervisor(sb)
        sup.start("web", "python3 app.py", port=8100)

        def broken(cmd):
            if "test -d" in cmd:
                return ("", 1)
            if "kill -0" in cmd:
                return ("", 1)
            return ("", 0)

        sb.handler = broken
        assert sup.restart("web").startswith("Error:")
        # Cause fixed → the SAME registration relaunches, no re-supplied spec.
        sb.handler = happy_handler(pid="777")
        out = sup.restart("web")
        assert "RUNNING" in out and "pid 777" in out
        assert sup._load()["web"]["command"] == "python3 app.py"

    def test_successful_restart_unchanged(self, tmp_path):
        sb = FakeSandbox(tmp_path, happy_handler())
        sup = ServiceSupervisor(sb)
        sup.start("web", "python3 app.py", port=8100)
        out = sup.restart("web")
        assert "RUNNING" in out
        assert "preserved" not in out


# ──────────────────────────────────────────────────────────────────────
# Fix 4 — _reap_dead dead code removed
# ──────────────────────────────────────────────────────────────────────

def test_reap_dead_is_gone():
    assert not hasattr(ServiceSupervisor, "_reap_dead")
