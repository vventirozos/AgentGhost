"""Supervised long-lived services inside the sandbox container (2026-07-11).

``execute`` wraps every command in ``timeout -k 5s <t>s`` and the container
is PID-isolated, so before this module the agent could BUILD a web app but
never KEEP it running — anything it started died with the exec (600 s cap),
and the 2026-07 feature survey called the gap out explicitly ("build and
host me a dashboard" is implied by the toolset and impossible).

The trick: the sandbox container is persistent (``sleep infinity``), and a
``setsid nohup``-detached process re-parents to the container's PID 1 when
its exec shell exits — so it survives the exec, the turn, and even agent
restarts (the container outlives the agent process). This module makes that
a *supervised* capability instead of an accident:

* one command script + pidfile + logfile per service under
  ``/workspace/.services/`` (host-visible via the bind mount, so the
  registry and logs are plain files the agent/operator can read);
* start / stop / restart / status / logs with liveness (``kill -0``) and an
  optional TCP port probe;
* hard rails: bounded service count, name whitelist, and the
  mock-server guard — ports 8000/8088 (the agent's own API / upstream LLM
  — the sandbox-loopback blind spot), 8080 (NetMon) and 9050 (Tor) are
  refused, as is any command that references 127.0.0.1:8000/:8088.

Reachability: in-sandbox consumers (the ``browser`` tool's embedded runner,
``execute``) reach a service at ``http://127.0.0.1:<port>`` — the browser's
SSRF guard admits loopback ports that appear in this registry (see
``active_service_ports``). Operator access from the HOST depends on the
sandbox network mode: ``host`` mode (Linux default) exposes the port
directly; ``bridge`` mode (macOS default) publishes the
``GHOST_SANDBOX_SERVICE_PORTS`` range (default 8100-8104) to
``127.0.0.1`` when the container is (re)created.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import threading
import time
from pathlib import Path
from typing import Dict, FrozenSet, Optional

logger = logging.getLogger("GhostAgent")

SERVICES_DIRNAME = ".services"
CONTAINER_SERVICES_DIR = "/workspace/.services"
CONTAINER_WORKDIR = "/workspace"

# 8000/8088: the agent's API / upstream LLM — inside the sandbox these are
# NOT reachable and standing in for them is the forbidden mock-server
# pathology (see tools/execute.py's guard). 8080: NetMon on this host
# (host-network Linux would collide). 9050: Tor (same rationale).
BLOCKED_PORTS: FrozenSet[int] = frozenset({8000, 8080, 8088, 9050})
SUGGESTED_PORTS = "8100-8104"

MAX_SERVICES = 5
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,31}$")
_FORBIDDEN_CMD_RE = re.compile(
    r"(?:127\.0\.0\.1|localhost|0\.0\.0\.0)\s*:\s*(?:8000|8088)\b")
_MAX_CMD_CHARS = 4000
_LOG_TAIL_BYTES = 8 * 1024


def _port_free(port: int) -> bool:
    """Best-effort: can 127.0.0.1:<port> be bound right now? Used to filter
    the publish list — see ``publishable_service_ports``."""
    import socket
    s = socket.socket()
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", int(port)))
        return True
    except OSError:
        return False
    finally:
        try:
            s.close()
        except OSError:
            pass


def publishable_service_ports(spec: Optional[str] = None) -> list:
    """``default_service_ports`` minus any port already taken on the host.

    Publishing a taken port makes ``containers.run`` fail OUTRIGHT (and
    leaves a created-but-unstarted container behind), so we filter first.
    This is not a corner case: a second agent instance — a throwaway for an
    ablation, or the test suite — cannot publish the same fixed host ports as
    the one already running, and must degrade to no-published-ports rather
    than failing to get a sandbox at all. The (rarer) bind race between this
    check and ``containers.run`` is handled by docker.py's retry.
    """
    return [p for p in default_service_ports(spec) if _port_free(p)]


def default_service_ports(spec: Optional[str] = None) -> list:
    """Parse a ports spec like ``"8100-8104"`` or ``"8100,8101"`` into a
    bounded list of ints. Empty/invalid → []. Shared by docker.py's
    publish step and the docs."""
    if spec is None:
        spec = os.environ.get("GHOST_SANDBOX_SERVICE_PORTS", SUGGESTED_PORTS)
    spec = (spec or "").strip()
    if not spec:
        return []
    ports: list = []
    try:
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-", 1)
                lo, hi = int(lo), int(hi)
                if lo > hi or hi - lo > 32:
                    continue
                ports.extend(range(lo, hi + 1))
            else:
                ports.append(int(part))
    except (TypeError, ValueError):
        return []
    return sorted({p for p in ports
                   if 1024 <= p <= 65535 and p not in BLOCKED_PORTS})[:32]


class ServiceSupervisor:
    """Manage named detached processes in the sandbox container.

    All methods are SYNC (they shell into the container via
    ``sandbox_manager.execute``) and return operator-readable strings —
    call them through ``asyncio.to_thread`` from tool code. State lives in
    ``<host_workspace>/.services/registry.json`` (host side of the bind
    mount), so it survives agent restarts alongside the container."""

    def __init__(self, sandbox_manager):
        self.sandbox = sandbox_manager
        self._lock = threading.Lock()

    # -- paths / registry ---------------------------------------------------

    @property
    def host_dir(self) -> Path:
        return Path(self.sandbox.host_workspace) / SERVICES_DIRNAME

    @property
    def _registry_path(self) -> Path:
        return self.host_dir / "registry.json"

    def _load(self) -> Dict[str, dict]:
        try:
            data = json.loads(self._registry_path.read_text())
            return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001 — absent/corrupt → empty
            return {}

    def _save(self, reg: Dict[str, dict]) -> None:
        self.host_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._registry_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(reg, indent=2))
        os.replace(tmp, self._registry_path)

    # -- container helpers ---------------------------------------------------

    def _exec(self, cmd: str, timeout: int = 30):
        out, code = self.sandbox.execute(cmd, timeout=timeout)
        return (out or ""), code

    def _pid_alive(self, pid) -> bool:
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return False
        _, code = self._exec(f"sh -c 'kill -0 {pid} 2>/dev/null'", timeout=15)
        return code == 0

    def _port_listening(self, port) -> bool:
        py = ("import socket,sys; s=socket.socket(); s.settimeout(1.5); "
              f"sys.exit(0 if s.connect_ex((\"127.0.0.1\",{int(port)}))==0 "
              "else 1)")
        _, code = self._exec(f"python3 -c {shlex.quote(py)}", timeout=15)
        return code == 0

    def _log_tail(self, name: str, lines: int = 25) -> str:
        try:
            raw = (self.host_dir / f"{name}.log").read_bytes()
            text = raw[-_LOG_TAIL_BYTES:].decode("utf-8", "replace")
            return "\n".join(text.splitlines()[-max(1, int(lines)):])
        except Exception:  # noqa: BLE001
            return "(no log output)"

    # -- validation ----------------------------------------------------------

    @staticmethod
    def _validate_name(name) -> Optional[str]:
        if not name or not _NAME_RE.match(str(name)):
            return ("Error: invalid service name — use 1-32 chars, letters/"
                    "digits/dash/underscore, starting with a letter "
                    "(e.g. 'dashboard').")
        return None

    @staticmethod
    def _validate_command(command) -> Optional[str]:
        if not command or not str(command).strip():
            return "Error: 'command' is required for start."
        if len(str(command)) > _MAX_CMD_CHARS:
            return f"Error: command too long (>{_MAX_CMD_CHARS} chars)."
        if _FORBIDDEN_CMD_RE.search(str(command)):
            return ("Error: this command references 127.0.0.1:8000/:8088 — "
                    "inside the sandbox those are NOT the agent API or the "
                    "upstream LLM (separate loopback), and standing in a "
                    "mock for them is forbidden. Host your service on "
                    f"another port (suggested: {SUGGESTED_PORTS}).")
        return None

    @staticmethod
    def _validate_port(port) -> Optional[str]:
        if port is None:
            return None
        try:
            port = int(port)
        except (TypeError, ValueError):
            return f"Error: port must be an integer, got {port!r}."
        if not (1024 <= port <= 65535):
            return f"Error: port {port} out of range (1024-65535)."
        if port in BLOCKED_PORTS:
            return (f"Error: port {port} is reserved (agent API / upstream "
                    f"LLM / NetMon / Tor). Use another port "
                    f"(suggested: {SUGGESTED_PORTS}).")
        return None

    # -- operations ----------------------------------------------------------

    def start(self, name: str, command: str, *, port=None,
              workdir: Optional[str] = None) -> str:
        for err in (self._validate_name(name),
                    self._validate_command(command),
                    self._validate_port(port)):
            if err:
                return err
        name = str(name)
        with self._lock:
            reg = self._load()
            entry = reg.get(name)
            if entry and self._pid_alive(entry.get("pid")):
                return (f"Error: service '{name}' is already running "
                        f"(pid {entry.get('pid')}). Use action='restart' "
                        f"to replace it, or 'stop' first.")
            alive = sum(1 for e in reg.values()
                        if e.get("name") != name
                        and self._pid_alive(e.get("pid")))
            if alive >= MAX_SERVICES:
                return (f"Error: {MAX_SERVICES} services already running — "
                        f"stop one first (action='status' to list).")

            # The command ships as a SCRIPT via the bind mount (no quoting
            # hazards), then launches detached: setsid gives it a fresh
            # process group (so stop can kill the whole tree) and the shell
            # exiting re-parents it to the container's PID 1 — out of reach
            # of execute()'s timeout.
            self.host_dir.mkdir(parents=True, exist_ok=True)
            (self.host_dir / f"{name}.cmd.sh").write_text(
                "#!/bin/sh\n" + str(command).rstrip() + "\n")
            try:
                (self.host_dir / f"{name}.log").unlink()
            except OSError:
                pass

            wd = str(workdir or CONTAINER_WORKDIR)
            inner = (
                f"cd {shlex.quote(wd)} && "
                f"setsid nohup sh {CONTAINER_SERVICES_DIR}/{name}.cmd.sh "
                f"> {CONTAINER_SERVICES_DIR}/{name}.log 2>&1 & echo $!"
            )
            out, code = self._exec(f"sh -c {shlex.quote(inner)}", timeout=30)
            if code != 0:
                return (f"Error: failed to launch '{name}' "
                        f"(exit {code}): {out.strip()[:400]}")
            pid = None
            for tok in reversed(out.split()):
                if tok.isdigit():
                    pid = int(tok)
                    break
            if pid is None:
                return (f"Error: could not determine '{name}' pid "
                        f"from launcher output: {out.strip()[:200]!r}")

            time.sleep(1.2)
            if not self._pid_alive(pid):
                tail = self._log_tail(name)
                return (f"Error: service '{name}' exited immediately.\n"
                        f"--- log tail ---\n{tail}")

            listening = None
            if port is not None:
                for _ in range(6):
                    if self._port_listening(port):
                        listening = True
                        break
                    time.sleep(1.0)
                else:
                    listening = False

            reg[name] = {
                "name": name, "command": str(command), "pid": pid,
                "port": int(port) if port is not None else None,
                "workdir": wd, "started_at": time.time(),
            }
            self._save(reg)

        lines = [f"Service '{name}' RUNNING (pid {pid})."]
        if port is not None:
            lines.append(
                f"In-sandbox URL: http://127.0.0.1:{port} — the browser and "
                f"execute tools reach it there"
                + (" (listening ✓)." if listening
                   else " (NOT listening yet — check action='logs' if it "
                        "stays down)."))
        lines.append(
            f"Logs: action='logs' name='{name}' · stop: action='stop'. "
            f"It survives across turns until stopped (or the sandbox "
            f"container is recreated).")
        return "\n".join(lines)

    def stop(self, name: str) -> str:
        err = self._validate_name(name)
        if err:
            return err
        name = str(name)
        with self._lock:
            reg = self._load()
            entry = reg.pop(name, None)
            if entry is None:
                return (f"Error: no service named '{name}' "
                        f"(action='status' lists them).")
            pid = entry.get("pid")
            was_alive = self._pid_alive(pid)
            if was_alive:
                # TERM the whole process group (setsid made pid the group
                # leader), escalate to KILL, with a plain-pid fallback.
                self._exec(f"sh -c 'kill -TERM -- -{int(pid)} "
                           f"2>/dev/null || kill -TERM {int(pid)} "
                           f"2>/dev/null'", timeout=15)
                time.sleep(1.0)
                if self._pid_alive(pid):
                    self._exec(f"sh -c 'kill -KILL -- -{int(pid)} "
                               f"2>/dev/null || kill -KILL {int(pid)} "
                               f"2>/dev/null'", timeout=15)
            self._save(reg)
        state = "stopped" if was_alive else "was already dead; removed"
        return (f"Service '{name}' {state}. Log kept at "
                f"{CONTAINER_SERVICES_DIR}/{name}.log")

    def restart(self, name: str) -> str:
        err = self._validate_name(name)
        if err:
            return err
        entry = self._load().get(str(name))
        if entry is None:
            return (f"Error: no service named '{name}' to restart "
                    f"(use action='start' with a command).")
        self.stop(name)
        return self.start(name, entry.get("command") or "",
                          port=entry.get("port"),
                          workdir=entry.get("workdir"))

    def status(self, name: Optional[str] = None) -> str:
        reg = self._load()
        if name:
            entries = {str(name): reg.get(str(name))}
            if entries[str(name)] is None:
                return f"Error: no service named '{name}'."
        else:
            entries = reg
        if not entries:
            return ("No services registered. Start one with action='start' "
                    "name='...' command='...' port=... "
                    f"(suggested ports: {SUGGESTED_PORTS}).")
        lines = []
        for n, e in entries.items():
            alive = self._pid_alive(e.get("pid"))
            state = "RUNNING" if alive else "DEAD (exited or container recreated)"
            part = f"- {n}: {state}, pid {e.get('pid')}"
            if e.get("port") is not None:
                if alive:
                    part += (f", http://127.0.0.1:{e['port']} "
                             f"{'listening ✓' if self._port_listening(e['port']) else 'NOT listening ✗'}")
                else:
                    part += f", port {e['port']}"
            up = time.time() - float(e.get("started_at") or 0)
            if alive and up > 0:
                part += f", up {int(up // 60)}m"
            part += f" · cmd: {str(e.get('command') or '')[:80]}"
            lines.append(part)
        return "Services:\n" + "\n".join(lines)

    def logs(self, name: str, lines: int = 60) -> str:
        err = self._validate_name(name)
        if err:
            return err
        if self._load().get(str(name)) is None and not (
                self.host_dir / f"{name}.log").exists():
            return f"Error: no service (or log) named '{name}'."
        try:
            lines = max(1, min(int(lines), 400))
        except (TypeError, ValueError):
            lines = 60
        tail = self._log_tail(str(name), lines=lines)
        return (f"--- {name} log (last {lines} lines) ---\n{tail}")

    def active_ports(self) -> FrozenSet[int]:
        """Ports of REGISTERED services (alive or not — a service that just
        crashed shouldn't flip the browser guard mid-run). Used by the
        browser tool's SSRF allowlist; registry-driven, so only ports the
        agent itself opened via this supervisor are admitted."""
        out = set()
        for e in self._load().values():
            p = e.get("port")
            if isinstance(p, int) and 1024 <= p <= 65535 \
                    and p not in BLOCKED_PORTS:
                out.add(p)
        return frozenset(out)


def get_service_supervisor(sandbox_manager) -> Optional[ServiceSupervisor]:
    """Get-or-create the supervisor cached on the sandbox manager (its
    owner). None when no sandbox is attached."""
    if sandbox_manager is None:
        return None
    sup = getattr(sandbox_manager, "_service_supervisor", None)
    if sup is None:
        sup = ServiceSupervisor(sandbox_manager)
        try:
            sandbox_manager._service_supervisor = sup
        except Exception:  # noqa: BLE001 — a mock may refuse attributes
            pass
    return sup


def active_service_ports(sandbox_manager) -> FrozenSet[int]:
    """Registry-driven loopback-port allowlist for the browser SSRF guard.
    Never raises; no sandbox → empty."""
    try:
        sup = get_service_supervisor(sandbox_manager)
        return sup.active_ports() if sup is not None else frozenset()
    except Exception:  # noqa: BLE001
        return frozenset()


__all__ = [
    "BLOCKED_PORTS", "MAX_SERVICES", "SUGGESTED_PORTS",
    "CONTAINER_SERVICES_DIR", "SERVICES_DIRNAME",
    "ServiceSupervisor", "default_service_ports", "publishable_service_ports",
    "get_service_supervisor", "active_service_ports",
]
