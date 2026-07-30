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
import secrets
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
# (host-network Linux would collide). 9050/9051: Tor SOCKS + CONTROL port.
# 9051 matters for defence-in-depth: a registered service on 9051 lands in
# active_service_ports, which the browser SSRF allowlist consults — and the
# browser guard explicitly names 127.0.0.1:9051 (Tor control) as must-block.
BLOCKED_PORTS: FrozenSet[int] = frozenset({8000, 8080, 8088, 9050, 9051})
SUGGESTED_PORTS = "8100-8104"

# Operator helpers that put a published sandbox port on the tailnet
# (2026-07-12). Exposing a service to the network is a HOST action, run by
# the operator — the agent lives in the sandbox and cannot (and should not)
# flip `tailscale serve` itself. The service report points here; the scripts
# self-discover the tailscale CLI + tailnet name. Overridable for a host
# whose ops scripts live elsewhere.
REMOTE_SERVE_SCRIPT = os.environ.get(
    "GHOST_REMOTE_SERVE_SCRIPT", "/Users/vasilis/Data/AI/bin/serve-remote.sh")
REMOTE_UNSERVE_SCRIPT = os.environ.get(
    "GHOST_REMOTE_UNSERVE_SCRIPT",
    "/Users/vasilis/Data/AI/bin/unserve-remote.sh")

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


def is_published_port(port: int, spec: Optional[str] = None,
                      published_ports: Optional[set] = None) -> bool:
    """True when docker publishes ``port`` to the host loopback — reachable at
    ``127.0.0.1:<port>`` on the host and thus a candidate for remote exposure.

    ``published_ports`` is the set docker ACTUALLY published for the live
    container (from the docker manager). When supplied it is authoritative;
    pass it whenever a manager is in reach. It matters in the ≥2-instance case
    (2026-07-15): a second agent — a throwaway for an ablation, the test suite
    — filters the already-taken ports out of its publish list, so it publishes
    NOTHING, yet the configured-range fallback below would still report the
    port as published and point the operator at the FIRST instance's forwarder.
    With no manager in reach, we fall back to the CONFIGURED range (the old
    behaviour): the single-instance common case, where free-at-create ⇒
    published."""
    try:
        p = int(port)
    except Exception:
        return False
    if published_ports is not None:
        return p in published_ports
    return p in set(default_service_ports(spec))


def remote_access_hint(port: int) -> str:
    """Operator recipe for reaching a published sandbox service from another
    device on the tailnet (2026-07-12). Assumes the app binds 0.0.0.0 (the
    manager exports ``HOST``/``PORT`` for it) so docker's bridge-publish
    actually forwards. The exposure step is a HOST action; see
    ``REMOTE_SERVE_SCRIPT``."""
    return (
        f"Remote access: published to the host at http://127.0.0.1:{port}. "
        f"To reach it from another device on your tailnet, run ON THE HOST:\n"
        f"    {REMOTE_SERVE_SCRIPT} {port}\n"
        f"→ serves https://<this-host>.<tailnet>.ts.net:{port}/ "
        f"(tear down: {REMOTE_UNSERVE_SCRIPT} {port}). The app must bind "
        f"0.0.0.0 — HOST=0.0.0.0 and PORT={port} are exported for it."
    )


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


# ── Project-scoped identity + port-lease helpers (2026-07-30, §4G) ────
#
# Services are PROJECT-OWNED resources: registry keys are scoped
# "<project_id>:<name>" (plain "<name>" for project-less utilities), so two
# projects can both run a "web" without colliding, and lifecycle coupling
# (archive/delete stops MY services) reads a first-class field instead of
# grepping the command string. Ports are LEASES the supervisor grants —
# the registry is the single source of truth for "who owns which port".

_PROJ_HINT_RE = re.compile(r"projects/([0-9a-f]{6,32})(?:/|\b)")

# Conservative "the command names its own port" patterns — used when the
# port argument is omitted so the allocator can treat the hardcoded number
# as the PREFERENCE (and substitute it if the lease lands elsewhere).
_CMD_PORT_RES = (
    re.compile(r"\bhttp\.server\s+(\d{4,5})\b"),
    re.compile(r"--port[=\s]+(\d{4,5})\b"),
    re.compile(r"(?<![\w-])-p\s+(\d{4,5})\b"),
    re.compile(r"\bPORT=(\d{4,5})\b"),
    re.compile(r"\bphp\s+-S\s+[\w.]+:(\d{4,5})\b"),
)


def entry_key(project_id, name) -> str:
    """Registry key for a service: ``<project_id>:<name>`` when owned,
    plain ``<name>`` for project-less services (also the legacy shape)."""
    pid = str(project_id or "").strip()
    return f"{pid}:{name}" if pid else str(name)


def split_key(key) -> tuple:
    """Inverse of :func:`entry_key` → ``(project_id | None, name)``."""
    key = str(key or "")
    if ":" in key:
        pid, name = key.split(":", 1)
        return (pid or None), name
    return None, key


def _file_stem(key) -> str:
    """Per-service filename stem (cmd.sh / log / pid). ':' is shell-hostile
    inside the container, so scoped keys map to ``<project>--<name>``."""
    return str(key).replace(":", "--")


def extract_command_port(command) -> Optional[int]:
    """Port literal the command itself binds, per the conservative patterns
    above; None when the command doesn't name one."""
    for rx in _CMD_PORT_RES:
        m = rx.search(str(command or ""))
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                continue
    return None


def substitute_command_port(command, old_port, new_port) -> str:
    """Rewrite a hardcoded port literal when the lease lands elsewhere —
    ``http.server 8102`` must become ``http.server 8103`` or the app binds
    the port the report doesn't name. Word-boundary, all occurrences."""
    return re.sub(rf"\b{int(old_port)}\b", str(int(new_port)),
                  str(command or ""))


class ServiceSupervisor:
    """Manage named detached processes in the sandbox container.

    All methods are SYNC (they shell into the container via
    ``sandbox_manager.execute``) and return operator-readable strings —
    call them through ``asyncio.to_thread`` from tool code. State lives in
    ``<host_workspace>/.services/registry.json`` (host side of the bind
    mount), so it survives agent restarts alongside the container."""

    def __init__(self, sandbox_manager):
        self.sandbox = sandbox_manager
        # Reentrant: restart() holds it across its stop()+start() pair (each
        # of which also acquires it) so a concurrent start of the same name
        # can't slip into the stop→start window (review 2026-07-22).
        self._lock = threading.RLock()

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

    def list_entries(self) -> list:
        """Public read-only snapshot of the registered services
        (``[{name, command, port, workdir, pid, project_id, key, …}]``).
        Added for the project release rehearsal (2026-07-25); ``key`` is the
        registry key — pass it to stop/restart for exact addressing across
        project scopes (2026-07-30)."""
        return [{**e, "key": k} for k, e in self._load().items()]

    # -- container helpers ---------------------------------------------------

    def _published_ports(self) -> Optional[set]:
        """The set docker ACTUALLY published for the live container, or None
        when unknown (no container created yet) — passed to is_published_port
        so a second instance that published nothing doesn't claim the first
        instance's ports (2026-07-15)."""
        try:
            return self.sandbox.published_service_ports()
        except Exception:
            return None

    def _exec(self, cmd: str, timeout: int = 30):
        out, code = self.sandbox.execute(cmd, timeout=timeout)
        return (out or ""), code

    def _pid_alive(self, pid) -> bool:
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return False
        # `kill -0` alone reports ZOMBIES as alive — and in this container
        # every dead orphan IS a zombie unless PID 1 reaps (sleep-infinity
        # never did; docker.py now runs with init=True, but keep this
        # zombie-proof for containers created before that). A zombie launcher
        # made a dead service look "already running", made stop() a no-op,
        # and suppressed start()'s exited-immediately diagnostic (observed
        # live 2026-07-12: three defunct [sh] launchers, 137s of thrash).
        # State = first field after the LAST ')' in /proc/<pid>/stat (comm
        # may contain spaces/parens, so field-splitting is unsafe).
        cmd = (f"sh -c 'kill -0 {pid} 2>/dev/null && "
               f"[ \"$(sed \"s/^.*) //\" /proc/{pid}/stat 2>/dev/null "
               f"| cut -d\" \" -f1)\" != Z ]'")
        _, code = self._exec(cmd, timeout=15)
        return code == 0

    def _port_listening(self, port) -> bool:
        py = ("import socket,sys; s=socket.socket(); s.settimeout(1.5); "
              f"sys.exit(0 if s.connect_ex((\"127.0.0.1\",{int(port)}))==0 "
              "else 1)")
        _, code = self._exec(f"python3 -c {shlex.quote(py)}", timeout=15)
        return code == 0

    def _port_bindable(self, port) -> bool:
        """Can a service BIND 0.0.0.0:<port> in the container right now?
        The allocator's probe — distinct from ``_port_listening`` (which
        asks whether something answers): a service binds 0.0.0.0 (the HOST
        export), so this is exactly the question a failed
        "exited immediately: address already in use" would answer later,
        asked BEFORE the launch instead."""
        py = ("import socket,sys; s=socket.socket(); "
              "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); "
              f"s.bind((\"0.0.0.0\",{int(port)})); s.close(); sys.exit(0)")
        _, code = self._exec(f"python3 -c {shlex.quote(py)}", timeout=15)
        return code == 0

    def _listeners(self) -> list:
        """``[(port, pid|None), …]`` for every TCP listener in the container
        (``ss`` is baked into the sandbox image). Best-effort; [] on any
        failure."""
        out, code = self._exec(
            "sh -c 'ss -H -ltnp 2>/dev/null'", timeout=15)
        if code != 0 or not out:
            return []
        found = []
        for line in out.splitlines():
            cols = line.split()
            if len(cols) < 4:
                continue
            addr = cols[3]
            port_s = addr.rsplit(":", 1)[-1]
            if not port_s.isdigit():
                continue
            m = re.search(r"pid=(\d+)", line)
            found.append((int(port_s), int(m.group(1)) if m else None))
        return found

    def _proc_cmdline(self, pid) -> str:
        try:
            out, code = self._exec(
                f"sh -c 'tr \"\\0\" \" \" < /proc/{int(pid)}/cmdline'",
                timeout=10)
        except (TypeError, ValueError):
            return ""
        return out.strip() if code == 0 else ""

    def _proc_cwd(self, pid) -> str:
        try:
            out, code = self._exec(
                f"readlink /proc/{int(pid)}/cwd", timeout=10)
        except (TypeError, ValueError):
            return ""
        return out.strip() if code == 0 else ""

    def _proc_project_hint(self, pid) -> Optional[str]:
        """Project id inferred from an arbitrary pid's cwd/cmdline — the
        attribution for listeners that bypassed this supervisor."""
        m = _PROJ_HINT_RE.search(self._proc_cwd(pid) or "")
        if not m:
            m = _PROJ_HINT_RE.search(self._proc_cmdline(pid) or "")
        return m.group(1) if m else None

    def _max_services(self) -> int:
        """Effective concurrent-service cap: the published-port range size
        when one is configured (each host-visible service needs a published
        port — the two numbers were silently the same 5 before), else the
        MAX_SERVICES floor."""
        n = len(default_service_ports())
        return n if n > 0 else MAX_SERVICES

    def _container_generation(self) -> Optional[str]:
        """Id of the LIVE sandbox container, or None when unknown (no
        container yet / stub manager). Stamped on registry entries at start;
        a mismatch at liveness time means the recorded pid belonged to a
        PREVIOUS container generation — the process is gone, and any
        same-numbered pid in the new container is an UNRELATED process
        (review 2026-07-22: PID recycling across a container recreate made
        dead services read RUNNING and pointed stop() at innocents)."""
        try:
            cid = getattr(getattr(self.sandbox, "container", None), "id", None)
            return str(cid) if cid else None
        except Exception:  # noqa: BLE001 — a mock may refuse attributes
            return None

    def _entry_alive(self, entry) -> bool:
        """Entry-level liveness: same container generation AND pid alive.
        A stamped entry from a DIFFERENT generation is dead by definition —
        never trust (or signal) its pid number in the new container. Entries
        without a stamp (legacy) or an unknown current generation fall back
        to the plain pid check."""
        if not isinstance(entry, dict):
            return False
        stamped = entry.get("container_id")
        if stamped:
            gen = self._container_generation()
            if gen and gen != stamped:
                return False
        return self._pid_alive(entry.get("pid"))

    def _holder_pid(self, port) -> Optional[int]:
        """Pid LISTENING on <port> inside the container, found via `ss`
        (iproute2, baked into the sandbox image); None when unknown."""
        out, _ = self._exec(
            "sh -c \"ss -H -ltnp 'sport = :%d' 2>/dev/null | "
            "grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2\"" % int(port),
            timeout=10)
        holder = (out or "").strip()
        return int(holder) if holder.isdigit() else None

    def _pid_ownership(self, target, owner) -> Optional[bool]:
        """Does <target> belong to <owner>'s process tree? True when it IS
        <owner>, or its process group / session leader is <owner> (setsid
        made the recorded service pid the leader of both — this also matches
        the historical mis-tracked `$!` orphan, whose real process kept the
        launcher's pgid). False on a POSITIVE mismatch. None when /proc gave
        no answer (process vanished mid-check, stub sandbox) — callers pick
        the safe direction for unknown. pgid/sid are fields 3/4 after the
        last ')' in /proc/<pid>/stat (comm may contain spaces/parens)."""
        try:
            target, owner = int(target), int(owner)
        except (TypeError, ValueError):
            return None
        if target == owner:
            return True
        out, code = self._exec(
            f"sh -c 'sed \"s/^.*) //\" /proc/{target}/stat 2>/dev/null "
            f"| cut -d\" \" -f3,4'", timeout=10)
        fields = (out or "").split()
        if code != 0 or len(fields) < 2:
            return None
        return str(owner) in fields[:2]

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

    # -- port leases ---------------------------------------------------------

    def _port_block_reason(self, port, reg, alive_map,
                           self_key=None) -> Optional[str]:
        """Why ``port`` cannot be granted right now, or None when it can.
        Checks live registry claims first (no container round-trip), then
        the container bind probe — which also catches UNREGISTERED holders,
        the class the old claim-check missed (solar-sim: an execute-leaked
        `python3 server.py &` held 8102 invisibly)."""
        port = int(port)
        for k, e in reg.items():
            if k == self_key or e.get("port") != port:
                continue
            if alive_map.get(k):
                proj = e.get("project_id")
                return (f"claimed by RUNNING service "
                        f"'{e.get('name') or k}'"
                        + (f" of project {proj}" if proj else ""))
        if not self._port_bindable(port):
            holder = self._holder_pid(port)
            if holder is not None:
                extra = ""
                cl = self._proc_cmdline(holder)
                if cl:
                    extra += f" `{cl[:60]}`"
                hint = self._proc_project_hint(holder)
                if hint:
                    extra += f" (project {hint})"
                return (f"held by an UNREGISTERED process — "
                        f"pid {holder}{extra}")
            return "not bindable (held by an unknown process)"
        return None

    def _allocate_port(self, reg, requested=None, self_key=None):
        """Grant a port lease: ``(port | None, notes)``.

        ``requested`` is a PREFERENCE — honored when grantable, otherwise
        the allocator falls through the published range and the notes say
        who was in the way (the report must explain a moved port, never
        silently relocate). Ports claimed by other REGISTERED entries are
        skipped even when those entries are dead — a dead service's port is
        its restart contract — unless nothing else is free, in which case
        the first reassignable dead claim in range order is taken (noted).
        Holders are NEVER killed. Returns ``(None, notes)`` when nothing
        is grantable."""
        notes = []
        alive_map = {k: self._entry_alive(e) for k, e in reg.items()
                     if k != self_key}
        if requested is not None:
            why = self._port_block_reason(requested, reg, alive_map,
                                          self_key)
            if why is None:
                # An explicit request may land on a DEAD entry's reserved
                # port — grant it (the request wins), but SAY SO: silently
                # double-claiming left the port map naming the dead entry
                # as owner while another service ran there (review
                # 2026-07-30).
                _dead_owner = next(
                    (k for k, e in reg.items()
                     if k != self_key and not alive_map.get(k)
                     and e.get("port") == int(requested)), None)
                if _dead_owner is not None:
                    notes.append(
                        f"port {int(requested)} was reserved by dead "
                        f"service '{reg[_dead_owner].get('name') or _dead_owner}'"
                        f" — its restart will be granted a new port")
                return int(requested), notes
            notes.append(f"requested port {int(requested)} is "
                         f"unavailable: {why}")
        span = default_service_ports()
        dead_claims = {int(e["port"]): k for k, e in reg.items()
                       if k != self_key and not alive_map.get(k)
                       and isinstance(e.get("port"), int)}
        deferred = []
        for cand in span:
            if requested is not None and cand == int(requested):
                continue
            if cand in dead_claims:
                deferred.append(cand)     # a dead service's restart contract
                continue
            why = self._port_block_reason(cand, reg, alive_map, self_key)
            if why is None:
                notes.append(f"assigned free port {cand}"
                             + (" instead" if requested is not None else ""))
                return cand, notes
        for cand in deferred:
            why = self._port_block_reason(cand, reg, alive_map, self_key)
            if why is None:
                notes.append(
                    f"assigned port {cand} (was reserved for dead service "
                    f"'{reg[dead_claims[cand]].get('name') or dead_claims[cand]}' — "
                    f"its restart will be granted a new port)")
                return cand, notes
        _span_txt = (os.environ.get("GHOST_SANDBOX_SERVICE_PORTS",
                                    SUGGESTED_PORTS)
                     if span else "none configured")
        notes.append(
            f"no free port in the published range ({_span_txt}) — "
            f"stop a service (action='status' shows the port map) or widen "
            f"GHOST_SANDBOX_SERVICE_PORTS (needs a sandbox recreate, which "
            f"kills running services)")
        return None, notes

    # -- operations ----------------------------------------------------------

    def start(self, name: str, command: str, *, port=None,
              workdir: Optional[str] = None, project_id=None) -> str:
        # Callers (restart, project tooling) may pass a scoped KEY as the
        # name — normalize to (project_id, display-name) first so validation
        # and file stems see the display name. An explicit key's project
        # WINS over the ambient bound project (review 2026-07-30: the
        # ambiguity error tells the model to "use the full key", so the
        # full key must not be silently re-scoped).
        if ":" in str(name or ""):
            _kp, _kn = split_key(name)
            project_id = _kp or project_id
            name = _kn
        # port=0 (or 'none') opts out of the port lease entirely — the rare
        # non-listening worker. Detected before validation (0 is out of the
        # valid bind range by design).
        _portless = port is not None and \
            str(port).strip().lower() in ("0", "none", "no", "off")
        if _portless:
            port = None
        for err in (self._validate_name(name),
                    self._validate_command(command),
                    self._validate_port(port)):
            if err:
                return err
        name = str(name)
        project_id = str(project_id).strip() if project_id else None
        with self._lock:
            reg = self._load()
            # Scope-limited resolution: the bound project's spelling of this
            # name, or a legacy unscoped entry (adopted into the project
            # below) — NEVER another project's service (see
            # _resolve_for_start). Case-variants remain one service
            # (the req-43 'WebOS' vs 'webos' duplicate + port conflict).
            key = self._resolve_for_start(reg, name, project_id)
            if key is not None:
                _, name = split_key(key)
            entry = reg.get(key) if key else None
            if entry and self._entry_alive(entry):
                return (f"Error: service '{name}' is already running "
                        f"(pid {entry.get('pid')}). Use action='restart' "
                        f"to replace it, or 'stop' first.")
            # Re-key a resolved LEGACY (unscoped) entry into the bound
            # project's scope: the replacement entry below records the
            # ownership the old one lacked.
            if key is not None and project_id and ":" not in key:
                reg.pop(key, None)
            key = entry_key(project_id, name)
            stem = _file_stem(key)
            _max = self._max_services()
            alive = sum(1 for k2, e2 in reg.items()
                        if k2 != key and self._entry_alive(e2))
            if alive >= _max:
                return (f"Error: {_max} services already running — "
                        f"stop one first (action='status' to list).")
            # ── Port lease (2026-07-30, §4G) ──
            # Services exist to expose a project's output over HTTP
            # (operator definition), so EVERY start is granted a lease
            # unless explicitly opted out with port=0. The requested port
            # (argument, or a literal the command itself names) is a
            # PREFERENCE; the allocator grants it when free or falls
            # through the published range, explaining any move. An occupied
            # preferred port is no longer an error OR a doomed launch — the
            # service starts on the granted port, $PORT tells the app, and
            # a hardcoded literal is rewritten below.
            _alloc_notes: list = []
            _blocked_cmd_port = None
            if not _portless:
                _requested = port
                if _requested is None:
                    _cmd_port = extract_command_port(command)
                    if _cmd_port is not None:
                        if self._validate_port(_cmd_port) is None:
                            _requested = _cmd_port
                            _alloc_notes.append(
                                f"no port argument — the command names "
                                f"port {_cmd_port}, using it as the "
                                f"preference")
                        else:
                            # The command hardcodes a RESERVED port (8080
                            # etc.) — allocate elsewhere and rewrite the
                            # literal below, or the app binds the reserved
                            # port anyway and only the late not-listening
                            # diagnostic would catch it (review 2026-07-30).
                            _blocked_cmd_port = _cmd_port
                            _alloc_notes.append(
                                f"the command names RESERVED port "
                                f"{_cmd_port} — a free port is assigned "
                                f"instead")
                port, _notes = self._allocate_port(
                    reg, requested=_requested, self_key=key)
                _alloc_notes.extend(_notes)
                if port is None:
                    return ("Error: could not grant a port for "
                            f"'{name}': " + "; ".join(_alloc_notes)
                            + "\n" + self._port_map_text(reg))
                _rewrite_from = None
                if _requested is not None and int(port) != int(_requested):
                    _rewrite_from = _requested
                elif _blocked_cmd_port is not None:
                    _rewrite_from = _blocked_cmd_port
                if _rewrite_from is not None \
                        and re.search(rf"\b{int(_rewrite_from)}\b",
                                      str(command)):
                    command = substitute_command_port(
                        command, _rewrite_from, port)
                    _alloc_notes.append(
                        f"rewrote the hardcoded {_rewrite_from} in the "
                        f"command to {port}")

            # The command ships as a SCRIPT via the bind mount (no quoting
            # hazards), then launches detached: setsid gives it a fresh
            # process group (so stop can kill the whole tree) and the shell
            # exiting re-parents it to the container's PID 1 — out of reach
            # of execute()'s timeout.
            # Export the assigned port so the app can BIND it instead of
            # hardcoding one (2026-07-12). Without this an app that hardcodes,
            # say, 5055 while the manager probes 8100 looks "started but not
            # listening", and the agent thrashes reconciling the mismatch. A
            # well-behaved app does `port = int(os.environ.get("PORT", …))`.
            # Both PORT (the de-facto convention: Flask/Heroku/gunicorn/many
            # frameworks read it) and the explicit GHOST_SERVICE_PORT are set.
            self.host_dir.mkdir(parents=True, exist_ok=True)

            # Resolve + VALIDATE the workdir BEFORE anything launches
            # (2026-07-12). The launch below runs `cd <wd> && setsid …` inside
            # an async subshell, where a cd failure is INVISIBLE: the service
            # log's redirection opens after the cd, so nothing is written
            # anywhere — the model saw only "not listening" and thrashed
            # through identical retries (observed live: workdir
            # '/projects/<id>' — a container-absolute path missing the
            # /workspace prefix — cost a 137s request 3 failed launches).
            # Default workdir: the bound project's workspace when it exists
            # (a project's service serves ITS output — `python3 -m
            # http.server` from /workspace would serve EVERY project's
            # files), else the sandbox root as before.
            if workdir is None and project_id:
                _proj_wd = f"{CONTAINER_WORKDIR}/projects/{project_id}"
                _, _p_code = self._exec(
                    f"test -d {shlex.quote(_proj_wd)}", timeout=10)
                if _p_code == 0:
                    workdir = _proj_wd
            wd = str(workdir or CONTAINER_WORKDIR)
            if not wd.startswith("/"):
                # Relative paths are relative to /workspace by contract.
                wd = f"{CONTAINER_WORKDIR}/{wd}"
            _, _wd_code = self._exec(f"test -d {shlex.quote(wd)}", timeout=10)
            if _wd_code != 0:
                _healed = None
                if not wd.startswith(CONTAINER_WORKDIR):
                    _cand = f"{CONTAINER_WORKDIR}{wd}"
                    _, _h = self._exec(f"test -d {shlex.quote(_cand)}",
                                       timeout=10)
                    if _h == 0:
                        _healed = _cand
                if _healed:
                    logger.info("manage_services: healed workdir %s -> %s",
                                wd, _healed)
                    wd = _healed
                else:
                    return (f"Error: workdir {wd!r} does not exist in the "
                            f"sandbox. Paths are inside the container — use a "
                            f"path relative to {CONTAINER_WORKDIR} (e.g. "
                            f"'projects/<id>') or the full "
                            f"{CONTAINER_WORKDIR}/... path. Nothing was "
                            f"launched.")

            # Strip a redundant leading `cd X && ` when workdir already puts
            # us in X — the model habitually passes BOTH, and the inner
            # relative cd then fails FROM the workdir (X/X doesn't exist),
            # killing the service with a confusing log.
            _cmd_str = str(command).rstrip()
            _m = re.match(r"^\s*cd\s+([^\s;&|]+)\s*&&\s*(.+)$", _cmd_str,
                          re.DOTALL)
            if _m:
                _tgt = _m.group(1).strip("'\"").rstrip("/")
                _tgt_abs = (_tgt if _tgt.startswith("/")
                            else f"{CONTAINER_WORKDIR}/{_tgt}")
                if _tgt_abs == wd.rstrip("/") or \
                        wd.rstrip("/").endswith("/" + _tgt.lstrip("/")):
                    logger.info("manage_services: dropped redundant 'cd %s' "
                                "(workdir=%s already covers it)", _tgt, wd)
                    _cmd_str = _m.group(2).strip()
            # HOST=0.0.0.0: bind the container's forwarding interface, not
            # loopback (2026-07-12). In bridge mode docker publishes host
            # 127.0.0.1:<port> -> container <port>, and a loopback-bound app
            # never receives the forwarded packets — so it's reachable from
            # the in-sandbox browser but NOT from the host or a remote device.
            # 0.0.0.0 is safe: the container is network-isolated. Frameworks
            # that read HOST (uvicorn/gunicorn --host, `flask run`, many Node
            # servers) pick it up; the report + docs tell hand-rolled apps to
            # honour it. This is the code half of "host something remotely-
            # reachable"; `tailscale serve` on the host is the other half.
            _env_prefix = ""
            if port is not None:
                _env_prefix = (f"export PORT={int(port)}\n"
                               f"export GHOST_SERVICE_PORT={int(port)}\n"
                               f"export HOST=0.0.0.0\n"
                               f"export GHOST_SERVICE_HOST=0.0.0.0\n")
            _pidfile_in = f"{CONTAINER_SERVICES_DIR}/{stem}.pid"
            # `exec` the command so the (setsid) shell BECOMES it — same pid,
            # still the session/group leader — so the pid we record IS the real
            # process (not a wrapper the shell forked and then exited from,
            # which left status() showing DEAD and stop() killing nothing). Only
            # for a single SIMPLE command; a compound one (`a && b`, pipes)
            # can't be exec'd, so it runs normally and the port-reclaim fallback
            # + group kill still cover it. Prefer `workdir=` over `cd x && …`.
            _run_line = (("exec " + _cmd_str)
                         if not re.search(r"[;&|\n]", _cmd_str) else _cmd_str)
            # Per-service credential (2026-07-30): a sandbox-hosted app is a
            # participant client of the agent (the chess coach calls
            # /api/game/move for its moves/coaching) but must NEVER hold the
            # master API key. The supervisor mints a token at start, the app
            # reads $GHOST_SERVICE_TOKEN, and the agent's game routes accept
            # it while the registry entry exists — stop() revokes it with
            # the entry. The auth rollout (2026-07-13) silently orphaned
            # this app class; discovered when the released chess coach
            # could no longer reach the agent at all.
            _svc_token = secrets.token_hex(16)
            (self.host_dir / f"{stem}.cmd.sh").write_text(
                "#!/bin/sh\n" + _env_prefix
                + f"export GHOST_SERVICE_TOKEN={_svc_token}\n"
                + f"export GHOST_SERVICE_NAME={shlex.quote(name)}\n"
                # Record THIS shell's pid. Under `setsid` it is the session/
                # group leader, so `kill -- -<pid>` reaps the whole tree. With
                # the exec above, $$ IS the service process — the registry
                # tracks the REAL pid, not the transient `$!` launcher that
                # stop() used to miss, orphaning the service (2026-07-12).
                + f"echo $$ > {_pidfile_in}\n"
                + _run_line + "\n")
            for _stale in (f"{stem}.log", f"{stem}.pid"):
                try:
                    (self.host_dir / _stale).unlink()
                except OSError:
                    pass

            # Persist the entry (with its credential) BEFORE the process
            # launches (2026-07-30 follow-up): a service's FIRST act may be
            # probing the agent with $GHOST_SERVICE_TOKEN, and validation
            # reads THIS registry — saving only after the launch + liveness
            # + port-probe sequence left a ~2-8s window in which the
            # service's own startup probe was 403'd. Observed live:
            # chess-coach-v2 resolves its base URL once at import, got
            # rejected in that window, and permanently fell back to the
            # wrong loopback address. pid=None marks it provisional (reads
            # as dead); every failure return below pops it again.
            reg[key] = {
                "name": name, "command": _cmd_str, "pid": None,
                "port": int(port) if port is not None else None,
                "portless": bool(_portless),
                "token": _svc_token,
                "workdir": wd, "started_at": time.time(),
                "project_id": project_id,
                "container_id": self._container_generation(),
            }
            self._save(reg)

            inner = (
                f"cd {shlex.quote(wd)} && "
                f"setsid nohup sh {CONTAINER_SERVICES_DIR}/{stem}.cmd.sh "
                f"> {CONTAINER_SERVICES_DIR}/{stem}.log 2>&1 & echo $!"
            )
            out, code = self._exec(f"sh -c {shlex.quote(inner)}", timeout=30)
            if code != 0:
                reg.pop(key, None)
                self._save(reg)
                return (f"Error: failed to launch '{name}' "
                        f"(exit {code}): {out.strip()[:400]}")
            # Prefer the pid the script recorded (the real session leader,
            # written to <name>.pid as its first action). Poll briefly for the
            # bind-mounted file to appear.
            pid = None
            pid_path = self.host_dir / f"{stem}.pid"
            for _ in range(20):        # ~2s
                try:
                    _txt = pid_path.read_text().strip()
                    if _txt.isdigit():
                        pid = int(_txt)
                        break
                except OSError:
                    pass
                time.sleep(0.1)
            if pid is None:
                # Fallback: the transient launcher pid from `echo $!`.
                for tok in reversed(out.split()):
                    if tok.isdigit():
                        pid = int(tok)
                        break
            if pid is None:
                reg.pop(key, None)
                self._save(reg)
                return (f"Error: could not determine '{name}' pid "
                        f"from launcher output: {out.strip()[:200]!r}")

            time.sleep(1.2)
            if not self._pid_alive(pid):
                reg.pop(key, None)
                self._save(reg)
                tail = self._log_tail(stem)
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

            # The TCP probe is container-wide — it can't tell WHO answered.
            # If the identified holder is positively foreign (not this pid,
            # not in its process group/session), the app failed to bind and
            # something else answers the port: say so instead of a false
            # "listening ✓" (review 2026-07-22). Unknown holders (probe gave
            # no pid) keep the old benefit of the doubt.
            foreign_holder = None
            if listening:
                _holder = self._holder_pid(port)
                if _holder is not None and \
                        self._pid_ownership(_holder, pid) is False:
                    foreign_holder = _holder

            reg[key] = {
                "name": name, "command": _cmd_str, "pid": pid,
                "port": int(port) if port is not None else None,
                # Persisted opt-out (review 2026-07-30): port=None alone is
                # ambiguous — restart must know this worker CHOSE no port.
                "portless": bool(_portless),
                # The participant credential exported as $GHOST_SERVICE_TOKEN
                # — valid while this entry exists (see valid_service_token).
                "token": _svc_token,
                "workdir": wd, "started_at": time.time(),
                # Ownership (2026-07-30, §4G): the project this service
                # exposes — lifecycle coupling and the briefing read this
                # field; None for project-less utilities.
                "project_id": project_id,
                # Container generation stamp: a recreate invalidates every
                # pid; comparing this at liveness time stops recycled pids
                # from reading as RUNNING (review 2026-07-22).
                "container_id": self._container_generation(),
            }
            self._save(reg)

        # Process alive but the requested port never came up. This is the
        # "app crashed after startup / missing dependency / bound a DIFFERENT
        # port" case — and the cause is almost always sitting in the service
        # log (e.g. `ModuleNotFoundError: No module named 'chess'`). Surface it
        # NOW (2026-07-12): previously this returned a vague "NOT listening
        # yet" with no diagnostics, so the agent proceeded to browse the URL,
        # got ERR_CONNECTION_REFUSED, and only THEN discovered the missing
        # deps — a whole browse→fail→install→restart cycle for an error that
        # was already captured. Same log-tail treatment the immediate-exit
        # branch above already gives.
        # Port answers, but by a process that provably is NOT this service —
        # the classic "address already in use" shape: the app failed to bind
        # and whatever already held the port keeps answering. Without this
        # check the report said "listening ✓" and the agent verified the
        # WRONG process via the browser (review 2026-07-22).
        if port is not None and foreign_holder is not None:
            tail = self._log_tail(stem)
            return (
                f"Service '{name}' started (pid {pid}) BUT port {port} is "
                f"answered by a DIFFERENT process (pid {foreign_holder}), not "
                f"'{name}' — it most likely failed to bind (address already "
                f"in use). What answers on http://127.0.0.1:{port} is NOT "
                f"this service. Stop whatever holds the port (action='status' "
                f"to check registered services) or restart '{name}' on "
                f"another port.\n--- {name} log tail ---\n{tail}"
            )

        if port is not None and listening is False:
            tail = self._log_tail(stem)
            return (
                f"Service '{name}' started (pid {pid}) but nothing is "
                f"listening on port {port} after ~6s — it likely FAILED to "
                f"bind (missing dependency, a crash on import, or the app "
                f"binds a different port). Check the log below BEFORE trying "
                f"to reach it:\n--- {name} log tail ---\n{tail}\n"
                f"Fix the cause (e.g. pip install the missing module, or point "
                f"the app at port {port}), then action='restart' name='{name}'."
            )

        lines = [f"Service '{name}' RUNNING (pid {pid})"
                 + (f" — project {project_id}" if project_id else "")
                 + "."]
        for _n in _alloc_notes:
            lines.append(f"Port lease: {_n}.")
        if port is not None:
            lines.append(
                f"In-sandbox URL: http://127.0.0.1:{port} — the browser and "
                f"execute tools reach it there (listening ✓). The port is "
                f"exported to the app as $PORT.")
            if is_published_port(port, published_ports=self._published_ports()):
                lines.append(remote_access_hint(port))
        lines.append(
            f"Logs: action='logs' name='{name}' · stop: action='stop'. "
            f"It survives across turns until stopped (or the sandbox "
            f"container is recreated).")
        return "\n".join(lines)

    def _kill_pgroup(self, pid) -> None:
        """TERM then KILL a whole process group (setsid made pid the leader),
        with a plain-pid fallback."""
        self._exec(f"sh -c 'kill -TERM -- -{int(pid)} 2>/dev/null || "
                   f"kill -TERM {int(pid)} 2>/dev/null'", timeout=15)
        time.sleep(1.0)
        if self._pid_alive(pid):
            self._exec(f"sh -c 'kill -KILL -- -{int(pid)} 2>/dev/null || "
                       f"kill -KILL {int(pid)} 2>/dev/null'", timeout=15)

    def _kill_port_holder(self, port, owner_pid=None) -> bool:
        """Kill whatever is LISTENING on <port> in the container — the safety
        net for an orphaned service whose tracked pid was wrong (the old `$!`
        bug), so the real process was left bound to the port. When
        ``owner_pid`` is given, a holder that PROVABLY belongs to a different
        process tree (not that pid, nor in its process group/session) is left
        alone — reclaim must never shoot a process this service doesn't own
        (review 2026-07-22: stop('dead-twin') used to kill whoever had since
        taken the port). Unknown ownership (holder vanished mid-check, stub
        exec) still reclaims: the historical mis-tracked orphan sits in the
        recorded pid's process group and DOES resolve, and a vanished holder
        makes the kill a no-op. Best-effort."""
        holder = self._holder_pid(port)
        if holder is None:
            return False
        if owner_pid is not None and \
                self._pid_ownership(holder, owner_pid) is False:
            logger.info(
                "manage_services: NOT reclaiming port %s — holder pid %s "
                "belongs to a different process tree than pid %s",
                port, holder, owner_pid)
            return False
        self._kill_pgroup(holder)
        return True

    def _kill_service(self, entry, others=()) -> bool:
        """Kill one service's process tree and reclaim its port. Returns True
        if anything was actually alive/reclaimed. Also drops its pidfile.
        ``others`` are the OTHER registry entries: when one of them is alive
        and claims the same port, the port legitimately belongs to IT now and
        the reclaim is skipped outright — never TERM/KILL a process a
        different registry entry owns (review 2026-07-22)."""
        pid = entry.get("pid")
        name = entry.get("name")
        port = entry.get("port")
        was_alive = bool(pid) and self._entry_alive(entry)
        if was_alive:
            self._kill_pgroup(pid)
        reclaimed = False
        if port is not None and self._port_listening(port):
            _other_owns = any(
                isinstance(o, dict) and o is not entry
                and o.get("port") == port and self._entry_alive(o)
                for o in others)
            if not _other_owns:
                reclaimed = self._kill_port_holder(port, owner_pid=pid)
        if name:
            _stem = _file_stem(entry_key(entry.get("project_id"), name))
            try:
                (self.host_dir / f"{_stem}.pid").unlink()
            except OSError:
                pass
        return was_alive or reclaimed

    # NB: an auto-reaper for dead registry entries (`_reap_dead`) used to sit
    # here, defined but never called. It was DELETED (review 2026-07-22)
    # rather than wired in: dead entries are load-bearing — restart-after-
    # death is the normal recovery flow and needs the stored command/port/
    # workdir — so auto-removing them on status() would break exactly that.
    # Cleanup stays explicit: per-name stop ("was already dead; removed") or
    # stop-all.

    @staticmethod
    def _resolve_name(reg: Dict[str, dict], name: str,
                      project_id=None) -> Optional[str]:
        """Registered key for ``name`` — project-aware (2026-07-30).

        Order: exact key (callers may hold a scoped key already) → exact
        within the bound project's scope → case-insensitive within scope →
        legacy exact/case-insensitive unscoped key → UNIQUE display-name
        match across all scopes (so "stop web" still works cross-project
        when only one 'web' exists). Ambiguity (the same display name in
        several projects, none of them bound) resolves to None — callers
        list the candidates via ``_matching_keys``.

        Case-insensitivity per req 43 (2026-07-17): 'WebOS' vs 'webos'
        spawned a duplicate service and an "Address already in use" kill
        dance — names differing only by case are one service."""
        name = str(name)
        # Key-shaped names (they carry a ':') address exactly; a PLAIN name
        # must try the bound project's scope FIRST — review 2026-07-30:
        # plain-exact-first let stop('web', project=A) kill a legacy
        # project-less 'web' while A's own 'aaa111:web' kept running.
        if ":" in name and name in reg:
            return name
        if project_id:
            scoped = f"{project_id}:{name}"
            if scoped in reg:
                return scoped
            hits = [k for k in reg if k.lower() == scoped.lower()]
            if len(hits) == 1:
                return hits[0]
        if name in reg:
            return name
        hits = [k for k in reg if k.lower() == name.lower()]
        if len(hits) == 1:
            return hits[0]
        hits = [k for k, e in reg.items()
                if str(e.get("name") or "").lower() == name.lower()]
        return hits[0] if len(hits) == 1 else None

    def _resolve_or_error(self, reg: Dict[str, dict], name: str,
                          project_id=None) -> tuple:
        """``(key, None)`` on resolution, ``(None, error_text)`` otherwise —
        the error DISTINGUISHES ambiguity (several projects own the name;
        candidates listed) from absence, for every verb. Review 2026-07-30:
        only stop() listed candidates; restart/status/logs answered "no
        service named 'web'" for an ambiguous name — actively false, and
        the model's natural next move was to re-create the service."""
        key = self._resolve_name(reg, name, project_id)
        if key is not None:
            return key, None
        _amb = self._matching_keys(reg, str(name))
        if len(_amb) > 1:
            _list = ", ".join(
                f"'{k}' (project {split_key(k)[0] or 'none'})"
                for k in _amb)
            return None, (f"Error: '{name}' is ambiguous — several projects "
                          f"have a service by that name: {_list}. Use the "
                          f"full key (e.g. name='{_amb[0]}').")
        return None, (f"Error: no service named '{name}' "
                      f"(action='status' lists them).")

    @staticmethod
    def _matching_keys(reg: Dict[str, dict], name: str) -> list:
        """All keys whose display name (or key) matches ``name`` case-
        insensitively — for the ambiguity error listing."""
        name = str(name).lower()
        return sorted(k for k, e in reg.items()
                      if k.lower() == name
                      or str(e.get("name") or "").lower() == name)

    def _resolve_for_start(self, reg: Dict[str, dict], name: str,
                           project_id=None) -> Optional[str]:
        """Resolution for START only: scope-limited — the bound project's
        key, or a DEAD legacy unscoped key (the pre-§4G migration path:
        re-keyed into the project by start). NEVER another project's
        service (a global unique match here would let project A's
        `start web` silently replace project B's running 'web'), and
        never a LIVE legacy entry either — that is someone's running
        service, so the bound project creates its own scoped twin instead
        of being blocked by "already running" (review 2026-07-30)."""
        name = str(name)
        primary = entry_key(project_id, name)
        if primary in reg:
            return primary
        hits = [k for k in reg if k.lower() == primary.lower()]
        if len(hits) == 1:
            return hits[0]
        if project_id:
            legacy = name if name in reg else None
            if legacy is None:
                l_hits = [k for k in reg if k.lower() == name.lower()]
                legacy = l_hits[0] if len(l_hits) == 1 else None
            if legacy is not None and ":" not in legacy \
                    and not self._entry_alive(reg[legacy]):
                return legacy
        return None

    def stop(self, name: str, project_id=None) -> str:
        if ":" not in str(name or ""):
            err = self._validate_name(name)
            if err:
                return err
        with self._lock:
            reg = self._load()
            key, _err = self._resolve_or_error(reg, name, project_id)
            if _err:
                return _err
            entry = reg.pop(key, None)
            if entry is None:
                return (f"Error: no service named '{name}' "
                        f"(action='status' lists them).")
            _disp = entry.get("name") or key
            # reg no longer contains the popped entry — the remaining values
            # are exactly the services the reclaim must not harm.
            was_alive = self._kill_service(entry, others=reg.values())
            self._save(reg)
        state = "stopped" if was_alive else "was already dead; removed"
        return (f"Service '{_disp}' {state}. Log kept at "
                f"{CONTAINER_SERVICES_DIR}/{_file_stem(key)}.log")

    def stop_all(self) -> str:
        """Stop EVERY registered service and reclaim their ports — the
        one-command cleanup for accumulated / orphaned services."""
        with self._lock:
            reg = self._load()
            if not reg:
                return "No services registered — nothing to stop."
            killed, cleared = [], []
            for nm, entry in list(reg.items()):
                _others = [e for n2, e in reg.items() if n2 != nm]
                (killed if self._kill_service(entry, others=_others)
                 else cleared).append(nm)
            reg.clear()
            self._save(reg)
        parts = [f"Stopped {len(killed) + len(cleared)} service(s)."]
        if killed:
            parts.append(f"Killed (running/orphaned): {', '.join(killed)}.")
        if cleared:
            parts.append(f"Cleared (already dead): {', '.join(cleared)}.")
        return " ".join(parts)

    def restart(self, name: str, project_id=None) -> str:
        if ":" not in str(name or ""):
            err = self._validate_name(name)
            if err:
                return err
        # Hold the (reentrant) lock across the whole stop→start pair: no
        # concurrent start of the same name can slip into the window, and a
        # failed relaunch can restore the registration atomically (review
        # 2026-07-22 — restart used to pop-and-save via stop() and then
        # re-validate in start(), so any relaunch failure ERASED the
        # service's command/port/workdir).
        with self._lock:
            reg = self._load()
            key, _err = self._resolve_or_error(reg, name, project_id)
            if _err:
                return _err if "ambiguous" in _err else (
                    f"Error: no service named '{name}' to restart "
                    f"(use action='start' with a command).")
            entry = dict(reg[key]) if key in reg else None
            if entry is None:
                return (f"Error: no service named '{name}' to restart "
                        f"(use action='start' with a command).")
            self.stop(key)
            # The stored port is a PREFERENCE like any other: if something
            # took it while the service was down, the allocator moves the
            # restart to a free port and the report says so. A persisted
            # portless opt-out (port=0 at start) STAYS portless — review
            # 2026-07-30: port=None was ambiguous, so restart force-granted
            # a lease to workers that need none (and hard-failed when the
            # range was full).
            out = self.start(key, entry.get("command") or "",
                             port=(0 if entry.get("portless")
                                   else entry.get("port")),
                             workdir=entry.get("workdir"),
                             project_id=entry.get("project_id"))
            if out.startswith("Error:"):
                reg2 = self._load()
                if key not in reg2:
                    reg2[key] = entry     # old pid/stamp → reads DEAD, truthfully
                    self._save(reg2)
                    out += (f"\n(The registration for '{key}' was preserved — "
                            f"fix the cause, then action='restart' "
                            f"name='{key}' again.)")
            return out

    def status(self, name: Optional[str] = None, project_id=None) -> str:
        reg = self._load()
        if name:
            key, _err = self._resolve_or_error(reg, name, project_id)
            if _err:
                return _err
            entries = {key: reg.get(key)}
        else:
            entries = reg
        if not entries:
            out = ("No services registered. Start one with action='start' "
                   "name='...' command='...' — omit 'port' to have a free "
                   "one assigned (exported to the app as $PORT).")
            _orph = self._orphans_text()
            return out + ("\n" + _orph if _orph else "")
        lines = []
        _dead = 0
        for n, e in entries.items():
            alive = self._entry_alive(e)
            if not alive:
                _dead += 1
            state = "RUNNING" if alive else "DEAD (exited or container recreated)"
            _disp = e.get("name") or n
            _proj = e.get("project_id")
            part = (f"- {_disp}"
                    + (f" [project {_proj}]" if _proj else "")
                    + f": {state}, pid {e.get('pid')}")
            if e.get("port") is not None:
                if alive:
                    _lp = self._port_listening(e['port'])
                    part += (f", http://127.0.0.1:{e['port']} "
                             f"{'listening ✓' if _lp else 'NOT listening ✗'}")
                    if _lp and is_published_port(
                            e['port'], published_ports=self._published_ports()):
                        part += (f" · remote: {REMOTE_SERVE_SCRIPT} "
                                 f"{e['port']}")
                else:
                    part += f", port {e['port']} (kept for restart)"
            up = time.time() - float(e.get("started_at") or 0)
            if alive and up > 0:
                part += f", up {int(up // 60)}m"
            part += f" · cmd: {str(e.get('command') or '')[:80]}"
            lines.append(part)
        if _dead and name is None:
            lines.append(f"({_dead} dead — action='restart' brings one back "
                         f"(a taken port is reassigned automatically); "
                         f"action='stop-all' clears them and reclaims ports.)")
        out = "Services:\n" + "\n".join(lines)
        if name is None:
            _map = self._port_map_text(reg)
            if _map:
                out += "\n" + _map
            # Orphans surfaced on EVERY full status, not just the empty-
            # registry branch (review 2026-07-30).
            _orph = self._orphans_text()
            if _orph:
                out += "\n" + _orph
        return out

    def logs(self, name: str, lines: int = 60, project_id=None) -> str:
        if ":" not in str(name or ""):
            err = self._validate_name(name)
            if err:
                return err
        _reg = self._load()
        key, _err = self._resolve_or_error(_reg, name, project_id)
        if key is not None:
            stem = _file_stem(key)
        elif _err and "ambiguous" in _err:
            return _err
        elif (self.host_dir / f"{_file_stem(str(name))}.log").exists():
            stem = _file_stem(str(name))
        else:
            return f"Error: no service (or log) named '{name}'."
        try:
            lines = max(1, min(int(lines), 400))
        except (TypeError, ValueError):
            lines = 60
        tail = self._log_tail(stem, lines=lines)
        return (f"--- {name} log (last {lines} lines) ---\n{tail}")

    # -- reconciliation (2026-07-30, §4G) ------------------------------------

    def reconcile(self) -> Dict[str, list]:
        """READ-ONLY join of the registry against the container's actual
        listeners. Returns ``{alive: [key…], dead: [key…], orphans:
        [{port, pid, cmdline, project_hint}…]}`` where orphans are
        listeners on watched ports (the published range + any registered
        port) that no RUNNING registered service accounts for — the class
        that held port 8102 invisibly in the solar-sim incident. Never
        starts, stops, or mutates anything."""
        reg = self._load()
        alive_map = {k: self._entry_alive(e) for k, e in reg.items()}
        watch = set(default_service_ports())
        watch.update(e.get("port") for e in reg.values()
                     if isinstance(e.get("port"), int))
        alive_ports = {reg[k].get("port") for k, a in alive_map.items() if a}
        orphans = []
        for p, pid in self._listeners():
            if p not in watch or p in alive_ports:
                continue
            o = {"port": p, "pid": pid, "cmdline": "", "project_hint": None}
            if pid:
                o["cmdline"] = self._proc_cmdline(pid)
                o["project_hint"] = self._proc_project_hint(pid)
            orphans.append(o)
        return {"alive": sorted(k for k, a in alive_map.items() if a),
                "dead": sorted(k for k, a in alive_map.items() if not a),
                "orphans": orphans}

    def reconcile_summary(self) -> Optional[str]:
        """One-line boot/awareness summary of :meth:`reconcile`, or None
        when there is nothing worth a line (empty registry, no orphans).
        Deliberately READ-ONLY: services are never auto-restarted on agent
        boot — this line is how the agent KNOWS without acting."""
        try:
            r = self.reconcile()
        except Exception:  # noqa: BLE001 — awareness must never break boot
            return None
        if not r["alive"] and not r["dead"] and not r["orphans"]:
            return None
        bits = []
        if r["alive"] or r["dead"]:
            bits.append(f"{len(r['alive'])} running, {len(r['dead'])} dead "
                        f"(registry kept — restart on demand)")
        for o in r["orphans"]:
            b = f"ORPHAN listener :{o['port']}"
            if o.get("pid"):
                b += f" pid {o['pid']}"
            if o.get("cmdline"):
                b += f" `{o['cmdline'][:40]}`"
            if o.get("project_hint"):
                b += f" (project {o['project_hint']})"
            bits.append(b)
        return "Services: " + "; ".join(bits)

    def _orphans_text(self) -> str:
        """Orphan-listener block for status output; '' when clean."""
        try:
            orphans = self.reconcile()["orphans"]
        except Exception:  # noqa: BLE001
            return ""
        if not orphans:
            return ""
        lines = ["Unregistered listeners (started OUTSIDE manage_services — "
                 "invisible to lifecycle/stop; adopt or kill):"]
        for o in orphans:
            b = f"  :{o['port']}"
            if o.get("pid"):
                b += f" pid {o['pid']}"
            if o.get("cmdline"):
                b += f" `{o['cmdline'][:60]}`"
            if o.get("project_hint"):
                b += f" · project {o['project_hint']}"
            b += (f" — adopt: action='adopt' name='<name>' "
                  f"port={o['port']}")
            lines.append(b)
        return "\n".join(lines)

    def _port_map_text(self, reg=None) -> str:
        """The lease table, one line per published port: RUNNING owner,
        dead reservation, unregistered occupant, or free."""
        reg = self._load() if reg is None else reg
        span = default_service_ports()
        if not span:
            return "Port map: no published range configured."
        listeners = {}
        for p, pid in self._listeners():
            listeners.setdefault(p, pid)
        lines = ["Port map (published range — host-reachable):"]
        for p in span:
            # Several entries can claim one port (an explicit request over
            # a dead reservation, noted at grant time) — the ALIVE claimant
            # is the truth; naming the dead one made the map lie about who
            # actually runs there (review 2026-07-30).
            claims = [(k, e) for k, e in reg.items() if e.get("port") == p]
            claim, alive = None, False
            for k, e in claims:
                if self._entry_alive(e):
                    claim, alive = (k, e), True
                    break
            if claim is None and claims:
                claim = claims[0]
            if claim is not None:
                k, e = claim
                _proj = e.get("project_id")
                _b = (f"  {p}: "
                      + ("RUNNING" if alive else "reserved by dead")
                      + f" '{e.get('name') or k}'"
                      + (f" · project {_proj}" if _proj else ""))
                if not alive and p in listeners:
                    _sq = listeners[p]
                    _b += (f" · currently OCCUPIED by unregistered"
                           + (f" pid {_sq}" if _sq else ""))
                lines.append(_b)
            elif p in listeners:
                pid = listeners[p]
                b = f"  {p}: OCCUPIED by unregistered"
                if pid:
                    b += f" pid {pid}"
                    cl = self._proc_cmdline(pid)
                    if cl:
                        b += f" `{cl[:50]}`"
                    hint = self._proc_project_hint(pid)
                    if hint:
                        b += f" · project {hint}"
                lines.append(b)
            else:
                lines.append(f"  {p}: free")
        return "\n".join(lines)

    def adopt(self, name: str, port, *, project_id=None) -> str:
        """Register an EXISTING unregistered listener as a service — the
        recovery path for daemons that were started outside the supervisor
        (execute `… &` leaks). The process is not touched; it simply
        becomes visible to status/stop/lifecycle. Restart uses the captured
        cmdline, so it is best-effort for adopted entries."""
        for err in (self._validate_name(name), self._validate_port(port)):
            if err:
                return err
        if port is None:
            return "Error: 'port' is required for adopt."
        port = int(port)
        with self._lock:
            reg = self._load()
            holder = self._holder_pid(port)
            if holder is None:
                return (f"Error: nothing is listening on port {port} — "
                        f"adopt attaches an existing unregistered listener "
                        f"to the registry (action='status' shows the port "
                        f"map).")
            for k2, e2 in reg.items():
                if e2.get("port") == port and self._entry_alive(e2):
                    return (f"Error: port {port} already belongs to "
                            f"registered service '{e2.get('name') or k2}'"
                            + (f" of project {e2['project_id']}"
                               if e2.get("project_id") else "") + ".")
            cl = self._proc_cmdline(holder)
            cwd = self._proc_cwd(holder)
            owner = (str(project_id).strip() if project_id
                     else self._proc_project_hint(holder))
            key = entry_key(owner, str(name))
            if key in reg and self._entry_alive(reg[key]):
                return (f"Error: service '{name}' is already registered and "
                        f"RUNNING — pick another name.")
            reg[key] = {
                "name": str(name),
                "command": cl or "(adopted — original command unknown)",
                "pid": holder, "port": port,
                "workdir": cwd or CONTAINER_WORKDIR,
                "started_at": time.time(),
                "project_id": owner,
                "container_id": self._container_generation(),
                "adopted": True,
            }
            self._save(reg)
        return (f"Adopted pid {holder} on port {port} as service '{name}'"
                + (f" (project {owner})" if owner else "")
                + f". It is now visible to status/stop/lifecycle; logs were "
                  f"not captured for its past output. Captured command: "
                  f"{(cl or '(unknown)')[:120]}")

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


def valid_service_token(sandbox_manager, token) -> bool:
    """True when ``token`` matches a REGISTERED service's minted
    ``$GHOST_SERVICE_TOKEN`` (2026-07-30). The registry is the root of
    trust, exactly like ``active_service_ports``: only the supervisor
    mints tokens, and stop()/stop-all revoke by removing the entry. Alive
    or dead doesn't matter — a service that just crashed mid-game should
    not flip auth for its own restart window. Constant-time comparison per
    candidate; iterates ALL entries (no early exit) so timing doesn't
    reveal which entry matched. Never raises; no sandbox → False."""
    try:
        if not token:
            return False
        sup = get_service_supervisor(sandbox_manager)
        if sup is None:
            return False
        tok_b = str(token).encode("utf-8", "ignore")
        ok = False
        for e in sup.list_entries():
            t = e.get("token")
            if t and secrets.compare_digest(
                    str(t).encode("utf-8", "ignore"), tok_b):
                ok = True
        return ok
    except Exception:  # noqa: BLE001 — auth helper: fail closed
        return False


__all__ = [
    "BLOCKED_PORTS", "MAX_SERVICES", "SUGGESTED_PORTS",
    "CONTAINER_SERVICES_DIR", "SERVICES_DIRNAME",
    "ServiceSupervisor", "default_service_ports", "publishable_service_ports",
    "get_service_supervisor", "active_service_ports",
    "is_published_port", "remote_access_hint",
    "REMOTE_SERVE_SCRIPT", "REMOTE_UNSERVE_SCRIPT",
    "entry_key", "split_key", "extract_command_port",
    "substitute_command_port", "valid_service_token",
]
