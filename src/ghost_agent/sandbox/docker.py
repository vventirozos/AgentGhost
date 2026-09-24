import logging
import os
import shlex
import re
import threading
import time
from pathlib import Path
from ..utils.logging import Icons, pretty_log

logger = logging.getLogger("GhostAgent")


def _owner_boot_id() -> str:
    """A stamp that changes across reboots, so a container labelled with
    a PID from BEFORE a reboot is never mistaken for a live owner (PIDs
    are reused — the postgres stale-lock incident on this box was exactly
    that). Falls back to the empty string, which reads as "unknown owner"
    and therefore as reapable, matching the pre-label behaviour."""
    try:
        import platform
        import subprocess
        if platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "kern.boottime"],
                                 capture_output=True, timeout=5)
            return (out.stdout or b"").decode("utf-8", "replace").strip()[:64]
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()[:64]
    except Exception:  # noqa: BLE001
        return ""


def _pid_is_live(pid: str) -> bool:
    """True when `pid` names a process on THIS host right now."""
    try:
        n = int(str(pid).strip())
    except (TypeError, ValueError):
        return False
    if n <= 0:
        return False
    try:
        os.kill(n, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True          # alive, owned by someone else
    except Exception:  # noqa: BLE001
        return False

CONTAINER_NAME = "ghost-agent-sandbox"
CONTAINER_WORKDIR = "/workspace"

#: The capabilities the sandbox KEEPS when `cap_drop=ALL` (the §4KF default).
#: Everything the image's jobs need as ROOT: package files (CHOWN,
#: DAC_OVERRIDE, FOWNER, FSETID), privilege DROPS to `_apt`/`debian-tor`
#: (SETUID, SETGID), signalling supervised children (KILL). Deliberately
#: absent: NET_RAW, NET_BIND_SERVICE, MKNOD, SYS_CHROOT, SETPCAP, SETFCAP,
#: AUDIT_WRITE, and of course NET_ADMIN (the egress rules go in through a
#: privileged exec so the model cannot undo them).
SANDBOX_KEPT_CAPS = ("CHOWN", "DAC_OVERRIDE", "FOWNER", "FSETID", "SETGID", "SETUID", "KILL")

#: The everyday CLI set added in provisioning marker v10 (§4KG): apt package
#: names, and the binaries that PROVE they landed (the marker is written only
#: after every one answers `command -v`). Three copies must agree — the v10
#: delta command, the full-provision apt line, and sandbox/Dockerfile — and
#: tests/test_4kg_sandbox_tools.py holds them to these two tuples.
SANDBOX_TOOL_PACKAGES = tuple("netcat-openbsd socat jq unzip zip tree nano bc gawk rsync pkg-config telnet whois p7zip-full poppler-utils ffmpeg imagemagick stockfish".split())
SANDBOX_TOOL_BINARIES = tuple("nc socat jq unzip zip tree nano bc gawk rsync pkg-config telnet whois 7z pdftotext ffmpeg convert stockfish".split())

# Client-side deadline (seconds) for a container exec when the caller gives no
# explicit one. docker-py's exec socket read blocks in poll() with NO timeout
# (verified in 7.1.0), so a wedged daemon would otherwise hang the calling
# thread forever — and the provision execs run while holding self._lock, which
# would wedge EVERY other turn's execute. This ceiling is generous on purpose:
# it must never fire for a legitimately-slow install (apt/pip/playwright), only
# for a genuinely stuck daemon. The per-command exec passes its own tighter
# deadline (the in-container `timeout Ns` budget + grace).
_EXEC_DAEMON_DEADLINE_S = float(os.environ.get("GHOST_EXEC_DAEMON_DEADLINE", "1200") or 1200)


# Grace added on top of a provision install's own in-container ``timeout N``
# budget when choosing its CLIENT-side wedge deadline. The client deadline MUST
# exceed the in-container cap: otherwise a healthy-but-slow install trips the
# client deadline first and is mis-diagnosed as a wedged daemon (which aborts
# provisioning, arms the backoff, and lets the abandoned worker keep installing
# so the retry double-installs). 300s covers docker-py stream drain plus the
# SIGTERM->SIGKILL grace ``timeout`` itself uses. Env-tunable; never below 0.
_PROVISION_EXEC_GRACE_S = max(
    0.0, float(os.environ.get("GHOST_PROVISION_EXEC_GRACE", "300") or 300))

_TIMEOUT_PREFIX_RE = re.compile(r"^\s*timeout\s+(?:-k\s*\S+\s+)?(\d+)")


def _provision_deadline_s(cmd) -> float:
    """CLIENT-side wedge deadline for a provision exec: its in-container
    ``timeout N`` budget + grace, so a legitimately-slow install can never
    trip the client deadline before its own in-container cap. A command with
    no ``timeout`` prefix (the fast sudoers/marker/probe execs) falls back to
    the module wedge default. The returned deadline is GUARANTEED to strictly
    exceed the parsed cap — the inversion this fixes cannot silently return."""
    m = _TIMEOUT_PREFIX_RE.match(str(cmd or ""))
    if not m:
        return _EXEC_DAEMON_DEADLINE_S
    cap = float(m.group(1))
    deadline = cap + _PROVISION_EXEC_GRACE_S
    # Invariant: never hand back a deadline that does not clear the cap.
    if deadline <= cap:
        deadline = cap + 1.0
    return deadline


# Ceiling on how much of a CLASSIC (non-jobs) exec's output the agent holds in
# RAM at once, mirroring sandbox/jobs.py's _LOG_READ_CAP. docker-py's own exec
# buffer is UNBOUNDED (it b"".join()s the whole stream), so this is strictly
# safer than the path it replaces; head+tail keeps both ends of a pathological
# output readable. Env-tunable (MB); floored at 1 KB. Read at import (docker.py
# convention — see _EXEC_DAEMON_DEADLINE_S) so no per-exec os.environ read lands
# on the hot path.
_CLASSIC_EXEC_RAM_CAP = max(
    1024, int(float(os.environ.get("GHOST_SANDBOX_EXEC_RAM_CAP_MB", "32") or 32)
              * 1024 * 1024))

# Kill switch for the classic-path output streamer. Default ON;
# GHOST_SANDBOX_EXEC_STREAM=0 reverts to buffering the whole exec output in
# agent RAM (the pre-fix behaviour), so a live regression can be disarmed with a
# restart and no redeploy. Module-load constant, same reason as the cap above.
_EXEC_STREAM_ENABLED = str(
    os.environ.get("GHOST_SANDBOX_EXEC_STREAM", "1")).strip().lower() \
    not in ("0", "false", "no", "off")


def _drain_stream_bounded(stream, cap, demux=False):
    """Iterate a docker exec byte stream into AT MOST ~cap bytes, kept as
    head + tail, returning ``(bytes, total_bytes_seen)``.

    Byte-IDENTICAL to buffering the whole stream when total <= cap (the normal
    path): the very frames docker-py's ``consume_socket_output`` would
    ``b"".join`` are concatenated here. Above cap, RAM is bounded to ~cap and
    the middle is elided with a marker — only the pathological-output regime
    differs, which is the entire point.
    """
    half = max(1, cap // 2)
    head = bytearray()
    tail = bytearray()
    total = 0
    overflow = False
    for chunk in stream:
        if not chunk:
            continue
        if demux:
            # (stdout, stderr) frames — combine, matching demux=False output.
            if isinstance(chunk, (tuple, list)):
                so, se = chunk
                chunk = (so or b"") + (se or b"")
            if not chunk:
                continue
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8", "replace")
        total += len(chunk)
        if not overflow:
            head += chunk
            if len(head) > cap:
                overflow = True
                spill = bytes(head[half:])
                del head[half:]
                tail += spill
        else:
            tail += chunk
        if overflow and len(tail) > 2 * half:
            del tail[:len(tail) - half]
    if not overflow:
        return bytes(head), total
    if len(tail) > half:
        del tail[:len(tail) - half]
    omitted = total - len(head) - len(tail)
    marker = (f"\n[... {omitted} bytes elided — output exceeded the "
              f"{cap}-byte agent RAM cap; head+tail kept ...]\n").encode()
    return bytes(head) + marker + bytes(tail), total


class SandboxDaemonTimeout(Exception):
    """A container exec exceeded its CLIENT-SIDE deadline — the docker daemon
    is likely wedged. Raised instead of blocking forever so the caller
    releases self._lock and the agent surfaces a clear error."""


class DockerSandbox:
    # Per-container-generation state, reset whenever a container is
    # (re)created. Class-level defaults so test stubs built via __new__
    # inherit them.
    #   _env_verified — the marker+chromium checks passed once for the
    #     current generation; skip re-probing them on every command.
    #   _tor_attempted — the Tor-only egress enforcement (§4FU) was already
    #     run for this generation. ⚠ A "generation" is a container START,
    #     not a container (§4FW): `docker start` gives a stopped container a
    #     FRESH network namespace — the iptables rules are gone — and none
    #     of its processes, so the in-container Tor is not running either.
    #     Every path that starts or recreates the container must reset this
    #     and go through `_enforce_egress_once`.
    #   _provision_backoff_until — after a failed provision, no reinstall
    #     before this wall-clock time; prevents a failing mirror from
    #     triggering a fresh multi-minute install on every command.
    _env_verified = False
    _tor_attempted = False
    _provision_backoff_until = 0.0
    #: The egress policy, set for real by __init__ (pinned in
    #: tests/test_sandbox_resume_egress.py). The class default exists for the
    #: same reason the flags above do — test stubs built via __new__ — and
    #: because §4FW made the RESUME path read it, where the review stubs of
    #: `_try_resume_stopped` never had it: an AttributeError there would
    #: abort a resume that used to work.
    tor_proxy = None

    # Capability flag read by tools/execute.py (see _run_in_sandbox): this
    # manager implements `execute_promotable`, so a command that outruns its
    # budget while still working is DETACHED as a job instead of killed. It
    # is an explicit opt-in rather than a hasattr probe because a MagicMock
    # stub answers hasattr for anything.
    supports_job_promotion = True

    def __init__(self, host_workspace: Path, tor_proxy: str = None,
                 network: str = None):
        import hashlib
        short_hash = hashlib.md5(str(host_workspace.absolute()).encode()).hexdigest()[:8]
        self.container_name = f"ghost-agent-sandbox-{short_hash}"
        try:
            import docker
            from docker.errors import NotFound, APIError
            self.docker_lib = docker
            self.NotFound = NotFound
            self.APIError = APIError
        except ImportError:
            logger.error("Docker library not found. pip install docker")
            raise

        try:
            self.client = self.docker_lib.from_env()
            self.client.ping()
        except self.docker_lib.errors.DockerException as handle_err:
            # Carry the half-built client on the exception so a caller
            # that never receives `self` can still close it. A client
            # constructed and then abandoned holds ~11 unix sockets, and
            # a nightly job that builds one per attempt reaches EMFILE
            # in days (§4BO).
            try:
                handle_err.client = getattr(self, "client", None)
            except Exception:  # noqa: BLE001
                pass
            import sys
            import os
            if sys.platform == "darwin":
                orb_sock = os.path.expanduser("~/.orbstack/run/docker.sock")
                target_sock = os.path.expanduser("~/.docker/run/docker.sock") # alternative fallback
                
                sock_to_use = orb_sock if os.path.exists(orb_sock) else target_sock if os.path.exists(target_sock) else None
                
                if sock_to_use:
                    try:
                        self.client = self.docker_lib.DockerClient(base_url=f"unix://{sock_to_use}")
                        self.client.ping()
                    except:
                        # The fallback client is OURS alone — the caller
                        # only ever sees handle_err.client (the from_env
                        # one), so an unping-able fallback must be closed
                        # HERE or its ~11 sockets leak on every failed
                        # construction (recurring since the lazy re-init
                        # path retries per backoff window).
                        _fb = self.__dict__.get("client")
                        if _fb is not None and _fb is not getattr(handle_err, "client", None):
                            try:
                                _fb.close()
                            except Exception:  # noqa: BLE001
                                pass
                        raise handle_err
                else:
                    raise handle_err
            else:
                raise handle_err
        self.host_workspace = host_workspace.absolute()
        self.tor_proxy = tor_proxy
        # EXPLICIT network mode for THIS manager, overriding the process-wide
        # GHOST_SANDBOX_NETWORK. An isolated replay needs `none`, and the env
        # var is the wrong lever for it: it is process-global, so a concurrent
        # live turn creating its own container would inherit the replay's
        # isolation (§4CL S1). Values: "host" | "bridge" | "none"; anything
        # else (including None) falls through to the env/platform default.
        _net = str(network or "").strip().lower()
        self.network_override = _net if _net in ("host", "bridge", "none") else None
        self.container = None
        self.image = "python:3.11-slim-bookworm"
        # The service ports docker ACTUALLY published for the live container
        # (loopback bridge-publish), set at (re)create. None until then →
        # is_published_port falls back to the configured range (2026-07-15).
        self._published_service_ports = None
        # Serializes ensure_running across threads. execute() is run via
        # asyncio.to_thread, so concurrent tool calls hit ensure_running
        # on different threads; without this they race container
        # creation/provisioning (409 name conflict, double apt/pip/
        # playwright install, racing image commit). docker-py client
        # models are not thread-safe either.
        self._lock = threading.Lock()

        # Readiness TTL: execute() calls ensure_running() before EVERY command,
        # and the full readiness probe is 3 docker round-trips (reload + host
        # touch/stat + echo) ≈ 100-400ms, serialized under _lock. When a
        # command has demonstrably just succeeded, the container is ready — so
        # a probe within _READY_TTL_S of the last confirmed-good exec is
        # skipped. Any exec failure / OCI error clears the stamp
        # (`invalidate_ready`) so the recreate path still triggers promptly.
        self._last_ready_ok = 0.0
        self._READY_TTL_S = 8.0

        pretty_log("Sandbox Init", f"Mounting {self.host_workspace} -> {CONTAINER_WORKDIR}", icon=Icons.SANDBOX_BOX)

    def binds_host_netns(self) -> bool:
        """True when the sandbox shares the HOST network namespace (docker
        ``--network host``): an explicit choice only, since §4KF. In that mode a
        service the agent hosts binds a real host port, so exporting
        ``HOST=0.0.0.0`` exposes it LAN-wide unauthenticated — sandbox.services
        consults this to bind loopback instead. Mirrors the create-time logic
        (per-manager override → GHOST_SANDBOX_NETWORK → bridge)."""
        # The CONTAINER is the truth when there is one: a Linux agent that
        # adopted a pre-§4KF `host`-mode container must still bind loopback,
        # whatever today's default says (§4KF review, MAJOR-3).
        if getattr(self, "container", None) is not None:
            try:
                mode = self._container_network_mode()
            except Exception:  # noqa: BLE001
                mode = ""
            if mode in ("host", "bridge", "none"):
                return mode == "host"
        _override = getattr(self, "network_override", None)
        if _override:
            return _override == "host"
        _net = os.environ.get("GHOST_SANDBOX_NETWORK", "").strip().lower()
        if _net in ("host", "bridge", "none"):
            return _net == "host"
        return False  # §4KF: bridge is the default on every platform

    def published_service_ports(self):
        """The ports docker ACTUALLY published to the host loopback for the
        live container, or None when unknown (no container created yet — the
        caller then falls back to the configured range). getattr-guarded for
        managers whose __init__ was bypassed in tests."""
        return getattr(self, "_published_service_ports", None)

    @staticmethod
    def _derive_published_ports(container) -> set:
        """The set of container ports docker actually has published to the host,
        read from the LIVE container's ``HostConfig.PortBindings`` — the ground
        truth. Used when we ADOPT a container we didn't create (a pre-existing
        one after a deploy-by-kill, or a 409 name-race adopt), where the set we
        computed for our own aborted create says nothing about reality. Returns
        an empty set on any read failure (host-network containers publish
        nothing, so empty is the correct default). ``{'8100/tcp': [...]}`` →
        ``{8100}``."""
        try:
            container.reload()
            bindings = (container.attrs.get("HostConfig", {})
                        or {}).get("PortBindings") or {}
            out = set()
            for key in bindings:
                # key is like "8100/tcp"; take the numeric port.
                port = str(key).split("/", 1)[0]
                if port.isdigit():
                    out.add(int(port))
            return out
        except Exception as e:  # noqa: BLE001 — best-effort; empty is safe
            logger.debug("could not derive published ports from container: %s", e)
            return set()

    def _ready_is_fresh(self) -> bool:
        # getattr defaults keep this safe when __init__ was bypassed (tests
        # construct via __new__ / a stub), mirroring _get_lock's lazy pattern.
        ttl = getattr(self, "_READY_TTL_S", 8.0)
        last = getattr(self, "_last_ready_ok", 0.0)
        return (
            self.container is not None
            and (time.monotonic() - last) < ttl
        )

    def mark_ready(self):
        """Stamp a confirmed-good moment (a command exited without an
        infrastructure error). Called by execute() after a successful run."""
        self._last_ready_ok = time.monotonic()

    def invalidate_ready(self):
        """Force the next ensure_running() to run the full probe — call on any
        exec failure / exit 126/127 / OCI error, since those are exactly the
        signals that the container/mount may be gone."""
        self._last_ready_ok = 0.0

    def get_stats(self):
        # Snapshot the handle: a concurrent ensure_running() reprovision can
        # reassign self.container between the None-check and the call.
        container = self.container
        if not container: return None
        try: return container.stats(stream=False)
        except: return None

    def _is_container_ready(self):
        """False = not ready. A transient daemon/API hiccup gets ONE retry
        before we conclude not-ready: a false negative here is destructive
        (ensure_running force-removes the container and reprovisions from
        scratch, killing any in-flight work), so a healthy container must
        not be nuked over a momentary API error. NotFound is definitive —
        the container really is gone — and gets no retry."""
        try:
            return self._probe_container_ready()
        except self.NotFound:
            return False
        except Exception:
            time.sleep(0.5)
            try:
                return self._probe_container_ready()
            except Exception:
                return False

    def _exec_run(self, cmd, deadline_s: float = None, **kwargs):
        """``self.container.exec_run`` with a CLIENT-SIDE deadline.

        docker-py's exec output read blocks in ``poll.poll()`` with no timeout,
        so a wedged daemon hangs the calling thread indefinitely. Since the
        provision execs hold ``self._lock``, that would wedge every other
        turn's ``execute`` with zero log output. We run the exec on a daemon
        thread and ``join`` with a deadline: on expiry we ABANDON the blocked
        worker (a Python thread can't be killed — but a daemon thread won't
        block process exit and leaks only until the daemon recovers / the
        process restarts) and raise, so the caller releases the lock and the
        agent recovers with a clear error instead of a silent infinite hang.
        """
        deadline = _EXEC_DAEMON_DEADLINE_S if deadline_s is None else deadline_s
        result = {}

        def _run():
            try:
                result["ok"] = self.container.exec_run(cmd, **kwargs)
            except BaseException as e:  # noqa: BLE001 — re-raised on the caller thread
                result["err"] = e

        t = threading.Thread(target=_run, name="sandbox-exec", daemon=True)
        t.start()
        t.join(timeout=deadline)
        if t.is_alive():
            raise SandboxDaemonTimeout(
                f"container exec exceeded its {deadline:.0f}s client deadline — "
                f"the docker daemon may be wedged (command abandoned)")
        if "err" in result:
            raise result["err"]
        return result["ok"]

    def _provision_exec(self, cmd, **kwargs):
        """A provision-time exec whose CLIENT-side wedge deadline is derived
        from the command's own in-container ``timeout N`` budget + grace (see
        :func:`_provision_deadline_s`). Provision execs hold ``self._lock``, so
        they still need a client deadline against a genuinely wedged daemon —
        but it must sit ABOVE the install's own cap, never below it. Bare
        ``_exec_run`` (deadline_s=None) used the 1200s wedge default, which is
        LESS than the 1800s pip/torch/playwright caps: the inversion this
        method exists to remove."""
        return self._exec_run(
            cmd, deadline_s=_provision_deadline_s(cmd), **kwargs)

    def _exec_run_streamed(self, cmd, *, cid, ram_cap, deadline_s=None,
                           **exec_kwargs):
        """Stream a container exec through a bounded head+tail sink instead of
        buffering the whole output in agent RAM, returning ``(output_bytes,
        exit_code)``.

        Reproduces ``Container.exec_run`` EXACTLY — same ``exec_create`` args,
        same ``exec_start`` stream, same ``exec_inspect`` for the exit code
        (docker-py source) — so the output is byte-identical to the buffered
        call for output <= ram_cap; only the memory profile differs for a
        pathological producer. Runs under the SAME client wedge deadline as
        ``_exec_run`` (a daemon thread joined with a timeout), so a wedged
        daemon can't hang the caller."""
        deadline = _EXEC_DAEMON_DEADLINE_S if deadline_s is None else deadline_s
        demux = bool(exec_kwargs.get("demux", False))
        api = self.client.api
        create_kw = {"stdout": True, "stderr": True}
        if exec_kwargs.get("workdir") is not None:
            create_kw["workdir"] = exec_kwargs["workdir"]
        if exec_kwargs.get("user"):
            create_kw["user"] = exec_kwargs["user"]
        if exec_kwargs.get("environment"):
            create_kw["environment"] = exec_kwargs["environment"]
        result = {}

        def _run():
            stream = None
            try:
                exec_id = api.exec_create(cid, cmd, **create_kw)["Id"]
                stream = api.exec_start(exec_id, stream=True, demux=demux)
                out, _total = _drain_stream_bounded(stream, ram_cap, demux=demux)
                code = api.exec_inspect(exec_id).get("ExitCode")
                result["ok"] = (out, code)
            except BaseException as e:  # noqa: BLE001 — re-raised on caller thread
                result["err"] = e
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:  # noqa: BLE001
                        pass

        t = threading.Thread(target=_run, name="sandbox-exec-stream", daemon=True)
        t.start()
        t.join(timeout=deadline)
        if t.is_alive():
            raise SandboxDaemonTimeout(
                f"container exec exceeded its {deadline:.0f}s client deadline — "
                f"the docker daemon may be wedged (streamed command abandoned)")
        if "err" in result:
            raise result["err"]
        return result["ok"]

    def _probe_container_ready(self):
        # Verify the volume mount is still valid (not a deleted host inode)
        # AND the container responds to exec — in ONE exec_run. Previously
        # this was reload() + stat + echo = 3 daemon round-trips; the reload
        # was redundant (a dead/stopped container fails the exec anyway) and
        # the two execs collapse into a single `stat <syncfile> && echo OK`.
        # We MUST run with workdir=CONTAINER_WORKDIR: if the host directory
        # inode was deleted + recreated, any command against the bind mount
        # returns a non-zero OCI error, which is exactly the not-ready signal.
        import uuid
        test_file = f".mount_sync_{uuid.uuid4().hex}"
        test_path = self.host_workspace / test_file

        try:
            test_path.touch(exist_ok=True)
            code, out = self._exec_run(
                f"sh -c 'stat {test_file} >/dev/null 2>&1 && echo OK'",
                workdir=CONTAINER_WORKDIR,
            )
        finally:
            if test_path.exists():
                test_path.unlink()

        if code != 0:
            return False
        if out is not None and isinstance(out, (bytes, bytearray)):
            if b"OK" not in out:
                return False
        return True

    def _try_resume_stopped(self) -> bool:
        """If ``self.container`` is merely stopped/paused (not gone), start it
        and re-probe readiness. Returns True if it came back ready — saving a
        full destroy+reprovision that would kill in-sandbox services and all
        runtime state. False → the caller proceeds to recreate."""
        c = self.container
        if c is None:
            return False
        try:
            c.reload()
            status = c.status
        except Exception:
            return False
        if status not in ("exited", "created", "paused"):
            return False
        try:
            if status == "paused":
                c.unpause()
            else:
                c.start()
            # ⚠ RELOAD AFTER START (§4BW R2 CRITICAL). docker-py's `start()`
            # does NOT refresh `.attrs`, and the reload above ran BEFORE it —
            # so without this, `attrs["State"]["StartedAt"]` keeps its
            # pre-stop value for the whole resumed lifetime. That silently
            # broke the R1 generation-stamp fix: a stop→resume keeps the
            # container id AND (via the stale attrs) the same StartedAt, so
            # the stamp is identical across the resume and a recycled pid
            # still reads ALIVE — the exact wrong-process kill the stamp
            # exists to prevent. One reload here, at the single event where
            # attrs go stale, keeps the discriminator honest without a
            # per-liveness-check reload.
            c.reload()
        except Exception as e:  # noqa: BLE001 — fall through to recreate
            logger.debug("sandbox resume failed (%s); will recreate", e)
            return False
        if self._is_container_ready():
            pretty_log(
                "Sandbox Resume",
                "Resumed stopped container (in-sandbox services + runtime "
                "state preserved)", icon=Icons.SANDBOX_BOX)
            # §4FW: a resumed container is a NEW egress generation, and this
            # path RETURNS before the creation path's enforcement. `docker
            # start` recreates the network namespace (the GHOST_TOR rules
            # are gone) and starts none of the old processes (Tor is not
            # running) — measured on the live box 2026-09-10, thirteen
            # minutes after a restart that resumed the sandbox: an empty nat
            # table and a plain curl from inside answering {"IsTor": false}
            # with the host's real address.
            # It self-healed LATER, which is why it was invisible: the next
            # ensure_running past the readiness TTL finds the container ready,
            # skips this branch and falls through to the tail, which enforces
            # before the command runs. What that leaves exposed is exactly
            # what the resume exists to preserve — in-sandbox SERVICES and
            # promoted jobs, which come back with the container and reach the
            # network with no rules until some later agent command triggers
            # the enforcement — plus any command inside the post-boot TTL.
            # Enforcing HERE means the container is never handed back as
            # ready without its rules.
            # `unpause` keeps both (the freezer holds the namespace and the
            # processes), so covering it costs one idempotent re-apply.
            self._tor_attempted = False
            self._privilege_checked = False
            self._enforce_egress_once()
            self.mark_ready()
            return True
        return False

    def ensure_running(self):
        # Hold the lock for the WHOLE check+provision. The actual command
        # exec in execute() runs AFTER this returns (lock released), so
        # commands still run in parallel — only the readiness/provision
        # step is serialized, which is exactly what must not race.
        with self._lock:
            return self._ensure_running_impl()

    def _ensure_running_impl(self):
        # Fast path: a command succeeded within the TTL, so the container +
        # mount were confirmed good microseconds-to-seconds ago. Skip the
        # 3-round-trip probe entirely. invalidate_ready() (called on any exec
        # failure) resets the stamp, so a broken container never rides the TTL.
        # A container WE cut off its network must be recreated even when the
        # readiness TTL is fresh (§4GJ round 3): the cut-off leaves the
        # container perfectly able to run commands, so `mark_ready` keeps
        # stamping it and the check below the short-circuit was never reached
        # — it stayed cut off for the life of the TTL, and every refreshing
        # command extended that. Guarded on the in-memory state so the hot
        # path costs one attribute read, not a docker round-trip: only a
        # container this process blocked can be cut off with a fresh stamp,
        # and one inherited from a previous process has no stamp at all.
        # ⚠ SKIP THE TTL, DO NOT RECREATE HERE (§4GK round 4). Round 3 called
        # `_recreate_if_cut_off()` at this point, and that STRICTLY BROKE the
        # working path: the call nulls `self.container` and stamps
        # `_cut_off_at`, arming the 300 s backoff. The container is then
        # re-adopted by name a few lines below, and the SECOND
        # `_recreate_if_cut_off()` — the one that sits after the adopt and
        # before the readiness check, where it can actually reach the
        # remove-and-provision branch — is suppressed by the backoff its own
        # earlier twin just wrote. Net effect: the container stayed cut off,
        # `_egress_state` was downgraded to "" (= never attempted), and that
        # disarmed both the tool-side refusal and the `_execute_impl` belt.
        # Reproduced. All this line ever needed to do is decline the TTL
        # short-circuit so the real path below runs.
        if not self._cut_off and self._ready_is_fresh():
            return

        # Track whether this call did any actual work. Most invocations are
        # no-ops (the container is already up and provisioned) and must stay
        # silent — `execute()` calls `ensure_running` before every shell
        # command, so any unconditional logging here floods the agent log.
        did_work = False
        try:
            if not self.container:
                self.container = self.client.containers.get(self.container_name)
                # Adopted a container we did NOT create this process (routine
                # after a deploy-by-kill — the container outlives the agent).
                # Its real publish set lives on the container, not in our
                # (None) stamp, so read it — otherwise is_published_port falls
                # back to the configured range, which over-claims when the
                # survivor was created portless.
                if getattr(self, "_published_service_ports", None) is None:
                    self._published_service_ports = self._derive_published_ports(self.container)
        except self.NotFound:
            pass

        # §4GI R3: a container that was cut off its network (this or a
        # previous generation) is recreated so enforcement can be retried —
        # a resumed cut-off container can never bootstrap Tor. Backed off.
        self._recreate_if_cut_off()
        if not (self.container and self._is_container_ready()):
            # Before destroying + reprovisioning: if the container merely
            # STOPPED (e.g. an RSS-watchdog restart called close(remove=False),
            # whose docstring promised a "fast resume" that never existed),
            # try to RESUME it. Recreating discards every in-sandbox service
            # and all runtime apt/pip additions for nothing.
            if self.container is not None and self._try_resume_stopped():
                return
            did_work = True
            pretty_log("Sandbox Provision", "Initializing high-performance environment…", icon=Icons.SANDBOX_BOX)
            try:
                try:
                    old = self.client.containers.get(self.container_name)
                    # ⚠ The name is md5(workspace)[:8] — 32 bits. A
                    # collision is ~1-in-4-billion, but the blast radius
                    # is force-removing the LIVE agent's sandbox mid-turn
                    # while an isolated replay provisions its own (§4CL S1
                    # review). Cheap insurance: only reclaim a container
                    # that actually mounts OUR workspace. A container we
                    # cannot read mounts for is reclaimed as before —
                    # "cannot tell" must not strand the sandbox.
                    _mine = True
                    try:
                        _srcs = [m.get("Source") for m
                                 in ((old.attrs or {}).get("Mounts") or [])
                                 if m.get("Source")]
                        if _srcs:
                            _want = os.path.realpath(str(self.host_workspace))
                            _mine = any(os.path.realpath(str(x)) == _want
                                        for x in _srcs)
                    except Exception as _mx:  # noqa: BLE001
                        logger.debug("mount check skipped: %s", _mx)
                    if not _mine:
                        raise RuntimeError(
                            f"container name {self.container_name} is held by "
                            f"a container mounting a DIFFERENT workspace — "
                            f"refusing to force-remove it")
                    old.remove(force=True)
                    time.sleep(1)
                except self.NotFound: pass

                # If the host workspace dir vanished, recreate it OURSELVES.
                # Otherwise the docker daemon auto-creates the bind-mount
                # source as root-owned, after which the host-side readiness
                # touch fails with PermissionError on every future command
                # (an unrecoverable recreate loop). Best-effort.
                try:
                    Path(self.host_workspace).mkdir(parents=True, exist_ok=True)
                except Exception as mkdir_err:
                    logger.debug(f"Sandbox workspace mkdir skipped: {mkdir_err}")

                import sys
                is_mac = sys.platform == "darwin"
                
                # 1g is far too tight for ML workloads (pandas/sklearn/torch
                # OOM silently). Allow override via env var so users can tune
                # without code changes; default raised to 4g.
                import os as _os
                mem_limit = _os.environ.get("GHOST_SANDBOX_MEM", "4g")
                run_kwargs = {
                    "image": self.image,
                    "command": "sleep infinity",
                    "name": self.container_name,
                    "detach": True,
                    "tty": True,
                    "volumes": {str(self.host_workspace): {'bind': CONTAINER_WORKDIR, 'mode': 'rw'}},
                    "mem_limit": mem_limit,
                }

                # Fork-bomb / runaway-process defense — always on. Tunable but
                # never unbounded by default. (Set GHOST_SANDBOX_PIDS=0 to
                # disable, e.g. for highly-parallel workloads.)
                try:
                    _pids = int(_os.environ.get("GHOST_SANDBOX_PIDS", "1024"))
                    if _pids > 0:
                        run_kwargs["pids_limit"] = _pids
                except (TypeError, ValueError):
                    run_kwargs["pids_limit"] = 1024

                # tini as PID 1 (2026-07-12). The container command is
                # `sleep infinity`, which never wait()s — every orphaned dead
                # child became a PERMANENT ZOMBIE ([sh]/[tor]/[headless_shell]
                # <defunct> accumulated in prod). Zombies pass `kill -0`, so
                # dead service launchers looked "already running", stop() was
                # a no-op against them, and the service manager's
                # exited-immediately diagnostic never fired (a 137s live
                # request burned 3 failed launches on this). init=true makes
                # docker run tini as PID 1, which reaps orphans on arrival.
                run_kwargs["init"] = True

                # Owner stamp (§4CL S1 review). The orphan sweeper's only
                # identity check is "not MY container_name", so a
                # per-solve container belonging to a run IN FLIGHT in
                # ANOTHER process becomes a reap candidate the moment it
                # passes the 30-minute age floor — and per-solve
                # candidates get no liveness check at all. A second agent
                # booting (an ablation throwaway, the test suite, a
                # manual restart) would then force-remove a live replay's
                # container, and the next `ensure_running` would silently
                # recreate it, discarding all in-sandbox state. The label
                # lets the sweeper ask the one question that separates
                # "orphaned by SIGKILL" from "somebody else is using it".
                try:
                    run_kwargs["labels"] = {
                        "ghost.owner_pid": str(_os.getpid()),
                        "ghost.owner_boot": _owner_boot_id(),
                    }
                except Exception as _lbl:  # noqa: BLE001
                    logger.debug("owner label skipped: %s", _lbl)

                # Capability hardening — ON by default (§4KF, 2026-09-24).
                # It used to be opt-in because of a stale reason: "passwordless
                # sudo (setuid) for apt installs would break". The exec user
                # is ROOT on macOS (on Linux `execute` runs as the host
                # uid:gid, where setuid sudo was already refused for an
                # unknown uid), so nothing the SANDBOX CODE does needs to GAIN
                # privilege; what apt/dpkg/pip/Tor/the service supervisor need
                # is to DROP it (SETUID/SETGID to `_apt` and `debian-tor`),
                # own package files (CHOWN/DAC_OVERRIDE/FOWNER/FSETID) and
                # signal children (KILL). Measured on a throwaway container
                # from the live image before this flipped: apt's privilege
                # drop, tor as debian-tor, chown, pip, Chromium headless, a
                # node service bind, killing a child — all fine; raw sockets,
                # mknod, chroot and plain-root iptables refused. The iptables
                # egress rules are applied through a PRIVILEGED exec, which
                # carries its own capabilities regardless of this set.
                # GHOST_SANDBOX_DROP_CAPS=0 opts OUT (the pre-§4KF opt-in
                # spelling "1" is accepted as the default it now is).
                if _os.environ.get("GHOST_SANDBOX_DROP_CAPS", "1").strip().lower() not in ("0", "false", "no", "off"):
                    run_kwargs["cap_drop"] = ["ALL"]
                    run_kwargs["cap_add"] = list(SANDBOX_KEPT_CAPS)
                    run_kwargs["security_opt"] = ["no-new-privileges"]

                # Network mode. `bridge` everywhere by default (§4KF): the
                # in-container Tor + iptables egress enforcement (§4FU) only
                # works in the container's OWN netns; under `host` the
                # in-container Tor cannot bind and the enforcement does not
                # apply. The old Linux default of `host` predates §4FU (it
                # existed so the browser could reach the HOST's Tor at
                # 127.0.0.1:9050, which the in-container Tor made moot).
                # GHOST_SANDBOX_NETWORK=host remains an explicit choice, and
                # `none` isolates a replay. Non-mac bridge gets
                # host.docker.internal — reachable only when no Tor egress is
                # enforced (the rules redirect every non-loopback TCP to the
                # TransPort, and Tor cannot dial the host gateway).
                _net = (getattr(self, "network_override", None)
                        or _os.environ.get("GHOST_SANDBOX_NETWORK", "")
                        ).strip().lower()
                if _net in ("host", "bridge", "none"):
                    run_kwargs["network_mode"] = _net
                    if _net == "bridge" and not is_mac:
                        run_kwargs["extra_hosts"] = {"host.docker.internal": "host-gateway"}
                else:
                    run_kwargs["network_mode"] = "bridge"
                    if not is_mac:
                        run_kwargs["extra_hosts"] = {"host.docker.internal": "host-gateway"}

                # Service-port publishing (sandbox/services.py, 2026-07-11).
                # In bridge mode a supervised in-sandbox service (e.g. a dev
                # server the agent hosts) is unreachable from the host; we
                # publish a small loopback-bound range so the OPERATOR can
                # open http://127.0.0.1:<port> in their own browser. Range
                # via GHOST_SANDBOX_SERVICE_PORTS ("8100-8104" default;
                # empty string disables). Host mode needs none (the service
                # binds host ports directly). Only takes effect when the
                # container is (re)created.
                _published = set()
                if run_kwargs.get("network_mode") == "bridge":
                    try:
                        from .services import publishable_service_ports
                        # Only ports actually FREE on the host: a second agent
                        # (a throwaway for an ablation, the test suite) can't
                        # publish the same fixed host ports as the instance
                        # already running, and must degrade to no-published-
                        # ports rather than fail to get a sandbox at all.
                        _svc_ports = publishable_service_ports()
                        if _svc_ports:
                            run_kwargs["ports"] = {
                                f"{p}/tcp": ("127.0.0.1", p)
                                for p in _svc_ports
                            }
                        _published = set(_svc_ports)
                    except Exception as _spx:
                        logger.debug(f"service-port publish skipped: {_spx}")
                # Record what was ACTUALLY published (may be empty for a 2nd
                # instance) so is_published_port consults reality, not the
                # configured range (2026-07-15). Host mode publishes none here.
                self._published_service_ports = _published

                # Check for cached environment image for instant boot.
                # NB: never mutate self.image — it must stay the pullable
                # base image. Pinning the cached tag on self.image meant
                # that if the cache was later deleted (docker rmi), the
                # fallback tried to pull "ghost-agent-base:latest" from
                # Docker Hub (404) instead of the real base image,
                # bricking the sandbox until process restart.
                boot_image = self.image
                try:
                    self.client.images.get("ghost-agent-base:latest")
                    boot_image = "ghost-agent-base:latest"
                except self.docker_lib.errors.ImageNotFound:
                    pass
                run_kwargs["image"] = boot_image

                # Skip the network round-trip when the image is already
                # present locally. Only on `ImageNotFound` do we pay for a
                # `pull`. Any other exception (transient daemon hiccup,
                # auth glitch on a private registry) is logged but
                # tolerated — the subsequent `containers.run` will surface
                # a more actionable error if the image is genuinely
                # unusable.
                try:
                    self.client.images.get(boot_image)
                except self.docker_lib.errors.ImageNotFound:
                    pretty_log("Sandbox Image", f"Pulling required Docker image: {boot_image}", icon=Icons.TOOL_DOWN)
                    try:
                        self.client.images.pull(boot_image)
                    except Exception as pull_err:
                        logger.warning(
                            f"Sandbox image pull failed ({type(pull_err).__name__}: {pull_err}); "
                            f"continuing — `containers.run` will surface the real error if the "
                            f"image is unavailable."
                        )
                except Exception as inspect_err:
                    logger.warning(
                        f"Sandbox image inspect failed ({type(inspect_err).__name__}: {inspect_err}); "
                        f"skipping pull and continuing — `containers.run` will surface the error "
                        f"if the image is genuinely missing."
                    )

                # CPU limit (configurable via GHOST_SANDBOX_CPU_QUOTA, default
                # 200000 = 2 CPUs at the standard 100000-µs period). Without
                # this a single runaway sandbox script can saturate the host.
                try:
                    cpu_quota = int(os.environ.get("GHOST_SANDBOX_CPU_QUOTA", "200000"))
                except ValueError:
                    cpu_quota = 200000
                # <= 0 means "no CPU cap" (mirroring GHOST_SANDBOX_PIDS=0).
                # Passing 0 through was rejected by the daemon ("CPU cfs
                # quota cannot be less than 1ms"), bricking creation.
                if cpu_quota > 0:
                    run_kwargs["cpu_period"] = 100000
                    run_kwargs["cpu_quota"] = cpu_quota

                try:
                    self.container = self.client.containers.run(**run_kwargs)
                except self.APIError as run_err:
                    msg = str(run_err).lower()
                    if "port is already allocated" in msg and run_kwargs.get("ports"):
                        # Lost the race between publishable_service_ports()'s
                        # bind-check and this run (another container grabbed
                        # the port in between). Published ports are an
                        # operator convenience, NOT worth a bricked sandbox —
                        # retry once without them.
                        #
                        # CRITICAL: a port-bind failure leaves the container
                        # CREATED-but-not-started, so it must be REMOVED first
                        # or the retry dies with a 409 name-in-use (observed:
                        # the retry's own 409 propagated and killed the
                        # sandbox entirely).
                        run_kwargs.pop("ports", None)
                        pretty_log(
                            "Sandbox Ports",
                            "service-port publish conflicted with another "
                            "process — container created WITHOUT published "
                            "ports (in-sandbox services are still reachable "
                            "by browser/execute).",
                            level="WARNING", icon=Icons.WARN,
                        )
                        try:
                            self.client.containers.get(
                                self.container_name).remove(force=True)
                        except Exception:  # noqa: BLE001 — nothing to clean
                            pass
                        self.container = self.client.containers.run(**run_kwargs)
                        # We retried WITHOUT ports → nothing is published. The
                        # stamp from line ~371 still claimed the ports; leaving
                        # it made is_published_port over-claim and the remote
                        # hint point the operator (via tailscale serve) at a
                        # FOREIGN process on that port. Correct it to empty.
                        self._published_service_ports = set()
                    elif getattr(run_err, "status_code", None) == 409 or "already in use" in msg:
                        # Another process (sharing this docker daemon and the
                        # workspace-derived container name) won the race
                        # between our remove and run — a 409 "name already in
                        # use". Adopt the existing container instead of dying.
                        self.container = self.client.containers.get(self.container_name)
                        # The adopted container's real publish set is whatever
                        # IT was created with, not our aborted create's — read
                        # it from the container itself.
                        self._published_service_ports = self._derive_published_ports(self.container)
                    else:
                        raise

                # New container generation → environment and tor state of
                # the previous generation no longer apply.
                self._env_verified = False
                self._tor_attempted = False
                self._privilege_checked = False
                # ⚠ AND NEITHER DOES THE CUT-OFF (§4GK round 6). `_cut_off`
                # was raised by `_block_egress_hard` and lowered ONLY by
                # `_recreate_if_cut_off` — which returns early when the
                # container is gone, when it is not actually cut off, or while
                # its 300 s backoff stands. So if the container died or was
                # removed inside that window (RSS watchdog, orphan sweep, a
                # readiness false negative) and was provisioned fresh here,
                # the flag stayed True against a healthy attached container
                # and every later command declined the readiness TTL — round
                # 4's "the TTL disabled for the life of the process" defect,
                # reached by another route. This is the arm-on-N-needs-a-
                # recorded-DONE class: the raise had one clearer, and it was
                # not on every path that ends the condition.
                self._cut_off = False

                for _ in range(10):
                    if self._is_container_ready(): break
                    time.sleep(1)
                else:
                    # Previously this fell through silently and provisioning
                    # proceeded against a container that never became ready,
                    # surfacing as confusing install failures downstream.
                    raise Exception(
                        f"Container {self.container_name} did not become "
                        f"ready within 10s of creation"
                    )

            except Exception as e:
                pretty_log("Sandbox Error", f"Failed to start: {e}", level="ERROR", icon=Icons.FAIL)
                raise e

        env_vars = {}
        # We don't set HTTP_PROXY for the sandbox because we don't want to route
        # heavy package installs through Tor to avoid timeouts and IP blocks.

        # Marker version: bump this (and the string in sandbox/Dockerfile)
        # whenever the provisioning surface changes in a way that prior
        # committed images can't be trusted to match.
        #
        # History:
        #   v1 (legacy): .supercharged — used `playwright install
        #                chromium` WITHOUT `--with-deps`, so the cached
        #                image was missing libnss3/libatk/etc and
        #                Chromium couldn't actually launch. The self-play
        #                log caught this: the agent discovered at
        #                runtime that browsers were broken, re-installed
        #                Chromium (still without deps), re-ran, still
        #                failed, burned ~100 s.
        #   v2:          .supercharged.v2 — ensures `--with-deps` ran.
        #                Images without the v2 marker are treated as
        #                un-provisioned and go through a full install.
        #   v3:          .supercharged.v3 — preinstalls the CPU PyTorch
        #                wheel. Without it, every "train a model" project
        #                hit `ModuleNotFoundError: torch` and ran a ~300 s
        #                `pip install torch` mid-task (observed live: the
        #                PetAI training task), often tripping the execute
        #                timeout. v2 images re-provision to pick torch up.
        #   v4:          .supercharged.v4 — adds `iproute2` (the `ss`
        #                socket/port inspector) and preinstalls `flask` +
        #                `python-chess`. "Host a web app / chess service"
        #                requests otherwise `pip install flask python-chess`
        #                mid-task (~24 s serial thrash, observed live on the
        #                chess-hosting flow). v3 images re-provision to pick
        #                these up.
        #   v5:          .supercharged.v5 — adds `stockfish` (the chess
        #                project's engine-opponent mode; a recreate must
        #                not silently drop the engine). v4 images
        #                re-provision to pick it up.
        #   v6:          .supercharged.v6 — adds `file`. The model verifies
        #                a download the standard way, `file x.pdf`, and the
        #                missing binary turned a successful 16 MB fetch into
        #                exit 127 and a failure strike (2026-09-09, request
        #                d50a34bd). First version upgraded IN PLACE — the
        #                delta below — not re-provisioned: a full provision
        #                is ~5 minutes on the operator's next request, the
        #                price every earlier bump paid.
        #   v7:          .supercharged.v7 — adds `xxd`. The live check of v6
        #                found the same shape one tool over: the model's
        #                verification chain is `file x.pdf && head -c 200
        #                x.pdf | xxd | head -5`, and the missing xxd scored
        #                the download a failure again.
        #   v8:          .supercharged.v8 — adds `lsof` and `dnsutils`
        #                (host/nslookup/dig). 415 trajectory files, every
        #                "command not found" counted: after file and xxd,
        #                these were the only tools the model reached for
        #                that the image lacked (lsof ×2, host/nslookup ×2).
        #   v9:          .supercharged.v9 — adds `iptables`, for the
        #                Tor-only egress rules (§4FU, sandbox/tor_egress.py).
        #   v10 (now):   .supercharged.v10 — the everyday CLI set the image
        #                lacked (§4KG, 2026-09-24): `nc`/`socat` (network
        #                plumbing a shell task reaches for; the UDP leak
        #                probe that used to call `nc` now observes EPERM from
        #                Python instead — nc's exit code says nothing),
        #                `unzip` (referenced by the sandbox code), `ffmpeg`
        #                (asked for and missing), and jq, zip, tree, nano,
        #                bc, gawk, rsync, pkg-config, telnet, whois, 7z,
        #                pdftotext, imagemagick. `stockfish` was installed
        #                since v5 but lives in /usr/games, OFF the exec PATH
        #                — v10 links it into /usr/local/bin. No
        #                ping/traceroute: they need NET_RAW, which §4KF
        #                dropped. v9 images upgrade in place; older ones
        #                take the full provision.
        marker_path = "/root/.supercharged.v10"
        # The marker this version supersedes, and the delta that lifts an
        # image from it to this one. Keep the three in step with the history
        # above when bumping again — and with SANDBOX_TOOL_PACKAGES /
        # SANDBOX_TOOL_BINARIES (pinned equal by tests/test_4kg_sandbox_tools).
        # Debian installs stockfish under /usr/games, which the exec PATH
        # (the python image's) does not include: it was "missing" on the
        # live container while the package was there. The symlink puts it on
        # PATH for the model and for the verify chain alike (measured 2026-09-24).
        prev_marker_path = "/root/.supercharged.v9"
        upgrade_delta_cmd = "timeout 1800 sh -c 'apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends netcat-openbsd socat jq unzip zip tree nano bc gawk rsync pkg-config telnet whois p7zip-full poppler-utils ffmpeg imagemagick stockfish && ln -sf /usr/games/stockfish /usr/local/bin/stockfish'"
        upgrade_delta_verify = "sh -c 'command -v nc && command -v socat && command -v jq && command -v unzip && command -v zip && command -v tree && command -v nano && command -v bc && command -v gawk && command -v rsync && command -v pkg-config && command -v telnet && command -v whois && command -v 7z && command -v pdftotext && command -v ffmpeg && command -v convert && command -v stockfish'"

        # The marker/chromium probes are two docker execs; running them
        # before EVERY command added latency for nothing. Verify once per
        # container generation (the flag is reset when a container is
        # created). Trade-off: if someone deletes chromium inside a live
        # container, detection now happens on the next recreate, not the
        # next command — provision-time gating (the v2 lesson) is intact.
        if self._env_verified:
            marker_ok = chromium_ok = True
        else:
            marker_ok = (self._exec_run(f"test -f {marker_path}")[0] == 0)
            chromium_ok = self._chromium_binary_present()
        # ── In-place upgrade from the previous marker (v6, 2026-09-09) ──
        # An image carrying the PREVIOUS marker has everything but this
        # version's delta. Install only the delta, VERIFY it landed, stamp
        # the new marker, commit. Any failure falls through to the full
        # provision below, which installs the same packages from scratch.
        # The delta RESPECTS the provisioning backoff: inside one, it is
        # skipped and the gate below reports the wait — a broken mirror must
        # not be hit with a 600 s apt on EVERY command while the provision
        # lock is held (review, 2026-09-09). It does not ARM the backoff
        # itself: a failed delta falls through to the full provision in the
        # same call, and that path arms the backoff on its own failure.
        if (not marker_ok and chromium_ok
                and time.time() >= self._provision_backoff_until):
            try:
                _prev_ok = (self._exec_run(f"test -f {prev_marker_path}")[0] == 0)
            except Exception:  # noqa: BLE001 — a probe failure is "not present"
                _prev_ok = False
            if _prev_ok:
                pretty_log(
                    "Sandbox Provision",
                    f"Upgrading {prev_marker_path.rsplit('.', 1)[-1]} → "
                    f"{marker_path.rsplit('.', 1)[-1]} in place (adds: "
                    f"{' '.join(SANDBOX_TOOL_BINARIES)})…",
                    icon=Icons.SANDBOX_BOX,
                )
                _verified = False
                _code, _out = None, b""
                try:
                    _code, _out = self._provision_exec(
                        upgrade_delta_cmd, environment=env_vars)
                    # The marker asserts the binary is there; only a probe
                    # that finds it may write the marker (the v2 lesson).
                    _verified = (
                        _code == 0
                        and self._exec_run(upgrade_delta_verify)[0] == 0)
                except Exception as _e:  # noqa: BLE001 — fall through to full
                    logger.debug("in-place sandbox upgrade raised: %s", _e)
                if _verified:
                    self._exec_run(f"touch {marker_path}")
                    self._exec_run(f"rm -f {prev_marker_path}")
                    try:
                        pretty_log("Sandbox Cache", "Committing fast-boot image cache…",
                                   icon=Icons.SANDBOX_BOX)
                        self.container.commit(repository="ghost-agent-base", tag="latest")
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"Failed to commit sandbox image cache: {e}")
                    marker_ok = True
                    did_work = True
                else:
                    # Say WHY (§4KG review): a mirror down over Tor, a verify
                    # chain that missed a binary and a timeout used to be
                    # indistinguishable — the delta's exit code and output
                    # were discarded and only an exception reached DEBUG.
                    _tail = ""
                    try:
                        _tail = (_out or b"").decode("utf-8", "replace").strip().splitlines()[-3:]
                        _tail = " | ".join(_tail)[:400]
                    except Exception:  # noqa: BLE001
                        _tail = ""
                    pretty_log(
                        "Sandbox Provision",
                        f"In-place upgrade did not verify (delta exit {_code if _code is not None else 'n/a'}"
                        f"{'; ' + _tail if _tail else ''}) — falling back to a full provision.",
                        level="WARNING", icon=Icons.WARN,
                    )

        if not marker_ok or not chromium_ok:
            if time.time() < self._provision_backoff_until:
                raise Exception(
                    "Sandbox provisioning failed recently; retrying in "
                    f"{int(self._provision_backoff_until - time.time())}s "
                    "(backoff prevents reinstall storms against a failing mirror)."
                )
            did_work = True
            # Pessimistic backoff: set BEFORE the installs, cleared on
            # success. If any install below raises, the next command won't
            # immediately re-run a multi-minute failing install while
            # holding the provision lock.
            self._provision_backoff_until = time.time() + 300.0
            if marker_ok and not chromium_ok:
                # The cached image claims to be provisioned but the
                # Chromium binary isn't actually on disk — the exact
                # silent-failure mode v2 exists to catch. Flag loudly;
                # the full install flow below will fix it.
                pretty_log(
                    "Sandbox Chromium",
                    "Provision marker present but Chromium binary missing. Reinstalling…",
                    level="WARNING",
                    icon=Icons.WARN,
                )
            pretty_log("Sandbox Provision", "Installing deep-learning stack (~60s)…", icon=Icons.SANDBOX_BOX)

            # Every install below is wrapped in the in-container `timeout`
            # binary (coreutils ships in slim-bookworm): these exec_runs
            # block a worker thread WHILE HOLDING self._lock, so an
            # unbounded mirror/CDN stall would wedge every concurrent tool
            # call in the agent. The caps are generous — they exist to
            # bound a stall, not to race a slow link.
            apt_cmd = "timeout 1800 sh -c 'export DEBIAN_FRONTEND=noninteractive; apt-get update && apt-get install -y sudo coreutils nodejs npm g++ curl wget git procps postgresql-client libpq-dev tor ripgrep sqlite3 iproute2 file xxd lsof dnsutils iptables netcat-openbsd socat jq unzip zip tree nano bc gawk rsync pkg-config telnet whois p7zip-full poppler-utils ffmpeg imagemagick stockfish --no-install-recommends && ln -sf /usr/games/stockfish /usr/local/bin/stockfish'"
            code, out = self._provision_exec(apt_cmd, environment=env_vars)
            if code != 0:
                err_msg = out.decode("utf-8", errors="replace") if out else "Unknown error"
                raise Exception(f"System package installation failed: {err_msg}")

            # No `ALL ALL=(ALL) NOPASSWD: ALL` sudoers line any more (§4KF): on
            # macOS the exec user is root, so the line granted nothing root
            # lacks; on Linux the exec user is the host uid:gid and the grant
            # would hand it root outright. The `sudo` PACKAGE stays (model
            # scripts say `sudo apt-get`; root through Debian's stock sudoers
            # works without a password). A cached image that still carries
            # the grant is stripped in `_settle_privileges_once`.

            if self.tor_proxy:
                code, out = self._provision_exec("timeout 600 pip install --no-cache-dir pysocks requests")
                if code != 0:
                    err_msg = out.decode("utf-8", errors="replace") if out else "Unknown error"
                    raise Exception(f"PySocks bootstrap failed: {err_msg}")

            install_cmd = (
                "timeout 1800 pip install --no-cache-dir "
                "numpy pandas scipy matplotlib seaborn plotly "
                "scikit-learn yfinance beautifulsoup4 networkx requests "
                "pylint black mypy bandit dill ipykernel jupyter_client "
                "pytest pytest-asyncio "
                "psycopg2-binary asyncpg sqlalchemy tabulate sqlglot playwright html2text lxml "
                "flask python-chess"
            )
            code, out = self._provision_exec(install_cmd, environment=env_vars)
            if code != 0:
                err_msg = out.decode("utf-8", errors="replace") if out else "Unknown error"
                raise Exception(f"Python package installation failed: {err_msg}")

            # PyTorch — CPU wheel only (the default GPU wheels pull ~2 GB of
            # CUDA the sandbox can't use). Preinstalled so "build/train a model"
            # projects don't `pip install torch` mid-task and trip the execute
            # timeout (observed live: PetAI's training task hit
            # ModuleNotFoundError: torch, then a 300 s install). Best-effort and
            # NON-fatal: a machine that can't reach the torch CDN should still
            # get a working sandbox (the agent falls back to a runtime install),
            # so a torch flake must not poison provisioning of everything else.
            pretty_log("Sandbox PyTorch", "Installing CPU PyTorch (~1m)…", icon=Icons.SANDBOX_BOX)
            torch_code, torch_out = self._provision_exec(
                "timeout 1800 pip install --no-cache-dir torch "
                "--index-url https://download.pytorch.org/whl/cpu",
                environment=env_vars,
            )
            if torch_code != 0:
                torch_err = (torch_out.decode("utf-8", errors="replace")
                             if torch_out else "unknown error")
                pretty_log(
                    "Sandbox PyTorch",
                    f"torch preinstall failed (non-fatal — runtime install still "
                    f"works): {torch_err[:200]}",
                    level="WARNING", icon=Icons.WARN,
                )
                
            # Unconditionally install Chromium inside this first-boot
            # block. The previous gate ran `from playwright.sync_api
            # import sync_playwright` and skipped the install when the
            # Python library was importable — but library-importable
            # does NOT imply the Chromium BINARY is on disk. (The pip
            # install above puts the library in place, which made the
            # probe pass every time and silently skip the binary
            # install on first provision.) The eval at 2026-04-23
            # 09:56 hit "headless_shell not found" for exactly this
            # reason and burned ~100 s of agent time recovering. We're
            # already inside the `test -f /root/.supercharged` outer
            # gate, so this only runs on a container that has never
            # been provisioned — the "re-download on every boot"
            # concern the old gate was trying to address can't happen.
            # If Chromium is somehow already present (e.g. the user
            # manually deleted the supercharged marker without wiping
            # the cache), `playwright install` short-circuits in ~1 s.
            pretty_log("Sandbox Chromium", "Installing headless Chromium (~2m)…", icon=Icons.TOOL_DOWN)
            pw_code, pw_out = self._provision_exec(
                "timeout 1800 python3 -m playwright install chromium --with-deps",
                environment=env_vars,
            )
            if pw_code != 0:
                # Fail loud: refuse to touch /root/.supercharged so a
                # failed Chromium download can't silently poison every
                # future boot into thinking the environment is ready.
                err_msg = pw_out.decode("utf-8", errors="replace") if pw_out else "Unknown error"
                raise Exception(
                    f"Playwright Chromium installation failed (exit {pw_code}): {err_msg}"
                )

            # Post-install sanity: verify the Chromium binary we just
            # installed is actually on disk before we set the marker.
            # This is the second line of defence behind `--with-deps
            # must exit 0` above — if the install exited 0 for some
            # weird reason but didn't produce a binary (network flake,
            # disk-full mid-extract), we'd rather fail loud here than
            # leave a v2-marked image that's still broken.
            if not self._chromium_binary_present():
                raise Exception(
                    "Playwright install reported success but no Chromium "
                    "binary found under /root/.cache/ms-playwright. "
                    "Refusing to mark container as provisioned."
                )

            self._exec_run(f"touch {marker_path}")
            # Remove any legacy v1 marker so a downgrade-then-upgrade
            # cycle doesn't leave stale state around — and the marker this
            # version supersedes, so an image never carries two.
            self._exec_run("rm -f /root/.supercharged")
            self._exec_run(f"rm -f {prev_marker_path}")

            # Cache the fully installed environment for instant future
            # startups. Committed UNCONDITIONALLY after a successful
            # provision: the old `if self.image != "ghost-agent-base"`
            # guard meant a container booted from a STALE cached image
            # (e.g. v2-era, forcing the full reinstall above) never wrote
            # the freshened image back — so every future recreation paid
            # the full multi-minute provision again, forever.
            try:
                pretty_log("Sandbox Cache", "Committing fast-boot image cache…", icon=Icons.SANDBOX_BOX)
                self.container.commit(repository="ghost-agent-base", tag="latest")
            except Exception as e:
                logger.warning(f"Failed to commit sandbox image cache: {e}")

            # Provision succeeded — lift the failure backoff.
            self._provision_backoff_until = 0.0

        self._env_verified = True

        # §4FU: Tor-only egress for the SANDBOX, enforced in its own network
        # namespace (see sandbox/tor_egress.py for the design and the spike
        # that settled it). Runs AFTER provisioning — apt, pip and the
        # Chromium download stay direct, which is what the old comment
        # wanted — and once per container generation. Under host
        # networking the sandbox shares the host's namespace and iptables
        # here would rewrite the HOST's traffic: not applied, said once.
        if self.tor_proxy and not self._tor_attempted:
            did_work = True
        self._enforce_egress_once()

        # Reached the end of ensure_running without raising → container +
        # mount + environment are all confirmed good. Stamp the readiness TTL
        # so the next command within the window skips the probe entirely.
        self.mark_ready()

        # Only announce readiness when this call actually had to bring the
        # environment up. Silent on the steady-state common path.
        if did_work:
            pretty_log("Sandbox Ready", "Environment Ready.", icon=Icons.OK)

    #: "enforced" | "blocked" (rules in, Tor not verified) | "unavailable"
    #: (host networking / no iptables) | "" (not attempted yet).
    #:
    #: ⚠ ITS ONLY READERS ARE IN THIS FILE AND `egress_gate` (§4GK round 4).
    #: This comment used to say "Read by get_stats() and the health report" —
    #: neither is true: `get_stats()` returns `container.stats()` and never
    #: touches it, and no reader exists outside the two modules. A docstring
    #: that names consumers which do not exist is how a silently-dead state
    #: flag survives a review; this flag was one, for a whole section.
    _egress_state = ""
    _egress_exit_ip = ""

    def egress_enforcement_attempted(self) -> bool:
        """§4GI R3: False until `_enforce_egress_once` has run for THIS
        container generation. The tool-side gate must not read an
        un-attempted state as "unavailable": enforcement runs inside
        `execute()` (`ensure_running`), so before the first call the state
        is simply unknown — refusing there re-created the §4DD outage
        (a lazily rebuilt sandbox refused every command forever)."""
        return bool(self._egress_state)

    def egress_is_enforced_or_blocked(self) -> bool:
        """§4GI: True when the sandbox cannot reach the internet directly —
        Tor-only rules are in ("enforced"/"blocked"), or the container was
        cut off its network because they could not be. False means DIRECT
        egress is possible (host networking, or a cut-off that itself
        failed): a caller under `--mandatory-tor` must refuse to run
        network-capable work. Reads state only; never blocks."""
        return self._egress_state in ("enforced", "blocked")

    def _block_egress_hard(self, why: str) -> None:
        """Fail closed BY CONSTRUCTION when the Tor-only rules could not be
        established (§4GI): disconnect the container from every docker
        network it is attached to, so the state is "blocked", not "direct".
        Before this, every such branch logged an ERROR, set
        `_egress_state = "unavailable"`, and left the sandbox serving
        `execute`/browser calls with cleartext egress — a flag nothing read.
        Never raises; the one branch that cannot be closed (a disconnect
        that itself fails) stays "unavailable" and is logged at CRITICAL."""
        nets = []
        networks_reported = False
        try:
            self.container.reload()
            _ns = (self.container.attrs or {}).get("NetworkSettings")
            networks_reported = isinstance(_ns, dict) and "Networks" in _ns
            nets = list(((_ns or {}).get("Networks") or {}).keys())
        except Exception:  # noqa: BLE001 — attrs may be stubbed
            nets = []
        if not nets and not networks_reported:
            nets = ["bridge"]          # attrs did not say: assume the default network
        if not nets:
            # Already cut off (a previous generation's disconnect persists
            # across docker stop/start): nothing to disconnect, and trying
            # "bridge" would raise "not connected" and read as unavailable
            # (R3 review). Blocked by construction; the next ensure_running
            # recreates the container (backoff below).
            self._set_egress_state("blocked")
            self._cut_off = True
            self._cut_off_at = time.time()
            pretty_log("Sandbox Egress",
                       f"{why} — the container has NO network attached "
                       "(already cut off): blocked; will recreate the sandbox "
                       "to retry enforcement.", level="ERROR", icon=Icons.FAIL)
            return
        failed = []
        for name in nets:
            try:
                self.client.networks.get(name).disconnect(self.container, force=True)
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{name}: {type(exc).__name__}: {exc}")
        if failed:
            self._set_egress_state("unavailable", "cut_off_failed")
            pretty_log(
                "Sandbox Egress",
                f"{why} — AND the container could not be cut off its network "
                f"({'; '.join(failed)[:200]}). Sandbox egress may be DIRECT: "
                "refuse network work until the sandbox is recreated.",
                level="CRITICAL", icon=Icons.FAIL,
            )
            return
        self._set_egress_state("blocked")
        self._cut_off = True
        self._cut_off_at = time.time()
        pretty_log(
            "Sandbox Egress",
            f"{why} — the container was DISCONNECTED from {', '.join(nets)}: "
            "no network at all rather than direct egress (fail-closed by "
            "construction; the sandbox is recreated to retry, at most every "
            f"{int(self._CUT_OFF_RECREATE_BACKOFF_S)}s).",
            level="ERROR", icon=Icons.FAIL,
        )

    #: A cut-off container is recreated (fresh network, enforcement retried)
    #: on the next ensure_running, but not more often than this — a
    #: persistent iptables fault would otherwise re-provision on every call.
    _CUT_OFF_RECREATE_BACKOFF_S = 300.0
    _cut_off_at = 0.0
    #: ⚠ "CUT OFF" IS NOT "BLOCKED" (§4GK round 5). `_egress_state == "blocked"`
    #: is ALSO the HEALTHY state written the moment the iptables rules load,
    #: and two branches then return leaving it there for the life of the
    #: container ("Tor is not running as debian-tor", "Tor did not bootstrap
    #: within the timeout"). Round 4's readiness guard keyed on that string,
    #: so in those regimes the readiness TTL was disabled on EVERY command —
    #: turning a once-per-8s docker probe into a per-command one, and raising
    #: the number of chances to hit `_is_container_ready`'s DESTRUCTIVE false
    #: negative (which force-removes the container and reprovisions, killing
    #: in-flight work) from once per TTL to once per command, forever. The
    #: live log shows that regime really occurs. This flag names the actual
    #: condition — we disconnected this container — and nothing else.
    _cut_off = False
    #: Which branch left the state "unavailable" — the tools' refusal names
    #: it, because the two have different remedies (§4GJ).
    _egress_unavailable_reason = ""

    def _set_egress_state(self, state: str, reason: str = "") -> None:
        """THE egress-state transition. The reason travels WITH the state
        (§4GJ round 3): it used to be written at one branch and never
        cleared, so a later disconnect-failure inherited the host-networking
        remedy ("set GHOST_SANDBOX_NETWORK=bridge") — advice that does
        nothing for a container whose disconnect failed. Every write goes
        through here; `tests/test_4gi_review_round2.py` enumerates the class
        and fails on a bare assignment."""
        self._egress_state = state
        self._egress_unavailable_reason = reason if state == "unavailable" else ""

    @staticmethod
    def _container_cut_off(container) -> bool:
        """True when the container has no docker network attached — the
        state `_block_egress_hard` leaves behind, which docker persists
        across stop/start (R3 review: nothing recreated it)."""
        try:
            container.reload()
            attrs = container.attrs or {}
            mode = str(((attrs.get("HostConfig") or {}).get("NetworkMode")) or "")
            if mode == "host":
                return False
            ns = attrs.get("NetworkSettings")
            # Docker always reports the `Networks` map; a container whose
            # attrs do not carry it (a stub, a half-inspected object) is NOT
            # evidence of a cut-off — reading absence as "cut off" sent every
            # test fake down the provisioning path (R3 round 2).
            if not isinstance(ns, dict) or "Networks" not in ns:
                return False
            return not (ns.get("Networks") or {})
        except Exception:  # noqa: BLE001 — attrs may be stubbed
            return False

    def _cut_off_recreate_due(self, now=None) -> bool:
        now = time.time() if now is None else now
        return (now - float(self._cut_off_at or 0.0)) >= self._CUT_OFF_RECREATE_BACKOFF_S

    def _recreate_if_cut_off(self) -> bool:
        """Drop a cut-off container so `_ensure_running_impl` provisions a
        fresh one (fresh network, enforcement retried). True when it did.
        Backed off by `_CUT_OFF_RECREATE_BACKOFF_S` so a persistent fault
        does not re-provision on every call."""
        if (self.container is not None and self._container_cut_off(self.container)
                and self._cut_off_recreate_due()):
            pretty_log("Sandbox Egress",
                       "the container has no network attached (cut off) — "
                       "recreating it to retry Tor-only enforcement",
                       level="WARNING", icon=Icons.WARN)
            self._cut_off_at = time.time()
            self._set_egress_state("")
            self._cut_off = False
            # A recreate is a NEW CONTAINER GENERATION, and enforcement is
            # per-generation: without this reset `_enforce_egress_once` is a
            # no-op on the replacement and it comes up with no Tor rules at
            # all. Every other generation boundary (resume, create) already
            # resets it; this one did not (§4GK round 4).
            self._tor_attempted = False
            self._privilege_checked = False
            self.container = None
            # Belt: `_ready_is_fresh` also requires a container, but making the
            # stamp stale here means the ordering holds even if that changes.
            self.invalidate_ready()
            return True
        return False

    def _container_network_mode(self) -> str:
        try:
            self.container.reload()
            return str(((self.container.attrs or {}).get("HostConfig") or {}).get("NetworkMode") or "")
        except Exception:  # noqa: BLE001
            return ""

    def _enforce_egress_once(self) -> None:
        """Enforce Tor-only egress for THIS container generation, once.

        The single entry point: creation and resume both call it, and the
        AST pin in `tests/test_sandbox_resume_egress.py` requires any future
        path that starts a container to do the same. Cheap and idempotent
        when the generation has already been enforced.
        """
        self._settle_privileges_once()
        if not self.tor_proxy or self._tor_attempted:
            return
        self._tor_attempted = True
        self._enforce_tor_egress()

    #: Intended create-time privilege set (§4KF), used to REPORT drift on a
    #: container this process adopted rather than created.
    _privilege_checked = False
    _privilege_drift = ""

    def _intended_privileges(self) -> dict:
        drop_on = os.environ.get("GHOST_SANDBOX_DROP_CAPS", "1").strip().lower() not in ("0", "false", "no", "off")
        return {
            "cap_drop": ["ALL"] if drop_on else [],
            "security_opt": ["no-new-privileges"] if drop_on else [],
            "network_mode": (getattr(self, "network_override", None)
                             or os.environ.get("GHOST_SANDBOX_NETWORK", "").strip().lower()
                             or "bridge"),
        }

    def _settle_privileges_once(self) -> None:
        """Once per container generation (§4KF): (1) strip the
        ``ALL ALL=(ALL) NOPASSWD: ALL`` sudoers grant IN PLACE — the cached
        ``ghost-agent-base:latest`` is a runtime commit that still carries it
        twice, and provisioning is skipped when the marker is present, so the
        recipe change alone never reaches a recreated container; (2) compare
        the adopted container's HostConfig with the intended privilege set
        and say so ONCE at WARNING when they differ — flags apply at create,
        so a container that outlived a deploy keeps its old ones silently
        otherwise. Never raises; never recreates (that discards state and is
        the operator's call: remove the container and the next turn rebuilds
        it with the current defaults)."""
        if self._privilege_checked or not getattr(self, "container", None):
            return
        self._privilege_checked = True
        try:
            code, out = self._exec_run(
                "sh -c " + shlex.quote(
                    "grep -qF 'NOPASSWD: ALL' /etc/sudoers 2>/dev/null && "
                    "sed -i '/NOPASSWD: ALL/d' /etc/sudoers && echo stripped || echo clean"),
                user="root")
            if code == 0 and b"stripped" in (out or b""):
                pretty_log("Sandbox Privileges", "removed the NOPASSWD sudoers grant left by the cached image",
                           icon=Icons.SANDBOX_BOX)
        except Exception as exc:  # noqa: BLE001
            logger.debug("sudoers strip skipped: %s", exc)
        try:
            self.container.reload()
            attrs = self.container.attrs
            hc = attrs.get("HostConfig") if isinstance(attrs, dict) else None
            if not isinstance(hc, dict):
                # A stub or a half-inspected object is NOT evidence of drift
                # (the same rule `_recreate_if_cut_off` learned: reading a
                # fake's absence as a state sent every test double down the
                # remedy path).
                return
            actual = {
                "cap_drop": [str(c).upper() for c in (hc.get("CapDrop") or [])],
                "security_opt": [str(o) for o in (hc.get("SecurityOpt") or [])],
                "network_mode": str(hc.get("NetworkMode") or ""),
            }
            want = self._intended_privileges()
            drift = []
            if set(actual["cap_drop"]) != set(want["cap_drop"]):
                drift.append(f"cap_drop {actual['cap_drop'] or 'none'} (intended {want['cap_drop'] or 'none'})")
            if set(actual["security_opt"]) != set(want["security_opt"]):
                drift.append(f"security_opt {actual['security_opt'] or 'none'} (intended {want['security_opt'] or 'none'})")
            if actual["network_mode"] and actual["network_mode"] != want["network_mode"]:
                drift.append(f"network {actual['network_mode']} (intended {want['network_mode']})")
            self._privilege_drift = "; ".join(drift)
            if drift:
                pretty_log(
                    "Sandbox Privileges",
                    f"container {self.container_name} predates the current privilege defaults: "
                    f"{self._privilege_drift}. Flags apply at creation — remove the container "
                    f"(docker rm -f {self.container_name}) and the next turn recreates it hardened.",
                    level="WARNING", icon=Icons.WARN)
        except Exception as exc:  # noqa: BLE001 — attrs may be stubbed
            logger.debug("privilege drift check skipped: %s", exc)

    def _enforce_tor_egress(self) -> None:
        """Make every connection the sandbox opens leave through Tor.

        Fail-closed by construction: the rules go in BEFORE Tor is known
        to be up, so from that moment nothing leaves except through the
        TransPort; if Tor never bootstraps (or dies later) the sandbox is
        offline, not exposed. The rules are loaded through a PRIVILEGED
        exec — the container's own capability set has no NET_ADMIN, so the
        model, root or not, cannot undo them. Never raises: a sandbox that
        cannot be enforced logs at ERROR and stays blocked.
        """
        from . import tor_egress as _te
        if self._container_network_mode() == "host":
            self._set_egress_state("unavailable", "host_networking")
            # §4GJ: name the CAUSE, once, at boot, at the level an operator
            # reads — and name a remedy that can actually be applied. This is
            # a CONFIGURATION state, not a transient fault: the container
            # cannot be cut off (it IS the host's namespace), so every
            # network-capable tool call is refused from here on, and telling
            # the operator to "recreate the sandbox" (the other unavailable
            # branch's advice) would be useless.
            pretty_log(
                "Sandbox Egress",
                "host networking: the sandbox shares the host's network namespace, "
                "so transparent Tor enforcement is NOT applied (rules here would "
                "rewrite the host's traffic) and the container cannot be cut off. "
                "Under --mandatory-tor every sandbox execute/browser call will be "
                "REFUSED until this is changed: set GHOST_SANDBOX_NETWORK=bridge "
                "and recreate the sandbox.",
                level="CRITICAL", icon=Icons.FAIL,
            )
            return
        try:
            if self._exec_run("sh -c 'command -v iptables && command -v tor'")[0] != 0:
                self._block_egress_hard(
                    "iptables or tor missing in the image — Tor-only egress NOT enforced "
                    "(provisioning incomplete; the v9 image carries both)")
                return
            self._exec_run(_te.write_torrc_cmd(), user="root")
            self._exec_run(_te.start_tor_cmd(), user="root")
            # Rules first — the fail-closed moment. Privileged exec: the
            # container has no NET_ADMIN of its own.
            code, out = self._exec_run(_te.apply_rules_cmd(), privileged=True)
            if code != 0:
                self._block_egress_hard(
                    f"iptables rules could not be loaded (exit {code}: "
                    f"{(out or b'').decode('utf-8', 'replace')[:160]}) — Tor-only egress "
                    "NOT enforced")
                return
            self._set_egress_state("blocked")
            if self._exec_run(_te.tor_running_as_expected_cmd())[0] != 0:
                pretty_log(
                    "Sandbox Egress",
                    f"rules loaded but Tor is not running as {_te.TOR_USER} — the sandbox "
                    "network is BLOCKED until it does (fail-closed).",
                    level="WARNING", icon=Icons.WARN,
                )
                return
            pretty_log("Sandbox Egress", "Tor-only rules loaded; waiting for the in-container Tor to bootstrap…",
                       icon=Icons.TOOL_DOWN)
            deadline = time.time() + _te.BOOTSTRAP_TIMEOUT_S
            booted = False
            while time.time() < deadline:
                if self._exec_run(_te.bootstrapped_cmd())[0] == 0:
                    booted = True
                    break
                time.sleep(_te.BOOTSTRAP_POLL_S)
            if not booted:
                pretty_log(
                    "Sandbox Egress",
                    f"Tor did not bootstrap within {int(_te.BOOTSTRAP_TIMEOUT_S)}s — the sandbox "
                    "network stays BLOCKED (fail-closed); it opens by itself when Tor is up.",
                    level="WARNING", icon=Icons.WARN,
                )
                return
            code, out = self._exec_run(_te.verify_cmd(), deadline_s=60.0)
            is_tor, ip = _te.parse_tor_check((out or b"").decode("utf-8", "replace"))
            if is_tor:
                self._set_egress_state("enforced")
                self._egress_exit_ip = ip
                pretty_log("Sandbox Egress", f"Tor-only egress ENFORCED — the sandbox exits via {ip}",
                           icon=Icons.OK)
            elif is_tor is False:
                # ⚠ CORROBORATE BEFORE ACTING (§4GK round 5). Round 4 made this
                # branch destructive — it disconnects the container now and the
                # recreate removes and reprovisions it 300 s later, taking every
                # in-sandbox service and every promoted job with it — while the
                # trigger stayed ONE answer from a third-party endpoint. That
                # endpoint is measurably unreliable for this sandbox (the live
                # log carries 6 unusable answers against 143 enforcements), and
                # the branch is also reached from the RESUME path, whose whole
                # purpose is to preserve those services. A single re-probe costs
                # one exec and removes the transient false positive; two
                # independent requests both answering "not Tor" is a leak.
                _code2, _out2 = self._exec_run(_te.verify_cmd(), deadline_s=60.0)
                _is_tor2, _ip2 = _te.parse_tor_check(
                    (_out2 or b"").decode("utf-8", "replace"))
                # ⚠ ONLY AN EXPLICIT "YES, TOR" CLEARS A MEASURED LEAK
                # (§4GK round 6). The first version cleared on anything that
                # was not an explicit second `false` — so an exec that never
                # ran, a timeout, an infra error or an HTML challenge on the
                # SECOND request turned a confirmed direct exit back into
                # "enforced", and logged that the first answer was the bad one.
                # The rule the docstring states is "two independent requests
                # both answering not-Tor is a leak"; its negation is "a second
                # request that AGREES it is Tor", not "anything else". The
                # cited base rate (6 unusable answers in 143 enforcements) is
                # exactly the probability of masking a real leak.
                if _code2 == 0 and _is_tor2 is True:
                    self._set_egress_state("enforced")
                    self._egress_exit_ip = _ip2 or ip
                    pretty_log(
                        "Sandbox Egress",
                        f"the first verification answered IsTor=false (IP {ip}) but the "
                        f"re-check did not agree ({_is_tor2!r}) — treating the first answer "
                        "as a bad response from the check endpoint, NOT as a leak. The "
                        "rules are loaded; egress stays enforced.",
                        level="WARNING", icon=Icons.WARN,
                    )
                    return
                # ⚠ A MEASURED LEAK MUST BE CLOSED, NOT LABELLED (§4GK round 4).
                # This branch used to call `_set_egress_state("blocked")` — which
                # writes a STRING and touches no network. `egress_is_enforced_or_blocked`
                # then answered True, `egress_gate.network_refusal` returned None,
                # and `execute`/`browser` kept running network work through a
                # container that had just been PROVEN to reach the internet
                # directly with the host's IP. That is the fail-open §4GI exists
                # to close, in its worst form: the state said "blocked" while the
                # measurement said "direct". `_block_egress_hard` disconnects the
                # container for real and is the same remedy every other
                # rules-not-established branch already used.
                pretty_log(
                    "Sandbox Egress",
                    f"LEAK: a plain request from the sandbox reached the internet directly "
                    f"(IsTor=false, IP {ip}) despite the rules — cutting the container off now; "
                    "investigate before using the sandbox.",
                    level="CRITICAL", icon=Icons.FAIL,
                )
                self._block_egress_hard(
                    f"a plain request from the sandbox reached the internet directly (IP {ip})")
            else:
                pretty_log(
                    "Sandbox Egress",
                    "rules loaded and Tor bootstrapped, but the verification request got no "
                    "usable answer (check.torproject.org unreachable or challenged) — treated as "
                    "enforced-unverified; re-checked on the next container generation.",
                    level="WARNING", icon=Icons.WARN,
                )
                self._set_egress_state("enforced")
        except Exception as exc:  # noqa: BLE001 — never take a turn down
            if self._egress_state in ("enforced", "blocked"):
                pretty_log("Sandbox Egress", f"enforcement step raised ({type(exc).__name__}: {exc}) — "
                           f"state={self._egress_state!r}", level="ERROR", icon=Icons.FAIL)
            else:
                # the rules never landed: cut the network rather than leave it direct
                self._block_egress_hard(
                    f"enforcement step raised before the rules landed "
                    f"({type(exc).__name__}: {exc})")

    def _chromium_binary_present(self) -> bool:
        """Check that Playwright's Chromium `headless_shell` is actually
        on disk inside the container.

        We cannot trust `/root/.supercharged*` alone: in the old flow,
        a successful `pip install playwright` (Python library) was the
        gate for marking the image provisioned, even though the
        Chromium BINARY was an entirely separate `playwright install`
        download that often hadn't run. The binary check defends against
        that silent-failure mode regardless of marker state.

        We glob rather than pin a specific Chromium version directory
        because Playwright versions bump chromium-NNNN/ numbers on
        every release.
        """
        if self.container is None:
            return False
        try:
            # `find -print -quit` exits as soon as the first match is
            # printed. Exit code 0 + non-empty stdout → present.
            code, out = self._exec_run(
                "sh -c '"
                "find /root/.cache/ms-playwright -type f "
                "\\( -name headless_shell -o -name chrome \\) "
                "-print -quit 2>/dev/null'"
            )
            if code != 0:
                return False
            stdout = (out or b"").decode("utf-8", errors="replace").strip()
            return bool(stdout)
        except Exception:
            return False

    # NB: no per-exec memory limit — Docker memory is a CONTAINER-level
    # setting (mem_limit from GHOST_SANDBOX_MEM at creation). The old
    # `memory_limit` parameter here was accepted but silently ignored,
    # implying a per-call cap that never applied; removed.
    # Monotonic counter for spill filenames (Date/time are unavailable to keep
    # runs reproducible; a counter is enough for uniqueness within a process).
    _spill_counter = 0
    _spill_counter_seeded = False

    def _spill_run_output(self, text: str):
        """Write the full run output to a log file under the workspace and
        return its workspace-relative path (readable via file_system), or None
        on failure. Bounded at 10 MB so a pathological output can't fill disk."""
        try:
            spill_dir = self.host_workspace / ".ghost_runs"
            spill_dir.mkdir(parents=True, exist_ok=True)
            # Seed the counter past any run_N.log left by a PRIOR process
            # (routine: plain-kill deploy under KeepAlive resets the class
            # counter to 0). Without this, run_1.log is clobbered and a stale
            # "saved to run_1.log" pointer in a long-lived project context now
            # points at unrelated new content. Seed once per process.
            if not getattr(type(self), "_spill_counter_seeded", False):
                _existing = 0
                for _f in spill_dir.glob("run_*.log"):
                    _stem = _f.stem[4:]  # strip "run_"
                    if _stem.isdigit():
                        _existing = max(_existing, int(_stem))
                type(self)._spill_counter = max(type(self)._spill_counter, _existing)
                type(self)._spill_counter_seeded = True
            type(self)._spill_counter += 1
            name = f"run_{type(self)._spill_counter}.log"
            capped = text[: 10 * 1024 * 1024]  # 10 MB hard ceiling
            # §4GI (2026-09-13): a FIXED name in a FIXED directory under the
            # bind mount, and the counter is announced to the model — the
            # §4DX class docker.py had missed. `mkdir(exist_ok=True)` passes
            # through a symlinked `.ghost_runs`, and `write_text` followed a
            # planted `run_{N+1}.log`. The dir-fd writer refuses a symlink at
            # either component atomically; on refusal the spill is simply
            # not made (the caller keeps the truncated inline output).
            from ..tools.file_system import write_text_nofollow_in_dir
            try:
                write_text_nofollow_in_dir(
                    spill_dir, name, capped.encode("utf-8", "replace").decode("utf-8"))
            except ValueError as ve:
                pretty_log("Sandbox Spill",
                           f"refused to spill run output: {ve}",
                           level="WARNING", icon=Icons.WARN)
                return None
            return f".ghost_runs/{name}"
        except Exception as e:
            logger.debug(f"run-output spill failed (non-critical): {e}")
            return None

    def execute_promotable(self, cmd: str, timeout: int = 600,
                           workdir: str = None,
                           spill_large_output: bool = False,
                           max_output_chars: int = None,
                           label: str = None, project_id=None,
                           cleanup_paths=None, identity: str = None):
        """``execute`` with one difference: a command that is still ALIVE and
        still PROGRESSING when the budget expires is DETACHED as a supervised
        job instead of killed (see :mod:`sandbox.jobs`).

        Returns a 3-tuple ``(output, exit_code, job_entry_or_None)``. A
        non-None ``job_entry`` means the command is still running — the exit
        code is 0 (a promotion is not a failure) and the output is whatever
        it had produced so far.

        Deliberately a SEPARATE method rather than a flag on ``execute``:
        promotion is only ever right for the ``execute`` TOOL's own runs.
        Internal callers (rg/find from file_system, the browser runner, the
        service supervisor's own probes) must keep the classic
        kill-at-the-budget contract, and a stub sandbox manager in a test
        simply won't have this attribute — call sites fall back to
        ``execute``.
        """
        return self._execute_impl(
            cmd, timeout=timeout, workdir=workdir,
            spill_large_output=spill_large_output,
            max_output_chars=max_output_chars,
            promotable=True, job_label=label, job_project_id=project_id,
            job_cleanup_paths=cleanup_paths, job_identity=identity)

    def execute(self, cmd: str, timeout: int = 600, workdir: str = None,
                spill_large_output: bool = False, max_output_chars: int = None,
                quiet: bool = False):
        """Run a command in the container under a hard ``timeout -k 5s``
        budget; returns ``(output, exit_code)``. A budget overrun is killed
        and surfaces as exit 124/137/143 — see :meth:`execute_promotable` for
        the tool path that detaches instead."""
        out, code, _job = self._execute_impl(
            cmd, timeout=timeout, workdir=workdir,
            spill_large_output=spill_large_output,
            max_output_chars=max_output_chars, quiet=quiet)
        return out, code

    #: Optional per-manager ceiling on any command's budget. None = the
    #: caller's timeout stands. An isolated replay sets it, because
    #: `tools/execute.py` passes a module constant (600 s) that is twice
    #: a replay leg's whole budget — and cancelling the leg's coroutine
    #: cannot stop the executor thread the command is running on, so the
    #: container gets force-removed and the workspace deleted while a
    #: process is still writing into it. Clamping at the SANDBOX is the
    #: only layer that works regardless of which caller passes what.
    max_exec_timeout = None

    def _execute_impl(self, cmd: str, timeout: int = 600, workdir: str = None,
                      spill_large_output: bool = False,
                      max_output_chars: int = None,
                      promotable: bool = False, job_label: str = None,
                      job_project_id=None, job_cleanup_paths=None,
                      job_identity=None, quiet: bool = False):
        """Shared body of execute / execute_promotable → ``(output,
        exit_code, job_entry_or_None)``. ``job_entry`` is always None on the
        classic path."""
        # Per-manager command ceiling (see `max_exec_timeout`). FIRST, so
        # every caller is clamped regardless of what it passed.
        _cap = getattr(self, "max_exec_timeout", None)
        if _cap:
            try:
                timeout = max(5, min(int(timeout), int(_cap)))
            except (TypeError, ValueError):
                pass
        try:
            # ensure_running() either just probed readiness (steady path)
            # or raised (provision path) — re-probing here doubled the
            # exec/host-IO overhead of EVERY command for no benefit. If
            # the container dies in the tiny gap before exec_run below,
            # the normal error path surfaces it.
            self.ensure_running()
            # §4GI R3: the belt at the layer where enforcement has DEFINITELY
            # run — the tool-side gate sees only the state before this call.
            if (getattr(self, "tor_proxy", None) and getattr(self, "_egress_state", "")
                    and not self.egress_is_enforced_or_blocked()):
                from .egress_gate import _refusal_text
                pretty_log("Sandbox Egress", "refusing command — egress unavailable",
                           level="ERROR", icon=Icons.SHIELD)
                return _refusal_text(self), 1, None

            # Promotable runs are supervised by sandbox/jobs.py, which owns
            # the budget itself (it has to still be holding the process when
            # the budget expires, which `timeout` never is — it has already
            # killed it). Resolved BEFORE the command string is built so the
            # log line matches what actually runs.
            job_sup = None
            if promotable:
                from .jobs import get_job_supervisor, jobs_enabled
                if jobs_enabled():
                    job_sup = get_job_supervisor(self)

            # Add -k 5s to ensure processes are killed if they ignore SIGTERM
            cmd_string = (cmd if job_sup is not None
                          else f"timeout -k 5s {timeout}s {cmd}")
            # `quiet` is for the job supervisor's own bookkeeping probes
            # (liveness, /proc I/O counters, kills). Those run every 30 s per
            # running job and once a minute per job from the reaper — logging
            # a 500-char `sh -c 'for f in /proc/…'` blob each time would bury
            # the stream the operator actually watches, in a subsystem whose
            # whole point is to be quiet in the background. They still reach
            # the durable log via logger.debug.
            if quiet:
                logger.debug("sandbox probe: %s", cmd_string[:200])
            else:
                pretty_log("Sandbox Exec", f"Command: {cmd_string}",
                           icon=Icons.TOOL_SHELL)

            # Cross-platform safe UID/GID fetching (Windows doesn't have getuid)
            user_id = os.getuid() if hasattr(os, 'getuid') else 1000
            group_id = os.getgid() if hasattr(os, 'getgid') else 1000
            
            import sys
            is_mac = sys.platform == "darwin"
            
            # workdir defaults to /workspace; a project-scoped caller passes
            # /workspace/projects/<id> so files written/run during a project
            # land under that subdir (easy per-project cleanup). The path is
            # under the bind-mounted root, so it exists in the container.
            exec_kwargs = {
                "workdir": workdir or CONTAINER_WORKDIR,
                "demux": False
            }
            if not is_mac:
                exec_kwargs["user"] = f"{user_id}:{group_id}"
            
            job_entry = None
            if job_sup is not None:
                # Detached-and-polled: the job supervisor launches the command
                # under setsid, watches it from the HOST side of the bind
                # mount, and at the budget either promotes it (still alive,
                # still progressing) or kills it exactly like `timeout` would
                # have (exit 124). No client deadline is needed — the poll
                # loop is ours and is bounded by the budget.
                stdout_bytes, exit_code, job_entry = job_sup.run(
                    cmd, timeout=timeout, workdir=exec_kwargs["workdir"],
                    label=job_label, project_id=job_project_id,
                    exec_kwargs=exec_kwargs,
                    cleanup_paths=job_cleanup_paths,
                    identity=job_identity)
            else:
                # The command self-limits via the in-container `timeout -k 5s Ns`
                # wrapper, so the client deadline only needs to catch a WEDGED
                # daemon (which never streams the process's EOF back): timeout +
                # grace. Without it a stuck daemon hangs this worker thread
                # forever.
                #
                # Output is STREAMED through a bounded head+tail sink so a
                # runaway producer (`yes`, `cat bigfile`) can't buffer 100s of
                # MB in agent RAM (the jobs path already caps at 32 MB; this
                # brings rg/find, the browser runner, execute.py heal retries
                # and GHOST_SANDBOX_JOBS=0 to parity). Normal-sized output is
                # byte-identical to the buffered call. Streaming is used ONLY
                # for a REAL container (a string `.id`): a MagicMock/None
                # container (tests) or the kill switch takes the buffered
                # `_exec_run` path, and any non-wedge streaming fault also falls
                # back to it, so a docker-py API shift can't take out execute().
                _cid = getattr(self.container, "id", None)
                _streamed = False
                if _EXEC_STREAM_ENABLED and isinstance(_cid, str) and _cid:
                    try:
                        stdout_bytes, exit_code = self._exec_run_streamed(
                            cmd_string, cid=_cid, ram_cap=_CLASSIC_EXEC_RAM_CAP,
                            deadline_s=timeout + 60, **exec_kwargs)
                        _streamed = True
                    except SandboxDaemonTimeout:
                        raise
                    except Exception as _stream_err:  # noqa: BLE001
                        logger.warning(
                            "streamed classic exec failed (%s: %s) — falling "
                            "back to buffered exec_run",
                            type(_stream_err).__name__, _stream_err)
                if not _streamed:
                    exec_result = self._exec_run(
                        cmd_string,
                        deadline_s=timeout + 60,
                        **exec_kwargs
                    )
                    stdout_bytes = exec_result.output
                    exit_code = exec_result.exit_code

            # Output handling. A sandbox script that prints multi-MB to stdout
            # would flood the model context with 100k+ tokens of garbage (and
            # the whole blob is already in RAM via exec_result.output). Two
            # modes:
            #   - spill_large_output (the execute TOOL path): keep the returned
            #     view SMALL (max_output_chars, default 24 KB head+tail) and
            #     write the FULL output to a run-log file under the workspace so
            #     the model can inspect it with file_system — truncation becomes
            #     an affordance instead of information loss.
            #   - default (rg/find/browser via sandbox_manager.execute): the
            #     legacy 256 KB head+tail, no spill, so those callers are
            #     unchanged.
            # pylint: disable=possibly-used-before-assignment
            # Both names are bound on every path into this read: the job
            # branch unpacks them, and in the else branch `_streamed` is the
            # guard — it is only True after the streamed call assigned both,
            # and `if not _streamed` binds them from the buffered exec
            # otherwise. pylint cannot follow the flag across the two
            # statements (§4GJ triage).
            output = ""
            if stdout_bytes:
                from ..utils.text_truncate import truncate_head_tail
                decoded = stdout_bytes.decode("utf-8", errors="replace")
                if spill_large_output:
                    budget = max_output_chars or 24 * 1024
                    trimmed, was_trunc, _dropped = truncate_head_tail(
                        decoded, budget, label="run output")
                    if was_trunc:
                        rel = self._spill_run_output(decoded)
                        pointer = (
                            f"\n[Full output ({len(decoded) // 1024} KB) saved to "
                            f"'{rel}' — inspect it with file_system "
                            f"operation='search' (find lines) or "
                            f"operation='read' start_line/end_line.]" if rel else ""
                        )
                        output = trimmed + pointer
                    else:
                        output = decoded
                else:
                    MAX_OUTPUT_CHARS = 256 * 1024  # 256 KB legacy cap
                    trimmed, _was, _dropped = truncate_head_tail(
                        decoded, MAX_OUTPUT_CHARS, label="sandbox 256KB cap",
                        head_frac=0.5)
                    output = trimmed

            if not output.strip() and exit_code != 0:
                 output = f"[SYSTEM ERROR]: Process failed (Exit {exit_code}) with no output."
            # A promoted job legitimately has no output yet (a silent
            # downloader promoted on file growth) — that is not an error, and
            # the caller's promotion banner supplies the explanation.
            if job_entry is not None and not output.strip():
                output = "(no output yet)"

            # Readiness TTL bookkeeping. exec_run returning at all means the
            # daemon + container are live, so a normal command (even one that
            # exits non-zero — a failing script is not an infra fault) confirms
            # readiness. Exit 126/127/128 are the OCI-level codes that a
            # deleted/recreated mount inode produces, so those INVALIDATE
            # instead — forcing a full reprobe (and reprovision) next call.
            if exit_code in (126, 127, 128):
                self.invalidate_ready()
            else:
                self.mark_ready()

            return output, exit_code, job_entry

        except Exception as e:
            # The container/daemon may be gone — force a full probe next time.
            self.invalidate_ready()
            # Mark this as a SANDBOX/INFRA failure, not a program failure. The
            # blanket exit 1 made an infra fault (a wedged daemon, the
            # remove-while-exec race, the provision-backoff refusal) look like
            # the model's own code failing, so it debugged its code and burned
            # strikes on a sandbox condition. The `[SANDBOX INFRA ERROR]` prefix
            # tells the model (and keeps execute.py's file-not-found heal from
            # firing on it — the heuristic doesn't match this text). A wedged
            # daemon gets its own explicit line.
            _wedged = isinstance(e, SandboxDaemonTimeout)
            pretty_log(
                "Sandbox Daemon Wedged" if _wedged else "Sandbox Exec Failed",
                f"{type(e).__name__}: {e}", icon=Icons.FAIL, level="ERROR")
            return (
                f"[SANDBOX INFRA ERROR — not your code] "
                f"{'docker daemon wedged; ' if _wedged else ''}{str(e)}", 1,
                None)

    #: Most containers one boot sweep will remove. A runaway backstop, not
    #: a policy — if a box ever has more than this many orphans, the
    #: operator should look rather than have them silently vanish.
    _SWEEP_CAP = 25

    #: A per-solve sandbox younger than this is assumed to belong to a
    #: solve that is genuinely in flight (a second agent process, a dev
    #: run) and is left alone. True orphans persist for hours to days.
    _SWEEP_MIN_AGE_S = 1800

    #: A ``ghostjobs-*`` detached-job container younger than this is spared
    #: unconditionally — a job's whole life is at most the exec budget plus
    #: job_ttl_s (bounded to 6h), so this is many multiples past any job
    #: ceiling. Only a container older than this is even a reap CANDIDATE, and
    #: it still has to pass the no-running-job and idle-process checks.
    _GHOSTJOB_REAP_MIN_AGE_S = float(
        os.environ.get("GHOST_SANDBOX_GHOSTJOB_MAX_AGE_H", "48") or 48) * 3600.0

    #: Client-side deadline for the ghostjob liveness probe. A wedged candidate
    #: must not hang the boot sweep; a timeout reads as LIVE (spare).
    _GHOSTJOB_LIVENESS_DEADLINE_S = 15.0

    def _is_per_solve_workspace(self, source: str) -> bool:
        """True when ``source`` is a throwaway per-solve workspace.

        THE CRITERION, and the first version of it was WRONG. I first
        swept "containers whose mount no longer exists on disk", on the
        theory that the workspace dies with the solve. A dry run against
        the real box refuted it: a SIGKILL is exactly the case that
        orphans a container, and it is also exactly the case where Python
        cannot run `TemporaryDirectory.cleanup()` — so the workspace
        SURVIVES. The criterion spared every orphan it was written for.

        What actually separates them is the KIND of workspace:

        * per-solve  → `tempfile.TemporaryDirectory()`, i.e. a `tmp*`
          basename under the system temp root. Nothing may outlive its
          solve, so one present at boot is by definition a leftover.
        * the agent's own sandbox → `$GHOST_HOME/sandbox`, a stable path
          outside the temp root. Never matches.
        * a DETACHED JOB → `ghostjobs-*`, which is designed to survive
          agent restarts (§4AX: a still-working command is detached, not
          killed). Sweeping those would destroy running work, and one was
          live on the box while this was written.
        """
        try:
            import tempfile
            root = os.path.realpath(tempfile.gettempdir())
            real = os.path.realpath(source)
            if os.path.commonpath([real, root]) != root:
                return False            # outside the temp root entirely
            base = os.path.basename(real)
            # `tmp` prefix = tempfile.TemporaryDirectory; anything else
            # under the temp root (ghostjobs-*, hand-made dirs) is NOT a
            # per-solve workspace and must be left alone.
            return base.startswith("tmp")
        except Exception:  # noqa: BLE001
            return False                # cannot tell → not an orphan

    def _owner_is_alive(self, container) -> bool:
        """True when the container carries an owner stamp naming a
        process that is STILL RUNNING on this host, from this boot.

        Without this the age floor is the only protection a per-solve
        container has, and a run legitimately in flight for more than 30
        minutes in another process (an overnight replay, a long bench
        item) is reaped out from under itself — which does not merely
        fail it, it makes the next `ensure_running` recreate the
        container and silently discard all in-sandbox state, so the run
        produces a verdict on a half-executed episode.

        Unlabelled containers (everything created before this shipped)
        answer False, i.e. reapable — the pre-label behaviour."""
        try:
            labels = ((container.attrs or {}).get("Config", {})
                      or {}).get("Labels") or {}
            pid = labels.get("ghost.owner_pid")
            if not pid:
                return False
            boot = labels.get("ghost.owner_boot") or ""
            if boot and boot != _owner_boot_id():
                return False          # pre-reboot PID: reuse, not owner
            return _pid_is_live(pid)
        except Exception as exc:  # noqa: BLE001
            logger.debug("owner liveness check skipped: %s", exc)
            return False

    def _container_age_s(self, container) -> float:
        """Seconds since creation; -1.0 when it cannot be determined
        (which the caller treats as "too young to touch")."""
        try:
            import datetime as _dt
            raw = str((container.attrs or {}).get("Created") or "")
            if not raw:
                return -1.0
            raw = raw.replace("Z", "+00:00")
            # docker reports nanoseconds; fromisoformat wants ≤6 digits
            if "." in raw:
                head, _, tail = raw.partition(".")
                frac = "".join(c for c in tail if c.isdigit())[:6]
                off = tail[len(frac):] if len(tail) > len(frac) else ""
                off = off.lstrip("0123456789")
                raw = f"{head}.{frac or '0'}{off or '+00:00'}"
            created = _dt.datetime.fromisoformat(raw)
            now = _dt.datetime.now(_dt.timezone.utc)
            return (now - created).total_seconds()
        except Exception:  # noqa: BLE001
            return -1.0

    def _workspace_has_running_job(self, ws_root) -> bool:
        """True when ``<ws_root>/.jobs/registry.json`` has a job row still in
        the ``running`` state — a LIVE detached job that must never be swept.
        A missing registry means no running job; any OTHER read/parse failure
        is treated as "cannot tell → assume live" and spares the container."""
        import json
        try:
            reg = os.path.join(ws_root, ".jobs", "registry.json")
            with open(reg, "r") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            return False           # no registry → no running job
        except Exception:  # noqa: BLE001
            return True            # unreadable → assume live, spare
        if not isinstance(data, dict):
            return True
        for row in data.values():
            if isinstance(row, dict) and str(row.get("state")) == "running":
                return True
        return False

    def _container_has_live_process(self, container) -> bool:
        """True (=> SPARE) unless the container can be POSITIVELY read as idle:
        an exec listing ONLY PID 1 + a sleep, or an exec that raises because
        the container is gone/stopped (nothing alive to protect). Anything
        unparseable is treated as LIVE. The exec is bounded by a short client
        deadline so a wedged candidate cannot hang the boot sweep — a timeout
        reads as LIVE."""
        box = {}

        def _probe():
            try:
                box["res"] = container.exec_run("ps -eo pid,comm --no-headers")
            except BaseException as e:  # noqa: BLE001 — inspected below
                box["err"] = e

        t = threading.Thread(target=_probe, name="ghostjob-liveness",
                             daemon=True)
        t.start()
        t.join(timeout=self._GHOSTJOB_LIVENESS_DEADLINE_S)
        if t.is_alive():
            return True            # probe wedged → cannot confirm idle → spare
        if "err" in box:
            err = box["err"]
            msg = str(err).lower()
            gone = (type(err).__name__ in ("NotFound", "APIError",
                                           "NullResource")
                    or "not running" in msg
                    or "no such container" in msg
                    or "is not running" in msg)
            return not gone        # gone/stopped → not live; else uncertain
        res = box.get("res")
        raw = getattr(res, "output", None)
        if raw is None and isinstance(res, tuple) and len(res) == 2:
            raw = res[1]
        if isinstance(raw, (bytes, bytearray)):
            text = raw.decode("utf-8", "replace")
        elif isinstance(raw, str):
            text = raw
        else:
            return True            # unparseable → cannot confirm idle → spare
        if not text.strip():
            return True            # empty read → cannot confirm → spare
        idle = {"sleep", "docker-init", "ps", "sh", "bash", "cat", "tini"}
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            comm = (parts[1] if len(parts) > 1 else parts[0]).strip()
            comm = comm.split()[0] if comm else ""
            base = os.path.basename(comm)
            if base and base not in idle:
                return True        # a real process → LIVE → spare
        return False               # only PID1 + sleep/ps → idle → reapable

    def _is_reapable_dead_ghostjob(self, container, sources, age_s) -> bool:
        """A ``ghostjobs-*`` detached-job container is reapable ONLY when it is
        UNAMBIGUOUSLY a dead leftover — the default is always to spare.

        ALL of: kill switch on; EVERY mount a ``ghostjobs-*`` dir under the
        system temp root; older than ``_GHOSTJOB_REAP_MIN_AGE_S``; NO
        ``running`` row in its ``.jobs`` registry; and an idle process table
        (or a gone container). A LIVE detached job — a running registry row OR
        a real in-container process — is ALWAYS spared, as is the agent's own
        sandbox (a non-ghostjobs mount) and any container we cannot read."""
        if os.environ.get("GHOST_SANDBOX_REAP_GHOSTJOBS", "1") == "0":
            return False
        import tempfile
        try:
            root = os.path.realpath(tempfile.gettempdir())
        except Exception:  # noqa: BLE001
            return False
        ghost_ws = []
        for s in sources:
            try:
                real = os.path.realpath(s)
                if os.path.commonpath([real, root]) != root:
                    return False       # a mount outside the temp root
            except Exception:  # noqa: BLE001
                return False
            if not os.path.basename(real).startswith("ghostjobs-"):
                return False           # a non-ghostjobs mount → not our case
            ghost_ws.append(real)
        if not ghost_ws:
            return False
        if age_s < self._GHOSTJOB_REAP_MIN_AGE_S:
            return False               # too young — a job may be in flight
        if any(self._workspace_has_running_job(w) for w in ghost_ws):
            return False               # a LIVE detached job — never touch
        if self._container_has_live_process(container):
            return False               # something real is running inside
        return True

    def sweep_orphaned_containers(self, max_remove: int = None) -> list:
        """Remove per-solve sandbox containers left behind by a kill
        mid-solve. Returns the names removed. Never raises — this runs at
        boot, where an exception is a dead agent.

        WHY (§4BO, 2026-08-15): the per-solve teardown now closes its
        sandbox properly, but a `finally` cannot run through SIGKILL
        (`launchctl kickstart -k`, a crash, a reboot). Those containers
        keep running against a workspace nothing will ever look up again;
        two were live on the box, aged 3 days and 43 minutes.

        SAFETY. A container is removed only when ALL of:
          * the name is `ghost-agent-sandbox-*` and not this instance's;
          * EVERY mount is a per-solve workspace (see
            `_is_per_solve_workspace` — this is what protects the agent's
            own sandbox and, critically, detached JOB containers);
          * it is older than `_SWEEP_MIN_AGE_S`, so a solve genuinely in
            flight in another process is never touched;
          * `GHOST_SANDBOX_SWEEP` is not "0".
        A container with no mounts is left alone — "cannot tell" must not
        read as "safe to delete". Running-vs-stopped is deliberately not
        part of the test: a per-solve container outliving its solve is
        orphaned whether or not a process is still inside it.
        """
        removed: list = []
        if os.environ.get("GHOST_SANDBOX_SWEEP", "1") != "1":
            logger.debug("sandbox sweep disabled by GHOST_SANDBOX_SWEEP=0")
            return removed
        cap = self._SWEEP_CAP if max_remove is None else max_remove
        try:
            containers = self.client.containers.list(all=True)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"sandbox sweep listing failed: {e}")
            return removed
        for c in containers:
            if len(removed) >= cap:
                logger.warning(
                    "sandbox sweep stopped at the %d-container cap — more "
                    "orphans remain; inspect `docker ps -a` by hand", cap)
                break
            try:
                name = getattr(c, "name", "") or ""
                if not name.startswith("ghost-agent-sandbox-"):
                    continue
                if name == getattr(self, "container_name", None):
                    continue
                sources = [m.get("Source") for m
                           in ((c.attrs or {}).get("Mounts") or [])
                           if m.get("Source")]
                if not sources:
                    continue          # cannot tell → leave it
                age = self._container_age_s(c)
                if all(self._is_per_solve_workspace(s) for s in sources):
                    if age < self._SWEEP_MIN_AGE_S:
                        continue      # may belong to a live solve
                    if self._owner_is_alive(c):
                        continue      # in flight in ANOTHER process
                elif not self._is_reapable_dead_ghostjob(c, sources, age):
                    # own sandbox, a LIVE detached job, or anything we cannot
                    # read as unambiguously dead — spare it.
                    continue
                c.remove(force=True)
                removed.append(name)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"sandbox sweep skipped {c}: {e}")
                continue
        return removed

    def close(self, remove: bool = False):
        """Tear down the sandbox container at agent shutdown.

        ``remove=False`` (the default) just stops the container, so the next
        start is a fast resume on the already-provisioned image. Pass
        ``remove=True`` to discard the container entirely (e.g. during
        tests or when the provisioning state is known to be corrupt).

        Failure is logged but never raised — close() is expected to run
        from signal handlers / shutdown hooks where exceptions are
        disruptive.
        """
        # ⚠ THE CLIENT CLOSE MUST SURVIVE EVERY EARLY EXIT (§4BO). The
        # lookup below returns early when no container exists — which is
        # the COMMON per-solve case — and that skipped the finally
        # entirely, so the first version of this fix still leaked. Own
        # try/finally, wrapping everything.
        try:
            self._close_container(remove)
        except Exception as e:  # noqa: BLE001
            # The docstring promises this never raises, and both callers
            # are teardown paths (a solve's `finally`, the RSS watchdog).
            # It DID raise: an exception in the container work propagated
            # straight through the finally below. Making the promise true
            # rather than deleting it — a teardown that throws is how one
            # failing container takes an idle loop with it.
            logger.debug(f"Sandbox close() container step failed: {e}")
        finally:
            if remove:
                self.container = None
                # `__init__` opens a client per instance via `from_env()`
                # and dream.py builds a FRESH DockerSandbox per solve, so
                # an unclosed pool leaks its unix sockets to
                # /var/run/docker.sock every time. Measured live: 228 unix
                # FDs after ~20 bench items, then `[Errno 24] Too many
                # open files` — the remaining items failed to solve AND
                # failed to append their ledger rows.
                #
                # Only on remove=True: remove=False means "stop but keep
                # it warm for a fast resume" and that instance must stay
                # usable. After remove=True the instance is dead.
                try:
                    closer = getattr(self.client, "close", None)
                    if callable(closer):
                        closer()
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"Sandbox client close failed: {e}")

    def _close_container(self, remove: bool):
        """Stop (and optionally remove) the container.

        ⚠ NOT thread-safe, and deliberately does NOT take `self._lock`.
        On the §4BO per-item timeout path the solve's `execute` may still
        be running in its own thread when this fires, so close+use can
        overlap on the `requests.Session` — noisy errors in the doomed
        exec, but no FD leak (a checked-out urllib3 connection is closed
        when returned to a cleared pool). Taking the lock would be worse:
        teardown would block for up to `_EXEC_DAEMON_DEADLINE_S`, which
        is exactly the wedge the timeout exists to end.
        """
        container = getattr(self, "container", None)
        if container is None:
            # Best-effort: maybe a container with our name exists from a
            # previous run that never bound to `self.container` — the
            # provisioning-failed-midway case, which is what an FD
            # exhaustion produces.
            try:
                container = self.client.containers.get(self.container_name)
            except self.NotFound:
                return
            except Exception as e:
                logger.debug(f"Sandbox close() lookup failed: {e}")
                return

        try:
            container.reload()
            if container.status == "running":
                try:
                    container.stop(timeout=5)
                except self.APIError as e:
                    logger.warning(f"Sandbox stop failed for {self.container_name}: {e}")
            if remove:
                try:
                    container.remove(force=True)
                except self.APIError as e:
                    logger.warning(f"Sandbox remove failed for {self.container_name}: {e}")
        except self.NotFound:
            pass
        except Exception as e:
            logger.debug(f"Sandbox close() failed: {e}")

# ── Lazy sandbox re-init (2026-08-26) ────────────────────────────────
# Boot init is ONE-SHOT: main.py constructs the live DockerSandbox once,
# and the constructor pings the daemon. When the agent boots seconds
# before the docker socket appears (08-26: agent up at 08:11:52,
# OrbStack's socket answering at ~08:12), the constructor raises,
# `context.sandbox_manager` is never assigned, and every execute/browser
# call fails with "Sandbox manager not initialized" until the next
# restart — 7 hours that day, while hourly isolated bench runs proved
# docker had recovered minutes after boot. These two functions are the
# retry path: registry.py calls ensure_sandbox_manager() when the live
# context has no manager, and a successful (backoff-gated) construction
# is assigned back onto the context so every other reader — file_system,
# manage_services, the tree lister — recovers on the same turn.
#
# Identity guard: ONLY the exact context object registered at boot is
# eligible. isolated_replay_context sets sandbox_manager=None on its
# copy.copy'd context ON PURPOSE (a network=none replay must never touch
# docker, §4CL), and a dream/bench fork owns its own explicitly-managed
# sandbox — a weakref identity check excludes every copy for free, so
# this path can never resurrect a sandbox that isolation deliberately
# detached.

_LAZY_REINIT_BACKOFF_S = 60.0
_lazy_ctx_ref = None            # weakref to THE live context (set at boot)
_lazy_lock = threading.Lock()   # one attempt at a time — never queue
_lazy_next_attempt = 0.0        # time.monotonic() gate between attempts


def register_lazy_sandbox(context) -> None:
    """Mark ``context`` as the one object whose ``sandbox_manager`` may be
    lazily (re)constructed. Called from main.py boot inside the
    ``find_spec("docker")`` branch — a docker-less install never
    registers, so ensure_sandbox_manager() stays a no-op there."""
    global _lazy_ctx_ref, _lazy_next_attempt
    import weakref
    _lazy_ctx_ref = weakref.ref(context)
    _lazy_next_attempt = 0.0


def _docker_endpoint_plausible() -> bool:
    """Cheap pre-check before paying for a DockerSandbox construction.

    Without any daemon endpoint the constructor's ping raises anyway, but
    when the daemon is WEDGED (socket present, not answering) the ping can
    block for the client timeout — so when NO plausible endpoint exists,
    skip the attempt for the price of a few stat() calls. A remote
    DOCKER_HOST (tcp/ssh) can't be stat'd; let the ping decide there."""
    host = os.environ.get("DOCKER_HOST", "").strip()
    if host:
        if host.startswith("unix://"):
            return os.path.exists(host[len("unix://"):])
        return True
    candidates = ["/var/run/docker.sock"]
    import sys
    if sys.platform == "darwin":
        candidates += [os.path.expanduser("~/.orbstack/run/docker.sock"),
                       os.path.expanduser("~/.docker/run/docker.sock")]
    return any(os.path.exists(c) for c in candidates)


def close_carried_client(exc) -> None:
    """Close the half-built docker client a failed DockerSandbox
    constructor carries on its exception (§4BO: ~11 unix sockets each;
    any caller that retries construction walks to EMFILE without this).
    Safe on any exception — missing attribute and a close() that raises
    are both swallowed."""
    for attr in ("client", "docker_client"):
        cl = getattr(exc, attr, None)
        try:
            if cl is not None:
                cl.close()
        except Exception:  # noqa: BLE001
            pass


def ensure_sandbox_manager(context):
    """Return a live sandbox manager for ``context``, constructing one if
    boot-time init failed and docker has since recovered.

    Never raises; returns None while the manager stays unavailable. The
    caller's error path ("Sandbox manager not initialized") is unchanged —
    this only shrinks how long that state can last. Blocking (the
    constructor pings the daemon): call via asyncio.to_thread from async
    code. ensure_running is NOT called here — execute() runs it before
    every command, so the container provisions itself on first use."""
    global _lazy_next_attempt
    sm = getattr(context, "sandbox_manager", None)
    if sm is not None:
        return sm
    ref = _lazy_ctx_ref
    if ref is None or ref() is not context:
        return None
    if time.monotonic() < _lazy_next_attempt:
        return None
    if not _lazy_lock.acquire(blocking=False):
        # Another thread is mid-attempt; if it succeeds, its result is
        # already on the context for the caller's NEXT read.
        return None
    try:
        # Re-check under the lock: a racing attempt may have just
        # succeeded (manager set) or failed (backoff armed).
        sm = getattr(context, "sandbox_manager", None)
        if sm is not None:
            return sm
        if time.monotonic() < _lazy_next_attempt:
            return None
        _lazy_next_attempt = time.monotonic() + _LAZY_REINIT_BACKOFF_S
        if not _docker_endpoint_plausible():
            return None
        try:
            candidate = DockerSandbox(context.sandbox_dir,
                                      getattr(context, "tor_proxy", None))
        except Exception as exc:  # noqa: BLE001
            # Boot leaked the carried client at most once; THIS path
            # retries every backoff window, so the close is load-bearing.
            close_carried_client(exc)
            logger.debug("lazy sandbox re-init failed (next attempt in "
                         "%.0fs): %s", _LAZY_REINIT_BACKOFF_S, exc)
            return None
        context.sandbox_manager = candidate
        pretty_log("Sandbox Recovered",
                   "Docker reachable again — sandbox manager rebuilt after "
                   "failed boot-time init (execute/browser re-enabled)",
                   icon=Icons.SANDBOX_BOX)
        return candidate
    finally:
        _lazy_lock.release()
