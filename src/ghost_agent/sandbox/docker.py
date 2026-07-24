import logging
import os
import threading
import time
from pathlib import Path
from ..utils.logging import Icons, pretty_log

logger = logging.getLogger("GhostAgent")

CONTAINER_NAME = "ghost-agent-sandbox"
CONTAINER_WORKDIR = "/workspace"

# Client-side deadline (seconds) for a container exec when the caller gives no
# explicit one. docker-py's exec socket read blocks in poll() with NO timeout
# (verified in 7.1.0), so a wedged daemon would otherwise hang the calling
# thread forever — and the provision execs run while holding self._lock, which
# would wedge EVERY other turn's execute. This ceiling is generous on purpose:
# it must never fire for a legitimately-slow install (apt/pip/playwright), only
# for a genuinely stuck daemon. The per-command exec passes its own tighter
# deadline (the in-container `timeout Ns` budget + grace).
_EXEC_DAEMON_DEADLINE_S = float(os.environ.get("GHOST_EXEC_DAEMON_DEADLINE", "1200") or 1200)


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
    #   _tor_attempted — the in-container tor spawn was already attempted
    #     for this generation (under host networking it can never bind,
    #     the host tor owns :9050 — retrying per command is pure waste).
    #   _provision_backoff_until — after a failed provision, no reinstall
    #     before this wall-clock time; prevents a failing mirror from
    #     triggering a fresh multi-minute install on every command.
    _env_verified = False
    _tor_attempted = False
    _provision_backoff_until = 0.0

    def __init__(self, host_workspace: Path, tor_proxy: str = None):
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
                        raise handle_err
                else:
                    raise handle_err
            else:
                raise handle_err
        self.host_workspace = host_workspace.absolute()
        self.tor_proxy = tor_proxy
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
        ``--network host``): the effective default on Linux. In that mode a
        service the agent hosts binds a real host port, so exporting
        ``HOST=0.0.0.0`` exposes it LAN-wide unauthenticated — sandbox.services
        consults this to bind loopback instead. Mirrors the create-time logic
        (GHOST_SANDBOX_NETWORK override → Linux=host / else bridge)."""
        import sys as _sys
        _net = os.environ.get("GHOST_SANDBOX_NETWORK", "").strip().lower()
        if _net in ("host", "bridge", "none"):
            return _net == "host"
        return _sys.platform.startswith("linux")

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
        except Exception as e:  # noqa: BLE001 — fall through to recreate
            logger.debug("sandbox resume failed (%s); will recreate", e)
            return False
        if self._is_container_ready():
            pretty_log(
                "Sandbox Resume",
                "Resumed stopped container (in-sandbox services + runtime "
                "state preserved)", icon=Icons.SANDBOX_BOX)
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
        if self._ready_is_fresh():
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
                is_linux = sys.platform.startswith("linux")
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

                # Optional capability hardening — OFF by default because the
                # sandbox provisions passwordless sudo (setuid) for in-container
                # apt installs, which `no-new-privileges` / `cap_drop=ALL` would
                # break. Operators who don't need in-sandbox package installs can
                # opt in with GHOST_SANDBOX_DROP_CAPS=1.
                if _os.environ.get("GHOST_SANDBOX_DROP_CAPS", "").lower() in ("1", "true", "yes"):
                    run_kwargs["cap_drop"] = ["ALL"]
                    run_kwargs["security_opt"] = ["no-new-privileges"]

                # Network mode. On Linux the default is `host` because the
                # in-sandbox browser must reach the host's Tor proxy at
                # 127.0.0.1:9050 (bridge would break Tor-routed browsing). This
                # also means sandboxed code shares the host's loopback — set
                # GHOST_SANDBOX_NETWORK=bridge (or none) to ISOLATE when you
                # don't rely on host-loopback services from the sandbox.
                _net = _os.environ.get("GHOST_SANDBOX_NETWORK", "").strip().lower()
                if _net in ("host", "bridge", "none"):
                    run_kwargs["network_mode"] = _net
                    if _net == "bridge" and not is_mac:
                        run_kwargs["extra_hosts"] = {"host.docker.internal": "host-gateway"}
                elif is_linux:
                    run_kwargs["network_mode"] = "host"
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
        #   v5 (now):    .supercharged.v5 — adds `stockfish` (the chess
        #                project's engine-opponent mode; a recreate must
        #                not silently drop the engine). v4 images
        #                re-provision to pick it up.
        marker_path = "/root/.supercharged.v5"

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
            apt_cmd = "timeout 900 sh -c 'apt-get update && apt-get install -y sudo coreutils nodejs npm g++ curl wget git procps postgresql-client libpq-dev tor ripgrep sqlite3 iproute2 stockfish'"
            code, out = self._exec_run(apt_cmd, environment=env_vars)
            if code != 0:
                err_msg = out.decode("utf-8", errors="replace") if out else "Unknown error"
                raise Exception(f"System package installation failed: {err_msg}")

            self._exec_run("sh -c 'echo \"ALL ALL=(ALL) NOPASSWD: ALL\" >> /etc/sudoers'")

            if self.tor_proxy:
                code, out = self._exec_run("timeout 600 pip install --no-cache-dir pysocks requests")
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
            code, out = self._exec_run(install_cmd, environment=env_vars)
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
            torch_code, torch_out = self._exec_run(
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
            pw_code, pw_out = self._exec_run(
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
            # cycle doesn't leave stale state around.
            self._exec_run("rm -f /root/.supercharged")

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

        # Ensure Tor is installed and running inside the container for
        # isolated browser proxying. Attempted once per container
        # generation: under host networking (the Linux default) an
        # in-container tor can NEVER bind :9050 (the host tor owns it in
        # the shared netns), so re-attempting the doomed spawn on every
        # command just added latency and flooded the log with
        # per-command "Environment Ready" lines.
        if self.tor_proxy and not self._tor_attempted:
            self._tor_attempted = True
            exit_code, _ = self._exec_run("test -f /usr/bin/tor")
            if exit_code != 0:
                did_work = True
                pretty_log("Sandbox Tor", "Installing isolated Tor daemon…", icon=Icons.TOOL_DOWN)
                self._exec_run("timeout 900 sh -c 'apt-get update && apt-get install -y tor'", user="root")

            code, _ = self._exec_run("pgrep -x tor")
            if code != 0:
                did_work = True
                self._exec_run("su - debian-tor -s /bin/sh -c 'tor --RunAsDaemon 1'", user="root")
                # Verify it actually came up; under host networking this is
                # EXPECTED to fail (host tor owns the port) — say so once
                # instead of silently retrying forever.
                time.sleep(0.5)
                code, _ = self._exec_run("pgrep -x tor")
                if code != 0:
                    pretty_log(
                        "Sandbox Tor",
                        "In-container Tor did not start (expected under host "
                        "networking, where the host Tor already serves :9050).",
                        level="WARNING", icon=Icons.WARN,
                    )

        # Reached the end of ensure_running without raising → container +
        # mount + environment are all confirmed good. Stamp the readiness TTL
        # so the next command within the window skips the probe entirely.
        self.mark_ready()

        # Only announce readiness when this call actually had to bring the
        # environment up. Silent on the steady-state common path.
        if did_work:
            pretty_log("Sandbox Ready", "Environment Ready.", icon=Icons.OK)

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
            path = spill_dir / name
            capped = text[: 10 * 1024 * 1024]  # 10 MB hard ceiling
            path.write_text(capped, encoding="utf-8", errors="replace")
            return f".ghost_runs/{name}"
        except Exception as e:
            logger.debug(f"run-output spill failed (non-critical): {e}")
            return None

    def execute(self, cmd: str, timeout: int = 600, workdir: str = None,
                spill_large_output: bool = False, max_output_chars: int = None):
        try:
            # ensure_running() either just probed readiness (steady path)
            # or raised (provision path) — re-probing here doubled the
            # exec/host-IO overhead of EVERY command for no benefit. If
            # the container dies in the tiny gap before exec_run below,
            # the normal error path surfaces it.
            self.ensure_running()

            # Add -k 5s to ensure processes are killed if they ignore SIGTERM
            cmd_string = f"timeout -k 5s {timeout}s {cmd}"
            pretty_log("Sandbox Exec", f"Command: {cmd_string}", icon=Icons.TOOL_SHELL)
            
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
            
            # The command self-limits via the in-container `timeout -k 5s Ns`
            # wrapper, so the client deadline only needs to catch a WEDGED
            # daemon (which never streams the process's EOF back): timeout +
            # grace. Without it a stuck daemon hangs this worker thread forever.
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

            return output, exit_code

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
                f"{'docker daemon wedged; ' if _wedged else ''}{str(e)}", 1)

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
        container = self.container
        if container is None:
            # Best-effort: maybe a container with our name exists from a
            # previous run that never bound to `self.container`.
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
        finally:
            if remove:
                self.container = None