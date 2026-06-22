import logging
import os
import threading
import time
from pathlib import Path
from ..utils.logging import Icons, pretty_log

logger = logging.getLogger("GhostAgent")

CONTAINER_NAME = "ghost-agent-sandbox"
CONTAINER_WORKDIR = "/workspace"

class DockerSandbox:
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
        # Serializes ensure_running across threads. execute() is run via
        # asyncio.to_thread, so concurrent tool calls hit ensure_running
        # on different threads; without this they race container
        # creation/provisioning (409 name conflict, double apt/pip/
        # playwright install, racing image commit). docker-py client
        # models are not thread-safe either.
        self._lock = threading.Lock()

        pretty_log("Sandbox Init", f"Mounting {self.host_workspace} -> {CONTAINER_WORKDIR}", icon=Icons.SANDBOX_BOX)

    def get_stats(self):
        if not self.container: return None
        try: return self.container.stats(stream=False)
        except: return None

    def _is_container_ready(self):
        try:
            self.container.reload()
            if self.container.status != "running":
                return False

            # Verify the volume mount is still valid (not a deleted host inode)
            import uuid
            test_file = f".mount_sync_{uuid.uuid4().hex}"
            test_path = self.host_workspace / test_file

            try:
                # Write to host
                test_path.touch(exist_ok=True)

                # We specifically MUST use workdir=CONTAINER_WORKDIR.
                # If the host directory inode was deleted + recreated,
                # running any command with workdir set to the bind mount
                # will immediately return exit code 128 (OCI breakout)
                exec_kwargs = {
                    "workdir": CONTAINER_WORKDIR,
                    "demux": True
                }
                exit_code, _ = self.container.exec_run(f"stat {test_file}", **exec_kwargs)
                if exit_code != 0:
                    return False
            finally:
                if test_path.exists():
                    test_path.unlink()

            # Final liveness probe: a tiny `echo OK` confirms the container
            # is responding to exec_run, not just sitting in `running` state
            # with a hung kernel. Any non-zero / non-"OK" output → fail-fast.
            try:
                code, out = self.container.exec_run("echo OK", workdir=CONTAINER_WORKDIR)
                if code != 0:
                    return False
                if out is not None and isinstance(out, (bytes, bytearray)):
                    if b"OK" not in out:
                        return False
            except Exception:
                return False

            return True
        except:
            return False

    def ensure_running(self):
        # Hold the lock for the WHOLE check+provision. The actual command
        # exec in execute() runs AFTER this returns (lock released), so
        # commands still run in parallel — only the readiness/provision
        # step is serialized, which is exactly what must not race.
        with self._lock:
            return self._ensure_running_impl()

    def _ensure_running_impl(self):
        # Track whether this call did any actual work. Most invocations are
        # no-ops (the container is already up and provisioned) and must stay
        # silent — `execute()` calls `ensure_running` before every shell
        # command, so any unconditional logging here floods the agent log.
        did_work = False
        try:
            if not self.container:
                self.container = self.client.containers.get(self.container_name)
        except self.NotFound:
            pass

        if not (self.container and self._is_container_ready()):
            did_work = True
            pretty_log("Sandbox Provision", "Initializing high-performance environment…", icon=Icons.SANDBOX_BOX)
            try:
                try:
                    old = self.client.containers.get(self.container_name)
                    old.remove(force=True)
                    time.sleep(1) 
                except self.NotFound: pass

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

                # Check for cached environment image for instant boot
                try:
                    self.client.images.get("ghost-agent-base:latest")
                    self.image = "ghost-agent-base:latest"
                    run_kwargs["image"] = self.image
                except self.docker_lib.errors.ImageNotFound:
                    pass

                # Skip the network round-trip when the image is already
                # present locally. Only on `ImageNotFound` do we pay for a
                # `pull`. Any other exception (transient daemon hiccup,
                # auth glitch on a private registry) is logged but
                # tolerated — the subsequent `containers.run` will surface
                # a more actionable error if the image is genuinely
                # unusable.
                try:
                    self.client.images.get(self.image)
                except self.docker_lib.errors.ImageNotFound:
                    pretty_log("Sandbox Image", f"Pulling required Docker image: {self.image}", icon=Icons.TOOL_DOWN)
                    try:
                        self.client.images.pull(self.image)
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
                run_kwargs["cpu_period"] = 100000
                run_kwargs["cpu_quota"] = cpu_quota

                try:
                    self.container = self.client.containers.run(**run_kwargs)
                except self.APIError as run_err:
                    # Another process (sharing this docker daemon and the
                    # workspace-derived container name) won the race
                    # between our remove and run — a 409 "name already in
                    # use". Adopt the existing container instead of dying.
                    msg = str(run_err).lower()
                    if getattr(run_err, "status_code", None) == 409 or "already in use" in msg:
                        self.container = self.client.containers.get(self.container_name)
                    else:
                        raise
                
                for _ in range(10):
                    if self._is_container_ready(): break
                    time.sleep(1)
                
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
        #   v3 (now):    .supercharged.v3 — preinstalls the CPU PyTorch
        #                wheel. Without it, every "train a model" project
        #                hit `ModuleNotFoundError: torch` and ran a ~300 s
        #                `pip install torch` mid-task (observed live: the
        #                PetAI training task), often tripping the execute
        #                timeout. v2 images re-provision to pick torch up.
        marker_path = "/root/.supercharged.v3"

        marker_ok = (self.container.exec_run(f"test -f {marker_path}")[0] == 0)
        chromium_ok = self._chromium_binary_present()
        if not marker_ok or not chromium_ok:
            did_work = True
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
            
            apt_cmd = "sh -c 'apt-get update && apt-get install -y sudo coreutils nodejs npm g++ curl wget git procps postgresql-client libpq-dev tor ripgrep sqlite3'"
            code, out = self.container.exec_run(apt_cmd, environment=env_vars)
            if code != 0:
                err_msg = out.decode("utf-8", errors="replace") if out else "Unknown error"
                raise Exception(f"System package installation failed: {err_msg}")
                
            self.container.exec_run("sh -c 'echo \"ALL ALL=(ALL) NOPASSWD: ALL\" >> /etc/sudoers'")
            
            if self.tor_proxy:
                code, out = self.container.exec_run("pip install --no-cache-dir pysocks requests")
                if code != 0:
                    err_msg = out.decode("utf-8", errors="replace") if out else "Unknown error"
                    raise Exception(f"PySocks bootstrap failed: {err_msg}")
            
            install_cmd = (
                "pip install --no-cache-dir "
                "numpy pandas scipy matplotlib seaborn plotly "
                "scikit-learn yfinance beautifulsoup4 networkx requests "
                "pylint black mypy bandit dill ipykernel jupyter_client "
                "pytest pytest-asyncio "
                "psycopg2-binary asyncpg sqlalchemy tabulate sqlglot playwright html2text lxml"
            )
            code, out = self.container.exec_run(install_cmd, environment=env_vars)
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
            torch_code, torch_out = self.container.exec_run(
                "pip install --no-cache-dir torch "
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
            pw_code, pw_out = self.container.exec_run(
                "python3 -m playwright install chromium --with-deps",
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

            self.container.exec_run(f"touch {marker_path}")
            # Remove any legacy v1 marker so a downgrade-then-upgrade
            # cycle doesn't leave stale state around.
            self.container.exec_run("rm -f /root/.supercharged")

            # Cache the fully installed environment for instant future startups
            if self.image != "ghost-agent-base:latest":
                try:
                    pretty_log("Sandbox Cache", "Committing fast-boot image cache…", icon=Icons.SANDBOX_BOX)
                    self.container.commit(repository="ghost-agent-base", tag="latest")
                except Exception as e:
                    logger.warning(f"Failed to commit sandbox image cache: {e}")

        # Ensure Tor is installed and running inside the container for isolated browser proxying
        if self.tor_proxy:
            exit_code, _ = self.container.exec_run("test -f /usr/bin/tor")
            if exit_code != 0:
                did_work = True
                pretty_log("Sandbox Tor", "Installing isolated Tor daemon…", icon=Icons.TOOL_DOWN)
                self.container.exec_run("sh -c 'apt-get update && apt-get install -y tor'", user="root")

            code, _ = self.container.exec_run("pgrep -x tor")
            if code != 0:
                did_work = True
                self.container.exec_run("su - debian-tor -s /bin/sh -c 'tor --RunAsDaemon 1'", user="root")

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
            code, out = self.container.exec_run(
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
    def execute(self, cmd: str, timeout: int = 600, workdir: str = None):
        try:
            self.ensure_running()
            if not self._is_container_ready():
                pretty_log("Sandbox Not Ready", "container refused to start",
                           icon=Icons.STOP, level="ERROR")
                return "Error: Container refused to start.", 1
 
 
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
            
            exec_result = self.container.exec_run(
                cmd_string,
                **exec_kwargs 
            )
            
            stdout_bytes = exec_result.output
            exit_code = exec_result.exit_code

            # Hard output cap. A sandbox script that prints multi-MB to
            # stdout used to materialise the entire blob into Python memory
            # and then forward it to the model context — capable of OOMing
            # the host or flooding the LLM with 100k+ tokens of garbage.
            # We keep the head AND tail (most error stack traces are at the
            # tail; setup logs at the head) and drop the middle.
            output = ""
            if stdout_bytes:
                MAX_OUTPUT_BYTES = 256 * 1024  # 256 KB
                total = len(stdout_bytes)
                if total > MAX_OUTPUT_BYTES:
                    half = MAX_OUTPUT_BYTES // 2
                    head = stdout_bytes[:half].decode("utf-8", errors="replace")
                    tail = stdout_bytes[-half:].decode("utf-8", errors="replace")
                    dropped = total - MAX_OUTPUT_BYTES
                    output = (
                        f"{head}\n\n"
                        f"[... {dropped} bytes truncated by sandbox 256KB cap — "
                        f"showing first 128KB and last 128KB ...]\n\n"
                        f"{tail}"
                    )
                else:
                    output = stdout_bytes.decode("utf-8", errors="replace")

            if not output.strip() and exit_code != 0:
                 output = f"[SYSTEM ERROR]: Process failed (Exit {exit_code}) with no output."

            return output, exit_code

        except Exception as e:
            pretty_log("Sandbox Exec Failed", f"{type(e).__name__}: {e}",
                       icon=Icons.FAIL, level="ERROR")
            return f"Container Execution Error: {str(e)}", 1

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