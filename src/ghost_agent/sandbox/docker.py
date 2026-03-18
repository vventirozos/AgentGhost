import logging
import os
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

        pretty_log("Sandbox Init", f"Mounting {self.host_workspace} -> {CONTAINER_WORKDIR}", icon=Icons.SYSTEM_BOOT)

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
                    
            return True
        except:
            return False

    def ensure_running(self):
        try:
            if not self.container:
                self.container = self.client.containers.get(self.container_name)
        except self.NotFound:
            pass 

        if not (self.container and self._is_container_ready()):
            pretty_log("Sandbox", "Initializing High-Performance Environment...", icon="⚙️")
            try:
                try:
                    old = self.client.containers.get(self.container_name)
                    old.remove(force=True)
                    time.sleep(1) 
                except self.NotFound: pass

                import sys
                is_linux = sys.platform.startswith("linux")
                is_mac = sys.platform == "darwin"
                
                run_kwargs = {
                    "image": self.image,
                    "command": "sleep infinity",
                    "name": self.container_name,
                    "detach": True,
                    "tty": True,
                    "volumes": {str(self.host_workspace): {'bind': CONTAINER_WORKDIR, 'mode': 'rw'}},
                    "mem_limit": "512m",
                }
                
                if is_linux:
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

                try:
                    self.client.images.get(self.image)
                except self.docker_lib.errors.ImageNotFound:
                    pretty_log("Sandbox", f"Pulling required Docker image: {self.image}", icon="📥")
                    self.client.images.pull(self.image)
                    
                self.container = self.client.containers.run(**run_kwargs)
                
                for _ in range(10):
                    if self._is_container_ready(): break
                    time.sleep(1)
                
            except Exception as e:
                pretty_log("Sandbox Error", f"Failed to start: {e}", level="ERROR")
                raise e

        env_vars = {}
        # We don't set HTTP_PROXY for the sandbox because we don't want to route
        # heavy package installs through Tor to avoid timeouts and IP blocks.

        exit_code, _ = self.container.exec_run("test -f /root/.supercharged")
        if exit_code != 0:
            pretty_log("Sandbox", "Installing Deep Learning Stack (Wait ~60s)...", icon="📦")
            
            apt_cmd = "sh -c 'apt-get update && apt-get install -y sudo coreutils nodejs npm g++ curl wget git procps postgresql-client libpq-dev'"
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
                "numpy pandas scipy matplotlib seaborn plotly"
                "scikit-learn yfinance beautifulsoup4 networkx requests "
                "pylint black mypy bandit dill "
                "psycopg2-binary asyncpg sqlalchemy tabulate sqlglot"
            )
            code, out = self.container.exec_run(install_cmd, environment=env_vars)
            if code != 0:
                err_msg = out.decode("utf-8", errors="replace") if out else "Unknown error"
                raise Exception(f"Python package installation failed: {err_msg}")

                
            self.container.exec_run("touch /root/.supercharged")

            # Cache the fully installed environment for instant future startups
            if self.image != "ghost-agent-base:latest":
                try:
                    pretty_log("Sandbox", "Committing fast-boot image cache...", icon=Icons.MEM_SAVE)
                    self.container.commit(repository="ghost-agent-base", tag="latest")
                except Exception as e:
                    logger.warning(f"Failed to commit sandbox image cache: {e}")

            pretty_log("Sandbox", "Environment Ready.", icon="✅")

    def execute(self, cmd: str, timeout: int = 300):
        try:
            self.ensure_running()
            if not self._is_container_ready():
                return "Error: Container refused to start.", 1
 
 
            # Add -k 5s to ensure processes are killed if they ignore SIGTERM
            cmd_string = f"timeout -k 5s {timeout}s {cmd}"
            pretty_log("Docker Execute Debug", f"Command: {cmd_string}", icon=Icons.TOOL_CODE)
            
            # Cross-platform safe UID/GID fetching (Windows doesn't have getuid)
            user_id = os.getuid() if hasattr(os, 'getuid') else 1000
            group_id = os.getgid() if hasattr(os, 'getgid') else 1000
            
            import sys
            is_mac = sys.platform == "darwin"
            
            exec_kwargs = {
                "workdir": CONTAINER_WORKDIR,
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

            output = ""
            if stdout_bytes: output = stdout_bytes.decode("utf-8", errors="replace")

            if not output.strip() and exit_code != 0:
                 output = f"[SYSTEM ERROR]: Process failed (Exit {exit_code}) with no output."

            return output, exit_code

        except Exception as e:
            return f"Container Execution Error: {str(e)}", 1