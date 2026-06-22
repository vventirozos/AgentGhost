import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from ghost_agent.sandbox.docker import DockerSandbox

@pytest.mark.asyncio
async def test_docker_sandbox_installs_dill_and_others():
    """
    Test that the DockerSandbox install_cmd includes the required packages,
    specifically ensuring `dill` was recently added.
    """
    # 1. Setup mock sandbox
    host_workspace = Path("/tmp/workspace")
    sandbox = DockerSandbox(host_workspace)
    sandbox.container = MagicMock()
    sandbox.container.status = "running"
    
    # 2. Mock exec code
    def exec_side_effect(cmd, **kwargs):
        if "test -f" in cmd:
            # Simulate first boot where it's not installed yet
            return (1, b"")
        if "from playwright" in cmd:
            # Simulate Playwright not yet installed so install runs
            return (1, b"ModuleNotFoundError")
        if "apt-get" in cmd or "sudoers" in cmd or "PySocks" in cmd or "pip install" in cmd:
            # Simulate successful install
            return (0, b"Success")
        if "touch" in cmd:
            return (0, b"")
        if "find /root/.cache/ms-playwright" in cmd:
            return (0, b"/root/.cache/ms-playwright/chromium-9999/chrome-linux/headless_shell\n")
        return (0, b"")
        
    sandbox.container.exec_run.side_effect = exec_side_effect
    
    # 3. Trigger the install by ensuring it runs
    with patch.dict("sys.modules", {"docker": MagicMock(), "docker.errors": MagicMock()}):
        with patch.object(sandbox, "_is_container_ready", return_value=True):
            sandbox.ensure_running()
            
            # 4. Grab all calls to exec_run
            calls = sandbox.container.exec_run.call_args_list
            
            # 5. Find the pip install call
            pip_install_call = None
            playwright_install_call = None
            torch_install_call = None
            for call in calls:
                args, _ = call
                cmd = args[0]
                if "pip install --no-cache-dir" in cmd:
                    if "torch" in cmd:
                        torch_install_call = cmd            # the CPU-torch preinstall
                    elif "pysocks" not in cmd:              # ignore the pysocks bootstrap
                        pip_install_call = cmd              # the main data/ML stack
                elif "playwright install chromium" in cmd:
                    playwright_install_call = cmd

            assert pip_install_call is not None, "Python package installation command was never called"
            assert playwright_install_call is not None, "Playwright browser installation command was never called"
            # torch is preinstalled (CPU wheel) so projects don't pip-install it
            # mid-task and trip the execute timeout.
            assert torch_install_call is not None, "CPU torch preinstall was never called"
            assert "download.pytorch.org/whl/cpu" in torch_install_call, (
                "torch must install from the CPU index, not the default GPU wheels")
            
            # 6. Verify required packages are in the install string
            packages = [
                "numpy", "pandas", "scipy", "matplotlib", "seaborn",
                "scikit-learn", "yfinance", "beautifulsoup4", "networkx", "requests",
                "pylint", "black", "mypy", "bandit", "dill", "ipykernel", "jupyter_client",
                "pytest", "pytest-asyncio",
                "playwright", "html2text", "lxml",
                "psycopg2-binary", "asyncpg", "sqlalchemy", "tabulate", "sqlglot"
            ]
            
            for pkg in packages:
                assert pkg in pip_install_call, f"Package '{pkg}' is missing from Sandbox installation"
