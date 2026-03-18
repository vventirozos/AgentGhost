import pytest
import sys
import os
from unittest.mock import MagicMock, patch, call
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from ghost_agent.sandbox.docker import DockerSandbox

def test_docker_ensure_running_installs_sudo():
    # Setup
    host_workspace = Path("/tmp/workspace")
    
    mock_client = MagicMock()
    mock_container = MagicMock()
    mock_container.status = "running"
    
    def exec_side_effect(cmd, **kwargs):
        if "test -f /root/.supercharged" in cmd:
            return (1, b"") # Return non-zero to trigger installation
        return (0, b"")
        
    mock_container.exec_run.side_effect = exec_side_effect
    
    # Execute within patch context to prevent global poisoning of imported docker
    with patch.dict('sys.modules', {'docker': MagicMock(), 'docker.errors': MagicMock()}):
        import sys
        
        sys.modules["docker"].from_env = MagicMock(return_value=mock_client)
        mock_client.containers.run.return_value = mock_container
        class MockNotFound(Exception): pass
        sys.modules["docker.errors"].NotFound = MockNotFound
        mock_client.containers.get.side_effect = MockNotFound()

        # Instantiate within patched scope
        sandbox = DockerSandbox(host_workspace)
        
        # We also need to mock _is_container_ready because ensure_running calls it first 
        # and we don't want it to consume our "stat" side effect logic here
        with patch.object(sandbox, "_is_container_ready", return_value=True):
            sandbox.container = mock_container
            sandbox.ensure_running()
    
    # Verify
    exec_calls = mock_container.exec_run.call_args_list
    
    installed_sudo = False
    configured_sudoers = False
    
    for call_obj in exec_calls:
        cmd = call_obj[0][0]
        if "apt-get install -y sudo" in cmd:
            installed_sudo = True
        if "ALL ALL=(ALL) NOPASSWD: ALL" in cmd and "/etc/sudoers" in cmd:
            configured_sudoers = True
            
    assert installed_sudo, "sudo package was not installed during sandbox initialization"
    assert configured_sudoers, "passwordless sudo was not configured during sandbox initialization"
