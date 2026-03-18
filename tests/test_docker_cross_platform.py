import pytest
import os
import sys
from unittest.mock import MagicMock, patch
from pathlib import Path

# We cannot patch sys.modules at the module level because it leaks to other tests in the suite.
# However, DockerSandbox doesn't import `docker` globally, it imports it dynamically.
from ghost_agent.sandbox.docker import DockerSandbox

def test_docker_execute_windows_compatibility():
    # Setup
    host_workspace = Path("/tmp/workspace")
    sandbox = DockerSandbox(host_workspace)
    sandbox.container = MagicMock()
    sandbox.container.status = "running"
    
    mock_result = MagicMock()
    mock_result.output = (b"stdout", b"stderr")
    mock_result.exit_code = 0
    
    def exec_side_effect(*args, **kwargs):
        if "test -f" in args[0]:
            return (0, b"")
        return mock_result

    with patch.dict("sys.modules", {"docker": MagicMock(), "docker.errors": MagicMock()}):
        sandbox.container.exec_run.side_effect = exec_side_effect
        # Mock os to NOT have getuid/getgid attributes
        # We use a spec that specifically excludes them or use `delattr` on a mock
        
        # Since we can't easily modify the real `os` module for the test process safely, 
        # we should patch `ghost_agent.sandbox.docker.os`
        
        with patch.object(sandbox, "_is_container_ready", return_value=True), patch("ghost_agent.sandbox.docker.os") as mock_os:
            # Configure mock_os to NOT have getuid/getgid
            del mock_os.getuid
            del mock_os.getgid
            
            # Ensure hasattr works as expected on the mock
            # MagicMock by default creates attributes on access. 
            # We need `hasattr(mock_os, 'getuid')` to return False.
            # This requires `spec` or `delattr` (if it was already there).
            
            # Better approach: verify that the code uses the fallback 1000:1000
            
            # Current logic in docker.py traces:
            # user_id = os.getuid()
            # group_id = os.getgid()
            
            # New logic will be:
            # user_id = os.getuid() if hasattr(os, 'getuid') else 1000
            
            # If we remove the attributes from the mock, code accessing them directly would fail (if it didn't check hasattr).
            # But hasattr on a MagicMock usually returns True because it creates them on fly.
            
            # Solution: Use `spec` with selected attributes only, OR explicitly set side_effect for access?
            # Actually, `del mock_os.getuid` ensures `hasattr` returns False on a Mock object if configured right? 
            # No, accessing it creates it.
            
            # Let's use `spec` to restrict available attributes.
            # But `os` has many.
            
            # Alternative: Mock `getattr`? No.
            
            # Let's try `del mock_os.getuid`.
            # If that fails, we can use `mock_os` with `spec=[]`.
            
            # Let's try creating a custom object or class that definitely lacks them.
            class FakeOS:
                pass
                
            with patch("ghost_agent.sandbox.docker.os", new=FakeOS()):
                with patch("sys.platform", "win32"):
                    # Run
                    sandbox.execute("echo hello")
                
                # Verify exec_run called with "user='1000:1000'"
                # args[0] is cmd, kwargs['user'] is the one we want.
                
                call_args = sandbox.container.exec_run.call_args
                assert call_args is not None
                assert "user" in call_args[1]
                assert call_args[1]["user"] == "1000:1000"

def test_docker_execute_linux_compatibility():
    # Setup
    host_workspace = Path("/tmp/workspace")
    sandbox = DockerSandbox(host_workspace)
    sandbox.container = MagicMock()
    sandbox.container.status = "running"
    
    # Mock exec_run to return a named tuple-like object expected by code: (exit_code, output) 
    # Wait, code says: `exec_result = self.container.exec_run(...)`
    # `stdout_bytes, stderr_bytes = exec_result.output`
    # `exit_code = exec_result.exit_code`
    # So exec_run returns an object with .output and .exit_code attrs.
    
    mock_result = MagicMock()
    mock_result.output = (b"stdout", b"stderr")
    mock_result.exit_code = 0
    
    def exec_side_effect(*args, **kwargs):
        # ensure_running calls with "test -f" and no demux -> expects tuple (exit_code, output)
        if "test -f" in args[0]:
            return (0, b"")
        # execute calls with demux=True -> expects object with .output and .exit_code
        return mock_result

    with patch.dict("sys.modules", {"docker": MagicMock(), "docker.errors": MagicMock()}):
        sandbox.container.exec_run.side_effect = exec_side_effect
        
        with patch.object(sandbox, "_is_container_ready", return_value=True), patch("ghost_agent.sandbox.docker.os") as mock_os:
            # Configure mock_os HAVE getuid/getgid
            mock_os.getuid.return_value = 1234
            mock_os.getgid.return_value = 5678
            
            # Manually ensure hasattr returns true (it does for mocks usually)
            
            with patch("sys.platform", "linux"):
                # Run
                sandbox.execute("echo linux")
            
            # Verify
            call_args = sandbox.container.exec_run.call_args
            assert call_args[1]["user"] == "1234:5678"

def test_docker_execute_mac_compatibility():
    # Setup
    host_workspace = Path("/tmp/workspace")
    sandbox = DockerSandbox(host_workspace)
    sandbox.container = MagicMock()
    sandbox.container.status = "running"
    
    mock_result = MagicMock()
    mock_result.output = (b"stdout", b"stderr")
    mock_result.exit_code = 0
    
    def exec_side_effect(*args, **kwargs):
        if "test -f" in args[0]:
            return (0, b"")
        return mock_result

    with patch.dict("sys.modules", {"docker": MagicMock(), "docker.errors": MagicMock()}):
        sandbox.container.exec_run.side_effect = exec_side_effect
        
        with patch.object(sandbox, "_is_container_ready", return_value=True), patch("sys.platform", "darwin"):
            
            # Run
            sandbox.execute("echo mac")
            
            # Verify
            call_args = sandbox.container.exec_run.call_args
            assert "user" not in call_args[1]
