
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from ghost_agent.sandbox.docker import DockerSandbox

def test_docker_timeout_kill_flag():
    # Setup
    host_workspace = Path("/tmp/workspace")
    sandbox = DockerSandbox(host_workspace)
    sandbox.container = MagicMock()
    sandbox.container.status = "running"
    
    # Mock exec_run to return valid result structure depending on call (tuple vs object)
    mock_result_obj = MagicMock()
    mock_result_obj.output = (b"", b"")
    mock_result_obj.exit_code = 0
    
    def exec_side_effect(*args, **kwargs):
        # ensure_running calls with "test -f" -> expects tuple (exit_code, output)
        if args and "test -f" in args[0]:
            return (0, b"")
        # execute calls with demux=True -> expects object with .output and .exit_code
        return mock_result_obj

    sandbox.container.exec_run.side_effect = exec_side_effect
    
    with patch.object(sandbox, "_is_container_ready", return_value=True), patch("ghost_agent.sandbox.docker.os") as mock_os:
        # Defaults
        mock_os.getuid.return_value = 1000
        mock_os.getgid.return_value = 1000
        
        sandbox.execute("sleep 10", timeout=30)
        
        # Verify
        # Check arguments passed to exec_run
        # ensure_running might call exec_run too (e.g. test -f). We need to find the one with our command.
        
        found_command = False
        for call in sandbox.container.exec_run.call_args_list:
            args, _ = call
            if args:
                cmd_arg = args[0]
                if "sleep 10" in cmd_arg:
                    assert "timeout -k 5s" in cmd_arg
                    assert "30s" in cmd_arg
                    found_command = True
                    break
        
        assert found_command, f"Command 'sleep 10' was not executed via exec_run. Calls: {sandbox.container.exec_run.call_args_list}"
