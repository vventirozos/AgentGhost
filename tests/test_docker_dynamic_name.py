import pytest
import os
import hashlib
from unittest.mock import MagicMock, patch
from pathlib import Path
from src.ghost_agent.sandbox.docker import DockerSandbox

@patch('docker.from_env')
def test_docker_dynamic_container_name(mock_from_env, tmp_path):
    """
    Test that DockerSandbox uses a dynamic container name based on the workspace path 
    to prevent volume mount conflicts across different agent sessions.
    """
    mock_client = MagicMock()
    mock_from_env.return_value = mock_client
    
    workspace1 = tmp_path / "workspace1"
    workspace2 = tmp_path / "workspace2"
    
    sandbox1 = DockerSandbox(workspace1)
    sandbox2 = DockerSandbox(workspace2)
    
    # Verify container names are different
    assert sandbox1.container_name != sandbox2.container_name
    
    # Verify the hash logic matches what we expect
    expected_hash1 = hashlib.md5(str(workspace1.absolute()).encode()).hexdigest()[:8]
    expected_hash2 = hashlib.md5(str(workspace2.absolute()).encode()).hexdigest()[:8]
    
    assert sandbox1.container_name == f"ghost-agent-sandbox-{expected_hash1}"
    assert sandbox2.container_name == f"ghost-agent-sandbox-{expected_hash2}"

    # Verify ensure_running attempts to get the right container name
    mock_client.containers.get.side_effect = Exception("Not found") # mock NotFound
    
    # Set exception class
    import docker
    try:
        sandbox1.NotFound = docker.errors.NotFound
    except AttributeError:
        # Create dummy exception if docker library isn't actually installed in test environment
        class NotFoundException(Exception): pass
        sandbox1.NotFound = NotFoundException
        mock_client.containers.get.side_effect = sandbox1.NotFound()

    try:
        sandbox1.ensure_running()
    except Exception:
        pass # it will fail creating since it's a mock, but we just want to verify '.get' was called
        
    mock_client.containers.get.assert_any_call(sandbox1.container_name)
    
