import pytest
import sys
import os
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from ghost_agent.sandbox.docker import DockerSandbox

# We remove the global sys.modules overwrite to avoid poisoning subsequent tests.

def test_docker_init_installs_packages(tmp_path):
    # Setup mock docker environment
    mock_client = MagicMock()
    mock_container = MagicMock()
    
    class MockNotFound(Exception): pass
    
    with patch.dict('sys.modules', {
        'docker': MagicMock(from_env=MagicMock(return_value=mock_client)),
        'docker.errors': MagicMock(NotFound=MockNotFound)
    }):
        mock_client.containers.get.side_effect = MockNotFound()

        mock_client.containers.run.return_value = mock_container
        mock_container.status = "running"
        
        def exec_run_side_effect(cmd, **kwargs):
            if hasattr(cmd, '__contains__'):
                if "test -f" in cmd:
                    return (1, b"")
                elif "apt-get" in cmd:
                    return (0, b"")
                elif "/etc/sudoers" in cmd:
                    return (0, b"")
                elif "pip install" in cmd:
                    return (0, b"")
                elif "touch" in cmd:
                    return (0, b"")
            return (0, b"")

        mock_container.exec_run.side_effect = exec_run_side_effect

        sandbox = DockerSandbox(host_workspace=tmp_path)
        sandbox._is_container_ready = MagicMock(return_value=True)
        sandbox.ensure_running()

        # Verify that the correct shell command was used for apt-get
        exec_calls = mock_container.exec_run.call_args_list
        
        apt_call_found = False
        for call in exec_calls:
            cmd_arg = call[0][0]
            if hasattr(cmd_arg, '__contains__') and "apt-get update" in cmd_arg:
                apt_call_found = True
                assert cmd_arg.startswith("sh -c 'apt-get update && apt-get install -y")
        
        assert apt_call_found, "The apt-get update && install command was not found in exec_run calls"


def test_docker_init_raises_on_apt_fail(tmp_path):
    mock_client = MagicMock()
    mock_container = MagicMock()
    
    class MockNotFound(Exception): pass

    with patch.dict('sys.modules', {
        'docker': MagicMock(from_env=MagicMock(return_value=mock_client)),
        'docker.errors': MagicMock(NotFound=MockNotFound)
    }):
        mock_client.containers.get.side_effect = MockNotFound()

        mock_client.containers.run.return_value = mock_container
        mock_container.status = "running"
        
        # Mock apt-get failure
        def exec_run_side_effect(cmd, **kwargs):
            if hasattr(cmd, '__contains__'):
                if "test -f" in cmd:
                    return (1, b"")
                elif "apt-get" in cmd:
                    return (1, b"E: Invalid operation update") # Fail
            return (0, b"")

        mock_container.exec_run.side_effect = exec_run_side_effect

        sandbox = DockerSandbox(host_workspace=tmp_path)
        sandbox._is_container_ready = MagicMock(return_value=True)
        
        with pytest.raises(Exception, match="System package installation failed: E: Invalid operation update"):
            sandbox.ensure_running()


def test_docker_init_raises_on_pip_fail(tmp_path):
    mock_client = MagicMock()
    mock_container = MagicMock()
    
    class MockNotFound(Exception): pass
    
    with patch.dict('sys.modules', {
        'docker': MagicMock(from_env=MagicMock(return_value=mock_client)),
        'docker.errors': MagicMock(NotFound=MockNotFound)
    }):
        mock_client.containers.get.side_effect = MockNotFound()

        mock_client.containers.run.return_value = mock_container
        mock_container.status = "running"
        
        # Mock pip install failure
        def exec_run_side_effect(cmd, **kwargs):
            if hasattr(cmd, '__contains__'):
                if "test -f" in cmd:
                    return (1, b"")
                elif "apt-get" in cmd:
                    return (0, b"")
                elif "/etc/sudoers" in cmd:
                    return (0, b"")
                elif "pip install" in cmd:
                    return (1, b"ERROR: Could not find a version...") # Fail
            return (0, b"")

        mock_container.exec_run.side_effect = exec_run_side_effect

        sandbox = DockerSandbox(host_workspace=tmp_path)
        sandbox._is_container_ready = MagicMock(return_value=True)
        
        with pytest.raises(Exception, match="Python package installation failed: ERROR: Could not find a version..."):
            sandbox.ensure_running()

def test_docker_init_tor_proxy_installs_pysocks(tmp_path):
    mock_client = MagicMock()
    mock_container = MagicMock()
    
    class MockNotFound(Exception): pass

    with patch.dict('sys.modules', {
        'docker': MagicMock(from_env=MagicMock(return_value=mock_client)),
        'docker.errors': MagicMock(NotFound=MockNotFound)
    }):
        mock_client.containers.get.side_effect = MockNotFound()

        mock_client.containers.run.return_value = mock_container
        mock_container.status = "running"
        
        def exec_run_side_effect(cmd, **kwargs):
            if hasattr(cmd, '__contains__'):
                if "test -f" in cmd:
                    return (1, b"")
                elif "apt-get" in cmd:
                    return (0, b"")
                elif "/etc/sudoers" in cmd:
                    return (0, b"")
                elif "pip install" in cmd:
                    return (0, b"")
                elif "touch" in cmd:
                    return (0, b"")
            return (0, b"")

        mock_container.exec_run.side_effect = exec_run_side_effect

        sandbox = DockerSandbox(host_workspace=tmp_path, tor_proxy="socks5://127.0.0.1:9050")
        sandbox._is_container_ready = MagicMock(return_value=True)
        sandbox.ensure_running()

        # Verify that containers.run was called
        run_call_kwargs = mock_client.containers.run.call_args[1]
        assert run_call_kwargs is not None

        # Verify that the correct shell commands were used
        exec_calls = mock_container.exec_run.call_args_list
        
        pysocks_bootstrap_found = False
        pip_install_found = False
        for call in exec_calls:
            cmd_arg = call[0][0]
            if hasattr(cmd_arg, '__contains__'):
                if "pip install --no-cache-dir pysocks requests" in cmd_arg:
                    pysocks_bootstrap_found = True
                    # verify it runs without environment variable proxies
                    assert not call[1].get('environment'), "Bootstrap should run without tor_proxy environment vars"
                elif "pip install --no-cache-dir numpy" in cmd_arg:
                    pip_install_found = True
                    assert not call[1].get('environment'), "Main install should run without tor_proxy environment vars"
        
        assert pysocks_bootstrap_found, "The PySocks bootstrap command was not found"
        assert pip_install_found, "The main pip install command was not found"
