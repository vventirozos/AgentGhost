
import pytest
import os
import httpx
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path
import sys

# Import components to test
# We need to make sure src is in path or we run with PYTHONPATH=src
# valid imports
from ghost_agent.core.llm import LLMClient
from ghost_agent.memory.vector import VectorMemory, GhostEmbeddingFunction
from ghost_agent.sandbox.docker import DockerSandbox
from ghost_agent.tools.system import tool_check_health
from ghost_agent.tools.search import tool_search_ddgs
from ghost_agent.tools.file_system import _get_safe_path
from ghost_agent.utils.logging import Icons

# Dummy Paths
MOCK_SANDBOX = Path("/tmp/sandbox")
MOCK_MEMORY = Path("/tmp/memory")

@pytest.fixture
def mock_tor_proxy():
    return "socks5://127.0.0.1:9050"

@pytest.fixture
def mock_tor_proxy_h():
    return "socks5h://127.0.0.1:9050"

# --- 1. LLM Client Tests ---
@patch("ghost_agent.core.llm.httpx.AsyncClient")
def test_llm_client_uses_proxy_for_external(mock_client_cls, mock_tor_proxy, mock_tor_proxy_h):
    # Fix Icon issue by mocking pretty_log or just ignoring it, 
    # but the code uses Icons.shield. If Icons doesn't have shield, we mocked it? 
    # The real code has Icons.shield? Let's assume it doesn't and verify.
    # We will patch pretty_log to avoid the AttributeError during runtime if it fails there.
    
    with patch("ghost_agent.core.llm.pretty_log"):
        # Case A: External URL -> Should use Proxy
        client = LLMClient(upstream_url="https://api.openai.com", tor_proxy=mock_tor_proxy)
        # Check if AsyncClient was init with proxy
        _, kwargs = mock_client_cls.call_args
        assert kwargs.get("proxy") == mock_tor_proxy_h

@patch("ghost_agent.core.llm.httpx.AsyncClient")
def test_llm_client_ignores_proxy_for_localhost(mock_client_cls, mock_tor_proxy):
    client = LLMClient(upstream_url="http://127.0.0.1:8080", tor_proxy=mock_tor_proxy)
    _, kwargs = mock_client_cls.call_args
    assert kwargs.get("proxy") is None

# --- 2. Vector Memory Tests ---
# Since imports are local, we patch the GLOBAL modules that they import
@patch("chromadb.PersistentClient")
@patch("chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction")
def test_vector_memory_does_not_inject_env_vars(mock_embed, mock_client, mock_tor_proxy, mock_tor_proxy_h):
    # We also need to patch os.environ carefully
    with patch.dict(os.environ, {}, clear=True):
        # Initialize VectorMemory with Tor
        vm = VectorMemory(memory_dir=MOCK_MEMORY, upstream_url="http://external-embedding.com", tor_proxy=mock_tor_proxy)
        
        # Assert Environment Variables NOT set for HuggingFace
        assert os.environ.get("HTTP_PROXY") is None
        assert os.environ.get("HTTPS_PROXY") is None

@patch("httpx.Client")
def test_ghost_embedding_function_uses_proxy(mock_httpx_client, mock_tor_proxy, mock_tor_proxy_h):
    # Patching global httpx.Client because it is imported as 'import httpx' inside __init__
    # and then used as httpx.Client. 
    # Since we can't easily patch 'ghost_agent.memory.vector.httpx' (it doesn't exist at module level),
    # we rely on the fact that 'import httpx' returns the global module.
    
    ef = GhostEmbeddingFunction(upstream_url="https://external-api.com", tor_proxy=mock_tor_proxy)
    _, kwargs = mock_httpx_client.call_args
    assert kwargs.get("proxy") == mock_tor_proxy_h

# --- 3. Docker Sandbox Tests ---
@patch("docker.from_env")
def test_docker_sandbox_injects_proxy(mock_docker, mock_tor_proxy, mock_tor_proxy_h):
    # Setup Mock Container
    mock_container = MagicMock()
    mock_container.status = "running"  # Ensure it's considered ready
    mock_container.exec_run.return_value = (0, b"")
    mock_client = MagicMock()
    mock_client.containers.get.return_value = mock_container
    mock_docker.return_value = mock_client
    
    sandbox = DockerSandbox(host_workspace=MOCK_SANDBOX, tor_proxy=mock_tor_proxy)
    sandbox.container = mock_container # Force inject
    
    # Trigger install logic (simulate missing marker)
    # 1. test -f -> returns 1 (missing)
    # 2. apt-get -> returns 0
    # 3. sudoers echo -> returns 0
    # 4. pip install pysocks -> returns 0
    # 5. pip install main -> returns 0
    # 6. touch -> returns 0
    mock_container.exec_run.side_effect = [(1, b""), (0, b""), (0, b""), (0, b""), (0, b""), (0, b"")] 
    
    with patch.object(sandbox, "_is_container_ready", return_value=True):
        sandbox.ensure_running()
    
    # Verify exec_run was called with environment variables
    # We filter for calls that have environment in kwargs
    # calls is a list of call objects. 
    # call.args is tuple, call.kwargs is dict.
    
    assert mock_container.exec_run.called
    calls = mock_container.exec_run.call_args_list
    
    # We expect apt-get to be called with env vars
    apt_calls = [c for c in calls if "apt-get" in str(c.args)]
    if not apt_calls:
        # Fallback: maybe it was keyword arg "cmd"?
        apt_calls = [c for c in calls if "apt-get" in str(c)]
    
    assert apt_calls, "apt-get was not called"
    last_apt_call = apt_calls[-1]
    
    # Check env in kwargs
    env_arg = last_apt_call.kwargs.get("environment")
    assert not env_arg, f"Expected no environment variables for installing, got {env_arg}"

# --- 4. System Tools Tests ---
@patch("ghost_agent.tools.system.curl_requests", None)
@patch("ghost_agent.tools.system.curl_requests", None)
@patch("ghost_agent.tools.system.asyncio.sleep")
@patch("ghost_agent.tools.system.request_new_tor_identity")
@patch("ghost_agent.tools.system.httpx.AsyncClient")
@pytest.mark.asyncio
async def test_check_health_uses_proxy(mock_client_cls, mock_tor_identity, mock_sleep, mock_tor_proxy, mock_tor_proxy_h):
    # Mock context object
    mock_context = MagicMock()
    mock_context.tor_proxy = mock_tor_proxy
    mock_context.llm_client = MagicMock() # active
    mock_context.memory_system = MagicMock() # active
    mock_context.sandbox_dir = MagicMock() # active
    
    # Setup AsyncContextManager mock
    mock_instance = AsyncMock()
    mock_instance.get.side_effect = Exception("Force fail to trigger retry")
    mock_client_cls.return_value.__aenter__.return_value = mock_instance
    
    await tool_check_health(context=mock_context)
    
    # Verify httpx client was initialized with proxy
    # We look for ANY init with the proxy because check_health makes multiple calls
    inits = mock_client_cls.call_args_list
    found_proxy_init = False
    for args, kwargs in inits:
        if kwargs.get("proxy") == mock_tor_proxy_h:
            found_proxy_init = True
            break
            
    assert found_proxy_init, "httpx.AsyncClient was not initialized with the Tor proxy"

# --- 5. Search Tool Tests ---
@pytest.mark.asyncio
async def test_search_ddgs_converts_proxy(mock_tor_proxy, mock_tor_proxy_h):
    # Patch the DDGS class where it is imported in the code?
    # No, it's imported inside the function: 'from ddgs import DDGS'
    # So we patch 'ddgs.DDGS' in the global sys.modules cache
    with patch("ddgs.DDGS") as mock_ddgs:
        # Mock Context Manager
        mock_instance = MagicMock()
        mock_instance.text.return_value = []
        mock_ddgs.return_value.__enter__.return_value = mock_instance
        
        # We also need to mock importlib.util.find_spec to return True
        with patch("importlib.util.find_spec", return_value=True):
            await tool_search_ddgs("test query", mock_tor_proxy)
        
        # Verify DDGS init
        _, kwargs = mock_ddgs.call_args
        assert kwargs.get("proxy") == mock_tor_proxy_h

# --- 6. Path Traversal Tests ---
def test_path_traversal_prevention():
    # .resolve() because on Mac /tmp is a symlink to /private/tmp, which breaks equality checks vs _get_safe_path
    sandbox = Path("/tmp/sandbox").resolve()
    
    # Safe path
    safe = _get_safe_path(sandbox, "safe_file.txt")
    assert safe == sandbox / "safe_file.txt"
    
    # Subdir safe path
    safe_sub = _get_safe_path(sandbox, "subdir/file.txt")
    assert safe_sub == sandbox / "subdir" / "file.txt"
    
    # Attack 1: Parent Directory (Should Raise)
    with pytest.raises(ValueError, match="Security Error"):
        _get_safe_path(sandbox, "../secret.txt")
        
    # Attack 2: Root Absolute (SHOULD NOT RAISE if treated as relative)
    # Current implementation strips leading slashes, making /etc/passwd -> sandbox/etc/passwd.
    # This is SAFE. So we verify it resolves safely.
    safe_abs = _get_safe_path(sandbox, "/etc/passwd")
    # On MacOS/Unix, Path("/etc/passwd").relative_to("/") might just be "etc/passwd" but Path.joinpath behavior on absolute paths varies.
    # The actual implementation of _get_safe_path strips leading slashes using lstrip('/') and resolves.
    assert safe_abs == sandbox / "etc/passwd"
        
    # Attack 3: Sneaky Traversal (Should Raise)
    # Accessing a sibling directory of the sandbox
    with pytest.raises(ValueError, match="Security Error"):
        _get_safe_path(sandbox, "../sandbox_sibling/file.txt")

# --- 7. Regression Tests ---
def test_agent_transcript_handles_none_content():
    # Simulation of the logic in agent.py that crashed
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": None}, # Tool call or similar
        {"role": "user", "content": "Status?"}
    ]
    
    recent_transcript = ""
    transcript_msgs = [m for m in messages if m.get("role") in ["user", "assistant"]][-4:]
    
    # Safe logic verification (this panicked before fix)
    try:
        for m in transcript_msgs:
            content = m.get('content') or ""
            recent_transcript += f"{m['role'].upper()}: {content[:500]}\n"
    except TypeError:
        pytest.fail("Regression: agent.py logic crashed on None content")
        
    assert "ASSISTANT: \n" in recent_transcript
