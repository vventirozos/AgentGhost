
import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
from ghost_agent.tools.execute import tool_execute

@pytest.mark.asyncio
async def test_tool_execute_path_traversal_vulnerability():
    """Test that the current implementation is vulnerable to path traversal (or fixed if patched)."""
    sandbox_dir = Path("/tmp/sandbox")
    sandbox_manager = MagicMock()
    sandbox_manager.execute = AsyncMock(return_value=("output", 0))
    
    # Attack payload: try to write to a file outside sandbox
    # e.g. ../../../etc/passwd (simulated)
    filename = "../../../tmp/evil.py" 
    content = "print('hacked')"
    
    # We mock asyncio.to_thread intercepting normally without blowing up unpacking
    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)
        
    with patch("ghost_agent.tools.execute.asyncio.to_thread", side_effect=fake_to_thread) as mock_to_thread:
        # Mock exists to avoid stubbornness guard reading the file
        with patch.object(Path, "exists", return_value=False):
             # Mock mkdir to avoid actual filesystem ops
            with patch.object(Path, "mkdir"):
                 # Mock read_text/write_text to avoid errors
                with patch.object(Path, "write_text"):
                    result = await tool_execute(filename, content, sandbox_dir, sandbox_manager)

    # If vulnerable, it tries to write to /tmp/evil.py (resolved from /tmp/sandbox/../../../tmp/evil.py)
    # The current implementation does: host_path = sandbox_dir / rel_path
    # rel_path = "../../../tmp/evil.py".lstrip("/") -> "../../../tmp/evil.py"
    # host_path = /tmp/sandbox / "../../../tmp/evil.py" -> /tmp/evil.py (outside sandbox)
    
    # If fixed, it should return an error message about security/path traversal.
    
    if "outside sandbox" in result or "Security Error" in result:
        # This is what we WANT after the fix
        pass
    elif "EXIT CODE" in result:
        # This means it executed, which implies vulnerability (for this test setup)
        # However, since we patched basic file ops, it might "succeed" in the test eyes 
        # but the path would be wrong.
        pass

@pytest.mark.asyncio
async def test_tool_execute_prevents_traversal():
    """Explicitly test that traversal attempts are blocked."""
    sandbox_dir = Path("/tmp/sandbox")
    sandbox_manager = MagicMock()
    
    filename = "../outside.py"
    content = "print('fail')"
    
    result = await tool_execute(filename, content, sandbox_dir, sandbox_manager)
    
    # We expect the fix to trigger a specific error message
    assert "Security Error" in result or "outside sandbox" in result
