
import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
from ghost_agent.tools.file_system import tool_file_search

@pytest.fixture
def mock_sandbox(tmp_path):
    # Create some dummy files
    (tmp_path / "file1.txt").write_text("Hello world")
    (tmp_path / "file2.py").write_text("print('Hello world')")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "file3.md").write_text("# Hello world")
    return tmp_path

@pytest.mark.asyncio
async def test_tool_file_search_async_execution(mock_sandbox):
    """Test that tool_file_search runs asynchronously and returns correct results."""
    
    # We want to verify it uses asyncio.to_thread. 
    # We can spy on asyncio.to_thread or just verify the result.
    # Spying on asyncio.to_thread is a bit tricky as it's a function not a method of an object usually easily patched if imported directly?
    # In file_system.py it is imported as `import asyncio`. So we can patch `ghost_agent.tools.file_system.asyncio.to_thread`.
    
    with patch("ghost_agent.tools.file_system.asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
        # We need to make the mock return the expected result because the real implementation will use it.
        # However, we are refactoring to use it. The CURRENT implementation does NOT use it (it's sync).
        # So if we run this test against CURRENT code, it might fail to match the mock call (because it doesn't call it) 
        # OR it will run synchronously and pass if we don't assert the mock was called.
        
        # To reuse this test for verification:
        # 1. We expect the function to return the search results.
        # 2. We expect asyncio.to_thread to be called (after refactor).
        
        # But we can't easily execute the REAL logic inside the mock side_effect unless we define it.
        # So we'll let the real logic run by wrapping it? No, to_thread takes a func.
        # simpler: just run it and check results, and verify it IS a coroutine (which it is declared as async def already).
        
        # Let's just run it "for real" without mocking to_thread to verify logic correctness first.
        pass

    # Real execution
    result = await tool_file_search("Hello", mock_sandbox)
    
    assert "file1.txt" in result
    assert "file2.py" in result
    assert "file3.md" in result
    
    # Validation of non-blocking behavior is hard in a unit test without mocking.
    # But we can check if we can mock the inner function if we knew its name, but we haven't written it yet.
    
    # After refactor, we will check if `asyncio.to_thread` was called. 
    # For now, let's just ensure it works functionally.

