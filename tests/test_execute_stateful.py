import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from ghost_agent.tools.execute import tool_execute

@pytest.mark.asyncio
async def test_tool_execute_stateful_wrapper():
    """
    Test that the tool_execute properly injects the dill wrapper
    when stateful=True and the file is a python script.
    """
    sandbox_dir = Path("/tmp/workspace")
    sandbox_manager = MagicMock()
    
    # We mock out the actual execution and file operations
    with patch("ghost_agent.tools.execute._get_safe_path") as mock_safe_path, \
         patch("ghost_agent.tools.execute.asyncio.to_thread") as mock_to_thread:
        
        mock_path_original = MagicMock()
        mock_path_original.stat.return_value.st_size = 0
        mock_path_wrapper = MagicMock()
        
        def safe_path_side_effect(sandbox_dir, filename):
            if str(filename).startswith("._"):
                return mock_path_wrapper
            return mock_path_original
            
        mock_safe_path.side_effect = safe_path_side_effect
        
        # When to_thread is called, we just return (b"output", 0) for the exec call
        async def mock_to_thread_impl(func, *args, **kwargs):
            if func == mock_path_original.write_text:
                mock_path_original.written_content = args[0]
                return None
            elif func == mock_path_wrapper.write_text:
                mock_path_wrapper.written_content = args[0]
                return None
            elif func == mock_path_original.parent.mkdir or func == mock_path_wrapper.parent.mkdir:
                return None
            elif func == mock_path_wrapper.unlink:
                return None
            elif func == sandbox_manager.execute:
                return ("output", 0)
            return func(*args, **kwargs)
            
        mock_to_thread.side_effect = mock_to_thread_impl
        
        original_code = "x = 10\nprint(x)"
        filename = "test_script.py"
        
        # Call tool_execute with stateful=True
        result = await tool_execute(
            filename=filename,
            content=original_code,
            sandbox_dir=sandbox_dir,
            sandbox_manager=sandbox_manager,
            stateful=True
        )
        
        # Verify the original file only gets the original code
        original_written = mock_path_original.written_content
        assert original_written == original_code
        
        # Verify the wrapper file gets the dill load/dump wrapper
        wrapper_written = mock_path_wrapper.written_content
        assert "import dill" in wrapper_written
        assert "dill.load_session" in wrapper_written
        assert "dill.dump_session" in wrapper_written
        assert original_code in wrapper_written
        assert "# --- AGENT CODE START ---" in wrapper_written
        assert "# --- AGENT CODE END ---" in wrapper_written
