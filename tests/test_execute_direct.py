import pytest
import asyncio
from unittest.mock import MagicMock, patch
from pathlib import Path
from ghost_agent.tools.execute import tool_execute

@pytest.mark.asyncio
async def test_execute_direct_python():
    # Setup
    mock_sandbox = MagicMock()
    # Return output indicating a successful print and exit code 0
    mock_sandbox.execute = MagicMock(return_value=("--- EXECUTION RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\nHello World\n", 0))
    
    mock_sandbox_dir = MagicMock()
    
    with patch("asyncio.to_thread") as mock_to_thread, \
         patch("ghost_agent.tools.execute._get_safe_path") as mock_safe_path:
        
        # Mock the path so it pretends the file doesn't exist yet (to avoid strict equality check error)
        mock_target_script = MagicMock()
        mock_target_script.exists.return_value = False
        mock_safe_path.return_value = mock_target_script
        
        # Let's cleanly mock asyncio.to_thread so it returns coroutines that evaluate to whatever the function returns
        async def mock_thread_impl(func, *args, **kwargs):
            if func == mock_sandbox.execute:
                return func(*args, **kwargs)
            return func(*args, **kwargs) if callable(func) else None
        
        mock_to_thread.side_effect = mock_thread_impl
        
        result = await tool_execute("calc_salary.py", "print('Hello World')", mock_sandbox_dir, mock_sandbox)
        
        # Verify the command ran directly with python3 -u
        mock_sandbox.execute.assert_any_call("python3 -u calc_salary.py")
        assert "Hello World" in result

@pytest.mark.asyncio
async def test_execute_direct_shell():
    mock_sandbox = MagicMock()
    mock_sandbox.execute = MagicMock(return_value=("done", 0))
    mock_sandbox_dir = MagicMock()
    
    with patch("asyncio.to_thread") as mock_to_thread, \
         patch("ghost_agent.tools.execute._get_safe_path") as mock_safe_path:
        
        mock_target_script = MagicMock()
        mock_target_script.exists.return_value = False
        mock_safe_path.return_value = mock_target_script
        
        async def mock_thread_impl(func, *args, **kwargs):
            if func == mock_sandbox.execute:
                return func(*args, **kwargs)
            return func(*args, **kwargs) if callable(func) else None
            
        mock_to_thread.side_effect = mock_thread_impl
        
        result = await tool_execute("script.sh", "echo 'hi'", mock_sandbox_dir, mock_sandbox)
        
        # Verify the command ran directly with bash
        mock_sandbox.execute.assert_called_once_with("bash script.sh")
