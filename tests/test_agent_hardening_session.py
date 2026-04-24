import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import inspect

from ghost_agent.tools.execute import tool_execute
from ghost_agent.core.agent import GhostAgent, GhostContext
from ghost_agent.core.dream import Dreamer

@pytest.mark.asyncio
async def test_tool_execute_direct_command():
    sandbox_manager = MagicMock()
    sandbox_manager.container = MagicMock()
    
    sandbox_manager.execute = MagicMock(return_value=("hello shell\n", 0))
    
    # Try calling the execute tool with a direct Bash command
    result = await tool_execute(command="echo 'hello shell'", sandbox_manager=sandbox_manager)
    
    assert "EXIT CODE: 0" in result
    assert "hello shell" in result
    
    # Verify the command was dispatched correctly
    sandbox_manager.execute.assert_called_once()
    args = sandbox_manager.execute.call_args[0]
    assert "bash -c" in args[0]
    assert "echo" in args[0]

def test_dreamer_setup_script_execution():
    # Verify the code structure directly because the loop is highly integrated
    source = inspect.getsource(Dreamer.synthetic_self_play)
    assert "setup_script and setup_script.strip():" in source
    assert "await asyncio.to_thread(isolated_context.sandbox_manager.execute, \"python3 .setup.py\", 60)" in source
