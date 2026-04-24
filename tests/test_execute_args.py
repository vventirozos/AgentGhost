import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
from ghost_agent.tools.execute import tool_execute
import shlex

@pytest.mark.asyncio
async def test_tool_execute_shell_argument_escaping():
    sandbox_dir = Path("/tmp/sandbox")
    sandbox_manager = MagicMock()
    sandbox_manager.execute = MagicMock(return_value=("output", 0))
    
    filename = "script.sh"
    content = "echo $1"
    
    malicious_arg = "'; echo HACKED; '"
    args = [malicious_arg]
    
    with patch("asyncio.to_thread", side_effect=lambda func, *a, **k: func(*a, **k) if not asyncio.iscoroutinefunction(func) else func(*a, **k)):
        with patch.object(Path, "write_text"), patch("os.chmod"):
             await tool_execute(filename, content, sandbox_dir, sandbox_manager, args=args)
             
             expected_cmd = "bash script.sh " + shlex.quote(malicious_arg)
             sandbox_manager.execute.assert_called_with(expected_cmd)
