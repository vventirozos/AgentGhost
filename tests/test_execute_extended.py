import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from ghost_agent.tools.execute import tool_execute

@pytest.fixture
def mock_sandbox_manager(mock_context):
    from ghost_agent.sandbox.docker import DockerSandbox
    # Mock the sandbox manager to avoid needing real Docker
    mock_sandbox = AsyncMock(spec=DockerSandbox)
    mock_sandbox.execute.return_value = ("Output: Hello", 0)
    mock_context.sandbox_manager = mock_sandbox
    return mock_context

@pytest.mark.asyncio
async def test_execute_python_simple(mock_sandbox_manager):
    # Test simple print
    # Actual Sig: tool_execute(filename, content, sandbox_dir, sandbox_manager, ...)
    # language inferred from extension
    code = "print('Hello')"
    res = await tool_execute(
        filename="test.py", 
        content=code, 
        sandbox_dir=mock_sandbox_manager.sandbox_dir,
        sandbox_manager=mock_sandbox_manager.sandbox_manager
    )
    assert "Output: Hello" in res
    assert "EXIT CODE: 0" in res

@pytest.mark.asyncio
async def test_execute_bash(mock_sandbox_manager):
    # Test bash command
    res = await tool_execute(
        filename="script.sh",
        content="echo Hello", 
        sandbox_dir=mock_sandbox_manager.sandbox_dir,
        sandbox_manager=mock_sandbox_manager.sandbox_manager
    )
    assert "Output: Hello" in res

@pytest.mark.asyncio
async def test_execute_syntax_error(mock_sandbox_manager):
    # Mock a failure
    mock_sandbox_manager.sandbox_manager.execute.return_value = ("SyntaxError: invalid syntax", 1)
    
    res = await tool_execute(
        filename="broken.py", 
        content="def broken", 
        sandbox_dir=mock_sandbox_manager.sandbox_dir,
        sandbox_manager=mock_sandbox_manager.sandbox_manager
    )
    assert "EXIT CODE: 1" in res
    assert "SyntaxError" in res

@pytest.mark.asyncio
async def test_execute_markdown_stripping(mock_sandbox_manager):
    # Ensure tool strips markdown
    code = "```python\nprint('Clean')\n```"
    mock_sandbox_manager.sandbox_manager.execute.return_value = ("Clean", 0)
    
    await tool_execute(
        filename="clean.py", 
        content=code, 
        sandbox_dir=mock_sandbox_manager.sandbox_dir,
        sandbox_manager=mock_sandbox_manager.sandbox_manager
    )
    
    # Check what was actually executed
    # The tool should call sandbox.execute with cleaned code
    args, _ = mock_sandbox_manager.sandbox_manager.execute.call_args
    executed_cmd = args[0]
    assert "```" not in executed_cmd
