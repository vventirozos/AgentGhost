import pytest
import asyncio
from pathlib import Path
from ghost_agent.tools.execute import tool_execute

# Mock Sandbox Manager
class MockSandbox:
    def execute(self, cmd, timeout=None):
        if "python3 -m black" in cmd: return "reformatted", 0
        if "error_script.py" in cmd:
             return 'File "error_script.py", line 2, in <module>\n    x = 1 / 0\nZeroDivisionError: division by zero', 1
        return "Hello World", 0

@pytest.mark.asyncio
async def test_execute_error_diagnostics(tmp_path):
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    
    mock_manager = MockSandbox()
    
    code = """
print("Start")
x = 1 / 0
print("End")
"""
    result = await tool_execute("error_script.py", code, sandbox_dir, mock_manager)
    
    assert "EXIT CODE: 1" in result
    assert "DIAGNOSTIC HINT" in result
    assert "Error detected at Line 2" in result
    assert "x = 1 / 0" in result
