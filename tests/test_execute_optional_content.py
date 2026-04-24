import pytest
import asyncio
from pathlib import Path
from src.ghost_agent.tools.execute import tool_execute

class MockSandboxManager:
    def __init__(self):
        self.commands_run = []
        
    def execute(self, cmd, timeout=None):
        self.commands_run.append(cmd)
        return "Executed", 0

@pytest.mark.asyncio
async def test_execute_optional_content_file_not_exist(tmp_path):
    sandbox_mgr = MockSandboxManager()
    
    # Passing no content and file doesn't exist
    result = await tool_execute(
        filename="test_script.py",
        content=None,
        sandbox_dir=tmp_path,
        sandbox_manager=sandbox_mgr
    )
    
    assert "File 'test_script.py' does not exist. You must provide 'content' to create it." in result
    assert "EXIT CODE: 1" in result

@pytest.mark.asyncio
async def test_execute_optional_content_file_exists(tmp_path):
    sandbox_mgr = MockSandboxManager()
    
    script_path = tmp_path / "existing_script.py"
    script_path.write_text("print('Hello World')")
    
    # Passing no content but file exists
    result = await tool_execute(
        filename="existing_script.py",
        content=None,
        sandbox_dir=tmp_path,
        sandbox_manager=sandbox_mgr
    )
    
    # Should not fail, should execute
    assert "EXIT CODE: 0" in result
    assert len(sandbox_mgr.commands_run) > 0
    assert any("existing_script.py" in cmd for cmd in sandbox_mgr.commands_run)

@pytest.mark.asyncio
async def test_execute_same_content_bypasses_write(tmp_path):
    sandbox_mgr = MockSandboxManager()
    
    script_path = tmp_path / "same_script.py"
    code = "print('Hello World')\n"
    script_path.write_text(code)
    
    # Get modified time before execution
    mtime_before = script_path.stat().st_mtime
    
    # Passing the exact same content
    result = await tool_execute(
        filename="same_script.py",
        content=code,
        sandbox_dir=tmp_path,
        sandbox_manager=sandbox_mgr
    )
    
    # Should execute successfully without returning the "EXACT SAME CODE" system error
    assert "EXIT CODE: 0" in result
    assert "EXACT SAME CODE SUBMITTED" not in result
    
    # The file should not have been written to (modified time should be same)
    mtime_after = script_path.stat().st_mtime
    assert mtime_before == mtime_after
