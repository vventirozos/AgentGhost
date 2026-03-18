import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock

# 1. Test binary file read guard
from ghost_agent.tools.file_system import tool_read_file
@pytest.mark.asyncio
async def test_read_binary_file_guard(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    image_file = sandbox / "test_image.png"
    # Write some non-decodable bytes
    image_file.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")
    
    result = await tool_read_file("test_image.png", sandbox)
    assert "Error" in result
    assert "appears to be a binary file" in result
    assert "vision_analysis" in result

# 2. Test execute forbidden modules guard
from ghost_agent.tools.execute import tool_execute
class MockSandbox:
    def execute(self, cmd, timeout=None):
        return "Executed", 0

@pytest.mark.asyncio
async def test_execute_forbidden_modules_extended(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    manager = MockSandbox()
    
    # Test one of the newly added forbidden modules
    code = "import vision_analysis\nprint('hacking')"
    result = await tool_execute("script.py", code, sandbox, manager)
    assert "FORBIDDEN IMPORT" in result
    assert "vision_analysis" in result

    code2 = "from delegate_to_swarm import something"
    result2 = await tool_execute("script2.py", code2, sandbox, manager)
    assert "FORBIDDEN IMPORT" in result2
    assert "delegate_to_swarm" in result2

# 3. Test memory ingestion binary guard
from ghost_agent.tools.memory import tool_gain_knowledge
@pytest.mark.asyncio
async def test_memory_binary_ingestion_guard(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    exe_file = sandbox / "program.exe"
    exe_file.write_text("mock binary data")
    
    mock_memory_system = MagicMock()
    # tool_gain_knowledge checks library first
    mock_memory_system.get_library.return_value = []
    
    result = await tool_gain_knowledge("program.exe", sandbox, mock_memory_system)
    assert "Disk Error" in result
    assert "Cannot ingest binary or media files" in result
