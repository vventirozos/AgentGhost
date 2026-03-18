import pytest
import os
from pathlib import Path
from ghost_agent.tools.file_system import (
    tool_read_file, tool_write_file, tool_list_files, tool_file_search, tool_inspect_file
)

@pytest.mark.asyncio
async def test_read_write_file(temp_dirs):
    sandbox = temp_dirs["sandbox"]
    
    # 1. Write File
    res = await tool_write_file(sandbox_dir=sandbox, filename="test.txt", content="Hello World")
    assert "SUCCESS" in res
    assert (sandbox / "test.txt").read_text() == "Hello World"
    
    # 2. Read File
    content = await tool_read_file(sandbox_dir=sandbox, filename="test.txt")
    assert content == "Hello World"

@pytest.mark.asyncio
async def test_path_traversal_prevention(temp_dirs):
    sandbox = temp_dirs["sandbox"]
    
    # Try to write outside sandbox
    result = await tool_write_file(sandbox_dir=sandbox, filename="../hack.txt", content="hack")
    assert "Security Error" in result or "Traversal attempt" in result

@pytest.mark.asyncio
async def test_list_files(temp_dirs):
    sandbox = temp_dirs["sandbox"]
    (sandbox / "a.txt").touch()
    (sandbox / "b.log").touch()
    (sandbox / "subdir").mkdir()
    (sandbox / "subdir" / "c.csv").touch()
    
    listing = await tool_list_files(sandbox_dir=sandbox)
    assert "a.txt" in listing
    assert "b.log" in listing
    assert "subdir/c.csv" in listing

@pytest.mark.asyncio
async def test_file_search(temp_dirs):
    sandbox = temp_dirs["sandbox"]
    (sandbox / "code.py").write_text("def main(): pass")
    (sandbox / "notes.txt").write_text("TODO: fix bug")
    
    # Search for code
    # Actual sig: tool_file_search(pattern: str, sandbox_dir: Path, filename: str = None)
    res = await tool_file_search(pattern="def main", sandbox_dir=sandbox)
    assert "code.py" in res
    
    # Search for notes
    res2 = await tool_file_search(pattern="TODO", sandbox_dir=sandbox)
    assert "notes.txt" in res2

@pytest.mark.asyncio
async def test_inspect_file(temp_dirs):
    sandbox = temp_dirs["sandbox"]
    f = sandbox / "data.csv"
    f.write_text("id,name\n1,alice\n2,bob")
    
    info = await tool_inspect_file(sandbox_dir=sandbox, filename="data.csv")
    # tool_inspect_file returns just the content lines, no metadata
    assert "id,name" in info
    assert "1,alice" in info
