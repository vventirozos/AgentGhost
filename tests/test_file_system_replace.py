import pytest
import os
from pathlib import Path
from ghost_agent.tools.file_system import tool_replace_text, tool_file_system

@pytest.fixture
def sandbox_dir(tmp_path):
    return tmp_path

@pytest.mark.asyncio
async def test_tool_replace_file_success(sandbox_dir):
    test_file = sandbox_dir / "test.txt"
    test_file.write_text("Hello World\nLine 2\nLine 3")
    
    res = await tool_replace_text("test.txt", "Line 2", "Replaced Line 2", sandbox_dir)
    assert "SUCCESS" in res
    
    content = test_file.read_text()
    assert content == "Hello World\nReplaced Line 2\nLine 3"

@pytest.mark.asyncio
async def test_tool_replace_file_multiple_matches(sandbox_dir):
    test_file = sandbox_dir / "test.txt"
    test_file.write_text("duplicate\nduplicate")
    
    res = await tool_replace_text("test.txt", "duplicate", "single", sandbox_dir)
    assert "WARNING: Replaced 2 identical occurrences" in res
    
    content = test_file.read_text()
    assert content == "single\nsingle"

@pytest.mark.asyncio
async def test_tool_replace_file_no_match(sandbox_dir):
    test_file = sandbox_dir / "test.txt"
    test_file.write_text("Hello World")
    
    res = await tool_replace_text("test.txt", "Goodbye", "Hello", sandbox_dir)
    assert "Error: The exact search block was NOT found" in res

@pytest.mark.asyncio
async def test_tool_replace_file_heuristic_match(sandbox_dir):
    test_file = sandbox_dir / "test.txt"
    test_file.write_text("def my_func():\n    print('hello')\n    return True")
    
    # Intentionally messed up indentation in the search block
    old_text = "def my_func():\n print('hello')\n return True"
    new_text = "def my_func():\n    print('world')\n    return False"
    
    res = await tool_replace_text("test.txt", old_text, new_text, sandbox_dir)
    assert "SUCCESS: Flexible match found" in res
    
    content = test_file.read_text()
    assert "print('world')" in content
    assert "return False" in content
    assert "print('hello')" not in content

@pytest.mark.asyncio
async def test_tool_replace_file_heuristic_multiple_matches(sandbox_dir):
    test_file = sandbox_dir / "test.txt"
    test_file.write_text("    print('hello')\n\n\n    print('hello')")
    
    # Add a trailing space to ensure it fails exact matching but succeeds in fuzzy matching
    old_text = "print('hello') "
    new_text = "print('world')"
    
    res = await tool_replace_text("test.txt", old_text, new_text, sandbox_dir)
    # The heuristic match warns of multiple flexible instances
    assert "Error: Multiple instances of this text block found" in res
    
    content = test_file.read_text()
    assert content == "    print('hello')\n\n\n    print('hello')"

@pytest.mark.asyncio
async def test_tool_file_system_replace_routing(sandbox_dir):
    test_file = sandbox_dir / "test.txt"
    test_file.write_text("Routing Test")
    
    res = await tool_file_system(
        operation="replace",
        sandbox_dir=sandbox_dir,
        path="test.txt",
        content="Routing Test",
        replace_with="Routed Success"
    )
    assert "SUCCESS" in res
    assert test_file.read_text() == "Routed Success"

@pytest.mark.asyncio
async def test_tool_file_system_replace_missing_args(sandbox_dir):
    test_file = sandbox_dir / "test.txt"
    test_file.write_text("Missing args")
    
    # Missing replace_with
    res1 = await tool_file_system(
        operation="replace",
        sandbox_dir=sandbox_dir,
        path="test.txt",
        content="Missing args"
    )
    assert "SUCCESS: Exact match found" in res1
    
    # Missing content
    res2 = await tool_file_system(
        operation="replace",
        sandbox_dir=sandbox_dir,
        path="test.txt",
        replace_with="New"
    )
    assert "Error: You must specify the exact 'content'" in res2
