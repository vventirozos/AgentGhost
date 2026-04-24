
import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch
from ghost_agent.tools.file_system import tool_list_files

@pytest.fixture
def mock_sandbox(tmp_path):
    (tmp_path / "file1.txt").write_text("content")
    (tmp_path / "subdir").mkdir()
    (tmp_path / ".hidden").touch()
    return tmp_path

@pytest.mark.asyncio
async def test_tool_list_files_formatting(mock_sandbox):
    """Test that tool_list_files formats output with trailing slashes for directories."""
    
    # We can run the real function since it uses os.listdir and Path.is_file
    # But we need to ensure the sorting and filtering is correct.
    
    result = await tool_list_files(mock_sandbox)
    
    assert "CURRENT SANDBOX DIRECTORY STRUCTURE:" in result
    
    # Check specific entries
    # Before fix: 📄 file1.txt, 📁 subdir
    # After fix:   file1.txt,   subdir/
    
    # We need to assertions that will FAIL now and PASS after fix.
    
    # The current code produces "📄 file1.txt" and "📁 subdir".
    # The user wants "  file1.txt" and "  subdir/".
    
    # Let's verify the NEW format in this test.
    # If we run this now, it should fail.
    
    assert "  file1.txt" in result
    # In the deep map, empty directories are implicitly skipped because we loop over files.
    # To test recursive mapping, create a file in subdir:
    (mock_sandbox / "subdir" / "file2.txt").write_text("content")
    result_with_subdir = await tool_list_files(mock_sandbox)
    assert "  subdir/file2.txt" in result_with_subdir
    
    # Ensure hidden files are skipped
    assert ".hidden" not in result
    
    # Ensure no emojis if user requested removing them (based on the snippet "  {f}")
    assert "📄" not in result
    assert "📁" not in result

@pytest.mark.asyncio
async def test_tool_list_files_ast_parsing(mock_sandbox):
    """Test that tool_list_files parses Python files and extracts top-level AST signatures."""
    
    # Create a Python file with some classes and functions
    py_code = '''
class MyClass:
    def method(self): pass

def my_func():
    pass

def another_func(x, y):
    return x + y
'''
    (mock_sandbox / "code.py").write_text(py_code)
    
    # Create another python file that has too many functions to test truncation
    many_funcs = "\n".join(f"def func_{i}(): pass" for i in range(10))
    (mock_sandbox / "huge.py").write_text(many_funcs)
    
    result = await tool_list_files(mock_sandbox)
    
    # Check that code.py has "class MyClass" and "def my_func()" appended via AST parser
    assert "  code.py  [class MyClass, def my_func(), def another_func()]" in result
    
    # Check that huge.py truncates after 5 signatures
    assert "  huge.py  [def func_0(), def func_1(), def func_2(), def func_3(), def func_4()...]" in result
    
    # Ensure it works smoothly with subdirectories
    (mock_sandbox / "subdir" / "nested.py").write_text("def nested_main(): pass")
    result_nested = await tool_list_files(mock_sandbox)
    assert "  subdir/nested.py  [def nested_main()]" in result_nested

