
import pytest
import pytest_asyncio
import asyncio
from pathlib import Path
from ghost_agent.tools.file_system import tool_read_file, tool_write_file, tool_list_files, tool_download_file, tool_file_system
from unittest.mock import patch, MagicMock, AsyncMock

@pytest.fixture
def sandbox(tmp_path):
    return tmp_path

@pytest.mark.asyncio
async def test_write_read_basic(sandbox):
    filename = "test.txt"
    content = "Hello Ghost"
    
    # Write
    result = await tool_write_file(filename, content, sandbox)
    assert "SUCCESS" in result
    assert (sandbox / filename).exists()
    assert (sandbox / filename).read_text() == content
    
    # Read
    read_content = await tool_read_file(filename, sandbox)
    assert read_content == content

@pytest.mark.asyncio
async def test_write_auto_mkdir(sandbox):
    # Verify it creates parent directories (self-healing)
    filename = "nested/deep/folder/test.txt"
    content = "Deep content"
    
    result = await tool_write_file(filename, content, sandbox)
    assert "SUCCESS" in result
    assert (sandbox / filename).exists()

@pytest.mark.asyncio
async def test_path_traversal_protection(sandbox):
    # Try to write outside sandbox
    # The tool uses sandbox / filename.lstrip("/"). 
    # Python's pathlib / operator usually handles ".." by resolving, 
    # BUT simple joining might be vulnerable if not checked.
    # Let's see how the tool behaves. 
    # buffer = sandbox / "../outside.txt" -> This actually resolves to sibling of sandbox.
    # However, the tool code does: sandbox_dir / str(filename).lstrip("/")
    # If filename is "../outside.txt", then sandbox / "../outside.txt".
    # We want to ensure the tool strictly keeps it inside.
    # NOTE: The current tool implementation does NOT explicitly check for ".." traversal escaping the sandbox root 
    # logic: `path = sandbox_dir / str(filename).lstrip("/")`
    # If I pass `../../etc/passwd`, it might try to write there if the user runs as root (very bad).
    # Let's TEST this. If it fails, we found a security bug to fix!
    
    filename = "../outside_attack.txt"
    content = "attack"
    
    # We perform the write
    await tool_write_file(filename, content, sandbox)
    
    # Check where it landed
    # Correct behavior: It should be inside the sandbox, potentially as "outside_attack.txt" or flattened, 
    # OR the tool should reject ".." components. 
    # The current code: `sandbox_dir / "../outside.txt"` -> resolves to `sandbox_dir.parent / "outside.txt"`
    
    # Let's verify if the file exists OUTSIDE the sandbox
    outside_path = sandbox.parent / "outside_attack.txt"
    inside_path = sandbox / "outside_attack.txt"
    
    # If outside_path exists, we have a vulnerability (and the test 'passes' capturing the bug behavior for now, 
    # or fails if we assert safety). 
    # We WANT to assert safety. So this test expects the tool to block it or neutralize it.
    
    assert not outside_path.exists(), "SECURITY FAIL: Path traversal allowed writing outside sandbox!"

@pytest.mark.asyncio
async def test_read_nonexistent(sandbox):
    result = await tool_read_file("ghost_file.txt", sandbox)
    assert "Error" in result
    assert "not found" in result

@pytest.mark.asyncio
async def test_write_empty_content(sandbox):
    result = await tool_write_file("empty.txt", "", sandbox)
    assert "Error" in result
    assert "empty" in result

class MockResponse:
    status_code = 200
    headers = {}

class MockStream:
    async def __aenter__(self):
        return MockResponse()
    async def __aexit__(self, *args):
        pass

class MockClient:
    def stream(self, method, url):
        return MockStream()
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        pass

@pytest.mark.asyncio
async def test_download_auto_heals_filename(sandbox, monkeypatch):
    # We mock tool_download_file itself to verify what filename it received
    mock_download = AsyncMock(return_value="SUCCESS")
    monkeypatch.setattr("ghost_agent.tools.file_system.tool_download_file", mock_download)

    proxy = None
    
    # 1. No filename, URL with no path -> should auto-heal to download.bin
    url1 = "https://example.com/"
    await tool_file_system("download", sandbox, tor_proxy=proxy, url=url1, path=None)
    mock_download.assert_called_with(url=url1, sandbox_dir=sandbox, tor_proxy=proxy, filename="download.bin")
    
    # 2. Empty string filename, URL with path -> should auto-heal to file.zip
    url2 = "https://example.com/file.zip"
    await tool_file_system("download", sandbox, url=url2, path="   ")
    mock_download.assert_called_with(url=url2, sandbox_dir=sandbox, tor_proxy=None, filename="file.zip")
    
    # 3. Same as URL
    await tool_file_system("download", sandbox, url=url2, path=url2)
    mock_download.assert_called_with(url=url2, sandbox_dir=sandbox, tor_proxy=None, filename="file.zip")
    
    # 4. Valid works (does not overwrite)
    await tool_file_system("download", sandbox, url=url2, path="custom.txt")
    mock_download.assert_called_with(url=url2, sandbox_dir=sandbox, tor_proxy=None, filename="custom.txt")
