import pytest
from pathlib import Path
from ghost_agent.tools.file_system import tool_read_document_chunked, tool_file_system

@pytest.fixture
def temp_dirs(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    return {"sandbox": sandbox}

@pytest.mark.asyncio
async def test_chunked_reading_text(temp_dirs):
    sandbox = temp_dirs["sandbox"]
    test_file = sandbox / "big_test.txt"
    
    # Create file exactly 20,000 bytes text
    content = "0123456789" * 2000 # 20kb
    test_file.write_text(content)
    
    # Read chunk 1 with size 8000
    res1 = await tool_read_document_chunked("big_test.txt", sandbox, page=1, chunk_size=8000)
    assert "Section 1 of 3" in res1
    assert content[:8000] in res1
    
    # Read chunk 2 with size 8000
    res2 = await tool_read_document_chunked("big_test.txt", sandbox, page=2, chunk_size=8000)
    assert "Section 2 of 3" in res2
    # Ensure overlap starting context is within the read chunk logic (8000 length minus 200 overlap offset * 1)
    # Check string size and ensure it bounded correctly without throwing errors
    assert len(res2) > 7000

@pytest.mark.asyncio
async def test_chunked_reading_router(temp_dirs):
    sandbox = temp_dirs["sandbox"]
    test_file = sandbox / "small_test.txt"
    test_file.write_text("Hello World!")
    
    # Route through file_system tool
    res = await tool_file_system("read_chunked", sandbox, path="small_test.txt", page=1, chunk_size=1000)
    assert "TEXT DATA" in res
    assert "Hello World!" in res
    
    # Out of bounds request should fail gracefully
    res_bounds = await tool_file_system("read_chunked", sandbox, path="small_test.txt", page=999, chunk_size=1000)
    assert "Error: Requested section 999 exceeds total sections" in res_bounds
