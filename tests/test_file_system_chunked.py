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

    # Out-of-range page on a SINGLE-section file: the model over-estimated
    # the size. Don't strike — serve the whole file so the turn progresses.
    res_bounds = await tool_file_system("read_chunked", sandbox, path="small_test.txt", page=999, chunk_size=1000)
    assert not res_bounds.lstrip().lower().startswith("error")  # not a strike
    assert "Hello World!" in res_bounds                          # real content served
    assert "smaller than you expected" in res_bounds            # nudge to stop paginating


@pytest.mark.asyncio
async def test_out_of_range_multisection_is_terminal_not_error(temp_dirs):
    # A genuinely chunked file where the model overshoots the end must get
    # terminal, NON-erroring guidance (no strike, no "request higher page").
    sandbox = temp_dirs["sandbox"]
    f = sandbox / "multi.txt"
    f.write_text("0123456789" * 2000)  # 20kb → 3 sections at chunk 8000
    res = await tool_read_document_chunked("multi.txt", sandbox, page=75, chunk_size=8000)
    assert not res.lstrip().lower().startswith("error")  # not a strike
    assert "only 3 sections" in res
    assert "do NOT request a higher page" in res.replace("\n", " ")


@pytest.mark.asyncio
async def test_last_section_footer_is_end_aware(temp_dirs):
    # The final in-range section must NOT tell the model to "use page=N+1"
    # (that footer is what produced the out-of-range requests).
    sandbox = temp_dirs["sandbox"]
    f = sandbox / "two.txt"
    f.write_text("0123456789" * 1200)  # 12kb → 2 sections at chunk 8000
    last = await tool_read_document_chunked("two.txt", sandbox, page=2, chunk_size=8000)
    assert "Section 2 of 2" in last
    assert "LAST section" in last
    assert "page=3" not in last  # never invite the non-existent next page
    # ...but a NON-last section still offers the continue hint
    first = await tool_read_document_chunked("two.txt", sandbox, page=1, chunk_size=8000)
    assert "Use page=2 to continue" in first
