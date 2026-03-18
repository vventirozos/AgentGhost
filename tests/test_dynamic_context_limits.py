import pytest
import asyncio
from unittest.mock import MagicMock, patch
from pathlib import Path
from ghost_agent.core.agent import GhostAgent, GhostContext
from ghost_agent.tools.file_system import tool_read_file, tool_read_document_chunked
from ghost_agent.tools.search import tool_deep_research

@pytest.fixture
def mock_agent():
    ctx = MagicMock(spec=GhostContext)
    ctx.llm_client = MagicMock()
    ctx.llm_client.vision_clients = None
    ctx.args = MagicMock()
    ctx.args.max_context = 100000 # Large context window
    agent = GhostAgent(context=ctx)
    agent.agent_semaphore = asyncio.Semaphore(10)
    return agent

def test_prepare_planning_context_dynamic_limit(mock_agent):
    # max_context = 100000. Limit is max(4000, 100000 * 3.5 * 0.1) = 35000
    tools_run = [{"content": "A" * 30000}]
    
    result = mock_agent._prepare_planning_context(tools_run)
    # Should NOT be truncated because 30000 < 35000
    assert "TRUNCATED" not in result
    assert "A" * 30000 in result

    tools_run_large = [{"content": "B" * 40000}]
    result_large = mock_agent._prepare_planning_context(tools_run_large)
    # Should BE truncated because 40000 > 35000
    assert "TRUNCATED" in result_large
    assert len(result_large) < 38000 # Approximately 35000 + some header/footer
    assert "B" * 35000 in result_large

def test_get_recent_transcript_dynamic_limit(mock_agent):
    # max_context = 100000. 
    # msg_limit = max(40, 100000 / 500) = 200 messages
    # char_limit = max(500, 100000 * 3.5 * 0.02) = 7000 chars

    # Test msg truncation: Provide 250 messages
    messages = [{"role": "user", "content": f"msg {i}"} for i in range(250)]
    recent = mock_agent._get_recent_transcript(messages)
    
    # Needs to be sliced to the last 200 messages
    assert len(recent.split("\n")) >= 200 
    assert "msg 50\n" in recent # the first msg is index 50
    assert "msg 49\n" not in recent

    # Test char_limit truncation: Provide 1 giant message
    long_msg = "C" * 10000
    messages_long = [{"role": "user", "content": long_msg}]
    recent_long = mock_agent._get_recent_transcript(messages_long)
    
    # Should truncate the content down to 7000 chars approx
    assert len(recent_long) < 7500
    assert len(recent_long) > 6900
    assert "C" * 6900 in recent_long

@pytest.mark.asyncio
async def test_tool_read_file_dynamic_limit(tmp_path):
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    
    # max_bytes = max(150000, int(100000 * 3.5 * 0.5)) = 175000
    
    # 1. Test size = 170000 (Should pass)
    pass_file = sandbox_dir / "pass.txt"
    with open(pass_file, "w") as f:
        f.write("A" * 170000)
        
    result_pass = await tool_read_file("pass.txt", sandbox_dir, max_context=100000)
    assert len(result_pass) == 170000
    assert "Error" not in result_pass
    
    # 2. Test size = 180000 (Should fail)
    fail_file = sandbox_dir / "fail.txt"
    with open(fail_file, "w") as f:
        f.write("B" * 180000)
        
    result_fail = await tool_read_file("fail.txt", sandbox_dir, max_context=100000)
    assert "Error: File 'fail.txt' is too large to read entirely" in result_fail
    assert "Limit is 170.9 KB" in result_fail # 175000 / 1024

@pytest.mark.asyncio
async def test_tool_read_document_chunked_dynamic_limit(tmp_path):
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    
    # max_chunk = max(30000, int(100000 * 3.5 * 0.2)) = 70000

    test_file = sandbox_dir / "doc.txt"
    with open(test_file, "w") as f:
        f.write("D" * 100000)
        
    # Requesting chunk_size=80000 should get capped to 70000
    result = await tool_read_document_chunked("doc.txt", sandbox_dir, page=1, chunk_size=80000, max_context=100000)
    
    # overlap is chunk_size//4
    # effective chunks vary, but the returned text chunk should be based on 70000
    # Actually, the tool reads `read_amount = chunk_size` bytes directly.
    # So the return string length is approx len("[TEXT DATA...]") + 70000
    assert len(result) > 70000
    assert len(result) < 71000
    
@pytest.mark.asyncio
async def test_deep_research_dynamic_limit():
    from unittest.mock import AsyncMock
    
    tor_proxy = None
    query = "test query"
    anonymous = False
    
    llm_client = MagicMock()
    llm_client.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": "Mock summary"}}]})
    
    long_html = "E" * 50000
    
    # Mock DuckDuckGo and the fetcher
    with patch("duckduckgo_search.DDGS", autospec=True) as mock_ddgs, \
         patch("ghost_agent.tools.search.helper_fetch_url_content", new_callable=AsyncMock) as mock_fetch:
         
        mock_ddgs.return_value.__enter__.return_value.text.return_value = [{"href": "http://example.com"}]
        mock_fetch.return_value = long_html
        
        # max_context = 100000
        # url_char_limit = max(15000, int(100000 * 3.5 * 0.2)) = 70000
        # The payload should contain the text sliced to 70000 chars, but text is only 50k here,
        # so let's make text 80000 config
        long_html = "E" * 80000
        mock_fetch.return_value = long_html
        
        result = await tool_deep_research(query, anonymous, tor_proxy, llm_client=llm_client, model_name="test", max_context=100000)
        
        # Verify the payload sizing logic
        assert llm_client.chat_completion.called
        call_args = llm_client.chat_completion.call_args[0][0]
        user_msg = call_args["messages"][0]["content"]
        
        # Check that exactly 15000 chars (approx) of 'E's are in the message
        # len("Extract ONLY the hard facts... Source text:\n") + 15000
        assert len(user_msg) > 15000
        assert len(user_msg) < 15500
        assert "E" * 15000 in user_msg
        assert "E" * 15001 not in user_msg
