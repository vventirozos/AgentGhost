import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

from ghost_agent.tools.system import tool_get_weather
from ghost_agent.tools.memory import tool_knowledge_base
from ghost_agent.tools.file_system import tool_file_system, tool_download_file
from ghost_agent.tools.database import tool_postgres_admin
from ghost_agent.tools.execute import tool_execute
from ghost_agent.tools.tasks import tool_schedule_task

@pytest.mark.asyncio
async def test_weather_proxy_none():
    # Verify tor_proxy=None does not throw AttributeError
    res = await tool_get_weather(tor_proxy=None, location="London")
    assert "REPORT" in res or "SYSTEM ERROR" in res # We just care it doesn't crash with AttributeError

@pytest.mark.asyncio
async def test_memory_scratchpad_removed():
    res = await tool_knowledge_base("scratchpad", Path("/tmp"), None)
    assert "Unknown action 'scratchpad'" in res

@pytest.mark.asyncio
async def test_file_system_write_missing_content():
    res = await tool_file_system("write", Path("/tmp"), path="test.txt", content=None)
    assert "mandatory for write operations" in res.lower()

@pytest.mark.asyncio
async def test_download_file_size_limit():
    url = "http://example.com/large"
    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"Content-Length": "50000001"}
    
    with patch("ghost_agent.tools.file_system.curl_requests", None), \
         patch("ghost_agent.tools.file_system.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        
        mock_context = MagicMock()
        mock_context.__aenter__.return_value = mock_resp
        mock_context.__aexit__.return_value = None
        mock_client.stream = MagicMock(return_value=mock_context)
        
        res = await tool_download_file(url, Path("/tmp"), None)
        assert "Error: File is too large" in res
        assert "50MB" in res

@pytest.mark.asyncio
async def test_database_statement_timeout():
    mock_psycopg = MagicMock()
    with patch.dict("sys.modules", {"psycopg2": mock_psycopg, "psycopg2.extras": MagicMock(), "tabulate": MagicMock()}):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_psycopg.connect.return_value = mock_conn
        
        # Action 'query'
        res = await tool_postgres_admin("query", "uri", "SELECT 1")
        
        # Verify execute was called with SET statement_timeout
        calls = mock_cur.execute.call_args_list
        assert calls[0][0][0] == "SET statement_timeout = 15000;"
        assert "SELECT 1" in calls[1][0][0]

@pytest.mark.asyncio
async def test_execute_stubbornness_large_file():
    mock_path = MagicMock()
    mock_path.exists.return_value = True
    
    mock_stat = MagicMock()
    # Mock size > 1MB
    mock_stat.st_size = 1_500_000
    mock_path.stat.return_value = mock_stat
    mock_path.parent.mkdir = MagicMock()
    
    with patch("ghost_agent.tools.execute._get_safe_path", return_value=mock_path):
        mock_manager = MagicMock()
        mock_manager.execute.return_value = ("Success", 0)
        
        # the read_text shouldn't be called because the file is too large
        mock_path.read_text = MagicMock()
        
        await tool_execute("test.py", "print('hello')", Path("/tmp"), mock_manager)
        
        # Ensure read_text wasn't called because of the size boundary
        mock_path.read_text.assert_not_called()

@pytest.mark.asyncio
async def test_schedule_task_uuid_and_interval_fallback():
    import apscheduler
    mock_scheduler = MagicMock()
    
    with patch("ghost_agent.tools.tasks.run_proactive_task_fn", new=MagicMock()):
        # Test valid interval
        await tool_schedule_task("test_job", "prompt", "interval: 30", mock_scheduler, None)
        call_args = mock_scheduler.add_job.call_args
        assert call_args[1]["seconds"] == 30
        
        job_id_1 = call_args[1]["id"]
        assert "task_" in job_id_1
        assert len(job_id_1.split("_")) == 3 # task_hash_uuid
        
        # Test hallucinated interval fallback
        await tool_schedule_task("test_job_2", "prompt", "interval: 30 minutes", mock_scheduler, None)
        call_args = mock_scheduler.add_job.call_args
        assert call_args[1]["seconds"] == 60 # Defaulted to 60 because "30 minutes" throws ValueError
