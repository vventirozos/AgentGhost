
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os

# Ensure the src directory is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ghost_agent.tools.system import tool_check_health

@pytest.fixture
def mock_context():
    context = MagicMock()
    context.tor_proxy = None
    context.llm_client = True
    context.memory_system = True
    context.sandbox_dir = True
    context.scheduler = MagicMock()
    context.scheduler.running = True
    context.scheduler.get_jobs.return_value = [1, 2]
    return context

@pytest.mark.asyncio
async def test_check_health_basic(mock_context):
    """Test health check under normal conditions with all dependencies available."""
    with patch("ghost_agent.tools.system.platform.system", return_value="Linux"), \
         patch("ghost_agent.tools.system.platform.release", return_value="5.15.0"), \
         patch("ghost_agent.tools.system.platform.machine", return_value="x86_64"), \
         patch("ghost_agent.tools.system.os.getloadavg", return_value=(0.5, 0.3, 0.1)), \
         patch("ghost_agent.tools.system.psutil") as mock_psutil, \
         patch("ghost_agent.tools.system.subprocess.run") as mock_run, \
         patch("ghost_agent.tools.system.curl_requests", None), \
         patch("ghost_agent.tools.system.httpx.AsyncClient") as mock_client_cls:

        # Mock psutil
        mock_psutil.cpu_percent.return_value = 10.5
        mock_mem = MagicMock()
        mock_mem.percent = 45.0
        mock_mem.used = 4000 * 1024**2
        mock_mem.total = 8000 * 1024**2
        mock_psutil.virtual_memory.return_value = mock_mem
        
        mock_disk = MagicMock()
        mock_disk.percent = 20.0
        mock_disk.free = 100 * 1024**3
        mock_psutil.disk_usage.return_value = mock_disk

        # Mock Docker
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "24.0.0"

        # Mock HTTPX
        mock_client = AsyncMock()
        mock_client.get.return_value.status_code = 200
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        result = await tool_check_health(context=mock_context)

        assert "System Status: Online" in result
        assert "OS: Linux 5.15.0 (x86_64)" in result
        assert "CPU Load (1/5/15 min): 0.50 / 0.30 / 0.10" in result
        assert "CPU Usage: 10.5%" in result
        assert "Memory: 45.0% used" in result
        assert "Docker: Active (Version 24.0.0)" in result
        assert "Internet: Connected (200)" in result
        assert "Agent Internals: LLM=Active, Memory=Active, Sandbox=Active, Scheduler=Running (2 jobs)" in result

@pytest.mark.asyncio
async def test_check_health_no_psutil(mock_context):
    """Test fallback when psutil is not available."""
    with patch("ghost_agent.tools.system.psutil", None), \
         patch("ghost_agent.tools.system.shutil.disk_usage", return_value=(1000, 500, 500)), \
         patch("ghost_agent.tools.system.subprocess.run") as mock_run, \
         patch("ghost_agent.tools.system.curl_requests", None), \
         patch("ghost_agent.tools.system.httpx.AsyncClient") as mock_client_cls:

        mock_run.return_value.returncode = 1 # Docker fail
        
        mock_client = AsyncMock()
        mock_client.get.return_value.status_code = 500 # Internet fail
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        result = await tool_check_health(context=mock_context)

        assert "System Status: Online" in result
        assert "Disk (/): 50.0% used" in result
        assert "Docker: Inactive or Not Found" in result
        assert "Internet: Connected (500)" in result # It still says connected if response is returned, just status code differs in string

@pytest.mark.asyncio
async def test_check_health_network_failure(mock_context):
    """Test handling of network exceptions."""
    with patch("ghost_agent.tools.system.psutil"), \
         patch("ghost_agent.tools.system.subprocess.run"), \
         patch("ghost_agent.tools.system.curl_requests", None), \
         patch("ghost_agent.tools.system.httpx.AsyncClient") as mock_client_cls:

        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Connection refused")
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        result = await tool_check_health(context=mock_context)

        assert "Internet: Disconnected or Blocked" in result
        assert "Tor: Not Configured" in result

@pytest.mark.asyncio
async def test_check_health_with_tor(mock_context):
    """Test Tor connection check."""
    mock_context.tor_proxy = "socks5://127.0.0.1:9050"
    
    with patch("ghost_agent.tools.system.psutil"), \
         patch("ghost_agent.tools.system.subprocess.run"), \
         patch("ghost_agent.tools.system.curl_requests", None), \
         patch("ghost_agent.tools.system.httpx.AsyncClient") as mock_client_cls:

        mock_client = AsyncMock()
        
        # Responses for internet check and tor check
        # We need to handle multiple calls to client.get
        # 1. Internet check
        # 2. Tor check
        
        response_internet = MagicMock()
        response_internet.status_code = 200
        
        response_tor = MagicMock()
        response_tor.status_code = 200
        response_tor.json.return_value = {"IsTor": True}

        mock_client.get.side_effect = [response_internet, response_tor]
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        result = await tool_check_health(context=mock_context)

        assert "Internet: Connected (200) [via Tor]" in result
        assert "Tor: Connected (Anonymous)" in result
