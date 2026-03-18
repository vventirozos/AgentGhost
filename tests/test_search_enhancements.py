import pytest
import asyncio
from unittest.mock import patch, MagicMock

from src.ghost_agent.tools.registry import TOOL_DEFINITIONS
from src.ghost_agent.tools.search import tool_search_ddgs, tool_deep_research

def test_registry_descriptions():
    web_search = next((t for t in TOOL_DEFINITIONS if t["function"]["name"] == "web_search"), None)
    deep_research = next((t for t in TOOL_DEFINITIONS if t["function"]["name"] == "deep_research"), None)

    assert web_search is not None
    assert deep_research is not None

    desc_ws = web_search["function"]["description"]
    desc_dr = deep_research["function"]["description"]

    assert "CRITICAL: Keep your queries concise" in desc_ws
    assert "site:wikipedia.org" in desc_ws

    assert "CRITICAL: Keep your queries concise" in desc_dr
    assert "site:wikipedia.org" in desc_dr

@pytest.mark.asyncio
@patch("ddgs.DDGS")
@patch("src.ghost_agent.tools.search.importlib.util.find_spec")
async def test_tool_search_ddgs_params(mock_find_spec, mock_ddgs):
    mock_find_spec.return_value = True
    
    mock_ddgs_instance = MagicMock()
    mock_ddgs.return_value.__enter__.return_value = mock_ddgs_instance
    mock_ddgs_instance.text.return_value = [{"title": "Test", "body": "Body", "href": "http://test.com"}]

    result = await tool_search_ddgs("test param query", None)

    mock_ddgs_instance.text.assert_any_call("test param query", max_results=10, region="wt-wt", safesearch="moderate")
    assert "1. Test" in result

@pytest.mark.asyncio
@patch("ddgs.DDGS")
@patch("src.ghost_agent.tools.search.helper_fetch_url_content")
@patch("src.ghost_agent.tools.search.importlib.util.find_spec")
async def test_tool_deep_research_params_and_sorting(mock_find_spec, mock_fetch, mock_ddgs):
    mock_find_spec.return_value = True
    mock_fetch.return_value = "Mocked content"
    
    mock_ddgs_instance = MagicMock()
    mock_ddgs.return_value.__enter__.return_value = mock_ddgs_instance
    
    mock_ddgs_instance.text.return_value = [
        {"href": "https://www.forbes.com/article"}, # Junk (Skipped)
        {"href": "https://www.tiktok.com/video"}, # Junk (Skipped)
        {"href": "https://en.wikipedia.org/wiki/Test"}, # Valid 1
        {"href": "https://developer.mozilla.org/en-US/"}, # Valid 2
        {"href": "https://random-news-site.com/blog"}, # Valid 3
        {"href": "https://github.com/test/test"}, # Valid 4
        {"href": "https://stackoverflow.com/questions/123"}, # Valid 5 (Dropped, only top 4)
        {"href": "https://normal-site.com/info"} # Valid 6 (Dropped)
    ]

    result = await tool_deep_research("test query", False, None)

    mock_ddgs_instance.text.assert_any_call("test query", max_results=15, region="wt-wt", safesearch="moderate")
    
    # Junk domains should be filtered out
    assert "https://www.forbes.com" not in result
    assert "tiktok.com" not in result
    
    # The first 4 valid domains should be included
    assert "https://en.wikipedia.org/wiki/Test" in result
    assert "https://developer.mozilla.org/en-US/" in result
    assert "https://random-news-site.com/blog" in result
    assert "https://github.com/test/test" in result
    
    # 5th and 6th valid domains should be dropped because max is 4
    assert "stackoverflow.com" not in result
    assert "normal-site.com/info" not in result

@pytest.mark.asyncio
@patch("ddgs.DDGS")
@patch("src.ghost_agent.tools.search.importlib.util.find_spec")
async def test_tool_search_ddgs_empty_list_triggers_tor_cycle(mock_find_spec, mock_ddgs):
    mock_find_spec.return_value = True
    
    mock_ddgs_instance = MagicMock()
    mock_ddgs.return_value.__enter__.return_value = mock_ddgs_instance
    mock_ddgs_instance.text.return_value = [] # Empty list to trigger ValueError
    
    # We pass an invalid tor_proxy parameter to force the ValueError to be swallowed
    # inside the Except block and it should sleep for 1 sec rather than crashing. 
    # Because there are 3 retries, the final error message should be returned:
    result = await tool_search_ddgs("test query", None)
    
    assert "DuckDuckGo returned ZERO results" in result
    assert mock_ddgs_instance.text.call_count >= 3


@pytest.mark.asyncio
@patch("ddgs.DDGS")
@patch("src.ghost_agent.tools.search.importlib.util.find_spec")
async def test_tool_deep_research_empty_list_triggers_tor_cycle(mock_find_spec, mock_ddgs):
    mock_find_spec.return_value = True
    
    mock_ddgs_instance = MagicMock()
    mock_ddgs.return_value.__enter__.return_value = mock_ddgs_instance
    mock_ddgs_instance.text.return_value = [] # Empty list to trigger error
    
    result = await tool_deep_research("test query", False, None)
    
    # Ensure it reaches the final failure block and the deep research fails gracefully
    assert "CRITICAL ERROR: Deep Research search phase failed." in result
    assert mock_ddgs_instance.text.call_count >= 3


@pytest.mark.asyncio
@patch("src.ghost_agent.tools.search.pretty_log")
@patch("ddgs.DDGS")
@patch("src.ghost_agent.tools.search.importlib.util.find_spec")
async def test_tool_search_ddgs_exception_logging(mock_find_spec, mock_ddgs, mock_pretty_log):
    mock_find_spec.return_value = True
    
    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.__enter__.return_value = mock_ddgs_instance
    mock_ddgs_instance.text.side_effect = Exception("Simulated Search Crash")
    mock_ddgs.return_value = mock_ddgs_instance

    await tool_search_ddgs("test params", None)
    
    assert mock_ddgs_instance.text.call_count == 3
    
    logs = [call.args[1] for call in mock_pretty_log.call_args_list]
    assert "Simulated Search Crash" in logs

@pytest.mark.asyncio
@patch("src.ghost_agent.tools.search.pretty_log")
@patch("ddgs.DDGS")
@patch("src.ghost_agent.tools.search.importlib.util.find_spec")
async def test_tool_deep_research_exception_logging(mock_find_spec, mock_ddgs, mock_pretty_log):
    mock_find_spec.return_value = True
    
    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.__enter__.return_value = mock_ddgs_instance
    mock_ddgs_instance.text.side_effect = Exception("Simulated DR Crash")
    mock_ddgs.return_value = mock_ddgs_instance

    result = await tool_deep_research("test params", False, None)
    
    assert mock_ddgs_instance.text.call_count == 3
    assert "CRITICAL ERROR: Deep Research search phase failed." in result
    
    # Verify the exception string was logged as a warning
    logs = [call.args[1] for call in mock_pretty_log.call_args_list]
    assert "Simulated DR Crash" in logs

from src.ghost_agent.utils.helpers import request_new_tor_identity

@patch("src.ghost_agent.utils.helpers.socket.socket")
@patch("src.ghost_agent.utils.helpers.platform.system")
@patch("src.ghost_agent.utils.helpers.subprocess.run")
def test_request_new_tor_identity_fallback_mac(mock_subprocess_run, mock_platform_system, mock_socket):
    # Setup socket to raise an exception (simulating connection refused)
    mock_socket_instance = MagicMock()
    mock_socket_instance.__enter__.return_value = mock_socket_instance
    mock_socket_instance.connect.side_effect = ConnectionRefusedError("[Errno 61] Connection refused")
    mock_socket.return_value = mock_socket_instance
    
    # Setup platform to simulate macOS
    mock_platform_system.return_value = "Darwin"
    
    # Run the function
    success, msg = request_new_tor_identity()
    
    # Assert socket connection was attempted
    mock_socket_instance.connect.assert_called_once_with(("127.0.0.1", 9051))
    
    # Assert fallback brew command was called
    mock_subprocess_run.assert_called_once_with(["brew", "services", "restart", "tor"], check=True, capture_output=True)
    
    # Assert success response
    assert success is True
    assert "via brew services restart" in msg

@patch("src.ghost_agent.utils.helpers.socket.socket")
@patch("src.ghost_agent.utils.helpers.platform.system")
@patch("src.ghost_agent.utils.helpers.subprocess.run")
def test_request_new_tor_identity_fallback_linux(mock_subprocess_run, mock_platform_system, mock_socket):
    # Setup socket to raise an exception (simulating connection refused)
    mock_socket_instance = MagicMock()
    mock_socket_instance.__enter__.return_value = mock_socket_instance
    mock_socket_instance.connect.side_effect = ConnectionRefusedError("[Errno 111] Connection refused")
    mock_socket.return_value = mock_socket_instance
    
    # Setup platform to simulate Linux
    mock_platform_system.return_value = "Linux"
    
    # Run the function
    success, msg = request_new_tor_identity()
    
    # Assert socket connection was attempted
    mock_socket_instance.connect.assert_called_once_with(("127.0.0.1", 9051))
    
    # Assert fallback systemctl command was called
    mock_subprocess_run.assert_called_once_with(["sudo", "-n", "systemctl", "restart", "tor"], check=True, capture_output=True)
    
    # Assert success response
    assert success is True
    assert "via systemctl restart" in msg

@patch("src.ghost_agent.utils.helpers.socket.socket")
@patch("src.ghost_agent.utils.helpers.platform.system")
@patch("src.ghost_agent.utils.helpers.subprocess.run")
def test_request_new_tor_identity_fallback_failure(mock_subprocess_run, mock_platform_system, mock_socket):
    # Setup socket to raise an exception (simulating connection refused)
    mock_socket_instance = MagicMock()
    mock_socket_instance.__enter__.return_value = mock_socket_instance
    mock_socket_instance.connect.side_effect = ConnectionRefusedError("[Errno 61] Connection refused")
    mock_socket.return_value = mock_socket_instance
    
    # Setup platform to simulate macOS
    mock_platform_system.return_value = "Darwin"
    
    # Setup subprocess to fail
    import subprocess
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(1, ["brew", "services", "restart", "tor"])
    
    # Run the function
    success, msg = request_new_tor_identity()
    
    # Assert socket connection was attempted
    mock_socket_instance.connect.assert_called_once_with(("127.0.0.1", 9051))
    
    # Assert fallback brew command was called
    mock_subprocess_run.assert_called_once_with(["brew", "services", "restart", "tor"], check=True, capture_output=True)
    
    # Assert failure response
    assert success is False
    assert "Fallback restart also failed" in msg
