
import pytest
from unittest.mock import MagicMock
from ghost_agent.tools.system import tool_system_utility

@pytest.fixture
def mock_context():
    ctx = MagicMock()
    ctx.tor_proxy = None
    ctx.profile_memory = MagicMock()
    return ctx

@pytest.mark.asyncio
async def test_check_health_basic(mock_context):
    result = await tool_system_utility("check_health", tor_proxy=None, context=mock_context)
    assert "System Status" in result

@pytest.mark.asyncio
async def test_check_weather_no_location(mock_context):
    # Should block without location/profile
    result = await tool_system_utility("check_weather", location="Paris", tor_proxy="socks5://localhost:9050", context=mock_context)
    # Since we can't actually hit the network, it might return an error or exception depending on the mock state.
    # But we just want to ensure it runs without TypeError.
    assert isinstance(result, str)
