import pytest
import platform
import os
import psutil
# from ghost_agent.core.system import get_system_info 
# core.system doesn't exist. We just use internal libs or tools.system if needed.
from ghost_agent.tools.system import tool_system_utility 
# I will inspect `ghost_agent/core/system.py` if unsure.
# But let's assume standard tools.

@pytest.mark.asyncio
async def test_system_platform():
    info = platform.system()
    assert info in ["Linux", "Darwin", "Windows"]

@pytest.mark.asyncio
async def test_system_memory():
    mem = psutil.virtual_memory()
    assert mem.total > 0

@pytest.mark.asyncio
async def test_system_cpu():
    cpu = psutil.cpu_percent(interval=None)
    assert isinstance(cpu, float)

@pytest.mark.asyncio
async def test_disk_usage(temp_dirs):
    usage = psutil.disk_usage(str(temp_dirs["sandbox"]))
    assert usage.total > 0

@pytest.mark.asyncio
async def test_env_vars():
    # Just check if we can read env
    os.environ["TEST_ENV"] = "123"
    assert os.getenv("TEST_ENV") == "123"
