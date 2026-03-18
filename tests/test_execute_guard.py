import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from src.ghost_agent.tools.execute import tool_execute

from pathlib import Path

@pytest.mark.asyncio
async def test_execute_sandbox_guard_blocks_native_tools():
    sandbox_manager = MagicMock()
    # Mocking successful sanitized code (bypasses sanitizer)
    content = "import knowledge_base\nkb = knowledge_base.KnowledgeBase()"
    
    result = await tool_execute("test.py", content, Path("mock_dir"), sandbox_manager)
    assert "SYSTEM ERROR" in result
    assert "FORBIDDEN IMPORT DETECTED" in result
    assert "knowledge_base" in result

@pytest.mark.asyncio
async def test_execute_sandbox_guard_allows_valid_imports():
    sandbox_manager = MagicMock()
    sandbox_manager.execute = MagicMock(return_value=("output", 0))
    content = "import os\nprint('hello')"
    
    # Needs a real Path for sandbox_dir
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox_dir = Path(tmpdir)
        result = await tool_execute("test.py", content, sandbox_dir, sandbox_manager)
        # Assuming the execution mocking bypasses actual shell but returns stdout
        assert "EXIT CODE: 0" in result
        
