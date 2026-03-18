import pytest
from unittest.mock import MagicMock
from src.ghost_agent.tools.execute import tool_execute
from pathlib import Path

@pytest.mark.asyncio
async def test_html_guard_with_html_tags():
    sandbox_dir = Path("/tmp/sandbox")
    sandbox_manager = MagicMock()
    
    html_content = "<html>\n<body>\n<h1>Hello</h1>\n</body>\n</html>"
    
    result = await tool_execute(
        filename="test.py", 
        content=html_content, 
        sandbox_dir=sandbox_dir, 
        sandbox_manager=sandbox_manager
    )
    
    assert "It looks like you are trying to write HTML, CSS, or JS" in result
    assert "DO NOT use the 'execute' tool to create web pages" in result

@pytest.mark.asyncio
async def test_html_guard_with_css_tags():
    sandbox_dir = Path("/tmp/sandbox")
    sandbox_manager = MagicMock()
    
    css_content = "body {\n    background-color: red;\n}"
    
    result = await tool_execute(
        filename="style.py", 
        content=css_content, 
        sandbox_dir=sandbox_dir, 
        sandbox_manager=sandbox_manager
    )
    
    assert "It looks like you are trying to write HTML, CSS, or JS" in result
    assert "DO NOT use the 'execute' tool to create web pages" in result

@pytest.mark.asyncio
async def test_html_guard_valid_python(tmp_path):
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    
    sandbox_manager = MagicMock()
    sandbox_manager.execute.return_value = ("Success", 0)
    
    python_content = "def hello():\n    print('World')\n\nhello()"
    
    result = await tool_execute(
        filename="test.py", 
        content=python_content, 
        sandbox_dir=sandbox_dir, 
        sandbox_manager=sandbox_manager
    )
    
    assert "It looks like you are trying to write HTML, CSS, or JS" not in result
