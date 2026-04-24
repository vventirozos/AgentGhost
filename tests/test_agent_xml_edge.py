import pytest
from unittest.mock import MagicMock
from ghost_agent.core.agent import GhostAgent

@pytest.fixture
def mock_context():
    context = MagicMock()
    context.sandbox_manager = MagicMock()
    context.sandbox_dir = "/tmp"
    context.args = MagicMock()
    context.args.perfect_it = False
    return context

@pytest.mark.asyncio
async def test_xml_self_closing_parameters(mock_context):
    """Test parsing of XML tool calls using <parameter name="x" value="y" /> self-closing syntax."""
    agent = GhostAgent(mock_context)
    
    # Qwen sometimes outputs this instead of the explicit <parameter>value</parameter> tags
    raw_response = """
    <think>Thinking</think>
    <tool_call>
    <function name="execute">
    <parameter name="command" value="ls -la" />
    <parameter name="timeout" value="30" />
    </function>
    </tool_call>
    """
    
    mock_msg = {"role": "assistant", "content": raw_response}
    
    agent._execute_turn = MagicMock(return_value=([{"name": "execute"}], "")) # We bypass the complex loop
    
    # Manually test the parsing block from agent.py
    import re
    from ghost_agent.core.agent import extract_json_from_text
    
    tool_calls = []
    blocks = re.split(r'<tool_call.*?>', raw_response, flags=re.IGNORECASE)
    for block in blocks[1:]:
        block_content = re.split(r'</tool_call.*?>', block, flags=re.IGNORECASE)[0]
        
        func_match = re.search(r'<function(?:\s+name=|=)(.*?)>', block_content, re.IGNORECASE)
        if func_match:
            func_name = func_match.group(1).strip().strip('"').strip("'")
            args_val = {}
            
            param_matches = list(re.finditer(r'<parameter(?:\s+name=|=)([^>]+)>(.*?)</parameter>', block_content, re.DOTALL | re.IGNORECASE))
            if param_matches:
                for p in param_matches:
                    p_name = p.group(1).strip().strip('"').strip("'")
                    p_val = p.group(2).strip()
                    args_val[p_name] = p_val
            else:
                alt_matches = re.finditer(r'<parameter\s+name=["\']([^"\']+)["\']\s+value=["\']([^"\']+)["\']\s*(?:/|>.*?</parameter>)', block_content, re.DOTALL | re.IGNORECASE)
                for p in alt_matches:
                    args_val[p.group(1)] = p.group(2)
            tool_calls.append({"name": func_name, "arguments": args_val})
            
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "execute"
    assert tool_calls[0]["arguments"]["command"] == "ls -la"
    assert tool_calls[0]["arguments"]["timeout"] == "30"


@pytest.mark.asyncio
async def test_xml_pure_json_fallback(mock_context):
    """Test parsing when the LLM outputs a raw JSON dictionary directly inside a <tool_call>."""
    agent = GhostAgent(mock_context)
    
    raw_response = """
    <tool_call>
    {"name": "file_system", "arguments": {"operation": "read", "filename": "access.log"}}
    </tool_call>
    """
    
    import re
    from ghost_agent.core.agent import extract_json_from_text
    
    tool_calls = []
    blocks = re.split(r'<tool_call.*?>', raw_response, flags=re.IGNORECASE)
    for block in blocks[1:]:
        block_content = re.split(r'</tool_call.*?>', block, flags=re.IGNORECASE)[0]
        
        func_match = None
        if '<function' not in block_content.lower():
            t_data = extract_json_from_text(block_content)
            if t_data and "name" in t_data:
                func_match = True
                tool_calls.append(t_data)
        
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "file_system"
    assert tool_calls[0]["arguments"]["operation"] == "read"
    assert tool_calls[0]["arguments"]["filename"] == "access.log"
