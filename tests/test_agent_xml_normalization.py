import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from ghost_agent.core.agent import GhostAgent, GhostContext

@pytest.fixture
def mock_context():
    context = MagicMock(spec=GhostContext)
    context.sandbox_manager = AsyncMock()
    context.sandbox_dir = "/tmp"
    context.args = MagicMock()
    context.args.perfect_it = False
    context.args.temperature = 0.5
    context.args.smart_memory = 0.0
    context.args.use_planning = False
    context.args.max_context = 4000
    context.profile_memory = MagicMock()
    context.profile_memory.get_context_string.return_value = ""
    context.scratchpad = MagicMock()
    context.scratchpad.list_all.return_value = ""
    context.memory_system = None
    context.skill_memory = None
    context.llm_client = MagicMock()
    context.llm_client.swarm_clients = None
    return context

@pytest.mark.asyncio
async def test_json_to_xml_schema_uses_tool_def(mock_context):
    """Test that the schema generator uses <tool_def> instead of <tool>."""
    agent = GhostAgent(mock_context)
    
    agent.available_tools = {
        "fake_tool": AsyncMock()
    }
    
    mock_chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": "SUCCESS", "tool_calls": []}}]})
    agent.context.llm_client.chat_completion = mock_chat_completion
    
    body = {"messages": [{"role": "user", "content": "please run the tool"}]}
    
    class FakeBgTasks:
        def add_task(self, *a, **k): pass
        
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions", return_value=[{"function": {"name": "fake_tool", "description": "a fake tool", "parameters": {"properties": {}, "required": []}}}]):
        await agent.handle_chat(body, FakeBgTasks())
        
    assert mock_chat_completion.call_count > 0
    payload = mock_chat_completion.call_args[0][0]

    # Tool schemas now ride in the user-message header (KV-cache-stable
    # system slot architectural change). Assert the marker appears in *any*
    # message of the payload, and confirm the system slot stays clean of
    # the legacy `<tool>` form.
    all_content = "\n".join(m["content"] for m in payload["messages"])
    assert "<tool_def>" in all_content
    assert "</tool_def>" in all_content
    system_msg = next(m["content"] for m in payload["messages"] if m["role"] == "system")
    assert "<tool>" not in system_msg
    # The system slot must NOT contain tool schemas anymore — that's the
    # whole point of the KV-cache-stable optimisation.
    assert "<tool_def>" not in system_msg

@pytest.mark.asyncio
async def test_xml_normalization_heals_tool_tag(mock_context):
    """Test that the pre-parser normalization heals <tool> into <tool_call>."""
    agent = GhostAgent(mock_context)
    
    invalid_content = '''<tool>\n<function name="fake_tool">\n<parameter name="arg">value</parameter>\n</function>\n</tool>'''
    
    mock_chat_completion = AsyncMock(side_effect=[
        {"choices": [{"message": {"content": invalid_content, "tool_calls": []}}]},
        {"choices": [{"message": {"content": "DONE", "tool_calls": []}}]}
    ])
    agent.context.llm_client.chat_completion = mock_chat_completion
    
    tool_mock = AsyncMock(return_value="executed_successfully")
    agent.available_tools = {"fake_tool": tool_mock}
    
    body = {"messages": [{"role": "user", "content": "do the fake tool"}]}
    
    class FakeBgTasks:
        def add_task(self, *a, **k): pass
        
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions", return_value=[{"function": {"name": "fake_tool", "description": "a fake tool", "parameters": {"properties": {"arg": {"type": "string"}}, "required": []}}}]):
         await agent.handle_chat(body, FakeBgTasks())
    
    tool_mock.assert_called_once()
    kwargs = tool_mock.call_args.kwargs
    # Check that arguments were parsed
    assert kwargs.get("arg") == "value"

@pytest.mark.asyncio
async def test_ui_scrubber_hides_hallucinated_tool_tags(mock_context):
    """Test that the UI scrubber removes <tool> tags in final_ai_content."""
    agent = GhostAgent(mock_context)
    
    # LLM returns <tool> and the tool throws an error to force it to return final response
    # Actually if the tool executes successfully, it will return the last output.
    invalid_content = 'Thinking process...\n<tool>\n<function name="fake_tool"></function>\n</tool>'
    
    mock_chat_completion = AsyncMock(side_effect=[
        {"choices": [{"message": {"content": invalid_content, "tool_calls": []}}]},
        {"choices": [{"message": {"content": "Final AI message", "tool_calls": []}}]}
    ])
    agent.context.llm_client.chat_completion = mock_chat_completion
    
    agent.available_tools = {"fake_tool": AsyncMock(return_value="tool_done")}
    
    body = {"messages": [{"role": "user", "content": "trigger fail"}]}
    
    class FakeBgTasks:
        def add_task(self, *a, **k): pass
    
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions", return_value=[{"function": {"name": "fake_tool"}}]):
         final_ai_content, _, _ = await agent.handle_chat(body, FakeBgTasks())
         
    # The returned content should NOT contain <tool> because it was healed into tool_call and then scrubbed
    # But wait, final_ai_content will be "Process finished successfully... tool_done" since tool executed
    # Let's say tool does not execute because the parser misses it? No, parser heals it now!
    assert "<tool>" not in final_ai_content
    assert "<tool_call>" not in final_ai_content

