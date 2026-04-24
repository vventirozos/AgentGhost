import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from ghost_agent.core.agent import GhostAgent, GhostContext
from ghost_agent.tools.file_system import _get_safe_path
from ghost_agent.tools.vision import tool_vision_analysis

@pytest.fixture
def mock_context():
    context = MagicMock(spec=GhostContext)
    
    mock_llm = AsyncMock()
    context.llm_client = mock_llm
    
    mock_args = MagicMock()
    mock_args.max_context = 10000
    mock_args.smart_memory = 0.0
    mock_args.use_planning = False
    mock_args.temperature = 0.7
    mock_args.perfect_it = False
    context.args = mock_args
    
    context.journal = MagicMock()
    context.memory_system = MagicMock()
    context.profile_memory = MagicMock()
    context.profile_memory.get_context_string = lambda: ""
    context.skill_memory = MagicMock()
    
    # Create an actual temp dir so list_files works properly
    temp_dir = tempfile.mkdtemp()
    context.sandbox_dir = Path(temp_dir)
    context.cached_sandbox_state = None
    
    return context

@pytest.mark.asyncio
async def test_json_root_key_healing(mock_context):
    mock_llm = mock_context.llm_client
    # Simulate LLM outputting pure JSON with the tool name as the root key
    mock_llm.chat_completion.side_effect = [
        {
            "choices": [{"message": {"content": '<tool_call>\n{"execute": {"filename": "test.py", "content": "print(1)"}}\n</tool_call>'}}]
        },
        {
            "choices": [{"message": {"content": "SUCCESS: I have finished."}}]
        }
    ]
    
    agent = GhostAgent(context=mock_context)
    agent.available_tools = {"execute": AsyncMock(return_value="executed_successfully")}
    
    messages = [{"role": "user", "content": "Run script"}]
    body = {"messages": messages, "model": "test-model"}
    await agent.handle_chat(body, background_tasks=MagicMock(), request_id="req123")
    
    tools_called = [m["name"] for m in body["messages"] if m.get("role") == "tool"]
    assert "execute" in tools_called, "JSON with root key tool name was not healed properly"

@pytest.mark.asyncio
async def test_system_parse_error_messaging(mock_context):
    mock_llm = mock_context.llm_client
    # Simulate an invalid tool block that triggers system_parse_error
    mock_llm.chat_completion.side_effect = [
        {
            "choices": [{"message": {"content": '<tool_call><invalid>1</invalid></tool_call>'}}]
        },
        {
            "choices": [{"message": {"content": "SUCCESS: I have finished."}}]
        }
    ]
    
    agent = GhostAgent(context=mock_context)
    agent.available_tools = {"execute": AsyncMock(return_value="executed_successfully")}
    
    messages = [{"role": "user", "content": "Run script"}]
    body = {"messages": messages, "model": "test-model"}
    await agent.handle_chat(body, background_tasks=MagicMock(), request_id="req123")
    
    err_msgs = [m["content"] for m in body["messages"] if m.get("role") == "tool" and m.get("name") == "system"]
    assert len(err_msgs) > 0
    # Recovery message is now reason-specific. For this fixture (an
    # <invalid>1</invalid> bare tag with no <function>), the branch is
    # "no_function_tag" — which must direct the model to the correct
    # shape. The old assertion on the verbatim legacy wording was
    # brittle — check behaviourally for the XML example instead.
    assert "SYSTEM ERROR" in err_msgs[0]
    assert "<function name=" in err_msgs[0]

@pytest.mark.asyncio
async def test_auto_diagnostic_sandbox_state_append(mock_context):
    mock_llm = mock_context.llm_client
    mock_llm.chat_completion.side_effect = [
        {
            "choices": [{"message": {"content": '{"name": "execute", "arguments": {"filename": "test.py", "content": "print(1)"}}'}}]
        },
        {
            "choices": [{"message": {"content": "SUCCESS"}}]
        }
    ]
    
    agent = GhostAgent(context=mock_context)
    # Simulate a tool throwing an exception to trigger auto-diagnostic
    agent.available_tools = {"execute": AsyncMock(side_effect=Exception("Critical meltdown"))}
    
    messages = [{"role": "user", "content": "Run script"}]
    body = {"messages": messages, "model": "test-model"}
    await agent.handle_chat(body, background_tasks=MagicMock(), request_id="req123")
    
    diag_msgs = [m for m in body["messages"] if m.get("role") == "user" and "AUTO-DIAGNOSTIC" in m.get("content", "")]
    assert len(diag_msgs) > 0
    diag_text = diag_msgs[0]["content"]
    assert "CURRENT SANDBOX DIRECTORY STRUCTURE" in diag_text or "[Empty]" in diag_text

def test_file_system_safe_path_stripping():
    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox_dir = Path(tmpdir)
        # Spaces should be stripped before the slash is processed
        result = _get_safe_path(sandbox_dir, "   /sneaky.txt   ")
        assert result.name == "sneaky.txt"
        assert result.parent.resolve() == sandbox_dir.resolve()

@pytest.mark.asyncio
async def test_vision_tool_missing_file_suggestion():
    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox_dir = Path(tmpdir)
        # Target a missing file
        res = await tool_vision_analysis(action="describe_picture", target="missing.jpg", sandbox_dir=sandbox_dir)
        assert "Error: File 'missing.jpg' not found." in res
        assert "Use the `file_system` tool with operation='list_files' to check the sandbox directory" in res
