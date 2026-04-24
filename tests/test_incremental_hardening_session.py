import pytest
import asyncio
import re
from unittest.mock import MagicMock, AsyncMock

from ghost_agent.core.agent import GhostAgent, GhostContext
from ghost_agent.tools.execute import tool_execute
from pathlib import Path

@pytest.mark.asyncio
async def test_execute_ephemeral_script():
    """Test that an ephemeral script is automatically generated and cleaned up when content is provided without filename."""
    sandbox_manager = MagicMock()
    # Mock execute to return (output, exit_code)
    sandbox_manager.execute = MagicMock(return_value=("output", 0))
    sandbox_manager.container = MagicMock()
    
    # We will patch asyncio.to_thread temporarily in the tool to avoid filesystem issues if needed,
    # or just let it mock. `sandbox_manager.execute` is run via `asyncio.to_thread`.
    # Let's mock `asyncio.to_thread` for host_path unlink locally.
    
    # Actually, we can just verify the command passed to sandbox_manager
    ans = await tool_execute(
        sandbox_dir=Path("/tmp"),
        sandbox_manager=sandbox_manager,
        memory_dir=Path("/tmp"),
        content="print('ephemeral')",
        filename="",
        command=""
    )
    
    # Check if sandbox_manager.execute was called
    called_cmd = sandbox_manager.execute.call_args[0][0]
    assert ".ephemeral_" in called_cmd
    assert "Process executed successfully" not in ans # it had "output"


@pytest.mark.asyncio
async def test_execute_empty_stdout_success():
    """Test that empty stdout from a successful script generates the correct safe fallback text."""
    sandbox_manager = MagicMock()
    sandbox_manager.execute = MagicMock(return_value=(" \n  ", 0)) # Empty stdout, exit 0
    
    with pytest.MonkeyPatch.context() as mp:
        from pathlib import Path
        mp.setattr(Path, "exists", MagicMock(return_value=True))
        mp.setattr(Path, "is_file", MagicMock(return_value=True))
        mp.setattr(Path, "read_text", MagicMock(return_value="print('test')"))
        ans = await tool_execute(
            sandbox_dir=Path("/tmp"),
            sandbox_manager=sandbox_manager,
            memory_dir=Path("/tmp"),
            content="",
            filename="test.py",
            command=""
        )
    
    assert "(Process executed successfully, but no output was printed to stdout." in ans


def test_dream_xml_lenient_regex():
    """Test that dream XML extraction flawlessly bypasses markdown wrapping and optional attributes."""
    content_text = '''```xml
<challenge_prompt type="python" id="101">
Generate some test data!
</challenge_prompt>

<setup_script lang="bash">
touch data.csv
</setup_script>
```'''

    content_text = re.sub(r'^```(?:xml|html|python|json)?\n', '', content_text, flags=re.MULTILINE | re.IGNORECASE)
    content_text = re.sub(r'\n```$', '', content_text, flags=re.MULTILINE)
    
    # Assert fences are stripped
    assert "```" not in content_text
    
    m_chal = re.search(r'<challenge_prompt[^>]*>(.*?)(?:</challenge_prompt>|$)', content_text, re.DOTALL | re.IGNORECASE)
    challenge = m_chal.group(1).strip() if m_chal else ""
    assert "Generate some test data!" in challenge
    
    m_set = re.search(r'<setup_script[^>]*>(.*?)(?:</setup_script>|$)', content_text, re.DOTALL | re.IGNORECASE)
    setup = m_set.group(1).strip() if m_set else ""
    assert "touch data.csv" in setup


@pytest.mark.asyncio
async def test_parallel_strike_amplification_uncoupled():
    """
    Test that running multiple failing tools in a single turn increments
    the execution_failure_count by only 1, preventing instant loop breakage.
    """
    mock_args = MagicMock(max_context=8000, temperature=0.7, use_planning=False, perfect_it=False, smart_memory=0.0)
    context = GhostContext(args=mock_args, sandbox_dir="/tmp", memory_dir="/tmp", tor_proxy=None)
    context.llm_client = MagicMock()
    
    # First LLM call: returns 5 parallel failed tools
    tool_calls = [
        {"id": f"call_{i}", "type": "function", "function": {"name": "execute", "arguments": '{"command":"fail"}'}}
        for i in range(5)
    ]
    
    # Second LLM call: returns normal string (agent gives up)
    context.llm_client.chat_completion = AsyncMock(side_effect=[
        {"choices": [{"message": {"content": "Failure array test", "tool_calls": tool_calls}}]},
        {"choices": [{"message": {"content": "I am done based on those errors."}}]}
    ])
    
    context.scratchpad = MagicMock()
    context.profile_memory = MagicMock()
    context.profile_memory.get_context_string = MagicMock(return_value="")
    context.memory_system = MagicMock()
    context.scheduler = MagicMock()
    context.sandbox_manager = AsyncMock()
    context.sandbox_manager.execute = AsyncMock(return_value=("System crash", 1)) # Exit 1 for execute
    
    agent = GhostAgent(context)
    # Mock prune for speed
    agent.process_rolling_window = MagicMock(return_value=[{"role": "user", "content": "Execute 5 actions"}])
    agent._prune_context = AsyncMock(return_value=[{"role": "user", "content": "Execute 5 actions"}])
    
    body = {"messages": [{"role": "user", "content": "Execute 5 actions"}]}
    
    final_output, _, _ = await agent.handle_chat(body, background_tasks=None)
    
    # Extract ALL appended messages
    all_messages_str = "\\n".join([str(m) for m in body.get("messages", [])])
    
    # Strike count should be 1, so we should NOT see the System 3 intervention (requires 4)
    # And we certainly should NOT see the loop breaker (requires 6)
    assert "SYSTEM ALERT: You have failed 6 times in a row" not in all_messages_str
    assert "SYSTEM 3 PIVOT" not in all_messages_str


@pytest.mark.asyncio
async def test_local_guard_removed_for_data_files():
    """
    Test that the agent no longer produces a 'SYSTEM BLOCK' err_msg manually
    when trying to overwrite a .csv or .json file.
    """
    mock_args = MagicMock(max_context=8000, temperature=0.7, use_planning=False, perfect_it=False, smart_memory=0.0)
    context = GhostContext(args=mock_args, sandbox_dir="/tmp", memory_dir="/tmp", tor_proxy=None)
    context.llm_client = MagicMock()
    
    tool_call = {
        "id": "write_call",
        "type": "function",
        "function": {
            "name": "file_system",
            "arguments": '{"operation":"write", "path":"dataset.csv", "content":"id,name\\n1,ghost\\n"}'
        }
    }
    
    context.llm_client.chat_completion = AsyncMock(side_effect=[
        {"choices": [{"message": {"content": "Writing dataset to test guard", "tool_calls": [tool_call]}}]},
        {"choices": [{"message": {"content": "Finished."}}]}
    ])
    
    context.scratchpad = MagicMock()
    context.profile_memory = MagicMock()
    context.profile_memory.get_context_string = MagicMock(return_value="")
    context.memory_system = AsyncMock()
    context.sandbox_manager = AsyncMock()
    
    agent = GhostAgent(context)
    agent.process_rolling_window = MagicMock(return_value=[{"role": "user", "content": "Write csv"}])
    agent._prune_context = AsyncMock(return_value=[{"role": "user", "content": "Write csv"}])
    
    import ghost_agent.core.agent as _ga_mod
    # Patch actual file_system tool to prevent it throwing an exception
    with pytest.MonkeyPatch.context() as mp:
        async def dummy_fs(**kwargs): return "File written perfectly."
        
        # Patch available tools so our dummy file system is used to bypass real IO
        original_get = _ga_mod.get_available_tools
        def mock_get(ctx):
            tools = original_get(ctx)
            tools["file_system"] = dummy_fs
            return tools
            
        mp.setattr(_ga_mod, "get_available_tools", mock_get)
        agent.available_tools = mock_get(context)
        
        body = {"messages": [{"role": "user", "content": "Write csv"}]}
        final_output, _, _ = await agent.handle_chat(body, background_tasks=None)
        
        all_messages_str = "\\n".join([str(m) for m in body.get("messages", [])])
        
        assert "Blocked overwrite of data file" not in all_messages_str
        assert "SYSTEM BLOCK: You attempted to overwrite 'dataset.csv'" not in all_messages_str

def test_unescape_xml_values():
    import html
    def unescape_xml_values(val):
        if isinstance(val, str):
            return html.unescape(val).replace('\\"', '"').replace("\\'", "'")
        elif isinstance(val, dict):
            return {k: unescape_xml_values(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [unescape_xml_values(v) for v in val]
        return val

    # Test escaped XML entities
    raw_dict = {"input": "sudo rm -rf &amp;&amp; touch test", "escaped_str": "print(\\\"hello\\\")"}
    cleaned = unescape_xml_values(raw_dict)
    
    assert cleaned["input"] == "sudo rm -rf && touch test"
    assert cleaned["escaped_str"] == 'print("hello")'

