
import pytest
from unittest.mock import MagicMock
from ghost_agent.core.agent import GhostAgent, GhostContext

@pytest.fixture
def mock_agent():
    ctx = MagicMock(spec=GhostContext)
    ctx.llm_client = MagicMock()
    ctx.llm_client.vision_clients = None
    ctx.args = MagicMock()
    ctx.args.max_context = 4000
    # Mocking attributes accessed in specific methods if needed
    agent = GhostAgent(context=ctx)
    return agent

def test_prepare_planning_context_truncation(mock_agent):
    # Case 1: Short output
    tools_run = [{"content": "Short output"}]
    result = mock_agent._prepare_planning_context(tools_run)
    assert result == "Tool [unknown]: Short output"

    # Case 2: Long output (Over 5000 chars)
    # create a string of 6000 chars
    long_content = "A" * 30000
    tools_run = [{"content": long_content}]
    result = mock_agent._prepare_planning_context(tools_run)
    
    # Expect truncation
    expected_marker = "\n\n... [TRUNCATED: Tool output too long. Showing top results only.]"
    
    # Result = "Tool [unknown]: " + 4800-chars + marker
    # The length depends on the "Tool [unknown]: " prefix length which is 16 chars
    # So 16 + 4800 + len(marker) = 4816 + 65 = 4881 approximately
    
    assert len(result) < 10000
    assert expected_marker in result
    assert "A" * 4000 in result
    assert result.endswith(expected_marker)

def test_process_rolling_window_sliding(mock_agent):
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "tool", "name": "exec", "content": "Result A"},
        {"role": "tool", "name": "exec", "content": "Result A"}, # Duplicate
        {"role": "assistant", "content": "Memory updated..."},  # Meta-chatter
        {"role": "assistant", "content": "Real response"}
    ]
    
    # The new logic is a pure sliding window. No deduplication, no filtering.
    clean = mock_agent.process_rolling_window(messages, max_tokens=1000)
    
    # Check that all messages are preserved (they all fit in 1000 tokens)
    assert len(clean) == 5
    
    tool_msgs = [m for m in clean if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assert tool_msgs[0]["content"] == "Result A"
    assert tool_msgs[1]["content"] == "Result A"
    
    assist_msgs = [m for m in clean if m["role"] == "assistant"]
    assert len(assist_msgs) == 2
    assert assist_msgs[1]["content"] == "Real response"

def test_agent_semaphore_initialization(mock_agent):
    """
    Verify turns are SERIALIZED (semaphore == 1) — 2026-07-07 (#22). Per-turn
    state (`last_user_content`, `current_project_id`) lives on the singleton
    context, so concurrent turns (esp. a cron job firing mid-user-turn) would
    clobber each other's project scope. One llama slot makes concurrency mostly
    illusory anyway, so serializing is near-free and closes that hazard.
    """
    assert mock_agent.agent_semaphore._value == 1

@pytest.mark.asyncio
async def test_agent_streaming(mock_agent):
    from unittest.mock import AsyncMock

    # With the planner disabled, the agent goes directly to final generation on the first turn.
    # Returning empty content + no tool_calls forces is_final_generation=True which triggers
    # stream_chat_completion when stream=True.
    async def mock_final(*args, **kwargs):
        return {
            "choices": [{"message": {"content": '{"thought": "done", "next_action_id": "none", "required_tool": "none"}', "tool_calls": []}}]
        }
    mock_agent.context.llm_client.chat_completion = MagicMock(side_effect=mock_final)

    # Mock the stream path
    async def fake_stream(*args, **kwargs):
        yield b"data: {\"choices\": [{\"delta\": {\"content\": \"hello\"}}]}\n\n"
        yield b"data: [DONE]\n\n"

    mock_agent.context.llm_client.stream_chat_completion = MagicMock(return_value=fake_stream())
    
    # Mock context config
    mock_agent.context.args = MagicMock()
    mock_agent.context.args.use_planning = True
    mock_agent.context.args.smart_memory = 0.0
    mock_agent.context.args.max_context = 4000
    mock_agent.context.args.temperature = 0.5
    mock_agent.context.scratchpad = MagicMock()
    mock_agent.context.scratchpad.list_all.return_value = ""
    mock_agent.context.profile_memory = MagicMock()
    mock_agent.context.memory_system = None
    mock_agent.context.skill_memory = None

    def sync_return(): return ""
    mock_agent.context.profile_memory.get_context_string.return_value = sync_return()

    body = {
        "messages": [{"role": "user", "content": "calculate math"}],
        "model": "test-model",
        "stream": True
    }
    
    bg_tasks = MagicMock()
    
    result_content, created_time, req_id = await mock_agent.handle_chat(body, bg_tasks)
    
    # Verify we get an async generator wrapper back
    assert hasattr(result_content, '__aiter__')
    
    # Verify the yielded content
    chunks = []
    async for chunk in result_content:
        chunks.append(chunk)
        
    assert len(chunks) == 2

    # The stream-scrub path (enabled on final-generation turns, which
    # this test exercises via use_planning + no tool calls) rewraps
    # upstream content chunks into synthetic SSE chunks with richer
    # metadata (id / created / model). The assertion is now
    # shape-tolerant: verify the content reached the client and the
    # second chunk is the [DONE] sentinel. Byte-exact passthrough is
    # not required and would regress the moment the wrapper learns to
    # scrub.
    import json as _json
    first = chunks[0].decode("utf-8")
    assert first.startswith("data: ") and first.endswith("\n\n")
    first_payload = _json.loads(first[len("data: "):].strip())
    assert first_payload["choices"][0]["delta"]["content"] == "hello"
    assert chunks[1] == b"data: [DONE]\n\n"

@pytest.mark.asyncio
async def test_agent_manage_skills_bypasses_meta_nudge(mock_agent):
    from unittest.mock import AsyncMock, patch
    
    # Simulate the agent returning a tool_call for "manage_skills"
    async def mock_chat_completion(*args, **kwargs):
        # We need a first pass where tool is called, then a second pass where it's a regular generation
        # Otherwise handle_chat keeps iterating if there's always a tool call
        if mock_chat_completion.call_count == 1:
            return {
                "choices": [{"message": {
                    "content": '{"thought": "deleting skill", "next_action_id": "none", "required_tool": "manage_skills"}', 
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "manage_skills", "arguments": '{"action": "delete", "skill_name": "x"}'}
                    }]
                }}]
            }
        else:
            return {
                "choices": [{"message": {
                    "content": '{"thought": "done", "next_action_id": "none", "required_tool": "none"}', 
                    "tool_calls": []
                }}]
            }
            
    mock_chat_completion.call_count = 0
    def side_effect(*args, **kwargs):
        mock_chat_completion.call_count += 1
        import asyncio
        return asyncio.ensure_future(mock_chat_completion(*args, **kwargs))
        
    mock_agent.context.llm_client.chat_completion = MagicMock(side_effect=side_effect)
    
    # Mocking execution to return success so it doesn't loop forever
    mock_agent.context.registry = MagicMock()
    
    async def fake_tool(**kwargs): return "Success"
    mock_agent.context.registry.get_available_tools.return_value = {
        "manage_skills": fake_tool
    }
    mock_agent.context.args = MagicMock(use_planning=False, smart_memory=0.0, max_context=4000, temperature=0.7)
    mock_agent.context.scratchpad = MagicMock()
    mock_agent.context.scratchpad.list_all.return_value = ""
    mock_agent.context.profile_memory = MagicMock()
    mock_agent.context.memory_system = MagicMock()
    mock_agent.context.skill_memory = MagicMock()
    
    def sync_return(): return ""
    mock_agent.context.profile_memory.get_context_string.return_value = sync_return()

    body = {
        "messages": [{"role": "user", "content": "delete custom skill xyz"}],
        "model": "test-model"
    }
    
    with patch("ghost_agent.core.agent.pretty_log") as mock_log:
        class FakeBgTasks:
            def add_task(self, *a, **k): pass
        
        result, _, _ = await mock_agent.handle_chat(body, FakeBgTasks())
        
        # Verify that pretty_log was NOT called with the "Checklist Nudge" warning
        for call in mock_log.call_args_list:
            if call and call[0]:
                if "Checklist Nudge" in call[0]:
                    pytest.fail("Agent triggered meta-task nudge even though manage_skills was called!")
