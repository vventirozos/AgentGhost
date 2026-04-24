import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path
from ghost_agent.core.agent import GhostAgent
from ghost_agent.core.dream import Dreamer
from ghost_agent.core.prompts import QWEN_TOOL_PROMPT, SYNTHETIC_CHALLENGE_PROMPT

@pytest.mark.asyncio
async def test_agent_think_tag_stripping():
    # Test that handle_chat strips <think> tags from the final message content
    mock_context = MagicMock()
    mock_context.args.perfect_it = False
    mock_context.args.use_planning = False
    mock_context.args.max_context = 4000
    mock_context.args.temperature = 0.5
    mock_context.profile_memory = None
    mock_context.skill_memory = None
    mock_context.graph_memory = None
    mock_context.memory_system = None
    mock_context.args.smart_memory = 0.0
    
    mock_client = AsyncMock()
    # Mock stream_chat_completion to return an async generator yielding <think> content
    async def mock_stream(*args, **kwargs):
        chunks = [
            b'data: {"choices": [{"delta": {"content": "<think>\\nI am thinking\\n</think>\\nHello User!"}}]}',
            b'data: [DONE]'
        ]
        for c in chunks:
            yield c
            
    mock_client.stream_chat_completion = mock_stream
    mock_context.llm_client = mock_client
    
    agent = GhostAgent(mock_context)
    body = {
        "model": "test-model",
        "stream": True,
        "messages": [{"role": "user", "content": "Run a calculation for me"}]
    }
    
    final_ai, ct, rid = await agent.handle_chat(body, None)
    
    # Verify the context was appended with the stripped content
    assert "<think>" not in final_ai
    assert "Hello User!" in final_ai

@pytest.mark.asyncio
async def test_agent_payload_penalties():
    # Test that the LLM request payload gets the presence and frequency penalties
    mock_context = MagicMock()
    mock_context.args.perfect_it = False
    mock_context.args.use_planning = False
    mock_context.args.max_context = 4000
    mock_context.args.temperature = 0.5
    mock_context.profile_memory = None
    mock_context.skill_memory = None
    mock_context.graph_memory = None
    mock_context.memory_system = None
    mock_context.args.smart_memory = 0.0
    
    mock_client = AsyncMock()
    async def mock_stream(*args, **kwargs):
        assert "payload" not in kwargs, "Just checking stream arguments"
        payload = args[0]
        assert payload.get("presence_penalty") == 0.20
        assert payload.get("frequency_penalty") == 0.20
        yield b'data: [DONE]'
        
    mock_client.stream_chat_completion = mock_stream
    mock_context.llm_client = mock_client
    
    agent = GhostAgent(mock_context)
    body = {
        "model": "test-model",
        "stream": True,
        "messages": [{"role": "user", "content": "Run a calculation for me"}]
    }
    
    await agent.handle_chat(body, None)

@pytest.mark.asyncio
async def test_dreamer_slate_wipe_on_rejection():
    # Test that the self-play loop wipes context on validation failure
    mock_context = MagicMock()
    mock_client = AsyncMock()
    
    # 1. Challenge generation
    async def mock_chat_comp(payload, **kwargs):
        return {
            "choices": [{"message": {"content": "<challenge_prompt>Do it.</challenge_prompt><validation_script>False</validation_script>"}}]
        }
    mock_client.chat_completion = mock_chat_comp
    mock_context.llm_client = mock_client
    
    # mock sandbox manager execute
    mock_sandbox = MagicMock()
    
    # Mocking different calls. First setup? Pycompile? Validator execution?
    def mock_execute(cmd, *args, **kwargs):
        if cmd == "python3 .setup.py":
            return "", 0
        if "py_compile" in cmd:
            return "", 0
        if cmd == "python3 .validator.py":
            return "Test Failed", 1 # Failure to trigger the slate wipe!
        return "", 0
        
    mock_sandbox.execute = mock_execute
    
    # mock the GhostAgent in the test
    with patch('ghost_agent.core.agent.GhostAgent') as MockAgent:
        mock_agent_instance = MockAgent.return_value
        
        # When handle_chat is called, it simulates the agent attempting the challenge
        async def mock_handle_chat(body_ref, *args, **kwargs):
            # We artificially add some tool attempts to the messages list
            body_ref["messages"].append({"role": "assistant", "content": "I failed"})
            body_ref["messages"].append({"role": "tool", "content": "Error"})
            return "Final output", 123, "req_123"
            
        mock_agent_instance.handle_chat = mock_handle_chat
        mock_agent_instance._get_recent_transcript.return_value = "fake transcript"
        mock_agent_instance.disabled_tools = set()
        mock_agent_instance.available_tools = {}
        
        dreamer = Dreamer(mock_context)
        with patch('ghost_agent.sandbox.docker.DockerSandbox') as DockMock:
            DockMock.return_value = mock_sandbox
            res = await dreamer.synthetic_self_play()
            
            # Check status
            assert "FAILURE (Exhausted 3 attempts)" in res

def test_prompt_updates():
    # Check that prompt files have the requested constraints
    assert "strip()" in SYNTHETIC_CHALLENGE_PROMPT
    assert "split both the actual output and expected output" in SYNTHETIC_CHALLENGE_PROMPT
    assert "ANTI-PARALYSIS" in QWEN_TOOL_PROMPT
    assert "strictly forbidden to brainstorm multiple alternatives" in QWEN_TOOL_PROMPT
