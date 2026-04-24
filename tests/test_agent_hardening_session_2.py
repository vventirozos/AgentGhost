import pytest
import json
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path

from ghost_agent.core.agent import GhostAgent, GhostContext
from ghost_agent.core.prompts import SYNTHETIC_CHALLENGE_PROMPT
from ghost_agent.core.dream import Dreamer

@pytest.fixture
def mock_agent():
    context = MagicMock(spec=GhostContext)
    context.args = MagicMock()
    context.args.use_planning = False
    context.args.smart_memory = 0.0
    context.args.max_context = 4000
    context.args.temperature = 0.7
    
    context.llm_client = MagicMock()
    context.sandbox_dir = None
    context.memory_system = None
    context.profile_memory = None
    context.semantic_memory = MagicMock()
    context.skill_memory = None
    context.journal = None
    context.scratchpad = MagicMock()
    context.scratchpad.list_all.return_value = ""
    
    agent = GhostAgent(context=context)
    return agent

@pytest.mark.asyncio
async def test_strip_think_tags_from_permanent_history(mock_agent):
    """
    Test that the deep cognitive loops are prevented by stripping <think> blocks
    from the permanent message history when finalizing an AI turn.
    """
    req_id = "test-req-think"
    
    async def mock_stream(*args, **kwargs):
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "This is my internal thought loop."}}]},
            {"choices": [{"delta": {"reasoning_content": "</think>"}}]},
            {"choices": [{"delta": {"content": "This is my actual final response."}}]}
        ]
        
        for c in chunks:
            yield f"data: {json.dumps(c)}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"
        
    mock_agent.context.llm_client.stream_chat_completion = mock_stream
    
    body = {
        "messages": [{"role": "user", "content": "Execute something"}],
        "stream": False
    }
    
    with patch("ghost_agent.core.agent.get_active_tool_definitions", return_value=[]):
        result = await mock_agent.handle_chat(body, background_tasks=MagicMock(), request_id=req_id)
        
    assert "This is my actual final response." in str(result)
    assert "<think>" not in str(result)
    assert "This is my internal thought loop." not in str(result)
    if isinstance(result, dict) and "messages" in result and len(result["messages"]) > 1:
        ai_turn = result["messages"][-1]
        assert "<think>" not in ai_turn.get("content", "")

def test_prompts_challenge_prompt_has_setup_script():
    """Verify that the synthetic self-play prompt mandates embedding the setup script strictly."""
    assert "exact Python code of your `setup_script`" in SYNTHETIC_CHALLENGE_PROMPT
    assert "avoid any schema typos" in SYNTHETIC_CHALLENGE_PROMPT

@pytest.mark.asyncio
async def test_dream_validator_script_not_leaked(disable_self_play_templates):
    """Verify that the rejection message gives the agent the feedback
    (expected vs actual diff) but does NOT leak the `.validator.py` source.

    Leaking the validator source turned every struggled-then-won cycle into
    an answer-key lookup — the agent copied validator constants instead of
    reasoning from the diff, and skill-gate lessons were memorised constants
    rather than transferable knowledge. The rejection prompt now contains
    the feedback field only."""
    context = MagicMock()
    context.args = MagicMock()
    context.args.use_planning = False
    context.args.temperature = 0.5
    context.args.smart_memory = 0.0
    context.memory_system = None
    context.llm_client = MagicMock()
    
    async def mock_generate_challenge(*args, **kwargs):
        return {
            "choices": [{"message": {"content": "<challenge_prompt>\nTask Description\n</challenge_prompt>\n<setup_script>\nx=1\n</setup_script>\n<validation_script>\nassert x==2\n</validation_script>"}}]
        }
    context.llm_client.chat_completion = AsyncMock(side_effect=mock_generate_challenge)
    dreamer = Dreamer(context)
    
    def fake_execute(cmd, *a, **k):
        if "validator.py" in cmd and "py_compile" not in cmd:
            return ("Assertion", 1)
        return ("OK", 0)
        
    with patch("ghost_agent.sandbox.docker.DockerSandbox.execute", side_effect=fake_execute):
        with patch("pathlib.Path.write_text"):
            with patch("ghost_agent.core.agent.GhostAgent.handle_chat", AsyncMock()) as mock_handle_chat:
    
                async def fake_handle_chat(body, *a, **k):
                    if any("SYSTEM JUDGE REJECTION" in str(m.get("content", "")) for m in body.get("messages", [])):
                        raise KeyboardInterrupt("Stop loop")
                    return ("Output", False, False)
                mock_handle_chat.side_effect = fake_handle_chat
    
                try:
                    await dreamer.synthetic_self_play()
                except KeyboardInterrupt:
                    pass
    
                for call in mock_handle_chat.call_args_list:
                    messages = call[0][0].get("messages", [])
                    for msg in messages:
                        if "SYSTEM JUDGE REJECTION" in str(msg.get("content", "")):
                            content = msg["content"]
                            # The feedback (validator's FAIL output / diff)
                            # must reach the agent — it's the signal it
                            # reasons from on retry.
                            assert "Validator feedback" in content
                            assert "Assertion" in content
                            # The `.validator.py` source must NOT be leaked.
                            assert "assert x==2" not in content
                            assert "hidden `.validator.py`" not in content
                            assert "```python" not in content
                            return
                pytest.fail("Did not find rejection feedback")
