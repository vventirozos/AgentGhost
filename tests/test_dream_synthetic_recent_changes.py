import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch

from ghost_agent.core.dream import Dreamer

def dict_to_xml(d):
    res = ""
    for k, v in d.items():
        res += f"<{k}>{v}</{k}>\n"
    return res

@pytest.fixture
def mock_context():
    context = MagicMock()
    context.memory_system = MagicMock()
    context.skill_memory = MagicMock()
    context.skill_memory.get_recent_failures.return_value = "No failures"
    
    context.llm_client = MagicMock()
    context.args = MagicMock()
    context.args.perfect_it = True
    context.args.smart_memory = 1.0
    context.sandbox_manager = MagicMock()
    context.sandbox_dir = "/tmp/mock"
    context.tor_proxy = None
    return context

@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_synthetic_self_play_final_instruction_appended(mock_ghost_agent_class, mock_docker_sandbox_class, mock_context):
    """
    Test that synthetic_self_play appends the system instruction
    warning the LLM to ignore the temporary mock files post-simulation.
    """
    dreamer = Dreamer(mock_context)
    
    llm_payload = {
        "challenge_prompt": "Test prompt",
        "validation_script": "assert True"
    }
    
    mock_context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": dict_to_xml(llm_payload)}}]
    })
    
    mock_agent_instance = MagicMock()
    mock_agent_instance.handle_chat = AsyncMock(return_value=("Code generated", None, None))
    mock_agent_instance._get_recent_transcript.return_value = "Mock transcript"
    mock_ghost_agent_class.return_value = mock_agent_instance
    
    mock_sandbox_instance = MagicMock()
    mock_sandbox_instance.execute.return_value = ("Success", 0)
    mock_docker_sandbox_class.return_value = mock_sandbox_instance
    
    result = await dreamer.synthetic_self_play("test-model")
    
    assert "SYSTEM INSTRUCTION: The self-play simulation took place in a temporary, isolated sandbox" in result
    assert "permanently destroyed" in result
    assert "DO NOT attempt to find, run, or execute 'solution.py'" in result


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_synthetic_self_play_setup_script_runs_once(mock_ghost_agent_class, mock_docker_sandbox_class, mock_context):
    """
    Verifies that the setup script is executed only ONCE during sandbox creation,
    and is NOT redundantly re-run right before validation executing.
    """
    dreamer = Dreamer(mock_context)
    
    llm_payload = {
        "setup_script": "print('hello mock data')",
        "challenge_prompt": "Read mock data",
        "validation_script": "assert True"
    }
    
    mock_context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": dict_to_xml(llm_payload)}}]
    })
    
    mock_agent_instance = MagicMock()
    mock_agent_instance.handle_chat = AsyncMock(return_value=("Code generated", None, None))
    mock_agent_instance._get_recent_transcript.return_value = "Mock transcript"
    mock_ghost_agent_class.return_value = mock_agent_instance
    
    mock_sandbox_instance = MagicMock()
    
    def side_effect(cmd, *args, **kwargs):
        if "python3 .setup.py" in cmd:
            return ("Setup success output", 0)
        else:
            return ("Validator success output", 0)
    
    mock_sandbox_instance.execute.side_effect = side_effect
    mock_docker_sandbox_class.return_value = mock_sandbox_instance
    
    result = await dreamer.synthetic_self_play("test-model")
    
    assert "SUCCESS" in result
    
    execute_calls = mock_sandbox_instance.execute.call_args_list
    setup_script_calls = [call for call in execute_calls if "python3 .setup.py" in call[0][0]]
    
    # Assert it was only called exactly once (not twice due to the removal)
    assert len(setup_script_calls) == 1
