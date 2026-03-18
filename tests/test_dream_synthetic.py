import pytest
import asyncio
import json
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

from ghost_agent.core.dream import Dreamer

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
async def test_synthetic_self_play_memory_proxies(mock_ghost_agent_class, mock_docker_sandbox_class, mock_context):
    """
    Test that the memory proxies (ReadOnlySkillMemory, ReadOnlyVectorMemory)
    correctly forward read operations and block write operations.
    """
    dreamer = Dreamer(mock_context)
    
    # Mock LLM to return challenge and validation script
    llm_payload = {
        "challenge_prompt": "Write a python script",
        "validation_script": "assert True"
    }
    mock_context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": json.dumps(llm_payload)}}]
    })
    
    # We want to capture the isolated_context passed to GhostAgent
    mock_agent_instance = MagicMock()
    mock_agent_instance.handle_chat = AsyncMock(return_value=("Code generated", None, None))
    mock_agent_instance._get_recent_transcript.return_value = "Mock transcript"
    mock_ghost_agent_class.return_value = mock_agent_instance
    
    # Mock DockerSandbox execute to succeed
    mock_sandbox_instance = MagicMock()
    mock_sandbox_instance.execute.return_value = ("Success", 0)
    mock_docker_sandbox_class.return_value = mock_sandbox_instance
    
    # Run synthetic_self_play
    result = await dreamer.synthetic_self_play("test-model")
    
    assert "SUCCESS" in result
    
    # Get the isolated_context passed to GhostAgent
    mock_ghost_agent_class.assert_called_once()
    isolated_context = mock_ghost_agent_class.call_args[0][0]
    
    # Test ReadOnlyVectorMemory
    isolated_vm = isolated_context.memory_system
    # Forwarded methods
    isolated_vm.search("q")
    mock_context.memory_system.search.assert_called_with("q")
    isolated_vm.search_advanced("q")
    mock_context.memory_system.search_advanced.assert_called_with("q")
    
    # Blocked methods
    isolated_vm.add("fact")
    mock_context.memory_system.add.assert_not_called()
    isolated_vm.smart_update("fact")
    mock_context.memory_system.smart_update.assert_not_called()
    isolated_vm.delete("id")
    mock_context.memory_system.delete.assert_not_called()
    
    # Test ReadOnlySkillMemory
    isolated_sm = isolated_context.skill_memory
    # Forwarded methods
    isolated_sm.get_playbook_context()
    mock_context.skill_memory.get_playbook_context.assert_called_once()
    
    # Blocked methods
    isolated_sm.learn_lesson("task", "mistake", "solution")
    mock_context.skill_memory.learn_lesson.assert_not_called()
    isolated_sm.save_playbook([])
    mock_context.skill_memory.save_playbook.assert_not_called()

@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_synthetic_self_play_tool_bleed_mitigation(mock_ghost_agent_class, mock_docker_sandbox_class, mock_context):
    dreamer = Dreamer(mock_context)
    
    llm_payload = {"challenge_prompt": "C", "validation_script": "V"}
    mock_context.llm_client.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": json.dumps(llm_payload)}}]})
    
    mock_agent_instance = MagicMock()
    mock_agent_instance.available_tools = {
        "manage_tasks": {}, "postgres_admin": {}, "update_profile": {},
        "learn_skill": {}, "delegate_to_swarm": {}, "system_utility": {},
        "self_play": {}, "file_system": {}
    }
    mock_agent_instance.handle_chat = AsyncMock(return_value=("Code", None, None))
    mock_agent_instance._get_recent_transcript.return_value = "Trace"
    mock_ghost_agent_class.return_value = mock_agent_instance
    
    mock_sandbox_instance = MagicMock()
    mock_sandbox_instance.execute.return_value = ("Success", 0)
    mock_docker_sandbox_class.return_value = mock_sandbox_instance
    
    await dreamer.synthetic_self_play("test-model")
    
    # Check that dangerous tools were popped
    popped_tools = ["manage_tasks", "postgres_admin", "update_profile", "learn_skill", "delegate_to_swarm", "system_utility", "self_play"]
    for tool in popped_tools:
        assert tool not in mock_agent_instance.available_tools
    
    # Safe tool should remain
    assert "file_system" in mock_agent_instance.available_tools

@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_synthetic_self_play_judge_trace_and_tdd(mock_ghost_agent_class, mock_docker_sandbox_class, mock_context):
    dreamer = Dreamer(mock_context)
    
    llm_payload = {"challenge_prompt": "Task", "validation_script": "assert 1 == 1"}
    mock_context.llm_client.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": json.dumps(llm_payload)}}]})
    
    mock_agent_instance = MagicMock()
    mock_agent_instance.handle_chat = AsyncMock(return_value=("Failure Code", None, None))
    mock_agent_instance._get_recent_transcript.return_value = "Detailed execution transcript limit"
    mock_ghost_agent_class.return_value = mock_agent_instance
    
    mock_sandbox_instance = MagicMock()
    # It fails on the first 4 attempts, succeeds on the 5th
    execute_responses = [("Fail 1", 1), ("Fail 2", 1), ("Fail 3", 1), ("Fail 4", 1), ("Success", 0)]
    mock_sandbox_instance.execute.side_effect = execute_responses
    mock_docker_sandbox_class.return_value = mock_sandbox_instance
    
    result = await dreamer.synthetic_self_play("test-model")
    
    assert "SUCCESS" in result
    
    # Check that _get_recent_transcript was called
    assert mock_agent_instance._get_recent_transcript.call_count == 6 # 5 attempts + 1 for post-mortem
    
    # Check that validation script was written (via checking to_thread calls implicitly or patching)
    # It's tested indirectly by the logic flowing successfully through TDD
    
@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_synthetic_self_play_dynamic_postmortem(mock_ghost_agent_class, mock_docker_sandbox_class, mock_context):
    dreamer = Dreamer(mock_context)
    
    llm_payload = {"challenge_prompt": "C", "validation_script": "V"}
    
    # First LLM call is challenge generation. Second is post-mortem.
    # We will simulate a perfect first-try completion.
    async def chat_mock(*args, **kwargs):
        messages = args[0]["messages"]
        system_content = messages[0]["content"] if messages else ""
        if "AI training coordinator" in system_content:
            return {"choices": [{"message": {"content": json.dumps(llm_payload)}}]}
        elif "Meta-Cognitive Analyst" in system_content:
            user_content = messages[1]["content"]
            # Assert that the dynamic prompt asked for Best Practice because to passed on attempt 0
            assert "effortlessly solved a simulated challenge on the first try" in user_content
            assert "'Best Practice' or 'Optimization'" in user_content
            return {"choices": [{"message": {"content": json.dumps({"task": "T", "mistake": "", "solution": "S"})}}]}
        return {}
    
    mock_context.llm_client.chat_completion = AsyncMock(side_effect=chat_mock)
    
    mock_agent_instance = MagicMock()
    mock_agent_instance.handle_chat = AsyncMock(return_value=("Code", None, None))
    mock_agent_instance._get_recent_transcript.return_value = "Trace"
    mock_ghost_agent_class.return_value = mock_agent_instance
    
    mock_sandbox_instance = MagicMock()
    mock_sandbox_instance.execute.return_value = ("Success", 0)  # Pass on attempt 0
    mock_docker_sandbox_class.return_value = mock_sandbox_instance
    
    await dreamer.synthetic_self_play("test-model")
    
    mock_context.skill_memory.learn_lesson.assert_called_once()
    
    # Now simulate failure on all attempts
    mock_context.llm_client.chat_completion.reset_mock()
    mock_context.skill_memory.learn_lesson.reset_mock()
    
    async def chat_mock_fail(*args, **kwargs):
        messages = args[0]["messages"]
        system_content = messages[0]["content"] if messages else ""
        if "AI training coordinator" in system_content:
            return {"choices": [{"message": {"content": json.dumps(llm_payload)}}]}
        elif "Meta-Cognitive Analyst" in system_content:
            user_content = messages[1]["content"]
            # Assert that the dynamic prompt asked for core error because it failed
            assert "core technical error or strategy flaw" in user_content
            return {"choices": [{"message": {"content": json.dumps({"task": "T", "mistake": "M", "solution": "S"})}}]}
        return {}
        
    mock_context.llm_client.chat_completion = AsyncMock(side_effect=chat_mock_fail)
    mock_sandbox_instance.execute.side_effect = None
    mock_sandbox_instance.execute.return_value = ("Failure", 1)  # Fail every attempt
    
    await dreamer.synthetic_self_play("test-model")
    
    mock_context.skill_memory.learn_lesson.assert_called_once()


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_synthetic_self_play_regex_fallback(mock_ghost_agent_class, mock_docker_sandbox_class, mock_context):
    """
    Test that if the LLM outputs malformed JSON (e.g., unescaped nested quotes), 
    the regex fallback correctly extracts the challenge_prompt and validation_script.
    """
    dreamer = Dreamer(mock_context)
    
    # Malformed JSON with unescaped nested quotes
    malformed_llm_payload = """
    {
      "setup_script": "",
      "challenge_prompt": "Write a python script that prints "hello".",
      "validation_script": "import subprocess\\nassert sum([1, 2]) == 3"
    }
    """
    
    # Mock LLM chat_completion to return the malformed string directly
    mock_context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": malformed_llm_payload}}]
    })
    
    mock_agent_instance = MagicMock()
    mock_agent_instance.handle_chat = AsyncMock(return_value=("Code generated", None, None))
    mock_agent_instance._get_recent_transcript.return_value = "Mock transcript"
    mock_ghost_agent_class.return_value = mock_agent_instance
    
    mock_sandbox_instance = MagicMock()
    mock_sandbox_instance.execute.return_value = ("Success", 0)
    mock_docker_sandbox_class.return_value = mock_sandbox_instance
    
    result = await dreamer.synthetic_self_play("test-model")
    
    # If the fallback failed, the result would be "Failed to extract challenge..."
    assert "SUCCESS" in result

@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_synthetic_self_play_scratchpad_report(mock_ghost_agent_class, mock_docker_sandbox_class, mock_context):
    """
    Test that synthetic_self_play writes a summary report to the agent's scratchpad 
    upon completion so that `scratch_list` can read it.
    """
    dreamer = Dreamer(mock_context)
    
    llm_payload = {
        "challenge_prompt": "Test scratchpad",
        "validation_script": "assert True"
    }
    
    # Needs two LLM calls: one for challenge, one for learning extraction
    async def chat_mock(*args, **kwargs):
        messages = args[0]["messages"]
        system_content = messages[0]["content"] if messages else ""
        if "AI training coordinator" in system_content:
            return {"choices": [{"message": {"content": json.dumps(llm_payload)}}]}
        elif "Meta-Cognitive Analyst" in system_content:
            return {"choices": [{"message": {"content": json.dumps({"task": "Test task", "mistake": "None", "solution": "Test solution"})}}]}
        return {}
        
    mock_context.llm_client.chat_completion = AsyncMock(side_effect=chat_mock)
    
    # Explicitly add a mock scratchpad to context
    mock_context.scratchpad = MagicMock()
    mock_context.scratchpad.get.return_value = "Challenge: Test scratchpad\nStatus: SUCCESS\nLearned task: Test task"
    
    mock_agent_instance = MagicMock()
    mock_agent_instance.handle_chat = AsyncMock(return_value=("Code generated", None, None))
    mock_agent_instance._get_recent_transcript.return_value = "Mock transcript"
    mock_ghost_agent_class.return_value = mock_agent_instance
    
    mock_sandbox_instance = MagicMock()
    mock_sandbox_instance.execute.return_value = ("Success", 0)
    mock_docker_sandbox_class.return_value = mock_sandbox_instance
    
    result = await dreamer.synthetic_self_play("test-model")
    
    # Verify that scratchpad.set was called with the report
    mock_context.scratchpad.set.assert_called_once()
    args, _ = mock_context.scratchpad.set.call_args
    assert args[0] == "Self-Play Report"
    assert "Challenge: Test scratchpad" in args[1]
    assert "SUCCESS" in args[1]
    assert "Learned task: Test task" in args[1]
    
    # Verify the report is also returned in the final string to prevent the LLM from getting stuck
    assert "SELF-PLAY POST-MORTEM REPORT:" in result
    assert "Challenge: Test scratchpad" in result
