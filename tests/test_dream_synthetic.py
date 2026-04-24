import pytest
import asyncio
import json
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

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
        "choices": [{"message": {"content": dict_to_xml(llm_payload)}}]
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
@pytest.mark.skip(reason="Tool bleed mitigation depends on removed logic")
async def test_synthetic_self_play_tool_bleed_mitigation(mock_ghost_agent_class, mock_docker_sandbox_class, mock_context):
    dreamer = Dreamer(mock_context)
    
    llm_payload = {"challenge_prompt": "C", "validation_script": "V"}
    mock_context.llm_client.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": dict_to_xml(llm_payload)}}]})
    
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
async def test_synthetic_self_play_judge_trace_and_tdd(mock_ghost_agent_class, mock_docker_sandbox_class, mock_context, disable_self_play_templates):
    dreamer = Dreamer(mock_context)
    
    llm_payload = {"challenge_prompt": "Task", "validation_script": "assert 1 == 1"}
    mock_context.llm_client.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": dict_to_xml(llm_payload)}}]})
    
    mock_agent_instance = MagicMock()
    mock_agent_instance.handle_chat = AsyncMock(return_value=("Failure Code", None, None))
    mock_agent_instance._get_recent_transcript.return_value = "Detailed execution transcript limit"
    mock_ghost_agent_class.return_value = mock_agent_instance
    
    mock_sandbox_instance = MagicMock()
    # It fails on the first 2 attempts, succeeds on the 3rd. Ignore the
    # setup/validator pre-flight shell commands — those are infrastructure
    # checks, not part of the attempt-count semantics this test models.
    def execute_side_effect(cmd, *args, **kwargs):
        if "py_compile" in cmd: return ("Syntax OK", 0)
        if ".preflight.py" in cmd: return ("Pre-flight OK", 0)
        # Stateful failure counting:
        if not hasattr(execute_side_effect, "calls"): execute_side_effect.calls = 0
        execute_side_effect.calls += 1
        if execute_side_effect.calls <= 2: return ("Fail", 1)
        return ("Success", 0)
        
    mock_sandbox_instance.execute.side_effect = execute_side_effect
    mock_docker_sandbox_class.return_value = mock_sandbox_instance
    
    result = await dreamer.synthetic_self_play("test-model")
    
    assert "SUCCESS" in result
    
    assert mock_agent_instance._get_recent_transcript.call_count == 3  # one capture per attempt (duplicate capture removed)
    
    # Check that validation script was written (via checking to_thread calls implicitly or patching)
    # It's tested indirectly by the logic flowing successfully through TDD
    
@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_synthetic_self_play_dynamic_postmortem(mock_ghost_agent_class, mock_docker_sandbox_class, mock_context, disable_self_play_templates):
    dreamer = Dreamer(mock_context)
    
    llm_payload = {"challenge_prompt": "C", "validation_script": "V"}
    
    # First LLM call is challenge generation. Second is post-mortem.
    # We will simulate a perfect first-try completion.
    async def chat_mock(*args, **kwargs):
        messages = args[0]["messages"]
        system_content = messages[0]["content"] if messages else ""
        if "AI training coordinator" in system_content:
            return {"choices": [{"message": {"content": dict_to_xml(llm_payload)}}]}
        elif "Meta-Cognitive Analyst" in system_content:
            user_content = messages[1]["content"]
            # Redesigned prompt tags the outcome explicitly so the
            # downstream extractor can route on it.
            assert "Outcome: FIRST_TRY_SUCCESS" in user_content
            assert "CORRECT-PATTERN" in user_content or "correct_pattern" in user_content
            return {"choices": [{"message": {"content": json.dumps({
                "task": "T", "mistake": "", "solution": "def fix():\n    pass",
                "trigger": "T", "anti_pattern": "", "correct_pattern": "def fix():\n    pass",
                "domains": ["python_general"], "confidence": 0.6,
            })}}]}
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
            return {"choices": [{"message": {"content": dict_to_xml(llm_payload)}}]}
        elif "Meta-Cognitive Analyst" in system_content:
            user_content = messages[1]["content"]
            # Failure-outcome tag replaces the old "core technical error"
            # prose — the structured extractor keys off the Outcome line.
            assert "Outcome: FAILED" in user_content
            return {"choices": [{"message": {"content": json.dumps({
                "task": "T", "mistake": "M", "solution": "def fix():\n    pass",
                "trigger": "T", "anti_pattern": "M", "correct_pattern": "def fix():\n    pass",
                "domains": ["python_general"], "confidence": 0.5,
            })}}]}
        return {}
        
    mock_context.llm_client.chat_completion = AsyncMock(side_effect=chat_mock_fail)
    mock_context.llm_client.chat_completion = AsyncMock(side_effect=chat_mock_fail)
    
    def side_effect_fail_all(cmd, *args, **kwargs):
        if "py_compile" in cmd: return ("Syntax OK", 0)
        if ".preflight.py" in cmd: return ("Pre-flight OK", 0)
        return ("Failure", 1)
        
    mock_sandbox_instance.execute.side_effect = side_effect_fail_all
    
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
    <setup_script></setup_script>
    <challenge_prompt>Write a python script that prints "hello".</challenge_prompt>
    <validation_script>import subprocess\nassert sum([1, 2]) == 3</validation_script>
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
async def test_synthetic_self_play_scratchpad_report(mock_ghost_agent_class, mock_docker_sandbox_class, mock_context, disable_self_play_templates):
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
            return {"choices": [{"message": {"content": dict_to_xml(llm_payload)}}]}
        elif "Meta-Cognitive Analyst" in system_content:
            return {"choices": [{"message": {"content": json.dumps({
                "task": "Test task", "mistake": "None",
                "solution": "def ok():\n    return 1",
                "trigger": "Test task", "anti_pattern": "None",
                "correct_pattern": "def ok():\n    return 1",
                "domains": ["python_general"], "confidence": 0.7,
            })}}]}
        return {}

    mock_context.llm_client.chat_completion = AsyncMock(side_effect=chat_mock)

    # Explicitly add a mock scratchpad to context
    mock_context.scratchpad = MagicMock()
    mock_context.scratchpad.get.return_value = "Challenge: Test scratchpad\nStatus: SUCCESS\nLearned trigger: Test task"

    mock_agent_instance = MagicMock()
    mock_agent_instance.handle_chat = AsyncMock(return_value=("Code generated", None, None))
    mock_agent_instance._get_recent_transcript.return_value = "Mock transcript"
    mock_ghost_agent_class.return_value = mock_agent_instance

    mock_sandbox_instance = MagicMock()
    mock_sandbox_instance.execute.return_value = ("Success", 0)
    mock_docker_sandbox_class.return_value = mock_sandbox_instance

    result = await dreamer.synthetic_self_play("test-model")

    # Deep-review fix #9: the report must now land in the scratchpad on
    # EVERY path, including the success path. Previously this was only
    # written in the "no skill gate fired" else-branch, so successful
    # learning runs left the next turn with no breadcrumb.
    mock_context.scratchpad.set.assert_called()
    args, kwargs = mock_context.scratchpad.set.call_args
    assert args[0] == "Self-Play Report"
    assert "Challenge: Test scratchpad" in args[1]
    assert "Status: SUCCESS" in args[1]

    # Verify the report is ALSO generated locally and appended to the final string
    assert "SELF-PLAY POST-MORTEM REPORT:" in result
    assert "Challenge: Test scratchpad" in result
    assert "Status: SUCCESS" in result
    # Redesigned report uses "Learned trigger" instead of "Learned task".
    assert "Learned trigger: Test task" in result

@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_synthetic_self_play_setup_script_success(mock_ghost_agent_class, mock_docker_sandbox_class, mock_context):
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
    
    # We want execute to succeed for both .setup.py and .validator.py
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
    
    # Verify execute was called with .setup.py
    execute_calls = mock_sandbox_instance.execute.call_args_list
    assert any("python3 .setup.py" in call[0][0] for call in execute_calls)


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_synthetic_self_play_setup_script_failure(mock_ghost_agent_class, mock_docker_sandbox_class, mock_context):
    dreamer = Dreamer(mock_context)
    
    llm_payload = {
        "setup_script": "raise Exception('Bad setup')",
        "challenge_prompt": "Read mock data",
        "validation_script": "assert True"
    }
    
    mock_context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": dict_to_xml(llm_payload)}}]
    })
    
    mock_sandbox_instance = MagicMock()
    
    def side_effect(cmd, *args, **kwargs):
        if "python3 .setup.py" in cmd:
            return ("Setup traceback error", 1)  # Failing exit code
        else:
            return ("Success", 0)
            
    mock_sandbox_instance.execute.side_effect = side_effect
    mock_docker_sandbox_class.return_value = mock_sandbox_instance
    
    result = await dreamer.synthetic_self_play("test-model")
    
    assert "Synthetic challenge generation failed during setup script execution" in result
    assert "Setup traceback error" in result

@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_synthetic_self_play_regex_fallback_eof(mock_ghost_agent_class, mock_docker_sandbox_class, mock_context):
    """
    Test that the regex fallback handles truncated strings (EOF without closing brace/comma).
    """
    dreamer = Dreamer(mock_context)
    
    # Truncated payload missing closing brace and quote on validation_script
    truncated_llm_payload = """
    <setup_script>print('setup')</setup_script>
    <challenge_prompt>Solve this task</challenge_prompt>
    <validation_script>assert True"""
    
    mock_context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": truncated_llm_payload}}]
    })
    
    mock_agent_instance = MagicMock()
    mock_agent_instance.handle_chat = AsyncMock(return_value=("Code generated", None, None))
    mock_agent_instance._get_recent_transcript.return_value = "Mock transcript"
    mock_ghost_agent_class.return_value = mock_agent_instance
    
    mock_sandbox_instance = MagicMock()
    mock_sandbox_instance.execute.return_value = ("Success", 0)
    mock_docker_sandbox_class.return_value = mock_sandbox_instance
    
    result = await dreamer.synthetic_self_play("test-model")
    
    # Validation should succeed and not crash with "Failed to extract challenge..."
    assert "SUCCESS" in result
