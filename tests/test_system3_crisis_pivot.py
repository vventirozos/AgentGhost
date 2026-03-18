"""
Tests for the System 3 "Crisis Pivot" feature in GhostAgent.

Covers:
  1. Prompt constants exist and are well-formed.
  2. _run_system_3_pivot happy-path: returns tree_update + justification.
  3. _run_system_3_pivot graceful failure: returns {} on any exception.
  4. _run_system_3_pivot graceful failure: returns {} when Generator returns no strategies.
  5. _run_system_3_pivot uses swarm when available.
  6. handle_chat interception: triggered at exactly execution_failure_count == 2.
  7. handle_chat interception: resets failure counter so agent gets a fresh attempt.
  8. handle_chat interception: injects the SYSTEM 3 PIVOT message into context.
  9. handle_chat interception: does NOT trigger on first failure (count == 1).
  10. handle_chat interception: skips pivot and falls to hard-limit when pivot returns {}.
"""

import copy
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from ghost_agent.core.prompts import (
    SYSTEM_3_GENERATION_PROMPT,
    SYSTEM_3_EVALUATOR_PROMPT,
)
from ghost_agent.core.agent import GhostAgent, GhostContext


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

class MockArgs:
    temperature = 0.7
    max_context = 8000
    smart_memory = 0.0
    use_planning = False
    perfect_it = False


def _make_agent(llm_responses=None):
    """Build a GhostAgent with a fully-mocked context."""
    ctx = MagicMock(spec=GhostContext)
    ctx.args = MockArgs()
    ctx.llm_client = MagicMock()
    ctx.llm_client.swarm_clients = None  # no swarm by default
    ctx.llm_client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": "ok", "tool_calls": []}}]}
    )
    if llm_responses is not None:
        ctx.llm_client.chat_completion = AsyncMock(side_effect=llm_responses)

    ctx.profile_memory = MagicMock()
    ctx.profile_memory.get_context_string = MagicMock(return_value="")
    ctx.skill_memory = MagicMock()
    ctx.skill_memory.get_playbook_context = MagicMock(return_value="")
    ctx.memory_system = MagicMock()
    ctx.memory_system.search = MagicMock(return_value="")
    ctx.cached_sandbox_state = None
    ctx.sandbox_dir = "/tmp/sandbox"
    ctx.scratchpad = MagicMock()
    ctx.scratchpad.list_all = MagicMock(return_value="None.")

    return GhostAgent(ctx)


def _llm_ok(content="ok"):
    return {"choices": [{"message": {"content": content, "tool_calls": []}}]}


def _execute_tool_call(exit_code=1, stderr="SyntaxError: invalid syntax"):
    """Build a sequence: assistant requests execute → tool returns failure."""
    return {
        "choices": [{
            "message": {
                "content": None,
                "tool_calls": [{
                    "id": "call_exec",
                    "type": "function",
                    "function": {
                        "name": "execute",
                        "arguments": json.dumps({"content": "bad code(", "language": "python"})
                    }
                }]
            }
        }]
    }


EXEC_FAIL_RESULT = f"EXIT CODE: 1\nSTDOUT/STDERR: SyntaxError: invalid syntax"


# ---------------------------------------------------------------------------
# 1. Prompt constants
# ---------------------------------------------------------------------------

class TestSystem3Prompts:
    def test_generation_prompt_exists_and_has_schema(self):
        assert SYSTEM_3_GENERATION_PROMPT, "SYSTEM_3_GENERATION_PROMPT is empty"
        assert "strategies" in SYSTEM_3_GENERATION_PROMPT
        assert "IDENTITY" in SYSTEM_3_GENERATION_PROMPT

    def test_evaluator_prompt_exists_and_has_schema(self):
        assert SYSTEM_3_EVALUATOR_PROMPT, "SYSTEM_3_EVALUATOR_PROMPT is empty"
        assert "tree_update" in SYSTEM_3_EVALUATOR_PROMPT
        assert "justification" in SYSTEM_3_EVALUATOR_PROMPT
        assert "winning_id" in SYSTEM_3_EVALUATOR_PROMPT

    def test_generation_prompt_specifies_three_approaches(self):
        assert "Approach A" in SYSTEM_3_GENERATION_PROMPT
        assert "Approach B" in SYSTEM_3_GENERATION_PROMPT
        assert "Approach C" in SYSTEM_3_GENERATION_PROMPT

    def test_evaluator_prompt_mentions_safety(self):
        # The evaluator should be mindful of loops/sandbox safety
        prompt_lower = SYSTEM_3_EVALUATOR_PROMPT.lower()
        assert "loop" in prompt_lower or "risk" in prompt_lower or "safe" in prompt_lower


# ---------------------------------------------------------------------------
# 2. _run_system_3_pivot – happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pivot_happy_path_returns_tree_update_and_justification():
    agent = _make_agent()

    gen_response = json.dumps({
        "strategies": [
            {"id": "A", "description": "Direct", "steps": ["step1"]},
            {"id": "B", "description": "Defensive", "steps": ["step1"]},
            {"id": "C", "description": "Creative", "steps": ["step1"]},
        ]
    })
    eval_response = json.dumps({
        "winning_id": "B",
        "justification": "Least likely to loop",
        "tree_update": {
            "id": "root",
            "description": "Recover from failure",
            "status": "IN_PROGRESS",
            "children": [{"id": "task_1", "description": "Try approach B", "status": "READY"}]
        }
    })

    agent.context.llm_client.chat_completion = AsyncMock(
        side_effect=[_llm_ok(gen_response), _llm_ok(eval_response)]
    )

    result = await agent._run_system_3_pivot(
        task_context="write a script",
        error_context="SyntaxError: bad code",
        sandbox_state="empty sandbox",
        model="test-model"
    )

    assert "tree_update" in result
    assert result["tree_update"]["id"] == "root"
    assert result["justification"] == "Least likely to loop"
    assert result["winning_id"] == "B"


# ---------------------------------------------------------------------------
# 3. _run_system_3_pivot – exception → returns {}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pivot_returns_empty_dict_on_llm_exception():
    agent = _make_agent()
    agent.context.llm_client.chat_completion = AsyncMock(
        side_effect=Exception("upstream down")
    )

    result = await agent._run_system_3_pivot(
        task_context="task", error_context="error", sandbox_state="state", model="model"
    )

    assert result == {}


# ---------------------------------------------------------------------------
# 4. _run_system_3_pivot – no strategies → returns {}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pivot_returns_empty_dict_when_generator_has_no_strategies():
    agent = _make_agent()

    # Generator returns JSON with empty strategies list
    agent.context.llm_client.chat_completion = AsyncMock(
        return_value=_llm_ok(json.dumps({"strategies": []}))
    )

    result = await agent._run_system_3_pivot(
        task_context="task", error_context="error", sandbox_state="state", model="model"
    )

    assert result == {}
    # Should only have called the LLM once (aborted before evaluator)
    agent.context.llm_client.chat_completion.assert_called_once()


# ---------------------------------------------------------------------------
# 5. _run_system_3_pivot – uses swarm when available
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pivot_uses_swarm_when_available():
    agent = _make_agent()
    agent.context.llm_client.swarm_clients = ["worker1"]  # swarm exists

    gen_response = json.dumps({
        "strategies": [{"id": "A", "description": "D", "steps": ["s"]}]
    })
    eval_response = json.dumps({
        "winning_id": "A",
        "justification": "best",
        "tree_update": {
            "id": "root", "description": "obj", "status": "IN_PROGRESS",
            "children": [{"id": "t1", "description": "step", "status": "READY"}]
        }
    })
    agent.context.llm_client.chat_completion = AsyncMock(
        side_effect=[_llm_ok(gen_response), _llm_ok(eval_response)]
    )

    await agent._run_system_3_pivot(
        task_context="t", error_context="e", sandbox_state="s", model="m"
    )

    calls = agent.context.llm_client.chat_completion.call_args_list
    assert all(c.kwargs.get("use_swarm") is True for c in calls), \
        "Both Generator and Evaluator calls should use use_swarm=True"


# ---------------------------------------------------------------------------
# 6. handle_chat: interception triggers at count == 2 and pivots the plan
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_chat_triggers_pivot_at_second_failure():
    """
    Scenario: execute tool fails twice → on the 2nd failure the pivot fires,
    returns a valid tree_update, and the agent continues with the new plan.
    """
    tree_update_json = {
        "id": "root",
        "description": "Recover task",
        "status": "IN_PROGRESS",
        "children": [{"id": "task_1", "description": "Try new approach", "status": "READY"}]
    }
    pivot_result = {
        "winning_id": "B",
        "justification": "Safer path chosen",
        "tree_update": tree_update_json,
    }

    exec_call = _execute_tool_call()
    final_answer = _llm_ok("Done!")

    # Turn 1: LLM requests execute → fail
    # Turn 2: LLM requests execute → fail (triggers pivot)
    # After pivot: LLM answers directly
    llm_sequence = [exec_call, exec_call, final_answer]

    agent = _make_agent(llm_responses=llm_sequence)
    # Register a mock execute tool that returns failure
    agent.available_tools["execute"] = AsyncMock(return_value=EXEC_FAIL_RESULT)

    body = {
        "messages": [{"role": "user", "content": "run my script"}],
        "model": "test-model"
    }

    with patch.object(agent, "_run_system_3_pivot", new=AsyncMock(return_value=pivot_result)) as mock_pivot, \
         patch("ghost_agent.core.agent.pretty_log"):

        result_content, _, _ = await agent.handle_chat(body, background_tasks=MagicMock())

    mock_pivot.assert_called_once()
    # Verify the pivot was called with the right signatures
    call_kwargs = mock_pivot.call_args.kwargs
    assert "error_context" in call_kwargs
    assert "task_context" in call_kwargs
    assert "sandbox_state" in call_kwargs
    assert call_kwargs["model"] == "test-model"


# ---------------------------------------------------------------------------
# 7. handle_chat: failure counter resets after successful pivot
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_chat_pivot_resets_failure_count():
    """
    After a successful pivot the execution_failure_count must be 0 so the
    agent gets a fresh 3-strike budget on the new strategy.
    The hard-limit (3 strikes) should NOT be hit in the same session.
    """
    tree_update_json = {
        "id": "root", "description": "Obj", "status": "IN_PROGRESS",
        "children": [{"id": "t1", "description": "step", "status": "READY"}]
    }
    pivot_result = {
        "winning_id": "A", "justification": "ok",
        "tree_update": tree_update_json,
    }

    exec_call = _execute_tool_call()
    final_answer = _llm_ok("All done!")

    # Two failures → pivot → final answer
    llm_sequence = [exec_call, exec_call, final_answer]

    agent = _make_agent(llm_responses=llm_sequence)
    agent.available_tools["execute"] = AsyncMock(return_value=EXEC_FAIL_RESULT)

    body = {
        "messages": [{"role": "user", "content": "run script"}],
        "model": "test-model"
    }

    hard_limit_messages = []
    original_chat_completion = agent.context.llm_client.chat_completion

    with patch.object(agent, "_run_system_3_pivot", new=AsyncMock(return_value=pivot_result)), \
         patch("ghost_agent.core.agent.pretty_log"):
        result_content, _, _ = await agent.handle_chat(body, background_tasks=MagicMock())

    # The hard-limit message must NOT appear in the final response
    assert "failed 3 times" not in result_content.lower(), \
        "Hard limit was triggered even though pivot should have reset the counter"


# ---------------------------------------------------------------------------
# 8. handle_chat: SYSTEM 3 PIVOT message is injected into context
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_chat_pivot_injects_user_message():
    """
    The pivot must inject a `role: user` message containing 'SYSTEM 3 PIVOT'
    and the justification into the message history before continuing.
    """
    tree_update_json = {
        "id": "root", "description": "Obj", "status": "IN_PROGRESS",
        "children": [{"id": "t1", "description": "step", "status": "READY"}]
    }
    pivot_result = {
        "winning_id": "A",
        "justification": "My unique justification string XYZ",
        "tree_update": tree_update_json,
    }

    exec_call = _execute_tool_call()
    captured_messages = []

    async def capturing_completion(payload, **kwargs):
        captured_messages.extend(copy.deepcopy(payload.get("messages", [])))
        # Return exec call for first two, then final answer
        if len([m for m in captured_messages if m.get("role") == "user" and "SYSTEM 3 PIVOT" in str(m.get("content", ""))]) > 0:
            return _llm_ok("Fixed!")
        return exec_call

    agent = _make_agent()
    agent.context.llm_client.chat_completion = AsyncMock(
        side_effect=[exec_call, exec_call, _llm_ok("Fixed!")]
    )
    agent.available_tools["execute"] = AsyncMock(return_value=EXEC_FAIL_RESULT)

    body = {
        "messages": [{"role": "user", "content": "do task"}],
        "model": "test-model"
    }

    injected_pivot_msgs = []

    async def mock_llm(payload, **kwargs):
        for m in payload.get("messages", []):
            if "SYSTEM 3 PIVOT" in str(m.get("content", "")):
                injected_pivot_msgs.append(m)
        # Return exec call twice, then final
        call_count = mock_llm.call_count
        mock_llm.call_count += 1
        if call_count < 2:
            return exec_call
        return _llm_ok("Done")

    mock_llm.call_count = 0
    agent.context.llm_client.chat_completion = mock_llm

    with patch.object(agent, "_run_system_3_pivot", new=AsyncMock(return_value=pivot_result)), \
         patch("ghost_agent.core.agent.pretty_log"):
        await agent.handle_chat(body, background_tasks=MagicMock())

    assert len(injected_pivot_msgs) > 0, "SYSTEM 3 PIVOT message was never injected"
    pivot_msg = injected_pivot_msgs[0]
    assert pivot_msg["role"] == "user"
    assert "My unique justification string XYZ" in pivot_msg["content"]


# ---------------------------------------------------------------------------
# 9. handle_chat: pivot does NOT fire on first failure (count == 1)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_chat_pivot_not_triggered_on_first_failure():
    """
    The pivot should only fire at count == 2, not at count == 1.
    """
    exec_call = _execute_tool_call()
    final_answer = _llm_ok("Recovered!")

    # Only one failure, then success
    llm_sequence = [exec_call, final_answer]

    agent = _make_agent(llm_responses=llm_sequence)
    # First call: fail; second call: success (exit code 0)
    agent.available_tools["execute"] = AsyncMock(
        side_effect=[EXEC_FAIL_RESULT, "EXIT CODE: 0\nSTDOUT/STDERR: success"]
    )

    body = {
        "messages": [{"role": "user", "content": "run script"}],
        "model": "test-model"
    }

    with patch.object(agent, "_run_system_3_pivot", new=AsyncMock(return_value={})) as mock_pivot, \
         patch("ghost_agent.core.agent.pretty_log"):
        await agent.handle_chat(body, background_tasks=MagicMock())

    mock_pivot.assert_not_called()


# ---------------------------------------------------------------------------
# 10. handle_chat: falls to hard-limit when pivot returns {}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_chat_falls_to_hard_limit_when_pivot_empty():
    """
    If _run_system_3_pivot returns {}, the interception block must not
    alter the counter or history, so the normal 3-strike hard limit fires
    on the 3rd failure.
    """
    exec_call = _execute_tool_call()
    final_answer = _llm_ok("Giving up.")

    llm_sequence = [exec_call, exec_call, exec_call, final_answer]

    agent = _make_agent(llm_responses=llm_sequence)
    agent.available_tools["execute"] = AsyncMock(return_value=EXEC_FAIL_RESULT)

    body = {
        "messages": [{"role": "user", "content": "run forever"}],
        "model": "test-model"
    }

    hard_limit_injected = []

    async def mock_llm(payload, **kwargs):
        for m in payload.get("messages", []):
            if "failed 3 times in a row" in str(m.get("content", "")).lower():
                hard_limit_injected.append(m)
        call_idx = mock_llm.call_count
        mock_llm.call_count += 1
        if call_idx < 3:
            return exec_call
        return final_answer

    mock_llm.call_count = 0
    agent.context.llm_client.chat_completion = mock_llm

    with patch.object(agent, "_run_system_3_pivot", new=AsyncMock(return_value={})), \
         patch("ghost_agent.core.agent.pretty_log"):
        await agent.handle_chat(body, background_tasks=MagicMock())

    assert len(hard_limit_injected) > 0, \
        "Hard-limit alert was not injected after pivot returned empty"
