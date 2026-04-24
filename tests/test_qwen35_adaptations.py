import pytest
import sys
import inspect
from unittest.mock import patch

def test_max_context_default():
    from ghost_agent.main import parse_args
    with patch.object(sys, 'argv', ['main.py']):
        args = parse_args()
        assert args.max_context == 65536, "Max context default is incorrect."

def test_prompt_updates():
    from ghost_agent.core.prompts import QWEN_TOOL_PROMPT, SYSTEM_PROMPT
    assert "<function name=\"function_name\">" in QWEN_TOOL_PROMPT, "QWEN_TOOL_PROMPT missing `<function name=\"function_name\">` xml syntax."
    assert "<parameter name=\"arg1\">" in QWEN_TOOL_PROMPT, "QWEN_TOOL_PROMPT missing `<parameter name=\"arg1\">` xml syntax."
    # The conversational persona was deliberately softened from "blunt /
    # aggressively efficient" to "neutral, friendly, helpful" — pin the
    # new wording so it can't silently regress.
    assert "neutral, friendly, and helpful" in SYSTEM_PROMPT, "SYSTEM_PROMPT missing softened conversational persona."
    assert "warm, conversational tone" in SYSTEM_PROMPT, "SYSTEM_PROMPT missing warm-tone instruction."
    assert "blunt, direct, and aggressively" not in SYSTEM_PROMPT, "Old aggressive persona must be removed."
    assert "natively multimodal and can physically see images" in SYSTEM_PROMPT, "SYSTEM_PROMPT missing multimodal vision capability."

def test_search_reddit_removed():
    from ghost_agent.tools.search import tool_search_ddgs, tool_deep_research
    ddgs_src = inspect.getsource(tool_search_ddgs)
    dr_src = inspect.getsource(tool_deep_research)
    assert "reddit.com" not in ddgs_src, "reddit.com is still in tool_search_ddgs junk list."
    assert "reddit.com" not in dr_src, "reddit.com is still in tool_deep_research junk list."

def test_agent_xml_generation():
    # The XML rendering logic now lives in a module-level helper
    # (_render_assistant_with_tool_calls), not inside GhostAgent. Inspect
    # both the class and the module so we keep working regardless of
    # where the code physically sits.
    from ghost_agent.core import agent as _agent_mod
    from ghost_agent.core.agent import GhostAgent, _render_assistant_with_tool_calls
    helper_src = inspect.getsource(_render_assistant_with_tool_calls)
    class_src = inspect.getsource(GhostAgent)
    combined = helper_src + "\n" + class_src
    # Ensure tool execution parser uses new syntax
    assert "<function name=\"{" in combined or "xml_call += f'<parameter name=\"{" in combined, "Agent does not assemble XML cleanly for Qwen 3.5."
    assert ".get(\"name\"" in combined

def test_agent_uncensored_stall_catcher():
    from ghost_agent.core.agent import GhostAgent
    agent_src = inspect.getsource(GhostAgent)
    assert "Conversational fallback removed for smarter models." in agent_src, "Fallback removal comment not found."

def test_agent_temperature_fixed_profiles():
    """Temperature is no longer passed through from args — two fixed
    profiles (coding vs general) are selected via get_sampling_params()."""
    from ghost_agent.core.agent import (
        GhostAgent,
        CODING_SAMPLING_PARAMS,
        GENERAL_SAMPLING_PARAMS,
        get_sampling_params,
    )
    agent_src = inspect.getsource(GhostAgent)
    assert "current_temp" not in agent_src
    assert "args.temperature" not in agent_src
    assert "get_sampling_params(" in agent_src
    assert CODING_SAMPLING_PARAMS["temperature"] == 0.6
    assert GENERAL_SAMPLING_PARAMS["temperature"] == 1.0
    assert get_sampling_params(True)["presence_penalty"] == 0
    assert get_sampling_params(False)["presence_penalty"] == 1.5
