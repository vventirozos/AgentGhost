import pytest
import sys
import inspect
from unittest.mock import patch

def test_qwen_model_id():
    from ghost_agent.utils.token_counter import QWEN_MODEL_ID
    assert QWEN_MODEL_ID == "Qwen/Qwen3.5-9B", "Token counter model ID is incorrect."

def test_max_context_default():
    from ghost_agent.main import parse_args
    with patch.object(sys, 'argv', ['main.py']):
        args = parse_args()
        assert args.max_context == 262144, "Max context default is incorrect."

def test_prompt_updates():
    from ghost_agent.core.prompts import QWEN_TOOL_PROMPT, SYSTEM_PROMPT
    assert "<function=function_name>" in QWEN_TOOL_PROMPT, "QWEN_TOOL_PROMPT missing `<function=function_name>` xml syntax."
    assert "<parameter=arg1>" in QWEN_TOOL_PROMPT, "QWEN_TOOL_PROMPT missing `<parameter=arg1>` xml syntax."
    assert "blunt, direct, and aggressively efficient" in SYSTEM_PROMPT, "SYSTEM_PROMPT missing aggressive conversational persona."
    assert "natively multimodal and can physically see images" in SYSTEM_PROMPT, "SYSTEM_PROMPT missing multimodal vision capability."

def test_search_reddit_removed():
    from ghost_agent.tools.search import tool_search_ddgs, tool_deep_research
    ddgs_src = inspect.getsource(tool_search_ddgs)
    dr_src = inspect.getsource(tool_deep_research)
    assert "reddit.com" not in ddgs_src, "reddit.com is still in tool_search_ddgs junk list."
    assert "reddit.com" not in dr_src, "reddit.com is still in tool_deep_research junk list."

def test_agent_xml_generation():
    from ghost_agent.core.agent import GhostAgent
    agent_src = inspect.getsource(GhostAgent)
    # Ensure tool execution parser uses new syntax
    assert "<function={" in agent_src or "xml_call += f'<parameter={" in agent_src, "Agent does not assemble XML cleanly for Qwen 3.5."
    assert ".get(\"name\"" in agent_src

def test_agent_uncensored_stall_catcher():
    from ghost_agent.core.agent import GhostAgent
    agent_src = inspect.getsource(GhostAgent)
    assert "len(clean_ui) > 0 and not is_valid_final" in agent_src, "Length constraint < 300 was not removed in stall catcher."
    assert "not legal advice|general information|consult a professional|not medical advice|own risk" in agent_src, "Uncensored bot disclaimers are not matched."

def test_agent_temperature_passthrough():
    from ghost_agent.core.agent import GhostAgent
    agent_src = inspect.getsource(GhostAgent)
    assert "current_temp = 0.15" not in agent_src, "Hardcoded DBA temperature 0.15 is still defined."
    assert "current_temp = 0.2" not in agent_src, "Hardcoded Coder temperature 0.2 is still defined."
    assert "current_temp = self.context.args.temperature" in agent_src, "Temperature doesn't passthrough context."
