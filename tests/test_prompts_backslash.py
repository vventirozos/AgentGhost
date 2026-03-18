
import pytest
from ghost_agent.core.prompts import CODE_SYSTEM_PROMPT

def test_code_system_prompt_backslash_ban():
    """Verify that CODE_SYSTEM_PROMPT contains the F-String Backslash Ban rule."""
    expected_rule = "F-STRING BACKSLASH BAN: Python 3.11 DOES NOT allow backslashes (\\) inside f-string expressions"
    assert expected_rule in CODE_SYSTEM_PROMPT, "CODE_SYSTEM_PROMPT is missing the F-String Backslash Ban rule."
