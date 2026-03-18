
import pytest
from ghost_agent.core.prompts import CODE_SYSTEM_PROMPT

def test_code_system_prompt_has_robustness_rules():
    """Verify that the CODE_SYSTEM_PROMPT includes critical engineering standards."""
    
    # Rule 1: Variable Safety
    assert "VARIABLE SAFETY" in CODE_SYSTEM_PROMPT, "Prompt missing VARIABLE SAFETY rule"
    assert "Initialize variables *before* `try` blocks" in CODE_SYSTEM_PROMPT
    
    # Rule 2: Data Flexibility
    assert "DATA FLEXIBILITY" in CODE_SYSTEM_PROMPT, "Prompt missing DATA FLEXIBILITY rule"
    assert "fallback to `ast.literal_eval`" in CODE_SYSTEM_PROMPT
    
    # Rule 3: Anti-Loop
    assert "ANTI-LOOP" in CODE_SYSTEM_PROMPT, "Prompt missing ANTI-LOOP rule"
    assert "DO NOT submit the exact same code again" in CODE_SYSTEM_PROMPT
    
    # Rule 4: No Backslashes
    assert "NO BACKSLASHES" in CODE_SYSTEM_PROMPT, "Prompt missing NO BACKSLASHES rule"
    assert "Do not use backslash" in CODE_SYSTEM_PROMPT

def test_code_system_prompt_has_observability():
    """Verify observability requirements."""
    assert "ABSOLUTE OBSERVABILITY" in CODE_SYSTEM_PROMPT
    assert "MUST use `print()`" in CODE_SYSTEM_PROMPT
