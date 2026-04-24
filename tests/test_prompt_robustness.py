
import pytest
from ghost_agent.core.prompts import SPECIALIST_SYSTEM_PROMPT

def test_code_system_prompt_has_robustness_rules():
    """Verify that the SPECIALIST_SYSTEM_PROMPT includes critical engineering standards."""
    
    # Rule 1: Variable Safety
    assert "VARIABLE SAFETY" in SPECIALIST_SYSTEM_PROMPT, "Prompt missing VARIABLE SAFETY rule"
    assert "Initialize variables cleanly" in SPECIALIST_SYSTEM_PROMPT
    
    # Rule 2: Data Flexibility
    assert "DATA SANITIZATION & FLEXIBILITY" in SPECIALIST_SYSTEM_PROMPT, "Prompt missing DATA SANITIZATION & FLEXIBILITY rule"
    assert "proactively clean strings" in SPECIALIST_SYSTEM_PROMPT
    
    # Rule 3: Anti-Loop
    assert "ANTI-LOOP" in SPECIALIST_SYSTEM_PROMPT, "Prompt missing ANTI-LOOP rule"
    assert "DO NOT submit the exact same code again" in SPECIALIST_SYSTEM_PROMPT
    
    # Rule 4: No Backslashes
    assert "NO BACKSLASHES" in SPECIALIST_SYSTEM_PROMPT, "Prompt missing NO BACKSLASHES rule"
    assert "Do not use backslash" in SPECIALIST_SYSTEM_PROMPT

def test_code_system_prompt_has_observability():
    """Verify observability requirements."""
    assert "STRICT OBSERVABILITY OVER DEFENSIVENESS" in SPECIALIST_SYSTEM_PROMPT
    assert "MUST use `print()`" in SPECIALIST_SYSTEM_PROMPT
