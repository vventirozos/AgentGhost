
import pytest
from ghost_agent.core.prompts import SPECIALIST_SYSTEM_PROMPT


def test_code_system_prompt_has_no_stale_f_string_rule():
    """The Python-3.11 f-string backslash ban was removed when the agent
    moved to Qwen 3.6 35B-A3 (and a Python 3.12+ sandbox).

    PEP 701 (merged in Python 3.12) lifted the f-string backslash
    restriction, so a blanket "NEVER use backslashes inside f-string
    expressions" rule now actively misleads the model away from legal
    code. This test pins the rule as REMOVED so a future revert (or a
    stray copy-paste of the old prompt text) is caught loudly.

    The older, narrower "NO BACKSLASHES: Do not use backslash `\\` for
    line continuation" guidance stays — that's about multi-line
    expressions, not f-strings, and is still good advice.
    """
    assert "F-STRING BACKSLASH BAN" not in SPECIALIST_SYSTEM_PROMPT
    assert "Python 3.11 DOES NOT allow backslashes" not in SPECIALIST_SYSTEM_PROMPT
    # The narrow line-continuation guidance must still be present.
    assert "NO BACKSLASHES" in SPECIALIST_SYSTEM_PROMPT
