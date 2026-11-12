
from ghost_agent.core.prompts import SYSTEM_PROMPT


def test_no_stale_f_string_backslash_rule():
    """PEP 701 (Python 3.12) lifted the f-string backslash restriction; the
    old blanket ban misled the model away from legal code. §4UI removed the
    coding persona that carried the remaining line-continuation rule — the
    main prompt must not reintroduce either."""
    assert "F-STRING BACKSLASH BAN" not in SYSTEM_PROMPT
    assert "Python 3.11 DOES NOT allow backslashes" not in SYSTEM_PROMPT
