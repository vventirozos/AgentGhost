import pytest

def test_prompts_contain_replace_instructions():
    """The EDITING EXISTING FILES section used to unconditionally forbid
    `file_system write` on existing files and mandate `replace`. That
    rule was softened when traces showed the whitespace-flexible
    matcher is brittle on multi-line blocks: for edits of more than ~3
    lines, a whole-file `write` is faster and more reliable. The
    section now teaches a TOOL-BY-EDIT-SIZE choice. Pin the invariants
    that still matter: the section exists, the Aider SEARCH/REPLACE
    format is documented, and the fact that write is the preferred
    choice for large edits is surfaced."""
    import os
    prompt_path = os.path.join(os.path.dirname(__file__), "../src/ghost_agent/core/prompts.py")
    with open(prompt_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "EDITING EXISTING FILES" in content
    assert "Aider" in content or "SEARCH/REPLACE" in content or "<<<< SEARCH" in content
    # Both tools must be named so the agent knows which to pick.
    assert "file_system" in content and "replace" in content and "write" in content
    # The size-based preference must be documented so future edits
    # don't silently revert to "always use replace".
    assert "PREFER `file_system write`" in content or "PREFER file_system write" in content

def test_prompts_contain_web_automation_instructions():
    import os
    prompt_path = os.path.join(os.path.dirname(__file__), "../src/ghost_agent/core/prompts.py")
    with open(prompt_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    assert "WEB AUTOMATION (HEADLESS BROWSER)" in content
    assert "async_playwright" in content
    assert "html2text" in content
    assert "--no-sandbox" in content
    assert "os._exit(0)" in content
