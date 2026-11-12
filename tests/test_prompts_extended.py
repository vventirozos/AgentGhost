import pytest

def test_the_edit_guidance_lives_in_the_file_system_schema():
    """§4UI: the coding persona's EDITING EXISTING FILES section (Aider
    SEARCH blocks, "prefer a whole-file write", the fix-script) contradicted
    the tool schema and was removed with the persona; `edit` is the way."""
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    from ghost_agent.core import prompts
    import inspect
    fs = next(t["function"]["description"] for t in TOOL_DEFINITIONS if t["function"]["name"] == "file_system")
    assert "edit" in fs
    assert "EDITING EXISTING FILES" not in inspect.getsource(prompts)


def test_the_web_automation_guidance_lives_in_the_tool_descriptions():
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    d = {t["function"]["name"]: t["function"]["description"] for t in TOOL_DEFINITIONS}
    assert "interact" in d["browser"]
    assert "top-level `await`" in d["execute"] and "sys.exit()" in d["execute"]
