"""§4FJ pins: the `workspace` tool no longer advertises itself as a place to
do work, and the diet bench can measure that change against the old text as a
fixed head. (The coding executor's think-probe pins went with the executor,
§4UY C18.)
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ghost_agent.tools import registry as R

# --- workspace description + the bench pair ----------------------------------

def _workspace_description():
    return next(t["function"]["description"] for t in R.TOOL_DEFINITIONS
                if t["function"]["name"] == "workspace")


def test_workspace_describes_a_ledger_and_names_the_tools_that_do_work():
    """World where it fails: the old text is back ("files", "show me what
    you've been doing in my project") and coding requests route to it."""
    d = _workspace_description()
    assert "ALREADY HAPPENED" in d and "NEVER call this to work on files" in d
    for tool in ("file_system", "execute", "browser", "manage_projects"):
        assert tool in d
    assert "in my project" not in d and "(files," not in d
    assert "'summary'" in d and "'search'" in d          # the actions survived


# (§4NM: the two scripts/tool_head_diet_bench.py pins left with scripts/)
def test_tuned_description_must_keep_the_baselines_never_and_only_sentences():
    """World where it fails: a promoted tool_description.workspace.json
    rewrites the text toward the reward, drops the NEVER sentence, passes
    the size cap, and the live head serves the confusion again while the
    registry pin stays green."""
    base = _workspace_description()
    pinned = R._pinned_sentences("workspace", base)
    assert pinned == R.TOOL_DESC_PINNED["workspace"], "a pin drifted out of the live baseline"
    assert any(s.startswith("NEVER call this") for s in pinned)
    assert R._validate_tool_description("workspace", base, base) is True
    shed = base.replace(next(s for s in pinned if s.startswith("NEVER")), "")
    assert R._validate_tool_description("workspace", base, shed) is False
    reordered = "Prefix. " + base
    assert R._validate_tool_description("workspace", base, reordered) is True
    # every pinned sentence of every tool must occur verbatim in its live baseline
    live = {t["function"]["name"]: t["function"]["description"] for t in R.TOOL_DEFINITIONS}
    for tool, pins in R.TOOL_DESC_PINNED.items():
        assert tool in live, tool
        for sent in pins:
            assert sent in live[tool], (tool, sent[:40])
    # unpinned tools keep the tuning contract the read-site invariants rely on
    assert R._pinned_sentences("web_search", live["web_search"]) == []
    assert R._validate_tool_description("web_search", live["web_search"], "Search anything.") is True


def test_wake_up_prefix_no_longer_frames_the_workspace_as_a_place_to_work():
    """§4FJ review C1: the prefix sits ABOVE the tool schemas on every turn."""
    from tests.test_workspace_recognition import (WorkspaceActivity, WorkspaceEvent,
                                                  build_workspace_prefix)
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        act = WorkspaceActivity(Path(d))
        act.append(WorkspaceEvent(kind="note", summary="one tracked change"))
        text = build_workspace_prefix(activity=act, state=None)
    assert text, "an empty prefix cannot be checked"
    assert "WHAT'S OUTSIDE OF ME" not in text and "what I'm looking at" not in text
    assert "ACTIVITY LEDGER" in text and "not a place to do work" in text
    assert "### WORKSPACE STATE" in text          # the substring three older pins rely on
