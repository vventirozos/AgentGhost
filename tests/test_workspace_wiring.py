"""End-to-end wiring tests for workspace continuity:

  * deep_research records research artifacts into workspace.
  * GhostContext exposes a workspace_model slot defaulting to None.
  * The prompt-assembly path splices the workspace prefix into the
    system prompt when workspace_model is enabled.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.workspace import WorkspaceModel


# ---------------------------------------------------------------------
# GhostContext default
# ---------------------------------------------------------------------


def test_ghost_context_has_workspace_model_attribute():
    from ghost_agent.core.agent import GhostContext
    args = SimpleNamespace(model="x", max_context=4000)
    ctx = GhostContext(args, "/tmp/sandbox", "/tmp/memory", None)
    assert hasattr(ctx, "workspace_model")
    assert ctx.workspace_model is None


# ---------------------------------------------------------------------
# deep_research artifact sink
# ---------------------------------------------------------------------


async def test_deep_research_records_artifacts_into_workspace(tmp_path: Path):
    """Bypass the network entirely by patching DDGS and the page
    fetcher, then assert deep_research records the discovered URLs."""
    from ghost_agent.tools import search as search_mod

    wm = WorkspaceModel(tmp_path)
    urls = [
        "https://paper.org/a",
        "https://paper.org/b",
        "https://paper.org/c",
    ]

    class _FakeDDGS:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def text(self, q, **kwargs):
            return [{"href": u, "title": u} for u in urls]

    # Patch the importlib find_spec to claim 'ddgs' is available, the
    # ddgs.DDGS class itself, and the page fetcher used downstream.
    with patch.object(search_mod, "importlib") as imp_mod:
        imp_mod.util.find_spec = MagicMock(return_value=object())
        with patch.dict("sys.modules", {"ddgs": MagicMock(DDGS=_FakeDDGS)}):
            with patch.object(
                search_mod, "helper_fetch_url_content",
                new=AsyncMock(return_value="dummy page text"),
            ):
                out = await search_mod.tool_deep_research(
                    query="test paper", workspace_model=wm,
                )
    assert "DEEP RESEARCH RESULT" in out
    # Each discovered URL should be in the workspace's seen-urls set
    # and have generated a research event.
    research_events = wm.activity.recent(limit=20, kind="research")
    seen_urls = {e.payload.get("url") for e in research_events}
    assert seen_urls == set(urls)
    for u in urls:
        assert wm.has_seen_url(u)


async def test_deep_research_dedups_on_repeat(tmp_path: Path):
    """A second deep_research call for the same URLs must not create
    duplicate research events."""
    from ghost_agent.tools import search as search_mod

    wm = WorkspaceModel(tmp_path)
    urls = ["https://paper.org/a"]

    class _FakeDDGS:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def text(self, q, **kwargs):
            return [{"href": u, "title": u} for u in urls]

    with patch.object(search_mod, "importlib") as imp_mod:
        imp_mod.util.find_spec = MagicMock(return_value=object())
        with patch.dict("sys.modules", {"ddgs": MagicMock(DDGS=_FakeDDGS)}):
            with patch.object(
                search_mod, "helper_fetch_url_content",
                new=AsyncMock(return_value="dummy"),
            ):
                await search_mod.tool_deep_research(query="x", workspace_model=wm)
                await search_mod.tool_deep_research(query="x again", workspace_model=wm)
    assert len(wm.activity.recent(limit=20, kind="research")) == 1


# ---------------------------------------------------------------------
# Prompt-assembly splices the workspace prefix
# ---------------------------------------------------------------------


def test_workspace_wakeup_prefix_renders_into_string(tmp_path: Path):
    """The prefix builder is what the agent splices into base_prompt.
    Verify it returns a non-empty string when there is workspace state."""
    target = tmp_path / "code.py"
    target.write_text("def main():\n    pass\n")
    wm = WorkspaceModel(tmp_path)
    wm.track_file(str(target), label="entry")
    wm.mark_session_boot()
    prefix = wm.build_wakeup_prefix()
    assert isinstance(prefix, str)
    assert "WORKSPACE STATE" in prefix
    assert "code.py" in prefix


def test_workspace_wakeup_prefix_empty_when_disabled(tmp_path: Path):
    wm = WorkspaceModel(tmp_path, enabled=False)
    assert wm.build_wakeup_prefix() == ""


def test_workspace_wakeup_prefix_empty_when_no_state(tmp_path: Path):
    wm = WorkspaceModel(tmp_path)
    assert wm.build_wakeup_prefix() == ""
