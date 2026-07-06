"""Tests for the verifier's web-artifact execution check.

Regression target (req EA): the text verifier said CONFIRMED (95%) on a
freshly built web app whose data.js had a parse error — every claim/
evidence pair read fine, but the page threw on load and the user found
out by clicking a dead button. When a turn WRITES web files, the entry
page must be loaded headless and an uncaught exception must refute the
answer regardless of how plausible the claim text is.
"""
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core.agent import GhostAgent, _web_artifacts_written


# ── extracting written web files from turn records ───────────────────
def test_extracts_web_files_from_success_messages():
    tools = [
        {"role": "tool", "name": "file_system",
         "content": "SUCCESS: Wrote 8214 chars to 'index.html'. "
                     "Script-side path (from sandbox cwd): 'index.html'."},
        {"role": "tool", "name": "file_system",
         "content": "SUCCESS: Exact match found and replaced in 'game.js'."},
        {"role": "tool", "name": "file_system",
         "content": "SUCCESS: Wrote 90 chars to 'notes.md'."},
    ]
    assert _web_artifacts_written(tools) == ["index.html", "game.js"]


def test_ignores_synthetic_failures_and_other_tools():
    tools = [
        {"role": "tool", "name": "file_system", "_synthetic": True,
         "content": "SUCCESS: Wrote 10 chars to 'fake.html'."},
        {"role": "tool", "name": "browser",
         "content": "SUCCESS: navigated to 'page.html'."},
        {"role": "tool", "name": "file_system",
         "content": "Error: could not write 'broken.js'."},
    ]
    assert _web_artifacts_written(tools) == []


def test_empty_and_none_are_safe():
    assert _web_artifacts_written(None) == []
    assert _web_artifacts_written([]) == []


# ── headless execution of the entry page ─────────────────────────────
def _bare_agent(tmp_path, browser_result, monkeypatch):
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = MagicMock()
    browser = AsyncMock(return_value=browser_result)
    agent.available_tools = {"browser": browser}
    monkeypatch.setattr(
        "ghost_agent.tools.file_system.project_scoped_sandbox",
        lambda ctx, stateful=False: (tmp_path, "/workspace"),
    )
    return agent, browser


async def test_html_artifact_clean_load(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("<html></html>")
    agent, browser = _bare_agent(
        tmp_path, "SUCCESS: navigated. Title: 'x'", monkeypatch)
    res = await agent._execute_web_artifact(["index.html"])
    assert res == ("index.html", "")
    assert browser.await_count == 1
    assert browser.call_args.kwargs["operation"] == "navigate"


async def test_html_artifact_with_uncaught_exception_returns_block(
        tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("<html></html>")
    diag = ("SUCCESS: navigated.\n⚠ UNCAUGHT JS EXCEPTIONS (2) — these "
            "crash the page silently:\n  • SyntaxError: Unexpected "
            "identifier 't'")
    agent, _ = _bare_agent(tmp_path, diag, monkeypatch)
    page_rel, block = await agent._execute_web_artifact(["index.html"])
    assert page_rel == "index.html"
    assert block.startswith("UNCAUGHT JS EXCEPTIONS")
    assert "Unexpected identifier" in block


async def test_js_only_edit_loads_sibling_index(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("<html></html>")
    (tmp_path / "game.js").write_text("var x = 1;")
    agent, browser = _bare_agent(tmp_path, "SUCCESS: navigated", monkeypatch)
    res = await agent._execute_web_artifact(["game.js"])
    assert res is not None and res[0] == "index.html"


async def test_navigate_url_is_absolute_container_path(tmp_path, monkeypatch):
    """Reliability bug: the probe built ``file://index.html`` (relative →
    parsed as a host → never loads), so WEB-EXEC silently 'skipped' on every
    build and a throwing page still got a text CONFIRMED. The URL must be an
    absolute ``file:///workspace/...`` path."""
    (tmp_path / "index.html").write_text("<html></html>")
    agent, browser = _bare_agent(tmp_path, "SUCCESS: navigated", monkeypatch)
    await agent._execute_web_artifact(["index.html"])
    url = browser.call_args.kwargs["url"]
    assert url.startswith("file:///workspace/"), url
    assert url == "file:///workspace/index.html"


async def test_navigate_url_scoped_project(tmp_path, monkeypatch):
    """When the sandbox is project-scoped, the container URL must carry the
    ``projects/<id>/`` segment (the mount is at the root, not the scope)."""
    proj = tmp_path / "projects" / "abc123"
    proj.mkdir(parents=True)
    (proj / "index.html").write_text("<html></html>")
    agent, browser = _bare_agent(proj, "SUCCESS: navigated", monkeypatch)
    monkeypatch.setattr(
        "ghost_agent.tools.file_system.project_scoped_sandbox",
        lambda ctx, stateful=False: (proj, "/workspace"),
    )
    res = await agent._execute_web_artifact(["index.html"])
    assert res == ("projects/abc123/index.html", "")
    assert browser.call_args.kwargs["url"] == \
        "file:///workspace/projects/abc123/index.html"


async def test_binding_gap_finds_deliverable_in_project_subdir(
        tmp_path, monkeypatch):
    """Live failure: a project-reuse turn left the deliverable in
    ``projects/<id>/index.html`` while project_scoped_sandbox read as
    UN-scoped (root). The old direct-path lookup missed it → 'skipped'. The
    newest-wins basename fallback must still find and load it."""
    (tmp_path / "projects" / "reuse99").mkdir(parents=True)
    (tmp_path / "projects" / "reuse99" / "index.html").write_text("<html></html>")
    # sandbox reads as the bare root (binding gap)
    agent, browser = _bare_agent(tmp_path, "SUCCESS: navigated", monkeypatch)
    monkeypatch.setattr(
        "ghost_agent.tools.file_system.project_scoped_sandbox",
        lambda ctx, stateful=False: (tmp_path, "/workspace"),
    )
    res = await agent._execute_web_artifact(["index.html"])
    assert res is not None and res[0] == "projects/reuse99/index.html"
    assert browser.call_args.kwargs["url"] == \
        "file:///workspace/projects/reuse99/index.html"


async def test_stale_fallback_file_does_not_certify(tmp_path, monkeypatch):
    """Live false-confirm: the turn's deliverable (projects/<new>/index.html)
    never landed on disk, the basename fallback found a 25-min-old index.html
    from an UNRELATED project, and WEB-EXEC reported it clean. A fallback
    match older than the freshness window must be rejected → inconclusive."""
    import os
    old = tmp_path / "projects" / "old_proj"
    old.mkdir(parents=True)
    stale = old / "index.html"
    stale.write_text("<html></html>")
    # age it well past the freshness window
    past = time.time() - 4000
    os.utime(stale, (past, past))
    agent, browser = _bare_agent(tmp_path, "SUCCESS: navigated", monkeypatch)
    monkeypatch.setattr(
        "ghost_agent.tools.file_system.project_scoped_sandbox",
        lambda ctx, stateful=False: (tmp_path, "/workspace"),
    )
    # the agent "wrote" a file in a project dir that isn't on disk
    res = await agent._execute_web_artifact(["projects/new_proj/index.html"])
    assert res is None
    browser.assert_not_awaited()


async def test_no_entry_page_is_inconclusive(tmp_path, monkeypatch):
    (tmp_path / "lonely.js").write_text("var x = 1;")
    agent, _ = _bare_agent(tmp_path, "SUCCESS", monkeypatch)
    assert await agent._execute_web_artifact(["lonely.js"]) is None


async def test_failed_navigation_is_inconclusive_not_clean(
        tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("<html></html>")
    agent, _ = _bare_agent(
        tmp_path, "Error: browser crashed before navigation", monkeypatch)
    assert await agent._execute_web_artifact(["index.html"]) is None


async def test_missing_browser_tool_is_inconclusive(tmp_path, monkeypatch):
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = MagicMock()
    agent.available_tools = {}
    assert await agent._execute_web_artifact(["index.html"]) is None


# ── the override inside _compute_verifier_verdict ────────────────────
async def test_exec_failure_overrides_text_confirmed(tmp_path, monkeypatch):
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict

    agent = GhostAgent.__new__(GhostAgent)
    agent.context = MagicMock()
    agent.available_tools = {}
    agent._is_strict_trivial_chat = lambda lc: False

    verifier = MagicMock()
    verifier.llm_client = MagicMock()
    verifier.verify_claim = AsyncMock(return_value=VerifyResult(
        verdict=VerifyVerdict.CONFIRMED, confidence=0.95,
        reasoning="claim matches evidence",
    ))
    agent.context.verifier = verifier

    agent._execute_web_artifact = AsyncMock(return_value=(
        "index.html",
        "UNCAUGHT JS EXCEPTIONS (2)\n  • SyntaxError: Unexpected identifier 't'",
    ))

    tools = [{"role": "tool", "name": "file_system",
              "content": "SUCCESS: Wrote 1000 chars to 'index.html'."}]
    v_result, last_tool = await agent._compute_verifier_verdict(
        tools_run_this_turn=tools,
        messages=[{"role": "user", "content": "build me a web game"}],
        final_ai_content="Done! The game is ready.",
        last_user_content="build me a web game",
        lc="build me a web game",
    )
    assert v_result is not None
    assert v_result.verdict == VerifyVerdict.REFUTED
    assert "index.html" in v_result.reasoning
    assert "Unexpected identifier" in v_result.reasoning


async def test_clean_exec_keeps_text_verdict(tmp_path, monkeypatch):
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict

    agent = GhostAgent.__new__(GhostAgent)
    agent.context = MagicMock()
    agent.available_tools = {}
    agent._is_strict_trivial_chat = lambda lc: False

    verifier = MagicMock()
    verifier.llm_client = MagicMock()
    confirmed = VerifyResult(
        verdict=VerifyVerdict.CONFIRMED, confidence=0.9, reasoning="ok")
    verifier.verify_claim = AsyncMock(return_value=confirmed)
    agent.context.verifier = verifier

    agent._execute_web_artifact = AsyncMock(return_value=("index.html", ""))

    tools = [{"role": "tool", "name": "file_system",
              "content": "SUCCESS: Wrote 1000 chars to 'index.html'."}]
    v_result, _ = await agent._compute_verifier_verdict(
        tools_run_this_turn=tools,
        messages=[{"role": "user", "content": "build me a web game"}],
        final_ai_content="Done! The game is ready.",
        last_user_content="build me a web game",
        lc="build me a web game",
    )
    assert v_result is confirmed
    assert v_result.confidence == 0.9  # exec-backed → NOT capped


# ── skipped/failed probe must cap a CONFIRMED (fail-safe, not fail-open) ──
def _verdict_agent(verdict_result):
    """Bare agent whose text verifier returns `verdict_result`."""
    from unittest.mock import AsyncMock, MagicMock
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = MagicMock()
    agent.available_tools = {}
    agent._is_strict_trivial_chat = lambda lc: False
    verifier = MagicMock()
    verifier.llm_client = MagicMock()
    verifier.verify_claim = AsyncMock(return_value=verdict_result)
    agent.context.verifier = verifier
    return agent


_WEB_WRITE_TOOLS = [{"role": "tool", "name": "file_system",
                     "content": "SUCCESS: Wrote 1000 chars to 'index.html'."}]


async def _verdict_for(agent):
    return await agent._compute_verifier_verdict(
        tools_run_this_turn=_WEB_WRITE_TOOLS,
        messages=[{"role": "user", "content": "build me a web game"}],
        final_ai_content="Done! The game is ready.",
        last_user_content="build me a web game",
        lc="build me a web game",
    )


async def test_skipped_exec_caps_text_confirmed():
    """Live failure (2026-06-20): WEB-EXEC logged 'skipped' and the verifier
    still said CONFIRMED 100% without ever executing the artifact. A skipped
    probe must cap a CONFIRMED below the 0.7 consumption threshold."""
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    agent = _verdict_agent(VerifyResult(
        verdict=VerifyVerdict.CONFIRMED, confidence=1.0, reasoning="looks right"))
    agent._execute_web_artifact = AsyncMock(return_value=None)  # probe skipped
    v_result, _ = await _verdict_for(agent)
    assert v_result.verdict == VerifyVerdict.CONFIRMED  # verdict kept
    assert v_result.confidence == GhostAgent._WEB_EXEC_SKIP_CONF_CAP
    assert v_result.confidence < 0.7
    assert "WEB-EXEC inconclusive" in v_result.reasoning


async def test_probe_crash_caps_confirmed():
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    agent = _verdict_agent(VerifyResult(
        verdict=VerifyVerdict.CONFIRMED, confidence=0.95, reasoning="ok"))
    agent._execute_web_artifact = AsyncMock(side_effect=RuntimeError("boom"))
    v_result, _ = await _verdict_for(agent)
    assert v_result.verdict == VerifyVerdict.CONFIRMED
    assert v_result.confidence == GhostAgent._WEB_EXEC_SKIP_CONF_CAP


async def test_skipped_exec_leaves_refuted_untouched():
    # Refuting is already the fail-safe direction — never weaken it.
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    agent = _verdict_agent(VerifyResult(
        verdict=VerifyVerdict.REFUTED, confidence=0.9, reasoning="wrong"))
    agent._execute_web_artifact = AsyncMock(return_value=None)
    v_result, _ = await _verdict_for(agent)
    assert v_result.verdict == VerifyVerdict.REFUTED
    assert v_result.confidence == 0.9


async def test_no_web_writes_no_cap():
    # A turn that wrote no web artifacts never runs the probe → no cap.
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    agent = _verdict_agent(VerifyResult(
        verdict=VerifyVerdict.CONFIRMED, confidence=0.95, reasoning="ok"))
    agent._execute_web_artifact = AsyncMock(return_value=None)
    tools = [{"role": "tool", "name": "file_system",
              "content": "SUCCESS: Wrote 90 chars to 'notes.md'."}]
    v_result, _ = await agent._compute_verifier_verdict(
        tools_run_this_turn=tools,
        messages=[{"role": "user", "content": "take a note"}],
        final_ai_content="Noted.",
        last_user_content="take a note",
        lc="take a note",
    )
    assert v_result.confidence == 0.95
    agent._execute_web_artifact.assert_not_awaited()
