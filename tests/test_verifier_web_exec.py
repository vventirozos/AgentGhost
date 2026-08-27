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
        # ⚠ These pin the `$` anchor on the web-extension test. Without a
        # file whose name merely CONTAINS a web extension, un-anchoring the
        # regex changes no assertion, and `data.jsonl` / `App.jsx` would be
        # handed to the headless entry-page probe as though they were pages.
        {"role": "tool", "name": "file_system",
         "content": "SUCCESS: Wrote 12 chars to 'data.jsonl'."},
        {"role": "tool", "name": "file_system",
         "content": "SUCCESS: Wrote 12 chars to 'App.jsx'."},
    ]
    assert _web_artifacts_written(tools) == ["index.html", "game.js"]


def test_ignores_synthetic_failures_and_other_tools():
    tools = [
        {"role": "tool", "name": "file_system", "_synthetic": True,
         "content": "SUCCESS: Wrote 10 chars to 'fake.html'."},
        # Must carry a mutation marker + quoted path so the tool-NAME gate
        # is what excludes it (see the M11 note in test_grounded_file_verify).
        {"role": "tool", "name": "run_skill",
         "content": "SUCCESS: Wrote 40 chars to 'page.html'."},
        {"role": "tool", "name": "file_system",
         "content": "Error: could not write 'broken.js'."},
        # ⚠ Load-bearing for the SUCCESS gate: this one carries mutation
        # wording AND a quoted path, so only the "must start with SUCCESS"
        # rule excludes it. Without it the gate could be deleted and this
        # test still passed (lens C, 2026-08-26).
        {"role": "tool", "name": "file_system",
         "content": "REJECTED: that replace would have written marker lines "
                    "into 'x.html'."},
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


# ── multi-page probing (2026-07-19 gap) ──────────────────────────────
# The turn created minesweeper.html AND corrupted index.html via replace;
# the probe picked the FIRST page only, so the mutated index.html was never
# loaded and the corruption shipped behind "WEB-EXEC clean". Every located
# html page written this turn must now load clean.

def _multi_agent(tmp_path, results_by_url, monkeypatch, default="SUCCESS: navigated"):
    """Agent whose browser mock answers per-URL from ``results_by_url``."""
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = MagicMock()

    async def _nav(**kwargs):
        return results_by_url.get(kwargs.get("url"), default)

    browser = AsyncMock(side_effect=_nav)
    agent.available_tools = {"browser": browser}
    monkeypatch.setattr(
        "ghost_agent.tools.file_system.project_scoped_sandbox",
        lambda ctx, stateful=False: (tmp_path, "/workspace"),
    )
    return agent, browser


async def test_second_page_exception_refutes(tmp_path, monkeypatch):
    (tmp_path / "minesweeper.html").write_text("<html></html>")
    (tmp_path / "index.html").write_text("<html></html>")
    diag = ("SUCCESS: navigated.\n⚠ UNCAUGHT JS EXCEPTIONS (1):\n"
            "  • SyntaxError: Unexpected token '='")
    agent, browser = _multi_agent(
        tmp_path,
        {"file:///workspace/index.html": diag},
        monkeypatch,
    )
    page_rel, block = await agent._execute_web_artifact(
        ["minesweeper.html", "index.html"])
    assert page_rel == "index.html"          # the FAILING page is named
    assert block.startswith("UNCAUGHT JS EXCEPTIONS")
    assert browser.await_count == 2          # both pages actually probed


async def test_all_pages_clean_names_every_page(tmp_path, monkeypatch):
    (tmp_path / "a.html").write_text("<html></html>")
    (tmp_path / "b.html").write_text("<html></html>")
    agent, browser = _multi_agent(tmp_path, {}, monkeypatch)
    page_rel, block = await agent._execute_web_artifact(["a.html", "b.html"])
    assert block == ""
    assert page_rel == "a.html, b.html"
    assert browser.await_count == 2


async def test_one_unloadable_page_is_inconclusive(tmp_path, monkeypatch):
    # A mutated page that fails to NAVIGATE must make the whole probe
    # inconclusive (None → confidence cap), never read as "clean".
    (tmp_path / "a.html").write_text("<html></html>")
    (tmp_path / "b.html").write_text("<html></html>")
    agent, _ = _multi_agent(
        tmp_path,
        {"file:///workspace/b.html": "Error: net::ERR_FILE_NOT_FOUND"},
        monkeypatch,
    )
    assert await agent._execute_web_artifact(["a.html", "b.html"]) is None


async def test_page_probe_cap(tmp_path, monkeypatch):
    for i in range(6):
        (tmp_path / f"p{i}.html").write_text("<html></html>")
    agent, browser = _multi_agent(tmp_path, {}, monkeypatch)
    res = await agent._execute_web_artifact(
        [f"p{i}.html" for i in range(6)])
    assert res is not None and res[1] == ""
    assert browser.await_count == GhostAgent._WEB_EXEC_MAX_PAGES


async def test_duplicate_candidates_probed_once(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("<html></html>")
    agent, browser = _multi_agent(tmp_path, {}, monkeypatch)
    res = await agent._execute_web_artifact(["index.html", "index.html"])
    assert res == ("index.html", "")
    assert browser.await_count == 1


# ── fetch-backed pages probed via the RUNNING service (2026-07-25) ────
# reqs 92a968fc & 646e9b7e: both turns skipped here ("calls the network")
# while the project's service was up on its port the whole time — broken
# JS shipped and the operator found it in their browser.

FETCH_PAGE = "<html><script>fetch('/api/items').then(r=>r.json())</script></html>"


def _fake_supervisor(monkeypatch, entries, alive=True):
    sup = MagicMock()
    sup.list_entries.return_value = entries
    sup._entry_alive.return_value = alive
    sup._port_listening.return_value = alive
    monkeypatch.setattr(
        "ghost_agent.sandbox.services.get_service_supervisor",
        lambda sm: sup,
    )
    return sup


async def test_fetch_page_probed_via_running_service(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text(FETCH_PAGE)
    agent, browser = _bare_agent(
        tmp_path, "SUCCESS: navigated. Title: 'x'", monkeypatch)
    _fake_supervisor(monkeypatch, [
        {"name": "app", "port": 8101, "workdir": "/workspace"}])
    res = await agent._execute_web_artifact(["index.html"])
    assert res == ("index.html", "")
    assert browser.await_count == 1
    assert browser.call_args.kwargs["url"] == "http://127.0.0.1:8101/index.html"


async def test_fetch_page_service_exception_refutes_via_http(
        tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text(FETCH_PAGE)
    agent, browser = _bare_agent(
        tmp_path,
        "Navigated.\nUNCAUGHT JS EXCEPTIONS\nTypeError: DataStore.ready "
        "is not a function", monkeypatch)
    _fake_supervisor(monkeypatch, [
        {"name": "app", "port": 8101, "workdir": "/workspace"}])
    res = await agent._execute_web_artifact(["index.html"])
    assert res is not None
    page_rel, block = res
    assert page_rel == "index.html"
    assert "DataStore.ready" in block


async def test_fetch_page_without_service_stays_inconclusive(
        tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text(FETCH_PAGE)
    agent, browser = _bare_agent(tmp_path, "SUCCESS", monkeypatch)
    monkeypatch.setattr(
        "ghost_agent.sandbox.services.get_service_supervisor",
        lambda sm: None,
    )
    res = await agent._execute_web_artifact(["index.html"])
    assert res is None                 # inconclusive, conf cap applies
    assert browser.await_count == 0


async def test_fetch_page_dead_service_stays_inconclusive(
        tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text(FETCH_PAGE)
    agent, browser = _bare_agent(tmp_path, "SUCCESS", monkeypatch)
    _fake_supervisor(monkeypatch, [
        {"name": "app", "port": 8101, "workdir": "/workspace"}],
        alive=False)
    res = await agent._execute_web_artifact(["index.html"])
    assert res is None
    assert browser.await_count == 0


async def test_fetch_page_longest_workdir_service_wins(tmp_path, monkeypatch):
    proj = tmp_path / "projects" / "abc123"
    proj.mkdir(parents=True)
    (proj / "index.html").write_text(FETCH_PAGE)
    agent, browser = _bare_agent(
        tmp_path, "SUCCESS: navigated.", monkeypatch)
    _fake_supervisor(monkeypatch, [
        {"name": "root", "port": 8000, "workdir": "/workspace"},
        {"name": "app", "port": 8101,
         "workdir": "/workspace/projects/abc123"},
    ])
    res = await agent._execute_web_artifact(["projects/abc123/index.html"])
    assert res is not None and res[1] == ""
    assert browser.call_args.kwargs["url"] == "http://127.0.0.1:8101/index.html"


async def test_plain_page_still_uses_file_url(tmp_path, monkeypatch):
    # No fetch/XHR → the file:// probe is sufficient; no service lookup.
    (tmp_path / "index.html").write_text("<html><body>static</body></html>")
    agent, browser = _bare_agent(
        tmp_path, "SUCCESS: navigated.", monkeypatch)
    _fake_supervisor(monkeypatch, [
        {"name": "app", "port": 8101, "workdir": "/workspace"}])
    res = await agent._execute_web_artifact(["index.html"])
    assert res == ("index.html", "")
    assert browser.call_args.kwargs["url"].startswith("file://")


# ── deleted / echoed artifacts must not reach the execution probe ─────
# Same 2026-08-26 root cause as the FILE-ARTIFACT refute: the extractor
# read every quoted web-extension token in every SUCCESS message, so a
# deleted scratch script was handed to the entry-page probe as though the
# turn had shipped it.
def test_deleted_scratch_js_is_not_a_web_artifact():
    tools = [
        {"role": "tool", "name": "file_system",
         "content": "SUCCESS: Wrote 8214 chars to 'index.html'."},
        {"role": "tool", "name": "file_system",
         "content": "SUCCESS: Wrote 300 chars to '_chk.js'."},
        {"role": "tool", "name": "file_system",
         "content": "SUCCESS: Deleted '_chk.js'."},
    ]
    assert _web_artifacts_written(tools) == ["index.html"]


def test_echoed_source_is_not_a_web_artifact():
    tools = [{
        "role": "tool", "name": "file_system",
        "content": "SUCCESS: Exact match found and replaced in 'app.html'. "
                   "VERIFY the change is what you intended:\n"
                   "--- REPLACED BLOCK (was) ---\n"
                   "<script src='vendor/legacy.js'></script>",
    }]
    assert _web_artifacts_written(tools) == ["app.html"]


def test_renamed_source_is_not_a_web_artifact():
    tools = [
        {"role": "tool", "name": "file_system",
         "content": "SUCCESS: Wrote 100 chars to 'old.html'."},
        {"role": "tool", "name": "file_system",
         "content": "SUCCESS: Renamed/Moved 'old.html' to 'new.html'."},
    ]
    assert _web_artifacts_written(tools) == ["new.html"]


# ── the retired-entry-page branch of the WEB-EXEC consumer ────────────
# ⚠ These pin agent.py's `_retired_pages` branch, which had NO coverage at
# all: deleting it outright left 137 tests green. `TestPathLedgerSuppress-
# ionIsVisible` tests only the HELPER that feeds it, and the nearest
# neighbour (`test_no_web_writes_no_cap`) uses a fixture where the branch
# is never entered in either direction. Both directions are needed — the
# cap must arm when the page is gone, and must NOT arm for a scratch file.
async def test_a_retired_entry_page_still_probes_and_arms_the_cap():
    """The turn wrote its only page and then renamed it away. `written` is
    empty, so a plain `if written:` would skip the probe AND its
    inconclusive cap — certifying at full confidence a build with no entry
    page on disk. WEB-EXEC has no claim-prose leg to catch that."""
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    agent = _verdict_agent(VerifyResult(
        verdict=VerifyVerdict.CONFIRMED, confidence=1.0, reasoning="ok"))
    agent._execute_web_artifact = AsyncMock(return_value=None)
    tools = [
        {"role": "tool", "name": "file_system",
         "content": "SUCCESS: Wrote 1000 chars to 'index.html'."},
        {"role": "tool", "name": "file_system",
         "content": "SUCCESS: Renamed/Moved 'index.html' to 'index.html.bak'."},
    ]
    v_result, _ = await agent._compute_verifier_verdict(
        tools_run_this_turn=tools,
        messages=[{"role": "user", "content": "build me a page"}],
        final_ai_content="Done, the page is ready.",
        last_user_content="build me a page",
        lc="build me a page",
    )
    agent._execute_web_artifact.assert_awaited()
    assert v_result.confidence == GhostAgent._WEB_EXEC_SKIP_CONF_CAP


async def test_a_retired_scratch_script_does_not_arm_the_cap():
    """The mirror, and the reason the helper narrows to .html/.htm: a
    tidied-up scratch .js must not penalise a turn whose deliverable was
    never a web page."""
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    agent = _verdict_agent(VerifyResult(
        verdict=VerifyVerdict.CONFIRMED, confidence=1.0, reasoning="ok"))
    agent._execute_web_artifact = AsyncMock(return_value=None)
    tools = [
        {"role": "tool", "name": "file_system",
         "content": "SUCCESS: Wrote 90 chars to 'report.md'."},
        {"role": "tool", "name": "file_system",
         "content": "SUCCESS: Wrote 40 chars to 'tmp/scratch.js'."},
        {"role": "tool", "name": "file_system",
         "content": "SUCCESS: Deleted 'tmp/scratch.js'."},
    ]
    v_result, _ = await agent._compute_verifier_verdict(
        tools_run_this_turn=tools,
        messages=[{"role": "user", "content": "write up the report"}],
        final_ai_content="Report written.",
        last_user_content="write up the report",
        lc="write up the report",
    )
    agent._execute_web_artifact.assert_not_awaited()
    assert v_result.confidence == 1.0


# ── FILE-ARTIFACT call-site behaviour (round 4) ───────────────────────
def _fs(content):
    return {"role": "tool", "name": "file_system", "content": content}


def _scope_to(monkeypatch, host_dir):
    monkeypatch.setattr(
        "ghost_agent.tools.file_system.project_scoped_sandbox",
        lambda ctx, stateful=False: (str(host_dir), "/workspace"))


async def _verdict_with(agent, tools, reply="Done."):
    return await agent._compute_verifier_verdict(
        tools_run_this_turn=tools,
        messages=[{"role": "user", "content": "do the thing"}],
        final_ai_content=reply,
        last_user_content="do the thing",
        lc="do the thing",
    )


async def test_a_second_spelling_does_not_evict_a_real_deliverable(
        tmp_path, monkeypatch):
    """⚠ The union deduped on the RAW spelling while the ledger keys are
    normalised, so `/workspace/a.js` and `a.js` each burned one of the eight
    slots. That eviction would silence the 2026-07-19 corrupted-
    index.html guard (it differs on 0 of 1,855 recorded turns — the right
    key for a shape the record has not yet produced, not a measured save) — the exact thing the mutated ∪ claimed union exists
    for. Here index.html is empty and must still be caught."""
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    _scope_to(monkeypatch, tmp_path)
    for n in ("a.js", "b.js", "c.js", "d.js"):
        (tmp_path / n).write_text("ok")
    (tmp_path / "index.html").write_text("")          # the corrupted one
    tools = [_fs(f"SUCCESS: Wrote 5 chars to '/workspace/{n}'.")
             for n in ("a.js", "b.js", "c.js", "d.js")]
    tools.append(_fs("SUCCESS: Wrote 0 chars to 'index.html'."))
    # The prose names the same four files by their BARE spelling, so under a
    # raw-string dedup each file occupies two of the eight slots and pushes
    # the corrupted index.html off the end.
    reply = ("Done. I saved a.js and wrote b.js, then created c.js "
             "and generated d.js.")
    agent = _verdict_agent(VerifyResult(
        verdict=VerifyVerdict.CONFIRMED, confidence=0.98, reasoning="ok"))
    agent._execute_web_artifact = AsyncMock(return_value=None)
    v_result, _ = await _verdict_with(agent, tools, reply=reply)
    assert v_result.verdict == VerifyVerdict.REFUTED
    assert "index.html" in (v_result.reasoning or "")


async def test_the_check_unscopes_to_the_sandbox_root_like_its_sibling(
        tmp_path, monkeypatch):
    """⚠ `_execute_web_artifact` already un-scopes (`sbx.parent.parent if
    sbx.parent.name == "projects"`) because a turn's writes can land at the
    sandbox ROOT while the binding still reads as scoped. This check did not,
    so two checks in the same method disagreed about where the workspace is
    and the scoped one refuted a file plainly on disk."""
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    proj = tmp_path / "projects" / "p1"
    proj.mkdir(parents=True)
    (tmp_path / "deliverable.md").write_text("real content, at the root")
    _scope_to(monkeypatch, proj)
    agent = _verdict_agent(VerifyResult(
        verdict=VerifyVerdict.CONFIRMED, confidence=0.95, reasoning="ok"))
    agent._execute_web_artifact = AsyncMock(return_value=None)
    v_result, _ = await _verdict_with(
        agent, [_fs("SUCCESS: Wrote 25 chars to 'deliverable.md'.")])
    assert v_result.verdict == VerifyVerdict.CONFIRMED
    assert v_result.confidence == 0.95


async def test_a_file_artifact_refute_does_not_discard_an_execution_refute(
        tmp_path, monkeypatch):
    """⚠ `v_result = _fa` was unconditional, so the SOFTER prose-level
    refute replaced the HARDER execution one. The repair directive is built
    from `issues`, so the model was told "a deliverable is missing" and never
    that the page throws on load — and the JS bug shipped."""
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    _scope_to(monkeypatch, tmp_path)
    (tmp_path / "index.html").write_text("<html>ok</html>")
    agent = _verdict_agent(VerifyResult(
        verdict=VerifyVerdict.CONFIRMED, confidence=0.95, reasoning="ok"))
    agent._execute_web_artifact = AsyncMock(
        return_value=("index.html", "SyntaxError: Unexpected token"))
    v_result, _ = await _verdict_with(
        agent,
        [_fs("SUCCESS: Wrote 10 chars to 'index.html'.")],
        reply="Done. I also saved the summary to notes.md.")
    assert v_result.verdict == VerifyVerdict.REFUTED
    joined = "; ".join(v_result.issues or [])
    assert "uncaught JS exception" in joined      # execution evidence kept
    assert "notes.md" in joined                   # and the missing file named


async def test_prose_claims_cannot_starve_the_mutated_file_guard(
        tmp_path, monkeypatch):
    """⚠ `_claimed` was concatenated before `_mutated` and the union then cut
    to 8 — and the prose extractor alone returns up to 8. So a chatty answer
    that named eight present files pushed the ledger arm out entirely, and a
    file this turn CORRUPTED went unchecked because the model had talked
    about other things. LLM prose must not outrank hard tool facts for a
    scarce slot."""
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    _scope_to(monkeypatch, tmp_path)
    names = [f"n{i}.md" for i in range(8)]
    for n in names:
        (tmp_path / n).write_text("present and fine")
    (tmp_path / "index.html").write_text("")          # corrupted this turn
    reply = "Done. " + " ".join(f"I saved {n}." for n in names)
    agent = _verdict_agent(VerifyResult(
        verdict=VerifyVerdict.CONFIRMED, confidence=0.98, reasoning="ok"))
    agent._execute_web_artifact = AsyncMock(return_value=None)
    v_result, _ = await _verdict_with(
        agent, [_fs("SUCCESS: Wrote 0 chars to 'index.html'.")], reply=reply)
    assert v_result.verdict == VerifyVerdict.REFUTED
    assert "index.html" in (v_result.reasoning or "")


async def test_the_merged_issue_survives_every_consumers_slice(
        tmp_path, monkeypatch):
    """⚠ The merge appended and then truncated to 4, while every consumer
    slices lower — `issues[:3]` for the repair directive and the verifier
    note, `[:2]` for the backfill reason. A standing refute with three issues
    therefore lost the file-artifact issue in all of them, which is strictly
    worse than the wholesale replace it was meant to improve on. Slots are
    reserved now."""
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    _scope_to(monkeypatch, tmp_path)
    (tmp_path / "index.html").write_text("<html>ok</html>")
    standing = VerifyResult(
        verdict=VerifyVerdict.CONFIRMED, confidence=0.95, reasoning="ok")
    agent = _verdict_agent(standing)
    agent._execute_web_artifact = AsyncMock(
        return_value=("index.html", "SyntaxError: Unexpected token"))
    v_result, _ = await _verdict_with(
        agent, [_fs("SUCCESS: Wrote 10 chars to 'index.html'.")],
        reply="Done. I also saved the summary to notes.md.")
    # what the repair directive and the backfill reason actually read
    assert "notes.md" in "; ".join(v_result.issues[:2])
    assert "uncaught JS exception" in "; ".join(v_result.issues[:2])


async def test_merging_the_same_refute_twice_is_idempotent(
        tmp_path, monkeypatch):
    """The in-loop verdict and the post-reply async re-verify run over the
    same accumulated tool list; the merge mutates the VerifyResult in place,
    and that object is cached and read by the late-verdict handler."""
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    _scope_to(monkeypatch, tmp_path)
    (tmp_path / "index.html").write_text("<html>ok</html>")
    agent = _verdict_agent(VerifyResult(
        verdict=VerifyVerdict.CONFIRMED, confidence=0.95, reasoning="ok"))
    agent._execute_web_artifact = AsyncMock(
        return_value=("index.html", "SyntaxError: Unexpected token"))
    tools = [_fs("SUCCESS: Wrote 10 chars to 'index.html'.")]
    reply = "Done. I also saved the summary to notes.md."
    first, _ = await _verdict_with(agent, tools, reply=reply)
    n_issues, n_also = len(first.issues), (first.reasoning or "").count("ALSO:")
    agent.context.verifier.verify_claim = AsyncMock(return_value=first)
    second, _ = await _verdict_with(agent, tools, reply=reply)
    assert len(second.issues) == n_issues
    assert (second.reasoning or "").count("ALSO:") == n_also


# ── the merge branch: three properties its first tests could not see ──
def _refuting_agent(confidence, issues):
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    return _verdict_agent(VerifyResult(
        verdict=VerifyVerdict.REFUTED, confidence=confidence,
        reasoning="the text judge's own objection", issues=list(issues)))


async def test_a_sub_threshold_refute_is_replaced_not_promoted(
        tmp_path, monkeypatch):
    """⚠ `max(conf)` on a 0.55 text refute stamps it 0.9 — pushing a verdict
    no gate would have acted on across the repair, backfill and lesson-scrub
    thresholds, carrying ITS issues while the grounded evidence is what gets
    truncated away. Below the gate the grounded result must replace it."""
    _scope_to(monkeypatch, tmp_path)
    agent = _refuting_agent(0.55, ["a weak textual objection"])
    agent._execute_web_artifact = AsyncMock(return_value=None)
    (tmp_path / "ok.md").write_text("present")
    v, _ = await _verdict_with(agent, [_fs("SUCCESS: Wrote 7 chars to 'ok.md'.")],
                               reply="Done. I saved notes.md.")
    joined = "; ".join(v.issues or [])
    assert "notes.md" in joined
    assert "weak textual objection" not in joined


async def test_the_grounded_issue_lands_inside_the_backfill_slice(
        tmp_path, monkeypatch):
    """The backfill reason reads `issues[:2]` and the repair directive
    `issues[:3]`. A standing refute with three issues therefore lost an
    APPENDED file-artifact issue in both."""
    _scope_to(monkeypatch, tmp_path)
    agent = _refuting_agent(0.85, ["claim A unsupported", "claim B hedged",
                                   "no evidence for C"])
    agent._execute_web_artifact = AsyncMock(return_value=None)
    (tmp_path / "ok.md").write_text("present")
    v, _ = await _verdict_with(agent, [_fs("SUCCESS: Wrote 7 chars to 'ok.md'.")],
                               reply="Done. I saved notes.md.")
    assert "notes.md" in "; ".join((v.issues or [])[:2])


async def test_merging_into_the_same_object_twice_changes_nothing(
        tmp_path, monkeypatch):
    """The in-loop verdict and the post-reply re-verify run over the same
    tool list, and the merge mutates the VerifyResult IN PLACE — an object
    that is cached and read again by the late-verdict handler."""
    _scope_to(monkeypatch, tmp_path)
    agent = _refuting_agent(0.85, ["claim A unsupported"])
    agent._execute_web_artifact = AsyncMock(return_value=None)
    (tmp_path / "ok.md").write_text("present")
    tools = [_fs("SUCCESS: Wrote 7 chars to 'ok.md'.")]
    reply = "Done. I saved notes.md."
    first, _ = await _verdict_with(agent, tools, reply=reply)
    n_issues = len(first.issues or [])
    n_also = (first.reasoning or "").count("ALSO:")
    second, _ = await _verdict_with(agent, tools, reply=reply)
    assert len(second.issues or []) == n_issues
    assert (second.reasoning or "").count("ALSO:") == n_also == 1


async def test_both_grounded_issues_reach_the_repair_directive(
        tmp_path, monkeypatch):
    """⚠ Interleaving alone is not the whole property — the ORDER inside it
    decides how much grounded evidence survives `issues[:3]`, which is what
    the repair directive reads. With two grounded issues (one missing, one
    empty) and two textual ones, leading with the grounded pair keeps both;
    leading with the text keeps one."""
    _scope_to(monkeypatch, tmp_path)
    (tmp_path / "hollow.md").write_text("")          # written but empty
    agent = _refuting_agent(0.85, ["claim A unsupported", "claim B hedged"])
    agent._execute_web_artifact = AsyncMock(return_value=None)
    v, _ = await _verdict_with(
        agent, [_fs("SUCCESS: Wrote 0 chars to 'hollow.md'.")],
        reply="Done. I saved notes.md and wrote hollow.md.")
    directive = "; ".join((v.issues or [])[:3])
    assert "notes.md" in directive and "hollow.md" in directive
