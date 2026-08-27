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
    # ⚠ *a/**k, deliberately: this stands in for a real signature, and when
    # that signature grew a keyword the positional-only stub raised
    # TypeError INSIDE the block's blanket except — the check silently
    # stopped running while every test stayed green. Same lesson as the
    # test_verdict_fact_recording stubs.
    monkeypatch.setattr(
        "ghost_agent.tools.file_system.project_scoped_sandbox",
        lambda ctx, *a, **k: (str(host_dir), "/workspace"))


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
    """The ledger-written empty index.html must refute no matter how the
    other writes are spelled. (Historically this pinned an eight-slot
    eviction through raw-spelling dedup in the claimed ∪ mutated union; the
    §4DH demotion removed prose from the hard list and the union with it,
    so the eviction is impossible by construction — the fixture survives as
    a plain corrupted-write pin.)"""
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
        [_fs("SUCCESS: Wrote 10 chars to 'index.html'."),
         _fs("SUCCESS: Wrote 20 chars to 'notes.md'.")],   # never landed
        reply="Done.")
    assert v_result.verdict == VerifyVerdict.REFUTED
    joined = "; ".join(v_result.issues or [])
    assert "uncaught JS exception" in joined      # execution evidence kept
    assert "notes.md" in joined                   # and the missing file named


async def test_prose_claims_cannot_starve_the_mutated_file_guard(
        tmp_path, monkeypatch):
    """⚠ Prose must not outrank hard tool facts for a scarce slot. Two
    designs failed here before the structural one: concatenating claims
    first let 8 prose claims evict the ledger arm entirely, and an
    interleave fixed the eviction while leaving prose in the hard list.
    Prose now rides the soft (emptiness-only) arm, so the hard list is
    ledger-only and starvation is impossible by construction — this pins
    that a corrupted ledger-written file refutes REGARDLESS of how chatty
    the answer is."""
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
        agent, [_fs("SUCCESS: Wrote 10 chars to 'index.html'."),
                _fs("SUCCESS: Wrote 20 chars to 'notes.md'.")],
        reply="Done.")
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
    tools = [_fs("SUCCESS: Wrote 10 chars to 'index.html'."),
             _fs("SUCCESS: Wrote 20 chars to 'notes.md'.")]
    reply = "Done."
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
    v, _ = await _verdict_with(
        agent, [_fs("SUCCESS: Wrote 7 chars to 'ok.md'."),
                _fs("SUCCESS: Wrote 20 chars to 'notes.md'.")],
        reply="Done.")
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
    v, _ = await _verdict_with(
        agent, [_fs("SUCCESS: Wrote 7 chars to 'ok.md'."),
                _fs("SUCCESS: Wrote 20 chars to 'notes.md'.")],
        reply="Done.")
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
    tools = [_fs("SUCCESS: Wrote 7 chars to 'ok.md'."),
             _fs("SUCCESS: Wrote 20 chars to 'notes.md'.")]   # never landed
    reply = "Done."
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
        agent, [_fs("SUCCESS: Wrote 0 chars to 'hollow.md'."),
                _fs("SUCCESS: Wrote 20 chars to 'notes.md'.")],
        reply="Done.")
    directive = "; ".join((v.issues or [])[:3])
    assert "notes.md" in directive and "hollow.md" in directive


# ── round 8: prose demoted to emptiness-only; race-free scoping ───────
async def test_a_prose_only_claim_never_refutes_on_absence(
        tmp_path, monkeypatch):
    """⚠ THE HEADLINE CHANGE. Every false FILE-ARTIFACT refute attested in
    the record was a prose capture refuting on the ABSENCE of a file the
    answer never really produced — on turns that often ran no file tool at
    all. Prose now rides the soft arm: a claimed-but-absent file is not
    evidence of anything, and the text verdict stands."""
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    _scope_to(monkeypatch, tmp_path)
    agent = _verdict_agent(VerifyResult(
        verdict=VerifyVerdict.CONFIRMED, confidence=0.95, reasoning="ok"))
    agent._execute_web_artifact = AsyncMock(return_value=None)
    v, _ = await _verdict_with(
        agent, [{"role": "tool", "name": "web_search",
                 "content": "RESULTS: ..."}],
        reply="As agreed, I saved the board to game_state.json earlier.")
    assert v.verdict == VerifyVerdict.CONFIRMED
    assert v.confidence == 0.95


async def test_a_prose_claimed_EMPTY_file_still_refutes(
        tmp_path, monkeypatch):
    """Prose keeps its teeth for emptiness: a claimed file that exists at 0
    bytes is a real defect the old parse caught, and still is."""
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    _scope_to(monkeypatch, tmp_path)
    (tmp_path / "summary.md").write_text("")
    agent = _verdict_agent(VerifyResult(
        verdict=VerifyVerdict.CONFIRMED, confidence=0.95, reasoning="ok"))
    agent._execute_web_artifact = AsyncMock(return_value=None)
    v, _ = await _verdict_with(
        agent, [{"role": "tool", "name": "web_search",
                 "content": "RESULTS: ..."}],
        reply="Done — I saved the summary to summary.md.")
    assert v.verdict == VerifyVerdict.REFUTED
    assert "summary.md" in "; ".join(v.issues or [])


async def test_a_ledger_written_file_still_refutes_on_absence(
        tmp_path, monkeypatch):
    """The demotion must not defang the ledger arm: a tool confirmation IS
    absence-grade evidence, and a written file that is gone still refutes."""
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    _scope_to(monkeypatch, tmp_path)
    agent = _verdict_agent(VerifyResult(
        verdict=VerifyVerdict.CONFIRMED, confidence=0.95, reasoning="ok"))
    agent._execute_web_artifact = AsyncMock(return_value=None)
    v, _ = await _verdict_with(
        agent, [_fs("SUCCESS: Wrote 20 chars to 'report.md'.")],
        reply="Done.")
    assert v.verdict == VerifyVerdict.REFUTED
    assert "report.md" in "; ".join(v.issues or [])


async def test_the_verdict_scopes_to_the_project_captured_at_spawn(
        tmp_path, monkeypatch):
    """⚠ RACE-FREE SCOPING. The verdict runs in a detached task up to ~60s
    after the turn, and `current_project_id` is process-global — a
    concurrent conversation's reconcile can repoint it mid-flight, resolving
    this turn's deliverables against the WRONG project's directory. The
    gated front door captures the id at spawn; the verdict must use the
    captured value, not re-read the global."""
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    seen = {}

    def _spy(ctx, *a, **k):
        seen.update(k)
        seen["stateful"] = k.get("stateful", a[0] if a else False)
        return (str(tmp_path), "/workspace")

    monkeypatch.setattr(
        "ghost_agent.tools.file_system.project_scoped_sandbox", _spy)
    agent = _verdict_agent(VerifyResult(
        verdict=VerifyVerdict.CONFIRMED, confidence=0.95, reasoning="ok"))
    agent._execute_web_artifact = AsyncMock(return_value=None)
    (tmp_path / "a.md").write_text("x")
    # the global says one thing; the captured value says another
    agent.context.current_project_id = "GLOBAL-DRIFTED"
    await agent._compute_verifier_verdict(
        tools_run_this_turn=[_fs("SUCCESS: Wrote 1 chars to 'a.md'.")],
        messages=[{"role": "user", "content": "x"}],
        final_ai_content="Done.", last_user_content="x", lc="x",
        project_id="captured-at-spawn")
    assert seen.get("explicit_project_id") == "captured-at-spawn"


async def test_a_captured_None_project_means_unscoped_not_reread(
        tmp_path, monkeypatch):
    """'No project at capture' must mean UNSCOPED — passing None through to
    `project_scoped_sandbox` would fall back to the mutable global, which by
    verdict time may point at a project this turn never ran under."""
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    seen = {}

    def _spy(ctx, *a, **k):
        seen.update(k)
        if a:
            seen["stateful"] = a[0]
        return (str(tmp_path), "/workspace")

    monkeypatch.setattr(
        "ghost_agent.tools.file_system.project_scoped_sandbox", _spy)
    agent = _verdict_agent(VerifyResult(
        verdict=VerifyVerdict.CONFIRMED, confidence=0.95, reasoning="ok"))
    agent._execute_web_artifact = AsyncMock(return_value=None)
    (tmp_path / "a.md").write_text("x")
    agent.context.current_project_id = "GLOBAL-DRIFTED"
    await agent._compute_verifier_verdict(
        tools_run_this_turn=[_fs("SUCCESS: Wrote 1 chars to 'a.md'.")],
        messages=[{"role": "user", "content": "x"}],
        final_ai_content="Done.", last_user_content="x", lc="x",
        project_id=None)
    assert seen.get("stateful") is True
    assert "explicit_project_id" not in seen


# ── round 9: every front door threads the captured scope ──────────────
def test_every_verdict_entry_point_threads_a_captured_project_id():
    """⚠ TRIPWIRE, because this exact miss happened three times in one
    round: the captured id was threaded to ONE consumer while the streamed
    spawn (the production-common path), the in-loop repair spawn and both
    timeout attaches kept reading the process-global — one verdict could
    scope its arms to different projects. Every direct call to
    `_compute_verifier_verdict(` must pass `project_id=`, and the verdict
    flow must contain no live `project_scoped_sandbox(self.context)` read.
    A new caller that forgets is exactly the next instance of the bug."""
    import inspect
    import re as _re
    from pathlib import Path as _P
    src = (_P(inspect.getfile(GhostAgent))).read_text(encoding="utf-8")
    # every direct call passes project_id= within its argument block
    for m in _re.finditer(r"self\._compute_verifier_verdict\(", src):
        window = src[m.end():m.end() + 800]
        args_block = window[: window.find(")\n") + 2 if ")\n" in window
                            else len(window)]
        assert ("project_id=" in args_block
                and "project_id=_PROJECT_ID_UNCAPTURED" not in args_block
                # ⚠ the live-global SPELLING is not a capture — an audit
                # showed `project_id=self.context.current_project_id` passed
                # this scan while being exactly the race it forbids.
                and "current_project_id" not in args_block), (
            "a _compute_verifier_verdict call site does not thread a "
            "CAPTURED project_id — passing nothing (or the sentinel, which "
            "is the same thing spelled louder) re-reads the process-global "
            "inside a detached task:\n"
            + src[max(0, m.start() - 200): m.end() + 300])
    # a REFERENCE to the method (aliasing, functools.partial) would evade
    # the call-site scan above entirely — forbid taking one at all: every
    # occurrence of the attribute must be an immediate call.
    for m in _re.finditer(
            r"self\._compute_verifier_verdict\b(?!\()"
            r"|type\(self\)\._compute_verifier_verdict"
            r"|['\"]_compute_verifier_verdict['\"]", src):
        raise AssertionError(
            "a bare reference to _compute_verifier_verdict (alias/partial?) "
            "evades the project-id tripwire — call it directly:\n"
            + src[max(0, m.start() - 200): m.end() + 200])
    # and the verdict arms themselves never read the global directly
    for fn in (GhostAgent._compute_verifier_verdict,
               GhostAgent._execute_web_artifact):
        body = inspect.getsource(fn)
        assert "project_scoped_sandbox(self.context)" not in body, (
            f"{fn.__name__} scopes through a live global read — use "
            f"_scoped_sandbox_for(project_id)")


async def test_the_gated_front_door_captures_and_threads_the_global(
        tmp_path, monkeypatch):
    """Behavioral half of the tripwire: drive the GATED door and assert the
    value it captured reaches the sandbox scoping as explicit_project_id."""
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    seen = {}

    def _spy(ctx, *a, **k):
        seen.update(k)
        return (str(tmp_path), "/workspace")

    monkeypatch.setattr(
        "ghost_agent.tools.file_system.project_scoped_sandbox", _spy)
    agent = _verdict_agent(VerifyResult(
        verdict=VerifyVerdict.CONFIRMED, confidence=0.95, reasoning="ok"))
    agent._execute_web_artifact = AsyncMock(return_value=None)
    agent._critic_gate_timeout = lambda: float("inf")
    agent._critic_async_enabled = lambda: False
    agent.context.current_project_id = "pid-at-spawn"
    agent.context.args = None
    (tmp_path / "a.md").write_text("x")
    await agent._compute_verifier_verdict_gated(
        tools_run_this_turn=[_fs("SUCCESS: Wrote 1 chars to 'a.md'.")],
        messages=[{"role": "user", "content": "x"}],
        final_ai_content="Done.", last_user_content="x", lc="x",
        trajectory_id="")
    assert seen.get("explicit_project_id") == "pid-at-spawn"


async def test_a_stomped_global_is_healed_from_the_conversation_binding(
        tmp_path, monkeypatch):
    """⚠ A global cleared mid-request by a concurrent reconcile used to be
    healed inside project_scoped_sandbox at resolve time; committing a
    captured None straight to the raw root would regress that. The capture
    heals through `_conversation_bound_project` while conversation_key is
    still this turn's own."""
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    seen = {}

    def _spy(ctx, *a, **k):
        seen.update(k)
        return (str(tmp_path), "/workspace")

    monkeypatch.setattr(
        "ghost_agent.tools.file_system.project_scoped_sandbox", _spy)
    monkeypatch.setattr(
        "ghost_agent.tools.file_system._conversation_bound_project",
        lambda ctx: "healed-pid")
    agent = _verdict_agent(VerifyResult(
        verdict=VerifyVerdict.CONFIRMED, confidence=0.95, reasoning="ok"))
    agent._execute_web_artifact = AsyncMock(return_value=None)
    agent._critic_gate_timeout = lambda: float("inf")
    agent._critic_async_enabled = lambda: False
    agent.context.current_project_id = None       # stomped
    agent.context.args = None
    (tmp_path / "a.md").write_text("x")
    await agent._compute_verifier_verdict_gated(
        tools_run_this_turn=[_fs("SUCCESS: Wrote 1 chars to 'a.md'.")],
        messages=[{"role": "user", "content": "x"}],
        final_ai_content="Done.", last_user_content="x", lc="x",
        trajectory_id="")
    assert seen.get("explicit_project_id") == "healed-pid"


async def test_web_exec_scopes_by_the_captured_id_not_the_global(
        tmp_path, monkeypatch):
    """⚠ The structural pin alone was gameable: routing through the helper
    with the SENTINEL still reads the global. Drive the real
    `_execute_web_artifact` and assert the sandbox scoping received the
    captured id."""
    seen = {}

    def _spy(ctx, *a, **k):
        seen.update(k)
        return (str(tmp_path), "/workspace")

    monkeypatch.setattr(
        "ghost_agent.tools.file_system.project_scoped_sandbox", _spy)
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = MagicMock()
    agent.available_tools = {"browser": AsyncMock()}
    await agent._execute_web_artifact(["index.html"],
                                      project_id="captured-web")
    assert seen.get("explicit_project_id") == "captured-web"


async def test_the_removal_downgrade_leaves_a_durable_trace(
        tmp_path, monkeypatch):
    """⚠ the downgrade decides verdicts; a mechanism whose firings live only
    in a one-day log is the exact "no durable record" defect the override
    provenance was built to close. The skip list rides the VerifyResult into
    the sidecar."""
    from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
    _scope_to(monkeypatch, tmp_path)
    agent = _verdict_agent(VerifyResult(
        verdict=VerifyVerdict.CONFIRMED, confidence=0.95, reasoning="ok"))
    agent._execute_web_artifact = AsyncMock(return_value=None)
    v, _ = await _verdict_with(
        agent,
        [_fs("SUCCESS: Wrote 9 chars to 'probe.py'."),
         {"role": "tool", "name": "execute",
          "content": "--- COMMAND RESULT ---\nEXIT CODE: 0"}],
        reply="Done.")
    assert v.verdict == VerifyVerdict.CONFIRMED
    assert getattr(v, "skipped_removable", None) == ["probe.py"]
