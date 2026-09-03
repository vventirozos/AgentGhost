"""Main-loop research write-back (2026-09-03, §4EK).

Live (request 463111ad): the main loop ran six web searches about the
project and threw the results away — only the conversation saw them, while
the coding leaves read the project's research/ files and nothing else. They
built from a 3 KB brief that said "talismans and stat allocation are not
covered" and shipped wrong facts.

Now every RELEVANT main-loop web_search is written to
``research/main-loop-findings.md`` (newest first, a repeated query replaces
its entry, the newest N survive), the research index is kept in step, and
``_gather_research_briefs`` — what the coding executor actually reads —
picks it up. Autonomous leaves (pinned project contexts) are excluded: their
research is already saved as a brief. The relevance verdict has ONE
authority, shared with the work-log gate.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.core import project_research as pr
from ghost_agent.core.project_research import (
    MAIN_LOOP_FINDINGS_MAX_SEARCHES,
    MAIN_LOOP_FINDINGS_SLUG,
    get_research_index,
    parse_search_results,
    record_main_loop_findings,
    request_relevant_to_project,
)
from ghost_agent.core.project_advancer import (
    _gather_research_briefs,
    pinned_project_context,
)
import ghost_agent.tools.search as search_mod


SEARCH_OUT = (
    "### 1. Somberstone Miner's Bell Bearing locations\n"
    "All five Somberstone bearings and where each one drops, region by region.\n"
    "[Source: https://example.org/somber-bearings]\n\n"
    "### 2. Miner's Bell Bearing guide\n"
    "The four regular bearings for standard smithing stones.\n"
    "[Source: https://example.org/miner-bearings]\n\n"
    "### 3. Junk without a link\nnothing\n[Source: #]"
)


@pytest.fixture
def store(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    return ProjectStore(tmp_path / "memory", sandbox_root=sandbox)


@pytest.fixture
def pid(store):
    p = store.create_project("Elden Ring Blasphemous Build Tracker", kind="CODING",
                             goal="Track bell bearings, talismans and stats")
    store.add_task(p, "Build the Somberstone bell bearings section with wiki links")
    return p


def _findings(store, pid):
    path = store.ensure_workspace(pid) / "research" / "main-loop-findings.md"
    return path.read_text(encoding="utf-8") if path.exists() else None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_parse_search_results_reads_the_tool_shape_and_drops_linkless_rows():
    rows = parse_search_results(SEARCH_OUT)
    assert [r["url"] for r in rows] == ["https://example.org/somber-bearings",
                                        "https://example.org/miner-bearings"]
    assert rows[0]["title"].startswith("Somberstone")
    assert parse_search_results("CRITICAL ERROR: 'ddgs' library is missing.") == []
    assert parse_search_results("") == []


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

def test_relevant_search_is_written_newest_first_with_sources(store, pid):
    rel = record_main_loop_findings(store, pid, "Somberstone bell bearing locations",
                                    SEARCH_OUT, ts=1_000)
    assert rel == "research/main-loop-findings.md"
    rel2 = record_main_loop_findings(store, pid, "best talismans for bleed build",
                                     SEARCH_OUT, ts=2_000)
    assert rel2 == rel
    text = _findings(store, pid)
    assert text.startswith("# Main-loop search findings (auto)")
    assert text.index("best talismans") < text.index("Somberstone bell bearing locations")
    assert "https://example.org/somber-bearings" in text
    assert "https://example.org/miner-bearings" in text
    assert "[Source: #]" not in text and "Junk without a link" not in text
    # Index kept in step, on both surfaces.
    idx = get_research_index(store, pid)
    entry = next(e for e in idx if e.get("slug") == MAIN_LOOP_FINDINGS_SLUG)
    assert entry["path"] == rel and "best talismans" in entry["summary_preview"]
    index_md = (store.ensure_workspace(pid) / "research" / "INDEX.md").read_text()
    assert rel in index_md


def test_off_topic_search_is_not_recorded(store, pid):
    assert record_main_loop_findings(
        store, pid, "weather forecast Athens tomorrow", SEARCH_OUT) is None
    assert _findings(store, pid) is None


def test_error_and_empty_outputs_are_not_recorded(store, pid):
    assert record_main_loop_findings(
        store, pid, "Somberstone bell bearing", "CRITICAL ERROR: search failed") is None
    assert record_main_loop_findings(store, pid, "Somberstone bell bearing", "") is None
    assert _findings(store, pid) is None


def test_repeated_query_replaces_its_own_entry(store, pid):
    for ts in (1, 2, 3):
        record_main_loop_findings(store, pid, "somberstone bell bearing locations",
                                  SEARCH_OUT, ts=ts * 1_000)
    text = _findings(store, pid)
    assert text.count("\n## ") == 1


def test_only_the_newest_searches_survive(store, pid):
    n = MAIN_LOOP_FINDINGS_MAX_SEARCHES + 4
    for i in range(n):
        record_main_loop_findings(store, pid, f"bell bearing query number {i}",
                                  SEARCH_OUT, ts=1_000 + i)
    text = _findings(store, pid)
    assert text.count("\n## ") == MAIN_LOOP_FINDINGS_MAX_SEARCHES
    assert f"query number {n - 1}" in text          # newest kept
    assert "query number 0 " not in text and "query number 0\n" not in text


def test_the_coding_executor_gatherer_picks_it_up(store, pid):
    record_main_loop_findings(store, pid, "Somberstone bell bearing locations",
                              SEARCH_OUT, ts=1_000)
    briefs = _gather_research_briefs(store, pid)
    assert "research/main-loop-findings.md" in briefs
    excerpt = briefs["research/main-loop-findings.md"]
    assert "Somberstone bell bearing locations" in excerpt
    assert "https://example.org/somber-bearings" in excerpt


# ---------------------------------------------------------------------------
# One authority for relevance
# ---------------------------------------------------------------------------

def test_agent_gate_delegates_to_the_shared_authority(store, pid, monkeypatch):
    from ghost_agent.core.agent import GhostAgent
    fake_self = SimpleNamespace(context=SimpleNamespace(_project_work_cmds=[]))
    assert GhostAgent._request_relevant_to_project(
        fake_self, store, pid, "add the bell bearings section") is True
    assert GhostAgent._request_relevant_to_project(
        fake_self, store, pid, "weather forecast Athens tomorrow") is False
    # Executed delegation: flip the authority and the agent's answer flips.
    monkeypatch.setattr(pr, "request_relevant_to_project", lambda *a, **k: False)
    assert GhostAgent._request_relevant_to_project(
        fake_self, store, pid, "add the bell bearings section") is False


def test_shared_authority_honours_project_dir_in_commands(store, pid):
    # An off-topic request text, but a command that names the project dir.
    assert request_relevant_to_project(
        store, pid, "please rerun that thing", cmds=[f"cd projects/{pid} && ls"]) is True
    assert request_relevant_to_project(
        store, pid, "please rerun that thing", cmds=["ls"]) is False
    # No significant token at all → False before the store is consulted.
    assert request_relevant_to_project(store, pid, "do it", cmds=[f"cd projects/{pid}"]) is False


# ---------------------------------------------------------------------------
# The tool hook
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_live_project_search_writes_back_and_says_so(store, pid, monkeypatch):
    async def fake_ddgs(query, tor_proxy):
        return SEARCH_OUT
    monkeypatch.setattr(search_mod, "tool_search_ddgs", fake_ddgs)
    ctx = SimpleNamespace(current_project_id=pid, project_store=store)
    out = await search_mod.tool_search(query="Somberstone bell bearing locations",
                                       context=ctx)
    assert SEARCH_OUT in out
    assert "saved to research/main-loop-findings.md" in out
    assert _findings(store, pid) is not None


@pytest.mark.asyncio
async def test_pinned_leaf_and_no_context_do_not_write_back(store, pid, monkeypatch):
    async def fake_ddgs(query, tor_proxy):
        return SEARCH_OUT
    monkeypatch.setattr(search_mod, "tool_search_ddgs", fake_ddgs)
    base = SimpleNamespace(current_project_id=None, project_store=store)
    pinned = pinned_project_context(base, pid)
    assert pinned.current_project_id == pid            # it IS project-scoped…
    out = await search_mod.tool_search(query="Somberstone bell bearing locations",
                                       context=pinned)
    assert "saved to" not in out                        # …but autonomous: no write-back
    assert _findings(store, pid) is None
    out = await search_mod.tool_search(query="Somberstone bell bearing locations")
    assert "saved to" not in out
    assert _findings(store, pid) is None


@pytest.mark.asyncio
async def test_off_topic_live_search_leaves_no_trace_in_the_output(store, pid, monkeypatch):
    async def fake_ddgs(query, tor_proxy):
        return SEARCH_OUT
    monkeypatch.setattr(search_mod, "tool_search_ddgs", fake_ddgs)
    ctx = SimpleNamespace(current_project_id=pid, project_store=store)
    out = await search_mod.tool_search(query="weather forecast Athens tomorrow", context=ctx)
    assert out == SEARCH_OUT
    assert _findings(store, pid) is None


@pytest.mark.asyncio
async def test_registry_hands_the_live_context_to_web_search(monkeypatch):
    """The wiring, executed: build the real tool map from a mock context and
    check that the web_search entry reaches tool_search WITH that context
    (a substring pin over the registry source was proven vacuous before —
    see test_autoadvance_project_scope.TestCallersArePinned)."""
    from unittest.mock import MagicMock
    from ghost_agent.tools import registry as reg
    ctx = MagicMock()
    ctx.args.anonymous = False
    ctx.tor_proxy = None
    tools = reg.get_available_tools(ctx)
    seen = {}

    async def rec(**kw):
        seen.update(kw)
        return "ok"
    monkeypatch.setattr(reg, "tool_search", rec)
    assert await tools["web_search"](query="x") == "ok"
    assert seen.get("context") is ctx and seen.get("query") == "x"
