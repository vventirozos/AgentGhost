"""The workspace tool's `search` action (alias `recall`).

Models repeatedly guessed ``workspace{action:"search"}`` — a failure
strike followed by a recovery turn (journal §4C). The guess is now a
real action: IDF-weighted keyword search over the workspace activity
log (``WorkspaceActivity.search``), mirroring selfhood's
``search_my_past``.
"""

import pytest

from ghost_agent.tools.registry import TOOL_DEFINITIONS
from ghost_agent.tools.workspace import _VALID_ACTIONS, tool_workspace
from ghost_agent.workspace import WorkspaceModel, pinned_event_project


@pytest.fixture
def wm(tmp_path):
    m = WorkspaceModel(tmp_path, enabled=True)
    m.note("deployed the falcon api server")
    m.note("backup ran with restic to the coldstore vault")
    m.record_command_outcome(
        command="python train.py --epochs 5", exit_code=0,
        duration_seconds=42.0,
    )
    m.record_research_artifact(
        url="https://example.com/pg20-release", title="PG20 notes",
        source="deep_research",
    )
    return m


# ---------------------------------------------------------------------------
# WorkspaceActivity.search scoring
# ---------------------------------------------------------------------------

def test_search_finds_by_summary_keyword(wm):
    hits = wm.activity.search("restic backup")
    assert hits and "restic" in hits[0].summary


def test_search_matches_payload_values(wm):
    # "train" only appears inside the command payload, not the summary...
    hits = wm.activity.search("train epochs")
    assert hits and hits[0].kind == "command"
    # ...and URL components match too.
    hits = wm.activity.search("pg20 release")
    assert hits and hits[0].kind == "research"


def test_search_rare_term_dominates(tmp_path):
    m = WorkspaceModel(tmp_path, enabled=True)
    for i in range(5):
        m.note(f"server maintenance pass {i}")
    m.note("server maintenance touched the zephyrium module")
    hits = m.activity.search("server zephyrium")
    assert "zephyrium" in hits[0].summary


def test_search_empty_and_short_queries(wm):
    assert wm.activity.search("") == []
    assert wm.activity.search("a b") == []  # tokens <= 2 chars are dropped
    assert wm.activity.search("nonexistentzz") == []


def test_search_events_facade_never_raises_when_disabled(tmp_path):
    m = WorkspaceModel(tmp_path, enabled=False)
    assert m.search_events("anything") == []


# ---------------------------------------------------------------------------
# tool dispatch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_action_search_happy_path(wm):
    out = await tool_workspace(action="search", query="restic", workspace_model=wm)
    assert "restic" in out and "SYSTEM ERROR" not in out


@pytest.mark.asyncio
async def test_action_recall_alias(wm):
    out = await tool_workspace(action="recall", query="falcon", workspace_model=wm)
    assert "falcon" in out and "SYSTEM ERROR" not in out


@pytest.mark.asyncio
async def test_search_result_carries_project_tag(tmp_path):
    m = WorkspaceModel(tmp_path, enabled=True)
    with pinned_event_project("abc123def456"):
        m.note("wrote the kestrel deployment script")
    out = await tool_workspace(action="search", query="kestrel", workspace_model=m)
    assert "abc123def456" in out


@pytest.mark.asyncio
async def test_search_without_query_names_the_missing_arg(wm):
    out = await tool_workspace(action="search", workspace_model=wm)
    assert "SYSTEM ERROR" in out and "query" in out


@pytest.mark.asyncio
async def test_search_accepts_near_miss_arg_names(wm):
    out = await tool_workspace(action="search", q="restic", workspace_model=wm)
    assert "restic" in out and "SYSTEM ERROR" not in out


@pytest.mark.asyncio
async def test_search_no_match_redirects_to_recall_tool(wm):
    out = await tool_workspace(
        action="search", query="qqqzzzunmatchable", workspace_model=wm,
    )
    assert "No workspace events match" in out
    assert "recall" in out  # nudge: stored facts/documents live elsewhere


@pytest.mark.asyncio
async def test_unknown_action_still_strikes(wm):
    out = await tool_workspace(action="bogus", workspace_model=wm)
    assert "SYSTEM ERROR" in out


# ---------------------------------------------------------------------------
# The two sources of truth (schema enum vs dispatch frozenset) must agree
# ---------------------------------------------------------------------------

def test_schema_advertises_search_and_query():
    ws = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "workspace")
    params = ws["function"]["parameters"]["properties"]
    assert "search" in params["action"]["enum"]
    assert "query" in params
    assert "'search'" in ws["function"]["description"]


def test_every_advertised_action_dispatches():
    ws = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "workspace")
    enum = set(ws["function"]["parameters"]["properties"]["action"]["enum"])
    assert enum <= _VALID_ACTIONS, (
        f"schema advertises actions the dispatcher rejects: {enum - _VALID_ACTIONS}"
    )
