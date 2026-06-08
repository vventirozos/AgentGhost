"""Tests for project-scoping the workspace wake-up prefix.

Root-cause fix for the cross-project bleed: the workspace model is global
(one activity log + narrative across all projects), so a fresh project's
wake-up prefix inherited a PRIOR project's `file://` research pulls
("recent pulls of index.html v6,7,8"). The model then chased a file that
only existed in the other project's sandbox — an infinite loop. Events are
now scoped to the active project, deriving an event's project from its
explicit stamp OR its `projects/<id>/` path (so existing on-disk data is
fixed without a migration).
"""
import pytest

from ghost_agent.workspace.activity import WorkspaceActivity
from ghost_agent.workspace.model import WorkspaceModel
from ghost_agent.workspace.recognition import (
    build_workspace_prefix,
    _narrative_is_cross_project,
)
from ghost_agent.workspace.schema import (
    WorkspaceEvent,
    derive_event_project_id,
    filter_events_for_project,
)

OLD = "991e52bce912"
NEW = "866dabcd1234"
OLD_URL = f"file:///workspace/projects/{OLD}/minecraft-clone/index.html?v=6"


# ── derivation ───────────────────────────────────────────────────────
def test_derive_prefers_explicit_stamp():
    ev = WorkspaceEvent(kind="note", project_id="ABCdef123456", summary="x")
    assert derive_event_project_id(ev) == "abcdef123456"  # lowercased


def test_derive_from_file_url_legacy_event():
    ev = WorkspaceEvent(kind="research", summary=f"pulled {OLD_URL}",
                        payload={"url": OLD_URL})
    assert derive_event_project_id(ev) == OLD


def test_web_research_is_project_agnostic():
    ev = WorkspaceEvent(kind="research", summary="pulled https://docs.python.org",
                        payload={"url": "https://docs.python.org"})
    assert derive_event_project_id(ev) == ""


def test_schema_roundtrip_preserves_project_id():
    ev = WorkspaceEvent(kind="note", project_id=NEW, summary="hi")
    assert WorkspaceEvent.from_dict(ev.to_dict()).project_id == NEW


# ── filtering ────────────────────────────────────────────────────────
def test_filter_drops_other_project_keeps_agnostic_and_self():
    evs = [
        WorkspaceEvent(kind="research", summary=f"pulled {OLD_URL}", payload={"url": OLD_URL}),
        WorkspaceEvent(kind="research", summary="pulled https://x.com", payload={"url": "https://x.com"}),
        WorkspaceEvent(kind="note", project_id=NEW, summary="mine"),
    ]
    kept = filter_events_for_project(evs, NEW)
    summaries = [e.summary for e in kept]
    assert "mine" in summaries
    assert any("x.com" in s for s in summaries)
    assert not any(OLD in s for s in summaries)   # other project dropped


def test_filter_no_active_project_keeps_everything():
    evs = [WorkspaceEvent(kind="research", summary=f"pulled {OLD_URL}", payload={"url": OLD_URL})]
    assert len(filter_events_for_project(evs, "")) == 1
    assert len(filter_events_for_project(evs, None)) == 1


# ── narrative suppression ────────────────────────────────────────────
def test_cross_project_narrative_detected():
    assert _narrative_is_cross_project(f"I pulled projects/{OLD}/index.html", NEW) is True
    assert _narrative_is_cross_project(f"I worked on projects/{NEW}/app.js", NEW) is False
    assert _narrative_is_cross_project("just chatting, no files", NEW) is False
    assert _narrative_is_cross_project(f"projects/{OLD}/x", "") is False  # no active → no scope


# ── build_workspace_prefix integration ───────────────────────────────
def test_prefix_hides_other_projects_pulls(tmp_path):
    act = WorkspaceActivity(tmp_path)
    act.append(WorkspaceEvent(kind="research", summary=f"pulled {OLD_URL}",
                              payload={"url": OLD_URL}))
    out = build_workspace_prefix(activity=act, state=None, narrative="",
                                 active_project_id=NEW)
    assert OLD not in out and "index.html" not in out


def test_prefix_shows_own_project_pulls(tmp_path):
    act = WorkspaceActivity(tmp_path)
    act.append(WorkspaceEvent(kind="research", summary=f"pulled {OLD_URL}",
                              payload={"url": OLD_URL}))
    out = build_workspace_prefix(activity=act, state=None, narrative="",
                                 active_project_id=OLD)
    assert "index.html" in out   # same project → shown


def test_prefix_suppresses_cross_project_narrative(tmp_path):
    act = WorkspaceActivity(tmp_path)
    narrative = f"My running summary: I pulled projects/{OLD}/minecraft-clone/index.html three times."
    out = build_workspace_prefix(activity=act, state=None, narrative=narrative,
                                 active_project_id=NEW)
    assert OLD not in out


# ── WorkspaceModel end-to-end ────────────────────────────────────────
def test_model_stamps_and_scopes(tmp_path):
    wm = WorkspaceModel(tmp_path, enabled=True)
    wm.current_project_id = OLD
    art = wm.record_research_artifact(url=OLD_URL, source="browser", note="v6")
    assert art is not None
    # A new project's prefix must NOT inherit the old project's pull.
    out_new = wm.build_wakeup_prefix(active_project_id=NEW)
    assert OLD not in out_new
    # The owning project still sees it.
    out_old = wm.build_wakeup_prefix(active_project_id=OLD)
    assert "index.html" in out_old
