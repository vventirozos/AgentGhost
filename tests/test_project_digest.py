"""Tests for the autonomous-progress digest (user-facing half of phase 2.95)."""

import pytest

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.core.project_digest import (
    DigestResult,
    summarize_since,
    render_digest,
    load_watermark,
    save_watermark,
)


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


# ──────────────────────────────────────────────────────────────────────
# summarize_since
# ──────────────────────────────────────────────────────────────────────

def test_summarize_empty(store):
    res = summarize_since(store, 0)
    assert res.has_content is False
    assert res.advanced == 0 and res.needs_user == []


def test_summarize_counts_advances_and_needs_user(store):
    pid = store.create_project("My Project")
    tid = store.add_task(pid, "build the thing")
    store.log_event(pid, tid, "autoadvance_step", {})
    store.log_event(pid, tid, "autoadvance_needs_user", {"description": "approve the deploy"})
    store.log_event(pid, None, "some_unrelated_event", {})  # advances watermark, not counted
    res = summarize_since(store, 0)
    assert res.advanced == 1
    assert res.projects_touched == 1
    assert len(res.needs_user) == 1
    assert res.needs_user[0] == ("My Project", "approve the deploy")
    assert res.new_event_id > 0


def test_needs_user_desc_falls_back_to_task(store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "the original task description")
    # human_gate_triggered carries no description in payload → fall back to task
    store.log_event(pid, tid, "human_gate_triggered", {"reason": "deletes prod"})
    res = summarize_since(store, 0)
    assert len(res.needs_user) == 1
    title, desc = res.needs_user[0]
    assert desc == "the original task description"


def test_watermark_gates_repeats(store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "t")
    store.log_event(pid, tid, "autoadvance_step", {})
    first = summarize_since(store, 0)
    assert first.advanced == 1
    # Re-scan from the new watermark → nothing new.
    again = summarize_since(store, first.new_event_id)
    assert again.has_content is False
    assert again.advanced == 0
    # A fresh advance after the watermark is picked up.
    store.log_event(pid, tid, "autoadvance_step", {})
    third = summarize_since(store, first.new_event_id)
    assert third.advanced == 1


def test_needs_user_project_rolled_same_tick_still_surfaces(store):
    # Regression: the advancer's tick rolls the project to NEEDS_USER in
    # the same batch that logs the needs-user event; the ACTIVE-only scan
    # hid exactly that batch, so the user was never told input is needed.
    pid = store.create_project("Waiting Proj")
    tid = store.add_task(pid, "approve the schema change")
    store.log_event(pid, tid, "autoadvance_needs_user",
                    {"description": "approve the schema change"})
    store.update_task(tid, status="NEEDS_USER")  # rolls project → NEEDS_USER
    assert store.get_project(pid)["status"] == "NEEDS_USER"
    res = summarize_since(store, 0)
    assert len(res.needs_user) == 1
    assert res.needs_user[0] == ("Waiting Proj", "approve the schema change")
    assert "need your input" in render_digest(res)


def test_done_project_terminal_batch_still_counted(store):
    # The steps that finished a project must not vanish from the digest
    # just because the project is no longer ACTIVE when the digest runs.
    pid = store.create_project("Ship It")
    tid = store.add_task(pid, "final step")
    store.log_event(pid, tid, "autoadvance_step", {})
    store.update_task(tid, status="DONE")  # rolls project → DONE
    assert store.get_project(pid)["status"] == "DONE"
    res = summarize_since(store, 0)
    assert res.advanced == 1
    assert ("Ship It", "DONE") in res.finished
    out = render_digest(res)
    assert "Ship It" in out and "DONE" in out


def test_failed_project_rollup_surfaces(store):
    pid = store.create_project("Doomed")
    tid = store.add_task(pid, "impossible step")
    store.update_task(tid, status="FAILED")  # rolls project → FAILED
    assert store.get_project(pid)["status"] == "FAILED"
    res = summarize_since(store, 0)
    assert ("Doomed", "FAILED") in res.finished
    assert res.has_content is True


def test_new_event_id_advances_past_irrelevant_events(store):
    pid = store.create_project("P")
    store.log_event(pid, None, "noise_a", {})
    store.log_event(pid, None, "noise_b", {})
    res = summarize_since(store, 0)
    assert res.has_content is False         # nothing surfaced
    assert res.new_event_id > 0             # but watermark moved past the noise


# ──────────────────────────────────────────────────────────────────────
# render_digest
# ──────────────────────────────────────────────────────────────────────

def test_render_empty():
    assert render_digest(DigestResult()) == ""


def test_render_advanced_only():
    out = render_digest(DigestResult(advanced=2, projects_touched=1, new_event_id=5))
    assert "While you were away" in out
    assert "2 task(s)" in out
    assert "need your input" not in out


def test_render_needs_user_caps_and_counts():
    nu = [("Proj", f"task {i}") for i in range(5)]
    out = render_digest(DigestResult(advanced=1, projects_touched=1, needs_user=nu),
                        max_needs_user=3)
    assert "5 now need your input" in out
    assert "task 0" in out and "task 2" in out
    assert "task 4" not in out          # capped
    assert "and 2 more" in out


# ──────────────────────────────────────────────────────────────────────
# watermark persistence
# ──────────────────────────────────────────────────────────────────────

def test_watermark_absent_is_none(tmp_path):
    assert load_watermark(tmp_path / "wm.json") is None


def test_watermark_roundtrip(tmp_path):
    p = tmp_path / "wm.json"
    save_watermark(p, 42)
    assert load_watermark(p) == 42


def test_watermark_save_never_raises(tmp_path):
    # parent is a file → save can't create the dir; must swallow, not raise.
    bad = tmp_path / "afile"
    bad.write_text("x")
    save_watermark(bad / "wm.json", 7)  # must not raise
