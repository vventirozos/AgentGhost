"""§4UO (2026-11-14, req 671eea0c): (1) the wake turn of a job the owner's request started edited the
project's browser.js four times and the change log never heard of it — a wake was neither the owner's turn
nor a project task's; (2) the delivered reply opened with two 566/571-char working notes written beside
`browser` checks, kept by the 400-char lookup cap although the turn was a build. Every test names the
world it fails in."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.core.reply_smoothing import WorkSegment, drop_pre_tool_segments
from ghost_agent.memory import projects as MP
from ghost_agent.memory.projects import ProjectStore
from ghost_agent.memory.scratchpad import Scratchpad
from ghost_agent.sandbox import jobs as sbx_jobs
from ghost_agent.tools import projects as TP
from ghost_agent.tools.projects import note_project_write
from ghost_agent.utils.logging import request_id_context
from tests.test_4um_living_projects import _built
from tests.test_job_landing_report_drain import _FakeSandbox

JID = "job-d36bc23e"


@pytest.fixture(autouse=True)
def _clean_pending():
    MP._PENDING_CHANGES.clear()
    yield
    MP._PENDING_CHANGES.clear()


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


def _ctx(tmp_path, store, starter):
    sbx = _FakeSandbox(tmp_path / "sandbox")
    sup = sbx_jobs.SandboxJobSupervisor(sbx)
    sbx._job_supervisor = sup
    sup._record_launch(JID, starter, "bash -c 'check …'")
    return SimpleNamespace(project_store=store, scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
                           graph_memory=None, contradiction_log=None, current_project_id=None,
                           llm_client=None, workspace_model=None, sandbox_dir=str(tmp_path / "sb"),
                           sandbox_manager=sbx,
                           last_user_content=f"SYSTEM: the background job {JID} you started has finished")


def _write_as(ctx, ws, rel, text, rid):
    tok = request_id_context.set(rid)
    try:
        (ws / rel).parent.mkdir(parents=True, exist_ok=True)
        (ws / rel).write_text(text)
        return note_project_write(ctx, ws / rel)
    finally:
        request_id_context.reset(tok)


# ── (1) the wake of the owner's job is logged ────────────────────────────────
def test_the_wake_of_an_owner_job_is_a_change_entry_saying_it_was_unattended(tmp_path, store):
    """Fails where the wake's edits reached the folder with no entry (#8 never written)."""
    pid, ws = _built(store)
    ctx = _ctx(tmp_path, store, starter="671eea0c00000000")
    store.add_change(pid, req_id="671eea0c00000000", ask="add a working browser to webos",
                     added=["js/apps/browser.js"], summary="Done")
    _write_as(ctx, ws, "js/apps/browser.js", "v2", JID)
    assert MP.flush_change_log(JID, reply="Tightened the blocked-site hint.") == 1
    e = store.list_changes(pid)[0]
    assert e["req_id"] == JID and e["changed"] == ["js/apps/browser.js"]
    assert e["ask"].startswith(f"(unattended follow-up after background job {JID} finished)")
    assert "add a working browser to webos" in e["ask"] and "SYSTEM:" not in e["ask"]
    assert ctx.current_project_id is None                 # a wake has no conversation focus to set


@pytest.mark.parametrize("starter", ["sched-abc123", "sub-xyz", "probe-1", ""])
def test_the_wake_of_a_background_or_probe_job_logs_nothing(tmp_path, store, starter):
    pid, ws = _built(store)
    ctx = _ctx(tmp_path, store, starter=starter)
    _write_as(ctx, ws, "x.js", "1", JID)
    MP.flush_change_log(JID, reply="")
    assert store.list_changes(pid) == []


def test_a_sub_job_wake_of_a_scheduled_tasks_job_never_counts(tmp_path, store):
    """A job a scheduled task started wakes as `sub-job-…` and stays unlogged."""
    ctx = _ctx(tmp_path, store, starter="sched-abc123")
    tok = request_id_context.set(f"sub-{JID}")
    try:
        assert TP._job_wake_starter(ctx) == "" and not TP._owner_change_turn(ctx)
    finally:
        request_id_context.reset(tok)


def test_a_simulated_wake_logs_nothing(tmp_path, store):
    ctx = _ctx(tmp_path, store, starter="671eea0c00000000")
    ctx.simulated = True
    tok = request_id_context.set(JID)
    try:
        assert not TP._owner_change_turn(ctx)
    finally:
        request_id_context.reset(tok)


def test_an_owner_turn_still_logs_with_its_own_ask(tmp_path, store):
    pid, ws = _built(store)
    ctx = _ctx(tmp_path, store, starter="671eea0c00000000")
    ctx.last_user_content = "make the dock bigger"
    _write_as(ctx, ws, "css/style.css", "x", "req-owner-9")
    MP.flush_change_log("req-owner-9", reply="Bigger.")
    assert store.list_changes(pid)[0]["ask"] == "make the dock bigger"


# ── (2) lookup notes in a work turn ──────────────────────────────────────────
ANSWER = ("Done, Vasilis — your webOS now has a real, working Browser, and it's live.\n\n"
          + "What you got: tabs, history, an omnibox and a start page. " * 30)   # ~1.8k, like the real one
NOTE1 = ("The Bing result is the one real gap: a genuinely blocked site did not trigger the overlay. "
         + "I stop theorizing and measure the exact cross-origin semantics with a controlled experiment. " * 5)


def test_a_long_note_beside_a_browser_check_in_a_build_turn_is_dropped():
    """Fails where the delivered reply opened with the 571-char note written beside a `browser` check."""
    assert 400 < len(NOTE1.strip()) < 1000
    text = NOTE1.strip() + "\n\n" + ANSWER
    out = drop_pre_tool_segments(text, [WorkSegment("Let me edit the file."), NOTE1.strip()])
    assert "Bing result" not in out and out.startswith("Done, Vasilis")


def test_a_long_note_in_a_pure_research_turn_is_still_kept():
    """A lookup-only turn's long note may hold part of the answer — unchanged."""
    text = NOTE1.strip() + "\n\n" + ANSWER
    assert drop_pre_tool_segments(text, [NOTE1.strip()]) == text.strip()


def test_the_work_rule_still_keeps_a_note_the_rest_does_not_outweigh():
    short_rest = "Done — it's live and verified end to end, with tabs, history and search."
    text = NOTE1.strip() + "\n\n" + short_rest
    out = drop_pre_tool_segments(text, [WorkSegment("x"), NOTE1.strip()])
    assert "Bing result" in out


# ── review r1 ────────────────────────────────────────────────────────────────
def test_a_wake_never_reopens_a_project_the_owner_closed(tmp_path, store):
    """MAJOR: fails where the wake's write moved a DONE project back to ACTIVE with nobody there."""
    pid, ws = _built(store)
    store.update_project(pid, status="DONE")
    ctx = _ctx(tmp_path, store, starter="671eea0c00000000")
    note = _write_as(ctx, ws, "js/apps/browser.js", "v3", JID)
    assert str(store.get_project(pid)["status"]).upper() == "DONE" and not note
    MP.flush_change_log(JID, reply="tweaked")
    e = store.list_changes(pid)[0]
    assert e["req_id"] == JID and not (e.get("meta") or {}).get("reopened_from")


def test_findings_beside_a_lookup_before_the_save_are_kept():
    """MODERATE: fails where "research X and save it" lost the 800-char findings written beside its last
    fetch because a later write made the turn a work turn."""
    findings = ("Finding: the three sources agree on the dates and the population figures. " * 11).strip()
    assert 400 < len(findings) < 1000
    summary = "Saved to report.md. In short: " + "the sources agree and the figures hold. " * 35
    text = findings + "\n\n" + summary
    out = drop_pre_tool_segments(text, [findings, WorkSegment("Saving.")])
    assert findings in out


# ── open items (2026-11-14, operator: "proceed with the still open items") ──────────────────────────
def _ctx_chain(tmp_path, store, rows):
    """A context whose host launch ledger holds ``rows`` {jid: starter}."""
    sbx = _FakeSandbox(tmp_path / "sandbox")
    sup = sbx_jobs.SandboxJobSupervisor(sbx)
    sbx._job_supervisor = sup
    for jid, starter in rows.items():
        sup._record_launch(jid, starter, "bash -c '…'")
    return SimpleNamespace(project_store=store, scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
                           graph_memory=None, contradiction_log=None, current_project_id=None,
                           llm_client=None, workspace_model=None, sandbox_dir=str(tmp_path / "sb"),
                           sandbox_manager=sbx, last_user_content="SYSTEM: the background job finished")


def test_a_job_started_by_a_wake_is_logged_for_the_owner(tmp_path, store):
    """Fails where the wake of a job that the owner's job's wake started wrote unlogged."""
    pid, ws = _built(store)
    ctx = _ctx_chain(tmp_path, store, {JID: "671eea0c00000000", "job-0a0a0a0a": JID})
    store.add_change(pid, req_id="671eea0c00000000", ask="add a working browser to webos", summary="Done")
    _write_as(ctx, ws, "js/apps/browser.js", "v4", "job-0a0a0a0a")
    assert MP.flush_change_log("job-0a0a0a0a", reply="ok") == 1
    e = store.list_changes(pid)[0]
    assert e["req_id"] == "job-0a0a0a0a" and "add a working browser to webos" in e["ask"]
    assert e["ask"].startswith("(unattended follow-up after background job job-0a0a0a0a finished)")


def test_a_job_started_by_an_owner_run_task_turn_is_logged(tmp_path, store):
    """Fails where the `sub-job-…` wake of a job an owner-run task turn started wrote unlogged."""
    pid, ws = _built(store)
    MP.set_request_parent("sub-ptask-77", "req-owner-77")
    ctx = _ctx_chain(tmp_path, store, {JID: "sub-ptask-77"})
    _write_as(ctx, ws, "a.js", "1", f"sub-{JID}")
    assert MP.flush_change_log(f"sub-{JID}", reply="ok") == 1
    assert store.list_changes(pid)[0]["ask"].startswith(f"(unattended follow-up after background job {JID} finished)")


@pytest.mark.parametrize("rows", [
    {JID: "sub-ptask-orphan"},                                 # a task turn with no owner root (unattended)
    {JID: "job-0b0b0b0b", "job-0b0b0b0b": "sched-x"},          # a chain ending in a scheduled task
    {JID: "job-0c0c0c0c"},                                     # a chain to a job this host never launched
    {JID: "job-0d0d0d0d", "job-0d0d0d0d": JID},                # a cycle
])
def test_chains_that_do_not_end_at_the_owner_log_nothing(tmp_path, store, rows):
    pid, ws = _built(store)
    ctx = _ctx_chain(tmp_path, store, rows)
    _write_as(ctx, ws, "x.js", "1", JID)
    MP.flush_change_log(JID, reply="")
    assert store.list_changes(pid) == []
