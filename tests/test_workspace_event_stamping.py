"""Event-stamping correctness for the workspace activity log.

Regression tests for the cross-context stamping race (journal §4B):
``record_*`` used to read the ONE process-global
``workspace_model.current_project_id``, so any writer outside the
chat-turn serialization (idle autoadvance tick, explicit-project
autoadvance batch, dream self-play temp agent) stamped its events with
whatever project the LAST chat turn had open. The fix is a task-local
ContextVar override (``set_event_project`` / ``pinned_event_project``)
that every stamp site reads first, falling back to the shared attribute
when unbound.
"""

import asyncio
from pathlib import Path

import pytest

from ghost_agent.workspace import (
    WorkspaceModel,
    pinned_event_project,
    set_event_project,
)


# ---------------------------------------------------------------------------
# ContextVar override semantics
# ---------------------------------------------------------------------------

def _events(wm, kind=None, limit=20):
    return wm.activity.recent(limit=limit, kind=kind)


def test_record_falls_back_to_shared_attribute(tmp_path):
    wm = WorkspaceModel(tmp_path, enabled=True)
    wm.current_project_id = "sharedproj"
    wm.record_command_outcome(command="echo hi", exit_code=0)
    (ev,) = _events(wm, kind="command")
    assert ev.project_id == "sharedproj"


@pytest.mark.asyncio
async def test_pinned_event_project_overrides_shared_attribute(tmp_path):
    wm = WorkspaceModel(tmp_path, enabled=True)
    wm.current_project_id = "chatproj"

    async def idle_work():
        with pinned_event_project("idleproj"):
            wm.record_command_outcome(command="idle build", exit_code=1)
            wm.note("idle note")
            wm.record_task_outcome(job_id="j1", task_name="t", outcome="passed")
            wm.record_research_artifact(url="https://example.com/x")

    # Run in a child task, the way the biological tick runs the phase.
    await asyncio.create_task(idle_work())
    for ev in _events(wm):
        assert ev.project_id == "idleproj", f"{ev.kind} stamped {ev.project_id!r}"
    # The pin is scoped: after the block, records revert to the attribute.
    wm.record_command_outcome(command="after", exit_code=0)
    after = [e for e in _events(wm, kind="command")
             if e.payload.get("command") == "after"]
    assert after and after[0].project_id == "chatproj"


def test_pinned_event_project_nests_and_restores(tmp_path):
    wm = WorkspaceModel(tmp_path, enabled=True)
    wm.current_project_id = "base"
    with pinned_event_project("outer"):
        with pinned_event_project("inner"):
            wm.note("innermost")
        wm.note("outermost")
    wm.note("unpinned")
    stamps = {e.summary: e.project_id for e in _events(wm, kind="note")}
    assert stamps == {"innermost": "inner", "outermost": "outer",
                      "unpinned": "base"}


@pytest.mark.asyncio
async def test_concurrent_tasks_stamp_independently(tmp_path):
    """The original race, reproduced: another context clobbers the shared
    attribute between a turn's stamp-sync and its record_* call. With the
    task-local override, each side keeps its own stamp."""
    wm = WorkspaceModel(tmp_path, enabled=True)

    async def chat_turn():
        # What handle_chat does at prompt assembly (agent.py ~5737).
        set_event_project("chatproj")
        wm.current_project_id = "chatproj"
        await asyncio.sleep(0.05)  # LLM call — the interleave window
        wm.record_command_outcome(command="chat cmd", exit_code=0)

    async def idle_tick():
        with pinned_event_project("idleproj"):
            # Simulate a writer mutating the SHARED attribute mid-flight.
            wm.current_project_id = ""
            await asyncio.sleep(0.01)
            wm.record_command_outcome(command="idle cmd", exit_code=1)

    await asyncio.gather(chat_turn(), idle_tick())
    stamps = {e.payload.get("command"): e.project_id
              for e in _events(wm, kind="command")}
    assert stamps["chat cmd"] == "chatproj"
    assert stamps["idle cmd"] == "idleproj"


def test_scan_tracked_fallback_stamps_override(tmp_path):
    """file_changed events derive the project from the path when possible;
    the FALLBACK (non-project path) must honor the override too."""
    wm = WorkspaceModel(tmp_path, enabled=True)
    p = tmp_path / "watched.txt"
    p.write_text("v1")
    wm.track_file(str(p))
    wm.scan_tracked()  # prime the snapshot (first scan may or may not emit)
    p.write_text("v2 with different, longer content")
    with pinned_event_project("pinnedproj"):
        wm.scan_tracked()
    pinned = [e for e in _events(wm, kind="file_changed")
              if e.project_id == "pinnedproj"]
    assert pinned, "the pinned scan's file_changed event must carry the pin"


# ---------------------------------------------------------------------------
# Call-site wiring guards (the fix spans an 11k-line file; these keep the
# three producer paths from silently losing their pin in a refactor)
# ---------------------------------------------------------------------------

def _src(relpath: str) -> str:
    root = Path(__file__).resolve().parents[1] / "src" / "ghost_agent"
    return (root / relpath).read_text(encoding="utf-8")


def test_dream_selfplay_detaches_shared_workspace_model():
    # The temp agent runs under its OWN semaphore; sharing the real
    # WorkspaceModel let it clobber the global stamp pointer and pollute
    # the real activity log with synthetic self-play outcomes.
    assert "isolated_context.workspace_model = None" in _src("core/dream.py")


def test_idle_autoadvance_pins_event_project():
    src = _src("core/agent.py")
    assert "pinned_event_project" in src, "idle autoadvance lost its event pin"
    assert "set_event_project" in src, "handle_chat lost its task-local stamp"


def test_manage_projects_autoadvance_pins_event_project():
    assert "pinned_event_project" in _src("tools/projects.py")
