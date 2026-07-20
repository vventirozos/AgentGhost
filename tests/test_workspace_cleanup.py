"""Unit tests for end-of-project workspace cleanup (mark-and-sweep).

Covers the sweep itself (`core.workspace_cleanup`), the store-level
`on_project_done` trigger firing only on the transition to DONE, the
deduped deliverable registration, and the safety guards (fail-safe on an
unreadable keep-set, no symlink escape, project dir preserved).
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core.planning import ProjectPlan, TaskStatus
from ghost_agent.core.workspace_cleanup import (
    sweep_project_workspace, _normalize_rel,
)
from ghost_agent.memory.projects import ProjectStore, ProjectStatus


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


def _proj_dir(store, pid):
    d = store.sandbox_root / "projects" / pid
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write(base, rel, content="x"):
    p = base / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


# ----------------------------------------------------------------- sweep core

def test_sweep_keeps_registered_recovers_sources_deletes_media(store):
    """Partial keep-set (chess incident, 2026-07-02): registration proves
    what IS a deliverable, never what isn't. Unregistered SOURCE files are
    recovered (kept + registered); unregistered media/binary scratch is
    what actually gets swept."""
    pid = store.create_project("P")
    pdir = _proj_dir(store, pid)
    plan = ProjectPlan(store, pid)
    tid = plan.add_task("t")

    _write(pdir, "report.pdf", "deliverable")
    _write(pdir, "screenshot.png", "junk")
    _write(pdir, "helper.py", "source — recovered, not scratch")
    _write(pdir, "sub/notes.txt", "source — recovered, not scratch")

    store.register_file_artifact(tid, "report.pdf")

    res = sweep_project_workspace(store, pid)

    assert res["status"] == "ok"
    assert (pdir / "report.pdf").exists()
    assert not (pdir / "screenshot.png").exists()
    # source files survive a partial keep-set and are now registered
    assert (pdir / "helper.py").exists()
    assert (pdir / "sub" / "notes.txt").exists()
    assert (pdir / "sub").exists()
    assert pdir.exists()
    assert set(res["deleted"]) == {"screenshot.png"}
    assert set(res["recovered"]) == {"helper.py", "sub/notes.txt"}
    assert set(res["kept"]) == {"report.pdf", "helper.py", "sub/notes.txt"}
    payloads = {a["payload"] for a in store.list_artifacts(project_id=pid)
                if a["kind"] == "file"}
    assert {"helper.py", "sub/notes.txt"} <= payloads


def test_partial_keepset_chess_scenario_build_survives(store):
    """Regression for the live incident: a 4-file web game with ONE file
    registered lost index.html and two JS modules to the sweep. All source
    files must survive; screenshots and caches must not."""
    pid = store.create_project("Chess Game")
    pdir = _proj_dir(store, pid)
    tid = ProjectPlan(store, pid).add_task("build the game")

    _write(pdir, "chess/index.html", "<html>")
    _write(pdir, "chess/js/chess-engine.js", "engine")
    _write(pdir, "chess/js/game-state.js", "state")
    _write(pdir, "chess/js/main.js", "loop")
    _write(pdir, "chess/board_rendered.png", "screenshot")
    _write(pdir, "chess/__pycache__/x.pyc", "bytecode")
    store.register_file_artifact(tid, "chess/js/game-state.js")

    res = sweep_project_workspace(store, pid)

    for kept in ("chess/index.html", "chess/js/chess-engine.js",
                 "chess/js/game-state.js", "chess/js/main.js"):
        assert (pdir / kept).exists(), kept
    assert not (pdir / "chess" / "board_rendered.png").exists()
    assert not (pdir / "chess" / "__pycache__").exists()
    assert set(res["recovered"]) == {"chess/index.html",
                                     "chess/js/chess-engine.js",
                                     "chess/js/main.js"}


def test_partial_keepset_recovery_registers_for_next_sweep(store):
    """Recovered sources are registered, so a second sweep keeps them via
    the normal keep-set with no recovery pass needed (idempotence)."""
    pid = store.create_project("P")
    pdir = _proj_dir(store, pid)
    tid = ProjectPlan(store, pid).add_task("t")
    _write(pdir, "kept.md", "registered")
    _write(pdir, "forgotten.js", "recovered")
    store.register_file_artifact(tid, "kept.md")

    first = sweep_project_workspace(store, pid)
    assert first["recovered"] == ["forgotten.js"]

    second = sweep_project_workspace(store, pid)
    assert "recovered" not in second
    assert set(second["kept"]) == {"kept.md", "forgotten.js"}
    assert (pdir / "forgotten.js").exists()


def test_partial_keepset_debris_named_sources_still_swept(store):
    """The debris classifier outranks the source classifier: temp_-prefixed
    scripts, .log files, and dotfiles are deleted even under partial keep."""
    pid = store.create_project("P")
    pdir = _proj_dir(store, pid)
    tid = ProjectPlan(store, pid).add_task("t")
    _write(pdir, "real.py", "keep")
    _write(pdir, "temp_probe.py", "debris")
    _write(pdir, "run.log", "debris")
    _write(pdir, ".hidden.js", "debris")
    store.register_file_artifact(tid, "real.py")

    res = sweep_project_workspace(store, pid)

    assert (pdir / "real.py").exists()
    assert not (pdir / "temp_probe.py").exists()
    assert not (pdir / "run.log").exists()
    assert not (pdir / ".hidden.js").exists()
    assert "recovered" not in res


def test_partial_keepset_registered_media_kept_unregistered_deleted(store):
    """Media stays protected only by registration under a partial keep-set —
    the empty-keep-set recovery (which keeps ambiguous media) is unchanged."""
    pid = store.create_project("P")
    pdir = _proj_dir(store, pid)
    tid = ProjectPlan(store, pid).add_task("t")
    _write(pdir, "logo.png", "deliverable")
    _write(pdir, "shot1.png", "scratch")
    store.register_file_artifact(tid, "logo.png")

    sweep_project_workspace(store, pid)

    assert (pdir / "logo.png").exists()
    assert not (pdir / "shot1.png").exists()


def test_partial_keepset_dry_run_reports_without_touching(store):
    pid = store.create_project("P")
    pdir = _proj_dir(store, pid)
    tid = ProjectPlan(store, pid).add_task("t")
    _write(pdir, "kept.md", "registered")
    _write(pdir, "forgotten.js", "would recover")
    _write(pdir, "shot.png", "would delete")
    store.register_file_artifact(tid, "kept.md")

    res = sweep_project_workspace(store, pid, dry_run=True)

    assert res["recovered"] == ["forgotten.js"]
    assert res["deleted"] == ["shot.png"]
    assert (pdir / "shot.png").exists()          # nothing touched
    payloads = {a["payload"] for a in store.list_artifacts(project_id=pid)
                if a["kind"] == "file"}
    assert "forgotten.js" not in payloads        # no registration on dry_run


def test_sweep_keeps_nested_deliverable_and_its_dir(store):
    pid = store.create_project("P")
    pdir = _proj_dir(store, pid)
    tid = ProjectPlan(store, pid).add_task("t")
    _write(pdir, "src/solver.py", "keep")
    _write(pdir, "src/scratch.tmp", "drop")
    store.register_file_artifact(tid, "src/solver.py")

    sweep_project_workspace(store, pid)

    assert (pdir / "src" / "solver.py").exists()
    assert not (pdir / "src" / "scratch.tmp").exists()
    assert (pdir / "src").exists()  # not empty -> not pruned


def test_sweep_removes_pycache(store):
    pid = store.create_project("P")
    pdir = _proj_dir(store, pid)
    _write(pdir, "__pycache__/mod.cpython-310.pyc", "bytecode")
    # nothing registered -> everything (incl. pycache) is scratch

    sweep_project_workspace(store, pid)

    assert not (pdir / "__pycache__").exists()


def test_sweep_empty_keepset_recovers_files_not_deletes_them(store):
    """A finished project that registered NO deliverables must NOT have its
    workspace wiped — the agent simply forgot to register. The sweep recovers
    every non-debris file (keeps + registers) and deletes only debris. (Live
    data-loss: a one-turn single-file build's index.html was the only file and
    got swept because task_update was called without deliverables=.)"""
    pid = store.create_project("P")
    pdir = _proj_dir(store, pid)
    ProjectPlan(store, pid).add_task("t")        # a task to hang artifacts on
    _write(pdir, "index.html", "<html></html>")
    _write(pdir, "app.js", "var x=1;")
    _write(pdir, "screenshot.png", "junk")       # ambiguous -> kept (conservative)
    _write(pdir, "__pycache__/m.pyc", "bytecode")  # debris -> deleted
    _write(pdir, ".browser_runner.py", "scaffold")  # debris -> deleted

    res = sweep_project_workspace(store, pid)

    assert res["status"] == "ok"
    assert (pdir / "index.html").exists()
    assert (pdir / "app.js").exists()
    # recovery is conservative: an image could be a deliverable, so it is kept
    # rather than risk-deleted when nothing was registered
    assert (pdir / "screenshot.png").exists()
    assert not (pdir / "__pycache__").exists()
    assert not (pdir / ".browser_runner.py").exists()
    # the deliverables were registered so future reads / sweeps see them
    payloads = {a["payload"] for a in store.list_artifacts(project_id=pid)
                if a["kind"] == "file"}
    assert "index.html" in payloads and "app.js" in payloads
    assert "index.html" in res["recovered"]


def test_sweep_debris_only_workspace_is_emptied(store):
    """The recovery must not leave debris behind: a workspace that is ALL
    debris yields an empty recovered keep-set, and the normal pass then
    deletes it (keeps the dir)."""
    pid = store.create_project("P")
    pdir = _proj_dir(store, pid)
    _write(pdir, "__pycache__/m.pyc", "bytecode")
    _write(pdir, "debug_trace.tmp", "junk")

    res = sweep_project_workspace(store, pid)

    assert res["status"] == "ok"
    assert not (pdir / "__pycache__").exists()
    assert not (pdir / "debug_trace.tmp").exists()
    assert pdir.exists()
    assert res.get("recovered") == []


def test_dry_run_reports_without_deleting(store):
    pid = store.create_project("P")
    pdir = _proj_dir(store, pid)
    tid = ProjectPlan(store, pid).add_task("t")
    _write(pdir, "keep.txt", "deliverable")
    _write(pdir, "junk.tmp", "scratch")
    store.register_file_artifact(tid, "keep.txt")  # non-empty keep-set: allowlist path

    res = sweep_project_workspace(store, pid, dry_run=True)

    assert res["deleted"] == ["junk.tmp"]
    assert (pdir / "junk.tmp").exists()  # untouched
    assert (pdir / "keep.txt").exists()


def test_dry_run_recovery_reports_without_registering(store):
    """Dry-run on an unregistered project reports what it WOULD delete and
    must not register recovered files (no state mutation on a dry run)."""
    pid = store.create_project("P")
    pdir = _proj_dir(store, pid)
    ProjectPlan(store, pid).add_task("t")
    _write(pdir, "index.html", "<html></html>")
    _write(pdir, "old.tmp", "scratch")

    res = sweep_project_workspace(store, pid, dry_run=True)

    assert res["deleted"] == ["old.tmp"]
    assert (pdir / "index.html").exists() and (pdir / "old.tmp").exists()
    assert "index.html" in res["recovered"]
    # nothing persisted
    assert store.list_artifacts(project_id=pid) == []


# ----------------------------------------------------------------- guards

def test_sweep_skips_when_no_sandbox_root(tmp_path):
    store = ProjectStore(tmp_path / "mem", sandbox_root=None)
    pid = store.create_project("P")
    res = sweep_project_workspace(store, pid)
    assert res["status"] == "skipped: no sandbox_root"


def test_sweep_skips_when_project_dir_missing(store):
    pid = store.create_project("P")
    # create_project makes an empty workspace dir; remove it to simulate a
    # project whose dir never materialized (e.g. created headless).
    (store.sandbox_root / "projects" / pid).rmdir()
    res = sweep_project_workspace(store, pid)
    assert res["status"] == "skipped: no project dir"


def test_sweep_ok_on_empty_dir(store):
    pid = store.create_project("P")  # empty workspace dir exists
    res = sweep_project_workspace(store, pid)
    assert res["status"] == "ok"
    assert res["deleted"] == []


def test_sweep_fail_safe_when_keepset_unreadable(store, monkeypatch):
    pid = store.create_project("P")
    pdir = _proj_dir(store, pid)
    _write(pdir, "precious.txt", "do not delete")

    def boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(store, "list_artifacts", boom)
    res = sweep_project_workspace(store, pid)

    assert res["status"] == "skipped: artifact read failed"
    assert (pdir / "precious.txt").exists()  # nothing deleted


def test_sweep_does_not_follow_symlink_out_of_tree(store, tmp_path):
    pid = store.create_project("P")
    pdir = _proj_dir(store, pid)
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "important.txt"
    target.write_text("must survive")
    link = pdir / "link.txt"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unsupported on this platform")

    sweep_project_workspace(store, pid)

    # the link (scratch) is removed, but its target outside the tree is intact
    assert not link.exists()
    assert target.exists()
    assert target.read_text() == "must survive"


def test_sweep_path_escape_id_deletes_nothing(store):
    """C1 regression (2026-07-20): `manage_projects action=cleanup
    project_id="../.."` skipped id resolution, and `_project_dir` then
    resolved to the sandbox PARENT (reproduced) — the walk deleted
    dotfiles/logs outside the sandbox. Defense-in-depth behind the tool
    boundary: the sweep must verify its resolved root is strictly inside
    `<sandbox>/projects/` and refuse without raising."""
    parent = store.sandbox_root.parent
    decoy = parent / "decoy_outside.log"  # debris-shaped, would be swept
    decoy.write_text("outside the sandbox")

    res = sweep_project_workspace(store, "../..")

    assert res["status"] == "skipped: path escape"
    assert res["deleted"] == []
    assert decoy.exists()


def test_sweep_rejects_separator_bearing_id(store):
    """Even an id that resolves INSIDE projects/ is rejected when it
    smuggles path components — ids are opaque tokens, not paths."""
    pid = store.create_project("P")
    _proj_dir(store, pid)
    res = sweep_project_workspace(store, f"x/../{pid}")
    assert res["status"] == "skipped: path escape"
    assert res["deleted"] == []


def test_sweep_absolute_path_deliverable_is_protected(store):
    """H9 regression (2026-07-20): a deliverable registered under its
    absolute sandbox path (`/workspace/projects/<id>/demo.png`) normalized
    to `workspace/projects/<id>/demo.png`, matched no walked rel, and the
    DONE sweep deleted the registered file. Both payload forms must
    protect the same file."""
    pid = store.create_project("P")
    pdir = _proj_dir(store, pid)
    tid = ProjectPlan(store, pid).add_task("t")
    _write(pdir, "demo.png", "the registered render")
    _write(pdir, "stray_shot.png", "scratch")
    # store the raw absolute-sandbox form, bypassing write-side normalization
    store.add_artifact(tid, "file", f"/workspace/projects/{pid}/demo.png")

    res = sweep_project_workspace(store, pid)

    assert (pdir / "demo.png").exists()
    assert not (pdir / "stray_shot.png").exists()
    assert "demo.png" in res["kept"]


def test_sweep_keeps_media_referenced_by_kept_source(store):
    """MED regression (2026-07-20): the referenced-media protection was
    wired only into the recurring tidy — the DONE sweep still deleted a
    `sprites.png` that the kept `index.html` points at, breaking the
    deliverable at the moment of completion."""
    pid = store.create_project("P")
    pdir = _proj_dir(store, pid)
    tid = ProjectPlan(store, pid).add_task("t")
    _write(pdir, "index.html", '<img src="sprites.png">')
    _write(pdir, "sprites.png", "asset the build points at")
    _write(pdir, "shot.png", "stray screenshot")
    store.register_file_artifact(tid, "index.html")

    res = sweep_project_workspace(store, pid)

    assert (pdir / "sprites.png").exists()
    assert not (pdir / "shot.png").exists()
    assert res["kept_referenced"] == ["sprites.png"]
    assert "sprites.png" in res["kept"]
    assert res["deleted"] == ["shot.png"]


def test_sweep_never_deletes_git_tree(store):
    """H8 (2026-07-20): `.git` sits in `_SCRATCH_DIRS` so recovery never
    registers its internals, but the sweep must not DELETE version-control
    state either — a DONE project's repo history is part of the work."""
    pid = store.create_project("P")
    pdir = _proj_dir(store, pid)
    tid = ProjectPlan(store, pid).add_task("t")
    _write(pdir, "main.py", "keep")
    _write(pdir, ".git/config", "[core]")
    _write(pdir, ".git/objects/ab/cdef", "blob")
    (pdir / ".git" / "refs" / "tags").mkdir(parents=True)
    _write(pdir, "__pycache__/m.pyc", "bytecode")
    store.register_file_artifact(tid, "main.py")

    res = sweep_project_workspace(store, pid)

    assert (pdir / ".git" / "config").exists()
    assert (pdir / ".git" / "objects" / "ab" / "cdef").exists()
    assert (pdir / ".git" / "refs" / "tags").is_dir()  # empty, never pruned
    assert not (pdir / "__pycache__").exists()         # real debris still swept
    assert not any(r.startswith(".git/") for r in res["deleted"])


# ----------------------------------------------------------------- normalize

@pytest.mark.parametrize("payload,expected", [
    ("report.pdf", "report.pdf"),
    ("./report.pdf", "report.pdf"),
    ("/report.pdf", "report.pdf"),
    ("sub/x.md", "sub/x.md"),
    ("projects/abc123/report.pdf", "report.pdf"),  # redundant prefix stripped
    ("../escape.txt", None),                        # traversal rejected
    ("sub/../../escape", None),
    ("", None),
    (".env", ".env"),                               # leading dot file preserved
    # H9 (2026-07-20): absolute sandbox paths must collapse to the same
    # project-relative key as the bare rel, or the registered deliverable
    # matches no walked path and gets swept. Shared contract with
    # `register_file_artifact`: project-relative POSIX, no leading `/`,
    # no `workspace/` segment, no `projects/<id>/` prefix, `..` rejected.
    ("/workspace/projects/abc123/demo.png", "demo.png"),
    ("workspace/projects/abc123/demo.png", "demo.png"),
    ("workspace/demo.png", "demo.png"),
    ("/workspace/projects/abc123/sub/x.md", "sub/x.md"),
])
def test_normalize_rel(payload, expected):
    assert _normalize_rel(payload, "abc123") == expected


# ----------------------------------------------------------------- registration dedup

def test_register_file_artifact_dedups(store):
    pid = store.create_project("P")
    tid = ProjectPlan(store, pid).add_task("t")
    a1 = store.register_file_artifact(tid, "report.pdf")
    a2 = store.register_file_artifact(tid, "./report.pdf")  # same path, normalized
    a3 = store.register_file_artifact(tid, "report.pdf")
    files = [a for a in store.list_artifacts(project_id=pid) if a["kind"] == "file"]
    assert len(files) == 1
    assert a1 == a2 == a3


def test_register_file_artifact_blank_is_noop(store):
    pid = store.create_project("P")
    tid = ProjectPlan(store, pid).add_task("t")
    assert store.register_file_artifact(tid, "   ") is None
    assert store.list_artifacts(project_id=pid) == []


# ----------------------------------------------------------------- trigger wiring

def test_hook_fires_only_on_transition_to_done(store):
    fired = []
    store.on_project_done = lambda pid: fired.append(pid)
    pid = store.create_project("P")
    plan = ProjectPlan(store, pid)
    t1 = plan.add_task("a")
    t2 = plan.add_task("b")

    plan.update_status(t1, TaskStatus.DONE, result="ok")
    assert fired == []  # project not DONE yet (t2 still open)

    plan.update_status(t2, TaskStatus.DONE, result="ok")
    assert fired == [pid]  # fired exactly once on rollup to DONE


def test_hook_does_not_fire_on_failed(store):
    fired = []
    store.on_project_done = lambda pid: fired.append(pid)
    pid = store.create_project("P")
    plan = ProjectPlan(store, pid)
    t1 = plan.add_task("a")
    plan.update_status(t1, TaskStatus.FAILED, failure_reason="nope")
    assert store.get_project(pid)["status"] == ProjectStatus.FAILED.value
    assert fired == []


def test_manual_project_done_fires_hook_once(store):
    fired = []
    store.on_project_done = lambda pid: fired.append(pid)
    pid = store.create_project("P")
    store.update_project(pid, status="DONE")
    store.update_project(pid, status="DONE")  # DONE -> DONE no-op
    assert fired == [pid]


@pytest.mark.asyncio
async def test_tool_deliverables_survive_sweep_on_last_task(store, tmp_path):
    """Through the real `manage_projects` tool: completing the *last* task
    with `deliverables=[...]` must register them BEFORE the rollup-to-DONE
    sweep fires, so the deliverable survives and scratch is removed."""
    from types import SimpleNamespace
    from ghost_agent.memory.scratchpad import Scratchpad
    from ghost_agent.tools.projects import tool_manage_projects

    store.on_project_done = lambda pid: sweep_project_workspace(store, pid)
    ctx = SimpleNamespace(
        project_store=store,
        scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
        graph_memory=None,
        contradiction_log=None,
        current_project_id=None,
    )
    await tool_manage_projects(ctx, action="create", title="Build it")
    pid = ctx.current_project_id
    await tool_manage_projects(ctx, action="task_decompose",
                               subtasks=["produce report"])
    tid = store.list_tasks(pid)[0]["id"]

    pdir = store.sandbox_root / "projects" / pid
    _write(pdir, "report.pdf", "the deliverable")
    _write(pdir, "scratch.png", "junk")

    await tool_manage_projects(
        ctx, action="task_update", task_id=tid, status="DONE",
        result="done", deliverables=["report.pdf"],
    )

    assert store.get_project(pid)["status"] == ProjectStatus.DONE.value
    assert (pdir / "report.pdf").exists()       # registered before sweep
    assert not (pdir / "scratch.png").exists()  # scratch removed


def test_end_to_end_completion_sweeps_workspace(store):
    # wire the real sweep as the hook, like main.py does
    store.on_project_done = lambda pid: sweep_project_workspace(store, pid)
    pid = store.create_project("P")
    pdir = _proj_dir(store, pid)
    plan = ProjectPlan(store, pid)
    tid = plan.add_task("build")
    _write(pdir, "final.txt", "keep me")
    _write(pdir, "scratch.log", "junk")
    store.register_file_artifact(tid, "final.txt")

    plan.update_status(tid, TaskStatus.DONE, result="done")

    assert store.get_project(pid)["status"] == ProjectStatus.DONE.value
    assert (pdir / "final.txt").exists()
    assert not (pdir / "scratch.log").exists()
