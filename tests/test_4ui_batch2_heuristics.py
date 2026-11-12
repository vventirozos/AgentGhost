"""§4UI Batch 2 (2026-11-12) — project text heuristics replaced by what the
plan declares (heuristics review of req 8646c6c8's workflow; the dense model
declares what each task is)."""
import pytest

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.tools import projects as TP
from ghost_agent.core import project_advancer as PA


# ── dedup: the declared file or the exact text, never the feature head ──────

def test_two_apps_of_one_webos_are_both_kept(tmp_path):
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("webOS", goal="a web desktop")
    kept, dropped = TP._filter_duplicate_subtasks(store, pid, [
        "App: Notes → apps/notes.js", "App: Calculator → apps/calc.js",
        "Backend: user API", "Backend: payments API"])
    assert len(kept) == 4 and not dropped


def test_a_replayed_plan_still_dedups(tmp_path):
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("webOS", goal="a web desktop")
    store.add_task(pid, "Core Shell: index.html skeleton → index.html")
    kept, dropped = TP._filter_duplicate_subtasks(store, pid, [
        "Core shell: the page skeleton → index.html",          # same declared file
        "Write the README.", "write the readme",                # same text
    ])
    assert kept == ["Write the README."] and len(dropped) == 2


# ── the create gate keeps only the owner's explicit "no project" ────────────

@pytest.mark.parametrize("ask,refused", [
    ("write me a snake game", False),
    ("Θέλω ένα παιχνίδι φιδάκι σε html", False),
    ("I need a webOS with apps", False),
    ("build a parser, but don't create a project", True),
])
def test_the_create_gate_reads_only_a_refusal(monkeypatch, ask, refused):
    from types import SimpleNamespace
    monkeypatch.setattr("ghost_agent.core.agent.turn_origin", lambda c: "user")
    ctx = SimpleNamespace(last_user_content=ask, recent_user_contents=[])
    assert bool(TP._create_not_requested(ctx)) is refused


# ── single-file is the plan's structure, not goal words ─────────────────────

def test_one_file_is_read_from_the_declared_deliverables(tmp_path):
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    a = store.create_project("Snake", goal="split the single-file prototype into modules")
    store.add_task(a, "Board → src/board.js")
    store.add_task(a, "Snake → src/snake.js")
    assert PA._plan_declares_one_file(store, a) is False
    b = store.create_project("Page", goal="a landing page")
    store.add_task(b, "Header → index.html")
    store.add_task(b, "Footer → ./index.html")
    assert PA._plan_declares_one_file(store, b) is True


# ── the visual close gate is for code deliverables ──────────────────────────

@pytest.mark.parametrize("desc,kind,want", [
    ("Write the HUD design doc → docs/hud.md", "", False),
    ("Research canvas fingerprinting defences", "research", False),
    ("Build the HUD overlay → hud.js", "", True),
    ("Build the WebGL renderer", "build", True),
    ("Build the WebGL renderer", "", True),
    ("Write the parser → parser.py", "", False),                   # not visual
])
def test_the_visual_gate_reads_what_the_task_makes(desc, kind, want):
    assert TP._is_visual_artifact_task(desc, kind) is want


# ── the advancer: a declared kind in the owner's run; CODING builds ─────────

from tests.test_4su_phase2 import _advance  # noqa: E402


async def test_a_declared_build_with_a_gate_word_runs_in_the_owners_run(tmp_path):
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    pid = store.create_project("webOS", goal="a web desktop")
    store.update_project(pid, status="ACTIVE")
    tid = store.add_task(pid, "Add a confirm dialog before deleting a file → apps/files.js", kind="build")
    res, calls, _, _ = await _advance(store, pid, owner=True)
    assert calls                                   # it ran — any wait now is the unverified-build hold
    assert "gate word" not in str(store.get_task(tid).get("result_summary") or "")


async def test_unattended_the_same_task_is_built_too(tmp_path):
    """§4UK (operator 11-12: no confirmations): gate words hold nothing, attended or not."""
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    pid = store.create_project("webOS", goal="a web desktop")
    store.update_project(pid, status="ACTIVE")
    tid = store.add_task(pid, "Add a confirm dialog before deleting a file → apps/files.js", kind="build")
    res, calls, _, _ = await _advance(store, pid, owner=False)
    assert calls                                   # it ran — any wait now is the unverified-build hold
    assert "gate word" not in str(store.get_task(tid).get("result_summary") or "")


@pytest.mark.parametrize("desc", ["Build the pathfinding module", "Add a resource loader for sprites",
                                  "Add preview pane to the image viewer"])
async def test_an_undeclared_coding_leaf_is_built_not_searched(tmp_path, desc):
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    pid = store.create_project("Game", goal="a browser game", kind="CODING")
    store.update_project(pid, status="ACTIVE")
    store.add_task(pid, desc)
    res, calls, searches, _ = await _advance(store, pid, owner=True)
    assert calls and "web_search" not in searches and "deep_research" not in searches


def test_one_list_may_stage_one_file_but_a_replay_is_dropped(tmp_path):
    """r1 m4: within one list only identical text is a duplicate (a plan may
    stage index.html over several tasks); against the open plan the declared
    file or the text is."""
    from ghost_agent.memory.projects import ProjectStore
    from ghost_agent.tools.projects import _filter_duplicate_subtasks
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "s")
    pid = store.create_project("P")
    kept, dropped = _filter_duplicate_subtasks(store, pid, [
        "Markup and styles → index.html", "Game logic → index.html", "Game logic → index.html"])
    assert kept == ["Markup and styles → index.html", "Game logic → index.html"] and len(dropped) == 1
    store.add_task(pid, "Markup and styles → index.html")
    kept, dropped = _filter_duplicate_subtasks(store, pid, ["Board and pieces → index.html", "Rules → rules.js"])
    assert kept == ["Rules → rules.js"] and dropped == ["Board and pieces → index.html"]
