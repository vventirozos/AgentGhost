"""§4QA-c: scripts/reopen_4qa_experiments.py on a scratch store (operator 2026-10-18:
"yes, reopen the experiments")."""
import importlib.util
from pathlib import Path

from ghost_agent.memory.projects import ProjectStore

ROOT = Path(__file__).resolve().parents[1]


def _load(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("reopen_4qa", ROOT / "scripts" / "reopen_4qa_experiments.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mem, sb = tmp_path / "mem", tmp_path / "sb"
    store = ProjectStore(mem, sandbox_root=sb)
    pid = store.create_project("Recursion", goal="run introspection loops on an LLM")
    layer = store.add_task(pid, "Task 3.1: Run Layer 0 → results/layer0-base.md")
    other = store.add_task(pid, "Task 1.1: Research → research/r.md")
    for t in (layer, other):
        store.update_task(t, status="DONE", result_summary="done")
    ws = Path(store.ensure_workspace(pid))
    (ws / "results").mkdir(parents=True)
    (ws / "results" / "layer0-base.md").write_text("Layer 0 response (written by the agent)")
    (ws / "results" / "meta-reflection.md").write_text("kept")
    store.describe_file(pid, "results/layer0-base.md", "layer 0 capture", "")
    for k, v in (("MEM", mem), ("SANDBOX", sb), ("PID", pid), ("REOPEN", {layer: "Layer 0"}),
                 ("SIMULATED", ["results/layer0-base.md", "tools/missing.py"])):
        monkeypatch.setattr(mod, k, v)
    return mod, store, pid, layer, other, ws


def test_the_dry_run_writes_nothing(monkeypatch, tmp_path, capsys):
    mod, store, pid, layer, other, ws = _load(monkeypatch, tmp_path)
    mod.main(apply=False)
    assert (ws / "results" / "layer0-base.md").is_file()
    assert store.get_task(layer)["status"] == "DONE"
    assert mod.MARKER not in (store.get_project(pid).get("metadata") or {})
    assert "dry run" in capsys.readouterr().out


def test_apply_moves_the_invented_text_aside_and_reopens_only_the_experiments(monkeypatch, tmp_path):
    mod, store, pid, layer, other, ws = _load(monkeypatch, tmp_path)
    mod.main(apply=True)
    assert not (ws / "results" / "layer0-base.md").exists()                     # the deliverable path is free
    assert (ws / "simulated" / "results" / "layer0-base.md").read_text().startswith("Layer 0")  # kept
    assert (ws / "results" / "meta-reflection.md").read_text() == "kept"        # not on the list
    mf = store.get_file_manifest(pid)
    assert "results/layer0-base.md" not in mf and "not data" in mf["simulated/results/layer0-base.md"]["desc"]
    t = store.get_task(layer)
    assert t["status"] == "PENDING" and "probe_model" in t["result_summary"]
    assert store.get_task(other)["status"] == "DONE"
    assert "§4QA-c" in store.get_ledger(pid)
    assert store.get_project(pid)["metadata"][mod.MARKER]


def test_a_second_apply_never_reopens_a_task_closed_since(monkeypatch, tmp_path):
    mod, store, pid, layer, other, ws = _load(monkeypatch, tmp_path)
    mod.main(apply=True)
    store.update_task(layer, status="DONE", result_summary="ran with probe_model")
    mod.main(apply=True)
    assert store.get_task(layer)["status"] == "DONE"
    assert store.get_ledger(pid).count("§4QA-c") == 1


def test_a_run_that_stopped_after_the_move_re_points_on_the_next_run(monkeypatch, tmp_path):
    mod, store, pid, layer, other, ws = _load(monkeypatch, tmp_path)
    (ws / "simulated" / "results").mkdir(parents=True)
    (ws / "results" / "layer0-base.md").replace(ws / "simulated" / "results" / "layer0-base.md")  # moved, not re-pointed
    mod.main(apply=True)
    mf = store.get_file_manifest(pid)
    assert "results/layer0-base.md" not in mf and "simulated/results/layer0-base.md" in mf


def test_the_briefing_says_how_many_tasks_are_open_when_it_shows_only_some(tmp_path):
    # live probe after the reopen: "OPEN TASKS (8)" with 16 open → "the 8 pending tasks";
    # §4QA-d: "(8 of 17 — call task_list)" did not change the answer, so all are listed
    from ghost_agent.core.prompts import build_project_briefing
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    pid = store.create_project("Recursion", goal="g")
    ids = [store.add_task(pid, f"Task {i} " + "x" * 100) for i in range(12)]
    for t in ids[:5]:
        store.update_task(t, status="NEEDS_USER")
    b = build_project_briefing(store, pid)
    assert "OPEN TASKS (12: 7 PENDING, 5 WAITING(owner)):" in b      # §4SU Phase 3
    assert all(f"[{t}]" in b for t in ids)
    assert b.count("x" * 100) == 8 + 1                  # 8 in full + NEXT TASK; the rest short
    assert "action=task_list" not in b
    few = store.create_project("Few", goal="g")
    store.add_task(few, "only one")
    assert "OPEN TASKS (1: 1 PENDING):" in build_project_briefing(store, few)


def test_past_the_listed_cap_the_briefing_points_at_task_list(tmp_path):
    from ghost_agent.core.prompts import build_project_briefing
    store = ProjectStore(tmp_path / "m", sandbox_root=tmp_path / "sb")
    pid = store.create_project("Big", goal="g")
    ids = [store.add_task(pid, f"Task {i}") for i in range(35)]
    b = build_project_briefing(store, pid)
    assert "OPEN TASKS (35: 35 PENDING) — 30 listed; for the rest call manage_projects action=task_list:" in b
    assert f"[{ids[29]}]" in b and f"[{ids[30]}]" not in b
