"""The one-off §4KW scope migration, driven on a fixture store."""
import json
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _store(tmp_path):
    sysd = tmp_path / "system"
    d = sysd / "trajectories" / "2026-09-30"
    d.mkdir(parents=True)
    rows = [{"id": "t1", "session_id": "s1", "task_kind": "user_request", "user_request": "Show me all projects!"},
            {"id": "t3", "session_id": "s3", "task_kind": "user_request",
             "user_request": "Investigate the building history at Alkiviadou street in detail " + " ".join(f"owner{i}" for i in range(80))},
            {"id": "t2", "session_id": "probe-1", "task_kind": "probe", "user_request": "a probe request"}]
    (d / "session-x.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (sysd / "memory").mkdir()
    pb = sysd / "memory" / "skills_playbook.json"
    pb.write_text(json.dumps([
        {"task": "show me all projects", "trigger": "show me all projects", "solution": "list them"},   # keyed
        {"task": "When listing projects, cross-check counts", "solution": "x"},                          # general
        {"task": "a probe request", "solution": "y"},
        {"task": ("Investigate the building history at Alkiviadou street in detail " + " ".join(f"owner{i}" for i in range(80)))[:400],
         "solution": "1. search"},                                                                       # cut at 400                                                     # a probe's request
    ]))
    return pb


def _run(monkeypatch, tmp_path, *args):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["x", *args])
    monkeypatch.setenv("GHOST_AGENT_PORT", "1")          # not the live agent's port
    runpy.run_path(str(REPO / "scripts" / "lesson_scope_migrate_4kw.py"), run_name="__main__")


def test_dry_run_changes_nothing(monkeypatch, tmp_path):
    pb = _store(tmp_path)
    before = pb.read_text()
    _run(monkeypatch, tmp_path)
    assert pb.read_text() == before


def test_apply_tags_only_lessons_keyed_to_a_recorded_user_request(monkeypatch, tmp_path):
    pb = _store(tmp_path)
    _run(monkeypatch, tmp_path, "--apply")
    rows = {r["task"]: r for r in json.loads(pb.read_text())}
    assert rows["show me all projects"]["scope"] == "request"
    assert rows["show me all projects"]["source_request"] == "Show me all projects!"   # the full request
    assert "scope" not in rows["When listing projects, cross-check counts"]
    # a probe's lesson is request-keyed too (second review: the Moon lesson)
    assert rows["a probe request"]["scope"] == "request"
    cut = next(r for t, r in rows.items() if t.startswith("Investigate the building"))
    assert cut["scope"] == "request" and len(cut["source_request"]) > 400
    assert list(pb.parent.glob("skills_playbook.json.pre-4kw-scope-*.bak"))
