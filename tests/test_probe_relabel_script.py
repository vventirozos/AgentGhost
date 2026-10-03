"""The one-off §4KW relabel: the operator's test requests that were recorded
as user traffic get what a header-marked probe would have written. Driven on
a fixture store."""
import json
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _jsonl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def _store(tmp_path):
    sysd = tmp_path / "system"
    traj = _jsonl(sysd / "trajectories" / "2026-09-22" / "session-x.jsonl", [
        {"id": "t1", "session_id": "probe4jr1", "task_kind": "user_request", "extra": {"req_id": "probe4jr1"}},
        {"id": "t2", "session_id": "reflect:probe4jr1", "task_kind": "reflection", "extra": {}},
        {"id": "t3", "session_id": "abc123", "task_kind": "user_request", "extra": {}},
        {"id": "t4", "session_id": "probe4jr10", "task_kind": "user_request", "extra": {}},   # not in the list
    ])
    cal = _jsonl(sysd / "calibration" / "calibration.jsonl", [
        {"req_id": "probe4jr1", "origin": "user"}, {"req_id": "abc123", "origin": "user"}])
    rrf = _jsonl(sysd / "rrf" / "observations.jsonl", [
        {"turn": "kf-live", "source": "graph"}, {"turn": "abc123", "source": "graph"}])
    (sysd / "memory").mkdir()
    (sysd / "memory" / "skills_playbook.json").write_text(json.dumps([
        {"task": "from a probe", "source_trajectory_id": "t1"},
        {"task": "real", "source_trajectory_id": "t3"},
        {"task": "When calling manage_services, ensure the output is pasted verbatim as the entire",
         "trigger": "When calling manage_services, ensure the output is pasted verbatim as the entire"},
        {"task": "When using manage_services, confirm the service is up", "trigger": "When using manage_services, confirm the service is up"}]))
    _jsonl(sysd / "selfhood" / "autobiographical.jsonl", [
        {"id": "d1", "trajectory_id": "t1"}, {"id": "d2", "trajectory_id": "t3"}, {"id": "d3", "trajectory_id": "t2"}])
    return traj, cal, rrf


def _run(monkeypatch, tmp_path, *args):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["x", *args])
    runpy.run_path(str(REPO / "scripts" / "probe_relabel_4kw.py"), run_name="__main__")


def test_dry_run_changes_nothing(monkeypatch, tmp_path):
    files = [*_store(tmp_path), tmp_path / "system" / "memory" / "skills_playbook.json"]
    before = [f.read_text() for f in files]
    _run(monkeypatch, tmp_path)
    assert [f.read_text() for f in files] == before
    assert not list(tmp_path.rglob("*.bak"))


def test_apply_writes_what_a_marked_probe_would_have(monkeypatch, tmp_path):
    traj, cal, rrf = _store(tmp_path)
    _run(monkeypatch, tmp_path, "--apply")
    rows = {json.loads(l)["id"]: json.loads(l) for l in traj.read_text().splitlines()}
    assert rows["t1"]["task_kind"] == "probe" and rows["t1"]["extra"]["relabelled_4kw"] == "user_request"
    assert rows["t2"]["task_kind"] == "probe" and rows["t2"]["extra"]["relabelled_4kw"] == "reflection"
    assert rows["t3"]["task_kind"] == "user_request" and rows["t4"]["task_kind"] == "user_request"
    assert [json.loads(l)["req_id"] for l in cal.read_text().splitlines()] == ["abc123"]
    assert [json.loads(l)["turn"] for l in rrf.read_text().splitlines()] == ["abc123"]
    pb = json.loads((tmp_path / "system" / "memory" / "skills_playbook.json").read_text())
    assert [x["task"] for x in pb] == ["real", "When using manage_services, confirm the service is up"]
    diary = tmp_path / "system" / "selfhood" / "autobiographical.jsonl"
    assert [json.loads(l)["id"] for l in diary.read_text().splitlines()] == ["d2"]
    for f in (traj, cal, rrf):
        assert list(f.parent.glob(f.name + ".pre-4kw-*.bak"))


def test_a_second_apply_is_a_no_op(monkeypatch, tmp_path):
    traj, cal, rrf = _store(tmp_path)
    _run(monkeypatch, tmp_path, "--apply")
    after = [f.read_text() for f in (traj, cal, rrf)]
    _run(monkeypatch, tmp_path, "--apply")
    assert [f.read_text() for f in (traj, cal, rrf)] == after


def test_the_relabelled_rows_leave_the_readers(monkeypatch, tmp_path):
    """The point of the relabel: the shared iterator skips them."""
    traj, _, _ = _store(tmp_path)
    _run(monkeypatch, tmp_path, "--apply")
    from ghost_agent.distill.collector import TrajectoryCollector
    c = TrajectoryCollector(root=tmp_path / "system" / "trajectories", session_id="reader")
    assert sorted(t.id for t in c.iter_trajectories()) == ["t3", "t4"]
