"""`scripts/memory_repair_4mi.py` — non-owner residue leaves the owner's
stores by a JOIN against the trajectory corpus, archived, backed up, dry by
default. Fails where a member/probe row survived, an owner row was taken, or
a removal left no archive."""
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import memory_repair_4mi as R  # noqa: E402


def _home(tmp_path):
    home = tmp_path
    day = home / "system" / "trajectories" / "2026-09-24"
    day.mkdir(parents=True)
    rows = [
        {"id": "m" * 32, "timestamp": "2026-09-24T13:58:30Z", "task_kind": "user_request", "duration_s": 5,
         "extra": {"requester_role": "member", "req_id": "slack-1"}},
        {"id": "p" * 32, "timestamp": "2026-09-24T15:00:00Z", "task_kind": "probe", "duration_s": 5,
         "extra": {"req_id": "probe-1"}},
        {"id": "o" * 32, "timestamp": "2026-09-24T18:00:00Z", "task_kind": "user_request", "duration_s": 5,
         "extra": {"requester_role": "", "req_id": "abcd1234"}},
    ]
    (day / "s.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    sh = home / "system" / "selfhood"; sh.mkdir(parents=True)
    (sh / "autobiographical.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"id": "a1", "trajectory_id": "m" * 32, "summary": "member"},
        {"id": "a2", "trajectory_id": "p" * 32, "summary": "probe"},
        {"id": "a3", "trajectory_id": "o" * 32, "summary": "owner"},
        {"id": "a4", "trajectory_id": "", "summary": "legacy"}]) + "\n")
    mem = home / "system" / "memory"; mem.mkdir(parents=True)
    con = sqlite3.connect(mem / "knowledge_graph.db")
    con.execute("CREATE TABLE triplets (subject TEXT, predicate TEXT, object TEXT, timestamp DATETIME, "
                "weight INTEGER DEFAULT 1, valid_from REAL, valid_until REAL)")
    con.executemany("INSERT INTO triplets (subject, predicate, object, timestamp) VALUES (?,?,?,?)", [
        ("user", "REQUESTED", "dvda schematic", "2026-09-24 13:58:40"),       # inside the member window
        ("user", "HAS_FOTINI_DESCRIPTION", "tall", "2026-09-06 07:35:19"),    # on-demand mirror
        ("user", "HAS_WIFE", "fotini", "2026-09-06 07:35:19"),               # a real owner fact
        ("owner topic", "IS", "kept", "2026-09-24 18:00:10"),                # inside the owner window
    ])
    con.commit(); con.close()
    ws = home / "system" / "workspace"; ws.mkdir(parents=True)
    (ws / "activity.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"id": "e1", "kind": "command", "timestamp": "2026-09-24T15:00:03Z", "payload": {}, "summary": "probe cmd", "project_id": ""},
        {"id": "e2", "kind": "command", "timestamp": "2026-09-24T18:00:03Z", "payload": {}, "summary": "owner cmd", "project_id": ""},
        {"id": "e3", "kind": "file_changed", "timestamp": "2026-09-24T15:00:03Z", "payload": {}, "summary": "project file", "project_id": "abc123def456"},
    ]) + "\n")
    (ws / "narrative.txt").write_text("probe-derived narrative")
    return home


def test_dry_run_classifies_and_writes_nothing(tmp_path):
    home = _home(tmp_path)
    cls, windows = R.classify_trajectories(home)
    assert cls["m" * 32] == "member" and cls["p" * 32] == "probe" and cls["o" * 32] == "owner"
    out = R.repair_autobiography(home, cls, apply=False)
    assert out["removed"] == 2 and out["by_class"] == {"member": 1, "probe": 1}
    assert R.repair_graph(home, windows, apply=False)["removed"] == 2
    assert R.repair_workspace(home, windows, apply=False)["removed"] == 1
    assert (home / "system" / "workspace" / "narrative.txt").is_file()
    assert len((home / "system" / "selfhood" / "autobiographical.jsonl").read_text().splitlines()) == 4


def test_apply_removes_archives_and_backs_up(tmp_path):
    home = _home(tmp_path)
    cls, windows = R.classify_trajectories(home)
    R.repair_autobiography(home, cls, apply=True)
    R.repair_graph(home, windows, apply=True)
    R.repair_workspace(home, windows, apply=True)
    sh = home / "system" / "selfhood"
    kept = [json.loads(l)["summary"] for l in (sh / "autobiographical.jsonl").read_text().splitlines()]
    assert kept == ["owner", "legacy"]
    assert (sh / "autobiographical.jsonl.pre-4mi.bak").is_file()
    removed = [json.loads(l) for l in (sh / "autobiographical.removed-4mi.jsonl").read_text().splitlines()]
    assert {r["reason"] for r in removed} == {"member-turn-4mi", "probe-turn-4mi"}
    con = sqlite3.connect(home / "system" / "memory" / "knowledge_graph.db")
    left = sorted(r[0] + "|" + r[1] for r in con.execute("SELECT subject, predicate FROM triplets"))
    assert left == ["owner topic|IS", "user|HAS_WIFE"]
    arch = [json.loads(l) for l in (home / "system" / "memory" / "graph_pruned_archive.jsonl").read_text().splitlines()]
    assert {a["reason"] for a in arch} == {"member-thread-4mi", "on-demand-mirror-4mi"}
    assert (home / "system" / "memory" / "knowledge_graph.db.pre-4mi.bak").is_file()
    ws = home / "system" / "workspace"
    assert [json.loads(l)["summary"] for l in (ws / "activity.jsonl").read_text().splitlines()] == ["owner cmd", "project file"]
    assert not (ws / "narrative.txt").is_file() and (ws / "narrative.txt.pre-4mi.bak").is_file()


def test_r2_a_window_covers_the_turns_body_not_the_time_after_it(tmp_path):
    """Fails where the window was (end, end + duration): a 1,418 s probe that
    ENDED at 02:13 swallowed the owner's 02:33 commands (live, 09-18)."""
    home = tmp_path
    day = home / "system" / "trajectories" / "2026-09-18"; day.mkdir(parents=True)
    (day / "s.jsonl").write_text(json.dumps(
        {"id": "p" * 32, "timestamp": "2026-09-18T02:13:32Z", "task_kind": "probe", "duration_s": 1418,
         "extra": {"req_id": "probe-ifs"}}) + "\n")
    cls, windows = R.classify_trajectories(home)
    assert R._in_window(R._parse_ts("2026-09-18T02:00:00Z"), windows) == "probe"      # inside the turn
    assert R._in_window(R._parse_ts("2026-09-18T02:33:00Z"), windows) == ""           # after it


def test_r2_the_correction_restores_what_the_wrong_window_removed(tmp_path):
    home = tmp_path
    day = home / "system" / "trajectories" / "2026-09-18"; day.mkdir(parents=True)
    (day / "s.jsonl").write_text(json.dumps(
        {"id": "p" * 32, "timestamp": "2026-09-18T02:13:32Z", "task_kind": "probe", "duration_s": 1418,
         "extra": {"req_id": "probe-ifs"}}) + "\n")
    ws = home / "system" / "workspace"; ws.mkdir(parents=True)
    owner_ev = {"id": "e9", "kind": "command", "timestamp": "2026-09-18T02:33:00Z", "payload": {}, "summary": "owner eckit", "project_id": ""}
    probe_ev = {"id": "e8", "kind": "command", "timestamp": "2026-09-18T02:00:00Z", "payload": {}, "summary": "probe cmd", "project_id": ""}
    (ws / "activity.jsonl").write_text(json.dumps({"id": "e1", "kind": "note", "timestamp": "2026-09-17T00:00:00Z",
                                                    "payload": {}, "summary": "older", "project_id": ""}) + "\n"
                                       + json.dumps({"id": "e2", "kind": "note", "timestamp": "2026-09-19T00:00:00Z",
                                                     "payload": {}, "summary": "newer", "project_id": ""}) + "\n")
    (ws / "activity.removed-4mi.jsonl").write_text(
        json.dumps({"reason": "probe-turn-4mi", "row": owner_ev}) + "\n"
        + json.dumps({"reason": "probe-turn-4mi", "row": probe_ev}) + "\n")
    cls, windows = R.classify_trajectories(home)
    out = R.correct_r2(home, windows, apply=True)
    assert out["workspace_restored"] == 1
    rows = [json.loads(l)["summary"] for l in (ws / "activity.jsonl").read_text().splitlines()]
    assert rows == ["older", "owner eckit", "newer"]                         # back in time order
    left = [json.loads(l)["row"]["summary"] for l in (ws / "activity.removed-4mi.jsonl").read_text().splitlines()]
    assert left == ["probe cmd"]
