"""The one-off §4KW memory repair, driven on a fixture store."""
import json
import runpy
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _store(tmp_path):
    mem = tmp_path / "system" / "memory"
    mem.mkdir(parents=True)
    (mem / "user_profile.json").write_text(json.dumps({
        "root": {"location": {"v": "Athens", "as_of": "x"}}, "interests": {"a": "b"}}))
    con = sqlite3.connect(str(mem / "episodic_memory.db"))
    con.execute("create table episodes (id integer primary key, trigger text, timestamp real)")
    con.execute("create table episode_actions (id integer primary key, episode_id integer, tool_name text)")
    for i in (1, 2, 3):
        con.execute("insert into episodes values (?,?,?)", (i, f"t{i}", 0.0))
        con.execute("insert into episode_actions (episode_id, tool_name) values (?, 'x')", (i,))
    con.commit(); con.close()
    (mem / "skills_playbook.json").write_text(json.dumps([
        {"task": "When responding to greetings, use system_utility to check the time or weather if"},
        {"task": "keep me"}]))
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps({"probe": [1], "member": [3]}))
    return mem, ids


def _run(monkeypatch, tmp_path, *args):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["x", *args])
    runpy.run_path(str(REPO / "scripts" / "memory_repair_4kw.py"), run_name="__main__")


def test_dry_run_changes_nothing(monkeypatch, tmp_path):
    mem, ids = _store(tmp_path)
    before = {p.name: p.read_bytes() for p in mem.iterdir()}
    _run(monkeypatch, tmp_path, "--episode-ids", str(ids))
    assert {p.name: p.read_bytes() for p in mem.iterdir()} == before


def test_apply_restores_the_profile_keeping_every_other_key(monkeypatch, tmp_path):
    mem, ids = _store(tmp_path)
    try:
        _run(monkeypatch, tmp_path, "--episode-ids", str(ids), "--apply")
    except Exception:
        pass   # the fixture has no chroma store: the episode/lesson halves stop there
    prof = json.loads((mem / "user_profile.json").read_text())
    assert prof["root"]["name"]["v"] == "Vasilis" and prof["root"]["location"]["v"] == "Athens"
    assert prof["relationships"]["wife_name"]["v"] == "Fotini" and prof["interests"] == {"a": "b"}
    assert list(mem.glob("user_profile.json.pre-4kw-repair-*.bak"))
