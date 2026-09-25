"""The one-off §4KJ cleanup stamps legacy `slack-` trajectories as member
(they were recorded before `extra["requester_role"]` existed), so the idle
phases stop treating them as the owner's. Driven on a fixture store."""
import json
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _store(tmp_path):
    d = tmp_path / "system" / "trajectories" / "2026-09-24"
    d.mkdir(parents=True)
    rows = [
        {"id": "a", "session_id": "slack-1", "extra": {}},
        {"id": "b", "session_id": "slack-2", "extra": {"requester_role": "owner"}},
        {"id": "c", "session_id": "abc", "extra": {}},
    ]
    (d / "s.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp_path / "system" / "memory").mkdir()
    return d / "s.jsonl"


def _run(monkeypatch, tmp_path, *args):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["x", *args])
    runpy.run_path(str(REPO / "scripts" / "member_data_cleanup_4kj.py"), run_name="__main__")


def test_dry_run_changes_nothing(monkeypatch, tmp_path):
    f = _store(tmp_path)
    before = f.read_text()
    _run(monkeypatch, tmp_path)
    assert f.read_text() == before


def test_apply_stamps_only_unstamped_slack_rows_and_backs_up(monkeypatch, tmp_path):
    f = _store(tmp_path)
    _run(monkeypatch, tmp_path, "--apply")
    rows = {json.loads(l)["id"]: json.loads(l) for l in f.read_text().splitlines()}
    assert rows["a"]["extra"]["requester_role"] == "member"
    assert rows["b"]["extra"]["requester_role"] == "owner"          # an explicit role is kept
    assert "requester_role" not in rows["c"]["extra"]               # not a slack- id
    assert list(f.parent.glob("s.jsonl.pre-4kj-*.bak"))


def test_apply_removes_only_member_foresight_rows(monkeypatch, tmp_path):
    """R10: member calls were predicted into the owner's foresight ledger."""
    _store(tmp_path)
    fs = tmp_path / "system" / "foresight"; fs.mkdir()
    led = fs / "predictions.jsonl"
    led.write_text("\n".join(json.dumps(r) for r in [
        {"req_id": "slack-1", "tool": "web_search"},      # legacy slack row → member
        {"req_id": "slack-2", "tool": "web_search"},      # explicit owner role → kept
        {"req_id": "abc", "tool": "execute"},
    ]) + "\n")
    before = led.read_text()
    _run(monkeypatch, tmp_path)
    assert led.read_text() == before                          # dry run
    _run(monkeypatch, tmp_path, "--apply")
    assert [json.loads(l)["req_id"] for l in led.read_text().splitlines()] == ["slack-2", "abc"]
    assert list(fs.glob("predictions.jsonl.pre-4kj-*.bak"))



def test_ledger_ids_follow_the_legacy_rule_and_backups_are_never_rewritten(monkeypatch, tmp_path, capsys):
    """R11: a legacy slack- id with no trajectory is a member's (step 1's
    rule), the dry run already counts legacy rows, the rotated `.1` is
    scanned, and a second --apply never rewrites an earlier backup."""
    _store(tmp_path)
    fs = tmp_path / "system" / "foresight"; fs.mkdir()
    (fs / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"req_id": "slack-1"}, {"req_id": "slack-orphan"}, {"req_id": "slack-2"}, {"req_id": 7}, {"req_id": "abc"}]) + "\n")
    (fs / "predictions.jsonl.1").write_text(json.dumps({"req_id": "slack-1"}) + "\n")
    _run(monkeypatch, tmp_path)
    assert "foresight rows removed: 3 (dry run)" in capsys.readouterr().out
    _run(monkeypatch, tmp_path, "--apply")
    kept = [json.loads(l)["req_id"] for l in (fs / "predictions.jsonl").read_text().splitlines()]
    assert kept == ["slack-2", 7, "abc"]
    assert (fs / "predictions.jsonl.1").read_text() == ""
    baks = sorted(fs.glob("*.bak"))
    snap = {b.name: b.read_text() for b in baks}
    _run(monkeypatch, tmp_path, "--apply")
    assert {b.name: b.read_text() for b in baks} == snap
    assert not list(fs.glob("*.bak.pre-4kj-*"))
