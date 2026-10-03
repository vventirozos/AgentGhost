"""The one-off §4KW r3 playbook repair: rows are named by exact trigger, an
unresolved name applies nothing, and a twin is stale when its text is not its
row's current embedding text."""
import json
import runpy
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ghost_agent.memory.skills import SkillMemory, _normalize_lesson, lesson_embedding_text

REPO = Path(__file__).resolve().parents[1]
SCRIPT = str(REPO / "scripts" / "lesson_repair_4kw_r3.py")


def _home(tmp_path, rows, archive=()):
    mem = tmp_path / "system" / "memory"
    mem.mkdir(parents=True)
    (mem / "skills_playbook.json").write_text(json.dumps(rows), encoding="utf-8")
    (mem / "skills_pruned_archive.jsonl").write_text(
        "".join(json.dumps({"lesson": l}) + "\n" for l in archive), encoding="utf-8")
    return mem


def _run(monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["x"])  # dry run
    return runpy.run_path(SCRIPT, run_name="not_main")


def test_every_named_row_must_resolve_or_nothing_is_applied(tmp_path, monkeypatch, capsys):
    mem = _home(tmp_path, [{"trigger": "unrelated", "solution": "s"}])
    m = _run(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as e:
        m["main"]()
    assert "nothing applied" in str(e.value)
    assert json.loads((mem / "skills_playbook.json").read_text()) == [{"trigger": "unrelated", "solution": "s"}]


def test_the_delete_everything_lesson_is_retracted():
    m = runpy.run_path(SCRIPT, run_name="not_main")
    assert "lots of stuff in your sandbox, clean it up" in m["SPEC"]["retract"]
    assert not set(m["SPEC"]["retract"]) & set(m["SPEC"]["scope"])


def test_a_twin_is_stale_only_when_its_text_is_not_the_row_text(tmp_path):
    m = runpy.run_path(SCRIPT, run_name="not_main")
    fresh = {"trigger": "Reading large files", "mistake": "m", "solution": "read in chunks"}
    old = {"trigger": "Parsing dates", "mistake": "m", "solution": "use ISO 8601 now"}
    sm = SkillMemory(tmp_path)
    sm.save_playbook([fresh, old])
    vm = MagicMock()
    vm.collection.get.return_value = {
        "metadatas": [{"trigger": fresh["trigger"]}, {"trigger": old["trigger"]}],
        "documents": [lesson_embedding_text(_normalize_lesson(fresh)),
                      lesson_embedding_text(_normalize_lesson(dict(old, solution="guess the format")))]}
    assert [l["trigger"] for l in m["stale_twins"](sm, vm)] == [old["trigger"]]


def test_an_apply_refuses_while_the_agent_listens(tmp_path, monkeypatch):
    """Fails in the world where the single-writer rule was convention only."""
    import socket
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    try:
        monkeypatch.setenv("GHOST_AGENT_PORT", str(srv.getsockname()[1]))
        m = runpy.run_path(SCRIPT, run_name="not_main")
        assert m["_agent_listening"]() is True
        monkeypatch.setenv("GHOST_AGENT_PORT", "1")
        assert m["_agent_listening"]() is False
    finally:
        srv.close()
