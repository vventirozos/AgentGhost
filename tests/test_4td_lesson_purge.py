"""§4TD fix round: `scripts/lesson_purge_4td.py`, the deploy step that makes `forget`'s "Nothing stored matches"
true. Driven on a REAL chromadb built by `VectorMemory`; no live store and no running agent is involved."""
import hashlib
import importlib.util
import json
import os
import socket
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "lesson_purge_4td.py"
CANARY = "Kosmas-keeps-the-spare-key-under-the-blue-pot"
BYSTANDERS = {
    "fact-1": ("Elena lives near the harbour", {"type": "manual"}),
    "id-1": ("Markos prefers tea", {"type": "identity"}),
    "typeless-1": ("a row from before types existed", {"timestamp": "2026-01-01T00:00:00Z"}),
    "acq-1": ("news_headlines fetches the feeds", {"type": "acquired_skill"}),
    "doc-1": ("a page of the manual", {"type": "document"}),
    "near-1": ("not a lesson, type only looks alike", {"type": "skills"}),
}
TWINS = {f"twin-{i}": (f"Bruno the dog {CANARY} {i}", {"type": "skill", "trigger": f"t{i}"}) for i in range(5)}


@pytest.fixture
def P(monkeypatch):
    spec = importlib.util.spec_from_file_location("lesson_purge_4td", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "agent_answers", lambda port=0: False)
    monkeypatch.setattr(mod, "BATCH", 2)    # every paged path pages
    return mod


@pytest.fixture
def home(tmp_path):
    from ghost_agent.memory.vector import VectorMemory
    h = tmp_path / "home"
    mem = h / "system" / "memory"
    mem.mkdir(parents=True)
    vm = VectorMemory(mem, "http://mock-url")
    rows = {**BYSTANDERS, **TWINS}
    vm.collection.add(ids=list(rows), documents=[v[0] for v in rows.values()],
                      metadatas=[v[1] for v in rows.values()])
    (mem / "skills_playbook.json").write_text(json.dumps([{"task": CANARY}]))
    (mem / "skills_playbook.lock").write_text("")
    (mem / "skills_pruned_archive.jsonl").write_text(json.dumps({"task": CANARY}) + "\n")
    (mem / "owner_notes.json").write_text("{}")
    return h


def _store(home):
    import chromadb
    from chromadb.config import Settings
    c = chromadb.PersistentClient(path=str(home / "system" / "memory"),
                                  settings=Settings(allow_reset=True, anonymized_telemetry=False))
    return c.get_collection("agent_memory", embedding_function=None)


def _ids(home):
    return set(_store(home).get(include=[])["ids"])


def _hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(root).rglob("*") if p.is_file()}


def test_only_the_twins_go_and_every_other_row_keeps_its_text_and_metadata(P, home, tmp_path):
    before = _store(home).get(ids=list(BYSTANDERS), include=["documents", "metadatas"])
    r = P.purge(home, tmp_path / "bk")
    assert r["rows"] == r["rows_deleted"] == r["exported"] == len(TWINS)
    assert _ids(home) == set(BYSTANDERS)
    after = _store(home).get(ids=list(BYSTANDERS), include=["documents", "metadatas"])
    assert sorted(zip(after["ids"], after["documents"], map(str, after["metadatas"]))) == \
        sorted(zip(before["ids"], before["documents"], map(str, before["metadatas"])))


def test_the_export_holds_ids_documents_and_metadatas(P, home, tmp_path):
    P.purge(home, tmp_path / "bk")
    data = json.loads((tmp_path / "bk" / P.EXPORT_NAME).read_text())
    assert data["count"] == len(TWINS) and set(data["ids"]) == set(TWINS)
    for i, d, m in zip(data["ids"], data["documents"], data["metadatas"]):
        assert d == TWINS[i][0] and m["trigger"] == TWINS[i][1]["trigger"]


def test_the_export_is_on_disk_and_complete_when_the_first_delete_runs(P, home, tmp_path, monkeypatch):
    seen, real = [], P.delete_rows

    def spy(collection, ids):
        seen.append(json.loads((tmp_path / "bk" / P.EXPORT_NAME).read_text())["count"])
        return real(collection, ids)
    monkeypatch.setattr(P, "delete_rows", spy)
    P.purge(home, tmp_path / "bk")
    assert seen == [len(TWINS)]


@pytest.mark.parametrize("damage", ["short", "other_text"])
def test_an_export_that_does_not_read_back_deletes_and_moves_nothing(P, home, tmp_path, monkeypatch, damage):
    real = P._load_export

    def bad(path):
        got = real(path)
        if damage == "short":
            got.popitem()
        else:
            got = {k: ("changed", v[1]) for k, v in got.items()}
        return got
    monkeypatch.setattr(P, "_load_export", bad)
    with pytest.raises(P.ExportBad):
        P.purge(home, tmp_path / "bk")
    assert _ids(home) == set(BYSTANDERS) | set(TWINS)
    assert (home / "system" / "memory" / "skills_playbook.json").exists()


def test_the_three_files_move_and_nothing_else_does(P, home, tmp_path):
    mem = home / "system" / "memory"
    r = P.purge(home, tmp_path / "bk")
    assert sorted(r["files_moved"]) == sorted(P.FILES)
    for n in P.FILES:
        assert not (mem / n).exists() and (tmp_path / "bk" / n).exists()
    assert CANARY in (tmp_path / "bk" / "skills_playbook.json").read_text()
    assert (mem / "owner_notes.json").exists()


def test_a_second_run_changes_nothing_and_keeps_the_export(P, home, tmp_path):
    P.purge(home, tmp_path / "bk")
    snap = _hashes(tmp_path / "bk")
    r = P.purge(home, tmp_path / "bk")
    assert (r["rows"], r["rows_deleted"], r["files_moved"]) == (0, 0, [])
    assert _hashes(tmp_path / "bk") == snap and _ids(home) == set(BYSTANDERS)


def test_a_run_after_a_partial_one_merges_the_export_and_replaces_no_file(P, home, tmp_path):
    P.purge(home, tmp_path / "bk")
    mem = home / "system" / "memory"
    _store(home).add(ids=["twin-late"], documents=["late"], metadatas=[{"type": "skill"}], embeddings=[[0.0] * 384])
    (mem / "skills_playbook.json").write_text("[]")
    P.purge(home, tmp_path / "bk")
    data = json.loads((tmp_path / "bk" / P.EXPORT_NAME).read_text())
    assert set(data["ids"]) == set(TWINS) | {"twin-late"} and data["count"] == len(TWINS) + 1
    assert CANARY in (tmp_path / "bk" / "skills_playbook.json").read_text()
    assert (tmp_path / "bk" / "skills_playbook.json.1").read_text() == "[]"


def test_an_unreadable_earlier_export_refuses_rather_than_overwrites(P, home, tmp_path):
    (tmp_path / "bk").mkdir()
    (tmp_path / "bk" / P.EXPORT_NAME).write_text("{torn")
    with pytest.raises(P.Refused):
        P.purge(home, tmp_path / "bk")
    assert (tmp_path / "bk" / P.EXPORT_NAME).read_text() == "{torn"
    assert _ids(home) == set(BYSTANDERS) | set(TWINS)


def test_the_dry_run_counts_the_same_and_changes_no_byte(P, home, tmp_path, capsys):
    snap = _hashes(home)
    assert P.main([str(home), str(tmp_path / "bk"), "--dry-run"]) == 0
    assert _hashes(home) == snap and not (tmp_path / "bk").exists()
    out = capsys.readouterr().out
    assert f"{len(TWINS)} vector rows" in out and "3 files" in out
    assert P.purge(home, tmp_path / "bk")["rows_deleted"] == len(TWINS)


@pytest.mark.parametrize("dry", [False, True])
def test_an_answering_agent_refuses_a_real_listener(home, tmp_path, monkeypatch, capsys, dry):
    spec = importlib.util.spec_from_file_location("lesson_purge_real", SCRIPT)
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        monkeypatch.setattr(P, "AGENT_PORT", srv.getsockname()[1])
        snap = _hashes(home)
        assert P.main([str(home), str(tmp_path / "bk")] + (["--dry-run"] if dry else [])) == 2
    assert "REFUSED" in capsys.readouterr().out
    assert _hashes(home) == snap and not (tmp_path / "bk").exists()


def test_the_default_port_is_the_agents():
    spec = importlib.util.spec_from_file_location("lesson_purge_port", SCRIPT)
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    assert P.AGENT_PORT == 8000


def test_only_a_refused_connection_counts_as_nobody_there(monkeypatch):
    spec = importlib.util.spec_from_file_location("lesson_purge_sock", SCRIPT)
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    with socket.socket() as s:      # bound and closed: a port nobody listens on
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    assert P.agent_answers(free) is False

    def timeout(*a, **k):
        raise socket.timeout("slow")
    monkeypatch.setattr(P.socket, "create_connection", timeout)
    assert P.agent_answers(free) is True


@pytest.mark.parametrize("kind", ["read_only", "a_file", "inside_memory"])
def test_a_bad_backup_dir_refuses_with_nothing_changed(P, home, tmp_path, kind):
    bk = tmp_path / "bk"
    if kind == "read_only":
        bk.mkdir()
        os.chmod(bk, 0o500)
    elif kind == "a_file":
        bk.write_text("x")
    else:
        bk = home / "system" / "memory" / "bk"
    try:
        snap = _hashes(home)
        with pytest.raises(P.Refused):
            P.purge(home, bk)
        assert _hashes(home) == snap
    finally:
        if kind == "read_only":
            os.chmod(bk, 0o700)


def test_a_home_with_no_store_creates_none(P, tmp_path):
    mem = tmp_path / "h" / "system" / "memory"
    mem.mkdir(parents=True)
    r = P.purge(tmp_path / "h", tmp_path / "bk")
    assert r["rows"] == 0 and not (mem / "chroma.sqlite3").exists()
    with pytest.raises(P.Refused):
        P.purge(tmp_path / "nothing", tmp_path / "bk")


def test_after_the_purge_forget_has_nothing_to_miss(P, home, tmp_path):
    """The point of the step: no row and no file under the memory dir quotes the lesson any more."""
    P.purge(home, tmp_path / "bk")
    got = _store(home).get(include=["documents"])
    assert not [d for d in got["documents"] if CANARY in d]
    mem = home / "system" / "memory"
    assert not [p for p in mem.iterdir() if p.is_file() and p.suffix in (".json", ".jsonl")
                and CANARY in p.read_text()]


def test_a_store_that_shrinks_by_more_than_was_asked_stops_the_step(P):
    class Greedy:
        n = 5

        def count(self):
            return self.n

        def delete(self, ids):
            self.n -= len(ids) + 1

        def get(self, ids, include):
            return {"ids": []}
    with pytest.raises(RuntimeError):
        P.delete_rows(Greedy(), ["a", "b"])


def test_another_writer_on_the_store_refuses(P, home, tmp_path, monkeypatch):
    from ghost_agent.memory import store_lock as L

    def held(*a, **k):
        raise L.StoreLockedError("held by another process")
    monkeypatch.setattr(L, "acquire_writer_lock", held)
    with pytest.raises(P.Refused):
        P.purge(home, tmp_path / "bk")
    assert _ids(home) == set(BYSTANDERS) | set(TWINS)


def test_a_row_already_exported_keeps_its_first_text(P, tmp_path):
    path = tmp_path / P.EXPORT_NAME
    P.write_export(path, {"t1": ("the first text", {"type": "skill"})})
    assert P.write_export(path, {"t1": ("different text", {"type": "skill"}), "t2": ("b", {"type": "skill"})}) == 2
    data = json.loads(path.read_text())
    assert dict(zip(data["ids"], data["documents"])) == {"t1": "the first text", "t2": "b"}


def test_an_export_whose_metadata_does_not_read_back_stops(P, tmp_path, monkeypatch):
    real = P._load_export
    monkeypatch.setattr(P, "_load_export", lambda path: {i: (d, {"type": "other"}) for i, (d, _m) in real(path).items()})
    with pytest.raises(P.ExportBad):
        P.write_export(tmp_path / P.EXPORT_NAME, {"t1": ("a", {"type": "skill"})})


def test_a_failed_delete_is_reported_and_a_second_run_finishes(P, home, tmp_path, monkeypatch, capsys):
    real = P.delete_rows
    monkeypatch.setattr(P, "delete_rows", lambda *a: (_ for _ in ()).throw(RuntimeError("disk trouble")))
    assert P.main([str(home), str(tmp_path / "bk")]) == 4
    assert "FAILED part-way" in capsys.readouterr().out
    assert set(TWINS) <= set(json.loads((tmp_path / "bk" / P.EXPORT_NAME).read_text())["ids"])
    monkeypatch.setattr(P, "delete_rows", real)
    assert P.main([str(home), str(tmp_path / "bk")]) == 0
    assert _ids(home) == set(BYSTANDERS)
