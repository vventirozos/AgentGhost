"""§4PP: (1) a summarised video's transcript is TEMPORARY in the knowledge
base and the store-hygiene job removes it after its expiry, unless the
owner's own message asks to keep it; (2) the Slack bot's conversation key
never reaches the durable session store (it failed the id charset on every
owner turn — a false "not durable" alarm)."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pytest

from ghost_agent.memory import temp_documents as TD


class FakeStore:
    """The four store calls the registry uses, over a tmp dir."""
    def __init__(self, tmp: Path, docs=(), outlines=None):
        self.outlines_file = tmp / "document_outlines.json"
        self.library = list(docs)
        self.outlines = dict(outlines or {})
        self.deleted = []

    def get_library(self):
        return list(self.library)

    def get_document_outline(self, f):
        return self.outlines.get(f, {})

    def delete_document_by_name(self, f):
        self.deleted.append(f)
        self.library.remove(f)
        return True, "Deleted"


YT = "yt-yYqvcd2fkxY.captions.el"


@pytest.mark.parametrize("text,want", [
    # kept AT INGEST only when the owner puts it in the store (r2 M2: a list
    # reading every keep flipped both ways); the rest is temporary and the
    # model keeps it on the owner's word with action='keep'
    ("summarise this video : https://www.youtube.com/watch?v=yYqvcd2fkxY", False),
    ("transcribe and summarize this video : https://youtu.be/x", False),
    ("summarise it and keep it short", False),
    ("keep it in English", False),
    ("keep it", False),
    ("save the transcript of this video", False),
    ("summarise this. use your file_system or knowledge_base tools", False),
    ("summarise this video about the React library", False),
    ("το βίντεο είναι στην βάση γνώσεων;", False),
    ("don't add it to the knowledge base", False),
    ("μην το βάλεις στη βάση γνώσεων", False),
    ("add this video to my knowledge base", True),
    ("ingest this into the kb", True),
    ("save this transcript in the library", True),
    ("βάλε το στη βάση γνώσεων", True),
    ("αποθήκευσε το βίντεο στη βάση γνώσεων", True),
])
def test_only_the_owners_words_keep_it(text, want):
    assert TD.owner_asked_to_keep(text) is want


def test_a_marked_document_is_removed_only_after_it_expires(tmp_path):
    ms = FakeStore(tmp_path, [YT, "notes.pdf"], {YT: {"video_id": "yYqvcd2fkxY"}})
    exp = TD.mark(ms, YT, reason="video transcript", now=1000.0, days=7)
    assert exp == 1000.0 + 7 * 86400 and TD.is_temporary(ms, YT)
    assert TD.sweep(ms, now=exp - 1)["removed"] == [] and ms.deleted == []
    out = TD.sweep(ms, now=exp + 1)
    assert out["removed"] == [YT] and ms.deleted == [YT] and TD.get(ms, YT) is None
    assert "notes.pdf" in ms.library                                  # never a non-temporary document


def test_a_kept_document_is_never_removed_and_never_re_marked(tmp_path):
    ms = FakeStore(tmp_path, [YT], {YT: {"video_id": "v"}})
    TD.mark(ms, YT, now=0.0, days=1)
    assert TD.keep(ms, YT, now=10.0)
    assert TD.mark(ms, YT) is None and not TD.is_temporary(ms, YT)
    assert TD.sweep(ms, now=10 ** 12)["removed"] == [] and ms.deleted == []


def test_the_sweep_marks_older_transcripts_and_nothing_else(tmp_path):
    """The 5 transcripts ingested before §4PP: temporary from the first sweep
    on — by their OUTLINE's video id, never by the filename (review r1 M1:
    an owner's `yt-notes.pdf` or a `.captions.` subtitle file was marked)."""
    ms = FakeStore(tmp_path, [YT, "yt-notes.pdf", "talk.captions.en.srt", "manual.pdf"],
                   {YT: {"video_id": "yYqvcd2fkxY"}, "manual.pdf": {"pages": 3}})
    out = TD.sweep(ms, now=5000.0)
    assert out["marked"] == [YT] and out["removed"] == []
    for other in ("yt-notes.pdf", "talk.captions.en.srt", "manual.pdf"):
        assert TD.get(ms, other) is None
    assert TD.get(ms, YT)["expires_at"] == 5000.0 + TD.ttl_days() * 86400


def test_an_entry_for_a_document_already_gone_is_dropped(tmp_path):
    ms = FakeStore(tmp_path, ["manual.pdf"])             # the library reads; the transcript is gone
    TD.mark(ms, YT, now=0.0)
    assert TD.sweep(ms, now=1.0)["dropped"] == [YT] and TD.get(ms, YT) is None


def test_an_unreadable_registry_removes_nothing(tmp_path, caplog):
    ms = FakeStore(tmp_path, [YT], {YT: {"video_id": "v"}})
    (tmp_path / TD.REGISTRY_FILENAME).write_text("{not json")
    with caplog.at_level(logging.WARNING, logger="GhostAgent"):
        out = TD.sweep(ms, now=10 ** 12)
    assert out == {"removed": [], "marked": [], "dropped": []} and ms.deleted == []
    assert "sweep skipped" in caplog.text


def test_a_failed_delete_keeps_the_entry_for_the_next_sweep(tmp_path):
    ms = FakeStore(tmp_path, [YT], {YT: {"video_id": "v"}})
    TD.mark(ms, YT, now=0.0, days=1)
    ms.delete_document_by_name = lambda f: (_ for _ in ()).throw(RuntimeError("busy"))
    assert TD.sweep(ms, now=10 ** 9)["removed"] == [] and TD.is_temporary(ms, YT)


@pytest.mark.parametrize("days,want", [("3", 3.0), ("-1", TD.DEFAULT_TTL_DAYS), ("x", TD.DEFAULT_TTL_DAYS)])
def test_the_ttl_is_configurable(monkeypatch, days, want):
    monkeypatch.setenv("GHOST_TEMP_DOC_DAYS", days)
    assert TD.ttl_days() == want


# ── the knowledge_base tool ────────────────────────────────────────────────

class _Res:
    def __init__(self, filename=YT, vid="yYqvcd2fkxY", message="SUCCESS: transcript stored"):
        self.ok, self.filename, self.video_id, self.message = True, filename, vid, message


def _as_owner(text, rid="web-4pp-1"):
    from ghost_agent.utils.logging import request_id_context
    from ghost_agent.utils.provenance import note_user_message
    tok = request_id_context.set(rid)
    note_user_message(text, rid)
    return tok


def test_a_summarised_video_is_stored_temporarily(tmp_path):
    from ghost_agent.tools.memory import _video_retention_note
    from ghost_agent.utils.logging import request_id_context
    ms = FakeStore(tmp_path, [YT], {YT: {"video_id": "yYqvcd2fkxY"}})
    tok = _as_owner("summarise this video : https://www.youtube.com/watch?v=yYqvcd2fkxY")
    try:
        note = _video_retention_note(ms, _Res(), set())
    finally:
        request_id_context.reset(tok)
    assert "Temporary:" in note and "action='keep'" in note and TD.is_temporary(ms, YT)


def test_a_video_the_owner_asked_to_save_is_kept(tmp_path):
    from ghost_agent.tools.memory import _video_retention_note
    from ghost_agent.utils.logging import request_id_context
    ms = FakeStore(tmp_path, [YT], {YT: {"video_id": "v"}})
    tok = _as_owner("add this video to my knowledge base https://youtu.be/yYqvcd2fkxY", "web-4pp-2")
    try:
        note = _video_retention_note(ms, _Res(), set())
    finally:
        request_id_context.reset(tok)
    assert note.startswith("\nKept:") and TD.get(ms, YT).get("kept_at")


def test_a_kept_video_summarised_again_stays_kept(tmp_path):
    from ghost_agent.tools.memory import _video_retention_note
    from ghost_agent.utils.logging import request_id_context
    ms = FakeStore(tmp_path, [YT], {YT: {"video_id": "v"}})
    TD.keep(ms, YT)
    tok = _as_owner("summarise this video again", "web-4pp-3")
    try:
        assert _video_retention_note(ms, _Res(), {YT}) == ""
    finally:
        request_id_context.reset(tok)
    assert not TD.is_temporary(ms, YT)


def test_a_failed_ingest_marks_nothing(tmp_path):
    from ghost_agent.tools.memory import _video_retention_note
    ms = FakeStore(tmp_path, [])
    assert _video_retention_note(ms, _Res(), set()) == "" and TD.get(ms, YT) is None


async def test_keep_is_the_owners_and_needs_no_word_list(tmp_path):
    """r2 M2: "yes please" to "shall I keep it?" was refused by the word list —
    the gate is the owner's turn; a probe or member cannot keep."""
    from ghost_agent.tools.memory import tool_knowledge_base
    from ghost_agent.utils.logging import request_id_context
    ms = FakeStore(tmp_path, [YT], {YT: {"video_id": "v"}})
    TD.mark(ms, YT)
    tok = _as_owner("yes please", "probe-4pp-4")
    try:
        out = await tool_knowledge_base(action="keep", filename=YT, memory_system=ms)
    finally:
        request_id_context.reset(tok)
    assert getattr(out, "reason_code", "") == "keep_not_asked" and TD.is_temporary(ms, YT)
    tok = _as_owner("yes please", "web-4pp-5")
    try:
        out = await tool_knowledge_base(action="keep", filename=YT, memory_system=ms)
        unknown = await tool_knowledge_base(action="keep", filename="nope.pdf", memory_system=ms)
    finally:
        request_id_context.reset(tok)
    assert str(out).startswith("SUCCESS") and not TD.is_temporary(ms, YT)
    assert getattr(unknown, "reason_code", "") == "keep_unknown_doc"


async def test_the_library_listing_says_what_is_temporary(tmp_path):
    from ghost_agent.tools.memory import tool_knowledge_base
    ms = FakeStore(tmp_path, [YT, "notes.pdf"], {YT: {"video_id": "v"}})
    TD.mark(ms, YT)
    out = await tool_knowledge_base(action="list_docs", memory_system=ms)
    line = next(ln for ln in str(out).splitlines() if YT in ln)
    assert "TEMPORARY — removed on" in line
    assert "TEMPORARY" not in next(ln for ln in str(out).splitlines() if "notes.pdf" in ln)


def test_keep_is_an_advertised_action():
    from ghost_agent.tools.memory import _KB_ACTIONS
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    kb = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "knowledge_base")
    assert "keep" in _KB_ACTIONS and "keep" in kb["function"]["parameters"]["properties"]["action"]["enum"]


# ── (2) the Slack conversation key ─────────────────────────────────────────

def test_a_slack_key_never_reaches_the_session_store(tmp_path, caplog):
    from fastapi.testclient import TestClient
    from tests.test_sessions_and_cancel import _make_app
    seen = {}

    async def capture(body, bg, request_id=None):
        seen["session_id"] = body.get("session_id")
        return ("the reply", 1, "r")
    app, _ = _make_app(tmp_path, handle_chat=capture)
    with caplog.at_level(logging.WARNING, logger="GhostAgent"), TestClient(app) as c:
        r = c.post("/api/chat", json={"model": "test-model", "stream": False,
                                      "session_id": "slack:dm:U0OWNER01",
                                      "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200
        assert c.get("/api/sessions").json()["sessions"] == []
    assert "session NOT persisted" not in caplog.text
    assert seen["session_id"] == "slack:dm:U0OWNER01"            # the state-of-work key still reaches the agent


def test_a_web_session_id_still_persists(tmp_path):
    from fastapi.testclient import TestClient
    from tests.test_sessions_and_cancel import _make_app
    app, _ = _make_app(tmp_path)
    with TestClient(app) as c:
        c.post("/api/chat", json={"model": "test-model", "stream": False, "session_id": "web-1",
                                  "messages": [{"role": "user", "content": "hi"}]})
        assert [s["id"] for s in c.get("/api/sessions").json()["sessions"]] == ["web-1"]


def test_a_probe_cannot_keep_a_video_but_its_ingest_is_temporary(tmp_path):
    from ghost_agent.tools.memory import _video_retention_note
    from ghost_agent.utils.logging import request_id_context
    ms = FakeStore(tmp_path, [YT], {YT: {"video_id": "v"}})
    tok = _as_owner("add this video to my knowledge base", "probe-4pp-1")
    try:
        note = _video_retention_note(ms, _Res(), set())
    finally:
        request_id_context.reset(tok)
    assert "Temporary:" in note and TD.is_temporary(ms, YT)


def test_a_read_only_replay_isolate_writes_no_registry(tmp_path):
    from ghost_agent.memory.readonly import ReadOnlyVectorMemory
    from ghost_agent.tools.memory import _video_retention_note
    ms = FakeStore(tmp_path, [YT], {YT: {"video_id": "v"}})
    assert _video_retention_note(ReadOnlyVectorMemory(ms), _Res(), set()) == ""
    assert not (tmp_path / TD.REGISTRY_FILENAME).exists()


async def test_the_store_hygiene_idle_job_runs_the_sweep(monkeypatch):
    """The wiring: an idle tick inside the store-hygiene window sweeps the
    owner's memory store (fails where the call is dropped or another store is passed)."""
    import datetime
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from ghost_agent.core.agent import GhostAgent
    calls = []
    monkeypatch.setattr(TD, "sweep", lambda ms, now=None: calls.append(ms) or {"removed": [], "marked": [], "dropped": []})
    ctx = MagicMock()
    ctx.llm_client = SimpleNamespace(foreground_tasks=0)
    ctx.last_activity_time = datetime.datetime.now() - datetime.timedelta(seconds=1800)
    ctx.journal = None
    ctx.trajectory_collector = None
    ctx.frontier_tracker = None
    ctx.memory_system.collection.get = MagicMock(return_value={"ids": []})
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    try:
        await agent._biological_tick()
    except Exception:  # noqa: BLE001 — other phases on a bare mock context may fail after it
        pass
    assert calls and calls[0] is ctx.memory_system


async def test_the_youtube_ingest_path_carries_the_note(tmp_path, monkeypatch):
    """Through the real knowledge_base call (battery T12): the tool result
    tells the model the transcript is temporary, and the registry holds it."""
    from ghost_agent.memory import youtube_ingest as YI
    from ghost_agent.tools.memory import tool_knowledge_base
    from ghost_agent.utils.logging import request_id_context
    ms = FakeStore(tmp_path, [], {})

    def fake_ingest(url, **kw):
        ms.library.append(YT)
        ms.outlines[YT] = {"video_id": "yYqvcd2fkxY"}
        return _Res()
    monkeypatch.setattr(YI, "ingest_youtube", fake_ingest)
    tok = _as_owner("summarise this video : https://www.youtube.com/watch?v=yYqvcd2fkxY", "web-4pp-9")
    try:
        out = await tool_knowledge_base(action="ingest_document",
                                        filename="https://www.youtube.com/watch?v=yYqvcd2fkxY",
                                        memory_system=ms, tor_proxy="socks5h://127.0.0.1:9050", sandbox_dir=tmp_path)
    finally:
        request_id_context.reset(tok)
    assert str(out).startswith("SUCCESS") and "Temporary:" in str(out) and TD.is_temporary(ms, YT)



def test_an_unreadable_library_changes_nothing(tmp_path):
    """review r1 M2: `get_library` answers [] on an error — the sweep then
    dropped every entry, kept ones too, and re-marked kept transcripts."""
    ms = FakeStore(tmp_path, [YT], {YT: {"video_id": "v"}})
    TD.keep(ms, YT)

    def broken():
        raise ValueError("library index is a dict, not a list")
    ms._load_library = broken
    out = TD.sweep(ms, now=10 ** 10)
    assert out == {"removed": [], "marked": [], "dropped": []} and TD.get(ms, YT).get("kept_at")


def test_an_empty_library_beside_registered_entries_changes_nothing(tmp_path):
    ms = FakeStore(tmp_path, [YT], {YT: {"video_id": "v"}})
    TD.keep(ms, YT)
    ms.library = []
    assert TD.sweep(ms, now=1.0)["dropped"] == [] and TD.get(ms, YT).get("kept_at")


def test_a_failed_ingest_renews_nothing(tmp_path):
    from ghost_agent.tools.memory import _video_retention_note
    ms = FakeStore(tmp_path, [YT], {YT: {"video_id": "v"}})
    bad = _Res(message="Error: captions unavailable")
    bad.ok = False
    assert _video_retention_note(ms, bad, {YT}) == "" and TD.get(ms, YT) is None


def test_the_real_store_loses_every_part_of_an_expired_transcript(tmp_path):
    """On the real VectorMemory: rows (the summary row too), catalogue,
    outline and ordered text go; another document stays."""
    from ghost_agent.memory.vector import VectorMemory
    vm = VectorMemory(tmp_path / "vm", "http://mock-url")
    vm._update_library_index(YT, "add")
    vm._update_library_index("yt-notes.pdf", "add")
    vm.set_document_outline(YT, {"video_id": "yYqvcd2fkxY"})
    vm.set_document_text(YT, {"passages": [[0, 1, "x"]]})
    vm.collection.add(ids=["a", "b", "c"], documents=["chunk", "summary", "pdf"], embeddings=[[0.1, 0.2]] * 3,
                      metadatas=[{"source": YT, "type": "document"}, {"source": YT, "type": "document_summary"},
                                 {"source": "yt-notes.pdf", "type": "document"}])
    assert TD.sweep(vm, now=0.0)["marked"] == [YT]
    assert TD.sweep(vm, now=10 ** 10)["removed"] == [YT]
    assert vm.get_library() == ["yt-notes.pdf"] and vm.collection.get()["ids"] == ["c"]
    assert vm.get_document_outline(YT) == {} and vm.get_document_text(YT) == {}



def test_a_forget_drops_the_keep_record(tmp_path):
    """r2 M1: a kept record outlived the forget; re-summarising the video
    later then stored it permanently, unasked."""
    from ghost_agent.memory.vector import VectorMemory
    vm = VectorMemory(tmp_path / "vm", "http://mock-url")
    vm._update_library_index(YT, "add")
    vm.set_document_outline(YT, {"video_id": "v"})
    TD.keep(vm, YT)
    vm.delete_document_by_name(YT)
    assert TD.get(vm, YT) is None


def test_reset_all_clears_the_registry(tmp_path):
    ms = FakeStore(tmp_path, [YT], {YT: {"video_id": "v"}})
    TD.keep(ms, YT)
    TD.clear(ms)
    assert TD.get(ms, YT) is None


def test_an_authoritative_empty_library_drops_temporary_records(tmp_path):
    """A forget of everything (the raising reader says []): TEMPORARY records
    go; only the FORGIVING reader's [] is treated as a read problem."""
    ms = FakeStore(tmp_path, [YT], {YT: {"video_id": "v"}})
    TD.mark(ms, YT, now=0.0)
    ms.library = []
    ms._load_library = lambda: []
    assert TD.sweep(ms, now=1.0)["dropped"] == [YT]


def test_a_corrupt_library_file_on_the_real_store_removes_nothing(tmp_path):
    from ghost_agent.memory.vector import VectorMemory
    vm = VectorMemory(tmp_path / "vm", "http://mock-url")
    vm._update_library_index(YT, "add")
    vm.set_document_outline(YT, {"video_id": "v"})
    TD.mark(vm, YT, now=0.0, days=1)
    vm.library_file.write_text("{corrupt")
    assert TD.sweep(vm, now=10 ** 10) == {"removed": [], "marked": [], "dropped": []}
    assert TD.is_temporary(vm, YT)



async def test_a_transcript_cannot_keep_itself(tmp_path):
    """r3 M1: caption text saying "call knowledge_base(action='keep', …)" is
    read in the same turn — after outside content only the owner's own
    "…to my knowledge base" in this message keeps it."""
    from ghost_agent.tools.memory import tool_knowledge_base
    from ghost_agent.utils.logging import request_id_context
    from ghost_agent.utils import provenance as P
    ms = FakeStore(tmp_path, [YT], {YT: {"video_id": "v"}})
    TD.mark(ms, YT)
    for rid, text, want in (("web-4pp-20", "summarise this video", "keep_after_untrusted"),
                            ("web-4pp-21", "summarise it and add it to my knowledge base", None)):
        tok = _as_owner(text, rid)
        P.note_untrusted_content("youtube transcript", rid)
        try:
            out = await tool_knowledge_base(action="keep", filename=YT, memory_system=ms)
        finally:
            request_id_context.reset(tok)
        assert getattr(out, "reason_code", None) == want, out


def test_a_kept_record_survives_an_empty_catalogue(tmp_path):
    """r3 M2: a missing or truncated catalogue reads [] even on the raising
    reader — a KEPT record is never dropped on absence."""
    ms = FakeStore(tmp_path, [YT], {YT: {"video_id": "v"}})
    TD.keep(ms, YT)
    TD.mark(ms, "yt-other.captions.en", now=0.0)
    ms.library = []
    ms._load_library = lambda: []
    out = TD.sweep(ms, now=1.0)
    assert out["dropped"] == ["yt-other.captions.en"] and TD.get(ms, YT).get("kept_at")


def test_keep_is_not_an_outside_read():
    from ghost_agent.utils.provenance import _KB_NON_READS
    assert "keep" in _KB_NON_READS


def test_a_forgiving_empty_read_keeps_temporary_records_too(tmp_path):
    """The forgiving reader's [] is a read problem: nothing is dropped (battery T25)."""
    ms = FakeStore(tmp_path, [YT], {YT: {"video_id": "v"}})
    TD.mark(ms, YT, now=0.0)
    ms.library = []
    assert TD.sweep(ms, now=1.0)["dropped"] == [] and TD.is_temporary(ms, YT)


def test_the_reconciler_drops_the_record_of_a_document_it_reaps(tmp_path):
    """r4: a catalogue line with no rows is reaped without delete_document_by_name."""
    from ghost_agent.memory.vector import VectorMemory
    vm = VectorMemory(tmp_path / "vm", "http://mock-url")
    vm._update_library_index(YT, "add")              # listed, but no rows
    TD.keep(vm, YT)
    report = vm.reconcile_indexes()
    assert YT in report.get("catalogue_dropped", []), report
    assert YT not in vm.get_library() and TD.get(vm, YT) is None
