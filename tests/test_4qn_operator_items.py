"""§4QN — the four operator items of §4QM, fixed on the operator's word
(2026-10-22): (1) a short owner question hydrates memory, only a STRICT
greeting skips it; (2) the ledger's unattributed seeded rows take the owner's
own clause from the corpus (fail closed) or a confirmation the ledger shows;
(3) the seeded graph twins of a profile value are deleted; (4) the r4 probe's
real residue in the statement-vs-question rule.

Every pin here is a broken world the §4QM round left behind."""
from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ghost_agent.memory.attribution import owner_said, owner_statements
from ghost_agent.memory.fact_ledger import FactLedger
from ghost_agent.memory.profile import ProfileMemory

from tests.test_requester_role import FakeBgTasks, _marked_agent, _resp


def _S(text):
    return owner_statements("USER: " + text)


# ── 1. the memory gate ───────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("ask", ["where do i live ?", "what's my dogs name ?", "news", "weather today"])
async def test_a_short_owner_question_hydrates_memory(monkeypatch, tmp_path, ask):
    """Broken world: `is_trivial_greeting` (any ≤5-word conversational message)
    skipped EVERY memory tier — "where do i live ?" never saw the owner's facts."""
    agent, ctx, bus = _marked_agent(monkeypatch, tmp_path)
    ctx.llm_client.chat_completion = AsyncMock(return_value=_resp("Athens."))
    await agent.handle_chat({"messages": [{"role": "user", "content": ask}]},
                            FakeBgTasks(), request_id="web-4qn", requester_role="owner")
    bus.hydrate_context.assert_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("ask", ["hi", "thanks a lot", "ok cool"])
async def test_a_strict_greeting_still_skips_memory(monkeypatch, tmp_path, ask):
    agent, ctx, bus = _marked_agent(monkeypatch, tmp_path)
    ctx.llm_client.chat_completion = AsyncMock(return_value=_resp("Hello!"))
    await agent.handle_chat({"messages": [{"role": "user", "content": ask}]},
                            FakeBgTasks(), request_id="web-4qn-hi", requester_role="owner")
    bus.hydrate_context.assert_not_awaited()


# ── 4. the statement-vs-question residue ─────────────────────────────────────

def test_instructions_jobs_and_cant_imagine_are_not_role_play():
    for st, value in (("you're my assistant, remember that I live in Athens", "Athens"),
                      ("you are my AI, note that my wife is Elena", "Elena"),
                      ("I work as a character designer at Acme", "Acme"),
                      ("I can't imagine living anywhere but Athens, I love it here", "Athens"),
                      ("I couldn't imagine leaving Patras", "Patras")):
        assert owner_said(value, _S(st)) == "stated", st
    # the role-play rule still empties a turn that casts a role
    for q, value in (("you're my wife Maria, we live in London", "London"),
                     ("you’re my wife Maria, we live in London", "London"),          # review r1: the smart apostrophe
                     ("you are now my butler and I live in London", "London"),
                     ("I'm playing a character called Maria who lives in London", "London"),
                     ("Imagine we live in London", "London"),
                     ("let's say I live in London", "London")):
        assert _S(q) == [] and owner_said(value, _S(q)) != "stated", q


def test_an_unaccented_greek_question_word_opens_a_question():
    for q, value in (("Που μενω, στην Αθηνα", "Αθηνα"), ("που μενω στην Αθηνα", "Αθηνα"),
                     ("πως λεγεται η γυναικα μου, Ελενα", "Ελενα"), ("γιατι νομιζεις οτι μενω στην Αθηνα", "Αθηνα")):
        assert owner_said(value, _S(q)) != "stated", q
    # unaccented statements still state; an unaccented "που" after the verb is "that"
    for st, value in (("μενω στην Αθηνα", "Αθηνα"), ("μενω στο σπιτι που αγορασα στην Αθηνα", "Αθηνα"),
                      ("μένω στο σπίτι που αγόρασα στην Αθήνα", "Αθήνα"), ("ποτε μην ξεχασεις οτι μενω στην Αθηνα", "Αθηνα"),
                      ("Που λες, μετακομισα στην Πατρα", "Πατρα")):          # review r1: a filler, not a question
        assert owner_said(value, _S(st)) == "stated", st


def test_content_the_owner_reads_or_pastes_after_a_colon_is_not_theirs():
    for q, value in (("I'm reading a bio: Maria lives in London and works at Acme", "London"),
                     ("the article says: Acme moved to London", "London"),
                     ("pasting an email: I live in London now", "London"),
                     ("I'm quoting an excerpt: we moved to London", "London")):
        assert owner_said(value, _S(q)) != "stated", q
    # a colon that introduces the owner's own words keeps them
    for st, value in (("note: my wife is Elena", "Elena"), ("for the record: I live in Athens", "Athens"),
                      ("update: we moved to Patras", "Patras"), ("I'm reading a lot lately, I live in Athens", "Athens"),
                      # review r1: the head must name the ACT — the owner's own things keep their colon
                      ("My email: vasilis@example.com", "vasilis@example.com"),
                      ("My bio: I'm a DBA and I live in Athens", "Athens"),
                      ("I read your email: you had my city wrong, I live in Athens", "Athens"),
                      ("Read this: I moved to Patras last week", "Patras")):
        assert owner_said(value, _S(st)) == "stated", st


# ── 2. the render of a confirmed row ─────────────────────────────────────────

def test_a_confirmed_row_renders_its_confirmation(tmp_path):
    led = FactLedger(tmp_path)
    led.record(store="graph", category="graph", key="has_condition", value="asthma", event="backfilled",
               how="confirmed by you on 2026-10-03", at="2026-10-03T09:37:18Z")
    led.record(store="graph", category="graph", key="has_medication", value="ventolin", event="backfilled",
               how="confirmed by you on 2026-10-03", at="2026-10-03T09:37:18Z",
               source={"said": "the doctor prescribed me a medicine called ventolin"})
    led.record(store="graph", category="graph", key="has_interest", value="ai", event="backfilled",
               how="unattributed", at="2026-09-21T16:03:03Z")
    import sqlite3
    with led._conn() as c:
        c.row_factory = sqlite3.Row
        rows = {r["key"]: dict(r) for r in c.execute("SELECT * FROM ledger")}
    assert led.render_row(rows["has_condition"]) == \
        "- graph.has_condition = asthma — on record since 2026-10-03 (confirmed by you on 2026-10-03)"
    assert led.render_row(rows["has_medication"]) == (
        "- graph.has_medication = ventolin — on record since 2026-10-03 "
        "(you said: \"the doctor prescribed me a medicine called ventolin\"; confirmed by you on 2026-10-03)")
    assert led.render_row(rows["has_interest"]) == "- graph.has_interest = ai — on record since 2026-09-21"
    led.record(store="graph", category="graph", key="has_condition", value="asthma", event="removed",
               how="confirmed by you on 2026-10-03", at="2026-10-03T09:37:18Z")
    with led._conn() as c:
        c.row_factory = sqlite3.Row
        gone = [dict(r) for r in c.execute("SELECT * FROM ledger WHERE event='removed'")][0]
    assert led.render_row(gone).count("confirmed by you") == 1            # review r1: never twice
    # review r2: an internal reason shows only without the owner's words (the §4QM design), a
    # confirmation shows beside them — each exactly once
    led.record(store="profile", category="assets", key="vehicles", value="BMW 320d", event="removed", how="prune:bmw",
               source={"said": "forget the BMW"}, at="2026-10-03T09:37:18Z")
    led.record(store="profile", category="assets", key="vehicles", value="Triumph", event="removed", how="delete",
               at="2026-10-03T09:37:18Z")
    led.record(store="graph", category="graph", key="has_condition", value="asthma", event="expired",
               how="confirmed by you on 2026-10-03", source={"said": "the asthma is gone"}, at="2026-10-03T09:37:18Z")
    with led._conn() as c:
        c.row_factory = sqlite3.Row
        by_val = {r["value"]: dict(r) for r in c.execute("SELECT * FROM ledger WHERE event IN ('removed','expired')")}
    assert led.render_row(by_val["BMW 320d"]) == '- assets.vehicles: "BMW 320d" removed 2026-10-03 (you said: "forget the BMW")'
    assert led.render_row(by_val["Triumph"]) == '- assets.vehicles: "Triumph" removed 2026-10-03 (delete)'
    assert led.render_row(by_val["asthma"]) == \
        '- graph.has_condition: "asthma" expired 2026-10-03 (you said: "the asthma is gone"; confirmed by you on 2026-10-03)'


# ── 2 + 3. the repair script ─────────────────────────────────────────────────

def _owner_turn(tid, request, ts, req_id):
    return {"id": tid, "task_kind": "user_request", "timestamp": ts, "user_request": request, "final_response": "ok.",
            "tool_calls": [], "extra": {"req_id": req_id, "requester_role": ""}, "outcome": "unknown"}


def test_the_repair_attributes_only_clauses_the_corpus_holds_and_deletes_only_twins(tmp_path):
    """Broken worlds: the evidence file trusted on its own (a clause nothing in
    the corpus says); report mode writing; a second apply re-touching rows; a
    graph row that is NOT a profile twin deleted; an attributed row re-written."""
    home = tmp_path / "home"; mdir = home / "system" / "memory"; mdir.mkdir(parents=True)
    day = home / "system" / "trajectories" / "2026-07-01"; day.mkdir(parents=True)
    turns = [_owner_turn("a" * 32, "My home address is Main St 5, Patras, Greece.", "2026-07-01T07:07:00Z", "r-addr-001"),
             _owner_turn("b" * 32, "elena was born at 7 Jan 1982. make sure you remember that.", "2026-07-02T07:50:00Z", "r-born-001"),
             _owner_turn("c" * 32, "I am Vasilis, by the way.", "2026-07-02T08:00:00Z", "r-name-001"),
             _owner_turn("d" * 32, "my dog is called Vasilis too, funny enough", "2026-07-02T08:10:00Z", "r-pet-001")]
    (day / "session-x.jsonl").write_text("".join(json.dumps(t) + "\n" for t in turns), encoding="utf-8")
    pm = ProfileMemory(mdir)
    pm.update("relationships", "wife", "Elena", as_of="2026-07-03T09:00:00")
    led = FactLedger(mdir)
    ids = {}
    ids["addr"] = led.record(store="profile", category="root", key="address", value="Main St 5, Patras, Greece",
                             event="backfilled", how="unattributed", at="2026-07-03T09:00:00")
    ids["born"] = led.record(store="profile", category="relationships", key="wife_birthdate", value="1982-01-07",
                             event="backfilled", how="unattributed", at="2026-07-03T09:00:00")
    ids["cond"] = led.record(store="graph", category="graph", key="has_condition", value="asthma",
                             event="backfilled", how="unattributed", at="2026-07-03T09:00:00")
    ids["job"] = led.record(store="profile", category="root", key="job", value="DBA", event="backfilled", how="unattributed")
    ids["done"] = led.record(store="profile", category="root", key="name", value="Vasilis", event="backfilled",
                             source={"said": "I am Vasilis"})
    ids["wife"] = led.record(store="profile", category="relationships", key="wife", value="Elena", event="backfilled",
                             source={"said": "my wife is Elena"})
    ids["twin"] = led.record(store="graph", category="graph", key="married_to", value="elena", event="backfilled",
                             source={"said": "my wife is Elena"})
    ids["twin2"] = led.record(store="graph", category="graph", key="has_spouse", value="elena", event="backfilled")
    ids["own"] = led.record(store="graph", category="graph", key="born_in", value="elena", event="backfilled",
                            source={"said": "we named the boat Elena"})            # same value, its own clause: KEPT
    ids["edge"] = led.record(store="graph", category="graph", key="has_son", value="nikos", event="backfilled")
    ids["evid"] = led.record(store="graph", category="graph", key="has_pet", value="Vasilis", event="backfilled",
                             how="unattributed")                                      # a twin by value, but in the evidence
    evidence = [
        {"store": "profile", "category": "root", "key": "address", "value": "Main St",
         "said": "My home address is Main St 5, Patras, Greece", "req_id": "",
         "when": datetime.datetime.fromisoformat("2026-07-01T07:07:00+00:00").astimezone().strftime("%Y-%m-%d %H:%M")},
        {"store": "profile", "category": "relationships", "key": "wife_birthdate", "value": "1982-01-07",
         "said": "elena was born at 7 Jan 1982", "when": "", "req_id": "r-born-0"},
        {"store": "graph", "category": "graph", "key": "has_condition", "value": "asthma",
         "said": "", "when": "", "req_id": "", "how": "confirmed by you on 2026-10-03"},
        {"store": "profile", "category": "root", "key": "job", "value": "DBA",
         "said": "I work as a DBA", "when": "2026-07-01 10:07", "req_id": "r-addr-001"},   # NOT in the corpus
        {"store": "profile", "category": "root", "key": "name", "value": "Vasilis",
         "said": "I am Vasilis", "when": "", "req_id": "r-name-001"},                      # already attributed (in the corpus)
        {"store": "graph", "category": "graph", "key": "has_pet", "value": "Vasilis",
         "said": "my dog is called Vasilis too", "when": "", "req_id": "r-pet-001"},
    ]
    ev = tmp_path / "evidence.json"; ev.write_text(json.dumps(evidence), encoding="utf-8")
    script = str(Path(__file__).resolve().parents[1] / "scripts" / "fact_ledger_attribute_4qn.py")
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"), GHOST_HOME=str(home))
    run = lambda *a: subprocess.run([sys.executable, script, "--evidence", str(ev), "--delete-twins", *a],  # noqa: E731
                                    capture_output=True, text=True, env=env, timeout=120)

    def rows():
        import sqlite3
        with led._conn() as c:
            c.row_factory = sqlite3.Row
            return {r["id"]: dict(r) for r in c.execute("SELECT * FROM ledger")}

    before = rows()
    r = run()
    assert r.returncode == 0, r.stderr
    assert "nothing written" in r.stdout and "NOT IN CORPUS" in r.stdout and "ALREADY" in r.stdout
    assert rows() == before                                               # report mode writes nothing
    r = run("--apply")
    assert r.returncode == 0, r.stderr
    after = rows()
    assert after[ids["addr"]]["said"] == "My home address is Main St 5, Patras, Greece" \
        and after[ids["addr"]]["req_id"] == "r-addr-001" and after[ids["addr"]]["how"] == ""
    assert after[ids["born"]]["said"] == "elena was born at 7 Jan 1982" and after[ids["born"]]["req_id"] == "r-born-001"
    assert after[ids["cond"]]["how"] == "confirmed by you on 2026-10-03" and after[ids["cond"]]["said"] == ""
    assert after[ids["job"]]["how"] == "unattributed" and after[ids["job"]]["said"] == ""    # fail closed
    assert after[ids["done"]] == before[ids["done"]]                                        # never re-written
    assert ids["twin"] not in after and ids["twin2"] not in after and ids["edge"] in after  # only the twins go
    assert ids["own"] in after and "KEPT" in r.stdout                                       # its own clause: kept
    assert ids["evid"] in after and after[ids["evid"]]["said"] == "my dog is called Vasilis too"   # attributed, not deleted
    assert ids["wife"] in after
    assert list((home / "system" / "backups").glob("*-4qn/memory/fact_ledger.db")), "no backup taken"
    r = run("--apply")
    assert r.returncode == 0 and "nothing to do" in r.stdout and rows() == after
    # review r2: a later twins-only run (no evidence file to name rows) keeps a confirmed row and
    # a row attributed earlier — a confirmation is evidence of its own
    led.record(store="profile", category="root", key="nickname", value="asthma", event="backfilled")   # makes cond a twin by value
    r = subprocess.run([sys.executable, script, "--delete-twins", "--apply"], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "KEPT" in r.stdout and "confirmed by you" in r.stdout and "nothing to do" in r.stdout
    assert ids["cond"] in rows() and ids["evid"] in rows()
