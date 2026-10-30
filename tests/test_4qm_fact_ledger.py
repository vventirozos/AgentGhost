"""§4QM — the owner's fact ledger (2026-10-21).

Property: every write, replacement, expiry and removal of an OWNER fact
(profile value, `user …` graph edge) leaves an append-only ledger row with
when, what, the event, the request id and the owner's own words; the ledger
never changes which facts are live; its lines reach a prompt or a recall
result only behind the switch / the probe arm. Each pin's docstring names the
broken world it catches."""
from __future__ import annotations

import ast
import asyncio
import inspect
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ghost_agent.memory import fact_ledger as FL
from ghost_agent.memory.fact_ledger import FactLedger
from ghost_agent.memory.profile import ProfileMemory


@pytest.fixture
def led(tmp_path):
    return FactLedger(tmp_path)


@pytest.fixture
def profile(tmp_path, led):
    pm = ProfileMemory(tmp_path)
    pm.ledger = led
    return pm


@pytest.fixture
def production(monkeypatch):
    monkeypatch.delenv(FL.ENV_FLAG, raising=False)


def _rows(led, **kw):
    return led.history(**kw) if kw else led.history(words=["x"], limit=0) or _all(led)


def _all(led):
    with led._conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM ledger ORDER BY id")]


# ── the ledger itself ───────────────────────────────────────────────────────

def test_record_history_and_render(led):
    """Broken worlds: an unknown event accepted; rows lost; a render that drops
    the date or the owner's words."""
    assert led.record(store="profile", category="root", key="location", value="Athens, Greece", event="stated",
                      source={"req_id": "r1", "said": "I live in Athens", "via": "profile"}, at="2026-07-07T09:00:00") > 0
    assert led.record(store="profile", category="root", key="location", value="Patras", event="replaced",
                      prev_value="Athens, Greece", source={"req_id": "r2", "said": "we moved to Patras"}) > 0
    assert led.record(store="profile", category="x", key="y", value="z", event="bogus") == 0
    rows = led.history("root", "location")
    assert [r["event"] for r in rows] == ["replaced", "stated"]
    lines = led.render_lines(rows)
    assert lines[1] == '- root.location = Athens, Greece — since 2026-07-07 (you said: "I live in Athens")'
    assert lines[0].startswith("- root.location = Patras — since ") and 'replacing "Athens, Greece"' in lines[0]
    prov = led.provenance_for("root", "location", "Athens, Greece")
    assert prov and prov["req_id"] == "r1"


def test_history_by_question_words_matches_key_value_or_the_owners_words(led):
    """Broken world: "what car did I have before?" finds nothing because the
    row's KEY is `vehicles` and only the value says "BMW"."""
    led.record(store="profile", category="assets", key="vehicles", value="BMW 118i", event="stated",
               source={"said": "my car is a BMW 118i"})
    led.record(store="profile", category="assets", key="vehicles", value="BMW 118i", event="removed", how="prune:bmw",
               source={"said": "we sold the BMW"})
    led.record(store="profile", category="pet", key="dog_name", value="Hanzo", event="stated")
    lines = led.lines_for_question("what car did I have before, the bmw?")
    assert len(lines) == 2 and all("vehicles" in ln for ln in lines) and 'removed' in lines[0]
    assert led.lines_for_question("what is my dogs name?") == ["- pet.dog_name = Hanzo — since " + FL._day(time.time())]
    assert led.lines_for_question("tell me about the weather") == []
    # a value restated many times is ONE line
    for _ in range(3):
        led.record(store="profile", category="pet", key="dog_name", value="Hanzo", event="restated")
    assert len(led.lines_for_question("my dog hanzo")) == 2


def test_said_is_clipped_and_plain_values_fold_stamped_shapes():
    assert FL.clip_said("word " * 100).endswith("…") and len(FL.clip_said("word " * 100)) <= FL.SAID_CHARS + 1
    assert FL._plain({"v": "Athens", "as_of": "2026-01-01"}) == "Athens"
    assert FL._plain([{"v": "a"}, "b", None]) == "a; b"


def test_current_source_reads_the_live_request_but_keeps_only_the_owners_clause():
    """Broken worlds: an in-turn write records no request id / no words; the
    WHOLE message is stored (review r1 MAJOR: a token pasted beside "my car
    is a Ducati" went into the ledger and back into prompts); a question
    clause is taken for the statement."""
    from ghost_agent.utils import logging as L
    from ghost_agent.memory.lesson_scope import current_request
    msg = "here is my token sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789 keep it; my dog is called Hanzo"
    t1 = L.request_id_context.set("req-123"); t2 = current_request.set(msg)
    try:
        src = FL.current_source(via="profile", value="Hanzo", field=("pet", "dog_name"))
        assert src == {"req_id": "req-123", "said": "my dog is called Hanzo", "via": "profile"}
        assert FL.current_source(via="profile") == {"req_id": "req-123", "said": "", "via": "profile"}
        assert FL.current_source(via="profile", value="Tesla", field=("assets", "car"))["said"] == ""
        current_request.set("do you remember my dog Hanzo")
        assert FL.current_source(via="profile", value="Hanzo", field=("pet", "dog_name"))["said"] == ""
    finally:
        current_request.reset(t2); L.request_id_context.reset(t1)


def test_the_owners_words_are_redacted_before_they_are_stored(led):
    """Broken world: a key, an e-mail or a password inside the owner's sentence
    is stored raw and rendered into a prompt line."""
    led.record(store="profile", category="assets", key="car", value="Ducati", event="stated",
               source={"said": "my car is a Ducati, mail me at vasilis@example.com, key sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"})
    r = led.history("assets", "car")[0]
    assert "example.com" not in r["said"] and "sk-ant" not in r["said"] and r["said"].startswith("my car is a Ducati")
    assert "<REDACTED" in r["said"]


# ── the profile writes into it ──────────────────────────────────────────────

def test_profile_update_records_stated_restated_replaced(profile, led):
    """Broken worlds: a singleton replacement recorded as a plain statement
    (the old value vanishes from history); a restatement recorded as new."""
    from ghost_agent.utils import logging as L
    from ghost_agent.memory.lesson_scope import current_request
    t1 = L.request_id_context.set("req-a"); t2 = current_request.set("my name is Vasilis")
    try:
        profile.update("root", "name", "Vasilis")
    finally:
        current_request.reset(t2); L.request_id_context.reset(t1)
    profile.update("root", "name", "Vasilis", source={"req_id": "req-b", "said": "yes, Vasilis", "via": "profile"})
    profile.update("root", "name", "Bill", source={"req_id": "req-c", "said": "call me Bill", "via": "profile"})
    rows = led.history("root", "name")
    assert [(r["event"], r["value"], r["prev_value"], r["req_id"], r["said"]) for r in rows] == [
        ("replaced", "Bill", "Vasilis", "req-c", "call me Bill"),
        ("restated", "Vasilis", "", "req-b", "yes, Vasilis"),
        ("stated", "Vasilis", "", "req-a", "my name is Vasilis")]
    # a merging key: new item = stated, the same item again = restated
    profile.update("interests", "hobbies", "chess", source={"said": "I play chess"})
    profile.update("interests", "hobbies", "cars", source={"said": "I like cars"})
    profile.update("interests", "hobbies", "chess", source={"said": "still chess"})
    assert [(r["event"], r["value"]) for r in led.history("interests", "hobbies")] == [
        ("restated", "chess"), ("stated", "cars"), ("stated", "chess")]
    assert profile.load()["interests"]["hobbies"] == ["chess", "cars"]     # the ledger changed nothing live


def test_profile_removals_record_removed_with_how(profile, led):
    """Broken world: a forgotten value leaves no trace of WHEN or HOW it went."""
    profile.update("assets", "vehicles", "BMW 118i"); profile.update("assets", "vehicles", "Ducati")
    profile.update("root", "company", "Acme"); profile.update("root", "company", "EvolMonkey")
    profile.update("pet", "dog_name", "Hanzo")
    assert profile.prune_value("assets", "vehicles", "bmw").startswith("Pruned")
    assert profile.remove_item("assets", "vehicles", "Ducati").startswith("Removed")
    assert profile.drop_previous("root", "company").startswith("Removed")
    assert profile.delete("pet", "dog_name").startswith("Removed")
    got = {(r["key"], r["value"], r["how"]) for r in _all(led) if r["event"] == "removed"}
    assert got == {("vehicles", "BMW 118i", "prune:bmw"), ("vehicles", "Ducati", "remove_item"),
                   ("company", "Acme", "drop_previous"), ("dog_name", "Hanzo", "delete")}
    assert "pet" not in profile.load() and profile.load()["root"]["company"] == "EvolMonkey"


def test_a_profile_without_a_ledger_writes_as_before(tmp_path):
    """Broken world: the hook raises when no ledger is attached (tests, the
    backfill's dry read, a boot where the ledger failed)."""
    pm = ProfileMemory(tmp_path)
    assert pm.ledger is None
    assert pm.update("root", "name", "Vasilis").startswith("Synchronized")
    assert pm.delete("root", "name").startswith("Removed")


def test_a_failing_ledger_never_breaks_the_store_write(profile, led, monkeypatch):
    monkeypatch.setattr(led, "record", MagicMock(side_effect=RuntimeError("disk")))
    assert profile.update("root", "name", "Vasilis").startswith("Synchronized")
    assert profile.load()["root"]["name"] == "Vasilis"


# ── the graph writes into it ────────────────────────────────────────────────

def test_graph_owner_edges_record_stated_restated_expired_removed(tmp_path, led):
    """Broken worlds: a world edge (no `user` end) recorded as an owner fact; a
    superseded owner edge expires without a row; a mirror delete leaves none."""
    from ghost_agent.memory.graph import GraphMemory
    g = GraphMemory(tmp_path)
    g.ledger = led
    g.add_triplets([{"subject": "user", "predicate": "LIVES_IN", "object": "Athens"},
                    {"subject": "postgresql", "predicate": "RELEASED_IN", "object": "1996"},
                    {"subject": "maria", "predicate": "LIVES_IN", "object": "London"}],
                   source={"req_id": "r1", "said": "I live in Athens", "via": "graph"})
    g.add_triplets([{"subject": "user", "predicate": "LIVES_IN", "object": "Athens"}])
    g.add_triplets([{"subject": "user", "predicate": "LIVES_IN", "object": "Patras"}],
                   source={"req_id": "r2", "said": "we moved to Patras"})
    g.sync_owner_field("dog_name", ["Hanzo"])
    g.sync_owner_field("dog_name", [])
    g.add_triplets([{"subject": "user", "predicate": "ASKED_ABOUT", "object": "weather"},
                    {"subject": "user", "predicate": "GREETED", "object": "ghost"}])
    g.add_triplets([{"subject": "user", "predicate": "MARRIED_TO", "object": "maria"}], source={"said": "my wife is Maria"})
    assert g.forget_entity("maria")[0] == 2                   # her MARRIED_TO edge and her own LIVES_IN edge
    assert g.delete_edge("user", "LIVES_IN", "patras") == 1   # a direct (non-mirror) delete IS recorded (battery G5)
    rows = _all(led)
    assert all(r["store"] == "graph" and r["category"] == "graph" for r in rows)
    keys = {r["key"] for r in rows}
    assert "released_in" not in keys and "london" not in {r["value"] for r in rows}   # no `user` end: not the owner's
    # the owner's chatter is not a life fact (review r1); the profile's mirror
    # edges are derived from a profile row the ledger already holds (review r1)
    assert not keys & {"asked_about", "greeted", "has_dog_name"}
    ev = [(r["event"], r["key"], r["value"]) for r in rows]
    assert ev[:2] == [("stated", "lives_in", "athens"), ("restated", "lives_in", "athens")]
    assert ("expired", "lives_in", "athens") in ev and ("stated", "lives_in", "patras") in ev
    assert ("removed", "married_to", "maria") in ev                     # battery G6
    assert ("removed", "lives_in", "patras") in ev
    exp = next(r for r in rows if r["event"] == "expired")
    assert exp["how"] == "superseded by patras" and exp["req_id"] == "r2"


def test_rows_are_written_only_after_the_store_commit(tmp_path, led, monkeypatch):
    """Broken world (review r1): a ledger row for an edge whose commit failed /
    a profile previous-value removal whose save failed — history says a write
    happened that never did."""
    import sqlite3
    from ghost_agent.memory import graph as G_mod
    g = G_mod.GraphMemory(tmp_path)
    g.ledger = led

    class _Boom(sqlite3.Connection):
        def commit(self):
            raise sqlite3.OperationalError("disk full")

    _real = sqlite3.connect
    monkeypatch.setattr(G_mod.sqlite3, "connect", lambda *a, **k: _real(*a, factory=_Boom, **k))
    with pytest.raises(sqlite3.OperationalError):
        g.add_triplets([{"subject": "user", "predicate": "LIVES_IN", "object": "Athens"}])
    monkeypatch.undo()
    assert _all(led) == []
    (tmp_path / "p").mkdir()
    pm = ProfileMemory(tmp_path / "p")
    pm.ledger = led
    pm.update("root", "location", "Athens")
    pm.update("root", "location", "Patras")
    n = len(_all(led))
    monkeypatch.setattr(pm, "save", lambda data: (_ for _ in ()).throw(OSError("read-only")))
    with pytest.raises(OSError):
        pm.drop_previous_mentioning("Athens")
    assert len(_all(led)) == n


def test_a_value_the_cap_pushes_out_is_recorded_removed(profile, led):
    """Broken world (review r1): the oldest hobby silently leaves the store at
    the cap and stays "since …" in the ledger for ever."""
    from ghost_agent.memory.profile import _MAX_VALUES_PER_KEY as cap
    for i in range(cap + 1):
        profile.update("interests", "hobbies", f"hobby{i}", source={"said": f"i like hobby{i}"})
    assert "hobby0" not in profile.load()["interests"]["hobbies"]
    gone = [r for r in led.history("interests", "hobbies", limit=100) if r["event"] == "removed"]
    assert [(r["value"], r["how"]) for r in gone] == [("hobby0", f"capped at {cap}")]


# ── the gate: bus and recall ────────────────────────────────────────────────

def _bus(led):
    from ghost_agent.core.bus import MemoryBus
    return MemoryBus(fact_ledger=led)


def test_history_tier_off_in_production_on_by_switch_or_arm(production, monkeypatch, led):
    """Broken worlds: history lines on a production turn with the switch off;
    the arm honoured on a non-probe; a non-self question getting history."""
    from ghost_agent.utils import logging as L
    led.record(store="profile", category="pet", key="dog_name", value="Hanzo", event="stated", source={"said": "my dog is Hanzo"})
    bus = _bus(led)
    q = "what's my dogs name?"
    assert asyncio.run(bus._fetch_history(q)) == []
    monkeypatch.setenv(FL.ENV_FLAG, "1")
    out = asyncio.run(bus._fetch_history(q))
    assert out and out[0]["source"] == "history" and "pet.dog_name = Hanzo" in out[0]["text"]
    assert asyncio.run(bus._fetch_history("what is the dog breed with the longest name?")) == []   # not about the owner
    monkeypatch.delenv(FL.ENV_FLAG)
    t1 = L.prompt_arm_context.set("fact_history"); t2 = L.request_id_context.set("probe-4qm-1")
    try:
        assert asyncio.run(bus._fetch_history(q)) == out
        L.request_id_context.set("owner-turn")
        assert asyncio.run(bus._fetch_history(q)) == []
    finally:
        L.request_id_context.reset(t2); L.prompt_arm_context.reset(t1)
    assert L.parse_prompt_arm("fact_history") == "fact_history"


def test_hydrated_history_skips_on_demand_fields_and_scrubs_the_street(monkeypatch, led, tmp_path):
    """§4LB broken worlds: "where do I live?" hydrates the street address row;
    the owner's sentence behind another field quotes the street verbatim."""
    monkeypatch.setenv(FL.ENV_FLAG, "1")
    from ghost_agent.core.bus import MemoryBus
    pm = ProfileMemory(tmp_path)
    pm.update("root", "address", "Makedonias 83 Thrakomakedones 13676, Athens, Greece")
    led.record(store="profile", category="root", key="address", value="Makedonias 83 Thrakomakedones 13676, Athens, Greece",
               event="stated", source={"said": "My home address is Makedonias 83 Thrakomakedones 13676"})
    led.record(store="profile", category="root", key="location", value="Athens, Greece", event="stated",
               source={"said": "I live at Makedonias 83 Thrakomakedones, Athens"})
    bus = MemoryBus(profile_memory=pm, fact_ledger=led)
    out = asyncio.run(bus._fetch_history("where do I live and what's my address?"))
    texts = [o["text"] for o in out]
    assert len(texts) == 1 and "root.location = Athens, Greece" in texts[0]
    assert "Makedonias" not in " ".join(texts) and "83" not in " ".join(texts)
    # recall (on demand) still shows the field itself
    from ghost_agent.tools.memory import tool_recall
    ms = MagicMock(); ms.search_advanced = MagicMock(return_value=[])
    rec = asyncio.run(tool_recall(query="what's my address?", memory_system=ms, fact_ledger=led))
    assert "root.address" in rec


def test_history_block_renders_under_its_header_and_logs_its_tier(monkeypatch, led, caplog):
    import logging as _l
    monkeypatch.setenv(FL.ENV_FLAG, "1")
    led.record(store="profile", category="pet", key="dog_name", value="Hanzo", event="stated")
    bus = _bus(led)
    with caplog.at_level(_l.INFO, logger="GhostAgent"):
        out = asyncio.run(bus.hydrate_context("what's my dogs name?", context_budget=4000, raw_user_text="what's my dogs name?"))
    assert out.startswith(FL.HEADER) and "pet.dog_name = Hanzo" in out
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("Hydration tiers:"))
    assert " h=1 " in line and " h1)" in line


def test_recall_shows_the_history_section_only_behind_the_switch(production, monkeypatch, led):
    """Broken worlds: recall shows history with the switch off; a self-question
    with no vector hit loses the section; a world question gets it."""
    from ghost_agent.tools.memory import tool_recall
    led.record(store="root", category="root", key="location", value="Athens, Greece", event="stated",
               source={"said": "I live in Athens"}, at="2026-07-07T00:00:00")
    ms = MagicMock(); ms.search_advanced = MagicMock(return_value=[])
    out = asyncio.run(tool_recall(query="since when do you know where I live?", memory_system=ms, fact_ledger=led))
    assert FL.HEADER not in out
    monkeypatch.setenv(FL.ENV_FLAG, "1")
    out = asyncio.run(tool_recall(query="since when do you know where I live?", memory_system=ms, fact_ledger=led))
    assert FL.HEADER in out and "root.location = Athens, Greece — since 2026-07-07" in out and out.startswith("SYSTEM: No stored memory matched; the owner's fact history")
    ms.search_advanced = MagicMock(return_value=[{"score": 0.25, "text": "User location is Athens, Greece", "metadata": {"source": "identity"}}])
    out = asyncio.run(tool_recall(query="where do i live?", memory_system=ms, fact_ledger=led))
    assert "SYSTEM: Found 1 memories" in out and FL.HEADER in out
    out = asyncio.run(tool_recall(query="what is the capital of greece", memory_system=ms, fact_ledger=led))
    assert FL.HEADER not in out


# ── wiring ──────────────────────────────────────────────────────────────────

def test_boot_attaches_the_ledger_to_the_context_and_both_writers(tmp_path):
    from ghost_agent.main import _wire_fact_ledger
    from ghost_agent.memory.graph import GraphMemory
    ctx = SimpleNamespace(memory_dir=str(tmp_path), profile_memory=ProfileMemory(tmp_path), graph_memory=GraphMemory(tmp_path))
    _wire_fact_ledger(ctx)
    assert isinstance(ctx.fact_ledger, FactLedger)
    assert ctx.profile_memory.ledger is ctx.fact_ledger and ctx.graph_memory.ledger is ctx.fact_ledger
    bare = SimpleNamespace(memory_dir=None)
    _wire_fact_ledger(bare)
    assert bare.fact_ledger is None


def test_both_bus_constructors_carry_the_ledger(tmp_path):
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent import main
    led = FactLedger(tmp_path)
    me = SimpleNamespace(context=SimpleNamespace(memory_bus=None, owner_memory_isolated=False, fact_ledger=led))
    assert GhostAgent._get_memory_bus(me).ledger is led
    tree = ast.parse(inspect.getsource(main.lifespan))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "MemoryBus"]
    assert calls and all(any(k.arg == "fact_ledger" for k in c.keywords) for c in calls)
    assert any(isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_wire_fact_ledger" for n in ast.walk(tree))


def test_the_extractor_names_the_owners_clause_behind_a_value():
    """Broken world: the background extractor's ledger rows carry the SYSTEM
    drain's empty request instead of the sentence the owner typed."""
    from ghost_agent.core.agent import _owner_sentence_for
    from ghost_agent.memory.attribution import PREDICATE_CUES, cue_for_field, owner_statements, owner_clause_for
    ep = "USER: My dog is called Hanzo. I don't drive a Tesla, we live in Athens.\nAI: noted.\nUSER: do you remember my wife Fotini"
    assert _owner_sentence_for("Hanzo", ep) == "My dog is called Hanzo"          # the owner's RAW words (review r1)
    assert _owner_sentence_for("Athens, Greece", ep) == "we live in Athens"
    assert _owner_sentence_for("Tesla", ep) == "" and _owner_sentence_for("", ep) == ""
    assert _owner_sentence_for("Fotini", ep) == ""                              # a question states nothing
    # the field's predicate cue beats an earlier clause that only shares the word (review r1)
    ep2 = "USER: I was in Athens last weekend for a wedding. We moved to Athens in May."
    assert owner_clause_for("Athens, Greece", ep2) == "I was in Athens last weekend for a wedding"
    assert owner_clause_for("Athens, Greece", ep2, cue=PREDICATE_CUES["LIVES_IN"]) == "We moved to Athens in May"
    assert cue_for_field("root", "location") == PREDICATE_CUES["LIVES_IN"]
    assert cue_for_field("pet", "dog_name") == PREDICATE_CUES["HAS_NAME"] and cue_for_field("assets", "car") == ""


def test_a_question_without_a_question_mark_is_not_a_statement():
    """Broken world (review r1 MAJOR): "do you remember my wife fotini",
    "tell me about my car bmw", "remind me where i live athens" passed as the
    owner's statements — attributed, and would have WRITTEN wife = fotini."""
    from ghost_agent.memory.attribution import owner_statements
    for q in ("do you remember my wife fotini", "tell me about my car bmw", "remind me where i live athens",
              "what is my dogs name", "have you got my address", "θυμάσαι τη γυναίκα μου", "πες μου που μένω"):
        assert owner_statements("USER: " + q) == [], q
    assert owner_statements("USER: remember my wife is called Fotini") == [("remember my wife is called fotini", False)]
    assert owner_statements("USER: we moved to Patras, not Athens.")[0][0] == "we moved to patras"


def test_extractor_rows_carry_the_turns_request_id():
    """Broken world (review r1): the drain's rows say req_id "" although the
    journal item was written inside the turn — the smart-memory payload must
    carry it, the drain must pass it on, and the profile write must use it
    (an AST walk of the three call sites)."""
    tree = ast.parse((Path(__file__).resolve().parents[1] / "src" / "ghost_agent" / "core" / "agent.py")
                     .read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]

    def _name(c):
        f = c.func
        return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")

    def _dict_keys(node):
        return {k.value for k in getattr(node, "keys", []) if isinstance(k, ast.Constant)}

    appends = [c for c in calls if _name(c) == "_journal_append_safe" and c.args
               and isinstance(c.args[0], ast.Constant) and c.args[0].value == "smart_memory"]
    assert len(appends) == 2 and all("req_id" in _dict_keys(c.args[1]) for c in appends)
    drains = [c for c in calls if _name(c) == "run_smart_memory_task"]
    assert drains and all(any(k.arg == "req_id" for k in c.keywords) for c in drains)
    writes = [c for c in calls if _name(c) == "to_thread" and c.args
              and isinstance(c.args[0], ast.Attribute) and c.args[0].attr == "update"
              and any(k.arg == "source" for k in c.keywords)]
    assert writes, "the extractor's profile write passes no source"
    for w in writes:
        src = next(k.value for k in w.keywords if k.arg == "source")
        rid = dict(zip([k.value for k in src.keys], src.values))["req_id"]
        assert not (isinstance(rid, ast.Constant) and rid.value == ""), "the extractor's req_id is the empty constant"
    gw = [c for c in calls if _name(c) == "to_thread" and c.args
          and isinstance(c.args[0], ast.Attribute) and c.args[0].attr == "add_triplets"
          and any(k.arg == "source" for k in c.keywords)]
    assert gw and all({"req_id", "via", "episode"} <= _dict_keys(next(k.value for k in c.keywords if k.arg == "source"))
                      for c in gw), "the extractor's graph write passes no source/episode (r2)"


def test_hydration_applies_the_profiles_whole_on_demand_rule(monkeypatch, led, tmp_path):
    """Broken world (review r1 MAJOR): only the policy set was excluded, so
    `root.appearance_description` (a physical description) and
    `root.home_address` rendered into the prompt; §4LB keeps every
    on-demand field (suffix, address-like key, street-looking value) out."""
    monkeypatch.setenv(FL.ENV_FLAG, "1")
    from ghost_agent.core.bus import MemoryBus
    pm = ProfileMemory(tmp_path)
    pm.ledger = led
    pm.update("root", "appearance_description", "brown hair, scar on the left cheek")
    pm.update("root", "home_address", "Makedonias 83 Thrakomakedones 13676")
    pm.update("root", "location", "Athens, Greece")
    pm.update("notes", "info", "parking spot: Makedonias 83")          # a street-looking VALUE under a plain key
    bus = MemoryBus(profile_memory=pm, fact_ledger=led)
    out = asyncio.run(bus._fetch_history("what did i tell you about my appearance description, home address and location?"))
    texts = " ".join(o["text"] for o in out)
    assert "root.location = Athens, Greece" in texts
    assert "scar" not in texts and "appearance" not in texts and "Makedonias" not in texts and "83" not in texts


def test_history_fails_closed_without_the_profiles_rules(monkeypatch, led):
    """Broken world (review r1): the scrubber raised → lines left unscrubbed."""
    monkeypatch.setenv(FL.ENV_FLAG, "1")
    from ghost_agent.core.bus import MemoryBus
    led.record(store="profile", category="root", key="location", value="Athens", event="stated", source={"said": "I live in Athens"})
    assert led.lines_for_question("where do i live?")                  # the row IS found without a profile …
    prof = MagicMock()
    prof.on_demand_fields = MagicMock(return_value=set())
    prof.is_on_demand = MagicMock(return_value=False)        # r2: a truthy MagicMock excluded every row
    prof.egress_scrubber = MagicMock(side_effect=RuntimeError("no profile"))
    bus = MemoryBus(profile_memory=prof, fact_ledger=led)
    assert asyncio.run(bus._fetch_history("where do i live?")) == []


def test_recall_history_only_for_a_question_about_the_owner(monkeypatch, led):
    """Broken world (review r1): "history of python" / "what changed in
    python 3.13" got `interests.languages = python` because a history word
    alone opened the gate; it only widens the budget."""
    monkeypatch.setenv(FL.ENV_FLAG, "1")
    from ghost_agent.tools.memory import tool_recall
    led.record(store="profile", category="interests", key="languages", value="python", event="stated")
    ms = MagicMock(); ms.search_advanced = MagicMock(return_value=[])
    for q in ("history of python", "what changed in python 3.13?"):
        assert FL.HEADER not in asyncio.run(tool_recall(query=q, memory_system=ms, fact_ledger=led))
    assert FL.HEADER in asyncio.run(tool_recall(query="since when do i use python?", memory_system=ms, fact_ledger=led))


def test_restatements_never_hide_the_older_events_a_question_is_about(led):
    """Broken world (review r1): the newest 24 rows were 24 restatements of
    one value, so "where did I live before Patras?" lost the replaced row."""
    led.record(store="profile", category="root", key="location", value="Athens", event="stated", at="2026-05-01T09:00:00")
    led.record(store="profile", category="root", key="location", value="Patras", event="replaced", prev_value="Athens",
               at="2026-07-07T09:00:00", source={"said": "we moved to Patras"})
    for i in range(40):
        led.record(store="profile", category="root", key="location", value="Patras", event="restated", at=1790000000 + i * 3600)
    lines = led.lines_for_question("did i live in athens before patras?")
    assert any('replacing "Athens"' in ln for ln in lines) and any("= Athens — since 2026-05-01" in ln for ln in lines)
    assert sum("confirmed again" in ln for ln in lines) == 1


def test_days_are_local_and_unreadable_stamps_are_undated(led):
    """Broken worlds (review r1): UTC days ("since yesterday" after midnight
    local); an unreadable stamp silently dated today."""
    import datetime as dt
    at = dt.datetime(2026, 7, 7, 0, 30).timestamp()                    # 00:30 LOCAL (UTC says the 6th here)
    assert FL._day(at) == "2026-07-07"
    assert FL._epoch("07/07/2026") is None and FL._epoch("") is None and FL._epoch(0) is None
    assert FL._epoch("2026-07-07 21:28:19 UTC") == FL._epoch("2026-07-07T21:28:19Z") == dt.datetime(
        2026, 7, 7, 21, 28, 19, tzinfo=dt.timezone.utc).timestamp()
    led.record(store="profile", category="x", key="y", value="z", event="stated", at="07/07/2026")
    assert abs(led.history("x", "y")[0]["at"] - time.time()) < 5


def test_provenance_skips_removals_and_expiries(led):
    """Battery F9: the newest row for a value may be its removal — provenance
    is the row that STATED it."""
    led.record(store="profile", category="pet", key="dog_name", value="Hanzo", event="stated", source={"req_id": "r1"})
    led.record(store="profile", category="pet", key="dog_name", value="Hanzo", event="removed", how="delete")
    p = led.provenance_for("pet", "dog_name", "Hanzo")
    assert p and p["event"] == "stated" and p["req_id"] == "r1"


# ── the backfill script ─────────────────────────────────────────────────────

def _owner_turn(tid, request, ts, req_id):
    return {"id": tid, "task_kind": "user_request", "timestamp": ts, "user_request": request, "final_response": "ok.",
            "tool_calls": [], "extra": {"req_id": req_id, "requester_role": ""}, "outcome": "unknown"}


def test_backfill_seeds_the_stores_history_once_and_truthfully(tmp_path):
    """Broken worlds (review r1 MAJORs 2–3, minors 6/11/12): the previous value
    seeded as the LIVE one; a fact attributed to a QUESTION; the graph's mirror
    edge seeded beside its profile row; an undated value dated today; a second
    --apply re-seeding every value whose spacing the ledger folds."""
    import os
    import subprocess
    import sys
    from ghost_agent.memory.graph import GraphMemory
    home = tmp_path / "home"
    mdir = home / "system" / "memory"
    mdir.mkdir(parents=True)
    day = home / "system" / "trajectories" / "2026-07-01"
    day.mkdir(parents=True)
    turns = [_owner_turn("a" * 32, "Hi. We moved to Patras last year, by the way.", "2026-07-01T10:00:00Z", "r-moved"),
             _owner_turn("b" * 32, "do you remember my wife fotini", "2026-07-02T10:00:00Z", "r-ask")]
    (day / "session-x.jsonl").write_text("".join(json.dumps(t) + "\n" for t in turns), encoding="utf-8")
    pm = ProfileMemory(mdir)
    pm.update("root", "location", "Athens, Greece", as_of="2026-05-01T09:00:00")
    pm.update("root", "location", "Patras", as_of="2026-07-03T09:00:00")
    pm.update("relationships", "wife", "Fotini", as_of="2026-07-03T09:00:00")
    data = pm.load_raw()
    data.setdefault("assets", {})["car"] = "Audi  A3"                   # legacy: no stamp, double space
    pm.save(data)
    g = GraphMemory(mdir)
    g.sync_owner_field("location", ["Patras"])                          # the mirror edge user HAS_LOCATION patras
    g.add_triplets([{"subject": "user", "predicate": "LIVES_IN", "object": "Patras"}])
    import sqlite3 as _sq
    with _sq.connect(str(mdir / "knowledge_graph.db")) as _c:          # a legacy edge with no valid_from
        _c.execute("INSERT INTO triplets(subject, predicate, object, weight, timestamp, valid_from)"
                   " VALUES ('user', 'HAS_CONDITION', 'asthma', 1, CURRENT_TIMESTAMP, NULL)")
    script = str(Path(__file__).resolve().parents[1] / "scripts" / "fact_ledger_backfill_4qm.py")
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"), GHOST_HOME=str(home))
    run = lambda *a: subprocess.run([sys.executable, script, *a], capture_output=True, text=True, env=env, timeout=120)  # noqa: E731
    r = run()
    assert r.returncode == 0, r.stderr
    assert FactLedger(mdir).count() == 0                                # a dry run writes nothing
    r = run("--apply")
    assert r.returncode == 0, r.stderr
    led = FactLedger(mdir)
    rows = {(x["store"], x["key"], x["value"]): dict(x) for x in led._conn().execute("SELECT * FROM ledger")}
    live = rows[("profile", "location", "Patras")]
    assert live["event"] == "replaced" and live["prev_value"] == "Athens, Greece"
    assert live["said"] == "We moved to Patras last year" and live["req_id"] == "r-moved"
    assert rows[("profile", "location", "Athens, Greece")]["event"] == "backfilled"
    wife = rows[("profile", "wife", "Fotini")]
    assert wife["said"] == "" and wife["how"] == "unattributed"         # a question attributes nothing
    car = rows[("profile", "car", "Audi A3")]
    assert "undated" in car["how"]
    # the profile row is the record: neither its mirror edge nor a same-valued edge under
    # another predicate is seeded (two rows would date one fact twice); an edge with no
    # profile twin is
    assert ("graph", "lives_in", "patras") not in rows and ("graph", "has_location", "patras") not in rows
    assert ("graph", "has_condition", "asthma") in rows
    asthma = rows[("graph", "has_condition", "asthma")]
    assert "undated" in asthma["how"]
    rendered = "\n".join(led.render_lines([asthma, car, rows[("profile", "location", "Athens, Greece")]]))
    assert "date unknown" in rendered and time.strftime("%Y-%m-%d") not in rendered     # r2: never "since today"
    assert "Athens, Greece — on record 2026-05-01, later replaced" in rendered
    assert "stated" not in rendered and "— since 2" not in rendered       # seeded rows never claim a statement date
    n = led.count()
    r = run("--apply")
    assert r.returncode == 0 and "already in the ledger" in r.stdout and FactLedger(mdir).count() == n


def test_requests_to_remember_are_statements_and_hidden_questions_are_not():
    """Broken worlds (review r2): the question opener dropped real statements —
    modal requests to remember, "did I mention", "when I was 20 …", the Greek
    "ποτέ μην …" imperative — and a greeting or a lead clause before the
    question hid it ("hi, do you remember my wife fotini" wrote wife = fotini)."""
    from ghost_agent.memory.attribution import owner_statements, owner_said
    kept = ("can you remember that I live in Athens", "Can you please remember my wife is called Fotini",
            "could you note that my employer is Acme", "would you remember that I moved to Patras",
            "May I remind you that I live in Athens", "Did I mention I moved to Patras",
            "When I was 20 I lived in Patras", "Ποτέ μην ξεχάσεις ότι η γυναίκα μου λέγεται Φωτεινή",
            "hey, we moved to Patras last year")
    for st in kept:
        assert owner_statements("USER: " + st), st
    dropped = ("hi, do you remember my wife fotini", "hey do you remember my wife fotini",
               "I wonder, do you remember my wife fotini", "ok so, what is my wife's name fotini",
               "have you got my wife fotini in your memory", "can you tell me my wife fotini")
    for q in dropped:
        assert "fotini" not in " ".join(c for c, _ in owner_statements("USER: " + q)), q
        assert owner_said("fotini", owner_statements("USER: " + q)) != "stated", q
    # "don't forget …" states, it does not negate (r2 note)
    assert owner_statements("USER: Don't forget I live in Athens.") == [("don't forget i live in athens", False)]
    assert owner_statements("USER: never forget my wife is Fotini") == [("never forget my wife is fotini", False)]
    assert owner_statements("USER: I don't live in Athens any more")[0][1] is True
    # review r3: a relative clause is not a question; a modal + "remind me"/"forget"
    # IS one; a wh-word before we/my/our opens a statement; "have I told you" states;
    # the Greek question word is read with its accent on the raw sentence
    for st, value in (("My wife, who is called Fotini, is a doctor.", "Fotini"),
                      ("My car, which is a BMW, is parked outside.", "BMW"),
                      ("I have two kids, who are called Nikos and Maria.", "Nikos"),
                      ("When we moved to Patras last year we bought a house.", "Patras"),
                      ("What my wife Fotini likes is hiking.", "Fotini"),
                      ("have I told you about my wife fotini", "fotini")):
        assert owner_said(value, owner_statements("USER: " + st)) == "stated", st
    for q, value in (("can you remind me where my wife fotini works", "fotini"),
                     ("could you remind me what car i drive, the bmw", "bmw"),
                     ("can you forget my ex-wife maria", "maria"),
                     ("Πότε σου είπα ότι μένω στην Αθήνα", "Αθήνα"),
                     ("Πού μένω, στην Αθήνα", "Αθήνα")):
        assert owner_said(value, owner_statements("USER: " + q)) != "stated", q
    assert owner_said("Αθήνα", owner_statements("USER: μένω στο σπίτι που αγόρασα στην Αθήνα")) == "stated"   # unaccented που = "that"
    # review r4 (102-sentence probe): an auxiliary before a noun phrase asks; an
    # indirect question asks; Greek ";" is the question mark; a tag question asks;
    # a colon lead hides nothing; instructions with also/just/make a note state;
    # Greek states person in the verb
    for q, value in (("is my wife fotini a doctor", "fotini"), ("was my previous car the audi", "audi"),
                     ("does my wife fotini still work at evangelismos", "fotini"),
                     ("I wonder if you remember my wife fotini", "fotini"),
                     ("I want to know if you still remember my wife fotini", "fotini"),
                     ("η γυναίκα μου είναι η Φωτεινή;", "Φωτεινή"), ("μένω ακόμα στην Αθήνα;", "Αθήνα"),
                     ("you know my wife fotini, right", "fotini"), ("quick question: is my wife fotini a doctor", "fotini"),
                     ("Any idea where I live, Athens or Patras", "Athens")):
        assert owner_said(value, owner_statements("USER: " + q)) != "stated", q
    for st, value in (("could you also remember that I live in Athens", "Athens"),
                      ("can you just note that my wife is Fotini", "Fotini"),
                      ("could you make a note that I live in Athens", "Athens"),
                      ("Can I ask you to remember that I live in Athens", "Athens"),
                      ("I don't know if you remember, but my wife is Fotini", "Fotini"),
                      ("Μετακόμισα στην Πάτρα", "Πάτρα"), ("Λοιπόν, μετακομίσαμε στην Πάτρα", "Πάτρα"),
                      ("Παντρεύτηκα τη Φωτεινή", "Φωτεινή")):
        assert owner_said(value, owner_statements("USER: " + st)) == "stated", st
    from ghost_agent.memory.attribution import owner_clause_for
    assert owner_clause_for("1985-03-05", "USER: my birthday is 1985-03-05") == "my birthday is 1985-03-05"
    assert owner_clause_for("the BMW", "USER: and the kids are fine, I drive a bmw") == "I drive a bmw"


def test_undated_and_previous_values_render_truthfully(led):
    """Broken worlds (review r2): a backfilled value with no stamp rendered
    "since <the backfill day>"; a backfilled PREVIOUS value rendered as live."""
    led.record(store="profile", category="assets", key="car", value="Audi A3", event="backfilled", how="undated")
    led.record(store="profile", category="assets", key="car", value="BMW", event="backfilled", how="previous value",
               at="2026-05-01T09:00:00")
    lines = led.render_lines(led.history("assets", "car"))
    assert "- assets.car = Audi A3 — on record since date unknown" in lines and time.strftime("%Y-%m-%d") not in " ".join(lines)
    assert "- assets.car = BMW — on record 2026-05-01, later replaced" in lines


def test_backfilled_rows_say_on_record_and_stated_rows_say_since(led):
    """Broken world (§4QM replay, h08/h09): a seeded graph row is dated by the
    edge's creation (2026-09-21) and rendered "since 2026-09-21 (you said: …)",
    so the model told the owner "you told me on 2026-09-21" about a fact stated
    in July. Only a row the live agent wrote at the owner's turn may say since."""
    led.record(store="graph", category="graph", key="owns", value="bmw 118i", event="backfilled",
               source={"said": "I own a BMW 118i"}, at="2026-09-21T16:03:03")
    led.record(store="profile", category="pet", key="dog_name", value="Hanzo", event="stated",
               source={"said": "my dogs name is Hanzo"}, at="2026-10-11T16:18:09")
    lines = led.render_lines(led.history("graph", "owns") + led.history("pet", "dog_name"))
    assert '- graph.owns = bmw 118i — on record since 2026-09-21 (you said: "I own a BMW 118i")' in lines
    assert '- pet.dog_name = Hanzo — since 2026-10-11 (you said: "my dogs name is Hanzo")' in lines
    assert not any("since 2026-09-21" in ln and "on record" not in ln for ln in lines)


def test_a_questions_words_match_every_row_of_a_fact_not_only_the_newest(led):
    """Broken world (review r2): the group is matched on its newest row only,
    so a wordless restatement hid the fact the older row's words had found."""
    led.record(store="profile", category="root", key="location", value="Athens", event="stated",
               source={"said": "we moved to Athens, I live there"}, at="2026-05-01T09:00:00")
    led.record(store="profile", category="root", key="location", value="Athens", event="stated", at="2026-06-01T09:00:00")
    lines = led.lines_for_question("where do i live?")
    assert len(lines) == 1 and lines[0].startswith("- root.location = Athens — since 2026-06-01")


def test_deleting_a_replaced_singleton_records_its_previous_value_too(profile, led):
    """Broken world (review r2): `delete` recorded the live value only; the
    previous value kept beside it vanished with its "since …" row intact."""
    profile.update("root", "location", "Athens")
    profile.update("root", "location", "Patras")
    profile.delete("root", "location")
    gone = [(r["value"], r["how"]) for r in led.history("root", "location") if r["event"] == "removed"]
    assert sorted(gone) == [("Athens", "delete (previous)"), ("Patras", "delete")]
    profile.update("root", "name", "Vasilis")                   # the seeded "User" is kept as previous …
    profile.delete("root", "name")
    assert [r["value"] for r in led.history("root", "name") if r["event"] == "removed"] == ["Vasilis"]   # … but is no fact (r3)


def test_the_extractors_graph_rows_name_the_owners_clause_per_edge(tmp_path, led):
    """Broken world (review r2): the extractor's graph rows carried the drain's
    empty request — no id, no words — while its profile rows had them."""
    from ghost_agent.memory.graph import GraphMemory
    g = GraphMemory(tmp_path)
    g.ledger = led
    ep = "USER: hi. We moved to Patras last year. my wife is Fotini\nAI: noted."
    g.add_triplets([{"subject": "user", "predicate": "LIVES_IN", "object": "Patras"},
                    {"subject": "user", "predicate": "MARRIED_TO", "object": "Fotini"}],
                   source={"req_id": "r9", "via": "extractor", "episode": ep})
    rows = {r["key"]: r for r in _all(led)}
    assert rows["lives_in"]["said"] == "We moved to Patras last year" and rows["lives_in"]["req_id"] == "r9"
    assert rows["married_to"]["said"] == "my wife is Fotini" and rows["married_to"]["via"] == "extractor"
