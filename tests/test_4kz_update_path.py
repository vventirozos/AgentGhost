"""§4KZ (2026-10-03): when a fact about the owner CHANGES. The profile is the
authority; its graph and vector mirrors are synced to it; no model call
deletes stored facts; the newest statement wins. Each test names the world it
fails in."""
import asyncio
import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core.agent import GhostAgent, GhostContext
from ghost_agent.memory.attribution import owner_said, owner_statements, PREDICATE_CUES
from ghost_agent.memory.graph import GraphMemory
from ghost_agent.memory.profile import ProfileMemory
from ghost_agent.memory.vector import VectorMemory
from ghost_agent.tools import memory as M
from ghost_agent.utils.logging import request_id_context


def _edges(gm):
    with sqlite3.connect(gm.db_path) as c:
        return set(c.execute("select subject, predicate, object from triplets where valid_until is null"))


def _vm():
    return VectorMemory(Path(tempfile.mkdtemp()), "http://mock-url")


def _identity(vm):
    return sorted(vm.collection.get(where={"type": "identity"}, include=["documents"])["documents"])


def _stores(tmp_path):
    return ProfileMemory(tmp_path), GraphMemory(tmp_path), _vm()


def _update(pm, gm, vm, cat, key, value, rid):
    t = request_id_context.set(rid)
    try:
        return asyncio.run(M.tool_update_profile(cat, key, value, profile_memory=pm, graph_memory=gm, memory_system=vm))
    finally:
        request_id_context.reset(t)


# ── the profile and its mirrors ──────────────────────────────────────────────
def test_a_changed_wife_name_leaves_one_value_in_every_store(tmp_path):
    """Fails where HAS_WIFE fotinh, fotini and φωτεινή all stayed live, and
    the vector kept every spelling."""
    pm, gm, vm = _stores(tmp_path)
    for i, v in enumerate(["Fotinh", "Fotini", "Φωτεινή"]):
        _update(pm, gm, vm, "relationships", "wife", v, f"req-{i}")
    wife = [e for e in _edges(gm) if e[0] == "user" and e[1] == "HAS_WIFE"]
    assert wife == [("user", "HAS_WIFE", "φωτεινή")]
    assert _identity(vm) == ["User wife is Φωτεινή"]


def test_a_value_changed_back_is_written(tmp_path):
    """Fails where the bus LRU dropped Athens → Berlin → Athens as a repeat
    and still said SUCCESS."""
    from ghost_agent.core.bus import MemoryBus
    pm, gm, vm = _stores(tmp_path)
    bus = MemoryBus(vector_memory=vm, graph_memory=gm, profile_memory=pm)
    for i, v in enumerate(["Athens", "Berlin", "Athens"]):
        t = request_id_context.set(f"req-{i}")
        try:
            asyncio.run(M.tool_update_profile("root", "location", v, memory_bus=bus))
        finally:
            request_id_context.reset(t)
    assert pm.load()["root"]["location"] == "Athens"
    assert [e for e in _edges(gm) if e[1] == "HAS_LOCATION"] == [("user", "HAS_LOCATION", "athens")]
    assert _identity(vm) == ["User location is Athens"]


def test_a_key_never_touches_its_sibling_key(tmp_path):
    """Fails where `smart_update` of "User wife is …" deleted "User
    wife_birthdate is …" (distance 0.25)."""
    pm, gm, vm = _stores(tmp_path)
    _update(pm, gm, vm, "relationships", "wife_birthdate", "January 10, 1982", "req-a")
    _update(pm, gm, vm, "relationships", "wife", "Fotini", "req-b")
    assert _identity(vm) == ["User wife is Fotini", "User wife_birthdate is January 10, 1982"]


def test_a_list_field_keeps_every_item(tmp_path):
    pm, gm, vm = _stores(tmp_path)
    _update(pm, gm, vm, "relationships", "children", "Thodoris", "req-a")
    _update(pm, gm, vm, "relationships", "children", "Leonidas", "req-b")
    assert _identity(vm) == ["User children is Leonidas", "User children is Thodoris"]


def test_a_refused_mirror_keeps_the_old_value(tmp_path):
    vm = _vm()
    vm.add("User car is a Fiat", {"type": "identity", "timestamp": "2026-01-01T00:00:00Z"})
    vm.add("User car is a BMW", {"type": "auto", "timestamp": "2026-01-01T00:00:00Z"})
    with pytest.raises(Exception):
        vm.sync_owner_field("car", ["a BMW"])
    assert _identity(vm) == ["User car is a Fiat"]


# ── the extractor ────────────────────────────────────────────────────────────
def _agent(tmp_path, reply):
    ctx = MagicMock(spec=GhostContext)
    ctx.args = MagicMock()
    ctx.args.smart_memory = 0.5
    ctx.llm_client = MagicMock()
    ctx.llm_client.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": json.dumps(reply)}}]})
    ctx.memory_system = _vm()
    ctx.graph_memory = GraphMemory(tmp_path)
    ctx.profile_memory = ProfileMemory(tmp_path)
    ctx.adaptive_threshold = None
    return GhostAgent(ctx)


def _run(a, episode, as_of=None):
    asyncio.run(a.run_smart_memory_task(episode, "m", 0.5, as_of=as_of))


def test_no_model_call_deletes_a_stored_fact(tmp_path):
    a = _agent(tmp_path, {"score": 0.95, "fact": "The user lives in Berlin.", "graph_triplets": []})
    a.context.memory_system.add("The user lives in Athens.", {"type": "auto", "timestamp": "2026-01-01T00:00:00Z"})
    _run(a, "USER: I moved, I live in Berlin now\nAI: noted")
    docs = a.context.memory_system.collection.get(include=["documents"])["documents"]
    assert "The user lives in Athens." in docs and a.context.llm_client.chat_completion.await_count == 1


def test_an_owner_profile_change_from_the_extractor_syncs_the_mirrors(tmp_path):
    """Fails where an extractor-driven change left HAS_LOCATION athens live."""
    a = _agent(tmp_path, {"score": 0.95, "fact": "The user lives in Berlin.",
                          "profile_update": {"category": "root", "key": "location", "value": "Berlin"},
                          "graph_triplets": [{"subject": "User", "predicate": "LIVES_IN", "object": "Berlin"}]})
    a.context.profile_memory.update("root", "location", "Athens")
    a.context.graph_memory.sync_owner_field("location", ["Athens"])
    _run(a, "USER: I moved to Berlin, I live there now\nAI: noted")
    e = _edges(a.context.graph_memory)
    assert ("user", "HAS_LOCATION", "berlin") in e and ("user", "HAS_LOCATION", "athens") not in e
    assert ("user", "LIVES_IN", "berlin") in e


def test_a_trip_never_replaces_the_home(tmp_path):
    """Fails in the world of 08-15: "I'm in Kyllini this weekend" expired
    the owner's home."""
    a = _agent(tmp_path, {"score": 0.9, "fact": "The user lives in Kyllini.",
                          "graph_triplets": [{"subject": "User", "predicate": "LIVES_IN", "object": "Kyllini"},
                                             {"subject": "User", "predicate": "LOCATED_IN", "object": "Kyllini"}]})
    a.context.graph_memory.add_triplets([{"subject": "user", "predicate": "LIVES_IN", "object": "athens"}])
    _run(a, "USER: I'm in Kyllini this weekend with the kids. Any beach tips?\nAI: …")
    e = _edges(a.context.graph_memory)
    assert ("user", "LIVES_IN", "athens") in e and ("user", "LIVES_IN", "kyllini") not in e


def test_a_role_play_across_sentences_writes_nothing(tmp_path):
    a = _agent(tmp_path, {"score": 0.95, "fact": "Likes living in London with Maria.",
                          "graph_triplets": [{"subject": "User", "predicate": "MARRIED_TO", "object": "Maria"},
                                             {"subject": "User", "predicate": "LIVES_IN", "object": "London"}]})
    a.context.graph_memory.add_triplets([{"subject": "user", "predicate": "MARRIED_TO", "object": "fotini"}])
    _run(a, "USER: Let's play a game. You are my wife Maria. We live in London.\nAI: Sure!")
    e = _edges(a.context.graph_memory)
    assert ("user", "MARRIED_TO", "fotini") in e and ("user", "MARRIED_TO", "maria") not in e
    assert not a.context.memory_system.collection.get(include=["documents"])["documents"]


def test_x_not_y_states_x_and_corrects_y(tmp_path):
    """Fails where "spelled Φωτεινή, not Fotini" was dropped whole."""
    a = _agent(tmp_path, {"score": 0.95, "fact": "",
                          "graph_triplets": [{"subject": "User", "predicate": "MARRIED_TO", "object": "Φωτεινή"},
                                             {"subject": "User", "predicate": "MARRIED_TO", "object": "not Fotini"}]})
    a.context.graph_memory.add_triplets([{"subject": "user", "predicate": "MARRIED_TO", "object": "fotini"}])
    _run(a, "USER: my wife's name is spelled Φωτεινή, not Fotini\nAI: noted")
    e = _edges(a.context.graph_memory)
    assert ("user", "MARRIED_TO", "φωτεινή") in e and ("user", "MARRIED_TO", "fotini") not in e


def test_a_correction_removes_only_that_kind_of_fact(tmp_path):
    """Fails where "I'm not going to Athens" deleted LIVES_IN, BORN_IN and
    WORKS_IN athens."""
    a = _agent(tmp_path, {"score": 0.1, "fact": "",
                          "graph_triplets": [{"subject": "User", "predicate": "TRAVELING_TO", "object": "not Athens"}]})
    a.context.graph_memory.add_triplets([{"subject": "user", "predicate": "LIVES_IN", "object": "athens"},
                                         {"subject": "user", "predicate": "BORN_IN", "object": "athens"},
                                         {"subject": "user", "predicate": "TRAVELING_TO", "object": "athens"}])
    _run(a, "USER: I'm not going to Athens this weekend after all\nAI: ok")
    e = _edges(a.context.graph_memory)
    assert ("user", "LIVES_IN", "athens") in e and ("user", "BORN_IN", "athens") in e
    assert ("user", "TRAVELING_TO", "athens") not in e


def test_we_sold_the_bmw_retracts_it_everywhere(tmp_path):
    """Fails where "we sold the BMW" never reached the extractor, and a sale
    left OWNS/DRIVES/HAS_CAR and the profile car live."""
    a = _agent(tmp_path, {"score": 0.9, "fact": "The user sold the BMW.",
                          "graph_triplets": [{"subject": "User", "predicate": "SOLD", "object": "BMW"}]})
    pm = a.context.profile_memory
    pm.update("assets", "car", "BMW 320d")
    a.context.graph_memory.add_triplets([{"subject": "user", "predicate": "OWNS", "object": "bmw"},
                                         {"subject": "user", "predicate": "HAS_CAR", "object": "bmw 320d"}])
    _run(a, "USER: we sold the BMW last week\nAI: ok")
    e = _edges(a.context.graph_memory)
    assert not any("bmw" in x[2] for x in e)
    assert "car" not in (pm.load().get("assets") or {})


def test_an_age_update_is_attributed_before_it_is_anchored(tmp_path):
    """Fails where "is now 10" became "born ~2016-04" BEFORE the gate, words
    the owner never typed, and every age update was refused."""
    a = _agent(tmp_path, {"score": 0.95, "fact": "The user's son Thodoris is now 10 years old.",
                          "profile_update": {"category": "relationships", "key": "son",
                                             "value": "Thodoris is now 10 years old"},
                          "graph_triplets": []})
    _run(a, "USER: my son Thodoris is now 10 years old\nAI: congrats")
    son = str(a.context.profile_memory.load()["relationships"]["son"])
    assert "born ~" in son and "now born" not in son


def test_ages_are_never_graph_facts(tmp_path):
    g = GraphMemory(tmp_path)
    g.add_triplets([{"subject": "leonidas", "predicate": "AGE", "object": "5"},
                    {"subject": "leonidas", "predicate": "HAS_AGE", "object": "5"},
                    {"subject": "leonidas", "predicate": "BORN_ON", "object": "2026-03-12"}])
    assert _edges(g) == {("leonidas", "HAS_BIRTHDATE", "2026-03-12")}


def test_an_older_statement_never_replaces_a_newer_one(tmp_path):
    """Fails where a re-queued older journal item expired "lives in Patras"
    back to Athens."""
    g = GraphMemory(tmp_path)
    g.add_triplets([{"subject": "user", "predicate": "LIVES_IN", "object": "patras"}], as_of=2000.0)
    g.add_triplets([{"subject": "user", "predicate": "LIVES_IN", "object": "athens"}], as_of=1000.0)
    assert ("user", "LIVES_IN", "patras") in _edges(g) and ("user", "LIVES_IN", "athens") not in _edges(g)


def test_journal_items_carry_when_they_were_said(tmp_path):
    from ghost_agent.memory.journal import MemoryJournal
    j = MemoryJournal(tmp_path)
    j.append("smart_memory", {"text": "USER: hi", "model": "m"})
    assert isinstance(j.load()[0].get("ts"), float)


@pytest.mark.parametrize("pred,stmt,ok", [("LIVES_IN", "i'm in kyllini this weekend", False),
                                          ("LIVES_IN", "we moved to kyllini", True),
                                          ("WORKS_AT", "i work at globex now", True),
                                          ("MARRIED_TO", "maria is a friend of mine", False)])
def test_a_single_valued_fact_needs_its_kind_of_statement(pred, stmt, ok):
    value = stmt.split()[-1] if pred != "MARRIED_TO" else "maria"
    if pred == "LIVES_IN":
        value = "kyllini"
    if pred == "WORKS_AT":
        value = "globex"
    said = owner_said(value, owner_statements("USER: " + stmt), cue=PREDICATE_CUES[pred])
    assert (said == "stated") is ok


@pytest.mark.parametrize("text", ["we sold the BMW", "We moved to Berlin last month.", "πουλήσαμε το BMW",
                                  "Μετακομίσαμε στο Βερολίνο."])
def test_we_reaches_the_extractor(tmp_path, text):
    a = _agent(tmp_path, {"score": 0.1, "fact": "", "graph_triplets": []})
    _run(a, f"USER: {text}\nAI: ok")
    assert a.context.llm_client.chat_completion.await_count == 1


# ── ranking ──────────────────────────────────────────────────────────────────
def test_the_profile_mirror_outranks_a_stale_synthesis():
    vm = _vm()
    vm.add("The user lives in Athens with his family and works nearby.",
           {"type": "synthesis", "timestamp": "2026-07-07T00:00:00Z"})
    vm.add("User location is Thrakomakedones", {"type": "identity", "timestamp": "2026-10-03T00:00:00Z"})
    sel = vm._search_selection("where does the user live, user location")
    assert sel and sel[0]["doc"] == "User location is Thrakomakedones"


def test_a_stale_auto_name_row_is_not_a_master_summary():
    vm = _vm()
    vm.add("The user's wife's name is Fotini.", {"type": "auto", "timestamp": "2026-01-01T00:00:00Z"})
    sel = vm._search_selection("what is the name of the user's wife")
    assert all("MASTER SUMMARY" not in vm._render_item(i) for i in sel)


def test_owner_facts_newest_first_with_dates(tmp_path):
    g = GraphMemory(tmp_path)
    g.add_triplets([{"subject": "user", "predicate": "LIVES_IN", "object": "athens"}])
    with sqlite3.connect(g.db_path) as c:
        c.execute("UPDATE triplets SET timestamp='2026-08-05 00:00:00'")
    g.add_triplets([{"subject": "user", "predicate": "HAS_ADDRESS", "object": "thrakomakedones"}])
    out = g.owner_facts_matching("where do I live, my home address")
    assert "Thrakomakedones" in out[0] and "(as of" in out[0]


@pytest.mark.parametrize("q", ["who is my wife", "my sons birthdays"])
def test_owner_facts_reach_wife_and_sons(tmp_path, q):
    g = GraphMemory(tmp_path)
    g.add_triplets([{"subject": "user", "predicate": "MARRIED_TO", "object": "fotini"},
                    {"subject": "user", "predicate": "HAS_SON", "object": "leonidas"},
                    {"subject": "leonidas", "predicate": "BORN_ON", "object": "2026-03-12"}])
    assert g.owner_facts_matching(q)


def test_hydration_carries_owner_facts_by_kind(tmp_path):
    from ghost_agent.core.bus import MemoryBus
    g = GraphMemory(tmp_path)
    g.add_triplets([{"subject": "user", "predicate": "LIVES_IN", "object": "thrakomakedones"}])
    items = asyncio.run(MemoryBus(graph_memory=g)._fetch_graph("where do I live?"))
    assert any("Thrakomakedones" in i["text"] for i in items)




def test_the_extractor_passes_when_it_was_said(tmp_path):
    """Fails where the drain's statement time was dropped and an older item
    replaced a newer home."""
    a = _agent(tmp_path, {"score": 0.1, "fact": "",
                          "graph_triplets": [{"subject": "User", "predicate": "LIVES_IN", "object": "Athens"}]})
    a.context.graph_memory.add_triplets([{"subject": "user", "predicate": "LIVES_IN", "object": "patras"}], as_of=2000.0)
    _run(a, "USER: I live in Athens\nAI: ok", as_of=1000.0)
    e = _edges(a.context.graph_memory)
    assert ("user", "LIVES_IN", "patras") in e and ("user", "LIVES_IN", "athens") not in e


def test_a_sale_is_never_stored_as_a_fact(tmp_path):
    a = _agent(tmp_path, {"score": 0.95, "fact": "The user sold the BMW last week.", "graph_triplets": []})
    _run(a, "USER: I sold the BMW last week\nAI: ok")
    assert not a.context.memory_system.collection.get(include=["documents"])["documents"]


def test_an_auto_name_row_never_outranks_the_profile_mirror():
    vm = _vm()
    vm.add("The user's wife's name is Fotini.", {"type": "auto", "timestamp": "2026-01-01T00:00:00Z"})
    vm.add("User wife is Φωτεινή", {"type": "identity", "timestamp": "2026-10-03T00:00:00Z"})
    sel = vm._search_selection("the user's wife name")
    assert sel and sel[0]["doc"] == "User wife is Φωτεινή"


def test_being_read_often_does_not_make_a_fact_newer():
    """Fails where `last_accessed` reset a stale row's age."""
    vm = _vm()
    vm.add("The user's favourite food is souvlaki.", {"type": "auto", "timestamp": "2025-01-01T00:00:00Z",
                                                       "last_accessed": "2026-10-03T00:00:00Z"})
    vm.add("The user's favourite food is gemista.", {"type": "auto", "timestamp": "2026-10-01T00:00:00Z"})
    sel = vm._search_selection("the user's favourite food")
    order = [i["doc"] for i in sel]
    assert order.index("The user's favourite food is gemista.") < order.index("The user's favourite food is souvlaki.")


def test_an_identity_question_keeps_its_query_relevance():
    """Fails where the canned probe claimed the rows and the bus gate then
    found no query-batch candidate for "what is my name?"."""
    vm = _vm()
    vm.add("The user's name is Bill.", {"type": "manual", "timestamp": "2026-10-01T00:00:00Z"})
    vm.add("The weather in Athens is sunny.", {"type": "auto", "timestamp": "2026-10-01T00:00:00Z"})
    items = vm.search_items("what is my name?", min_relevance_dist=0.42)
    assert any("Bill" in i["text"] for i in items)
