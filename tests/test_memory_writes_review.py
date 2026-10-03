"""Fresh review of owner-memory writes (profile, graph, episodes, forget,
reset_all) — 2026-10-03. Each test names the world it fails in."""
import asyncio
import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ghost_agent.memory.graph import GraphMemory
from ghost_agent.memory.profile import ProfileMemory
from ghost_agent.tools import memory as M


class _Vec:
    class collection:                                   # noqa: N801
        @staticmethod
        def query(**k):
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        @staticmethod
        def get(**k):
            return {"ids": [], "documents": [], "metadatas": []}

        @staticmethod
        def delete(**k):
            pass

    def get_library(self):
        return []

    def delete_document_by_name(self, n):
        pass

    def delete_fragment(self, t):
        pass


def _stores(tmp_path):
    pm = ProfileMemory(tmp_path)
    pm.update("root", "name", "Vasilis")
    pm.update("relationships", "wife_name", "Fotini")
    pm.update("assets", "vehicles", "BMW 118i, Ducati Streetfighter V4S, Sym scooter")
    pm.update("interests", "hobbies", "cars, jiu jitsu")
    pm.update("root", "address", "12 Oak Street, Athens")
    gm = GraphMemory(tmp_path)
    gm.add_triplets([{"subject": "leonidas", "predicate": "IS_SON_OF", "object": "fotini"},
                     {"subject": "user", "predicate": "MARRIED_TO", "object": "fotini"},
                     {"subject": "user", "predicate": "HAS_CHILD", "object": "leonidas"},
                     {"subject": "mortimer", "predicate": "IS_A", "object": "iguana"},
                     {"subject": "evolmonkey", "predicate": "IS_A", "object": "postgresql services company"}])
    return pm, gm


def _forget(tmp_path, target, pm, gm, ep=None):
    return asyncio.run(M.tool_unified_forget(target, tmp_path / "nosb", _Vec(), pm, gm, episodic_memory=ep))


def _edges(gm):
    with sqlite3.connect(gm.db_path) as c:
        return set(c.execute("select subject, predicate, object from triplets where valid_until is null"))


# ── forget: expansion, profile, graph ────────────────────────────────────────
def test_forgetting_a_child_never_reaches_his_mother(tmp_path):
    """Fails in the world where the expansion followed IS_SON_OF to fotini and
    deleted the owner's wife's facts and `user MARRIED_TO fotini`."""
    pm, gm = _stores(tmp_path)
    _forget(tmp_path, "Leonidas", pm, gm)
    assert ("user", "MARRIED_TO", "fotini") in _edges(gm)
    assert pm.load()["relationships"]["wife_name"] == "Fotini"


def test_an_alias_is_still_followed_and_a_class_never_reaches_its_instances(tmp_path):
    pm, gm = _stores(tmp_path)
    # a class link is not followed at all (third review: `hermes IS_A llm`
    # swept every edge naming "llm"); an alias is, both ways
    assert gm.get_connected_entities("mortimer") == []
    gm.add_triplets([{"subject": "bobby", "predicate": "ALSO_KNOWN_AS", "object": "robert"}])
    assert gm.get_connected_entities("bobby") == ["robert"] and gm.get_connected_entities("robert") == ["bobby"]


@pytest.mark.parametrize("target,field", [("BMW", "assets.vehicles"), ("jiu jitsu", "interests.hobbies"),
                                          ("Athens", "root.address")])
def test_a_field_that_mentions_the_target_is_listed_not_deleted(tmp_path, target, field):
    """Fails in the world where forgetting one vehicle deleted all three."""
    pm, gm = _stores(tmp_path)
    out = _forget(tmp_path, target, pm, gm)
    cat, key = field.split(".")
    assert key in pm.load()[cat] and field in out and "NOT changed" in out


@pytest.mark.parametrize("target", ["name", "user", "Vasilis", "wife"])
def test_the_owners_identity_is_never_forgotten_by_a_loose_word(tmp_path, target):
    pm, gm = _stores(tmp_path)
    _forget(tmp_path, target, pm, gm)
    assert pm.load()["root"]["name"] == "Vasilis"
    assert ("user", "MARRIED_TO", "fotini") in _edges(gm)


def test_an_explicit_field_is_forgotten(tmp_path):
    pm, gm = _stores(tmp_path)
    _forget(tmp_path, "root.address", pm, gm)
    assert "address" not in pm.load()["root"]


def test_a_list_item_is_pruned_alone(tmp_path):
    pm, gm = _stores(tmp_path)
    pm.update("assets", "pets", "Hanzo the dog")
    pm.update("assets", "pets", "Mortimer the iguana")
    _forget(tmp_path, "Mortimer", pm, gm)
    assert pm.load()["assets"]["pets"] in ("Hanzo the dog", ["Hanzo the dog"])


# ── graph decay ──────────────────────────────────────────────────────────────
def test_decay_keeps_owner_facts_and_prunes_noise(tmp_path):
    """Fails in the world where every weight-1 edge older than 45 days went —
    the owner's marriage, children and vehicles among them."""
    gm = GraphMemory(tmp_path)
    gm.add_triplets([{"subject": "user", "predicate": "OWNS", "object": "bmw 118i"},
                     {"subject": "leonidas", "predicate": "IS_SON_OF", "object": "fotini"},
                     {"subject": "weather", "predicate": "MENTIONS", "object": "rain"}])
    with sqlite3.connect(gm.db_path) as c:
        c.execute("UPDATE triplets SET timestamp = datetime('now', '-90 days')")
    gm.prune_stale_edges(max_age_days=45)
    assert _edges(gm) == {("user", "OWNS", "bmw 118i"), ("leonidas", "IS_SON_OF", "fotini")}


# ── update_profile and the store ─────────────────────────────────────────────
@pytest.mark.parametrize("value", [[], {}, "null", "​", "undefined"])
def test_an_empty_looking_value_changes_nothing(tmp_path, value):
    pm = ProfileMemory(tmp_path)
    pm.update("root", "name", "Vasilis")
    out = asyncio.run(M.tool_update_profile(category="root", key="name", value=value, profile_memory=pm))
    assert "Nothing was changed" in out and pm.load()["root"]["name"] == "Vasilis"


@pytest.mark.parametrize("value", [None, "", "null", " "])
def test_the_store_refuses_an_empty_value(tmp_path, value):
    """Fails in the world where an extractor's null was stored as "None"."""
    pm = ProfileMemory(tmp_path)
    pm.update("root", "name", "Vasilis")
    assert pm.update("root", "name", value).startswith("Error")
    assert pm.load()["root"]["name"] == "Vasilis"


def test_a_replaced_singleton_keeps_one_previous_value(tmp_path):
    pm = ProfileMemory(tmp_path)
    pm.update("root", "name", "Vasilis")
    out = pm.update("root", "name", "Leonidas")
    assert "(was: 'Vasilis')" in out
    pm.update("root", "name", "Thodoris")
    raw = pm.load_raw()["root"]["name"]
    assert raw["previous"]["v"] == "Leonidas" and "previous" not in raw["previous"]


def test_deleting_a_field_removes_its_graph_edge_and_every_fragment(tmp_path):
    pm = ProfileMemory(tmp_path)
    pm.update("assets", "pets", "Hanzo")
    pm.update("assets", "pets", "Mortimer")
    gm = GraphMemory(tmp_path)
    gm.add_triplets([{"subject": "user", "predicate": "HAS_PETS", "object": "hanzo"},
                     {"subject": "user", "predicate": "HAS_PETS", "object": "mortimer"}])
    vec = MagicMock()
    asyncio.run(M.tool_update_profile(category="assets", key="pets", value="", profile_memory=pm,
                                      memory_system=vec, graph_memory=gm))
    assert not _edges(gm)
    # §4KZ: the vector mirror is SYNCED to the (now empty) field
    vec.sync_owner_field.assert_called_once_with("pets", [])


def test_the_dedup_check_reads_the_canonical_field(tmp_path):
    pm = ProfileMemory(tmp_path)
    pm.update("assets", "vehicle", "Volvo")                 # filed as assets.car
    out = asyncio.run(M.tool_update_profile(category="assets", key="vehicle", value="Volvo", profile_memory=pm))
    assert out.startswith("NOOP")


# ── reset_all ────────────────────────────────────────────────────────────────
# (§4KX r8) the request-wording gate opened on negations and questions and
# refused "yes": reset_all is now preview → confirm in a later turn
@pytest.mark.parametrize("request_text", ["wipe all your memory", "Don't wipe all your memory!", "clean up the sandbox"])
def test_reset_all_never_wipes_in_the_call_that_asks(request_text):
    """Fails in the world where one model call wiped every store."""
    from ghost_agent.memory.lesson_scope import current_request
    vec = MagicMock()
    vec.collection.get.return_value = {"ids": ["a"]}
    tok = current_request.set(request_text)
    try:
        out = asyncio.run(M.tool_knowledge_base(action="reset_all", memory_system=vec, graph_memory=MagicMock()))
    finally:
        current_request.reset(tok)
    assert str(out).startswith("PREVIEW") and not vec.collection.delete.called


# ── episodes ─────────────────────────────────────────────────────────────────
def test_forget_removes_the_episodes_that_name_the_entity(tmp_path):
    from ghost_agent.memory.episodes import EpisodicMemory
    ep = EpisodicMemory(tmp_path)
    ep.record_episode("what did Fotini say about the trip", context="", outcome="ok", lesson="")
    ep.record_episode("weather in Athens", context="", outcome="ok", lesson="")
    pm, gm = _stores(tmp_path)
    out = _forget(tmp_path, "Fotini", pm, gm, ep)
    assert ep.count() == 1 and "Episodes: Forgot 1" in out
    rows = [json.loads(l) for l in (tmp_path / "episodes_forgotten.jsonl").read_text().splitlines()]
    assert rows[0]["trigger"].startswith("what did Fotini")


def test_the_expansion_anchors_on_the_exact_node_and_never_reaches_instances(tmp_path):
    gm = GraphMemory(tmp_path)
    gm.add_triplets([{"subject": "mortimer", "predicate": "IS_A", "object": "iguana"},
                     {"subject": "postgresql db", "predicate": "IS_A", "object": "database"}])
    assert gm.get_connected_entities("iguana") == []          # a class never reaches its instances
    assert gm.get_connected_entities("postgresql") == []      # "postgresql db" is another node


def test_episodes_are_kept_when_their_archive_cannot_be_written(tmp_path):
    from ghost_agent.memory.episodes import EpisodicMemory
    ep = EpisodicMemory(tmp_path)
    ep.record_episode("what did Fotini say", context="", outcome="ok", lesson="")
    (tmp_path / "episodes_forgotten.jsonl").mkdir()              # unwritable as a file
    assert ep.forget_mentions("fotini") == 0 and ep.count() == 1


def test_episodes_match_the_whole_word_only(tmp_path):
    from ghost_agent.memory.episodes import EpisodicMemory
    ep = EpisodicMemory(tmp_path)
    ep.record_episode("join the fotinista club", context="", outcome="ok", lesson="")
    assert ep.forget_mentions("fotini") == 0 and ep.count() == 1
