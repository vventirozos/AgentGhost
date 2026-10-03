"""Re-review of the §4KX fixes (memory writes + lesson producers),
2026-10-03. Each test names the world it fails in."""
import asyncio
import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ghost_agent.memory.graph import GraphMemory
from ghost_agent.memory.profile import ProfileMemory
from ghost_agent.memory.skills import SkillMemory
from ghost_agent.tools import memory as M


class _Vec:
    class collection:                                   # noqa: N801
        query = staticmethod(lambda **k: {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]})
        get = staticmethod(lambda **k: {"ids": [], "documents": [], "metadatas": []})
        delete = staticmethod(lambda **k: None)

    def get_library(self):
        return []

    def delete_document_by_name(self, n):
        pass

    def delete_fragment(self, t):
        pass


def _edges(gm):
    with sqlite3.connect(gm.db_path) as c:
        return set(c.execute("select subject, predicate, object from triplets where valid_until is null"))


def _owner(tmp_path):
    pm = ProfileMemory(tmp_path)
    pm.update("root", "name", "Vasilis")
    pm.update("interests", "topics", "llm inference on apple silicon")
    pm.update("assets", "pet", "Mortimer the iguana")
    pm.update("assets", "car", "Tesla Model 3")
    pm.update("assets", "vehicles", "BMW 118i, Ducati, Sym")
    pm.update("relationships", "wife_name", "Fotini")
    gm = GraphMemory(tmp_path)
    gm.add_triplets([{"subject": "hermes", "predicate": "IS_A", "object": "llm"},
                     {"subject": "user", "predicate": "RESEARCHES", "object": "local llms"},
                     {"subject": "user", "predicate": "INTERESTED_IN", "object": "llm inference"},
                     {"subject": "user", "predicate": "HAS_NAME", "object": "vasilis"},
                     {"subject": "vasilis", "predicate": "WORKS_AT", "object": "evolmonkey"},
                     {"subject": "evolmonkey", "predicate": "IS_A", "object": "postgresql services company"},
                     {"subject": "user", "predicate": "MARRIED_TO", "object": "fotini"},
                     {"subject": "user", "predicate": "LIVES_IN", "object": "thrakomakedones near athens"},
                     {"subject": "anna", "predicate": "WAS_FOUND_IN", "object": "athens"},
                     {"subject": "user", "predicate": "OWNS", "object": "tesla model 3"},
                     {"subject": "user", "predicate": "HAS_ADDRESS", "object": "athens"},
                     {"subject": "user", "predicate": "ALSO_KNOWN_AS", "object": "the boss"},
                     {"subject": "the boss", "predicate": "RUNS", "object": "the office"},
                     {"subject": "vasilis", "predicate": "ALSO_KNOWN_AS", "object": "vas"},
                     {"subject": "vas", "predicate": "PLAYS", "object": "chess"}])
    return pm, gm


def _forget(tmp_path, target, pm, gm, ep=None):
    return asyncio.run(M.tool_unified_forget(target, tmp_path / "nosb", _Vec(), pm, gm, episodic_memory=ep))


# ── forget ───────────────────────────────────────────────────────────────────
def test_forgetting_an_instance_never_sweeps_its_class(tmp_path):
    """Fails in the world where `hermes IS_A llm` led the forget to delete
    every edge and profile item naming "llm" — the owner's interests."""
    pm, gm = _owner(tmp_path)
    _forget(tmp_path, "hermes", pm, gm)
    e = _edges(gm)
    assert ("user", "RESEARCHES", "local llms") in e and ("user", "INTERESTED_IN", "llm inference") in e
    assert pm.load()["interests"]["topics"]
    assert ("hermes", "IS_A", "llm") not in e


@pytest.mark.parametrize("target", ["user", "Vasilis"])
def test_no_expansion_and_no_graph_change_from_a_hub_word_or_the_owners_name(tmp_path, target):
    """Fails in the world where `forget user` expanded through the owner's
    alias ("the boss") or the owner's name through his ("vas"), or where the
    graph leg ran for them at all."""
    pm, gm = _owner(tmp_path)
    before = _edges(gm)
    _forget(tmp_path, target, pm, gm)
    assert _edges(gm) == before


def test_an_expansion_never_lands_on_the_owners_name(tmp_path):
    pm, gm = _owner(tmp_path)
    gm.add_triplets([{"subject": "vasilis", "predicate": "LIKES", "object": "jazz"},
                     {"subject": "vasilis", "predicate": "ALSO_KNOWN_AS", "object": "billy"}])
    _forget(tmp_path, "billy", pm, gm)                # alias of the owner → "vasilis" must not be swept
    assert ("vasilis", "LIKES", "jazz") in _edges(gm)


def test_an_owner_fact_that_mentions_a_place_is_kept_and_reported(tmp_path):
    """Fails in the world where `forget Athens` deleted the owner's
    residence edge (it MENTIONS athens) while the profile leg kept the
    address."""
    pm, gm = _owner(tmp_path)
    out = _forget(tmp_path, "Athens", pm, gm)
    e = _edges(gm)
    assert ("user", "LIVES_IN", "thrakomakedones near athens") in e
    assert ("user", "HAS_ADDRESS", "athens") in e and "kept" in out        # an owner fact ON the node
    assert ("anna", "WAS_FOUND_IN", "athens") not in e
    rows = [json.loads(l) for l in (tmp_path / gm._ARCHIVE_FILENAME).read_text().splitlines()]
    assert any(r["reason"] == "forget_entity" and r["subject"] == "anna" for r in rows)


def test_a_description_is_not_the_entity_but_a_model_is(tmp_path):
    pm, gm = _owner(tmp_path)
    _forget(tmp_path, "postgresql", pm, gm)
    assert ("evolmonkey", "IS_A", "postgresql services company") in _edges(gm)
    assert GraphMemory._node_is("tesla model 3", "tesla") and GraphMemory._node_is("postgresql 17", "postgresql")
    assert not GraphMemory._node_is("postgresql services company", "postgresql")
    assert not GraphMemory._node_is("teslamotors 3", "tesla")


def test_forgetting_a_family_member_removes_the_family_edge(tmp_path):
    pm, gm = _owner(tmp_path)
    _forget(tmp_path, "Fotini", pm, gm)
    assert ("user", "MARRIED_TO", "fotini") not in _edges(gm)


@pytest.mark.parametrize("target,field,gone", [("Mortimer", ("assets", "pet"), True),
                                               ("my old car Tesla", ("assets", "car"), True),
                                               ("BMW", ("assets", "vehicles"), False)])
def test_a_value_about_the_entity_is_forgotten_a_list_is_not(tmp_path, target, field, gone):
    """Fails in the world where `forget Mortimer` left "Mortimer the iguana"
    (over-correction), or `forget BMW` deleted all three vehicles."""
    pm, gm = _owner(tmp_path)
    _forget(tmp_path, target, pm, gm)
    assert (field[1] not in pm.load()[field[0]]) is gone


def test_an_attribute_target_lists_its_fields(tmp_path):
    pm, gm = _owner(tmp_path)
    out = _forget(tmp_path, "my wife", pm, gm)
    assert "relationships.wife_name" in out and pm.load()["relationships"]["wife_name"] == "Fotini"


def test_forget_reaches_a_replaced_value(tmp_path):
    pm = ProfileMemory(tmp_path)
    pm.update("relationships", "son", "Leonidas")
    pm.update("relationships", "son", "Thodoris")
    _forget(tmp_path, "Leonidas", pm, GraphMemory(tmp_path))
    assert "previous" not in pm.load_raw()["relationships"]["son"]


def test_an_unchanged_value_keeps_no_previous(tmp_path):
    pm = ProfileMemory(tmp_path)
    pm.update("root", "name", "Vasilis")
    pm.update("root", "name", "Vasilis")
    assert "previous" not in pm.load_raw()["root"]["name"]


# ── episodes ─────────────────────────────────────────────────────────────────
def _ep(tmp_path):
    from ghost_agent.memory.episodes import EpisodicMemory
    ep = EpisodicMemory(tmp_path)
    ep.record_episode("what did Fotini say about the trip", context="", outcome="ok", lesson="",
                      actions=[{"tool_name": "recall", "tool_args": "{}", "result": "x", "success": 1}])
    ep.record_episode("weather in Athens today", context="", outcome="ok", lesson="")
    return ep


def test_a_mention_keeps_the_episode_a_family_member_does_not(tmp_path):
    """Fails in the world where `forget athens` deleted every episode naming
    it (18 live) — an episode is a record of a whole turn."""
    pm, gm = _owner(tmp_path)
    ep = _ep(tmp_path)
    out = _forget(tmp_path, "Athens", pm, gm, ep)
    assert ep.count() == 2 and "kept" in out
    _forget(tmp_path, "Fotini", pm, gm, ep)
    assert ep.count() == 1
    rec = json.loads((tmp_path / "episodes_forgotten.jsonl").read_text().splitlines()[0])
    assert rec["actions"] and rec["actions"][0]["result"] == "x"      # actions archived too
    with sqlite3.connect(tmp_path / "episodic_memory.db") as c:
        assert c.execute("select count(*) from episode_actions where episode_id = ?", (rec["id"],)).fetchone()[0] == 0


def test_forgotten_episodes_lose_their_vector_twins(tmp_path):
    from ghost_agent.memory.episodes import EpisodicMemory
    ep = EpisodicMemory(tmp_path)
    ep.record_episode("Fotini called", context="", outcome="ok", lesson="")
    vec = MagicMock()
    assert ep.forget_mentions("fotini", vec) == 1
    assert vec.collection.delete.called


def test_episodes_match_any_field_not_only_the_trigger(tmp_path):
    from ghost_agent.memory.episodes import EpisodicMemory
    ep = EpisodicMemory(tmp_path)
    ep.record_episode("a family question", context="", outcome="asked Fotini", lesson="")
    assert ep.count_mentions("fotini") == 1


# ── graph ────────────────────────────────────────────────────────────────────
def test_decay_prunes_chatter_and_keeps_life_facts(tmp_path):
    """Fails in the world where every `user` edge was protected (the graph's
    main noise permanent) or where `.*SON.*` protected PERSON/REASON."""
    gm = GraphMemory(tmp_path)
    gm.add_triplets([{"subject": "user", "predicate": "GREETED", "object": "ghost"},
                     {"subject": "user", "predicate": "HAS_FAVORITE_COLOR", "object": "green"},
                     {"subject": "fotini", "predicate": "MARRIED_TO", "object": "user"},
                     {"subject": "neil", "predicate": "WAS_FIRST_PERSON_ON_MOON", "object": "1969"},
                     {"subject": "task", "predicate": "HAS_NAME_SUGGESTIONS", "object": "x"}])
    with sqlite3.connect(gm.db_path) as c:
        c.execute("UPDATE triplets SET timestamp = datetime('now', '-90 days')")
    gm.prune_stale_edges(max_age_days=45)
    assert _edges(gm) == {("user", "HAS_FAVORITE_COLOR", "green"), ("fotini", "MARRIED_TO", "user")}


def test_delete_edge_archives_and_matches_the_predicate(tmp_path):
    gm = GraphMemory(tmp_path)
    gm.add_triplets([{"subject": "user", "predicate": "HAS_CAR", "object": "volvo"},
                     {"subject": "user", "predicate": "LIKES", "object": "volvo"}])
    assert gm.delete_edge("user", "HAS_CAR", "volvo") == 1
    assert _edges(gm) == {("user", "LIKES", "volvo")}
    rows = [json.loads(l) for l in (tmp_path / gm._ARCHIVE_FILENAME).read_text().splitlines()]
    assert rows[-1]["predicate"] == "HAS_CAR" and rows[-1]["reason"] == "delete_edge"


def test_deleting_a_synonym_field_removes_the_canonical_edge(tmp_path):
    pm = ProfileMemory(tmp_path)
    pm.update("assets", "vehicle", "Volvo")                       # filed as assets.car
    gm = GraphMemory(tmp_path)
    gm.add_triplets([{"subject": "user", "predicate": "HAS_CAR", "object": "volvo"}])
    asyncio.run(M.tool_update_profile(category="assets", key="vehicle", value="", profile_memory=pm, graph_memory=gm))
    assert not _edges(gm)


# ── update_profile and reset_all ─────────────────────────────────────────────
def test_the_model_is_told_what_it_overwrote(tmp_path):
    pm = ProfileMemory(tmp_path)
    pm.update("root", "name", "Vasilis")
    out = asyncio.run(M.tool_update_profile(category="root", key="name", value="Bob", profile_memory=pm))
    assert "(was: 'Vasilis')" in out


@pytest.mark.parametrize("value", ["none", "None", "N/A"])
def test_none_is_a_real_answer(tmp_path, value):
    pm = ProfileMemory(tmp_path)
    assert not pm.update("health", "allergies", value).startswith("Error")


# (§4KX r8) the request-wording gate is gone: reset_all is preview → confirm
# in a later turn — tests/test_forget_confirm_r8.py
def test_the_reset_all_preview_says_what_it_would_erase_and_erases_nothing():
    vec = MagicMock()
    vec.collection.count.return_value = 7
    gm = MagicMock()
    out = asyncio.run(M.tool_knowledge_base(action="reset_all", memory_system=vec, graph_memory=gm))
    assert "vector memory" in str(out) and "7 rows" in str(out) and "episode, lesson" not in str(out)
    vec.collection.delete.assert_not_called()
    gm.wipe_all.assert_not_called()


# ── producers ────────────────────────────────────────────────────────────────
def test_the_final_turn_directive_never_reaches_a_dream_fragment():
    """Fails in the world where only the marker sentence was dropped and the
    rest of the directive became the rule "state the findings and name the
    sources read"."""
    from ghost_agent.core.agent import _FORCED_FINAL_ANSWER_DIRECTIVE as F
    from ghost_agent.core.dream import _strip_harness
    assert _strip_harness("ran the parser. " + F + " The output had 3 rows.") == "ran the parser. The output had 3 rows."
    tail = "mbedded AI opponent when the user said YOU will play) is a violation — fix the artifact NOW. The board rendered."
    assert _strip_harness(tail) == "The board rendered."


@pytest.mark.parametrize("source,written", [("dream", False), ("reflection", True)])
def test_harness_framing_is_refused_from_unattended_producers_only(tmp_path, source, written):
    """A user's own lesson about an input.txt or a validator is legitimate."""
    sm = SkillMemory(tmp_path)
    out = sm.learn_lesson("When the user's script reads input.txt", "Assumed the file exists",
                          "Check that input.txt exists before reading it.", source=source)
    assert (out == "written") is written


def test_a_self_play_framed_rule_is_refused(tmp_path):
    sm = SkillMemory(tmp_path)
    assert sm.learn_lesson("When executing self-play tasks", "none",
                           "When executing self-play tasks, always state the findings.", source="dream") is None


@pytest.mark.parametrize("rule,ctx,share,general", [
    ("When parsing JSON lines, always wrap json.loads in try/except.", "FIRST_ERROR: json.loads raised", 1.0, True),
    ("When a page returns 403, always switch to the browser tool.", "HTTP 403 Forbidden", 1.0, True),
    ("Use a shared threading.Event to signal consumers to stop.", "a producer with threading.Event", 0.5, True),
    ("If Jan is 30 now, two years ago he was 28.", "Jan is 30 years old now", 0.5, False),
    ("Always copy report.txt to backup.txt first.", "copy report.txt to backup.txt", 0.5, False),
    ("When an HTTP request times out, always retry with backoff.", "the HTTP request to the api timed out", 0.5, True),
    ("Leonidas-style age questions: verify the date", "how old is leonidas now ?", 0.5, False),
])
def test_generality_names_files_and_numbers(rule, ctx, share, general):
    from ghost_agent.memory.lesson_scope import is_general_text
    assert is_general_text(rule, ctx, max_shared_share=share) is general


def test_learn_skill_keeps_a_request_shaped_lesson_for_that_request(tmp_path):
    """(§4KX r8) only the model's OWN lesson; a dictated one stays general."""
    sm = SkillMemory(tmp_path)
    req = "restart postgres on the evolmonkey box and check it is up"
    out = sm.learn_lesson("When restarting postgres on the evolmonkey box", "Did not check readiness",
                          "Always run pg_isready after restarting postgres.", source="learn_skill",
                          generality_context=req)
    row = sm._load_playbook()[0]
    assert out == "written" and row["scope"] == "request" and row["source_request"] == req
    assert out.scope == "request"


@pytest.mark.parametrize("req", ["Learn this: when you restart postgres on the evolmonkey box, run pg_isready after",
                                 "remember: restart postgres on the evolmonkey box, then run pg_isready",
                                 "θυμήσου: μετά το restart του postgres στο evolmonkey box τρέξε pg_isready"])
def test_a_lesson_the_user_dictates_is_kept_general(tmp_path, req):
    """Fails in the world where 7 of 10 dictated lessons were silently
    scoped to the one request that taught them."""
    sm = SkillMemory(tmp_path)
    out = sm.learn_lesson("When restarting postgres on the evolmonkey box", "Did not check readiness",
                          "Always run pg_isready after restarting postgres.", source="learn_skill",
                          generality_context=req)
    row = sm._load_playbook()[0]
    assert out == "written" and row.get("scope", "general") != "request" and out.scope == "general"


def _archive(tmp_path, trigger, reason, rotated=False):
    name = "skills_pruned_archive.jsonl" + (".1" if rotated else "")
    with open(tmp_path / name, "a") as fh:
        fh.write(json.dumps({"reason": reason, "lesson": {"trigger": trigger}}) + "\n")


@pytest.mark.parametrize("reason,blocked", [("removed_by_trigger", True), ("retract:abc", False),
                                            ("prune:low_utility", False), ("duplicate_trigger_r5", False)])
def test_only_an_operator_retraction_is_a_tombstone(tmp_path, reason, blocked):
    sm = SkillMemory(tmp_path)
    _archive(tmp_path, "When parsing dates from logs", reason)
    out = sm.learn_lesson("When parsing dates from logs", "guessed", "Use ISO 8601.", source="dream")
    assert (out is None) is blocked


def test_a_rotated_archive_still_tombstones(tmp_path):
    sm = SkillMemory(tmp_path)
    _archive(tmp_path, "When parsing dates from logs", "removed_by_trigger", rotated=True)
    assert sm.learn_lesson("When parsing dates from logs", "guessed", "Use ISO 8601.", source="dream") is None


def test_a_new_tombstone_takes_effect_without_a_restart(tmp_path):
    import os
    import time
    sm = SkillMemory(tmp_path)
    _archive(tmp_path, "unrelated", "removed_by_trigger")
    sm._tombstones()
    time.sleep(0.01)
    _archive(tmp_path, "When parsing dates from logs", "removed_by_trigger")
    os.utime(tmp_path / "skills_pruned_archive.jsonl")
    assert sm.learn_lesson("When parsing dates from logs", "guessed", "Use ISO 8601.", source="dream") is None


def test_a_live_row_is_reinforced_despite_a_tombstone_and_a_plan_is_never_blocked(tmp_path):
    sm = SkillMemory(tmp_path)
    t = "Parallel processing with strict output ordering requirements"
    sm.save_playbook([{"trigger": t, "task": t, "mistake": "m", "solution": "Map results back to input order.",
                       "source": "self_play", "frequency": 1}])
    _archive(tmp_path, t, "removed_by_trigger")
    assert sm.learn_lesson(t, "m", "Map results back to input order.", source="self_play") == "reinforced"
    req = "copy the notes to the backup folder"
    _archive(tmp_path, req, "removed_by_trigger")
    assert sm.learn_lesson(req, "m", "cp notes backup/", trigger=req, scope="request", source_request=req,
                           source="reflection") == "written"


@pytest.mark.parametrize("mistake,none", [
    ("None observed; the solution was direct", True), ("No mistakes were made", True), ("N/A - no mistake", True),
    ("No error handling around json.loads", False), ("None of the paths were quoted", False),
    ("Nothing was flushed", False), ("There was no retry after the 429", False),
])
def test_no_mistake_means_the_whole_statement(mistake, none):
    from ghost_agent.memory.lesson_quality import _is_mistake_less
    assert _is_mistake_less(mistake) is none


def test_the_fix_and_mistake_stay_a_pair_even_when_the_new_mistake_is_none(tmp_path):
    sm = SkillMemory(tmp_path)
    t = "distilled(tool/files): paths"
    sm.learn_lesson(t, "Assuming paths exist", "Check the path exists.", trigger=t, source="distilled")
    sm.learn_lesson(t, "", "Always validate every path against the filesystem before reading it.",
                    trigger=t, source="distilled")
    row = sm._load_playbook()[0]
    assert row["mistake"] == "none" and row["solution"].startswith("Always validate")


def test_the_same_text_ignores_case_and_punctuation():
    from ghost_agent.memory.lesson_quality import is_actionable_lesson
    assert is_actionable_lesson("Use ISO-8601.", "use iso 8601", "Dates") is False


def test_the_dream_cache_stamp_keeps_every_namespace_on_disk(tmp_path):
    from ghost_agent.core.dream import _load_dream_cache, _stamp_dream_cache
    sm = SimpleNamespace(file_path=tmp_path / "skills_playbook.json")
    _stamp_dream_cache(SimpleNamespace(skill_memory=sm), "auto", {"a"})
    _stamp_dream_cache(SimpleNamespace(skill_memory=sm), "traj_selfplay", {"b"})   # a fresh context: a restart
    assert _load_dream_cache(SimpleNamespace(skill_memory=sm)) == {"auto": frozenset({"a"}),
                                                                   "traj_selfplay": frozenset({"b"})}
