"""§4KX r8 (2026-10-03): forget is PREVIEW → CONFIRM, reset_all likewise,
and the matching fixes from the re-review of the §4KX deploy. Each test names
the world it fails in."""
import asyncio
import json
import sqlite3
from unittest.mock import MagicMock

import pytest

from ghost_agent.memory.episodes import EpisodicMemory
from ghost_agent.memory.graph import GraphMemory
from ghost_agent.memory.profile import ProfileMemory, mentions
from ghost_agent.memory.skills import SkillMemory
from ghost_agent.tools import memory as M
from ghost_agent.utils.logging import request_id_context
from tests.test_memory_producers_rereview import _Vec, _edges, _forget, _owner


def _kb(pm, gm, ep=None, sandbox=None, vec=None, **kw):
    return asyncio.run(M.tool_knowledge_base(memory_system=vec or _Vec(), sandbox_dir=sandbox, profile_memory=pm,
                                             graph_memory=gm, episodic_memory=ep, **kw))


def _token(out) -> str:
    return str(out).split("confirm='")[1].split("'")[0]


def _as_request(rid):
    """Run the next calls as request ``rid`` (another turn)."""
    return request_id_context.set(rid)


@pytest.fixture(autouse=True)
def _turn_one():
    tok = request_id_context.set("req-preview")
    yield
    request_id_context.reset(tok)


def _family(tmp_path):
    pm = ProfileMemory(tmp_path)
    pm.update("root", "name", "Vasilis")
    pm.update("root", "company", "EvolMonkey")
    pm.update("relationships", "wife_name", "Fotini")
    pm.update("relationships", "fotini_birthday", "3 March")
    pm.update("relationships", "fotini_description", "kind, loves the sea")
    gm = GraphMemory(tmp_path)
    gm.add_triplets([{"subject": "user", "predicate": "MARRIED_TO", "object": "fotini"},
                     {"subject": "fotini", "predicate": "HAS_BIRTHDAY", "object": "3 march"},
                     {"subject": "leonidas", "predicate": "IS_SON_OF", "object": "fotini"},
                     {"subject": "user", "predicate": "WORKS_AT", "object": "evolmonkey"},
                     {"subject": "ektoras", "predicate": "WORKS_AT", "object": "acme"},
                     {"subject": "user", "predicate": "OWNS", "object": "pista gp rr"},
                     {"subject": "project", "predicate": "USES", "object": "node.js"}])
    return pm, gm


# ── preview → confirm ────────────────────────────────────────────────────────
def test_the_preview_deletes_nothing_and_numbers_what_it_would(tmp_path):
    """Fails in the world where forget deleted in the call that asked."""
    pm, gm = _family(tmp_path)
    sb = tmp_path / "sb"
    sb.mkdir()
    (sb / "fotini.txt").write_text("x")
    ep = EpisodicMemory(tmp_path)
    ep.record_episode("what did Fotini say", context="", outcome="ok", lesson="")
    before_e, before_p = _edges(gm), pm.load()
    out = _kb(pm, gm, ep, sb, action="forget", target="Fotini")
    assert str(out).startswith("PREVIEW") and "1. " in out and "confirm='" in out
    assert _edges(gm) == before_e and pm.load() == before_p and (sb / "fotini.txt").exists()
    assert ep.count_mentions("fotini") == 1


def test_the_turn_that_previewed_cannot_confirm(tmp_path):
    """Fails in the world where the model previews and confirms in one breath."""
    pm, gm = _family(tmp_path)
    tok = _token(_kb(pm, gm, action="forget", target="ektoras"))
    out = _kb(pm, gm, action="forget", confirm=tok)
    assert "NOT deleted" in str(out) and ("ektoras", "WORKS_AT", "acme") in _edges(gm)


def test_the_users_next_turn_deletes_exactly_the_default_items(tmp_path):
    pm, gm = _family(tmp_path)
    tok = _token(_kb(pm, gm, action="forget", target="ektoras"))
    t = _as_request("req-next")
    try:
        out = _kb(pm, gm, action="forget", confirm=tok, items="all")
    finally:
        request_id_context.reset(t)
    e = _edges(gm)
    assert ("ektoras", "WORKS_AT", "acme") not in e and "✅" in out
    assert ("user", "WORKS_AT", "evolmonkey") in e and len(e) == 6


def test_a_listed_owner_fact_is_deleted_only_when_picked_by_number(tmp_path):
    """Fails in the world where some owner facts could be removed by NO tool
    (a sold car's `user OWNS`)."""
    pm, gm = _family(tmp_path)
    out = _kb(pm, gm, action="forget", target="pista gp rr")
    assert "(nothing by default)" in out and "1. graph user OWNS pista gp rr" in out
    tok = _token(out)
    t = _as_request("req-next")
    try:
        assert "NOT deleted" in str(_kb(pm, gm, action="forget", confirm=tok, items="all"))
        _kb(pm, gm, action="forget", confirm=tok, items="1")
    finally:
        request_id_context.reset(t)
    assert ("user", "OWNS", "pista gp rr") not in _edges(gm)


def test_a_token_works_once_and_an_unknown_one_is_refused(tmp_path):
    pm, gm = _family(tmp_path)
    tok = _token(_kb(pm, gm, action="forget", target="ektoras"))
    t = _as_request("req-next")
    try:
        _kb(pm, gm, action="forget", confirm=tok)
        again = _kb(pm, gm, action="forget", confirm=tok)
        bogus = _kb(pm, gm, action="forget", confirm="deadbeef")
    finally:
        request_id_context.reset(t)
    assert "unknown or expired" in str(again) and "unknown or expired" in str(bogus)


@pytest.mark.parametrize("rid", ["job-1234", "sched-9", "sub-abc"])
def test_an_autonomous_turn_cannot_confirm(tmp_path, rid):
    """Fails in the world where a background job said yes for the user."""
    pm, gm = _family(tmp_path)
    tok = _token(_kb(pm, gm, action="forget", target="ektoras"))
    t = _as_request(rid)
    try:
        out = _kb(pm, gm, action="forget", confirm=tok)
    finally:
        request_id_context.reset(t)
    assert "NOT deleted" in str(out) and ("ektoras", "WORKS_AT", "acme") in _edges(gm)


def test_the_file_executor_rechecks_the_path_at_delete_time(tmp_path):
    """Fails in the world where a file swapped for a link between preview and
    confirm was deleted THROUGH the link."""
    sb = tmp_path / "sb"
    sb.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("keep me")
    (sb / "notes.txt").write_text("x")
    pm, gm = _family(tmp_path)
    tok = _token(_kb(pm, gm, sandbox=sb, action="forget", target="notes.txt"))
    (sb / "notes.txt").unlink()
    (sb / "notes.txt").symlink_to(outside)
    t = _as_request("req-next")
    try:
        out = _kb(pm, gm, sandbox=sb, action="forget", confirm=tok)
    finally:
        request_id_context.reset(t)
    assert "Refused" in out and outside.read_text() == "keep me"


def test_the_file_executor_honours_the_released_lock(tmp_path, monkeypatch):
    sb = tmp_path / "sb"
    sb.mkdir()
    (sb / "notes.txt").write_text("x")
    pm, gm = _family(tmp_path)
    tok = _token(_kb(pm, gm, sandbox=sb, action="forget", target="notes.txt"))
    import ghost_agent.tools.file_system as fs
    monkeypatch.setattr(fs, "_released_write_block", lambda *a, **k: "released")
    t = _as_request("req-next")
    try:
        out = _kb(pm, gm, sandbox=sb, action="forget", confirm=tok)
    finally:
        request_id_context.reset(t)
    assert "RELEASED" in out and (sb / "notes.txt").exists()


# ── reset_all ────────────────────────────────────────────────────────────────
def _wipe_store():
    vec = MagicMock()
    vec.collection.count.return_value = 3
    vec.collection.get.return_value = {"ids": ["a", "b", "c"], "metadatas": [{}, {}, {}]}
    vec.library_file = None
    return vec


def test_reset_all_is_two_steps(tmp_path):
    """Fails in the world where one reset_all call wiped every store, or the
    old wording gate opened on "Don't wipe all your memory!"."""
    vec, gm = _wipe_store(), MagicMock()
    out = _kb(None, gm, vec=vec, action="reset_all")
    assert str(out).startswith("PREVIEW") and not vec.collection.delete.called and not gm.wipe_all.called
    tok = _token(out)
    assert "NOT executed" in str(_kb(None, gm, vec=vec, action="reset_all", confirm=tok))
    t = _as_request("req-next")
    try:
        done = _kb(None, gm, vec=vec, action="reset_all", confirm=tok)
    finally:
        request_id_context.reset(t)
    assert "Wiped" in str(done) and vec.collection.delete.called and gm.wipe_all.called


def test_a_forget_token_cannot_confirm_a_wipe(tmp_path):
    pm, gm = _family(tmp_path)
    tok = _token(_kb(pm, gm, action="forget", target="ektoras"))
    vec = _wipe_store()
    t = _as_request("req-next")
    try:
        out = _kb(pm, gm, vec=vec, action="reset_all", confirm=tok)
    finally:
        request_id_context.reset(t)
    assert "NOT executed" in str(out) and not vec.collection.delete.called


# ── matching ─────────────────────────────────────────────────────────────────
def test_fotinis_birthday_takes_only_the_birthday(tmp_path):
    """Fails in the world where "forget Fotini's birthday" deleted the
    marriage, the son's edge and the wife-name field."""
    pm, gm = _family(tmp_path)
    _forget(tmp_path, "Fotini's birthday", pm, gm)
    e, rel = _edges(gm), pm.load()["relationships"]
    assert ("fotini", "HAS_BIRTHDAY", "3 march") not in e and "fotini_birthday" not in rel
    assert ("user", "MARRIED_TO", "fotini") in e and ("leonidas", "IS_SON_OF", "fotini") in e
    assert rel["wife_name"] == "Fotini" and "fotini_description" in rel


def test_a_category_before_the_name_is_not_a_qualifier(tmp_path):
    assert M._qualifiers_of("my old car Tesla") == [] and M._qualifiers_of("the birthday of Fotini") == ["birthday"]


def test_a_greek_name_is_found_in_every_store(tmp_path):
    """Fails in the world where Greek names were found in no store (ASCII
    split, SQLite's ASCII-only LIKE, no accent folding)."""
    pm = ProfileMemory(tmp_path)
    pm.update("relationships", "wife_name", "Φωτεινή")
    gm = GraphMemory(tmp_path)
    gm.add_triplets([{"subject": "user", "predicate": "MARRIED_TO", "object": "Φωτεινή"},
                     {"subject": "Φωτεινή", "predicate": "LIKES", "object": "sea"}])
    ep = EpisodicMemory(tmp_path)
    ep.record_episode("τι είπε η ΦΩΤΕΙΝΗ;", context="", outcome="ok", lesson="")
    out = _kb(pm, gm, ep, action="forget", target="φωτεινη")
    assert "wife_name" in out and "MARRIED_TO" in out and "LIKES" in out and "episode #" in out


@pytest.mark.parametrize("value,target,hit", [("Φωτεινή", "φωτεινη", True), ("Ελένη και Νίκος", "ελενη", True),
                                              ("language", "age", False), ("node.js 20", "node.js", True),
                                              ("my tesla model 3", "tesla model", True),
                                              ("teslamodel", "tesla model", False),
                                              ("tesla modelling", "tesla model", False),
                                              ("Φωτεινη", "Φωτεινή", True)])
def test_the_shared_mention_rule(value, target, hit):
    assert mentions(value, target) is hit


@pytest.mark.parametrize("target,edge", [("node.js", ("project", "USES", "node.js")),
                                         ("pista-gp rr", ("user", "OWNS", "pista gp rr"))])
def test_dotted_and_hyphenated_names_reach_the_graph(tmp_path, target, edge):
    """Fails in the world where "node.js" became one U+2024 token and
    "pista-gp" two words, so neither ever reached the graph."""
    pm, gm = _family(tmp_path)
    out = _kb(pm, gm, action="forget", target=target)
    assert "graph " + " ".join(edge) in out


def test_another_persons_durable_fact_is_theirs_not_yours(tmp_path):
    """Fails in the world where a public person's WORKS_AT could not be
    forgotten and was reported as a fact about the owner."""
    pm, gm = _family(tmp_path)
    out = _forget(tmp_path, "ektoras", pm, gm)
    assert ("ektoras", "WORKS_AT", "acme") not in _edges(gm) and "facts about you" not in out


def test_forgetting_a_family_person_takes_the_fields_named_after_them(tmp_path):
    """Fails in the world where `fotini_description` outlived the graph."""
    pm, gm = _family(tmp_path)
    _forget(tmp_path, "Fotini", pm, gm)
    rel = pm.load().get("relationships", {})
    assert "fotini_description" not in rel and "fotini_birthday" not in rel


def test_an_owner_field_and_its_graph_fact_agree(tmp_path):
    """Fails in the world where `forget EvolMonkey` deleted root.company while
    the graph kept `user WORKS_AT evolmonkey`."""
    pm, gm = _family(tmp_path)
    _forget(tmp_path, "EvolMonkey", pm, gm)
    assert pm.load()["root"]["company"] == "EvolMonkey" and ("user", "WORKS_AT", "evolmonkey") in _edges(gm)


def test_the_episode_leg_searches_the_entity_not_the_phrase(tmp_path):
    """Fails in the world where "my wife Fotini" searched episodes for the
    whole phrase while the graph leg searched "fotini"."""
    pm, gm = _family(tmp_path)
    ep = EpisodicMemory(tmp_path)
    ep.record_episode("what did Fotini say about the trip", context="", outcome="ok", lesson="")
    _forget(tmp_path, "my wife Fotini", pm, gm, ep)
    assert ep.count_mentions("fotini") == 0


def test_a_degraded_profile_says_it_was_not_searched(tmp_path):
    pm, gm = _family(tmp_path)
    pm._degraded = True
    pm.load = lambda: {}
    out = _forget(tmp_path, "ektoras", pm, gm)
    assert "could not be read" in out


# ── expansion guards, behind ALIAS edges ─────────────────────────────────────
@pytest.mark.parametrize("alias,extra", [
    ("ghost", []),                                                     # stoplisted hub
    ("bigdog", [{"subject": "bigdog", "predicate": f"R{i}", "object": f"n{i}"} for i in range(9)]),   # degree
    ("mortimers pal", []),                                             # contains the target
])
def test_the_expansion_never_follows_an_alias_into_a_hub(tmp_path, alias, extra):
    """Fails in the world where the hub stoplist, the degree cap or the
    substring guard of the alias expansion was gone."""
    pm = ProfileMemory(tmp_path)
    gm = GraphMemory(tmp_path)
    gm.add_triplets([{"subject": "mortimer", "predicate": "ALSO_KNOWN_AS", "object": alias},
                     {"subject": alias, "predicate": "RUNS_ON", "object": "mac studio"}] + extra)
    _forget(tmp_path, "mortimer", pm, gm)
    assert (alias, "RUNS_ON", "mac studio") in _edges(gm)


def test_a_true_alias_is_followed(tmp_path):
    pm = ProfileMemory(tmp_path)
    gm = GraphMemory(tmp_path)
    gm.add_triplets([{"subject": "mortimer", "predicate": "ALSO_KNOWN_AS", "object": "morty"},
                     {"subject": "morty", "predicate": "EATS", "object": "lettuce"}])
    _forget(tmp_path, "mortimer", pm, gm)
    assert ("morty", "EATS", "lettuce") not in _edges(gm)


# ── graph primitives ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("node,entity,hit", [("tesla model 3", "tesla", True), ("teslamotors", "tesla", False),
                                             ("athens, greece", "athens", False),
                                             ("postgresql services company", "postgresql", False),
                                             ("Φωτεινή", "φωτεινη", True), ("pista-gp rr", "pista gp rr", True)])
def test_a_node_is_the_entity_or_its_version(node, entity, hit):
    assert GraphMemory._node_is(node, entity) is hit


def test_forget_entity_archives_before_it_deletes(tmp_path):
    gm = GraphMemory(tmp_path)
    gm.add_triplets([{"subject": "ektoras", "predicate": "WORKS_AT", "object": "acme"}])
    gm.forget_entity("ektoras")
    rows = [json.loads(x) for x in open(tmp_path / "graph_pruned_archive.jsonl")]
    assert rows[-1]["reason"] == "forget_entity" and rows[-1]["subject"] == "ektoras"


def test_delete_edge_touches_only_a_live_row(tmp_path):
    """Fails in the world where the executor deleted an EXPIRED (history) row
    it never listed."""
    gm = GraphMemory(tmp_path)
    gm.add_triplets([{"subject": "user", "predicate": "OWNS", "object": "volvo"}])
    with sqlite3.connect(gm.db_path) as c:
        c.execute("INSERT INTO triplets (subject, predicate, object, valid_until) VALUES ('user','OWNS','bmw', 1)")
    assert gm.delete_edge("user", "OWNS", "bmw") == 0 and gm.delete_edge("user", "OWNS", "volvo") == 1
    with sqlite3.connect(gm.db_path) as c:
        assert c.execute("SELECT object FROM triplets").fetchall() == [("bmw",)]


# ── decay ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("edge,kept", [(("user", "HAS_CONDITION", "heart failure"), True),
                                       (("user", "HAS_MEDICATION", "entresto"), True),
                                       (("user", "HAS_PROFESSION", "doctor"), True),
                                       (("user", "LIVES_IN", "athens, greece"), True),
                                       (("user", "LOCATED_IN", "kyllini"), False),   # §4KZ: presence decays
                                       (("subscription", "LOCATED_IN", "different database"), False),
                                       (("user", "IS_AT", "kyllini"), False),
                                       (("user", "HAS_TEST_COLOUR", "teal"), False),
                                       (("user", "ALLERGIC_TO", "penicillin"), True),
                                       (("fotini", "HAS_PROFESSION", "teacher"), True),
                                       (("user", "HAS_COMPANION", "fotini"), True)])
def test_decay_keeps_health_and_place_facts_and_drops_probe_noise(tmp_path, edge, kept):
    """Fails in the world where decay pruned the owner's heart condition and
    medication, or kept a probe's HAS_TEST_COLOUR forever."""
    gm = GraphMemory(tmp_path)
    gm.add_triplets([dict(zip(("subject", "predicate", "object"), edge))])
    with sqlite3.connect(gm.db_path) as c:
        c.execute("UPDATE triplets SET timestamp = '2020-01-01 00:00:00'")
    gm.prune_stale_edges(max_age_days=30)
    assert (edge in _edges(gm)) is kept


# ── lesson producers ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("rule,ctx", [
    ("Check the GitHub Actions cache before rebuilding", "my GitHub build is slow"),
    ("Pin the Node.js version in CI", "set up CI for my Node.js app"),
    ("Use PostgreSQL's EXPLAIN ANALYZE before adding an index", "why is my PostgreSQL query slow"),
])
def test_a_product_name_is_not_a_requests_name(rule, ctx):
    """Fails in the world where GitHub/Node.js/PostgreSQL made a dream
    window's rule "specific"."""
    from ghost_agent.memory.lesson_scope import is_general_text
    assert is_general_text(rule, ctx, max_shared_share=1.0)


def test_a_dotted_file_is_still_the_requests():
    from ghost_agent.memory.lesson_scope import is_general_text
    assert not is_general_text("Read app.js before editing", "fix app.js", max_shared_share=1.0)


@pytest.mark.parametrize("word", ["Determining", "Running", "Stopped", "Matches", "Used"])
def test_inflected_english_is_not_a_name(word):
    from ghost_agent.memory import lesson_scope as ls
    ls._is_english_word("a")
    _is_english_word = ls._is_english_word
    if not ls._ENGLISH:
        pytest.skip("no system word list")
    assert _is_english_word(word)


@pytest.mark.parametrize("text,harness", [
    ("Add a pre-execution validator to the CLI", False), ("pydantic validators run before save", False),
    ("Write a validator function for emails", False), ("Satisfy the validator by echoing input.txt", True),
    ("the hidden-tests check stdout", True), ("Graders want the exact token", True),
    ("Avoid coded stand-ins for the opponent", True),
])
def test_the_harness_framing_rule(text, harness):
    from ghost_agent.memory.skills import _VALIDATOR_FRAMING_RE
    assert bool(_VALIDATOR_FRAMING_RE.search(text)) is harness


@pytest.mark.parametrize("mistake,none", [
    ("None observed in the final successful execution path.", True), ("No issues found", True),
    ("No errors, but the loop ran twice", False), ("None. The first try failed though", False),
    ("None; the first attempt crashed", False),
])
def test_no_mistake_unless_a_failure_is_named(mistake, none):
    from ghost_agent.memory.lesson_quality import _is_mistake_less
    assert _is_mistake_less(mistake) is none


def test_a_learn_skill_lesson_is_never_absorbed_by_another_producers_row(tmp_path):
    """Fails in the world where a dictated rule vanished into a dream row at
    distance 0.05."""
    sm = SkillMemory(tmp_path)
    a = "When a web page returns 403, try an archived copy"
    sm.save_playbook([{"trigger": a, "task": a, "mistake": "m", "solution": "Use archive.org", "source": "dream"}])
    sm._find_duplicate_lesson = lambda *x, **k: {"source": "vector", "trigger": a, "text": "", "distance": 0.05}
    sm.learn_lesson("If a site answers 403, use a cached copy", "m", "Use a cached copy.", MagicMock(),
                    source="learn_skill", generality_context="remember: on a 403 use a cached copy")
    assert len(sm._load_playbook()) == 2


def test_the_learn_skill_tool_says_when_a_lesson_is_for_this_request_only(tmp_path):
    from ghost_agent.memory.lesson_scope import current_request
    sm = SkillMemory(tmp_path)
    tok = current_request.set("restart postgres on the evolmonkey box and check it is up")
    try:
        out = asyncio.run(M.tool_learn_skill("When restarting postgres on the evolmonkey box", "Did not check",
                                             "Always run pg_isready after restarting postgres.", skill_memory=sm))
    finally:
        current_request.reset(tok)
    assert "THIS request only" in out


def test_a_short_harness_scrap_is_stripped():
    from ghost_agent.core.dream import _harness_texts, _strip_harness
    texts = _harness_texts()
    scrap = next((t[:9] for t in texts if len(t) >= 9), None)
    if scrap is None:
        pytest.skip("no harness text")
    assert scrap.strip() not in _strip_harness(f"Keep this rule. {scrap}").lower()


class _FactStore:
    """A vector store holding one conversational fact that names Ektoras."""
    def __init__(self):
        self.deleted, self.docs_deleted = [], []
        store = self

        class _C:
            def query(self, **k):
                return {"ids": [["f1"]], "documents": [["Ektoras works at Acme"]], "metadatas": [[{"type": "auto"}]],
                        "distances": [[0.9]]}

            def delete(self, ids=None, **k):
                store.deleted += list(ids or [])

            def count(self):
                return 1
        self.collection = _C()

    def get_library(self):
        return ["ektoras.pdf"]

    def delete_document_by_name(self, n):
        self.docs_deleted.append(n)


def test_the_preview_never_touches_the_vector_store(tmp_path):
    """Fails in the world where the preview deleted vector facts and
    documents while saying "nothing was deleted"."""
    pm, gm = _family(tmp_path)
    vec = _FactStore()
    out = _kb(pm, gm, vec=vec, action="forget", target="ektoras")
    assert vec.deleted == [] and vec.docs_deleted == []
    assert "fact 'Ektoras works at Acme'" in out and "document 'ektoras.pdf'" in out
    t = _as_request("req-next")
    try:
        _kb(pm, gm, vec=vec, action="forget", confirm=_token(out))
    finally:
        request_id_context.reset(t)
    assert vec.deleted == ["f1"] and vec.docs_deleted == ["ektoras.pdf"]


def test_a_qualified_forget_keeps_the_family_members_episodes(tmp_path):
    """Fails in the world where "Fotini's birthday" deleted every episode
    naming her."""
    pm, gm = _family(tmp_path)
    ep = EpisodicMemory(tmp_path)
    ep.record_episode("what did Fotini say about the trip", context="", outcome="ok", lesson="")
    _forget(tmp_path, "Fotini's birthday", pm, gm, ep)
    assert ep.count_mentions("fotini") == 1


def test_a_qualifier_matches_its_stem(tmp_path):
    """birthdate ~ HAS_BIRTHDAY: fails where only the exact word matched."""
    pm, gm = _family(tmp_path)
    _forget(tmp_path, "Fotini birthdate", pm, gm)
    assert ("fotini", "HAS_BIRTHDAY", "3 march") not in _edges(gm) and ("user", "MARRIED_TO", "fotini") in _edges(gm)


def test_a_greek_family_member_is_family(tmp_path):
    """Fails where the family check compared unfolded names, so a Greek
    wife's episodes were only listed, never in the default list."""
    gm = GraphMemory(tmp_path)
    gm.add_triplets([{"subject": "user", "predicate": "MARRIED_TO", "object": "Φωτεινή"}])
    assert gm.is_owner_family("φωτεινη") and gm.is_owner_family("ΦΩΤΕΙΝΗ")


def test_no_episode_is_deleted_when_the_archive_cannot_be_written(tmp_path):
    ep = EpisodicMemory(tmp_path)
    ep.record_episode("what did Fotini say", context="", outcome="ok", lesson="")
    (tmp_path / "episodes_forgotten.jsonl").mkdir()
    assert ep.delete_episodes([i for i, _ in ep.mention_previews("fotini")]) == 0
    assert ep.count_mentions("fotini") == 1


def test_a_later_default_wins_over_an_earlier_listing():
    """Fails where an item first LISTED by one leg stayed out of "all" after
    another leg chose to delete it."""
    pl = M._Plan()
    pl.add("profile_field", {"category": "a", "key": "b"}, "x", default=False)
    pl.add("profile_field", {"category": "a", "key": "b"}, "x")
    assert M._pick(pl.items, "all")[0] == pl.items



# ── r8 fresh-eye review ──────────────────────────────────────────────────────
def test_the_preview_keeps_the_sweeps_warnings_and_partial_names(tmp_path):
    """Fails where the preview dropped ambiguous/partial file names and said
    "Nothing stored matches"."""
    sb = tmp_path / "sb"
    (sb / "a").mkdir(parents=True)
    (sb / "b").mkdir()
    (sb / "a" / "index.html").write_text("x")
    (sb / "b" / "index.html").write_text("x")
    (sb / "atlas_notes.md").write_text("x")
    pm, gm = _family(tmp_path)
    out = _kb(pm, gm, sandbox=sb, action="forget", target="index.html")
    assert "Nothing stored matches" not in out and "file a/index.html" in out and "(nothing by default)" in out
    class _Lib(_Vec):
        def get_library(self):
            return ["atlas_manual.pdf"]
    out2 = _kb(pm, gm, sandbox=sb, vec=_Lib(), action="forget", target="atlas")
    assert "file atlas_notes.md" in out2 and "document 'atlas_manual.pdf'" in out2.split("Also found")[1]


def test_the_preview_reports_a_store_error(tmp_path):
    pm, gm = _family(tmp_path)

    class _Broken(_Vec):
        def get_library(self):
            raise RuntimeError("chroma down")
    out = _kb(pm, gm, vec=_Broken(), action="forget", target="ektoras")
    assert "chroma down" in out


@pytest.mark.parametrize("old,typed", [("José Silva", "jose"), ("Θεσσαλονίκη", "θεσσαλονικη")])
def test_the_preview_finds_an_accented_previous_value(tmp_path, old, typed):
    pm = ProfileMemory(tmp_path)
    pm.update("root", "name", old)
    pm.update("root", "name", "Vasilis")
    out = _kb(pm, GraphMemory(tmp_path), action="forget", target=typed)
    assert "previous value" in out


def test_the_executor_says_when_a_field_was_already_gone(tmp_path):
    pm, gm = _family(tmp_path)
    tok = _token(_kb(pm, gm, action="forget", target="relationships.fotini_description"))
    pm.delete("relationships", "fotini_description")
    t = _as_request("req-next")
    try:
        out = _kb(pm, gm, action="forget", confirm=tok)
    finally:
        request_id_context.reset(t)
    assert "already gone" in out and "✅" not in out


def test_the_executor_deletes_the_field_it_showed(tmp_path):
    """Fails where a legacy `root.car` was shown and `assets.car` deleted
    (the canonical map)."""
    pm = ProfileMemory(tmp_path)
    pm.update("assets", "car", "Tesla Model 3")
    raw = pm.load_raw()
    raw.setdefault("root", {})["car"] = "BMW 118i"
    pm.save(raw)
    pl = M._Plan()
    pl.add("profile_field", {"category": "root", "key": "car"}, "profile root.car = 'BMW 118i'")
    M._execute_item(pl.items[0], None, pm, None, None, None)
    d = pm.load()
    assert "car" not in d.get("root", {}) and d["assets"]["car"] == "Tesla Model 3"


def test_an_expired_token_is_refused_at_confirm(tmp_path):
    pm, gm = _family(tmp_path)
    tok = _token(_kb(pm, gm, action="forget", target="ektoras"))
    M._FORGET_PLANS[tok]["ts"] -= 2 * M._PLAN_TTL_S
    t = _as_request("req-next")
    try:
        out = _kb(pm, gm, action="forget", confirm=tok)
    finally:
        request_id_context.reset(t)
    assert "expired" in str(out) and ("ektoras", "WORKS_AT", "acme") in _edges(gm)


@pytest.mark.parametrize("rid", ["SYSTEM", "bench-1a2b3c", "replay-1-a"])
def test_no_request_and_bench_or_replay_traffic_cannot_confirm(tmp_path, rid):
    pm, gm = _family(tmp_path)
    tok = _token(_kb(pm, gm, action="forget", target="ektoras"))
    t = _as_request(rid)
    try:
        out = _kb(pm, gm, action="forget", confirm=tok)
    finally:
        request_id_context.reset(t)
    assert "NOT deleted" in str(out)


def test_a_preview_the_user_never_saw_cannot_be_confirmed(tmp_path):
    pm, gm = _family(tmp_path)
    t = _as_request("job-77")
    try:
        tok = _token(_kb(pm, gm, action="forget", target="ektoras"))
    finally:
        request_id_context.reset(t)
    t = _as_request("req-next")
    try:
        out = _kb(pm, gm, action="forget", confirm=tok)
    finally:
        request_id_context.reset(t)
    assert "NOT deleted" in str(out) and ("ektoras", "WORKS_AT", "acme") in _edges(gm)


@pytest.mark.parametrize("items,ok", [([1], True), (1, True), ("1,7", False), ("x", False)])
def test_a_selection_is_parsed_or_refused_whole(tmp_path, items, ok):
    pm, gm = _family(tmp_path)
    tok = _token(_kb(pm, gm, action="forget", target="ektoras"))
    t = _as_request("req-next")
    try:
        out = _kb(pm, gm, action="forget", confirm=tok, items=items)
    finally:
        request_id_context.reset(t)
    assert (("ektoras", "WORKS_AT", "acme") not in _edges(gm)) is ok
    if not ok:
        assert "not on the list" in str(out)



# ── r8 fresh-eye review: matching ────────────────────────────────────────────
def test_a_qualified_target_never_defaults_an_owner_fact(tmp_path):
    """Fails where "EvolMonkey work" put `user WORKS_AT evolmonkey` in the
    default list."""
    pm, gm = _family(tmp_path)
    out = _kb(pm, gm, action="forget", target="EvolMonkey work")
    assert "(nothing by default)" in out and "graph user WORKS_AT evolmonkey" in out


def test_company_is_not_companion(tmp_path):
    pm, gm = _family(tmp_path)
    gm.add_triplets([{"subject": "user", "predicate": "HAS_COMPANION", "object": "fotini"}])
    _forget(tmp_path, "Fotini's company", pm, gm)
    assert ("user", "HAS_COMPANION", "fotini") in _edges(gm)


@pytest.mark.parametrize("target", ["Leonidas birthdate", "the birthday of Leonidas"])
def test_a_birthday_reaches_born_edges(tmp_path, target):
    pm, gm = _family(tmp_path)
    gm.add_triplets([{"subject": "leonidas", "predicate": "BORN_ON", "object": "2010-05-01"}])
    _forget(tmp_path, target, pm, gm)
    assert ("leonidas", "BORN_ON", "2010-05-01") not in _edges(gm) and ("leonidas", "IS_SON_OF", "fotini") in _edges(gm)


def test_a_family_forget_takes_the_owner_field_edges_named_after_her(tmp_path):
    """Fails where `user HAS_FOTINI_DESCRIPTION …` outlived the forget and was
    not even listed."""
    pm, gm = _family(tmp_path)
    gm.add_triplets([{"subject": "user", "predicate": "HAS_FOTINI_DESCRIPTION", "object": "kind, loves the sea"}])
    _forget(tmp_path, "my wife Fotini", pm, gm)
    assert not any(e[1] == "HAS_FOTINI_DESCRIPTION" for e in _edges(gm))
    assert "fotini_description" not in pm.load().get("relationships", {})


def test_someone_elses_owner_field_edge_is_only_listed(tmp_path):
    pm, gm = _family(tmp_path)
    gm.add_triplets([{"subject": "user", "predicate": "HAS_EKTORAS_OPINION", "object": "smart"}])
    out = _kb(pm, gm, action="forget", target="ektoras")
    assert "graph user HAS_EKTORAS_OPINION smart (a fact about you)" in out.split("Also found")[1]


@pytest.mark.parametrize("stored,typed", [("rené lacoste", "rene lacoste"), ("panerithraikos g.o.", "panerithraikos g.o."),
                                          ("vale of tempe", "the vale of tempe")])
def test_names_with_accents_dots_and_inner_words_are_found(tmp_path, stored, typed):
    pm = ProfileMemory(tmp_path)
    gm = GraphMemory(tmp_path)
    gm.add_triplets([{"subject": stored, "predicate": "IS_IN", "object": "somewhere"}])
    out = _kb(pm, gm, action="forget", target=typed)
    assert "IS_IN somewhere" in out


def test_an_accented_owner_name_is_the_owner(tmp_path):
    pm = ProfileMemory(tmp_path)
    pm.update("root", "name", "Βασίλης")
    assert M._is_owner_name(pm, "βασιλης")


def test_a_blind_profile_holds_the_graph_back(tmp_path):
    """Fails where an unreadable profile hid the owner's name and the graph
    leg defaulted the owner's own edges."""
    pm, gm = _family(tmp_path)
    gm.add_triplets([{"subject": "vasilis", "predicate": "HAS_BIRTHDATE", "object": "1980"}])
    pm._degraded = True
    pm.load = lambda: {}
    _forget(tmp_path, "Vasilis", pm, gm)
    assert ("vasilis", "HAS_BIRTHDATE", "1980") in _edges(gm)


def test_a_dotted_list_item_is_pruned(tmp_path):
    pm = ProfileMemory(tmp_path)
    pm.update("interests", "stack", ["node.js", "postgres"])
    _forget(tmp_path, "node.js", pm, GraphMemory(tmp_path))
    v = pm.load()["interests"]["stack"]
    assert "node.js" not in (v if isinstance(v, list) else [v])


# ── r8 fresh-eye review: lesson producers ────────────────────────────────────
@pytest.mark.parametrize("req,dictated", [
    ("the game never starts, the ball never gets in the pinball", False),
    ("what do you remember about leonidas?", False), ("remember who she is ?", False),
    ("για ποιο λόγο το αποτέλεσμα είναι λάθος", False), ("Τι έμαθες σήμερα;", False), ("απάντα μου σύντομα", False),
    ("I always wanted to get better at chess", False), ("the never-ending loop again", False),
    ("why does it always use the old config?", False), ("it never run the tests", False),
    ("Remember this: always run pg_isready after a restart", True), ("from now on, use metric units", True),
    ("Always use UTC in logs.", True), ("θυμήσου: τρέχε pg_isready μετά το restart", True),
    ("note that the API key lives in ~/.ghost_api_key", True),
])
def test_dictation_is_an_imperative_not_a_word(req, dictated):
    from ghost_agent.memory.skills import user_dictates_lesson
    assert user_dictates_lesson(req) is dictated


def test_dream_rewordings_of_a_learn_skill_rule_merge_into_it(tmp_path):
    """Fails where every rewording of a learn_skill rule became its own row."""
    sm = SkillMemory(tmp_path)
    a = "When a web page returns 403, try an archived copy"
    sm.save_playbook([{"trigger": a, "task": a, "mistake": "m", "solution": "Use archive.org", "source": "learn_skill"}])
    sm._find_duplicate_lesson = lambda *x, **k: {"source": "vector", "trigger": a, "text": "", "distance": 0.03}
    for t in ("If a site answers 403, use a cached copy", "On HTTP 403 fetch an archived snapshot"):
        sm.learn_lesson(t, "m", "Use a cached copy.", MagicMock(), source="dream")
    assert len(sm._load_playbook()) == 1


def test_concurrent_learn_skill_calls_report_their_own_scope(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    sm = SkillMemory(tmp_path)
    req_specific = ("When restarting postgres on the evolmonkey box", "Did not check",
                    "Always run pg_isready after restarting postgres.",
                    "restart postgres on the evolmonkey box and check it is up")
    general = ("When a shell command times out", "Waited forever", "Always set a timeout on shell commands.",
               "why do my scripts hang sometimes")

    def go(t):
        return sm.learn_lesson(t[0], t[1], t[2], source="learn_skill", generality_context=t[3])
    with ThreadPoolExecutor(2) as ex:
        a, b = ex.map(go, [req_specific, general])
    assert a.scope == "request" and b.scope == "general"


def test_a_dropped_legacy_learn_skill_is_not_reported_as_saved(tmp_path):
    sm = SkillMemory(tmp_path)
    out = asyncio.run(M.tool_learn_skill("x", "none", "ok", skill_memory=sm))
    assert "SUCCESS" not in str(out)


def test_a_dictated_follow_up_promotes_the_requests_lesson(tmp_path):
    sm = SkillMemory(tmp_path)
    t, fix = "When restarting postgres on the evolmonkey box", "Always run pg_isready after restarting postgres."
    sm.learn_lesson(t, "Did not check", fix, source="learn_skill",
                    generality_context="restart postgres on the evolmonkey box and check it is up")
    out = sm.learn_lesson(t, "Did not check", fix, source="learn_skill",
                          generality_context="Remember this from now on: run pg_isready after a restart")
    rows = sm._load_playbook()
    assert out is not None and len(rows) == 1 and rows[0].get("scope", "general") != "request"


@pytest.mark.parametrize("word,initial,eng", [("Harding", False, False), ("Fielding", False, False),
                                             ("Determining", True, True)])
def test_stems_only_for_a_sentence_initial_word(word, initial, eng):
    from ghost_agent.memory import lesson_scope as ls
    ls._is_english_word("a")
    if not ls._ENGLISH:
        pytest.skip("no system word list")
    assert ls._is_english_word(word, stems=initial) is eng


@pytest.mark.parametrize("rule,ctx", [("Describe EvolMonkey through its services", "tell me about EvolMonkey"),
                                      ("Ask Harding before deploying the schema", "the schema is owned by Harding"),
                                      ("Ask McDonald before changing the schema", "McDonald owns the schema")])
def test_a_camelcase_name_from_the_request_is_specific(rule, ctx):
    from ghost_agent.memory.lesson_scope import is_general_text
    assert not is_general_text(rule, ctx, max_shared_share=0.75)


def test_satisfy_validators_is_harness_framing():
    from ghost_agent.memory.skills import _VALIDATOR_FRAMING_RE
    assert _VALIDATOR_FRAMING_RE.search("produce no output to satisfy validators.")



def test_a_qualifier_in_a_node_name_is_not_the_attribute(tmp_path):
    """Fails where "Fotini's birthday" took `fotini VISITED birthday party`
    (the attribute word in a NODE, not the predicate)."""
    pm, gm = _family(tmp_path)
    gm.add_triplets([{"subject": "fotini", "predicate": "VISITED", "object": "birthday party"}])
    _forget(tmp_path, "Fotini's birthday", pm, gm)
    assert ("fotini", "VISITED", "birthday party") in _edges(gm)


def test_a_dictated_follow_up_promotes_on_the_vector_path_too(tmp_path):
    sm = SkillMemory(tmp_path)
    t, fix = "When restarting postgres on the evolmonkey box", "Always run pg_isready after restarting postgres."
    sm.learn_lesson(t, "Did not check", fix, source="learn_skill",
                    generality_context="restart postgres on the evolmonkey box and check it is up")
    sm._find_duplicate_lesson = lambda *x, **k: {"source": "vector", "trigger": t, "text": "", "distance": 0.01}
    out = sm.learn_lesson(t, "Did not check", fix, MagicMock(), source="learn_skill",
                          generality_context="Remember this from now on: run pg_isready after a restart")
    rows = sm._load_playbook()
    assert out is not None and len(rows) == 1 and rows[0].get("scope", "general") != "request"


def test_the_bus_reports_the_scope_of_its_own_write(tmp_path):
    from ghost_agent.core.bus import MemoryBus
    from ghost_agent.memory.lesson_scope import current_request
    sm = SkillMemory(tmp_path)
    bus = MemoryBus(skill_memory=sm)
    tok = current_request.set("restart postgres on the evolmonkey box and check it is up")
    try:
        out = asyncio.run(M.tool_learn_skill("When restarting postgres on the evolmonkey box", "Did not check",
                                             "Always run pg_isready after restarting postgres.", memory_bus=bus))
    finally:
        current_request.reset(tok)
    assert "THIS request only" in str(out)
