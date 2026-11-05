"""Retrieval review 2026-10-31 (§4SN): how the owner's stored memory gets back
into a prompt. Each test names the replayed miss or live failure it pins; the
stores are real (tmp dirs)."""
import asyncio
import datetime as dt
import sqlite3
import tempfile
from pathlib import Path

import pytest

from ghost_agent.memory.graph import GraphMemory
from ghost_agent.memory.profile import ProfileMemory
from ghost_agent.memory.vector import VectorMemory
from ghost_agent.tools import memory as M


def _vm():
    return VectorMemory(Path(tempfile.mkdtemp()), "http://mock-url")


def _g(tmp_path, *edges):
    g = GraphMemory(tmp_path)
    g.add_triplets([{"subject": s, "predicate": p, "object": o} for s, p, o in edges], raw=True)
    return g


# ── R1: a fact asked for by its kind, in the words people use ───────────────
@pytest.mark.parametrize("q", [
    "τι προβλήματα υγείας έχω;",          # genitive — the key is υγεια
    "do I have heart problems?",
    "what's wrong with my heart?",
    "έχω πρόβλημα με την καρδιά μου;",
])
def test_health_facts_are_reached_by_common_phrasings(tmp_path, q):
    g = _g(tmp_path, ("user", "HAS_CONDITION", "asthma"), ("user", "TAKES_MEDICATION", "ventolin"))
    got = g.owner_facts_matching(q, 8)
    assert any("asthma" in x for x in got), got


@pytest.mark.parametrize("q, want", [
    ("what pets do I have", "bruno"),
    ("what motorcycle do I ride", "triumph"),
    ("what cars do I have", "triumph"),
    ("what's my kid's name", "kosmas"),
    ("what do I specialise in", "postgresql"),
    ("τι κατοικίδια έχω;", "bruno"),
])
def test_synonyms_of_a_kind_reach_its_field(tmp_path, q, want):
    g = _g(tmp_path, ("user", "HAS_PET_DOG_NAME", "bruno"), ("user", "HAS_ASSETS_VEHICLES", "triumph bonneville"),
           ("user", "HAS_RELATIONSHIPS_SONS", "kosmas"), ("user", "SPECIALIZES_IN", "postgresql"),
           ("user", "HAS_RELATIONSHIPS_WIFE_NAME", "elena"))
    got = g.owner_facts_matching(q, 8)
    assert any(want in x for x in got), got


@pytest.mark.parametrize("q", ["best heart rate monitor", "a bike route to the coast", "the cat command in bash",
                               "υγείας υπουργείο νέα"])
def test_the_new_kind_words_open_on_an_owner_question_only(tmp_path, q):
    g = _g(tmp_path, ("user", "HAS_CONDITION", "asthma"), ("user", "HAS_PET_DOG_NAME", "bruno"),
           ("user", "HAS_ASSETS_VEHICLES", "triumph bonneville"))
    got = g.owner_facts_matching(q, 8, extended=False)
    assert not any(x in " ".join(got) for x in ("asthma", "bruno", "triumph")), got


def test_greek_inflection_matching_is_greek_only():
    K = GraphMemory._CATEGORY_WORDS
    assert GraphMemory._greek_kind_key("υγειασ", K) == "υγεια"
    assert GraphMemory._greek_kind_key("παιδιων", K) == "παιδια"
    assert GraphMemory._greek_kind_key("προβληματα", K) == ""
    assert GraphMemory._greek_kind_key("homes", K) == ""                # English stays exact
    assert GraphMemory._greek_kind_key("μεν", K) == ""


# ── R2: a street address under an ordinary key stays out of the graph ───────
def test_a_street_address_under_location_is_not_mirrored(tmp_path):
    pm, gm = ProfileMemory(tmp_path), GraphMemory(tmp_path)
    pm.update("root", "location", "Kaledonias 83, Kalopetres 10101")
    asyncio.run(M.sync_owner_mirrors("root", "location", pm, gm, None))
    with sqlite3.connect(gm.db_path) as c:
        objs = [r[0] for r in c.execute("select object from triplets where valid_until is null")]
    assert not any("kaledonias" in o for o in objs), objs
    assert not any("Kaledonias" in x for x in gm.owner_facts_matching("where do I live?", 8))


def test_an_ordinary_location_is_still_mirrored(tmp_path):
    pm, gm = ProfileMemory(tmp_path), GraphMemory(tmp_path)
    pm.update("root", "location", "Athens, Greece")
    asyncio.run(M.sync_owner_mirrors("root", "location", pm, gm, None))
    assert any("athens" in x for x in gm.owner_facts_matching("where do I live?", 8))


# ── R3: the date shown is when the fact was learned ─────────────────────────
def test_a_mirror_edge_is_dated_when_the_profile_learned_it(tmp_path):
    """Live: the 10-30 re-sync made every owner edge "as of 2026-10-30"
    beside the profile's "as of 2026-07-07"; the model chose the newer."""
    pm, gm, vm = ProfileMemory(tmp_path), GraphMemory(tmp_path), _vm()
    pm.update("root", "location", "Athens, Greece", as_of="2026-07-07T21:28:19")
    asyncio.run(M.sync_owner_mirrors("root", "location", pm, gm, vm))
    got = gm.owner_facts_matching("where do I live?", 8)
    assert any("(as of 2026-07-07)" in x for x in got), got
    ts = vm.collection.get(where={"type": "identity"}, include=["metadatas"])["metadatas"][0]["timestamp"]
    assert ts.startswith("2026-07-07")


def test_a_stated_edge_shows_its_stated_date_not_its_last_touch(tmp_path):
    gm = GraphMemory(tmp_path)
    said = dt.datetime(2026, 8, 2, tzinfo=dt.timezone.utc).timestamp()
    gm.add_triplets([{"subject": "user", "predicate": "VISITED", "object": "nafplio"}], as_of=said)
    gm.add_triplets([{"subject": "user", "predicate": "VISITED", "object": "nafplio"}], as_of=said)   # a bump
    got = gm.owner_facts_matching("where have I visited", 8)
    assert any("(as of 2026-08-02)" in x for x in got), got


# ── R4: names are shown as stored, not re-cased ─────────────────────────────
def test_names_are_not_title_cased():
    line = GraphMemory._format_path((("user", "HAS_ASSETS_VEHICLES", "bmw 320d"),), 1)
    assert line == "- (User) -[HAS_ASSETS_VEHICLES]-> (bmw 320d)"
    line = GraphMemory._format_path((("user", "HAS_ROOT_COMPANY", "orbitdata - postgresql services"),), 1)
    assert "Orbitdata" not in line and "Postgresql" not in line


# ── R5 / R6 / R9: what `recall` prints ──────────────────────────────────────
def _recall(q, pm, gm, vm):
    return asyncio.run(M.tool_recall(q, memory_system=vm, graph_memory=gm, profile_memory=pm))


def test_recall_prints_an_owner_edge_once(tmp_path):
    """Live: every memory-usage edge printed twice — once with " (as of …)"."""
    pm, gm, vm = ProfileMemory(tmp_path), GraphMemory(tmp_path), _vm()
    gm.add_triplets([{"subject": "user", "predicate": "HAS_PREFERENCES_MEMORY_USAGE", "object": "high"}], raw=True)
    out = _recall("is high memory usage ok for my preferences?", pm, gm, vm)
    assert out.count("HAS_PREFERENCES_MEMORY_USAGE") == 1, out


def test_the_word_user_does_not_open_the_owners_edges():
    """Live: "what's the postgres user home directory" printed 15 owner edges."""
    edges = ["- (User) -[HAS_RELATIONSHIPS_WIFE_BIRTHDATE]-> (1982-01-07)", "- (User) -[VISITED]-> (nafplio)",
             "- (postgres) -[HAS_HOME]-> (/var/lib/postgresql)"]
    kept = M._graph_edges_on_topic("what's the postgres user home directory", edges)
    assert kept == ["- (postgres) -[HAS_HOME]-> (/var/lib/postgresql)"]


def test_a_one_line_past_request_with_no_shared_word_is_not_shown(tmp_path):
    """Live: "status update !" (retrieval_count 325) sat MEDIUM under 9 of 18
    owner questions."""
    pm, gm, vm = ProfileMemory(tmp_path), GraphMemory(tmp_path), _vm()
    vm.add("status update !", {"type": "episode", "timestamp": "2026-07-26T19:32:19Z"})
    vm.add("compare postgres logical replication slots", {"type": "episode", "timestamp": "2026-07-26T19:32:19Z"})
    out = _recall("what do you know about logical replication?", pm, gm, vm)
    assert "status update" not in out
    assert "logical replication slots" in out


# ── R8: the owner's own age is worked out ───────────────────────────────────
def test_a_birthdate_field_carries_its_age(tmp_path):
    pm = ProfileMemory(tmp_path)
    pm.update("root", "birthdate", "1980-01-26")
    pm.update("relationships", "wife_birthdate", "1982-01-07")
    s = pm.get_context_string()
    today = dt.date.today()
    age = today.year - 1980 - ((today.month, today.day) < (1, 26))
    assert f"birthdate: 1980-01-26 → {age} years old" in s
    assert "wife_birthdate: 1982-01-07 → " in s
    assert pm.load()["root"]["birthdate"] == "1980-01-26"                  # the value itself unchanged


def test_a_born_value_is_not_glossed_twice(tmp_path):
    pm = ProfileMemory(tmp_path)
    pm.update("relationships", "son_birthdate", "born 2016-11-22")
    assert pm.get_context_string().count("→") == 1


def test_the_restamp_script_only_moves_mirror_dates_earlier(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "restamp", Path(__file__).resolve().parents[1] / "scripts" / "restamp_owner_mirrors_4sn.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pm, gm = ProfileMemory(tmp_path), GraphMemory(tmp_path)
    pm.update("root", "location", "Athens, Greece", as_of="2026-07-07T21:28:19")
    pm.update("root", "name", "Vasilis", as_of="2026-09-04T07:52:51")
    late = dt.datetime(2026, 10, 30, tzinfo=dt.timezone.utc).timestamp()
    early = dt.datetime(2026, 7, 27, tzinfo=dt.timezone.utc).timestamp()
    gm.add_triplets([{"subject": "user", "predicate": "HAS_LOCATION", "object": "athens, greece"}], raw=True, as_of=late)
    gm.add_triplets([{"subject": "user", "predicate": "HAS_NAME", "object": "vasilis"}], raw=True, as_of=early)
    gm.add_triplets([{"subject": "user", "predicate": "VISITED", "object": "nafplio"}], as_of=late)
    rows = mod.plan(tmp_path)
    assert [(p, v) for _r, p, v, _vf, _e in rows] == [("HAS_LOCATION", "athens, greece")]
    assert mod.main(["--memory-dir", str(tmp_path), "--apply"]) == 0
    assert mod.plan(tmp_path) == []
    assert list(tmp_path.glob("knowledge_graph.db.pre-4sn-*"))
    assert any("(as of 2026-07-07)" in x for x in gm.owner_facts_matching("where do I live?", 8))


# ── review round 1 ──────────────────────────────────────────────────────────
def test_a_row_without_valid_from_sorts_by_its_timestamp(tmp_path):
    g = GraphMemory(tmp_path)
    old = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc).timestamp()
    g.add_triplets([{"subject": "user", "predicate": "VISITED", "object": f"town{i}"} for i in range(25)], as_of=old)
    with sqlite3.connect(g.db_path) as c:
        c.execute("INSERT INTO triplets (subject, predicate, object, weight, timestamp, valid_from) "
                  "VALUES ('user', 'VISITED', 'legacy', 1, '2025-01-01 00:00:00', NULL)")
    got = g.owner_facts_matching("where have I visited", 20)
    assert not any("legacy" in x for x in got), got            # the oldest, not first


def test_recall_dedups_a_weighted_edge_too(tmp_path):
    pm, gm, vm = ProfileMemory(tmp_path), GraphMemory(tmp_path), _vm()
    for _ in range(3):
        gm.add_triplets([{"subject": "user", "predicate": "HAS_PET_DOG_NAME", "object": "bruno"}], raw=True)
    out = _recall("what pets do I have, is bruno my dog?", pm, gm, vm)
    assert out.count("HAS_PET_DOG_NAME") == 1, out


def test_a_past_request_with_a_three_letter_topic_is_shown(tmp_path):
    pm, gm, vm = ProfileMemory(tmp_path), GraphMemory(tmp_path), _vm()
    vm.add("my car makes a noise", {"type": "episode", "timestamp": "2026-07-26T19:32:19Z"})
    vm.add("proceed", {"type": "episode", "timestamp": "2026-07-26T19:32:19Z"})
    out = _recall("what did I say about my car yesterday", pm, gm, vm)
    assert "my car makes a noise" in out and "CONTENT: proceed" not in out


def test_education_is_not_children():
    K = GraphMemory._CATEGORY_WORDS
    assert GraphMemory._greek_kind_key("παιδεια", K) == ""
    assert GraphMemory._greek_kind_key("δουλειεσ", K) == "δουλεια"


# ── battery survivors (round 1) ─────────────────────────────────────────────
def test_a_question_without_content_words_keeps_every_past_request():
    from ghost_agent.tools.memory import _episode_on_topic
    assert _episode_on_topic("and so?", "status update !")


def test_the_restamp_apply_takes_the_writer_lock(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "restamp_lock", Path(__file__).resolve().parents[1] / "scripts" / "restamp_owner_mirrors_4sn.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pm, gm = ProfileMemory(tmp_path), GraphMemory(tmp_path)
    pm.update("root", "location", "Athens, Greece", as_of="2026-07-07T21:28:19")
    late = dt.datetime(2026, 10, 30, tzinfo=dt.timezone.utc).timestamp()
    gm.add_triplets([{"subject": "user", "predicate": "HAS_LOCATION", "object": "athens, greece"}], raw=True, as_of=late)

    def _locked(md, who):
        raise SystemExit("refusing to run: held")
    monkeypatch.setattr(mod, "assert_no_other_writer", _locked)
    with pytest.raises(SystemExit):
        mod.main(["--memory-dir", str(tmp_path), "--apply"])
    assert not list(tmp_path.glob("knowledge_graph.db.pre-4sn-*"))     # nothing written


# ── review round 2 ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("q", ["what did we talk about this morning?", "what did we discuss last night?",
                               "τι συζητήσαμε χθες;"])
def test_a_when_question_keeps_past_requests(q):
    from ghost_agent.tools.memory import _episode_on_topic
    assert _episode_on_topic(q, "my car makes a noise")


def test_plurals_reach_a_past_request():
    from ghost_agent.tools.memory import _episode_on_topic
    assert _episode_on_topic("what did I say about my kids", "my kid's school starts monday")
    assert _episode_on_topic("what did I say about my cars", "my car makes a noise")


def test_greek_function_words_are_not_a_topic():
    from ghost_agent.tools.memory import _episode_on_topic
    assert not _episode_on_topic("τι σου είπα για τον σκύλο μου", "τι καιρό θα κάνει για τον γάμο μου")


@pytest.mark.parametrize("w, want", [("γιο", "γιοσ"), ("γιου", "γιοσ"), ("κορες", "κορη"), ("παιδικο", ""),
                                     ("σπιτικο", ""), ("υγειας", "υγεια"), ("παιδιων", "παιδια")])
def test_greek_inflections_are_case_endings(w, want):
    K = GraphMemory._CATEGORY_WORDS
    got = GraphMemory._greek_kind_key(w, K)
    assert got == want or (want and got.replace("ς", "σ") == want), got


# ── review round 3 ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("q", ["what did we talk about the other day?", "what did I ask you two days ago?",
                               "what have we been talking about?", "τι σου ζήτησα χθες;", "τι είχαμε πει χθες;",
                               "τι συζητήσαμε την προηγούμενη εβδομάδα;"])
def test_more_when_questions_keep_past_requests(q):
    from ghost_agent.tools.memory import _episode_on_topic
    assert _episode_on_topic(q, "my car makes a noise")


def test_ma_nouns_reach_their_key():
    K = GraphMemory._CATEGORY_WORDS
    assert GraphMemory._greek_kind_key("ονοματα", K) == "ονομα"
