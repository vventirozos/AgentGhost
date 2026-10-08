"""§4MJ (2026-10-08): memory hydration — what reaches every prompt. Each test
names the world it FAILS in, measured by replaying 80-239 real owner requests
through the real bus on a copy of the live stores."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core.bus import MemoryBus, _episode_is_hydratable, _GRAPH_HUB_WORDS
from ghost_agent.core.sessions import _densest_window


# ── episodes ────────────────────────────────────────────────────────────────
def test_a_failed_episode_without_a_lesson_is_not_hydrated():
    """Fails where 87 of 195 injected episodes were failures rendered as the
    failed reply's text ("FAILURE — I hit a hard limit…"); 2 were relevant."""
    assert not _episode_is_hydratable({"trigger": "hello ghost", "outcome_success": 0, "outcome": "I hit a hard limit"}, "hello ghost how are you")
    assert _episode_is_hydratable({"trigger": "x", "outcome_success": 0, "lesson": "check the port"}, "x")
    assert _episode_is_hydratable({"trigger": "x", "outcome_success": 1}, "x")
    assert _episode_is_hydratable({"trigger": "x"}, "x")                 # no outcome recorded: unknown, kept


def test_a_greek_request_needs_shared_words_not_the_english_embedder():
    """Fails where every Greek request received the store's Greek episodes
    whatever their topic (any two Greek sentences sit ~0.1 apart)."""
    q = "ψάξε το μασονικό βιογραφικό του Στέφανου Παϊπέτη"
    assert not _episode_is_hydratable({"trigger": "τα εγκλήματα του Μιχαήλ Βόδα", "outcome_success": 1}, q)
    assert _episode_is_hydratable({"trigger": "βιογραφικό του Στέφανου Παϊπέτη", "outcome_success": 1}, q)
    assert _episode_is_hydratable({"trigger": "fix the docker probe", "outcome_success": 1}, "how did I fix the docker probe")


@pytest.mark.asyncio
async def test_the_episodic_tier_applies_the_filter():
    ep = MagicMock()
    ep.search_similar = MagicMock(return_value=[
        {"trigger": "hello ghost", "outcome": "I hit a hard limit", "outcome_success": 0},
        {"trigger": "hello ghost", "outcome": "Hi!", "outcome_success": 1}])
    ep.format_episode = lambda e: e["outcome"]
    items = await MemoryBus(episodic_memory=ep)._fetch_episodic("hello ghost, what's going on today")
    assert [i["text"] for i in items] == ["Hi!"]


def test_stored_tool_call_markup_is_stripped_from_a_rendered_episode():
    from ghost_agent.memory.episodes import EpisodicMemory
    line = EpisodicMemory.format_episode({"trigger": "t", "outcome_success": 1,
                                          "outcome": "<tool_call>{\"name\": \"execute\"}</tool_call> done"})
    assert "<tool_call>" not in line and "done" in line


# ── graph ───────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_hydration_never_seeds_a_generic_hub_word_or_a_fuzzy_match(tmp_path):
    """Fails where 'project' seeded the node holding every project's title
    (WebOS chains on 16% of real turns) and 'sitting' fuzzy-seeded
    `stealthing`."""
    from ghost_agent.memory.graph import GraphMemory
    gm = GraphMemory(tmp_path)
    gm.add_triplets([{"subject": "project", "predicate": "HAS_TITLE", "object": "webos"},
                     {"subject": "stealthing", "predicate": "IS_DEFINED_AS", "object": "something"},
                     {"subject": "chess coach", "predicate": "USES", "object": "stockfish"}])
    bus = MemoryBus(graph_memory=gm)
    out = await bus._fetch_graph("resume the project and keep sitting on the chess coach engine")
    text = " ".join(i["text"] for i in out).lower()
    assert "webos" not in text and "stealthing" not in text
    assert "stockfish" in text
    assert "project" in _GRAPH_HUB_WORDS
    assert gm.get_neighborhood(["sitting"], 15) != []                  # recall keeps the fuzzy tier
    assert gm.get_neighborhood(["sitting"], 15, fuzzy=False) == []


# ── sessions ────────────────────────────────────────────────────────────────
def test_a_long_reply_matches_only_where_the_terms_sit_together():
    """Fails where two 5.9 KB replies matched 23 of 80 unrelated requests on
    any two words and injected their opening, not the match."""
    reply = ("intro " * 300) + "postgres here " + ("filler " * 300) + "migration plan there"
    lowered = reply.lower()
    start, n = _densest_window(lowered, {"postgres", "migration"}, 240)
    assert n == 1                                                       # never together in 240 chars
    reply2 = ("intro " * 300) + "the postgres migration plan step one"
    start2, n2 = _densest_window(reply2.lower(), {"postgres", "migration"}, 240)
    assert n2 == 2 and "postgres" in reply2[start2:start2 + 240]


def test_the_session_search_returns_the_matching_window(tmp_path):
    """Driven through the real store: the hit's text is the window around
    the terms, and a reply where the terms are far apart is no hit."""
    from ghost_agent.core.sessions import SessionStore
    st = SessionStore(tmp_path)
    sid = st.create(title="db work").id
    near = ("background " * 200) + "the postgres migration needs a rollback plan"
    assert st.append_turn(sid, [{"role": "user", "content": "plan it please"}], near)
    hits = st.search_messages("postgres migration rollback", 5)
    assert hits and "postgres migration" in hits[0]["text"] and hits[0]["text"].startswith("…")
    sid2 = st.create(title="other").id
    far = ("background " * 200) + "postgres" + (" filler" * 200) + " migration"
    assert st.append_turn(sid2, [{"role": "user", "content": "something unrelated"}], far)
    st._list_memo = (0, [])                     # drop the 30 s list memo
    assert all(h["session_id"] != sid2 for h in st.search_messages("postgres migration", 5))


# ── budget ──────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_the_budget_scales_on_the_users_words_not_the_expansion():
    bus = MemoryBus()
    seen = {}

    def fmt(fused, max_chars):
        seen["max_chars"] = max_chars
        return "", []
    bus._format_markdown_with_survivors = fmt
    bus._fetch_all_tiers = AsyncMock(return_value=[])
    bus._log_hydration = lambda *a, **k: None
    expanded = "Context: " + "word " * 60 + "| User intent: and the second one?"
    await bus.hydrate_context(expanded, context_budget=4000, raw_user_text="and the second one?")
    assert seen["max_chars"] == 4000


# ── the refit population ────────────────────────────────────────────────────
def test_the_refit_reads_only_the_gated_era(tmp_path):
    from ghost_agent.core.dream import Dreamer, RRF_OBSERVATIONS_CLEAN_SINCE
    ledger = tmp_path / "rrf" / "observations.jsonl"; ledger.parent.mkdir(parents=True)
    old = [{"intent": "procedural", "source": "graph", "success": False, "turn": f"o{i}",
            "ts": "2026-09-16T00:00:00Z"} for i in range(60)]
    ledger.write_text("\n".join(json.dumps(r) for r in old) + "\n")
    bus = MemoryBus(usefulness_ledger_path=ledger)
    d = Dreamer(SimpleNamespace(memory_bus=bus, memory_system=MagicMock()))
    assert d._refit_rrf_weights(min_observations=30) is False          # every row predates the gates
    assert RRF_OBSERVATIONS_CLEAN_SINCE == "2026-09-24T18:31"


# ── the public channel ──────────────────────────────────────────────────────
def test_the_autobiography_capture_refuses_a_public_reply():
    """Fails where an owner turn in a public channel (a member's diagnosis in
    the thread) was written into the owner's autobiography."""
    import ast, inspect
    from ghost_agent.core import agent as A
    tree = ast.parse(inspect.getsource(A))
    for n in ast.walk(tree):
        if isinstance(n, ast.If) and "requester_is_member" in ast.dump(n.test) and "_SelfModel" in ast.dump(n.test):
            assert "reply_is_public" in ast.dump(n.test)
            return
    pytest.fail("autobiography capture gate not found")


def test_no_owner_name_in_a_tool_schema():
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    assert "Fotini" not in json.dumps(TOOL_DEFINITIONS)


# ── lens 3 (code) ───────────────────────────────────────────────────────────
def test_a_chain_contained_in_a_longer_chain_is_not_returned_twice(tmp_path):
    """Fails where '(User)-[DESIGNS]->(X)-[REPRESENTS]->(Y)' and its own
    sub-path '(X)-[REPRESENTS]->(Y)' took two of the graph tier's 6 slots
    (102 pairs on 16% of real turns)."""
    from ghost_agent.memory.graph import GraphMemory
    gm = GraphMemory(tmp_path)
    gm.add_triplets([{"subject": "user", "predicate": "DESIGNS", "object": "ouroboros"},
                     {"subject": "ouroboros", "predicate": "REPRESENTS", "object": "recursion"}])
    out = gm.get_neighborhood(["ouroboros"], 15, fuzzy=False)
    assert any("Designs" in l.title() or "DESIGNS" in l for l in out)
    assert not any(l.startswith("- (Ouroboros) -[REPRESENTS]->") for l in out), out


@pytest.mark.parametrize("q,intent", [
    ("hello ghost, how are you?", "contextual"),
    ("how are things today", "contextual"),
    ("how old is leonidas?", "factual"),
    ("how do I fix the docker probe", "procedural"),
    ("how to configure nginx", "procedural"),
    ("should I drive or walk to the car wash", "contextual"),
    ("what?", "factual"),
    ("hello ghost, what's up ?", "contextual"),
    ("good morning ghost, what's the news ?", "contextual"),
])
def test_the_intent_classifier_reads_how_and_punctuation(q, intent):
    """Fails where greetings and 'how old' came out procedural (33 of 36
    procedural turns were not how-to) and 'what?' was not 'what'."""
    assert MemoryBus._classify_query_intent(q) == intent


@pytest.mark.asyncio
async def test_a_turns_stash_survives_an_overlapping_turns_hydration():
    """Fails where the single stash slot was taken by a second turn's
    hydration before the first turn's (staggered) judge read it."""
    bus = MemoryBus()
    bus._fetch_all_tiers = AsyncMock(return_value=[[{"source": "graph", "text": "- (A) -[R]-> (B)"}]])
    bus._log_hydration = lambda *a, **k: None
    bus._credit_surfaced = lambda s: None
    await bus.hydrate_context("first question about A", turn_id="t1")
    await bus.hydrate_context("second question about A", turn_id="t2")
    assert "t1" in bus._hydration_by_turn and "t2" in bus._hydration_by_turn
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": "USED: 1"}}]})
    await bus.judge_hydration_usefulness("reply about A", llm, turn_id="t1")
    assert "t1" not in bus._hydration_by_turn and "t2" in bus._hydration_by_turn


def test_the_active_session_is_excluded_before_the_limit(tmp_path):
    from ghost_agent.core.sessions import SessionStore
    st = SessionStore(tmp_path)
    active = st.create(title="active").id
    for i in range(9):
        st.append_turn(active, [{"role": "user", "content": f"postgres migration step {i}"}],
                       f"postgres migration answer {i}")
    other = st.create(title="older").id
    st.append_turn(other, [{"role": "user", "content": "the postgres migration from last week"}], "ok")
    st._list_memo = (0, [])
    hits = st.search_messages("postgres migration", 8, exclude_session_id=active)
    assert hits and all(h["session_id"] == other for h in hits)


@pytest.mark.asyncio
async def test_hydrate_passes_k10_and_consensus_normalisation():
    """M01/M02 (lens 3): nothing pinned what hydrate_context passes to RRF."""
    bus = MemoryBus()
    bus._fetch_all_tiers = AsyncMock(return_value=[[{"source": "graph", "text": "x"}]])
    bus._log_hydration = lambda *a, **k: None
    seen = {}
    orig = bus._reciprocal_rank_fusion

    def spy(lists, **kw):
        seen.update(kw)
        return orig(lists, **kw)
    bus._reciprocal_rank_fusion = spy
    await bus.hydrate_context("a question about x")
    assert seen["k"] == 10 and seen["normalize_consensus"] is True


@pytest.mark.asyncio
async def test_a_simple_query_gets_exactly_the_callers_budget():
    """M06 (lens 3): the existing pin was masked by the per-source cap."""
    bus = MemoryBus()
    bus._fetch_all_tiers = AsyncMock(return_value=[])
    bus._log_hydration = lambda *a, **k: None
    seen = {}
    bus._format_markdown_with_survivors = lambda fused, max_chars: (seen.setdefault("m", max_chars), ("", []))[1]
    await bus.hydrate_context("where is the car", context_budget=4000)
    assert seen["m"] == 4000


def test_intent_is_classified_on_the_users_words():
    """M07 (lens 3): the expansion's assistant prose must not pick the intent."""
    q = "Context: how to fix the error, avoid the mistake, follow the steps | User intent: who wrote it?"
    assert MemoryBus._classify_query_intent(MemoryBus._intent_source_text(q, "who wrote it?")) == "factual"


# ── the retired vector tier and project episodes (operator decisions) ───────
@pytest.mark.vector_tier_off
def test_the_vector_tier_is_retired_by_default_in_production():
    """Fails where the vector tier queried the store every turn for 0
    injected items (every admissible type is a twin of another tier)."""
    import os, importlib, ghost_agent.core.bus as B
    assert os.getenv("GHOST_BUS_VECTOR_TIER") in (None, "", "0")
    assert B.MemoryBus._VECTOR_TIER_ENABLED is False


@pytest.mark.vector_tier_off
@pytest.mark.asyncio
async def test_a_retired_vector_tier_never_queries_the_store():
    vec = MagicMock()
    vec.search_items = MagicMock(return_value=[{"text": "x", "type": "auto", "id": "1"}])
    assert await MemoryBus(vector_memory=vec)._fetch_vector("anything at all") == []
    assert vec.search_items.call_count == 0


def test_a_deleted_projects_episodes_are_matched_by_id_and_distinctive_title(tmp_path):
    """Fails where 16% of episodes named deleted projects and were recalled;
    and where a title shared with a LIVE project ('Chess Coach' vs the live
    'Chess Coach v3') or a project LIST in a tool result matched."""
    from ghost_agent.memory.episodes import EpisodicMemory as E
    em = E(tmp_path)
    import sqlite3
    with sqlite3.connect(em.db_path) as c:
        c.execute("DELETE FROM episodes")
        for trig, out in [("resume 30d5d5b65c38 and start the server", "ok"),
                          ("play a game with the Chess Coach v3", "ok"),
                          ("create a new project called 'prince of persia'", "ok"),
                          ("show me all projects", "ok")]:
            c.execute("INSERT INTO episodes (trigger, context, outcome, outcome_success, timestamp) VALUES (?,?,?,?,?)",
                      (trig, "", out, 1, "2026-10-01T00:00:00"))
        ep_list = c.execute("SELECT id FROM episodes WHERE trigger='show me all projects'").fetchone()[0]
        c.execute("INSERT INTO episode_actions (episode_id, action_order, tool_name, tool_args, result) VALUES (?,?,?,?,?)",
                  (ep_list, 0, "manage_projects", "{}", "Prince of Persia (bd75420e2d96), Chess Coach (30d5d5b65c38)"))
    live = ["Chess Coach v3", "WebOS"]
    assert not E.project_title_is_distinctive("Chess Coach", live)
    assert not E.project_title_is_distinctive("Meta", live)
    assert E.project_title_is_distinctive("Prince of Persia", live)
    trig = lambda ids: sorted(r[0] for r in sqlite3.connect(em.db_path).execute(
        f"SELECT trigger FROM episodes WHERE id IN ({','.join(map(str, ids)) or 0})"))
    assert trig(em.project_mention_ids("30d5d5b65c38", "Chess Coach", live)) == ["resume 30d5d5b65c38 and start the server"]
    assert trig(em.project_mention_ids("bd75420e2d96", "Prince of Persia", live)) == ["create a new project called 'prince of persia'"]
    assert em.forget_project("30d5d5b65c38", "Chess Coach", live) == 1
    arch = (tmp_path / "episodes_forgotten.jsonl").read_text()
    assert "project deleted: 30d5d5b65c38" in arch


def test_hard_delete_forgets_the_projects_episodes():
    import ast, inspect
    from ghost_agent.tools import projects as P
    src = ast.parse(inspect.getsource(P))
    calls = [n for n in ast.walk(src) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_forget_project_episodes"]
    assert calls, "hard delete no longer forgets the project's episodes"
    ctx = SimpleNamespace(episodic_memory=MagicMock(), project_store=SimpleNamespace(
        list_projects=lambda: [{"id": "a", "title": "Chess Coach v3"}, {"id": "gone", "title": "Chess Coach"}]),
        memory_system=None)
    ctx.episodic_memory.forget_project = MagicMock(return_value=2)
    assert P._forget_project_episodes(ctx, "gone", "Chess Coach") == 2
    kw = ctx.episodic_memory.forget_project.call_args.kwargs
    assert kw["live_titles"] == ["Chess Coach v3"] and kw["title"] == "Chess Coach"


def test_the_judge_spawner_reads_this_turns_own_stash():
    """r2 MAJOR: the spawner checked only the single slot and returned when an
    overlapping turn had hydrated since — the per-turn stash was never read."""
    from ghost_agent.core.agent import GhostAgent
    agent = GhostAgent.__new__(GhostAgent)
    bus = MemoryBus()
    bus._hydration_by_turn["t1"] = {"turn_id": "t1", "survivors": [{"source": "graph", "text": "x"}], "ts": 0}
    bus.last_hydration = {"turn_id": "t2", "survivors": [{"source": "graph", "text": "y"}], "ts": 0}
    spawned = []
    bus.judge_hydration_usefulness = lambda *a, **k: spawned.append(k.get("turn_id")) or __import__("asyncio").sleep(0)
    agent.context = SimpleNamespace(memory_bus=bus, llm_client=MagicMock(), args=SimpleNamespace(model="m"))
    import ghost_agent.core.agent as A
    orig = A.turn_may_teach
    A.turn_may_teach = lambda ctx: True
    try:
        import ghost_agent.utils.logging as L
        o2 = L.spawn_bg
        L.spawn_bg = lambda coro, name="": (coro.close() if hasattr(coro, "close") else None, spawned.append("spawned"))
        try:
            agent._judge_hydration_safe("reply", turn_id="t1")
        finally:
            L.spawn_bg = o2
    finally:
        A.turn_may_teach = orig
    assert spawned, "the judge for t1 was not spawned"


@pytest.mark.asyncio
async def test_a_greek_follow_up_is_judged_on_the_users_words():
    ep = MagicMock()
    ep.search_similar = MagicMock(return_value=[{"trigger": "τα εγκλήματα του Μιχαήλ Βόδα", "outcome": "x", "outcome_success": 1}])
    ep.format_episode = lambda e: e["trigger"]
    bus = MemoryBus(episodic_memory=ep)
    expanded = "Context: here is a long English answer about the history of the city | User intent: και ο δήμαρχος;"
    assert await bus._fetch_episodic(expanded, raw_user_text="και ο δήμαρχος;") == []
