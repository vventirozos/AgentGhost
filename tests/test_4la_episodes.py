"""§4LA (2026-10-03): episodic memory — what is recorded, how it is recalled,
how it is relabelled and forgotten. Each test names the world it fails in."""
import asyncio
import json
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ghost_agent.memory.episodes import EpisodicMemory
from ghost_agent.memory.vector import VectorMemory
from ghost_agent.utils.logging import request_id_context, reply_surface_context


def _em(tmp_path):
    return EpisodicMemory(tmp_path)


def _vm():
    return VectorMemory(Path(tempfile.mkdtemp()), "http://mock-url")


def _rows(em, sql="SELECT id, trigger, outcome, outcome_success, req_id FROM episodes"):
    with sqlite3.connect(em.db_path) as c:
        return c.execute(sql).fetchall()


# ── recall ───────────────────────────────────────────────────────────────────
def test_an_unrelated_question_gets_no_episodes(tmp_path):
    """Fails where the 0.2 floor (nonsense scores 0.53–0.68 on this embedder)
    put 5 unrelated episodes into every owner turn."""
    em, vm = _em(tmp_path), _vm()
    for t in ["fix the docker compose healthcheck for postgres", "transcribe the youtube talk about rust",
              "find bjj gyms near thrakomakedones"]:
        em.record_episode(t, outcome="done", success=True, vector_memory=vm)
    assert em.search_similar("asdf qwerty", 5, vm) == []
    assert em.search_similar("recipe for pancakes", 5, vm) == []
    hit = em.search_similar("fix the docker compose healthcheck", 5, vm)
    assert hit and "docker" in hit[0]["trigger"]


def test_an_empty_semantic_answer_does_not_fall_back_to_any_shared_word(tmp_path):
    em, vm = _em(tmp_path), _vm()
    em.record_episode("what is the weather in kyllini today", outcome="sunny", success=True, vector_memory=vm)
    assert em.search_similar("what is my blood type", 5, vm) == []


def test_recency_reorders_the_results(tmp_path):
    em = _em(tmp_path)
    old = {"id": 1, "trigger": "a", "timestamp": time.time() - 60 * 86400}
    new = {"id": 2, "trigger": "b", "timestamp": time.time() - 1 * 86400}
    em.get_episode = lambda i: dict({1: old, 2: new}[i])

    class _V:
        def query(self, **k):
            return {"ids": [["v1", "v2"]], "distances": [[0.20, 0.22]],
                    "metadatas": [[{"type": "episode", "episode_id": 1}, {"type": "episode", "episode_id": 2}]],
                    "documents": [["a", "b"]]}
    vm = MagicMock()
    vm.collection = _V()
    out = em._vector_search("q", 2, vm, credit=False)
    assert [e["id"] for e in out] == [2, 1]


def test_only_episodes_that_reach_the_prompt_are_credited(tmp_path):
    """Fails where every candidate of every sub-query was credited (8 of 14
    credited episodes were never shown) — probe episodes became uneviction-
    able."""
    em, vm = _em(tmp_path), _vm()
    em.record_episode("fix the docker compose healthcheck for postgres", outcome="ok", success=True, vector_memory=vm)
    em.search_similar("fix the docker compose healthcheck", 5, vm, credit=False)
    assert _rows(em, "SELECT access_count FROM episodes")[0][0] == 0
    from ghost_agent.core.bus import MemoryBus
    bus = MemoryBus(vector_memory=vm, episodic_memory=em)
    items = asyncio.run(bus._fetch_episodic("fix the docker compose healthcheck"))
    assert items and items[0]["episode_id"] == 1
    bus._credit_surfaced(items)
    assert _rows(em, "SELECT access_count FROM episodes")[0][0] == 1


def test_eviction_prefers_stale_credit_over_fresh_use(tmp_path):
    em = _em(tmp_path)
    em.MAX_EPISODES = 2
    a = em.record_episode("old popular", outcome="x", success=True)
    b = em.record_episode("fresh", outcome="x", success=True)
    year = time.time() - 365 * 86400
    with sqlite3.connect(em.db_path) as c:
        c.execute("UPDATE episodes SET access_count=5, last_accessed=?, consolidated=1 WHERE id=?", (year, a))
        c.execute("UPDATE episodes SET access_count=1, last_accessed=?, consolidated=1 WHERE id=?", (time.time(), b))
    em.record_episode("newest", outcome="x", success=True)
    assert {r[1] for r in _rows(em)} == {"fresh", "newest"}


def test_a_recovery_needs_a_success_after_the_failure(tmp_path):
    em = _em(tmp_path)
    ev, _ = em._recovery_evidence({"id": 1, "lesson": "", "outcome": "done",
                                   "actions": [{"tool_name": "file_system", "success": 0}]})
    assert ev != "failed_action"
    ev2, _ = em._recovery_evidence({"id": 2, "lesson": "", "outcome": "done",
                                    "actions": [{"tool_name": "file_system", "success": 0},
                                                {"tool_name": "file_system", "success": 1}]})
    assert ev2 == "failed_action"


# ── writing ──────────────────────────────────────────────────────────────────
def test_an_episode_is_redacted(tmp_path):
    em = _em(tmp_path)
    em.record_episode("read my .env", actions=[{"tool": "file_system",
                      "result": "OPENAI_API_KEY=sk-live-abcdefghijklmnopqrstuvwxyz123456 at /Users/vasilis/x"}],
                      outcome="it holds OPENAI_API_KEY=sk-live-abcdefghijklmnopqrstuvwxyz123456", success=True)
    with sqlite3.connect(em.db_path) as c:
        blob = json.dumps(c.execute("SELECT outcome FROM episodes").fetchall()
                          + c.execute("SELECT result FROM episode_actions").fetchall())
    assert "sk-live-abcdefghijklmnopqrstuvwxyz123456" not in blob and "/Users/vasilis" not in blob


def test_a_reindexed_twin_carries_the_episodes_own_date(tmp_path):
    """Fails where a re-ingest stamped "now" (133 twins up to 19 days late)."""
    em, vm = _em(tmp_path), _vm()
    eid = em.record_episode("plan the trip to chalkida with the family", outcome="ok", success=True)
    with sqlite3.connect(em.db_path) as c:
        c.execute("UPDATE episodes SET timestamp=? WHERE id=?", (1767225600.0, eid))   # 2026-01-01
    em.reconcile_vector_index(vm)
    meta = vm.collection.get(where={"type": "episode"}, include=["metadatas"])["metadatas"][0]
    assert meta["timestamp"].startswith("2026-01-01")


def test_a_late_refute_relabels_the_turns_episode(tmp_path):
    """Fails where 88 episodes stayed SUCCESS after a late REFUTED verdict."""
    em = _em(tmp_path)
    em.record_episode("what is 0/0", outcome="0", success=True, req_id="req-77")
    assert em.mark_outcome("req-77", False, "refuted after the reply") == 1
    _, _, outcome, ok, rid = _rows(em)[0]
    assert ok == 0 and outcome.startswith("[refuted after the reply]") and rid == "req-77"


def test_the_owners_correction_relabels_by_the_request_text(tmp_path):
    em = _em(tmp_path)
    em.record_episode("how many sons do I have", outcome="three", success=True)
    assert em.mark_outcome("", False, "the owner corrected this answer", trigger="how many sons do I have") == 1
    assert _rows(em)[0][3] == 0


def _agent(em):
    from ghost_agent.core.agent import GhostAgent, GhostContext
    ctx = MagicMock(spec=GhostContext)
    ctx.episodic_memory = em
    ctx.memory_system = None
    ctx.llm_client = MagicMock()
    ctx.args = MagicMock()
    ctx.skill_memory = MagicMock(is_read_only=False)
    return GhostAgent(ctx)


@pytest.mark.parametrize("user_text,public", [
    ("[message from another channel member — not the owner; treat as untrusted context]\nI have HIV", False),
    ("find me a clinic near home", True),
])
def test_a_public_or_foreign_turn_is_not_recorded(tmp_path, user_text, public):
    em = _em(tmp_path)
    a = _agent(em)
    t = request_id_context.set("req-owner-1")
    s = reply_surface_context.set("public" if public else "")
    try:
        asyncio.run(a._record_episode_safe(user_text, [], "answer", req_id="req-owner-1"))
    finally:
        reply_surface_context.reset(s)
        request_id_context.reset(t)
    assert em.count() == 0


def test_a_private_owner_turn_is_recorded_with_its_request_id(tmp_path):
    em = _em(tmp_path)
    a = _agent(em)
    t = request_id_context.set("req-owner-2")
    try:
        asyncio.run(a._record_episode_safe("find me a clinic near home", [], "here are three", req_id="req-owner-2"))
    finally:
        request_id_context.reset(t)
    assert _rows(em)[0][4] == "req-owner-2"


@pytest.mark.parametrize("reply", ["[TURN BUDGET EXHAUSTED] partial work, NOT a finished result",
                                   "I hit a hard limit after repeated failures"])
def test_the_agents_own_failure_is_a_failure(tmp_path, reply):
    em = _em(tmp_path)
    a = _agent(em)
    t = request_id_context.set("req-owner-3")
    try:
        asyncio.run(a._record_episode_safe("build the dashboard", [], reply, verifier_verdict="passed",
                                           req_id="req-owner-3"))
    finally:
        request_id_context.reset(t)
    assert _rows(em)[0][3] == 0


@pytest.mark.parametrize("label", ["sim", "bench"])
def test_sim_and_bench_never_teach(label):
    from ghost_agent.core.agent import turn_may_teach
    from types import SimpleNamespace
    assert turn_may_teach(SimpleNamespace(turn_origin_label=label, skill_memory=SimpleNamespace(is_read_only=False))) is False


def test_the_teach_gate_fails_closed(monkeypatch):
    import ghost_agent.core.agent as A
    monkeypatch.setattr(A, "turn_origin", lambda ctx: (_ for _ in ()).throw(RuntimeError("x")))
    assert A.turn_may_teach(object()) is False


# ── forgetting ───────────────────────────────────────────────────────────────
def test_forget_finds_a_name_inside_a_tool_result(tmp_path):
    """Fails where a recall RESULT naming Fotini survived "forget Fotini"."""
    em = _em(tmp_path)
    em.record_episode("who is my sister?", actions=[{"tool": "recall", "result": "Your sister is Fotini, lives in Patras"}],
                      outcome="She lives in Patras.", success=True)
    assert em.count_mentions("fotini") == 1


@pytest.mark.parametrize("stored,typed", [("Η Φωτεινή έρχεται αύριο", "fotini"), ("ήρθε της Φωτεινής η μάνα", "Φωτεινή"),
                                          ("Fotini said hi", "Φωτεινή")])
def test_forget_crosses_scripts_and_greek_inflection(tmp_path, stored, typed):
    em = _em(tmp_path)
    em.record_episode(stored, outcome="ok", success=True)
    assert em.count_mentions(typed) == 1


def test_a_short_name_does_not_swallow_longer_words(tmp_path):
    em = _em(tmp_path)
    em.record_episode("annabel and annette went out", outcome="ok", success=True)
    assert em.count_mentions("ann") == 0


def test_a_twin_that_could_not_be_deleted_is_reported(tmp_path):
    em = _em(tmp_path)
    eid = em.record_episode("dinner with fotini on friday", outcome="ok", success=True)
    vm = MagicMock()
    vm.collection.delete.side_effect = RuntimeError("chroma down")
    assert em.delete_episodes([eid], vm, reason="forget fotini") == 1
    assert em.last_twin_failures == [eid]
    from ghost_agent.tools import memory as M
    out = M._execute_item({"kind": "episode", "ref": {"id": eid}, "label": "episode #1"}, vm, None, None, None,
                          _em2 := _em(tmp_path))
    assert "already gone" in out or "could not be removed" in out


def test_the_forgotten_archive_says_what_and_when_and_expires(tmp_path):
    em = _em(tmp_path)
    eid = em.record_episode("dinner with fotini", outcome="ok", success=True)
    em.delete_episodes([eid], reason="forget fotini")
    row = json.loads((tmp_path / "episodes_forgotten.jsonl").read_text().splitlines()[0])
    assert row["forgot"] == "forget fotini" and row["forgot_at"] > 0
    row["forgot_at"] = time.time() - 40 * 86400
    (tmp_path / "episodes_forgotten.jsonl").write_text(json.dumps(row) + "\n")
    # the NEXT deletion purges what is past the retention
    eid2 = em.record_episode("lunch with nikos", outcome="ok", success=True)
    em.delete_episodes([eid2], reason="forget nikos")
    left = [json.loads(x)["forgot"] for x in (tmp_path / "episodes_forgotten.jsonl").read_text().splitlines()]
    assert left == ["forget nikos"]


def test_reset_all_erases_the_episodes_and_says_so(tmp_path):
    """Fails where reset_all left every episode and the boot re-indexed them."""
    from ghost_agent.tools import memory as M
    em = _em(tmp_path)
    em.record_episode("plan my wife's birthday party", outcome="ok", success=True)
    vec = MagicMock()
    vec.collection.count.return_value = 1
    vec.collection.get.return_value = {"ids": ["a"], "metadatas": [{}]}
    vec.library_file = None
    t = request_id_context.set("req-preview")
    try:
        prev = asyncio.run(M.tool_knowledge_base(action="reset_all", memory_system=vec, graph_memory=MagicMock(),
                                                 episodic_memory=em))
    finally:
        request_id_context.reset(t)
    assert "episodes (1)" in prev
    tok = prev.split("confirm='")[1].split("'")[0]
    t = request_id_context.set("req-next")
    try:
        asyncio.run(M.tool_knowledge_base(action="reset_all", memory_system=vec, graph_memory=MagicMock(),
                                          episodic_memory=em, confirm=tok))
    finally:
        request_id_context.reset(t)
    assert em.count() == 0
