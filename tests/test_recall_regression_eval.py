"""Recall regression eval — golden-set hit-rate guard (2026-07-14).

Until now an embedder swap, rerank tweak, distance-gate change, or RRF
reweighting had NO regression guard except live vibes: the embedder.json
fingerprint catches a *wrong model*, nothing catches *worse retrieval*.
This harness seeds a small golden corpus across ALL five hydration tiers
(vector facts + distractors, a graph triplet, a skill lesson, an episode,
a stored session) and asserts end-to-end hit-rates through the REAL
pipeline — real BGE embeddings, real Chroma, real RRF fusion via
MemoryBus.hydrate_context.

Thresholds are set with headroom below current behavior (measured 2026-07-14:
vector paraphrase recall 100%) so the suite only fails on a real regression,
not on embedder jitter. If this file fails after a retrieval-stack change,
the change made recall WORSE — do not loosen the floor without measuring.
"""

from types import SimpleNamespace

import pytest

from ghost_agent.core.bus import MemoryBus
from ghost_agent.core.sessions import SessionStore
from ghost_agent.memory.episodes import EpisodicMemory
from ghost_agent.memory.graph import GraphMemory
from ghost_agent.memory.skills import SkillMemory
from ghost_agent.memory.vector import VectorMemory


# fact → paraphrased queries that must retrieve it
GOLDEN_VECTOR = [
    ("The user's name is Vasilis and he lives in Athens, Greece.",
     ["what is the name of the user", "where does the user live"]),
    ("The user's favorite chess opening is the Alekhine defense.",
     ["which chess opening does the user prefer"]),
    ("The ghost agent runs on port 8000 under a launchd supervisor.",
     ["what port does the ghost agent listen on"]),
    ("Backups of the memory store run every friday night.",
     ["when do the memory store backups run"]),
]

DISTRACTORS = [
    "The mitochondria is the powerhouse of the cell.",
    "Mount Olympus is the highest mountain in Greece.",
    "The 1997 chess match between Kasparov and Deep Blue ended 3.5-2.5.",
    "Docker containers share the host kernel.",
    "The speed of light is 299792458 meters per second.",
    "Starlink satellites orbit at roughly 550 kilometers.",
    "SQLite databases are single files on disk.",
    "The uConsole ships with a Raspberry Pi compute module.",
]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("recall_eval")
    for sub in ("vec", "graph", "skill", "epi"):
        (root / sub).mkdir()
    vm = VectorMemory(root / "vec", "http://mock-url")
    for fact, _ in GOLDEN_VECTOR:
        vm.add(fact, {"type": "manual"})
    for d in DISTRACTORS:
        vm.add(d, {"type": "auto"})

    gm = GraphMemory(root / "graph")
    gm.add_triplets([
        {"subject": "vasilis", "predicate": "WORKS_AT", "object": "evolmonkey"},
        {"subject": "ghost", "predicate": "RUNS_ON", "object": "port 8000"},
    ])

    sm = SkillMemory(root / "skill")
    sm.learn_lesson(
        "deploy fails with EACCES", "retrying the copy blindly",
        "When a deploy fails with EACCES, verify the publish path ownership first.",
        memory_system=vm,
    )

    epi = EpisodicMemory(root / "epi")
    epi.record_episode(
        trigger="restart of the llama server crashed with a metal assert",
        outcome="fixed by sending a single signal and waiting",
        success=True,
        lesson="never double-signal llama-server on shutdown",
        cluster_id="ops",
        vector_memory=vm,
    )

    ss = SessionStore(root / "sessions")
    ss.append_turn(
        "sessk8s",
        [{"role": "user", "content": "when shall we do the kubernetes deployment?"}],
        "We agreed the kubernetes deployment happens friday after the backup.",
    )

    bus = MemoryBus(vector_memory=vm, graph_memory=gm, skill_memory=sm,
                    episodic_memory=epi, session_store=ss)
    return SimpleNamespace(vm=vm, bus=bus)


class TestVectorMatchGate:
    """The proactive-injection vector gate (2026-07-15): a query with no
    strong match injects nothing, but the recall TOOL path (no gate) stays
    best-effort."""

    def test_off_topic_gated_to_empty(self, env):
        # No golden/distractor item is a strong match for this — the gate
        # returns [].
        assert env.vm.search_items("what is the weather like on mars today",
                                   inject_identity=False,
                                   min_relevance_dist=0.42) == []

    def test_on_topic_survives_the_gate(self, env):
        items = env.vm.search_items("which chess opening does the user prefer",
                                    inject_identity=False, min_relevance_dist=0.42)
        assert any("Alekhine" in (it.get("text") or "") for it in items)

    def test_tool_path_is_best_effort_without_gate(self, env):
        # Same off-topic query with NO gate (the recall tool path) still
        # returns its best-effort matches — an explicit ask is answered.
        assert env.vm.search_items("what is the weather like on mars today",
                                   inject_identity=False) != []


class TestVectorTier:
    def test_every_golden_fact_is_reachable(self, env):
        """Each seeded fact must be retrievable by at least one paraphrase
        through the raw vector tier (embeddings + tiers + rerank)."""
        for fact, queries in GOLDEN_VECTOR:
            found = False
            for q in queries:
                items = env.vm.search_items(q, inject_identity=False)
                if any(fact in (it.get("text") or "") for it in items):
                    found = True
                    break
            assert found, f"vector tier lost golden fact: {fact!r}"

    def test_paraphrase_hit_rate_floor(self, env):
        """Aggregate per-query hit rate with headroom (measured 1.0)."""
        total = hits = 0
        for fact, queries in GOLDEN_VECTOR:
            for q in queries:
                total += 1
                items = env.vm.search_items(q, inject_identity=False)
                if any(fact in (it.get("text") or "") for it in items):
                    hits += 1
        assert hits / total >= 0.75, f"vector paraphrase hit-rate regressed: {hits}/{total}"


class TestHydrationEndToEnd:
    async def test_hydration_hit_rate_floor(self, env):
        """Golden facts must survive fusion + budget + relevance gates."""
        total = hits = 0
        for fact, queries in GOLDEN_VECTOR:
            for q in queries:
                total += 1
                out = await env.bus.hydrate_context(q)
                if fact in out:
                    hits += 1
        assert hits / total >= 0.75, f"hydration hit-rate regressed: {hits}/{total}"

    async def test_graph_tier_surfaces_triplet(self, env):
        out = await env.bus.hydrate_context("who does vasilis work for")
        assert "evolmonkey" in out.lower()

    async def test_skill_tier_surfaces_lesson(self, env):
        out = await env.bus.hydrate_context("how do I fix a deploy that fails with EACCES")
        assert "publish path" in out

    async def test_session_tier_surfaces_conversation(self, env):
        out = await env.bus.hydrate_context("when is the kubernetes deployment happening")
        assert "friday" in out.lower()

    async def test_episodic_tier_surfaces_episode(self, env):
        out = await env.bus.hydrate_context(
            "the llama server crashed with a metal assert during restart")
        assert "llama" in out.lower()

    async def test_off_topic_query_does_not_surface_golden_facts(self, env):
        """The relevance gates must keep unrelated memories out of the
        prompt — injection of everything would also 'pass' hit-rate tests.

        FIXED 2026-07-15 (was xfail): the fix is NOT the RRF _RELEVANCE_FLOOR
        (measured incapable — RRF scores discard embedding distance), but the
        vector best-match gate `_VECTOR_MATCH_FLOOR`: an off-topic query whose
        closest vector item is beyond ~0.42 injects no vector context. Several
        off-topic queries checked so the gate isn't overfit to one phrasing."""
        for q in ("what is the weather like on mars today",
                  "how do quantum computers factor integers",
                  "recipe for chocolate chip cookies"):
            out = await env.bus.hydrate_context(q)
            assert "Alekhine" not in out, f"off-topic {q!r} surfaced a golden fact"
            assert "EACCES" not in out, f"off-topic {q!r} surfaced a golden fact"
