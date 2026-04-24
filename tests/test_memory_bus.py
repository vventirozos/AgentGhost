"""Unit tests for the Cognitive Event Bus (MemoryBus)."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core.bus import MemoryBus


# =================================================================== fixtures


@pytest.fixture
def mocks():
    vec = MagicMock()
    vec.search = MagicMock(return_value="Doc one\n---\nDoc two")
    vec.add = MagicMock()

    graph = MagicMock()
    graph.get_neighborhood = MagicMock(return_value=[
        "- (User) -[OWNS]-> (Dog)",
        "- (Dog) -[NAMED]-> (Max)",
    ])
    graph.add_triplets = MagicMock(return_value=2)

    skill = MagicMock()
    skill.get_playbook_context = MagicMock(return_value="## RECENT LESSONS\n1. Don't recurse forever.")
    skill.learn_lesson = MagicMock()

    profile = MagicMock()
    profile.update = MagicMock()

    return {"vector": vec, "graph": graph, "skill": skill, "profile": profile}


@pytest.fixture
def bus(mocks):
    return MemoryBus(
        vector_memory=mocks["vector"],
        graph_memory=mocks["graph"],
        skill_memory=mocks["skill"],
        profile_memory=mocks["profile"],
    )


# ============================================================== RRF math unit


def test_rrf_basic_fusion():
    a = [{"source": "vector", "text": "X"}, {"source": "vector", "text": "Y"}]
    b = [{"source": "graph", "text": "Z"}, {"source": "graph", "text": "X"}]
    fused = MemoryBus._reciprocal_rank_fusion([a, b], k=60)
    # Five distinct (source, text) pairs total.
    assert len(fused) == 4
    # Each entry must have a positive RRF score.
    assert all(score > 0 for _, score in fused)
    # Ordering must be score-desc and stable.
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_top_rank_outscores_lower():
    a = [{"source": "vector", "text": f"chunk_{i}"} for i in range(5)]
    fused = MemoryBus._reciprocal_rank_fusion([a], k=60)
    # First chunk has the lowest rank index -> highest RRF score.
    assert fused[0][0]["text"] == "chunk_0"
    assert fused[-1][0]["text"] == "chunk_4"


def test_rrf_handles_empty_input():
    assert MemoryBus._reciprocal_rank_fusion([]) == []
    assert MemoryBus._reciprocal_rank_fusion([[], []]) == []


def test_rrf_dedupes_across_rankers():
    a = [{"source": "vector", "text": "shared"}]
    b = [{"source": "vector", "text": "shared"}]
    fused = MemoryBus._reciprocal_rank_fusion([a, b], k=60)
    # Same (source, text) → single fused entry whose score is the SUM
    # of both rankers' contributions. With intent-weighted RRF, the
    # default intent ("contextual") gives vector a weight of 1.5.
    assert len(fused) == 1
    vector_weight = MemoryBus._INTENT_WEIGHTS["contextual"]["vector"]
    assert fused[0][1] == pytest.approx(2 * (vector_weight / 61))


# =========================================================== query expansion


def test_extract_query_terms_filters_stopwords_and_short_tokens():
    words = MemoryBus._extract_query_terms("Tell me about the (Neo's) link to the Matrix!")
    assert "neo's" in words
    assert "matrix" in words
    assert "tell" in words           # 4 chars, kept
    assert "user" in words           # always appended
    assert "the" not in words        # 3 chars, dropped
    assert "about" not in words      # stopword


def test_extract_query_terms_caps_and_appends_user():
    long_query = " ".join([f"word{i}" for i in range(40)])
    words = MemoryBus._extract_query_terms(long_query)
    # 25 cap + the appended 'user' sentinel = 26 max.
    assert len(words) == 26
    assert words[-1] == "user"


# =========================================================== hydrate_context


@pytest.mark.asyncio
async def test_hydrate_context_fans_out_in_parallel(bus, mocks):
    """All three subsystems must be queried for a single hydrate call."""
    out = await bus.hydrate_context("tell me about my dog")
    mocks["vector"].search.assert_called_once_with("tell me about my dog")
    mocks["graph"].get_neighborhood.assert_called_once()
    mocks["skill"].get_playbook_context.assert_called_once()
    # Combined Markdown contains markers from each section.
    assert "TOPOLOGICAL KNOWLEDGE GRAPH" in out
    assert "MEMORY CONTEXT" in out
    assert "SKILL PLAYBOOK" in out
    assert "Doc one" in out and "Doc two" in out
    assert "(User) -[OWNS]-> (Dog)" in out


@pytest.mark.asyncio
async def test_hydrate_context_actually_concurrent(bus, mocks):
    """hydrate_context must use asyncio.gather, not sequential awaits."""
    timings = {"started": 0, "finished": 0}

    real_to_thread = asyncio.to_thread

    async def slow_to_thread(func, *args, **kwargs):
        timings["started"] += 1
        await asyncio.sleep(0.05)
        result = func(*args, **kwargs)
        timings["finished"] += 1
        return result

    with patch("ghost_agent.core.bus.asyncio.to_thread", side_effect=slow_to_thread):
        import time
        t0 = time.monotonic()
        await bus.hydrate_context("anything")
        elapsed = time.monotonic() - t0

    # Three 50ms sleeps in series would be >=150ms; concurrent ≈50ms.
    assert elapsed < 0.13, f"hydrate_context appears sequential ({elapsed:.3f}s)"
    assert timings["started"] == 3
    assert timings["finished"] == 3


@pytest.mark.asyncio
async def test_hydrate_context_empty_query_short_circuits(bus, mocks):
    out = await bus.hydrate_context("")
    assert out == ""
    mocks["vector"].search.assert_not_called()
    mocks["graph"].get_neighborhood.assert_not_called()
    mocks["skill"].get_playbook_context.assert_not_called()


@pytest.mark.asyncio
async def test_hydrate_context_swallows_subsystem_errors(mocks):
    mocks["vector"].search = MagicMock(side_effect=RuntimeError("vector down"))
    bus = MemoryBus(
        vector_memory=mocks["vector"],
        graph_memory=mocks["graph"],
        skill_memory=mocks["skill"],
    )
    out = await bus.hydrate_context("hello world")
    # Vector failure must not abort the whole hydrate; graph + skill still appear.
    assert "TOPOLOGICAL KNOWLEDGE GRAPH" in out
    assert "MEMORY CONTEXT" not in out
    assert "SKILL PLAYBOOK" in out


@pytest.mark.asyncio
async def test_hydrate_context_truncates_to_char_budget(bus, mocks):
    big = "x" * 200_000
    mocks["vector"].search = MagicMock(return_value=big)
    out = await bus.hydrate_context("query", max_chars=500)
    # Either the per-section budget or the global truncation marker fires;
    # both are valid evidence the formatter refused to dump 200 KB.
    assert ("[... TRUNCATED]" in out) or ("truncated for budget" in out) or ("[...]" in out)
    assert len(out) < 1500


@pytest.mark.asyncio
async def test_hydrate_context_skips_skill_no_lessons(bus, mocks):
    mocks["skill"].get_playbook_context = MagicMock(return_value="No lessons learned yet.")
    out = await bus.hydrate_context("anything")
    assert "SKILL PLAYBOOK" not in out


@pytest.mark.asyncio
async def test_hydrate_context_handles_missing_subsystems():
    bus = MemoryBus()  # no stores at all
    out = await bus.hydrate_context("hello")
    assert out == ""


# =============================================================== publish_fact


@pytest.mark.asyncio
async def test_publish_fact_routes_text_to_vector(bus, mocks):
    rep = await bus.publish_fact("insert_fact", {"text": "the sky is blue"})
    mocks["vector"].add.assert_called_once()
    args, _ = mocks["vector"].add.call_args
    assert args[0] == "the sky is blue"
    assert rep["vector"] == "ok"
    assert rep["graph"] == "skip"


@pytest.mark.asyncio
async def test_publish_fact_routes_triplets_to_graph(bus, mocks):
    triplets = [{"subject": "user", "predicate": "OWNS", "object": "dog"}]
    rep = await bus.publish_fact("insert_fact", {"text": "I own a dog", "triplets": triplets})
    mocks["graph"].add_triplets.assert_called_once_with(triplets)
    assert rep["graph"].startswith("ok")
    assert rep["vector"] == "ok"


@pytest.mark.asyncio
async def test_publish_fact_routes_profile_update(bus, mocks):
    rep = await bus.publish_fact("update_profile", {
        "profile_update": {"category": "preferences", "key": "color", "value": "blue"},
    })
    mocks["profile"].update.assert_called_once_with("preferences", "color", "blue")
    assert rep["profile"] == "ok"


@pytest.mark.asyncio
async def test_publish_fact_routes_skill_lesson(bus, mocks):
    rep = await bus.publish_fact("learn_skill", {
        "skill": {"task": "T", "mistake": "M", "solution": "S"},
    })
    mocks["skill"].learn_lesson.assert_called_once_with(
        "T", "M", "S", memory_system=mocks["vector"]
    )
    assert rep["skill"] == "ok"


@pytest.mark.asyncio
async def test_publish_fact_concurrent_fanout(bus, mocks):
    """All four subsystems must run inside one asyncio.gather."""
    started = []

    real_to_thread = asyncio.to_thread

    async def tracking(func, *args, **kwargs):
        started.append(func)
        await asyncio.sleep(0.03)
        return func(*args, **kwargs)

    with patch("ghost_agent.core.bus.asyncio.to_thread", side_effect=tracking):
        import time
        t0 = time.monotonic()
        await bus.publish_fact("compound", {
            "text": "fact",
            "triplets": [{"subject": "a", "predicate": "B", "object": "c"}],
            "profile_update": {"category": "x", "key": "y", "value": "z"},
            "skill": {"task": "t", "mistake": "m", "solution": "s"},
        })
        elapsed = time.monotonic() - t0
    assert len(started) == 4
    assert elapsed < 0.10  # Sequential would be ≈120ms.


@pytest.mark.asyncio
async def test_publish_fact_isolates_subsystem_failures(mocks):
    mocks["graph"].add_triplets = MagicMock(side_effect=RuntimeError("graph down"))
    bus = MemoryBus(
        vector_memory=mocks["vector"],
        graph_memory=mocks["graph"],
        profile_memory=mocks["profile"],
    )
    rep = await bus.publish_fact("insert_fact", {
        "text": "still works",
        "triplets": [{"subject": "a", "predicate": "B", "object": "c"}],
    })
    assert rep["vector"] == "ok"
    assert rep["graph"].startswith("error")
    mocks["vector"].add.assert_called_once()
