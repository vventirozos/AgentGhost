"""Regression tests for the 2026-07-22 bus + RRF-weight fixes.

Covers the eight findings verified against the live production RRF ledger
(`$GHOST_HOME/system/rrf/`). Companion to tests/test_memory_bus.py and
tests/test_bus_per_item_fusion.py (which must stay green); reuses the same
MagicMock-store harness style.

Findings, most-severe first (1 and 2 pinned the weight FIT, which went with
the usefulness ledger in §4NM — rrf_weights.py is loader-only now):
  3 HIGH — the budget emission loop skips an oversized item and keeps
           filling, still emitting later tiers' lazy headers.
  4 HIGH — intent is classified on the raw user text when supplied, not the
           expanded/anaphora-resolved query.
  5 MED  — the caller's context_budget is respected and the documented
           up-scaling is reachable.
  6 MED  — consensus is normalised by tier redundancy so a keyword-stable
           tier (graph) can't monopolise the fused head.
  7 MED  — the off-topic vector gate is dropped only for a genuine arity
           mismatch, and that drop is logged.
  8 IMPR — one per-hydration instrumentation line (per-tier counts).
Plus: sub-query validation/dedup, and the source-required RRF key.
"""
import asyncio
import logging

import pytest
from unittest.mock import MagicMock

from ghost_agent.core.bus import MemoryBus


# =====================================================================
# 3 HIGH — budget loop skips oversized items and keeps filling
# =====================================================================

def test_budget_loop_skips_oversized_and_keeps_filling():
    """A big vector item at rank 2 must NOT discard the small
    items ranked below it — nor drop their lazily-emitted headers."""
    headers_cost = (len("### MEMORY CONTEXT:") + 1
                    + len("### TOPOLOGICAL KNOWLEDGE GRAPH:") + 1)
    small_vector = "do X"      # tiny
    small_graph = "- A->B"    # tiny
    fused = [
        ({"source": "vector", "text": "seed", "mem_id": "v0"}, 0.99),
        ({"source": "vector", "text": "B" * 900, "mem_id": "v1"}, 0.90),  # oversized
        ({"source": "vector", "text": small_vector, "mem_id": "v2"}, 0.80),
        ({"source": "graph", "text": small_graph}, 0.70),
    ]
    budget = (headers_cost + len("seed") + 1
              + len(small_vector) + 1 + len(small_graph) + 1 + 5)
    out, survivors = MemoryBus._format_markdown_with_survivors(fused, max_chars=budget)
    # The oversized item is skipped, everything else survives — headers too.
    assert "B" * 900 not in out
    assert small_vector in out and small_graph in out
    assert "TOPOLOGICAL KNOWLEDGE GRAPH" in out
    texts = [s["text"] for s in survivors]
    assert small_vector in texts and small_graph in texts
    assert "B" * 900 not in texts


def test_budget_loop_first_item_too_big_still_truncates():
    """The nothing-emitted-yet path is unchanged: truncate the top item in
    place rather than returning empty context (either truncation marker)."""
    fused = [({"source": "vector", "text": "Z" * 500, "mem_id": "a"}, 0.9)]
    out, survivors = MemoryBus._format_markdown_with_survivors(fused, max_chars=80)
    assert ("[...]" in out) or ("[... TRUNCATED]" in out)
    assert len(survivors) == 1
    assert 0 < len(out) <= 80 + len("\n\n[... TRUNCATED]\n\n")


# =====================================================================
# 4 HIGH — intent classified on raw user text when provided
# =====================================================================

def test_classify_prefers_raw_user_text():
    """"and the docs?" is contextual; classifying the whole EXPANDED query
    (the pre-fix behaviour) is procedural from the previous reply's prose."""
    expanded = ("Context: Here is how to FIX the ERROR: follow these STEPS "
                "and DEBUG the workflow | User intent: and the docs?")
    # Pre-fix: classifying the full expanded string → procedural (leaked from
    # the assistant's previous prose).
    assert MemoryBus._classify_query_intent(expanded) == "procedural"
    # Fix, explicit raw text: classify on the user's own words → contextual.
    src = MemoryBus._intent_source_text(expanded, raw_user_text="and the docs?")
    assert src == "and the docs?"
    assert MemoryBus._classify_query_intent(src) == "contextual"
    # Fix, marker fallback (no raw text): the user-intent TAIL is used, not
    # the full string → also contextual, not procedural.
    assert MemoryBus._classify_query_intent(
        MemoryBus._intent_source_text(expanded)) == "contextual"


def test_intent_source_text_falls_back_to_expansion_tail():
    """No raw text supplied → classify the user-intent tail, not the prose."""
    expanded = ("Context: FIX the ERROR DEBUG STEPS workflow "
                "| User intent: who is the user")
    src = MemoryBus._intent_source_text(expanded)
    assert src.strip() == "who is the user"
    assert MemoryBus._classify_query_intent(src) == "factual"


def test_intent_source_text_plain_query_unchanged():
    assert MemoryBus._intent_source_text("plain query") == "plain query"


@pytest.mark.asyncio
async def test_hydrate_threads_raw_user_text():
    """hydrate_context accepts raw_user_text and routes it into classification."""
    graph = MagicMock()
    graph.owner_facts_matching = MagicMock(return_value=["- (User)-[IS]->(Vasilis)"])
    bus = MemoryBus(graph_memory=graph)
    expanded = ("Context: FIX the ERROR DEBUG the STEPS workflow procedure "
                "| User intent: who am I?")
    # If classified on the expanded string → procedural (graph weight 0.5).
    # With raw_user_text "who am I?" → factual (graph weight 2.0), so the
    # owner's fact must appear (§4NM: the graph tier carries only those).
    out = await bus.hydrate_context(expanded, raw_user_text="who am I?")
    assert "TOPOLOGICAL KNOWLEDGE GRAPH" in out


# =====================================================================
# 5 MED — caller budget respected, scaling reachable
# =====================================================================

@pytest.mark.asyncio
async def test_caller_budget_respected_for_simple_query():
    """A simple query gets exactly the caller's budget, not a 6000 floor."""
    graph = MagicMock()
    graph.owner_facts_matching = MagicMock(return_value=[f"fact {i} " + "x" * 300 for i in range(20)])
    bus = MemoryBus(graph_memory=graph)
    out = await bus.hydrate_context("what is my simple question",
                                    context_budget=1000)
    assert out, "nothing hydrated — the budget assertion below would be vacuous"
    # Old code floored at 6000; the fix honours 1000. Allow the truncation
    # tail; the body must be far under 6000.
    assert len(out) < 2000, f"budget not respected: {len(out)} chars"


@pytest.mark.asyncio
async def test_complex_query_scales_above_base_budget():
    """A >30-word query reaches up to 3x the base budget (capped 12000)."""
    big = "y" * 40000
    graph = MagicMock()
    graph.owner_facts_matching = MagicMock(return_value=[big])
    bus = MemoryBus(graph_memory=graph)
    complex_q = "what is my " + " ".join(["word"] * 40)
    out = await bus.hydrate_context(complex_q, context_budget=4000)
    # 4000 base → 12000 for a complex query. Must exceed the simple-query
    # budget of 4000, proving the documented scaling is reachable.
    assert len(out) > 6000, f"complex scaling unreachable: {len(out)} chars"
    assert len(out) < 12500


# =====================================================================
# 6 MED — consensus normalised by tier redundancy
# =====================================================================

def test_consensus_normalisation_caps_stable_tier():
    """A keyword-stable tier that returns the SAME edge in every sub-query
    must not out-bank a tier that returns different items per sub-query."""
    # graph returns the identical edge in 4 sub-query lists (stable);
    # vector returns 4 different docs (one per sub-query).
    graph_lists = [[{"source": "graph", "text": "EDGE"}] for _ in range(4)]
    vector_lists = [[{"source": "vector", "text": f"doc{i}"}] for i in range(4)]
    ranked = graph_lists + vector_lists

    naive = MemoryBus._reciprocal_rank_fusion(
        ranked, k=10, intent="contextual", normalize_consensus=False)
    normed = MemoryBus._reciprocal_rank_fusion(
        ranked, k=10, intent="contextual", normalize_consensus=True)

    naive_scores = {it["text"]: s for it, s in naive}
    normed_scores = {it["text"]: s for it, s in normed}
    # Naively, the 4x-accumulated EDGE dwarfs each single doc.
    assert naive_scores["EDGE"] > naive_scores["doc0"]
    # Normalised by redundancy (graph emitted 4, distinct 1 → factor 4),
    # EDGE's score drops to a single-emission level, no longer dominating.
    assert normed_scores["EDGE"] < naive_scores["EDGE"]
    assert normed_scores["EDGE"] == pytest.approx(naive_scores["EDGE"] / 4)
    # vector docs (distinct per sub-query → factor 1) are untouched.
    assert normed_scores["doc0"] == pytest.approx(naive_scores["doc0"])


def test_consensus_normalisation_preserves_intra_tier_order():
    """Redundancy scaling divides a tier uniformly → its internal ranking
    (the genuine consensus reward) is unchanged."""
    sq1 = [{"source": "vector", "text": "CONSENSUS"},
           {"source": "vector", "text": "SINGLE"}]
    sq2 = [{"source": "vector", "text": "CONSENSUS"}]
    fused = MemoryBus._reciprocal_rank_fusion(
        [sq1, sq2], k=10, normalize_consensus=True)
    order = [it["text"] for it, _ in fused]
    assert order.index("CONSENSUS") < order.index("SINGLE")


# =====================================================================
# 7 MED — off-topic vector gate: narrow fallback + log
# =====================================================================

# =====================================================================
# 8 IMPROVEMENT — per-hydration instrumentation line
# =====================================================================

@pytest.mark.asyncio
async def test_hydration_emits_instrumentation_line(caplog):
    vec = MagicMock()
    graph = MagicMock()
    graph.owner_facts_matching = MagicMock(return_value=[])
    # episodic/session unwired (None); vector wired but retired (§4NM) —
    # both read "-".
    bus = MemoryBus(vector_memory=vec, graph_memory=graph)
    with caplog.at_level(logging.INFO, logger="GhostAgent"):
        await bus.hydrate_context("tell me about the thing")
    line = next((r.message for r in caplog.records
                 if "Hydration tiers" in r.message), None)
    assert line is not None, "no instrumentation line emitted"
    # graph 0 (no owner question), the retired vector tier and the unwired
    # ones '-'. All tags, survivors, intent.
    assert "v=-" in line          # retired, even though wired
    assert "g=0" in line
    assert " s=" not in line      # §4TD: no lesson tier
    assert "e=-" in line          # episodic unwired
    assert "sess=-" in line       # session unwired
    assert "survivors" in line and "intent=" in line


# =====================================================================
# Sub-query validation + dedup; source-required RRF key
# =====================================================================





def test_rrf_requires_source_and_does_not_raise():
    """A source-less ranked list is dropped (not positionally guessed) and
    an item missing 'source' never raises out of fusion."""
    good = [{"source": "vector", "text": "V"}]
    headless = [{"text": "no source here"}]  # first item has no source
    fused = MemoryBus._reciprocal_rank_fusion([good, headless], k=10)
    texts = [it["text"] for it, _ in fused]
    assert "V" in texts
    assert "no source here" not in texts  # dropped, not crashed


def test_rrf_backcompat_default_k_and_no_normalisation():
    """Direct classmethod callers keep k=60 and the exact legacy arithmetic
    (normalize_consensus defaults off)."""
    a = [{"source": "graph", "text": "X"}]
    b = [{"source": "graph", "text": "X"}]
    fused = MemoryBus._reciprocal_rank_fusion([a, b])  # k defaults 60
    sw = MemoryBus._INTENT_WEIGHTS["contextual"]["graph"]
    assert fused[0][1] == pytest.approx(2 * (sw / 61))
