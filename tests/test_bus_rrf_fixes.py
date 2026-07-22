"""Regression tests for the 2026-07-22 bus + RRF-weight fixes.

Covers the eight findings verified against the live production RRF ledger
(`$GHOST_HOME/system/rrf/`). Companion to tests/test_memory_bus.py and
tests/test_bus_per_item_fusion.py (which must stay green); reuses the same
MagicMock-store harness style.

Findings, most-severe first:
  1 CRIT — the fit no longer collapses realistic (~0.14) used-rates toward
           WEIGHT_MIN; it degrades toward the hand-set priors and NEVER
           inverts them.
  2 HIGH — per-TURN normalisation removes the items-per-turn confound so a
           verbose tier is not mistaken for a useless one.
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
import json
import logging

import pytest
from unittest.mock import MagicMock

from ghost_agent.core.bus import MemoryBus
from ghost_agent.core.rrf_weights import (
    DEFAULT_INTENT_WEIGHTS,
    WEIGHT_MIN,
    WEIGHT_MAX,
    MAX_DEVIATION,
    fit_intent_weights,
    _per_turn_lift,
)


# =====================================================================
# Live-shaped per-cell counts, taken from the production observations
# ledger tail (1506 obs, overall used-rate 0.1428) on 2026-07-22.
# (intent, source): (n, used)
# =====================================================================
LIVE_CELLS = {
    ("contextual", "vector"):   (501, 47),   # rate 0.094
    ("contextual", "episodic"): (300, 42),   # 0.140
    ("contextual", "session"):  (224, 65),   # 0.290
    ("contextual", "graph"):    (81, 9),     # 0.111
    ("procedural", "skill"):    (72, 4),     # 0.056
    ("factual", "vector"):      (55, 7),     # 0.127
    ("procedural", "episodic"): (54, 4),     # 0.074
    ("procedural", "graph"):    (37, 5),     # 0.135
    ("procedural", "session"):  (37, 15),    # 0.405
    ("procedural", "vector"):   (35, 4),     # 0.114
    ("contextual", "skill"):    (30, 1),     # 0.033
    ("factual", "session"):     (28, 7),     # 0.250
    ("factual", "graph"):       (24, 0),     # 0.000  <- the collapse cell
    ("factual", "skill"):       (17, 2),     # 0.118
    ("factual", "episodic"):    (11, 3),     # 0.273
}


def _live_shaped_obs(with_turns=False):
    """Flat (intent, source, success[, turn]) list matching LIVE_CELLS."""
    obs = []
    turn = 0
    for (intent, source), (n, used) in LIVE_CELLS.items():
        for i in range(n):
            success = i < used
            if with_turns:
                # Spread each cell across turns; the exact grouping doesn't
                # matter for these assertions, only that turn ids exist.
                obs.append((intent, source, success, f"t{turn}"))
                turn += 1
            else:
                obs.append((intent, source, success))
    return obs


# =====================================================================
# 1 CRIT — calibration degrades toward priors, never inverts
# =====================================================================

def test_live_rates_do_not_collapse_toward_min():
    """The exact bug: on the live ~0.14 used-rate every well-sampled cell
    was crushed toward WEIGHT_MIN. It must now stay near its prior."""
    fitted = fit_intent_weights(_live_shaped_obs(), min_obs_per_cell=20)
    # factual/graph: n=24 used=0 → live weights.json had 0.1 (a 6.6x
    # inversion of its 2.0 prior). It must stay the dominant factual tier.
    fg = fitted["factual"]["graph"]
    assert fg > 1.5, f"factual/graph collapsed to {fg}"
    assert fg > fitted["factual"]["vector"], "graph must still beat vector (factual)"
    # procedural/skill: n=72 used=4 → live had 0.311 (BELOW session's 0.424).
    ps = fitted["procedural"]["skill"]
    assert ps > 1.4, f"procedural/skill collapsed to {ps}"
    assert ps > fitted["procedural"]["session"], "skill must still beat session (procedural)"


def test_no_gross_prior_inversion_on_live_data():
    """The dominant hand-set tiers stay dominant, and no LOW-prior tier
    leapfrogs to the top. (A mild reordering of two near-equal priors —
    contextual vector 1.5 vs session 1.2 — from real evidence is intended
    learning, NOT the 2.0→0.1 collapse finding 1 warns about.)"""
    fitted = fit_intent_weights(_live_shaped_obs(), min_obs_per_cell=20)
    # The two deliberate 2.0 priors must remain their intent's top tier.
    assert max(fitted["factual"], key=fitted["factual"].get) == "graph"
    assert max(fitted["procedural"], key=fitted["procedural"].get) == "skill"
    # contextual's winner must be one of its two HIGHEST-prior tiers, never a
    # 1.0-prior also-ran vaulting to the front.
    ctx_top = max(fitted["contextual"], key=fitted["contextual"].get)
    assert ctx_top in {"vector", "session"}, f"weak tier {ctx_top} took the top"


def test_no_cell_deviates_beyond_the_clamp():
    """A learned cell may re-order tiers mildly, never explode past the
    MAX_DEVIATION band around its prior."""
    fitted = fit_intent_weights(_live_shaped_obs(), min_obs_per_cell=20)
    for intent, prior in DEFAULT_INTENT_WEIGHTS.items():
        for source, pw in prior.items():
            w = fitted[intent][source]
            lo = max(WEIGHT_MIN, pw / MAX_DEVIATION)
            hi = min(WEIGHT_MAX, pw * MAX_DEVIATION)
            assert lo - 1e-6 <= w <= hi + 1e-6, (
                f"{intent}/{source}={w} outside [{lo},{hi}]")


def test_coinflip_reproduces_the_prior():
    """A cell used at exactly the corpus base rate earns lift 1 → prior."""
    # Two cells so the corpus base rate is 0.5, and graph sits on it.
    obs = ([("factual", "graph", i % 2 == 0) for i in range(60)]
           + [("factual", "vector", i % 2 == 0) for i in range(60)])
    out = fit_intent_weights(obs, min_obs_per_cell=20)
    assert abs(out["factual"]["graph"] - 2.0) < 0.15
    assert abs(out["factual"]["vector"] - 1.0) < 0.15


def test_thin_sample_keeps_prior_untouched():
    """Below the per-cell floor the prior is returned verbatim."""
    obs = [("factual", "graph", False)] * 5
    out = fit_intent_weights(obs, min_obs_per_cell=20)
    assert out["factual"]["graph"] == DEFAULT_INTENT_WEIGHTS["factual"]["graph"]


def test_degenerate_sample_is_refused():
    """All-failure / all-success carry no cross-tier contrast → priors stand
    (this is the anti-collapse guard, not the old 'rate 0 → MIN')."""
    all_fail = fit_intent_weights([("factual", "graph", False)] * 40,
                                  min_obs_per_cell=20)
    assert all_fail["factual"]["graph"] == DEFAULT_INTENT_WEIGHTS["factual"]["graph"]
    all_win = fit_intent_weights([("factual", "graph", True)] * 40,
                                 min_obs_per_cell=20)
    assert all_win["factual"]["graph"] == DEFAULT_INTENT_WEIGHTS["factual"]["graph"]


def test_fit_still_returns_full_matrix_and_is_json_safe():
    fitted = fit_intent_weights(_live_shaped_obs(with_turns=True))
    assert set(fitted) == set(DEFAULT_INTENT_WEIGHTS)
    for sw in fitted.values():
        assert set(sw) >= set(DEFAULT_INTENT_WEIGHTS["contextual"])
    json.dumps(fitted)  # serialisable for save_intent_weights


# =====================================================================
# 2 HIGH — per-turn normalisation removes the items/turn confound
# =====================================================================

def test_per_turn_lift_rewards_used_share_over_footprint():
    """A lean tier that earns the used set each turn lifts >1; a verbose
    tier that is most of the injections but earns nothing lifts <1."""
    rows = []
    for t in range(50):
        tid = f"t{t}"
        for _ in range(6):
            rows.append(("contextual", "vector", False, tid))   # verbose, never used
        rows.append(("contextual", "session", True, tid))        # lean, always used
    lift = _per_turn_lift(rows)
    assert lift[("contextual", "session")][0] > 1.0
    assert lift[("contextual", "vector")][0] < 1.0


def test_per_turn_absent_ids_fall_back_to_pooled():
    """A row without a turn id disables the turn estimator (mixed ledger)."""
    assert _per_turn_lift([("contextual", "vector", True, None)]) is None


def test_verbose_useful_tier_not_penalised_for_verbosity():
    """The finding-2 scenario end to end: vector is the SOLE useful tier but
    injects many items/turn (per-item rate capped ~0.36). Turn-normalised, it
    is up-weighted; the never-used lean tier is down-weighted."""
    obs = []
    for t in range(50):
        tid = f"t{t}"
        for j in range(6):
            obs.append((("contextual"), "vector", j == 0, tid))  # 1 of 6 used
        obs.append(("contextual", "graph", False, tid))
    fitted = fit_intent_weights(obs, min_obs_per_cell=20)
    assert fitted["contextual"]["vector"] > DEFAULT_INTENT_WEIGHTS["contextual"]["vector"]
    assert fitted["contextual"]["graph"] < DEFAULT_INTENT_WEIGHTS["contextual"]["graph"]


# =====================================================================
# 3 HIGH — budget loop skips oversized items and keeps filling
# =====================================================================

def test_budget_loop_skips_oversized_and_keeps_filling():
    """A big vector item at rank 2 must NOT discard the small skill + graph
    items ranked below it — nor drop their lazily-emitted headers."""
    headers_cost = (len("### MEMORY CONTEXT:") + 1
                    + len("### SKILL PLAYBOOK (lessons from prior runs — "
                          "follow to avoid repeats):") + 1
                    + len("### TOPOLOGICAL KNOWLEDGE GRAPH:") + 1)
    small_skill = "do X"      # tiny
    small_graph = "- A->B"    # tiny
    fused = [
        ({"source": "vector", "text": "seed", "mem_id": "v0"}, 0.99),
        ({"source": "vector", "text": "B" * 900, "mem_id": "v1"}, 0.90),  # oversized
        ({"source": "skill", "text": small_skill, "trigger": "t"}, 0.80),
        ({"source": "graph", "text": small_graph}, 0.70),
    ]
    budget = (headers_cost + len("seed") + 1
              + len(small_skill) + 1 + len(small_graph) + 1 + 5)
    out, survivors = MemoryBus._format_markdown_with_survivors(fused, max_chars=budget)
    # The oversized item is skipped, everything else survives — headers too.
    assert "B" * 900 not in out
    assert small_skill in out and small_graph in out
    assert "SKILL PLAYBOOK" in out
    assert "TOPOLOGICAL KNOWLEDGE GRAPH" in out
    texts = [s["text"] for s in survivors]
    assert small_skill in texts and small_graph in texts
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
    vec = MagicMock()
    vec.search_items = MagicMock(return_value=[
        {"id": "g1", "text": "graph fact", "score": 0.1}])
    graph = MagicMock()
    graph.get_neighborhood = MagicMock(return_value=["- (User)-[IS]->(Vasilis)"])
    bus = MemoryBus(vector_memory=vec, graph_memory=graph)
    expanded = ("Context: FIX the ERROR DEBUG the STEPS workflow procedure "
                "| User intent: who is the user")
    # If classified on the expanded string → procedural (graph weight 0.5).
    # With raw_user_text "who is the user" → factual (graph weight 2.0), so
    # the graph edge must outrank / appear.
    out = await bus.hydrate_context(expanded, raw_user_text="who is the user")
    assert "TOPOLOGICAL KNOWLEDGE GRAPH" in out


# =====================================================================
# 5 MED — caller budget respected, scaling reachable
# =====================================================================

@pytest.mark.asyncio
async def test_caller_budget_respected_for_simple_query():
    """A simple query gets exactly the caller's budget, not a 6000 floor."""
    vec = MagicMock()
    vec.search_items = MagicMock(return_value=[
        {"id": f"id-{i}", "text": "x" * 300, "score": 0.1} for i in range(20)])
    bus = MemoryBus(vector_memory=vec)
    out = await bus.hydrate_context("short simple query",
                                    context_budget=1000)
    # Old code floored at 6000; the fix honours 1000. Allow the truncation
    # tail; the body must be far under 6000.
    assert len(out) < 2000, f"budget not respected: {len(out)} chars"


@pytest.mark.asyncio
async def test_complex_query_scales_above_base_budget():
    """A >30-word query reaches up to 3x the base budget (capped 12000)."""
    big = "y" * 40000
    vec = MagicMock()
    vec.search_items = MagicMock(return_value=[
        {"id": "big", "text": big, "score": 0.1}])
    bus = MemoryBus(vector_memory=vec)
    complex_q = " ".join(["word"] * 40)
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

@pytest.mark.asyncio
async def test_gate_passed_to_modern_search_items():
    """A search_items that accepts the kwarg is called WITH the gate."""
    vec = MagicMock()

    def search_items(query, inject_identity=True, min_relevance_dist=None):
        search_items.seen = min_relevance_dist
        return [{"id": "a", "text": "hit", "score": 0.1}]
    vec.search_items = search_items
    bus = MemoryBus(vector_memory=vec)
    await bus._fetch_vector("query")
    assert search_items.seen == MemoryBus._VECTOR_MATCH_FLOOR


@pytest.mark.asyncio
async def test_gate_dropped_only_for_arity_and_is_logged(caplog):
    """A legacy stub whose search_items lacks the kwarg is called
    positionally AND the dropped gate is logged."""
    vec = MagicMock()

    def legacy(query):  # no min_relevance_dist
        return [{"id": "a", "text": "hit", "score": 0.1}]
    vec.search_items = legacy
    bus = MemoryBus(vector_memory=vec)
    with caplog.at_level(logging.WARNING, logger="GhostAgent"):
        items = await bus._fetch_vector("query")
    assert items and items[0]["text"] == "hit"
    assert any("gate DROPPED" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_internal_typeerror_is_not_silently_retried():
    """A TypeError raised INSIDE a gate-accepting search_items must surface
    as a swallowed fetch error (empty list), NOT a silent gate-off retry."""
    calls = []

    def search_items(query, min_relevance_dist=None):
        calls.append(min_relevance_dist)
        raise TypeError("boom inside search")
    vec = MagicMock()
    vec.search_items = search_items
    bus = MemoryBus(vector_memory=vec)
    items = await bus._fetch_vector("query")
    assert items == []
    # Called exactly once (with the gate) — no positional gate-off retry.
    assert calls == [MemoryBus._VECTOR_MATCH_FLOOR]


def test_accepts_relevance_floor_detection():
    assert MemoryBus._accepts_relevance_floor(lambda q, min_relevance_dist=None: None)
    assert MemoryBus._accepts_relevance_floor(lambda q, **kw: None)
    assert not MemoryBus._accepts_relevance_floor(lambda q: None)


# =====================================================================
# 8 IMPROVEMENT — per-hydration instrumentation line
# =====================================================================

@pytest.mark.asyncio
async def test_hydration_emits_instrumentation_line(caplog):
    vec = MagicMock()
    vec.search_items = MagicMock(return_value=[
        {"id": "v1", "text": "vec doc", "score": 0.1}])
    graph = MagicMock()
    graph.get_neighborhood = MagicMock(return_value=["- (A)-[R]->(B)"])
    # skill wired but empty; episodic/session unwired (None).
    skill = MagicMock()
    skill.get_playbook_items = MagicMock(return_value=[])
    bus = MemoryBus(vector_memory=vec, graph_memory=graph, skill_memory=skill)
    with caplog.at_level(logging.INFO, logger="GhostAgent"):
        await bus.hydrate_context("tell me about the thing")
    line = next((r.message for r in caplog.records
                 if "Hydration tiers" in r.message), None)
    assert line is not None, "no instrumentation line emitted"
    # vector produced 1 candidate, skill 0 (wired-but-empty), episodic
    # unwired → '-'. All five tier tags present, plus survivors + intent.
    assert "v=1" in line
    assert "s=0" in line          # skill wired but empty
    assert "e=-" in line          # episodic unwired
    assert "sess=-" in line       # session unwired
    assert "survivors" in line and "intent=" in line


# =====================================================================
# Sub-query validation + dedup; source-required RRF key
# =====================================================================

def test_extend_sub_queries_filters_and_dedups():
    subs = ["original query"]
    MemoryBus._extend_sub_queries(subs, [
        "original query",            # echo of the original → dropped
        "a valid distinct subquery",  # kept
        "",                          # empty → dropped
        "ok",                        # < 3 chars → dropped
        None,                        # non-string → dropped
        {"not": "a string"},         # non-string → dropped
        "a valid distinct subquery",  # duplicate → dropped
        "another good one",          # kept
    ])
    assert subs == ["original query", "a valid distinct subquery",
                    "another good one"]


def test_extend_sub_queries_respects_limit():
    subs = ["q0"]
    MemoryBus._extend_sub_queries(subs, [f"candidate number {i}" for i in range(10)],
                                  limit=3)
    assert len(subs) == 4  # original + 3


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
    a = [{"source": "vector", "text": "X"}]
    b = [{"source": "vector", "text": "X"}]
    fused = MemoryBus._reciprocal_rank_fusion([a, b])  # k defaults 60
    vw = MemoryBus._INTENT_WEIGHTS["contextual"]["vector"]
    assert fused[0][1] == pytest.approx(2 * (vw / 61))
