"""Tests for the core/main wiring fixes from the deep-audit remediation pass.

Covers:
* Dead modules now wired in main.py (AdaptiveThreshold, EpisodicMemory,
  Verifier; §4NN removed ContradictionLog and UncertaintyTracker)
* tool_failure.get_fallback_hint reachable from agent error path
* llm.py circuit-breaker filtering on vision/coding/image-gen pools
* llm.py vision/coding retry loops break when all nodes exhausted

These are unit tests against pure functions / class behaviour — no
network, no Docker, no upstream LLM.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from ghost_agent.core.llm import LLMClient
from ghost_agent.core.verifier import Verifier
from ghost_agent.memory.adaptive_threshold import AdaptiveThreshold
from ghost_agent.memory.episodes import EpisodicMemory
from ghost_agent.tools.tool_failure import get_fallback_hint


# ---------------------------------------------------------------------------
# Dead-module instantiability: each one must be constructible from a
# memory_dir alone (or from no args, for stateless trackers). main.py wires
# them with these exact signatures, so a regression here breaks startup.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# main.py imports the wiring symbols at module load. Verify the import path
# is correct so a renamed module breaks loudly here, not at server start.
# ---------------------------------------------------------------------------

def test_main_imports_wire_dead_modules():
    import ghost_agent.main as main_mod
    for sym in (
        "AdaptiveThreshold",
        "EpisodicMemory",
        "Verifier",
    ):
        assert hasattr(main_mod, sym), f"main.py missing import: {sym}"


# ---------------------------------------------------------------------------
# tool_failure fallback-hint wiring: known tool/error pair returns a hint;
# unknown returns None. Agent error path imports this lazily to enrich
# tool failure context for the LLM.
# ---------------------------------------------------------------------------

def test_get_fallback_hint_known_pair_returns_hint():
    hint = get_fallback_hint("execute", "ModuleNotFoundError: No module named 'foo'")
    assert hint is not None and "pip install" in hint.lower()


def test_get_fallback_hint_unknown_tool_returns_none():
    assert get_fallback_hint("no_such_tool_xyz", "anything") is None


def test_get_fallback_hint_unknown_error_returns_none():
    assert get_fallback_hint("execute", "totally novel error string") is None


def test_get_fallback_hint_handles_empty_input():
    assert get_fallback_hint("", "err") is None
    assert get_fallback_hint("execute", "") is None
    assert get_fallback_hint("execute", None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# llm.py circuit-breaker filtering on vision/coding/image-gen pools.
# Build a real LLMClient with two nodes; trip the breaker on the first;
# verify we get the second node back, not the dead one.
# ---------------------------------------------------------------------------

def _make_llm_with_pool(pool_attr: str):
    """Construct an LLMClient with a two-node pool and return (client, urls)."""
    nodes = [
        {"url": "http://node-a", "model": "alpha"},
        {"url": "http://node-b", "model": "beta"},
    ]
    # We only care about the specific pool we patch in.
    client = LLMClient(
        upstream_url="http://upstream",
        tor_proxy=None,
        worker_nodes=[],
        visual_nodes=nodes if pool_attr == "vision_clients" else None,
        coding_nodes=nodes if pool_attr == "coding_clients" else None,
        image_gen_nodes=nodes if pool_attr == "image_gen_clients" else None,
    )
    return client, [n["url"] for n in nodes]


def test_get_vision_node_skips_tripped_breaker():
    client, urls = _make_llm_with_pool("vision_clients")
    # Trip the first node; the round-robin must skip it.
    for _ in range(20):
        client.circuit_breaker.record_failure(urls[0])
    chosen = client.get_vision_node()
    assert chosen is not None
    assert chosen["url"] != urls[0], "tripped vision node should be skipped"


def test_get_coding_node_skips_tripped_breaker():
    client, urls = _make_llm_with_pool("coding_clients")
    for _ in range(20):
        client.circuit_breaker.record_failure(urls[0])
    chosen = client.get_coding_node()
    assert chosen is not None
    assert chosen["url"] != urls[0]


def test_get_image_gen_node_skips_tripped_breaker():
    client, urls = _make_llm_with_pool("image_gen_clients")
    for _ in range(20):
        client.circuit_breaker.record_failure(urls[0])
    chosen = client.get_image_gen_node()
    assert chosen is not None
    assert chosen["url"] != urls[0]


def test_get_vision_node_targeted_match_respects_breaker():
    client, urls = _make_llm_with_pool("vision_clients")
    for _ in range(20):
        client.circuit_breaker.record_failure(urls[0])
    # Targeting "alpha" should NOT return the dead alpha node.
    chosen = client.get_vision_node(target_model="alpha")
    assert chosen is None or chosen["url"] != urls[0]


def test_get_vision_node_all_tripped_falls_back_to_first():
    client, urls = _make_llm_with_pool("vision_clients")
    for url in urls:
        for _ in range(20):
            client.circuit_breaker.record_failure(url)
    # All tripped — getter still returns *some* node (the first) so the
    # call surface stays non-null; the breaker cooldown will widen on
    # the inevitable next failure.
    chosen = client.get_vision_node()
    assert chosen is not None


# ---------------------------------------------------------------------------
# llm.py source-level guard: vision/coding retry loops MUST break when
# every node has been tried. We assert the guard text is present in the
# source rather than synthesizing a real network failure storm.
# ---------------------------------------------------------------------------


def test_a_pool_never_re_posts_to_a_node_it_already_tried():
    """⚠ REPLACES TWO VACUOUS SOURCE-TEXT TESTS.

    They asserted `"exhausted" in src.lower()` — a word that appears ONLY in
    comments — and `src.count("if node in tried_nodes:\\n") >= 2`, which
    counts 10 occurrences of which only 5 are guards (the rest are re-select
    branches). Mutation-proven: turning all five guards into `pass`, i.e.
    making them completely inoperative, left BOTH tests green (R2 lens B,
    item 4).

    The guard is load-bearing: without it an already-tried node is re-POSTed,
    burning a second full timeout and recording a second breaker failure from
    one logical call. Measure the POSTs."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from ghost_agent.core.llm import LLMClient

    calls = {"a": 0, "b": 0}

    def _node(tag, url):
        cli = MagicMock()

        async def _fail(*a, **kw):
            calls[tag] += 1
            raise RuntimeError(f"{tag} down")

        cli.post = AsyncMock(side_effect=_fail)
        return {"url": url, "model": f"m-{tag}", "client": cli, "name": tag}

    c = LLMClient("http://upstream.invalid:8080")
    c.worker_clients = [_node("a", "http://a"), _node("b", "http://b")]
    c._worker_index = 0
    # Trip both breakers so the selector's is_available() filter is exercised
    # too — this is the state in which the guard matters.
    for url in ("http://a", "http://b"):
        for _ in range(5):
            c.circuit_breaker.record_failure(url)

    try:
        asyncio.run(c._do_chat_completion(
            {"model": "m", "messages": []}, use_worker=True,
            off_main_only=True, timeout=5.0))
    except Exception:
        pass

    # ⚠ PER-NODE, not the total. `total <= len(pool)` is satisfied by
    # hammering ONE node twice — which is exactly the defect — because with
    # both breakers open the selector returns pool[0] every time. The first
    # version of this replacement passed with all five guards neutered.
    assert max(calls.values()) <= 1, (
        f"a node was POSTed to {max(calls.values())} times for ONE logical "
        f"call ({calls}) — the exhaustion guard is inoperative, so an "
        f"already-tried node burns a second full timeout and records a "
        f"second breaker failure")
    assert sum(calls.values()) <= len(c.worker_clients), calls
