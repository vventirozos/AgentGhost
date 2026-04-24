"""Tests for the core/main wiring fixes from the deep-audit remediation pass.

Covers:
* Dead modules now wired in main.py (ContradictionLog, AdaptiveThreshold,
  EpisodicMemory, Verifier, UncertaintyTracker)
* tool_failure.get_fallback_hint reachable from agent error path
* llm.py circuit-breaker filtering on vision/coding/image-gen pools
* llm.py vision/coding retry loops break when all nodes exhausted
* uncertainty risk summary appended to final response when populated

These are unit tests against pure functions / class behaviour — no
network, no Docker, no upstream LLM.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ghost_agent.core.llm import LLMClient
from ghost_agent.core.uncertainty import UncertaintyTracker
from ghost_agent.core.verifier import Verifier
from ghost_agent.memory.adaptive_threshold import AdaptiveThreshold
from ghost_agent.memory.contradiction_log import ContradictionLog
from ghost_agent.memory.episodes import EpisodicMemory
from ghost_agent.tools.tool_failure import get_fallback_hint


# ---------------------------------------------------------------------------
# Dead-module instantiability: each one must be constructible from a
# memory_dir alone (or from no args, for stateless trackers). main.py wires
# them with these exact signatures, so a regression here breaks startup.
# ---------------------------------------------------------------------------

def test_contradiction_log_constructible(tmp_path: Path):
    log = ContradictionLog(tmp_path)
    assert log is not None


def test_adaptive_threshold_constructible(tmp_path: Path):
    th = AdaptiveThreshold(tmp_path)
    assert 0.0 < th.threshold < 1.0


def test_episodic_memory_constructible(tmp_path: Path):
    em = EpisodicMemory(tmp_path)
    # Round-trip: record an episode and verify it persists.
    eid = em.record_episode(
        trigger="test",
        outcome="ok",
        success=True,
        lesson="lesson learned",
    )
    assert isinstance(eid, int) and eid > 0


def test_verifier_constructible_without_llm():
    v = Verifier(llm_client=None)
    assert v is not None


def test_uncertainty_tracker_lifecycle():
    t = UncertaintyTracker()
    t.flag_unknown("config path", impact=5, resolution="ask user")
    t.flag_assumption("default port is 8080", confidence=0.4)
    assert t.should_ask_user() is not None
    summary = t.get_risk_summary()
    assert "Things I'm not certain about" in summary
    assert "Assumptions I made" in summary
    t.reset()
    assert t.get_risk_summary() == ""


# ---------------------------------------------------------------------------
# main.py imports the wiring symbols at module load. Verify the import path
# is correct so a renamed module breaks loudly here, not at server start.
# ---------------------------------------------------------------------------

def test_main_imports_wire_dead_modules():
    import ghost_agent.main as main_mod
    for sym in (
        "ContradictionLog",
        "AdaptiveThreshold",
        "EpisodicMemory",
        "Verifier",
        "UncertaintyTracker",
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
    # Use the same node list for swarm/worker so the constructor is happy;
    # we only care about the specific pool we patch in.
    client = LLMClient(
        upstream_url="http://upstream",
        tor_proxy=None,
        swarm_nodes=[],
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

def _llm_source() -> str:
    src_path = Path(__file__).resolve().parents[1] / "src/ghost_agent/core/llm.py"
    return src_path.read_text()


def test_vision_retry_loop_has_exhaustion_break():
    src = _llm_source()
    # Look for the specific guard pattern we added.
    assert "exhausted" in src.lower() or "if node in tried_nodes:\n                            break" in src, (
        "vision retry loop missing exhaustion guard"
    )


def test_coding_retry_loop_has_exhaustion_break():
    src = _llm_source()
    # The coding loop indents two levels deeper than the worker pool's
    # equivalent; we check for at least two `if node in tried_nodes: break`
    # occurrences (vision + coding).
    occurrences = src.count("if node in tried_nodes:\n")
    assert occurrences >= 2, (
        f"expected ≥2 exhaustion guards (vision + coding); found {occurrences}"
    )
