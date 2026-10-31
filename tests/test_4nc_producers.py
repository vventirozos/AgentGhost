"""§4NC (2026-10-10, operator decisions): background producers no longer write
lessons (only approved rules and the owner's dictation), graph links and
episodes are no longer auto-loaded into prompts (recall keeps them), and the
idle busywork stops. §4NM removed the producers themselves (reflection, the
post-mortem engine, REM) and the GHOST_PRODUCER_LESSONS / GHOST_POSTMORTEM_ENGINE
switches: `producer_may_write` is now a hard rule, so this file pins PRODUCTION
with no switch to clear."""
from __future__ import annotations

import ast
import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def production(monkeypatch):
    for k in ("GHOST_BUS_GRAPH_TIER", "GHOST_BUS_EPISODIC_TIER"):
        monkeypatch.delenv(k, raising=False)


# ── lesson writers ────────────────────────────────────────────────────

@pytest.mark.parametrize("source,written", [
    ("dream", False), ("dream_pattern", False), ("distilled", False), ("self_play", False),
    ("bench", False), ("reflection", False), ("journal_postmortem", False), ("episode", False),
    ("perfection_protocol", False), ("", False),
    ("learn_skill", True), ("operator_repair", True),
])
def test_only_the_owners_and_the_operators_lessons_are_written(tmp_path, production, source, written):
    from ghost_agent.memory.skills import SkillMemory
    sm = SkillMemory(tmp_path)
    t = f"when checking a disk quota on a {source or 'blank'} host"
    w = sm.learn_lesson(t, "guessed the number", "Run df -h and quote the line.", trigger=t, source=source)
    assert (w is not None) is written


def _fn(module, name):
    tree = ast.parse(inspect.getsource(module))
    return next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)


def _calls(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)
            and (getattr(n.func, "id", "") == name or getattr(n.func, "attr", "") == name)]


# ── graph links and episodes stay out of prompts ──────────────────────

def _bus():
    from ghost_agent.core.bus import MemoryBus
    graph, episodic = MagicMock(), MagicMock()
    bus = MemoryBus.__new__(MemoryBus)
    bus.graph, bus.episodic, bus.vector = graph, episodic, None
    return bus, graph, episodic


def test_graph_and_episodes_are_not_loaded_into_prompts(production, monkeypatch):
    """§4NM: the topical graph edges and the episodes never hydrate a prompt
    — the switches that brought them back were deleted with the code."""
    monkeypatch.setenv("GHOST_BUS_GRAPH_TIER", "1")
    monkeypatch.setenv("GHOST_BUS_EPISODIC_TIER", "1")
    bus, graph, episodic = _bus()
    assert asyncio.run(bus._fetch_graph("postgresql latest version")) == []
    assert not hasattr(bus, "_fetch_episodic")
    assert not graph.get_neighborhood.called and not episodic.method_calls


def test_recall_does_not_go_through_the_hydration_tiers():
    """Recall keeps graph and episodes: nothing outside the bus's hydration
    fan-out calls the two fetchers."""
    import pathlib, ghost_agent
    root = pathlib.Path(ghost_agent.__file__).parent
    users = [p for p in root.rglob("*.py") if p.name != "bus.py"
             and any(isinstance(n, ast.Attribute) and n.attr in ("_fetch_graph", "_fetch_episodic")
                     for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))))]
    assert users == []


# ── busywork ──────────────────────────────────────────────────────────

# ── r1 review ─────────────────────────────────────────────────────────

def test_store_maintenance_still_runs_without_rem():
    """r1 MAJOR: graph pruning, node compression and the reconcile lived only
    in REM's tail — skipping REM stopped all three. §4NM removed REM (and the
    RRF refit) outright; the housekeeping pass still calls the maintenance."""
    from ghost_agent.core import dream
    tree = ast.parse(inspect.getsource(dream))
    maint = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "_store_maintenance")
    used = {n.attr for n in ast.walk(maint) if isinstance(n, ast.Attribute)}
    for name in ("prune_stale_edges", "_compress_graph_nodes", "_reconcile_memory_stores"):
        assert name in used, name
    run = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "dream")
    assert _calls(run, "_store_maintenance")


def test_store_maintenance_runs_at_most_every_six_hours_on_the_skip_path():
    from ghost_agent.core.dream import Dreamer
    assert Dreamer.STORE_MAINTENANCE_EVERY_S == 6 * 3600


def test_a_refused_perfect_it_lesson_logs_no_save(monkeypatch, production):
    from ghost_agent.core import agent
    src = ast.unparse(_fn(agent, "_perfect_it_generate_and_learn"))
    # §4NV: the source is refused since §4NC, so there is no "saved" line at all
    assert "Saved optimization strategy" not in src and "Internal Learning" not in src


def test_owner_facts_still_reach_a_question_about_the_owner(production):
    bus, graph, episodic = _bus()
    graph.owner_facts_matching = MagicMock(return_value=["User LIVES_IN Athens (as of 2026-10-01)"])
    out = asyncio.run(bus._fetch_graph("where do I live?"))
    assert out == [{"source": "graph", "text": "User LIVES_IN Athens (as of 2026-10-01)"}]
    assert asyncio.run(bus._fetch_graph("latest postgresql version")) == []


def test_the_model_facing_text_promises_no_lesson_making():
    import json
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    from ghost_agent.core import prompts
    # §4NM: self_play left the registry; §4SI: dream_mode and its SLEEP/REST line too
    names = {d["function"]["name"] for d in TOOL_DEFINITIONS}
    assert not {"self_play", "dream_mode"} & names
    texts = [v for v in vars(prompts).values() if isinstance(v, str)]
    assert not any("SLEEP/REST" in v or "extract heuristics" in v for v in texts)


