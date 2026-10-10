"""§4NC (2026-10-10, operator decisions): background producers no longer write
lessons (only approved rules and the owner's dictation), graph links and
episodes are no longer auto-loaded into prompts (recall keeps them), and the
idle busywork stops. Each test CLEARS the conftest switches that keep the
producers' own tests exercising their code — this file pins PRODUCTION."""
from __future__ import annotations

import ast
import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def production(monkeypatch):
    for k in ("GHOST_PRODUCER_LESSONS", "GHOST_BUS_GRAPH_TIER", "GHOST_BUS_EPISODIC_TIER",
              "GHOST_POSTMORTEM_ENGINE"):
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


def test_the_switch_reopens_every_writer(tmp_path, monkeypatch):
    from ghost_agent.memory.skills import SkillMemory
    monkeypatch.setenv("GHOST_PRODUCER_LESSONS", "1")
    t = "when checking a disk quota on a dream host"
    assert SkillMemory(tmp_path).learn_lesson(t, "m", "Run df -h.", trigger=t, source="dream") is not None


def _fn(module, name):
    tree = ast.parse(inspect.getsource(module))
    return next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)


def _calls(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)
            and (getattr(n.func, "id", "") == name or getattr(n.func, "attr", "") == name)]


def test_the_lesson_only_phases_are_gated_where_they_run():
    """Dream's episode consolidation, failure distillation and REM, reflection
    and the journal post-mortem only write lessons — they no longer spend
    model time when lessons are off."""
    from ghost_agent.core import dream, agent
    run = next(n for n in ast.walk(ast.parse(inspect.getsource(dream)))
               if isinstance(n, ast.AsyncFunctionDef) and _calls(n, "_consolidate_episodes")
               and _calls(n, "distill_failure_clusters"))
    assert len(_calls(run, "_ple")) >= 3                       # episodes, distill, REM
    src = ast.unparse(run)
    assert src.index("_ple()") < src.index("_consolidate_episodes")
    assert src.index("if not _ple():\n", src.index("_daily_store_care")) < src.index("Entering REM cycle")
    agent_src = ast.unparse(ast.parse(inspect.getsource(agent)))
    assert agent_src.index("if not _ple_r():") < agent_src.index("_refl_trajs = await")
    assert "_ple_j()" in agent_src


def test_the_postmortem_engine_is_off_unless_asked(production, monkeypatch):
    from ghost_agent.core import agent
    assert agent._postmortem_engine_on() is False
    monkeypatch.setenv("GHOST_POSTMORTEM_ENGINE", "1")
    assert agent._postmortem_engine_on() is True
    tree = ast.parse(inspect.getsource(agent))
    assert any(isinstance(n, ast.BoolOp) and any(getattr(getattr(v, "func", None), "id", "") == "_postmortem_engine_on"
                                                 for v in n.values) for n in ast.walk(tree))


# ── graph links and episodes stay out of prompts ──────────────────────

def _bus():
    from ghost_agent.core.bus import MemoryBus
    graph, episodic = MagicMock(), MagicMock()
    bus = MemoryBus.__new__(MemoryBus)
    bus.graph, bus.episodic, bus.vector = graph, episodic, None
    return bus, graph, episodic


def test_graph_and_episodes_are_not_loaded_into_prompts(production):
    bus, graph, episodic = _bus()
    assert asyncio.run(bus._fetch_graph("postgresql latest version")) == []
    assert asyncio.run(bus._fetch_episodic("postgresql latest version")) == []
    assert not graph.method_calls and not episodic.method_calls


def test_the_switches_bring_them_back(monkeypatch):
    monkeypatch.setenv("GHOST_BUS_GRAPH_TIER", "1")
    monkeypatch.setenv("GHOST_BUS_EPISODIC_TIER", "1")
    bus, graph, episodic = _bus()
    assert bus._GRAPH_TIER_ENABLED and bus._EPISODIC_TIER_ENABLED


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

def test_skills_auto_skips_an_unchanged_trajectory_store(tmp_path):
    from ghost_agent.core.agent import _trajectories_unchanged
    (tmp_path / "d").mkdir()
    f = tmp_path / "d" / "session-1.jsonl"
    f.write_text("{}\n")
    owner, coll = SimpleNamespace(), SimpleNamespace(root=tmp_path)
    assert _trajectories_unchanged(owner, coll) is False          # first look: run
    assert _trajectories_unchanged(owner, coll) is True           # nothing new: skip
    f.write_text("{}\n{}\n")
    assert _trajectories_unchanged(owner, coll) is False          # a new turn: run
    assert _trajectories_unchanged(owner, SimpleNamespace(root=tmp_path / "missing")) in (True, False)


def test_a_macro_mint_skip_is_a_debug_line(monkeypatch):
    from ghost_agent.core import agent
    seen = []
    monkeypatch.setattr(agent, "pretty_log", lambda *a, **k: seen.append(k.get("level")))
    monkeypatch.setattr(agent, "macro_mint_skip_is_new", lambda *a: True)
    agent.report_macro_mint_skip(SimpleNamespace(), "m", ("a", "b"), "why")
    assert seen == ["DEBUG"]


# ── r1 review ─────────────────────────────────────────────────────────

def test_store_maintenance_still_runs_without_rem():
    """r1 MAJOR: graph pruning, node compression, the reconcile and the RRF
    refit lived only in REM's tail — skipping REM stopped all four."""
    from ghost_agent.core import dream
    tree = ast.parse(inspect.getsource(dream))
    maint = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "_store_maintenance")
    used = {n.attr for n in ast.walk(maint) if isinstance(n, ast.Attribute)}
    for name in ("prune_stale_edges", "_compress_graph_nodes", "_reconcile_memory_stores", "_refit_rrf_weights"):
        assert name in used, name
    run = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "dream")
    skips = [n for n in ast.walk(run) if isinstance(n, ast.If) and isinstance(n.test, ast.UnaryOp)
             and isinstance(n.test.operand, ast.Call) and getattr(n.test.operand.func, "id", "") == "_ple"
             and any(isinstance(r, ast.Return) for r in n.body)]
    assert len(skips) == 1 and _calls(skips[0], "_store_maintenance")


def test_store_maintenance_runs_at_most_every_six_hours_on_the_skip_path():
    from ghost_agent.core.dream import Dreamer
    assert Dreamer.STORE_MAINTENANCE_EVERY_S == 6 * 3600


def test_the_lesson_only_llm_calls_are_not_made_when_lessons_are_off():
    """r1 MAJOR: the journal post-mortem, Perfect-It and correction reflection
    still made their model call and threw the lesson away."""
    from ghost_agent.core import agent
    src = ast.unparse(ast.parse(inspect.getsource(agent)))
    i = src.index("elif item['type'] == 'post_mortem':")
    assert src.index("_ple_pm()", i) < src.index("_execute_post_mortem", i)
    assert "if reflector is None or not _ple_cr():" in src
    assert "producer_lessons_enabled():" in src[src.index("_deferred_perfect_it") - 900: src.index("_deferred_perfect_it")]


def test_a_refused_perfect_it_lesson_logs_no_save(monkeypatch, production):
    from ghost_agent.core import agent
    src = ast.unparse(_fn(agent, "_perfect_it_generate_and_learn"))
    assert "if _w_pp is not None:" in src


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
    for name in ("dream_mode", "self_play"):
        t = json.dumps(next(d for d in TOOL_DEFINITIONS if d["function"]["name"] == name))
        assert "lesson" in t and "Active Memory Consolidation" not in t and "training curriculum" not in t
    texts = [v for v in vars(prompts).values() if isinstance(v, str) and "SLEEP/REST" in v]
    assert texts and not any("extract heuristics" in v for v in texts)


def test_self_play_says_no_lesson_when_none_was_saved():
    from ghost_agent.core import dream
    consts = [n.value for n in ast.walk(ast.parse(inspect.getsource(dream)))
              if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert any("No lesson saved" in c for c in consts)
