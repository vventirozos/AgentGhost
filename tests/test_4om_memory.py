"""§4OM — free-form graph writes off (operator "A"); profile mirrors kept."""
import ast

from ghost_agent.memory.graph import freeform_writes_enabled
from tests.pin_ast import tree, calls, U


def test_free_form_graph_writes_are_off_by_default(monkeypatch):
    monkeypatch.delenv("GHOST_GRAPH_FREEFORM", raising=False)
    assert freeform_writes_enabled() is False
    for v in ("1", "on", "true"):
        monkeypatch.setenv("GHOST_GRAPH_FREEFORM", v)
        assert freeform_writes_enabled() is True


def test_every_free_form_writer_reads_the_switch():
    from ghost_agent.core import agent as ag
    from ghost_agent.tools import memory as mem
    a = [U(n.test) for n in ast.walk(tree(ag)) if isinstance(n, ast.If)]
    assert any("graph_triplets" in x and "_gfw()" in x for x in a)                 # the per-turn extractor
    m = [U(n.test) for n in ast.walk(tree(mem)) if isinstance(n, ast.If)]
    assert sum(1 for x in m if "_gfw()" in x and "llm_client" in x) == 2          # remember: bus + legacy


async def test_remember_writes_no_graph_edges_when_off(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock
    from ghost_agent.tools import memory as mem
    monkeypatch.delenv("GHOST_GRAPH_FREEFORM", raising=False)
    spawned = []
    monkeypatch.setattr(mem, "spawn_bg", lambda coro, name=None: (spawned.append(name), coro.close()))
    ms = MagicMock()
    out = await mem.tool_remember(text="My bike is a Triumph", memory_system=ms, graph_memory=MagicMock(),
                                  llm_client=AsyncMock(), model_name="m")
    assert "graph-extract" not in spawned and "graph-extract-legacy" not in spawned, (out, spawned)


def test_owner_profile_mirrors_still_write_the_graph():
    from ghost_agent.tools import memory as mem
    assert hasattr(mem, "sync_owner_mirrors")
    from ghost_agent.memory import graph as g
    src_calls = [U(c.func) for c in calls(tree(g), "add_triplets")]
    assert src_calls                                                               # the mirror writer is not gated
