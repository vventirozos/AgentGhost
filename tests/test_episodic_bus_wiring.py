"""EpisodicMemory must be passed into MemoryBus so `_fetch_episodic`
actually sees an instance during context hydration. Before this wiring,
the bus's episodic slot stayed None even though main.py constructed
an EpisodicMemory instance on the context.
"""

import pytest
from unittest.mock import MagicMock


class TestMemoryBusEpisodicSlot:
    def test_memory_bus_accepts_episodic_kwarg(self):
        from ghost_agent.core.bus import MemoryBus
        ep = MagicMock()
        bus = MemoryBus(episodic_memory=ep)
        assert bus.episodic is ep

    async def test_fetch_episodic_uses_instance(self):
        """When an episodic memory is wired, _fetch_episodic calls through
        to its search_similar + format_for_context methods."""
        from ghost_agent.core.bus import MemoryBus
        ep = MagicMock()
        ep.search_similar = MagicMock(return_value=[{"id": 1, "text": "prior"}])
        ep.format_for_context = MagicMock(return_value="past episode: prior")
        bus = MemoryBus(episodic_memory=ep)
        items = await bus._fetch_episodic("how to fix X")
        assert len(items) == 1
        assert items[0]["source"] == "episodic"
        assert "past episode: prior" in items[0]["text"]

    async def test_fetch_episodic_noop_without_instance(self):
        from ghost_agent.core.bus import MemoryBus
        bus = MemoryBus(episodic_memory=None)
        assert await bus._fetch_episodic("any") == []


class TestAgentLazyBusIncludesEpisodic:
    """When the agent lazily builds a MemoryBus from context (tests path),
    the episodic_memory slot on context must be threaded through."""

    def test_agent_lazy_bus_propagates_episodic(self):
        from ghost_agent.core.agent import GhostAgent, GhostContext
        import argparse, tempfile
        from pathlib import Path

        args = argparse.Namespace(
            max_context=4096, smart_memory=0.0, anonymous=True,
            model="test", verbose=False, no_memory=True,
        )
        with tempfile.TemporaryDirectory() as td:
            ctx = GhostContext(args, Path(td), Path(td), "socks5://none")
            ctx.memory_bus = None  # force the lazy path
            sentinel = MagicMock()
            ctx.episodic_memory = sentinel
            agent = GhostAgent(ctx)
            bus = agent._get_memory_bus()
            assert bus.episodic is sentinel


class TestHydrationIntegratesEpisodic:
    """End-to-end: a hydration call with an episodic instance produces an
    episodic section in the fused markdown output."""

    async def test_hydrate_includes_episodic_section(self):
        from ghost_agent.core.bus import MemoryBus
        ep = MagicMock()
        ep.search_similar = MagicMock(return_value=[{"id": 1, "text": "prior fix"}])
        ep.format_for_context = MagicMock(
            return_value="past episode: used subprocess.run to fix the auth bug"
        )
        bus = MemoryBus(episodic_memory=ep)
        out = await bus.hydrate_context("how do I fix the auth bug")
        assert "PAST EPISODES" in out
        assert "subprocess.run" in out
