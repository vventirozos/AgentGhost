"""Regression: knowledge_base insert_fact must not hang the turn on a stalled
graph-triplet extractor, and must store the fact regardless.

Live functional finding (2026-07-04, functional hunt unit 3): "Add this to your
knowledge base: … Bluefin …" hung the turn to the client timeout (>150s) and a
follow-up query could not find Bluefin. Root cause: `tool_remember`'s bus-aware
path `await`ed the graph-triplet extraction LLM call INLINE, BEFORE the
`publish_fact()` that stores the fact. That inline call parked on
`_wait_for_foreground_clear()` (which waits until `foreground_requests==0` — but
the request it waits for is THIS turn → self-deadlock, 600s ceiling), so the
turn hung AND the fact was never stored.

Fix: publish the fact IMMEDIATELY (triplets are best-effort enrichment), then
extract triplets in a fire-and-forget background task (bounded by
`_GRAPH_EXTRACT_TIMEOUT_S`) and add them to the graph off the critical path —
where `is_background=True` is finally correct because the turn does not await it.
"""
from __future__ import annotations

import asyncio

import pytest

from ghost_agent.tools import memory as M
from ghost_agent.utils import logging as _glog


class _HangingClient:
    async def chat_completion(self, payload, **kw):
        await asyncio.sleep(3600)  # never returns — simulates a stalled worker


class _FastClient:
    def __init__(self, content):
        self._c = content
        self.calls = []

    async def chat_completion(self, payload, **kw):
        self.calls.append(kw)
        return {"choices": [{"message": {"content": self._c}}]}


class _FakeGraph:
    def __init__(self):
        self.added = []

    def add_triplets(self, triplets):
        self.added.append(triplets)
        return len(triplets)


class _FakeBus:
    def __init__(self, graph=None):
        self.published = []
        self.vector = None  # skip the dedup short-circuit
        self.graph = graph

    async def publish_fact(self, action, payload):
        self.published.append((action, payload))


# Graph extraction now schedules through the unified spawn_bg registry
# (utils.logging._BG_TASKS), not the removed module-local set.
@pytest.fixture(autouse=True)
async def _clear_bg_tasks():
    yield
    for t in list(_glog._BG_TASKS):
        t.cancel()
    _glog._BG_TASKS.clear()


async def _drain_bg():
    tasks = list(_glog._BG_TASKS)
    if tasks:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5.0)


async def test_turn_returns_immediately_even_if_extractor_hangs(monkeypatch):
    monkeypatch.setattr(M, "_GRAPH_EXTRACT_TIMEOUT_S", 0.2)
    bus = _FakeBus(graph=_FakeGraph())
    # The tool must return fast — the hang is confined to a fire-and-forget
    # background task, not the turn.
    res = await asyncio.wait_for(
        M.tool_remember("The codename is Bluefin.", memory_bus=bus, llm_client=_HangingClient()),
        timeout=2.0,
    )
    assert "stored" in res.lower()
    # Fact published immediately, without waiting on triplet extraction.
    action, payload = bus.published[0]
    assert action == "insert_fact"
    assert payload["text"] == "The codename is Bluefin."
    assert payload["triplets"] == []


async def test_triplets_added_to_graph_in_background():
    graph = _FakeGraph()
    bus = _FakeBus(graph=graph)
    client = _FastClient('{"graph_triplets": [{"subject": "Bluefin", "predicate": "HAS_DEADLINE", "object": "Aug 15"}]}')
    res = await M.tool_remember("Bluefin deadline is Aug 15", memory_bus=bus, llm_client=client)
    assert "stored" in res.lower()
    # Published immediately with empty triplets (fact stored regardless).
    assert bus.published[0][1]["triplets"] == []
    await _drain_bg()
    # Triplets land in the graph via the background task…
    assert graph.added == [[{"subject": "Bluefin", "predicate": "HAS_DEADLINE", "object": "Aug 15"}]]
    # …and the background extraction correctly used is_background=True.
    assert client.calls[0].get("is_background") is True
    assert client.calls[0].get("use_worker") is True


async def test_fact_stored_with_no_llm_client():
    bus = _FakeBus(graph=_FakeGraph())
    res = await M.tool_remember("A plain fact", memory_bus=bus, llm_client=None)
    assert "stored" in res.lower()
    assert bus.published[0][1]["triplets"] == []
    # No extraction task without an llm_client.
    assert not _glog._BG_TASKS


async def test_empty_text_rejected():
    res = await M.tool_remember("", memory_bus=_FakeBus(), llm_client=None)
    assert "MANDATORY" in res
