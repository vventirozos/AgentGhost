"""Tests for the project auto-research subsystem (core/project_research.py)
and its wiring into the manage_projects tool + project briefing.

No real LLM or network: a fake tool_runner returns canned search output,
and the context has no llm_client so summarisation takes the heuristic
fallback path. That keeps the tests deterministic while still exercising
search → summarise → persist → index → surface end-to-end.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.memory.scratchpad import Scratchpad
from ghost_agent.core import project_research as pr
from ghost_agent.core.project_research import (
    research_topic, research_project, persist_research_from_output,
    propose_topics, get_research_index, _slugify, _extract_sources,
    _first_line, _strip_think, _message_text,
)
from ghost_agent.core.prompts import build_project_briefing
from ghost_agent.tools.projects import tool_manage_projects


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


@pytest.fixture
def context(tmp_path, store):
    # No llm_client → heuristic summariser; no real registry → search must
    # be injected via tool_runner.
    return SimpleNamespace(
        project_store=store,
        scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
        graph_memory=None,
        contradiction_log=None,
        current_project_id=None,
    )


SEARCH_OUT = (
    "### 1. BGE-M3 overview\nA multilingual embedding model.\n"
    "[Source: http://example.com/bge]\n\n"
    "### 2. Benchmarks\nStrong MTEB scores.\n[Source: http://example.com/bench]"
)


def _runner_factory(output=SEARCH_OUT):
    calls = []

    async def runner(name, args):
        calls.append((name, args))
        return output

    return runner, calls


# ============================================================ pure helpers

def test_slugify_is_path_safe():
    assert _slugify("Hello, World! / ../etc") == "hello-world-etc"
    assert _slugify("") == "topic"
    assert "/" not in _slugify("a/b/c") and ".." not in _slugify("..")


def test_extract_sources_dedups_and_skips_hash():
    out = "[Source: http://a.com]\n[Source: http://a.com]\n[Source: #]"
    assert _extract_sources(out) == ["http://a.com"]


def test_first_line_skips_markdown_heading():
    # The preview must be a real sentence, not the '## Summary' heading.
    assert _first_line("## Summary\nReal sentence here.") == "Real sentence here."
    assert _first_line("# H\n- bullet point") == "bullet point"
    assert _first_line("") == ""


class _FakeLLM:
    """Minimal chat_completion stub recording how it was called."""

    def __init__(self, content):
        self.content = content
        self.calls = []

    async def chat_completion(self, payload, **kw):
        self.calls.append((payload, kw))
        return {"choices": [{"message": {"content": self.content}}]}


def test_strip_think_removes_reasoning():
    assert _strip_think("<think>plan plan</think>Real answer") == "Real answer"
    # Unclosed <think> (budget exhausted mid-reasoning) → nothing usable.
    assert _strip_think("<think>still thinking and never closed") == ""
    assert _strip_think("plain text") == "plain text"


def test_message_text_falls_back_to_reasoning_content():
    # content empty (budget went to <think>) → salvage reasoning_content.
    resp = {"choices": [{"message": {
        "content": "", "reasoning_content": "The key point is X."}}]}
    assert _message_text(resp) == "The key point is X."
    # content present → preferred, with think stripped.
    resp2 = {"choices": [{"message": {
        "content": "<think>x</think>Clean answer.", "reasoning_content": "y"}}]}
    assert _message_text(resp2) == "Clean answer."


async def test_research_uses_llm_summary_with_no_think_payload(store):
    """With an LLM client present the brief uses the model summary; the call
    disables thinking (the upstream is a reasoning model that otherwise burns
    the whole budget inside <think> and returns empty content), is a plain
    foreground call, and the index preview skips the heading."""
    llm = _FakeLLM("## Summary\nKyoto is loveliest in autumn and spring.\n\n"
                   "## Key findings\n- Foliage peaks mid-November.")
    ctx = SimpleNamespace(
        project_store=store, graph_memory=None, contradiction_log=None,
        current_project_id=None, args=SimpleNamespace(model="m"), llm_client=llm,
    )
    pid = store.create_project("P", goal="plan a trip")
    runner, _ = _runner_factory()
    rr = await research_topic(ctx, pid, "Kyoto seasons", tool_runner=runner)

    assert "loveliest in autumn" in rr.summary
    payload, kw = llm.calls[0]
    # No-think hard + soft switches, bounded, plain foreground.
    assert payload["chat_template_kwargs"]["enable_thinking"] is False
    assert payload["messages"][-1]["content"].rstrip().endswith("/no_think")
    assert kw.get("is_background") is not True
    assert kw.get("use_worker") is not True
    assert kw.get("timeout")
    idx = get_research_index(store, pid)
    assert idx[0]["summary_preview"].startswith("Kyoto is loveliest")


# ============================================================ research_topic

async def test_research_topic_writes_brief_and_indexes(context, store, tmp_path):
    pid = store.create_project("P", goal="study embeddings")
    runner, calls = _runner_factory()
    rr = await research_topic(context, pid, "BGE-M3 embeddings", tool_runner=runner)

    assert rr.ok
    assert rr.path == "research/bge-m3-embeddings.md"
    assert calls and calls[0][0] == "web_search"
    # File persisted in the project workspace.
    ws = store.ensure_workspace(pid)
    brief = ws / "research" / "bge-m3-embeddings.md"
    assert brief.exists()
    body = brief.read_text()
    assert "# Research: BGE-M3 embeddings" in body
    assert "http://example.com/bge" in body  # sources section
    # Index entry + event recorded.
    idx = get_research_index(store, pid)
    assert len(idx) == 1 and idx[0]["slug"] == "bge-m3-embeddings"
    assert idx[0]["sources_count"] == 2
    assert store.list_events(pid, event_type="research_added")
    # INDEX.md mirror written.
    assert (ws / "research" / "INDEX.md").exists()


async def test_research_topic_without_runner_still_records(context, store):
    """No search runner available → still writes a sourceless brief so the
    topic is recorded rather than silently dropped."""
    pid = store.create_project("P")
    rr = await research_topic(context, pid, "some topic", tool_runner=None)
    assert rr.ok
    assert rr.sources == []
    assert get_research_index(store, pid)[0]["slug"] == "some-topic"


async def test_research_topic_empty_topic_rejected(context, store):
    pid = store.create_project("P")
    rr = await research_topic(context, pid, "   ", tool_runner=None)
    assert not rr.ok and "empty" in rr.error


async def test_re_research_same_topic_replaces_index_entry(context, store):
    pid = store.create_project("P")
    runner, _ = _runner_factory()
    await research_topic(context, pid, "Topic X", tool_runner=runner)
    await research_topic(context, pid, "topic x", tool_runner=runner)  # same slug
    idx = get_research_index(store, pid)
    assert len(idx) == 1  # replaced, not duplicated


async def test_persist_from_output_no_second_search(context, store):
    pid = store.create_project("P")
    rr = await persist_research_from_output(context, pid, "Inline topic", SEARCH_OUT)
    assert rr.ok
    assert len(rr.sources) == 2
    assert get_research_index(store, pid)[0]["slug"] == "inline-topic"


# ============================================================ topic discovery

async def test_propose_topics_fallback_uses_open_tasks(context, store):
    pid = store.create_project("P", goal="build a thing")
    store.add_task(pid, "research vector databases")
    store.add_task(pid, "research embedding models")
    done = store.add_task(pid, "already done")
    store.update_task(done, status="DONE")
    topics = await propose_topics(context, pid, n=5)
    # DONE task excluded; open ones used as fallback topics.
    assert "research vector databases" in topics
    assert "research embedding models" in topics
    assert "already done" not in topics


async def test_research_project_explicit_topics(context, store):
    pid = store.create_project("P")
    runner, _ = _runner_factory()
    results = await research_project(
        context, pid, topics=["topic a", "topic b"], tool_runner=runner)
    assert len(results) == 2 and all(r.ok for r in results)
    slugs = {e["slug"] for e in get_research_index(store, pid)}
    assert slugs == {"topic-a", "topic-b"}


async def test_research_project_auto_derives_when_no_topics(context, store):
    pid = store.create_project("P", goal="g")
    store.add_task(pid, "research alpha")
    runner, _ = _runner_factory()
    results = await research_project(context, pid, tool_runner=runner, max_topics=3)
    assert results and any(r.ok for r in results)
    assert any(e["topic"] == "research alpha" for e in get_research_index(store, pid))


# ============================================================ briefing awareness

def test_briefing_includes_research_notes(store):
    pid = store.create_project("P", goal="g")
    # Seed an index entry directly (mirrors what research_topic writes).
    meta = {"research_index": [
        {"topic": "Vector DBs", "slug": "vector-dbs",
         "path": "research/vector-dbs.md", "ts": 1.0,
         "summary_preview": "pgvector vs qdrant", "sources_count": 3},
    ]}
    store.update_project(pid, metadata=meta)
    briefing = build_project_briefing(store, pid)
    assert "RESEARCH NOTES" in briefing
    assert "Vector DBs" in briefing
    assert "research/vector-dbs.md" in briefing


# ============================================================ tool surface

async def test_tool_research_single_topic(context, store):
    await tool_manage_projects(context, action="create", title="Research Proj")
    pid = context.current_project_id
    # Inject a registry-free search by monkeypatching the runner resolution:
    # research_topic falls back to writing a sourceless brief when no runner
    # is available, which is enough to exercise the tool path deterministically.
    out = await tool_manage_projects(context, action="research", topic="Quantum error correction")
    data = json.loads(out)
    assert data["researched"] == "Quantum error correction"
    assert data["path"] == "research/quantum-error-correction.md"
    # research_list reflects it.
    out2 = await tool_manage_projects(context, action="research_list")
    listed = json.loads(out2)["research"]
    assert any(r["slug"] == "quantum-error-correction" for r in listed)


async def test_tool_research_auto_derive(context, store):
    await tool_manage_projects(context, action="create", title="Auto Proj", goal="explore X")
    pid = context.current_project_id
    await tool_manage_projects(context, action="task_add", description="research subtopic one")
    out = await tool_manage_projects(context, action="research", max_topics=2)
    data = json.loads(out)
    assert data["count"] >= 1
    assert any("subtopic one" in r["topic"] for r in data["researched"])


async def test_tool_research_requires_project(context):
    out = await tool_manage_projects(context, action="research", topic="x")
    assert out.startswith("ERROR")  # no active project
