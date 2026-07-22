"""Contention fix for core/project_research.py LLM calls (MED, 2026-07-22).

The research summariser (`_llm_complete`) used to issue plain FOREGROUND
`chat_completion` calls with a stale "no worker node is configured here"
rationale. It is reached from TWO invocation contexts:

  * idle/background — agent.py idle autoadvance (Phase 2.95) → advance_once
    → persist_research_from_output; dream passes; HTTP/Slack project routes.
    Here a foreground call bumps `foreground_tasks` and contends with a live
    user turn for the single main-model slot (the bug).
  * a user's synchronous path — manage_projects action=research/advance
    inside a live turn (which holds LLMClient.foreground_requests > 0 for
    the whole request; api/routes.py:_mark_foreground). Here is_background
    would park the call behind its OWN request — the 600s self-stall
    documented in tools/delegate.py.

The fix captures the invocation context ONCE at each public entry point via
the delegate.py invariant (`foreground_requests > 0` ⇔ an interactive turn
is in flight; 0 in idle/autoadvance contexts) and threads `is_background`
into the LLM calls. It is a plain MAIN-model call (no worker/critic pool),
so per the LLMClient contract is_background alone is the fix and
`off_main_only` must NOT be passed. Success output must be unchanged.

No real LLM: a recording stub captures the chat_completion kwargs.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import inspect
from types import SimpleNamespace

import pytest

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.core import project_research as pr
from ghost_agent.core.project_research import (
    research_topic, research_project, persist_research_from_output,
    propose_topics, get_research_index, _idle_invocation, _resolve_background,
)


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


SEARCH_OUT = (
    "### 1. Result one\nSome finding.\n[Source: http://example.com/one]\n\n"
    "### 2. Result two\nAnother finding.\n[Source: http://example.com/two]"
)

SUMMARY_MD = ("## Summary\nMain-model summary of the findings.\n\n"
              "## Key findings\n- finding one\n- finding two")


class _RecordingLLM:
    """chat_completion stub recording (payload, kwargs) per call.

    ``foreground_requests`` mimics LLMClient's active-user-request counter;
    pass ``fg=None`` to build a stub WITHOUT the attribute (a minimal client
    the auto-detect must treat as "can't tell → stay foreground")."""

    def __init__(self, content=SUMMARY_MD, fg=0):
        self.content = content
        self.calls = []
        if fg is not None:
            self.foreground_requests = fg

    async def chat_completion(self, payload, **kw):
        self.calls.append((payload, kw))
        return {"choices": [{"message": {"content": self.content}}]}


def _ctx(store, llm):
    return SimpleNamespace(
        project_store=store, graph_memory=None, contradiction_log=None,
        current_project_id=None, args=SimpleNamespace(model="m"),
        llm_client=llm,
    )


def _runner(output=SEARCH_OUT):
    async def run(name, args):
        return output
    return run


# ================================================== invocation-context detect

def test_idle_invocation_reads_foreground_requests(store):
    # Idle/autoadvance context: counter present and 0 → idle.
    assert _idle_invocation(_ctx(store, _RecordingLLM(fg=0))) is True
    # A live user turn holds the counter > 0 → NOT idle.
    assert _idle_invocation(_ctx(store, _RecordingLLM(fg=2))) is False
    # Counter missing (minimal stub) → can't tell → NOT idle (fail-safe:
    # never park a call that might be inline on its own user request).
    assert _idle_invocation(_ctx(store, _RecordingLLM(fg=None))) is False
    # No llm_client at all → NOT idle.
    assert _idle_invocation(SimpleNamespace(llm_client=None)) is False


def test_resolve_background_explicit_wins(store):
    ctx_idle = _ctx(store, _RecordingLLM(fg=0))
    ctx_busy = _ctx(store, _RecordingLLM(fg=1))
    assert _resolve_background(ctx_idle, None) is True
    assert _resolve_background(ctx_busy, None) is False
    # Explicit override beats auto-detect in both directions.
    assert _resolve_background(ctx_idle, False) is False
    assert _resolve_background(ctx_busy, True) is True


# ================================================== advancer persist path

async def test_idle_persist_marks_llm_call_background(store):
    """The autoadvancer path (persist_research_from_output) in an idle
    context must pass is_background=True — and never off_main_only /
    use_worker (plain main call)."""
    llm = _RecordingLLM(fg=0)
    ctx = _ctx(store, llm)
    pid = store.create_project("P", goal="g")
    rr = await persist_research_from_output(ctx, pid, "Idle topic", SEARCH_OUT)

    assert rr.ok
    assert len(llm.calls) == 1
    payload, kw = llm.calls[0]
    assert kw.get("is_background") is True
    # Plain main call: no off-main pool flags, so no OffMainNodeUnavailable
    # path exists to handle.
    assert kw.get("off_main_only") is not True
    assert kw.get("use_worker") is not True
    assert kw.get("use_critic") is not True


async def test_user_turn_persist_stays_foreground(store):
    """advance_once reached via manage_projects during a LIVE user turn
    (foreground_requests > 0) must stay foreground — is_background there
    parks the call behind its own request (delegate.py's 600s self-stall)."""
    llm = _RecordingLLM(fg=1)
    ctx = _ctx(store, llm)
    pid = store.create_project("P", goal="g")
    rr = await persist_research_from_output(ctx, pid, "Turn topic", SEARCH_OUT)

    assert rr.ok
    _, kw = llm.calls[0]
    assert kw.get("is_background") is False


async def test_counterless_client_keeps_foreground_status_quo(store):
    """A client without the foreground_requests counter (minimal stubs,
    exotic clients) cannot prove no user turn is in flight → foreground."""
    llm = _RecordingLLM(fg=None)
    ctx = _ctx(store, llm)
    pid = store.create_project("P", goal="g")
    await persist_research_from_output(ctx, pid, "Stub topic", SEARCH_OUT)
    _, kw = llm.calls[0]
    assert kw.get("is_background") is False


async def test_explicit_background_override_beats_autodetect(store):
    llm = _RecordingLLM(fg=1)  # looks like a live user turn...
    ctx = _ctx(store, llm)
    pid = store.create_project("P", goal="g")
    await persist_research_from_output(ctx, pid, "Forced", SEARCH_OUT,
                                       background=True)  # ...caller knows best
    _, kw = llm.calls[0]
    assert kw.get("is_background") is True


# ================================================== tool-facing entry points

async def test_research_topic_idle_marks_background(store):
    llm = _RecordingLLM(fg=0)
    ctx = _ctx(store, llm)
    pid = store.create_project("P", goal="g")
    rr = await research_topic(ctx, pid, "Idle research", tool_runner=_runner())
    assert rr.ok
    _, kw = llm.calls[0]
    assert kw.get("is_background") is True


async def test_research_topic_captures_context_before_search(store):
    """The background decision is captured at run START: a user turn that
    begins DURING the web search must not flip an idle run back to
    foreground (the idle run should park politely, not contend)."""
    llm = _RecordingLLM(fg=0)
    ctx = _ctx(store, llm)
    pid = store.create_project("P", goal="g")

    async def turn_arrives_mid_search(name, args):
        llm.foreground_requests = 1  # user turn starts mid-run
        return SEARCH_OUT

    rr = await research_topic(ctx, pid, "Race topic",
                              tool_runner=turn_arrives_mid_search)
    assert rr.ok
    _, kw = llm.calls[0]
    assert kw.get("is_background") is True


async def test_propose_topics_idle_marks_background(store):
    llm = _RecordingLLM(content="topic alpha\ntopic beta", fg=0)
    ctx = _ctx(store, llm)
    pid = store.create_project("P", goal="build a thing")
    store.add_task(pid, "do something")
    topics = await propose_topics(ctx, pid, n=3)
    assert topics == ["topic alpha", "topic beta"]
    _, kw = llm.calls[0]
    assert kw.get("is_background") is True


async def test_research_project_threads_one_decision_through_run(store):
    """research_project resolves background ONCE and threads it to
    propose_topics AND every research_topic — every LLM call in an
    idle-started multi-topic run is background, even if a user turn
    arrives between topics."""
    llm = _RecordingLLM(content="only topic", fg=0)
    ctx = _ctx(store, llm)
    pid = store.create_project("P", goal="g")
    store.add_task(pid, "seed task")

    results = await research_project(ctx, pid, tool_runner=_runner(),
                                     max_topics=2)
    assert results and all(r.ok for r in results)
    assert llm.calls  # propose + summarize calls
    for _, kw in llm.calls:
        assert kw.get("is_background") is True


# ================================================== output unchanged + hygiene

async def test_success_output_unchanged_on_background_path(store):
    """The fix must not change WHAT research computes: same brief content,
    same no-think payload shape, same index entry."""
    llm = _RecordingLLM(fg=0)
    ctx = _ctx(store, llm)
    pid = store.create_project("P", goal="g")
    rr = await persist_research_from_output(ctx, pid, "Same output", SEARCH_OUT)

    assert rr.ok
    assert "Main-model summary" in rr.summary
    assert rr.sources == ["http://example.com/one", "http://example.com/two"]
    payload, _ = llm.calls[0]
    # No-think switches preserved (reasoning upstream burns budget otherwise).
    assert payload["chat_template_kwargs"]["enable_thinking"] is False
    assert payload["messages"][-1]["content"].rstrip().endswith("/no_think")
    assert payload["stream"] is False
    idx = get_research_index(store, pid)
    assert idx and idx[0]["slug"] == "same-output"
    assert idx[0]["summary_preview"].startswith("Main-model summary")


def test_llm_complete_threads_flag_and_comment_updated():
    """Source-level guards: the closure threads is_background into
    chat_completion, never passes off_main_only (plain main call), and the
    stale 'no worker node is configured' rationale is gone."""
    src = inspect.getsource(pr._llm_complete)
    assert "is_background=is_background" in src
    assert "off_main_only" not in src.split('"""')[2]  # not in the CODE body
    assert "no worker node is configured" not in src
    module_src = inspect.getsource(pr)
    assert "no worker node is configured" not in module_src


def test_module_imports_cleanly():
    import importlib
    importlib.reload(pr)
    assert callable(pr._llm_complete)
    assert callable(pr._idle_invocation)
