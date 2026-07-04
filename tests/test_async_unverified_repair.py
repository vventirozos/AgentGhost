"""Async-critic mode must still refuse to finalise on an untested write.

Regression for the 2026-07-04 chess session: with ``--critic-nodes``
(``GHOST_CRITIC_ASYNC=1``) the in-loop VERIFIER-GATE AUTO-REPAIR block was
skipped ENTIRELY, so six consecutive turns finalised on ``terminal_chess.py``
writes that were never executed — every crash (module shadowing, IndexError,
NameError, hallucinated API) shipped to the user, who became the test
harness. The post-loop gate only APPENDED a note ("flagged INCOMPLETE") and
shipped anyway.

The fix: the unverified-mutation check is a pure predicate over the turn's
tool records (``_is_unverified_mutation`` — no LLM call, no latency), so
async mode now runs it inline and triggers the same bounded "actually RUN
it" re-entry as sync mode, while the LLM verdict itself stays deferred to
the post-loop gate (off the critical path).
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from ghost_agent.core.agent import GhostAgent, GhostContext
from ghost_agent.core.verifier import VerifyResult, VerifyVerdict


@pytest.fixture
def agent():
    ctx = MagicMock(spec=GhostContext)
    ctx.args = MagicMock()
    ctx.args.temperature = 0.7
    ctx.args.max_context = 8000
    ctx.args.smart_memory = 0.0
    ctx.args.use_planning = False
    ctx.args.model = "Qwen-Test"
    ctx.llm_client = MagicMock()
    ctx.profile_memory = MagicMock()
    ctx.profile_memory.get_context_string.return_value = ""
    ctx.skill_memory = MagicMock()
    ctx.skill_memory.get_context_string.return_value = ""
    ctx.memory_system = MagicMock()
    ctx.memory_system.search = MagicMock(return_value="")
    ctx.cached_sandbox_state = None
    ctx.sandbox_dir = "/tmp/sandbox"
    ctx.verifier = None
    return GhostAgent(ctx)


def _tool_call(name="file_system",
               args='{"operation": "write", "path": "app.py", "content": "x=1"}',
               tid="t1"):
    return {"choices": [{"message": {"content": "working", "tool_calls": [
        {"id": tid, "function": {"name": name, "arguments": args}}]}}]}


def _final(text):
    return {"choices": [{"message": {"content": text, "tool_calls": []}}]}


async def _drive(agent, user, seq):
    calls = []

    async def _spy(payload, *a, **k):
        calls.append([m.get("content", "") for m in payload.get("messages", [])])
        return seq[min(len(calls) - 1, len(seq) - 1)]

    agent.context.llm_client.chat_completion = AsyncMock(side_effect=_spy)
    body = {"messages": [{"role": "user", "content": user}],
            "model": "Qwen-Test"}
    with patch("ghost_agent.core.agent.pretty_log"):
        result, _, _ = await agent.handle_chat(
            body, background_tasks=MagicMock())
    return result, calls


# --------------------------------------------------------------------------
# Core regression: async mode repairs an unverified mutation in-loop
# --------------------------------------------------------------------------

async def test_async_mode_repairs_unverified_mutation(agent, monkeypatch):
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    agent.available_tools["file_system"] = AsyncMock(
        return_value="SUCCESS: wrote 120 bytes to app.py")

    result, calls = await _drive(
        agent, "create app.py that prints 1", [
            _tool_call(),
            _final("I wrote app.py — you can run it now."),  # untested write
            _final("I ran app.py; it executes cleanly."),    # repaired
        ])

    # The repair re-entry fired: a third LLM call happened and its
    # conversation carries the "UNVERIFIED change" directive.
    assert len(calls) == 3
    third = "\n".join(calls[-1])
    assert "UNVERIFIED change" in third
    assert "ran app.py" in result
    # The unrepaired optimistic answer did not ship.
    assert "you can run it now" not in result


async def test_async_mode_no_repair_when_write_was_run(agent, monkeypatch):
    # A write FOLLOWED by an execute is verified — no repair round.
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    agent.available_tools["file_system"] = AsyncMock(
        return_value="SUCCESS: wrote 120 bytes to app.py")
    agent.available_tools["execute"] = AsyncMock(
        return_value="OUTPUT: 1\nScript completed with exit code 0")

    result, calls = await _drive(
        agent, "create app.py that prints 1 and run it", [
            _tool_call(),
            _tool_call(name="execute",
                       args='{"content": "print(1)"}', tid="t2"),
            _final("Done — app.py runs and prints 1."),
        ])

    # The most recent substantive tool is the execute — verified evidence,
    # so no repair directive is ever injected.
    assert not any("UNVERIFIED change" in "\n".join(c) for c in calls)
    assert "Done" in result


async def test_async_mode_repair_budget_is_bounded(agent, monkeypatch):
    # The model keeps writing without running: exactly ONE repair round
    # (_MAX_VERIFIER_REPAIRS), then the answer ships with the post-loop note.
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    agent.available_tools["file_system"] = AsyncMock(
        return_value="SUCCESS: wrote 120 bytes to app.py")

    result, calls = await _drive(
        agent, "create app.py", [
            _tool_call(),
            _final("Wrote it."),                       # → repair round 1
            _tool_call(tid="t2"),                      # writes AGAIN
            _final("Rewrote it, still never ran it."),  # budget exhausted
        ])

    # Count directive MESSAGES in the final conversation (an injected
    # directive persists in `messages`, so counting calls would double).
    directives = sum(
        1 for m in calls[-1] if "UNVERIFIED change" in m)
    assert directives == 1  # bounded: no second in-loop re-entry
    assert "Rewrote it" in result


# --------------------------------------------------------------------------
# Async semantics preserved: the LLM verdict itself stays off the loop
# --------------------------------------------------------------------------

async def test_async_mode_never_blocks_on_inloop_verdict(agent, monkeypatch):
    # Even with a verifier installed, async mode must not await a blocking
    # in-loop verdict — verify_* is only reached via the post-loop gate.
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    # Pure-async gate: never wait for the background verdict.
    monkeypatch.setenv("GHOST_CRITIC_GATE_TIMEOUT", "0")
    verifier = MagicMock()
    verifier.llm_client = MagicMock()
    inloop_calls = AsyncMock(return_value=VerifyResult(
        verdict=VerifyVerdict.CONFIRMED, confidence=0.9,
        reasoning="", issues=[]))
    verifier.verify_claim = inloop_calls
    verifier.verify_code_output = inloop_calls
    verifier.verify_visual = AsyncMock(return_value=None)
    agent.context.verifier = verifier
    agent.available_tools["execute"] = AsyncMock(return_value="OUTPUT: 42")

    result, calls = await _drive(
        agent, "compute the answer", [
            _tool_call(name="execute", args='{"content": "print(42)"}'),
            _final("The answer is 42."),
        ])

    # No repair fired (execute is not a mutation) and no in-loop verdict
    # inserted a repair turn — async's non-blocking property itself is
    # covered by test_critic_async.py; here we assert the fix did not
    # smuggle a blocking repair into the non-mutation path.
    assert not any("UNVERIFIED change" in "\n".join(c) for c in calls)
    assert "42" in result


async def test_sync_mode_unchanged_by_the_fix(agent, monkeypatch):
    # Flag off → the original sync behaviour (repair fires there too).
    monkeypatch.delenv("GHOST_CRITIC_ASYNC", raising=False)
    agent.available_tools["file_system"] = AsyncMock(
        return_value="SUCCESS: wrote 120 bytes to app.py")

    result, calls = await _drive(
        agent, "create app.py that prints 1", [
            _tool_call(),
            _final("I wrote app.py."),
            _final("I ran app.py; it works."),
        ])

    assert len(calls) == 3
    assert "UNVERIFIED change" in "\n".join(calls[-1])
    assert "ran app.py" in result
