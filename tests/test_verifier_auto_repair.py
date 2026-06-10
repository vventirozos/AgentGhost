"""Verifier-gate AUTO-REPAIR loop (bounded in-loop re-attempt).

When the verifier REFUTES the final answer (high confidence) — or the
turn finalised on an unverified mutation — the agent gets up to
`_MAX_VERIFIER_REPAIRS` extra in-loop attempts to FIX the issue (the
critique is injected and the turn loop re-runs via `continue`) instead
of shipping a noted-but-wrong answer. These tests drive the real
`handle_chat` with a mocked LLM + verifier and assert the re-entry,
the bound, the single-verifier-pass cost on the clean path, and that
error/abort answers are never repaired.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from ghost_agent.core.agent import GhostAgent, GhostContext
from ghost_agent.core.verifier import VerifyResult, VerifyVerdict


# --------------------------------------------------------------------------
# Fixture
# --------------------------------------------------------------------------

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
    # verifier is attached per-test (None by default → gate skipped).
    ctx.verifier = None
    return GhostAgent(ctx)


def _verdict(v, conf=0.95, issues=None, reasoning="reason"):
    return VerifyResult(verdict=v, confidence=conf, reasoning=reasoning, issues=issues or [])


def _make_verifier(verdicts):
    """A verifier mock whose verify_* calls return `verdicts` in order.
    The same counter backs verify_claim and verify_code_output so a test
    can assert the TOTAL number of verifier passes regardless of which
    shape the gate chose."""
    v = MagicMock()
    v.llm_client = MagicMock()  # truthy → gate active
    shared = AsyncMock(side_effect=list(verdicts))
    v.verify_claim = shared
    v.verify_code_output = shared
    v.verify_visual = AsyncMock(return_value=None)
    return v, shared


def _tool_call(name="execute", args='{"content": "print(40+2)"}', tid="t1"):
    return {"choices": [{"message": {"content": "working", "tool_calls": [
        {"id": tid, "function": {"name": name, "arguments": args}}]}}]}


def _final(text):
    return {"choices": [{"message": {"content": text, "tool_calls": []}}]}


async def _run(agent, user, llm_side_effects):
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=list(llm_side_effects))
    body = {"messages": [{"role": "user", "content": user}], "model": "Qwen-Test"}
    with patch("ghost_agent.core.agent.pretty_log"):
        result, _, _ = await agent.handle_chat(body, background_tasks=MagicMock())
    return result


# --------------------------------------------------------------------------
# Core: REFUTED → repair → CONFIRMED
# --------------------------------------------------------------------------

async def test_refuted_then_repaired_ships_corrected_answer(agent):
    agent.available_tools["execute"] = AsyncMock(return_value="OUTPUT: 7")
    verifier, vmock = _make_verifier([
        _verdict(VerifyVerdict.REFUTED, issues=["wrong number"]),  # turn-2 finalisation
        _verdict(VerifyVerdict.CONFIRMED),                          # turn-3 (post-repair)
    ])
    agent.context.verifier = verifier

    result = await _run(agent, "compute the answer and report it", [
        _tool_call(),
        _final("The answer is 7."),         # refuted
        _final("Corrected: the answer is 42."),  # repaired → confirmed
    ])

    assert "42" in result          # shipped the repaired answer
    assert "The answer is 7." not in result  # not the refuted one
    assert vmock.await_count == 2  # one verdict per finalisation, no double-call


async def test_repair_injects_critique_into_conversation(agent):
    agent.available_tools["execute"] = AsyncMock(return_value="OUTPUT: 7")
    verifier, _ = _make_verifier([
        _verdict(VerifyVerdict.REFUTED, issues=["off by a lot"]),
        _verdict(VerifyVerdict.CONFIRMED),
    ])
    agent.context.verifier = verifier

    captured = {}
    orig = agent.context.llm_client
    # Spy on the messages handed to the 3rd (post-repair) LLM call.
    calls = []
    async def _spy(payload, *a, **k):
        calls.append([m.get("content", "") for m in payload.get("messages", [])])
        seq = [_tool_call(), _final("7"), _final("Corrected: 42")]
        return seq[len(calls) - 1]
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=_spy)
    body = {"messages": [{"role": "user", "content": "compute it"}], "model": "Qwen-Test"}
    with patch("ghost_agent.core.agent.pretty_log"):
        await agent.handle_chat(body, background_tasks=MagicMock())

    # The 3rd call's message history must contain the verifier critique.
    third = "\n".join(calls[-1])
    assert "REFUTED" in third and "off by a lot" in third


# --------------------------------------------------------------------------
# Clean path: CONFIRMED → no repair, exactly one verifier pass
# --------------------------------------------------------------------------

async def test_confirmed_no_repair_single_verifier_pass(agent):
    agent.available_tools["execute"] = AsyncMock(return_value="OUTPUT: 42")
    verifier, vmock = _make_verifier([_verdict(VerifyVerdict.CONFIRMED)])
    agent.context.verifier = verifier

    result = await _run(agent, "compute the answer", [
        _tool_call(),
        _final("The answer is 42."),
    ])

    assert "42" in result
    # Verdict computed once at finalisation and REUSED post-loop (cache):
    # no double LLM verifier call on the clean success path.
    assert vmock.await_count == 1


# --------------------------------------------------------------------------
# Bound: budget is respected (no infinite repair)
# --------------------------------------------------------------------------

async def test_repair_budget_is_bounded(agent):
    agent.available_tools["execute"] = AsyncMock(return_value="OUTPUT: 7")
    # Verifier REFUTES every time. With _MAX_VERIFIER_REPAIRS == 1 there
    # must be exactly ONE repair, then the answer ships (with the note).
    verifier, vmock = _make_verifier([
        _verdict(VerifyVerdict.REFUTED, issues=["still wrong"]),
        _verdict(VerifyVerdict.REFUTED, issues=["still wrong"]),
        _verdict(VerifyVerdict.REFUTED, issues=["still wrong"]),
    ])
    agent.context.verifier = verifier

    result = await _run(agent, "compute the answer", [
        _tool_call(),
        _final("Attempt one: 7."),
        _final("Attempt two: 8."),
        _final("Attempt three: 9."),   # should never be reached
    ])

    # Exactly one repair: the 2nd finalisation (turn 3) does not repair
    # (budget spent), so the post-loop gate verifies it → 2 verifier passes.
    assert vmock.await_count == 2
    assert "Attempt two: 8." in result   # the single repaired answer ships
    assert "Attempt three" not in result
    assert agent._MAX_VERIFIER_REPAIRS == 1


# --------------------------------------------------------------------------
# Unverified mutation → repair (works even with no verifier installed)
# --------------------------------------------------------------------------

async def test_unverified_mutation_triggers_repair_without_verifier(agent):
    # No verifier; the last action is a file write that was never run.
    agent.context.verifier = None
    agent.available_tools["file_system"] = AsyncMock(return_value="SUCCESS: wrote 120 bytes to app.py")

    calls = []
    async def _spy(payload, *a, **k):
        calls.append([m.get("content", "") for m in payload.get("messages", [])])
        seq = [
            _tool_call(name="file_system", args='{"operation": "write", "path": "app.py", "content": "x=1"}'),
            _final("I wrote app.py."),               # unverified mutation → repair
            _final("I ran app.py; it executes cleanly."),  # repaired
        ]
        return seq[len(calls) - 1]
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=_spy)
    body = {"messages": [{"role": "user", "content": "create app.py that prints 1"}], "model": "Qwen-Test"}
    with patch("ghost_agent.core.agent.pretty_log"):
        result, _, _ = await agent.handle_chat(body, background_tasks=MagicMock())

    # A repair round fired: the "UNVERIFIED change" directive was injected
    # and the loop re-ran to a second answer.
    assert len(calls) == 3
    third = "\n".join(calls[-1])
    assert "UNVERIFIED change" in third
    assert "ran app.py" in result


# --------------------------------------------------------------------------
# Safety: error/abort answers are never repaired
# --------------------------------------------------------------------------

async def test_no_repair_on_trivial_chat(agent):
    # A strict trivial greeting never reaches the full loop / verifier.
    verifier, vmock = _make_verifier([_verdict(VerifyVerdict.REFUTED)])
    agent.context.verifier = verifier
    result = await _run(agent, "hi", [_final("Hello!")])
    assert vmock.await_count == 0  # gate never engaged
