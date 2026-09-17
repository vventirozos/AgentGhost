"""The no-progress steer names the held write (§4HE, 2026-09-15).

Request 5fa6aa97: the model composed an 18,875-char rewrite of its report
as `replace` without `replace_with`; the tool rejected it and HELD the
payload (§4HC) with a one-line redemption route. The model re-read the
file twice instead; the breaker steered it to "the mutating action" in the
abstract; it then delivered the answer with nothing written. The steer now
names the exact pending write when the tool is holding one for that path.
Executed end to end through `handle_chat`; each pin names the world it
fails in.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core.agent import GhostAgent
from ghost_agent.tools import file_system as FS

_PATH = "revolut_report.md"
_REQ = {"messages": [{"role": "user", "content":
                      "Update the forensic report with the KELA findings and save it."}],
        "model": "Qwen-Test"}


@pytest.fixture(autouse=True)
def _no_leftover_holds():
    # The hold store is module-global and keyed by resolved path; the
    # mock context's sandbox is the same dir across tests, so a hold
    # planted by one pin would be the "pending write" of the next.
    FS._HELD_REPLACE_CONTENT.clear()
    yield
    FS._HELD_REPLACE_CONTENT.clear()


@pytest.fixture
def agent(mock_context):
    return GhostAgent(mock_context)


def _drive(agent, n_reads=3):
    seen = []
    state = {"n": 0}

    async def fake(payload, *a, **kw):
        seen.append(payload)
        state["n"] += 1
        if state["n"] <= n_reads:
            return {"choices": [{"message": {"content": None, "tool_calls": [{
                "id": f"c{state['n']}",
                "function": {"name": "file_system",
                             "arguments": '{"operation": "read", "path": "%s"}' % _PATH}}]}}]}
        return {"choices": [{"message": {"content": "Done.", "tool_calls": []}}]}

    agent.context.llm_client.chat_completion = AsyncMock(side_effect=fake)
    agent.available_tools["file_system"] = AsyncMock(
        return_value=f"--- {_PATH} CONTENTS ---\n# Report\nsame text every time\n")
    return seen


def _steers(seen):
    """The SYSTEM ALERT messages THIS drive appended: whatever the last
    LLM payload carries that the first did not. The mock context keeps
    conversation history across tests, so an earlier pin's steer would
    otherwise be read as this pin's."""
    first = seen[0].get("messages", [])
    return [m["content"] for m in seen[-1].get("messages", [])
            if m not in first and m.get("role") == "user"
            and "SYSTEM ALERT" in str(m.get("content", ""))]


@pytest.mark.asyncio
async def test_a_pending_hold_is_named_in_the_steer(agent):
    """FAILS IF: the steer stays abstract ("call the mutating action") —
    the live world."""
    sandbox = FS.project_scoped_sandbox(agent.context)[0]
    FS._hold_rejected_content(sandbox, _PATH, "# Report\n" + "x" * 500)
    seen = _drive(agent)
    await agent.handle_chat(_REQ, background_tasks=MagicMock())
    steers = _steers(seen)
    assert steers, "the no-progress steer never fired"
    assert any("content='<<HELD>>'" in s and _PATH in s and "509-char" in s for s in steers), steers[-1]


@pytest.mark.asyncio
async def test_no_hold_no_redemption_line(agent):
    """FAILS IF: the line is emitted unconditionally — a model with nothing
    held would be told to redeem a write that does not exist."""
    seen = _drive(agent)
    await agent.handle_chat(_REQ, background_tasks=MagicMock())
    steers = _steers(seen)
    assert steers, "the no-progress steer never fired"
    assert not any("<<HELD>>" in s for s in steers)


def test_the_peek_does_not_consume_the_hold(tmp_path):
    """FAILS IF: naming the hold redeems it — the model's `<<HELD>>` write
    would then be rejected as stale."""
    FS._hold_rejected_content(tmp_path, "a.md", "payload")
    assert FS.held_content_chars(tmp_path, "a.md") == 7
    assert FS.held_content_chars(tmp_path, "a.md") == 7
    assert FS._redeem_held_content(tmp_path, "a.md") == "payload"
    assert FS.held_content_chars(tmp_path, "a.md") == 0


def test_an_expired_hold_reads_as_absent(tmp_path, monkeypatch):
    """FAILS IF: the peek ignores the TTL — it would name a write the
    redeem path will refuse."""
    FS._hold_rejected_content(tmp_path, "b.md", "payload")
    real = FS.time.monotonic
    monkeypatch.setattr(FS.time, "monotonic", lambda: real() + FS._HELD_CONTENT_TTL_S + 1)
    assert FS.held_content_chars(tmp_path, "b.md") == 0
