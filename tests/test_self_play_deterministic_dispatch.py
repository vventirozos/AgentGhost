"""Deterministic terminal-tool dispatch for explicit self-play commands.

Prior bug: a bare "self play" ran the cycle only on the FIRST ask. On
every repeat ("self play again") the agent replayed the previous turn's
text summary instead of re-firing the tool. Root cause: the
direct-from-tool-summary bypass persists the cycle result as a PLAIN-TEXT
assistant turn with NO tool_call, and the memory bus re-hydrates it as
"Context: Self-play complete…", so the model's only in-context example
says the right response to "self play" is to reprint that text — which it
does (identical ~126-char reply, tool=- in the metacog trace), never
re-running.

Fix: when the user's message is unambiguously a single-cycle self-play
command, the agent dispatches the `self_play` tool deterministically
BEFORE the LLM turn and records it in history as a real tool_call +
result. These tests pin both halves: the pure intent gate, and the
end-to-end "runs every time, even with poisoned history" behaviour.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ghost_agent.core.agent import (
    GhostAgent,
    GhostContext,
    _is_single_self_play_command,
    _is_dream_command,
    _explicit_terminal_command,
)


# --------------------------------------------------------------------------
# Pure intent gate
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "self play",
    "self play again",
    "self-play",
    "selfplay",
    "run self-play",
    "do another self play cycle please",
    "run a practice cycle",
    "training cycle",
])
def test_single_self_play_command_matches(text):
    assert _is_single_self_play_command(text) is True


@pytest.mark.parametrize("text", [
    "",
    "what is self play and how does it work in reinforcement learning?",  # prose mention
    "keep practicing self play until i stop you",                          # loop intent
    "run self play in a loop continuously",                                # loop intent
    "self play over and over",                                             # loop intent
    "build me a flask app",                                                # unrelated
    # Mentions self-play but buried in a long task message → leave to model:
    "before you start, note that the self play module needs a new test harness wired",
])
def test_non_single_self_play_command_rejected(text):
    assert _is_single_self_play_command(text) is False


@pytest.mark.parametrize("text", [
    "dream mode",
    "go to sleep",
    "time to sleep",
    "consolidate memories",
    "consolidate your memories",
    "memory consolidation",
    "go to sleep now please",
])
def test_dream_command_matches(text):
    assert _is_dream_command(text) is True


@pytest.mark.parametrize("text", [
    "",
    "i had an interesting dream about a flask app last night",  # prose, >6 words + "rest"/"dream" traps
    "tell me how memory consolidation works in the brain in detail",  # prose mention
    "build me a flask app",                                          # unrelated
])
def test_non_dream_command_rejected(text):
    assert _is_dream_command(text) is False


def test_explicit_terminal_command_routing():
    assert _explicit_terminal_command("self play again") == "self_play"
    assert _explicit_terminal_command("dream mode") == "dream_mode"
    assert _explicit_terminal_command("go to sleep") == "dream_mode"
    assert _explicit_terminal_command("build me a flask app") is None
    # Loop phrasing is not a single-cycle command → left to the model.
    assert _explicit_terminal_command("keep practicing in a loop") is None


# --------------------------------------------------------------------------
# End-to-end dispatch
# --------------------------------------------------------------------------

@pytest.fixture
def agent():
    context = MagicMock(spec=GhostContext)
    context.llm_client = MagicMock()
    context.llm_client.vision_clients = None
    # No worker pool → query-expansion / routing helpers take their legacy
    # synchronous fallback instead of awaiting a MagicMock `route`.
    context.llm_client.worker_clients = None
    context.memory_bus = None
    context.sandbox_dir = "/tmp/sandbox"
    context.args = MagicMock()
    context.args.shell = "bash"
    context.args.max_context = 8000
    context.args.temperature = 0.5
    context.args.smart_memory = 0.0
    context.args.use_planning = False
    context.args.model = "qwen3.6"
    context.args.perfect_it = False
    context.profile_memory = MagicMock()
    context.profile_memory.get_context_string.return_value = ""
    context.memory_system = None
    context.skill_memory = None
    context.scratchpad = MagicMock()
    context.scratchpad.list_all.return_value = ""
    return GhostAgent(context)


class FakeBgTasks:
    def add_task(self, *a, **k):
        pass


_SELF_PLAY_RESULT = (
    "Synthetic Self-Play cycle completed. Final Status: SUCCESS.\n"
    "SELF-PLAY POST-MORTEM REPORT:\n"
    "Challenge: python_general\n"
    "Status: SUCCESS (in 1 attempts)\n"
    "Cluster: python_general  Compression delta: +0.5\n"
    "Skill gate: cluster mastered — skipping skill write.\n\n"
    "CURIOSITY: cluster=python_general compression_delta=+0.5\n\n"
    "SYSTEM INSTRUCTION: ... DO NOT call the `self_play` tool again."
)


@pytest.mark.asyncio
async def test_explicit_self_play_dispatches_without_consulting_llm(agent):
    """A bare "self play" must run the tool deterministically and never
    even reach the main LLM — the model's indecision can't swallow it."""
    agent.context.llm_client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": "ignored", "tool_calls": []}}]}
    )
    self_play_mock = AsyncMock(return_value=_SELF_PLAY_RESULT)
    agent.available_tools = {"self_play": self_play_mock}

    body = {"messages": [{"role": "user", "content": "self play"}]}

    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[{"function": {"name": "self_play"}}]):
        await agent.handle_chat(body, FakeBgTasks())

    assert self_play_mock.call_count == 1, "self_play should run exactly once"
    assert agent.context.llm_client.chat_completion.call_count == 0, (
        "deterministic dispatch must bypass the main LLM turn entirely"
    )


@pytest.mark.asyncio
async def test_self_play_reruns_even_with_poisoned_history(agent):
    """The reported regression: a prior 'Self-play complete…' text summary
    in history (the memory-bus / in-context priming) must NOT stop a repeat
    'self play again' from actually re-firing the tool."""
    # If the LLM were consulted it would (buggily) replay the summary text.
    agent.context.llm_client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {
            "content": "Self-play complete.\n\nChallenge: python_general — SUCCESS (in 1 attempts).",
            "tool_calls": [],
        }}]}
    )
    self_play_mock = AsyncMock(return_value=_SELF_PLAY_RESULT)
    agent.available_tools = {"self_play": self_play_mock}

    body = {"messages": [
        {"role": "user", "content": "self play"},
        {"role": "assistant", "content": "Self-play complete.\n\nChallenge: python_general — SUCCESS (in 1 attempts)."},
        {"role": "user", "content": "self play again"},
    ]}

    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[{"function": {"name": "self_play"}}]):
        await agent.handle_chat(body, FakeBgTasks())

    assert self_play_mock.call_count == 1, (
        "self_play must re-fire on the repeat ask despite the poisoned history"
    )
    assert agent.context.llm_client.chat_completion.call_count == 0


@pytest.mark.asyncio
async def test_dispatch_returns_distilled_summary(agent):
    """The deterministic path must surface the distilled, user-facing
    self-play summary (the cluster/status line), not the raw blob with its
    CURIOSITY / SYSTEM INSTRUCTION trailers."""
    self_play_mock = AsyncMock(return_value=_SELF_PLAY_RESULT)
    agent.available_tools = {"self_play": self_play_mock}

    body = {"messages": [{"role": "user", "content": "self play"}]}

    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[{"function": {"name": "self_play"}}]):
        result = await agent.handle_chat(body, FakeBgTasks())

    assert self_play_mock.call_count == 1
    final_content = result[0] if isinstance(result, tuple) else result
    assert isinstance(final_content, str)
    assert "Self-play complete." in final_content
    assert "python_general" in final_content
    # Internal trailers meant for other consumers must not leak to the user.
    assert "SYSTEM INSTRUCTION" not in final_content
    assert "CURIOSITY" not in final_content


@pytest.mark.asyncio
async def test_loop_request_is_left_to_the_model(agent):
    """A continuous-loop phrasing ('keep practicing … in a loop') must NOT
    be force-dispatched as a single cycle — it routes through the model so
    it can pick self_play_loop."""
    agent.context.llm_client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": "Okay.", "tool_calls": []}}]}
    )
    self_play_mock = AsyncMock(return_value=_SELF_PLAY_RESULT)
    agent.available_tools = {"self_play": self_play_mock}

    body = {"messages": [{"role": "user", "content": "keep practicing self play in a loop"}]}

    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[{"function": {"name": "self_play"}}]):
        await agent.handle_chat(body, FakeBgTasks())

    # Deterministic single-cycle dispatch must NOT have fired; the model
    # turn ran instead (it chose plain text in this mock, which is fine —
    # the point is the fast path stayed out of the way).
    assert self_play_mock.call_count == 0
    assert agent.context.llm_client.chat_completion.call_count >= 1


_DREAM_RESULT = (
    "Active Memory Consolidation complete. Merged 3 episodic memories.\n\n"
    "SYSTEM: SESSION FINISHED. STAND BY."
)


@pytest.mark.asyncio
async def test_explicit_dream_dispatches_without_consulting_llm(agent):
    """dream_mode is the other terminal tool subject to the same replay bug;
    an explicit 'go to sleep' must dispatch deterministically and bypass the
    LLM, even with a poisoned prior-summary turn in history."""
    agent.context.llm_client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {
            "content": "Dream cycle complete.\n\nActive Memory Consolidation complete.",
            "tool_calls": [],
        }}]}
    )
    dream_mock = AsyncMock(return_value=_DREAM_RESULT)
    agent.available_tools = {"dream_mode": dream_mock}

    body = {"messages": [
        {"role": "user", "content": "go to sleep"},
        {"role": "assistant", "content": "Dream cycle complete.\n\nActive Memory Consolidation complete."},
        {"role": "user", "content": "go to sleep"},
    ]}

    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[{"function": {"name": "dream_mode"}}]):
        result = await agent.handle_chat(body, FakeBgTasks())

    assert dream_mock.call_count == 1
    assert agent.context.llm_client.chat_completion.call_count == 0
    final_content = result[0] if isinstance(result, tuple) else result
    assert isinstance(final_content, str)
    assert "Dream cycle complete." in final_content
    # The STAND BY trailer is for other consumers — must not leak.
    assert "STAND BY" not in final_content
