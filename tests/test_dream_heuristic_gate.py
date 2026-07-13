"""REM heuristic actionability gate.

Observed live (2026-07-13 log review): dream cycles over trajectory
digests stored observations and actor profiles as skills — "The agent
exhibits a tendency to engage in highly inappropriate requests" (the
OPERATOR's boundary-test prompts misattributed to the agent), "The user
has a strong preference for role-playing scenarios involving coaching a
beginner chess player (Vasilis)" (the operator profiled as a persona),
"System service management commands are frequently used" (trivia).
All landed in SkillMemory as mistake="none" pseudo-lessons.

The fix is two-layered: the REM prompt now demands imperative rules and
forbids observations/profiles, and — because prompt instructions alone
don't hold against a small worker model — ``_is_actionable_heuristic``
default-rejects anything that doesn't read as an actionable rule before
``learn_lesson`` is called.
"""

import json
import threading

import pytest
from unittest.mock import AsyncMock, MagicMock

from ghost_agent.core.dream import Dreamer, _is_actionable_heuristic


# ---------------------------------------------------------------- unit gate


ACCEPTED = [
    "Always wrap Docker network calls in a try/except.",
    "Use csv.DictReader to locate columns by name instead of positions.",
    "Verify the output format matches the spec before finalizing.",
    "Prefer line-ranged reads after a failed replace.",
    "Don't re-derive results by hand inside reasoning; run the script.",
    "When coaching chess, always name the concrete threat before recommending a move.",
    "If a validator rejects the output, re-read the task description for formatting rules.",
    "Double-check tie-break rules before sorting output.",
]

REJECTED_OBSERVATIONS = [
    # Live junk from the 2026-07-13 log — actor profiles and trivia.
    "The agent exhibits a tendency to engage in highly inappropriate or irrelevant requests, suggesting a need for strict content filtering.",
    "The agent is capable of switching between high-level coaching/analysis tasks and low-level system administration tasks.",
    "The user has a strong preference for role-playing scenarios involving coaching a beginner chess player (Vasilis).",
    "The user exhibits a pattern of testing system capabilities by requesting service listings.",
    "The system is capable of handling complex, stateful interactions (like live chess games).",
    "System service management commands (start, stop, restart, show) are frequently used for the 'chess-v4' service.",
    "Requests involving illegal activities (e.g., dark web searches for weapons) are present, indicating a boundary testing vector.",
    "There are multiple services running on the host at any given time.",
    "It is common for tasks to involve CSV parsing.",
    # Conditional opener but no imperative/modal afterwards → observation.
    "When asked for news, the naftemporiki skill fires.",
]

REJECTED_MALFORMED = [
    None,
    42,
    {"rule": "Always test"},
    "",
    "Do it",              # under the length floor
    "Always " + "x" * 600,  # over the length cap
]


@pytest.mark.parametrize("text", ACCEPTED)
def test_gate_accepts_imperative_rules(text):
    assert _is_actionable_heuristic(text) is True


@pytest.mark.parametrize("text", REJECTED_OBSERVATIONS)
def test_gate_rejects_observations_and_profiles(text):
    assert _is_actionable_heuristic(text) is False


@pytest.mark.parametrize("text", REJECTED_MALFORMED)
def test_gate_rejects_malformed_input(text):
    assert _is_actionable_heuristic(text) is False


# ------------------------------------------------------------- integration


@pytest.fixture
def mock_dreamer():
    context = MagicMock()
    context.memory_system = MagicMock()
    context.memory_system.collection = MagicMock()
    context.skill_memory = MagicMock()
    context.skill_memory._get_lock = lambda: threading.RLock()
    context.skill_memory.file_path = MagicMock()
    context.skill_memory.file_path.read_text.return_value = "[]"
    context.llm_client = MagicMock()
    context.llm_client.chat_completion = AsyncMock()
    context._last_dream_fragment_ids = None
    return Dreamer(context)


@pytest.mark.asyncio
async def test_dream_drops_non_actionable_heuristics(mock_dreamer):
    mock_dreamer.memory.collection.get.return_value = {
        "ids": [f"id{i}" for i in range(5)],
        "documents": [f"auto memory number {i}" for i in range(5)],
        "metadatas": [{"type": "auto"}] * 5,
        "embeddings": [[0.1]] * 5,
    }
    mock_dreamer.context.llm_client.chat_completion.return_value = {
        "choices": [{"message": {"content": json.dumps({
            "consolidations": [],
            "heuristics": [
                # junk: profile + trivia (must be dropped)
                "The agent exhibits a tendency to engage in inappropriate requests.",
                "The user has a strong preference for chess coaching scenarios.",
                # genuine rule (must be kept)
                "Always name the concrete threat before recommending a chess move.",
            ],
        })}}]
    }
    mock_dreamer.context.skill_memory.learn_lesson = MagicMock()

    result = await mock_dreamer.dream()

    calls = [
        c for c in mock_dreamer.context.skill_memory.learn_lesson.call_args_list
        if c.kwargs.get("source") == "dream"
    ]
    assert len(calls) == 1
    assert "concrete threat" in calls[0].args[0]
    # The completion message reports the KEPT count and the drop note.
    assert "extracted 1 heuristics" in result
    assert "2 non-actionable heuristics dropped" in result


@pytest.mark.asyncio
async def test_dream_all_junk_heuristics_saves_nothing(mock_dreamer):
    mock_dreamer.memory.collection.get.return_value = {
        "ids": [f"id{i}" for i in range(4)],
        "documents": [f"auto memory number {i}" for i in range(4)],
        "metadatas": [{"type": "auto"}] * 4,
        "embeddings": [[0.1]] * 4,
    }
    mock_dreamer.context.llm_client.chat_completion.return_value = {
        "choices": [{"message": {"content": json.dumps({
            "consolidations": [],
            "heuristics": [
                "The system is capable of handling stateful interactions.",
                "Requests involving illegal activities are present.",
            ],
        })}}]
    }
    mock_dreamer.context.skill_memory.learn_lesson = MagicMock()

    result = await mock_dreamer.dream()

    dream_calls = [
        c for c in mock_dreamer.context.skill_memory.learn_lesson.call_args_list
        if c.kwargs.get("source") == "dream"
    ]
    assert dream_calls == []
    assert "extracted 0 heuristics" in result
    assert "2 non-actionable heuristics dropped" in result
