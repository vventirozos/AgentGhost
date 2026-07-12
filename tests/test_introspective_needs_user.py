"""Regression: an INTROSPECTIVE task must never be classified needs_user
(found live 2026-07-12).

`_NEEDS_USER_KEYWORDS` matches bare substrings, so a task ABOUT decision-making
was mistaken for a task REQUIRING a decision:

    "Illusion of Agency: Evaluate whether I truly 'choose' responses or merely
     predict them. Analyze decision-making as probabilistic sampling vs
     deterministic selection."
                          ^^^^^^ the word "choose" → NEEDS_USER

The task then JAMMED — autoadvance skips NEEDS_USER, so it could never be
advanced — and the agent burned THREE user requests (~4 min) investigating
before answering "I just need you to say proceed", which was useless AND wrong.

There is nothing for a human to decide in "analyse your own X". An explicit
[HUMAN_GATE: …] postcondition still wins (enforce_human_gate is separate).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.core.project_advancer import (
    advance_once, classify_task, is_self_referential,
)

# The exact task that jammed in production.
JAMMED = ("Illusion of Agency: Evaluate whether I truly 'choose' responses or "
          "merely predict them. Analyze decision-making as probabilistic "
          "sampling vs deterministic selection.")


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


def _ctx(analysis="## Agency\nSampling is not choosing."):
    llm = SimpleNamespace()
    llm.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": analysis}}]})
    return SimpleNamespace(llm_client=llm, args=SimpleNamespace(model="qwen"),
                           project_store=None)


class TestKeywordFalsePositive:
    def test_the_bare_keyword_still_matches(self):
        # classify_task itself is unchanged — it DOES see "choose". The guard
        # lives where BOTH classifier paths converge (advance_once), because
        # the LLM classifier mis-fires on this wording too.
        assert classify_task(JAMMED) == "needs_user"

    def test_but_the_task_is_recognised_as_introspective(self):
        assert is_self_referential(JAMMED) is True


class TestAdvanceOnceOverride:
    def _run(self, store, ctx, description, classifier=None):
        pid = store.create_project("Meta", kind="GENERAL")
        store.add_task(pid, description)
        ctx.project_store = store
        return pid, asyncio.run(advance_once(
            ctx, pid,
            tool_runner=AsyncMock(return_value="web results"),
            llm_classifier=classifier,
        ))

    def test_introspective_needs_user_is_overridden(self, store):
        """The headline: the jammed task now ADVANCES instead of stalling."""
        ctx = _ctx()
        pid, res = self._run(store, ctx, JAMMED)
        assert res.classification != "needs_user"
        tasks = store.list_tasks(pid)
        assert tasks[0]["status"] == "DONE"
        # …and it was answered from the agent's own knowledge, not web-searched.
        ctx.llm_client.chat_completion.assert_awaited()

    def test_override_applies_to_the_llm_classifier_path_too(self, store):
        # A GENERAL project consults the LLM classifier; it mis-fires on this
        # wording as well, so the guard must catch that path.
        ctx = _ctx()
        classifier = AsyncMock(return_value="needs_user")
        pid, res = self._run(store, ctx, JAMMED, classifier=classifier)
        assert res.classification != "needs_user"
        assert store.list_tasks(pid)[0]["status"] == "DONE"

    def test_genuine_human_decision_still_needs_user(self, store):
        # Not self-referential ⇒ the keyword is a TRUE positive. Don't
        # over-correct: a real approval task must still route to the human.
        ctx = _ctx()
        pid, res = self._run(
            store, ctx, "Choose the production database vendor and approve it")
        assert res.classification == "needs_user"
        assert store.list_tasks(pid)[0]["status"] == "NEEDS_USER"

    def test_explicit_human_gate_still_wins(self, store):
        """An introspective task carrying an EXPLICIT gate must still stop —
        enforce_human_gate is separate and must not be defeated."""
        ctx = _ctx()
        pid = store.create_project("Meta", kind="GENERAL")
        store.add_task(
            pid,
            "Analyze your own reasoning, then publish it "
            "[HUMAN_GATE: operator sign-off]",
            postconditions=["HUMAN_GATE: operator sign-off"],
        )
        ctx.project_store = store
        res = asyncio.run(advance_once(ctx, pid,
                                       tool_runner=AsyncMock(return_value="x")))
        assert res.classification == "needs_user"
        assert store.list_tasks(pid)[0]["status"] == "NEEDS_USER"


class TestSourceWiring:
    SRC = (Path(__file__).resolve().parents[1] / "src" / "ghost_agent"
           / "core" / "project_advancer.py").read_text()

    def test_override_is_before_the_needs_user_branch(self):
        i_guard = self.SRC.index(
            'if classification == "needs_user" and is_self_referential(')
        i_branch = self.SRC.index('if classification == "needs_user":')
        assert i_guard < i_branch
