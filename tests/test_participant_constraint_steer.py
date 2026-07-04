"""Participant-mode constraint enforcement (2026-07-04 chess regression).

The user asked to "play chess against each other. with YOU, not a generated
chess AI"; the constraint was captured on project create — and then violated
three times (a positional engine, a random engine, a positional engine
again) because (a) the autoadvance coding executor's spec prompt NEVER saw
the project's stored constraints, and (b) the post-write constraint steer
carried no architecture guidance, so the model rationalised the engine as
"the evaluation function IS me".

Coverage here:
* ``has_participant_constraint`` — detection over the REAL captured clauses
  from the live session, plus negative controls.
* ``PARTICIPANT_STEER`` — names the two valid designs and /api/game/move.
* ``_generate_build_spec`` — the spec prompt now renders project constraints
  and appends the participant directive when one is detected.
* ``build_coding_task`` — forwards ``constraints=`` (it swallows unknown
  kwargs via ``**_ignored``, so a signature regression would fail silently).
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.utils.constraints import (
    PARTICIPANT_STEER,
    extract_constraints,
    has_participant_constraint,
)


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------

class TestHasParticipantConstraint:
    def test_live_session_create_message(self):
        # Verbatim from the 2026-07-04 session (request 38).
        msg = ("Create a new project, it's goal will be to play chess "
               "against each other. with YOU, not a generated chess AI , "
               "i want to play chess with you. don't use html / js etc.. "
               "make it turn based from the terminal.")
        assert has_participant_constraint(extract_constraints(msg))

    def test_live_session_follow_up(self):
        # Verbatim from request 21.
        msg = ("that works. now. instead of the game being against a python "
               "chess ai, i want you to change it so it's you, ghost that "
               "will read the moves and play against me.")
        assert has_participant_constraint(extract_constraints(msg))

    def test_stored_project_constraint(self):
        # As captured on the project record at create time.
        cons = ["with YOU - Ghost plays directly, not a generated chess AI",
                "don't use html / js etc"]
        assert has_participant_constraint(cons)

    def test_plain_negation_is_not_participant(self):
        cons = extract_constraints(
            "fix the bug in parser.py, don't touch the tests")
        assert not has_participant_constraint(cons)

    def test_your_does_not_false_positive(self):
        assert not has_participant_constraint(
            ["play a song with your speaker"])

    def test_empty_and_none_safe(self):
        assert not has_participant_constraint([])
        assert not has_participant_constraint(None)
        assert not has_participant_constraint([None, ""])


class TestParticipantSteerText:
    def test_names_the_endpoint(self):
        assert "/api/game/move" in PARTICIPANT_STEER
        assert "127.0.0.1:8000" in PARTICIPANT_STEER

    def test_names_both_designs_and_the_violation(self):
        s = PARTICIPANT_STEER
        assert "(A)" in s and "(B)" in s
        assert "random.choice" in s and "minimax" in s
        # The exact rationalisation seen live must be pre-empted.
        assert "sandbox" in s.lower()


# --------------------------------------------------------------------------
# Coding-executor spec prompt
# --------------------------------------------------------------------------

def _spec_llm(captured):
    """LLM mock that records the payload and returns a minimal valid spec."""
    async def _cc(payload, **kw):
        captured.append(payload)
        return {"choices": [{"message": {"content": json.dumps({
            "files": [{"path": "game.py", "content": "print('ok')\n"}],
            "verify": "", "summary": "s", "ledger": ""})}}]}
    return MagicMock(chat_completion=AsyncMock(side_effect=_cc))


class TestSpecPromptConstraints:
    async def _gen(self, constraints):
        from ghost_agent.core.coding_executor import _generate_build_spec
        captured = []
        llm = _spec_llm(captured)
        spec, was_empty = await _generate_build_spec(
            llm, "m", "Build the chess game task", "",
            constraints=constraints)
        user_msg = captured[0]["messages"][1]["content"]
        return spec, user_msg

    async def test_constraints_rendered_into_prompt(self):
        _, user = await self._gen(["don't use html / js etc"])
        assert "EXPLICIT USER CONSTRAINTS (PROJECT-WIDE)" in user
        assert "don't use html / js etc" in user

    async def test_participant_constraint_appends_steer(self):
        _, user = await self._gen(
            ["with YOU - Ghost plays directly, not a generated chess AI"])
        assert "PARTICIPANT-MODE ARCHITECTURE" in user
        assert "/api/game/move" in user

    async def test_non_participant_constraint_no_steer(self):
        _, user = await self._gen(["don't use html / js etc"])
        assert "PARTICIPANT-MODE ARCHITECTURE" not in user

    async def test_no_constraints_prompt_unchanged(self):
        _, user = await self._gen(None)
        assert "EXPLICIT USER CONSTRAINTS" not in user
        assert "PARTICIPANT-MODE ARCHITECTURE" not in user


class TestBuildCodingTaskForwarding:
    async def test_constraints_kwarg_reaches_spec_generation(self):
        from ghost_agent.core import coding_executor as ce
        cons = ["with YOU - Ghost plays directly"]
        seen = {}

        async def _fake_spec(llm, model, description, ledger, **kw):
            seen.update(kw)
            # Empty spec ends the attempt loop quickly.
            return {}, False

        ctx = MagicMock()
        ctx.llm_client = MagicMock()
        ctx.args.model = "m"
        with patch.object(ce, "_generate_build_spec", side_effect=_fake_spec):
            await ce.build_coding_task(
                ctx, "task", tool_runner=AsyncMock(return_value="ok"),
                constraints=cons, max_attempts=1)
        assert seen.get("constraints") == cons


# --------------------------------------------------------------------------
# In-turn post-write steer (agent.py)
# --------------------------------------------------------------------------

class TestInTurnSteer:
    @pytest.fixture
    def agent(self):
        from ghost_agent.core.agent import GhostAgent, GhostContext
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

    async def _drive(self, agent, user):
        calls = []
        seq = [
            {"choices": [{"message": {"content": "", "tool_calls": [
                {"id": "t1", "function": {"name": "file_system",
                 "arguments": json.dumps({
                     "operation": "write", "path": "game.py",
                     "content": "print(1)"})}}]}}]},
            {"choices": [{"message": {
                "content": "Done.", "tool_calls": []}}]},
        ]

        async def _spy(payload, **kw):
            calls.append([m.get("content", "")
                          for m in payload.get("messages", [])])
            return seq[min(len(calls) - 1, len(seq) - 1)]

        agent.available_tools["file_system"] = AsyncMock(
            return_value="SUCCESS: wrote 8 chars to 'game.py'.")
        agent.context.llm_client.chat_completion = AsyncMock(
            side_effect=_spy)
        body = {"messages": [{"role": "user", "content": user}],
                "model": "Qwen-Test"}
        with patch("ghost_agent.core.agent.pretty_log"):
            await agent.handle_chat(body, background_tasks=MagicMock())
        return calls

    async def test_participant_request_gets_architecture_steer(self, agent):
        calls = await self._drive(
            agent,
            "build a terminal chess game where YOU play against me, "
            "don't use html")
        joined = "\n".join("\n".join(c) for c in calls[1:])
        assert "SYSTEM ALERT (constraint check)" in joined
        assert "PARTICIPANT-MODE ARCHITECTURE" in joined
        assert "/api/game/move" in joined

    async def test_plain_constraint_request_no_architecture_steer(
            self, agent):
        calls = await self._drive(
            agent, "write game.py that prints 1, don't use numpy")
        joined = "\n".join("\n".join(c) for c in calls[1:])
        assert "SYSTEM ALERT (constraint check)" in joined
        assert "PARTICIPANT-MODE ARCHITECTURE" not in joined
