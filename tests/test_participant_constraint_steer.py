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


# --------------------------------------------------------------------------
# Engine-pattern detection (2026-07-05: the deterministic write guard)
# --------------------------------------------------------------------------

from ghost_agent.utils.constraints import (  # noqa: E402
    find_engine_patterns,
    participant_write_violation,
)

# Condensed from the ACTUAL violating artifact of the 2026-07-05 session
# (projects/bb181a973669/terminal_chess.py, ghost_choose_move).
LIVE_ENGINE_SNIPPET = """
import chess

def ghost_choose_move(board):
    legal_moves = list(board.legal_moves)
    captures = [m for m in legal_moves if board.is_capture(m)]
    if captures:
        import random
        return random.choice(captures)
    import random
    return random.choice(legal_moves)
"""

# The sanctioned design (B) thin client — must NEVER trip the guard.
THIN_CLIENT_SNIPPET = """
import chess
import requests

def ghost_move(board, history):
    r = requests.post("http://127.0.0.1:8000/api/game/move",
                      json={"fen": board.fen(), "history": history},
                      timeout=180)
    r.raise_for_status()
    data = r.json()
    board.push(board.parse_san(data["move"]))
    return data
"""

PARTICIPANT_CONS = [
    "with YOU - Ghost plays directly, not a generated chess AI",
]


class TestFindEnginePatterns:
    def test_live_artifact_random_picker_flagged(self):
        hits = find_engine_patterns(LIVE_ENGINE_SNIPPET)
        assert any("random" in h for h in hits)

    def test_minimax_flagged_anywhere(self):
        assert find_engine_patterns("def minimax(board, depth): pass")

    def test_alpha_beta_and_piece_square_flagged(self):
        hits = find_engine_patterns(
            "// alpha-beta search over PIECE_SQUARE tables")
        assert len(hits) == 2

    def test_stockfish_flagged(self):
        assert find_engine_patterns("engine = Stockfish(path)")

    def test_math_random_in_js_board_flagged(self):
        content = ("<script src='chess.js'></script><script>"
                   "const m = moves[Math.floor(Math.random()*moves.length)];"
                   "</script>")
        assert find_engine_patterns(content)

    def test_coin_toss_without_game_context_is_clean(self):
        # random.* alone (no move-generation markers) must not trip the
        # guard — benign randomness in unrelated scripts stays writable.
        assert find_engine_patterns(
            "import random\nside = random.choice(['heads', 'tails'])") == []

    def test_thin_client_is_clean(self):
        assert find_engine_patterns(THIN_CLIENT_SNIPPET) == []

    def test_empty_and_none_safe(self):
        assert find_engine_patterns("") == []
        assert find_engine_patterns(None) == []


class TestParticipantWriteViolation:
    def test_blocks_live_engine_write(self):
        msg = participant_write_violation(
            PARTICIPANT_CONS,
            {"operation": "write", "path": "terminal_chess.py",
             "content": LIVE_ENGINE_SNIPPET})
        assert msg is not None
        assert "SYSTEM BLOCK" in msg
        assert "/api/game/move" in msg          # names the sanctioned design
        assert "terminal_chess.py" in msg

    def test_replace_with_body_is_scanned(self):
        # replace→write auto-promotion means replace_with can become the
        # whole file body — it must be scanned too.
        msg = participant_write_violation(
            PARTICIPANT_CONS,
            {"operation": "replace", "path": "game.py",
             "content": "old text",
             "replace_with": "def minimax(b, d): pass"})
        assert msg is not None

    def test_no_participant_constraint_never_blocks(self):
        assert participant_write_violation(
            ["don't use html / js etc"],
            {"operation": "write", "path": "t.py",
             "content": LIVE_ENGINE_SNIPPET}) is None

    def test_clean_thin_client_not_blocked(self):
        assert participant_write_violation(
            PARTICIPANT_CONS,
            {"operation": "write", "path": "client.py",
             "content": THIN_CLIENT_SNIPPET}) is None

    def test_non_write_operations_ignored(self):
        assert participant_write_violation(
            PARTICIPANT_CONS,
            {"operation": "read", "path": "terminal_chess.py"}) is None

    def test_empty_inputs_safe(self):
        assert participant_write_violation([], {"operation": "write",
                                                "content": "x"}) is None
        assert participant_write_violation(PARTICIPANT_CONS, None) is None
        assert participant_write_violation(
            PARTICIPANT_CONS, {"operation": "write", "content": ""}) is None


# --------------------------------------------------------------------------
# Project-constraint replay into the request (agent.py helpers)
# --------------------------------------------------------------------------

from types import SimpleNamespace  # noqa: E402


class TestProjectConstraintHelpers:
    def _fake_agent(self, constraints=None, pid="p1", store_raises=False):
        from ghost_agent.core.agent import GhostAgent
        store = MagicMock()
        if store_raises:
            store.get_project.side_effect = RuntimeError("db gone")
        else:
            store.get_project.return_value = {
                "metadata": {"constraints": list(constraints or [])}}
        fake = SimpleNamespace(
            context=SimpleNamespace(project_store=store,
                                    current_project_id=pid))
        fake._project_constraints_for = (
            lambda p, limit=5: GhostAgent._project_constraints_for(
                fake, p, limit))
        fake._active_project_constraints = (
            lambda limit=5: GhostAgent._active_project_constraints(
                fake, limit))
        fake._store = store
        return fake

    def test_returns_stored_constraints(self):
        fake = self._fake_agent(["no html", "with YOU - Ghost plays"])
        assert fake._active_project_constraints() == [
            "no html", "with YOU - Ghost plays"]

    def test_empty_without_bound_project(self):
        fake = self._fake_agent(["no html"], pid=None)
        assert fake._active_project_constraints() == []

    def test_store_error_is_safe(self):
        fake = self._fake_agent(store_raises=True)
        assert fake._active_project_constraints() == []

    def test_merge_dedups_and_arms_steer(self):
        from ghost_agent.core.agent import GhostAgent
        fake = self._fake_agent(["No HTML", "with YOU - Ghost plays"])
        with patch("ghost_agent.core.agent.pretty_log"):
            merged, block, pending = GhostAgent._merge_project_constraints(
                fake, ["no html"])  # dup differs only by case
        assert merged == ["no html", "with YOU - Ghost plays"]
        assert "EXPLICIT USER CONSTRAINTS (CURRENT REQUEST)" in block
        assert pending is True

    def test_merge_empty_everything(self):
        from ghost_agent.core.agent import GhostAgent
        fake = self._fake_agent([])
        merged, block, pending = GhostAgent._merge_project_constraints(
            fake, [])
        assert merged == [] and block == "" and pending is False

    def test_merge_replays_project_referenced_by_path(self):
        # The C5 2026-07-05 escape: traceback pasted into a FRESH chat
        # (no binding, pid=None) but naming the project on every line.
        from ghost_agent.core.agent import GhostAgent
        fake = self._fake_agent(
            ["with YOU - Ghost plays directly, not a coded AI"], pid=None)
        traceback_text = (
            'Traceback (most recent call last):\n  File '
            '"/Users/x/Data/AI/Data/sandbox/projects/6d0dd7371d17/chess/'
            'chess_client.py", line 41, in post_move\nTimeoutError: timed out')
        with patch("ghost_agent.core.agent.pretty_log"):
            merged, block, pending = GhostAgent._merge_project_constraints(
                fake, [], traceback_text)
        assert merged == ["with YOU - Ghost plays directly, not a coded AI"]
        assert pending is True
        fake._store.get_project.assert_any_call("6d0dd7371d17")

    def test_merge_bound_and_referenced_dedup(self):
        # Bound project and path-referenced project return the same
        # clauses — must not double up.
        from ghost_agent.core.agent import GhostAgent
        fake = self._fake_agent(["No HTML"], pid="p1")
        with patch("ghost_agent.core.agent.pretty_log"):
            merged, _, _ = GhostAgent._merge_project_constraints(
                fake, [], "see projects/abcdef123456/main.py")
        assert merged == ["No HTML"]

    def test_merge_plain_text_no_reference_no_binding(self):
        from ghost_agent.core.agent import GhostAgent
        fake = self._fake_agent(["No HTML"], pid=None)
        merged, block, pending = GhostAgent._merge_project_constraints(
            fake, [], "the board is not lined up")
        assert merged == [] and pending is False


# --------------------------------------------------------------------------
# In-turn integration: "proceed." with stored project constraints
# --------------------------------------------------------------------------

def _make_agent():
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


class TestProjectConstraintReplayInTurn:
    """The 2026-07-05 regression: the request that wrote the engine said
    just "proceed." — no clause in the message, so neither the steer nor
    any guard armed. Stored project constraints must now cover it."""

    @pytest.fixture
    def agent(self):
        return _make_agent()

    async def _drive(self, agent, user, write_content):
        calls = []
        seq = [
            {"choices": [{"message": {"content": "", "tool_calls": [
                {"id": "t1", "function": {"name": "file_system",
                 "arguments": json.dumps({
                     "operation": "write", "path": "game.py",
                     "content": write_content})}}]}}]},
            {"choices": [{"message": {
                "content": "Done.", "tool_calls": []}}]},
        ]

        async def _spy(payload, **kw):
            calls.append([m.get("content", "")
                          for m in payload.get("messages", [])])
            return seq[min(len(calls) - 1, len(seq) - 1)]

        fs_mock = AsyncMock(
            return_value="SUCCESS: wrote to 'game.py'.")
        agent.available_tools["file_system"] = fs_mock
        agent.context.llm_client.chat_completion = AsyncMock(
            side_effect=_spy)
        body = {"messages": [{"role": "user", "content": user}],
                "model": "Qwen-Test"}
        with patch("ghost_agent.core.agent.pretty_log"):
            await agent.handle_chat(body, background_tasks=MagicMock())
        return calls, fs_mock

    async def test_proceed_arms_steer_from_project_constraints(self, agent):
        with patch.object(agent, "_active_project_constraints",
                          return_value=list(PARTICIPANT_CONS)):
            calls, fs_mock = await self._drive(
                agent, "proceed.", "print(1)")
        joined = "\n".join("\n".join(c) for c in calls[1:])
        assert fs_mock.called                       # clean write dispatched
        assert "SYSTEM ALERT (constraint check)" in joined
        assert "PARTICIPANT-MODE ARCHITECTURE" in joined
        assert "/api/game/move" in joined

    async def test_proceed_engine_write_is_blocked(self, agent):
        with patch.object(agent, "_active_project_constraints",
                          return_value=list(PARTICIPANT_CONS)):
            calls, fs_mock = await self._drive(
                agent, "proceed.", LIVE_ENGINE_SNIPPET)
        joined = "\n".join("\n".join(c) for c in calls[1:])
        assert not fs_mock.called                   # never reached the tool
        assert "SYSTEM BLOCK (participant-mode engine guard)" in joined
        assert "/api/game/move" in joined

    async def test_proceed_without_project_constraints_untouched(
            self, agent):
        with patch.object(agent, "_active_project_constraints",
                          return_value=[]):
            calls, fs_mock = await self._drive(
                agent, "proceed.", LIVE_ENGINE_SNIPPET)
        joined = "\n".join("\n".join(c) for c in calls[1:])
        assert fs_mock.called                       # no constraint, no block
        assert "SYSTEM BLOCK" not in joined
        assert "SYSTEM ALERT (constraint check)" not in joined
