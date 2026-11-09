"""Tests for the conversational-filler-without-XML guard.

The guard (agent.py:~3283) catches the case where the model writes
narration like "I'll execute the script now" but forgets to emit the
actual `<tool_call>` block. The naive substring-match version false-
positived on any casual reply that used a tool name as an ordinary
English word (`execute` is a verb, `forget` is a verb, `file system`
/ `knowledge base` come up in meta-conversation). See the 23:09 log
where a user asked about AI consciousness and the guard fired on
"structured execution via tools", trapping the model in a spurious
tool-call loop.

These tests pin the new two-signal rule (word-boundary match AND an
explicit intent marker) by reconstructing the guard's matching logic
as a pure predicate.
"""

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
AGENT_SRC = REPO_ROOT / "src" / "ghost_agent" / "core" / "agent.py"


# §4TY (2026-11-09): the guard's predicate is the real helper — the old
# replica (intent anywhere + tool name anywhere) drifted from nothing and
# pinned the shape that rejected req 8ecb1424 twice.
from ghost_agent.core.agent import _filler_promised_tool

TOOL_NAMES = [
    "execute", "file_system", "knowledge_base", "forget",
    "deep_think", "deep_research", "image_generation",
    "self_play", "manage_tasks", "learn_skill",
]


def matches_filler_guard(reply_text: str) -> list[str]:
    """The tool the guard would flag, as a list; empty = it would not fire."""
    hit = _filler_promised_tool(reply_text, TOOL_NAMES)
    return [hit] if hit else []


# ---------------------------------------------------------------------------
# FALSE-POSITIVES — the shapes that broke the user's session. None of
# these should trigger the guard.
# ---------------------------------------------------------------------------


class TestGuardDoesNotFireOnCasualConversation:
    def test_philosophical_reply_mentioning_execution(self):
        """The exact shape from the 23:09 log — a conversational reply
        about AI consciousness that mentions 'execution' in passing."""
        reply = (
            "What's it like to be me? My thought is more like structured "
            "execution via tools — pattern-matching on probabilities, "
            "then handing off to code when precision matters."
        )
        assert matches_filler_guard(reply) == []

    def test_reply_mentioning_execute_as_verb(self):
        reply = "I executed that task yesterday, but today I'm just chatting."
        assert matches_filler_guard(reply) == []

    def test_reply_about_forgetting_in_english_sense(self):
        reply = "Humans forget things all the time; memory is selective."
        assert matches_filler_guard(reply) == []

    def test_reply_discussing_file_system_concept(self):
        reply = "A file system is a way of organising persistent data."
        assert matches_filler_guard(reply) == []

    def test_reply_about_knowledge_base_concept(self):
        reply = "Your knowledge base is basically a vector store over past chats."
        assert matches_filler_guard(reply) == []

    def test_reply_explaining_deep_research(self):
        reply = "Deep research usually means iterated retrieval and synthesis."
        assert matches_filler_guard(reply) == []

    def test_reply_asking_about_tool_names(self):
        """User asked 'what tools do you have?' — reply legitimately
        enumerates them without intending to call any."""
        reply = (
            "The main ones are execute, file_system, and knowledge_base — "
            "each handles a different concern."
        )
        assert matches_filler_guard(reply) == []

    def test_word_execution_does_not_match_execute(self):
        """Word-boundary check — `execution` / `executive` / `executed`
        must not false-match `execute`."""
        reply = "Structured execution is the idea, and I'll think about it."
        # "I'll" is an intent marker, so has_intent=True. But "execute"
        # (word-boundary) doesn't actually appear — only "execution"
        # does. So no tool should match.
        assert matches_filler_guard(reply) == []


# ---------------------------------------------------------------------------
# TRUE POSITIVES — the shapes the guard MUST still catch. If any of
# these stop triggering, legitimate tool-promise forgetfulness slips
# through.
# ---------------------------------------------------------------------------


class TestGuardStillFiresOnRealForgetfulness:
    def test_ill_execute_without_tool_call(self):
        reply = "I'll `execute` the script to check the output."
        assert "execute" in matches_filler_guard(reply)

    def test_let_me_execute(self):
        reply = "Let me use the execute tool for that command now."
        assert "execute" in matches_filler_guard(reply)

    def test_running_file_system_write(self):
        reply = "Running file_system write to save the config."
        assert "file_system" in matches_filler_guard(reply)

    def test_calling_knowledge_base(self):
        reply = "Calling knowledge_base to look that up."
        assert "knowledge_base" in matches_filler_guard(reply)

    def test_i_will_use_deep_research(self):
        reply = "I will use deep_research to dig into this."
        # "I will" is an intent marker; deep_research matches with a
        # word boundary.
        assert "deep_research" in matches_filler_guard(reply)

    def test_now_im_calling_execute(self):
        reply = "Now I'm calling `execute` with the rendered script."
        assert "execute" in matches_filler_guard(reply)


# ---------------------------------------------------------------------------
# Source-level guards — keep the replica in sync with the real code.
# ---------------------------------------------------------------------------


class TestSourceWiring:
    def test_plain_english_verb_is_not_a_tool_promise(self):
        """§4TY: a one-word tool name that is an English verb counts only
        when written as a tool (backticks / "… tool")."""
        assert matches_filler_guard("Let me execute that command now.") == []
        assert matches_filler_guard("I'll forget about it, promise.") == []

    def test_source_no_longer_uses_raw_substring_match(self):
        """Regression guard: the old naive form was
            `t.replace("_", " ") in clean_ui.lower().replace("_", " ")`
        — that line must not reappear."""
        src = AGENT_SRC.read_text()
        assert 't.replace("_", " ") in clean_ui.lower().replace("_", " ")' not in src
