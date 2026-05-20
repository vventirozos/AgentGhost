"""Tests for the self_state tool (proposal item #1).

The tool is the agent's explicit write path into its own cross-session
SelfStateThread — open questions, unfinished threads, mood.
"""

from pathlib import Path

from ghost_agent.selfhood import SelfModel
from ghost_agent.tools.self_state import tool_self_state


async def test_note_and_list_open_question(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    out = await tool_self_state(
        action="note_question",
        text="Why do trapdoor functions feel asymmetric?",
        self_model=sm,
    )
    assert "Recorded open question" in out
    listing = await tool_self_state(action="list", self_model=sm)
    assert "trapdoor functions" in listing
    assert len(sm.state.open_questions()) == 1


async def test_resolve_question_by_text_substring(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    await tool_self_state(action="note_question",
                          text="Why is the sky blue?", self_model=sm)
    out = await tool_self_state(action="resolve_question",
                                text="sky", self_model=sm)
    assert "resolved" in out.lower()
    assert sm.state.open_questions() == []


async def test_resolve_question_no_match(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    await tool_self_state(action="note_question", text="A real question",
                          self_model=sm)
    out = await tool_self_state(action="resolve_question",
                                text="nonexistent", self_model=sm)
    assert "No open question matched" in out


async def test_unfinished_thread_lifecycle(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    add = await tool_self_state(action="add_unfinished",
                                text="finish the parser rewrite", self_model=sm)
    assert "Recorded unfinished thread" in add
    assert len(sm.state.unfinished_threads()) == 1
    close = await tool_self_state(action="close_unfinished",
                                  text="parser", self_model=sm)
    assert "Closed" in close
    assert sm.state.unfinished_threads() == []


async def test_set_mood(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    out = await tool_self_state(action="set_mood", mood="inquisitive",
                                evidence="a hard problem is bugging me",
                                self_model=sm)
    assert "inquisitive" in out
    assert sm.state.mood().label == "inquisitive"


async def test_state_written_surfaces_in_wakeup_prefix(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    await tool_self_state(action="note_question",
                          text="What is the halting problem really about?",
                          self_model=sm)
    prefix = sm.build_wakeup_prefix()
    assert "halting problem" in prefix


async def test_disabled_self_model_is_graceful(tmp_path: Path):
    sm = SelfModel(root=tmp_path, enabled=False)
    out = await tool_self_state(action="note_question", text="x", self_model=sm)
    assert "unavailable" in out.lower()


async def test_none_self_model_is_graceful():
    out = await tool_self_state(action="list", self_model=None)
    assert "unavailable" in out.lower()


async def test_invalid_action_and_missing_args(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    bad = await tool_self_state(action="explode", self_model=sm)
    assert "SYSTEM ERROR" in bad
    no_text = await tool_self_state(action="note_question", self_model=sm)
    assert "SYSTEM ERROR" in no_text
    no_mood = await tool_self_state(action="set_mood", self_model=sm)
    assert "SYSTEM ERROR" in no_mood
