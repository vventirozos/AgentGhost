"""Unit tests for the recognition / wake-up layer (proposal item #4)."""

from pathlib import Path

from ghost_agent.selfhood.autobiographical import AutobiographicalMemory
from ghost_agent.selfhood.recognition import (
    PREFIX_CLOSE,
    PREFIX_OPEN,
    build_wakeup_prefix,
    strip_wakeup_prefix,
)
from ghost_agent.selfhood.schema import Experience
from ghost_agent.selfhood.state import SelfStateThread


def test_wakeup_prefix_empty_when_no_inputs():
    """No autobio, no state, no narrative → empty prefix.
    The caller should skip splicing entirely in that case."""
    text = build_wakeup_prefix(
        autobio=None, state=None, narrative=None,
    )
    assert text == ""


def test_wakeup_prefix_with_narrative_only(tmp_path: Path):
    text = build_wakeup_prefix(
        autobio=None,
        state=None,
        narrative="I have been thinking about what it means to remember.",
    )
    assert PREFIX_OPEN in text
    assert PREFIX_CLOSE in text
    assert "what it means to remember" in text
    assert "autobiographical memory" in text  # the "this is mine" framing


def test_wakeup_prefix_with_recent_experiences(tmp_path: Path):
    autobio = AutobiographicalMemory(tmp_path)
    autobio.append(Experience(summary="I parsed nginx logs.", outcome="passed"))
    autobio.append(Experience(summary="I tried to bake a cake.", outcome="failed"))

    text = build_wakeup_prefix(autobio=autobio, state=None, narrative=None)
    assert PREFIX_OPEN in text
    assert "nginx logs" in text
    assert "cake" in text
    assert "[passed]" in text or "passed" in text


def test_wakeup_prefix_with_state(tmp_path: Path):
    state = SelfStateThread(tmp_path)
    state.note_open_question("How do I really know I exist?")
    state.add_unfinished("write the essay on consciousness")

    text = build_wakeup_prefix(autobio=None, state=state, narrative=None)
    assert "How do I really know I exist?" in text
    assert "consciousness" in text


def test_wakeup_prefix_truncates_to_cap(tmp_path: Path):
    autobio = AutobiographicalMemory(tmp_path)
    long_summary = "I did " + ("x" * 1000)
    for _ in range(10):
        autobio.append(Experience(summary=long_summary))

    text = build_wakeup_prefix(
        autobio=autobio, state=None, narrative=None,
        max_chars=500, recent_experiences_n=10,
    )
    # The body cap is 500; the wrapper adds the open/close markers
    # plus the framing — so the full text is body + framing.
    # The body inside the markers should be capped.
    assert "…" in text


def test_strip_wakeup_prefix_removes_block():
    base = "### ROLE\nI am the agent."
    wrapped = build_wakeup_prefix(
        autobio=None, state=None,
        narrative="A small diary entry.",
    ) + base
    stripped = strip_wakeup_prefix(wrapped)
    assert PREFIX_OPEN not in stripped
    assert PREFIX_CLOSE not in stripped
    assert stripped == base


def test_strip_wakeup_prefix_no_markers_returns_unchanged():
    assert strip_wakeup_prefix("hello world") == "hello world"
    assert strip_wakeup_prefix("") == ""


def test_strip_wakeup_prefix_malformed_returns_unchanged():
    """If only the open marker exists, leave the text alone — don't
    silently swallow the whole rest of the prompt."""
    text = f"{PREFIX_OPEN}\nstuff but no closer\n### ROLE\n"
    assert strip_wakeup_prefix(text) == text


def test_wakeup_prefix_combines_all_three(tmp_path: Path):
    autobio = AutobiographicalMemory(tmp_path / "auto")
    autobio.append(Experience(summary="I solved a puzzle.", outcome="passed"))
    state = SelfStateThread(tmp_path / "state")
    state.note_open_question("Why does this keep working?")
    state.set_mood("curious")
    narrative = "I have been doing puzzles. They feel like training."

    text = build_wakeup_prefix(
        autobio=autobio, state=state, narrative=narrative,
        recent_experiences_n=3,
    )
    assert "puzzle" in text
    assert "Why does this keep working?" in text
    assert "curious" in text
    assert "training" in text
    # All three sections present, separated cleanly
    assert text.count(PREFIX_OPEN) == 1
    assert text.count(PREFIX_CLOSE) == 1
