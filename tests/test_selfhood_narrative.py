"""Unit tests for the narrative summariser (proposal item #5)."""

import asyncio
import json
from pathlib import Path

import pytest

from ghost_agent.selfhood.autobiographical import AutobiographicalMemory
from ghost_agent.selfhood.narrative import NarrativeSummariser
from ghost_agent.selfhood.schema import Experience
from ghost_agent.selfhood.state import SelfStateThread


async def test_narrative_noop_when_no_experiences(tmp_path: Path):
    autobio = AutobiographicalMemory(tmp_path)
    n = NarrativeSummariser(tmp_path)
    out = await n.regenerate(autobio=autobio)
    assert out == ""
    assert not (tmp_path / "narrative.md").exists()


async def test_narrative_template_fallback_when_no_critique_fn(tmp_path: Path):
    autobio = AutobiographicalMemory(tmp_path)
    for i in range(3):
        autobio.append(Experience(summary=f"I did thing {i}.", outcome="passed"))
    n = NarrativeSummariser(tmp_path, critique_fn=None)
    out = await n.regenerate(autobio=autobio)
    assert out != ""
    assert out.startswith("Lately")
    # Persisted
    assert (tmp_path / "narrative.md").read_text(encoding="utf-8") == out


async def test_narrative_uses_critique_fn_when_supplied(tmp_path: Path):
    autobio = AutobiographicalMemory(tmp_path)
    autobio.append(Experience(summary="I helped a researcher.", outcome="passed"))

    captured_prompts = []

    async def fake_critique(prompt: str) -> str:
        captured_prompts.append(prompt)
        return "Today I was helpful. Yesterday too. Tomorrow we'll see."

    n = NarrativeSummariser(tmp_path, critique_fn=fake_critique)
    out = await n.regenerate(autobio=autobio)
    assert "Today I was helpful" in out
    assert len(captured_prompts) == 1
    assert "I helped a researcher" in captured_prompts[0]
    assert "(no persisted state)" in captured_prompts[0]


async def test_narrative_falls_back_when_critique_returns_empty(tmp_path: Path):
    autobio = AutobiographicalMemory(tmp_path)
    autobio.append(Experience(summary="I tried something."))

    async def empty_critique(prompt: str) -> str:
        return ""

    n = NarrativeSummariser(tmp_path, critique_fn=empty_critique)
    out = await n.regenerate(autobio=autobio)
    assert out != ""  # template fallback wrote something
    assert out.startswith("Lately")


async def test_narrative_falls_back_when_critique_raises(tmp_path: Path):
    autobio = AutobiographicalMemory(tmp_path)
    autobio.append(Experience(summary="I tried something."))

    async def boom(prompt: str) -> str:
        raise RuntimeError("LLM down")

    n = NarrativeSummariser(tmp_path, critique_fn=boom)
    out = await n.regenerate(autobio=autobio)
    assert out != ""
    assert out.startswith("Lately")


async def test_narrative_history_appends(tmp_path: Path):
    autobio = AutobiographicalMemory(tmp_path)
    autobio.append(Experience(summary="A turn."))

    call_count = [0]

    async def fake_critique(prompt: str) -> str:
        call_count[0] += 1
        return f"Run {call_count[0]}."

    n = NarrativeSummariser(tmp_path, critique_fn=fake_critique)
    await n.regenerate(autobio=autobio)
    await n.regenerate(autobio=autobio)
    await n.regenerate(autobio=autobio)

    history_path = tmp_path / "narrative.history.jsonl"
    assert history_path.exists()
    lines = [l for l in history_path.read_text(encoding="utf-8").splitlines() if l]
    assert len(lines) == 3
    records = [json.loads(l) for l in lines]
    assert all(r["used_llm"] is True for r in records)
    assert records[-1]["text"] == "Run 3."

    # The "latest" file holds only the most recent text.
    assert (tmp_path / "narrative.md").read_text(encoding="utf-8") == "Run 3."


async def test_narrative_includes_state_in_prompt(tmp_path: Path):
    autobio = AutobiographicalMemory(tmp_path / "auto")
    autobio.append(Experience(summary="Did a turn."))
    state = SelfStateThread(tmp_path / "state")
    state.note_open_question("Why does X happen?")

    captured = []

    async def fake_critique(prompt: str) -> str:
        captured.append(prompt)
        return "Some diary text."

    n = NarrativeSummariser(tmp_path, critique_fn=fake_critique)
    await n.regenerate(autobio=autobio, state=state)
    assert "Why does X happen?" in captured[0]


async def test_narrative_disabled_returns_empty(tmp_path: Path):
    autobio = AutobiographicalMemory(tmp_path)
    autobio.append(Experience(summary="hi"))
    n = NarrativeSummariser(tmp_path, enabled=False)
    out = await n.regenerate(autobio=autobio)
    assert out == ""
    assert not (tmp_path / "narrative.md").exists()


def test_narrative_latest_empty_when_never_run(tmp_path: Path):
    n = NarrativeSummariser(tmp_path)
    assert n.latest() == ""


async def test_narrative_latest_returns_most_recent(tmp_path: Path):
    autobio = AutobiographicalMemory(tmp_path)
    autobio.append(Experience(summary="x"))
    n = NarrativeSummariser(tmp_path)
    out = await n.regenerate(autobio=autobio)
    assert n.latest() == out
