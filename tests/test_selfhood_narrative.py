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
    # A NEW experience lands between regenerations — each run therefore
    # sees changed input and must persist. (Identical-input runs are
    # skipped by the idempotency guard; see the dedicated tests below.)
    autobio = AutobiographicalMemory(tmp_path)

    call_count = [0]

    async def fake_critique(prompt: str) -> str:
        call_count[0] += 1
        return f"Run {call_count[0]}."

    n = NarrativeSummariser(tmp_path, critique_fn=fake_critique)
    for i in range(3):
        autobio.append(Experience(summary=f"A turn {i}."))
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


async def test_narrative_skips_regeneration_on_unchanged_input(tmp_path: Path):
    autobio = AutobiographicalMemory(tmp_path)
    autobio.append(Experience(summary="I fixed the parser.", outcome="passed"))

    call_count = [0]

    async def fake_critique(prompt: str) -> str:
        call_count[0] += 1
        return f"Run {call_count[0]}."

    n = NarrativeSummariser(tmp_path, critique_fn=fake_critique)
    first = await n.regenerate(autobio=autobio)
    assert first == "Run 1."

    # Nothing new happened → no LLM call, no persist, empty return.
    second = await n.regenerate(autobio=autobio)
    assert second == ""
    assert call_count[0] == 1
    history = (tmp_path / "narrative.history.jsonl").read_text(encoding="utf-8")
    assert len([l for l in history.splitlines() if l]) == 1
    assert n.latest() == "Run 1."

    # A new experience unblocks the guard.
    autobio.append(Experience(summary="I shipped the fix.", outcome="passed"))
    third = await n.regenerate(autobio=autobio)
    assert third == "Run 2."


async def test_narrative_failed_persist_does_not_commit_key(
    tmp_path: Path, monkeypatch,
):
    # A transient disk error during persist must NOT commit the
    # idempotency key — otherwise the guard passes via the OLDER
    # narrative file on the next run and staleness becomes permanent.
    autobio = AutobiographicalMemory(tmp_path)
    autobio.append(Experience(summary="I fixed the parser.", outcome="passed"))

    async def fake_critique(prompt: str) -> str:
        return "Fresh entry."

    n = NarrativeSummariser(tmp_path, critique_fn=fake_critique)
    # An OLDER narrative already on disk — what the guard would pass via.
    (tmp_path / "narrative.md").write_text("Stale entry.", encoding="utf-8")

    calls = [0]
    real_write_text = Path.write_text

    def flaky_write_text(self, *args, **kwargs):
        if self.name.endswith(".md.tmp"):
            calls[0] += 1
            if calls[0] == 1:
                raise OSError("disk full")
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", flaky_write_text)

    await n.regenerate(autobio=autobio)
    assert n._last_input_key == ""  # not committed on failure
    assert n.latest() == "Stale entry."

    # Same input again → guard must not skip; the retry persists.
    second = await n.regenerate(autobio=autobio)
    assert second == "Fresh entry."
    assert n.latest() == "Fresh entry."
    assert n._last_input_key != ""


async def test_narrative_history_compaction_caps_file(tmp_path: Path):
    from ghost_agent.selfhood.narrative import (
        _HISTORY_COMPACT_KEEP_LINES,
        _HISTORY_COMPACT_MAX_BYTES,
    )
    n = NarrativeSummariser(tmp_path)
    history_path = tmp_path / "narrative.history.jsonl"
    # Prefill the audit history past the byte cap with fat records.
    pad = "x" * 4096
    with history_path.open("w", encoding="utf-8") as f:
        for i in range(300):
            f.write(json.dumps({"timestamp": f"t{i}", "text": pad}) + "\n")
    assert history_path.stat().st_size > _HISTORY_COMPACT_MAX_BYTES

    assert n._persist("fresh", used_llm=False, source_count=1) is True

    lines = [l for l in history_path.read_text(
        encoding="utf-8").splitlines() if l]
    assert len(lines) == _HISTORY_COMPACT_KEEP_LINES
    assert history_path.stat().st_size <= _HISTORY_COMPACT_MAX_BYTES
    # Newest tail kept — the just-persisted record is last, the oldest
    # prefilled records are gone.
    assert json.loads(lines[-1])["text"] == "fresh"
    assert json.loads(lines[0])["timestamp"] != "t0"


async def test_narrative_filters_trivial_experiences(tmp_path: Path):
    # Live failure shape (2026-07-13): ping-shaped turns (no tools, no
    # verdict, tiny request) dominated the recent window, so the diary
    # opened with 'Lately, I worked on "reply with just: pong"'. With an
    # informative experience available anywhere in the wider window, the
    # trivial ones must not reach the prompt.
    autobio = AutobiographicalMemory(tmp_path)
    autobio.append(Experience(
        summary='I worked on "debug the tor circuit racing". I reached for execute and it passed.',
        outcome="passed", tools_used=["execute"],
        user_first_words="debug the tor circuit racing",
    ))
    for _ in range(6):
        autobio.append(Experience(
            summary='I worked on "reply with just: pong". I reasoned through it without tools without a verdict either way.',
            outcome="unknown", user_first_words="reply with just: pong",
        ))

    captured = []

    async def fake_critique(prompt: str) -> str:
        captured.append(prompt)
        return "A proper diary entry."

    n = NarrativeSummariser(tmp_path, critique_fn=fake_critique)
    out = await n.regenerate(autobio=autobio)
    assert out == "A proper diary entry."
    assert "tor circuit racing" in captured[0]
    assert "reply with just: pong" not in captured[0]


async def test_narrative_all_trivial_window_still_writes(tmp_path: Path):
    # When EVERY recent experience is trivial, fall back to the
    # unfiltered slice — a thin diary beats an empty one.
    autobio = AutobiographicalMemory(tmp_path)
    for i in range(3):
        autobio.append(Experience(
            summary=f'I worked on "ping {i}". I reasoned through it without tools without a verdict either way.',
            outcome="unknown", user_first_words=f"ping {i}",
        ))
    n = NarrativeSummariser(tmp_path, critique_fn=None)
    out = await n.regenerate(autobio=autobio)
    assert out.startswith("Lately")


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
