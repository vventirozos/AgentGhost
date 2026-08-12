"""`knowledge_base` must be findable by the VERB a model is holding.

§4AW measured the failure precisely: the tool was advertised on all 16
tool-carrying payloads, sat 3rd of 44, and its description already said
"Do NOT write or install transcription code" — and the model installed
openai-whisper anyway, then planned a 4-step Whisper pipeline for a request
that is two steps. Presence was never the problem; FINDABILITY BY NEED was.
A guard built on top of that measurement was reviewed and reverted, so this
is the other lever: the tool now answers to the word the model looks for,
says so first, and the planner is told the step does not exist.
"""

import asyncio
import json
import re
from pathlib import Path

import pytest

import ghost_agent.tools.memory as mem
from ghost_agent.tools.registry import TOOL_DEFINITIONS


def _kb_schema():
    for entry in TOOL_DEFINITIONS:
        fn = entry.get("function", {})
        if fn.get("name") == "knowledge_base":
            return fn
    raise AssertionError("knowledge_base is not advertised at all")


# ------------------------------------------------------- the alias itself

@pytest.mark.asyncio
@pytest.mark.parametrize("action", [
    "transcribe", "TRANSCRIBE", " transcribe ", "transcribe_document",
    "transcription", "ingest", "ingest_file", "ingest_document",
])
async def test_every_name_for_this_capability_reaches_the_ingest_handler(
        action, monkeypatch):
    seen = []

    async def _fake_gain(target, sandbox_dir, memory_system):
        seen.append(target)
        return "ok"

    monkeypatch.setattr(mem, "tool_gain_knowledge", _fake_gain)
    out = await mem.tool_knowledge_base(action=action, filename="talk.mp4")
    assert seen == ["talk.mp4"], f"{action!r} did not reach ingest"
    assert out == "ok"


@pytest.mark.asyncio
async def test_an_unknown_action_is_still_rejected(monkeypatch):
    """Aliasing must not become a catch-all that swallows typos into an
    ingest of whatever `filename` happened to be passed."""
    out = await mem.tool_knowledge_base(action="frobnicate", filename="x.mp4")
    assert "Unknown action" in str(out)


@pytest.mark.asyncio
async def test_the_other_actions_are_untouched(monkeypatch):
    """The alias map must not shadow a real action."""
    hits = []
    monkeypatch.setattr(mem, "tool_remember",
                        lambda *a, **k: _async("insert_fact", hits))
    monkeypatch.setattr(mem, "tool_query_document",
                        lambda *a, **k: _async("query", hits))
    await mem.tool_knowledge_base(action="insert_fact", fact="f")
    await mem.tool_knowledge_base(action="query", filename="d", question="q")
    assert hits == ["insert_fact", "query"]


async def _async(label, sink):
    sink.append(label)
    return label


# --------------------------------------------------- how it is advertised

def test_the_capability_leads_the_description():
    """It used to appear at 20% of a 537-char run-on, as a subordinate
    clause, with the prohibition trailing at 82% — measured, and measured
    NOT to work."""
    desc = _kb_schema()["description"]
    idx = desc.lower().find("transcrib")
    assert idx >= 0
    assert idx < len(desc) * 0.10, (
        f"'transcrib' appears at {100 * idx // len(desc)}% of the "
        f"description — a model scanning for a transcriber will not reach it")


def test_the_description_carries_a_worked_call():
    """Models pick a tool up from a concrete call far more reliably than
    from prose about what it can do."""
    desc = _kb_schema()["description"]
    assert "knowledge_base(action='transcribe', filename=" in desc


def test_the_description_names_what_NOT_to_build():
    desc = _kb_schema()["description"].lower()
    for banned in ("whisper", "ffmpeg", "speech-to-text", "transcribe.py"):
        assert banned in desc, f"the description should name {banned!r}"
    assert "one call" in desc


def test_transcribe_is_advertised_in_the_action_enum():
    """The enum is short and always read — it is where a model with a verb
    in mind actually looks."""
    action = _kb_schema()["parameters"]["properties"]["action"]
    assert "transcribe" in action["enum"]
    assert action["enum"][0] == "transcribe", (
        "lead with it: this is the name the failing case reaches for")
    assert "same action" in action.get("description", "").lower()


def test_the_filename_param_refuses_a_video_URL():
    """The observed plan tried to hand a YouTube URL straight to ingest.
    The file must be downloaded first — that step IS real work."""
    fn = _kb_schema()["parameters"]["properties"]["filename"]["description"]
    assert "NOT a YouTube" in fn or "not a youtube" in fn.lower()
    assert "download the file first" in fn.lower()


# --------------------------------------------------------- the plan itself

def test_the_planner_is_told_the_step_does_not_exist():
    """The failure the operator saw was a PLAN — tasks 2 and 3 of four were
    'Whisper Transcription' and 'Knowledge Base Integration', both of which
    are one existing tool call. The planning rules are where that is
    decided."""
    from ghost_agent.core.prompts import PLANNING_SYSTEM_PROMPT
    rules = PLANNING_SYSTEM_PROMPT
    assert "TRANSCRIPTION AND INGESTION ARE ONE STEP" in rules
    assert "transcribe.py" in rules
    assert "Downloading the media first IS legitimate work" in rules, (
        "the rule must not over-correct into forbidding the download too")
