"""Tests for the operating-principles substrate (selfhood/values.py),
its SelfModel facade, the wake-up-prefix surfacing, the alignment gate,
and the self_state note_principle tool action."""

import pytest

from ghost_agent.selfhood.values import ValuesThread, MAX_PRINCIPLES
from ghost_agent.selfhood.model import SelfModel
from ghost_agent.tools.self_state import tool_self_state


# ──────────────────────────────────────────────────────────────────────
# ValuesThread
# ──────────────────────────────────────────────────────────────────────

def test_note_and_persist(tmp_path):
    vt = ValuesThread(tmp_path)
    p = vt.note_principle("I verify before asserting")
    assert p is not None
    assert len(vt.principles()) == 1
    # Reload from disk → survives.
    vt2 = ValuesThread(tmp_path)
    assert [x.text for x in vt2.principles()] == ["I verify before asserting"]


def test_dedup_case_insensitive(tmp_path):
    vt = ValuesThread(tmp_path)
    vt.note_principle("I prefer reversible actions")
    vt.note_principle("i PREFER reversible ACTIONS")
    assert len(vt.principles()) == 1


def test_cap_most_recent_wins(tmp_path):
    vt = ValuesThread(tmp_path)
    for i in range(MAX_PRINCIPLES + 5):
        vt.note_principle(f"principle number {i}")
    ps = vt.principles()
    assert len(ps) == MAX_PRINCIPLES
    assert ps[-1].text == f"principle number {MAX_PRINCIPLES + 4}"  # newest kept


def test_remove_and_clear(tmp_path):
    vt = ValuesThread(tmp_path)
    p = vt.note_principle("X")
    assert vt.remove_principle(p.id) is True
    assert vt.principles() == []
    vt.note_principle("Y")
    vt.clear()
    assert vt.principles() == []


def test_empty_and_blank(tmp_path):
    vt = ValuesThread(tmp_path)
    assert vt.note_principle("") is None
    assert vt.note_principle("   ") is None
    assert vt.format_as_prefix() == ""
    assert vt.as_text() == ""


def test_format_as_prefix_and_as_text(tmp_path):
    vt = ValuesThread(tmp_path)
    vt.note_principle("I verify before asserting")
    pref = vt.format_as_prefix()
    assert "operating principles" in pref.lower()
    assert "I verify before asserting" in pref
    assert "- I verify before asserting" in vt.as_text()


def test_corrupt_file_recovers(tmp_path):
    (tmp_path / "values.json").write_text("{ not valid json")
    vt = ValuesThread(tmp_path)
    assert vt.principles() == []  # degrades to empty, no crash
    assert vt.note_principle("recovered") is not None


# ──────────────────────────────────────────────────────────────────────
# SelfModel facade + prefix
# ──────────────────────────────────────────────────────────────────────

def test_selfmodel_note_and_prefix(tmp_path):
    sm = SelfModel(tmp_path, enabled=True)
    sm.note_principle("I prefer reversible actions")
    assert any(p.text == "I prefer reversible actions" for p in sm.principles())
    prefix = sm.build_wakeup_prefix()
    assert "I prefer reversible actions" in prefix
    assert "operating principles" in prefix.lower()


def test_selfmodel_disabled_noop(tmp_path):
    sm = SelfModel(tmp_path, enabled=False)
    assert sm.note_principle("x") is None
    assert sm.principles() == []
    assert sm.principles_text() == ""


# ──────────────────────────────────────────────────────────────────────
# alignment gate
# ──────────────────────────────────────────────────────────────────────

async def test_alignment_no_principles_is_aligned(tmp_path):
    sm = SelfModel(tmp_path, enabled=True)

    async def crit(_p):
        raise AssertionError("should not be called when no principles")

    aligned, note = await sm.evaluate_response_alignment("anything", critique_fn=crit)
    assert aligned is True
    assert note == ""


async def test_alignment_violation(tmp_path):
    sm = SelfModel(tmp_path, enabled=True)
    sm.note_principle("I verify before asserting")

    async def crit(_p):
        return "VERDICT: VIOLATION\nThe response asserts a number without verifying it."

    aligned, note = await sm.evaluate_response_alignment(
        "The project has 1,623 lines.", critique_fn=crit)
    assert aligned is False
    assert "VIOLATION" in note.upper()


async def test_alignment_confirmed(tmp_path):
    sm = SelfModel(tmp_path, enabled=True)
    sm.note_principle("I verify before asserting")

    async def crit(_p):
        return "VERDICT: ALIGNED\nThe response verified via a tool first."

    aligned, _ = await sm.evaluate_response_alignment("...", critique_fn=crit)
    assert aligned is True


async def test_alignment_fail_open_on_error(tmp_path):
    sm = SelfModel(tmp_path, enabled=True)
    sm.note_principle("I verify before asserting")

    async def crit(_p):
        raise RuntimeError("judge down")

    aligned, note = await sm.evaluate_response_alignment("...", critique_fn=crit)
    assert aligned is True  # gate must never block on its own failure
    assert note == ""


# ──────────────────────────────────────────────────────────────────────
# self_state tool action
# ──────────────────────────────────────────────────────────────────────

async def test_tool_note_principle(tmp_path):
    sm = SelfModel(tmp_path, enabled=True)
    out = await tool_self_state(action="note_principle",
                                text="I prefer reversible actions",
                                self_model=sm)
    assert "operating principle" in out.lower()
    # And it shows up in list.
    listed = await tool_self_state(action="list", self_model=sm)
    assert "I prefer reversible actions" in listed


async def test_tool_note_principle_requires_text(tmp_path):
    sm = SelfModel(tmp_path, enabled=True)
    out = await tool_self_state(action="note_principle", text="", self_model=sm)
    assert "required" in out.lower()
