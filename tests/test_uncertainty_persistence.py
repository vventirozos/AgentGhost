"""Tests for the uncertainty-gate enhancements (proposal item #6).

Covers durable persistence, recurring-blind-spot detection, the
prompt-injection context, the hedge-scanner, and the flag_uncertainty
tool that populates the tracker.
"""

from pathlib import Path

from ghost_agent.core.uncertainty import UncertaintyTracker
from ghost_agent.tools.uncertainty_tool import tool_flag_uncertainty


def test_flags_persist_to_disk(tmp_path: Path):
    log = tmp_path / "uncertainty_log.jsonl"
    t = UncertaintyTracker(persist_path=log)
    t.flag_unknown("the user's timezone", impact=4)
    t.flag_assumption("prod runs python 3.11", confidence=0.5)
    assert log.exists()
    lines = [l for l in log.read_text().splitlines() if l.strip()]
    assert len(lines) == 2


def test_reset_does_not_wipe_persisted_log(tmp_path: Path):
    log = tmp_path / "u.jsonl"
    t = UncertaintyTracker(persist_path=log)
    t.flag_unknown("something", impact=3)
    t.reset()
    assert t.unknowns == []          # in-memory cleared
    assert log.exists()              # durable log survives
    assert len([l for l in log.read_text().splitlines() if l.strip()]) == 1


def test_recurring_unknowns_detects_repeated_blind_spots(tmp_path: Path):
    log = tmp_path / "u.jsonl"
    t = UncertaintyTracker(persist_path=log)
    t.flag_unknown("the deployment target", impact=4)
    t.reset()
    t.flag_unknown("the deployment target", impact=4)
    t.reset()
    t.flag_unknown("a one-off thing", impact=2)
    recurring = t.recurring_unknowns(min_count=2)
    assert len(recurring) == 1
    assert recurring[0][0] == "the deployment target"
    assert recurring[0][1] == 2


def test_persisted_context_renders_recurring(tmp_path: Path):
    log = tmp_path / "u.jsonl"
    t = UncertaintyTracker(persist_path=log)
    for _ in range(2):
        t.flag_unknown("which database backend", impact=4)
        t.reset()
    ctx = t.persisted_context()
    assert "RECURRING UNCERTAINTIES" in ctx
    assert "which database backend" in ctx


def test_persisted_context_empty_without_persistence():
    t = UncertaintyTracker()  # no persist_path
    t.flag_unknown("x", impact=5)
    assert t.persisted_context() == ""
    assert t.recurring_unknowns() == []


def test_scan_text_for_uncertainty_finds_hedges():
    text = (
        "The result is 42. I'm assuming the input file is UTF-8 encoded. "
        "I could not verify the upstream schema. Everything else checks out."
    )
    hits = UncertaintyTracker.scan_text_for_uncertainty(text)
    assert any("assuming" in h.lower() for h in hits)
    assert any("could not verify" in h.lower() for h in hits)


def test_scan_text_no_false_positive_on_confident_text():
    hits = UncertaintyTracker.scan_text_for_uncertainty(
        "The answer is 42. The file is UTF-8. This is correct."
    )
    assert hits == []


def test_should_ask_user_gate():
    t = UncertaintyTracker()
    # Low-impact unknown does not trip the gate.
    t.flag_unknown("a cosmetic detail", impact=2, resolution="ask user")
    assert t.should_ask_user() is None
    # Critical unknown needing the user does.
    t.flag_unknown("which production database to target", impact=5,
                   resolution="ask user")
    q = t.should_ask_user()
    assert q is not None
    assert "which production database to target" in q


# --------------------------------------------------------------------------
# flag_uncertainty tool
# --------------------------------------------------------------------------

async def test_flag_uncertainty_tool_unknown():
    t = UncertaintyTracker()
    out = await tool_flag_uncertainty(
        action="unknown", text="the deadline", impact=4,
        uncertainty_tracker=t,
    )
    assert "deadline" in out
    assert "critical unknown" in out  # impact 4 + ask user → escalation note
    assert len(t.unknowns) == 1


async def test_flag_uncertainty_tool_assumption_and_list():
    t = UncertaintyTracker()
    await tool_flag_uncertainty(action="assumption", text="prod is py3.11",
                                confidence=0.6, uncertainty_tracker=t)
    assert len(t.assumptions) == 1
    listing = await tool_flag_uncertainty(action="list", uncertainty_tracker=t)
    assert "prod is py3.11" in listing


async def test_flag_uncertainty_tool_guards():
    t = UncertaintyTracker()
    assert "SYSTEM ERROR" in await tool_flag_uncertainty(
        action="bogus", uncertainty_tracker=t)
    assert "SYSTEM ERROR" in await tool_flag_uncertainty(
        action="unknown", uncertainty_tracker=t)  # missing text
    assert "unavailable" in await tool_flag_uncertainty(
        action="unknown", text="x", uncertainty_tracker=None)
