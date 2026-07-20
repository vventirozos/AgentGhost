"""Tests for the uncertainty-gate enhancements (proposal item #6).

Covers durable persistence, recurring-blind-spot detection, the
prompt-injection context, the hedge-scanner, and the flag_uncertainty
tool that populates the tracker.
"""

import json
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
# Resolution persistence — a resolved unknown must stop counting as a
# recurring blind-spot (the log used to record only flag events, so
# resolve_unknown left the readers claiming "unresolved across multiple
# past turns" forever).
# --------------------------------------------------------------------------

def test_resolve_appends_resolution_record(tmp_path: Path):
    log = tmp_path / "u.jsonl"
    t = UncertaintyTracker(persist_path=log)
    u = t.flag_unknown("the deployment target", impact=4)
    assert t.resolve_unknown(u, "prod-eu") is True
    recs = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    assert recs[-1]["kind"] == "unknown_resolved"
    assert recs[-1]["text"] == "the deployment target"
    assert recs[-1]["value"] == "prod-eu"


def test_resolved_unknown_excluded_from_recurring(tmp_path: Path):
    log = tmp_path / "u.jsonl"
    t = UncertaintyTracker(persist_path=log)
    for _ in range(2):
        t.flag_unknown("the deployment target", impact=4)
        t.reset()
    u = t.flag_unknown("the deployment target", impact=4)
    t.resolve_unknown(u, "prod-eu")
    assert t.recurring_unknowns(min_count=2) == []
    assert t.persisted_context() == ""


def test_reflag_after_resolution_counts_fresh(tmp_path: Path):
    """A resolution clears PRIOR flags only — re-flagging afterwards is a
    genuine recurrence and must start a fresh count."""
    log = tmp_path / "u.jsonl"
    t = UncertaintyTracker(persist_path=log)
    for _ in range(3):
        t.flag_unknown("the api schema", impact=4)
        t.reset()
    u = t.flag_unknown("the api schema", impact=4)
    t.resolve_unknown(u, "v2 openapi doc")
    t.reset()
    for _ in range(2):
        t.flag_unknown("the api schema", impact=4)
        t.reset()
    recurring = t.recurring_unknowns(min_count=2)
    assert recurring == [("the api schema", 2)]


def test_resolution_only_clears_matching_text(tmp_path: Path):
    log = tmp_path / "u.jsonl"
    t = UncertaintyTracker(persist_path=log)
    for _ in range(2):
        t.flag_unknown("the timezone", impact=3)
        t.reset()
    u = t.flag_unknown("the locale", impact=3)
    t.resolve_unknown(u, "el_GR")
    recurring = t.recurring_unknowns(min_count=2)
    assert recurring == [("the timezone", 2)]


def test_resolve_by_index_also_persists(tmp_path: Path):
    log = tmp_path / "u.jsonl"
    t = UncertaintyTracker(persist_path=log)
    t.flag_unknown("the port number", impact=3)
    assert t.resolve_unknown(0, "8080") is True
    recs = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    assert recs[-1]["kind"] == "unknown_resolved"


def test_failed_resolve_writes_nothing(tmp_path: Path):
    log = tmp_path / "u.jsonl"
    t = UncertaintyTracker(persist_path=log)
    t.flag_unknown("something", impact=3)
    assert t.resolve_unknown(99, "nope") is False
    recs = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    assert all(r["kind"] != "unknown_resolved" for r in recs)


def test_legacy_log_without_resolution_records_still_parses(tmp_path: Path):
    """Backward compatibility: logs written before resolution records
    existed contain only flag events (plus whatever junk survived a
    crash) and must read exactly as before."""
    log = tmp_path / "u.jsonl"
    old_lines = [
        json.dumps({"ts": 1.0, "kind": "unknown", "text": "the db backend",
                    "impact": 4, "resolution": "ask user"}),
        json.dumps({"ts": 2.0, "kind": "assumption", "text": "py3.11",
                    "confidence": 0.5, "basis": ""}),
        "not json at all",
        json.dumps({"ts": 3.0, "kind": "unknown", "text": "the db backend",
                    "impact": 4, "resolution": "ask user"}),
    ]
    log.write_text("\n".join(old_lines) + "\n")
    t = UncertaintyTracker(persist_path=log)
    assert t.recurring_unknowns(min_count=2) == [("the db backend", 2)]


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
