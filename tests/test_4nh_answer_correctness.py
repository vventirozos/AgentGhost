"""§4NH (2026-10-10): answer correctness — the bugs and instruments the
census root-causing found, and the no-tool knowledge check feeding replays."""
from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ── the redactor ate figures from stored tables ───────────────────────

@pytest.mark.parametrize("text,kept", [
    ("2020 - 1,100 2021 - 1,343 2022 - 1,272", True),       # a table of figures and years
    ("price 3.100 2021", True),
    ("the 300 2024 models", True),               # a figure and a year, space-separated
])
def test_figures_and_years_are_not_phone_numbers(text, kept):
    """'1,100 2021' became <REDACTED_PHONE> in the stored trajectory, so a
    census grader saw invented figures the model had in fact read."""
    from ghost_agent.distill.redact import redact_text
    assert (redact_text(text) == text) is kept


@pytest.mark.parametrize("phone", ["call 555-0123 now", "+30 210 5550123", "(212) 555-0123", "212-555-0123",
                                   # review r1: forms a comma/dot lookbehind would have leaked
                                   "Τηλ.210 555 0123", "phones: 210 555 0123,210 555 0124", "Tel.(210) 555-0123",
                                   "call me at,555-0123"])
def test_real_phone_numbers_are_still_redacted(phone):
    from ghost_agent.distill.redact import redact_text
    out = redact_text(phone)
    assert "<REDACTED_PHONE>" in out and not any(g in out for g in ("555", "0124"))


# ── the project work log trusted the reply ────────────────────────────

def _wl_agent(failed: dict):
    from ghost_agent.core.agent import GhostAgent
    a = GhostAgent.__new__(GhostAgent)
    store = MagicMock()
    store.get_project.return_value = {"id": "p1", "title": "report", "goal": "write the report pdf"}
    a.context = SimpleNamespace(current_project_id="p1", project_store=store,
                                _project_work_files={"report.pdf"}, _project_work_tools={"file_system": 3},
                                _project_work_failed_tools=dict(failed), _project_work_cmds=[],
                                _turn_failure_texts=[], _project_work_pending=None)
    return a, store


@pytest.mark.parametrize("failed,outcome", [({"file_system": 3}, "had_failures"), ({}, "completed")])
def test_a_turn_whose_writes_failed_is_not_recorded_completed(failed, outcome):
    """3 failed file_system writes were recorded 'completed', and the reply's
    claimed PDF became the next briefing's fact."""
    a, store = _wl_agent(failed)
    asyncio.run(a._write_project_work_log_safe(
        last_user_content="write the report pdf for the project", final_ai_content="Done — report.pdf is written.",
        execution_failure_count=0))
    assert store.add_work_log.called, "the relevance gate skipped the write — the test proves nothing"
    kw = store.add_work_log.call_args.kwargs
    assert kw["outcome"] == outcome
    assert kw["note"].startswith("[unverified;") is bool(failed)


# ── introspect's overview kept its head and lost its tail ─────────────

def test_the_overview_keeps_its_last_sections_when_long(monkeypatch):
    """a97ca03e: the cut at the cap (then 3,200 chars) dropped the short sections at the end
    (§4NM: the post-mortem defects section is gone; the workspace line is
    the tail that must survive)."""
    from ghost_agent.tools import introspect as I
    monkeypatch.setattr(I, "_overview_selfhood", lambda sm, context=None: "S" * 900)
    monkeypatch.setattr(I, "_overview_activity", lambda ctx: "A" * 900)

    async def _long(ctx):
        return "L" * (I._OVERVIEW_MAX_CHARS - 1700)     # with the two 900s: over the cap (§4SV-b raised it)
    monkeypatch.setattr(I, "_overview_learning", _long)
    monkeypatch.setattr(I, "_overview_workspace", lambda ctx: "Workspace: 2 projects")
    out = asyncio.run(I._render_overview(None, SimpleNamespace()))
    assert len(out) <= I._OVERVIEW_MAX_CHARS
    assert "Workspace: 2 projects" in out
    assert "action='learning' for the rest" in out   # §4NN: the longest surface left gives way first


# ── no-tool factual answers are checked, and feed the replay ──────────

@pytest.mark.parametrize("answer,ok", [
    ("For emerging markets on IBKR look at VWRD and EIMI; VWRD tracks about 3,800 stocks and has done since 2012, "
     "so it is the broad choice for a small monthly amount.", True),
    ("Use pg_last_subscription_replay_lsn() on the subscriber and check pg_stat_subscription for the lag "
     "right after the cutover, before you stop writes on the old server.", True),
    ("Sure, happy to help with that whenever you are ready, just tell me what you need next and I will "
     "take a look at it as soon as you send it over.", False),
])
def test_a_short_answer_with_specific_facts_is_checked(answer, ok, monkeypatch):
    monkeypatch.setenv("GHOST_KNOWLEDGE_SHADOW", "1")         # §4OH: off by default; this tests the gate when on
    from ghost_agent.core.knowledge_shadow import eligible
    assert eligible("what should I buy for emerging markets exposure", answer) is ok


def test_the_rate_limit_lets_a_follow_up_through():
    """The wrong ETF follow-up came 87 s after the first answer and was skipped."""
    from ghost_agent.core import knowledge_shadow as K
    assert K.MIN_INTERVAL_S <= 90


def test_a_knowledge_error_seeds_the_failure_replay_and_nothing_else(tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_KNOWLEDGE_SHADOW", "1")         # §4OH: off by default, and off seeds nothing
    """Review r1 (MAJOR ×2): routed through `_verdicts_by_trajectory` it could
    pull a proven lesson (lesson withdrawal reads it) and never reached the
    replay (both pickers admit tool turns only). Now a separate source."""
    from ghost_agent.core.owner_seeds import _verdicts_by_trajectory, knowledge_errors
    (tmp_path / "verifier").mkdir()
    rows = [{"ts": time.time(), "trajectory_id": "t-err", "verdict": "errors",
             "issues": [{"quote": "VEM", "why": "not Vanguard's EU EM ETF"}]},
            {"ts": time.time(), "trajectory_id": "t-ok", "verdict": "ok", "issues": []},
            {"ts": time.time(), "trajectory_id": "t-noissue", "verdict": "errors", "issues": []}]
    (tmp_path / "verifier" / "knowledge_shadow.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert knowledge_errors(tmp_path, 7) == {"t-err"}
    assert "t-err" not in _verdicts_by_trajectory(tmp_path, 7)          # lesson withdrawal never sees it


def test_the_replay_picks_a_flagged_no_tool_answer(tmp_path, monkeypatch):
    from ghost_agent.core import failure_replay as FR
    from ghost_agent.core import owner_seeds as OS
    t = SimpleNamespace(id="t-err", user_request="which emerging markets ETF is best for a small monthly amount",
                        final_response="VEM.", tool_calls=[], extra={"req_id": "web-1"}, outcome="unknown",
                        timestamp="2026-10-10T05:04:00Z", session_id="s")
    coll = MagicMock()
    coll.iter_trajectories.return_value = [t]
    monkeypatch.setattr(OS, "knowledge_errors", lambda home, days: {"t-err"})
    monkeypatch.setattr("ghost_agent.memory.skills.trajectory_may_teach", lambda tr, consumer="": True)
    case = FR.pick_case(coll, tmp_path)
    assert case is not None and case["source_id"] == "t-err" and case["signal"] == "knowledge_errors"


def test_a_passed_turn_keeps_its_note_plain():
    """Review r1: a write that failed then succeeded on retry, verifier PASSED —
    the note must not say "unverified" next to "verifier:passed"."""
    a, store = _wl_agent({"file_system": 1})
    asyncio.run(a._write_project_work_log_safe(
        last_user_content="write the report pdf for the project", final_ai_content="Done — report.pdf is written.",
        execution_failure_count=0, verifier_backfill=("passed", "ok")))
    kw = store.add_work_log.call_args.kwargs
    assert not kw["note"].startswith("[unverified")


def test_learning_warnings_lead_the_overview_section(monkeypatch):
    """Review r1: the shrink cuts this section from its END — the ⚠ lines
    must come first so a cut drops headlines, not warnings."""
    from ghost_agent.tools import introspect as I
    monkeypatch.setattr(I, "_learning_report_cached",
                        lambda md, args, allow_stale=True: ("LESSONS: 3 live\nCALIBRATION: ranking\n⚠ CONSUMER DEAD x", 0))
    out = asyncio.run(I._overview_learning(SimpleNamespace(memory_dir="/tmp", args=None)))
    assert out.index("⚠ CONSUMER DEAD") < out.index("LESSONS:")


def test_the_knowledge_shadow_is_off_by_default(monkeypatch):
    """§4OH: measured precision 0/3, recall 0/2 on its live verdicts, and its
    "errors" seed the failure replay — off unless asked for."""
    from ghost_agent.core.knowledge_shadow import enabled
    monkeypatch.delenv("GHOST_KNOWLEDGE_SHADOW", raising=False)
    assert enabled() is False
    for v in ("1", "true", "on"):
        monkeypatch.setenv("GHOST_KNOWLEDGE_SHADOW", v)
        assert enabled() is True
    monkeypatch.setenv("GHOST_KNOWLEDGE_SHADOW", "0")
    assert enabled() is False
