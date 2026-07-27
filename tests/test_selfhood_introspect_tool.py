"""Tests for the introspect tool (read-only selfhood snapshot).

The tool's job is to render a deterministic, first-person view of the
agent's selfhood — stats, narrative, recent experiences, and an
IDF-weighted recall over the autobiographical log. It must:

  * degrade cleanly when selfhood is disabled or absent,
  * surface real content from a populated SelfModel,
  * reject unknown / malformed actions without crashing the turn,
  * be wired into the LLM-facing TOOL_DEFINITIONS and the dispatch
    table built by ``get_available_tools``.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ghost_agent.selfhood import SelfModel
from ghost_agent.selfhood.schema import Experience
from ghost_agent.tools.introspect import tool_introspect


async def _populate(sm: SelfModel) -> None:
    """Drop a handful of experiences + state onto a SelfModel.

    Uses the public capture/state APIs so the test mirrors what the
    agent's hot path does, not the on-disk format directly."""
    sm.capture_turn(
        trajectory_id="t-1",
        user_request="help me migrate the postgres schema",
        tool_names=["postgres_admin"],
        outcome="passed",
        final_response="ran the migration",
    )
    sm.capture_turn(
        trajectory_id="t-2",
        user_request="trapdoor functions feel asymmetric",
        tool_names=[],
        outcome="unknown",
        final_response="discussed",
    )
    sm.capture_turn(
        trajectory_id="t-3",
        user_request="debug a flaky parser",
        tool_names=["execute"],
        outcome="failed",
        final_response="ran out of ideas",
        failure_reason="off-by-one in lookahead",
    )
    sm.state.set_mood("curious", "a hard problem is bugging me")
    sm.state.note_open_question("Why do trapdoor functions feel asymmetric?")


async def test_default_action_renders_summary(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    await _populate(sm)
    out = await tool_introspect(self_model=sm)
    assert "Who I am" in out
    assert "Experiences on file: 3" in out
    assert "Recent things I remember doing" in out
    # The summary surfaces the most recent experience.
    assert "parser" in out


async def test_explicit_summary_action_matches_default(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    await _populate(sm)
    default = await tool_introspect(self_model=sm)
    explicit = await tool_introspect(action="summary", self_model=sm)
    assert default == explicit


async def test_stats_action_reports_counts(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    await _populate(sm)
    out = await tool_introspect(action="stats", self_model=sm)
    assert "Experiences on file: 3" in out
    assert "Open questions: 1" in out
    assert "curious" in out
    # Topic clusters should mention at least one bucket the seed turns hit.
    assert "Topic clusters:" in out


async def test_narrative_action_empty_when_none_written(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    out = await tool_introspect(action="narrative", self_model=sm)
    assert "No narrative" in out


async def test_narrative_action_returns_persisted_diary(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    # Write a narrative directly via the summariser's persisted path so
    # the tool surfaces it without invoking the async LLM regeneration.
    sm.narrative.path.parent.mkdir(parents=True, exist_ok=True)
    sm.narrative.path.write_text(
        "I have been chewing on trapdoor functions for days.",
        encoding="utf-8",
    )
    out = await tool_introspect(action="narrative", self_model=sm)
    assert "trapdoor functions" in out


async def test_recent_action_returns_last_n(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    await _populate(sm)
    out = await tool_introspect(action="recent", limit=2, self_model=sm)
    # Most recent two: the trapdoor turn and the parser turn.
    assert "parser" in out
    assert "trapdoor" in out
    assert "postgres" not in out  # falls outside the window


async def test_recent_action_empty_log_is_graceful(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    out = await tool_introspect(action="recent", self_model=sm)
    assert "no experiences" in out.lower()


async def test_recent_limit_clamped_to_max(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    await _populate(sm)
    # An absurd limit must not blow up; clamp logic is internal.
    out = await tool_introspect(action="recent", limit=10_000, self_model=sm)
    # Still gets every experience we wrote (3 turns).
    assert "parser" in out and "trapdoor" in out and "postgres" in out


async def test_recall_action_requires_query(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    await _populate(sm)
    out = await tool_introspect(action="recall", self_model=sm)
    assert "SYSTEM ERROR" in out
    assert "query" in out.lower()


async def test_recall_action_surfaces_relevant_past(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    await _populate(sm)
    out = await tool_introspect(
        action="recall", query="trapdoor", self_model=sm,
    )
    assert "trapdoor" in out.lower()
    # Unrelated turns should NOT bubble up — postgres has nothing to do
    # with trapdoor functions, and the IDF scorer should rank it out.
    assert "postgres" not in out.lower()


async def test_recall_action_no_match_explains_quietly(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    await _populate(sm)
    out = await tool_introspect(
        action="recall", query="quantum gravity", self_model=sm,
    )
    assert "nothing" in out.lower()


async def test_disabled_self_model_is_graceful(tmp_path: Path):
    sm = SelfModel(root=tmp_path, enabled=False)
    out = await tool_introspect(self_model=sm)
    assert "unavailable" in out.lower()


async def test_none_self_model_is_graceful():
    out = await tool_introspect(self_model=None)
    assert "unavailable" in out.lower()


async def test_invalid_action_returns_system_error(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    out = await tool_introspect(action="explode", self_model=sm)
    assert "SYSTEM ERROR" in out
    assert "action" in out.lower()


async def test_internal_exception_is_swallowed(tmp_path: Path):
    """A broken SelfModel internal must not propagate up — the
    introspect path is secondary to the user turn and the agent should
    keep going if recall throws."""
    sm = SelfModel(root=tmp_path)
    await _populate(sm)
    sm.recall_relevant = MagicMock(side_effect=RuntimeError("disk on fire"))
    # The SelfModel.recall_relevant wrapper itself catches; we go one
    # level deeper and stub the underlying autobio.search_my_past so
    # the exception reaches the tool's outer try/except.
    sm.autobio.search_my_past = MagicMock(side_effect=RuntimeError("disk on fire"))
    sm.recall_relevant = SelfModel.recall_relevant.__get__(sm, SelfModel)
    out = await tool_introspect(
        action="recall", query="anything", self_model=sm,
    )
    # SelfModel.recall_relevant catches, so we get the "nothing matches"
    # path, not the outer error formatter — either is acceptable as
    # long as we did not raise.
    assert isinstance(out, str) and out  # never empty, never crashed


# ---------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------


def test_introspect_appears_in_tool_definitions():
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    names = [t["function"]["name"] for t in TOOL_DEFINITIONS]
    assert "introspect" in names
    # The definition must describe the read-only intent and list the
    # supported actions — these are the load-bearing fields the model
    # reads at tool-selection time.
    spec = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "introspect")
    desc = spec["function"]["description"]
    assert "READ-ONLY" in desc
    enum = spec["function"]["parameters"]["properties"]["action"]["enum"]
    assert set(enum) >= {"summary", "stats", "narrative", "recent", "recall"}


def test_introspect_dispatch_lambda_passes_self_model(tmp_path: Path):
    """The dispatcher must bind ``context.self_model`` into the tool —
    this is the wiring that lets the agent introspect at all."""
    from ghost_agent.tools.registry import get_available_tools

    sm = SelfModel(root=tmp_path)
    ctx = SimpleNamespace(
        self_model=sm,
        # All the other context attributes the rest of the registry
        # reaches for; mocked so we can exercise the introspect lambda
        # without standing up a full agent.
        args=SimpleNamespace(
            anonymous=False, max_context=4000, model="qwen", default_db="",
        ),
        tor_proxy=None,
        profile_memory=MagicMock(),
        sandbox_dir=str(tmp_path),
        sandbox_manager=None,
        memory_dir=str(tmp_path),
        memory_system=MagicMock(),
        graph_memory=None,
        skill_memory=MagicMock(),
        llm_client=MagicMock(image_gen_clients=None),
        scratchpad=MagicMock(),
        scheduler=MagicMock(),
        memory_bus=None,
        uncertainty_tracker=None,
        metacog=None,
    )
    tools = get_available_tools(ctx)
    assert "introspect" in tools
    assert callable(tools["introspect"])


# ──────────────────────────────────────────────────────────────────────
# action='activity' — the on-demand background-activity report
# (2026-07-17: the finalize banner went notify-only; this is where the
# routine maintenance answers "what did you do while I was away?")
# ──────────────────────────────────────────────────────────────────────


def _ctx_with_ledger(tmp_path: Path):
    from ghost_agent.core.autonomous_activity import ActivityLog
    log = ActivityLog(tmp_path / "activity.jsonl")
    return SimpleNamespace(activity_log=log), log


async def test_activity_renders_info_and_notify(tmp_path: Path):
    ctx, log = _ctx_with_ledger(tmp_path)
    log.record("dream", "REM cycle ran (memory consolidation)")
    log.record("prm_train", "value model refit on 879 samples")
    log.record("scheduled_task", "'netmon-check': all hosts up",
               severity="notify")
    out = await tool_introspect(action="activity", context=ctx)
    assert "REM cycle ran" in out
    assert "value model refit" in out
    assert "netmon-check" in out


async def test_activity_works_without_self_model(tmp_path: Path):
    """The ledger is not the SelfModel — a disabled selfhood must not
    block the activity report."""
    ctx, log = _ctx_with_ledger(tmp_path)
    log.record("dream", "REM cycle ran")
    out = await tool_introspect(action="activity", context=ctx,
                                self_model=None)
    assert "REM cycle ran" in out
    assert "unavailable" not in out.lower()


async def test_activity_without_ledger_degrades_cleanly():
    out = await tool_introspect(action="activity",
                                context=SimpleNamespace(activity_log=None))
    assert "not attached" in out


async def test_activity_clamps_bad_hours_and_limit(tmp_path: Path):
    ctx, log = _ctx_with_ledger(tmp_path)
    log.record("dream", "REM cycle ran")
    out = await tool_introspect(action="activity", context=ctx,
                                hours="garbage", limit="also-garbage")
    assert "REM cycle ran" in out


def test_activity_in_tool_definition_enum():
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    intro = next(d for d in TOOL_DEFINITIONS
                 if d["function"]["name"] == "introspect")
    assert "activity" in intro["function"]["parameters"][
        "properties"]["action"]["enum"]
    assert "hours" in intro["function"]["parameters"]["properties"]
    assert "while I was away" in intro["function"]["description"]


def test_learning_action_is_described_not_just_enumerated():
    """'learning' was in the enum but absent from the description text, so
    the model could never discover it (tool selection is description-
    driven). The stale 'All reads route through your SelfModel' claim was
    false for activity/learning."""
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    intro = next(d for d in TOOL_DEFINITIONS
                 if d["function"]["name"] == "introspect")
    desc = intro["function"]["description"]
    assert "learning" in intro["function"]["parameters"][
        "properties"]["action"]["enum"]
    assert "'learning'" in desc
    assert "All reads route through your SelfModel" not in desc


async def test_activity_never_raises_even_when_render_blows_up(tmp_path: Path,
                                                               monkeypatch):
    """The activity branch returns BEFORE the selfhood try-block; it must
    carry its own guard so the tool's never-raises contract holds."""
    ctx, log = _ctx_with_ledger(tmp_path)
    log.record("dream", "REM cycle ran")
    import ghost_agent.core.autonomous_activity as aa

    def _boom(*a, **k):
        raise RuntimeError("render exploded")

    monkeypatch.setattr(aa, "render_activity_report", _boom)
    out = await tool_introspect(action="activity", context=ctx)
    assert "Activity report failed" in out
    assert "RuntimeError" in out


async def test_activity_read_failure_is_not_reported_as_quiet(tmp_path: Path,
                                                              monkeypatch):
    """A dead ledger read must render as a read error — NOT as the calm
    'No background activity recorded' a genuinely quiet window produces."""
    ctx, log = _ctx_with_ledger(tmp_path)
    log.record("dream", "REM cycle ran")
    monkeypatch.setattr(log, "read_since",
                        MagicMock(side_effect=OSError("disk gone")))
    out = await tool_introspect(action="activity", context=ctx)
    assert "read error" in out
    assert "No background activity" not in out


async def test_activity_notes_a_window_the_tail_scan_cannot_cover(
        tmp_path: Path, monkeypatch):
    """The tail scan is byte/record-capped; when the requested window
    reaches further back than the scan, the report must say so instead of
    silently under-reporting (no silent caps)."""
    import ghost_agent.tools.introspect as intro_mod
    monkeypatch.setattr(intro_mod, "_ACTIVITY_TAIL_BYTES", 512)
    ctx, log = _ctx_with_ledger(tmp_path)
    for i in range(60):  # ~6KB of records — far past the 512-byte cap
        log.record("dream", f"REM cycle {i} ran with a reasonably long summary")
    out = await tool_introspect(action="activity", context=ctx, hours=24)
    assert "scan capped at the ledger tail" in out


async def test_activity_no_truncation_note_when_window_is_covered(
        tmp_path: Path):
    ctx, log = _ctx_with_ledger(tmp_path)
    for i in range(5):
        log.record("dream", f"REM cycle {i} ran")
    out = await tool_introspect(action="activity", context=ctx)
    assert "scan capped" not in out


# ──────────────────────────────────────────────────────────────────────
# action='learning' — the learning-health telemetry hookup
# ──────────────────────────────────────────────────────────────────────


async def test_learning_renders_report(tmp_path: Path):
    md = tmp_path / "memory"
    md.mkdir()
    (md / "skills_playbook.json").write_text("[]")
    out = await tool_introspect(action="learning",
                                context=SimpleNamespace(memory_dir=md))
    assert "LEARNING HEALTH" in out


async def test_learning_works_without_self_model(tmp_path: Path):
    """Like 'activity', 'learning' reads stores, not the SelfModel — a
    disabled selfhood must not block it."""
    md = tmp_path / "memory"
    md.mkdir()
    out = await tool_introspect(action="learning", self_model=None,
                                context=SimpleNamespace(memory_dir=md))
    assert "LEARNING HEALTH" in out


async def test_learning_without_memory_dir_degrades():
    out = await tool_introspect(action="learning",
                                context=SimpleNamespace(memory_dir=None))
    assert "memory_dir unavailable" in out


async def test_learning_render_failure_degrades(tmp_path: Path, monkeypatch):
    import ghost_agent.core.learning_health as lh

    def _boom(*a, **k):
        raise RuntimeError("report exploded")

    monkeypatch.setattr(lh, "render_learning_health", _boom)
    out = await tool_introspect(action="learning",
                                context=SimpleNamespace(memory_dir=tmp_path))
    assert "Learning health unavailable" in out
    assert "RuntimeError" in out


# ──────────────────────────────────────────────────────────────────────
# principles + experience ages in the selfhood renders (2026-07-27)
# ──────────────────────────────────────────────────────────────────────


async def test_stats_reports_principle_count(tmp_path: Path):
    sm = SelfModel(root=tmp_path)
    await _populate(sm)
    sm.note_principle("Prefer reversible actions over clever ones.")
    out = await tool_introspect(action="stats", self_model=sm)
    assert "Operating principles: 1" in out


async def test_summary_surfaces_principles(tmp_path: Path):
    """The values layer is behaviour-shaping by design — 'tell me about
    yourself' must render it, not hide it."""
    sm = SelfModel(root=tmp_path)
    await _populate(sm)
    sm.note_principle("Prefer reversible actions over clever ones.")
    out = await tool_introspect(self_model=sm)
    assert "My operating principles:" in out
    assert "reversible actions" in out


async def test_recent_lines_carry_an_age(tmp_path: Path):
    """Selfhood views were time-blind; each experience line now carries a
    compact age like the activity report does."""
    sm = SelfModel(root=tmp_path)
    await _populate(sm)
    out = await tool_introspect(action="recent", self_model=sm)
    assert "(just now)" in out


def test_exp_age_is_defensive():
    from ghost_agent.tools.introspect import _exp_age
    assert _exp_age("not-a-timestamp") == ""
    assert _exp_age("") == ""
    assert _exp_age(None) == ""


def test_count_and_clusters_track_appends(tmp_path: Path):
    """count()/cluster_counts() now ride the (mtime,size)-cached search
    index — a fresh append must still be reflected (cache invalidation),
    and corrupt lines must not count as experiences."""
    sm = SelfModel(root=tmp_path)
    sm.capture_turn(trajectory_id="t-1", user_request="fix the parser",
                    tool_names=[], outcome="passed", final_response="done")
    assert sm.autobio.count() == 1
    sm.capture_turn(trajectory_id="t-2", user_request="write sql migration",
                    tool_names=[], outcome="passed", final_response="done")
    assert sm.autobio.count() == 2
    # A corrupt line is not an experience.
    with sm.autobio.path.open("a", encoding="utf-8") as fh:
        fh.write("{not json}\n")
    assert sm.autobio.count() == 2
    assert sum(sm.autobio.cluster_counts().values()) <= 2
