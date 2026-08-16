"""Tests for biological watchdog phase 2.7 (PRM retrain).

Mirrors the structure of test_reflection_biological_tick.py: a small
helper builds a mocked context shaped to trigger phase 2.7, then we
assert the cooldown-anchor / activity-clock / gating invariants.
"""

import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core.agent import GhostAgent
from ghost_agent.distill.schema import Outcome, ToolCall, Trajectory
from ghost_agent.prm.scorer import PRMScorer


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _balanced_corpus():
    """Returns a list of synthetic trajectories balanced enough for
    PRMTrainer to actually fit. ``PRMTrainer`` defaults: min_trajectories=5,
    min_samples=20, min_class_fraction=0.05."""
    passing = [
        Trajectory(
            user_request=f"good {i}",
            outcome=Outcome.PASSED.value,
            tool_calls=[
                ToolCall(name="scratchpad", arguments={"action": "store"})
                for _ in range(3)
            ],
            n_steps=3,
        )
        for i in range(8)
    ]
    failing = [
        Trajectory(
            user_request=f"bad {i}",
            outcome=Outcome.FAILED.value,
            tool_calls=[
                ToolCall(name="execute", arguments={"command": "x"}, error="boom")
                for _ in range(3)
            ],
            n_steps=3,
        )
        for i in range(8)
    ]
    return passing + failing


def _make_ctx(*, idle_secs: float, prm_scorer=None,
              collector=None, args=None, memory_dir=None,
              checkpoint_path=None):
    """Build a mocked context shaped so phases 1 / 2 / 2.5 / 2.6 short-
    circuit and we observe phase 2.7 in isolation."""
    ctx = MagicMock()
    ctx.memory_system = MagicMock()
    # llm_client.foreground_tasks = 0 so the tick runs.
    ctx.llm_client = SimpleNamespace(foreground_tasks=0)
    # Phase 1: journal None → short-circuits.
    ctx.journal = None
    # Phase 2: empty memory collection → dream skipped.
    ctx.memory_system.collection.get = MagicMock(return_value={"ids": []})
    # Phase 2.5: reflector None → reflection skipped.
    ctx.reflector = None
    # last_activity_time positions us in the requested idle window.
    ctx.last_activity_time = datetime.datetime.now() - datetime.timedelta(seconds=idle_secs)

    if args is None:
        args = MagicMock()
        args.model = "test-model"
        # Use the static cooldown by default.
        args.prm_train_cooldown = None
    # Phase 2.7 now SKIPS when the PRM has no live consumer (2026-07-27):
    # both `.score()` (MCTS turn-start, module-gated off) and
    # `.uncertainty()` (frontier self-play) were dead in production, so the
    # retrain was writing a checkpoint nothing read. These tests exercise
    # the training path, so give them a live consumer explicitly — a
    # MagicMock attribute would NOT satisfy the `is True` check by design
    # (that strictness is what stops mocked contexts from silently
    # re-enabling the phase everywhere).
    if not isinstance(getattr(args, "frontier_selfplay", None), bool):
        args.frontier_selfplay = True
    ctx.args = args

    ctx.frontier_tracker = None
    ctx.trajectory_collector = collector
    ctx.prm_scorer = prm_scorer
    ctx.memory_dir = memory_dir
    ctx._prm_checkpoint_path = checkpoint_path
    ctx.mcts_reasoner = None  # not under test here
    return ctx


async def _tick(ctx, *, suppress_other_phases: bool = True):
    """Run one tick. By default, pre-sets every non-PRM cooldown anchor
    to 'just fired' so phase 2.7 is observed in isolation.

    Phase 2.6 (skills_auto) shares ``iter_trajectories`` with phase 2.7,
    so without suppression a test that asserts on collector calls is
    flaky against the order phases fire."""
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    if suppress_other_phases:
        now = datetime.datetime.now()
        agent._last_dream_at = now
        agent._last_reflection_at = now
        agent._last_skills_auto_at = now
        agent._last_selfplay_at = now
    await agent._biological_tick()
    return agent


# ──────────────────────────────────────────────────────────────────────
# Cooldown constant ordering
# ──────────────────────────────────────────────────────────────────────

def test_prm_cooldown_constant_is_long_enough():
    """PRM retrain is more expensive than skills_auto and benefits less
    from immediate refire — its cooldown should be at least as long as
    skills_auto (currently 7200 s)."""
    assert GhostAgent._PRM_TRAIN_COOLDOWN >= GhostAgent._SKILLS_AUTO_COOLDOWN


# ──────────────────────────────────────────────────────────────────────
# Phase fires when wired
# ──────────────────────────────────────────────────────────────────────

async def test_phase_27_fires_when_collector_and_scorer_present(tmp_path: Path):
    collector = MagicMock()
    collector.iter_trajectories = MagicMock(
        side_effect=lambda **kw: iter(_balanced_corpus())
    )
    scorer = PRMScorer()
    args = MagicMock()
    args.model = "test"
    args.prm_train_cooldown = None
    ctx = _make_ctx(
        idle_secs=1200,
        prm_scorer=scorer, collector=collector,
        args=args,
        checkpoint_path=tmp_path / "prm.json",
    )
    agent = await _tick(ctx)
    # Anchor was advanced
    assert agent._last_prm_train_at > datetime.datetime.min
    # Scorer hot-swap occurred.
    assert scorer.has_model is True


async def test_phase_27_skipped_when_collector_missing():
    scorer = PRMScorer()
    ctx = _make_ctx(idle_secs=1200, prm_scorer=scorer, collector=None)
    await _tick(ctx)
    assert scorer.has_model is False


async def test_phase_27_skipped_when_scorer_missing():
    collector = MagicMock()
    collector.iter_trajectories = MagicMock(return_value=iter(_balanced_corpus()))
    ctx = _make_ctx(idle_secs=1200, prm_scorer=None, collector=collector)
    # Should run without raising.
    await _tick(ctx)


# ──────────────────────────────────────────────────────────────────────
# Idle-window gating (mirrors reflection / skills_auto)
# ──────────────────────────────────────────────────────────────────────

async def test_phase_27_does_not_fire_below_900s_idle():
    collector = MagicMock()
    collector.iter_trajectories = MagicMock(return_value=iter(_balanced_corpus()))
    scorer = PRMScorer()
    ctx = _make_ctx(idle_secs=300, prm_scorer=scorer, collector=collector)
    await _tick(ctx)
    assert scorer.has_model is False


async def test_phase_27_does_not_fire_above_3600s_idle():
    """Above 3600 s, phase 3 (self-play) takes over. PRM retrain is
    scoped to the 900-3600 s window."""
    collector = MagicMock()
    collector.iter_trajectories = MagicMock(return_value=iter(_balanced_corpus()))
    scorer = PRMScorer()
    ctx = _make_ctx(idle_secs=4000, prm_scorer=scorer, collector=collector)
    await _tick(ctx)
    assert scorer.has_model is False


# ──────────────────────────────────────────────────────────────────────
# Activity clock + cooldown anchor invariants
# ──────────────────────────────────────────────────────────────────────

async def test_phase_27_never_resets_activity_clock(tmp_path: Path):
    """Same rule as phases 1 / 2 / 2.5 / 2.6: must NOT touch
    ctx.last_activity_time. Otherwise phase 3 (self-play) would never
    fire."""
    collector = MagicMock()
    collector.iter_trajectories = MagicMock(return_value=iter(_balanced_corpus()))
    scorer = PRMScorer()
    ctx = _make_ctx(
        idle_secs=1200, prm_scorer=scorer, collector=collector,
        checkpoint_path=tmp_path / "prm.json",
    )
    activity_before = ctx.last_activity_time
    await _tick(ctx)
    assert ctx.last_activity_time == activity_before


async def test_phase_27_advances_anchor_on_exception(tmp_path: Path):
    """Critical invariant: an exception mid-fit must NOT leave the
    cooldown un-advanced. The collector raising should still result
    in the anchor moving forward — otherwise the failing fit refires
    every 60 s for the rest of the idle window."""
    collector = MagicMock()
    collector.iter_trajectories = MagicMock(
        side_effect=RuntimeError("simulated read failure"),
    )
    scorer = PRMScorer()
    ctx = _make_ctx(
        idle_secs=1200, prm_scorer=scorer, collector=collector,
        checkpoint_path=tmp_path / "prm.json",
    )
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    agent._last_prm_train_at = datetime.datetime.min
    await agent._biological_tick()
    assert agent._last_prm_train_at > datetime.datetime.min


async def test_phase_27_respects_cooldown(tmp_path: Path):
    """Pre-set anchor to 'just fired' — phase must skip and not call
    iter_trajectories."""
    collector = MagicMock()
    collector.iter_trajectories = MagicMock(return_value=iter(_balanced_corpus()))
    scorer = PRMScorer()
    args = MagicMock()
    args.prm_train_cooldown = None
    args.model = "test"
    ctx = _make_ctx(
        idle_secs=1200, prm_scorer=scorer, collector=collector,
        args=args,
        checkpoint_path=tmp_path / "prm.json",
    )
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    # Suppress all phases except 2.7 so we can assert iter_trajectories
    # wasn't called *by phase 2.7*. (Phase 2.6 / skills_auto also
    # consumes iter_trajectories — without the suppression it fires
    # first and the assertion is flaky.)
    now = datetime.datetime.now()
    agent._last_dream_at = now
    agent._last_reflection_at = now
    agent._last_skills_auto_at = now
    agent._last_selfplay_at = now
    agent._last_prm_train_at = now  # cooldown not yet elapsed
    await agent._biological_tick()
    collector.iter_trajectories.assert_not_called()
    assert scorer.has_model is False


async def test_phase_27_honours_user_supplied_cooldown(tmp_path: Path):
    """``--prm-train-cooldown`` overrides the static class constant."""
    collector = MagicMock()
    collector.iter_trajectories = MagicMock(return_value=iter(_balanced_corpus()))
    scorer = PRMScorer()
    args = MagicMock()
    args.prm_train_cooldown = 60  # 1 min — much shorter than the 3 h default
    args.model = "test"
    ctx = _make_ctx(
        idle_secs=1200, prm_scorer=scorer, collector=collector,
        args=args,
        checkpoint_path=tmp_path / "prm.json",
    )
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = ctx
    # Suppress other phases so iter_trajectories is observed only via
    # phase 2.7.
    now = datetime.datetime.now()
    agent._last_dream_at = now
    agent._last_reflection_at = now
    agent._last_skills_auto_at = now
    agent._last_selfplay_at = now
    # PRM anchor 5 minutes ago — within the static 3 h cooldown but
    # past the user's 1-min override.
    agent._last_prm_train_at = now - datetime.timedelta(seconds=300)
    await agent._biological_tick()
    # The user's shorter cooldown should let the phase fire.
    collector.iter_trajectories.assert_called_once()
    assert scorer.has_model is True


# ──────────────────────────────────────────────────────────────────────
# Bail behaviour — bad fits must NOT swap a stale model in
# ──────────────────────────────────────────────────────────────────────

async def test_phase_27_does_not_swap_when_trainer_bails(tmp_path: Path):
    """A trainer that bails (e.g., not enough trajectories) must NOT
    publish a model into the live scorer — otherwise the agent's plan
    scoring would be junk for the next 3 hours."""
    # One trajectory → far below trainer's min_trajectories floor.
    collector = MagicMock()
    collector.iter_trajectories = MagicMock(
        return_value=iter([Trajectory(
            user_request="x",
            outcome=Outcome.PASSED.value,
            tool_calls=[ToolCall(name="a")],
        )])
    )
    scorer = PRMScorer()
    ctx = _make_ctx(
        idle_secs=1200, prm_scorer=scorer, collector=collector,
        checkpoint_path=tmp_path / "prm.json",
    )
    await _tick(ctx)
    # Scorer remains un-trained.
    assert scorer.has_model is False


# ──────────────────────────────────────────────────────────────────────
# MCTS auto-plug-in on first successful fit
# ──────────────────────────────────────────────────────────────────────

async def test_phase_27_plugs_scorer_into_mcts_on_first_fit(tmp_path: Path):
    """Lifespan attached the scorer to context but couldn't yet plug
    it into MCTS because no model was loaded. Phase 2.7's first
    successful fit should bridge them."""
    collector = MagicMock()
    collector.iter_trajectories = MagicMock(
        side_effect=lambda **kw: iter(_balanced_corpus())
    )
    scorer = PRMScorer()
    mcts = MagicMock()
    mcts.prm_scorer = None  # not yet plugged in
    ctx = _make_ctx(
        idle_secs=1200, prm_scorer=scorer, collector=collector,
        checkpoint_path=tmp_path / "prm.json",
    )
    ctx.mcts_reasoner = mcts
    await _tick(ctx)
    assert scorer.has_model is True
    assert mcts.prm_scorer is scorer


# ──────────────────────────────────────────────────────────────────────
# Wire-or-retire: don't train a model nothing reads (2026-07-27)
# ──────────────────────────────────────────────────────────────────────

async def test_phase_27_skips_when_no_consumer_is_live(tmp_path: Path):
    """The PRM's two consumers can BOTH be off in production: `.score()`
    is behind the module-gated MCTS turn-start hint, and `.uncertainty()`
    behind `--frontier-selfplay` (default False, absent from the live
    launcher). In that state the retrain burned an idle slot every
    cooldown to write a checkpoint no code path read — 41 such retrains
    in one recent ledger window — while logging "value model refit",
    which reads like learning progress."""
    collector = MagicMock()
    collector.iter_trajectories = MagicMock(
        side_effect=lambda **kw: iter(_balanced_corpus())
    )
    scorer = PRMScorer()
    args = MagicMock()
    args.model = "test-model"
    args.prm_train_cooldown = None
    args.frontier_selfplay = False          # consumer OFF
    # §4BN / R2 MAJ-2: the PRODUCER is explicitly ON here. `online_update`
    # REFINES an existing model and refuses to bootstrap one, so it can
    # never answer "does anything READ the model?" — counting it would
    # resume training for a model nothing consumes (the 41-retrains
    # defect). §4BM registered exactly that widening; §4BN retracted it.
    #
    # Setting this flag is what makes THIS test the pin on the retraction.
    # Two earlier attempts pinned it structurally instead — a source
    # substring, then an AST walk over the gate expression — and R2 showed
    # the AST version still missed four real widenings (a follow-up
    # statement, a sidecar local, an `or _helper(ctx)`, and deleting the
    # branch outright) while FALSE-failing the honest DRY refactor. Both
    # were lexical proxies for a semantic property; this is the property
    # itself, and it is spelling-independent by construction.
    args.prm_online_update = True
    ctx = _make_ctx(
        idle_secs=1200, prm_scorer=scorer, collector=collector, args=args,
        checkpoint_path=tmp_path / "prm.json",
    )
    await _tick(ctx)
    assert scorer.has_model is False, (
        "trained despite having no consumer — if --prm-online-update was "
        "just added to the retrain gate, that is the §4BN retraction being "
        "undone: it is a producer, not a reader")
    assert not (tmp_path / "prm.json").exists()


async def test_phase_27_skips_when_module_gate_on_but_no_deep_reason(tmp_path: Path):
    """R3 MAJOR-1 — the gate read ONE conjunct of a two-conjunct consumer.

    `.score()`'s call site (core/agent.py, MCTS turn-start) requires
    `_MCTS_TURNSTART_ENABLED and ctx.mcts_reasoner is not None`, and the
    reasoner only exists under --deep-reason. The retrain gate read the
    constant alone, so in THIS configuration it trained and wrote a
    checkpoint while nothing on the box could read a PRM value — the
    41-wasted-retrains defect, live, with three instruments (boot warning,
    learning-health row, skip log) simultaneously reporting that nothing
    reads the model.
    """
    from ghost_agent.core import agent as _ag
    collector = MagicMock()
    collector.iter_trajectories = MagicMock(
        side_effect=lambda **kw: iter(_balanced_corpus())
    )
    scorer = PRMScorer()
    args = MagicMock()
    args.model = "test-model"
    args.prm_train_cooldown = None
    args.frontier_selfplay = False
    args.deep_reason = False   # argparse always supplies a bool
    ctx = _make_ctx(
        idle_secs=1200, prm_scorer=scorer, collector=collector, args=args,
        checkpoint_path=tmp_path / "prm.json",
    )
    ctx.mcts_reasoner = None            # no --deep-reason ⇒ nothing calls .score()
    with patch.object(_ag, "_MCTS_TURNSTART_ENABLED", True):
        await _tick(ctx)
        _why_in_scope = _ag.prm_consumer_why_no_reader(ctx)
    assert scorer.has_model is False, (
        "trained with the module gate ON but no MCTS reasoner — the "
        "constant is NECESSARY, not SUFFICIENT; nothing can call .score()")
    assert not (tmp_path / "prm.json").exists()

    # R4 MAJOR-4: and the SKIP LOG must not blame the conjunct that is
    # already satisfied. It used to hardcode "module-gated off" here,
    # sending the operator to flip a constant that is True in this run —
    # the same defect R3 caught in the boot warning, at the site that fix
    # never grepped for.
    why = _why_in_scope
    assert "--deep-reason is not set" in why
    assert "module-gated off" not in why, \
        "tells the operator to enable something that is already enabled"


def test_the_LIVE_configs_cause_arm_is_pinned():
    """R32 CRIT-1 — the arm the live box actually renders had no pin.

    The equality pin below patches `_MCTS_TURNSTART_ENABLED = True`, a
    config that cannot occur in production (flipping that constant is a
    source edit). The live launcher passes `--deep-reason`, so
    `mcts_reasoner` exists and the gate is False — which takes a DIFFERENT
    arm of `prm_consumer_why_no_reader`, feeding the phase-2.7 skip log
    (the one §4BN message that reaches an operator every ~3h), the twin
    log, and both boot WARNINGs. Appending the §4BM framing to that arm was
    green across 802 tests.

    Pin every arm by equality, with the live one first."""
    import types
    from ghost_agent.core import agent as _ag

    def _why(gate, deep, reasoner, frontier=False, collector=True):
        prev = _ag._MCTS_TURNSTART_ENABLED
        _ag._MCTS_TURNSTART_ENABLED = gate
        try:
            ctx = types.SimpleNamespace(
                mcts_reasoner=object() if reasoner else None,
                trajectory_collector=object() if collector else None,
                args=types.SimpleNamespace(frontier_selfplay=frontier,
                                           deep_reason=deep))
            return _ag.prm_consumer_why_no_reader(ctx)
        finally:
            _ag._MCTS_TURNSTART_ENABLED = prev

    # THE LIVE CONFIG: --deep-reason on, module gate off, no frontier.
    assert _why(False, True, True) == (
        "MCTS turn-start hint is module-gated off"
        " and --frontier-selfplay is not enabled"), _why(False, True, True)
    # gate off, no reasoner, flag not set
    assert _why(False, False, False) == (
        "MCTS turn-start hint is off on both counts "
        "(module-gated off, and --deep-reason is not set)"
        " and --frontier-selfplay is not enabled"), _why(False, False, False)
    # flag set but construction failed
    assert _why(False, True, False) == (
        "--deep-reason WAS set but no MCTS reasoner exists — its "
        "construction failed at boot (see 'Deep Reasoning Failed'), so "
        "nothing can call .score(); the turn-start hint is also "
        "module-gated off"
        " and --frontier-selfplay is not enabled"), _why(False, True, False)
    # gate on, reasoner present → .score() live
    assert _why(True, True, True) == (
        "MCTS turn-start hint is live"
        " and --frontier-selfplay is not enabled"), _why(True, True, True)


async def test_phase_27_skip_actually_logs_and_says_why(tmp_path: Path,
                                                        monkeypatch):
    """R5 MAJOR-1: the skip LOG had no pin at all — the previous test
    computed `prm_consumer_why_no_reader(ctx)` itself and asserted on
    that, pinning the helper and never the delivery. R5 hardcoded the old
    string back into the log (116 green), and then DELETED the entire skip
    `pretty_log` (116 green). "Skip AND say why" is the whole 2026-07-27
    fix and the loudness §4BN exists for; capture the real emission."""
    from ghost_agent.core import agent as _ag
    emitted = []
    monkeypatch.setattr(_ag, "pretty_log",
                        lambda *a, **k: emitted.append((a, k)))
    collector = MagicMock()
    collector.iter_trajectories = MagicMock(
        side_effect=lambda **kw: iter(_balanced_corpus())
    )
    scorer = PRMScorer()
    args = MagicMock()
    args.model = "test-model"
    args.prm_train_cooldown = None
    args.frontier_selfplay = False
    args.deep_reason = False   # argparse always supplies a bool
    ctx = _make_ctx(
        idle_secs=1200, prm_scorer=scorer, collector=collector, args=args,
        checkpoint_path=tmp_path / "prm.json",
    )
    ctx.mcts_reasoner = None
    with patch.object(_ag, "_MCTS_TURNSTART_ENABLED", True):
        await _tick(ctx)
        _why_at_skip = _ag.prm_consumer_why_no_reader(ctx)

    skips = [e for e in emitted if e[0] and e[0][0] == "PRM Retrain"]
    assert skips, ("phase 2.7 skipped SILENTLY — the operator sees an idle "
                   "pass that did nothing, with no reason given")
    body = skips[0][0][1]
    # R29 CRIT-2: this is the ONE §4BN message that actually reaches an
    # operator in production — it fires every ~3h on the live box, and for
    # 28 rounds it was the retracted §4BM framing because the process was
    # stale. It was pinned by substrings while its rendered siblings got
    # equality pins. Derive it here: everything except the cause clause is
    # fixed text, and the cause clause has its own generator.
    _expected_body = (
        "skipped — both value-reading consumers are off ("
        # R30 CRIT-1: this used `_why_at_skip` — the cause helper's OWN
        # return — so both sides moved together and appending the retracted
        # §4BM string to that helper was green across 477 tests. That is
        # verbatim the circularity R29 graded CRIT-1 on the loudness file,
        # committed one artifact over in the same round. Recompute the
        # cause here from the same inputs production uses.
        + ("MCTS turn-start hint is module-gated ON but --deep-reason is "
           "not set, so no reasoner exists to call .score()"
           " and --frontier-selfplay is not enabled")
        + "). Training would produce a checkpoint neither "
        "reads; enable either to resume. "
        "(--prm-online-update is a PRODUCER, not a "
        "consumer — correctly not counted here; see §4BN.)")
    assert body == _expected_body, (
        "the production skip message is not what its cause generator "
        f"derives — text was reworded or appended:\n  got:      {body!r}\n"
        f"  expected: {_expected_body!r}")
    assert "--deep-reason is not set" in body, (
        f"skip log does not name the missing conjunct: {body!r}")
    assert "module-gated off" not in body, (
        "skip log blames the module gate, which is ON in this run — it "
        "sends the operator to enable something already enabled")
    assert "PRODUCER" in body, \
        "skip log no longer records why --prm-online-update is excluded"


async def test_phase_27_resumes_when_both_score_conjuncts_are_live(tmp_path: Path):
    """The other direction: with the module gate ON *and* a reasoner
    present, `.score()` is genuinely reachable and training must resume —
    otherwise the fix above would be an unconditional off switch."""
    from ghost_agent.core import agent as _ag
    collector = MagicMock()
    collector.iter_trajectories = MagicMock(
        side_effect=lambda **kw: iter(_balanced_corpus())
    )
    scorer = PRMScorer()
    args = MagicMock()
    args.model = "test-model"
    args.prm_train_cooldown = None
    args.frontier_selfplay = False
    args.deep_reason = False   # argparse always supplies a bool
    ctx = _make_ctx(
        idle_secs=1200, prm_scorer=scorer, collector=collector, args=args,
        checkpoint_path=tmp_path / "prm.json",
    )
    ctx.mcts_reasoner = MagicMock()
    with patch.object(_ag, "_MCTS_TURNSTART_ENABLED", True):
        await _tick(ctx)
    assert scorer.has_model is True, "both conjuncts live but did not train"


async def test_phase_27_resumes_when_frontier_consumer_enabled(tmp_path: Path):
    """The skip is a runtime check, not a deletion — enabling either
    consumer must resume training with no code change."""
    collector = MagicMock()
    collector.iter_trajectories = MagicMock(
        side_effect=lambda **kw: iter(_balanced_corpus())
    )
    scorer = PRMScorer()
    args = MagicMock()
    args.model = "test-model"
    args.prm_train_cooldown = None
    args.frontier_selfplay = True           # consumer ON
    ctx = _make_ctx(
        idle_secs=1200, prm_scorer=scorer, collector=collector, args=args,
        checkpoint_path=tmp_path / "prm.json",
    )
    await _tick(ctx)
    assert scorer.has_model is True


async def test_phase_27_skip_still_advances_the_cooldown(tmp_path: Path):
    """A skip must consume the cooldown anchor; otherwise the phase
    re-evaluates on every single idle tick."""
    collector = MagicMock()
    collector.iter_trajectories = MagicMock(
        side_effect=lambda **kw: iter(_balanced_corpus())
    )
    args = MagicMock()
    args.model = "test-model"
    args.prm_train_cooldown = None
    args.frontier_selfplay = False
    args.deep_reason = False   # argparse always supplies a bool
    ctx = _make_ctx(
        idle_secs=1200, prm_scorer=PRMScorer(), collector=collector, args=args,
        checkpoint_path=tmp_path / "prm.json",
    )
    agent = await _tick(ctx)
    assert agent._last_prm_train_at > datetime.datetime.min
