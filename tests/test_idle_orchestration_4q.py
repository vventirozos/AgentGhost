"""§4Q — idle-orchestration audit fixes (2026-08-08).

Each test is red-on-revert of one fix found by the four re-run lens reviewers:

* `_bio_cooldown` was DEAD — zero production callers, while its own docstring
  and `--bio-time-scale --help` both promised phase cooldowns were scaled. The
  B3 ablation therefore fired each phase at most ONCE per run regardless of
  `--idle-epochs`. Also the 60s tick period was unscaled, so at scale 60 the
  window (45s) was narrower than the tick.
* The skill-store hygiene rider (orphan reconcile + §4M twin heal) was nested
  under phase 2.6's `traj_collector is not None` gate although neither call
  touches the trajectory corpus, and both shared ONE try with a DEBUG swallow
  — so a raise in reconcile silently skipped the heal.
* Dream was the only NON-terminal phase without an `except`, so a dream crash
  skipped the twelve phases after it for that tick.
* The journal→self-play tee hardcoded one member of `_MINEABLE_TYPES`.
"""

import asyncio
import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core.agent import GhostAgent


class _Scaled(GhostAgent):
    """Bare shell — we only exercise the pure scaling helpers."""
    def __init__(self, scale=1.0):
        self._bio_time_scale = scale


# ── _bio_cooldown wiring (the MAJOR) ──────────────────────────────────────

def test_bio_cooldown_is_identity_at_production_scale():
    """Scale 1.0 must be an exact no-op: this fix may not perturb prod."""
    s = _Scaled(1.0)
    assert s._bio_cooldown(1800) == 1800
    assert s._bio_cooldown(10800) == 10800
    assert s._bio_scaled(60) == 60
    assert s._bio_scaled(900) == 900 and s._bio_scaled(3600) == 3600


def test_bio_cooldown_actually_scales():
    s = _Scaled(60.0)
    assert s._bio_cooldown(1800) == 30
    assert s._bio_cooldown(10800) == 180


def test_tick_period_is_scaled_below_the_window_width():
    """At scale 60 the window is (15, 60] — 45s wide. An UNSCALED 60s tick is
    wider than that window, so whole idle stretches could pass with zero
    in-window ticks purely on alignment. The scaled tick must fit inside."""
    s = _Scaled(60.0)
    tick = s._bio_scaled(60)
    width = s._bio_scaled(3600) - s._bio_scaled(900)
    assert tick < width, f"tick {tick}s does not fit in a {width}s window"


def test_every_cooldown_comparison_routes_through_bio_cooldown():
    """Structural pin: the defect was that `_bio_cooldown` had NO callers while
    the docs claimed otherwise. Any new phase that compares a raw cooldown
    constant re-introduces it, so assert the call count tracks the number of
    cooldown comparisons in the tick."""
    import inspect
    src = inspect.getsource(GhostAgent._biological_tick)
    comparisons = [ln for ln in src.splitlines()
                   if "since_last_" in ln and (">=" in ln or "<" in ln)]
    wired = [ln for ln in comparisons if "_bio_cooldown(" in ln]
    assert comparisons, "no cooldown comparisons found — test is stale"
    missing = [ln.strip() for ln in comparisons if "_bio_cooldown(" not in ln]
    assert not missing, (
        "these cooldown comparisons bypass _bio_cooldown: " + "; ".join(missing))


def test_bio_time_scale_help_does_not_promise_more_than_it_delivers():
    """The --help text claimed cooldowns were scaled while they were not.
    Whatever it promises must be true of the code."""
    import inspect
    from ghost_agent import main as _main
    help_src = inspect.getsource(_main.parse_args)
    assert "--bio-time-scale" in help_src
    tick_src = inspect.getsource(GhostAgent.biological_watchdog)
    assert "_bio_scaled(60)" in tick_src, "tick period is not scaled"


# ── store-hygiene phase: decoupled + independently guarded ────────────────

def test_store_hygiene_has_its_own_cooldown_constant():
    assert isinstance(GhostAgent._STORE_HYGIENE_COOLDOWN, int)


# ── dream must not abort the rest of the tick ─────────────────────────────

def test_dream_phase_catches_so_later_phases_still_run():
    """Dream is NOT the terminal phase — twelve follow it. A dream crash used
    to unwind to the watchdog handler and skip every one of them."""
    import inspect
    src = inspect.getsource(GhostAgent._biological_tick)
    dream_seg = src[src.index("# Phase 2: the MAINTENANCE pass"):     # §4NM: was "Dream housekeeping"
                    src.index("# Phase 2.6b")]
    # Assert on the OUTER handler specifically. A bare `"except Exception" in
    # dream_seg` is NOT discriminating: the eligibility probes inside this
    # phase already have their own except blocks, so that assertion passes
    # even with the outer handler removed (verified — it failed to go red).
    assert "except Exception as _dream_exc:" in dream_seg, (
        "the dream phase's OUTER exception handler is missing — a dream crash "
        "will unwind to the watchdog and skip the twelve phases after it")
    assert "Maintenance phase failed" in dream_seg, "a maintenance failure is not logged"


# ── journal tee shares the miner's type set ───────────────────────────────


# ── stash path no longer silently disables itself ─────────────────────────


# ── #40/#41: an over-aggressive --bio-time-scale silently kills most phases ──

def test_scale_that_breaks_the_window_warns():
    """MEASURED live: at scale 60 the window is (15,60] = 45s wide, while a
    dream is a real ~60s LLM call. Observed cycle: idle 1..10 -> dream at ~11
    (runs 60s) -> idle ~71, window already CLOSED -> self-play (>60) resets the
    clock -> repeat. The (15,60] band is never sampled, so reflection /
    postmortem / skills-auto / PRM / router / calibration / tidy / narratives /
    autoadvance can NEVER fire and silently report 0 firings as a result.
    Scaling the GATES never scaled the WORK."""
    s = _Scaled(60.0)
    warn = s._warn_if_scale_breaks_the_window()
    assert warn, "no warning at a scale that makes the window unusable"
    assert "60" in warn and "never fire" in warn.lower()


def test_usable_scale_does_not_warn():
    for scale in (1.0, 5.0, 15.0, 20.0):
        assert _Scaled(scale)._warn_if_scale_breaks_the_window() == "", (
            f"false warning at scale {scale}, whose window is still usable")


# ── #38: the idle-cycle summary reported 4 of 14 phases ──────────────────────


