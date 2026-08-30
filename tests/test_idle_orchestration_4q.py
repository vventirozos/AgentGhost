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


def test_hygiene_not_nested_under_trajectory_collector_gate():
    """The heal is a SkillMemory↔vector repair with no trajectory dependency.
    Nested under phase 2.6's collector gate it was dead under
    --no-trajectories, so lessons whose twin went missing stayed dark."""
    import inspect
    src = inspect.getsource(GhostAgent._biological_tick)
    idx_heal = src.index("heal_missing_twins")
    # Walk back to the enclosing phase header and confirm it is 2.6b (its own
    # phase), not the skills-auto phase body.
    header = src.rindex("# Phase ", 0, idx_heal)
    assert "2.6b" in src[header:header + 40], (
        "heal is no longer inside its own phase — check it did not get "
        "re-nested under the trajectory-collector gate")


def test_reconcile_failure_cannot_skip_the_heal():
    """A raise in reconcile_vector_orphans must NOT prevent heal_missing_twins:
    they were in one try, and reconcile genuinely raises (it calls
    _load_playbook, which re-raises OSError by design)."""
    import ast
    import inspect
    import textwrap
    src = inspect.getsource(GhostAgent._biological_tick)
    seg = src[src.index("# Phase 2.6b"):src.index("# Phase 2.7")]
    # ⚠ `seg.count("try:") >= 2` CANNOT SEE NESTING. Moving the heal's try
    # INSIDE the reconcile's try keeps both `try:` tokens, both
    # `logger.warning`s and no `logger.debug` — and it survived 1,420 tests.
    # `reconcile_vector_orphans` calls `_load_playbook`, which re-raises
    # OSError by design, and this project has a live root-owned-file failure
    # class: one unreadable playbook then silently skips `heal_missing_twins`
    # for the process lifetime (the audit that added it measured 36/50
    # lessons dark), with a warning that names reconcile and never mentions
    # healing.
    assert "logger.debug" not in seg, "hygiene failures must not be DEBUG-only"
    assert seg.count("logger.warning") >= 2

    # Parse the WHOLE method and select by line range — a slice of a method
    # body is not independently parseable.
    ded = textwrap.dedent(src)
    lines = ded.split("\n")
    lo = next(i for i, l in enumerate(lines, 1) if "# Phase 2.6b" in l)
    hi = next(i for i, l in enumerate(lines, 1) if "# Phase 2.7" in l)
    tree = ast.parse(ded)
    tries = [n for n in ast.walk(tree)
             if isinstance(n, ast.Try) and lo <= n.lineno < hi]
    assert len(tries) >= 2, "reconcile and heal must have separate try blocks"
    # SIBLINGS, not nested: no `try` may contain another.
    nested = [t for t in tries
              if any(isinstance(x, ast.Try) and x is not t
                     for x in ast.walk(t))]
    assert not nested, (
        "the heal's try is INSIDE the reconcile's try — a reconcile failure "
        "skips the heal entirely, which is the defect this phase split exists "
        "to prevent")
    # ...and each must actually call the thing it guards.
    _calls = [ast.unparse(t) for t in tries]
    assert any("reconcile_vector_orphans" in c for c in _calls), _calls
    assert any("heal_missing_twins" in c for c in _calls), _calls


# ── dream must not abort the rest of the tick ─────────────────────────────

def test_dream_phase_catches_so_later_phases_still_run():
    """Dream is NOT the terminal phase — twelve follow it. A dream crash used
    to unwind to the watchdog handler and skip every one of them."""
    import inspect
    src = inspect.getsource(GhostAgent._biological_tick)
    dream_seg = src[src.index("# Phase 2: Deep REM Dream"):
                    src.index("# Phase 2.5: Reflection")]
    # Assert on the OUTER handler specifically. A bare `"except Exception" in
    # dream_seg` is NOT discriminating: the eligibility probes inside this
    # phase already have their own except blocks, so that assertion passes
    # even with the outer handler removed (verified — it failed to go red).
    assert "except Exception as _dream_exc:" in dream_seg, (
        "the dream phase's OUTER exception handler is missing — a dream crash "
        "will unwind to the watchdog and skip the twelve phases after it")
    assert "Dream phase failed" in dream_seg, "dream failure is not logged"


# ── journal tee shares the miner's type set ───────────────────────────────

def test_tee_uses_the_miners_type_set_not_a_hardcoded_member():
    import inspect
    from ghost_agent.core.journal_challenges import _MINEABLE_TYPES
    src = inspect.getsource(GhostAgent.process_journal_queue)
    # Assert on the FILTER EXPRESSION, not merely on the name appearing
    # somewhere: a bare `"_MINEABLE_TYPES" in src` is satisfied by the import
    # line alone, so it stayed green with the hardcoded filter restored
    # (verified — it failed to go red).
    # Scope to the TEE block. A whole-method check would also catch the
    # legitimate `elif item["type"] == "post_mortem":` consumption dispatch
    # further down, which is unrelated and correct.
    tee = src[src.index("Tee the mineable subset"):src.index("Hippocampus")]
    assert "in _MINEABLE_TYPES]" in tee, (
        "the tee's filter does not use _MINEABLE_TYPES")
    assert '== "post_mortem"' not in tee, (
        "the tee still hardcodes one member of the miner's type set")
    assert "failure" in _MINEABLE_TYPES  # the member that was being dropped


# ── stash path no longer silently disables itself ─────────────────────────

def test_stash_disabled_without_home_is_announced_not_silent(monkeypatch, caplog):
    """The stash IS correctly disabled without GHOST_HOME — a home-relative
    fallback would make test runs and tool processes write into the user's real
    home directory (`test_stash_disabled_without_home` in
    tests/test_dream_bugfixes_2026_07_20.py pins that protection, and it caught
    exactly that regression when this fix was first attempted the wrong way).

    The genuine defect Lens-B named is the SILENCE: the subsystem went inert
    with no operator-visible trace. So the contract is "disabled AND said so"."""
    from ghost_agent.core import journal_challenges as jc
    monkeypatch.delenv("GHOST_HOME", raising=False)
    monkeypatch.setattr(jc, "_WARNED_NO_HOME", False)
    with caplog.at_level("WARNING"):
        assert jc._stash_path() is None, (
            "must stay disabled — a home fallback would write to the real ~")
    assert any("GHOST_HOME" in r.message for r in caplog.records), (
        "going inert must not be silent")


def test_stash_disabled_warning_fires_once_per_process(monkeypatch, caplog):
    from ghost_agent.core import journal_challenges as jc
    monkeypatch.delenv("GHOST_HOME", raising=False)
    monkeypatch.setattr(jc, "_WARNED_NO_HOME", False)
    with caplog.at_level("WARNING"):
        for _ in range(5):
            jc._stash_path()
    hits = [r for r in caplog.records if "GHOST_HOME" in r.message]
    assert len(hits) == 1, f"expected one warning per process, got {len(hits)}"


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


def test_ablation_default_scale_is_usable():
    """The harness default caused the broken measurement; it must self-check."""
    import sys, inspect
    sys.path.insert(0, "scripts")
    import ablation_trackb3 as B3
    src = inspect.getsource(B3.main) if hasattr(B3, "main") else ""
    if not src:
        import re
        src = pathlib_read = open("scripts/ablation_trackb3.py").read()
    import re
    m = re.search(r'"--time-scale".*?default=([0-9.]+)', src, re.S)
    assert m, "could not find the --time-scale default"
    default = float(m.group(1))
    assert _Scaled(default)._warn_if_scale_breaks_the_window() == "", (
        f"the ablation's default --time-scale {default:g} makes most idle "
        "phases structurally unreachable")


# ── #38: the idle-cycle summary reported 4 of 14 phases ──────────────────────

def test_idle_summary_reports_reflection():
    """MEASURED in production 2026-08-09: reflection RAN twice in 4h46m, yet
    the "idle cycle: ran ..." summary never mentioned it — the phase does work
    but never appended to `_idle_ran`. The summary's own docstring claims it
    answers "did the nightly loop dream / REFLECT / self-play", so it was
    lying by omission, and it misled this very investigation."""
    import inspect
    src = inspect.getsource(GhostAgent._biological_tick)
    assert '_idle_ran.append("reflection")' in src


def test_idle_summary_covers_the_phases_that_do_real_work():
    """Guard against the summary drifting back to a partial view.
    §4CB R1 A-F4 / R2: six more phases gained appends (prm, router, tidy,
    selfhood-narrative, stale-questions, workspace-narrative) — removing
    five of them passed every pre-R2 test (lens-A mutant M3), so the full
    label set is pinned here alongside the pre-existing postmortem/bench."""
    import inspect
    src = inspect.getsource(GhostAgent._biological_tick)
    for label in ("dream", "reflection", "skills-auto", "self-play",
                  "store-hygiene", "calibration", "autoadvance",
                  "postmortem", "bench", "prm", "router", "tidy",
                  "selfhood-narrative", "stale-questions",
                  "workspace-narrative"):
        assert f'_idle_ran.append("{label}")' in src, f"{label} is silent"


def test_summary_appends_are_not_on_skip_paths():
    """A phase must be reported only when it WORKS. The reflection phase has a
    fingerprint SKIP that also advances its anchor; appending there would
    report work that never happened."""
    import inspect
    src = inspect.getsource(GhostAgent._biological_tick)
    seg = src[src.index("# Phase 2.5: Reflection"):src.index("# Phase 2.5c")]
    skip_idx = seg.index("_reflection_corpus_fp', None)")
    append_idx = seg.index('_idle_ran.append("reflection")')
    assert append_idx > skip_idx, (
        "the reflection append sits on/before the fingerprint-skip path")
