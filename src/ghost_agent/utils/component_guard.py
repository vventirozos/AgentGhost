"""Real-component vs test-double guard.

Lives here because both `core.dream` and `core.failure_distill`
need it and neither should import the other.
"""

from __future__ import annotations


def _is_real_component(obj) -> bool:
    """True when ``obj`` is a real ghost_agent component, not a test double.

    ⚠ MUST tolerate BOTH import shapes. Production launches
    ``python -m src.ghost_agent.main`` from the repo root, so every module is
    ``src.ghost_agent.*``; the test suite runs with ``PYTHONPATH=src`` and gets
    ``ghost_agent.*``. The original guard was
    ``type(obj).__module__.startswith("ghost_agent")``, which is therefore
    ALWAYS FALSE IN PRODUCTION and always true under test — so five idle
    subsystems were silently inert on the live agent for weeks while their
    tests passed: failure-cluster distillation, the outcome-gated lesson
    PRUNE, the REM project-digest pass, file-manifest backfill, and the
    journal-mined self-play curriculum. Live proof at the time of the fix:
    `failure_distill_state.json` had never been created, and all 20 records in
    `selfplay/journal_stash.json` were `replayed: False`.
    """
    mod = type(obj).__module__ or ""
    return mod.startswith("ghost_agent") or mod.startswith("src.ghost_agent")


__all__ = ["_is_real_component", "is_real_component"]

is_real_component = _is_real_component
