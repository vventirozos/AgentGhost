"""Earn-your-keep prune overrides — the bridge between the measurement harness
(`scripts/earn_keep.py`, which DECIDES a subsystem doesn't earn its keep) and
prod boot (`main.py`, which APPLIES the decision).

The harness writes ``$GHOST_HOME/system/earn_keep/pruned.json`` — a small,
human-editable, fully-reversible record of subsystems it has auto-pruned on a
sustained "doesn't help + costs latency" verdict. At boot, ``main.py`` reads it
and flips the corresponding config off (loudly). Deleting an entry (or the
file) restores the subsystem on the next restart.

This module is deliberately pure stdlib and imports NOTHING from the package,
so ``main.py`` can import it and apply the env-kind prunes BEFORE it imports
``core.agent`` (whose module-level toggle constants read their env at import
time). The single ``SUBSYSTEMS`` catalog below is the one source of truth for
both the harness (which ablation arm measures each subsystem) and the boot-time
apply (which arg/env flips it off).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── The subsystem catalog ───────────────────────────────────────────────────
# One entry per independently-ablatable cognitive subsystem.
#   arm            — the ablation-matrix config name ("full_no_<x>") whose
#                    paired delta vs "full" measures this subsystem's marginal
#                    contribution (scripts/ablation_paired.py CONFIG_FLAGS).
#   kind           — "arg" (flip an argparse attribute) or "env" (set an env
#                    var read by a module-level toggle constant).
#   target         — the arg attribute name, or the env var name.
#   disabled_value — the value that turns the subsystem OFF.
#   protected      — True → measured/reported but NEVER auto-pruned (a proven
#                    or correctness-load-bearing subsystem).
#   costs          — True → carries a latency/compute cost, so pruning it when
#                    it doesn't help is a real win. A free-but-useless subsystem
#                    (costs=False) is never auto-pruned (harmless to keep).
#   track          — "A" (in-session, single-shot paired) or "B" (cross-session
#                    / idle-loop). Track A is measurable without an overnight run.
SUBSYSTEMS: Dict[str, Dict[str, Any]] = {
    "metacog": {
        "arm": "full_no_metacog", "kind": "arg", "target": "enable_metacog",
        "disabled_value": False, "protected": False, "costs": True, "track": "A",
    },
    "deep_reason": {
        "arm": "full_no_deepreason", "kind": "arg", "target": "deep_reason",
        "disabled_value": False, "protected": False, "costs": True, "track": "A",
    },
    "preflight": {
        "arm": "full_no_preflight", "kind": "arg", "target": "enable_preflight_guard",
        "disabled_value": False, "protected": False, "costs": True, "track": "A",
    },
    "self_model": {
        "arm": "full_no_selfmodel", "kind": "arg", "target": "no_self_model",
        "disabled_value": True, "protected": False, "costs": False, "track": "A",
    },
    "workspace_model": {
        "arm": "full_no_workspacemodel", "kind": "arg", "target": "no_workspace_model",
        "disabled_value": True, "protected": False, "costs": False, "track": "A",
    },
    "hypothesis": {
        "arm": "full_no_hypothesis", "kind": "env", "target": "GHOST_HYPOTHESIS_GROUNDING",
        "disabled_value": "0", "protected": False, "costs": True, "track": "A",
    },
    # Protected — measured for the record, never auto-pruned.
    "verifier": {
        "arm": "full_no_verifier", "kind": "arg", "target": "no_verifier",
        "disabled_value": True, "protected": True, "costs": True, "track": "A",
    },
    "memory": {
        "arm": "thin", "kind": "arg", "target": "no_memory",
        "disabled_value": True, "protected": True, "costs": False, "track": "B",
    },
    # Track B (idle / cross-session) — arg-toggled, so prod-apply works today;
    # measured only once the Track-B LOO is wired (Phase 2).
    "reflection": {
        "arm": "full_no_reflection", "kind": "arg", "target": "no_reflection",
        "disabled_value": True, "protected": False, "costs": False, "track": "B",
    },
    # dream / self-play run real LLM work in the idle window (memory
    # consolidation, self-play sessions) → costs=True: pruning a costed idle
    # loop that doesn't earn its keep frees genuine compute.
    "dream": {
        "arm": "full_no_dream", "kind": "arg", "target": "no_dream",
        "disabled_value": True, "protected": False, "costs": True, "track": "B",
    },
    "self_play": {
        "arm": "full_no_selfplay", "kind": "arg", "target": "no_self_play",
        "disabled_value": True, "protected": False, "costs": True, "track": "B",
    },
}

PROTECTED = frozenset(n for n, s in SUBSYSTEMS.items() if s["protected"])


def _pruned_path(ghost_home) -> Optional[Path]:
    if not ghost_home:
        return None
    return Path(ghost_home) / "system" / "earn_keep" / "pruned.json"


def load_pruned(ghost_home) -> Dict[str, Any]:
    """Read the prune-override record. Returns {} on any problem (absent file —
    the common case — malformed JSON, unreadable). NEVER raises: a bad override
    file must never break boot."""
    try:
        p = _pruned_path(ghost_home)
        if not p or not p.exists():
            return {}
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_pruned(ghost_home, pruned: Dict[str, Any]) -> None:
    p = _pruned_path(ghost_home)
    if not p:
        raise ValueError("GHOST_HOME is required to save prune overrides")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(pruned, indent=2))
    os.replace(tmp, p)


def record_prune(ghost_home, subsystem: str, evidence: Dict[str, Any],
                 pruned_at: str) -> Dict[str, Any]:
    """Add/refresh a prune entry. Refuses to prune a protected or unknown
    subsystem (the harness enforces this too — belt and suspenders). ``pruned_at``
    is passed in (callers stamp the time; this module stays clock-free/testable)."""
    if subsystem not in SUBSYSTEMS:
        raise ValueError(f"unknown subsystem {subsystem!r}")
    if subsystem in PROTECTED:
        raise ValueError(f"{subsystem!r} is PROTECTED — never auto-prune")
    pruned = load_pruned(ghost_home)
    pruned[subsystem] = {"pruned_at": pruned_at, "evidence": evidence}
    save_pruned(ghost_home, pruned)
    return pruned


def _valid_targets(pruned: Dict[str, Any], kind: str) -> List[str]:
    """Pruned subsystems of the requested kind, skipping unknown/protected
    names (a hand-edited file could contain anything)."""
    out = []
    for name in pruned:
        spec = SUBSYSTEMS.get(name)
        if spec and spec["kind"] == kind and name not in PROTECTED:
            out.append(name)
    return out


def apply_env_prunes(pruned: Dict[str, Any], environ=os.environ) -> List[str]:
    """Set the env vars for env-kind pruned subsystems. MUST run before
    ``core.agent`` is imported (its toggle constants read env at import). Returns
    the names applied. Never raises."""
    applied = []
    try:
        for name in _valid_targets(pruned, "env"):
            spec = SUBSYSTEMS[name]
            environ[spec["target"]] = str(spec["disabled_value"])
            applied.append(name)
    except Exception:
        pass
    return applied


def apply_arg_prunes(args, pruned: Dict[str, Any]) -> List[str]:
    """Flip the argparse attributes for arg-kind pruned subsystems off. Returns
    the names applied. Never raises — a failure means the un-pruned config boots,
    which is the safe direction."""
    applied = []
    try:
        for name in _valid_targets(pruned, "arg"):
            spec = SUBSYSTEMS[name]
            setattr(args, spec["target"], spec["disabled_value"])
            applied.append(name)
    except Exception:
        pass
    return applied
