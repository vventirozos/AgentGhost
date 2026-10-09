"""§4MS (operator, 2026-10-09: "retarget to my failures"): self-play practises
only what the OWNER's real turns got wrong.

Measured over 14 days: 341 self-play runs, 93% passed on the first try, 35
lesson saves (~8 distinct), hydrated into an owner turn twice — both
irrelevant; the curriculum (CSV, log parsing, Playwright) matched nothing the
owner asks. Now a self-play run needs a SEED: a real owner turn from the last
`OWNER_SEED_DAYS` days that FAILED and has not been practised yet. No seed —
no self-play (and no counterfactual replay: the whole phase waits).

A seed is used once: its id is kept in ``selfplay/owner_seeds_used.json``.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("GhostAgent")

OWNER_SEED_DAYS = 14.0
USED_FILENAME = "owner_seeds_used.json"
_USED_MAX = 2000
_REQUEST_CHARS = 400


def _used_path(home_system: Path) -> Path:
    return Path(home_system) / "selfplay" / USED_FILENAME


def load_used(home_system) -> set:
    if home_system is None:
        return set()
    try:
        return set(json.loads(_used_path(home_system).read_text(encoding="utf-8")) or [])
    except Exception:  # noqa: BLE001 — no file: nothing used yet
        return set()


def mark_used(home_system, traj_id: str) -> None:
    if home_system is None:
        return
    p = _used_path(home_system)
    try:
        used = list(load_used(home_system))
        if traj_id and traj_id not in used:
            used.append(str(traj_id))
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(used[-_USED_MAX:]), encoding="utf-8")
        tmp.replace(p)
    except Exception:  # noqa: BLE001 — a lost mark costs one repeat practice
        logger.warning("owner seed mark not saved for %s", traj_id, exc_info=True)


def is_owner_failure(traj) -> bool:
    """A real owner turn (the frontier consumer's REAL_ONLY kinds, no member,
    no probe, no interruption) whose outcome is FAILED — and that was a
    SKILL failure: it used tools, and it was not aborted (r1: the live
    seeds were an aborted "quick brown fox", a refused request, a sandbox
    clean-up and a forget confirmation — nothing to practise)."""
    try:
        from ..memory.skills import trajectory_may_teach
        if not trajectory_may_teach(traj, consumer="frontier"):
            return False
        if str(getattr(traj, "outcome", "") or "").lower() != "failed":
            return False
        if "[ATTEMPT_ABORTED" in str(getattr(traj, "final_response", "") or ""):
            return False
        return bool(getattr(traj, "tool_calls", None))
    except Exception:  # noqa: BLE001
        return False


def pick_owner_failure_seed(collector, home_system: Path, *, days: float = OWNER_SEED_DAYS,
                            now: Optional[float] = None) -> Optional[dict]:
    """The newest unpractised owner failure as a self-play seed, or None.
    The seed's ``hint`` asks for a self-contained challenge of the SAME
    shape — the skill the turn needed, not its data."""
    if collector is None:
        return None
    used = load_used(home_system)
    from ..memory.skills import iter_teachable
    try:
        trajs = list(iter_teachable(collector.iter_trajectories(since_days=days), consumer="frontier"))
    except TypeError:
        trajs = list(iter_teachable(collector.iter_trajectories(), consumer="frontier"))
    except Exception:  # noqa: BLE001
        return None
    for t in reversed(trajs):                     # newest first
        tid = str(getattr(t, "id", "") or "")
        if not tid or tid in used or not is_owner_failure(t):
            continue
        req = " ".join(str(getattr(t, "user_request", "") or "").split())[:_REQUEST_CHARS]
        why = " ".join(str(getattr(t, "failure_reason", "") or "").split())[:200]
        if not req:
            continue
        hint = ("Build the challenge around the SKILL this real request needed — the agent got it "
                f"wrong recently. Request: \"{req}\"" + (f" What went wrong: {why}." if why else "")
                + " Make the challenge self-contained, offline and machine-checkable; use invented "
                  "data, never the request's names or private details.")
        return {"mode": "owner_failure", "cluster_key": getattr(t, "cluster", None) or None,
                "hint": hint, "source_id": tid, "picked_at": now or time.time()}
    return None
