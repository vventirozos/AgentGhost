"""§4SX Phase D4: an autopilot build that could not get a lab waits for the labs host, and starts again by itself.

A leaf (``core/coding_loop.py``) may use `lab` for a disposable machine. When the labs host does not answer, or
has no room, the build cannot do its work — that is not the task failing. FAILED is never retried unattended, a
BLOCKED child sets off the failure cascade (the parent blocks and its alternatives are used up, ``planning.py``
``_check_parent_failure``), and a NEEDS_USER rolls the project to NEEDS_USER and stops its autopilot. So the task
goes back to PENDING and the hold is kept in the project's metadata (``lab_waits``: {task_id: {code, why, at}});
the unattended pick skips a held task until ``/v1/health`` answers with room, then the hold lifts. The owner's own
run never skips it (it takes the task, and the claim clears the hold).

Codes: ``labs_down`` (the host could not be reached), ``labs_no_answer`` (a create the host took and never
answered), ``lab_capacity`` (``lab_limit`` / ``budget_exceeded`` / ``disk_low``) and ``labs_not_ready`` / ``labs_no_tor`` (§4UD F15:
the host answered that it cannot make a lab RIGHT NOW — its firewall rules are being re-applied, its published
ports are all leased, the create met a ``busy`` lock; or its Tor gateway is not up. These failed the task, and a
FAILED task is never retried unattended). Nothing here reads prose: the lab tool's ``reason_code`` decides, and
the size a refused create needed arrives as a number (``needed_mb``), not as a line of the result text.

The codes a reader outside this module branches on are ``lab_capacity`` and "anything else" (the batch's stop
reason, `project_advancer`): a new code is a "the labs host could not make a lab now" stop.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional, Set

logger = logging.getLogger("GhostAgent")

#: the lab tool's reason_code → the hold it means
REASON_TO_WAIT = {
    "lab_labs_unreachable": "labs_down",
    # review r2 #14: a host that took the request and hung. review r3 #4: its own code — the create may still have
    # landed (an untracked lab the host's idle TTL ends), so the hold waits NO_ANSWER_COOLDOWN_S and a free slot,
    # not just a health answer (an at-once retry piled up orphan labs)
    "lab_no_answer": "labs_no_answer",
    "lab_lab_limit": "lab_capacity",
    "lab_budget_exceeded": "lab_capacity",
    "lab_disk_low": "lab_capacity",
    # §4UD F15: transient host conditions — a hold, not a failed task
    "lab_tor_unavailable": "labs_no_tor",
    "lab_firewall_missing": "labs_not_ready",
    "lab_ports_exhausted": "labs_not_ready",
    "lab_busy": "labs_not_ready",
}
WAIT_CODES = frozenset(REASON_TO_WAIT.values())
META_KEY = "lab_waits"
#: room the health answer must show before a capacity hold lifts
ROOM_MB = 1024
#: review r3 #1: a create needs disk above the HOST's floor (``disk_floor_mb`` in /v1/health; the host default when an
#: older host does not say) plus this margin — 2 GB lifted a disk_low hold the next create refused again
ROOM_DISK_MB = 10240
DISK_MARGIN_MB = 1024
NO_ANSWER_COOLDOWN_S = 900
#: a ``labs_not_ready`` hold waits at least this long (a Tor bootstrap, the firewall re-apply on the host's next
#: reaper tick, a lease freed by a destroy) before the health answer may lift it — never a tight retry loop
NOT_READY_COOLDOWN_S = 180
PREFIX = "waiting for the labs host"
#: review r2 #6: a task that waited this many times and still could not get a lab goes to the owner
MAX_ROUNDS = 3

_LOCK = threading.Lock()
#: build key → (wait code, needed_mb) — the last "no lab" answer a leaf of that build got (review r2 #7: keyed by the
#: BUILD, not the project — two concurrent builds of one project never read each other's answers)
_SEEN: Dict[str, tuple] = {}

import contextvars as _cv
#: the build key of the leaf running in this context ("<leaf_base>"); set by core/coding_loop.run_leaf_turn
BUILD_KEY: "_cv.ContextVar[str]" = _cv.ContextVar("lab_build_key", default="")
#: the leaf (attempt) id running in this context — the lab inventory row of a lab it creates carries it, and only
#: those labs are destroyed when the leaf ends (review r2 #5)
LEAF_ID: "_cv.ContextVar[str]" = _cv.ContextVar("lab_leaf_id", default="")
#: the leaf ids whose turn (or its end clean-up) is running in THIS process — `leaf_reaper.sweep_ended_leaves`
#: never touches these (§4UD F1). Added and removed by core/coding_loop.run_leaf_turn.
RUNNING_LEAVES: Set[str] = set()


def note_outcome(build_key: Optional[str], reason_code: str, ok: bool, action: str = "", needed_mb: int = 0) -> None:
    """Called by the lab tool for autopilot calls inside a build: remember a "no lab" answer to a CREATE; only a lab
    that was actually created clears it (review r2 #3: a `lab list` after a refused create must not). review r3 #3:
    only a create is noted — a `lab templates` during a blip must not turn a build's real failure into a lab wait.
    ``needed_mb``: the host's own number for a refused create (§4UD F15: it was parsed out of the result text)."""
    if not build_key or action != "create":
        return
    code = REASON_TO_WAIT.get(str(reason_code or ""))
    with _LOCK:
        if code:
            try:
                _need = max(0, int(needed_mb or 0))
            except (TypeError, ValueError):
                _need = 0
            _SEEN[build_key] = (code, _need)
            while len(_SEEN) > 256:                       # a build that later succeeded never takes its note
                _SEEN.pop(next(iter(_SEEN)))
        elif ok and action == "create":
            _SEEN.pop(build_key, None)


def take_seen(build_key: Optional[str]) -> tuple:
    """(code, needed_mb) this build hit and no later create cleared — and forget it. ("", 0) when none."""
    if not build_key:
        return "", 0
    with _LOCK:
        return _SEEN.pop(build_key, None) or ("", 0)


def holds_of(project: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    md = (project or {}).get("metadata") or {}
    h = md.get(META_KEY) if isinstance(md, dict) else None
    return {str(k): v for k, v in h.items() if isinstance(v, dict)} if isinstance(h, dict) else {}


def _mutate(store, project_id: str, fn) -> None:
    """review r2 #8: one atomic read-modify-write of the holds (two ticks never lose each other's)."""
    def _m(md):
        h = md.get(META_KEY) if isinstance(md.get(META_KEY), dict) else {}
        md[META_KEY] = fn(dict(h))
        return md
    if callable(getattr(store, "_atomic_metadata_update", None)):
        store._atomic_metadata_update(project_id, _m)
    else:
        md = dict((store.get_project(project_id) or {}).get("metadata") or {})
        store.update_project(project_id, metadata={META_KEY: _m(md)[META_KEY]})


def rounds_of(store, project_id: str, task_id: str) -> int:
    """How many lab waits this task already had (kept across holds; cleared when the task gets a lab)."""
    try:
        md = (store.get_project(project_id) or {}).get("metadata") or {}
        return int(((md.get(META_KEY + "_rounds") or {}) if isinstance(md, dict) else {}).get(str(task_id)) or 0)
    except Exception:  # noqa: BLE001
        return 0


def set_hold(store, project_id: str, task_id: str, code: str, why: str, needed_mb: int = 0) -> int:
    """Record the hold; returns the task's round number (1 = its first wait)."""
    out = {"n": 0}
    try:
        def _h(h):
            h[str(task_id)] = {"code": code, "why": str(why or "")[:240], "at": time.time(),
                               "needed_mb": int(needed_mb or 0)}
            return h
        _mutate(store, project_id, _h)

        def _r(md):
            r = dict(md.get(META_KEY + "_rounds") or {}) if isinstance(md.get(META_KEY + "_rounds"), dict) else {}
            r[str(task_id)] = out["n"] = int(r.get(str(task_id)) or 0) + 1
            md[META_KEY + "_rounds"] = r
            return md
        if callable(getattr(store, "_atomic_metadata_update", None)):
            store._atomic_metadata_update(project_id, _r)
    except Exception:  # noqa: BLE001
        logger.warning("lab wait not recorded for task %s", task_id, exc_info=True)
    return out["n"]


def clear_rounds(store, project_id: str, task_id: str) -> None:
    try:
        def _r(md):
            r = md.get(META_KEY + "_rounds")
            if isinstance(r, dict) and str(task_id) in r:
                md[META_KEY + "_rounds"] = {k: v for k, v in r.items() if k != str(task_id)}
            return md
        if callable(getattr(store, "_atomic_metadata_update", None)):
            store._atomic_metadata_update(project_id, _r)
    except Exception:  # noqa: BLE001
        logger.warning("lab wait rounds not cleared for %s", task_id, exc_info=True)


def clear_holds(store, project_id: str, task_ids) -> None:
    ids = {str(t) for t in (task_ids or ())}
    if not ids:
        return
    try:
        if not ids & set(holds_of(store.get_project(project_id))):
            return
        _mutate(store, project_id, lambda h: {k: v for k, v in h.items() if k not in ids})
    except Exception:  # noqa: BLE001
        logger.warning("lab wait not cleared for %s", sorted(ids), exc_info=True)


def has_room(health: Any, code: str, needed_mb: int = 0, held_at: Optional[float] = None) -> bool:
    """Does a ``/v1/health`` answer lift a hold of ``code``? ``labs_down``: the host answers and is ok.
    ``labs_no_answer``: also ``NO_ANSWER_COOLDOWN_S`` since the hold and a free lab slot. ``lab_capacity``: a free
    lab slot, ``ROOM_MB`` (or the size the create needed) in the disposable pool and in the host's memory, and disk
    above the host's floor. ``labs_not_ready``: ``NOT_READY_COOLDOWN_S`` since the hold, the host ok (its ``ok``
    is false while firewall rules are missing) and its Tor gateway up (``tor`` == "ok"; a host that does not say
    is not held on it)."""
    if not isinstance(health, dict) or health.get("ok") is not True:
        return False
    if code == "labs_down":
        return True
    if code in ("labs_not_ready", "labs_no_tor"):
        try:
            waited = time.time() - float(held_at or 0) >= NOT_READY_COOLDOWN_S
        except (TypeError, ValueError):
            return False
        # only the hold a missing Tor gateway caused waits for it (a lab with no internet needs none)
        return waited and (code != "labs_no_tor" or str(health.get("tor", "ok") or "") == "ok")
    try:
        if code == "labs_no_answer":
            return (time.time() - float(held_at or 0) >= NO_ANSWER_COOLDOWN_S
                    and int(health["labs"]) < int(health["max_labs"]))
        free_pool = int(health["ram_budget_mb"]) - int(health["kept_mb"]) - int(health["disposable_mb_used"])
        need = max(ROOM_MB, int(needed_mb or 0))          # review r2 #6: the size the refused create needed
        return (int(health["labs"]) < int(health["max_labs"]) and free_pool >= need
                and int(health["mem_available_mb"]) >= need + 512 and int(health["disk_free_mb"]) >= int(health.get("disk_floor_mb") or ROOM_DISK_MB) + DISK_MARGIN_MB)
    except (KeyError, TypeError, ValueError):
        return False


async def held_ids(context, store, project_id: str, project: Optional[Dict[str, Any]] = None) -> Set[str]:
    """The task ids the unattended pick must skip now. One health call, only when there is a hold. A hold
    whose task is gone, or no longer PENDING, is dropped; holds the health answer lifts are dropped and logged.
    Never raises (an error keeps every hold: skipping is the safe side)."""
    holds = holds_of(project if project is not None else store.get_project(project_id))
    if not holds:
        return set()
    try:
        pending = {str(t.get("id")) for t in (store.list_tasks(project_id) or [])
                   if str(t.get("status") or "").upper() in ("PENDING", "READY")}
    except Exception:  # noqa: BLE001
        return set(holds)
    stale = [k for k in holds if k not in pending]
    live = {k: v for k, v in holds.items() if k in pending}
    health, unconfigured = None, False
    if live:
        try:
            from ..tools.labs import _call, labs_configured
            if labs_configured(context):
                st, h = await _call(context, "GET", "/v1/health", origin="autopilot", timeout=8)
                health = h if st == 200 else None
            else:
                unconfigured = True           # no labs host any more: nothing to wait for
        except Exception:  # noqa: BLE001
            health = None
    lifted = [k for k, v in live.items()
              if unconfigured or has_room(health, str(v.get("code") or ""), int(v.get("needed_mb") or 0),
                                          held_at=float(v.get("at") or 0))]
    if stale or lifted:
        clear_holds(store, project_id, stale + lifted)
        for k in lifted:
            try:
                store.log_event(project_id, k, "lab_wait_lifted", {"code": live[k].get("code")})
            except Exception:  # noqa: BLE001
                pass
    return {k for k in live if k not in lifted}
