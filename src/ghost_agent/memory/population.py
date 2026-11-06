"""Which population is writing right now, and which stored turns a learner
may read.

Kept here since the lesson store was removed (§4TD, 2026-11-06 — these lived
in ``memory/skills.py``, which went with it). Nothing in this module reads or
writes a store.

* ``write_origin`` — the population of the request in flight, from the
  request-id contextvar the turn loop sets first thing (§4FB): a real owner
  request, the agent's own machinery, a diagnostic probe, or a non-owner.
* ``usage_credit_blocked`` — may this request bump usage counters (episode
  ``access_count``, vector ``retrieval_count``)?
* ``trajectory_may_teach`` / ``iter_teachable`` — the ONE filter a reader of
  the stored trajectory corpus goes through (today: the experience store).
"""
from __future__ import annotations

#: "user" = a real owner request; "auto" = idle machinery or an internal
#: (sched-/job-/sub-) turn; "probe" = a diagnostic; "member" = a non-owner's
#: turn, on any client.
ORIGIN_USER = "user"
ORIGIN_AUTO = "auto"
ORIGIN_PROBE = "probe"
ORIGIN_MEMBER = "member"


def write_origin() -> str:
    """The population writing RIGHT NOW, from the request-id contextvar — the
    ONE derivation, so no caller has to know. Idle phases run under the
    contextvar default ("SYSTEM") and are "auto"; internal prefixes
    (sched-/job-/sub-) are "auto"; a probe id is "probe"; a member's request
    is "member"; any other live request id is "user". Never raises."""
    try:
        from ..utils.logging import is_probe_request_id, request_id_context
        rid = str(request_id_context.get() or "")
    except Exception:  # noqa: BLE001
        return ORIGIN_AUTO
    if not rid or rid == "SYSTEM":
        return ORIGIN_AUTO
    try:
        if is_probe_request_id(rid):
            return ORIGIN_PROBE
    except Exception:  # noqa: BLE001
        pass
    try:
        from ..utils.logging import requester_is_member
        if requester_is_member():
            return ORIGIN_MEMBER
    except Exception:  # noqa: BLE001
        pass
    try:
        from ..core.autonomous_activity import is_internal_request
        if is_internal_request(rid):
            return ORIGIN_AUTO
    except Exception:  # noqa: BLE001
        pass
    return ORIGIN_USER


def usage_credit_blocked() -> bool:
    """§4KW (fresh audit): may THIS request bump usage counters (episode
    ``access_count``, vector ``retrieval_count``)? Not for a probe, a member
    or an internal (job-/sched-/sub-) turn — a probe's hydration was raising
    the counters that decide eviction on the stores just cleaned of probe
    data. Never raises."""
    try:
        if write_origin() in (ORIGIN_PROBE, ORIGIN_MEMBER):
            return True
        from ..utils.logging import request_id_context
        from ..core.autonomous_activity import is_internal_request
        return bool(is_internal_request(str(request_id_context.get() or "")))
    except Exception:  # noqa: BLE001
        return False


def iter_teachable(trajectories, consumer: str = ""):
    """The trajectories a learner may read: `trajectory_may_teach` applied to
    an iterable (or a collector's `iter_trajectories()` result).

    §4MI: with ``consumer`` named, only real ``user_request`` rows are
    admitted (§4NM: every consumer is real-only). Measured before this: the
    training set held 162 `reflection` copies and 66 coding `leaf` rows
    beside 1,907 real requests — the same request text labelled both "hard"
    (its source FAILED) and "easy" (the reflection re-run)."""
    for t in (trajectories or []):
        if trajectory_may_teach(t, consumer=consumer):
            yield t


def trajectory_may_teach(traj, consumer: str = "") -> bool:
    """The member rule applied to a STORED trajectory: may something be
    learned from it later, by an idle phase? The request-time gate
    (`turn_may_teach`) reads the request context, which idle phases run
    WITHOUT — under "SYSTEM" — so on 2026-09-24 an idle cycle learned from a
    channel member's turn nine hours after the request-time gate had blocked
    it. The trajectory carries the requester's role in
    `extra["requester_role"]` (recorded by `_record_turn_trajectory`); no
    client name is involved. Never raises.

    §4MI: ``consumer`` adds the task-kind rule (only ``user_request`` rows),
    and a reflection COPY is judged by its source's role as well (the copy
    used to carry no role, so a member's text re-entered through it)."""
    try:
        extra = getattr(traj, "extra", None) or {}
        _role = str(extra.get("requester_role") or extra.get("source_requester_role") or "").strip().lower()
        if _role == "member":
            return False
        if str(extra.get("source_task_kind") or "") == "probe":     # r2: a probe's reflection copy
            return False
        # §4MS: a turn cut from outside (disconnect, shutdown) was not the
        # agent's attempt — no reader learns from it
        if "[TURN_INTERRUPTED]" in str(getattr(traj, "final_response", "") or ""):
            return False
        if consumer:
            # §4NM: every consumer is real-only now
            if str(getattr(traj, "task_kind", "") or "") != "user_request":
                return False
        return True
    except Exception:  # noqa: BLE001
        return not consumer


__all__ = ["ORIGIN_USER", "ORIGIN_AUTO", "ORIGIN_PROBE", "ORIGIN_MEMBER", "write_origin",
           "usage_credit_blocked", "iter_teachable", "trajectory_may_teach"]
