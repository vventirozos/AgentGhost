"""§4SU Phase 3 — why a task is waiting, as FIELDS (not text marks).

Before this module the reason a task waited lived in its ``result_summary``
text: "… — not closed unattended" (a hold the owner's run takes back),
"human gate: …" (a park only the owner releases), "not closed: <gate>: …"
(a roll-up refusal), "flagged for human review" (a needs-user park). Thirteen
string checks read those marks, and most review MAJORs in §4SU Phases 0–2
sat in them (a mark quoted as evidence, a mark restated, a mark that lost
its prefix).

A waiting task (status NEEDS_USER or BLOCKED) now carries three columns:

  wait_on    who it waits on:
               owner       — the owner must decide / approve / do it
               owner_turn  — a HOLD: the owner's own run takes it back and runs it
               dependency  — another task (a failed child, an open depends_on)
  wait_code  what kind of wait (``human_gate``, ``review``, ``rollup_<gate>``,
             a hold kind such as ``unverified_build`` / ``model_experiment`` /
             ``gate_form``, ``child_failed`` (the failure cascade's block —
             the only one lifted automatically), ``blocked`` (any other
             block), ``asked``, ``paused``)
  wait_why   the short human reason (also what the owner reads)

Leaving a waiting status clears all three (the store does it, for every
writer). A writer that sets a waiting status with no ``wait_on`` gets the
safe default (``owner`` / ``dependency``) — never a hold, so a missed writer
fails safe (the owner decides), it cannot make a task re-run unattended.

The result_summary keeps its human text; NOTHING reads it as state any more.
The text marks are parsed in exactly one place: :func:`wait_from_marks`, the
one-off migration of rows written before this phase.

The model and the briefing see five states (:func:`view_state`): PENDING,
RUNNING, DONE, FAILED, WAITING(on, why). In storage READY folds into
PENDING, and a task-level PAUSED (a phase the owner paused; never set by the
system — audit 11-02) into NEEDS_USER waiting on the owner (code "paused",
so a paused phase still holds its subtree back); the
stored IN_PROGRESS / NEEDS_USER / BLOCKED names stay (the web console, Slack
and the database keep their vocabulary — operator 11-02).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

WAITING_STATUSES = frozenset({"NEEDS_USER", "BLOCKED"})
WAIT_ON = ("owner", "owner_turn", "dependency")
#: task statuses folded in storage: READY is PENDING; a PAUSED task (a phase
#: the owner paused — never set by the system) is WAITING on the owner
FOLDED = {"READY": "PENDING", "PAUSED": "NEEDS_USER"}
#: the wait a folded PAUSED gets
PAUSED_WAIT = ("owner", "paused", "paused by the owner")

_HELD_TAIL = "not closed unattended"


def fold_status(status: Any) -> str:
    s = str(getattr(status, "value", status) or "").strip().upper()
    return FOLDED.get(s, s)


def default_wait(status: str) -> Tuple[str, str]:
    """(wait_on, wait_code) for a waiting status written with none."""
    raw = str(getattr(status, "value", status) or "").strip().upper()
    if raw == "PAUSED":
        return PAUSED_WAIT[:2]
    # r5: a BLOCKED written with no wait is the model's / a writer's own block
    # ("waiting on the vendor key"), NOT the failure cascade's — only the
    # cascade writes ``child_failed``, the one block lifted automatically
    return ("dependency", "blocked") if fold_status(status) == "BLOCKED" else ("owner", "asked")


def wait_from_marks(status: str, summary: str) -> Tuple[str, str, str]:
    """The ONE reader of the pre-Phase-3 text marks — the row migration.
    Returns (wait_on, wait_code, wait_why); ("", "", "") for a task that is
    not waiting."""
    if str(status or "").strip().upper() == "PAUSED":
        return PAUSED_WAIT
    st = fold_status(status)
    if st not in WAITING_STATUSES:
        return "", "", ""
    text = str(summary or "").strip()
    low = text.lower()
    why = text[:300]
    if st == "BLOCKED":
        return "dependency", "child_failed", why
    if low.endswith(_HELD_TAIL):
        return "owner_turn", "held", why
    if low.startswith("human gate:"):
        return "owner", "human_gate", text[len("human gate:"):].strip()[:300]
    if low.startswith("not closed:"):
        rest = text[len("not closed:"):].strip()
        gate = rest.split(":", 1)[0].strip().lower().replace(" ", "_") if ":" in rest else ""
        return "owner", (f"rollup_{gate}" if gate else "rollup"), rest[:300]
    if low == "flagged for human review":
        return "owner", "review", why
    return "owner", "asked", why


def is_hold(task) -> bool:
    """A hold the owner's own run takes back (``wait_on == owner_turn``)."""
    return _get(task, "wait_on") == "owner_turn" and fold_status(_get(task, "status")) == "NEEDS_USER"


def is_waiting_on_owner(task, code: Optional[str] = None, prefix: Optional[str] = None) -> bool:
    if fold_status(_get(task, "status")) not in WAITING_STATUSES or _get(task, "wait_on") != "owner":
        return False
    c = _get(task, "wait_code")
    if code is not None and c != code:
        return False
    if prefix is not None and not c.startswith(prefix):
        return False
    return True


def view_state(task) -> Dict[str, str]:
    """The five-state view: {state, on?, why?}."""
    st = fold_status(_get(task, "status"))
    if st in WAITING_STATUSES:
        on = _get(task, "wait_on") or default_wait(st)[0]
        out = {"state": "WAITING", "on": on}
        why = _get(task, "wait_why")
        if why:
            out["why"] = why[:200]
        return out
    return {"state": {"IN_PROGRESS": "RUNNING"}.get(st, st)}


def view_label(task) -> str:
    """One short label: ``WAITING(owner_turn: why)``, ``RUNNING``…"""
    v = view_state(task)
    if v["state"] != "WAITING":
        return v["state"]
    return f"WAITING({v['on']}" + (f": {v['why'][:80]}" if v.get("why") else "") + ")"


def _get(task, key: str) -> str:
    v = task.get(key) if isinstance(task, dict) else getattr(task, key, "")
    v = getattr(v, "value", v)          # a TaskStatus enum → its value
    return str(v or "")


__all__ = ["WAITING_STATUSES", "WAIT_ON", "FOLDED", "fold_status", "default_wait",
           "wait_from_marks", "is_hold", "is_waiting_on_owner", "view_state", "view_label"]
