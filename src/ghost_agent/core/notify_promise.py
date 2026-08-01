"""Project-scoped notify-promise persistence (2026-08-01).

Request 5299219b carried "proceed with all tasks, notify me in slack" and
died 3.9 seconds in (an agent restart landed mid-turn). The ask lived only
in that request's text: the follow-up "resume … proceed" requests re-ran
the work, the project completed — and no ping fired, because the
per-request detector and finish-line backstop can only see the CURRENT
message. An explicit notify ask is a delivery requirement on the WORK, not
on the message that happened to carry it.

So it persists like constraints do (which already replay into every
request): ``capture_promise`` stamps the ask into the project's metadata at
REQUEST START — before the first LLM call, so an early death can't lose it
— and ``fire_promise_if_settled`` delivers when the project reaches
DONE/FAILED, from whichever path gets it there:

  * the finalize chain (both the non-streamed and streamed twins), with
    model-notified dedupe — a successful ``notify_operator`` call or a
    fired per-request backstop consumes the flag silently;
  * main.py's ``on_project_done`` hook, for idle-loop completions that
    have no finalize (the hook stands down while a foreground request is
    active — that request's finalize owns delivery).

The flag is consumed with an ATOMIC pop (``_atomic_metadata_update``'s
BEGIN IMMEDIATE read-modify-write), so exactly one caller delivers even
when the hook and finalize race. Delivery respects ``notify_operator``'s
hourly budget; a rate-limited attempt LEAVES the flag for a later one.
Every function is fail-safe: never raises into a caller.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger("GhostAgent")

PROMISE_KEY = "notify_on_done"
_TERMINAL = ("DONE", "FAILED")


def capture_promise(store, project_id, user_text, req_id: str = "") -> bool:
    """Persist the notify ask on ``project_id``. A repeat ask refreshes the
    stored entry. Skips projects already settled (the reply itself answers
    those). Returns True when the flag is stored."""
    try:
        if store is None or not project_id:
            return False
        proj = store.get_project(project_id)
        if not proj:
            return False
        if str(proj.get("status") or "").upper() in _TERMINAL:
            return False
        entry = {
            "ask": " ".join(str(user_text or "").split())[:200],
            "ts": time.time(),
            "req_id": str(req_id or ""),
        }

        def _mut(meta):
            meta[PROMISE_KEY] = entry
            return meta

        return store._atomic_metadata_update(project_id, _mut) is not None
    except Exception:
        logger.debug("notify-promise capture failed", exc_info=True)
        return False


def peek_promise(store, project_id) -> Optional[dict]:
    """The stored promise entry, or None. Read-only."""
    try:
        proj = store.get_project(project_id) if store else None
        val = ((proj or {}).get("metadata") or {}).get(PROMISE_KEY)
        return dict(val) if isinstance(val, dict) else None
    except Exception:  # noqa: BLE001
        return None


def consume_promise(store, project_id) -> Optional[dict]:
    """Atomically pop the promise — exactly one caller gets the entry."""
    try:
        holder: dict = {}

        def _mut(meta):
            val = meta.pop(PROMISE_KEY, None)
            if isinstance(val, dict):
                holder["v"] = val
            return meta

        store._atomic_metadata_update(project_id, _mut)
        return holder.get("v")
    except Exception:
        logger.debug("notify-promise consume failed", exc_info=True)
        return None


def fire_promise_if_settled(store, activity_log, project_id, *,
                            model_notified: bool = False,
                            req_id: str = "",
                            headline: str = "") -> bool:
    """Deliver the stored promise when ``project_id`` reached DONE/FAILED.

    No-op (flag left in place) while the project is still open, when no
    flag exists, or when delivery is rate-limited — a later attempt keeps
    the promise instead. ``model_notified=True`` (the model already made a
    successful notify_operator call this turn, or the per-request backstop
    fired) consumes the flag WITHOUT a second ping. Returns True only when
    a record was written."""
    try:
        if store is None or not project_id:
            return False
        proj = store.get_project(project_id)
        if not proj:
            return False
        status = str(proj.get("status") or "").upper()
        if status not in _TERMINAL:
            return False
        if peek_promise(store, project_id) is None:
            return False
        if model_notified:
            consume_promise(store, project_id)  # promise kept elsewhere
            return False
        if activity_log is None:
            return False
        from ..tools.notify_tool import _note_sent, _rate_limited
        from .autonomous_activity import SEVERITY_NOTIFY
        if _rate_limited():
            return False  # leave the flag — a later attempt may deliver
        entry = consume_promise(store, project_id)
        if entry is None:
            return False  # another caller won the atomic pop
        title = str(proj.get("title") or project_id)[:60]
        head = " ".join(str(headline or "").split())[:220]
        if status == "DONE":
            msg = (f"Project '{title}' is DONE — "
                   + (head or "all tasks complete."))
        else:
            msg = (f"Project '{title}' FAILED — "
                   + (head or "see chat for details."))
        meta = {"auto": "project notify promise",
                "project_id": str(project_id)}
        if req_id:
            meta["req_id"] = str(req_id)
        ok = activity_log.record("project", msg[:500],
                                 severity=SEVERITY_NOTIFY, **meta)
        if not ok:
            # A transient ledger-write failure must not EAT the promise the
            # atomic pop just consumed — put it back (setdefault: never
            # clobber a newer ask stored in the meantime) so a later settle
            # check retries delivery.
            def _restore(m):
                m.setdefault(PROMISE_KEY, entry)
                return m

            try:
                store._atomic_metadata_update(project_id, _restore)
            except Exception:  # noqa: BLE001
                logger.debug("notify-promise restore failed", exc_info=True)
            return False
        _note_sent()
        try:
            from ..utils.logging import Icons, pretty_log
            pretty_log(
                "Notify Guard",
                f"kept the stored notify promise: project "
                f"'{title}' {status}",
                icon=Icons.NOTIFY_OUT,
            )
        except Exception:  # noqa: BLE001
            pass
        return True
    except Exception:
        logger.debug("notify-promise fire failed", exc_info=True)
        return False


__all__ = ["PROMISE_KEY", "capture_promise", "peek_promise",
           "consume_promise", "fire_promise_if_settled"]
