"""``notify_operator`` — deliberate agent→operator push (2026-07-11).

The outbound pipeline (activity ledger → push transports → Slack poller →
owner DM) shipped earlier today, but the MODEL had no way to write to it:
the only notify-severity producers were automatic (needs-user project
events, scheduled-turn conclusions). So "…and report back in Slack" was an
instruction the agent could not actually follow — the worst failure mode
being a claimed-but-impossible delivery. This tool closes that gap: one
call writes a notify-severity record, and every configured delivery leg
(webhook/ntfy push, the Slack bot's poller, the next-turn digest) takes it
from there with zero new plumbing.

Honesty contract: the returned confirmation names the delivery channels
that are ACTUALLY live instead of asserting "sent to Slack" blindly.

§4TB (2026-11-05) made that contract true. Delivery is by POLLING: a client
(the Slack bot, the web console, a handheld) asks `/api/notifications/pending`
and acks what it showed. The confirmation used to list every client that had
EVER acked — one silent for five days was still "(poller, ≤30s)" — and it
always added "next-turn digest", although a record written in an owner turn
is left out of the digest by design. Now: a client is a channel only while it
is polling (within `CONSUMER_POLLING_WINDOW_S`); one that is not is named as
not polling, with how long ago it last did; the digest is named only when the
digest's own filter would show the record; and the text says "queued" — the
tool cannot know a message was read.

HELD UNTIL THE REQUEST ENDS (2026-11-07, req 298833ef). "Notify me in Slack
when it's done": the model called this after one render, the message went out
at once, and in the next turn the model found the dashboard showed made-up
prices and spent five more minutes fixing them. The owner had a "done ✅" for
work that was not done. Now a call inside an open request is HELD: the
message is written to the ledger when the request ends (`flush_held`, called
from handle_chat's finally), so it reports the finished work. A later call in
the same request REPLACES the held message. Files the request changes after
the message was written are named under it; a request that ends early (Stop,
an error, a shutdown) says so. A call outside a request (an idle job, a
background runner) is sent at once, as before. A held message is lost only if
the process dies without unwinding (SIGKILL, power loss).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from ..core.autonomous_activity import (
    ActivityRecord, CONSUMER_POLLING_WINDOW_S, SEVERITY_NOTIFY, _age_str,
    consumer_polls, digest_shows, get_activity_log, valid_consumer_name,
)

logger = logging.getLogger("GhostAgent")

PHASE = "agent_message"
_MAX_MESSAGE_CHARS = 500

# Spam bound: a runaway loop must not be able to page the operator's phone
# every few seconds. Deliberate reports are rare; 12/hour is generous.
_MAX_PER_HOUR = 12
_sent_timestamps: list = []


def _rate_limited() -> bool:
    """CHECK only — does NOT consume a slot. The slot is committed via
    ``_note_sent`` after the record is actually written, so a failed write
    doesn't burn budget (the old version appended here, so a record-write
    failure still counted against the 12/hour cap)."""
    now = time.time()
    cutoff = now - 3600.0
    while _sent_timestamps and _sent_timestamps[0] < cutoff:
        _sent_timestamps.pop(0)
    return len(_sent_timestamps) >= _MAX_PER_HOUR


def _note_sent() -> None:
    """Commit one slot against the hourly budget — only after a successful send."""
    _sent_timestamps.append(time.time())


# --- held notifications (one per open request) ------------------------------
_HOLD_LOCK = threading.Lock()
#: request id -> None while the request is open and nothing is held, or the
#: held entry {"message", "context", "changed": [labels]}
_OPEN: "OrderedDict[str, Optional[dict]]" = OrderedDict()
_OPEN_CAP = 256
#: how many changed things the note under a sent message names
_CHANGED_NAMES_MAX = 6


def _rid(req_id=None) -> str:
    if req_id is not None:
        return str(req_id or "")
    try:
        from ..utils.logging import request_id_context
        return str(request_id_context.get() or "")
    except Exception:  # noqa: BLE001
        return ""


def open_request(req_id) -> None:
    """handle_chat: a notification called from now on is held until
    `flush_held(req_id)`. Call it only where the matching flush is certain
    (right before the try whose finally flushes)."""
    rid = _rid(req_id)
    if not rid or rid == "SYSTEM":
        return
    with _HOLD_LOCK:
        _OPEN[rid] = _OPEN.get(rid)
        _OPEN.move_to_end(rid)
        while len(_OPEN) > _OPEN_CAP:
            # an open request with nothing held goes first; a held message
            # is dropped only when every entry holds one
            empty = next((k for k, v in _OPEN.items() if v is None and k != rid), None)
            if empty is not None:
                _OPEN.pop(empty)
            else:
                _OPEN.popitem(last=False)


def note_change(label: str, req_id=None) -> None:
    """The request changed something (a file, the project) — named under a
    message held BEFORE it. No-op when nothing is held."""
    rid = _rid(req_id)
    label = " ".join(str(label or "").split())[:120]
    if not rid or not label:
        return
    with _HOLD_LOCK:
        held = _OPEN.get(rid)
        if held is not None and label not in held["changed"]:
            held["changed"].append(label)


def _changed_note(changed) -> str:
    if not changed:
        return ""
    shown = ", ".join(changed[:_CHANGED_NAMES_MAX])
    more = len(changed) - _CHANGED_NAMES_MAX
    return (f" (Changed after this message was written: {shown}"
            + (f", +{more} more" if more > 0 else "") + ".)")


def flush_held(req_id=None, *, ended_early: str = "") -> Optional[str]:
    """handle_chat's finally: close the request and write its held message,
    if any, to the ledger. ``ended_early`` names why the request did not
    finish normally ("it was stopped", "it ended with an error"). Returns the
    text written, or None. Never raises."""
    rid = _rid(req_id)
    if not rid:
        return None
    with _HOLD_LOCK:
        held = _OPEN.pop(rid, None)
    if not held:
        return None
    text = held["message"] + _changed_note(held["changed"])
    if ended_early:
        text += (f" (The request ended early — {ended_early} — so this may not be "
                 f"the final state.)")
    try:
        log = get_activity_log(held["context"])
        if log is None or not log.record(PHASE, text, severity=SEVERITY_NOTIFY,
                                         **({"req_id": rid} if rid != "SYSTEM" else {})):
            logger.warning("held notification for request %s could not be written: %s", rid[:8], text[:120])
            return None
    except Exception:  # noqa: BLE001
        logger.warning("held notification for request %s failed", rid[:8], exc_info=True)
        return None
    _note_sent()
    try:
        from ..utils.logging import pretty_log, Icons
        pretty_log("Notify Operator",
                   "held message sent now the request ended"
                   + (f" — {len(held['changed'])} change(s) named under it" if held["changed"] else "")
                   + (f" — ended early: {ended_early}" if ended_early else ""),
                   icon=Icons.NOTIFY_OUT)
    except Exception:  # noqa: BLE001
        pass
    return text


def _reset_holds_for_tests() -> None:
    with _HOLD_LOCK:
        _OPEN.clear()


#: how many clients the "Not reachable now" line names before it counts the rest
_QUIET_NAMES_MAX = 6


def _registered_consumers(context):
    """``(names, readable)`` — the consumers with a stored position (they
    acknowledged at least once, so a record queued now is served to them),
    from the consumers file, under ANY name: a client called "my phone"
    receives the record like one called "slack" (review r2: hidden, the
    result said nobody was registered). The names are raw — what is shown is
    decided where it is shown. A null position is no position.
    ``readable`` is False when the file is there and cannot be understood."""
    import json as _json
    path = Path(str(context.memory_dir)).parent / "notify_consumers.json"
    if not path.exists():
        return set(), True
    try:
        data = _json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return set(), False
    if not isinstance(data, dict):
        return set(), False
    names = set()
    for n, v in data.items():
        try:
            if isinstance(n, str) and v is not None and int(v) >= 0:
                names.add(n)
        except (TypeError, ValueError):
            continue
    return names, True


def _delivery_channels(context, now=None):
    """``(channels, quiet, registered, readable)`` — the legs that will carry
    a notification queued now; the known clients that will not until they come
    back; whether ANY client is registered; whether the client list could be
    read.

    A client is a channel only while it is POLLING and REGISTERED: it asked
    the pending endpoint within ``CONSUMER_POLLING_WINDOW_S`` and it has a
    stored position (a client that never acknowledged is re-baselined past
    every record on each poll, so it receives nothing). No cadence is claimed:
    a channel is listed with how long ago it last polled. A client is NAMED
    only when its name is a plain token (`valid_consumer_name`); any other is
    counted — "another client (name not shown; …)". Never overclaims; never
    raises."""
    channels, quiet = [], []
    live_unnamed, quiet_unnamed = [], []      # seconds since the last poll / "last polled …" texts
    try:
        notifier = getattr(context, "outbound_notifier", None)
        if notifier is not None and getattr(notifier, "configured", False):
            channels.append("push (webhook/ntfy)")
    except Exception:  # noqa: BLE001
        pass
    try:
        registered, readable = _registered_consumers(context)
    except Exception:  # noqa: BLE001
        registered, readable = set(), False
    try:
        t = time.time() if now is None else float(now)
        polls = consumer_polls(context)
        for n in sorted(registered | set(polls)):        # a set: no name twice
            last = polls.get(n)
            live = last is not None and (t - last) <= CONSUMER_POLLING_WINDOW_S
            named = valid_consumer_name(n)
            if n in registered and live:
                age = max(0, int(t - last))
                if named:
                    channels.append(f"the operator's '{n}' client (last polled {age}s ago)")
                else:
                    live_unnamed.append(age)
            elif n in registered:
                when = f"last polled {_age_str(last, t)}" if last is not None else "no poll since the agent started"
                if named:
                    quiet.append(f"the '{n}' client (not polling now — {when}; it receives this at its next poll)")
                else:
                    quiet_unnamed.append(when)
            elif live and readable and named:
                # (it receives this only if it then acknowledges a starting
                # point it was given BEFORE this message; one that never
                # acknowledges is re-baselined past it on every poll)
                quiet.append(f"the '{n}' client (connected, but not registered yet: it has not "
                             "acknowledged its starting point — it may not receive this)")
        if len(live_unnamed) == 1:
            channels.append(f"another client (name not shown; last polled {live_unnamed[0]}s ago)")
        elif live_unnamed:
            channels.append(f"{len(live_unnamed)} other clients (names not shown; each polled within "
                            f"the last {max(live_unnamed)}s)")
        # first on its line: the names after it are the ones the line may cut
        if len(quiet_unnamed) == 1:
            quiet.insert(0, f"another client (name not shown; not polling now — {quiet_unnamed[0]}; "
                            "it receives this at its next poll)")
        elif quiet_unnamed:
            quiet.insert(0, f"{len(quiet_unnamed)} other clients (names not shown; not polling now — "
                            "they receive this at their next poll)")
    except Exception:  # noqa: BLE001
        pass
    return channels, quiet, bool(registered), readable


def _in_next_digest(message: str, req_id) -> bool:
    """Will a later turn's digest show the record just written? Asked of the
    digest's OWN filter (`digest_shows`, with the banner's notify-only
    setting) on a record shaped like the one the ledger got — a notification
    sent in an owner turn is not shown there (the owner read the reply)."""
    try:
        rec = ActivityRecord(
            ts=time.time(), phase=PHASE, summary=message, severity=SEVERITY_NOTIFY,
            meta=({"req_id": str(req_id)} if req_id and req_id != "SYSTEM" else {}))
        return digest_shows(rec, severities=(SEVERITY_NOTIFY,))
    except Exception:  # noqa: BLE001 — unknown: do not promise it
        return False


async def tool_notify_operator(message: str = None, context=None, **kwargs):
    # --- PARAMETER HALLUCINATION HEALING ---
    message = (message or kwargs.get("text") or kwargs.get("summary")
               or kwargs.get("body") or kwargs.get("content") or "")
    message = " ".join(str(message).split())
    if not message:
        return ("Error: 'message' is required — one short line describing "
                "what the operator should know.")
    if len(message) > _MAX_MESSAGE_CHARS:
        message = message[:_MAX_MESSAGE_CHARS - 1].rstrip() + "…"

    # A probe never pages the owner or spends the hourly quota (§4LZ A-F3:
    # 12 probe calls used the whole quota and the owner's next real
    # notification was refused). It is told what WOULD have been sent.
    try:
        from ..utils.logging import is_probe_request_id, request_id_context, request_origin_context, ORIGIN_PROBE
        if (is_probe_request_id(str(request_id_context.get() or ""))
                or str(request_origin_context.get() or "") == ORIGIN_PROBE):
            from .outcome import ToolOutcome
            # the head says NOT SENT and tells the model what to say: the
            # first wording drew "the message was sent" in the reply (§4LZ probe)
            return ToolOutcome.ok(f"PROBE — NOT SENT. This is a test (probe) request, so nothing "
                                  f"was delivered and nothing was recorded. Tell the user the "
                                  f"notification was NOT sent because this was a test request. "
                                  f"(It would have said: {message})")
    except Exception:  # noqa: BLE001
        pass

    log = get_activity_log(context)
    if log is None:
        return ("Error: the activity ledger is not attached — operator "
                "notifications are unavailable in this session.")

    if _rate_limited():
        return (f"Error: notification rate limit reached "
                f"({_MAX_PER_HOUR}/hour) — the message was NOT sent. "
                f"Batch your updates into fewer notifications.")

    # a notification written after reading outside content says so: a page's
    # "urgent: verify your account at <link>" must not read as the agent's own
    # alert, on the phone or in the next turn's digest (§4MB)
    try:
        from ..utils.provenance import untrusted_seen
        if untrusted_seen() and not message.startswith(OUTSIDE_CONTENT_PREFIX):
            message = OUTSIDE_CONTENT_PREFIX + message
    except Exception:  # noqa: BLE001
        pass

    # Held until the request ends (see the module docstring): the ledger
    # gets it from `flush_held`, with what changed after it.
    _hold_rid = _rid()
    with _HOLD_LOCK:
        _holding = bool(_hold_rid) and _hold_rid in _OPEN
        _replaced = _holding and _OPEN.get(_hold_rid) is not None
        if _holding:
            _OPEN[_hold_rid] = {"message": message, "context": context, "changed": []}
    if _holding:
        try:
            _ch, _q, _reg, _readable = _delivery_channels(context)
        except Exception:  # noqa: BLE001
            _ch, _reg, _readable = [], False, False
        _now = (f"Clients polling now: {', '.join(_ch)}." if _ch
                else "Which client will show it is not known." if not _readable
                else "No client is polling right now — it is delivered when one reconnects."
                if _reg else "No client is registered for notifications yet — it stays readable "
                             "in the activity ledger.")
        return (("Notification HELD (it replaces the one held earlier in this request). "
                 if _replaced else "Notification HELD. ")
                + "It is sent to the operator when this request ends, so it reports the "
                f"finished work. {_now}\nMessage: {message}\n"
                "Only the LAST notify_operator call of this request is sent: if you change, fix "
                "or find anything after this, call notify_operator again with the corrected line. "
                "File writes, project work and changing commands after this call are named "
                "under the message when it is sent. "
                "It goes ONLY to the operator (their own notification channels). No channel and "
                "no other person receives it — if a channel post or a message to someone else "
                "was asked for, tell the user that was NOT done.")

    # Stamp the writing turn's request id so the finalize digest can skip
    # records THIS turn authored — without it, the "while you were away"
    # banner echoes the notification the same reply just sent (observed on
    # the first live test).
    try:
        from ..utils.logging import request_id_context
        req_id = request_id_context.get()
    except Exception:  # noqa: BLE001
        req_id = ""
    ok = log.record(PHASE, message, severity=SEVERITY_NOTIFY,
                    **({"req_id": req_id} if req_id and req_id != "SYSTEM"
                       else {}))
    if not ok:
        return "Error: failed to write the notification record."

    _note_sent()  # commit the rate-limit slot only after a successful write
    channels, quiet, registered, readable = _delivery_channels(context)
    in_digest = _in_next_digest(message, req_id)
    # §4TB: QUEUED, and through what — never "delivered" (the tool cannot know
    # it was read), never a client that is not there, never a digest that will
    # not show it.
    if channels:
        head = (f"Notification queued for the operator via: "
                f"{', '.join(channels + (['next-turn digest'] if in_digest else []))}.")
    elif not readable:
        head = ("Notification queued in the activity ledger; which client will show it is not "
                "known.")
    elif registered:
        head = ("Notification queued; no client is polling right now — it is delivered "
                "when one reconnects.")
    else:
        # review: "delivered when one reconnects" is false when no client ever
        # acknowledged — a first-contact client is baselined PAST this record
        head = ("Notification queued in the activity ledger; no client is registered for "
                "notifications yet — a client that connects later starts after this message "
                "and will not receive it (it stays readable in the activity ledger).")
    if not channels and in_digest:
        head += " It will also show in the next-turn digest."
    if not readable:
        head += "\nClients: the client list could not be read."
    if quiet:
        more = len(quiet) - _QUIET_NAMES_MAX
        head += (f"\nNot reachable now: {'; '.join(quiet[:_QUIET_NAMES_MAX])}"
                 + (f"; +{more} more" if more > 0 else "") + ".")
    # where it goes, and where it does NOT (§4ME F1: "posted in #general" and
    # "sent to @someone" were reported after a DM to the owner)
    return (f"{head}\nMessage: {message}\n"
            f"It goes ONLY to the operator (their own notification channels). No channel and "
            f"no other person receives it — if a channel post or a message to someone else "
            f"was asked for, tell the user that was NOT done.")


#: the label on a notification sent after reading outside content (§4MB)
OUTSIDE_CONTENT_PREFIX = "[written after reading outside content] "


NOTIFY_OPERATOR_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "notify_operator",
        "description": (
            "Queue a short NOTIFICATION for the operator on their own "
            "notification channels (Slack DM, the web console; phone push "
            "only when configured), for when they are NOT watching. It is "
            "sent when this request ENDS; a later call replaces an earlier "
            "one. Use when asked to 'report back', 'notify me', 'ping me on "
            "Slack', or when long work finishes with news for now. NOT for "
            "normal replies, NOT a substitute for your final response — send "
            "the one-line headline, keep the detail in your reply. It "
            "reaches ONLY the operator: it cannot post to a Slack channel "
            "or message anyone else — if asked to, say plainly that it was "
            "NOT done."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": (
                        "One short, self-contained line (≤500 chars) — the "
                        "operator reads it on a phone. State the outcome, "
                        "not the process."
                    ),
                },
            },
            "required": ["message"],
        },
    },
}
