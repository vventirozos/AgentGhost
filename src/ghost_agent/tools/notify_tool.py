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
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

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
            "only when configured) — for when they are NOT watching this "
            "conversation. Use when asked to 'report back', 'notify me', "
            "'ping me on Slack', or when long background work finishes with "
            "something the operator should hear about now. NOT for normal "
            "replies, and NOT a substitute for your final response — send "
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
