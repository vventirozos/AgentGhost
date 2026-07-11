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
that are ACTUALLY live (push transport configured? has a Slack consumer
ever polled?) instead of asserting "sent to Slack" blindly.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from ..core.autonomous_activity import (
    get_activity_log, load_consumer_offset, SEVERITY_NOTIFY,
)

logger = logging.getLogger("GhostAgent")

PHASE = "agent_message"
_MAX_MESSAGE_CHARS = 500

# Spam bound: a runaway loop must not be able to page the operator's phone
# every few seconds. Deliberate reports are rare; 12/hour is generous.
_MAX_PER_HOUR = 12
_sent_timestamps: list = []


def _rate_limited() -> bool:
    now = time.time()
    cutoff = now - 3600.0
    while _sent_timestamps and _sent_timestamps[0] < cutoff:
        _sent_timestamps.pop(0)
    if len(_sent_timestamps) >= _MAX_PER_HOUR:
        return True
    _sent_timestamps.append(now)
    return False


def _delivery_channels(context) -> list:
    """Which legs of the pipeline are actually live — never overclaim."""
    channels = []
    try:
        notifier = getattr(context, "outbound_notifier", None)
        if notifier is not None and getattr(notifier, "configured", False):
            channels.append("push (webhook/ntfy)")
    except Exception:  # noqa: BLE001
        pass
    try:
        consumers_path = (Path(str(context.memory_dir)).parent
                          / "notify_consumers.json")
        if load_consumer_offset(consumers_path, "slack") is not None:
            channels.append("Slack DM (bot poller, ≤30s)")
    except Exception:  # noqa: BLE001
        pass
    channels.append("next-turn digest")
    return channels


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

    log = get_activity_log(context)
    if log is None:
        return ("Error: the activity ledger is not attached — operator "
                "notifications are unavailable in this session.")

    if _rate_limited():
        return (f"Error: notification rate limit reached "
                f"({_MAX_PER_HOUR}/hour) — the message was NOT sent. "
                f"Batch your updates into fewer notifications.")

    ok = log.record(PHASE, message, severity=SEVERITY_NOTIFY)
    if not ok:
        return "Error: failed to write the notification record."

    channels = _delivery_channels(context)
    return (f"Notification queued for the operator via: "
            f"{', '.join(channels)}.\nMessage: {message}")


NOTIFY_OPERATOR_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "notify_operator",
        "description": (
            "Send a short PUSH NOTIFICATION to the operator (Slack DM / "
            "phone) — for when they are NOT watching this conversation. Use "
            "when asked to 'report back', 'notify me', 'ping me on Slack', "
            "or when long background work finishes with something the "
            "operator should hear about now. NOT for normal replies (they "
            "already see your answer here), and NOT a substitute for your "
            "final response — send the one-line headline, keep the detail "
            "in your reply."
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
