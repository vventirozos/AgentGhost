"""Ghost Agent Slack bot — OWNER-LOCKED single-operator client.

Rewritten 2026-07-11. This bot is the OWNER'S private line to the agent:
it replies to exactly ONE Slack user and ignores everyone else, fail-closed.

Authorization model
-------------------
* The owner is resolved at startup from ``GHOST_SLACK_OWNER`` (a Slack user
  id like ``U0123ABCDEF``) or, failing that, looked up from
  ``GHOST_SLACK_OWNER_EMAIL`` via ``users.lookupByEmail`` (needs the
  ``users:read.email`` scope). With neither resolvable the bot REFUSES TO
  START — an unlocked bot must never run by accident.
* Every inbound event (mention or DM) passes ``is_owner_message``: authored
  by the owner, not a bot, not a message subtype (edits/deletes/joins).
  Anyone else is IGNORED SILENTLY — no "unauthorized" reply, because a
  reply confirms the bot exists and invites probing; attempts are logged
  with user + channel instead.
* Thread context is owner-filtered too: only the owner's messages and the
  bot's own replies are ever forwarded to the agent. A third party posting
  into the same channel thread cannot inject prompt content or smuggle
  files into the sandbox by riding the owner's conversation.

Also fixed in this rewrite (the bot had rotted while unused):
* the payload pinned a long-gone model name — the agent validates it and
  404s anything mismatched; the field is now OMITTED so the server's
  configured model always applies;
* the live-status tailer grepped for ``[{request_id}]``, which the pretty
  stream never prints (full id appears only on the BEGIN frame, then a
  2-char tag) — it never matched a single line. It now arms on the BEGIN
  frame and reads emoji until the END frame, which is sound because agent
  turns are globally serialized (one foreground request at a time). Default
  log path corrected to the agent's live stream;
* file attachments now go through ``POST /api/upload`` (multipart,
  X-Ghost-Key) instead of writing into a locally-mounted sandbox dir, so
  the bot no longer has to share a filesystem with the agent;
* the emoji→status map was refreshed to the current Icons set;
* outbound notifications (the 2026-07-11 "mouth" feature) now default to a
  DM with the OWNER when ``GHOST_NOTIFY_SLACK_CHANNEL`` is unset; set it to
  ``off`` to disable.
"""

import argparse
import asyncio
import logging
import os
import re
import uuid

import httpx
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GhostSlackBot")

app = AsyncApp(token=os.environ.get("SLACK_BOT_TOKEN"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GHOST_API_URL = os.environ.get("GHOST_API_URL", "http://localhost:8000/api/chat")
GHOST_API_BASE = re.sub(r"/api/chat/?$", "", GHOST_API_URL)
# Never ship a default secret — the key must be EXPLICITLY configured. An
# EMPTY value is allowed and means "the agent runs with auth disabled"
# (prod uses --api-key ""): we then send NO X-Ghost-Key header at all. A
# whitespace placeholder used to reach httpx verbatim and broke every call
# with "Illegal header value b' '" (leading/trailing whitespace is illegal
# in a header value), so the key is stripped.
_raw_key = os.environ.get("GHOST_API_KEY")
if _raw_key is None:
    raise SystemExit(
        "GHOST_API_KEY environment variable is required (no default). Set "
        "it to the agent's --api-key value, or explicitly EMPTY "
        "(GHOST_API_KEY=) if the agent runs with auth disabled.")
GHOST_API_KEY = _raw_key.strip()
# The one auth-header source of truth — every agent-API call uses this.
AUTH_HEADERS = {"X-Ghost-Key": GHOST_API_KEY} if GHOST_API_KEY else {}

# The agent's LIVE pretty-log stream (for the status tailer). The old default
# pointed at the bot's own stderr in a path tree that doesn't exist.
DEFAULT_AGENT_LOG = "/Users/vasilis/Data/AI/Logs/ghost-agent.log"
LOG_FILE_PATH = os.environ.get("GHOST_AGENT_LOG", DEFAULT_AGENT_LOG)

MAINTENANCE_MODE = False

# Resolved at startup (see resolve_owner_id / main). The gate treats an
# unresolved owner as "authorize nobody" — fail-closed.
OWNER_ID: str | None = None
BOT_USER_ID: str | None = None

# ---------------------------------------------------------------------------
# Status map — keep in sync with utils/logging.py Icons (first match wins).
# ---------------------------------------------------------------------------
EMOJI_MAP = {
    "💭": "Thinking...",
    "📋": "Planning...",
    "🧩": "Recalling context...",
    "💬": "Asking LLM...",
    "🤖": "LLM responding...",
    "🌐": "Searching web...",
    "🔬": "Deep research...",
    "🧅": "Dark-web search...",
    "🌎": "Browsing...",
    "🐍": "Writing code...",
    "🐚": "Running command...",
    "💾": "Writing file...",
    "📖": "Reading file...",
    "🔍": "Scanning files...",
    "👀": "Inspecting file...",
    "📥": "Downloading...",
    "📝": "Saving memory...",
    "🔎": "Reading memory...",
    "🎨": "Generating image...",
    "📄": "Building PDF...",
    "🔧": "Worker node...",
    "📡": "Background activity...",
    "🎓": "Learning...",
    "🔄": "Retrying...",
    "🔶": "Warning...",
    "🛑": "Stopping...",
    "✅": "Finishing up...",
    "❌": "Hit an error...",
}


# ---------------------------------------------------------------------------
# Owner lock
# ---------------------------------------------------------------------------

def is_owner_message(event: dict, owner_id: str | None) -> bool:
    """The single authorization gate. True ONLY for a normal, human-authored
    message from the owner. Fail-closed on every edge: no resolved owner,
    bot-authored, or any message subtype (edits, deletes, channel joins —
    those carry mutated payloads and must never trigger the agent)."""
    if not owner_id:
        return False
    if not isinstance(event, dict):
        return False
    if event.get("bot_id") or event.get("subtype"):
        return False
    return event.get("user") == owner_id


def _log_unauthorized(kind: str, event: dict) -> None:
    """Attempts are logged, never answered — a reply would confirm the bot
    exists and invite probing. The log line is the operator's audit trail."""
    logger.warning(
        "IGNORED %s from unauthorized user=%s channel=%s",
        kind, event.get("user"), event.get("channel"),
    )


async def resolve_owner_id() -> str | None:
    """GHOST_SLACK_OWNER (explicit user id) wins; otherwise look up
    GHOST_SLACK_OWNER_EMAIL. Returns None when neither resolves — the
    caller refuses to start."""
    explicit = os.environ.get("GHOST_SLACK_OWNER", "").strip()
    if explicit:
        return explicit
    email = os.environ.get("GHOST_SLACK_OWNER_EMAIL", "").strip()
    if email:
        try:
            resp = await app.client.users_lookupByEmail(email=email)
            if resp.get("ok"):
                return resp["user"]["id"]
        except Exception as e:  # noqa: BLE001
            logger.error(
                "users.lookupByEmail failed (%s) — the app may be missing the "
                "users:read.email scope. Set GHOST_SLACK_OWNER to your Slack "
                "user id (U…) instead.", e,
            )
    return None


async def get_bot_user_id() -> str | None:
    global BOT_USER_ID
    if not BOT_USER_ID:
        try:
            auth_test = await app.client.auth_test()
            BOT_USER_ID = auth_test.get("user_id")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to get bot user ID: {e}")
    return BOT_USER_ID


# ---------------------------------------------------------------------------
# Thread context (owner-filtered)
# ---------------------------------------------------------------------------

async def build_thread_context(channel_id: str, thread_ts: str,
                               current_event_ts: str) -> list:
    """LLM message history for a thread — OWNER and BOT messages only.

    The filter is part of the authorization boundary, not a convenience:
    without it, anyone in a shared channel could seed a thread with prompt
    content (or file attachments) that the owner's next mention would
    forward to the agent as trusted history."""
    bot_user_id = await get_bot_user_id()
    context_messages: list = []

    try:
        response = await app.client.conversations_replies(
            channel=channel_id, ts=thread_ts)
        if not response.get("ok"):
            return context_messages

        for msg in response.get("messages", []):
            msg_ts = msg.get("ts")
            try:
                if float(msg_ts) > float(current_event_ts):
                    continue
            except (TypeError, ValueError):
                continue

            text = msg.get("text", "")
            is_current = (msg_ts == current_event_ts)
            is_bot = bool(msg.get("user") == bot_user_id or msg.get("bot_id"))

            if is_bot:
                context_messages.append({"role": "assistant", "content": text})
                continue

            # Owner-only from here. Anyone else's message in the thread is
            # dropped entirely — content AND files.
            if not is_owner_message(msg, OWNER_ID):
                continue

            file_notes = []
            if not is_current:  # the current event's files upload separately
                for f in msg.get("files", []):
                    filename = await upload_file_to_agent(f)
                    if filename:
                        file_notes.append(_file_note(filename))
            if file_notes:
                notes_text = "\n\n" + "\n".join(file_notes)
                text = text + notes_text if text else notes_text.strip()

            if bot_user_id:
                text = re.sub(f"<@{bot_user_id}>", "", text).strip()
            if text:
                context_messages.append({"role": "user", "content": text})

    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to fetch thread context: {e}")

    return context_messages


# ---------------------------------------------------------------------------
# File ingestion: Slack → agent sandbox via /api/upload
# ---------------------------------------------------------------------------

def _file_note(filename: str) -> str:
    return (f"[SYSTEM NOTE: The user attached a file named '{filename}'. It "
            f"has been uploaded to your sandbox. Use your file_system or "
            f"knowledge_base tools to interact with it.]")


async def upload_file_to_agent(file_info: dict) -> str | None:
    """Fetch a Slack attachment and hand it to the agent via /api/upload.

    The old path wrote directly into a locally-mounted sandbox dir, which
    only worked when the bot shared a filesystem (and the exact path) with
    the agent. The API does its own traversal/containment checks and lands
    the file in the active project's scope; we still basename() the Slack
    name (attacker-controlled metadata) as defense-in-depth."""
    url = file_info.get("url_private_download")
    filename = os.path.basename(file_info.get("name") or "")
    if not url or not filename or filename in (".", ".."):
        return None
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            slack_headers = {
                "Authorization": f"Bearer {os.environ.get('SLACK_BOT_TOKEN')}"}
            resp = await client.get(url, headers=slack_headers)
            resp.raise_for_status()
            up = await client.post(
                f"{GHOST_API_BASE}/api/upload",
                headers=AUTH_HEADERS,
                files={"file": (filename, resp.content)},
            )
            if up.status_code == 200:
                return filename
            logger.error(f"/api/upload returned {up.status_code} for {filename}")
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to ingest file {filename}: {e}")
    return None


# ---------------------------------------------------------------------------
# Live status: tail the agent's pretty stream while a request runs
# ---------------------------------------------------------------------------

def scan_log_line(line: str, rid8: str, armed: bool):
    """Pure per-line scanner for the tailer (testable without the loop).

    Returns ``(armed, event)`` where event is None, ``("status", emoji,
    text)`` or ``("end",)``. The pretty stream prints the FULL request id
    only on its BEGIN frame (afterwards just a 2-char tag), so we arm on
    our BEGIN and attribute emoji lines to our request until the END frame
    — sound because agent turns are globally serialized: exactly one
    foreground request is inside a BEGIN/END window at a time."""
    if not armed:
        if rid8 in line and "request started" in line:
            return True, None
        return False, None
    if "request finished" in line:
        return False, ("end",)
    for emoji, status_text in EMOJI_MAP.items():
        if emoji in line:
            return True, ("status", emoji, status_text)
    return True, None


async def tail_logs(request_id: str, say, thread_ts: str | None = None):
    """Post/update a one-line status message in Slack while the agent works.
    Best-effort: any failure here must never affect the real reply."""
    current_status_msg = None
    last_emoji = None
    armed = False
    rid8 = request_id[:8]

    try:
        while not os.path.exists(LOG_FILE_PATH):
            await asyncio.sleep(0.5)

        with open(LOG_FILE_PATH, "r", errors="replace") as log_file:
            log_file.seek(0, os.SEEK_END)
            while True:
                line = log_file.readline()
                if not line:
                    await asyncio.sleep(0.1)
                    continue

                armed, event = scan_log_line(line, rid8, armed)
                if event is None:
                    continue
                if event[0] == "end":
                    break
                _, emoji, status_text = event
                if emoji == last_emoji:
                    continue
                last_emoji = emoji
                msg_text = f"{emoji} {status_text}"
                try:
                    if current_status_msg:
                        await app.client.chat_update(
                            channel=say.channel,
                            ts=current_status_msg["ts"],
                            text=msg_text,
                        )
                    else:
                        current_status_msg = await say(
                            text=msg_text, thread_ts=thread_ts)
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to update status: {e}")
    except asyncio.CancelledError:
        pass
    finally:
        if current_status_msg:
            try:
                await app.client.chat_delete(
                    channel=say.channel, ts=current_status_msg["ts"])
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Formatting + the request pipeline
# ---------------------------------------------------------------------------

def format_for_slack(text: str) -> str:
    """Translates standard Markdown to Slack's mrkdwn format, ignoring code blocks."""
    parts = re.split(r'(```.*?```|`.*?`)', text, flags=re.DOTALL)
    for i in range(len(parts)):
        if i % 2 == 0:
            # Bold: **text** -> *text*
            parts[i] = re.sub(r'\*\*(.*?)\*\*', r'*\1*', parts[i])
            # Links: [Text](URL) -> <URL|Text>
            parts[i] = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<\2|\1>', parts[i])
            # Headers: ### Header -> *Header*
            parts[i] = re.sub(r'^(#{1,6})\s+(.+)$', r'*\2*', parts[i], flags=re.MULTILINE)
    return "".join(parts)


async def _process_message(messages: list, say, thread_ts: str | None = None,
                           event_files: list | None = None):
    request_id = str(uuid.uuid4())[:8]
    log_task = asyncio.create_task(tail_logs(request_id, say, thread_ts))

    try:
        for file_info in (event_files or []):
            filename = await upload_file_to_agent(file_info)
            if filename:
                messages[-1]["content"] += "\n\n" + _file_note(filename)

        async with httpx.AsyncClient(timeout=3600.0) as client:
            # NOTE: no model field — the agent 404s any name that isn't its
            # configured model (the old pinned name broke every request
            # after a model upgrade); omitting it always matches.
            payload = {"messages": messages, "stream": False}
            headers = {**AUTH_HEADERS, "X-Request-ID": request_id}
            response = await client.post(GHOST_API_URL, json=payload,
                                         headers=headers)

            if response.status_code != 200:
                await say(text=f"Error: Agent returned {response.status_code}",
                          thread_ts=thread_ts)
                return

            data = response.json()
            ai_content = data["choices"][0]["message"]["content"]

            # Pull image links out of the reply; they upload natively below.
            images = re.findall(r'!\[.*?\]\(/api/download/([^)]+)\)', ai_content)
            clean_content = re.sub(
                r'!\[.*?\]\(/api/download/[^)]+\)', '', ai_content).strip()
            formatted_content = format_for_slack(clean_content)

            uploaded = []  # (filename, bytes)
            for img_name in images:
                # `img_name` comes from a regex over the agent reply ([^)]+ —
                # may contain / or ..). The /api/download endpoint enforces
                # its own containment; basename() keeps the Slack title sane.
                safe_name = os.path.basename(img_name)
                if not safe_name or safe_name in (".", ".."):
                    continue
                try:
                    dl = await client.get(
                        f"{GHOST_API_BASE}/api/download/{img_name}",
                        headers=AUTH_HEADERS,
                    )
                    if dl.status_code == 200:
                        uploaded.append((safe_name, dl.content))
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to download image {img_name}: {e}")

            if formatted_content:
                await say(text=formatted_content, thread_ts=thread_ts)
            elif not uploaded:
                await say(text="Error: Agent returned an empty response.",
                          thread_ts=thread_ts)
            else:
                await say(text="Here is your image:", thread_ts=thread_ts)

            for safe_name, blob in uploaded:
                try:
                    await app.client.files_upload_v2(
                        channel=say.channel, thread_ts=thread_ts,
                        file=blob, filename=safe_name, title=safe_name,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to upload {safe_name} to Slack: {e}")

    except Exception as e:  # noqa: BLE001
        await say(text=f"System Error: {str(e)}", thread_ts=thread_ts)
    finally:
        log_task.cancel()
        try:
            await log_task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# Event handlers — every entry point passes the owner gate FIRST
# ---------------------------------------------------------------------------

@app.event("app_mention")
async def handle_mention(event, say):
    if not is_owner_message(event, OWNER_ID):
        _log_unauthorized("mention", event)
        return

    thread_ts = event.get("thread_ts") or event.get("ts")
    if MAINTENANCE_MODE:
        await say(text="The agent is currently down for maintenance.",
                  thread_ts=thread_ts)
        return

    messages = await build_thread_context(
        event.get("channel"), thread_ts, event.get("ts"))
    if not messages:
        user_text = re.sub(r"<@.*?>", "", event.get("text", "")).strip()
        messages = [{"role": "user", "content": user_text}]

    await _process_message(messages, say, thread_ts, event.get("files"))


@app.event("message")
async def handle_direct_message(event, say):
    # DMs only; the owner gate also drops bots and message subtypes
    # (edits / deletes / joins).
    if event.get("channel_type") != "im":
        return
    if not is_owner_message(event, OWNER_ID):
        _log_unauthorized("DM", event)
        return

    user_text = event.get("text", "").strip()
    if not user_text and not event.get("files"):
        return

    thread_ts = event.get("thread_ts")
    if MAINTENANCE_MODE:
        await say(text="The agent is currently down for maintenance.",
                  thread_ts=thread_ts)
        return

    fetch_ts = thread_ts or event.get("ts")
    messages = await build_thread_context(
        event.get("channel"), fetch_ts, event.get("ts"))
    if not messages:
        messages = [{"role": "user", "content": user_text}]

    await _process_message(messages, say, thread_ts, event.get("files"))


# ---------------------------------------------------------------------------
# Outbound notifications (2026-07-11): poll the agent's notification feed and
# push notify-severity autonomous events (needs-user project tasks, scheduled-
# turn conclusions, ...) — the PROACTIVE half of the bot; everything above
# only speaks when spoken to. Destination: GHOST_NOTIFY_SLACK_CHANNEL if set
# (a channel id C… or user id U…; "off" disables), else a DM to the OWNER —
# so push works out of the box on an owner-locked bot. The delivery watermark
# lives on the AGENT side (consumer="slack") so bot restarts never
# re-deliver; ack happens after chat_postMessage so a mid-delivery crash
# re-serves rather than drops.
# ---------------------------------------------------------------------------
NOTIFY_CHANNEL = os.environ.get("GHOST_NOTIFY_SLACK_CHANNEL", "").strip()
NOTIFY_POLL_SECONDS = float(os.environ.get("GHOST_NOTIFY_POLL_SECONDS", "30"))
_NOTIFY_CONSUMER = "slack"
_poller_task = None  # strong ref — a bare create_task result can be GC'd

_PHASE_EMOJI = {
    "project": ":raising_hand:",
    "scheduled_task": ":alarm_clock:",
    "agent_message": ":speech_balloon:",
    "service": ":gear:",
    "job": ":package:",
}


def format_notification(rec: dict) -> str:
    phase = str(rec.get("phase", "") or "event")
    summary = str(rec.get("summary", "") or "")
    icon = _PHASE_EMOJI.get(phase, ":satellite_antenna:")
    return f"{icon} *[{phase}]* {summary}"


async def notification_poller(channel: str):
    headers = AUTH_HEADERS
    logger.info(
        f"Notification poller ON → {channel} every {NOTIFY_POLL_SECONDS:.0f}s")
    while True:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.get(
                    f"{GHOST_API_BASE}/api/notifications/pending",
                    params={"consumer": _NOTIFY_CONSUMER, "limit": 20},
                    headers=headers,
                )
                if r.status_code != 200:
                    raise RuntimeError(f"pending poll HTTP {r.status_code}")
                data = r.json()
                records = data.get("records") or []
                watermark = data.get("watermark")
                if records:
                    text = "\n".join(format_notification(rec)
                                     for rec in records)
                    await app.client.chat_postMessage(channel=channel,
                                                      text=text)
                if watermark is not None and (records or data.get("baseline")):
                    await client.post(
                        f"{GHOST_API_BASE}/api/notifications/ack",
                        json={"consumer": _NOTIFY_CONSUMER,
                              "watermark": watermark},
                        headers=headers,
                    )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"notification poller: {e}")
        await asyncio.sleep(NOTIFY_POLL_SECONDS)


async def main():
    global LOG_FILE_PATH, MAINTENANCE_MODE, OWNER_ID, _poller_task
    parser = argparse.ArgumentParser(description="Ghost Agent Slack Bot (owner-locked)")
    parser.add_argument("--log-file", type=str, default=LOG_FILE_PATH,
                        help="Agent pretty-log stream to tail for live status "
                             "(env GHOST_AGENT_LOG)")
    parser.add_argument("--maintenance", type=str, choices=["on", "off"],
                        default="off", help="Enable or disable maintenance mode")
    args = parser.parse_args()

    LOG_FILE_PATH = args.log_file
    if args.maintenance == "on":
        MAINTENANCE_MODE = True
        logger.info("Maintenance mode ENABLED. The bot will not process queries.")

    # Fail-closed owner lock: refuse to start unlocked.
    OWNER_ID = await resolve_owner_id()
    if not OWNER_ID:
        raise SystemExit(
            "Owner lock is unresolved — set GHOST_SLACK_OWNER to your Slack "
            "user id (profile → three dots → 'Copy member ID', looks like "
            "U0123ABCDEF) or GHOST_SLACK_OWNER_EMAIL to your Slack email "
            "(requires the users:read.email scope). The bot will not start "
            "without an owner."
        )
    logger.info(f"Owner lock ACTIVE — replying only to {OWNER_ID}")

    notify_dest = NOTIFY_CHANNEL or OWNER_ID
    if notify_dest.lower() in ("off", "none", "0", "false"):
        logger.info("Notification poller OFF (GHOST_NOTIFY_SLACK_CHANNEL=off)")
    else:
        _poller_task = asyncio.create_task(notification_poller(notify_dest))

    handler = AsyncSocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    await handler.start_async()


if __name__ == "__main__":
    asyncio.run(main())
