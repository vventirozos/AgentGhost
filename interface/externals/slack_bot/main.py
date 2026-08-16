"""Ghost Agent Slack bot — owner-anchored client with an OPEN-CHANNEL mode.

Rewritten 2026-07-11 as the OWNER'S private line (reply to exactly ONE user,
fail-closed). 2026-08-13: open-channel mode added AT THE OPERATOR'S REQUEST
to bring real user traffic into the agent's outcome ledgers, plus 👍/👎
reaction feedback that writes human outcome labels via ``/api/feedback``.

Authorization model
-------------------
* The owner is resolved at startup from ``GHOST_SLACK_OWNER`` (a Slack user
  id like ``U0123ABCDEF``) or, failing that, looked up from
  ``GHOST_SLACK_OWNER_EMAIL`` via ``users.lookupByEmail`` (needs the
  ``users:read.email`` scope). With neither resolvable the bot REFUSES TO
  START — even open-channel mode needs the owner as the notification
  destination and the always-authorized feedback labeler.
* ``GHOST_SLACK_OPEN_CHANNEL`` (default ON since 2026-08-13): channel
  mentions from ANY normal human member are answered, and their thread
  messages/files are forwarded as context — channel members are users, by
  operator decision. Bots and message subtypes (edits/deletes/joins) are
  still rejected on every path. Set ``GHOST_SLACK_OPEN_CHANNEL=0`` to
  restore the strict owner lock.
* DMs are OWNER-ONLY in both modes: "everybody in the channel" is a channel
  grant, not a workspace-wide one. Anyone else is IGNORED SILENTLY — no
  "unauthorized" reply, because a reply confirms the bot exists and invites
  probing; attempts are logged with user + channel instead.
* Feedback reactions (👍/👎 on a bot reply) are accepted from the OWNER or
  from the REQUESTER of that specific reply — a third party cannot label
  someone else's turn, and a SHADOW-BANNED user cannot label at all (that
  path writes outcome labels that feed learning, so a message-only ban
  would be cosmetic). Needs the ``reactions:read`` scope + the
  ``reaction_added`` bot event subscription in the Slack app config.

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
import json
import logging
import logging.handlers
import os
import re
import time
import uuid
from collections import OrderedDict

import httpx
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

# Log hygiene (2026-08-01). The old `basicConfig(INFO)` sent EVERYTHING to
# stderr → launchd's ghost-slack-bot.err, which nothing rotates: httpx logs
# every poll request at INFO (2 lines / 30s ≈ 5.8k lines/day), the file hit
# 11 MB in 3 weeks, and the wedge diagnostic this bot's history says to
# watch for ("200-OK polls with zero `delivered` lines", 2026-07-13) was
# buried in its own noise. Now:
#   * INFO and up go to a self-ROTATING file (the .log launchd points at,
#     which had been empty since Jul 11) — 5 MB × 3 backups, no newsyslog
#     or sudo required;
#   * stderr (the launchd .err) gets WARNING+ only, so it stays quiet and
#     crash tracebacks stand out;
#   * httpx/httpcore request-level INFO is silenced outright — the poller
#     logs its own `delivered N` lines and an hourly heartbeat instead.
_LOG_FMT = "%(asctime)s %(levelname)s:%(name)s:%(message)s"
# NOT ghost-slack-bot.log: launchd holds that open as StandardOutPath, and
# a rotation RENAME under launchd's fd would silently divert process stdout
# into a backup (then an unlinked inode) until restart. The rotating INFO
# file is its own sibling path.
_BOT_LOG_PATH = os.environ.get(
    "GHOST_SLACKBOT_LOG",
    "/Users/vasilis/Data/AI/Logs/ghost-slack-bot.info.log")
_stderr_handler = logging.StreamHandler()
_stderr_handler.setLevel(logging.WARNING)
_log_handlers: list = [_stderr_handler]
_file_log_error: str | None = None
if _BOT_LOG_PATH:  # explicitly EMPTY (tests) → stderr only, no file
    try:
        _file_handler = logging.handlers.RotatingFileHandler(
            _BOT_LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=3,
            encoding="utf-8")
        _log_handlers.append(_file_handler)
    except OSError as _e:  # unwritable path must never keep the bot down
        _file_log_error = str(_e)
logging.basicConfig(level=logging.INFO, format=_LOG_FMT,
                    handlers=_log_handlers)
for _noisy in ("httpx", "httpcore"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger("GhostSlackBot")
if _file_log_error:
    # Loud, or the fallback recreates "quiet log = can't tell wedged from
    # healthy" with zero indication: INFO ('delivered N', the heartbeat)
    # would silently go nowhere.
    logger.warning("INFO file logging is DARK — %s unwritable (%s); only "
                   "WARNING+ reaches stderr", _BOT_LOG_PATH, _file_log_error)

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


def _env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in (
        "1", "true", "yes", "on")


# Open-channel mode (2026-08-13, operator-requested): channel mentions from
# any human member are answered. DEFAULT ON — the operator asked for the
# flag to be enabled on startup to bring real traffic into the outcome
# ledgers. GHOST_SLACK_OPEN_CHANNEL=0 restores the strict owner lock.
OPEN_CHANNEL: bool = _env_flag("GHOST_SLACK_OPEN_CHANNEL", "1")

# Resolved at startup (see resolve_owner_id / main). The gate treats an
# unresolved owner as "authorize nobody" — fail-closed.
OWNER_ID: str | None = None
BOT_USER_ID: str | None = None
# The bot's own workspace/org ids, from auth.test (R1 review; widened to a
# SET in R2 — Enterprise Grid payloads can carry the enterprise E… id where
# a T… id is expected, which would have locked out legitimate same-org
# users). Slack Connect delivers events for EXTERNAL-org users with a
# perfectly normal `user` id — without a team check, "any human member"
# silently meant "anyone Slack delivers an event for". The check is
# BEST-EFFORT: an event that carries no team field passes (many payload
# shapes omit it), so it narrows the Slack-Connect surface rather than
# sealing it. Empty set (auth.test failed) → main() forces the owner lock
# for the run rather than running open-channel blind (R2: fail-closed).
BOT_TEAM_IDS: set = set()


def _env_id_set(name: str) -> frozenset:
    """Comma/space-separated Slack user ids from the environment."""
    raw = str(os.environ.get(name, "") or "")
    return frozenset(p for p in raw.replace(",", " ").split() if p)


# Shadow ban (operator-requested). Messages, mentions and REACTIONS from
# these Slack user ids are dropped as if the bot never saw them — no reply,
# no error, nothing that distinguishes being banned from the bot simply
# being idle. That silence is not new behaviour: `is_authorized_message`
# already answers nothing on rejection, deliberately, because a reply
# "confirms the bot exists and invites probing". A ban just routes a
# specific user down the path that already exists.
#
# STATIC by choice: read once at startup, so a ban takes a bot restart.
# SILENT by choice: attempts log at DEBUG only, which keeps them out of the
# operator's live stream (WARNING+) while leaving them recoverable.
#
# The OWNER can never be banned — a typo in this variable must not be able
# to lock the operator out of their own bot.
SHADOW_BANNED: frozenset = _env_id_set("GHOST_SLACK_SHADOW_BAN")

# ---------------------------------------------------------------------------
# Status map — keep in sync with utils/logging.py Icons (first match wins).
# ---------------------------------------------------------------------------
EMOJI_MAP = {
    "💭": "Thinking...",
    "🧠": "Reasoning...",
    "📋": "Planning...",
    "🔮": "Consulting precedent...",
    "🧩": "Recalling context...",
    "🧭": "Routing...",
    "🎯": "Targeting...",
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
    "🧪": "Verifying...",
    "🔗": "Checking constraints...",
    "🔒": "Guarding...",
    "🐘": "Querying Postgres...",
    "📣": "Notifying...",
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


def is_shadow_banned(user_id: str | None, owner_id: str | None = None) -> bool:
    """True when this user's events must be dropped as if unseen.

    The owner is exempt unconditionally: a mistyped id in
    ``GHOST_SLACK_SHADOW_BAN`` must not be able to lock the operator out
    of their own bot.
    """
    if not user_id:
        return False
    if owner_id and user_id == owner_id:
        return False
    return user_id in SHADOW_BANNED


def _is_open_surface(channel_id: str | None) -> bool:
    """True only for ordinary channels (``C…`` conversation ids) — the ONLY
    surface the open grant covers. IMs are ``D…`` and group DMs / legacy
    private channels are ``G…``; treating anything non-``C`` as private is
    the fail-closed default (R1 review CRIT: ``app_mention`` fires in IMs
    and group DMs too, and carries no ``channel_type``, so without a
    surface check "@Ghost" in a stranger's DM bypassed the owner lock)."""
    return str(channel_id or "").startswith("C")


def is_authorized_message(event: dict, owner_id: str | None,
                          open_channel: bool) -> bool:
    """Authorization gate for message-shaped events.

    Owner-locked (``open_channel=False``): identical to
    ``is_owner_message``. Open-channel mode: any normal HUMAN-authored
    message from the bot's OWN workspace passes — bot-authored messages
    and message subtypes (edits/deletes/joins carry mutated payloads) are
    rejected in BOTH modes, an event with no ``user`` never passes, and a
    Slack-Connect external-org author (``team``/``user_team`` differing
    from ``BOT_TEAM_ID``) never rides the open grant.

    Callers own the SURFACE decision: pass
    ``open_channel and _is_open_surface(channel_id)`` so DMs/group DMs
    stay owner-only in both modes — the open grant is scoped to channels
    the operator put the bot in, not to private lines."""
    if not isinstance(event, dict):
        return False
    if event.get("bot_id") or event.get("subtype"):
        return False
    if is_shadow_banned(event.get("user"), owner_id):
        logger.debug("shadow-banned message user=%s channel=%s",
                     event.get("user"), event.get("channel"))
        return False
    if open_channel and event.get("user"):
        ev_team = event.get("user_team") or event.get("team")
        if BOT_TEAM_IDS and ev_team and ev_team not in BOT_TEAM_IDS:
            return is_owner_message(event, owner_id)
        return True
    return is_owner_message(event, owner_id)


def _log_unauthorized(kind: str, event: dict) -> None:
    """Attempts are logged, never answered — a reply would confirm the bot
    exists and invite probing. The log line is the operator's audit trail.

    Only a GENUINE stranger attempt earns WARNING (which reaches the
    launchd ``.err``): subtype-only rejections are routine plumbing — a
    ``message_changed`` in a DM carries ``channel_type: im`` and no
    top-level ``user``, so every edit used to spam
    ``IGNORED DM from unauthorized user=None`` into the quiet stream the
    2026-08-01 log-hygiene work protects (R1 review). Subtype rejections
    log at INFO *with the subtype name* (R2 review: DEBUG made the
    dropped-``file_share``-DM class invisible — a hole should be visible
    in the log even when it is accepted)."""
    if event.get("subtype"):
        # Allowlist (R3 review): blanket-INFO re-created the exact spam R1
        # removed — the bot's own tail_logs chat_update/chat_delete calls
        # fire message_changed/message_deleted events per emoji transition,
        # 10-40 per tool-heavy DM turn. Only the subtypes that represent a
        # REAL dropped human message stay visible.
        if event.get("subtype") in ("file_share", "thread_broadcast"):
            logger.info("ignored %s subtype=%s channel=%s user=%s",
                        kind, event.get("subtype"), event.get("channel"),
                        event.get("user"))
        else:
            logger.debug("ignored %s subtype=%s channel=%s",
                         kind, event.get("subtype"), event.get("channel"))
        return
    if event.get("bot_id") or not event.get("user"):
        logger.debug("ignored %s (bot/userless) channel=%s",
                     kind, event.get("channel"))
        return
    logger.warning(
        "IGNORED %s from unauthorized user=%s channel=%s",
        kind, event.get("user"), event.get("channel"),
    )


# ---------------------------------------------------------------------------
# Reply index + reaction feedback (2026-08-13)
#
# Every reply the bot posts is remembered as (channel, ts) → {req_id,
# requester}, so a later 👍/👎 reaction on that message can be turned into a
# human outcome label via the agent's /api/feedback. Bounded, and persisted
# best-effort to a small JSON file so labels survive a bot restart (reactions
# often arrive minutes after the reply). Explicitly EMPTY path (tests)
# disables persistence.
# ---------------------------------------------------------------------------
REPLY_INDEX_PATH = os.environ.get(
    "GHOST_SLACK_REPLY_INDEX",
    "/Users/vasilis/Data/AI/Logs/ghost-slack-reply-index.json")
_REPLY_INDEX: "OrderedDict[str, dict]" = OrderedDict()
_REPLY_INDEX_MAX = 500

# Reaction name → feedback signal. Slack sends skin-tone variants as
# "+1::skin-tone-3" — classify on the base name.
_POSITIVE_REACTIONS = {"+1", "thumbsup"}
_NEGATIVE_REACTIONS = {"-1", "thumbsdown"}

# Rate limiter for the unindexed-thumb INFO line (see handle_reaction).
_last_unindexed_log_ts = 0.0


def classify_reaction(name: str | None) -> str | None:
    base = str(name or "").split("::", 1)[0].strip().lower()
    if base in _POSITIVE_REACTIONS:
        return "positive"
    if base in _NEGATIVE_REACTIONS:
        return "negative"
    return None


def _reply_key(channel: str | None, ts: str | None) -> str:
    return f"{channel or ''}:{ts or ''}"


def _load_reply_index() -> None:
    if not REPLY_INDEX_PATH:
        return
    try:
        with open(REPLY_INDEX_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(k, str) and isinstance(v, dict):
                    _REPLY_INDEX[k] = v
            while len(_REPLY_INDEX) > _REPLY_INDEX_MAX:
                _REPLY_INDEX.popitem(last=False)
    except FileNotFoundError:
        pass
    except Exception as e:  # noqa: BLE001 — a corrupt index must not keep the bot down
        logger.warning("reply index unreadable (%s) — starting empty", e)


def _save_reply_index() -> None:
    if not REPLY_INDEX_PATH:
        return
    try:
        # Atomic replace, not truncate-in-place: the documented deploy is
        # `launchctl kickstart -k` (a signal mid-flight) and KeepAlive
        # respawns on crash — a torn write would corrupt the file and
        # _load_reply_index would silently start EMPTY, defeating the
        # restart-survival this persistence exists for (R1 review).
        tmp = f"{REPLY_INDEX_PATH}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(dict(_REPLY_INDEX), f)
        os.replace(tmp, REPLY_INDEX_PATH)
    except Exception as e:  # noqa: BLE001 — persistence is best-effort
        logger.warning("reply index save failed: %s", e)


def register_reply(channel: str | None, ts: str | None, req_id: str,
                   requester: str | None) -> None:
    """Remember a posted reply so a later reaction can be attributed."""
    if not (channel and ts and req_id):
        return
    _REPLY_INDEX[_reply_key(channel, ts)] = {
        "req_id": req_id,
        "requester": requester or "",
        "t": time.time(),
    }
    while len(_REPLY_INDEX) > _REPLY_INDEX_MAX:
        _REPLY_INDEX.popitem(last=False)
    _save_reply_index()


# Entries older than the agent's trajectory-scan window always 404 server-
# side — expire them on lookup so the miss is honest instead of a doomed
# round-trip (matches core/feedback._SCAN_DAYS = 8).
_REPLY_INDEX_TTL_S = 8 * 86400.0


def lookup_reply(channel: str | None, ts: str | None) -> dict | None:
    entry = _REPLY_INDEX.get(_reply_key(channel, ts))
    if entry is None:
        return None
    try:
        if time.time() - float(entry.get("t") or 0) > _REPLY_INDEX_TTL_S:
            _REPLY_INDEX.pop(_reply_key(channel, ts), None)
            return None
    except (TypeError, ValueError):
        pass
    return entry


def build_feedback_payload(entry: dict, signal: str, reactor: str,
                           owner_id: str | None,
                           reaction_name: str = "") -> dict:
    """The /api/feedback body for a classified reaction. ``source``
    carries the authority CLASS (owner vs requester); the note is
    deliberately EMPTY (R1 review): a note becomes the trajectory's
    ``failure_reason`` on negatives, and "slack reaction :-1: by U…" fed
    to the reflection LLM as the REPORTED FAILURE REASON produces
    hallucinated diagnoses — the server-side default ("human negative
    feedback") is the honest corpus row. The reactor id lives in the
    bot's own log, not the training corpus. ``reaction_name`` is kept in
    the signature for the log line's benefit only."""
    role = "owner" if reactor and reactor == owner_id else "requester"
    return {
        "request_id": entry.get("req_id") or "",
        "signal": signal,
        "source": f"slack:{role}",
        "note": "",
    }


async def post_feedback(payload: dict) -> int | None:
    """POST a label to the agent. One retry after a short pause on 404 (a
    reaction landing within a second of the reply can race the trajectory
    flush) AND on 5xx (the agent restarting — the exact window this ship's
    deploy creates; reactions are never replayed, so a dropped label is
    gone for good). Returns the final HTTP status, or None on transport
    failure."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            for attempt in (1, 2):
                r = await client.post(
                    f"{GHOST_API_BASE}/api/feedback",
                    json=payload, headers=AUTH_HEADERS)
                if attempt == 1 and (r.status_code == 404
                                     or r.status_code == 429
                                     or r.status_code >= 500):
                    await asyncio.sleep(5.0)
                    continue
                return r.status_code
    except Exception as e:  # noqa: BLE001 — feedback must never crash the bot
        logger.warning("feedback POST failed: %s", e)
    return None


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
            # Same call carries the workspace/org ids — the open gate's
            # Slack-Connect check (is_authorized_message) reads them.
            for key in ("team_id", "enterprise_id"):
                val = auth_test.get(key)
                if val:
                    BOT_TEAM_IDS.add(val)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to get bot user ID: {e}")
    return BOT_USER_ID


# ---------------------------------------------------------------------------
# Thread context (owner-filtered)
# ---------------------------------------------------------------------------

async def build_thread_context(channel_id: str, thread_ts: str,
                               current_event_ts: str) -> list:
    """LLM message history for a thread — AUTHORIZED messages only.

    The filter is part of the authorization boundary, not a convenience.
    Owner-locked mode: owner + bot messages only — without that, anyone in
    a shared channel could seed a thread with prompt content (or file
    attachments) that the owner's next mention would forward to the agent
    as trusted history. Open-channel mode widens the grant to every normal
    human member (their text AND files) — channel members are users by
    operator decision; bot-authored messages and subtypes stay excluded."""
    bot_user_id = await get_bot_user_id()
    context_messages: list = []
    if bot_user_id is None:
        # Without our own id we can't tell our replies from foreign bots'
        # (both carry bot_id) — the thread would degrade into consecutive
        # user turns with the agent blind to what it already said (R2
        # review). Fall back to the caller's single-message path.
        logger.warning("bot user id unresolved — skipping thread context")
        return context_messages
    # The SURFACE decision applies HERE too (R2 review HIGH: the handler
    # gated the open grant by surface, but this builder re-authorized a
    # stranger's history under the bare flag — bypassing the gate at
    # exactly the boundary it protects).
    open_here = OPEN_CHANNEL and _is_open_surface(channel_id)

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

            # ONLY this bot's own messages become role=assistant. A foreign
            # bot's message used to ride the same branch — i.e. any Jira/
            # webhook/workflow app posting into the thread could put words
            # in the agent's own mouth as highest-trust history (R1 review;
            # open-channel mode makes shared channels the NORM). Foreign
            # bot-authored messages are dropped entirely, in both modes.
            if bot_user_id and msg.get("user") == bot_user_id:
                context_messages.append({"role": "assistant", "content": text})
                continue
            if msg.get("bot_id"):
                continue

            # Authorization filter. Owner-locked (or a private surface):
            # anyone else's message in the thread is dropped entirely —
            # content AND files. Open mode on a channel surface: any
            # same-team human member's message is context.
            if not is_authorized_message(msg, OWNER_ID, open_here):
                continue

            file_notes = []
            if not is_current:  # the current event's files upload separately
                for f in msg.get("files", []):
                    filename = await upload_file_to_agent(f)
                    if filename:
                        file_notes.append(_file_note(filename))
                    else:
                        # Mirror _process_message's honesty note (R3): an
                        # oversized attachment announced on turn N must not
                        # silently vanish from turn N+1's rebuilt context.
                        _nm = os.path.basename(
                            str(f.get("name") or "attachment"))
                        file_notes.append(
                            f"[SYSTEM NOTE: The user attached a file named "
                            f"'{_nm}' but it could NOT be ingested (too "
                            f"large or the fetch failed).]")
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


# Slack file ids recently ingested → (sandbox filename, upload ts).
# build_thread_context rebuilds the WHOLE thread history on every message,
# so without this each subsequent turn re-uploaded every earlier attachment
# — re-POSTing the original over the agent's copy and silently CLOBBERING
# any edits the agent had made to it in the meantime ("fix this file" flows
# lost their fix on the owner's next message). A cache hit keeps the
# [SYSTEM NOTE] in the rebuilt context but skips the network round-trips.
# TTL-bounded (not process-lifetime): the upload lands in the ACTIVE
# project's scope and sandbox sweeps can delete it, so an old hit could
# assert a file that no longer exists where the note claims — a few hours
# kills the every-message clobber while re-asserting reality daily-ish.
# In-memory only: a bot restart re-uploads once (pre-existing behaviour).
_UPLOADED_FILE_IDS: dict = {}
_UPLOADED_FILE_IDS_MAX = 512
_UPLOAD_CACHE_TTL_S = 6 * 3600.0

# Attachment size cap (R1 review): the fetch buffers the WHOLE file in RAM
# and re-POSTs it whole — and open-channel mode means ANY member's thread
# attachment triggers it on the next mention. One multi-GB video = heap
# spike → OOM → launchd respawn → cache lost → repeat.
try:
    # OverflowError: GHOST_SLACK_MAX_UPLOAD_MB=inf parses as float('inf')
    # and int() of it raises AT IMPORT — the launchd crash-loop class.
    _MAX_UPLOAD_BYTES = int(float(
        os.environ.get("GHOST_SLACK_MAX_UPLOAD_MB", "50")) * 1024 * 1024)
except (TypeError, ValueError, OverflowError):
    _MAX_UPLOAD_BYTES = 50 * 1024 * 1024
if _MAX_UPLOAD_BYTES <= 0:
    # 0/negative reads as "no cap", not "block every attachment" (R2).
    _MAX_UPLOAD_BYTES = float("inf")


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
    size = file_info.get("size")
    if isinstance(size, int) and size > _MAX_UPLOAD_BYTES:
        logger.warning(
            "skipping attachment %s: %d bytes exceeds the %d-byte cap "
            "(GHOST_SLACK_MAX_UPLOAD_MB)", filename, size, _MAX_UPLOAD_BYTES)
        return None
    file_id = str(file_info.get("id") or "")
    if file_id and file_id in _UPLOADED_FILE_IDS:
        cached_name, cached_ts = _UPLOADED_FILE_IDS[file_id]
        if time.time() - cached_ts < _UPLOAD_CACHE_TTL_S:
            return cached_name
        _UPLOADED_FILE_IDS.pop(file_id, None)  # expired → re-upload
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
                if file_id:
                    while len(_UPLOADED_FILE_IDS) >= _UPLOADED_FILE_IDS_MAX:
                        _UPLOADED_FILE_IDS.pop(
                            next(iter(_UPLOADED_FILE_IDS)))
                    _UPLOADED_FILE_IDS[file_id] = (filename, time.time())
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
                           event_files: list | None = None,
                           requester: str | None = None):
    request_id = str(uuid.uuid4())[:8]
    log_task = asyncio.create_task(tail_logs(request_id, say, thread_ts))

    try:
        for file_info in (event_files or []):
            filename = await upload_file_to_agent(file_info)
            if filename:
                note = _file_note(filename)
            else:
                # A skipped/failed ingest must not be SILENT to the model
                # (R2 review): "analyze this" + a capped 60MB file used to
                # get a confident answer that ignored the file.
                raw_name = os.path.basename(
                    str(file_info.get("name") or "attachment"))
                note = (f"[SYSTEM NOTE: The user attached a file named "
                        f"'{raw_name}' but it could NOT be ingested (too "
                        f"large or the fetch failed). Say so rather than "
                        f"answering as if you had read it.]")
            # Placement (R3 review — the R2 "always a fresh turn" version
            # 422'd caption-less attachments via the empty fallback message
            # AND made the note the turn's recorded `user_request`):
            #   * empty trailing user message (caption-less attachment
            #     fallback) → the note BECOMES the request;
            #   * non-empty trailing user message (the requester's own
            #     caption, the common case on both entry paths) → glue;
            #   * anything else (thread ends on an assistant reply) →
            #     fresh user turn.
            # Residual edge accepted: an empty-caption mention in a thread
            # whose last human message is another member's glues the note
            # there — rare, and preferable to poisoning `user_request`.
            if messages and messages[-1].get("role") == "user" \
                    and not str(messages[-1].get("content") or "").strip():
                messages[-1]["content"] = note
            elif messages and messages[-1].get("role") == "user":
                messages[-1]["content"] += "\n\n" + note
            else:
                messages.append({"role": "user", "content": note})

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

            # The authoritative correlation handle for feedback labels. The
            # agent echoes X-Request-ID as "chatcmpl-<req_id>" but may have
            # UNIQUIFIED it on collision — trust the response, not the
            # locally generated id.
            agent_req_id = str(data.get("id") or "")
            if agent_req_id.startswith("chatcmpl-"):
                agent_req_id = agent_req_id[len("chatcmpl-"):]
            agent_req_id = agent_req_id or request_id

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

            posted = None
            if formatted_content:
                posted = await say(text=formatted_content,
                                   thread_ts=thread_ts)
            elif not uploaded:
                await say(text="Error: Agent returned an empty response.",
                          thread_ts=thread_ts)
            else:
                posted = await say(text="Here is your image:",
                                   thread_ts=thread_ts)
            # Remember the reply so a 👍/👎 reaction on it can become a
            # human outcome label. Error messages are deliberately NOT
            # registered — a thumbs-down on "Agent returned 500" would
            # label a turn the agent never completed.
            if posted is not None:
                try:
                    register_reply(
                        posted.get("channel") or getattr(say, "channel", None),
                        posted.get("ts"),
                        agent_req_id, requester)
                except Exception as e:  # noqa: BLE001 — bookkeeping only
                    logger.warning("reply registration failed: %s", e)

            for safe_name, blob in uploaded:
                try:
                    await app.client.files_upload_v2(
                        channel=say.channel, thread_ts=thread_ts,
                        file=blob, filename=safe_name, title=safe_name,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to upload {safe_name} to Slack: {e}")

    except Exception as e:  # noqa: BLE001
        # Generic text to the channel — open-channel mode means any member
        # reads this, and exception strings carry paths/URLs/internals
        # (R2 review). The detail goes to the log.
        logger.warning("processing failed for req %s: %s: %s",
                       request_id, type(e).__name__, e)
        await say(text="System error — the request could not be completed. "
                       "The operator can check the bot log for details.",
                  thread_ts=thread_ts)
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
    # 1:1 IMs are the MESSAGE handler's surface: Slack fires BOTH events
    # for "@Ghost …" typed in a DM, and processing here too ran the agent
    # twice per keystroke — two turns, two replies, two index entries
    # (R2 review). Group DMs / private G-channels stay here (the message
    # handler only takes channel_type == "im") under the owner-only gate.
    if str(event.get("channel") or "").startswith("D"):
        return
    # Owner always; any same-team human member in open-channel mode — but
    # ONLY on a channel surface. app_mention also fires for "@Ghost" inside
    # group DMs (with no channel_type field), and without the surface check
    # that mention bypassed the DM owner-lock entirely (R1 review CRIT).
    if not is_authorized_message(
            event, OWNER_ID,
            OPEN_CHANNEL and _is_open_surface(event.get("channel"))):
        _log_unauthorized("mention", event)
        return

    thread_ts = event.get("thread_ts") or event.get("ts")
    if MAINTENANCE_MODE:
        await say(text="The agent is currently down for maintenance.",
                  thread_ts=thread_ts)
        return

    messages = await build_thread_context(
        event.get("channel"), thread_ts, event.get("ts"))
    user_text = re.sub(r"<@.*?>", "", event.get("text", "")).strip()
    # Guard on "no USER content at all", not just an empty list (R4: a
    # rebuilt thread of assistant-only entries let a bare "@Ghost" ship a
    # request with zero user messages).
    if not any(m.get("role") == "user" for m in messages):
        if not user_text and not event.get("files"):
            # A bare "@Ghost" with nothing else — don't burn an agent turn
            # on an empty request (R2 review).
            await say(text="👋 Mention me with a question or a task.",
                      thread_ts=thread_ts)
            return
        messages.append({"role": "user", "content": user_text})

    await _process_message(messages, say, thread_ts, event.get("files"),
                           requester=event.get("user"))


@app.event("message")
async def handle_direct_message(event, say):
    # DMs only; the owner gate also drops bots and message subtypes
    # (edits / deletes / joins). DELIBERATELY owner-only in BOTH modes —
    # open-channel is a channel grant, not a workspace-wide one.
    if event.get("channel_type") != "im":
        return
    if not is_owner_message(event, OWNER_ID):
        _log_unauthorized("DM", event)
        return

    # Strip any literal @mention token — the thread-context builder strips
    # it, but the fallback path used to ship "<@U…> what's the weather"
    # verbatim whenever the context fetch failed (R3 review).
    user_text = re.sub(r"<@.*?>", "", event.get("text", "")).strip()
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

    await _process_message(messages, say, thread_ts, event.get("files"),
                           requester=event.get("user"))


@app.event("reaction_added")
async def handle_reaction(event, say=None):
    """👍/👎 on a bot reply → human outcome label via /api/feedback.

    Accepted only from the OWNER or the REQUESTER of that specific reply.
    Reactions on messages the bot didn't post (or whose index entry was
    evicted) are ignored silently. ``reaction_removed`` is deliberately
    not handled — a changed mind posts the opposite reaction and the
    sidecar's last-write-wins resolves it. Requires the ``reactions:read``
    scope + ``reaction_added`` bot event in the Slack app config."""
    try:
        item = event.get("item") or {}
        if item.get("type") != "message":
            return
        # A shadow ban that covered only MESSAGES would be cosmetic: this
        # path writes human outcome LABELS via /api/feedback, and those
        # feed learning. A banned user could still thumb their own earlier
        # replies and poison the training signal while appearing ignored.
        # Checked before `classify_reaction` so a banned user's reaction is
        # indistinguishable from any other unhandled emoji.
        if is_shadow_banned(event.get("user"), OWNER_ID):
            logger.debug("shadow-banned reaction user=%s channel=%s",
                         event.get("user"), item.get("channel"))
            return
        signal = classify_reaction(event.get("reaction"))
        if not signal:
            return
        entry = lookup_reply(item.get("channel"), item.get("ts"))
        if not entry:
            # A THUMB on a message we don't index is the case an operator
            # needs to see (missing reactions:read looks identical to a
            # wiped index otherwise — R1 review) — but in an open channel
            # colleagues thumb EACH OTHER constantly (R2 review), so it is
            # rate-limited to one INFO line per hour; the rest go to DEBUG.
            global _last_unindexed_log_ts
            now = time.time()
            if now - _last_unindexed_log_ts > 3600.0:
                _last_unindexed_log_ts = now
                logger.info(
                    "thumb reaction on unindexed message %s:%s — not one "
                    "of my replies, or the index was evicted/reset "
                    "(rate-limited; further misses log at DEBUG)",
                    item.get("channel"), item.get("ts"))
            else:
                logger.debug("thumb on unindexed message %s:%s",
                             item.get("channel"), item.get("ts"))
            return
        reactor = str(event.get("user") or "")
        if not reactor or (reactor != OWNER_ID
                           and reactor != entry.get("requester")):
            logger.info(
                "ignored %s reaction from non-party user=%s on req %s",
                signal, reactor, entry.get("req_id"))
            return
        payload = build_feedback_payload(
            entry, signal, reactor, OWNER_ID,
            reaction_name=str(event.get("reaction") or ""))
        status = await post_feedback(payload)
        if status is not None and 200 <= status < 300:
            logger.info("feedback %s for req %s by %s (:%s:) → HTTP %s",
                        signal, entry.get("req_id"), reactor,
                        event.get("reaction"), status)
        else:
            # WARNING reaches the launchd .err — a label that failed to
            # land must not hide at INFO in the sidecar log (R1 review:
            # "the feature can be 100% dead and every surface says
            # healthy").
            logger.warning(
                "feedback %s for req %s by %s FAILED → HTTP %s (label "
                "dropped; reactions are not replayed)",
                signal, entry.get("req_id"), reactor, status)
    except Exception as e:  # noqa: BLE001 — reactions must never crash the bot
        logger.warning(f"reaction handler failed: {e}")


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
try:
    NOTIFY_POLL_SECONDS = float(
        os.environ.get("GHOST_NOTIFY_POLL_SECONDS", "30"))
except (TypeError, ValueError):
    # A .env typo ("30s") used to raise at IMPORT — before any handler
    # existed — so launchd KeepAlive respawned the bot forever at the
    # throttle interval with nothing but a traceback in .err (the exact
    # crash-loop class the 2026-08-13 missing-.env incident produced).
    logger.warning("GHOST_NOTIFY_POLL_SECONDS is not a number — using 30")
    NOTIFY_POLL_SECONDS = 30.0
# Floor it: 0/negative parses fine but asyncio.sleep(0) is a tight loop
# hammering the agent and burying the heartbeat (R2 review).
NOTIFY_POLL_SECONDS = max(5.0, NOTIFY_POLL_SECONDS)
_NOTIFY_CONSUMER = "slack"
_poller_task = None  # strong ref — a bare create_task result can be GC'd

_PHASE_EMOJI = {
    "project": ":raising_hand:",
    "scheduled_task": ":alarm_clock:",
    "agent_message": ":speech_balloon:",
    "service": ":gear:",
    "job": ":package:",
}

# Human labels for the Slack line — mirrors the agent's digest labels
# (core/autonomous_activity._PHASE_LABELS). Raw slugs like
# `scheduled_task` read as debug output on a phone.
_PHASE_LABELS = {
    "project": "project",
    "scheduled_task": "scheduled task",
    "agent_message": "agent",
    "service": "service",
    "job": "background job",
    "open_questions": "open questions",
}

# Records older than this get an age suffix: after downtime/a wedge the
# poller re-serves the backlog, and without the age an hours-old event
# reads as breaking news (the 2026-07-13 unwedge delivered 2-day-old
# "needs your input" items as if current).
_STALE_AFTER_S = 300.0


def _age_suffix(ts: float, now: float | None = None) -> str:
    try:
        delta = (now if now is not None else time.time()) - float(ts or 0)
    except (TypeError, ValueError):
        return ""
    if not ts or delta < _STALE_AFTER_S:
        return ""
    if delta < 5400:
        return f"  _({int(delta // 60)}m ago)_"
    if delta < 129600:
        return f"  _({delta / 3600:.1f}h ago)_"
    return f"  _({delta / 86400:.1f}d ago)_"


def format_notification(rec: dict, now: float | None = None) -> str:
    phase = str(rec.get("phase", "") or "event")
    summary = str(rec.get("summary", "") or "")
    icon = _PHASE_EMOJI.get(phase, ":satellite_antenna:")
    label = _PHASE_LABELS.get(phase, phase)
    return f"{icon} *[{label}]* {summary}{_age_suffix(rec.get('ts'), now)}"


# Last watermark this process successfully acked. The server's watermark on
# an IDLE ledger is the same number every poll — re-acking it was a no-op
# POST plus a notify_consumers.json rewrite every 30s, around the clock
# (and kept that file's mtime permanently fresh, killing its documented
# value as a staleness diagnostic). The ack is skipped ONLY when the
# watermark equals what we already acked; an empty response whose watermark
# ADVANCED is still acked — that exact case is the 2026-07-13 wedge and its
# regression test.
_last_acked_watermark = None


async def poll_and_deliver_once(client, channel: str, poster=None) -> int:
    """One poll → deliver → ack cycle. Returns the number of records
    delivered. Factored out of the loop so the ack contract is unit-
    testable (``poster`` defaults to the live Slack client's
    chat_postMessage; tests inject a fake).
    """
    global _last_acked_watermark
    poster = poster or app.client.chat_postMessage
    r = await client.get(
        f"{GHOST_API_BASE}/api/notifications/pending",
        params={"consumer": _NOTIFY_CONSUMER, "limit": 20},
        headers=AUTH_HEADERS,
    )
    if r.status_code != 200:
        raise RuntimeError(f"pending poll HTTP {r.status_code}")
    data = r.json()
    if data.get("enabled") is False:
        # No activity ledger on the agent (boot/config gap). Its watermark
        # is a literal 0 — acking that would OVERWRITE the stored consumer
        # offset with 0, and when the ledger comes back the first-contact
        # baseline is bypassed and the ENTIRE notify history replays.
        return 0
    records = data.get("records") or []
    watermark = data.get("watermark")
    if records:
        text = "\n".join(format_notification(rec) for rec in records)
        await poster(channel=channel, text=text)
        logger.info(f"delivered {len(records)} notification(s) → {channel}")
    # Ack every response whose watermark MOVED (or that delivered records),
    # after delivery. The 2026-07-13 contract stands: an empty response can
    # still carry an ADVANCED watermark (the scan window was all non-notify
    # lines) and skipping that ack wedged the consumer for two days. What's
    # new is only the idle-identity skip — re-acking the same number was
    # pure churn. Ordering is preserved: a crash between post and ack
    # re-serves rather than drops. NOTE the skip compares against the last
    # SUCCESSFUL (2xx) ack — recording a failed ack as done would convert
    # one agent-side 500 into a permanently suppressed retry.
    if watermark is not None and (records
                                  or watermark != _last_acked_watermark):
        ack = await client.post(
            f"{GHOST_API_BASE}/api/notifications/ack",
            json={"consumer": _NOTIFY_CONSUMER, "watermark": watermark},
            headers=AUTH_HEADERS,
        )
        if 200 <= ack.status_code < 300:
            _last_acked_watermark = watermark
        else:
            logger.warning(f"ack HTTP {ack.status_code} — will retry "
                           f"watermark {watermark} next poll")
    return len(records)


# Hourly heartbeat: one INFO line summarizing the poller's last hour. With
# httpx request-logging silenced, a QUIET log needs a positive liveness
# signal — the 2026-07-13 wedge was diagnosed by the ABSENCE of `delivered`
# lines among poll noise; now the heartbeat states polls/delivered/errors
# outright, so a wedged or dead poller is visible in two lines of log.
_HEARTBEAT_EVERY_S = 3600.0


class _PollerHeartbeat:
    def __init__(self, every_s: float = _HEARTBEAT_EVERY_S):
        self.every_s = float(every_s)
        self.polls = 0
        self.delivered = 0
        self.errors = 0
        self._last_beat = time.time()

    def note(self, delivered: int = 0, error: bool = False,
             now: float | None = None) -> str | None:
        """Record one cycle; returns the heartbeat line when due."""
        now = now if now is not None else time.time()
        self.polls += 1
        self.delivered += max(0, int(delivered))
        if error:
            self.errors += 1
        if now - self._last_beat < self.every_s:
            return None
        line = (f"poller heartbeat: {self.polls} poll(s), "
                f"{self.delivered} delivered, {self.errors} error(s) "
                f"in the last {(now - self._last_beat) / 60:.0f}m")
        self.polls = self.delivered = self.errors = 0
        self._last_beat = now
        return line


async def notification_poller(channel: str):
    logger.info(
        f"Notification poller ON → {channel} every {NOTIFY_POLL_SECONDS:.0f}s")
    hb = _PollerHeartbeat()
    client = None
    while True:
        try:
            if client is None:
                # One persistent client (connection pooling) instead of a
                # new TCP+client per 30s poll; rebuilt after any error.
                client = httpx.AsyncClient(timeout=15.0)
            n = await poll_and_deliver_once(client, channel)
            beat = hb.note(n)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"notification poller: {e}")
            beat = hb.note(0, error=True)
            if client is not None:
                try:
                    await client.aclose()
                except Exception:  # noqa: BLE001
                    pass
                client = None
        if beat:
            logger.info(beat)
        await asyncio.sleep(NOTIFY_POLL_SECONDS)


async def main():
    global LOG_FILE_PATH, MAINTENANCE_MODE, OWNER_ID, OPEN_CHANNEL, \
        _poller_task
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
    if OPEN_CHANNEL:
        logger.info(
            "OPEN-CHANNEL mode ACTIVE (GHOST_SLACK_OPEN_CHANNEL=1) — "
            "channel mentions from ANY human member are answered and their "
            "thread context is forwarded; DMs remain owner-only (%s); "
            "feedback reactions accepted from the owner or the requester",
            OWNER_ID)
    else:
        logger.info(f"Owner lock ACTIVE — replying only to {OWNER_ID}")

    _load_reply_index()
    if _REPLY_INDEX:
        logger.info(f"reply index loaded: {len(_REPLY_INDEX)} entries")

    # Resolve bot user + WORKSPACE ids up front: the open gate's
    # Slack-Connect check depends on BOT_TEAM_IDS, and resolving it lazily
    # would leave the first events of every boot unchecked. FAIL-CLOSED
    # (R2 review): boots on this box demonstrably hit transient auth
    # windows; running open-channel with the team check blind reopens the
    # exact hole it closes. Retry, then fall back to the owner lock for
    # the run rather than exit — notifications and owner chat stay up.
    for attempt in range(3):
        await get_bot_user_id()
        if BOT_USER_ID:
            break
        if attempt < 2:   # no pointless sleep after the final attempt
            await asyncio.sleep(5.0)
    if OPEN_CHANNEL and not BOT_TEAM_IDS:
        OPEN_CHANNEL = False
        logger.warning(
            "auth.test never resolved the workspace id — OPEN-CHANNEL mode "
            "DISABLED for this run (owner lock enforced); restart to retry")

    notify_dest = NOTIFY_CHANNEL or OWNER_ID
    if notify_dest.lower() in ("off", "none", "0", "false"):
        logger.info("Notification poller OFF (GHOST_NOTIFY_SLACK_CHANNEL=off)")
    else:
        _poller_task = asyncio.create_task(notification_poller(notify_dest))

    handler = AsyncSocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    await handler.start_async()


if __name__ == "__main__":
    asyncio.run(main())
