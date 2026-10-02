"""The handheld's calls to the agent beyond ``/api/chat`` itself. Qt-free.

Sessions, turn cancel, human feedback and pending notifications — the four
things the web UI could do and this client could not. Every function takes
the HTTP client as an argument (an ``httpx.AsyncClient`` in production, a
double in tests), so the contracts are executed off the device:

* **sessions** — the conversation survives a restart, and one started in the
  browser can be opened here (``/sessions``, ``/open N``).
* **cancel** — "stop" must stop the AGENT's turn, not just this client's
  listening; and it must stop OUR turn, never whichever one holds the lock.
* **feedback** — a thumb on the last reply. Human labels are the scarcest
  signal the agent's learning has, and this device produced none.
* **notifications** — what the agent did while nobody was asking, shown on
  the device that is always on. The ack contract is the Slack bot's, learned
  the hard way there (see :class:`NotifyPoller`).
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
import time
import uuid

SESSION_PATH = os.environ.get("GHOST_SESSION_STATE", "~/.ghost_session_id")
# The agent's own guard (core/sessions.py `_ID_RE`): an id outside it is
# accepted by /api/chat and then silently NOT persisted.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}\Z")
FEEDBACK_SOURCE = "clockwork"
NOTIFY_CONSUMER = "clockwork"


def headers(key: str, request_id: str = "") -> dict:
    h = {"X-Ghost-Key": key}
    if request_id:
        h["X-Request-ID"] = request_id
    return h


# ── errors, in words ────────────────────────────────────────────────────────
# The transcript used to show `fault → ConnectError: All connection attempts
# failed` and `fault → HTTP 503`. Both are true and neither says what to do.
# Matched by class NAME so this module needs no httpx import (and so a double
# in a test is described the same way as the real thing).
_UNREACHABLE = ("ConnectError", "ConnectTimeout", "ConnectionRefusedError",
                "gaierror", "NetworkError")
_TIMEOUTS = ("ReadTimeout", "WriteTimeout", "PoolTimeout", "TimeoutError")
_DROPPED = ("RemoteProtocolError", "ReadError", "WriteError",
            "ConnectionResetError", "IncompleteRead")


def is_unreachable(exc) -> bool:
    return type(exc).__name__ in _UNREACHABLE


def describe_error(exc, base: str = "") -> str:
    """An exception from a call to ``base`` as one line the operator can use."""
    name = type(exc).__name__
    where = base.split("://", 1)[-1] if base else "the agent"
    if name in _UNREACHABLE:
        return f"cannot reach {where} — is it up, and is this device on the network?"
    if name in _TIMEOUTS:
        return f"{where} did not answer in time"
    if name in _DROPPED:
        return f"{where} dropped the connection (restarting?)"
    detail = " ".join(str(exc).split())[:200]
    return f"{name}: {detail}" if detail else name


def describe_http_error(status: int, body: str = "") -> str:
    """A non-200 answer: the server's own message when it sent one."""
    detail = ""
    try:
        import json as _json
        data = _json.loads(body or "")
        if isinstance(data, dict):
            err = data.get("error", data.get("detail"))
            if isinstance(err, dict):
                err = err.get("message")
            detail = str(err or "").strip()
    except ValueError:
        detail = ""
    if not detail:
        detail = body or ""
    # One line, bounded: a 422's `detail` is a LIST of validation errors, and
    # its repr ran to three thousand characters in the transcript.
    detail = " ".join(detail.split())[:200]
    hint = {401: "the API key was refused (~/.ghost_api_key)",
            403: "the API key was refused (~/.ghost_api_key)",
            503: "the agent is starting up or busy"}.get(int(status), "")
    parts = [f"HTTP {status}"] + [p for p in (detail, hint) if p]
    return " — ".join(parts)


# ── session id ──────────────────────────────────────────────────────────────

def new_session_id() -> str:
    return "cw-" + uuid.uuid4().hex[:16]


def valid_session_id(sid) -> bool:
    return isinstance(sid, str) and bool(_SESSION_ID_RE.match(sid))


def load_session_id(path=None):
    """The remembered session id, or None. Never raises."""
    try:
        with open(os.path.expanduser(path or SESSION_PATH)) as fh:
            sid = fh.read().strip()
    except OSError:
        return None
    return sid if valid_session_id(sid) else None


def save_session_id(sid: str, path=None) -> bool:
    """Remember ``sid``. Atomic, like facestate: a handheld loses power
    mid-write as a matter of course."""
    if not valid_session_id(sid):
        return False
    target = os.path.expanduser(path or SESSION_PATH)
    try:
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(target) or ".",
                                   prefix=".session-")
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(sid + "\n")
            os.replace(tmp, target)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except OSError:
        return False
    return True


async def fetch_session(client, base: str, key: str, sid: str):
    """The stored conversation: ``(messages, status)``.

    ``status`` is ``"ok"``, ``"missing"`` (a new id — nothing stored yet, which
    is normal), ``"disabled"`` (the agent runs without a session store, so the
    client must keep carrying history itself) or ``"error"``.
    """
    try:
        r = await client.get(f"{base}/api/sessions/{sid}", headers=headers(key))
    except Exception:  # noqa: BLE001 — unreachable agent: not this call's to raise
        return [], "error"
    if r.status_code == 404:
        return [], "missing"
    if r.status_code == 503:
        return [], "disabled"
    if r.status_code != 200:
        return [], "error"
    try:
        msgs = r.json().get("messages")
    except Exception:  # noqa: BLE001
        return [], "error"
    return (msgs if isinstance(msgs, list) else []), "ok"


async def list_sessions(client, base: str, key: str, limit: int = 8):
    """Recent conversations, newest first: ``[{id, title, updated_at,
    message_count}]``. Empty on any failure."""
    try:
        r = await client.get(f"{base}/api/sessions", params={"limit": limit},
                             headers=headers(key))
        data = r.json() if r.status_code == 200 else {}
    except Exception:  # noqa: BLE001
        return []
    rows = data.get("sessions") if isinstance(data, dict) else None
    return [s for s in (rows or []) if isinstance(s, dict) and s.get("id")][:limit]


def history_for_model(messages) -> list:
    """A stored conversation as the history this client replays.

    Role and content only, and the content OBJECT as stored: the agent aligns
    a replayed history against its own by ``(role, str(content))``, so a
    message this client re-serialised differently would be appended again.
    """
    # A message with no content (a stored assistant turn that was only tool
    # calls) is dropped: stripped of its tool plumbing it would reach the
    # model as an assistant message saying nothing.
    return [{"role": m["role"], "content": m["content"]}
            for m in (messages or [])
            if isinstance(m, dict) and m.get("role") in ("user", "assistant")
            and isinstance(m.get("content"), (str, list)) and m["content"]]


def age_text(ts, now=None) -> str:
    try:
        d = max(0, int((time.time() if now is None else now) - float(ts)))
    except (TypeError, ValueError):
        return ""
    if d < 90:
        return "just now"
    if d < 5400:
        return f"{d // 60} min ago"
    if d < 129600:
        return f"{d // 3600} h ago"
    return f"{d // 86400} d ago"


# ── streamed frames ─────────────────────────────────────────────────────────

def new_request_id() -> str:
    """This turn's id, minted HERE so the turn can be cancelled before the
    agent has sent a single frame — which is the whole thinking phase, i.e.
    exactly when stop gets pressed. Same shape as the agent's own ids."""
    return uuid.uuid4().hex[:8]


def frame_request_id(data):
    """The agent's request id from a stream frame (``chatcmpl-<id>``), bare.

    Read from the frames rather than trusted from what was sent: the agent
    may uniquify a colliding id, and feedback must name the id the trajectory
    was actually filed under.
    """
    rid = data.get("id") if isinstance(data, dict) else None
    if not isinstance(rid, str) or not rid.startswith("chatcmpl-"):
        return None
    return rid[len("chatcmpl-"):] or None


def frame_unlabelable(data) -> bool:
    """``ghost.labelable: false`` — the trivial fast path answered and wrote
    no trajectory, so a thumb could never land."""
    ghost = data.get("ghost") if isinstance(data, dict) else None
    return isinstance(ghost, dict) and ghost.get("labelable") is False


def frame_content(data) -> str:
    """The text a stream frame carries, or ``""`` — for ANY frame shape.

    ⚠ The inline version was ``data["choices"][0].get("delta", {})``, and a
    frame whose ``choices`` is an empty list raised IndexError out of the
    stream: the reply ended in "fault → IndexError", its last sentence was
    never spoken, and a held reply was lost whole. The agent asks its model
    server for usage, and the usage frame is exactly that shape
    (``{"choices": [], "usage": {…}}``).
    """
    if not isinstance(data, dict):
        return ""
    message = data.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str) \
            and message["content"]:
        return message["content"]
    choices = data.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else None
    delta = first.get("delta") if isinstance(first, dict) else None
    content = delta.get("content") if isinstance(delta, dict) else None
    return content if isinstance(content, str) else ""


def frame_error(data):
    """The message of an ERROR frame (one with no ``choices``), else None."""
    if not isinstance(data, dict) or not data.get("error") or "choices" in data:
        return None
    err = data["error"]
    return str(err.get("message") if isinstance(err, dict) else err)


# ── turns: is it ours, and stop it ──────────────────────────────────────────

def _turns(payload) -> list:
    rows = payload.get("turns") if isinstance(payload, dict) else None
    return [t for t in (rows or []) if isinstance(t, dict)]


def background_busy(payload, session_id, own_request_id=None) -> bool:
    """A turn that is NOT ours holds the lock (a dream, self-play, another
    client) — the face shows a second, slower breath for it."""
    for t in _turns(payload):
        if not t.get("running"):
            continue
        if own_request_id and t.get("request_id") == own_request_id:
            continue
        if session_id and t.get("session_id") == session_id:
            continue
        return True
    return False


async def fetch_turns(client, base: str, key: str):
    """``/api/turns`` payload, or None when the agent did not answer — which
    is also this client's reachability probe."""
    try:
        r = await client.get(f"{base}/api/turns", headers=headers(key))
    except Exception:  # noqa: BLE001
        return None
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except Exception:  # noqa: BLE001
        return None
    return data if isinstance(data, dict) else None


CANCEL_RETRY_S = 1.0


async def cancel_turn(client, base: str, key: str, request_id,
                      hard: bool = False, sleep=asyncio.sleep):
    """Stop this client's turn on the AGENT. Returns ``(outcome, detail)``:

    * ``"cancelled"`` — the agent took it;
    * ``"finished"``  — the agent has no such turn: it has already ended;
    * ``"unknown"``   — there is no id to name, so NOTHING was cancelled;
    * ``"failed"``    — the agent refused or did not answer.

    ⚠ Ownership is the REQUEST ID and nothing else. The first version of this
    fell back, on a 404, to "the turn in my session" and then "the turn whose
    text starts like mine" — and a reviewer cancelled a browser turn with it
    (a shared session after ``/open``; a preview that merely began with the
    same word). This client mints its own id for every turn and reads the id
    the agent actually used off the first frame, so a 404 has exactly two
    meanings, neither of which is "look for another turn":

      * the turn is over — the common case; or
      * the turn is not registered YET (stop pressed in the first moments).
        One retry a second later tells the two apart, so a turn that then
        registers and runs is not reported as stopped.
    """
    if not request_id:
        return "unknown", ""
    body = {"request_id": request_id}
    if hard:
        body["hard"] = True
    try:
        for attempt in (0, 1):
            r = await client.post(f"{base}/api/turn/cancel", json=body,
                                  headers=headers(key))
            try:
                data = r.json()
            except Exception:  # noqa: BLE001
                data = {}
            if r.status_code == 200 and isinstance(data, dict) and data.get("cancelled"):
                return "cancelled", request_id
            if r.status_code != 404:
                return "failed", f"HTTP {r.status_code}"
            if attempt == 0:
                await sleep(CANCEL_RETRY_S)
        return "finished", ""
    except Exception as e:  # noqa: BLE001
        return "failed", f"{type(e).__name__}: {e}"


# ── feedback ────────────────────────────────────────────────────────────────

FEEDBACK_RETRY_S = 4.0


async def send_feedback(client, base: str, key: str, request_id: str,
                        signal: str, note: str = "", sleep=asyncio.sleep):
    """Label a finished turn ``positive`` / ``negative``. ``(ok, detail)``.

    One delayed retry on 404/429/5xx, exactly as the web UI does: the
    trajectory is written moments AFTER the stream ends, so a thumb pressed
    the instant the reply lands gets a 404 that a second attempt four seconds
    later does not; and a 503 is the agent restarting.
    """
    if signal not in ("positive", "negative"):
        return False, "bad signal"
    if not request_id:
        return False, "no reply to rate"
    body = {"request_id": request_id, "signal": signal,
            "source": FEEDBACK_SOURCE}
    if note:
        body["note"] = str(note)[:500]
    detail = ""
    for attempt in (0, 1):
        try:
            r = await client.post(f"{base}/api/feedback", json=body,
                                  headers=headers(key))
        except Exception as e:  # noqa: BLE001
            detail = f"{type(e).__name__}: {e}"
        else:
            if 200 <= r.status_code < 300:
                return True, ""
            try:
                detail = str(r.json().get("error") or "")
            except Exception:  # noqa: BLE001
                detail = ""
            detail = detail or f"HTTP {r.status_code}"
            if not (r.status_code in (404, 429) or r.status_code >= 500):
                return False, detail
        if attempt == 0:
            await sleep(FEEDBACK_RETRY_S)
    return False, detail


# ── notifications ───────────────────────────────────────────────────────────

# The Slack bot's labels (interface/externals/slack_bot), so one record reads
# the same on both surfaces.
_PHASE_LABELS = {
    "project": "project",
    "scheduled_task": "scheduled task",
    "agent_message": "agent",
    "service": "service",
    "job": "background job",
    "open_questions": "open questions",
    "gepa_autonomy": "GEPA autonomy",
}
# Past this a record carries its age: after downtime the backlog is re-served,
# and without the age an hours-old event reads as breaking news.
NOTIFY_STALE_S = 300.0


def format_notification(rec: dict, now=None) -> str:
    """One activity record as a single plain line:
    ``[scheduled task] backup finished · 2 h ago``."""
    phase = str(rec.get("phase") or "event")
    label = _PHASE_LABELS.get(phase, phase.replace("_", " "))
    summary = " ".join(str(rec.get("summary") or "").split())
    age = ""
    try:
        ts = float(rec.get("ts") or 0)
        if ts and (time.time() if now is None else now) - ts >= NOTIFY_STALE_S:
            age = age_text(ts, now)
    except (TypeError, ValueError):
        age = ""
    return f"[{label}] {summary}" + (f" · {age}" if age else "")


class NotifyPoller:
    """Poll → deliver → ack, with the ack rules the Slack bot paid for.

    * ``enabled: false`` is NEVER acked: its watermark is a literal 0, and
      acking it overwrites the stored offset — when the ledger comes back the
      first-contact baseline is bypassed and the whole history replays.
    * the ack follows delivery, so a crash between the two re-serves rather
      than drops;
    * a watermark that MOVED is acked even with no records (the scan window
      was all non-notify lines — skipping that ack once wedged a consumer for
      two days), while re-acking an unchanged watermark is skipped;
    * only a 2xx counts as acked — recording a failed ack as done turns one
      agent-side error into a permanently suppressed retry.
    """

    def __init__(self, base: str, key: str, consumer: str = NOTIFY_CONSUMER,
                 limit: int = 20):
        self.base, self.key = base, key
        self.consumer, self.limit = consumer, limit
        self.last_acked = None

    async def poll(self, client):
        """``(records, watermark)``; watermark is None when nothing may be
        acked (agent unreachable, ledger disabled, malformed reply)."""
        try:
            r = await client.get(
                f"{self.base}/api/notifications/pending",
                params={"consumer": self.consumer, "limit": self.limit},
                headers=headers(self.key))
        except Exception:  # noqa: BLE001
            return [], None
        if r.status_code != 200:
            return [], None
        try:
            data = r.json()
        except Exception:  # noqa: BLE001
            return [], None
        if not isinstance(data, dict) or data.get("enabled") is False:
            return [], None
        wm = data.get("watermark")
        if type(wm) is not int:
            # Nothing could be acked — and records shown without an ack are
            # served again on the next poll, and the one after. A reply that
            # cannot be acknowledged is malformed: deliver none of it.
            return [], None
        records = [x for x in (data.get("records") or []) if isinstance(x, dict)]
        return records, wm

    def needs_ack(self, records, watermark) -> bool:
        return watermark is not None and (bool(records)
                                          or watermark != self.last_acked)

    async def ack(self, client, watermark) -> bool:
        try:
            r = await client.post(
                f"{self.base}/api/notifications/ack",
                json={"consumer": self.consumer, "watermark": watermark},
                headers=headers(self.key))
        except Exception:  # noqa: BLE001
            return False
        if 200 <= r.status_code < 300:
            self.last_acked = watermark
            return True
        return False

    async def cycle(self, client, deliver) -> int:
        """One poll → ``deliver(records)`` → ack. Returns records delivered.
        If ``deliver`` raises, nothing is acked and the records come back."""
        records, watermark = await self.poll(client)
        if records:
            deliver(records)
        if self.needs_ack(records, watermark):
            await self.ack(client, watermark)
        return len(records)
