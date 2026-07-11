"""Durable server-side conversations (2026-07-11).

Before this, conversation history was entirely CLIENT-carried: the web UI's
``localStorage``, a Slack thread, or a manual ``workspace/save`` zip. Switch
device or clear the cache and the conversation was gone — and each client
held a *different* fragment of history, so a conversation started in Slack
could not be continued in the web UI. Ironic, given cross-session MEMORY is
this agent's proven strength (Track B: 98% recall vs 0% control): the memory
substrate persisted while the conversations themselves did not.

A session is one JSON file under ``$GHOST_HOME/system/sessions/``:
``{id, title, created_at, updated_at, messages: [...]}``. Sessions live
entirely in the API layer — the route merges the stored history into the
request before ``handle_chat`` and appends the turn afterwards — so the
agent's turn logic is untouched by this feature.

Fail-safe by contract: every method degrades (returns None / [] / False)
rather than raising into a request.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("GhostAgent")

MAX_SESSIONS = 200          # oldest-updated evicted past this
MAX_MESSAGES_PER_SESSION = 400
MAX_TITLE_CHARS = 80
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_ROLES = ("system", "user", "assistant", "tool", "function")


def _now() -> float:
    return time.time()


@dataclass
class Session:
    id: str
    title: str = ""
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    messages: List[dict] = field(default_factory=list)

    def summary(self) -> dict:
        """List-view shape — no message bodies (a session list must stay
        cheap even with 200 sessions of 400 messages)."""
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": len(self.messages),
        }

    def to_dict(self) -> dict:
        d = self.summary()
        d["messages"] = self.messages
        return d


def _clean_messages(messages) -> List[dict]:
    """Keep only well-shaped chat messages. A malformed entry is dropped
    rather than persisted — a corrupt session file would otherwise poison
    every future turn that loads it."""
    out: List[dict] = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role not in _ROLES:
            continue
        content = m.get("content")
        if not isinstance(content, (str, list)) and content is not None:
            continue
        keep = {"role": role, "content": content}
        # Preserve tool-call plumbing so a resumed conversation's tool
        # pairing stays intact.
        for k in ("tool_calls", "tool_call_id", "name"):
            if k in m:
                keep[k] = m[k]
        out.append(keep)
    return out


def derive_title(messages) -> str:
    """First user message, collapsed — the session's human label."""
    for m in messages or []:
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):  # multimodal: take the text parts
                c = " ".join(str(p.get("text", "")) for p in c
                             if isinstance(p, dict))
            text = " ".join(str(c or "").split())
            if text:
                return text[:MAX_TITLE_CHARS]
    return "New conversation"


def merge_history(stored: List[dict], incoming: List[dict]) -> List[dict]:
    """Build the message list for a turn from the STORED history and what
    the client sent, tolerating both client styles:

    * a *thin* client sends only the new message(s) → ``stored + incoming``;
    * a *fat* client (the current web UI) replays the whole conversation →
      ``incoming`` already contains the stored history as its prefix, so it
      is used as-is instead of being duplicated.

    Detection compares (role, content) pairs over the stored prefix — this
    is what stops a fat client from doubling the conversation on every turn.
    """
    stored = stored or []
    incoming = incoming or []
    if not stored:
        return list(incoming)
    if len(incoming) >= len(stored):
        def _key(m):
            return (m.get("role"), str(m.get("content") or ""))
        if all(_key(a) == _key(b)
               for a, b in zip(stored, incoming[:len(stored)])):
            return list(incoming)   # fat client — already carries the history
    return list(stored) + list(incoming)


class SessionStore:
    """One JSON file per session. Thread-safe; never raises."""

    def __init__(self, root):
        self.root = Path(root)
        self._lock = threading.Lock()

    # -- paths --------------------------------------------------------------

    def _path(self, session_id: str) -> Optional[Path]:
        sid = str(session_id or "")
        if not _ID_RE.match(sid):
            return None          # path-traversal guard: ids are opaque tokens
        return self.root / f"{sid}.json"

    # -- CRUD ---------------------------------------------------------------

    def create(self, title: str = "") -> Optional[Session]:
        sess = Session(id=uuid.uuid4().hex[:16],
                       title=" ".join(str(title or "").split())[:MAX_TITLE_CHARS])
        return sess if self._write(sess) else None

    def get(self, session_id: str) -> Optional[Session]:
        path = self._path(session_id)
        if path is None or not path.exists():
            return None
        try:
            d = json.loads(path.read_text())
            return Session(
                id=str(d.get("id") or session_id),
                title=str(d.get("title") or ""),
                created_at=float(d.get("created_at") or 0.0),
                updated_at=float(d.get("updated_at") or 0.0),
                messages=_clean_messages(d.get("messages")),
            )
        except Exception as e:  # noqa: BLE001 — corrupt file → treated absent
            logger.debug("session %s unreadable: %s", session_id, e)
            return None

    def _write(self, sess: Session) -> bool:
        try:
            with self._lock:
                self.root.mkdir(parents=True, exist_ok=True)
                path = self._path(sess.id)
                if path is None:
                    return False
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(sess.to_dict(), ensure_ascii=False))
                os.replace(tmp, path)
            return True
        except Exception as e:  # noqa: BLE001
            logger.debug("session write failed: %s", e)
            return False

    def delete(self, session_id: str) -> bool:
        path = self._path(session_id)
        if path is None or not path.exists():
            return False
        try:
            path.unlink()
            return True
        except OSError as e:
            logger.debug("session delete failed: %s", e)
            return False

    def list(self, limit: int = 50) -> List[dict]:
        """Session summaries, most recently updated first."""
        out = []
        try:
            for p in self.root.glob("*.json"):
                try:
                    d = json.loads(p.read_text())
                    out.append({
                        "id": str(d.get("id") or p.stem),
                        "title": str(d.get("title") or ""),
                        "created_at": float(d.get("created_at") or 0.0),
                        "updated_at": float(d.get("updated_at") or 0.0),
                        "message_count": len(d.get("messages") or []),
                    })
                except Exception:  # noqa: BLE001 — skip a corrupt file
                    continue
        except Exception as e:  # noqa: BLE001 — absent dir → empty list
            logger.debug("session list failed: %s", e)
            return []
        out.sort(key=lambda s: s["updated_at"], reverse=True)
        try:
            limit = max(1, min(int(limit), MAX_SESSIONS))
        except (TypeError, ValueError):
            limit = 50
        return out[:limit]

    # -- turn append --------------------------------------------------------

    def append_turn(self, session_id: str, user_messages: List[dict],
                    assistant_content: str) -> bool:
        """Append this turn's new user message(s) + the assistant reply.

        Called by the chat route AFTER the turn completes, so a failed turn
        never leaves a dangling user message with no reply. Creates the
        session if it doesn't exist yet (a client may pick its own id).
        """
        sess = self.get(session_id)
        if sess is None:
            path = self._path(session_id)
            if path is None:
                return False
            sess = Session(id=str(session_id))
        new_msgs = _clean_messages(user_messages)
        if assistant_content:
            new_msgs.append({"role": "assistant",
                             "content": str(assistant_content)})
        if not new_msgs:
            return False
        sess.messages.extend(new_msgs)
        # Bound the stored history: keep leading system messages + the most
        # recent tail (mirrors the agent's own 500-message hard cap).
        if len(sess.messages) > MAX_MESSAGES_PER_SESSION:
            systems = [m for m in sess.messages if m.get("role") == "system"]
            rest = [m for m in sess.messages if m.get("role") != "system"]
            sess.messages = systems + rest[-(MAX_MESSAGES_PER_SESSION
                                             - len(systems)):]
        if not sess.title:
            sess.title = derive_title(sess.messages)
        sess.updated_at = _now()
        ok = self._write(sess)
        self._evict()
        return ok

    def _evict(self) -> None:
        """Drop the least-recently-updated sessions past ``MAX_SESSIONS``."""
        try:
            sessions = self.list(limit=MAX_SESSIONS + 50)
            for s in sessions[MAX_SESSIONS:]:
                self.delete(s["id"])
        except Exception as e:  # noqa: BLE001
            logger.debug("session eviction skipped: %s", e)


def get_session_store(context) -> Optional[SessionStore]:
    """The context-attached store (wired in main.py), or None."""
    return getattr(context, "session_store", None)


__all__ = [
    "MAX_SESSIONS", "MAX_MESSAGES_PER_SESSION",
    "Session", "SessionStore",
    "derive_title", "merge_history", "get_session_store",
]
