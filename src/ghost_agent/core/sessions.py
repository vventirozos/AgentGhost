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
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("GhostAgent")

MAX_SESSIONS = 200          # oldest-updated evicted past this
MAX_MESSAGES_PER_SESSION = 400
MAX_TITLE_CHARS = 80
# ⚠ `\Z`, not `$`: Python's `$` also matches immediately BEFORE a trailing
# newline, so `"web-x\n"` passed this guard while the interface's
# `re.fullmatch` rejected it — the store wrote `web-x\n.json`, which the UI
# could neither open nor delete (R2 lens C).
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}\Z")


def _warn_unpersistable(session_id) -> None:
    """An id the charset guard rejects means the session is NOT durable —
    every turn in it is dropped on the floor. That was silent on BOTH early
    returns, so a client minting a slightly-wrong id (the interface's proxy
    guard used to allow dots and 128 chars, this one does not) looked like it
    was persisting and never was (review R1 M10). The operator watches the
    live stream: data loss belongs there."""
    logger.warning(
        "session NOT persisted: id %r fails the id charset (%s) — this "
        "conversation is not durable", session_id, _ID_RE.pattern)
_ROLES = ("system", "user", "assistant", "tool", "function")

# Search stopwords for the hydration tier (mirrors core.bus._STOPWORDS —
# high-frequency chat words that would make every conversation "match").
_SEARCH_STOPWORDS = {
    "this", "that", "what", "with", "from", "your", "have", "they", "will",
    "would", "could", "should", "just", "about", "like", "when", "then",
    "there", "their", "them", "some", "make", "please", "know", "want",
    "need", "tell", "give", "show",
}


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


def _coerce_messages(seq) -> List[dict]:
    """Keep only dict-shaped messages. The module contract is "never raises
    into a request", and `merge_history_detail` broke it: a str/None entry
    (or a str passed where a list belongs) died on `.get` with an
    AttributeError. The only live caller validates shape first, so this was
    a latent fail-open rather than an outage — the kind that surfaces the
    day a second caller appears (R2 lens C)."""
    if isinstance(seq, dict) or isinstance(seq, str):
        return []
    try:
        return [m for m in (seq or []) if isinstance(m, dict)]
    except TypeError:
        return []


def _msg_key(m) -> tuple:
    return (m.get("role"), str(m.get("content") or ""))


def _lcs_match(stored: List[dict], incoming: List[dict]) -> List[tuple]:
    """Longest common subsequence of (role, content) keys, as index pairs.

    THE alignment primitive, and the reason this module was rewritten
    (§4BU C1). The previous design asked "do stored and incoming differ by
    at most ONE message?" — a fixed tolerance. Every aborted turn leaves the
    client holding a partial reply where the server holds the full one, i.e.
    one substitution, PERMANENTLY. Two aborts in a session therefore
    exceeded the tolerance, fat detection failed, and the whole conversation
    was concatenated — reproduced end to end: stored 6 → merged 13 with 4
    duplicated messages, and the corruption never healed.

    An LCS makes the decision count-INDEPENDENT: what matters is how much of
    the conversation two views share, not how many edits separate them.
    O(n·m) on lists bounded by MAX_MESSAGES_PER_SESSION, and both sides are
    small (≤400), so this is microseconds.
    """
    n, m = len(stored), len(incoming)
    if not n or not m:
        return []
    # Row-wise DP; keep only two rows (the traceback is rebuilt from choices).
    # For the sizes involved a full table is fine and much simpler to read.
    # Hoist the keys: `_msg_key` stringifies `content`, and for LIST content
    # (multimodal parts) that re-serialises the whole payload on every cell —
    # measured 4.5s for a 160x160 merge, 59ms with the keys precomputed.
    skeys = [_msg_key(x) for x in stored]
    ikeys = [_msg_key(x) for x in incoming]
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        ki = skeys[i]
        row, nxt = table[i], table[i + 1]
        for j in range(m - 1, -1, -1):
            row[j] = (nxt[j + 1] + 1 if ki == ikeys[j]
                      else max(nxt[j], row[j + 1]))
    pairs: List[tuple] = []
    i = j = 0
    while i < n and j < m:
        if skeys[i] == ikeys[j]:
            pairs.append((i, j))
            i += 1
            j += 1
        elif table[i + 1][j] >= table[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def merge_history(stored: List[dict], incoming: List[dict]) -> List[dict]:
    """Build the message list for a turn. See `merge_history_detail`."""
    return merge_history_detail(stored, incoming)[0]


def merge_history_detail(stored: List[dict],
                         incoming: List[dict]) -> Tuple[List[dict], List[dict]]:
    """``(merged, new_tail)`` — the prompt for this turn, and the messages
    this turn genuinely ADDS to the durable store.

    ONE RULE: **stored, plus whatever of incoming is not already accounted
    for.** No client style is inferred, nothing is replaced, and there is no
    tolerance threshold anywhere.

    That is the point. The previous design classified the client — thin /
    fat / diverged / stale — and acted differently for each, with the
    classifier tuned by counts and ratios. Every threshold was a place to be
    wrong, and it was: a tolerance of ONE diverged message meant the SECOND
    aborted stream in a session made fat detection fail, and the whole
    conversation was concatenated every turn from then on (reproduced:
    stored 6 -> merged 13, 4 messages duplicated, permanent). Working
    through the classifier's cases produced three more bugs (a stale tab
    missing MIDDLE messages had them deleted; a two-message conversation
    with one abort stopped aligning; a stale tab whose only reply diverged
    duplicated its whole prefix). The lesson is the shape, not the
    individual bugs: an ambiguous intent inferred from content does not
    become reliable by adding thresholds.

    Appending only the unmatched tail needs no classifier and gives the two
    invariants that matter, by construction:

      * **nothing is dropped** — stored is always a prefix of the result, so
        no divergence shape can delete durable history;
      * **nothing is duplicated** — only messages after the last aligned
        position are appended, so a replay of the whole conversation adds
        exactly its new tail. Growth is linear in turns.

    Assistant and duplicate-system messages are stripped from the tail:
    assistant turns originate server-side (an aborted stream's partial stub
    is the usual imposter, and the real reply is appended by `append_turn`),
    and a thin client re-sends its system prompt every turn, which used to
    grow an unbounded run of duplicates.

    Two accepted consequences, both the safe side of an ambiguity the
    protocol cannot resolve (there are no message ids or sequence numbers):

    1. A diverged client's copy never replaces the server's, so the store
       keeps the FULL reply of an aborted turn rather than the partial one
       the user saw. That is information kept, not lost.
    2. When the user retypes a message verbatim ("ok", "continue") it is
       appended even though identical text is already stored — because the
       alternative is dropping a genuinely new message, which loses the turn
       entirely. Under multi-tab interleaving this can leave one duplicated
       short message in history; harmless, and strictly better than the turn
       vanishing.
    """
    stored = _coerce_messages(stored)
    incoming = _coerce_messages(incoming)
    if not stored:
        return list(incoming), list(incoming)
    if not incoming:
        return stored, []

    pairs = _lcs_match(stored, incoming)

    # THE LAST MESSAGE OF A REPLAY IS NEVER "ALREADY STORED" when it is a
    # user message AND stored continues past the match. The operator retypes
    # short replies, and a greedy alignment matched the newly typed one
    # against an identical older message and consumed it: the model answered
    # the PREVIOUS turn and nothing was persisted. Requiring that stored
    # continue past the match keeps a plain fat replay (whose trailing
    # message really is the stored one) aligned.
    if (pairs and pairs[-1][1] == len(incoming) - 1
            and incoming[-1].get("role") == "user"
            and pairs[-1][0] < len(stored) - 1):
        # ⚠ Keep the ALIGNMENT for the lead/extras split even though this
        # pair is no longer treated as "already stored". Dropping it from
        # `pairs` outright made `first_j` fall back to len(incoming), which
        # collapsed the whole client prefix — including cap-EVICTED messages
        # — into `extras`, i.e. straight into the durable file, undoing the
        # eviction the lead mechanism exists to respect (R2 lens C).
        aligned = pairs
        pairs = pairs[:-1]
    else:
        aligned = pairs

    first_j = aligned[0][1] if aligned else len(incoming)
    if pairs:
        last_j = pairs[-1][1]
    elif aligned:
        # M5 consumed the ONLY pair. `extras` must start AT the message it
        # un-matched (so the retyped turn is genuinely new) and no earlier —
        # falling back to -1 swept the client's cap-evicted prefix into the
        # durable file (R2 lens C).
        last_j = aligned[-1][1] - 1
    else:
        last_j = -1
    stored_systems = {str(m.get("content") or "") for m in stored
                      if m.get("role") == "system"}

    # THE PROMPT MAY BE MORE GENEROUS THAN THE STORE. When the client still
    # holds messages the store's cap EVICTED, they sit before the first
    # aligned position — restoring them gives this turn the context the
    # client actually has, while `new_tail` deliberately excludes them so
    # the cap's decision is not undone in the durable file. This is the one
    # place the two return values differ in kind rather than in length.
    def _is_abort_stub(m) -> bool:
        """A client-side assistant message the client itself marked as an
        aborted stream. The web client writes `content + "\n\n*[Aborted]*"`
        (app.js), so the MARKER is the identity — not a heuristic.

        ⚠ This was a prefix test, and it had three false positives (R4 lens C):
        (1) `if not text: return True` classified every EMPTY-content assistant
        as a stub, which is the shape of a normal tool-calling turn — so a
        legitimate tool round in `lead`, and (via `_drop_orphan_tools`) the
        search results an earlier answer was based on, both vanished from the
        prompt; (2) a complete short reply that happens to PREFIX a longer
        stored one ("Done." vs "Done. I also updated the docs.") was deleted,
        producing the consecutive-user-messages shape `_admissible` exists to
        avoid; (3) any reply merely MENTIONING the marker was compared on its
        head only. Identity, not a property — the marker at the END, or an
        empty message that carries no tool calls (nothing to say and nothing
        to do)."""
        text = str(m.get("content") or "")
        if text.rstrip().endswith("*[Aborted]*"):
            return True
        return not text and not m.get("tool_calls")

    def _admissible(m, *, in_lead: bool) -> bool:
        """One predicate, two regions, and the difference is deliberate.

        ``extras`` becomes ``new_tail`` — it is PERSISTED — and assistant
        turns originate server-side, so every client assistant there is
        dropped (`append_turn` writes the real reply).

        ``lead`` is prompt-only: the client's copy of history the cap
        EVICTED. Dropping every assistant there was an over-correction from
        the R2 fix — it deleted complete replies to evicted turns, emitting
        consecutive user messages and removing most of the context `lead`
        exists to restore (R3 lens C). Drop only the abort stub, which is
        what the R2 case was actually about: a truncated reply sitting
        immediately before the full one."""
        role = m.get("role")
        if role == "assistant":
            return in_lead and not _is_abort_stub(m)
        return not (role == "system"
                    and str(m.get("content") or "") in stored_systems)

    def _drop_orphan_tools(msgs):
        """A ``tool`` message whose ``tool_call_id`` has no surviving
        assistant is a protocol error: most OpenAI-shaped servers reject a
        tool result with no preceding tool_call, and `_clean_messages`
        preserves `tool_call_id` into the DURABLE file — so one orphan
        poisons every future turn of that session (R3 lens C). Filtering
        assistants without filtering their tool results created them."""
        live, any_calls = set(), False
        for m in msgs:
            if m.get("role") != "assistant":
                continue
            calls = m.get("tool_calls")
            # ⚠ `for call in (m.get("tool_calls") or [])` iterated whatever the
            # client sent: `tool_calls: 7` raised TypeError out of the merge,
            # which `routes.py` calls OUTSIDE its try — so a plain-text 500
            # bypassed the OpenAI-shaped error envelope, breaking this
            # module's "never raises into a request" contract. And the route's
            # own validator PERMITS the shape (it allows `content: null` when
            # `tool_calls` is truthy and never checks its type) — R4 lens C.
            if not isinstance(calls, (list, tuple)):
                continue
            any_calls = True
            for call in calls:
                if isinstance(call, dict) and call.get("id"):
                    live.add(str(call["id"]))
        out = []
        for m in msgs:
            if m.get("role") == "tool":
                cid = m.get("tool_call_id")
                if cid is None:
                    # An unidentified result is admissible only if SOME
                    # surviving assistant made calls — otherwise it is the
                    # same orphan by another spelling.
                    if not any_calls:
                        continue
                elif str(cid) not in live:
                    # ⚠ Unless no surviving assistant declared ids at all, in
                    # which case dropping the result creates the MIRROR error:
                    # tool_calls in the prompt with no matching result
                    # (R4 lens C F4).
                    if live or not any_calls:
                        continue
            out.append(m)
        return out

    lead = []
    if aligned and first_j > 0:
        lead = _drop_orphan_tools(
            [m for m in incoming[:first_j] if _admissible(m, in_lead=True)])

    extras = _drop_orphan_tools(
        [m for m in incoming[last_j + 1:] if _admissible(m, in_lead=False)])

    # ⚠ A TAIL REPLAY-FILTER WAS TRIED HERE AND REVERTED (R4). The idea: only
    # the last element of `extras` is the turn being taken, so anything before
    # it that the store already holds is replay and need not be written twice.
    # It removed the multi-tab re-adds it targeted (117 -> 0 at 3 tabs with a
    # cap of 30) and it LOST USER TURNS in the live configuration:
    #
    #   stored   = [u "ok", a A1, u "q2", a A2]     (tab A did the "ok" turn)
    #   incoming = [u "q2", a A2, u "ok", u "q3"]   (tab B's "ok" turn ERRORED;
    #                                                it is replaying to recover)
    #   -> tail = [u "q3"]      tab B's "ok" is gone from the store forever
    #
    # The flaw is that "the store already holds this text" is not the same as
    # "this client is replaying it": `remaining` also counted messages the
    # client never saw — another tab's turns, and the pre-eviction era. A turn
    # that FAILED to persist is not the last element of the tail, so the safety
    # argument ("the new turn is the element this never examines") only covered
    # the current turn, not the one being recovered.
    #
    # Measured end to end with a client model that includes failed turns:
    # 2 clients 35 -> 115 user messages lost, 3 clients 60 -> 126. And at the
    # LIVE cap (400, i.e. effectively uncapped for real conversations) it
    # removed ZERO re-adds while adding 66 losses at 3 tabs. It was tuned on
    # the pathological regime and only cost in the real one.
    #
    # So the accepted position stands, and it is the one the design was chosen
    # for: a duplicate is recoverable, a lost turn is not. Resolving this
    # properly needs message ids in the protocol, which is a client change.
    return lead + stored + extras, list(extras)


class SessionStore:
    """One JSON file per session. Thread-safe; never raises."""

    # Hydration-tier search bounds: only the most-recently-updated sessions
    # are indexed (conversation recall is recency-heavy), snippets are
    # clipped, and one summaries scan is memoized briefly so the 4 RAG-fusion
    # sub-queries of a single hydration share it instead of re-globbing.
    _SEARCH_MAX_SESSIONS = 50
    _SNIPPET_CHARS = 240
    _LIST_MEMO_TTL = 2.0

    def __init__(self, root):
        self.root = Path(root)
        self._lock = threading.Lock()
        # session_id -> (mtime, messages) for search; bounded below.
        self._search_cache: Dict[str, tuple] = {}
        self._list_memo: tuple = (0.0, [])

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
        except json.JSONDecodeError as e:
            # §4M (Lens A MINOR): corrupt-as-absent left the corrupt bytes
            # in place for the NEXT save to overwrite — the conversation
            # history unrecoverable. Preserve a timestamped sidecar first
            # (same policy as playbook/profile/auto_skills), then absent.
            try:
                import time as _time
                backup = path.with_suffix(
                    f".corrupt-{int(_time.time())}")
                os.replace(path, backup)
                logger.warning(
                    "session %s was corrupt (%s); preserved as %s",
                    session_id, e, backup.name)
            except OSError as rename_err:
                logger.error("session %s corrupt AND rename failed: %s",
                             session_id, rename_err)
            return None
        except Exception as e:  # noqa: BLE001 — unreadable → treated absent
            logger.debug("session %s unreadable: %s", session_id, e)
            return None

    def _write(self, sess: Session) -> bool:
        try:
            with self._lock:
                self.root.mkdir(parents=True, exist_ok=True)
                path = self._path(sess.id)
                if path is None:
                    _warn_unpersistable(sess.id)
                    return False
                tmp = path.with_suffix(".tmp")
                # §4M (Lens A MINOR): fsync before the rename so power
                # loss can't promote a torn/empty session file.
                with open(tmp, "w", encoding="utf-8") as fh:
                    fh.write(json.dumps(sess.to_dict(), ensure_ascii=False))
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, path)
            return True
        except Exception as e:  # noqa: BLE001
            # Losing the durable copy of a conversation is not a debug-level
            # event: at debug the operator's live stream showed nothing while
            # every turn silently failed to persist (review R1 M10).
            logger.warning("session write FAILED for %s (%s): %s",
                           sess.id, type(e).__name__, e)
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

    # -- search (hydration tier) ---------------------------------------------

    def _cached_messages(self, session_id: str) -> List[dict]:
        """A session's messages via an mtime-keyed cache, so repeated
        hydration scans only re-parse files that actually changed."""
        path = self._path(session_id)
        if path is None or not path.exists():
            return []
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return []
        cached = self._search_cache.get(session_id)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        sess = self.get(session_id)
        msgs = sess.messages if sess else []
        with self._lock:
            self._search_cache[session_id] = (mtime, msgs)
            while len(self._search_cache) > self._SEARCH_MAX_SESSIONS + 10:
                self._search_cache.pop(next(iter(self._search_cache)))
        return msgs

    def search_messages(self, query: str, limit: int = 5) -> List[dict]:
        """Keyword search over stored conversations — the raw-conversation
        memory tier (2026-07-14). Sessions were previously replay-only:
        durable, but unreachable by any retrieval path.

        Token-overlap scoring (sessions are not vector-indexed; embedding
        400 messages × 200 sessions is not worth it for this tier), scoped
        to the ``_SEARCH_MAX_SESSIONS`` most-recently-updated sessions.
        Multi-term queries must match ≥2 distinct terms — a single shared
        word is noise, not conversation recall. Returns
        ``[{session_id, title, role, text, score}]`` best-first.
        Fail-safe: returns [] rather than raising."""
        try:
            terms = {w.lower().strip('.,?!;"\'()[]{}')
                     for w in str(query or "").split()}
            terms = {w for w in terms if len(w) > 3 and w not in _SEARCH_STOPWORDS}
            if not terms:
                return []
            floor = 2 if len(terms) >= 2 else 1

            now = time.time()
            expires, summaries = self._list_memo
            if now >= expires:
                summaries = self.list(limit=self._SEARCH_MAX_SESSIONS)
                self._list_memo = (now + self._LIST_MEMO_TTL, summaries)

            hits: List[dict] = []
            for meta in summaries:
                for m in self._cached_messages(meta["id"]):
                    if m.get("role") not in ("user", "assistant"):
                        continue
                    content = m.get("content")
                    if not isinstance(content, str) or len(content) < 8:
                        continue
                    lowered = content.lower()
                    score = sum(1 for t in terms if t in lowered)
                    if score >= floor:
                        hits.append({
                            "session_id": meta["id"],
                            "title": meta["title"] or "untitled",
                            "role": m["role"],
                            "text": content[:self._SNIPPET_CHARS],
                            "score": score,
                        })
            hits.sort(key=lambda h: h["score"], reverse=True)
            return hits[:max(1, int(limit))]
        except Exception as e:  # noqa: BLE001 — hydration must never break
            logger.debug("session search failed: %s", e)
            return []

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
                # ⚠ This return, not _write's, is the one a client-picked id
                # actually hits — the whole conversation is dropped here and
                # it was silent.
                _warn_unpersistable(session_id)
                return False
            sess = Session(id=str(session_id))
        new_msgs = _clean_messages(user_messages)
        # A turn's new content must END in the user message that prompted
        # the reply we are about to append. A fat client (the web UI)
        # replays the whole conversation and optimistically renders its
        # OWN copy of the assistant reply before the server persists — so
        # the tail past the stored prefix can end in an assistant message.
        # Left in place, that assistant plus the freshly generated
        # `assistant_content` produced TWO consecutive assistant messages
        # for the same turn (§4N/#31, reproduced 2026-08-08). merge_history's
        # stale-tab branch already strips tail assistants; the fat-replay
        # branch does not, and the caller slices them through to here — so
        # this is the single site that owns the "assistant turns originate
        # server-side" invariant for the append path. Interior assistants
        # (followed by a later user message — a genuinely recovered missed
        # turn) are kept; only TRAILING ones are dropped.
        while new_msgs and new_msgs[-1].get("role") == "assistant":
            new_msgs.pop()
        if assistant_content:
            new_msgs.append({"role": "assistant",
                             "content": str(assistant_content)})
        if not new_msgs:
            return False
        sess.messages.extend(new_msgs)
        # Bound the stored history: keep leading system messages + the most
        # recent tail (mirrors the agent's own 500-message hard cap).
        # ``keep`` must be clamped: with >= MAX systems, ``rest[-0:]`` is the
        # WHOLE tail and a negative keep re-grows it — the cap silently
        # stopped truncating (found 2026-07-15).
        if len(sess.messages) > MAX_MESSAGES_PER_SESSION:
            systems = [m for m in sess.messages if m.get("role") == "system"]
            rest = [m for m in sess.messages if m.get("role") != "system"]
            # ⚠ Reserve room for the CONVERSATION. Keeping every system
            # message first means a session accumulating distinct system
            # prompts drives `keep` to 0 and deletes the entire dialogue,
            # leaving a file of nothing but system messages (reproduced at
            # 420 turns — R3 lens C). Systems are context; the conversation
            # is the point. Cap the systems at a quarter of the budget.
            sys_budget = max(1, MAX_MESSAGES_PER_SESSION // 4)
            # The MOST RECENT ones: past the budget, keeping the oldest means
            # the prompt evicted is the CURRENT one, which is backwards
            # (R4 lens C F7).
            systems = systems[-sys_budget:]
            keep = max(0, MAX_MESSAGES_PER_SESSION - len(systems))
            sess.messages = systems + (rest[-keep:] if keep else [])
        if not sess.title:
            sess.title = derive_title(sess.messages)
        sess.updated_at = _now()
        ok = self._write(sess)
        self._evict()
        return ok

    def _evict(self) -> None:
        """Drop the least-recently-updated sessions past ``MAX_SESSIONS``.

        Enumerates the files directly: ``list()`` clamps its limit at
        ``MAX_SESSIONS``, so going through it can never SEE the overflow —
        the slice past the cap was always empty and eviction was dead code
        (found 2026-07-15). File mtime tracks ``updated_at``: every update
        rewrites the file."""
        try:
            paths = sorted(self.root.glob("*.json"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
            for p in paths[MAX_SESSIONS:]:
                self.delete(p.stem)
        except Exception as e:  # noqa: BLE001
            logger.debug("session eviction skipped: %s", e)


def get_session_store(context) -> Optional[SessionStore]:
    """The context-attached store (wired in main.py), or None."""
    return getattr(context, "session_store", None)


__all__ = [
    "MAX_SESSIONS", "MAX_MESSAGES_PER_SESSION",
    "Session", "SessionStore",
    "derive_title", "merge_history", "merge_history_detail", "get_session_store",
]
