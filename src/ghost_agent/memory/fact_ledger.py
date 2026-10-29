"""The owner's fact ledger — when a fact was stated, replaced or removed, and
by which of the owner's words (§4QM, 2026-10-21).

WHY. A profile value knows when it was learned (`as_of`, §4EM) and nothing
else: a replaced singleton keeps one `previous`, a removed list value goes
to a 30-day archive, an owner graph edge expires with a log line. Nothing
records WHICH request stated a fact or what the owner actually typed, so
"since when do you know where I live?", "what car did I have before?" and
"why do you think my wife is called X?" have no answer, and a wrong owner
fact cannot be traced to the line that put it there.

WHAT. One append-only SQLite table, `fact_ledger.db`, beside the stores. The
profile and graph writers call `record()` at the moment they change an owner
fact (stated / restated / replaced / removed / expired), with the store,
field, value, the previous value when one was replaced, the request id and
the owner's own sentence when the writer knows it (`current_source()` reads
the live request; background writers pass what they matched). The ledger
NEVER decides what is live — reading it changes nothing — and a writer that
cannot reach it still writes its store (the ledger is best-effort, logged).

READS. `history()` for a field or for the words of a question;
`render_lines()` for a prompt or a `recall` result; `provenance_for()` for
one live value. Hydration of these lines is behind `GHOST_FACT_HISTORY` /
the probe arm `fact_history` until the paired replay proves it (§4QM).
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger("GhostAgent")

DB_FILENAME = "fact_ledger.db"
EVENTS = ("stated", "restated", "replaced", "removed", "expired", "backfilled")
SAID_CHARS = 200
ENV_FLAG = "GHOST_FACT_HISTORY"
ARM = "fact_history"
HEADER = "### WHAT YOU TOLD ME, AND WHEN (your own facts with their dates and your words):"
#: a self-question that asks about the HISTORY of a fact, not just its value
HISTORY_WORDS_RE = re.compile(
    r"\b(?:since when|how long|when did i|what did i (?:tell|say)|did i (?:ever|tell|say|mention)|"
    r"before (?:that|this|the)|used to|previous(?:ly)?|why do you (?:think|believe|say)|how do you know|"
    r"where did you (?:get|learn)|first (?:time|mention)|changed?|history|still)\b"
    r"|από πότε|πότε (?:σου )?είπα|πριν|γιατί (?:νομίζεις|πιστεύεις)|άλλαξ", re.IGNORECASE)

_STOP = frozenset("""a an the and or of in on at for to is are was were my me i you your do does did what
which who when where why how have has had tell know remember about still ever since before that this
these those it its μου σου το τα της του και να σε με για από""".split())


def fact_history_enabled() -> bool:
    """ON in production only by the env switch; ON for a labelled probe that
    selected the ``fact_history`` arm (the replay). `prompt_arm()` is "" on
    every production request."""
    if os.environ.get(ENV_FLAG, "0").strip().lower() in ("1", "true", "on", "yes"):
        return True
    try:
        from ..utils.logging import prompt_arm
        return prompt_arm() == ARM
    except Exception:  # noqa: BLE001
        return False


def current_source(via: str = "", value=None, field=None) -> Dict[str, str]:
    """The live request's id and the owner's CLAUSE behind ``value``, for a
    writer that runs INSIDE the turn (the update_profile tool, the bus).
    Never the whole message (review r1 MAJOR: a token pasted beside "my car
    is a Ducati" was stored and replayed into prompts): the one clause of the
    owner's statements that carries the value — preferring the clause with
    the field's predicate cue ("moved", "wife", "called"), then the widest
    coverage — or "" when none does. Background writers (the extractor) pass
    their own: the request they consolidate is not the one in flight. Never
    raises."""
    out = {"req_id": "", "said": "", "via": str(via or "")}
    try:
        from ..utils.logging import request_id_context
        out["req_id"] = str(request_id_context.get() or "")[:64]
    except Exception:  # noqa: BLE001
        pass
    if value is None:
        return out
    try:
        from .lesson_scope import current_request
        from .attribution import owner_clause_for, cue_for_field
        from ..utils.provenance import added_user_messages
        cue = cue_for_field(*(field or ("", ""))) if field else None
        # §4RZ step 4: the clause can come from a message the user added mid-turn
        _said = "\n".join("USER: " + t for t in [str(current_request.get() or "")] + added_user_messages())
        out["said"] = clip_said(owner_clause_for(str(value), _said, cue=cue))
    except Exception:  # noqa: BLE001
        pass
    return out


def redact_said(text) -> str:
    """The owner's words as the trajectory recorder keeps them: secrets,
    keys, card numbers and paths masked (distill/redact). Best-effort: on a
    broken redactor the words are DROPPED, never stored raw."""
    t = str(text or "")
    if not t:
        return ""
    try:
        from ..distill.redact import redact_text
        return str(redact_text(t) or "")
    except Exception:  # noqa: BLE001
        return ""


def clip_said(text) -> str:
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(t) <= SAID_CHARS:
        return t
    cut = t[:SAID_CHARS]
    return cut.rsplit(" ", 1)[0] + "…"


def query_words(text: str) -> List[str]:
    toks = re.findall(r"[^\W\d_]{3,}|\d{4}", str(text or "").lower())
    return [w for w in toks if w not in _STOP]


def _day(at) -> str:
    """The LOCAL day (review r1: UTC days said "since yesterday" for a fact
    stated after midnight local time)."""
    try:
        return _dt.datetime.fromtimestamp(float(at)).strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001
        return "undated"


def _epoch(at) -> Optional[float]:
    """An epoch for a stamp (epoch, ISO text, "YYYY-MM-DD HH:MM:SS[ UTC]");
    ``None`` when there is none or it cannot be read — the CALLER decides
    what an unknown date means (`record` → now; the backfill → undated),
    never a silent "today" (review r1)."""
    if at is None or at == "":
        return None
    if isinstance(at, (int, float)):
        return float(at) if float(at) > 0 else None
    s = re.sub(r"\s*(?:UTC|GMT)$", "", str(at).strip()).replace("Z", "+00:00")
    try:
        d = _dt.datetime.fromisoformat(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=_dt.timezone.utc)
        return d.timestamp()
    except Exception:  # noqa: BLE001
        return None


def _plain(value) -> str:
    """A stored value (plain, stamped dict, or a list of either) as text."""
    if isinstance(value, dict) and "v" in value:
        value = value.get("v")
    if isinstance(value, list):
        return "; ".join(_plain(v) for v in value if v not in (None, ""))
    return re.sub(r"\s+", " ", str(value if value is not None else "")).strip()[:300]


class FactLedger:
    def __init__(self, memory_dir):
        self.path = Path(memory_dir) / DB_FILENAME
        self._lock = threading.RLock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(str(self.path), timeout=10)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    at REAL NOT NULL, store TEXT NOT NULL, category TEXT NOT NULL, key TEXT NOT NULL,
                    value TEXT NOT NULL, event TEXT NOT NULL, prev_value TEXT DEFAULT '',
                    req_id TEXT DEFAULT '', said TEXT DEFAULT '', via TEXT DEFAULT '', how TEXT DEFAULT '');
                CREATE INDEX IF NOT EXISTS ledger_field ON ledger(category, key, at);
            """)

    # -- writes ---------------------------------------------------------------

    def record(self, *, store: str, category: str, key: str, value, event: str, prev_value=None,
               source: Optional[Dict[str, str]] = None, how: str = "", at=None) -> int:
        """Append one row. ``event`` ∈ EVENTS. Never raises: a ledger that
        cannot be written is logged and the store write it describes stands."""
        try:
            ev = str(event or "").strip().lower()
            if ev not in EVENTS:
                raise ValueError(f"unknown ledger event {event!r}")
            src = dict(source or {})
            at_e = _epoch(at)
            if at_e is None:
                at_e = time.time()
            with self._lock, self._conn() as c:
                cur = c.execute(
                    "INSERT INTO ledger(at, store, category, key, value, event, prev_value, req_id, said, via, how)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (at_e, str(store or "")[:16], str(category or "").strip().lower()[:80],
                     str(key or "").strip().lower()[:120], _plain(value), ev, _plain(prev_value),
                     str(src.get("req_id") or "")[:64], clip_said(redact_said(src.get("said"))),
                     str(src.get("via") or "")[:32], str(how or "")[:80]))
                return int(cur.lastrowid or 0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("fact ledger: row not written (%s: %s)", type(exc).__name__, exc)
            return 0

    # -- reads ----------------------------------------------------------------

    def count(self) -> int:
        with self._lock, self._conn() as c:
            return int(c.execute("SELECT COUNT(*) FROM ledger").fetchone()[0])

    def history(self, category: Optional[str] = None, key: Optional[str] = None, words: Iterable[str] = (),
                limit: int = 20, distinct: bool = False) -> List[dict]:
        """Rows for a field (exact category/key when given) and/or rows whose
        key, value or the owner's words share a query word; newest first.
        ``distinct``: one row per (field, value, event) — its NEWEST — found in
        SQL, so a value restated a hundred times cannot push the older
        replaced/expired rows a question is about out of the scan (review r1)."""
        words = [str(w).lower() for w in words if str(w).strip()]
        with self._lock, self._conn() as c:
            if category or key:
                sql = "SELECT * FROM ledger WHERE 1=1"
                args: list = []
                if category:
                    sql += " AND category = ?"; args.append(str(category).lower())
                if key:
                    sql += " AND key = ?"; args.append(str(key).lower())
                rows = [dict(r) for r in c.execute(sql + " ORDER BY at DESC, id DESC LIMIT ?", args + [int(limit)])]
            elif words:
                if distinct:
                    # the newest row of each group renders; the question's words are
                    # matched against EVERY row's words (review r2: a restatement with
                    # no words hid the group the older row's words had found)
                    sql = ("SELECT id, MAX(at) AS at, store, category, key, value, event, prev_value, req_id, said,"
                           " via, how, GROUP_CONCAT(said, ' ') AS said_all, GROUP_CONCAT(prev_value, ' ') AS prev_all"
                           " FROM ledger GROUP BY category, key, value, event ORDER BY at DESC, id DESC")
                else:
                    sql = "SELECT * FROM ledger ORDER BY at DESC, id DESC LIMIT 2000"
                rows = [dict(r) for r in c.execute(sql)]
                rows = [r for r in rows if _row_matches(r, words)][: int(limit)]
            else:
                rows = []
        return rows

    def provenance_for(self, category: str, key: str, value) -> Optional[dict]:
        """The newest row that stated/restated/backfilled exactly this value."""
        want = _plain(value)
        for r in self.history(category, key, limit=200):
            if r["value"] == want and r["event"] in ("stated", "restated", "replaced", "backfilled"):
                return r
        return None

    @staticmethod
    def render_row(r: dict) -> str:
        field = f"{r.get('category')}.{r.get('key')}".strip(".")
        how = str(r.get("how") or "").strip()
        # a backfilled value without a stamp is dated by the backfill RUN:
        # never "since today" (review r2)
        when = "date unknown" if "undated" in how else _day(r.get("at"))
        said = str(r.get("said") or "").strip()
        notes = [f"you said: \"{said}\""] if said else []
        if how.startswith("confirmed by you"):
            notes.append(how)      # §4QN: a fact the owner confirmed in person, with no clause to quote
        tail = f" ({'; '.join(notes)})" if notes else ""
        ev = r.get("event")
        v = str(r.get("value") or "")
        if ev == "backfilled" and how.startswith("previous value"):
            return f"- {field} = {v} — on record {when}, later replaced{tail}"
        if ev == "backfilled":
            # a seeded row is dated by the STORE's stamp (a profile's last write, a
            # graph edge's creation), not by the owner's statement — never "since"
            # (§4QM replay: a graph row dated 2026-09-21 became "you told me on
            # 2026-09-21" for a fact the owner stated in July)
            return f"- {field} = {v} — on record since {when}{tail}"
        if ev == "stated":
            return f"- {field} = {v} — since {when}{tail}"
        if ev == "restated":
            return f"- {field} = {v} — confirmed again {when}{tail}"
        if ev == "replaced":
            prev = str(r.get("prev_value") or "")
            return f"- {field} = {v} — since {when}, replacing \"{prev}\"{tail}"
        if ev in ("removed", "expired"):
            # the owner's words tell the story; an internal reason ("prune:bmw", "delete")
            # shows only without them, and a confirmation is already in the notes (§4QN r2)
            how_t = f" ({how})" if how and not said and not how.startswith("confirmed by you") else ""
            return f"- {field}: \"{v}\" {ev} {when}{tail}{how_t}"
        return f"- {field} = {v} — {ev} {when}{tail}"

    def render_lines(self, rows: Iterable[dict], limit: int = 8) -> List[str]:
        out = []
        for r in rows:
            out.append(self.render_row(r))
            if len(out) >= limit:
                break
        return out

    def lines_for_question(self, question: str, limit: int = 8, exclude_fields=(), scrub=None,
                           exclude=None) -> List[str]:
        """The history lines a self-question earns: rows whose field or value
        or the owner's words share a content word with the question.
        ``exclude_fields``: ``{"category.key", …}`` never rendered (the
        profile's on-demand fields — the street address, a physical
        description — ride no every-turn prompt, §4LB); ``exclude``: a
        predicate ``(category, key, value) -> bool`` with the SAME reach as
        the profile's own on-demand rule (`is_on_demand`: `*_description`
        keys, address-like keys, street-looking values — review r1: the
        fixed set alone let `root.appearance_description` through);
        ``scrub``: a callable applied to each line (the profile's egress
        scrubber, so an address quoted inside the owner's words of ANOTHER
        field is masked)."""
        rows = self.history(words=query_words(question), limit=limit * 3, distinct=True)
        excl = {str(f).lower() for f in (exclude_fields or ())}
        kept = []
        for r in rows:            # one row per (field, value, event) already: `distinct`
            if f"{r['category']}.{r['key']}".lower() in excl:
                continue
            if callable(exclude):
                try:
                    if exclude(r["category"], r["key"], r["value"]):
                        continue
                except Exception:  # noqa: BLE001
                    continue          # an undecidable field stays out of the prompt
            kept.append(r)
        lines = self.render_lines(kept, limit=limit)
        if callable(scrub):
            lines = [scrub(ln) or ln for ln in lines]
        return lines


def _row_matches(r: dict, words: List[str]) -> bool:
    hay = " ".join(str(r.get(k) or "") for k in ("category", "key", "value", "prev_value", "said",
                                                 "prev_all", "said_all")).lower()
    hay = hay.replace("_", " ")
    return any(re.search(r"(?<![^\W_])" + re.escape(w) + r"(?![^\W_])", hay) for w in words)
