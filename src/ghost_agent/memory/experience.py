"""Outcome-labelled experience memory (§4QL, 2026-10-21).

WHY. The 10-21 census put knowledge answers at 29 % right, most of the
failures being specifics invented on no-tool turns, and the one lesson that
ever moved an answer (§4ON) was the owner's version rule — "go to the vendor
page", a memory of WHERE evidence came from. The agent already holds every
owner turn with its tool calls (the trajectory corpus) and ~300 HUMAN labels
for them (the gold/census grades, the owner's 👍/👎), but nothing joined the
two to retrieval; the unlabelled episode tier that once hydrated prompts was
measured 15 % relevant / 19 % harmful (§4NC) and retired.

WHAT. One row per past OWNER turn that a human graded: the request, the
grade and the grader's reason, the tool steps it took, the sources that
answered it (URLs the browser opened without error) and the ones that
failed, and — for a graded RIGHT answer to a question whose answer does not
move — a short gist of the answer. On an owner turn whose request is close
to a stored one (cosine ≥ ``SIMILARITY_FLOOR`` on the same embedder the
vector store uses, or the same wording), the bus hydrates at most two lines:
how the best RIGHT case was answered, and what went wrong in the best
WRONG/PARTIAL case. Never a probe's, a member's, a public-channel or an
ungraded turn; never a verdict of the agent's own judges (the late verifier,
the reaction judge) — those were 9/17 wrong on real traffic (§4OW).

LABELS, in precedence: a gold grade (``system/census/gold/<date>/grades.jsonl``,
keyed by trajectory id) > a census grade (``system/eval/census/<date>.jsonl``,
keyed by req_id) > the owner's own thumb in the corrections sidecar
(``human_feedback:cli|web|slack:owner`` — a Slack *requester*'s reaction may be
a member's and is not used). A thumb is coarser than a grade ("satisfied",
not "every specific right"), so a thumb-labelled case never renders its
answer text, only its steps and sources.

OFF by default: ``GHOST_EXPERIENCE_TIER=1`` turns the tier on for owner
turns; a labelled probe can select it with ``X-Ghost-Arm: experience`` for
the paired replay that has to prove it first. The index is rebuilt by
``ExperienceMemory.sync`` (the idle store maintenance, and the operator's
``scripts/experience_sync_4ql.py``) — one writer, idempotent, from the
trajectory collector and the label files; ``excluded`` holds trajectory ids
held out of the index for an experiment.
"""
from __future__ import annotations

import ast
import datetime as _dt
import json
import logging
import math
import os
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlsplit

logger = logging.getLogger("GhostAgent")

DB_FILENAME = "experience.db"
#: below this cosine a stored request is not "a turn like this one" — measured
#: offline on 640 owner turns: at 0.80 "correct." matched "yes" and a greeting
#: matched "i'm good, how are you?"; identical wording always qualifies
SIMILARITY_FLOOR = 0.85
#: a request with fewer content words than this is a go-ahead or a greeting
#: ("ok", "yes do it", "hello ghost", "correct.") — it names nothing a past
#: turn could share; "give me the news" (give, news) still qualifies
MIN_CONTENT_WORDS = 2
#: most cases rendered on one turn: the best RIGHT one and the best bad one
MAX_CASES = 2
MAX_LINE_CHARS = 700
ANSWER_GIST_CHARS = 220
REASON_CHARS = 240
GRADES = ("RIGHT", "PARTIAL", "WRONG", "FAILED-HONEST")
GOOD_GRADES = frozenset({"RIGHT"})
LABEL_RANK = {"gold": 3, "census": 2, "thumb": 1}
#: thumb sources that are the OWNER's own reaction (a Slack requester may be a member)
OWNER_THUMB_SOURCES = ("human_feedback:cli", "human_feedback:web", "human_feedback:slack:owner",
                       "human_feedback:clockwork", "human_feedback:api",
                       "user_correction",   # the owner's own next-message correction (always a failure)
                       "operator", "operator_overlay")   # the operator's manual relabel (passed/failed only)
#: a thumb's default note is not a diagnosis
_EMPTY_REASONS = frozenset({"human negative feedback", "human positive feedback", "none", "n/a", "-"})
ARM = "experience"
ENV_FLAG = "GHOST_EXPERIENCE_TIER"
#: bump when what `sync` DERIVES from a trajectory (steps, sources, the
#: answer gist, the moving rule) changes: rows of an older version are
#: recomputed on the next sync, their vectors kept
DERIVED_VERSION = 5
#: the kinds whose RIGHT answer may be shown as "answer then": a knowledge
#: question or a no-tool answer; a status/introspection or a tool-driven turn
#: ("execute → introspect → scratchpad") is a reading of its day
ANSWER_KINDS = frozenset({"knowledge", "no-tool"})
HEADER = ("### EXPERIENCE (how past turns like this one went, graded by a human — "
          "a record of THEN, not facts about now; reuse the steps and sources, "
          "not the answers):")

_STOP = frozenset("""a an the and or but so to of in on at for with from by as is are was were be been
being do does did have has had i me my mine you your it its this that these those we our us they them
their he she his her what which who whom when where why how can could would should will shall may might
must please pls ok okay yes no not now then there here also too very just about into over under again
ghost hello hi hey good morning evening afternoon thanks thank της του το τα τον την και να σε με για
από στο στη στην είναι ειναι μου σου μας θα τι πώς πως που ποιο ποια ποιος""".split())

#: a question whose answer is a reading of THAT day — the weather, the news,
#: a price, "now" — beside the episode store's version/price rule; such a
#: case renders its steps and sources, never the answer it gave then
_TIME_BOUND_RE = re.compile(
    r"\b(?:weather|forecast|news|headlines?|today|tonight|tomorrow|yesterday|now|currently|right now|"
    r"this (?:week|month|year)|latest|newest|current|price|prices|stock|score|scores|standings|"
    r"σήμερα|σημερα|αύριο|αυριο|καιρός|καιρος|ειδήσεις|ειδησεις|τώρα|τωρα|τιμή|τιμη)\b", re.IGNORECASE)

#: tools whose argument names a page/stream the turn READ
_URL_ARG_KEYS = ("url", "link", "page_url", "video_url")
_SEARCH_TOOLS = frozenset({"web_search", "deep_research", "darkweb_search", "darkweb_research", "fact_check"})


def experience_tier_enabled() -> bool:
    """ON for production owner turns only by the env switch; ON for a labelled
    probe that selected the ``experience`` arm (so the paired replay can
    measure it before the switch flips). ``prompt_arm()`` is "" on every
    production request."""
    if os.environ.get(ENV_FLAG, "0").strip().lower() in ("1", "true", "on", "yes"):
        return True
    try:
        from ..utils.logging import prompt_arm
        return prompt_arm() == ARM
    except Exception:  # noqa: BLE001
        return False


def normalize_request(text: str) -> str:
    """The wording folded: case, punctuation, whitespace, angle-bracketed
    links (Slack wraps URLs in <>), so a repeat reads as the same request."""
    t = str(text or "").lower().replace("’", "'")
    t = re.sub(r"<(https?://[^>|\s]+)(?:\|[^>]*)?>", r"\1", t)
    t = re.sub(r"[^\w\s/:.?=&%-]+", " ", t)
    # sentence punctuation at a token's end is not wording ("postgresql ?" = "PostgreSQL?");
    # the URL characters above are kept only inside a token
    t = re.sub(r"\s*[?.!,;:]+(?=\s|$)", "", t)
    return re.sub(r"\s+", " ", t).strip()


def content_words(text: str) -> List[str]:
    toks = re.findall(r"[^\W\d_]{2,}|\d{2,}", normalize_request(text))
    return [w for w in toks if w not in _STOP]


def _args_of(call: dict) -> dict:
    a = call.get("arguments")
    if isinstance(a, dict):
        return a
    s = str(a or "").strip()
    if not s:
        return {}
    for loader in (json.loads, ast.literal_eval):
        try:
            v = loader(s)
            return v if isinstance(v, dict) else {}
        except Exception:  # noqa: BLE001
            continue
    return {}


def _page_only(url: str) -> str:
    """scheme + host + path: a query string is the owner's own trip dates,
    search terms or tokens (review: `omio.com/…?dateFrom=2026-10-26&passengers=1`)."""
    try:
        sp = urlsplit(url)
        return f"{sp.scheme}://{sp.netloc}{sp.path}" if sp.netloc else url
    except Exception:  # noqa: BLE001
        return url


def _host(url: str) -> str:
    try:
        return urlsplit(url).netloc.lower().removeprefix("www.")
    except Exception:  # noqa: BLE001
        return ""


def _failed(call: dict) -> bool:
    """A stored tool row failed when the recorder kept an error, or the
    result opens with the failure banner the loop writes (§4PD: failure is
    the TYPE, but a stored row only has text — the banner head is the one
    mark every failed browser/search row carries)."""
    if str(call.get("error") or "").strip():
        return True
    head = str(call.get("result") or "").lstrip()[:160].lower()
    return head.startswith("[failure banner]") or head.startswith("not done:")


_BANNER_NOISE_RE = re.compile(r"\[failure banner\]|-{3}\s*browser result\s*-{3}|-{3}\s*\w+ result\s*-{3}", re.IGNORECASE)
_BANNER_TAIL_RE = re.compile(r"\s+(?:op|hint|url|---)\s*:.*$", re.IGNORECASE | re.DOTALL)


def _error_head(call: dict) -> str:
    """The failure's own words, not the page's: the recorder's ``error``, else
    the failure banner's status clause — never anything after ``hint:`` /
    ``op:`` (the banner is followed by whatever the page sent)."""
    raw = str(call.get("error") or "").strip() or str(call.get("result") or "")[:300]
    head = _BANNER_NOISE_RE.sub(" ", raw)
    head = _BANNER_TAIL_RE.sub("", head)
    return re.sub(r"\s+", " ", head).strip()[:90]


def extract_steps(tool_calls: Iterable[Any]) -> Tuple[List[str], List[str], List[str]]:
    """``(steps, sources, failed_sources)`` from a trajectory's tool calls.
    A step is ``name`` or ``name(FAILED)``; a source is a URL a non-failed
    call read; a failed source is ``host — <error head>``. Page prose is
    never kept: only the agent's own call arguments."""
    steps: List[str] = []
    sources: List[str] = []
    failed: List[str] = []
    for c in tool_calls or ():
        if not isinstance(c, dict):
            try:
                c = {"name": getattr(c, "name", ""), "arguments": getattr(c, "arguments", {}),
                     "result": getattr(c, "result", ""), "error": getattr(c, "error", "")}
            except Exception:  # noqa: BLE001
                continue
        name = str(c.get("name") or "").strip().lower().replace("-", "_")
        if not name:
            continue
        bad = _failed(c)
        args = _args_of(c)
        detail = ""
        url = ""
        for k in _URL_ARG_KEYS:
            v = args.get(k)
            if isinstance(v, str) and v.startswith(("http://", "https://")) and "<REDACTED" not in v:
                url = _page_only(v.strip())[:200]
                break
        if name in _SEARCH_TOOLS:
            q = args.get("query") or args.get("q") or args.get("topic")
            if isinstance(q, str) and q.strip():
                detail = json.dumps(q.strip()[:80], ensure_ascii=False)
        elif url:
            detail = _host(url)
        steps.append(f"{name}({detail})" if detail else name)
        if bad:
            steps[-1] += " FAILED"
        if url:
            if bad:
                failed.append(f"{_host(url)} — {_error_head(c)}".rstrip(" —"))
            else:
                sources.append(url)
    # one mention per source, order kept
    sources = list(dict.fromkeys(sources))
    failed = list(dict.fromkeys(failed))
    return steps[:12], sources[:6], failed[:4]


def is_moving_target(text: str) -> bool:
    """A question whose answer is a reading of its day: the episode store's
    rule (§4MM: "latest version", "price now" — imported lazily so this leaf
    module stays importable alone) or a time-bound word in the request."""
    t = str(text or "")
    if _TIME_BOUND_RE.search(t):
        return True
    try:
        from .episodes import is_moving_target_question
        return bool(is_moving_target_question(t))
    except Exception:  # noqa: BLE001
        return False


def kind_of(tool_calls: Iterable[Any]) -> str:
    names = []
    for c in tool_calls or ():
        n = c.get("name") if isinstance(c, dict) else getattr(c, "name", "")
        names.append(str(n or "").lower())
    if "image_generation" in names:
        return "image"
    if not names:
        return "no-tool"
    if all(n in _SEARCH_TOOLS or n in ("browser", "news_headlines", "fact_check") for n in names):
        return "knowledge"
    return "tools"   # a build, a file write, a service, an execute: its reply is a status, not an answer


# ── labels ──────────────────────────────────────────────────────────────────

def load_labels(system_dir) -> Dict[str, Tuple[str, str, str]]:
    """``{trajectory_id: (grade, source, reason)}`` from the human label files
    under ``system/``. Gold grades are keyed by trajectory id; census grades
    by req_id (resolved by the caller through ``req_id → tid``: this function
    returns them under ``"req:<req_id>"`` keys for ``sync`` to resolve); the
    owner's thumbs come from the corrections sidecar. Judge verdicts
    (``verifier_late``, the reaction judge) are NOT labels."""
    sysd = Path(system_dir)
    out: Dict[str, Tuple[str, str, str]] = {}

    def _put(key: str, grade: str, source: str, reason: str) -> None:
        grade = str(grade or "").strip().upper()
        if grade not in GRADES or not key:
            return
        prev = out.get(key)
        if prev is None or LABEL_RANK.get(source, 0) >= LABEL_RANK.get(prev[1], 0):
            out[key] = (grade, source, _clip_reason(reason))

    for p in sorted((sysd / "census" / "gold").glob("*/grades.jsonl")) if (sysd / "census" / "gold").is_dir() else []:
        for d in _jsonl(p):
            _put(str(d.get("tid") or ""), d.get("grade"), "gold", d.get("reason"))
    for p in sorted((sysd / "eval" / "census").glob("*.jsonl")) if (sysd / "eval" / "census").is_dir() else []:
        for d in _jsonl(p):
            rid = str(d.get("req_id") or "")
            if rid:
                _put("req:" + rid, d.get("grade"), "census", d.get("reason"))
    corr = sysd / "trajectories" / "corrections.jsonl"
    if corr.is_file():
        for d in _jsonl(corr):
            src = str(d.get("source") or "")
            if src not in OWNER_THUMB_SOURCES:
                continue
            grade = "RIGHT" if d.get("outcome") == "passed" else "WRONG" if d.get("outcome") == "failed" else ""
            _put(str(d.get("trajectory_id") or ""), grade, "thumb", d.get("reason"))
    return out


_FIGURE_RE = re.compile(r"(?<![\w.])(?:[$€£¥]\s?)?v?\d+(?:[.,]\d+)+(?!\w|\.\d)|(?<![\w.])\d{1,3}(?:,\d{3})+(?!\w|\.\d)")


def mask_figures(text: str) -> str:
    """Versions, prices and other dotted/comma figures in a moving-target
    case's reason → ‹figure›: the grader's "shows 18.6" is the answer of
    THEN, and a block that carries it lets the model parrot it without a
    tool (replay design review, point 10). Years and plain integers stay."""
    return _FIGURE_RE.sub("‹figure›", str(text or ""))


def _clip_reason(reason) -> str:
    """The grader's reason, one line, cut at a sentence end inside the cap
    (a reason cut mid-clause — "18.4 came from an undated snippet, and the" —
    reads as if it ended there)."""
    r = re.sub(r"\s+", " ", str(reason or "")).strip()
    if r.lower() in _EMPTY_REASONS:
        return ""
    if len(r) <= REASON_CHARS:
        return r
    cut = r[:REASON_CHARS]
    end = max(cut.rfind(". "), cut.rfind("; "), cut.rfind("! "), cut.rfind("? "))
    return (cut[:end + 1] if end >= REASON_CHARS // 2 else cut.rsplit(" ", 1)[0] + "…").strip()


def best_label(*candidates):
    """The highest-ranked of the labels found under a turn's keys (tid and
    req_id) — a thumb must not beat a census grade because it was looked up
    first (review)."""
    best, best_rank = None, -1
    for c in candidates:
        rank = LABEL_RANK.get(c[1], 0) if c else -1
        if c and rank > best_rank:
            best, best_rank = tuple(c), rank
    return best


def _int(v) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def is_system_command(request: str) -> bool:
    """A command the system answers without the model (§4PY): its reply is a
    listing of THEN, never an answer worth repeating. §4SZ: the commands'
    OWN matchers — the loose pattern that stood here ("show … tests",
    "list … rules") also took "show me the failing tests" for one."""
    from ..core.owner_rules import is_rule_command
    from ..selfhood.mirror import is_mirror_command
    return is_rule_command(request) or is_mirror_command(request)


def _jsonl(path: Path) -> Iterable[dict]:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if isinstance(d, dict):
                    yield d
    except OSError:
        return


# ── the store ───────────────────────────────────────────────────────────────

class ExperienceMemory:
    """SQLite rows + the request embedding kept beside each row (384 floats);
    similarity is a brute-force cosine over a few hundred vectors, so this
    store never touches the vector collection (no phantom retrieval counts,
    no store lock, nothing new for `recall` to rank)."""

    def __init__(self, memory_dir, embed_fn: Optional[Callable[[List[str]], Any]] = None):
        self.path = Path(memory_dir) / DB_FILENAME
        self.embed_fn = embed_fn
        self._lock = threading.RLock()
        self._cache: Optional[List[dict]] = None
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(str(self.path), timeout=10)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS experience (
                    tid TEXT PRIMARY KEY, req_id TEXT DEFAULT '', ts REAL NOT NULL,
                    request TEXT NOT NULL, request_norm TEXT NOT NULL, kind TEXT DEFAULT '',
                    grade TEXT NOT NULL, label_source TEXT NOT NULL, reason TEXT DEFAULT '',
                    steps TEXT DEFAULT '[]', sources TEXT DEFAULT '[]', failed_sources TEXT DEFAULT '[]',
                    answer TEXT DEFAULT '', moving INTEGER DEFAULT 0, vec BLOB,
                    labelled_at TEXT DEFAULT '', derived_v INTEGER DEFAULT 1);
                CREATE TABLE IF NOT EXISTS excluded (tid TEXT PRIMARY KEY, note TEXT DEFAULT '');
            """)
            cols = {r[1] for r in c.execute("PRAGMA table_info(experience)")}
            if "derived_v" not in cols:
                c.execute("ALTER TABLE experience ADD COLUMN derived_v INTEGER DEFAULT 1")

    # -- writes ---------------------------------------------------------------

    def set_excluded(self, tids: Iterable[str], note: str = "") -> int:
        """Hold these trajectories OUT of the index (an experiment's own cases,
        leave-one-out); `sync` drops them and never re-adds them until cleared."""
        tids = [str(t) for t in tids if str(t)]
        with self._lock, self._conn() as c:
            c.executemany("INSERT OR REPLACE INTO excluded(tid, note) VALUES (?, ?)",
                          [(t, str(note or "")) for t in tids])
            if tids:
                c.executemany("DELETE FROM experience WHERE tid = ?", [(t,) for t in tids])
        self._cache = None
        return len(tids)

    def clear_excluded(self) -> int:
        with self._lock, self._conn() as c:
            n = c.execute("SELECT COUNT(*) FROM excluded").fetchone()[0]
            c.execute("DELETE FROM excluded")
        self._cache = None
        return int(n)

    def excluded(self) -> List[str]:
        with self._lock, self._conn() as c:
            return [r[0] for r in c.execute("SELECT tid FROM excluded")]

    def sync(self, trajectories: Iterable[Any], labels: Dict[str, Tuple[str, str, str]]) -> Dict[str, int]:
        """Rebuild the index from ``trajectories`` (collector rows: dicts or
        ``Trajectory`` objects) and ``labels`` (`load_labels`). Only an
        OWNER's delivered ``user_request`` turn with a human label and a
        request of ≥ ``MIN_CONTENT_WORDS`` content words is indexed; a row
        already present with the same label is left alone (its vector is
        kept); a changed label rewrites the row; a trajectory no longer
        labelled, or excluded, is removed. Returns counts."""
        from ..core.failure_census import is_owner_turn, delivered, _traj_dict, _blind_reply
        counts = {"seen": 0, "eligible": 0, "added": 0, "updated": 0, "removed": 0, "skipped_short": 0}
        with self._lock:
            excluded = set(self.excluded())
            with self._conn() as c:
                have = {r["tid"]: (r["grade"], r["label_source"], r["reason"]) for r in
                        c.execute("SELECT tid, grade, label_source, reason FROM experience")}
                stale = {r[0] for r in c.execute("SELECT tid FROM experience WHERE derived_v != ?",
                                                 (DERIVED_VERSION,))}
            keep: set = set()
            seen: set = set()
            pending: List[dict] = []
            for tr in trajectories:
                counts["seen"] += 1
                t = _traj_dict(tr)
                tid = str(t.get("id") or "")
                if not tid or tid in excluded or not is_owner_turn(t) or not delivered(t):
                    continue
                seen.add(tid)
                ex = t.get("extra") or {}
                if str(ex.get("surface") or "") == "public":
                    continue          # a public-channel reply is not the owner's private record (§4LA)
                rid = str(ex.get("req_id") or "")
                lab = best_label(labels.get(tid), labels.get("req:" + rid) if rid else None)
                if not lab:
                    continue
                followup = _int(ex.get("conv_user_turns")) > 1
                request = str(t.get("user_request") or "").strip()
                if len(content_words(request)) < MIN_CONTENT_WORDS:
                    counts["skipped_short"] += 1
                    continue
                counts["eligible"] += 1
                keep.add(tid)
                _lab_stored = (lab[0], lab[1], mask_figures(lab[2]) if is_moving_target(request) else lab[2])
                if have.get(tid) == _lab_stored and tid not in stale:
                    continue
                steps, sources, failed = extract_steps(t.get("tool_calls") or [])
                moving = is_moving_target(request)
                kind = kind_of(t.get("tool_calls") or [])
                reason = mask_figures(lab[2]) if moving else lab[2]
                answer = ""
                # a follow-up's wording is context-free ("is there a way to run it on a
                # mac?"): its answer belongs to THAT conversation's subject (review)
                if (lab[0] in GOOD_GRADES and lab[1] != "thumb" and not moving and kind in ANSWER_KINDS
                        and not followup and not is_system_command(request)):
                    answer = re.sub(r"\s+", " ", _blind_reply(str(t.get("final_response") or "")))[:ANSWER_GIST_CHARS]
                pending.append({
                    "tid": tid, "req_id": rid, "ts": _epoch(t.get("timestamp")), "request": request[:500],
                    "request_norm": normalize_request(request)[:500], "kind": kind,
                    "grade": lab[0], "label_source": lab[1], "reason": reason, "steps": steps, "sources": sources,
                    "failed_sources": failed, "answer": answer, "moving": int(moving),
                    "new": tid not in have,
                })
            vecs: Dict[str, bytes] = {}
            need = [p for p in pending if p["new"]]
            if need and self.embed_fn is not None:
                try:
                    raw = self.embed_fn([p["request"] for p in need])
                    for p, v in zip(need, raw):
                        vecs[p["tid"]] = _pack(v)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("experience sync: embedding failed (%s: %s) — new rows not indexed",
                                   type(exc).__name__, exc)
                    pending = [p for p in pending if not p["new"]]
            with self._conn() as c:
                for p in pending:
                    if p["new"]:
                        if p["tid"] not in vecs:
                            continue
                        c.execute(
                            "INSERT OR REPLACE INTO experience(tid, req_id, ts, request, request_norm, kind, grade, label_source,"
                            " reason, steps, sources, failed_sources, answer, moving, vec, labelled_at, derived_v)"
                            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (p["tid"], p["req_id"], p["ts"], p["request"], p["request_norm"], p["kind"], p["grade"],
                             p["label_source"], p["reason"], json.dumps(p["steps"]), json.dumps(p["sources"]),
                             json.dumps(p["failed_sources"]), p["answer"], p["moving"], vecs[p["tid"]], _now(),
                             DERIVED_VERSION))
                        counts["added"] += 1
                    else:
                        c.execute(
                            "UPDATE experience SET grade=?, label_source=?, reason=?, kind=?, steps=?, sources=?,"
                            " failed_sources=?, answer=?, moving=?, request_norm=?, labelled_at=?, derived_v=?"
                            " WHERE tid=?",
                            (p["grade"], p["label_source"], p["reason"], p["kind"], json.dumps(p["steps"]),
                             json.dumps(p["sources"]), json.dumps(p["failed_sources"]), p["answer"], p["moving"],
                             p["request_norm"], _now(), DERIVED_VERSION, p["tid"]))
                        counts["updated"] += 1
                # only a trajectory this walk SAW and found unlabelled is gone: one the
                # walk did not reach (an archived day, a narrower window) keeps its row
                gone = [t for t in have if t in seen and t not in keep]
                if gone:
                    c.executemany("DELETE FROM experience WHERE tid = ?", [(t,) for t in gone])
                    counts["removed"] = len(gone)
            self._cache = None
        return counts

    # -- reads ----------------------------------------------------------------

    def count(self) -> int:
        with self._lock, self._conn() as c:
            return int(c.execute("SELECT COUNT(*) FROM experience").fetchone()[0])

    def _stamp(self) -> Tuple[float, int]:
        try:
            st = os.stat(self.path)
            return (st.st_mtime, st.st_size)
        except OSError:
            return (0.0, 0)

    def _rows(self) -> List[dict]:
        """The rows, cached per process and re-read when the file changed on
        disk — the operator's sync script and the agent's idle maintenance
        both write it, and a hold-out set for a replay must take effect on
        the running agent without a restart."""
        with self._lock:
            stamp = self._stamp()
            if self._cache is None or stamp != getattr(self, "_cache_stamp", None):
                self._cache = None
                self._cache_stamp = stamp
            if self._cache is None:
                with self._conn() as c:
                    rows = []
                    for r in c.execute("SELECT * FROM experience"):
                        d = dict(r)
                        d["vec"] = _unpack(d.get("vec"))
                        for k in ("steps", "sources", "failed_sources"):
                            try:
                                d[k] = json.loads(d.get(k) or "[]")
                            except Exception:  # noqa: BLE001
                                d[k] = []
                        rows.append(d)
                self._cache = rows
            return list(self._cache)

    def similar(self, request: str, k: int = 4, exclude_tid: str = "") -> List[dict]:
        """Stored cases like ``request``: identical wording first (sim 1.0),
        then cosine ≥ ``SIMILARITY_FLOOR``. A request under the content-word
        floor matches nothing. Each row carries ``sim``."""
        request = str(request or "").strip()
        if len(content_words(request)) < MIN_CONTENT_WORDS:
            return []
        rows = [r for r in self._rows() if r["tid"] != str(exclude_tid or "")]
        if not rows:
            return []
        norm = normalize_request(request)
        qv = None
        if self.embed_fn is not None:
            try:
                qv = _unit(list(self.embed_fn([request])[0]))
            except Exception as exc:  # noqa: BLE001
                logger.debug("experience: query embedding failed: %s", exc)
                qv = None
        scored: List[dict] = []
        for r in rows:
            if r["request_norm"] == norm:
                sim = 1.0
            elif qv is not None and r.get("vec"):
                if len(r["vec"]) != len(qv) and not getattr(self, "_dim_warned", False):
                    self._dim_warned = True
                    logger.warning("experience: stored vectors are %d-d but the embedder gives %d-d — "
                                   "only identical wording matches until the index is rebuilt",
                                   len(r["vec"]), len(qv))
                sim = _cos(qv, r["vec"])
                if sim < SIMILARITY_FLOOR:
                    continue
            else:
                continue
            d = dict(r)
            d["sim"] = round(float(sim), 4)
            scored.append(d)
        scored.sort(key=lambda d: (-d["sim"], -float(d.get("ts") or 0)))
        return scored[:max(1, int(k))]

    # -- rendering ------------------------------------------------------------

    @staticmethod
    def select(rows: List[dict]) -> List[dict]:
        """At most one RIGHT case and one bad case (WRONG > PARTIAL >
        FAILED-HONEST, by similarity then recency), the RIGHT one first."""
        good = sorted([r for r in rows if r.get("grade") in GOOD_GRADES],
                      key=lambda r: (-float(r.get("sim") or 0), -float(r.get("ts") or 0)))
        bad_rank = {"WRONG": 0, "PARTIAL": 1, "FAILED-HONEST": 2}
        bad = sorted([r for r in rows if r.get("grade") in bad_rank],
                     key=lambda r: (-float(r.get("sim") or 0), bad_rank[r["grade"]], -float(r.get("ts") or 0)))
        out = good[:1] + bad[:1]
        return out[:MAX_CASES]

    @staticmethod
    def render_case(r: dict) -> str:
        when = _day(r.get("ts"))
        q = str(r.get("request") or "").replace("\n", " ")[:120]
        grade = str(r.get("grade") or "")
        steps = r.get("steps") or []
        did = " → ".join(steps[:8]) if steps else "no tool was called"
        line = f"- {when}, asked \"{q}\" — graded {grade}"
        if grade in GOOD_GRADES:
            line += f". It worked by: {did}."
            if r.get("sources"):
                line += " Sources that answered it: " + ", ".join(r["sources"][:4]) + "."
            if r.get("answer"):
                line += f" Answer then (may have changed since): \"{r['answer']}\""
        else:
            why = str(r.get("reason") or "").strip().rstrip(".;,")
            if r.get("moving"):
                why = mask_figures(why)
            if grade == "FAILED-HONEST":
                line += f": it could not be done{(' — ' + why) if why else ''}. It tried: {did}."
            else:
                line += f"{(': ' + why) if why else ''}. It did: {did}. Do not repeat that."
            if r.get("failed_sources"):
                line += " Sites that failed then: " + "; ".join(r["failed_sources"][:3]) + "."
            elif grade != "FAILED-HONEST" and r.get("sources"):
                line += " Pages it read: " + ", ".join(r["sources"][:3]) + "."
        return line[:MAX_LINE_CHARS]

    def lines_for(self, request: str, k: int = 4, exclude_tid: str = "") -> List[str]:
        return [self.render_case(r) for r in self.select(self.similar(request, k=k, exclude_tid=exclude_tid))]


# ── the one sync entry point (idle maintenance + the operator script) ──────

def sync_experience_index(context=None, *, store: Optional["ExperienceMemory"] = None,
                          system_dir=None) -> Dict[str, int]:
    """Rebuild the index from the trajectory collector and the label files.
    From a ``context``: its ``experience_memory`` and ``memory_dir``'s parent
    (``$GHOST_HOME/system``). Returns the sync counts, or ``{}`` when there is
    no store or no system dir (a ``--no-memory`` boot, a mocked context)."""
    store = store or (getattr(context, "experience_memory", None) if context is not None else None)
    if store is None or not hasattr(store, "sync"):
        return {}
    if system_dir is None and context is not None:
        mdir = getattr(context, "memory_dir", None)
        if isinstance(mdir, (str, Path)) and Path(mdir).is_dir():
            system_dir = Path(mdir).parent
    if system_dir is None or not Path(system_dir).is_dir():
        return {}
    from ..distill.collector import TrajectoryCollector
    from .population import iter_teachable
    col = TrajectoryCollector(root=Path(system_dir) / "trajectories", enabled=False)
    labels = load_labels(system_dir)
    # the ONE filter every learner reads the corpus through (members, probes,
    # synthetic rows out) — `sync` re-checks the owner/delivered rule per row
    # the whole corpus, archived days included: with `since_days` the collector skips
    # `archive/`, and a labelled turn whose day was archived would read as unlabelled
    return store.sync(iter_teachable(col.iter_trajectories(include_archive=True), consumer="experience"), labels)


# ── small helpers ───────────────────────────────────────────────────────────

def _now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _day(ts) -> str:
    try:
        return _dt.datetime.utcfromtimestamp(float(ts)).strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001
        return "undated"


def _epoch(ts) -> float:
    if isinstance(ts, (int, float)):
        return float(ts)
    s = str(ts or "").strip().replace("Z", "+00:00")
    try:
        d = _dt.datetime.fromisoformat(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=_dt.timezone.utc)
        return d.timestamp()
    except Exception:  # noqa: BLE001
        return 0.0


def _unit(v: List[float]) -> List[float]:
    n = math.sqrt(sum(float(x) * float(x) for x in v)) or 1.0
    return [float(x) / n for x in v]


def _cos(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)))


def _pack(v) -> bytes:
    import struct
    u = _unit([float(x) for x in v])
    return struct.pack(f"<{len(u)}f", *u)


def _unpack(blob) -> List[float]:
    import struct
    if not blob:
        return []
    n = len(blob) // 4
    try:
        return list(struct.unpack(f"<{n}f", bytes(blob)[: n * 4]))
    except Exception:  # noqa: BLE001
        return []
