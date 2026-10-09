from ..utils.json_store import open_append  # torn-tail-safe JSONL appends (§4MF)
import sqlite3
import threading
import difflib
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Iterable, Optional, Tuple

import networkx as nx

logger = logging.getLogger("GhostAgent")


@lru_cache(maxsize=512)
def _entity_pattern(target: str) -> "re.Pattern":
    """Whole-word/whole-phrase matcher for an entity name.

    ``target`` must match as a complete token (or token sequence) inside the
    node name — optionally pluralised — never as a bare substring:
    ``lois`` matches ``lois lane`` and ``lois's dog``, ``game`` matches
    ``games`` and ``chess game``, but ``tin`` does NOT match ``testing`` and
    ``game`` does NOT match ``gamedev``. ``[^\\W_]`` is "unicode alphanumeric,
    underscore excluded", so ``chess_game`` still matches ``game``.
    """
    return re.compile(r"(?<![^\W_])" + re.escape(target) + r"s?(?![^\W_])")



def _fold(text) -> str:
    """Case- and accent-folded text (Greek "Φωτεινή" ~ "φωτεινη")."""
    import unicodedata
    t = unicodedata.normalize("NFD", str(text or "").casefold())
    return "".join(ch for ch in t if not unicodedata.combining(ch))

#: world-fact predicates whose value is a moving target (§4ME F3, moved here
#: §4MM so every writer and the one-off cleanup share it). Whole
#: `_`-separated words: "RELEASED_IN 2008" is a stable fact.
TRANSIENT_WORLD_PREDICATE = re.compile(
    r"(?:^|_)(?:VERSION|LATEST|CURRENT|NEWEST|PRICE|COSTS?|STOCK|RATE|SCORE|RANK|RANKING)(?:_|$)",
    re.IGNORECASE)


#: §4MR: the predicate's WORDS were the only signal — "postgresql HAS_RELEASE
#: 18.4", "bitcoin TRADES_AT $61,000" were stored. The OBJECT's shape says it
#: too: a version number under a release-ish predicate, or a money amount.
_VERSION_OBJECT_RE = re.compile(r"^v?\d+(?:\.\d+){1,3}[a-z0-9.+-]*$", re.IGNORECASE)
_RELEASEISH_PREDICATE_RE = re.compile(r"(?:^|_)(?:RELEASES?|VERSIONS?|BUILDS?|EDITION)(?:_|$)", re.IGNORECASE)
_MONEY_OBJECT_RE = re.compile(
    r"^[$€£¥]\s?\d|\d[\d,.]*\s?(?:usd|eur|gbp|€|\$|£|dollars?|euros?|btc|ευρώ|ευρω)\b", re.IGNORECASE)


def is_moving_target_world_fact(predicate, obj) -> bool:
    """A world fact whose value moves — by its predicate's words, or by its
    object's shape (a version under a release predicate, an amount of money).
    "RELEASED_IN 2008" stays: a year is not a version."""
    p, o = str(predicate or ""), str(obj or "").strip()
    if TRANSIENT_WORLD_PREDICATE.search(p):
        return True
    if _VERSION_OBJECT_RE.match(o) and _RELEASEISH_PREDICATE_RE.search(p):
        return True
    return bool(_MONEY_OBJECT_RE.search(o))


class GraphMemory:
    """Knowledge graph with SQLite persistence + in-memory NetworkX routing.

    SQLite remains the source of truth on disk. A `nx.MultiDiGraph` mirror is
    held in memory and used for spreading-activation traversal in
    `get_neighborhood`. Both stores are kept in sync by `add_triplets`,
    `delete_by_target`, `wipe_all`, and `execute_graph_compression`.
    """

    #: Predicates that are SINGLE-VALUED: a new object supersedes the old one,
    #: which gets a `valid_until` stamp instead of accumulating a contradiction.
    #: Every entry here MUST be one-to-one *for any subject* — a multi-valued
    #: predicate in this set silently expires real knowledge on every new
    #: extraction. `OWNS`/`IS` used to live here and destroyed 19 of the
    #: operator's ownership facts (you own many things; X IS many things);
    #: likewise `HAS_PET`, `HAS_TASK`, `HAS_FEATURE`, `HAS_NAME` (written with
    #: the generic subject `project`) and `HAS_STATE`/`HAS_FEN` (position logs)
    #: are deliberately absent. Comparison is case-insensitive (uppercased).
    _FUNCTIONAL_PREDICATES = {
        # Biographical
        # (§4KZ: LOCATED_IN is PRESENCE — a weekend in Kyllini expired the
        # owner's home; home is LIVES_IN, and a birth date is one date)
        "WORKS_AT", "LIVES_IN", "DRIVES", "MARRIED_TO", "HAS_BIRTHDATE",
        "BORN_IN", "STUDIES_AT", "EMPLOYED_BY",
        "HAS_AGE", "HAS_LOCATION",
        # §4KY: a new profession REPLACES the old one ("I'm a DBA" left
        # "doctor" live beside it)
        "HAS_PROFESSION", "HAS_OCCUPATION", "WORKS_AS", "HAS_JOB",
        # Operational — written by the agent itself, single-valued by
        # construction (a process has one status/one pid at a time).
        "HAS_STATUS", "STATUS", "HAS_PID",
    }

    #: Subjects too GENERIC for functional expiry to be safe. The extractor
    #: routinely writes per-entity facts under aggregate nouns (live rows:
    #: `project HAS_STATUS done/active/needs_user` — three DIFFERENT
    #: projects) — expiring "all other objects of (project, HAS_STATUS)"
    #: would erase other projects' real statuses on every write. Same
    #: rationale that kept HAS_NAME out of _FUNCTIONAL_PREDICATES entirely;
    #: these subjects just make ANY functional predicate unsafe.
    _EXPIRY_GENERIC_SUBJECTS = {
        "project", "task", "app", "service", "process", "system",
        "it", "this", "that",
    }

    def __init__(self, memory_dir: Path):
        self.db_path = memory_dir / "knowledge_graph.db"
        self._lock = threading.RLock()
        self.nx_graph: nx.MultiDiGraph = nx.MultiDiGraph()
        # Cached node-name list for _map_words_to_seeds. Rebuilding
        # list(nx_graph.nodes()) on EVERY query word was O(nodes) per turn
        # (the graph is the only uncapped memory tier); this snapshots it and
        # invalidates on any edge mutation.
        self._node_list_cache: Optional[List[str]] = None
        self._init_db()
        self.initialize_graph()

    def _invalidate_node_cache(self):
        self._node_list_cache = None

    def _nodes_snapshot(self) -> List[str]:
        if self._node_list_cache is None:
            self._node_list_cache = list(self.nx_graph.nodes())
        return self._node_list_cache

    # ------------------------------------------------------------------ setup

    def _init_db(self):
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS triplets (
                        subject TEXT,
                        predicate TEXT,
                        object TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(subject, predicate, object)
                    )
                ''')
                try:
                    conn.execute('ALTER TABLE triplets ADD COLUMN weight INTEGER DEFAULT 1')
                except Exception:
                    pass
                # Temporal columns: valid_from marks when the fact became true,
                # valid_until marks when it was superseded (NULL = still current).
                try:
                    conn.execute('ALTER TABLE triplets ADD COLUMN valid_from REAL')
                except Exception:
                    pass
                try:
                    conn.execute('ALTER TABLE triplets ADD COLUMN valid_until REAL')
                except Exception:
                    pass
                conn.execute('CREATE INDEX IF NOT EXISTS idx_subj ON triplets(subject)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_obj ON triplets(object)')
                conn.commit()

    def initialize_graph(self, include_expired: bool = False):
        """Hydrate the in-memory NetworkX graph from the SQLite triplets table.

        By default only loads temporally-valid edges (valid_until IS NULL).
        Pass ``include_expired=True`` to load the full history."""
        with self._lock:
            self.nx_graph = nx.MultiDiGraph()
            self._invalidate_node_cache()
            with sqlite3.connect(self.db_path) as conn:
                if include_expired:
                    query = 'SELECT subject, predicate, object, weight FROM triplets'
                else:
                    query = 'SELECT subject, predicate, object, weight FROM triplets WHERE valid_until IS NULL'
                cursor = conn.execute(query)
                for s, p, o, w in cursor:
                    self._upsert_edge(s, p, o, int(w or 1))

    # ----------------------------------------------------------- graph mirror

    def _upsert_edge(self, s: str, p: str, o: str, weight: int):
        """Add or reinforce a single edge in the in-memory graph."""
        existing_key = None
        if self.nx_graph.has_edge(s, o):
            for k, data in self.nx_graph[s][o].items():
                if data.get("predicate") == p:
                    existing_key = k
                    break
        if existing_key is not None:
            self.nx_graph[s][o][existing_key]["weight"] = weight
        else:
            self.nx_graph.add_edge(s, o, predicate=p, weight=weight)
            self._invalidate_node_cache()  # new nodes may have appeared

    def _remove_edge(self, s: str, p: str, o: str):
        if not self.nx_graph.has_edge(s, o):
            return
        kill = [k for k, data in self.nx_graph[s][o].items()
                if data.get("predicate") == p]
        for k in kill:
            self.nx_graph.remove_edge(s, o, key=k)
        for n in (s, o):
            if n in self.nx_graph and self.nx_graph.degree(n) == 0:
                self.nx_graph.remove_node(n)
                self._invalidate_node_cache()

    # ------------------------------------------------------------ public CRUD

    #: §4KZ: ONE predicate per kind of fact. The live graph held the owner's
    #: spouse six ways and each son's birth date three ways; correcting one
    #: copy left the others live.
    _CANONICAL_PREDICATES = {
        "HAS_SPOUSE": "MARRIED_TO", "HAS_WIFE": "MARRIED_TO", "HAS_HUSBAND": "MARRIED_TO", "IS_MARRIED_TO": "MARRIED_TO",
        "HAS_BIRTH_DATE": "HAS_BIRTHDATE", "BORN_ON": "HAS_BIRTHDATE", "WAS_BORN_ON": "HAS_BIRTHDATE",
        "HAS_BIRTHDAY": "HAS_BIRTHDATE", "BIRTHDATE": "HAS_BIRTHDATE",
        "RESIDES_IN": "LIVES_IN", "HAS_HOME_IN": "LIVES_IN", "LIVES_AT": "LIVES_IN",
        "TAKES_MEDICATION": "HAS_MEDICATION", "HAS_SONS": "HAS_SON", "HAS_CHILDREN": "HAS_CHILD",
    }
    #: an AGE is a measurement that is wrong a year later — the birth date is
    #: the fact (live: "Leonidas AGE 5" from "5 months")
    _DECAYING_PREDICATES = {"AGE", "HAS_AGE", "IS_AGED", "AGED"}

    @classmethod
    def canonical_predicate(cls, predicate) -> str:
        p = str(predicate or "").upper().strip()
        return cls._CANONICAL_PREDICATES.get(p, p)

    def add_triplets(self, triplets: List[Dict[str, str]], as_of: float = None, raw: bool = False):
        """``as_of`` (epoch): WHEN the triplets were stated. A single-valued
        edge stated EARLIER than the live one does not replace it (§4KZ: a
        re-queued older journal item expired "lives in Patras" back to
        Athens). ``raw``: the predicate as given (the profile's own mirror
        edges, `sync_owner_field`)."""
        if not triplets:
            return 0
        added = 0
        import time as _time
        now = _time.time()
        stated = float(as_of) if as_of else now
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                for t in triplets:
                    # Shape guard BEFORE any attribute access: LLM extractors
                    # do emit list/tuple-shaped or `relation`-keyed triplets
                    # (the display helper `_tri` handles both defensively).
                    # A single malformed one used to raise AttributeError out
                    # of the whole batch — and at the smart-memory call site
                    # that aborted the turn's fact embed AND profile write,
                    # since graph ingestion runs first.
                    if not isinstance(t, dict):
                        logger.debug("add_triplets: skipped non-dict triplet %r",
                                     str(t)[:80])
                        continue
                    s = t.get("subject", "")
                    p = t.get("predicate", "") or t.get("relation", "")
                    o = t.get("object", "")
                    if not (s and p and o):
                        continue
                    sn = str(s).lower().strip()
                    pn = str(p).upper().strip() if raw else self.canonical_predicate(p)
                    on = str(o).lower().strip()
                    if not (sn and pn and on):
                        continue
                    if pn in self._DECAYING_PREDICATES and not raw:
                        continue
                    # §4MM: a world fact whose value moves (VERSION, LATEST,
                    # PRICE…) is never stored, at ANY writer: "postgresql
                    # HAS_VERSION 18.4" (July) sat beside 18.6, outranked it,
                    # and answered "latest version?" wrong twice. F3 (§4ME)
                    # gated one writer only. An owner fact is not a world fact.
                    if (not raw and is_moving_target_world_fact(pn, on)
                            and "user" not in (sn, on)            # r2: the owner's own statement
                            and not self._is_owner_fact(sn, pn, on)):
                        continue
                    try:
                        # Temporal conflict resolution: if the same subject+predicate
                        # exists with a DIFFERENT object, expire the old edge instead
                        # of accumulating contradictions. Only applies to "functional"
                        # predicates (one-to-one) like WORKS_AT, LIVES_IN, DRIVES —
                        # see `_FUNCTIONAL_PREDICATES`. Multi-valued predicates
                        # (LIKES, KNOWS, HAS_*, OWNS, IS) are excluded: expiring
                        # those is silent data loss, not an update.
                        conflicting = []
                        if (pn in self._FUNCTIONAL_PREDICATES
                                and sn not in self._EXPIRY_GENERIC_SUBJECTS):
                            conflicting = conn.execute(
                                '''SELECT object, valid_from FROM triplets
                                   WHERE subject = ? AND predicate = ? AND object != ?
                                   AND valid_until IS NULL''',
                                (sn, pn, on)
                            ).fetchall()
                            if any(float(vf or 0) > stated for _, vf in conflicting):
                                # a NEWER value is live: this older statement
                                # neither replaces it nor joins it
                                logger.info("graph: %s %s '%s' is older than the live value — not applied",
                                            sn, pn, on)
                                continue
                            conflicting = [(o_,) for o_, _ in conflicting]
                        if conflicting:
                            conn.execute(
                                '''UPDATE triplets SET valid_until = ?
                                   WHERE subject = ? AND predicate = ? AND object != ?
                                   AND valid_until IS NULL''',
                                (now, sn, pn, on)
                            )
                            # Expiry is destructive-by-retrieval (expired edges
                            # vanish from every read path), so it is NEVER silent.
                            for (old_obj,) in conflicting:
                                logger.warning(
                                    "graph expiry: %s %s '%s' superseded by '%s'",
                                    sn, pn, old_obj, on,
                                )
                            # Remove expired edges from the in-memory graph
                            for (old_obj,) in conflicting:
                                self._remove_edge(sn, pn, old_obj)

                        cursor = conn.execute(
                            '''INSERT INTO triplets (subject, predicate, object, weight, timestamp, valid_from)
                               VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP, ?)
                               ON CONFLICT(subject, predicate, object)
                               DO UPDATE SET weight = weight + 1, timestamp = CURRENT_TIMESTAMP, valid_until = NULL''',
                            (sn, pn, on, stated)
                        )
                        if cursor.rowcount > 0:
                            added += 1
                        # Read back the resulting weight so the mirror stays accurate
                        row = conn.execute(
                            'SELECT weight FROM triplets WHERE subject=? AND predicate=? AND object=?',
                            (sn, pn, on)
                        ).fetchone()
                        weight = int(row[0]) if row and row[0] is not None else 1
                        self._upsert_edge(sn, pn, on, weight)
                    except Exception as e:
                        # A dropped write deflates `added` — never silently:
                        # multi-process access is assumed elsewhere, so
                        # sqlite errors here are real signal.
                        logger.warning(
                            "add_triplets: dropped (%s %s %s): %s",
                            sn, pn, on, e)
                conn.commit()
        return added

    @staticmethod
    def _entity_matches(node: str, target: str) -> bool:
        """True when ``node`` names the entity ``target`` (already normalised).

        Whole-token match, NOT substring: `forget("tin")` must not reach
        `testing`/`printing`, and `forget("game")` must not reach `gamedev`.
        Multi-word names still match on any complete token run, so `lois`
        reaches `lois lane` and `mortimer` reaches `mortimer the iguana`.
        """
        n = (node or "").strip().lower()
        if not n or not target:
            return False
        if n == target:
            return True
        return _entity_pattern(target).search(n) is not None

    #: Blast-radius guard for `delete_by_target`. Whole-token matching alone
    #: does not bound a forget: `game` is a legitimate token of 108 rows (8%)
    #: of the live graph. A forget at or above this many rows is therefore
    #: downgraded to a temporal expiry — it disappears from every read path
    #: (mirror, neighborhood, recent triplets) exactly like a delete, but
    #: stays recoverable via `get_expired_triplets`. Small, surgical forgets
    #: keep the original hard-delete semantics.
    _FORGET_SOFT_EXPIRE_MIN_ROWS = 50

    #: §4M (Lens A MAJOR-3): archive-before-delete belongs to EVERY
    #: destructive path (the playbook learned this the hard way; the graph
    #: — the only uncapped tier and the July data-loss site — never got
    #: it: 884 live rows vs 1407 in the 07-22 backup, unattributable
    #: precisely because no archive existed).
    _ARCHIVE_FILENAME = "graph_pruned_archive.jsonl"

    def _archive_rows(self, reason: str, rows) -> bool:
        """Append doomed triplets to the JSONL archive BEFORE deletion.
        Rows are ``(subject, predicate, object[, weight[, timestamp]])``.
        Returns True only when every row was durably appended — callers
        on unattended paths fail closed (skip or soft-expire) on False."""
        import json as _json
        import os as _os
        import time as _time
        rows = list(rows or [])
        if not rows:
            return True
        try:
            path = Path(self.db_path).parent / self._ARCHIVE_FILENAME
            ts = _time.time()
            # §4MN: the validity window too — without it a restored EXPIRED
            # edge came back as current (best-effort: a row may be gone)
            _valid = {}
            try:
                with sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True) as _rc:
                    for row in rows:
                        _v = _rc.execute("SELECT valid_from, valid_until FROM triplets WHERE subject=? "
                                         "AND predicate=? AND object=?", (row[0], row[1], row[2])).fetchone()
                        if _v:
                            _valid[(row[0], row[1], row[2])] = _v
            except Exception:  # noqa: BLE001
                _valid = {}
            with open_append(path) as fh:
                for row in rows:
                    rec = {"archived_at": ts, "reason": reason,
                           "subject": row[0], "predicate": row[1],
                           "object": row[2]}
                    if len(row) > 3:
                        rec["weight"] = row[3]
                    if len(row) > 4:
                        rec["timestamp"] = row[4]
                    if (row[0], row[1], row[2]) in _valid:
                        rec["valid_from"], rec["valid_until"] = _valid[(row[0], row[1], row[2])]
                    fh.write(_json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                _os.fsync(fh.fileno())
            return True
        except Exception as e:
            logger.error("graph archive-before-delete failed (%s): %s",
                         reason, e)
            return False

    #: relation tokens of the owner's family
    _FAMILY_TOKENS = {"MARRIED", "SPOUSE", "WIFE", "HUSBAND", "PARTNER", "CHILD", "CHILDREN", "SON", "DAUGHTER",
                      "PARENT", "MOTHER", "FATHER", "SIBLING", "BROTHER", "SISTER", "FAMILY", "PET"}

    def is_owner_family(self, name: str) -> bool:
        """Is ``name`` a node the owner (``user``) is linked to by a family
        relation (either direction, current edges)?"""
        n = _fold(name).strip()
        if not n:
            return False
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    """SELECT subject, predicate, object FROM triplets WHERE valid_until IS NULL AND
                       (lower(subject)='user' OR lower(object)='user')""").fetchall()
        return any(set(str(p or "").upper().split("_")) & self._FAMILY_TOKENS
                   for s_, p, o_ in rows
                   if _fold(o_ if str(s_).strip().lower() == "user" else s_).strip() == n)

    def delete_edge(self, subject: str, predicate: str, obj: str) -> int:
        """Delete ONE current edge (archived first, like every graph delete).
        Used when `update_profile` deletes the field that minted it
        (profile-writes review: `user HAS_<KEY> <value>` outlived the
        field). Returns the number of rows removed."""
        s_, p_, o_ = str(subject).lower().strip(), str(predicate).strip(), str(obj).lower().strip()
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    """SELECT rowid, subject, predicate, object, COALESCE(weight, 1), timestamp FROM triplets
                       WHERE valid_until IS NULL AND predicate=? AND (subject=? OR lower(subject)=?)
                       AND (object=? OR lower(object)=?)""",
                    (p_, str(subject).strip(), s_, str(obj).strip(), o_)).fetchall()
                if not rows or not self._archive_rows("delete_edge", [r[1:] for r in rows]):
                    return 0
                conn.executemany("DELETE FROM triplets WHERE rowid = ?", [(r[0],) for r in rows])
                conn.commit()
                rows = [r[1:] for r in rows]
            for r in rows:
                self._remove_edge(r[0], r[1], r[2])
        return len(rows)

    @staticmethod
    def _node_is(node: str, entity: str) -> bool:
        """The node IS the entity, or a short name starting with it ("tesla
        model 3" for "tesla") — not a phrase or list that mentions it
        ("thrakomakedones near athens", "athens, greece")."""
        # folded both sides (re-review: "Φωτεινή" never matched "φωτεινη")
        # …and hyphen/underscore ~ space ("pista-gp" names "pista gp")
        n = " ".join(re.sub(r"[-_]+", " ", _fold(node)).split()).rstrip(".,;:!?")
        e = " ".join(re.sub(r"[-_]+", " ", _fold(entity)).split()).rstrip(".,;:!?")
        if not n or not e:
            return False
        if n == e:
            return True
        if not n.startswith(e + " ") or re.search(r"[,;/&]|\band\b|\bnear\b|\bof\b|\bin\b", n):
            return False
        # a VERSION or MODEL after the name ("tesla model 3", "postgresql
        # 17"), never a description ("postgresql services company")
        rest = n[len(e):].split()
        return len(rest) <= 3 and any(ch.isdigit() for ch in "".join(rest))

    @staticmethod
    def _entity_candidates(conn, e: str) -> list:
        """Live rows that MAY name ``e`` (rowid, s, p, o, weight, ts).
        sqlite's LIKE/lower() fold ASCII only, so every live row is read and
        the folded `_node_is` decides."""
        cols = "rowid, subject, predicate, object, COALESCE(weight, 1), timestamp"
        # every live row: an ASCII entity ("rene lacoste") must still find a
        # stored "rené lacoste" (r8 review) — 6 ms on the live 1k-edge graph
        return conn.execute(f"SELECT {cols} FROM triplets WHERE valid_until IS NULL").fetchall()

    def owner_field_edges(self, entity: str) -> list:
        """Live `user HAS_<…>` edges whose PREDICATE names ``entity`` — the
        graph twins of profile fields like `fotini_description` (r8 review:
        they outlived a family forget and were not even listed)."""
        words = [w for w in re.split(r"[\s_\-]+", _fold(entity)) if w]
        if not words:
            return []
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute("SELECT subject, predicate, object FROM triplets WHERE valid_until IS NULL "
                                    "AND lower(subject) = 'user'").fetchall()
        return [r for r in rows if set(words) <= set(_fold(r[1]).split("_"))]

    #: a CATEGORY word in a question and the predicate words it covers
    #: ("what health conditions do I have" → HAS_CONDITION, TAKES_MEDICATION)
    _CATEGORY_WORDS = {
        "health": {"condition", "conditions", "medication", "medications", "diagnosis", "diagnosed", "allergy",
                   "allergies", "allergic", "disease", "health", "illness", "takes"},
        "medical": {"condition", "medication", "diagnosis", "diagnosed", "allergy", "allergic", "disease", "health"},
        "medicine": {"medication", "medications", "takes"}, "medicines": {"medication", "medications", "takes"},
        "meds": {"medication", "medications", "takes"}, "drugs": {"medication", "medications"},
        "pills": {"medication", "medications"}, "illness": {"condition", "disease", "illness", "diagnosis"},
        "sick": {"condition", "disease", "illness"}, "allergic": {"allergy", "allergies", "allergic"},
        "job": {"profession", "occupation", "works", "employed", "employer", "job"},
        "work": {"profession", "occupation", "works", "employed", "employer", "work"},
        "profession": {"profession", "occupation"}, "occupation": {"profession", "occupation"},
        "family": {"married", "spouse", "wife", "husband", "son", "sons", "daughter", "daughters", "child",
                   "children", "parent", "parents", "mother", "father", "sibling", "brother", "sister", "companion"},
        "partner": {"married", "spouse", "wife", "husband", "partner", "companion"},
        "wife": {"married", "spouse", "wife"}, "husband": {"married", "spouse", "husband"},
        "spouse": {"married", "spouse", "wife", "husband"},
        "sons": {"son", "sons", "child", "children"}, "son": {"son", "sons", "child", "children"},
        "daughter": {"daughter", "daughters", "child", "children"},
        "birthdays": {"birth", "birthday", "birthdate", "born"},
        "kids": {"son", "sons", "daughter", "daughters", "child", "children"},
        "children": {"son", "sons", "daughter", "daughters", "child", "children"},
        "live": {"lives", "resides", "home", "address", "located"}, "home": {"lives", "resides", "home", "address"},
        "address": {"address", "lives", "resides", "home"},
        "own": {"owns", "owned", "own", "has"}, "car": {"car", "vehicle", "owns", "drives"},
        "birthday": {"birth", "birthday", "birthdate", "born"},
        # Greek (folded)
        "υγεια": {"condition", "conditions", "medication", "medications", "diagnosis", "allergy", "disease", "health"},
        "φαρμακα": {"medication", "medications", "takes"}, "φαρμακο": {"medication", "medications", "takes"},
        "δουλεια": {"profession", "occupation", "works", "employed"}, "οικογενεια": {"married", "son", "sons",
                                                                                   "daughter", "child", "children"},
    }
    _INFL = ("", "s", "es", "ed", "ing", "ies")

    @classmethod
    def _word_names_predicate_token(cls, w: str, tok: str) -> bool:
        if w == tok:
            return True
        k = 0
        for a, b in zip(w, tok):
            if a != b:
                break
            k += 1
        return k >= 4 and w[k:] in cls._INFL and tok[k:] in cls._INFL

    def owner_facts_matching(self, query: str, limit: int = 20) -> List[str]:
        """The owner's facts (`user …` edges) whose PREDICATE names a word of
        the query or a category it covers — "what health conditions do I have"
        finds `user HAS_CONDITION heart failure`. The neighbourhood lookup
        seeds on NODE names, so a fact asked for by its kind was unreachable
        (§4KX r8 probe: heart failure and Entresto were stored and never
        recalled)."""
        words = [w for w in re.findall(r"\w+", _fold(query)) if len(w) >= 3]
        if not words:
            return []
        # common verbs of a REQUEST are not the name of a fact (§4LB r2:
        # "do you know" matched every `User KNOWS …`, "show me" REQUESTED)
        stop = {"has", "have", "the", "and", "what", "who", "which", "user", "you", "about",
                "know", "knows", "show", "tell", "view", "see", "find", "give", "look", "want",
                "need", "request", "requested", "ask", "asked", "make", "get", "help", "use", "can"}
        asked = set(words) - stop
        implied = set()
        for w in words:
            implied |= self._CATEGORY_WORDS.get(w, set())
        implied -= stop
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                # NEWEST first, and the date is shown (§4KZ: by weight, a
                # stale "located at kyllini" outranked the current home and
                # the model could not tell which was newer)
                rows = conn.execute(
                    "SELECT subject, predicate, object, COALESCE(weight, 1), timestamp FROM triplets "
                    "WHERE valid_until IS NULL AND (lower(subject) = 'user' OR lower(object) = 'user') "
                    "ORDER BY timestamp DESC").fetchall()
        out = []
        for s_, p_, o_, _w, _ts in rows:
            raw = _fold(p_).split("_")
            toks = [t for t in raw if t and t not in ("has", "is", "of", "to", "at", "in", "on")]
            hit = any(self._word_names_predicate_token(w, t) for w in asked for t in toks)
            # a category reaches only a fact ABOUT the owner, not a task in
            # progress (`user WORKS_ON pinball.html` is not a job)
            if not hit and "on" not in raw and "working" not in raw:
                hit = any(self._word_names_predicate_token(w, t) for w in implied for t in toks)
            if hit:
                out.append(self._format_path(((s_, p_, o_),), 1) + (f" (as of {str(_ts)[:10]})" if _ts else ""))
                if len(out) >= limit:
                    break
        return out

    def owner_edges_naming(self, value: str, predicate: str = None) -> list:
        """Live edges with `user` at one end whose OTHER end IS ``value``
        (folded) — what an owner's correction ("I'm not a doctor") removes.
        With ``predicate``, only that kind of fact (§4KZ: "I'm not going to
        Athens" deleted LIVES_IN, BORN_IN and WORKS_IN athens)."""
        v = str(value or "").strip()
        _p = self.canonical_predicate(predicate) if predicate else None
        if len(v) < 2:
            return []
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute("SELECT subject, predicate, object FROM triplets WHERE valid_until IS NULL "
                                    "AND (lower(subject) = 'user' OR lower(object) = 'user')").fetchall()
        return [r for r in rows
                if self._node_is(r[2] if str(r[0]).strip().lower() == "user" else r[0], v)
                and (_p is None or self.canonical_predicate(r[1]) == _p)]

    def sync_owner_field(self, key: str, values) -> tuple:
        """Make the owner's ``user HAS_<KEY>`` edges EQUAL ``values`` (the
        profile field, now): stale ones deleted (archived), missing ones
        added. §4KZ: a profile change left the old HAS_WIFE/HAS_EMPLOYER/
        HAS_LOCATION edges live beside the new. Returns (added, removed)."""
        pred = "HAS_" + str(key or "").upper().replace(" ", "_")
        want = {str(v).lower().strip() for v in (values if isinstance(values, list) else [values])
                if v not in (None, "") and str(v).strip()}
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                live = [r[0] for r in conn.execute(
                    "SELECT object FROM triplets WHERE valid_until IS NULL AND lower(subject) = 'user' "
                    "AND predicate = ?", (pred,)).fetchall()]
        removed = sum(self.delete_edge("user", pred, o) for o in live if str(o).lower().strip() not in want)
        added = self.add_triplets([{"subject": "user", "predicate": pred, "object": v}
                                   for v in want if v not in {str(o).lower().strip() for o in live}], raw=True)
        return added, removed

    def count_edges(self) -> int:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                return conn.execute("SELECT COUNT(*) FROM triplets WHERE valid_until IS NULL").fetchone()[0] or 0

    def preview_forget_entity(self, entity: str) -> tuple:
        """What `forget_entity` WOULD do, deleting nothing:
        ``(doomed_edges, kept_owner_facts)`` as (subject, predicate, object)."""
        e = str(entity or "").strip().lower()
        if len(e) < 3:
            return [], []
        family = self.is_owner_family(e)
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                rows = [r[1:4] for r in self._entity_candidates(conn, e)]
        hits = [r for r in rows if self._node_is(r[0], e) or self._node_is(r[2], e)]
        kept = [r for r in hits if self._is_owner_life_fact(*r) and not family]
        return [r for r in hits if r not in kept], kept

    @classmethod
    def _is_owner_life_fact(cls, subject, predicate, obj) -> bool:
        """A durable fact ABOUT THE OWNER (one end is `user`). `forget`
        keeps these unless asked by name; another person's facts are not
        the owner's (re-review: Ektoras Koufontinas's parents could not be
        forgotten and were reported as "facts about you")."""
        if "user" not in (str(subject or "").strip().lower(), str(obj or "").strip().lower()):
            return False
        return cls._is_owner_fact(subject, predicate, obj)

    def forget_entity(self, entity: str) -> tuple:
        """`forget`'s graph leg (third review): delete the edges where a node
        IS ``entity`` (archived, as every graph delete). An OWNER fact (see
        `_is_owner_fact`) is deleted only when its other end is the owner's
        family member being forgotten; any other owner fact is returned as
        kept, for the report. Returns ``(deleted, kept_owner_facts)``."""
        e = str(entity or "").strip().lower()
        if len(e) < 3:
            return 0, []
        family = self.is_owner_family(e)
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                rows = self._entity_candidates(conn, e)
                hits = [r for r in rows if self._node_is(r[1], e) or self._node_is(r[3], e)]
                kept = [r for r in hits if self._is_owner_life_fact(r[1], r[2], r[3]) and not family]
                doomed = [r for r in hits if r not in kept]
                if doomed:
                    if not self._archive_rows("forget_entity", [(r[1], r[2], r[3], r[4], r[5]) for r in doomed]):
                        return 0, [(r[1], r[2], r[3]) for r in kept]
                    conn.executemany("DELETE FROM triplets WHERE rowid = ?", [(r[0],) for r in doomed])
                    conn.commit()
            for r in doomed:
                self._remove_edge(r[1], r[2], r[3])
        if doomed:
            self._invalidate_node_cache()
        return len(doomed), [(r[1], r[2], r[3]) for r in kept]

    #: generic subjects the extractor files per-project facts under
    #: (`project HAS_ID 7b62e5e533d1`, `project HAS_TITLE <title>`)
    _PROJECT_GENERIC_NODES = {"project", "projects", "the project"}

    @staticmethod
    def project_title_key(title) -> str:
        """The one title comparison `forget_project` and its caller share:
        folded (case + accents), hyphen/underscore as space, whitespace
        collapsed. Review: the caller's `.lower()` check and this `_fold`
        disagreed on "Café"/"Cafe", so a twin's title edge was deleted."""
        return " ".join(re.sub(r"[-_]+", " ", _fold(title)).split())

    def forget_project(self, project_id: str, title: str = "",
                       forget_title: bool = True) -> int:
        """A hard-deleted project's graph leg: delete every edge (current AND
        expired, archived first) that names the project. Live 2cb40b10: after
        the user deleted projects, `user RESUMES project 7b62e5e533d1` and
        `project 7b62e5e533d1 TESTED 4-layer recursive cascade` kept telling
        the model "the user runs this experiment as a project", and it built
        one again.

        Matches a node holding ``project_id`` as a whole token (the tool's
        `project:<id>`, the extractor's `project <id>`, a bare id under
        `project HAS_ID`) and the `task:<tid>` nodes the project owned. When
        ``forget_title`` (the caller passes False while another project still
        has that title) it also matches the generic `project HAS_TITLE <title>`
        edge and the node NAMED by the title — the extractor's main shape
        (`ai self awareness exploration TESTED self-awareness emergence`
        survived the first version). The title node is matched only for a
        title of two or more words: a one-word title ("Chess") is also a
        topic the owner talks about. On that node an owner life fact (see
        `_is_owner_life_fact`) is kept. Shared concept nodes
        (`technique:bayesian`) lose only this project's edge. If the archive
        write fails the rows are soft-expired (recoverable), as
        `delete_by_target` does. Returns the number of rows removed."""
        pid = str(project_id or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{8,}", pid):
            return 0
        id_re = re.compile(rf"(?<![0-9a-z]){re.escape(pid)}(?![0-9a-z])")
        key = self.project_title_key
        title_k = key(title) if forget_title else ""
        node_k = title_k if len(title_k.split()) >= 2 else ""
        cols = "rowid, subject, predicate, object, COALESCE(weight, 1), timestamp"
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(f"SELECT {cols} FROM triplets").fetchall()
                names = lambda r: id_re.search(_fold(r[1])) or id_re.search(_fold(r[3]))
                own = [r for r in rows if names(r)]
                tasks = {_fold(r[3]).strip() for r in own
                         if str(r[2]).upper() == "HAS_TASK"
                         and _fold(r[3]).strip().startswith("task:")}

                def titled(r):
                    if not title_k:
                        return False
                    if (str(r[2]).upper() == "HAS_TITLE"
                            and _fold(r[1]).strip() in self._PROJECT_GENERIC_NODES
                            and key(r[3]) == title_k):
                        return True
                    return bool(node_k) and node_k in (key(r[1]), key(r[3])) \
                        and not self._is_owner_life_fact(r[1], r[2], r[3])

                doomed = [r for r in rows if names(r)
                          or _fold(r[1]).strip() in tasks or _fold(r[3]).strip() in tasks
                          or titled(r)]
                if not doomed:
                    return 0
                if self._archive_rows("forget_project", [r[1:] for r in doomed]):
                    conn.executemany("DELETE FROM triplets WHERE rowid = ?", [(r[0],) for r in doomed])
                else:
                    import time as _time
                    logger.warning("graph forget_project %s: archive failed — soft-expiring "
                                   "%d row(s) instead of deleting", pid, len(doomed))
                    now = _time.time()
                    conn.executemany(
                        "UPDATE triplets SET valid_until = COALESCE(valid_until, ?) WHERE rowid = ?",
                        [(now, r[0]) for r in doomed])
                conn.commit()
            for r in doomed:
                self._remove_edge(r[1], r[2], r[3])
        self._invalidate_node_cache()
        return len(doomed)

    def delete_by_target(self, target: str) -> int:
        if not target or len(target.strip()) < 3:
            return 0
        t_norm = target.lower().strip()
        like = f"%{t_norm}%"
        import time as _time
        now = _time.time()
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                # LIKE is only a cheap prefilter — the authoritative test is
                # `_entity_matches`, which requires a whole-token hit. The old
                # code deleted on the raw LIKE, so `forget("tin")` hard-deleted
                # 83 unrelated rows of the production graph with no undo.
                candidates = conn.execute(
                    '''SELECT rowid, subject, predicate, object,
                              COALESCE(weight, 1), timestamp FROM triplets
                       WHERE subject LIKE ? OR object LIKE ?''',
                    (like, like)
                ).fetchall()
                doomed = [
                    row for row in candidates
                    if self._entity_matches(row[1], t_norm)
                    or self._entity_matches(row[3], t_norm)
                ]
                if not doomed:
                    return 0
                live_rows = conn.execute(
                    'SELECT COUNT(*) FROM triplets WHERE valid_until IS NULL'
                ).fetchone()[0] or 0
                if len(doomed) >= self._FORGET_SOFT_EXPIRE_MIN_ROWS:
                    logger.warning(
                        "graph forget '%s': %d/%d live rows matched — EXPIRING "
                        "instead of deleting (recoverable via get_expired_triplets)",
                        t_norm, len(doomed), live_rows,
                    )
                    conn.executemany(
                        'UPDATE triplets SET valid_until = ? WHERE rowid = ?',
                        [(now, row[0]) for row in doomed],
                    )
                elif self._archive_rows(
                        "delete_by_target",
                        # §4M R2 NIT-2: full 5-tuples like prune/wipe — a
                        # forget-archive row without weight/age loses what
                        # recovery needs.
                        [(row[1], row[2], row[3], row[4], row[5])
                         for row in doomed]):
                    conn.executemany(
                        'DELETE FROM triplets WHERE rowid = ?',
                        [(row[0],) for row in doomed],
                    )
                else:
                    # Archive failed → the forget still honors the user's
                    # intent, but RECOVERABLY: soft-expire instead of
                    # hard-delete (same shape as the big-batch guard).
                    logger.warning(
                        "graph forget '%s': archive failed — soft-expiring "
                        "%d row(s) instead of deleting", t_norm, len(doomed))
                    conn.executemany(
                        'UPDATE triplets SET valid_until = ? WHERE rowid = ?',
                        [(now, row[0]) for row in doomed],
                    )
                deleted = len(doomed)
                conn.commit()
            for row in doomed:
                self._remove_edge(row[1], row[2], row[3])
        return deleted

    #: Generic hub nodes that link to nearly everything; expanding a
    #: `forget` to these would wipe unrelated knowledge, so they are never
    #: returned as connected entities. Pronouns are not enough — the live
    #: graph's real hubs are operational nouns (`assistant` deg 54, `project`
    #: 49, `system` 43, `done` 14). Also used as a guard in
    #: `_map_words_to_seeds` so a longer query word cannot re-seed a hub.
    _ENTITY_EXPANSION_STOPLIST = {
        "user", "me", "i", "you", "it", "this", "that", "they", "them",
        "he", "she", "we", "thing", "things",
        "assistant", "agent", "ghost", "system", "project", "projects",
        "task", "tasks", "todo", "done", "bug", "bugs", "issue", "issues",
        "file", "files", "code", "error", "errors", "service", "services",
        "status", "test", "tests", "data", "session", "model", "tool",
        "tools", "memory", "goal", "goals", "feature", "features",
    }

    #: Dynamic companion to the stoplist: any neighbour with a degree above
    #: this is a hub by measurement, whatever its name, and is never followed
    #: by the forget expansion (forgetting a chess bot must not reach `webos`).
    _EXPANSION_MAX_DEGREE = 8

    #: predicates that name the SAME thing — the only edges the forget
    #: expansion follows (both directions)
    _ALIAS_PREDICATE_RE = re.compile(r"ALIAS|AKA|ALSO_KNOWN_AS|KNOWN_AS|NICKNAME\w*|HAS_NICKNAME|SAME_AS|HAS_ALIAS")

    def get_connected_entities(self, target: str, limit: int = 8) -> List[str]:
        """Return distinct entity names directly (1 hop) connected to
        ``target``.

        Lets ``forget`` expand an entity wipe to its tightly-coupled
        neighbours: forgetting ``mortimer`` surfaces ``iguana`` (from a
        ``mortimer IS_A iguana`` edge) so the alias tombstone goes too.

        Every returned name is fed back into ``delete_by_target`` by the
        caller, so this is the amplifier of any forget: it is bounded on
        three axes — the anchor must match a WHOLE token of the node name
        (not a substring), only CURRENT edges are followed, and hub
        neighbours (by stoplist or by measured degree) are never returned.
        """
        if not target or len(target.strip()) < 3:
            return []
        t = target.lower().strip()
        like = f"%{t}%"
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    '''SELECT subject, object, predicate FROM triplets
                       WHERE (subject LIKE ? OR object LIKE ?)
                       AND valid_until IS NULL''',
                    (like, like)
                ).fetchall()
            out: List[str] = []
            seen = set()
            for s, o, pred in rows:
                # only an IDENTITY edge names the same thing twice ("mortimer
                # IS_A iguana"); a relationship names ANOTHER entity —
                # `forget leonidas` followed IS_SON_OF to fotini and deleted
                # the owner's wife's facts (profile-writes review)
                _p = str(pred or "").upper()
                # ALIASES only (another name for the same thing): a class
                # link (`hermes IS_A llm`) led the forget to sweep every
                # edge and fact naming the class word (third review)
                if not self._ALIAS_PREDICATE_RE.fullmatch(_p):
                    continue
                # the target must BE the node (not a token of it: `forget
                # postgresql` reached "evolmonkey IS_A postgresql services
                # company"), and a class link runs one way only: the thing →
                # its class ("mortimer IS_A iguana"); a class's instances are
                # never followed. Aliases run both ways.
                _sn, _on = str(s or "").lower().strip(), str(o or "").lower().strip()
                if _sn == t:
                    neighbours = (o,)
                elif _on == t:
                    neighbours = (s,)
                else:
                    continue
                for node in neighbours:
                    n = (node or "").lower().strip()
                    # Skip the target's own variants, hub nodes, and tiny tokens.
                    if not n or len(n) < 3 or t in n or n in t:
                        continue
                    if n in self._ENTITY_EXPANSION_STOPLIST:
                        continue
                    if n in self.nx_graph and \
                            self.nx_graph.degree(n) > self._EXPANSION_MAX_DEGREE:
                        continue
                    if n not in seen:
                        seen.add(n)
                        out.append(n)
                        if len(out) >= limit:
                            return out
            return out

    def wipe_all(self):
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                # Explicit operator intent ("reset all") — archive is
                # best-effort here: proceed with the wipe either way, but
                # a failure to preserve is logged loudly by the helper.
                try:
                    rows = conn.execute(
                        """SELECT subject, predicate, object,
                                  COALESCE(weight, 1), timestamp
                           FROM triplets""").fetchall()
                    self._archive_rows("wipe_all", rows)
                except Exception as e:
                    logger.error("graph wipe_all: pre-wipe archive read "
                                 "failed: %s", e)
                conn.execute('DELETE FROM triplets')
                conn.commit()
            self.nx_graph = nx.MultiDiGraph()
            self._invalidate_node_cache()

    def prune_stale_edges(self, max_age_days: int = 45, keep_min_weight: int = 1) -> int:
        """Forget low-signal stale edges — the graph's decay story.

        The graph is the only uncapped memory tier (vector 5000, episodes 500,
        skills 50); non-functional predicates (LIKES/HAS/KNOWS/…) accumulate
        forever, and weight-1 edges older than a threshold are almost always
        one-off extractor noise that dilutes retrieval. This deletes currently-
        valid edges with ``weight <= keep_min_weight`` and a ``timestamp``
        older than ``max_age_days`` from BOTH the DB and the in-memory mirror.
        Reinforced edges (weight > threshold) are kept regardless of age —
        weight IS the decay signal, previously stored but never used for
        forgetting. Returns the number of edges pruned. Idempotent, best-effort.
        Intended to run from the dream cycle."""
        removed = 0
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cutoff_expr = f"datetime('now', '-{int(max_age_days)} days')"
                    rows = conn.execute(
                        f"""SELECT subject, predicate, object,
                                   COALESCE(weight, 1), timestamp
                            FROM triplets
                            WHERE valid_until IS NULL
                              AND COALESCE(weight, 1) <= ?
                              AND timestamp < {cutoff_expr}""",
                        (int(keep_min_weight),),
                    ).fetchall()
                    # an OWNER fact never decays by age (profile-writes review:
                    # `user MARRIED_TO fotini`, `HAS_CHILD leonidas`, the
                    # HAS_<KEY> edges update_profile writes, `user OWNS …` —
                    # stated once, never reinforced, due to go at 45 days)
                    rows = [r for r in rows if not self._is_owner_fact(r[0], r[1], r[2])]
                    # Unattended dream-driven scrub: archive or DON'T prune.
                    if not self._archive_rows("prune_stale_edges", rows):
                        return 0
                    for s, p, o, _w, _ts in rows:
                        conn.execute(
                            "DELETE FROM triplets WHERE subject=? AND predicate=? AND object=?",
                            (s, p, o),
                        )
                        self._remove_edge(s, p, o)
                        removed += 1
                    conn.commit()
                if removed:
                    self._invalidate_node_cache()
        except Exception as e:
            logger.warning("graph prune_stale_edges failed: %s", e)
        return removed

    #: relation WORDS that describe a life (whole tokens of the predicate —
    #: third review: `.*SON.*` matched PERSON/REASON/SEASON, `.*NAME.*`
    #: HAS_NAME_SUGGESTIONS)
    _DURABLE_TOKENS = {"MARRIED", "SPOUSE", "WIFE", "HUSBAND", "PARTNER", "CHILD", "CHILDREN", "SON", "SONS",
                       "DAUGHTER", "DAUGHTERS", "PARENT", "PARENTS", "MOTHER", "FATHER", "SIBLING", "BROTHER",
                       "SISTER", "FAMILY", "BIRTH", "BIRTHDATE", "BIRTHDAY", "BORN", "OWNS", "OWNED", "OWN",
                       "RESIDES", "LIVES", "HOME", "ADDRESS", "EMPLOYED", "EMPLOYER", "PET", "PETS",
                       # health and profession (re-review: decay had pruned the owner's
                       # heart condition and medication)
                       "CONDITION", "CONDITIONS", "MEDICATION", "MEDICATIONS", "DIAGNOSIS", "DIAGNOSED",
                       "ALLERGY", "ALLERGIES", "ALLERGIC", "HEALTH", "DISEASE", "PROFESSION", "COMPANION"}
    #: a `user HAS_<KEY>` written by a probe or a test is noise, not a fact
    #: about the owner's life (re-review: HAS_TEST_COLOUR kept forever)
    _NOISE_TOKENS = {"TEST", "TESTING", "PROBE", "DEMO", "EXAMPLE", "DUMMY", "TEMP", "TMP", "SAMPLE", "FAKE",
                     # §4KY: the AGENT's / a project's state is not the owner's
                     # life (48 of 96 protected owner edges were this)
                     "PROJECT", "PROJECTS", "SKILL", "SKILLS", "TASK", "TASKS", "SANDBOX", "FILE", "FILES",
                     "DOCUMENTATION", "STAT", "STATS", "LEARNING", "CODENAME", "WORKSPACE", "RESOURCE",
                     "INTROSPECTION", "COMPETENCE", "SESSION"}
    #: …and whole predicates (WORKS_ON a chat project is not employment)
    #: (LOCATED_IN: where a place IS — not LOCATED_NEAR / IS_AT, a trip)
    _DURABLE_PREDICATES = {"HAS_NAME", "IS_NAMED", "NAMED", "WORKS_AT", "WORKS_FOR"}

    @classmethod
    def _is_owner_fact(cls, subject, predicate, obj) -> bool:
        """An edge decay must keep: a durable relation (family, birth,
        ownership, residence, work, a name) whoever its subject, or a field
        `update_profile` wrote (`user HAS_<KEY> …`). The owner's chatter
        (`user GREETED ghost`, `user ASKED_ABOUT weather`) is not a fact
        about the owner's life and still decays (third review: protecting
        every `user` edge made the graph's main noise permanent)."""
        p = str(predicate or "").upper()
        if set(p.split("_")) & cls._NOISE_TOKENS:
            return False
        if p in cls._DURABLE_PREDICATES or set(p.split("_")) & cls._DURABLE_TOKENS:
            return True
        return str(subject or "").strip().lower() == "user" and p.startswith("HAS_")

    def get_recent_triplets(self, limit: int = 100) -> List[Dict[str, str]]:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                # Only CURRENT facts (valid_until IS NULL) — otherwise a
                # superseded/expired triplet ("bob WORKS_AT google", later
                # replaced by "meta") is returned alongside the live one and,
                # if surfaced into context, contradicts the current fact.
                cursor = conn.execute(
                    'SELECT subject, predicate, object FROM triplets '
                    'WHERE valid_until IS NULL ORDER BY timestamp DESC LIMIT ?',
                    (limit,)
                )
                return [dict(row) for row in cursor.fetchall()]

    def propose_merge_candidates(self, max_candidates: int = 12,
                                 neighbor_window: int = 5,
                                 fuzzy_cutoff: float = 0.90) -> List[Dict[str, str]]:
        """Deterministic near-duplicate node pairs for dream-time compression.

        Two tiers, distinguished by ``kind``:

        - ``"safe"`` — names identical after stripping punctuation/whitespace
          ("new-york" vs "new york"): mergeable without confirmation.
        - ``"fuzzy"`` — high-similarity lexicographic neighbors (plural forms,
          trailing typos): the caller must get an LLM same-entity confirmation
          before merging — string similarity alone conflates distinct entities
          ("new"/"news").

        Fuzzy comparison is bounded to each node's ``neighbor_window`` sorted
        neighbors (near-duplicates share prefixes) so the pass stays
        ~O(n log n) on the only uncapped memory tier. Canonical direction: the
        higher-degree node survives as ``new_node`` (ties: the longer name).
        Read-only — the merge policy lives in core/dream.py."""
        import re

        def _norm(name: str) -> str:
            return re.sub(r"[\s\-_./'\"]+", "", name)

        with self._lock:
            nodes = [n for n in self._nodes_snapshot()
                     if isinstance(n, str) and len(n) > 2 and not n.isdigit()]
            deg = {n: self.nx_graph.degree(n) for n in nodes}

        out: List[Dict[str, str]] = []
        seen: set = set()

        def _add(a: str, b: str, kind: str):
            key = tuple(sorted((a, b)))
            if key in seen:
                return
            seen.add(key)
            # Higher-degree node survives; tie broken toward the longer name.
            if (deg.get(a, 0), len(a)) >= (deg.get(b, 0), len(b)):
                old, new = b, a
            else:
                old, new = a, b
            out.append({"old_node": old, "new_node": new, "kind": kind})

        def _digit_distinct(a: str, b: str) -> bool:
            # §4MI: two names whose NUMBERS differ are versions, models or
            # ids ("qwen3.6-35b" / "qwen3.5-35b", "1.10" / "1.1.0",
            # "server1" / "server2") — never a spelling variant. `_norm`
            # strips the dots, so these were "safe" merges or 0.90-ratio
            # fuzzy candidates re-asked every dream since 08-25. Names
            # whose numbers agree ("topic 0" / "topic-0") stay candidates.
            _da, _db = re.findall(r"\d+", a), re.findall(r"\d+", b)
            if not _da or not _db:
                return False
            # r2: the NUMBERS decide — "ubuntu 22.04" / "ubuntu-22-04" are
            # one thing; "1.10" / "1.1.0" have different numbers anyway
            return _da != _db

        by_norm: Dict[str, List[str]] = {}
        for n in nodes:
            by_norm.setdefault(_norm(n), []).append(n)
        for variants in by_norm.values():
            # r2: pair each variant with the first one it is NOT
            # digit-distinct from (comparing only against variants[0] left
            # two equal names unpaired when the first was the odd one)
            for i, other in enumerate(variants[1:], 1):
                base = next((v for v in variants[:i] if not _digit_distinct(v, other)), None)
                if base is not None:
                    _add(base, other, "safe")

        ordered = sorted(nodes)
        for i, a in enumerate(ordered):
            if len(out) >= max_candidates:
                break
            for b in ordered[i + 1:i + 1 + neighbor_window]:
                if _norm(a) == _norm(b):
                    continue  # tier-1 pair (or already merged direction)
                if _digit_distinct(a, b):
                    continue  # §4MI: two versions are two things
                if difflib.SequenceMatcher(None, a, b).ratio() >= fuzzy_cutoff:
                    _add(a, b, "fuzzy")

        return out[:max_candidates]

    def execute_graph_compression(self, merges: List[Dict[str, str]]) -> int:
        ops = 0
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for m in merges:
                    old_node = m.get("old_node", "").lower().strip()
                    new_node = m.get("new_node", "").lower().strip()
                    if old_node and new_node and old_node != new_node:
                        try:
                            # Snapshot every triple that touches old_node on
                            # EITHER side (a self-loop old->old matches once),
                            # then rewrite old_node -> new_node wherever it
                            # appears and merge each rewritten triple into the
                            # target. We carry valid_from/valid_until through so
                            # a superseded (expired) fact does NOT come back as
                            # current, and rewrite both endpoints so an
                            # old->old self-loop survives as new->new instead of
                            # being deleted. Weights sum; temporal state merges
                            # current-wins (see _merge_triplet_row).
                            cursor.execute(
                                '''SELECT subject, predicate, object, weight, valid_from, valid_until
                                   FROM triplets WHERE subject = ? OR object = ?''',
                                (old_node, old_node))
                            src_rows = cursor.fetchall()
                            # §4MI: a merge rewrites and DELETES rows with no
                            # record — 40 merges in two months, none
                            # reconstructible, and the docs promised the
                            # archive. Fail closed like every other
                            # destructive path here.
                            if not self._archive_rows(
                                    f"compress:{old_node}->{new_node}",
                                    [(s_, p_, o_, w_) for s_, p_, o_, w_, _vf, _vu in src_rows]):
                                logger.warning("graph compression skipped %r -> %r: archive unwritable",
                                               old_node, new_node)
                                continue
                            logger.info("graph compression: %r -> %r (%d row(s))",
                                        old_node, new_node, len(src_rows))
                            for subj, pred, obj, w, vfrom, vuntil in src_rows:
                                n_subj = new_node if subj == old_node else subj
                                n_obj = new_node if obj == old_node else obj
                                if n_subj == n_obj and subj != obj:
                                    # The edge ran BETWEEN the two merged nodes
                                    # ("new-york SAME_AS new york"). Rewriting it
                                    # mints a self-loop that duplicates the merged
                                    # node's whole neighbourhood in every
                                    # spreading-activation chain. Drop it — a
                                    # genuine old->old self-loop (subj == obj)
                                    # still migrates to new->new below.
                                    continue
                                self._merge_triplet_row(
                                    cursor, n_subj, pred, n_obj,
                                    int(w or 1), vfrom, vuntil)
                            # Delete the migrated source rows. Rewritten targets
                            # never reference old_node, so this only removes the
                            # originals, never the freshly-merged copies.
                            cursor.execute(
                                "DELETE FROM triplets WHERE subject = ? OR object = ?",
                                (old_node, old_node))
                            # A subject-side merge unions two different objects of
                            # the same functional predicate ("bob WORKS_AT google"
                            # + "bobby WORKS_AT meta"), minting two mutually
                            # exclusive CURRENT facts. add_triplets' conflict
                            # resolution never runs here, so re-apply it.
                            self._reconcile_functional_conflicts(cursor, new_node)
                            ops += 1
                        except Exception as e:
                            logger.debug(f"Graph compression merge failed: {type(e).__name__}: {e}")
                conn.commit()
            # Rebuild mirror after a structural rewrite — simpler than diffing.
            self.initialize_graph()
        return ops

    def _reconcile_functional_conflicts(self, cursor, subject: str) -> int:
        """Re-apply single-valued-predicate expiry for one subject.

        `add_triplets` resolves functional conflicts at write time, but node
        compression rewrites rows straight into SQLite and can leave two
        CURRENT objects under the same functional predicate. Newest wins
        (latest ``valid_from``, then ``timestamp``, then insertion order);
        the losers get a ``valid_until`` stamp. Returns the number expired."""
        import time as _time
        now = _time.time()
        expired = 0
        rows = cursor.execute(
            '''SELECT rowid, predicate, object, COALESCE(valid_from, 0), timestamp
               FROM triplets WHERE subject = ? AND valid_until IS NULL''',
            (subject,)).fetchall()
        by_pred: Dict[str, List] = {}
        for rid, pred, obj, vfrom, ts in rows:
            pred_u = str(pred or "").upper()
            if pred_u in self._FUNCTIONAL_PREDICATES:
                by_pred.setdefault(pred_u, []).append((vfrom, ts or "", rid, pred, obj))
        for pred_u, items in by_pred.items():
            if len(items) < 2:
                continue
            items.sort(reverse=True)  # newest valid_from / timestamp / rowid first
            winner = items[0]
            for vfrom, ts, rid, pred, obj in items[1:]:
                cursor.execute(
                    'UPDATE triplets SET valid_until = ? WHERE rowid = ?', (now, rid))
                expired += 1
                logger.warning(
                    "graph expiry (merge): %s %s '%s' superseded by '%s'",
                    subject, pred, obj, winner[4],
                )
        return expired

    @staticmethod
    def _merge_triplet_row(cursor, subject: str, predicate: str, obj: str,
                           weight: int, valid_from, valid_until) -> None:
        """Insert (subject, predicate, obj) or, if it already exists, merge into
        it: weights sum, valid_from keeps the earliest, and valid_until is
        current-wins — if EITHER the incoming or existing row is current
        (valid_until IS NULL) the result is current; if both are expired we keep
        the later (max) expiry. Used by graph compression so a node merge never
        resurrects a superseded fact and never double-counts weight."""
        cursor.execute(
            '''SELECT weight, valid_from, valid_until FROM triplets
               WHERE subject = ? AND predicate = ? AND object = ?''',
            (subject, predicate, obj))
        existing = cursor.fetchone()
        if existing is None:
            cursor.execute(
                '''INSERT INTO triplets (subject, predicate, object, weight, timestamp, valid_from, valid_until)
                   VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?)''',
                (subject, predicate, obj, int(weight or 1), valid_from, valid_until))
            return
        ex_w, ex_from, ex_until = existing
        merged_w = int(ex_w or 1) + int(weight or 1)
        if ex_until is None or valid_until is None:
            merged_until = None
        else:
            merged_until = max(ex_until, valid_until)
        froms = [v for v in (ex_from, valid_from) if v is not None]
        merged_from = min(froms) if froms else None
        cursor.execute(
            '''UPDATE triplets SET weight = ?, timestamp = CURRENT_TIMESTAMP,
                                   valid_from = ?, valid_until = ?
               WHERE subject = ? AND predicate = ? AND object = ?''',
            (merged_w, merged_from, merged_until, subject, predicate, obj))

    def get_expired_triplets(self, subject: str = None, limit: int = 50) -> List[Dict]:
        """Return expired (superseded) triplets, optionally filtered by subject."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                if subject:
                    cursor = conn.execute(
                        '''SELECT subject, predicate, object, valid_from, valid_until
                           FROM triplets WHERE valid_until IS NOT NULL AND subject = ?
                           ORDER BY valid_until DESC LIMIT ?''',
                        (subject.lower().strip(), limit)
                    )
                else:
                    cursor = conn.execute(
                        '''SELECT subject, predicate, object, valid_from, valid_until
                           FROM triplets WHERE valid_until IS NOT NULL
                           ORDER BY valid_until DESC LIMIT ?''',
                        (limit,)
                    )
                return [dict(row) for row in cursor.fetchall()]

    # -------------------------------------------------------- spreading act'n

    #: Minimum length / minimum length-ratio for a node name that is a
    #: FRAGMENT of the query word ('ai' from 'aiohttp', 'user' from
    #: 'username'). That direction re-seeds generic ego hubs from unrelated
    #: words and blows the ego graph into every turn's context — the exact
    #: hydration that `bus._STOPWORDS` exists to prevent. The opposite
    #: direction (word inside a longer node: 'germ' -> 'germany') narrows
    #: instead of widening and stays unrestricted.
    _SEED_FRAGMENT_MIN_LEN = 4
    _SEED_FRAGMENT_MIN_RATIO = 0.75

    def _seed_containment_ok(self, node: str, word: str) -> bool:
        """Guard for the substring tier of `_map_words_to_seeds`."""
        if node == word:
            return True
        # Never reach a hub node except by an exact query word.
        if node in self._ENTITY_EXPANSION_STOPLIST:
            return False
        if node in word:  # node is a fragment of the word — the risky direction
            if len(node) < self._SEED_FRAGMENT_MIN_LEN:
                return False
            if len(node) < self._SEED_FRAGMENT_MIN_RATIO * len(word):
                return False
        elif word in node:
            # §4MI: the OTHER direction was unguarded — "back" seeded
            # `xtrabackup`, "brown" seeded a description node. A word
            # reaches a longer node only as a WHOLE token of it (r2: any
            # length — "grid", "ring", "port" are real tokens — and Unicode
            # word boundaries, so a Greek fragment is not a token either).
            import re as _re
            if _re.search(r"(?<!\w)" + _re.escape(word) + r"(?!\w)", node):
                return True
            # …or the START of a token, when the word is most of it
            # ("germ" → `germany`, "german" → `germany`); never its middle
            # or end ("back" ⊄ `xtrabackup`)
            for _tok in _re.findall(r"\w+", node):
                if (_tok.startswith(word) and len(word) >= 4
                        and len(word) >= 0.5 * len(_tok)):
                    return True
            return False
        return True

    def _map_words_to_seeds(self, words: Iterable[str], fuzzy: bool = True) -> List[str]:
        """Map free-form query words to exact node names in the graph.

        Strategy per word: exact match → substring containment → difflib
        fuzzy fallback. Words shorter than 3 chars are skipped.
        """
        if not self.nx_graph or self.nx_graph.number_of_nodes() == 0:
            return []
        seeds: List[str] = []
        seen = set()
        all_nodes = self._nodes_snapshot()
        for w in words:
            wl = str(w).lower().strip()
            if len(wl) < 3:
                continue
            matches: List[str] = []
            if wl in self.nx_graph:
                matches = [wl]
            else:
                _raw_sub = [n for n in all_nodes if (wl in n or n in wl)]
                substr = [n for n in _raw_sub if self._seed_containment_ok(n, wl)]
                if _raw_sub and not substr:
                    # r2 review: every substring hit was REFUSED — the word
                    # is a fragment, not a name; falling to the fuzzy tier
                    # turned "tool" into `pool` and "word" into `work`
                    matches = []
                elif substr:
                    # Prefer the closest length match for stable ordering
                    matches = sorted(substr, key=lambda n: (abs(len(n) - len(wl)), n))[:3]
                elif not fuzzy:
                    # §4MJ: hydration asks for no fuzzy seeds — 91 graph
                    # items on 80 real turns came only from them and 0 were
                    # relevant ("sitting"→`stealthing`, "going"→`coding`)
                    matches = []
                else:
                    # Same hub protection on the fuzzy tier: 'systemd' is a
                    # 0.92 difflib match for the 'system' hub.
                    matches = [
                        m for m in difflib.get_close_matches(wl, all_nodes, n=3, cutoff=0.7)
                        if m not in self._ENTITY_EXPANSION_STOPLIST
                    ]
            for m in matches:
                if m not in seen:
                    seen.add(m)
                    seeds.append(m)
        return seeds

    def _out_edges(self, node: str) -> List[Tuple[str, str, str, int]]:
        return [(node, nb, d.get("predicate", ""), int(d.get("weight", 1) or 1))
                for _, nb, _, d in self.nx_graph.out_edges(node, keys=True, data=True)]

    def _in_edges(self, node: str) -> List[Tuple[str, str, str, int]]:
        return [(pr, node, d.get("predicate", ""), int(d.get("weight", 1) or 1))
                for pr, _, _, d in self.nx_graph.in_edges(node, keys=True, data=True)]

    def _spreading_activation(self, seed: str,
                              path_scores: Dict[Tuple, int],
                              max_hops: int = 3) -> None:
        """Multi-hop BFS from `seed`, recording every directed chain of
        length 1 to `max_hops` that is naturally readable.
        Score = sum of edge weights along the chain.

        3-hop enables complex reasoning like:
        "A works_at B, B is_owned_by C, C is_located_in D"
        """
        if seed not in self.nx_graph:
            return
        out1 = self._out_edges(seed)
        in1 = self._in_edges(seed)

        def bump(chain: Tuple[Tuple[str, str, str], ...], score: int):
            prev = path_scores.get(chain)
            if prev is None or prev < score:
                path_scores[chain] = score

        # 1-hop forward and backward
        for s, o, p, w in out1:
            bump(((s, p, o),), w)
        for s, o, p, w in in1:
            bump(((s, p, o),), w)

        # 2-hop forward: seed -> Y -> Z
        for s1, y, p1, w1 in out1:
            for _, z, p2, w2 in self._out_edges(y):
                if z == seed:
                    continue
                bump(((s1, p1, y), (y, p2, z)), w1 + w2)

                # 3-hop forward: seed -> Y -> Z -> W
                if max_hops >= 3:
                    for _, w_node, p3, w3 in self._out_edges(z):
                        if w_node == seed or w_node == y:
                            continue
                        bump(((s1, p1, y), (y, p2, z), (z, p3, w_node)), w1 + w2 + w3)

        # 2-hop backward: Z -> Y -> seed   (Y is the in-neighbour of seed)
        for x, o1, p1, w1 in in1:
            for z, _, p2, w2 in self._in_edges(x):
                if z == seed:
                    continue
                bump(((z, p2, x), (x, p1, o1)), w1 + w2)

                # 3-hop backward: W -> Z -> Y -> seed
                if max_hops >= 3:
                    for w_node, _, p3, w3 in self._in_edges(z):
                        if w_node == seed or w_node == x:
                            continue
                        bump(((w_node, p3, z), (z, p2, x), (x, p1, o1)), w1 + w2 + w3)

        # Through-chain: X -> seed -> Y
        for x, _, p1, w1 in in1:
            for _, y, p2, w2 in out1:
                if x == y:
                    continue
                bump(((x, p1, seed), (seed, p2, y)), w1 + w2)

    @staticmethod
    def _format_path(chain: Tuple[Tuple[str, str, str], ...], score: int) -> str:
        first = chain[0][0]
        parts = [f"({first.title()})"]
        for s, p, o in chain:
            parts.append(f"-[{p}]->")
            parts.append(f"({o.title()})")
        line = "- " + " ".join(parts)
        if score > len(chain):
            line += f" [Score {score}]"
        return line

    #: the agent's OWN log lines ("ai RESPONDED_TO user"). Only as a SUBJECT:
    #: "user HAS_INTEREST ai" is the owner's fact, and "ghost IS_A framework" /
    #: a database named "agent" are real entities (§4LB r2)
    _AGENT_NODES = frozenset({"ai", "assistant", "system", "the assistant", "the ai"})

    def get_neighborhood(self, words: List[str], global_limit: int = 25, fuzzy: bool = True) -> List[str]:
        """Spreading-activation GraphRAG over the in-memory NetworkX graph.

        1. Map query words to exact graph nodes (fuzzy → exact matching).
        2. Run a 3-hop BFS from each seed, scoring chains by edge-weight sum
           (a chain contained in a higher-ranked chain is dropped, §4MJ).
        3. Return the highest-scoring directed paths formatted for the LLM.
        """
        with self._lock:
            seeds = self._map_words_to_seeds(words, fuzzy=fuzzy)
            if not seeds:
                return []
            path_scores: Dict[Tuple, int] = {}
            for seed in seeds:
                self._spreading_activation(seed, path_scores)
            if not path_scores:
                return []
            # §4LB: a chain through the AGENT's own nodes (ai / assistant /
            # system) is a log of what the agent did, not knowledge — 49% of
            # injected graph items ("(Ai)-[RESPONDED_TO]->(User)-[HAS_SON]->…"
            # for a question about MoE models)
            path_scores = {c: v for c, v in path_scores.items()
                           if not any(str(t[0]).lower() in self._AGENT_NODES for t in c)}
            sorted_paths = sorted(
                path_scores.items(),
                key=lambda item: (item[1], len(item[0])),
                reverse=True
            )
            # §4MJ: a chain and its own sub-path were two items (a 3-hop
            # chain always outranks its prefixes) — 102 duplicate pairs on
            # 16% of real turns, each taking one of the tier's 6 slots
            kept: List[Tuple[Tuple, int]] = []
            for chain, score in sorted_paths:
                n = len(chain)
                if any(len(k) > n and any(k[i:i + n] == chain for i in range(len(k) - n + 1))
                       for k, _ in kept):
                    continue
                kept.append((chain, score))
                if len(kept) >= global_limit:
                    break
            return [self._format_path(chain, score) for chain, score in kept]
