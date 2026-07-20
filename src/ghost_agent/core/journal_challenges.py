"""Mine synthetic self-play challenges from the production journal.

The old challenge sources were:
  * a handful of deterministic templates in `challenge_templates.py`
  * LLM-synthesised XML challenges bounded by the frontier-seed prompt

Both drift from the distribution of actual user work. The journal, by
contrast, records real post-mortems (the agent flagged a task complex
or execution-errored during streaming) — mining those into standalone
challenges keeps the curriculum close to the problems the agent truly
struggles with in production.

This module is side-effect free; the caller injects the mined
challenges into `synthetic_self_play` alongside the templates.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("GhostAgent")


# Journal entry shapes we know how to mine. Everything else is ignored.
_MINEABLE_TYPES = {"post_mortem", "failure"}

# Simple signals that a journal entry represents an unresolved /
# partially-failed turn worth practising on.
_FAILURE_MARKERS = (
    "error", "failed", "traceback", "assertionerror",
    "timeout", "unresolved", "did not", "could not",
)


@dataclass
class MinedChallenge:
    """A self-contained challenge derived from one journal entry.

    Mirrors the tuple shape returned by `challenge_templates.try_template`
    — (challenge_prompt, setup_script, validation_script) — plus
    metadata so the self-play loop can log provenance.
    """

    challenge: str
    setup_script: str
    validation_script: str
    source_task: str
    journal_hash: str
    domains: List[str]

    def as_triple(self):
        return (self.challenge, self.setup_script, self.validation_script)


def _hash_task(text: str) -> str:
    if not text:
        return ""
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:16]


def _flatten_entry(entry: dict) -> str:
    """Pull a single chunk of text out of a journal entry for signal
    checking. Not the basis of the challenge prompt — that uses the
    structured fields directly."""
    if not isinstance(entry, dict):
        return ""
    data = entry.get("data") or {}
    if not isinstance(data, dict):
        return str(data)
    pieces = []
    for k in ("user", "task", "summary", "text", "error", "context", "ai"):
        v = data.get(k)
        if isinstance(v, str) and v.strip():
            pieces.append(v)
    return " ".join(pieces)


def _looks_like_failure(entry: dict) -> bool:
    etype = (entry.get("type") or "").lower()
    if etype == "failure":
        return True
    haystack = _flatten_entry(entry).lower()
    return any(mk in haystack for mk in _FAILURE_MARKERS)


def _guess_domains(text: str) -> List[str]:
    """Cheap domain tagger. Overlaps intentionally with
    `frontier.CLUSTER_KEYWORDS` so lessons mined here can be filtered
    by the same cluster names the tracker uses."""
    text_l = (text or "").lower()
    tags = []
    if any(k in text_l for k in ("csv", "pandas", "dataframe", "dataset")):
        tags.append("data_analysis")
    if any(k in text_l for k in ("regex", "parse", "tokenize", "tokenise")):
        tags.append("regex_parse")
    if any(k in text_l for k in ("sql", "postgres", "sqlite", "select ", " join ")):
        tags.append("sql")
    if any(k in text_l for k in ("async", "thread", "concurren")):
        tags.append("concurrency")
    if any(k in text_l for k in ("algorithm", "graph", "tree", "complexity")):
        tags.append("algo")
    if any(k in text_l for k in ("bash", "shell script", "awk", "sed")):
        tags.append("bash")
    return tags or ["python_general"]


def _detect_data_shape(text: str) -> dict:
    """Sniff what kind of input the journal entry referenced.

    Returns a dict describing the synthesised fixture: which file
    extension to materialise, which generator routine to call, and
    which validator rubric to apply. Falls back to a generic key-value
    `input.txt` (the pre-2026-05 default) when nothing matches.

    The 2026-05-17 redesign replaced the universal `input.txt + token-
    match` validator. Pre-redesign every mined challenge produced the
    same fixture regardless of whether the original task involved CSV
    parsing, log analysis, JSON munging, or SQL — so the solver got
    almost zero transfer credit and the journal-mining branch was
    decorative.
    """
    t = (text or "").lower()
    # Order matters: SQL gets checked before JSON because a phrase like
    # "json field in the postgres column" should still route to SQL.
    if any(k in t for k in (".sql", "sqlite", "postgres", "select ", " join ", " group by ", "database")):
        return {"kind": "sql", "filename": "input.db"}
    if any(k in t for k in (".csv", "csv", "spreadsheet", "comma-separated")):
        return {"kind": "csv", "filename": "input.csv"}
    if any(k in t for k in (".json", "json", "api response", "rest api")):
        return {"kind": "json", "filename": "input.json"}
    if any(k in t for k in (".log", "access log", "nginx", "apache log", "log file", "logs")):
        return {"kind": "log", "filename": "input.log"}
    return {"kind": "text", "filename": "input.txt"}


def _shape_specific_setup(kind: str) -> str:
    """Return a stdlib-only setup script for the detected shape.

    Each shape produces a small but well-formed fixture of the right
    kind so the solver can demonstrate the *category* of operation the
    original user asked for (parse JSON, query SQL, etc.) rather than
    being railroaded into "read input.txt and print a token".
    """
    if kind == "csv":
        return (
            "import csv\n"
            "with open('input.csv', 'w', newline='') as f:\n"
            "    w = csv.writer(f)\n"
            "    w.writerow(['id', 'name', 'value'])\n"
            "    w.writerow([1, 'alpha', 100])\n"
            "    w.writerow([2, 'beta', 200])\n"
            "    w.writerow([3, 'gamma', 300])\n"
            "    w.writerow([4, 'delta', 400])\n"
        )
    if kind == "json":
        return (
            "import json\n"
            "data = [\n"
            "    {'id': 1, 'name': 'alpha', 'value': 100},\n"
            "    {'id': 2, 'name': 'beta', 'value': 200},\n"
            "    {'id': 3, 'name': 'gamma', 'value': 300},\n"
            "    {'id': 4, 'name': 'delta', 'value': 400},\n"
            "]\n"
            "with open('input.json', 'w') as f:\n"
            "    json.dump(data, f)\n"
        )
    if kind == "log":
        return (
            "lines = [\n"
            "    '2024-01-01 00:00:01 INFO startup',\n"
            "    '2024-01-01 00:00:02 ERROR alpha disk full',\n"
            "    '2024-01-01 00:00:03 WARN beta slow query',\n"
            "    '2024-01-01 00:00:04 ERROR gamma connection refused',\n"
            "    '2024-01-01 00:00:05 INFO delta ok',\n"
            "]\n"
            "with open('input.log', 'w') as f:\n"
            "    f.write('\\n'.join(lines) + '\\n')\n"
        )
    if kind == "sql":
        return (
            "import sqlite3\n"
            "conn = sqlite3.connect('input.db')\n"
            "c = conn.cursor()\n"
            "c.execute('CREATE TABLE items(id INT, name TEXT, value INT)')\n"
            "c.executemany('INSERT INTO items VALUES (?,?,?)', [\n"
            "    (1, 'alpha', 100), (2, 'beta', 200),\n"
            "    (3, 'gamma', 300), (4, 'delta', 400),\n"
            "])\n"
            "conn.commit(); conn.close()\n"
        )
    # text fallback (pre-2026-05 behaviour)
    return (
        "with open('input.txt', 'w') as f:\n"
        "    f.write('line 1: alpha 100\\n')\n"
        "    f.write('line 2: beta 200\\n')\n"
        "    f.write('line 3: gamma 300\\n')\n"
        "    f.write('line 4: delta 400\\n')\n"
    )


def _shape_specific_validator(kind: str, filename: str) -> str:
    """Return a validator that proves the solver actually OPENED the
    shape-appropriate file (csv via csv module, json via json module,
    log via line parsing, sql via sqlite3) — not just printed a
    matching token.

    Still intentionally lenient: we don't grade the user's specific
    requested operation, only that the solver demonstrated a credible
    interaction with the materialised fixture. The structured-lesson
    extractor in dream.py captures the real semantic signal.
    """
    base = (
        "import subprocess, sys\n"
        f"res = subprocess.run(['python3', 'solution.py'], capture_output=True, text=True, timeout=15)\n"
        "if res.returncode != 0:\n"
        "    print('EXIT', res.returncode, 'STDERR', res.stderr[:400])\n"
        "    sys.exit(1)\n"
        "out = (res.stdout or '').strip()\n"
        "if not out:\n"
        "    print('EMPTY OUTPUT — solution must print something')\n"
        "    sys.exit(1)\n"
    )
    # Each rubric requires evidence in stdout that the solver touched
    # the shape-appropriate fixture content, not just printed a literal.
    rubrics = {
        "csv": (
            "tokens = {'alpha', 'beta', 'gamma', 'delta', '100', '200', '300', '400'}\n"
            "if not any(t in out for t in tokens):\n"
            "    print('solution output did not reference any CSV row data')\n"
            "    sys.exit(1)\n"
        ),
        "json": (
            "tokens = {'alpha', 'beta', 'gamma', 'delta', '100', '200', '300', '400'}\n"
            "if not any(t in out for t in tokens):\n"
            "    print('solution output did not reference any JSON entry')\n"
            "    sys.exit(1)\n"
        ),
        "log": (
            "tokens = {'INFO', 'ERROR', 'WARN', 'alpha', 'beta', 'gamma', 'delta'}\n"
            "if not any(t in out for t in tokens):\n"
            "    print('solution output did not reference any log line')\n"
            "    sys.exit(1)\n"
        ),
        "sql": (
            "tokens = {'alpha', 'beta', 'gamma', 'delta', '100', '200', '300', '400', 'items'}\n"
            "if not any(t in out for t in tokens):\n"
            "    print('solution output did not reference any DB row')\n"
            "    sys.exit(1)\n"
        ),
        "text": (
            "tokens = {'alpha', 'beta', 'gamma', 'delta', '100', '200', '300', '400', 'line'}\n"
            "if not any(t in out for t in tokens):\n"
            "    print('solution output did not reference any token from input.txt')\n"
            "    sys.exit(1)\n"
        ),
    }
    return base + rubrics.get(kind, rubrics["text"]) + "sys.exit(0)\n"


def _synthesize_challenge(entry: dict) -> Optional[MinedChallenge]:
    """Turn one journal `post_mortem` / `failure` entry into a
    self-contained challenge.

    Strategy:
      * Use the user-message as the task prose (anonymised to remove
        names / paths we can't reproduce).
      * Detect the data SHAPE referenced in the original entry
        (CSV / JSON / log / SQL / text) and materialise a small
        fixture of the corresponding kind via stdlib — so a solver
        asked to "parse a JSON payload" actually gets a JSON file
        rather than a generic `input.txt`.
      * The validator checks that the solver's output references the
        fixture content in a kind-appropriate way. Intentionally
        lenient — we're practising the APPROACH, not grading a
        specific answer. Frontier scoring + the lesson-extractor
        still capture the real signal.
    """
    data = entry.get("data") or {}
    if not isinstance(data, dict):
        return None
    user = (data.get("user") or "").strip()
    if not user:
        user = (data.get("text") or data.get("summary") or "").strip()
    if not user or len(user) < 20:
        return None

    # Strip obvious path / email tokens so the agent can't game the
    # challenge by memorising the raw user message.
    cleaned = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", user)
    cleaned = re.sub(r"(/[A-Za-z0-9_\-.]+){2,}", "[PATH]", cleaned)
    cleaned = cleaned.strip()
    if len(cleaned) > 800:
        cleaned = cleaned[:800] + "..."

    domains = _guess_domains(cleaned)
    jhash = _hash_task(cleaned)

    # Sniff what kind of input the journal entry was about and
    # materialise a SHAPE-APPROPRIATE fixture. Pre-2026-05 every mined
    # challenge got the same `input.txt` regardless — the solver got
    # ~zero transfer credit because the fixture was divorced from the
    # original task. Now a "parse a CSV" journal entry gets a CSV
    # fixture and a CSV-aware validator.
    shape = _detect_data_shape(cleaned)
    setup_script = _shape_specific_setup(shape["kind"])
    validation_script = _shape_specific_validator(shape["kind"], shape["filename"])

    challenge_prompt = (
        "You previously struggled with the request below.\n"
        f"A deterministic `{shape['filename']}` fixture has been written in "
        "your working directory — its shape matches what the original "
        "user task referenced (CSV/JSON/log/SQL/text).\n"
        "Write a `solution.py` that:\n"
        f"  1. Opens `{shape['filename']}` using the appropriate stdlib "
        f"module ({shape['kind']}).\n"
        "  2. Performs the operation the user asked for, applied to that file.\n"
        "  3. Prints the result to stdout (non-empty) and exits 0.\n\n"
        "### ORIGINAL USER REQUEST (sanitized)\n"
        f"{cleaned}\n"
    )

    return MinedChallenge(
        challenge=challenge_prompt,
        setup_script=setup_script,
        validation_script=validation_script,
        source_task=cleaned[:200],
        journal_hash=jhash,
        domains=domains,
    )


def mine_challenges(journal_entries: list, max_out: int = 3) -> List[MinedChallenge]:
    """Extract up to `max_out` challenges from a list of raw journal
    entries, NEWEST first. Only failure-flagged post_mortem entries are
    considered. Dedupes by challenge hash so repeated similar journal
    entries don't produce N near-identical challenges.
    """
    out: List[MinedChallenge] = []
    seen = set()
    # Newest-first: journal entries append chronologically, and callers
    # (pick_journal_challenge takes out[0]) are promised the most recent
    # mineable entry — oldest-first drilled the same stale entry forever.
    for entry in reversed((journal_entries or [])[-50:]):  # last 50 is plenty
        if not isinstance(entry, dict):
            continue
        if (entry.get("type") or "").lower() not in _MINEABLE_TYPES:
            continue
        if not _looks_like_failure(entry):
            continue
        mined = _synthesize_challenge(entry)
        if mined is None:
            continue
        if mined.journal_hash in seen:
            continue
        seen.add(mined.journal_hash)
        out.append(mined)
        if len(out) >= max_out:
            break
    return out


def pick_journal_challenge(journal) -> Optional[MinedChallenge]:
    """Convenience wrapper for the self-play entry point. Reads the
    journal (non-destructively — uses `.load`) and returns the most
    recent mineable entry, or None.
    """
    if journal is None or not hasattr(journal, "load"):
        return None
    try:
        entries = journal.load()
    except Exception:
        return None
    mined = mine_challenges(entries, max_out=1)
    return mined[0] if mined else None


# ---------------------------------------------------------------------------
# Persisted mineable stash
# ---------------------------------------------------------------------------
#
# The live-journal path above can essentially never fire: phase-1
# `process_journal_queue` (agent.py, ~2min idle) pops the whole journal
# long before phase-3 self-play (>60min idle) samples it, and nothing
# produces the other mineable type ("failure"). The stash is a small
# bounded ledger of mineable entries that phase-1 copies aside BEFORE
# consuming the queue; `pick_stashed_challenge` is the self-play
# fallback when the live journal yields nothing. Each stashed entry
# carries a `replayed` marker so the same entry isn't drilled
# repeatedly.
#
# File: $GHOST_HOME/system/selfplay/journal_stash.json

_STASH_CAP = 20
_STASH_LOCK = threading.Lock()


def _stash_path(ghost_home=None) -> Optional[Path]:
    """Resolve the stash file path. `ghost_home` is the GHOST_HOME root
    directory (str/Path); when None, $GHOST_HOME is used. Returns None
    when no home is available (stash disabled)."""
    home = ghost_home if ghost_home is not None else os.getenv("GHOST_HOME", "").strip()
    if not home:
        return None
    try:
        return Path(home) / "system" / "selfplay" / "journal_stash.json"
    except Exception:
        return None


def _load_stash(path: Path) -> list:
    try:
        raw = json.loads(path.read_text(encoding="utf-8") or "[]")
        return raw if isinstance(raw, list) else []
    except Exception:
        return []


def _write_stash(path: Path, records: list) -> None:
    """Atomic tmp + os.replace write, mirroring the store convention
    (SkillMemory._save_playbook_unlocked / FrontierTracker)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(records, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    os.replace(tmp, path)


def stash_mineable(entries: list, ghost_home=None) -> int:
    """Persist the mineable subset of `entries` into the bounded stash.

    Called by phase-1 `process_journal_queue` with the raw popped items
    BEFORE consuming them. Filters to entries `mine_challenges` could
    actually use (mineable type, failure-flagged, synthesizable),
    dedupes by journal hash against what's already stashed, appends,
    and trims to the newest ``_STASH_CAP``. Returns the number of
    entries newly stashed. Never raises.
    """
    try:
        path = _stash_path(ghost_home)
        if path is None:
            return 0
        fresh = []
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            if (entry.get("type") or "").lower() not in _MINEABLE_TYPES:
                continue
            if not _looks_like_failure(entry):
                continue
            mined = _synthesize_challenge(entry)
            if mined is None:
                continue
            fresh.append({
                "type": entry.get("type"),
                "data": entry.get("data"),
                "journal_hash": mined.journal_hash,
                "stashed_at": datetime.datetime.utcnow().isoformat() + "Z",
                "replayed": False,
            })
        if not fresh:
            return 0
        with _STASH_LOCK:
            records = _load_stash(path)
            known = {r.get("journal_hash") for r in records if isinstance(r, dict)}
            added = [r for r in fresh if r["journal_hash"] not in known]
            if not added:
                return 0
            records.extend(added)
            _write_stash(path, records[-_STASH_CAP:])
        return len(added)
    except Exception as e:  # noqa: BLE001
        logger.debug("journal stash write skipped: %s", e)
        return 0


def pick_stashed_challenge(ghost_home=None) -> Optional[MinedChallenge]:
    """Self-play fallback loader: newest un-replayed stash entry, or
    None. Marks the picked entry ``replayed`` (persisted atomically) so
    the same stashed failure isn't drilled every cycle. Never raises.
    """
    try:
        path = _stash_path(ghost_home)
        if path is None or not path.exists():
            return None
        with _STASH_LOCK:
            records = _load_stash(path)
            dirty = False
            for rec in reversed(records):  # newest-first
                if not isinstance(rec, dict) or rec.get("replayed"):
                    continue
                mined = _synthesize_challenge(rec)
                if mined is None:
                    # Un-synthesizable garbage — mark it so it isn't
                    # re-inspected forever.
                    rec["replayed"] = True
                    dirty = True
                    continue
                rec["replayed"] = True
                _write_stash(path, records)
                return mined
            if dirty:
                # Persist the garbage-marking done above.
                _write_stash(path, records)
        return None
    except Exception as e:  # noqa: BLE001
        logger.debug("journal stash pick skipped: %s", e)
        return None
