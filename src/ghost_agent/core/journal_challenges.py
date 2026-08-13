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

# One-shot latch so the "stash disabled" warning (see _stash_path) is emitted
# once per process rather than on every drain.
_WARNED_NO_HOME = False


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


_LOG_SHAPE_RE = re.compile(
    # ⚠ Word boundaries closed "the login"/"blogs", but bare `log` is also
    # a VERB — "log in to the box and fix the login page" mined as a
    # log-file exercise (round-4 review, third appearance of this class).
    # The singular noun therefore needs a determiner or a qualifier in
    # front, and the verb senses "log in/into/out/a ticket" are excluded.
    r"\.log\b"
    r"|\b(?:access|error|server|system|application|app|build|run|nginx|"
    r"apache|the|a|an|this|that|these|those|my|your|our|its|their|his|her)"
    r"\s+logs?\b"
    r"|\blog\s*file\b|\blog\s+(?:analysis|data|entries)\b"
    # A bare plural is the NOUN when a data verb governs it ("parse logs",
    # "could not match logs") and the VERB when a subject noun precedes it
    # ("the service logs errors"). Requiring a determiner alone refused the
    # former, so the governing verb is accepted too.
    r"|\b(?:pars\w*|analy[sz]\w*|match\w*|check\w*|read\w*|count\w*|"
    r"search\w*|grep\w*|tail\w*|summari[sz]\w*|scan\w*|inspect\w*|"
    r"review\w*|process\w*|filter\w*|sort\w*)\s+logs?\b",
    re.IGNORECASE)


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
        return {"kind": "sql", "filename": "input.db", "explicit": True}
    if any(k in t for k in (".csv", "csv", "spreadsheet", "comma-separated")):
        return {"kind": "csv", "filename": "input.csv", "explicit": True}
    if any(k in t for k in (".json", "json", "api response", "rest api")):
        return {"kind": "json", "filename": "input.json", "explicit": True}
    # ⚠ WORD BOUNDARIES. A substring test made "the log" match "the LOGIN",
    # "the LOGIC", "the LOGOUT" and "logs" match "bLOGS", so UI-bug prose
    # ("fix the login page") became a log exercise — the incoherence
    # generator re-opened through the over-block fix (round-2 NEW-2). The
    # live stash holds UI-bug post-mortems of exactly that shape.
    if _LOG_SHAPE_RE.search(t):
        return {"kind": "log", "filename": "input.log", "explicit": True}
    # ⚠ THE FALLBACK IS THE INCOHERENCE GENERATOR (§4BE, 2026-08-13). It
    # hands ANY request a generic `input.txt`, and the mined harness then
    # promises the solver "perform the operation the user asked for, applied
    # to that file". For "Run the composed skill youtube_transcribe on this
    # URL" that promise is unmeetable: the observed run really executed the
    # macro (8 Tor exit rotations, twice — live egress from a "synthetic"
    # exercise), then satisfied the validator by printing the fixture and
    # scored passed=True. `explicit` records whether the fixture actually
    # stands in for data the request referenced; the mining path refuses
    # when it does not. The `kind`/`filename` keys are unchanged so every
    # other consumer behaves exactly as before.
    # ⚠ ARTIFACT-level evidence only. A first cut accepted any of
    # {file,line,row,record,content,output,…} anywhere in the prose, which
    # let a live chess prompt and "use your file_system tool" qualify as
    # "text data". The fixture may only stand in for something the request
    # actually names as DATA.
    _texty = any(k in t for k in (
        ".txt", ".md", ".log", ".csv", ".json", ".xml", ".yaml", ".yml",
        "the file", "this file", "that file", "a file called", "text file",
        "the text", "this text", "the data", "this data", "the contents",
        "the output of", "these lines", "the lines", "each line",
        "the records", "the rows", "the entries", "the document",
        "the transcript", "the report", "line by line"))
    return {"kind": "text", "filename": "input.txt", "explicit": _texty}


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


# A mined challenge is a REAL past user message replayed VERBATIM to a solver
# that holds the real toolset — live Postgres, the real filesystem, the real
# service supervisor. Anything the operator once asked for destructively
# therefore becomes an UNATTENDED destructive action with nobody in the loop,
# and self-play runs at journal_prob=0.75 once the frontier saturates.
#
# Found live, un-replayed and one pick away from execution: "Run this EXACT
# SQL via postgres_admin action=query, do not modify it ... SELECT 1; DROP
# TABLE web_order_line_options_old;". The only thing standing in front of it
# was `validators.py`'s multi-statement gate — a single generic guard, doing
# a job it was never scoped for.
#
# Practising a destructive request teaches nothing that a read-only variant
# does not, so these are dropped rather than rewritten.
_UNSAFE_CHALLENGE_RE = re.compile(
    r"\b("
    r"drop\s+(table|database|schema|index|view|role|user)"
    r"|truncate\s+(table\s+)?\w"
    r"|delete\s+from\b"
    r"|alter\s+(table|database|schema|role|user)\b"
    r"|update\s+\w+\s+set\b"
    r"|insert\s+into\b"
    r"|grant\s+\w+\s+(on|to)\b|revoke\s+\w+\s+(on|from)\b"
    r"|rm\s+-[rRf]{1,3}\b|rmdir\b|mkfs\b|dd\s+if=|shred\b"
    r"|git\s+(push|reset\s+--hard|clean\s+-[a-z]*f)"
    r"|launchctl\s+(bootout|unload)|systemctl\s+(stop|disable)"
    r"|kill(all)?\s+-9|pkill\b"
    r"|chmod\s+-R\b|chown\s+-R\b"
    r"|>\s*/dev/(sd|disk)"
    r")",
    re.IGNORECASE,
)

# The verbatim-execution framing is independently disqualifying: it exists to
# defeat exactly the judgement the solver would otherwise apply.
_VERBATIM_EXEC_RE = re.compile(
    r"(exact(ly)?|verbatim|as-is|do\s+not\s+(modify|change|alter|edit))",
    re.IGNORECASE,
)


# ── Transferability gate (§4BE, 2026-08-13) ─────────────────────────────
# A mined challenge wraps the original request in a FIXED harness: "a
# `<fixture>` has been written in your working directory — write a
# solution.py that opens it and performs the operation the user asked for,
# APPLIED TO THAT FILE". That framing is only coherent when the original
# operation is DATA PROCESSING. For a request that acts on the live world
# ("Run the composed skill youtube_transcribe on this URL"), "apply it to
# input.txt" is meaningless, and the observed result (2026-08-12 17:36) was
# the worst of both: the solver ran the REAL macro — 8 Tor exit rotations,
# twice, genuine egress from a "synthetic" exercise — then satisfied the
# validator by printing the fixture's lines, scoring passed=True and minting
# a lesson from an exercise that never cohered.
#
# Both halves of that are label noise: a fabricated PASS enters frontier
# scoring, and the lesson extractor mines "genuine lessons" from a run whose
# task and grader disagreed. Refusing to mine is strictly better than
# grading an incoherent exercise.
#
# ⚠ FALSE-POSITIVE DIRECTION IS THE SAFE ONE: refusing a minable challenge
# costs one skipped self-play cycle (the template bank and LLM-gen still
# supply material); accepting a non-transferable one writes a fake label
# into the corpus AND can touch the network. Tuned against the live
# trajectory corpus — see tests/test_label_noise_leaks_4be.py.
_NON_TRANSFERABLE_RE = re.compile(
    r"\b("
    # invoking the agent's own tools / macros by name
    r"run\s+the\s+(?:composed\s+)?(?:skill|macro)|invoke\s+the\s+(?:macro|skill)"
    r"|create\s+(?:a\s+)?(?:new\s+)?(?:skill|macro|composed\s+skill|project)"
    r"|define\s+(?:a\s+)?(?:new\s+)?(?:composed\s+)?skill"
    r"|manage_composed_skills|youtube_transcribe|self[_\s-]?play"
    # network / fetch work that cannot be replayed against a local fixture
    r"|download|upload|scrape|crawl|browse|screenshot"
    r"|search\s+(?:the\s+)?(?:web|internet|online)"
    r"|https?://|www\.|\.com\b|\.gr\b|youtube"
    # live services / infrastructure
    r"|restart\s+the\s+(?:service|server|agent|daemon)"
    r"|start\s+the\s+(?:service|server)|deploy\b"
    r"|systemctl|launchctl|ssh\b|tailscale"
    # interactive UI work
    r"|click\s+(?:on\s+)?the|open\s+the\s+browser"
    # live interaction with the operator or a running game/session
    # The live chess COACHING prompt is the corpus's largest family (62
    # instances) and contains none of {chess, "you are playing"} — it opens
    # "You are Ghost, coaching Vasilis" and talks about FEN/moves.
    r"|you\s+are\s+playing|live\s+(?:game|chess|session)|chess|coaching"
    r"|next\s+move|your\s+move|\bfen\b"
    # driving the agent's OWN tools rather than processing data
    r"|use\s+your\s+\w+\s+tool|your\s+sandbox|file_system\s+tool"
    # open-ended web lookups ("search what X will be", "dark web search")
    r"|dark\s*web|fact[\s-]?check|latest\s+version"
    # `search` is egress ONLY when it is not aimed at local data
    r"|search(?!\s+(?:the\s+|this\s+|that\s+)?(?:log|file|csv|json|db|"
    r"database|table|data|records?|text|output|manual))"
    # naming any of the agent's own tools (or a tool-call literal) means the
    # task is about DRIVING THE AGENT, not processing a fixture
    r"|knowledge[\s_-]?base|introspect|postgres_admin|manage_projects"
    r"|web[\s_-]?search|file[\s_-]?system|delegate|action\s*=|in\s+the\s+sandbox"
    r"|in\s+project\b|your\s+(?:workspace|system|tools?)\b"
    # research/lookup questions are network work however they are phrased
    r"|deep\s+research|\bresearch\b|what\s+(?:new\s+)?features"
    r"|pg_stat|show\s+me\s+all\s+(?:databases|tables)"
    # asking about the agent's own live state — not reproducible offline
    r"|(?:your|my|all|the)\s+(?:custom\s+|acquired\s+|composed\s+)?"
    r"(?:memory|lessons?|skills?|macros?|profile|uptime|projects?)\b"
    r")",
    re.IGNORECASE,
)


# The harness promises the solver "performs THE OPERATION the user asked
# for". A declarative statement ("I am located in Athens", "My hobbies
# are …") names no operation, so the exercise degenerates to "print the
# fixture" and its PASS says nothing. Require an actual task verb — an
# allowlist, because "is this a task?" is semantic and a denylist of
# non-tasks would leak (the project's own rule).
_TASK_VERB_RE = re.compile(
    # ⚠ STEMS, not whole words. The first cut matched `parse\b`, which does
    # not match "parsing" — the stash fixture "First stashed failure about
    # json parsing" was refused and four stash tests went red. Inflection is
    # the norm in real requests ("counting", "sorted", "extracted"), so the
    # allowlist matches stems and lets the suffix fall where it may.
    # ⚠ STEM + INFLECTION + BOUNDARY. Dropping \b entirely made this
    # near-vacuous: `read` matched "ready", `count` matched "country",
    # `sum` matched "summer", `list` matched "listen" — 15 of 16 declarative
    # non-tasks became "tasks" (round-2 review NEW-1). The suffix group is
    # what allows "parsing" while still requiring the word to END there.
    r"\b(?:pars|count|list|find|search|extract|summari[sz]|sort|filter|"
    r"comput|calculat|convert|transform|check|read|"
    r"process|generat|writ|build|fix|debug|test|compar|merg|split|"
    r"group|aggregat|renam|replac|clean|validat|report|show|print|"
    r"detect|match|join|render|pick|select|"
    # quantity / aggregate phrasings that carry the operation without a
    # classic imperative verb ("how many rows have status=error?",
    # "what is the average of the value column", "top 10 IPs")
    r"averag|median|total|duplicat|analys|analyz)"
    # NEW-4: `per \w+` and `which \w+` are not operations ("as per usual
    # the csv is attached", "the csv, which is why I asked"). Dropped —
    # "which entries are duplicated?" is carried by the `duplicat` stem.
    r"(?:e|es|ed|ing|s|is)?\b"
    # phrasings that carry the operation without an imperative verb
    r"|\bhow\s+many\b|\btop\s+\d+\b|\bturn\s+th(?:is|at)\b"
    # irregular inflections, spelled out so the shared suffix group stays
    # tight (folding in y/ies would make `read`+`y` match "ready" again)
    # ⚠ y-STEMS and DOUBLED-CONSONANT gerunds cannot ride the shared
    # suffix group — `verif`+`y` and `identif`+`y` were DEAD verbs (round-4
    # review), silently refusing "verify that the csv has no missing values"
    # and one real corpus entry. Adding `y` to the shared group instead
    # would re-admit `read`+`y` = "ready", so they are spelled out.
    r"|\bquer(?:y|ies|ied|ying)\b|\banalys(?:is|es)\b"
    r"|\bverif(?:y|ies|ied|ying)\b|\bidentif(?:y|ies|ied|ying)\b"
    r"|\bdebug(?:s|ged|ging)?\b|\bsplit(?:s|ting)?\b"
    r"|\bmap(?:s|ped|ping)?\b|\bdedup(?:e|es|ed|ing|licat(?:e|es|ed|ing))?\b"
    r"|\bbreakdown\b|\bplot(?:s|ted|ting)?\b|\bgrep\b|\btail\b",
    re.IGNORECASE,
)


def _is_transferable_challenge(text: str) -> bool:
    """True when the original request can honestly be re-posed as
    'do this to the fixture in your working directory'.

    The mined harness demands a `solution.py` over a local file. A request
    to run a macro, hit the network, restart a service or drive a browser
    cannot be applied to a file — grading such a run produces a fabricated
    PASS and a lesson from an incoherent exercise (§4BE). Nor can a
    declarative statement that asks for no operation at all."""
    t = text or ""
    if _NON_TRANSFERABLE_RE.search(t):
        return False
    return bool(_TASK_VERB_RE.search(t))


def _is_unsafe_challenge(text: str) -> bool:
    """True when a mined user message must never be replayed by self-play."""
    if not text:
        return False
    if _UNSAFE_CHALLENGE_RE.search(text):
        return True
    # "run this exactly, do not modify it" + any SQL/shell verb.
    if _VERBATIM_EXEC_RE.search(text) and re.search(
            r"\b(sql|query|command|shell|bash|psql|postgres_admin|execute)\b",
            text, re.IGNORECASE):
        return True
    return False


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
    if _is_unsafe_challenge(user):
        logger.warning(
            "self-play: refusing to mine a destructive challenge from the "
            "journal (%r...) — a replayed user message runs against the LIVE "
            "toolset", user[:80])
        return None
    # §4BE: the harness can only pose DATA-PROCESSING work. Checked on the
    # raw user text, before anonymisation, so a stripped URL cannot hide a
    # network task from the gate.
    if not _is_transferable_challenge(user):
        logger.info(
            "self-play: skipping a NON-TRANSFERABLE journal challenge "
            "(%r...) — the mined harness asks for a solution.py over a local "
            "fixture, and this request acts on the live world, so grading it "
            "would fabricate a PASS and mine a lesson from an incoherent "
            "exercise", user[:80])
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
    if not shape.get("explicit"):
        logger.info(
            "self-play: skipping a journal challenge with NO data signal "
            "(%r...) — the harness would materialise a generic input.txt and "
            "then ask the solver to apply an operation that was never about a "
            "file, which grades an incoherent exercise (§4BE)", cleaned[:80])
        return None
    setup_script = _shape_specific_setup(shape["kind"])
    validation_script = _shape_specific_validator(shape["kind"], shape["filename"])

    challenge_prompt = (
        "You previously struggled with the request below.\n"
        f"A deterministic `{shape['filename']}` fixture has been written in "
        "your working directory — its shape matches what the original "
        # ⚠ THE PARENTHETICAL USED TO READ "(CSV/JSON/log/SQL/text)" AND THAT
        # ONE WORD MISFILED EVERY MINED CHALLENGE (2026-08-11, §4AT-B).
        # `classify_cluster` tests `\bsql\b` FIRST, so the boilerplate itself —
        # not the task — decided the cluster: 23 of 23 stored journal runs
        # classified as `sql`, including CSV, log, JSON and shell tasks, and
        # `sql` climbed to expert tier on work that was never SQL. The list is
        # descriptive prose for the model; it does not need the literal token
        # that the classifier keys on.
        "user task referenced (tabular, structured, log-style or plain "
        "text).\n"
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
        # §4Q Lens-B flagged that returning None here makes BOTH stash_mineable
        # and pick_stashed_challenge permanently inert on a deployment that
        # doesn't export GHOST_HOME — the silent-inoperative-subsystem class.
        # The complaint is right, but the FIX IS NOT to fall back to the
        # agent's `~/ghost_llamacpp` default: this module is imported by test
        # runs and by any tool process, and a home-relative fallback would make
        # them WRITE INTO THE USER'S REAL HOME. `tests/test_dream_bugfixes_
        # 2026_07_20.py::test_stash_disabled_without_home` pins that protection
        # deliberately (it caught exactly that regression on the first attempt).
        # So: stay disabled — but stop being SILENT about it, which was the
        # actual defect. Warn once per process, not per call, so a hot loop
        # can't spam the operator's stream.
        global _WARNED_NO_HOME
        if not _WARNED_NO_HOME:
            _WARNED_NO_HOME = True
            logger.warning(
                "GHOST_HOME is not set — the self-play journal stash is "
                "DISABLED (stash_mineable/pick_stashed_challenge are no-ops), "
                "so the journal-replay curriculum will not run. Export "
                "GHOST_HOME to enable it.")
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
