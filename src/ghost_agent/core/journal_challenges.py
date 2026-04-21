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

import hashlib
import re
from dataclasses import dataclass
from typing import List, Optional


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


def _synthesize_challenge(entry: dict) -> Optional[MinedChallenge]:
    """Turn one journal `post_mortem` / `failure` entry into a
    self-contained challenge.

    Strategy:
      * Use the user-message as the task prose (anonymised to remove
        names / paths we can't reproduce).
      * Generate a deterministic mock text file that captures the SHAPE
        of whatever the agent was working with (so the solver has SOME
        data to read). We can't faithfully reproduce production data
        here, so we build a small, bounded `input.txt` and ask the
        solver to demonstrate the REQUESTED operation on it.
      * The validator treats any exit-0 solution that produces non-
        empty, well-formed output as a pass. This is intentionally
        lenient — we're practising the APPROACH, not grading a
        specific answer. Frontier scoring + the lesson-extractor still
        capture the real signal.
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

    # The setup script writes a tiny deterministic input.txt that the
    # solver is expected to read. Keep this stdlib-only and small —
    # the self-play sandbox asserts stdlib-only in setup scripts.
    setup_script = (
        "with open('input.txt', 'w') as f:\n"
        "    f.write('line 1: alpha 100\\n')\n"
        "    f.write('line 2: beta 200\\n')\n"
        "    f.write('line 3: gamma 300\\n')\n"
        "    f.write('line 4: delta 400\\n')\n"
    )

    # A deliberately lenient validator: the solver must produce SOME
    # stdout, exit 0, and demonstrate that it read input.txt (checked
    # by matching at least one of the tokens present in the file).
    # We accept any reasonable interpretation of the task — this is
    # pretending to be the real problem, not a graded test.
    validation_script = (
        "import subprocess, sys\n"
        "res = subprocess.run(['python3', 'solution.py'], capture_output=True, text=True, timeout=15)\n"
        "if res.returncode != 0:\n"
        "    print('EXIT', res.returncode, 'STDERR', res.stderr[:400])\n"
        "    sys.exit(1)\n"
        "out = (res.stdout or '').strip()\n"
        "if not out:\n"
        "    print('EMPTY OUTPUT — solution must print something')\n"
        "    sys.exit(1)\n"
        "tokens = {'alpha', 'beta', 'gamma', 'delta', '100', '200', '300', '400', 'line'}\n"
        "if not any(t in out for t in tokens):\n"
        "    print('solution output did not reference any token from input.txt')\n"
        "    sys.exit(1)\n"
        "sys.exit(0)\n"
    )

    challenge_prompt = (
        "You previously struggled with the request below.\n"
        "A deterministic `input.txt` file (4 numbered lines, each with a word "
        "and a number) has already been written in your working directory.\n"
        "Write a `solution.py` that:\n"
        "  1. Opens `input.txt`, reads the lines.\n"
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
    entries. Only failure-flagged post_mortem entries are considered.
    Dedupes by challenge hash so repeated similar journal entries don't
    produce N near-identical challenges.
    """
    out: List[MinedChallenge] = []
    seen = set()
    for entry in (journal_entries or [])[-50:]:  # last 50 is plenty
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
