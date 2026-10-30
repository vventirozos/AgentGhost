"""Real per-task coding executor for autonomous batches.

The first version stopped *web-search* theatrical completion (a build task
marked DONE having only researched). But a deeper failure remained: on a
single-file app, each task REGENERATED the whole file with ``write`` and
OVERWROTE the previous task's work, while frontend tasks (no runnable shell
``verify``) had nothing to catch the regression — so three tasks "completed"
left a 2.7KB shell with neither the File Explorer nor the Snake game in it
(observed live). This module now defends against that:

  * **Edit, don't clobber.** A file entry may carry ``edits`` (find/replace)
    to ADD a feature to an existing file without re-sending the whole thing.
  * **Non-regression guard.** If a task returns FULL ``content`` for a file
    that already exists, the new content must be a SUPERSET — not smaller,
    and not dropping the file's existing identifiers. A regression is
    refused (and retried with feedback, then failed) rather than written.
  * **Frontend gate.** When a task has no shell ``verify`` and writes HTML,
    the file is headless-rendered and an uncaught JS exception fails it — a
    shell can't verify a browser UI, so without this a hollow page passes.
  * **Retry with feedback.** A rejected write or failed verify is regenerated
    once with the failure reason, then the task fails loudly (stopping the
    batch loop) instead of being marked DONE on shallow output.

Still bounded — the project task tree does the decomposition; this builds ONE
leaf well (a spec call, one streamed call per ``generate`` file — §4SD —, N
writes/edits, one verify), up to ``MAX_ATTEMPTS`` times within the leaf's
time budget (``_leaf_deadline``).
"""

from __future__ import annotations

import ast
import asyncio
import contextvars
import json
import logging
import os
import re
import time
import weakref
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("GhostAgent")

ToolRunner = Callable[[str, Dict[str, Any]], Awaitable[str]]

# 16 (was 8): a single leaf like "scaffold the app" can legitimately emit a
# dozen+ files in one coherent spec from a capable model; 8 silently dropped the
# overflow. The task tree still does the coarse decomposition, so this only
# bounds runaway specs, not normal multi-file work.
MAX_FILES = 16
# §4SD (2026-10-29, operator: "disable the coding executor's 30k-char
# abort"): the 30K-char reasoning ceiling of the streamed spec call is gone,
# like the main loop's 200K one. It was a length abort for an older model
# family; a think phase is now bounded by the spec call's own max_tokens, and
# a budget eaten with no content takes the same no-think retry the ceiling
# did. The n-gram loop probe stays.
# Upper bound on a written file. Must comfortably exceed a fully-accreted
# single-file app: with append, the result is old+new, and old can be ~200 KB
# (the gather cap). A tight 60 KB truncated the growing index.html mid-JS,
# cutting off </body></html> and breaking the page once the OS got big.
MAX_CONTENT_CHARS = 400_000
# 4 (was 2): one spec call + 3 feedback retries. A capable model often fixes a
# verify failure on attempt 3-4; the old hard stop at 2 converted a recoverable
# build into a batch-halting FAILURE. Each retry is fed the exact failure reason.
MAX_ATTEMPTS = 4
# Think policy (2026-09-03, §4EI). Live: a coding model that drafts the WHOLE
# file inside its think block hit the (since removed, §4SD) 30K ceiling on every leaf — 6 of 6
# thinking attempts in request 463111ad, ~115 s each, zero content — and only
# the no-think retry ever produced a spec. Thinking still earns its cost on a
# model that plans in 3-15K, so the policy is adaptive rather than a switch:
# after this many CONSECUTIVE aborted think phases (n-gram loop, or
# the whole budget eaten with no content) the spec call starts with thinking
# disabled; one clean think phase (content produced) resets the streak. The
# streak is kept PER LLM CLIENT (weak-keyed), so the one long-lived production
# client adapts once per process while test doubles never see each other.
#   GHOST_CODING_THINK_SKIP_AFTER=<n>   default 2
#   0 = never think on spec calls;  negative = never adapt (legacy behaviour).
# A process restart (deploy), a clean think phase, or (§4FJ) a successful
# probe think after GHOST_CODING_THINK_PROBE_AFTER no-think calls resets it.
THINK_SKIP_AFTER_ABORTS_DEFAULT = 2
# Rejected spec outputs are kept VERBATIM under
# $GHOST_HOME/system/coding_executor_failures/ (newest UNPARSED_SPEC_KEEP
# files). The 400-char log preview could not say WHY four complete-looking
# retry outputs failed to parse in 463111ad — and GHOST_LLM_RECORD is off in
# production — so the next rejection must leave the whole text behind.
UNPARSED_SPEC_KEEP = 30
# A COMPLETELY empty upstream response (content=0 reasoning=0) is contention/
# infra, not the model failing — feedback retries can't fix it, so cap them
# separately (with a small backoff) rather than burning all MAX_ATTEMPTS
# hammering the server instantly (observed live: 8 empty calls in <3s).
MAX_EMPTY_RETRIES = 2
_EMPTY_BACKOFF_S = 1.5
_REGRESS_SHRINK_RATIO = 0.85   # a rewrite below this fraction of the old size regresses
# §4SD (operator 2026-10-29: "given that a project could handle large files …
# do it"): the spec call returned every file's COMPLETE text inside one JSON
# reply — thinking plus every file in 16,384 tokens, escaped, and a reply cut
# anywhere lost every file of the attempt. A NEW file is now named in the spec
# with ``generate`` (what it must contain) and written by its own streamed
# call as raw text: its output limit is the room the server's window leaves
# (up to 64k, never under 16k — `agent.output_cap_for_tool_turn`), and a cut
# costs only that file's tail, which is continued round by round under the
# main loop's cut-write rules (core/write_continuation). The result goes
# through `_apply_file` as ``content``, so every write guard still applies.
GENERATE_ROUNDS = 6            # continuation rounds after the first call
# A round's output limit is also what the time left can hold (review r9: a
# 16k round — ~18 min — started 3 min before an unattended step's cut, and
# the files of the attempt were lost): tokens ≤ (left − reserve) × a slow
# rate; under the minimum the file fails as out of time. The reserve keeps
# room for verify and the build gates after the last file.
GENERATE_TOK_PER_S = 12.0      # slower than the dense main's measured 15–30
GENERATE_RESERVE_S = 300.0
GENERATE_MIN_TOKENS = 2048
#: the least time a new attempt (a spec call, then the files) starts with
GENERATE_MIN_ATTEMPT_S = GENERATE_RESERVE_S + GENERATE_MIN_TOKENS / GENERATE_TOK_PER_S
_GENERATE_HEAD_CHARS = 1200    # what a continuation round is shown of the file
_GENERATE_TAIL_CHARS = 1500
_REPEAT_TAIL_LINES = 40         # r19: the cut round's tail the repetition test reads
# A leaf's wall clock: background work has no request deadline, and a 64k
# call is ~73 min at ~15 tok/s. `GHOST_LEAF_BUILD_MAX_S`, default 2 h; an
# unattended step's own cap (project_advancer) lowers it via
# `leaf_deadline_context`. A call may take the window's room only when the
# time left could hold it (the main loop's `turn_room_fits_time` rule).
LEAF_BUILD_MAX_S_DEFAULT = 7200.0
#: monotonic deadline set by the caller that bounds this leaf (0 = none)
leaf_deadline_context: contextvars.ContextVar = contextvars.ContextVar(
    "coding_leaf_deadline", default=0.0)
#: §4SD open items r5: does the caller wait for this build when it is stopped?
#: The unattended step does not (it reopens the leaf at once), so a stopped
#: build there must not write late (project_advancer `_with_leaf_deadline`)
caller_waits_context: contextvars.ContextVar = contextvars.ContextVar(
    "coding_caller_waits", default=True)


def _stop_requested() -> bool:
    """§4SD open items r5 (review MAJOR): the turn's tool stop was sent — the
    build's verify run was killed by it (exit 130), so what follows is a stop,
    not a failed attempt (a FAILED leaf the reopen could not undo)."""
    try:
        from ..utils import tool_stop as _ts
        return _ts.requested()
    except Exception:  # noqa: BLE001
        return False


def _leaf_budget_s() -> float:
    try:
        v = float(os.getenv("GHOST_LEAF_BUILD_MAX_S") or LEAF_BUILD_MAX_S_DEFAULT)
    except ValueError:
        return LEAF_BUILD_MAX_S_DEFAULT
    return v if 0 < v < float("inf") else LEAF_BUILD_MAX_S_DEFAULT


def _leaf_deadline() -> float:
    """The monotonic time this leaf must be done by: its own budget, never
    past a caller's deadline. The request it runs inside is read live, call
    by call (`_time_left_s`)."""
    out = time.monotonic() + _leaf_budget_s()
    try:
        outer = float(leaf_deadline_context.get() or 0.0)
    except (TypeError, ValueError):
        outer = 0.0
    return min(out, outer) if outer > 0 else out


def _request_left_s() -> Optional[float]:
    """The most time the request this leaf runs inside can still have: its
    ceiling WITH every extension it could earn (`extension_ceiling_s`), less
    what it has spent; None = no bound (background work; the web and the CLI
    with no ceiling). Review r9: an owner's Slack "do the next task" — a
    30-min request — took a 64k (~73 min) file call; r10: the remaining time
    of the CURRENT deadline froze a web request's leaf at its base clock,
    though each written part earns it more (`_note_round_progress`)."""
    try:
        from ..utils import logging as _glog
        rid = str(_glog.request_id_context.get() or "")
        ceil = float(_glog.extension_ceiling_s(rid) or 0.0)
        if not (0 < ceil < float("inf")):
            return None
        spent = _glog.request_elapsed_s(rid)
        return None if spent is None else ceil - float(spent)
    except Exception:  # noqa: BLE001 — no request clock = the leaf's own
        return None


def _time_left_s(deadline: float) -> float:
    left = deadline - time.monotonic()
    req = _request_left_s()
    return min(left, req) if req is not None else left


def _note_round_progress(path: str) -> None:
    """§4RA: a generated part of a CODE file is coding progress, as a continued
    write's part is in the main loop (`write_continuation._note_progress`) —
    a request earns its extension while the leaf writes."""
    try:
        from ..utils import logging as _glog
        from .write_continuation import _note_progress
        _note_progress(str(_glog.request_id_context.get() or ""), path)
    except Exception:  # noqa: BLE001
        pass


@dataclass
class CodingResult:
    ok: bool
    summary: str
    files: List[str] = field(default_factory=list)
    ledger_note: str = ""
    detail: str = ""
    #: §4PG (R2-1): a verify command the leaf gave RAN and exited 0 — a
    #: SUMMARY claim is not evidence; unattended, an unverified build is held
    verified: bool = False
    #: §4SD: the leaf's time ran out (not the build) — the advancer puts the
    #: task back to READY as for a timed-out step, not FAILED
    out_of_time: bool = False


def _short(text: str, n: int = 180) -> str:
    return " ".join((text or "").split())[:n]


def _replace_failure_kind(out: str) -> str:
    """Why a `file_system` replace did not apply — OBSERVED, not guessed.

    The caller feeds this into the next attempt's prompt behind "YOUR PREVIOUS
    ATTEMPT FAILED:", so a wrong label costs retries. `file_system` rejects a
    replace for several distinct reasons and only one of them is a missing
    anchor; calling them all "anchor not found" sends the model looking for a
    better SEARCH block when the defect is in its REPLACE block.

    Unknown shapes deliberately return a NEUTRAL phrase rather than the most
    likely guess — "did not apply" plus the tool's own text is honest, and the
    raw text is always appended by the caller anyway.
    """
    head = (out or "")[:400].lower()
    if "syntax error" in head or "would introduce a syntax" in head:
        return "the REPLACEMENT breaks syntax — fix the replacement, not the anchor"
    # ⚠ MISSING FILE BEFORE MISSING ANCHOR (review round 4). `file_system`
    # returns `Error: '<f>' not found.` when the TARGET DOES NOT EXIST
    # (file_system.py:1362) — which the generic "not found" rule below claimed
    # as an anchor miss, steering the model to hunt for a better SEARCH block
    # against a path that isn't there. An unfixable hypothesis that burns the
    # whole retry budget, and path desync is a known live failure mode here.
    # This is the very defect the function was written to remove, reproduced
    # inside the fix.
    if "not found." in head and "block" not in head:
        return "the FILE does not exist at that path — check the path first"
    if "multiple instances" in head or "ambiguous" in head:
        return "anchor is AMBIGUOUS — include more surrounding context"
    # ⚠ THESE LITERALS ARE COPIED FROM `file_system.py`, NOT IMAGINED. The
    # first version invented "none of the blocks matched" / "no blocks
    # matched" / "did not match" — none of which appear anywhere in the tree.
    # The real multi-block failure is
    #   "SYSTEM INSTRUCTION: None of the SEARCH/REPLACE blocks matched in '<f>'"
    # (file_system.py:1671), with per-block detail "Could not find block:"
    # (:1637). The most common anchor failure was falling through to the
    # neutral branch while the correct one sat directly above it.
    if ("blocks matched" in head or "could not find block" in head
            or "no match" in head or "not found" in head):
        return "anchor not found"
    if "marker" in head or "<<<<" in head or "====" in head:
        return "the block contains SEARCH/REPLACE markers"
    if "system block" in head or "not applied" in head:
        return "the write was BLOCKED"
    return "see the tool output"


def _op_ok(out: str) -> bool:
    """A file_system op reports ``SUCCESS: …`` on success; a no-match replace
    returns ``SYSTEM INSTRUCTION: … NOT found``. So success == a SUCCESS head.

    Anchored on the HEAD, not a substring: the no-match message embeds the
    filename (``… NOT found in 'payment_success.html'``), so a SUBSTRING
    test for "success" matched a failed replace on any ``*success*``-named
    file (common in web deliverables) and counted the un-applied edit as
    applied — closing the leaf DONE with the file unchanged. Mirrors the
    anchored discipline of the sibling ``_looks_like_write_error``."""
    # A migrated tool ANSWERS this. Its sibling below was the FIFTH reader
    # of the same question and the only one this round's sweep missed — four
    # were migrated and it was not, which is the round-over-round pattern
    # this work keeps reproducing. PARTIAL still counts as written: a file
    # that landed but does not parse HAS changed, and the syntax diagnostic
    # is what reports that.
    _st = getattr(out, "status", None)
    if _st is not None:
        _sv = str(getattr(_st, "value", _st))
        if _sv in ("rejected", "failed"):
            return False
        if _sv == "partial":
            return True
    head = (out or "").strip()[:80].lower()
    return head.startswith("success")


def _looks_like_write_error(out: str) -> bool:
    """Conservative write-failure check (writes rarely fail; don't abort a good
    build on a chatty success message). ``SYSTEM INSTRUCTION:`` heads are
    file_system REFUSALS (e.g. the empty-content write refusal) — missing them
    closed tasks DONE with the file never written (2026-07-20 review).

    Reads the STATUS first. It matched three lowercase heads and ignored the
    status entirely, so on the typed path 36 live refusals still slipped
    through — 14 ``REJECTED: that replace would introduce a syntax error``,
    11 pre-flight guard blocks, 5 empty-write blocks, 4 rejected SQL. The
    caller then does ``touched.add(path)`` and advances the task with the
    file never written, unattended. ADD-only: an ``ok`` status falls through
    to the prose rules below."""
    _st = getattr(out, "status", None)
    if _st is not None and str(getattr(_st, "value", _st)) in ("rejected",
                                                              "failed"):
        return True
    head = (out or "").strip()[:80].lower()
    return (
        not out
        or head.startswith("system error")
        or head.startswith("system instruction")
        or head.startswith("error:")
        # Anchored, not substring: a SUCCESSFUL write to a file whose name
        # contains these words ("SUCCESS: Wrote … to 'security error
        # handler.py'") must not read as a write error. The real marker is
        # a "Security Error:" head raised by file_system's path guard.
        or head.startswith("security error")
    )


def _syntax_fail_reason(path: str, out: str) -> Optional[str]:
    """Extract the post-write syntax diagnostic from a file_system result,
    shaped as an apply-failure reason — or None when the file parses.

    file_system's write/replace paths append ``⚠ SYNTAX CHECK FAILED: …``
    (ast/node-backed, HTML <script> blocks included) to an otherwise-
    successful result when the file left on disk does NOT parse. The
    interactive loop reads that warning in-context and fixes it next turn;
    this executor used to discard it — observed live 2026-07-14: five
    consecutive autoadvance tasks each rewrote index.html carrying the same
    duplicate-identifier SyntaxError, every write was flagged, every task
    still closed DONE, and the broken build was only caught when the final
    turn browsed the page. Returning the diagnostic here feeds the
    retry-with-feedback loop (the model gets the exact line and can fix it
    with `edits`); on exhaust the task fails honestly instead of piling more
    features onto a file that doesn't parse.
    """
    text = str(out or "")
    idx = text.find("SYNTAX CHECK FAILED")
    if idx < 0:
        return None
    diag = " ".join(text[idx:].split())[:400]
    return (f"{path} is on disk but does NOT parse — {diag} "
            f"Fix the syntax error with `edits` (do not rewrite the whole "
            f"file); the task cannot complete while the file is broken.")


def _looks_like_missing_file(out: str) -> bool:
    """True when a file_system read says the target does not exist (as opposed
    to failing for some other reason — too large, budget-refused, IO error)."""
    head = " ".join((out or "").split()).lower()[:200]
    return head.startswith("error") and (
        "does not exist" in head or "not found" in head)


async def _read_live_file(tool_runner: ToolRunner,
                          path: str) -> Tuple[Optional[str], Optional[str]]:
    """Read ``path``'s CURRENT on-disk content through file_system. Returns
    ``(content, None)`` on success, ``("", "missing")`` when the file does not
    exist yet, and ``(None, reason)`` when the live content cannot be
    determined (too large / budget-refused / errored). Callers must NOT
    substitute the prompt snapshot in that last case: the snapshot is
    truncated by the gatherer's per-file/budget caps, and writing old+new
    reconstructed from it AMPUTATES the on-disk tail (2026-07-20 review)."""
    try:
        out = await tool_runner("file_system", {"operation": "read", "path": path})
    except Exception as e:
        return None, f"read errored: {e}"
    text = str(out or "")
    # §4QA-d r20: a project-scoped miss is answered with the sandbox-ROOT
    # file of the same name ("NOTE: … serving the file at the sandbox ROOT"):
    # that is NOT this project's file — taken as the baseline it refused a
    # new main.py as "smaller than the file that already exists", and an
    # append built on the root file's content
    if text.startswith("NOTE:") and "serving the file at the sandbox ROOT" in text[:600]:
        return "", "missing"
    marker = f"--- {path} CONTENTS ---\n"
    idx = text.find(marker)
    if idx != -1:
        return text[idx + len(marker):], None
    if _looks_like_missing_file(text):
        return "", "missing"
    return None, _short(text, 140)


async def _is_root_fallback(tool_runner: ToolRunner, path: str) -> bool:
    """§4QA-d r21: True when ``path`` is missing from the project and
    file_system would serve the sandbox-ROOT file of that name. A whole read
    of a big root file answers "Error: … too large" WITHOUT the fallback
    NOTE (only content replies carry it), so that error refused to create a
    NEW project file. A one-line ranged read carries the NOTE."""
    try:
        out = str(await tool_runner("file_system", {"operation": "read", "path": path,
                                                     "start_line": 1, "end_line": 1}) or "")
    except Exception:  # noqa: BLE001
        return False
    return out.startswith("NOTE:") and "serving the file at the sandbox ROOT" in out[:600]


async def _refresh_existing(tool_runner: ToolRunner, existing: Dict[str, str],
                            fresh: set, paths: List[str]) -> None:
    """Re-read ``paths`` from disk into the ``existing`` snapshot between retry
    attempts. A failed attempt can leave PARTIAL edits on disk (``_apply_edits``
    stops at the first bad edit, after earlier ones applied) — without this the
    retry prompt re-rendered the PRE-edit excerpt, the model re-emitted an edit
    whose anchor was already gone, and every attempt burned on "anchor not
    found" (2026-07-20 review). Best-effort: an unreadable file keeps its stale
    entry (it is only prompt material), but is dropped from ``fresh`` so the
    append path re-reads instead of trusting it."""
    for path in paths:
        live, err = await _read_live_file(tool_runner, path)
        if live is not None and err is None:
            existing[path] = live
            fresh.add(path)
        elif err == "missing":
            existing.pop(path, None)
            fresh.discard(path)
        else:
            fresh.discard(path)


# Distinctive structural identifiers (HTML ids, function/def/class/const names,
# CSS id/class selectors) used to detect that a rewrite dropped prior work.
_ANCHOR_RE = re.compile(
    r'id="([\w-]{3,})"'
    r"|id='([\w-]{3,})'"
    r"|function\s+([A-Za-z_]\w{2,})"
    r"|def\s+([A-Za-z_]\w{2,})"
    r"|class\s+([A-Za-z_]\w{2,})"
    r"|const\s+([A-Za-z_]\w{2,})"
    r"|#([\w-]{3,})\s*\{"
    r"|\.([\w-]{3,})\s*\{"
)


def _structural_anchors(text: str) -> set:
    out = set()
    for m in _ANCHOR_RE.finditer(text or ""):
        for g in m.groups():
            if g:
                out.add(g)
    return out


def _py_parse_error(text: str) -> Optional[str]:
    """The SyntaxError diagnostic for Python source ``text``, or None when it
    parses. In-process and cheap — used to refuse a write BEFORE it lands,
    because the post-write check catches the damage only after a working file
    on disk has already been replaced with a broken one."""
    try:
        ast.parse(text or "")
        return None
    except SyntaxError as se:
        return f"{se.msg} (line {se.lineno}, col {se.offset})"
    except Exception:  # noqa: BLE001 — e.g. NUL bytes; not a syntax verdict
        return None


_PY_TOPLEVEL_DEF_RE = re.compile(r"^(?:def|class)\s+([A-Za-z_]\w*)",
                                 re.MULTILINE)


def _py_append_guard(base: str, snippet: str, path: str) -> Optional[str]:
    """Why appending ``snippet`` to the WORKING Python file ``base`` must be
    refused — or None when the append is a safe extension.

    Unlike HTML (where appended <script> blocks are IIFE-isolated), Python
    concatenation has no isolation: a spec that re-emits an implementation
    for a file that already implements the task lands old+new on disk —
    duplicate defs, a second ``__main__`` block, and usually a SyntaxError
    from a snippet that assumed different context. Observed live 2026-08-01
    (Mini AI, request b7e516b9): autoadvance re-ran an already-built task,
    appended a second full implementation into a tested, working core.py,
    and left a file that failed to parse at line 379 — three repair attempts
    later the task was FAILED and the project rolled up FAILED. Only guards
    a base that PARSES: a mid-repair broken file keeps the old flow (the
    post-write check reports it with the fresh diagnostic)."""
    if not (base or "").strip() or not path.lower().endswith(".py"):
        return None
    if _py_parse_error(base) is not None:
        return None
    dup = sorted(set(_PY_TOPLEVEL_DEF_RE.findall(base))
                 & set(_PY_TOPLEVEL_DEF_RE.findall(snippet or "")))
    if dup:
        return (f"append to {path} refused: it RE-DEFINES identifiers that "
                f"already exist in the file ({', '.join(dup[:6])}) — that is "
                f"a duplicate implementation, not an extension. The file "
                f"already implements this; if the task is already done, "
                f'return {{"files": [], "verify": "<cmd that proves it '
                f'works>"}}. To CHANGE existing code use `edits`.')
    if "__main__" in (snippet or "") and "__main__" in base:
        return (f"append to {path} refused: the file already has an "
                f"`if __name__ == \"__main__\"` block — appending a second "
                f"entry point duplicates it. Use `edits` to change the "
                f"existing block, or return files: [] with a verify if the "
                f"task is already done.")
    merged_err = _py_parse_error(_smart_append(base, (snippet or "").strip(),
                                               path))
    if merged_err:
        return (f"append to {path} refused BEFORE writing: the merged file "
                f"would not parse ({merged_err}) while the current file on "
                f"disk parses cleanly. Do not break a working file — use "
                f"`edits`, or files: [] with a verify if the task is "
                f"already done.")
    return None


_ESCAPE_MAP = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "'": "'",
               "\\": "\\"}


def _unescape_common(s: str) -> str:
    """Decode the common backslash escapes (\\n \\t \\r \\" \\' \\\\) in a
    single left-to-right walk. NOT ``unicode_escape`` — that codec mangles
    non-ASCII (UTF-8 text round-trips through latin-1) and decodes escapes
    we don't want to guess about."""
    out: List[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            mapped = _ESCAPE_MAP.get(s[i + 1])
            if mapped is not None:
                out.append(mapped)
                i += 2
                continue
        out.append(c)
        i += 1
    return "".join(out)


def _maybe_unescape_double_escaped(content: str, path: str) -> Optional[str]:
    """Bounded repair for DOUBLE-ESCAPED Python spec content — the JSON
    string's escapes arrived literally, so the whole file is one line full
    of two-char ``\\n`` sequences. Observed live 2026-08-01 (req 50855398):
    data_generator.py was written this way, three executor retries re-broke
    it, the task FAILED, and the interactive loop burned ~180s discovering
    the shape. Returns the repaired text ONLY when every check passes:
    the original does not parse, it has the single-line-escaped shape, and
    the unescaped candidate DOES ``ast.parse`` — a tolerant repair must
    bound its OUTPUT, not just its trigger. Otherwise None (caller writes
    the original and the normal syntax-failure feedback loop applies)."""
    if not path.lower().endswith(".py") or not content:
        return None
    if content.count("\n") > 2 or content.count("\\n") < 5:
        return None
    if _py_parse_error(content) is None:
        return None  # parses as-is — leave it alone
    candidate = _unescape_common(content)
    if _py_parse_error(candidate) is None:
        return candidate
    return None


# Imported lazily from project_advancer so there is ONE definition of the
# marker — a second copy of the literal is exactly how the producer and the
# consumer drift apart (the defect class this file already carries scars for).
def _snap_truncated_mark() -> str:
    from .project_advancer import SNAPSHOT_TRUNCATED_MARK
    return SNAPSHOT_TRUNCATED_MARK


def _regression_reason(old: Optional[str], new: str) -> Optional[str]:
    """Why writing ``new`` over the existing ``old`` would LOSE work — or None
    if it safely extends. None for a brand-new file (no prior work to lose)."""
    if not old:
        return None
    new = new or ""
    if len(new) < len(old) * _REGRESS_SHRINK_RATIO:
        return (f"the new content ({len(new)} bytes) is smaller than the file "
                f"that already exists ({len(old)} bytes) — it discards prior "
                f"work. Do NOT rewrite the whole file; use `append` to ADD only "
                f"your new feature")
    old_anchors = _structural_anchors(old)
    if old_anchors:
        missing = sorted(a for a in old_anchors if a not in new)
        if len(missing) > max(1, int(len(old_anchors) * 0.4)):
            return (f"the new content drops existing identifiers {missing[:6]} "
                    f"— do NOT rewrite the file; use `append` to ADD your "
                    f"feature, keeping everything already there")
    return None


# Match an INLINE <script> (no src=) and capture its body for IIFE-wrapping.
_INLINE_SCRIPT_RE = re.compile(
    r"(<script\b(?![^>]*\bsrc\s*=)[^>]*>)(.*?)(</script\s*>)",
    re.IGNORECASE | re.DOTALL)

# An inline event handler (onclick="startGame()", onload="init()") — the names
# it calls MUST remain at global scope.
_INLINE_HANDLER_RE = re.compile(r"\bon\w+\s*=\s*[\"']([^\"']*)[\"']", re.IGNORECASE)
_IDENT_RE = re.compile(r"[A-Za-z_$][\w$]*")
# A script body that intentionally exposes a global (window.x = …, globalThis.x = …).
_EXPOSES_GLOBAL_RE = re.compile(r"\b(?:window|globalThis)\s*\.\s*[\w$]+\s*=", re.IGNORECASE)


def _declares_name(body: str, name: str) -> bool:
    """True if ``body`` declares ``name`` at a scope IIFE-wrapping would hide."""
    n = re.escape(name)
    return bool(
        re.search(r"\b(?:function|var|let|const|class)\s+" + n + r"\b", body)
        or re.search(r"\b" + n + r"\s*=\s*(?:function\b|\(|async\b)", body))


def _isolate_scripts(fragment: str) -> str:
    """Wrap each inline <script> body in an IIFE so an APPENDED app block can't
    redeclare another block's top-level identifiers (``function initGame``,
    ``makeDraggable``, …) — those redeclarations were SyntaxErrors that broke the
    WHOLE page on load (observed live — the verifier REFUTED the built OS as
    "throws on load").

    BUT wrapping is skipped when the body must expose globals, because IIFE-
    scoping them away SILENTLY breaks the page (observed: a strong model wires
    ``<button onclick="startGame()">`` to a top-level ``function startGame()``;
    wrapping hid it and the button did nothing, with no error the model could
    see). A body is left UNWRAPPED when it (a) assigns to ``window.``/
    ``globalThis.``, or (b) declares a name referenced by an inline ``on*=``
    handler in the same fragment. Already-wrapped bodies are left alone too."""
    needed = set()
    for m in _INLINE_HANDLER_RE.finditer(fragment):
        needed.update(_IDENT_RE.findall(m.group(1)))

    def _wrap(m):
        open_tag, body, close_tag = m.group(1), m.group(2), m.group(3)
        s = body.strip()
        if not s or s.startswith(("(function", "(()", "(async", "!function")):
            return m.group(0)
        if _EXPOSES_GLOBAL_RE.search(body):
            return m.group(0)
        if any(_declares_name(body, n) for n in needed):
            return m.group(0)
        return f"{open_tag}\n(function(){{\n{body}\n}})();\n{close_tag}"
    return _INLINE_SCRIPT_RE.sub(_wrap, fragment)


def _smart_append(old: str, new: str, path: str) -> str:
    """Place ``new`` into ``old``. For an HTML file, isolate the appended
    inline scripts (so apps can't clobber each other's globals) and insert
    just BEFORE the closing </body> (or </html>) so scripts/markup land INSIDE
    the document; otherwise append to the end. New files just get ``new``."""
    is_html = path.lower().endswith((".html", ".htm"))
    if is_html:
        new = _isolate_scripts(new)
    if not old:
        return new + "\n"
    if is_html:
        low = old.lower()
        for anchor in ("</body>", "</html>"):
            idx = low.rfind(anchor)
            if idx != -1:
                return old[:idx] + new + "\n" + old[idx:]
    return old.rstrip() + "\n\n" + new + "\n"


def _file_excerpt(content: str, head: int = 2000, tail: int = 1000) -> str:
    """A compact head+tail view of a file — enough for the model to see its
    structure and pick insertion anchors, without re-sending the whole thing."""
    content = content or ""
    if len(content) <= head + tail:
        return content
    omitted = len(content) - head - tail
    return f"{content[:head]}\n  …({omitted} bytes omitted)…\n{content[-tail:]}"


def _render_existing(existing_files: Optional[Dict[str, str]], single_file: bool) -> str:
    if not existing_files:
        return ""
    parts: List[str] = []
    for path, content in existing_files.items():
        if content:
            parts.append(f"--- {path} ({len(content)} bytes) ---\n"
                         f"{_file_excerpt(content)}")
        else:
            parts.append(f"--- {path} (exists; large) ---")
    # 40 KB (was 12 KB): with a 65 K-token context window the prompt budget is
    # large, and an 800/400 excerpt of a grown index.html hid the middle — where
    # an `edits` find/replace anchor often lives, so the model guessed an anchor
    # that byte-failed. Give it real structural visibility of the files it grows.
    body = "\n\n".join(parts)[:40_000]
    lead = (
        "\nEXISTING PROJECT FILES (excerpts) — your task ADDS to these; never "
        "recreate or shrink them. USE `append`: a file entry "
        '{"path":"index.html","append":"<self-contained <script>…</script> '
        'and/or <div>…</div> for this feature>"} — you write ONLY the new code '
        "and the system places it correctly (for HTML, inside the document "
        "before </body>). Do NOT re-send the file's existing content, and do "
        "NOT guess `edits` anchors — `append` needs no anchor and is the "
        "reliable way to grow the app. A NEW file of more than ~100 lines: "
        "`generate` it.\n"
        "SCOPING (important): your appended <script> runs in an ISOLATED scope, "
        "so it CANNOT see or clobber other tasks' globals — and they can't see "
        "yours. Therefore: (1) keep helper names local (a second `function "
        "initGame` in another block is fine, they don't collide); (2) wire your "
        "UI INSIDE your block — attach event listeners to the elements you "
        "create, or to your desktop/taskbar entry; (3) if the shell must launch "
        "you, expose ONE entry point on window with a UNIQUE name "
        "(e.g. `window.openSnake = ...`), never a generic global the shell "
        "guesses.\n"
    )
    if single_file:
        lead = ("\nThis is a SINGLE-FILE project: every task ADDS to the same "
                "growing file(s) with `append`/`edits` — never rewrite from "
                "scratch." + lead)
    return lead + body + "\n"


# A file path/name mentioned in a task description ("Implement: core.py - the
# model core"). Used to notice that the task's deliverable ALREADY EXISTS.
_TASK_NAMED_FILE_RE = re.compile(
    r"(?<![\w./-])(?:\./)*(\.?[\w][\w./-]{0,60}\.(?:py|js|mjs|cjs|json|html?|htm|css|md|"
    r"sh|yml|yaml|toml|txt))(?!\w)(?!\.\w)", re.IGNORECASE)   # §4QA-d: "…into h.py." ends a sentence


def _task_named_existing(description: str,
                         existing_files: Optional[Dict[str, str]]) -> List[str]:
    """The existing-file keys that the task description names (exact path or
    basename match). Non-empty means a previous turn probably already built
    this task's deliverable."""
    if not existing_files:
        return []
    named = {m.group(1).lstrip("./").lower()
             for m in _TASK_NAMED_FILE_RE.finditer(description or "")}
    if not named:
        return []
    out = []
    for key in existing_files:
        k = key.replace("\\", "/").lstrip("./").lower()
        if k in named or k.rsplit("/", 1)[-1] in named:
            out.append(key)
    return out


def _render_already_built(description: str,
                          existing_files: Optional[Dict[str, str]]) -> str:
    """Steer for the spec model when the task's named deliverable already
    exists. Without this, autoadvance re-picking a task that an interactive
    turn already finished re-IMPLEMENTED the file (2026-08-01 Mini AI: a
    second full core.py appended into the working one → SyntaxError → task
    FAILED → project FAILED). The executor's verify-only path ("files": [] +
    "verify") exists exactly for this; the model just never knew to take it."""
    matched = _task_named_existing(description, existing_files)
    if not matched:
        return ""
    return (
        "\nALREADY ON DISK: this task names file(s) that already EXIST in "
        "the project: " + ", ".join(sorted(matched)[:6]) + ". A previous "
        "turn may have already done this work — check their excerpts above "
        "FIRST. If the file already implements the task, return "
        '{"files": [], "verify": "<shell command that proves it works>"} — '
        "do NOT re-send or re-implement it (a duplicate implementation "
        "appended to a working file breaks it). Only emit file entries for "
        "code that is genuinely missing, and prefer `edits` for changes.\n")


def _render_constraints(constraints: Optional[List[str]]) -> str:
    """User-mandated constraints from the project record. Before this block
    the executor never saw them — the 2026-07-04 chess session's first "coded
    AI opponent" violation was written by an autoadvance leaf whose spec
    prompt contained no trace of "with YOU - Ghost plays directly, not a
    generated chess AI". §4SD: a generated file's call sees them too."""
    if not constraints:
        return ""
    from ..utils.constraints import (
        PARTICIPANT_STEER, has_participant_constraint,
        render_constraint_block,
    )
    out = "\n" + render_constraint_block(
        list(constraints), header="EXPLICIT USER CONSTRAINTS (PROJECT-WIDE)") + "\n"
    if has_participant_constraint(list(constraints)):
        out += "\n" + PARTICIPANT_STEER + "\n"
    return out


def _render_research(research_context: Optional[Dict[str, str]]) -> str:
    """Render the project's research briefs as READ-ONLY reference: the design
    decisions the agent researched and saved. Without this the build ignored
    its own research (observed live: a careful research brief written, then
    never used). Excerpts only — the model consults, does not reproduce."""
    if not research_context:
        return ""
    parts: List[str] = []
    for path, excerpt in research_context.items():
        if excerpt and excerpt.strip():
            parts.append(f"--- {path} ---\n{excerpt.strip()}")
    if not parts:
        return ""
    body = "\n\n".join(parts)[:8_000]
    return (
        "\nPROJECT RESEARCH (reference — design decisions you already "
        "researched and saved; build CONSISTENTLY with these, do NOT re-derive "
        "or contradict them; these are NOT files to edit):\n" + body + "\n")


def _repair_json_string_newlines(text: str) -> str:
    """Escape literal control characters inside JSON string literals.

    The dominant malformed-spec shape in the live logs (reqs 92a968fc,
    8ec512cf — three occurrences in one afternoon): the model emits an
    otherwise-valid files spec whose ``find``/``append``/``content``
    string values contain RAW newlines/tabs instead of ``\\n``/``\\t``,
    which is invalid JSON, so every parse attempt fails and the whole
    generation is retried with feedback (~30-60s each). A simple
    in-string state walk (quote toggling, backslash escapes honoured)
    turns the literal control chars into their escape sequences; other
    defects (unescaped inner quotes) are left alone — this is a
    best-effort last resort reached only after normal parsing failed."""
    out: List[str] = []
    in_str = False
    esc = False
    for ch in text:
        if in_str:
            if esc:
                esc = False
                out.append(ch)
                continue
            if ch == "\\":
                esc = True
                out.append(ch)
                continue
            if ch == '"':
                in_str = False
                out.append(ch)
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
            out.append(ch)
        else:
            if ch == '"':
                in_str = True
            out.append(ch)
    return "".join(out)


def _recover_spec_escaping_newlines(channels, extract, usable) -> dict:
    """Re-parse each raw channel after escaping literal control chars
    inside string literals, over the JSON candidate region (first ``{``
    to last ``}`` — leading prose would desync the quote walk).
    Returns the first usable spec, else ``{}``."""
    for raw in channels:
        if not raw or "{" not in raw:
            continue
        start, end = raw.find("{"), raw.rfind("}")
        if end <= start:
            continue
        cand = extract(_repair_json_string_newlines(raw[start:end + 1]),
                       repair_truncated=True) or {}
        if usable(cand):
            logger.info(
                "coding_executor: spec recovered by escaping literal "
                "newlines inside JSON string values")
            return cand
    return {}


# ---------------------------------------------------------------------------
# Think policy — see THINK_SKIP_AFTER_ABORTS_DEFAULT
# ---------------------------------------------------------------------------

_THINK_STATE: "weakref.WeakKeyDictionary[Any, Dict[str, int]]" = weakref.WeakKeyDictionary()
_THINK_STATE_FALLBACK: Dict[int, Dict[str, int]] = {}


def _think_skip_after() -> int:
    raw = (os.getenv("GHOST_CODING_THINK_SKIP_AFTER") or "").strip()
    if not raw:
        return THINK_SKIP_AFTER_ABORTS_DEFAULT
    try:
        return int(raw)
    except ValueError:
        return THINK_SKIP_AFTER_ABORTS_DEFAULT


# §4FJ (2026-09-07): the disarm path. Once the streak had tripped, "a clean
# think phase resets" could never happen — no think phase ran any more — so
# the no-think regime was sticky until a restart, and the leaf bench measured
# that regime at 2/6 DONE against 6/6 for think-with-fallback in the same
# process (journal §4FI). Now, after GHOST_CODING_THINK_PROBE_AFTER (default
# 3) no-think spec calls — or immediately after a verify failure on a spec
# that was built no-think — ONE spec call thinks again as a probe: a clean
# probe resets the streak (thinking is back), an aborted probe extends the
# no-think window by another PROBE_AFTER calls. <= 0 disables probing (the
# pre-§4FJ sticky behaviour).
THINK_PROBE_AFTER_DEFAULT = 3


def _think_probe_after() -> int:
    raw = (os.getenv("GHOST_CODING_THINK_PROBE_AFTER") or "").strip()
    if not raw:
        return THINK_PROBE_AFTER_DEFAULT
    try:
        return int(raw)
    except ValueError:
        return THINK_PROBE_AFTER_DEFAULT


# Per-client policy record. `skipped_think_last`: the LAST spec call did not
# ATTEMPT a think phase (it went straight to the no-think call). After a
# probe that aborted and fell back to the no-think retry this is 0 — a think
# phase WAS attempted — which is exactly what stops the verify hook from
# re-probing at once (see _note_nothink_verify_failure).
_THINK_KEYS: Dict[str, int] = {"aborts": 0, "skipping": 0, "nothink_calls": 0,
                               "probe": 0, "skipped_think_last": 0}


def _think_state(llm) -> Dict[str, int]:
    """Per-client streak record; keys as in ``_THINK_KEYS``. A record created
    before §4FJ (two keys) is migrated in place with the safe defaults —
    a fresh window, no pending probe."""
    try:
        st = _THINK_STATE.get(llm)
        if st is None:
            st = dict(_THINK_KEYS)
            _THINK_STATE[llm] = st
    except TypeError:  # not weak-referenceable / unhashable double
        st = _THINK_STATE_FALLBACK.setdefault(id(llm), dict(_THINK_KEYS))
    for k, v in _THINK_KEYS.items():
        st.setdefault(k, v)
    return st


def _probe_due(llm) -> bool:
    """In the tripped regime, has the no-think window elapsed?"""
    k = _think_probe_after()
    return k > 0 and _think_state(llm)["nothink_calls"] >= k


def _think_disabled_now(llm) -> bool:
    """Should this spec call skip the think phase entirely? A pure predicate
    (§4FJ review M-2): arming the probe is the CALL SITE's job
    (`_arm_probe`), so a telemetry reader can ask without side effects."""
    n = _think_skip_after()
    if n < 0:
        return False
    if n == 0:
        return True
    st = _think_state(llm)
    if st["aborts"] < n:
        return False
    return not _probe_due(llm)


def _arm_probe(llm) -> None:
    """The spec call about to stream a think phase is the scheduled probe."""
    st = _think_state(llm)
    st["probe"] = 1
    logger.warning(
        "coding_executor: probe think phase after %d no-think spec calls "
        "(GHOST_CODING_THINK_PROBE_AFTER=%d) — a clean think re-enables "
        "thinking, an aborted one extends the no-think window",
        st["nothink_calls"], _think_probe_after())


def _note_nothink_call(llm) -> None:
    """A spec call that started without a think phase (the tripped regime)."""
    st = _think_state(llm)
    st["nothink_calls"] += 1
    st["skipped_think_last"] = 1


def _note_nothink_verify_failure(llm, label: str = "") -> None:
    """The spec just built WITHOUT attempting a think phase failed its
    verify: probe thinking on the very next spec call instead of waiting out
    the window. Not armed after a probe that aborted and fell back to the
    no-think retry (`skipped_think_last` is 0 there) — thinking was just
    tried; so a failing leaf pays at most floor(max_attempts / 2) probes
    (§4FJ review MAJOR 2, pinned). Smoke/constraint-gate failures do not arm
    it on purpose: the verify is the spec's own contract, the gates are
    mechanical checks the model never saw."""
    st = _think_state(llm)
    k = _think_probe_after()
    if st["skipping"] and st["skipped_think_last"] and k > 0 and st["nothink_calls"] < k:
        st["nothink_calls"] = k
        logger.warning(
            "coding_executor: verify failed on a no-think spec%s — the next "
            "spec call probes thinking (GHOST_CODING_THINK_PROBE_AFTER=%d)",
            f" ({label})" if label else "", k)


def _note_think_inconclusive(llm) -> None:
    """The streamed think phase returned NOTHING on either channel (an
    upstream error frame or an empty response — contention, a restarting
    server — not evidence about thinking). §4FJ review MAJOR 1: scored as a
    clean think this wiped the streak and printed "the probe succeeded" for
    a zero-byte response. The probe is disarmed; the window is left as it
    was, so the next call probes again once the upstream is back."""
    st = _think_state(llm)
    if st["probe"]:
        logger.warning("coding_executor: probe think phase got an empty "
                       "upstream response — inconclusive, window unchanged")
    st["probe"] = 0
    st["skipped_think_last"] = 0


def _note_think_outcome(llm, *, aborted: bool) -> None:
    """Record one streamed think phase: aborted (zero content) or clean."""
    n = _think_skip_after()
    st = _think_state(llm)
    st["skipped_think_last"] = 0
    was_probe = st["probe"]
    st["probe"] = 0
    if not aborted:
        if st["aborts"]:
            logger.warning("coding_executor: think phase produced content — "
                           "abort streak reset (was %d)%s", st["aborts"],
                           " — the probe succeeded, thinking is back on"
                           if was_probe else "")
        st["aborts"] = 0
        st["skipping"] = 0
        st["nothink_calls"] = 0
        return
    st["aborts"] += 1
    if was_probe:
        st["nothink_calls"] = 0
        logger.warning(
            "coding_executor: probe think phase aborted too — no-think window "
            "extended by %d spec calls", _think_probe_after())
    if n > 0 and st["aborts"] >= n and not st["skipping"]:
        st["skipping"] = 1
        logger.warning(
            "coding_executor: %d consecutive think phases aborted with no "
            "content — spec calls now start with thinking DISABLED "
            "(GHOST_CODING_THINK_SKIP_AFTER=%d; a clean think phase, a probe "
            "after GHOST_CODING_THINK_PROBE_AFTER=%d no-think calls, or a "
            "restart resets)", st["aborts"], n, _think_probe_after())


# JSON grammar for the no-think spec call (2026-09-03, §4EI follow-up).
# The five kept rejections were all STRUCTURAL slips (a dropped `{`, `]` or
# `]}` at the files boundary). A grammar makes those impossible at the
# sampler: llama-server's OpenAI-compatible `response_format` compiles the
# schema to GBNF and constrains every generated token. The schema is
# deliberately loose — only the shape that failed (an ARRAY of OBJECTS with
# a path) is required; extra keys and any edits item shape are allowed, so
# it cannot starve a legitimate spec. Thinking calls never get it: a think
# block is not JSON. Never applied to the main turn loop (different path).
#   GHOST_CODING_SPEC_JSON_GRAMMAR   unset/1 = json_schema (default),
#                                    object  = json_object (shape-free),
#                                    0/off   = no grammar.
SPEC_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "files": {"type": "array", "items": {
            "type": "object",
            "properties": {"path": {"type": "string"},
                           "generate": {"type": "string"},
                           "content": {"type": "string"},
                           "append": {"type": "string"},
                           "edits": {"type": "array",
                                     "items": {"type": "object"}}},
            "required": ["path"]}},
        "verify": {"type": "string"},
        "summary": {"type": "string"},
        "ledger": {"type": "string"},
    },
    "required": ["files"],
}


def _spec_response_format() -> Optional[Dict[str, Any]]:
    mode = (os.getenv("GHOST_CODING_SPEC_JSON_GRAMMAR") or "1").strip().lower()
    if mode in ("0", "off", "false", "no", "none"):
        return None
    if mode == "object":
        return {"type": "json_object"}
    return {"type": "json_schema",
            "json_schema": {"name": "build_spec", "schema": SPEC_JSON_SCHEMA}}


async def _spec_nothink_call(llm, model: str, sys_hint: str, user: str,
                             is_background: bool) -> Tuple[str, str]:
    """The thinking-disabled spec call: ``/no_think`` soft-switch +
    ``enable_thinking=False`` hard-switch + a system nudge — the same recipe
    as project_research._llm_call and dream.py — plus the JSON grammar
    (see SPEC_JSON_SCHEMA). Returns (content, reasoning). Used both as the
    retry after an aborted think phase and, once the think policy has
    flipped, as the FIRST call. A backend that rejects `response_format`
    (or any error while the grammar is on) gets ONE retry without it, so a
    grammar can never make the leaf fail on its own."""
    def _payload(grammar: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        p: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_hint +
                 "\nDo NOT emit a <think> block — output the JSON object directly."},
                {"role": "user", "content": user + "\n\n/no_think"},
            ],
            "temperature": 0.3, "max_tokens": 16384, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if grammar:
            p["response_format"] = grammar
        return p

    from .llm import BackgroundDeferred as _Deferred
    grammar = _spec_response_format()
    if is_background:
        # §4SD: streamed, so it can give the slot to the owner (`_spec_stream`);
        # a rejected grammar ends the stream with an error frame, not a raise —
        # an empty answer under the grammar is the cue for the retry without it
        def _stream_payload(g):
            return {k: v for k, v in _payload(g).items() if k != "stream"}
        try:
            content, reasoning, _ = await _spec_stream(llm, _stream_payload(grammar), True)
        except Exception as e:  # noqa: BLE001 — see docstring
            if not grammar or isinstance(e, _Deferred):      # review r1: a deferral is not a grammar failure
                raise
            logger.warning("coding_executor: no-think spec call with the JSON "
                           "grammar failed (%s: %s) — retrying once without it",
                           type(e).__name__, _short(str(e), 160))
            content, reasoning = "", ""
        if grammar and not content.strip() and not reasoning.strip():
            content, reasoning, _ = await _spec_stream(llm, _stream_payload(None), True)
        return content, reasoning
    kw = _main_call_kw(is_background, "coding spec")
    try:
        resp = await llm.chat_completion(_payload(grammar), is_background=is_background, **kw)
    except Exception as e:  # noqa: BLE001 — see docstring
        if not grammar or isinstance(e, _Deferred):
            raise
        logger.warning("coding_executor: no-think spec call with the JSON "
                       "grammar failed (%s: %s) — retrying once without it",
                       type(e).__name__, _short(str(e), 160))
        resp = await llm.chat_completion(_payload(None), is_background=is_background, **kw)
    msg = ((resp or {}).get("choices", [{}])[0].get("message", {})) or {}
    return msg.get("content") or "", msg.get("reasoning_content") or ""


def _unparsed_spec_dir():
    home = (os.getenv("GHOST_HOME") or "").strip()
    if not home:
        return None
    from pathlib import Path
    return Path(home) / "system" / "coding_executor_failures"


def _keep_unparsed_spec_output(content: str, reasoning: str,
                        note: str = "") -> Optional[str]:
    """Keep the raw channels of a rejected spec call on disk (never raises).
    Returns the file path, or None when GHOST_HOME is unset or the write
    failed. Newest ``UNPARSED_SPEC_KEEP`` files survive."""
    d = _unparsed_spec_dir()
    if d is None:
        return None
    try:
        import time as _time
        d.mkdir(parents=True, exist_ok=True)
        stamp = _time.strftime("%Y%m%dT%H%M%S", _time.gmtime())
        fn = d / f"{stamp}_{os.getpid()}_{_time.time_ns() % 1_000_000:06d}.txt"
        fn.write_text(
            f"# coding_executor rejected spec output\n# {note}\n"
            f"# content_chars={len(content)} reasoning_chars={len(reasoning)}\n"
            f"===== CONTENT =====\n{content}\n"
            f"===== REASONING =====\n{reasoning}\n", encoding="utf-8")
        files = sorted(d.glob("*.txt"))   # names sort by time
        for old in files[:max(0, len(files) - UNPARSED_SPEC_KEEP)]:
            try:
                old.unlink()
            except OSError:
                pass
        return str(fn)
    except Exception:  # noqa: BLE001 — a diagnostic must never fail the leaf
        logger.debug("coding_executor: rejected-spec dump failed", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Spec structure repair (2026-09-03, §4EI live proof)
# ---------------------------------------------------------------------------
# The rejected-output dump finally showed WHY complete-looking no-think specs
# failed to parse: not quote escaping but two structural slips at the
# `files` array boundary — corpus tests/fixtures/spec_structure_4ei, 4 of 4
# rejections in one leaf:
#   * `…"}]` never emitted: a top-level key ("verify"/"summary"/"ledger")
#     arrives while the files array is still open  → the `]` was dropped;
#   * `…"},"path":…`: an entry key arrives directly inside the files array
#     → the `{` opening the next file object was dropped.
# Braces balance either way, so the brace scan says "malformed" and every
# salvage path fails. The repair inserts exactly the one character the
# parser's own error offset and the open-container stack call for, and
# nothing else; the result must strictly json.loads to a dict.

_SPEC_TOP_KEYS = frozenset({"files", "verify", "summary", "ledger"})
_SPEC_ENTRY_KEYS = frozenset({"path", "generate", "content", "append", "edits", "find",
                              "replace", "after", "before", "insert"})


def _container_stack(t: str, upto: int) -> str:
    """Open containers (`{`/`[`) at offset ``upto``, string-aware."""
    st: List[str] = []
    in_str = esc = False
    for ch in t[:upto]:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            st.append(ch)
        elif ch in "}]":
            if st:
                st.pop()
    return "".join(st)


def _strip_fences(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _repair_spec_structure(text: str, max_fixes: int = 4
                           ) -> Optional[Tuple[str, List[str]]]:
    """Repair the two boundary slips described above. Each fix is applied
    at the parser's error offset ONLY when the misplaced token is a known
    spec key and the container stack matches; anything else returns None.
    Returns ``(fixed_text, fixes)`` — never an unrepaired or unchanged
    text, so a caller cannot mistake a plain parse for a repair."""
    body = _strip_fences(text)
    start = body.find("{")
    if start < 0:
        return None
    body = body[start:]
    fixes: List[str] = []
    for _ in range(max_fixes):
        try:
            obj = json.loads(body)
        except json.JSONDecodeError as e:
            pos = e.pos
        else:
            return (body, fixes) if fixes and isinstance(obj, dict) else None
        if pos >= len(body):
            # Third slip (live proof, 5th kept output): the output simply
            # ENDS with `files` and the top object still open — the model
            # dropped the closing `]}` (finish_reason was stop, not a cut).
            # Close what is open, innermost first; an EOF inside a string
            # makes the next parse fail elsewhere and returns None.
            stack = _container_stack(body, len(body))
            if not stack:
                return None
            body += "".join("]" if c == "[" else "}" for c in reversed(stack))
            fixes.append(f"closed {len(stack)} open container(s) at end of output")
            continue
        if body[pos] == "]":
            # Fourth slip (post-deploy sanity leaf, think path): the `}`
            # closing the last file entry dropped — `…"append":"…"],"verify"`.
            # Only when the parser is inside an object that sits directly in
            # an array; the `]` then belongs to that array.
            stack = _container_stack(body, pos)
            if stack.endswith("[{"):
                body = body[:pos] + "}" + body[pos:]
                fixes.append("closed an entry object before `]`")
                continue
            return None
        if body[pos] != ":":
            return None
        # The string just before `pos` is a key that landed in the wrong
        # container: `"key":` where the parser expected `,` or `]`.
        q_end = body.rfind('"', 0, pos)
        q_start = body.rfind('"', 0, q_end) if q_end > 0 else -1
        if q_start < 0:
            return None
        key = body[q_start + 1:q_end]
        before = body[:q_start].rstrip()
        if not before.endswith(","):
            return None
        comma = len(before) - 1
        stack = _container_stack(body, q_start)
        if key in _SPEC_TOP_KEYS and stack == "{[":
            body = body[:comma] + "]" + body[comma:]
            fixes.append(f'closed `files` before "{key}"')
        elif key in _SPEC_ENTRY_KEYS and stack.endswith("["):
            body = body[:q_start] + "{" + body[q_start:]
            fixes.append(f'opened an entry object before "{key}"')
        else:
            return None
    return None


def _parse_spec_channels(content: str, reasoning: str, extract, usable) -> dict:
    """The full spec-extraction chain over both model channels: content,
    reasoning, both joined, raw-newline recovery, then the structural
    repair. Returns the first USABLE spec, else whatever the chain last
    produced (the caller treats non-usable as a rejection)."""
    spec = extract(content, repair_truncated=True) or {}
    if not usable(spec) and reasoning:
        spec = extract(reasoning, repair_truncated=True) or {}
        if not usable(spec):
            # A spec split across the closing </think> boundary.
            spec = extract(f"{reasoning}\n{content}", repair_truncated=True) or spec
    if not usable(spec):
        spec = _recover_spec_escaping_newlines(
            (content, reasoning), extract, usable) or spec
    if not usable(spec):
        for channel in (content, reasoning):
            if not (channel or "").strip():
                continue
            rep = _repair_spec_structure(channel)
            if not rep:
                continue
            cand = extract(rep[0]) or {}
            if usable(cand):
                logger.warning("coding_executor: spec structure repaired — %s",
                               "; ".join(rep[1]))
                return cand
    return spec


# ---------------------------------------------------------------------------
# Pre-write guards shared by append and content (2026-09-03, §4EI)
# ---------------------------------------------------------------------------

_WEB_SUFFIXES = (".html", ".htm", ".js", ".mjs", ".cjs", ".ts", ".jsx", ".tsx",
                 ".css", ".scss", ".vue", ".svelte")
# A line that is ONLY an elision marker — `...` / `…`, optionally inside a
# comment (`// ...`, `/* ... */`, `<!-- ... -->`). In JS/CSS such a line never
# parses; in HTML it is a text node that reads as "rest omitted". Python is
# deliberately NOT covered: `...` is a valid statement there.
_ELISION_LINE_RE = re.compile(
    r"^\s*(?:<!--\s*|/\*\s*|//\s*)?(?:\.\.\.|…)\s*(?:\*/\s*|-->\s*)?$")


def _placeholder_elision_reason(text: str, path: str) -> Optional[str]:
    """Why ``text`` must not be written to ``path``: it elides code with
    placeholder lines. Live 2026-09-03: the Talismans leaf appended a block
    with `...` in both its markup and its script; the file then failed the
    syntax check on that line for every retry."""
    if not path.lower().endswith(_WEB_SUFFIXES):
        return None
    hits = [i for i, line in enumerate((text or "").split("\n"), 1)
            if _ELISION_LINE_RE.match(line)]
    if not hits:
        return None
    where = ", ".join(f"line {h}" for h in hits[:4])
    return (f"{path} refused BEFORE writing: the spec contains {len(hits)} "
            f"placeholder line(s) ('...' at {where}) — elided code is not "
            f"code and does not parse. Emit the COMPLETE block; if it is too "
            f"long, put the feature in a separate file or split the task.")


def _html_keys(text: str, top_level_only: bool) -> List[Tuple[str, str]]:
    """``(attr, value)`` pairs for ``id`` and ``data-*`` attributes — on every
    element, or only on the DEPTH-0 elements of a fragment."""
    from html.parser import HTMLParser
    void = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
            "meta", "param", "source", "track", "wbr"}
    out: List[Tuple[str, str]] = []

    class _P(HTMLParser):
        depth = 0

        def _take(self, attrs):
            for k, v in attrs:
                if v is not None and (k == "id" or k.startswith("data-")):
                    out.append((k, v))

        def handle_starttag(self, tag, attrs):
            if not top_level_only or self.depth == 0:
                self._take(attrs)
            if tag not in void:
                self.depth += 1

        def handle_startendtag(self, tag, attrs):
            if not top_level_only or self.depth == 0:
                self._take(attrs)

        def handle_endtag(self, tag):
            if tag not in void and self.depth > 0:
                self.depth -= 1

    p = _P()
    p.feed(text or "")
    return out


def _html_duplicate_key_reason(base: str, fragment: str,
                               path: str) -> Optional[str]:
    """Why an HTML ``fragment`` must not be appended to ``base``: one of its
    TOP-LEVEL elements carries an ``id``/``data-*`` key the file already has
    (or the fragment uses twice). Live 2026-09-03: two leaves each appended a
    second ``<section data-section="bearings">``; the SPA's
    ``querySelector('[data-section=…]')`` matches the first element, so both
    deliverables were unreachable — and both tasks closed DONE."""
    if not path.lower().endswith((".html", ".htm")):
        return None
    try:
        top = _html_keys(fragment, top_level_only=True)
        have = set(_html_keys(base, top_level_only=False)) if base else set()
    except Exception:  # noqa: BLE001 — a guard must never raise
        return None
    seen: set = set()
    for k, v in top:
        if (k, v) in have or (k, v) in seen:
            return (f"append to {path} refused BEFORE writing: its top-level "
                    f"element carries {k}=\"{v}\", which the file already has "
                    f"(or the fragment uses twice). A second element with the "
                    f"same key is unreachable — selectors and routers match "
                    f"the FIRST one — so this is not an addition. Use `edits` "
                    f"to fill or change the existing {k}=\"{v}\" element, or "
                    f"give the new element a NEW key.")
        seen.add((k, v))
    return None


async def _restore_file(tool_runner: ToolRunner, path: str, content: str) -> bool:
    """Write ``content`` back to ``path`` after a syntax-failed append. True
    when the write went through (the file is the pre-append state again)."""
    try:
        out = await tool_runner(
            "file_system", {"operation": "write", "path": path, "content": content})
    except Exception as e:  # noqa: BLE001
        logger.warning("coding_executor: restore of %s after a syntax-failed "
                       "append errored: %s", path, e)
        return False
    if _looks_like_write_error(out):
        logger.warning("coding_executor: restore of %s after a syntax-failed "
                       "append was rejected: %s", path, _short(out))
        return False
    return True


_DIAG_LINE_RE = re.compile(r"\bline\s+(\d+)")


def _appended_block_start_line(base: str, path: str) -> int:
    """1-based line where ``_smart_append`` places the new block in ``base``:
    the line holding the last ``</body>`` (else ``</html>``) for HTML, two
    lines past the last non-blank line otherwise."""
    base = base or ""
    if path.lower().endswith((".html", ".htm")):
        low = base.lower()
        for anchor in ("</body>", "</html>"):
            idx = low.rfind(anchor)
            if idx != -1:
                return base[:idx].count("\n") + 1
    return base.rstrip().count("\n") + 3


def _syntax_error_before_block(base: str, path: str, reason: str) -> Optional[int]:
    """Given the reason ``_syntax_fail_reason`` produced (it carries the
    sandbox diagnostic), the reported line number when it points BEFORE the
    appended block — the file was already broken before this append; None
    when the line is inside or after the block, or no line is reported.
    Lines AFTER the block still count as the block's: an unterminated
    construct is reported at the next token or EOF — the tail the append
    pushed down."""
    text = str(reason or "")
    idx = text.find("SYNTAX CHECK FAILED")
    m = _DIAG_LINE_RE.search(text[idx:] if idx >= 0 else text)
    if not m:
        return None
    line = int(m.group(1))
    return line if line < _appended_block_start_line(base, path) else None


def _reverted_append_reason(path: str, out: str) -> str:
    text = str(out or "")
    idx = text.find("SYNTAX CHECK FAILED")
    diag = " ".join(text[idx:].split())[:400] if idx >= 0 else ""
    return (f"the block appended to {path} does NOT parse — {diag} The append "
            f"was REVERTED: {path} is back to its previous state, so there is "
            f"nothing to fix in place. Re-emit the WHOLE block as a corrected "
            f"`append` (complete code, no '...' elisions, no re-declared "
            f"identifiers).")


#: §4SD: the longest a stopped build spends writing its finished files
STOP_FLUSH_S = 15.0

#: a stream's finish (`_stream_file_text`) or abort reason
#: (`_stream_spec_completion`) when a background stream gave the slot to the owner
YIELDED = "yielded"
_MAX_YIELDS = 12


class _OwnerWatch:
    """r22: a background stream holds the one main slot for minutes (a spec)
    to an hour (a 64k file) — it yields as soon as a USER request starts or a
    foreground LLM call waits (the owner waited behind it). r23: the baseline
    is read at the FIRST chunk — after the client's park admitted the stream
    (read before it, a call admitted after a yield, or inside an image
    render's idle window, never yielded again)."""

    def __init__(self, llm, on: bool):
        self.llm, self.on, self.base = llm, bool(on), None

    def owner_waits(self) -> bool:
        """Called at every chunk."""
        if not self.on:
            return False
        now = int(getattr(self.llm, "foreground_requests", 0) or 0)
        if self.base is None:
            self.base = now
            return False
        return now > self.base or int(getattr(self.llm, "foreground_tasks", 0) or 0) > 0


def _main_call_kw(is_background: bool, what: str) -> Dict[str, Any]:
    """§4SD: a FOREGROUND executor call (the owner's own build) does not open
    with the head, so it displaces the owner's conversation in the slot — it
    takes the §4SC side path: the conversation is saved first and the
    request's next head call restores it. A background call has no owner
    conversation of its own to keep (the client's background path does)."""
    return {} if is_background else {"side_main": True, "task_label": what}


async def _stream_spec_completion(llm, payload: Dict[str, Any],
                                  is_background: bool, *, may_yield: bool = True):
    """Stream the build-spec generation, accumulating the content and
    reasoning channels, and ABORT a runaway think phase early.

    The non-streaming call this replaces could not see a think-loop until
    the full 16384-token budget was burned (~4 minutes on the live box,
    observed twice in one request, 92a968fc 2026-07-25) — the guard then
    fired on the corpse. Streaming lets the abort land while the loop is
    still forming. One abort condition, evaluated ONLY while the
    content channel is still empty (once the JSON spec starts, reasoning
    is done and generated code repeats lines legitimately):

      * ``_detect_thinking_loop`` — the exact-n-gram detector the main
        turn loop uses. Deliberately the ONLY heuristic here: the first
        deploy also ran the paragraph-repeat detector and it aborted the
        thinking of EVERY coding leaf in the first live autoadvance run
        (one at just 3,019 chars) — spec planning legitimately restates
        the task and field lists, so that detector stays out of this
        path (2026-07-25 second deploy).

    The 30K-char length ceiling that sat beside it is removed (§4SD): a
    long think runs to the call's max_tokens, and the caller's no-content
    branch retries it with thinking off.

    Returns ``(content, reasoning, abort_reason)`` where ``abort_reason``
    is ``None`` (clean), ``"loop"``, or `YIELDED` (§4SD: a background call
    gave the slot to the owner — ``may_yield``; `_spec_stream` calls again).
    Stall/mid-stream
    errors surface as an SSE error event from the llm layer — the
    accumulator stops there and the caller's existing no-spec
    diagnostics handle the partial output."""
    import json as _json
    from .stream_guards import (
        THINKING_LOOP_PROBE_EVERY,
        _detect_thinking_loop,
    )
    import inspect as _inspect
    streamer = getattr(llm, "stream_chat_completion", None)
    if streamer is None or not _inspect.isasyncgenfunction(streamer):
        # Non-streaming client (test doubles — including MagicMocks whose
        # auto-attributes would otherwise masquerade as a streamer — and
        # alternate backends): degrade to the single-shot call — correct
        # output, no early loop abort.
        resp = await llm.chat_completion(
            {**payload, "stream": False}, is_background=is_background,
            **_main_call_kw(is_background, "coding spec"))
        msg = ((resp or {}).get("choices", [{}])[0].get("message", {})) or {}
        return (msg.get("content") or "",
                msg.get("reasoning_content") or "", None)
    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    c_len = r_len = 0
    next_probe = THINKING_LOOP_PROBE_EVERY
    aborted = None
    agen = streamer(payload, is_background=is_background,
                    **_main_call_kw(is_background, "coding spec"))
    watch = _OwnerWatch(llm, is_background and may_yield)
    finished = False
    try:
        async for raw in agen:
            line = (raw.decode("utf-8", "replace") if isinstance(raw, bytes)
                    else str(raw)).strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = _json.loads(data)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            if obj.get("error"):
                # Stall / mid-stream break — already logged upstream.
                break
            choice = ((obj.get("choices") or [{}])[0]) or {}
            finished = finished or bool(choice.get("finish_reason"))
            # r1/r2: after the frame is read — a spec whose finish arrived is kept
            if not finished and watch.owner_waits():
                aborted = YIELDED
                break
            delta = choice.get("delta") or {}
            rc = delta.get("reasoning_content") or ""
            cc = delta.get("content") or ""
            if rc:
                reasoning_parts.append(rc)
                r_len += len(rc)
            if cc:
                content_parts.append(cc)
                c_len += len(cc)
            if c_len == 0 and r_len >= next_probe:
                next_probe = r_len + THINKING_LOOP_PROBE_EVERY
                buf = "".join(reasoning_parts)
                reasoning_parts = [buf]  # keep the join amortised
                if _detect_thinking_loop(buf):
                    aborted = "loop"
                    break
    finally:
        try:
            await agen.aclose()
        except Exception:  # noqa: BLE001 — closing a broken stream
            pass
    return "".join(content_parts), "".join(reasoning_parts), aborted


async def _spec_stream(llm, payload: Dict[str, Any], is_background: bool):
    """§4SD: `_stream_spec_completion`, called again after it yielded the slot
    to the owner (the partial is dropped; the next call waits in the client
    until the owner is done) — at most `_MAX_YIELDS` times, then it runs on."""
    for n in range(_MAX_YIELDS + 1):
        content, reasoning, aborted = await _stream_spec_completion(
            llm, payload, is_background, may_yield=n < _MAX_YIELDS)
        if aborted != YIELDED:
            return content, reasoning, aborted
        logger.info("coding_executor: the spec call gave the slot to the owner (%d/%d) — it runs again after",
                    n + 1, _MAX_YIELDS)
    return content, reasoning, aborted        # unreachable: the last call never yields


async def _generate_build_spec(llm, model: str, description: str, ledger: str, *,
                               existing_files: Optional[Dict[str, str]] = None,
                               single_file: bool = False,
                               feedback: str = "",
                               research_context: Optional[Dict[str, str]] = None,
                               is_background: bool = False,
                               constraints: Optional[List[str]] = None) -> Tuple[dict, bool]:
    """Ask the model for a JSON build spec.

    Returns ``(spec, was_empty)``: ``spec`` is ``{}`` on any failure, and
    ``was_empty`` is True when the upstream returned a COMPLETELY empty
    completion (no content AND no reasoning). An empty completion is an
    upstream/infra symptom (contention, a dropped response) — NOT the model
    failing to produce a spec — so the caller backs off and reports it honestly
    instead of hammering feedback retries that can't help."""
    sys_hint = (
        "You are building ONE task inside a larger project. Output ONLY a JSON "
        "object — no prose, no markdown fences — with this shape:\n"
        '{"files":[{"path":"name.ext", <ONE OF generate|content|append|edits>}],'
        '"verify":"shell cmd that exits 0 iff it works, or \\"\\"",'
        '"summary":"one line","ledger":"one durable fact or \\"\\""}\n'
        "Per file choose EXACTLY ONE of:\n"
        '  "generate": for a brand-NEW file of more than ~100 lines — DO NOT '
        "write its code here: describe what it must contain (purpose, "
        "structure, the functions/ids/APIs other files rely on, how it "
        "connects to them). The system then writes that file in a separate "
        "call with its own output budget, showing it this plan.\n"
        '  "content": the FULL file text — for a SHORT brand-NEW file ONLY. It must '
        "be COMPLETE and VALID (a full HTML page must include </body></html>); a file that "
        "looks truncated is rejected.\n"
        '  "append": to ADD new code to an EXISTING file — you write ONLY the '
        "new code (a self-contained <script>/<style>/<div> or function). For "
        "HTML the system inserts it before </body>, so you do NOT need an "
        "anchor. Use this for pure ADDITIONS (a new feature/app/section).\n"
        '  "edits": [{"find":"EXACT existing text","replace":"…"}] — to MODIFY '
        "existing code (fix a bug, change a value, refactor, rename); the find "
        "text must byte-match. Use this for any CHANGE to code that already "
        "exists — you cannot append your way to a modification.\n"
        "Pick by INTENT: adding → append; changing existing code → edits.\n"
        "Rules: complete runnable code (no TODOs/stubs); BARE project-relative "
        "paths (no /workspace, sandbox/, projects/<id>/); only the files THIS "
        "task needs; prefer a runnable verify that exercises what you built. "
        'When the deliverable ALREADY exists and works, "files" may be [] '
        "with a verify that proves it — never re-implement working code. "
        "The verify must NOT kill or restart processes/services (no fuser -k, "
        "pkill, kill) — if a service is involved, probe the one already "
        "running (e.g. curl its port).\n"
        "Think briefly (a short plan), then answer. Do NOT draft file contents "
        "inside your thinking — code belongs in the JSON output only; thinking "
        "and the JSON share one output limit, and a reply cut there loses the "
        "attempt. Never append a second copy of a section/container the file "
        "already has (same id or data-* key): fill the existing one with "
        "`edits`, or give the new one a new key."
    )
    user = f"TASK: {description}\n"
    user += _render_constraints(constraints)
    if ledger:
        user += ("\nPROJECT LEDGER (existing files / APIs / conventions — build "
                 f"CONSISTENTLY with these):\n{ledger}\n")
    user += _render_research(research_context)
    user += _render_existing(existing_files, single_file)
    user += _render_already_built(description, existing_files)
    if feedback:
        user += (f"\nYOUR PREVIOUS ATTEMPT FAILED: {feedback}\n"
                 "Produce a corrected spec that fixes exactly this.\n")
    # STREAMED (2026-07-25) so a thinking loop is aborted within a probe
    # interval instead of burning the whole token budget first — see
    # _stream_spec_completion. max_tokens rationale unchanged: 16384
    # (raised from 8192, itself raised from 4096) because the output is a
    # JSON spec with a COMPLETE file embedded as `content` — a cut
    # mid-string yields invalid JSON and a truncated file, and a reasoning
    # model spends part of the budget in `reasoning_content` before the
    # JSON even starts. §4SD: a big NEW file no longer rides in this JSON —
    # the spec names it with `generate` and `_generate_file` writes it in
    # its own call; `content` here is for short files.
    think_off = _think_disabled_now(llm)
    if think_off:
        # Think policy (§4EI): the streak of aborted think phases has hit the
        # threshold (or the operator set the knob to 0) — skip the ~2-minute
        # think phase that has been producing nothing and go straight to the
        # call that has been producing specs. WARNING so the operator can
        # count the window between the trip and the probe (§4FJ review M-5).
        _note_nothink_call(llm)
        logger.warning("coding_executor: spec call starts with thinking disabled "
                       "(no-think call %d/%d in this window; abort streak %d)",
                       _think_state(llm)["nothink_calls"], _think_probe_after(),
                       _think_state(llm)["aborts"])
        content, reasoning = await _spec_nothink_call(
            llm, model, sys_hint, user, is_background)
        loop_aborted = None
    else:
        _skip = _think_skip_after()
        if _skip > 0 and _think_state(llm)["aborts"] >= _skip and _probe_due(llm):
            _arm_probe(llm)
        try:
            content, reasoning, loop_aborted = await _spec_stream(llm, {
                "model": model,
                "messages": [{"role": "system", "content": sys_hint},
                             {"role": "user", "content": user}],
                "temperature": 0.3, "max_tokens": 16384,
            }, is_background)
        except BaseException:
            _note_think_inconclusive(llm)      # no stale probe flag (§4FJ M-1)
            raise
        if not content.strip() and not reasoning.strip() and not loop_aborted:
            _note_think_inconclusive(llm)
        else:
            _note_think_outcome(
                llm, aborted=bool(loop_aborted)
                or (not content.strip() and bool(reasoning.strip())))
    # Reasoning models (Qwen via llama.cpp) emit their chain-of-thought in a
    # separate `reasoning_content` field. When the think block consumes the
    # whole token budget without closing, the parser routes EVERYTHING there
    # and leaves `content` empty — so the JSON build spec lives entirely in the
    # reasoning channel. Reading only `content` then logged `len=0` and FAILED
    # the task with "model produced no file spec" (observed live: 5 coding
    # leaves in one project killed this way). Fall back to the reasoning
    # channel, mirroring core/agent.py (~2973) and project_research.py.
    from .agent import extract_json_from_text

    def _usable(s) -> bool:
        if not isinstance(s, dict):
            return False
        files = s.get("files")
        if isinstance(files, list) and files:
            # An entry must carry a payload (§4EI): the truncation repair can
            # salvage `[{"path": "app.js"}]` from a broken spec, and a
            # path-only entry is not a spec — accepting it skipped the
            # no-think retry AND the rejected-output dump, then failed the
            # attempt as "no writable files".
            if any(isinstance(f, dict) and (
                    str(f.get("content") or "").strip()
                    or str(f.get("generate") or "").strip()
                    or str(f.get("append") or "").strip()
                    or (isinstance(f.get("edits"), list) and f.get("edits")))
                   for f in files):
                return True
        # A VERIFY-ONLY spec ("the deliverable already exists — just prove
        # it works") is a first-class answer (2026-08-01): the caller's
        # no-files path honours the verify and closes the task without
        # rebuilding. Before this, a verify-only spec in the content
        # channel was "not usable", so the reasoning-channel fallback
        # could clobber it and the retry loop steered the model into
        # re-implementing files that already worked.
        return bool(str(s.get("verify") or "").strip())

    spec = _parse_spec_channels(content, reasoning, extract_json_from_text, _usable)
    if (not _usable(spec) and not content.strip() and reasoning.strip()
            and not think_off):
        # The think phase produced no JSON: the stream guard aborted it
        # (exact n-gram loop — the reason is
        # named in the log so live tuning has data), or the think block
        # consumed the whole max_tokens budget before the JSON ever
        # started (observed live 2026-07-06: content=0, reasoning=40-62k
        # chars, twice in one project — each a ~5-minute generation
        # lost). Thinking earns its cost on attempt 1, but once it has
        # looped or eaten the budget the only productive move is ONE retry
        # with thinking off — same recipe as project_research._llm_call and
        # dream.py: /no_think soft-switch + enable_thinking=False
        # hard-switch + a system nudge.
        if loop_aborted == "loop":
            logger.warning(
                "coding_executor: exact n-gram thinking loop aborted "
                "mid-stream (content=0, reasoning=%d chars) — retrying "
                "once with thinking disabled", len(reasoning.strip()))
        else:
            logger.warning(
                "coding_executor: think block consumed the whole budget "
                "(content=0, reasoning=%d chars) — retrying once with "
                "thinking disabled", len(reasoning.strip()))
        content, reasoning = await _spec_nothink_call(
            llm, model, sys_hint, user, is_background)
        spec = _parse_spec_channels(content, reasoning, extract_json_from_text, _usable)
    was_empty = not content.strip() and not reasoning.strip()
    if not _usable(spec):
        # Diagnostic: the model returned no usable file spec. Log a window of
        # the raw output (BOTH channels) so we can see WHY (prose? broken
        # JSON-escaped code? truncated mid-think? — or a fully EMPTY upstream
        # response, content=0 reasoning=0, which is contention, not the model).
        raw = (content or reasoning).strip()
        # Keep the WHOLE output (§4EI): the head/tail window below could not
        # explain four complete-looking rejections in one request.
        kept = None if was_empty else _keep_unparsed_spec_output(
            content, reasoning, note=f"think_off={think_off}")
        logger.warning(
            "coding_executor: no file spec parsed (content=%d reasoning=%d%s). "
            "RAW head: %s ||| tail: %s%s",
            len(content.strip()), len(reasoning.strip()),
            " — EMPTY upstream response" if was_empty else "",
            raw[:400].replace("\n", "\\n"), raw[-200:].replace("\n", "\\n"),
            (f" ||| full output kept at {kept}" if kept else
             ("" if was_empty else
              " ||| (set GHOST_HOME to keep the full output on disk)")))
        # Contract (docstring): {} on every failure. A salvaged-but-unusable
        # dict (path-only entries) must not leak to the caller as a spec.
        spec = {}
    return spec, was_empty


_GENERATE_SYS = (
    "You write ONE complete file of a software project. Reply with the file's "
    "text ONLY, from its first line — no markdown fence, no explanation, no "
    "JSON. The code must be complete and runnable: no TODOs, stubs, "
    "placeholders or elided sections. When the file is complete, put the line "
    "{marker} on its own last line."
)


async def _window_tokens(llm) -> int:
    """§4SD: the main server's per-slot window (``/props`` n_ctx), cached on
    the client for 10 min (60 s after a failure); 0 = unknown → the 16k floor."""
    now = time.monotonic()
    c = getattr(llm, "_ghost_window_cache", None)
    if isinstance(c, tuple) and len(c) == 2 and isinstance(c[1], float) and now < c[1]:
        return int(c[0])
    n = 0
    try:
        from . import head_saves as _hs
        base = _hs.server_base(str(getattr(llm, "upstream_url", "") or ""))
        if base:
            p = await asyncio.to_thread(_hs._http, base + "/props", None, 5.0)
            n = int(((p or {}).get("default_generation_settings") or {}).get("n_ctx") or 0)
    except Exception:  # noqa: BLE001 — unknown window = the floor
        n = 0
    try:
        llm._ghost_window_cache = (max(0, n), now + (600.0 if n > 0 else 60.0))
    except Exception:  # noqa: BLE001 — a client that takes no attributes re-reads
        pass
    return max(0, n)


async def _file_call_cap(llm, messages, deadline: float) -> int:
    """§4SD: a generated file's call gets the room the window leaves after its
    prompt (`agent.output_cap_for_tool_turn`, ≤ 64k, ≥ 16k; 16k when the
    window is unknown), never more than the leaf's time left can hold
    (`GENERATE_TOK_PER_S`, less `GENERATE_RESERVE_S`). 0 = no time for a
    useful round (under `GENERATE_MIN_TOKENS`)."""
    from .agent import DEFAULT_TOOL_TURN_MAX_TOKENS, output_cap_for_tool_turn
    by_time = int((_time_left_s(deadline) - GENERATE_RESERVE_S) * GENERATE_TOK_PER_S)
    if by_time < GENERATE_MIN_TOKENS:
        return 0
    cap = DEFAULT_TOOL_TURN_MAX_TOKENS
    if by_time > cap:
        win = await _window_tokens(llm)
        if win > 0:
            from ..utils.token_counter import estimate_tokens
            used = await asyncio.to_thread(
                lambda: sum(estimate_tokens(str(m.get("content") or "")) + 8 for m in messages))
            cap = output_cap_for_tool_turn(used, win)
    return min(cap, by_time)


async def _stream_file_text(llm, payload: Dict[str, Any], is_background: bool) -> Tuple[str, Optional[str]]:
    """One generated-file call, STREAMED (a 64k reply is over an hour; the
    client's non-streamed read timeout is far shorter). Returns the content
    channel and the finish reason ("length" = cut by max_tokens; None = the
    stream broke). A client without a streamer (test doubles) gets the
    single-shot call."""
    import inspect as _inspect
    streamer = getattr(llm, "stream_chat_completion", None)
    if streamer is None or not _inspect.isasyncgenfunction(streamer):
        resp = await llm.chat_completion({**payload, "stream": False}, is_background=is_background,
                                         **_main_call_kw(is_background, "generated file"))
        choice = ((resp or {}).get("choices") or [{}])[0] or {}
        return ((choice.get("message") or {}).get("content") or "", choice.get("finish_reason"))
    parts: List[str] = []
    finish: Optional[str] = None
    agen = streamer(payload, is_background=is_background,
                    **_main_call_kw(is_background, "generated file"))
    # r22/r23 (`_OwnerWatch`): a yielded round ends as a cut, and the next
    # round's call waits for the owner
    watch = _OwnerWatch(llm, is_background)
    try:
        async for raw in agen:
            if finish is None and watch.owner_waits():
                finish = YIELDED
                break
            line = (raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)).strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(obj, dict):
                continue
            if obj.get("error"):
                finish = None                 # a broken stream: not a clean end
                break
            ch = ((obj.get("choices") or [{}])[0]) or {}
            cc = (ch.get("delta") or {}).get("content") or ""
            if cc:
                parts.append(cc)
            if ch.get("finish_reason"):
                finish = ch.get("finish_reason")
    finally:
        try:
            await agen.aclose()
        except Exception:  # noqa: BLE001 — closing a broken stream
            pass
    return "".join(parts), finish


def _render_reference(files: Dict[str, str]) -> str:
    """The project's other files for a generated file's call: read-only
    reference, without the spec's `append`/`edits` steer (review r9: a
    raw-text writer told "USE `append`: a file entry {…}" can answer in JSON,
    and that text would be the file)."""
    parts = [f"--- {p} ({len(c)} bytes) ---\n{_file_excerpt(c)}" if c else f"--- {p} (exists; large) ---"
             for p, c in files.items()]
    if not parts:
        return ""
    return ("\nOTHER FILES OF THE PROJECT (read-only reference, excerpts — use "
            "their names, ids and APIs; do not copy them into this file):\n"
            + "\n\n".join(parts)[:40_000] + "\n")


def _generate_prompt(path: str, brief: str, *, description: str, ledger: str, plan: list,
                     shown_files: Dict[str, str], previous: str = "", feedback: str = "",
                     research_context: Optional[Dict[str, str]],
                     constraints: Optional[List[str]]) -> list:
    from .write_continuation import END_MARKER
    user = f"TASK: {description}\n"
    user += _render_constraints(constraints)
    if ledger:
        user += ("\nPROJECT LEDGER (existing files / APIs / conventions — build "
                 f"CONSISTENTLY with these):\n{ledger}\n")
    user += _render_research(research_context)
    user += _render_reference({p: c for p, c in shown_files.items() if _norm_path(p) != _norm_path(path)})
    if previous:
        # review r10: the leaf's own earlier version was shown as "another
        # file, do not copy" — a retry then wrote a different file
        # r20: only a file the failure names is told it failed
        how = ("the attempt failed — fix in it what the failure below is about; keep what works"
               if feedback else "keep it unless the plan changed it")
        user += f"\nYOUR PREVIOUS VERSION OF {path} ({how}):\n{_file_excerpt(previous)}\n"
    if feedback:
        user += f"\nTHE PREVIOUS ATTEMPT FAILED:\n{_short(feedback, 1500)}\n"
    lines = []
    for f in plan:
        if not isinstance(f, dict) or not str(f.get("path") or "").strip():
            continue
        what = (str(f.get("generate") or "").strip() or
                ("edited in place" if f.get("edits") else
                 "code appended to it" if f.get("append") else "written in full"))
        lines.append(f"- {str(f.get('path')).strip()}: {_short(what, 600)}")
    if lines:
        user += "\nTHE PLAN FOR THIS TASK (every file it touches):\n" + "\n".join(lines) + "\n"
    user += (f"\nWRITE NOW the complete file: {path}\nIt must contain: {brief}\n"
             "Match the names, ids and APIs the other files of the plan use.\n")
    return [{"role": "system", "content": _GENERATE_SYS.format(marker=END_MARKER)},
            {"role": "user", "content": user}]


class OutOfTime(str):
    """A failure reason that means the TIME ran out, not the file or the
    model (review r11: matched by a "out of time" prefix, so a file named
    so read as a timeout)."""


async def _generate_file(llm, model: str, fspec: dict, *, description: str, ledger: str,
                         plan: list, shown_files: Dict[str, str],
                         research_context: Optional[Dict[str, str]],
                         constraints: Optional[List[str]], is_background: bool,
                         deadline: float, previous: str = "",
                         feedback: str = "") -> Tuple[Optional[str], Optional[str]]:
    """§4SD: write one ``generate`` entry's file as raw text — the first call
    sized by the window's room and the time left, then continuation rounds
    while the output limit cuts it, under the main loop's cut-write rules
    (core/write_continuation): the file is complete at the end marker, or
    when the model ends a round that wrote text; a round that starts over,
    repeats the file, is prose about it, or is empty is discarded; two such
    rounds fail. Returns ``(text, None)`` or
    ``(None, reason)`` — an `OutOfTime` reason when the time ran out."""
    from .agent import DEFAULT_TOOL_TURN_MAX_TOKENS
    from .llm import BackgroundDeferred, _disable_thinking
    from .write_continuation import (continuation_messages, drop_chatter,
                                     is_repetition, restarts_from_head, strip_tail_overlap,
                                     to_last_complete_line)
    from ..tools.file_system import restarts_file
    path = str(fspec.get("path") or "").strip()
    brief = str(fspec.get("generate") or "").strip()
    ask = f"{description}\n{path}: {brief}"
    text, done, bad, rounds = "", False, 0, 0
    time_capped, err, one_line, yields = False, "", False, 0
    while rounds <= GENERATE_ROUNDS and not done:
        if not text:
            messages = _generate_prompt(path, brief, description=description, ledger=ledger,
                                        plan=plan, shown_files=shown_files, previous=previous,
                                        feedback=feedback, research_context=research_context,
                                        constraints=constraints)
        else:
            messages = continuation_messages(path, ask, text[:_GENERATE_HEAD_CHARS],
                                             text[-_GENERATE_TAIL_CHARS:], rounds, bad)
        cap = await _file_call_cap(llm, messages, deadline)
        if cap <= 0:
            return None, OutOfTime(f"out of time: the leaf's time budget could not hold another "
                                   f"call for {path} ({len(text):,} chars written)")
        time_capped = cap < DEFAULT_TOOL_TURN_MAX_TOKENS   # review r12: the last round's, not sticky
        one_line = False
        payload = {"model": model, "messages": messages, "temperature": 0.3, "max_tokens": cap}
        _disable_thinking(payload)            # the plan was the thinking; this is the file
        rounds += 1
        try:
            raw, finish = await _stream_file_text(llm, payload, is_background)
        except (BackgroundDeferred, asyncio.CancelledError):
            raise                             # the owner is back / the step was cut
        except Exception as e:  # noqa: BLE001 — a broken call is a bad round (review r10: keep the text)
            raw, finish, err = "", None, _short(str(e), 160)
        opening = text or _THINK_HEAD_RE.sub("", raw or "", count=1)   # r19: how the FILE opened
        cont, done = _clean_file_round(raw, wrapped=bool(_WRAP_OPEN_RE.match(opening.lstrip())))
        yielded = finish == YIELDED
        if yielded:
            yields += 1
            # r22: the owner came — a cut that costs no round (bounded: the
            # next call waits for the owner, then defers; past
            # `_MAX_YIELDS` a yield is an ordinary round)
            finish, rounds = "length", rounds - (1 if yields <= _MAX_YIELDS else 0)
            logger.info("coding_executor: generated %s — the owner is back; the round stops at %d chars",
                        path, len(cont))
        # The first call is never judged. r21 — one rule per file kind:
        # CODE: a continuation round that opens with a chatter line (no code
        # character in it) is ONLY that line → discarded, or, with the
        # marker, the file ends there; with lines after it, the opener goes
        # when it names the file/rest/code or ends with ":" (r16–r20: the
        # lines after it are never judged — "    return t", "  fi\ndone" are
        # code with no code character). PROSE/MARKDOWN: the shared rule — a
        # short chatter round is discarded unless it ended (its last paragraph).
        chatter = False
        if text and _file_kind(path) == "code":
            first, _, body = cont.lstrip("\r\n").partition("\n")
            if _is_opener(cont, first):
                if not body.strip():
                    if done:
                        cont = ""             # "Sure! It is complete.<marker>" — the file ends
                    else:
                        chatter = True
                elif _strip_opener_ok(path, cont):
                    cont = body.lstrip("\r\n")   # the rest keeps its own whitespace
        elif text:
            chatter = drop_chatter(cont)[1] and not done and finish != "stop"
        if yielded and chatter:
            cont, chatter = "", False         # r23: a cut-off opener fragment — no bad round
        if finish != "stop" and not done:
            kept = to_last_complete_line(cont)   # a cut (or broken) reply keeps whole lines
            one_line = bool(cont.strip()) and not kept.strip()
            cont = kept
        if text:
            cont = strip_tail_overlap(text[-_GENERATE_TAIL_CHARS:], cont)
        if not cont.strip() and not done:
            bad += 0 if yielded else 1        # an empty round ends nothing (review r11)
            logger.warning("coding_executor: generated %s — round %d discarded (%s)", path, rounds,
                           err or ("one line longer than a reply" if one_line else "empty"))
            if bad >= 2:
                break
            err = ""
            continue
        # r19: a stuck model is still repeating when the limit cuts it — judge
        # repetition on a CUT round's tail only (a level file's literal map,
        # a board of equal rows, a wall row repeated per level is content)
        stuck = (finish != "stop" and not done
                 and is_repetition("\n".join(cont.splitlines()[-_REPEAT_TAIL_LINES:]), text))
        restart = bool(cont.strip() and text and (restarts_file(text[:_GENERATE_HEAD_CHARS], cont)
                                                  or restarts_from_head(text[:_GENERATE_HEAD_CHARS], cont)
                                                  or stuck))
        if restart or chatter:
            bad += 1
            done = False
            logger.warning("coding_executor: generated %s — round %d discarded (started over / "
                           "repeated / prose)", path, rounds)
            if bad >= 2:
                break
            continue
        text += cont
        # §4SD review r14: the file is complete at the marker, or when the model
        # ENDED a round that wrote text (finish "stop"); an empty round after a
        # cut ends nothing. A confirming round (r11–r13) only invited echoes,
        # remnants and prose into the file.
        if finish == "stop":
            done = True                       # (an empty "stop" round was discarded above)
        if len(text) > MAX_CONTENT_CHARS:
            return None, f"{path} grew past {MAX_CONTENT_CHARS:,} chars — split it into smaller files"
        if cont.strip():
            _note_round_progress(path)
        logger.info("coding_executor: generated %s — round %d: +%d chars (now %d)%s",
                    path, rounds, len(cont), len(text), " — complete" if done else "")
    if not done:
        if bad >= 2:
            why = (f"the call failed: {err}" if err else
                   "the reply was only a <think> block" if not text and _THINK_HEAD_RE.match(raw or "") else
                   "a single line longer than one reply — write it on many lines" if one_line else
                   "they were empty, started over, repeated the file or were prose")
            return None, f"writing {path} failed: two continuation rounds had to be discarded ({why})"
        if time_capped:
            return None, OutOfTime(f"out of time: {path} was still unfinished after {GENERATE_ROUNDS} "
                                   f"continuation rounds the time left kept short ({len(text):,} chars)")
        return None, (f"{path} was still unfinished after {GENERATE_ROUNDS} continuation rounds "
                      f"({len(text):,} chars) — split it into smaller files")
    text = _unwrap_file(text, path)
    if not text.strip():
        return None, f"the model wrote nothing for {path}"
    return text, None


#: r18: the block, then whole blank lines — never the next line's indentation
_THINK_HEAD_RE = re.compile(r"\A\s*(?:<think>(?:.*?</think>|.*\Z)|</think>)(?:[ \t]*\r?\n)*",
                            re.DOTALL | re.IGNORECASE)   # r19: an orphan close too
_PROSE_EXTS = frozenset({"md", "markdown", "mdx", "rmd", "qmd", "txt", "text", "rst", "adoc", "asciidoc",
                         "tex", "log", "org"})
#: a character prose about a file does not use (r18/r21: ' ` ( ) are prose —
#: "Here's", "`app.py`", "(continued)")
_CODE_CHAR_RE = re.compile(r"[={}\[\];<>\"$#|&\\]")
#: openers the shared list lacks (r22: "Let me continue", "Okay, here is",
#: "Below is", a curly "Here’s")
#: (r25: at column 0 only — an indented docstring line is the file's)
_MORE_OPENERS_RE = re.compile(r"\A[\r\n]*\(?(?:let me continue\b|ok(?:ay)?[,!.](?=\s|\Z)|below is\b|here’s\b|"
                              r"(?:the )?(?:rest|remainder) of the (?:file|code)\b|"
                              # r24: a "the file is done" sentence before the marker
                              r"(?:the|this) file is (?:now |already )?complete\b|this completes\b|"
                              r"that(?:'s|’s| is) the (?:complete|whole|full) file\b|no further (?:content|changes)\b)",
                              re.IGNORECASE)
#: drop_chatter's own test for a long round's opener line (write_continuation)
_OPENER_NAMES_RE = re.compile(r"\b(?:file|rest|remaining|continu\w*|code)\b", re.IGNORECASE)
_PROSE_NAMES = frozenset({"readme", "changelog", "license", "notes", "authors", "contributing"})


def _file_kind(path: str) -> str:
    """'markdown', 'prose' or 'code' — by the file's name (``README.md.j2`` is
    markdown; an extensionless README is prose; ``notes.js``, a Makefile and
    an .html page are code, r18)."""
    name = str(path or "").replace("\\", "/").rsplit("/", 1)[-1].lower()
    parts = name.split(".")
    exts = set(parts[1:]) if len(parts) > 1 else set()
    if exts & {"md", "markdown", "mdx", "rmd", "qmd"}:
        return "markdown"
    if exts & _PROSE_EXTS or (not exts and parts[0] in _PROSE_NAMES):
        return "prose"
    return "code"


def _is_opener(cont: str, first: str) -> bool:
    """A chatter start ("Here's", "Sure,", "I'll continue") on a line with no
    code character — parens and quotes are prose here ("(continued)",
    "Here's"); "continuing = False" is code (r21)."""
    from .write_continuation import _CHATTER_RE
    return bool(_CHATTER_RE.match(cont) or _MORE_OPENERS_RE.match(cont)) and not _CODE_CHAR_RE.search(first)


def _strip_opener_ok(path: str, cont: str) -> bool:
    """A continuation round's first line goes as chatter only in a CODE file,
    only as the shared rule's opener (a chatter start that names the file /
    the rest / the code, or ends with ":") with more lines after it, and only
    when it reads as prose — no code character in it (r16: "continuing =
    True" went; a GUIDE.md's "Here is the code:" is content; r18: "Here's"
    and "`app.py`" are prose — ' and ` are not code characters here)."""
    first, _, rest = cont.lstrip("\r\n").partition("\n")
    if not _is_opener(cont, first) or not rest.strip():   # (called for code files only)
        return False
    return bool(_OPENER_NAMES_RE.search(first) or first.rstrip().endswith(":"))


def _clean_file_round(raw: str, *, wrapped: bool = False) -> Tuple[str, bool]:
    """(text, done) from a file call's reply — the file's own text is never
    touched (review r15: `write_continuation.clean_round` removed a
    ``<think>`` anywhere — code that handles LLM output lost lines — and
    unwrapped fences round by round): only a LEADING think block goes
    (thinking is off; a template may still emit one), and the end marker
    ends the file at the FIRST line that ends with it — on its own or glued
    (``};<<<END OF FILE>>>``); everything after that line goes (r17/r18:
    "Done!", a usage example, a repeated marker), except a bare fence on the
    very next non-blank line when the file opened with a wrapper (r19), kept
    for `_unwrap_file`. A marker in bold, in backticks or with a full stop
    (``**<<<END OF FILE>>>**``) is the marker (r19/r21); text glued before
    it keeps its line verbatim (r20: ``from m import *`` lost its star). A string constant
    that holds it (``END = "<<<END OF FILE>>>"``) is text; a marker line
    inside a multi-line string ends the file (residue: the marker is the
    protocol's own sentinel)."""
    from .write_continuation import END_MARKER
    t = _THINK_HEAD_RE.sub("", str(raw or ""), count=1)
    lines = t.split("\n")
    fence_re = re.compile(r"^[ \t]*`{3,}[ \t]*\r?$")
    for i, ln in enumerate(lines):
        alone = re.fullmatch(r"\s*([*`]*)\s*" + re.escape(END_MARKER) + r"\s*([*`.]*)\s*", ln)
        core = ln.rstrip().rstrip("*.").rstrip()       # r21: "x = 1<marker>." too
        if alone and alone.group(1).count("`") == alone.group(2).count("`"):
            before = ""                       # alone: **bold**, `ticked`, with a full stop (r19/r21)
        elif core.endswith(END_MARKER):
            before = core[:-len(END_MARKER)]  # glued — kept verbatim (r20: "from m import *", a template's `)
        else:
            continue
        head = "\n".join(lines[:i] + ([before] if before.strip() else []))
        head += "\n" if head else ""
        nxt = next((x for x in lines[i + 1:] if x.strip()), "")
        if wrapped and fence_re.match(nxt):
            head += nxt.rstrip("\r") + "\n"
        return head, True
    return t, False


_WRAP_OPEN_RE = re.compile(r"\A(`{3,})([\w.+-]*)[ \t]*\r?\n")
_ANY_FENCE_RE = re.compile(r"^[ \t]*```")


def _unwrap_file(text: str, path: str) -> str:
    """The prompt says no fence; a file the model wrapped anyway loses its
    wrapper — ONE rule on the whole finished text, as the write tool's own
    unwrap (review r9–r14: five round-by-round rules each cut real text or
    kept a fence): a fence on the first line and a bare close of the same
    backticks on the last line both go; a first-line fence that no line
    closes (the file's fences are unbalanced) goes alone. A markdown file's
    own blocks are not wrappers: there only ```markdown/```md or a
    4+-backtick fence is. Anything else (prose around a fence, a fence
    mid-file) is left to the write tool's syntax check and the gates."""
    if _WRAP_OPEN_RE.match(text.lstrip(" \t\r\n")):
        text = text.lstrip(" \t\r\n")       # r19: a blank line or space before the wrapper
    m = _WRAP_OPEN_RE.match(text)
    if not m:
        return text
    ticks, info = len(m.group(1)), m.group(2).lower()
    kind = _file_kind(path)
    markdown = kind == "markdown"
    if markdown and ticks < 4 and info not in ("markdown", "md"):
        return text
    if kind == "prose":
        return text                       # r15/r17: a .txt/.rst/README file's first fence is its own
    lines = text.rstrip("\r\n").split("\n")
    closes = len(lines) > 1 and bool(re.match(r"^[ \t]*`{%d}[ \t]*\r?$" % ticks, lines[-1]))
    odd = sum(1 for ln in lines if _ANY_FENCE_RE.match(ln)) % 2
    if closes and (not odd or not markdown):
        # a code file: first + last (r15: a ``` inside a string made the count
        # odd, the close stayed, and the write tool's own unwrap cut text)
        return "\n".join(lines[1:-1]) + "\n"
    if odd:
        return text[m.end():]             # an opener no line closes (a README's own last block keeps its close)
    return text


def _feedback_for(path: str, feedback: str, is_own: bool) -> str:
    """The last failure for ONE generated file's call: the leaf's own files
    get it always (r21: a pytest trace names only the test file, the bug is
    in the code — r20's routing by name told the code "keep it"), a new file
    when the failure names it (r20: b.js, refused before it was written, got
    nothing)."""
    if not feedback:
        return ""
    base = path.replace("\\", "/").rsplit("/", 1)[-1]
    named = bool(base) and re.search(r"(?<![\w.-])" + re.escape(base) + r"(?![\w-])", feedback) is not None
    return feedback if is_own or named else ""


async def _disk_text(tool_runner: ToolRunner, path: str, sent: str) -> str:
    """The file as the write tool LEFT it (r18: it unwraps a fenced payload
    and the read normalises CRLF — `own` held the sent text, so a retry read
    the leaf's own file as an owner's edit and refused the regeneration).
    Unreadable: the sent text (every guard then applies — a refusal costs a
    retry)."""
    live, _ = await _read_live_file(tool_runner, path)
    return live if live is not None else sent


def _is_generate_entry(f) -> bool:
    return isinstance(f, dict) and bool(str(f.get("generate") or "").strip())


def _norm_path(p: str) -> str:
    p = str(p or "").strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return os.path.normpath(p) if p else p


#: the private mark `_materialize_generated` puts on a ``content`` entry it
#: wrote from a ``generate`` one (`_apply_file`'s own-file rule reads it)
_GENERATED_KEY = "_ghost_generated"


async def _materialize_generated(llm, model: str, files: list, *, description: str, ledger: str,
                                 existing: Dict[str, str], own: Dict[str, str],
                                 pending: Dict[str, Tuple[str, str]], tool_runner: Optional[ToolRunner],
                                 research_context: Optional[Dict[str, str]],
                                 constraints: Optional[List[str]], is_background: bool,
                                 deadline: float, feedback: str = "") -> Tuple[list, Optional[str]]:
    """§4SD: every ``generate`` entry becomes a ``content`` entry, written by
    `_generate_file`; each file's call is shown the project's files, the plan's
    other written-in-full files and those generated before it, so they agree.

    ``generate`` is for NEW files, and for ones this leaf wrote itself
    (``own``: path → the text written; a retry regenerates them, shown its
    previous version and the failure): an existing one changes through
    ``edits``/``append`` (a rewrite from a description would drop what it
    does; the snapshot may be truncated). Every entry is checked before any
    file is generated. A file generated but not yet written waits in
    ``pending`` (path → (brief, text)) — a later file's failure does not lose
    it; the next attempt reuses it for the same brief (review r10/r11). On a
    failure the generated entries that completed are returned with the reason."""
    files = [{k: v for k, v in f.items() if k != _GENERATED_KEY} if isinstance(f, dict) else f
             for f in files]                  # review r12: the mark is the system's, never the model's
    gen = [f for f in files if _is_generate_entry(f)]
    if not gen:
        return files, None
    known = {_norm_path(k) for k in existing}
    others = {_norm_path(str(f.get("path") or "")) for f in files
              if isinstance(f, dict) and not _is_generate_entry(f)}
    seen: set = set()
    for f in gen:
        if not isinstance(f.get("path"), str):
            return files, f"a `generate` entry's path is not a string ({f.get('path')!r})"
        path = f["path"].strip()
        np_ = _norm_path(path)
        if not np_ or np_ in (".", "..") or np_.startswith("../"):
            return files, f"a `generate` entry has no usable path ({path!r})"
        if any(f.get(k) for k in ("content", "append", "edits")):
            # review r10: {"generate": "the engine", "content": "// TODO"} wrote the stub
            return files, (f"{path}: an entry has `generate` AND `content`/`append`/`edits` — "
                           "use exactly one")
        if np_ in others:
            return files, (f"{path} is named by a `generate` entry and another entry — name each "
                           "file once")
        if np_ in seen:
            # review r11: "part one"/"part two" of one file — the second was the first again
            return files, (f"{path} is named by two `generate` entries — generate it once "
                           "(it has its own output budget), or add to it later with `append`")
        seen.add(np_)
        if np_ in own:
            # r19: the leaf's own file — unless the owner edited it since (the
            # regeneration is shown the leaf's text and would drop the edit);
            # unreadable (over the read cap): every write guard still judges it
            live, _ = await _read_live_file(tool_runner, path) if tool_runner is not None else (None, "")
            if live is None or live == own[np_]:
                continue
            return files, (f"{path} was changed since this task wrote it (by the owner?) — keep that "
                           "change: fix it with `edits` or `append`, not `generate`")
        exists = np_ in known
        if not exists and tool_runner is not None:
            live, lerr = await _read_live_file(tool_runner, path)
            if lerr != "missing" and live is None and await _is_root_fallback(tool_runner, path):
                lerr = "missing"              # §4QA-d r21: a big ROOT file of that name is not this one
            exists = not (lerr == "missing")  # unknown = do not rewrite blind
        if exists:
            return files, (f"{path} already exists: `generate` is for NEW files — change it "
                           f"with `edits` (or add to it with `append`)")
    shown: Dict[str, str] = dict(existing)
    for f in files:
        if isinstance(f, dict) and not _is_generate_entry(f) and isinstance(f.get("content"), str):
            shown[str(f.get("path") or "").strip()] = f["content"]   # a sibling written in full
    out: list = []
    done_entries: list = []
    generated: Dict[str, str] = {}          # r20: this spec's generated files — shown FIRST (the 40k cut)
    for f in files:
        if not _is_generate_entry(f):
            out.append(f)
            continue
        path = str(f.get("path") or "").strip()
        np_ = _norm_path(path)
        brief = str(f.get("generate") or "").strip()
        hit = pending.get(np_)
        text = hit[1] if hit else None    # same brief: `pending` holds only this plan's entries (r21)
        if text is None:
            ref = dict(reversed(list(generated.items())))
            ref.update({k: v for k, v in shown.items() if k not in ref})
            text, reason = await _generate_file(
                llm, model, f, description=description, ledger=ledger, plan=files,
                shown_files=ref, research_context=research_context,
                constraints=constraints, is_background=is_background, deadline=deadline,
                previous=own.get(np_, ""), feedback=_feedback_for(path, feedback, np_ in own))
            if reason:
                return done_entries, reason
            pending[np_] = (brief, text)
        generated[path] = text            # the later files' reference (shown first, r20)
        entry = ({k: v for k, v in f.items() if k != "generate"}
                 | {"content": text, _GENERATED_KEY: True})
        out.append(entry)
        done_entries.append(entry)
    return out, None


async def _apply_edits(tool_runner: ToolRunner, path: str, edits: list,
                       touched: Optional[set] = None) -> Optional[str]:
    """Apply find/replace (or after/insert) edits to an existing file. Returns
    a failure reason, or None if at least one edit applied cleanly. ``touched``
    collects paths a mutating op was issued for, so the retry loop knows what
    to re-read from disk (a failed batch can leave EARLIER edits applied)."""
    applied = 0
    touched = touched if touched is not None else set()
    cur: Optional[str] = None   # live content; None = unknown / stale
    for ed in edits[:20]:
        if not isinstance(ed, dict):
            continue
        find, rep = ed.get("find"), ed.get("replace")
        after = ed.get("after")
        before = ed.get("before")
        ins = ed.get("insert")
        anchor: Optional[str] = None
        insert_after = False
        if isinstance(find, str) and find and isinstance(rep, str):
            args = {"operation": "replace", "path": path,
                    "content": find, "replace_with": rep}
        elif isinstance(after, str) and after and isinstance(ins, str):
            anchor, insert_after = after, True
            args = {"operation": "replace", "path": path,
                    "content": after, "replace_with": after + "\n" + ins}
        elif isinstance(before, str) and before and isinstance(ins, str):
            # Insert BEFORE an anchor (e.g. "</body>") — the common HTML case.
            anchor, insert_after = before, False
            args = {"operation": "replace", "path": path,
                    "content": before, "replace_with": ins + "\n" + before}
        else:
            continue
        if anchor is not None:
            # file_system's exact-match replace substitutes EVERY occurrence,
            # so an insert anchored on a tag that appears 3x landed 3 copies
            # of the fragment (2026-07-20 review). When the anchor is
            # ambiguous, splice ONE fragment at the FIRST occurrence and
            # write the result ourselves; a unique (or non-byte-matching)
            # anchor keeps the replace path — its fuzzy matching and syntax
            # rollback are worth more than the splice.
            if cur is None:
                cur, _rerr = await _read_live_file(tool_runner, path)
            if isinstance(cur, str) and cur.count(anchor) > 1:
                i = cur.index(anchor)
                pos = i + len(anchor) if insert_after else i
                frag = ("\n" + ins) if insert_after else (ins + "\n")
                new = cur[:pos] + frag + cur[pos:]
                if len(new) > MAX_CONTENT_CHARS:
                    return (f"{path} would exceed {MAX_CONTENT_CHARS} chars "
                            f"({len(new)}) — split this feature into a "
                            f"separate file")
                try:
                    out = await tool_runner("file_system", {
                        "operation": "write", "path": path, "content": new})
                except Exception as e:
                    touched.add(path)   # state unknown — refresh next attempt
                    return f"edit on {path} errored: {e}"
                if _looks_like_write_error(out):
                    return f"edit on {path} was rejected: {_short(out)}"
                touched.add(path)
                cur = new
                applied += 1
                last_out = out
                continue
        try:
            out = await tool_runner("file_system", args)
        except Exception as e:
            touched.add(path)   # state unknown — refresh next attempt
            return f"edit on {path} errored: {e}"
        if not _op_ok(out):
            # a no-match replace provably did not mutate — no refresh needed
            #
            # ⚠ NAME THE ACTUAL REASON (2026-08-11, §4AT-F). Every non-SUCCESS
            # replace was reported as "anchor not found", which is a diagnosis,
            # not an observation — and often the wrong one. `file_system`
            # rejects a replace for at least three distinct reasons, and the
            # feedback string is prefixed with "YOUR PREVIOUS ATTEMPT FAILED:"
            # into the next spec prompt, so a wrong label steers every retry at
            # the wrong hypothesis. Live on req 6e9efd6a: a syntax-regression
            # rollback ("REJECTED: that replace would introduce a syntax error
            # and was NOT applied … unexpected indent") was relabelled
            # "anchor not found", and the model spent 20 minutes and 14
            # attempts hunting for a better ANCHOR while the defect was its
            # replacement's INDENTATION.
            return (f"edit on {path} did not apply "
                    f"({_replace_failure_kind(out)}): {_short(out)}")
        touched.add(path)
        applied += 1
        last_out = out
        cur = None   # the replace mutated the file in a way we didn't model
    if applied == 0:
        return f"no usable edits for {path}"
    # The LAST edit's result reflects the file's final on-disk state — the
    # replace path appends the same syntax diagnostic writes get. (Replaces
    # that would INTRODUCE breakage are already REJECTED by file_system's
    # rollback guard and caught by _op_ok above; this catches edits that
    # leave an already-broken file still broken.)
    sfail = _syntax_fail_reason(path, last_out)
    if sfail:
        return sfail
    return None


def _merge_feedback(new: str, old: str) -> str:
    """The new failure first, the one the retry was fixing kept (bounded)."""
    if not old or old == new or old in new:
        return new
    return f"{new}\n\nEARLIER: {_short(old, 1500)}"


async def _apply_file(tool_runner: ToolRunner, fspec: dict,
                      existing_files: Dict[str, str],
                      fresh: Optional[set] = None,
                      touched: Optional[set] = None,
                      own: Optional[Dict[str, str]] = None) -> Tuple[Optional[str], Optional[str]]:
    """Apply one file entry. Returns ``(written_path|None, fail_reason|None)``.
    A blank entry returns (None, None) — skipped, not a failure.

    ``existing_files`` is the PROMPT snapshot — possibly truncated by the
    gatherer's per-file/budget caps, or missing the path entirely (file-count
    cap) — never a source of truth for what is on disk. ``fresh`` marks paths
    whose snapshot entry IS authoritative (just written or re-read from disk
    this task); the append path trusts those and live-reads everything else.
    ``touched`` collects paths whose ON-DISK state (possibly) changed, so a
    failed attempt's retry re-reads exactly those."""
    if not isinstance(fspec, dict):
        return (None, None)
    path = (fspec.get("path") or "").strip()
    if not path:
        return (None, None)
    snap = existing_files if isinstance(existing_files, dict) else {}
    fresh = fresh if fresh is not None else set()
    touched = touched if touched is not None else set()
    old = snap.get(path)

    # APPEND — the easiest, safest incremental primitive: the model emits ONLY
    # the new snippet and the executor places it (for HTML, just before the
    # closing tag so scripts/markup land inside the document; otherwise at the
    # end). A strict superset, so it can never regress. (Re-emitting the whole
    # file choked the small model — empty specs — and brittle edit anchors
    # failed; smart append sidesteps both.)
    append = fspec.get("append")
    if isinstance(append, str) and append.strip():
        # Base the append on the LIVE file, not the snapshot: writing
        # old+snippet from a truncated snapshot AMPUTATED the on-disk tail,
        # and from an absent one (>file-cap projects) REPLACED the whole
        # file with just the snippet (2026-07-20 review). The invariant: an
        # append must never shorten or replace the existing file.
        if path in fresh and isinstance(old, str):
            base = old
        else:
            live, lerr = await _read_live_file(tool_runner, path)
            if lerr == "missing":
                base = ""
            elif live is None:
                return (None, f"append to {path} refused: the current on-disk "
                              f"content could not be read ({lerr}) — appending "
                              f"from a possibly-truncated snapshot could delete "
                              f"the file's tail. Put this feature in a separate "
                              f"NEW file instead")
            else:
                base = live
        # Python append guard (2026-08-01): refuse BEFORE writing when the
        # append would duplicate an existing implementation or leave a
        # working file unparseable — the post-write syntax check catches the
        # breakage only after the working file on disk is already gone.
        pyguard = _py_append_guard(base, append, path)
        if pyguard:
            return (None, pyguard)
        # §4EI pre-write guards: elided code never parses, and a top-level
        # element that re-uses an existing id/data-* key is unreachable.
        placeholder = _placeholder_elision_reason(append, path)
        if placeholder:
            return (None, placeholder)
        dupkey = _html_duplicate_key_reason(base, append, path)
        if dupkey:
            return (None, dupkey)
        new = _smart_append(base, append.strip(), path)
        # §4QA: the same floor as a full write — an append onto nothing that
        # leaves only "..." is still a hollow document
        from .task_deliverable import elided_document_reason
        hollow = elided_document_reason(new, path)
        if hollow:
            return (None, f"{path} refused BEFORE writing: {hollow}")
        if len(new) > MAX_CONTENT_CHARS:
            # Never truncate (it would cut off closing tags and break the
            # file) — fail so the task can be split or done by hand.
            return (None, f"{path} would exceed {MAX_CONTENT_CHARS} chars "
                          f"({len(new)}) — split this feature into a separate file")
        try:
            out = await tool_runner(
                "file_system", {"operation": "write", "path": path, "content": new})
        except Exception as e:
            touched.add(path)   # state unknown — refresh next attempt
            return (None, f"append failed for {path}: {e}")
        if _looks_like_write_error(out):
            return (None, f"append rejected for {path}: {_short(out)}")
        touched.add(path)
        sfail = _syntax_fail_reason(path, out)
        if sfail:
            # Revert (2026-09-03, §4EI). An append is a pure addition, so the
            # pre-append content is a complete, known state — put it back.
            # Left on disk, the broken block poisoned every retry: the model
            # answered "line 387 does not parse" with ANOTHER append, the
            # file grew 18K→22K, and the same line failed four times.
            pre = _syntax_error_before_block(base, path, sfail)
            if pre is not None:
                # The reported line precedes the appended block: the file was
                # broken BEFORE this append. Reverting would hide that and
                # blame the new block — keep the taint flow and say where the
                # fault really is.
                return (None, f"{sfail} NOTE: line {pre} is in the EXISTING "
                              f"file, before the block you appended — the "
                              f"file was already broken; fix that line with "
                              f"`edits`.")
            if base.strip() and await _restore_file(tool_runner, path, base):
                snap[path] = base
                fresh.add(path)
                return (None, _reverted_append_reason(path, out))
            return (None, sfail)
        # Record the just-written content so a SECOND append to the same path
        # in this spec builds on it instead of the pre-write state — two
        # appends computed from the same stale base made the second write
        # discard the first (2026-07-20 review).
        snap[path] = new
        fresh.add(path)
        return (path, None)

    edits = fspec.get("edits")
    if isinstance(edits, list) and edits and path not in snap:
        # §4QA-d: the snapshot holds at most 12 files, so an `edits` entry for
        # a real file beyond them was refused, and the feedback ("name it in
        # `files` so it is read first") pointed at a read that did not exist:
        # every retry failed the same way (43c19f8b7511). Read it now: edits
        # are exact-match, so a guessed anchor fails without writing, and the
        # path goes to `touched` so the next attempt's prompt shows the file.
        live, lerr = await _read_live_file(tool_runner, path)
        if lerr != "missing" and live is None and await _is_root_fallback(tool_runner, path):
            lerr = "missing"
        if lerr == "missing":
            return (None, f"{path} does not exist, so its {len(edits)} edit(s) "
                          f"cannot apply — write it with `content` (or, over ~100 lines, "
                          f"`generate`)")
        touched.add(path)
        if live is None:
            # r21 m2: too large / SAMPLE ONLY — the file exists and exact-match
            # edits need no copy of it: apply them, without a snapshot entry
            reason = await _apply_edits(tool_runner, path, edits, touched=touched)
            return (None, reason) if reason else (path, None)
        snap[path] = live
        fresh.add(path)
    if isinstance(edits, list) and edits and path in snap:
        reason = await _apply_edits(tool_runner, path, edits, touched=touched)
        if reason:
            return (None, reason)
        # Edits mutate the file in ways we didn't model in-memory — drop any
        # fresh claim so a later append re-reads instead of trusting it.
        fresh.discard(path)
        return (path, None)

    content = fspec.get("content")
    if content is None:
        # ⚠ A DROPPED EDIT MUST NOT READ AS "NOTHING TO DO" (2026-08-11,
        # §4AT-F). `edits` are only applied when `path in snap`, and the
        # prompt snapshot is hard-capped at 12 files. On a larger project an
        # `edits` entry for an unsnapshotted file fell through to here,
        # `content` is None, and this returned (None, None) — "skipped, not a
        # failure": zero tool calls, no reason, nothing in `written`. A
        # sibling entry that DID apply then made the spec look successful, so
        # the leaf closed DONE claiming both landed.
        # (§4QA-d: an unsnapshotted file is now read above, so `path in snap`
        # holds whenever `edits` is non-empty and this is unreachable for it.)
        return (None, None)
    if not isinstance(content, str):
        content = str(content)
    # Double-escaped spec content repair (2026-08-01): the model sometimes
    # emits the file as a one-line JSON-escaped string that survives
    # extraction verbatim. Writing it as-is lands a broken file on disk
    # that the retry loop demonstrably cannot fix (req 50855398: three
    # attempts, same corruption). The repair is accepted ONLY when the
    # unescaped text ast-parses — see _maybe_unescape_double_escaped.
    _repaired = _maybe_unescape_double_escaped(content, path)
    if _repaired is not None:
        logger.warning(
            "coding_executor: %s content arrived double-escaped (one line, "
            "literal \\n sequences) — unescaped repair parses; writing the "
            "repaired text instead", path)
        content = _repaired
    if len(content) > MAX_CONTENT_CHARS:
        # Fail loudly — never silently slice a full file mid-code (it would cut
        # a function/closing tag and write BROKEN code). Mirrors the append
        # path's "never truncate" stance; the retry can split into files.
        return (None, f"{path} content is {len(content)} chars (> "
                      f"{MAX_CONTENT_CHARS}) — split it across multiple files "
                      f"instead of emitting one oversized file")
    placeholder = _placeholder_elision_reason(content, path)
    if placeholder:
        return (None, placeholder)
    # §4QA: a document of "..." is no deliverable (live: results/meta-reflection.md,
    # 3 bytes, closed DONE on `test -f`)
    from .task_deliverable import elided_document_reason
    hollow = elided_document_reason(content, path)
    if hollow:
        return (None, f"{path} refused BEFORE writing: {hollow}")

    # Truncation guard: an HTML file written via full `content` that has no
    # closing </html> was almost certainly cut off at the token cap (observed:
    # task 1's shell ended mid-JS, breaking every later append). Don't build on
    # a broken file — fail so the retry produces a smaller COMPLETE one.
    # §4SD r22: only a full PAGE (it opened <html>) — a Jinja child template
    # or a partial never closes one and was refused on every attempt
    if (path.lower().endswith((".html", ".htm")) and content
            and "<html" in content.lower() and "</html>" not in content.lower()):
        return (None, f"{path} looks truncated (missing </html>) — produce a "
                      "COMPLETE file with closing tags (a NEW file of more than "
                      "~100 lines: `generate` it — it gets its own output budget), or "
                      "a smaller one and add features in later tasks via append")

    # ⚠ NEVER JUDGE A REWRITE AGAINST A PREFIX (2026-08-11, §4AT-F).
    # `_gather_project_files` stores only a prefix once its 400 KB budget is
    # nearly spent, and that truncation used to be silent — so a 300 KB
    # index.html snapshotted as 4 KB made a 20 KB full rewrite look like
    # GROWTH (20000 > 4000×0.85) and the non-regression guard waved through an
    # overwrite that destroyed 280 KB. For `.py` the prefix usually fails
    # `ast.parse`, which disables the overwrite guard entirely.
    # A marked entry means "unknown baseline": re-read the file from disk,
    # which is what the append path already does for the same reason.
    if old and _snap_truncated_mark() in old:
        live, _lerr = await _read_live_file(tool_runner, path)
        if live is not None:
            old = live
        else:
            # Could not re-read: refuse rather than compare against a prefix.
            # A refusal costs one retry; a wrong pass costs the file.
            return (None, f"cannot verify a full overwrite of {path}: its "
                          f"snapshot is a truncated prefix and the file could "
                          f"not be re-read — use `edits` instead of `content`")
    elif path not in snap:
        # §4QA-d review: a file past the 12-file snapshot had NO baseline, so
        # both guards below were skipped — a 6 KB module with 200 defs was
        # replaced by 22 bytes. A file the snapshot never saw is not a new
        # file: read it, and refuse when it exists but cannot be read.
        # file_system answers a read with "--- p CONTENTS ---", "--- p SAMPLE
        # ONLY (…) ---" (a big long-line file: it EXISTS, r20) or "Error: …"
        # (too large, budget, permission); those refuse — any other text is
        # no read result at all and keeps the old no-baseline path.
        live, lerr = await _read_live_file(tool_runner, path)
        if live is not None and lerr is None:
            old = live
        elif (lerr != "missing" and str(lerr or "").lstrip().lower().startswith(
                ("error", "read errored", "---"))
              and not await _is_root_fallback(tool_runner, path)):
            return (None, f"cannot verify a full overwrite of {path}: it is not "
                          f"in the file snapshot and could not be read ({lerr}) "
                          f"— use `edits` instead of `content`")

    # §4SD review r10: a file this leaf generated itself, still as it wrote
    # it, is rewritten as the new file it was — the overwrite guards would
    # refuse a resampled (shorter) version with "use `append`"
    # (only for a regenerated entry — review r11: a later plain `content`
    # stub replaced a 40-function generated file — and only the size/anchor
    # guard: a regeneration that does not parse never replaces one that does)
    reg_old = old
    if own and old is not None and fspec.get(_GENERATED_KEY) and _norm_path(path) in own:
        # review r12: judged by the file ON DISK — a regeneration runs for
        # long; an owner's edit meanwhile is not this leaf's file any more
        live, _lerr = await _read_live_file(tool_runner, path)
        if live is None:
            # review r14: unreadable (over the read cap, a spent read budget)
            # — it cannot be shown to be still the leaf's: every guard applies
            # (a refusal costs a retry; a wrong pass costs the owner's edit)
            pass
        elif live == own[_norm_path(path)]:
            reg_old = None
        else:
            old = reg_old = live
    # Refuse a full-overwrite that would lose prior work on an existing file.
    reg = _regression_reason(reg_old, content)
    if reg:
        return (None, f"refused to overwrite {path}: {reg}")
    # Python overwrite guard (2026-08-01), sibling of the append guard:
    # never replace a PARSING .py file with content that does not parse —
    # refuse pre-write so the working file survives on disk. Brand-new /
    # already-broken files keep the post-write check (same feedback, and a
    # broken new file blocks nothing that worked before).
    if (path.lower().endswith(".py") and old and (old or "").strip()
            and _py_parse_error(old) is None):
        new_err = _py_parse_error(content)
        if new_err:
            return (None, f"write to {path} refused BEFORE writing: the new "
                          f"content does not parse ({new_err}) while the "
                          f"current file on disk parses cleanly. Fix the "
                          f"syntax (or use `edits`) and retry.")
    try:
        out = await tool_runner(
            "file_system", {"operation": "write", "path": path, "content": content})
    except Exception as e:
        touched.add(path)   # state unknown — refresh next attempt
        return (None, f"write failed for {path}: {e}")
    if _looks_like_write_error(out):
        return (None, f"write rejected for {path}: {_short(out)}")
    touched.add(path)
    sfail = _syntax_fail_reason(path, out)
    if sfail:
        return (None, sfail)
    snap[path] = content
    fresh.add(path)
    return (path, None)


# Kill-shaped fragments in a spec's `verify` command. A verify must TEST,
# not manage processes: the live 2026-07-25 incident (req 8ec512cf) ran
# `fuser -k 8101/tcp; node server.js &` as its verify — it murdered the
# SUPERVISED jj-calendar service, left an orphan node that died with the
# shell, and the operator's browser hit a dead port two turns later.
# `kill -0` (pure liveness probe, signals nothing) stays allowed.
# §4QA: a verify that only asks whether the file EXISTS (`test -f x`, `[ -s x ]`,
# `ls x`, plus `&& echo OK`) checks nothing the write did not already prove.
# One line only, and no command substitution: "echo 'running'\npytest" and
# "echo $(python t.py)" run real code (fresh-eye MAJOR-5).
_EXISTENCE_ONLY_VERIFY_RE = re.compile(
    r"^(?![^\n]*\n\s*\S)(?![\s\S]*(?:\$\(|`))\s*(?:(?:test\s+-[efs]|\[\s+-[efs])\s+\S+(?:\s+\])?|ls(?:\s+-\w+)*\s+\S+"
    r"|echo(?:[ \t]+[^&|;\n]*)?|true)"
    r"(?:\s*(?:&&|;)\s*(?:(?:test\s+-[efs]|\[\s+-[efs])\s+\S+(?:\s+\])?|ls(?:\s+-\w+)*\s+\S+"
    r"|echo(?:[ \t]+[^&|;\n]*)?|true))*\s*$")

_VERIFY_KILL_RE = re.compile(
    r"\bfuser\s+-[a-zA-Z]*k"          # fuser -k / -km <port>/tcp
    r"|\bpkill\b"
    r"|\bkillall\b"
    r"|\bkill\s+(?!-0\b)\S"           # kill <pid>, kill -9 …, kill $(…)
    r"|xargs\s+(?:-\S+\s+)*kill\b"
)


async def _run_verify(tool_runner: ToolRunner, spec: dict,
                      written: List[str]) -> Optional[str]:
    """Failure reason or None — see `_run_verify_ex` for whether it RAN."""
    reason, _ran = await _run_verify_ex(tool_runner, spec, written)
    return reason


async def _run_verify_ex(tool_runner: ToolRunner, spec: dict,
                         written: List[str]):
    """Gate the build with the model's shell ``verify`` command, if any.
    Returns a failure reason, or None when the build passes (or gave no
    verify).

    NOTE: there is deliberately NO per-task headless render. In an
    incremental / single-file build no intermediate task produces a complete
    working page — the shell exists before the apps, an app exists before the
    next — so rendering each task would fail legitimate in-progress work
    (observed live: the Core-shell task built real files, then a render of the
    appless shell "crashed" on a not-yet-defined reference, the retry returned
    an empty spec, and a good task was marked FAILED). The non-regression
    guard in ``_apply_file`` is the real protection against shallow/clobbering
    output; whole-app rendering belongs to a final verification task, not to
    every tick."""
    verify = (spec.get("verify") or "").strip() if isinstance(spec, dict) else ""
    if not verify:
        return None, False
    if _VERIFY_KILL_RE.search(verify):
        # Skip, don't fail: the FILES may be perfectly good — only the
        # verify command is malformed intent. Degrades to the no-verify
        # path; the request-level gates (WEB-EXEC via the running
        # service, unverified-mutation cap) still stand between an
        # untested build and a confident success.
        logger.warning(
            "coding_executor: verify command wants to kill processes — "
            "SKIPPED (a verify must test, not manage services; probe the "
            "already-running service instead). Command: %s",
            _short(verify, 160))
        return None, False                 # §4PG r3: skipped is not verified
    if _EXISTENCE_ONLY_VERIFY_RE.match(verify):
        # still RUN (a missing file must fail), but it is not verification
        try:
            vout = await tool_runner("execute", {"command": verify})
        except Exception as e:
            return f"verify errored: {e}", True
        from .project_advancer import _looks_like_failure
        if _looks_like_failure(vout):
            return f"verify failed: {_short(vout)}", True
        return None, False
    try:
        vout = await tool_runner("execute", {"command": verify})
    except Exception as e:
        return f"verify errored: {e}", True
    from .project_advancer import _looks_like_failure
    if _looks_like_failure(vout):
        return f"verify failed: {_short(vout)}", True
    return None, True


async def build_coding_task(
    context,
    description: str,
    *,
    tool_runner: Optional[ToolRunner],
    ledger: str = "",
    existing_files: Optional[Dict[str, str]] = None,
    research_context: Optional[Dict[str, str]] = None,
    single_file: bool = False,
    max_files: int = MAX_FILES,
    max_attempts: int = MAX_ATTEMPTS,
    is_background: bool = False,
    constraints: Optional[List[str]] = None,
    **_ignored,
) -> CodingResult:
    """Build one coding leaf: spec → write/edit (non-regressively) → verify,
    retrying once with feedback before failing. See module docs.

    ``existing_files`` ({path: content}) is the project's CURRENT workspace
    (captured before this task), so a leaf EXTENDS prior files instead of
    overwriting them. ``single_file`` strengthens the "grow one file" steer.

    ``is_background`` routes the spec-generation LLM call through the client's
    background lane (waits for foreground to clear, capped concurrency). The
    IDLE autoadvancer sets it so its spec calls defer to a user who starts
    typing mid-build; the user-initiated ``manage_projects autoadvance`` tool
    leaves it False — the user is actively waiting on that batch.
    """
    llm = getattr(context, "llm_client", None)
    _verify_ran_ok = False                   # §4PG r3
    if llm is None or tool_runner is None:
        return CodingResult(False, "coding executor unavailable (no llm/tool_runner)")
    # §4FG: the ONE seam every caller inherits — `GHOST_CODING_EXECUTOR=agentic`
    # or the project's metadata `executor: agentic` routes the leaf to the
    # bounded edit-test loop (core/coding_loop.py); default stays the spec
    # executor below until the leaf bench says otherwise.
    from .coding_loop import build_coding_task_agentic, executor_kind
    _pid = _ignored.get("project_id") or getattr(context, "current_project_id", None)
    if executor_kind(context, project_id=_pid, description=description,
                     existing_files=existing_files) == "agentic":
        return await build_coding_task_agentic(
            context, description, tool_runner=tool_runner, ledger=ledger,
            existing_files=existing_files, research_context=research_context,
            single_file=single_file, is_background=is_background,
            constraints=constraints, project_id=_pid)
    model = getattr(getattr(context, "args", None), "model", "default")
    # Work on a COPY: attempts update the snapshot in place (just-written
    # content, between-attempt disk refresh) and the caller's dict must not
    # be mutated under it.
    existing = dict(existing_files or {})
    fresh: set = set()          # paths whose `existing` entry mirrors disk
    prev_touched: List[str] = []

    last = "build failed"
    feedback = ""
    last_written: List[str] = []
    empty_responses = 0
    deadline = _leaf_deadline()             # §4SD: a leaf's wall clock
    own: Dict[str, str] = {}                # §4SD: path → generated text this leaf wrote (a retry may regenerate it)
    pending: Dict[str, Tuple[str, str]] = {}   # §4SD: path → (brief, generated text) not written yet
    async def _write_pending() -> List[str]:
        # r20: generated files a refused batch never reached are written out
        # (through every guard) when the time ends, as the OutOfTime path does
        kept_: List[str] = []
        for _np, (_brief, _text) in list(pending.items()):
            _p, _r = await _apply_file(tool_runner, {"path": _np, "content": _text, _GENERATED_KEY: True},
                                       existing, fresh, set(), own=own)
            if _p and not _r:
                kept_.append(_p)
                pending.pop(_np, None)
        return kept_

    async def _write_pending_on_stop() -> None:
        # §4SD open items r3 (review MAJOR): a stop (Slack's hard limit, the
        # owner's Stop) cancels the build — the finished files are written
        # first, shielded and bounded (a write does not read the tool stop)
        try:
            await asyncio.wait_for(asyncio.shield(_write_pending()), STOP_FLUSH_S)
        except BaseException:  # noqa: BLE001 — the stop goes on whatever happens here
            logger.warning("coding_executor: finished files not all written on the stop")

    try:
        for _attempt in range(max(1, max_attempts)):
            if _stop_requested():
                raise asyncio.CancelledError()
            if _attempt and _time_left_s(deadline) < GENERATE_MIN_ATTEMPT_S:
                kept = await _write_pending()
                wrote = f" — wrote {', '.join(kept)} (not verified yet)" if kept else ""
                if _attempt < 2:
                    # review r13: ONE attempt that ran out of room — a slow leaf,
                    # not a failing build: back to READY for the next step
                    return CodingResult(False, _short(f"out of time after one attempt{wrote} — {last}", 280),
                                        files=kept or last_written, detail=feedback, out_of_time=True)
                # review r10/r12: no spec call starts with seconds left; a build
                # that kept FAILING is a failure (FAILED), not a timeout
                return CodingResult(False, _short(f"no time for another attempt{wrote} — {last}", 280),
                                    files=kept or last_written, detail=feedback)
            if prev_touched:
                # The previous attempt touched files and failed — possibly with
                # PARTIAL edits already on disk. Re-read them so this attempt's
                # prompt and guards see the CURRENT state; re-rendering the stale
                # pre-edit excerpt made the model re-emit already-applied edits
                # that burned every retry on "anchor not found" (2026-07-20).
                await _refresh_existing(tool_runner, existing, fresh, prev_touched)
                prev_touched = []
            try:
                spec, was_empty = await _generate_build_spec(
                    llm, model, description, ledger,
                    existing_files=existing, single_file=single_file,
                    feedback=feedback, research_context=research_context,
                    is_background=is_background, constraints=constraints)
            except Exception as e:  # pragma: no cover - LLM/network variance
                from .llm import BackgroundDeferred as _BgDeferred
                if isinstance(e, _BgDeferred):
                    await _write_pending()      # r25: this plan's finished files first (as below)
                    raise                       # §4PG (R2-3): the owner is back — the advancer reopens the leaf
                return CodingResult(False, f"build-spec generation failed: {e}")

            # A fully EMPTY upstream response is contention/infra, not a code
            # problem: feedback can't fix it. Back off briefly and retry, but cap
            # these separately so we don't burn every attempt hammering the server
            # (observed live: 8 instant empty calls). Report it honestly on exhaust.
            if was_empty:
                empty_responses += 1
                if empty_responses >= MAX_EMPTY_RETRIES:
                    return CodingResult(
                        False,
                        f"LLM returned empty responses ({empty_responses}x) — likely "
                        "upstream contention/overload, not a code problem; retry later",
                        files=last_written)
                await asyncio.sleep(_EMPTY_BACKOFF_S)
                feedback = ""   # nothing useful to feed back about an empty response
                continue

            files = spec.get("files") if isinstance(spec, dict) else None
            # r21/r22: a generated text waits only for THIS plan's same entry — a
            # path the plan now writes another way, renames, re-briefs or drops
            # (a verify-only plan drops all) is not written later from an older
            # plan (the time gate wrote it over the newer attempt's file)
            _plan = {_norm_path(str(f.get("path") or "")): str(f.get("generate") or "").strip()
                     for f in (files if isinstance(files, list) else [])[:max_files] if _is_generate_entry(f)}
            _real_plan = (isinstance(files, list) and bool(files)) or (
                isinstance(spec, dict) and bool(str(spec.get("verify") or "").strip())
                and not _VERIFY_KILL_RE.search(str(spec.get("verify") or "")))   # r24: refused below — no plan
            for _k in [k for k, (b, _t) in pending.items() if _real_plan and _plan.get(k) != b]:
                pending.pop(_k, None)             # r23: an unusable reply is no plan — it drops nothing
            if not isinstance(files, list) or not files:
                # Empty files + a verify command means "nothing to write — the
                # deliverable already exists; just check it works". This happens
                # when a prior interactive turn already built the file and the
                # autoadvance tick re-picks the task (observed live: the Model
                # Architecture task FAILED in autoadvance though model.py existed
                # and ran). Honour the verify instead of failing on no-files.
                _verify = (spec.get("verify") or "").strip() if isinstance(spec, dict) else ""
                if _verify and _VERIFY_KILL_RE.search(_verify):
                    # _run_verify skips kill-shaped verifies as a PASS on the
                    # rationale "the files may be good" — but here there ARE no
                    # files: the verify is the only check, so skipping it would
                    # close the task on nothing (review finding, 2026-08-01).
                    last = ("verify-only spec refused: its verify command "
                            "manages processes (kill/pkill/fuser -k) and there "
                            "are no files, so nothing would be verified. Provide "
                            "a verify that TESTS the existing deliverable (run "
                            "the script, curl the already-running service).")
                    feedback = last
                    continue
                if _verify:
                    vfail, _vran = await _run_verify_ex(tool_runner, spec, [])
                    if vfail:
                        _note_nothink_verify_failure(llm, _short(description, 60))
                    if not vfail:
                        summary = (spec.get("summary") if isinstance(spec, dict) else "") \
                            or "verified existing deliverable (nothing to build)"
                        ledger_note = (spec.get("ledger") if isinstance(spec, dict) else "") or ""
                        return CodingResult(True, _short(summary, 300), files=[],
                                            ledger_note=_short(ledger_note, 200),
                                            verified=bool(_vran))   # §4PG: it ran; §4QA: an existence check is not a run
                    # Verify failed — fall through with that as the reason so the
                    # retry/feedback path gets the real error, not "no file spec".
                    last = vfail
                    feedback = vfail
                    continue
                msg = ("model produced no file spec for the task (a NEW file of more than "
                       "~100 lines: name it with `generate` instead of writing it in `content`)")
                # Don't let an empty retry overwrite a more informative failure
                # (e.g. a real verify error) from a prior attempt.
                if last == "build failed":
                    last = msg
                feedback = msg
                continue

            # §4SD: files named with `generate` are written now, one call each
            from .llm import BackgroundDeferred as _Deferred
            try:
                files, gfail = await _materialize_generated(
                    llm, model, files[:max_files], description=description, ledger=ledger,
                    existing=existing, own=own, pending=pending, tool_runner=tool_runner,
                    research_context=research_context, constraints=constraints,
                    is_background=is_background, deadline=deadline, feedback=feedback)
            except _Deferred:
                # r24: the owner kept the slot past the park — the files this plan
                # finished are written (through every guard), as when time runs out
                await _write_pending()
                raise
            if isinstance(gfail, OutOfTime):
                # review r10: no further attempt fits; the files that completed are
                # written (through every guard) so the next step builds on them
                # instead of generating them again — the unattended step's cap
                # (~44 min) is shorter than some files take
                kept: List[str] = []
                for f in files:
                    path, reason = await _apply_file(tool_runner, f, existing, fresh, set(), own=own)
                    pending.pop(_norm_path(str(f.get("path") or "")), None)
                    if path and not reason:
                        kept.append(path)
                kept += await _write_pending()    # r21: this plan's files an earlier attempt finished
                if _attempt >= 2:
                    # r19: a build that kept FAILING is a failure (FAILED) however
                    # its time ended — as the gate above
                    return CodingResult(False, _short(f"no time for another attempt — {last}", 280),
                                        files=kept or last_written, detail=feedback)
                return CodingResult(False, _short(gfail + (f" — wrote {', '.join(kept)} (not verified yet); "
                                                           "the task goes back to READY" if kept else ""), 280),
                                    files=kept or last_written, detail=feedback, out_of_time=True)
            if gfail:
                feedback = _merge_feedback(gfail, feedback)   # review r9: keep the reason the retry must fix
                last = gfail
                continue

            touched: set = set()
            written: List[str] = []
            fail: Optional[str] = None
            for f in files[:max_files]:
                path, reason = await _apply_file(tool_runner, f, existing, fresh, touched, own=own)
                _gen = isinstance(f, dict) and bool(f.get(_GENERATED_KEY))
                _np = _norm_path(str(f.get("path") or "")) if _gen else ""
                if reason:
                    fail = reason
                    if _gen:
                        pending.pop(_np, None)            # refused: generate it anew, not again
                        if (str(f.get("path") or "").strip() in touched
                                and isinstance(f.get("content"), str)):
                            # review r11 (r19: `path` is None here — the write landed, its check failed)
                            own[_np] = await _disk_text(tool_runner, str(f.get("path") or "").strip(),
                                                        f["content"])
                    break
                if path:
                    written.append(path)
                    if _gen:
                        pending.pop(_np, None)
                        own[_np] = await _disk_text(tool_runner, path, f["content"])   # §4SD: on disk
                    elif _norm_path(path) in own:
                        # review r12: its own later edit/append — the file is still the leaf's
                        _live, _ = await _read_live_file(tool_runner, path)
                        if _live is not None:
                            own[_norm_path(path)] = _live
                        # (unreadable — over the read cap: it stays the leaf's, r13)
            if fail:
                # r20: a failed batch can have changed the leaf's own files (edit 1
                # of 2 landed; an edit left it broken) — still the leaf's, not an
                # owner's edit the next `generate` must keep
                for _p in touched:
                    if _norm_path(_p) in own:
                        _live, _ = await _read_live_file(tool_runner, _p)
                        if _live is not None:
                            own[_norm_path(_p)] = _live
            prev_touched = sorted(touched)
            last_written = written
            if fail:
                last = fail
                # review r10: a refused regeneration replaced the verify error
                feedback = _merge_feedback(fail, feedback) if own else fail
                continue
            if not written:
                last = "build spec produced no writable files"
                feedback = last
                continue

            vfail, _verify_ran_ok = await _run_verify_ex(tool_runner, spec, written)
            _verify_ran_ok = bool(_verify_ran_ok) and not vfail
            if vfail:
                _note_nothink_verify_failure(llm, _short(description, 60))   # §4FJ
                last = vfail
                feedback = vfail
                continue

            # Post-build gates (2026-07-08, from the chess-session post-mortem).
            # Smoke first (mechanical, cheap): py_compile every written .py and
            # sweep Flask routes with test_client — three crash-on-first-touch
            # bugs shipped in one session because no handler was ever exercised.
            from .build_gates import constraint_gate, files_from_specs, smoke_gate
            sfail = await smoke_gate(tool_runner, written)
            if sfail:
                last = sfail
                feedback = sfail
                continue
            # Then the constraint audit (one background LLM call): the spec
            # verify checks "does it run"; this checks "is it what the user
            # ALLOWED" — the session's engine-instead-of-Ghost build passed
            # every mechanical check while violating the stated constraint.
            if constraints:
                cok, creason = await constraint_gate(
                    context, constraints, files_from_specs(files),
                    is_background=is_background)
                if not cok:
                    last = creason
                    feedback = creason
                    continue

            summary = (spec.get("summary") if isinstance(spec, dict) else "") or \
                f"wrote {', '.join(written)}"
            ledger_note = (spec.get("ledger") if isinstance(spec, dict) else "") or ""
            return CodingResult(True, _short(summary, 300), files=written,
                                ledger_note=_short(ledger_note, 200),
                                verified=bool(_verify_ran_ok))   # §4PG r3: the verify RAN and passed
        if _stop_requested():
            raise asyncio.CancelledError()
    except asyncio.CancelledError:
        # §4SD open items r3/r4 (review MAJORs): a stop (Slack's hard limit, the
        # owner's Stop) cancels the build at ANY await of an attempt — the
        # finished files waiting in `pending` are written first. r5: an owner's
        # (foreground) build only — the unattended step reopens its leaf without
        # waiting for this task, and a late flush landed in the next build
        # (r5: keyed on the caller, not `is_background` — a scheduled batch's
        # build is foreground and unattended)
        if not is_background and caller_waits_context.get():
            await _write_pending_on_stop()
        raise

    return CodingResult(False, _short(last, 280), files=last_written, detail=feedback)
