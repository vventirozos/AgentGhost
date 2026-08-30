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
leaf well (a spec call, N writes/edits, one verify), at most twice.
"""

from __future__ import annotations

import ast
import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("GhostAgent")

ToolRunner = Callable[[str, Dict[str, Any]], Awaitable[str]]

# 16 (was 8): a single leaf like "scaffold the app" can legitimately emit a
# dozen+ files in one coherent spec from a capable model; 8 silently dropped the
# overflow. The task tree still does the coarse decomposition, so this only
# bounds runaway specs, not normal multi-file work.
MAX_FILES = 16
# Reasoning ceiling for the streamed spec call: chain-of-thought still
# running with ZERO content at this size is the budget-burn shape (live
# failures ran 40-75K chars before eating the whole 16384-token budget);
# normal spec planning lands in 3-15K. Cutting at 30K saves ~3 minutes per
# runaway while never touching a legitimate think phase.
SPEC_REASONING_ABORT_CHARS = 30_000
# Upper bound on a written file. Must comfortably exceed a fully-accreted
# single-file app: with append, the result is old+new, and old can be ~200 KB
# (the gather cap). A tight 60 KB truncated the growing index.html mid-JS,
# cutting off </body></html> and breaking the page once the OS got big.
MAX_CONTENT_CHARS = 400_000
# 4 (was 2): one spec call + 3 feedback retries. A capable model often fixes a
# verify failure on attempt 3-4; the old hard stop at 2 converted a recoverable
# build into a batch-halting FAILURE. Each retry is fed the exact failure reason.
MAX_ATTEMPTS = 4
# A COMPLETELY empty upstream response (content=0 reasoning=0) is contention/
# infra, not the model failing — feedback retries can't fix it, so cap them
# separately (with a small backoff) rather than burning all MAX_ATTEMPTS
# hammering the server instantly (observed live: 8 empty calls in <3s).
MAX_EMPTY_RETRIES = 2
_EMPTY_BACKOFF_S = 1.5
_REGRESS_SHRINK_RATIO = 0.85   # a rewrite below this fraction of the old size regresses


@dataclass
class CodingResult:
    ok: bool
    summary: str
    files: List[str] = field(default_factory=list)
    ledger_note: str = ""
    detail: str = ""


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
    marker = f"--- {path} CONTENTS ---\n"
    idx = text.find(marker)
    if idx != -1:
        return text[idx + len(marker):], None
    if _looks_like_missing_file(text):
        return "", "missing"
    return None, _short(text, 140)


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
        "reliable way to grow the app.\n"
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
    r"(?<![\w./-])([\w][\w./-]{0,60}\.(?:py|js|mjs|cjs|json|html?|htm|css|md|"
    r"sh|yml|yaml|toml|txt))(?![\w.])", re.IGNORECASE)


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


async def _stream_spec_completion(llm, payload: Dict[str, Any],
                                  is_background: bool):
    """Stream the build-spec generation, accumulating the content and
    reasoning channels, and ABORT a runaway think phase early.

    The non-streaming call this replaces could not see a think-loop until
    the full 16384-token budget was burned (~4 minutes on the live box,
    observed twice in one request, 92a968fc 2026-07-25) — the guard then
    fired on the corpse. Streaming lets the abort land while the loop is
    still forming. Two abort conditions, both evaluated ONLY while the
    content channel is still empty (once the JSON spec starts, reasoning
    is done and generated code repeats lines legitimately):

      * ``_detect_thinking_loop`` — the exact-n-gram detector the main
        turn loop uses. Deliberately the ONLY heuristic here: the first
        deploy also ran the paragraph-repeat detector and it aborted the
        thinking of EVERY coding leaf in the first live autoadvance run
        (one at just 3,019 chars) — spec planning legitimately restates
        the task and field lists, so that detector stays out of this
        path (2026-07-25 second deploy).
      * ``SPEC_REASONING_ABORT_CHARS`` ceiling — reasoning still going
        with zero content at 30K chars is the budget-burn shape (the
        live failures ran 40-75K); cutting there saves ~3 minutes
        without touching normal 3-15K planning.

    Returns ``(content, reasoning, abort_reason)`` where ``abort_reason``
    is ``None`` (clean), ``"loop"`` or ``"ceiling"``. Stall/mid-stream
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
            {**payload, "stream": False}, is_background=is_background)
        msg = ((resp or {}).get("choices", [{}])[0].get("message", {})) or {}
        return (msg.get("content") or "",
                msg.get("reasoning_content") or "", None)
    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    c_len = r_len = 0
    next_probe = THINKING_LOOP_PROBE_EVERY
    aborted = None
    agen = streamer(payload, is_background=is_background)
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
            delta = (((obj.get("choices") or [{}])[0]) or {}).get("delta") or {}
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
                if r_len >= SPEC_REASONING_ABORT_CHARS:
                    aborted = "ceiling"
                    break
    finally:
        try:
            await agen.aclose()
        except Exception:  # noqa: BLE001 — closing a broken stream
            pass
    return "".join(content_parts), "".join(reasoning_parts), aborted


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
        '{"files":[{"path":"name.ext", <ONE OF content|append|edits>}],'
        '"verify":"shell cmd that exits 0 iff it works, or \\"\\"",'
        '"summary":"one line","ledger":"one durable fact or \\"\\""}\n'
        "Per file choose EXACTLY ONE of:\n"
        '  "content": the FULL file text — for a brand-NEW file ONLY. It must '
        "be COMPLETE and VALID (HTML must include </body></html>); a file that "
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
        "running (e.g. curl its port)."
    )
    user = f"TASK: {description}\n"
    if constraints:
        # User-mandated constraints from the project record. Before this
        # block the executor never saw them — the 2026-07-04 chess session's
        # first "coded AI opponent" violation was written by an autoadvance
        # leaf whose spec prompt contained no trace of "with YOU - Ghost
        # plays directly, not a generated chess AI".
        from ..utils.constraints import (
            PARTICIPANT_STEER, has_participant_constraint,
            render_constraint_block,
        )
        user += "\n" + render_constraint_block(
            list(constraints),
            header="EXPLICIT USER CONSTRAINTS (PROJECT-WIDE)") + "\n"
        if has_participant_constraint(list(constraints)):
            user += "\n" + PARTICIPANT_STEER + "\n"
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
    # JSON even starts.
    content, reasoning, loop_aborted = await _stream_spec_completion(llm, {
        "model": model,
        "messages": [{"role": "system", "content": sys_hint},
                     {"role": "user", "content": user}],
        "temperature": 0.3, "max_tokens": 16384,
    }, is_background)
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
            return True
        # A VERIFY-ONLY spec ("the deliverable already exists — just prove
        # it works") is a first-class answer (2026-08-01): the caller's
        # no-files path honours the verify and closes the task without
        # rebuilding. Before this, a verify-only spec in the content
        # channel was "not usable", so the reasoning-channel fallback
        # could clobber it and the retry loop steered the model into
        # re-implementing files that already worked.
        return bool(str(s.get("verify") or "").strip())

    spec = extract_json_from_text(content, repair_truncated=True) or {}
    if not _usable(spec) and reasoning:
        spec = extract_json_from_text(reasoning, repair_truncated=True) or {}
        if not _usable(spec):
            # Last resort: scan reasoning + content together (a spec split
            # across the closing </think> boundary).
            spec = extract_json_from_text(
                f"{reasoning}\n{content}", repair_truncated=True) or spec
    if not _usable(spec):
        # Raw-newline-in-string repair (see _repair_json_string_newlines).
        spec = _recover_spec_escaping_newlines(
            (content, reasoning), extract_json_from_text, _usable) or spec
    if not _usable(spec) and not content.strip() and reasoning.strip():
        # The think phase produced no JSON: the stream guard aborted it
        # (exact n-gram loop or the 30K reasoning ceiling — the reason is
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
        elif loop_aborted == "ceiling":
            logger.warning(
                "coding_executor: reasoning ceiling (%d chars) hit with no "
                "content — aborted mid-stream, retrying once with thinking "
                "disabled", SPEC_REASONING_ABORT_CHARS)
        else:
            logger.warning(
                "coding_executor: think block consumed the whole budget "
                "(content=0, reasoning=%d chars) — retrying once with "
                "thinking disabled", len(reasoning.strip()))
        resp = await llm.chat_completion({
            "model": model,
            "messages": [
                {"role": "system", "content": sys_hint +
                 "\nDo NOT emit a <think> block — output the JSON object directly."},
                {"role": "user", "content": user + "\n\n/no_think"},
            ],
            "temperature": 0.3, "max_tokens": 16384, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }, is_background=is_background)
        msg = ((resp or {}).get("choices", [{}])[0].get("message", {})) or {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        spec = extract_json_from_text(content, repair_truncated=True) or {}
        if not _usable(spec) and reasoning:
            spec = extract_json_from_text(reasoning, repair_truncated=True) or spec
        if not _usable(spec):
            spec = _recover_spec_escaping_newlines(
                (content, reasoning), extract_json_from_text, _usable) or spec
    was_empty = not content.strip() and not reasoning.strip()
    if not _usable(spec):
        # Diagnostic: the model returned no usable file spec. Log a window of
        # the raw output (BOTH channels) so we can see WHY (prose? broken
        # JSON-escaped code? truncated mid-think? — or a fully EMPTY upstream
        # response, content=0 reasoning=0, which is contention, not the model).
        raw = (content or reasoning).strip()
        logger.warning(
            "coding_executor: no file spec parsed (content=%d reasoning=%d%s). "
            "RAW head: %s ||| tail: %s",
            len(content.strip()), len(reasoning.strip()),
            " — EMPTY upstream response" if was_empty else "",
            raw[:400].replace("\n", "\\n"), raw[-200:].replace("\n", "\\n"))
    return spec, was_empty


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


async def _apply_file(tool_runner: ToolRunner, fspec: dict,
                      existing_files: Dict[str, str],
                      fresh: Optional[set] = None,
                      touched: Optional[set] = None) -> Tuple[Optional[str], Optional[str]]:
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
        new = _smart_append(base, append.strip(), path)
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
            return (None, sfail)
        # Record the just-written content so a SECOND append to the same path
        # in this spec builds on it instead of the pre-write state — two
        # appends computed from the same stale base made the second write
        # discard the first (2026-07-20 review).
        snap[path] = new
        fresh.add(path)
        return (path, None)

    edits = fspec.get("edits")
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
        if isinstance(edits, list) and edits:
            return (None,
                    f"{path} was not in the file snapshot (capped at 12 "
                    f"files), so its {len(edits)} edit(s) were NOT applied — "
                    f"name it in `files` so it is read first, or split this "
                    f"task")
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

    # Truncation guard: an HTML file written via full `content` that has no
    # closing </html> was almost certainly cut off at the token cap (observed:
    # task 1's shell ended mid-JS, breaking every later append). Don't build on
    # a broken file — fail so the retry produces a smaller COMPLETE one.
    if (path.lower().endswith((".html", ".htm")) and content
            and "</html>" not in content.lower()):
        return (None, f"{path} looks truncated (missing </html>) — produce a "
                      "COMPLETE, smaller file with closing tags; add features "
                      "in later tasks via append")

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

    # Refuse a full-overwrite that would lose prior work on an existing file.
    reg = _regression_reason(old, content)
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
_VERIFY_KILL_RE = re.compile(
    r"\bfuser\s+-[a-zA-Z]*k"          # fuser -k / -km <port>/tcp
    r"|\bpkill\b"
    r"|\bkillall\b"
    r"|\bkill\s+(?!-0\b)\S"           # kill <pid>, kill -9 …, kill $(…)
    r"|xargs\s+(?:-\S+\s+)*kill\b"
)


async def _run_verify(tool_runner: ToolRunner, spec: dict,
                      written: List[str]) -> Optional[str]:
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
        return None
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
        return None
    try:
        vout = await tool_runner("execute", {"command": verify})
    except Exception as e:
        return f"verify errored: {e}"
    from .project_advancer import _looks_like_failure
    if _looks_like_failure(vout):
        return f"verify failed: {_short(vout)}"
    return None


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
    if llm is None or tool_runner is None:
        return CodingResult(False, "coding executor unavailable (no llm/tool_runner)")
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
    for _attempt in range(max(1, max_attempts)):
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
                vfail = await _run_verify(tool_runner, spec, [])
                if not vfail:
                    summary = (spec.get("summary") if isinstance(spec, dict) else "") \
                        or "verified existing deliverable (nothing to build)"
                    ledger_note = (spec.get("ledger") if isinstance(spec, dict) else "") or ""
                    return CodingResult(True, _short(summary, 300), files=[],
                                        ledger_note=_short(ledger_note, 200))
                # Verify failed — fall through with that as the reason so the
                # retry/feedback path gets the real error, not "no file spec".
                last = vfail
                feedback = vfail
                continue
            msg = "model produced no file spec for the task"
            # Don't let an empty retry overwrite a more informative failure
            # (e.g. a real verify error) from a prior attempt.
            if last == "build failed":
                last = msg
            feedback = msg
            continue

        touched: set = set()
        written: List[str] = []
        fail: Optional[str] = None
        for f in files[:max_files]:
            path, reason = await _apply_file(tool_runner, f, existing, fresh, touched)
            if reason:
                fail = reason
                break
            if path:
                written.append(path)
        prev_touched = sorted(touched)
        last_written = written
        if fail:
            last = fail
            feedback = fail
            continue
        if not written:
            last = "build spec produced no writable files"
            feedback = last
            continue

        vfail = await _run_verify(tool_runner, spec, written)
        if vfail:
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
                            ledger_note=_short(ledger_note, 200))

    return CodingResult(False, _short(last, 280), files=last_written, detail=feedback)
