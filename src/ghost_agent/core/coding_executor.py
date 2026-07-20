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


def _op_ok(out: str) -> bool:
    """A file_system op reports ``SUCCESS: …`` on success; a no-match replace
    returns ``SYSTEM INSTRUCTION: … NOT found``. So success == a SUCCESS head."""
    return "success" in (out or "").lower()[:200]


def _looks_like_write_error(out: str) -> bool:
    """Conservative write-failure check (writes rarely fail; don't abort a good
    build on a chatty success message). ``SYSTEM INSTRUCTION:`` heads are
    file_system REFUSALS (e.g. the empty-content write refusal) — missing them
    closed tasks DONE with the file never written (2026-07-20 review)."""
    head = (out or "").strip()[:80].lower()
    return (
        not out
        or head.startswith("system error")
        or head.startswith("system instruction")
        or head.startswith("error:")
        or "security error" in head
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
        "task needs; prefer a runnable verify that exercises what you built."
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
    if feedback:
        user += (f"\nYOUR PREVIOUS ATTEMPT FAILED: {feedback}\n"
                 "Produce a corrected spec that fixes exactly this.\n")
    resp = await llm.chat_completion({
        "model": model,
        "messages": [{"role": "system", "content": sys_hint},
                     {"role": "user", "content": user}],
        # 16384 (raised from 8192, itself raised from 4096). The output is a
        # JSON spec with a COMPLETE file embedded as `content` — a cut mid-string
        # yields invalid JSON and a truncated file. 8192 (~24-32 KB) was far
        # below the 400 KB this module permits a file to be, and a reasoning
        # model spends part of the budget in `reasoning_content` before the JSON
        # even starts — so a brand-new full file plus a think preamble could
        # still truncate. Give complete files real room.
        "temperature": 0.3, "max_tokens": 16384, "stream": False,
    }, is_background=is_background)
    msg = ((resp or {}).get("choices", [{}])[0].get("message", {})) or {}
    content = msg.get("content") or ""
    # Reasoning models (Qwen via llama.cpp) emit their chain-of-thought in a
    # separate `reasoning_content` field. When the think block consumes the
    # whole token budget without closing, the parser routes EVERYTHING there
    # and leaves `content` empty — so the JSON build spec lives entirely in the
    # reasoning channel. Reading only `content` then logged `len=0` and FAILED
    # the task with "model produced no file spec" (observed live: 5 coding
    # leaves in one project killed this way). Fall back to the reasoning
    # channel, mirroring core/agent.py (~2973) and project_research.py.
    reasoning = msg.get("reasoning_content") or ""
    from .agent import extract_json_from_text

    def _usable(s) -> bool:
        return bool(isinstance(s, dict) and isinstance(s.get("files"), list) and s.get("files"))

    spec = extract_json_from_text(content, repair_truncated=True) or {}
    if not _usable(spec) and reasoning:
        spec = extract_json_from_text(reasoning, repair_truncated=True) or {}
        if not _usable(spec):
            # Last resort: scan reasoning + content together (a spec split
            # across the closing </think> boundary).
            spec = extract_json_from_text(
                f"{reasoning}\n{content}", repair_truncated=True) or spec
    if not _usable(spec) and not content.strip() and reasoning.strip():
        # The think block consumed the whole max_tokens budget before the
        # JSON ever started: content is empty and everything sits in the
        # reasoning channel as prose (observed live 2026-07-06: content=0,
        # reasoning=40-62k chars, twice in one project — each a ~5-minute
        # generation lost). Thinking earns its cost on attempt 1, but once
        # it has eaten the budget the only productive move is ONE retry with
        # thinking off — same recipe as project_research._llm_call and
        # dream.py: /no_think soft-switch + enable_thinking=False
        # hard-switch + a system nudge.
        logger.warning(
            "coding_executor: think block consumed the whole budget "
            "(content=0, reasoning=%d chars) — retrying once with thinking "
            "disabled", len(reasoning.strip()))
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
            return f"edit on {path} did not apply (anchor not found): {_short(out)}"
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
        return (None, None)
    if not isinstance(content, str):
        content = str(content)
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

    # Refuse a full-overwrite that would lose prior work on an existing file.
    reg = _regression_reason(old, content)
    if reg:
        return (None, f"refused to overwrite {path}: {reg}")
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
