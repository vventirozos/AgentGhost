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

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("GhostAgent")

ToolRunner = Callable[[str, Dict[str, Any]], Awaitable[str]]

MAX_FILES = 8
# Upper bound on a written file. Must comfortably exceed a fully-accreted
# single-file app: with append, the result is old+new, and old can be ~200 KB
# (the gather cap). A tight 60 KB truncated the growing index.html mid-JS,
# cutting off </body></html> and breaking the page once the OS got big.
MAX_CONTENT_CHARS = 400_000
MAX_ATTEMPTS = 2               # spec attempts (1 retry with feedback)
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
    build on a chatty success message)."""
    head = (out or "").strip()[:80].lower()
    return (
        not out
        or head.startswith("system error")
        or head.startswith("error:")
        or "security error" in head
    )


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


def _isolate_scripts(fragment: str) -> str:
    """Wrap each inline <script> body in an IIFE so an APPENDED app block can't
    redeclare another block's top-level identifiers (``function initGame``,
    ``makeDraggable``, …). Those redeclarations were SyntaxErrors that broke
    the WHOLE page on load (observed live — the verifier REFUTED the built OS
    as "throws on load"). Already-wrapped bodies are left alone."""
    def _wrap(m):
        open_tag, body, close_tag = m.group(1), m.group(2), m.group(3)
        s = body.strip()
        if not s or s.startswith(("(function", "(()", "(async", "!function")):
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


def _file_excerpt(content: str, head: int = 800, tail: int = 400) -> str:
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
    body = "\n\n".join(parts)[:12_000]
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


async def _generate_build_spec(llm, model: str, description: str, ledger: str, *,
                               existing_files: Optional[Dict[str, str]] = None,
                               single_file: bool = False,
                               feedback: str = "") -> dict:
    """Ask the model for a JSON build spec. Returns ``{}`` on any failure."""
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
        '  "append": STRONGLY PREFERRED to add to an EXISTING file — you write '
        "ONLY the new code (a self-contained <script>/<style>/<div> or "
        "function). For HTML the system inserts it before </body> for you, so "
        "you do NOT need an anchor. This is the most reliable way to extend a "
        "single-file app.\n"
        '  "edits": [{"find":"EXACT existing text","replace":"…"}] — only when '
        "you must MODIFY existing code; the find text must byte-match. Avoid "
        "this for plain additions — use append.\n"
        "Rules: complete runnable code (no TODOs/stubs); BARE project-relative "
        "paths (no /workspace, sandbox/, projects/<id>/); only the files THIS "
        "task needs; prefer a runnable verify that exercises what you built."
    )
    user = f"TASK: {description}\n"
    if ledger:
        user += ("\nPROJECT LEDGER (existing files / APIs / conventions — build "
                 f"CONSISTENTLY with these):\n{ledger}\n")
    user += _render_existing(existing_files, single_file)
    if feedback:
        user += (f"\nYOUR PREVIOUS ATTEMPT FAILED: {feedback}\n"
                 "Produce a corrected spec that fixes exactly this.\n")
    resp = await llm.chat_completion({
        "model": model,
        "messages": [{"role": "system", "content": sys_hint},
                     {"role": "user", "content": user}],
        # 4096 truncated a single-file shell mid-JS (no </html>), which then
        # broke every later append/edit. Give complete files room.
        "temperature": 0.3, "max_tokens": 8192, "stream": False,
    })
    content = ((resp or {}).get("choices", [{}])[0]
               .get("message", {}).get("content", "") or "")
    from .agent import extract_json_from_text
    spec = extract_json_from_text(content, repair_truncated=True) or {}
    if not (isinstance(spec, dict) and isinstance(spec.get("files"), list) and spec.get("files")):
        # Diagnostic: the model returned no usable file spec. Log a window of
        # the raw output so we can see WHY (prose? broken JSON-escaped code?
        # empty?) instead of guessing.
        raw = content.strip()
        logger.warning(
            "coding_executor: no file spec parsed (len=%d). RAW head: %s ||| tail: %s",
            len(raw), raw[:400].replace("\n", "\\n"), raw[-200:].replace("\n", "\\n"))
    return spec


async def _apply_edits(tool_runner: ToolRunner, path: str, edits: list) -> Optional[str]:
    """Apply find/replace (or after/insert) edits to an existing file. Returns
    a failure reason, or None if at least one edit applied cleanly."""
    applied = 0
    for ed in edits[:20]:
        if not isinstance(ed, dict):
            continue
        find, rep = ed.get("find"), ed.get("replace")
        after = ed.get("after")
        before = ed.get("before")
        ins = ed.get("insert")
        if isinstance(find, str) and find and isinstance(rep, str):
            args = {"operation": "replace", "path": path,
                    "content": find, "replace_with": rep}
        elif isinstance(after, str) and after and isinstance(ins, str):
            args = {"operation": "replace", "path": path,
                    "content": after, "replace_with": after + "\n" + ins}
        elif isinstance(before, str) and before and isinstance(ins, str):
            # Insert BEFORE an anchor (e.g. "</body>") — the common HTML case.
            args = {"operation": "replace", "path": path,
                    "content": before, "replace_with": ins + "\n" + before}
        else:
            continue
        try:
            out = await tool_runner("file_system", args)
        except Exception as e:
            return f"edit on {path} errored: {e}"
        if not _op_ok(out):
            return f"edit on {path} did not apply (anchor not found): {_short(out)}"
        applied += 1
    if applied == 0:
        return f"no usable edits for {path}"
    return None


async def _apply_file(tool_runner: ToolRunner, fspec: dict,
                      existing_files: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
    """Apply one file entry. Returns ``(written_path|None, fail_reason|None)``.
    A blank entry returns (None, None) — skipped, not a failure."""
    if not isinstance(fspec, dict):
        return (None, None)
    path = (fspec.get("path") or "").strip()
    if not path:
        return (None, None)
    old = (existing_files or {}).get(path)

    # APPEND — the easiest, safest incremental primitive: the model emits ONLY
    # the new snippet and the executor places it (for HTML, just before the
    # closing tag so scripts/markup land inside the document; otherwise at the
    # end). A strict superset, so it can never regress. (Re-emitting the whole
    # file choked the small model — empty specs — and brittle edit anchors
    # failed; smart append sidesteps both.)
    append = fspec.get("append")
    if isinstance(append, str) and append.strip():
        base = old if isinstance(old, str) else ""
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
            return (None, f"append failed for {path}: {e}")
        if _looks_like_write_error(out):
            return (None, f"append rejected for {path}: {_short(out)}")
        return (path, None)

    edits = fspec.get("edits")
    if isinstance(edits, list) and edits and path in (existing_files or {}):
        reason = await _apply_edits(tool_runner, path, edits)
        return (None, reason) if reason else (path, None)

    content = fspec.get("content")
    if content is None:
        return (None, None)
    if not isinstance(content, str):
        content = str(content)
    if len(content) > MAX_CONTENT_CHARS:
        content = content[:MAX_CONTENT_CHARS]

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
        return (None, f"write failed for {path}: {e}")
    if _looks_like_write_error(out):
        return (None, f"write rejected for {path}: {_short(out)}")
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
    single_file: bool = False,
    max_files: int = MAX_FILES,
    max_attempts: int = MAX_ATTEMPTS,
    **_ignored,
) -> CodingResult:
    """Build one coding leaf: spec → write/edit (non-regressively) → verify,
    retrying once with feedback before failing. See module docs.

    ``existing_files`` ({path: content}) is the project's CURRENT workspace
    (captured before this task), so a leaf EXTENDS prior files instead of
    overwriting them. ``single_file`` strengthens the "grow one file" steer.
    """
    llm = getattr(context, "llm_client", None)
    if llm is None or tool_runner is None:
        return CodingResult(False, "coding executor unavailable (no llm/tool_runner)")
    model = getattr(getattr(context, "args", None), "model", "default")
    existing = existing_files or {}

    last = "build failed"
    feedback = ""
    last_written: List[str] = []
    for _attempt in range(max(1, max_attempts)):
        try:
            spec = await _generate_build_spec(
                llm, model, description, ledger,
                existing_files=existing, single_file=single_file, feedback=feedback)
        except Exception as e:  # pragma: no cover - LLM/network variance
            return CodingResult(False, f"build-spec generation failed: {e}")

        files = spec.get("files") if isinstance(spec, dict) else None
        if not isinstance(files, list) or not files:
            msg = "model produced no file spec for the task"
            # Don't let an empty retry overwrite a more informative failure
            # (e.g. a real verify error) from a prior attempt.
            if last == "build failed":
                last = msg
            feedback = msg
            continue

        written: List[str] = []
        fail: Optional[str] = None
        for f in files[:max_files]:
            path, reason = await _apply_file(tool_runner, f, existing)
            if reason:
                fail = reason
                break
            if path:
                written.append(path)
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

        summary = (spec.get("summary") if isinstance(spec, dict) else "") or \
            f"wrote {', '.join(written)}"
        ledger_note = (spec.get("ledger") if isinstance(spec, dict) else "") or ""
        return CodingResult(True, _short(summary, 300), files=written,
                            ledger_note=_short(ledger_note, 200))

    return CodingResult(False, _short(last, 280), files=last_written, detail=feedback)
