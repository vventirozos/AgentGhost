"""Real per-task coding executor for autonomous batches.

``advance_once``'s default coding path generates ONE shell command — too
weak for genuine build tasks, which produced *theatrical completion*: a
task marked DONE having written nothing (observed live — "create hello.txt"
ran a web_search and was marked done, no file written).

``build_coding_task`` builds a leaf FOR REAL and bounded:

  1. ask the model for a file SPEC — one or more files with FULL contents,
     plus an optional ``verify`` shell command and a one-line ledger fact;
  2. write each file via the ``file_system`` tool;
  3. run ``verify`` via ``execute``;
  4. report success with the files produced — or FAIL loudly (which stops
     the batch loop for the user) instead of marking shallow work done.

It is deliberately NOT an open-ended agent loop: the project's task tree
already decomposes the work, so this just builds ONE leaf well — a single
spec call, N writes, one verify. That keeps per-task context bounded, the
whole point of advancing a big project as a loop of ticks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger("GhostAgent")

ToolRunner = Callable[[str, Dict[str, Any]], Awaitable[str]]

MAX_FILES = 8
MAX_CONTENT_CHARS = 60_000  # per-file guard against a runaway generation


@dataclass
class CodingResult:
    ok: bool
    summary: str
    files: List[str] = field(default_factory=list)
    ledger_note: str = ""
    detail: str = ""


def _short(text: str, n: int = 180) -> str:
    return " ".join((text or "").split())[:n]


def _looks_like_write_error(out: str) -> bool:
    """True when a file_system write clearly failed. Conservative: writes
    rarely fail (the path heal absorbs most mistakes), so only an explicit
    error prefix counts — we don't want to abort a good build on a chatty
    success message."""
    head = (out or "").strip()[:80].lower()
    return (
        not out
        or head.startswith("system error")
        or head.startswith("error:")
        or "security error" in head
    )


async def _generate_build_spec(llm, model: str, description: str, ledger: str) -> dict:
    """Ask the model for a JSON build spec. Returns ``{}`` on any failure."""
    sys_hint = (
        "You are building ONE task inside a larger project. Output ONLY a "
        "JSON object — no prose, no markdown fences — with this exact shape:\n"
        '{"files":[{"path":"relative/name.ext","content":"FULL file contents"}],'
        '"verify":"a shell command that exits 0 iff the task works, or \\"\\"",'
        '"summary":"one line: what you built",'
        '"ledger":"one durable fact for the project (a file, an API/function '
        'name, a convention), or \\"\\""}\n'
        "Rules: write COMPLETE files (never snippets or TODOs); paths are "
        "project-relative BARE names (no /workspace, no sandbox/, no "
        "projects/<id>/); only the files THIS task needs; prefer a real, "
        "runnable verify (python3 -c '...', node -e '...', or a test command) "
        "that actually exercises what you built."
    )
    user = f"TASK: {description}\n"
    if ledger:
        user += ("\nPROJECT LEDGER (existing files / APIs / conventions — build "
                 f"CONSISTENTLY with these, reuse them):\n{ledger}\n")
    resp = await llm.chat_completion({
        "model": model,
        "messages": [{"role": "system", "content": sys_hint},
                     {"role": "user", "content": user}],
        "temperature": 0.3, "max_tokens": 4096, "stream": False,
    })
    content = ((resp or {}).get("choices", [{}])[0]
               .get("message", {}).get("content", "") or "")
    # repair_truncated salvages a spec cut at the token cap — a partial build
    # that then fails `verify` is still better than no attempt, and verify is
    # the real correctness gate here (this is NOT a raw tool-call arg parse).
    from .agent import extract_json_from_text
    return extract_json_from_text(content, repair_truncated=True) or {}


async def build_coding_task(
    context,
    description: str,
    *,
    tool_runner: Optional[ToolRunner],
    ledger: str = "",
    max_files: int = MAX_FILES,
) -> CodingResult:
    """Build one coding leaf: spec → write files → verify. See module docs."""
    llm = getattr(context, "llm_client", None)
    if llm is None or tool_runner is None:
        return CodingResult(False, "coding executor unavailable (no llm/tool_runner)")
    model = getattr(getattr(context, "args", None), "model", "default")

    try:
        spec = await _generate_build_spec(llm, model, description, ledger)
    except Exception as e:  # pragma: no cover - LLM/network variance
        return CodingResult(False, f"build-spec generation failed: {e}")

    files = spec.get("files") if isinstance(spec, dict) else None
    if not isinstance(files, list) or not files:
        return CodingResult(False, "model produced no file spec for the task")

    written: List[str] = []
    for f in files[:max_files]:
        if not isinstance(f, dict):
            continue
        path = (f.get("path") or "").strip()
        content = f.get("content")
        if not path or content is None:
            continue
        if not isinstance(content, str):
            content = str(content)
        if len(content) > MAX_CONTENT_CHARS:
            content = content[:MAX_CONTENT_CHARS]
        try:
            out = await tool_runner(
                "file_system", {"operation": "write", "path": path, "content": content})
        except Exception as e:
            return CodingResult(False, f"write failed for {path}: {e}", files=written)
        if _looks_like_write_error(out):
            return CodingResult(False, f"write rejected for {path}: {_short(out)}",
                                files=written)
        written.append(path)

    if not written:
        return CodingResult(False, "build spec produced no writable files")

    verify = (spec.get("verify") or "").strip() if isinstance(spec, dict) else ""
    if verify:
        try:
            vout = await tool_runner("execute", {"command": verify})
        except Exception as e:
            return CodingResult(False, f"verify errored: {e}", files=written)
        from .project_advancer import _looks_like_failure
        if _looks_like_failure(vout):
            return CodingResult(
                False, f"verify failed for {', '.join(written)}: {_short(vout)}",
                files=written, detail=(vout or "")[:400])

    summary = (spec.get("summary") if isinstance(spec, dict) else "") or \
        f"wrote {', '.join(written)}"
    ledger_note = (spec.get("ledger") if isinstance(spec, dict) else "") or ""
    return CodingResult(True, _short(summary, 300), files=written,
                        ledger_note=_short(ledger_note, 200))
