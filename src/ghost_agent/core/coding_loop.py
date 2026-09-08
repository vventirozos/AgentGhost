"""Agentic coding-leaf executor: a bounded edit-test loop (§4FG, 2026-09-07).

Why. An autonomous coding leaf (`coding_executor.build_coding_task`) is one
spec call that returns WHOLE FILES as JSON, then N writes and one verify, at
most a few times. §4EI showed what that costs on the live model: files
drafted inside `<think>` until the reasoning ceiling fires, structural slips
at the `files` boundary, and a grammar-constrained no-think retry — the exact
quadrant arXiv 2604.03616 measures as worst (JSON-required output costs 3–9
pp on Qwen3; a grammar does not remove it). Three of six leaves failed in a
49-minute run. The literature's shape for small open models is the opposite:
the model EDITS files with a tool and RUNS the tests, iterating on the
witness (Claw-SWE-Bench arXiv 2606.12344: apply failures 69.1% → <1.5% when
the model edits and the runner exports the patch; Cursor: model-specific edit
shapes gave an order of magnitude fewer tool errors).

What. The leaf runs as a bounded turn loop on an ISOLATED context that keeps
the project pinned (file_system paths resolve inside the project workspace)
but writes no memory, teaches nothing, and carries `task_kind="leaf"` so no
user-population ledger admits it. Tools: `file_system` and `execute` only.
The model works with thinking ON, then ends with two lines the executor
parses — `VERIFY: <shell command that exits 0 iff the work is correct>` and
`SUMMARY: <one line>`. Files touched are found by DIFFING the workspace
before and after (never by parsing the model's claims). The existing gates
then run unchanged: the verify command, `smoke_gate`, `constraint_gate`. On
failure a SECOND attempt starts FRESH with the failure's witness (the verify
output head) — arXiv 2607.26117 / 2609.02892: repair conditioned on the failed
attempt loses to a fresh attempt unless it carries a concrete witness.

Selection (`executor_kind`): `GHOST_CODING_EXECUTOR=agentic` or the project's
`metadata["executor"] == "agentic"`; default stays "spec" until the leaf bench
(`scripts/leaf_bench.py`) says otherwise. The seam is the top of
`build_coding_task`, so every caller inherits it.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from ..utils.logging import Icons, pretty_log

logger = logging.getLogger("GhostAgent")

ToolRunner = Callable[[str, Dict[str, Any]], Awaitable[str]]

from ..utils.helpers import env_positive

LEAF_ALLOWED_TOOLS = ("file_system", "execute")
LEAF_MAX_TURNS = int(env_positive("GHOST_LEAF_MAX_TURNS", 14.0))
LEAF_TIMEOUT_S = env_positive("GHOST_LEAF_TIMEOUT_S", 600.0)
LEAF_MAX_ATTEMPTS = 2
LEAF_TASK_KIND = "leaf"

_VERIFY_LINE_RE = re.compile(r"^\s*VERIFY:\s*(.+?)\s*$", re.I | re.M)
_SUMMARY_LINE_RE = re.compile(r"^\s*SUMMARY:\s*(.+?)\s*$", re.I | re.M)


# §4FL routing by shape (2026-09-08). The deciding bench (§4FI/§4FK, 24
# pairs) separated the two executors by ONE property: whether the leaf grows
# an existing file. On six small fresh leaves the arms were at parity three
# times; on a single app.py grown over six leaves the spec executor — which
# must re-emit the whole file — finished 15/24 against the loop's 23/24
# (p = 0.0078), and its failures began once the file existed at ~100 lines
# (crud 3/4, page 2/4, csv 1/4). So the default is a rule, not a flip: a
# leaf that EXTENDS an existing file of GHOST_LEAF_GROW_LINES (80) lines or
# more runs the loop; a fresh or small leaf keeps the cheaper spec path
# (mean 102 s vs 246 s per leaf). `metadata.executor` and the env override
# both ways; GHOST_LEAF_ROUTE_BY_SHAPE=0 turns the rule off.
GROW_LINES_DEFAULT = 80


def _grow_lines() -> int:
    return env_positive("GHOST_LEAF_GROW_LINES", GROW_LINES_DEFAULT)


def leaf_shape(description: str, existing_files: Optional[Dict[str, str]]) -> Tuple[str, str, int]:
    """("growing", path, lines) when the leaf names an existing file (by path
    or basename, as a whole word) that already holds >= GHOST_LEAF_GROW_LINES
    lines; else ("fresh", "", 0). The largest named file decides."""
    text = str(description or "")
    if not text or not existing_files:
        return ("fresh", "", 0)
    best, best_lines = "", 0
    for path, content in existing_files.items():
        p = str(path or "").strip()
        if not p:
            continue
        names = {p, os.path.basename(p)}
        if not any(re.search(r"(?<![\w/.-])" + re.escape(n) + r"(?![\w/-])", text) for n in names if n):
            continue
        lines = len(str(content or "").splitlines())
        if lines > best_lines:
            best, best_lines = p, lines
    if best and best_lines >= _grow_lines():
        return ("growing", best, best_lines)
    return ("fresh", "", 0)


def executor_kind(context, project_id: Optional[str] = None, *,
                  description: str = "",
                  existing_files: Optional[Dict[str, str]] = None) -> str:
    """"agentic" | "spec". Env wins; else the project's metadata (either
    value); else the SHAPE rule (a leaf growing an existing file of
    GHOST_LEAF_GROW_LINES+ lines runs the loop); else spec. ``project_id``
    is the leaf's project (passed by the advancer); the context's binding is
    only a fallback — an autoadvance tick carries none."""
    env = os.getenv("GHOST_CODING_EXECUTOR", "").strip().lower()
    if env in ("agentic", "spec"):
        return env
    try:
        pid = project_id or getattr(context, "current_project_id", None)
        store = getattr(context, "project_store", None)
        if pid and store is not None:
            meta = (store.get_project(pid) or {}).get("metadata") or {}
            want = str(meta.get("executor") or "").strip().lower()
            if want in ("agentic", "spec"):
                return want
    except Exception:  # noqa: BLE001
        pass
    if os.getenv("GHOST_LEAF_ROUTE_BY_SHAPE", "1").strip().lower() not in ("0", "false", "off", "no"):
        shape, path, lines = leaf_shape(description, existing_files)
        if shape == "growing":
            logger.warning("coding_loop: routing leaf to the agentic loop — it extends %s "
                        "(%d lines >= GHOST_LEAF_GROW_LINES=%d)", path, lines, _grow_lines())
            return "agentic"
    return "spec"


def _workspace_dir(context, project_id: Optional[str] = None) -> Optional[Path]:
    try:
        pid = project_id or getattr(context, "current_project_id", None)
        store = getattr(context, "project_store", None)
        if pid and store is not None:
            ws = (store.get_project(pid) or {}).get("workspace_dir")
            if ws:
                return Path(str(ws))
    except Exception:  # noqa: BLE001
        pass
    return None


_SNAPSHOT_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".pytest_cache", ".venv"}


def snapshot_workspace(root: Optional[Path]) -> Dict[str, str]:
    """{relative path: sha1 of content} for every regular file under root."""
    out: Dict[str, str] = {}
    if root is None or not root.exists():
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SNAPSHOT_SKIP_DIRS and not d.startswith(".")]
        for fn in filenames:
            if fn.startswith("."):
                continue
            p = Path(dirpath) / fn
            try:
                if p.stat().st_size > 5_000_000:
                    continue
                h = hashlib.sha1(p.read_bytes()).hexdigest()[:16]
            except OSError:
                continue
            out[str(p.relative_to(root)).replace(os.sep, "/")] = h
    return out


def diff_snapshots(before: Dict[str, str], after: Dict[str, str]) -> List[str]:
    """Paths created or changed (deleted paths are not 'written')."""
    return sorted(p for p, h in after.items() if before.get(p) != h)


def parse_leaf_reply(text: str) -> Tuple[str, str]:
    """(verify_command, summary) from the final reply; '' when absent.
    `VERIFY: none` → ''. Only the LAST occurrence counts (the model may
    quote the contract while thinking aloud)."""
    verify = ""
    m = _VERIFY_LINE_RE.findall(text or "")
    if m:
        v = m[-1].strip().strip("`")
        verify = "" if v.lower() in ("none", "n/a", "-", "") else v
    summary = ""
    s = _SUMMARY_LINE_RE.findall(text or "")
    if s:
        summary = s[-1].strip()
    return verify, summary


def _leaf_prompt(description: str, ledger: str, existing_files: Optional[Dict[str, str]],
                 research_context: Optional[Dict[str, str]], single_file: bool,
                 witness: str) -> str:
    files = sorted((existing_files or {}).keys())
    files_line = (", ".join(files[:40]) + (f" (+{len(files) - 40} more)" if len(files) > 40 else "")) or "(empty)"
    research = ""
    if research_context:
        parts = []
        for k, v in list(research_context.items())[:4]:
            parts.append(f"- {k}: {str(v)[:600]}")
        research = "RESEARCH BRIEFS:\n" + "\n".join(parts) + "\n\n"
    single = ("This is a SINGLE-FILE app: extend the existing file with targeted "
              "`file_system operation=replace` edits — never rewrite it wholesale.\n\n"
              if single_file else "")
    wit = (f"A PREVIOUS ATTEMPT FAILED VERIFICATION. Start from the workspace as it is now. "
           f"The failing evidence was:\n{witness[:1500]}\n\n") if witness else ""
    return (
        f"BUILD TASK (one leaf of a project): {description}\n\n"
        f"PROJECT LEDGER (what was built before):\n{(ledger or '(none)')[:2500]}\n\n"
        f"EXISTING FILES: {files_line}\n\n"
        f"{research}{single}{wit}"
        "HOW TO WORK: you have `file_system` (read / write / replace / list_files / search) and "
        "`execute` (a shell in the project directory). Read what exists before editing. Make the "
        "change with targeted edits, then RUN it — run the tests, import the module, start the "
        "script — and fix what the output shows. Prefer small verified steps over one big write. "
        "Do not touch files the task does not need. Do not ask the user anything; nobody is present.\n\n"
        "WHEN DONE, end your final reply with exactly these two lines:\n"
        "VERIFY: <one shell command that exits 0 iff this task's work is correct, e.g. "
        "`python -m pytest -q tests/test_x.py`; or `VERIFY: none` if nothing is runnable>\n"
        "SUMMARY: <one line: what you built and how you checked it>"
    )


def build_leaf_context(context, *, leaf_id: str, project_id: Optional[str] = None):
    """An isolated context for one leaf: project PINNED (explicitly — the
    advancer's context carries no binding), workspace shared, memory
    read-only, nothing recorded into user-population ledgers."""
    from ..memory.readonly import (ReadOnlyGraphMemory, ReadOnlySkillMemory,
                                   ReadOnlyVectorMemory)
    iso = copy.copy(context)
    # Keep sandbox_dir; PIN the project id: file_system must resolve the
    # PROJECT workspace, which is the whole point of the leaf. Copying the
    # context's binding is not enough — idle ticks and the HTTP route run
    # with `current_project_id=None` (the 2026-07-08 root-vs-project class).
    pid = project_id or getattr(context, "current_project_id", None)
    if pid:
        iso.current_project_id = pid
        # §4FH C1: `handle_chat` runs the conversation reconciler first thing,
        # which parks any project without a binding for the turn's
        # fingerprint — it parked every leaf's pin and the files went to the
        # sandbox root. The reconciler honours this marker and leaves a
        # pinned leaf alone; `run_leaf_turn` checks the pin AFTER the turn.
        iso._leaf_pinned_project = pid
    iso.workspace_model = None
    iso.episodic_memory = None
    iso.journal = None
    iso.turn_origin_label = LEAF_TASK_KIND
    iso.trajectory_task_kind = LEAF_TASK_KIND
    iso.trajectory_extra_static = {"leaf_id": leaf_id}
    iso.trajectory_user_request_override = None
    iso.scheduler = None
    iso.profile_memory = None
    iso.memory_bus = None
    try:
        iso.memory_system = ReadOnlyVectorMemory(context.memory_system)
        iso.skill_memory = ReadOnlySkillMemory(context.skill_memory)
        iso.graph_memory = ReadOnlyGraphMemory(getattr(context, "graph_memory", None))
    except Exception as e:  # noqa: BLE001
        logger.debug("leaf read-only memory wrap skipped: %s", e)
    iso.args = copy.copy(context.args)
    iso.args.perfect_it = False
    iso.args.smart_memory = 0.0
    if not getattr(iso.args, "native_tools", False):
        iso.args.native_tools = True
    for attr in ("verifier", "uncertainty_tracker", "mcts_reasoner", "hypothesis_tester",
                 "frontier_tracker", "metacog", "postmortem_engine", "reflector",
                 "prm_scorer", "complexity_dispatcher", "calibration_tracker"):
        try:
            setattr(iso, attr, None)
        except Exception:  # noqa: BLE001
            pass
    iso._subagent_allowed_tools = frozenset(LEAF_ALLOWED_TOOLS)
    return iso


class _BackgroundOnlyLLM:
    """Route every completion through the client's background lane (waits
    for foreground to clear, capped concurrency) — the same wrapper
    `build_subagent_context` uses. §4FH M1: the first leaf runner took
    `is_background` and never used it, so an idle-tick leaf competed with a
    live user turn for the single slot."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def chat_completion(self, payload, *a, **kw):
        kw["is_background"] = True
        return await self._inner.chat_completion(payload, *a, **kw)

    async def stream_chat_completion(self, payload, *a, **kw):
        kw["is_background"] = True
        async for chunk in self._inner.stream_chat_completion(payload, *a, **kw):
            yield chunk


async def run_leaf_turn(context, *, leaf_id: str, prompt: str, is_background: bool,
                        max_turns: int = LEAF_MAX_TURNS,
                        timeout_s: float = LEAF_TIMEOUT_S,
                        project_id: Optional[str] = None) -> str:
    """One bounded agentic run; returns the final reply text. Tool surface
    is contained the same way `run_subagent` does it (fail CLOSED)."""
    from .agent import GhostAgent
    iso = build_leaf_context(context, leaf_id=leaf_id, project_id=project_id)
    if is_background and getattr(iso, "llm_client", None) is not None:
        iso.llm_client = _BackgroundOnlyLLM(iso.llm_client)
    agent = GhostAgent(iso)
    allow = set(LEAF_ALLOWED_TOOLS)
    try:
        from ..tools.registry import TOOL_DEFINITIONS
        advertised = {t["function"]["name"] for t in TOOL_DEFINITIONS}
        agent.disabled_tools = (advertised | set(agent.available_tools)) - allow
        agent.available_tools = {k: v for k, v in agent.available_tools.items() if k in allow}
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"leaf tool containment failed ({type(e).__name__}: {e})") from e
    agent.max_turns_override = max(2, int(max_turns))
    body = {"model": getattr(iso.args, "model", "default"),
            "messages": [{"role": "user", "content": prompt}], "stream": False}
    content, _, _ = await asyncio.wait_for(
        agent.handle_chat(body, background_tasks=None, request_id=f"sub-leaf-{leaf_id}"),
        timeout=max(30.0, float(timeout_s)))
    # §4FH C1: the pin must have survived the turn — if anything parked it,
    # the leaf's files went somewhere other than the project and the
    # attempt is a failure, not a success with an empty diff.
    _want = project_id or getattr(context, "current_project_id", None)
    if _want and getattr(iso, "current_project_id", None) != _want:
        raise RuntimeError(
            f"leaf lost its project pin during the turn "
            f"(wanted {_want}, got {getattr(iso, 'current_project_id', None)!r})")
    return str(content or "").strip()


async def build_coding_task_agentic(context, description: str, *, tool_runner: ToolRunner,
                                    ledger: str = "", existing_files=None,
                                    research_context=None, single_file: bool = False,
                                    max_attempts: int = LEAF_MAX_ATTEMPTS,
                                    is_background: bool = False, constraints=None,
                                    project_id: Optional[str] = None,
                                    **_ignored):
    """Same contract as `coding_executor.build_coding_task` → `CodingResult`.
    ``project_id`` pins the leaf's project for the workspace diff and the
    isolated context (the advancer passes it; the context may carry none)."""
    from .coding_executor import CodingResult, _run_verify, _short
    from .build_gates import constraint_gate, smoke_gate
    ws = _workspace_dir(context, project_id)
    witness = ""
    detail_parts: List[str] = []
    leaf_base = hashlib.sha1(f"{time.time()}|{description[:80]}".encode()).hexdigest()[:8]
    for attempt in range(1, max(1, int(max_attempts)) + 1):
        leaf_id = f"{leaf_base}-a{attempt}"
        before = snapshot_workspace(ws)
        prompt = _leaf_prompt(description, ledger, existing_files, research_context,
                              single_file, witness)
        pretty_log("Leaf Loop", f"attempt {attempt}/{max_attempts} · {_short(description, 70)}",
                   icon=Icons.BRAIN_PLAN)
        try:
            reply = await run_leaf_turn(context, leaf_id=leaf_id, prompt=prompt,
                                        is_background=is_background,
                                        project_id=project_id)
        except asyncio.TimeoutError:
            witness = f"attempt {attempt} timed out after {LEAF_TIMEOUT_S:.0f}s"
            detail_parts.append(witness)
            continue
        except Exception as e:  # noqa: BLE001
            witness = f"attempt {attempt} crashed: {type(e).__name__}: {e}"
            detail_parts.append(witness)
            continue
        after = snapshot_workspace(ws)
        written = diff_snapshots(before, after)
        verify, summary = parse_leaf_reply(reply)
        if not written:
            # §4FH C1: a leaf that changed nothing INSIDE the project workspace
            # cannot be DONE — the six theatrical DONEs had exactly this shape
            # (files at the sandbox root, a verify that passed through
            # execute's retry-from-root heal). A verify-only leaf is the spec
            # executor's business, not this loop's.
            witness = ("the attempt changed no files inside the project workspace"
                       + ("" if verify else " and gave no VERIFY command"))
            detail_parts.append(witness)
            continue
        reason = await _run_verify(tool_runner, {"verify": verify}, written)
        if reason is None:
            reason = await smoke_gate(tool_runner, written)
        if reason is None and constraints and written and ws is not None:
            files: Dict[str, str] = {}
            for rel in written[:8]:
                try:
                    files[rel] = (ws / rel).read_text(errors="replace")[:4000]
                except OSError:
                    continue
            ok, why = await constraint_gate(context, list(constraints), files,
                                            is_background=is_background)
            if not ok:
                reason = f"constraint gate: {why}"
        if reason is None:
            note = (f"leaf(agentic): {summary or _short(description, 120)} · "
                    f"files={len(written)} · verify={'yes' if verify else 'none'}")
            return CodingResult(True, summary or _short(description, 160), written, note,
                                detail="; ".join(detail_parts))
        witness = reason
        detail_parts.append(f"attempt {attempt}: {reason}")
        pretty_log("Leaf Loop", f"attempt {attempt} failed — {_short(reason, 120)}",
                   level="WARNING", icon=Icons.WARN)
    return CodingResult(False, f"agentic leaf failed after {max_attempts} attempt(s): "
                        f"{_short(witness, 200)}", [], "", detail="; ".join(detail_parts))
