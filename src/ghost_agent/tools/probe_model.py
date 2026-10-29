"""``probe_model`` — run prompts on a model for real and keep the raw answers
(§4QA, 2026-10-18).

Project 448d5afa86f0 tested "3-layer recursive thinking → self-awareness"
with no way to call a model: every "Layer N response" was written by the
agent into a results file in one go (with invented 2024 timestamps), the
"cross-model validation" script carried hard-coded ``simulate_model_response``
texts labelled temperature 0.3/0.9, and the final report said the hypothesis
was "robustly confirmed". The model said it outright: "Since I can't spin up
separate LLM instances, I'll simulate…".

This tool is the missing instrument. Each prompt goes to a model in a FRESH
context (no conversation, no tools); ``chain=true`` feeds every prompt the
previous answers (a recursive cascade: answer → "reflect on that" → …). The
SYSTEM writes the transcript — prompt, verbatim response, model, node,
temperature, time — so a results file can cite answers that were really
produced, and the reader can tell them from prose.

Main-slot etiquette: inside the owner's turn the call is FOREGROUND (a
background call parks behind its own request — the §4QA self-stall). Fresh
messages displace the main slot's cached conversation, so the turn's next
step re-reads it (one re-prefill per probe call, ~30 s on a long chat);
``node="worker"`` runs on the worker node (a different model — usable as a
second subject) and leaves the main slot alone.
"""

from __future__ import annotations

import logging
import os
import time
import re
from pathlib import Path, PurePosixPath
from typing import Any, List, Optional

from .outcome import ToolOutcome

logger = logging.getLogger("GhostAgent")

MAX_PROMPTS = 8
MAX_PROMPT_CHARS = 4000
#: §4QA-e: 600 cut every answer of the 448d5afa86f0 runs (11 of 11 runs, the
#: model's own answers run ~530–810 tokens), and a chain fed the cut answers
#: on; at ~87 tok/s 8 × 3000 still fits TOTAL_TIMEOUT_S
DEFAULT_MAX_TOKENS = 1500
MAX_TOKENS_CAP = 3000
_RETURN_CHARS = 12000
#: the whole call's wall clock: 8 prompts × 180 s had no bound (fresh-eye)
TOTAL_TIMEOUT_S = 600
_PER_CALL_TIMEOUT_S = 180

PROBE_MODEL_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "probe_model",
        "description": (
            "Run prompts on a language model FOR REAL and get its verbatim answers — the instrument "
            "for any experiment, test, control condition or comparison whose data are model outputs. "
            "Never write a model's 'response' yourself in a results file: run it here and cite the "
            "saved transcript. Each prompt runs in a FRESH context (no conversation, no tools). "
            "chain=true sends every prompt with the previous prompts and answers (a recursive "
            "cascade). node='main' is the model you run on; node='worker' is a different model on "
            "the worker node (a second subject). save_to (a new .md path in the project) makes the "
            "SYSTEM write the transcript, unedited. A results file that cites the transcript must quote "
            "it word for word — at task close a quote that seems reworded is reported to you and "
            "to the owner."),
        "parameters": {
            "type": "object",
            "properties": {
                "prompts": {"type": "array", "items": {"type": "string"},
                            "description": f"1–{MAX_PROMPTS} prompts, sent in order."},
                "chain": {"type": "boolean",
                          "description": "true: each prompt sees the earlier prompts and answers. Default false (independent)."},
                "temperature": {"type": "number", "description": "0–1.5, default 0.7."},
                "max_tokens": {"type": "integer", "description": f"Per answer, 64–{MAX_TOKENS_CAP}, default {DEFAULT_MAX_TOKENS}."},
                "node": {"type": "string", "enum": ["main", "worker"], "description": "Default 'main'."},
                "system": {"type": "string", "description": "Optional system prompt for the subject."},
                "save_to": {"type": "string",
                            "description": "Optional NEW project-relative .md path for the transcript (e.g. results/layer-runs/run1.md)."},
            },
            "required": ["prompts"],
        },
    },
}


def _clamp(v, lo, hi, default):
    try:
        return max(lo, min(hi, type(default)(v)))
    except (TypeError, ValueError):
        return default


def _answer(data):
    """(text, cut) — ``cut`` when the answer stopped at max_tokens."""
    try:
        ch = (data or {}).get("choices", [{}])[0] or {}
        msg = ch.get("message", {}) or {}
    except (AttributeError, IndexError):
        return "", False
    return str(msg.get("content") or "").strip(), ch.get("finish_reason") == "length"


def _fenced(text: str) -> str:
    """The response inside a fence longer than any backtick run in it, so a
    response holding "## 3. Prompt" or "### Response" lines cannot pose as
    sections of the transcript."""
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{text}\n{fence}"


def _cut_note(rows: List[dict], chain: bool, max_tokens: int) -> str:
    """§4QA-e: how many answers stopped at max_tokens — and, in a chain, that
    the later prompts were answered on top of a cut answer."""
    cut = [i for i, r in enumerate(rows, 1) if r.get("cut")]
    if not cut:
        return ""
    note = f"{len(cut)} of {len(rows)} answers were cut off at max_tokens={max_tokens} (answer(s) {', '.join(map(str, cut))})"
    if chain and cut[0] < len(rows):
        later = (f"prompt {len(rows)} was" if cut[0] + 1 == len(rows)
                 else f"prompts {cut[0] + 1}–{len(rows)} were")
        note += f"; in this chain, {later} answered after a cut answer"
    return note


def _cut_warning(meta: dict) -> str:
    if not meta.get("cut_note"):
        return ""
    return (f"WARNING: {meta['cut_note']} — if complete answers matter, run again with a higher "
            f"max_tokens (up to {MAX_TOKENS_CAP}) and a new save_to. ")


def _transcript(rows: List[dict], meta: dict) -> str:
    head = [
        "# Model probe transcript",
        "",
        "_Recorded by `probe_model`: the system wrote this file. Each response is the model's "
        "verbatim output, unedited._",
        "",
        f"- model: {meta['model']} · node: {meta['node']} · temperature: {meta['temperature']} · "
        f"chain: {str(meta['chain']).lower()} · thinking: off · recorded {meta['when']}",
    ]
    if meta.get("system"):
        # quoted like a prompt: a raw system text could forge rows (r9)
        head.append("- system prompt:\n\n" + "\n".join("> " + ln for ln in meta["system"].splitlines() or [""]))
    if meta.get("cut_note"):
        head.append(f"- ⚠ {meta['cut_note']}")
    out = ["\n".join(head)]
    for i, r in enumerate(rows, 1):
        quoted = "\n".join("> " + ln for ln in r["prompt"].splitlines() or [""])
        body = _fenced(r["response"]) if r["response"] else "_(empty)_"
        if r.get("cut"):
            body += f"\n\n_(cut off at max_tokens={meta['max_tokens']} — the answer is incomplete)_"
        who = f" ({r['model']})" if r.get("model") and r["model"] != meta["model"] else ""
        out.append(f"## {i}. Prompt\n\n{quoted}\n\n### Response{who}\n\n{body}")
    if meta.get("stopped"):
        out.append(f"_{meta['stopped']}_")
    return "\n\n".join(out) + "\n"


async def tool_probe_model(context, prompts: Any = None, chain: bool = False,
                           temperature: Any = 0.7, max_tokens: Any = DEFAULT_MAX_TOKENS,
                           node: str = "main", system: Optional[str] = None,
                           save_to: Optional[str] = None, **_kw) -> str:
    if isinstance(prompts, str):
        prompts = [prompts]
    if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) and p.strip() for p in prompts):
        return ToolOutcome.rejected("REJECTED: `prompts` must be a list of 1–8 non-empty strings. Nothing ran.",
                                    reason_code="bad_prompts")
    if len(prompts) > MAX_PROMPTS:
        return ToolOutcome.rejected(f"REJECTED: {len(prompts)} prompts — at most {MAX_PROMPTS} per call "
                                    f"(make several calls). Nothing ran.", reason_code="too_many_prompts")
    node = "worker" if str(node or "").lower() == "worker" else "main"
    save_rel = None
    if save_to:
        save_rel = str(save_to).strip().lstrip("/")
        if save_rel.startswith("workspace/"):
            save_rel = save_rel[len("workspace/"):]
        p = PurePosixPath(save_rel)
        if p.suffix.lower() != ".md" or ".." in p.parts or not p.name:
            return ToolOutcome.rejected("REJECTED: save_to must be a project-relative .md path "
                                        "(e.g. results/run1.md). Nothing ran.", reason_code="bad_save_to")
    target = None
    if save_rel:
        # existence on the path the writer RESOLVES ("/workspace/projects/<id>/
        # results/r.md" is results/r.md) — checked before any call (MAJOR-1)
        from .file_system import _get_safe_path, project_scoped_sandbox
        sandbox_dir, _ = project_scoped_sandbox(context)
        if sandbox_dir is None:
            return ToolOutcome.rejected("REJECTED: no workspace to save into — call without save_to, "
                                        "or open the project first. Nothing ran.", reason_code="no_workspace")
        try:
            target = _get_safe_path(sandbox_dir, save_rel)
            # inside THIS workspace only: "projects/<other>/r.md" resolved
            # into another project, a released one included (r9)
            Path(target).resolve().relative_to(Path(sandbox_dir).resolve())
        except (ValueError, OSError) as e:
            return ToolOutcome.rejected(f"REJECTED: save_to is not a usable path in this project's "
                                        f"workspace ({e}). Nothing ran.", reason_code="bad_save_to")
        # the workspace folder as it is NOW, before minutes of model calls: a
        # root swapped for a symlink meanwhile is refused at save (r12 MAJOR)
        try:
            _st_root = os.stat(str(sandbox_dir))
            root_id = (_st_root.st_dev, _st_root.st_ino)
        except OSError as e:
            return ToolOutcome.rejected(f"REJECTED: the workspace is not usable ({e}). Nothing ran.",
                                        reason_code="no_workspace")
        from .file_system import _released_write_block
        _rb = _released_write_block(getattr(context, "project_store", None), sandbox_dir, target)
        if _rb:
            return ToolOutcome.rejected(f"REJECTED: {_rb} Nothing ran.", reason_code="released")
        if os.path.lexists(target):
            return ToolOutcome.rejected(f"REJECTED: {save_rel} already exists — a transcript is never "
                                        "overwritten; pick a new save_to. Nothing ran.",
                                        reason_code="save_target_exists")
    llm = getattr(context, "llm_client", None)
    if llm is None:
        return ToolOutcome.failed("Error: no model client is available. Nothing ran.",
                                  world_changed=False, reason_code="no_llm")
    temperature = _clamp(temperature, 0.0, 1.5, 0.7)
    max_tokens = _clamp(max_tokens, 64, MAX_TOKENS_CAP, DEFAULT_MAX_TOKENS)
    model = getattr(getattr(context, "args", None), "model", "default")
    msgs: List[dict] = [{"role": "system", "content": str(system)[:2000]}] if system else []
    rows: List[dict] = []
    stopped = ""
    from ..core.project_research import _idle_invocation
    from ..core.llm import served_leg
    from ..utils.aio import wait_for
    _bg = _idle_invocation(context)          # an unattended caller parks behind the owner
    t0 = time.monotonic()
    for prompt in prompts:
        left = TOTAL_TIMEOUT_S - (time.monotonic() - t0)
        if left < 30:
            stopped = (f"Stopped after {len(rows)} of {len(prompts)} prompts: the call's "
                       f"{TOTAL_TIMEOUT_S} s limit was reached.")
            break
        prompt = prompt[:MAX_PROMPT_CHARS]
        convo = (msgs if chain else list(msgs[:1] if system else [])) + [{"role": "user", "content": prompt}]
        try:
            # the whole wait is bounded here, not only between prompts (r9):
            # slot waits and retries ran past the call's limit
            data = await wait_for(llm.chat_completion(
                {"model": model, "messages": convo, "temperature": temperature,
                 "max_tokens": max_tokens, "stream": False,
                 "chat_template_kwargs": {"enable_thinking": False}},
                # foreground inside the turn: a background call parks behind
                # its own request (§4QA); the worker leg never touches main
                use_worker=(node == "worker"), off_main_only=(node == "worker"),
                is_background=_bg, task_label="probe-model",
                # §4SC-b: a foreground probe on main does not open with the
                # head — a side call (the conversation is saved first)
                side_main=(not _bg and node == "main"),
                timeout=min(_PER_CALL_TIMEOUT_S, int(left)), total_budget=left), left)
            served = served_leg(data).get("served_by") or ""
            if node == "worker" and served and served != "worker":
                raise RuntimeError(f"the worker did not answer (served by {served}) — "
                                   "no second subject is available")
        except Exception as e:  # noqa: BLE001
            if not rows:
                return ToolOutcome.failed(
                    f"Error: the {node} model call failed on prompt 1 ({type(e).__name__}: {e}). "
                    "Nothing ran to the end; nothing was saved.",
                    world_changed=False, reason_code="probe_failed")
            # keep the answers already paid for (r9): save them, say where it stopped
            stopped = (f"Stopped after {len(rows)} of {len(prompts)} prompts: the {node} model call "
                       f"failed on prompt {len(rows) + 1} ({type(e).__name__}: {str(e)[:200]}).")
            break
        answer, cut = _answer(data)
        rows.append({"prompt": prompt, "response": answer, "cut": cut,
                     "model": str((data or {}).get("model") or "") if isinstance(data, dict) else ""})
        if chain:
            msgs = convo + [{"role": "assistant", "content": answer}]
    # the model that ANSWERED, not the one configured for main (r9 MAJOR:
    # a worker run was recorded under the main model's name)
    _served = [r["model"] for r in rows if r.get("model")]
    meta = {"model": (_served[0] if _served else model), "node": node, "temperature": temperature, "chain": bool(chain),
            "when": time.strftime("%Y-%m-%d %H:%M"), "system": (str(system)[:200] if system else ""),
            "max_tokens": max_tokens, "stopped": stopped, "cut_note": _cut_note(rows, chain, max_tokens)}
    text = _transcript(rows, meta)
    saved = ""
    if save_rel:
        # written EXCLUSIVELY: an exists-check then the writer's replace let
        # two probes finishing together overwrite each other (r9). A temp
        # file is hard-linked to the name — link() fails if it exists. Both
        # go through ONE directory handle reached from the workspace without
        # following a link: a folder swapped for a symlink during the minutes
        # of model calls cannot redirect it (r10 MAJOR-4).
        dfd, tmp_name, name = None, "", Path(target).name
        try:
            # walked from the workspace one folder at a time, never through a
            # link (O_NOFOLLOW per component, created in place): a path check
            # then an open is a race a swapped symlink wins (r11 m3)
            root = str(sandbox_dir)
            rel_parent = os.path.relpath(str(Path(target).parent), root)
            if rel_parent.startswith(".."):              # /var vs /private/var spellings
                root = str(Path(sandbox_dir).resolve())
                rel_parent = os.path.relpath(str(Path(target).parent), root)
            parts = [] if rel_parent == "." else rel_parent.split(os.sep)
            if any(p_ in ("", "..") for p_ in parts):
                raise ValueError("the target is outside this workspace")
            # released while the prompts ran (the check above is minutes old)
            _rb_now = _released_write_block(getattr(context, "project_store", None), sandbox_dir, target)
            if _rb_now:
                raise ValueError(_rb_now)
            dfd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            _st = os.fstat(dfd)
            if (_st.st_dev, _st.st_ino) != root_id:
                raise ValueError("the workspace folder changed while the probe ran")
            for part in parts:
                try:
                    os.mkdir(part, 0o755, dir_fd=dfd)
                except FileExistsError:
                    pass
                nfd = os.open(part, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                              dir_fd=dfd)
                os.close(dfd)
                dfd = nfd
            data_b = text.encode("utf-8", errors="replace")       # a lone surrogate (r10)
            tmp_name = f".pm.{os.getpid()}.{time.monotonic_ns()}.tmp"    # short: a long save_to still fits
            fd = os.open(tmp_name, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
                         0o644, dir_fd=dfd)
            with os.fdopen(fd, "wb") as fh:
                fh.write(data_b)
            os.link(tmp_name, name, src_dir_fd=dfd, dst_dir_fd=dfd, follow_symlinks=False)
        except FileExistsError:
            return ToolOutcome.partial(
                _cut_warning(meta) + f"The probe ran, but {save_rel} already exists and was NOT overwritten — call again "
                f"with a new save_to to keep this run. Transcript:\n\n" + text[:_RETURN_CHARS],
                world_changed=False, reason_code="save_target_exists")
        except (OSError, ValueError) as e:
            return ToolOutcome.partial(_cut_warning(meta) + f"{stopped + ' ' if stopped else ''}The probe ran, but saving failed: {e}"
                                       "\n\n" + text[:_RETURN_CHARS],
                                       world_changed=False, reason_code="save_failed")
        finally:
            if dfd is not None:
                if tmp_name:
                    try:
                        os.unlink(tmp_name, dir_fd=dfd)
                    except OSError:
                        pass
                os.close(dfd)
        saved = f"Saved the transcript to {save_rel} (written by the system — cite it as the data). "
        # §4QA-e: the bytes kept where the sandbox cannot reach — a results
        # file citing this transcript is checked against THEM at task close
        _sd = Path(str(sandbox_dir))
        from ..core.probe_transcripts import record
        if not (_sd.parent.name == "projects" and record(
                getattr(getattr(context, "project_store", None), "memory_dir", None), _sd.name, save_rel, text)):
            saved += ("(It could not be kept for the quote check — a write-up citing it is not "
                      "checked against it at task close.) ")
    try:
        logger.info("probe_model: %d prompt(s) on %s, chain=%s, T=%s%s", len(rows), node, chain,
                    temperature, f", saved {save_rel}" if saved else "")
    except Exception:  # noqa: BLE001
        pass
    body = text if len(text) <= _RETURN_CHARS else text[:_RETURN_CHARS] + (
        "\n…(transcript continues in the saved file)" if saved else
        f"\n…({len(text) - _RETURN_CHARS} more characters NOT shown and NOT saved — call again with "
        "save_to to keep every answer)")
    saved = _cut_warning(meta) + saved
    if stopped:
        return ToolOutcome.partial(f"{stopped} {saved}\n\n{body}", world_changed=bool(saved),
                                   reason_code=("probe_time_limit" if "limit was reached" in stopped
                                                else "probe_failed_partial"))
    return ToolOutcome.ok(f"SUCCESS: {len(rows)} prompt(s) answered by the {node} model. {saved}\n\n{body}",
                          world_changed=bool(saved))


__all__ = ["PROBE_MODEL_TOOL_DEFINITION", "tool_probe_model"]
