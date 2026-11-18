"""§4UT: live progress for the owner's streamed request — a step list like Claude Code's.

Operator 2026-11-16: "is there any way to show the progress in a request like claude code does?" →
"could we use nova for 2.?" → "go" (web and CLI only, English only, the steps must be hideable).

After each tool batch the agent sends the client two frames through the reply tap
(``ReplyTap.progress``), never as reply text:

1. **the step**, at once, with no model call: its tools and counts and a fixed summary ("Ran 2 commands,
   read 1 file"), so a client always has something to show;
2. **the sentence**, about a second later: one first-person line the worker node (Nova, the small side
   model) writes from the user's request, the step's own intent (the end of its thinking), the calls and
   the start of their results — "I read the portfolio app's styles and concluded no CSS changes are
   needed." It is attached to its step by ``seq``.

Measured on live steps (11-16, Gemma E4B): ~1 s a line; with the intent note the line says what a step
CONCLUDED, without it only what the command printed. English only: in Greek the small model leaked
Cyrillic and misspelled verbs (operator: "english only").

Rules: Nova only (``off_main_only`` — never the main model's slot); one call in flight per request, and
the steps that finish meanwhile collapse into the newest (a 37-call turn is ~10–15 lines, not 37); a line
that is not one clean English sentence, that promises the next step, or that comes after the answer has
started, is dropped. Owner's streamed requests only (the route makes a tap for nothing else: no member,
no piped CLI); never a probe (``X-Ghost-Origin: probe`` — unless ``GHOST_PROGRESS_PROBES=1``, for tests).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("GhostAgent")

ENV = "GHOST_PROGRESS"
#: the most a sentence may wait for Nova; a later one is not a progress line any more
NARRATE_TIMEOUT_S = 4.0
NARRATE_MAX_TOKENS = 60
LINE_MAX_CHARS = 200
REQUEST_CHARS = 300
INTENT_CHARS = 400
CALLS_SHOWN = 3
ARG_CHARS = 120
RESULT_CHARS = 200
#: kept in the turn record (``extra.progress``) so the lines can be judged later
KEPT_STEPS = 60

#: what is not one clean English line: another script (the small model leaked Cyrillic into Greek), more
#: than one sentence, a promise of what comes next (a progress line narrates what WAS done)
_FOREIGN_RE = re.compile(r"[Ͱ-ϿЀ-ӿ֐-ۿ぀-ヿ一-鿿가-힯]")
_PROMISE_RE = re.compile(r"\b(?:i['’]ll|i\s+will|i\s+am\s+going\s+to|i['’]m\s+going\s+to|next,?\s+i|now\s+i['’]ll|"
                         r"let\s+me|i\s+plan\s+to|then\s+i['’]ll)\b", re.I)
_SENTENCE_END_RE = re.compile(r"[.!?](?:\s+|$)")


def worker_is_off_main(llm) -> bool:
    """Is there a worker node that is not the main model's box? Without one, an ``off_main_only`` call does
    not raise: it waits for the foreground to clear and then runs on the main slot."""
    try:
        from .llm import _node_key
        nodes = getattr(llm, "worker_clients", None) or []
        main = _node_key(str(getattr(llm, "upstream_url", "") or ""))
        return any(isinstance(n, dict) and n.get("url") and (not main or _node_key(n["url"]) != main)
                   for n in nodes)
    except Exception:  # noqa: BLE001
        return False


def enabled() -> bool:
    return os.environ.get(ENV, "1").strip().lower() not in ("0", "false", "no", "off")


# ── the step: tools, counts, a fixed summary (no model) ──────────────────────

#: a tool (and, for file_system, its operation) → the verb phrase of the summary: (one, many)
_VERBS = {
    "execute": ("ran a command", "ran {n} commands"),
    "shell": ("ran a command", "ran {n} commands"),
    "file_system.read": ("read a file", "read {n} files"),
    "file_system.edit": ("edited a file", "edited {n} files"),
    "file_system.write": ("wrote a file", "wrote {n} files"),
    "file_system.append": ("wrote a file", "wrote {n} files"),
    "file_system.search": ("searched the files", "searched the files {n} times"),
    "file_system.list": ("listed a folder", "listed {n} folders"),
    "file_system.list_files": ("listed a folder", "listed {n} folders"),
    "file_system.read_chunked": ("read a file", "read {n} files"),
    "file_system.outline": ("read a file", "read {n} files"),
    "file_system.symbols": ("read a file", "read {n} files"),
    "file_system.find": ("searched the files", "searched the files {n} times"),
    "file_system.replace": ("edited a file", "edited {n} files"),
    "file_system.rename": ("renamed a file", "renamed {n} files"),
    "file_system.move": ("moved a file", "moved {n} files"),
    "file_system.copy": ("copied a file", "copied {n} files"),
    "file_system.delete": ("deleted a file", "deleted {n} files"),
    "file_system.download": ("downloaded a file", "downloaded {n} files"),
    "web_search": ("searched the web", "searched the web {n} times"),
    "deep_research": ("researched the web", "researched the web {n} times"),
    "darkweb_search": ("searched the dark web", "searched the dark web {n} times"),
    "darkweb_research": ("researched the dark web", "researched the dark web {n} times"),
    "fact_check": ("checked a fact", "checked {n} facts"),
    "youtube_transcribe": ("read a YouTube video", "read {n} YouTube videos"),
    "vision_analysis": ("looked at an image", "looked at {n} images"),
    "system_utility": ("checked the system", "checked the system {n} times"),
    "report_pdf": ("made a PDF report", "made {n} PDF reports"),
    "scratchpad": ("took a note", "took {n} notes"),
    "workspace": ("looked at the workspace", "looked at the workspace {n} times"),
    "browser": ("used the browser", "used the browser {n} times"),
    "fetch_url": ("read a web page", "read {n} web pages"),
    "recall": ("searched its memory", "searched its memory {n} times"),
    "remember": ("saved a memory", "saved {n} memories"),
    "knowledge_base": ("searched the documents", "searched the documents {n} times"),
    "manage_projects": ("updated the project", "updated the project {n} times"),
    "manage_tasks": ("updated a scheduled task", "updated {n} scheduled tasks"),
    "image_generation": ("made an image", "made {n} images"),
    "vision": ("looked at an image", "looked at {n} images"),
    "weather": ("checked the weather", "checked the weather {n} times"),
    "news_headlines": ("read the news", "read the news {n} times"),
    "portfolio": ("checked the portfolio", "checked the portfolio {n} times"),
    "introspect": ("looked at its own state", "looked at its own state {n} times"),
    "labs": ("worked on a lab", "worked on a lab {n} times"),
    "checklist": ("worked through the checklist", "worked through the checklist {n} times"),
}


def tool_key(name: str, args: Any) -> str:
    """``file_system.read`` for a file operation, the tool's name otherwise."""
    name = str(name or "").strip() or "tool"
    if name == "file_system" and isinstance(args, dict):
        op = str(args.get("operation") or "").strip().lower()
        if op:
            return f"file_system.{op}"
    return name


def summary(counts: Dict[str, int]) -> str:
    """"Ran 2 commands, read 1 file" — the step in words, in the order the tools first ran. Pure."""
    parts = []
    for key, n in counts.items():
        try:
            n = int(n)
        except (TypeError, ValueError):
            continue
        if n <= 0:
            continue
        one, many = _VERBS.get(key) or _VERBS.get(key.split(".")[0]) or (
            f"used {key.split('.')[0].replace('_', ' ')}", f"used {key.split('.')[0].replace('_', ' ')} {{n}} times")
        parts.append(one if n == 1 else many.format(n=n))
    if not parts:
        return ""
    text = ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1]
    return text[0].upper() + text[1:]


# ── the sentence (Nova) ──────────────────────────────────────────────────────

def _clip(text: Any, n: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def intent_of(reasoning: str, content: str = "") -> str:
    """The end of the step's thinking (its last two sentences) — why it ran these calls; the text before
    the call when there is no thinking."""
    src = " ".join(str(reasoning or "").split()) or " ".join(str(content or "").split())
    if not src:
        return ""
    sents = [s for s in re.split(r"(?<=[.!?])\s+", src) if s.strip()]
    tail = " ".join(sents[-2:]) if sents else src
    return tail[-INTENT_CHARS:]


def _redact(text: str) -> str:
    try:
        from ..distill.redact import redact_text
        return redact_text(text)
    except Exception:  # noqa: BLE001 — never show a result the redactor could not read
        return ""


def build_prompt(request: str, intent: str, calls: List[Dict[str, Any]]) -> str:
    """The narrator's prompt (the one measured live on 11-16, plus: say the result, never the next
    step). ``calls``: ``{"tool", "args", "result"}``, the first ``CALLS_SHOWN``."""
    lines = []
    for c in calls[:CALLS_SHOWN]:
        args = c.get("args")
        if isinstance(args, dict):      # r1: redact each value WHOLE, then cut (a cut token escaped the patterns)
            args = ", ".join(f"{k}={_clip(_redact(str(v)[:2000]), 60)}" for k, v in list(args.items())[:4])
        else:
            args = _redact(str(args or "")[:2000])
        # r1: cut BEFORE redacting (a 300k result took ~80 ms on the event loop), with room for the patterns
        res = _redact(str(c.get("result") or "")[: RESULT_CHARS * 4])
        lines.append(f"- {c.get('tool')}: {_clip(args, ARG_CHARS)}\n"
                     f"  result (start): {_clip(res, RESULT_CHARS)}")
    more = len(calls) - CALLS_SHOWN
    if more > 0:
        lines.append(f"- … and {more} more call(s)")
    return (
        "You narrate an AI assistant's work to its user, one step at a time, like a progress line.\n"
        f"USER'S REQUEST: {_clip(_redact(str(request or '')[:2000]), REQUEST_CHARS)}\n"
        f"YOUR OWN NOTE (why you did it): {_redact(intent) if intent else '(none)'}\n"
        "STEP JUST DONE:\n" + "\n".join(lines) + "\n\n"
        "Write ONE short sentence (max 22 words) in English, as the assistant speaking to the user in the first "
        "person (\"I ...\"): what you just did and what it showed or what you concluded — take the conclusion "
        "from your own note when it has one, and state the result when there is one. Plain words (the purpose, "
        "not the command). Past tense. Never say what you will do next. No greeting, no quotes.")


def clean_line(text: Any) -> str:
    """The narrator's answer as one shown line, or "" when it is not one clean English sentence."""
    s = " ".join(str(text or "").split()).strip().strip('"“”\'')
    if not s or len(s) > LINE_MAX_CHARS or _FOREIGN_RE.search(s) or _PROMISE_RE.search(s):
        return ""
    if len(_SENTENCE_END_RE.findall(s.rstrip(".!?") + " ")) > 0:
        return ""                                   # a sentence end before the last: more than one sentence
    if s.startswith(("{", "[", "<", "-", "*", "#")):
        return ""
    return s if s.endswith((".", "!", "?")) else s + "."


class ProgressNarrator:
    """One per streamed owner request (``for_request``). ``after_batch`` is called by the turn loop after
    each tool batch; ``close`` when the answer starts. Never raises."""

    def __init__(self, tap, llm_client, request: str):
        self.tap = tap
        self.llm = llm_client
        self.request = str(request or "")
        self.seq = 0
        self.closed = False
        self.steps: List[Dict[str, Any]] = []      # kept for the turn record
        self._pending: Optional[Dict[str, Any]] = None
        self._running = False
        self._task = None

    @classmethod
    def for_request(cls, context, request: str, req_id: str) -> Optional["ProgressNarrator"]:
        if not enabled():
            return None
        try:
            from .reply_tap import reply_tap_context
            tap = reply_tap_context.get()
            # r1: only a client that shows steps asks for them (the web console's proxy and the CLI send
            # `X-Ghost-Progress: 1`); the handheld ignored the frames and its turns still cost Nova calls
            if tap is None or not hasattr(tap, "progress") or not getattr(tap, "progress_wanted", False):
                return None
            from ..utils.logging import is_probe_request_id
            if is_probe_request_id(req_id) and os.environ.get("GHOST_PROGRESS_PROBES", "0") != "1":
                return None
            llm = getattr(context, "llm_client", None)
            if not worker_is_off_main(llm):     # r1: without an off-main worker the call would wait for, then
                return None                     # take, the main model's slot right after the turn
            return cls(tap, llm, request)
        except Exception as e:  # noqa: BLE001
            logger.debug("progress narrator off: %s", e)
            return None

    # ── the loop's side ─────────────────────────────────────────────────
    def after_batch(self, rows: List[Dict[str, Any]], reasoning: str = "", content: str = "") -> None:
        """The batch's tool rows (what it appended to ``tools_run_this_turn``): send the step now, ask Nova
        for its sentence in the background."""
        if self.closed:
            return
        try:
            calls, counts = [], {}
            for r in rows or []:
                if not isinstance(r, dict) or not str(r.get("name") or "").strip():
                    continue                        # a tool result row carries its tool's name
                if r.get("_synthetic"):             # r1: refused/blocked before it ran — not work done
                    continue
                res = r.get("content")
                args = getattr(res, "call_args", None)
                key = tool_key(r.get("name"), args)
                counts[key] = counts.get(key, 0) + 1
                calls.append({"tool": key, "args": args if isinstance(args, dict) else "",
                              "result": str(res or "")})
            if not calls:
                return
            self.seq += 1
            step = {"seq": self.seq, "tools": counts, "summary": summary(counts), "at": round(time.time(), 1)}
            self.steps.append(dict(step))
            del self.steps[:-KEPT_STEPS]
            self.tap.progress(step)
            if self.llm is None:
                return
            self._pending = {"seq": self.seq, "intent": intent_of(reasoning, content), "calls": calls}
            if not self._running:
                self._running = True
                from ..utils.logging import spawn_bg
                self._task = spawn_bg(self._drain(), name="progress-narrator")
        except Exception as e:  # noqa: BLE001 — progress never breaks a turn
            logger.debug("progress step skipped: %s: %s", type(e).__name__, e)

    def close(self) -> None:
        """The answer has started: a sentence that comes later is not progress any more."""
        if self.closed:
            return
        self.closed = True
        self._pending = None
        t, self._task = self._task, None
        if t is not None and not t.done():          # r1: a call still in flight is not waited for
            try:
                t.cancel()
            except Exception:  # noqa: BLE001
                pass

    # ── Nova ────────────────────────────────────────────────────────────
    async def _drain(self) -> None:
        try:
            while self._pending is not None and not self.closed:
                job, self._pending = self._pending, None        # the newest step; older ones collapsed
                line = await self._ask(job)
                # the answer's text already on screen (the last generation streams before the loop ends)
                if line and not self.closed and not str(getattr(self.tap, "sent", "") or "").strip():
                    self.tap.progress({"seq": job["seq"], "text": line})
                    for s in self.steps:
                        if s.get("seq") == job["seq"]:
                            s["text"] = line
                    try:
                        from ..utils.logging import pretty_log, Icons
                        pretty_log("Progress", line, icon=Icons.PROGRESS)
                    except Exception:  # noqa: BLE001
                        pass
        finally:
            self._running = False

    async def _ask(self, job: Dict[str, Any]) -> str:
        payload = {"messages": [{"role": "user", "content": build_prompt(self.request, job["intent"], job["calls"])}],
                   "temperature": 0.2, "max_tokens": NARRATE_MAX_TOKENS, "stream": False,
                   "chat_template_kwargs": {"enable_thinking": False}}
        try:
            data = await asyncio.wait_for(
                self.llm.chat_completion(payload, use_worker=True, is_background=True, off_main_only=True,
                                         timeout=NARRATE_TIMEOUT_S, task_label="progress"),
                timeout=NARRATE_TIMEOUT_S + 0.5)
            text = ((data or {}).get("choices") or [{}])[0].get("message", {}).get("content", "")
        except Exception as e:  # noqa: BLE001 — Nova busy, slow or gone: the step keeps its summary
            logger.debug("progress line skipped: %s: %s", type(e).__name__, e)
            return ""
        return clean_line(_redact(clean_line(text)))


#: the step lists of the latest requests, for their turn record (``extra.progress``)
_KEPT: "OrderedDict[str, List[Dict[str, Any]]]"
from collections import OrderedDict as _OD
_KEPT = _OD()
_KEPT_MAX = 64


def keep_steps(req_id: str, steps: List[Dict[str, Any]]) -> None:
    if not req_id or not steps:
        return
    _KEPT[str(req_id)] = steps
    _KEPT.move_to_end(str(req_id))
    while len(_KEPT) > _KEPT_MAX:
        _KEPT.popitem(last=False)


def steps_of(req_id: str) -> List[Dict[str, Any]]:
    """The steps request ``req_id`` showed (with their sentences), or []."""
    return [dict(s) for s in _KEPT.get(str(req_id or ""), [])]


def frame_bytes(chunk_id: str, created: int, model: str, progress: Dict[str, Any]) -> bytes:
    """The SSE frame: no choices, ``ghost.progress`` FIRST — the route tells it from reply text by its
    opening bytes (``is_progress_frame``) and never counts it as the reply having started."""
    obj = {"ghost": {"progress": progress}, "id": chunk_id, "object": "chat.completion.chunk",
           "created": int(created), "model": model, "choices": []}
    return f"data: {json.dumps(obj)}\n\n".encode("utf-8")


_PROGRESS_PREFIX = b'data: {"ghost": {"progress"'


def is_progress_frame(chunk: Any) -> bool:
    return isinstance(chunk, (bytes, bytearray)) and bytes(chunk).startswith(_PROGRESS_PREFIX)
