"""§4MP: token streaming for the tools-on generation.

Since the planner arm was concluded (§4LF, 2026-10-04) every answer comes
from the tools-on call, which streams from llama-server internally but whose
text reached a streaming client only after the whole reply was finalized —
4–13 s (p90 ~18 s) after its first token. A `ReplyTap` forwards that text as
it is generated, under these rules:

* **hold** — nothing is sent before ``HOLD_CHARS`` of content: 84% of
  tool-call generations write no text first, and of those that do the p90
  is 238 chars (§4MP lens, 825 generations). A call inside the hold sends
  nothing.
* **may release** — the agent says when the text can be an answer at all:
  with thinking on and no reasoning channel yet, the content may itself be
  unparsed reasoning (§4KO/§4KP, r2 M2) and is held.
* **latch** — a call, think block, prompt bleed or raw-JSON call in the
  content (the agent's SHARED detectors, passed in as ``unsafe`` — r2 M3:
  a private copy missed four call shapes) or a native call delta stops
  forwarding for the generation; text already shown is retracted at once
  (r2: it used to stay on screen for the whole tool phase). The text from
  an unclosed ``<`` is never sent.
* **commit or retract** — when the request ends, the finalized reply either
  extends what was sent exactly (the rest goes out) or it does not: a
  ``ghost.retract`` frame tells the client to drop what it showed, and the
  whole reply follows. A new generation after shown text retracts first.

The tap is created by the chat route for an owner's streamed request and
reached through ``reply_tap_context``; the agent only feeds it. No tap — the
route's old path, byte for byte.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("GhostAgent")

reply_tap_context: "contextvars.ContextVar[Optional[ReplyTap]]" = contextvars.ContextVar(
    "reply_tap", default=None)

#: content sent before this much text exists could still turn out to be the
#: preamble of a tool call (p90 238 chars before a call)
HOLD_CHARS = 240
#: the fallback when no detector is passed (unit use): call/think/result
#: tags and chat-template tokens
_BASIC_UNSAFE_RE = re.compile(
    r"<\s*/?\s*(?:tool_call|tool(?=[^\w]|$)|function|parameter|think|tool_response)|<\|im_|<\|endoftext", re.I)
#: every marker is shorter than this — the re-scan window behind new text
#: (r2: re-scanning the whole buffer on every delta was quadratic)
_SCAN_OVERLAP = 96
FRAME_CHARS = 15


def _default_unsafe(text: str) -> bool:
    """The agent's shared detector when it is loaded (always, in the
    service); the basic tag check otherwise."""
    try:
        from .agent import reply_tap_unsafe
    except Exception:  # noqa: BLE001
        return bool(_BASIC_UNSAFE_RE.search(text))
    return reply_tap_unsafe(text) or bool(_BASIC_UNSAFE_RE.search(text))


def _frame(chunk_id: str, created: int, model: str, delta: Dict[str, Any],
           finish: Optional[str] = None, ghost: Optional[Dict[str, Any]] = None,
           extra: Optional[Dict[str, Any]] = None) -> bytes:
    obj: Dict[str, Any] = {"id": chunk_id, "object": "chat.completion.chunk", "created": created,
                           "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if extra:
        obj.update(extra)
    if ghost:
        obj["ghost"] = {**(obj.get("ghost") or {}), **ghost}
    return f"data: {json.dumps(obj)}\n\n".encode("utf-8")


class ReplyTap:
    """One per streamed request. The agent calls `begin_generation`, `feed`
    and `tool_call_seen`; the route drains `queue` while the request runs and
    calls `finish` with the finalized reply."""

    def __init__(self, req_id: str, model: str, created: int, hold_chars: int = HOLD_CHARS):
        self.chunk_id = f"chatcmpl-{req_id}"
        self.model = model
        self.created = int(created)
        self.hold_chars = int(hold_chars)
        self.queue: "asyncio.Queue[bytes]" = asyncio.Queue()
        self.sent = ""              # the text the client shows now
        self._gen_raw = ""
        self._gen_sent = 0          # how much of this generation went out
        self._scanned = 0           # how much of it the unsafe check has seen
        self._open = False
        self._latched = False
        self._role_sent = False
        self._unsafe: Callable[[str], bool] = _default_unsafe
        self.retracts: List[str] = []
        self.released_generations = 0
        self.outcome = "not streamed"

    # ── the agent's side ──────────────────────────────────────────────
    def bind(self, req_id: str) -> None:
        """The id the turn was filed under (a colliding client id is renamed
        `id#2` by the turn registry — r2: frames carried the route's)."""
        if req_id:
            self.chunk_id = f"chatcmpl-{req_id}"

    def begin_generation(self, unsafe: Optional[Callable[[str], bool]] = None) -> None:
        if self.sent:
            self._retract("a new generation")
        if unsafe is not None:
            self._unsafe = unsafe
        self._gen_raw, self._gen_sent, self._scanned = "", 0, 0
        self._open, self._latched = True, False

    def tool_call_seen(self) -> None:
        self._latch("a tool call followed")

    def end_generation(self) -> None:
        self._open = False

    def feed(self, full_content: str, may_release: bool = True) -> None:
        """``full_content``: this generation's content so far."""
        if not self._open or self._latched:
            return
        text = full_content or ""
        self._gen_raw = text
        start = max(0, self._scanned - _SCAN_OVERLAP)
        if text.lstrip().startswith("{"):       # the raw-JSON call shape (agent.py fallback parser)
            self._latch("raw JSON (a call shape)")
            return
        if self._unsafe(text[start:]):
            self._latch("call, think or bleed markup")
            return
        self._scanned = len(text)
        if not may_release:
            return
        if not self._gen_sent and len(text.strip()) < self.hold_chars:
            return
        safe_end = len(text)
        lt = text.rfind("<", max(0, len(text) - 32))
        if lt >= 0 and ">" not in text[lt:] and "\n" not in text[lt:]:
            safe_end = lt                       # may be the start of a marker
        if safe_end <= self._gen_sent:
            return
        piece = text[self._gen_sent:safe_end]
        if not self._gen_sent:
            piece = piece.lstrip()
            if not piece:
                return
            self.released_generations += 1
        self._gen_sent = safe_end
        self._send_text(piece)

    # ── the route's side ──────────────────────────────────────────────
    def finish(self, final_text: str, extra: Optional[Dict[str, Any]] = None,
               created: Optional[int] = None) -> List[bytes]:
        """The frames that close the reply: the rest of ``final_text`` when
        it extends what was sent exactly, else a retract and all of it."""
        final_text = final_text or ""
        out: List[bytes] = []
        if created:
            self.created = int(created)
        _shown = self.sent.replace("\r", "")        # finalize drops "\r" (§4MR: CRLF always retracted)
        if self.sent and final_text.startswith(_shown):
            rest = final_text[len(_shown):]
            outcome = "committed"
        elif self.sent and final_text.rstrip() == self.sent.rstrip():
            rest = ""                           # only trailing whitespace differs
            outcome = "committed"
        else:
            had_sent = bool(self.sent)
            if had_sent:
                out.append(self._retract_frame("the reply was rewritten"))
            rest = final_text
            outcome = "retracted" if (had_sent or self.retracts) else "not streamed"
        if not self._role_sent:
            out.append(_frame(self.chunk_id, self.created, self.model, {"role": "assistant"}, extra=extra))
            self._role_sent = True
        for i in range(0, len(rest), FRAME_CHARS):
            out.append(_frame(self.chunk_id, self.created, self.model,
                              {"content": rest[i:i + FRAME_CHARS]}, extra=extra))
        out.append(_frame(self.chunk_id, self.created, self.model, {}, finish="stop", extra=extra))
        out.append(b"data: [DONE]\n\n")
        self.outcome = outcome
        return out

    def retract_before_stream(self) -> List[bytes]:
        """A forced final (or an error) follows: drop what was shown first."""
        return [self._retract_frame("a streamed final follows")] if self.sent else []

    # ── internals ─────────────────────────────────────────────────────
    def _latch(self, why: str) -> None:
        self._latched = True
        if self.sent:
            self._retract(why)

    def _send_text(self, text: str) -> None:
        if not self._role_sent:
            self.queue.put_nowait(_frame(self.chunk_id, self.created, self.model, {"role": "assistant"}))
            self._role_sent = True
        for i in range(0, len(text), FRAME_CHARS):
            self.queue.put_nowait(_frame(self.chunk_id, self.created, self.model,
                                         {"content": text[i:i + FRAME_CHARS]}))
        self.sent += text

    def _retract_frame(self, why: str) -> bytes:
        self.retracts.append(why)
        self.sent = ""
        return _frame(self.chunk_id, self.created, self.model, {}, ghost={"retract": True})

    def _retract(self, why: str) -> None:
        self.queue.put_nowait(self._retract_frame(why))
