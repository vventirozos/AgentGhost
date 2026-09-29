"""§4KP — the reasoning/content split llama-server gets wrong.

llama-server's reasoning parser ends the reasoning at the FIRST ``</think>`` it
sees — including one the model merely WRITES ("reasoning blocks end with
`</think>`"). Everything after that mention reaches the caller as CONTENT: the
rest of the reasoning, the real close, the answer. Measured on the live server
(2026-09-29):

* a real close leaves the reasoning ending in ``\\n`` (the model closes on a
  line of its own) — streamed and non-streamed alike;
* a mention split leaves it ending MID-LINE ("...end with `") and the content
  resumes that same line ("`.\\n ...").

So a reasoning channel that stops mid-line is the tell. The repair then looks
for the real close later in the content (the tag starting its own line, then a
newline): what is before it is reasoning, what is after it is the answer. With
no later close the model wrote its answer INSIDE the reasoning (probe-ff2dd1d7:
reasoning "…\\n\\nA closing `", content "` tag signals…"); only when the tag was
inside an open code span on that line is the answer rebuilt from the last
reasoning paragraph + the tag + the content. Anything else is left as the
server sent it.

Streams: a split is detected at the reasoning→content transition, before any
content byte is forwarded; only then is content held (until the real close or
the end of the stream). Every other stream passes through byte-identical.
"""
from __future__ import annotations

import copy
import json
import re
from typing import Any, AsyncIterator, Dict, Optional, Tuple

# The real close: the tag at the start of its own line, then a newline (or the
# end of the text, once the text is complete).
_REAL_CLOSE_RE = re.compile(r'(?:\A|\n)[ \t]*</think(?:ing)?[ \t]*>[ \t]*(?:\r?\n|\Z)', re.IGNORECASE)
_REAL_CLOSE_OPEN_RE = re.compile(r'(?:\A|\n)[ \t]*</think(?:ing)?[ \t]*>[ \t]*\r?\n', re.IGNORECASE)


def is_mention_split(reasoning: Any) -> bool:
    """True when the reasoning channel stops mid-line — the parser ended it at
    a ``</think>`` the model wrote inside a line, not at a real close."""
    if not isinstance(reasoning, str) or not reasoning.strip():
        return False
    return not reasoning.rstrip(" \t").endswith("\n")


def _open_code_span(reasoning: str) -> bool:
    """The last reasoning line has an unclosed inline code span — the tag was
    written inside backticks (a fence opener does not count)."""
    last = reasoning[reasoning.rfind("\n") + 1:]
    return last.replace("```", "").count("`") % 2 == 1


def resolve_split(reasoning: str, content: str, *, complete: bool, rescue: bool = True,
                  pos: int = 0) -> Optional[Tuple[str, str]]:
    """``(reasoning, content)`` repaired, or None when nothing can be decided
    yet (``complete=False``) or nothing should change. ``rescue=False`` (a
    generation cut by its token cap) never rebuilds an answer from the
    reasoning; ``pos`` starts the close search there (a stream re-scans only
    the new tail).

    Call only when :func:`is_mention_split` holds for ``reasoning``."""
    m = (_REAL_CLOSE_RE if complete else _REAL_CLOSE_OPEN_RE).search(content, max(0, pos))
    if m:
        return reasoning + "</think>" + content[:m.start()], content[m.end():].lstrip("\r\n")
    if not complete or not rescue:
        return None
    if _open_code_span(reasoning):
        cut = reasoning.rfind("\n\n")
        head = reasoning[cut + 2:] if cut >= 0 else reasoning
        return (reasoning[:cut] if cut >= 0 else ""), (head + "</think>" + content).lstrip()
    return None


def repair_message(result: Any) -> bool:
    """Repair a non-streamed completion's first message in place. True when
    something changed. Never raises. An EMPTY content is the token cap cutting
    the reasoning (callers detect it by that emptiness) — never an answer."""
    try:
        choice = result["choices"][0]
        msg = choice["message"]
        reasoning, content = msg.get("reasoning_content"), msg.get("content")
        if not isinstance(content, str) or not content.strip() or not is_mention_split(reasoning):
            return False
        fixed = resolve_split(reasoning, content, complete=True,
                              rescue=choice.get("finish_reason") != "length")
        if fixed is None:
            return False
        msg["reasoning_content"], msg["content"] = fixed
        return True
    except Exception:  # noqa: BLE001 — a repair never breaks a call
        return False


def _frame(chunk: Any) -> Optional[Dict[str, Any]]:
    try:
        text = chunk.decode("utf-8") if isinstance(chunk, (bytes, bytearray)) else str(chunk)
        text = text.strip()
        if not text.startswith("data: {") or "\n" in text:
            return None
        d = json.loads(text[6:])
        return d if isinstance(d, dict) else None
    except Exception:  # noqa: BLE001
        return None


def _delta(frame: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        d = frame["choices"][0].get("delta") if frame else None
        return d if isinstance(d, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _is_end(chunk: Any, frame: Optional[Dict[str, Any]]) -> bool:
    if frame is None:
        try:
            text = chunk.decode("utf-8") if isinstance(chunk, (bytes, bytearray)) else str(chunk)
        except Exception:  # noqa: BLE001
            return False
        return text.strip() == "data: [DONE]"
    if "error" in frame and "choices" not in frame:
        return True
    try:
        return bool(frame["choices"][0].get("finish_reason"))
    except Exception:  # noqa: BLE001
        return False


def _without_content(frame: Dict[str, Any]) -> bytes:
    """An end frame whose content was folded into the held text: its
    finish_reason / usage still reach the caller."""
    d = copy.deepcopy(frame)
    try:
        d["choices"][0]["delta"] = {k: v for k, v in (d["choices"][0].get("delta") or {}).items() if k != "content"}
    except Exception:  # noqa: BLE001
        pass
    return f"data: {json.dumps(d, ensure_ascii=False)}\n\n".encode("utf-8")


def _synth(template: Dict[str, Any], key: str, text: str) -> bytes:
    d = copy.deepcopy(template)
    for k in ("usage", "timings", "logprobs"):
        d.pop(k, None)
    d["choices"] = [{"index": 0, "delta": {key: text}, "finish_reason": None}]
    return f"data: {json.dumps(d, ensure_ascii=False)}\n\n".encode("utf-8")


#: What the consumer receives for each upstream frame folded into a hold: an
#: SSE comment every client skips. It keeps the consumer's heartbeat and cancel
#: checks turning; content-fed guards (loop / flood detectors) see nothing
#: until release, which is why a hold is bounded (`HOLD_MAX_CHARS`, the first
#: native tool_calls frame).
HELD_TICK = b": held\n\n"
#: A hold longer than this is released unchanged: no recorded split carries
#: more than a paragraph of reasoning past the mention.
HOLD_MAX_CHARS = 8000


async def repair_stream(chunks: AsyncIterator[Any]) -> AsyncIterator[Any]:
    """Pass an upstream SSE stream through, repairing a mention split (see the
    module docstring). Unaffected streams are yielded chunk for chunk. Only a
    generation that ended with ``finish_reason: stop`` may have its answer
    rebuilt from the reasoning; a cut one (length, a stall/abort error frame,
    a stream that just ends) is released as the server sent it."""
    reasoning = ""
    state = "reasoning"          # → "pass" | "hold"
    held_text = ""
    held_frames: list = []       # non-content frames that arrived while holding
    template: Optional[Dict[str, Any]] = None
    trim_lead = False            # a close resolved mid-stream: drop the blank line after it
    try:
        async for chunk in chunks:
            if state == "pass":
                if trim_lead:
                    frame = _frame(chunk)
                    c = _delta(frame).get("content")
                    if isinstance(c, str) and c:
                        trimmed = c.lstrip("\r\n")
                        if not trimmed:
                            # the blank line goes; the frame's other fields
                            # (finish_reason, tool_calls, …) do not
                            others = {k for k, v in _delta(frame).items() if k != "content" and v}
                            yield _without_content(frame) if (others or _finish(frame)) else HELD_TICK
                            continue
                        trim_lead = False
                        if trimmed != c:
                            yield _with_content(frame, trimmed)
                            continue
                yield chunk
                continue
            frame = _frame(chunk)
            delta = _delta(frame)
            content = delta.get("content")
            has_content = isinstance(content, str) and bool(content)
            if state == "reasoning":
                rc = delta.get("reasoning_content")
                if isinstance(rc, str):
                    reasoning += rc
                if not has_content:
                    yield chunk
                    continue
                if not is_mention_split(reasoning):
                    state = "pass"
                    yield chunk
                    continue
                state, template = "hold", frame
            # holding — the close search restarts at the last newline already held
            scan_from = max(0, held_text.rfind("\n"))
            if has_content:
                held_text += content
            ends = _is_end(chunk, frame)
            if ends or delta.get("tool_calls") or len(held_text) > HOLD_MAX_CHARS:
                # decide on what is held, then release in order: the end of the
                # generation, a native call (content is over; the flood guards
                # must see the calls), or a hold past its bound
                _clean_end = ends and _finish(frame) == "stop"
                # only a clean stop sees the text as complete: a cut ending in
                # "\n</think>" is a mention cut at a line start, not a close
                outs, closed_empty = _flush(reasoning, held_text, template, complete=_clean_end,
                                            rescue=_clean_end)
                for out in outs:
                    yield out
                trim_lead = closed_empty and not ends
                for f in held_frames:
                    yield f
                held_text, held_frames, state = "", [], "pass"
                yield _without_content(frame) if has_content else chunk
                continue
            if has_content:
                fixed = resolve_split(reasoning, held_text, complete=False, pos=scan_from)
                if fixed is not None:
                    yield _synth(template, "reasoning_content", fixed[0][len(reasoning):])
                    if fixed[1]:
                        yield _synth(template, "content", fixed[1])
                    else:
                        trim_lead = True
                    for f in held_frames:
                        yield f
                    held_frames, state = [], "pass"
                    continue
            else:
                held_frames.append(chunk)
            yield HELD_TICK
        if state == "hold":
            # the stream just ended — no finish frame: a cut generation
            for out in _flush(reasoning, held_text, template, complete=False, rescue=False)[0]:
                yield out
            for f in held_frames:
                yield f
    finally:
        aclose = getattr(chunks, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:  # noqa: BLE001
                pass


def _with_content(frame: Dict[str, Any], text: str) -> bytes:
    d = copy.deepcopy(frame)
    d["choices"][0]["delta"] = dict(d["choices"][0].get("delta") or {}, content=text)
    return f"data: {json.dumps(d, ensure_ascii=False)}\n\n".encode("utf-8")


def _finish(frame: Optional[Dict[str, Any]]) -> Optional[str]:
    try:
        return frame["choices"][0].get("finish_reason") if frame else None
    except Exception:  # noqa: BLE001
        return None


def _flush(reasoning: str, held_text: str, template: Optional[Dict[str, Any]], *,
           complete: bool = True, rescue: bool = True):
    """The frames that release a hold, and whether a close was found with no
    answer text after it yet. ``complete=False`` (a hold released early: its
    bound, a native call) accepts only a close already followed by a newline —
    a mention cut at a line start is not one."""
    if template is None or not held_text:
        return [], False
    fixed = resolve_split(reasoning, held_text, complete=complete, rescue=rescue)
    if fixed is None:
        return [_synth(template, "content", held_text)], False
    out = []
    if fixed[0] != reasoning and fixed[0].startswith(reasoning):
        out.append(_synth(template, "reasoning_content", fixed[0][len(reasoning):]))
    if fixed[1]:
        out.append(_synth(template, "content", fixed[1]))
    return out, not fixed[1]
