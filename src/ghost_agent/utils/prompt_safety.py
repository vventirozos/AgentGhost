"""Content the agent READ must never become chat-template control (§4MB).

llama-server tokenises the rendered chat prompt with special-token parsing,
so the TEXT ``<|im_end|>\\n<|im_start|>system`` inside a web page, a file or
a transcript became the real role tokens — the page opened a genuine system
turn (measured on the live server: four special tokens out of a tool
result). Every Qwen control tag is a single special token: ``<|…|>``,
``<tool_response>``, ``<tool_call>``, ``<think>``.

:func:`defuse_text` runs where a tool result ENTERS the conversation (the
dispatcher's tool message, and the sites that re-wrap tool results as user
text in ``<tool_response>`` tags), and :func:`defuse_payload` re-applies it
to every ``role: "tool"`` message at the client's send sites. It rewrites the
role-forging tags so they tokenise as ordinary characters (``<|`` → ``‹|``,
``</tool_response>`` → ``‹/tool_response›``).

User and system messages are NOT touched: they carry the agent's OWN tool
instructions and ``<tool_response>`` wrappers (the review found the first
cut rewrote ``QWEN_TOOL_PROMPT``'s ``<tool_call>`` examples into a syntax the
parser refuses). Deterministic, and a no-op when nothing matches.
"""
from __future__ import annotations

import re
from typing import Any, Dict

_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]{1,40})\|>")
#: the tag that closes the tool result early; ``<tool_call>`` / ``<think>``
#: inside a tool result forge nothing (calls are parsed from the MODEL's
#: output) and stay, so a project file that contains them still reads — and
#: edits — exactly as written
_ROLE_TAG_RE = re.compile(r"<(/?)(tool_response)>")


def defuse_text(text: str, role: str = "tool") -> str:
    """``text`` with the ROLE-FORGING control tags made inert: every
    ``<|…|>`` special and ``<tool_response>``/``</tool_response>``."""
    if not text or "<" not in text:
        return text
    out = _SPECIAL_RE.sub(lambda m: f"‹|{m.group(1)}|›", text)
    return _ROLE_TAG_RE.sub(lambda m: f"‹{m.group(1)}{m.group(2)}›", out)


def _defuse_content(content: Any, role: str):
    if isinstance(content, str):
        new = defuse_text(content, role)
        return new if new != content else content
    if isinstance(content, list):
        changed, parts = False, []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                t = defuse_text(part["text"], role)
                if t != part["text"]:
                    part = {**part, "text": t}
                    changed = True
            parts.append(part)
        return parts if changed else content
    return content


def defuse_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The payload with every TOOL message's text defused — a NEW
    dict/list only where something changed; the caller's objects are never
    mutated."""
    msgs = payload.get("messages") if isinstance(payload, dict) else None
    if not isinstance(msgs, list):
        return payload
    new_msgs, changed = [], False
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "tool" and "content" in m:
            c = _defuse_content(m["content"], str(m.get("role") or ""))
            if c is not m["content"]:
                m = {**m, "content": c}
                changed = True
        new_msgs.append(m)
    return {**payload, "messages": new_msgs} if changed else payload
