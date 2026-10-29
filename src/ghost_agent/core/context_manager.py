# src/ghost_agent/core/context_manager.py
"""Progressive Context Compression.

Instead of a binary "everything fits" / "emergency prune" model, this module
progressively compresses older messages as the context window fills up.

Compression levels:
 L0 — No compression (everything verbatim)
 L1 — Summarize old tool outputs (keep last N turns full)
 L2 — Collapse repeated file reads into digest
 L3 — Merge planning updates into current plan state only
 L4 — Emergency prune (system + last user + last tool)
"""

import logging
from collections import OrderedDict
from ..utils.logging import pretty_log, Icons
from typing import Any, Dict, List, Optional

logger = logging.getLogger("GhostAgent")


class ContextManager:
    """Manages context window budget with progressive compression."""

    # How many recent turns to keep at full fidelity per compression level
    FULL_FIDELITY_TURNS = {0: 999, 1: 6, 2: 4, 3: 2, 4: 1}

    def __init__(self, max_tokens: int = 65536,
                 token_estimator=None):
        self.max_tokens = max_tokens
        self._estimate_tokens = token_estimator or self._default_estimate
        self._compression_level = 0
        # Bounded LRU-ish cache (insertion-ordered dict) so a long-lived
        # session with many distinct large tool outputs can't grow it without
        # bound. Oldest entries are evicted past the cap.
        self._summaries_cache: "OrderedDict[str, str]" = OrderedDict()
        self._summaries_cache_max = 256

    @staticmethod
    def _default_estimate(messages: List[dict]) -> int:
        """Rough token estimate: ~4 chars per token."""
        total = 0
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, str):
                total += len(content) // 4
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        total += len(str(part.get("text", ""))) // 4
            # §4OE-3: a native tool call's ARGUMENTS are prompt bytes too — a
            # 70 KB `write` body counted as nothing, so write-heavy turns never
            # reached the compressor until the L4 prune
            for tc in m.get("tool_calls") or []:
                if isinstance(tc, dict):
                    total += len(str(((tc.get("function") or {}).get("arguments")) or "")) // 4
        return total

    @property
    def compression_level(self) -> int:
        return self._compression_level

    def compress_if_needed(self, messages: List[dict],
                           llm_summarizer=None, max_level: int = 4) -> List[dict]:
        """Apply progressive compression if context budget is stressed.

        Parameters
        ----------
        messages : list of message dicts
        llm_summarizer : optional async callable(text) -> summary_text
            If provided, used for L1 summarization. If None, uses truncation.
        max_level : cap on the compression level (default 4). The agent hot
            path passes ``max_level=3`` so this only ever does CONTENT
            compression (every message preserved → tool-call/result pairing
            intact) and never L4 message-dropping — the existing
            ``_prune_context`` remains the L4 emergency.

        Returns
        -------
        Possibly-compressed message list.
        """
        usage = self._estimate_tokens(messages)

        # Determine compression level based on usage ratio
        ratio = usage / self.max_tokens if self.max_tokens > 0 else 0
        prev_level = self._compression_level
        if ratio < 0.60:
            self._compression_level = 0
            return messages
        elif ratio < 0.75:
            self._compression_level = 1
        elif ratio < 0.85:
            self._compression_level = 2
        elif ratio < 0.95:
            self._compression_level = 3
        else:
            self._compression_level = 4
        self._compression_level = min(self._compression_level, max_level)
        if self._compression_level == 0:
            return messages

        # Surface compaction on the monitored stream when it ESCALATES — the
        # operator otherwise can't see when answers begin degrading from
        # dropped context (this was logger.debug only, invisible live).
        if self._compression_level > prev_level:
            pretty_log(
                "Context Compaction",
                f"L{self._compression_level} ratio={ratio:.0%} ≈{usage}/{self.max_tokens} tok",
                icon=Icons.CUT,
                level=("WARNING" if self._compression_level >= 3 else "INFO"),
            )

        logger.debug(
            "ContextManager: ratio=%.2f level=%d tokens≈%d/%d",
            ratio, self._compression_level, usage, self.max_tokens,
        )

        return self._apply_compression(messages, self._compression_level)

    def reset_for_request(self) -> None:
        """§4N C-MINOR-2: the compression level persists on this cached
        instance ACROSS requests, and the escalation log fires only when the
        level INCREASES vs the previous call — so the second and later
        repeat-pressure requests compacted SILENTLY, defeating the "operator
        sees when answers begin degrading" purpose. Reset the level at each
        request boundary so every request's first escalation is logged."""
        self._compression_level = 0

    def _apply_compression(self, messages: List[dict],
                           level: int) -> List[dict]:
        """Apply the specified compression level."""
        if level == 0:
            return messages

        if level >= 4:
            return self._emergency_prune(messages)

        keep_full = self.FULL_FIDELITY_TURNS.get(level, 3)

        # Split: system messages + conversation messages
        system_msgs = [m for m in messages if m.get("role") == "system"]
        conv_msgs = [m for m in messages if m.get("role") != "system"]

        if len(conv_msgs) <= keep_full:
            return messages

        # Keep the last `keep_full` turns at full fidelity
        old_msgs = conv_msgs[:-keep_full]
        recent_msgs = conv_msgs[-keep_full:]

        # §4N C-MAJOR-1: the ORIGINAL GOAL (first conversation message —
        # always a user turn) must survive every compression stage. The LLM
        # prune downstream deliberately preserves `original_goal`, but this
        # deterministic ladder runs FIRST and L3 truncated it to 500 chars,
        # destroying any constraint stated past that point. Never compress
        # the goal message here — identify it by object identity so a later
        # message that happens to equal it is unaffected.
        goal_msg = next((m for m in conv_msgs if m.get("role") == "user"), None)
        # § context R1 A-F5b: the NEWEST user message carries the constraints
        # governing the CURRENT work — "protect the first user message" was a
        # name-proxy for "the instruction that governs the work". With the
        # live tail shape [..., user_current, assistant, tool] and
        # keep_full=2, L3 center-cut the current request itself (a hard
        # constraint at char 900 destroyed, reachable every turn via
        # compress_if_needed). Same identity rule as the goal.
        # §4RZ r6: the turn's request is the row handle_chat MARKED — the
        # newest user row is only a stand-in for it (a steer, an effort note or
        # a queued message can be newer); the stand-in remains without a mark
        newest_user_msg = (next((m for m in conv_msgs if m.get("_request")), None)
                           or next((m for m in reversed(conv_msgs)
                                    if m.get("role") == "user"), None))
        # §4OE-3: the ids of tool calls whose result is in and succeeded — a
        # 70 KB `write` argument stayed verbatim until the L4 prune (any
        # assistant message with a tool call was left whole); once the file
        # holds the bytes the prompt needs the path, not the body
        succeeded_ids = {str(m.get("tool_call_id")) for m in conv_msgs
                         if m.get("role") == "tool" and m.get("tool_call_id")
                         and str(m.get("content") or "").lstrip().startswith("SUCCESS")}
        compressed_old = []
        for msg in old_msgs:
            if msg is goal_msg or msg is newest_user_msg or msg.get("_queued"):
                # goal + current request are sacrosanct; so is what the user
                # sent while the turn ran (§4RZ r7: protected as the newest
                # user row until the request was marked)
                compressed_old.append(msg)
                continue
            msg = self._compact_written_args(msg, succeeded_ids)
            compressed = self._compress_message(msg, level)
            if compressed is not None:
                compressed_old.append(compressed)

        return system_msgs + compressed_old + recent_msgs

    def _compress_message(self, msg: dict, level: int) -> Optional[dict]:
        """Compress a single message based on the level."""
        role = msg.get("role", "")
        content = msg.get("content", "")

        if not isinstance(content, str):
            return msg

        # L1: Summarize tool outputs
        if level >= 1 and role == "tool":
            return self._summarize_tool_output(msg)

        # L2: Collapse verbose assistant messages with tool calls
        if level >= 2 and role == "assistant" and len(content) > 2000:
            # §4N C-MAJOR-4: a naive head/tail truncation of an assistant
            # message that carries `<tool_call>` markup splices the "[...
            # compressed ...]" marker INSIDE the JSON args while leaving
            # both <tool_call>/</tool_call> markers intact — history that
            # teaches the model malformed call syntax. Leave tool-call
            # messages whole (they are bounded by the model's own output;
            # L3 + the downstream tail-cap still bound total context).
            if "<tool_call>" in content or "</tool_call>" in content:
                return msg
            return self._keep_outcome(
                msg, self._truncate_with_summary(content, max_len=800))

        # L3: Compress user messages too (keep first 500 chars)
        if level >= 3 and role == "user" and len(content) > 1000:
            return {
                **msg,
                "content": content[:500] + "\n[... message truncated for context budget]",
            }

        return msg

    #: a written body longer than this is compacted once its write succeeded
    WRITTEN_ARGS_COMPACT_OVER = 2000
    WRITTEN_ARGS_HEAD = 240

    @classmethod
    def _compact_written_args(cls, msg: dict, succeeded_ids: set) -> dict:
        """§4OE-3: a native `file_system` write/append/edit call whose result
        said SUCCESS keeps its path and a head of the body; the body itself is
        on disk. The message object is copied, never mutated in place."""
        if not isinstance(msg, dict) or msg.get("role") != "assistant" or not msg.get("tool_calls"):
            return msg
        import json as _json
        import re as _re
        changed = False
        new_calls = []
        compact_args = {}                     # call index → compacted arguments JSON
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = (tc or {}).get("function") or {}
            args = fn.get("arguments")
            if (str(tc.get("id")) in succeeded_ids and fn.get("name") == "file_system"
                    and isinstance(args, str) and len(args) > cls.WRITTEN_ARGS_COMPACT_OVER):
                try:
                    a = _json.loads(args)
                except Exception:  # noqa: BLE001
                    a = None
                if isinstance(a, dict) and a.get("operation") in ("write", "append", "edit", "replace"):
                    body_key = next((k for k in ("content", "new_string", "replace_with") if isinstance(a.get(k), str)), None)
                    if body_key and len(a[body_key]) > cls.WRITTEN_ARGS_COMPACT_OVER:
                        n = len(a[body_key])
                        a = dict(a)
                        a[body_key] = (a[body_key][:cls.WRITTEN_ARGS_HEAD]
                                       + f"\n[... {n:,} chars written to the file — compacted from this "
                                         f"history; read the file for its text ...]")
                        new_args = _json.dumps(a, ensure_ascii=False)
                        tc = {**tc, "function": {**fn, "arguments": new_args}}
                        compact_args[i] = new_args
                        changed = True
            new_calls.append(tc)
        if not changed:
            return msg
        out = {**msg, "tool_calls": new_calls}
        # the inline lane (review 2): when the model's own `<tool_call>` text stays
        # in `content`, THAT is what reaches the wire. The live dialect is
        # `<function=file_system><parameter=content>…</parameter>` (JSON blocks
        # exist too). Matched by IDENTITY — tool + path — never by position: a
        # scrubbed parse error or a schema echo shifts positions.
        content = msg.get("content")
        if isinstance(content, str) and "<tool_call>" in content:
            by_path = {}
            for i in compact_args:
                try:
                    a = _json.loads(compact_args[i])
                    by_path[str(a.get("path") or "")] = a
                except Exception:  # noqa: BLE001
                    pass

            def _note(a):
                k = next((k for k in ("content", "new_string", "replace_with") if k in a), "content")
                return a[k], k

            def _sub(m):
                block = m.group(0)
                if "<function=file_system>" in block:
                    pm = _re.search(r"<parameter=path>\s*(.*?)\s*</parameter>", block, _re.S)
                    path = pm.group(1).strip() if pm else ""
                    a = by_path.get(path)
                    if a is None:
                        return block
                    head, k = _note(a)
                    return _re.sub(r"(<parameter=" + _re.escape(k) + r">)(.*?)(</parameter>)",
                                   lambda mm: mm.group(1) + head + mm.group(3), block, count=1, flags=_re.S)
                jm = _re.search(r"<tool_call>\s*(\{.*\})\s*</tool_call>", block, _re.S)
                if jm:
                    try:
                        j = _json.loads(jm.group(1))
                    except Exception:  # noqa: BLE001
                        return block
                    args = j.get("arguments") if isinstance(j, dict) else None
                    if (isinstance(j, dict) and j.get("name") == "file_system" and isinstance(args, dict)
                            and str(args.get("path") or "") in by_path):
                        j = {**j, "arguments": by_path[str(args.get("path") or "")]}
                        return "<tool_call>\n" + _json.dumps(j, ensure_ascii=False) + "\n</tool_call>"
                return block
            out["content"] = _re.sub(r"<tool_call>.*?</tool_call>", _sub, content, flags=_re.S)
        return out

    @staticmethod
    def _keep_outcome(msg: dict, new_text: str) -> dict:
        """Rewrite a message's content while KEEPING what the tool said.

        Two defects in one line of `{**msg, "content": <str>}`:

        1. `ToolOutcome` is a `str` subclass, so the rewrite returns a plain
           `str` and the status is destroyed.
        2. The keyword line-filter below HOISTS an `[error]`/`exception` line
           up into the first 120 characters, which is exactly the window the
           shared failure sniffer reads — so a SUCCESSFUL call acquired an
           error-looking head it never had.

        Together those flipped **67 corpus labels on 42 turns, 64 of them
        SUCCESS -> FAILED**, and the compressed list is what
        `_record_turn_trajectory` writes. `context_compressed` marks the text
        as no longer being the tool's own output, so the corpus stops
        re-deriving a label from it.
        """
        old = msg.get("content")
        st = getattr(old, "status", None)
        if st is None:
            return {**msg, "content": new_text}
        from ..tools.outcome import ToolOutcome
        return {**msg, "content": ToolOutcome(
            new_text, status=old.status, world_changed=old.world_changed,
            reason_code="context_compressed", declared=old.declared,
            # the call's arguments and wall time survive the cut (§4LQ N6)
            call_args=getattr(old, "call_args", None),
            duration_s=getattr(old, "duration_s", None))}

    def _summarize_tool_output(self, msg: dict) -> dict:
        """Compress a tool output message to its essential information."""
        content = msg.get("content", "")
        if not isinstance(content, str):
            return msg

        tool_name = msg.get("name", "tool")

        if len(content) <= 500:
            return msg

        # Check cache
        cache_key = f"{tool_name}:{hash(content)}"
        if cache_key in self._summaries_cache:
            self._summaries_cache.move_to_end(cache_key)  # LRU bump
            return self._keep_outcome(msg, self._summaries_cache[cache_key])

        # Heuristic compression: keep first and last lines, error lines
        lines = content.split("\n")
        if len(lines) <= 10:
            return msg

        compressed_lines = []
        compressed_lines.extend(lines[:3])  # First 3 lines (usually headers/structure)

        # Keep error/important lines
        for line in lines[3:-3]:
            lower = line.lower()
            if any(kw in lower for kw in ("error", "exception", "fail", "warning", "result", "total", "summary")):
                compressed_lines.append(line)

        compressed_lines.append(f"[... {len(lines) - 6} lines compressed]")
        compressed_lines.extend(lines[-3:])  # Last 3 lines (usually results)

        summary = "\n".join(compressed_lines)
        # Only use compression if it actually saves space
        if len(summary) < len(content) * 0.8:
            self._summaries_cache[cache_key] = summary
            self._summaries_cache.move_to_end(cache_key)
            while len(self._summaries_cache) > self._summaries_cache_max:
                self._summaries_cache.popitem(last=False)  # evict oldest
            return self._keep_outcome(msg, summary)
        return msg

    @staticmethod
    def _truncate_with_summary(text: str, max_len: int = 800) -> str:
        """Truncate text keeping the start and end."""
        if len(text) <= max_len:
            return text
        half = max_len // 2
        return (
            text[:half]
            + f"\n[... {len(text) - max_len} chars compressed ...]\n"
            + text[-half:]
        )

    @staticmethod
    def _emergency_prune(messages: List[dict]) -> List[dict]:
        """Level 4: Keep only system prompt + last user message + last tool result."""
        system_msgs = [m for m in messages if m.get("role") == "system"]
        last_user = None
        last_tool = None
        for m in reversed(messages):
            if m.get("role") == "user" and last_user is None:
                last_user = m
            elif m.get("role") == "tool" and last_tool is None:
                last_tool = m
            if last_user and last_tool:
                break

        result = list(system_msgs)
        # §4RZ step 3: the newest user row can be a message the user sent
        # mid-turn (or a loop steer) — keep the turn's REQUEST (the row
        # handle_chat marks) and what the user sent while it ran (newest
        # first, the summarising prune's 8,000-character budget, each row
        # whole, stopping at the first that does not fit — s3-r7), in order,
        # beside the newest user row. (Level 4 is not
        # reached by the turn loop today — it calls with max_level=3 — but
        # where it is, this list replaces the messages it was given.)
        _users = [m for m in messages if m.get("role") == "user"]
        _keep = {id(m) for m in _users if m.get("_request")}
        _left = 8000
        for m in reversed([m for m in _users if m.get("_queued")]):
            _n = min(4000, len(str(m.get("content") or "")))
            if _n > _left:
                break
            _left -= _n
            _keep.add(id(m))
        if last_user is not None:
            _keep.add(id(last_user))
        result += [m for m in _users if id(m) in _keep]
        if last_tool:
            # Truncate tool output in emergency
            content = last_tool.get("content", "")
            if isinstance(content, str) and len(content) > 1000:
                last_tool = ContextManager._keep_outcome(
                    last_tool, content[:1000] + "\n[EMERGENCY TRUNCATED]")
            result.append(last_tool)
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Return compression statistics."""
        return {
            "compression_level": self._compression_level,
            "max_tokens": self.max_tokens,
            "cached_summaries": len(self._summaries_cache),
        }
