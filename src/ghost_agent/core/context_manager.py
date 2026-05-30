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
        return total

    @property
    def compression_level(self) -> int:
        return self._compression_level

    def compress_if_needed(self, messages: List[dict],
                           llm_summarizer=None) -> List[dict]:
        """Apply progressive compression if context budget is stressed.

        Parameters
        ----------
        messages : list of message dicts
        llm_summarizer : optional async callable(text) -> summary_text
            If provided, used for L1 summarization. If None, uses truncation.

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

        compressed_old = []
        for msg in old_msgs:
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
            return {
                **msg,
                "content": self._truncate_with_summary(content, max_len=800),
            }

        # L3: Compress user messages too (keep first 500 chars)
        if level >= 3 and role == "user" and len(content) > 1000:
            return {
                **msg,
                "content": content[:500] + "\n[... message truncated for context budget]",
            }

        return msg

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
            return {**msg, "content": self._summaries_cache[cache_key]}

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
            return {**msg, "content": summary}
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
        if last_user:
            result.append(last_user)
        if last_tool:
            # Truncate tool output in emergency
            content = last_tool.get("content", "")
            if isinstance(content, str) and len(content) > 1000:
                last_tool = {**last_tool, "content": content[:1000] + "\n[EMERGENCY TRUNCATED]"}
            result.append(last_tool)
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Return compression statistics."""
        return {
            "compression_level": self._compression_level,
            "max_tokens": self.max_tokens,
            "cached_summaries": len(self._summaries_cache),
        }
