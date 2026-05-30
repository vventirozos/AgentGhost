"""Tool fallback chains.

When a tool fails, instead of burning a strike and hoping the LLM picks
a better tool next turn, we can automatically suggest (or attempt) the
next-best tool in a predefined fallback chain. This reduces wasted turns
on transient failures and gives the agent a recovery path.

The chains are advisory — they're injected as hints into the diagnostic
message so the LLM makes the final decision, rather than being executed
automatically (which could cause unintended side effects).
"""

import logging
from typing import Optional

logger = logging.getLogger("GhostAgent")

# Fallback chains: tool_name → list of (fallback_tool, hint) pairs.
# When tool_name fails, suggest the first applicable fallback.
FALLBACK_CHAINS = {
    "deep_research": [
        ("web_search", "Try a simpler web_search with a focused query instead of deep_research."),
        ("recall", "Check if the answer is already in memory using recall."),
    ],
    "web_search": [
        ("deep_research", "Try deep_research for a more thorough investigation."),
        ("recall", "Check if you already have this information in memory."),
    ],
    "execute": [
        ("file_system", "Read the file first with file_system to check for syntax errors before re-executing."),
    ],
    "postgres_admin": [
        ("execute", "If the database connection failed, try executing SQL via a Python script with execute."),
    ],
    "vision_analysis": [
        ("file_system", "If vision analysis failed, try reading the file as text with file_system."),
    ],
    "delegate_to_swarm": [
        ("execute", "If swarm delegation failed, try executing the task directly with execute."),
    ],
}


def get_fallback_hint(failed_tool: str, error_text: str = "") -> Optional[str]:
    """Get a fallback suggestion for a failed tool.

    Returns a hint string suggesting the next tool to try, or None if
    no fallback chain exists for this tool.
    """
    chain = FALLBACK_CHAINS.get(failed_tool)
    if not chain:
        return None

    error_lower = (error_text or "").lower()

    for fallback_tool, hint in chain:
        # Skip if the error suggests the fallback would also fail
        if fallback_tool == "web_search" and "captcha" in error_lower:
            continue
        if fallback_tool == "execute" and "docker" in error_lower:
            continue
        return f"FALLBACK SUGGESTION: {hint}"

    return None
