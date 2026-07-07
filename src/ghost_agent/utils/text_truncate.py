"""One head+tail truncation policy, shared across the tool layer.

Before this, four call sites each rolled their own truncation with different
budgets and message shapes (docker 256 KB, execute.py's dead 512 KB layer,
file_search 40 KB, browser 64 KB), so the effective limit on what could be
injected into the model's context depended on which code path won — the drift
class the truncation layers were meant to prevent. This module is the single
source of the head+tail policy; callers pass a budget and a label.
"""
from typing import Tuple


def truncate_head_tail(text: str, budget: int, label: str = "output",
                       head_frac: float = 0.25) -> Tuple[str, bool, int]:
    """Trim ``text`` to ``budget`` chars keeping the HEAD and TAIL.

    The tail matters most for tool output (Python tracebacks print the
    exception type at the very end; setup/context is at the head), so the tail
    gets the larger share by default (head_frac=0.25 → 25% head, 75% tail).

    Returns ``(result, was_truncated, dropped_chars)``. When ``text`` fits, it
    is returned unchanged with ``was_truncated=False``.
    """
    if not isinstance(text, str) or budget <= 0 or len(text) <= budget:
        return text, False, 0
    head_n = max(0, int(budget * head_frac))
    tail_n = max(0, budget - head_n)
    head = text[:head_n]
    tail = text[-tail_n:] if tail_n else ""
    dropped = len(text) - head_n - tail_n
    marker = (
        f"\n\n[... {dropped} chars truncated ({label}) — "
        f"showing first {head_n // 1024 or head_n} "
        f"{'KB' if head_n >= 1024 else 'chars'} and last "
        f"{tail_n // 1024 or tail_n} "
        f"{'KB' if tail_n >= 1024 else 'chars'} of {len(text) // 1024 or len(text)} "
        f"{'KB' if len(text) >= 1024 else 'chars'} total ...]\n\n"
    )
    return f"{head}{marker}{tail}", True, dropped
