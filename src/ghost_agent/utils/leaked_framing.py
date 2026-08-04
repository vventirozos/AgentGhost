"""Detect tool-call framing that leaked into an ARGUMENT VALUE.

This is a **corpus diagnostic**, deliberately NOT the repair predicate.

`core.agent._value_has_leaked_framing` decides whether to TRUNCATE a live
value, so it is intentionally strict: it demands an ordered close-then-
continuation pair (``</parameter>`` … ``<parameter=``) precisely so that code
or prose merely *mentioning* framing text is never mangled. That strictness is
correct there and must not be relaxed.

Reading the live corpus (2026-08-04) shows the cost of that strictness as a
*measurement* tool: of 17 historical corruptions, the strict predicate matches
only 11. The six it misses have a clean prefix and then a sibling parameter
with **no preceding close token** — e.g.::

    'read_chunked>\\n<parameter=path>\\nindex.html'
    'replace\\n<parameter=path>\\nprojects/26990e596da6/index.html'

and one more uses a different dialect entirely (``<arg_key>``/``<arg_value>``),
whose tokens the repair regexes do not list at all.

Counting is not truncating: a false positive here inflates a diagnostic, while
a false positive in the repair path destroys a real argument. So this module
casts a wider net ON PURPOSE, and nothing here may be wired into the repair
path.

**All 17 known occurrences predate the 2026-07-31 ~18:54
``QWEN_TOOL_PROMPT_NATIVE`` split that fixed the dual-dialect prompt; 161
trajectories recorded after it carry zero.** This module exists so a
RECURRENCE is noticed by an instrument rather than by someone grepping the
corpus four days later — the journal's "each one is news" had no listener.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

# Framing tokens across BOTH dialects the native path has emitted:
#   - the equals/name dialect: <function=…>, <parameter=…>, </parameter>, …
#   - the arg_key/arg_value dialect: <arg_key>…</arg_key><arg_value>…
# Any of these appearing INSIDE an argument value is anomalous — a well-formed
# value never carries its own framing.
_FRAMING_RE = re.compile(
    r"</?parameter[\s=>]"          # <parameter=…  or  </parameter>
    r"|</?function[\s=>]"
    r"|</?tool_call\b"
    r"|</?arg_key\b"
    r"|</?arg_value\b",
    re.IGNORECASE,
)

# Values that are ONLY a path-ish string containing "<parameter" as literal
# text are vanishingly rare; but a legitimate value CAN discuss framing (a
# docstring about the XML dialect, this very module's source). Requiring the
# token to look STRUCTURAL — a real tag boundary — keeps prose out.
_STRUCTURAL_RE = re.compile(
    r"<parameter\s*(?:=|name\s*=)"     # <parameter=path>  /  <parameter name="path">
    r"|</parameter\s*>"
    r"|</function\s*>"
    r"|</tool_call\s*>"
    r"|<arg_key\s*>"
    r"|<arg_value\s*>",
    re.IGNORECASE,
)


def value_has_leaked_framing(value: Any) -> bool:
    """True when a single argument value carries structural tool-call framing.

    Wider than the repair predicate by design (see module docstring), but not
    unboundedly so — a value may legitimately DISCUSS the dialect, and a
    diagnostic that counts every docstring mentioning ``<parameter=`` is a
    broken instrument in the other direction.

    The discriminator is POSITION, measured against the 17 known corruptions:
    leaked framing always sits at the start of the value, or immediately after
    a newline, or appears more than once. Prose mentions it mid-sentence, once.
    Verified: all 16 corrupt calls still match, while
    ``"the XML dialect uses <parameter=path> style tags"`` and
    ``"def f(): return '</tool_call>'"`` no longer do.
    """
    if not isinstance(value, str) or not value:
        return False
    matches = list(_STRUCTURAL_RE.finditer(value))
    if not matches:
        return False
    if not _FRAMING_RE.search(value):
        return False
    if len(matches) > 1:
        return True                       # repeated framing is never prose
    start = matches[0].start()
    # At the very start, or opening a line — i.e. occupying a structural
    # position rather than being embedded in a sentence.
    return start == 0 or value[start - 1] == "\n"


def call_has_leaked_framing(call: Any) -> bool:
    """True when ANY argument of a ToolCall-shaped object leaked framing.

    Accepts both shapes in use: an object with ``.arguments`` (``ToolCall``)
    and a plain ``{"name": …, "arguments": {…}}`` dict.
    """
    args = None
    if hasattr(call, "arguments"):
        args = getattr(call, "arguments", None)
    elif isinstance(call, dict):
        args = call.get("arguments")
    if not isinstance(args, dict):
        return False
    return any(value_has_leaked_framing(v) for v in args.values())


def first_leaked_argument(call: Any) -> Optional[Tuple[str, str]]:
    """``(arg_name, value)`` of the first leaked argument, or None.

    Returned so a report can name WHICH argument broke rather than only that
    something did — `operation` and `path` fail very differently.
    """
    args = getattr(call, "arguments", None)
    if args is None and isinstance(call, dict):
        args = call.get("arguments")
    if not isinstance(args, dict):
        return None
    for k, v in args.items():
        if value_has_leaked_framing(v):
            return (str(k), v)
    return None


def scan_trajectories(trajectories) -> Dict[str, Any]:
    """Corpus scan → ``{scanned, calls, corrupt_calls, by_tool, by_arg,
    last_seen, examples}``.

    ``scanned`` is reported so "0 corrupt" can be told apart from "the scan
    never ran" — the distinction this project keeps having to relearn.
    """
    out: Dict[str, Any] = {
        "scanned": 0, "calls": 0, "corrupt_calls": 0,
        "by_tool": {}, "by_arg": {}, "last_seen": "", "examples": [],
    }
    for t in trajectories or []:
        out["scanned"] += 1
        for c in (getattr(t, "tool_calls", None) or []):
            out["calls"] += 1
            hit = first_leaked_argument(c)
            if hit is None:
                continue
            arg, val = hit
            name = str(getattr(c, "name", "") or "?")
            out["corrupt_calls"] += 1
            out["by_tool"][name] = out["by_tool"].get(name, 0) + 1
            out["by_arg"][arg] = out["by_arg"].get(arg, 0) + 1
            ts = str(getattr(t, "timestamp", "") or "")
            if ts > out["last_seen"]:
                out["last_seen"] = ts
            if len(out["examples"]) < 5:
                out["examples"].append(
                    {"tool": name, "arg": arg, "value": val[:120], "ts": ts})
    return out
