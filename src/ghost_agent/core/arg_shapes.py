"""§4VN: a list or an object that arrived as a string, where the tool's schema
says list or object.

A tool call written as TEXT (the ``<tool_call>`` form the prompt teaches, read
by the agent's own parser) carries every parameter as text. For a parameter
the tool declares as an array or an object the model writes JSON there —
``[{"action": "goto", …}]`` — and the tool then received that JSON as a
string: the browser's ``interact`` answered "requires a non-empty 'actions'
list" four times in one request (2026-11-26, Inkling 2 Mini; 10 of 45 tool
calls in that run came as text). A native call carries the list itself.

This is NOT the blanket decoding of arguments §4RD removed, and must not grow
into it: a parameter the schema types as a string (file content, a command, a
regex) reaches the tool exactly as written. Only a parameter whose declared
type is ``array`` or ``object`` — and only when the string IS that JSON — is
replaced by the value.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

_OPEN = {"array": "[", "object": "{"}


def structured_params(tool_name: str, definitions: Iterable[dict]) -> Dict[str, str]:
    """``{parameter: "array" | "object"}`` for a tool, from its schema. A
    parameter that may ALSO be a string (a list of types, ``anyOf``) is left
    out: its text is a legitimate value."""
    for d in definitions or ():
        fn = (d or {}).get("function") or {}
        if fn.get("name") != tool_name:
            continue
        out = {}
        for name, spec in ((fn.get("parameters") or {}).get("properties") or {}).items():
            kind = (spec or {}).get("type") if isinstance(spec, dict) else None
            if isinstance(kind, str) and kind in _OPEN:
                out[name] = kind
        return out
    return {}


def decode_structured(tool_name: str, args: Any, definitions: Optional[Iterable[dict]]) -> List[str]:
    """Replace, in ``args``, each string that stands where ``tool_name``
    declares an array or an object and that parses as exactly that. Returns
    the names replaced. Anything else — a string that is not that JSON, a
    parameter of another type, an unknown tool — is left untouched."""
    if not isinstance(args, dict):
        return []
    changed = []
    for name, kind in structured_params(tool_name, definitions or ()).items():
        value = args.get(name)
        if not isinstance(value, str):
            continue
        text = value.strip()
        if not text.startswith(_OPEN[kind]):
            continue
        try:
            parsed = json.loads(text)
        except (ValueError, RecursionError):        # not that JSON (or nested past any use): as written
            continue
        args[name] = parsed                 # text that opens with "[" (or "{") and parses IS a list (an object)
        changed.append(name)
    return changed
