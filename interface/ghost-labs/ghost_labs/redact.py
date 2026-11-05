"""Secret redaction for exec/logs/files/http output (contract v2 §11.20).

Transcript hygiene, NOT a security boundary: a process inside the lab can
always transform a secret in ways no filter anticipates.

Two rule sets are built from the same forms: text rules (``build``/``apply``)
and byte rules (``build_bytes``/``apply_bytes``) for output that is not valid
UTF-8 (review 11-05 #8: a file with one bad byte came back unredacted).
``cut_tail``/``cut_head`` redact a window fetched with a margin and THEN cut
(review 11-05 #9): a secret straddling the cut is replaced whole, never halved.
"""
from __future__ import annotations

import base64
import binascii
import json
import re
import shlex
import urllib.parse
from typing import Dict, Iterable, List, Tuple, Union

Secrets = Union[Dict[str, str], Iterable[Tuple[str, str]]]

MIN_LEN = 6  # shorter values would redact ordinary words
_BREAK = r"(?:\r?\n)?"     # a line break (with optional CR) may sit between any two characters
_BREAK_B = rb"(?:\r?\n)?"


def _forms(value: str) -> List[str]:
    b = value.encode()
    forms = {
        value,
        base64.b64encode(b).decode(),
        base64.b64encode(b).decode().rstrip("="),
        base64.urlsafe_b64encode(b).decode().rstrip("="),
        urllib.parse.quote(value, safe=""),
        urllib.parse.quote_plus(value),
        binascii.hexlify(b).decode(),
        binascii.hexlify(b).decode().upper(),
        json.dumps(value)[1:-1],
        # the form /run/ghost/secrets.env holds (review 11-05 #10: a value with ' is written split
        # as 'a'"'"'b', which no other form matches)
        shlex.quote(value),
    }
    return sorted((f for f in forms if len(f) >= MIN_LEN), key=len, reverse=True)


def _usable(secrets: Secrets):
    """(name, value) pairs from a dict or an iterable of pairs (one name may carry several values)."""
    for name, value in (secrets.items() if isinstance(secrets, dict) else secrets):
        if value and len(value) >= MIN_LEN:
            yield name, value


def build(secrets: Secrets) -> List[Tuple[re.Pattern, str]]:
    rules: List[Tuple[re.Pattern, str]] = []
    for name, value in _usable(secrets):
        for form in _forms(value):
            pat = _BREAK.join(re.escape(ch) for ch in form)
            rules.append((re.compile(pat), f"<secret:{name}>"))
    return rules


def build_bytes(secrets: Secrets) -> List[Tuple[re.Pattern, bytes]]:
    """The same forms as ``build``, matched on the UTF-8 bytes (a break may sit between any two bytes)."""
    rules: List[Tuple[re.Pattern, bytes]] = []
    for name, value in _usable(secrets):
        for form in _forms(value):
            pat = _BREAK_B.join(re.escape(bytes([x])) for x in form.encode())
            rules.append((re.compile(pat), f"<secret:{name}>".encode()))
    return rules


def max_match(secrets: Secrets) -> int:
    """An upper bound on the length (bytes) of one redactable match: the longest form, with a CRLF
    allowed between every two bytes."""
    longest = max((len(f.encode()) for _, v in _usable(secrets) for f in _forms(v)), default=0)
    return 3 * longest


def margin(secrets: Secrets) -> int:
    """How much more than the wanted size to fetch before redacting and cutting."""
    return 2 * max_match(secrets) + 16


def apply(text: str, rules: Iterable[Tuple[re.Pattern, str]]) -> str:
    for pat, repl in rules:
        text = pat.sub(repl, text)
    return text


def apply_bytes(data: bytes, rules: Iterable[Tuple[re.Pattern, bytes]]) -> bytes:
    for pat, repl in rules:
        data = pat.sub(repl, data)
    return data


def _spans(data: bytes, rules) -> List[Tuple[int, int]]:
    return [m.span() for pat, _ in rules for m in pat.finditer(data)]


def cut_tail(window: bytes, keep: int, rules, truncated: bool) -> bytes:
    """The last ``keep`` bytes of ``window``, redacted. ``truncated``: the window is itself the tail of
    something longer (fetched with ``margin`` extra bytes), so a partial secret may open it; that part is
    never returned. The cut moves back over any match it would split (the result can be a little longer)."""
    if len(window) <= keep and not truncated:
        return apply_bytes(window, rules)
    p = max(0, len(window) - keep)
    spans = _spans(window, rules)
    moved = True
    while moved:
        moved = False
        for s, e in spans:
            if s < p < e:
                p, moved = s, True
    return apply_bytes(window[p:], rules)


def cut_head(window: bytes, keep: int, rules) -> bytes:
    """The first ``keep`` bytes of ``window`` (fetched with ``margin`` extra), redacted. The cut moves
    forward over any match it would split."""
    if len(window) <= keep:
        return apply_bytes(window, rules)
    p = keep
    spans = _spans(window, rules)
    moved = True
    while moved:
        moved = False
        for s, e in spans:
            if s < p < e:
                p, moved = e, True
    return apply_bytes(window[:p], rules)
