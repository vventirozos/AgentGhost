"""Secret redaction for exec/logs/files/http output (contract v2 §11.20).

Transcript hygiene, NOT a security boundary: a process inside the lab can
always transform a secret in ways no filter anticipates.
"""
from __future__ import annotations

import base64
import binascii
import json
import re
import urllib.parse
from typing import Dict, Iterable, List, Tuple

MIN_LEN = 6  # shorter values would redact ordinary words


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
    }
    return sorted((f for f in forms if len(f) >= MIN_LEN), key=len, reverse=True)


def build(secrets: Dict[str, str]) -> List[Tuple[re.Pattern, str]]:
    rules: List[Tuple[re.Pattern, str]] = []
    for name, value in secrets.items():
        if not value or len(value) < MIN_LEN:
            continue
        for form in _forms(value):
            # allow a line break (with optional CR) between any two characters: values split across lines
            pat = r"(?:\r?\n)?".join(re.escape(ch) for ch in form)
            rules.append((re.compile(pat), f"<secret:{name}>"))
    return rules


def apply(text: str, rules: Iterable[Tuple[re.Pattern, str]]) -> str:
    for pat, repl in rules:
        text = pat.sub(repl, text)
    return text
