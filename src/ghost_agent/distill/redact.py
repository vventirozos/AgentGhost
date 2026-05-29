"""Trajectory redaction.

Ghost's sandbox output genuinely does contain sensitive material — API
keys, Tor circuit identifiers, absolute paths pointing at the operator's
home directory, Slack tokens. Even though the trajectory store is
local-only, redacting at write time means:

  1. The on-disk corpus is safe to hand to a future training script
     (SFT / GRPO) without re-scanning.
  2. If the trajectory store is ever accidentally copied off the box
     (backup, mis-configured sync), the blast radius is much smaller.

Patterns are opinionated but conservative: prefer false positives
(redacting something that wasn't actually a secret) over false negatives
(leaking a real one).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Pattern, Tuple

from .schema import Trajectory


# Each pattern is a (name, regex, replacement) triple. `name` is used
# only for observability — counting how often each rule fires.
_BuiltinRule = Tuple[str, Pattern[str], str]

_BUILTIN_RULES: List[_BuiltinRule] = [
    # Bearer / Authorization headers. The optional quotes around the
    # field name and around the value cover both HTTP header form
    # (`Authorization: Bearer xxx`) and JSON form (`"Authorization":
    # "Bearer xxx"`).
    ("auth_header", re.compile(
        r"(?i)(authorization\"?\s*:\s*\"?)(bearer\s+[^\s\"',]+)"
    ), r"\1<REDACTED_BEARER>"),

    # OpenAI / Anthropic / generic sk- prefixed keys
    ("openai_key", re.compile(r"sk-[A-Za-z0-9_\-]{16,}"), "<REDACTED_API_KEY>"),
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}"), "<REDACTED_API_KEY>"),

    # Slack tokens
    ("slack_bot_token", re.compile(r"xoxb-[0-9A-Za-z\-]{10,}"), "<REDACTED_SLACK_TOKEN>"),
    ("slack_app_token", re.compile(r"xapp-[0-9A-Za-z\-]{10,}"), "<REDACTED_SLACK_TOKEN>"),
    ("slack_user_token", re.compile(r"xoxp-[0-9A-Za-z\-]{10,}"), "<REDACTED_SLACK_TOKEN>"),

    # GitHub personal access tokens
    ("github_pat", re.compile(r"ghp_[A-Za-z0-9]{20,}"), "<REDACTED_GITHUB_PAT>"),
    ("github_oauth", re.compile(r"gho_[A-Za-z0-9]{20,}"), "<REDACTED_GITHUB_TOKEN>"),

    # AWS access key IDs (distinctive prefix + length)
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}"), "<REDACTED_AWS_KEY>"),

    # Generic hex-looking tokens in env-var assignments
    ("env_assignment_secret", re.compile(
        r"((?:GHOST_API_KEY|SLACK_BOT_TOKEN|SLACK_APP_TOKEN|OPENAI_API_KEY|ANTHROPIC_API_KEY|HF_TOKEN|HUGGINGFACE_TOKEN|AWS_SECRET_ACCESS_KEY)\s*[=:]\s*)[^\s\"',]+",
    ), r"\1<REDACTED>"),

    # JSON-style "api_key": "..." / "token": "..."
    ("json_secret_field", re.compile(
        r'("(?:api[_-]?key|access[_-]?token|secret[_-]?key|auth[_-]?token|password|passwd)"\s*:\s*")[^"]+(")',
        re.IGNORECASE,
    ), r"\1<REDACTED>\2"),

    # .onion hostnames
    ("tor_onion", re.compile(r"\b[a-z2-7]{16,56}\.onion\b"), "<REDACTED_ONION>"),

    # Absolute /Users/<name>/ paths — replace the name segment only
    ("macos_home", re.compile(r"/Users/[^/\s\"':]+"), "/Users/<user>"),

    # Absolute /home/<name>/ paths
    ("linux_home", re.compile(r"/home/[^/\s\"':]+"), "/home/<user>"),

    # Email addresses (fairly conservative — requires real-looking TLD chars)
    ("email", re.compile(
        r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,24}"
    ), "<REDACTED_EMAIL>"),

    # IPv4 addresses outside loopback + private ranges. We redact ALL
    # v4 addresses because a leaked address even from a private range
    # reveals local topology, and we can always un-redact in analysis.
    # Loopback is kept readable for debugging.
    ("ipv4", re.compile(
        r"\b(?!127\.)(?!0\.0\.0\.0\b)"
        r"((?:25[0-5]|2[0-4]\d|[01]?\d\d?)"
        r"(?:\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)){3})\b"
    ), "<REDACTED_IP>"),
]


@dataclass
class RedactionConfig:
    """Tunable redaction. Defaults cover everything; callers can narrow
    or extend as needed.

    `extra_rules` are appended to built-ins and run in declaration
    order — earlier rules take priority on overlapping matches.
    """

    disabled_rules: Iterable[str] = field(default_factory=tuple)
    extra_rules: List[_BuiltinRule] = field(default_factory=list)


def redact_text(text: str, config: RedactionConfig | None = None) -> str:
    if not text:
        return text
    cfg = config or RedactionConfig()
    disabled = set(cfg.disabled_rules or ())
    out = text
    for name, rx, repl in _BUILTIN_RULES:
        if name in disabled:
            continue
        out = rx.sub(repl, out)
    for name, rx, repl in cfg.extra_rules:
        if name in disabled:
            continue
        out = rx.sub(repl, out)
    return out


# Dict keys whose *string* value is a secret regardless of the value's
# own shape. The standalone `redact_text` rules are self-identifying
# (they key off the value: `sk-…`, `AKIA…`, `…@…`). But a bare token
# under a telling key — `{"password": "hunter2"}`, `{"Authorization":
# "Bearer …"}` — has nothing in the *value* to match on once it's been
# split out of its surrounding JSON. Keying off the field name closes
# that gap. Compared case-insensitively with `-`/space normalised to `_`.
_SENSITIVE_KEYS = frozenset({
    "authorization", "api_key", "apikey", "access_token", "accesstoken",
    "secret_key", "secretkey", "auth_token", "authtoken", "token",
    "password", "passwd", "pwd", "secret", "cookie", "set_cookie",
    "private_key", "privatekey", "client_secret", "refresh_token",
    "session_token", "x_api_key", "credentials", "credential",
})


def _is_sensitive_key(key) -> bool:
    return str(key).strip().lower().replace("-", "_").replace(" ", "_") in _SENSITIVE_KEYS


def _redact_value(value, _r):
    """Recursively redact every string *leaf* inside a nested container.

    Tool-call arguments are arbitrary JSON: a secret can sit inside a
    list (``env=["OPENAI_API_KEY=sk-…"]``) or a nested dict
    (``headers={"Authorization": "Bearer …"}``). A shallow "redact only
    top-level str values" pass leaks all of those. We walk dict/list/
    tuple containers and redact each string leaf, leaving non-string
    scalars (int/float/bool/None) untouched so the structure round-trips.
    A string value under a sensitive *key* (see `_SENSITIVE_KEYS`) is
    replaced wholesale, since the value alone may not be self-identifying.
    """
    if isinstance(value, str):
        return _r(value)
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if isinstance(v, str) and _is_sensitive_key(k):
                out[k] = "<REDACTED>"
            else:
                out[k] = _redact_value(v, _r)
        return out
    if isinstance(value, list):
        return [_redact_value(v, _r) for v in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(v, _r) for v in value)
    return value


def redact_trajectory(traj: Trajectory, config: RedactionConfig | None = None) -> Trajectory:
    """Return a redacted copy of `traj`. Does not mutate the input.

    Redaction applies to every free-text field: system_prompt,
    user_request, planning_output, final_response, failure_reason,
    each tool_call's arguments (string leaves at ANY nesting depth),
    result, and error, and the `extra` / `validator_signal` metadata
    dicts (both are serialized by `to_jsonl`, so a secret captured into
    a runner metric would otherwise reach the on-disk corpus unredacted).
    """
    cfg = config or RedactionConfig()
    from dataclasses import replace

    def _r(s: str) -> str:
        return redact_text(s, cfg)

    redacted_calls = []
    for tc in traj.tool_calls:
        # Pass the whole arguments dict through _redact_value so the
        # dict branch applies sensitive-key redaction to top-level keys
        # too (not just nested ones).
        new_args = _redact_value(tc.arguments, _r)
        redacted_calls.append(type(tc)(
            name=tc.name,
            arguments=new_args,
            result=_r(tc.result or ""),
            error=_r(tc.error or ""),
            duration_s=tc.duration_s,
        ))

    return replace(
        traj,
        system_prompt=_r(traj.system_prompt or ""),
        user_request=_r(traj.user_request or ""),
        planning_output=_r(traj.planning_output) if traj.planning_output is not None else None,
        final_response=_r(traj.final_response or ""),
        failure_reason=_r(traj.failure_reason or ""),
        tool_calls=redacted_calls,
        extra=_redact_value(traj.extra, _r),
        validator_signal=_redact_value(traj.validator_signal, _r),
    )
