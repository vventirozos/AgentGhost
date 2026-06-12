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
from typing import Callable, Iterable, List, Pattern, Tuple, Union

from .schema import Trajectory


# Each pattern is a (name, regex, replacement) triple. `name` is used
# only for observability — counting how often each rule fires. The
# replacement is either a plain string or a match-callable (re.sub
# accepts both) — the callable form lets a rule validate the match
# before committing to the rewrite (see the Luhn-gated credit-card rule).
_BuiltinRule = Tuple[str, Pattern[str], Union[str, Callable[["re.Match[str]"], str]]]


def _luhn_ok(digits: str) -> bool:
    """True when `digits` passes the Luhn checksum (every real PAN does)."""
    if not digits.isdigit() or not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _redact_cc_if_luhn(m: "re.Match[str]") -> str:
    """Redact a 13-19 digit run ONLY when it Luhn-validates.

    A bare digit-run rule redacted every bigint id / sequence value /
    epoch-millis literal that wandered through a trajectory, corrupting
    the stored SQL/code corpus. Real card numbers always pass Luhn;
    arbitrary numeric literals pass ~10% of the time — so this keeps
    every true positive while dropping ~90% of the false ones.
    """
    return "<REDACTED_CC>" if _luhn_ok(re.sub(r"\D", "", m.group(0))) else m.group(0)

_BUILTIN_RULES: List[_BuiltinRule] = [
    # PEM private-key blocks (multi-line) — most specific, run first so the
    # whole block collapses before any sub-pattern nibbles at it.
    ("pem_private_key", re.compile(
        r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY-----"
        r"[\s\S]+?-----END (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY-----"
    ), "<REDACTED_PRIVATE_KEY>"),

    # Authorization headers (HTTP header form + JSON form). All schemes:
    # Basic carries reversible base64(user:pass), Token/Digest carry
    # credentials too — a Bearer-only rule leaked every other scheme.
    ("auth_header", re.compile(
        r"(?i)(authorization\"?\s*:\s*\"?)((?:bearer|basic|token|digest)\s+[^\s\"',]+)"
    ), r"\1<REDACTED_BEARER>"),

    # JWTs (header.payload.signature, all base64url).
    ("jwt", re.compile(
        r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
    ), "<REDACTED_JWT>"),

    # Anthropic keys FIRST (more specific) so they get their own label
    # before the generic openai `sk-` rule would swallow them.
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}"), "<REDACTED_API_KEY>"),
    # OpenAI / generic `sk-` (hyphen) prefixed keys.
    ("openai_key", re.compile(r"sk-[A-Za-z0-9_\-]{16,}"), "<REDACTED_API_KEY>"),
    # Stripe secret/restricted keys (underscore form — NOT caught by sk-).
    ("stripe_key", re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}\b"), "<REDACTED_API_KEY>"),
    # Google API keys.
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}\b"), "<REDACTED_API_KEY>"),

    # Slack tokens (bot/app/user/legacy).
    ("slack_bot_token", re.compile(r"xoxb-[0-9A-Za-z\-]{10,}"), "<REDACTED_SLACK_TOKEN>"),
    ("slack_app_token", re.compile(r"xapp-[0-9A-Za-z\-]{10,}"), "<REDACTED_SLACK_TOKEN>"),
    ("slack_user_token", re.compile(r"xox[pasr]-[0-9A-Za-z\-]{10,}"), "<REDACTED_SLACK_TOKEN>"),

    # GitHub tokens: classic ghp_/gho_/ghu_/ghs_ AND fine-grained github_pat_.
    ("github_finegrained_pat", re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "<REDACTED_GITHUB_PAT>"),
    ("github_pat", re.compile(r"gh[posu]_[A-Za-z0-9]{20,}"), "<REDACTED_GITHUB_TOKEN>"),

    # AWS access key IDs.
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "<REDACTED_AWS_KEY>"),

    # Credentials embedded in a DB / broker connection URI: redact only the
    # password between `user:` and `@host` (host/scheme are topology, not secret).
    # Username is OPTIONAL (`*` not `+`): the canonical Redis requirepass
    # form is `redis://:password@host` with an empty user — requiring a
    # username char leaked exactly that form.
    ("conn_uri_password", re.compile(
        r"((?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|rediss|amqp|amqps|mssql)://[^\s:\"'/]*:)"
        r"([^\s:\"'@/]+)(@)"
    ), r"\1<REDACTED>\3"),

    # Named secret env-var assignments (NAME=value or "NAME": "value").
    # The `\"?` BEFORE the separator is load-bearing: in the JSON form the
    # key's closing quote sits between the name and the colon, and without
    # it `{"GHOST_API_KEY": "..."}` never matched — leaking exactly the
    # named, non-self-identifying secrets (AWS secret keys have no prefix)
    # this rule exists for.
    ("env_assignment_secret", re.compile(
        r"((?:GHOST_API_KEY|SLACK_BOT_TOKEN|SLACK_APP_TOKEN|OPENAI_API_KEY|ANTHROPIC_API_KEY|HF_TOKEN|HUGGINGFACE_TOKEN|AWS_SECRET_ACCESS_KEY|GITHUB_TOKEN|GH_TOKEN|GOOGLE_API_KEY|STRIPE_SECRET_KEY)\"?\s*[=:]\s*\"?)[^\s\"',]+",
    ), r"\1<REDACTED>"),

    # Generic ALL-CAPS secret-shaped env assignment (…KEY/TOKEN/SECRET/
    # PASSWORD/PASSWD/CREDENTIAL = value). Prefers false positives.
    # Same optional closing quote before the separator as above.
    ("generic_secret_assignment", re.compile(
        r"\b([A-Z][A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)S?)(\"?\s*[=:]\s*\"?)([^\s\"',]+)"
    ), r"\1\2<REDACTED>"),

    # Lowercase form/query-string secrets (`client_secret=…`, `password=…`
    # in curl bodies and URLs). OAuth token exchanges and login form posts
    # are conventionally lowercase; the ALL-CAPS rule above and the JSON
    # rule below both miss the `key=value` spelling.
    ("form_secret_assignment", re.compile(
        r"(?i)\b((?:[a-z0-9_\-]*(?:password|passwd|secret|token|api_key|apikey))=)([^\s&\"',]+)"
    ), r"\1<REDACTED>"),

    # JSON-style "api_key": "..." / "token": "..."
    ("json_secret_field", re.compile(
        r'("(?:api[_-]?key|access[_-]?token|secret[_-]?key|auth[_-]?token|password|passwd|client[_-]?secret|refresh[_-]?token)"\s*:\s*")[^"]+(")',
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

    # IPv4 addresses outside loopback. Loopback is kept readable for debugging.
    ("ipv4", re.compile(
        r"\b(?!127\.)(?!0\.0\.0\.0\b)"
        r"((?:25[0-5]|2[0-4]\d|[01]?\d\d?)"
        r"(?:\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)){3})\b"
    ), "<REDACTED_IP>"),

    # IPv6 addresses: full form (>=4 hextet groups so `::1` loopback stays
    # readable) PLUS `::`-compressed forms — most real-world IPv6 is
    # compressed (`2001:db8::8a2e:370:7334`), and the full-form-only rule
    # leaked all of them. Bare `::1` still stays readable (the compressed
    # alternative needs at least one hextet before the `::`).
    ("ipv6", re.compile(
        r"\b(?:(?:[0-9A-Fa-f]{1,4}:){3,7}[0-9A-Fa-f]{1,4}"
        r"|(?:[0-9A-Fa-f]{1,4}:)+:(?:[0-9A-Fa-f]{1,4}:)*[0-9A-Fa-f]{1,4})\b"
    ), "<REDACTED_IP>"),

    # Phone numbers and credit-card numbers (ported from the selfhood diary
    # redactor so the higher-stakes corpus is at least as strict as the diary).
    # Credit cards are Luhn-gated (see _redact_cc_if_luhn) so bigint ids and
    # other long numeric literals in stored SQL/code survive intact.
    ("credit_card", re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"), _redact_cc_if_luhn),
    # A phone match must carry phone STRUCTURE — a leading `+`, a
    # parenthesised area code, or internal space/dash separators. The old
    # pattern's core (`\d{3}[ -]?\d{4}` with everything else optional)
    # matched any bare 7-10 digit integer, so `LIMIT 1000000` in a stored
    # SQL trajectory became `LIMIT <REDACTED_PHONE>`. Bare unseparated
    # digit runs are now left alone — code corpora are full of them and
    # an unformatted local number is not recoverable PII worth that cost.
    ("phone", re.compile(
        r"(?<!\d)(?:"
        # +country, optional area code, separators optional: +1 (212) 555-0123, +306912345678
        r"\+\d{1,3}[ -]?(?:\(?\d{2,4}\)?[ -]?)?\d{3}[ -]?\d{4}"
        # parenthesised area code: (212) 555-0123
        r"|\(\d{2,4}\)[ -]?\d{3}[ -]?\d{4}"
        # separator-delimited groups: 212-555-0123, 30 210 5550123
        r"|(?:\d{1,3}[ -])?\d{2,4}[ -]\d{3}[ -]?\d{4}"
        # 7-digit local WITH separator: 555-0123
        r"|\d{3}[ -]\d{4}"
        r")(?!\d)"
    ), "<REDACTED_PHONE>"),
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


# Compound names that END in a secret-ish suffix but are structurally
# public/benign — redacting them would corrupt otherwise-useful tool args
# (DB column names, key-pair public halves) for no privacy gain.
_BENIGN_COMPOUND_KEYS = frozenset({
    "primary_key", "foreign_key", "sort_key", "partition_key", "row_key",
    "public_key", "ssh_public_key", "cache_key", "idempotency_key",
})

_SENSITIVE_KEY_SUFFIXES = (
    "_key", "_token", "_secret", "_password", "_passwd", "_pwd",
    "_credential", "_credentials", "apikey",
)


def _is_sensitive_key(key) -> bool:
    norm = str(key).strip().lower().replace("-", "_").replace(" ", "_")
    if norm in _SENSITIVE_KEYS:
        return True
    if norm in _BENIGN_COMPOUND_KEYS:
        return False
    # Compound env-style names (GHOST_API_KEY, DB_PASSWORD, HF_TOKEN…):
    # exact-match keying missed every one of them, so structured tool
    # args like {"env": {"GHOST_API_KEY": …}} leaked the very key this
    # agent itself uses.
    return norm.endswith(_SENSITIVE_KEY_SUFFIXES)


def _redact_subtree(value):
    """Replace EVERY string leaf under `value` with ``<REDACTED>``.

    Used when a key is sensitive (see ``_SENSITIVE_KEYS``) but its value is
    a container — e.g. ``{"credentials": ["alice", "hunter2"]}`` or
    ``{"authorization": {"value": "tok"}}`` — where the opaque secret has
    nothing in its own shape to match on. Without this, only direct string
    values under a sensitive key were redacted and the container-valued
    case leaked.
    """
    if isinstance(value, str):
        return "<REDACTED>"
    if isinstance(value, dict):
        return {k: _redact_subtree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_subtree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_redact_subtree(v) for v in value)
    return value


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
            if _is_sensitive_key(k):
                # Sensitive key → redact the WHOLE value (string or container).
                out[k] = "<REDACTED>" if isinstance(v, str) else _redact_subtree(v)
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
