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


# Words that precede a dotted-numeric run which is NOT a host. `7.2.1.2` is a
# structurally valid dotted quad, so no pattern can separate it from an
# address — the context can. These are the false-positive classes MEASURED in
# the live corpus (the PostgreSQL manual is a known RAG source), encoded the
# same way `_BENIGN_COMPOUND_KEYS` encodes its own.
# Deliberately ONLY document-structure and versioning words. `rule`, `item`,
# `step`, `table` and bare `v` were here and were removed: each doubles as a
# plausible config KEY (`rule: 10.0.0.5`), so keeping them traded a corrupted
# manual section number for a leaked host. When the two error directions
# conflict in a redactor, the leak is the one that matters.
_IPV4_NON_HOST_CUES = (
    "section", "chapter", "figure", "appendix", "clause",
    "version", "release", "rev",
)

# `Name/1.2.3.4` is a version only for these names. Everything else before a
# `/` is treated as a path segment and the quad is redacted — see
# `_ipv4_repl`. Kept short and explicit on purpose: the one measured false
# positive was a Chrome UA inside stored Python source, and guessing at the
# shape instead of the name is what leaked real hosts.
_UA_PRODUCT_TOKENS = frozenset((
    "mozilla", "chrome", "chromium", "safari", "firefox", "edge", "edg",
    "opera", "applewebkit", "gecko", "webkit", "trident", "msie",
    "curl", "wget", "python-requests", "okhttp", "httpx", "urllib",
    "postman", "insomnia", "node-fetch", "axios", "go-http-client",
))


def _ipv4_repl(m) -> str:
    """Redact a dotted quad unless the context says it is not a host.

    A quad preceded by `/` is redacted, FULL STOP, unless the token before
    the slash is a known user-agent product name. `/` must not simply be
    excluded by a lookbehind — doing that spared `http://<host>`,
    `https://<host>:8080/api` and `/var/log/<host>.log`.

    The first attempt at the exception was structural ("a product token is a
    word not itself preceded by a separator") and leaked broadly: any path
    segment containing `-`, `_`, `.` or `:` defeated it, so
    `/var/log/my-app/10.0.0.9`, `http://example.com/10.0.0.9` and
    `/opt/ghost_agent/10.0.0.9` all passed through verbatim — 36 of 83 real
    directories on the live box were leak-triggering prefixes, and a path
    segment of 39+ chars leaked regardless of content. There is no
    structural difference between `logs/10.0.0.9` and `Chrome/120.0.0.0`;
    only the NAME distinguishes them. So the exception is an explicit
    allowlist, and everything not on it is treated as a host.

    That deliberately re-corrupts nothing measured except unknown product
    tokens, and it errs toward redaction: a false positive mangles stored
    text, a false negative publishes an address.
    """
    head = m.string[:m.start()]
    # Never scan across a line break: `rstrip()` used to eat `\n`, so a
    # markdown heading ("## Section\n10.0.0.9") suppressed redaction on the
    # NEXT line. A cue only speaks for its own line.
    line = head.rsplit("\n", 1)[-1][-60:]
    if line.endswith("/"):
        stem = line[:-1]
        word = ""
        for ch in reversed(stem):
            if ch.isalnum():
                word = ch + word
            else:
                break
        return (m.group(0) if word.lower() in _UA_PRODUCT_TOKENS
                else "<REDACTED_IP>")
    # Last alphabetic token before the match, ignoring punctuation/space.
    tail = line.rstrip()
    tail = tail.rstrip(".:#§-_ \t")
    word = ""
    for ch in reversed(tail):
        if ch.isalpha():
            word = ch + word
        elif word:
            break
        else:
            break
    if word.lower() in _IPV4_NON_HOST_CUES:
        return m.group(0)
    return "<REDACTED_IP>"


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
        # Password runs up to the `@` delimiter. The old class excluded `:`
        # and `/`, so a password CONTAINING either (`admin:aB/cD3f@host`, a
        # base64/colon-bearing password) truncated the match before `@`, the
        # rule failed to fire, and the credential leaked verbatim into the
        # corpus. Only whitespace / quotes / the `@` delimiter now terminate it.
        r"([^\s\"'@]+)(@)"
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

    # JSON-style "api_key": "..." / "token": "...". MUST run before
    # form_secret_assignment: on a double-quoted JSON pair the form
    # rule's value charset stops at the first escaped quote
    # (`"a\"b_secret"` → redacts only `a\`, leaks the tail), while this
    # rule's escape-aware value body consumes the whole string.
    ("json_secret_field", re.compile(
        # Value body tolerates escaped quotes (`"a\"b_secret"`): the old
        # `[^"]+` stopped at the first inner quote, redacting only `a\` and
        # leaking the rest.
        r'("(?:api[_-]?key|access[_-]?token|secret[_-]?key|auth[_-]?token|password|passwd|client[_-]?secret|refresh[_-]?token)"\s*:\s*")(?:\\.|[^"\\])+(")',
        re.IGNORECASE,
    ), r"\1<REDACTED>\2"),

    # Lowercase / mixed-case secret assignments: form/query-string bodies
    # (`client_secret=…`), YAML/config style (`password: hunter2`,
    # `api_key: sk_live_…`) and spaced prose (`db_password = hunter2`).
    # OAuth token exchanges and login form posts are conventionally
    # lowercase; the ALL-CAPS rule above and the quoted-JSON rule above
    # both miss these spellings. Only names ENDING in a word from the
    # sensitive list fire — a bare `key: value` never matches. `pwd`/`auth`
    # need a `_`/`-` boundary (or to stand alone) so prose like
    # `oauth: client_credentials` isn't redacted. Separators stay on one
    # line ([ \t], not \s) so a name at end-of-line can't eat the next line.
    ("form_secret_assignment", re.compile(
        r"(?i)\b((?:[a-z0-9_\-]*"
        r"(?:password|passwd|secret|token|api_key|apikey|access_key|private_key)"
        r"|(?:[a-z0-9_\-]+[_\-])?(?:pwd|auth))"
        r"[\"']?[ \t]*[=:][ \t]*[\"']?)([^\s&\"',]+)"
    ), r"\1<REDACTED>"),

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
    #
    # STRUCTURE GUARD (added after measuring the live corpus): the sibling
    # `credit_card` rule got a Luhn check and `phone` got a separator
    # requirement precisely to stop numeric false positives corrupting stored
    # text — `ipv4` had neither, and most of the 41 live `<REDACTED_IP>` hits
    # were NOT hosts. It ate PostgreSQL manual section numbers ("see Section
    # 7.2.1"), a Chrome UA version inside stored Python source
    # ("Chrome/120.0.0.0"), part of a loopback ("127.0.0.1" -> the last octet),
    # and digits inside an opaque CDN token.
    #
    # (An earlier version of this comment said "33 of 41". Re-measured
    # 2026-08-04 by classifying every marker: the split is ~22 non-host /
    # ~19 host-shaped, and of those only one — 192.168.215.2, in the Flask
    # `* Running on` line — is a genuine host on this box; the rest are
    # synthetic fixture addresses from coding exercises. The false-positive
    # problem was real; its size was overstated.)
    #
    # The guards, all cheap and all lossless for real addresses:
    #   * not preceded by a digit or a dot — kills the tail of a longer
    #     dotted-numeric run (section numbers, version strings, and
    #     "127.0.0.1" whose 127. prefix the negative lookahead already spares).
    #     Slash and hyphen were in this class and were REMOVED: they leaked
    #     every URL host, every path-embedded address, and the second address
    #     of an "a-b" range. Those shapes are now separated by context in
    #     `_ipv4_repl`, which is where a judgement call belongs;
    #   * not followed by a dot+digit — kills "1.2.3.4.5" style enumerations;
    #   * at least one octet > 255-impossible-as-a-version, i.e. reject the
    #     all-small-numbers shape that version strings take, UNLESS it is
    #     preceded by an address-ish cue. Rather than guess, require that the
    #     match is not immediately inside a word character on either side.
    ("ipv4", re.compile(
        # No leading `\b`. `(?<![\d.])` already prevents matching the tail of
        # a longer dotted-numeric run, and the word boundary additionally
        # required a NON-word char before the first octet — which a
        # JSON-escaped newline does not provide. Measured on the live corpus:
        # `"STDOUT/STDERR:\\n10.0.0.2 12\\n10.0.0.4 11..."` left seven
        # addresses in the clear, because the `n` of the literal `\n` escape
        # is a word character. Serialized tool output is exactly where
        # addresses live.
        r"(?<![\d.])"
        r"(?!127\.)(?!0\.0\.0\.0\b)"
        r"((?:25[0-5]|2[0-4]\d|[01]?\d\d?)"
        r"(?:\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)){3})\b"
        r"(?![\d.]*\d)"
    ), _ipv4_repl),

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
    # Letter-excluding boundaries: a 13-19 digit run INSIDE a longer
    # alphanumeric token (32-hex trajectory/request ids) is an identifier,
    # not a card — Luhn alone passes ~10% of those and the redaction
    # corrupts referential ids downstream.
    # The DOT matters, but ASYMMETRICALLY. A long float in stored JSON —
    # e.g. `"ballY": 1863.6640967916142` — has a 13+ digit run after the
    # decimal point that passes Luhn ~10% of the time, and redacting it
    # corrupts the record. So a dot BEFORE the run disqualifies it.
    # After the run the guard must be `(?!\.\d)`, not `(?!\.)`: a plain
    # trailing-dot veto let a card at the end of a sentence — "my card is
    # 4111111111111111." — escape redaction ENTIRELY, which is the failure
    # direction that actually matters here. `\.\d` still vetoes the other
    # float shape (`12345678901234.5`).
    # Boundaries also exclude `-`, `_` and `|`. Luhn passes ~10% of long
    # digit runs by chance, and the corpus is full of them INSIDE delimited
    # identifiers — measured, 95 of 147 Luhn-valid runs in the live corpus
    # were redacted and NONE was a card:
    #   `unsplash.com/photo-1504567991286-3a325b4b0e98`  (53 occurrences)
    #   `1785569266492|E105|START|5`                     (epoch-millis logs)
    #   `linkedin.com/.../activity-7395082818684583936-PW89`
    # A real PAN is written after a space, `:` or start-of-line, never
    # welded into a hyphen- or pipe-delimited token. Cards written WITH
    # dashes (`4111-1111-1111-1111`) are unaffected: the separators are
    # consumed inside the match, not at its edges.
    ("credit_card", re.compile(r"(?<![0-9A-Za-z._|-])(?:\d[ -]?){13,19}(?![0-9A-Za-z_|-])(?!\.\d)"), _redact_cc_if_luhn),
    # A phone match must carry phone STRUCTURE — a leading `+`, a
    # parenthesised area code, or internal space/dash separators. The old
    # pattern's core (`\d{3}[ -]?\d{4}` with everything else optional)
    # matched any bare 7-10 digit integer, so `LIMIT 1000000` in a stored
    # SQL trajectory became `LIMIT <REDACTED_PHONE>`. Bare unseparated
    # digit runs are now left alone — code corpora are full of them and
    # an unformatted local number is not recoverable PII worth that cost.
    # Boundaries exclude letters, `_` and `-`, not just digits. A bare `\d`
    # boundary let the pattern eat a digit group out of the MIDDLE of a
    # hyphenated identifier: measured in the mined bench pool,
    # `…/renecannao_postgresql-184-1710-16…` became
    # `…/renecannao_postgresql-<REDACTED_PHONE>-16…`, corrupting a URL inside
    # a case whose `fact_swap` fault operates on exactly those digits. A real
    # phone number is not preceded or followed by `-`/`_`/a letter.
    ("phone", re.compile(
        r"(?<![\dA-Za-z_-])(?:"
        # +country, optional area code, separators optional: +1 (212) 555-0123, +306912345678
        r"\+\d{1,3}[ -]?(?:\(?\d{2,4}\)?[ -]?)?\d{3}[ -]?\d{4}"
        # parenthesised area code: (212) 555-0123
        r"|\(\d{2,4}\)[ -]?\d{3}[ -]?\d{4}"
        # separator-delimited groups: 212-555-0123, 30 210 5550123
        r"|(?:\d{1,3}[ -])?\d{2,4}[ -]\d{3}[ -]?\d{4}"
        # 7-digit local WITH separator: 555-0123
        r"|\d{3}[ -]\d{4}"
        r")(?![\dA-Za-z_-])"
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
