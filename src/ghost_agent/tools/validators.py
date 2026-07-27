"""Pre-execution validators for shell + SQL — roadmap phase 1.4.

These run BEFORE a candidate command/statement is dispatched to its
executor. The goal is to catch shape errors (unclosed quotes, missing
semicolons in DDL, obviously-destructive forms) at a phase where the
agent's prompt history can still be enriched with a diagnostic, rather
than after a tool call has corrupted host state.

Contract for every validator:
    ``validate_X(text: str) -> (ok: bool, reason: str)``

Validators are deliberately CONSERVATIVE — they false-negative
(reject safe statements that happen to look risky) rather than
false-positive. A rejection is a hint to the planner, not a final
verdict; the agent can re-emit with a clarification or ask the user.

Validators never raise. A validator that itself crashes returns
``(True, "validator-error: <type>")`` so the bug doesn't break a
production turn. The bug surfaces in logs at debug level.
"""

from __future__ import annotations

import logging
import re
import shlex
from typing import Tuple

logger = logging.getLogger("GhostAgent")


# ──────────────────────────────────────────────────────────────────────
# Shell validator
# ──────────────────────────────────────────────────────────────────────

# Patterns that we flat-out refuse to dispatch. These are the
# canonical "you almost certainly do not want this" forms — exotic
# delete-the-whole-disk variants. Anchored at word boundaries so a
# benign substring ("description") doesn't trip the deny-list.
_SHELL_DENY: tuple = (
    # rm with BOTH -r and -f flags (any order/spelling, combined or split)
    # targeting a DANGEROUS path: absolute (/...), root glob (/*), home (~ /
    # $HOME), or a quoted form of those. Relative deletes (rm -rf ./build)
    # are intentionally NOT blocked. The two lookaheads require an `r` flag
    # and an `f` flag in the rm invocation before the target.
    re.compile(
        r"\brm\b(?=[^|;&]*\s-\w*r)(?=[^|;&]*\s-\w*f)[^|;&]*\s+"
        r"(?:/(?:\s|$|\*|\w[^\s'\"]*)|~(?:\s|$|/)|\$\{?HOME\}?"
        r"|['\"]\s*(?:/[^'\"]*|~[^'\"]*|\$\{?HOME\}?)['\"])",
        re.IGNORECASE,
    ),
    # dd to a raw device
    re.compile(r"\bdd\b[^|;&]*of=/dev/(?:sd|nvme|hd)"),
    # mkfs / fdisk / shred against a device
    re.compile(r"\bmkfs(?:\.[a-z0-9]+)?\b\s+/dev/"),
    re.compile(r"\bshred\b\s+/dev/"),
    # Fork bomb
    re.compile(r":\(\)\s*\{\s*:\|\s*:&\s*\}\s*;\s*:"),
    # Chmod 777 on root / system dirs
    re.compile(r"\bchmod\b\s+(?:-R\s+)?(?:0?777|a\+rwx)\b\s+/(?:bin|etc|usr|sys|root)\b"),
    # Download piped straight into an interpreter — common malware shape.
    # SHELLS stay fully blocked (curl | sh, | bash -s, …). Scripting
    # interpreters block only when the pipe feeds them AS THE PROGRAM —
    # bare (`| python3`) or explicit stdin (`| python3 -`). With `-m mod`,
    # `-c code`, or a script file the piped bytes are DATA, not code:
    # `curl … | python3 -m json.tool` is a legitimate verification pattern
    # the old blanket rule false-positived on (2026-07-25 audit: the block
    # made the agent substitute a WEAKER check instead of pretty-printing
    # its POST result).
    re.compile(
        r"\b(?:curl|wget|fetch)\b[^|]+\|\s*(?:sudo\s+)?"
        r"(?:sh|bash|zsh|dash|ksh)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:curl|wget|fetch)\b[^|]+\|\s*(?:sudo\s+)?"
        r"(?:python[0-9.]*|perl|ruby|node|php)\s*(?:-\s*)?(?:\||;|&|$)",
        re.IGNORECASE,
    ),
)


def validate_shell(cmd: str) -> Tuple[bool, str]:
    """Validate a shell command's shape and reject obviously-destructive
    forms. Returns ``(ok, reason)``.

    ``ok=True`` means "shape looks plausible, not on the deny list"
    — NOT "this command is safe in your environment". The host-level
    sandbox is still the authoritative safety boundary.
    """
    if not cmd or not cmd.strip():
        return False, "empty command"
    s = cmd.strip()
    # Shape check: must shlex-parse. Unclosed quotes are the most common
    # LLM emission bug ("echo 'hello world").
    try:
        tokens = shlex.split(s, posix=True)
    except ValueError as e:
        return False, f"shell syntax: {e}"
    if not tokens:
        return False, "empty after parsing"
    # Deny list
    for pat in _SHELL_DENY:
        if pat.search(s):
            return False, f"deny-listed pattern: {pat.pattern[:60]}"
    return True, ""


# ──────────────────────────────────────────────────────────────────────
# SQL validator
# ──────────────────────────────────────────────────────────────────────

# A (possibly schema-qualified, possibly quoted) table reference:
#   users | public.users | "users" | "public"."users" | `users`
# Using a bare `\w+` here silently exempted schema-qualified and quoted
# targets from the destructive-statement guards below (DELETE FROM
# public.users with no WHERE passed validation).
_SQL_TABLE = (
    r'(?:"[^"]+"|`[^`]+`|\w+)'
    r'(?:\s*\.\s*(?:"[^"]+"|`[^`]+`|\w+))*'
)

# Statement keywords that we treat as DESTRUCTIVE and require an explicit
# WHERE clause (or, for DROP/TRUNCATE, an explicit confirmation flag).
#
# These are deliberately NOT anchored at the start of the statement. The
# anchored versions were bypassed by anything that put the verb elsewhere:
#   WITH d AS (DELETE FROM t RETURNING *) SELECT count(*) FROM d
#   /* comment */ DROP TABLE t
# Both were live holes (2026-07-27 audit). Matching is done against a MASKED
# statement (literals / comments / dollar-quoted bodies blanked) so a keyword
# inside a string can never trip them.
_SQL_UNGUARDED_DELETE = re.compile(
    r"\bdelete\s+from\s+" + _SQL_TABLE, re.IGNORECASE)
_SQL_UNGUARDED_UPDATE = re.compile(
    r"\bupdate\s+" + _SQL_TABLE + r"\s+set\b", re.IGNORECASE)
_SQL_DROP = re.compile(r"\bdrop\s+(?:table|schema|database|view|index)\b",
                       re.IGNORECASE)
_SQL_TRUNCATE = re.compile(r"\btruncate\b", re.IGNORECASE)
# Lightweight statement-shape checks — catch unbalanced quotes/parens.
_SQL_SINGLE_QUOTE = "'"
_SQL_INNER_PARENS = re.compile(r"\([^()]*\)")
# Opening marker of a dollar-quoted body: $$ or $tag$ (PostgreSQL).
_SQL_DOLLAR_OPEN = re.compile(r"\$([A-Za-z_]\w*)?\$")


def _strip_sql_parens(s: str) -> str:
    """Remove balanced parenthesised groups (subqueries, expressions),
    innermost-first, so a WHERE that lives only inside a subquery does not
    count as the statement's own WHERE clause."""
    prev = None
    while prev != s:
        prev = s
        s = _SQL_INNER_PARENS.sub(" ", s)
    return s


def _mask_sql(s: str) -> Tuple[str, dict]:
    """Blank out everything that is DATA rather than statement structure —
    single-quoted literals, ``--`` line comments, ``/* */`` block comments
    (PostgreSQL nests them) and dollar-quoted bodies — replacing each with
    spaces so character offsets are preserved.

    Returns ``(masked, flags)``. Structural checks (paren balance, statement
    splitting, destructive-verb detection) run on the MASKED text so neither
    a keyword inside a string nor a semicolon inside a comment can fool them.
    """
    out = []
    flags = {"unterminated_quote": False, "unterminated_comment": False,
             "unterminated_dollar": False, "has_dollar_body": False}
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == "'":
            j = i + 1
            closed = False
            while j < n:
                if s[j] == "'":
                    if j + 1 < n and s[j + 1] == "'":   # '' escape
                        j += 2
                        continue
                    closed = True
                    j += 1
                    break
                j += 1
            if not closed:
                flags["unterminated_quote"] = True
            out.append(" " * (j - i))
            i = j
            continue
        if c == "-" and i + 1 < n and s[i + 1] == "-":
            j = s.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
            continue
        if c == "/" and i + 1 < n and s[i + 1] == "*":
            depth, j = 1, i + 2
            while j < n and depth:
                if s.startswith("/*", j):
                    depth += 1
                    j += 2
                elif s.startswith("*/", j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            if depth:
                flags["unterminated_comment"] = True
            out.append(" " * (j - i))
            i = j
            continue
        if c == "$":
            m = _SQL_DOLLAR_OPEN.match(s, i)
            if m:
                tag = m.group(0)
                end = s.find(tag, m.end())
                flags["has_dollar_body"] = True
                if end < 0:
                    flags["unterminated_dollar"] = True
                    out.append(" " * (n - i))
                    i = n
                    continue
                j = end + len(tag)
                out.append(" " * (j - i))
                i = j
                continue
        out.append(c)
        i += 1
    return "".join(out), flags


def _split_sql_statements(masked: str, original: str) -> list:
    """Split on top-level ``;`` using the MASKED text for offsets, returning
    the ORIGINAL slices. Semicolons inside literals/comments/dollar bodies
    are already blanked, so they cannot split a statement."""
    parts, start = [], 0
    for idx, ch in enumerate(masked):
        if ch == ";":
            seg = original[start:idx]
            if seg.strip():
                parts.append(seg)
            start = idx + 1
    tail = original[start:]
    if tail.strip():
        parts.append(tail)
    return parts


def _scope_slice(masked: str, start: int) -> str:
    """Text from ``start`` to the end of its ENCLOSING paren group (or the
    statement end) — so a CTE-wrapped ``DELETE`` is tested for a WHERE
    within its own parenthesised body, not the whole statement."""
    depth = 0
    for i in range(start, len(masked)):
        ch = masked[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                return masked[start:i]
            depth -= 1
    return masked[start:]


def validate_sql(stmt: str, confirm: bool = False) -> Tuple[bool, str]:
    """Validate a SQL statement. Returns ``(ok, reason)``.

    Rejects unguarded DELETE/UPDATE, raw DROP/TRUNCATE, unbalanced
    quotes/parens, and multi-statement bundles whose parts don't each pass
    the same guards.

    Self-contained by design: this used to try ``sqlparse`` for statement
    splitting and fall through to `^`-anchored regexes when the import
    failed. ``sqlparse`` was never a declared dependency and is NOT
    installed, so the multi-statement guard was inoperative in production —
    ``SELECT 1; DROP TABLE t`` validated clean and psycopg2 ran the whole
    batch (2026-07-27 audit, verified live). Splitting is now done here,
    against a masked copy of the statement, with no optional import.

    ``confirm=True`` skips the DROP/TRUNCATE block — the caller has
    explicitly acknowledged a destructive DDL (see ``postgres_admin``'s
    ``confirm`` parameter).
    """
    if not stmt or not stmt.strip():
        return False, "empty statement"
    s = stmt.strip()

    masked, flags = _mask_sql(s)

    # Quote / paren balance — cheap shape check that catches most LLM
    # token-truncation failures before they reach the DB.
    if flags["unterminated_quote"]:
        return False, "unbalanced single quotes"
    if flags["unterminated_comment"]:
        return False, "unterminated block comment"
    if flags["unterminated_dollar"]:
        return False, "unterminated dollar-quoted string"
    # Parens counted on the MASKED text — a valid `WHERE note = 'a)'` must
    # not be rejected as unbalanced.
    if masked.count("(") != masked.count(")"):
        return False, "unbalanced parentheses"

    # Multi-statement bundles: each part must pass the same guards on its
    # own. Fail-closed and independent of any optional parser.
    parts = _split_sql_statements(masked, s)
    if len(parts) > 1:
        for p in parts:
            ok, reason = validate_sql(p, confirm=confirm)
            if not ok:
                return False, f"multi-stmt: {reason}"
        return True, ""
    if not parts:
        return False, "empty statement"

    # A dollar-quoted body (DO $$…$$, function bodies) can execute arbitrary
    # dynamic SQL that no static check can see through — treat as
    # unanalysable and require explicit confirmation.
    if flags["has_dollar_body"] and not confirm:
        return False, ("dollar-quoted body (DO block / function body) cannot be "
                       "statically checked — requires confirm=true")

    # Destructive-statement guards, matched ANYWHERE in the masked statement
    # (see the pattern definitions for why anchoring was unsafe). The WHERE
    # test is scoped to the match's own paren group so a CTE-wrapped DELETE
    # is judged on its own body.
    _m = _SQL_UNGUARDED_DELETE.search(masked)
    if _m:
        scope = _strip_sql_parens(_scope_slice(masked, _m.end()))
        if not re.search(r"\bwhere\b", scope, re.IGNORECASE):
            return False, "DELETE without WHERE clause"
    _m = _SQL_UNGUARDED_UPDATE.search(masked)
    if _m:
        # Only block when there is no WHERE at the STATEMENT level: neither a
        # WHERE inside a string ('no where here', already masked) nor one
        # inside a subquery (SET x=(SELECT … WHERE …)) counts as the outer one.
        scope = _strip_sql_parens(_scope_slice(masked, _m.end()))
        if not re.search(r"\bwhere\b", scope, re.IGNORECASE):
            return False, "UPDATE without WHERE clause"
    if not confirm:
        if _SQL_DROP.search(masked):
            return False, "DROP statement requires confirm=true"
        if _SQL_TRUNCATE.search(masked):
            return False, "TRUNCATE statement requires confirm=true"

    return True, ""


__all__ = ["validate_shell", "validate_sql"]
