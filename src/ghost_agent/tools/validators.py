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
    # rm -rf / and friends
    re.compile(r"\brm\b\s+(?:-[A-Za-z]*r[A-Za-z]*f|-[A-Za-z]*f[A-Za-z]*r)\b\s+/(?:\s|$)"),
    re.compile(r"\brm\b\s+-rf\s+/(?:\s|$)"),
    re.compile(r"\brm\b\s+(?:-[A-Za-z]*r[A-Za-z]*f|-[A-Za-z]*f[A-Za-z]*r)\b\s+~(?:\s|$)"),
    # dd to a raw device
    re.compile(r"\bdd\b[^|;&]*of=/dev/(?:sd|nvme|hd)"),
    # mkfs / fdisk / shred against a device
    re.compile(r"\bmkfs(?:\.[a-z0-9]+)?\b\s+/dev/"),
    re.compile(r"\bshred\b\s+/dev/"),
    # Fork bomb
    re.compile(r":\(\)\s*\{\s*:\|\s*:&\s*\}\s*;\s*:"),
    # Chmod 777 on root / system dirs
    re.compile(r"\bchmod\b\s+(?:-R\s+)?(?:0?777|a\+rwx)\b\s+/(?:bin|etc|usr|sys|root)\b"),
    # Curl|sh pattern — common malware delivery shape
    re.compile(r"\bcurl\b[^|]+\|\s*(?:sh|bash|zsh)\b"),
    re.compile(r"\bwget\b[^|]+\|\s*(?:sh|bash|zsh)\b"),
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

# Statement keywords that we treat as DESTRUCTIVE and require an explicit
# WHERE clause (or, for DROP/TRUNCATE, an explicit confirmation flag).
_SQL_UNGUARDED_DELETE = re.compile(
    r"^\s*delete\s+from\s+\w+\s*(?:returning|;|\s*$)", re.IGNORECASE)
_SQL_UNGUARDED_UPDATE = re.compile(
    r"^\s*update\s+\w+\s+set\s+[^;]*?(?:;|\s*$)", re.IGNORECASE)
_SQL_DROP = re.compile(r"^\s*drop\s+(?:table|schema|database|view|index)\b",
                       re.IGNORECASE)
_SQL_TRUNCATE = re.compile(r"^\s*truncate\b", re.IGNORECASE)
# Lightweight statement-shape checks — catch unbalanced quotes/parens.
_SQL_SINGLE_QUOTE = "'"


def validate_sql(stmt: str) -> Tuple[bool, str]:
    """Validate a SQL statement. Returns ``(ok, reason)``.

    Tries to use ``sqlparse`` for tokenisation when available; falls
    back to regex shape checks. Either path rejects unguarded
    DELETE/UPDATE, raw DROP/TRUNCATE, and unbalanced quotes/parens.
    """
    if not stmt or not stmt.strip():
        return False, "empty statement"
    s = stmt.strip()

    # Quote / paren balance — cheap shape check that catches most LLM
    # token-truncation failures before they reach the DB.
    if s.count(_SQL_SINGLE_QUOTE) % 2 != 0:
        # Allow doubled single-quotes (SQL escape) by stripping them first
        if s.replace("''", "").count(_SQL_SINGLE_QUOTE) % 2 != 0:
            return False, "unbalanced single quotes"
    if s.count("(") != s.count(")"):
        return False, "unbalanced parentheses"

    # Destructive-statement guard.
    if _SQL_UNGUARDED_DELETE.match(s):
        return False, "DELETE without WHERE clause"
    if _SQL_UNGUARDED_UPDATE.match(s):
        # Only block when there is no WHERE anywhere in the statement.
        if not re.search(r"\bwhere\b", s, re.IGNORECASE):
            return False, "UPDATE without WHERE clause"
    if _SQL_DROP.match(s):
        return False, "DROP statement requires explicit confirmation"
    if _SQL_TRUNCATE.match(s):
        return False, "TRUNCATE statement requires explicit confirmation"

    # Try sqlparse for a deeper parse if installed.
    try:
        import sqlparse  # type: ignore
        parsed = sqlparse.parse(s)
        if not parsed:
            return False, "sqlparse returned no statements"
        # Reject multi-statement bundles unless the caller explicitly
        # asked for them — common when the LLM stitches several lines
        # and the DB driver runs them as one transaction.
        non_empty = [p for p in parsed if str(p).strip()]
        if len(non_empty) > 1:
            # Multi-statement is okay only if each individual statement
            # already passes the same guards.
            for p in non_empty:
                ok, reason = validate_sql(str(p))
                if not ok:
                    return False, f"multi-stmt: {reason}"
    except ImportError:
        # sqlparse not installed — fall through with regex-only checks.
        pass
    except Exception as exc:
        logger.debug("sqlparse failed in validate_sql: %s", exc)

    return True, ""


__all__ = ["validate_shell", "validate_sql"]
