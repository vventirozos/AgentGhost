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

# Postgres built-ins that reach outside the database: server-side file I/O,
# large-object import/export, and `COPY … TO/FROM PROGRAM` (command
# execution). Matched on the MASKED statement, so a table or column merely
# NAMED like one of these inside a string literal cannot trip it.
#
# `copy` is matched only in its dangerous forms — `COPY … TO/FROM PROGRAM`
# and `COPY … FROM '/path'` — so ordinary `COPY t FROM STDIN` (the bulk-load
# path a normal task uses) still validates.
_SQL_FS_PRIMITIVE = re.compile(
    r"\b(?:"
    r"pg_read_file|pg_read_binary_file|pg_stat_file|pg_ls_dir|"
    r"pg_ls_logdir|pg_ls_waldir|pg_ls_archive_statusdir|pg_ls_tmpdir|"
    r"pg_ls_logicalsnapdir|pg_ls_logicalmapdir|pg_ls_replslotdir|"
    r"pg_ls_summariesdir|"
    r"lo_import|lo_export"
    r")\b",
    re.IGNORECASE)

# Postgres features that grant code execution or host I/O outright. These
# are not "dangerous if misused" — each is a complete escape on its own, and
# none has a legitimate use from an agent tool:
#
#   * an UNTRUSTED procedural language (the `u` suffix) runs arbitrary code
#     as the server's OS user. `CREATE EXTENSION plperlu` plus a one-line
#     function is host RCE — and on this box `ghost` is a SUPERUSER with
#     `trust` on loopback, so nothing else is in the way;
#   * `ALTER SYSTEM` writes postgresql.auto.conf, and
#     `session_preload_libraries` then loads an attacker .so on the next
#     connection;
#   * `dblink` / the FDWs open outbound connections FROM the database —
#     egress the process-wide socket guard cannot see, because libpq
#     bypasses it;
#   * `pg_file_write` / `pg_file_unlink` / `pg_logdir_ls` (adminpack) are
#     host filesystem writes.
#
# `CREATE EXTENSION` is refused wholesale rather than allow-listed: whether
# an extension is trusted is a property of the installed catalogue, not of
# the statement text, so it cannot be decided here. An operator installs
# extensions by hand.
_SQL_SERVER_ESCAPE = re.compile(
    # Extensions and untrusted procedural languages. `["\s]*` after LANGUAGE
    # because Postgres accepts the quoted identifier `LANGUAGE "plperlu"`,
    # which the unquoted pattern could not see.
    r"\bcreate\s+(?:or\s+replace\s+)?(?:trusted\s+)?extension\b"
    r"|\blanguage\s+[\"\s]*(?:pl)?(?:perl|python|tcl|r|java|sh|v8)\w*u\b"
    # ⚠ `LANGUAGE C` / `LANGUAGE internal` — the reason a NAME deny-list
    # cannot work on its own. `CREATE FUNCTION f(...) AS 'evil.so','f'
    # LANGUAGE C` loads an arbitrary shared object, and
    # `... AS 'pg_read_file_off_len' LANGUAGE internal` RENAMES the exact
    # primitive `_SQL_FS_PRIMITIVE` blocks — after which `SELECT myread(...)`
    # scans clean. Any deny-list of names is one `CREATE FUNCTION` away from
    # irrelevant, so the renaming mechanism itself is refused.
    r"|\blanguage\s+[\"\s]*(?:c|internal)\b"
    # `LOAD '/tmp/evil.so'` is a one-statement shared-object load. Matched
    # as a leading STATEMENT keyword: `_mask_sql` has already blanked the
    # path literal by the time this runs, so there is no quote left to
    # anchor on, and a bare `\bload\b` would refuse a column named `load`.
    r"|\A\s*load\b|;\s*load\b"
    # ALTER SYSTEM writes postgresql.auto.conf; ALTER ROLE/DATABASE ... SET
    # reaches the SAME GUCs (session_preload_libraries) per-role.
    r"|\balter\s+system\b"
    # ⚠ NARROWED to the GUCs that load code. The first version matched any
    # `ALTER ROLE|DATABASE … SET`, which refused every routine migration:
    # `ALTER ROLE app SET search_path = …`, `SET statement_timeout = …`,
    # `ALTER DATABASE db SET timezone = …`. Standard DDL, killed with a
    # security banner that `confirm=true` could not open.
    r"|\balter\s+(?:role|user|database)\b[^;]*\bset\b[^;]*"
    r"(?:preload_libraries|dynamic_library_path)\b"
    # Privilege escalation.
    r"|\b(?:alter|create)\s+(?:role|user)\b[^;]*\bsuperuser\b"
    # The predefined roles are durable capability grants that never say
    # "superuser": `pg_execute_server_program` is command execution,
    # `pg_read_server_files` / `pg_write_server_files` are host file I/O.
    r"|\bpg_(?:execute_server_program|read_server_files|write_server_files)\b"
    # Outbound connections FROM the database — egress the process-wide
    # socket guard cannot see, because libpq bypasses it. Matched as a
    # function CALL, so a table named `dblink_cache` is not an escape.
    r"|\bdblink(?:_connect|_connect_u|_exec|_open|_fetch|_close|"
    r"_send_query|_get_result|_cancel_query)?\s*\("
    r"|\b(?:postgres_fdw|file_fdw)\b"
    # ⚠ SHAPES, NOT NAMES. The docstring above names this class — outbound
    # connections the socket guard cannot see because libpq bypasses it —
    # and the first version blocked `dblink(` plus two wrapper NAMES while
    # leaving the statements that do it open. `CREATE SUBSCRIPTION` is the
    # cleanest: one superuser statement that dials an attacker host at
    # execution time. `dblink_fdw` slipped the function-call anchor, and
    # `CREATE FOREIGN TABLE … OPTIONS (program …)` is host RCE.
    r"|\b(?:create|alter)\s+subscription\b"
    r"|\b(?:create|alter|drop)\s+server\b"
    # ALTER/DROP too: the twins that REPOINT an existing foreign
    # server at an attacker host, or swap its credentials.
    r"|\b(?:create|alter|drop)\s+user\s+mapping\b"
    r"|\bimport\s+foreign\s+schema\b"
    r"|\bforeign\s+data\s+wrapper\b"
    r"|\b(?:create|alter)\s+foreign\s+table\b"
    # adminpack host filesystem writes.
    r"|\bpg_file_write\b|\bpg_file_unlink\b|\bpg_file_rename\b"
    r"|\bpg_logdir_ls\b|\bpg_reload_conf\b",
    re.IGNORECASE)

# Dynamic SQL inside a function body defeats every static scan:
#   DO $$ BEGIN EXECUTE 'pg_read' || '_file(''/etc/passwd'')'; END $$
# The string is assembled at run time, so no pattern above can see it. A
# deny-list cannot win that argument; the construction is refused instead.
_SQL_DYNAMIC_IN_BODY = re.compile(r"\bexecute\s+(?!immediate\b)", re.IGNORECASE)

# ⚠ ANCHORED. Matching `\bcopy\b` anywhere refused
# `CREATE TABLE audit (id serial, copy text)` and
# `SELECT copy FROM ledger` — a column legitimately named `copy`. COPY is a
# STATEMENT, so it can only lead one.
_SQL_COPY = re.compile(r"\A\s*copy\b", re.IGNORECASE)
_SQL_COPY_PROGRAM = re.compile(r"\bprogram\b", re.IGNORECASE)
# `COPY … TO PROGRAM` / `… FROM PROGRAM`, wherever it appears.
_SQL_COPY_PROGRAM_FORM = re.compile(
    # `COPY` must START a statement or a plpgsql body clause — otherwise
    # `SELECT copy FROM program` matched, which is the same false positive
    # the `\A\s*copy` anchor was added to fix, one rule over.
    r"(?:\A|;|\$\$|\$[A-Za-z_]\w*\$|\bbegin\b|\bthen\b|\bloop\b|\belse\b)\s*"
    r"copy\b[^;]*?\b(?:to|from)\s+program\b",
    re.IGNORECASE | re.DOTALL)
# The only two endpoints of COPY that stay inside the client connection.
_SQL_COPY_SAFE_ENDPOINT = re.compile(r"\b(?:stdin|stdout)\b", re.IGNORECASE)


def _copy_reaches_the_host(masked: str) -> bool:
    """True when a COPY statement's endpoint is a host file or a command.

    ⚠ AN ALLOW-LIST, NOT A DENY-LIST, and it has to be. `_mask_sql` blanks
    string literals before any guard runs, so `COPY t FROM '/etc/passwd'`
    arrives here as `COPY t FROM` plus spaces — there is no path left to
    pattern-match against, and the first attempt at this rule (matching
    `from\s+'`) passed it clean. What survives masking is the SAFE spelling:
    `STDIN` / `STDOUT` are bare keywords. So a COPY is refused unless it
    names one of those, which also catches every file path without needing
    to see it.

    Checked at statement level rather than on the `TO`/`FROM` keyword,
    because `COPY (SELECT x FROM t) TO STDOUT` has an inner FROM that a
    keyword-anchored rule reads as the endpoint.
    """
    # ⚠ TWO RULES, because the two dangerous forms need different anchors.
    #
    # `COPY … TO/FROM PROGRAM` is unambiguous ANYWHERE: no column named
    # `copy` is ever followed by `TO PROGRAM`. It must not be anchored to a
    # statement head, because anchoring it is exactly what re-opened
    # `DO $$ COPY t TO PROGRAM 'id'; $$` when the head-anchor was added to
    # fix the `SELECT copy FROM ledger` false positive.
    if _SQL_COPY_PROGRAM_FORM.search(masked):
        return True
    # The file form (`COPY t FROM '/etc/passwd'`) survives masking as
    # `COPY t FROM` plus spaces — indistinguishable from `SELECT copy FROM
    # t` unless COPY leads the statement, which as a STATEMENT it always
    # does. (Inside a plpgsql body a bare COPY is not valid anyway; it would
    # have to go through EXECUTE, which is refused as dynamic SQL.)
    if not any(_SQL_COPY.match(part.strip())
               for part in masked.split(";")):
        return False
    # (A `\bprogram\b` search anywhere in a COPY statement used to live
    # here. It was redundant — `_SQL_COPY_PROGRAM_FORM` and the
    # STDIN/STDOUT allow-list cover every real form — and it refused
    # `COPY courses (id, program) FROM STDIN`, a column legitimately named
    # `program`. Mutation-tested: neutering it changed no test result.)
    return not _SQL_COPY_SAFE_ENDPOINT.search(masked)
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


def _mask_sql(s: str, *, keep_dollar: bool = False) -> Tuple[str, dict]:
    """Blank out everything that is DATA rather than statement structure —
    single-quoted literals, ``--`` line comments, ``/* */`` block comments
    (PostgreSQL nests them) and dollar-quoted bodies — replacing each with
    spaces so character offsets are preserved.

    Returns ``(masked, flags)``. Structural checks (paren balance, statement
    splitting, destructive-verb detection) run on the MASKED text so neither
    a keyword inside a string nor a semicolon inside a comment can fool them.

    ``keep_dollar=True`` leaves dollar-quoted bodies VERBATIM while still
    masking literals and comments. A dollar body is executable CODE, not
    data, so blanking it hides exactly what the host-primitive guards need
    to see: ``DO $$ BEGIN PERFORM pg_read_file('/etc/passwd'); END $$``
    masks down to ``DO`` plus whitespace and validated clean under
    ``confirm=true``. Scanning the RAW statement instead would over-refuse
    on any literal containing a keyword (``'create extension is just
    text'``), so the guards get this middle form: literals masked, code
    visible. §4DX round 2.
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
                    out.append(s[i:n] if keep_dollar else " " * (n - i))
                    i = n
                    continue
                j = end + len(tag)
                if keep_dollar:
                    # ⚠ MASK THE BODY'S OWN LITERALS AND COMMENTS.
                    #
                    # Copying the body verbatim made the probe commit the
                    # very sin `keep_dollar` exists to avoid, one nesting
                    # level down: a plpgsql function whose body merely
                    # MENTIONS `copy`, `create extension`, `dblink` or a
                    # `pg_*` name in a comment or a RAISE NOTICE string was
                    # refused unconditionally, and `confirm=true` could not
                    # open it. The top-level pin missed it because the
                    # top-level literal is still masked correctly.
                    #
                    # Recursing keeps the body's CODE visible (so a real
                    # `pg_read_file` call inside a DO block is still seen)
                    # while blanking its data. `_mask_sql` preserves
                    # character offsets, so the recursion cannot shift them.
                    _body = s[m.end():end]
                    out.append(tag)
                    out.append(_mask_sql(_body, keep_dollar=True)[0])
                    out.append(tag)
                else:
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

    # ⚠ SANDBOX ESCAPE — NEVER GATED BY `confirm`.
    #
    # `postgres_admin` runs IN THE AGENT PROCESS, on the host, outside the
    # Docker sandbox and outside every `_get_safe_path` root. The `ghost`
    # role is a Postgres SUPERUSER and `pg_hba.conf` grants `trust` on
    # loopback TCP, so these built-ins are the whole filesystem and a shell:
    #
    #   SELECT pg_read_file('~/Data/AI/.ghost_api_key')  -> the 0600 master key
    #   COPY (SELECT 1) TO PROGRAM 'sh -c ...'           -> host RCE as the user
    #
    # Both were demonstrated end-to-end through the real tool on 2026-08-29.
    # Every one of these validated CLEAN before this block existed: the guard
    # above stops `DROP TABLE` — the clumsy destructive thing — and passed
    # the complete escape.
    #
    # Unlike DROP/TRUNCATE this is NOT a confirmable operation. `confirm` is
    # the model's own flag; it exists so a deliberate destructive DDL can
    # proceed, not so file reads and command execution can be opted into. A
    # legitimate schema task never needs any of these.
    #
    # ⚠ THIS IS THE SECOND LAYER, NOT THE FIX. It closes the agent's own tool
    # path. It does NOT stop a compromised sandbox container connecting
    # straight to `host.docker.internal:5432`, which `trust` + superuser
    # accepts with no validator in the way. That needs the role dropped from
    # superuser and `trust` replaced with `scram-sha-256` — host config, and
    # the operator's call.
    # ⚠ SCANNED ON THE DOLLAR-VISIBLE PROBE TOO. See `_mask_sql`'s
    # `keep_dollar`: a DO block or function body is executable code the
    # normal mask blanks, so a host primitive hidden inside one validated
    # clean whenever the model set `confirm=true` — a flag the MODEL sets.
    _probe, _ = _mask_sql(s, keep_dollar=True)
    _m = _SQL_FS_PRIMITIVE.search(masked) or _SQL_FS_PRIMITIVE.search(_probe)
    if _m:
        return False, (
            f"'{_m.group(0)}' reads or writes the HOST filesystem through the "
            "database server, outside every sandbox. Refused unconditionally "
            "(confirm=true does not enable it)."
        )
    if _copy_reaches_the_host(masked) or _copy_reaches_the_host(_probe):
        return False, (
            "COPY to/from a host FILE or PROGRAM runs outside every sandbox. "
            "Only COPY ... FROM STDIN / TO STDOUT is allowed. Refused "
            "unconditionally (confirm=true does not enable it)."
        )
    if flags["has_dollar_body"] and _SQL_DYNAMIC_IN_BODY.search(_probe):
        return False, (
            "dynamic SQL (EXECUTE) inside a function or DO body cannot be "
            "checked statically — the statement is assembled at run time, so "
            "`EXECUTE 'pg_read' || '_file(...)'` defeats every guard here. "
            "Refused unconditionally. Write the statement literally instead."
        )
    _esc = (_SQL_SERVER_ESCAPE.search(masked)
            or _SQL_SERVER_ESCAPE.search(_probe))
    if _esc:
        return False, (
            f"'{_esc.group(0)}' grants code execution or host I/O through the "
            "database server (untrusted procedural language, ALTER SYSTEM, "
            "dblink/FDW outbound connection, or adminpack file write). "
            "Refused unconditionally — confirm=true does not enable it. Ask "
            "the operator if an extension is genuinely needed."
        )

    return True, ""


__all__ = ["validate_shell", "validate_sql"]
