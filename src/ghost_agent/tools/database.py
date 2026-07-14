import asyncio
import logging
import threading
from typing import Optional
from ..utils.logging import Icons, pretty_log
from ..utils.sanitizer import extract_code_from_markdown

logger = logging.getLogger("GhostAgent")

# Simple connection pool: cache one connection per URI to avoid
# opening/closing on every call during multi-step SQL workflows.
#
# A psycopg2 connection is NOT safe for concurrent use by multiple
# threads, yet tool calls are dispatched concurrently via
# `asyncio.to_thread` (see core/agent.py's parallel tool gather). Two
# `postgres_admin` calls against the same URI would otherwise share one
# connection across two worker threads and corrupt the protocol stream
# ("lost synchronization with server"). We therefore serialize all use
# of a given URI's connection behind a per-URI lock.
_connection_pool: dict = {}
_conn_locks: dict = {}
_pool_lock = threading.Lock()  # guards the two module dicts above


def _get_uri_lock(connection_string: str) -> threading.Lock:
    """Return the per-URI lock serializing use of that URI's connection."""
    with _pool_lock:
        lk = _conn_locks.get(connection_string)
        if lk is None:
            lk = threading.Lock()
            _conn_locks[connection_string] = lk
        return lk


def _evict_connection(connection_string: str):
    """Drop the cached connection for a URI (after a connection error)."""
    with _pool_lock:
        _connection_pool.pop(connection_string, None)


def _get_connection(connection_string: str):
    """Get or create a cached connection for the given URI.

    The caller MUST hold the per-URI lock from `_get_uri_lock` — this
    function reads/writes the shared connection for the URI.
    """
    import psycopg2
    with _pool_lock:
        conn = _connection_pool.get(connection_string)
    if conn is not None:
        try:
            # Check if connection is still alive
            closed = getattr(conn, 'closed', 0)
            if closed:
                conn = None
        except Exception:
            conn = None
    if conn is None:
        conn = psycopg2.connect(connection_string)
        conn.autocommit = True
        with _pool_lock:
            _connection_pool[connection_string] = conn
    return conn


async def tool_postgres_admin(action: str = None, connection_string: Optional[str] = None, query: Optional[str] = None, table_name: Optional[str] = None, default_uri: Optional[str] = None, timeout_ms: Optional[int] = None, confirm: bool = False, **kwargs):
    if not action:
        return "SYSTEM ERROR: The 'action' parameter is MANDATORY. You must specify it."
    pretty_log("Postgres Admin", f"Action: {action}", icon=Icons.POSTGRES)
    try:
        import psycopg2
        import psycopg2.extras
        from tabulate import tabulate
    except ImportError:
        return "Error: psycopg2 or tabulate library is missing."

    _supplied_conn = connection_string  # what the LLM actually passed (may be None)
    connection_string = connection_string or default_uri
    if not connection_string:
        return "Error: connection_string is required and no default is configured. Ask the user for the DB URI."

    # Host restriction (SSRF / internal-DB-probe guard): when a default URI
    # is configured, the LLM may NOT redirect the connection to a DIFFERENT
    # host. Only honor an LLM-supplied connection_string whose target matches
    # the configured default. (With no default configured the operator has
    # explicitly opted into arbitrary URIs.)
    #
    # The comparison key resolves libpq's URI-QUERY-STRING overrides, not just
    # the netloc: `?hostaddr=` is the ACTUAL TCP target (the netloc host becomes
    # mere TLS SNI), while `?host=`/`?port=`/`?dbname=` also override the netloc.
    # A key built from netloc alone saw only the innocent host and passed — so
    # `…/prod?hostaddr=10.0.0.99` connected to 10.0.0.99 while reading as
    # db.internal (verified 2026-07-14). Parsed with urllib (urlparse + parse_qs)
    # DIRECTLY — deliberately NOT psycopg2's parse_dsn, so the guard stays
    # deterministic even where a test double or a partial psycopg2 install makes
    # parse_dsn unavailable (a mocked parse_dsn returning a non-dict would have
    # silently reopened the bypass).
    if default_uri and _supplied_conn and _supplied_conn != default_uri:
        from urllib.parse import urlparse, parse_qs

        def _dsn_target(uri):
            p = urlparse(uri)
            q = parse_qs(p.query)

            def _last(key):
                vals = q.get(key)
                return vals[-1] if vals else ""  # libpq: last duplicate wins
            # hostaddr (literal IP endpoint) wins over host, query-host over
            # netloc-host — matches libpq's endpoint precedence. `p.port`
            # raises ValueError on a non-numeric port; that propagates to the
            # outer handler as a formatted parse error rather than crashing
            # the tool.
            host = (_last("hostaddr") or _last("host") or (p.hostname or "")).lower()
            port = str(_last("port") or (p.port or 5432))
            dbname = _last("dbname") or (p.path or "").lstrip("/")
            return (host, port, dbname)
        try:
            _sup_key, _def_key = _dsn_target(_supplied_conn), _dsn_target(default_uri)
        except Exception:
            return "Error: could not parse the supplied connection_string."
        if _sup_key != _def_key:
            return (
                f"Error: refused connection to {_sup_key[0]}:{_sup_key[1]}/{_sup_key[2]!r}; "
                f"only the configured database "
                f"{_def_key[0]}:{_def_key[1]}/{_def_key[2]!r} is allowed. Omit "
                f"connection_string to use the default. (Note: host/hostaddr/"
                f"port/dbname in the URI query string are resolved and checked "
                f"too — they cannot be used to redirect the connection.)"
            )

    if query:
        query = extract_code_from_markdown(query)

    # Pre-execution SQL validator (roadmap phase 1.4). Runs UNCONDITIONALLY
    # — the destructive-statement guard (unguarded DELETE/UPDATE, raw
    # DROP/TRUNCATE, multi-statement, unbalanced quotes) is a safety boundary
    # that must not depend on the optional ``--enable-metacog`` uplift. Extra
    # metacog telemetry is emitted only when the bundle is wired. We validate
    # only `query`/`explain_analyze`; `schema`/`activity` use hand-crafted SQL
    # that never sees user content. `confirm=true` allows DROP/TRUNCATE.
    # `confirm` gates DROP/TRUNCATE. Coerce like a truthy STRING, not bool():
    # tool args arrive as strings (see the timeout_ms note below), and
    # bool("false") / bool("0") / bool("no") are all True — so an explicit
    # confirm="false" would AUTHORIZE a destructive statement. Only the
    # affirmative tokens count.
    _confirmed = (confirm is True
                  or str(confirm).strip().lower() in ("true", "1", "yes", "y"))
    _metacog = kwargs.get("_metacog_bundle")
    if query and action in ("query", "explain_analyze"):
        try:
            from .validators import validate_sql
            ok, reason = validate_sql(query, confirm=_confirmed)
        except Exception as _vexc:
            # Fails OPEN (a validator bug must not brick a mostly-read tool)
            # but at WARNING, not debug: this disables the destructive-
            # statement guard, so the operator (who monitors the live stream)
            # must SEE that it happened rather than have it buried.
            logger.warning("SQL validator crashed — destructive-statement "
                           "guard DISABLED for this call: %s", _vexc)
            ok, reason = True, ""
        if not ok:
            if _metacog is not None and getattr(_metacog, "enabled", False):
                try:
                    from ..core.metacog_log import (
                        emit as _mc_emit, Subsystem as _mc_ss, LEVEL_WARN,
                    )
                    _mc_emit(
                        _mc_ss.VALID, level=LEVEL_WARN,
                        verdict="block", tool="sql",
                        action=action, reason=reason, sql_head=query[:60],
                    )
                    _metacog.count(validator_block=True)
                except Exception:
                    pass
            return (
                f"SYSTEM BLOCK: SQL statement rejected by pre-execution "
                f"validator: {reason}. The query was not run. Re-emit with a "
                f"WHERE clause, or pass confirm=true if you genuinely need to "
                f"DROP/TRUNCATE."
            )

    # Default timeout: 15s, but allow override for complex queries.
    # `timeout_ms` arrives from LLM tool-args (frequently as a string)
    # and used to be interpolated straight into a SET statement — both a
    # crash risk and a SQL-injection sink (e.g. "0; DROP TABLE x--" on an
    # autocommit connection, where multi-statement batches execute). We
    # coerce to int and clamp to a sane band so it can only ever be a
    # bare integer; the SET is parameterized too as defense in depth.
    try:
        effective_timeout = int(timeout_ms) if timeout_ms else 15000
    except (TypeError, ValueError):
        effective_timeout = 15000
    effective_timeout = max(100, min(effective_timeout, 600000))

    def _run_action(cur):
        # Statement timeout is SESSION-scoped on the cached autocommit
        # connection, so a prior call's `SET statement_timeout=100` would
        # leak into THIS call and cause spurious cancellations on a healthy
        # DB. Set it at the top of every action (not just query) so each
        # action runs under its own bound. Parameterized (int-clamped above).
        cur.execute("SET statement_timeout = %s", (effective_timeout,))
        if action == "schema":
            # Parameterized — table_name comes from an LLM and could
            # otherwise carry an injection payload ("'; DROP TABLE--").
            # Bounded + byte-capped like `query`: a wide DB (hundreds of
            # tables) would otherwise flood the model context with every
            # column of every public table, unbounded.
            _SCHEMA_ROW_CAP = 1000
            sql = ("SELECT table_name, column_name, data_type "
                   "FROM information_schema.columns WHERE table_schema = 'public' "
                   "ORDER BY table_name, ordinal_position")
            if table_name:
                sql = ("SELECT table_name, column_name, data_type "
                       "FROM information_schema.columns "
                       "WHERE table_schema = 'public' AND table_name = %s "
                       "ORDER BY ordinal_position")
                cur.execute(sql, (table_name,))
            else:
                cur.execute(sql)
            rows = cur.fetchmany(_SCHEMA_ROW_CAP + 1)
            if not rows: return "No schema found."
            _schema_truncated = len(rows) > _SCHEMA_ROW_CAP
            out = tabulate(rows[:_SCHEMA_ROW_CAP], headers="keys", tablefmt="pipe")
            if _schema_truncated:
                out += (f"\n\n[... schema truncated at {_SCHEMA_ROW_CAP} columns; "
                        "pass a specific table_name, or query information_schema "
                        "directly with your own filter, for the rest.]")
            return out

        elif action == "activity":
            # Bounded row count + ordering so the model sees the
            # longest-running queries first. Previously this was
            # unbounded and would flood the prompt context with
            # hundreds of `pg_stat_activity` rows on a busy DB.
            cur.execute(
                "SELECT pid, state, query, extract(epoch from now() - query_start) as duration_sec "
                "FROM pg_stat_activity "
                "WHERE state != 'idle' "
                "ORDER BY duration_sec DESC NULLS LAST "
                "LIMIT 50;"
            )
            rows = cur.fetchall()
            if not rows: return "No active queries."
            output = tabulate(rows, headers="keys", tablefmt="pipe")
            if len(rows) == 50:
                output += "\n\n[... showing 50 longest-running rows; LIMIT 50 applied. Use a direct `query` with your own filter for more detail. ...]"
            return output

        elif action in ["query", "explain_analyze"]:
            if not query: return "Error: query parameter required."
            sql = query
            if action == "explain_analyze" and not sql.upper().strip().startswith("EXPLAIN"):
                sql = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) {sql}"

            # statement_timeout already SET at the top of _run_action (for
            # every action), so no per-query SET is needed here.
            cur.execute(sql)
            if cur.description:
                # We fetch one extra row beyond our display cap so we
                # can detect "there are more than 300 rows" without
                # paying the cost of materialising the full result
                # set just to call `len()` on it.
                rows = cur.fetchmany(301)
                if not rows: return "Query executed successfully. No rows returned."
                truncated = len(rows) > 300
                output = tabulate(rows[:300], headers="keys", tablefmt="pipe")
                # Byte cap: the 300-ROW limit gives no protection against a
                # single huge cell (e.g. SELECT repeat('x', 1e8)) — the whole
                # value would materialise into the model context / host memory.
                _MAX_OUTPUT_CHARS = 200_000
                if len(output) > _MAX_OUTPUT_CHARS:
                    output = (
                        output[:_MAX_OUTPUT_CHARS]
                        + f"\n\n[... result truncated at {_MAX_OUTPUT_CHARS} chars "
                        "(a cell or the row set was very large). Narrow the "
                        "SELECT columns or add a LIMIT.]"
                    )
                    return output
                if truncated:
                    # We can't cheaply get the true total without
                    # `SELECT COUNT(*)` reissue against an arbitrary
                    # query (subquery wrapping is fragile), so we
                    # surface the cap honestly — the model now knows
                    # the result was clamped and how to dig deeper.
                    output += (
                        "\n\n[Returned 300 rows (max display limit). "
                        "The full result set may contain MORE rows. "
                        "Re-run with `LIMIT N OFFSET M` or wrap the query in "
                        "`SELECT count(*) FROM (...) sub` to know the true total.]"
                    )
                else:
                    output += f"\n\n[Returned {len(rows)} of {len(rows)} rows.]"
                return output
            return "Query executed successfully. No rows returned."
        else:
            return f"Error: Unknown action: {action}"

    # psycopg2's connection-level exception types. Resolved defensively:
    # under test doubles these attributes may not be real exception
    # classes, so we filter to genuine BaseException subclasses and match
    # with isinstance (never `except <non-class>`, which raises TypeError).
    _conn_err_types = tuple(
        t for t in (getattr(psycopg2, "OperationalError", None),
                    getattr(psycopg2, "InterfaceError", None))
        if isinstance(t, type) and issubclass(t, BaseException)
    )

    def execute_db():
        # Serialize all use of this URI's shared connection — psycopg2
        # connections are not thread-safe and tool dispatch is concurrent.
        with _get_uri_lock(connection_string):
            last_err = None
            for attempt in range(2):  # one retry on a stale/broken connection
                conn = None
                try:
                    conn = _get_connection(connection_string)
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                        return _run_action(cur)
                except Exception as e:
                    # Always discard the (possibly broken) connection.
                    last_err = e
                    if conn is not None:
                        try:
                            conn.close()
                        except Exception:
                            pass
                    _evict_connection(connection_string)
                    # A connection-level failure (server dropped an idle
                    # conn, network blip) is retried once with a fresh
                    # connection. Query-level errors (bad SQL) are not —
                    # the statement itself is the problem.
                    is_conn_err = bool(_conn_err_types) and isinstance(e, _conn_err_types)
                    if is_conn_err and attempt == 0:
                        logger.debug("DB connection-level error, retrying with a fresh connection: %s", e)
                        continue
                    pretty_log("Postgres Error", f"{type(e).__name__}: {str(e)[:160]}",
                               icon=Icons.FAIL, level="ERROR")
                    return f"Error: Postgres execution failed - {str(e)}"
            return f"Error: Postgres execution failed - {last_err}"

    result = await asyncio.to_thread(execute_db)
    if result.startswith("Error:"):
        return result
    return f"### POSTGRES RESULT ###\n{result}"
