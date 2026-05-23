import asyncio
import logging
from typing import Optional
from ..utils.logging import Icons, pretty_log
from ..utils.sanitizer import extract_code_from_markdown

logger = logging.getLogger("GhostAgent")

# Simple connection pool: cache one connection per URI to avoid
# opening/closing on every call during multi-step SQL workflows.
_connection_pool: dict = {}


def _get_connection(connection_string: str):
    """Get or create a cached connection for the given URI."""
    import psycopg2
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
        _connection_pool[connection_string] = conn
    return conn


async def tool_postgres_admin(action: str = None, connection_string: Optional[str] = None, query: Optional[str] = None, table_name: Optional[str] = None, default_uri: Optional[str] = None, timeout_ms: Optional[int] = None, **kwargs):
    if not action:
        return "SYSTEM ERROR: The 'action' parameter is MANDATORY. You must specify it."
    pretty_log("Postgres Admin", f"Action: {action}", icon=Icons.POSTGRES)
    try:
        import psycopg2
        import psycopg2.extras
        from tabulate import tabulate
    except ImportError:
        return "Error: psycopg2 or tabulate library is missing."

    connection_string = connection_string or default_uri
    if not connection_string:
        return "Error: connection_string is required and no default is configured. Ask the user for the DB URI."

    if query:
        query = extract_code_from_markdown(query)

    # Metacog pre-execution SQL validator (roadmap phase 1.4). Only
    # active when the bundle is wired (i.e. ``--enable-metacog`` set).
    # Validator rejects unparseable SQL, unguarded DELETE/UPDATE, and
    # raw DROP/TRUNCATE before the connection is even opened. We only
    # validate the `query` and `explain_analyze` actions because
    # `schema` and `activity` go through hand-crafted SQL on lines
    # 62-83 that never sees user content.
    _metacog = kwargs.get("_metacog_bundle")
    if (_metacog is not None and getattr(_metacog, "enabled", False)
            and query and action in ("query", "explain_analyze")):
        try:
            from .validators import validate_sql
            from ..core.metacog_log import (
                emit as _mc_emit, Subsystem as _mc_ss, LEVEL_WARN,
            )
            ok, reason = validate_sql(query)
            if not ok:
                _mc_emit(
                    _mc_ss.VALID, level=LEVEL_WARN,
                    verdict="block", tool="sql",
                    action=action, reason=reason,
                    sql_head=query[:60],
                )
                try:
                    _metacog.count(validator_block=True)
                except Exception:
                    pass
                return (
                    f"SYSTEM BLOCK: SQL statement rejected by pre-execution "
                    f"validator: {reason}. The query was not run. Re-emit with "
                    f"a WHERE clause, or set `confirm=true` if you genuinely "
                    f"need to drop/truncate."
                )
        except Exception as _vexc:
            logger.debug("metacog SQL validator crashed: %s", _vexc)

    # Default timeout: 15s, but allow override for complex queries
    effective_timeout = timeout_ms or 15000

    def execute_db():
        conn = None
        try:
            conn = _get_connection(connection_string)
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if action == "schema":
                    # Parameterized — table_name comes from an LLM and could
                    # otherwise carry an injection payload ("'; DROP TABLE--").
                    sql = "SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema = 'public'"
                    if table_name:
                        sql += " AND table_name = %s"
                        cur.execute(sql, (table_name,))
                    else:
                        cur.execute(sql)
                    rows = cur.fetchall()
                    if not rows: return "No schema found."
                    return tabulate(rows, headers="keys", tablefmt="pipe")
                
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
                    
                    cur.execute(f"SET statement_timeout = {effective_timeout};")
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
                    return f"Unknown action: {action}"
        except Exception as e:
            # On error, close and discard the pooled connection to avoid
            # reusing a broken connection on the next call.
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
                _connection_pool.pop(connection_string, None)
            return f"Error: Postgres execution failed - {str(e)}"

    result = await asyncio.to_thread(execute_db)
    if result.startswith("Error:"):
        return result
    return f"### POSTGRES RESULT ###\n{result}"
