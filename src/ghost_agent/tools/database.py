import asyncio
from typing import Optional
from ..utils.logging import Icons, pretty_log
from ..utils.sanitizer import extract_code_from_markdown

async def tool_postgres_admin(action: str, connection_string: Optional[str] = None, query: Optional[str] = None, table_name: Optional[str] = None, default_uri: Optional[str] = None, **kwargs):
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

    def execute_db():
        conn = None
        try:
            conn = psycopg2.connect(connection_string)
            conn.autocommit = True
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if action == "schema":
                    sql = "SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema = 'public'"
                    if table_name:
                        sql += f" AND table_name = '{table_name}'"
                    cur.execute(sql)
                    rows = cur.fetchall()
                    if not rows: return "No schema found."
                    return tabulate(rows, headers="keys", tablefmt="pipe")
                
                elif action == "activity":
                    cur.execute("SELECT pid, state, query, extract(epoch from now() - query_start) as duration_sec FROM pg_stat_activity WHERE state != 'idle';")
                    rows = cur.fetchall()
                    if not rows: return "No active queries."
                    return tabulate(rows, headers="keys", tablefmt="pipe")
                
                elif action in ["query", "explain_analyze"]:
                    if not query: return "Error: query parameter required."
                    sql = query
                    if action == "explain_analyze" and not sql.upper().strip().startswith("EXPLAIN"):
                        sql = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) {sql}"
                    
                    cur.execute("SET statement_timeout = 15000;")
                    cur.execute(sql)
                    if cur.description:
                        rows = cur.fetchmany(301)
                        if not rows: return "Query executed successfully. No rows returned."
                        output = tabulate(rows[:300], headers="keys", tablefmt="pipe")
                        if len(rows) > 300:
                            output += f"\n\n... [Truncated {len(rows)-300} rows]"
                        return output
                    return "Query executed successfully. No rows returned."
                else:
                    return f"Unknown action: {action}"
        except Exception as e:
            return f"Error: Postgres execution failed - {str(e)}"
        finally:
            if conn:
                conn.close()

    result = await asyncio.to_thread(execute_db)
    if result.startswith("Error:"):
        return result
    return f"### POSTGRES RESULT ###\n{result}"
