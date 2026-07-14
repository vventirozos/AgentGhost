import pytest
import sys
from unittest.mock import MagicMock, patch
from ghost_agent.tools.database import tool_postgres_admin


# Autouse fixture: patch `psycopg2` / `tabulate` only for the duration
# of each test in this file, then roll the change back.
#
# BEFORE: this module patched `sys.modules["psycopg2"] = MagicMock()`
# at import time, which permanently poisoned the interpreter's module
# cache. Any test file imported AFTER this one that tried a normal
# `import tabulate` got the `MagicMock` back and failed pytest
# collection with `ValueError: tabulate.__spec__ is not set`. The
# failure was pytest-order-dependent and flaky: the full suite
# sometimes passed, but running `test_database_tool.py` before
# `test_vision_integration.py` in isolation always tripped it.
# Moving the patch into a fixture with `patch.dict(...)` guarantees
# the original modules (or their absence) are restored when the test
# finishes.
@pytest.fixture(autouse=True)
def _mock_db_modules(monkeypatch):
    monkeypatch.setitem(sys.modules, "psycopg2", MagicMock())
    monkeypatch.setitem(sys.modules, "psycopg2.extras", MagicMock())
    monkeypatch.setitem(sys.modules, "tabulate", MagicMock())
    yield

@pytest.mark.asyncio
async def test_postgres_admin_missing_deps():
    """Test error message when dependencies are missing."""
    with patch.dict(sys.modules, {"psycopg2": None}):
        result = await tool_postgres_admin("query", "postgres://user:pass@localhost:5432/db", "SELECT 1")
        assert "Error: psycopg2 or tabulate library is missing" in result

@pytest.mark.asyncio
async def test_postgres_admin_missing_connection_string():
    """Test error when connection string is missing."""
    result = await tool_postgres_admin("query", "", "SELECT 1")
    assert "Error: connection_string is required" in result

@pytest.mark.asyncio
async def test_postgres_admin_schema_with_table_name():
    """Test that table_name is now passed parameterized, not f-string interpolated."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    with patch("psycopg2.connect", return_value=mock_conn):
        with patch("tabulate.tabulate", return_value="| ok |"):
            # schema now uses fetchmany (row-capped, 2026-07-14) + ORDER BY.
            mock_cursor.fetchmany.return_value = [{"table_name": "users", "column_name": "id", "data_type": "int"}]
            await tool_postgres_admin("schema", "postgres://uri", table_name="users")

            # Must be parameterized — second positional arg is the params tuple.
            mock_cursor.execute.assert_called_with(
                "SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema = 'public' AND table_name = %s ORDER BY ordinal_position",
                ("users",),
            )


@pytest.mark.asyncio
async def test_postgres_admin_schema_injection_blocked():
    """A SQL injection payload as table_name must NOT end up in the literal SQL."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    payload = "users'; DROP TABLE users; --"
    with patch("psycopg2.connect", return_value=mock_conn):
        with patch("tabulate.tabulate", return_value=""):
            mock_cursor.fetchmany.return_value = []
            await tool_postgres_admin("schema", "postgres://uri", table_name=payload)

            sql_arg, params_arg = mock_cursor.execute.call_args.args
            assert "DROP TABLE" not in sql_arg
            assert params_arg == (payload,)


@pytest.mark.asyncio
async def test_postgres_admin_query_execution():
    """Test successful query execution."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    
    # Mock context manager behavior
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    
    # Mock psycopg2.connect
    with patch("psycopg2.connect", return_value=mock_conn):
        # Mock tabulate
        with patch("tabulate.tabulate", return_value="| id | name |\n|----|------|\n| 1  | test |"):
            mock_cursor.fetchall.return_value = [{"id": 1, "name": "test"}]
            mock_cursor.description = True
            
            result = await tool_postgres_admin("query", "postgres://uri", "SELECT * FROM users")
            
            assert "### POSTGRES RESULT ###" in result
            assert "| id | name |" in result
            mock_cursor.execute.assert_called_with("SELECT * FROM users")

@pytest.mark.asyncio
async def test_postgres_admin_explain_analyze():
    """Test explain analyze auto-formatting."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    
    with patch("psycopg2.connect", return_value=mock_conn):
        with patch("tabulate.tabulate", return_value="Plan"):
            mock_cursor.fetchall.return_value = [{"Plan": "..."}]
            mock_cursor.description = True
            
            await tool_postgres_admin("explain_analyze", "postgres://uri", "SELECT * FROM users")
            
            # Verify EXPLAIN ANALYZE was prepended
            mock_cursor.execute.assert_called_with("EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) SELECT * FROM users")

@pytest.mark.asyncio
async def test_postgres_admin_schema_no_table_name():
    """Without table_name, the schema query runs unparameterised."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    with patch("psycopg2.connect", return_value=mock_conn):
        with patch("tabulate.tabulate", return_value="Schema Table"):
            mock_cursor.fetchmany.return_value = [{"table_name": "users"}]
            await tool_postgres_admin("schema", "postgres://uri")
            mock_cursor.execute.assert_called_with(
                "SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema = 'public' ORDER BY table_name, ordinal_position"
            )


# -----------------------------------------------------------------
# timeout_ms: int-coercion / clamping / parameterization
# (regression: timeout_ms was f-string-interpolated raw into a SET
#  statement → crash on non-int + SQL-injection sink on autocommit)
# -----------------------------------------------------------------

def _find_set_timeout_call(mock_cursor):
    """Return the (args, params) of the SET statement_timeout execute call."""
    for call in mock_cursor.execute.call_args_list:
        if call.args and isinstance(call.args[0], str) and "statement_timeout" in call.args[0]:
            return call.args
    return None


async def _run_query_with_timeout(timeout_ms):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.description = True
    mock_cursor.fetchmany.return_value = [{"x": 1}]
    with patch("psycopg2.connect", return_value=mock_conn):
        with patch("tabulate.tabulate", return_value="| x |"):
            result = await tool_postgres_admin(
                "query", "postgres://uri-timeout", "SELECT 1", timeout_ms=timeout_ms
            )
    return result, mock_cursor


@pytest.mark.asyncio
async def test_timeout_ms_string_does_not_crash_and_defaults():
    """A non-numeric timeout_ms (LLM often passes strings) must not crash;
    it falls back to the 15000ms default."""
    result, cur = await _run_query_with_timeout("not-a-number")
    assert not result.startswith("Error:")
    set_call = _find_set_timeout_call(cur)
    assert set_call == ("SET statement_timeout = %s", (15000,))


@pytest.mark.asyncio
async def test_timeout_ms_injection_payload_neutralized():
    """An injection payload in timeout_ms can never reach the SQL text."""
    result, cur = await _run_query_with_timeout("0; DROP TABLE users--")
    # int() fails → default; parameterized → payload is gone entirely.
    set_call = _find_set_timeout_call(cur)
    assert set_call == ("SET statement_timeout = %s", (15000,))
    # No execute call's SQL string contains the injection.
    for call in cur.execute.call_args_list:
        if call.args and isinstance(call.args[0], str):
            assert "DROP TABLE" not in call.args[0]


@pytest.mark.asyncio
async def test_timeout_ms_parameterized_and_clamped():
    """Valid ints are used; out-of-band values are clamped to [100, 600000]."""
    _, cur = await _run_query_with_timeout(5000)
    assert _find_set_timeout_call(cur) == ("SET statement_timeout = %s", (5000,))

    _, cur = await _run_query_with_timeout(99999999)
    assert _find_set_timeout_call(cur) == ("SET statement_timeout = %s", (600000,))

    _, cur = await _run_query_with_timeout(1)
    assert _find_set_timeout_call(cur) == ("SET statement_timeout = %s", (100,))


# -----------------------------------------------------------------
# per-URI connection lock + stale-connection retry
# -----------------------------------------------------------------

def test_uri_lock_is_per_uri_and_reused():
    import threading
    from ghost_agent.tools.database import _get_uri_lock
    a1 = _get_uri_lock("postgres://A")
    a2 = _get_uri_lock("postgres://A")
    b = _get_uri_lock("postgres://B")
    assert a1 is a2          # same URI → same lock (serializes that connection)
    assert a1 is not b       # different URI → different lock
    assert isinstance(a1, type(threading.Lock()))


@pytest.mark.asyncio
async def test_connection_retry_on_operational_error(monkeypatch):
    """A connection-level error (server dropped an idle conn) is retried
    once with a fresh connection instead of surfacing an error."""
    import ghost_agent.tools.database as db

    class _OpErr(Exception):
        pass

    class _IfErr(Exception):
        pass

    fake = MagicMock()
    fake.OperationalError = _OpErr
    fake.InterfaceError = _IfErr
    fake.extras = MagicMock()
    monkeypatch.setitem(sys.modules, "psycopg2", fake)
    monkeypatch.setitem(sys.modules, "psycopg2.extras", fake.extras)

    # Ensure a clean pool for this URI.
    uri = "postgres://retry-uri"
    db._evict_connection(uri)

    # First connection's cursor context raises OperationalError; second works.
    conn_bad = MagicMock()
    conn_bad.cursor.return_value.__enter__.side_effect = _OpErr("server closed the connection")
    conn_good = MagicMock()
    good_cursor = MagicMock()
    conn_good.cursor.return_value.__enter__.return_value = good_cursor
    good_cursor.fetchall.return_value = [{"table_name": "users"}]
    fake.connect.side_effect = [conn_bad, conn_good]

    with patch("tabulate.tabulate", return_value="| ok |"):
        result = await tool_postgres_admin("schema", uri)

    assert "POSTGRES RESULT" in result          # retry succeeded
    assert fake.connect.call_count == 2          # reconnected after eviction
    db._evict_connection(uri)
