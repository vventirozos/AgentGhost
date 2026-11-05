"""11-05 prod probe PG: told `postgres_admin` was gone, the model ran `psql` in the sandbox (exit 0 through `|| true`),
read the socket error and spent 6 minutes trying to initdb/start its own server to the strike cap. The DATABASES prompt
line did not stop it. The result of a database client that found no server now carries the route (`lab`), whether
the command exited 0 or not, for a command and for a script."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ghost_agent.tools import execute as E
from ghost_agent.tools.execute import _no_db_server_note, tool_execute

PSQL_SOCKET = ('psql: error: connection to server on socket "/var/run/postgresql/.s.PGSQL.5432" failed: '
               'No such file or directory\n\tIs the server running locally and accepting connections on that socket?')
PSYCOPG_TCP = ('psycopg2.OperationalError: connection to server at "localhost" (127.0.0.1), port 5432 failed: '
               'Connection refused\n\tIs the server running on that host and accepting TCP/IP connections?')
OLD_LIBPQ = 'psql: could not connect to server: Connection refused'
MYSQL = "ERROR 2002 (HY000): Can't connect to local MySQL server through socket '/run/mysqld/mysqld.sock' (2)"


def _mgr(returns):
    mgr = MagicMock()
    mgr.execute = MagicMock(return_value=returns)
    return mgr


@pytest.fixture
def labs_on(monkeypatch):
    import ghost_agent.tools.labs as L
    monkeypatch.setattr(L, "labs_configured", lambda ctx: True)


@pytest.mark.parametrize("out", [PSQL_SOCKET, PSYCOPG_TCP, OLD_LIBPQ, MYSQL])
def test_every_client_shape_of_no_server_gets_the_note(out, labs_on):
    note = _no_db_server_note(out, SimpleNamespace())
    assert "NO DATABASE SERVER HERE" in note and "lab create" in note and "Do NOT try to install, initdb" in note


@pytest.mark.parametrize("out", ["", "SELECT 1\n 1", "ERROR: relation \"t\" does not exist",
                                 "server closed the connection unexpectedly"])
def test_other_output_gets_no_note(out):
    assert _no_db_server_note(out, SimpleNamespace()) == ""


def test_without_a_labs_host_the_note_says_no_server_is_available(monkeypatch):
    import ghost_agent.tools.labs as L
    monkeypatch.setattr(L, "labs_configured", lambda ctx: False)
    note = _no_db_server_note(PSQL_SOCKET, SimpleNamespace())
    assert "no labs host is configured" in note and "lab create" not in note


def test_a_caller_with_no_context_never_states_labs_are_unconfigured():
    """review r19: the acquired-skill and create_skill paths passed no context and were told "no labs host"."""
    note = _no_db_server_note(PSQL_SOCKET, None)
    assert "no labs host is configured" not in note and "if the `lab` tool is available" in note


@pytest.mark.parametrize("out", [
    'src/db.py:12:    raise RuntimeError("could not connect to server: Connection refused")',      # grep -rn of code
    "2026-11-05 app: retrying; psql: error: connection to server on socket x failed",            # a log line quoting it
    "opened connection to server at 10.0.0.5:8080\n3 tests failed",                              # two unrelated lines
    'psql: error: connection to server at "db.example.com" (93.1.2.3), port 5432 failed: timeout',  # a REMOTE database
])
def test_quoted_joined_or_remote_shapes_get_no_note(out):
    assert _no_db_server_note(out, SimpleNamespace()) == ""


@pytest.mark.parametrize("out", [
    "OSError: Multiple exceptions: [Errno 111] Connect call failed ('127.0.0.1', 5432)",
    "ConnectionRefusedError: [Errno 111] Connect call failed ('127.0.0.1', 5432)",
    "ERROR 2002 (HY000): Can't connect to local server through socket '/run/mysqld/mysqld.sock' (2)",
])
def test_asyncpg_and_mariadb_shapes_get_the_note(out, labs_on):
    assert "NO DATABASE SERVER HERE" in _no_db_server_note(out, SimpleNamespace())


async def test_an_exit_0_psql_command_carries_the_note(tmp_path, labs_on):
    """the live shape: `psql -c 'SELECT 1' 2>&1 || true` exits 0."""
    res = await tool_execute(command="psql -c 'SELECT 1' 2>&1 || true", sandbox_dir=tmp_path,
                             sandbox_manager=_mgr((PSQL_SOCKET, 0)), run_context=SimpleNamespace())
    assert "EXIT CODE: 0" in res and "NO DATABASE SERVER HERE" in res and "lab create" in res


async def test_a_failed_psql_command_carries_the_note(tmp_path, labs_on):
    res = await tool_execute(command="psql -c 'SELECT 1'", sandbox_dir=tmp_path,
                             sandbox_manager=_mgr((PSQL_SOCKET, 2)), run_context=SimpleNamespace())
    assert "EXIT CODE: 2" in res and "NO DATABASE SERVER HERE" in res


async def test_a_script_that_cannot_connect_carries_the_note(tmp_path, labs_on):
    res = await tool_execute(filename="q.py", content="import psycopg2\n", sandbox_dir=tmp_path,
                             sandbox_manager=_mgr((PSYCOPG_TCP, 1)), run_context=SimpleNamespace())
    assert "NO DATABASE SERVER HERE" in str(res)


async def test_an_ordinary_command_is_unchanged(tmp_path):
    res = await tool_execute(command="echo hi", sandbox_dir=tmp_path, sandbox_manager=_mgr(("hi", 0)))
    assert "NO DATABASE SERVER" not in res


def test_the_note_is_not_read_as_a_timeout_or_retryable_failure():
    """the strike classifier reads result text — the note must not look transient"""
    note = _no_db_server_note(PSQL_SOCKET, None).lower()
    assert "timed out" not in note and "timeout" not in note and "try again" not in note
    assert E._NO_DB_SERVER_RE.pattern


# --- review r20 -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("out", [
    'E       psycopg2.OperationalError: connection to server on socket "/var/run/postgresql/.s.PGSQL.5432" failed: '
    'No such file or directory',                                                                      # pytest
    'psycopg.OperationalError: connection failed: connection to server at "127.0.0.1", port 5432 failed: '
    'Connection refused',                                                                             # psycopg 3
    "- host: 'localhost', port: '5432', hostaddr: '::1': connection failed: connection to server at \"::1\", "
    "port 5432 failed: Connection refused",                                                           # psycopg 3 multi
    '(psycopg2.OperationalError) connection to server at "localhost" (::1), port 5432 failed: Connection refused',
    'sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) connection to server at "localhost" (127.0.0.1), '
    'port 5432 failed: Connection refused',
    "ERROR 2003 (HY000): Can't connect to MySQL server on '127.0.0.1:3306' (111)",                   # mysql -h127.0.0.1
    "pymysql.err.OperationalError: (2003, \"Can't connect to MySQL server on 'localhost' ([Errno 111] Connection "
    "refused)\")",
    "mysql.connector.errors.DatabaseError: 2003 (HY000): Can't connect to MySQL server on 'localhost:3306' (111)",
])
def test_r20_client_shapes_get_the_note(out, labs_on):
    assert "NO DATABASE SERVER HERE" in _no_db_server_note(out, SimpleNamespace())


@pytest.mark.parametrize("out", [
    'psql: error: connection to server on socket "/var/run/postgresql/.s.PGSQL.5432" failed: FATAL:  role "x" does '
    'not exist',
    'psql: error: connection to server at "127.0.0.1", port 5432 failed: FATAL:  password authentication failed for '
    'user "postgres"',
    "ERROR 2003 (HY000): Can't connect to MySQL server on 'db.example.com:3306' (111)",
    "ERROR 1045 (28000): Access denied for user 'root'@'localhost' (using password: YES)",
])
def test_r20_a_server_that_answered_or_a_remote_one_gets_no_note(out):
    """r20 M2: the note states there is no server — false when one answered with FATAL/denied."""
    assert _no_db_server_note(out, SimpleNamespace()) == ""


def test_r20_labs_configured_raising_is_not_stated_as_unconfigured(monkeypatch):
    import ghost_agent.tools.labs as L

    def boom(ctx):
        raise RuntimeError("settings unreadable")
    monkeypatch.setattr(L, "labs_configured", boom)
    note = _no_db_server_note(PSQL_SOCKET, SimpleNamespace())
    assert "no labs host is configured" not in note and "if the `lab` tool is available" in note


def test_r20_the_client_line_after_a_traceback_still_counts(labs_on):
    """battery D7 survived: every pinned shape was the FIRST line; real output has a traceback above it."""
    out = ("Traceback (most recent call last):\n  File \"q.py\", line 2, in <module>\n"
           "    conn = psycopg2.connect(dbname='x')\n" + PSYCOPG_TCP)
    assert "NO DATABASE SERVER HERE" in _no_db_server_note(out, SimpleNamespace())


@pytest.mark.parametrize("out", ["/var/run/postgresql:5432 - no response", "localhost:5432 - no response"])
def test_r20_pg_isready_no_response_gets_the_note(out, labs_on):
    assert "NO DATABASE SERVER HERE" in _no_db_server_note(out, SimpleNamespace())


def test_r20_pg_isready_on_a_remote_host_gets_no_note():
    assert _no_db_server_note("db.example.com:5432 - no response", SimpleNamespace()) == ""


@pytest.mark.parametrize("out", [
    "ERROR 2003 (HY000): Can't connect to server on '127.0.0.1' (111)",                  # MariaDB client, TCP
    "ERROR 2002 (HY000): Can't connect to server on '127.0.0.1' (115)",                  # MariaDB non-blocking connect
    "MySQLdb.OperationalError: (2002, \"Can't connect to server on '127.0.0.1' (115)\")",  # mysqlclient
    "MySQLdb.OperationalError: (2002, \"Can't connect to local server through socket '/run/mysqld/mysqld.sock' (2)\")",
    "pymysql.err.OperationalError: (2003, \"Can't connect to MySQL server on '127.0.0.1' ([Errno 99] Cannot assign "
    "requested address)\")",
    "Error: connect ECONNREFUSED 127.0.0.1:5432",                                         # node-postgres
])
def test_r21_mariadb_mysqlclient_and_node_shapes_get_the_note(out, labs_on):
    """r21 MAJOR: Debian's default mysql client is MariaDB, whose TCP wording has no "MySQL"."""
    assert "NO DATABASE SERVER HERE" in _no_db_server_note(out, SimpleNamespace())


@pytest.mark.parametrize("out", ["ERROR 2003 (HY000): Can't connect to server on 'db.example.com' (111)",
                                 "Error: connect ECONNREFUSED 127.0.0.1:8080"])
def test_r21_remote_mariadb_or_a_non_db_port_gets_no_note(out):
    assert _no_db_server_note(out, SimpleNamespace()) == ""
