
import pytest
import sys
from unittest.mock import MagicMock, patch, AsyncMock

# Retrieve the function to test
from ghost_agent.tools.database import tool_postgres_admin

@pytest.fixture
def mock_postgres_env():
    """Mock psycopg2 and tabulate modules in sys.modules."""
    mock_psycopg2 = MagicMock()
    mock_tabulate_module = MagicMock()
    # Ensure tabulate() function returns a string, otherwise `startswith` on a Mock is True!
    mock_tabulate_module.tabulate.return_value = "id | name\n---+-----\n 1 | test"
    
    with patch.dict(sys.modules, {
        "psycopg2": mock_psycopg2, 
        "psycopg2.extras": MagicMock(),
        "tabulate": mock_tabulate_module
    }):
        yield mock_psycopg2, mock_tabulate_module

@pytest.mark.asyncio
async def test_tool_postgres_admin_schema_success(mock_postgres_env):
    """Test retrieving schema successfully."""
    mock_psycopg2, mock_tabulate_module = mock_postgres_env
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    
    # Setup mock returns
    mock_psycopg2.connect.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.fetchall.return_value = [
        {"table_name": "users", "column_name": "id", "data_type": "integer"},
        {"table_name": "users", "column_name": "name", "data_type": "text"}
    ]
    
    result = await tool_postgres_admin("schema", "postgres://user:pass@localhost:5432/db")
    
    # Verify connection usage
    mock_psycopg2.connect.assert_called_with("postgres://user:pass@localhost:5432/db")
    assert mock_conn.autocommit is True
    mock_conn.close.assert_called_once()
    
    # Verify cursor usage
    mock_cursor.execute.assert_called()
    assert "SELECT table_name" in mock_cursor.execute.call_args[0][0]
    
    # Verify output format
    assert "### POSTGRES RESULT ###" in result
    # The result content comes from our mock_tabulate_module.tabulate.return_value
    assert "id | name" in result

@pytest.mark.asyncio
async def test_tool_postgres_admin_query_sanitization(mock_postgres_env):
    """Test that queries are sanitized (markdown stripped)."""
    mock_psycopg2, mock_tabulate_module = mock_postgres_env
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    
    mock_psycopg2.connect.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.description = True
    mock_cursor.fetchall.return_value = [{"id": 1}]
    
    query_with_markdown = "```sql\nSELECT * FROM users\n```"
    expected_sql = "SELECT * FROM users"
    
    await tool_postgres_admin("query", "db_uri", query=query_with_markdown)
    
    # Check that the sanitized query was executed
    args, _ = mock_cursor.execute.call_args
    assert args[0].strip() == expected_sql

@pytest.mark.asyncio
async def test_tool_postgres_admin_connection_error_handling(mock_postgres_env):
    """Test handling of connection errors."""
    mock_psycopg2, _ = mock_postgres_env
    mock_psycopg2.connect.side_effect = Exception("Connection failed")
    
    result = await tool_postgres_admin("query", "db_uri", query="SELECT 1")
    
    assert "Error: Postgres execution failed - Connection failed" in result
    assert "### POSTGRES RESULT ###" not in result

@pytest.mark.asyncio
async def test_tool_postgres_admin_query_execution_error(mock_postgres_env):
    """Test handling of errors during query execution."""
    mock_psycopg2, _ = mock_postgres_env
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    
    mock_psycopg2.connect.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.execute.side_effect = Exception("Syntax error")
    
    result = await tool_postgres_admin("query", "db_uri", query="SELECT * FROM bad_table")
    
    assert "Error: Postgres execution failed - Syntax error" in result
    # Ensure connection is still closed
    mock_conn.close.assert_called_once()

@pytest.mark.asyncio
async def test_tool_postgres_admin_missing_connection_string(mock_postgres_env):
    """Test missing connection string returns error."""
    result = await tool_postgres_admin("query", "", query="SELECT 1")
    assert "Error: connection_string is required" in result

@pytest.mark.asyncio
async def test_tool_postgres_admin_missing_query_param(mock_postgres_env):
    """Test query action without query parameter."""
    mock_psycopg2, _ = mock_postgres_env
    mock_conn = MagicMock()
    mock_psycopg2.connect.return_value = mock_conn
    
    result = await tool_postgres_admin("query", "db_uri", query=None)
    assert "Error: query parameter required" in result
    mock_conn.close.assert_called_once()

@pytest.mark.asyncio
async def test_tool_postgres_admin_explain_analyze_prefix(mock_postgres_env):
    """Test that explain analyze action adds EXPLAIN prefix if missing."""
    mock_psycopg2, _ = mock_postgres_env
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    
    mock_psycopg2.connect.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.description = True
    
    await tool_postgres_admin("explain_analyze", "db_uri", query="SELECT 1")
    
    args, _ = mock_cursor.execute.call_args
    assert args[0].startswith("EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)")
    assert "SELECT 1" in args[0]
