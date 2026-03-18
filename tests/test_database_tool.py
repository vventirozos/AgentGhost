import pytest
import sys
from unittest.mock import MagicMock, patch
from ghost_agent.tools.database import tool_postgres_admin

# Mock modules that might not be installed in the test environment
sys.modules["psycopg2"] = MagicMock()
sys.modules["psycopg2.extras"] = MagicMock()
sys.modules["tabulate"] = MagicMock()

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
async def test_postgres_admin_schema():
    """Test schema retrieval."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    
    with patch("psycopg2.connect", return_value=mock_conn):
        with patch("tabulate.tabulate", return_value="Schema Table"):
            mock_cursor.fetchall.return_value = [{"table_name": "users"}]
            
            await tool_postgres_admin("schema", "postgres://uri", table_name="users")
            
            expected_sql = "SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'users'"
            mock_cursor.execute.assert_called_with(expected_sql)
