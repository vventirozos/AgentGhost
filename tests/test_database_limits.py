
import pytest
from unittest.mock import MagicMock, patch
import sys
from ghost_agent.tools.database import tool_postgres_admin

@pytest.mark.asyncio
async def test_database_fetch_limit_enforced():
    # Mock psycopg2 and tabulate in sys.modules so the import inside the function gets our mock
    mock_psycopg2 = MagicMock()
    mock_psycopg2_extras = MagicMock()
    mock_tabulate = MagicMock()
    
    # We need to make sure 'tabulate' module has a 'tabulate' function
    mock_tabulate.tabulate.return_value = "Mock Table Output"

    with patch.dict(sys.modules, {
        "psycopg2": mock_psycopg2, 
        "psycopg2.extras": mock_psycopg2_extras,
        "tabulate": mock_tabulate
    }):
        # Setup mock connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        
        mock_psycopg2.connect.return_value = mock_conn
        # conn.cursor(...) returns a context manager
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        # Setup mock cursor description
        mock_cursor.description = [("col1", "type_code")]
        
        # Determine the behavior for fetchmany
        limited_result = [{"col1": i} for i in range(301)]
        mock_cursor.fetchmany.return_value = limited_result
        mock_cursor.fetchall.side_effect = Exception("Should not call fetchall")
        
        # Execute
        result = await tool_postgres_admin(
            action="query", 
            connection_string="postgres://user:pass@localhost:5432/db", 
            query="SELECT * FROM huge_table"
        )
        
        # Verification
        # 1. Ensure fetchmany(101) was called instead of fetchall()
        mock_cursor.fetchmany.assert_called_once_with(301)
        mock_cursor.fetchall.assert_not_called()
        
        # 2. Result should contain the mock table output or indication of truncation
        # The tool code appends truncation info if len > 100
        assert "Mock Table Output" in result
        assert "... [Truncated" in result
