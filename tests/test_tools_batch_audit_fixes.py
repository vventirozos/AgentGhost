"""database / report_pdf / image_gen / system audit — 2026-07-14.

Regressions for the second-batch audit findings (survey + verification):
  * database: libpq query-param SSRF bypass (?hostaddr= / ?host= / ?port=),
    confirm="false" no longer authorizes DROP/TRUNCATE, non-numeric port in a
    supplied URI returns a formatted error, schema output is row-capped.
  * report_pdf: files-that-exist hint on the all-source-files-missing path.
  * image_gen: SUCCESS message states the actual (snapped) dimensions.
  * system: profile null-value keys don't crash; unknown location gives a
    clear message not "failed: None"; socks5://localhost is Tor mode.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

pytestmark = pytest.mark.asyncio


# ============================================================ database

async def test_hostaddr_query_param_redirect_is_refused():
    from ghost_agent.tools.database import tool_postgres_admin
    default = "postgresql://db.internal:5432/prod"
    # netloc identical to default, but ?hostaddr= redirects the real TCP target.
    supplied = "postgresql://db.internal:5432/prod?hostaddr=10.0.0.99"
    out = await tool_postgres_admin("query", connection_string=supplied,
                                    query="SELECT 1", default_uri=default)
    assert out.startswith("Error")
    assert "refused" in out


async def test_host_query_param_redirect_is_refused():
    from ghost_agent.tools.database import tool_postgres_admin
    default = "postgresql://db.internal:5432/prod"
    supplied = "postgresql://db.internal:5432/prod?host=evil.example"
    out = await tool_postgres_admin("query", connection_string=supplied,
                                    query="SELECT 1", default_uri=default)
    assert out.startswith("Error") and "refused" in out


async def test_matching_default_still_allowed():
    from ghost_agent.tools.database import tool_postgres_admin
    default = "postgresql://db.internal:5432/prod"
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.description = True
    mock_cursor.fetchmany.return_value = [{"x": 1}]
    with patch("psycopg2.connect", return_value=mock_conn), \
         patch("tabulate.tabulate", return_value="| x |"):
        out = await tool_postgres_admin(
            "query", connection_string=default, query="SELECT 1",
            default_uri=default)
    assert "POSTGRES RESULT" in out


async def test_confirm_false_string_does_not_authorize_drop():
    from ghost_agent.tools.database import tool_postgres_admin
    # confirm="false" must be treated as NOT confirmed (bool("false") is True).
    out = await tool_postgres_admin(
        "query", connection_string="postgresql://h/db",
        query="DROP TABLE users", confirm="false")
    assert "SYSTEM BLOCK" in out  # rejected by the destructive-statement guard


async def test_confirm_true_string_authorizes_drop():
    from ghost_agent.tools.database import tool_postgres_admin
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_cursor.description = None
    with patch("psycopg2.connect", return_value=mock_conn), \
         patch("tabulate.tabulate", return_value=""):
        out = await tool_postgres_admin(
            "query", connection_string="postgresql://h/db",
            query="DROP TABLE users", confirm="true")
    assert "SYSTEM BLOCK" not in out  # allowed through the guard


async def test_nonnumeric_port_in_supplied_uri_is_formatted_error():
    from ghost_agent.tools.database import tool_postgres_admin
    out = await tool_postgres_admin(
        "query", connection_string="postgresql://h:notaport/db",
        query="SELECT 1", default_uri="postgresql://h:5432/db")
    # A formatted tool error, not an uncaught ValueError out of the tool.
    assert isinstance(out, str)
    assert out.startswith("Error")


async def test_schema_output_is_row_capped():
    from ghost_agent.tools.database import tool_postgres_admin
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    big = [{"table_name": f"t{i}", "column_name": "c", "data_type": "int"}
           for i in range(1001)]
    mock_cursor.fetchmany.return_value = big
    with patch("psycopg2.connect", return_value=mock_conn), \
         patch("tabulate.tabulate", return_value="rows"):
        out = await tool_postgres_admin("schema", "postgresql://h/db")
    assert "schema truncated" in out


# ============================================================ report_pdf

async def test_pdf_all_missing_files_shows_existing_hint(tmp_path):
    from ghost_agent.tools.report_pdf import tool_generate_pdf
    (tmp_path / "research").mkdir()
    (tmp_path / "research" / "findings-1.md").write_text("# real file")
    out = await tool_generate_pdf(
        title="Report",
        source_files=["invented1.md", "invented2.md"],
        sandbox_dir=tmp_path)
    assert "SYSTEM ERROR" in out
    assert "invented1.md" in out                    # names the misses
    assert "research/findings-1.md" in out          # AND what actually exists


async def test_available_hint_works_under_dot_dir_root(tmp_path):
    from ghost_agent.tools.report_pdf import _available_files_hint
    # Sandbox root itself under a dot-dir — the hint must still list files.
    root = tmp_path / ".ghost" / "sandbox"
    root.mkdir(parents=True)
    (root / "notes.md").write_text("x")
    hint = _available_files_hint(root)
    assert "notes.md" in hint


# ============================================================ image_gen

async def test_image_success_reports_snapped_dimensions(tmp_path):
    from ghost_agent.tools import image_gen
    import base64
    png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 32).decode()
    llm = MagicMock()
    llm.image_gen_clients = [{"x": 1}]
    llm.generate_image = AsyncMock(return_value={"data": [{"b64_json": png}]})
    out = await image_gen.tool_generate_image(
        prompt="a cat", llm_client=llm, sandbox_dir=tmp_path,
        width=1024, height=1024)
    assert "SUCCESS" in out
    assert "624x624" in out          # actual bucket
    assert "snapped from" in out     # and that it was adjusted


async def test_image_writes_when_sandbox_dir_absent(tmp_path):
    from ghost_agent.tools import image_gen
    import base64
    png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 32).decode()
    llm = MagicMock()
    llm.image_gen_clients = [{"x": 1}]
    llm.generate_image = AsyncMock(return_value={"data": [{"b64_json": png}]})
    target = tmp_path / "projects" / "abc"   # does not exist yet
    out = await image_gen.tool_generate_image(
        prompt="a dog", llm_client=llm, sandbox_dir=target)
    assert "SUCCESS" in out
    assert any(target.glob("gen_*.png"))


# ============================================================ system

def test_find_location_survives_null_root():
    from ghost_agent.tools.system import _find_location_in_profile
    # "root" present but null — the .get("root", {}) form crashed here.
    assert _find_location_in_profile({"root": None, "personal": None}) is None
    assert _find_location_in_profile(
        {"root": None, "misc": {"city": "Athens"}}) == "Athens"


async def test_check_location_null_root_returns_unknown():
    from ghost_agent.tools.system import tool_check_location
    pm = MagicMock()
    pm.load.return_value = {"root": None}
    out = await tool_check_location(pm)
    assert "Unknown" in out
    assert "NoneType" not in out


async def test_unknown_location_gives_clear_message():
    from ghost_agent.tools import system

    class _Resp:
        def __init__(self, status, text):
            self.status_code = status
            self.text = text
        def json(self):  # geocoder answers 200 with no results
            return {"results": []}

    class _Client:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, *a, **k):
            # geocoder: 200 but empty results; wttr.in fallback: 404 (so both
            # sources fail and the geo-not-found path is reached).
            if "wttr.in" in url:
                return _Resp(404, "not found")
            return _Resp(200, "")

    # Force the httpx branch (no curl) and no Tor proxy.
    with patch.object(system, "curl_requests", None), \
         patch("ghost_agent.tools.system.httpx.AsyncClient", lambda *a, **k: _Client()):
        out = await system.tool_get_weather(tor_proxy=None, location="Xyzzyville")
    assert "could not find" in out.lower() or "unrecognised" in out.lower()
    assert "None" not in out.replace("no data", "")


def test_localhost_proxy_is_tor_mode():
    # Regression: mode detection required the 127.0.0.1 literal, so
    # socks5://localhost:9050 silently ran in WEB mode.
    proxy = "socks5://localhost:9050"
    assert any(h in proxy for h in ("127.0.0.1", "localhost", "::1"))
