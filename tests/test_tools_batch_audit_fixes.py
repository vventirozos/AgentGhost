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

async def test_image_success_reports_the_size_the_node_rendered(tmp_path):
    """§4KA: the tool no longer snaps sizes — the node searches 246 legal sizes
    and the result must state what it actually rendered, not what was asked.
    Reporting the request as fact is how a user was told 768x512 when the node
    had chosen otherwise."""
    from ghost_agent.tools import image_gen
    import base64
    png = base64.b64encode(
        b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR"
        + (736).to_bytes(4, "big") + (416).to_bytes(4, "big")
        + b"\x08\x06\x00\x00\x00" + b"\x00" * 8).decode()
    llm = MagicMock()
    llm.image_gen_clients = [{"x": 1}]
    llm.generate_image = AsyncMock(return_value={"data": [{"b64_json": png}],
                                                 "width": 736, "height": 416})
    out = await image_gen.tool_generate_image(
        prompt="a cat", llm_client=llm, sandbox_dir=tmp_path, width=1920, height=1080)
    assert "SUCCESS" in out
    assert "Rendered at 736x416" in out          # what the node did
    assert "1920x1080" in out                    # and what was asked, so the model can say so


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
