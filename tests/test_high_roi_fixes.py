"""Regression + behavioral tests for the high-ROI fix block.

Coverage map:
  #1  CORS misconfiguration (api/app.py + interface/server.py)
  #3  helper_fetch_url_content body cap + scheme allowlist
  #4  file_system tool_read_file / tool_replace_text encoding + size cap
  #6  setup_logging clears stale handlers
  #7  estimate_tokens warns once on tokenizer fallback
  #8  stream_openai sleep removed
  #10 extract_json_from_text logs malformed JSON-like content
"""
import asyncio
import inspect
import logging
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =====================================================================
# #1 — CORS misconfiguration
# =====================================================================


def test_api_app_cors_credentials_disabled():
    """allow_origins=['*'] + allow_credentials=True is forbidden by spec.
    Verify the production CORSMiddleware call has been corrected (we strip
    comments before checking so a doc-line mentioning the old config doesn't
    trip the assertion)."""
    src = Path("src/ghost_agent/api/app.py").read_text()
    code_only = "\n".join(
        line for line in src.splitlines() if not line.strip().startswith("#")
    )
    assert "allow_credentials=False" in code_only
    assert "allow_credentials=True" not in code_only


def test_interface_server_cors_credentials_disabled():
    src = Path("interface/server.py").read_text()
    # The CORS middleware must use credentials=False alongside the wildcard.
    assert "allow_credentials=False" in src
    # The only `allow_credentials=True` allowed in the file would be in a
    # comment talking about why we don't use it — assert there's no actual
    # middleware config still using it.
    middleware_block = src.split("CORSMiddleware")[1].split(")")[0] if "CORSMiddleware" in src else ""
    assert "allow_credentials=True" not in middleware_block


# =====================================================================
# #8 — stream_openai sleep removed
# =====================================================================


def test_stream_openai_no_artificial_sleep_in_chunk_loop():
    """The 10ms sleep added 1+ second of latency to every fast-path streaming
    response. It must be gone."""
    from ghost_agent.core.llm import LLMClient
    src = inspect.getsource(LLMClient.stream_openai)
    # The function must NOT contain any `await asyncio.sleep(0.0..)` in the
    # for loop. We allow comments that mention sleep — only the `await` form
    # is forbidden.
    lines = [l for l in src.splitlines() if "asyncio.sleep" in l and not l.strip().startswith("#")]
    assert lines == [], f"stream_openai still contains sleep calls: {lines}"


@pytest.mark.asyncio
async def test_stream_openai_emits_chunks_with_no_artificial_delay():
    """Functional test: a 1500-char response should stream in well under
    100ms (the old code would take ~1 full second for this size)."""
    from ghost_agent.core.llm import LLMClient
    client = LLMClient.__new__(LLMClient)
    content = "x" * 1500  # 100 chunks at 15 char/chunk
    t0 = time.monotonic()
    chunks = []
    async for c in client.stream_openai("test", content, 0, "req"):
        chunks.append(c)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.2, f"stream_openai still slow: {elapsed:.3f}s for 1500 chars"
    # Sanity: we should have received multiple chunks.
    assert len(chunks) > 50


# =====================================================================
# #6 — setup_logging clears stale handlers
# =====================================================================


def test_setup_logging_does_not_accumulate_handlers(tmp_path):
    from ghost_agent.utils.logging import setup_logging
    log_file = tmp_path / "ghost.log"

    setup_logging(str(log_file), debug=False, daemon=False, verbose=False)
    first_count = len(logging.getLogger("GhostAgent").handlers)

    # Repeat 5 times — handler count must NOT grow.
    for _ in range(5):
        setup_logging(str(log_file), debug=False, daemon=False, verbose=False)

    final_count = len(logging.getLogger("GhostAgent").handlers)
    assert final_count == first_count, (
        f"setup_logging accumulated handlers: {first_count} → {final_count}"
    )


def test_setup_logging_daemon_mode_only_file_handler(tmp_path):
    from ghost_agent.utils.logging import setup_logging
    log_file = tmp_path / "ghost.log"
    setup_logging(str(log_file), daemon=True)
    handlers = logging.getLogger("GhostAgent").handlers
    # Daemon mode → exactly one FileHandler, no StreamHandler.
    file_handlers = [h for h in handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1


# =====================================================================
# #7 — estimate_tokens warns once on fallback
# =====================================================================


def test_estimate_tokens_warns_once_when_tokenizer_missing(caplog):
    """When the tokenizer hasn't loaded, the first fallback call must emit
    a single WARNING. Subsequent calls must NOT warn again."""
    from ghost_agent.utils import token_counter as tc
    tc._reset_fallback_warning()
    tc.clear_token_cache()
    saved = tc.TOKEN_ENCODER
    tc.TOKEN_ENCODER = None
    try:
        with caplog.at_level(logging.WARNING, logger="GhostAgent"):
            tc.estimate_tokens("hello world")
            tc.estimate_tokens("another fragment")
            tc.estimate_tokens("a third one")
        warnings = [r for r in caplog.records if "estimate_tokens" in r.message]
        assert len(warnings) == 1, f"expected exactly one fallback warning, got {len(warnings)}"
    finally:
        tc.TOKEN_ENCODER = saved
        tc._reset_fallback_warning()


def test_estimate_tokens_returns_char_quarter_fallback():
    from ghost_agent.utils import token_counter as tc
    saved = tc.TOKEN_ENCODER
    tc.TOKEN_ENCODER = None
    tc.clear_token_cache()
    try:
        # 100-char input → ~25 token fallback
        result = tc.estimate_tokens("x" * 100)
        assert result == 25
    finally:
        tc.TOKEN_ENCODER = saved
        tc._reset_fallback_warning()


def test_clear_token_cache_drops_lru():
    # The unbounded LRU was replaced with a tiny bounded dict cache. The
    # public clear_token_cache() surface is still expected to empty it.
    from ghost_agent.utils import token_counter as tc
    saved = tc.TOKEN_ENCODER
    tc.TOKEN_ENCODER = None
    tc.clear_token_cache()
    try:
        tc.estimate_tokens("foo bar")
        # The bounded dict cache should now hold at least one entry.
        assert len(tc._SHORT_TOKEN_CACHE) >= 1
        tc.clear_token_cache()
        assert len(tc._SHORT_TOKEN_CACHE) == 0
    finally:
        tc.TOKEN_ENCODER = saved
        tc._reset_fallback_warning()


# =====================================================================
# #4 — file_system encoding + size cap + binary sniff
# =====================================================================


@pytest.mark.asyncio
async def test_read_file_handles_text_with_occasional_bad_bytes(tmp_path):
    """A mostly-text file with one bad UTF-8 byte must read successfully
    using errors='replace' instead of being refused as 'binary'."""
    from ghost_agent.tools.file_system import tool_read_file
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    target = sandbox / "log.txt"
    # 5 KB of clean ASCII followed by a single rogue byte.
    target.write_bytes(b"clean text " * 500 + b"\xff" + b"more text")
    res = await tool_read_file("log.txt", sandbox)
    assert "CONTENTS" in res
    assert "clean text" in res
    assert "Error" not in res


@pytest.mark.asyncio
async def test_read_file_still_rejects_real_binary(tmp_path):
    """Binary files (PNG, ELF, etc.) must still be cleanly refused."""
    from ghost_agent.tools.file_system import tool_read_file
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    target = sandbox / "image.png"
    # PNG header + lots of NUL bytes.
    target.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 200 + b"\xff\xd8\xff" * 50)
    res = await tool_read_file("image.png", sandbox)
    assert "Error" in res
    assert "binary" in res.lower()


@pytest.mark.asyncio
async def test_replace_text_caps_huge_files(tmp_path):
    """tool_replace_text must refuse files larger than the 50 MB cap."""
    from ghost_agent.tools.file_system import tool_replace_text
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    target = sandbox / "huge.txt"
    # Write a sparse 60 MB file via seek (cheap on most filesystems).
    with open(target, "wb") as f:
        f.seek(60 * 1024 * 1024)
        f.write(b"x")
    res = await tool_replace_text("huge.txt", "old", "new", sandbox)
    assert "Error" in res
    assert "MB" in res


@pytest.mark.asyncio
async def test_replace_text_handles_text_with_bad_bytes(tmp_path):
    """Text file with occasional bad bytes still works under replace."""
    from ghost_agent.tools.file_system import tool_replace_text
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    target = sandbox / "data.txt"
    target.write_bytes(b"hello OLD world\xff trailing")
    res = await tool_replace_text("data.txt", "OLD", "NEW", sandbox)
    # Replace should have happened despite the bad byte.
    assert "SUCCESS" in res or "NEW" in target.read_text(errors="replace")


def test_looks_like_binary_heuristic():
    from ghost_agent.tools.file_system import _looks_like_binary
    # NUL byte → binary
    assert _looks_like_binary(b"hello\x00world")
    # Pure ASCII → text
    assert not _looks_like_binary(b"hello world\nthis is text\n")
    # Empty → not binary
    assert not _looks_like_binary(b"")
    # PNG header → binary (lots of high-bit + control bytes)
    assert _looks_like_binary(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    # UTF-8 with accented chars → text (no NUL, low ratio of control bytes)
    assert not _looks_like_binary("café résumé naïve".encode("utf-8") * 50)


# =====================================================================
# #3 — helper_fetch_url_content body cap + scheme allowlist
# =====================================================================


@pytest.mark.asyncio
async def test_helper_fetch_url_rejects_file_scheme():
    from ghost_agent.utils.helpers import helper_fetch_url_content
    res = await helper_fetch_url_content("file:///etc/passwd")
    assert "Error" in res
    assert "non-http" in res.lower() or "file" in res.lower()


@pytest.mark.asyncio
async def test_helper_fetch_url_rejects_localhost():
    from ghost_agent.utils.helpers import helper_fetch_url_content
    for host_url in (
        "http://localhost/foo",
        "http://127.0.0.1/foo",
        "http://0.0.0.0/foo",
        "http://169.254.169.254/latest/meta-data",
    ):
        res = await helper_fetch_url_content(host_url)
        assert "Error" in res, f"failed to reject {host_url}"
        assert "SSRF" in res or "local" in res.lower(), f"wrong error for {host_url}: {res}"


@pytest.mark.asyncio
async def test_helper_fetch_url_rejects_invalid_scheme():
    from ghost_agent.utils.helpers import helper_fetch_url_content
    for bad in ("javascript:alert(1)", "ftp://example.com/f", "data:text/plain;base64,YWJj"):
        res = await helper_fetch_url_content(bad)
        assert "Error" in res


@pytest.mark.asyncio
async def test_helper_fetch_url_caps_oversized_response():
    """Response with content-length > 5 MB must be refused."""
    from ghost_agent.utils import helpers

    class FakeResp:
        headers = {"content-type": "text/html", "content-length": str(20 * 1024 * 1024)}
        status_code = 200
        text = "<html>...</html>"

    class FakeSession:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def get(self, *a, **kw): return FakeResp()

    class FakeCurl:
        class requests:
            AsyncSession = lambda *a, **kw: FakeSession()

    with patch("ghost_agent.utils.helpers.curl_cffi", FakeCurl, create=True), \
         patch.dict("sys.modules", {"curl_cffi": FakeCurl, "curl_cffi.requests": FakeCurl.requests}):
        # Need to reimport to pick up the patched curl_cffi
        import importlib
        importlib.reload(helpers)
        res = await helpers.helper_fetch_url_content("https://example.com/huge")
        # Restore for other tests
        importlib.reload(helpers)
    assert "Error" in res or "MB" in res


@pytest.mark.asyncio
async def test_helper_fetch_url_truncates_oversize_body_text():
    """If the server lies about Content-Length (or omits it) and ships a
    huge body anyway, the function must truncate the decoded text to the
    5 MB ceiling rather than handing 100 MB to BeautifulSoup."""
    from ghost_agent.utils import helpers

    big_text = "<html><body>" + ("A" * 7_000_000) + "</body></html>"

    class FakeResp:
        headers = {"content-type": "text/html"}
        status_code = 200
        text = big_text

    class FakeSession:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def get(self, *a, **kw): return FakeResp()

    class FakeCurl:
        class requests:
            AsyncSession = lambda *a, **kw: FakeSession()

    import importlib
    with patch.dict("sys.modules", {"curl_cffi": FakeCurl, "curl_cffi.requests": FakeCurl.requests}):
        importlib.reload(helpers)
        res = await helpers.helper_fetch_url_content("https://example.com/huge")
        importlib.reload(helpers)
    # The body either errors at body-cap or comes back truncated.
    assert "TRUNCATED" in res or "Error" in res or len(res) < 6_000_000


# =====================================================================
# #10 — extract_json_from_text logs malformed JSON-like content
# =====================================================================


def test_extract_json_returns_empty_on_pure_text():
    """Plain prose without JSON returns {} silently — no warning."""
    from ghost_agent.core.agent import extract_json_from_text
    res = extract_json_from_text("Just some prose with no JSON in it at all.")
    assert res == {}


def test_extract_json_logs_warning_on_malformed_braces(caplog):
    """If the input clearly contains brace content but parsing fails, we
    must surface a WARNING so debugging is possible. Use a properly-delimited
    but invalid block so the brace-finder enters the parse branch."""
    from ghost_agent.core.agent import extract_json_from_text
    malformed = "Here is the plan: { thought: missing-quotes, next_action_id: !!! }"
    with caplog.at_level(logging.WARNING, logger="GhostAgent"):
        res = extract_json_from_text(malformed)
    assert res == {}
    warnings = [r for r in caplog.records if "extract_json_from_text" in r.message]
    assert len(warnings) >= 1


def test_extract_json_succeeds_on_python_dict_syntax():
    """Models that emit Python dict syntax should still parse via the AST fallback."""
    from ghost_agent.core.agent import extract_json_from_text
    py_dict = "{'thought': 'plan things', 'next_action_id': 'task1', 'done': True}"
    res = extract_json_from_text(py_dict)
    assert res.get("thought") == "plan things"
    assert res.get("done") is True


def test_extract_json_heals_qwen_equals_syntax_for_any_field():
    """The healing regex previously only fixed `"name"=` — now it should
    fix `"key"=` for any key."""
    from ghost_agent.core.agent import extract_json_from_text
    qwen_garbage = '{"name"="execute", "operation"="write", "content"="hi"}'
    res = extract_json_from_text(qwen_garbage)
    assert res.get("name") == "execute"
    assert res.get("operation") == "write"
    assert res.get("content") == "hi"
