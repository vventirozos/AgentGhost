"""SSRF-on-redirect in download + onion body-cap (PROJECT_JOURNAL §4B).

`tool_download_file` validated only the ORIGINAL URL while both HTTP clients
auto-followed redirects, so a non-Tor 302 → 169.254.169.254 / 127.0.0.1 / a LAN
host bypassed the guard. The fix disables auto-redirect and follows hops
manually, re-validating each Location. Separately, the darkweb onion fetch read
the whole body into RAM before capping it (a chunked no-Content-Length body
OOMs the host); it now streams with a hard byte ceiling.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.utils.helpers import url_ssrf_reason
from ghost_agent.tools.file_system import _download_redirect_target, tool_download_file


# --------------------------------------------- redirect-hop decision (pure)


def test_redirect_to_internal_is_blocked():
    for loc in ("http://127.0.0.1:9051/", "http://169.254.169.254/latest/",
                "http://10.0.0.5/x", "//169.254.169.254/x"):
        tgt, err = _download_redirect_target(302, {"location": loc}, "http://pub.com", url_ssrf_reason)
        assert tgt is None and err and "SSRF" in err, loc


def test_redirect_to_public_is_followed():
    tgt, err = _download_redirect_target(302, {"location": "http://other.com/f"},
                                         "http://pub.com", url_ssrf_reason)
    assert tgt == "http://other.com/f" and err is None


def test_relative_redirect_resolved_against_current_url():
    tgt, err = _download_redirect_target(301, {"location": "/deeper/file"},
                                         "http://pub.com/a/b", url_ssrf_reason)
    assert tgt == "http://pub.com/deeper/file" and err is None


def test_non_redirect_is_final():
    assert _download_redirect_target(200, {}, "http://pub.com", url_ssrf_reason) == (None, None)


def test_all_redirect_codes_covered():
    for code in (301, 302, 303, 307, 308):
        tgt, err = _download_redirect_target(code, {"location": "http://127.0.0.1/x"},
                                             "http://pub.com", url_ssrf_reason)
        assert err and "SSRF" in err


# --------------------------------------------- end-to-end: httpx 302 → internal


def _stream_ctx(status, headers, body_chunks=()):
    resp = MagicMock()
    resp.status_code = status
    resp.headers = headers

    async def _aiter():
        for c in body_chunks:
            yield c
    resp.aiter_bytes = _aiter
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx


@pytest.mark.asyncio
async def test_download_blocks_redirect_to_metadata(tmp_path):
    """A public URL 302-redirecting to the cloud metadata IP must be blocked,
    and nothing written."""
    with patch("ghost_agent.tools.file_system.curl_requests", None), \
         patch("ghost_agent.tools.file_system.httpx.AsyncClient") as cls:
        client = MagicMock()
        cls.return_value.__aenter__ = AsyncMock(return_value=client)
        cls.return_value.__aexit__ = AsyncMock(return_value=None)
        # First (and only) request returns a 302 to the metadata service.
        client.stream = MagicMock(return_value=_stream_ctx(
            302, {"location": "http://169.254.169.254/latest/meta-data/"}))

        res = await tool_download_file("http://evil.example/pull", tmp_path, None, "out.bin")
        assert "SSRF" in res
        assert not (tmp_path / "out.bin").exists()


@pytest.mark.asyncio
async def test_download_follows_safe_redirect_then_writes(tmp_path):
    """A redirect to a public host is followed and the final body written."""
    responses = [
        _stream_ctx(302, {"location": "http://cdn.example/real"}),
        _stream_ctx(200, {"Content-Length": "6"}, [b"abc", b"def"]),
    ]
    with patch("ghost_agent.tools.file_system.curl_requests", None), \
         patch("ghost_agent.tools.file_system.httpx.AsyncClient") as cls:
        client = MagicMock()
        cls.return_value.__aenter__ = AsyncMock(return_value=client)
        cls.return_value.__aexit__ = AsyncMock(return_value=None)
        client.stream = MagicMock(side_effect=responses)

        res = await tool_download_file("http://start.example/x", tmp_path, None, "out.bin")
        assert "SUCCESS" in res
        assert (tmp_path / "out.bin").read_bytes() == b"abcdef"


@pytest.mark.asyncio
async def test_download_bounds_redirect_loop(tmp_path):
    """An endless public→public redirect chain terminates with an error, not a
    hang."""
    loop_ctx = lambda: _stream_ctx(302, {"location": "http://a.example/loop"})
    with patch("ghost_agent.tools.file_system.curl_requests", None), \
         patch("ghost_agent.tools.file_system.httpx.AsyncClient") as cls:
        client = MagicMock()
        cls.return_value.__aenter__ = AsyncMock(return_value=client)
        cls.return_value.__aexit__ = AsyncMock(return_value=None)
        client.stream = MagicMock(side_effect=lambda *a, **k: loop_ctx())

        res = await tool_download_file("http://a.example/loop", tmp_path, None, "out.bin")
        assert "too many redirects" in res.lower()


# --------------------------------------------- onion body-cap (streaming)


@pytest.mark.asyncio
async def test_onion_fetch_streams_with_byte_cap():
    """_fetch_raw_html must STOP reading at the cap instead of materializing a
    giant chunked body via r.text. Patches the curl_cffi Session (the default
    branch) to stream effectively-unbounded chunks."""
    import ghost_agent.tools.darkweb_search as D
    import curl_cffi.requests as creq

    consumed = {"chunks": 0}

    class _Resp:
        status_code = 200
        headers = {"content-type": "text/html"}  # no content-length (chunked)

        def iter_content(self):
            while True:  # unbounded — the byte cap must break the loop
                consumed["chunks"] += 1
                yield b"x" * (256 * 1024)

        def close(self):
            pass

    class _Session:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, *a, **k): return _Resp()

    with patch.object(creq, "Session", _Session):
        status, text = await D._fetch_raw_html("http://x.onion/", None, 5.0)

    assert status == 200
    assert len(text) <= D._MAX_ONION_BODY_BYTES  # capped, not unbounded
    assert consumed["chunks"] < 1000  # stopped early, didn't read forever
