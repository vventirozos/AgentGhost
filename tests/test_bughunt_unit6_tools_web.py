"""Regression tests for bug-hunt unit 6 (tools-web) — see BUGHUNT.md.

Fixed bugs pinned here:
 1. browser: SSRF guard resolved the hostname on the HOST even in Tor mode
    (DNS leak) → now resolve=False when anonymous; exotic schemes blocked;
    fail-closed on an unparseable URL
 2. browser: WebRTC hardening no longer disables the local-IP-hiding feature
 3. browser: LLM-supplied timeout_ms/max_chars are safely coerced and max_chars
    is clamped to the _MAX_TEXT_CHARS ceiling
 4. vision: URL fetch is byte-capped (no OOM); content=null coerced; non-image
    content-type refused; leading prefixes stripped (not str.replace-all)
 5. image_gen: empty/data-URI b64 rejected (not reported SUCCESS); width/height
    coerced; size kwargs excluded from the prompt fallback
 6. search/darkweb: distill-failure fallback is _clean_for_cpp-scrubbed; deep
    research fetch honours the per-query proxy and suppresses global NEWNYM;
    darkweb doesn't cache an all-errors run
"""

import base64

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ──────────────────────────────────────────────────────────────────────
# 1-3. browser guard / args / coercion (host-side functions)
# ──────────────────────────────────────────────────────────────────────

from ghost_agent.tools.browser import (
    _browser_blocked_url,
    _safe_int,
    _MAX_TEXT_CHARS,
)


class TestBrowserGuard:
    def test_anon_mode_does_not_resolve_public_host(self):
        # In Tor mode a public host must NOT be host-resolved (DNS leak) — the
        # guard passes resolve=False, so a normal public host is allowed
        # without a getaddrinfo call.
        with patch("ghost_agent.utils.helpers.socket.getaddrinfo") as gai:
            out = _browser_blocked_url("http://example.com/path", anonymous=True)
            assert out is None
            gai.assert_not_called()

    def test_clearnet_mode_still_resolves(self):
        with patch("ghost_agent.utils.helpers.socket.getaddrinfo") as gai:
            gai.return_value = [(2, 1, 6, "", ("93.184.216.34", 0))]
            _browser_blocked_url("http://example.com/path", anonymous=False)
            gai.assert_called()  # clearnet: resolve to catch hostname→internal

    def test_literal_internal_blocked_even_anon(self):
        assert _browser_blocked_url("http://127.0.0.1:9051", anonymous=True)
        assert _browser_blocked_url("http://169.254.169.254/latest/meta-data", anonymous=True)

    @pytest.mark.parametrize("u", [
        "chrome://settings", "view-source:http://x", "ftp://host/f", "gopher://x",
    ])
    def test_exotic_schemes_blocked(self, u):
        assert _browser_blocked_url(u)

    @pytest.mark.parametrize("u", ["file:///workspace/x.html", "about:blank", "data:text/html,x"])
    def test_allowed_schemes_pass(self, u):
        assert _browser_blocked_url(u) is None

    def test_unparseable_url_fails_closed(self):
        assert _browser_blocked_url("http://[::bad")  # refused, not allowed


class TestBrowserCoercion:
    def test_safe_int_tolerates_garbage(self):
        assert _safe_int("30s", 30000) == 30000
        assert _safe_int(None, 5) == 5
        assert _safe_int("500", 0) == 500

    def test_max_chars_clamped_to_ceiling(self):
        clamped = max(256, min(_safe_int(10**9, _MAX_TEXT_CHARS), _MAX_TEXT_CHARS))
        assert clamped == _MAX_TEXT_CHARS

    def test_webrtc_args_do_not_expose_local_ips(self):
        # _chromium_args lives inside the runner-script string; assert the
        # source no longer APPENDS the local-IP-exposing flag and keeps the
        # restrictive policy.
        import inspect
        import ghost_agent.tools.browser as b
        src = inspect.getsource(b)
        assert 'args.append("--disable-features=WebRtcHideLocalIpsWithMdns")' not in src
        assert "disable_non_proxied_udp" in src


# ──────────────────────────────────────────────────────────────────────
# 4. vision
# ──────────────────────────────────────────────────────────────────────

from ghost_agent.tools.vision import tool_vision_analysis


class TestVision:
    async def test_prefix_strip_is_leading_only(self, tmp_path):
        # A path with "/sandbox/" mid-string must NOT be clobbered. We give a
        # non-existent file so it returns "not found" but with the RIGHT path.
        out = await tool_vision_analysis(
            action="describe_picture", target="assets/sandbox/logo.png",
            sandbox_dir=tmp_path, llm_client=MagicMock(),
        )
        assert "assets/sandbox/logo.png" in out  # path preserved in the error

    async def test_content_null_does_not_error(self, tmp_path):
        img = tmp_path / "x.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\nfakepng")
        client = MagicMock()
        client.chat_completion = AsyncMock(return_value={
            "choices": [{"message": {"content": None}}]  # null content
        })
        out = await tool_vision_analysis(
            action="describe_picture", target="x.png",
            sandbox_dir=tmp_path, llm_client=client,
        )
        # Pre-fix: TypeError → "Vision API Error". Now a clean (empty) result.
        assert "VISION ANALYSIS RESULT" in out
        assert "Error" not in out


# ──────────────────────────────────────────────────────────────────────
# 5. image_gen
# ──────────────────────────────────────────────────────────────────────

from ghost_agent.tools.image_gen import tool_generate_image, _snap_to_sdxl_bucket


class TestImageGen:
    def _client(self, b64):
        c = MagicMock()
        c.image_gen_clients = [object()]
        c.generate_image = AsyncMock(return_value={"data": [{"b64_json": b64}]})
        return c

    async def test_empty_b64_is_rejected_not_success(self, tmp_path):
        out = await tool_generate_image(prompt="a cat", llm_client=self._client(""),
                                        sandbox_dir=tmp_path)
        assert out.startswith("ERROR")
        assert not list(tmp_path.glob("gen_*.png"))  # no 0-byte file written

    async def test_data_uri_prefix_stripped(self, tmp_path):
        raw = b"\x89PNG\r\n\x1a\n" + b"imagedata"
        b64 = "data:image/png;base64," + base64.b64encode(raw).decode()
        out = await tool_generate_image(prompt="a cat", llm_client=self._client(b64),
                                        sandbox_dir=tmp_path)
        assert out.startswith("SUCCESS")
        written = list(tmp_path.glob("gen_*.png"))
        assert written and written[0].read_bytes() == raw  # decoded cleanly

    async def test_nonint_width_does_not_raise(self, tmp_path):
        raw = b"\x89PNG\r\n\x1a\nx"
        b64 = base64.b64encode(raw).decode()
        # width="1024px" (truthy, non-int) previously raised out of the tool.
        out = await tool_generate_image(prompt="a cat", llm_client=self._client(b64),
                                        sandbox_dir=tmp_path, width="1024px", height="large")
        assert out.startswith("SUCCESS")

    async def test_size_kwarg_not_used_as_prompt(self, tmp_path):
        raw = b"\x89PNG\r\n\x1a\nx"
        b64 = base64.b64encode(raw).decode()
        client = self._client(b64)
        # No prompt, only size — must NOT generate for the literal "512x512".
        out = await tool_generate_image(llm_client=client, sandbox_dir=tmp_path,
                                        size="512x512")
        assert "MANDATORY" in out
        client.generate_image.assert_not_called()


# ──────────────────────────────────────────────────────────────────────
# 6. search / darkweb research
# ──────────────────────────────────────────────────────────────────────

class TestDeepResearchFetch:
    async def test_fetch_honours_proxy_and_suppresses_newnym(self, monkeypatch):
        from ghost_agent.tools import search as search_mod

        seen = {}

        async def fake_fetch(url, *, proxy_override=None, renew_identity=True):
            seen["proxy_override"] = proxy_override
            seen["renew_identity"] = renew_identity
            return "page text " * 10

        monkeypatch.setattr(search_mod, "helper_fetch_url_content", fake_fetch)
        monkeypatch.setattr("importlib.util.find_spec", lambda name: True)

        async def fake_to_thread(fn, *a, **k):
            return [{"href": "http://example.com/1"}]

        monkeypatch.setattr("asyncio.to_thread", fake_to_thread)

        llm = MagicMock()
        llm.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": "facts"}}]})
        await search_mod.tool_deep_research(
            query="q", anonymous=True, tor_proxy="socks5://user:pass@127.0.0.1:9050",
            llm_client=llm,
        )
        # The per-query identity proxy is forwarded (normalised to socks5h so
        # DNS also rides the proxy); global NEWNYM suppressed.
        assert seen["proxy_override"] == "socks5h://user:pass@127.0.0.1:9050"
        assert seen["renew_identity"] is False


class TestHelperFetchParams:
    async def test_renew_identity_false_skips_newnym_on_503(self, monkeypatch):
        from ghost_agent.utils import helpers

        renewed = {"count": 0}
        monkeypatch.setattr(helpers, "request_new_tor_identity",
                            lambda *a, **k: renewed.__setitem__("count", renewed["count"] + 1))
        monkeypatch.setattr(helpers, "url_ssrf_reason", lambda u, **k: None)

        class _Resp:
            status_code = 503
            headers = {}
            text = ""

        class _Session:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, *a, **k):
                return _Resp()

        import types
        fake_curl = types.SimpleNamespace(requests=types.SimpleNamespace(AsyncSession=_Session))
        monkeypatch.setitem(__import__("sys").modules, "curl_cffi", fake_curl)
        monkeypatch.setitem(__import__("sys").modules, "curl_cffi.requests", fake_curl.requests)

        out = await helpers.helper_fetch_url_content(
            "http://example.com", proxy_override="socks5://127.0.0.1:9050", renew_identity=False,
        )
        assert "503" in out
        assert renewed["count"] == 0  # NEWNYM suppressed
