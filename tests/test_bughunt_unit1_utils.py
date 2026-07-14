"""Regression tests for bug-hunt unit 1 (utils/) — see BUGHUNT.md.

One test class per fixed bug:
1. sanitizer: loose-fence scan is linear (no quadratic backtracking DoS)
2. sanitizer: fix_python_syntax never mutates already-valid code
3. egress_guard: multicast destinations are allowed (LAN discovery)
4. stylometry: scrub_query never drops leading content keywords
5. token_counter: short-token cache is thread-safe under eviction
6. helpers: binary short-circuit checks the URL *path* (query strings)
7. logging: setup_logging tolerates a bare filename (no dirname)
8. telemetry: HostSnapshot.healthy honours the disk threshold
"""

import math
import threading
import time

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from ghost_agent.utils.sanitizer import (
    extract_code_from_markdown,
    fix_python_syntax,
    sanitize_code,
    _loose_fence_bodies,
)
from ghost_agent.utils.egress_guard import is_allowed_host
from ghost_agent.utils.stylometry import scrub_query
from ghost_agent.utils import token_counter
from ghost_agent.utils.helpers import helper_fetch_url_content
from ghost_agent.utils.logging import setup_logging
from ghost_agent.utils.telemetry import HostSnapshot


# ──────────────────────────────────────────────────────────────────────
# 1. Loose-fence scan must be linear, not quadratic
# ──────────────────────────────────────────────────────────────────────

class TestFenceScanLinear:
    def test_unclosed_fence_with_huge_space_run_is_fast(self):
        # Pre-fix: the lazy-body regex backtracked quadratically on an
        # opening fence followed by spaces and NO closing fence
        # (16K spaces ≈ 1.2s, ~minutes at the 200KB cap).
        text = "```" + " " * 190_000
        t0 = time.monotonic()
        extract_code_from_markdown(text)
        assert time.monotonic() - t0 < 2.0

    def test_trailing_open_fence_after_valid_block_is_fast(self):
        # The blowup also triggered on the tail AFTER a valid block.
        text = "```python\nx = 1\n```\n```" + " " * 150_000
        t0 = time.monotonic()
        out = extract_code_from_markdown(text)
        assert time.monotonic() - t0 < 2.0
        assert "x = 1" in out

    def test_loose_scan_equivalent_to_old_regex_semantics(self):
        # Must match the old lazy-body regex exactly (including the greedy
        # language-tag run eating a first bare word — "inline"/"a" below).
        import re
        old = re.compile(r'```[ \t]*[a-zA-Z0-9_.+-]*[ \t\n]?(.*?)```', re.DOTALL)
        cases = [
            "``` inline body ```",
            "```python\ncode\n```",
            "```a``` mid ```b```",
            "no fences here",
            "```abc",
            "```py\nx=1\n```\ntail ```\nrest\n```",
            "```\n\ncode```",
            "a```b```c```d```",
        ]
        for text in cases:
            assert _loose_fence_bodies(text) == old.findall(text), text

    def test_extraction_still_picks_longest_block(self):
        text = "```\nshort\n``` prose ```python\nthe = 'real payload line'\nsecond = 2\n```"
        out = extract_code_from_markdown(text)
        assert "real payload" in out
        assert "short" not in out


# ──────────────────────────────────────────────────────────────────────
# 2. fix_python_syntax must not mutate valid code
# ──────────────────────────────────────────────────────────────────────

class TestValidCodeUntouched:
    @pytest.mark.parametrize("code", [
        'msg = "Ready?Set?Go?Now"',
        'url = "https://x/a?b?c?d"',
        "pat = r'foo?bar?baz?qux'",
        'q = "???"',
    ])
    def test_stutter_shapes_inside_valid_literals_survive(self, code):
        # Pre-fix: the ?-stutter strip ran before any validity check and
        # silently corrupted valid string literals; the result still
        # parsed, so the corruption was committed and executed.
        assert fix_python_syntax(code) == code

    def test_sanitize_code_end_to_end_preserves_valid_literal(self):
        code = 'msg = "Ready?Set?Go?Now"\nprint(msg)'
        out, err = sanitize_code(code, "script.py")
        assert err is None
        assert 'Ready?Set?Go?Now' in out

    def test_broken_code_still_healed(self):
        healed = fix_python_syntax('x = "abc\nprint(x)')
        assert healed == 'x = "abc"\nprint(x)'

    def test_stutter_still_stripped_from_broken_code(self):
        out = fix_python_syntax("import os?????")
        assert "????" not in out


# ──────────────────────────────────────────────────────────────────────
# 3. Multicast egress is LAN discovery, not public egress
# ──────────────────────────────────────────────────────────────────────

class TestMulticastAllowed:
    @pytest.mark.parametrize("addr", [
        "224.0.0.251",        # mDNS
        "239.255.255.250",    # SSDP
        "ff02::fb",           # mDNS v6
        "ff02::1",            # all-nodes link-local
    ])
    def test_multicast_allowed(self, addr):
        # Pre-fix: CPython reports multicast as is_global=True, so the
        # guard raised MandatoryTorError on mDNS/SSDP — traffic the
        # module docstring explicitly permits.
        assert is_allowed_host(addr) is True

    def test_public_unicast_still_blocked(self):
        assert is_allowed_host("8.8.8.8") is False
        assert is_allowed_host("2001:4860:4860::8888") is False

    def test_private_and_loopback_still_allowed(self):
        assert is_allowed_host("192.168.1.10") is True
        assert is_allowed_host("127.0.0.1") is True


# ──────────────────────────────────────────────────────────────────────
# 4. scrub_query keeps content keywords
# ──────────────────────────────────────────────────────────────────────

class TestScrubKeepsContent:
    @pytest.mark.parametrize("query", [
        "search algorithms in python",
        "lookup table sql tutorial",
        "thank you card template",
    ])
    def test_leading_content_keywords_survive(self, query):
        # Pre-fix these lost their first token ("search", "lookup",
        # "thank") — violating the never-drop-a-content-keyword contract.
        assert scrub_query(query) == query

    def test_framed_forms_still_stripped(self):
        assert scrub_query("search for rust tutorials please") == "rust tutorials"
        assert scrub_query("look up mitochondria function") == "mitochondria function"
        assert scrub_query("can you please find me a plumber thanks!") == "a plumber"

    def test_trailing_gratitude_stripped(self):
        assert scrub_query("best gpu under 500 thanks") == "best gpu under 500"
        assert scrub_query("best gpu under 500 thank you") == "best gpu under 500"

    def test_idempotent_on_content_gratitude(self):
        q = "thank you card template"
        assert scrub_query(scrub_query(q)) == scrub_query(q)


# ──────────────────────────────────────────────────────────────────────
# 5. Token cache thread safety
# ──────────────────────────────────────────────────────────────────────

class TestTokenCacheThreadSafety:
    def test_concurrent_estimates_do_not_race(self, monkeypatch):
        # Tiny cap → every insert evicts → maximum iterate/mutate overlap.
        # Pre-fix the unlocked next(iter(dict)) eviction could raise
        # "dictionary changed size during iteration" under concurrency.
        monkeypatch.setattr(token_counter, "_SHORT_TOKEN_CACHE", {})
        monkeypatch.setattr(token_counter, "_SHORT_TOKEN_CACHE_MAX", 4)
        errors = []

        def hammer(tag):
            try:
                for i in range(3000):
                    token_counter.estimate_tokens(f"{tag}-{i}-payload")
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=hammer, args=(t,)) for t in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert len(token_counter._SHORT_TOKEN_CACHE) <= 5


# ──────────────────────────────────────────────────────────────────────
# 6. Binary short-circuit must consider the URL path, not the raw URL
# ──────────────────────────────────────────────────────────────────────

class TestBinaryUrlPathCheck:
    @pytest.mark.asyncio
    async def test_pdf_with_query_string_short_circuits(self):
        with patch("ghost_agent.utils.helpers.url_ssrf_reason", lambda u, **k: None), \
             patch("ghost_agent.utils.helpers.httpx.AsyncClient") as mock_client_cls, \
             patch.dict("sys.modules", {"curl_cffi": None, "curl_cffi.requests": None}):
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            # Server lies: generic content-type on a PDF URL.
            mock_resp.headers = {"content-type": "text/html"}
            mock_resp.text = "%PDF-1.7 binary garbage"
            mock_client.get.return_value = mock_resp

            result = await helper_fetch_url_content("http://example.com/report.pdf?dl=1")
            assert "binary file" in result


# ──────────────────────────────────────────────────────────────────────
# 7. setup_logging with a bare filename
# ──────────────────────────────────────────────────────────────────────

class TestSetupLoggingBareFilename:
    def test_bare_filename_does_not_crash(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # Pre-fix: os.makedirs("") raised FileNotFoundError.
        setup_logging("bare.log")
        assert (tmp_path / "bare.log").exists()


# ──────────────────────────────────────────────────────────────────────
# 8. HostSnapshot.healthy honours disk
# ──────────────────────────────────────────────────────────────────────

class TestSnapshotHealthyDisk:
    def _snap(self, disk):
        return HostSnapshot(ts=time.time(), cpu_percent=10.0, mem_percent=30.0,
                            mem_available_mb=4000.0, disk_percent=disk,
                            proc_count=100)

    def test_full_disk_is_unhealthy(self):
        # Pre-fix: healthy ignored disk entirely while the poller warned
        # at DEFAULT_DISK_HIGH — the two access patterns disagreed.
        assert self._snap(95.0).healthy is False

    def test_normal_disk_is_healthy(self):
        assert self._snap(20.0).healthy is True

    def test_nan_disk_does_not_flag(self):
        assert self._snap(math.nan).healthy is True
