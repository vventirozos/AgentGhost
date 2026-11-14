"""§4UN-fetch (operator 2026-11-13, internet-tools review): downloads and page fetches that report what really
happened and keep what is not the content out.

* the download's retries ride a fresh SOCKS-isolated circuit each (the old Tor "identity rotation" was a no-op:
  control port closed, agent unprivileged) and 429 is retried; failures are reported by kind (Tor down, timeout,
  stall, cut at N MB) — "refuses Tor exits" only after refusals; a size-scaled wall clock and a stall timer;
* the receipt checks the bytes against the name's type for every name (JSON error as .jpg, HTML as data.csv,
  HTML after leading comments); a URL name with no extension takes Content-Disposition / Content-Type;
* the shared page fetch: plain-identity retry on 403/429, the page's own <meta charset>, main text only,
  challenge/consent/JS-only walls → BLOCKED; the quarantine's clock scales and a page body is PARTIAL;
* browser: "Download is starting" advice, more challenge titles, no automation flag."""
from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from ghost_agent.tools import file_system as FS
from ghost_agent.utils import content_check as CC
from ghost_agent.utils import helpers as H

TOR = "socks5://127.0.0.1:9050"


# ───────────────────────────── content_check ─────────────────────────────

@pytest.mark.parametrize("name,ct,head,bad", [
    ("photo.jpg", "image/jpeg", b'{"error":"rate limited"}', "JSON"),
    ("report.pdf", "application/pdf", b" " * 600 + b"<!-- cdn -->\n<!doctype html><html>", "HTML page"),
    ("data.csv", "text/html", b"<!DOCTYPE html><html><head>", "HTML page"),
    ("pic.avif", "text/html", b"<html><body>denied", "HTML page"),
    ("deck.pptx", "text/html", b"<html>", "HTML page"),
    ("paper.pdf", "application/pdf", b"Access denied", "plain text"),
    ("empty.zip", "application/zip", b"", "EMPTY"),
])
def test_a_body_that_is_not_the_named_file_is_named(name, ct, head, bad):
    assert bad in CC.body_mismatch(name, ct, head, total=len(head))


@pytest.mark.parametrize("name,ct,head", [
    ("report.pdf", "application/pdf", b"%PDF-1.7\n"),
    ("data.csv", "text/csv", b"a,b\n1,2\n"),
    ("download.bin", "text/plain", b"hello there"),          # a nameless URL's default name holds anything
    ("notes.md", "text/markdown", b"# <html> in a heading"),
    ("page.html", "text/html", b"<!doctype html><html>"),
    ("song.mp3", "audio/mpeg", b"ID3\x04\x00binary"),
])
def test_the_real_file_passes(name, ct, head):
    assert CC.body_mismatch(name, ct, head, total=len(head)) == ""


def test_another_binary_type_is_a_note_not_a_failure():
    assert CC.kind_note("x.jpg", b"\x89PNG\r\n\x1a\n" + b"x" * 20) == "the bytes are png, not .jpg"
    assert CC.kind_note("x.jpg", b"\xff\xd8\xff\xe0") == ""


@pytest.mark.parametrize("title,text,url,want", [
    ("Making sure you're not a bot!", "Making sure you're not a bot! Loading…", "https://git.example/", "bot challenge"),
    ("Before you continue to YouTube", "Before you continue to YouTube We use cookies and data to deliver and "
     "maintain Google services", "https://consent.youtube.com/m?continue=x", "consent wall"),
    ("App", "You need to enable JavaScript to run this app.", "https://spa.example/", "JavaScript-only"),
    ("Request unsuccessful", "Request unsuccessful. Incapsula incident ID: 123", "https://x.example/", "bot challenge"),
    ("DDoS-Guard", "Checking your browser before accessing x.example", "https://x.example/", "bot challenge"),
])
def test_walls_in_front_of_the_content_are_named(title, text, url, want):
    assert want in CC.challenge_reason(200, title, text, url)


def test_an_article_about_bot_checks_is_not_a_wall():
    long = ("Cloudflare's 'Just a moment' page and Anubis' 'Making sure you're not a bot' gate are both "
            "proof-of-work or passive checks. ") * 60
    assert CC.challenge_reason(200, "How bot walls work", long, "https://blog.example/") == ""
    assert CC.challenge_reason(200, "Before You Continue Reading: Our Guide", "short", "https://x/") == ""


def test_the_main_text_drops_menus_banners_and_forms():
    html = (b'<html><head><title>T</title></head><body><header>SITE MENU</header><nav>Home News</nav>'
            b'<div id="cookie-banner">We use cookies</div><form>Search</form><article><h1>Big news</h1><p>'
            + b"The real sentence of the story. " * 30 + b'</p></article><aside>Related links</aside>'
            b'<footer>Copyright</footer></body></html>')
    t = CC.extract_main_text(html)
    assert t.startswith("Big news The real sentence")
    for junk in ("SITE MENU", "Home News", "cookies", "Search", "Related links", "Copyright"):
        assert junk not in t


def test_a_page_without_an_article_keeps_its_body_and_a_boilerplate_only_page_keeps_its_text():
    t = CC.extract_main_text("<html><body><nav>menu</nav><p>Short answer: 42.</p></body></html>")
    assert t == "Short answer: 42."
    assert CC.extract_main_text("<html><body><header>Only a header here</header></body></html>") == \
        "Only a header here"


def test_page_meta_reads_title_and_dates():
    html = ('<html><head><title>Fallback</title><meta property="og:title" content="Release notes 19">'
            '<meta property="article:published_time" content="2026-03-02T10:00:00Z">'
            '<script type="application/ld+json">{"@graph":[{"dateModified":"2026-04-05"}]}</script>'
            '</head><body></body></html>')
    m = CC.page_meta(html)
    assert m == {"title": "Release notes 19", "published": "2026-03-02", "modified": "2026-04-05",
                 "date": "2026-04-05"}
    assert CC.page_meta("<html><title>x</title></html>", {"Last-Modified": "Wed, 21 Oct 2025 07:28:00 GMT"})[
        "date"] == "2025-10-21"
    assert CC.page_meta("<html><title>x</title></html>")["date"] == ""


def test_pdf_text_reads_pages_and_refuses_non_pdfs():
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    for i in range(3):
        doc.new_page().insert_text((72, 72), f"Page {i + 1} says hello")
    data = doc.tobytes()
    assert CC.pdf_text(data) == "Page 1 says hello\nPage 2 says hello\nPage 3 says hello"
    assert CC.pdf_text(data, max_pages=1) == "Page 1 says hello"
    assert CC.pdf_text(b"<html>not a pdf") == ""


def test_the_charset_comes_from_the_header_then_the_page_never_a_guess():
    assert H.page_charset("text/html; charset=ISO-8859-7", b"") == "ISO-8859-7"
    assert H.page_charset("text/html", b'<meta charset="windows-1253">') == "windows-1253"
    assert H.page_charset("text/html", b'<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-7">') \
        == "iso-8859-7"
    assert H.page_charset("text/html", b"<html>") == "utf-8"
    assert H.page_charset("text/html", b'<meta charset="no-such-codec">') == "utf-8"


# ───────────────────────────── the download ─────────────────────────────

class _Resp:
    def __init__(self, status, headers=None, body=b"", chunks=None, raise_after=None, hang_after=None, delay=0.0):
        self.status_code, self.headers, self._body = status, headers or {}, body
        self._chunks, self._raise_after, self._hang_after = chunks, raise_after, hang_after
        self._delay = delay

    async def aiter_content(self):
        for i, c in enumerate(self._chunks or [self._body]):
            if self._hang_after is not None and i >= self._hang_after:
                await asyncio.sleep(3600)
            if self._delay:
                await asyncio.sleep(self._delay)
            yield c
            if self._raise_after is not None and i + 1 >= self._raise_after:
                raise ConnectionError("Connection reset by peer")

    async def aclose(self):
        pass


class _Session:
    sessions = []
    responder = None

    def __init__(self, *a, **k):
        _Session.sessions.append(k)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, **k):
        r = _Session.responder(len(_Session.sessions) - 1, url)
        if isinstance(r, BaseException):
            raise r
        return r


def _download(tmp_path, responder, *, url="https://files.example/report.pdf", filename="report.pdf", tor=TOR,
              name_from_url=False):
    _Session.sessions, _Session.responder = [], responder
    rotations = []
    with patch.object(FS, "curl_requests", SimpleNamespace(AsyncSession=_Session)), \
         patch("ghost_agent.utils.egress_guard.resolve_egress_proxy", lambda tor_proxy, url=None: tor_proxy), \
         patch.object(FS, "request_new_tor_identity", lambda: rotations.append(1)):
        out = asyncio.run(FS.tool_download_file(url, tmp_path, tor, filename=filename, name_from_url=name_from_url))
    return str(out), rotations


def test_each_attempt_rides_its_own_circuit_and_a_429_is_retried(tmp_path):
    def responder(i, url):
        return _Resp(429, {"server": "nginx"}) if i < 2 else _Resp(200, {"Content-Length": "8"}, b"%PDF-1.7")
    out, rotations = _download(tmp_path, responder)
    assert out.startswith("SUCCESS"), out
    assert rotations == []                                   # never the no-op daemon rotation
    px = [str((k.get("proxies") or {}).get("https")) for k in _Session.sessions]
    assert len(px) == 3 and len(set(px)) == 3 and all("127.0.0.1:9050" in p for p in px), px
    assert _Session.sessions[1].get("impersonate") is None   # the plain identity after the first refusal


def test_a_localhost_socks_proxy_is_tor_mode(tmp_path):
    out, _ = _download(tmp_path, lambda i, u: _Resp(403, {"server": "cloudflare"}), tor="socks5h://localhost:9050")
    assert len(_Session.sessions) == 3 and "over Tor" in out and "refuses this agent from Tor exits" in out


def test_tor_down_is_said_at_once_not_blamed_on_the_site(tmp_path):
    err = Exception("Failed to perform, curl: (7) Failed to connect to 127.0.0.1 port 9050 after 0 ms: "
                    "Couldn't connect to server")
    out, _ = _download(tmp_path, lambda i, u: err)
    assert "Tor is not reachable on this machine" in out and len(_Session.sessions) == 1
    assert "refuses" not in out


def test_timeouts_and_a_cut_transfer_are_not_called_refusals(tmp_path):
    out, _ = _download(tmp_path, lambda i, u: Exception("curl: (28) Operation timed out after 60000 ms"))
    assert "did not answer in time" in out and "refuses" not in out and "did not refuse" in out
    big = [b"%PDF-" + b"x" * 999_995] + [b"y" * 1_000_000] * 3
    out, _ = _download(tmp_path, lambda i, u: _Resp(200, {"Content-Length": "9000000"}, chunks=big, raise_after=3))
    assert "cut in the middle of the file" in out and "refuses" not in out
    assert not (tmp_path / "report.pdf").exists()            # nothing partial left behind


def test_a_stalled_transfer_ends_on_the_stall_timer(tmp_path, monkeypatch):
    monkeypatch.setattr(FS, "_DOWNLOAD_STALL_S", 0.2)
    out, _ = _download(tmp_path, lambda i, u: _Resp(200, {}, chunks=[b"%PDF-1", b"x"], hang_after=1))
    assert "stalled" in out and "refuses" not in out


def test_the_wall_clock_scales_with_the_size():
    assert FS._download_deadline_s(None) == FS._DOWNLOAD_MAX_S       # unknown size: the stall timer + the cap
    assert FS._download_deadline_s(5_000_000) == pytest.approx(60 + 100)
    assert FS._download_deadline_s(10**9) == FS._DOWNLOAD_MAX_S


def test_a_malformed_content_length_is_unknown_not_a_crash(tmp_path):
    out, _ = _download(tmp_path, lambda i, u: _Resp(200, {"Content-Length": "8, 8"}, b"%PDF-1.7"))
    assert out.startswith("SUCCESS"), out


@pytest.mark.parametrize("name,ct,body", [
    ("photo.jpg", "image/jpeg", b'{"error":"rate limited"}'),
    ("data.csv", "text/html", b"<!DOCTYPE html><html><body>Sign in</body></html>"),
])
def test_a_wrong_body_is_partial_never_success(tmp_path, name, ct, body):
    out, _ = _download(tmp_path, lambda i, u: _Resp(200, {"Content-Type": ct}, body),
                       url=f"https://files.example/{name}", filename=name)
    assert out.startswith("PARTIAL") and "Do not use" in out, out


def test_a_nameless_pdf_takes_the_content_type_extension_and_the_final_url_is_said(tmp_path):
    def responder(i, url):
        if url.endswith("/abs/2401.12345"):
            return _Resp(302, {"location": "https://arxiv.example/pdf/2401.12345"})
        return _Resp(200, {"Content-Type": "application/pdf"}, b"%PDF-1.7 body")
    out, _ = _download(tmp_path, responder, url="https://arxiv.example/abs/2401.12345", filename="2401.12345",
                       name_from_url=True)
    assert "to '2401.12345.pdf'" in out and (tmp_path / "2401.12345.pdf").exists(), out
    assert "Final URL after redirects: https://arxiv.example/pdf/2401.12345" in out


def test_content_disposition_names_a_nameless_download_but_never_a_chosen_name(tmp_path):
    hdr = {"Content-Type": "application/octet-stream", "Content-Disposition": 'attachment; filename="Q3 report.pdf"'}
    out, _ = _download(tmp_path, lambda i, u: _Resp(200, hdr, b"%PDF-1.7"), url="https://x.example/get?id=7",
                       filename="get", name_from_url=True)
    assert "to 'Q3 report.pdf'" in out, out
    out, _ = _download(tmp_path, lambda i, u: _Resp(200, hdr, b"%PDF-1.7"), url="https://x.example/get?id=7",
                       filename="mine", name_from_url=False)
    assert "to 'mine'" in out, out


# ───────────────────────────── the shared page fetch ─────────────────────────────

def _fetch(responses, url="https://news.example/story"):
    calls = []

    class _S:
        def __init__(self, *a, **k):
            calls.append(k)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, *a, **k):
            r = responses.pop(0)
            if isinstance(r, BaseException):
                raise r
            return r

    from tests.conftest import make_streaming_resp  # noqa: F401  (shape reference)
    fake = types.SimpleNamespace(requests=types.SimpleNamespace(AsyncSession=_S))
    with patch("ghost_agent.utils.helpers.url_ssrf_reason", lambda u, **k: None), \
         patch.dict(sys.modules, {"curl_cffi": fake, "curl_cffi.requests": fake.requests}), \
         patch("ghost_agent.utils.helpers.asyncio.sleep", AsyncMock()):
        out = asyncio.run(H.helper_fetch_url_content(url, proxy_override="socks5h://127.0.0.1:9050"))
    return out, calls


def _page(status, body, ct="text/html"):
    from tests.conftest import make_streaming_resp
    return make_streaming_resp(status, body, content_type=ct)


def test_a_challenge_page_is_a_blocked_fetch_not_content():
    out, _ = _fetch([_page(200, "<html><head><title>Just a moment...</title></head><body>Checking your browser "
                                "before accessing news.example</body></html>")])
    assert isinstance(out, H.FetchError) and out.startswith("Error: BLOCKED") and "bot challenge" in out


def test_a_429_retries_with_the_plain_identity_on_a_fresh_circuit():
    out, calls = _fetch([_page(429, "slow down"), _page(200, "<html><body><p>The story text.</p></body></html>")])
    assert out == "The story text."
    assert calls[0].get("impersonate") == "chrome" and calls[1].get("impersonate") is None
    assert calls[0]["proxies"]["https"] != calls[1]["proxies"]["https"]


def test_a_greek_page_is_decoded_by_its_meta_charset():
    body = '<html><head><meta charset="windows-1253"></head><body><p>Καλημέρα κόσμε</p></body></html>'
    out, _ = _fetch([_page(200, body.encode("cp1253"), ct="text/html")])
    assert out == "Καλημέρα κόσμε"


def test_the_fetch_returns_the_main_text_without_menus():
    body = ("<html><body><nav>Home Sports Weather</nav><main><p>" + "The actual article sentence. " * 30 +
            "</p></main><footer>About us</footer></body></html>")
    out, _ = _fetch([_page(200, body)])
    assert out.startswith("The actual article sentence.") and "Home Sports" not in out and "About us" not in out


# ───────────────────────────── quarantine ─────────────────────────────

def test_the_quarantine_clock_scales_with_the_file():
    from ghost_agent.tools import quarantine as Q
    assert Q.fetch_deadline_s(None) == Q.FETCH_UNKNOWN_SIZE_S
    assert Q.fetch_deadline_s(25_000_000) == pytest.approx(Q.FETCH_TIMEOUT_S + 1000)
    assert Q.fetch_deadline_s(50_000_000) == Q.FETCH_MAX_S


async def test_an_onion_page_instead_of_the_file_is_partial(tmp_path, monkeypatch):
    from ghost_agent.tools import quarantine as Q
    from ghost_agent.utils.logging import request_id_context
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    (tmp_path / "sandbox").mkdir()

    async def _s(url, proxy, tmp):
        Path(tmp).write_bytes(b"<!DOCTYPE html><html><body>Login required</body></html>")
        return None, url, 55
    monkeypatch.setattr(Q, "_stream_download", _s)
    tok = request_id_context.set("web-test-owner")
    try:
        out = await Q.quarantine_download("http://" + "a" * 56 + ".onion/f.zip", TOR)
    finally:
        request_id_context.reset(tok)
    assert str(out).startswith("PARTIAL") and "not the file" in str(out)


# ───────────────────────────── browser ─────────────────────────────

def test_a_download_navigation_gets_the_download_advice():
    from ghost_agent.tools.browser_support import _runner_failure_hint, browser_outcome_note
    hint = _runner_failure_hint("Page.goto: Download is starting\nCall log: navigating to https://x/a.pdf")
    assert "file_system(operation='download')" in hint
    assert browser_outcome_note("Page.goto: Download is starting") == "the address is a download, not a page"


def test_the_new_walls_are_challenge_titles_and_the_runner_drops_the_automation_flag():
    from ghost_agent.tools.browser_routes import blocked_page_reason
    for title in ("Making sure you're not a bot!", "Before you continue to YouTube", "Request unsuccessful",
                  "One more step", "Pardon Our Interruption"):
        assert blocked_page_reason({"status": 200, "title": title, "url": "https://x/", "length": 300}), title
    assert not blocked_page_reason({"status": 200, "title": "Before You Continue Reading: Our Guide",
                                    "url": "https://x/", "length": 300})
    pytest.importorskip("playwright")
    from ghost_agent.tools import browser_runner as R
    assert "--disable-blink-features=AutomationControlled" in R._chromium_args(None)
    assert R._PASSING_CHALLENGE_RE.match("Just a moment...") and not R._PASSING_CHALLENGE_RE.match("News today")


# ───────────────────────────── review round 1 (§4UN-fetch r1) ─────────────────────────────

# MAJOR 1 — wrappers whose class merely CONTAINS a boilerplate word held the article

@pytest.mark.parametrize("wrapper", [
    '<div class="cmp-container"><div class="cmp-text">{}</div></div>',                    # AEM components
    '<div id="primary" class="content-area has-sidebar">{}</div>',                          # WordPress
    '<div class="subscriber-content">{}</div>',
    '<div class="paywall">{}</div>',
    '<div class="post-related-tags-wrap entry">{}</div>',
])
def test_r1_a_wrapper_named_like_boilerplate_keeps_the_story(wrapper):
    story = "<h1>Port strike ends</h1><p>" + "Dockworkers returned to work on Monday after a deal. " * 20 + "</p>"
    html = "<html><body>" + wrapper.format(story) + "<footer>Copyright 2026 Agency</footer></body></html>"
    t = CC.extract_main_text(html)
    assert "Dockworkers returned" in t and "Copyright" not in t, t[:120]


def test_r1_r4_a_consent_banner_goes_and_class_named_blocks_stay():
    # r4: the class-name rules (share bars, related posts, sidebars, modals) were removed — a story next to
    # them may keep their text; only a short consent banner goes
    html = ('<html><body><div class="cookie-banner">We use cookies</div><div class="share-bar">Share</div>'
            '<div class="related-posts">Other story</div><p>' + "Real text here. " * 40 + "</p></body></html>")
    t = CC.extract_main_text(html)
    assert "Real text here." in t and "cookies" not in t


def test_r1_a_cleaned_page_under_a_fifth_of_the_raw_text_falls_back_to_the_raw_text():
    html = ('<html><body><div class="modal"><p>' + "The only real paragraph sits in a modal-classed div. " * 20
            + "</p></div><p>Tiny.</p></body></html>")
    assert "The only real paragraph" in CC.extract_main_text(html)


# MAJOR 2 — a short real page whose title starts like a wall, or that quotes a wall phrase

@pytest.mark.parametrize("title", ["Cloudflare outage takes down X", "CAPTCHA farms are booming",
                                   "Access denied: court rules on the data request"])
def test_r1_a_short_news_page_with_a_wall_like_title_is_read(title):
    text = ("The story begins here and runs for a while with real reporting about the event. " * 14)[:1100]
    assert CC.challenge_reason(200, title, text, "https://news.example/a") == ""


def test_r1_a_page_quoting_a_wall_phrase_deep_in_its_text_is_read():
    text = ("Scammers send fake prompts to trick people. " * 22) + ' One read "Verify you are human" and asked ' \
           "for a phone number. " + "More reporting follows. " * 10
    assert len(text) > 800
    assert CC.challenge_reason(200, "Fake prompts spread", text, "https://news.example/b") == ""
    assert CC.challenge_reason(200, "React docs", "Intro " * 200 + "You need to enable JavaScript to run this "
                               "app. " + "More " * 50, "https://docs.example/") == ""


@pytest.mark.parametrize("title,text", [
    ("Cloudflare", "Please wait"), ("Attention Required! | Cloudflare", "Sorry, you have been blocked"),
    ("Captcha", "Type the characters"), ("Just a moment...", "x" * 300),
    ("Shop", "Verify you are human by completing the action below. " + "x" * 900),
])
def test_r1_real_walls_are_still_walls(title, text):
    assert CC.challenge_reason(200, title, text, "https://shop.example/") != ""


# MAJOR 3 — one clock across attempts; a slow transfer that made progress is not retried

def test_r1_a_transfer_too_slow_for_its_size_is_reported_once_not_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(FS, "_DOWNLOAD_BASE_S", 0.4)
    monkeypatch.setattr(FS, "_DOWNLOAD_MIN_RATE", 10**12)
    chunks = [b"%PDF-" + b"x" * 99_995] + [b"y" * 100_000] * 89
    out, _ = _download(tmp_path, lambda i, u: _Resp(200, {"Content-Length": "9000000"}, chunks=chunks, delay=0.1))
    assert len(_Session.sessions) == 1, out
    assert "too slow over Tor" in out and "of 9.0 MB" in out and "refuses" not in out


def test_r1_attempts_share_one_clock(tmp_path, monkeypatch):
    monkeypatch.setattr(FS, "_DOWNLOAD_MAX_S", 1.0)      # stall 0.3 + pause 0.5 + 0.2 left for attempt 2
    monkeypatch.setattr(FS, "_DOWNLOAD_STALL_S", 0.3)
    out, _ = _download(tmp_path, lambda i, u: _Resp(200, {}, chunks=[b"x"], hang_after=0))
    assert len(_Session.sessions) == 2, (len(_Session.sessions), out)     # the 3rd attempt had no time left
    assert out.startswith("Error") and "refuses" not in out


def test_r1_an_unknown_size_file_streaming_steadily_is_not_cut(tmp_path, monkeypatch):
    monkeypatch.setattr(FS, "_DOWNLOAD_UNKNOWN_SIZE_S", 0.3, raising=False)     # the old fixed budget
    monkeypatch.setattr(FS, "_DOWNLOAD_STALL_S", 0.5)
    chunks = [b"%PDF-1.7"] + [b"z" * 1000] * 8
    out, _ = _download(tmp_path, lambda i, u: _Resp(200, {}, chunks=chunks, delay=0.08))
    assert out.startswith("SUCCESS"), out


def test_r1_the_stream_low_speed_limit_is_not_longer_than_the_stall_timer(tmp_path):
    _download(tmp_path, lambda i, u: _Resp(200, {}, b"%PDF-1.7"))
    assert _Session.sessions[0]["timeout"] <= FS._DOWNLOAD_STALL_S


# MAJOR 4 — a server-derived name must not overwrite an unread workspace file

def test_r1_a_server_name_never_overwrites_an_unread_file(tmp_path):
    (tmp_path / "2401.12345.pdf").write_bytes(b"%PDF-1.4 the user's own annotated copy")
    out, _ = _download(tmp_path, lambda i, u: _Resp(200, {"Content-Type": "application/pdf"}, b"%PDF-1.7 new"),
                       url="https://arxiv.example/pdf/2401.12345", filename="2401.12345", name_from_url=True)
    assert (tmp_path / "2401.12345.pdf").read_bytes() == b"%PDF-1.4 the user's own annotated copy"
    assert out.startswith("SUCCESS") and "to '2401.12345-2.pdf'" in out, out
    assert (tmp_path / "2401.12345-2.pdf").read_bytes() == b"%PDF-1.7 new"


def test_r1_a_script_name_takes_the_servers_type():
    hdr = {"Content-Type": "application/pdf"}
    assert FS._name_from_response("get.php", hdr) == "get.pdf"
    assert FS._name_from_response("file.aspx", {"Content-Disposition": 'attachment; filename="r.xlsx"'}) == "r.xlsx"
    assert FS._name_from_response("page.php", {"Content-Type": "text/html"}) == "page.php"
    assert FS._name_from_response("report.pdf", hdr) == "report.pdf"


# MAJOR 5 — a real .xls/.doc that is an HTML table or TSV is the file

@pytest.mark.parametrize("name,head", [
    ("export.xls", b'<html xmlns:x="urn:schemas-microsoft-com:office:excel"><body><table><tr><td>1</td>'),
    ("export.xls", b"name\tqty\nbolts\t40\nnuts\t25\n"),
    ("letter.doc", b'<html xmlns:w="urn:schemas-microsoft-com:office:word"><body><p>Dear Elena,</p>'),
    ("letter.doc", b"{\\rtf1\\ansi Dear Elena,}"),
])
def test_r1_html_or_text_office_exports_are_the_file(name, head):
    assert CC.body_mismatch(name, "application/vnd.ms-excel", head, total=len(head)) == ""
    assert CC.kind_note(name, head)            # said, not refused


def test_r1_a_wall_under_an_office_name_is_still_partial():
    head = b"<html><head><title>Just a moment...</title></head><body>Checking your browser</body></html>"
    assert "HTML page" in CC.body_mismatch("export.xls", "text/html", head, total=len(head))


# minors

def test_r1_page_charset_quotes_bom_and_clean_utf8():
    assert H.page_charset('text/html; charset="ISO-8859-7"', b"") == "ISO-8859-7"
    assert H.page_charset("text/html; charset=windows-1253", b"\xef\xbb\xbf<p>x</p>") == "utf-8-sig"
    utf8 = '<meta charset="iso-8859-1"><p>Καλημέρα</p>'.encode("utf-8")
    assert H.page_charset("text/html", utf8) == "utf-8"
    assert H.page_charset("text/html", '<meta charset="windows-1253"><p>Καλημέρα</p>'.encode("cp1253")) \
        == "windows-1253"


@pytest.mark.parametrize("url,body,partial", [
    ("feed.xml", b'<?xml version="1.0"?><rss><channel></channel></rss>', False),
    ("logo.svg", b'<svg xmlns="http://www.w3.org/2000/svg"></svg>', False),
    ("dump.zip", b"<!DOCTYPE html><html><body>Login</body></html>", True),
    ("download?id=3", b"<!DOCTYPE html><html><body>Login</body></html>", True),
])
async def test_r1_the_quarantine_judges_a_page_by_the_urls_name(tmp_path, monkeypatch, url, body, partial):
    from ghost_agent.tools import quarantine as Q
    from ghost_agent.utils.logging import request_id_context
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    (tmp_path / "sandbox").mkdir()

    async def _s(u, proxy, tmp):
        Path(tmp).write_bytes(body)
        return None, u, len(body)
    monkeypatch.setattr(Q, "_stream_download", _s)
    tok = request_id_context.set("web-test-owner")
    try:
        out = await Q.quarantine_download("http://" + "b" * 56 + ".onion/" + url, TOR)
    finally:
        request_id_context.reset(tok)
    assert str(out).startswith("PARTIAL") is partial, str(out)[:160]


class _Page:
    def __init__(self, titles, period=0.3):
        self._titles, self._t0, self._period = titles, None, period
        self.handlers = []

    async def title(self):
        import time as _t
        if self._t0 is None:
            self._t0 = _t.monotonic()
        i = min(len(self._titles) - 1, int((_t.monotonic() - self._t0) / self._period))
        return self._titles[i]

    def on(self, ev, fn):
        self.handlers.append(fn)

    def remove_listener(self, ev, fn):
        self.handlers.remove(fn)

    async def wait_for_load_state(self, *a, **k):
        return None

    async def expect_navigation(self, *a, **k):          # the old wait held this for the whole budget
        await asyncio.sleep(k.get("timeout", 8000) / 1000)
        raise TimeoutError


async def test_r1_the_challenge_wait_polls_the_title_and_honours_the_budget():
    pytest.importorskip("playwright")
    import time as _t
    from ghost_agent.tools import browser_runner as R
    p = _Page(["Just a moment...", "Just a moment...", "The real page"])
    t0 = _t.monotonic()
    await R._await_challenge_pass(p, 3.0)
    assert _t.monotonic() - t0 < 1.5 and p.handlers == []
    t0 = _t.monotonic()
    await R._await_challenge_pass(_Page(["Just a moment..."]), 0.4)
    assert _t.monotonic() - t0 < 1.0
    assert R._challenge_budget_s(30_000, _t.monotonic() - 29.0) < 1.0
    assert R._challenge_budget_s(30_000, _t.monotonic()) == R.CHALLENGE_WAIT_S


def test_r1_an_encrypted_pdf_reads_as_empty():
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "secret")
    data = doc.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="u", owner_pw="o")
    assert CC.pdf_text(data) == ""


# ───────────────────────────── review round 2 (§4UN-fetch r2) ─────────────────────────────

@pytest.mark.parametrize("name,head", [
    ("report.xls", b"<html><head><title>Sign in \xe2\x80\x93 Reports Portal</title></head><body><form>Your session "
                   b"expired. Sign in again.</form></body></html>"),
    ("report.xls", b"<html><head><title>404 Not Found</title></head><body><h1>Not Found</h1></body></html>"),
    ("report.xls", b"Access denied. You do not have permission to view this export."),
    ("letter.doc", b"<html><body><p>Please log in to continue</p></body></html>"),
    ("letter.doc", b"Error 500: internal server error"),
])
def test_r2_an_office_name_over_a_login_or_error_page_is_partial(name, head):
    assert CC.body_mismatch(name, "text/html", head, total=len(head)) != ""


def test_r2_r3_a_story_in_a_boilerplate_named_node_under_a_big_menu_is_never_lost():
    # r3 (operator principle via the coordinator): never lose the content — the menu may come back
    menu = "<nav>" + " ".join(f"Section{i}" for i in range(900)) + "</nav>"
    story = "<p>" + "A short real story sentence here. " * 30 + "</p>"
    html = f"<html><body>{menu}<div class='modal'>{story}</div><p>tail</p></body></html>"
    assert "A short real story" in CC.extract_main_text(html)


@pytest.mark.parametrize("cls", ["govuk-cookie-banner", "cc-window cc-banner", "cookie-notice-container",
                                 "js-cookie-consent"])
def test_r2_prefixed_consent_banners_go(cls):
    banner = f'<div class="{cls}">' + "We and our 900 partners use cookies to store and access data. " * 12 \
        + "</div>"
    story = "Short story: the bridge reopens Friday after repairs. " * 6
    html = f"<html><body>{banner}<p>{story}</p></body></html>"
    t = CC.extract_main_text(html)
    assert t == story.strip(), t[:120]


def test_r2_a_javascript_phrase_on_a_real_story_is_not_a_wall():
    text = "Please enable JavaScript to see comments. " + "The council voted on the budget today. " * 60
    assert 800 < len(text) < 3000
    assert CC.challenge_reason(200, "Budget vote", text, "https://news.example/") == ""
    assert "JavaScript" in CC.challenge_reason(200, "App", "Please enable JavaScript to use this site.", "https://a/")


def test_r2_a_retry_gets_its_own_floor_not_the_first_attempts_leftovers(tmp_path, monkeypatch):
    monkeypatch.setattr(FS, "_DOWNLOAD_BASE_S", 0.6)
    monkeypatch.setattr(FS, "_DOWNLOAD_MIN_RATE", 10**12)
    monkeypatch.setattr(FS, "_DOWNLOAD_STALL_S", 0.3)
    body = [b"%PDF-" + b"x" * 995] + [b"y" * 1000] * 4

    def responder(i, url):
        if i == 0:     # sends a little, then stalls at ~0.4 s (of a 0.6 s budget)
            return _Resp(200, {"Content-Length": "5000"}, chunks=body, delay=0.1, hang_after=1)
        return _Resp(200, {"Content-Length": "5000"}, chunks=body, delay=0.1)   # needs ~0.5 s
    out, _ = _download(tmp_path, responder)
    assert out.startswith("SUCCESS"), out


def test_r2_a_retry_starved_by_the_overall_cap_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(FS, "_DOWNLOAD_MAX_S", 1.0)
    monkeypatch.setattr(FS, "_DOWNLOAD_STALL_S", 0.3)
    body = [b"%PDF-" + b"x" * 995] + [b"y" * 1000] * 20

    def responder(i, url):
        if i == 0:
            return _Resp(200, {}, chunks=body, hang_after=1)                    # stalls at once
        return _Resp(200, {}, chunks=body, delay=0.1)                            # needs 2 s, has ~0.2
    out, _ = _download(tmp_path, responder)
    assert "overall" in out and "stalled" in out and "too slow" not in out, out


def test_r2_connect_and_first_byte_keep_a_full_minute(tmp_path):
    _download(tmp_path, lambda i, u: _Resp(200, {}, b"%PDF-1.7"))
    k = _Session.sessions[0]
    assert k["timeout"] >= 60 and k["timeout"] <= FS._DOWNLOAD_STALL_S


def test_r2_the_onion_connect_keeps_90_s():
    from ghost_agent.tools import quarantine as Q
    assert Q.FETCH_TIMEOUT_S == 90 and Q.STALL_S >= Q.FETCH_TIMEOUT_S


# ───────────────────────────── review round 3 (§4UN-fetch r3): never lose the content ─────────────────────────────

_LONG = "The committee published its findings on the harbour project this week. " * 30


def test_r3_an_aspnet_webforms_page_keeps_its_body():
    html = (f'<html><body><form id="form1" method="post"><div class="wrap"><p>{_LONG}</p></div></form>'
            '<footer>Copyright 2026 City Council</footer></body></html>')
    t = CC.extract_main_text(html)
    assert "harbour project" in t and "Copyright" not in t
    html2 = (f'<html><body><form id="form1"><h1>Notice</h1><p>Short notice text.</p></form>'
             f'<div>{"Unrelated listing text. " * 20}</div></body></html>')
    assert "Short notice text." in CC.extract_main_text(html2)


def test_r3_a_cookie_policy_page_is_the_content():
    policy = "This site uses cookies to remember your settings; this policy explains which and why. " * 40
    html = f'<html><body><h1>Cookie Policy</h1><div class="cookie-notice-content">{policy}</div></body></html>'
    assert "explains which and why" in CC.extract_main_text(html)
    html = f'<html><body><div class="cookie-policy-container"><p>{policy}</p></div></body></html>'
    assert "explains which and why" in CC.extract_main_text(html)


def test_r3_a_consent_named_wrapper_around_the_whole_page_is_not_lost():
    story = "The museum reopens on Saturday with a new wing for maritime history. " * 12
    html = f'<html><body><div id="consent-wrapper" class="container"><p>{story}</p></div></body></html>'
    assert "maritime history" in CC.extract_main_text(html)


def test_r3_r4_a_header_holding_the_whole_page_is_not_lost():
    # r4: a header outside <article>/<main> goes; here it held the page, so the raw-text fallback returns it
    # (with the menu — the accepted, lesser harm)
    html = (f'<html><body><header><h2>Opening hours</h2><p>{_LONG}</p></header>'
            '<nav>Home About</nav></body></html>')
    assert "harbour project" in CC.extract_main_text(html)


@pytest.mark.parametrize("name,head,ok", [
    ("big.xls", b"<html><head><style>" + b"td{color:red}\n" * 800 + b"</style></head><body><table><tr><td>1", True),
    ("styled.xls", b"<html><head><style>td{mso-number-format:'0.00'}</style></head><body><p>1</p>", True),
    ("titled.xls", b"Sales report Q3\nregion\tsales\nnorth\t40\nsouth\t35\n", True),
    ("codes.xls", b"code\nA1\nB2\nC3\nD4\nE5\nF6\n", False),     # r4: no single-column acceptance
    ("err.xls", b"Error\nThe export could not be generated because the session expired.\nTry again later.\n", False),
    ("memo.doc", b"MIME-Version: 1.0\r\nContent-Type: multipart/related; boundary=\"x\"\r\n\r\n--x\r\n", True),
    ("login.xls", b"<html><body><form action='/login'><table><tr><td>User</td><td><input name=u></td></tr><tr>"
                  b"<td><input type=\"password\" name=p></td></tr></table></form></body></html>", False),
])
def test_r3_office_export_marks(name, head, ok):
    assert (CC.body_mismatch(name, "text/html", head, total=len(head)) == "") is ok


def test_r3_curl_too_slow_after_bytes_is_a_stall_not_a_timeout():
    e = Exception("Failed to perform, curl: (28) Operation too slow. Less than 1 bytes/sec transferred the last "
                  "60 seconds")
    assert FS._failure_kind(e, 1000) == "stall"
    assert FS._failure_kind(e, 0) == "timeout"


async def test_r3_the_quarantine_reports_curls_too_slow_as_a_stall(home, tmp_path):
    from ghost_agent.tools import quarantine as Q

    class _R:
        status_code, headers = 200, {}

        async def aiter_content(self):
            yield b"PK\x03\x04data"
            raise Exception("Failed to perform, curl: (28) Operation too slow. Less than 1 bytes/sec")

        async def aclose(self):
            pass

    class _S:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, *a, **k):
            return _R()
    fake = types.SimpleNamespace(AsyncSession=_S)
    with patch.dict(sys.modules, {"curl_cffi": types.SimpleNamespace(requests=fake), "curl_cffi.requests": fake}):
        err, _, n = await Q._stream_download("http://" + "c" * 56 + ".onion/f.zip", TOR, tmp_path / "t")
    assert err and "stalled" in err and n == 0, err


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    return tmp_path


def test_r3_a_64kb_head_cut_mid_character_is_still_text():
    row = "Ελλάδα,Αθήνα,42\n".encode("utf-8")
    head = (b"country,city,n\n" + row * 5000)[:65536 - 1]          # ends inside a 2-byte character
    assert CC.sniff_kind(head) == "text"
    assert CC.body_mismatch("data.csv", "text/csv", head, total=10**6) == ""


# ───────────────────────────── review round 4 (§4UN-fetch r4): the minimal conservative extractor ──────────────────

def test_r4_faq_accordion_questions_are_kept():
    qa = "".join(f'<h3><button aria-expanded="false">Question {i}: how do refunds work?</button></h3>'
                 f'<div class="panel"><p>Answer {i}: refunds take five days.</p></div>' for i in range(6))
    t = CC.extract_main_text(f"<html><body><main><h1>FAQ</h1>{qa}</main></body></html>")
    assert "Question 3: how do refunds work?" in t and "Answer 3" in t


def test_r4_an_article_header_with_subtitle_and_time_is_kept():
    html = ('<html><body><header>SITE MENU</header><article><header><h1>Bridge reopens</h1>'
            '<p class="subtitle">Repairs finished early</p><time datetime="2026-11-12">12 Nov 2026</time></header>'
            f'<p>{_LONG}</p><footer>Filed under: Transport</footer></article><footer>Copyright</footer></body></html>')
    t = CC.extract_main_text(html)
    assert t.startswith("Bridge reopens Repairs finished early 12 Nov 2026")
    assert "Filed under: Transport" in t and "SITE MENU" not in t and "Copyright" not in t


def test_r4_class_named_blocks_forms_and_buttons_stay():
    html = (f'<html><body><div class="sidebar">Opening hours: 9-5</div><form><button>Subscribe</button></form>'
            f'<div role="complementary">Map link</div><p>{_LONG}</p></body></html>')
    t = CC.extract_main_text(html)
    for kept in ("Opening hours", "Subscribe", "Map link", "harbour project"):
        assert kept in t


@pytest.mark.parametrize("name,head,ok", [
    ("home.xls", b"<html><body><table width=100%><tr><td><a href=/>Home</a></td></tr></table><div>"
                 + b"Welcome to our company website, where we explain everything about our products. " * 8
                 + b"</div></body></html>", False),
    ("data.xls", b"<html><body><table><tr><td>north</td><td>40</td></tr><tr><td>south</td><td>35</td></tr>"
                 b"</table><p>Total</p></body></html>", True),
    ("codes.xls", b"code\nA1\nB2\nC3\nD4\nE5\nF6\n", False),
    ("two.xls", b"a,b\n1,2\n", False),
    ("three.xls", b"Sales Q3\na,b\n1,2\n3,4\n", True),
    ("ragged.xls", b"a,b,c\n1,2\nthe end, really\n", False),
])
def test_r4_office_exports(name, head, ok):
    assert (CC.body_mismatch(name, "text/html", head, total=len(head)) == "") is ok


@pytest.mark.parametrize("final", ["https://files.example/", "https://files.example/index.html",
                                   "https://files.example/login.html"])
def test_r4_a_file_link_redirected_to_a_page_is_partial(tmp_path, final):
    def responder(i, url):
        if url.endswith("report.pdf"):
            return _Resp(302, {"location": final})
        # r5: the rule applies to a page body only (a real PDF behind a gateway is the file)
        return _Resp(200, {"Content-Type": "application/pdf"}, b"<html><body>Sign in</body></html>")
    out, _ = _download(tmp_path, responder)
    assert out.startswith("PARTIAL") and "redirected to a page" in out, out


# ───────────────────────────── review round 5 (§4UN-fetch r5) ─────────────────────────────

def test_r5_a_real_pdf_behind_a_gateway_is_success(tmp_path):
    def responder(i, url):
        if url.endswith("report.pdf"):
            return _Resp(302, {"location": "https://dl.example/?f=report.pdf&sig=abc"})
        return _Resp(200, {"Content-Type": "application/pdf"}, b"%PDF-1.7 body")
    out, _ = _download(tmp_path, responder)
    assert out.startswith("SUCCESS"), out


def test_r5_a_redirect_to_an_html_page_with_html_bytes_is_still_partial(tmp_path):
    def responder(i, url):
        if url.endswith("report.pdf"):
            return _Resp(302, {"location": "https://files.example/"})
        return _Resp(200, {"Content-Type": "application/octet-stream"}, b"Welcome to Files Example")
    out, _ = _download(tmp_path, responder)
    assert out.startswith("PARTIAL") and "redirected to a page" in out, out


def test_r5_several_articles_without_main_are_all_kept():
    html = ("<html><body><article><h1>How do I reset my router?</h1><p>" + "It keeps dropping. " * 30
            + "</p></article><article><p>Answer: hold the reset button for 30 seconds.</p></article>"
            "<article><p>Answer: update the firmware first.</p></article></body></html>")
    t = CC.extract_main_text(html)
    assert "reset my router" in t and "hold the reset button" in t and "update the firmware" in t


def test_r5_a_header_with_the_title_and_byline_beside_a_div_body_is_kept():
    html = ('<html><body><header class="site"><a>Home</a> <a>News</a></header>'
            '<header class="article-header"><h1>Harbour plan approved</h1><time datetime="2026-11-12">12 Nov'
            f'</time> By A. Writer</header><div class="body"><p>{_LONG}</p></div>'
            '<footer>Copyright</footer></body></html>')
    t = CC.extract_main_text(html)
    assert t.startswith("Harbour plan approved 12 Nov By A. Writer") and "Home News" not in t, t[:80]


def test_r5_role_main_is_main_and_footnote_asides_stay():
    html = ('<html><body><div class="menu">Menu items here</div><div role="main"><p>' + "Body text. " * 80
            + '</p><aside class="footnote">1. Source: the 2025 census.</aside></div>'
            '<aside role="note">Note: figures are provisional.</aside><aside class="ad">Buy now</aside>'
            '<aside role="doc-footnote">2. Revised in March.</aside></body></html>')
    t = CC.extract_main_text(html)
    assert t.startswith("Body text.") and "Menu items" not in t and "Source: the 2025 census" in t
    full = CC.extract_main_text(html.replace('role="main"', 'class="x"'))
    assert "figures are provisional" in full and "Revised in March" in full and "Buy now" not in full


# ───────────────────────────── review round 6 (§4UN-fetch r6) ─────────────────────────────

def test_r6_teaser_cards_never_hide_the_story_outside_them():
    cards = "".join(f"<article><h3>Teaser {i}</h3><p>{'Short teaser text for another story. ' * 3}</p></article>"
                    for i in range(24))
    story = "The city council approved the new harbour plan after a long debate on Tuesday. " * 19
    html = f"<html><body><div class='story'><h1>Harbour plan approved</h1><p>{story}</p></div>{cards}</body></html>"
    t = CC.extract_main_text(html)
    assert "approved the new harbour plan" in t and "Teaser 3" in t


# ───────────────────────────── review round 7 (§4UN-fetch r7) ─────────────────────────────

def test_r7_a_title_header_before_main_survives_when_main_is_returned_alone():
    html = ('<html><body><header class="site">Home News Sport</header><header class="page-head">'
            '<h1>Rates held at 2%</h1><time datetime="2026-11-12">12 Nov 2026</time></header><main><p>'
            + "The central bank kept its main rate unchanged on Thursday. " * 30 + "</p></main></body></html>")
    t = CC.extract_main_text(html)
    assert t.startswith("Rates held at 2% 12 Nov 2026 The central bank kept"), t[:80]
    assert "Home News Sport" not in t


# §4UN-fetch final review: server-chosen text inside the download receipt must not let the verifier's
# file ledger capture a fake path, and a redirected-to-page PARTIAL names the final URL once.
def test_a_hostile_type_and_final_url_cannot_plant_a_ledger_path(tmp_path):
    from ghost_agent.core.agent import _fs_path_ledger
    from ghost_agent.tools.file_system import _download_receipt
    target = tmp_path / "f.pdf"
    target.write_bytes(b"%PDF-1.7 hello")
    out = _download_receipt("https://a.example/x", "f.pdf", target, 14, "application/pdf) b",
                            b"%PDF-1.7 hello", final_url="https://b.example/it's/f.pdf")
    alive, _ = _fs_path_ledger([{"role": "tool", "name": "file_system", "content": out}])
    assert alive == ["f.pdf"], (alive, out)
    assert "it%27s" in out


def test_a_redirect_to_a_page_names_the_final_url_once(tmp_path):
    from ghost_agent.tools.file_system import _download_receipt
    target = tmp_path / "f.pdf"
    page = b"<!doctype html><html><head><title>Sign in</title></head><body>log in</body></html>"
    target.write_bytes(page)
    out = _download_receipt("https://a.example/f.pdf", "f.pdf", target, len(page), "text/html", page,
                            final_url="https://a.example/login.html")
    assert out.count("https://a.example/login.html") == 1, out


def test_a_quote_in_the_asked_url_cannot_plant_a_ledger_path(tmp_path):
    from ghost_agent.core.agent import _fs_path_ledger
    from ghost_agent.tools.file_system import _download_receipt
    target = tmp_path / "real.bin"
    target.write_bytes(b"\x00\x01binary")
    out = _download_receipt("http://a.example/it' to 'evil.py", "real.bin", target, 8,
                            "application/octet-stream", b"\x00\x01binary")
    alive, _ = _fs_path_ledger([{"role": "tool", "name": "file_system", "content": out}])
    assert alive == ["real.bin"], (alive, out)
