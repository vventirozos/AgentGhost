"""§4QE — a page/post link given to vision_analysis (slack-55bbb388, 2026-10-20).

A member asked "explain what we're seeing here https://www.threads.com/share/…".
vision_analysis fetched the share link twice: Threads answered 429, the tool
said "Vision API Error: Client error '429 Too Many Requests'", the loop booked it
as transient ("too many requests") and the model retried, then searched for the
share code three times. 57 s to say it could not see the post."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ghost_agent.tools import vision as vision_mod
from ghost_agent.tools.outcome import OutcomeStatus
from ghost_agent.tools.tool_failure import FailureClass, classify_tool_failure, is_blocked_page_result
from ghost_agent.tools.vision import tool_vision_analysis

SHARE = "https://www.threads.com/share/BAeTnkKv8F/"


def _llm():
    llm = SimpleNamespace()
    llm.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": "a caption"}}]})
    return llm


def _no_ssrf(monkeypatch):
    import ghost_agent.utils.helpers as helpers
    monkeypatch.setattr(helpers, "url_ssrf_reason", lambda url, *, resolve=True: None)


def _serve(monkeypatch, status, ctype="text/html; charset=utf-8", body=b"<html></html>"):
    """An httpx stand-in answering every GET with one response; records URLs."""
    fetched = []

    class _Resp:
        status_code = status
        headers = {"content-type": ctype}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def aiter_bytes(self):
            yield body

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def stream(self, method, url):
            fetched.append(url)
            return _Resp()
    monkeypatch.setattr(vision_mod.httpx, "AsyncClient", _Client)
    return fetched


async def _look(target):
    llm = _llm()
    res = await tool_vision_analysis(action="describe_picture", target=target, llm_client=llm,
                                     sandbox_dir=None, tor_proxy="socks5://127.0.0.1:9050")
    return res, llm


# ── a post link on a social host: refused before any request leaves ──

@pytest.mark.parametrize("url", [
    SHARE, "https://threads.net/@a/post/x", "https://x.com/a/status/1",
    "https://m.facebook.com/story.php?id=1", "https://www.instagram.com/p/abc/",
    "https://vm.tiktok.com/ZM123/",
])
async def test_a_post_link_is_refused_without_a_fetch(monkeypatch, url):
    _no_ssrf(monkeypatch)
    fetched = _serve(monkeypatch, 200)
    res, llm = await _look(url)
    assert fetched == [] and not llm.chat_completion.await_count
    assert "post/page link, not an image file" in res and "nothing was downloaded" in res
    assert "screenshot" in res and "Do NOT retry it or search for it" in res
    assert classify_tool_failure(res)[0] is not FailureClass.RETRYABLE


@pytest.mark.parametrize("url", [
    "https://pbs.twimg.com/media/abc.jpg",          # the image CDN, not the post host
    "https://www.threads.com/media/shot.JPG",       # a file on a post host
    "https://notthreads.com/share/x",               # only a suffix of the name
    "https://example.org/x.com/a",                  # the host is not x.com
])
async def test_an_image_file_or_another_host_is_fetched(monkeypatch, url):
    _no_ssrf(monkeypatch)
    fetched = _serve(monkeypatch, 200)
    await _look(url)
    assert fetched == [url]


# ── the site's HTTP error: a site refusal, not the vision model's, not transient ──

async def test_a_429_from_the_site_is_a_blocked_download_not_a_vision_error(monkeypatch):
    _no_ssrf(monkeypatch)
    fetched = _serve(monkeypatch, 429)
    res, llm = await _look("https://cdn.example.org/a.png")
    assert fetched and not llm.chat_completion.await_count
    assert res.startswith("STATUS: BLOCKED (HTTP 429) — cdn.example.org refused the download")
    assert "vision model was never reached" in res and "will not pass" in res
    assert "Vision API" not in res
    assert res.status is OutcomeStatus.FAILED
    # the loop's routing: blocked-page branch (no strike), never "transient"
    assert is_blocked_page_result(res)
    assert classify_tool_failure(res)[0] is not FailureClass.RETRYABLE


@pytest.mark.parametrize("status", [403, 451])
async def test_any_other_refusal_is_blocked_too(monkeypatch, status):
    _no_ssrf(monkeypatch)
    _serve(monkeypatch, status)
    res, _ = await _look("https://cdn.example.org/a.png")
    assert res.startswith(f"STATUS: BLOCKED (HTTP {status})") and is_blocked_page_result(res)


@pytest.mark.parametrize("status", [404, 410])
async def test_a_missing_file_is_an_ordinary_failure_with_a_strike(monkeypatch, status):
    # r1 MAJOR: a 404 is usually a URL the model guessed (§4PC) — a BLOCKED
    # head would let guessed variants run with no strike at all
    _no_ssrf(monkeypatch)
    _serve(monkeypatch, status)
    res, _ = await _look("https://cdn.example.org/a.png")
    assert res.startswith(f"Error (HTTP {status}): there is no file at this URL on cdn.example.org")
    assert "never reached" in res and "Do not try variants" in res
    assert not is_blocked_page_result(res) and res.status is OutcomeStatus.FAILED
    assert classify_tool_failure(res)[0] is not FailureClass.RETRYABLE


@pytest.mark.parametrize("status", [500, 503])
async def test_a_server_fault_is_not_called_a_refusal(monkeypatch, status):
    # r1 MAJOR: "refused … will not pass" inside a result the loop formats as
    # TRANSIENT (\b503\b) told the model both things
    _no_ssrf(monkeypatch)
    _serve(monkeypatch, status)
    res, _ = await _look("https://cdn.example.org/a.png")
    assert res.startswith(f"Error (HTTP {status}): cdn.example.org failed to serve the file")
    assert "will not pass" not in res and "refused" not in res
    assert not is_blocked_page_result(res) and "Vision API" not in res


async def test_a_short_link_landing_on_a_post_is_refused_at_the_hop(monkeypatch):
    _no_ssrf(monkeypatch)
    fetched = []

    class _Resp:
        status_code = 301
        headers = {"location": SHARE}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def stream(self, method, url):
            fetched.append(url)
            return _Resp()
    monkeypatch.setattr(vision_mod.httpx, "AsyncClient", _Client)
    res, _ = await _look("https://t.co/abc")
    assert fetched == ["https://t.co/abc"] and "post/page link" in res and SHARE in res


def test_the_advice_names_the_screenshot_route():
    # r1 MINOR: "open it in the browser" meets the same 429/login wall; the
    # screenshot gets pixels
    assert "browser(operation='screenshot'" in vision_mod._PAGE_LINK_ADVICE
    assert "run vision_analysis on that file" in vision_mod._PAGE_LINK_ADVICE


async def test_the_refusal_names_the_host_that_answered_after_a_redirect(monkeypatch):
    _no_ssrf(monkeypatch)
    hops = iter([(302, {"location": "https://img.other.org/a.png"}), (403, {})])

    class _Resp:
        def __init__(self):
            self.status_code, self.headers = next(hops)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def stream(self, method, url):
            return _Resp()
    monkeypatch.setattr(vision_mod.httpx, "AsyncClient", _Client)
    res, _ = await _look("https://example.org/a.png")
    assert "img.other.org refused the download" in res


# ── a web page that is not an image: said plainly ──

@pytest.mark.parametrize("ctype", ["text/html; charset=utf-8", "application/xhtml+xml"])
async def test_a_web_page_is_named_as_a_page(monkeypatch, ctype):
    _no_ssrf(monkeypatch)
    _serve(monkeypatch, 200, ctype=ctype)
    res, llm = await _look("https://news.example.org/story")
    assert not llm.chat_completion.await_count
    assert f"is a web page ({ctype.split(';')[0]}), not an image or PDF" in res
    assert "Do NOT retry it" in res and "screenshot" in res
    assert classify_tool_failure(res)[0] is not FailureClass.RETRYABLE


async def test_other_non_image_types_keep_the_plain_refusal(monkeypatch):
    _no_ssrf(monkeypatch)
    _serve(monkeypatch, 200, ctype="application/json", body=b"{}")
    res, _ = await _look("https://api.example.org/x")
    assert "returned content-type 'application/json', not an image or PDF" in res


# ── the member notice ──

def test_the_member_notice_says_vision_reads_files_and_post_links_get_an_ask():
    from ghost_agent.core.agent import _MEMBER_CAPABILITY_NOTICE as N
    can, _, cannot = N.partition("You CANNOT")
    assert "never a web page or a post" in can
    assert "A social-media post or share link" in cannot
    assert "reply at once that you cannot open links on this channel" in cannot
    assert "paste the post's text or attach a screenshot" in cannot


# ── §4QE-b: a member's bare post link is answered by the system ──
# Live probes after §4QE: told "reply at once", the model still searched for the
# share code in 3 of 3 runs (12–47 s) before saying the same thing.

from ghost_agent.core.agent import member_post_link_reply  # noqa: E402


def _user(text):
    return [{"role": "user", "content": text}]


@pytest.mark.parametrize("text,site", [
    (f"explain what we're seeing here {SHARE}", "Threads"),
    (f"<{SHARE}>", "Threads"),                                        # Slack's link wrapping
    ("what is this? https://x.com/someone/status/123.", "X"),        # trailing punctuation
    ("https://www.instagram.com/p/abc/ https://vm.tiktok.com/Z1/ thoughts?", "Instagram"),
    ("what's this https://www.facebook.com/share/p/1/", "Facebook"),
    ("https://www.threads.com/@someone/post/C1x", "Threads"),
    ("https://fb.watch/abc/", "Facebook"),
    ("https://www.tiktok.com/@u/video/7123", "TikTok"),
    ("https://www.facebook.com/story.php?story_fbid=1&id=2", "Facebook"),
    ("https://www.facebook.com/watch/?v=123", "Facebook"),
    ("https://www.facebook.com/photo/?fbid=9", "Facebook"),
    ("what is this café thing https://x.com/u/status/1", "X"),        # r2: accents are still English
])

def test_a_bare_post_link_gets_the_system_reply(text, site):
    r = member_post_link_reply(_user(text))
    assert r.startswith("I can't open links on this channel") and f" {site} post" in r
    assert "Paste the post's text here or attach a screenshot" in r


@pytest.mark.parametrize("messages", [
    _user("what's the weather in Athens?"),                               # no link
    _user(f"compare {SHARE} with https://news.example.org/story"),        # a non-post link too
    _user("https://pbs.twimg.com/media/abc.jpg what is in this image"),  # an image file
    _user("what is this? https://www.threads.com/share/a.jpg."),         # an image file + full stop
    _user("<https://www.threads.com/share/a.jpg|a.jpg> what is it"),     # Slack's <url|label> form
    _user(f"{SHARE} " + "word " * 13),                                    # a real question around it
    [{"role": "user", "content": f"look {SHARE}\n\n[SYSTEM NOTE: The user attached a file named 'shot.png'.]"}],
    [{"role": "user", "content": "[SYSTEM NOTE: The user attached a file named 'shot.png'.]"},
     {"role": "assistant", "content": "ok"}, {"role": "user", "content": f"and this {SHARE}"}],
    # r1: a thread may already hold the post's text or the bot's description of it
    [{"role": "user", "content": "here is the post text: lorem ipsum"},
     {"role": "assistant", "content": "ok"}, {"role": "user", "content": f"explain this {SHARE}"}],
    [{"role": "assistant", "content": "earlier reply"}, {"role": "user", "content": f"explain {SHARE}"}],
    [{"role": "user", "content": SHARE}, {"role": "assistant", "content": "I can't open it"},
     {"role": "user", "content": "ok thanks"}],
    # r1 MAJOR: a profile or a home page is answerable from knowledge or search
    _user("is https://www.tiktok.com safe for kids?"),
    _user("who is this guy https://x.com/someone"),
    _user("https://www.instagram.com/someone/"),
    _user("https://fb.watch/"),
    # r2: profile tabs and the Watch home page are not posts
    _user("who is this https://www.facebook.com/someone/posts"),
    _user("https://www.instagram.com/someone/reels/"),
    _user("https://x.com/someone/photo"),
    _user("https://www.facebook.com/watch"),
    _user("https://www.facebook.com/watch/"),
    # r1: the fixed reply is English — another script goes to the model
    _user(f"τι βλέπουμε εδώ {SHARE}"),
    [{"role": "user", "content": [{"type": "text", "text": SHARE}, {"type": "image_url", "image_url": {}}]}],
    [], None,
])
def test_anything_else_goes_to_the_model(messages):
    assert member_post_link_reply(messages) == ""


def test_a_system_prompt_does_not_count_as_a_turn():
    assert member_post_link_reply([{"role": "system", "content": "sys"}] + _user(SHARE))


def test_the_article_fits_the_site():
    assert "and an X post" in member_post_link_reply(_user("https://x.com/u/status/1"))
    assert "and an Instagram post" in member_post_link_reply(_user("https://www.instagram.com/p/a/"))
    assert "and a Threads post" in member_post_link_reply(_user(SHARE))


def test_twelve_words_is_the_limit():
    assert member_post_link_reply(_user(f"{SHARE} " + "w " * 12))
    assert not member_post_link_reply(_user(f"{SHARE} " + "w " * 13))


def _chat_agent(tmp_path):
    from unittest.mock import MagicMock
    from ghost_agent.core.agent import GhostAgent
    (tmp_path / "memory").mkdir(exist_ok=True)
    context = MagicMock()
    context.llm_client.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": "MODEL TEXT"}}]})
    context.args.smart_memory = 0.0
    context.args.temperature = 0.1
    context.args.max_context = 8000
    context.args.perfect_it = False
    context.sandbox_dir = "/tmp"
    context.memory_dir = str(tmp_path / "memory")
    context.memory_system = MagicMock()
    context.profile_memory = MagicMock()
    context.profile_memory.get_context_string.return_value = ""
    agent = GhostAgent(context)
    agent.available_tools = {"web_search": AsyncMock(return_value="results")}
    return agent


@pytest.mark.parametrize("role,by_system", [("member", True), ("owner", False), ("", False)])
async def test_handle_chat_answers_only_a_member_without_the_model(tmp_path, monkeypatch, role, by_system):
    from ghost_agent.core import agent as A
    monkeypatch.setattr(A, "reply_is_public", lambda: False)
    agent = _chat_agent(tmp_path)
    content, _, rid = await agent.handle_chat(
        {"messages": _user(f"explain what we're seeing here {SHARE}"), "stream": False}, [],
        requester_role=role)
    if by_system:
        assert content.startswith("I can't open links on this channel")
        assert agent.context.llm_client.chat_completion.await_count == 0
        assert agent.available_tools["web_search"].await_count == 0
        assert agent.is_trivial_reply(rid) is True
    else:
        assert agent.context.llm_client.chat_completion.await_count >= 1
        assert "I can't open links on this channel" not in content


def test_the_vision_description_says_the_url_is_the_file_itself():
    from unittest.mock import MagicMock
    from ghost_agent.tools.registry import get_active_tool_definitions
    fn = next(t["function"] for t in get_active_tool_definitions(MagicMock())
              if t["function"]["name"] == "vision_analysis")
    assert "URL of the image or PDF FILE itself" in fn["description"]
    assert "social-media post is NOT an image" in fn["description"]
    assert "never a page or post link" in fn["parameters"]["properties"]["target"]["description"]

