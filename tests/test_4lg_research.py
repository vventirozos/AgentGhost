"""§4LG (2026-10-04): web + dark-web research — what a refute may rest on,
whether a correction reaches the owner, what the tools say they read, and the
fetches that kept failing. Each test names the world it fails in."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
from tests.test_critic_async import agent  # noqa: F401 — the fixture
from ghost_agent.utils.logging import (reply_surface_context, request_id_context, request_origin_context,
                                       requester_role_context, ORIGIN_PROBE)


# ── a refute must rest on the evidence, not the judge's memory ───────────────
_CLAIM = "The final boss of Shadow of the Erdtree is Promised Consort Radahn."
_EV = "[web_search] Promised Consort Radahn is the final boss of the Shadow of the Erdtree DLC."


@pytest.mark.parametrize("issue,claim,evidence,memory", [
    ("The actual final boss is Messiah, not Promised Consort Radahn.", _CLAIM, _EV, True),
    ("Released in 1991, not 1990.", "Python was released in 1990.", "Python first appeared", True),
    ("The claim names Lexar Professional, which is not in the evidence.",
     "Buy the Lexar Professional card.", "SanDisk Extreme 1TB", False),            # unsupported: stands
    ("Latest stable is 3.14.6, not 3.13.4.", "The latest 3.13 patch is 3.13.4.",
     "Python 3.14.6 released; 3.13.8 security fix", False),                        # counter-fact IS evidence
    ("The evidence says 2017; the claim says 2016.", "It launched in 2016.", "it launched in 2017", False),
    # a sentence's first word is not a counter-fact ("Unsupported" is the audit's word)
    ("Unsupported: the claim's price of 49 euros.", "It costs 49 euros.", "price list unavailable", False),
])
def test_a_refute_from_memory_is_recognised(issue, claim, evidence, memory):
    """Fails where "the actual final boss is Messiah" refuted a correct
    answer and the reflection wrote it as a lesson."""
    from ghost_agent.core.verifier import _refute_rests_on_memory
    assert _refute_rests_on_memory([issue], claim, evidence) is memory


def test_a_memory_refute_is_downgraded_and_an_evidence_refute_stands(monkeypatch):
    # §4LH final review: the guard is OFF by default (it fired once in the
    # replay, wrongly); this pins what it does when switched on
    monkeypatch.setenv("GHOST_VERIFY_MEMORY_REFUTE_GUARD", "1")
    from ghost_agent.core.verifier import Verifier as ClaimVerifier
    mem = VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=0.95, reasoning="judge",
                       issues=["The actual final boss is Messiah, not Promised Consort Radahn."])
    out = ClaimVerifier._guard_memory_refute(None, mem, _CLAIM, _EV)
    assert out.verdict == VerifyVerdict.UNCERTAIN and out.confidence <= 0.5
    ok = VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=0.9, reasoning="judge",
                      issues=["The evidence says 2017; the claim says 2016."])
    assert ClaimVerifier._guard_memory_refute(None, ok, "It launched in 2016.", "launched in 2017") is ok


def test_the_guard_runs_on_every_verdict(monkeypatch):
    """Fails where the guard existed but verify_claim never applied it."""
    monkeypatch.setenv("GHOST_VERIFY_MEMORY_REFUTE_GUARD", "1")
    from ghost_agent.core.verifier import Verifier as ClaimVerifier
    v = ClaimVerifier.__new__(ClaimVerifier)
    mem = VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=0.95, reasoning="judge",
                       issues=["The actual final boss is Messiah, not Promised Consort Radahn."])
    monkeypatch.setattr(ClaimVerifier, "_verify_claim_incumbent", AsyncMock(return_value=mem))
    monkeypatch.setenv("GHOST_VERIFY_CLAIM_BINDING", "0")
    import ghost_agent.core.verifier as V
    monkeypatch.setattr(V, "_claim_binding_primary_enabled", lambda: False)
    monkeypatch.setattr(V, "_claim_binding_refute_first_enabled", lambda: False)
    out = asyncio.run(v.verify_claim(_CLAIM, _EV, "who is the final boss?"))
    assert out.verdict == VerifyVerdict.UNCERTAIN


# ── a late correction reaches the owner ──────────────────────────────────────
def _owner_ctx(role="owner", surface="", origin="", rid="web-1", request="who is the final boss?"):
    from ghost_agent.memory.lesson_scope import current_request
    toks = [(v, v.set(x)) for v, x in ((requester_role_context, role), (reply_surface_context, surface),
                                       (request_origin_context, origin), (request_id_context, rid),
                                       (current_request, request))]
    return toks


@pytest.mark.parametrize("role,surface,origin,rid,sent", [
    ("owner", "", "", "web-1", True), ("member", "", "", "slack-1", False),
    ("owner", "public", "", "slack-2", False), ("owner", "", ORIGIN_PROBE, "web-3", False),
    ("owner", "", "", "sched-task_1", False)])
def test_a_late_correction_is_sent_to_the_owner_only(monkeypatch, role, surface, origin, rid, sent):
    """Fails where the correction waited 15 min for a next message in the same
    conversation — 0 of 11 queued corrections were ever shown."""
    from ghost_agent.core.agent import GhostAgent
    import ghost_agent.core.autonomous_activity as aa
    import ghost_agent.tools.notify_tool as nt
    log = MagicMock()
    log.record.return_value = True
    monkeypatch.setattr(aa, "get_activity_log", lambda c: log)
    monkeypatch.setattr(nt, "_rate_limited", lambda: False)
    toks = _owner_ctx(role, surface, origin, rid)
    try:
        out = GhostAgent._notify_owner_correction(SimpleNamespace(context=SimpleNamespace()), "the boss is X")
    finally:
        for v, t in reversed(toks):
            v.reset(t)
    assert out is sent and log.record.called is sent
    if sent:
        msg = log.record.call_args.args[1]
        assert "who is the final boss?" in msg and "the boss is X" in msg


def test_a_late_refute_sends_the_notice(monkeypatch, agent):
    """Fails where the notice existed but the late-refute path never sent it."""
    from ghost_agent.core.agent import GhostAgent
    agent.context.skill_memory = MagicMock()
    agent._pending_corrections = []
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    sent = []
    monkeypatch.setattr(GhostAgent, "_notify_owner_correction", lambda self, note: sent.append(note) or True)
    with patch("ghost_agent.core.agent.pretty_log"), patch("ghost_agent.core.agent._glog.spawn_task"):
        agent._record_late_verdict(VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=0.9, reasoning="r",
                                                issues=["wrong number"]), trajectory_id="traj-1")
    assert sent and "wrong number" in sent[0]


# ── what the research tools say they read ────────────────────────────────────
def test_the_research_rule_is_in_the_system_prompt():
    from ghost_agent.core.prompts import SYSTEM_PROMPT
    assert "RESEARCH ANSWERS" in SYSTEM_PROMPT and "Never invent a URL" in SYSTEM_PROMPT
    assert "never describe a thread, post or discussion you did not open" in SYSTEM_PROMPT
    # the live D2 probe took a clone page from onion search as "DuckDuckGo's official onion"
    assert "OFFICIAL .onion address comes only from that organisation's own clearnet site" in SYSTEM_PROMPT


def test_dark_web_results_say_no_page_was_opened(monkeypatch):
    """Fails where directory TITLES were described as active forum threads."""
    import ghost_agent.tools.darkweb_search as D
    ranked = [{"url": "http://abc.onion/x", "title": "Forum index", "snippet": "", "engines": ["ahmia"]}]
    monkeypatch.setattr(D, "_darkweb_search_raw", AsyncMock(return_value=(ranked, [], False, 1)))
    monkeypatch.setattr(D, "_cache_get", lambda k: None, raising=False)
    out = asyncio.run(D.tool_darkweb_search(query="mh370 forum", tor_proxy="socks5://x"))
    assert "TITLES and snippets only — no page was opened" in out


def test_onions_are_read_on_topic_first_and_homepages_last():
    """Fails where engine agreement alone picked the pages: 32 of 36 read
    were "No relevant information", 25 of them homepages."""
    from ghost_agent.tools.darkweb_search import rank_for_reading
    ranked = [{"url": "http://home1.onion/", "title": "Welcome", "snippet": ""},
              {"url": "http://b.onion/threads/mh370-theories", "title": "MH370 theories", "snippet": ""},
              {"url": "http://c.onion/", "title": "MH370 archive", "snippet": ""}]
    out = [r["url"] for r in rank_for_reading(ranked, "mh370 theories")]
    assert out[0].endswith("/threads/mh370-theories") and out[-1] == "http://home1.onion/"
    # topic beats path: an off-topic deep page never outranks an on-topic homepage
    mixed = [{"url": "http://shop.onion/cat/item-7", "title": "Buy gift cards", "snippet": ""},
             {"url": "http://c.onion/", "title": "MH370 archive", "snippet": ""}]
    assert rank_for_reading(mixed, "mh370 theories")[0]["url"] == "http://c.onion/"


def test_dark_web_research_reads_in_the_reading_order(monkeypatch):
    """Fails where the ranking existed but darkweb_research still FETCHED in
    the engines' order (asserted on the fetch, not on the ranker's output —
    the first version passed with the ranking computed and ignored)."""
    import ghost_agent.tools.darkweb_search as D
    ranked = [{"url": "http://home1.onion/", "title": "Welcome", "snippet": "", "engines": ["a"]},
              {"url": "http://b.onion/threads/mh370", "title": "MH370 theories", "snippet": "", "engines": ["a"]}]
    monkeypatch.setattr(D, "_darkweb_search_raw", AsyncMock(return_value=(ranked, [], False, 2)))
    fetched = []

    async def _fetch(url, tor_proxy):
        fetched.append(url)
        return ""
    monkeypatch.setattr(D, "_fetch_onion_text", _fetch)
    try:
        asyncio.run(D.tool_darkweb_research(query="mh370 theories", tor_proxy="socks5://x", max_sources=1,
                                            llm_client=None))
    except Exception:  # noqa: BLE001 — the synthesis half is not under test here
        pass
    assert fetched and fetched[0] == "http://b.onion/threads/mh370"


# ── fetches that kept failing ────────────────────────────────────────────────
def test_a_site_that_failed_twice_is_skipped_until_it_loads(monkeypatch):
    """Fails where 60 of 65 re-fetches of an already-failed host failed again
    (one host: 19 attempts, all HTTP/2 errors)."""
    import ghost_agent.tools.browser as B
    import ghost_agent.tools.host_memo as H
    H._HOST_FAILS.clear()
    u = "https://slow.example.it/page"
    H._mark_host_failed(u, "net::ERR_HTTP2_PROTOCOL_ERROR")
    assert H._dead_host_notice(u) is None
    H._mark_host_failed(u, "net::ERR_HTTP2_PROTOCOL_ERROR")
    assert "failed 2 times" in H._dead_host_notice(u)
    H._mark_host_ok(u)
    assert H._dead_host_notice(u) is None
    H._mark_host_failed("https://x.org", "selector #a not found")
    H._mark_host_failed("https://x.org", "selector #a not found")
    assert H._dead_host_notice("https://x.org") is None              # not a site failure
    monkeypatch.setenv("GHOST_DEAD_HOST_MEMO", "0")
    H._mark_host_failed(u, "timeout")
    H._mark_host_failed(u, "timeout")
    assert H._dead_host_notice(u) is None


@pytest.mark.asyncio
async def test_the_browser_refuses_a_remembered_dead_site_without_running(monkeypatch, tmp_path):
    import ghost_agent.tools.browser as B
    import ghost_agent.tools.host_memo as H
    H._HOST_FAILS.clear()
    u = "https://slow.example.it/page"
    for _ in range(2):
        H._mark_host_failed(u, "Page.goto: Timeout 30000ms exceeded.")
    sm = MagicMock()
    out = await B.tool_browser(operation="navigate", url=u, sandbox_dir=tmp_path, sandbox_manager=sm,
                               tor_proxy="socks5://127.0.0.1:9050")
    assert "failed 2 times" in str(out) and not sm.execute.called


@pytest.mark.asyncio
async def test_a_navigation_timeout_is_not_retried_by_default(monkeypatch, tmp_path):
    """Fails where the commit-milestone retry ran on every timeout (it
    recovered 1 of 17, ~30 s each) and the hint invited another attempt."""
    import ghost_agent.tools.browser as B
    import ghost_agent.tools.host_memo as H
    H._HOST_FAILS.clear()
    monkeypatch.delenv("GHOST_BROWSER_COMMIT_RETRY", raising=False)
    sm = MagicMock()
    sm.execute.return_value = ("[BROWSER_ERR] Timeout 30000ms exceeded navigating to https://a.example/", 1)
    monkeypatch.setattr(B, "_ensure_runner", AsyncMock(return_value=None), raising=False)
    out = await B.tool_browser(operation="navigate", url="https://a.example/", sandbox_dir=tmp_path,
                               sandbox_manager=sm, tor_proxy="socks5://127.0.0.1:9050")
    navs = [c for c in sm.execute.call_args_list if "commit" in str(c)]
    assert not navs
    assert "another attempt usually times out" in str(out)



@pytest.mark.asyncio
async def test_a_clean_load_clears_a_sites_strikes(monkeypatch, tmp_path):
    """Fails where a recovered site stayed one strike from a ban forever."""
    import ghost_agent.tools.browser as B
    import ghost_agent.tools.host_memo as H
    H._HOST_FAILS.clear()
    u = "https://flaky.example/page"
    H._mark_host_failed(u, "Timeout 30000ms exceeded")
    sm = MagicMock()
    sm.execute.return_value = ('[BROWSER_OK] {"url": "https://flaky.example/page", "title": "Flaky", '
                               '"status": 200, "text": "' + "real article text " * 40 + '"}', 0)
    await B.tool_browser(operation="navigate", url=u, sandbox_dir=tmp_path, sandbox_manager=sm,
                         tor_proxy="socks5://127.0.0.1:9050")
    assert "flaky.example" not in H._HOST_FAILS


@pytest.mark.parametrize("tool", ["darkweb_research", "fact_check", "web_search"])
def test_research_tools_count_as_external_evidence(tool):
    """Fails where darkweb_research and fact_check ranked below the outside-
    world tools in the verifier's digest."""
    from ghost_agent.core.agent import _evidence_is_external
    assert _evidence_is_external({"name": tool, "content": "x"})


@pytest.mark.asyncio
async def test_a_blocked_page_is_a_strike_too(monkeypatch, tmp_path):
    """Fails where a 403 / bot-challenge page (reported BLOCKED, not as a
    runner failure) never counted toward the host memory."""
    import ghost_agent.tools.browser as B
    import ghost_agent.tools.host_memo as H
    H._HOST_FAILS.clear()
    u = "https://walled.example/article"
    sm = MagicMock()
    sm.execute.return_value = ('[BROWSER_OK] {"url": "https://walled.example/article", "title": "Just a moment...", '
                               '"status": 403, "text": "Checking your browser"}', 0)
    for _ in range(2):
        out = await B.tool_browser(operation="navigate", url=u, sandbox_dir=tmp_path, sandbox_manager=sm,
                                   tor_proxy="socks5://127.0.0.1:9050")
        assert "BLOCKED" in str(out)
    assert H._dead_host_notice(u)


# ── onion addresses that cannot load (live probe D2, second run) ─────────────
def _onion_sb(out, code):
    from unittest.mock import MagicMock
    sb = MagicMock()
    sb.calls = []
    sb.execute = lambda cmd, timeout=300, **kw: (sb.calls.append(cmd), (out, code))[1]
    return sb


@pytest.mark.asyncio
async def test_a_v2_onion_is_refused_before_it_is_dialled(tmp_path):
    """Fails where DuckDuckGo's retired 16-character v2 address was dialled,
    failed, and became "Tor's SOCKS proxy is failing for all .onion
    addresses" plus "official v3 address" in the reply."""
    from ghost_agent.tools.browser import tool_browser
    sb = _onion_sb('[BROWSER_OK] {"status": 200, "url": "x", "title": "T", "text": "b", "length": 1, "truncated": false}\n', 0)
    out = await tool_browser(operation="navigate", url="http://3g2upl4pq6kufc4m.onion/",
                             sandbox_dir=tmp_path, sandbox_manager=sb)
    assert not sb.calls
    assert "v2 onion address" in out and "NOT a current address" in out and "Tor working" in out
    await tool_browser(operation="navigate",
                       url="http://duckduckgogg42xjoc72x3sjasowoarfbgcmvfimaftt6twagswzczad.onion/",
                       sandbox_dir=tmp_path, sandbox_manager=sb)
    assert len(sb.calls) == 1                     # a v3 address is dialled as before
    await tool_browser(operation="navigate", url="http://abcdefghijklmnop.co.uk/",
                       sandbox_dir=tmp_path, sandbox_manager=sb)
    assert len(sb.calls) == 2                     # a clearnet host is never an onion


@pytest.mark.asyncio
async def test_the_first_tor_failure_on_an_onion_says_it_is_not_tor(tmp_path):
    """Fails where a first ERR_SOCKS_CONNECTION_FAILED got the generic
    browser advice; after three one-off failures the model diagnosed Tor
    and tried to restart it."""
    from ghost_agent.tools.browser import tool_browser
    sb = _onion_sb("[BROWSER_ERR] net::ERR_SOCKS_CONNECTION_FAILED\n", 1)
    url = "http://bww2yrsf4gooxcwwnkek7dzzbnq3rvf3w4vm4pys5zid4r2xwqj5wfqd.onion/"
    out = await tool_browser(operation="navigate", url=url, sandbox_dir=tmp_path, sandbox_manager=sb)
    assert len(sb.calls) >= 1
    assert "does NOT mean Tor is broken" in out and "different" in out.lower()
    clear = _onion_sb("[BROWSER_ERR] net::ERR_SOCKS_CONNECTION_FAILED\n", 1)
    out2 = await tool_browser(operation="navigate", url="https://example.com/",
                              sandbox_dir=tmp_path, sandbox_manager=clear)
    assert "does NOT mean Tor is broken" not in out2   # a clearnet SOCKS error is OUR side



def test_the_memory_refute_guard_is_off_by_default(monkeypatch):
    """§4LH final review: replayed over the recorded refutes it fired once, on
    a right refute ("stork, not pelican"), and missed its own motivating case."""
    monkeypatch.delenv("GHOST_VERIFY_MEMORY_REFUTE_GUARD", raising=False)
    from ghost_agent.core.verifier import Verifier as ClaimVerifier
    mem = VerifyResult(verdict=VerifyVerdict.REFUTED, confidence=0.95, reasoning="judge",
                       issues=["The actual final boss is Messiah, not Promised Consort Radahn."])
    assert ClaimVerifier._guard_memory_refute(None, mem, _CLAIM, _EV) is mem


@pytest.mark.parametrize("url,cause,strikes", [
    ("https://en.wikipedia.org/wiki/Nope", "blocked: HTTP 404", False),          # one page, not the site
    ("https://site.org/a", "blocked: HTTP 403 — bot challenge", True),
    ("https://github.com/x", "Locator.click: Timeout 30000ms exceeded", False),   # the page's shape
    ("https://site.org/a", "Page.goto: Timeout 30000ms exceeded.", True),
    ("https://site.org/2024/03/4031-story", "net::ERR_CONNECTION_RESET at https://site.org/2024/03/4031-story", False),
    ("http://localhost:3000/", "Page.goto: Timeout 30000ms exceeded.", False),   # our own dev server
    ("http://192.168.1.5/", "Page.goto: Timeout 30000ms exceeded.", False),
])
def test_only_a_site_level_failure_strikes_the_site(url, cause, strikes):
    """Fails where a 404, a selector timeout, a URL containing "403", or a
    slow local dev server banned the whole host for 6 h (§4LH final review)."""
    import ghost_agent.tools.host_memo as H
    H._HOST_FAILS.clear()
    for _ in range(2):
        H._mark_host_failed(url, cause)
    assert bool(H._dead_host_notice(url)) is strikes


def test_a_host_is_keyed_with_its_port():
    import ghost_agent.tools.host_memo as H
    H._HOST_FAILS.clear()
    for _ in range(2):
        H._mark_host_failed("https://site.org:8443/a", "Page.goto: net::ERR_HTTP2_PROTOCOL_ERROR")
    assert H._dead_host_notice("https://site.org:8443/b") and not H._dead_host_notice("https://site.org/b")
