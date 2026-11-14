"""§4NJ (2026-10-10): research yield — where a deep_research source is lost.

408 sources since 09-01: 33% yielded facts; 29% failed at fetch and the slot
was simply lost; YouTube pages never yielded a fact (0/27); WAF bot-check
pages were distilled as sources; a run where every loaded source said
"nothing relevant" read as a success."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from ghost_agent.core.node_throughput import DistillPlan
from ghost_agent.tools import search as S
from ghost_agent.utils.helpers import FetchError

BOT_CHECK = "One moment, please... Please wait while your request is being verified..."   # verbatim, 73 chars
FORBIDDEN = ("Error: Access Denied (403) from {u}. Application-level forbidden — Tor rotation will not help.")
def _err(text):
    """§4PD: a fetch failure is the fetcher's TYPE, never a page's first word"""
    return FetchError(text)


class _Client:
    def __init__(self, reply=lambda text: "- Fact: ZFS RAIDZ expansion arrived in OpenZFS 2.3."):
        self.reply = reply
        self.sent = []

    def plan_distill(self, budget_s, **kw):
        return DistillPlan(12_000, 300, True, "", 220.0, 12.0, 5, 40.0)

    def note_distill_density(self, *a, **k):
        return False

    async def chat_completion(self, payload, **kw):
        text = payload["messages"][-1]["content"]
        self.sent.append(text)
        return {"choices": [{"message": {"content": self.reply(text)}}]}


def _results(n):
    return [{"href": f"https://site{i}.example/article-{i}", "title": f"t{i}", "body": "b"} for i in range(n)]


async def _research(monkeypatch, results, fetch, client=None):
    monkeypatch.setattr(S.importlib.util, "find_spec", lambda name: True)
    monkeypatch.setattr(S, "_race_search_wave", AsyncMock(return_value=list(results)))
    fetched = []

    async def _fetch(url, **kw):
        fetched.append(url)
        return fetch(url)
    monkeypatch.setattr(S, "helper_fetch_url_content", _fetch)
    out = await S.tool_deep_research("openzfs vs btrfs reliability", False, None,
                                     llm_client=client if client is not None else _Client(), max_context=240_000)
    return str(out), out, fetched


# ── video pages are not read as sources ───────────────────────────────

@pytest.mark.parametrize("url,video", [
    ("https://www.youtube.com/watch?v=kvwRArxxRP0", True), ("https://youtu.be/kvwRArxxRP0", True),
    ("https://m.youtube.com/watch?v=x", True), ("https://www.youtube.com/shorts/abc", True),
    ("https://www.youtube.com/@channel/about", False), ("https://blog.youtube/news-and-events/x", False),
    ("https://example.com/watch?v=1", False)])
def test_video_pages_are_recognised(url, video):
    assert S.is_video_page_url(url) is video


def test_selection_skips_video_pages_and_takes_the_next_result():
    res = [{"href": "https://www.youtube.com/watch?v=a"}] + _results(9)
    picked = [r["href"] for r in S.select_research_sources(res)]
    assert "https://www.youtube.com/watch?v=a" not in picked and len(picked) == S.RESEARCH_MAX_SOURCES
    assert picked[-1] == "https://site7.example/article-7"


def test_a_batch_of_only_videos_is_still_read():
    res = [{"href": f"https://www.youtube.com/watch?v={i}"} for i in range(3)]
    assert len(S.select_research_sources(res)) == 3


# ── a bot-check page is a failed fetch ────────────────────────────────

@pytest.mark.parametrize("text,check", [
    (BOT_CHECK, True), ("Just a moment...\nEnable JavaScript and cookies to continue", True),
    ("Article: how the WAF says 'Please wait while your request is being verified' to bots. " * 20, False),
    ("ZFS vs Btrfs, a comparison of reliability and RAID.", False), ("", False)])
def test_bot_check_pages_are_recognised(text, check):
    assert S.is_bot_check_page(text) is check


async def test_a_bot_check_page_is_reported_failed_not_distilled(monkeypatch):
    client = _Client()
    text, out, fetched = await _research(monkeypatch, _results(1), lambda u: BOT_CHECK, client)
    assert "bot-check page" in text and not client.sent
    assert getattr(out, "reason_code", "") == "research_all_sources_failed"


# ── a failed source hands its slot to the next result ─────────────────

async def test_a_failed_fetch_is_backfilled_from_the_reserve(monkeypatch):
    res = _results(12)
    bad = {res[0]["href"], res[1]["href"]}
    text, out, fetched = await _research(
        monkeypatch, res, lambda u: _err(FORBIDDEN.format(u=u)) if u in bad else "OpenZFS 2.3 added RAIDZ expansion. " * 40)
    reserves = [r["href"] for r in res[8:10]]
    assert all(u in fetched for u in reserves), fetched
    assert "https://site10.example/article-10" not in fetched          # only as many as failed
    assert text.count("[EDGE EXTRACTED FACTS]") == 8 and text.count("Access Denied (403)") == 2
    # r1: coverage is counted in slots — both failures were replaced
    assert "SOURCE FAILURES" not in text and "[2 source(s) failed to load and were replaced" in text
    assert getattr(out, "reason_code", "") in ("", None)


async def test_a_backfill_stops_at_its_per_slot_cap(monkeypatch):
    res = _results(15)
    text, out, fetched = await _research(monkeypatch, res, lambda u: _err(FORBIDDEN.format(u=u)))
    assert len(fetched) <= S.RESEARCH_MAX_SOURCES * (1 + S.RESEARCH_BACKFILL_PER_SLOT)
    assert len(set(fetched)) == len(fetched) == 15                     # 8 + all 7 reserves, none twice
    assert getattr(out, "reason_code", "") == "research_all_sources_failed"


async def test_no_backfill_when_every_source_loads(monkeypatch):
    res = _results(12)
    _, _, fetched = await _research(monkeypatch, res, lambda u: "A real page. " * 40)
    assert sorted(fetched) == sorted(r["href"] for r in res[:8])


async def test_no_backfill_after_the_phase_deadline(monkeypatch):
    res = _results(12)
    monkeypatch.setattr(S, "_RESEARCH_PHASE_TIMEOUT_S", 0.0)
    text, _, fetched = await _research(monkeypatch, res, lambda u: _err(FORBIDDEN.format(u=u)))
    assert not set(fetched) & {r["href"] for r in res[8:]}
    assert not any(r["href"] in text for r in res[8:]), "a reserve was attempted past the deadline"


# ── every loaded source said "nothing relevant" ───────────────────────

@pytest.mark.parametrize("reply", ["No relevant information found.", "NO RELEVANT FACTS",
                                   "No hard facts explicitly relevant to the query were found."])
async def test_all_sources_empty_is_said_plainly(monkeypatch, reply):
    text, out, _ = await _research(monkeypatch, _results(3), lambda u: "A page. " * 40, _Client(lambda t: reply))
    assert "[⚠ NO RELEVANT FACTS: none of the 3 sources that loaded" in text
    assert getattr(out, "reason_code", "") == "research_no_relevant_facts"


async def test_one_source_with_facts_is_not_a_no_facts_report(monkeypatch):
    replies = iter(["No relevant information found.", "- OpenZFS 2.3 added RAIDZ expansion.", "NO RELEVANT FACTS"])
    text, out, _ = await _research(monkeypatch, _results(3), lambda u: "A page. " * 40,
                                   _Client(lambda t: next(replies)))
    assert "NO RELEVANT FACTS:" not in text and getattr(out, "reason_code", "") != "research_no_relevant_facts"


def test_the_evidence_gate_reads_the_no_facts_report_as_empty():
    from ghost_agent.core.evidence_gate import assess_turn_evidence
    body = ("--- DEEP RESEARCH RESULT ---\n[⚠ NO RELEVANT FACTS: none of the 5 sources that loaded had facts "
            "relevant to the query. Do not present …]\n\n### SOURCE: https://a\n[EDGE EXTRACTED FACTS]:\nNo relevant information found.")
    a = assess_turn_evidence([{"name": "deep_research", "content": body}])
    assert a.consulted == 1 and a.substantive == 0


@pytest.mark.parametrize("block,has", [
    ("### SOURCE: https://a\n[EDGE EXTRACTED FACTS]:\n- RAIDZ expansion in 2.3", True),
    ("### SOURCE: https://a\n[EDGE EXTRACTED FACTS]:\nNo relevant information found.", False),
    ("### SOURCE: https://a\n[EDGE EXTRACTED FACTS]:\nNO RELEVANT FACTS", False),
    ("### SOURCE: https://a\n[EDGE EXTRACTED FACTS]:\n", False),
    ("### SOURCE: https://a\nError: Access Denied (403)", False),
    ("### SOURCE: https://a\nraw page text about ZFS\n[...truncated...]", True)])
def test_distilled_block_has_facts(block, has):
    assert S.distilled_block_has_facts(block) is has


async def test_a_bot_check_is_retried_on_the_next_circuit(monkeypatch):
    """The second attempt goes out on another circuit and may get the page."""
    seen = {}

    def _fetch(u):
        seen[u] = seen.get(u, 0) + 1
        return BOT_CHECK if seen[u] == 1 else "OpenZFS 2.3 added RAIDZ expansion. " * 40
    text, out, fetched = await _research(monkeypatch, _results(1), _fetch)
    assert fetched.count("https://site0.example/article-0") == 2
    assert "[EDGE EXTRACTED FACTS]" in text and "bot-check" not in text


async def test_a_slot_that_raises_is_counted_lost(monkeypatch):
    """A slot that dies (here: a cancelled fetch) is reported, not silently dropped."""
    import asyncio
    res = _results(10)

    def _fetch(u):
        if u == res[0]["href"]:
            raise asyncio.CancelledError()
        return "A page. " * 40
    text, out, _ = await _research(monkeypatch, res, _fetch)
    assert "1 of 8 sources could not be fetched" in text
    assert getattr(out, "reason_code", "") == "research_sources_partial"


async def test_a_backfill_log_failure_loses_nothing(monkeypatch):
    """r1 MINOR: the log call sat outside every catch — a failing log sink
    turned a slot (and its finished source) into a lost one."""
    real = S.pretty_log

    def _log(tag, *a, **k):
        if tag == "Research Backfill":
            raise RuntimeError("log sink down")
        return real(tag, *a, **k)
    monkeypatch.setattr(S, "pretty_log", _log)
    res = _results(10)
    text, out, fetched = await _research(monkeypatch, res,
                                         lambda u: _err(FORBIDDEN.format(u=u)) if u == res[0]["href"] else "A page. " * 40)
    assert res[8]["href"] in fetched and "SOURCE FAILURES" not in text


# ── fresh-reader round r1 ─────────────────────────────────────────────

@pytest.mark.parametrize("facts", [
    "The provided text does not state the exact date, but it reports the launch is planned for Q2 2027 and costs $40M.",
    "No relevant information about 2026; however the 2025 final was won by Spain 2-1.",
    "No relevant information found for EX60. " + "EX40 range: 240 miles (WLTP), 82 kWh battery, 200 kW charging. " * 4])
def test_a_hedged_extract_still_has_facts(facts):
    assert S.distilled_block_has_facts(f"### SOURCE: https://a\n[EDGE EXTRACTED FACTS]:\n{facts}") is True


@pytest.mark.parametrize("facts", ["There is no information about the query in the text.",
                                   "The page does not contain relevant facts."])
# §4UN-deep r1: "The source does not mention OpenZFS." is no longer folded — only refusals that say no
# relevant/hard facts or information about the query are; a negative statement stays (never lose a fact)
def test_more_refusal_wordings_are_recognised(facts):
    assert S.distilled_block_has_facts(f"### SOURCE: https://a\n[EDGE EXTRACTED FACTS]:\n{facts}") is False


async def test_two_empty_pages_beside_failures_are_partial_not_no_facts(monkeypatch):
    """r1 MAJOR: 6 of 8 failed + 2 loaded-but-empty was booked "no relevant
    facts", and the gate read it as no evidence — against the coverage banner."""
    res = _results(8)
    good = {res[0]["href"], res[1]["href"]}
    text, out, _ = await _research(monkeypatch, res, lambda u: "A page. " * 40 if u in good else _err(FORBIDDEN.format(u=u)),
                                   _Client(lambda t: "No relevant information found."))
    assert "NO RELEVANT FACTS:" not in text and "6 of 8 sources could not be fetched" in text
    assert getattr(out, "reason_code", "") == "research_sources_partial"


async def test_three_empty_pages_beside_a_failure_still_say_no_facts(monkeypatch):
    res = _results(4)
    text, out, _ = await _research(monkeypatch, res,
                                   lambda u: _err(FORBIDDEN.format(u=u)) if u == res[0]["href"] else "A page. " * 40,
                                   _Client(lambda t: "No relevant information found."))
    assert "[⚠ NO RELEVANT FACTS: none of the 3 sources that loaded" in text
    assert getattr(out, "reason_code", "") == "research_sources_partial"     # coverage status wins the code


async def test_an_error_reading_result_is_a_failed_source_and_is_backfilled(monkeypatch):
    """r1 MINOR: the helper's "Error reading <url>: …" (no colon) reached the
    distiller as page text and its slot was never backfilled."""
    res = _results(9)
    client = _Client()
    text, out, fetched = await _research(
        monkeypatch, res,
        lambda u: _err(f"Error reading {u}: Failed to perform, curl: (56) Connection closed") if u == res[0]["href"]
        else "A page. " * 40, client)
    assert res[8]["href"] in fetched
    assert not any("Error reading" in t for t in client.sent)
    assert "### SOURCE: https://site0.example/article-0\nError: Error reading" in text


async def test_a_reserve_the_deadline_never_ran_is_not_a_failed_source(monkeypatch):
    """r1 MINOR: a reserve popped with time left, then queued behind the
    semaphore past the deadline, was booked as one more failed source."""
    import asyncio
    # timeline (sem=3): site0 fails at once and pops the reserve with 10 s left;
    # site1 frees a slot at 0.2 s; the 2 s pages free the rest at ~2 s/2.2 s
    # (last primary starts with ~7.8 s ≥ 7 s); the reserve, queued behind
    # them, gets a slot at ~4 s with ~6 s < 7 s → never attempted
    monkeypatch.setattr(S, "_RESEARCH_PHASE_TIMEOUT_S", 10.0)
    monkeypatch.setattr(S, "_MIN_URL_BUDGET_S", 7.0)
    res = _results(10)
    monkeypatch.setattr(S.importlib.util, "find_spec", lambda name: True)
    monkeypatch.setattr(S, "_race_search_wave", AsyncMock(return_value=list(res)))
    fetched = []

    async def _fetch(url, **kw):
        fetched.append(url)
        if url == res[0]["href"]:
            return _err(FORBIDDEN.format(u=url))
        await asyncio.sleep(0.2 if url == res[1]["href"] else 2.0)
        return "A page. " * 40
    monkeypatch.setattr(S, "helper_fetch_url_content", _fetch)
    out = await S.tool_deep_research("openzfs vs btrfs reliability", False, None, llm_client=_Client(), max_context=240_000)
    text = str(out)
    assert all(r["href"] in fetched for r in res[1:8]), "a primary hit the deadline — the timeline is off"
    assert res[8]["href"] not in text, "the never-attempted reserve was reported"
    assert "1 of 8 sources could not be fetched" in text and "research phase deadline" not in text
