"""§4NK (2026-10-10): darkweb_research gets the §4NJ yield fixes.

Stored darkweb_research sources: 58 from 10 calls; 45 "nothing relevant",
11 facts. Bare homepages: 0 facts in 25 reads; pages with a path: 11 of 29."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

import ghost_agent.tools.darkweb_search as D
from ghost_agent.tools import search as S
from tests.test_4nj_research_yield import BOT_CHECK, _Client
from ghost_agent.utils.helpers import FetchError  # §4PD: a fetch failure is a TYPE


def _row(url, title="mh370 theories", snippet=""):
    return {"url": url, "title": title, "snippet": snippet, "engines": ["a"]}


async def _research(monkeypatch, ranked, fetch, client=None, max_sources=3):
    monkeypatch.setattr(D, "_darkweb_search_raw", AsyncMock(return_value=(list(ranked), [], False, len(ranked))))
    monkeypatch.setattr(D, "_cache_get", lambda k: None)
    monkeypatch.setattr(D, "_cache_put", lambda k, v: None)
    fetched = []

    async def _fetch(url, tor_proxy):
        fetched.append(url)
        return fetch(url)
    monkeypatch.setattr(D, "_fetch_onion_text", _fetch)
    out = await D.tool_darkweb_research(query="mh370 theories", tor_proxy="socks5://127.0.0.1:9050",
                                        llm_client=client if client is not None else _Client(),
                                        max_context=240_000, max_sources=max_sources)
    return str(out), out, fetched


PAGE = "MH370 disappeared on 8 March 2014 with 239 people aboard. " * 30


@pytest.mark.parametrize("url,on_topic,ok", [
    ("http://abc.onion/", False, False), ("http://abc.onion", False, False),
    ("http://abc.onion/", True, True),                                   # §4LG: topic beats path
    ("http://abc.onion/threads/mh370", False, True), ("http://abc.onion/index.php?topic=4921.0", False, True),
    ("http://abc.onion/js/get-banners.js", True, False), ("http://abc.onion/static/site.css", True, False),
    ("http://abc.onion/search/?q=mh370+theories", True, False)])
def test_onion_reading_candidates(url, on_topic, ok):
    assert D.is_onion_reading_candidate(url, on_topic) is ok


async def test_off_topic_homepages_and_assets_are_not_read_while_pages_exist(monkeypatch):
    ranked = [_row("http://home.onion/", title="Welcome", snippet="market"),
              _row("http://a.onion/js/app.js"), _row("http://b.onion/threads/mh370"),
              _row("http://c.onion/t/42")]
    _, _, fetched = await _research(monkeypatch, ranked, lambda u: PAGE)
    assert fetched == ["http://b.onion/threads/mh370", "http://c.onion/t/42"]


async def test_only_homepages_are_still_read(monkeypatch):
    ranked = [_row("http://home1.onion/", title="Welcome"), _row("http://home2.onion/", title="Index")]
    _, _, fetched = await _research(monkeypatch, ranked, lambda u: PAGE)
    assert sorted(fetched) == ["http://home1.onion/", "http://home2.onion/"]


async def test_a_ddos_guard_page_is_a_failed_source_and_is_backfilled(monkeypatch):
    ranked = [_row(f"http://s{i}.onion/thread/{i}") for i in range(5)]
    client = _Client()
    text, out, fetched = await _research(monkeypatch, ranked,
                                         lambda u: BOT_CHECK if u.startswith("http://s0.") else PAGE, client)
    assert "http://s3.onion/thread/3" in fetched                       # the first reserve
    assert not any("request is being verified" in t for t in client.sent)
    assert "bot-check page from http://s0.onion/thread/0" in text
    assert "SOURCE FAILURES" not in text and "[1 hidden service(s) failed to load and were replaced" in text


async def test_an_unreplaceable_failure_is_counted_in_slots(monkeypatch):
    ranked = [_row(f"http://s{i}.onion/thread/{i}") for i in range(3)]
    text, out, _ = await _research(monkeypatch, ranked,
                                   lambda u: FetchError("Error: received status 404 from x") if u.startswith("http://s0.") else PAGE)
    assert "[⚠ SOURCE FAILURES: 1 of 3 hidden services" in text
    assert getattr(out, "reason_code", "") == "darkweb_research_sources_partial"


async def test_all_loaded_onions_empty_is_said_plainly(monkeypatch):
    ranked = [_row(f"http://s{i}.onion/thread/{i}") for i in range(3)]
    text, out, _ = await _research(monkeypatch, ranked, lambda u: PAGE, _Client(lambda t: "No relevant information found."))
    assert "[⚠ NO RELEVANT FACTS: none of the 3 hidden services that loaded" in text
    assert getattr(out, "reason_code", "") == "darkweb_research_no_relevant_facts"


async def test_a_cached_no_facts_report_keeps_its_reason_code(monkeypatch):
    cached = "--- DARK-WEB RESEARCH RESULT ---\n[⚠ NO RELEVANT FACTS: none of the 3 hidden services…]\n"
    monkeypatch.setattr(D, "_cache_get", lambda k: cached)
    out = await D.tool_darkweb_research(query="mh370 theories", tor_proxy="socks5://127.0.0.1:9050",
                                        llm_client=_Client(), max_context=240_000)
    assert getattr(out, "reason_code", "") == "darkweb_research_no_relevant_facts"


def test_the_evidence_gate_reads_the_darkweb_no_facts_report_as_empty():
    from ghost_agent.core.evidence_gate import assess_turn_evidence
    body = ("--- DARK-WEB RESEARCH RESULT ---\n[⚠ NO RELEVANT FACTS: none of the 3 hidden services that loaded "
            "had facts …]\n\n### SOURCE: http://s0.onion/t\n[EDGE EXTRACTED FACTS]:\nNo relevant information found.")
    a = assess_turn_evidence([{"name": "darkweb_research", "content": body}])
    assert a.consulted == 1 and a.substantive == 0


# ── fresh-reader round r1 ─────────────────────────────────────────────

@pytest.mark.parametrize("url", ["http://abc.onion/leak-dump.txt", "http://abc.onion/canary.txt",
                                 "http://abc.onion/api/status.json", "http://abc.onion/feed.xml"])
def test_text_documents_are_read(url):
    """r1: on onion sites a paste, dump or canary IS the content."""
    assert D.is_onion_reading_candidate(url) is True


async def test_a_slot_whose_reserves_also_failed_is_not_called_replaced(monkeypatch):
    ranked = [_row(f"http://s{i}.onion/thread/{i}") for i in range(5)]
    bad = {"http://s0.onion/thread/0", "http://s3.onion/thread/3", "http://s4.onion/thread/4"}
    text, out, _ = await _research(monkeypatch, ranked,
                                   lambda u: FetchError("Error: received status 404 from x") if u in bad else PAGE)
    # §4UN-darkres r1: the banner counts what was TRIED (the [SOURCES:] totals), not slots
    assert "replaced" not in text and "[⚠ SOURCE FAILURES: 3 of 5 hidden services" in text
    assert "[SOURCES: tried 5, loaded 2, timed out 0, failed 3, skipped 0]" in text


async def test_a_run_whose_reserve_the_deadline_stopped_is_not_cached(monkeypatch):
    """r1: a reserve the deadline stopped left `_phase_truncated` False and the
    transiently truncated report was cached."""
    monkeypatch.setattr(D, "_darkweb_search_raw", AsyncMock(return_value=(
        [_row(f"http://s{i}.onion/thread/{i}") for i in range(3)], [], False, 3)))
    monkeypatch.setattr(D, "_cache_get", lambda k: None)
    puts = []
    monkeypatch.setattr(D, "_cache_put", lambda k, v: puts.append(k))

    async def _slots(urls, reserves, bounded, **kw):
        kw["stats"]["deadline_cut"] = 1
        return [[(u, f"### SOURCE: {u}\n[EDGE EXTRACTED FACTS]:\n- MH370 vanished in 2014.\n")] for u in urls]
    monkeypatch.setattr(S, "run_research_slots", _slots)
    await D.tool_darkweb_research(query="mh370 theories", tor_proxy="socks5://127.0.0.1:9050",
                                  llm_client=_Client(), max_context=240_000, max_sources=2)
    assert puts == []


def test_slot_coverage_counts_a_replacement_only_when_it_loaded():
    ok = "### SOURCE: u\n[EDGE EXTRACTED FACTS]:\n- fact"
    bad = "### SOURCE: u\nError: received status 404"
    slots = [[("a", bad), ("b", ok)], [("c", bad), ("d", bad), ("e", bad)], [("f", ok)], RuntimeError("x")]
    urls, blocks, lost, n, failed, replaced = S.slot_coverage(slots)
    assert (lost, n, failed, replaced) == (1, 4, 1, 1) and urls[-1] is None and len(blocks) == 6


# ── the extractor: an all-"none" run is rechecked once ────────────────

class _Seq(_Client):
    """Answers the first `n_none` extractions "nothing relevant", then facts."""
    def __init__(self, n_none, then="- MH370 vanished on 8 March 2014 with 239 aboard."):
        self.n = n_none
        super().__init__(lambda t: "No relevant information found." if len(self.sent) <= self.n else then)


async def _deep(monkeypatch, client, n=3, page=None):
    from tests.test_4nj_research_yield import _research, _results
    return await _research(monkeypatch, _results(n), lambda u: page or ("MH370 report text. " * 60 + u), client)


async def test_a_worker_fault_is_caught_by_the_recheck(monkeypatch):
    """10-10 19:4x: every source "none", on-topic pages included; the same
    request re-run gave facts. The recheck finds facts → every source is
    re-extracted and no false "no facts" reaches the model."""
    client = _Seq(3)
    text, out, _ = await _deep(monkeypatch, client)
    assert len(client.sent) == 6 and "NO RELEVANT FACTS:" not in text
    assert text.count("MH370 vanished") == 3
    assert client.sent[3] == client.sent[0] or client.sent[3] in client.sent[:3]   # the identical request


async def test_a_genuine_no_facts_run_costs_one_recheck(monkeypatch):
    client = _Client(lambda t: "No relevant information found.")
    text, out, _ = await _deep(monkeypatch, client)
    assert len(client.sent) == 4 and "[⚠ NO RELEVANT FACTS: none of the 3 sources" in text


async def test_the_largest_page_is_rechecked_first(monkeypatch):
    client = _Client(lambda t: "No relevant information found.")
    from tests.test_4nj_research_yield import _research, _results
    res = _results(3)
    sizes = {res[0]["href"]: 50, res[1]["href"]: 400, res[2]["href"]: 100}
    await _research(monkeypatch, res, lambda u: "MH370 text. " * sizes[u], client)
    assert len(client.sent[3]) == max(len(t) for t in client.sent[:3])


async def test_no_recheck_when_any_source_had_facts(monkeypatch):
    replies = iter(["No relevant information found.", "- fact one", "No relevant information found."])
    client = _Client(lambda t: next(replies))
    await _deep(monkeypatch, client)
    assert len(client.sent) == 3


async def test_no_recheck_without_time_left(monkeypatch):
    monkeypatch.setattr(S, "RECHECK_MIN_BUDGET_S", 10_000.0)
    client = _Client(lambda t: "No relevant information found.")
    text, _, _ = await _deep(monkeypatch, client)
    assert len(client.sent) == 3 and "NO RELEVANT FACTS:" in text


async def test_a_failing_recheck_leaves_the_report_as_it_was(monkeypatch):
    class _Boom(_Client):
        async def chat_completion(self, payload, **kw):
            if len(self.sent) >= 3:
                raise RuntimeError("worker down")
            return await super().chat_completion(payload, **kw)
    client = _Boom(lambda t: "No relevant information found.")
    text, out, _ = await _deep(monkeypatch, client)
    assert "[⚠ NO RELEVANT FACTS: none of the 3 sources" in text


async def test_the_onion_path_is_rechecked_too(monkeypatch):
    ranked = [_row(f"http://s{i}.onion/thread/{i}") for i in range(3)]
    client = _Seq(3)
    text, out, _ = await _research(monkeypatch, ranked, lambda u: PAGE + u, client)
    assert len(client.sent) == 6 and "NO RELEVANT FACTS:" not in text and text.count("MH370 vanished") == 3


async def test_both_tools_send_the_one_extractor_prompt(monkeypatch):
    """Behaviour, not source text: what each tool SENDS is `extract_prompt`."""
    c1 = _Client()
    await _deep(monkeypatch, c1, n=1, page="ZFS page text. " * 40)
    c2 = _Client()
    await _research(monkeypatch, [_row("http://s0.onion/thread/0")], lambda u: PAGE, c2)
    assert c1.sent[0].startswith(S.extract_prompt("openzfs vs btrfs reliability", "")[:-1].split("Source text:")[0])
    assert c2.sent[0].startswith(S.extract_prompt("mh370 theories", "").split("Source text:")[0])
    assert c1.sent[0].split("Source text:")[0] != c2.sent[0].split("Source text:")[0]   # the query is in it


# ── fresh-reader round r2 (the recheck) ───────────────────────────────

def _blocks(n):
    urls = [f"https://s{i}.example/p" for i in range(n)]
    blocks = [f"### SOURCE: {u}\n[EDGE EXTRACTED FACTS]:\nNo relevant information found.\n" for u in urls]
    return urls, blocks


class _Scripted:
    def __init__(self, answers):
        self.answers, self.sent = answers, []

    async def chat_completion(self, payload, **kw):
        text = payload["messages"][0]["content"]
        self.sent.append((text, kw))
        a = self.answers(text)
        if isinstance(a, Exception):
            raise a
        return {"choices": [{"message": {"content": a}}]}


async def test_one_failing_recheck_does_not_stop_the_others_and_is_reported():
    import time
    urls, blocks = _blocks(3)
    texts = {urls[0]: ("big " * 300 + "s0", 200, 5.0), urls[1]: ("mid s1", 200, 5.0), urls[2]: ("s2", 200, 5.0)}
    c = _Scripted(lambda t: RuntimeError("busy") if t.endswith("mid s1") else "- a real fact")
    r = await S.recheck_all_none(blocks, urls, texts, "q", c, "m", deadline=time.monotonic() + 300, slot_wait=1.0)
    assert r.anomaly and r.fixed == 2 and r.unchecked == 1
    assert "1 source(s) could not be re-checked" in S.recheck_note(r)
    assert all(kw["timeout"] <= 65.0 for _, kw in c.sent)                  # capped per call, not the whole phase


async def test_rechecks_run_in_parallel():
    import asyncio, time
    urls, blocks = _blocks(4)
    texts = {u: (u, 200, 1.0) for u in urls}
    live = {"now": 0, "max": 0}

    class _Slow(_Scripted):
        async def chat_completion(self, payload, **kw):
            live["now"] += 1; live["max"] = max(live["max"], live["now"])
            await asyncio.sleep(0.05)
            live["now"] -= 1
            return await super().chat_completion(payload, **kw)
    r = await S.recheck_all_none(blocks, urls, texts, "q", _Slow(lambda t: "- fact"), "m",
                                 deadline=time.monotonic() + 300, slot_wait=1.0, fanout=3)
    assert r.fixed == 4 and live["max"] == 3


async def test_a_page_whose_plan_does_not_fit_is_left_unchecked():
    import time
    urls, blocks = _blocks(3)
    texts = {urls[0]: ("big " * 300, 200, 5.0), urls[1]: ("a", 200, 500.0), urls[2]: ("b", 200, 500.0)}
    c = _Scripted(lambda t: "- fact")
    r = await S.recheck_all_none(blocks, urls, texts, "q", c, "m", deadline=time.monotonic() + 120, slot_wait=10.0)
    assert r.fixed == 1 and r.unchecked == 2 and len(c.sent) == 1


async def test_a_none_recheck_never_replaces_a_block():
    import time
    urls, blocks = _blocks(3)
    texts = {u: (u + " x" * i, 200, 1.0) for i, u in enumerate(urls)}
    seq = iter(["- fact", "", "No relevant information found."])
    r = await S.recheck_all_none(blocks, urls, texts, "q", _Scripted(lambda t: next(seq)), "m",
                                 deadline=time.monotonic() + 300, slot_wait=1.0, fanout=1)
    assert r.fixed == 1 and r.unchecked == 0
    assert sum("No relevant information found." in b for b in r.contents) == 2


async def test_a_url_read_twice_is_rechecked_once():
    import time
    urls, blocks = _blocks(2)
    urls, blocks = urls + [urls[0]], blocks + [blocks[0]]
    texts = {urls[0]: ("big " * 50, 200, 1.0), urls[1]: ("small", 200, 1.0)}
    c = _Scripted(lambda t: "- fact")
    await S.recheck_all_none(blocks, urls, texts, "q", c, "m", deadline=time.monotonic() + 300, slot_wait=1.0)
    assert len(c.sent) == 2


async def test_a_no_facts_darkweb_report_is_never_cached(monkeypatch):
    """r2: a "no facts" may be the worker fault itself — a cached copy
    would repeat it for 5 minutes."""
    puts = []
    ranked = [_row(f"http://s{i}.onion/thread/{i}") for i in range(3)]
    monkeypatch.setattr(D, "_darkweb_search_raw", AsyncMock(return_value=(ranked, [], False, 3)))
    monkeypatch.setattr(D, "_cache_get", lambda k: None)
    monkeypatch.setattr(D, "_cache_put", lambda k, v: puts.append(k))

    async def _fetch(url, tor_proxy):
        return PAGE
    monkeypatch.setattr(D, "_fetch_onion_text", _fetch)
    out = await D.tool_darkweb_research(query="mh370 theories", tor_proxy="socks5://127.0.0.1:9050",
                                        llm_client=_Client(lambda t: "No relevant information found."),
                                        max_context=240_000, max_sources=3)
    assert getattr(out, "reason_code", "") == "darkweb_research_no_relevant_facts" and puts == []


class _FaultThenBusy(_Client):
    """3 first answers "none" (the fault); the first recheck finds facts; every
    later recheck fails (worker busy)."""
    def __init__(self):
        super().__init__(lambda t: "No relevant information found." if len(self.sent) <= 3
                         else "- MH370 vanished on 8 March 2014.")

    async def chat_completion(self, payload, **kw):
        if len(self.sent) >= 4:
            self.sent.append(payload["messages"][-1]["content"])
            raise RuntimeError("worker busy")
        return await super().chat_completion(payload, **kw)


async def test_deep_research_says_which_sources_could_not_be_rechecked(monkeypatch):
    text, out, _ = await _deep(monkeypatch, _FaultThenBusy())
    assert "2 source(s) could not be re-checked in time" in text and "NO RELEVANT FACTS:" not in text


async def test_an_onion_run_with_unrechecked_sources_is_not_cached(monkeypatch):
    puts = []
    ranked = [_row(f"http://s{i}.onion/thread/{i}") for i in range(3)]
    monkeypatch.setattr(D, "_darkweb_search_raw", AsyncMock(return_value=(ranked, [], False, 3)))
    monkeypatch.setattr(D, "_cache_get", lambda k: None)
    monkeypatch.setattr(D, "_cache_put", lambda k, v: puts.append(k))

    async def _fetch(url, tor_proxy):
        return PAGE + url
    monkeypatch.setattr(D, "_fetch_onion_text", _fetch)
    out = await D.tool_darkweb_research(query="mh370 theories", tor_proxy="socks5://127.0.0.1:9050",
                                        llm_client=_FaultThenBusy(), max_context=240_000, max_sources=3)
    assert "could not be re-checked" in str(out) and puts == []


# ── fresh-reader round r3 ─────────────────────────────────────────────

async def test_a_malformed_plan_estimate_never_escapes():
    import time
    urls, blocks = _blocks(2)
    texts = {urls[0]: ("big " * 40, 200, "not-a-number"), urls[1]: ("s", 200, None)}
    r = await S.recheck_all_none(blocks, urls, texts, "q", _Scripted(lambda t: "- fact"), "m",
                                 deadline=time.monotonic() + 300, slot_wait=1.0)
    assert r.fixed == 2


async def test_the_largest_page_that_fits_is_rechecked_first():
    import time
    urls, blocks = _blocks(2)
    texts = {urls[0]: ("big " * 300, 200, 1000.0), urls[1]: ("small", 200, 5.0)}
    c = _Scripted(lambda t: "- fact")
    r = await S.recheck_all_none(blocks, urls, texts, "q", c, "m", deadline=time.monotonic() + 120, slot_wait=1.0)
    assert r.anomaly and c.sent[0][0].endswith("small") and r.unchecked == 1


async def test_a_hung_recheck_is_cut_at_its_budget():
    import asyncio, time

    class _Hang(_Scripted):
        async def chat_completion(self, payload, **kw):
            await asyncio.sleep(30)
    urls, blocks = _blocks(1)
    t0 = time.monotonic()
    r = await S.recheck_all_none(blocks, urls, {urls[0]: ("x", 200, 0.0)}, "q", _Hang(lambda t: ""), "m",
                                 deadline=time.monotonic() + 30, slot_wait=1.0, per_call_s=0.5)
    assert time.monotonic() - t0 < 5 and not r.anomaly
