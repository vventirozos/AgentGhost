"""§4UY — internet tools (findings N1–N3, N7, N9–N12, N14–N18 of §4UX-1; N4, N6 and N5/N8/N13/N19 have
their own files: test_4uy_internet_egress.py, test_4uy_internet_provenance.py, test_4uy_internet_browser.py).

Each test names the world it fails in: the code as it was before §4UY. Fictional data only."""
from __future__ import annotations

import ast
import asyncio
import json
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core import link_grounding as LG
from ghost_agent.tools import darkweb_search as D
from ghost_agent.tools import search as S
from ghost_agent.tools import tool_failure as TF
from ghost_agent.tools.fallback_chains import FALLBACK_CHAINS, get_fallback_chain_hint
from ghost_agent.tools.outcome import OutcomeStatus, ToolOutcome
from tests.test_4un_deep_research import _Client

SRC = Path(__file__).resolve().parents[1] / "src" / "ghost_agent"
PROXY = "socks5h://127.0.0.1:9050"

TRANSPORT = ("ERROR: web search failed on TRANSPORT, not on the query — 9 engine answer(s) were refusals "
             "(HTTP 403/429 or a captcha page) and 0 attempt(s) could not connect or timed out over Tor (0 "
             "answered empty). Rephrasing will not help right now. Retry the SAME query once after other "
             "work, or answer from your own knowledge and say web search was unreachable.")


def _ddgs_module(by_engine):
    mod = MagicMock(); cls = MagicMock(); mod.DDGS = cls
    inst = MagicMock(); cls.return_value.__enter__.return_value = inst
    inst.text.side_effect = by_engine
    return mod, inst


def _drive(monkeypatch, script, query):
    """web_search with its waves replaced: script(query, wave, stats, kw) -> rows. (out, calls)."""
    calls = []

    async def _wave(q, proxy, wave, max_results=20, stats=None, **kw):
        calls.append((q, wave, dict(kw)))
        return list(script(q, wave, stats, kw) or [])
    monkeypatch.setattr(S, "_race_search_wave", _wave)
    monkeypatch.setattr(S, "_proxy_reachable", lambda *a, **k: True)
    monkeypatch.setattr(S, "_page_reads_on", lambda: False)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    S._SEARCH_CACHE.clear()
    return asyncio.run(S.tool_search_ddgs(query, PROXY)), calls


# ── N1: an internet tool's failure carries its own next step — no second reader ────────────────

def test_no_hint_table_and_no_chain_covers_an_internet_tool():
    """Fails where `_FALLBACK_HINTS["web_search"]` turned the words "captcha page" of a TRANSPORT failure
    into "retry with a SHORTER query", and the chains sent web_search ↔ deep_research (same engines)."""
    assert not (set(TF._FALLBACK_HINTS) & LG.RESEARCH_TOOLS)
    assert not (set(FALLBACK_CHAINS) & LG.RESEARCH_TOOLS)
    for tool in LG.RESEARCH_TOOLS:
        for text in (TRANSPORT, "ERROR: web search returned ZERO results", "CAPTCHA", "search phase failed",
                     "CRITICAL ERROR: Deep Research search phase failed"):
            assert TF.get_fallback_hint(tool, text) is None
            assert get_fallback_chain_hint(tool, text) is None


def test_the_real_transport_failure_gets_no_contradicting_hint(monkeypatch):
    def script(q, wave, stats, kw):
        stats.setdefault("cats", {})["blocked"] = stats.get("cats", {}).get("blocked", 0) + 3
        return []
    out, _ = _drive(monkeypatch, script, "electric car battery lifespan")
    assert "failed on TRANSPORT" in out
    assert TF.get_fallback_hint("web_search", out) is None and get_fallback_chain_hint("web_search", out) is None


# ── N2: deep research never reads a batch about the query's first word ──────────────────────────

Q_LOW = "revolut breach leaked passports"
LOW_ROWS = [{"href": f"https://bank{i}.example/revolut", "title": f"Revolut app review {i}",
             "body": "Revolut is a fintech company offering cards, savings and accounts across Europe."}
            for i in range(6)]
ON_ROWS = [{"href": f"https://news{i}.example/leak", "title": "Revolut breach: leaked passports",
            "body": "A breach at Revolut leaked customer passports, the company said."} for i in range(3)]


async def _research(monkeypatch, waves, query=Q_LOW, **kw):
    """deep_research with scripted waves (one row list per call, the last repeated) and a recording fetch."""
    monkeypatch.setattr(S.importlib.util, "find_spec", lambda name: True)
    seen = []

    async def _wave(q, proxy, wave, max_results=15, stats=None, **k):
        seen.append(q)
        return list(waves[min(len(seen) - 1, len(waves) - 1)])
    monkeypatch.setattr(S, "_race_search_wave", _wave)
    fetched = []

    async def _fetch(url, **k):
        fetched.append(url)
        return "Revolut breach: passports leaked. " * 20
    monkeypatch.setattr(S, "helper_fetch_url_content", _fetch)
    out = await S.tool_deep_research(query, False, None, llm_client=_Client(), max_context=240_000, **kw)
    return out, fetched, seen


@pytest.mark.asyncio
async def test_deep_research_reads_no_page_when_every_batch_is_low_relevance(monkeypatch):
    """Fails where deep research raced without the low-relevance rule and read all six pages."""
    out, fetched, seen = await _research(monkeypatch, [LOW_ROWS])
    assert fetched == []
    assert isinstance(out, ToolOutcome) and out.status is OutcomeStatus.FAILED
    assert out.reason_code == "research_low_relevance" and S.RESEARCH_LOW_HEAD in out
    assert "Revolut app review 0" in out and len(seen) >= 2                 # the second wave was tried


@pytest.mark.asyncio
async def test_deep_research_reads_the_wave_that_is_about_the_query(monkeypatch):
    out, fetched, _ = await _research(monkeypatch, [LOW_ROWS, ON_ROWS])
    assert fetched and all("news" in u for u in fetched)
    assert not any("bank" in u for u in fetched)


def test_a_batch_with_too_little_text_is_not_judged_low():
    """The off-topic rule's own evidence floor: bare rows are no evidence."""
    rows = [{"href": "https://a.example/x", "title": "t", "body": "b"}]
    assert S.research_batch_is_low(Q_LOW, rows) is False
    assert S.research_batch_is_low(Q_LOW, LOW_ROWS) is True
    assert S.research_batch_is_low(Q_LOW, ON_ROWS) is False


def test_every_race_is_soft_on_a_low_batch():
    """Fails where only web_search (a context variable) raced softly: the counter-search and deep research
    took the first low batch."""
    def by_engine(q, **kw):
        if kw["backend"] == "duckduckgo":
            return [{"title": "Kubernetes pricing", "body": "cloud", "href": "https://k.example/a"}]
        time.sleep(0.5)
        return [{"title": "Kubernetes ingress nginx timeout", "body": "ingress nginx timeout",
                 "href": "https://d.example"}]
    mod, _ = _ddgs_module(by_engine)
    with patch.dict("sys.modules", {"ddgs": mod}):
        rows = asyncio.run(S._race_search_wave("kubernetes ingress nginx timeout", None, 0))
    assert rows[0]["href"] == "https://d.example"


# ── N3: a low batch whose rivals were only refused gets a wave on fresh circuits ────────────────

def test_low_batch_unheard_reads_the_rivals():
    assert S.low_batch_unheard("yandex", [("yandex", "low-relevance"), ("yahoo", "blocked: HTTP 403"),
                                          ("brave", "engine brave timed out after 12s")]) is True
    assert S.low_batch_unheard("yandex", [("yandex", "low-relevance"), ("yahoo", "No results found.")]) is False
    assert S.low_batch_unheard("yandex", [("yandex", "low-relevance"), ("yahoo", "off-topic")]) is False
    assert S.low_batch_unheard("yandex", [("yandex", "low-relevance"), ("yahoo", "low-relevance")]) is False
    # §4VA: the Brave onion is a general engine like the others — its answer counts as heard
    assert S.low_batch_unheard("yandex", [("yandex", "low-relevance"), (S.BRAVE_ONION_ENGINE, "empty"),
                                          ("yahoo", "blocked: x")]) is False


def _race_stats(by_engine, monkeypatch):
    monkeypatch.setattr(S, "LOW_RELEVANCE_GRACE_S", 0.3)
    mod, _ = _ddgs_module(by_engine)
    stats = {}
    with patch.dict("sys.modules", {"ddgs": mod}):
        rows = asyncio.run(S._race_search_wave("kubernetes ingress nginx timeout", None, 0, stats=stats))
    return rows, stats


def test_the_race_says_when_the_low_batch_stood_only_because_every_rival_was_refused(monkeypatch):
    def refused(q, **kw):
        if kw["backend"] == "yandex":
            return [{"title": "Kubernetes pricing", "body": "cloud", "href": "https://k.example/a"}]
        raise Exception("blocked: HTTP 403 from engine")
    rows, stats = _race_stats(refused, monkeypatch)
    assert rows and stats["low_unheard"] is True

    def answered(q, **kw):
        if kw["backend"] == "yandex":
            return [{"title": "Kubernetes pricing", "body": "cloud", "href": "https://k.example/a"}]
        raise Exception("No results found.")
    rows, stats = _race_stats(answered, monkeypatch)
    assert rows and stats["low_unheard"] is False


def test_web_search_runs_a_second_wave_for_an_unheard_low_batch(monkeypatch):
    """Fails where the low batch ended the search (12 of 146 live calls; the model re-searched 12/12)."""
    low = [{"title": "Kubernetes pricing", "body": "cloud costs", "href": "https://k.example/a"}]
    good = [{"title": "Kubernetes ingress nginx timeout", "body": "ingress nginx timeout settings",
             "href": "https://docs.example/ingress"}]

    def script(q, wave, stats, kw):
        stats["low_unheard"] = wave == 0
        return low if wave == 0 else good
    out, calls = _drive(monkeypatch, script, "kubernetes ingress nginx timeout")
    assert [w for _, w, _ in calls] == [0, 1]
    assert "docs.example/ingress" in out and S.LOW_RELEVANCE_NOTE not in out

    def script2(q, wave, stats, kw):
        stats["low_unheard"] = True
        return low
    out, calls = _drive(monkeypatch, script2, "kubernetes ingress nginx timeout")
    assert [w for _, w, _ in calls] == [0, 1] and S.LOW_RELEVANCE_NOTE in out   # the first low batch stands


# ── N7: the sources reach the reply mechanically ───────────────────────────────────────────────

DR_ROW = {"name": "deep_research", "content": (
    "--- DEEP RESEARCH RESULT ---\n### SOURCE: https://news.example/elena-award\n"
    "[PAGE] Date: published 2026-10-01 | Title: \"Award\"\n[EDGE EXTRACTED FACTS]:\n- Elena won the award.\n")}
WS_ROW = {"name": "web_search", "content": "### 1. Markos opens a bakery\nsnippet\n[Source: https://town.example/markos]"}
ELENA_REPLY = "Elena won the award in October, the jury said, by a wide margin."


def test_the_sources_line_lists_the_pages_read():
    note = LG.sources_note(ELENA_REPLY, [DR_ROW])
    assert note.startswith("\n\n" + LG.SOURCES_HEAD) and "- https://news.example/elena-award" in note


def test_no_sources_line_when_the_reply_links_one_or_nothing_was_seen():
    assert LG.sources_note("Elena won it (https://news.example/elena-award).", [DR_ROW]) == ""
    assert LG.sources_note("Done.", [{"name": "execute", "content": "ok"}]) == ""
    assert LG.sources_note("", [DR_ROW]) == ""


def test_a_search_only_answer_lists_the_results_as_snippets():
    note = LG.sources_note("Markos opened a bakery on the town square this spring.", [WS_ROW])
    assert LG.SOURCES_SNIPPET_HEAD in note and "https://town.example/markos" in note


def test_the_sources_line_is_bounded():
    rows = [{"name": "browser", "content": f"--- BROWSER RESULT ---\nSTATUS: OK\nURL: https://s{i}.example/p\n"}
            for i in range(9)]
    note = LG.sources_note("An answer that is long enough to be a report of the pages.", rows)
    assert note.count("\n- https://") == LG.SOURCES_MAX and f"(+{9 - LG.SOURCES_MAX} more)" in note


@pytest.mark.asyncio
async def test_finalize_appends_the_sources_to_the_reply(tmp_path):
    """Fails where the reply went out with no source (0 of 95 live research replies carried a URL)."""
    from tests.test_finalize_stream_pins import _fs, make_fin_agent
    a = make_fin_agent()
    a.context.memory_dir = str(tmp_path / "mem")
    out, _, _ = await a._finalize_and_return(_fs(final_ai_content=ELENA_REPLY,
                                                 tools_run_this_turn=[DR_ROW]))
    assert ELENA_REPLY in out
    assert LG.SOURCES_HEAD in out and "https://news.example/elena-award" in out


@pytest.mark.asyncio
async def test_a_streamed_reply_gets_the_sources_after_the_answer():
    from tests.test_4oh_open_items import _client_view, _drain
    frames, durable = await _drain("", [], [ELENA_REPLY], tools=[DR_ROW])
    view = _client_view(frames)
    assert view.index("Elena won the award") < view.index(LG.SOURCES_HEAD)
    assert "https://news.example/elena-award" in view and "https://news.example/elena-award" in durable


# §4UY N7 r2 (independent review): the list is for research answers to a reader, from public pages only

def test_a_coding_turn_gets_no_sources_list():
    """Reviewer repros A/B: a browser check of the agent's own app, a downloaded dependency."""
    br = "--- BROWSER RESULT ---\nSTATUS: OK\nURL: http://localhost:5000/\nTITLE: My app\nTEXT: hello"
    assert LG.sources_note("The app works: the page shows hello, and the form submits.",
                           [{"name": "browser", "content": br}]) == ""
    dl = "SUCCESS: Downloaded 'https://raw.example.com/x/y/main/data.csv' to data.csv (2 KB)"
    assert LG.sources_note("Done, the tests pass (12 passed) with the new data file.",
                           [{"name": "file_system", "content": dl}]) == ""


def test_a_short_or_form_fixed_answer_gets_no_list():
    """Reviewer repros C/D: a number-only answer; a short Greek answer; a request that fixes the form."""
    ws = {"name": "web_search", "content": "### 1. BTC price\nsnippet 91,000 USD\n[Source: https://coins.example/btc]"}
    assert LG.sources_note("91000", [ws]) == ""
    assert LG.sources_note("Η τιμή είναι 91.000 δολάρια.", [ws]) == ""
    long_reply = "The price is 91,000 USD according to the market page, as of this morning."
    assert LG.sources_note(long_reply, [ws])                                    # the control
    for req in ("Reply only with the number", "answer in JSON", "give me the price, no links please",
                "Answer only with the year", "Find the price of Bitcoin, just the number", "no citations"):
        assert LG.sources_note(long_reply, [ws], req) == "", req
    # review r2: words that shape the CONTENT do not fix the reply's form (reviewer's requests)
    for req in ("Search the web and give me just the top 3 restaurants in Athens",
                "What's new with the Mars mission? Give me only the important parts",
                "Look up the latest Python release and write only a short summary",
                "Can you research heat pumps for me and tell me exactly which one to buy?",
                "Find reviews of the Framework laptop, just give me a quick overview",
                "Research why there are no sources of lithium in Greece"):
        assert LG.sources_note(long_reply, [ws], req), req
    assert LG.sources_note('{"price": 91000, "currency": "USD", "source": "market page", "ok": true}', [ws]) == ""


@pytest.mark.parametrize("url,public", [
    ("https://news.example/a", True), ("http://localhost:8000/x", False), ("http://127.0.0.1/x", False),
    ("http://10.0.0.5/x", False), ("http://192.168.1.4/", False), ("http://100.101.102.103/", False),
    ("https://eva.tail0000.ts.net/x", False), ("http://printer.local/", False), ("http://intranet/", False),
    ("http://[::1]:8080/", False), ("http://" + "a" * 56 + ".onion/", False), ("https://8.8.8.8/dns", True)])
def test_only_public_pages_are_listed(url, public):
    assert LG.is_public_source_url(url) is public
    row = {"name": "browser", "content": f"--- BROWSER RESULT ---\nSTATUS: OK\nURL: {url}\n"}
    assert bool(LG.sources_note(ELENA_REPLY, [row])) is public


def test_a_contained_turn_gets_no_list(monkeypatch):
    from types import SimpleNamespace
    assert LG.sources_for_reply(ELENA_REPLY, [DR_ROW], "", SimpleNamespace())
    assert LG.sources_for_reply(ELENA_REPLY, [DR_ROW], "", SimpleNamespace(_subagent_allowed_tools=frozenset({"x"}))) == ""
    assert LG.sources_for_reply(ELENA_REPLY, [DR_ROW], "", SimpleNamespace(trajectory_extra_static={"leaf_id": "l1"})) == ""
    monkeypatch.setenv("GHOST_REPLY_SOURCES", "0")
    assert LG.sources_for_reply(ELENA_REPLY, [DR_ROW], "", SimpleNamespace()) == ""


@pytest.mark.asyncio
async def test_finalize_reads_the_request_and_the_turn(tmp_path):
    """The finalize sink passes the owner's request (a form-fixing ask gets no list) and the context."""
    from tests.test_finalize_stream_pins import _fs, make_fin_agent
    a = make_fin_agent()
    a.context.memory_dir = str(tmp_path / "mem")
    out, _, _ = await a._finalize_and_return(_fs(final_ai_content=ELENA_REPLY, tools_run_this_turn=[DR_ROW],
                                                 last_user_content="Tell me who won, no links please"))
    assert LG.SOURCES_HEAD not in out
    a = make_fin_agent()
    a.context.memory_dir = str(tmp_path / "mem2")
    a.context.trajectory_extra_static = {"leaf_id": "leaf-1"}
    out, _, _ = await a._finalize_and_return(_fs(final_ai_content=ELENA_REPLY, tools_run_this_turn=[DR_ROW]))
    assert LG.SOURCES_HEAD not in out


@pytest.mark.asyncio
async def test_the_verifier_judges_the_reply_without_the_list(tmp_path):
    """Fails where the list was appended before the verifier gate (the judge read the code's links as claims)."""
    from tests.test_finalize_stream_pins import _fs, make_fin_agent
    a = make_fin_agent()
    a.context.memory_dir = str(tmp_path / "mem")
    a.context.verifier = MagicMock()
    seen = []

    async def _gate(**kw):
        seen.append(kw.get("final_ai_content"))
        return None, None
    a._compute_verifier_verdict_gated = _gate
    out, _, _ = await a._finalize_and_return(_fs(final_ai_content=ELENA_REPLY, tools_run_this_turn=[DR_ROW]))
    assert LG.SOURCES_HEAD in out
    assert seen and all(LG.SOURCES_HEAD not in str(x) for x in seen)


@pytest.mark.asyncio
async def test_the_streamed_verifier_judges_the_reply_without_the_list():
    """The late (streamed) verdict reads the reply as streamed — less the code's list appended after it."""
    from tests.test_4oh_open_items import _state
    from tests.test_finalize_stream_pins import make_stream_agent, sse
    a = make_stream_agent()
    a.context.args.no_verifier = False
    a.context.verifier = MagicMock()
    a.context.journal = MagicMock()
    a.context.metacog = MagicMock(); a.context.metacog.enabled = False
    a._journal_append_safe = AsyncMock(); a._record_episode_safe = AsyncMock()
    a._judge_hydration_safe = MagicMock(); a._write_project_work_log_safe = AsyncMock()
    a._record_turn_trajectory = MagicMock()
    a._attach_late_verdict_handler = MagicMock()
    seen = []

    async def _verdict(**kw):
        seen.append(kw.get("final_ai_content"))
        return None
    a._compute_verifier_verdict = _verdict

    async def final_stream(payload, use_coding=False):
        yield sse({"content": ELENA_REPLY})
        yield b"data: [DONE]\n\n"
    a.context.llm_client.stream_chat_completion = final_stream
    reg = MagicMock(); reg.is_cancelled.return_value = False
    st = _state(reg, "", [])
    st.stream_tools_snapshot = [DR_ROW]
    gen, _, _ = a._stream_final_generation(st)
    _ = [c async for c in gen]
    await asyncio.sleep(0.05)
    durable = a._record_turn_trajectory.call_args.kwargs["final_content"]
    assert LG.SOURCES_HEAD in durable
    assert seen and ELENA_REPLY in seen[0] and LG.SOURCES_HEAD not in seen[0]


def test_every_final_reply_sink_uses_the_one_rule():
    """Enumeration: `sources_note` is called only through `sources_for_reply` (the leaf/off-switch gate)."""
    tree = ast.parse((SRC / "core" / "agent.py").read_text())
    names = {getattr(n.func, "id", getattr(n.func, "attr", "")) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert "sources_note" not in names and "_sources_note" not in names
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert "sources_note" not in imported and "sources_for_reply" in imported


def test_voice_never_reads_the_list_on_the_handheld():
    import sys
    sys.path.insert(0, str(SRC.parents[1] / "interface" / "externals" / "clockwork_ghost"))
    import speech
    note = LG.sources_note(ELENA_REPLY, [DR_ROW, WS_ROW])
    text = ELENA_REPLY + note
    for step in (1, 5, 13, len(text)):
        ch = speech.SpeechChunker()
        out = []
        for i in range(0, len(text), step):
            out += ch.feed(text[i:i + step])
        out += ch.flush()
        spoken = " ".join(out)
        assert "Elena won the award" in spoken and "link" not in spoken and "Sources" not in spoken, (step, out)


# the web client's three speech paths: tests/test_4kp_think_split_and_clients.py (one helper,
# `_stripSourcesList`, read through the suite's app.js fixture)


# ── N9: dead engines sit out; an empty page is kept once for the operator ─────────────────────

@pytest.fixture
def _ledger(monkeypatch):
    monkeypatch.setattr(S, "_ENGINE_WAVES", __import__("collections").deque(maxlen=4000))
    monkeypatch.setattr(S, "_bench_state", {"waves": 0})
    return S._ENGINE_WAVES


def _waves(ledger, n, pairs, t0=None, step=60.0):
    # §4VA r1: recent by default — a benched engine whose last outcome is over half an hour old is
    # raced again as a probe (tests/test_4va_search_engines.py), and these pin the rest itself
    import time as _time
    t0 = (_time.time() - 30.0 - (n - 1) * step) if t0 is None else t0
    for i in range(n):
        ledger.append((t0 + i * step, tuple(pairs)))


def test_an_engine_that_only_refused_for_thirty_waves_sits_out(_ledger):
    _waves(_ledger, 30, [("duckduckgo", "blocked"), ("yandex", "win"), ("brave", "blocked"), ("yahoo", "empty")])
    raced = S.raced_engines(None)
    assert "duckduckgo" not in raced and "brave" not in raced and {"yandex", "yahoo"} <= set(raced)


def test_one_answer_or_a_short_burst_keeps_an_engine_racing(_ledger):
    _waves(_ledger, 29, [("duckduckgo", "blocked"), ("yandex", "win")])
    _waves(_ledger, 1, [("duckduckgo", "empty"), ("yandex", "win")], t0=2_000_000.0)
    assert "duckduckgo" in S.raced_engines(None)
    _ledger.clear()
    _waves(_ledger, 40, [("brave", "blocked"), ("yandex", "win")], step=1.0)       # 40 s of bad Tor
    assert "brave" in S.raced_engines(None)


def test_a_benched_engine_is_probed_and_two_always_race(_ledger):
    _waves(_ledger, 30, [(e, "blocked") for e in S._RACE_ENGINES])
    # never below two (the Brave onion is switched off in the suite and needs a proxy — conftest, §4VA)
    assert S.raced_engines(None) == tuple(e for e in S._RACE_ENGINES if e != S.BRAVE_ONION_ENGINE)
    _ledger.clear()
    _waves(_ledger, 30, [("duckduckgo", "blocked"), ("yandex", "win"), ("yahoo", "win"), ("brave", "win")])
    seen = [("duckduckgo" in S.raced_engines(None)) for _ in range(S.ENGINE_BENCH_PROBE_EVERY)]
    assert seen.count(True) == 1


def test_an_empty_engine_page_is_kept_once_as_structure_only(monkeypatch, tmp_path):
    """r2 (review): the query echoes in forms no strip enumerates — partial words, encodings, attributes.
    Only the page's outline is kept: tags, class/id names, text-run sizes."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.setattr(S, "_EMPTY_SAMPLED", set())
    q = "elena kosmas bakery"
    page = ('<html><head><title>Elena Kosmas Bakery - Search</title><script>var q="elena+kosmas"</script></head>'
            '<body class="srp consent-wall"><form action="/s?p=elena%20kosm" id="q-elenakos">'
            '<input value="ELENA KOSMAS"></form><p>We value your privacy. elen kosm bak&#101;ry</p>'
            '<a href="https://r.example/?q=kosmas-bakery">x</a></body></html>')
    p = S.keep_empty_page_sample("yahoo", page, q)
    assert p and Path(p).parent == tmp_path / "system" / "search_empty_samples"
    body = Path(p).read_text().lower()
    assert "class:consent-wall" in body and "<form>" in body and "#text" in body
    for frag in ("elen", "kosm", "bak", "privacy", "r.example", "%20"):
        assert frag not in body, frag
    assert S.keep_empty_page_sample("yahoo", page, q) is None      # once per engine


def test_the_engine_guard_keeps_a_sample_of_a_zero_result_200(monkeypatch):
    kept = []
    monkeypatch.setattr(S, "keep_empty_page_sample", lambda e, h, q: kept.append(e))
    eng = MagicMock()
    eng.name = "yahoo"; eng.search_method = "GET"
    eng.http_client.request.return_value = MagicMock(status_code=200, text="<html>nothing here</html>")
    eng.post_extract_results.return_value = []
    assert S._guarded_engine_search(eng, "markos bakery") == []
    assert kept == ["yahoo"]


# ── N10: deep_research reads the pages the model names ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_deep_research_reads_named_urls_without_a_query(monkeypatch):
    onion = "http://" + "a" * 56 + ".onion/x"
    out, fetched, seen = await _research(
        monkeypatch, [ON_ROWS], query="",
        urls=["https://docs.example/report.pdf", "https://news.example/a", onion, "ftp://x.example/f"])
    assert fetched[:2] == ["https://docs.example/report.pdf", "https://news.example/a"]
    assert len(fetched) == 2 and seen == []                                       # no search ran
    assert "darkweb_research reads those" in out and "not an http(s) address" in out


@pytest.mark.asyncio
async def test_named_urls_come_first_and_the_query_fills_the_rest(monkeypatch):
    out, fetched, seen = await _research(monkeypatch, [ON_ROWS], urls=["https://given.example/p"])
    assert fetched[0] == "https://given.example/p" and any("news" in u for u in fetched[1:])
    assert seen


def test_the_given_url_rule():
    read, skipped = S.research_given_urls(["https://a.example/x", "https://a.example/x", "javascript:alert(1)",
                                          "http://" + "b" * 56 + ".onion/", 7]
                                         + [f"https://s{i}.example/" for i in range(10)])
    assert read[0] == "https://a.example/x" and len(read) == S.RESEARCH_MAX_SOURCES and read.count(read[0]) == 1
    assert any("onion" in s for s in skipped) and any("more than" in s for s in skipped)
    assert sum("not an http(s) address" in s for s in skipped) == 2


# ── N11: a page web_search could not read is told ─────────────────────────────────────────────

def test_a_failed_page_read_is_named_in_the_result(monkeypatch):
    from ghost_agent.utils.helpers import FetchError

    async def _fetch(url, **k):
        return FetchError("Error: HTTP 403 Forbidden")
    monkeypatch.setattr(S, "helper_fetch_url_content", _fetch)
    rows = [{"href": "https://a.example/py", "title": "Python 3.14 release date", "body": "x"}]
    out = asyncio.run(S.read_top_pages("python 3.14 release date", rows, None))
    assert out.startswith(S.PAGE_READS_NOT_READ_HEAD) and "https://a.example/py — the fetch failed" in out
    assert S.web_read_urls(out) == []                                             # not a page read


# ── N12: the recency limit the winner ignored is said ─────────────────────────────────────────

def test_the_engines_own_request_builder_says_whether_it_limits():
    assert S.engine_honours_timelimit("yahoo") is True
    assert S.engine_honours_timelimit("yandex") is False                          # random searchid is noise
    assert S.engine_honours_timelimit("brave_onion") is False                     # §4VA: not a ddgs engine: never claimed
    assert S.engine_honours_timelimit("no-such-engine") is False


def test_a_time_limited_wave_won_by_an_engine_that_ignores_it_says_so(monkeypatch):
    monkeypatch.setattr(S, "engine_honours_timelimit", lambda e: not e.startswith("yandex"))
    S._SEARCH_CACHE.clear()

    def by_engine(q, **kw):
        if kw["backend"] == "yandex":
            return [{"title": "Latest Python release 3.14", "body": "latest python release notes 3.14",
                     "href": "https://py.example/rel"}]
        raise Exception("blocked: HTTP 403")
    mod, _ = _ddgs_module(by_engine)
    monkeypatch.setattr(S, "_proxy_reachable", lambda *a, **k: True)
    monkeypatch.setattr(S, "_page_reads_on", lambda: False)
    monkeypatch.setattr(S.importlib.util, "find_spec", lambda name: True)
    with patch.dict("sys.modules", {"ddgs": mod}):
        out = asyncio.run(S.tool_search_ddgs("latest python release", None))
    assert "recency limit was NOT applied" in out and "(yandex)" in out
    assert S.timelimit_ignored_note(None) == ""


# ── N14: the dark-web first wave ends soon after an answer; dark blocks carry a date line ───────

@pytest.mark.asyncio
async def test_the_dark_first_wave_ends_a_grace_after_the_first_answer(monkeypatch):
    monkeypatch.setattr(D, "DARK_GRACE_AFTER_ANSWER_S", 0.2)

    async def fast():
        await asyncio.sleep(0.05)
        return [{"url": "http://x.onion/"}]

    async def slow():
        await asyncio.sleep(5)
        return [{"url": "http://y.onion/"}]

    async def failing():
        raise RuntimeError("down")
    t0 = time.monotonic()
    res = await D._first_wave_with_grace([fast(), slow(), failing()])
    assert time.monotonic() - t0 < 1.0
    assert res[0] == [{"url": "http://x.onion/"}] and isinstance(res[1], asyncio.TimeoutError)
    assert isinstance(res[2], RuntimeError)


@pytest.mark.asyncio
async def test_no_answer_means_the_wave_waits_for_everyone(monkeypatch):
    monkeypatch.setattr(D, "DARK_GRACE_AFTER_ANSWER_S", 0.05)

    async def empty():
        return []

    async def late():
        await asyncio.sleep(0.3)
        return [{"url": "http://z.onion/"}]
    res = await D._first_wave_with_grace([empty(), late()])
    assert res == [[], [{"url": "http://z.onion/"}]]


def test_a_loaded_dark_block_gets_the_page_line():
    meta = D._dark_page_meta('<html><head><title>Kosmas forum</title>'
                             '<meta property="article:published_time" content="2026-03-04T10:00:00Z">'
                             '</head><body>x</body></html>')
    block = "### SOURCE: http://k.onion/t\n[EDGE EXTRACTED FACTS]:\n- a fact\n"
    out = D.with_page_line(block, meta)
    assert out.split("\n")[1] == '[PAGE] Date: published 2026-03-04 | Title: "Kosmas forum"'
    assert S.block_body(out).startswith("[EDGE EXTRACTED FACTS]")
    failed = "### SOURCE: http://k.onion/t\nError: timed out\n"
    assert D.with_page_line(failed, meta) == failed
    assert D.with_page_line(out, meta) == out                                     # never twice


# ── N15: the sentinel and repeats leave the extract ────────────────────────────────────────────

def test_the_extract_drops_a_none_line_beside_facts_and_repeats():
    assert S.tidy_extract("- Elena founded it in 2019.\n- elena founded it in 2019\nNONE") == "- Elena founded it in 2019."
    assert S.tidy_extract("NONE\n NONE.") == "NONE"
    assert S.tidy_extract("- a\n- b") == "- a\n- b"


def test_both_research_tools_get_tidy_blocks_from_the_shared_slots():
    block = "### SOURCE: https://a.example/\n[EDGE EXTRACTED FACTS]:\n- Markos is 40.\nNONE\n- Markos is 40.\n"
    urls, blocks, *_ = S.slot_coverage([[("https://a.example/", block)]])
    assert blocks[0] == "### SOURCE: https://a.example/\n[EDGE EXTRACTED FACTS]:\n- Markos is 40.\n"
    assert S.distilled_block_has_facts(blocks[0])
    only = "### SOURCE: https://b.example/\n[EDGE EXTRACTED FACTS]:\nNONE\nNONE\n"
    assert not S.distilled_block_has_facts(S.slot_coverage([[("https://b.example/", only)]])[1][0])


@pytest.mark.asyncio
async def test_the_recheck_extract_is_tidied_too():
    class _C:
        async def chat_completion(self, payload, **kw):
            return {"choices": [{"message": {"content": "- Kosmas fact.\nNONE\n- Kosmas fact."}}]}

        def plan_distill(self, *a, **k):
            from ghost_agent.core.node_throughput import DistillPlan
            return DistillPlan(8000, 300, True, "", 220.0, 12.0, 5, 40.0)
    page = "### SOURCE: https://k.example/\n[EDGE EXTRACTED FACTS]:\nNONE\n"
    rc = await S.recheck_all_none([page], ["https://k.example/"], {"https://k.example/": ("Kosmas text " * 20, 300)},
                                  "kosmas", _C(), "m", deadline=time.monotonic() + 30, slot_wait=1.0,
                                  per_call_s=10.0, fanout=1)
    assert rc.contents[0].count("Kosmas fact.") == 1 and "NONE" not in rc.contents[0]


# ── N16: a title is one heading ───────────────────────────────────────────────────────────────

def test_a_result_title_is_its_first_heading_only():
    from lxml import html as LH
    item = LH.fromstring('<div class="relsrch"><div class="Title"><h3><a>What is the Boiling Point? - '
                         '<span>WorldAtlas</span></a></h3><h3><a>Boiling Point by Altitude</a></h3></div></div>')
    parts = item.xpath(".//div[contains(@class, 'Title')]//h3//text()")
    assert "".join(parts) == "What is the Boiling Point? - WorldAtlasBoiling Point by Altitude"   # the defect
    assert "".join(S.first_heading_parts(parts)) == "What is the Boiling Point? - WorldAtlas"
    assert S.first_heading_parts(["plain", " strings"]) == ["plain", " strings"]


def test_fact_check_names_the_page_by_its_own_title():
    block = ('### SOURCE: https://w.example/boil\n[PAGE] Date: undated | Title: "Boiling point of water"\n'
             '[EDGE EXTRACTED FACTS]:\n- Water boils at 100 C at sea level.\n')
    out = S.factcheck_sources(block, {"https://w.example/boil": "Boiling PointWorldAtlasAnother title"})
    assert "title: Boiling point of water" in out and "WorldAtlas" not in out
    assert S.page_line_title("### SOURCE: u\n[PAGE] Date: undated | Title: (none)\nx") == ""


# ── N17: the descriptions agree on site: and name no engine ────────────────────────────────────

def test_the_search_descriptions_agree_and_name_no_engine():
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    defs = {d["function"]["name"]: json.dumps(d, ensure_ascii=False) for d in TOOL_DEFINITIONS}
    for name in ("web_search", "deep_research"):
        assert "'site:host terms' does not restrict the results" in defs[name]
        assert not any(e in defs[name].lower() for e in S._RACE_ENGINES + ("mojeek", "brave"))
    assert "Torgle" not in defs["darkweb_search"] and "Torch" not in defs["darkweb_search"]


@pytest.mark.asyncio
async def test_deep_research_reads_the_site_hosts_pages_first(monkeypatch):
    rows = ON_ROWS + [{"href": "https://official.example/leak", "title": "Revolut breach leaked passports",
                       "body": "Official statement on the leaked passports breach at Revolut."}]
    _, fetched, _ = await _research(monkeypatch, [rows], query="revolut breach leaked passports site:official.example")
    assert fetched[0] == "https://official.example/leak"


# ── N18: the class comes from what the tool counted, not from its words ────────────────────────

def test_a_declared_reason_decides_the_class():
    assert TF.classify_tool_failure(TRANSPORT)[1].lower().startswith("timed")       # the defect, by prose
    assert TF.classify_tool_failure(TRANSPORT, reason_code="search_refused") == (
        TF.FailureClass.RETRYABLE, "declared search_refused")
    low = "DEEP RESEARCH READ NO PAGES — low relevance: - Nginx timeout tuning — https://x.example"
    assert TF.classify_tool_failure(low, reason_code="research_low_relevance")[0] is TF.FailureClass.UNKNOWN
    assert TF.classify_tool_failure("[SOURCES: tried 2, loaded 0, timed out 0, failed 2]",
                                    reason_code="darkweb_research_all_sources_failed")[0] is TF.FailureClass.UNKNOWN
    assert TF.classify_tool_failure(TRANSPORT, reason_code="not_declared")[0] is TF.FailureClass.RETRYABLE


def test_web_search_failures_carry_their_counted_reason(monkeypatch):
    def refused(q, wave, stats, kw):
        stats.setdefault("cats", {})["blocked"] = stats.get("cats", {}).get("blocked", 0) + 3
        return []
    out, _ = _drive(monkeypatch, refused, "electric car battery lifespan")
    assert out.status is OutcomeStatus.FAILED and out.reason_code == "search_refused"

    def unreachable(q, wave, stats, kw):
        stats.setdefault("cats", {})["conn-error"] = stats.get("cats", {}).get("conn-error", 0) + 3
        return []
    out, _ = _drive(monkeypatch, unreachable, "electric car battery lifespan")
    assert out.reason_code == "search_unreachable"

    def empty(q, wave, stats, kw):
        stats.setdefault("cats", {})["empty"] = stats.get("cats", {}).get("empty", 0) + 4
        return []
    out, _ = _drive(monkeypatch, empty, "electric car battery lifespan")
    assert out.reason_code == "search_zero_results"
    monkeypatch.setattr(S, "_proxy_reachable", lambda *a, **k: False)
    out = asyncio.run(S._tool_search_ddgs("electric car battery lifespan", PROXY))
    assert out.reason_code == "search_proxy_down"


def test_every_declared_reason_is_one_a_tool_returns():
    """The table is not a list of hopes: each code is set somewhere in the tools."""
    text = "\n".join(p.read_text() for p in (SRC / "tools").glob("*.py"))
    for code in TF.DECLARED_FAILURE_CLASS:
        assert f'"{code}"' in text, code


def test_the_loop_hands_the_declared_reason_to_the_classifier():
    """Enumeration over ALL of src/: every `classify_tool_failure` call passes a reason_code that is not a
    constant (r2: the mood reader `mood_excuses_failure` read the text alone)."""
    sites = []
    for path in SRC.rglob("*.py"):
        for n in ast.walk(ast.parse(path.read_text())):
            if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", "")) == "classify_tool_failure":
                sites.append((path.name, n))
    assert len(sites) >= 2
    for _, s in sites:
        kw = [k for k in s.keywords if k.arg == "reason_code"]
        assert kw and not isinstance(kw[0].value, ast.Constant), ast.dump(s)
    tree = ast.parse((SRC / "core" / "agent.py").read_text())
    sites = [s for f, s in sites if f == "agent.py"]
    # …and the loop's name is set from the failing call's outcome
    assert any(isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "last_error_reason" for t in n.targets)
               and "reason_code" in ast.dump(n.value) for n in ast.walk(tree))


# ── N14/N16/N18 through the real paths ─────────────────────────────────────────────────────────

def test_the_extractor_titles_a_result_by_its_first_heading():
    """Fails where the title path's every heading was glued into one title (live fact_check)."""
    from types import SimpleNamespace
    from lxml import html as LH
    eng = SimpleNamespace(
        pre_process_html=lambda h: h, extract_tree=lambda h: LH.fromstring(h),
        items_xpath="//div[@class='r']", result_type=lambda: SimpleNamespace(),
        elements_xpath={"title": ".//h3//text()", "href": ".//a/@href"})
    page = ('<html><body><div class="r"><h3><a href="https://w.example/boil">What is the Boiling Point? - '
            '<span>WorldAtlas</span></a></h3><h3><a>Boiling Point by Altitude</a></h3></div></body></html>')
    rows = S._fixed_extract_results(eng, page)
    assert rows[0].title == "What is the Boiling Point? - WorldAtlas" and rows[0].href == "https://w.example/boil"


class _DatedPage(__import__("http.server").server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):                                           # noqa: N802
        body = ('<html><head><title>Kosmas forum</title><meta property="article:published_time" '
                'content="2026-03-04T10:00:00Z"></head><body><p>'
                + "The forum reported a new ransomware leak on 12 November 2026. " * 30
                + "</p></body></html>").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):                                  # noqa: N802
        pass


@pytest.fixture
def dated_base():
    import socketserver
    import threading

    class _Srv(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True
    srv = _Srv(("127.0.0.1", 0), _DatedPage)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.mark.asyncio
async def test_a_dark_source_says_when_its_page_was_written(monkeypatch, dated_base):
    """Fails where dark blocks carried no [PAGE] line (not even "undated")."""
    from tests.onion_v3 import v3_label
    from tests.test_4un_darkres import _route_to
    from tests.test_4nj_research_yield import _Client as _DClient
    onion = f"http://{v3_label('k1')}.onion/t"
    _route_to(monkeypatch, dated_base, lambda u: "/page")
    monkeypatch.setattr(D, "_darkweb_search_raw", AsyncMock(return_value=([], [], False, 0)))
    monkeypatch.setattr(D, "_cache_get", lambda k: None)
    monkeypatch.setattr(D, "_cache_put", lambda k, v: None)
    out = await D.tool_darkweb_research(query="ransomware leak", tor_proxy="socks5://127.0.0.1:9050",
                                        llm_client=_DClient(), max_context=240_000, max_sources=1, urls=[onion])
    assert "[PAGE] Date: published 2026-03-04" in str(out)


@pytest.mark.asyncio
async def test_dark_research_counts_why_every_named_page_failed(monkeypatch):
    """N18: all named pages timed out → a retryable reason; any other failure among them → not."""
    from tests.onion_v3 import v3_label
    from tests.test_4un_darkres import _research as _dark
    from ghost_agent.utils.helpers import FetchError
    a, b = (f"http://{v3_label(s)}.onion/x" for s in ("t1", "t2"))
    _, out, _, _ = await _dark(monkeypatch, [], lambda u: D._FetchTimeout(f"Error: Fetch of {u} timed out"),
                               urls=[a, b], query=None)
    assert out.reason_code == "darkweb_research_all_sources_timed_out"
    assert TF.classify_tool_failure(str(out), reason_code=out.reason_code)[0] is TF.FailureClass.RETRYABLE
    _, out, _, _ = await _dark(monkeypatch, [], lambda u: (D._FetchTimeout(f"Error: Fetch of {u} timed out") if u == a
                                                           else FetchError(f"Error: received status 404 from {u}")),
                               urls=[a, b], query=None)
    assert out.reason_code == "darkweb_research_all_sources_failed"


# ── review r2: N18 mood reader, N2 deadline, N2 paraphrase batches ──────────────────────────────

def test_the_mood_reads_the_declared_reason_not_the_titles():
    """Fails where `mood_excuses_failure` classified the text: a low-relevance report listing a page titled
    "…timeout…" read as a transient failure and was excused."""
    from ghost_agent.core.agent import mood_excuses_failure
    low = ToolOutcome.failed(S.RESEARCH_LOW_HEAD + ":\n- Nginx timeout tuning — https://x.example/a",
                             reason_code="research_low_relevance")
    assert mood_excuses_failure(low) is False
    assert mood_excuses_failure(str(low), "research_low_relevance") is False
    refused = ToolOutcome.failed("ERROR: web search failed — 3 refusals.", reason_code="search_refused")
    assert mood_excuses_failure(refused) is True
    assert mood_excuses_failure("ERROR: web search failed — 3 refusals.", "search_refused") is True


def test_the_mood_signal_carries_the_declared_reason_of_each_call():
    """Through the real turn reader: the reconstructed call keeps its outcome's reason."""
    from tests.test_finalize_stream_pins import make_fin_agent
    a = make_fin_agent()
    low = ToolOutcome.failed(S.RESEARCH_LOW_HEAD + ":\n- Nginx timeout tuning — https://x.example/a",
                             reason_code="research_low_relevance")
    msgs = [{"role": "user", "content": "research nginx"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "deep_research", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "name": "deep_research", "content": low}]
    calls = a._reconstruct_tool_calls(msgs)
    assert calls and getattr(calls[0], "declared_reason", "") == "research_low_relevance"


@pytest.mark.asyncio
async def test_deep_research_runs_no_wave_its_bound_cannot_hold(monkeypatch):
    """r2: low batches send deep research on to more waves; a caller's bound (fact_check) caps them."""
    _, fetched, seen = await _research(monkeypatch, [LOW_ROWS], deadline_s=S._race_wave_deadline())
    assert len(seen) == 1 and fetched == []
    out, _, seen = await _research(monkeypatch, [[]], deadline_s=S._race_wave_deadline())
    # review r2: the bound, not the engines, ended it — said so (not "no engine answered")
    assert len(seen) == 1 and out.reason_code == "research_time_bound" and "time bound" in out
    assert "no engine answered" not in out
    _, _, seen = await _research(monkeypatch, [LOW_ROWS])
    assert len(seen) >= 2                                              # the control: no bound


@pytest.mark.parametrize("query,rows", [
    ("average lifespan domestic cats indoor versus outdoor comparison statistics", [
        {"title": "How Long Do Cats Live? Indoor vs. Outdoor Cat Lifespan",
         "body": "Indoor cats typically live 12-18 years, while outdoor cats often live 2-5 years due to hazards "
                 "such as traffic and disease.", "href": "https://a.example/cats"},
        {"title": "Cat life expectancy - Encyclopedia", "body": "The life expectancy of a house cat is about 15 "
                 "years. Free-roaming felines have shorter lives.", "href": "https://b.example/cat"}]),
    ("effects of intermittent fasting on insulin sensitivity in adults clinical trials 2024", [
        {"title": "Time-restricted eating improves insulin resistance: a randomized trial",
         "body": "Participants who ate within an 8-hour window had lower fasting insulin and HOMA-IR after 12 "
                 "weeks.", "href": "https://c.example/x"},
        {"title": "Intermittent fasting and type 2 diabetes", "body": "Reviews of studies show modest "
                 "improvements in glucose control.", "href": "https://d.example/y"}]),
])
def test_a_paraphrasing_batch_is_not_low(query, rows):
    """Reviewer cases: on-topic results in other words are read (the rule must not starve long queries)."""
    assert S.research_batch_is_low(query, rows) is False


# ── §4UY review r3: every reason code the internet tools return is declared ─────────────────────────────

_INTERNET_TOOL_MODULES = ("search.py", "darkweb_search.py", "browser.py", "quarantine.py")


def _returned_reason_codes():
    """AST walk: every string a `reason_code=` of an internet tool can carry (constants anywhere in the
    expression — an `x if c else y` gives both), and the code argument of web_search's `_search_failed`."""
    tools = Path(TF.__file__).resolve().parent
    found, dynamic = set(), []
    for mod in _INTERNET_TOOL_MODULES:
        tree = ast.parse((tools / mod).read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            exprs = []
            if isinstance(n, ast.keyword) and n.arg == "reason_code":
                exprs.append(n.value)
            if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_search_failed" and len(n.args) > 1:
                exprs.append(n.args[1])
            for e in exprs:
                if any(isinstance(x, ast.JoinedStr) for x in ast.walk(e)):
                    dynamic.append((mod, e.lineno))
                tested = {id(c) for x in ast.walk(e) if isinstance(x, ast.Compare) for c in ast.walk(x)}
                found |= {x.value for x in ast.walk(e)
                          if isinstance(x, ast.Constant) and isinstance(x.value, str) and id(x) not in tested}
    return found, dynamic


def test_every_reason_code_an_internet_tool_returns_is_declared():
    """Reverse direction of `test_every_declared_reason_is_one_a_tool_returns`. Fails where a code is
    returned that the table does not know (its class came from the message's words: a page title with
    "timeout" in it read as one) or is built at run time (no table can hold it)."""
    found, dynamic = _returned_reason_codes()
    assert dynamic == [], dynamic
    missing = sorted(c for c in found if c and c not in TF.DECLARED_FAILURE_CLASS)
    assert missing == [], missing
    assert len(found) >= 35


@pytest.mark.parametrize("code", ["research_time_bound", "factcheck_research_timeout", "factcheck_verify_timeout"])
def test_a_bound_cut_says_answer_from_what_you_have(code):
    """A caller's time bound cut the work: not a transient fault to retry."""
    cls, why = TF.classify_tool_failure("ERROR: timed out", reason_code=code)
    assert cls is not TF.FailureClass.RETRYABLE and why == f"declared {code}"


@pytest.mark.asyncio
async def test_fact_check_passes_the_researchs_own_reason_through():
    """Fails where the wrapper replaced every inner reason with `factcheck_research_failed` (RETRYABLE): a
    bound cut inside deep_research became "retry it"."""
    from tests.test_4tb_fact_check import _llm
    inner = ToolOutcome.failed(S.RESEARCH_BOUND_CUT_TEXT, world_changed=False, reason_code="research_time_bound")
    out = await S.tool_fact_check(query="Elena's bakery opened in 2019", llm_client=_llm(),
                                  deep_research_callable=AsyncMock(return_value=inner))
    assert out.status is OutcomeStatus.FAILED and out.reason_code == "research_time_bound"
    assert "time bound" in str(out) and "transient" not in str(out).split("\n")[0]
    inner2 = ToolOutcome.failed("no page could be read", world_changed=False)
    out2 = await S.tool_fact_check(query="Elena's bakery opened in 2019", llm_client=_llm(),
                                   deep_research_callable=AsyncMock(return_value=inner2))
    assert out2.reason_code == "factcheck_research_failed"
    assert TF.classify_tool_failure(str(out2), reason_code=out2.reason_code)[0] is TF.FailureClass.RETRYABLE


@pytest.mark.parametrize("req,suppressed", [
    ("Find the price, no sources please", True), ("Answer without sources", True),
    ("Don't include the sources", True), ("do not add any links", True), ("Summarise it, no sources.", True),
    ("Don’t include the sources", True), ("Please don’t add links", True),
    ("Research why there are no sources of lithium in Greece", False),
    ("Are there places without sources of clean water?", False), ("no sources for this claim exist?", False)])
def test_r3_no_sources_please_suppresses_the_list(req, suppressed):
    note = LG.sources_note(ELENA_REPLY, [WS_ROW], req)
    assert (note == "") is suppressed, (req, note)


# ── 2026-11-19 (live functional run): a URL's own closing parenthesis stays ─────────────────────────────────
@pytest.mark.parametrize("raw,want", [
    ("https://en.wikipedia.org/wiki/Python_(programming_language)", "https://en.wikipedia.org/wiki/Python_(programming_language)"),
    ("https://en.wikipedia.org/wiki/Python_(programming_language).", "https://en.wikipedia.org/wiki/Python_(programming_language)"),
    ("https://example.org/a)", "https://example.org/a"),
    ("https://example.org/a).", "https://example.org/a"),
    ("https://example.org/a,", "https://example.org/a"),
    ("https://example.org/f(x)/g(y))", "https://example.org/f(x)/g(y)"),
    ("https://example.org/a", "https://example.org/a"), ("", "")])
def test_a_result_url_keeps_its_own_parenthesis(raw, want):
    from ghost_agent.core.link_grounding import trim_result_url
    assert trim_result_url(raw) == want


def test_the_sources_list_shows_a_parenthesised_link_whole():
    from ghost_agent.core.link_grounding import fetched_urls
    url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    assert fetched_urls([{"name": "deep_research", "content": f"### SOURCE: {url}\nPython was released in 1991.\n"}]) == [url]
    assert fetched_urls([{"name": "browser", "content": f"--- BROWSER RESULT ---\nSTATUS: OK\nURL: {url}\ntext"}]) == [url]
