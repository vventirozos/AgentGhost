"""§4LH (2026-10-04): the §4LG open and deferred items. Each test names the
world it fails in."""
import asyncio
import datetime
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
from tests.test_critic_async import agent, _final, _make_verifier, _verdict  # noqa: F401 — fixture

_V3 = "duckduckgogg42xjoc72x3sjasowoarfbgcmvfimaftt6twagswzczad.onion"
_V3_OTHER = "zqktlwiuavvvqqt4ybvgvi7tyo4hjl5xgfuvpdf6otjiycgwqbym2qad.onion"
_V2 = "3g2upl4pq6kufc4m.onion"


# ── the verifier's onion check ───────────────────────────────────────────────
def _confirmed():
    return VerifyResult(verdict=VerifyVerdict.CONFIRMED, confidence=0.95, reasoning="judge",
                        issues=[], unverified_facts=["2014"])


@pytest.mark.parametrize("claim,hay,demoted", [
    (f"DuckDuckGo's official v3 address is {_V2}.", _V2, True),        # retired, shown as current
    (f"The old {_V2} was retired in 2021; the current one is {_V3}.", _V3, False),
    (f"The official address is {_V3}.", "nothing about it", True),     # from memory
    (f"The official address is {_V3}.", f"loaded http://{_V3}/ title DuckDuckGo", False),
])
def test_an_onion_address_is_not_confirmed_from_memory(claim, hay, demoted):
    """Fails where the judge CONFIRMED the retired v2 address as DuckDuckGo's
    "official v3" (live probe D2, second run)."""
    from ghost_agent.core.verifier import Verifier
    out = Verifier._guard_onion_claims(None, _confirmed(), claim, hay)
    assert (out.verdict == VerifyVerdict.UNCERTAIN) is demoted
    if demoted:
        assert out.confidence <= 0.5 and out.issues and out.unverified_facts == ["2014"]


def test_the_onion_check_runs_on_every_verdict(monkeypatch):
    from ghost_agent.core.verifier import Verifier
    import ghost_agent.core.verifier as V
    v = Verifier.__new__(Verifier)
    monkeypatch.setattr(Verifier, "_verify_claim_incumbent", AsyncMock(return_value=_confirmed()))
    monkeypatch.setattr(V, "_claim_binding_primary_enabled", lambda: False)
    monkeypatch.setattr(V, "_claim_binding_refute_first_enabled", lambda: False)
    out = asyncio.run(v.verify_claim(f"The official address is {_V2}.", f"search: {_V2}", "find ddg onion"))
    assert out.verdict == VerifyVerdict.UNCERTAIN
    monkeypatch.setenv("GHOST_VERIFY_ONION_GUARD", "0")
    out = asyncio.run(v.verify_claim(f"The official address is {_V2}.", f"search: {_V2}", "find ddg onion"))
    assert out.verdict == VerifyVerdict.CONFIRMED


# ── links the turn never saw ────────────────────────────────────────────────
def test_an_invented_deep_link_is_removed_and_a_seen_one_kept():
    """Fails where facebook.com/GKTeamBJJ/ — in no result — reached the user."""
    from ghost_agent.core.link_grounding import ground_links, haystack_from
    hay = haystack_from([{"role": "user", "content": "find the gym"},
                         {"role": "assistant", "content": "see https://facebook.com/GKTeamBJJ/",
                          "tool_calls": [{"function": {"arguments": '{"url": "https://www.gkteam.gr/about"}'}}]},
                         {"role": "tool", "content": "result: https://www.bbc.co.uk/news/articles/c1"}])
    reply = ("Sources: [BBC](https://www.bbc.co.uk/news/articles/c1), https://facebook.com/GKTeamBJJ/ "
             "and [their page](https://gkteam.gr/about). See also python.org and "
             "[made up](https://example.com/a/b).")
    out, removed = ground_links(reply, hay)
    assert "https://www.bbc.co.uk/news/articles/c1" in out          # in a result
    assert "https://gkteam.gr/about" in out                         # the agent navigated to it
    assert "GKTeamBJJ" not in out and "facebook.com" in out         # bare: the domain stays
    assert "example.com/a/b" not in out and "made up" in out        # markdown: the label stays
    assert "python.org" in out                                      # a bare domain is never touched
    assert len(removed) == 2 and "_Removed 2 links" in out


def test_an_onion_address_in_no_result_is_removed():
    from ghost_agent.core.link_grounding import ground_links, haystack_from
    hay = haystack_from([{"role": "tool", "content": f"opened http://{_V3}/ — DuckDuckGo"}])
    out, removed = ground_links(f"Official: `{_V3}`. Another: {_V3_OTHER} and http://{_V3_OTHER}/x", hay)
    assert _V3 in out and _V3_OTHER not in out and len(removed) == 2


@pytest.mark.asyncio
async def test_finalize_grounds_links_on_a_research_turn_only(agent):
    """Fails where the check is not wired into the delivered reply."""
    agent.available_tools["web_search"] = AsyncMock(return_value="1. Python 3.13 — https://python.org/downloads/release/3136")
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=[
        {"choices": [{"message": {"content": "", "tool_calls": [
            {"id": "t1", "function": {"name": "web_search", "arguments": '{"query": "python"}'}}]}}]},
        _final("See https://python.org/downloads/release/3136 and https://python.org/invented/page."),
    ])
    with patch("ghost_agent.core.agent.pretty_log"):
        out, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "latest python?"}]},
                                            background_tasks=MagicMock())
    assert "release/3136" in out and "invented/page" not in out
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=[
        _final("Try https://python.org/invented/page.")])
    with patch("ghost_agent.core.agent.pretty_log"):
        out, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "a python link?"}]},
                                            background_tasks=MagicMock())
    assert "invented/page" in out                                   # no research tool: untouched
    agent.available_tools["execute"] = AsyncMock(return_value="OUTPUT: ok")
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=[
        {"choices": [{"message": {"content": "", "tool_calls": [
            {"id": "t1", "function": {"name": "execute", "arguments": '{"content": "print(1)"}'}}]}}]},
        _final("Ran it. Docs: https://python.org/invented/page.")])
    with patch("ghost_agent.core.agent.pretty_log"):
        out, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "run it"}]},
                                            background_tasks=MagicMock())
    assert "invented/page" in out                                   # a non-research tool: untouched


# ── text written alongside a tool call ──────────────────────────────────────
@pytest.mark.asyncio
async def test_the_reply_does_not_open_with_the_working_narration(agent, monkeypatch):
    """Fails where the D2 reply opened "The known DuckDuckGo onion address
    didn't respond. Let me find the official one…"."""
    agent.available_tools["web_search"] = AsyncMock(return_value=f"1. DuckDuckGo onion — {_V3}")
    narration = "The known address didn't respond. Let me find the official one."

    def script():
        return [{"choices": [{"message": {"content": narration, "tool_calls": [
                    {"id": "t1", "function": {"name": "web_search", "arguments": '{"query": "ddg onion"}'}}]}}]},
                _final(f"DuckDuckGo's onion address is {_V3}. It is listed on DuckDuckGo's own help pages and loaded over Tor with the title 'DuckDuckGo - Protection. Privacy. Peace of mind.', which matches the clearnet site, so this is the official service.")]
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=script())
    with patch("ghost_agent.core.agent.pretty_log"):
        out, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "ddg onion?"}]},
                                            background_tasks=MagicMock())
    assert out.startswith("DuckDuckGo's onion address") and narration not in out
    monkeypatch.setenv("GHOST_DROP_PRE_TOOL_TEXT", "0")
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=script())
    with patch("ghost_agent.core.agent.pretty_log"):
        out, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "ddg onion again?"}]},
                                            background_tasks=MagicMock())
    assert narration in out


def test_long_or_only_pre_tool_text_is_kept():
    from ghost_agent.core.reply_smoothing import drop_pre_tool_segments, PRE_TOOL_SEGMENT_MAX
    long_seg = "Partial answer: " + "x" * PRE_TOOL_SEGMENT_MAX
    assert drop_pre_tool_segments(long_seg + "\n\nMore.", [long_seg]) == long_seg + "\n\nMore."
    assert drop_pre_tool_segments("Checking.", ["Checking."]) == "Checking."   # never empties


# ── the search hedge ─────────────────────────────────────────────────────────
class _FakeDDGS:
    """Engines on the first circuits hang empty; a hedge circuit answers."""
    def __init__(self, **kw):
        self.proxy = str(kw.get("proxy") or "")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def text(self, q, **kw):
        if "yandh" in self.proxy:
            return [{"title": "Python 3.13", "body": "release", "href": "https://python.org/x"}]
        time.sleep(1.2)
        return []


@pytest.mark.asyncio
@pytest.mark.parametrize("hedge_s,wins", [("0.3", True), ("0", False)])
async def test_a_wave_nobody_wins_hedges_on_fresh_circuits(monkeypatch, hedge_s, wins):
    """Fails where a wave with no early winner waited the full deadline (13%
    of first waves, median 12 s) before wave 2."""
    from ghost_agent.tools.search import _race_search_wave
    mod = MagicMock()
    mod.DDGS = _FakeDDGS
    monkeypatch.setenv("GHOST_SEARCH_HEDGE_S", hedge_s)
    t0 = time.monotonic()
    with patch.dict("sys.modules", {"ddgs": mod}):
        res = await _race_search_wave("python release", "socks5h://127.0.0.1:9050", 0)
    took = time.monotonic() - t0
    assert bool(res) is wins
    if wins:
        assert took < 1.1                                   # did not wait for the hung engines


# ── the cache key: trivial variation only ──────────────────────────────────
@pytest.mark.parametrize("a,b,same", [
    ("python asyncio tutorial", "Python  asyncio tutorial?", True),
    ("postgres 18 release notes", "release notes postgres 18", False),   # §4LI: order is kept
    ("israel attacks iran", "iran attacks israel", False),
    ("python 3.12", "python 3.13", False),
])
def test_only_trivial_variation_shares_a_cache_entry(a, b, same):
    """§4LI: the §4LH word-set key added 11 hits on 1,555 recorded queries
    (0.7%) and merged subject/object order ("israel attacks iran") — reverted
    to the exact normalised key."""
    from ghost_agent.tools.search import _norm_cache_key
    assert (_norm_cache_key(a) == _norm_cache_key(b)) is same


# ── torch is a fallback ──────────────────────────────────────────────────────
def _engine_results(n_primary):
    def fake(engine, query, tor_proxy, exclude):
        async def run():
            calls.append(engine["name"])
            if engine["name"] == "torch":
                return [{"url": f"http://{'t' * 55}{i}.onion/", "title": "t"} for i in range(2)]
            return [{"url": f"http://{engine['name'][:1] * 55}{i}.onion/", "title": "p"} for i in range(n_primary)]
        return run()
    calls = []
    return fake, calls


@pytest.mark.parametrize("n_primary,env,torch_asked", [(3, "1", False), (1, "1", True), (3, "0", True)])
def test_torch_is_asked_only_when_the_others_come_back_thin(monkeypatch, n_primary, env, torch_asked):
    """Fails where torch — useful in 22% of searches, 34 timeouts at 38 s —
    ran on every search."""
    import ghost_agent.tools.darkweb_search as D
    fake, calls = _engine_results(n_primary)
    monkeypatch.setattr(D, "_query_engine", fake)
    monkeypatch.setenv("GHOST_ONION_FALLBACK_TIER", env)
    monkeypatch.delenv("GHOST_ONION_ENGINES", raising=False)
    ranked, skipped, all_skipped, n = asyncio.run(D._darkweb_search_raw("q", "socks5h://x:1"))
    assert ("torch" in calls) is torch_asked
    assert n == len(calls)                                  # "asked N engines" counts what was asked


def test_an_engine_override_can_mark_a_fallback(monkeypatch):
    import ghost_agent.tools.darkweb_search as D
    monkeypatch.setenv("GHOST_ONION_ENGINES", json.dumps([
        {"name": "a", "url": "http://a/?q={q}"}, {"name": "b", "url": "http://b/?q={q}", "tier": "fallback"}]))
    assert [e.get("tier") for e in D._load_engines()] == [None, "fallback"]


# ── full tool results beside the trajectory ──────────────────────────────────
def test_a_long_tool_result_is_kept_in_full_beside_the_row(tmp_path, monkeypatch):
    """Fails where every row cut tool results at 4,000 chars (all 53 browser
    results in the §4LG audit) and onion addresses were redacted, so a reply's
    facts could not be checked against the page it read."""
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.distill.collector import TrajectoryCollector
    from ghost_agent.distill.schema import Trajectory
    page = f"Page at http://{_V3}/ " + "word " * 3000 + "THE-END"
    calls = GhostAgent._reconstruct_tool_calls([
        {"role": "assistant", "tool_calls": [{"id": "t1", "function": {"name": "browser", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "name": "browser", "content": page}])
    col = TrajectoryCollector(tmp_path / "trajectories", session_id="s1")
    traj = Trajectory(tool_calls=calls, user_request="read it")
    row_path = col.append(traj)
    assert col.append_full_results(traj) == 1
    row = json.loads(row_path.read_text().splitlines()[0])
    assert "full_result" not in row["tool_calls"][0]
    assert len(row["tool_calls"][0]["result"]) <= 4000 and "<REDACTED_ONION>" in row["tool_calls"][0]["result"]
    side = col.results_path(datetime.datetime.utcnow())
    assert side.parent.parent.name == "trajectory_results"         # NOT under the row root
    rec = json.loads(side.read_text().splitlines()[0])
    assert rec["trajectory_id"] == traj.id and rec["result"].endswith("THE-END") and _V3 in rec["result"]
    assert not list((tmp_path / "trajectories").rglob("*results*"))
    monkeypatch.setenv("GHOST_TRAJ_FULL_RESULTS", "0")
    assert col.append_full_results(traj) == 0


@pytest.mark.asyncio
async def test_the_recorder_writes_the_sidecar(agent, tmp_path):
    from ghost_agent.distill.collector import TrajectoryCollector
    agent.context.trajectory_collector = TrajectoryCollector(tmp_path / "trajectories", session_id="s2")
    agent.available_tools["web_search"] = AsyncMock(return_value="result " * 1200)
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=[
        {"choices": [{"message": {"content": "", "tool_calls": [
            {"id": "t1", "function": {"name": "web_search", "arguments": '{"query": "x"}'}}]}}]},
        _final("Here is the answer.")])
    with patch("ghost_agent.core.agent.pretty_log"):
        await agent.handle_chat({"messages": [{"role": "user", "content": "search x"}]},
                                background_tasks=MagicMock())
    assert list((tmp_path / "trajectory_results").rglob("session-s2.jsonl"))


# ── a member's web research is verified ─────────────────────────────────────
@pytest.mark.parametrize("names,verified", [
    (["web_search"], True), (["web_search", "darkweb_search"], True),
    (["image_generation"], False), (["web_search", "vision_analysis"], False), ([], False)])
def test_which_member_turns_are_research(names, verified, monkeypatch):
    from ghost_agent.core.agent import _member_research_turn
    assert _member_research_turn([{"name": n, "content": "x"} for n in names]) is verified
    monkeypatch.setenv("GHOST_VERIFY_MEMBER_RESEARCH", "0")
    assert _member_research_turn([{"name": n, "content": "x"} for n in names]) is False


@pytest.mark.asyncio
async def test_a_members_web_research_is_verified_after_the_reply(agent, monkeypatch):
    """Fails where 0 of 25 member research turns were verified (by design,
    R8) — a web-only answer reads nothing of the owner's."""
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    verifier, vmock = _make_verifier([_verdict(VerifyVerdict.REFUTED, issues=["the year is 2017"])])
    agent.context.verifier = verifier
    agent.context.skill_memory = MagicMock()
    agent._active_constraint_note = MagicMock(return_value="OWNER PROJECT: secret plan || USER REQUEST: ")
    agent.available_tools["web_search"] = AsyncMock(return_value="It launched in 2017. https://x.org/a")
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=[
        {"choices": [{"message": {"content": "", "tool_calls": [
            {"id": "t1", "function": {"name": "web_search", "arguments": '{"query": "launch year"}'}}]}}]},
        _final("It launched in 2016.")])
    with patch("ghost_agent.core.agent.pretty_log"):
        out, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "when did it launch?"}]},
                                            background_tasks=MagicMock(), requester_role="member")
    assert "2016" in out
    for _ in range(100):
        await asyncio.sleep(0.01)
        if vmock.await_count:
            break
    assert vmock.await_count == 1
    ctx_arg = " ".join(str(a) for a in vmock.await_args.args) + " ".join(
        str(v) for v in vmock.await_args.kwargs.values())
    assert "secret plan" not in ctx_arg                     # the owner's project stays out


# ── review fixes (§4LH round 2) ──────────────────────────────────────────────
@pytest.mark.asyncio
async def test_a_members_verdict_never_looks_at_the_owners_images(agent, monkeypatch):
    """Fails where the visual arm, never member-gated, resolved an image name
    from a web result into the OWNER's sandbox on a member's research turn."""
    import ghost_agent.core.agent as A
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    verifier, vmock = _make_verifier([_verdict(VerifyVerdict.CONFIRMED)])
    agent.context.verifier = verifier
    agent.context.skill_memory = MagicMock()
    spy = MagicMock(return_value=(None, None))
    monkeypatch.setattr(A, "_select_visual_evidence", spy)
    agent.available_tools["web_search"] = AsyncMock(return_value="logo: https://cdn.x/owner_private_logo.png")
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=[
        {"choices": [{"message": {"content": "", "tool_calls": [
            {"id": "t1", "function": {"name": "web_search", "arguments": '{"query": "ddg logo"}'}}]}}]},
        _final("The logo appears blue.")])
    with patch("ghost_agent.core.agent.pretty_log"):
        await agent.handle_chat({"messages": [{"role": "user", "content": "what colour does the logo appear?"}]},
                                background_tasks=MagicMock(), requester_role="member")
    for _ in range(100):
        await asyncio.sleep(0.01)
        if vmock.await_count:
            break
    assert vmock.await_count == 1 and not spy.called


def test_an_answer_written_beside_a_bookkeeping_call_is_kept():
    """Fails where a full answer written alongside a `remember` call was
    dropped and only the final "I've saved a note" was delivered."""
    from ghost_agent.core.agent import _all_lookup_calls
    assert _all_lookup_calls([{"function": {"name": "web_search"}}, {"function": {"name": "browser"}}])
    assert not _all_lookup_calls([{"function": {"name": "web_search"}}, {"function": {"name": "remember"}}])
    assert not _all_lookup_calls([])


def test_only_a_whole_paragraph_is_dropped():
    from ghost_agent.core.reply_smoothing import drop_pre_tool_segments
    body = "Here is the table. Done. " + "Row data follows in the attached file for each region. " * 5
    assert drop_pre_tool_segments(body + "\n\nDone.", ["Done."]) == body.strip()


@pytest.mark.asyncio
async def test_an_answer_written_beside_a_confirming_search_is_kept(agent):
    """§4LI: fails where "The capital of Australia is Canberra." written
    beside a search was dropped and only "The search confirms it." shipped."""
    agent.available_tools["web_search"] = AsyncMock(return_value="Canberra is the capital of Australia.")
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=[
        {"choices": [{"message": {"content": "The capital of Australia is Canberra.", "tool_calls": [
            {"id": "t1", "function": {"name": "web_search", "arguments": '{"query": "capital of australia"}'}}]}}]},
        _final("The search confirms it.")])
    with patch("ghost_agent.core.agent.pretty_log"):
        out, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "capital of australia?"}]},
                                            background_tasks=MagicMock())
    assert "Canberra" in out


@pytest.mark.parametrize("reply,hay_extra,kept", [
    ("See https://el.wikipedia.org/wiki/Τορ", "https://el.wikipedia.org/wiki/%CE%A4%CE%BF%CF%81", True),
    ("Read https://x.org/guide#install", "https://x.org/guide", True),
    ("Install from https://x.org/guide…", "https://x.org/guide", True),
    ("Run `curl https://x.org/never/seen`.", "", True),             # code is left alone
    ("Old: 3g2upl4pq6kufc4m.onion (v2, retired)", "", True),        # v2 is the verifier's job
])
def test_a_link_the_turn_did_see_is_not_removed(reply, hay_extra, kept):
    from ghost_agent.core.link_grounding import ground_links, haystack_from
    hay = haystack_from([{"role": "tool", "content": "results: " + hay_extra}])
    out, removed = ground_links(reply, hay)
    assert (not removed) is kept, (out, removed)


def test_a_link_from_an_earlier_answer_is_seen_but_the_reply_cannot_vouch_for_itself():
    from ghost_agent.core.link_grounding import ground_links, haystack_from
    msgs = [{"role": "user", "content": "q1"},
            {"role": "assistant", "content": "Earlier: https://prev.example.org/article/42"},
            {"role": "user", "content": "more?"},
            {"role": "assistant", "content": "Also https://new.example.org/made/up"}]
    out, removed = ground_links("As I said, https://prev.example.org/article/42 and https://new.example.org/made/up",
                                haystack_from(msgs))
    assert "prev.example.org/article/42" in out and removed == ["https://new.example.org/made/up"]


def test_an_unseen_image_keeps_its_alt_text_and_a_garbled_onion_goes():
    from ghost_agent.core.link_grounding import ground_links
    out, removed = ground_links("![logo](https://x.org/a/b.png) and duckduckgogg42xjoc72x3sj3owoq.onion", "")
    assert out.startswith("logo and [onion address removed") and "!" not in out.split(" and ")[0]
    assert len(removed) == 2


@pytest.mark.parametrize("claim,issue", [
    (f"Η παλιά διεύθυνση {_V2} έχει καταργηθεί.", False),   # Greek: the v2 check abstains
    ("The address is duckduckgogg42xjoc72x3sj3owoqfbhsf7tt3v5y4qt1.onion.", True),   # impossible length
])
def test_onion_issues_abstain_across_scripts_and_catch_garbled_addresses(claim, issue):
    from ghost_agent.core.link_grounding import onion_claim_issues
    # the haystack holds the address: a garbled one is an issue even when a page printed it
    out = onion_claim_issues(claim, claim)
    assert bool(out) is issue
    if issue:
        assert "not a valid onion address" in out[0]


def test_the_removed_links_line_is_not_judged_as_the_models_words():
    from ghost_agent.core.reply_smoothing import strip_system_notes
    t = "Answer.\n\n_Removed 2 links that appear in no page or search result I read this turn._"
    assert strip_system_notes(t) == "Answer."


@pytest.mark.asyncio
async def test_a_hedge_ends_with_the_first_engines(monkeypatch):
    """Fails where a hedge engine kept its full timeout from 6 s, so a wave
    nobody won ran 16 s instead of 12."""
    import ghost_agent.tools.search as S

    class Hang:
        def __init__(self, **kw):
            self.t = float(kw.get("timeout") or 1)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def text(self, q, **kw):
            time.sleep(self.t)
            return []
    mod = MagicMock()
    mod.DDGS = Hang
    monkeypatch.setattr(S, "_engine_timeout", lambda e: 2.0)
    monkeypatch.setenv("GHOST_SEARCH_HEDGE_S", "0.6")
    t0 = time.monotonic()
    with patch.dict("sys.modules", {"ddgs": mod}):
        assert await S._race_search_wave("q words", "socks5h://127.0.0.1:9050", 0) == []
    assert time.monotonic() - t0 < 2.4


@pytest.mark.parametrize("a,b", [("flights athens to london", "flights london to athens"),
                                 ("convert usd to eur", "convert eur to usd"),
                                 (".net framework", "net framework")])
def test_order_carrying_queries_do_not_share_a_cache_entry(a, b):
    from ghost_agent.tools.search import _norm_cache_key
    assert _norm_cache_key(a) != _norm_cache_key(b)


def test_the_torch_fallback_has_a_short_deadline(monkeypatch):
    """Fails where the fallback ran a second full 38 s deadline after the
    first wave."""
    import ghost_agent.tools.darkweb_search as D

    def fake(engine, query, tor_proxy, exclude):
        async def run():
            if engine["name"] == "torch":
                await asyncio.sleep(3)
            return []
        return run()
    monkeypatch.setattr(D, "_query_engine", fake)
    monkeypatch.setattr(D, "_FALLBACK_DEADLINE_S", 0.2)
    monkeypatch.delenv("GHOST_ONION_ENGINES", raising=False)
    t0 = time.monotonic()
    asyncio.run(D._darkweb_search_raw("q", "socks5h://x:1"))
    assert time.monotonic() - t0 < 1.5


def test_the_sidecar_lands_in_the_rows_day_and_frees_the_text(tmp_path):
    from ghost_agent.distill.collector import TrajectoryCollector
    from ghost_agent.distill.schema import ToolCall, Trajectory
    col = TrajectoryCollector(tmp_path / "trajectories", session_id="s3")
    tc = ToolCall(name="browser", result="x" * 4000, full_result="x" * 9000)
    traj = Trajectory(tool_calls=[tc])
    row = tmp_path / "trajectories" / "2026-10-03" / "session-s3.jsonl"
    assert col.append_full_results(traj, row) == 1
    assert (tmp_path / "trajectory_results" / "2026-10-03" / "session-s3.jsonl").exists()
    assert tc.full_result == ""


def test_a_members_correction_surfaces_only_beside_the_answer_it_corrects(agent):
    """Fails where two members' threads that open with the same words shared
    one conversation tag, so A's correction could open B's reply."""
    import time as _t
    from ghost_agent.core.agent import _reply_tag
    from ghost_agent.utils.logging import requester_role_context
    tok = requester_role_context.set("member")
    try:
        first = [{"role": "user", "content": "hi"}]
        fp = agent._conversation_fingerprint(first)
        agent._pending_corrections = [{"note": "the year is 2017", "ts": _t.monotonic(), "traj": "t",
                                       "conv": f"{fp}|r{_reply_tag('It launched in 2016, per the site.')}"}]
        other = first + [{"role": "assistant", "content": "Hello! How can I help?"}, {"role": "user", "content": "x"}]
        agent._consume_pending_corrections(other, conv_fp=fp)
        assert agent._take_active_correction() == "" and len(agent._pending_corrections) == 1
        mine = first + [{"role": "assistant", "content": "It launched in *2016*, per the site."},
                        {"role": "user", "content": "sure?"}]
        agent._consume_pending_corrections(mine, conv_fp=fp)
        assert "the year is 2017" in agent._take_active_correction()
    finally:
        requester_role_context.reset(tok)


@pytest.mark.asyncio
async def test_the_in_loop_verdict_is_reused_after_the_narration_drop(agent):
    """Fails where the judged text kept the narration finalize then dropped:
    the in-loop verdict was discarded and recomputed (~25 s)."""
    agent.available_tools["web_search"] = AsyncMock(return_value="The answer is 42 (source: x.org).")
    verifier, vmock = _make_verifier([_verdict(VerifyVerdict.CONFIRMED), _verdict(VerifyVerdict.CONFIRMED)])
    agent.context.verifier = verifier
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=[
        {"choices": [{"message": {"content": "Let me search for it.", "tool_calls": [
            {"id": "t1", "function": {"name": "web_search", "arguments": '{"query": "the answer"}'}}]}}]},
        _final("The answer is 42." + " The source states it plainly in its summary, and the figure matches"
               " the value reported in the original article, so there is no disagreement between the two." * 2)])
    with patch("ghost_agent.core.agent.pretty_log"):
        out, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "what is the answer?"}]},
                                            background_tasks=MagicMock())
    assert out.startswith("The answer is 42") and vmock.await_count == 1


@pytest.mark.asyncio
async def test_a_member_never_waits_on_a_verdict_in_sync_mode(agent, monkeypatch):
    """Fails where a member's research turn, now verifiable, blocked the
    reply on the verdict in sync mode (inline in the loop or at the gate)."""
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "0")
    monkeypatch.setenv("GHOST_CRITIC_GATE_TIMEOUT", "30")

    async def slow(*a, **k):
        await asyncio.sleep(1.5)
        return _verdict(VerifyVerdict.CONFIRMED)
    verifier, vmock = _make_verifier([])
    vmock.side_effect = slow
    agent.context.verifier = verifier
    agent.context.skill_memory = MagicMock()
    agent.available_tools["web_search"] = AsyncMock(return_value="It launched in 2017.")
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=[
        {"choices": [{"message": {"content": "", "tool_calls": [
            {"id": "t1", "function": {"name": "web_search", "arguments": '{"query": "launch"}'}}]}}]},
        _final("It launched in 2017.")])
    t0 = time.monotonic()
    with patch("ghost_agent.core.agent.pretty_log"):
        await agent.handle_chat({"messages": [{"role": "user", "content": "when did it launch?"}]},
                                background_tasks=MagicMock(), requester_role="member")
    assert time.monotonic() - t0 < 1.2


@pytest.mark.asyncio
async def test_a_members_late_correction_is_bound_to_its_answer(agent, monkeypatch):
    """Fails where the late handler queued a member's correction under the
    bare first-message tag, which another member's thread can share."""
    from ghost_agent.core.agent import _reply_tag
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    verifier, vmock = _make_verifier([_verdict(VerifyVerdict.REFUTED, conf=0.97, issues=["the year is 2017"])])
    agent.context.verifier = verifier
    agent.context.skill_memory = MagicMock()
    agent.available_tools["web_search"] = AsyncMock(return_value="It launched in 2017.")
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=[
        {"choices": [{"message": {"content": "", "tool_calls": [
            {"id": "t1", "function": {"name": "web_search", "arguments": '{"query": "launch"}'}}]}}]},
        _final("It launched in 2016.")])
    with patch("ghost_agent.core.agent.pretty_log"):
        out, _, _ = await agent.handle_chat({"messages": [{"role": "user", "content": "when did it launch?"}]},
                                            background_tasks=MagicMock(), requester_role="member")
    for _ in range(200):
        await asyncio.sleep(0.01)
        if agent._pending_corrections:
            break
    convs = [c.get("conv", "") for c in agent._pending_corrections if isinstance(c, dict)]
    assert convs and all(c.endswith("|r" + _reply_tag(out)) for c in convs)


@pytest.mark.asyncio
async def test_a_member_never_waits_at_the_gate_after_a_failed_search(agent, monkeypatch):
    """The in-loop branch is skipped when a tool failed; the post-loop gate
    then decides, and for a member it must not wait (sync mode, budget 30 s)."""
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "0")
    monkeypatch.setenv("GHOST_CRITIC_GATE_TIMEOUT", "30")

    async def slow(*a, **k):
        await asyncio.sleep(1.5)
        return _verdict(VerifyVerdict.CONFIRMED)
    verifier, vmock = _make_verifier([])
    vmock.side_effect = slow
    agent.context.verifier = verifier
    agent.context.skill_memory = MagicMock()
    agent.available_tools["web_search"] = AsyncMock(
        return_value="Error: search failed — ZERO results from every engine over Tor.")
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=[
        {"choices": [{"message": {"content": "", "tool_calls": [
            {"id": "t1", "function": {"name": "web_search", "arguments": '{"query": "launch"}'}}]}}]},
        _final("The search failed, so I could not find the launch year.")])
    t0 = time.monotonic()
    with patch("ghost_agent.core.agent.pretty_log"):
        await agent.handle_chat({"messages": [{"role": "user", "content": "when did it launch?"}]},
                                background_tasks=MagicMock(), requester_role="member")
    assert time.monotonic() - t0 < 1.2


@pytest.mark.parametrize("text", ["\n" * 100000, ". " * 50000, "[" * 50000, "`" * 33000, "http://" * 20000,
                                  "a" * 60 + ".onion " * 10000, "![" * 30000, "_Removed 1 link " * 20000])
def test_the_link_checks_are_linear(text):
    """Fails where a pattern backtracks: the first REMOVED_NOTE_RE (a leading
    unbounded newline run) took 3.2 s on 100k newlines in strip_system_notes."""
    from ghost_agent.core.link_grounding import ground_links, onion_claim_issues
    from ghost_agent.core.reply_smoothing import strip_system_notes
    t0 = time.perf_counter()
    ground_links(text, "")
    onion_claim_issues(text, "")
    strip_system_notes(text)
    assert time.perf_counter() - t0 < 1.0
