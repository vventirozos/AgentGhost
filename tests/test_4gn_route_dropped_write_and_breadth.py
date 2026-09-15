"""§4GN (2026-09-14): three defects the cf45e352 RE-RUN exposed.

Request da4c17ba re-ran the September-2026 Revolut OSINT brief on the fixed
tree. It came back `verified · 0.74` (was `failed · 0.15`), refused to
manufacture an attribution, and never crashed — but it showed three things
the first run had hidden:

1. **The OCR route had no entrance.** §4GL made `vision_analysis` reachable
   and the model DID reach for it — "Let me try vision_analysis on the
   notification screenshot. But I need to find the image URL" — and then
   could not get one: X/Twitter and Telegram serve images from scripted,
   expiring blob URLs that `extract_text` never returns. The screenshot went
   unread again, for a new reason. The route that exists is
   `browser(operation="screenshot")` → PNG → `extract_text_picture`, and
   nothing said so.

2. **The reply claimed a file it never wrote.** It ended "The report is saved
   to `/workspace/…md`" while the log said `Dropping 1 tool_call(s) —
   final-generation turn (names=['file_system'])`. The honesty note was
   appended, but it disclaimed the FUTURE ("described above as about to
   happen"), which does not touch a past-tense claim — and FILE-ARTIFACT
   reported "clean … checked for emptiness only — no files written this
   turn", because prose claims are deliberately not absence-grade.

3. **Breadth cost too much.** Two dark-web queries ran against a brief
   enumerating BreachForums, Exploit, RaidForums mirrors, paste sites and
   leak indexes — each phrasing was a whole sequential tool call.

The world each pin fails in: a tree where the screenshot→OCR route is not
stated where the model looks, where a turn that admits it ate its write can
still certify the file, or where extra phrasings are dropped, merged without
corroboration, or leak one circuit across queries.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ghost_agent.core import agent as agent_mod
from ghost_agent.core import prompts as prompts_mod
from ghost_agent.core.agent import (GhostAgent, _dropped_mutation_note,
                                    _dropped_write_admitted,
                                    _dropped_write_paths)
from ghost_agent.core.verifier import VerifyResult, VerifyVerdict
from ghost_agent.tools import darkweb_search as D
from ghost_agent.tools import registry as registry_mod
from ghost_agent.tools.outcome import ToolOutcome
from tests.helpers import make_context


# ── 1. the OCR route, stated where the model actually looks ───────────────

def _tool_def(name):
    ctx = make_context()
    for d in registry_mod.get_active_tool_definitions(ctx):
        fn = d.get("function") or {}
        if fn.get("name") == name:
            return fn
    raise AssertionError(f"{name} is not in the active tool definitions")


def test_the_vision_tool_says_what_to_do_when_there_is_no_image_url():
    """The blocker was not "vision exists?" — §4GL settled that — it was
    "I have a post, not an image URL"."""
    target = (_tool_def("vision_analysis")["parameters"]["properties"]
              ["target"]["description"]).lower()
    assert "screenshot" in target and "browser" in target
    assert "blob" in target or "expiring" in target, target
    assert "out_path" in target, target


def test_the_browser_tool_says_a_screenshot_is_how_you_read_someone_elses_image():
    op = (_tool_def("browser")["parameters"]["properties"]
          ["operation"]["description"]).lower()
    assert "extract_text_picture" in op, op
    assert "vision_analysis" in op


def test_the_system_prompt_names_the_two_step_route_not_just_the_tool():
    sp = prompts_mod.SYSTEM_PROMPT
    assert 'browser(operation="screenshot"' in sp
    assert "extract_text_picture" in sp
    low = sp.lower()
    i = low.find("a screenshot is evidence")
    assert i >= 0, "the §4GL bullet is gone — re-point this pin"
    bullet = low[i:i + 1400]
    assert "could not find the image url" in bullet, bullet[-400:]


# ── 2. a turn that ate its own write cannot certify the file ──────────────

NOTE = _dropped_mutation_note(["file_system"], ["/workspace/out.md"])

EVIDENCE_ROW = {
    "role": "tool", "tool_call_id": "c1", "name": "web_search",
    "content": ToolOutcome.ok(
        "### 1. A source\n[Source: https://example.org/a]\n" + "detail " * 80,
        call_args={"query": "revolut breach"}),
}


def test_the_note_contradicts_the_past_tense_and_names_the_file():
    assert "/workspace/out.md" in NOTE
    assert "does NOT exist" in NOTE
    low = NOTE.lower()
    assert "written, saved or created was actually written" in low, NOTE
    # a drop with no path still produces the note, just without the name
    bare = _dropped_mutation_note(["file_system"])
    assert bare and "/workspace" not in bare
    # …and a dropped terminal tool stays silent, which is the point of the guard
    assert _dropped_mutation_note(["self_play"], ["/x.md"]) == ""


def test_the_dropped_paths_come_from_the_calls_own_arguments():
    calls = [
        {"function": {"name": "file_system",
                      "arguments": '{"operation":"write","path":"/report.md"}'}},
        {"function": {"name": "file_system",
                      "arguments": {"operation": "write", "path": " /b.md "}}},
        {"function": {"name": "web_search", "arguments": '{"query":"x"}'}},
        {"function": {"name": "file_system", "arguments": "{not json"}},
    ]
    assert _dropped_write_paths(calls) == ["/report.md", "/b.md"]
    assert _dropped_write_paths(None) == []


def test_the_drop_site_feeds_the_note_the_calls_own_arguments():
    """`_dropped_write_paths` being correct buys nothing if the place that
    eats the call never calls it. Read the wiring from the AST: inside
    `handle_chat`, every `_dropped_mutation_note(...)` that runs on the
    finish-line drop must be handed the parsed paths."""
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(agent_mod))
    fed = False
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "id", "") == "_dropped_mutation_note"):
            continue
        for arg in node.args[1:]:
            if (isinstance(arg, ast.Call)
                    and getattr(arg.func, "id", "") == "_dropped_write_paths"):
                fed = True
    assert fed, ("no call site passes _dropped_write_paths(...) to the note — "
                 "the note cannot name the file it ate")


def test_only_a_note_that_names_file_system_is_absence_grade():
    assert _dropped_write_admitted(NOTE) is True
    assert _dropped_write_admitted(_dropped_mutation_note(["execute"])) is False
    assert _dropped_write_admitted("I saved it to /workspace/out.md") is False


def _agent_with_a_judge_that_confirms(sandbox):
    async def _confirm(*a, **k):
        return VerifyResult(verdict=VerifyVerdict.CONFIRMED, confidence=0.95,
                            reasoning="the judge is happy", issues=[])
    ctx = make_context()
    ctx.verifier = SimpleNamespace(llm_client=object(), verify_claim=_confirm,
                                   verify_code_output=_confirm)
    agent = GhostAgent(ctx)
    agent._scoped_sandbox_for = lambda *a, **k: str(sandbox)
    return agent


async def _verdict(agent, reply):
    v, _lt = await agent._compute_verifier_verdict(
        tools_run_this_turn=[EVIDENCE_ROW], messages=[],
        final_ai_content=reply, last_user_content="investigate and report",
        lc="investigate and report", req_id="r1", trajectory_id="t1")
    return v


async def test_a_claimed_file_is_refuted_when_the_turn_admits_it_ate_the_write(tmp_path):
    """req da4c17ba: 24 minutes of work ending in a sentence that was false
    when it was written. The file is not there; the turn's own note says the
    write never ran; the verdict must not be CONFIRMED."""
    agent = _agent_with_a_judge_that_confirms(tmp_path)
    v = await _verdict(agent, "Done. The report is saved to /workspace/out.md." + NOTE)
    assert v is not None and v.verdict == VerifyVerdict.REFUTED, v
    assert "out.md" in (v.reasoning or ""), v.reasoning


async def test_the_same_claim_WITHOUT_the_note_is_still_not_refuted(tmp_path):
    """The false-refute class stays closed. A reply may mention a file that
    predates the turn — that is why prose claims ride the emptiness-only arm
    — and nothing here changes it. The two worlds differ ONLY by the note."""
    agent = _agent_with_a_judge_that_confirms(tmp_path)
    v = await _verdict(agent, "Done. The report is saved to /workspace/out.md.")
    assert v is not None and v.verdict != VerifyVerdict.REFUTED, v


async def test_a_claimed_file_that_REALLY_EXISTS_survives_the_note(tmp_path):
    """The note arms the absence check; it does not fabricate absence."""
    (tmp_path / "out.md").write_text("# the real report\nbody\n")
    agent = _agent_with_a_judge_that_confirms(tmp_path)
    v = await _verdict(agent, "Done. The report is saved to /workspace/out.md." + NOTE)
    assert v is not None and v.verdict != VerifyVerdict.REFUTED, v


# ── 3. breadth in one round ───────────────────────────────────────────────

def test_the_query_set_caps_dedupes_and_sanitizes():
    """The cap bounds FETCH COST (engines × phrasings). A duplicate costs
    nothing to drop, so it must not eat a slot — "alpha  breach" normalises
    onto the main query, and "delta dump" takes the slot it would have
    wasted. The fifth phrasing is past the cap and goes."""
    qs = D._query_set("alpha breach", ["alpha  breach", "beta leak",
                                       "gamma paste", "delta dump", "eps x"])
    assert qs[0] == "alpha breach"
    assert "beta leak" in qs and "gamma paste" in qs
    assert len(qs) == 1 + D._MAX_EXTRA_QUERIES, qs
    assert qs.count("alpha breach") == 1, qs
    assert "delta dump" in qs, qs                      # the duplicate's slot
    assert "eps x" not in qs, qs                       # past the cap
    assert D._query_set("solo", None) == ["solo"]
    assert D._query_set("solo", "one more") == ["solo", "one more"]
    assert D._query_set("solo", [None, 7, {"q": "x"}]) == ["solo"]


def _stub_raw(seen, rows_by_query):
    async def _raw(query, proxy, max_results=12):
        seen.append((query, proxy))
        return (rows_by_query.get(query, []), [], False, 4)
    return _raw


#: ⚠ `zzzz` is DISCOVERED FIRST and corroborated by nothing; `aaaa` is
#: discovered second and surfaced again by the other phrasing. Any fixture
#: where the corroborated row is also the first row cannot tell ranking from
#: insertion order — and a mutant that deletes the sort survives it.
ROWS = {
    "alpha breach": [{"url": "http://zzzz.onion/first", "title": "Z", "snippet": "",
                      "engines": {"ahmia"}, "indexes": {"ahmia"}},
                     {"url": "http://aaaa.onion/x", "title": "A", "snippet": "s1",
                      "engines": {"ahmia"}, "indexes": {"ahmia"}},
                     {"url": "http://bbbb.onion/y", "title": "B", "snippet": "",
                      "engines": {"torch"}, "indexes": {"torch"}}],
    "beta leak": [{"url": "http://aaaa.onion/other", "title": "A2", "snippet": "s2",
                   "engines": {"torch"}, "indexes": {"torch"}},
                  {"url": "http://cccc.onion/z", "title": "C", "snippet": "",
                   "engines": {"ahmia"}, "indexes": {"ahmia"}}],
}


async def test_every_phrasing_runs_and_corroboration_across_them_ranks_first():
    seen = []
    with patch.object(D, "_darkweb_search_raw", _stub_raw(seen, ROWS)):
        out = await D.tool_darkweb_search(query="alpha breach",
                                          extra_queries=["beta leak"],
                                          tor_proxy="socks5h://127.0.0.1:9050")
    assert [q for q, _p in seen] == ["alpha breach", "beta leak"], seen
    assert "2 phrasings" in out
    # aaaa.onion was surfaced by BOTH phrasings and two indexes, so it leads
    # the row DISCOVERED BEFORE IT on one index — corroboration, not order.
    assert out.index("aaaa.onion") < out.index("zzzz.onion"), out
    assert out.index("aaaa.onion") < out.index("cccc.onion"), out
    assert "matched: alpha breach; beta leak" in out, out


async def test_one_query_still_produces_the_old_output_shape():
    """Everything downstream reads this text (and the 5-minute cache stores
    it), so the single-query call must be byte-identical to what it was."""
    seen = []
    with patch.object(D, "_darkweb_search_raw", _stub_raw(seen, ROWS)):
        out = await D.tool_darkweb_search(query="alpha breach",
                                          tor_proxy="socks5h://127.0.0.1:9050")
    assert len(seen) == 1
    assert "phrasings" not in out and "matched:" not in out
    assert out.startswith("[Dark-web search — onion results, engines reached:")


async def test_each_phrasing_gets_its_own_anonymous_circuit():
    """The per-query identity tag exists to stop cross-query linkability —
    running the set on one circuit would relink exactly what it separates."""
    seen = []
    with patch.object(D, "_darkweb_search_raw", _stub_raw(seen, ROWS)):
        await D.tool_darkweb_search(query="alpha breach",
                                    extra_queries=["beta leak"],
                                    anonymous=True,
                                    tor_proxy="socks5h://127.0.0.1:9050")
    proxies = [p for _q, p in seen]
    assert len(set(proxies)) == 2, proxies
    assert all("@" in p for p in proxies), proxies      # identity-tagged


async def test_one_failing_phrasing_does_not_lose_the_others():
    async def _raw(query, proxy, max_results=12):
        if query == "beta leak":
            raise RuntimeError("tor said no")
        return (ROWS[query], [], False, 4)
    with patch.object(D, "_darkweb_search_raw", _raw):
        out = await D.tool_darkweb_search(query="alpha breach",
                                          extra_queries=["beta leak"],
                                          tor_proxy="socks5h://127.0.0.1:9050")
    assert "aaaa.onion" in out and "bbbb.onion" in out


def test_the_schema_offers_the_extra_phrasings():
    props = _tool_def("darkweb_search")["parameters"]["properties"]
    assert props["extra_queries"]["type"] == "array"
    assert props["extra_queries"]["items"]["type"] == "string"
    desc = _tool_def("darkweb_search")["description"].lower()
    assert "same round" in desc and "extra_queries" in desc
