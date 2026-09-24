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

NOTE = _dropped_mutation_note(["file_system"], ["/workspace/out.md"],
                              creating=["/workspace/out.md"])
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


# ---------------------------------------------------------------------------
# §4KC (2026-09-23, req 76b3602e): the note must not disclaim what DID land
# ---------------------------------------------------------------------------

def test_a_landed_edit_is_not_disclaimed_by_a_later_drop():
    """Live: one `edit` succeeded at +40 s, a redundant second file_system
    call was dropped at the finish line, and the reply said "nothing
    described above as written … was actually written" over a file that was
    verifiably changed. Fails in that world."""
    note = _dropped_mutation_note(["file_system"], ["/b.md"],
                                  landed=["kc_probe.py"], creating=["/b.md"])
    assert note.startswith("\n\n" + agent_mod._DROPPED_NOTE_HEAD)
    assert "file_system" in note[:400]              # the §4GH strip/detect contract
    assert "kc_probe.py" in note and "DID land" in note
    assert "nothing described above" not in note
    assert "/b.md" in note and "does NOT exist" in note   # the drop is still named
    # with nothing landed, the original full disclaimer is unchanged
    assert "written, saved or created was actually written" in \
        _dropped_mutation_note(["file_system"], ["/b.md"], landed=[])


def test_both_drop_sites_pass_what_landed():
    """Wiring, read from the AST: every finish-line `_dropped_mutation_note`
    call in the turn loop is handed `landed=` from the tool results."""
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(agent_mod))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "id", "") == "_dropped_mutation_note"]
    assert len(calls) == 2, len(calls)
    for c in calls:
        kw = {k.arg: k.value for k in c.keywords}
        assert "landed" in kw, ast.dump(c)[:200]
        assert getattr(kw["landed"].func, "id", "") == "_files_mutated_this_turn"
        # §4KC r3: BOTH sites hand over the dropped paths and which of them
        # are creating ops — site 2 used to pass names only, so a forced-
        # final-dropped WRITE read "was not changed" instead of "does NOT
        # exist" and the landed-reconciliation was inoperative there.
        assert "creating" in kw, ast.dump(c)[:200]
        assert not (isinstance(kw["creating"], ast.List) and not kw["creating"].elts), "creating=[] literal"
        assert len(c.args) >= 2, "paths positional missing"
        assert not (isinstance(c.args[1], ast.List) and not c.args[1].elts), "paths=[] literal"


def test_a_dropped_read_produces_no_disclaimer_and_a_dropped_edit_says_not_changed():
    """[§4KC r2] Neither the gate nor the path list looked at `operation`:
    a dropped verify-`read` produced the full "nothing was written" note,
    and a dropped `edit` on an existing file was told it "does NOT exist"."""
    from ghost_agent.core.agent import (_dropped_fs_calls,
                                        _dropped_mutating_names)
    read = [{"function": {"name": "file_system",
                          "arguments": {"operation": "read", "path": "README.md"}}}]
    assert _dropped_mutating_names(read) == []
    assert _dropped_write_paths(read) == []
    assert _dropped_mutation_note(_dropped_mutating_names(read),
                                  _dropped_write_paths(read)) == ""

    edit = [{"function": {"name": "file_system",
                          "arguments": {"operation": "edit", "path": "app.py",
                                        "old_string": "a", "new_string": "b"}}}]
    assert _dropped_mutating_names(edit) == ["file_system"]
    assert _dropped_fs_calls(edit) == [("edit", "app.py")]
    note = _dropped_mutation_note(["file_system"], ["app.py"])
    assert "`app.py` was not changed" in note and "does NOT exist" not in note

    # a dropped call on a file that ALSO landed this turn is not "missing"
    # and not "unchanged" — the earlier call changed it; only the note's
    # "final pending action" clause remains
    note = _dropped_mutation_note(["file_system"], ["app.py"], landed=["app.py"])
    assert "DID land" in note and "app.py" in note
    assert "does NOT exist" not in note and "was not changed" not in note
    # other tools keep their name-based gate
    assert _dropped_mutating_names([{"function": {"name": "execute", "arguments": {}}}]) == ["execute"]


def test_round3_the_note_names_destinations_dedupes_and_reconciles_redone_writes():
    """[§4KC r3] copy/rename/move create at the DESTINATION (the note said
    the SOURCE "does NOT exist"); aliased spellings of one file were listed
    thrice; and a write dropped on a forced-final miss then RE-DONE by the
    repair re-entry was still disclaimed."""
    from ghost_agent.core.agent import (_dropped_entries, _dropped_fs_calls,
                                        _still_pending_drops)
    copy = [{"function": {"name": "file_system",
                          "arguments": {"operation": "copy", "path": "src.md",
                                        "destination": "dst.md"}}}]
    assert _dropped_fs_calls(copy) == [("copy", "dst.md")]
    note = _dropped_mutation_note(["file_system"], ["dst.md"], creating=["dst.md"])
    assert "`dst.md` does NOT exist" in note and "src.md" not in note
    # the dispatcher's alias chain
    alias = [{"function": {"name": "file_system",
                           "arguments": {"operation": "write", "filename": "b.md"}}}]
    assert _dropped_write_paths(alias) == ["b.md"]
    # dedupe by normalised key
    note = _dropped_mutation_note(["file_system"], ["a.md", "a.md", "/workspace/a.md"],
                                  creating=["a.md"])
    assert note.count("a.md") == 1, note
    # reconcile: a later file_system SUCCESS retires the dropped fs entry
    entries = _dropped_entries(alias, tools_seen=2)
    assert entries == [("file_system", "write", "b.md", 2)]
    runs = [{"name": "file_system", "content": "read stuff"},
            {"name": "web_search", "content": "…"},
            {"name": "file_system", "content": "SUCCESS: Wrote 3 chars to 'b.md'. Script-side path (from sandbox cwd): 'b.md'."}]
    assert _still_pending_drops(entries, runs) == []
    assert _still_pending_drops(entries, runs[:2]) == entries
    # [r4 pre-fix] a later SUCCESS on an UNRELATED file must not retire it
    other = runs[:2] + [{"name": "file_system", "content":
                         "SUCCESS: Wrote 3 chars to 'z.md'. Script-side path (from sandbox cwd): 'z.md'."}]
    assert _still_pending_drops(entries, other) == entries
    # …but the same file under the tool's other spelling does
    alias_hit = runs[:2] + [{"name": "file_system", "content":
                             "SUCCESS: edited — replaced 1 occurrence of old_string (line 1) in '/workspace/b.md'."}]
    assert _still_pending_drops(entries, alias_hit) == []
    # a non-file_system drop has no such signal and stays pending
    ex = _dropped_entries([{"function": {"name": "execute", "arguments": {}}}], 0)
    assert _still_pending_drops(ex, runs) == ex
    assert "pending last" not in _dropped_mutation_note(["file_system"], ["x.md"], landed=["y.md"])


def test_round4_every_op_reconciles_and_a_delete_has_its_own_verb():
    """[§4KC r4] the reconcile knew only the produce shapes, so a re-done
    copy/rename/delete stayed pending; and a dropped delete read "was not
    changed"."""
    from ghost_agent.core.agent import (_confirmed_paths_any_op,
                                        _dropped_entries, _still_pending_drops)
    def entry(op, path):
        return [(("file_system", op, path, 0))]
    assert _still_pending_drops(entry("copy", "b.md"),
        [{"name": "file_system", "content": "SUCCESS: Copied 'a.md' to 'b.md'."}]) == []
    assert _still_pending_drops(entry("rename", "b.md"),
        [{"name": "file_system", "content": "SUCCESS: Renamed/Moved 'a.md' to 'b.md'."}]) == []
    assert _still_pending_drops(entry("delete", "old.md"),
        [{"name": "file_system", "content": "SUCCESS: Deleted 'old.md'."}]) == []
    assert _still_pending_drops(entry("write", "b.md"),
        [{"name": "file_system", "content": "[FAILURE BANNER] x\nSUCCESS: Wrote 3 chars to 'b.md'. Script-side path (from sandbox cwd): 'b.md'."}]) == []
    # unrelated names never retire
    assert _still_pending_drops(entry("copy", "b.md"),
        [{"name": "file_system", "content": "SUCCESS: Copied 'a.md' to 'c.md'."}]) == entry("copy", "b.md")
    assert _confirmed_paths_any_op("REJECTED: nope") == set()
    note = _dropped_mutation_note(["file_system"], ["old.md"], deleting=["old.md"])
    assert "`old.md` was not deleted" in note and "not changed" not in note
    note = _dropped_mutation_note(["file_system"], ["a.md", "old.md", "new.md"],
                                  creating=["new.md"], deleting=["old.md"])
    assert "`new.md` does NOT exist" in note and "`old.md` was not deleted" in note \
        and "`a.md` was not changed" in note


def test_round4_both_drop_sites_pass_deleting():
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(agent_mod))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", "") == "_dropped_mutation_note"]
    assert len(calls) == 2
    for c in calls:
        kw = {k.arg: k.value for k in c.keywords}
        assert "deleting" in kw and not (isinstance(kw["deleting"], ast.List) and not kw["deleting"].elts)


def test_round5_reconcile_is_op_aware_and_a_landed_delete_is_named():
    """[§4KC r5] a dropped WRITE was retired by a later `Deleted` of the same
    name (the file is gone — the note must stand); a dropped DELETE of a
    file that landed this turn was filtered out of the note entirely."""
    from ghost_agent.core.agent import _still_pending_drops
    w = [("file_system", "write", "a.md", 0)]
    d = [("file_system", "delete", "a.md", 0)]
    deleted = [{"name": "file_system", "content": "SUCCESS: Deleted 'a.md'."}]
    wrote = [{"name": "file_system", "content":
              "SUCCESS: Wrote 3 chars to 'a.md'. Script-side path (from sandbox cwd): 'a.md'."}]
    moved = [{"name": "file_system", "content": "SUCCESS: Renamed/Moved 'a.md' to 'b.md'."}]
    assert _still_pending_drops(w, deleted) == w          # write not satisfied by a delete
    assert _still_pending_drops(d, wrote) == d            # delete not satisfied by a write
    assert _still_pending_drops(d, deleted) == []
    assert _still_pending_drops(d, moved) == []           # moved away = gone
    assert _still_pending_drops(w, wrote) == []
    note = _dropped_mutation_note(["file_system"], ["a.md"], landed=["a.md"], deleting=["a.md"])
    assert "`a.md` was not deleted" in note


def test_round6_a_landed_file_is_never_also_missing():
    """[R6] a dropped write AND a dropped delete of a file that landed said
    "DID land" and "does NOT exist" in one sentence."""
    note = _dropped_mutation_note(["file_system"], ["b.txt"], landed=["b.txt"],
                                  creating=["b.txt"], deleting=["b.txt"])
    assert "DID land" in note and "does NOT exist" not in note
    assert "`b.txt` was not deleted" in note
