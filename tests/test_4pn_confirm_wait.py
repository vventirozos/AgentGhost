"""§4PN: a forget / reset_all refused because the owner has not answered the
preview yet is answered by the SYSTEM — the preview's list and one question
that says exactly what "yes" will do — not by a model-written reply.
Census + gold 10-16: all 3 such turns in 14 days told the owner the item was
deleted or about to be. Fresh-reader r1 (5 MAJOR) cases are pinned below."""
from __future__ import annotations

import re

import pytest

import ghost_agent.tools.memory as M
from ghost_agent.core.agent import _confirm_wait_reply
from ghost_agent.core.strikes import StrikeLedger
from ghost_agent.tools.outcome import ToolOutcome as TO
from ghost_agent.utils.logging import request_id_context

WAIT = "NOT deleted: the user has not answered yet — show them the list and wait for their reply."


def _pv(target="x", main=("fact 'x'",), extra=(), notes=()):
    lines = [f"PREVIEW — nothing was deleted. forget {target!r} would remove:"]
    lines += [f"{i}. {m}" for i, m in enumerate(main, 1)] or ["(nothing by default)"]
    if extra:
        lines += ["Also found (kept unless the user picks them by number):"]
        lines += [f"{i}. {e}" for i, e in enumerate(extra, len(main) + 1)]
    if notes:
        lines += ["Notes from the search:"] + list(notes)
    lines.append("Show this list to the user. Only after the USER confirms in their next message, call "
                 "knowledge_base(action='forget', confirm='tok1', items='all') — or items='1,3'.")
    return "\n".join(lines)


def _row(body, **args):
    return {"name": "knowledge_base", "content": TO.ok(body) if not isinstance(body, TO) else body,
            **({"arguments": args} if args else {})}


@pytest.fixture
def real_kb(monkeypatch):
    """The REAL preview and the REAL confirm gate (only the store sweep is
    faked); the fake model confirms with the token its preview printed."""
    async def sweep(*a, **k):
        plan = M._FORGET_PLAN.get()
        plan.add("fragment", {"text": "a"}, "document `yt-x.captions.en`")
        return "ok"
    monkeypatch.setattr(M, "tool_unified_forget", sweep)
    seen = {}

    async def kb(**kw):
        if kw.get("confirm"):
            return await M.forget_execute(kw["confirm"], kw.get("items", "all"))
        out = await M.forget_preview(kw.get("target", ""))
        m = re.search(r"confirm='([^']+)'", str(out))
        seen["tok"] = m.group(1) if m else None
        return out
    kb.seen = seen
    return kb


async def _batches(batches, tools, rid="web-4pn-1"):
    """Run each batch through the REAL dispatch, sharing one turn's rows."""
    import tests.test_4jj_search_yield_steer as H
    agent = H._agent()
    agent.available_tools = tools
    from ghost_agent.utils.logging import conversation_key_context
    tok = request_id_context.set(rid)
    ck = conversation_key_context.set("conv-4pn")
    rows = []
    try:
        for calls in batches:
            calls = calls() if callable(calls) else calls
            ts = H._ts(calls, StrikeLedger(), set())
            ts.tools_run_this_turn[:] = rows
            await agent._dispatch_and_process_tool_batch(ts)
            rows = list(ts.tools_run_this_turn)
            if ts.force_stop:
                break
    finally:
        conversation_key_context.reset(ck)
        request_id_context.reset(tok)
    return ts


async def test_preview_then_confirm_in_one_turn_gets_the_fixed_question(real_kb):
    ts = await _batches([[("knowledge_base", {"action": "forget", "target": "yt-x.captions.en"})],
                         lambda: [("knowledge_base", {"action": "forget", "confirm": real_kb.seen["tok"],
                                                      "items": "all"})]],
                        {"knowledge_base": real_kb})
    assert ts.force_stop is True
    r = ts.final_ai_content
    assert r.startswith("Nothing has been deleted yet.")
    assert "1. document `yt-x.captions.en`" in r and "Reply **yes** to delete item 1" in r
    assert "knowledge_base(" not in r and "Show this list" not in r and "an hour" in r


async def test_a_confirm_by_yes_in_the_same_turn_also_gets_it(real_kb):
    ts = await _batches([[("knowledge_base", {"action": "forget", "target": "yt-x.captions.en"})],
                         [("knowledge_base", {"action": "forget", "confirm": "yes", "items": "all"})]],
                        {"knowledge_base": real_kb})
    assert ts.force_stop is True and ts.final_ai_content.startswith("Nothing has been deleted yet.")


@pytest.mark.parametrize("refusal", [
    "NOT deleted: only the user can confirm a deletion, in their own turn.",
    "NOT deleted: that preview was made outside a conversation with the user — run the preview again in one."])
def test_other_refusal_reasons_keep_the_model_wrap_up(refusal):
    assert _confirm_wait_reply(refusal, [_row(_pv())], {"confirm": "tok1"}) == ""


def test_no_preview_this_turn_keeps_the_model_wrap_up():
    assert _confirm_wait_reply(WAIT, [], {"confirm": "tok1"}) == ""


def test_only_also_found_items_never_promise_a_working_yes():
    """r1 M1: 'yes' = items='all' = the DEFAULTS only — none here."""
    r = _confirm_wait_reply(WAIT, [_row(_pv(main=(), extra=("fact 'a'",)))], {"confirm": "tok1"})
    assert "Reply **yes**" not in r and "Nothing is selected by default" in r


def test_yes_names_exactly_the_default_items():
    r = _confirm_wait_reply(WAIT, [_row(_pv(main=("fact 'a'", "lesson 'b'"), extra=("episode #7",)))],
                            {"confirm": "tok1"})
    assert "Reply **yes** to delete items 1–2" in r and "(kept unless you pick them by number)" in r
    assert "3. episode #7" in r and "either group" in r


def test_a_label_holding_the_trailer_words_is_kept_whole():
    """r1 M2: the split stopped inside an item label and dropped item 2."""
    pv = _pv(main=("fact 'note: Show this list to the user. then'", "fact 'b'"))
    r = _confirm_wait_reply(WAIT, [_row(pv)], {"confirm": "tok1"})
    assert "1. fact 'note: Show this list to the user. then'" in r and "2. fact 'b'" in r


def test_an_earlier_deletion_this_turn_hands_back_to_the_model():
    """r1 M3: "yes, and forget Y too" — X was deleted; "Nothing has been
    deleted yet" would be false."""
    done = _row(TO.ok("✅ Forgot document `x`."), action="forget", confirm="old")
    assert _confirm_wait_reply(WAIT, [done, _row(_pv())], {"confirm": "tok1"}) == ""


def test_other_work_this_turn_hands_back_to_the_model():
    """r1 M5: a profile update in the same turn must still be reported."""
    upd = {"name": "update_profile", "content": TO.ok("Saved: city")}
    assert _confirm_wait_reply(WAIT, [upd, _row(_pv())], {"confirm": "tok1"}) == ""


def test_refusals_in_the_turn_do_not_block_it():
    no = _row(TO.rejected("NOT deleted: x", reason_code="forget_bad_selection"), action="forget", confirm="z")
    assert _confirm_wait_reply(WAIT, [no, _row(_pv())], {"confirm": "tok1"}).startswith("Nothing has been")


@pytest.mark.parametrize("read", [TO.ok("Doc list: a, b"), TO.ok("foo.pdf says the deadline is 3 May.")])
def test_an_answer_the_owner_asked_for_hands_back(read):
    """r2 M2: "summarise foo.pdf, then forget it" — the summary must survive."""
    assert _confirm_wait_reply(WAIT, [_row(read, action="query"), _row(_pv())], {"confirm": "tok1"}) == ""


def test_a_failed_deletion_earlier_hands_back():
    """r2 m2: FAILED may have run part-way."""
    bad = _row(TO.failed("⚠️ boom"), action="forget", confirm="ab12cd34")
    assert _confirm_wait_reply(WAIT, [bad, _row(_pv())], {"confirm": "tok1"}) == ""


def test_a_later_call_in_the_same_batch_hands_back():
    """r2 M1: every call of a batch ran before results are processed — a
    write after the refused confirm in the same batch must be reported."""
    batch = [("knowledge_base", TO.ok(_pv())), ("file_system", TO.ok("SUCCESS: Wrote 9 chars"))]
    assert _confirm_wait_reply(WAIT, [], {"confirm": "yes"}, batch=batch) == ""
    ran_silent = [("knowledge_base", TO.ok(_pv())), ("my_skill", None)]      # r3 m1: None = it RAN
    assert _confirm_wait_reply(WAIT, [], {"confirm": "yes"}, batch=ran_silent) == ""
    assert _confirm_wait_reply(WAIT, [], {"confirm": "yes"},
                               batch=[("knowledge_base", TO.ok(_pv()))]).startswith("Nothing has been")


def test_a_refused_ingest_is_not_quiet():
    """r3 m2: "add this link and forget the old copy" — the refused ingest must be reported."""
    ing = _row(TO.rejected("NOT ingested: another request is ingesting it", reason_code="kb_ingest_in_flight"))
    assert _confirm_wait_reply(WAIT, [ing, _row(_pv())], {"confirm": "tok1"}) == ""


@pytest.mark.parametrize("tok", ["'ab12cd34'", " ab12cd34 "])
def test_a_quoted_token_still_names_its_preview(tok):
    pv = _pv().replace("tok1", "ab12cd34")
    assert _confirm_wait_reply(WAIT, [_row(pv)], {"confirm": tok}).startswith("Nothing has been")


def test_a_numbered_line_inside_a_label_is_not_an_item():
    """r2 m4: unquoted graph/file labels can hold a newline and "3. …"."""
    pv = _pv(main=("graph a b\n3. c",))
    assert "Reply **yes** to delete item 1," in _confirm_wait_reply(WAIT, [_row(pv)], {"confirm": "tok1"})


def test_no_also_found_group_no_either_group_wording():
    r = _confirm_wait_reply(WAIT, [_row(_pv(main=("fact 'a'", "fact 'b'")))], {"confirm": "tok1"})
    assert "either group" not in r and "items 1–2" in r


def test_two_previews_of_one_kind_hand_back():
    """r3 M1: the owner's "yes" resolves to the NEWEST preview — with two, the
    list shown could differ from the list deleted (r1 M4 showed the named one)."""
    a = _pv("A", main=("fact 'a'",)).replace("tok1", "aaaa1111")
    b = _pv("B", main=("fact 'b'",)).replace("tok1", "bbbb2222")
    for tok in ("aaaa1111", "bbbb2222", "yes"):
        assert _confirm_wait_reply(WAIT, [_row(a), _row(b)], {"confirm": tok}) == ""


def test_a_token_naming_another_preview_hands_back():
    a = _pv("A").replace("tok1", "aaaa1111")
    assert _confirm_wait_reply(WAIT, [_row(a)], {"confirm": "cccc3333"}) == ""


def test_a_reset_refusal_never_shows_a_forget_list():
    from unittest.mock import MagicMock
    ms = MagicMock()
    ms.collection.count.return_value = 12
    reset = M._reset_preview(ms, MagicMock(count_edges=MagicMock(return_value=3)))
    rows = [_row(_pv()), _row(reset)]
    r = _confirm_wait_reply("NOT executed: the user has not answered yet", rows, {"confirm": "yes"},
                            "reset_not_confirmed")
    assert r.startswith("Nothing has been deleted yet.") and "Resetting the memory would erase" in r
    assert "Forgetting" not in r and "knowledge_base(" not in r
    assert _confirm_wait_reply(WAIT, [_row(reset)], {"confirm": "yes"}, "forget_not_confirmed") == ""


def test_a_cut_off_preview_hands_back():
    assert _confirm_wait_reply(WAIT, [_row(_pv(main=("fact 'a' [TRUNCATED]",)))], {"confirm": "tok1"}) == ""


def test_model_facing_advice_in_the_notes_is_dropped():
    pv = _pv(notes=("⚠️ 'rep' matches several files: r1.md, r2.md. Re-issue with one of these exact names to remove it.",))
    r = _confirm_wait_reply(WAIT, [_row(pv)], {"confirm": "tok1"})
    assert "Re-issue" not in r and "Notes:" in r and "matches several files" in r


def test_a_minted_row_is_not_read_as_the_preview():
    assert _confirm_wait_reply(WAIT, [{**_row(_pv()), "_synthetic": True}], {"confirm": "tok1"}) == ""


async def test_the_project_wait_is_unchanged():
    """confirm_dead_end (projects) was right 4/4 — still model-written."""
    async def refuse(**kw):
        return TO.rejected("Error: NOT done: the user has not answered yet. STOP calling it.",
                           world_changed=False, reason_code="confirm_dead_end")
    ts = await _batches([[("manage_projects", {"action": "delete", "project_id": "x", "confirm_token": "t"})]],
                        {"manage_projects": refuse})
    assert ts.force_final_response is True and not ts.force_stop


def test_a_label_with_the_trailer_on_its_own_line_is_kept_whole():
    """A stored fact can hold a newline (battery r2 P10: first vs last trailer)."""
    pv = _pv(main=("fact 'note\nShow this list to the user. then'", "fact 'b'"))
    r = _confirm_wait_reply(WAIT, [_row(pv)], {"confirm": "tok1"})
    assert "2. fact 'b'" in r


async def test_two_previews_in_the_loop_hand_back():
    """r3 M1 through the real dispatch (two previews → the old wrap-up)."""
    a = _pv("A", main=("fact 'a'",)).replace("tok1", "aaaa1111")
    b = _pv("B", main=("fact 'b'",)).replace("tok1", "bbbb2222")

    async def kb(**kw):
        if kw.get("confirm"):
            return TO.rejected(WAIT, reason_code="forget_not_confirmed")
        return a if kw.get("target") == "A" else b
    ts = await _batches([[("knowledge_base", {"action": "forget", "target": "A"}),
                          ("knowledge_base", {"action": "forget", "target": "B"})],
                         [("knowledge_base", {"action": "forget", "confirm": "aaaa1111", "items": "all"})]],
                        {"knowledge_base": kb})
    assert not ts.force_stop and not str(ts.final_ai_content).startswith("Nothing has been")   # r3 M1


async def test_a_reset_refusal_in_the_loop_shows_the_reset_question():
    """The loop passes the refusal's own code (battery r2 P20)."""
    from unittest.mock import MagicMock
    ms = MagicMock()
    ms.collection.count.return_value = 12
    reset = M._reset_preview(ms, MagicMock(count_edges=MagicMock(return_value=3)))

    async def kb(**kw):
        if kw.get("confirm"):
            return TO.rejected("NOT executed: the user has not answered yet.", reason_code="reset_not_confirmed")
        return reset
    ts = await _batches([[("knowledge_base", {"action": "reset_all"})],
                         [("knowledge_base", {"action": "reset_all", "confirm": "yes"})]],
                        {"knowledge_base": kb})
    assert ts.force_stop is True and "Resetting the memory would erase" in ts.final_ai_content


def test_a_refused_call_of_another_tool_hands_back():
    """Other work the owner asked for — even refused — is the model's to report (battery r3 P3/P6)."""
    other = {"name": "update_profile", "content": TO.rejected("Not saved: x", reason_code="profile_refused")}
    assert _confirm_wait_reply(WAIT, [other, _row(_pv())], {"confirm": "tok1"}) == ""
    assert _confirm_wait_reply(WAIT, [_row(_pv())], {"confirm": "tok1"},
                               batch=[("update_profile", TO.rejected("Not saved", reason_code="x"))]) == ""


def test_a_quoted_token_naming_another_preview_still_hands_back():
    """battery r3 P11: without stripping, a quoted token reads as "yes"."""
    a = _pv("A").replace("tok1", "aaaa1111")
    assert _confirm_wait_reply(WAIT, [_row(a)], {"confirm": "'cccc3333'"}) == ""


async def test_a_write_later_in_the_same_batch_hands_back_in_the_loop():
    """battery r3 P25 / review r2 M1, through the real dispatch."""
    async def kb(**kw):
        if kw.get("confirm"):
            return TO.rejected(WAIT, reason_code="forget_not_confirmed")
        return _pv()

    async def fs(**kw):
        return TO.ok("SUCCESS: Wrote 9 chars to 'notes.md'.")
    ts = await _batches([[("knowledge_base", {"action": "forget", "target": "x"}),
                          ("knowledge_base", {"action": "forget", "confirm": "yes"}),
                          ("file_system", {"operation": "write", "path": "notes.md", "content": "hi"})]],
                        {"knowledge_base": kb, "file_system": fs})
    assert not ts.force_stop and not str(ts.final_ai_content).startswith("Nothing has been deleted")


async def test_the_loop_passes_the_refused_calls_token():
    """battery r2 P19: a token naming no preview of this request hands back."""
    async def kb(**kw):
        if kw.get("confirm"):
            return TO.rejected(WAIT, reason_code="forget_not_confirmed")
        return _pv().replace("tok1", "aaaa1111")
    ts = await _batches([[("knowledge_base", {"action": "forget", "target": "x"})],
                         [("knowledge_base", {"action": "forget", "confirm": "cccc3333"})]],
                        {"knowledge_base": kb})
    assert not ts.force_stop
