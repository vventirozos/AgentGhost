"""§4OH — the open items closed after §4OG: unopened citations disclosed in
the reply, the streamed reply's working narration retracted to match the
record, the drain's report write refused when the project changed, and a
Stop during a streamed final persisting the STATE OF WORK."""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ghost_agent.core import agent as ag
from ghost_agent.core.agent import StreamState, unopened_citation_note, _persist_stream_stop_state
from ghost_agent.tools.outcome import ToolOutcome
from tests.pin_ast import tree, U, calls, call_kw, ifs, assigns
from tests.test_finalize_stream_pins import make_stream_agent, sse


def _write_row(path, content):
    out = ToolOutcome.ok(f"SUCCESS: Wrote {len(content)} chars to '{path}'. Script-side path (from sandbox cwd): '{path}'.")
    out.call_args = {"operation": "write", "path": path, "content": content}
    return {"name": "file_system", "content": out}


def _nav_row(url):
    return {"name": "browser", "args": {"operation": "navigate", "url": url},
            "content": f"--- BROWSER RESULT ---\nSTATUS: OK\nOP: navigate\nURL: {url}\nHTTP_STATUS: 200\nLENGTH: 10\n"}


# ── (a) unopened citations are disclosed ────────────────────────────────── #

def test_the_reply_discloses_citations_the_request_never_opened(tmp_path):
    """Fails where the §4OG link check was advisory only: the §4OG research
    probe's report kept three sources it had not opened, and the owner was
    not told which."""
    body = ("# Report\n\n## Sources\n- https://ex.org/a\n- https://ex.org/b\n- https://never.example/p\n"
            "- `https://in-code.example/x`\n")
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / "report.md").write_text(body)
    rows = [_nav_row("https://ex.org/a"), _nav_row("https://www.ex.org/b/"), _write_row("r/report.md", body)]
    note = unopened_citation_note(rows, (tmp_path,))
    assert note.startswith("\n\n_Source check: 1 link this request wrote into the file was not opened during this request")
    assert "https://never.example/p" in note and "`report.md`" in note
    assert "ex.org/a" not in note and "in-code.example" not in note
    # every link read → nothing; nothing written → nothing; a REFUSED write → nothing
    assert unopened_citation_note(rows + [_nav_row("https://never.example/p")], (tmp_path,)) == ""
    assert unopened_citation_note(rows[:2], (tmp_path,)) == ""
    refused = {"name": "file_system", "content": ToolOutcome.rejected("REJECTED: no")}
    refused["content"].call_args = {"operation": "write", "path": "r/report.md"}
    assert unopened_citation_note(rows[:2] + [refused], (tmp_path,)) == ""
    # only URLs THIS request wrote: an append of new text flags nothing old (reader M9)
    app = ToolOutcome.ok("SUCCESS: Appended 20 chars to 'r/report.md'. Script-side path (from sandbox cwd): 'r/report.md'.")
    app.call_args = {"operation": "append", "path": "r/report.md", "content": "\n- https://ex.org/a\n"}
    assert unopened_citation_note(rows[:2] + [{"name": "file_system", "content": app}], (tmp_path,)) == ""
    # a downloaded URL and a darkweb_research source count as read (reader M9)
    from ghost_agent.core.link_grounding import fetched_urls
    assert "https://dl.example/f.pdf" in fetched_urls([{"name": "file_system", "content": "SUCCESS: Downloaded 'https://dl.example/f.pdf' to 'f.pdf'."}])
    assert "http://abc.onion/p" in fetched_urls([{"name": "darkweb_research", "content": "### SOURCE: http://abc.onion/p\ntext\n"}])
    # the tool's own path spelling is tried when the call's path does not resolve (reader m5)
    alt = ToolOutcome.ok("SUCCESS: Wrote 9 chars to 'sandbox/r/report.md'. Script-side path (from sandbox cwd): 'r/report.md'.")
    alt.call_args = {"operation": "write", "path": "sandbox/projects/zz/r/report.md", "content": body}
    assert "never.example" in unopened_citation_note(rows[:2] + [{"name": "file_system", "content": alt}], (tmp_path,))
    # the FILE is the truth (the model fixed it after the link-check note): read from disk
    (tmp_path / "r" / "report.md").write_text("# Report\n- https://ex.org/a\n")
    assert unopened_citation_note(rows, (tmp_path,)) == ""
    # containment: a path that escapes the root is not read, even when the file exists (battery mutant H3)
    root = tmp_path / "root"; root.mkdir()
    (tmp_path / "outside.md").write_text("see https://outside.example/page\n")
    nav = _nav_row("https://ex.org/a")
    assert unopened_citation_note([nav, _write_row("../outside.md", "see https://outside.example/page")], (root,)) == ""
    (root / "inside.md").write_text("see https://outside.example/page\n")
    assert "outside.example" in unopened_citation_note([nav, _write_row("inside.md", "see https://outside.example/page")], (root,))
    # r2 M4: a request that fetched nothing (a rewrite of earlier research) gets no note
    assert unopened_citation_note([_write_row("inside.md", "see https://outside.example/page")], (root,)) == ""
    # r3: a download IS a fetch
    dl = {"name": "file_system", "content": "SUCCESS: Downloaded 'https://dl.example/f.pdf' to 'f.pdf'."}
    assert "outside.example" in unopened_citation_note([dl, _write_row("inside.md", "see https://outside.example/page")], (root,))
    # r2 m5: an EDIT that adds a link is written content too
    ed = ToolOutcome.ok("SUCCESS: edited 'inside.md'.")
    ed.call_args = {"operation": "edit", "path": "inside.md", "old_string": "see", "new_string": "see https://outside.example/page"}
    assert "outside.example" in unopened_citation_note([nav, {"name": "file_system", "content": ed}], (root,))


def test_both_reply_paths_append_the_source_check():
    t = tree(ag)
    sites = calls(t, "unopened_citation_note")
    assert len(sites) == 2, "the non-stream finalize and the streamed drain both disclose"
    assert {U(c.args[0]) for c in sites} == {"tools_run_this_turn", "stream_tools_snapshot"}


# ── (b) the streamed reply drops the working narration ──────────────────── #

NARR = "Two remaining fixes: the CLI parser and the income statement."
BEAT = "Let me run the whole suite again to confirm."
ANSWER = ("All 96 tests pass. Here is the summary of the package:\n\n"
          + "\n".join(f"| module{i} | does thing {i} in the ledger package, described at length |" for i in range(12)))


def _state(reg, prefix, segs, **kw):
    base = dict(session_id="", created_time=123, current_trajectory_id="t1",
                execution_failure_count=0, force_stop=False, forget_was_called=False,
                has_coding_intent=True, is_final_generation=True, last_user_content="build it",
                last_was_failure=False, lc="build it", messages=[{"role": "user", "content": "build it"}],
                model="m", payload={"messages": [{"role": "user", "content": "build it"}]}, req_id="rid0og01",
                stream_conv_fp="fp", stream_messages_snapshot=[], stream_model="m", stream_prefix=prefix,
                stream_tools_snapshot=[{"name": "execute", "content": "2 passed"}], stream_verify_messages=[],
                was_complex_task=True, _active_turn=None, _proj_task_closed_this_req=False, _turn_reg=reg,
                pre_tool_segments=tuple(segs), ff_breaker=False, ff_report_write_used=False, ff_breaker_reason="")
    base.update(kw)
    return StreamState(**base)


async def _drain(prefix, segs, deltas, tools=None, sandbox=None, work=True):
    from ghost_agent.core.reply_smoothing import WorkSegment
    a = make_stream_agent()
    if sandbox is not None:
        a.context.sandbox_dir = sandbox
        a.context.current_project_id = None
    a.context.args.no_verifier = True
    a.context.journal = MagicMock()
    a.context.metacog = MagicMock(); a.context.metacog.enabled = False
    a._journal_append_safe = AsyncMock(); a._record_episode_safe = AsyncMock()
    a._judge_hydration_safe = MagicMock(); a._write_project_work_log_safe = AsyncMock()
    a._record_turn_trajectory = MagicMock()
    a._attach_late_verdict_handler = MagicMock()

    async def _no_verdict(**kw):
        return None
    a._compute_verifier_verdict = _no_verdict

    async def final_stream(payload, use_coding=False):
        for d in deltas:
            yield sse({"content": d})
        yield b"data: [DONE]\n\n"
    a.context.llm_client.stream_chat_completion = final_stream
    reg = MagicMock(); reg.is_cancelled.return_value = False
    st = _state(reg, prefix, [WorkSegment(x) if work else x for x in segs])
    if tools is not None:
        st.stream_tools_snapshot = list(tools)
    gen, _, _ = a._stream_final_generation(st)
    frames = [json.loads(c.decode()[6:]) for c in [c async for c in gen]
              if c.startswith(b"data: ") and c.strip() != b"data: [DONE]"]
    durable = (a._record_turn_trajectory.call_args.kwargs["final_content"]
               if a._record_turn_trajectory.called else "")
    return frames, durable


def _client_view(frames):
    text = ""
    for f in frames:
        if (f.get("ghost") or {}).get("retract") is True:
            text = ""
            continue
        text += ((f.get("choices") or [{}])[0].get("delta") or {}).get("content") or ""
    return text


@pytest.mark.asyncio
async def test_the_streamed_reply_drops_step_announcements_and_matches_the_record():
    """Fails where the prefix delivered the tool turns' "Let me …" beats ahead
    of the answer while the record dropped them. Three earlier designs (a late
    retract; a release after the answer; a hold-by-position rule) broke other
    things — now only strict beats go, nothing comes back, one text everywhere."""
    frames, durable = await _drain(BEAT + "\n\n" + NARR + "\n\n", [BEAT, NARR], [ANSWER])
    view = _client_view(frames)
    assert BEAT not in view and NARR in view and view.index(NARR) < view.index("All 96 tests pass")
    assert not any((f.get("ghost") or {}).get("retract") for f in frames)
    assert durable.strip() == view.strip()


@pytest.mark.asyncio
@pytest.mark.parametrize("answer,closing,final", [
    # reader r3 shape A: the whole answer written beside a pointless call, then a beat, then a short final
    ("Here's the breakdown: 6 stack tests and 7 queue tests, all passing. Final count: 13 tests.",
     "Let me double-check the file contents.", "Done — all 13 tests passed, no fixes needed."),
    # B: a lookup answer, then a confirmation
    ("The capital of Australia is Canberra (per the government's own site).", "The search confirms it.", "Confirmed."),
    # C: a work answer, then a close
    ("Fixed both bugs; All 96 tests pass and the trial balance is 6000.00 on both sides.",
     "Closing task 3 now.", "Task 3 closed."),
    # r3 MAJOR 2: classifier "beats" that are answers
    ("Perfect! All 96 tests pass and the trial balance is 6000.00 on both sides.", "", "Task closed."),
    ("Great, the capital is Canberra.", "", "Task closed."),
    ("Okay, your flight leaves at 14:35 from gate B12.", "", "Task closed."),
    ("I'll keep it short: the server is up on port 8101.", "", "Task closed."),
])
async def test_an_answer_written_beside_a_tool_call_is_never_lost(answer, closing, final):
    segs = [answer] + ([closing] if closing else [])
    prefix = "".join(x + "\n\n" for x in segs)
    frames, durable = await _drain(prefix, segs, [final])
    view = _client_view(frames)
    assert answer in view and answer in durable and durable.strip() == view.strip()
    if closing.startswith("Let me"):
        assert closing not in view


def test_split_pre_tool_segments_takes_strict_beats_only():
    from ghost_agent.core.reply_smoothing import split_pre_tool_segments, _strict_beat
    for b in ("Let me run the suite again.", "Now I'll fix the parser.", "I need to check the logs first.",
              "Next, let me open the release notes.", "Time to write the report."):
        assert _strict_beat(b), b
    # r4: an opener does not make a beat — a fact, a question, a refusal or a reminder after it stays
    for b in ("I'll remind you tomorrow at 9.", "I will not run that command — it would wipe your home directory.",
              "I need to correct myself — the capital is Canberra, not Sydney.", "Let me pause here — should I delete the old data?",
              "Let me know if you want the PDF version too.", "OK, I'll stop here — 3 of 5 tests still fail.",
              "Let me check — the price is 42 EUR.", "Let me be clear, the server is down and the backup failed.",
              "I'll use port 8103 instead since 8102 is taken by Chess Coach.", "Let me fix both:",
              "Now I need to fix the transcribe detection in cli.py — it looks for the wrong word.",
              "Let me check the logs. Let me open the config file.",                                   # two sentences (battery B4)
              "Let me read the whole configuration directory carefully and compare every setting against the documented defaults for this release today."):  # > 120 (B8)
        assert not _strict_beat(b), b
    for b in ("Perfect! All 96 tests pass.", "Great, the capital is Canberra.", "I'll keep it short: the server is up on 8101.",
              "Let me explain. The parser was wrong because of the header.", "Two remaining fixes: the parser.",
              "The task is complete. Let me mark it done.", "Let me " + "x" * 300):
        assert not _strict_beat(b), b
    pre = "BANNER line\n\n   indented\n\n\nLet me check the logs.\n\nKeep this paragraph.\n\n"
    kept, held = split_pre_tool_segments(pre, ["Let me check the logs.\n\nKeep this paragraph."])
    # paragraph granularity; the prefix's own whitespace is kept (r2 m6)
    assert held == ["Let me check the logs."] and kept == "BANNER line\n\n   indented\n\n\nKeep this paragraph.\n\n"
    kept, held = split_pre_tool_segments("Let me check. And more text.\n\n", ["Let me check."])
    assert held == []                                                          # not a whole paragraph
    assert split_pre_tool_segments("", ["x"]) == ("", []) and split_pre_tool_segments("a\n\n", []) == ("a\n\n", [])


# ── (e) the drain's report write and a Stop mid-drain ───────────────────── #

@pytest.mark.asyncio
async def test_the_drain_report_write_is_refused_when_the_project_changed():
    """Fails where the drain's file write used the LIVE project id — a request
    that started meanwhile had re-pointed it."""
    a = make_stream_agent()
    fs = AsyncMock(return_value="SUCCESS: Wrote 10 chars to 'r.md'. Script-side path (from sandbox cwd): 'r.md'.")
    a.available_tools = {"file_system": fs}
    a.context.current_project_id = "bbbbbbbbbbbb"
    calls_ = [{"id": "c1", "function": {"name": "file_system",
                                         "arguments": json.dumps({"operation": "write", "path": "r.md", "content": "# R\n"})}}]
    landed, rows = await a._run_streamed_report_writes(calls_, [], project_id="aaaaaaaaaaaa")
    assert landed == [] and fs.await_count == 0
    assert rows and rows[0].get("_synthetic") and getattr(rows[0]["content"], "reason_code", "") == "project_changed"
    landed, rows = await a._run_streamed_report_writes(calls_, [], project_id="bbbbbbbbbbbb")
    assert fs.await_count == 1


def test_a_stop_during_the_streamed_final_persists_the_state_of_work():
    """Fails where only handle_chat's finally persisted it — before the drain,
    so a Stop pressed while the final streamed left nothing to resume."""
    pad = MagicMock()
    rows = [{"name": "execute", "content": "2 passed"}]
    assert _persist_stream_stop_state(MagicMock(scratchpad=pad), "sess-1", rows, "build it", "") is True
    key, val = pad.set.call_args.args[:2]
    assert key == "_state_of_work:sess-1" and "stopped while it streamed" in val["text"]
    pad.reset_mock()
    assert _persist_stream_stop_state(MagicMock(scratchpad=pad), "sess-1", rows, "x", "member") is False
    assert _persist_stream_stop_state(MagicMock(scratchpad=pad), "", rows, "x", "") is False
    assert _persist_stream_stop_state(MagicMock(scratchpad=pad), "s", [{"name": "x", "_synthetic": True}], "x", "") is False
    assert not pad.set.called
    t = tree(ag.GhostAgent._stream_final_generation)
    guarded = ifs(t, "not _drained or _cancelled_mid_drain")                 # reader M7: a cooperative Stop drains to the end
    assert guarded and calls(guarded[0], "_persist_stream_stop_state")
    assert any(U(a.value) == "bool(_drain_flags.get('cut'))" for a in assigns(t, "_cancelled_mid_drain"))
    assert any(U(a.value) == "True" for a in assigns(t, "_drain_flags['cut']"))   # r2 m4: set only where the Stop cut
    assert any(U(a.value) == "True" for a in assigns(t, "_drained"))


def test_our_notes_hand_nothing_over_and_are_stripped_for_the_judges():
    """Reader M8/m4: the note names the file in a code span — it made "I could
    not finish report.md" look delivered, and the judges read it as the model's."""
    from ghost_agent.distill.schema import ToolCall
    from ghost_agent.core.agent import delivered_ok_from_calls
    from ghost_agent.core.reply_smoothing import strip_system_notes
    wrote = ToolCall(name="file_system", arguments={"operation": "write", "path": "r/report.md"},
                     result="SUCCESS: Wrote 9 chars to 'r/report.md'. Script-side path (from sandbox cwd): 'r/report.md'.")
    note = "\n\n_Source check: 1 link this request wrote into the file was not opened during this request (x) — `report.md`: https://n.example._"
    assert not delivered_ok_from_calls([wrote], "I could not finish report.md." + note)
    assert strip_system_notes("Answer text." + note) == "Answer text."


@pytest.mark.asyncio
async def test_the_streamed_source_check_reaches_the_record(tmp_path):
    """Battery P6 / reader M1: a note appended to full_content only was absent
    from the record (and erased by the old retract)."""
    body = "# R\n- https://never.example/p\n"
    (tmp_path / "r.md").write_text(body)
    frames, durable = await _drain("", [], [ANSWER], tools=[_nav_row("https://ex.org/a"), _write_row("r.md", body)],
                                   sandbox=tmp_path)
    view = _client_view(frames)
    assert "_Source check: 1 link" in view and "never.example" in view
    assert "_Source check: 1 link" in durable


@pytest.mark.asyncio
async def test_the_record_does_not_drop_what_the_stream_kept():
    """Battery P2: once the prefix split decided, the record must not apply
    its own drop — a short note kept in the stream before a long final would
    otherwise be dropped from the record only."""
    ans = "The balance is 6000.00 on both sides."
    frames, durable = await _drain(ans + "\n\n", [ans], [ANSWER])
    view = _client_view(frames)
    assert view.startswith(ans) and ans in durable and durable.strip() == view.strip()


