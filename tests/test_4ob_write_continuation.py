"""§4OB (2026-10-12): a write the output limit cut is continued by the
SYSTEM, round by round, until the file is complete. Every test names the
world it fails in."""
import asyncio
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.core import write_continuation as W
from ghost_agent.tools import file_system as FS
from ghost_agent.utils.logging import request_id_context


class FakeLLM:
    """Scripted replies: each item is (content, finish_reason)."""
    def __init__(self, replies):
        self.replies, self.payloads = list(replies), []

    async def chat_completion(self, payload, timeout=None, task_label="", side_main=False):
        self.payloads.append(payload)
        if not self.replies:
            return {"choices": [{"message": {"content": W.END_MARKER}, "finish_reason": "stop"}]}
        content, finish = self.replies.pop(0)
        return {"choices": [{"message": {"content": content}, "finish_reason": finish}]}


@pytest.fixture
def ctx(tmp_path, request):
    sb = tmp_path / "sandbox"
    sb.mkdir()
    tok = request_id_context.set(f"req-4ob-{request.node.name[:30]}")
    yield SimpleNamespace(sandbox_dir=sb, current_project_id=None,
                          args=SimpleNamespace(model="m", temperature=0.6), llm_client=None)
    request_id_context.reset(tok)
    FS._IN_PROGRESS.clear()


HEAD = "#!/usr/bin/env python3\nfacts = [\n"


def _saved_part(ctx, path, body, more_parts=True):
    """The salvaged first part, as the tool writes it (in progress)."""
    return FS.tool_file_system(operation="write", sandbox_dir=ctx.sandbox_dir, path=path,
                               content=body, more_parts=more_parts, cut_off_saved=70000)


def _run(ctx, path, body, llm):
    ctx.llm_client = llm
    return asyncio.run(W.continue_cut_write(ctx, path, _saved_part(ctx, path, body), ask="write 20 facts"))


def test_a_cut_write_is_continued_until_the_marker_and_syntax_checked(ctx):
    """Fails where the model was handed the cut note and had to build the
    append calls itself (probe-oa-facts: 21 min, six rewrites)."""
    body = HEAD + "".join(f'    "fact {i}",\n' for i in range(8))
    llm = FakeLLM([("".join(f'    "fact {i}",\n' for i in range(8, 16)), "length"),      # cut again
                   ("".join(f'    "fact {i}",\n' for i in range(16, 20)) + "]\nprint(len(facts))\n"
                    + W.END_MARKER + "\n", "stop")])
    r = _run(ctx, "g.py", body, llm)
    text = (ctx.sandbox_dir / "g.py").read_text()
    assert text.count('"fact') == 20 and text.rstrip().endswith("print(len(facts))")
    assert "in 3 parts" in str(r) and "continued automatically" in str(r) and "SYNTAX" not in str(r)
    assert str(r).startswith("SUCCESS: Wrote ") and " chars to 'g.py'. Script-side path (from sandbox cwd): 'g.py'." in str(r)
    assert not FS._is_in_progress(ctx.sandbox_dir / "g.py")
    p = llm.payloads[0]
    assert p["max_tokens"] == W.CONTINUATION_MAX_TOKENS and "tools" not in p
    assert p["chat_template_kwargs"]["enable_thinking"] is False
    assert '"fact 7",' in p["messages"][1]["content"]                 # the tail was shown


def test_a_round_that_starts_the_file_over_is_discarded(ctx):
    body = HEAD + '    "fact 0",\n'
    llm = FakeLLM([(HEAD.replace("python3", "python") + '    "fact A",\n', "stop"),   # a restart: dropped
                   ('    "fact 1",\n]\n' + W.END_MARKER, "stop")])
    r = _run(ctx, "g.py", body, llm)
    text = (ctx.sandbox_dir / "g.py").read_text()
    assert text.count("#!/usr/bin/env") == 1 and text.count('"fact 1"') == 1 and "complete" in str(r)
    assert "discarded" in llm.payloads[1]["messages"][1]["content"] or "started the file over" in \
        llm.payloads[1]["messages"][1]["content"]


def test_two_bad_rounds_abort_and_leave_the_file_in_progress(ctx):
    body = HEAD + '    "fact 0",\n'
    llm = FakeLLM([(HEAD + "x\n", "stop"), (HEAD + "y\n", "stop"), ("never asked", "stop")])
    r = _run(ctx, "g.py", body, llm)
    assert len(llm.payloads) == 2 and (ctx.sandbox_dir / "g.py").read_text() == body
    assert "CUT OFF" in str(r) and FS._is_in_progress(ctx.sandbox_dir / "g.py")


def test_rounds_run_out_leave_the_file_in_progress_with_the_note(ctx):
    body = HEAD + '    "fact 0",\n'
    llm = FakeLLM([("".join(f'    "fact {r}-{i}",\n' for i in range(5)), "length")
                   for r in range(W.CUT_WRITE_CONTINUATIONS + 2)])
    r = _run(ctx, "g.py", body, llm)
    assert len(llm.payloads) == W.CUT_WRITE_CONTINUATIONS
    assert "CUT OFF" in str(r) and FS._is_in_progress(ctx.sandbox_dir / "g.py")
    # review: the note is rebuilt from the file as it NOW ends, not the first part's end
    assert f"fact {W.CUT_WRITE_CONTINUATIONS - 1}-4" in str(r) and "6 more part(s)" in str(r)
    assert str(r).startswith("SUCCESS: Wrote ") and not getattr(r, "is_failure", False)   # an OK result: no strike
    assert (ctx.sandbox_dir / "g.py").read_text().count('"fact') == 1 + 5 * W.CUT_WRITE_CONTINUATIONS


def test_a_refused_first_part_is_not_continued(ctx):
    """The guards still rule: when the tool refused the cut write, nothing
    is continued and the model is never called."""
    (ctx.sandbox_dir / "owner.html").write_text("<!DOCTYPE html>\n<html><body>owner</body></html>\n")
    llm = FakeLLM([("x", "stop")])
    ctx.llm_client = llm
    inner = FS.tool_file_system(operation="write", sandbox_dir=ctx.sandbox_dir, path="owner.html",
                                content="<!DOCTYPE html>\n<html>\n" + "x\n" * 900, more_parts=True,
                                cut_off_saved=70000)
    r = asyncio.run(W.continue_cut_write(ctx, "owner.html", inner, ask="x"))
    assert getattr(r, "reason_code", None) == "tool_call_cut_off" and llm.payloads == []


def test_fences_chatter_tail_repeats_and_the_marker_alone(ctx):
    body = HEAD + '    "fact 0",\n    "fact 1",\n'
    llm = FakeLLM([("```python\n" + '    "fact 1",\n    "fact 2",\n```\n', "stop"),   # fence + tail repeat
                   ("Here is the rest of the file:\n]\n", "stop"),                      # prose: dropped
                   ("]\n" + W.END_MARKER, "stop")])
    r = _run(ctx, "g.py", body, llm)
    text = (ctx.sandbox_dir / "g.py").read_text()
    assert text == body + '    "fact 2",\n]\n' and "complete" in str(r)
    # the marker alone: §4SH r8 — the saved part was CUT after its last whole line, so
    # "done" with nothing added is not proof; the file stays as saved, still open
    body2 = "x = 1\n"
    llm2 = FakeLLM([(W.END_MARKER, "stop")])
    r2 = _run(ctx, "h.py", body2, llm2)
    assert "is complete" not in str(r2) and (ctx.sandbox_dir / "h.py").read_text() == body2


def test_a_syntax_error_at_the_end_is_reported_as_partial(ctx):
    body = HEAD + '    "fact 0",\n'
    llm = FakeLLM([("]\nprint(\n" + W.END_MARKER, "stop")])
    r = _run(ctx, "g.py", body, llm)
    assert getattr(r, "reason_code", None) == "written_but_syntax_failed" and "SYNTAX CHECK FAILED" in str(r)
    assert "continued automatically" in str(r)


def test_the_time_budget_stops_the_rounds(ctx, monkeypatch):
    monkeypatch.setattr(W, "_time_left", lambda rid: False)
    llm = FakeLLM([("y\n" + W.END_MARKER, "stop")])
    r = _run(ctx, "g.py", HEAD + '    "fact 0",\n', llm)
    assert llm.payloads == [] and "CUT OFF" in str(r)


def test_the_stats_note_tells_the_model_how_many_items_it_has(ctx):
    body = HEAD + "".join(f'    "a distinct ocean fact number {i} here",\n' for i in range(30))
    llm = FakeLLM([("]\n" + W.END_MARKER, "stop")])
    ctx.llm_client = llm
    asyncio.run(W.continue_cut_write(ctx, "g.py", _saved_part(ctx, "g.py", body),
                                     ask="a page with 300 facts about the ocean"))
    u = llm.payloads[0]["messages"][1]["content"]
    assert "30 list items" in u and "numbers in the request: 300" in u


def test_the_dispatcher_continues_only_a_salvaged_cut_write():
    """Enumerated over the parsed source: the wrap sits right after the
    tool coroutine is built, gated on the out-of-band cut marker and a
    write operation."""
    import ghost_agent.core.agent as A
    tree = ast.parse(Path(A.__file__).read_text(encoding="utf-8"))
    hits = [n for n in ast.walk(tree) if isinstance(n, ast.If)
            and "_cut_off_saved" in ast.unparse(n.test) and "'write'" in ast.unparse(n.test)
            and any(isinstance(c, ast.Call) and getattr(c.func, "id", "") == "continue_cut_write"
                    for c in ast.walk(n))]
    assert len(hits) == 1
    assigns = [n for n in ast.walk(hits[0]) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
               and getattr(n.value.func, "id", "") == "continue_cut_write"]
    assert len(assigns) == 1 and [ast.unparse(t) for t in assigns[0].targets] == ["_coro"]   # it REPLACES the coroutine
    call = assigns[0].value
    assert ast.unparse(call.args[2]) == "_coro" and any(k.arg == "ask" for k in call.keywords)


def test_an_identical_re_emission_of_the_file_is_repaired_not_lost(ctx):
    """A round that re-sends the whole file verbatim and then continues:
    the repeated part is stripped and the new lines are kept."""
    body = HEAD + '    "fact 0",\n'
    llm = FakeLLM([(body + '    "fact 1",\n', "stop"), ("]\n" + W.END_MARKER, "stop")])
    r = _run(ctx, "g.py", body, llm)
    text = (ctx.sandbox_dir / "g.py").read_text()
    assert text == body + '    "fact 1",\n]\n' and "in 3 parts" in str(r)


def test_a_round_cut_mid_line_is_trimmed_to_its_last_complete_line(ctx):
    """Fails where a round the output limit cut mid-line was appended with
    its half line, which the next round then continued from a wrong tail."""
    body = HEAD + '    "fact 0",\n'
    llm = FakeLLM([('    "fact 1",\n    "fact 2",\n    "fac', "length"),
                   ('    "fact 2",\n]\n' + W.END_MARKER, "stop")])
    _run(ctx, "g.py", body, llm)
    text = (ctx.sandbox_dir / "g.py").read_text()
    assert '"fac\n' not in text and text.count('"fact 2"') == 1 and text.endswith("]\n")


def test_the_model_is_shown_the_files_real_tail(ctx):
    """Fails where only the head was shown: on a long file the model could
    not know where to continue (the head is the first 1,200 chars)."""
    body = HEAD + "".join(f'    "a long enough fact number {i} to fill the file",\n' for i in range(120))
    llm = FakeLLM([("]\n" + W.END_MARKER, "stop")])
    _run(ctx, "g.py", body, llm)
    u = llm.payloads[0]["messages"][1]["content"]
    assert "fact number 119" in u.split("ENDS with")[1]
    assert "fact number 119" not in u.split("ENDS with")[0]      # the head alone did not carry it


def test_the_verifiers_file_ledger_parses_the_continued_result(ctx):
    """Fails where the result said "Wrote N bytes … in K parts" — a shape the
    write-result parsers (FILE-ARTIFACT, the ledgers) do not read, so the
    largest files dropped out of verification (review CRITICAL)."""
    from ghost_agent.core.agent import _fs_path_ledger
    body = HEAD + '    "fact 0",\n'
    llm = FakeLLM([("]\n" + W.END_MARKER, "stop")])
    r = _run(ctx, "g.py", body, llm)
    produced, _ = _fs_path_ledger([{"name": "file_system", "content": str(r)}])
    assert "g.py" in produced


def test_a_lone_closing_brace_the_file_repeats_is_kept(ctx):
    """Fails where the tail stripper dropped a `}` or `</div>` that the
    continuation legitimately starts with (review MAJOR)."""
    assert W.strip_tail_overlap("  }\n}", "}\n\nfunction b(){\n") == "}\n\nfunction b(){\n"
    assert W.strip_tail_overlap("</div>\n</div>\n</body>", "</div>\n</body>\n</html>\n") == "</div>\n</body>\n</html>\n"
    # a substantive repeat of the shown tail IS stripped
    tail = "".join(f'    "a long fact line number {i} in the file",\n' for i in range(10))
    cont = "".join(f'    "a long fact line number {i} in the file",\n' for i in range(4, 10)) + '    "new",\n'
    assert W.strip_tail_overlap(tail, cont) == '    "new",\n'


def test_the_finished_file_is_syntax_checked_when_the_marker_comes_alone(ctx):
    """Fails where round 1 (no marker, more_parts) was never checked and the
    marker-only round 2 closed the file without a check (review MAJOR)."""
    body = HEAD + '    "fact 0",\n'
    llm = FakeLLM([("]\nprint(\n", "stop"), (W.END_MARKER, "stop")])
    r = _run(ctx, "g.py", body, llm)
    assert getattr(r, "reason_code", None) == "written_but_syntax_failed" and "SYNTAX CHECK FAILED" in str(r)


def test_content_that_merely_starts_like_prose_is_kept():
    assert W.is_chatter("The remaining items are listed below.\n- a\n- b\n" * 20) is False
    assert W.is_chatter("Here is the rest of the file:\n]\n") is True
    assert W.is_chatter("Certainly not, she said. " * 30) is False


def test_a_fence_the_file_itself_needs_is_kept():
    kept, _ = W.clean_round("```python\nprint(1)\n```\n\nMore prose after the example.\n")
    assert kept.startswith("```python")                          # not wholly wrapped: content
    stripped, _ = W.clean_round("```js\nconst a = 1;\n```\n")
    assert stripped == "const a = 1;\n"


def test_an_empty_reply_after_a_stop_round_ends_the_file(ctx):
    body = HEAD + '    "fact 0",\n'
    llm = FakeLLM([("]\n", "stop"), ("", "stop")])
    r = _run(ctx, "g.py", body, llm)
    assert "complete" in str(r) and (ctx.sandbox_dir / "g.py").read_text() == body + "]\n"


def test_the_total_wall_clock_bounds_the_rounds(ctx, monkeypatch):
    monkeypatch.setattr(W, "CONTINUATION_TOTAL_S", 0.0)
    llm = FakeLLM([("]\n" + W.END_MARKER, "stop")])
    r = _run(ctx, "g.py", HEAD + '    "fact 0",\n', llm)
    assert llm.payloads == [] and "CUT OFF" in str(r)


def test_a_short_distinct_line_the_file_repeats_is_kept():
    """`end` / `done` lines that a file repeats are content, not an echo."""
    assert W.strip_tail_overlap("x = 1\nend", "end\nnext\n") == "end\nnext\n"
    assert W.strip_tail_overlap("x = 1\n    \"a fact line\",", "    \"a fact line\",\nnext\n") == "next\n"
    # review A: a line the shown tail itself repeats is boilerplate the file keeps, not an echo
    tail = "    self.assertTrue(ok)\n    x = 1\n    self.assertTrue(ok)"
    assert W.strip_tail_overlap(tail, "    self.assertTrue(ok)\n    y = 2\n") == "    self.assertTrue(ok)\n    y = 2\n"


def test_long_content_that_opens_like_chatter_is_kept():
    long_prose = "Here is the story of the harbour, told at length " * 10 + "\nMore lines follow.\n"
    assert W.is_chatter(long_prose) is False                 # no colon, not short
    assert W.is_chatter("Here is the remaining code:\n" + "x = 1\n" * 100) is True


def test_an_empty_reply_after_a_stop_round_asks_no_further(ctx):
    body = HEAD + '    "fact 0",\n'
    llm = FakeLLM([("]\n", "stop"), ("", "stop"), ("never", "stop")])
    _run(ctx, "g.py", body, llm)
    assert len(llm.payloads) == 2


# ── review round 2 ───────────────────────────────────────────────────────── #

def test_a_give_up_is_not_booked_as_a_failure(ctx):
    """Fails where the give-up result was a PARTIAL outcome — a strike for
    the arm that saved MORE of the file, while the plain cut-save (an OK
    string + note) got none (review MAJOR)."""
    body = HEAD + '    "fact 0",\n'
    llm = FakeLLM([("".join(f'    "fact {r}-{i}",\n' for i in range(5)), "length")
                   for r in range(W.CUT_WRITE_CONTINUATIONS + 2)])
    r = _run(ctx, "g.py", body, llm)
    assert not getattr(r, "is_failure", False) and str(r).startswith("SUCCESS: Wrote ")
    assert "IN PROGRESS" in str(r) and FS._is_in_progress(ctx.sandbox_dir / "g.py")


def test_a_stop_press_ends_the_rounds(ctx):
    """Fails where a cooperative Stop was seen only at the turn boundary and
    up to 6 rounds / 10 min of continuation kept running (review MAJOR)."""
    body = HEAD + '    "fact 0",\n'
    llm = FakeLLM([("    \"fact 1\",\n", "length")] * 5)
    ctx.llm_client = llm
    calls = {"n": 0}

    def stop_after_one():
        calls["n"] += 1
        return calls["n"] > 1
    r = asyncio.run(W.continue_cut_write(ctx, "g.py", _saved_part(ctx, "g.py", body), ask="x",
                                         cancelled=stop_after_one))
    assert len(llm.payloads) == 1 and "cancelled" in str(r)


def test_a_python_or_markdown_round_that_restarts_from_line_one_is_dropped(ctx):
    """Fails where only HTML/XML/#! restarts were caught — a .py restart was
    appended whole, duplicating the file (review MAJOR)."""
    body = "import os\nimport sys\n\nfacts = [\n    \"fact 0\",\n"
    llm = FakeLLM([("import os\nimport sys\n\nfacts = [\n    \"fact 0\",\n    \"fact 1\",\n]\n", "stop"),
                   ("    \"fact 1\",\n]\n" + W.END_MARKER, "stop")])
    r = _run(ctx, "m.py", body, llm)
    text = (ctx.sandbox_dir / "m.py").read_text()
    assert text.count("import os") == 1 and text == body + '    "fact 1",\n]\n' and "complete" in str(r)


def test_a_long_round_with_a_chatter_opener_loses_only_that_line(ctx):
    cont, discarded = W.drop_chatter("Here is the rest of the file:\n" + "line\n" * 100)
    assert not discarded and cont.startswith("line\n")
    cont, discarded = W.drop_chatter("Here is the breakdown of costs:\n" + "item\n" * 100)
    assert not discarded and cont.startswith("item\n")          # a colon-ended opener: dropped line only
    cont, discarded = W.drop_chatter("Here is the story of the harbour " * 10 + "\n" + "line\n" * 50)
    assert not discarded and cont.startswith("Here is the story")  # content: kept whole
    assert W.drop_chatter("Here is the rest:\n]\n")[1] is True   # short: discarded


def test_a_re_emitted_end_with_the_marker_completes_the_file(ctx):
    """Fails where a duplicate-append refusal of the re-emitted closers
    (all structural, never stripped) plus the marker counted as a bad round
    twice and gave up on a complete file (review)."""
    closers = "".join("</div>\n" for _ in range(40)) + "</body>\n</html>\n"
    body = "<!DOCTYPE html>\n<html>\n<body>\n" + "".join(f"<div>{i}</div>\n" for i in range(40)) + closers
    for name in ("p.html", "p.xml"):                   # §4SH r5: a judged kind and an unjudged one
        llm = FakeLLM([(closers + W.END_MARKER, "stop")])
        r = _run(ctx, name, body, llm)
        assert "complete" in str(r) and (ctx.sandbox_dir / name).read_text() == body
        assert len(llm.payloads) == 1                  # the refusal + marker ended it; no second round


def test_the_dispatcher_passes_a_real_cancel_check():
    import ghost_agent.core.agent as A
    tree = ast.parse(Path(A.__file__).read_text(encoding="utf-8"))
    call = next(c for c in ast.walk(tree) if isinstance(c, ast.Call)
                and getattr(c.func, "id", "") == "continue_cut_write")
    kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
    assert "is_cancelled" in kw.get("cancelled", "")


def test_a_long_python_file_restarted_from_line_one_is_dropped(ctx):
    """The tail no longer contains the head (the file is longer than the
    shown tail), so only the head rule can catch a restart."""
    body = "import os\nimport sys\n\nfacts = [\n" + "".join(f'    "a long fact line number {i} here",\n' for i in range(80))
    llm = FakeLLM([("import os\nimport sys\n\nfacts = [\n    \"fresh start\",\n]\n", "stop"),
                   ("]\n" + W.END_MARKER, "stop")])
    r = _run(ctx, "m.py", body, llm)
    text = (ctx.sandbox_dir / "m.py").read_text()
    assert text.count("import os") == 1 and "fresh start" not in text and text == body + "]\n"
    assert "complete" in str(r)


# ── probe-ob: over-generation and repetition ─────────────────────────────── #

def _facts(n, start=0, word="ocean"):
    return "".join(f'    "A distinct {word} fact number {i} written out here.",\n' for i in range(start, start + n))


def test_a_file_that_already_has_the_asked_items_is_only_closed(ctx):
    """Fails where the saved part held ~1,000 facts for a request of 300
    and six continuation rounds added 170 KB more (probe-ob 18:08)."""
    body = HEAD + _facts(320)
    llm = FakeLLM([(_facts(40, 320), "length"),                          # kept going: discarded
                   ("]\nprint(len(facts))\n" + W.END_MARKER, "stop")])
    ctx.llm_client = llm
    r = asyncio.run(W.continue_cut_write(ctx, "g.py", _saved_part(ctx, "g.py", body),
                                         ask="a page with 300 facts about the ocean"))
    u = llm.payloads[0]["messages"][1]["content"]
    assert "already holds 320 list items" in u and "Do NOT add any more items" in u
    # probe 4: a close round thinks (no enable_thinking=False) under a short cap; a normal round does not
    p0 = llm.payloads[0]
    assert (p0.get("chat_template_kwargs") or {}).get("enable_thinking") is not False
    assert p0["max_tokens"] == W.CLOSE_ROUND_MAX_TOKENS < W.CONTINUATION_MAX_TOKENS
    text = (ctx.sandbox_dir / "g.py").read_text()
    assert text.count("fact number") == 320 and text.rstrip().endswith("print(len(facts))") and "complete" in str(r)


def test_close_mode_allows_a_few_items_but_not_more(ctx):
    """Round 4: a closing round may carry a short footer list (a nav of 4-8
    links) — `_CLOSE_MODE_MAX_ITEMS` — but not a continued enumeration."""
    body = HEAD + _facts(320)
    llm = FakeLLM([(_facts(8, 320) + "]\nprint(len(facts))\n" + W.END_MARKER, "stop")])   # 8: allowed
    ctx.llm_client = llm
    r = asyncio.run(W.continue_cut_write(ctx, "g.py", _saved_part(ctx, "g.py", body),
                                         ask="a page with 300 facts about the ocean"))
    text = (ctx.sandbox_dir / "g.py").read_text()
    assert text.count("fact number") == 328 and len(llm.payloads) == 1 and "complete" in str(r)


def test_an_empty_round_cut_inside_its_thinking_is_not_the_end(ctx):
    """Round 8: after a stop-round, an EMPTY reply means "nothing to add" —
    unless the cap cut it (finish "length"): a close round that spent its
    4k tokens thinking must be retried, not booked as complete."""
    body = HEAD + _facts(320)
    llm = FakeLLM([(_facts(2, 320), "stop"),                       # a stop-round without the marker
                   ("", "length"),                                  # cap cut inside the thinking
                   ("]\nprint(len(facts))\n" + W.END_MARKER, "stop")])
    ctx.llm_client = llm
    r = asyncio.run(W.continue_cut_write(ctx, "g.py", _saved_part(ctx, "g.py", body),
                                         ask="a page with 300 facts about the ocean"))
    text = (ctx.sandbox_dir / "g.py").read_text()
    assert text.rstrip().endswith("print(len(facts))") and len(llm.payloads) == 3 and "complete" in str(r)


def test_a_close_round_that_kept_listing_is_salvaged_not_discarded(ctx):
    """Fails where probe 3 (500 facts, 1,652 already saved) had both
    close-only rounds discarded for adding items and the file stayed
    unclosed (`'[' was never closed`): the items are dropped, the closers
    and the remaining code are kept."""
    body = HEAD + _facts(320)
    llm = FakeLLM([(_facts(40, 320) + "]\nprint(len(facts))\n" + W.END_MARKER, "stop")])
    ctx.llm_client = llm
    r = asyncio.run(W.continue_cut_write(ctx, "g.py", _saved_part(ctx, "g.py", body),
                                         ask="a page with 300 facts about the ocean"))
    text = (ctx.sandbox_dir / "g.py").read_text()
    assert text.count("fact number") == 320 and text.rstrip().endswith("print(len(facts))")
    assert len(llm.payloads) == 1 and "complete" in str(r)
    assert W.drop_list_items(_facts(6) + "]\n") == ("]\n", 6)
    assert W.drop_list_items(_facts(6)) == ("", 6)                      # nothing to keep → still a discard
    five = _facts(5) + "]\n"
    assert W.drop_list_items(five) == (five, 0)                         # review A: a 5-string list is code


def _dicts(n, start=0):
    return "".join(f'    {{\n        "name": "Item number {i} long name",\n        "category": "tools",\n'
                   f'        "description": "A sentence describing item {i} in full.",\n    }},\n'
                   for i in range(start, start + n))


def test_a_dict_close_round_loses_its_dicts_and_keeps_the_code(ctx):
    """Fails where a close round of 30 more dicts had no item "run" (dict
    lines never matched the item regex) and was appended whole — close mode
    armed but never bit for a list of dicts (fresh-eye review, 21:10)."""
    body = "CATALOG = [\n" + _dicts(420)
    closing = "]\n\ndef find(term):\n    return [e for e in CATALOG if term in e['name']]\n" + W.END_MARKER
    llm = FakeLLM([(_dicts(30, 420) + closing, "stop")])
    ctx.llm_client = llm
    r = asyncio.run(W.continue_cut_write(ctx, "c.py", _saved_part(ctx, "c.py", body),
                                         ask="a module with a list of exactly 400 dicts"))
    text = (ctx.sandbox_dir / "c.py").read_text()
    assert text.count('"name":') == 420 and "def find(term)" in text and "complete" in str(r)
    assert "Do NOT add any more items" in llm.payloads[0]["messages"][1]["content"]
    # a one-line object list is counted and salvaged too
    from ghost_agent.core.agent import list_item_stats
    ones = "".join(f'    {{"name": "item number {i}", "qty": {i}}},\n' for i in range(60))
    assert len(list_item_stats(ones, "a list of 40 items")[0]) == 60
    assert W.drop_list_items(ones + "]\n") == ("]\n", 60)
    # five dicts are not a listing run; six are (counted as units, not lines)
    assert W.drop_list_items(_dicts(5) + "]\n")[1] == 0 and W.drop_list_items(_dicts(6) + "]\n") == ("]\n", 6)
    # nested dicts stay one entry each (review: the scan used to stop at the inner `},`)
    nested = "".join(f'    {{\n        "name": "Item number {i} long name",\n        "meta": {{\n'
                     f'            "source": "catalogue {i}",\n        }},\n'
                     f'        "description": "A sentence describing item {i} in full.",\n    }},\n' for i in range(8))
    assert W.drop_list_items(nested + "]\n") == ("]\n", 8)
    assert len(list_item_stats(nested, "a list of 5 records")[0]) == 8


def test_a_close_round_whose_items_form_no_run_is_kept_whole(ctx):
    """r2: nine quoted lines interleaved with code (a `# note` after each) are
    not a listing run; the round is closing code and is appended whole."""
    body = HEAD + _facts(320)
    closing = "]\nEXTRA = [\n" + "".join(f'    "an extra labelled string number {i} here",\n    # note {i}\n'
                                           for i in range(9)) + "]\nprint(len(facts), len(EXTRA))\n" + W.END_MARKER
    llm = FakeLLM([(closing, "stop")])
    ctx.llm_client = llm
    r = asyncio.run(W.continue_cut_write(ctx, "g.py", _saved_part(ctx, "g.py", body),
                                         ask="a page with 300 facts about the ocean"))
    text = (ctx.sandbox_dir / "g.py").read_text()
    assert text.count("extra labelled string") == 9 and text.rstrip().endswith("print(len(facts), len(EXTRA))")
    assert len(llm.payloads) == 1 and "complete" in str(r)


def test_a_close_round_of_dict_entries_is_closing_code_not_a_listing(ctx):
    """A close-mode round whose quoted lines form no listing run (a 12-entry
    dict the writer needs) is appended whole — `list_item_stats` counts
    them, `drop_list_items` finds nothing to drop, and discarding it left
    the file unclosed."""
    body = HEAD + _facts(320)
    closing = "]\nlabels = {\n" + "".join(f'    "label number {i}": "a label text {i}",\n' for i in range(12)) \
              + "}\nprint(len(facts), len(labels))\n" + W.END_MARKER
    llm = FakeLLM([(closing, "stop")])
    ctx.llm_client = llm
    r = asyncio.run(W.continue_cut_write(ctx, "g.py", _saved_part(ctx, "g.py", body),
                                         ask="a page with 300 facts about the ocean"))
    text = (ctx.sandbox_dir / "g.py").read_text()
    assert text.count("label number") == 12 and text.rstrip().endswith("print(len(facts), len(labels))")
    assert len(llm.payloads) == 1 and "complete" in str(r)


def test_salvage_keeps_dict_entries_docstrings_and_lone_quoted_lines():
    """Round 6: the salvage must not eat the closing code's own quoted
    lines — dict entries, a one-line docstring, a lone string — only RUNS
    of listed items go."""
    closing = (']\n\nmeta = {\n    "title": "Facts about the ocean page",\n    "author": "ghost",\n'
               '    "lang": "en-GB",\n}\n\ndef write():\n    """Write the page to disk."""\n'
               '    tpl = "<li>{fact}</li> rendered here",\n    open("out.html", "w").write(tpl)\n')
    text, dropped = W.drop_list_items(_facts(12) + closing)
    assert dropped == 12 and text == closing
    # seven dict entries in a row are still a dict, not a listing run
    seven = "cfg = {\n" + "".join(f'    "config key number {i}": "value number {i}",\n' for i in range(7)) + "}\n"
    assert W.drop_list_items(_facts(12) + "]\n" + seven) == ("]\n" + seven, 12)
    two = _facts(2) + "]\n"
    assert W.drop_list_items(two) == (two, 0)                           # two items are not a listing
    # a one-line docstring right after five trailing items is not the sixth item of a listing
    tail = _facts(5) + '"""Facts above; the writer follows."""\n' + "]\n"
    assert W.drop_list_items(tail) == (tail, 0)
    # round 7: blank lines inside a listing do not split it into non-runs
    assert W.drop_list_items(_facts(2) + "\n" + _facts(2, 2) + "\n" + _facts(2, 4) + "]\n") == ("]\n", 6)


def test_a_round_of_repeated_lines_is_discarded_and_two_abort(ctx):
    """Fails where a no-think round echoing one fact thousands of times was
    appended (probe-ob: 6,733 quoted strings, "Fish are eaten by
    oystercatchers." over and over)."""
    body = HEAD + _facts(30)
    echo = '    "Fish are eaten by oystercatchers.",\n' * 60
    llm = FakeLLM([(echo, "length"), (echo, "length"), ("never", "stop")])
    ctx.llm_client = llm
    # the request names NO count, so close mode stays off and only the repetition check can act
    r = asyncio.run(W.continue_cut_write(ctx, "g.py", _saved_part(ctx, "g.py", body),
                                         ask="a page of ocean facts"))
    assert len(llm.payloads) == 2 and (ctx.sandbox_dir / "g.py").read_text() == body
    assert "CUT OFF" in str(r)
    assert "repeated its end or lines already in it" in llm.payloads[1]["messages"][1]["content"]


def test_repetition_needs_a_real_share_of_repeats():
    assert W.is_repetition("".join(f"line {i}\n" for i in range(40)), set()) is False
    assert W.is_repetition("the same line\n" * 12, set()) is True
    old_text = "".join(f"an old line of the file number {i}\n" for i in range(40))
    assert W.is_repetition(old_text, old_text) is True                  # the file's own block re-emitted
    assert W.is_repetition("".join(f"an old line of the file number {i}\n" for i in range(5)), old_text) is False  # too short
    shuffled = "".join(f"an old line of the file number {i}\n" for i in range(39, -1, -1))
    assert W.is_repetition(shuffled, old_text) is False                 # the same LINES, not the same block
    assert W.is_repetition("the same line\nthe same line\nthe same line\n", set()) is False   # too short to judge


def test_honest_repeats_are_not_repetition():
    """Round 3: a test file whose cases share `assert r.ok`, idiom lines that
    stand in the file (`return None`, `except Exception:`) and a refrain are
    legitimate — only a RUN of identical lines or copied long lines count."""
    tests = "".join(f"def test_case_{i}():\n    r = run({i})\n    assert r.ok\n    assert r.err is None\n\n"
                    for i in range(12))
    assert W.is_repetition(tests, set()) is False
    file_lines = {"return None", "pass", "else:", "except Exception:  # noqa: BLE001", "fi", "done",
                  "if x:", "try:"}
    code = "".join(f"def f{i}(x):\n    if x:\n        return None\n    try:\n        go({i})\n"
                   f"    except Exception:  # noqa: BLE001\n        pass\n" for i in range(12))
    assert W.is_repetition(code, file_lines) is False     # 5 of 7 lines per function stand in the file
    # round 4: ten long lines of which half are shared test idioms is an honest continuation
    idioms = {"assert resp.status_code == 200", "monkeypatch.setattr(mod, 'run', fake)"}
    tests2 = "".join(f"def test_endpoint_number_{i}(monkeypatch):\n    monkeypatch.setattr(mod, 'run', fake)\n"
                     f"    resp = client.get('/thing/{i}')\n    assert resp.status_code == 200\n\n" for i in range(6))
    assert W.is_repetition(tests2, idioms) is False
    # review A: a unittest round whose every line stands somewhere in the file (other
    # tests use the same bodies) is honest — only the SAME 10-line block is a copy
    file_text = "".join(f"def test_old_{i}(self):\n    response = self.client.get('/old/{i}')\n"
                        f"    self.assertEqual(response.status_code, 200)\n    self.assertTrue(response.ok)\n\n"
                        for i in range(12))
    round_text = "".join(f"def test_new_{i}(self):\n    response = self.client.get('/new/{i}')\n"
                         f"    self.assertEqual(response.status_code, 200)\n    self.assertTrue(response.ok)\n\n"
                         for i in range(12))
    assert W.is_repetition(round_text, file_text) is False
    assert W.is_repetition(file_text, file_text) is True
    # r2: structural wrappers must not hide a copied block — the filter runs on both sides
    wrapped = "".join(f'<div class="f">\n  <li>A copied ocean fact number {i} here.</li>\n</div>\n' for i in range(12))
    bare = "".join(f"  <li>A copied ocean fact number {i} here.</li>\n" for i in range(12))
    assert W.is_repetition(bare, wrapped) is True
    # a block of trivially short lines (`x=1`, `y=2`…) is idiom, not a copied block
    short = "".join(f"v{i}=1\n" for i in range(12))
    assert W.is_repetition(short, "HEAD\n" + short + "TAIL\n") is False
    assert W.is_repetition("</section>\n" * 6 + "</main>\n" * 4, set()) is False   # closers, structural
    assert W.is_repetition("|---|---|---|\n" * 4 + "# -----\n" * 4, set()) is False
    assert W.is_repetition("    pass\n" * 6 + "done\n" * 5, set()) is False   # short idioms never make a run
    poem = "".join(f"verse line number {i} about the sea\nand the tide comes in again\n" for i in range(10))
    assert W.is_repetition(poem, set()) is False
    # a stuck model: the same long line four times in a row, anywhere in the round
    stuck = "".join(f"a fresh line {i}\n" for i in range(20)) + '    "Fish are eaten by oystercatchers.",\n' * 6
    assert W.is_repetition(stuck, set()) is True
    assert W.is_repetition("".join(f"a fresh line {i}\n" for i in range(20)) + "(NULL, NULL, NULL),\n" * 5, set()) is False
    assert W.is_repetition("<tr>\n" + "<td></td>\n" * 8 + "</tr>\n", set()) is False     # tags-only lines are structural
    # copying the file: a long round of long lines that already stand in it
    old = "".join(f"a long copied fact line number {i} of the file\n" for i in range(30))
    assert W.is_repetition(old + "fresh\n", "HEAD\n" + old + "TAIL\n") is True
    # round 4: alternating two or three lines is as stuck as repeating one
    abab = "".join(f"fresh line {i}\n" for i in range(10)) + "    \"Fish are eaten by gulls.\",\n    \"Gulls eat fish.\",\n" * 4
    assert W.is_repetition(abab, set()) is True
    abc = ("    \"Alpha fact one.\",\n    \"Beta fact two.\",\n    \"Gamma fact three.\",\n") * 4
    assert W.is_repetition(abc, set()) is True
    assert W.is_repetition(("    \"Alpha fact one.\",\n    \"Beta fact two.\",\n") * 2, set()) is False


def test_unit_numbers_are_not_item_counts():
    """Round 3: "under 500 words", "last 30 days", "retry 10 times", "age 25
    to 80" name no item count; "300 facts" and "top 100 movies of 1999" do."""
    from ghost_agent.core.agent import list_item_stats
    for ask in ("a story under 500 words", "a 1000 word story", "the last 30 days", "retry 10 times",
                "a 300-line script", "wait 45 seconds", "a 20 page report"):
        assert list_item_stats("", ask)[1] == [], ask
    assert list_item_stats("", "300 facts, 20 per section")[1] == [20, 300]
    assert list_item_stats("", "give me 50 to 60 facts")[1] == [60]         # "50 to" is a range bound, not a count
    assert list_item_stats("", "top 100 movies of 1999")[1] == [100]
    # round 4: a thousands separator and a count that ends the request
    assert list_item_stats("", "a page with 1,000 facts")[1] == [1000]
    assert list_item_stats("", "a page with 2,000 facts")[1] == [2000]     # a comma-written number is not a year
    assert list_item_stats("", "a page with 2000 facts")[1] == []          # ambiguous with a year: left out
    assert list_item_stats("", "make the list between 20 and 30")[1] == [20, 30]


def test_docstrings_are_not_list_items_and_a_minority_list_does_not_close():
    """Review A: a 20-function module with docstrings + "20 subcommands" armed
    close mode on round 1; r2: a 25-string NAMES list in a 200-line module is
    not the file's dominant shape either."""
    from ghost_agent.core.agent import list_item_stats
    module = "".join(f'def cmd_{i}(args):\n    """Run subcommand number {i} of the tool."""\n'
                     f'    value = compute({i})\n    return value\n\n' for i in range(30))
    assert list_item_stats(module, "a CLI with 20 subcommands")[0] == []
    assert W.close_mode_note(module, "a CLI with 20 subcommands") == ""
    names = "NAMES = [\n" + "".join(f'    "subcommand name number {i}",\n' for i in range(25)) + "]\n"
    assert len(list_item_stats(names + module, "a CLI with 25 subcommands")[0]) == 25
    assert W.close_mode_note(names + module, "a CLI with 25 subcommands") == ""     # 25 of ~200 lines
    # r2: an item with a description line and a source line (one line in four) is still the list
    page = "".join(f"<li>A distinct ocean fact number {i} written out here.</li>\n<p>Longer description {i}.</p>\n"
                   f"<span>Source {i}</span>\n<small>note {i}</small>\n" for i in range(300))
    assert W.close_mode_note(page, "a page with 300 facts").startswith("The file already holds 300 list items")


def test_dict_entries_are_not_list_items():
    """A 400-dict list has 1,200 `"key": "value",` lines; counting them as
    items armed close mode at ~134 dicts and told the model to close the
    list early (found by a direct check before the big-file probes)."""
    from ghost_agent.core.agent import list_item_stats
    dicts = "CATALOG = [\n" + "".join(
        f'    {{\n        "name": "Item number {i} long name",\n        "category": "tools and hardware",\n'
        f'        "description": "A sentence describing item {i} in full.",\n    }},\n' for i in range(120))
    items, asked = list_item_stats(dicts, "a module with a list of 400 dicts")
    assert len(items) == 120 and asked == [400]                  # one item per dict, not per key line
    assert W.close_mode_note(dicts, "a module with a list of 400 dicts") == ""
    # probe-big (21:05): 600 dicts for a 400-dict ask ran six rounds to 232 KB and stayed open
    many = "CATALOG = [\n" + "".join(
        f'    {{\n        "name": "Item number {i} long name",\n        "category": "tools",\n'
        f'        "description": "A sentence describing item {i} in full.",\n    }},\n' for i in range(420))
    assert W.close_mode_note(many, "a module with a list of 400 dicts").startswith("The file already holds 420 list items")
    wide = "ROWS = [\n" + "".join(
        "    {\n" + "".join(f'        "field number {k}": "value {i} {k}",\n' for k in range(6)) + "    },\n"
        for i in range(420))
    assert W.close_mode_note(wide, "a list of 400 records").startswith("The file already holds 420")   # 6-key dicts
    # Allman-style code braces are not entries
    allman = "".join(f"void f{i}(void)\n{{\n    run({i});\n    stop({i});\n}}\n" for i in range(40))
    assert list_item_stats(allman, "30 functions")[0] == []


def test_html_list_items_count_as_items():
    """Round 3: close mode was silent for an HTML page whose facts are `<li>`
    lines — the common deliverable."""
    from ghost_agent.core.agent import list_item_stats
    html = "<ul>\n" + "".join(f"  <li>A distinct ocean fact number {i} written out here.</li>\n" for i in range(40)) + "</ul>\n"
    items, asked = list_item_stats(html, "a page with 30 facts")
    assert len(items) == 40 and asked == [30]
    assert W.close_mode_note(html, "a page with 30 facts").startswith("The file already holds 40 list items")


def test_list_item_stats_counts_items_and_asked():
    from ghost_agent.core.agent import list_item_stats
    items, asked = list_item_stats(_facts(25), "give me 300 facts, 12-20 words each, by 2026")
    assert len(items) == 25 and asked == [300]


def test_repeated_closing_tags_are_not_repetition():
    assert W.is_repetition("</div>\n" * 40 + "</body>\n</html>\n", set()) is False
