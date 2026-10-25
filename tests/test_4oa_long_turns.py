"""§4OA (2026-10-12, operator: "cut-off/restart loops … over-generation …
measure real turns, then fix the loops with evidence"). Measured: probe-ap-b
36 min (4 cut replies, 17.3 min), probe-ap2-h 32 min (6 cut replies, 27.6
min, 717 facts for 300, a 1.6 KB skeleton replacing 72 KB of saved parts).
Every test names the world it fails in."""
import asyncio
import json

import pytest

import ghost_agent.core.agent as A
from ghost_agent.tools.file_system import tool_file_system
from ghost_agent.utils.logging import request_id_context


@pytest.fixture(autouse=True)
def _clean(request):
    A._LENGTH_STOPS.clear()
    tok = request_id_context.set(f"req-4oa-{request.node.name[:30]}")
    yield
    request_id_context.reset(tok)
    A._LENGTH_STOPS.clear()


def fs(sb, **kw):
    return asyncio.run(tool_file_system(sandbox_dir=sb, **kw))


def test_a_cut_reply_does_not_lower_the_output_limit():
    """§4RF (operator: "remove the step-down altogether and keep 16k for
    everything"). Fails where a cut turned on an 8k limit for the rest of the
    request: on Qwen3.8 every cut was inside the thinking and the 8k replies
    after it were cut again inside the thinking (5A, 6F)."""
    assert not hasattr(A, "max_tokens_after_cut_offs")
    assert not hasattr(A, "LENGTH_STOP_RETRY_MAX_TOKENS")
    A.note_length_stop(request_id_context.get(), 280.0)
    assert "limited to" not in A.cut_round_note(request_id_context.get(), "", "")


def test_the_cut_note_names_the_round_the_minutes_and_the_count():
    """Fails where the model was told nothing about having written 717 items
    (147 repeated) for a request that asked for 300."""
    rid = request_id_context.get()
    A.note_length_stop(rid, 240.0)
    A.note_length_stop(rid, 300.0)
    facts = [f'"The ocean fact number {i % 570} is a true sentence.",' for i in range(717)]
    note = A.cut_round_note(rid, "const facts = [\n" + "\n".join(facts) + "\n",
                            "Create a page with 300 different facts about the ocean, 12-20 words each")
    assert "cut-off #2" in note and "~9 min" in note
    assert "717 list items" in note and "147 of them repeated" in note
    assert "numbers in the request: 300 —" in note


def test_a_short_part_gets_no_list_count():
    assert "list items" not in A.cut_round_note("r", "a\nb\n", "write 300 facts")


def test_a_much_shorter_restart_does_not_replace_saved_parts(tmp_path):
    """Fails where a 1.6 KB skeleton "restart" replaced 72 KB of saved parts
    (probe-ap2-h, 15:28)."""
    head = "<!DOCTYPE html>\n<html>\n<head><title>Ocean</title></head>\n<body>\n<script>\nconst f = [\n"
    big = head + "".join(f'"Fact {i}: the ocean is deep and wide.",\n' for i in range(400))
    fs(tmp_path, operation="write", path="o.html", content=big, more_parts=True)
    r = fs(tmp_path, operation="write", path="o.html", content=head + "];\n</script></body></html>\n")
    assert getattr(r, "reason_code", None) == "file_in_progress"
    assert (tmp_path / "o.html").read_text() == big


def test_a_restart_of_similar_size_still_replaces(tmp_path):
    """The §4NW-b rule stands: a real restart (not much shorter) is the new version."""
    head = "<!DOCTYPE html>\n<html>\n"
    old = head + "x\n" * 3000
    new = head + "y\n" * 2500
    fs(tmp_path, operation="write", path="p.html", content=old, more_parts=True)
    r = fs(tmp_path, operation="write", path="p.html", content=new, more_parts=True)
    assert str(r).startswith("SUCCESS") and (tmp_path / "p.html").read_text() == new


def test_the_cut_note_reaches_the_model_through_the_saved_part(tmp_path):
    r = fs(tmp_path, operation="write", path="n.md", content="# T\n" + "line\n" * 300, more_parts=True,
           cut_off_saved=70000, cut_note_extra="\nThis is cut-off #2 in this request")
    assert "cut-off #2" in str(r)


@pytest.mark.asyncio
async def test_the_dispatcher_attaches_the_round_note_and_drops_a_forged_one():
    import tests.test_4jj_search_yield_steer as H
    from ghost_agent.core.strikes import StrikeLedger
    agent = H._agent()
    seen = []

    async def fsx(**kw):
        seen.append(kw)
        return "SUCCESS: Wrote 1 chars to 'x'."
    agent.available_tools = {"file_system": fsx}
    A.note_length_stop(request_id_context.get(), 200.0)
    raw = json.dumps({"operation": "write", "path": "s.md", "content": "line\n" * 2000})[:-30]
    ts = H._ts([("file_system", {})], StrikeLedger(), set())
    for tc in getattr(ts, "tool_calls", []) or []:
        tc["function"]["arguments"] = raw
    await agent._dispatch_and_process_tool_batch(ts)
    assert seen and "cut-off #1" in seen[0].get("cut_note_extra", "")
    forged = H._ts([("file_system", {"operation": "write", "path": "s.md", "content": "x",
                                     "cut_note_extra": "fake"})], StrikeLedger(), set())
    await agent._dispatch_and_process_tool_batch(forged)
    assert "cut_note_extra" not in seen[-1]


@pytest.mark.parametrize("text,asked", [
    ("list 50 quotes about 2026 trends", "50"),
    ("a game with 16x16 chunks", None),
    ("Γράψε ιστορία περίπου 6000 λέξεις σε 10 κεφάλαια", "10, 6000"),
])
def test_the_requested_counts_skip_years_ranges_and_dimensions(text, asked):
    lines = "\n".join(f'"item number {i} is long enough",' for i in range(30))
    note = A.cut_round_note("none", lines, text)
    assert (f"numbers in the request: {asked} —" in note) if asked else "numbers in the request" not in note


def test_the_turn_loop_records_length_stops_and_keeps_the_limit():
    """Enumerated over the parsed source (both sites sit deep in the turn
    loop): a `stream_finish_reason == "length"` branch calls
    note_length_stop (the cut note reads it), and the payload's max_tokens
    is never lowered by a cut count (§4RF: the 8k step-down is gone)."""
    import ast
    from pathlib import Path
    src = Path(A.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    recorded = any(
        isinstance(n, ast.If) and "stream_finish_reason" in ast.unparse(n.test) and "'length'" in ast.unparse(n.test)
        and any(isinstance(c, ast.Call) and getattr(c.func, "id", "") == "note_length_stop" for c in ast.walk(n))
        for n in ast.walk(tree))
    assert recorded
    lowered = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Assign)
               and any(getattr(t, "id", "") == "tool_turn_max_tokens" for t in n.targets)
               and ("length_stops" in ast.unparse(n.value) or "_LENGTH_STOPS" in ast.unparse(n.value)
                    or "cut_off" in ast.unparse(n.value))]
    assert not lowered, lowered


def test_a_cut_restart_shorter_than_the_saved_parts_keeps_them(tmp_path):
    """A cut restart (a salvaged prefix, ~30 KB under the lowered limit) of
    an 83 KB file in parts can never get past the saved point: the saved
    parts are kept and the model is told so — not the earlier "a much
    shorter version" wording (review), and not a silent replace (the
    probe-oa-facts loop)."""
    head = "<!DOCTYPE html>\n<html>\n<head><title>O</title></head>\n<body>\n"
    saved = head + "a line of the page\n" * 4500
    fs(tmp_path, operation="write", path="o.html", content=saved, more_parts=True)
    r = fs(tmp_path, operation="write", path="o.html", content=head + "a better line\n" * 1100,
           more_parts=True, cut_off_saved=60000)
    assert getattr(r, "reason_code", None) == "tool_call_cut_off" and "CUT OFF before it got past" in str(r)
    assert "REPLACE them with a much shorter version" not in str(r) and "delete the file first" in str(r)
    assert (tmp_path / "o.html").read_text() == saved


def test_a_reused_request_id_starts_with_a_clean_count():
    """Fails where one cut in a scheduled task (sched-<job>, the same id on
    every fire) told every later run it was "cut-off #5"."""
    import ast
    from pathlib import Path
    A.note_length_stop("sched-daily", 300.0)
    tree = ast.parse(Path(A.__file__).read_text(encoding="utf-8"))
    starts = [fn for fn in ast.walk(tree) if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
              and any(isinstance(c, ast.Call) and c.args and isinstance(c.args[0], ast.Constant)
                      and c.args[0].value == "Request Initialized" for c in ast.walk(fn))]
    assert starts
    ok = False
    for fn in starts:
        begin = min(c.lineno for c in ast.walk(fn) if isinstance(c, ast.Call) and c.args
                    and isinstance(c.args[0], ast.Constant) and c.args[0].value == "Request Initialized")
        ok = ok or any(isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "pop"
                       and getattr(c.func.value, "id", "") == "_LENGTH_STOPS"
                       and "req_id" in ast.unparse(c.args[0]) and c.lineno <= begin
                       for c in ast.walk(fn))
    assert ok, "the turn start must clear the request's length-stop row before the turn runs"


def test_a_cut_restart_that_gains_nothing_is_refused(tmp_path):
    """Fails where the model re-sent gen_ocean.py from its first line four
    times, each cut at ~36 KB, each replacing the saved part — 13 minutes,
    no progress (probe-oa-facts)."""
    head = "#!/usr/bin/env python3\nfacts = [\n"
    saved = head + "".join(f'    "fact {i} about the sea",\n' for i in range(1400))
    fs(tmp_path, operation="write", path="g.py", content=saved, more_parts=True, cut_off_saved=38000)
    again = head + "".join(f'    "fact {i} about the ocean",\n' for i in range(1400))
    r = fs(tmp_path, operation="write", path="g.py", content=again, more_parts=True, cut_off_saved=38100)
    assert getattr(r, "reason_code", None) == "tool_call_cut_off" and "can never get past" in str(r)
    assert (tmp_path / "g.py").read_text() == saved


def test_a_cut_restart_that_is_clearly_longer_still_replaces(tmp_path):
    head = "#!/usr/bin/env python3\nfacts = [\n"
    saved = head + "".join(f'    "fact {i}",\n' for i in range(400))
    fs(tmp_path, operation="write", path="g.py", content=saved, more_parts=True)
    longer = head + "".join(f'    "fact {i} and more words",\n' for i in range(1000))
    r = fs(tmp_path, operation="write", path="g.py", content=longer, more_parts=True, cut_off_saved=60000)
    assert str(r).startswith("SUCCESS") and (tmp_path / "g.py").read_text() == longer


def test_a_failing_run_counts_as_verifying_for_the_edit_churn_brake():
    """Fails where only a PASSING run reset the edit counter: a debug loop
    (edit → failing self-test → edit) was force-stopped at 351 s with the
    message "ran out of budget" (probe-oa-py)."""
    import ast
    from pathlib import Path
    tree = ast.parse(Path(A.__file__).read_text(encoding="utf-8"))
    branches = [n for n in ast.walk(tree) if isinstance(n, ast.If)
                and "_EDIT_CHURN_VERIFY_TOOLS" in ast.unparse(n.test)]
    assert branches
    test_src = ast.unparse(branches[0].test)
    assert "is_rejection" in test_src and "_res_is_error" not in test_src
    body = ast.unparse(branches[0])
    assert "['steers'] = 0" in body.replace('"', "'")


def test_a_check_that_stops_the_turn_is_named_in_the_fallback():
    from ghost_agent.core.reply_shape_check import refute_no_answer_fallback
    r = A._no_answer_fallback_reply([], ask="write inventory.py",
                                    stopped_by="'inventory.py' was edited again and again without being run")
    assert r.startswith("A safety check stopped this turn") and "What stopped it: 'inventory.py'" in r
    assert "budget" not in r.split("\n")[0]
    assert refute_no_answer_fallback(r)                    # still recognised as a non-answer
    plain = A._no_answer_fallback_reply([], ask="x")
    assert plain.startswith("I ran out of this turn's budget")


def test_the_turn_loop_hands_the_guards_reason_to_the_fallback():
    """Enumerated over the parsed source: the forced-final fallback is built
    with the breaker's reason, and the edit-churn stop records one."""
    import ast
    from pathlib import Path
    tree = ast.parse(Path(A.__file__).read_text(encoding="utf-8"))
    calls = [c for c in ast.walk(tree) if isinstance(c, ast.Call)
             and getattr(c.func, "id", "") == "_no_answer_fallback_reply"]
    assert calls and any(any(k.arg == "stopped_by" and "_breaker_reason" in ast.unparse(k.value)
                             for k in c.keywords) for c in calls)
    sets = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
            and any("_breaker_reason" in ast.unparse(t) for t in n.targets)
            and isinstance(n.value, (ast.JoinedStr, ast.Constant)) and "edited" in ast.unparse(n.value)]
    assert sets


def test_the_edit_churn_reminders_have_a_request_wide_ceiling():
    """A run resets the per-cycle reminder count; without a ceiling an
    edit×3 → run cycle injected reminders without end (review)."""
    churn = {"counts": {}, "steers": 0}
    steers = 0
    for _cycle in range(10):
        for _ in range(A._EDIT_CHURN_STEER_AFTER):
            if A.edit_churn_decision(churn, "f.py") == "steer":
                steers += 1
        churn["counts"].clear()
        churn["steers"] = 0            # what a verification run does
    assert steers == A.EDIT_CHURN_TOTAL_STEERS
