"""§4PH (2026-10-15) — how the agent writes code in an interactive turn.

Four fresh-eye readers over the coding turn (workflow + prompts, execute
result reading, the verifier's mechanical claim arms, a blind grade of the
two released apps) and a 158-turn corpus: 82 replies said works / tested /
verified, 27 of them ran nothing; 1 of 158 ran a test runner; the only "did
you run it" guard keyed on the turn's LAST tool, so a trailing `ls` cleared
it; `|| true` and `; echo done` laundered failing programs into EXIT CODE 0;
the same-error breaker never fired when the numbers changed.

Every pin here is behavioural (the ratchet rejects source-text pins).
"""
from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from ghost_agent.core import agent as A
from ghost_agent.core import code_claims as CC
from ghost_agent.core.strikes import error_line_fingerprint, normalise_volatile
from ghost_agent.tools import execute as ex
from ghost_agent.tools.execute import _normalise_exit, _normalise_pipe_exit, tool_execute
from ghost_agent.tools.outcome import OutcomeStatus, ToolOutcome
from ghost_agent.tools.tool_failure import FailureClass, classify_tool_failure, exec_exit_code
from ghost_agent.utils.constraints import request_forbids_running

from tests import pin_ast as P


# ── record builders ─────────────────────────────────────────────────

def _fs(path, n=3000):
    return {"role": "tool", "name": "file_system",
            "content": ToolOutcome(f"SUCCESS: Wrote {n} chars to '{path}'.",
                                   call_args={"operation": "write", "path": path})}


def _edit(path):
    return {"role": "tool", "name": "file_system",
            "content": ToolOutcome(f"SUCCESS: edited — replaced 1 occurrence of old_string (line 3) in '{path}'.",
                                   call_args={"operation": "edit", "path": path})}


def _ex(command=None, code=0, out="ok", *, filename=None, content=None, args=None, promoted=False):
    if promoted:
        from ghost_agent.sandbox.jobs import PROMOTED_RESULT_BANNER
        text = f"{PROMOTED_RESULT_BANNER}\nstill running"
        st = OutcomeStatus.UNRESOLVED
    else:
        text = f"--- COMMAND RESULT ---\nEXIT CODE: {code}\nSTDOUT/STDERR:\n{out}"
        st = OutcomeStatus.OK if code == 0 else OutcomeStatus.FAILED
    ca = {k: v for k, v in (("command", command), ("filename", filename),
                            ("content", content), ("args", args)) if v is not None}
    return {"role": "tool", "name": "execute", "content": ToolOutcome(text, status=st, call_args=ca)}


def _browser(url, ok=True):
    text = f"--- BROWSER RESULT ---\nSTATUS: {'OK' if ok else 'ERROR'}\nloaded"
    return {"role": "tool", "name": "browser",
            "content": ToolOutcome(text, status=OutcomeStatus.OK if ok else OutcomeStatus.FAILED,
                                   call_args={"operation": "screenshot", "url": url})}


def _mgr(returns):
    mgr = MagicMock()
    mgr.execute = MagicMock(return_value=returns)
    return mgr


# ═════════════════════════ B. the claim arm (R1-1, R3-1, R3-2, R3-4) ═════════════════════════

@pytest.mark.parametrize("text", [
    "I wrote app.py and tested it — it works.",
    "All tests pass.",
    "The script works as expected on the sample data.",
    "Confirmed working: the calendar renders every month.",
    "I ran it successfully.",
    "I tested it — works.",
    "Tested, works.",
    "Το έτρεξα και δουλεύει.",
])
def test_claim_detector_positives(text):
    assert CC.claims_code_works(text)


@pytest.mark.parametrize("text", [
    "I could not test it here — run it on your machine.",
    "Written, not run: the change is untested.",
    "It should work once you install the package.",
    "I cannot confirm it works without the database.",
    "The page is written; I haven't tested it.",
    "Here is the function you asked for.",
    "If the tests pass, deploy it.",
    # r3 reader M1: mechanism prose and honest reporting are not claims
    "This works by splitting the input on commas and summing each column.",
    "A regex which works on both Windows and POSIX paths.",
    "I ran the tests and 2 of 5 failed — the parser needs a fix.",
    "I checked that the file exists.",
    "Written, not run. I ran the linter only.",
    "Να πώς δουλεύει: διαβάζει το αρχείο και τυπώνει τα σύνολα.",
    "Here is how it works: the loop reads one line at a time.",
])
def test_claim_detector_negatives(text):
    assert not CC.claims_code_works(text)


def test_trailing_ls_does_not_count_as_a_run():
    """R3-1 S1: write 3,000 chars → `execute ls` → the file was never run."""
    never, failed = CC.unbacked_code_writes([_fs("app.py"), _ex("ls -la")])
    assert never == ["app.py"] and failed == []


def test_read_back_and_python_version_do_not_count_either():
    never, _ = CC.unbacked_code_writes([
        _fs("app.py"),
        {"role": "tool", "name": "file_system", "content": ToolOutcome("import os\n...", call_args={"operation": "read"})},
        _ex("python --version", out="Python 3.10"),
    ])
    assert never == ["app.py"]


def test_running_the_file_backs_it():
    never, failed = CC.unbacked_code_writes([_fs("app.py"), _ex("python3 app.py")])
    assert never == [] and failed == []


def test_a_failed_run_is_named_as_failed():
    """R3-2 S4: the only run of the file FAILED — the file is not 'tested'."""
    never, failed = CC.unbacked_code_writes([
        _fs("app.py"), _ex("python3 app.py", code=1, out="Traceback (most recent call last):\nNameError: x")])
    assert never == [] and failed == ["app.py"]


def test_a_run_before_the_last_edit_tested_an_older_version():
    never, _ = CC.unbacked_code_writes([_fs("app.py"), _ex("python3 app.py"), _edit("app.py")])
    assert never == ["app.py"]


def test_a_suite_run_after_the_write_backs_every_code_file_but_not_a_page():
    never, _ = CC.unbacked_code_writes([
        _fs("app.py"), _fs("util.py"), _fs("index.html"),
        _ex("python -m pytest tests/ -q", out="3 passed")])
    assert never == ["index.html"]


def test_a_browser_load_backs_a_page_and_a_failed_load_does_not():
    assert CC.unbacked_code_writes([_fs("index.html"), _browser("http://127.0.0.1:8000/x/index.html")]) == ([], [])
    assert CC.unbacked_code_writes([_fs("index.html"), _browser("file:///workspace/index.html", ok=False)]) == ([], ["index.html"])
    # a BLOCKED page (a 4xx / challenge) under an ok-status outcome is not a load either
    blocked = {"role": "tool", "name": "browser",
               "content": ToolOutcome("--- BROWSER RESULT ---\nSTATUS: BLOCKED\n403", status=OutcomeStatus.OK,
                                      call_args={"operation": "screenshot", "url": "http://127.0.0.1:8000/index.html"})}
    assert CC.unbacked_code_writes([_fs("index.html"), blocked]) == ([], ["index.html"])


def test_a_heredoc_write_through_execute_is_in_the_ledger():
    """R3-4: `cat > app.py <<'EOF'` is a write, not a run."""
    never, _ = CC.unbacked_code_writes([_ex("cat > app.py <<'EOF'\nprint(1)\nEOF")])
    assert never == ["app.py"]
    # …and the same command running it afterwards IS a run
    assert CC.unbacked_code_writes([_ex("cat > app.py <<'EOF'\nprint(1)\nEOF\npython3 app.py")]) == ([], [])


def test_execute_with_filename_and_content_writes_and_runs():
    assert CC.unbacked_code_writes([_ex(filename="probe.py", content="print(1)")]) == ([], [])
    # an ephemeral script that imports the written module ran it
    assert CC.unbacked_code_writes([_fs("calc.py"), _ex(content="import calc\nprint(calc.add(1, 2))")]) == ([], [])


def test_module_and_import_runs_back_the_file():
    """r3 reader M2: `python -m app`, `-c 'import calc'`, `uvicorn main:app`
    and a served root are the common 'run it' shapes."""
    assert CC.unbacked_code_writes([_fs("app.py"), _ex("python3 -m app")]) == ([], [])
    assert CC.unbacked_code_writes([_fs("calc.py"), _ex("python3 -c 'import calc; print(calc.add(1, 2))'")]) == ([], [])
    assert CC.unbacked_code_writes([_fs("main.py"), _ex("uvicorn main:app --port 8000", promoted=True),
                                    _ex("curl -s http://127.0.0.1:8000/")]) == (["main.py"], [])   # a detached start is not a run
    assert CC.unbacked_code_writes([_fs("main.py"), _ex("timeout 5 uvicorn main:app --port 8000", code=124, out="")]) == ([], ["main.py"])
    assert CC.unbacked_code_writes([_fs("index.html"), _browser("http://127.0.0.1:8000/")]) == ([], [])
    assert CC.unbacked_code_writes([_fs("app.py"), _ex("python3 -m pytest -q", out="1 passed")]) == ([], [])


def test_a_detached_run_is_not_a_run():
    never, _ = CC.unbacked_code_writes([_fs("app.py"), _ex("python3 app.py", promoted=True)])
    assert never == ["app.py"]


def test_prose_files_are_not_code():
    assert CC.unbacked_code_writes([_fs("report.md"), _fs("data.json")]) == ([], [])


def test_the_note_fires_on_a_claim_over_an_unrun_file_whatever_the_last_tool():
    tools = [_fs("app.py"), _ex("ls")]
    note, failed = A._unverified_mutation_note(tools[-1], "build me an app", tools,
                                               reply="I wrote app.py and tested it — it works.")
    assert failed is True
    assert "Unverified claim" in note and "app.py" in note and "never run" in note


def test_the_note_names_a_failed_run():
    tools = [_fs("app.py"), _ex("python3 app.py", code=1, out="Traceback (most recent call last):\nNameError: x"),
             _ex("echo ok")]
    note, failed = A._unverified_mutation_note(tools[-1], "build me an app", tools, reply="All tests pass.")
    assert failed is True and "FAILED" in note


def test_no_claim_keeps_the_operator_decision():
    """A write with no claim and a trailing tool: the 2026-09-23 footer
    decision holds — nothing is appended."""
    tools = [_fs("app.py"), _ex("ls")]
    assert A._unverified_mutation_note(tools[-1], "build me an app", tools,
                                       reply="Here is app.py; I have not run it.") == ("", False)


def test_a_small_write_with_a_false_claim_is_still_marked():
    """R3-7: the size exemption was about the footer, not about a false 'tested'."""
    tools = [_fs("run.sh", n=400), _ex("ls")]
    note, failed = A._unverified_mutation_note(tools[-1], "write a backup script", tools,
                                               reply="run.sh is tested and working.")
    assert failed is True and "run.sh" in note


def test_dont_run_it_is_said_not_failed():
    tools = [_fs("app.py"), _ex("ls")]
    note, failed = A._unverified_mutation_note(tools[-1], "write app.py, do not run it", tools,
                                               reply="Done — it works.")
    assert failed is False and "Not run, as you asked" in note and "not backed by a run" in note


def test_the_write_last_shape_keeps_its_note():
    tools = [_fs("app.py")]
    note, failed = A._unverified_mutation_note(tools[-1], "build me an app", tools, reply="Done, it works.")
    assert failed is True and "final action" in note


def test_a_backed_claim_is_clean():
    tools = [_fs("app.py"), _ex("python3 app.py"), _ex("ls")]
    assert A._unverified_mutation_note(tools[-1], "build me an app", tools, reply="Tested — it works.") == ("", False)


def test_the_repair_directive_names_the_file():
    d = CC.unbacked_claim_directive("tested, it works", [_fs("app.py"), _ex("ls")])
    assert "app.py" in d and "RUN it now" in d
    assert CC.unbacked_claim_directive("tested, it works", [_fs("app.py"), _ex("python3 app.py")]) == ""


def test_both_delivery_paths_pass_the_reply_to_the_note():
    """The claim arm reads the reply: finalize passes its final content, the
    stream drain its accumulated content — a path that passes nothing is the
    old last-tool proxy."""
    t_fin = P.tree(A.GhostAgent._finalize_and_return)
    t_str = P.tree(A.GhostAgent._stream_final_generation)
    for t, want in ((t_fin, "final_ai_content"), (t_str, "full_content")):
        cs = P.calls(t, "_unverified_mutation_note")
        assert cs, "note call missing"
        assert any(P.U(P.call_kw(c, "reply")) == want for c in cs), [P.U(c) for c in cs]


def test_the_in_loop_repair_consults_the_claim_arm():
    t = P.tree(A.GhostAgent._run_internal_turn)
    assert P.calls(t, "unbacked_claim_directive"), "in-loop repair never asks the claim arm"
    vals = [P.U(a.value) for a in P.assigns(t, "_unverified")]
    # BOTH verdict branches (async and sync) consult the arm…
    assert sum("_claim_directive" in v for v in vals) >= 2, vals
    # …and the repair directive is the arm's (it names the file) when it fired
    dvals = [P.U(a.value) for a in P.assigns(t, "_directive")]
    assert any(v.startswith("_claim_directive or") for v in dvals), dvals


def test_streamed_work_log_books_the_unverified_turn_failed():
    """R3-3: the stream drain stamped the trajectory failed and booked the
    work_log 'completed' with the claim as its note."""
    t = P.tree(A.GhostAgent._stream_final_generation)
    cs = [c for c in P.calls(t, "_write_project_work_log_safe")
          if P.U(P.call_kw(c, "final_ai_content")) == "_treated_content"]
    assert cs, "streamed work_log call not found"
    for c in cs:
        v = P.call_kw(c, "verifier_backfill")
        assert v is not None and P.U(v) != "None", P.U(c)
        assert "_um_failed_stream" in P.U(v)


# ═════════════════════════ B'. absence refute (R3-5) ═════════════════════════

def test_execute_is_removal_capable_by_what_it_ran():
    assert A._is_removal_capable("execute", "x", {"command": "python3 report.py"}) is False
    assert A._is_removal_capable("execute", "x", {"command": "rm -f report.py"}) is True
    assert A._is_removal_capable("execute", "x", {"command": "mv report.py old/"}) is True
    assert A._is_removal_capable("execute", "x", {"content": "import os\nos.remove('report.py')"}) is True
    assert A._is_removal_capable("execute", "x", {"content": "import shutil\nshutil.rmtree('build')"}) is True
    assert A._is_removal_capable("execute", "x", {"command": "python3 -m pytest -q"}) is False
    assert A._is_removal_capable("execute", "x", {"command": "ls form charm"}) is False
    assert A._is_removal_capable("execute", "x", {"content": "s = line.replace('a', 'b')"}) is False   # r3: str.replace
    assert A._is_removal_capable("execute", "x", {"command": "make clean && make"}) is True
    assert A._is_removal_capable("execute", "x", None) is True        # a legacy row: fail-open


def test_write_then_run_keeps_the_missing_file_refutable():
    tools = [_fs("report.py"), _ex("python3 report.py")]
    assert "report.py" not in A._keys_removable_after_write(tools)
    tools_rm = [_fs("report.py"), _ex("python3 report.py; rm report.py")]
    assert "report.py" in A._keys_removable_after_write(tools_rm)


# ═════════════════════════ B''. write-time checks + the no-run matcher ═════════════════════════

def test_a_shell_script_gets_a_parse_check_at_write(tmp_path):
    from ghost_agent.tools.file_system import _syntax_feedback
    bad = tmp_path / "run.sh"
    bad.write_text("#!/bin/bash\nif [ -f x ; then\n  echo hi\n")
    good = tmp_path / "ok.sh"
    good.write_text("#!/bin/bash\nif [ -f x ]; then\n  echo hi\nfi\n")
    assert asyncio.run(_syntax_feedback(bad, "run.sh"))
    assert asyncio.run(_syntax_feedback(good, "ok.sh")) == ""
    # r3: a bash-4 construct the sandbox accepts is not an error on the host's bash 3.2
    b4 = tmp_path / "b4.sh"
    b4.write_text("#!/bin/bash\ncmd 2>&1 |& tee log\n")
    assert asyncio.run(_syntax_feedback(b4, "b4.sh")) == ""


@pytest.mark.parametrize("text,want", [
    ("the script must never execute anything downloaded", False),
    ("never execute untrusted input", False),
    ("do not run it", True),
    # r3 reader M5: a relative clause names the thing not to run
    ("please don't run the script that I paste below, just review it", True),
    ("do not execute the code which follows; only explain it", True),
    ("don't run it from the sandbox, I will run it myself", True),
    ("no need to run anything", True),
    ("don't forget to run the tests", False),
])
def test_forbids_running_qualified_objects(text, want):
    assert request_forbids_running(text) is want


# ═════════════════════════ C. strike fingerprints (R1-2) ═════════════════════════

def test_assertion_values_do_not_make_a_new_error():
    a = error_line_fingerprint("--- EXECUTION RESULT ---\nEXIT CODE: 1\nSTDOUT/STDERR:\nAssertionError: expected 372.00, got 369.50", tool="execute")
    b = error_line_fingerprint("--- EXECUTION RESULT ---\nEXIT CODE: 1\nSTDOUT/STDERR:\nAssertionError: expected 372.00, got 371.20", tool="execute")
    assert a and a == b


def test_different_missing_modules_are_different_errors():
    a = error_line_fingerprint("EXIT CODE: 1\nModuleNotFoundError: No module named 'foo'", tool="execute")
    b = error_line_fingerprint("EXIT CODE: 1\nModuleNotFoundError: No module named 'bar'", tool="execute")
    assert a != b
    assert normalise_volatile("No module named 'foo'", keep_quoted=True) == "No module named 'foo'"
    assert normalise_volatile("No module named 'foo'") == "No module named '…'"


def test_a_moving_line_number_is_the_same_error():
    a = error_line_fingerprint('EXIT CODE: 1\n  File "x.py", line 12, in <module>\nValueError: bad at index 3', tool="execute")
    b = error_line_fingerprint('EXIT CODE: 1\n  File "x.py", line 15, in <module>\nValueError: bad at index 7', tool="execute")
    assert a == b


def test_same_class_failing_runs_are_not_progress():
    rec = {"writes": 3, "runs": 0}
    A.note_run_outcome(rec, ok=False, output="EXIT CODE: 1\nAssertionError: expected 372.00, got 369.50")
    assert rec["progress"] is True                   # the first failure is not a stall
    A.note_run_outcome(rec, ok=False, output="EXIT CODE: 1\nAssertionError: expected 372.00, got 371.20")
    assert rec["progress"] is False
    A.note_run_outcome(rec, ok=False, output="EXIT CODE: 1\nTypeError: unsupported operand")
    assert rec["progress"] is True                   # a different error is a move


# ═════════════════════════ D. execute results (R2) ═════════════════════════

PYTEST_FAIL = "FAILED tests/test_a.py::test_c - assert 1 == 2\n1 failed, 2 passed in 0.3s"
TRACE = "Traceback (most recent call last):\n  File \"x.py\", line 1, in <module>\nModuleNotFoundError: No module named 'foo'"


@pytest.mark.parametrize("command,out,want", [
    ("python3 -m pytest tests/ -q || true", PYTEST_FAIL, 1),
    ("python3 -m pytest tests/ -q; echo done", PYTEST_FAIL, 1),
    ("python3 x.py; ls", TRACE, 1),
    ("python3 x.py\nls", TRACE, 1),
    ("cd x && python3 y.py", TRACE, 0),                      # && is not a launderer
    ("python3 x.py && echo ok", TRACE, 0),                   # …even with a runner first: a caught error under 0
    ("python3 x.py", TRACE, 0),                              # a caught error under 0 is the same-error arm's
    ("python3 -c 'print(1); print(2)'", TRACE, 0),           # the `;` is quoted
    ("python3 x.py; ls", "all good", 0),
])
def test_list_laundering_is_undone(command, out, want):
    assert _normalise_exit(command, 0, out) == want


async def test_laundered_failure_reads_as_a_failure_with_the_note(tmp_path):
    res = await tool_execute(command="python3 -m pytest tests/ -q || true",
                             sandbox_dir=tmp_path, sandbox_manager=_mgr((PYTEST_FAIL, 0)))
    assert "EXIT CODE: 1" in res and "LAST command of the list" in res
    assert classify_tool_failure(str(res))[0] != FailureClass.RETRYABLE
    assert exec_exit_code(str(res)) == 1


async def test_a_missing_exit_status_is_a_failure_not_a_zero(tmp_path):
    """R2-3: docker's `None` read as `EXIT CODE: 0` on the command path and
    `EXIT CODE: None` on the script path."""
    res = await tool_execute(command="python3 x.py", sandbox_dir=tmp_path,
                             sandbox_manager=_mgr((TRACE, None)))
    assert "EXIT CODE: 1" in res and "NO exit status" in res
    res2 = await tool_execute(filename="x.py", content="print(1)", sandbox_dir=tmp_path,
                              sandbox_manager=_mgr(("", None)))
    assert "EXIT CODE: 1" in res2 and "EXIT CODE: None" not in res2


async def test_own_timeout_is_not_the_budget_and_not_transient(tmp_path):
    res = await tool_execute(command="timeout 60 python3 slow.py", sandbox_dir=tmp_path,
                             sandbox_manager=_mgr(("", 124)))
    assert "EXIT CODE: 124" in res
    assert "own `timeout`" in res and "timed out / killed after" not in res
    assert classify_tool_failure(str(res))[0] != FailureClass.RETRYABLE


def test_the_budget_kill_is_a_strike_and_infra_is_transient():
    budget = ("--- EXECUTION RESULT ---\nEXIT CODE: 124 (timed out / killed after 600s)\nSTDOUT/STDERR:\n"
              "--- 💡 DIAGNOSTIC HINT ---\nExit 124 means the process was KILLED — most likely it exceeded the 600s execution budget")
    assert classify_tool_failure(budget)[0] != FailureClass.RETRYABLE
    infra = ("--- EXECUTION RESULT ---\nEXIT CODE: 1\nSTDOUT/STDERR:\n"
             "[SANDBOX INFRA ERROR — not your code] docker daemon wedged; re-run the command")
    assert classify_tool_failure(infra)[0] == FailureClass.RETRYABLE
    infra2 = ("--- EXECUTION RESULT ---\nEXIT CODE: 137\nSTDOUT/STDERR:\n"
              "[SANDBOX INFRA ERROR] the sandbox container stopped or restarted while this ran")
    assert classify_tool_failure(infra2)[0] == FailureClass.RETRYABLE


async def test_a_backgrounded_process_is_only_started(tmp_path):
    res = await tool_execute(command="nohup python3 app.py > app.log 2>&1 &", sandbox_dir=tmp_path,
                             sandbox_manager=_mgr(("", 0)))
    assert "EXIT CODE: 0" in res and "only that it STARTED" in res
    plain = await tool_execute(command="python3 app.py", sandbox_dir=tmp_path,
                               sandbox_manager=_mgr(("", 0)))
    assert "STARTED" not in plain


async def test_background_note_ignores_heredoc_and_quoted_ampersands(tmp_path):
    """r3 reader M4: `Tom &amp; Jerry` inside a heredoc is data."""
    res = await tool_execute(command="cat > index.html <<'EOF'\n<p>Tom &amp; Jerry</p>\nEOF",
                             sandbox_dir=tmp_path, sandbox_manager=_mgr(("", 0)))
    assert "STARTED" not in res
    res2 = await tool_execute(command="git commit -m 'a & b'", sandbox_dir=tmp_path, sandbox_manager=_mgr(("", 0)))
    assert "STARTED" not in res2
    res3 = await tool_execute(command="python3 srv.py & sleep 1; curl -s localhost:9/; kill %1",
                              sandbox_dir=tmp_path, sandbox_manager=_mgr(("", 0)))
    assert "STARTED" not in res3


@pytest.mark.parametrize("command,out", [
    ("cat app.log; echo ---", TRACE),                       # r3 reader M6: a READER of failure text
    ("grep -h Traceback logs/a.log; echo done", TRACE),
    ("git log -1; echo", "FAILED tests/test_a.py::test_c fix\n"),
])
def test_list_laundering_needs_a_runner(command, out):
    assert _normalise_exit(command, 0, out) == 0


async def test_a_missing_status_keeps_the_pipestatus_marker_readable(tmp_path):
    """r3 reader: the no-status note landed BEFORE the end-anchored marker read."""
    out = "x\n__GHOST_PIPESTATUS=2 0\n"
    res = await tool_execute(command="python3 x.py | grep foo", sandbox_dir=tmp_path,
                             sandbox_manager=_mgr((out, None)))
    assert "__GHOST_PIPESTATUS" not in res and "EXIT CODE: 1" in res


async def test_a_124_without_a_timeout_wrapper_is_not_blamed_on_one(tmp_path):
    res = await tool_execute(command="python3 slow.py", sandbox_dir=tmp_path,
                             sandbox_manager=_mgr(("", 124)))
    assert "EXIT CODE: 124" in res and "NOT the budget kill" in res
    assert "own `timeout`" not in res and "timed out / killed after" not in res
    assert classify_tool_failure(str(res))[0] != FailureClass.RETRYABLE


def test_broken_pipe_text_forgives_only_an_exit_1():
    out = "Traceback (most recent call last):\nBrokenPipeError: [Errno 32] Broken pipe"
    assert _normalise_pipe_exit("python3 gen.py | head -5", 1, out) == 0
    assert _normalise_pipe_exit("python3 gen.py | head -5", 124, out) == 124
    assert _normalise_pipe_exit("python3 gen.py | head -5", 2, out) == 2
    assert _normalise_pipe_exit("python3 gen.py | head -5", 141, "") == 0


def test_job_banner_counts_only_at_the_start_of_a_result():
    assert exec_exit_code("[sandbox job job-1 finished — EXIT CODE: 0]\n$ cmd\nout") == 0
    assert exec_exit_code("[FAILURE BANNER] x\n[sandbox job job-1 finished — EXIT CODE: 2]\n$ cmd") == 2
    # r3 reader M3: a COLLECTED job's banner sits under the collect header — still its code
    collect = ("Collected 1 finished job(s):\n\n--- job-1: build ---\n"
               "[sandbox job job-1 finished — EXIT CODE: 2]\n$ python3 build.py\nboom")
    assert exec_exit_code(collect) == 2
    assert CC._exec_ok({"name": "execute", "content": collect}) is False
    quoting = ("--- COMMAND RESULT ---\nEXIT CODE: 1\nSTDOUT/STDERR:\n"
               "earlier: [sandbox job job-1 finished — EXIT CODE: 0] then it broke")
    assert exec_exit_code(quoting) == 1
    # …and a banner QUOTED at a line start (a `cat` of a saved job result) under
    # the result's own EXIT CODE line is still the result's code
    quoting2 = ("--- COMMAND RESULT ---\nEXIT CODE: 1\nSTDOUT/STDERR:\n"
                "[sandbox job job-1 finished — EXIT CODE: 0]\n$ python3 old.py\nfine")
    assert exec_exit_code(quoting2) == 1


def test_pipestatus_trailer_reports_the_first_failing_stage():
    bash = "/bin/bash"
    if not os.path.exists(bash):
        pytest.skip("no bash")
    cmd = "set -o pipefail; sh -c 'echo x; exit 2' | grep zzz" + ex._PIPESTATUS_TRAILER
    rc = subprocess.run([bash, "-c", cmd], capture_output=True, text=True).returncode
    assert rc == 2


# ═════════════════════════ A. the standing rules (R1-3, R1-4) ═════════════════════════

def test_the_prompt_asks_for_a_run_before_a_claim():
    # §4UI: the coding persona is gone; the main prompt carries the rules
    from ghost_agent.core.prompts import SYSTEM_PROMPT
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    SP = SYSTEM_PROMPT
    assert "run it (or its tests) before you say it works" in SP   # a saved file is run
    assert "only about something you ran THIS turn" in SP
    assert "|| true" in SP                                   # exit 0 is the shell's last command
    assert "verifier will REFUTE" not in SP                  # stale: the judge is off
    assert "DO NOT use `file_system` to save it or `execute` to run it" not in SP
    d = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "execute")["function"]["description"]
    assert "LAST RESORT" not in d and "run the file (or its tests)" in d
    assert "- EXECUTION: Use `execute` ONLY for" not in SYSTEM_PROMPT


# ═════════════════════════ D'. the sandbox boundary (R2-3) ═════════════════════════

def test_streamed_exec_waits_for_the_daemon_to_record_the_exit():
    """docker-py's `exec_inspect` reads `ExitCode: None` when the stream EOFs
    before the daemon records the exit; that None became 0 downstream."""
    from ghost_agent.sandbox.docker import DockerSandbox
    sb = DockerSandbox.__new__(DockerSandbox)
    api = MagicMock()
    api.exec_create.return_value = {"Id": "e1"}
    api.exec_start.return_value = iter([b"out\n"])
    api.exec_inspect.side_effect = [{"ExitCode": None}, {"ExitCode": None}, {"ExitCode": 2}]
    sb.client = MagicMock()
    sb.client.api = api
    _out, code = sb._exec_run_streamed("python3 x.py", cid="c1", ram_cap=1 << 20, deadline_s=10)
    assert code == 2
    assert api.exec_inspect.call_count == 3


def test_a_converging_test_summary_is_not_the_same_error():
    """The numeric collapse must not erase test COUNTS: `9 failed, 3 passed`
    then `8 failed, 4 passed` under exit 0 read as one error five times and
    the same-error breaker forced a blocker report on a converging loop."""
    a = error_line_fingerprint("EXIT CODE: 0\n=== 9 failed, 3 passed in 0.1s ===", tool="execute")
    b = error_line_fingerprint("EXIT CODE: 0\n=== 8 failed, 4 passed in 0.1s ===", tool="execute")
    assert a and b and a != b
