"""§4OG-1 — the development loop, second pass (after the §4OF evaluation):
futility runs keyed by the SUITE, a green suite is never a stall, program
output is never "transient", tests-vs-source edit sides in TASK STATE, a
bounded turn-budget extension, coverage+version receipts, per-request receipt
cleanup, and a prompt that states one working directory."""
import asyncio

from ghost_agent.core import agent as ag
from ghost_agent.core.agent import (futility_keys_in_command, is_suite_command, futility_tier, note_run_outcome,
                                    build_is_converging, turn_budget_extension_fits, TURN_BUDGET_EXTENSION,
                                    FUTILITY_REPORT_WRITES, FUTILITY_REPORT_RUNS)
from ghost_agent.core import task_state as TS
from ghost_agent.tools import file_system as FS
from ghost_agent.tools.file_system import tool_file_system
from ghost_agent.tools.tool_failure import classify_tool_failure, FailureClass, is_program_output
from tests.pin_ast import tree, U, calls, call_kw, assigns, assigned_values, ifs, whiles


def fs(sb, **kw):
    return asyncio.run(tool_file_system(sandbox_dir=sb, **kw))


# ── (a) runs keyed by the suite ─────────────────────────────────────────── #

def test_a_suite_run_is_attributed_to_every_code_file_written_this_request():
    """Fails where `python -m pytest tests/ -q` named no file, so `runs` stayed
    0 and the §4OE-2 progress term never ran live (§4OF defect 1)."""
    written = {"app.py", "test_app.py", "cli.py"}
    assert futility_keys_in_command("cd /workspace/p && python -m pytest tests/ -q --import-mode=importlib",
                                    written_keys=written) >= written
    assert futility_keys_in_command("python -m unittest discover", written_keys=written) >= written
    assert futility_keys_in_command("npm test", written_keys=written) >= written
    assert futility_keys_in_command("go test ./...", written_keys=written) >= written
    # a command naming one file still counts only that file; `-m pkg.mod` names its module
    assert futility_keys_in_command("python3 gen.py --fast", written_keys=written) == {"gen.py"}
    assert futility_keys_in_command("python -m ledger.cli post", written_keys=written) == {"cli.py"}
    assert futility_keys_in_command("ls -la", written_keys=written) == set()
    assert is_suite_command("pytest -q") and not is_suite_command("python3 inventory.py --selftest")
    # the WIRING (an AST walk of the turn loop): the attribution site hands the request's written
    # keys to the keyer and notes the run's outcome (mutant G3 of the §4OG battery)
    t = tree(ag)
    kc = [c for c in calls(t, "futility_keys_in_command") if U(call_kw(c, "written_keys")) == "_si.keys()"]
    assert kc, "the attribution site does not pass the written keys"
    assert any(U(call_kw(c, "ok")) == "not _res_is_error" for c in calls(t, "note_run_outcome"))


def test_two_green_suite_runs_are_not_a_stall_but_a_green_countless_rerun_is():
    """Fails where the report tier ignored the exit code: feature → 23 passed →
    feature → 23 passed forced a blocker report on a converging build (§4OF
    defect 1b). The breaker's founding case — exit-0 runs whose output stayed
    wrong (xrick) — must still report."""
    rec = {"writes": FUTILITY_REPORT_WRITES, "runs": FUTILITY_REPORT_RUNS}
    note_run_outcome(rec, ok=True, output="===== 23 passed in 0.4s =====")
    note_run_outcome(rec, ok=True, output="===== 23 passed in 0.4s =====")
    assert rec["progress"] is False and rec["ok"] is True and rec["counted"] is True
    assert futility_tier({"app.py": rec}, True, False)[0] == ""
    note_run_outcome(rec, ok=False, output="===== 2 failed, 21 passed in 0.4s =====")
    note_run_outcome(rec, ok=False, output="===== 2 failed, 21 passed in 0.4s =====")
    assert futility_tier({"app.py": rec}, True, False)[0] == "report"        # failing and stalled: reported
    x = {"writes": FUTILITY_REPORT_WRITES, "runs": FUTILITY_REPORT_RUNS}
    note_run_outcome(x, ok=True, output="counts: 12 sprites (expected 300)")
    note_run_outcome(x, ok=True, output="counts: 12 sprites (expected 300)")
    assert x["counted"] is False and futility_tier({"extract.py": x}, True, False)[0] == "report"


# ── (i) program output is deterministic ─────────────────────────────────── #

def test_a_failing_pytest_run_is_never_transient():
    """Fails where `urlopen(req, timeout=5)` in a traceback booked the run
    RETRYABLE and told the model the same call may pass (3 of 5 failing runs
    in the §4OE probe)."""
    out = ("FAILED tests/test_api.py::TestCreate::test_timeout - AssertionError\n"
           "    resp = urlopen(req, timeout=5)\n"
           "===== 1 failed, 22 passed in 0.9s =====\nEXIT CODE: 1")
    cls, why = classify_tool_failure(out)
    assert cls != FailureClass.RETRYABLE and is_program_output(out)
    assert classify_tool_failure("Traceback ...\nrequests.exceptions.Timeout: read timed out\nEXIT CODE: 1")[0] == FailureClass.DIAGNOSTIC
    # the sandbox's own absence inside program output is still transient
    assert classify_tool_failure("sandbox busy — try again\nEXIT CODE: 1")[0] == FailureClass.RETRYABLE
    # a tool's own error is classified as before
    assert classify_tool_failure("Error: navigation timed out after 30s")[0] == FailureClass.RETRYABLE
    assert not is_program_output("Error: navigation timed out after 30s")


# ── (j) which side the loop edits ───────────────────────────────────────── #

def _row(name, args, content):
    return {"name": name, "args": args, "content": content}


def test_task_state_says_when_only_tests_were_edited_after_a_failing_run():
    """Fails where 18 of 26 edits went to test files and nothing noticed
    (§4OF defect 10)."""
    rows = [
        _row("file_system", {"operation": "write", "path": "p/ledger.py"}, "SUCCESS: Wrote 100 chars to 'p/ledger.py'. Script-side path (from sandbox cwd): 'p/ledger.py'."),
        _row("file_system", {"operation": "write", "path": "p/tests/test_ledger.py"}, "SUCCESS: Wrote 90 chars to 'p/tests/test_ledger.py'. Script-side path (from sandbox cwd): 'p/tests/test_ledger.py'."),
        _row("execute", {"command": "python -m pytest p/tests -q"}, "FAILED p/tests/test_ledger.py::test_a - assert 1 == 2\n1 failed in 0.2s\nEXIT CODE: 1"),
        _row("file_system", {"operation": "edit", "path": "p/tests/test_ledger.py"}, "SUCCESS: edited 'p/tests/test_ledger.py' (1 replacement)."),
    ]
    sides = TS.edit_sides(rows)
    assert sides["tests"] == 2 and sides["source"] == 1 and sides["since_fail_tests"] == 1 and sides["since_fail_source"] == 0
    text = TS.render_task_state(rows, ask="build it", turn=5)
    assert "1 to source files, 2 to test files" in text and "only TEST files were edited" in text
    # a source edit after the failure clears the warning; a refused edit does not count
    rows.append(_row("file_system", {"operation": "edit", "path": "p/ledger.py"}, "REJECTED: old_string not found"))
    assert "only TEST files were edited" in TS.render_task_state(rows, ask="", turn=6)
    rows.append(_row("file_system", {"operation": "edit", "path": "p/ledger.py"}, "SUCCESS: edited 'p/ledger.py' (1 replacement)."))
    assert "only TEST files were edited" not in TS.render_task_state(rows, ask="", turn=7)
    # a NEW failing run resets the since-failure counters: a later test-only edit warns again (mutant G12)
    rows.append(_row("execute", {"command": "python -m pytest p/tests -q"}, "FAILED p/tests/test_ledger.py::test_b\n1 failed in 0.2s\nEXIT CODE: 1"))
    rows.append(_row("file_system", {"operation": "edit", "path": "p/tests/test_ledger.py"}, "SUCCESS: edited 'p/tests/test_ledger.py' (1 replacement)."))
    s2 = TS.edit_sides(rows)
    assert s2["since_fail_source"] == 0 and s2["since_fail_tests"] == 1
    assert "only TEST files were edited" in TS.render_task_state(rows, ask="", turn=8)
    assert TS.is_test_path("tests/test_x.py") and TS.is_test_path("src/a_test.go") and TS.is_test_path("app.spec.ts")
    assert not TS.is_test_path("src/contest.py") and not TS.is_test_path("latest/app.py")


# ── (k) the turn budget extends once for a converging build ─────────────── #

def test_a_converging_build_gets_one_bounded_extension():
    """Fails where the 40-turn wall fired on the turn the tests went green
    (§4OE probe, turn 38 of 40)."""
    assert build_is_converging({"app.py": {"writes": 3, "runs": 2, "progress": True, "improved": True}})
    assert not build_is_converging({"app.py": {"writes": 3, "runs": 2, "progress": True, "improved": False}})   # a first run
    assert not build_is_converging({"app.py": {"writes": 3, "runs": 1, "progress": True, "improved": True}})    # one run only
    assert not build_is_converging({}) and not build_is_converging(None)
    assert turn_budget_extension_fits(None, 300.0)                     # no deadline
    assert turn_budget_extension_fits(900.0, 300.0)
    assert not turn_budget_extension_fits(400.0, 300.0)                # the report would eat it
    assert TURN_BUDGET_EXTENSION == 10
    t = tree(ag)
    assert whiles(t, "turn + 1 < effective_max_turns"), "the turn loop is not the extensible while"
    assert "True" in assigned_values(t, "_budget_extended")
    assert any(isinstance(a, __import__("ast").AugAssign) and U(a.value) == "TURN_BUDGET_EXTENSION"
               for a in assigns(t, "effective_max_turns"))


def test_chunked_pages_add_up_and_a_sample_read_is_not_a_receipt(tmp_path):
    body = "".join(f"line {i}\n" for i in range(1000))
    (tmp_path / "big.py").write_text(body)
    fs(tmp_path, operation="read_chunked", path="big.py", page=1, chunk_size=4000)
    assert FS.read_coverage(tmp_path / "big.py") < 0.8
    for pg in range(2, 4):
        fs(tmp_path, operation="read_chunked", path="big.py", page=pg, chunk_size=4000)
    assert FS.read_coverage(tmp_path / "big.py") >= 0.99
    assert str(fs(tmp_path, operation="write", path="big.py", content="x = 1\n")).startswith("SUCCESS")


def test_overlapping_ranged_reads_do_not_double_count(tmp_path):
    """Mutant G24: counting an overlap twice would let 1-150 + 100-200 of a
    300-line file (67 % seen) pass the 80 % floor as 84 %."""
    lines = [f"x{i} = {i}\n" for i in range(300)]
    (tmp_path / "o.py").write_text("".join(lines))
    fs(tmp_path, operation="read", path="o.py", start_line=1, end_line=150)
    fs(tmp_path, operation="read", path="o.py", start_line=100, end_line=200)
    want = sum(map(len, lines[:200])) / sum(map(len, lines))    # §4TJ: coverage is by size
    assert abs(FS.read_coverage(tmp_path / "o.py") - want) < 0.001
    assert getattr(fs(tmp_path, operation="write", path="o.py", content="x = 1\n"), "reason_code", None) == "unread_rewrite"
    fs(tmp_path, operation="read", path="o.py", start_line=200, end_line=260)
    assert FS.read_coverage(tmp_path / "o.py") >= 0.8
    assert str(fs(tmp_path, operation="write", path="o.py", content="x = 1\n")).startswith("SUCCESS")


def test_a_replace_or_direct_append_by_this_request_keeps_its_own_file_current(tmp_path):
    """Mutants G26/M7: a `replace` (or the §4OB continuation's DIRECT append)
    that did not refresh the written stamp made the next whole write look
    like an outside change."""
    import asyncio
    from ghost_agent.tools.file_system import tool_append_file
    assert str(fs(tmp_path, operation="write", path="r.py", content="a = 1\nb = 2\n")).startswith("SUCCESS")
    assert str(fs(tmp_path, operation="replace", path="r.py", old_string="a = 1", new_string="a = 9")).startswith("SUCCESS")
    assert str(fs(tmp_path, operation="write", path="r.py", content="a = 9\nb = 2\nc = 3\n")).startswith("SUCCESS")
    assert str(asyncio.run(tool_append_file("r.py", "d = 4\n", tmp_path))).startswith("SUCCESS")
    assert str(fs(tmp_path, operation="write", path="r.py", content="a = 10\n")).startswith("SUCCESS")


def test_the_suite_regex_matches_runner_invocations_only():
    """Reader M4: `cat tox.ini`, `pip install pytest-cov`, `grep nose setup.cfg`
    counted as suite runs and fed pip output to the progress term."""
    for cmd in ("cat tox.ini", "pip install pytest-cov", "grep nose setup.cfg", "head -5 pytest.ini",
                "git log --oneline tox", "ls tests", "python3 app.py --selftest", "echo pytest done",
                "pytest --co -q", "pytest --help", "tox -e lint", "make check-format"):
        assert not is_suite_command(cmd), cmd
    for cmd in ("pytest -q", "python -m pytest tests/ -q --import-mode=importlib", "cd /workspace/p && pytest",
                "PYTHONPATH=p python3 -m pytest p/tests", "npm test", "npx jest", "go test ./...", "cargo test",
                "python -m unittest discover -s tests", "make test", "tox -e py311", "node --test",
                "timeout 60 python -m pytest", "/usr/bin/python3 -m pytest", ".venv/bin/python -m pytest tests",
                "uv run pytest", "poetry run pytest -q", "coverage run -m pytest", "env X=1 pytest", "python -X dev -m pytest"):
        assert is_suite_command(cmd), cmd


def test_the_steer_tier_leaves_a_converging_loop_alone_and_convergence_needs_an_improvement():
    """Reader M3/M6: with suite attribution every TDD loop reaches the steer;
    a loop whose last run improved must not be told to STOP patching, and
    one failing first run is not "converging"."""
    rec = {"writes": 3, "runs": 0}
    note_run_outcome(rec, ok=False, output="5 failed, 2 passed in 0.3s"); rec["runs"] += 1
    assert rec["improved"] is False and not build_is_converging({"app.py": rec})
    note_run_outcome(rec, ok=False, output="2 failed, 5 passed in 0.3s"); rec["runs"] += 1
    assert rec["improved"] is True and futility_tier({"app.py": rec}, False, False)[0] == ""
    assert build_is_converging({"app.py": rec})
    note_run_outcome(rec, ok=False, output="2 failed, 5 passed in 0.3s"); rec["runs"] += 1
    assert rec["improved"] is False and futility_tier({"app.py": rec}, False, False)[0] == "steer"
    x = {"writes": 3, "runs": 2}
    note_run_outcome(x, ok=True, output="rows=12 (expected 300)"); note_run_outcome(x, ok=True, output="rows=12 (expected 300)")
    assert futility_tier({"gen.py": x}, False, False)[0] == "steer"            # the founding case still steers


def test_network_failures_inside_program_output_stay_transient():
    """Reader M5: pip over Tor / git clone failures are the network, not the code."""
    assert classify_tool_failure("pip ... ReadTimeoutError: HTTPSConnectionPool(host='pypi.org') Read timed out.\nEXIT CODE: 1")[0] == FailureClass.RETRYABLE
    assert classify_tool_failure("fatal: unable to access 'https://x/': Connection reset by peer\nEXIT CODE: 128")[0] == FailureClass.RETRYABLE
    # …but a TEST that hits the network fails the same way every time (reader 2 M5)
    assert classify_tool_failure("FAILED tests/test_client.py::test_retry - ConnectionResetError: Connection reset by peer\n1 failed in 0.3s\nEXIT CODE: 1")[0] != FailureClass.RETRYABLE
    assert classify_tool_failure("FAILED tests/test_dns.py::test_x - socket.gaierror: Temporary failure in name resolution\n1 failed, 3 passed in 0.3s\nEXIT CODE: 1")[0] != FailureClass.RETRYABLE
    assert classify_tool_failure("Error: Sandbox container is not running\nEXIT CODE: 1")[0] == FailureClass.RETRYABLE


def test_receipts_survive_until_the_streamed_drain_is_done():
    """Reader C1: the handle_chat finally forgot the receipts BEFORE the
    streamed forced-final report write ran in the drain."""
    t = tree(ag)
    guarded = ifs(t, "not _stream_owns_unregister")
    assert any(calls(g, "_ffr") for g in guarded), "receipts are forgotten regardless of the stream"
    assert calls(t, "_ffr2"), "the stream wrapper's finally does not forget the receipts"


def test_chunked_receipts_use_the_readers_clamp_and_stride(tmp_path):
    """Reader M2: a requested chunk_size=200000 recorded 200,000 bytes of
    coverage for an 80,000-byte page."""
    (tmp_path / "big.txt").write_text("z" * 160_600)
    fs(tmp_path, operation="read_chunked", path="big.txt", page=1, chunk_size=200000)
    assert FS.read_coverage(tmp_path / "big.txt") < 0.6
    fs(tmp_path, operation="read_chunked", path="big.txt", page=2, chunk_size=200000)
    fs(tmp_path, operation="read_chunked", path="big.txt", page=3, chunk_size=200000)
    assert FS.read_coverage(tmp_path / "big.txt") >= 0.99


def test_receipts_are_forgotten_per_request(tmp_path):
    """Fails where the receipt tables were one 512-slot LRU across requests
    (an early read of a long request evicted; a later request inherited)."""
    from ghost_agent.utils.logging import request_id_context
    tok = request_id_context.set("req-4og-a")
    try:
        (tmp_path / "a.py").write_text("a = 1\n")
        fs(tmp_path, operation="read", path="a.py")
        assert FS.read_coverage(tmp_path / "a.py") == 1.0
        assert FS.forget_request_receipts("req-4og-a") >= 1
        assert FS.read_coverage(tmp_path / "a.py") == 0.0
        assert getattr(fs(tmp_path, operation="write", path="a.py", content="a = 2\n"), "reason_code", None) == "unread_rewrite"
    finally:
        request_id_context.reset(tok)


def test_the_prompt_states_one_working_directory_and_a_reachable_pypi():
    """Fails where the prompt said the shell ALWAYS starts in /workspace two
    lines after saying it starts in the project dir, and that the sandbox has
    no package network while pip works over Tor (§4OF)."""
    from ghost_agent.core.prompts import SPECIALIST_SYSTEM_PROMPT
    p = SPECIALIST_SYSTEM_PROMPT
    assert "Shell ALWAYS starts in `/workspace`" not in p
    assert "the shell starts in the active project directory" in p
    assert "SAME working directory for every command" in p
    assert "has no package network" not in p
    assert "works over Tor" in p


def test_task_state_gives_each_written_file_its_download_link_last():
    """§4OI (iv): "give me the download link" was answered with the vendor's
    download page; the model had never been shown the file's own link."""
    rows = [
        _row("file_system", {"operation": "write", "path": "probe/pg18.md"},
             "SUCCESS: Wrote 9 chars to 'probe/pg18.md'. Script-side path (from sandbox cwd): 'probe/pg18.md'."),
        _row("execute", {"command": "python -m pytest -q"}, "FAILED t.py::a\n1 failed in 0.1s\nEXIT CODE: 1"),
    ]
    text = TS.render_task_state(rows, ask="write it", turn=2)
    assert "[pg18.md](/api/download/probe/pg18.md)" in text
    # r1 M2: the route's spelling — the script-side path, under the project prefix, URL-encoded
    ws = [_row("file_system", {"operation": "write", "path": "/workspace/My Report (v2).md"},
               "SUCCESS: Wrote 9 chars to '/workspace/My Report (v2).md'. Script-side path (from sandbox cwd): 'My Report (v2).md'.")]
    t2 = TS.render_task_state(ws, turn=1, link_prefix="projects/abc123abc123/")
    assert "(/api/download/projects/abc123abc123/My%20Report%20%28v2%29.md)" in t2 and "/api/download/workspace/" not in t2
    # r1 m7: only whole links that fit the cap
    many = [_row("file_system", {"operation": "write", "path": f"d/{'n' * 90}{i}.md"},
                 f"SUCCESS: Wrote 9 chars to 'd/{'n' * 90}{i}.md'. Script-side path (from sandbox cwd): 'd/{'n' * 90}{i}.md'.")
            for i in range(8)]
    t3 = TS.render_task_state(many, turn=1)
    n_links = t3.count("](/api/download/")
    assert 1 <= n_links < 6                                                    # the budget bit: some fit, not all
    assert not t3.rstrip().endswith("…")                                      # nothing cut by the cap
    assert n_links == t3.count(".md)")                                         # only whole links (battery K4)
    assert len(t3.strip()) <= TS.TASK_STATE_MAX_CHARS
    assert text.index("Last failure") < text.index("/api/download/probe/pg18.md")     # the cap cuts links first
    assert "download page is not the file" in text
    assert "/api/download/" not in TS.render_task_state([rows[1]], ask="x", turn=1)    # no file, no link line
