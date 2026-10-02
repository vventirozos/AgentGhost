"""§4KT (2026-09-30) — two subsystems from the §4KS residue.

(1) `core/strikes.py::error_line`: the failure-word scan is for PROGRAM
    OUTPUT (`execute`, `jobs`); every other tool returns CONTENT whose words
    are its subject — page text, a file, JSON, a snippet — and fails only by
    what it declares. Measured on the 9,119 tool results in the corpus: the
    success-shaped false positives (96 searches, 324 loaded pages, 450 file
    reads, 134 project records, 81 introspect lines) go to zero; the execute
    exit-0 class (§4IB) keeps 220 of 221 (the one lost was "terror").
(2) The trajectory `temperature` is the sampling temperature of the reply
    (every user/probe/leaf row carried the schema default 0.0), and a
    streamed turn prints the same `Turn Outcome` line as a non-streamed one,
    from one emitter, into the same late-correction ring.
"""
import asyncio
import ast
import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import ghost_agent.core.agent as agent_mod
from ghost_agent.core.agent import GhostAgent, StreamState, evidence_digest
from ghost_agent.core.strikes import (
    error_line, error_line_fingerprint, PROGRAM_OUTPUT_TOOLS, _ERROR_LINE_RE)
from ghost_agent.tools.outcome import ToolOutcome

from tests.test_4ks_log_review_fixes import _chat_context, _turn, _Bg
from tests.test_finalize_stream_pins import make_stream_agent, sse, _make_stream_state

SRC = Path(__file__).resolve().parents[1] / "src" / "ghost_agent"

# ══════════════════════════════════════════════════════════════════════
# 1 — error_line: content is not a failure
# ══════════════════════════════════════════════════════════════════════
# Verbatim heads of REAL corpus results the old rule listed as errors.
_CONTENT_HITS = [
    ("web_search", "### 1. Nucleus market\nAn illegal darkweb marketplace has given up "
                   "selling guns in the wake of the Paris terror attacks.\n[Source: https://a/]"),
    ("web_search", "### 1. PostgreSQL version\nUse \"locate bin/postgres\" if not found. "
                   "postgres (PostgreSQL) 9.6.1\n[Source: https://b/]"),
    ("web_search", "### 1. Rwanda\nThe Rwandan genocide occurred from 7 April to 19 July 1994.\n"
                   "[Source: https://c/]\n\n[Note: these search engines do not support site "
                   "restrictions — `site:x.com` was removed]"),
    ("browser", "--- BROWSER RESULT ---\nSTATUS: OK\nOP: navigate\nURL: https://x/\n"
                "HTTP_STATUS: 200\nTITLE: t\n--- CONSOLE ---\n• [error] release/19.0:553:1 — "
                "Applying inline style violates the following Content Security Policy directive\n"
                "• [error] favicon.ico:1:1 — Failed to load resource: 404"),
    ("file_system", "def run():\n    try:\n        ...\n    except ValueError as e:\n"
                    "        raise RuntimeError('cannot parse') from e\n"),
    ("manage_projects", '{"events": [{"type": "task_failed", "payload": {"error": "x not found"}}]}'),
    ("introspect", "(failure ticks flow; no fail-only lesson yet — expected at a high turn pass rate)"),
    ("manage_composed_skills", "▎ Still running at 90s — promoted to background job job-6c30d9c2. "
                               "Do NOT re-run this command — it was NOT killed and is still working."),
    ("manage_services", "Service 'web' RUNNING (pid 4242).\nIn-sandbox URL: http://127.0.0.1:5000\n"
                        "⚠ NOT reachable from the host: port 5000 is not published by the sandbox "
                        "— the user's browser cannot open it and serve-remote would map nothing."),
    ("vision_analysis", "The photo shows a road sign reading ERROR 404 painted on a wall."),
]


@pytest.mark.parametrize("tool,text", _CONTENT_HITS)
def test_content_with_failure_words_is_not_an_error(tool, text):
    assert error_line(text, tool=tool) == ""
    assert error_line_fingerprint(text, tool=tool) == ""


@pytest.mark.parametrize("tool,text,expected", [
    # what a CONTENT tool's failure looks like: a head the tool writes…
    ("web_search", "ERROR: No results found. The internet might be blocking your request.",
     "ERROR: No results found. The internet might be blocking your request."),
    ("file_system", "Error: could not write 'broken.js'.", "Error: could not write 'broken.js'."),
    ("manage_services", "Error: port 8080 is reserved (agent API). Nothing was stopped.",
     "Error: port 8080 is reserved (agent API). Nothing was stopped."),
    ("recall", "SYSTEM ERROR: The 'query' parameter is MANDATORY.",
     "SYSTEM ERROR: The 'query' parameter is MANDATORY."),
    ("browser", "--- BROWSER RESULT ---\nSTATUS: BLOCKED (HTTP 404)\nOP: navigate\nHINT: …",
     "STATUS: BLOCKED (HTTP 404)"),
    ("browser", "--- BROWSER RESULT ---\nSTATUS: ERROR\nPage.goto: net::ERR_CONNECTION_REFUSED",
     "Page.goto: net::ERR_CONNECTION_REFUSED"),          # the message under the status
    ("browser", "--- BROWSER RESULT ---\nSTATUS: ERROR", "STATUS: ERROR"),
    ("web_search", "CRITICAL ERROR: 'ddgs' library is missing. Search is impossible.",
     "CRITICAL ERROR: 'ddgs' library is missing. Search is impossible."),
    ("image_generation", "ERROR generating image: node unreachable", "ERROR generating image: node unreachable"),
])
def test_a_content_tools_failure_head_is_its_error_line(tool, text, expected):
    assert error_line(text, tool=tool) == expected


# The turn loop records every rejected result as `[FAILURE BANNER] <label>\n<result>`
# (agent.py, the `_failure_shaped` branch) — the shapes the R1 review found
# the digest printing as `[FAILURE BANNER] Traceback (most recent call last):`.
_BANNERED = [
    # the producer's verbatim statement (services.py: "… it likely FAILED to
    # bind …") IS the tool's own failure line; the log tail follows it
    ("manage_services", ToolOutcome.failed(
        "[FAILURE BANNER] Traceback (most recent call last):\n"
        "Service 'web' started (pid 7) but nothing is listening on port 8100 after ~6s — "
        "it likely FAILED to bind (missing dependency, a crash on import, or the app binds a "
        "different port). Check the log below BEFORE trying to reach it:\n"
        "--- web log tail ---\nTraceback (most recent call last):\n  File app.py, line 1\n"
        "ModuleNotFoundError: No module named 'flask'", world_changed=True,
        reason_code="service_failed_to_bind"),
     "Service 'web' started (pid 7) but nothing is listening on port 8100 after ~6s — it likely "
     "FAILED to bind (missing dependency, a crash on import, or the app binds a different port). "
     "Check the log below BEFORE trying to reach it:"),
    # the formatter's verbatim interact shape (tools/browser.py): the
    # `ACTIONS:` summary and a `NOTE:` nudge come BEFORE the per-action lines
    # (R3 review: the first failure-word line was the summary)
    ("browser", ToolOutcome.partial(
        "[FAILURE BANNER] --- BROWSER RESULT ---\n--- BROWSER RESULT ---\nSTATUS: OK\n"
        "OP: interact\nNOTE: You've navigated to this page 3 times this session — a change or "
        "an error, not another load, is the next step.\nFINAL_URL: https://x/form\n"
        "FINAL_TITLE: Form\nACTIONS: 2 OK, 2 errors (of 4 total)\n"
        "CONSOLE (1 error/warning):\n• [error] favicon.ico — Failed to load resource: 404\n"
        "--- PER-ACTION RESULTS ---\n  [0] OK fill #name\n  [1] OK fill #email\n"
        "  [2] ERR click: TimeoutError: Page.click: Timeout 30000ms exceeded waiting for #go\n"
        "  [3] ERR click: TimeoutError: Page.click: Timeout 30000ms exceeded waiting for #ok",
        world_changed=True),
     "[2] ERR click: TimeoutError: Page.click: Timeout 30000ms exceeded waiting for #go"),  # the FIRST
    # the runner's implicit-goto abort: index -1 (R4 review)
    ("browser", ToolOutcome.failed(
        "--- BROWSER RESULT ---\nSTATUS: OK\nOP: interact\nFINAL_URL: https://example.org/error\n"
        "FINAL_TITLE: \nACTIONS: 0 OK, 1 error (of 1 total)\n"
        "⚠ SEQUENCE ABORTED: initial_goto_failed. Remaining actions were NOT executed because the "
        "initial navigation failed — page.click/fill/extract on an error page would have just "
        "timed out one-by-one. Fix the URL and retry the whole interact call.\n"
        "--- PER-ACTION RESULTS ---\n"
        "  [-1] ERR goto: initial navigation failed (TimeoutError): Page.goto: Timeout 30000ms exceeded.",
        world_changed=False, reason_code="browser_interact_failed"),
     "[-1] ERR goto: initial navigation failed (TimeoutError): Page.goto: Timeout 30000ms exceeded."),
    ("file_system", ToolOutcome.rejected(
        "[FAILURE BANNER] SYSTEM INSTRUCTION: The search block was NOT found in 'a.py'.\n"
        "SYSTEM INSTRUCTION: The search block was NOT found in 'a.py'. Re-read the file."),
     "SYSTEM INSTRUCTION: The search block was NOT found in 'a.py'. Re-read the file."),
    # a bannered plain string (no status object) reads the same way
    ("browser", "[FAILURE BANNER] Error 1006 Ray ID: a3bf\n--- BROWSER RESULT ---\nSTATUS: OK\n"
                "HTTP_STATUS: 500\nTITLE: Error 1006 Ray ID: a3bf",
     "TITLE: Error 1006 Ray ID: a3bf"),
    # a body with no failure word at all: its first line, never the banner
    ("file_system", "[FAILURE BANNER] REJECTED\nThe file is outside the workspace and was left alone.",
     "The file is outside the workspace and was left alone."),
    # R2 review: real rows carry the tool's HINT prose, console bullets and
    # file snippets AFTER the statement — the last failure-word line named
    # those. A content tool's own statement is the first such line.
    ("browser", ToolOutcome.failed(
        "[FAILURE BANNER] --- BROWSER RESULT ---\n--- BROWSER RESULT ---\nSTATUS: ERROR\n"
        "Runner failed (exit 1): ValueError: selector 'p' did not match any element\n"
        "--- HINT ---\nIf this is a navigation timeout, try wait_until='domcontentloaded' "
        "or raise timeout_ms.", world_changed=False, reason_code="browser_runner_failed"),
     "Runner failed (exit 1): ValueError: selector 'p' did not match any element"),
    ("browser", ToolOutcome.failed(
        "[FAILURE BANNER] --- BROWSER RESULT ---\n--- BROWSER RESULT ---\n"
        "STATUS: BLOCKED (HTTP 403 — bot challenge)\nOP: navigate\nURL: https://x/\n"
        "--- CONSOLE ---\n• [error] security/revolut-1:1:1 — Failed to load resource: 403",
        world_changed=False, reason_code="browser_blocked"),
     "STATUS: BLOCKED (HTTP 403 — bot challenge)"),
    ("file_system", ToolOutcome.rejected(
        "SYSTEM INSTRUCTION: The search block was NOT found in 'app.py'. Re-read.\n"
        " 1. check the indentation\n 2. check the quotes\n"
        " 3. If two replace attempts have already failed, rewrite the file.\n"
        "--- current file around the block ---\n"
        ">>>  96:  return jsonify(error=\"date must match\"), 400"),
     "SYSTEM INSTRUCTION: The search block was NOT found in 'app.py'. Re-read."),
    # a PARTIAL run is a declared failure too (a browser interact with a failed op)
    ("browser", ToolOutcome.partial(
        "--- BROWSER RESULT ---\nSTATUS: OK\nOP: interact\nFINAL_URL: https://x/\nFINAL_TITLE: t\n"
        "ACTIONS: 1 OK, 1 error (of 2 total)\n"
        "--- PER-ACTION RESULTS ---\n  [0] OK fill #q\n  [3] ERR click: Timeout 3000ms exceeded",
        world_changed=True),
     "[3] ERR click: Timeout 3000ms exceeded"),
]


@pytest.mark.parametrize("tool,res,expected", _BANNERED)
def test_a_bannered_result_names_the_failure_inside_it(tool, res, expected):
    """Never the loop's own marker, never the label: the last failure line
    of the result, as the pre-§4KT digest showed it."""
    line = error_line(res, tool=tool)
    assert line == expected
    assert "[FAILURE BANNER]" not in line


def test_the_digest_never_prints_the_loops_marker():
    """Each bannered row through the real digest (one at a time: the digest
    lists its five most frequent errors): the tool's own statement, never
    the marker, a hint sentence, a console bullet or a file snippet."""
    from ghost_agent.core.strikes import exception_signature, normalise_volatile
    for tool, res, expected in _BANNERED:
        out = evidence_digest([{"name": tool, "content": res}], ask="x")
        assert "[FAILURE BANNER]" not in out
        errs = out.split("Distinct errors hit:")[1].split("\n")[0]
        key = " ".join(exception_signature(normalise_volatile(expected)).split())
        assert key[:28] in errs, (expected, errs)
        assert "HINT" not in errs and "[error]" not in errs and "jsonify" not in errs, errs


def test_a_declared_program_failure_still_names_its_last_line():
    """Program output keeps the LAST rule under a declared status too: a
    probe prints its verdict after its attempts."""
    res = ToolOutcome.failed("ERR: attempt 1: SpecError [pl]\nERR: attempt 2: SpecError [pl]\n"
                             "verdict: all attempts failed", world_changed=False,
                             reason_code="shell_failed")
    assert error_line(res, tool="execute") == "verdict: all attempts failed"


def test_a_declared_status_wins_over_any_prose():
    """A `ToolOutcome` that says FAILED is an error whatever it says; one that
    says OK with the same text is not (content tools) — the outcome-consumers
    R3 rule, now on both sides."""
    text = "Service 'web' started (pid 7) but nothing is listening on port 8100 after ~6s"
    failed = ToolOutcome.failed(text, world_changed=True, reason_code="service_failed_to_bind")
    assert error_line(failed, tool="manage_services") == text
    assert error_line(ToolOutcome.ok(text), tool="manage_services") == ""
    assert error_line(ToolOutcome.rejected("Error: no service named 'x'"), tool="manage_services") \
        == "Error: no service named 'x'"


@pytest.mark.parametrize("tool", sorted(PROGRAM_OUTPUT_TOOLS) + [None])
@pytest.mark.parametrize("text,expected", [
    # §4IB: the script caught its own error and printed it under exit 0
    ("Loading spec…\ncannot build grid without 'type'\nEXIT CODE: 0", "cannot build grid without 'type'"),
    ("Traceback (most recent call last):\n  File x\nModuleNotFoundError: No module named 'flask'\nEXIT CODE: 0",
     "ModuleNotFoundError: No module named 'flask'"),
    ("arkanoid: ✗ FAILED\nEXIT CODE: 0", "arkanoid: ✗ FAILED"),
    ("ERR: [pl] SpecError\nEXIT CODE: 0", "ERR: [pl] SpecError"),
    ("all 12 tests passed\nEXIT CODE: 0", ""),
    # a CamelCase exception name is an error; a word that merely ENDS in
    # "error" is not (§4KT: "terror" matched under IGNORECASE)
    ("ValueError: bad literal\nEXIT CODE: 0", "ValueError: bad literal"),
    ("the Paris terror attacks of 2015\nEXIT CODE: 0", ""),
    ("Terror in the streets\nEXIT CODE: 0", ""),
    ("[error] release/19.0 — CSP violation", "[error] release/19.0 — CSP violation"),
    ("error: BaseException | None = None", "error: BaseException | None = None"),
    ("ERROR 403 Forbidden", "ERROR 403 Forbidden"),
])
def test_program_output_is_scanned_for_its_printed_error(tool, text, expected):
    """The LAST failure line wins for program output — and an unnamed
    caller (legacy) is read as program output."""
    kw = {} if tool is None else {"tool": tool}
    assert error_line(text, **kw) == expected


def test_a_jobs_result_is_a_programs_output():
    """A detached job's output is program output (the CLI's own lines)."""
    assert error_line("[CLI] Exiting due to transcription failure.", tool="jobs") \
        == "[CLI] Exiting due to transcription failure."
    assert "jobs" in PROGRAM_OUTPUT_TOOLS and "execute" in PROGRAM_OUTPUT_TOOLS
    assert "web_search" not in PROGRAM_OUTPUT_TOOLS and "browser" not in PROGRAM_OUTPUT_TOOLS


def test_program_output_tools_are_registered_tools():
    """R1 review: the set named an `execute_python` no registry has."""
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    names = {d["function"]["name"] for d in TOOL_DEFINITIONS}
    for t in PROGRAM_OUTPUT_TOOLS:
        assert t in names, t


def test_a_url_is_not_part_of_the_error():
    """R3 review: one `net::ERR_HTTP2_PROTOCOL_ERROR` keyed as eleven, one
    per page — a URL is the thing that was tried."""
    from ghost_agent.core.strikes import error_line_fingerprint
    a = "--- BROWSER RESULT ---\nSTATUS: ERROR\nRunner failed (exit 1): Error: Page.goto: net::ERR_HTTP2_PROTOCOL_ERROR at https://a.it/en/x"
    b = "--- BROWSER RESULT ---\nSTATUS: ERROR\nRunner failed (exit 1): Error: Page.goto: net::ERR_HTTP2_PROTOCOL_ERROR at https://b.org/y?z=1"
    assert error_line_fingerprint(a, tool="browser") == error_line_fingerprint(b, tool="browser") != ""
    # file:// too, and a quoted URL followed by another quoted literal
    # (R4 review: `\S+` ate the closing quote and split what the quote rule joined)
    c = "--- BROWSER RESULT ---\nSTATUS: ERROR\nPage.goto: net::ERR_FILE_NOT_FOUND at file:///workspace/projects/a/index.html"
    d = "--- BROWSER RESULT ---\nSTATUS: ERROR\nPage.goto: net::ERR_FILE_NOT_FOUND at file:///workspace/projects/b/index.html"
    assert error_line_fingerprint(c, tool="browser") == error_line_fingerprint(d, tool="browser") != ""
    e = "ERR 'https://a.com/x' 'p1': not found"
    f = "ERR 'https://b.com/y' 'p2': not found"
    assert error_line_fingerprint(e) == error_line_fingerprint(f) != ""
    g = 'ERR "https://a.com/x" "p1": not found'
    h = 'ERR "https://b.com/y" "p2": not found'
    assert error_line_fingerprint(g) == error_line_fingerprint(h) != ""


@pytest.mark.parametrize("line,signature", [
    # a named class wins, wherever a bare `Error` label sits before it (R5)
    ("Error processing item_16: ValueError: bad literal", "ValueError: bad literal"),
    ("Syntax Error Detected: SyntaxError: invalid syntax", "SyntaxError: invalid syntax"),
    ("error occurred while loading x=16: RuntimeError: y", "RuntimeError: y"),
    # the bare class only when none is named; case-sensitive, whole word
    ("[0] ERR goto: Error: Page.goto: net::X at URL", "Error: Page.goto: net::X at URL"),
    ("[-1] ERR goto: initial navigation failed (Error): Page.goto: net::X at URL",
     "Error: Page.goto: net::X at URL"),                    # the runner's parenthesised class
    ("Errors: 3 of 9", "Errors: 3 of 9"),
    ("job done with 3 Errors: none fatal", "job done with 3 Errors: none fatal"),   # whole word
    ("the Paris terror attacks", "the Paris terror attacks"),
    ("error: something", "error: something"),
    ("an error occurred while loading x=16", "an error occurred while loading x=16"),  # case
])
def test_exception_signature(line, signature):
    from ghost_agent.core.strikes import exception_signature
    assert exception_signature(line) == signature


def test_one_action_error_at_two_indices_is_one_digest_key():
    """R4 review: Playwright's bare `Error` class has no prefix, so the
    exception signature kept the `[n] ERR goto:` label and one error at two
    indices was two "distinct errors"."""
    from ghost_agent.core.strikes import exception_signature
    a = exception_signature("[0] ERR goto: Error: Page.goto: net::ERR_CONNECTION_REFUSED at URL")
    b = exception_signature("[2] ERR goto: Error: Page.goto: net::ERR_CONNECTION_REFUSED at URL")
    assert a == b == "Error: Page.goto: net::ERR_CONNECTION_REFUSED at URL"


def test_the_exception_name_alternative_is_case_sensitive():
    assert _ERROR_LINE_RE.search("TimeoutError") and _ERROR_LINE_RE.search("Exception")
    assert _ERROR_LINE_RE.search("an error occurred") and _ERROR_LINE_RE.search("[ERROR]")
    assert not _ERROR_LINE_RE.search("terror") and not _ERROR_LINE_RE.search("Terror")
    assert not _ERROR_LINE_RE.search("mirrors")


def test_evidence_digest_lists_only_failures_the_request_hit():
    """At the call site: the digest gets the tool NAME and the outcome
    OBJECT. A search whose snippet says "not found" is not an error hit;
    a search the tool declared failed is; a script's printed error is."""
    runs = [
        {"name": "web_search", "content": ToolOutcome.ok(_CONTENT_HITS[1][1])},
        {"name": "web_search", "content": ToolOutcome.failed(
            "ERROR: No results found. Try a different query.", world_changed=False,
            reason_code="search_empty")},
        {"name": "execute", "content": ToolOutcome.ok(
            "Traceback (most recent call last):\nModuleNotFoundError: No module named 'flask'\nEXIT CODE: 0")},
        {"name": "browser", "content": ToolOutcome.ok(_CONTENT_HITS[3][1])},
        # a declared failure whose text has no failure head: only the STATUS says so
        {"name": "manage_services", "content": ToolOutcome.failed(
            "Service 'web' started (pid 7) but nothing is listening on port 8100 after ~6s",
            world_changed=True, reason_code="service_failed_to_bind")},
    ]
    out = evidence_digest(runs, ask="find the postgres version")
    assert "Distinct errors hit:" in out
    errs = out.split("Distinct errors hit:")[1].split("\n")[0]
    assert "ModuleNotFoundError" in errs and "No results found" in errs
    assert "nothing is listening on port 8100" in errs
    assert "not found. postgres" not in errs and "CSP" not in errs and "TimeoutError" not in errs


def test_every_error_line_call_names_its_tool():
    """Enumeration: no consumer may fall back to the legacy (program-output)
    scan by omitting the tool."""
    found = 0
    for path in SRC.rglob("*.py"):
        if path.name == "strikes.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", getattr(node.func, "id", ""))
                    in ("error_line", "error_line_fingerprint")):
                found += 1
                assert "tool" in {k.arg for k in node.keywords}, f"{path.name}:{node.lineno}"
    assert found >= 3, "the walk found no callers — the enumeration is blind"


# ══════════════════════════════════════════════════════════════════════
# 2 — the recorder: temperature, and the streamed Turn Outcome line
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("payload,expected", [
    ({"temperature": 0.6}, 0.6), ({"temperature": 1}, 1.0), ({"temperature": 0.0}, 0.0),
    ({"messages": []}, None), ({}, None), (None, None), ("x", None),
    ({"temperature": "0.6"}, None), ({"temperature": True}, None),
])
def test_sampling_temperature(payload, expected):
    assert GhostAgent._sampling_temperature(payload) == expected


def _collector():
    coll = MagicMock()
    coll.append.return_value = "/tmp/traj.jsonl"
    coll.redact.side_effect = lambda s: s
    return coll


async def test_a_recorded_turn_carries_the_temperature_it_ran_at(tmp_path):
    """Through a real non-streamed turn: the row's temperature is the one
    the last request was sent with (the log's `Temp N.NN`), not 0.0."""
    ctx = _chat_context(tmp_path)
    ctx.trajectory_collector = _collector()

    async def empty(*a, **kw):
        return (None, {"name": "execute_python"})

    agent, logged = await _turn(ctx, verdict=empty, req_id="req4kt-t1")
    temps = [float(m.group(1)) for t, c in logged
             for m in [re.search(r"Temp (\d+\.\d+)", c)] if t == "LLM Request" and m]
    assert temps, "no request was logged — the pin is blind"
    rows = [c.args[0] for c in ctx.trajectory_collector.append.call_args_list]
    assert rows, "no trajectory was recorded — the pin is blind"
    assert rows[-1].temperature == temps[-1] != 0.0


async def test_a_non_streamed_turn_still_prints_its_reply(tmp_path):
    """The emitter prints `Final Reply` for finalize (the drain prints its
    own, before the record)."""
    ctx = _chat_context(tmp_path)

    async def empty(*a, **kw):
        return (None, {"name": "execute_python"})

    agent, logged = await _turn(ctx, verdict=empty, req_id="req4kt-fr")
    outcomes = [i for i, (t, c) in enumerate(logged) if t == "Turn Outcome"]
    replies = [i for i, (t, c) in enumerate(logged) if t == "Final Reply"]
    assert len(outcomes) == 1 and len(replies) == 1
    assert replies[0] > outcomes[0]                 # the reply follows the line


def test_a_shape_failed_row_relabels_the_calibration_sample():
    """§4EE R3's third mirror, now inside the shared emitter: a shape
    FAILED re-labels the request's calibration sample."""
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = MagicMock()
    agent.context._recent_turn_outcome = None          # a MagicMock would auto-vivify the ring
    agent.context.calibration_tracker = MagicMock()
    agent._turn_confidence = lambda rid: None
    with patch("ghost_agent.core.agent.pretty_log"):
        agent._emit_turn_outcome_line(
            req_id="r-shape", trajectory_id="t-shape", final_content="raw dump",
            tools=[], execution_failure_count=0, exec_terminal=False,
            unacked_total_failure=False, budget_exhausted=False, shape_failed=True)
    agent.context.calibration_tracker.record_shape_failure.assert_called_once_with("r-shape")
    assert agent.context._recent_turn_outcome["t-shape"]["state"] == "failed"
    agent.context.calibration_tracker.reset_mock()
    with patch("ghost_agent.core.agent.pretty_log"):
        agent._emit_turn_outcome_line(
            req_id="r-ok", trajectory_id="t-ok", final_content="fine", tools=[],
            execution_failure_count=0, exec_terminal=False, unacked_total_failure=False,
            budget_exhausted=False, shape_failed=False)
    agent.context.calibration_tracker.record_shape_failure.assert_not_called()


def test_an_unknown_temperature_keeps_the_schema_default(tmp_path):
    ctx = _chat_context(tmp_path)
    ctx.trajectory_collector = _collector()
    agent = GhostAgent(ctx)
    agent._record_turn_trajectory(messages=[{"role": "user", "content": "x"}],
                                  final_content="y", req_id="r", model="m")
    assert ctx.trajectory_collector.append.call_args.args[0].temperature == 0.0
    agent._record_turn_trajectory(messages=[{"role": "user", "content": "x"}],
                                  final_content="y", req_id="r", model="m", temperature=0.6)
    assert ctx.trajectory_collector.append.call_args.args[0].temperature == 0.6


async def _drain(agent, deltas, **state_overrides):
    async def final_stream(payload, use_coding=False):
        for d in deltas:
            yield sse({"content": d})
        yield b"data: [DONE]\n\n"
    agent.context.llm_client.stream_chat_completion = final_stream
    reg = MagicMock()
    reg.is_cancelled.return_value = False
    ss = _make_stream_state(reg)
    if state_overrides:
        ss = StreamState(**{**ss.__dict__, **state_overrides})
    gen, _, _ = agent._stream_final_generation(ss)
    return [c async for c in gen]


def _streamed(monkeypatch):
    a = make_stream_agent()
    a._record_calibration_safe = AsyncMock()
    a._record_turn_trajectory = MagicMock(return_value=None)
    logged = []
    monkeypatch.setattr(agent_mod, "pretty_log",
                        lambda title, content=None, **kw: logged.append((title, str(content), kw)))
    return a, logged


async def test_a_streamed_turn_records_its_temperature(monkeypatch):
    a, logged = _streamed(monkeypatch)
    await _drain(a, ["The answer."], payload={"messages": [], "temperature": 0.6})
    assert a._record_turn_trajectory.call_args.kwargs["temperature"] == 0.6


async def test_a_streamed_turn_prints_a_turn_outcome_line(monkeypatch):
    """The line finalize prints for every other turn, from the same
    emitter — state, confidence, tools, chars — and a ring entry a late
    verdict can correct."""
    a, logged = _streamed(monkeypatch)
    a._set_last_confidence(MagicMock(composite=0.71), "rid00001")
    tools = [{"role": "tool", "name": "execute", "content": "EXIT CODE: 0\nok"}]
    await _drain(a, ["The streamed ", "answer."], stream_tools_snapshot=tools,
                 current_trajectory_id="traj-4kt-1")
    lines = [(c, kw) for t, c, kw in logged if t == "Turn Outcome"]
    assert len(lines) == 1, logged
    text, kw = lines[0]
    assert text.startswith("ok") and "confidence 0.71" in text
    assert "tools: execute" in text and "20 chars" in text
    assert kw.get("level", "INFO") == "INFO"
    ring = a.context._recent_turn_outcome
    assert ring["traj-4kt-1"]["state"] == "ok" and ring["traj-4kt-1"]["tools"] == ["execute"]
    # the reply itself is printed ONCE (before the record), not again by the emitter
    assert sum(1 for t, _, _ in logged if t == "Final Reply") == 1


async def test_a_streamed_turn_that_ended_in_failure_says_so(monkeypatch):
    a, logged = _streamed(monkeypatch)
    tools = [{"role": "tool", "name": "execute", "content": "EXIT CODE: 1\nboom"}]
    await _drain(a, ["I could not run it; the script crashed with boom."],
                 stream_tools_snapshot=tools, execution_failure_count=2,
                 last_was_failure=True, current_trajectory_id="traj-4kt-2")
    text, kw = [(c, kw) for t, c, kw in logged if t == "Turn Outcome"][0]
    assert text.startswith("failed"), text
    assert kw.get("level") == "WARNING"
    assert a.context._recent_turn_outcome["traj-4kt-2"]["exec_terminal"] is True


async def test_a_streamed_shape_failed_row_prints_failed(monkeypatch):
    """R1 review: the drain's `shape_failed` input was unpinned — a streamed
    turn whose recorded row the shape heuristics marked FAILED printed
    `ok` (the §4EE F2 defect, on the web-UI path)."""
    from ghost_agent.distill.schema import Outcome
    a, logged = _streamed(monkeypatch)
    row = MagicMock()
    row.outcome = Outcome.FAILED.value
    row.failure_reason = "raw tool dump pasted as the answer"
    a._record_turn_trajectory = MagicMock(return_value=row)
    await _drain(a, ["<tool_call>{}</tool_call> EXIT CODE: 0"], current_trajectory_id="traj-4kt-4")
    text, kw = [(c, kw) for t, c, kw in logged if t == "Turn Outcome"][0]
    assert text.startswith("failed") and kw.get("level") == "WARNING"
    assert a.context._recent_turn_outcome["traj-4kt-4"]["shape_failed"] is True


async def test_a_streamed_unacknowledged_total_failure_is_carried_to_the_ring(monkeypatch):
    """…and `unacked_total_failure`: a late PASS on an all-tools-failed,
    unacknowledged streamed turn must not print `CORRECTED failed →
    verified` (§4EE 2b)."""
    a, logged = _streamed(monkeypatch)
    tools = [{"role": "tool", "name": "execute", "content": "EXIT CODE: 1\nboom"},
             {"role": "tool", "name": "execute", "content": "EXIT CODE: 1\nboom again"}]
    await _drain(a, ["Done — the report is ready and everything worked."],
                 stream_tools_snapshot=tools, execution_failure_count=2,
                 last_was_failure=True, current_trajectory_id="traj-4kt-5")
    entry = a.context._recent_turn_outcome["traj-4kt-5"]
    assert entry["exec_terminal"] is True and entry["unacked_total_failure"] is True


async def test_the_streamed_line_follows_the_calibration_record(monkeypatch):
    """R1 review: the emitter ran BEFORE the calibration record, where its
    shape re-label finds no `turn` sample and the confidence stamp is not
    yet written — the reverse of finalize."""
    a, logged = _streamed(monkeypatch)
    order = []
    a._record_calibration_safe = AsyncMock(side_effect=lambda **kw: order.append("calibration"))
    real = a._emit_turn_outcome_line
    a._emit_turn_outcome_line = lambda **kw: (order.append("outcome"), real(**kw))[1]
    await _drain(a, ["x"], current_trajectory_id="traj-4kt-6")
    assert order == ["calibration", "outcome"]


async def test_the_streamed_line_survives_a_recorder_fault(monkeypatch):
    """The emit sits outside the RECORD's try: a recorder exception must
    not drop the line and the ring entry (R1 review)."""
    a, logged = _streamed(monkeypatch)
    a._record_turn_trajectory = MagicMock(side_effect=RuntimeError("disk full"))
    await _drain(a, ["fine"], current_trajectory_id="traj-4kt-7")
    assert any(t == "Turn Outcome" for t, _, _ in logged), logged
    assert "traj-4kt-7" in a.context._recent_turn_outcome


def test_the_stream_state_has_no_budget_flag():
    """A streamed turn returns from inside the loop; the for-else budget
    flag can never be set for it (R1 review: the plumbing was dead)."""
    import dataclasses
    assert "turn_budget_exhausted" not in {f.name for f in dataclasses.fields(StreamState)}


def test_one_turn_outcome_emitter_reached_from_both_paths():
    """Enumeration: the plain `Turn Outcome` line is printed by ONE method,
    and both the finalize path and the streamed drain call it — the late
    CORRECTED line is the only other writer of that title."""
    tree = ast.parse((SRC / "core" / "agent.py").read_text(encoding="utf-8"))
    owners, callers = set(), set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for call in ast.walk(fn):
            if not isinstance(call, ast.Call):
                continue
            if (call.args and isinstance(call.args[0], ast.Constant)
                    and call.args[0].value == "Turn Outcome"):
                owners.add(fn.name)
            if getattr(call.func, "attr", "") == "_emit_turn_outcome_line":
                callers.add(fn.name)
    assert "_emit_turn_outcome_line" in owners
    assert len(owners) == 2                      # the emitter + the late correction
    assert {"_finalize_and_return", "_stream_final_generation"} <= callers, callers
