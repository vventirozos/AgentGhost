"""§4RA: a coding request's deadline moves out while code is still being
written — base budget (GHOST_MAX_REQUEST_S, never past the client's timeout),
15-minute steps EARNED ON READ by coding progress in the last 30 minutes,
a ceiling (GHOST_MAX_REQUEST_EXTENDED_S, never past the client's timeout,
only for a client that sent one)."""
import ast
import pathlib
import time

import pytest

from ghost_agent.utils import logging as L
from ghost_agent.core.agent import (client_deadline_line, deadline_needs_report,
                                    effective_report_floor, is_code_progress)

ROOT = pathlib.Path(__file__).resolve().parents[1]
RID = "4ra0test"
STEP = L.EXTENSION_STEP_S


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("GHOST_MAX_REQUEST_S", "3600")
    monkeypatch.setenv("GHOST_MAX_REQUEST_EXTENDED_S", "10800")
    tok = L.client_deadline_context.set(10800.0)          # the web interface in production
    yield monkeypatch
    L.client_deadline_context.reset(tok)


def _open(elapsed, progress_ago=None):
    now = time.monotonic()
    st = {"started": now - elapsed, "origin": None}
    if progress_ago is not None:
        st["progress_t"] = now - progress_ago
        st["progress_what"] = "wrote app.js"
    with L._REQ_STATE_LOCK:
        L._REQ_STATE[RID] = st


@pytest.fixture
def req():
    yield _open
    with L._REQ_STATE_LOCK:
        L._REQ_STATE.pop(RID, None)


def _client(s):
    L.client_deadline_context.set(float(s))


@pytest.mark.parametrize("client,base", [(0, 3600), (1800, 1800), (10800, 3600), (3600, 3600)])
def test_the_base_is_the_server_cap_never_past_the_client(env, client, base):
    _client(client)
    assert L.base_deadline_s(RID) == base


def test_background_work_keeps_the_client_deadline_only_and_is_never_extended(env):
    _client(0)
    assert L.base_deadline_s("sched-x") == 0.0
    _client(900)
    assert L.base_deadline_s("sched-x") == 900.0
    assert L.extension_ceiling_s("sched-x") == 900.0


@pytest.mark.parametrize("ext,client,ceil", [("0", 10800, 3600), ("", 10800, 3600), ("bad", 10800, 3600),
                                             ("10800", 0, 3600),          # no header: never extended (CLI, Slack, scripts)
                                             ("10800", 10800, 10800), ("10800", 7200, 7200),
                                             ("10800", 3600, 3600), ("1200", 10800, 3600)])
def test_the_ceiling(env, ext, client, ceil):
    env.setenv("GHOST_MAX_REQUEST_EXTENDED_S", ext)
    _client(client)
    assert L.extension_ceiling_s(RID) == ceil


def test_no_progress_no_extension(env, req):
    req(3500)
    assert L.request_deadline_s(RID) == 3600
    assert L.request_extension_s(RID) == 0


def test_stale_progress_no_extension(env, req):
    req(3500, progress_ago=L.PROGRESS_WINDOW_S + 5)
    assert L.request_deadline_s(RID) == 3600


def test_far_from_the_deadline_no_extension_yet(env, req):
    req(3600 - STEP - 10, progress_ago=5)
    assert L.request_deadline_s(RID) == 3600


def test_near_the_deadline_with_progress_one_step(env, req):
    req(3600 - STEP + 10, progress_ago=5)
    assert L.request_deadline_s(RID) == 3600 + STEP
    assert L.request_extension_s(RID) == STEP
    assert L.request_deadline_s(RID) == 3600 + STEP              # reading again grants nothing more


def test_a_long_turn_that_overshot_gets_enough_steps_at_once(env, req):
    # review M2: a 27B write turn that ran ~300 s past the deadline
    req(3900, progress_ago=10)
    d = L.request_deadline_s(RID)
    assert d - 3900 >= STEP                                      # more than a step left, not a used-up grant
    rem = L.request_remaining_s(RID)
    assert not deadline_needs_report(rem, effective_report_floor(d), False, False)


def test_progress_from_a_whole_long_turn_ago_still_counts(env, req):
    req(3500, progress_ago=1000)                                 # the write landed one long turn ago
    assert L.request_deadline_s(RID) > 3600


def test_never_past_the_ceiling(env, req):
    req(10700, progress_ago=5)
    assert L.request_deadline_s(RID) == 10800
    assert L.request_extension_s(RID) == 10800 - 3600            # battery R9: the stored total lands exactly
    env.setenv("GHOST_MAX_REQUEST_EXTENDED_S", "5000")          # a room that is not a whole number of steps
    with L._REQ_STATE_LOCK:
        L._REQ_STATE[RID]["extended_s"] = 0.0
    assert L.request_deadline_s(RID) == 5000
    assert L.request_extension_s(RID) == 1400
    env.setenv("GHOST_MAX_REQUEST_EXTENDED_S", "10800")
    req(11500, progress_ago=5)
    assert L.request_deadline_s(RID) == 10800
    assert L.request_remaining_s(RID) < 0


def test_never_past_the_client(env, req):
    _client(5000)
    req(4900, progress_ago=5)
    assert L.request_deadline_s(RID) == 5000


def test_a_client_without_a_timeout_is_never_extended(env, req):
    _client(0)
    req(3500, progress_ago=5)
    assert L.request_deadline_s(RID) == 3600


def test_a_lowered_ceiling_caps_what_was_already_earned(env, req):
    req(3500, progress_ago=5)
    assert L.request_deadline_s(RID) == 3600 + STEP
    env.setenv("GHOST_MAX_REQUEST_EXTENDED_S", "4000")
    assert L.request_deadline_s(RID) == 4000


def test_every_in_turn_check_sees_the_extension(env, req):
    # review M3: the preflight / image / continuation readers all go through request_remaining_s
    req(3600 - 400, progress_ago=5)
    assert L.request_remaining_s(RID) > STEP


def test_note_code_progress_marks_the_open_request_only(env, req):
    req(3500)
    L.note_code_progress(RID, "wrote x.js")
    L.note_code_progress("never-opened", "wrote y.js")
    assert L.request_progress_what(RID) == "wrote x.js"
    assert L.request_progress_what("never-opened") == ""
    assert L.request_deadline_s(RID) == 3600 + STEP


def test_the_extension_ends_with_the_request(env, req):
    req(3500, progress_ago=5)
    L.request_deadline_s(RID)
    with L._REQ_STATE_LOCK:
        L._REQ_STATE.pop(RID)
    assert L.request_extension_s(RID) == 0.0


def test_no_deadline_at_all_stays_none(env):
    _client(0)
    env.setenv("GHOST_MAX_REQUEST_S", "0")
    assert L.request_deadline_s(RID) == 0.0
    assert L.request_remaining_s(RID) is None


@pytest.mark.parametrize("fname,op,target,err,ran,want", [
    ("file_system", "write", "webos/app.js", False, False, True),
    ("file_system", "append", "webos/INDEX.HTML", False, False, True),
    ("file_system", "edit", "site/style.css", False, False, True),
    ("file_system", "replace", "a.py", False, False, True),
    ("file_system", "write", "notes.md", False, False, False),
    ("file_system", "write", "report.txt", False, False, False),
    ("file_system", "read", "webos/app.js", False, False, False),
    ("file_system", "", "webos/app.js", False, False, False),        # not mutating
    ("file_system", "delete", "webos/app.js", False, False, False),  # review m2
    ("file_system", "download", "lib.js", False, False, False),
    ("file_system", "move", "a.js", False, False, False),
    ("file_system", "write", "webos/app.js", True, False, False),    # failed / refused
    ("file_system", "write", None, False, False, False),
    ("execute", "", "", False, True, True),
    ("execute", "", "", False, False, False),
    ("execute", "", "", True, True, False),
    ("browser", "", "http://x/index.html", False, False, False),
])
def test_what_counts_as_coding_progress(fname, op, target, err, ran, want):
    assert is_code_progress(fname, op, target, err, ran_written=ran) is want


def test_the_budget_line_names_the_ceiling_only_when_there_is_one():
    line = client_deadline_line(3600, 10800)
    assert line.startswith("time budget 60 min") and "extendable to 180 min" in line
    assert "extendable" not in client_deadline_line(3600)
    assert "extendable" not in client_deadline_line(3600, 3600)


# ── the progress hook, run as the turn loop runs it ───────────────────────── #

def _hook_tree():
    """The §4RA progress block of the dispatch loop, AS PARSED: the `try` in
    `_dispatch_and_process_tool_batch` whose body calls `note_code_progress`."""
    tree = ast.parse((ROOT / "src/ghost_agent/core/agent.py").read_text())
    fn = [f for f in ast.walk(tree) if isinstance(f, ast.AsyncFunctionDef)
          and f.name == "_dispatch_and_process_tool_batch"][0]
    tries = [t for t in ast.walk(fn) if isinstance(t, ast.Try)
             and any(isinstance(n, ast.Attribute) and n.attr == "note_code_progress"
                     for st in t.body for n in ast.walk(st))]
    hook = min(tries, key=lambda t: t.end_lineno - t.lineno)          # the innermost: the hook itself
    assert isinstance(hook.body[0], ast.Assign) and hook.body[0].targets[0].id == "_written"
    return fn, hook


def _hook_code():
    _fn, t = _hook_tree()
    return compile(ast.fix_missing_locations(ast.Module(body=t.body, type_ignores=[])), "<4ra-hook>", "exec")


class _Ctx:
    pass


def _x(cmd):
    import json
    return "execute:" + json.dumps({"command": cmd}, sort_keys=True)       # the production a_hash


# §4SD review: a run earns against writes not made before (``new_writes``);
# ``writes`` counts every write (the futility churn count)
_P, _S = {"progress": True, "writes": 1, "new_writes": 1}, {"progress": False, "writes": 1, "new_writes": 1}
_SEEN = {"progress": True, "writes": 1, "new_writes": 1, "writes_at_earn": 1}   # already counted since the last write
_IMPROVED = {"progress": True, "writes": 1, "new_writes": 1, "writes_at_earn": 1, "improved": True}


@pytest.mark.parametrize("fname,op,mut,target,cmd,written,err,want", [
    ("file_system", "write", True, "web/app.js", "", {}, False, "wrote web/app.js"),
    ("file_system", "append", False, "web/index.html", "", {}, False, "wrote web/index.html"),   # r2: the alias
    ("file_system", "write", True, "notes.md", "", {}, False, ""),
    ("file_system", "write", True, "web/app.js", "", {}, True, ""),
    ("execute", "", True, "", _x("cat app.js"), {}, False, ""),                   # r1 M1: names a file, wrote none
    ("execute", "", True, "", _x("grep -n def main.py"), {"other.py": _P}, False, ""),
    ("execute", "", True, "", _x("python3 main.py"), {"main.py": _P}, False, "ran a file it wrote"),
    ("execute", "", True, "", _x("node --check app.js"), {"app.js": _P}, False, "ran a file it wrote"),
    ("execute", "", True, "", _x("cat app.js"), {"app.js": _S}, False, ""),       # r2 M-E: re-read, no new output
    ("execute", "", True, "", _x("python3 main.py"), {"main.py": _S}, False, ""),  # identical re-run
    ("execute", "", True, "", _x("python3 main.py"), {"main.py": _P}, True, "ran a file it wrote"),  # failing, converging
    ("execute", "", True, "", _x("python3 main.py"), {"main.py": dict(_SEEN)}, False, ""),   # r3 M1: no write since
    ("execute", "", True, "", _x("python3 main.py"), {"main.py": dict(_IMPROVED)}, True, ""),     # r4: no rewrite
])
def test_the_hook(env, req, fname, op, mut, target, cmd, written, err, want):
    _run_hook(fname, op, mut, target, cmd, {k: dict(v) for k, v in written.items()}, err)
    assert L.request_progress_what(RID) == want


def _run_hook(fname, op, mut, target, cmd, written, err, unresolved=False, rejection=False):
    from ghost_agent.core import agent as A
    _open(100)
    self = _Ctx()
    self.context = _Ctx()
    self.context._script_iter = written
    outcome = _Ctx()
    outcome.is_rejection = rejection
    g = {**vars(A), "self": self, "fname": fname, "ptool_op": op, "is_mutating": mut, "ptarget": target,
         "a_hash": cmd, "_res_is_error": err, "_res_unresolved": unresolved, "_outcome": outcome}
    tok = L.request_id_context.set(RID)
    try:
        exec(_hook_code(), g)
    finally:
        L.request_id_context.reset(tok)


@pytest.mark.parametrize("unresolved,rejection", [(True, False), (False, True)])
def test_refused_or_unresolved_writes_and_runs_are_not_progress(env, req, unresolved, rejection):
    _run_hook("file_system", "write", True, "web/app.js", "", {}, False, unresolved, rejection)
    assert L.request_progress_what(RID) == ""
    _run_hook("execute", "", True, "", _x("python3 main.py"), {"main.py": dict(_P)}, False, unresolved, rejection)
    assert L.request_progress_what(RID) == ""


def test_a_counted_run_needs_a_new_write_before_it_counts_again(env, req):
    rec = {"main.py": dict(_P)}
    _run_hook("execute", "", True, "", _x("python3 main.py"), rec, False)
    assert L.request_progress_what(RID) == "ran a file it wrote"
    _run_hook("execute", "", True, "", _x("python3 main.py"), rec, False)          # same file, no write: alternating
    assert L.request_progress_what(RID) == ""                                      # errors / a volatile port
    rec["main.py"]["writes"] += 1
    rec["main.py"]["new_writes"] += 1
    _run_hook("execute", "", True, "", _x("python3 main.py"), rec, False)
    assert L.request_progress_what(RID) == "ran a file it wrote"


def test_a_refused_run_does_not_use_up_the_next_runs_earn(env, req):
    rec = {"main.py": dict(_P)}
    _run_hook("execute", "", True, "", _x("python3 main.py"), rec, False, rejection=True)
    assert L.request_progress_what(RID) == ""
    _run_hook("execute", "", True, "", _x("python3 main.py"), rec, False)
    assert L.request_progress_what(RID) == "ran a file it wrote"


@pytest.mark.parametrize("outputs", [
    ["3 failed, 7 passed", "2 failed, 8 passed"] * 3,                    # a flaky suite
    ["2 passed", "10 passed"] * 3,                                        # a subset, then the whole suite
    ["ConnectionRefusedError: x", "TimeoutError: y"] * 3,                 # alternating errors
    [f"served on port {40000 + 7 * i} pid {13 * i}" for i in range(6)],   # volatile output
])
def test_the_real_verdict_and_the_hook_earn_once_per_rewrite(env, req, outputs):
    # r4: the hook fed hand-made records; here note_run_outcome makes them, as in production
    from ghost_agent.core.agent import note_run_outcome
    rec = {"main.py": {"writes": 1, "new_writes": 1, "runs": 0}}
    earned = []
    for i, out in enumerate(outputs):
        rec["main.py"]["runs"] += 1
        note_run_outcome(rec["main.py"], ok="failed" not in out and "Error" not in out, output=out)
        _run_hook("execute", "", True, "", _x("python3 main.py"), rec, "failed" in out or "Error" in out)
        earned.append(L.request_progress_what(RID) == "ran a file it wrote")
    assert earned[0] and sum(earned) == 1                       # the first run after the write earns, no other
    rec["main.py"]["writes"] += 1                               # a rewrite brings the earn back
    rec["main.py"]["new_writes"] += 1
    rec["main.py"]["runs"] += 1
    note_run_outcome(rec["main.py"], ok=True, output="something new")
    _run_hook("execute", "", True, "", _x("python3 main.py"), rec, False)
    assert L.request_progress_what(RID) == "ran a file it wrote"


def test_a_write_naming_a_tracked_file_does_not_use_up_its_run(env, req):
    rec = {"main.py": dict(_P)}
    _run_hook("file_system", "write", True, "main.py", "file_system:{\"path\": \"main.py\"}", rec, False)
    assert rec["main.py"].get("writes_at_earn", 0) == 0
    _run_hook("execute", "", True, "", _x("python3 main.py"), rec, False)
    assert L.request_progress_what(RID) == "ran a file it wrote"


def test_the_continuation_and_the_advancer_use_the_sized_reserve(env, req, monkeypatch):
    # battery R30/R31: without the request id the reserve is the flat 300 s, not this request's
    from ghost_agent.core import agent as A, write_continuation as WC, project_advancer as PA
    seen = []
    monkeypatch.setattr(A, "effective_report_floor", lambda d, req_id=None: seen.append(req_id) or 30.0)
    req(100)
    WC._time_left(RID)
    tok = L.request_id_context.set(RID)
    try:
        PA._batch_would_cross_deadline(10.0)
    finally:
        L.request_id_context.reset(tok)
    assert seen == [RID, RID]


def test_the_reserve_is_noted_by_the_function_that_computes_it(env, req):
    req(100)
    floor = effective_report_floor(3600.0, RID)
    with L._REQ_STATE_LOCK:
        assert L._REQ_STATE[RID]["floor_s"] == floor


def test_the_turn_start_check_reads_again_after_noting_the_reserve():
    tree = _tree("src/ghost_agent/core/agent.py")
    for f in ast.walk(tree):
        if not isinstance(f, ast.AsyncFunctionDef):
            continue
        body = [n for n in ast.walk(f) if isinstance(n, ast.Call)]
        dnr = [c for c in body if isinstance(c.func, ast.Name) and c.func.id == "deadline_needs_report"
               and any(isinstance(a, ast.Name) and a.id == "_remaining_s" for a in c.args)]
        if not dnr:
            continue
        reads = sorted(c.lineno for c in body if isinstance(c.func, ast.Attribute)
                       and c.func.attr == "request_remaining_s" and c.lineno < dnr[0].lineno
                       and dnr[0].lineno - c.lineno < 40)
        floors = sorted(c.lineno for c in body if isinstance(c.func, ast.Name)
                        and c.func.id == "effective_report_floor" and c.lineno < dnr[0].lineno
                        and dnr[0].lineno - c.lineno < 40)
        assert len(reads) >= 2 and floors and reads[-1] > floors[0], (reads, floors)
        return
    pytest.fail("turn-start deadline check not found")


def test_production_values(env, req):
    env.setenv("GHOST_MAX_REQUEST_S", "3600")
    env.setenv("GHOST_MAX_REQUEST_EXTENDED_S", "10800")
    _client(14400)
    assert (L.base_deadline_s(RID), L.extension_ceiling_s(RID)) == (3600, 10800)


def test_the_hook_runs_after_the_futility_record():
    fn, hook = _hook_tree()
    runs = [n for n in ast.walk(fn) if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Subscript)
            and isinstance(n.target.slice, ast.Constant) and n.target.slice.value == "runs"]
    assert runs and max(r.lineno for r in runs) < hook.lineno   # the run's verdict exists when the hook reads it


def test_a_slow_models_reserve_widens_the_window(env, req):
    # r2 M-A: a reserve above one step forced the report before a second step could be earned
    req(3600 - 1100, progress_ago=5)
    assert L.request_deadline_s(RID) == 3600                      # 1100 s left: outside a bare step
    L.note_report_floor(RID, 1125.0)
    assert L.request_deadline_s(RID) - (3600 - 1100) >= 1125.0 + STEP


def test_the_extension_holds_off_a_slow_models_report(env, req):
    req(3600 - 1100, progress_ago=5)
    floor = 1125.0
    L.note_report_floor(RID, floor)
    rem = L.request_remaining_s(RID)
    assert not deadline_needs_report(rem, floor, False, False)


@pytest.mark.parametrize("base,ceil", [("10903.1", "30638.8"), ("3600.7", "10800.3"), ("1.1", "7.7")])
def test_the_earning_loop_ends_exactly_at_the_ceiling(env, req, base, ceil):
    # r2 M-D: `base + (ceil - base) < ceil` held in floating point and the loop spun under the lock
    env.setenv("GHOST_MAX_REQUEST_S", base)
    env.setenv("GHOST_MAX_REQUEST_EXTENDED_S", ceil)
    _client(float(ceil))
    req(float(ceil) + 10, progress_ago=5)
    assert L.request_deadline_s(RID) == pytest.approx(float(ceil))


def test_a_continued_code_write_is_progress(env, req):
    from ghost_agent.core.write_continuation import _note_progress
    req(100)
    _note_progress(RID, "notes.md")
    assert L.request_progress_what(RID) == ""
    _note_progress(RID, "web/index.html")
    assert L.request_progress_what(RID) == "continued web/index.html"


def test_the_continuation_marks_progress_for_the_first_part_and_each_append():
    tree = _tree("src/ghost_agent/core/write_continuation.py")
    fn = [f for f in ast.walk(tree) if isinstance(f, ast.AsyncFunctionDef) and f.name == "continue_cut_write"][0]
    calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
             and c.func.id == "_note_progress"]
    assert len(calls) == 2
    loops = [w for w in ast.walk(fn) if isinstance(w, ast.While)]
    assert sum(any(c is n for n in ast.walk(w)) for c in calls for w in loops) == 1   # one before, one inside


def _tree(rel):
    return ast.parse((ROOT / rel).read_text())


def test_the_hook_is_in_the_tool_result_loop_and_progress_is_logged():
    tree = _tree("src/ghost_agent/core/agent.py")
    fns = [f for f in ast.walk(tree) if isinstance(f, ast.AsyncFunctionDef)
           and any(isinstance(n, ast.Attribute) and n.attr == "note_code_progress" for n in ast.walk(f))]
    assert [f.name for f in fns] == ["_dispatch_and_process_tool_batch"]
    logs = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "pretty_log" and n.args and isinstance(n.args[0], ast.Constant)
            and n.args[0].value == "Time Budget"]
    assert len(logs) == 1


def test_the_slack_bot_sends_its_30_minute_budget_and_waits_longer():
    # §4SD (operator 10-29: "set a limit to slack for 30 mins"): the bot sends
    # X-Ghost-Client-Timeout (GHOST_SLACK_REQUEST_S, default 1800); its own wait
    # still covers a turn queued behind a long build
    tree = _tree("interface/externals/slack_bot/main.py")
    sent = [v for d in ast.walk(tree) if isinstance(d, ast.Dict)
            for k, v in zip(d.keys, d.values)
            if isinstance(k, ast.Constant) and k.value == "X-Ghost-Client-Timeout"]
    assert len(sent) == 1 and "SLACK_REQUEST_S" in {n.id for n in ast.walk(sent[0]) if isinstance(n, ast.Name)}
    defaults = {c.args[0].value: c.args[1].value for c in ast.walk(tree) if isinstance(c, ast.Call)
                and isinstance(c.func, ast.Attribute) and c.func.attr == "get" and len(c.args) == 2
                and isinstance(c.args[0], ast.Constant) and isinstance(c.args[1], ast.Constant)
                and c.args[0].value in ("GHOST_SLACK_AGENT_TIMEOUT", "GHOST_SLACK_REQUEST_S")}
    assert defaults == {"GHOST_SLACK_AGENT_TIMEOUT": "15300", "GHOST_SLACK_REQUEST_S": "1800"}
    clients = [c for c in ast.walk(tree) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
               and c.func.attr == "AsyncClient"
               and any(k.arg == "timeout" and isinstance(k.value, ast.Name) and k.value.id == "AGENT_TIMEOUT_S"
                       for k in c.keywords)]
    assert len(clients) == 1
