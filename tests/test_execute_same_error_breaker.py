"""§4IB — the same error twenty times under exit 0 is a loop.

THE LIVE FAILURE (req 21b295ef, 2026-09-17). Turns 19–39: `Grid('reduced_gg,
npoints=127')`-style specs, every run printing `Grid: cannot build grid
without 'type'`, every run EXIT 0 because the script wrapped the call in
try/except and printed `ERR: {spec!r} -> {e}`. No strike (`note_failure`
needs a failure), no action breaker (`execute` is a mutating tool and
skipped), and even keyed on the result the output carried a catalog of
memory addresses that differed every run — no two results could share a
fingerprint. The planner said "I've spent 25+ turns on this" and kept
going; the proxy cut the request at 1800 s.

The signal is the ERROR LINE with volatile tokens (addresses, hex ids,
times, temp names, the quoted thing-tried) normalised, counted under the
command HEAD. Steer at 3, force a blocker REPORT at 5.

World where each pin fails: the normaliser stops collapsing addresses or
quoted attempts, a clean result grows a fingerprint, the execute branch
stops feeding the ledger, or the consumer stops steering/reporting.
"""
import ast
import inspect

import pytest

from ghost_agent.core import strikes as st
from ghost_agent.core.strikes import (EXECUTE_SAME_ERROR_HARD_STOP, EXECUTE_SAME_ERROR_STEER,
                                      StrikeLedger, error_line, error_line_fingerprint,
                                      normalise_volatile, note_repeated_action)

CATALOG = ("Grid: cannot build grid without 'type', choices are: Generator\n"
           "    027b27853ea2c4b96436708a777b2bc0  --  0xaaab1a134190\n"
           "    09131429766e7737c087d3a8d7073dc9  --  0xaaab1a1112b0\n")


def _run(spec, addr):
    """One live-shaped output: the model's ERR: line + the catalog dump."""
    return (f"ERR: {spec!r} -> SpecError: [Grid: cannot build grid without 'type']\n"
            + CATALOG.replace("0xaaab1a134190", addr))


# --- the error line ----------------------------------------------------------

def test_error_line_is_the_last_failure_naming_line():
    out = ("total 76\ndrwxr-xr-x . \n---ECKIT---\nTraceback (most recent call last):\n"
           "  File \"<string>\", line 1\nModuleNotFoundError: No module named 'eckit'")
    assert error_line(out).startswith("ModuleNotFoundError")


def test_a_declared_failure_has_an_error_line_whatever_its_prose():
    """Outcome-consumers R3: a classifier reads the STATUS, not only the
    text. A refusal whose wording matches no error pattern still names
    its failure on its first line; an ok status with the same text does not."""
    from ghost_agent.tools.outcome import ToolOutcome
    quiet = "the request was declined by policy\nnothing ran"
    assert error_line(ToolOutcome.rejected(quiet)) == "the request was declined by policy"
    assert error_line(ToolOutcome.ok(quiet)) == ""
    assert error_line_fingerprint(ToolOutcome.rejected(quiet)) != ""
    assert error_line(ToolOutcome.unresolved("EXIT CODE: 0 (STILL RUNNING)")) == ""


def test_clean_output_has_no_error_line_and_no_fingerprint():
    assert error_line("count = 9592\nexit code 0") == ""
    assert error_line_fingerprint("count = 9592\nexit code 0") == ""
    assert error_line_fingerprint("") == ""


@pytest.mark.parametrize("out", [
    "python3: can't open file '/workspace/x.py': [Errno 2] No such file or directory",
    "bash: line 1:   807 Killed                  python3 ifs_grid_oxford.py",
    "ERROR: No matching distribution found for ecmwf-api\n=== exit: 1 ===",
    "Grid: cannot build grid without 'type', choices are: Generator",
])
def test_live_failure_shapes_have_an_error_line(out):
    assert error_line(out)


# --- the normaliser -----------------------------------------------------------

def test_volatile_tokens_collapse():
    s = normalise_volatile("at 0xaaab1a134190 pid 1234 12:34:56 2026-09-17 /tmp/_ghost_inline_ab12.py "
                           "'reduced_gg, npoints=127' \"x\"")
    assert "0xaaab" not in s and "12:34:56" not in s and "2026-09-17" not in s
    assert "_ghost_inline" not in s and "npoints=127" not in s
    assert "0xADDR" in s and "'…'" in s


def test_meaningful_digits_are_kept():
    """A count or a line number is signal, not noise."""
    s = normalise_volatile("line 37: too many indices; count=149")
    assert "37" in s and "149" in s


# --- the fingerprint: the live sequence collapses ------------------------------

def test_twenty_variants_share_one_fingerprint():
    specs = ["reduced_gg, npoints=127", "reduced_gg, npoints=128", "reduced_gg, npts=127",
             "reduced gaussian grid, npoints=127", "reduced_gg, N=128"]
    addrs = ["0xaaab026651d0", "0xaaab025fcc40", "0xaaab02661800", "0xaaab02654a90", "0xaaab1a147350"]
    fps = {error_line_fingerprint(_run(s, a)) for s, a in zip(specs, addrs)}
    assert len(fps) == 1 and "" not in fps


def test_a_different_error_is_a_different_fingerprint():
    a = error_line_fingerprint(_run("reduced_gg, npoints=127", "0xaaab026651d0"))
    b = error_line_fingerprint("ERR: 'x' -> TypeError: __cinit__() takes at most 1 positional argument")
    assert a != b


# --- counting under the ledger ------------------------------------------------

def test_ledger_trips_at_the_steer_threshold_on_the_live_sequence():
    led = StrikeLedger()
    fp = error_line_fingerprint(_run("a", "0x1111"))
    counts = []
    for i in range(EXECUTE_SAME_ERROR_STEER + 2):
        _, cnt, tripped = led.note_action("execute", "python3 (same error)", fp,
                                          threshold=EXECUTE_SAME_ERROR_STEER)
        counts.append((cnt, tripped))
    assert counts[EXECUTE_SAME_ERROR_STEER - 2] == (EXECUTE_SAME_ERROR_STEER - 1, False)
    assert counts[EXECUTE_SAME_ERROR_STEER - 1] == (EXECUTE_SAME_ERROR_STEER, True)
    assert counts[-1][0] == EXECUTE_SAME_ERROR_STEER + 2


def test_thresholds_are_ordered_and_small():
    assert 2 <= EXECUTE_SAME_ERROR_STEER < EXECUTE_SAME_ERROR_HARD_STOP <= 8


# --- the sites ---------------------------------------------------------------

def _handle_chat():
    """The whole module tree: the dispatch loop and its consumers have been
    extracted out of `handle_chat` over time, and the pin is about the
    module having exactly one such site, wherever it lives."""
    from ghost_agent.core import agent as ag
    return ast.parse(inspect.getsource(ag))


def _calls(tree, attr_or_name):
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name == attr_or_name:
                out.append(n)
    return out


def test_execute_branch_feeds_the_ledger_with_the_error_fingerprint():
    tree = _handle_chat()
    fps = _calls(tree, "error_line_fingerprint")
    assert len(fps) == 1
    exec_actions = [c for c in _calls(tree, "note_action")
                    if c.args and isinstance(c.args[0], ast.Constant) and c.args[0].value == "execute"]
    assert len(exec_actions) == 1
    kw = {k.arg: k.value for k in exec_actions[0].keywords}
    assert getattr(kw.get("threshold"), "attr", "") == "EXECUTE_SAME_ERROR_STEER"
    heads = _calls(tree, "_cmd_head")
    assert len(heads) == 1                       # keyed on the command HEAD, not the heredoc
    # …and the branch is REACHABLE: it hangs off `exit_code_val == 0`, the
    # exit-0 case the strike counter cannot see (mutant X1 made it `if False`)
    owners = [n for n in ast.walk(tree) if isinstance(n, ast.If)
              and any(c is fps[0] for c in ast.walk(ast.Module(body=n.body, type_ignores=[])))]
    # every enclosing `if` is an owner (fname == "execute", …); the exit-0
    # gate must be one of them
    assert any(ast.unparse(o.test) == "exit_code_val == 0" for o in owners), \
        [ast.unparse(o.test) for o in owners]


def test_consumer_uses_the_execute_hard_stop_tier():
    """The execute class steers at 3 and reports at 5 — the generic tier
    (3) would abort before the steer ever ran (mutant X6)."""
    tree = _handle_chat()
    tiers = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
             and any(getattr(t, "id", "") == "_hard_n" for t in n.targets)
             and "EXECUTE_SAME_ERROR_HARD_STOP" in ast.unparse(n.value)]
    assert len(tiers) == 1
    guards = [n for n in ast.walk(tree) if isinstance(n, ast.If)
              and ast.unparse(n.test) == "_exec_same_err"
              and any(c is tiers[0] for c in ast.walk(ast.Module(body=n.body, type_ignores=[])))]
    assert len(guards) == 1


def test_consumer_has_the_steer_and_the_report_branches():
    fn = _handle_chat()
    texts = [n.value for n in ast.walk(fn) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert any("STOP trying variations. Write your FINAL answer now" in t for t in texts)
    assert any("Re-running variations of the same call is not progress" in t for t in texts)
    # the hard branch forces a final generation (a report), never an abort marker
    for n in ast.walk(fn):
        if (isinstance(n, ast.If) and isinstance(n.test, ast.BoolOp)
                and any(isinstance(v, ast.Name) and v.id == "_exec_same_err" for v in n.test.values)
                and any(isinstance(v, ast.Compare) for v in n.test.values)):
            # the BODY of this branch only — `ast.walk(n)` would also walk the
            # `orelse` chain, i.e. the generic abort branch below it
            body = ast.Module(body=n.body, type_ignores=[])
            assigned = [t.id for s in ast.walk(body) if isinstance(s, ast.Assign)
                        for t in s.targets if isinstance(t, ast.Name)]
            body_consts = [c.value for c in ast.walk(body) if isinstance(c, ast.Constant) and isinstance(c.value, str)]
            if any("STOP trying variations" in c for c in body_consts):
                assert "force_final_response" in assigned
                assert not any("ATTEMPT_ABORTED" in c for c in body_consts)
                break
    else:
        raise AssertionError("hard-stop branch for the execute same-error class not found")
