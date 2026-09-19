"""§4II — when the model will not write the report, the system writes the
digest.

THE LIVE FAILURE (probe ifs21133…, 2026-09-17, sixth IFS/Oxford re-test).
Every breaker fired in order (futility steer turn 14, same-error steer
turn 22, futility REPORT tier turn 23 with thinking off), the report turn
and its retry both answered with narration + a `file_system` call, and
the honest fallback shipped — reading, in full: "Last evidence gathered
(`file_system` on probe.py): SUCCESS: Wrote 936 chars to 'probe.py'."
Twenty-four steps, seven distinct errors, four files on disk, and the
user got one write confirmation.

`evidence_digest` builds the report the model would not: the ask, the
files left in place (the verifier's own path ledger), the distinct errors
with counts (the §4IE signature), the last clean tool output, the tools
run. `_no_answer_fallback_reply` ships it under the §4GH head, so the
shape check still refutes it as the non-answer it is.

World where each pin fails: the digest loses a section, counts labelled
variants as different errors, names a deleted file, picks an errored run
as "clean", or the fallback stops carrying it / stops being refuted.
"""
import pytest

from ghost_agent.core import reply_shape_check as rsc
from ghost_agent.core.agent import _no_answer_fallback_reply, evidence_digest
from ghost_agent.tools.outcome import ToolOutcome

RESULT = "--- COMMAND RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\n{}\n"


def _t(name, content, **args):
    return {"role": "tool", "tool_call_id": "c", "name": name,
            "content": ToolOutcome.ok(content, call_args=args)}


LIVE = [
    _t("execute", RESULT.format("Python 3.11.15\neckit OK"), command="python3 --version"),
    _t("file_system", "SUCCESS: Wrote 1387 chars to 'probe.py'. Script-side path (from sandbox cwd): 'probe.py'.",
       operation="write", path="probe.py"),
    _t("execute", RESULT.format("ERR  str reduced_gg npts=31: RuntimeError: SpecError: [Grid: cannot build grid without 'type']"),
       command="cd /workspace && python3 probe.py"),
    _t("execute", RESULT.format("ERR  dict nxacc16: RuntimeError: SpecError: [pl]"), command="python3 probe.py"),
    _t("execute", RESULT.format("ERR  dict pl1: RuntimeError: SpecError: [pl]"), command="python3 probe.py"),
    _t("execute", RESULT.format("ERR  dict npts31: RuntimeError: SpecError: [pl]"), command="python3 probe.py"),
    _t("web_search", "### 1. How to get help in Windows", query="eckit Grid reduced_gg"),
    _t("file_system", "SUCCESS: Wrote 3108 chars to 'oxford_grid.py'. Script-side path (from sandbox cwd): 'oxford_grid.py'.",
       operation="write", path="oxford_grid.py"),
    _t("execute", RESULT.format("Total points: 5122\nSaved oxford_grid.png"), command="cd /workspace && python3 oxford_grid.py"),
    _t("file_system", "SUCCESS: Wrote 200 chars to 'scratch.py'. Script-side path (from sandbox cwd): 'scratch.py'.",
       operation="write", path="scratch.py"),
    _t("file_system", "SUCCESS: Deleted 'scratch.py'.", operation="delete", path="scratch.py"),
    _t("execute", RESULT.format("ERR: AttributeError module 'eckit.geo' has no attribute 'Spec'"), command="python3 -c x"),
]


def test_digest_has_every_section_and_counts_the_labelled_variants_once():
    d = evidence_digest(LIVE, ask="find which grid points … around Oxford? Can you plot them?")
    assert d.startswith("You asked: find which grid points")
    assert "Files this request wrote and left in place: `probe.py`, `oxford_grid.py`" in d
    assert "scratch.py" not in d                                   # written then deleted
    assert "`RuntimeError: SpecError: [pl]` ×3" in d                # three labels, one error
    assert "cannot build grid without" in d and "×1" in d
    assert "AttributeError" in d
    assert "Last clean tool output, `execute` (cd /workspace && python3 oxford_grid.py): " in d
    assert "Saved oxford_grid.png" in d
    assert "Tools run: execute ×7, file_system ×4, web_search ×1" in d


def test_digest_is_empty_without_runs_and_skips_synthetic_rows():
    assert evidence_digest([], ask="x") == ""
    assert evidence_digest(None) == ""
    assert evidence_digest([{"name": "plan", "content": "…", "_synthetic": True}], ask="x") == ""


def test_digest_last_clean_output_is_a_clean_one():
    rows = [_t("execute", RESULT.format("ok 1"), command="a"),
            _t("execute", RESULT.format("Traceback (most recent call last):\nValueError: bad"), command="b")]
    d = evidence_digest(rows)
    assert "Last clean tool output, `execute` (a): " in d
    assert "(b)" not in d
    assert "`ValueError: bad` ×1" in d


def test_digest_caps_the_file_list_and_the_error_list():
    rows = [_t("file_system", f"SUCCESS: Wrote 10 chars to 'f{i}.py'. Script-side path (from sandbox cwd): 'f{i}.py'.",
               operation="write", path=f"f{i}.py") for i in range(15)]
    rows += [_t("execute", RESULT.format(f"{chr(65 + i)}xError: x"), command="c") for i in range(8)]
    d = evidence_digest(rows)
    assert d.count("`f") == 12 and " …" in d
    assert d.count("xError: x` ×1") == 5                           # top five of eight


# ── the fallback carries the digest and stays a refuted shape ─────────

def test_fallback_ships_the_digest_under_the_no_answer_head():
    reply = _no_answer_fallback_reply(LIVE, ask="plot the Oxford grid points")
    assert reply.startswith(rsc.FALLBACK_HEADS["no_answer"])
    assert "You asked: plot the Oxford grid points" in reply
    assert "Files this request wrote and left in place" in reply
    assert "Distinct errors hit" in reply
    assert "the task is NOT finished" in reply
    assert rsc.refute_no_answer_fallback(reply)       # still the non-answer arm
    assert "No tool this turn returned usable evidence" in _no_answer_fallback_reply([])
    empty = _no_answer_fallback_reply([], ask="plot the grid")
    assert empty.startswith(rsc.FALLBACK_HEADS["no_answer"]) and "You asked: plot the grid" in empty
    assert "No tool this turn returned usable evidence" in empty
