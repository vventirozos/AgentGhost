"""§4OE-2 — the development loop over many calls: the write-over-unread gate
(§4KB restored), an outcome-aware futility breaker, pytest failure names in
the strike line, and a prompt that tells the truth about the sandbox."""
import asyncio

from ghost_agent.core import agent as ag
from ghost_agent.core.agent import futility_tier, note_run_outcome, failure_preview, FUTILITY_REPORT_WRITES, FUTILITY_REPORT_RUNS
from ghost_agent.tools import file_system as FS
from ghost_agent.tools.file_system import tool_file_system


def fs(sb, **kw):
    return asyncio.run(tool_file_system(sandbox_dir=sb, **kw))


def test_a_whole_file_write_over_an_unread_existing_file_is_refused(tmp_path):
    """Fails where the §4KB gate had been lost (its comment survived): a
    400-line app.py never read this request was silently replaced by a
    120-line guess — 16 such pushes in the edit ladder (§4OD)."""
    (tmp_path / "app.py").write_text("".join(f"def f{i}():\n    return {i}\n\n" for i in range(100)))
    r = fs(tmp_path, operation="write", path="app.py", content="def f0():\n    return 'guess'\n")
    assert getattr(r, "reason_code", None) == "unread_rewrite" and "has not read it" in str(r)
    assert (tmp_path / "app.py").read_text().count("def f") == 100            # untouched
    # a new file is never gated, nor its later parts; a FIRST part over an existing unread file is
    assert str(fs(tmp_path, operation="write", path="new.py", content="x = 1\n")).startswith("SUCCESS")
    assert str(fs(tmp_path, operation="write", path="app2.py", content="A = [\n", more_parts=True)).startswith("SUCCESS")
    assert str(fs(tmp_path, operation="write", path="app2.py", content="    1,\n", append=True, more_parts=True)).startswith("SUCCESS")
    (tmp_path / "old.py").write_text("keep = True\n")
    assert getattr(fs(tmp_path, operation="write", path="old.py", content="A = [\n", more_parts=True), "reason_code", None) == "unread_rewrite"
    # §4OG-1 (h): a 3-line ranged read of a 300-line file is NOT a receipt (it unlocked a
    # whole-file discard in §4OE-2, and the refusal text pointed there)
    assert "(lines 1-3)" in str(fs(tmp_path, operation="read", path="app.py", start_line=1, end_line=3))
    r = fs(tmp_path, operation="write", path="app.py", content="def f0():\n    return 'read first'\n")
    assert getattr(r, "reason_code", None) == "unread_rewrite" and "read only about 1%" in str(r)
    # ranged reads ADD UP: 80 % of the lines seen is a receipt
    assert "(lines 1-200)" in str(fs(tmp_path, operation="read", path="app.py", start_line=1, end_line=200))
    assert "(lines 190-245)" in str(fs(tmp_path, operation="read", path="app.py", start_line=190, end_line=245))
    assert str(fs(tmp_path, operation="write", path="app.py", content="def f0():\n    return 'read first'\n")).startswith("SUCCESS")
    # an edit is NOT a read: it makes the earlier view stale
    (tmp_path / "b.py").write_text("a = 1\nb = 2\n")
    assert getattr(fs(tmp_path, operation="write", path="b.py", content="a = 10\nb = 2\nc = 3\n"), "reason_code", None) == "unread_rewrite"
    assert str(fs(tmp_path, operation="edit", path="b.py", old_string="a = 1", new_string="a = 10")).startswith("SUCCESS")
    assert getattr(fs(tmp_path, operation="write", path="b.py", content="a = 10\nb = 2\nc = 3\n"), "reason_code", None) == "unread_rewrite"
    # a whole-file read unlocks it …
    (tmp_path / "c.py").write_text("c = 1\n")
    fs(tmp_path, operation="read", path="c.py")
    assert str(fs(tmp_path, operation="write", path="c.py", content="c = 2\n")).startswith("SUCCESS")
    # … and this request's own edit of a file it only READ voids that view (read → edit → a whole
    # write from the PRE-edit view would silently discard the edit — §4OF defect 8, reader M1)
    (tmp_path / "d.py").write_text("d = 1\nkeep = True\n")
    fs(tmp_path, operation="read", path="d.py")
    assert str(fs(tmp_path, operation="edit", path="d.py", old_string="d = 1", new_string="d = 2")).startswith("SUCCESS")
    r = fs(tmp_path, operation="write", path="d.py", content="d = 1\nkeep = True\n")
    assert getattr(r, "reason_code", None) == "unread_rewrite" and "changed since this request read it" in str(r)
    assert (tmp_path / "d.py").read_text() == "d = 2\nkeep = True\n"
    fs(tmp_path, operation="read", path="d.py")
    assert str(fs(tmp_path, operation="write", path="d.py", content="d = 3\n")).startswith("SUCCESS")
    # a file this request WROTE whole stays its own through its own edits, replaces and appends …
    assert str(fs(tmp_path, operation="edit", path="d.py", old_string="d = 3", new_string="d = 30")).startswith("SUCCESS")
    assert str(fs(tmp_path, operation="replace", path="d.py", old_string="d = 30", new_string="d = 31")).startswith("SUCCESS")
    assert str(fs(tmp_path, operation="write", path="d.py", content="tail\n", append=True)).startswith("SUCCESS")
    assert str(fs(tmp_path, operation="write", path="d.py", content="d = 4\n")).startswith("SUCCESS")
    # … but a change by anything ELSE (a script, another process) leaves the view stale: read again
    (tmp_path / "d.py").write_text("d = 5\n")
    r = fs(tmp_path, operation="write", path="d.py", content="d = 6\n")
    assert getattr(r, "reason_code", None) == "unread_rewrite" and "changed since this request" in str(r)
    assert (tmp_path / "d.py").read_text() == "d = 5\n"
    fs(tmp_path, operation="read", path="d.py")
    assert str(fs(tmp_path, operation="write", path="d.py", content="d = 6\n")).startswith("SUCCESS")


def test_the_futility_breaker_does_not_report_a_loop_that_is_making_progress():
    """Fails where 6 edits + 4 runs forced a blocker report even when every
    run passed more tests (35 fires, 3 forced reports, §4OD)."""
    rec = {"writes": FUTILITY_REPORT_WRITES, "runs": FUTILITY_REPORT_RUNS}
    note_run_outcome(rec, ok=False, output="=== 5 failed, 7 passed ===")
    assert rec["progress"] is True                                          # review 5: a first result is not "stalled"
    note_run_outcome(rec, ok=False, output="=== 5 failed, 7 passed ===")
    assert rec["progress"] is False                                          # no progress (§4UI: no report tier)
    note_run_outcome(rec, ok=False, output="=== 3 failed, 7 passed, 2 errors ===")   # fewer failures, same passes
    assert rec["progress"] is True and futility_tier({"app.py": rec}, True, False)[0] == ""   # failures falling
    note_run_outcome(rec, ok=False, output="=== 3 failed, 7 passed, 2 errors ===")
    assert rec["progress"] is False                                          # stalled again: the same counts
    note_run_outcome(rec, ok=True, output="12 passed")                    # 3 failed → 0: progress
    assert futility_tier({"app.py": rec}, True, False)[0] == ""
    note_run_outcome(rec, ok=True, output="=== 9 failed, 3 passed ===")   # the shell said ok, pytest did not
    assert rec["progress"] is False
    # review 5: a green run records fails=0, so a later regression is NOT progress; the summary
    # line's count wins over an earlier "1 failed assertion" in captured output
    g = {"writes": 6, "runs": 4}
    note_run_outcome(g, ok=False, output="3 failed, 9 passed"); note_run_outcome(g, ok=True, output="12 passed")
    assert g["fails"] == 0
    note_run_outcome(g, ok=False, output="captured: 1 failed assertion\n=== 2 failed, 10 passed ===")
    assert g["progress"] is False and g["fails"] == 2
    # review 4: with counts, a feature-per-step loop ("12 passed", "13 passed", …) is progress and
    # timing noise ("in 0.04s" vs "in 0.05s") is not; without counts, a green run with the SAME
    # output is the xrick shape — no progress
    r2 = {"writes": 6, "runs": 4}
    note_run_outcome(r2, ok=True, output="12 passed in 0.04s"); note_run_outcome(r2, ok=True, output="13 passed in 0.05s")
    assert r2["progress"] is True
    note_run_outcome(r2, ok=True, output="13 passed in 0.06s")
    assert r2["progress"] is False                                        # same count, only the timing moved
    r3 = {"writes": 6, "runs": 4}
    note_run_outcome(r3, ok=True, output="rows: 1200 (took 0.4s)"); note_run_outcome(r3, ok=True, output="rows: 1200 (took 0.5s)")
    assert r3["progress"] is False                                       # §4UI: no report tier
    note_run_outcome(r3, ok=True, output="rows: 1337 (took 0.4s)")
    assert r3["progress"] is True
    # review 6: one transient count-less failure between converging runs is not a stall …
    r4 = {"writes": 6, "runs": 4}
    note_run_outcome(r4, ok=False, output="11 passed"); note_run_outcome(r4, ok=False, output="12 passed")
    note_run_outcome(r4, ok=False, output="Timeout: the command ran past 120 s")
    assert r4["progress"] is True and futility_tier({"app.py": r4}, True, False)[0] == ""
    # … the same failure twice in a row is
    note_run_outcome(r4, ok=False, output="Timeout: the command ran past 120 s")
    assert r4["progress"] is False                                       # §4UI: no report tier
    # the "repeat" needs the PREVIOUS run to have failed too: a green run then the same text failing is new
    r6 = {"writes": 6, "runs": 4}
    note_run_outcome(r6, ok=True, output="rows 12"); note_run_outcome(r6, ok=False, output="rows 12")
    assert r6["progress"] is True
    # count runs end a count-less failure streak: timeout, 12 passed, 13 passed, timeout → transient again
    r7 = {"writes": 6, "runs": 4}
    for ok, out in ((False, "Timeout"), (False, "12 passed"), (False, "13 passed"), (False, "Timeout")):
        note_run_outcome(r7, ok=ok, output=out)
    assert r7["progress"] is True
    # a stuck exit-0 script whose output carries a clock or an id does not fake progress
    r5 = {"writes": 6, "runs": 4}
    note_run_outcome(r5, ok=True, output="14:01:03 rows=12 pid=4411"); note_run_outcome(r5, ok=True, output="14:02:09 rows=12 pid=4512")
    assert r5["progress"] is False
    assert futility_tier({"app.py": {"writes": 3, "runs": 2}}, False, False)[0] == "steer"   # the steer is unchanged


def test_the_strike_preview_names_the_failing_tests():
    out = ("============ test session starts ============\n"
           "tests/test_app.py::test_list PASSED\n"
           "tests/test_app.py::test_create FAILED\n"
           "FAILED tests/test_app.py::test_create - AssertionError: 201 != 200\n"
           "FAILED tests/test_app.py::test_delete - KeyError: 'id'\n"
           "FAILED tests/test_app.py::test_update - ValueError\n"
           "FAILED tests/test_app.py::test_get - TypeError\n"
           "=== 4 failed, 1 passed in 0.04s ===\n")
    pv = failure_preview(out)
    assert pv.startswith("FAILED: tests/test_app.py::test_create, tests/test_app.py::test_delete, tests/test_app.py::test_update (+1 more)")
    assert "Traceback" in failure_preview("Traceback (most recent call last):\n  File x\nValueError: boom") or "ValueError: boom" in failure_preview("Traceback (most recent call last):\n  File x\nValueError: boom")


def test_the_prompt_tells_the_truth_about_the_sandbox():
    # §4UI: the sandbox facts moved to the `execute` description (the
    # definition the model is sent, not the source file)
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    text = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "execute")["function"]["description"]
    assert "SANDBOX: Python 3.11" in text and "pytest" in text and "python-chess" in text and "already installed" in text
    assert "`pip install python-chess` in the sandbox if missing" not in text
    assert "--import-mode=importlib" in text


def test_the_other_whole_file_paths_honour_the_gate_and_the_right_reads_stamp(tmp_path):
    """Review: `replace` auto-promoted to a whole-file write bypassed the gate
    (a 30-function module replaced by a 7-line guess); a paged read did not
    count as a read; a 4 KB SAMPLE of a 500 KB file did."""
    (tmp_path / "mod.py").write_text("".join(f"def f{i}():\n    return {i}\n\n" for i in range(30)))
    guess = "import sys\n\ndef f0():\n    return 'guess'\n\nif __name__ == '__main__':\n    f0()\n"   # a complete module
    r = fs(tmp_path, operation="replace", path="mod.py", content=guess)
    assert getattr(r, "reason_code", None) == "unread_rewrite" and (tmp_path / "mod.py").read_text().count("def f") == 30
    fs(tmp_path, operation="read_chunked", path="mod.py", page=1)                 # a paged read is a read
    r2 = fs(tmp_path, operation="replace", path="mod.py", content=guess)
    assert str(r2).startswith("SUCCESS") and "auto-promoted" in str(r2)
    big = tmp_path / "data.txt"
    big.write_text("".join("x" * 1500 + "\n" for _ in range(400)))            # 600 KB of long lines → a SAMPLE read
    sample = str(fs(tmp_path, operation="read", path="data.txt"))
    assert "SAMPLE ONLY (" in sample
    assert getattr(fs(tmp_path, operation="write", path="data.txt", content="tiny\n"), "reason_code", None) == "unread_rewrite"


def test_a_download_over_an_unread_existing_file_is_refused_before_any_network(tmp_path):
    (tmp_path / "data.bin").write_bytes(b"owner bytes")
    r = fs(tmp_path, operation="download", url="http://example.org/new.bin", path="data.bin",
           tor_proxy="socks5://127.0.0.1:9050")
    assert getattr(r, "reason_code", None) == "unread_rewrite" and (tmp_path / "data.bin").read_bytes() == b"owner bytes"
