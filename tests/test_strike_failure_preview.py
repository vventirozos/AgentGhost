"""§4IB — the strike line names the failure, not the head of the output.

Request 21b295ef: `execution fail — Strike 1/6 (diagnostic) -> total 76
drwxr-xr-x 1 root root 384 …` — an `ls` listing was the whole preview
while `ModuleNotFoundError: No module named 'eckit'` sat at the end of the
same output. The operator reads the stream; the line has to carry the
failure.

World where each pin fails: the preview goes back to the head, a
traceback previews as its header instead of its exception, or the site
stops using the helper.
"""
import ast
import inspect

from ghost_agent.core import agent as ag
from ghost_agent.core.agent import failure_preview

LS_THEN_TRACEBACK = ("total 76\ndrwxr-xr-x 1 root root 384 Sep 17 10:52 .\n-rw-r--r-- 1 root root 10244 .DS_Store\n"
                     "---ECKIT---\nTraceback (most recent call last):\n  File \"<string>\", line 1, in <module>\n"
                     "ModuleNotFoundError: No module named 'eckit'")


def test_traceback_previews_as_its_exception_line():
    assert failure_preview(LS_THEN_TRACEBACK) == "ModuleNotFoundError: No module named 'eckit'"


def test_head_only_listing_is_never_the_preview_when_a_failure_line_exists():
    assert not failure_preview(LS_THEN_TRACEBACK).startswith("total 76")


def test_killed_and_exit_echo_are_found():
    assert failure_preview("T205: ok\nbash: line 1:  1008 Killed   timeout 400 python3 x.py\nexit=137") == "exit=137"
    assert "Killed" in failure_preview("stuff\nbash: line 1:   807 Killed                  python3 x.py")


def test_a_declared_failure_previews_its_first_line():
    """Outcome-consumers R3: the status outranks the prose — a refusal
    with no error-shaped word is still previewed as its banner line."""
    from ghost_agent.tools.outcome import ToolOutcome
    quiet = "the request was declined by policy\nnothing ran\n" + "z" * 300
    assert failure_preview(ToolOutcome.rejected(quiet)) == "the request was declined by policy"
    assert failure_preview(ToolOutcome.ok(quiet)) == "z" * 200


def test_no_failure_line_falls_back_to_the_tail():
    body = "line1\nline2\n" + "z" * 300
    p = failure_preview(body)
    assert p == "z" * 200


def test_preview_is_bounded_and_flat():
    p = failure_preview("Error: " + "x" * 1000 + "\n" + "y" * 50)
    assert len(p) <= 200 and "\n" not in p


def test_site_uses_the_helper_for_execute_bodies():
    tree = ast.parse(inspect.getsource(ag))
    uses = [c for c in ast.walk(tree) if isinstance(c, ast.Call)
            and getattr(c.func, "id", "") == "failure_preview"]
    assert len(uses) == 1
    arg = uses[0].args[0]
    assert "STDOUT/STDERR:" in ast.unparse(arg)
