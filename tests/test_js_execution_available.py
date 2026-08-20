"""node must be present, because several suites' real coverage depends on it.

`tests.helpers.eval_js` SKIPS when node is missing. That is the right local
behaviour (a skip is honest about not having run), but it has a systemic
failure mode: on a box without node,
``tests/test_webui_console_review.py`` reports "11 passed, 15 skipped" and
every behavioural pin in it — the ones that killed `void 0 && addMessage(…)`
and `authFailed = false && …`, which text assertions could not — is silently
gone while the run reads green (R2 lens B/C).

One loud failure is better than fifteen quiet skips.
"""

import shutil
import subprocess

from tests.helpers import eval_js


def test_node_is_available():
    assert shutil.which("node"), (
        "node is not on PATH. Several suites assert JS BEHAVIOUR by executing "
        "extracted helpers (tests.helpers.eval_js); without node they degrade "
        "to green skips and stop testing anything. Install node, or accept "
        "that the client-side pins are inert on this host.")


def test_the_executor_actually_executes():
    """A harness that silently returns None would make every eval_js
    assertion vacuous in a different way."""
    assert eval_js("function f(x) { return x * 2; }", "f(21)") == 42
    assert eval_js("function g() { return {a: [1, 2]}; }", "g()") == {"a": [1, 2]}


def test_a_thrown_error_is_a_FAILURE_not_a_silent_null():
    """The whole reason for executing rather than grepping is to catch
    ReferenceErrors like the one R1 nearly shipped (reading a variable from
    another function's scope inside a catch block)."""
    import pytest
    with pytest.raises(RuntimeError) as exc:
        eval_js("function f() { return notDefinedAnywhere; }", "f()")
    assert "notDefinedAnywhere" in str(exc.value)


def test_node_can_run_a_module_with_top_level_await():
    proc = subprocess.run(
        [shutil.which("node"), "--input-type=module", "-e",
         "const v = await Promise.resolve(7); process.stdout.write(String(v));"],
        capture_output=True, text=True, timeout=20)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "7"
