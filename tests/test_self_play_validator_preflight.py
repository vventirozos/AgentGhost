"""Self-play validator pre-flight: catches module-scope bugs that
`py_compile` silently accepts.

Background: on 2026-04-17 the agent's self-play generator produced a
validator.py that referenced an undefined variable `best_group_stats`
at module scope. `py_compile` (the old pre-flight) only checks syntax,
so the bug slipped through and the solver wasted ~3 minutes before the
real validator run failed with `NameError`.

The fix is to `ast.parse` + `exec` the validator in a namespace where
`__name__ == "__dry_run__"`, which:
  * catches module-scope NameError / ImportError / ModuleNotFoundError
  * skips `if __name__ == "__main__":` blocks so well-formed
    validators that gate their heavy work behind the main guard
    don't execute twice (once during pre-flight, once for real).

These tests exec the SAME pre-flight source that dream.py writes into
the sandbox, against representative inputs, and assert the classification
matches what we need.
"""

import os
import subprocess
import tempfile
import textwrap

import pytest


# Keep this source in lockstep with the `preflight_src` literal inside
# `src/ghost_agent/core/dream.py`. Both versions exist because dream.py
# writes the script into a Docker sandbox over the execute() shell
# protocol, whereas the test just runs it under the host interpreter.
PREFLIGHT_SRC = (
    "import ast, sys, pathlib\n"
    "src = pathlib.Path('.validator.py').read_text()\n"
    "try:\n"
    "    ast.parse(src)\n"
    "except SyntaxError as e:\n"
    "    print(f'PRE-FLIGHT SyntaxError at line {e.lineno}: {e.msg}', file=sys.stderr)\n"
    "    sys.exit(1)\n"
    "try:\n"
    "    exec(compile(src, '.validator.py', 'exec'), {'__name__': '__dry_run__'})\n"
    "except (NameError, ImportError, ModuleNotFoundError) as e:\n"
    "    print(f'PRE-FLIGHT {type(e).__name__}: {e}', file=sys.stderr)\n"
    "    sys.exit(2)\n"
    "except SystemExit:\n"
    "    pass\n"
    "except Exception:\n"
    "    pass\n"
)


def _run_preflight(validator_source: str) -> subprocess.CompletedProcess:
    """Write the validator + pre-flight to a temp dir and run it,
    returning the CompletedProcess so each test can assert on both
    exit code and stderr."""
    td = tempfile.mkdtemp(prefix="sp_preflight_")
    try:
        with open(os.path.join(td, ".validator.py"), "w") as f:
            f.write(validator_source)
        with open(os.path.join(td, ".preflight.py"), "w") as f:
            f.write(PREFLIGHT_SRC)
        return subprocess.run(
            ["python3", ".preflight.py"],
            cwd=td, capture_output=True, text=True, timeout=15,
        )
    finally:
        import shutil
        shutil.rmtree(td, ignore_errors=True)


class TestRejectsModuleScopeNameError:
    """The exact bug class from the 08:46 self-play log."""

    def test_rejects_undefined_variable_at_module_scope(self):
        validator = textwrap.dedent("""
            groups = {}
            # Module-scope reference to a never-defined name. py_compile
            # accepts this; the new pre-flight must reject it.
            top = sorted(best_group_stats['items'].items(),
                         key=lambda x: x[1], reverse=True)[:3]
            print(top)
        """).lstrip()
        r = _run_preflight(validator)
        assert r.returncode == 2, (
            f"Expected exit 2 (module-scope error). Got {r.returncode}. "
            f"stderr: {r.stderr!r}"
        )
        assert "NameError" in r.stderr
        assert "best_group_stats" in r.stderr

    def test_rejects_missing_import_at_module_scope(self):
        validator = textwrap.dedent("""
            import nonexistent_module_xyz123  # ModuleNotFoundError at import time
            print('unreachable')
        """).lstrip()
        r = _run_preflight(validator)
        assert r.returncode == 2
        assert "ModuleNotFoundError" in r.stderr or "ImportError" in r.stderr


class TestRejectsSyntaxErrors:
    """Backwards-compat: anything py_compile would have caught, pre-flight must too."""

    def test_rejects_syntax_error(self):
        validator = "def broken(:\n    pass\n"  # malformed def
        r = _run_preflight(validator)
        assert r.returncode == 1, (
            f"Syntax errors must exit 1 so the caller can distinguish them from "
            f"name errors (exit 2). Got {r.returncode}, stderr: {r.stderr!r}"
        )
        assert "SyntaxError" in r.stderr


class TestAcceptsCleanValidators:
    """No false rejections — clean validators must pass."""

    def test_accepts_main_guarded_validator(self):
        # This is the canonical shape: imports + defs at module scope,
        # all real work behind the __main__ guard. Pre-flight must NOT
        # execute the heavy comparison logic.
        validator = textwrap.dedent('''
            import csv
            from collections import defaultdict

            def main():
                totals = defaultdict(float)
                # This path never runs under pre-flight because
                # __name__ == "__dry_run__" and the guard below
                # blocks it.
                with open("sales.csv") as f:
                    for row in csv.DictReader(f):
                        totals[row["category"]] += float(row["amount"])
                return totals

            if __name__ == "__main__":
                print(main())
        ''').lstrip()
        r = _run_preflight(validator)
        assert r.returncode == 0, (
            f"Clean validator rejected by pre-flight. stderr: {r.stderr!r}"
        )

    def test_accepts_validator_with_safe_module_scope_compute(self):
        # Some validators do light computation at module scope (e.g.,
        # deriving constants). As long as no undefined names are
        # referenced and imports resolve, pre-flight must pass.
        validator = textwrap.dedent("""
            EXPECTED_COLUMNS = ["category", "region", "revenue"]
            TOLERANCE = 0.01

            def compare(a, b):
                return abs(float(a) - float(b)) < TOLERANCE

            if __name__ == "__main__":
                print(EXPECTED_COLUMNS, TOLERANCE)
        """).lstrip()
        r = _run_preflight(validator)
        assert r.returncode == 0, f"Safe module-scope work rejected: {r.stderr!r}"

    def test_accepts_validator_with_module_scope_assertion_failure(self):
        # If the validator does `assert` work at module scope against
        # a mock file the setup already created, an AssertionError is
        # NOT what we're guarding against here — pre-flight catches
        # NameError/ImportError specifically. AssertionError at module
        # scope gets swallowed (falls through to the bare `except
        # Exception: pass`) because the real validator run catches it
        # with proper reporting later.
        validator = textwrap.dedent("""
            # Pretend assertions against a file that doesn't exist yet.
            # pre-flight should NOT fail on these — they're not in our
            # catch list.
            ok = True
            assert ok, "would only fail if ok were False"
        """).lstrip()
        r = _run_preflight(validator)
        assert r.returncode == 0
