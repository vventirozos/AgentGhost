"""Tests for the validator self-test gate in core/dream.py.

The gate catches "LLM generated a validator that crashes on its own
expected_output" bugs — the classic manifestation is:

    expected_lines.append(f"{ip} {total_size} {error_rate:.2f}%")
    ...
    exp_ip, exp_size, exp_rate = exp_parts
    float(exp_rate)  # ValueError: "60.00%"

These crash regardless of what the solver outputs, so they burn every
retry. The gate instruments the validator to dump its expected output,
then re-runs the real validator against a probe solution that echoes
that output. A correct validator must exit 0 on its own expected
output; a self-contradicting one crashes and gets rejected.

Also covers the widened runtime crash detector (validator-frame
ValueError/TypeError/etc. now classified as a validator crash).
"""

import ast
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

from ghost_agent.core.dream import (
    _instrument_validator_for_self_test,
    _extract_selftest_dump,
    _looks_like_validator_crash,
    _SELFTEST_DUMP_START,
    _SELFTEST_DUMP_END,
    _SELFTEST_PROBE_EXIT_CODE,
)


# ---------------------------------------------------------------------------
# _instrument_validator_for_self_test
# ---------------------------------------------------------------------------


class TestInstrumentValidator:
    def test_returns_none_on_syntax_error(self):
        assert _instrument_validator_for_self_test("def broken(:") is None

    def test_returns_none_when_no_subprocess_run_solution_py(self):
        """Validator that doesn't run solution.py isn't something we
        know how to instrument — bail cleanly."""
        src = dedent("""
            import json
            data = json.load(open('data.json'))
            assert data['ok']
        """)
        assert _instrument_validator_for_self_test(src) is None

    def test_inserts_probe_and_truncates_solution_run(self):
        """The probe must (a) include expected_output construction and
        (b) replace the `subprocess.run(solution.py)` tail so the probe
        exits without actually running the solver."""
        src = dedent("""
            import subprocess
            expected_output = "42"
            result = subprocess.run(['python3', 'solution.py'], capture_output=True, text=True)
            assert result.stdout.strip() == expected_output
        """)
        probed = _instrument_validator_for_self_test(src)
        assert probed is not None
        # Probe markers present.
        assert "GHOST SELFTEST PROBE" in probed
        assert _SELFTEST_DUMP_START in probed
        assert _SELFTEST_DUMP_END in probed
        # Probe must SHORT-CIRCUIT: the original subprocess.run line is
        # replaced by the probe that raises SystemExit(42) — so
        # `subprocess.run(` on solution.py must NOT survive in the
        # probed source. (A surviving subprocess.run would re-invoke
        # the solver, defeating the whole gate.)
        assert "subprocess.run(" not in probed
        # But the expected_output construction above the run call must
        # survive — that's what we're dumping.
        assert 'expected_output = "42"' in probed

    def test_probe_is_syntactically_valid_python(self):
        src = dedent("""
            import subprocess
            expected_output = "ok"
            result = subprocess.run(['python3', 'solution.py'], capture_output=True, text=True)
            assert result.stdout.strip() == expected_output
        """)
        probed = _instrument_validator_for_self_test(src)
        ast.parse(probed)  # must not raise


# ---------------------------------------------------------------------------
# End-to-end via subprocess — the critical guarantee is that running the
# probed script actually emits the expected-output dump we can recover.
# ---------------------------------------------------------------------------


class TestProbeRunsAndDumpsExpected:
    def test_probe_emits_markers_and_dump(self, tmp_path):
        src = dedent("""
            import subprocess
            expected_output = "hello world"
            # Pretend we're about to run solution.py
            result = subprocess.run(['python3', 'solution.py'], capture_output=True, text=True)
        """)
        probed = _instrument_validator_for_self_test(src)
        probe_path = tmp_path / ".validator_selftest.py"
        probe_path.write_text(probed)
        r = subprocess.run(
            [sys.executable, str(probe_path)],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == _SELFTEST_PROBE_EXIT_CODE
        dumped = _extract_selftest_dump(r.stdout)
        assert dumped == "hello world"

    def test_probe_dumps_joined_expected_lines(self, tmp_path):
        src = dedent("""
            import subprocess
            expected_lines = []
            for x in ['a', 'b', 'c']:
                expected_lines.append(f"line:{x}")
            result = subprocess.run(['python3', 'solution.py'], capture_output=True, text=True)
        """)
        probed = _instrument_validator_for_self_test(src)
        probe_path = tmp_path / ".validator_selftest.py"
        probe_path.write_text(probed)
        r = subprocess.run(
            [sys.executable, str(probe_path)],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == _SELFTEST_PROBE_EXIT_CODE
        dumped = _extract_selftest_dump(r.stdout)
        assert dumped == "line:a\nline:b\nline:c"

    def test_probe_exits_with_no_var_code_when_nothing_matches(self, tmp_path):
        """A validator that doesn't use any of our candidate names
        still gets instrumented (it passes the subprocess.run check),
        but the probe exits 43 so the gate knows to skip."""
        src = dedent("""
            import subprocess
            totally_unrelated_name = "42"
            result = subprocess.run(['python3', 'solution.py'], capture_output=True, text=True)
        """)
        probed = _instrument_validator_for_self_test(src)
        probe_path = tmp_path / ".validator_selftest.py"
        probe_path.write_text(probed)
        r = subprocess.run(
            [sys.executable, str(probe_path)],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == 43
        assert "GHOST_SELFTEST_NO_EXPECTED_VAR" in r.stderr


# ---------------------------------------------------------------------------
# _extract_selftest_dump
# ---------------------------------------------------------------------------


class TestExtractDump:
    def test_extracts_between_markers(self):
        s = f"prelude\n{_SELFTEST_DUMP_START}\nhello\n{_SELFTEST_DUMP_END}\npostlude\n"
        assert _extract_selftest_dump(s) == "hello"

    def test_missing_markers_returns_none(self):
        assert _extract_selftest_dump("no markers here") is None


# ---------------------------------------------------------------------------
# _looks_like_validator_crash
# ---------------------------------------------------------------------------


class TestCrashDetection:
    def test_flags_validator_frame_traceback(self):
        feedback = (
            'Traceback (most recent call last):\n'
            '  File "/workspace/.validator.py", line 76, in <module>\n'
            '    float(exp_rate)\n'
            "ValueError: could not convert string to float: '60.00%'\n"
        )
        assert _looks_like_validator_crash(feedback) is True

    def test_ignores_solution_frame_traceback(self):
        """If the traceback's last frame is solution.py, that's a
        SOLVER crash (legitimate retry), not a validator crash."""
        feedback = (
            'Traceback (most recent call last):\n'
            '  File "/workspace/solution.py", line 10, in <module>\n'
            '    raise ValueError("boom")\n'
            'ValueError: boom\n'
        )
        assert _looks_like_validator_crash(feedback) is False

    def test_no_traceback_is_not_a_crash(self):
        assert _looks_like_validator_crash("FAIL expected=100 actual=99") is False


# ---------------------------------------------------------------------------
# End-to-end: the production bug (float('60.00%')) must be caught.
# We reproduce the exact validator shape from the 16:15 log and show
# that a probe solution echoing the validator's expected_output makes
# the ORIGINAL validator crash — which is what the gate will reject.
# ---------------------------------------------------------------------------


class TestProductionBugReproduction:
    def test_percent_suffix_validator_crashes_on_own_expected(self, tmp_path):
        """Reproduction of the production bug: validator builds
        expected lines WITH `%` and then calls `float()` on the
        field. Running a probe solution that echoes exactly the
        validator's expected_output must surface a ValueError in
        .validator.py — which is what _looks_like_validator_crash
        keys on so the gate can reject."""
        validator_src = dedent("""
            import subprocess
            # Simulate the bug: expected_output has '%' suffix.
            expected_lines = ["10.0.1.1 500 60.00%", "10.0.2.5 300 40.00%"]
            expected_output = "\\n".join(expected_lines)

            result = subprocess.run(
                ['python3', 'solution.py'],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0:
                print(f"FAIL exit={result.returncode}")
                exit(1)
            actual_output = result.stdout.strip()
            exp_lines = expected_output.split('\\n')
            act_lines = actual_output.split('\\n')
            for exp_line, act_line in zip(exp_lines, act_lines):
                exp_parts = exp_line.split()
                act_parts = act_line.split()
                exp_ip, exp_size, exp_rate = exp_parts
                act_ip, act_size, act_rate = act_parts
                # The bug: float on a '60.00%' string crashes.
                if abs(float(exp_rate) - float(act_rate)) > 0.01:
                    print("rate mismatch")
                    exit(1)
            print("PASS")
        """).strip()

        # Step 1: instrument → extract expected_output.
        probed = _instrument_validator_for_self_test(validator_src)
        assert probed is not None
        probe_path = tmp_path / ".validator_selftest.py"
        probe_path.write_text(probed)
        pr = subprocess.run(
            [sys.executable, str(probe_path)],
            capture_output=True, text=True, timeout=10,
        )
        assert pr.returncode == _SELFTEST_PROBE_EXIT_CODE
        dumped = _extract_selftest_dump(pr.stdout)
        assert dumped is not None
        assert "60.00%" in dumped  # our reproduction has the % suffix

        # Step 2: write a probe solution.py that echoes the dump,
        # plus the real validator.
        (tmp_path / "solution.py").write_text(
            "import sys\n"
            f"sys.stdout.write({dumped!r})\n"
        )
        (tmp_path / ".validator.py").write_text(validator_src)

        # Step 3: run the real validator. It MUST crash with a
        # traceback in .validator.py (not solution.py).
        vr = subprocess.run(
            [sys.executable, ".validator.py"],
            cwd=tmp_path, capture_output=True, text=True, timeout=10,
        )
        assert vr.returncode != 0
        combined = (vr.stdout or "") + (vr.stderr or "")
        assert _looks_like_validator_crash(combined)
        assert "ValueError" in combined
        assert ".validator.py" in combined[-800:]

    def test_well_formed_validator_passes_self_test(self, tmp_path):
        """A validator that correctly strips the '%' before float()
        must pass the self-test (no crash when given its own
        expected output)."""
        validator_src = dedent("""
            import subprocess
            expected_lines = ["10.0.1.1 500 60.00", "10.0.2.5 300 40.00"]
            expected_output = "\\n".join(expected_lines)

            result = subprocess.run(
                ['python3', 'solution.py'],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0:
                print(f"FAIL exit={result.returncode}")
                exit(1)
            actual_output = result.stdout.strip()
            exp_lines = expected_output.split('\\n')
            act_lines = actual_output.split('\\n')
            if len(exp_lines) != len(act_lines):
                print("line count mismatch")
                exit(1)
            for exp_line, act_line in zip(exp_lines, act_lines):
                exp_parts = exp_line.split()
                act_parts = act_line.split()
                # No % suffix — float() works.
                if abs(float(exp_parts[2]) - float(act_parts[2])) > 0.01:
                    print("rate mismatch")
                    exit(1)
            print("PASS")
            exit(0)
        """).strip()

        probed = _instrument_validator_for_self_test(validator_src)
        probe_path = tmp_path / ".validator_selftest.py"
        probe_path.write_text(probed)
        pr = subprocess.run(
            [sys.executable, str(probe_path)],
            capture_output=True, text=True, timeout=10,
        )
        dumped = _extract_selftest_dump(pr.stdout)
        assert dumped is not None

        (tmp_path / "solution.py").write_text(
            f"import sys\nsys.stdout.write({dumped!r})\n"
        )
        (tmp_path / ".validator.py").write_text(validator_src)

        vr = subprocess.run(
            [sys.executable, ".validator.py"],
            cwd=tmp_path, capture_output=True, text=True, timeout=10,
        )
        # Well-formed validator MUST exit 0 on its own expected output.
        assert vr.returncode == 0, f"well-formed validator rejected its own expected output:\n{vr.stdout}\n{vr.stderr}"


# ---------------------------------------------------------------------------
# Widened runtime crash detector (fix B) — structural test against
# dream.py source. The production log at 16:17 showed all 3 attempts
# burnt on a validator-frame ValueError because the old detector only
# looked for NameError/ImportError/etc.
# ---------------------------------------------------------------------------


class TestPerfectItSkipDuringSelfPlay:
    """The Perfect-It follow-up LLM call ran unconditionally at the
    end of every successful heavy-tool turn — including the inner
    turn of a self-play sub-agent. Its `learn_lesson` write landed
    in the ReadOnlySkillMemory wrapper (no-op), but:
      (a) the misleading 'Saved optimization strategy to playbook'
          log line fired on the sub-agent's request ID (production
          trace 16:36, request C6), and
      (b) a full ~15s follow-up LLM call burned per self-play cycle
          generating suggestions that were silently discarded.

    The fix marks ReadOnlySkillMemory with `is_read_only = True`
    and agent.py skips the whole Perfect-It block when the marker
    is present.
    """

    def test_readonly_skill_memory_has_is_read_only_marker(self):
        """The marker must live on the class (not an instance
        attribute) so `getattr(wrapper, 'is_read_only', False)`
        works on any wrapper instance built inside self-play."""
        import inspect
        from ghost_agent.core import dream as dream_module
        src = inspect.getsource(dream_module)
        # The marker is defined inside the isolated synthetic_self_play
        # scope; structural check against the source is sufficient.
        assert "is_read_only = True" in src

    def test_perfect_it_block_checks_read_only_marker(self):
        """The entry condition for the Perfect-It block must gate on
        `is_read_only`, so self-play inner turns skip the whole block."""
        import inspect
        from ghost_agent.core.agent import GhostAgent
        src = inspect.getsource(GhostAgent)
        # The marker check must be near the Perfect-It trigger.
        assert "is_read_only" in src
        perfect_it_idx = src.index('"Perfect It Protocol"')
        marker_idx = src.rindex("is_read_only", 0, perfect_it_idx)
        # Marker check must appear within ~1 KB before the log
        # line — i.e., it guards the entry, not some unrelated path.
        assert perfect_it_idx - marker_idx < 1500


class TestWidenedRuntimeCrashDetector:
    def test_fatal_markers_include_valueerror_and_friends(self):
        import inspect
        from ghost_agent.core import dream as dream_module
        src = inspect.getsource(dream_module)
        # Find the fatal_markers tuple used by the runtime circuit breaker.
        assert '"ValueError"' in src
        assert '"TypeError"' in src
        assert '"KeyError"' in src
        assert '"IndexError"' in src
        assert '"AttributeError"' in src
