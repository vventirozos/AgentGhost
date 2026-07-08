"""Tests for the self-play reference-solution consistency gate (2026-07-08).

Live failure this gate exists for: an LLM-generated critical-path challenge
whose validator hardcoded ``expected duration=10`` while its own setup data
(tasks.json) yields 25. The echo self-test gate passed it (the validator
doesn't CRASH on its own expected output — it's merely wrong about the
data), the solver failed 3/3 attempts on CORRECT code, the frontier
recorded a bogus -1.0 delta on the `algo` cluster, and a misleading
"you skimmed a constraint" lesson was learned.

The fix has three parts:
  1. The generation prompt demands a <reference_solution> block that
     computes the answer FROM the setup files at runtime.
  2. A static gate (validate_reference_solution) rejects references that
     never open the setup's files — a hardcoded reference would agree
     with an equally-hardcoded validator, making the sandbox gate vacuous.
  3. A sandbox gate runs the reference against the real setup data and
     discards the challenge unless the validator passes it.
"""
import inspect

from ghost_agent.core.dream import (
    validate_reference_solution,
    validate_challenge_quality,
)


SETUP = """
import json
tasks = [{"task_id": "A", "duration": 5, "dependencies": []}]
with open("tasks.json", "w") as f:
    json.dump(tasks, f)
"""


class TestValidateReferenceSolution:
    def test_reference_computing_from_setup_file_passes(self):
        ref = """
import json
with open("tasks.json") as f:
    tasks = json.load(f)
print(sum(t["duration"] for t in tasks))
"""
        ok, reason = validate_reference_solution(SETUP, ref)
        assert ok, reason

    def test_hardcoded_reference_is_rejected(self):
        # The vacuous-gate shape: prints values without reading the data.
        ref = 'print("10")\nprint("B")\n'
        ok, reason = validate_reference_solution(SETUP, ref)
        assert not ok
        assert "tasks.json" in reason
        assert "hardcoded" in reason

    def test_dynamic_path_reference_passes(self):
        # A reference resolving files dynamically can't be proven
        # hardcoded from literals — same leniency as the validator check.
        ref = """
import json, pathlib
for p in pathlib.Path(".").glob("*.json"):
    print(json.loads(p.read_text()))
"""
        ok, _ = validate_reference_solution(SETUP, ref)
        assert ok

    def test_missing_reference_or_setup_is_not_gated_here(self):
        # Absence is handled (and logged) by the caller; the static gate
        # itself only judges present pairs.
        assert validate_reference_solution("", 'print("x")')[0] is True
        assert validate_reference_solution(SETUP, "")[0] is True


class TestGateWiring:
    """The orchestration lives inside a very large coroutine; pin the
    load-bearing wiring by source inspection so a refactor can't silently
    drop the gate."""

    def _source(self):
        import ghost_agent.core.dream as dream
        return inspect.getsource(dream)

    def test_generation_prompt_demands_reference_solution(self):
        src = self._source()
        assert "<reference_solution>" in src
        # The contract line: compute from the setup files, never hardcode.
        assert "REFERENCE SOLUTION" in src
        assert "MUST pass your validator" in src

    def test_sandbox_gate_runs_reference_then_validator_and_discards(self):
        src = self._source()
        # The gate writes the reference as solution.py, runs it, then runs
        # the validator, and discards on mismatch.
        assert "Reference-solution consistency gate" in src
        assert "internally inconsistent" in src
        # It must also restore the post-setup snapshot afterwards so the
        # solver never sees the reference's side effects.
        gate_idx = src.index("Reference-solution consistency gate")
        gate_tail = src[gate_idx:gate_idx + 4000]
        assert "_preflight_restore" in gate_tail
        assert "ref_path.unlink" in gate_tail

    def test_static_gate_wired_into_gen_loop(self):
        src = self._source()
        assert "validate_reference_solution(" in src.replace(
            "def validate_reference_solution(", "", 1
        )
