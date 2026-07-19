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

Fail-closed extension (2026-07-19 log eval): a data-backed challenge whose
generation OMITTED the <reference_solution> used to be accepted with the
consistency gate silently skipped ("fail-open") — and the only solver
failure that night was exactly the class the gate exists to catch (a
validator ordering quirk the challenge text never stated). Now the
orchestrator first attempts a targeted repair (regenerate ONLY the missing
block, mirroring the validator repair), and rejects the challenge if the
repair doesn't produce a usable reference. Challenges WITHOUT a
setup_script stay exempt: there is no data for the validator to disagree
with, so the gate has nothing to check.
"""
import inspect
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core.dream import (
    Dreamer,
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

    def test_omission_is_fail_closed_not_fail_open(self):
        src = self._source()
        # The fail-open acceptance is gone…
        assert "consistency gate will be SKIPPED" not in src
        # …replaced by targeted repair + rejection.
        assert "Reference Repair" in src
        assert "targeted repair did not produce a usable one" in src
        assert '"stop": ["</reference_solution>"]' in src

    def test_generation_prompt_demands_explicit_output_order(self):
        # The 2026-07-19 failure was an ordering ambiguity: correct values,
        # unstated order. The prompt must forbid implicit ordering.
        src = self._source()
        assert "EXPLICIT OUTPUT ORDER" in src


# ── functional: the generation loop end-to-end ───────────────────────────────

SETUP_CSV = """import csv
rows = [["u1", "10.0"], ["u2", "20.0"]]
with open("data.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["user", "amount"])
    w.writerows(rows)
"""

VALIDATOR_CSV = """import subprocess, csv
with open("data.csv") as f:
    rows = list(csv.reader(f))[1:]
exp = sum(float(r[1]) for r in rows)
out = subprocess.run(["python3", "solution.py"], capture_output=True, text=True)
ok = abs(float(out.stdout.strip()) - float(exp)) < 0.01
raise SystemExit(0 if ok else 1)
"""

REFERENCE_CSV = """import csv
with open("data.csv") as f:
    rows = list(csv.reader(f))[1:]
print(sum(float(r[1]) for r in rows))
"""

GEN_XML_NO_REFERENCE = (
    "<challenge_prompt>Sum the amount column of data.csv and print the "
    "total.</challenge_prompt>\n"
    f"<setup_script>{SETUP_CSV}</setup_script>\n"
    f"<validation_script>{VALIDATOR_CSV}</validation_script>\n"
)


def _make_context(tmp_path):
    context = MagicMock()
    context.memory_system = MagicMock()
    context.skill_memory = MagicMock()
    context.skill_memory.get_recent_failures.return_value = "No failures"
    context.llm_client = MagicMock()
    context.args = MagicMock()
    context.args.perfect_it = True
    context.args.smart_memory = 1.0
    context.sandbox_manager = MagicMock()
    context.sandbox_dir = str(tmp_path)
    context.tor_proxy = None
    context.scratchpad = MagicMock()
    context.frontier_tracker = None
    return context


def _make_sandbox():
    sandbox = MagicMock()

    def execute(cmd, *a, **kw):
        if "py_compile" in cmd:
            return ("Syntax OK", 0)
        return ("", 0)

    sandbox.execute.side_effect = execute
    return sandbox


def _is_reference_repair(payload):
    return payload.get("stop") == ["</reference_solution>"]


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_omitted_reference_is_repaired_and_challenge_accepted(
    mock_agent_cls, mock_sandbox_cls, tmp_path, disable_self_play_templates
):
    ctx = _make_context(tmp_path)
    repair_calls = []

    async def llm(payload, **kw):
        if _is_reference_repair(payload):
            repair_calls.append(payload)
            return {"choices": [{"message": {"content":
                f"<reference_solution>{REFERENCE_CSV}</reference_solution>"}}]}
        return {"choices": [{"message": {"content": GEN_XML_NO_REFERENCE}}]}

    ctx.llm_client.chat_completion = AsyncMock(side_effect=llm)

    mock_agent = MagicMock()
    mock_agent.handle_chat = AsyncMock(return_value=("ok", None, None))
    mock_agent._get_recent_transcript.return_value = "transcript"
    mock_agent_cls.return_value = mock_agent
    mock_sandbox_cls.return_value = _make_sandbox()

    dreamer = Dreamer(ctx)
    result = await dreamer.synthetic_self_play("test-model")

    assert len(repair_calls) == 1, "exactly one targeted repair expected"
    assert "generation failed the quality gate" not in str(result)
    # The repaired challenge was ACCEPTED: the solver actually ran.
    mock_agent.handle_chat.assert_awaited()


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_unrepairable_omission_rejects_challenge(
    mock_agent_cls, mock_sandbox_cls, tmp_path, disable_self_play_templates
):
    ctx = _make_context(tmp_path)
    repair_calls = []

    async def llm(payload, **kw):
        if _is_reference_repair(payload):
            repair_calls.append(payload)
            # repair also fails to produce the block
            return {"choices": [{"message": {"content": "no block here"}}]}
        return {"choices": [{"message": {"content": GEN_XML_NO_REFERENCE}}]}

    ctx.llm_client.chat_completion = AsyncMock(side_effect=llm)

    mock_agent = MagicMock()
    mock_agent.handle_chat = AsyncMock(return_value=("ok", None, None))
    mock_agent_cls.return_value = mock_agent
    mock_sandbox_cls.return_value = _make_sandbox()

    dreamer = Dreamer(ctx)
    result = await dreamer.synthetic_self_play("test-model")

    # fail-closed: every gen attempt repaired-then-rejected, solver never ran
    assert len(repair_calls) == 3
    assert "failed the quality gate" in str(result)
    mock_agent.handle_chat.assert_not_awaited()


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_no_setup_challenge_stays_exempt(
    mock_agent_cls, mock_sandbox_cls, tmp_path, disable_self_play_templates
):
    # Without a setup_script there is no data for the validator to disagree
    # with — the consistency gate has nothing to check, so omission must
    # not trigger repair or rejection (pre-existing toy-challenge shape).
    ctx = _make_context(tmp_path)
    repair_calls = []

    async def llm(payload, **kw):
        if _is_reference_repair(payload):
            repair_calls.append(payload)
            return {"choices": [{"message": {"content": "no block"}}]}
        return {"choices": [{"message": {"content":
            "<challenge_prompt>Print the string OK.</challenge_prompt>\n"
            "<validation_script>import subprocess\n"
            "out = subprocess.run(['python3', 'solution.py'], "
            "capture_output=True, text=True)\n"
            "raise SystemExit(0 if out.stdout.strip() == 'OK' else 1)\n"
            "</validation_script>"}}]}

    ctx.llm_client.chat_completion = AsyncMock(side_effect=llm)

    mock_agent = MagicMock()
    mock_agent.handle_chat = AsyncMock(return_value=("ok", None, None))
    mock_agent._get_recent_transcript.return_value = "transcript"
    mock_agent_cls.return_value = mock_agent
    mock_sandbox_cls.return_value = _make_sandbox()

    dreamer = Dreamer(ctx)
    result = await dreamer.synthetic_self_play("test-model")

    assert repair_calls == []
    assert "failed the quality gate" not in str(result)
    mock_agent.handle_chat.assert_awaited()
