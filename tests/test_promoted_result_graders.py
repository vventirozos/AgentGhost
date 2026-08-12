"""A promoted `execute` result must not be read as SUCCESS by anything.

The promotion contract makes an unfinished command look successful on
purpose: `EXIT CODE: 0` is what keeps the turn loop from scoring a transient
strike (see sandbox/jobs.py). But roughly a dozen other consumers read an
execute result to decide whether WORK SUCCEEDED, and two of those decisions
are consequential and unattended:

  * ``project_advancer.classify_verify_result`` — a "pass" marks a project
    task DONE, in the idle autoadvancer, with no human present;
  * ``acquired_skills`` TDD gate — a pass INSTALLS a skill.

Prose is not a contract, so the result carries ``PROMOTED_RESULT_MARKER`` in
its first line and each grader tests for it. These are the red-on-revert pins
for that wiring; without them the feature ships verification theatre.
"""

import re

import pytest

from ghost_agent.sandbox.jobs import (
    PROMOTED_RESULT_MARKER, is_promoted_result,
)
from ghost_agent.tools.execute import _promoted_result


JOB = {"id": "job-1a2b3c4d", "log": ".jobs/job-1a2b3c4d.log",
       "promoted_at": 1000.0, "deadline_at": 1000.0 + 2700}


def _result(output="partial"):
    return _promoted_result(dict(JOB), 600, output)


# ------------------------------------------------------------ the marker

def test_the_marker_rides_in_the_first_line():
    """Consumers slice the head of a result (`[:120]`, `[:200]`). A marker
    further down would be invisible to exactly the readers that matter."""
    res = _result()
    first = res.splitlines()[0]
    assert PROMOTED_RESULT_MARKER in first
    assert first.index(PROMOTED_RESULT_MARKER) < 40
    assert is_promoted_result(res[:120])


def test_the_exit_code_banner_still_parses_as_zero():
    """The two properties must hold TOGETHER: machine-readable "unfinished"
    for the graders, and exit 0 for the strike accounting."""
    res = _result()
    assert re.search(r"EXIT CODE:\s*(\d+)", res).group(1) == "0"
    assert is_promoted_result(res)


def test_is_promoted_result_is_false_for_ordinary_results():
    for ordinary in (
        "--- COMMAND RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\nok",
        "--- EXECUTION RESULT ---\nEXIT CODE: 1\nSTDOUT/STDERR:\nboom",
        "", None, 42,
    ):
        assert is_promoted_result(ordinary) is False


# --------------------------------------------------- the verify gate (CRIT)

def test_verify_gate_calls_a_promoted_result_inconclusive():
    """`classify_verify_result` exists to reject success-SHAPED output that
    verified nothing. A promoted result is a fourth such shape: exit 0, no
    error sentinel, and the command has NOT finished."""
    from ghost_agent.core.project_advancer import classify_verify_result
    assert classify_verify_result(_result()) == "inconclusive"
    # The ordinary verdicts are unchanged.
    assert classify_verify_result(
        "--- COMMAND RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\nok") == "pass"
    assert classify_verify_result(
        "--- COMMAND RESULT ---\nEXIT CODE: 1\nSTDOUT/STDERR:\nno") == "fail"


@pytest.mark.asyncio
async def test_the_fail_closed_runner_rewrites_a_promoted_verify():
    """End of the chain: the coding executor's `_run_verify` reads
    `_looks_like_failure`, so the inconclusive verdict has to become an
    explicit ERROR at the seam or the task is still marked DONE."""
    from ghost_agent.core.project_advancer import (
        _verify_fail_closed_runner, _looks_like_failure,
    )

    async def _runner(tool_name, tool_args):
        return _result("pytest collected 812 items")

    wrapped = _verify_fail_closed_runner(_runner)
    out = await wrapped("execute", {"command": "pytest -q"})
    assert out.startswith("ERROR: verify inconclusive")
    assert _looks_like_failure(out) is True, (
        "the executor must see this as a failed verify, not a pass")
    # The original output is preserved for retry feedback, with its banner
    # neutralised so it cannot smuggle an EXIT CODE: 0 past the ERROR.
    assert "pytest collected 812 items" in out
    assert "EXIT-CODE:" in out


# ----------------------------------------------------- the TDD gate (MAJOR)

@pytest.mark.asyncio
async def test_the_tdd_gate_does_not_install_a_skill_on_a_promoted_run(tmp_path):
    """Drives the REAL entry point. The previous version recomputed the
    gate's arithmetic in its own body and then grepped the source — which
    the surviving import line satisfied, so deleting the predicate from
    production left it green."""
    from ghost_agent.tools.acquired_skills import (
        AcquiredSkillManager, tool_create_skill,
    )

    class _Sbx:
        supports_job_promotion = True

        def execute(self, cmd, timeout=None, **kw):
            return _result("collected 3 items"), 0

        def execute_promotable(self, cmd, timeout=None, **kw):
            return _result("collected 3 items"), 0, {"id": "job-1a2b3c4d"}

    mgr = AcquiredSkillManager(base_dir=tmp_path)
    out = await tool_create_skill(
        sandbox_dir=tmp_path, memory_dir=tmp_path, sandbox_manager=_Sbx(),
        name="my_skill", description="d",
        parameters_schema='{"type": "object", "properties": {}}',
        python_code="def run():\n    return 'ok'\n",
        test_payload="{}")
    assert "my_skill" not in mgr._load_registry(), (
        "a skill must not be installed on a TDD run that never finished")
    assert "fail" in str(out).lower()


def test_skill_telemetry_class_is_not_ok_for_a_promoted_run():
    from ghost_agent.tools.registry import _acquired_skill_result_class
    assert _acquired_skill_result_class(_result()) == "infra"
    assert _acquired_skill_result_class(
        "--- COMMAND RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\nok") == "ok"


# ------------------------------------------------------- the world model

def test_foresight_is_not_graded_on_a_promoted_result():
    """Grading `ok=True` would teach the world model that this command shape
    SUCCEEDS on evidence that does not exist; the honest state is unresolved."""
    import inspect
    from ghost_agent.core import agent as agent_mod
    src = inspect.getsource(agent_mod)
    idx = src.index("_fs_resolve(")
    window = src[max(0, idx - 1200):idx]
    assert "_fs_promoted(str_res)" in window, (
        "foresight resolution must skip promoted results")


def test_metacog_skips_a_promoted_result():
    import inspect
    from ghost_agent.core import agent as agent_mod
    src = inspect.getsource(agent_mod)
    idx = src.index('_mc = getattr(self.context, "metacog", None)')
    window = src[idx:idx + 900]
    assert "_mc_promoted(str_res)" in window, (
        "a detached command is neither a competence success nor a failure")


# ----------------------------------------- the rest of the grader surface

def test_the_advancers_other_failure_check_also_refuses_it():
    """`_looks_like_failure` is the SIBLING of the fixed function and the one
    a tick consults when a task has NO verify command — it decides between
    DONE and retry. Missing it meant a task could still be marked DONE,
    unattended, on a command that had not finished."""
    from ghost_agent.core.project_advancer import _looks_like_failure
    assert _looks_like_failure(_result()) is True
    assert _looks_like_failure(
        "--- COMMAND RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\nok") is False


def test_self_play_reward_counts_it_as_a_tool_error():
    """Otherwise an unfinished run scores as a clean one and the reward is
    computed on work that never happened."""
    from ghost_agent.core.self_play_scoring import count_tool_errors
    msgs = [{"role": "tool", "content": _result()}]
    assert count_tool_errors(msgs) == 1
    clean = [{"role": "tool", "content": "--- COMMAND RESULT ---\nEXIT CODE: 0"}]
    assert count_tool_errors(clean) == 0


def test_a_promoted_macro_step_is_not_a_successful_step():
    from ghost_agent.tools.composed_skills import _step_result_ok
    assert _step_result_ok(_result()) is False
    assert _step_result_ok(
        "--- COMMAND RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\nok") is True


@pytest.mark.asyncio
async def test_the_smoke_gate_does_not_fail_open_on_an_unfinished_run():
    """A mangled run fails OPEN (None = pass) by design; an unfinished one
    must not — it is still going, and "no verdict yet" is not "build fine"."""
    from ghost_agent.core import build_gates

    async def _runner(tool_name, tool_args):
        return _result("starting smoke…")

    out = await build_gates.smoke_gate(_runner, ["app.py"])
    assert out is not None, "an unfinished smoke run must not pass the gate"
    assert "did not finish" in out


@pytest.mark.asyncio
async def test_an_unfinished_test_confirms_no_hypothesis():
    """`get_surviving` keeps `consistent is True`, so an unfinished test that
    reads as "clean" would CONFIRM its hypothesis on no evidence."""
    from ghost_agent.core.hypothesis import Hypothesis, HypothesisTester

    async def _exec(tool, action):
        return _result("running the repro…")

    h = Hypothesis(description="the parser drops the last row",
                   test_action="python3 t.py")
    out = await HypothesisTester().test_hypotheses_parallel([h], _exec)
    assert out[0].consistent is not True, (
        "a test that has not finished is evidence for nothing")

    async def _exec_clean(tool, action):
        return "--- COMMAND RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\nall good"

    h2 = Hypothesis(description="x", test_action="python3 t.py")
    out2 = await HypothesisTester().test_hypotheses_parallel([h2], _exec_clean)
    assert out2[0].consistent is True, "an ordinary clean run still confirms"


def test_the_preflight_world_changed_reset_is_gated_in_agent_py():
    import inspect
    from ghost_agent.core import agent as agent_mod
    src = inspect.getsource(agent_mod)
    idx = src.index("_pf_world_mut and not _pf_exec_failed")
    assert "_pf_promoted" in src[idx:idx + 120], (
        "an unfinished mutation must not clear the pre-flight failure guard")
    # …and the flag must be bound OUTSIDE the execute-only branch, or the
    # first non-execute tool call raises NameError.
    bind = src.index("_pf_promoted = _sbx_promoted(str_res)")
    gate = src.index('if fname == "execute" and not _res_is_error:')
    assert bind < gate, "_pf_promoted must be bound unconditionally"


# ------------------------------------------------------ the workspace ledger

def test_the_workspace_summary_carries_the_note(tmp_path):
    """Every reader of this ledger renders `summary` only, so a note was
    write-only — and a still-running command rendered as a completed
    success."""
    from ghost_agent.workspace.model import WorkspaceModel
    model = WorkspaceModel(tmp_path)
    if not getattr(model, "enabled", False):
        pytest.skip("workspace model disabled in this environment")
    model.record_command_outcome(
        command="yt-dlp x", exit_code=0, duration_seconds=600.0,
        note="still running — promoted to background job job-1a2b3c4d")
    events = model.activity.recent(limit=5)
    assert events, "the outcome was not recorded"
    summary = events[0].get("summary") if isinstance(events[0], dict) \
        else events[0].summary
    assert "promoted to background job job-1a2b3c4d" in summary
