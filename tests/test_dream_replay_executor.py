"""§4CM D1-D3 — validator synthesis, the paired executor, the batch.

The engine turns one recorded episode into execution-graded
counterfactual verdicts. Everything downstream will treat those verdicts
as LABELS, so the pins here are almost all about refusing to produce one:

  * a synthesised validator is screened statically, then SELF-TESTED
    against the episode's recorded outcome — a check that cannot fail is
    not a check, and its own synthesis is not evidence that it works;
  * a leg that did not produce a verdict is not a null result;
  * legs within an arm that disagree ABSTAIN — that is the stochasticity
    the pairs exist to detect, and calling it an effect is how a label
    source becomes a noise source (§4BE);
  * the perturbation is applied to the DATA, not to a code path, so
    "this lesson was not available" is the property itself rather than a
    lexical proxy for it.
"""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core import isolation as RE_ISO
from ghost_agent.core import replay_engine as RE


# ------------------------------------------------------------------ #
# D1 — the validator                                                 #
# ------------------------------------------------------------------ #

_GOOD = "import sys, os\nsys.exit(0 if os.path.exists('out.txt') else 1)\n"


def test_a_usable_validator_is_admitted():
    ok, why = RE.validator_is_admissible(_GOOD)
    assert ok is True and why == ""


@pytest.mark.parametrize("src,frag", [
    ("", "empty"),
    ("   ", "empty"),
    ("import subprocess\nsubprocess.run(['ls'])", "subprocess"),
    ("import socket", "socket"),
    ("import requests", "requests"),
    ("import shutil\nshutil.rmtree('/x')", "rmtree"),
    ("import os\nos.remove('x')", "os.remove"),
    ("def broken(:\n  pass", "syntax"),
])
def test_a_validator_that_is_not_a_check_is_rejected(src, frag):
    """A check that shells out, reaches the network or deletes things is
    not a check — it is a second agent with no supervision."""
    ok, why = RE.validator_is_admissible(src)
    assert ok is False and frag in why


def test_an_oversized_validator_is_rejected():
    ok, why = RE.validator_is_admissible("x = 1\n" * 5000)
    assert ok is False and "chars" in why


async def test_synthesis_strips_a_markdown_fence():
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value={"choices": [
        {"message": {"content": "```python\n" + _GOOD + "```"}}]})
    out = await RE.synthesize_validator(SimpleNamespace(
        user_request="make out.txt", tool_calls=[], final_response="done"),
        llm)
    assert out.startswith("import sys") and "```" not in out


async def test_synthesis_returns_empty_rather_than_an_unscreened_script():
    """An episode with no validator is simply not replayable, which is a
    better outcome than a validator nobody screened."""
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value={"choices": [
        {"message": {"content": "import subprocess\nsubprocess.run('x')"}}]})
    out = await RE.synthesize_validator(SimpleNamespace(
        user_request="q", tool_calls=[], final_response=""), llm)
    assert out == ""


async def test_synthesis_survives_an_upstream_failure():
    llm = MagicMock()
    llm.chat_completion = AsyncMock(side_effect=RuntimeError("upstream down"))
    assert await RE.synthesize_validator(
        SimpleNamespace(user_request="q", tool_calls=[],
                        final_response=""), llm) == ""
    assert await RE.synthesize_validator(SimpleNamespace(), None) == ""


def test_the_prompt_carries_the_trace_not_just_the_request():
    """A validator written from the request alone cannot know what
    "accomplished" looks like for THIS episode."""
    traj = SimpleNamespace(tool_calls=[
        SimpleNamespace(name="file_system",
                        arguments={"operation": "write", "path": "out.txt"},
                        result="wrote 3 lines")])
    trace = RE._trace_for_prompt(traj)
    assert "file_system" in trace and "out.txt" in trace
    assert "wrote 3 lines" in trace


# ------------------------------------------------------------------ #
# The perturbation is applied to the DATA                            #
# ------------------------------------------------------------------ #

def _real_skill_memory(tmp_path, triggers):
    from ghost_agent.memory.skills import SkillMemory
    d = tmp_path / "real-memory"
    d.mkdir(parents=True, exist_ok=True)
    sm = SkillMemory(d)
    sm.save_playbook([{"trigger": t, "anti_pattern": "a",
                       "correct_pattern": "c"} for t in triggers])
    return sm


def test_a_withheld_lesson_is_absent_from_the_store(tmp_path):
    """Not filtered out of a rendered string, and not intercepted on one
    of the two retrieval surfaces — ABSENT. A store that does not contain
    the lesson is the property itself, and every ranking, dedup and
    quarantine rule downstream keeps working unchanged."""
    real = _real_skill_memory(tmp_path, ["keep me", "drop me"])
    sm = RE.apply_lesson_perturbation(tmp_path / "fork" / ".memory", real,
                                      withhold="drop me")
    triggers = {it["trigger"] for it in sm._load_playbook()}
    assert triggers == {"keep me"}
    # …and the REAL store is untouched.
    assert {it["trigger"] for it in real._load_playbook()} == {
        "keep me", "drop me"}


def test_the_control_arm_keeps_every_lesson(tmp_path):
    real = _real_skill_memory(tmp_path, ["a", "b"])
    sm = RE.apply_lesson_perturbation(tmp_path / "fork" / ".memory", real,
                                      withhold="")
    assert {it["trigger"] for it in sm._load_playbook()} == {"a", "b"}


def test_an_injected_lesson_is_present(tmp_path):
    real = _real_skill_memory(tmp_path, ["a"])
    sm = RE.apply_lesson_perturbation(
        tmp_path / "fork" / ".memory", real,
        inject={"trigger": "new one", "anti_pattern": "x",
                "correct_pattern": "y"})
    assert {it["trigger"] for it in sm._load_playbook()} == {"a", "new one"}


def test_the_forked_store_is_writable_and_thrown_away(tmp_path):
    """A replay's own lesson writes land in the fork and die with it."""
    real = _real_skill_memory(tmp_path, ["a"])
    fork = tmp_path / "fork" / ".memory"
    sm = RE.apply_lesson_perturbation(fork, real)
    sm.save_playbook([{"trigger": "written by the replay"}])
    assert {it["trigger"] for it in real._load_playbook()} == {"a"}
    assert (fork / "skills_playbook.json").exists()


def test_withholding_a_lesson_that_is_not_there_asks_nothing(tmp_path):
    real = _real_skill_memory(tmp_path, ["a"])
    assert RE._withheld_was_present(real, "a") is True
    assert RE._withheld_was_present(real, "gone since the recording") is False
    assert RE._withheld_was_present(real, "") is False


# ------------------------------------------------------------------ #
# The step-deny perturbation                                         #
# ------------------------------------------------------------------ #

async def test_step_deny_blocks_the_first_matching_call_then_free_runs():
    """Forward re-execution: after the denial the agent decides
    everything else itself. Nothing is spliced into a recorded suffix —
    61-94% of post-fork actions get rewritten, so a spliced suffix scores
    a world that never happens."""
    ran = []

    async def _tool(**kw):
        ran.append(kw)
        return "real result"

    agent = SimpleNamespace(available_tools={"execute": _tool},
                            disabled_tools=set())
    state = RE._install_step_deny(agent, "execute")
    denied = await agent.available_tools["execute"](i=0)
    assert denied.startswith(RE.STEP_DENY_MARKER)
    assert state["fired"] is True
    assert await agent.available_tools["execute"](i=1) == "real result"
    assert [k["i"] for k in ran] == [1]


async def test_step_deny_matches_on_CONTENT_not_position():
    """The replay free-runs from a different starting state, so it does
    NOT emit the recorded sequence. A positional denial against
    recorded `[fs, fs, execute]` and replayed `[fs, execute, …]` fires on
    the wrong call — or on none — and the perturbed arm becomes
    byte-identical to the control arm."""
    seen = []

    def _mk(name):
        async def _t(**kw):
            seen.append(name)
            return name
        return _t

    agent = SimpleNamespace(
        available_tools={"file_system": _mk("file_system"),
                         "execute": _mk("execute")},
        disabled_tools=set())
    state = RE._install_step_deny(agent, "execute")
    # A replay whose call order differs from the recording:
    await agent.available_tools["file_system"]()
    out = await agent.available_tools["execute"]()
    assert out.startswith(RE.STEP_DENY_MARKER) and state["fired"] is True
    assert seen == ["file_system"]


async def test_a_step_deny_that_never_fires_is_reported():
    """The replay need not call the recorded tool at all. When it does
    not, the perturbed arm IS the control arm."""
    agent = SimpleNamespace(available_tools={"file_system": None},
                            disabled_tools=set())
    state = RE._install_step_deny(agent, "execute")
    assert state["fired"] is False


async def test_step_deny_survives_a_dispatch_miss_rebuild():
    """`_rebuild_available_tools` fires on any tool-name miss — its own
    docstring says models hallucinate variants routinely — and REPLACES
    the dispatch dict from the registry, dropping the wrapper silently."""
    calls = []

    async def _tool(**kw):
        calls.append(1)
        return "real result"

    class _Agent:
        def __init__(self):
            self.available_tools = {"execute": _tool}
            self.disabled_tools = set()

        def _rebuild_available_tools(self):
            self.available_tools = {"execute": _tool}   # from the registry
            return self.available_tools

    agent = _Agent()
    state = RE._install_step_deny(agent, "execute")
    agent._rebuild_available_tools()          # the hallucinated-name path
    out = await agent.available_tools["execute"]()
    assert out.startswith(RE.STEP_DENY_MARKER), \
        "the rebuild dropped the perturbation and nothing noticed"
    assert state["fired"] is True and calls == []


def test_the_denial_reads_as_synthetic_to_the_corpus():
    """The call never ran, so it carries no transition information and
    its label would be inverted."""
    from ghost_agent.core.foresight import is_synthetic_result
    assert is_synthetic_result(RE.STEP_DENY_MARKER + " 'execute' is …")




# ------------------------------------------------------------------ #
# D2 — the paired verdict                                            #
# ------------------------------------------------------------------ #

def _leg(arm, passed=None, reason=""):
    return RE.ReplayLeg(arm=arm, passed=passed, reason=reason)


def test_consistent_arms_that_differ_are_an_effect():
    v, _ = RE.decide_verdict([_leg("control", True)] * 2,
                             [_leg("perturbed", False)] * 2)
    assert v == RE.VERDICT_MATTERED_POS
    v, _ = RE.decide_verdict([_leg("control", False)] * 2,
                             [_leg("perturbed", True)] * 2)
    assert v == RE.VERDICT_MATTERED_NEG


def test_consistent_arms_that_agree_are_no_effect():
    for outcome in (True, False):
        v, _ = RE.decide_verdict([_leg("control", outcome)] * 2,
                                 [_leg("perturbed", outcome)] * 2)
        assert v == RE.VERDICT_NO_EFFECT


def test_legs_within_an_arm_that_disagree_abstain():
    """That is the stochasticity the pairs exist to detect. Calling it an
    effect is how a label source becomes a noise source."""
    v, why = RE.decide_verdict(
        [_leg("control", True), _leg("control", False)],
        [_leg("perturbed", False)] * 2)
    assert v == RE.VERDICT_ABSTAIN and "stochastic" in why


def test_an_ungradable_leg_abstains_rather_than_counting_as_a_failure():
    """A missing leg is not a null result, and this corpus is far too
    small to absorb guesses."""
    for bad in (_leg("perturbed", None, "leg exceeded 300s"),
                _leg("perturbed", None, "sandbox infra fault"),
                _leg("perturbed", None, "validator inconclusive (exit 2)")):
        v, why = RE.decide_verdict([_leg("control", True)] * 2,
                                   [_leg("perturbed", True), bad])
        assert v == RE.VERDICT_ABSTAIN
        assert bad.reason in why


def test_an_empty_arm_abstains():
    v, why = RE.decide_verdict([_leg("control", True)], [])
    assert v == RE.VERDICT_ABSTAIN and "no legs" in why


def test_a_network_failure_is_not_a_verdict():
    for msg in ("Temporary failure in name resolution",
                "curl: (6) Could not resolve host: example.com",
                "OSError: [Errno 101] Network is unreachable"):
        assert RE._network_failure(msg) is True
    assert RE._network_failure("AssertionError: expected 3 rows") is False


# ------------------------------------------------------------------ #
# D2 — a leg, end to end against a fake sandbox                      #
# ------------------------------------------------------------------ #

@pytest.fixture(autouse=True)
def _preflight_clears(monkeypatch):
    """The batch tests exercise batch LOGIC. The preflight has its own
    tests, and leaving the real one in place makes every batch test
    depend on whether the operator's Docker happens to be running —
    which is exactly the kind of environment coupling that makes a suite
    lie."""
    monkeypatch.setattr(RE, "preflight", lambda: (True, "test"))


@pytest.fixture(autouse=True)
def _reset_shared_state():
    """Two pieces of state leak between cases, and both bit.

    `next_exit` is a CLASS attribute on the fake sandbox, shared across
    tests, and the suite runs in random order.

    The isolation BACKOFF is module-level and process-wide by design (it
    is what stops an unattended batch re-paying a failing provision on
    every item). A test that deliberately makes the sandbox unavailable
    therefore arms a 15-minute refusal for every test after it — which is
    the module working correctly and a harness that lies."""
    from ghost_agent.core.isolation import reset_backoff_for_tests
    _FakeSandbox.next_exit = 0
    _FakeSandbox.neg_exit = 1
    reset_backoff_for_tests()
    yield
    _FakeSandbox.next_exit = 0
    _FakeSandbox.neg_exit = 1
    reset_backoff_for_tests()


class _FakeSandbox:
    """A sandbox that DISCRIMINATES, because the thing under test is a
    validator that must. `neg_exit` is what the validator returns in the
    negative-control fork (empty workspace, no agent run); `next_exit` is
    what it returns in a real leg. A fake that returned the same code for
    both would make the negative control unfalsifiable — which is the
    defect it exists to catch, one level up."""

    next_exit = 0
    neg_exit = 1

    def __init__(self, workspace=None, *a, **kw):
        self.workspace = str(workspace or "")
        self.exit_code = kw.get("exit_code", 0)

    def ensure_running(self):
        pass

    def close(self, remove=False):
        pass

    def execute(self, cmd, timeout=None):
        if "dream-negctl" in self.workspace:
            return ("empty workspace", _FakeSandbox.neg_exit)
        return ("checked out.txt", _FakeSandbox.next_exit)


def _ctx(tmp_path):
    ctx = MagicMock()
    ctx.sandbox_dir = str(tmp_path / "live-ws")
    Path(ctx.sandbox_dir).mkdir(parents=True, exist_ok=True)
    ctx.memory_dir = tmp_path / "live-memory"
    ctx.memory_dir.mkdir(parents=True, exist_ok=True)
    ctx.tor_proxy = None
    ctx.args = MagicMock()
    ctx.args.native_tools = False
    return ctx


def _spec(**over):
    d = {"spec_id": "abc123", "trajectory_id": "t1",
         "perturbation": RE.PERTURB_VERIFY_TOGGLE, "target": "verifier",
         "fork_step": 0, "user_request": "make out.txt",
         "recorded_outcome": "passed", "n_steps": 3}
    d.update(over)
    return d


class _FakeAgent:
    def __init__(self, context):
        self.context = context
        self.disabled_tools = set()
        from ghost_agent.tools.registry import TOOL_DEFINITIONS
        self.available_tools = {t["function"]["name"]: object()
                                for t in TOOL_DEFINITIONS}

    async def handle_chat(self, body, background_tasks=None,
                          request_id=None):
        body.setdefault("messages", []).append(
            {"role": "tool", "content": "ok"})
        return ("done", None, None)


async def test_a_leg_grades_on_the_validator_exit_code(tmp_path,
                                                       monkeypatch):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    _FakeSandbox.next_exit = 0
    with patch("ghost_agent.sandbox.docker.DockerSandbox", _FakeSandbox), \
            patch("ghost_agent.core.agent.GhostAgent", _FakeAgent):
        leg = await RE.run_leg(_ctx(tmp_path), _spec(), arm="control",
                               validator=_GOOD)
    assert leg.passed is True and leg.validator_exit == 0
    _FakeSandbox.next_exit = 1
    with patch("ghost_agent.sandbox.docker.DockerSandbox", _FakeSandbox), \
            patch("ghost_agent.core.agent.GhostAgent", _FakeAgent):
        leg = await RE.run_leg(_ctx(tmp_path), _spec(), arm="control",
                               validator=_GOOD)
    assert leg.passed is False


async def test_an_inconclusive_validator_is_not_a_failure(tmp_path,
                                                          monkeypatch):
    """Exit 2 is the validator's own "I cannot check this from the
    filesystem". Charging it to the agent is the §4AO label-noise class."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    _FakeSandbox.next_exit = 2
    with patch("ghost_agent.sandbox.docker.DockerSandbox", _FakeSandbox), \
            patch("ghost_agent.core.agent.GhostAgent", _FakeAgent):
        leg = await RE.run_leg(_ctx(tmp_path), _spec(), arm="control",
                               validator=_GOOD)
    assert leg.passed is None and "inconclusive" in leg.reason


async def test_a_sandbox_banner_is_not_charged_to_the_agent(tmp_path,
                                                            monkeypatch):
    """`docker.execute` never raises — it RETURNS this banner. Self-play
    already paid for reading it as a genuine failure."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))

    class _Banner(_FakeSandbox):
        def execute(self, cmd, timeout=None):
            return ("[SANDBOX INFRA ERROR — not your code] daemon gone", 1)

    with patch("ghost_agent.sandbox.docker.DockerSandbox", _Banner), \
            patch("ghost_agent.core.agent.GhostAgent", _FakeAgent):
        leg = await RE.run_leg(_ctx(tmp_path), _spec(), arm="control",
                               validator=_GOOD)
    assert leg.passed is None and "infra" in leg.reason


async def test_a_leg_applies_containment_before_it_runs(tmp_path,
                                                        monkeypatch):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    _FakeSandbox.next_exit = 0
    seen = {}

    class _Capturing(_FakeAgent):
        async def handle_chat(self, body, background_tasks=None,
                              request_id=None):
            seen["tools"] = set(self.available_tools)
            seen["req_id"] = request_id
            return await super().handle_chat(body, background_tasks,
                                             request_id)

    with patch("ghost_agent.sandbox.docker.DockerSandbox", _FakeSandbox), \
            patch("ghost_agent.core.agent.GhostAgent", _Capturing):
        await RE.run_leg(_ctx(tmp_path), _spec(), arm="control",
                         validator=_GOOD)
    assert "postgres_admin" not in seen["tools"]
    assert "notify_operator" not in seen["tools"]
    assert seen["req_id"].startswith("replay-")


async def test_a_leg_respects_the_shared_deadline(tmp_path, monkeypatch):
    """A budget is a deadline, not a duration: per-leg durations that
    each look reasonable add up to a night."""
    import time
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    leg = await RE.run_leg(_ctx(tmp_path), _spec(), arm="control",
                           validator=_GOOD,
                           deadline=time.monotonic() - 1.0)
    assert leg.passed is None and "deadline" in leg.reason


async def test_an_unavailable_sandbox_abstains_rather_than_failing(
        tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))

    class _Dead:
        def __init__(self, *a, **kw):
            raise RuntimeError("docker daemon is not running")

    with patch("ghost_agent.sandbox.docker.DockerSandbox", _Dead):
        leg = await RE.run_leg(_ctx(tmp_path), _spec(), arm="control",
                               validator=_GOOD)
    assert leg.passed is None and "isolation unavailable" in leg.reason


# ------------------------------------------------------------------ #
# D2 — run_spec                                                      #
# ------------------------------------------------------------------ #

async def test_run_spec_writes_one_credit_row(tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    _FakeSandbox.next_exit = 0
    with patch("ghost_agent.sandbox.docker.DockerSandbox", _FakeSandbox), \
            patch("ghost_agent.core.agent.GhostAgent", _FakeAgent):
        rec = await RE.run_spec(_ctx(tmp_path), _spec(), validator=_GOOD,
                                n_pairs=2)
    assert rec["verdict"] == RE.VERDICT_NO_EFFECT
    assert rec["control_pass"] == [True, True]
    assert rec["pert_pass"] == [True, True]
    assert rec["validator_hash"]
    rows = list(RE.iter_credits())
    assert len(rows) == 1 and rows[0]["spec_id"] == "abc123"


async def test_run_spec_does_not_run_the_perturbed_arm_on_a_dead_control(
        tmp_path, monkeypatch):
    """Nothing to compare against — and running it anyway spends the
    night's budget producing abstains."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    _FakeSandbox.next_exit = 2          # inconclusive → ungradable
    with patch("ghost_agent.sandbox.docker.DockerSandbox", _FakeSandbox), \
            patch("ghost_agent.core.agent.GhostAgent", _FakeAgent):
        rec = await RE.run_spec(_ctx(tmp_path), _spec(), validator=_GOOD,
                                n_pairs=2)
    assert rec["verdict"] == RE.VERDICT_ABSTAIN
    assert rec["pert_pass"] == []


# ------------------------------------------------------------------ #
# D3 — the batch                                                     #
# ------------------------------------------------------------------ #

def test_the_engine_is_off_by_default(monkeypatch):
    monkeypatch.delenv("GHOST_DREAM_REPLAY", raising=False)
    assert RE._enabled() is False
    monkeypatch.setenv("GHOST_DREAM_REPLAY", "1")
    assert RE._enabled() is True


async def test_a_disabled_batch_does_nothing(monkeypatch, tmp_path):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.delenv("GHOST_DREAM_REPLAY", raising=False)
    out = await RE.run_batch(MagicMock())
    assert out["planned"] == 0 and out["skipped"] == {"disabled": 1}


async def test_the_batch_skips_an_episode_whose_validator_disagrees(
        tmp_path, monkeypatch):
    """THE self-test. The control leg must reproduce the recording; if it
    does not, either the world drifted or the validator is wrong, and
    neither can be told apart from the outcome the perturbed leg would
    produce."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.setenv("GHOST_DREAM_REPLAY", "1")
    traj = SimpleNamespace(
        id="t1", task_kind="user_request", outcome="passed", n_steps=3,
        user_request="make out.txt", final_response="done",
        extra={"hydrated_lessons": ["a lesson"]},
        tool_calls=[SimpleNamespace(
            name="file_system",
            arguments={"operation": "write", "path": "out.txt"},
            result="ok", error="")] * 3)
    ctx = _ctx(tmp_path)
    ctx.llm_client.chat_completion = AsyncMock(return_value={"choices": [
        {"message": {"content": _GOOD}}]})
    _FakeSandbox.next_exit = 1          # control FAILS, recording says passed
    with patch("ghost_agent.sandbox.docker.DockerSandbox", _FakeSandbox), \
            patch("ghost_agent.core.agent.GhostAgent", _FakeAgent), \
            patch.object(RE.EpisodeSource, "_iter_real",
                         lambda self: iter([traj])):
        out = await RE.run_batch(ctx, limit=2)
    assert out["validated"] == 0
    assert out["skipped"].get("validator_disagreed_with_the_recording") == 1
    assert out["verdicts"] == []


async def test_the_batch_runs_specs_once_the_self_test_agrees(tmp_path,
                                                              monkeypatch):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.setenv("GHOST_DREAM_REPLAY", "1")
    traj = SimpleNamespace(
        id="t1", task_kind="user_request", outcome="passed", n_steps=3,
        user_request="make out.txt", final_response="done",
        extra={"hydrated_lessons": ["a lesson"]},
        tool_calls=[SimpleNamespace(
            name="file_system",
            arguments={"operation": "write", "path": "out.txt"},
            result="ok", error="")] * 3)
    ctx = _ctx(tmp_path)
    ctx.llm_client.chat_completion = AsyncMock(return_value={"choices": [
        {"message": {"content": _GOOD}}]})
    _FakeSandbox.next_exit = 0
    with patch("ghost_agent.sandbox.docker.DockerSandbox", _FakeSandbox), \
            patch("ghost_agent.core.agent.GhostAgent", _FakeAgent), \
            patch.object(RE.EpisodeSource, "_iter_real",
                         lambda self: iter([traj])):
        out = await RE.run_batch(ctx, limit=2)
    assert out["validated"] == 1 and len(out["verdicts"]) == 2
    by_kind = {r["perturbation"]: r for r in out["verdicts"]}
    assert by_kind[RE.PERTURB_VERIFY_TOGGLE]["verdict"] == RE.VERDICT_NO_EFFECT
    # …and the withhold ABSTAINS, because its target is not in the store
    # this replay would run against. 88.5% of real withhold specs are in
    # that position: the lesson was pruned between the recording and the
    # replay, so control and perturbed would be identical and the verdict
    # would be sampling noise wearing a `no_effect` label.
    wh = by_kind[RE.PERTURB_LESSON_WITHHOLD]
    assert wh["verdict"] == RE.VERDICT_ABSTAIN
    assert wh["applied"] is False
    assert "no longer in the store" in wh["why"]


async def test_a_spec_is_only_burned_once_it_has_been_ANSWERED(
        tmp_path, monkeypatch):
    """The defect this pins: the planner used to append every picked spec
    to the durable ledger, and the dedup read THAT ledger — so a spec
    skipped for any reason (docker down, no admissible validator, an
    ungradable control) was marked "already asked" forever, with no row
    anywhere saying why. A week of a wedged daemon would consume ~21
    specs of a 222-spec corpus invisibly, and the abstain rate would read
    HEALTHIER the more specs were lost that way."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    traj = SimpleNamespace(
        id="t1", task_kind="user_request", outcome="passed", n_steps=3,
        user_request="q", final_response="",
        extra={"hydrated_lessons": ["L1", "L2"]},
        tool_calls=[SimpleNamespace(
            name="file_system",
            arguments={"operation": "write", "path": "out.txt"},
            result="ok", error="")] * 3)
    ctx = _ctx(tmp_path)
    with patch.object(RE.EpisodeSource, "_iter_real",
                      lambda self: iter([traj])):
        first = await RE.plan_batch(ctx, limit=2)
        # Planned but never answered → still available tomorrow.
        again = await RE.plan_batch(ctx, limit=2)
        assert {s["spec_id"] for s in first} == {s["spec_id"] for s in again}
        # …and once one produces a credit row, it is not re-asked.
        RE.write_credits([{"spec_id": first[0]["spec_id"],
                           "verdict": RE.VERDICT_ABSTAIN, "ts": "t"}])
        third = await RE.plan_batch(ctx, limit=2)
    assert first[0]["spec_id"] not in {s["spec_id"] for s in third}


async def test_even_an_ABSTAIN_counts_as_answered(tmp_path, monkeypatch):
    """An abstain is a measurement — the engine looked and could not
    tell. Re-asking it every night would spend the corpus on the specs
    least likely to yield anything."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    RE.write_credits([{"spec_id": "abc", "verdict": RE.VERDICT_ABSTAIN,
                       "ts": "t"}])
    assert "abc" in RE.known_spec_ids()


def test_the_batch_summary_names_the_abstains_and_the_skips():
    """A night that produced three abstains and a night that produced
    three verdicts are different nights, and a count of "3" hides which."""
    line = RE.batch_summary({
        "planned": 5, "validated": 2,
        "verdicts": [{"verdict": RE.VERDICT_ABSTAIN},
                     {"verdict": RE.VERDICT_ABSTAIN},
                     {"verdict": RE.VERDICT_MATTERED_POS}],
        "skipped": {"no_admissible_validator": 3},
    })
    assert "2x abstain" in line and "1x mattered_pos" in line
    assert "no_admissible_validator" in line


def test_credit_stats_reports_the_abstain_rate(tmp_path, monkeypatch):
    """A replay engine whose output is mostly abstains is not producing
    labels, and an aggregate that hides that reads as throughput."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    assert RE.credit_stats()["present"] is False
    RE.write_credits([
        {"spec_id": "a", "verdict": RE.VERDICT_ABSTAIN, "ts": "t"},
        {"spec_id": "b", "verdict": RE.VERDICT_ABSTAIN, "ts": "t"},
        {"spec_id": "c", "verdict": RE.VERDICT_NO_EFFECT, "ts": "t"},
        {"spec_id": "d", "verdict": RE.VERDICT_MATTERED_POS, "ts": "t"},
    ])
    st = RE.credit_stats()
    assert st["verdicts"] == 4 and st["decisive"] == 2
    assert st["abstain_rate"] == 0.5


# ------------------------------------------------------------------ #
# The preflight                                                      #
# ------------------------------------------------------------------ #

_REAL_PREFLIGHT = RE.preflight

def _healthy(monkeypatch, **over):
    monkeypatch.setattr(RE, "preflight", _REAL_PREFLIGHT)
    """A box that clears every floor, so each test can break exactly one."""
    import psutil
    import shutil as _sh
    vals = {"avail": 8 << 30, "swap": 4 << 30, "disk": 40 << 30}
    vals.update(over)
    monkeypatch.setattr(psutil, "virtual_memory",
                        lambda: SimpleNamespace(available=vals["avail"]))
    monkeypatch.setattr(psutil, "swap_memory",
                        lambda: SimpleNamespace(free=vals["swap"],
                                                total=vals["swap"] * 2))
    monkeypatch.setattr(_sh, "disk_usage",
                        lambda p: SimpleNamespace(free=vals["disk"]))

    class _Images:
        def get(self, name):
            return SimpleNamespace(id="sha256:deadbeef")

    class _Client:
        closed = False
        images = _Images()

        def ping(self):
            return True

        def close(self):
            _Client.closed = True

    import docker as _docker
    monkeypatch.setattr(_docker, "from_env", lambda **kw: _Client())
    return _Client


def test_a_healthy_box_clears_and_says_what_it_checked(monkeypatch):
    cl = _healthy(monkeypatch)
    ok, why = RE.preflight()
    assert ok is True, why
    assert "MB free" in why and "docker answering" in why
    assert cl.closed is True, "the preflight leaked a docker client"


def test_the_preflight_refuses_a_memory_starved_box(monkeypatch):
    """The journal's own operational note: abort a run if swap_free <
    250 MB. A batch spawns a container per leg."""
    _healthy(monkeypatch, avail=100 * 1024 * 1024)
    ok, why = RE.preflight()
    assert ok is False and "available" in why


def test_the_preflight_refuses_a_box_that_is_already_swapping(monkeypatch):
    import psutil
    _healthy(monkeypatch)
    monkeypatch.setattr(psutil, "swap_memory",
                        lambda: SimpleNamespace(free=10 * 1024 * 1024,
                                                total=4 << 30))
    ok, why = RE.preflight()
    assert ok is False and "swapping" in why


def test_no_swap_allocated_is_the_HEALTHIEST_state_not_the_worst(
        monkeypatch):
    """macOS allocates swap dynamically. On a box that has never needed
    it, `swapusage` reads `total = 0.00M, used = 0.00M, free = 0.00M` —
    and the first version of this gate read that as maximally starved and
    refused. Measured on the live box with 17.1 GB available and
    llama-server up: it called the healthiest possible state "only 0 MB
    swap free". Same shape as §4BR — a floor applied to the wrong
    statistic."""
    import psutil
    _healthy(monkeypatch)
    monkeypatch.setattr(psutil, "swap_memory",
                        lambda: SimpleNamespace(free=0, total=0))
    ok, why = RE.preflight()
    assert ok is True, why
    assert "no swap allocated" in why


def test_the_preflight_refuses_when_the_temp_disk_is_full(monkeypatch):
    """The batch's whole footprint is temp directories, and nothing
    bounds what a replayed turn writes into one."""
    _healthy(monkeypatch, disk=100 * 1024 * 1024)
    ok, why = RE.preflight()
    assert ok is False and "temp" in why


def test_the_preflight_probes_the_DAEMON_not_the_package(monkeypatch):
    """`find_spec("docker")` passes with OrbStack stopped — which is the
    actual failure mode on this box, and the one that made every leg
    raise while the batch burned its specs anyway."""
    _healthy(monkeypatch)
    import docker as _docker

    class _Dead:
        def ping(self):
            raise RuntimeError("Error while fetching server API version")

        def close(self):
            pass

    monkeypatch.setattr(_docker, "from_env", lambda **kw: _Dead())
    ok, why = RE.preflight()
    assert ok is False and "daemon is not answering" in why


def test_an_unchecked_precondition_is_not_a_pass(monkeypatch):
    """The preflight's own rule: a precondition it cannot READ must be
    reported, not cleared. The first version appended "memory UNCHECKED"
    to the notes and returned True."""
    import sys as _sys
    _healthy(monkeypatch)
    # Make `import psutil` fail the way it would on a box without it,
    # WITHOUT patching builtins.__import__ (which breaks pytest itself).
    monkeypatch.setitem(_sys.modules, "psutil", None)
    ok, why = RE.preflight()
    assert ok is False and "UNCHECKED" in why


# ------------------------------------------------------------------ #
# The idle phase                                                     #
# ------------------------------------------------------------------ #

def test_the_idle_phase_is_registered_as_gated():
    """`GHOST_DREAM_REPLAY` defaults OFF, so a zero must report the gate
    rather than manufacture an alarm — the `bench` precedent."""
    from ghost_agent.core.autonomous_activity import (
        EXPECT_GATED, PHASE_EXPECTATION, _PHASE_LABELS,
    )
    assert PHASE_EXPECTATION.get("dream_replay") == EXPECT_GATED
    assert _PHASE_LABELS.get("dream_replay")


def test_the_phase_cooldown_leaves_room_inside_the_liveness_window():
    from ghost_agent.core.agent import GhostAgent
    assert GhostAgent._DREAM_REPLAY_COOLDOWN * 4 <= 86400


# ================================================================== #
# R3 — the two-lens review of the Dream engine                       #
# ================================================================== #

def test_the_replay_denylist_is_a_real_superset_of_BOTH_precedents():
    """The first version's docstring CLAIMED to be a superset of
    self-play's list while omitting `web_search` and `deep_research` —
    i.e. real host-process egress, which `network="none"` does not touch.
    Nothing could see the gap because self-play's list was an inline
    literal with nothing to import."""
    from ghost_agent.core.dream import SELF_PLAY_FORBIDDEN_TOOLS
    from ghost_agent.core.isolation import REPLAY_FORBIDDEN_TOOLS
    from ghost_agent.core.subagent import FORBIDDEN_TOOLS

    assert SELF_PLAY_FORBIDDEN_TOOLS <= REPLAY_FORBIDDEN_TOOLS, \
        sorted(SELF_PLAY_FORBIDDEN_TOOLS - REPLAY_FORBIDDEN_TOOLS)
    assert FORBIDDEN_TOOLS <= REPLAY_FORBIDDEN_TOOLS, \
        sorted(FORBIDDEN_TOOLS - REPLAY_FORBIDDEN_TOOLS)


@pytest.mark.parametrize("tool", [
    "web_search", "deep_research", "fact_check",
    "darkweb_search", "darkweb_research", "browser",
    "vision_analysis", "image_generation",
])
def test_no_host_process_egress_survives_the_denylist(tool):
    """`network="none"` covers the CONTAINER. These reach the internet
    from the agent's own process, carrying `context.tor_proxy` — and a
    replayed episode whose local routes are all denied will reach for a
    search."""
    from ghost_agent.core.isolation import REPLAY_FORBIDDEN_TOOLS
    assert tool in REPLAY_FORBIDDEN_TOOLS


async def test_a_replay_can_only_reach_fork_scoped_tools(tmp_path,
                                                         monkeypatch):
    """The executed half: enumerate what a leg's agent is actually left
    holding."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    seen = {}

    class _Capturing(_FakeAgent):
        async def handle_chat(self, body, background_tasks=None,
                              request_id=None):
            seen["tools"] = set(self.available_tools)
            return await super().handle_chat(body, background_tasks,
                                             request_id)

    with patch("ghost_agent.sandbox.docker.DockerSandbox", _FakeSandbox), \
            patch("ghost_agent.core.agent.GhostAgent", _Capturing):
        await RE.run_leg(_ctx(tmp_path), _spec(), arm="control",
                         validator=_GOOD)
    egress = {"web_search", "deep_research", "browser", "darkweb_search",
              "vision_analysis", "image_generation", "fact_check"}
    assert not (seen["tools"] & egress), sorted(seen["tools"] & egress)


async def test_a_perturbed_store_keeps_the_simulation_marker(tmp_path):
    """`skill_memory.is_read_only` is the ONE derivation the repo uses
    for "this is not real traffic" — `turn_origin`, the foresight predict
    hook, the metacog competence write and about nine other gates all
    read it. Swapping in a plain SkillMemory silently re-armed two
    PRODUCTION corpora on 59% of every night's legs."""
    from ghost_agent.core.agent import turn_origin

    real = _real_skill_memory(tmp_path, ["a"])
    sm = RE.apply_lesson_perturbation(tmp_path / "fork" / ".memory", real,
                                      withhold="a")
    assert getattr(sm, "is_read_only", False) is True
    ctx = SimpleNamespace(skill_memory=sm, turn_origin_label=None)
    assert turn_origin(ctx) == "sim"


async def test_every_perturbation_leaves_the_leg_reading_as_a_sim(
        tmp_path, monkeypatch):
    """End to end, for each kind — the marker is what keeps a replay out
    of the foresight ledger and the competence prior."""
    from ghost_agent.core.agent import turn_origin

    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    real = _real_skill_memory(tmp_path, ["a lesson"])
    seen = {}

    class _Capturing(_FakeAgent):
        async def handle_chat(self, body, background_tasks=None,
                              request_id=None):
            seen[self.context.__class__.__name__] = turn_origin(self.context)
            seen["origin"] = turn_origin(self.context)
            return await super().handle_chat(body, background_tasks,
                                             request_id)

    for kind, target in ((RE.PERTURB_LESSON_WITHHOLD, "a lesson"),
                         (RE.PERTURB_VERIFY_TOGGLE, "verifier"),
                         (RE.PERTURB_STEP_DENY, "file_system")):
        ctx = _ctx(tmp_path)
        ctx.skill_memory = real
        with patch("ghost_agent.sandbox.docker.DockerSandbox",
                   _FakeSandbox), \
                patch("ghost_agent.core.agent.GhostAgent", _Capturing):
            await RE.run_leg(ctx, _spec(perturbation=kind, target=target),
                             arm="perturbed", validator=_GOOD)
        assert seen.get("origin") == "sim", f"{kind} disarmed the marker"


# ------------------------------------------------------------------ #
# The control arm must be the RECORDED condition                     #
# ------------------------------------------------------------------ #

async def test_the_control_arm_has_a_verifier(tmp_path, monkeypatch):
    """The live agent constructs one unconditionally at boot, and the
    verifier is priority 1 and 3 in `resolve_turn_outcome` — i.e. the top
    signal that produced the very `recorded_outcome` the self-test
    compares against. A verifier-less control is a third condition, not
    the turn as recorded."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    seen = {}

    class _Capturing(_FakeAgent):
        async def handle_chat(self, body, background_tasks=None,
                              request_id=None):
            seen[body["messages"][0]["content"][:4] + str(len(seen))] = (
                self.context.verifier)
            seen["verifier"] = self.context.verifier
            return await super().handle_chat(body, background_tasks,
                                             request_id)

    with patch("ghost_agent.sandbox.docker.DockerSandbox", _FakeSandbox), \
            patch("ghost_agent.core.agent.GhostAgent", _Capturing):
        await RE.run_leg(_ctx(tmp_path), _spec(), arm="control",
                         validator=_GOOD)
    assert seen["verifier"] is not None


async def test_verify_toggle_REMOVES_the_verifier_on_the_perturbed_arm(
        tmp_path, monkeypatch):
    """"Toggle" means remove. The first version had it backwards —
    control had none and the perturbed arm added one — which inverted the
    sign of every verdict the kind produced."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    seen = {}

    class _Capturing(_FakeAgent):
        async def handle_chat(self, body, background_tasks=None,
                              request_id=None):
            seen["verifier"] = self.context.verifier
            return await super().handle_chat(body, background_tasks,
                                             request_id)

    with patch("ghost_agent.sandbox.docker.DockerSandbox", _FakeSandbox), \
            patch("ghost_agent.core.agent.GhostAgent", _Capturing):
        await RE.run_leg(_ctx(tmp_path),
                         _spec(perturbation=RE.PERTURB_VERIFY_TOGGLE),
                         arm="perturbed", validator=_GOOD)
    assert seen["verifier"] is None


# ------------------------------------------------------------------ #
# An unapplied perturbation is ungradable                            #
# ------------------------------------------------------------------ #

def test_an_unapplied_perturbation_is_not_gradable():
    """The single most load-bearing property in the module: every way a
    perturbation can silently fail produces a perturbed arm identical to
    the control arm, and therefore a confident `no_effect` about a
    counterfactual that never happened."""
    leg = RE.ReplayLeg(arm="perturbed", passed=True, applied=False)
    assert RE._leg_is_gradable(leg) is False
    v, why = RE.decide_verdict([RE.ReplayLeg(arm="control", passed=True)] * 2,
                               [leg] * 2)
    assert v == RE.VERDICT_ABSTAIN


async def test_a_withhold_whose_target_is_gone_abstains(tmp_path,
                                                        monkeypatch):
    """Measured 2026-08-22: 88.5% of real withhold specs target a lesson
    that has since been pruned or quarantined. Control and perturbed
    stores would be identical."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    ctx = _ctx(tmp_path)
    ctx.skill_memory = _real_skill_memory(tmp_path, ["still here"])
    with patch("ghost_agent.sandbox.docker.DockerSandbox", _FakeSandbox), \
            patch("ghost_agent.core.agent.GhostAgent", _FakeAgent):
        leg = await RE.run_leg(
            ctx, _spec(perturbation=RE.PERTURB_LESSON_WITHHOLD,
                       target="pruned last week"),
            arm="perturbed", validator=_GOOD)
    assert leg.applied is False and "no longer in the store" in leg.reason


# ------------------------------------------------------------------ #
# The withhold must cover BOTH retrieval surfaces                    #
# ------------------------------------------------------------------ #

def _chroma_result(triggers):
    return {"documents": [[f"TRIGGER: {t}\nbody" for t in triggers]],
            "distances": [[0.1] * len(triggers)],
            "metadatas": [[{"trigger": t, "type": "skill"}
                           for t in triggers]],
            "ids": [[f"id{i}" for i in range(len(triggers))]]}


def test_the_withhold_covers_the_VECTOR_surface():
    """Hydration is vector-FIRST: it queries Chroma for `type="skill"`
    documents and only then looks the trigger up in the JSON to pick a
    nicer rendering — falling back to the EMBEDDED DOCUMENT when the
    lookup misses. A JSON-only withhold left the lesson in the prompt,
    merely rendered differently, on 59% of this engine's specs."""
    real = MagicMock()
    real.collection.query.return_value = _chroma_result(["keep", "drop"])
    ro = RE.withholding_memory(real, "drop")
    out = ro.collection.query(query_texts=["x"], n_results=5)
    assert [m["trigger"] for m in out["metadatas"][0]] == ["keep"]
    assert len(out["documents"][0]) == 1 and len(out["distances"][0]) == 1
    assert len(out["ids"][0]) == 1


def test_the_withhold_keeps_the_parallel_lists_aligned():
    """Dropping one column and not another silently misaligns
    documents/distances/metadatas — worse than not filtering at all."""
    real = MagicMock()
    real.collection.query.return_value = _chroma_result(["a", "b", "c"])
    ro = RE.withholding_memory(real, "b")
    out = ro.collection.query(query_texts=["x"], n_results=5)
    keys = ("documents", "distances", "metadatas", "ids")
    lengths = {len(out[k][0]) for k in keys}
    assert lengths == {2}
    assert out["metadatas"][0][1]["trigger"] == "c"
    assert out["documents"][0][1].startswith("TRIGGER: c")


def test_the_control_arm_withholds_nothing():
    real = MagicMock()
    real.collection.query.return_value = _chroma_result(["a", "b"])
    ro = RE.withholding_memory(real, "")
    assert len(ro.collection.query()["documents"][0]) == 2


def test_the_withhold_still_blocks_writes():
    """It is layered on the read-only façade, not instead of it."""
    real = MagicMock()
    ro = RE.withholding_memory(real, "x")
    ro.add("y")
    real.add.assert_not_called()


# ------------------------------------------------------------------ #
# The validator's negative control                                   #
# ------------------------------------------------------------------ #

async def test_a_validator_that_passes_an_empty_workspace_is_rejected(
        tmp_path, monkeypatch):
    """`import sys; sys.exit(0)` passes the static screen and then agrees
    with the recording on every `passed` episode with probability 1.0 —
    47 of the 67 real ones — and would report `no_effect` for every
    perturbation of that episode forever. Agreement is exactly what a
    constant validator gives you, so the agreement test cannot see it."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.setenv("GHOST_DREAM_REPLAY", "1")
    traj = SimpleNamespace(
        id="t1", task_kind="user_request", outcome="passed", n_steps=3,
        user_request="q", final_response="", extra={},
        tool_calls=[SimpleNamespace(
            name="file_system",
            arguments={"operation": "write", "path": "out.txt"},
            result="ok", error="")] * 3)
    ctx = _ctx(tmp_path)
    ctx.llm_client.chat_completion = AsyncMock(return_value={"choices": [
        {"message": {"content": "import sys\nsys.exit(0)\n"}}]})
    _FakeSandbox.neg_exit = 0          # it passes an EMPTY workspace
    with patch("ghost_agent.sandbox.docker.DockerSandbox", _FakeSandbox), \
            patch("ghost_agent.core.agent.GhostAgent", _FakeAgent), \
            patch.object(RE.EpisodeSource, "_iter_real",
                         lambda self: iter([traj])):
        out = await RE.run_batch(ctx, limit=1)
    assert out["validated"] == 0
    assert out["skipped"].get("validator_passes_an_empty_workspace") == 1


async def test_the_negative_control_runs_no_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    built = []

    class _Counting(_FakeAgent):
        def __init__(self, context):
            built.append(1)
            super().__init__(context)

    with patch("ghost_agent.sandbox.docker.DockerSandbox", _FakeSandbox), \
            patch("ghost_agent.core.agent.GhostAgent", _Counting):
        leg = await RE.run_validator_only(_ctx(tmp_path), _GOOD)
    assert built == [], "the negative control ran the agent"
    assert leg.passed is False        # neg_exit default 1


# ------------------------------------------------------------------ #
# The verdict's sign                                                 #
# ------------------------------------------------------------------ #

def test_the_verdict_sign_is_normalised_across_perturbations():
    """`mattered_pos` means opposite things for a perturbation that
    REMOVES and one that ADDS. A consumer summing them across kinds is
    adding opposite facts."""
    P, N = RE.VERDICT_MATTERED_POS, RE.VERDICT_MATTERED_NEG
    assert RE._verdict_sign(RE.PERTURB_LESSON_WITHHOLD, P) == "helped"
    assert RE._verdict_sign(RE.PERTURB_STEP_DENY, P) == "helped"
    assert RE._verdict_sign(RE.PERTURB_VERIFY_TOGGLE, P) == "helped"
    assert RE._verdict_sign(RE.PERTURB_LESSON_INJECT, P) == "hurt"
    assert RE._verdict_sign(RE.PERTURB_LESSON_INJECT, N) == "helped"
    assert RE._verdict_sign(RE.PERTURB_LESSON_WITHHOLD,
                            RE.VERDICT_NO_EFFECT) == ""


async def test_the_credit_row_carries_the_sign_and_the_applied_flag(
        tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    with patch("ghost_agent.sandbox.docker.DockerSandbox", _FakeSandbox), \
            patch("ghost_agent.core.agent.GhostAgent", _FakeAgent):
        rec = await RE.run_spec(_ctx(tmp_path), _spec(), validator=_GOOD,
                                n_pairs=1)
    assert "sign" in rec and "applied" in rec
    assert rec["applied"] is True


# ------------------------------------------------------------------ #
# The batch bounds itself and can be stopped                         #
# ------------------------------------------------------------------ #

def test_the_batch_budget_fits_inside_its_callers_bound():
    """The first version's own constants could not: 3 specs x a FRESH
    1500 s per-spec deadline, plus a self-test and a negative-control leg
    per trajectory, is up to 5,400 s inside an hour-long wait_for. The
    bound would fire every time and the warning would blame the spec
    deadline, which had held perfectly."""
    assert RE.DEFAULT_BATCH_TIMEOUT_S < 3600.0
    assert RE.DEFAULT_SPEC_TIMEOUT_S <= RE.DEFAULT_BATCH_TIMEOUT_S


async def test_a_spec_deadline_is_clamped_to_the_batch(tmp_path,
                                                       monkeypatch):
    import time
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    with patch("ghost_agent.sandbox.docker.DockerSandbox", _FakeSandbox), \
            patch("ghost_agent.core.agent.GhostAgent", _FakeAgent):
        rec = await RE.run_spec(_ctx(tmp_path), _spec(), validator=_GOOD,
                                n_pairs=1,
                                batch_deadline=time.monotonic() - 1.0)
    assert rec["verdict"] == RE.VERDICT_ABSTAIN
    # The control legs all hit the clamped deadline, so the perturbed arm
    # never ran — which is itself the point: a batch out of time stops.
    assert rec["control_pass"] == [None] and rec["pert_pass"] == []


async def test_the_kill_switch_stops_an_in_flight_batch(tmp_path,
                                                        monkeypatch):
    """Re-checked between trajectories. Without it the only way to end up
    to an hour of unattended container work is to kill the process."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.setenv("GHOST_DREAM_REPLAY", "1")
    trajs = [SimpleNamespace(
        id=f"t{i}", task_kind="user_request", outcome="passed", n_steps=3,
        user_request="q", final_response="", extra={},
        tool_calls=[SimpleNamespace(
            name="file_system",
            arguments={"operation": "write", "path": "out.txt"},
            result="ok", error="")] * 3)
        for i in range(3)]
    ctx = _ctx(tmp_path)
    ctx.llm_client.chat_completion = AsyncMock(return_value={"choices": [
        {"message": {"content": _GOOD}}]})
    calls = {"n": 0}
    real_leg = RE.run_leg

    async def _flip_after_first(*a, **kw):
        calls["n"] += 1
        if calls["n"] >= 1:
            monkeypatch.setenv("GHOST_DREAM_REPLAY", "0")
        return await real_leg(*a, **kw)

    monkeypatch.setattr(RE, "run_leg", _flip_after_first)
    with patch("ghost_agent.sandbox.docker.DockerSandbox", _FakeSandbox), \
            patch("ghost_agent.core.agent.GhostAgent", _FakeAgent), \
            patch.object(RE.EpisodeSource, "_iter_real",
                         lambda self: iter(trajs)):
        out = await RE.run_batch(ctx, limit=3)
    assert out["stopped_early"], "the master switch could not stop the batch"


async def test_the_batch_stands_down_when_the_box_gets_busy(tmp_path,
                                                            monkeypatch):
    """Resources change under a batch that spawns a container per leg; a
    single reading at entry is not a guarantee for an hour."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.setenv("GHOST_DREAM_REPLAY", "1")
    traj = SimpleNamespace(
        id="t1", task_kind="user_request", outcome="passed", n_steps=3,
        user_request="q", final_response="", extra={},
        tool_calls=[SimpleNamespace(
            name="file_system",
            arguments={"operation": "write", "path": "out.txt"},
            result="ok", error="")] * 3)
    monkeypatch.setattr(RE, "preflight", lambda: (False, "swap exhausted"))
    with patch.object(RE.EpisodeSource, "_iter_real",
                      lambda self: iter([traj])):
        out = await RE.run_batch(_ctx(tmp_path), limit=1)
    assert "preflight stood down mid-batch" in out["stopped_early"]
    assert out["verdicts"] == []


def test_the_summary_says_when_a_batch_stopped_early():
    line = RE.batch_summary({
        "planned": 3, "validated": 0, "verdicts": [],
        "skipped": {}, "stopped_early": "batch deadline reached"})
    assert "STOPPED EARLY" in line and "deadline" in line


# ------------------------------------------------------------------ #
# Resource hygiene                                                   #
# ------------------------------------------------------------------ #

def test_a_process_reclaims_its_OWN_leaked_forks(tmp_path, monkeypatch):
    """`sweep_fork_workspaces` spares any fork whose owner PID is alive —
    which is this process — so a fork leaked at 03:00 was never reclaimed
    while the agent ran."""
    import json as _json
    import os as _os
    from ghost_agent.core.isolation import (
        FORK_PREFIX, OWNER_STAMP, sweep_own_forks,
    )

    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    mine = tmp_path / (FORK_PREFIX + "mine")
    theirs = tmp_path / (FORK_PREFIX + "theirs")
    for d, pid in ((mine, _os.getpid()), (theirs, 999999)):
        d.mkdir()
        (d / OWNER_STAMP).write_text(_json.dumps({"pid": pid, "boot": "b"}))
    removed = sweep_own_forks()
    assert removed == [str(mine)]
    assert theirs.exists(), "another process's fork was reaped"


async def test_a_leg_clamps_the_sandbox_command_budget(tmp_path,
                                                       monkeypatch):
    """`tools/execute.py` passes a 600 s module constant — twice a leg's
    whole budget — on a thread the leg's cancellation cannot stop. The
    leg would return, the container be force-removed and the workspace
    rmtree'd while a process was still writing into it."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    caps = []

    class _Recording(_FakeSandbox):
        def __setattr__(self, name, value):
            if name == "max_exec_timeout":
                caps.append(value)
            object.__setattr__(self, name, value)

    with patch("ghost_agent.sandbox.docker.DockerSandbox", _Recording), \
            patch("ghost_agent.core.agent.GhostAgent", _FakeAgent):
        await RE.run_leg(_ctx(tmp_path), _spec(), arm="control",
                         validator=_GOOD, leg_timeout_s=120.0)
    assert caps, "the leg did not cap the sandbox's command budget"
    assert caps[0] <= 120, caps
    from ghost_agent.tools.execute import _EXEC_TIMEOUT_S
    assert caps[0] < _EXEC_TIMEOUT_S, (
        "the cap must be BELOW the tool's own default, or it changes "
        "nothing for the case it exists for")


def test_the_sandbox_honours_a_command_budget_cap():
    """The clamp lives at the SANDBOX because that is the only layer that
    works regardless of which caller passes what."""
    from ghost_agent.sandbox.docker import DockerSandbox

    sb = DockerSandbox.__new__(DockerSandbox)
    seen = {}

    def _fake_run(cmd, timeout, **kw):
        seen["timeout"] = timeout
        raise RuntimeError("stop here — the prologue already ran")

    sb.max_exec_timeout = 45
    sb._get_lock = lambda: __import__("threading").RLock()
    try:
        DockerSandbox._execute_impl(sb, "echo hi", timeout=600)
    except Exception:
        pass
    # The clamp runs in the prologue, before anything that can fail.
    assert DockerSandbox.max_exec_timeout is None      # off by default
    sb2 = DockerSandbox.__new__(DockerSandbox)
    assert getattr(sb2, "max_exec_timeout", "missing") is None


# ------------------------------------------------------------------ #
# The five mutants that survived the first R3 batch                  #
# ------------------------------------------------------------------ #

async def test_a_withhold_leg_gets_a_WITHHOLDING_vector_store(tmp_path,
                                                              monkeypatch):
    """The JSON store is only half the surface. Pinned on the object the
    leg actually hands the agent, because the docstring's claim to cover
    both surfaces was false for the first version and no test could see
    it."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    seen = {}

    class _Capturing(_FakeAgent):
        async def handle_chat(self, body, background_tasks=None,
                              request_id=None):
            coll = getattr(self.context.memory_system, "collection", None)
            seen["kind"] = type(coll).__name__
            seen["want"] = getattr(coll, "_want", None)
            return await super().handle_chat(body, background_tasks,
                                             request_id)

    ctx = _ctx(tmp_path)
    ctx.skill_memory = _real_skill_memory(tmp_path, ["drop me"])
    with patch("ghost_agent.sandbox.docker.DockerSandbox", _FakeSandbox), \
            patch("ghost_agent.core.agent.GhostAgent", _Capturing):
        await RE.run_leg(
            ctx, _spec(perturbation=RE.PERTURB_LESSON_WITHHOLD,
                       target="drop me"),
            arm="perturbed", validator=_GOOD)
    assert seen["kind"] == "_WithholdingCollection"
    assert seen["want"] == "drop me"


async def test_a_step_deny_that_never_fires_makes_the_leg_ungradable(
        tmp_path, monkeypatch):
    """The replay free-runs from a different starting state, so it need
    not call the recorded tool at all. When it does not, the perturbed
    arm IS the control arm — and a `no_effect` about a counterfactual
    that never happened is worse than no verdict."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    with patch("ghost_agent.sandbox.docker.DockerSandbox", _FakeSandbox), \
            patch("ghost_agent.core.agent.GhostAgent", _FakeAgent):
        leg = await RE.run_leg(
            _ctx(tmp_path),
            _spec(perturbation=RE.PERTURB_STEP_DENY,
                  target="a_tool_this_replay_never_calls"),
            arm="perturbed", validator=_GOOD)
    assert leg.applied is False
    assert "never fired" in leg.reason
    assert RE._leg_is_gradable(leg) is False


async def test_a_leg_forks_from_the_LIVE_workspace(tmp_path, monkeypatch):
    """The recorded turn ran against a populated sandbox. Replaying into
    an empty tempdir means every task whose success is defined by files
    that already existed fails in the control arm — and the self-test
    then discards the episode as "the validator disagrees"."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    ctx = _ctx(tmp_path)
    marker = Path(ctx.sandbox_dir) / "already_here.txt"
    marker.write_text("state from before the turn")
    seen = {}

    class _Looking(_FakeAgent):
        async def handle_chat(self, body, background_tasks=None,
                              request_id=None):
            ws = Path(self.context.sandbox_dir)
            seen["found"] = (ws / "already_here.txt").exists()
            return await super().handle_chat(body, background_tasks,
                                             request_id)

    with patch("ghost_agent.sandbox.docker.DockerSandbox", _FakeSandbox), \
            patch("ghost_agent.core.agent.GhostAgent", _Looking):
        await RE.run_leg(ctx, _spec(), arm="control", validator=_GOOD)
    assert seen["found"] is True, "the leg replayed into an EMPTY workspace"
    # …and it is a COPY: the live sandbox is untouched.
    assert marker.read_text() == "state from before the turn"


async def test_the_kill_switch_is_checked_between_TRAJECTORIES(
        tmp_path, monkeypatch):
    """Two checks, and they cover different windows. The per-spec one
    cannot stop a batch between two episodes' validator syntheses, which
    is where most of a night's LLM budget goes."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.setenv("GHOST_DREAM_REPLAY", "1")
    trajs = [SimpleNamespace(
        id=f"t{i}", task_kind="user_request", outcome="passed", n_steps=3,
        user_request="q", final_response="", extra={},
        tool_calls=[SimpleNamespace(
            name="file_system",
            arguments={"operation": "write", "path": "out.txt"},
            result="ok", error="")] * 3)
        for i in range(3)]
    ctx = _ctx(tmp_path)
    calls = {"n": 0}

    async def _synth(traj, llm, **kw):
        calls["n"] += 1
        return _GOOD

    async def _negctl(context, validator, **kw):
        # Flip the switch during the FIRST episode's negative control, so
        # that episode is skipped BEFORE any spec runs. The per-spec check
        # therefore never fires, and only a check between trajectories can
        # stop the second episode's validator synthesis — which is where
        # most of a night's LLM budget goes.
        monkeypatch.setenv("GHOST_DREAM_REPLAY", "0")
        return RE.ReplayLeg(arm="negative_control", passed=None,
                            reason="stubbed")

    monkeypatch.setattr(RE, "synthesize_validator", _synth)
    monkeypatch.setattr(RE, "run_validator_only", _negctl)
    with patch("ghost_agent.sandbox.docker.DockerSandbox", _FakeSandbox), \
            patch("ghost_agent.core.agent.GhostAgent", _FakeAgent), \
            patch.object(RE.EpisodeSource, "_iter_real",
                         lambda self: iter(trajs)):
        out = await RE.run_batch(ctx, limit=3)
    assert out["stopped_early"] == "disabled mid-batch"
    assert calls["n"] == 1, (
        f"the batch synthesised {calls['n']} validators after being "
        f"disabled — the check is not between trajectories")


@pytest.mark.parametrize("cap,passed,expected", [
    (45, 600, 45),        # clamped — the case it exists for
    (45, 10, 10),         # a caller under the cap keeps its own budget
    (None, 600, 600),     # off by default: no cap, no change
    (1, 600, 5),          # never below the 5s floor
])
def test_the_sandbox_clamps_a_command_budget(cap, passed, expected,
                                             monkeypatch):
    """Executed against the REAL `_execute_impl` prologue. The clamp
    lives at the sandbox because that is the only layer that works
    regardless of which caller passes what — and `tools/execute.py`
    passes a 600 s module constant."""
    from ghost_agent.sandbox.docker import DockerSandbox

    seen = {}
    sb = DockerSandbox.__new__(DockerSandbox)
    sb.max_exec_timeout = cap
    sb.container = None

    def _stop_here(self):
        raise RuntimeError("prologue done")

    # `_execute_impl`'s body immediately enters a try that touches the
    # container; recording `timeout` from the frame is enough.
    import sys as _sys

    def _trace(frame, event, arg):
        if event == "line" and frame.f_code.co_name == "_execute_impl":
            if "timeout" in frame.f_locals:
                seen["timeout"] = frame.f_locals["timeout"]
        return _trace

    _sys.settrace(_trace)
    try:
        DockerSandbox._execute_impl(sb, "echo hi", timeout=passed)
    except Exception:
        pass
    finally:
        _sys.settrace(None)
    assert seen.get("timeout") == expected


def test_the_command_cap_is_off_by_default():
    """A ceiling that applied to the LIVE sandbox would silently shorten
    every real command."""
    from ghost_agent.sandbox.docker import DockerSandbox
    assert DockerSandbox.max_exec_timeout is None
    assert DockerSandbox._execute_impl.__doc__, \
        "the clamp was inserted above the docstring and ate it"


def test_the_preflight_refuses_without_the_provisioned_base_image(
        monkeypatch):
    """A container created with `network=none` cannot provision itself —
    `apt-get update` has no interfaces. A routine marker bump (v1→v5
    already happened) would turn every leg into a raise, arm the backoff,
    and burn the batch's specs with nothing naming the image."""
    _healthy(monkeypatch)
    import docker as _docker

    class _NoImage:
        class images:
            @staticmethod
            def get(name):
                raise RuntimeError("404 Client Error: image not found")

        def ping(self):
            return True

        def close(self):
            pass

    monkeypatch.setattr(_docker, "from_env", lambda **kw: _NoImage())
    ok, why = RE.preflight()
    assert ok is False and "base image" in why


def test_the_credit_aggregate_does_not_add_opposite_facts(tmp_path,
                                                          monkeypatch):
    """`mattered_pos` means "the thing helped" for a perturbation that
    REMOVES and "the thing hurt" for one that ADDS. Summing them across
    kinds adds opposite facts, and that sum is what the health page
    renders."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    RE.write_credits([
        {"spec_id": "a", "verdict": RE.VERDICT_MATTERED_POS, "ts": "t",
         "perturbation": RE.PERTURB_LESSON_WITHHOLD, "sign": "helped",
         "noise_floor": 0.03},
        {"spec_id": "b", "verdict": RE.VERDICT_MATTERED_POS, "ts": "t",
         "perturbation": RE.PERTURB_LESSON_INJECT, "sign": "hurt",
         "noise_floor": 0.03},
    ])
    st = RE.credit_stats()
    assert st["by_verdict"][RE.VERDICT_MATTERED_POS] == 2
    assert st["by_sign"] == {"helped": 1, "hurt": 1}
    assert set(st["by_kind"]) == {RE.PERTURB_LESSON_WITHHOLD,
                                  RE.PERTURB_LESSON_INJECT}


def test_a_verdict_below_its_own_noise_floor_is_counted_separately(
        tmp_path, monkeypatch):
    """A `mattered_*` on a task whose pass rate makes the paired rule
    mislabel a quarter of decided nulls is not a result — and it was
    sitting in the same aggregate as the ones that are."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    RE.write_credits([
        {"spec_id": "a", "verdict": RE.VERDICT_MATTERED_POS, "ts": "t",
         "perturbation": RE.PERTURB_STEP_DENY, "sign": "helped",
         "noise_floor": 0.03},      # clean
        {"spec_id": "b", "verdict": RE.VERDICT_MATTERED_POS, "ts": "t",
         "perturbation": RE.PERTURB_STEP_DENY, "sign": "helped",
         "noise_floor": 0.26},      # a coin flip with a label
    ])
    st = RE.credit_stats()
    assert st["mattered_above_noise_floor"] == 1
    assert st["mattered_below_noise_floor"] == 1


def test_the_unapplied_count_is_surfaced(tmp_path, monkeypatch):
    """A high number means the specs are STALE — their lessons pruned,
    their steps never re-emitted — not that the world is null."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    RE.write_credits([
        {"spec_id": "a", "verdict": RE.VERDICT_ABSTAIN, "ts": "t",
         "applied": False},
        {"spec_id": "b", "verdict": RE.VERDICT_NO_EFFECT, "ts": "t",
         "applied": True},
    ])
    assert RE.credit_stats()["unapplied"] == 1


@pytest.mark.parametrize("p,n,expected", [
    (0.5, 2, 0.500), (0.7, 2, 0.2622), (0.8, 2, 0.1107),
    (0.7, 3, 0.1353), (0.8, 3, 0.0303), (0.9, 3, 0.0027),
])
def test_the_noise_floor_arithmetic(p, n, expected):
    """`2·pⁿ·qⁿ/(pⁿ+qⁿ)²` — the rate at which the unanimity rule
    mislabels a NULL perturbation as an effect. It is why `n_pairs` is 3
    and why the pre-registered 0.90 specificity bar is unreachable on a
    stochastic task at n=2."""
    assert abs(RE._noise_floor(p, n) - expected) < 5e-4


def test_n_pairs_is_the_value_the_arithmetic_asks_for():
    assert RE.DEFAULT_N_PAIRS >= 3
    assert RE._noise_floor(0.8, RE.DEFAULT_N_PAIRS) <= 0.10


@pytest.mark.parametrize("bad", [
    "import os\nos.system('rm -rf /')",
    "import os\nos.popen('curl x')",
    "eval(open('x').read())",
    "__import__('subprocess')",
])
def test_the_validator_screen_names_the_whole_class(bad):
    """A screen that names `subprocess` and not `os.system` reads as a
    closed door and is not one."""
    ok, why = RE.validator_is_admissible(bad)
    assert ok is False, bad


def test_a_corrected_outcome_is_a_different_question(tmp_path):
    """406 correction rows exist in the trajectory sidecar. An episode
    later corrected passed→failed is a DIFFERENT question, and without
    the outcome in the identity the dedup blocks re-asking while the
    credit row on disk keeps an outcome that is now wrong."""
    a = RE.ReplaySpec(trajectory_id="t1", perturbation="verify_toggle",
                      recorded_outcome="passed")
    b = RE.ReplaySpec(trajectory_id="t1", perturbation="verify_toggle",
                      recorded_outcome="failed")
    assert a.spec_id != b.spec_id


def test_a_network_error_past_the_truncation_is_still_seen():
    """`validator_output` is cut to 1,000 chars. Classifying on the cut
    copy missed a chatty validator's network error and graded it as a
    genuine failure."""
    chatty = ("x" * 2000) + "Temporary failure in name resolution"
    assert RE._network_failure(chatty) is True
    assert RE._network_failure(chatty[:1000]) is False


async def test_an_uncheckable_episode_is_not_a_vacuous_validator(
        tmp_path, monkeypatch):
    """Exit 2 is the validator's reserved "I cannot check this from the
    filesystem" — an honest statement about the EPISODE (a conversational
    turn whose deliverable was a reply), not a defect in the check. The
    first live smoke lumped the two together, which hid how much of the
    corpus is simply not replayable: 2 of 16 sampled episodes."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    monkeypatch.setenv("GHOST_DREAM_REPLAY", "1")
    traj = SimpleNamespace(
        id="t1", task_kind="user_request", outcome="passed", n_steps=3,
        user_request="write me a note about how the run felt", 
        final_response="", extra={},
        # ⚠ Produces an artifact, so D0 admits it — the point of THIS
        # test is the validator's exit 2, which now only fires for
        # episodes that made something whose correctness a filesystem
        # cannot judge. A pure-introspect turn no longer reaches the
        # validator at all: `REJECT_NO_ARTIFACT` catches it in triage,
        # which is the whole saving.
        tool_calls=[SimpleNamespace(
            name="file_system",
            arguments={"operation": "write", "path": "note.md"},
            result="ok", error="")] * 3)
    ctx = _ctx(tmp_path)
    ctx.llm_client.chat_completion = AsyncMock(return_value={"choices": [
        {"message": {"content": _GOOD}}]})
    _FakeSandbox.neg_exit = 2          # "cannot check from the filesystem"
    with patch("ghost_agent.sandbox.docker.DockerSandbox", _FakeSandbox), \
            patch("ghost_agent.core.agent.GhostAgent", _FakeAgent), \
            patch.object(RE.EpisodeSource, "_iter_real",
                         lambda self: iter([traj])):
        out = await RE.run_batch(ctx, limit=1)
    assert out["skipped"].get("episode_not_filesystem_checkable") == 1
    assert "validator_passes_an_empty_workspace" not in out["skipped"]
    assert out["validated"] == 0


# ------------------------------------------------------------------ #
# tool_ablate — the seeded perturbation D4's positive rests on        #
# ------------------------------------------------------------------ #

class _NoExecAgent(_FakeAgent):
    """An agent whose surface is missing one of the ablation's targets."""

    def __init__(self, context):
        super().__init__(context)
        self.available_tools.pop("execute", None)


async def test_a_PARTIAL_ablation_is_not_an_applied_perturbation(tmp_path,
                                                                 monkeypatch):
    """The check was `if _ablated and not run.ablated_tools` — a
    truthiness test on the INTERSECTION. Ablating file_system+execute
    when only one is in the surface passed it, the agent reached the
    artefact with the survivor, the validator passed, and every seeded
    positive silently became a NO_EFFECT miss: "the engine cannot detect
    perturbations", with no diagnostic anywhere."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    _FakeSandbox.next_exit = 0
    spec = _spec(perturbation=RE.PERTURB_TOOL_ABLATE,
                 target="file_system,execute")
    with patch("ghost_agent.sandbox.docker.DockerSandbox", _FakeSandbox), \
            patch("ghost_agent.core.agent.GhostAgent", _NoExecAgent):
        leg = await RE.run_leg(_ctx(tmp_path), spec, arm="perturbed",
                               validator=_GOOD)
    assert leg.applied is False
    assert "incomplete" in leg.reason and "execute" in leg.reason
    assert leg.passed is None


async def test_a_COMPLETE_ablation_applies_and_removes_both_tools(
        tmp_path, monkeypatch):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    _FakeSandbox.next_exit = 1          # no artefact ⇒ the validator fails
    spec = _spec(perturbation=RE.PERTURB_TOOL_ABLATE,
                 target="file_system,execute")
    seen = {}
    real_build = RE_ISO.IsolatedRun.build_agent

    def _spy(self, agent_cls=None, extra_forbidden=()):
        agent = real_build(self, agent_cls=agent_cls,
                           extra_forbidden=extra_forbidden)
        seen["ablated"] = set(self.ablated_tools)
        seen["allowed"] = set(self.allowed_tools)
        return agent

    monkeypatch.setattr(RE_ISO.IsolatedRun, "build_agent", _spy)
    with patch("ghost_agent.sandbox.docker.DockerSandbox", _FakeSandbox), \
            patch("ghost_agent.core.agent.GhostAgent", _FakeAgent):
        leg = await RE.run_leg(_ctx(tmp_path), spec, arm="perturbed",
                               validator=_GOOD)
    assert leg.applied is True and leg.passed is False
    assert seen["ablated"] == {"file_system", "execute"}
    assert not ({"file_system", "execute"} & seen["allowed"])


async def test_the_ablation_is_inert_on_the_control_arm(tmp_path,
                                                        monkeypatch):
    """The control arm must be the RECORDED condition. A perturbation
    that also fires on control is two identical arms with a label."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    _FakeSandbox.next_exit = 0
    spec = _spec(perturbation=RE.PERTURB_TOOL_ABLATE,
                 target="file_system,execute")
    seen = {}
    real_build = RE_ISO.IsolatedRun.build_agent

    def _spy(self, agent_cls=None, extra_forbidden=()):
        agent = real_build(self, agent_cls=agent_cls,
                           extra_forbidden=extra_forbidden)
        seen["ablated"] = set(self.ablated_tools)
        return agent

    monkeypatch.setattr(RE_ISO.IsolatedRun, "build_agent", _spy)
    with patch("ghost_agent.sandbox.docker.DockerSandbox", _FakeSandbox), \
            patch("ghost_agent.core.agent.GhostAgent", _FakeAgent):
        leg = await RE.run_leg(_ctx(tmp_path), spec, arm="control",
                               validator=_GOOD)
    assert seen["ablated"] == set()
    assert leg.applied is True and leg.passed is True


def test_tool_ablate_REMOVES_so_mattered_pos_means_the_tools_HELPED():
    """`_REMOVES` was not updated when the kind was added, so
    `_verdict_sign` filed it as ADDITIVE — "control passed, ablated leg
    failed" came out as "the tools HURT". `credit_stats` aggregates
    `by_sign`, so the first `write=True` run poisons the ledger with
    opposite facts summed together."""
    assert RE._verdict_sign(RE.PERTURB_TOOL_ABLATE,
                            RE.VERDICT_MATTERED_POS) == "helped"
    assert RE._verdict_sign(RE.PERTURB_TOOL_ABLATE,
                            RE.VERDICT_MATTERED_NEG) == "hurt"
    assert RE._verdict_sign(RE.PERTURB_LESSON_INJECT,
                            RE.VERDICT_MATTERED_POS) == "hurt"


def test_every_perturbation_kind_has_a_DECIDED_sign():
    """A kind that nobody classified defaults to ADDITIVE and reports the
    opposite of the truth. Adding one must fail here, not in the ledger."""
    adds = {RE.PERTURB_LESSON_INJECT}
    for kind in RE.PERTURB_KINDS:
        assert (kind in RE._REMOVES) ^ (kind in adds), (
            f"{kind} is in neither _REMOVES nor the ADDS set — its sign "
            f"is an accident of a default")
    assert "PERTURB_TOOL_ABLATE" in RE.__all__


async def test_an_EMPTY_ablation_target_is_not_an_applied_perturbation(
        tmp_path, monkeypatch):
    """R1: the partial-ablation guard covered `some of the targets were
    missing` but not `there were no targets`. An empty (or all-comma)
    target ablates nothing, so the perturbed arm is byte-identical to
    control and `decide_verdict` returns a confident `no_effect` about a
    counterfactual that never happened — which is the single thing
    `ReplayLeg.applied` exists to prevent."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    _FakeSandbox.next_exit = 0
    for target in ("", "   ", ",,"):
        spec = _spec(perturbation=RE.PERTURB_TOOL_ABLATE, target=target)
        with patch("ghost_agent.sandbox.docker.DockerSandbox", _FakeSandbox), \
                patch("ghost_agent.core.agent.GhostAgent", _FakeAgent):
            leg = await RE.run_leg(_ctx(tmp_path), spec, arm="perturbed",
                                   validator=_GOOD)
        assert leg.applied is False, target
        assert "no target" in leg.reason


def test_an_unclassified_perturbation_has_NO_sign(monkeypatch):
    """`perturbation in _REMOVES` sent everything unlisted down the
    ADDITIVE branch, so a typo, a case difference, an empty value or a
    kind nobody classified reported the exact OPPOSITE sign with full
    confidence — and `credit_stats` sums `by_sign` across kinds."""
    for bogus in ("", "tool_ablat", "TOOL_ABLATE", "something_new"):
        assert RE._verdict_sign(bogus, RE.VERDICT_MATTERED_POS) == "", bogus
        assert RE._verdict_sign(bogus, RE.VERDICT_MATTERED_NEG) == "", bogus
    # and the classified ones still answer
    assert RE._verdict_sign(RE.PERTURB_TOOL_ABLATE,
                            RE.VERDICT_MATTERED_POS) == "helped"
    assert RE._verdict_sign(RE.PERTURB_LESSON_INJECT,
                            RE.VERDICT_MATTERED_POS) == "hurt"


def test_the_two_sign_sets_PARTITION_the_registered_kinds():
    assert not (set(RE._REMOVES) & set(RE._ADDS))
    assert set(RE.PERTURB_KINDS) == set(RE._REMOVES) | set(RE._ADDS)


async def test_phase_3c_also_re_reads_the_clock_and_the_foreground_lock(
        tmp_path, monkeypatch):
    """R2 adjacent finding: 3c gated on the TICK-TOP `idle_secs` and did
    not check `foreground_requests` at all — the defect just fixed in
    3d, in the phase whose hour-long batch 3d's own comment cites as the
    reason to re-read. A batch spawns a container per leg, so starting
    one against a live user turn is the most expensive version of this
    mistake in the file."""
    import datetime
    from unittest.mock import AsyncMock
    from ghost_agent.core.agent import GhostAgent, GhostContext
    monkeypatch.setenv("GHOST_DREAM_REPLAY", "1")

    class _MidTick:
        foreground_tasks = 0

        def __init__(self):
            self._left = 1

        @property
        def foreground_requests(self):
            if self._left > 0:
                self._left -= 1
                return 0
            return 1

    ctx = MagicMock(spec=GhostContext)
    ctx.args = MagicMock()
    ctx.args.model = "m"
    ctx.args.no_dream = True
    ctx.args.no_self_play = False
    ctx.args.no_bench = True
    ctx.llm_client = _MidTick()
    ctx.memory_system = MagicMock()
    ctx.memory_system.collection.get.return_value = {"ids": []}
    ctx.profile_memory = MagicMock()
    ctx.scratchpad = MagicMock()
    ctx.skill_memory = None
    ctx.graph_memory = None
    ctx.journal = None
    ctx.sandbox_dir = "/tmp/sandbox"
    ctx.last_activity_time = (datetime.datetime.now()
                              - datetime.timedelta(seconds=9000))
    agent = GhostAgent(ctx)
    agent._last_selfplay_at = datetime.datetime.now()
    agent._last_bench_at = datetime.datetime.now()
    with patch("ghost_agent.core.replay_engine.run_batch",
               new=AsyncMock()) as batch:
        await agent._biological_tick()
    batch.assert_not_awaited()


async def test_phase_3c_advances_its_anchor_when_the_gate_is_CLOSED(
        tmp_path, monkeypatch):
    """`GHOST_DREAM_REPLAY` off is the DEFAULT, and that path fell
    through the anchor — so the gate re-imported and re-probed on every
    60-second tick for the whole idle stretch. 3d was fixed for exactly
    this and the rule was written in capitals; the round that edited 3c
    FOR CONSISTENCY WITH 3d copied the clock re-read and left the anchor
    behind."""
    import datetime
    from ghost_agent.core.agent import GhostAgent, GhostContext
    monkeypatch.delenv("GHOST_DREAM_REPLAY", raising=False)
    ctx = MagicMock(spec=GhostContext)
    ctx.args = MagicMock()
    ctx.args.model = "m"
    ctx.args.no_dream = True
    ctx.args.no_self_play = False
    ctx.args.no_bench = True
    ctx.llm_client = MagicMock()
    ctx.llm_client.foreground_tasks = 0
    ctx.llm_client.foreground_requests = 0
    ctx.memory_system = MagicMock()
    ctx.memory_system.collection.get.return_value = {"ids": []}
    ctx.profile_memory = MagicMock()
    ctx.scratchpad = MagicMock()
    ctx.skill_memory = None
    ctx.graph_memory = None
    ctx.journal = None
    ctx.sandbox_dir = "/tmp/sandbox"
    ctx.last_activity_time = (datetime.datetime.now()
                              - datetime.timedelta(seconds=9000))
    agent = GhostAgent(ctx)
    agent._last_selfplay_at = datetime.datetime.now()
    agent._last_bench_at = datetime.datetime.now()
    from unittest.mock import AsyncMock
    with patch("ghost_agent.core.replay_engine.run_batch",
               new=AsyncMock()) as batch:
        await agent._biological_tick()
    # the gate is CLOSED, so nothing may run…
    batch.assert_not_awaited()
    # …and the anchor must still advance, or the phase re-imports and
    # re-probes on every 60-second tick for the whole idle stretch.
    assert agent._last_dream_replay_at > datetime.datetime.min


# ---- one command must not be able to eat a whole leg ---------------- #

@pytest.mark.parametrize("leg,elapsed,want", [
    # ⚠ THE MEASURED DEFECT. A replayed episode ran `python3 -m
    # http.server 8080` — a foreground server that never returns — and
    # was handed `timeout -k 5s 569s` out of a 600 s leg, killing the
    # leg and the episode. The cap makes the leg size irrelevant.
    (600.0, 0.0, 120),
    (600.0, 100.0, 120),
    (240.0, 0.0, 120),
    # …and raising the LEG cap must not raise what one command can waste,
    # which is exactly what made the 240 -> 600 change worse.
    (300.0, 0.0, 120),
    (1200.0, 0.0, 120),
])
def test_ONE_COMMAND_cannot_spend_the_whole_leg(leg, elapsed, want):
    assert RE._cmd_budget(leg, elapsed) == want


@pytest.mark.parametrize("leg,elapsed,want", [
    # The remaining-leg bound still wins when it is the tighter one: a
    # command outliving its leg blocks an executor thread the leg's
    # cancellation cannot reach.
    (120.0, 0.0, 90),
    (100.0, 20.0, 50),
])
def test_the_REMAINING_leg_still_bounds_a_short_leg(leg, elapsed, want):
    assert RE._cmd_budget(leg, elapsed) == want


@pytest.mark.parametrize("leg,elapsed", [(600.0, 560.0), (600.0, 590.0),
                                         (60.0, 0.0), (30.0, 25.0)])
def test_a_nearly_spent_leg_still_gives_a_USABLE_slice(leg, elapsed):
    """The 30 s floor is deliberate and is not a third bound: handing a
    command one second produces a guaranteed-useless timeout rather than
    a shorter one."""
    assert RE._cmd_budget(leg, elapsed) == 30


def test_the_cap_is_STRICTLY_below_a_leg_budget():
    """The identity the whole fix rests on. If the cap ever equals or
    exceeds a leg's budget it stops bounding anything, and the defect
    returns silently — a leg is for MANY turns, one command is not."""
    assert RE.REPLAY_MAX_CMD_S < RE.DEFAULT_LEG_TIMEOUT_S
    # …and it must leave real room to recover after a timeout: a leg
    # should fit at least two capped commands plus turns around them.
    assert RE.REPLAY_MAX_CMD_S * 2 <= RE.DEFAULT_LEG_TIMEOUT_S


@pytest.mark.parametrize("leg_timeout,want,exact", [
    (300.0, 120, True),   # the cap binds, and elapsed cannot move it
    (600.0, 120, True),   # …and a bigger leg does NOT raise it
    # ⚠ NOT EXACT, and the executed pin is how I learned that: the pure
    # helper gives 90 for a 120 s leg at elapsed=0, but by the time
    # `run_leg` reaches the clamp real seconds have passed, so the
    # sandbox is handed 89. A test that demanded 90 would be pinning a
    # fiction the call site never produces.
    (120.0, 90, False),   # the remaining-leg bound binds instead
])
async def test_run_leg_ACTUALLY_CLAMPS_the_sandbox(tmp_path, monkeypatch,
                                                   leg_timeout, want,
                                                   exact):
    """⚠ AN EXECUTED PIN, NOT A TOKEN ONE. The defect lived at the CALL
    SITE, so asserting the helper's name appears in the source proves
    nothing about what the sandbox is handed — and a source-text pin is
    the exact failure mode this file keeps finding elsewhere. This drives
    the real `run_leg` and reads the value off the sandbox it built."""
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    seen = []

    class _RecordingSandbox(_FakeSandbox):
        def __setattr__(self, k, v):
            if k == "max_exec_timeout":
                seen.append(v)
            object.__setattr__(self, k, v)

    _FakeSandbox.next_exit = 0
    with patch("ghost_agent.sandbox.docker.DockerSandbox",
               _RecordingSandbox), \
            patch("ghost_agent.core.agent.GhostAgent", _FakeAgent):
        await RE.run_leg(_ctx(tmp_path), _spec(), arm="control",
                         validator=_GOOD, leg_timeout_s=leg_timeout)
    assert seen, "run_leg never clamped the sandbox at all"
    got = seen[-1]
    if exact:
        assert got == want, seen
    else:
        # …still strictly under the cap, which is the fact that matters:
        # the remaining-leg bound won, not REPLAY_MAX_CMD_S.
        assert got < RE.REPLAY_MAX_CMD_S, seen
        assert want - 10 <= got <= want, seen
