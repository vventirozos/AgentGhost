"""§4BF Track 2 flip (ii) — the text-graded bank flavor + the tts_bon
bench arm consumer.

Four surfaces, each with the failure mode that motivated its pin:

  * banks: `graded_on` must stay OPTIONAL (adding it to ITEM_FIELDS
    would silently empty every pre-§4BF bank on disk) and the
    gsm8k_text validator's exit codes are load-bearing (5 = the seam
    didn't run = infra, never an agent failure);
  * dream's solve loop: the answer.txt seam must hand the validator the
    turn's ACTUAL final text, exit-5 must triage as infra, and the
    bench-local verifier must exist ONLY for final_response items (the
    C4 null stays for everything else);
  * the lesson-verify rerun: its pristine-restore deletes answer.txt,
    so without its own seam a text-graded lesson could never verify —
    the silent-inoperative class;
  * handle_chat's BoN block: the bench-scoped arm decides for enrolled
    turns (trigger stamped on BOTH arms), the env default only for
    unenrolled ones.
"""
import json
import subprocess
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.eval import banks
from ghost_agent.core import experiments as ex
from ghost_agent.core.dream import Dreamer


_ROW = {"question": "Tom has 3 boxes of 4 pens. He gives away 5. "
                    "How many pens does he have left?",
        "answer": "3*4 = <<3*4=12>>12 pens.\n12-5 = <<12-5=7>>7\n#### 7"}


# ──────────────────────────────────────────────────────────────────
# banks: the flavor itself
# ──────────────────────────────────────────────────────────────────

def test_item_fields_do_not_require_graded_on():
    # Mutation pin: adding graded_on to ITEM_FIELDS makes load_bank's
    # all-fields-present check drop every pre-§4BF row on disk.
    assert "graded_on" not in banks.ITEM_FIELDS


def test_item_graded_on_defaults_and_passthrough():
    assert banks.item_graded_on({}) == "artifact"
    assert banks.item_graded_on({"graded_on": ""}) == "artifact"
    assert banks.item_graded_on({"graded_on": "sideways"}) == "artifact"
    assert banks.item_graded_on(
        {"graded_on": "final_response"}) == "final_response"


def test_normalize_gsm8k_text_shape():
    it = banks.normalize_gsm8k_text(_ROW, 3)
    assert it["bank"] == "gsm8k_text"
    assert it["item_id"] == "gsm8k_text-3"
    assert it["graded_on"] == "final_response"
    # The reply contract the wobble-band A/B leans on: answer in the
    # reply, final numeric answer last.
    assert "in your reply" in it["challenge"]
    assert "FINAL NUMERIC ANSWER" in it["challenge"]
    # The validator grades answer.txt, never a solver artifact.
    assert "answer.txt" in it["validation_script"]
    assert "solution.py" not in it["validation_script"]


def test_gsm8k_text_roundtrips_through_bank_io(tmp_path):
    it = banks.normalize_gsm8k_text(_ROW, 0)
    banks.write_bank([it], "gsm8k_text", home=str(tmp_path))
    loaded = banks.load_bank("gsm8k_text", home=str(tmp_path))
    assert len(loaded) == 1
    assert banks.item_graded_on(loaded[0]) == "final_response"


def _run_validator(tmp_path, item, reply=None):
    if reply is not None:
        (tmp_path / "answer.txt").write_text(reply, encoding="utf-8")
    (tmp_path / ".validator.py").write_text(item["validation_script"],
                                            encoding="utf-8")
    return subprocess.run(["python3", ".validator.py"], cwd=tmp_path,
                          capture_output=True, text=True, timeout=20)


def test_text_validator_exit_codes(tmp_path):
    it = banks.normalize_gsm8k_text(_ROW, 0)
    # correct answer, last-line style
    assert _run_validator(tmp_path, it, "Step by step…\n7\n").returncode == 0
    # commas + prose around the number still resolve (last number wins)
    assert _run_validator(
        tmp_path, it, "12 pens minus 5 gives\n7").returncode == 0
    # wrong number → 1, and the gold is never leaked in the feedback
    p = _run_validator(tmp_path, it, "The answer is 9")
    assert p.returncode == 1
    assert "7" not in p.stdout
    # no number at all → 4 (a real solver failure — format was pinned)
    assert _run_validator(tmp_path, it, "I cannot solve this").returncode == 4
    # empty reply → 4, not a crash (the harness writes "" for an empty
    # final — that must grade as a failure, not as infra)
    assert _run_validator(tmp_path, it, "").returncode == 4


def test_text_validator_missing_answer_txt_is_exit_5(tmp_path):
    it = banks.normalize_gsm8k_text(_ROW, 0)
    p = _run_validator(tmp_path, it, reply=None)
    assert p.returncode == 5
    assert "infra" in p.stdout.lower()


def test_verify_reference_uses_answer_txt_for_text_items():
    it = banks.normalize_gsm8k_text(_ROW, 0)
    assert banks.verify_item_against_reference(it, "So the total is\n7")
    assert not banks.verify_item_against_reference(it, "So the total is\n8")


def test_verify_reference_still_runs_solutions_for_artifact_items():
    it = banks.normalize_gsm8k(_ROW, 0)
    assert banks.verify_item_against_reference(it, "print(7)")
    assert not banks.verify_item_against_reference(it, "print(8)")


# ──────────────────────────────────────────────────────────────────
# experiments: the arm's registry contract
# ──────────────────────────────────────────────────────────────────

def test_tts_bon_trigger_and_context_keys():
    assert ex.TRIGGER_KEYS.get("tts_bon") == "tts_bon_fired"
    # BoN substitutes the recorded final — replay consumers must be
    # able to exclude treatment turns.
    assert "tts_bon_fired" in ex.CONTEXT_MUTATING_KEYS


# ──────────────────────────────────────────────────────────────────
# dream: the solve-loop seams
# ──────────────────────────────────────────────────────────────────

_TEXT_ITEM = {
    "challenge": "What is 3*4-5? End your reply with the number.",
    "setup_script": banks._NO_SETUP,
    # The REAL generated validator, so the seam is tested against the
    # exact script production ships.
    "validation_script": banks.normalize_gsm8k_text(_ROW, 0)[
        "validation_script"],
}
_TEXT_META = {"bank": "gsm8k_text", "item_id": "gsm8k_text-0",
              "cluster": "math_text", "graded_on": "final_response"}

_FINAL_GOOD = "Three boxes of four is 12, minus 5:\n7"


def _make_context(tmp_path):
    context = MagicMock()
    context.memory_system = MagicMock()
    context.skill_memory = MagicMock()
    context.skill_memory.get_recent_failures.return_value = "No failures"
    context.llm_client = MagicMock()
    context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "{}"}}]})
    context.args = MagicMock()
    context.args.perfect_it = True
    context.args.smart_memory = 1.0
    context.sandbox_manager = MagicMock()
    context.sandbox_dir = str(tmp_path)
    context.tor_proxy = None
    context.scratchpad = MagicMock()
    context.frontier_tracker = None
    context.calibration_tracker = MagicMock()
    context.calibration_tracker.record_bench_validator_verdict = MagicMock(
        return_value=True)
    return context


def _wire_agent(mock_agent_cls, final_text):
    mock_agent = MagicMock()

    async def fake_handle_chat(body, **kw):
        body.setdefault("messages", []).extend([
            {"role": "assistant", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "ok"},
        ])
        return (final_text, None, None)

    mock_agent.handle_chat = AsyncMock(side_effect=fake_handle_chat)
    mock_agent._get_recent_transcript.return_value = "t" * 300
    mock_agent.disabled_tools = set()
    mock_agent.available_tools = {}
    mock_agent_cls.return_value = mock_agent
    return mock_agent


def _answer_txt_grading_sandbox(mock_sandbox_cls, seen):
    """A sandbox whose `.validator.py` step REALLY runs the validator in
    the temp sandbox dir — grading whatever answer.txt dream wrote —
    and records what it saw. This is what makes the seam test
    end-to-end: the PASS below can only come from dream's write."""
    sandbox = MagicMock()

    def execute(cmd, *a, **kw):
        if "py_compile" in cmd or ".setup.py" in cmd:
            return ("OK", 0)
        if ".validator.py" in cmd:
            d = Path(mock_sandbox_cls.call_args.args[0])
            seen.append((d / "answer.txt").read_text(encoding="utf-8")
                        if (d / "answer.txt").exists() else None)
            p = subprocess.run(["python3", ".validator.py"], cwd=d,
                               capture_output=True, text=True, timeout=20)
            return (p.stdout, p.returncode)
        return ("", 0)

    sandbox.execute.side_effect = execute
    mock_sandbox_cls.return_value = sandbox
    return sandbox


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_text_item_graded_on_the_reply_via_answer_txt(
        mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch,
        disable_self_play_templates):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    ctx = _make_context(tmp_path)
    _wire_agent(mock_agent_cls, _FINAL_GOOD)
    seen = []
    _answer_txt_grading_sandbox(mock_sandbox_cls, seen)

    dreamer = Dreamer(ctx)
    await dreamer.synthetic_self_play(
        "test-model", injected_challenge=dict(_TEXT_ITEM),
        bench_meta=dict(_TEXT_META))

    # The validator saw the turn's EXACT final text and passed it.
    assert seen == [_FINAL_GOOD]
    assert dreamer.last_bench_result and dreamer.last_bench_result["passed"]


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_text_item_wrong_reply_fails_normally(
        mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch,
        disable_self_play_templates):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    ctx = _make_context(tmp_path)
    _wire_agent(mock_agent_cls, "I think it is 9")
    seen = []
    _answer_txt_grading_sandbox(mock_sandbox_cls, seen)

    dreamer = Dreamer(ctx)
    await dreamer.synthetic_self_play(
        "test-model", injected_challenge=dict(_TEXT_ITEM),
        bench_meta=dict(_TEXT_META))

    assert dreamer.last_bench_result and not dreamer.last_bench_result["passed"]
    # A wrong ANSWER is charged to the agent: all three attempts graded.
    assert len(seen) == 3


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_missing_seam_exit_5_is_infra_not_agent_failure(
        mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch,
        disable_self_play_templates):
    # Simulate the seam NOT having run: the sandbox deletes answer.txt
    # before grading, so the validator exits 5 — the run must abort as
    # infra with NO fail verdict charged at the oracle rank.
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    ctx = _make_context(tmp_path)
    _wire_agent(mock_agent_cls, _FINAL_GOOD)
    sandbox = MagicMock()

    def execute(cmd, *a, **kw):
        if "py_compile" in cmd or ".setup.py" in cmd:
            return ("OK", 0)
        if ".validator.py" in cmd:
            d = Path(mock_sandbox_cls.call_args.args[0])
            (d / "answer.txt").unlink(missing_ok=True)
            p = subprocess.run(["python3", ".validator.py"], cwd=d,
                               capture_output=True, text=True, timeout=20)
            return (p.stdout, p.returncode)
        return ("", 0)

    sandbox.execute.side_effect = execute
    mock_sandbox_cls.return_value = sandbox

    dreamer = Dreamer(ctx)
    await dreamer.synthetic_self_play(
        "test-model", injected_challenge=dict(_TEXT_ITEM),
        bench_meta=dict(_TEXT_META))

    assert dreamer.last_bench_result
    assert not dreamer.last_bench_result["passed"]
    # INFRA, not an agent failure — and the loop aborted (1 attempt).
    assert "INFRA" in str(dreamer.last_bench_result.get("status") or "")
    assert mock_agent_cls.return_value.handle_chat.await_count == 1
    # No ground-truth negative at the strongest source rank.
    assert not any(
        c.args[1] is False for c in
        ctx.calibration_tracker.record_bench_validator_verdict
        .call_args_list)


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_bench_verifier_rebound_only_for_text_graded_items(
        mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch,
        disable_self_play_templates):
    from ghost_agent.core.verifier import Verifier
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    seen_verifiers = {}

    def capture(kind):
        def _wire(mock_cls, final):
            agent = _wire_agent(mock_cls, final)

            async def fake(body, **kw):
                solve_ctx = mock_cls.call_args_list[0].args[0]
                seen_verifiers[kind] = solve_ctx.verifier
                return (final, None, None)

            agent.handle_chat = AsyncMock(side_effect=fake)
            return agent
        return _wire

    # Text-graded run → a real bench-local Verifier on the isolate.
    ctx = _make_context(tmp_path)
    capture("text")(mock_agent_cls, _FINAL_GOOD)
    seen = []
    _answer_txt_grading_sandbox(mock_sandbox_cls, seen)
    await Dreamer(ctx).synthetic_self_play(
        "test-model", injected_challenge=dict(_TEXT_ITEM),
        bench_meta=dict(_TEXT_META))
    assert isinstance(seen_verifiers["text"], Verifier)

    # Artifact-graded run → the C4 null holds.
    mock_agent_cls.reset_mock()
    ctx2 = _make_context(tmp_path)
    capture("artifact")(mock_agent_cls, "done")
    sandbox = MagicMock()
    sandbox.execute.side_effect = lambda cmd, *a, **kw: ("OK", 0)
    mock_sandbox_cls.return_value = sandbox
    await Dreamer(ctx2).synthetic_self_play(
        "test-model",
        injected_challenge={"challenge": "avg a list",
                            "setup_script": banks._NO_SETUP,
                            "validation_script": "import sys; sys.exit(0)"},
        bench_meta={"bank": "mbpp", "item_id": "mbpp-7",
                    "cluster": "algo", "graded_on": "artifact"})
    assert seen_verifiers["artifact"] is None


# ──────────────────────────────────────────────────────────────────
# handle_chat: the tts_bon arm consumer (REAL handle_chat)
# ──────────────────────────────────────────────────────────────────

class _FakeBgTasks:
    def add_task(self, *a, **k):
        pass


def _bench_chat_context(tmp_path, item_id, spec=True):
    from ghost_agent.core.agent import GhostContext
    from ghost_agent.core.calibration import CalibrationTracker
    from ghost_agent.distill.collector import TrajectoryCollector

    (tmp_path / "system" / "memory").mkdir(parents=True, exist_ok=True)
    (tmp_path / "system" / "experiments.json").write_text(json.dumps({
        "salt": "it",
        "experiments": ([{"name": "tts_bon", "scope": "bench"}]
                        if spec else []),
    }), encoding="utf-8")

    context = MagicMock(spec=GhostContext)
    context.llm_client = MagicMock()
    context.llm_client.vision_clients = None
    context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "All done — solved.",
                                 "tool_calls": []}}]})
    context.sandbox_dir = str(tmp_path)
    context.args = MagicMock()
    context.args.shell = "bash"
    context.args.max_context = 8000
    context.args.temperature = 0.5
    context.args.smart_memory = 0.0
    context.args.use_planning = False
    context.args.model = "test-model"
    context.args.perfect_it = False
    context.profile_memory = MagicMock()
    context.profile_memory.get_context_string.return_value = ""
    context.memory_system = None
    context.skill_memory = types.SimpleNamespace(
        is_read_only=True,
        last_playbook_triggers=[],
        get_recent_failures=lambda *a, **k: "No failures")
    context.scratchpad = MagicMock()
    context.scratchpad.list_all.return_value = ""
    context.memory_dir = tmp_path / "system" / "memory"
    context.calibration_tracker = CalibrationTracker(
        tmp_path / "system" / "calibration")
    bench_root = tmp_path / "system" / "bench" / "trajectories"
    context.trajectory_collector = TrajectoryCollector(
        root=bench_root, session_id="bench-gsm8k_text")
    context.trajectory_task_kind = "bench"
    context.turn_origin_label = "bench"
    context.trajectory_extra_static = {"bench_bank": "gsm8k_text",
                                       "bench_item": item_id}
    context.trajectory_user_request_override = "What is 3*4-5?"
    context._last_bench_calib_req_id = ""
    context._last_bench_traj_id = ""
    return context, bench_root


def _bench_rows(bench_root):
    rows = []
    for day in sorted(p for p in bench_root.iterdir() if p.is_dir()):
        for f in sorted(day.glob("session-*.jsonl")):
            rows += [json.loads(l) for l in f.read_text().splitlines()
                     if l.strip()]
    return rows


def _arm_for_item(tmp_path, item_id):
    reg = ex.load_registry(tmp_path / "system" / "experiments.json")
    return reg.assign_all(f"bench|gsm8k_text|{item_id}",
                          scope=ex.SCOPE_BENCH)["tts_bon"]


def _find_items_for_both_arms(tmp_path):
    """Assignment is deterministic (salt 'it'): probe item ids until one
    of each arm is found, so the tests below assert EXACT behavior
    instead of branching on whatever arm they happened to draw."""
    (tmp_path / "system").mkdir(parents=True, exist_ok=True)
    (tmp_path / "system" / "experiments.json").write_text(json.dumps({
        "salt": "it",
        "experiments": [{"name": "tts_bon", "scope": "bench"}],
    }), encoding="utf-8")
    found = {}
    for i in range(64):
        arm = _arm_for_item(tmp_path, f"gsm8k_text-{i}")
        found.setdefault(arm, f"gsm8k_text-{i}")
        if len(found) == 2:
            return found
    raise AssertionError("no both-arm split in 64 items")


def _uncertain_verdict():
    from ghost_agent.core.verifier import VerifyVerdict
    return types.SimpleNamespace(
        verdict=VerifyVerdict.UNCERTAIN, confidence=0.5,
        issues=[], reasoning="cannot fully confirm", suspects=None)


async def _run_chat_turn(context, req_id):
    from ghost_agent.core.agent import GhostAgent
    agent = GhostAgent(context)
    agent.thinking_budget_override = "selfplay"
    body = {"messages": [{"role": "user", "content": "What is 3*4-5?"}]}
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[]), \
         patch("ghost_agent.core.agent._find_substantive_tool_for_verifier",
               return_value={"name": "execute_python"}), \
         patch.object(GhostAgent, "_compute_verifier_verdict",
                      AsyncMock(return_value=(
                          _uncertain_verdict(),
                          {"name": "execute_python"}))), \
         patch.object(GhostAgent, "_adaptive_bon_final",
                      AsyncMock(return_value=("BON WINNER", {
                          "substituted": True, "winner": 1,
                          "candidates": 2}))) as bon:
        final, _, _ = await agent.handle_chat(body, _FakeBgTasks(),
                                              request_id=req_id)
    return final, bon


@pytest.mark.asyncio
async def test_treatment_turn_runs_bon_and_stamps_true(
        tmp_path, monkeypatch):
    monkeypatch.delenv("GHOST_EXPERIMENTS", raising=False)
    monkeypatch.delenv("GHOST_TTS_ADAPTIVE_BON", raising=False)
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "0")
    ex.reset_registry_cache()
    items = _find_items_for_both_arms(tmp_path)
    context, bench_root = _bench_chat_context(
        tmp_path, items[ex.TREATMENT])

    final, bon = await _run_chat_turn(context, "bench-t2treat1")

    assert bon.await_count == 1
    assert final == "BON WINNER"
    traj = _bench_rows(bench_root)[-1]
    assert traj["extra"]["experiments"]["tts_bon"] == ex.TREATMENT
    assert traj["extra"]["tts_bon_fired"] is True


@pytest.mark.asyncio
async def test_control_turn_withholds_bon_and_stamps_false(
        tmp_path, monkeypatch):
    monkeypatch.delenv("GHOST_EXPERIMENTS", raising=False)
    # Env ON must NOT override a control assignment — the arm decides
    # for enrolled turns.
    monkeypatch.setenv("GHOST_TTS_ADAPTIVE_BON", "1")
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "0")
    ex.reset_registry_cache()
    items = _find_items_for_both_arms(tmp_path)
    context, bench_root = _bench_chat_context(
        tmp_path, items[ex.CONTROL])

    final, bon = await _run_chat_turn(context, "bench-t2ctrl1")

    assert bon.await_count == 0
    assert final == "All done — solved."
    traj = _bench_rows(bench_root)[-1]
    assert traj["extra"]["experiments"]["tts_bon"] == ex.CONTROL
    assert traj["extra"]["tts_bon_fired"] is False


@pytest.mark.asyncio
async def test_no_wobble_means_no_stamp_and_no_bon(tmp_path, monkeypatch):
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.core.verifier import VerifyVerdict
    monkeypatch.delenv("GHOST_EXPERIMENTS", raising=False)
    monkeypatch.delenv("GHOST_TTS_ADAPTIVE_BON", raising=False)
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "0")
    ex.reset_registry_cache()
    items = _find_items_for_both_arms(tmp_path)
    context, bench_root = _bench_chat_context(
        tmp_path, items[ex.TREATMENT])
    confirmed = types.SimpleNamespace(
        verdict=VerifyVerdict.CONFIRMED, confidence=0.95,
        issues=[], reasoning="ok", suspects=None)

    agent = GhostAgent(context)
    agent.thinking_budget_override = "selfplay"
    body = {"messages": [{"role": "user", "content": "What is 3*4-5?"}]}
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[]), \
         patch("ghost_agent.core.agent._find_substantive_tool_for_verifier",
               return_value={"name": "execute_python"}), \
         patch.object(GhostAgent, "_compute_verifier_verdict",
                      AsyncMock(return_value=(
                          confirmed, {"name": "execute_python"}))), \
         patch.object(GhostAgent, "_adaptive_bon_final",
                      AsyncMock()) as bon:
        await agent.handle_chat(body, _FakeBgTasks(),
                                request_id="bench-t2nowob1")

    assert bon.await_count == 0
    traj = _bench_rows(bench_root)[-1]
    # Presence of the key MEANS the trigger fired — a confident turn
    # must not enter the triggered-only block of either arm.
    assert "tts_bon_fired" not in traj["extra"]


@pytest.mark.asyncio
async def test_unenrolled_turn_falls_back_to_the_env_default(
        tmp_path, monkeypatch):
    # No tts_bon spec in the registry: arm_for returns "" and the env
    # default decides, exactly as before the flip. With it ON the BoN
    # runs but no trigger is stamped (there is no experiment to read).
    monkeypatch.delenv("GHOST_EXPERIMENTS", raising=False)
    monkeypatch.setenv("GHOST_TTS_ADAPTIVE_BON", "1")
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "0")
    ex.reset_registry_cache()
    context, bench_root = _bench_chat_context(
        tmp_path, "gsm8k_text-0", spec=False)

    final, bon = await _run_chat_turn(context, "bench-t2env1")

    assert bon.await_count == 1
    assert final == "BON WINNER"
    traj = _bench_rows(bench_root)[-1]
    assert "tts_bon_fired" not in traj.get("extra", {})


@pytest.mark.asyncio
async def test_unenrolled_turn_env_off_means_no_bon(tmp_path, monkeypatch):
    monkeypatch.delenv("GHOST_EXPERIMENTS", raising=False)
    monkeypatch.delenv("GHOST_TTS_ADAPTIVE_BON", raising=False)
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "0")
    ex.reset_registry_cache()
    context, bench_root = _bench_chat_context(
        tmp_path, "gsm8k_text-0", spec=False)

    final, bon = await _run_chat_turn(context, "bench-t2envoff1")

    assert bon.await_count == 0
    assert final == "All done — solved."


# ──────────────────────────────────────────────────────────────────
# R1 review pins (§4BF Track 2, 2026-08-13)
# ──────────────────────────────────────────────────────────────────

_NOTE_TAIL = "\n\n---\n**Verifier note:** step 2 of the derivation was not re-checked"


def test_r1_tolerance_cap_makes_large_golds_discriminating(tmp_path):
    # R1 bank MAJ-3: with a purely relative tolerance, gold 1,450,000
    # accepted ±1 (shipped on disk as gsm8k-611). The cap must fail an
    # off-by-one at ANY magnitude, in BOTH validator flavors.
    row = {"question": "Big factory output?", "answer": "#### 1450000"}
    text_it = banks.normalize_gsm8k_text(row, 0)
    assert _run_validator(tmp_path, text_it, "So:\n1450001").returncode == 1
    assert _run_validator(tmp_path, text_it, "So:\n1450000").returncode == 0
    art_it = banks.normalize_gsm8k(row, 0)
    assert banks.verify_item_against_reference(art_it, "print(1450000)")
    assert not banks.verify_item_against_reference(art_it, "print(1450001)")


def test_r1_tolerance_stays_tight_at_small_golds(tmp_path):
    # R1 bank MAJ-4 mutation: loosening _REL_TOL to 1e-3 survived the
    # old tests (which only distinguished 7 from 9).
    it = banks.normalize_gsm8k_text(_ROW, 0)
    assert _run_validator(tmp_path, it, "7.01").returncode == 1
    assert _run_validator(tmp_path, it, "7.0000001").returncode == 0


def test_r1_cluster_and_anticheat_pins():
    # R1 bank MAJ-4: reverting the cluster to "python_general" made
    # bench text lessons domain-eligible on real python queries while
    # passing every test. R1 bank MAJ-1: the dot-file prohibition was
    # missing from this flavor alone.
    it = banks.normalize_gsm8k_text(_ROW, 0)
    assert it["cluster"] == "math_text"
    assert "dot-files" in it["challenge"]


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_r1_seam_strips_verifier_note_false_fail_direction(
        mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch,
        disable_self_play_templates):
    # R1 CRIT (two independent reviewers): a correct answer whose final
    # carries the finalize-appended Verifier note (which ends in a
    # digit) graded FAIL because the note's number displaced the answer.
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    ctx = _make_context(tmp_path)
    _wire_agent(mock_agent_cls, _FINAL_GOOD + _NOTE_TAIL)
    seen = []
    _answer_txt_grading_sandbox(mock_sandbox_cls, seen)

    dreamer = Dreamer(ctx)
    await dreamer.synthetic_self_play(
        "test-model", injected_challenge=dict(_TEXT_ITEM),
        bench_meta=dict(_TEXT_META))

    # The validator saw the STRIPPED reply — and passed it.
    assert seen == [_FINAL_GOOD]
    assert dreamer.last_bench_result and dreamer.last_bench_result["passed"]


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_r1_seam_strips_verifier_note_false_pass_direction(
        mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch,
        disable_self_play_templates):
    # The worse direction: a WRONG answer whose refute note quotes the
    # correct value must still grade FAIL.
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    ctx = _make_context(tmp_path)
    _wire_agent(mock_agent_cls,
                "I believe it is 9"
                "\n\n---\n**Verifier note:** the evidence computes 7, not 9")
    seen = []
    _answer_txt_grading_sandbox(mock_sandbox_cls, seen)

    dreamer = Dreamer(ctx)
    await dreamer.synthetic_self_play(
        "test-model", injected_challenge=dict(_TEXT_ITEM),
        bench_meta=dict(_TEXT_META))

    assert dreamer.last_bench_result and not dreamer.last_bench_result["passed"]
    assert all("Verifier note" not in (s or "") for s in seen)


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_r1_empty_final_grades_as_agent_failure_not_infra(
        mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch,
        disable_self_play_templates):
    # R1 pin (both dream+bank reviewers): guarding the seam write with
    # `if final_ai_content:` would turn every empty-final give-up into
    # never-charged INFRA. The seam must write "" and grade exit-4.
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    ctx = _make_context(tmp_path)
    _wire_agent(mock_agent_cls, "")
    seen = []
    _answer_txt_grading_sandbox(mock_sandbox_cls, seen)

    dreamer = Dreamer(ctx)
    await dreamer.synthetic_self_play(
        "test-model", injected_challenge=dict(_TEXT_ITEM),
        bench_meta=dict(_TEXT_META))

    assert dreamer.last_bench_result
    assert not dreamer.last_bench_result["passed"]
    assert "INFRA" not in str(dreamer.last_bench_result.get("status") or "")
    assert seen == ["", "", ""]


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_r1_directory_named_answer_txt_is_cleared_not_infra(
        mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch,
        disable_self_play_templates):
    # R1 dream MAJ-3: a solver `mkdir answer.txt` made the seam write
    # raise into the blanket infra handler — an INFRA_ABORT escape
    # hatch from the fail label. The seam clears it and grades the
    # actual reply.
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    ctx = _make_context(tmp_path)
    mock_agent = _wire_agent(mock_agent_cls, _FINAL_GOOD)

    async def sabotage(body, **kw):
        d = Path(mock_sandbox_cls.call_args.args[0])
        (d / "answer.txt").mkdir(exist_ok=True)
        body.setdefault("messages", []).extend([
            {"role": "assistant", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "ok"},
        ])
        return (_FINAL_GOOD, None, None)

    mock_agent.handle_chat = AsyncMock(side_effect=sabotage)
    seen = []
    _answer_txt_grading_sandbox(mock_sandbox_cls, seen)

    dreamer = Dreamer(ctx)
    await dreamer.synthetic_self_play(
        "test-model", injected_challenge=dict(_TEXT_ITEM),
        bench_meta=dict(_TEXT_META))

    assert seen == [_FINAL_GOOD]
    assert dreamer.last_bench_result and dreamer.last_bench_result["passed"]


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_r1_lesson_verify_rerun_sees_fresh_stripped_answer(
        mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch,
        disable_self_play_templates):
    # R1 dream reviewer: dropping the lesson-verify seam (or its
    # graded_on threading) silently made every text-graded lesson
    # unverifiable — no prior test reached _verify_lesson_helpful with
    # a final_response item. Struggled-then-won: fail, pass, verify.
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    ctx = _make_context(tmp_path)
    # domains must come from the guard's fixed taxonomy — "math_text"
    # (the CLUSTER) is deliberately NOT a lesson domain: the cluster-
    # fallback fill for text-bench lessons drops at the guard, and only
    # extractor-labeled (content-true) domains mint. "algo" is what the
    # extractor plausibly emits for arithmetic strategy lessons.
    lesson_json = (
        '{"trigger": "multi-step arithmetic word problems", '
        '"anti_pattern": "skipping the units conversion step", '
        '"correct_pattern": "convert units before summing", '
        '"domains": ["algo"], "confidence": 0.7, '
        '"task": "word math", "mistake": "unit skip", '
        '"solution": "convert first"}')
    ctx.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": lesson_json}}]})

    mock_agent = MagicMock()
    finals = ["The answer is 9" + _NOTE_TAIL, _FINAL_GOOD, _FINAL_GOOD]
    calls = {"n": 0}

    async def fake_handle_chat(body, **kw):
        i = min(calls["n"], len(finals) - 1)
        calls["n"] += 1
        body.setdefault("messages", []).extend([
            {"role": "assistant", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "ok"},
        ])
        return (finals[i], None, None)

    mock_agent.handle_chat = AsyncMock(side_effect=fake_handle_chat)
    mock_agent._get_recent_transcript.return_value = "t" * 300
    mock_agent.disabled_tools = set()
    mock_agent.available_tools = {}
    mock_agent_cls.return_value = mock_agent

    seen = []
    _answer_txt_grading_sandbox(mock_sandbox_cls, seen)

    dreamer = Dreamer(ctx)
    await dreamer.synthetic_self_play(
        "test-model", injected_challenge=dict(_TEXT_ITEM),
        bench_meta=dict(_TEXT_META))

    assert dreamer.last_bench_result and dreamer.last_bench_result["passed"]
    # Attempt 1 (wrong, note stripped), attempt 2 (right), verify run.
    assert len(seen) == 3
    assert seen[2] == _FINAL_GOOD
    assert all(s is not None and "Verifier note" not in s for s in seen)


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_r1_wrapper_states_the_reply_contract_for_text_items(
        mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch,
        disable_self_play_templates):
    # R1 bank MAJ-2: the artifact wrapper ("stop as soon as your script
    # exits 0 … do not re-derive the answer") steered text-graded
    # solvers away from restating the number in the reply.
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    bodies = {}

    def capture(kind):
        agent = MagicMock()

        async def fake(body, **kw):
            bodies.setdefault(kind, str(body["messages"][0]["content"]))
            return ("7", None, None)

        agent.handle_chat = AsyncMock(side_effect=fake)
        agent._get_recent_transcript.return_value = "t" * 300
        agent.disabled_tools = set()
        agent.available_tools = {}
        mock_agent_cls.return_value = agent

    ctx = _make_context(tmp_path)
    capture("text")
    seen = []
    _answer_txt_grading_sandbox(mock_sandbox_cls, seen)
    await Dreamer(ctx).synthetic_self_play(
        "test-model", injected_challenge=dict(_TEXT_ITEM),
        bench_meta=dict(_TEXT_META))

    mock_agent_cls.reset_mock()
    ctx2 = _make_context(tmp_path)
    capture("artifact")
    sandbox = MagicMock()
    sandbox.execute.side_effect = lambda cmd, *a, **kw: ("OK", 0)
    mock_sandbox_cls.return_value = sandbox
    await Dreamer(ctx2).synthetic_self_play(
        "test-model",
        injected_challenge={"challenge": "avg a list",
                            "setup_script": banks._NO_SETUP,
                            "validation_script": "import sys; sys.exit(0)"},
        bench_meta={"bank": "mbpp", "item_id": "mbpp-7",
                    "cluster": "algo", "graded_on": "artifact"})

    assert "DELIVERABLE is" in bodies["text"]
    assert "Stop as soon as your script" not in bodies["text"]
    assert "Stop as soon as your script" in bodies["artifact"]
    assert "DELIVERABLE is" not in bodies["artifact"]


def test_r1_backfill_yields_to_bench_oracle(tmp_path):
    # R1 dream MAJ-1: the late verifier verdict (source="verifier_late",
    # yield_to_human=True) overwrote the bank oracle's ground truth in
    # the bench corrections sidecar (latest-wins). Oracles now stand.
    from ghost_agent.distill.collector import TrajectoryCollector
    from ghost_agent.distill.schema import Trajectory
    col = TrajectoryCollector(root=tmp_path / "traj", session_id="s")
    traj = Trajectory(user_request="q", outcome="unknown",
                      task_kind="bench")
    col.append(traj)
    assert col.update_outcome(traj.id, "passed", "",
                              source="bench_validator",
                              yield_to_human=True) is True
    res = col.update_outcome(traj.id, "failed", "llm thinks not",
                             source="verifier_late", yield_to_human=True)
    assert res == "withheld"
    rows = {t.id: t for t in col.iter_trajectories()}
    assert rows[traj.id].outcome == "passed"
    # Humans still outrank the oracle (rank order preserved).
    assert col.update_outcome(traj.id, "failed", "operator says no",
                              source="human_feedback:thumbs") is True
    assert col.update_outcome(traj.id, "passed", "",
                              source="bench_validator",
                              yield_to_human=True) == "withheld"


@pytest.mark.asyncio
async def test_r1_async_critic_verdict_lands_and_arms_the_trigger(
        tmp_path, monkeypatch):
    # R1 consumer C1/T1: every prior consumer test pinned sync mode; the
    # production config (GHOST_CRITIC_ASYNC=1) had zero coverage — a
    # one-line kill of the async await branch survived the suite.
    from ghost_agent.core.agent import GhostAgent
    monkeypatch.delenv("GHOST_EXPERIMENTS", raising=False)
    monkeypatch.delenv("GHOST_TTS_ADAPTIVE_BON", raising=False)
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    ex.reset_registry_cache()
    items = _find_items_for_both_arms(tmp_path)
    context, bench_root = _bench_chat_context(
        tmp_path, items[ex.TREATMENT])

    final, bon = await _run_chat_turn(context, "bench-t2async1")

    assert bon.await_count == 1
    assert final == "BON WINNER"
    traj = _bench_rows(bench_root)[-1]
    assert traj["extra"]["tts_bon_fired"] is True


@pytest.mark.asyncio
async def test_r1_async_verdict_missing_budget_means_no_trigger(
        tmp_path, monkeypatch):
    # The starvation shape itself: a verdict slower than the await
    # budget leaves the cache verdict-less — no stamp, no BoN, and the
    # late handler is attached instead.
    import asyncio as _aio
    from ghost_agent.core.agent import GhostAgent
    monkeypatch.delenv("GHOST_EXPERIMENTS", raising=False)
    monkeypatch.delenv("GHOST_TTS_ADAPTIVE_BON", raising=False)
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    monkeypatch.setenv("GHOST_CRITIC_REPAIR_BUDGET", "0.05")
    ex.reset_registry_cache()
    items = _find_items_for_both_arms(tmp_path)
    context, bench_root = _bench_chat_context(
        tmp_path, items[ex.TREATMENT])

    async def slow_verdict(*a, **kw):
        await _aio.sleep(1.0)
        return (_uncertain_verdict(), {"name": "execute_python"})

    agent = GhostAgent(context)
    agent.thinking_budget_override = "selfplay"
    body = {"messages": [{"role": "user", "content": "What is 3*4-5?"}]}
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[]), \
         patch("ghost_agent.core.agent._find_substantive_tool_for_verifier",
               return_value={"name": "execute_python"}), \
         patch.object(GhostAgent, "_compute_verifier_verdict",
                      side_effect=slow_verdict), \
         patch.object(GhostAgent, "_attach_late_verdict_handler",
                      MagicMock()) as late, \
         patch.object(GhostAgent, "_adaptive_bon_final",
                      AsyncMock()) as bon:
        final, _, _ = await agent.handle_chat(body, _FakeBgTasks(),
                                              request_id="bench-t2slow1")

    assert bon.await_count == 0
    assert final == "All done — solved."
    assert late.called
    traj = _bench_rows(bench_root)[-1]
    assert "tts_bon_fired" not in traj.get("extra", {})


@pytest.mark.asyncio
async def test_r1_repairing_final_never_runs_bon(tmp_path, monkeypatch):
    # R1 consumer T2: `if not _do_repair:` → `if True:` survived the
    # suite. Wobble (UNCERTAIN) + unverified-mutation repair on the same
    # final: the repair path must win and BoN must stay out.
    from ghost_agent.core.agent import GhostAgent
    monkeypatch.delenv("GHOST_EXPERIMENTS", raising=False)
    monkeypatch.delenv("GHOST_TTS_ADAPTIVE_BON", raising=False)
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "0")
    ex.reset_registry_cache()
    items = _find_items_for_both_arms(tmp_path)
    context, bench_root = _bench_chat_context(
        tmp_path, items[ex.TREATMENT])

    agent = GhostAgent(context)
    agent.thinking_budget_override = "selfplay"
    body = {"messages": [{"role": "user", "content": "What is 3*4-5?"}]}
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[]), \
         patch("ghost_agent.core.agent._find_substantive_tool_for_verifier",
               return_value={"name": "file_system"}), \
         patch("ghost_agent.core.agent._is_unverified_mutation",
               return_value=True), \
         patch.object(GhostAgent, "_compute_verifier_verdict",
                      AsyncMock(return_value=(
                          _uncertain_verdict(),
                          {"name": "file_system"}))), \
         patch.object(GhostAgent, "_adaptive_bon_final",
                      AsyncMock()) as bon:
        await agent.handle_chat(body, _FakeBgTasks(),
                                request_id="bench-t2repair1")

    assert bon.await_count == 0


def test_r1_bench_turns_get_a_patient_repair_budget(tmp_path, monkeypatch):
    # R1 consumer C1 remedy: bench turns await the verdict for 90s (deep
    # idle — the wait is free); live turns keep 25s; the env override
    # still wins everywhere.
    from ghost_agent.core.agent import GhostAgent
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    monkeypatch.delenv("GHOST_CRITIC_REPAIR_BUDGET", raising=False)
    ex.reset_registry_cache()
    context, _ = _bench_chat_context(tmp_path, "gsm8k_text-0")
    agent = GhostAgent(context)
    assert agent._critic_repair_await_budget() == 90.0
    context.turn_origin_label = None
    assert agent._critic_repair_await_budget() == 25.0
    monkeypatch.setenv("GHOST_CRITIC_REPAIR_BUDGET", "12")
    context.turn_origin_label = "bench"
    assert agent._critic_repair_await_budget() == 12.0


# ──────────────────────────────────────────────────────────────────
# R2 review pins (fixes-to-fixes round, 2026-08-13)
# ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_r2_symlink_answer_txt_is_unlinked_not_followed(
        mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch,
        disable_self_play_templates):
    # R2 MAJ-1: `ln -s .validator.py answer.txt` made the HOST-side seam
    # write follow the link and overwrite the ORACLE with reply text —
    # a reply that is valid Python exiting 0 then graded PASS at oracle
    # rank. The seam must unlink the LINK, never touch its target.
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    ctx = _make_context(tmp_path)
    mock_agent = _wire_agent(mock_agent_cls, "import sys\nsys.exit(0)\n")

    async def sabotage(body, **kw):
        d = Path(mock_sandbox_cls.call_args.args[0])
        (d / "answer.txt").unlink(missing_ok=True)
        (d / "answer.txt").symlink_to(d / ".validator.py")
        return ("import sys\nsys.exit(0)\n", None, None)

    mock_agent.handle_chat = AsyncMock(side_effect=sabotage)
    seen = []
    _answer_txt_grading_sandbox(mock_sandbox_cls, seen)

    dreamer = Dreamer(ctx)
    await dreamer.synthetic_self_play(
        "test-model", injected_challenge=dict(_TEXT_ITEM),
        bench_meta=dict(_TEXT_META))

    # The validator survived (it RAN and graded the reply text, which
    # carries no number → exit 4 → an agent-charged FAILURE) and the
    # oracle was never a PASS.
    assert dreamer.last_bench_result
    assert not dreamer.last_bench_result["passed"]
    assert "INFRA" not in str(dreamer.last_bench_result.get("status") or "")
    assert seen and all(s == "import sys\nsys.exit(0)\n" for s in seen)


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_r2_symlink_to_dir_answer_txt_is_cleared_not_infra(
        mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch,
        disable_self_play_templates):
    # R2 MAJ-1 sibling: a symlink TO A DIRECTORY defeated the is_dir()
    # guard (rmtree refuses symlinks; ignore_errors swallowed it; the
    # write then raised into the infra handler — the escape hatch).
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    ctx = _make_context(tmp_path)
    mock_agent = _wire_agent(mock_agent_cls, _FINAL_GOOD)

    async def sabotage(body, **kw):
        d = Path(mock_sandbox_cls.call_args.args[0])
        (d / "some_dir").mkdir(exist_ok=True)
        (d / "answer.txt").unlink(missing_ok=True)
        (d / "answer.txt").symlink_to(d / "some_dir")
        return (_FINAL_GOOD, None, None)

    mock_agent.handle_chat = AsyncMock(side_effect=sabotage)
    seen = []
    _answer_txt_grading_sandbox(mock_sandbox_cls, seen)

    dreamer = Dreamer(ctx)
    await dreamer.synthetic_self_play(
        "test-model", injected_challenge=dict(_TEXT_ITEM),
        bench_meta=dict(_TEXT_META))

    assert seen == [_FINAL_GOOD]
    assert dreamer.last_bench_result and dreamer.last_bench_result["passed"]


def test_r2_multiline_verifier_note_is_flattened_to_one_block():
    # R2 MAJ-2: issues/reasoning are raw LLM strings — a blank line
    # inside the appended note pushed its tail past strip_system_notes'
    # end-anchored no-\n\n regex, resurrecting the R1 CRIT. The writer
    # now enforces the single-block invariant, so the strip removes the
    # whole note again.
    import re as _re
    from ghost_agent.core.reply_smoothing import strip_system_notes
    issues = "step 2 unchecked.\n\nAlso the total should be 42"
    flat = _re.sub(r"\s*\n\s*", " ", issues).strip()
    final = f"…the answer is 7\n\n---\n**Verifier note:** {flat}"
    assert strip_system_notes(final) == "…the answer is 7"
    # And the UNFLATTENED shape really is the trap (documents why the
    # writer-side invariant exists — if reply_smoothing ever handles
    # multi-block notes natively, this assert flips and the flattening
    # can be retired).
    raw = f"…the answer is 7\n\n---\n**Verifier note:** {issues}"
    assert "42" in strip_system_notes(raw)


@pytest.mark.asyncio
async def test_r2_verify_in_window_stamp_rides_gate_reaching_finals(
        tmp_path, monkeypatch):
    # R2 rules MAJ-8: the starvation triage needs a durable "was a
    # verdict present at the decision point" bit — stamped on enrolled
    # turns regardless of wobble. R3 scope correction: only finals that
    # REACH the verifier gate (clean, non-repairing, no execution
    # failures) carry it; other exits form the rule's trigger-INELIGIBLE
    # bucket by their stamp ABSENCE.
    monkeypatch.delenv("GHOST_EXPERIMENTS", raising=False)
    monkeypatch.delenv("GHOST_TTS_ADAPTIVE_BON", raising=False)
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "0")
    ex.reset_registry_cache()
    items = _find_items_for_both_arms(tmp_path)
    context, bench_root = _bench_chat_context(
        tmp_path, items[ex.TREATMENT])

    # Wobble verdict → stamp True alongside the trigger.
    final, bon = await _run_chat_turn(context, "bench-t2win1")
    traj = _bench_rows(bench_root)[-1]
    assert traj["extra"]["verify_in_window"] is True
    assert traj["extra"]["tts_bon_fired"] is True


@pytest.mark.asyncio
async def test_r2_verify_in_window_false_when_no_verdict(
        tmp_path, monkeypatch):
    from ghost_agent.core.agent import GhostAgent
    monkeypatch.delenv("GHOST_EXPERIMENTS", raising=False)
    monkeypatch.delenv("GHOST_TTS_ADAPTIVE_BON", raising=False)
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "0")
    ex.reset_registry_cache()
    items = _find_items_for_both_arms(tmp_path)
    context, bench_root = _bench_chat_context(
        tmp_path, items[ex.TREATMENT])

    agent = GhostAgent(context)
    agent.thinking_budget_override = "selfplay"
    body = {"messages": [{"role": "user", "content": "What is 3*4-5?"}]}
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[]), \
         patch("ghost_agent.core.agent._find_substantive_tool_for_verifier",
               return_value={"name": "execute_python"}), \
         patch.object(GhostAgent, "_compute_verifier_verdict",
                      AsyncMock(return_value=(
                          None, {"name": "execute_python"}))), \
         patch.object(GhostAgent, "_adaptive_bon_final",
                      AsyncMock()) as bon:
        await agent.handle_chat(body, _FakeBgTasks(),
                                request_id="bench-t2nowin1")

    assert bon.await_count == 0
    traj = _bench_rows(bench_root)[-1]
    assert traj["extra"]["verify_in_window"] is False
    assert "tts_bon_fired" not in traj["extra"]


@pytest.mark.asyncio
async def test_r2_late_pending_stamp_marks_missed_window(
        tmp_path, monkeypatch):
    # Triage bit 2: verdict exists but missed the await window —
    # distinguishable from "no verdict possible".
    import asyncio as _aio
    from ghost_agent.core.agent import GhostAgent
    monkeypatch.delenv("GHOST_EXPERIMENTS", raising=False)
    monkeypatch.delenv("GHOST_TTS_ADAPTIVE_BON", raising=False)
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    monkeypatch.setenv("GHOST_CRITIC_REPAIR_BUDGET", "0.05")
    ex.reset_registry_cache()
    items = _find_items_for_both_arms(tmp_path)
    context, bench_root = _bench_chat_context(
        tmp_path, items[ex.TREATMENT])

    async def slow_verdict(*a, **kw):
        await _aio.sleep(1.0)
        return (_uncertain_verdict(), {"name": "execute_python"})

    agent = GhostAgent(context)
    agent.thinking_budget_override = "selfplay"
    body = {"messages": [{"role": "user", "content": "What is 3*4-5?"}]}
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[]), \
         patch("ghost_agent.core.agent._find_substantive_tool_for_verifier",
               return_value={"name": "execute_python"}), \
         patch.object(GhostAgent, "_compute_verifier_verdict",
                      side_effect=slow_verdict), \
         patch.object(GhostAgent, "_attach_late_verdict_handler",
                      MagicMock()), \
         patch.object(GhostAgent, "_adaptive_bon_final", AsyncMock()):
        await agent.handle_chat(body, _FakeBgTasks(),
                                request_id="bench-t2late1")

    traj = _bench_rows(bench_root)[-1]
    assert traj["extra"]["verify_late_pending"] is True
    assert traj["extra"]["verify_in_window"] is False


# ──────────────────────────────────────────────────────────────────
# R3 review pins (2026-08-13)
# ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_r3_hardlinked_answer_txt_cannot_truncate_the_oracle(
        mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch,
        disable_self_play_templates):
    # R3 MAJ-1: `os.link(".validator.py", "answer.txt")` passed both the
    # symlink and dir checks, and the host-side write truncated the
    # SHARED INODE — .validator.py became the reply text, and any reply
    # that is valid Python exiting 0 graded PASS at oracle rank. The
    # fresh-regular-file discipline (unlink + O_EXCL) writes a new inode
    # and leaves the oracle untouched.
    import os as _os
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    ctx = _make_context(tmp_path)
    mock_agent = _wire_agent(mock_agent_cls, "print('ok')")

    async def sabotage(body, **kw):
        d = Path(mock_sandbox_cls.call_args.args[0])
        (d / "answer.txt").unlink(missing_ok=True)
        _os.link(d / ".validator.py", d / "answer.txt")
        return ("print('ok')", None, None)

    mock_agent.handle_chat = AsyncMock(side_effect=sabotage)
    seen = []
    _answer_txt_grading_sandbox(mock_sandbox_cls, seen)

    dreamer = Dreamer(ctx)
    await dreamer.synthetic_self_play(
        "test-model", injected_challenge=dict(_TEXT_ITEM),
        bench_meta=dict(_TEXT_META))

    # The oracle survived, RAN, and graded the reply (no number → 4 →
    # agent-charged failure). Never a PASS, never infra.
    assert dreamer.last_bench_result
    assert not dreamer.last_bench_result["passed"]
    assert "INFRA" not in str(dreamer.last_bench_result.get("status") or "")
    assert seen and all(s == "print('ok')" for s in seen)


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_r3_fifo_answer_txt_cannot_wedge_the_seam(
        mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch,
        disable_self_play_templates):
    # R3 MAJ-2: a FIFO at answer.txt made the host-side open() block
    # forever with no reader — wedging the idle orchestrator (§4U). The
    # unconditional unlink removes it; the write completes and grades.
    import os as _os
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    ctx = _make_context(tmp_path)
    mock_agent = _wire_agent(mock_agent_cls, _FINAL_GOOD)

    async def sabotage(body, **kw):
        d = Path(mock_sandbox_cls.call_args.args[0])
        (d / "answer.txt").unlink(missing_ok=True)
        _os.mkfifo(d / "answer.txt")
        return (_FINAL_GOOD, None, None)

    mock_agent.handle_chat = AsyncMock(side_effect=sabotage)
    seen = []
    _answer_txt_grading_sandbox(mock_sandbox_cls, seen)

    dreamer = Dreamer(ctx)
    # The whole point: this must FINISH (a wedge here hangs the test
    # suite, which is the failure signal).
    await dreamer.synthetic_self_play(
        "test-model", injected_challenge=dict(_TEXT_ITEM),
        bench_meta=dict(_TEXT_META))

    assert seen == [_FINAL_GOOD]
    assert dreamer.last_bench_result and dreamer.last_bench_result["passed"]


def test_r3_production_note_builder_flattens_multiline_issues():
    # R3 MAJ-4: no test drove the PRODUCTION note writer with multiline
    # issues — deleting the flattening survived the suite. The builder
    # is now the module-level `_verifier_note_block` (the only author
    # of the marker; the post-loop gate is its sole call site), so the
    # invariant is pinned on the production code itself: whatever the
    # LLM put in issues/reasoning, the note is ONE block and
    # strip_system_notes round-trips to the bare reply.
    from ghost_agent.core.agent import _verifier_note_block
    from ghost_agent.core.reply_smoothing import strip_system_notes
    base = "…so the answer is 7"
    for nasty in (
        "step 2 unchecked.\n\nAlso the total should be 42",
        "line one\n\n\n\nline two 42",
        "already flat 42",
        "",
    ):
        final = base + _verifier_note_block(nasty)
        assert "\n\n" not in final[len(base) + 2:], nasty
        stripped = strip_system_notes(final)
        assert stripped == base, (nasty, stripped)


@pytest.mark.asyncio
async def test_r4_note_call_site_ships_flattened_note_on_final(
        tmp_path, monkeypatch):
    # R4 MAJ-1: the builder was pinned but the CALL SITE wasn't — an
    # inline reversion to the old f-string (no flattening) survived the
    # suite. Drive REAL handle_chat down the repair-budget-spent shape
    # (_MAX_VERIFIER_REPAIRS=0 skips the in-loop gate, so the post-loop
    # gate recomputes and annotates the SHIPPED final) and assert the
    # emitted note is one strippable block.
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.core.verifier import VerifyVerdict
    from ghost_agent.core.reply_smoothing import strip_system_notes
    monkeypatch.delenv("GHOST_EXPERIMENTS", raising=False)
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "0")
    monkeypatch.setattr(GhostAgent, "_MAX_VERIFIER_REPAIRS", 0)
    ex.reset_registry_cache()
    context, bench_root = _bench_chat_context(tmp_path, "gsm8k_text-0",
                                              spec=False)
    context.verifier = MagicMock()
    context.verifier.llm_client = MagicMock()
    # The gate reads the VERIFIER's client for the critic-pool decision
    # — an auto-mocked truthy critic_clients pushes the verdict to the
    # background late path and the note lands on no final at all.
    context.verifier.llm_client.critic_clients = None
    context.llm_client.critic_clients = None
    context.llm_client.worker_clients = None
    refuted = types.SimpleNamespace(
        verdict=VerifyVerdict.REFUTED, confidence=0.9,
        issues=["step 2 unchecked.\n\nAlso the total should be 42"],
        reasoning="", suspects=None)

    agent = GhostAgent(context)
    agent.thinking_budget_override = "selfplay"
    body = {"messages": [{"role": "user", "content": "What is 3*4-5?"}]}
    with patch("ghost_agent.core.agent.pretty_log"), \
         patch("ghost_agent.core.agent.get_active_tool_definitions",
               return_value=[]), \
         patch("ghost_agent.core.agent._find_substantive_tool_for_verifier",
               return_value={"name": "execute_python"}), \
         patch.object(GhostAgent, "_compute_verifier_verdict",
                      AsyncMock(return_value=(
                          refuted, {"name": "execute_python"}))):
        final, _, _ = await agent.handle_chat(body, _FakeBgTasks(),
                                              request_id="bench-t4note1")

    assert "**Verifier note:**" in final
    stripped = strip_system_notes(final)
    assert "Verifier note" not in stripped
    assert "42" not in stripped
    # R5: exact round-trip — filler injected BEFORE the note (which the
    # tail-anchored strip cannot remove) escaped the three asserts above.
    assert stripped == "All done — solved."
