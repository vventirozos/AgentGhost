"""§4BF Track 2 flip (ii) — the tts_bon consumer in handle_chat and the
verifier-note round trip.

What is left pins LIVE behaviour: the tts_bon trigger/context-key contract,
the env-default BoN decision for unenrolled turns (no spec in the registry),
the repair-path precedence over BoN, the late-verdict handler on a starved
await, the oracle-vs-late-verifier rank in the trajectory collector, and the
single-block verifier note that `strip_system_notes` round-trips.

§4NM: the text-graded bench bank flavor (eval/banks.py), dream's bench solve
loop seams (answer.txt), the bench-scoped tts_bon arm (enrolled bench turns,
`verify_in_window` / `verify_late_pending` stamps) and the bench repair
budget were removed with the bench subsystem; their tests went with them.
"""
import json
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest



# ──────────────────────────────────────────────────────────────────
# experiments: the arm's registry contract
# ──────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────
# handle_chat: the tts_bon arm consumer (REAL handle_chat)
# ──────────────────────────────────────────────────────────────────

class _FakeBgTasks:
    def add_task(self, *a, **k):
        pass


def _bench_chat_context(tmp_path, item_id):
    from ghost_agent.core.agent import GhostContext
    from ghost_agent.distill.collector import TrajectoryCollector

    (tmp_path / "system" / "memory").mkdir(parents=True, exist_ok=True)
    (tmp_path / "system" / "experiments.json").write_text(json.dumps({
        "salt": "it",
        "experiments": [],
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
    context.args.model = "test-model"
    context.args.perfect_it = False
    context.profile_memory = MagicMock()
    context.profile_memory.get_context_string.return_value = ""
    context.memory_system = None
    context.simulated = True
    context.scratchpad = MagicMock()
    context.scratchpad.list_all.return_value = ""
    context.memory_dir = tmp_path / "system" / "memory"
    bench_root = tmp_path / "system" / "bench" / "trajectories"
    context.trajectory_collector = TrajectoryCollector(
        root=bench_root, session_id="bench-gsm8k_text")
    context.trajectory_task_kind = "bench"
    context.turn_origin_label = "bench"
    context.trajectory_extra_static = {"bench_bank": "gsm8k_text",
                                       "bench_item": item_id}
    context.trajectory_user_request_override = "What is 3*4-5?"
    context._last_bench_traj_id = ""
    return context, bench_root


def _bench_rows(bench_root):
    rows = []
    for day in sorted(p for p in bench_root.iterdir() if p.is_dir()):
        for f in sorted(day.glob("session-*.jsonl")):
            rows += [json.loads(l) for l in f.read_text().splitlines()
                     if l.strip()]
    return rows

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
async def test_no_wobble_means_no_stamp_and_no_bon(tmp_path, monkeypatch):
    from ghost_agent.core.agent import GhostAgent
    from ghost_agent.core.verifier import VerifyVerdict
    monkeypatch.delenv("GHOST_EXPERIMENTS", raising=False)
    monkeypatch.delenv("GHOST_TTS_ADAPTIVE_BON", raising=False)
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "0")
    context, bench_root = _bench_chat_context(tmp_path, "gsm8k_text-0")
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
    # must not enter the wobble-band block at all.
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
    context, bench_root = _bench_chat_context(tmp_path, "gsm8k_text-0")

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
    context, bench_root = _bench_chat_context(tmp_path, "gsm8k_text-0")

    final, bon = await _run_chat_turn(context, "bench-t2envoff1")

    assert bon.await_count == 0
    assert final == "All done — solved."


# ──────────────────────────────────────────────────────────────────
# R1 review pins (§4BF Track 2, 2026-08-13)
# ──────────────────────────────────────────────────────────────────

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
    context, bench_root = _bench_chat_context(tmp_path, "gsm8k_text-0")

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
    # final: the repair path must win and BoN must stay out. (§4NM: the
    # env default is ON here so that BoN WOULD fire without the repair
    # precedence — the bench-scoped treatment arm that used to arm it is
    # gone.)
    from ghost_agent.core.agent import GhostAgent
    monkeypatch.delenv("GHOST_EXPERIMENTS", raising=False)
    monkeypatch.setenv("GHOST_TTS_ADAPTIVE_BON", "1")
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "0")
    context, bench_root = _bench_chat_context(tmp_path, "gsm8k_text-0")

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


def test_r1_live_repair_budget_outlasts_the_vote_budget(tmp_path, monkeypatch):
    # R1 consumer C1 remedy, live half (§4NM: the 90 s bench-turn branch
    # went with the bench subsystem): live turns get the LIVE budget and
    # the env override still wins.
    #
    # The live default was 25s until 2026-08-22 and is 65s now (§4CK: the
    # REFUTED p90 is 63.8s, so 25s caught 5 refutes in 23). Asserted against
    # the real constant rather than a literal, so this pin keeps testing
    # the RELATIONSHIP — live > vote budget, env-override wins — instead
    # of freezing a number that is expected to be re-tuned.
    from ghost_agent.core.agent import GhostAgent
    monkeypatch.setenv("GHOST_CRITIC_ASYNC", "1")
    monkeypatch.delenv("GHOST_CRITIC_REPAIR_BUDGET", raising=False)
    context, _ = _bench_chat_context(tmp_path, "gsm8k_text-0")
    context.turn_origin_label = None
    agent = GhostAgent(context)
    from ghost_agent.core.verifier import _VOTE_BUDGET_DEFAULT_S

    _live = agent._critic_repair_await_budget()
    assert _live == 65.0
    # The pairing that makes the number mean something: the await must
    # outlast the ceiling the verdict bounds itself by, or it expires while
    # the verdict is still legitimately running and buys nothing.
    assert _live > _VOTE_BUDGET_DEFAULT_S
    monkeypatch.setenv("GHOST_CRITIC_REPAIR_BUDGET", "12")
    assert agent._critic_repair_await_budget() == 12.0


# ──────────────────────────────────────────────────────────────────
# R2 review pins (fixes-to-fixes round, 2026-08-13)
# ──────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────
# R3 review pins (2026-08-13)
# ──────────────────────────────────────────────────────────────────

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
    context, bench_root = _bench_chat_context(tmp_path, "gsm8k_text-0")
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
