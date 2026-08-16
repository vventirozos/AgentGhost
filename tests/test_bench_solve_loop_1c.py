"""§4BF 1c — the bench solve loop end-to-end (R1 review pins).

Runs the REAL ``Dreamer.synthetic_self_play`` with an injected bench item
(mocked GhostAgent + DockerSandbox, real bench trajectory root under a tmp
GHOST_HOME) and pins the seams the R1 reviewers proved were unpinned:

  * the mint tag — a bench struggled-then-won lesson carries
    ``source="bench"`` (mutating the tag to "self_play" survived the whole
    pre-R1 suite);
  * the frontier skip + neutral shape — a live FrontierTracker records
    NOTHING for bench, and a bench failure never mints (the
    self-play-shaped ``is_new_cluster=True`` default minted a junk lesson
    per failed item when the pin was removed);
  * the oracle verdict write-back — each attempt's verdict reaches the
    calibration re-label AND the bench trajectory corrections sidecar
    (without it bench arm stats stayed UNKNOWN forever);
  * the infra ordering — a sandbox banner defers/withholds the fail-side
    verdict instead of writing a permanent ground-truth negative;
  * the novelty pin — a bench FIRST-TRY pass defers (novelty is a frontier
    concept bench never accumulates; unpinned it minted unverified lessons
    on ~every pass).
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core.dream import Dreamer
from ghost_agent.memory.frontier import FrontierTracker


_LESSON_JSON = (
    '{"trigger": "computing aggregate statistics over a parsed list", '
    '"anti_pattern": "recomputing the aggregate inside the loop", '
    '"correct_pattern": "accumulate once, then divide by the count after '
    'the loop finishes", '
    '"domains": ["algo"], "confidence": 0.7, '
    '"task": "aggregate stats", "mistake": "loop recompute", '
    '"solution": "accumulate once"}'
)

_ITEM = {
    "challenge": "Write a python function that averages a list.",
    "setup_script": "# bench item: no setup files required\n",
    "validation_script": "import sys; sys.exit(0)",
}

_META = {"bank": "mbpp", "item_id": "mbpp-7", "cluster": "algo"}


def _make_context(tmp_path, frontier_tracker=None):
    context = MagicMock()
    context.memory_system = MagicMock()
    context.skill_memory = MagicMock()
    context.skill_memory.get_recent_failures.return_value = "No failures"
    context.llm_client = MagicMock()
    context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": _LESSON_JSON}}]})
    context.args = MagicMock()
    context.args.perfect_it = True
    context.args.smart_memory = 1.0
    context.sandbox_manager = MagicMock()
    context.sandbox_dir = str(tmp_path)
    context.tor_proxy = None
    context.scratchpad = MagicMock()
    context.frontier_tracker = frontier_tracker
    context.calibration_tracker = MagicMock()
    context.calibration_tracker.record_bench_validator_verdict = MagicMock(
        return_value=True)
    return context


def _stateful_sandbox(validator_results):
    """validator_results: list of (output, exit_code) consumed per
    `.validator.py` execution; the last entry repeats (the verify run)."""
    sandbox = MagicMock()
    state = {"i": 0}

    def execute(cmd, *a, **kw):
        if "py_compile" in cmd:
            return ("Syntax OK", 0)
        if ".setup.py" in cmd:
            return ("Setup OK", 0)
        if ".validator.py" in cmd:
            out = validator_results[min(state["i"],
                                        len(validator_results) - 1)]
            state["i"] += 1
            return out
        return ("", 0)

    sandbox.execute.side_effect = execute
    return sandbox


def _wire_agent(mock_agent_cls, stamp_only_first=False):
    """One shared mock agent whose handle_chat stamps the oracle join
    handles onto the SOLVE context (the first-constructed agent's context
    is dream's isolated_context), simulating what the real finalize does.

    ``stamp_only_first`` models an attempt whose calibration/trajectory
    writes were SKIPPED (truncated turn, metacog off) — the stale-join
    clears must then make later verdicts join NOTHING, not the previous
    attempt's rows."""
    mock_agent = MagicMock()
    counter = {"n": 0}

    async def fake_handle_chat(body, **kw):
        counter["n"] += 1
        # R4 pin: bench req_ids are PREFIXED by construction — the
        # no-collision claim in the calibration join and the docs rests
        # on it, and reverting to bare uuids survived the suite.
        _rid = kw.get("request_id")
        assert isinstance(_rid, str) and _rid.startswith("bench-"), _rid
        if not (stamp_only_first and counter["n"] > 1):
            solve_ctx = mock_agent_cls.call_args_list[0].args[0]
            solve_ctx._last_bench_calib_req_id = f"rq-{counter['n']}"
            solve_ctx._last_bench_traj_id = f"tj-{counter['n']}"
        body.setdefault("messages", []).extend([
            {"role": "assistant", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "ok"},
        ])
        return ("done", None, None)

    mock_agent.handle_chat = AsyncMock(side_effect=fake_handle_chat)
    mock_agent._get_recent_transcript.return_value = "t" * 300
    mock_agent.disabled_tools = set()
    mock_agent.available_tools = {}
    mock_agent_cls.return_value = mock_agent
    return mock_agent


def _corrections(tmp_path):
    p = (tmp_path / "system" / "bench" / "trajectories"
         / "corrections.jsonl")
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


async def _run(ctx, **kw):
    dreamer = Dreamer(ctx)
    await dreamer.synthetic_self_play("test-model",
                                      injected_challenge=dict(_ITEM),
                                      bench_meta=dict(_META), **kw)
    return dreamer


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_struggled_then_won_mints_bench_tag_and_writes_verdicts(
    mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch,
    disable_self_play_templates,
):
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    ft = FrontierTracker(tmp_path)
    ctx = _make_context(tmp_path, frontier_tracker=ft)
    _wire_agent(mock_agent_cls)
    mock_sandbox_cls.return_value = _stateful_sandbox(
        [("FAIL: expected 5 got 3", 1), ("Success", 0)])

    dreamer = await _run(ctx)

    # The run concluded as a bench SUCCESS in 2 attempts.
    assert dreamer.last_bench_result and dreamer.last_bench_result["passed"]
    # Mint tag (R1: mutating this to "self_play" survived the old suite).
    assert ctx.skill_memory.learn_lesson.called
    assert ctx.skill_memory.learn_lesson.call_args.kwargs["source"] == "bench"
    # Frontier stays REAL-ONLY — the live tracker recorded nothing.
    assert ft.get_cluster_stats("algo").get("runs", 0) == 0
    # Oracle verdicts: attempt 1 failed, attempt 2 passed — both stores.
    verdicts = [c.args for c in
                ctx.calibration_tracker.record_bench_validator_verdict
                .call_args_list]
    assert ("rq-1", False) in verdicts and ("rq-2", True) in verdicts
    # EXACTLY the two attempts — the lesson-VERIFY re-run (which fires on
    # this struggled-then-won shape) must contribute no third verdict and
    # no third correction row (R1 three-lens finding: the verify ctx
    # inherited every bench write path).
    assert len(verdicts) == 2
    rows = _corrections(tmp_path)
    assert len(rows) == 2
    got = {r["trajectory_id"]: r["outcome"] for r in rows}
    assert got.get("tj-1") == "failed" and got.get("tj-2") == "passed"
    assert all(r.get("source") == "bench_validator" for r in rows)


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_bench_failure_never_mints_even_with_a_live_tracker(
    mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch,
    disable_self_play_templates,
):
    # R1 mutation pin: removing the bench frontier skip re-arms the
    # self-play default (is_new_cluster=True) → every failed item minted
    # a junk lesson through "first failure on new cluster".
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    ft = FrontierTracker(tmp_path)
    ctx = _make_context(tmp_path, frontier_tracker=ft)
    _wire_agent(mock_agent_cls)
    mock_sandbox_cls.return_value = _stateful_sandbox(
        [("FAIL: expected 5 got 3", 1)])

    dreamer = await _run(ctx)

    assert dreamer.last_bench_result and not dreamer.last_bench_result["passed"]
    assert not ctx.skill_memory.learn_lesson.called
    assert ft.get_cluster_stats("algo").get("runs", 0) == 0
    # Every attempt's fail verdict still reached both stores.
    assert ctx.calibration_tracker.record_bench_validator_verdict.call_count == 3
    assert {r["outcome"] for r in _corrections(tmp_path)} == {"failed"}


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_first_try_pass_defers_the_lesson(
    mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch,
    disable_self_play_templates,
):
    # R1: novelty degenerates to ~1.0 for bench (no recorded winners), so
    # without the None pin every first-try pass minted UNVERIFIED lessons.
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    ctx = _make_context(tmp_path, frontier_tracker=FrontierTracker(tmp_path))
    _wire_agent(mock_agent_cls)
    mock_sandbox_cls.return_value = _stateful_sandbox([("Success", 0)])

    dreamer = await _run(ctx)

    assert dreamer.last_bench_result and dreamer.last_bench_result["passed"]
    assert not ctx.skill_memory.learn_lesson.called
    # The pass verdict still reached both stores.
    assert ctx.calibration_tracker.record_bench_validator_verdict \
        .call_args.args == ("rq-1", True)
    assert _corrections(tmp_path)[0]["outcome"] == "passed"


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_sandbox_banner_withholds_the_fail_verdict(
    mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch,
    disable_self_play_templates,
):
    # R1 CRIT: the oracle re-label ran BEFORE infra triage — a wedged
    # Docker daemon was written as a permanent ground-truth negative at
    # the strongest source rank, uncorrectable by the idempotency guard.
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    ctx = _make_context(tmp_path, frontier_tracker=FrontierTracker(tmp_path))
    _wire_agent(mock_agent_cls)
    mock_sandbox_cls.return_value = _stateful_sandbox(
        [("[SANDBOX INFRA ERROR — not your code] docker wedged", 1)])

    dreamer = await _run(ctx)

    assert not ctx.calibration_tracker.record_bench_validator_verdict.called
    assert _corrections(tmp_path) == []
    assert not ctx.skill_memory.learn_lesson.called
    # The run concluded as infra, not as an agent failure.
    status = (dreamer.last_bench_result or {}).get("status", "")
    assert status.startswith("INFRA_ABORT")


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_transient_banner_does_not_suppress_later_fail_verdicts(
    mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch,
    disable_self_play_templates,
):
    # R2 review MAJOR: the deferred fail-verdict guard read the STICKY
    # run-scoped infra flag — one transient banner on attempt 1 silently
    # suppressed the genuine fail verdicts of attempts 2-3, recreating
    # the exact "bench rows never resolve" inertness 1c exists to fix.
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    ctx = _make_context(tmp_path, frontier_tracker=FrontierTracker(tmp_path))
    _wire_agent(mock_agent_cls)
    mock_sandbox_cls.return_value = _stateful_sandbox([
        ("[SANDBOX INFRA ERROR — not your code] transient", 1),
        ("FAIL: expected 5 got 3", 1),
        ("FAIL: expected 5 got 3", 1),
    ])

    await _run(ctx)

    verdicts = [c.args for c in
                ctx.calibration_tracker.record_bench_validator_verdict
                .call_args_list]
    assert verdicts == [("rq-2", False), ("rq-3", False)]
    got = {r["trajectory_id"]: r["outcome"] for r in _corrections(tmp_path)}
    assert got == {"tj-2": "failed", "tj-3": "failed"}


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_skipped_write_joins_nothing_not_the_previous_attempt(
    mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch,
    disable_self_play_templates,
):
    # R2 review pin for the stale-join clears: attempt 1 hits a banner
    # (handles stamped), attempt 2 PASSES but its writes were skipped (no
    # stamp). Without the pre-attempt clears, attempt 2's PASS verdict
    # would join attempt 1's rows — a banner attempt re-labeled as a
    # ground-truth pass.
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    ctx = _make_context(tmp_path, frontier_tracker=FrontierTracker(tmp_path))
    _wire_agent(mock_agent_cls, stamp_only_first=True)
    mock_sandbox_cls.return_value = _stateful_sandbox([
        ("[SANDBOX INFRA ERROR — not your code] transient", 1),
        ("Success", 0),
    ])

    await _run(ctx)

    assert not ctx.calibration_tracker.record_bench_validator_verdict.called
    assert _corrections(tmp_path) == []


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_bench_isolate_gets_its_own_experiments_ring(
    mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch,
    disable_self_play_templates,
):
    # R3 review: 1c made bench turns ENROLLED, and copy.copy shares the
    # live 16-slot experiments ring — ~30 bench slots/night could evict a
    # live streamed turn's pending arm. The bench block must rebind a
    # fresh ring on the isolate (exactly like the two correction caches).
    from collections import OrderedDict
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    ctx = _make_context(tmp_path, frontier_tracker=FrontierTracker(tmp_path))
    live_ring = OrderedDict()
    live_ring["live-req"] = {"arms": {"use_planning": "treatment"},
                             "flags": {}}
    ctx._experiment_arms_recent = live_ring
    _wire_agent(mock_agent_cls)
    mock_sandbox_cls.return_value = _stateful_sandbox([("Success", 0)])

    await _run(ctx)

    iso = mock_agent_cls.call_args_list[0].args[0]
    assert iso._experiment_arms_recent is not live_ring
    # The live ring kept exactly its pre-seeded entry — no bench eviction.
    assert list(live_ring.keys()) == ["live-req"]
    # R4 CRIT pin: the isolate must carry NO production SelfModel — with
    # a collector attached, the finalize's selfhood capture became
    # reachable and bench attempts were writing first-person diary
    # entries into the live autobiographical store.
    assert iso.self_model is None


@pytest.mark.asyncio
@patch("ghost_agent.sandbox.docker.DockerSandbox")
@patch("ghost_agent.core.agent.GhostAgent")
async def test_bench_isolate_rebinds_the_full_ring_inventory(
    mock_agent_cls, mock_sandbox_cls, tmp_path, monkeypatch,
    disable_self_play_templates,
):
    # R5 review: each round found ONE shared LRU the previous round's
    # reset list missed (corrections → calib → experiments → turn_facts /
    # turn_outcome / surfaced_triggers). This pins the WHOLE inventory:
    # every copy.copy-shared ring the finalize writes through must be a
    # fresh object on the bench isolate.
    from collections import OrderedDict
    monkeypatch.setenv("GHOST_HOME", str(tmp_path))
    ctx = _make_context(tmp_path, frontier_tracker=FrontierTracker(tmp_path))
    ring_names = (
        "_recent_trajectories_for_correction",
        "_recent_calib_for_correction",
        "_experiment_arms_recent",
        "_turn_facts_recent",
        "_recent_turn_outcome",
        "_surfaced_triggers_by_traj",
    )
    shared = {}
    for name in ring_names:
        od = OrderedDict()
        od[f"live-{name}"] = object()
        setattr(ctx, name, od)
        shared[name] = od
    _wire_agent(mock_agent_cls)
    mock_sandbox_cls.return_value = _stateful_sandbox([("Success", 0)])

    await _run(ctx)

    iso = mock_agent_cls.call_args_list[0].args[0]
    for name in ring_names:
        assert getattr(iso, name) is not shared[name], \
            f"{name} is still ALIASED to the live ring on the bench isolate"
        assert list(shared[name].keys()) == [f"live-{name}"], \
            f"{name}: the live ring was mutated by the bench run"


def test_verify_ctx_is_stripped_of_every_bench_write_path():
    # Source-window pin (honestly labeled as such): the verify re-run's
    # sanitization cannot be exercised behaviorally in this harness
    # because GhostAgent is fully mocked — but its removal is exactly the
    # three-lens R1 finding, so its PRESENCE is pinned here and the
    # integration tests above pin its observable half (no third verdict
    # or correction row on the struggled-then-won shape).
    import inspect
    import ghost_agent.core.dream as dream_mod
    src = inspect.getsource(dream_mod)
    i = src.index("_verify_ctx = copy.copy(isolated_context)")
    window = src[i:i + 1800]
    for line in ("_verify_ctx.trajectory_collector = None",
                 "_verify_ctx.trajectory_task_kind = None",
                 "_verify_ctx.turn_origin_label = None",
                 "_verify_ctx.trajectory_extra_static = None",
                 "_verify_ctx.trajectory_user_request_override = None",
                 '_verify_ctx._last_bench_calib_req_id = ""',
                 '_verify_ctx._last_bench_traj_id = ""'):
        assert line in window, f"verify-ctx sanitization missing: {line}"
