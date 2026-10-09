"""§4MS (2026-10-09): the idle cycle — what runs while the owner is away.
Each test names the measured defect it fails on."""
from __future__ import annotations

import ast
import asyncio
import textwrap
import inspect
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.distill.schema import Trajectory

ROOT = Path(__file__).resolve().parents[1]


# ── owner safety ──────────────────────────────────────────────────────

def _bare_client():
    from ghost_agent.core.llm import LLMClient
    c = LLMClient.__new__(LLMClient)
    c.foreground_tasks = 0
    c.foreground_requests = 0
    c.main_slot_idle_windows = 0
    c._foreground_lock = asyncio.Lock()
    return c


async def test_a_parked_background_call_is_deferred_not_released_into_a_live_request(monkeypatch):
    """44 of 105 parks ran their call INSIDE a live owner request at 600 s;
    one owner call took 500 s instead of ~30 s."""
    from ghost_agent.core import llm as L
    c = _bare_client()
    c.foreground_requests = 1
    real = asyncio.sleep

    async def instant(_s):
        await real(0)
    monkeypatch.setattr(asyncio, "sleep", instant)
    with pytest.raises(L.BackgroundDeferred, match="selfplay-judge"):
        await c._wait_for_foreground_clear("selfplay-judge")


async def test_a_parked_call_runs_once_the_request_ends(monkeypatch):
    c = _bare_client()
    c.foreground_requests = 1
    real = asyncio.sleep
    n = {"i": 0}

    async def instant(_s):
        n["i"] += 1
        if n["i"] == 5:
            c.foreground_requests = 0
        await real(0)
    monkeypatch.setattr(asyncio, "sleep", instant)
    await c._wait_for_foreground_clear("x")          # returns, no raise


def _agent_with_llm(foreground=0):
    from ghost_agent.core.agent import GhostAgent
    a = GhostAgent.__new__(GhostAgent)
    a.context = SimpleNamespace(llm_client=SimpleNamespace(foreground_requests=foreground))
    return a


async def test_an_idle_job_stops_when_the_owner_arrives():
    """A self-play tick ran up to 2,140 s and never re-checked for the owner:
    19 owner turns waited a median 37 s for their first call (12 s clean)."""
    a = _agent_with_llm()
    cleaned = []

    async def job():
        try:
            await asyncio.sleep(30)
        finally:
            cleaned.append(1)
    t0 = time.monotonic()
    runner = asyncio.ensure_future(a._run_idle_job(job(), "self-play"))
    await asyncio.sleep(0.2)
    a.context.llm_client.foreground_requests = 1
    assert await runner is None
    assert time.monotonic() - t0 < 5 and cleaned == [1]


async def test_an_idle_job_has_a_wall_cap():
    a = _agent_with_llm()

    async def job():
        await asyncio.sleep(30)
    t0 = time.monotonic()
    assert await a._run_idle_job(job(), "self-play", cap_s=0.5) is None
    assert time.monotonic() - t0 < 5


async def test_an_idle_job_that_finishes_returns_its_result():
    a = _agent_with_llm()

    async def job():
        return {"replayed": 2}
    assert await a._run_idle_job(job(), "counterfactual") == {"replayed": 2}


def _agent_tree():
    import ghost_agent.core.agent as A
    return ast.parse(inspect.getsource(A))


def test_self_play_and_counterfactual_run_only_through_the_bounded_runner():
    """No bare await of synthetic_self_play / run_counterfactual_batch in the
    idle tick (the bench keeps its own wait_for)."""
    tree = _agent_tree()
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "_biological_tick")
    bad = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Await) and isinstance(n.value, ast.Call):
            f = n.value.func
            name = getattr(f, "attr", None) or getattr(f, "id", None)
            if name in ("synthetic_self_play", "run_counterfactual_batch"):
                bad.append(name)
    assert bad == []


# ── memory quality gates ──────────────────────────────────────────────

@pytest.mark.parametrize("text,invents", [
    ("When generating images, use the available imagination/creative parameters to enrich the scene.", True),
    ("Call web_search with the freshness_days argument for news.", True),
    ("Use browser operation=extract_text with max_chars=8000 before answering from snippets.", False),
    ("When a page 404s, pivot to web_search with a narrower query.", False),
    ("Pass the --force flag when the build cache is stale.", False),
    ("In the script, set retry_count=3 before the loop.", False),
    # r1: real dream rows that were flagged wrongly
    ("When calling manage_projects, ensure the action parameter matches the intended operation.", False),
    ("If page.query_selector fails in the browser tool, wait for the selector before extracting.", False),
])
def test_an_idle_lesson_naming_a_tool_interface_must_name_real_ones(text, invents):
    """A dream heuristic told the model to use "imagination/creative
    parameters" — none exist — and it reached an owner turn on 10-06."""
    from ghost_agent.memory.lesson_quality import heuristic_invents_interface
    assert heuristic_invents_interface(text) is invents


def test_the_live_tool_set_is_checked_not_only_the_static_list():
    """image_generation is advertised per context; the static list lacks it."""
    from ghost_agent.memory.lesson_quality import heuristic_invents_interface
    live = [{"type": "function", "function": {"name": "image_generation", "parameters": {
        "type": "object", "properties": {"prompt": {"type": "string"}, "subjects": {"type": "array"}}}}}]
    assert heuristic_invents_interface("Pass the `imagination_prompt` to image_generation.", live) is True
    assert heuristic_invents_interface("Pass the people in `subjects` to image_generation.", live) is False


def test_a_probe_turn_cannot_credit_a_lesson_as_helpful(tmp_path):
    """Lesson counters (pruning and graduation read them) were credited by
    probe turns: #104 44/42 with zero owner hydrations."""
    from ghost_agent.memory import skills as S
    from ghost_agent.utils.logging import request_id_context
    tok = request_id_context.set("probe-abc12345")
    try:
        assert S.usage_credit_blocked() is True
        sm = S.SkillMemory(tmp_path)
        assert sm.record_helpful_retrieval("anything") in (None, False, 0)
    finally:
        request_id_context.reset(tok)


def test_every_counter_method_uses_the_full_usage_gate():
    import ghost_agent.memory.skills as S
    for name in ("record_retrieval", "record_helpful_retrieval", "credit_recent_retrievals",
                 "record_retrievals_bulk"):
        tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(S.SkillMemory, name))))
        calls = {getattr(n.func, "id", "") for n in ast.walk(tree) if isinstance(n, ast.Call)}
        assert "usage_credit_blocked" in calls, name


def _dedent(src):
    import textwrap
    return textwrap.dedent(src)


@pytest.mark.parametrize("module,attr", [
    ("ghost_agent.core.foresight", None), ("ghost_agent.core.dream", None),
    ("ghost_agent.core.replay_engine", None), ("ghost_agent.optim.tool_fixtures", None),
])
def test_every_trajectory_reader_names_its_consumer(module, attr):
    """Five readers called iter_teachable WITHOUT a consumer: only member
    turns were filtered, PROBE turns got through."""
    import importlib
    tree = ast.parse(inspect.getsource(importlib.import_module(module)))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", getattr(n.func, "attr", "")) == "iter_teachable"]
    assert calls and all(any(k.arg == "consumer" for k in c.keywords) for c in calls), module


def test_a_probe_turn_never_moves_the_owners_competence_prior():
    tree = _agent_tree()
    sites = [n for n in ast.walk(tree) if isinstance(n, ast.If)
             and any(isinstance(b, ast.Expr) and isinstance(b.value, ast.Call)
                     and getattr(b.value.func, "attr", "") == "record_outcome"
                     and getattr(getattr(b.value.func, "value", None), "id", "") == "_mc"
                     for b in n.body)]
    assert sites and all("turn_origin" in ast.unparse(s.test) for s in sites)


def test_an_interrupted_turn_is_not_a_failure_and_teaches_nothing():
    """A client disconnect / restart was booked FAILED and taught: reflection
    reinforced a lesson from a turn the owner re-sent 2 s later."""
    from ghost_agent.distill.outcome_heuristics import TURN_INTERRUPTED_MARKER, classify_chat_outcome
    from ghost_agent.memory.skills import trajectory_may_teach
    t = Trajectory(user_request="explain x",
                   final_response=f"partial…\n\n{TURN_INTERRUPTED_MARKER} Turn aborted: cancelled: client disconnected.")
    assert classify_chat_outcome(t).outcome == "unknown"
    assert trajectory_may_teach(t) is False and trajectory_may_teach(t, consumer="reflection") is False
    aborted = Trajectory(user_request="x", final_response="[ATTEMPT_ABORTED_TURN] Turn aborted: stop.")
    # rule 0 must hold even where another rule would promote the turn
    cut_in_a_loop = Trajectory(user_request="x", extra={"loop_breaker": "repeat"},
                               final_response=f"{TURN_INTERRUPTED_MARKER} Turn aborted: shutdown.")
    assert classify_chat_outcome(cut_in_a_loop).outcome == "unknown"
    assert classify_chat_outcome(aborted).outcome == "failed"      # an owner Stop stays as it was


def test_a_disconnect_records_the_interruption_marker():
    tree = _agent_tree()
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", "") == "_record_aborted_turn"
             and any(k.arg == "reason" and "disconnected" in ast.unparse(k.value) for k in n.keywords)]
    assert calls and all(any(k.arg == "marker" and "interruption_marker" in ast.unparse(k.value)
                             for k in c.keywords) for c in calls)


def test_a_cancel_at_the_deadline_is_the_agents_failure(monkeypatch):
    """r1: the interface's 1800 s timeout reads like a disconnect — two of
    seven owner "disconnects" were the agent being too slow."""
    import ghost_agent.utils.logging as G
    from ghost_agent.core.agent import interruption_marker
    monkeypatch.setattr(G, "request_remaining_s", lambda rid: 2.0)
    assert interruption_marker("r1") == "[ATTEMPT_ABORTED_TURN]"
    monkeypatch.setattr(G, "request_remaining_s", lambda rid: 900.0)
    assert interruption_marker("r1") == "[TURN_INTERRUPTED]"
    monkeypatch.setattr(G, "request_remaining_s", lambda rid: None)
    assert interruption_marker("r1") == "[TURN_INTERRUPTED]"


# ── no-op churn ───────────────────────────────────────────────────────

def test_reflection_reads_only_failures_it_has_not_read():
    """152 of 157 passes reflected nothing ("0 of 29, dup-skipped 29")."""
    from ghost_agent.core.agent import fresh_reflectable
    a, b, c = (Trajectory(outcome="failed"), Trajectory(outcome="failed"), Trajectory(outcome="passed"))
    refl = SimpleNamespace(_is_reflectable=lambda t: t.outcome == "failed")
    assert fresh_reflectable([a, b, c], {a.id}, refl) == [b]
    assert fresh_reflectable([a, b], {a.id, b.id}, refl) == []


async def test_reflection_skips_without_a_new_failure_and_runs_with_one():
    from unittest.mock import AsyncMock, MagicMock
    from tests.test_reflection_biological_tick import _make_ctx, _tick
    from ghost_agent.reflection.loop import ReflectionRunReport
    for rows, expect in (([], 0), ([Trajectory(outcome="failed")], 1)):
        r = MagicMock()
        r.run = AsyncMock(return_value=ReflectionRunReport(seen_failures=1, reflected_ok=1))
        col = MagicMock()
        col.iter_trajectories = MagicMock(return_value=iter(rows))
        await _tick(_make_ctx(idle_secs=1200, reflector=r, collector=col))
        assert r.run.await_count == expect


def test_the_self_play_and_bench_anchors_reach_disk_when_set():
    """A restart mid-phase lost the anchor: self-play re-ran 85 min after a
    4 h one, bench inside its 6 h cooldown."""
    tree = _agent_tree()
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "_biological_tick")
    for anchor in ("_last_selfplay_at", "_last_bench_at"):
        found = False
        for body in (getattr(n, f, None) for n in ast.walk(fn) for f in ("body", "orelse", "finalbody")):
            if not isinstance(body, list):
                continue
            for i, st in enumerate(body[:-1]):
                if (isinstance(st, ast.Assign) and any(getattr(t, "attr", "") == anchor for t in st.targets)
                        and "now" in ast.unparse(st.value)
                        and "_sync_idle_anchors" in ast.unparse(body[i + 1])):
                    found = True
        assert found, anchor


# ── operator decisions ────────────────────────────────────────────────

class _Collector:
    def __init__(self, rows):
        self.rows = rows

    def iter_trajectories(self, since_days=None):
        return iter(self.rows)


def test_self_play_is_seeded_only_by_an_unpractised_owner_failure(tmp_path):
    """Operator: "retarget to my failures" — 341 runs, 93% first-try passes,
    no effect on owner turns."""
    from ghost_agent.core import owner_seeds as O
    from ghost_agent.distill.schema import ToolCall
    owner_fail = Trajectory(task_kind="user_request", outcome="failed", user_request="sum the CSV by month",
                            failure_reason="wrong totals", tool_calls=[ToolCall(name="execute")])
    no_tools = Trajectory(task_kind="user_request", outcome="failed", user_request="chat only")
    aborted = Trajectory(task_kind="user_request", outcome="failed", user_request="quick brown fox",
                         tool_calls=[ToolCall(name="execute")], final_response="[ATTEMPT_ABORTED_TURN] x")
    probe_fail = Trajectory(task_kind="probe", outcome="failed", user_request="probe")
    member_fail = Trajectory(task_kind="user_request", outcome="failed", user_request="m",
                             extra={"requester_role": "member"})
    passed = Trajectory(task_kind="user_request", outcome="passed", user_request="ok",
                        tool_calls=[ToolCall(name="execute")])
    interrupted = Trajectory(task_kind="user_request", outcome="failed", user_request="cut",
                             final_response="[TURN_INTERRUPTED] Turn aborted")
    col = _Collector([owner_fail, probe_fail, member_fail, passed, interrupted, no_tools, aborted])
    seed = O.pick_owner_failure_seed(col, tmp_path)
    assert seed and seed["source_id"] == owner_fail.id and seed["mode"] == "owner_failure"
    assert "sum the CSV by month" in seed["hint"] and "wrong totals" in seed["hint"]
    O.mark_used(tmp_path, owner_fail.id)
    assert O.pick_owner_failure_seed(col, tmp_path) is None


def test_no_owner_failure_means_no_self_play():
    """The phase raises its own stop before any self-play or replay when no
    seed exists (AST: the seed pick precedes the Dreamer)."""
    tree = _agent_tree()
    src = ast.unparse(next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)
                           and n.name == "_biological_tick"))
    i_seed = src.index("pick_owner_failure_seed")
    assert src.index("raise _NoOwnerSeed()") > i_seed
    assert src.index("dreamer = Dreamer(ctx)", i_seed) > src.index("raise _NoOwnerSeed()")
    assert "seed_override=_owner_seed" in src


def _bench_home(tmp_path, rows):
    b = tmp_path / "system" / "bench"
    b.mkdir(parents=True)
    (b / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(tmp_path)


def test_a_saturated_bank_is_retired_from_the_idle_walk(tmp_path, monkeypatch):
    """Operator: "retire saturated banks" — 94.5% pass, read by no owner turn."""
    from ghost_agent.eval import banks as B
    rows = ([{"bank": "gsm8k", "passed": True, "status": "SUCCESS"}] * 29
            + [{"bank": "gsm8k", "passed": False, "status": "FAILED"}]
            + [{"bank": "mbpp", "passed": i % 2 == 0, "status": "SUCCESS"} for i in range(30)]
            + [{"bank": "hard", "passed": True, "status": "SUCCESS"}] * 10
            + [{"bank": "mbpp", "passed": True, "status": "NO_RESULT"}] * 40)
    home = _bench_home(tmp_path, rows)
    monkeypatch.setattr(B, "bench_dir", lambda h=None: Path(home) / "system" / "bench")
    assert B.saturated_banks(home) == ["gsm8k"]            # mbpp 50%; "hard" has too few runs; NO_RESULT ignored
    monkeypatch.setenv("GHOST_BENCH_KEEP_BANKS", "gsm8k")
    assert B.saturated_banks(home) == []


def test_the_idle_bench_skips_saturated_banks_and_an_operator_drain_does_not():
    tree = _agent_tree()
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "pick_next_item"]
    assert calls and all(any(k.arg == "skip_saturated" and "not _drain_active" in ast.unparse(k.value)
                             for k in c.keywords) for c in calls)


# ── the cleanup ───────────────────────────────────────────────────────

def test_the_cleanup_plan_matches_the_preview_the_operator_confirmed():
    import importlib.util
    spec = importlib.util.spec_from_file_location("cleanup4ms", ROOT / "scripts" / "idle_cleanup_4ms.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    rows = [
        {"source": "dream", "trigger": "When generating images, use the available imagination/creative parameters.",
         "solution": "When generating images, use the available imagination/creative parameters."},
        {"source": "dream", "trigger": "When deleting a knowledge base document, confirm the specific item ID or name before execu",
         "solution": "When deleting a knowledge base document, confirm the specific item ID or name before execution."},
        {"source": "reflection", "source_trajectory_id": "de485e9e15aa", "trigger": "When renaming a parameter"},
        {"source": "dream", "trigger": "When running commands that operate on a file, verify the file path exists first",
         "solution": "When running commands that operate on a file, verify the file path exists first.",
         "retrievals": 44, "helpful_retrievals": 42},
        {"source": "dream", "trigger": "When asked to perform a task, prioritize using the most direct tool available (e",
         "solution": "When asked to perform a task, prioritize using the most direct tool available (e.g., use `manage_projects`)."},
        {"source": "self_play", "trigger": "SQL aggregation", "solution": "Use GROUP BY."},
    ]
    retract, reset, backfill = m.plan_playbook(rows)
    assert sorted(w for _, w in retract) == ["imagination-parameters", "probe-rooted",
                                             "stale-confirm-token-workaround"]
    assert reset == [3]
    assert [i for i, _a, _b in backfill] == [1, 4]
    assert backfill[1][2].endswith("(e.g., use `manage_projects`).")



def test_a_seeded_run_is_generated_from_the_brief_never_a_template_or_journal_pick():
    """r1: the seed's hint reached no generator — every seeded run became a
    random CSV/SQL template (all live seeds have no cluster)."""
    import ghost_agent.core.dream as D
    tree = ast.parse(inspect.getsource(D))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "synthetic_self_play")
    src = ast.unparse(fn)
    assert "_tpl = None if _owner_mode else try_template(" in src
    branch_tests = [ast.unparse(n.test) for n in ast.walk(fn) if isinstance(n, ast.If)
                    and "gen_ok" in ast.unparse(n.test)
                    and ("_try_journal_challenge" in ast.unparse(n) or "pick_random_template" in ast.unparse(n))
                    and "force_template" not in ast.unparse(n.test)]
    assert branch_tests and all("_owner_mode" in t for t in branch_tests), branch_tests


def test_a_seed_is_used_only_by_a_run_that_concluded_and_no_seed_is_a_declined_heartbeat():
    tree = _agent_tree()
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "_biological_tick")
    ifs = [n for n in ast.walk(fn) if isinstance(n, ast.If) and "_mark_seed_used" in ast.unparse(n)
           and not any("_mark_seed_used" in ast.unparse(b) for b in [n.test])]
    assert any("last_self_play_status" in ast.unparse(n.test) for n in ifs)
    handlers = [h for h in ast.walk(fn) if isinstance(h, ast.ExceptHandler) and getattr(h.type, "id", "") == "_NoOwnerSeed"]
    assert handlers and "'self_play', 'declined'" in ast.unparse(handlers[0])


def test_the_cleanup_applies_completely_and_a_rerun_changes_nothing(tmp_path):
    """r1: --apply crashed after rewriting the playbook and a re-run was
    refused; the backfill orphaned rows from retrieval (dropped)."""
    import subprocess
    import sys as _sys
    import chromadb
    from chromadb.config import Settings
    mem = tmp_path / "system" / "memory"
    (mem / "acquired_skills").mkdir(parents=True)
    rows = [
        {"source": "dream", "trigger": "When generating images, use the available imagination/creative parameters.",
         "task": "x", "solution": "When generating images, use the available imagination/creative parameters."},
        {"source": "dream", "trigger": "When running commands that operate on a file, verify the file path exists first",
         "task": "y", "retrievals": 44, "helpful_retrievals": 42, "failed_retrievals": 5},
        {"source": "self_play", "trigger": "SQL aggregation", "task": "z"},
    ]
    (mem / "skills_playbook.json").write_text(json.dumps(rows))
    (mem / "acquired_skills" / "skills_registry.json").write_text(json.dumps(
        {"extract_html_content": {"name": "extract_html_content"}, "news_headlines": {"name": "news_headlines"}}))
    (mem / "acquired_skills" / "extract_html_content.py").write_text("x = 1\n")
    col = chromadb.PersistentClient(path=str(mem), settings=Settings(anonymized_telemetry=False)) \
        .get_or_create_collection("agent_memory")
    col.add(ids=["bad", "ok"], embeddings=[[0.1] * 8, [0.2] * 8],
            documents=["The user is aware of potential pro-China alignment biases in AI systems.", "keep me"])
    del col
    env = {**os.environ, "GHOST_HOME": str(tmp_path) + "/", "PYTHONPATH": str(ROOT / "src"), "HF_HUB_OFFLINE": "1"}
    outs = []
    for _ in range(2):
        p = subprocess.run([_sys.executable, str(ROOT / "scripts" / "idle_cleanup_4ms.py"), "--apply"],
                           env=env, capture_output=True, text=True, timeout=300)
        assert p.returncode == 0, p.stderr[-800:]
        outs.append(json.loads(p.stdout[p.stdout.index("{"):]))
    assert outs[0]["retracted"] == 1 and outs[0]["vector_deleted"] == 1 and outs[0]["skill_retired"] == "extract_html_content"
    assert outs[1]["retracted"] == 0 and outs[1]["vector_deleted"] == 0 and outs[1]["skill_retired"] is None
    after = json.loads((mem / "skills_playbook.json").read_text())
    assert len(after) == 2 and after[0]["retrievals"] == 0 and after[0]["failed_retrievals"] == 5
    arch = [json.loads(l) for l in (mem / "skills_pruned_archive.jsonl").read_text().splitlines()]
    assert arch[0]["reason"] == "removed_by_trigger" and isinstance(arch[0]["lesson"], dict)
    col = chromadb.PersistentClient(path=str(mem), settings=Settings(anonymized_telemetry=False)).get_collection("agent_memory")
    assert col.get(ids=["bad"])["ids"] == [] and col.get(ids=["ok"])["ids"] == ["ok"]
    assert (mem / "acquired_skills" / "retired" / "extract_html_content.py").exists()


def test_two_snapshots_in_one_second_both_publish(tmp_path, monkeypatch):
    """§4MS battery: a cleanup re-run inside the same second collided with
    the first snapshot's name and the publish failed (ENOTEMPTY)."""
    import time as _time
    from ghost_agent.memory import snapshot as S
    (tmp_path / "system" / "memory").mkdir(parents=True)
    (tmp_path / "system" / "memory" / "x.json").write_text("{}")
    _fixed = _time.gmtime(0)
    monkeypatch.setattr(S.time, "gmtime", lambda *a: _fixed)
    a = S.take_snapshot(tmp_path, None, "t")
    b = S.take_snapshot(tmp_path, None, "t")
    assert a["ok"] and b["ok"] and a["path"] != b["path"]



async def test_with_no_owner_failure_the_self_play_phase_never_builds_a_dreamer(monkeypatch):
    """Behaviour, not text: no seed → no Dreamer, no self-play, a declined
    heartbeat."""
    from unittest.mock import AsyncMock, MagicMock, patch
    from tests.test_biological_watchdog import _make_agent
    import ghost_agent.core.owner_seeds as OS
    monkeypatch.setattr(OS, "pick_owner_failure_seed", lambda *a, **k: None)
    agent = _make_agent(idle_seconds=4000)
    attempts = []
    agent._record_idle_attempt = lambda phase, result="entered": attempts.append((phase, result))
    with patch("ghost_agent.core.dream.Dreamer") as MockDreamer, \
            patch("ghost_agent.core.agent.random.random", return_value=0.05):
        await agent._biological_tick()
    MockDreamer.assert_not_called()
    assert ("self_play", "declined") in attempts


def test_an_owner_seed_is_the_runs_seed_and_skips_the_frontier_pick():
    import ghost_agent.core.dream as D
    seed = D.initial_self_play_seed({"mode": "owner_failure", "hint": "practise X", "source_id": "t1",
                                     "cluster_key": None})
    assert seed["mode"] == "owner_failure" and seed["hint"] == "practise X"
    assert D.initial_self_play_seed(None)["mode"] == "cold_start"
    assert D.initial_self_play_seed({"mode": "owner_failure", "hint": ""})["mode"] == "cold_start"
    tree = ast.parse(inspect.getsource(D))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "synthetic_self_play")
    assert "seed = initial_self_play_seed(seed_override)" in ast.unparse(fn)
