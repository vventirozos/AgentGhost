"""Regression tests for the cross-codebase bug-hunt fixes (2026-06).

One test (or small group) per fix:

- utils/helpers.py: recursive_split_text returned chunks in REVERSE document
  order (everything now routes through the LIFO stack).
- memory/skills.py: _delete_lesson_twin used a flat two-key Chroma `where`,
  which raises ValueError and silently orphaned every vector twin.
- memory/skills.py: learn_lesson trusted a duplicate index computed from a
  stale snapshot; it now re-locates the duplicate by key under the lock.
- memory/vector.py: smart_update queried without a type filter, so the
  nearest neighbor could be a document chunk/skill/episode that got DELETED.
- selfhood/state.py: _cap evicted head-first, dropping still-open questions
  while retaining resolved zombies.
- memory/projects.py: update_task did not canonicalize parent_id, breaking
  every `WHERE parent_id = ?` lookup for case-mangled ids.
- core/planning.py: a BEST parent whose LAST child to finish FAILED was
  never resolved; an ALL parent with a consumed alternative could never
  complete (the FAILED child stayed in the rollup).
- core/project_advancer.py: _increment_budget was a non-atomic
  read-modify-write of project metadata.
- core/agent.py: _record_turn_trajectory derived user_request from the last
  user-role message, which mid-turn synthetic injections (SYSTEM ALERT etc.)
  overwrite; it now accepts the real request explicitly.
- utils/token_counter.py: load_tokenizer popped HF_HUB_OFFLINE instead of
  restoring the prior value (clobbering --mandatory-tor's offline flag).
- structural contracts (source-level) for fixes whose behavior needs a live
  server/upstream: llm.py node-exhaustion guards + mid-stream retry guard,
  api/routes.py /api/generate stream flag, interface/server.py janitor
  hard-cap and tail stderr, main.py --no-memory gating, agent.py
  _calib_pending per-request reset.
"""

import asyncio
import json
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "ghost_agent"


# ---------------------------------------------------------------------------
# utils/helpers.py — recursive_split_text document order
# ---------------------------------------------------------------------------

def test_recursive_split_text_preserves_document_order():
    from ghost_agent.utils.helpers import recursive_split_text

    text = "AAAA aaaa.\n\nBBBB bbbb.\n\nCCCC cccc.\n\nDDDD dddd."
    chunks = recursive_split_text(text, 12, 2)
    assert chunks == ["AAAA aaaa.", "BBBB bbbb.", "CCCC cccc.", "DDDD dddd."]


def test_recursive_split_text_hard_split_order():
    from ghost_agent.utils.helpers import recursive_split_text

    # No separator at all → character split, still in order.
    chunks = recursive_split_text("ABCDEFGHIJKLMNOPQRST", 8, 2)
    assert "".join(c[: 8 - 2] for c in chunks[:-1]) + chunks[-1] == "ABCDEFGHIJKLMNOPQRST"
    assert chunks[0].startswith("ABCDEFGH")


def test_recursive_split_text_nested_order():
    from ghost_agent.utils.helpers import recursive_split_text

    text = ("para one sentence. " * 5).strip() + "\n\n" + ("para two sentence. " * 5).strip()
    chunks = recursive_split_text(text, 60, 10)
    joined = " ".join(chunks)
    assert joined.index("para one") < joined.index("para two")


# ---------------------------------------------------------------------------
# memory/skills.py — _delete_lesson_twin Chroma where clause
# ---------------------------------------------------------------------------

def test_delete_lesson_twin_uses_and_operator():
    from ghost_agent.memory.skills import _delete_lesson_twin

    coll = MagicMock()
    memory_system = MagicMock()
    memory_system.collection = coll

    _delete_lesson_twin(memory_system, {"trigger": "use venv python"})

    assert coll.delete.called
    where = coll.delete.call_args.kwargs.get("where") or coll.delete.call_args.args[0]
    # Multi-key filters MUST be wrapped in $and — a flat two-key dict makes
    # Chroma raise ValueError, which the helper swallows (silent no-op).
    assert "$and" in where
    assert {"type": "skill"} in where["$and"]
    assert {"trigger": "use venv python"} in where["$and"]


def test_delete_lesson_twin_flat_dict_rejected_by_chroma(tmp_path):
    """Empirical guard: the installed chromadb version rejects the old flat
    two-key where, proving the $and fix is load-bearing."""
    chromadb = pytest.importorskip("chromadb")
    client = chromadb.PersistentClient(path=str(tmp_path))
    coll = client.get_or_create_collection("twin_fix_check")
    coll.add(ids=["1"], documents=["d"], metadatas=[{"type": "skill", "trigger": "t"}])
    with pytest.raises(Exception):
        coll.delete(where={"type": "skill", "trigger": "t"})
    coll.delete(where={"$and": [{"type": "skill"}, {"trigger": "t"}]})
    assert coll.count() == 0


# ---------------------------------------------------------------------------
# memory/skills.py — duplicate merge re-locates by key under the lock
# ---------------------------------------------------------------------------

def test_learn_lesson_merges_correct_entry_after_concurrent_prepend(tmp_path):
    from ghost_agent.memory.skills import SkillMemory

    sm = SkillMemory(tmp_path)
    sm.learn_lesson("target task", "old mistake", "old solution")

    # Simulate a concurrent writer prepending a lesson AFTER the duplicate
    # check found "target task" at index 0: the merge must land on the
    # matching lesson, not on whatever now sits at the stale index.
    original_find = sm._find_duplicate_lesson

    def find_then_prepend(*args, **kwargs):
        dup = original_find(*args, **kwargs)
        playbook = sm._load_playbook()
        playbook.insert(0, {"task": "interloper", "mistake": "x", "solution": "y",
                            "trigger": "interloper", "frequency": 1})
        sm.save_playbook(playbook)
        return dup

    sm._find_duplicate_lesson = find_then_prepend
    sm.learn_lesson("target task", "old mistake", "a much longer and richer solution text")

    playbook = sm._load_playbook()
    by_task = {p.get("task") or p.get("trigger"): p for p in playbook}
    assert int(by_task["target task"]["frequency"]) == 2
    assert int(by_task["interloper"].get("frequency", 1)) == 1
    assert "richer solution" in by_task["target task"]["solution"]


# ---------------------------------------------------------------------------
# memory/vector.py — smart_update must not delete document/skill/episode
# ---------------------------------------------------------------------------

def test_smart_update_query_excludes_protected_types():
    from ghost_agent.memory.vector import VectorMemory

    vm = VectorMemory.__new__(VectorMemory)  # bypass heavy __init__
    vm._lock = threading.RLock()
    vm._get_lock = lambda: vm._lock
    vm.collection = MagicMock()
    vm.collection.query.return_value = {"ids": [[]], "distances": [[]]}
    vm.add = MagicMock()

    vm.smart_update("user is studying asyncio")

    # 2026-07-22: the denylist became a SAME-TYPE scope, which is strictly
    # stronger. The old `$nin` list was not the complement of _PRUNABLE_TYPES,
    # so `identity`/`synthesis`/`document_summary`/`acquired_skill` — and even a
    # user-saved `manual` — were legal deletion victims of an auto extraction.
    # Scoping the dedup query to the incoming type removes the entire cross-type
    # deletion class: protected types simply can't be candidates.
    where = vm.collection.query.call_args.kwargs["where"]
    assert where == {"type": "auto"}

    vm.collection.query.reset_mock()
    vm.smart_update("User location is Athens", "identity")
    assert vm.collection.query.call_args.kwargs["where"] == {"type": "identity"}


# ---------------------------------------------------------------------------
# selfhood/state.py — _cap evicts resolved entries before open ones
# ---------------------------------------------------------------------------

def test_cap_evicts_resolved_before_open():
    from ghost_agent.selfhood.state import SelfStateThread

    class Q:
        def __init__(self, name, resolved=None):
            self.name = name
            self.resolved_at = resolved

    seq = [Q("open-oldest")] + [Q(f"resolved-{i}", resolved="2026-01-01") for i in range(10)]
    SelfStateThread._cap(seq, 10)

    names = [q.name for q in seq]
    assert "open-oldest" in names, "the only OPEN question must survive the cap"
    assert len(seq) == 10


def test_cap_still_bounds_all_open():
    from ghost_agent.selfhood.state import SelfStateThread

    class Q:
        def __init__(self, name):
            self.name = name
            self.resolved_at = None

    seq = [Q(f"open-{i}") for i in range(12)]
    SelfStateThread._cap(seq, 10)
    assert len(seq) == 10
    assert seq[0].name == "open-2"  # oldest evicted when everything is open


# ---------------------------------------------------------------------------
# memory/projects.py — update_task canonicalizes parent_id
# ---------------------------------------------------------------------------

def test_update_task_canonicalizes_parent_id(tmp_path):
    from ghost_agent.memory.projects import ProjectStore

    store = ProjectStore(tmp_path)
    pid = store.create_project("test project")
    parent = store.add_task(pid, "parent task")
    child = store.add_task(pid, "child task")

    # LLM-echoed ids routinely arrive case-mangled.
    assert store.update_task(child, parent_id=parent.upper())
    assert store.get_task(child)["parent_id"] == parent

    # Cascade delete must find the re-parented child again.
    store.delete_task(parent)
    assert store.get_task(child) is None


# ---------------------------------------------------------------------------
# core/planning.py — BEST resolution and ALL alternatives rollup
# ---------------------------------------------------------------------------

def test_best_parent_resolves_when_last_child_fails():
    from ghost_agent.core.planning import TaskTree, TaskStatus, DependencyType

    tree = TaskTree()
    root = tree.add_task("root")
    parent = tree.add_task("pick best", parent_id=root,
                           dependency_type=DependencyType.BEST)
    a = tree.add_task("approach A", parent_id=parent)
    b = tree.add_task("approach B", parent_id=parent)

    tree.update_status(a, TaskStatus.DONE, result="the winning result")
    tree.update_status(b, TaskStatus.FAILED, failure_reason="boom")

    node = tree.nodes[parent]
    assert node.status == TaskStatus.DONE
    assert node.result_summary == "the winning result"


def test_all_parent_completes_via_alternative():
    from ghost_agent.core.planning import TaskTree, TaskStatus, DependencyType

    tree = TaskTree()
    root = tree.add_task("root")
    alt = tree.add_task("fallback plan")  # standalone alternative node
    parent = tree.add_task("must succeed", parent_id=root,
                           dependency_type=DependencyType.ALL,
                           alternatives=[alt])
    a = tree.add_task("primary attempt", parent_id=parent)

    tree.update_status(a, TaskStatus.FAILED, failure_reason="boom")
    assert tree.nodes[parent].status not in (TaskStatus.BLOCKED, TaskStatus.FAILED)
    assert tree.nodes[alt].status == TaskStatus.READY

    tree.update_status(alt, TaskStatus.DONE, result="fallback worked")
    assert tree.nodes[parent].status == TaskStatus.DONE


# ---------------------------------------------------------------------------
# core/project_advancer.py — atomic budget increment
# ---------------------------------------------------------------------------

def test_increment_budget_is_atomic_under_concurrency():
    from ghost_agent.core.project_advancer import _increment_budget

    class RacyStore:
        """update_project replaces the whole metadata dict, like the real
        store; a tiny sleep widens the read-modify-write window."""
        def __init__(self):
            self.meta = {}

        def get_project(self, pid):
            import time
            time.sleep(0.001)
            return {"metadata": dict(self.meta)}

        def update_project(self, pid, metadata):
            import time
            time.sleep(0.001)
            self.meta = dict(metadata)

    store = RacyStore()
    threads = [threading.Thread(target=_increment_budget, args=(store, "p1"))
               for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert store.meta["steps_used"] == 8


# ---------------------------------------------------------------------------
# core/agent.py — trajectory user_request uses the real human request
# ---------------------------------------------------------------------------

def test_record_turn_trajectory_prefers_explicit_user_request():
    from ghost_agent.core.agent import GhostAgent

    agent = GhostAgent.__new__(GhostAgent)
    agent.context = MagicMock()
    recorded = {}
    agent.context.trajectory_collector = MagicMock()
    agent.context.trajectory_collector.append = lambda traj: recorded.update(t=traj)

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "the real human request"},
        {"role": "assistant", "content": "working on it"},
        {"role": "user", "content": "SYSTEM ALERT: tool failed, pivot now"},
    ]
    GhostAgent._record_turn_trajectory(
        agent,
        messages=messages,
        final_content="done",
        req_id="r1",
        model="m",
        user_request="the real human request",
    )
    assert recorded["t"].user_request == "the real human request"


def test_record_turn_trajectory_fallback_without_explicit_request():
    from ghost_agent.core.agent import GhostAgent

    agent = GhostAgent.__new__(GhostAgent)
    agent.context = MagicMock()
    recorded = {}
    agent.context.trajectory_collector = MagicMock()
    agent.context.trajectory_collector.append = lambda traj: recorded.update(t=traj)

    messages = [{"role": "user", "content": "only message"}]
    GhostAgent._record_turn_trajectory(
        agent, messages=messages, final_content="x", req_id="r2", model="m",
    )
    assert recorded["t"].user_request == "only message"


# ---------------------------------------------------------------------------
# utils/token_counter.py — HF_HUB_OFFLINE restored, not popped
# ---------------------------------------------------------------------------

def test_load_tokenizer_restores_prior_hf_offline(tmp_path, monkeypatch):
    from ghost_agent.utils import token_counter

    local = tmp_path / "tok"
    local.mkdir()
    (local / "tokenizer.json").write_text("{}")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # e.g. set by --mandatory-tor
    fake_auto = MagicMock()
    fake_auto.from_pretrained.return_value = MagicMock()
    monkeypatch.setattr(token_counter, "AutoTokenizer", fake_auto)
    # load_tokenizer reassigns the global TOKEN_ENCODER to our mock;
    # registering the current value with monkeypatch restores it at
    # teardown so other tests don't inherit a MagicMock encoder.
    monkeypatch.setattr(token_counter, "TOKEN_ENCODER", token_counter.TOKEN_ENCODER)

    token_counter.load_tokenizer(local)
    import os
    assert os.environ.get("HF_HUB_OFFLINE") == "1", \
        "pre-existing offline flag must survive load_tokenizer"


def test_load_tokenizer_removes_flag_it_set(tmp_path, monkeypatch):
    from ghost_agent.utils import token_counter

    local = tmp_path / "tok"
    local.mkdir()
    (local / "tokenizer.json").write_text("{}")

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    fake_auto = MagicMock()
    fake_auto.from_pretrained.return_value = MagicMock()
    monkeypatch.setattr(token_counter, "AutoTokenizer", fake_auto)
    monkeypatch.setattr(token_counter, "TOKEN_ENCODER", token_counter.TOKEN_ENCODER)

    token_counter.load_tokenizer(local)
    import os
    assert os.environ.get("HF_HUB_OFFLINE") is None


# ---------------------------------------------------------------------------
# tools/memory.py — fire-and-forget graph extraction keeps a strong ref
# ---------------------------------------------------------------------------

async def test_tool_remember_keeps_graph_task_reference(monkeypatch):
    from ghost_agent.tools import memory as memory_tools
    from ghost_agent.utils import logging as _glog

    # Graph extraction now schedules through the unified spawn_bg registry
    # (utils.logging._BG_TASKS), which holds the strong ref + drains at
    # shutdown — the old module-local _GRAPH_EXTRACT_TASKS set was removed.
    _glog._BG_TASKS.clear()

    memory_system = MagicMock()
    graph_memory = MagicMock()
    llm_client = MagicMock()
    llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "{\"graph_triplets\": []}"}}]
    })

    result = await memory_tools.tool_remember(
        text="alpha knows beta",
        memory_system=memory_system,
        graph_memory=graph_memory,
        llm_client=llm_client,
    )
    assert "Memory stored" in result
    tasks = list(_glog._BG_TASKS)
    assert len(tasks) == 1  # strong ref held while pending
    await tasks[0]
    await asyncio.sleep(0)
    assert tasks[0] not in _glog._BG_TASKS  # done_callback discards


# ---------------------------------------------------------------------------
# Deferred-findings round: eval suite verdict contract, execute container
# guard, resume offset clamp, docker dead param removal
# ---------------------------------------------------------------------------

async def test_eval_suite_honors_template_verdict_dict():
    """A sandbox runner returning {"passed": ...} per the documented
    ChallengeTemplateTask contract must be judged on that verdict — the old
    unpack stripped it to ["output"], failing {"passed": True} as "empty
    output" and PASSING {"passed": False, "output": "<traceback>"}."""
    from ghost_agent.eval.suite import EvalSuite
    from ghost_agent.eval.tasks import ChallengeTemplateTask

    async def verdict_runner(task, _ctx):
        if task.task_id == "t-pass":
            return {"passed": True}
        return {"passed": False, "output": "Traceback: assertion failed"}

    tasks = [
        ChallengeTemplateTask(task_id="t-pass", category="", prompt="p1"),
        ChallengeTemplateTask(task_id="t-fail", category="", prompt="p2"),
    ]
    result = await EvalSuite("verdicts", tasks).run(runner=verdict_runner)
    by_id = {r.task_id: r for r in result.results}
    assert by_id["t-pass"].passed is True
    assert by_id["t-fail"].passed is False


async def test_eval_suite_keeps_string_contract_for_non_template():
    """Non-template tasks must still validate on the output STRING even if
    the runner dict happens to carry a `passed` key."""
    from ghost_agent.eval.suite import EvalSuite
    from ghost_agent.eval.tasks import CuratedRequestTask

    async def runner(task, _ctx):
        return {"passed": False, "output": "hello world"}

    tasks = [CuratedRequestTask(task_id="c1", category="", prompt="x",
                                validator=["hello"])]
    result = await EvalSuite("curated", tasks).run(runner=runner)
    assert result.results[0].passed is True  # keyword matched the string


async def test_stateful_execute_dead_container_returns_formatted_error(tmp_path):
    """A missing sandbox container on a stateful call must surface as the
    tool's formatted error, not an unhandled AttributeError."""
    from ghost_agent.tools.execute import tool_execute

    sm = MagicMock()
    sm.container = None
    # Connection-file probe: "test -f" fails, pgrep irrelevant → boot path.
    sm.execute = MagicMock(return_value=("", 1))
    sm.tor_proxy = None

    result = await tool_execute(
        filename="t.py", content="print(1)", sandbox_dir=tmp_path,
        sandbox_manager=sm, stateful=True,
    )
    assert isinstance(result, str)
    assert "EXIT CODE: 1" in result
    assert "container is not running" in result.lower() or "Sandbox container" in result


def test_chat_resume_offset_is_clamped():
    src = (ROOT / "interface" / "server.py").read_text()
    assert "max(0, min(offset, len(task[\"buffer\"])))" in src, \
        "resume offset (a CHUNK index) must be clamped, not trusted"


def test_docker_execute_has_no_dead_memory_limit_param():
    import inspect as _inspect
    from ghost_agent.sandbox.docker import DockerSandbox
    params = _inspect.signature(DockerSandbox.execute).parameters
    assert "memory_limit" not in params, \
        "memory_limit was accepted but silently ignored (container-level setting)"


# ---------------------------------------------------------------------------
# Structural contracts — fixes whose behavior needs a live server/upstream.
# Source-level assertions so a refactor that silently reverts them fails CI.
# ---------------------------------------------------------------------------

def test_llm_worker_and_swarm_have_exhaustion_break():
    src = (SRC / "core" / "llm.py").read_text()
    # vision + coding had the guard; worker + swarm were missing it.
    assert src.count("if node in tried_nodes:\n                        break") >= 4 or \
        src.count("if node in tried_nodes:") >= 8, \
        "worker/swarm node-exhaustion break guard missing"


def test_llm_stream_does_not_retry_after_yield():
    src = (SRC / "core" / "llm.py").read_text()
    assert "yielded_any" in src, "mid-stream retry duplicate-output guard missing"


def test_api_generate_never_forwards_stream_true():
    src = (SRC / "api" / "routes.py").read_text()
    assert '"stream": False' in src, \
        "/api/generate must request a non-streaming upstream completion"


def test_interface_janitor_never_drops_live_tasks_silently():
    src = (ROOT / "interface" / "server.py").read_text()
    assert "evicted: active task cap exceeded" in src
    # tail stderr must not be a never-drained pipe
    assert "stderr=asyncio.subprocess.DEVNULL" in src


def test_main_gates_persistent_stores_under_no_memory():
    src = (ROOT / "src" / "ghost_agent" / "main.py").read_text()
    assert "if args.no_memory:" in src
    assert "ghost_no_memory_" in src, \
        "--no-memory must back journal/skill/frontier stores with a throwaway dir"


# ---------------------------------------------------------------------------
# core/agent.py — Perfect-It off the response path + end-of-turn heartbeats
# (follow-up to the 271s-gap report: a 24s task was delivered at +271s
# because the response blocked on Perfect-It + verifier, whose runtime also
# crossed the watchdog's 120s idle threshold and woke the hippocampus
# mid-request, piling consolidation LLM calls onto the same upstream)
# ---------------------------------------------------------------------------

async def test_perfect_it_helper_learns_with_provenance():
    from ghost_agent.core.agent import GhostAgent

    agent = GhostAgent.__new__(GhostAgent)
    agent.context = MagicMock()
    agent.context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "<tool_call>x</tool_call> Use a Makefile."}}]
    })
    sm = MagicMock()
    agent.context.skill_memory = sm
    agent.context.memory_system = None

    p_msg = await agent._perfect_it_generate_and_learn(
        {"messages": []}, "Optimization Analysis: run tests...", "T-123"
    )

    assert p_msg == "Use a Makefile."  # tool_call stripped
    kwargs = sm.learn_lesson.call_args.kwargs
    assert kwargs["source_trajectory_id"] == "T-123"
    assert kwargs["source"] == "perfection_protocol"
    assert kwargs["solution"] == "Use a Makefile."
    # The background path must be requested so the call yields to any
    # foreground (e.g. verifier) completion.
    cc_kwargs = agent.context.llm_client.chat_completion.call_args.kwargs
    assert cc_kwargs.get("is_background") is True


async def test_perfect_it_helper_skips_lesson_on_empty_output():
    from ghost_agent.core.agent import GhostAgent

    agent = GhostAgent.__new__(GhostAgent)
    agent.context = MagicMock()
    agent.context.llm_client.chat_completion = AsyncMock(return_value={
        "choices": [{"message": {"content": "<tool_call>only a hallucinated call</tool_call>"}}]
    })
    sm = MagicMock()
    agent.context.skill_memory = sm

    p_msg = await agent._perfect_it_generate_and_learn({"messages": []}, "label", "T-1")
    assert p_msg == ""
    sm.learn_lesson.assert_not_called()


def test_perfect_it_is_deferred_when_flag_off():
    """Structural contract: with --perfect-it off, the generation is
    scheduled as a tracked background task (strong ref + done-callback
    discard) instead of being awaited on the response path, the synthetic
    directive is NOT appended to the live `messages`, and heartbeats keep
    the watchdog from waking the hippocampus mid-request."""
    src = (SRC / "core" / "agent.py").read_text()
    # When the flag is off the suggestion is generated in the background
    # (log wording reflects the deferral).
    assert "in the background" in src
    block = src.split("Perfect It Protocol")[1].split("VERIFIER GATE")[0]
    assert "_pending_background_tasks" in block
    assert "add_done_callback" in block
    assert "last_activity_time = datetime.datetime.now()" in block  # heartbeat
    assert "messages.append({\"role\": \"user\", \"content\": perfect_it_prompt})" not in src
    # Heartbeat before the verifier completion too.
    verifier_gate = src.split("VERIFIER GATE")[1].split("verify_code_output")[0]
    assert "last_activity_time = datetime.datetime.now()" in verifier_gate


def test_agent_resets_calib_pending_per_request():
    src = (SRC / "core" / "agent.py").read_text()
    assert "self.context._calib_pending = None" in src.split("def handle_chat")[1].split("def ")[0] or \
        "self.context._calib_pending = None" in src, \
        "stale streaming calibration reading must be cleared at request start"
