"""Tests for Phase 7 safety rails: budgets, human-gates, contradictions,
and the suggestion-only promotion heuristic."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from types import SimpleNamespace

import pytest

from ghost_agent.memory.projects import ProjectStore
from ghost_agent.core.project_advancer import advance_once
from ghost_agent.core.project_safety import (
    BudgetDecision, PromotionSuggestion,
    check_budget, record_runtime,
    enforce_human_gate, HUMAN_GATE_PREFIX,
    detect_contradiction, route_contradiction,
    should_suggest_promotion,
    MIN_TURNS_FOR_SUGGESTION, MIN_PLAN_NODES_FOR_SUGGESTION,
)


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


@pytest.fixture
def context(store):
    return SimpleNamespace(
        project_store=store,
        scratchpad=None,
        graph_memory=None,
        contradiction_log=None,
        current_project_id=None,
    )


# --------------------------------------------------------------------- budgets

def test_check_budget_unbounded_by_default():
    d = check_budget({})
    assert d.allowed is True


def test_check_budget_blocks_on_exhausted_steps():
    d = check_budget({"steps_cap": 2, "steps_used": 2})
    assert d.allowed is False
    assert "steps" in d.reason


def test_check_budget_blocks_on_exhausted_runtime():
    d = check_budget({"runtime_cap_seconds": 60, "runtime_used_seconds": 61})
    assert d.allowed is False
    assert "runtime" in d.reason


def test_check_budget_blocks_on_exhausted_tool_calls():
    d = check_budget({"tool_call_cap": 10, "tool_call_used": 10})
    assert d.allowed is False
    assert "tool_calls" in d.reason


def test_check_budget_ok_when_partial():
    d = check_budget({"steps_cap": 5, "steps_used": 3,
                      "runtime_cap_seconds": 60, "runtime_used_seconds": 10})
    assert d.allowed is True
    assert "steps" in d.remaining


def test_record_runtime_accumulates(store):
    pid = store.create_project("P")
    record_runtime(store, pid, seconds=1.5, tool_calls=2)
    record_runtime(store, pid, seconds=2.0, tool_calls=1)
    meta = store.get_project(pid)["metadata"]
    assert abs(meta["runtime_used_seconds"] - 3.5) < 1e-6
    assert meta["tool_call_used"] == 3


def test_record_runtime_ignores_negative(store):
    pid = store.create_project("P")
    record_runtime(store, pid, seconds=-1, tool_calls=-1)
    meta = store.get_project(pid)["metadata"]
    assert "runtime_used_seconds" not in meta


def test_record_runtime_silent_on_missing_project(store):
    # Should not raise
    record_runtime(store, "no-such-id", seconds=1)


async def test_runtime_cap_blocks_advance(context, store):
    pid = store.create_project("P", metadata={
        "runtime_cap_seconds": 0, "runtime_used_seconds": 0.1,
    })
    store.add_task(pid, "Research something")
    r = await advance_once(context, pid)
    assert r.classification == "blocked"
    assert "runtime" in r.summary


async def test_advance_records_runtime_and_tool_calls(context, store):
    pid = store.create_project("P")
    store.add_task(pid, "Research foo")

    async def runner(name, args):
        return "ok"

    await advance_once(context, pid, tool_runner=runner)
    meta = store.get_project(pid)["metadata"]
    assert meta["tool_call_used"] == 1
    assert meta["runtime_used_seconds"] >= 0.0


# --------------------------------------------------------------------- human gates

def test_enforce_human_gate_returns_reason():
    task = {"postconditions": [f"{HUMAN_GATE_PREFIX} needs CTO sign-off"]}
    assert enforce_human_gate(task) == "needs CTO sign-off"


def test_enforce_human_gate_returns_default_when_empty():
    task = {"postconditions": [HUMAN_GATE_PREFIX]}
    out = enforce_human_gate(task)
    assert out and "human approval" in out.lower()


def test_enforce_human_gate_none_when_absent():
    assert enforce_human_gate({"postconditions": []}) is None
    assert enforce_human_gate({}) is None


async def test_human_gate_forces_needs_user(context, store):
    pid = store.create_project("P")
    tid = store.add_task(pid, "Deploy the new schema",
                         postconditions=[f"{HUMAN_GATE_PREFIX} cto approval"])

    async def runner(name, args):
        return "ready to ship"

    r = await advance_once(context, pid, tool_runner=runner)
    assert r.classification == "needs_user"
    assert store.get_task(tid)["status"] == "NEEDS_USER"
    evs = store.list_events(pid, event_type="human_gate_triggered")
    assert evs


# --------------------------------------------------------------------- contradictions

def test_detect_contradiction_catches_flip():
    assert detect_contradiction("fix is confirmed working",
                                "fix was denied") is not None


def test_detect_contradiction_tolerates_no_flip():
    assert detect_contradiction("hello world", "hello earth") is None


def test_detect_contradiction_handles_empty():
    assert detect_contradiction("", "anything") is None
    assert detect_contradiction("anything", "") is None


def test_detect_contradiction_agreeing_negatives_not_flagged():
    # Regression: 'supported' matched inside its own negation 'not
    # supported', so two AGREEING summaries were flagged as contradictory.
    assert detect_contradiction("streaming is not supported on this backend",
                                "we found streaming is not supported") is None
    assert detect_contradiction("that cast is unsafe",
                                "the cast is unsafe here too") is None
    assert detect_contradiction("driver is incompatible with kernel 6.8",
                                "confirmed the driver is incompatible") is None


def test_detect_contradiction_agreeing_positives_not_flagged():
    assert detect_contradiction("the migration is safe",
                                "the migration looks safe") is None
    assert detect_contradiction("SSE is supported by the proxy",
                                "SSE is supported end to end") is None


def test_detect_contradiction_genuine_negation_flip_flagged():
    # A real flip on the substring-containing pairs must still fire,
    # in both directions.
    assert detect_contradiction("the api is supported on linux",
                                "the api is not supported on linux") is not None
    assert detect_contradiction("the api is not supported on linux",
                                "the api is supported on linux") is not None
    assert detect_contradiction("the migration is safe",
                                "the migration is unsafe") is not None
    assert detect_contradiction("driver is incompatible with 6.8",
                                "driver is compatible with 6.8") is not None


def test_detect_contradiction_markers_are_word_bounded():
    # 'eyes'/'nothing' must not read as the yes/no markers.
    assert detect_contradiction("closed my eyes during the demo",
                                "nothing notable happened") is None


def test_route_contradiction_without_log_returns_false():
    assert route_contradiction(None, "new", ["old"]) is False


def test_route_contradiction_writes_to_log():
    calls = []

    class FakeLog:
        def record(self, new_fact, old_facts, deleted_ids, reason):
            calls.append((new_fact, old_facts, reason))

    ok = route_contradiction(FakeLog(), "new fact", ["old fact"],
                             reason="flip")
    assert ok is True
    assert calls[0] == ("new fact", ["old fact"], "flip")


def test_route_contradiction_survives_exception():
    class BoomLog:
        def record(self, **kw):
            raise RuntimeError("db dead")

    assert route_contradiction(BoomLog(), "x", ["y"]) is False


async def test_autoadvance_records_contradiction_event(context, store):
    pid = store.create_project("P")
    # Root task with two children — one DONE with "yes", one PENDING
    root = store.add_task(pid, "root")
    done_child = store.add_task(pid, "check A", parent_id=root)
    store.update_task(done_child, status="DONE",
                      result_summary="the build is confirmed green")
    store.add_task(pid, "check B", parent_id=root)

    async def runner(name, args):
        return "the build is denied — failing CI"

    calls = []

    class FakeLog:
        def record(self, new_fact, old_facts, deleted_ids, reason):
            calls.append(reason)

    context.contradiction_log = FakeLog()
    r = await advance_once(context, pid, tool_runner=runner)
    assert r.ok is True
    ev = store.list_events(pid, event_type="contradiction_detected")
    assert ev
    assert calls  # contradiction was routed to the log


# --------------------------------------------------------------------- suggestion heuristic

def test_suggestion_blocked_when_already_in_project():
    s = should_suggest_promotion(
        user_turns=["a"] * 20, assistant_turns=["b"] * 20,
        sandbox_writes=5, plan_node_count=10,
        already_in_project=True,
    )
    assert s.should_suggest is False
    assert "already in project" in s.reason


def test_suggestion_blocked_when_managing_projects():
    """No promotion nudge while the user is administering projects (list /
    delete / switch). Regression: the nudge fired right after a
    `delete project`, even quoting a stale first-turn as the title."""
    s = should_suggest_promotion(
        user_turns=["show me all the projects", "delete project a64a41803358"]
        + ["x"] * MIN_TURNS_FOR_SUGGESTION,
        assistant_turns=["b"] * MIN_TURNS_FOR_SUGGESTION,
        sandbox_writes=3, plan_node_count=5,  # would otherwise strongly trigger
        already_in_project=False,
        managing_projects=True,
    )
    assert s.should_suggest is False
    assert "managing projects" in s.reason


def test_suggestion_blocked_below_thresholds():
    s = should_suggest_promotion(
        user_turns=["a"], assistant_turns=["b"],
        sandbox_writes=0, plan_node_count=0,
        already_in_project=False,
    )
    assert s.should_suggest is False


def test_turns_alone_no_longer_trigger():
    # Depth WITHOUT durable output = just a long conversation, not a
    # project (the "12 turns titled hello" false positive).
    s = should_suggest_promotion(
        user_turns=["a"] * MIN_TURNS_FOR_SUGGESTION,
        assistant_turns=["b"] * MIN_TURNS_FOR_SUGGESTION,
        sandbox_writes=0, plan_node_count=0,
        already_in_project=False,
    )
    assert s.should_suggest is False


def test_suggestion_needs_turns_AND_writes():
    # Enough turns AND enough writes = a real promotable effort.
    from ghost_agent.core.project_safety import MIN_WRITES_FOR_SUGGESTION
    s = should_suggest_promotion(
        user_turns=["Build a log parser"] * MIN_TURNS_FOR_SUGGESTION,
        assistant_turns=["b"] * MIN_TURNS_FOR_SUGGESTION,
        sandbox_writes=MIN_WRITES_FOR_SUGGESTION, plan_node_count=0,
        already_in_project=False,
    )
    assert s.should_suggest is True
    assert "turns" in s.reason and "sandbox" in s.reason


def test_enough_turns_but_too_few_writes_does_not_trigger():
    from ghost_agent.core.project_safety import MIN_WRITES_FOR_SUGGESTION
    s = should_suggest_promotion(
        user_turns=["Build a thing"] * MIN_TURNS_FOR_SUGGESTION,
        assistant_turns=["b"] * MIN_TURNS_FOR_SUGGESTION,
        sandbox_writes=MIN_WRITES_FOR_SUGGESTION - 1, plan_node_count=0,
        already_in_project=False,
    )
    assert s.should_suggest is False


def test_writes_alone_without_turns_does_not_trigger():
    # A couple of writes in a 1-turn chat isn't a project either — both
    # signals are now required (unless the planner built a real plan).
    s = should_suggest_promotion(
        user_turns=["Research BGE embeddings"],
        assistant_turns=["a"],
        sandbox_writes=2, plan_node_count=0,
        already_in_project=False,
    )
    assert s.should_suggest is False


def test_suggestion_triggered_on_plan_depth():
    s = should_suggest_promotion(
        user_turns=["Do a thing"], assistant_turns=["ok"],
        sandbox_writes=0, plan_node_count=MIN_PLAN_NODES_FOR_SUGGESTION,
        already_in_project=False,
    )
    assert s.should_suggest is True


def test_suggestion_title_derived_from_first_user_turn():
    s = should_suggest_promotion(
        user_turns=(["Build a log parser for Nginx"]
                    + ["more"] * MIN_TURNS_FOR_SUGGESTION),
        assistant_turns=["ok"] * 10,
        sandbox_writes=3, plan_node_count=0,
        already_in_project=False,
    )
    assert s.suggested_title.startswith("Build a log parser")


def test_suggestion_title_truncated():
    s = should_suggest_promotion(
        user_turns=["x" * 200], assistant_turns=["a"],
        sandbox_writes=0, plan_node_count=MIN_PLAN_NODES_FOR_SUGGESTION,
        already_in_project=False,
    )
    assert len(s.suggested_title) <= 80
    assert s.suggested_title.endswith("…")


def test_greeting_first_turn_is_skipped_for_title():
    # The "hello" bug: a greeting first turn must not become the title.
    s = should_suggest_promotion(
        user_turns=(["hello!", "hey ghost"]
                    + ["Build a CSV analyzer"]
                    + ["tweak it"] * MIN_TURNS_FOR_SUGGESTION),
        assistant_turns=["ok"] * 12,
        sandbox_writes=3, plan_node_count=0,
        already_in_project=False,
    )
    assert s.should_suggest is True
    assert s.suggested_title.startswith("Build a CSV analyzer")
    assert "hello" not in s.suggested_title.lower()


def test_all_greetings_falls_back_to_first_nonempty():
    # If every early turn is a greeting, don't return empty — use the
    # first non-empty turn rather than "Untitled effort".
    s = should_suggest_promotion(
        user_turns=(["hi", "thanks"] + ["ok"] * MIN_TURNS_FOR_SUGGESTION),
        assistant_turns=["b"] * 12,
        sandbox_writes=3, plan_node_count=0,
        already_in_project=False,
    )
    assert s.suggested_title  # non-empty
    assert s.suggested_title != "Untitled effort"


def test_suggestion_never_auto_promotes():
    """The suggestion heuristic must never create a project — it only
    returns a recommendation. This is a design invariant: if a future
    refactor adds an 'auto_promote' side-effect, this test catches it.
    """
    import ast
    from ghost_agent.core import project_safety

    src = open(project_safety.__file__).read()
    tree = ast.parse(src)

    # Find all function call *names* in the module's executable code.
    # Docstring mentions and comments are not AST nodes, so they pass.
    forbidden_calls = {"create_project", "add_task", "delete_project",
                       "tool_manage_projects", "promote_from_context"}
    offenders: list = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name in forbidden_calls:
                offenders.append(name)
    assert offenders == [], f"project_safety must not mutate projects: {offenders}"
