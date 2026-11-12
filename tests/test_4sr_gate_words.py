"""§4SR — a gate is a human's, decided by whole words; a batch runs past it.

req 4609b359: "proceed with all tasks" ran 0 of 10 — task 1's description
said "…self-confirmation…", the substring "confirm" gated it, and the batch
stopped with nine independent tasks behind it. The owner path's LLM
classifier never answered (main model, thinking on, 8 tokens), and the
worker one over-gates ("Write final comprehensive report").
"""

import ast
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import pytest

from ghost_agent.core import project_advancer as PA
from ghost_agent.core.project_advancer import (
    AdvanceManyResult, AdvanceResult, advance_many, advance_once, classify_task,
    default_llm_classifier, settle_gate,
)
from ghost_agent.memory.projects import ProjectStore
from ghost_agent.memory.scratchpad import Scratchpad
from ghost_agent.tools.projects import _advance_batch_instruction, tool_manage_projects

SRC = Path(__file__).resolve().parents[1] / "src" / "ghost_agent"

# the real task of req 4609b359, verbatim
REQ_4609 = (
    "Deep-dive research — formal & philosophical foundations of recursive self-reference: "
    "Gödel's incompleteness & self-reference, Tarski's undefinability of truth, Hofstadter's "
    "strange loop (GEB 1979, I Am a Strange Loop 2007), higher-order theories of consciousness "
    "(Rosenthal, Carruthers 2009 'How we know our own minds'), and pin down what 'self-affirming' "
    "formally means (fixed-point stability, self-verification, self-confirmation). Write synthesis "
    "with source list and gap analysis to research/01-formal-foundations.md.")


# ------------------------------------------------------------- the gate words


NEGATIVES = [
    REQ_4609,
    "Score each reply for self-affirmation markers (stable self-model, confirmation phrasing)",
    "Verify PostgreSQL 18.4 as latest stable without explicit confirmation from the tools",
    "Survey transcendental arguments for self-knowledge",     # s-e-n-d inside a word
    "Collect unpublished preprints on metacognition",
    "Summarise confirmed cases per region",
    "Review papers published since 2020",
    "Explain the self-confirm loop in the model",
    "Model the decider stage of the pipeline",
    "Map approvers and senders in the email graph",
    "Read about the pre-approved list",
    "Plan the send-off party for the team",
    "Build an HTTP client that sends requests",               # a form, no person
    "Owner record: the decider stage",                         # a person, no form
    # r2 #5: common nouns are not roles
    "Implement a context manager that confirms the lock is released",
    "Add a connection manager that sends heartbeats",
    "Verify the package manager sends telemetry",
    "Write tests confirming the operator is used by the planner",
    "Survey human evaluation studies published since 2020",
    # r2 #6: identifiers
    "Write publish.py to export the notes",
    "Call confirm() on exit",
    # r3 m1: dotted / path / backticked identifiers
    "Mock socket.send in the tests",
    "Call client.publish with the payload",
    "Add the /api/publish endpoint",
    "Implement `confirm` handler",
    # r3 m4: a role named, not bound to the gate word
    "Validate the user's input before sending it",
    "Parse the customer's order and log a confirmation id",
    "Report which roles are schema owners, as published in pg_catalog",
    "List the table owners and when ownership was decided",
    "Generate legal moves and confirmed by perft",
    "Collect unpublished preprints by the user group",
    "The user sends a message to the bot",
    "Legal must have approved the contract first",        # no deciding role: the classifier's call
    "",
]


# r4: another form of a gate word is the classifier's call — never gated on
# the words alone, gated when the classifier says needs_user, not otherwise
OTHER_FORMS = [
    "Get the owner's confirmation before deploying",
    "Owner decides the hosting provider",
    "Get the budget pre-approved by the CTO",
    "Await the team lead's decision on the database",
    "Get the user's confirmation of the scope before building",
    "Owners' approval needed before the launch",
    "Get the user\u2019s confirmation of scope",
    "The CTO has approved the spend",
    "Approval from the client is required before launch",
    "Wait for my approval",
    "Store the user's confirmation token",        # r4 #3: the classifier says coding
    "Query rows approved by user 5",
    "Per the owner's decision, use BRIN",
    "Wait until the client has sent the signed contract",    # r5 #2: missing forms
    "Collect approvals from the board before launch",
    "Owner picks one of the logos",
    "Get confirmations from both vendors",
    "Await the owner\u2019s choice of hosting",               # r6 #3
    "Publish/deploy the site",                                # r6 #1: a slash next to a gate
    "Approve/deploy the release",                             # word is the classifier's call
    "Wait for owner approval/sign-off",
    "Owner decision/approval needed before deploy",
    "Add rate limiting to the api/send endpoint",             # (so is a path: the classifier
    "Copy the build into publish/ for deploy",                #  says coding, it stays coding)
    "Confirm(y/n) before deploying",                          # r10: an identifier shape is the
    "Approve = yes",                                          # classifier's call too
    "set confirm=False in config",
    "Mock socket.send in the tests",
    "Ship the approval-gated release",                        # r11: a hyphen on the right
]


# ------------------------------------------------------------- settle_gate


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb")


def _ctx(store):
    return SimpleNamespace(project_store=store, llm_client=None,
                           args=SimpleNamespace(model="m"))


async def _search(name, args):
    return "stub results: " + "x" * 400


# ------------------------------------------------------------- the one classifier

class _LLM:
    def __init__(self, content=None, exc=None):
        self.content, self.exc, self.calls = content, exc, []

    async def chat_completion(self, payload, **kw):
        self.calls.append((payload, kw))
        if self.exc:
            raise self.exc
        return {"choices": [{"message": {"content": self.content}}]}


def _clf(llm, **kw):
    return default_llm_classifier(SimpleNamespace(llm_client=llm, args=SimpleNamespace(model="m")), **kw)


@pytest.mark.asyncio
async def test_classifier_runs_on_the_worker_with_thinking_off():
    llm = _LLM("coding")
    assert await _clf(llm)("Build app.py") == "coding"
    payload, kw = llm.calls[0]
    assert kw.get("use_worker") is True and kw.get("off_main_only") is True
    assert not kw.get("side_main")
    assert payload["chat_template_kwargs"]["enable_thinking"] is False
    assert kw.get("is_background") is False
    assert kw.get("require_healthy") is True and kw.get("timeout") <= 10
    await _clf(llm, background=True)("x")
    assert llm.calls[1][1].get("is_background") is True


@pytest.mark.asyncio
@pytest.mark.parametrize("content,desc,expected", [
    ("coding or research", "Summarize the literature", "research"),   # two labels: no answer
    ("research or coding", "Build app.py with an API", "coding"),
    ("", "Summarize the literature", "research"),
    (None, "Summarize the literature", "research"),
    ("I think", "Summarize the literature", "research"),
    ("", "Approve the plan", "research"),              # no label → the keyword bucket (§4UK: no word gate)
    ("", "Build app.py with an API", "coding"),
    ("needs_user", "Summarize the literature", "needs_user"),
    ("Research.", "Build app.py", "research"),
])
async def test_classifier_label_or_keyword_fallback(content, desc, expected):
    assert await _clf(_LLM(content))(desc) == expected


@pytest.mark.asyncio
async def test_classifier_failure_calls_the_hook_and_falls_back():
    seen = []
    clf = _clf(_LLM(exc=RuntimeError("no worker")), on_error=seen.append)
    assert await clf("Approve the plan") == "research"      # §4UK: the keyword bucket has no gate
    assert len(seen) == 1 and isinstance(seen[0], RuntimeError)


@pytest.mark.asyncio
async def test_a_raising_hook_does_not_escape():
    def boom(_e):
        raise ValueError("hook")
    clf = _clf(_LLM(exc=RuntimeError("x")), on_error=boom)
    assert await clf("Summarise the literature") == "research"


def test_no_llm_no_classifier():
    assert default_llm_classifier(SimpleNamespace(llm_client=None)) is None


def test_one_classifier_prompt_in_the_source():
    """R1 enumeration: the bucket prompt exists ONCE (CLASSIFY_PROMPT). A
    second copy is a second classifier that can drift (the idle loop's did)."""
    hits = []
    for p in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and "Classify this task into EXACTLY one word" in node.value:
                hits.append(str(p.relative_to(SRC)))
    assert hits == ["core/project_advancer.py"], hits


# ------------------------------------------------------------- the batch

class _FakeStore:
    def __init__(self, status, ledger=None, waits=None):
        self.status, self.ledger, self.waits = status, ledger, waits or {}

    def get_task(self, tid):
        # §4SU Phase 3: a hold is the task's wait_on field
        return {"status": self.status.get(tid), "wait_on": self.waits.get(tid, "")}

    def get_project(self, pid):
        return {"status": "ACTIVE"}

    def list_tasks(self, pid):
        return [{"id": k, "status": v} for k, v in (self.ledger or self.status).items()]


def _script(monkeypatch, results):
    seq = list(results)

    async def fake(context, project_id, **kw):
        return seq.pop(0)
    monkeypatch.setattr(PA, "advance_once", fake)
    monkeypatch.setattr(PA, "advance_unattended", fake)
    monkeypatch.setattr(PA, "_paused_since", lambda *a, **k: False)
    return seq


@pytest.mark.asyncio
async def test_batch_goes_past_a_gate(monkeypatch):
    ctx = SimpleNamespace(project_store=_FakeStore(
        {"t1": "NEEDS_USER", "t2": "DONE", "t3": "DONE"}))
    _script(monkeypatch, [
        AdvanceResult(True, "t1", "needs_user", "task requires human input"),
        AdvanceResult(True, "t2", "research", "ok"),
        AdvanceResult(True, "t3", "research", "ok"),
        AdvanceResult(True, None, "idle", "no ready leaf"),
    ])
    r = await advance_many(ctx, "p", owner_requested=True, max_tasks=None, stop_on_fail=False)
    assert [a["task_id"] for a in r.advanced] == ["t1", "t2", "t3"]
    assert r.stop_reason == "needs_user"


@pytest.mark.asyncio
async def test_safe_mode_still_stops_at_a_gate(monkeypatch):
    ctx = SimpleNamespace(project_store=_FakeStore({"t1": "NEEDS_USER", "t2": "DONE"}))
    left = _script(monkeypatch, [
        AdvanceResult(True, "t1", "needs_user", "x"),
        AdvanceResult(True, "t2", "research", "should not run"),
    ])
    r = await advance_many(ctx, "p", owner_requested=True, max_tasks=None, stop_on_fail=True)
    assert r.stop_reason == "needs_user" and len(left) == 1
    assert r.gated == ["t1"] and r.ran == 0             # r4: booked as a gate


@pytest.mark.asyncio
async def test_a_task_that_ran_and_was_held_is_one_of_the_n(monkeypatch):
    """r3 MAJOR: an unattended close holds a task that RAN ("not closed
    unattended"). It is a hold, not a gate, and it is one of the N — count=1
    runs one task, not four."""
    ctx = SimpleNamespace(project_store=_FakeStore({"t1": "NEEDS_USER", "t2": "NEEDS_USER"}))
    left = _script(monkeypatch, [
        AdvanceResult(True, "t1", "research", f"task ran via web_search; held for the owner: x — {PA._HELD_MARK}"),
        AdvanceResult(True, "t2", "research", "should not run"),
    ])
    r = await advance_many(ctx, "p", owner_requested=False, max_tasks=1, stop_on_fail=False)
    assert r.stop_reason == "count_reached" and len(left) == 1
    assert r.ran == 1 and r.held == ["t1"] and r.gated == [] and r.not_run == []
    instr = _advance_batch_instruction(r, 1)
    assert instr.startswith("Autonomous batch ran 1 task(s)")
    assert "1 held for the owner's own run" in instr and "at a gate" not in instr


@pytest.mark.asyncio
async def test_a_hold_made_before_running_is_a_hold_not_a_gate(monkeypatch):
    # the unattended model-experiment hold: classification needs_user, never ran
    ctx = SimpleNamespace(project_store=_FakeStore({"t1": "NEEDS_USER", "t2": "DONE"},
                                                   waits={"t1": "owner_turn"}))
    _script(monkeypatch, [
        AdvanceResult(True, "t1", "needs_user", f"model experiment? — {PA._HELD_MARK}"),
        AdvanceResult(True, "t2", "research", "ok"),
        AdvanceResult(True, None, "idle", "x"),
    ])
    r = await advance_many(ctx, "p", owner_requested=False, max_tasks=1, stop_on_fail=False)
    assert r.held == ["t1"] and r.gated == [] and r.not_run == ["t1"]
    assert r.ran == 1 and r.count == 2 and r.stop_reason == "count_reached"


@pytest.mark.asyncio
async def test_a_task_gated_twice_stops_the_batch(monkeypatch):
    ctx = SimpleNamespace(project_store=_FakeStore({"t1": "NEEDS_USER"}))
    left = _script(monkeypatch, [AdvanceResult(True, "t1", "needs_user", "x")] * 5)
    r = await advance_many(ctx, "p", owner_requested=True, max_tasks=None, stop_on_fail=False)
    assert r.stop_reason == "needs_user"
    assert r.count == 2 and len(left) == 3
    assert r.ran == 0                                     # r3 m5: nothing ran


@pytest.mark.asyncio
async def test_a_count_stop_with_a_failure_keeps_count_reached(monkeypatch):
    """r6 #4: two of four ran (one FAILED), two still READY — more may run;
    "Advanced every task it could" would be false."""
    ctx = SimpleNamespace(project_store=_FakeStore(
        {"t1": "FAILED", "t2": "DONE", "t3": "READY", "t4": "READY"}))
    _script(monkeypatch, [
        AdvanceResult(True, "t1", "coding", "failed"),
        AdvanceResult(True, "t2", "coding", "ok"),
        AdvanceResult(True, "t3", "coding", "must not run"),
    ])
    r = await advance_many(ctx, "p", owner_requested=True, max_tasks=2, stop_on_fail=False)
    assert r.count == 2 and r.stop_reason == "count_reached"


@pytest.mark.asyncio
async def test_a_failure_outranks_the_gate(monkeypatch):
    ctx = SimpleNamespace(project_store=_FakeStore(
        {"t1": "NEEDS_USER", "t2": "FAILED", "t3": "DONE"}))
    _script(monkeypatch, [
        AdvanceResult(True, "t1", "needs_user", "x"),
        AdvanceResult(True, "t2", "coding", "failed"),
        AdvanceResult(True, "t3", "coding", "ok"),
        AdvanceResult(True, None, "idle", "x"),
    ])
    r = await advance_many(ctx, "p", owner_requested=True, max_tasks=None, stop_on_fail=False)
    assert r.stop_reason == "completed_with_failures"


@pytest.mark.asyncio
async def test_an_earlier_failure_in_the_ledger_outranks_the_gate(monkeypatch):
    ctx = SimpleNamespace(project_store=_FakeStore(
        {"t1": "NEEDS_USER", "t2": "DONE"},
        ledger={"t0": "FAILED", "t1": "NEEDS_USER", "t2": "DONE"}))
    _script(monkeypatch, [
        AdvanceResult(True, "t1", "needs_user", "x"),
        AdvanceResult(True, "t2", "research", "ok"),
        AdvanceResult(True, None, "idle", "x"),
    ])
    r = await advance_many(ctx, "p", owner_requested=True, max_tasks=None, stop_on_fail=False)
    assert r.stop_reason == "project_failed"


@pytest.mark.asyncio
async def test_a_gate_is_not_one_of_the_n_and_count_reached_stays(monkeypatch):
    """r1 M2: "do the next 2" past a gate runs 2 tasks, and the stop says more
    remain (count_reached), not "nothing else can run"."""
    ctx = SimpleNamespace(project_store=_FakeStore(
        {"t1": "NEEDS_USER", "t2": "DONE", "t3": "DONE"}))
    left = _script(monkeypatch, [
        AdvanceResult(True, "t1", "needs_user", "x"),
        AdvanceResult(True, "t2", "research", "ok"),
        AdvanceResult(True, "t3", "research", "ok"),
        AdvanceResult(True, "t4", "research", "should not run"),
    ])
    r = await advance_many(ctx, "p", owner_requested=True, max_tasks=2, stop_on_fail=False)
    assert [a["task_id"] for a in r.advanced] == ["t1", "t2", "t3"]
    assert r.stop_reason == "count_reached" and len(left) == 1
    # r2 #3: the model is told 2 ran, not 3
    assert r.ran == 2 and r.gated == ["t1"]
    instr = _advance_batch_instruction(r, 2)
    assert instr.startswith("Autonomous batch ran 2 task(s)")
    assert "1 more left waiting at a gate" in instr


@pytest.mark.asyncio
async def test_the_hard_cap_past_a_gate_stays_hard_cap(monkeypatch):
    ctx = SimpleNamespace(project_store=_FakeStore({"g": "NEEDS_USER", "t": "DONE"}))
    _script(monkeypatch, [AdvanceResult(True, "g", "needs_user", "x")]
            + [AdvanceResult(True, "t", "research", "ok")] * 10)
    r = await advance_many(ctx, "p", owner_requested=True, max_tasks=None,
                           stop_on_fail=False, hard_cap=3)
    assert r.stop_reason == "hard_cap" and r.count == 4


@pytest.mark.asyncio
async def test_a_cap_stop_with_a_failure_stays_hard_cap(monkeypatch):
    """r7 NIT: a FAILED task in the batch does not make a cap stop "every
    task it could"."""
    ctx = SimpleNamespace(project_store=_FakeStore({"f": "FAILED", "t": "DONE"}))
    _script(monkeypatch, [AdvanceResult(True, "f", "coding", "failed")]
            + [AdvanceResult(True, "t", "research", "ok")] * 10)
    r = await advance_many(ctx, "p", owner_requested=True, max_tasks=None,
                           stop_on_fail=False, hard_cap=3)
    assert r.stop_reason == "hard_cap" and r.count == 3


@pytest.mark.asyncio
async def test_a_count_above_the_cap_is_a_cap_stop(monkeypatch):
    """r7 #2: count=100 stops at the cap, not at "the requested number"."""
    ctx = SimpleNamespace(project_store=_FakeStore({"t": "DONE"}))
    _script(monkeypatch, [AdvanceResult(True, "t", "research", "ok")] * 10)
    r = await advance_many(ctx, "p", owner_requested=True, max_tasks=100,
                           stop_on_fail=False, hard_cap=3)
    assert r.stop_reason == "hard_cap" and r.count == 3
    r = await advance_many(ctx, "p", owner_requested=True, max_tasks=3,
                           stop_on_fail=False, hard_cap=3)
    assert r.stop_reason == "count_reached"


@pytest.mark.asyncio
async def test_the_backstop_on_the_tick_that_reached_the_count_is_count_reached(monkeypatch):
    """r8 NIT: gate, gate, run with max_tasks=1, hard_cap=2 — the range runs
    out on the tick that reached the count."""
    ctx = SimpleNamespace(project_store=_FakeStore({"g1": "NEEDS_USER", "g2": "NEEDS_USER", "t": "DONE"}))
    _script(monkeypatch, [AdvanceResult(True, "g1", "needs_user", "x"),
                          AdvanceResult(True, "g2", "needs_user", "x"),
                          AdvanceResult(True, "t", "research", "ok")])
    r = await advance_many(ctx, "p", owner_requested=True, max_tasks=1,
                           stop_on_fail=False, hard_cap=2)
    assert r.ran == 1 and r.stop_reason == "count_reached"


@pytest.mark.asyncio
async def test_a_model_experiment_is_not_counted_as_ran(monkeypatch):
    """r8 #2: payload `ran` and the text agree — the held experiment did not run."""
    ctx = SimpleNamespace(project_store=_FakeStore({"f": "FAILED", "e": "NEEDS_USER"}))
    _script(monkeypatch, [AdvanceResult(True, "f", "coding", "failed"),
                          AdvanceResult(True, "e", "blocked", PA.MODEL_EXPERIMENT_PREFIX + "x"),
                          AdvanceResult(True, None, "idle", "no READY/PENDING leaf")])
    r = await advance_many(ctx, "p", owner_requested=True, max_tasks=None, stop_on_fail=False)
    # §4SU Phase 5: the batch goes on past the experiment; the ledger's FAILED
    # task then outranks it as the stop, and the experiment is still listed
    assert r.ran == 1 and "e" in r.not_run and r.experiments == ["e"]
    assert r.stop_reason == "completed_with_failures"


@pytest.mark.asyncio
async def test_gates_past_the_backstop_with_a_failure_stay_hard_cap(monkeypatch):
    """r7 NIT: the backstop is a cap stop too."""
    ctx = SimpleNamespace(project_store=_FakeStore(
        {"f": "FAILED", **{f"g{i}": "NEEDS_USER" for i in range(20)}}))
    _script(monkeypatch, [AdvanceResult(True, "f", "coding", "failed")]
            + [AdvanceResult(True, f"g{i}", "needs_user", "x") for i in range(20)])
    r = await advance_many(ctx, "p", owner_requested=True, max_tasks=None,
                           stop_on_fail=False, hard_cap=3)
    assert r.stop_reason == "hard_cap"


@pytest.mark.asyncio
async def test_gates_alone_cannot_spin_past_the_backstop(monkeypatch):
    ctx = SimpleNamespace(project_store=_FakeStore({f"g{i}": "NEEDS_USER" for i in range(20)}))
    left = _script(monkeypatch, [AdvanceResult(True, f"g{i}", "needs_user", "x") for i in range(20)])
    r = await advance_many(ctx, "p", owner_requested=True, max_tasks=None,
                           stop_on_fail=False, hard_cap=3)
    assert r.count == 6 and r.stop_reason == "hard_cap" and len(left) == 14


# ------------------------------------------------------------- what the model reads

def test_gate_instruction_says_all_is_not_an_answer():
    instr = _advance_batch_instruction(
        AdvanceManyResult([{"status": "NEEDS_USER"}, {"status": "DONE"}], "needs_user", None), None)
    assert "NOT the answer to a gate" in instr
    assert "do that task yourself" in instr
    assert "does not run them" in instr
    assert "Paused" not in instr


def test_other_stops_say_rerunning_autoadvance_is_how_to_continue():
    instr = _advance_batch_instruction(AdvanceManyResult([{"status": "FAILED"}], "failed", None), None)
    assert "calling autoadvance again is how to continue" in instr
    assert "re-running autoadvance after a fix" not in instr


@pytest.fixture(autouse=True)
def _owner_turn():
    from ghost_agent.utils.logging import request_id_context
    tok = request_id_context.set("req-owner-4sr")
    yield
    request_id_context.reset(tok)


@pytest.fixture
def tctx(tmp_path, monkeypatch):
    import ghost_agent.tools.registry as reg

    async def _web_search(query="", **_kw):
        return f"stub results for {query}"
    monkeypatch.setattr(reg, "get_available_tools", lambda ctx: {"web_search": _web_search})
    return SimpleNamespace(
        project_store=ProjectStore(tmp_path / "mem", sandbox_root=tmp_path / "sb"),
        scratchpad=Scratchpad(persist_path=tmp_path / "sp.db"),
        graph_memory=None, contradiction_log=None, current_project_id=None, llm_client=None)


@pytest.mark.asyncio
async def test_a_gate_word_task_runs_with_the_rest(tctx):
    """End to end on a real store: the 4609b359 shape — §4UK (operator 11-12: no confirmations): a task
    whose words once gated it ("Approve …") runs like the rest; nothing waits for the owner."""
    await tool_manage_projects(tctx, action="create", title="Gate first")
    pid = tctx.current_project_id
    await tool_manage_projects(tctx, action="task_decompose", subtasks=[
        "Approve the research budget", "research topic A", "research topic B"])
    out = json.loads(await tool_manage_projects(tctx, action="autoadvance", count="all"))
    st = {t["description"]: t["status"] for t in tctx.project_store.list_tasks(pid)}
    assert set(st.values()) == {"DONE"}
    assert out["stop_reason"] != "needs_user" and not out.get("needs_user_tasks")
    assert out["gated"] == 0 and out["ran"] == 3
    assert out["agent_instruction"].startswith("Autonomous batch ran 3 task(s)")


@pytest.mark.asyncio
async def test_tool_names_a_waiting_task_on_any_stop(tctx, monkeypatch):
    await tool_manage_projects(tctx, action="create", title="Waiting")
    pid = tctx.current_project_id
    await tool_manage_projects(tctx, action="task_decompose", subtasks=["research topic A"])
    tid = tctx.project_store.add_task(pid, "Publish the notes")
    tctx.project_store.update_task(tid, status="NEEDS_USER")

    async def fake_many(*a, **k):
        return AdvanceManyResult([], "deadline", None)
    monkeypatch.setattr(PA, "advance_many", fake_many)
    out = json.loads(await tool_manage_projects(tctx, action="autoadvance", count="all"))
    assert out["stop_reason"] == "deadline"
    assert [t["task_id"] for t in out["needs_user_tasks"]] == [tid]
    assert "not their answer" in out["agent_instruction"]


# ------------------------------------------------------------- the idle loop's wiring

@pytest.mark.asyncio
async def test_the_idle_tick_classifies_with_the_shared_classifier_in_the_background():
    """Through ``_biological_tick``: fails where the idle autoadvance builds
    its own classifier again, runs it in the foreground lane, or drops the
    failure hook (the phase's 'had an error' report)."""
    import datetime
    from unittest.mock import AsyncMock, MagicMock, patch
    from tests.test_biological_watchdog import _make_agent
    import ghost_agent.core.agent as AGM

    agent = _make_agent(idle_seconds=1200, memory_ids=0)
    agent.context.args.autoadvance_idle = True
    agent.context.project_store = MagicMock()
    agent._last_autoadvance_at = datetime.datetime(2000, 1, 1)
    agent.context.llm_client.chat_completion = AsyncMock(side_effect=RuntimeError("worker down"))
    got = {}

    async def fake_unattended(ctx, pid, *, llm_classifier, **kw):
        got["label"] = await llm_classifier("Approve the plan")
        return AdvanceResult(True, "t1", got["label"], "x")

    failed = []
    with patch("ghost_agent.core.project_advancer.idle_candidates",
               return_value=[{"id": "p1", "metadata": {}}]), \
         patch("ghost_agent.core.project_advancer.advance_unattended", fake_unattended), \
         patch.object(AGM, "_idle_phase_failed", side_effect=lambda s, ran, ph, e: failed.append((ph, e))), \
         patch("ghost_agent.core.dream.Dreamer"), \
         patch("ghost_agent.core.agent.random.random", return_value=0.99):
        await agent._biological_tick()
    assert got.get("label") == "research"            # keyword fallback after the failure (§4UK: no word gate)
    _call = agent.context.llm_client.chat_completion.await_args
    payload, kw = _call.args[0], _call.kwargs
    assert payload["messages"][0]["content"].startswith(PA.CLASSIFY_PROMPT)
    assert kw.get("is_background") is True and kw.get("off_main_only") is True
    assert [p for p, _ in failed] == ["autoadvance"]


# ------------------------------------------------------------- r1 M1 / M3 on the real tool

@pytest.mark.asyncio
async def test_an_earlier_failure_outranks_the_gate_on_a_real_store(tctx):
    """r1 M1: the project rolls up to NEEDS_USER (blocked branch, not idle);
    an earlier FAILED task was hidden behind "needs_user"."""
    await tool_manage_projects(tctx, action="create", title="Old failure")
    pid = tctx.current_project_id
    await tool_manage_projects(tctx, action="task_decompose", subtasks=[
        "research topic Z", "Approve the research budget", "research topic A"])
    z = next(t["id"] for t in tctx.project_store.list_tasks(pid) if t["description"] == "research topic Z")
    tctx.project_store.update_task(z, status="FAILED", failure_reason="earlier run")
    out = json.loads(await tool_manage_projects(tctx, action="autoadvance", count="all"))
    assert out["stop_reason"] == "project_failed"
    assert [t["id"] for t in out["failed_tasks"]] == [z]
    assert not out.get("needs_user_tasks")          # §4UK: the "Approve …" task ran


@pytest.mark.asyncio
async def test_a_count_stop_with_a_failure_is_partial_and_lists_it(tctx, monkeypatch):
    """r7 #1: count_reached with a FAILED task must still carry failed_tasks,
    the repair text, and a PARTIAL (not ok) outcome."""
    await tool_manage_projects(tctx, action="create", title="Count fail")
    pid = tctx.current_project_id
    f = tctx.project_store.add_task(pid, "research topic F")
    d = tctx.project_store.add_task(pid, "research topic D")
    tctx.project_store.update_task(f, status="FAILED", failure_reason="boom")
    tctx.project_store.update_task(d, status="DONE")

    async def fake_many(*a, **k):
        return AdvanceManyResult([{"task_id": f, "status": "FAILED", "summary": "x"},
                                  {"task_id": d, "status": "DONE", "summary": "y"}],
                                 "count_reached", 2)
    monkeypatch.setattr(PA, "advance_many", fake_many)
    raw = await tool_manage_projects(tctx, action="autoadvance", count="2")
    out = json.loads(raw)
    assert out["stop_reason"] == "count_reached"
    assert [t["id"] for t in out["failed_tasks"]] == [f]
    assert "When the user wants a failed task fixed" in out["agent_instruction"]
    from ghost_agent.tools.outcome import OutcomeStatus
    assert raw.status is OutcomeStatus.PARTIAL


@pytest.mark.asyncio
async def test_a_gate_only_batch_on_a_failed_ledger_is_failed_not_partial(tctx, monkeypatch):
    """r8 #1: count=1 is a gate tick, nothing ran — FAILED, world unchanged."""
    await tool_manage_projects(tctx, action="create", title="Gate only")
    pid = tctx.current_project_id
    z = tctx.project_store.add_task(pid, "research topic Z")
    g = tctx.project_store.add_task(pid, "Approve the budget")
    tctx.project_store.update_task(z, status="FAILED", failure_reason="old")
    tctx.project_store.update_task(g, status="NEEDS_USER")

    async def fake_many(*a, **k):
        return AdvanceManyResult([{"task_id": g, "status": "NEEDS_USER", "summary": "x",
                                   "classification": "needs_user"}],
                                 "project_failed", None, gated=[g], not_run=[g])
    monkeypatch.setattr(PA, "advance_many", fake_many)
    raw = await tool_manage_projects(tctx, action="autoadvance", count="all")
    from ghost_agent.tools.outcome import OutcomeStatus
    assert raw.status is OutcomeStatus.FAILED and raw.world_changed is False
    assert json.loads(raw)["ran"] == 0


@pytest.mark.asyncio
async def test_a_zero_advance_batch_is_failed_not_partial(tctx, monkeypatch):
    """§4GJ, migrated from test_outcome_consumers_r5's AST pin: a batch that
    advanced NOTHING on a FAILED project must not say "PART OF THIS LANDED"."""
    await tool_manage_projects(tctx, action="create", title="Zero")
    pid = tctx.current_project_id
    z = tctx.project_store.add_task(pid, "research topic Z")
    tctx.project_store.update_task(z, status="FAILED", failure_reason="old")

    async def fake_many(*a, **k):
        return AdvanceManyResult([], "project_failed", None)
    monkeypatch.setattr(PA, "advance_many", fake_many)
    raw = await tool_manage_projects(tctx, action="autoadvance", count="all")
    from ghost_agent.tools.outcome import OutcomeStatus
    assert raw.status is OutcomeStatus.FAILED and raw.world_changed is False


def _held(store, pid, desc):
    tid = store.add_task(pid, desc)
    store.update_task(tid, status="NEEDS_USER",
                      result_summary=f"model experiment? — in your turn — {PA._HELD_MARK}",
                      wait_on="owner_turn")
    return tid


@pytest.mark.asyncio
async def test_a_model_experiment_stop_does_not_also_ask_the_user(tctx, monkeypatch):
    """r1 M3: "do it NOW" and "name them and ask" for the same held task."""
    await tool_manage_projects(tctx, action="create", title="Held")
    pid = tctx.current_project_id
    tid = _held(tctx.project_store, pid, "Run Layer 0 probe on the model")

    async def fake_many(*a, **k):
        return AdvanceManyResult([{"task_id": tid, "status": "NEEDS_USER",
                                   "summary": "model experiment: Run Layer 0"}], "model_experiment", None)
    monkeypatch.setattr(PA, "advance_many", fake_many)
    out = json.loads(await tool_manage_projects(tctx, action="autoadvance", count="all"))
    assert "needs_user_tasks" not in out
    assert [t["task_id"] for t in out["held_tasks"]] == [tid]
    assert "do it NOW" in out["agent_instruction"]
    assert "not their answer" not in out["agent_instruction"]
    assert "Held for the owner's own run" not in out["agent_instruction"]


@pytest.mark.asyncio
async def test_holds_only_needs_no_answer(tctx, monkeypatch):
    await tool_manage_projects(tctx, action="create", title="Held only")
    pid = tctx.current_project_id
    tid = _held(tctx.project_store, pid, "Run Layer 1 probe on the model")

    async def fake_many(*a, **k):
        return AdvanceManyResult([], "needs_user", None)
    monkeypatch.setattr(PA, "advance_many", fake_many)
    out = json.loads(await tool_manage_projects(tctx, action="autoadvance", count="all"))
    assert "needs_user_tasks" not in out
    assert [t["task_id"] for t in out["held_tasks"]] == [tid]
    assert "picks one up once nothing it depends on is still open" in out["agent_instruction"]
    assert "NOT the answer to a gate" not in out["agent_instruction"]
    # r2 #2: rebuilt from the base text, not replacing it
    assert "Report from the summaries above" in out["agent_instruction"]
    assert out["agent_instruction"].startswith("Autonomous batch ran 0 task(s) (0 DONE)")


@pytest.mark.asyncio
async def test_a_hold_on_another_stop_is_named_as_a_hold(tctx, monkeypatch):
    await tool_manage_projects(tctx, action="create", title="Held deadline")
    pid = tctx.current_project_id
    tid = _held(tctx.project_store, pid, "Run Layer 2 probe on the model")
    gid = tctx.project_store.add_task(pid, "Publish the notes")
    tctx.project_store.update_task(gid, status="NEEDS_USER", result_summary="flagged for human review")

    async def fake_many(*a, **k):
        return AdvanceManyResult([], "deadline", None)
    monkeypatch.setattr(PA, "advance_many", fake_many)
    out = json.loads(await tool_manage_projects(tctx, action="autoadvance", count="all"))
    assert [t["task_id"] for t in out["needs_user_tasks"]] == [gid]
    assert [t["task_id"] for t in out["held_tasks"]] == [tid]
    assert "not their answer" in out["agent_instruction"]
    assert "Held for the owner's own run" in out["agent_instruction"]


@pytest.mark.asyncio
async def test_unattended_holds_only_says_they_were_held_for_lack_of_an_owner(tctx, monkeypatch):
    """r3 MAJOR: a scheduled batch whose tasks ran and were held — the text
    must not say they wait on other tasks."""
    import ghost_agent.tools.projects as TP
    await tool_manage_projects(tctx, action="create", title="Sched")
    pid = tctx.current_project_id
    tid = _held(tctx.project_store, pid, "Verify claim 1 against sources")
    monkeypatch.setattr(TP, "_owner_turn_now", lambda: False)
    tctx.project_store.update_project(pid, metadata={"autopilot": True})

    async def fake_many(*a, **k):
        return AdvanceManyResult([{"task_id": tid, "classification": "research",
                                   "status": "NEEDS_USER", "summary": "held"}],
                                 "needs_user", None, [], [tid], [])
    monkeypatch.setattr(PA, "advance_many", fake_many)
    out = json.loads(await tool_manage_projects(tctx, action="autoadvance", count="all"))
    instr = out.get("agent_instruction", "")
    assert [t["task_id"] for t in out["held_tasks"]] == [tid], out
    assert "held because this run is unattended" in instr
    assert "wait on other tasks" not in instr and "depends on" not in instr
    assert instr.startswith("Autonomous batch ran 1 task(s)")
    assert "Report from the summaries above" in instr


@pytest.mark.asyncio
async def test_a_long_hold_reason_keeps_its_tail(tctx, monkeypatch):
    await tool_manage_projects(tctx, action="create", title="Long")
    pid = tctx.current_project_id
    tid = tctx.project_store.add_task(pid, "Verify the pipeline")
    tctx.project_store.update_task(tid, status="NEEDS_USER",
                                   result_summary="wrote x " + "y" * 400 + f" — needs a real check — {PA._HELD_MARK}",
                                   wait_on="owner_turn")

    async def fake_many(*a, **k):
        return AdvanceManyResult([], "deadline", None)
    monkeypatch.setattr(PA, "advance_many", fake_many)
    out = json.loads(await tool_manage_projects(tctx, action="autoadvance", count="all"))
    res = out["held_tasks"][0]["result"]
    assert res.endswith(f"needs a real check — {PA._HELD_MARK}") and len(res) <= 200


# ------------------------------------------------------------- r4

class _NUStore(_FakeStore):
    def get_project(self, pid):
        return {"status": "NEEDS_USER"}


@pytest.mark.asyncio
@pytest.mark.parametrize("summary,expected", [
    ("project is NEEDS_USER, not ACTIVE", "needs_user"),
    ("budget exhausted: 40/40 steps", "budget_or_inactive"),      # r4 MAJOR
    ("tool-call cap reached", "budget_or_inactive"),
])
async def test_only_the_not_active_refusal_on_a_needs_user_project_is_needs_user(monkeypatch, summary, expected):
    ctx = SimpleNamespace(project_store=_NUStore({}))
    _script(monkeypatch, [AdvanceResult(True, None, "blocked", summary)])
    r = await advance_many(ctx, "p", owner_requested=True, max_tasks=None, stop_on_fail=False)
    assert r.stop_reason == expected


@pytest.mark.asyncio
async def test_a_gate_in_the_paused_branch_is_booked_as_a_gate(monkeypatch):
    ctx = SimpleNamespace(project_store=_FakeStore({"t1": "NEEDS_USER"}))
    _script(monkeypatch, [AdvanceResult(True, "t1", "needs_user", "task requires human input")])
    monkeypatch.setattr(PA, "_paused_since", lambda *a, **k: True)
    r = await advance_many(ctx, "p", owner_requested=False, max_tasks=None, stop_on_fail=False)
    assert r.stop_reason == "autopilot_paused"
    assert r.gated == ["t1"] and r.ran == 0


@pytest.mark.asyncio
async def test_a_ran_then_held_task_in_the_paused_branch_is_a_hold(monkeypatch):
    ctx = SimpleNamespace(project_store=_FakeStore({"t1": "NEEDS_USER"}))
    _script(monkeypatch, [AdvanceResult(True, "t1", "research", f"held — {PA._HELD_MARK}")])
    monkeypatch.setattr(PA, "_paused_since", lambda *a, **k: True)
    r = await advance_many(ctx, "p", owner_requested=False, max_tasks=None, stop_on_fail=False)
    assert r.held == ["t1"] and r.ran == 1 and r.not_run == []


@pytest.mark.asyncio
async def test_a_ran_then_held_task_stopping_safe_mode_is_a_hold(monkeypatch):
    ctx = SimpleNamespace(project_store=_FakeStore({"t1": "NEEDS_USER"}))
    _script(monkeypatch, [AdvanceResult(True, "t1", "research", f"held — {PA._HELD_MARK}")])
    r = await advance_many(ctx, "p", owner_requested=False, max_tasks=None, stop_on_fail=True)
    assert r.stop_reason == "needs_user" and r.held == ["t1"] and r.ran == 1


def test_tally_units_are_kept_apart():
    """r4 #5: 'ran N (D DONE, H held…)' are among N; gates and holds made
    before running are extra ticks."""
    b = AdvanceManyResult(
        [{"task_id": "x", "status": "NEEDS_USER"}, {"task_id": "a", "status": "DONE"},
         {"task_id": "b", "status": "NEEDS_USER"}, {"task_id": "g", "status": "NEEDS_USER"}],
        "count_reached", 2, gated=["g"], held=["x", "b"], not_run=["x", "g"])
    assert b.ran == 2
    instr = _advance_batch_instruction(b, 2)
    assert instr.startswith("Autonomous batch ran 2 task(s) (1 DONE, 1 held for the owner's own run; "
                            "1 more left waiting at a gate; 1 more held before running)")


@pytest.mark.asyncio
async def test_the_batch_goes_past_a_model_experiment_to_independent_tasks(monkeypatch):
    """§4SU Phase 5: an experiment no longer stops the owner's batch."""
    ctx = SimpleNamespace(project_store=_FakeStore({"e": "PENDING", "t2": "DONE", "t3": "DONE"}))
    seen = []
    seq = [AdvanceResult(True, "e", "blocked", PA.MODEL_EXPERIMENT_PREFIX + ": run the prompts"),
           AdvanceResult(True, "t2", "research", "ok"),
           AdvanceResult(True, "t3", "research", "ok"),
           AdvanceResult(True, None, "idle", "no ready leaf")]

    async def fake(context, project_id, **kw):
        seen.append(set(kw.get("exclude_ids") or ()))
        return seq.pop(0)
    monkeypatch.setattr(PA, "advance_once", fake)
    monkeypatch.setattr(PA, "_paused_since", lambda *a, **k: False)
    r = await advance_many(ctx, "p", owner_requested=True, max_tasks=None, stop_on_fail=False)
    assert r.stop_reason == "model_experiment" and r.ran == 2 and r.experiments == ["e"]
    assert "e" in seen[1] and "e" in seen[2]              # the pick skips it after
    text = _advance_batch_instruction(r, None)
    assert "Task e" in text and "Autonomous batch ran 2 task(s)" in text


@pytest.mark.asyncio
async def test_a_count_still_counts_only_the_tasks_that_ran(monkeypatch):
    ctx = SimpleNamespace(project_store=_FakeStore({"e": "PENDING", "t2": "DONE", "t3": "DONE"}))
    left = _script(monkeypatch, [AdvanceResult(True, "e", "blocked", PA.MODEL_EXPERIMENT_PREFIX + ": x"),
                                 AdvanceResult(True, "t2", "research", "ok"),
                                 AdvanceResult(True, "t3", "research", "ok")])
    r = await advance_many(ctx, "p", owner_requested=True, max_tasks=1, stop_on_fail=False)
    assert r.ran == 1 and r.stop_reason == "model_experiment" and len(left) == 1


@pytest.mark.asyncio
async def test_an_unattended_batch_still_stops_at_a_model_experiment(monkeypatch):
    ctx = SimpleNamespace(project_store=_FakeStore({"e": "PENDING", "t2": "DONE"}))
    left = _script(monkeypatch, [AdvanceResult(True, "e", "blocked", PA.MODEL_EXPERIMENT_PREFIX + ": x"),
                                 AdvanceResult(True, "t2", "research", "ok")])
    r = await advance_many(ctx, "p", owner_requested=False, max_tasks=None, stop_on_fail=False)
    assert r.stop_reason == "model_experiment" and len(left) == 1
