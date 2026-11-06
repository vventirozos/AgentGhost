"""§4TD (2026-11-06): the lesson store is removed — `memory/skills.py`, its
retrieval, `learn_skill`, `list_lessons`, the lesson rows of every report. What
this file holds to, by driving the code:

  * nothing under src/ imports the removed modules; neither tool is advertised
    or dispatchable, and a stale call gets the ORDINARY unknown-tool reply;
  * an old `skills_playbook.json` is never read, rewritten or recreated, and no
    prompt part carries a lesson; the old vector twins are never served;
  * the owner's thumb still records a label, moves the mood and returns the
    hint;
  * THE DANGEROUS PART: every guard that read `skill_memory.is_read_only` now
    reads one explicit, fail-closed flag. Each guard below is driven twice: as
    the simulated turn it guards, and as an ordinary turn.

Fictional owner data only (Elena, dog Bruno).
"""
import ast
import asyncio
import importlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ghost_agent.core import agent as A
from ghost_agent.core.agent import GhostAgent, turn_origin
from ghost_agent.core.isolation import is_simulated_context, isolated_replay_context, mark_simulated
from ghost_agent.tools import registry as R
from tests.helpers import FakeBgTasks, make_context
from tests.test_4tb_replan_removed import _strings
from tests.test_turnloop_r1_fixes import _call, _ts

SRC = Path(__file__).resolve().parent.parent / "src" / "ghost_agent"
REMOVED_MODULES = ("ghost_agent.memory.skills", "ghost_agent.memory.lesson_scope", "ghost_agent.memory.lesson_quality")
REMOVED_TOOLS = ("learn_skill", "list_lessons")
LESSON_CANARY = "BRUNO-LESSON-CANARY always ask Elena first"


def _playbook(memory_dir) -> Path:
    f = Path(memory_dir) / "skills_playbook.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps([{"task": LESSON_CANARY, "trigger": LESSON_CANARY, "mistake": "m", "solution": LESSON_CANARY,
                              "retrievals": 9, "source": "learn_skill", "origin": "user"}]))
    return f


def _agent_on_registry(ctx):
    agent = GhostAgent(ctx)
    agent.available_tools = R.get_available_tools(ctx)
    return agent


# ── the modules ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("module", REMOVED_MODULES)
def test_a_removed_module_cannot_be_imported(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_no_module_under_src_imports_a_removed_module():
    """Lazy imports inside functions as well: the walk is over the whole tree."""
    gone = {m.rsplit(".", 1)[1] for m in REMOVED_MODULES}
    seen = 0
    hits = []
    for f in sorted(SRC.rglob("*.py")):
        seen += 1
        for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Import):
                names = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                names = [(n.module or "")] + [f"{n.module or ''}.{a.name}" for a in n.names]
            else:
                continue
            if any(x.rsplit(".", 1)[-1] in gone and "memory" in x for x in names) or any(
                    x.split(".")[0] in gone for x in names if isinstance(n, ast.ImportFrom) and n.level):
                hits.append((str(f), n.lineno))
    assert seen > 100 and hits == []


def test_the_read_only_wrapper_is_gone_and_the_context_has_no_lesson_store():
    import ghost_agent.core.isolation as I
    import ghost_agent.memory.readonly as RO
    assert not hasattr(I, "ReadOnlySkillMemory") and not hasattr(RO, "ReadOnlySkillMemory")
    live = A.GhostContext(args=SimpleNamespace(), sandbox_dir=None, memory_dir=None, tor_proxy=None)
    assert not hasattr(live, "skill_memory")
    assert live.simulated is False and is_simulated_context(live) is False


# ── the tools ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", REMOVED_TOOLS)
def test_a_removed_tool_is_neither_advertised_nor_dispatchable(mock_context, name):
    assert name not in {d["function"]["name"] for d in R.TOOL_DEFINITIONS}
    assert name not in {d["function"]["name"] for d in R.get_active_tool_definitions(mock_context, serve_tuned=False)}
    assert name not in R.get_available_tools(mock_context)


@pytest.mark.asyncio
@pytest.mark.parametrize("name,args", [
    ("learn_skill", {"task": "feed Bruno", "mistake": "late", "solution": "feed him at noon"}),
    ("list_lessons", {"scope": "all"}),
])
async def test_a_stale_call_gets_the_ordinary_unknown_tool_reply_through_the_real_dispatch(mock_context, name, args):
    """With the old playbook on disk and a custom tool in the registry: nothing
    is written, the custom tool is not run in its place, the rebuild does not
    bring the name back."""
    from ghost_agent.tools.acquired_skills import AcquiredSkillManager
    playbook = _playbook(mock_context.memory_dir)
    before = playbook.read_bytes()
    mgr = AcquiredSkillManager(mock_context.memory_dir, None)
    mgr.save_skill("bruno_walk_report", "report the walk", {}, "print(1)")
    reg_file = next(Path(mock_context.memory_dir).rglob("skills_registry.json"))
    reg_before = json.loads(reg_file.read_text())["bruno_walk_report"]
    mock_context._subagent_allowed_tools = None
    agent = _agent_on_registry(mock_context)
    agent.disabled_tools = set()
    ts = _ts(tool_calls=[_call(name, args)])
    await agent._dispatch_and_process_tool_batch(ts)
    syn = [m for m in ts.tools_run_this_turn if m.get("_synthetic")]
    assert len(syn) == 1 and len(ts.tools_run_this_turn) == 1
    reply = syn[0]["content"]
    assert f"Unknown tool '{name}'" in reply and getattr(reply, "reason_code", None) == "unknown_tool"
    assert "bruno_walk_report" not in reply or "create_skill" in reply      # the ordinary list, or none
    assert "learn a lesson" not in reply.lower() and "playbook" not in reply.lower()
    assert ts.execution_failure_count == 1
    assert name not in agent.available_tools
    assert playbook.read_bytes() == before
    assert json.loads(reg_file.read_text())["bruno_walk_report"]["usage_count"] == reg_before["usage_count"]


@pytest.mark.parametrize("spelled", ["learn_skill", "learnSkill", "learn-skill", "LEARN_SKILL",
                                     "list_lessons", "listLessons", "list-lessons", "learn_skills", "list_lesson"])
def test_the_healer_never_turns_a_stale_name_into_a_live_tool(spelled):
    """`create_skill` is a close neighbour of `learn_skill`: a stale call must
    not run it. The name stays what it was (and so reaches the unknown-tool
    reply) however it is spelled."""
    live = ["create_skill", "manage_skills", "knowledge_base", "recall"]
    assert GhostAgent._canonicalise_tool_name(spelled, live) not in live


def test_every_list_of_tool_names_has_lost_the_removed_names():
    """The lists as data, not as text: the module-level frozensets and dicts."""
    from ghost_agent.core.isolation import REPLAY_FORBIDDEN_TOOLS
    collections = [REPLAY_FORBIDDEN_TOOLS]
    for mod in (A,):
        for key, value in vars(mod).items():
            if isinstance(value, (frozenset, set, tuple, list, dict)) and key != "_REMOVED_TOOL_NAMES":
                collections.append(value)
    checked = 0
    for c in collections:
        try:
            members = list(c)
        except TypeError:
            continue
        checked += 1
        for gone in REMOVED_TOOLS:
            assert gone not in members, gone
    assert checked > 50
    assert {"learnskill", "listlessons"} <= set(A._REMOVED_TOOL_NAMES)


def test_no_model_facing_text_names_a_removed_tool_or_teaches_lessons(mock_context):
    from ghost_agent.core.prompts import SYSTEM_PROMPT
    defs = list(R.TOOL_DEFINITIONS) + list(R.get_active_tool_definitions(mock_context, serve_tuned=False))
    seen = 0
    for d in defs:
        for path, text in _strings(d):
            if not path.endswith("/description"):
                continue
            seen += 1
            low = text.lower()
            for word in REMOVED_TOOLS + ("lesson", "playbook"):
                assert word not in low, (d["function"]["name"], path, word)
    assert seen > 60
    for word in REMOVED_TOOLS + ("lesson", "playbook"):
        assert word not in SYSTEM_PROMPT.lower(), word


# ── old data on disk ─────────────────────────────────────────────────────────

def _all_prompts(ctx) -> str:
    out = []
    for call in ctx.llm_client.chat_completion.await_args_list:
        payload = call.args[0] if call.args else call.kwargs.get("payload", {})
        for m in (payload or {}).get("messages", []):
            out.append(str(m.get("content") or ""))
    return "\n".join(out)


@pytest.mark.asyncio
async def test_an_old_playbook_is_not_read_rewritten_or_shown_on_a_turn(tmp_path):
    ctx = make_context()
    ctx.memory_dir = tmp_path / "system" / "memory"
    playbook = _playbook(ctx.memory_dir)
    before, stamp = playbook.read_bytes(), playbook.stat().st_mtime_ns
    agent = GhostAgent(ctx)
    real_open = open

    def spying_open(file, *a, **k):
        assert "skills_playbook" not in str(file), "something opened the old playbook"
        return real_open(file, *a, **k)
    with patch("builtins.open", spying_open):
        await agent.handle_chat({"messages": [{"role": "user", "content": "what should Bruno eat tonight?"}], "model": "m"},
                                FakeBgTasks())
    assert LESSON_CANARY not in _all_prompts(ctx)
    assert playbook.read_bytes() == before and playbook.stat().st_mtime_ns == stamp


@pytest.mark.asyncio
async def test_a_turn_and_an_idle_cycle_do_not_create_a_playbook(tmp_path):
    from ghost_agent.core.dream import Dreamer
    ctx = make_context()
    ctx.memory_dir = tmp_path / "system" / "memory"
    ctx.memory_dir.mkdir(parents=True)
    ctx.project_store = None
    ctx.trajectory_collector = None
    ctx.graph_memory = None
    ctx._last_store_maintenance = None
    await GhostAgent(ctx).handle_chat({"messages": [{"role": "user", "content": "tell me about dogs like Bruno"}],
                                       "model": "m"}, FakeBgTasks())
    await Dreamer(ctx).dream("m")
    assert not list(tmp_path.rglob("skills_playbook*"))
    assert not list(tmp_path.rglob("skills_pruned_archive*"))


@pytest.fixture
def twin_store(tmp_path):
    """A REAL store holding a lesson's old vector copy beside an owner's fact."""
    from ghost_agent.memory.vector import VectorMemory
    vm = VectorMemory(tmp_path / "vec", "http://mock-url")
    vm.collection.add(ids=["twin-1", "fact-1"],
                      documents=[f"Bruno the dog {LESSON_CANARY}", "Bruno the dog sleeps on the blue sofa"],
                      metadatas=[{"type": "skill", "timestamp": "2026-10-01T10:00:00Z"},
                                 {"type": "manual", "timestamp": "2026-10-01T10:00:00Z"}])
    return vm


def test_a_twin_is_never_returned_by_search_advanced(twin_store):
    rows = twin_store.search_advanced("Bruno the dog", limit=5, record_retrievals=False)
    assert rows, "the search found nothing at all: the test proves nothing"
    assert not [r for r in rows if LESSON_CANARY in str(r)]
    assert [r for r in rows if "blue sofa" in str(r)]


def test_a_twin_asked_for_by_its_type_returns_nothing(twin_store):
    assert twin_store.search_advanced("Bruno the dog", limit=5, where={"type": "skill"}, record_retrievals=False) == []


def test_a_twin_is_never_returned_by_search_items_or_search(twin_store):
    items = twin_store.search_items("Bruno the dog sleeps", inject_identity=False)
    assert items and not [i for i in items if LESSON_CANARY in str(i)]
    text = twin_store.search("Bruno the dog sleeps", inject_identity=False, record_retrievals=False)
    assert "blue sofa" in str(text) and LESSON_CANARY not in str(text)


def test_a_twin_the_store_returns_anyway_is_dropped_by_the_second_filter():
    """The post-filter by itself (a store that ignores `where`)."""
    from ghost_agent.memory.vector import VectorMemory, is_retired_row, without_retired_rows
    vm = VectorMemory.__new__(VectorMemory)
    vm.collection = MagicMock()
    vm.collection.query.return_value = {"ids": [["a", "b"]], "documents": [["twin", "fact"]],
                                        "metadatas": [[{"type": "skill"}, {"type": "manual"}]], "distances": [[0.1, 0.2]]}
    vm._get_lock = MagicMock()
    vm._get_lock.return_value.__enter__ = lambda s: None
    vm._get_lock.return_value.__exit__ = lambda s, *a: False
    rows = vm.search_advanced("q", limit=2, record_retrievals=False)
    assert [r["text"] for r in rows] == ["fact"]
    assert is_retired_row({"type": " Skill "}) and is_retired_row(object()) and not is_retired_row({"type": "auto"})
    assert not is_retired_row(None) and not is_retired_row({})
    assert without_retired_rows(None) == {"type": {"$nin": ["skill"]}}
    assert without_retired_rows({"type": "skill"}) is None
    assert without_retired_rows({"type": "manual"}) == {"type": "manual"}
    assert without_retired_rows({"type": {"$nin": ["document"]}}) == {"type": {"$nin": ["document", "skill"]}}
    assert without_retired_rows({"source": "x"}) == {"$and": [{"source": "x"}, {"type": {"$nin": ["skill"]}}]}


@pytest.mark.asyncio
async def test_recall_and_the_project_probe_do_not_serve_a_twin(twin_store):
    from ghost_agent.tools.memory import tool_recall
    from ghost_agent.tools.projects import _not_found_with_recall
    out = await tool_recall(query="Bruno the dog sleeps", memory_system=twin_store)
    assert "blue sofa" in str(out) and LESSON_CANARY not in str(out)
    probe = await _not_found_with_recall(SimpleNamespace(memory_system=twin_store), "Bruno the dog sleeps")
    assert LESSON_CANARY not in str(probe)


# ── the owner's thumb ────────────────────────────────────────────────────────

def _feedback_app(tmp_path, playbook_dir=None):
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from ghost_agent.api.routes import router
    from ghost_agent.distill.collector import TrajectoryCollector
    from ghost_agent.distill.schema import Outcome, Trajectory
    from ghost_agent.selfhood import SelfModel
    import datetime as dt
    coll = TrajectoryCollector(root=tmp_path / "traj", session_id="s")
    t = Trajectory(session_id="r1", user_request="how old is Bruno?", final_response="nine", task_kind="user_request",
                   outcome=Outcome.UNKNOWN.value, extra={"req_id": "r1"})
    t.timestamp = dt.datetime.utcnow().isoformat() + "Z"
    assert coll.append(t) is not None
    sm = SelfModel(root=tmp_path / "self")
    app = FastAPI()
    app.include_router(router)
    agent = MagicMock()
    agent.context = SimpleNamespace(args=SimpleNamespace(api_key="test-key"), trajectory_collector=coll,
                                    _recent_trajectories_for_correction={}, self_model=sm, turn_origin_label=None,
                                    memory_dir=tmp_path / "memory")
    app.state.agent = agent
    return app, coll, t, sm, agent


@pytest.mark.parametrize("signal,outcome,hinted,label", [("negative", "failed", True, "down"),
                                                         ("positive", "passed", False, "content")])
def test_the_thumb_records_the_label_moves_the_mood_and_returns_the_hint(tmp_path, signal, outcome, hinted, label):
    from fastapi.testclient import TestClient
    from ghost_agent.core.owner_rules import FEEDBACK_HINT
    app, coll, t, sm, agent = _feedback_app(tmp_path)
    playbook = _playbook(tmp_path / "memory")
    before = playbook.read_bytes()
    with TestClient(app) as c:
        r = c.post("/api/feedback", headers={"X-Ghost-Key": "test-key"}, json={"request_id": "r1", "signal": signal, "source": "web"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] and body["outcome"] == outcome
    assert coll.latest_correction(t.id)["outcome"] == outcome
    assert sm.mood().label == label
    assert (body.get("hint") == FEEDBACK_HINT) is hinted and ("hint" in body) is hinted
    agent._drop_pending_corrections_for.assert_called_once_with(t.id)
    # no lesson side effect: the file and everything under the memory dir as they were
    assert playbook.read_bytes() == before and [p.name for p in (tmp_path / "memory").iterdir()] == [playbook.name]
    assert not [c for c in agent.method_calls if "lesson" in c[0] or "flush" in c[0]]


# ── THE MARKER ───────────────────────────────────────────────────────────────

def test_the_flag_is_set_on_the_replay_copy_and_the_live_context_stays_as_it_was(tmp_path):
    live = SimpleNamespace(sandbox_dir=str(tmp_path), memory_dir=str(tmp_path / "m"), args=SimpleNamespace(perfect_it=True),
                           simulated=False)

    async def go():
        async with isolated_replay_context(live, with_sandbox=False) as run:
            return is_simulated_context(run.context), turn_origin(run.context)
    assert asyncio.run(go()) == (True, "sim")
    assert live.simulated is False and is_simulated_context(live) is False


def test_the_flag_is_set_on_a_coding_leaf():
    from ghost_agent.core import coding_loop as CL
    base = SimpleNamespace(sandbox_dir="/nonexistent/sb", current_project_id=None, memory_system=None, graph_memory=None,
                           args=SimpleNamespace(perfect_it=True, smart_memory=0.5, native_tools=False),
                           workspace_model=None, journal=None, llm_client=object(), simulated=False)
    leaf = CL.build_leaf_context(base, leaf_id="L1", project_id="p1")
    assert is_simulated_context(leaf) and not is_simulated_context(base)


def test_a_leaf_that_will_not_take_the_flag_is_refused():
    from ghost_agent.core import coding_loop as CL

    class Refusing(SimpleNamespace):
        def __setattr__(self, k, v):
            if k == "simulated":
                raise AttributeError("read only")
            super().__setattr__(k, v)
    base = Refusing(sandbox_dir="/nonexistent/sb", current_project_id=None, memory_system=None, graph_memory=None,
                    args=SimpleNamespace(perfect_it=True, smart_memory=0.5, native_tools=False),
                    workspace_model=None, journal=None, llm_client=object())
    with pytest.raises(RuntimeError, match="simulated"):
        CL.build_leaf_context(base, leaf_id="L1", project_id="p1")


def test_the_flag_fails_closed():
    class Unreadable:
        @property
        def simulated(self):
            raise OSError("no")
    assert is_simulated_context(Unreadable()) is True
    assert is_simulated_context(None) is False
    assert is_simulated_context(MagicMock()) is False, "a double's auto-attribute is not a stamp"
    assert is_simulated_context(SimpleNamespace(simulated=1)) is False
    assert is_simulated_context(SimpleNamespace()) is False
    ns = SimpleNamespace()
    mark_simulated(ns)
    assert ns.simulated is True and is_simulated_context(ns) is True


@pytest.mark.parametrize("sim,expected", [(True, "sim"), (False, "user")])
def test_site_turn_origin(sim, expected):
    """Guards: the population every ledger gates on (`== "user"`)."""
    assert turn_origin(SimpleNamespace(simulated=sim)) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("sim,identified", [(True, False), (False, True)])
async def test_site_verifier_ledger_identity(monkeypatch, sim, identified):
    """Guards: a simulated verdict must not land in the production escalation ledger."""
    monkeypatch.setenv("GHOST_VERIFIER_JUDGE", "1")
    seen = {}

    class V:
        llm_client = object()

        async def verify_claim(self, claim, evidence, context="", *, high_stakes=False, trace=None, **_k):
            seen["trace"] = trace
            return None
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = SimpleNamespace(verifier=V(), args=SimpleNamespace(no_verifier=False), simulated=sim)
    agent._active_constraint_note = lambda limit=5, **_k: ""
    await agent._compute_verifier_verdict(tools_run_this_turn=[{"name": "web_search", "content": "x"}], messages=[],
                                          final_ai_content="answer", last_user_content="q", lc="q",
                                          req_id="r1", trajectory_id="t1")
    assert "trace" in seen
    assert (seen["trace"] is not None) is identified


@pytest.mark.parametrize("sim,owned", [(True, False), (False, True)])
def test_site_pending_corrections_queue(tmp_path, sim, owned):
    """Guards: a simulated copy shares memory_dir and must neither load nor save the owner's corrections."""
    agent = GhostAgent.__new__(GhostAgent)
    agent.context = SimpleNamespace(memory_dir=str(tmp_path), owner_memory_isolated=False, simulated=sim)
    assert (agent._pending_corrections_path() is not None) is owned


def _exec_result():
    return "--- EXECUTION RESULT ---\nEXIT CODE: 0\nSTDOUT/STDERR:\nanswer 42\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("sim,stopped", [(True, True), (False, False)])
async def test_site_solution_py_short_circuit(mock_context, sim, stopped):
    """Guards: only a simulated attempt skips the confirmation turn after `python3 solution.py` exits 0."""
    mock_context.simulated = sim
    agent = GhostAgent(mock_context)
    agent.available_tools = {"execute": AsyncMock(return_value=_exec_result())}
    args = {"command": "python3 solution.py"}
    ts = _ts(tool_calls=[_call("execute", args)],
             msg={"role": "assistant", "content": "", "tool_calls": [_call("execute", args)]})
    await agent._dispatch_and_process_tool_batch(ts)
    assert (ts.final_ai_content == "solution.py executed successfully (exit 0).") is stopped
    assert bool(ts.force_stop) is stopped


def _turn_ctx(sim, **over):
    ctx = make_context(**over)
    ctx.simulated = sim
    ctx.memory_bus = None
    ctx.self_model = None
    return ctx


def _resp(content, tool_calls=None):
    return {"choices": [{"message": {"role": "assistant", "content": content, "tool_calls": tool_calls or []}}]}


def _tool_call(name, args, tid="c1"):
    return {"id": tid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


@pytest.mark.asyncio
@pytest.mark.parametrize("sim,generated", [(True, False), (False, True)])
async def test_site_perfect_it(monkeypatch, sim, generated):
    """Guards: a simulated turn spends no model call on an optimisation suggestion."""
    ctx = _turn_ctx(sim)
    ctx.args.perfect_it = True
    agent = GhostAgent(ctx)
    agent.available_tools = {"execute": AsyncMock(return_value=_exec_result())}
    called = []

    async def fake(payload, foreground=False):
        called.append(foreground)
        return "use a loop"
    agent._perfect_it_generate = fake
    ctx.llm_client.chat_completion = AsyncMock(side_effect=[
        _resp("", [_tool_call("execute", {"content": "print(42)"})]),
        _resp("The answer is 42."), _resp("The answer is 42.")])
    monkeypatch.setenv("GHOST_EVIDENCE_GATE", "0")
    await agent.handle_chat({"messages": [{"role": "user", "content": "run a script that prints the answer"}], "model": "m"},
                            FakeBgTasks())
    assert bool(called) is generated


@pytest.mark.asyncio
@pytest.mark.parametrize("sim,fired", [(True, False), (False, True)])
async def test_site_notify_backstop_at_finalize(monkeypatch, sim, fired):
    """Guards: a simulated prompt that says "notify me" must not page the owner."""
    backstop = MagicMock(return_value=True)
    monkeypatch.setattr(A, "_notify_promise_backstop", backstop)
    ctx = _turn_ctx(sim)
    await GhostAgent(ctx).handle_chat({"messages": [{"role": "user", "content": "explain dogs and notify me when done"}],
                                       "model": "m"}, FakeBgTasks())
    assert backstop.called is fired


@pytest.mark.asyncio
@pytest.mark.parametrize("sim,fired", [(True, False), (False, True)])
async def test_site_stored_promise_fire_at_finalize(monkeypatch, sim, fired):
    """Guards: a simulated turn must not consume or fire a promise stored on the owner's project."""
    import ghost_agent.core.notify_promise as NP
    fire = MagicMock(return_value=False)
    monkeypatch.setattr(NP, "fire_promise_if_settled", fire)
    monkeypatch.setattr(NP, "peek_promise", lambda *a, **k: None)
    ctx = _turn_ctx(sim)
    ctx.current_project_id = "p1"
    ctx.project_store = MagicMock()
    import ghost_agent.tools.projects as TP
    monkeypatch.setattr(TP, "begin_request_focus", lambda *a, **k: None)   # keep the binding
    await GhostAgent(ctx).handle_chat({"messages": [{"role": "user", "content": "explain how dogs sleep"}], "model": "m"},
                                      FakeBgTasks())
    assert fire.called is fired


@pytest.mark.asyncio
@pytest.mark.parametrize("sim,captured", [(True, False), (False, True)])
async def test_site_notify_promise_capture(monkeypatch, sim, captured):
    """Guards: a simulated prompt echoing "notify me" stores no promise on the owner's project."""
    import ghost_agent.core.notify_promise as NP
    capture = MagicMock(return_value=False)
    monkeypatch.setattr(NP, "capture_promise", capture)
    monkeypatch.setattr(A, "_user_asked_for_notification", lambda *a, **k: True)
    ctx = _turn_ctx(sim)
    ctx.current_project_id = "p1"
    ctx.project_store = MagicMock()
    import ghost_agent.tools.projects as TP
    monkeypatch.setattr(TP, "begin_request_focus", lambda *a, **k: None)   # keep the binding
    await GhostAgent(ctx).handle_chat({"messages": [{"role": "user", "content": "explain dogs and notify me when done"}],
                                       "model": "m"}, FakeBgTasks())
    assert capture.called is captured


@pytest.mark.asyncio
@pytest.mark.parametrize("sim,consumed", [(True, False), (False, True)])
async def test_site_warmup_head_check(caplog, sim, consumed):
    """Guards: an isolate's head differs by design; it must not raise a false MISS or use up the one-shot check."""
    ctx = _turn_ctx(sim)
    ctx.profile_memory = MagicMock()
    ctx.profile_memory.get_context_string = MagicMock(return_value="")
    ctx._warmed_sys_hash = "not-the-real-hash"
    with caplog.at_level(logging.WARNING, logger="GhostAgent"):
        await GhostAgent(ctx).handle_chat({"messages": [{"role": "user", "content": "explain how dogs sleep"}], "model": "m"},
                                          FakeBgTasks())
    assert (ctx._warmed_sys_hash is None) is consumed
    assert any("prefix warmup MISS" in r.getMessage() for r in caplog.records) is consumed


async def _stream_turn(monkeypatch, sim, project=None):
    """The real streamed tail, every drain consumer as in `test_stream_treated_view`."""
    from tests.test_finalize_stream_pins import make_stream_agent, sse
    from ghost_agent.core.agent import StreamState
    a = make_stream_agent()
    a.context.simulated = sim
    a.context.args.smart_memory = 0.0
    a.context.args.no_verifier = True
    a.context.verifier = None
    a.context.project_store = MagicMock() if project else None

    async def final_stream(payload, use_coding=False):
        yield sse({"content": "Dogs sleep a lot."})
        yield b"data: [DONE]\n\n"
    a.context.llm_client.stream_chat_completion = final_stream
    reg = MagicMock()
    reg.is_cancelled.return_value = False
    st = StreamState(created_time=123, current_trajectory_id="t1", execution_failure_count=0, force_stop=False,
                     forget_was_called=False, has_coding_intent=False, is_final_generation=True,
                     last_user_content="explain dogs and notify me when done", last_was_failure=False,
                     lc="explain dogs and notify me when done",
                     messages=[{"role": "user", "content": "explain dogs and notify me when done"}], model="m",
                     payload={"messages": []}, req_id="rid00001", stream_conv_fp="fp", stream_messages_snapshot=[],
                     stream_model="m", stream_prefix="", stream_tools_snapshot=[], stream_verify_messages=[],
                     was_complex_task=False, _active_turn=None, _proj_task_closed_this_req=False, _turn_reg=reg)
    if project:
        a.context.current_project_id = project
    gen, _, _ = a._stream_final_generation(st)
    [c async for c in gen]
    await asyncio.sleep(0.02)


@pytest.mark.asyncio
@pytest.mark.parametrize("sim,fired", [(True, False), (False, True)])
async def test_site_notify_backstop_in_the_streamed_tail(monkeypatch, sim, fired):
    backstop = MagicMock(return_value=True)
    monkeypatch.setattr(A, "_notify_promise_backstop", backstop)
    await _stream_turn(monkeypatch, sim)
    assert backstop.called is fired


@pytest.mark.asyncio
@pytest.mark.parametrize("sim,fired", [(True, False), (False, True)])
async def test_site_stored_promise_fire_in_the_streamed_tail(monkeypatch, sim, fired):
    import ghost_agent.core.notify_promise as NP
    fire = MagicMock(return_value=False)
    monkeypatch.setattr(NP, "fire_promise_if_settled", fire)
    monkeypatch.setattr(NP, "peek_promise", lambda *a, **k: None)
    monkeypatch.setattr(A, "_notify_promise_backstop", MagicMock(return_value=False))
    await _stream_turn(monkeypatch, sim, project="p1")
    assert fire.called is fired


def test_the_guards_read_the_flag_through_one_function():
    """No guard reads the attribute by itself: a private truth test would turn
    every MagicMock context into a simulated one, or the other way round."""
    tree = ast.parse((SRC / "core" / "agent.py").read_text(encoding="utf-8"))
    readers = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "simulated"
               and not isinstance(n.ctx, ast.Store)]        # `GhostContext.__init__` sets the plain False
    getattrs = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "getattr"
                and len(n.args) >= 2 and isinstance(n.args[1], ast.Constant) and n.args[1].value == "simulated"]
    assert readers == [] and getattrs == []
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "is_simulated_context"]
    assert len(calls) == 8, "a guard was added or dropped: give it a site test above and change this count"
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "is_read_only"]


# ── the views ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_learning_report_and_the_overview_render_without_a_lesson_row(tmp_path):
    from ghost_agent.core.learning_health import collect_learning_health, render_learning_health
    from ghost_agent.core.liveness import YIELD_PROBES, yield_all
    from ghost_agent.tools.introspect import tool_introspect
    md = tmp_path / "system" / "memory"
    _playbook(md)
    assert "lessons" not in collect_learning_health(md)
    report = render_learning_health(md)
    assert "LEARNING HEALTH" in report
    for gone in ("LESSONS:", "lessons.playbook", "playbook", LESSON_CANARY):
        assert gone not in report, gone
    assert [p.name for p in YIELD_PROBES] == ["skills.acquired", "prompts.gepa", "verifier.rubric_shadow"]
    assert not [r for r in yield_all(tmp_path)["rows"] if "lesson" in r["name"]]
    ctx = SimpleNamespace(memory_dir=str(md), args=None, activity_log=None)
    full = await tool_introspect(action="learning", context=ctx)
    assert "LEARNING HEALTH" in str(full) and "LESSONS:" not in str(full)
    overview = await tool_introspect(action="overview", self_model=None, context=ctx)
    assert "Background loops" in str(overview)
    assert "LESSONS" not in str(overview) and "lesson" not in str(overview).lower()


def test_the_health_check_says_the_owners_rules_and_nothing_of_lessons(tmp_path):
    from ghost_agent.core import owner_rules as OR
    from tests.test_4sy_health_internals import _ctx, _run
    ctx = _ctx(tmp_path)
    _playbook(ctx.memory_dir)
    (tmp_path / "system" / "memory" / "skills_playbook.json").write_text("{torn")     # a damaged one is nobody's problem
    OR.rule_command("learn this rule: answer in English", tmp_path / "system")
    by, problems, _ = _run(ctx)
    assert by["Rules"] == "1 owner rule (of at most 10, shown on every turn)"
    assert "Learning" not in by
    assert not [p for p in problems if "lesson" in p or "playbook" in p]
    assert "lesson" not in " ".join(by.values()).lower()


# ── what stays ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_custom_tools_still_work_end_to_end(mock_context):
    """create_skill / manage_skills, through the real dispatch."""
    mock_context._subagent_allowed_tools = None
    agent = _agent_on_registry(mock_context)
    agent.disabled_tools = set()
    assert "create_skill" in agent.available_tools and "manage_skills" in agent.available_tools
    assert "acquire_skill" not in agent.available_tools
    ts = _ts(tool_calls=[_call("manage_skills", {"action": "list"})])
    await agent._dispatch_and_process_tool_batch(ts)
    assert len(ts.tools_run_this_turn) == 1
    said = str(ts.tools_run_this_turn[0]["content"])
    assert "Unknown tool" not in said and "lesson" not in said.lower()


def test_the_manage_skills_and_introspect_descriptions_stand_alone():
    by = {d["function"]["name"]: d["function"] for d in R.TOOL_DEFINITIONS}
    assert "create_skill" in by
    assert "skill" in by["manage_skills"]["description"].lower()
    assert "lesson" not in by["manage_skills"]["description"].lower()
    assert "learning" in by["introspect"]["parameters"]["properties"]["action"]["enum"]


def test_the_population_predicates_kept_their_answers():
    """`write_origin` and friends moved; the stores that read them did not change."""
    from ghost_agent.distill.schema import Trajectory
    from ghost_agent.memory import population as P
    from ghost_agent.utils.logging import request_id_context, requester_role_context
    for rid, want in (("SYSTEM", P.ORIGIN_AUTO), ("abc12345", P.ORIGIN_USER), ("probe-abc", P.ORIGIN_PROBE)):
        tok = request_id_context.set(rid)
        try:
            assert P.write_origin() == want
        finally:
            request_id_context.reset(tok)
    t1, t2 = requester_role_context.set("member"), request_id_context.set("web-1")
    try:
        assert P.write_origin() == P.ORIGIN_MEMBER and P.usage_credit_blocked() is True
    finally:
        request_id_context.reset(t2); requester_role_context.reset(t1)
    owner = Trajectory(user_request="x", session_id="o", extra={"requester_role": "owner"})
    member = Trajectory(user_request="x", session_id="m", extra={"requester_role": "member"})
    assert [t.session_id for t in P.iter_teachable([owner, member])] == ["o"]


# ── review round: behaviour whose pin went with a lesson test ────────────────

@pytest.mark.parametrize("rid,credited", [
    ("probe-abc12345", False), ("sched-abc12345", False), ("job-abc12345", False), ("sub-abc12345", False),
    ("abc12345", True),
])
def test_only_an_owner_turn_raises_the_real_usage_counters(tmp_path, twin_store, rid, credited):
    """A probe's or an internal turn's hydration raised the counters that decide eviction on stores
    just cleaned of probe data (§4KW). Through both real counters."""
    from ghost_agent.memory.episodes import EpisodicMemory
    from ghost_agent.utils.logging import request_id_context
    (tmp_path / "ep").mkdir()
    em = EpisodicMemory(tmp_path / "ep")
    eid = em.record_episode("Bruno's walk", success=True)
    tok = request_id_context.set(rid)
    try:
        twin_store.bump_retrievals(["fact-1"])
        em._credit_surfaced_episodes([eid])
    finally:
        request_id_context.reset(tok)
    meta = twin_store.collection.get(ids=["fact-1"], include=["metadatas"])["metadatas"][0]
    assert (int(meta.get("retrieval_count", 0)) == 1) is credited
    import sqlite3
    from contextlib import closing
    with closing(sqlite3.connect(em.db_path)) as conn:
        count = conn.execute("SELECT access_count FROM episodes WHERE id=?", (eid,)).fetchone()[0]
    assert (count == 1) is credited


@pytest.mark.asyncio
async def test_a_perfect_it_reply_loses_its_tool_call_markup():
    ctx = make_context()
    ctx.llm_client.chat_completion = AsyncMock(return_value=_resp("<tool_call>x</tool_call> Use a Makefile."))
    assert await GhostAgent(ctx)._perfect_it_generate({"messages": []}, foreground=True) == "Use a Makefile."


def test_forget_matches_a_target_across_accents():
    """A name typed with its accent finds the fact stored without it, in both directions."""
    from ghost_agent.tools.memory import _target_word_coverage
    assert _target_word_coverage("Η Ελενη μένει στην πόλη", "Ελένη") == 1.0
    assert _target_word_coverage("Η Ελένη μένει στην πόλη", "Ελενη") == 1.0
    assert _target_word_coverage("Ο Μάρκος μένει στην πόλη", "Ελένη") == 0.0


@pytest.mark.asyncio
async def test_the_turn_loop_publishes_the_request_for_its_readers():
    """`current_request`: the fact ledger reads it for the owner's own words, the forget tool for
    the request a preview was made in. Read back from inside the turn, by its real reader."""
    from ghost_agent.memory.request_text import current_request
    from ghost_agent.tools.memory import _current_request_text
    tok = current_request.set("stale from an earlier turn")
    seen = []

    async def llm(*a, **k):
        seen.append(_current_request_text())
        return _resp("Dogs sleep a lot.")
    try:
        ctx = _turn_ctx(False)
        ctx.llm_client.chat_completion = AsyncMock(side_effect=llm)
        await GhostAgent(ctx).handle_chat({"messages": [{"role": "user", "content": "explain how Bruno sleeps"}],
                                           "model": "m"}, FakeBgTasks())
    finally:
        current_request.reset(tok)
    assert seen and set(seen) == {"explain how Bruno sleeps"}


def _tc(name):
    return {"function": {"name": name}}


@pytest.mark.parametrize("name", ["remember", "update_profile", "forget", "reset_all"])
def test_a_memory_write_batch_keeps_the_text_beside_it(name):
    """"Saved that." beside a memory write IS the answer, never working narration."""
    assert not A._narration_batch([_tc(name)])
    assert A._narration_batch([_tc(name), _tc("file_system")])


def test_a_knowledge_base_confirmation_is_no_evidence_for_the_verifier():
    """Bookkeeping: a short confirmation does not summon the judge; the listing it precedes does."""
    kb = {"name": "knowledge_base", "content": "OK: fact saved."}
    assert A._find_substantive_tool_for_verifier([kb]) is None
    assert A._find_substantive_tool_for_verifier([{"name": "web_search", "content": "x" * 300}, kb])["name"] == "web_search"


@pytest.mark.asyncio
async def test_a_profile_update_clears_the_loop_breakers_memory(mock_context):
    """A profile write changes what a re-read would say, so an identical read after it is no repeat."""
    mock_context._subagent_allowed_tools = None
    agent = _agent_on_registry(mock_context)
    agent.disabled_tools = set()
    agent.available_tools["update_profile"] = AsyncMock(return_value="SUCCESS: saved.")
    agent.available_tools["recall"] = AsyncMock(return_value="found Bruno")
    ts = _ts(tool_calls=[_call("update_profile", {"category": "pets", "key": "dog", "value": "Bruno"})])
    ts.seen_tools.add("hash-of-an-earlier-read")
    await agent._dispatch_and_process_tool_batch(ts)
    assert "hash-of-an-earlier-read" not in ts.seen_tools
    ts = _ts(tool_calls=[_call("recall", {"query": "Bruno"})])
    ts.seen_tools.add("hash-of-an-earlier-read")
    await agent._dispatch_and_process_tool_batch(ts)
    assert "hash-of-an-earlier-read" in ts.seen_tools     # a read clears nothing


def test_the_ambient_search_drops_a_twin_the_store_returns_anyway(twin_store):
    """The second filter of the ambient path by itself: a store that ignores `where`."""
    real = twin_store.collection

    class IgnoresWhere:
        def __getattr__(self, name):
            return getattr(real, name)

        def query(self, **kw):
            kw.pop("where", None)
            return real.query(**kw)
    twin_store.collection = IgnoresWhere()
    rows = twin_store.search_items("Bruno the dog")
    assert rows, "the search found nothing at all: the test proves nothing"
    assert not [r for r in rows if LESSON_CANARY in str(r)]


@pytest.mark.asyncio
async def test_hydration_carries_the_owners_fact_through_the_graph_tier(tmp_path):
    from ghost_agent.core.bus import MemoryBus
    from ghost_agent.memory.graph import GraphMemory
    g = GraphMemory(tmp_path)
    g.add_triplets([{"subject": "user", "predicate": "LIVES_IN", "object": "kalopetres"}])
    assert "kalopetres" in await MemoryBus(graph_memory=g).hydrate_context("where do I live?")


@pytest.mark.asyncio
@pytest.mark.parametrize("name,real", [("create_skill", "tool_create_skill"), ("manage_skills", "tool_manage_skills")])
async def test_each_custom_tool_name_reaches_its_own_function(monkeypatch, mock_context, name, real):
    called = {}
    for fn in ("tool_create_skill", "tool_manage_skills"):
        monkeypatch.setattr(R, fn, AsyncMock(side_effect=lambda *a, _fn=fn, **k: called.setdefault(_fn, True)))
    await R.get_available_tools(mock_context)[name]()
    assert list(called) == [real]


@pytest.mark.asyncio
async def test_the_record_nudge_names_the_rule_command_after_the_head():
    """"remember this: always verify the path": the model cannot save a rule, the owner's command does."""
    calls = []

    async def llm(payload, **k):
        calls.append(json.loads(json.dumps(payload.get("messages") or [], default=str)))
        return _resp("Understood, I will keep that in mind.")
    ctx = _turn_ctx(False)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=llm)
    await GhostAgent(ctx).handle_chat({"messages": [{"role": "user", "content": "remember this: always verify the path"}],
                                       "model": "m"}, FakeBgTasks())
    assert len(calls) >= 2, "the nudge never fired"
    nudge = [m for m in calls[1] if m["role"] == "user" and "REMINDER" in str(m["content"])]
    assert nudge and "learn this rule: <the rule>" in nudge[0]["content"] and "update_profile" in nudge[0]["content"]
    head = [m for m in calls[0] if m["role"] == "system"]
    assert head and "<the rule>" not in str(head)
    assert [m for m in calls[1] if m["role"] == "system"][0]["content"] == head[0]["content"]


@pytest.mark.asyncio
async def test_the_record_nudge_does_not_offer_the_command_where_it_does_not_run():
    """On a public surface the rule command is not run: the nudge must not promise it."""
    from ghost_agent.utils import logging as L
    calls = []

    async def llm(payload, **k):
        calls.append(json.loads(json.dumps(payload.get("messages") or [], default=str)))
        return _resp("Understood, I will keep that in mind.")
    ctx = _turn_ctx(False)
    ctx.llm_client.chat_completion = AsyncMock(side_effect=llm)
    with patch("ghost_agent.core.agent.reply_is_public", lambda: True):
        await GhostAgent(ctx).handle_chat({"messages": [{"role": "user", "content": "remember this: always verify the path"}],
                                           "model": "m"}, FakeBgTasks())
    assert len(calls) >= 2, "the nudge never fired"
    nudge = [m for m in calls[1] if m["role"] == "user" and "REMINDER" in str(m["content"])]
    assert nudge and "update_profile" in nudge[0]["content"]
    assert "learn this rule" not in nudge[0]["content"]


def test_the_commands_own_placeholder_is_not_stored_as_a_rule(tmp_path):
    from ghost_agent.core import owner_rules as OR
    note = OR.rule_command("learn this rule: <the rule>", tmp_path / "system")
    assert "nothing was added" in note.banner
    assert OR.rules(tmp_path / "system") == []
    assert "answer <briefly>" in OR.rule_command("learn this rule: answer <briefly> in English", tmp_path / "system").banner
