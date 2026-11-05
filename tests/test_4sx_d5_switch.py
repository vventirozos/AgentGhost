"""§4SX Phase D5: the switch — `run` advertised, `manage_services` / `jobs` as aliases mapped onto `run`, every
tool-name set wired by meaning, promoted `execute` jobs recorded as `job` runs, the briefing's RUNNING block, the
leaf's runs ended with the leaf, the Phase C operator route gone (docs/labs/PHASE_D.md D5-design)."""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from ghost_agent.core import agent as AG
from ghost_agent.core import lab_waits as W
from ghost_agent.core import request_scope as RS
from ghost_agent.memory.run_registry import RunRegistry, registry_for
from ghost_agent.tools import runs as R
from ghost_agent.tools.outcome import OutcomeStatus, ToolOutcome


def _ctx(tmp_path, pid="41f745517fe4", conv="conv1"):
    mem = tmp_path / "memory"
    mem.mkdir(exist_ok=True)
    return SimpleNamespace(args=SimpleNamespace(labs_host="", labs_key_file=""), memory_dir=mem,
                           sandbox_dir=tmp_path, sandbox_manager=None, tor_proxy=None,
                           current_project_id=pid, conversation_key=conv)


@pytest.fixture
def scope():
    toks = []

    def _set(ctx, origin="owner"):
        toks.append(RS.install(RS.RequestScope(conversation_key=ctx.conversation_key,
                                               project_id=ctx.current_project_id, origin=origin, context=ctx)))
    yield _set
    for t in reversed(toks):
        RS.reset(t)


def _row(reg, name, owner, *, lifetime="server", backend="services", ref="k", origin="owner", meta=None):
    rid = reg.add(name=name, place="workspace", lifetime=lifetime, command="x", port=8101, owner=owner,
                  origin=origin, backend=backend, backend_ref=ref)
    if meta:
        reg.set_meta(rid, **meta)
    return rid


# ------------------------------------------------------------------ schema + dispatch

def test_run_is_in_the_schema_and_the_old_tools_are_not():
    from ghost_agent.tools.registry import TOOL_DEFINITIONS
    names = [d["function"]["name"] for d in TOOL_DEFINITIONS]
    assert names.count("run") == 1 and "manage_services" not in names and "jobs" not in names
    d = next(d for d in TOOL_DEFINITIONS if d["function"]["name"] == "run")
    assert d["function"]["parameters"]["properties"]["lifetime"]["enum"] == ["job", "server"]


async def test_the_registry_dispatches_run_and_the_aliases_with_the_context(monkeypatch):
    from ghost_agent.tools import registry as REG
    seen = []

    async def fake(name):
        async def f(context=None, **kw):
            seen.append((name, context, kw))
            return "ok"
        return f
    monkeypatch.setattr(REG, "tool_run", await fake("run"))
    monkeypatch.setattr(REG, "manage_services_alias", await fake("ms"))
    monkeypatch.setattr(REG, "jobs_alias", await fake("jobs"))
    ctx = SimpleNamespace(args=SimpleNamespace(), sandbox_manager=None, memory_dir=None, llm_client=None,
                          scratchpad=None, profile_memory=None, memory_system=None, tor_proxy=None,
                          current_project_id=None, scheduler=None, skill_memory=None)
    tools = REG.get_available_tools(ctx)
    await tools["run"](action="status")
    await tools["manage_services"](action="status")
    await tools["jobs"](action="status")
    assert [(n, c is ctx) for n, c, _ in seen] == [("run", True), ("ms", True), ("jobs", True)]


def test_the_leaf_gets_run():
    from ghost_agent.core import coding_loop as CL
    assert "run" in CL.LEAF_ALLOWED_TOOLS


def test_the_phase_c_route_is_removed():
    from ghost_agent.api import routes as RT
    assert not [r for r in RT.router.routes if "run-probe" in getattr(r, "path", "")]


# ------------------------------------------------------------------ name sets, by meaning

@pytest.mark.parametrize("action,read", [("status", True), ("logs", True), ("url", True), ("collect", True),
                                         ("start", False), ("stop", False), ("restart", False)])
def test_run_reads_and_changes(action, read):
    assert R.is_read_call({"action": action}) is read
    assert R.is_read_call(f'{{"action": "{action}"}}') is read
    assert AG._call_mutated_world("run", {"action": action}, True) is (not read)
    assert AG._starts_new_work("run", {"action": action}) is (not read)


def test_a_dropped_run_read_is_not_a_dropped_change():
    calls = [{"function": {"name": "run", "arguments": '{"action": "status"}'}},
             {"function": {"name": "run", "arguments": '{"action": "stop", "name": "api"}'}}]
    assert AG._dropped_mutating_names(calls) == ["run"]
    note = AG._dropped_mutation_note(["run"])
    assert note and "run" in note and "written" not in note.split("Only the results")[0]


def test_the_sets_hold_run_by_meaning():
    from ghost_agent.core.claim_binding import SOURCE_EXCLUDED_TOOLS
    from ghost_agent.core.evidence_gate import EVIDENCE_TOOLS
    from ghost_agent.core.isolation import REPLAY_FORBIDDEN_TOOLS
    from ghost_agent.core.strikes import READWRITE_LOOP_TOOLS
    from ghost_agent.core.task_state import EVIDENCE_TOOLS as TS_EVIDENCE
    from ghost_agent.memory.egress import CODE_TOOLS, OUTBOUND_TOOLS
    from ghost_agent.tools.composed_skills import MACRO_FORBIDDEN_STEP_TOOLS
    assert "run" in REPLAY_FORBIDDEN_TOOLS and "run" in MACRO_FORBIDDEN_STEP_TOOLS
    assert OUTBOUND_TOOLS["run"] == "url_literals" and "run" in CODE_TOOLS
    assert "run" in SOURCE_EXCLUDED_TOOLS and "run" in EVIDENCE_TOOLS and "run" in TS_EVIDENCE
    assert "run" in READWRITE_LOOP_TOOLS
    for s in (AG._EDIT_CHURN_VERIFY_TOOLS, AG._EXTERNAL_EVIDENCE_TOOLS, AG._WORK_TOOLS, AG._BATCH_WALL_TOOLS,
              AG._MUTATING_TOOLS_FOR_DROP_NOTE):
        assert "run" in s


_COLLECT_OK = "[run bench exited exit=0]\nEXIT CODE: 0\n3 passed"
_COLLECT_FAIL = "[run bench exited exit=1]\nEXIT CODE: 1\nTraceback (most recent call last):\nValueError: x"
_LOGS = "[run api logs]\nERROR: relation users does not exist\n"


def test_only_a_collect_is_program_output_and_its_exit_decides():
    from ghost_agent.core.strikes import is_program_output
    from ghost_agent.distill.outcome_heuristics import _run_result_failed
    assert is_program_output("run", _COLLECT_OK) and is_program_output("run", _COLLECT_FAIL)
    assert not is_program_output("run", _LOGS)
    assert is_program_output("run", "[FAILURE BANNER] EXIT CODE: 1\n" + _COLLECT_FAIL)
    assert _run_result_failed(_COLLECT_FAIL) and not _run_result_failed(_COLLECT_OK)
    assert not _run_result_failed(_LOGS)                       # a log's ERROR is its subject
    assert not _run_result_failed("[run b exited]\nEXIT CODE: none\n")
    assert _run_result_failed("RUN ERROR: no run named 'x' here")


def test_the_evidence_gate_reads_a_run_by_its_own_head():
    from ghost_agent.core.evidence_gate import _empty_reason
    assert _empty_reason("run", {}, _COLLECT_FAIL, False) == "run: error"
    assert _empty_reason("run", {}, _COLLECT_OK, False) is None
    assert _empty_reason("run", {}, _LOGS, False) is None
    assert _empty_reason("run", {}, "[run bench — running] no result yet; collect it later", False) == \
        "run: no answer"


def test_a_runs_output_is_outside_content_its_records_are_not():
    from ghost_agent.utils.provenance import is_untrusted_source
    assert is_untrusted_source("run", {"action": "logs"}) and is_untrusted_source("run", {"action": "collect"})
    assert not is_untrusted_source("run", {"action": "status"}) and not is_untrusted_source("run", {"action": "url"})


# ------------------------------------------------------------------ batch order

@pytest.mark.parametrize("name,args,prev,pargs,want", [
    ("run", {"action": "start", "name": "api", "where": "workspace"}, "execute", {"command": "make"}, True),
    ("execute", {"command": "curl $URL"}, "run", {"action": "start", "name": "api"}, True),
    ("execute", {"command": "ls"}, "run", {"action": "status"}, False),
    ("run", {"action": "logs", "name": "api"}, "run", {"action": "start", "name": "api"}, True),
    ("run", {"action": "logs", "name": "db"}, "run", {"action": "start", "name": "api"}, False),
    ("run", {"action": "status"}, "run", {"action": "start", "name": "api"}, True),
    ("run", {"action": "start", "name": "a", "where": "lab_0000aa01/web"}, "lab",
     {"action": "put", "lab_id": "lab_0000aa01"}, True),
    ("run", {"action": "start", "name": "a", "where": "lab_0000aa01/web"}, "lab",
     {"action": "put", "lab_id": "lab_0000bb02"}, False),
    ("run", {"action": "stop", "name": "shop"}, "lab", {"action": "switch", "lab_id": "lab_0000aa01"}, True),
    ("lab", {"action": "destroy", "lab_id": "lab_0000aa01"}, "run",
     {"action": "start", "name": "a", "where": "lab_0000aa01/web"}, True),
    ("lab", {"action": "exec", "lab_id": "lab_0000bb02"}, "run",
     {"action": "start", "name": "a", "where": "lab_0000aa01/web"}, False),
    ("lab", {"action": "switch", "lab_id": "lab_0000bb02"}, "run", {"action": "stop", "name": "shop"}, True),
    ("lab", {"action": "destroy", "lab_id": "lab_0000aa01"}, "run", {"action": "logs", "name": "a"}, False),
])
def test_a_run_waits_for_what_it_depends_on(name, args, prev, pargs, want):
    assert AG._batch_call_depends(name, args, prev, pargs) is want


async def test_the_batch_wires_runs_into_the_waits():
    order = []

    async def step(tag, d=0.0):
        await asyncio.sleep(d)
        order.append(tag)
    meta = [("execute", 0, 0, 0, 0, 0, 0, 0, 0, {"command": "make"}),
            ("run", 0, 0, 0, 0, 0, 0, 0, 0, {"action": "start", "name": "api", "where": "workspace"})]
    coros = [(0, step("build", 0.05)), (1, step("start"))]
    wired = AG._wire_batch_waits(coros, meta)
    await asyncio.gather(*(c for _, c in wired))
    assert order == ["build", "start"]


# ------------------------------------------------------------------ the aliases

async def test_manage_services_start_is_a_workspace_server_run(monkeypatch):
    seen = []

    async def fake_run(context=None, **kw):
        seen.append(kw)
        return ToolOutcome("[run api started]", status=OutcomeStatus.OK, world_changed=True)
    monkeypatch.setattr(R, "tool_run", fake_run)
    out = await R.manage_services_alias(context=None, action="start", service="My_API", cmd="node s.js",
                                        port=8104, workdir="app")
    assert seen == [{"action": "start", "name": "my-api", "command": "cd app && node s.js", "where": "workspace",
                     "lifetime": "server", "port": 8104}]
    assert out.status is OutcomeStatus.OK and out.world_changed is True
    assert "`manage_services` is now `run`" in out and "'My_API' → 'my-api'" in out
    seen.clear()
    await R.manage_services_alias(context=None, action="logs", name="api", tail=20)
    await R.manage_services_alias(context=None, action="list")
    assert seen == [{"action": "logs", "name": "api", "lines": 20}, {"action": "status"}]


@pytest.mark.parametrize("action", ["stop-all", "stop_all", "adopt", "purge"])
async def test_manage_services_actions_with_no_run_twin_are_refused(action):
    out = await R.manage_services_alias(context=None, action=action)
    assert out.status is OutcomeStatus.REJECTED and "`run`" in out


async def test_a_name_that_cannot_be_a_run_name_is_refused():
    out = await R.manage_services_alias(context=None, action="stop", name="___")
    assert out.status is OutcomeStatus.REJECTED and "not a run name" in out


async def test_jobs_maps_onto_the_run_row_of_that_job(tmp_path, scope, monkeypatch):
    ctx = _ctx(tmp_path)
    scope(ctx)
    reg = registry_for(ctx)
    _row(reg, "job-1a2b3c4d", f"project:{ctx.current_project_id}", lifetime="job", backend="jobs",
         ref="job-1a2b3c4d")
    seen = []

    async def fake_run(context=None, **kw):
        seen.append(kw)
        return ToolOutcome("r", status=OutcomeStatus.OK, world_changed=False)
    monkeypatch.setattr(R, "tool_run", fake_run)
    await R.jobs_alias(context=ctx, action="collect", job_id="job-1a2b3c4d")
    await R.jobs_alias(context=ctx, action="cancel", job_id="job-1a2b3c4d")
    await R.jobs_alias(context=ctx, action="status")
    assert seen == [{"action": "collect", "name": "job-1a2b3c4d"}, {"action": "stop", "name": "job-1a2b3c4d"},
                    {"action": "status"}]
    old = []

    async def fake_jobs(action=None, job_id=None, context=None, **kw):
        old.append((action, job_id))
        return "old jobs"
    import ghost_agent.tools.delegate as D
    monkeypatch.setattr(D, "tool_jobs", fake_jobs)
    assert await R.jobs_alias(context=ctx, action="collect", job_id="job-99999999") == "old jobs"
    assert old == [("collect", "job-99999999")]                     # promoted before the switch: still read
    out = await R.jobs_alias(context=ctx, action="spawn")
    assert out.status is OutcomeStatus.REJECTED


# ------------------------------------------------------------------ promotion

def test_a_promoted_job_is_a_job_run_named_after_it(tmp_path, scope):
    ctx = _ctx(tmp_path)
    scope(ctx)
    job = {"id": "job-1a2b3c4d", "command": "sleep 900"}
    assert R.record_promoted_job(ctx, job) == "job-1a2b3c4d"
    reg = registry_for(ctx)
    rows = reg.rows(f"project:{ctx.current_project_id}", live_only=True)
    assert [(r["name"], r["lifetime"], r["backend"], r["backend_ref"], r["origin"]) for r in rows] == \
        [("job-1a2b3c4d", "job", "jobs", "job-1a2b3c4d", "owner")]
    assert R.record_promoted_job(ctx, {**job, "duplicate": True}) == "job-1a2b3c4d"     # no second row
    assert len(reg.rows(live_only=True)) == 1
    assert R.record_promoted_job(ctx, {"id": "job-55555555", "duplicate": True}) == ""
    assert R.record_promoted_job(None, job) == "" and R.record_promoted_job(ctx, {"id": "BAD ID"}) == ""
    assert R.run_name_for_job(ctx, "job-1a2b3c4d") == "job-1a2b3c4d" and R.run_name_for_job(ctx, "job-0") == ""


def test_a_members_promoted_job_records_no_run(tmp_path, scope):
    ctx = _ctx(tmp_path)
    scope(ctx, origin="member")
    assert R.record_promoted_job(ctx, {"id": "job-1a2b3c4d"}) == ""


def test_the_banner_names_run_when_the_job_is_a_run():
    from ghost_agent.tools.execute import _promotion_banner, _promoted_result
    job = {"id": "job-1a2b3c4d", "log": "/workspace/.jobs/x.log"}
    b = _promotion_banner(job, 95, "job-1a2b3c4d")
    assert "run(action='collect', name='job-1a2b3c4d')" in b and "jobs(" not in b
    assert "jobs(action='collect', job_id='job-1a2b3c4d')" in _promotion_banner(job, 95)
    r = _promoted_result(job, 95, "", "job-1a2b3c4d")
    assert r.status is OutcomeStatus.UNRESOLVED and "run(action='status', name='job-1a2b3c4d')" in r


async def test_execute_gets_the_context_to_record_the_run(tmp_path, monkeypatch):
    """Behaviour: the registry's execute entry hands its context to tool_execute; R1 enumeration: both promotion
    sites in execute.py record the run from that context."""
    import ast
    from ghost_agent.tools import registry as REG
    seen = {}

    async def fake_exec(**kw):
        seen.update(kw)
        return "ok"
    monkeypatch.setattr(REG, "tool_execute", fake_exec)
    ctx = SimpleNamespace(args=SimpleNamespace(labs_host="", labs_key_file=""), sandbox_manager=None,
                          memory_dir=tmp_path, llm_client=None, scratchpad=None, profile_memory=None,
                          memory_system=None, tor_proxy=None, current_project_id=None, scheduler=None,
                          skill_memory=None, sandbox_dir=tmp_path)
    tools = REG.get_available_tools(ctx)
    await tools["execute"](command="echo hi")
    assert seen.get("run_context") is ctx
    tree = ast.parse(Path(AG.__file__).parent.parent.joinpath("tools", "execute.py").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", getattr(n.func, "attr", "")) == "record_promoted_job"]
    assert len(calls) == 2
    for n in calls:
        a0 = n.args[0]
        assert isinstance(a0, ast.Call) and ast.unparse(a0) == "kwargs.get('run_context')"


def test_the_landed_job_hint_names_run_collect(tmp_path, scope):
    from ghost_agent.main import _collect_hint
    ctx = _ctx(tmp_path)
    scope(ctx)
    R.record_promoted_job(ctx, {"id": "job-1a2b3c4d"})
    assert _collect_hint({"id": "job-1a2b3c4d"}, ctx) == "run(action='collect', name='job-1a2b3c4d')"
    assert _collect_hint({"id": "job-00000000"}, ctx) == "jobs(action='collect')"
    assert _collect_hint({"started_via": "run", "label": "bench"}) == "run(action='collect', name='bench')"


# ------------------------------------------------------------------ briefing

def test_the_briefing_lists_runs_and_labs_and_a_run_backed_service_once(tmp_path):
    from ghost_agent.core.prompts import build_project_briefing
    from ghost_agent.memory.lab_inventory import inventory_for
    from ghost_agent.memory.projects import ProjectStore
    store = ProjectStore(tmp_path / "ps", sandbox_root=tmp_path / "sb")
    pid = store.create_project("Shop")
    ctx = _ctx(tmp_path, pid=pid)
    reg = registry_for(ctx)
    _row(reg, "api", f"project:{pid}", ref="svc-key-1")
    inventory_for(ctx).record_create("lab_0000aa01", owner=f"project:{pid}", origin="owner", template="postgres",
                                     label="db")
    running = R.project_running_summary(ctx, pid)
    assert [(r["kind"], r["name"]) for r in running] == [("run", "api"), ("lab", "lab_0000aa01")]
    services = [{"name": "api", "key": "svc-key-1", "port": 8101}, {"name": "legacy", "key": "old", "port": 8102}]
    text = build_project_briefing(store, pid, services=services, running=running)
    assert "RUNNING (" in text and "run api · server · workspace · port 8101 · running (checked " in text
    assert "lab lab_0000aa01 · postgres · db" in text
    assert "  - legacy" in text and "  - api · port" not in text           # the run-backed service: once
    assert "RUNNING" not in build_project_briefing(store, pid, services=None, running=None)


def test_the_running_summary_reads_no_backend(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    monkeypatch.setattr(R, "_probe", lambda *a, **k: (_ for _ in ()).throw(AssertionError("probed")))
    monkeypatch.setattr(R, "_supervisors", lambda *a, **k: (_ for _ in ()).throw(AssertionError("docker")))
    assert R.project_running_summary(ctx, ctx.current_project_id) == []
    assert R.project_running_summary(ctx, "") == []


# ------------------------------------------------------------------ the leaf's runs and kept apps

async def test_the_leaf_stops_only_its_own_runs(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    reg = registry_for(ctx)
    own = f"project:{ctx.current_project_id}"
    _row(reg, "mine", own, origin="autopilot", meta={"leaf": "t1-a1"})
    _row(reg, "other", own, origin="autopilot", meta={"leaf": "t2-a1"})
    _row(reg, "owners", own)
    _row(reg, "shop", own, lifetime="app", backend="labs_app", ref="p/shop", meta={"leaf": "t1-a1"})
    stopped = []

    async def fake_stop(context, reg_, row, origin):
        stopped.append((row["name"], origin))
        return ToolOutcome("stopped", status=OutcomeStatus.OK)
    monkeypatch.setattr(R, "_stop", fake_stop)
    assert await R.stop_leaf_runs(ctx, own, "t1-a1") == ["mine"]
    assert stopped == [("mine", "autopilot")]
    assert await R.stop_leaf_runs(ctx, own, "") == []


async def test_the_leaf_turn_ends_its_runs_before_its_labs(monkeypatch):
    from ghost_agent.core import coding_loop as CL
    from ghost_agent.tools import labs as L
    order = []

    async def runs_(c, owner, leaf):
        order.append(("runs", owner, leaf))
        return []

    async def labs_(c, owner, leaf):
        order.append(("labs", owner, leaf))
        return []
    monkeypatch.setattr(R, "stop_leaf_runs", runs_)
    monkeypatch.setattr(L, "destroy_leaf_labs", labs_)

    class _Boom:
        def __init__(self, iso):
            self.available_tools, self.disabled_tools = {}, set()

        async def handle_chat(self, *a, **k):
            raise RuntimeError("boom")
    import ghost_agent.core.agent as A
    monkeypatch.setattr(A, "GhostAgent", _Boom)
    monkeypatch.setattr(CL, "build_leaf_context", lambda c, **k: SimpleNamespace(args=SimpleNamespace(model="m")))
    with pytest.raises(RuntimeError):
        await CL.run_leaf_turn(SimpleNamespace(), leaf_id="t1-a1", prompt="p", is_background=False, project_id="p1")
    assert [o[0] for o in order] == ["runs", "labs"] and order[0][1:] == ("project:p1", "t1-a1")


@pytest.mark.parametrize("action", ["start", "stop", "restart"])
async def test_unattended_work_never_starts_or_stops_a_kept_app(tmp_path, scope, action):
    ctx = _ctx(tmp_path)
    scope(ctx, origin="autopilot")
    reg = registry_for(ctx)
    _row(reg, "shop", f"project:{ctx.current_project_id}", lifetime="app", backend="labs_app", ref="p/shop")
    out = await R.tool_run(context=ctx, action=action, name="shop")
    assert out.status is OutcomeStatus.REJECTED and "production" in out


def test_a_live_run_outcome_is_judged_by_its_declared_status():
    """r41/r3: a live ToolOutcome's own status decides before any prose."""
    from types import SimpleNamespace as NS
    from ghost_agent.distill.outcome_heuristics import _run_result_failed

    class _R(str):
        pass
    bad = _R(_COLLECT_OK); bad.status = NS(value="failed"); bad.declared = True
    good = _R(_COLLECT_FAIL); good.status = NS(value="ok"); good.declared = True
    undeclared = _R(_COLLECT_FAIL); undeclared.status = NS(value="ok"); undeclared.declared = False
    assert _run_result_failed(bad) and not _run_result_failed(good)
    assert _run_result_failed(undeclared)                      # undeclared → the stored-row prose rule
