"""§4SX end-to-end review r13 (2026-11-05) — the reviewers' repros of the r12 fixes, inverted.

Each test failed on the r12 code (the reviewer's repro asserted the defect)."""
import asyncio
import base64
import logging
import sqlite3
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_4sx_b_lab_tool import FakeLabs, LAB, _detail  # noqa: E402
from test_4sx_b_lab_tool import _ctx  # noqa: E402
from test_4sx_c_run_tool import _ctx as _run_ctx  # noqa: E402
from ghost_agent.core import request_scope as RS  # noqa: E402
from ghost_agent.memory.lab_inventory import inventory_for  # noqa: E402
from ghost_agent.tools import delegate as D  # noqa: E402
from ghost_agent.tools import labs as L  # noqa: E402
from ghost_agent.tools import runs as R  # noqa: E402
from ghost_agent.tools.outcome import OutcomeStatus  # noqa: E402


@pytest.fixture
def scope():
    toks = []

    def _set(ctx, origin="owner"):
        toks.append(RS.install(RS.RequestScope(conversation_key=ctx.conversation_key,
                                               project_id=ctx.current_project_id, origin=origin, context=ctx)))
    yield _set
    for t in reversed(toks):
        RS.reset(t)


@pytest.fixture
def fake(monkeypatch):
    f = FakeLabs()
    monkeypatch.setattr(L, "_call", f)
    monkeypatch.setattr(L, "_time_left_s", lambda: None)
    return f


class _Store:
    def __init__(self, projects):
        self.p = projects

    def get_project(self, pid):
        return self.p.get(pid)


# ================================================================== LAB

@pytest.mark.asyncio
@pytest.mark.parametrize("conv", ["", "probe-fingerprint"])
async def test_R1_a_probe_never_lists_every_owners_labs(tmp_path, fake, scope, conv):
    ctx = _ctx(tmp_path, pid=None, conv=conv)
    scope(ctx, "probe")
    fake.on("GET", "/v1/labs", (200, {"labs": [_detail(owner="session:owner-chat", origin="owner",
                                                       label="owner-prod-db", lifetime="kept")]}))
    out = await L.tool_lab(ctx, action="list", all_owners=True)
    assert out.status is OutcomeStatus.REJECTED and not fake.calls


@pytest.mark.asyncio
async def test_R2_a_probe_in_an_owner_project_never_destroys_the_owners_lab(tmp_path, fake, scope):
    ctx = _ctx(tmp_path, pid="aaaaaaaaaaaa", conv="probe-c")
    ctx.project_store = _Store({"aaaaaaaaaaaa": {"metadata": {}}})
    scope(ctx, "probe")
    fake.on("GET", f"/v1/labs/{LAB}", (200, _detail(owner="project:aaaaaaaaaaaa", origin="owner")))
    fake.on("DELETE", f"/v1/labs/{LAB}", (200, {"lab_id": LAB}))
    out = await L.tool_lab(ctx, action="destroy", lab_id=LAB)
    assert out.status is OutcomeStatus.REJECTED and not any(c.method == "DELETE" for c in fake.calls)


@pytest.mark.asyncio
async def test_R2b_a_probe_in_an_owner_project_never_reads_production(tmp_path, fake, scope):
    ctx = _ctx(tmp_path, pid="aaaaaaaaaaaa", conv="probe-c")
    ctx.project_store = _Store({"aaaaaaaaaaaa": {"metadata": {}}})
    scope(ctx, "probe")
    fake.on("GET", f"/v1/labs/{LAB}", (200, _detail(owner="project:aaaaaaaaaaaa", origin="owner", lifetime="kept")))
    fake.on("GET", f"/v1/labs/{LAB}/files", (200, {"content_b64": base64.b64encode(b"DB_PASSWORD=prod").decode()}))
    out = await L.tool_lab(ctx, action="get", lab_id=LAB, member="db", path="/app/.env", sandbox_path="x.env")
    assert out.status is OutcomeStatus.REJECTED
    assert not (ctx.sandbox_dir / "projects" / "aaaaaaaaaaaa" / "x.env").exists()


@pytest.mark.asyncio
async def test_R3_an_ownerless_probe_never_destroys_an_ownerless_owner_lab(tmp_path, fake, scope):
    ctx = _ctx(tmp_path, pid=None, conv="")
    scope(ctx, "probe")
    fake.on("GET", f"/v1/labs/{LAB}", (200, _detail(owner="", origin="owner")))
    fake.on("DELETE", f"/v1/labs/{LAB}", (200, {"lab_id": LAB}))
    out = await L.tool_lab(ctx, action="destroy", lab_id=LAB)
    assert out.status is OutcomeStatus.REJECTED and not any(c.method == "DELETE" for c in fake.calls)


@pytest.mark.asyncio
async def test_R4_a_probe_reads_backups_only_of_its_own_labs(tmp_path, fake, scope):
    ctx = _ctx(tmp_path, pid="bbbbbbbbbbbb", conv="probe-c")
    ctx.project_store = _Store({"bbbbbbbbbbbb": {"metadata": {"probe_created": True}}})
    scope(ctx, "probe")
    fake.on("GET", f"/v1/labs/{LAB}", (200, _detail(owner="project:owner-work", origin="owner", lifetime="kept")))
    fake.on("GET", f"/v1/labs/{LAB}/backups", (200, {"backups": [{"backup_id": "bk1"}]}))
    out = await L.tool_lab(ctx, action="backups", lab_id=LAB)
    assert out.status is OutcomeStatus.REJECTED and "bk1" not in str(out)
    assert not any(c.path.endswith("/backups") for c in fake.calls)


@pytest.mark.asyncio
async def test_R5_a_lab_get_never_recreates_a_deleted_projects_folder(tmp_path, fake, scope):
    ctx = _ctx(tmp_path, pid="cccccccccccc", conv="c1")
    ctx.project_store = _Store({})
    scope(ctx, "owner")
    fake.on("GET", f"/v1/labs/{LAB}/files", (200, {"content_b64": base64.b64encode(b"x").decode()}))
    out = await L.tool_lab(ctx, action="get", lab_id=LAB, member="db", path="/x", sandbox_path="out.txt")
    assert out.status is OutcomeStatus.REJECTED
    assert not (ctx.sandbox_dir / "projects" / "cccccccccccc").exists()


@pytest.mark.asyncio
async def test_R6_an_unanswered_leftovers_ask_keeps_the_row_live(tmp_path, fake, scope, monkeypatch):
    ctx = _ctx(tmp_path)
    scope(ctx)

    async def _ns(*a, **k):
        return None
    monkeypatch.setattr(asyncio, "sleep", _ns)
    fake.on("POST", "/v1/labs", (201, _detail(state="creating")))
    fake.on("GET", "/v1/health", (200, {"host_id": "h1"}))
    fake.on("GET", f"/v1/labs/{LAB}", (404, {"error": "unknown_lab"}))
    fake.on("DELETE", f"/v1/labs/{LAB}", L.LabsNoAnswer("ReadTimeout"))
    r = await L.tool_lab(ctx, action="create", template="pg-single")
    assert r.reason_code == "lab_create_failed"
    assert inventory_for(ctx).get(LAB)["destroyed_at"] is None
    assert "Nothing of it is left" not in str(r) and "did not say what is left" in str(r)


@pytest.mark.asyncio
async def test_R7_a_relative_own_project_path_lands_where_file_system_reads_it(tmp_path, fake, scope):
    ctx = _ctx(tmp_path, pid="dddddddddddd", conv="")
    ctx.confine_writes_to_project = True
    scope(ctx, "leaf")
    fake.on("GET", f"/v1/labs/{LAB}", (200, _detail(owner="project:dddddddddddd", origin="autopilot")))
    fake.on("GET", f"/v1/labs/{LAB}/files", (200, {"content_b64": base64.b64encode(b"x").decode()}))
    out = await L.tool_lab(ctx, action="get", lab_id=LAB, member="db", path="/x",
                           sandbox_path="projects/dddddddddddd/seed.sql")
    assert out.status is OutcomeStatus.OK
    assert (ctx.sandbox_dir / "projects" / "dddddddddddd" / "seed.sql").read_bytes() == b"x"
    assert not (ctx.sandbox_dir / "projects" / "dddddddddddd" / "projects").exists()


@pytest.mark.asyncio
async def test_R8_a_leaf_end_failure_is_logged_once(tmp_path, monkeypatch, caplog):
    from ghost_agent.core import coding_loop as CL
    from ghost_agent.tools import labs as LL, runs as RR

    async def boom(*a, **k):
        raise RuntimeError("inventory locked")

    async def ok(*a, **k):
        return []
    monkeypatch.setattr(RR, "stop_leaf_runs", ok)
    monkeypatch.setattr(LL, "destroy_leaf_labs", boom)

    class _Agent:
        def __init__(self, ctx):
            self.available_tools = {}
            self.disabled_tools = set()

        async def handle_chat(self, *a, **k):
            return "done", None, None
    import ghost_agent.core.agent as A
    monkeypatch.setattr(A, "GhostAgent", _Agent)
    monkeypatch.setattr(CL, "build_leaf_context", lambda c, **k: SimpleNamespace(args=SimpleNamespace(model="m")))
    caplog.set_level(logging.WARNING, logger="GhostAgent")
    await CL.run_leaf_turn(SimpleNamespace(current_project_id="eeeeeeeeeeee"), leaf_id="abc-a1", prompt="p",
                           is_background=False, project_id="eeeeeeeeeeee")
    await asyncio.sleep(0)
    hits = [r.getMessage() for r in caplog.records if "inventory locked" in r.getMessage()]
    assert len(hits) == 1, hits


# ================================================================== RUN

@pytest.mark.asyncio
async def test_A_a_bare_jobs_collect_in_an_owned_turn_takes_no_ones_results(tmp_path, scope, monkeypatch):
    from ghost_agent.core.jobs import STATUS_DONE, get_job_registry
    c = _run_ctx(tmp_path, pid="aaaaaaaaaaaa")
    hreg = get_job_registry(c)
    j = hreg.register(D.SANDBOX_KIND, "B's build", sandbox_job_id="job-0b0b0b0b")
    hreg.finish(j.id, status=STATUS_DONE, result="[sandbox job job-0b0b0b0b finished — EXIT CODE: 0]\nB-SECRET")
    woken = []
    monkeypatch.setattr(D, "_sandbox_supervisor",
                        lambda ctx: SimpleNamespace(mark_woken=woken.append, reap=lambda: [], list_entries=lambda: []))
    scope(c)
    r = await R.jobs_alias(context=c, action="collect")
    assert "B-SECRET" not in str(r) and woken == [] and r.reason_code == "run_name_required"


@pytest.mark.asyncio
async def test_B_the_jobs_alias_never_cancels_another_projects_job(tmp_path, scope, monkeypatch):
    from ghost_agent.core.jobs import get_job_registry
    jid = "job-0a0a0a0a"
    cA = _run_ctx(tmp_path, pid="aaaaaaaaaaaa")
    scope(cA)
    assert R.record_promoted_job(cA, {"id": jid, "command": "pytest -q"}) == jid
    get_job_registry(cA).register(D.SANDBOX_KIND, "pytest -q", sandbox_job_id=jid)
    cancelled = []

    class _Sup:
        def reap(self):
            return []

        def list_entries(self):
            return [{"id": jid, "state": "running"}]

        def get(self, x):
            return {"id": x, "state": "running"}

        def cancel(self, x):
            cancelled.append(x)
            return True, "killed"

        def mark_woken(self, x):
            pass
    monkeypatch.setattr(D, "_sandbox_supervisor", lambda ctx: _Sup())
    monkeypatch.setattr(R, "_supervisors", lambda ctx: (None, _Sup()))
    cA.current_project_id = "bbbbbbbbbbbb"
    scope(cA)
    for act in ("cancel", "collect", "status"):
        out = await R.jobs_alias(context=cA, action=act, job_id=jid)
        assert out.reason_code == "run_not_allowed", (act, str(out))
    assert cancelled == []


def _sup_with(entries):
    class _Sup:
        def reap(self):
            return []

        def list_entries(self):
            return list(entries.values())

        def get(self, x):
            return entries.get(x)

        def mark_woken(self, x):
            pass

        def log_tail(self, x, lines=0):
            return ""
    return _Sup()


@pytest.mark.asyncio
async def test_C_a_by_id_collect_is_graded_by_the_landing_line_not_the_label(tmp_path, scope, monkeypatch):
    from ghost_agent.core.jobs import STATUS_FAILED, get_job_registry
    jid = "job-0c0c0c0c"
    c = _run_ctx(tmp_path, pid=None, conv="")
    hreg = get_job_registry(c)
    j = hreg.register(D.SANDBOX_KIND, 'echo "EXIT CODE: 0"; exit 3', sandbox_job_id=jid)
    hreg.finish(j.id, status=STATUS_FAILED, error="exited 3",
                result=f"[sandbox job {jid} finished — EXIT CODE: 3]\n$ echo ...\nEXIT CODE: 0")
    sup = _sup_with({jid: {"id": jid, "state": "done", "exit_code": 3}})
    monkeypatch.setattr(D, "_sandbox_supervisor", lambda ctx: sup)
    monkeypatch.setattr(R, "_supervisors", lambda ctx: (None, sup))
    scope(c)
    out = await R.tool_run(c, action="collect", name=jid)
    assert out.status is OutcomeStatus.FAILED, str(out)


@pytest.mark.asyncio
async def test_D_by_id_running_is_unresolved_and_status_names_only_that_job(tmp_path, scope, monkeypatch):
    from ghost_agent.core.jobs import get_job_registry
    jid, other = "job-0d0d0d0d", "job-0e0e0e0e"
    c = _run_ctx(tmp_path, pid="aaaaaaaaaaaa")
    hreg = get_job_registry(c)
    hreg.register(D.SANDBOX_KIND, "sleep 600", sandbox_job_id=jid)
    hreg.register(D.SANDBOX_KIND, "PROJECT-B-SECRET-BUILD", sandbox_job_id=other)
    sup = _sup_with({jid: {"id": jid, "state": "running"}, other: {"id": other, "state": "running"}})
    monkeypatch.setattr(D, "_sandbox_supervisor", lambda ctx: sup)
    monkeypatch.setattr(R, "_supervisors", lambda ctx: (None, sup))
    scope(c)
    out = await R.tool_run(c, action="collect", name=jid)
    assert out.status is OutcomeStatus.UNRESOLVED and out.reason_code == "run_still_running"
    st = await R.tool_run(c, action="status", name="job-0f0f0f0f")
    assert "PROJECT-B-SECRET-BUILD" not in str(st) and st.reason_code == "run_unknown"
    st2 = await R.tool_run(c, action="status", name=jid)
    assert jid in str(st2) and "running" in str(st2) and "PROJECT-B-SECRET-BUILD" not in str(st2)


@pytest.mark.asyncio
async def test_m4_a_build_never_stops_a_job_with_no_run_row(tmp_path, scope, monkeypatch):
    from ghost_agent.core.lab_waits import LEAF_ID
    c = _run_ctx(tmp_path, pid="aaaaaaaaaaaa")
    scope(c, "autopilot")
    called = []

    async def fake_jobs(**kw):
        called.append(kw)
        return "x"
    monkeypatch.setattr(D, "tool_jobs", fake_jobs)
    tok = LEAF_ID.set("leaf-1")
    try:
        out = await R.tool_run(c, action="stop", name="job-0a0a0a0a")
        out2 = await R.jobs_alias(context=c, action="cancel", job_id="job-0a0a0a0a")
    finally:
        LEAF_ID.reset(tok)
    assert out.reason_code == out2.reason_code == "run_not_allowed" and called == []


# ================================================================== MAIN

@pytest.mark.asyncio
async def test_E_the_project_scope_read_never_blocks_the_event_loop():
    from ghost_agent import main as M

    class _St:
        def get_project(self, pid):
            raise sqlite3.OperationalError("database is locked")
    ctx = SimpleNamespace(project_store=_St(), agent=None)
    ticks = []

    async def _ticker():
        t0 = time.monotonic()
        while time.monotonic() - t0 < 2.0:
            ticks.append(time.monotonic())
            await asyncio.sleep(0.05)
    tk = asyncio.create_task(_ticker())
    await asyncio.sleep(0.06)
    content, _, _ = await M._handle_chat_foreground(ctx, {}, "job-x", project_id="aaaaaaaaaaaa")
    await tk
    gap = max(b - a for a, b in zip(ticks, ticks[1:]))
    assert content.startswith(M.PROJECT_TURN_SKIPPED)
    assert gap < 0.5, gap
