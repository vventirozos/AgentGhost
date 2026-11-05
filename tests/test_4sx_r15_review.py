"""Pins for the fresh-eye review of 2026-11-05 round 15 (the r14 diff), inverted from the reviewer's repros.

M1: a job's REGISTRY id reaches the run-row ownership walls too (`runs._job_refs`); m2: a cancelled job collected by
id is FAILED; m3: `lab put/get` resolve EVERY spelling as file_system does and show where it landed (`shown_path`)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_4sx_c_run_tool import _ctx as _run_ctx  # noqa: E402
from ghost_agent.core import request_scope as RS  # noqa: E402
from ghost_agent.tools import delegate as D  # noqa: E402
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


class _Sup:
    def __init__(self, entries):
        self.entries, self.cancelled, self.woken = entries, [], []

    def reap(self):
        return []

    def list_entries(self):
        return list(self.entries.values())

    def get(self, x):
        return self.entries.get(x)

    def mark_woken(self, x):
        self.woken.append(x)

    def log_tail(self, x, lines=0):
        return "A-SECRET-OUTPUT"

    def cancel(self, x):
        self.cancelled.append(x)
        self.entries[x] = {**self.entries[x], "state": "cancelled"}
        return True, "killed"


def _setup(tmp_path, monkeypatch, scope, state="running"):
    """Project A promotes a command exactly as execute does: a registry mirror (no started_by) + a run row."""
    from ghost_agent.core.jobs import get_job_registry
    sid = "job-1a1a1a1a"
    c = _run_ctx(tmp_path, pid="aaaaaaaaaaaa")
    hreg = get_job_registry(c)
    from ghost_agent.tools.execute import _register_promoted_job
    _register_promoted_job(hreg, {"id": sid, "command": "A-SECRET-BUILD"}, "A-SECRET-BUILD")
    rj = [j for j in hreg.list(kind=D.SANDBOX_KIND)][0]
    assert rj.id != sid
    ent = {"id": sid, "state": state, "command": "A-SECRET-BUILD", "exit_code": 0 if state == "done" else None}
    sup = _Sup({sid: ent})
    monkeypatch.setattr(D, "_sandbox_supervisor", lambda ctx: sup)
    monkeypatch.setattr(R, "_supervisors", lambda ctx: (None, sup))
    scope(c)
    assert R.record_promoted_job(c, {"id": sid, "command": "A-SECRET-BUILD"}) == sid
    # now the owner works in project B (same process, same job registry)
    c.current_project_id = "bbbbbbbbbbbb"
    scope(c)
    return c, sid, rj.id, sup


@pytest.mark.asyncio
async def test_m1_project_b_cannot_stop_project_a_job_by_registry_id(tmp_path, monkeypatch, scope):
    c, sid, rid, sup = _setup(tmp_path, monkeypatch, scope)
    ctl = await R.tool_run(c, action="stop", name=sid)               # control: the sandbox id is walled (r12 F4)
    assert ctl.reason_code == "run_not_allowed" and sup.cancelled == []
    out = await R.tool_run(c, action="stop", name=rid)               # the registry id is walled the same way
    assert out.reason_code == "run_not_allowed" and sup.cancelled == []


@pytest.mark.asyncio
async def test_m1_project_b_cannot_read_project_a_job_by_registry_id(tmp_path, monkeypatch, scope):
    c, sid, rid, sup = _setup(tmp_path, monkeypatch, scope, state="done")
    ctl = await R.tool_run(c, action="collect", name=sid)
    assert ctl.reason_code == "run_not_allowed"
    st = await R.tool_run(c, action="status", name=rid)
    assert st.reason_code == "run_not_allowed" and "A-SECRET-BUILD" not in str(st)
    out = await R.tool_run(c, action="collect", name=rid)
    assert out.reason_code == "run_not_allowed" and "A-SECRET-OUTPUT" not in str(out)
    assert sid not in sup.woken                                      # A's wake is not suppressed


@pytest.mark.asyncio
async def test_m1_the_jobs_alias_keeps_the_wall_for_a_registry_id(tmp_path, monkeypatch, scope):
    c, sid, rid, sup = _setup(tmp_path, monkeypatch, scope)
    ctl = await R.jobs_alias(context=c, action="cancel", job_id=sid)
    assert ctl.reason_code == "run_not_allowed"
    out = await R.jobs_alias(context=c, action="cancel", job_id=rid)
    assert out.reason_code == "run_not_allowed" and sup.cancelled == []


@pytest.mark.asyncio
async def test_m2_a_cancelled_job_collected_by_id_is_failed(tmp_path, monkeypatch, scope):
    """The owned run-row collect books a cancelled/stopped job FAILED; the by-id collect books it OK."""
    from ghost_agent.core.jobs import STATUS_CANCELLED, get_job_registry
    sid = "job-2b2b2b2b"
    c = _run_ctx(tmp_path, pid=None, conv="")
    hreg = get_job_registry(c)
    j = hreg.register(D.SANDBOX_KIND, "pytest -q", sandbox_job_id=sid)
    hreg.finish(j.id, status=STATUS_CANCELLED, error="sandbox job cancelled",
                result=f"[sandbox job {sid} cancelled]\n$ pytest -q\n")
    sup = _Sup({sid: {"id": sid, "state": "cancelled", "command": "pytest -q"}})
    monkeypatch.setattr(D, "_sandbox_supervisor", lambda ctx: sup)
    monkeypatch.setattr(R, "_supervisors", lambda ctx: (None, sup))
    scope(c)
    out = await R.tool_run(c, action="collect", name=sid)
    assert out.status is OutcomeStatus.FAILED and out.reason_code == "run_job_failed"


@pytest.mark.asyncio
async def test_m1_the_owner_of_the_job_still_reaches_it_by_its_registry_id(tmp_path, monkeypatch, scope):
    c, sid, rid, sup = _setup(tmp_path, monkeypatch, scope)
    c.current_project_id = "aaaaaaaaaaaa"                            # back in project A
    scope(c)
    out = await R.jobs_alias(context=c, action="cancel", job_id=rid)
    assert out.reason_code != "run_not_allowed" and sup.cancelled == [sid]


def test_m3_a_leading_slash_name_is_the_file_file_system_wrote(tmp_path):
    """file_system honours `/seed.sql` as `<scope>/seed.sql` (its documented leading-slash fix); the lab refuses it."""
    from test_4sx_b_lab_tool import _ctx
    from ghost_agent.tools import labs as L
    from ghost_agent.tools.file_system import _get_safe_path
    ctx = _ctx(tmp_path, pid=None)
    root = ctx.sandbox_dir.resolve()
    (root / "seed.sql").write_text("x")
    assert _get_safe_path(root, "/seed.sql").resolve() == root / "seed.sql"
    p, err = L._sandbox_path(ctx, "/seed.sql")
    assert err == "" and p == root / "seed.sql"
    assert L.shown_path(ctx, p) == "/workspace/seed.sql"
