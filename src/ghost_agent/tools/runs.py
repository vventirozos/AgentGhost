"""§4SX Phase C: the `run` tool — everything that keeps running, anywhere (UNIFIED_DESIGN §3.2/§4,
docs/labs/PHASE_C.md).

Built ALONGSIDE `manage_services` / `jobs`, which are unchanged. It is NOT advertised to the model in
Phase C (design r1 M2: a different tool list re-prefills the head); the Phase D switch advertises it,
adds aliases and removes the old tools.

Where a run lives decides who runs it — no third launcher (design C1):

* ``server`` @ ``workspace`` → the services supervisor (same lease, token, project scoping);
* ``job``    @ ``workspace`` → the jobs supervisor's ``start_detached`` (same script, TTL, cap);
* ``job`` / ``server`` @ ``lab_<id>/<member>`` → the labs host's detached job (``exec detach``),
  wrapped in ``timeout`` (45 min, the workspace job TTL — design r1 M4);
* ``app`` → Phase D (kept machines, ``/keep``; design r1 M1).

The run registry (``memory/run_registry.py``) is a host-side POINTER; every state shown is probed from
the backend at the time it is shown.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import shlex
import time
from typing import Any, Dict, Optional, Tuple

from ..utils.logging import Icons, pretty_log
from .outcome import OutcomeStatus, ToolOutcome

logger = logging.getLogger("GhostAgent")

ACTIONS = ("start", "status", "logs", "stop", "restart", "collect", "url")
LIFETIMES = ("job", "server", "app")
#: §4SX D5: the actions that only READ — open on a status question, never a mutation, never a dropped change
READ_ACTIONS = frozenset({"status", "logs", "url", "collect"})
_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{0,30}$")
_LAB_WHERE_RE = re.compile(r"^(lab_[0-9a-f]{8})/([a-z][a-z0-9-]{0,30})$")
#: a lab job is killed after this (the workspace job TTL, design r1 M4)
LAB_JOB_TTL_S = 2700
LOG_LINES_DEFAULT = 60
LOG_LINES_MAX = 400
OUT_CAP = 6 * 1024
#: states a run cannot leave (stop/restart/logs report them instead of acting)
_ENDED = frozenset({"stopped", "exited", "failed", "lost", "replaced", "expired"})
#: one start at a time per (owner, name) (review r1 M4)
_START_LOCKS: Dict[Tuple[str, str], asyncio.Lock] = {}


def _stop_asked() -> bool:
    """The user's stop for this call (§4RZ) — read INSIDE the shielded start (r4 minor 2): the
    shield keeps a started backend step whole, but nothing may START after the user said stop."""
    try:
        from ..utils import tool_stop
        return tool_stop.requested()
    except Exception:  # noqa: BLE001
        return False


def is_read_call(args: Any) -> bool:
    """True for a `run` call that only reads (by its action)."""
    if isinstance(args, str):
        try:
            import json as _json
            args = _json.loads(args, strict=False)
        except ValueError:
            return False
    return isinstance(args, dict) and str(args.get("action") or "").strip().lower() in READ_ACTIONS


_NOT_STARTED = "RUN: not started — stopped by the user before it began."


async def _svc_undo(svc, key: str) -> bool:
    """Stop a service a start/restart could not record; True only when it is verifiably gone or dead
    (r4: `svc.stop` returns a failure — it does not raise — when the process survives TERM+KILL)."""
    try:
        await asyncio.to_thread(svc.stop, key)
        e = {x.get("key"): x for x in await asyncio.to_thread(svc.list_entries)}.get(key)
        return e is None or await asyncio.to_thread(svc._entry_state, e) is False
    except Exception:  # noqa: BLE001
        return False


def _start_lock(owner: str, name: str) -> asyncio.Lock:
    lk = _START_LOCKS.get((owner, name))
    if lk is None:
        lk = _START_LOCKS[(owner, name)] = asyncio.Lock()
    return lk


def _out(text: str, status: OutcomeStatus, *, changed: Optional[bool], code: str = None) -> ToolOutcome:
    return ToolOutcome(text, status=status, reason_code=code, world_changed=changed)


def _refuse(text: str, code: str = "run_bad_request") -> ToolOutcome:
    return _out(f"RUN ERROR: {text}", OutcomeStatus.REJECTED, changed=False, code=code)


def _cap(text: str) -> str:
    text = text or ""
    return text if len(text) <= OUT_CAP else f"[… {len(text) - OUT_CAP} chars cut …]\n" + text[-OUT_CAP:]


def services_key_half(owner: str) -> Optional[str]:
    """The services registry's ``<project>`` key half for a run owner (design r1 H3): a project owner
    is its id; a conversation owner is ``c-<sha1[:12]>`` (the registry only takes safe ids — a bare
    name would let one conversation's `web` replace another's, or a legacy one)."""
    kind, _, ident = str(owner or "").partition(":")
    if not ident:
        return None
    if kind == "project":
        return ident if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", ident) else None
    if kind == "conversation":
        return "c-" + hashlib.sha1(ident.encode("utf-8")).hexdigest()[:12]
    return None


def _project_of(owner: str) -> Optional[str]:
    kind, _, ident = str(owner or "").partition(":")
    return ident if kind == "project" and ident else None


def _caller(context) -> Tuple[Optional[str], str, str]:
    """(origin, owner, refusal) — the lab tool's rule: members and replays are refused."""
    from .labs import _caller as _lab_caller
    origin, owner, refusal = _lab_caller(context)
    if refusal:
        return None, "", "runs are the owner's — not available on this channel"
    return origin, owner, ""


def _supervisors(context):
    sm = getattr(context, "sandbox_manager", None)
    svc = jobs = None
    try:
        from ..sandbox.services import get_service_supervisor
        svc = get_service_supervisor(sm) if sm is not None else None
    except Exception:  # noqa: BLE001
        svc = None
    try:
        from ..sandbox.jobs import get_job_supervisor
        jobs = get_job_supervisor(sm) if sm is not None else None
    except Exception:  # noqa: BLE001
        jobs = None
    return svc, jobs


# ------------------------------------------------------------------ probes

async def _probe(context, row: Dict[str, Any], origin: str) -> Tuple[str, str, Dict[str, Any]]:
    """(state, note, detail) of a run, asked of its backend now. ``detail`` carries backend data
    (exit_code, output_tail, port) for the caller."""
    be, ref = row["backend"], row["backend_ref"]
    svc, jobs = _supervisors(context)
    if be == "services":
        if svc is None:
            return "unknown", "the sandbox is not running", {}
        ents = {e.get("key"): e for e in await asyncio.to_thread(svc.list_entries)}
        e = ents.get(ref)
        if e is None:
            return "lost", "its service row is gone (stopped outside `run`)", {}
        if row["backend_token"] and str(e.get("token") or "") != row["backend_token"]:
            return "replaced", "another start reused the name (manage_services?)", {"port": e.get("port")}
        alive = await asyncio.to_thread(svc._entry_state, e)
        if alive is True:
            return "running", "", {"port": e.get("port")}
        if alive is None:
            return "unknown", "the sandbox did not answer the probe", {"port": e.get("port")}
        return "exited", "its process is not running", {"port": e.get("port")}
    if be == "jobs":
        if jobs is None:
            return "unknown", "the sandbox is not running", {}
        await asyncio.to_thread(jobs.reap)
        e = next((x for x in await asyncio.to_thread(jobs.list_entries) if x.get("id") == ref), None)
        if e is None:
            return "lost", "its job row is gone", {}
        st = str(e.get("state") or "")
        code = e.get("exit_code")
        mapped = {"running": "running", "done": "exited", "expired": "expired",
                  "cancelled": "stopped", "lost": "lost"}.get(st, "unknown")
        return mapped, (f"exit {code}" if code is not None else ""), {"exit_code": code}
    if be == "labs_job":
        from .labs import LabsNoAnswer, LabsUnreachable, _call
        lab_id, _, job_id = ref.partition("/")
        try:
            st, body = await _call(context, "GET", f"/v1/labs/{lab_id}/jobs/{job_id}", origin=origin,
                                   params={"tail_bytes": OUT_CAP})
        except (LabsUnreachable, LabsNoAnswer) as e:
            return "unknown", f"the labs host did not answer ({e})", {}
        body = body if isinstance(body, dict) else {}
        if st == 404 and body.get("error") == "unknown_lab":
            from .labs import lab_on_other_host
            other = await lab_on_other_host(context, lab_id, origin)
            if other:                                            # review 11-05 M3: not ours to call destroyed
                from .labs import other_host_phrase
                return "unknown", f"its lab was made on {other_host_phrase(other)}", {}
            return "lost", "its lab was destroyed", {}           # design r1 L1
        if st == 404 and body.get("error") == "unknown_job":
            return "lost", "the lab no longer knows the job", {}  # review r1 minor 3
        if st >= 300:
            return "unknown", f"labs host answered {st} {body.get('error') or ''}".strip(), {}
        state = str(body.get("state") or "")
        detail = {"exit_code": body.get("exit_code"), "output_tail": body.get("output_tail") or ""}
        if state == "running":
            return "running", "", detail
        if body.get("killed"):
            return "stopped", "killed", detail
        if state == "exited":
            code = body.get("exit_code")
            return "exited", (f"exit {code}" if code is not None else ""), detail
        return "lost", "the lab lost track of it", detail
    if be == "labs_app":
        return await _probe_app(context, row, origin)
    return "unknown", f"unknown backend {be}", {}


#: a kept app's host state → the run state (§4SX D3 r1 H4: none of these END the row while its lab lives)
_APP_STATE = {"up": "running", "starting": "running", "restarting": "running", "stopped": "app_stopped",
              "crashloop": "crashloop", "staged": "staged"}


async def _app_serving(context, owner: str, lineage: str, app: str, origin: str) -> Tuple[str, Dict[str, Any]]:
    """("ok"|"none"|"unknown:<why>", {serving: detail-or-None, staged: [lab ids]}) — the owner's kept labs that
    carry this (lineage, app); the SERVING one is the version whose app is mounted (design r1 M6: resolved at probe
    time, so a switch/rollback never needs the row repointed)."""
    from .labs import LabsNoAnswer, LabsUnreachable, _call
    try:
        st, body = await _call(context, "GET", "/v1/labs", origin=origin, params={"owner": owner})
        if st != 200 or not isinstance(body, dict):
            return f"unknown:labs host answered {st}", {}
        serving, staged = None, []
        for lab in body.get("labs") or []:
            if not isinstance(lab, dict) or str(lab.get("lifetime") or "") != "kept":
                continue
            if lab.get("lineage") and str(lab.get("lineage")) != lineage:
                continue
            st2, det = await _call(context, "GET", f"/v1/labs/{lab.get('lab_id')}", origin=origin)
            if st2 != 200 or not isinstance(det, dict):
                if st2 == 404:
                    continue
                return f"unknown:labs host answered {st2} for {lab.get('lab_id')}", {}
            a = det.get("app") if isinstance(det.get("app"), dict) else {}
            if str(a.get("name") or "") != app or str(a.get("lineage") or lineage) != lineage:
                continue
            if a.get("mounted"):
                serving = dict(det, app=a)
            else:
                staged.append(str(det.get("lab_id")))
    except (LabsUnreachable, LabsNoAnswer) as e:
        return f"unknown:the labs host did not answer ({e})", {}
    if serving is None and not staged:
        return "none", {}
    return "ok", {"serving": serving, "staged": staged}


async def _probe_app(context, row, origin) -> Tuple[str, str, Dict[str, Any]]:
    lineage, _, app = str(row["backend_ref"]).partition("/")
    how, found = await _app_serving(context, row["owner"], lineage, app, origin)
    if how.startswith("unknown:"):
        return "unknown", how.split(":", 1)[1], {}
    if how == "none":
        return "lost", "its kept lab was destroyed", {}
    srv = found.get("serving")
    staged = found.get("staged") or []
    if srv is None:
        return "staged", f"no version serves; staged in {', '.join(staged)} — `lab switch`", {"staged": staged}
    a = srv["app"]
    state = _APP_STATE.get(str(a.get("state") or ""), "unknown")
    bits = []
    if a.get("last_error") and str(a.get("state") or "") != "up":      # code r1 #9: a `restarting` app's error too
        bits.append(str(a.get("last_error"))[:160])
    if a.get("stopped_by_owner") and state == "app_stopped":
        bits.append("stopped by the owner")
    if staged:
        bits.append(f"next version {', '.join(staged)} staged")
    detail = {"lab_id": srv.get("lab_id"), "url": a.get("url") or "", "port": a.get("port"),
              "member": a.get("member"), "restarts": a.get("restarts"), "app_state": a.get("state"),
              "last_error": a.get("last_error"), "stopped_by_owner": bool(a.get("stopped_by_owner")),
              "staged": staged}
    return state, "; ".join(bits), detail


async def _app_op(context, reg, row, origin, op: str) -> ToolOutcome:
    """`run start|stop|restart` of an `app` run = POST /app/{op} on the SERVING lab (owner-only host-side; a probe
    only on its own probe lab)."""
    from .labs import LabsNoAnswer, LabsUnreachable, _call
    state, note, detail = await _probe(context, row, origin)
    if state == "unknown":
        return _out(f"RUN ERROR: could not find {row['name']!r}'s serving lab — {note}.", OutcomeStatus.UNRESOLVED,
                    changed=False, code="run_unreachable")
    if state in _ENDED:
        reg.set_state(row["run_id"], state, note)
        return _refuse(f"{row['name']!r} is {state} ({note}) — keep a lab again to bring it back", "run_ended")
    if state == "staged":
        return _refuse(f"{row['name']!r}: {note}", "run_app_staged")
    lab_id = detail["lab_id"]
    try:
        st, body = await _call(context, "POST", f"/v1/labs/{lab_id}/app/{op}", origin=origin, timeout=120)
    except LabsUnreachable as e:
        return _out(f"RUN ERROR (labs_unreachable): the labs host did not answer ({e}) — nothing was changed; "
                    "try again later.", OutcomeStatus.FAILED, changed=False, code="run_unreachable")   # code r1 #12
    except LabsNoAnswer as e:
        return _out(f"RUN ERROR (no_answer): the labs host did not answer the {op} ({e}) — `run status "
                    f"name={row['name']}` shows what happened.", OutcomeStatus.UNRESOLVED, changed=None,
                    code="run_no_answer")
    body = body if isinstance(body, dict) else {}
    if st == 403:
        return _refuse(f"{op} of the app {row['name']!r} is the owner's ({body.get('message') or 'owner_only'})",
                       "run_not_allowed")
    if st >= 300:
        return _out(f"RUN ERROR: the labs host refused the {op} ({st} {body.get('error')}: "
                    f"{str(body.get('message') or '')[:200]}).", OutcomeStatus.FAILED, changed=False,
                    code="run_stop_failed" if op == "stop" else "run_start_failed")
    a = body.get("app") if isinstance(body.get("app"), dict) else {}
    new = _APP_STATE.get(str(a.get("state") or ""), "unknown")
    probe = body.get("probe") if isinstance(body.get("probe"), dict) else {}
    reg.set_state(row["run_id"], new if new != "unknown" else state, f"{op} by {origin}")
    pretty_log("Run App", f"{row['name']} {op} on {lab_id} → {a.get('state')} by {origin}", icon=Icons.STOP)
    ok = (new == "app_stopped") if op == "stop" else (new == "running" and probe.get("ok") is not False)
    url = a.get("url") or detail.get("url") or ""
    return _out(("" if ok else "RUN ERROR (app_not_up): ") + f"[run {row['name']} app {op}] {a.get('state') or '?'}"
                + (f" — {url}" if url and op != "stop" else "")
                + ("" if ok else f" — the check did not pass: {str(probe or a.get('last_error') or '')[:200]}"),
                OutcomeStatus.OK if ok else OutcomeStatus.FAILED, changed=True,
                code=None if ok else "run_app_not_up")


async def _probed(context, reg, row, origin) -> Tuple[str, str, Dict[str, Any]]:
    state, note, detail = await _probe(context, row, origin)
    if state != "unknown":
        reg.set_state(row["run_id"], state, note)
    return state, note, detail


def _line(row, state, note, detail) -> str:
    port = detail.get("port") or row.get("port")
    bits = [f"{row['name']}", row["lifetime"], row["place"], state + (f" ({note})" if note else "")]
    if detail.get("url"):
        bits.append(str(detail["url"]))               # an `app` run's stable address
    elif port:
        bits.append(f":{port}")
    return "  " + "  ".join(bits) + f"  [{row['run_id']}]"


# ------------------------------------------------------------------ the tool

_FAILED_HEAD = "RUN ERROR"
_UNRESOLVED_HEAD = "RUN UNRESOLVED"
#: the collect head — the same pattern the stored grader reads (outcome_heuristics._RUN_COLLECT_RE; pinned equal)
_COLLECT_HEAD_RE = re.compile(r"\[run [a-z][a-z0-9-]{0,30} (?!logs\b|app\b)([a-z_]+)[^\]\n]*\]\s*\nEXIT CODE: (-?\d+|none)\b")


def _headed(out):
    """review r7: ONE head rule for every `run` result, applied where it leaves the tool (`tool_run`, and the
    aliases' `_with_hint`) — `RUN ERROR` heads a result exactly when its status is FAILED/REJECTED. A stored row
    is re-graded from its text alone, and the pass-through refusals (`LAB ERROR (…)`, `SANDBOX EGRESS BLOCKED`,
    a failed restart's "The old server…") and the UNRESOLVED `RUN ERROR (no_answer)` results each read wrong
    there. A collect keeps its own head (`[run x exited …]` + `EXIT CODE:` decides it)."""
    if not isinstance(out, ToolOutcome):
        return out
    text = str(out)
    failed = out.status in (OutcomeStatus.FAILED, OutcomeStatus.REJECTED)
    if failed and not (text.startswith(_FAILED_HEAD) or _COLLECT_HEAD_RE.match(text)):
        # ON the first line (review r8 M1: a head line of its own became every failure's "error line" — the
        # digest and the task state named "RUN ERROR — not done:" ×3 and lost each reason)
        text = f"{_FAILED_HEAD} — {text}"
    elif not failed and text.startswith(_FAILED_HEAD):
        text = _UNRESOLVED_HEAD + text[len(_FAILED_HEAD):]
    elif (out.status is OutcomeStatus.UNRESOLVED and not text.startswith(_UNRESOLVED_HEAD)
          and not _COLLECT_HEAD_RE.match(text)):
        # review r9: every UNRESOLVED result says so in its head too ("stop sent, but the backend did not answer
        # the check", "no result yet", …) — a stored row has only its text, and each shape read as a success
        text = f"{_UNRESOLVED_HEAD} — {text}"
    else:
        return out
    return ToolOutcome(text, status=out.status, world_changed=out.world_changed, reason_code=out.reason_code,
                       declared=out.declared, call_args=out.call_args, duration_s=out.duration_s)


async def tool_run(context=None, action: str = None, name: str = None, command: str = None,
                   where: str = None, lifetime: str = None, port=None, lines=None, **kwargs) -> ToolOutcome:
    return _headed(await _tool_run(context, action, name, command, where, lifetime, port, lines, **kwargs))


_JOB_ID_RE = re.compile(r"job-[0-9a-f]{6,16}")


def _job_refs(context, job_id: str) -> set:
    """Every id a background job is known by: the one given plus, for a REGISTRY id, its sandbox id — the id its run
    row is filed under (`backend_ref`). Review 11-05 r15 M1: the run-row ownership walls matched the sandbox id only,
    so project B stopped/collected project A's job by its registry id (what `run status` prints)."""
    out = {str(job_id or "")}
    try:
        from ..core.jobs import get_job_registry
        j = get_job_registry(context).get(str(job_id or ""))
        sid = str(((j.meta or {}) if j is not None else {}).get("sandbox_job_id") or "")
        if sid:
            out.add(sid)
    except Exception:  # noqa: BLE001
        logger.debug("job refs of %s not read", job_id, exc_info=True)
    return out


async def _job_by_id(context, action: str, job_id: str) -> ToolOutcome:
    """`run status|collect|stop name=<job id>` (and a bare `run collect`, owner-less turns only) on a background job
    with no run row of this caller, under the jobs tool's own rules (§4NS: the owner's turn, or the job's starter /
    its wake).

    Review 11-05 r14: ONE record decides — the jobs registry's row, found by EITHER id (`_resolve_job`: the registry
    id `run status` prints, or the sandbox id the promotion banner quotes) and judged by its OWN status (the landing
    already normalised the exit code: `… | head` SIGPIPE is DONE, exit≠0 is FAILED, expired/lost are FAILED). The r11–r13
    layers read the supervisor (one id namespace only) and parsed the result text (a command could print the line)."""
    from ..core.jobs import STATUS_DONE, STATUS_RUNNING, get_job_registry
    from .delegate import _resolve_job, _sync_sandbox_jobs, job_visible_to_caller, tool_jobs
    if not job_id:
        return await tool_jobs(action="collect", context=context)
    if action == "stop" and _leaf_meta().get("leaf"):
        # r13 m4: a build stops only the runs IT started — a job with no run row is not one of them
        return _refuse(f"{job_id} was not started by this build — a build reads it, never stops it", "run_not_allowed")
    try:
        reg = get_job_registry(context)
        await asyncio.to_thread(_sync_sandbox_jobs, reg, context)
        job = _resolve_job(reg, job_id)
    except Exception:  # noqa: BLE001
        logger.debug("job %s not resolved", job_id, exc_info=True)
        job = None
    if job is None:
        return _refuse(f"no background job {job_id} (finished jobs are kept for a while only; `run status` lists "
                       "this conversation's/project's runs)", "run_unknown")
    if not job_visible_to_caller(job):
        return _refuse(f"job {job_id} was started by another request — it is not yours to read or stop",
                       "run_not_allowed")
    if action == "status":
        sid = str((job.meta or {}).get("sandbox_job_id") or "")
        return _out(f"[job {job.id}{f' (sandbox {sid})' if sid and sid != job.id else ''} {job.status}]: "
                    f"{str(job.label or '')[:80]}"
                    + (" — still running; collect it when it has ended" if job.status == STATUS_RUNNING else
                       f" — `run collect name={job_id}` reads its result"), OutcomeStatus.OK, changed=False)
    out = await tool_jobs(action={"stop": "cancel"}.get(action, action), job_id=job_id, context=context)
    if action == "stop":
        # review r2: a cancel stamps no report, so the reaper never ends the job's run row — end it here
        try:
            _, _jobs = _supervisors(context)
            sid = str((job.meta or {}).get("sandbox_job_id") or job_id)
            _e = await asyncio.to_thread(_jobs.get, sid) if _jobs is not None else None
            if _e:
                await asyncio.to_thread(end_landed_job_runs, context, [_e])
        except Exception:  # noqa: BLE001
            logger.debug("stop by id: run row not ended", exc_info=True)
        return out
    if getattr(out, "status", None) is not OutcomeStatus.OK:
        return out
    now = reg.get(job.id) or job
    if now.status == STATUS_RUNNING:
        return ToolOutcome(str(out), status=OutcomeStatus.UNRESOLVED, world_changed=False,
                           reason_code="run_still_running")
    if now.status != STATUS_DONE:
        # failed, cancelled (r15 m2: the owned collect books anything but a clean exit as failed), or unknown
        return ToolOutcome(str(out), status=OutcomeStatus.FAILED, world_changed=False, reason_code="run_job_failed")
    return out


async def _tool_run(context=None, action: str = None, name: str = None, command: str = None,
                    where: str = None, lifetime: str = None, port=None, lines=None, **kwargs) -> ToolOutcome:
    action = str(action or "").strip().lower()
    if action not in ACTIONS:
        return _refuse(f"unknown action {action!r}. Valid: {', '.join(ACTIONS)}.")
    origin, owner, refusal = _caller(context)
    if refusal:
        return _refuse(refusal, "run_not_allowed")
    _jn = str(name or "").strip()
    if not owner:
        # review: a job's wake turn and a scheduled task run with no project and no conversation — they still read,
        # collect and cancel the workspace jobs THEY may reach (the jobs tool's own §4NS rule: the owner's turn, or
        # the job's starter / its wake), as `run status/collect/stop name=<job id>`
        if action == "status" and not _jn:
            from .delegate import tool_jobs
            return await tool_jobs(action="status", context=context)
        if (action == "collect" and not _jn) or (action in ("status", "collect", "stop") and _JOB_ID_RE.fullmatch(_jn)):
            return await _job_by_id(context, action, _jn)
        return _refuse("this request has no project and no conversation — runs are filed under one. A background "
                       "job's own result: `run collect name=<job id>`; a project's runs: switch to the project "
                       "first (manage_projects action=switch).", "run_no_owner")
    from ..memory.run_registry import registry_for
    reg = registry_for(context)
    if reg is None:
        return _out("RUN ERROR: the run registry is unavailable — nothing was started or changed.",
                    OutcomeStatus.FAILED, changed=False, code="run_registry_unavailable")
    _by_ref = None
    if (action in ("status", "collect", "stop", "logs") and _JOB_ID_RE.fullmatch(_jn)
            and reg.latest_by_name(owner, _jn) is None):
        _ids = _job_refs(context, _jn)
        _refs = [r for r in reg.rows(None) if r["backend"] == "jobs" and r["backend_ref"] in _ids]
        _mine = [r for r in _refs if r["owner"] == owner]
        if _mine:
            # review 11-05 r12 F1: a job THIS owner started with `run` (the jobs tool never adopts those — `stop` by
            # id answered "not running" while it ran) is handled on its own run row, whatever its run name now is
            _by_ref = _mine[-1]
        elif _refs:
            # review 11-05 r12 F4: filed under another owner — reached by id it was stopped / collected from here
            return _refuse(f"{_jn} is a run of {_refs[-1]['owner']} — runs are read and stopped only from the "
                           "conversation or project that started them", "run_not_allowed")
        elif action != "logs":
            # review 11-05 W-M2: a background job with no run row (promoted before the switch) is read and stopped by
            # its job id through `run` — the one route the results name (never `jobs`)
            return await _job_by_id(context, action, _jn)
    if action == "collect" and not _jn:
        # review 11-05 r12 F6: a bare collect in an owned turn names nothing — it was every project's unread jobs
        # (each marked collected and its wake suppressed)
        return _refuse("name the run to collect (`run status` lists this conversation's/project's runs; a background "
                       "job by its id: `run collect name=job-…`)", "run_name_required")
    if _by_ref is not None:
        return await _job_row_op(context, reg, _by_ref, origin, action, lines)
    if action == "status":
        return await _status(context, reg, owner, origin, name)
    name = str(name or "").strip()
    if not _NAME_RE.match(name):
        return _refuse("`name` must match ^[a-z][a-z0-9-]{0,30}$ (e.g. `api`, `bench-1`)")
    if action == "start":
        _app_row = reg.live_by_name(owner, name)
        if _app_row is not None and _app_row["lifetime"] == "app":
            # §4SX D3 r1 H4: an app run is started again on its kept lab, never re-created here
            if origin == "probe" and _app_row["origin"] != "probe":
                return _refuse(f"{name!r} was kept by {_app_row['origin'] or 'the owner'} — a probe changes only "
                               "its own runs", "run_not_allowed")
            if origin == "autopilot":
                return _refuse(_AUTOPILOT_APP, "run_not_allowed")
            return await _locked(owner, name, _app_op(context, reg, _app_row, origin, "start"))
        return await _locked(owner, name, _start(context, reg, owner, origin, name, command, where, lifetime, port))
    if action == "stop":
        # review 11-05 R-m9: a start of this name still in flight records its row AFTER this read — wait it out
        # (the start lock), so the stop finds the run instead of answering "no run named" while it comes up
        async with _start_lock(owner, name):
            pass
    row = reg.live_by_name(owner, name) if action in ("stop", "restart", "url") else reg.latest_by_name(owner, name)
    if row is None and action == "restart":
        # review r2 minor 5: a crashed server restarts whether or not a status probed it first
        last = reg.latest_by_name(owner, name)
        if last is not None and last["state"] in ("exited", "failed") and last["lifetime"] == "server":
            row = last
    if row is None:
        last = reg.latest_by_name(owner, name)
        if last is not None:
            code = "run_replaced" if last["state"] == "replaced" else "run_ended"
            return _refuse(f"{name!r} is not running — it is {last['state']}"
                           + (f" ({last['state_note']})" if last["state_note"] else "")
                           + (". A start outside `run` took its name; that one is not stopped or restarted here"
                              if code == "run_replaced" else ""), code)
        return _refuse(f"no run named {name!r} here — `run status` lists them", "run_unknown")
    if origin == "probe" and row["origin"] != "probe" and action in ("stop", "restart"):
        return _refuse(f"{name!r} was started by {row['origin'] or 'the owner'} — a probe changes only its own runs",
                       "run_not_allowed")
    if row["lifetime"] == "app" and origin == "autopilot" and action in ("stop", "restart"):
        return _refuse(_AUTOPILOT_APP, "run_not_allowed")
    _leaf = str(_leaf_meta().get("leaf") or "")
    if _leaf and action in ("stop", "restart") and _leaf not in (_row_leaf(row), _row_started_leaf(row)):
        # review: an autopilot build (leaf) stops/restarts only the runs IT started — the owner's servers and jobs,
        # and another build's, are read (`status/logs/collect/url`), never changed
        return _refuse(f"{name!r} was not started by this build — a build reads it, never stops or restarts it",
                       "run_not_allowed")
    if row["lifetime"] == "app":
        if action == "url":
            return await _app_url(context, reg, row, origin)
        if action == "logs":
            return await _app_logs(context, row, origin, lines)
        if action in ("stop", "restart"):
            return await _locked(owner, name, _app_op(context, reg, row, origin, action))
    if action == "logs":
        return await _logs(context, row, origin, lines)
    if action == "collect":
        return await _collect(context, reg, row, origin)
    if action == "url":
        return _url(row)
    if action == "stop":
        # shielded (review 11-05 R-m9): a cancelled turn still records what the backend stop did — unshielded, the row
        # stayed "running" and the next probe booked "stopped outside `run`"
        return await asyncio.shield(_stop(context, reg, row, origin))
    return await _locked(owner, name, _restart(context, reg, row, origin))


async def _job_row_op(context, reg, row, origin, action, lines) -> ToolOutcome:
    """`run status|collect|stop|logs name=<job id>` on the caller's own workspace-job row (r12 F1) — the named-row
    rules (probe / build may stop only their own), whatever name the row has now."""
    if action == "status":
        if row["ended_at"] is not None:
            return _out("RUNS:\n" + _line(row, row["state"], row["state_note"], {}), OutcomeStatus.OK, changed=False)
        state, note, detail = await _probed(context, reg, row, origin)
        return _out("RUNS (probed now):\n" + _line(row, state, note, detail), OutcomeStatus.OK, changed=False)
    if action == "logs":
        return await _logs(context, row, origin, lines)
    if action == "collect":
        return await _collect(context, reg, row, origin)
    # stop
    if row["ended_at"] is not None:
        return _refuse(f"{row['backend_ref']} is not running — it is {row['state']}"
                       + (f" ({row['state_note']})" if row["state_note"] else ""), "run_ended")
    if origin == "probe" and row["origin"] != "probe":
        return _refuse(f"{row['backend_ref']} was started by {row['origin'] or 'the owner'} — a probe changes only "
                       "its own runs", "run_not_allowed")
    _leaf = str(_leaf_meta().get("leaf") or "")
    if _leaf and _leaf not in (_row_leaf(row), _row_started_leaf(row)):
        return _refuse(f"{row['backend_ref']} was not started by this build — a build reads it, never stops it",
                       "run_not_allowed")
    return await asyncio.shield(_stop(context, reg, row, origin))


#: §4SX D5: kept apps are production — unattended work reads them (status/logs/url), never starts/stops them
_AUTOPILOT_APP = ("a kept app is production (the owner's) — unattended work reads it (`run status/logs/url`), "
                  "never starts, stops or restarts it")


def _row_leaf(row) -> str:
    from ..memory.run_registry import RunRegistry
    return str(RunRegistry.meta_of(row).get("leaf") or "")


def _row_started_leaf(row) -> str:
    """The leaf whose `execute` detached this job (review r2: it may stop it — the leaf's END does not, that reads
    ``leaf`` only)."""
    from ..memory.run_registry import RunRegistry
    return str(RunRegistry.meta_of(row).get("started_leaf") or "")


def _leaf_meta() -> Dict[str, Any]:
    """§4SX D5: a run an autopilot leaf starts carries the leaf id IN its insert (review: tagging after the shielded
    start was skipped when the leaf was cancelled mid-start) — the leaf's end stops it (stop_leaf_runs)."""
    try:
        from ..core.lab_waits import LEAF_ID
        return {"leaf": LEAF_ID.get() or None}
    except Exception:  # noqa: BLE001
        return {}


async def stop_leaf_runs(context, owner: str, leaf_id: str) -> list:
    """§4SX D5: stop the job/server runs THIS leaf started (meta leaf == leaf_id) and left running — never
    another leaf's, a turn's or the owner's. Returns the names stopped. Never raises."""
    gone: list = []
    if not leaf_id:
        return gone
    pending = [t for t in (_LEAF_INFLIGHT.get(str(leaf_id)) or ()) if not t.done()]
    if pending:
        # review r2 M1: a start the leaf's cancel left running records its row after this read — wait for it
        try:
            await asyncio.wait(pending, timeout=LEAF_INFLIGHT_WAIT_S)
        except Exception:  # noqa: BLE001
            logger.warning("leaf starts not awaited", exc_info=True)
    # AFTER the wait (a start that finished inside it is read below, not stopped twice): one still going stops
    # its own row when it lands (_inflight_done)
    _LEAF_ENDED[str(leaf_id)] = (context, owner)
    while len(_LEAF_ENDED) > 64:
        _old = next(iter(_LEAF_ENDED))
        _LEAF_ENDED.pop(_old, None)
        for _k in [k for k in _LEAF_STOP_LOCKS if k[1] == _old]:
            _LEAF_STOP_LOCKS.pop(_k, None)
    # keyed by the running loop too (review r5: an asyncio.Lock binds to the first loop that contends on it)
    lock = _LEAF_STOP_LOCKS.setdefault((id(asyncio.get_running_loop()), str(leaf_id)), asyncio.Lock())
    async with lock:
        return await _stop_leaf_rows(context, owner, leaf_id, gone)


async def _stop_leaf_rows(context, owner: str, leaf_id: str, gone: list) -> list:
    """The row pass of `stop_leaf_runs`, one at a time per leaf (it reads the rows AFTER the previous pass ended
    them, so a row is stopped once)."""
    try:
        from ..memory.run_registry import RunRegistry, registry_for
        reg = registry_for(context)
        for row in (reg.rows(owner, live_only=True) if reg is not None else []):
            if row["lifetime"] == "app" or str(RunRegistry.meta_of(row).get("leaf") or "") != str(leaf_id):
                continue
            try:
                out = await _stop(context, reg, row, "autopilot")
                if getattr(out, "status", None) is OutcomeStatus.OK:
                    gone.append(row["name"])
                else:
                    logger.warning("leaf run %s not stopped: %s", row["name"], str(out)[:200])
            except Exception:  # noqa: BLE001
                logger.warning("leaf run %s not stopped", row["name"], exc_info=True)
    except Exception:  # noqa: BLE001
        logger.warning("leaf runs not read", exc_info=True)
    if gone:
        pretty_log("Run Stop", f"leaf ended: stopped its run(s) {', '.join(gone)}", icon=Icons.SANDBOX_BOX)
    return gone


async def _locked(owner: str, name: str, coro) -> ToolOutcome:
    """Run a start/restart under its (owner, name) lock as ONE shielded task (review r3 minor 3): a
    cancelled turn released the lock while the shielded backend step still ran, and a quick second
    start of the name raced it into two live rows. The task finishes (and records) on its own; it
    catches its own errors, so nothing ends unretrieved."""
    async def _task():
        async with _start_lock(owner, name):
            if _stop_asked():
                coro.close()
                return _out(_NOT_STARTED, OutcomeStatus.REJECTED, changed=False, code="run_stopped_by_user")
            try:
                return await coro
            except Exception as e:  # noqa: BLE001
                return _out(f"RUN ERROR: {type(e).__name__}: {e}", OutcomeStatus.FAILED, changed=None,
                            code="run_internal")
    t = asyncio.ensure_future(_task())
    leaf = str(_leaf_meta().get("leaf") or "")
    if leaf:
        # review r2 M1: a leaf cancelled mid-start ends (stop_leaf_runs) while this shielded start is still going —
        # its row lands after the leaf's end read the rows. The end waits for these first.
        _LEAF_INFLIGHT.setdefault(leaf, set()).add(t)
        t.add_done_callback(lambda _t, _l=leaf: _inflight_done(_l, _t))
    return await asyncio.shield(t)


#: leaf id → its start/restart tasks still running (review r2 M1)
_LEAF_INFLIGHT: Dict[str, set] = {}
#: how long a leaf's end waits for its in-flight starts (inside coding_loop's 120 s cleanup bound)
LEAF_INFLIGHT_WAIT_S = 40.0          # review r3: leaves the run stops + lab destroys room in the 120 s bound
#: leaf id → (context, owner) of a leaf whose end already ran: a start that lands after it stops its own row
_LEAF_ENDED: Dict[str, tuple] = {}
#: the late stops in flight (a reference, so none is collected while it runs)
_LATE_STOPS: set = set()
#: leaf id → the lock its stops run under (review r4: two late starts landing together stopped the first row twice)
_LEAF_STOP_LOCKS: Dict[tuple, asyncio.Lock] = {}


def _inflight_done(leaf: str, t) -> None:
    s = _LEAF_INFLIGHT.get(leaf)
    if s is not None:
        s.discard(t)
        if not s:
            _LEAF_INFLIGHT.pop(leaf, None)
    ended = _LEAF_ENDED.get(leaf)
    if ended is not None and not t.cancelled():
        # review r3: this start outlived the leaf's wait — its row landed after the end read the rows; stop it now
        try:
            _bg = asyncio.ensure_future(stop_leaf_runs(ended[0], ended[1], leaf))
            _LATE_STOPS.add(_bg)                          # review r4: held until done (never collected mid-run)
            _bg.add_done_callback(_LATE_STOPS.discard)
        except Exception:  # noqa: BLE001
            logger.warning("late leaf run not stopped", exc_info=True)


async def _start(context, reg, owner, origin, name, command, where, lifetime, port) -> ToolOutcome:
    command = str(command or "").strip()
    where = str(where or "").strip()
    lifetime = str(lifetime or "").strip().lower()
    if not command:
        return _refuse("`command` is required")
    if not where:
        return _refuse("`where` is required: `workspace` or `lab_<id>/<member>` — there is no default")
    if lifetime not in LIFETIMES:
        return _refuse(f"`lifetime` must be one of {', '.join(LIFETIMES)}")
    if lifetime == "app":
        return _refuse("an `app` run is made by `lab keep` on a project's lab (a kept machine) — `run` then starts, "
                       "stops and restarts it by name; for anything else use `server`", "run_app_via_keep")
    try:
        port = int(port) if port not in (None, "") else None
    except (TypeError, ValueError):
        return _refuse("`port` must be a number")
    if port is not None and not 1 <= port <= 65535:
        return _refuse("`port` must be 1–65535")
    existing = reg.live_by_name(owner, name)
    if existing is not None:
        state, _, _ = await _probed(context, reg, existing, origin)
        if state in ("running", "unknown"):
            return _refuse(f"a run named {name!r} is already {state} ({existing['place']}) — stop or restart it",
                           "run_exists")
    if where == "workspace":
        # review: the guards `execute` runs first — a workspace command naming the labs host (or carrying its
        # key) and one probing the agent's own ports are refused here too
        from .labs import lab_reach_refusal
        _refused = lab_reach_refusal(context, {"command": command}, "run")
        if _refused is not None:
            return _refused
        from .execute import _AGENT_PORT_PROBE_MSG, _command_probes_agent_port
        if _command_probes_agent_port(command):
            return _out(_AGENT_PORT_PROBE_MSG, OutcomeStatus.REJECTED, changed=False, code="run_agent_port")
        _blocked = _shell_guard_refusal(context, owner, command) or await _egress_refusal(context)
        if _blocked is not None:
            return _blocked
        if lifetime == "server":
            return await _start_ws_server(context, reg, owner, origin, name, command, port)
        return await _start_ws_job(context, reg, owner, origin, name, command)
    m = _LAB_WHERE_RE.match(where)
    if not m:
        return _refuse("`where` must be `workspace` or `lab_<8 hex>/<member>` (e.g. lab_0123abcd/db)")
    if lifetime == "server" and port is None:
        return _refuse("a `server` on a lab needs its `port` (what `lab http` probes)")
    return await _start_lab(context, reg, owner, origin, name, command, m.group(1), m.group(2), lifetime, port)


async def _start_ws_server(context, reg, owner, origin, name, command, port) -> ToolOutcome:
    svc, _ = _supervisors(context)
    if svc is None:
        return _out("RUN ERROR: the sandbox is not running — nothing was started.", OutcomeStatus.FAILED,
                    changed=False, code="sandbox_unavailable")
    half = services_key_half(owner)
    if half is None:
        return _refuse("this owner has no safe services key", "run_no_owner")
    from ..sandbox.services import entry_key
    key = entry_key(half, name)
    ents = {e.get("key"): e for e in await asyncio.to_thread(svc.list_entries)}
    before = ents.get(key)
    if before is not None and await asyncio.to_thread(svc._entry_state, before) is not False:
        # design r1 H3: never replace a service `run` did not start
        return _refuse(f"a service {name!r} is already registered here (started outside `run`) — pick another name",
                       "run_exists")
    if name in ents:
        # review r1 minor 4: a start would re-key the legacy unscoped row into ours (deleting it)
        return _refuse(f"a legacy service {name!r} is registered (manage_services, no project) — pick another name",
                       "run_exists")

    async def _go():
        # shielded (review r1 M3): a cancelled turn still records what it started;
        # it catches its own errors (r2 minor 1: none may end unretrieved)
        try:
            return await _go_inner()
        except Exception as e:  # noqa: BLE001
            return f"{type(e).__name__}: {e}", None, None, False

    async def _go_inner():
        if _stop_asked():
            return _NOT_STARTED, None, None, False
        text = await asyncio.to_thread(svc.start, name, command, port=port, project_id=half)
        row = {e.get("key"): e for e in await asyncio.to_thread(svc.list_entries)}.get(key)
        ok = (row is not None and row.get("pid") and row.get("token")
              and (before is None or row.get("token") != before.get("token"))
              and await asyncio.to_thread(svc._entry_state, row) is not False)
        if not ok:
            return text, row, None, False
        rid = reg.add(name=name, place="workspace", lifetime="server", command=command, port=row.get("port"),
                      owner=owner, origin=origin, backend="services", backend_ref=key,
                      backend_token=str(row.get("token")), meta=_leaf_meta())
        if rid is None:                     # not recorded → not left running untracked
            return text, row, ("undone" if await _svc_undo(svc, key) else "kept"), True
        return text, row, rid, True
    text, row, rid, ok = await asyncio.shield(_go())
    if not ok:
        return _out(f"RUN ERROR: the server did not start.\n{_cap(str(text))}", OutcomeStatus.FAILED,
                    changed=None, code="run_start_failed")
    if rid in ("undone", "kept"):
        what = ("so it was stopped again" if rid == "undone" else
                f"and stopping it again FAILED — it may still run as service {key}")
        return _out(f"RUN ERROR: the run registry did not record the start, {what}.",
                    OutcomeStatus.FAILED, changed=None, code="run_registry_unavailable")
    pretty_log("Run Start", f"{name} server workspace :{row.get('port')} [{rid}] by {origin} {owner}",
               icon=Icons.SANDBOX_BOX)
    return _out(f"[run {name} started — server in the workspace, port {row.get('port')}, {rid}]\n"
                f"Still running: this is not a result. `run status`/`run logs name={name}` show it.\n"
                f"{_cap(str(text))}", OutcomeStatus.OK, changed=True)


async def _egress_refusal(context) -> Optional[ToolOutcome]:
    """`execute`'s first gate (§4GI): the sandbox egress must be Tor-only or cut off (review 11-05 r12 F3: a `run`
    job launches through the raw exec, which skips the check `sandbox.execute` makes — with egress neither enforced
    nor blocked it left with the host's address). Enforcement not attempted yet (a fresh sandbox) is run now; still
    not attempted = refused (fail closed)."""
    from ..sandbox.egress_gate import network_refusal
    sm = getattr(context, "sandbox_manager", None)
    if sm is None or not getattr(sm, "tor_proxy", None):
        return None
    attempted = getattr(sm, "egress_enforcement_attempted", None)
    try:
        if callable(attempted) and not attempted():
            await asyncio.to_thread(sm.ensure_running)
            if not attempted():
                return _out("RUN ERROR: SANDBOX EGRESS UNVERIFIED — the sandbox's Tor-only egress has not been set up "
                            "yet, so nothing was started. Try again in a moment.", OutcomeStatus.FAILED,
                            changed=False, code="egress_unavailable")
    except Exception as e:  # noqa: BLE001
        return _out(f"RUN ERROR: the sandbox could not be started ({type(e).__name__}) — nothing was started.",
                    OutcomeStatus.FAILED, changed=False, code="sandbox_unavailable")
    blk = network_refusal(sm)
    if blk is not None:
        return _out(f"RUN ERROR: {blk}", OutcomeStatus.FAILED, changed=False, code="egress_unavailable")
    return None


def _shell_guard_refusal(context, owner: str, command: str) -> Optional[ToolOutcome]:
    """The guards `execute` runs on a sandbox shell command, for a workspace `run` (review 11-05 R-M3: a command
    `execute` refused — the destructive deny-list, a write into a RELEASED project, a YouTube download, a detached
    `… &` server — ran unrefused as a `run` job or server). None = allowed. Fail CLOSED on the deny-list (a crashing
    validator never lets the command through)."""
    from .execute import _daemonized_server_block, _released_shell_block, _youtube_download_block
    from .validators import validate_shell
    try:
        ok, why = validate_shell(command)
    except Exception as e:  # noqa: BLE001
        ok, why = False, f"the shell validator failed ({type(e).__name__})"
    if not ok:
        return _out(f"RUN ERROR: SYSTEM BLOCK: the command was refused by the pre-execution validator: {why}. "
                    "It was NOT started.", OutcomeStatus.REJECTED, changed=False, code="run_shell_blocked")
    proj = _project_of(owner)
    rb = _released_shell_block(getattr(context, "project_store", None), command,
                               workdir=f"/workspace/projects/{proj}" if proj else "")
    if rb:
        return _out(f"RUN ERROR: {str(rb)}", OutcomeStatus.REJECTED, changed=False, code="run_released_blocked")
    for blk, code in ((_youtube_download_block, "run_youtube_blocked"), (_daemonized_server_block, "run_daemon_blocked")):
        m = blk(command)
        if m:
            return _out(f"RUN ERROR: {m}", OutcomeStatus.REJECTED, changed=False, code=code)
    return None


async def _start_ws_job(context, reg, owner, origin, name, command) -> ToolOutcome:
    _, jobs = _supervisors(context)
    if jobs is None:
        return _out("RUN ERROR: the sandbox is not running (or background jobs are off) — nothing was started.",
                    OutcomeStatus.FAILED, changed=False, code="sandbox_unavailable")
    proj = _project_of(owner)
    wd = f"/workspace/projects/{proj}" if proj else None
    async def _go():
        if _stop_asked():
            return None, "stopped by the user before it began", None
        try:
            # through bash -c with pipefail, as a promoted `execute` job is (review 11-05 R-M1: the runner script is
            # `#!/bin/sh` + `timeout … <cmd>`, so `cd app && pytest`, `FOO=1 x`, `source …`, loops ran as
            # `timeout cd …` → exit 127, and only the first command of `a && b` was under the ceiling)
            from .execute import _bash_c
            entry, reason = await asyncio.to_thread(jobs.start_detached, _bash_c(command), workdir=wd, label=name,
                                                    project_id=proj)
        except Exception as e:  # noqa: BLE001 — e.g. a missing project folder (review r1 minor 7)
            return None, f"{type(e).__name__}: {e}", None
        if entry is None or entry.get("duplicate"):
            return entry, reason, None
        rid = reg.add(name=name, place="workspace", lifetime="job", command=command, port=None, owner=owner,
                      origin=origin, backend="jobs", backend_ref=str(entry["id"]), meta=_leaf_meta())
        if rid is None:
            try:
                await asyncio.to_thread(jobs.cancel, entry["id"])
            except Exception:  # noqa: BLE001
                pass
        return entry, reason, rid
    entry, reason, rid = await asyncio.shield(_go())
    if entry is None:
        return _out(f"RUN ERROR: the job did not start: {reason}", OutcomeStatus.FAILED, changed=False,
                    code="run_start_failed")
    if entry.get("duplicate"):
        # (no id: the twin may be another conversation's — review r1 minor 7)
        return _refuse("this exact command is already running in the workspace — not starting a second copy",
                       "run_exists")
    if rid is None:
        return _out("RUN ERROR: the run registry did not record the start, so the job was cancelled again.",
                    OutcomeStatus.FAILED, changed=None, code="run_registry_unavailable")
    pretty_log("Run Start", f"{name} job workspace {entry['id']} [{rid}] by {origin} {owner}",
               icon=Icons.JOB_PROMOTE)
    return _out(f"[run {name} started — job in the workspace, {entry['id']}, {rid}]\n"
                f"Still running: this is not a result. `run collect name={name}` reads it when it ends "
                f"(killed after {int(entry.get('deadline_at', 0) - entry.get('promoted_at', 0)) // 60} min).",
                OutcomeStatus.OK, changed=True)


def _lab_start_refusal(det: Dict[str, Any], origin: str, owner: str, lab_id: str) -> Optional[ToolOutcome]:
    """Who may start a run on this lab (its host detail ``det``): a probe only on a probe's lab; unattended work
    only on ITS OWN project's disposable labs (review 11-05 R-M2: the lab tool's rule — the host gets no owner, so it
    cannot tell; a kept lab is production). None = allowed. Read BEFORE a restart's stop too (r12 F5)."""
    theirs = str(det.get("owner") or "")
    if origin == "probe" and str(det.get("origin") or "") != "probe":
        return _refuse(f"{lab_id} is not a probe's lab — a probe starts runs only on its own labs", "run_not_allowed")
    if origin == "autopilot":
        if not owner or theirs != owner:
            return _refuse(f"{lab_id} belongs to {theirs or 'someone else'} — unattended work runs only on its own "
                           f"project's labs ({owner or 'no project'}); create one", "run_not_allowed")
        if str(det.get("lifetime") or "") == "kept":
            return _refuse(f"{lab_id} is KEPT (production) — unattended work never runs commands on a production "
                           "machine; use a disposable lab of the project", "run_not_allowed")
    return None


async def _start_lab(context, reg, owner, origin, name, command, lab_id, member, lifetime, port,
                     meta: Optional[Dict[str, Any]] = None) -> ToolOutcome:
    from .labs import LabsNoAnswer, LabsUnreachable, _call, _service_reply, _unreachable, labs_configured
    if not labs_configured(context):
        return _refuse("no labs host is configured — a run can only go to the workspace", "labs_unconfigured")
    # --foreground (live acceptance 11-04): plain `timeout` makes its OWN process group, so the host's
    # group kill on `run stop` missed the server — it kept running (and its port) while the stop said
    # "stopped". In the foreground it stays in the job's group; the TTL still kills the command.
    wrapped = f"timeout --foreground -k 30 {LAB_JOB_TTL_S} bash -c {shlex.quote(command)}"
    # whose lab (review r1 minor 5, the lab tool's rule): a probe changes only a probe's lab;
    # another owner's lab is named in the result
    try:
        st, det = await _call(context, "GET", f"/v1/labs/{lab_id}", origin=origin)
    except LabsUnreachable as e:
        return _unreachable(context, str(e))
    except LabsNoAnswer as e:
        return _out(f"RUN ERROR (no_answer): the labs host did not answer ({e}) — nothing was started.",
                    OutcomeStatus.FAILED, changed=False, code="run_no_answer")
    if st >= 300:
        return _service_reply(st, det, context, lab_id)
    det = det if isinstance(det, dict) else {}
    theirs = str(det.get("owner") or "")
    _no = _lab_start_refusal(det, origin, owner, lab_id)
    if _no is not None:
        return _no
    other = f" (this lab belongs to {theirs})" if theirs and theirs != owner else ""
    place = f"{lab_id}/{member}"

    async def _go():
        # shielded (review r2 M2): POST → record → (rollback) completes even if the turn is cancelled
        if _stop_asked():
            return "error", "stopped by the user before it began", None, None
        try:
            st, body = await _call(context, "POST", f"/v1/labs/{lab_id}/exec", origin=origin,
                                   json_body={"member": member, "cmd": wrapped, "detach": True})
        except LabsUnreachable as e:
            return "unreachable", str(e), None, None
        except LabsNoAnswer as e:
            return "no_answer", str(e), None, None
        except Exception as e:  # noqa: BLE001
            return "error", f"{type(e).__name__}: {e}", None, None
        if st >= 300:
            return "refused", (st, body), None, None
        job_id = str((body or {}).get("job_id") or "")
        if not job_id:
            return "no_id", "", None, None
        rid = reg.add(name=name, place=place, lifetime=lifetime, command=command, port=port, owner=owner,
                      origin=origin, backend="labs_job", backend_ref=f"{lab_id}/{job_id}",
                      meta=_leaf_meta() if meta is None else meta)
        if rid is not None:
            return "ok", "", job_id, rid
        try:
            dst, dbody = await _call(context, "DELETE", f"/v1/labs/{lab_id}/jobs/{job_id}", origin=origin)
            killed = dst < 300 and isinstance(dbody, dict) and dbody.get("state") != "running"
        except Exception:  # noqa: BLE001
            killed = False
        return "unrecorded", killed, job_id, None
    kind, info, job_id, rid = await asyncio.shield(_go())
    if kind == "unreachable":
        return _unreachable(context, info)
    if kind == "no_answer":
        return _out(f"RUN ERROR (no_answer): the labs host did not answer ({info}) — the run may have started; "
                    f"check with `lab status {lab_id}` before starting it again.", OutcomeStatus.UNRESOLVED,
                    changed=None, code="run_no_answer")
    if kind == "error":
        return _out(f"RUN ERROR: the start failed: {info}", OutcomeStatus.FAILED, changed=None,
                    code="run_start_failed")
    if kind == "refused":
        return _service_reply(info[0], info[1], context, lab_id)
    if kind == "no_id":
        return _out("RUN ERROR: the labs host started nothing it could name (no job id).", OutcomeStatus.FAILED,
                    changed=None, code="run_start_failed")
    if kind == "unrecorded":
        what = (f"so {job_id} was killed again" if info else
                f"and killing {job_id} again was NOT confirmed — it may still run (killed after "
                f"{LAB_JOB_TTL_S // 60} min); check `lab status {lab_id}`")
        return _out(f"RUN ERROR: the run registry did not record the start, {what}.", OutcomeStatus.FAILED,
                    changed=None, code="run_registry_unavailable")
    pretty_log("Run Start", f"{name} {lifetime} {place} {job_id} [{rid}] by {origin} {owner}", icon=Icons.LAB_EXEC)
    reach = f" Reach port {port} with `lab http lab_id={lab_id} member={member} port={port}`." if port else ""
    return _out(f"[run {name} started — {lifetime} on {place}{other}, {job_id}, {rid}]\n"
                f"Still running: this is not a result. Killed after {LAB_JOB_TTL_S // 60} min.{reach}",
                OutcomeStatus.OK, changed=True)


async def _status(context, reg, owner, origin, name) -> ToolOutcome:
    rows = reg.rows(owner)
    if name:
        rows = [r for r in rows if r["name"] == str(name).strip()]
        if not rows:
            return _refuse(f"no run named {name!r} here", "run_unknown")
        rows = rows[-1:]
    else:
        rows = [r for r in rows if r["ended_at"] is None]
    if not rows:
        return _out("No runs here. (`run start name=… command=… where=workspace|lab_<id>/<member> "
                    "lifetime=job|server`)", OutcomeStatus.OK, changed=False)
    out = []
    for r in rows:
        if r["ended_at"] is not None:
            out.append(_line(r, r["state"], r["state_note"], {}))
            continue
        state, note, detail = await _probed(context, reg, r, origin)
        out.append(_line(r, state, note, detail))
    return _out("RUNS (probed now):\n" + "\n".join(out), OutcomeStatus.OK, changed=False)


def _lines(lines) -> int:
    try:
        n = int(lines)
    except (TypeError, ValueError):
        n = LOG_LINES_DEFAULT
    return max(1, min(LOG_LINES_MAX, n))


async def _logs(context, row, origin, lines) -> ToolOutcome:
    n = _lines(lines)
    svc, jobs = _supervisors(context)
    if row["backend"] == "services":
        if svc is None:
            return _out("RUN ERROR: the sandbox is not running.", OutcomeStatus.FAILED, changed=False,
                        code="sandbox_unavailable")
        if row["ended_at"] is not None:
            # r2 minor 4: an ended run is shown as it ended (a `run stop` removed its service row —
            # that is not "stopped outside run"); the full key never falls back to another service
            state, note = row["state"], row["state_note"]
        else:
            state, note, _ = await _probe(context, row, origin)
        if state == "replaced" or (state == "lost" and row["ended_at"] is None):
            # review r1 M1: by name the services lookup falls back to ANOTHER project's service
            return _refuse(f"{row['name']!r} is {state} ({note}) — its log is not this run's", "run_ended")
        text = await asyncio.to_thread(svc.logs, row["backend_ref"], n)
        head = f"[run {row['name']} logs" + (f" — {state}" if row["ended_at"] is not None else "") + "]"
        return _out(f"{head}\n{_cap(str(text))}", OutcomeStatus.OK, changed=False)
    if row["backend"] == "jobs":
        if jobs is None:
            return _out("RUN ERROR: the sandbox is not running.", OutcomeStatus.FAILED, changed=False,
                        code="sandbox_unavailable")
        text = await asyncio.to_thread(jobs.log_tail, row["backend_ref"], n)
        return _out(f"[run {row['name']} logs]\n{_cap(str(text))}", OutcomeStatus.OK, changed=False)
    state, note, detail = await _probe(context, row, origin)
    if row["ended_at"] is not None and state != "unknown":
        state = row["state"]                       # r2 minor 4: its recorded end, not a re-probe's guess
    if state == "unknown":
        return _out(f"RUN ERROR: could not read the logs — {note}.", OutcomeStatus.FAILED, changed=False,
                    code="run_unreachable")
    tail = "\n".join(str(detail.get("output_tail") or "").splitlines()[-n:])
    return _out(f"[run {row['name']} logs — {state}]\n{_cap(tail)}", OutcomeStatus.OK, changed=False)


def _delivers(jobs, job_id: str) -> bool:
    """Is THIS turn the one a job's result is delivered to — the owner's turn (or internal code), or the request
    that started the job (the jobs tool's §4NS rule)? Never raises (False)."""
    try:
        from ..utils.logging import request_id_context, request_kind
        caller = str(request_id_context.get() or "")
        if not caller or request_kind(caller) in ("owner", "system"):
            return True
        if request_kind(caller) in ("job", "background") and caller.endswith(str(job_id)):
            return True                       # review r3: the job's own wake (`job-<id>`, `sub-job-<id>`) — §4NS _mine
        rec = (jobs._load_launches() or {}).get(str(job_id))
        return isinstance(rec, dict) and str(rec.get("started_by") or "") == caller
    except Exception:  # noqa: BLE001
        return False


async def _collect(context, reg, row, origin) -> ToolOutcome:
    if row["lifetime"] != "job":
        return _refuse(f"{row['name']!r} is a {row['lifetime']} — `collect` reads a finished job; use `logs`")
    if row["ended_at"] is not None:
        # r2 minor 4: an ended run keeps its recorded end (a stopped lab job re-probes as `lost`);
        # the probe only fetches its output
        _s, _n, detail = await _probe(context, row, origin)
        if _s == "unknown":
            return _out(f"[run {row['name']} {row['state']}] could not read its result now ({_n}).",
                        OutcomeStatus.UNRESOLVED, changed=False, code="run_unreachable")
        state, note = row["state"], row["state_note"]
        if detail.get("exit_code") is None:
            # r3 minor 1: the recorded end ("exit N") when the backend no longer has the code
            import re as _re
            _m = _re.fullmatch(r"exit (-?\d+)", str(note or "").strip())
            if _m:
                detail = {**detail, "exit_code": int(_m.group(1))}
    else:
        state, note, detail = await _probed(context, reg, row, origin)
    if state in ("running", "unknown"):
        return _out(f"[run {row['name']} — {state}{(': ' + note) if note else ''}] no result yet; collect it "
                    "when it has ended.", OutcomeStatus.UNRESOLVED, changed=False, code="run_still_running")
    code = detail.get("exit_code")
    if row["backend"] == "jobs":
        _, jobs = _supervisors(context)
        tail = await asyncio.to_thread(jobs.log_tail, row["backend_ref"], 80) if jobs is not None else ""
        # review (§4NS): the SAME exit reading as the wake and the old `jobs collect` — the stage-status marker is
        # ours, and a `| head` / `find … 2>/dev/null` that exited 1/141 with output is not a failure
        try:
            from .execute import _normalise_exit, _take_pipestatus
            tail, _ = _take_pipestatus(str(tail or ""))
            if isinstance(code, int) and code != 0:
                code = _normalise_exit(row["command"] or "", code, tail)
        except Exception:  # noqa: BLE001
            logger.debug("collect: exit not normalised", exc_info=True)
        if jobs is not None and _delivers(jobs, row["backend_ref"]):
            # review (§4NS probe-d28bd3a4): a job its owner or its starter collected is delivered — its wake must
            # not report it again. review r2: anyone else's collect (a leaf, a scheduled task reading the owner's
            # job) leaves the wake, which carries the owner's waiting work on
            try:
                await asyncio.to_thread(jobs.mark_woken, row["backend_ref"])
            except Exception:  # noqa: BLE001
                logger.debug("collect: wake not marked", exc_info=True)
    else:
        tail = str(detail.get("output_tail") or "")
    head = (f"[run {row['name']} {state}" + (f" exit={code}" if code is not None else "")
            + (" — the exit code was not recorded: judge it by the output" if state == "exited" and code is None
               else "") + "]")
    ok = state == "exited" and code == 0
    # review r3: an `exited` job whose exit nobody knows is unresolved, not failed (the stored verdict agrees)
    unknown = state == "exited" and code is None
    return _out(f"{head}\nEXIT CODE: {code if code is not None else 'none'}\n{_cap(str(tail))}",
                OutcomeStatus.OK if ok else OutcomeStatus.UNRESOLVED if unknown else OutcomeStatus.FAILED,
                changed=False, code=None if ok else "run_exit_unknown" if unknown else "run_job_failed")


async def _app_url(context, reg, row, origin) -> ToolOutcome:
    state, note, detail = await _probed(context, reg, row, origin)
    if state == "unknown":
        return _out(f"RUN ERROR: could not read {row['name']!r}'s address — {note}.", OutcomeStatus.UNRESOLVED,
                    changed=False, code="run_unreachable")
    if not detail.get("url"):
        return _out(f"{row['name']}: {state}" + (f" ({note})" if note else "") + " — no published address.",
                    OutcomeStatus.OK, changed=False)
    return _out(f"{row['name']}: {detail['url']} (stable address on the tailnet; {state}"
                + (f", {note}" if note else "") + ")", OutcomeStatus.OK, changed=False)


async def _app_logs(context, row, origin, lines) -> ToolOutcome:
    from .labs import LabsNoAnswer, LabsUnreachable, _call
    state, note, detail = await _probe(context, row, origin)
    if not detail.get("lab_id"):
        return _out(f"RUN ERROR: {row['name']!r} has no serving lab ({state}{': ' + note if note else ''}).",
                    OutcomeStatus.FAILED, changed=False, code="run_unreachable")
    try:
        st, body = await _call(context, "GET", f"/v1/labs/{detail['lab_id']}/logs", origin=origin,
                               params={"member": detail.get("member"), "tail": _lines(lines)})
    except (LabsUnreachable, LabsNoAnswer) as e:
        return _out(f"RUN ERROR: could not read the logs — {e}.", OutcomeStatus.FAILED, changed=False,
                    code="run_unreachable")
    if st != 200 or not isinstance(body, dict):
        return _out(f"RUN ERROR: the labs host answered {st} for the logs.", OutcomeStatus.FAILED, changed=False,
                    code="run_unreachable")
    return _out(f"[run {row['name']} logs — {state} on {detail['lab_id']}/{detail.get('member')}]\n"
                + _cap(str(body.get("logs") or "")), OutcomeStatus.OK, changed=False)


def _url(row) -> ToolOutcome:
    if row["lifetime"] != "server":
        return _refuse(f"{row['name']!r} is a {row['lifetime']} — only a server has an address")
    if not row["port"]:
        return _out(f"{row['name']}: a portless server — it has no address.", OutcomeStatus.OK, changed=False)
    if row["place"] == "workspace":
        return _out(f"{row['name']}: http://127.0.0.1:{row['port']}/ inside the workspace (the browser tool "
                    "reaches it; it is not published on the tailnet).", OutcomeStatus.OK, changed=False)
    lab_id, _, member = row["place"].partition("/")
    return _out(f"{row['name']}: not exposed (a disposable lab). Probe it with `lab http lab_id={lab_id} "
                f"member={member} port={row['port']}`.", OutcomeStatus.OK, changed=False)


async def _stop(context, reg, row, origin) -> ToolOutcome:
    svc, jobs = _supervisors(context)
    be = row["backend"]
    # probe FIRST (review r1 M1, minor 1): a run that already ended is reported as it is,
    # never "stopped" over its real end, and nothing is sent by name to a backend
    state, note, _ = await _probe(context, row, origin)
    if state == "replaced":
        reg.set_state(row["run_id"], "replaced", note)
        return _refuse(f"{row['name']!r} was replaced by a start outside `run` — not stopping that one",
                       "run_replaced")
    if state in _ENDED:
        reg.set_state(row["run_id"], state, note)
        return _out(f"[run {row['name']}] already {state}" + (f" ({note})" if note else "") + " — nothing to stop.",
                    OutcomeStatus.OK, changed=False)
    if be == "labs_app":
        if state in ("app_stopped", "staged"):
            return _out(f"[run {row['name']}] already {state} — nothing to stop.", OutcomeStatus.OK, changed=False)
        return await _app_op(context, reg, row, origin, "stop")
    if be == "services":
        if svc is None:
            return _out("RUN ERROR: the sandbox is not running.", OutcomeStatus.FAILED, changed=False,
                        code="sandbox_unavailable")
        text = await asyncio.to_thread(svc.stop, row["backend_ref"])
    elif be == "jobs":
        if jobs is None:
            return _out("RUN ERROR: the sandbox is not running.", OutcomeStatus.FAILED, changed=False,
                        code="sandbox_unavailable")
        _ok, text = await asyncio.to_thread(jobs.cancel, row["backend_ref"])
        if not _ok:
            # r2 minor 3: it may have ended on its own between the probe and the cancel — record
            # THAT end; but a refused kill ("could not kill … left RUNNING", r3 MAJOR 1) is a failure
            st2, n2, _ = await _probe(context, row, origin)
            if st2 in _ENDED:
                reg.set_state(row["run_id"], st2, n2)
                return _out(f"[run {row['name']}] already {st2}" + (f" ({n2})" if n2 else "") + f" — {text}",
                            OutcomeStatus.OK, changed=False)
            if st2 == "running":
                return _out(f"RUN ERROR: {row['name']!r} is still running — the stop failed: {text}",
                            OutcomeStatus.FAILED, changed=False, code="run_stop_failed")
            return _out(f"[run {row['name']}] the stop was refused ({text}) and the check got no answer "
                        f"({n2}) — `run status name={row['name']}` shows it.", OutcomeStatus.UNRESOLVED,
                        changed=None, code="run_unverified")
    else:
        from .labs import LabsNoAnswer, LabsUnreachable, _call, _unreachable
        lab_id, _, job_id = row["backend_ref"].partition("/")
        try:
            st, body = await _call(context, "DELETE", f"/v1/labs/{lab_id}/jobs/{job_id}", origin=origin)
        except LabsUnreachable as e:
            return _unreachable(context, str(e))
        except LabsNoAnswer as e:
            return _out(f"RUN ERROR (no_answer): the labs host did not answer ({e}) — it may have stopped; "
                        "check with `run status`.", OutcomeStatus.UNRESOLVED, changed=None, code="run_no_answer")
        body = body if isinstance(body, dict) else {}
        if st == 404 and body.get("error") == "unknown_lab":
            from .labs import lab_on_other_host
            other = await lab_on_other_host(context, lab_id, origin)
            if other:                                            # review 11-05 M3
                from .labs import other_host_phrase
                return _out(f"RUN ERROR: run {row['name']} is on a lab made on {other_host_phrase(other)} — it was "
                            "NOT stopped.", OutcomeStatus.FAILED, changed=False,
                            code="run_stop_failed")
            reg.set_state(row["run_id"], "lost", "its lab was destroyed")
            return _out(f"[run {row['name']}] already gone: its lab was destroyed.", OutcomeStatus.OK, changed=False)
        if st >= 300:
            return _out(f"RUN ERROR: the labs host refused the stop ({st} {body.get('error')}).",
                        OutcomeStatus.FAILED, changed=False, code="run_stop_failed")
        # the DELETE's own answer decides (review r1 minor 2: a later GET never says `killed`)
        if body.get("state") == "running":
            return _out(f"RUN ERROR: {row['name']!r} is still running after the stop.", OutcomeStatus.FAILED,
                        changed=None, code="run_stop_failed")
        reg.set_state(row["run_id"], "stopped", "killed" if body.get("killed") else "")
        pretty_log("Run Stop", f"{row['name']} {row['lifetime']} {row['place']} [{row['run_id']}] by {origin}",
                   icon=Icons.STOP)
        return _out(f"[run {row['name']} stopped]\n{job_id} {body.get('state')}", OutcomeStatus.OK, changed=True)
    state, note, _ = await _probe(context, row, origin)
    if state in ("running",):
        return _out(f"RUN ERROR: {row['name']!r} is still running after the stop.\n{_cap(str(text))}",
                    OutcomeStatus.FAILED, changed=None, code="run_stop_failed")
    if state == "unknown":
        return _out(f"[run {row['name']}] stop sent, but the backend did not answer the check ({note}) — "
                    f"`run status name={row['name']}` shows it.\n{_cap(str(text))}", OutcomeStatus.UNRESOLVED,
                    changed=None, code="run_unverified")
    # r4 minor 1: `svc.stop` removes the service row, so the check reads `lost` — that is THIS stop
    reg.set_state(row["run_id"], "stopped", "stopped by run" if state == "lost" else note)
    pretty_log("Run Stop", f"{row['name']} {row['lifetime']} {row['place']} [{row['run_id']}] by {origin}",
               icon=Icons.STOP)
    return _out(f"[run {row['name']} stopped]\n{_cap(str(text))}", OutcomeStatus.OK, changed=True)


async def _restart(context, reg, row, origin) -> ToolOutcome:
    if row["backend"] == "labs_app":
        return await _app_op(context, reg, row, origin, "restart")
    if row["lifetime"] == "job":
        return _refuse("a job is not restarted — `run start` it again (a job runs once)")
    if row["backend"] == "services":
        svc, _ = _supervisors(context)
        if svc is None:
            return _out("RUN ERROR: the sandbox is not running.", OutcomeStatus.FAILED, changed=False,
                        code="sandbox_unavailable")
        # r12: a restart runs the stored command again — under the start's guards as they are NOW (a project
        # released since the start, a deny-list rule added)
        _blocked = _shell_guard_refusal(context, row["owner"], row["command"] or "") or await _egress_refusal(context)
        if _blocked is not None:
            return _blocked
        state, note, _ = await _probe(context, row, origin)
        if state in ("replaced", "lost"):
            reg.set_state(row["run_id"], state, note)
            return _refuse(f"{row['name']!r} is {state} ({note}) — not restarting by name (it could reach "
                           "another project's service)", "run_replaced" if state == "replaced" else "run_ended")
        async def _go():
            # shielded (review r2 M2): restart → record completes even if the turn is cancelled
            if _stop_asked():
                return "not_started", _NOT_STARTED, None
            try:
                text = await asyncio.to_thread(svc.restart, row["backend_ref"])
                e = {x.get("key"): x for x in await asyncio.to_thread(svc.list_entries)}.get(row["backend_ref"])
                alive = await asyncio.to_thread(svc._entry_state, e) if e is not None else False
            except Exception as ex:  # noqa: BLE001
                return "error", f"{type(ex).__name__}: {ex}", None
            live_row = row["ended_at"] is None          # r4 minor 3: never rewrite a recorded end
            new_token = bool(e and e.get("token") and str(e.get("token")) != row["backend_token"])
            if new_token and alive is not False:
                rid = reg.add(name=row["name"], place="workspace", lifetime="server", command=row["command"],
                              port=e.get("port"), owner=row["owner"], origin=origin, backend="services",
                              backend_ref=row["backend_ref"], backend_token=str(e.get("token")),
                              # review: a leaf's server keeps its leaf tag across a restart (else its end
                              # skipped it — a workspace server has no TTL)
                              meta={"leaf": _row_leaf(row) or None})
                if rid is None:
                    # r3 minor 2: never left running untracked
                    gone = await _svc_undo(svc, row["backend_ref"])
                    if live_row:
                        reg.set_state(row["run_id"], "stopped" if gone else "lost",
                                      "restarted, then stopped: not recorded" if gone
                                      else "restarted but not recorded, and the stop failed")
                    return "unrecorded", text, gone
                if live_row:
                    reg.set_state(row["run_id"], "stopped", f"restarted as {rid}")
                return "ok", text, (e, rid)
            if not new_token and alive is True:
                # r2 M1: the old process survived the kill — the run is still RUNNING, nothing changed
                if row["ended_at"] is None:
                    reg.set_state(row["run_id"], "running", "a restart could not stop it", revive=True)
                return "survived", text, None
            if alive is None:
                return "unverified", text, None          # r4 minor 3: no answer is not "failed"
            if live_row:
                reg.set_state(row["run_id"], "failed", "the restart did not come up")
            return "failed", text, None
        kind, text, extra = await asyncio.shield(_go())
        if kind == "ok":
            e, rid = extra
            return _out(f"[run {row['name']} restarted — port {e.get('port')}, {rid}]\n{_cap(str(text))}",
                        OutcomeStatus.OK, changed=True)
        if kind == "survived":
            return _out(f"RUN ERROR: the restart could not stop the running server — it is still running as "
                        f"before.\n{_cap(str(text))}", OutcomeStatus.FAILED, changed=False, code="run_restart_failed")
        if kind == "not_started":
            return _out(text, OutcomeStatus.REJECTED, changed=False, code="run_stopped_by_user")
        if kind == "unverified":
            return _out(f"[run {row['name']}] restart sent, but the sandbox did not answer the check — "
                        f"`run status name={row['name']}` shows it.\n{_cap(str(text))}", OutcomeStatus.UNRESOLVED,
                        changed=None, code="run_unverified")
        if kind == "unrecorded":
            what = ("so the server was stopped again" if extra else
                    f"and stopping it again FAILED — it may still run as service {row['backend_ref']}")
            return _out(f"RUN ERROR: the run registry did not record the restart, {what}.",
                        OutcomeStatus.FAILED, changed=True, code="run_registry_unavailable")
        return _out(f"RUN ERROR: the restart did not come up.\n{_cap(str(text))}", OutcomeStatus.FAILED,
                    changed=None, code="run_start_failed")
    # a lab server: stop, then start the same command again
    if origin in ("autopilot", "probe"):
        # r12 F5: the start's own rule is checked BEFORE the stop — it was after, so a refused start left the
        # server stopped ("The old server WAS stopped; starting it again failed: … KEPT")
        from .labs import LabsNoAnswer, LabsUnreachable, _call
        _lid = row["place"].partition("/")[0]
        try:
            _st, _det = await _call(context, "GET", f"/v1/labs/{_lid}", origin=origin)
        except (LabsUnreachable, LabsNoAnswer) as e:
            return _out(f"RUN ERROR (no_answer): the labs host did not answer ({e}) — nothing was stopped or "
                        "started.", OutcomeStatus.FAILED, changed=False, code="run_no_answer")
        if _st == 200 and isinstance(_det, dict):
            _no = _lab_start_refusal(_det, origin, row["owner"], _lid)
            if _no is not None:
                return _no
    stopped = await _stop(context, reg, row, origin)
    if stopped.status is not OutcomeStatus.OK:
        return stopped
    lab_id, _, member = row["place"].partition("/")
    # review r2: a restart is the same run — it keeps the old row's leaf on both paths (workspace and lab)
    started = await _start_lab(context, reg, row["owner"], origin, row["name"], row["command"], lab_id, member,
                               row["lifetime"], row["port"], meta={"leaf": _row_leaf(row) or None})
    if started.status is not OutcomeStatus.OK:
        was = bool(getattr(stopped, "world_changed", False))
        lead = ("The old server WAS stopped; starting it again failed:" if was
                else "The old server had already ended; starting it again failed:")
        return ToolOutcome(f"{lead}\n{started}", status=started.status, reason_code=started.reason_code,
                           world_changed=True if was else started.world_changed)
    return started


# ------------------------------------------------------------------ reconcile + boot line (design C3/C5)

async def reconcile(context) -> Dict[str, Any]:
    """Check the run registry against the workspace supervisors (design C3): every live WORKSPACE row is
    probed now and the probe written back (a backend row that vanished → `lost`); lab rows are only counted —
    nothing reads a lab in the background (design r1 M4: a read touches the lab and keeps it alive).
    Rows of the old tools are REPORTED, never adopted: services with no run pointing at them, and
    `run`-started jobs (``started_via="run"``) no live run points at."""
    from ..memory.run_registry import registry_for
    out: Dict[str, Any] = {"live": 0, "probed": 0, "unknown": 0, "ended": {}, "labs": 0,
                           "untracked_services": [], "orphan_run_jobs": []}
    reg = registry_for(context)
    if reg is None:
        out["error"] = "the run registry is unavailable"
        return out
    rows = reg.rows(live_only=True)
    out["live"] = len(rows)
    for r in rows:
        if r["backend"] not in ("services", "jobs"):
            out["labs"] += 1
            continue
        try:
            state, note, _ = await _probed(context, reg, r, "autopilot")
        except Exception as e:  # noqa: BLE001 — review: one raising probe must not end the whole pass
            state, note = "unknown", f"{type(e).__name__}: {e}"
        out["probed"] += 1
        if state == "unknown":
            out["unknown"] += 1
        elif state in _ENDED:
            out["ended"][r["run_id"]] = state + (f" ({note})" if note else "")
    svc, jobs = _supervisors(context)
    # the rows live at the START of this pass (review: a run server that just read `exited` still has its
    # service row — that row is the run's own, not one "outside run")
    # (r2: plus the rows live NOW — a `run start` that landed during this pass is its own run, not a stray)
    live_refs = {(r["backend"], r["backend_ref"]) for r in rows + reg.rows(live_only=True)}
    if svc is not None:
        try:
            for e in await asyncio.to_thread(svc.list_entries):
                k = str(e.get("key") or "")
                if k and ("services", k) not in live_refs:
                    out["untracked_services"].append(k)
        except Exception as e:  # noqa: BLE001
            out["error"] = f"services: {type(e).__name__}: {e}"
    if jobs is not None:
        try:
            for e in await asyncio.to_thread(jobs.list_entries):
                if (e.get("started_via") == "run" and e.get("state") == "running"
                        and ("jobs", str(e.get("id"))) not in live_refs):
                    out["orphan_run_jobs"].append(str(e.get("id")))
        except Exception as e:  # noqa: BLE001
            out["error"] = f"jobs: {type(e).__name__}: {e}"
    return out


def reconcile_line(rep: Dict[str, Any]) -> str:
    if rep.get("error") == "the run registry is unavailable":
        return "run registry unavailable"
    ended = rep["ended"]
    bits = [f"{rep['live']} live run(s)",
            f"workspace {rep['probed']} re-probed"
            + (f" ({len(ended)} ended: " + ", ".join(f"{k} {v}" for k, v in list(ended.items())[:4]) + ")"
               if ended else "")
            + (f", {rep['unknown']} unanswered" if rep["unknown"] else ""),
            f"lab {rep['labs']} not probed"]
    if rep["untracked_services"]:
        bits.append(f"{len(rep['untracked_services'])} service(s) outside run")
    if rep["orphan_run_jobs"]:
        bits.append(f"run job(s) with no run row: {', '.join(rep['orphan_run_jobs'][:4])}")
    if rep.get("error"):
        bits.append(f"partial: {rep['error']}")
    return " · ".join(bits)


_BOOT_TASKS: set = set()


def start_runs_boot_line(context) -> None:
    """Schedule the one startup line (design C5: a boot line, no boot adoption) on the running loop."""
    t = asyncio.get_running_loop().create_task(runs_boot_line(context))
    _BOOT_TASKS.add(t)
    t.add_done_callback(_BOOT_TASKS.discard)


async def runs_boot_line(context) -> None:
    try:
        rep = await reconcile(context)
        warn = bool(rep["ended"] or rep["orphan_run_jobs"] or rep.get("error"))
        pretty_log("Runs", reconcile_line(rep), level="WARNING" if warn else "INFO",
                   icon=Icons.WARN if warn else Icons.SANDBOX_BOX)
    except Exception as e:  # noqa: BLE001 — a background boot task never raises
        pretty_log("Runs", f"reconcile failed: {type(e).__name__}: {e}", level="WARNING", icon=Icons.WARN)


# ------------------------------------------------------------------ kept-app down notice (§4SX D3 D, design r1 M8)

#: seconds between sweeps of the kept apps
APP_WATCH_EVERY_S = 300


def _app_is_down(state: str, detail: Dict[str, Any]) -> bool:
    """crashloop; no version serving (code r1 #9: a failed recovery left only staged versions); a host
    `restarting` (crashed, in backoff — two polls in a row is a slow crash loop, r1 #9); stopped with an error
    the owner did not cause."""
    if state in ("crashloop", "staged") or detail.get("app_state") == "restarting":
        return True
    return state == "app_stopped" and bool(detail.get("last_error")) and not detail.get("stopped_by_owner")


#: kept labs already looked at by the adopt step (lab_id) — a lab is read once per process
_ADOPT_SEEN: set = set()


def _project_known(context, owner: str) -> Optional[bool]:
    """True: the store has ``project:<id>``; False: the store POSITIVELY has no such row; None: cannot tell (no store,
    an error, not a project owner)."""
    pid = owner[len("project:"):] if owner.startswith("project:") else ""
    get = getattr(getattr(context, "project_store", None), "get_project", None)
    if not pid or not callable(get):
        return None
    try:
        return bool(get(pid))
    except Exception:  # noqa: BLE001
        return None


def _is_own_project(context, owner: str) -> bool:
    """True only when ``project:<id>`` is a project in THIS agent's store (fails closed: no store, an error or no row
    → False). Review 11-05 r17: the labs host is shared (its own integration tests make kept `project:itest…` labs);
    the adopt step filed them as the owner's app runs, and the down sweep would notify the owner about them."""
    return _project_known(context, owner) is True


async def _adopt_app_rows(context, reg) -> int:
    """code r1 #3: a keep the agent never heard back from (or whose row was not written) leaves the host's app
    unknown to `run`, the release gate and this sweep — record a row for each kept project lab whose app has no
    live row. One host list per sweep; a lab's detail is read once per process. Never raises."""
    from .labs import _call, _record_app_run
    n = 0
    try:
        st, body = await _call(context, "GET", "/v1/labs", origin="autopilot")
        if st != 200 or not isinstance(body, dict):
            return 0
        for lab in body.get("labs") or []:
            if not isinstance(lab, dict) or str(lab.get("lifetime") or "") != "kept":
                continue
            lid, owner = str(lab.get("lab_id") or ""), str(lab.get("owner") or "")
            if not lid or lid in _ADOPT_SEEN or not owner.startswith("project:"):
                continue
            if not await asyncio.to_thread(_is_own_project, context, owner):
                continue                      # r17: not a project of this agent (another client of the host)
            st2, det = await _call(context, "GET", f"/v1/labs/{lid}", origin="autopilot")
            if st2 != 200 or not isinstance(det, dict):
                continue
            _ADOPT_SEEN.add(lid)
            a = det.get("app") if isinstance(det.get("app"), dict) else {}
            if not a.get("name"):
                continue
            have = reg.live_by_name(owner, str(a["name"]))
            if have is not None:
                continue                      # a row exists (this app's, or a clash `lab keep` already refused)
            await _record_app_run(context, str(det.get("origin") or "owner"), det)
            n += 1
            pretty_log("Runs", f"adopted kept app {a['name']} of {owner} ({lid}) — the keep's answer was lost",
                       icon=Icons.SANDBOX_BOX)
    except Exception as e:  # noqa: BLE001
        logger.warning("kept-app adopt skipped: %s: %s", type(e).__name__, e)
    return n


async def app_down_sweep(context) -> Dict[str, Any]:
    """Probe every live `app` run once. An app down on TWO polls in a row (a host restart heals in one tick) opens a
    down period; ONE owner notice per sweep names every app newly down; a period is marked notified only when the
    notice was written; the period closes when the app is up again. A host that does not answer changes nothing."""
    from ..memory.run_registry import RunRegistry, registry_for
    reg = registry_for(context)
    out = {"probed": 0, "down": [], "notified": False, "unknown": 0}
    if reg is None:
        return out
    await _adopt_app_rows(context, reg)
    # code r1 #10: a probe's test app never pages the owner
    rows = [r for r in reg.rows(live_only=True) if r["backend"] == "labs_app" and r.get("origin") != "probe"]
    # r17 (review): a row adopted before the adopt step checked the project (a shared host's test app filed as the
    # owner's) is ended here, never probed or paged about
    _known: Dict[str, Optional[bool]] = {}
    keep = []
    for r in rows:
        o = str(r.get("owner") or "")
        if o.startswith("project:"):
            if o not in _known:
                _known[o] = await asyncio.to_thread(_project_known, context, o)
            if _known[o] is False:          # the store POSITIVELY has no such project: never ours — end the row
                reg.set_state(r["run_id"], "lost", "not a project of this agent — its kept lab is another client's "
                              "of the labs host")
                continue
        keep.append(r)                      # ours, or cannot tell (no store / store error): probed as before
    rows = keep
    fresh = []
    now = time.time()
    for row in rows:
        state, note, detail = await _probe(context, row, "autopilot")      # reads only
        out["probed"] += 1
        if state == "unknown":
            out["unknown"] += 1
            continue
        reg.set_state(row["run_id"], state, note)
        m = RunRegistry.meta_of(row)
        if _app_is_down(state, detail):
            polls = int(m.get("down_polls") or 0) + 1
            reg.set_meta(row["run_id"], down_polls=polls, down_since=m.get("down_since") or now)
            out["down"].append(row["name"])
            if polls >= 2 and not m.get("down_notified"):
                fresh.append((row, state, note, detail))
        elif m.get("down_polls") or m.get("down_notified"):
            reg.set_meta(row["run_id"], down_polls=None, down_since=None, down_notified=None)
    if fresh:
        lines = [f"{r['name']} ({r['owner'].partition(':')[2] or r['owner']}): {st}"
                 + (f" — {n}" if n else "") + (f" {d.get('url')}" if d.get("url") else "") for r, st, n, d in fresh]
        msg = "Kept app DOWN — " + "; ".join(lines)
        if _notify_owner(context, msg[:500], [r["run_id"] for r, *_ in fresh]):
            out["notified"] = True
            for r, *_ in fresh:
                reg.set_meta(r["run_id"], down_notified=now)
        pretty_log("Kept App Down", msg[:300], icon=Icons.FAIL, level="WARNING")
    return out


def _notify_owner(context, msg: str, run_ids) -> bool:
    try:
        from ..core.autonomous_activity import SEVERITY_NOTIFY, get_activity_log
        from .notify_tool import PHASE as _PHASE, _note_sent, _rate_limited
        log = get_activity_log(context)
        if log is None or _rate_limited():
            return False
        ok = log.record(_PHASE, msg, severity=SEVERITY_NOTIFY, auto="kept app down", run_ids=list(run_ids))
        if ok:
            _note_sent()
        return bool(ok)
    except Exception:  # noqa: BLE001
        logger.warning("kept-app down notice not recorded", exc_info=True)
        return False


async def watch_kept_apps(context) -> None:
    """The background loop (main.py): one sweep every APP_WATCH_EVERY_S; never raises."""
    while True:
        await asyncio.sleep(APP_WATCH_EVERY_S)
        try:
            from .labs import labs_configured
            if labs_configured(context):
                await app_down_sweep(context)
        except Exception:  # noqa: BLE001
            logger.warning("kept-app sweep failed", exc_info=True)


# ------------------------------------------------------------------ the schema (§4SX D5: advertised; replaces
# manage_services + jobs, which stay as aliases for one release cycle)

RUN_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "run",
        "description": (
            "Everything that KEEPS RUNNING after this call returns — a web server or daemon (`server`), a long "
            "command (`job`), or a kept app (`app`). Each run has a NAME, unique in this project (or "
            "conversation), and lives WHERE you say: `workspace` (the sandbox) or `lab_<id>/<member>` (a lab "
            "machine from the `lab` tool). NEVER background a process in execute ('cmd &') — start it here. "
            "start: a `server` gets a port (the command reads $PORT; ask for one with `port`); a `job` runs to "
            "its end (45 min at most) and `collect` reads its result. An `app` is made by `lab keep` and is "
            "started/stopped/restarted here by name. status: every run with its probed state (or one by "
            "`name`). logs: the run's recent output. url: where a server/app answers. A start is NOT proof the "
            "work is done — check its logs/collect/url before saying it works."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": list(ACTIONS)},
                "name": {"type": "string",
                         "description": "The run's name, ^[a-z][a-z0-9-]{0,30}$ (e.g. `api`, `bench-1`). Required "
                                        "except for status."},
                "command": {"type": "string",
                            "description": "start: the foreground shell command, e.g. 'python3 -m http.server "
                                           "$PORT' or 'cd app && node server.js'."},
                "where": {"type": "string",
                          "description": "start: `workspace`, or `lab_<id>/<member>` for a lab machine. "
                                         "Required — there is no default."},
                "lifetime": {"type": "string", "enum": ["job", "server"],
                             "description": "start: `server` (keeps running, has a port) or `job` (runs to its "
                                            "end; read it with collect)."},
                "port": {"type": "integer",
                         "description": "start a workspace server: OPTIONAL preferred port (a free one is given "
                                        "and reported if it is taken); the command reads $PORT."},
                "lines": {"type": "integer", "description": "logs: how many recent lines (default 60)."},
            },
            "required": ["action"],
        },
    },
}


# ------------------------------------------------------------------ §4SX D5: promotion + the old names

def record_promoted_job(context, job: Optional[Dict[str, Any]]) -> str:
    """An `execute` command promoted to a background job is a `job` run named after its job id, so `run
    status/collect/stop name=<id>` read and stop it (D5-design). Returns the run name, or '' when no row was
    recorded (the banner then names the `jobs` way, which still works). Never raises."""
    try:
        jid = str((job or {}).get("id") or "")
        if not jid or not _NAME_RE.match(jid) or context is None:
            return ""
        origin, owner, refusal = _caller(context)
        if refusal or not owner:
            return ""
        from ..memory.run_registry import registry_for
        reg = registry_for(context)
        if reg is None:
            return ""
        if any(r["backend"] == "jobs" and r["backend_ref"] == jid for r in reg.rows(owner, live_only=True)):
            return jid                                  # a duplicate promotion of a job already recorded
        if job.get("duplicate"):
            return ""
        # review: NO leaf tag — a long `pip install`/`pytest` a leaf's execute detached must finish after the leaf
        # (its wake carries on), never be cancelled by the leaf's end
        rid = reg.add(name=jid, place="workspace", lifetime="job", command=str(job.get("command") or ""),
                      port=None, owner=owner, origin=origin or "autopilot", backend="jobs", backend_ref=jid,
                      meta={"started_leaf": _leaf_meta().get("leaf")})
        if rid is None:
            return ""
        return jid
    except Exception:  # noqa: BLE001
        logger.warning("promoted job not recorded as a run", exc_info=True)
        return ""


def run_name_for_job(context, job_id: str, owner: Optional[str] = None) -> str:
    """The run row's name for a workspace job id, live or ended ('' when none; review r2: the reaper ends the row
    before it writes the hint). ``owner`` given → only that owner's rows (review: another owner's row made `jobs
    collect` map to a `run` this caller cannot reach)."""
    try:
        from ..memory.run_registry import registry_for
        reg = registry_for(context)
        for r in reversed(reg.rows(owner) if reg is not None else []):
            if r["backend"] == "jobs" and r["backend_ref"] == str(job_id):
                # review r3: a REUSED name resolves (latest_by_name) to the newer job — then the name is not this
                # job's, and the caller falls back to the job id (`jobs collect` would have stopped the wrong job)
                cur = reg.latest_by_name(r["owner"], r["name"])
                return str(r["name"]) if cur is not None and cur["run_id"] == r["run_id"] else ""
    except Exception:  # noqa: BLE001
        pass
    return ""


def _with_hint(out, hint: str) -> ToolOutcome:
    """The alias's one-line hint BELOW the `run` result (review r2: above it, the hint moved the result's own head
    — `RUN ERROR`, `[run x exited]` — off the start every head reader anchors on); the status is kept."""
    out = _headed(out)
    # review r8 minor 3: every field kept (call_args/duration_s/declared were dropped here)
    return ToolOutcome(f"{out}\n{hint}", status=getattr(out, "status", OutcomeStatus.OK),
                       reason_code=getattr(out, "reason_code", None),
                       world_changed=getattr(out, "world_changed", None),
                       declared=getattr(out, "declared", True), call_args=getattr(out, "call_args", None),
                       duration_s=getattr(out, "duration_s", None))


def _alias_name(name: Any) -> str:
    n = re.sub(r"[^a-z0-9-]+", "-", str(name or "").strip().lower()).strip("-")
    n = re.sub(r"^[^a-z]+", "", n)[:31].rstrip("-")
    return n


async def manage_services_alias(context=None, action: str = None, name: str = None, command: str = None,
                                port=None, lines=None, workdir: str = None, **kwargs) -> ToolOutcome:
    """`manage_services` is not advertised any more (§4SX D5); a model that remembers it is mapped to `run`
    (one release cycle). start → `run start where=workspace lifetime=server`; stop/restart/logs/status → the
    same on `run`; anything else is refused with the `run` way to do it."""
    action = str(action or kwargs.get("operation") or kwargs.get("op") or "").strip().lower()
    name = name or kwargs.get("service") or kwargs.get("service_name")
    command = command or kwargs.get("cmd") or kwargs.get("script")
    port = port if port is not None else kwargs.get("service_port")
    if str(port).strip() in ("0", ""):
        port = None          # review 11-05 R-m10: manage_services' portless opt-out (port=0) — `run` assigns one
    lines = lines if lines is not None else (kwargs.get("tail") or kwargs.get("n"))
    if action in ("list", "ls", ""):
        action = "status"
    if action in ("log", "tail"):
        action = "logs"
    if action not in ("start", "stop", "restart", "logs", "status"):
        return _refuse(f"`manage_services {action}` is gone — use the `run` tool: `run status` lists every run; "
                       "`run stop name=…` stops one.", "run_alias_refused")
    rn = _alias_name(name) if name else ""
    if name and not _NAME_RE.match(rn or "-"):
        return _refuse(f"{name!r} is not a run name — use the `run` tool with a name like `api` "
                       "(^[a-z][a-z0-9-]{0,30}$).")
    if action == "start" and workdir and command:
        command = f"cd {shlex.quote(str(workdir))} && {command}"
    args: Dict[str, Any] = {"action": action}
    if rn:
        args["name"] = rn
    if action == "start":
        args.update(command=command, where="workspace", lifetime="server")
        if port is not None:
            args["port"] = port
    if action == "logs" and lines is not None:
        args["lines"] = lines
    out = await tool_run(context=context, **args)
    shown = " ".join(f"{k}={v}" for k, v in args.items() if k not in ("command",))
    renamed = f" (name {name!r} → {rn!r})" if name and rn != str(name) else ""
    return _with_hint(out, f"(`manage_services` is now `run` — this ran as `run {shown}`{renamed}; use `run` "
                           "next time.)")


async def jobs_alias(context=None, action: str = None, job_id=None, **kwargs) -> ToolOutcome:
    """`jobs` is not advertised any more (§4SX D5): status → `run status`; collect/cancel <job_id> → `collect`
    / `stop` on the run row of that job; a job with no run row is still read by the old jobs tool (a job
    promoted before the switch)."""
    action = str(action or kwargs.get("operation") or "status").strip().lower()
    job_id = str(job_id or kwargs.get("id") or kwargs.get("job") or "").strip()
    if action in ("list", "", "status") and not job_id:
        return _with_hint(await tool_run(context=context, action="status"),
                          "(`jobs` is now `run` — this ran as `run status`; use `run` next time.)")
    if action in ("result", "get", "fetch"):
        action = "collect"
    if action == "list":
        action = "status"
    want = {"status": "status", "collect": "collect", "cancel": "stop"}.get(action)
    if want is None:
        return _refuse(f"`jobs {action}` is gone — use the `run` tool (`run status`, `run collect name=…`, "
                       "`run stop name=…`).", "run_alias_refused")
    _o, _owner, _r = _caller(context)
    if _owner and not _r and want == "collect" and not job_id:
        # r13 M1: the `run collect` rule (r12 F6) — a bare collect in an owned turn took every project's results
        return _refuse("name the job to collect (`run status` lists this conversation's/project's runs; a background "
                       "job by its id: `run collect name=job-…`)", "run_name_required")
    if _owner and not _r and job_id:
        try:
            from ..memory.run_registry import registry_for
            _reg = registry_for(context)
            _ids = _job_refs(context, job_id)       # r15 M1: the registry id names the same job
            _theirs = [r for r in (_reg.rows(None) if _reg is not None else [])
                       if r["backend"] == "jobs" and r["backend_ref"] in _ids and r["owner"] != _owner]
            _mine_refs = [r["backend_ref"] for r in (_reg.rows(_owner) if _reg is not None else [])
                          if r["backend"] == "jobs" and r["backend_ref"] in _ids]
            _mine = bool(_mine_refs)
            if _mine_refs:
                job_id = _mine_refs[-1]               # this owner's row is filed under that id: act on IT
        except Exception:  # noqa: BLE001
            _theirs, _mine = [], False
        if _theirs and not _mine:
            # r13 M2: the r12 F4 rule — another owner's job is not reached through the old name either
            return _refuse(f"{job_id} is a run of {_theirs[-1]['owner']} — runs are read and stopped only from the "
                           "conversation or project that started them", "run_not_allowed")
    rn = run_name_for_job(context, job_id, _owner) if job_id and _owner and not _r else ""
    if not rn:
        row = _job_row(context, job_id, _owner) if job_id and _owner and not _r else None
        if row is not None:
            # review r4: the job's run name was REUSED by a newer job — act on THIS job's own row (the jobs tool
            # never held a `run`-started job, so the fallback below answered "no job …")
            return await _old_job_row_op(context, row, _o, want)
        if job_id:
            # r14: the same by-id route as `run … name=<job id>` (its rules, its grading) — the jobs tool's own
            # status ignored the id and listed every job
            return await _job_by_id(context, want, job_id)
        from .delegate import tool_jobs
        return await tool_jobs(action=action, job_id=None, context=context, **kwargs)
    return _with_hint(await tool_run(context=context, action=want, name=rn),
                      f"(`jobs` is now `run` — this ran as `run {want} name={rn}`; use `run` next time.)")


def _job_row(context, job_id: str, owner: str):
    """The newest run row (any state) of this owner whose backend is this workspace job, or None."""
    try:
        from ..memory.run_registry import registry_for
        reg = registry_for(context)
        for r in reversed(reg.rows(owner) if reg is not None else []):
            if r["backend"] == "jobs" and r["backend_ref"] == str(job_id):
                return r
    except Exception:  # noqa: BLE001
        pass
    return None


async def _old_job_row_op(context, row, origin, want: str) -> ToolOutcome:
    """`jobs collect/status/cancel job_id=<id>` on a job whose run name a newer job took (review r4)."""
    from ..memory.run_registry import registry_for
    reg = registry_for(context)
    hint = (f"(`jobs` is now `run`; the name {row['name']!r} now belongs to a newer run, so this read job "
            f"{row['backend_ref']} by its id.)")
    if want == "collect":
        return _with_hint(await _collect(context, reg, row, origin), hint)
    if want == "status":
        return _with_hint(_out("RUNS:\n" + _line(row, row["state"], row["state_note"], {}), OutcomeStatus.OK,
                               changed=False), hint)
    if row["ended_at"] is not None:
        return _refuse(f"job {row['backend_ref']} is not running — it is {row['state']}"
                       + (f" ({row['state_note']})" if row["state_note"] else "")
                       + f". `run stop name={row['name']}` would stop the NEWER run of that name.", "run_ended")
    # still live under a name a newer run took (nearly unreachable: a start ends an exited row first) — stopped
    # here by its own row, under the rules `run stop` applies
    if origin == "probe" and row["origin"] != "probe":
        return _refuse(f"job {row['backend_ref']} was started by {row['origin'] or 'the owner'} — a probe changes "
                       "only its own runs", "run_not_allowed")
    _leaf = str(_leaf_meta().get("leaf") or "")
    if _leaf and _leaf not in (_row_leaf(row), _row_started_leaf(row)):
        return _refuse(f"job {row['backend_ref']} was not started by this build", "run_not_allowed")
    return _with_hint(await _stop(context, reg, row, origin), hint)


_JOB_END = {"done": "exited", "expired": "expired", "cancelled": "stopped", "lost": "lost"}


def end_landed_job_runs(context, entries) -> int:
    """Review (§4SX D5 M5): a workspace job that landed ends its run row — nothing else did (the reaper and the wake
    never wrote back), so the briefing's RUNNING block listed finished jobs as running for good. Called with the
    reaper's landed entries (local registry only, no docker exec). Returns rows ended. Never raises."""
    n = 0
    try:
        from ..memory.run_registry import registry_for
        reg = registry_for(context)
        if reg is None:
            return 0
        landed = {str(e.get("id") or ""): e for e in (entries or []) if isinstance(e, dict)
                  and str(e.get("state") or "") in _JOB_END}
        if not landed:
            return 0
        for r in reg.rows(live_only=True):
            e = landed.get(str(r["backend_ref"] or "")) if r["backend"] == "jobs" else None
            if e is None:
                continue
            code = e.get("exit_code")
            reg.set_state(r["run_id"], _JOB_END[str(e.get("state"))], f"exit {code}" if code is not None else "",
                          only_live=True)
            n += 1
    except Exception:  # noqa: BLE001
        logger.warning("landed jobs not written to their runs", exc_info=True)
    return n


def project_running_summary(context, project_id: str) -> list:
    """§4SX D5: the project briefing's RUNNING block — the project's live run rows (the state each had when last
    probed, and when) and its live labs from the lab inventory. Local records only: no probe, no host call, no
    docker exec per turn (D6 M9). ``[{kind, name, lifetime, place, state, at, port, service_key}]``."""
    out: list = []
    if not project_id:
        return out
    owner = f"project:{project_id}"
    try:
        from ..memory.run_registry import registry_for
        reg = registry_for(context)
        _rows = reg.rows(owner, live_only=True) if reg is not None else []
        try:
            # a job that landed while no reaper ran (local registry read, no exec) ends here too
            _, _jobs = _supervisors(context)
            if _jobs is not None and any(r["backend"] == "jobs" for r in _rows):
                _ents = [e for e in _jobs.list_entries()
                         if str(e.get("id") or "") in {r["backend_ref"] for r in _rows if r["backend"] == "jobs"}]
                if end_landed_job_runs(context, _ents):
                    _rows = reg.rows(owner, live_only=True)
        except Exception:  # noqa: BLE001
            logger.debug("running summary: job states not read", exc_info=True)
        # review: the NEWEST twelve (oldest-first hid a new run behind old ones)
        for r in sorted(_rows, key=lambda x: float(x.get("started_at") or 0), reverse=True)[:12]:
            out.append({"kind": "run", "name": r["name"], "lifetime": r["lifetime"], "place": r["place"],
                        "state": r["state"], "at": r.get("last_probe_at"), "port": r.get("port"),
                        "service_key": r["backend_ref"] if r["backend"] == "services" else ""})
    except Exception:  # noqa: BLE001 — the briefing never breaks a turn
        logger.debug("running summary: runs skipped", exc_info=True)
    try:
        from ..memory.lab_inventory import inventory_for
        inv = inventory_for(context)
        for r in (inv.rows(owner, live_only=True) if inv is not None else [])[:8]:
            out.append({"kind": "lab", "name": r["lab_id"], "lifetime": "", "place": r.get("template") or "",
                        "state": "leftovers" if r.get("leftovers") else "", "at": None, "port": None,
                        "label": r.get("label") or "", "service_key": ""})
    except Exception:  # noqa: BLE001
        logger.debug("running summary: labs skipped", exc_info=True)
    return out
