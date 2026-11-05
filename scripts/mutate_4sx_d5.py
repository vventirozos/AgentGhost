#!/usr/bin/env python3
"""§4SX Phase D5 battery — the `run` switch (tools/runs.py aliases, promotion, leaf runs, owner-less turns, the
RUNNING block), its wiring (core/agent.py batch order, strikes, egress, isolation, composed skills, coding_loop,
outcome_heuristics, prompts, projects inventory, main reaper). Coverage-selected, parallel
(scripts/battery_cov.py); the real src/ is never mutated. No-op must SURVIVE, known-bad must DIE.
    python3 scripts/mutate_4sx_d5.py [-jN] [all|check|rerun ID …]
"""
import os, subprocess, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = 'src/ghost_agent/tools/runs.py'
RR = 'src/ghost_agent/memory/run_registry.py'
AG = 'src/ghost_agent/core/agent.py'
CL = 'src/ghost_agent/core/coding_loop.py'
OH = 'src/ghost_agent/distill/outcome_heuristics.py'
PR = 'src/ghost_agent/core/prompts.py'
PJ = 'src/ghost_agent/tools/projects.py'
MN = 'src/ghost_agent/main.py'
ST = 'src/ghost_agent/core/strikes.py'
EG = 'src/ghost_agent/memory/egress.py'
TF = 'src/ghost_agent/tools/tool_failure.py'
EVG = 'src/ghost_agent/core/evidence_gate.py'
SK = 'src/ghost_agent/core/strikes.py'
LB = 'src/ghost_agent/tools/labs.py'
IS = 'src/ghost_agent/core/isolation.py'
CS = 'src/ghost_agent/tools/composed_skills.py'
MUT = [
    ('CONTROL known-bad', R, '    if action not in ACTIONS:\n        return _refuse(f"unknown action', '    if False:\n        return _refuse(f"unknown action'),
    ('CONTROL no-op', R, '    """True for a `run` call that only reads (by its action)."""', '    """True for a `run` call that only reads (by its action)."""  # no-op'),
    # M1/M2 leaf tags
    ('R1 promoted-leaf-tag', R, 'meta={"started_leaf": _leaf_meta().get("leaf")})', 'meta={"started_leaf": _leaf_meta().get("leaf"), **_leaf_meta()})'),
    ('R2 ws-job-no-leaf', R, 'origin=origin, backend="jobs", backend_ref=str(entry["id"]), meta=_leaf_meta())', 'origin=origin, backend="jobs", backend_ref=str(entry["id"]))'),
    ('R3 ws-server-no-leaf', R, 'backend_token=str(row.get("token")), meta=_leaf_meta())', 'backend_token=str(row.get("token")))'),
    ('R5 restart-drops-leaf', R, '                              meta={"leaf": _row_leaf(row) or None})', '                              meta=None)'),
    ('R6 registry-meta-ignored', RR, '            _m = _json.dumps({k: v for k, v in (meta or {}).items() if v is not None}, sort_keys=True)', '            _m = "{}"'),
    ('R6b registry-keeps-none', RR, '{k: v for k, v in (meta or {}).items() if v is not None}', '{k: v for k, v in (meta or {}).items()}'),
    # M3 collect
    ('R7 collect-no-normalise', R, '                code = _normalise_exit(row["command"] or "", code, tail)', '                pass'),
    ('R8 collect-keeps-marker', R, '            tail, _ = _take_pipestatus(str(tail or ""))', '            pass'),
    ('R9 collect-no-woken', R, '                await asyncio.to_thread(jobs.mark_woken, row["backend_ref"])', '                pass'),
    # M4 owner-less
    ('R10 ownerless-refuse-job', R, '        if action in ("status", "collect", "stop") and re.fullmatch(r"job-[0-9a-f]{6,16}", _jn):', '        if False:'),
    ('R12 ownerless-stop-not-cancel', R, 'action={"stop": "cancel"}.get(action, action)', 'action=action'),
    ('R13 ownerless-status-refused', R, '        if action == "status" and not _jn:\n            from .delegate import tool_jobs', '        if False:\n            from .delegate import tool_jobs'),
    ('R18 name-any-owner', R, '        for r in reversed(reg.rows(owner) if reg is not None else []):\n            if r["backend"] == "jobs" and r["backend_ref"] == str(job_id):\n                # review r3', '        for r in reversed(reg.rows(None) if reg is not None else []):\n            if r["backend"] == "jobs" and r["backend_ref"] == str(job_id):\n                # review r3'),
    ('R19 alias-owner-ignored', R, 'rn = run_name_for_job(context, job_id, _owner) if job_id and _owner and not _r else ""', 'rn = run_name_for_job(context, job_id) if job_id else ""'),
    # minors
    ('R14 leaf-may-stop-any', R, '    if _leaf and action in ("stop", "restart") and _leaf not in', '    if False and _leaf and action in ("stop", "restart") and _leaf not in'),
    ('R16 no-lab-reach', R, '        _refused = lab_reach_refusal(context, {"command": command}, "run")', '        _refused = None'),
    ('R17 no-agent-port', R, '        if _command_probes_agent_port(command):', '        if False:'),
    ('R20 jobs-list-refused', R, '    if action == "list":\n        action = "status"', '    if False:\n        action = "status"'),
    ('R21 ms-log-refused', R, '    if action in ("log", "tail"):\n        action = "logs"', '    if False:\n        action = "logs"'),
    ('R36 autopilot-app-start', R, '            if origin == "autopilot":\n                return _refuse(_AUTOPILOT_APP, "run_not_allowed")', '            if False:\n                return _refuse(_AUTOPILOT_APP, "run_not_allowed")'),
    ('R37 autopilot-app-stop', R, '    if row["lifetime"] == "app" and origin == "autopilot" and action in ("stop", "restart"):', '    if False:'),
    ('R31 stop-is-a-read', R, 'READ_ACTIONS = frozenset({"status", "logs", "url", "collect"})', 'READ_ACTIONS = frozenset({"status", "logs", "url", "collect", "stop"})'),
    # M5 RUNNING
    ('R22 landed-not-ended', R, '            reg.set_state(r["run_id"], _JOB_END[str(e.get("state"))], f"exit {code}" if code is not None else "",', '            None and reg.set_state(r["run_id"], _JOB_END[str(e.get("state"))], f"exit {code}" if code is not None else "",'),
    ('R23 landed-any-state', R, '                  and str(e.get("state") or "") in _JOB_END}', '}'),
    ('R24 summary-oldest', R, 'key=lambda x: float(x.get("started_at") or 0), reverse=True)[:12]', 'key=lambda x: float(x.get("started_at") or 0), reverse=False)[:12]'),
    ('R25 summary-no-sync', R, '                if end_landed_job_runs(context, _ents):', '                if False:'),
    ('R26 reaper-no-writeback', MN, '                await asyncio.to_thread(end_landed_job_runs, context, changed)', '                pass'),
    ('R27 stale-mark-gone', PR, '                if r.get("service_key") and str(r["service_key"]) in _stale_keys:', '                if False:'),
    ('R27b services-listed-twice', PR, '    services = [s for s in (services or []) if str(s.get("key") or "") not in _run_keys]', '    services = list(services or [])'),
    ('R28 inventory-dup', PJ, '                           and str(e.get("id") or "") not in _run_jobs]', ']'),
    # classifier
    ('R29 classifier-status-ignored', OH, '        return str(getattr(_st, "value", _st)) not in ("ok", "unresolved")\n    h = _without_banner_lines(result)\n    # review r7', '        pass\n    h = _without_banner_lines(result)\n    # review r7'),
    ('R30 classifier-none-failed', OH, '    return code not in ("0", "none")', '    return code != "0"'),
    # wiring
    ('R32 batch-run-after-execute', AG, '    if name == "run":\n        if prev_name == "execute":\n            return True', '    if name == "run":\n        if prev_name == "execute":\n            return False'),
    ('R33 batch-lab-after-run', AG, '        return bool(m and lid and m.group(1) == lid) or act in _LAB_APP_ACTIONS', '        return act in _LAB_APP_ACTIONS'),
    ('R33b batch-read-blocks', AG, '    # prev_name == "run"\n    if _rd(pa):\n        return False', '    # prev_name == "run"\n    if False:\n        return False'),
    ('R34 leaf-runs-not-ended', CL, '                    await stop_leaf_runs(context, f"project:{_want}", str(leaf_id))', '                    pass'),
    ('R35 leaf-no-run', CL, 'LEAF_ALLOWED_TOOLS = ("file_system", "execute", "lab", "run")', 'LEAF_ALLOWED_TOOLS = ("file_system", "execute", "lab")'),
    ('R38 strike-any-run-output', ST, '        return bool(_RUN_COLLECT_HEAD_RE.match(_o.strip()))', '        return True'),
    ('R39 egress-run-free', EG, '    "run": "url_literals",', ''),
    ('R40 replay-may-run', IS, '    "lab", "run",', '    "lab",'),
    ('R41 macro-may-run', CS, '    "run",           # §4SX D5: a process that keeps running is a turn\'s own work too', ''),
    # review r2
    ('R42 inflight-not-awaited', R, '    if pending:\n        # review r2 M1', '    if False:\n        # review r2 M1'),
    ('R43 inflight-not-registered', R, '        _LEAF_INFLIGHT.setdefault(leaf, set()).add(t)', '        pass'),
    ('R44 delivers-anyone', R, '        if jobs is not None and _delivers(jobs, row["backend_ref"]):', '        if jobs is not None:'),
    ('R45 delivers-starter-ignored', R, '        return isinstance(rec, dict) and str(rec.get("started_by") or "") == caller', '        return False'),
    ('R45b delivers-background-ok', R, '        if not caller or request_kind(caller) in ("owner", "system"):', '        if True:'),
    ('R46 started-leaf-ignored', R, 'action in ("stop", "restart") and _leaf not in (_row_leaf(row), _row_started_leaf(row))', 'action in ("stop", "restart") and _leaf != _row_leaf(row)'),
    ('R47 promoted-no-started-leaf', R, 'meta={"started_leaf": _leaf_meta().get("leaf")})', 'meta=None)'),
    ('R48 name-live-only', R, '        for r in reversed(reg.rows(owner) if reg is not None else []):\n            if r["backend"] == "jobs" and r["backend_ref"] == str(job_id):\n                # review r3', '        for r in reversed(reg.rows(owner, live_only=True) if reg is not None else []):\n            if r["backend"] == "jobs" and r["backend_ref"] == str(job_id):\n                # review r3'),
    ('R49 lab-restart-context-leaf', R, 'row["port"], meta={"leaf": _row_leaf(row) or None})', 'row["port"])'),
    ('R50 only-live-ignored', RR, '+ (" AND ended_at IS NULL" if only_live else "")', '+ ""'),
    ('R51 landed-overwrites', R, '                          only_live=True)', '                          only_live=False)'),
    ('R52 hint-above', R, '    return ToolOutcome(f"{out}\\n{hint}",', '    return ToolOutcome(f"{hint}\\n{out}",'),
    ('R53 stored-expired-ok', OH, '    if state != "exited":\n        return True', '    if False:\n        return True'),
    ('R54 ownerless-stop-row-stale', R, '            if action == "stop":\n                # review r2', '            if False:\n                # review r2'),
    # review r3 + §4SX E
    ('R55 live-unknown-failed', R, 'OutcomeStatus.OK if ok else OutcomeStatus.UNRESOLVED if unknown else OutcomeStatus.FAILED', 'OutcomeStatus.OK if ok else OutcomeStatus.FAILED'),
    ('R56 alias-rows-generic', OH, '    if _tn == "run" or (_tn in ("jobs", "manage_services")', '    if _tn == "run" or (False and _tn in ("jobs", "manage_services")'),
    ('R57 no-wake-arm', R, '        if request_kind(caller) in ("job", "background") and caller.endswith(str(job_id)):', '        if False:'),
    ('R58 name-reuse', R, '                return str(r["name"]) if cur is not None and cur["run_id"] == r["run_id"] else ""', '                return str(r["name"])'),
    ('R59 no-late-stop', R, '    if ended is not None and not t.cancelled():', '    if False:'),
    ('R60 no-sql-hint', 'src/ghost_agent/tools/labs.py', '        _sql = ("" if not ({"query", "sql", "connection_string"} & set(kwargs)) else', '        _sql = ("" if True else'),
    ('R61 postgres-alias-gone', AG, '        "postgresadmin": "lab",      # matched normalised: also postgres_admin / postgres-admin\n', ''),
    ('R62 postgres-short-alias-gone', AG, '        "postgres": "lab",\n', ''),
    # review r4
    ('R63 reused-old-row-not-used', R, '        if row is not None:\n            # review r4', '        if False:\n            # review r4'),
    ('R64 old-row-cancel-allowed', R, '    if row["ended_at"] is not None:\n        return _refuse(f"job {row[\'backend_ref\']} is not running', '    if False:\n        return _refuse(f"job {row[\'backend_ref\']} is not running'),
    ('R65 late-stops-unlocked', R, '    async with lock:\n        return await _stop_leaf_rows(context, owner, leaf_id, gone)', '    return await _stop_leaf_rows(context, owner, leaf_id, gone)'),
    ('R66 unknown-is-running', AG, '                                       and getattr(_outcome, "reason_code", None)\n                                       != "run_exit_unknown"),', '                                       ),'),
    ('R67 hint-reuse-label', MN, '        if (entry or {}).get("started_via") == "run":\n            # review r4', '        if False:\n            # review r4'),
    ('R69 unknown-head-note', R, '            + (" — the exit code was not recorded: judge it by the output" if state == "exited" and code is None\n               else "") + "]")', '            + "]")'),
    ('R70 unknown-still-in-ok', TF, '                   and not o.get("exit_unknown")]', '                   ]'),
    ('R71 unknown-line-dropped', TF, '    if unknown_ops:\n        stopped +=', '    if False:\n        stopped +='),
    ('R72 agent-exit-unknown-flag', AG, '"exit_unknown": getattr(_outcome, "reason_code", None) == "run_exit_unknown",', '"exit_unknown": False,'),
    ('R73 banner-not-stripped', OH, '    h = _without_banner_lines(result)\n    # review r7', '    h = str(result or "").strip()\n    # review r7'),
    ('R74 alias-banner-not-stripped', OH, 'and _without_banner_lines(result).startswith(("[run ",', 'and str(result).startswith(("[run ",'),
    ('R75 lock-not-loop-keyed', R, '_LEAF_STOP_LOCKS.setdefault((id(asyncio.get_running_loop()), str(leaf_id)), asyncio.Lock())', '_LEAF_STOP_LOCKS.setdefault((0, str(leaf_id)), asyncio.Lock())'),
    ('R76 reused-cancel-probe-guard', R, '    if origin == "probe" and row["origin"] != "probe":\n        return _refuse(f"job {row[\'backend_ref\']} was started by {row[\'origin\'] or \'the owner\'} — a probe changes "\n                       "only its own runs", "run_not_allowed")\n    _leaf = str(_leaf_meta().get("leaf") or "")\n    if _leaf and _leaf not in (_row_leaf(row), _row_started_leaf(row)):', '    _leaf = str(_leaf_meta().get("leaf") or "")\n    if _leaf and _leaf not in (_row_leaf(row), _row_started_leaf(row)):'),
    ('R76b reused-cancel-leaf-guard', R, '    if origin == "probe" and row["origin"] != "probe":\n        return _refuse(f"job {row[\'backend_ref\']} was started by {row[\'origin\'] or \'the owner\'} — a probe changes "\n                       "only its own runs", "run_not_allowed")\n    _leaf = str(_leaf_meta().get("leaf") or "")\n    if _leaf and _leaf not in (_row_leaf(row), _row_started_leaf(row)):', '    if origin == "probe" and row["origin"] != "probe":\n        return _refuse(f"job {row[\'backend_ref\']} was started by {row[\'origin\'] or \'the owner\'} — a probe changes "\n                       "only its own runs", "run_not_allowed")\n    _leaf = str(_leaf_meta().get("leaf") or "")\n    if False and _leaf not in (_row_leaf(row), _row_started_leaf(row)):'),
    ('R77 sql-hint-ignores-config', LB, '                if labs_configured(context) else', '                if True else'),
    ('R79 alias-runs-prefix', OH, '"RUN:", "RUNS", "No runs here"', '"RUN:", "No runs here"'),
    ('R80 finished-count-unknown', TF, 'for o in op_outcomes + unknown_ops) or "none"', 'for o in op_outcomes) or "none"'),
    ('R81 finished-n-unknown', TF, '{len(op_outcomes) + len(unknown_ops)} finished', '{len(op_outcomes)} finished'),
    ('R82 head-failed-off', R, '    if failed and not (text.startswith(_FAILED_HEAD) or _COLLECT_HEAD_RE.match(text)):', '    if False:'),
    ('R83 head-unresolved-off', R, '    elif not failed and text.startswith(_FAILED_HEAD):', '    elif False:'),
    ('R84 head-any-run-prefix', R, 'or _COLLECT_HEAD_RE.match(text)):', 'or text.startswith("[run ")):'),
    ('R85 hint-not-headed', R, '    out = _headed(out)\n    # review r8 minor 3', '    # review r8 minor 3'),
    ('R86 tool-run-not-headed', R, '    return _headed(await _tool_run(context, action,', '    return (await _tool_run(context, action,'),
    ('R87 head-drops-reason', R, 'world_changed=out.world_changed, reason_code=out.reason_code,', 'world_changed=out.world_changed, reason_code=None,'),
    ('R88 head-own-line', R, '        text = f"{_FAILED_HEAD} — {text}"', '        text = f"{_FAILED_HEAD} — not done:\\n{text}"'),
    ('R92 oh-logs-collect', OH, '_RUN_COLLECT_RE = re.compile(r"\\[run [a-z][a-z0-9-]{0,30} (?!logs\\b|app\\b)([a-z_]+)', '_RUN_COLLECT_RE = re.compile(r"\\[run [a-z][a-z0-9-]{0,30} ([a-z_]+)'),
    ('R93 strikes-logs-program', SK, '{0,30} (?!logs\\b|app\\b)[^\\]\\n]*', '{0,30} [^\\]\\n]*'),
    ('R94 hint-drops-fields', R, '                       declared=getattr(out, "declared", True), call_args=getattr(out, "call_args", None),\n                       duration_s=getattr(out, "duration_s", None))', '                       )'),
    ('R95 head-drops-world', R, '    return ToolOutcome(text, status=out.status, world_changed=out.world_changed,', '    return ToolOutcome(text, status=out.status, world_changed=None,'),
    ('R89 gate-unresolved-substantive', EVG, '        if (run_text_unresolved(text) or re.match', '        if (False or re.match'),
    ('R96 head-unresolved-prefix-off', R, '    elif (out.status is OutcomeStatus.UNRESOLVED and not text.startswith(_UNRESOLVED_HEAD)', '    elif (False and not text.startswith(_UNRESOLVED_HEAD)'),
    ('R97 head-unresolved-collect', R, '          and not _COLLECT_HEAD_RE.match(text)):\n', '          ):\n'),
    ('R98 stored-unresolved-off', OH, '    if run_text_unresolved(result):', '    if False:'),
    ('R99 unresolved-collect-none', OH, '    return bool(m and m.group(1) == "exited" and m.group(2) == "none")', '    return False'),
    ('R100 unresolved-banner', OH, '    h = _without_banner_lines(result)\n    if h.startswith("RUN UNRESOLVED"):', '    h = str(result or "").strip()\n    if h.startswith("RUN UNRESOLVED"):'),
    ('R101 unresolved-status-flip', OH, '        return str(getattr(_st, "value", _st)) == "unresolved"\n    h = _without_banner_lines(result)\n    if h.startswith("RUN UNRESOLVED")', '        return str(getattr(_st, "value", _st)) != "unresolved"\n    h = _without_banner_lines(result)\n    if h.startswith("RUN UNRESOLVED")'),
    ('R102 unresolved-gets-text', OH, '    if run_text_unresolved(result):', '    if run_text_unresolved(_t):'),
    ('R103 head-twice', R, '    elif (out.status is OutcomeStatus.UNRESOLVED and not text.startswith(_UNRESOLVED_HEAD)', '    elif (out.status is OutcomeStatus.UNRESOLVED'),
]
PINS = ["tests/test_4sx_d5_review.py", "tests/test_4sx_b_lab_tool.py", "tests/test_4sx_d5_switch.py", "tests/test_4sx_c_run_tool.py",
        "tests/test_4sx_c_run_battery.py"]


def _tests():
    pat = (r"tools.runs\|tools import runs\|run_registry\|_batch_call_depends\|outcome_heuristics\|"
           r"build_project_briefing\|_project_inventory\|LEAF_ALLOWED\|is_program_output\|egress\|REPLAY_FORBIDDEN\|"
           r"MACRO_FORBIDDEN\|_collect_hint\|run_leaf_turn\|_canonicalise_tool_name\|tool_lab")
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py"))),
                         capture_output=True, text=True).stdout.split()
    return sorted({os.path.relpath(t, ROOT) for t in out} | set(PINS))
def main():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import battery_cov as BC
    args = sys.argv[1:] or ["all"]
    workers = 6
    if args[0].startswith("-j"):
        workers = int(args[0][2:]); args = args[1:] or ["all"]
    if args[0] == "check":
        srcs = sorted({m[1] for m in MUT})
        originals = {f: open(os.path.join(ROOT, f)).read() for f in srcs}
        bad = [(lab, originals[f].count(old)) for lab, f, old, new in MUT if originals[f].count(old) != 1]
        print("anchors:", "OK" if not bad else bad); sys.exit(1 if bad else 0)
    batch = [m for m in MUT if m[0].split(" ")[0] in set(args[1:]) or m[0].startswith("CONTROL")] if args[0] == "rerun" else MUT
    full = _tests()
    srcs = sorted({m[1] for m in MUT})
    originals = {f: open(os.path.join(ROOT, f)).read() for f in srcs}
    bad = [lab for lab, f, old, new in batch if originals[f].count(old) != 1]
    if bad:
        print("ANCHOR NOT UNIQUE/FOUND:", bad); sys.exit(2)
    cov = BC.build_map(srcs, full, log=lambda m: print(m, flush=True))
    sel = {lab: BC.tests_for(cov, f, old, originals[f]) for lab, f, old, new in batch}
    union = sorted({i for v in sel.values() if v for i in v})

    def select(label, f, old):
        if label.startswith("CONTROL no-op"):
            return union, f"cov {len(union)} tests (all covering)"
        ids = sel.get(label)
        return (sorted(set(ids) | set(PINS)), f"cov {len(ids)} tests + pins") if ids else (PINS, "fallback pins")
    print(f"{len(full)} test files; {len(batch)} mutants; {len(union)} covering tests; {workers} trees", flush=True)
    res = BC.run_parallel(batch, originals, select, workers=workers, log=lambda m: print(m, flush=True))
    by = {}
    for _, st, _, _ in res:
        by[st] = by.get(st, 0) + 1
    print("SUMMARY " + " ".join(f"{k.lower()}={v}" for k, v in sorted(by.items())), flush=True)
    for lab, st, tier, tail in res:
        if st != "KILLED":
            print(f"  {st:8} {lab} — [{tier}] {tail}", flush=True)


if __name__ == "__main__":
    main()
