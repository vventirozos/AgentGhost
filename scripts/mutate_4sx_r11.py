#!/usr/bin/env python3
"""§4SX end-to-end review r11 battery (2026-11-05) — every agent-side fix of the re-review: lab put/get no-follow,
`lab list` never retiring unseen rows, other-host 404s, destroy no-answer, autopilot/probe walls, create texts,
batch order, run bash -c / shell guards / autopilot walls / job-id routing / revive, egress aliases, evidence gate,
the unanswered summary line, the boot leaf reaper, coding_loop leaf end, unattended project scope, deleted-project
folders. Coverage-selected, parallel (scripts/battery_cov.py); the real src/ is never mutated.
No-op must SURVIVE, known-bad must DIE.
    python3 scripts/mutate_4sx_r11.py [-jN] [all|check|rerun ID …]
"""
import os, subprocess, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LB = 'src/ghost_agent/tools/labs.py'
R = 'src/ghost_agent/tools/runs.py'
RR = 'src/ghost_agent/memory/run_registry.py'
AG = 'src/ghost_agent/core/agent.py'
CL = 'src/ghost_agent/core/coding_loop.py'
LR = 'src/ghost_agent/core/leaf_reaper.py'
MN = 'src/ghost_agent/main.py'
FS = 'src/ghost_agent/tools/file_system.py'
EG = 'src/ghost_agent/memory/egress.py'
EVG = 'src/ghost_agent/core/evidence_gate.py'
TF = 'src/ghost_agent/tools/tool_failure.py'
MUT = [
    ('CONTROL known-bad', LB, '        if action in KEPT_ACTIONS:                 # production: refused before any host call',
     '        if False:                 # production: refused before any host call'),
    ('CONTROL no-op', LB, '#: the id the configured labs host last gave on /v1/health', '#: the id the configured labs host last gave on /v1/health  (no-op)'),
    # --- lab tool
    ('L1 put-follows-link', LB, '            data = _read_inside(context, src, PUT_MAX_BYTES + 1)',
     '            data = src.read_bytes()[:PUT_MAX_BYTES + 1]'),
    ('L2 get-follows-link', LB, '            await _aio.to_thread(write_bytes_nofollow_rel, root, str(dst.relative_to(root)), data)   # off the loop', '            dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(data)'),
    ('L3 list-retires-other-host', LB,
     '                if not r.get("destroyed_at") and r.get("host_id") and r["host_id"] != host_id:',
     '                if False:'),
    ('L4 list-retires-unseen', LB, '                    out.append(await _settle_unlisted(context, origin, inv, r))',
     '                    inv.record_destroy(r["lab_id"], by="expired"); out.append("gone")'),
    ('L5 settle-ignores-200', LB,
     '        if st == 200:\n            return f"{lid} ({tpl}) — left out of the list, but the labs host still has it',
     '        if False:\n            return f"{lid} ({tpl}) — left out of the list, but the labs host still has it'),
    ('L6 settle-drops-leftovers', LB, '    if st == 200 and b.get("leftovers"):\n        inv.mark_leftovers',
     '    if False:\n        inv.mark_leftovers'),
    ('L7 destroy-noanswer-generic', LB, '        except LabsNoAnswer as e:            # nothing was deleted yet',
     '        except ZeroDivisionError as e:            # nothing was deleted yet'),
    ('L8 other-host-any-origin-closes', LB,
     '                if origin != "owner":\n                    return _service_reply(st, d, context, lab_id)',
     '                if False:\n                    return _service_reply(st, d, context, lab_id)'),
    ('L9 other-host-not-closed', LB, '                _inv0.record_destroy(lab_id, by="host_changed (closed by the owner)")',
     '                pass'),
    ('L10 probe-wall-off', LB,
     '        why = await _autopilot_owner_refusal(context, lab_id, owner, action, who="probe")',
     '        why = ""'),
    ('L11 autopilot-kept-open', LB,
     '    if (who == "autopilot" and st == 200 and isinstance(d, dict) and str(d.get("lifetime") or "") == "kept"',
     '    if (False and who == "autopilot" and st == 200 and isinstance(d, dict) and str(d.get("lifetime") or "") == "kept"'),
    ('L12 autopilot-status-refused', LB, '            and action != "status"):\n        return (f"{lab_id} is KEPT',
     '            ):\n        return (f"{lab_id} is KEPT'),
    ('L13 non200-someone-else', LB, '    if st not in (200, 404):\n        # review 11-05 m9',
     '    if False:\n        # review 11-05 m9'),
    ('L14 ready-404-not-failed', LB, '        elif st2 == 404 and isinstance(d2, dict) and d2.get("error") == "unknown_lab":\n            # review 11-05 m9', '        elif False:\n            # review 11-05 m9'),
    ('L15 evicted-not-recorded', LB,
     '                inv.record_destroy(eid, by="the labs host (evicted to make room for the owner\'s lab)")',
     '                pass'),
    ('L16 host-id-not-memoed', LB, '        _LAST_HOST_ID["id"] = hid', '        pass'),
    ('L17 other-host-no-ask', LB, '    now_id = _LAST_HOST_ID.get("id") or await _host_id(context, origin)',
     '    now_id = hid'),
    # --- run tool
    ('R1 job-no-bash', R, '            entry, reason = await asyncio.to_thread(jobs.start_detached, _bash_c(command), workdir=wd, label=name,',
     '            entry, reason = await asyncio.to_thread(jobs.start_detached, command, workdir=wd, label=name,'),
    ('R2 shell-guard-off', R, '        _blocked = _shell_guard_refusal(context, owner, command) or await _egress_refusal(context)', '        _blocked = await _egress_refusal(context)'),
    ('R3 validator-fail-open', R, '        ok, why = False, f"the shell validator failed ({type(e).__name__})"',
     '        ok, why = True, ""'),
    ('R4 daemon-unguarded', R, '(_daemonized_server_block, "run_daemon_blocked")):', '):'),
    ('R5 autopilot-other-lab', R, '        if not owner or theirs != owner:\n            return _refuse(f"{lab_id} belongs to',
     '        if False:\n            return _refuse(f"{lab_id} belongs to'),
    ('R6 autopilot-kept-lab', R, '        if str(det.get("lifetime") or "") == "kept":\n            return _refuse(f"{lab_id} is KEPT',
     '        if False:\n            return _refuse(f"{lab_id} is KEPT'),
    ('R8 owned-id-no-route', R, '            and reg.latest_by_name(owner, _jn) is None):\n        _ids', '            and False):\n        _ids'),
    ('R9 stop-by-id-row-live', R, '            await asyncio.to_thread(end_landed_job_runs, context, [_e])', '            pass'),
    ('R10 live-over-ended', RR, '                    c.execute("UPDATE runs SET state=?, state_note=?, last_probe_at=? WHERE run_id=? "\n                              "AND ended_at IS NULL"',
     '                    c.execute("UPDATE runs SET state=?, state_note=?, last_probe_at=? WHERE run_id=? "\n                              ""'),
    ('R11 revive-keeps-ended', RR, 'last_probe_at=?, ended_at=NULL WHERE run_id=?"', 'last_probe_at=? WHERE run_id=?"'),
    ('R12 restart-no-revive', R, 'reg.set_state(row["run_id"], "running", "a restart could not stop it", revive=True)',
     'reg.set_state(row["run_id"], "running", "a restart could not stop it")'),
    # --- wiring
    ('W1 files-unordered', AG, '    if act in ("put", "get", "deploy") and pact in ("put", "get", "deploy"):\n        # workspace files',
     '    if False:\n        # workspace files'),
    ('W2 lineage-unordered', AG, '    if act in _LAB_LINEAGE_ACTIONS and pact in _LAB_LINEAGE_ACTIONS:', '    if False:'),
    ('W4 from-lab-ignored', AG, '    if (flab and flab == plid) or (pflab and pflab == lid):', '    if False:'),
    ('W5 aliases-unordered', AG, '    if name not in ("manage_services", "jobs"):\n        return name, args',
     '    if True:\n        return name, args'),
    ('W6 unanswered-is-running', AG, '                                       and getattr(_outcome, "reason_code", None) not in _NO_ANSWER_CODES),',
     '                                       ),'),
    ('W7 egress-alias-free', EG, '    "manage_services": "url_literals",', ''),
    ('W8 run-start-evidence', EVG, '        if name == "run" and str((args or {}).get("action") or "").strip().lower() in ("start", "stop", "restart"):',
     '        if False:'),
    ('W9 unanswered-line-gone', TF, '    unanswered_ops = [o for o in op_outcomes if o.get("unanswered") and not o.get("interrupted")]',
     '    unanswered_ops = []'),
    # --- leaf / projects
    ('P1 reaper-no-runs', LR, '                done["runs"] += await stop_leaf_runs(context, owner, leaf)', '                pass'),
    ('P2 reaper-any-origin', LR, '            if leaf and r.get("origin") == "autopilot" and str(r.get("owner") or "").startswith("project:"):',
     '            if leaf and str(r.get("owner") or "").startswith("project:"):'),
    ('P3 reaper-apps', LR, '            if leaf and row.get("lifetime") != "app" and', '            if leaf and'),
    ('P4 seen-kept-across-attempts', CL, '        _drop_seen(leaf_base)', '        pass'),
    ('P6 deleted-dir-recreated', FS, '            if not sub.is_dir() and _project_row_gone(context, pid):',
     '            if False:'),
    # --- r12
    ('X1 ownership-noanswer-open', LB, '        return (f"the labs host did not answer the ownership check of {lab_id} ({e}) — nothing was sent; "\n                "try again")',
     '        return ""'),
    ('X3 probe-status-open', LB, '    if origin == "probe" and action not in ("templates", "create", "list") and (', '    if origin == "probe" and action not in ("templates", "create", "list", "status") and ('),
    ('X4 ownerless-probe-blocked', LB, '        if (st == 200 and isinstance(d, dict) and not str(d.get("owner") or "")\n                and str(d.get("origin") or "") == "probe"):\n            return ""',
     '        if False:\n            return ""'),
    ('X5 leaf-confine-off', LB, '    if getattr(context, "confine_writes_to_project", False) is True and p != scoped and scoped not in p.parents:',
     '    if False:'),
    ('X6 closed-says-destroyed', LB, '        if row and row.get("destroyed_at") and str(row.get("destroyed_by") or "").startswith("host_changed"):',
     '        if False:'),
    ('X7 ready-any-404', LB, '        elif st2 == 404 and isinstance(d2, dict) and d2.get("error") == "unknown_lab":', '        elif st2 == 404:'),
    ('X8 failed-build-leftovers-dropped', LB, '                if _left:                          # the row stays LIVE', '                if False:                          # the row stays LIVE'),
    ('X9 leaf-other-host-destroyed', LB, '            if st == 404 and await lab_on_other_host(context, lid):', '            if False:'),
    ('X10 fifo-blocks', FS, 'refuses any non-regular file\n        flags = (os.O_WRONLY | os.O_CREAT | os.O_TRUNC\n                 | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))', 'refuses any non-regular file\n        flags = (os.O_WRONLY | os.O_CREAT | os.O_TRUNC\n                 | getattr(os, "O_NOFOLLOW", 0))'),
    ('X11 nonregular-written', FS, '            if not _stat.S_ISREG(os.fstat(fd).st_mode):', '            if False:'),
    ('X12 deleted-project-writes', FS, '        _gone = _deleted_project_block(sandbox_dir, kwargs.get("project_store"))', '        _gone = None'),
    ('Y1 own-job-id-to-jobs', R, '        if _mine:\n            # review 11-05 r12 F1', '        if False:\n            # review 11-05 r12 F1'),
    ('Y2 other-owner-job-id', R, '        elif _refs:\n            # review 11-05 r12 F4', '        elif False:\n            # review 11-05 r12 F4'),
    ('Y3 owned-bare-collect', R, '    if action == "collect" and not _jn:\n        # review 11-05 r12 F6', '    if False:\n        # review 11-05 r12 F6'),
    ('Y6 egress-off', R, '        _blocked = _shell_guard_refusal(context, owner, command) or await _egress_refusal(context)',
     '        _blocked = _shell_guard_refusal(context, owner, command)'),
    ('Y7 egress-not-ensured', R, '            await asyncio.to_thread(sm.ensure_running)\n            if not attempted():', '            if False:'),
    ('Y8 restart-stop-first', R, '            _no = _lab_start_refusal(_det, origin, row["owner"], _lid)\n            if _no is not None:\n                return _no',
     '            _no = None'),
    ('Y9 restart-unguarded', R, '        _blocked = _shell_guard_refusal(context, row["owner"], row["command"] or "") or await _egress_refusal(context)',
     '        _blocked = None'),
    ('Z1 run-after-get-unordered', AG, '            if pact in ("put", "get", "deploy") and not m and act not in ("status", "logs", "url", "collect"):',
     '            if False:'),
    ('Z2 backup-unordered', AG, '_LAB_LINEAGE_ACTIONS = frozenset({"keep", "deploy", "rehearse", "switch", "rollback", "restore", "destroy",\n                                  "backup"})',
     '_LAB_LINEAGE_ACTIONS = frozenset({"keep", "deploy", "rehearse", "switch", "rollback", "restore", "destroy"})'),
    ('Z3 alias-name-raw', AG, '            a["name"] = _alias_name(a.get("name") or a.get("service") or a.get("service_name")) or a.get("name")', '            pass'),
    ('Z4 archived-runs', MN, '    if "archiv" in status.lower() or "delet" in status.lower():\n        return None, f"project {pid} is {status.lower()}"', '    if False:\n        return None, ""'),
    ('Z5 store-error-runs', MN, '    if err:\n        return None, f"project {pid} could not be read ({err[:120]})"', '    if err:\n        return None, ""'),
    ('Z6 skip-runs-anyway', MN, '    if _why:\n        for v, t in reversed(_clean):', '    if False:\n        for v, t in reversed(_clean):'),
    ('Z7 retry-loses-owner', MN, '                    _owner = str((_tools_tasks._load_task_store().get(_base_id) or {}).get("owner") or "")',
     '                    _owner = str((_tools_tasks._load_task_store().get(job_id) or {}).get("owner") or "")'),
    # --- r13
    ('V1 probe-all-owners', LB, '    if origin == "probe" and action == "list" and (_truthy(all_owners) or not owner):', '    if origin == "probe" and action == "list" and (not owner):'),
    ('V2 probe-origin-ignored', LB, '    if who == "probe" and st == 200 and isinstance(d, dict) and str(d.get("origin") or "") != "probe":', '    if False:'),
    ('V3 probe-destroy-origin', LB, '        if origin == "probe" and (theirs != owner or str(d.get("origin") or "") != "probe"):', '        if origin == "probe" and theirs != owner:'),
    ('V4 probe-kept-unwalled', LB, '            action not in KEPT_ACTIONS or (lab_id and not _kept_refusal(origin, context))):', '            action not in KEPT_ACTIONS):'),
    ('V5 lab-deleted-project', LB, '    if _gone is not None:\n        return None, str(_gone)', '    if False:\n        return None, str(_gone)'),
    ('V6 rel-not-safe-path', LB, '        p = _get_safe_path(scoped, rel).resolve()', '        p = (scoped / rel).resolve()'),
    ('V7 failed-build-guess', LB, '                elif _known:\n                    inv.record_destroy(lab_id, by="create_failed")', '                else:\n                    inv.record_destroy(lab_id, by="create_failed")'),
    ('V8 by-id-leaf-stop', R, '    if action == "stop" and _leaf_meta().get("leaf"):\n        # r13 m4', '    if False:\n        # r13 m4'),
    ('V11 alias-bare-collect', R, '    if _owner and not _r and want == "collect" and not job_id:', '    if False:'),
    ('V12 alias-other-owner', R, '        if _theirs and not _mine:', '        if False:'),
    ('V14 scope-on-loop', MN, '    _scope, _why = await asyncio.to_thread(_project_turn_scope, context, project_id)', '    _scope, _why = _project_turn_scope(context, project_id)'),
    # r14: the by-id job path reads the registry row
    ('N1 by-id-visible-off', R, '    if not job_visible_to_caller(job):', '    if False:'),
    ('N2 by-id-running-ok', R, '    if now.status == STATUS_RUNNING:', '    if False:'),
    ('N4 by-id-one-namespace', R, '        job = _resolve_job(reg, job_id)', '        job = reg.get(job_id)'),
    ('N5 alias-raw-jobs', R, '            return await _job_by_id(context, want, job_id)',
     '            from .delegate import tool_jobs\n            return await tool_jobs(action=action, job_id=job_id, context=context)'),
    ('N6 probe-list-all', LB, '            labs = [l for l in labs if str(l.get("origin") or "") == "probe"]', '            pass'),
    # r15
    ('N7 shown-raw-spelling', LB, "        return ToolOutcome(f\"[{lab_id}/{a['member']}] got {a['path']} → {shown_path(context, dst)} (",
     "        return ToolOutcome(f\"[{lab_id}/{a['member']}] got {a['path']} → {a['sandbox_path']} ("),
    ('N8 refs-one-namespace', R, '        if sid:\n            out.add(sid)', '        if False:\n            out.add(sid)'),
    ('N9 cancelled-ok', R, '    if now.status != STATUS_DONE:', '    if now.status == "failed":'),
    ('N10 alias-act-on-given-id', R, '                job_id = _mine_refs[-1]', '                pass'),
    # r17
    ('N11 adopt-foreign-projects', R, '            if not await asyncio.to_thread(_is_own_project, context, owner):', '            if False:'),
    ('N12 sweep-pages-foreign', R, '            if _known[o] is False:', '            if False:'),
    ('N13 sweep-ends-on-error', R, '            if _known[o] is False:', '            if not _known[o]:'),
]
PINS = ["tests/test_4sx_r11_review.py", "tests/test_4sx_r14_review.py", "tests/test_4sx_r15_review.py", "tests/test_4sx_r17_review.py", "tests/test_4sx_r12_review.py", "tests/test_4sx_r13_review.py", "tests/test_4sx_b_lab_tool.py", "tests/test_4sx_c_run_tool.py",
        "tests/test_4sx_d5_review.py", "tests/test_4sx_d4_autopilot_labs.py", "tests/test_4sx_r16_battery_pins.py"]


def _tests():
    pat = (r"tools.runs\|tools import runs\|tools import labs\|tools.labs\|run_registry\|_batch_call_depends\|"
           r"_batch_as_run\|leaf_reaper\|egress\|evidence_gate\|summarize_multi_op\|_unattended_project_scope\|"
           r"project_scoped_sandbox\|run_leaf_turn\|tool_lab")
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
