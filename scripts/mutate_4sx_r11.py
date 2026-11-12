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
EX = 'src/ghost_agent/tools/execute.py'
MUT = [
    ('CONTROL known-bad', LB, '        if action in KEPT_ACTIONS:                 # production: refused before any host call',
     '        if False:                 # production: refused before any host call'),
    ('CONTROL no-op', LB, '#: the id the configured labs host last gave on /v1/health', '#: the id the configured labs host last gave on /v1/health  (no-op)'),
    # --- lab tool
    ('L1 put-follows-link', LB, '            data = _read_inside(context, src, PUT_MAX_BYTES + 1)',
     '            data = src.read_bytes()[:PUT_MAX_BYTES + 1]'),
    ('L2 get-follows-link', LB, '    await asyncio.to_thread(write_bytes_nofollow_rel, root, str(Path(p).relative_to(root)), data)', '    Path(p).parent.mkdir(parents=True, exist_ok=True); Path(p).write_bytes(data)'),
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
    ('L12 autopilot-status-refused', LB, '            and kept_touch_refusal("autopilot", action)):', '            ):'),
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
    ('W1 files-unordered', AG, '    if act in _LAB_FILE_ACTIONS and pact in _LAB_FILE_ACTIONS:\n        # workspace files', '    if False:\n        # workspace files'),
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
    ('P3 reaper-apps', LR, '            if (leaf and row.get("lifetime") != "app" and row.get("origin") == "autopilot"', '            if (leaf and row.get("origin") == "autopilot"'),
    ('P4 seen-kept-across-attempts', CL, '        _drop_seen(leaf_base)', '        pass'),
    ('P6 deleted-dir-recreated', FS, '            if not sub.is_dir() and _project_row_gone(context, pid):',
     '            if False:'),
    # --- r12
    ('X1 ownership-noanswer-open', LB, '        return (f"the labs host did not answer the ownership check of {lab_id} ({e}) — nothing was sent; "\n                "try again")',
     '        return ""'),
    ('X3 probe-status-open', LB, '    if origin == "probe" and action not in ("templates", "create", "list") and (', '    if origin == "probe" and action not in ("templates", "create", "list", "status") and ('),
    ('X4 ownerless-probe-blocked', LB, '        if (st == 200 and isinstance(d, dict) and not str(d.get("owner") or "")\n                and str(d.get("origin") or "") == "probe"):\n            return ""',
     '        if False:\n            return ""'),
    ('X5 leaf-confine-off', LB, '    if getattr(context, "confine_writes_to_project", False) is True and p != scoped and scoped not in p.parents:', '    if False:'),
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
    ('Z1 run-after-get-unordered', AG, '            if pact in _LAB_FILE_ACTIONS and not m and act not in ("status", "logs", "url", "collect"):', '            if False:'),
    ('Z2 backup-unordered', LB, 'LINEAGE_ACTIONS = frozenset((KEPT_ACTIONS - READ_ACTIONS) | {"destroy"})', 'LINEAGE_ACTIONS = frozenset((KEPT_ACTIONS - READ_ACTIONS - {"backup"}) | {"destroy"})'),
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
    # 11-05 prod PG: the no-DB-server note on execute results
    ('D1 dbnote-never', EX, '    if not _NO_DB_SERVER_RE.search(str(output or "")):', '    if True:'),
    ('D2 dbnote-not-on-exit0', EX, '                f"STDOUT/STDERR:\\n{output}{_host_proc_note}{_cd_note}{_db_note}")',
     '                f"STDOUT/STDERR:\\n{output}{_host_proc_note}{_cd_note}")'),
    ('D3 dbnote-not-on-failure', EX, '                ), _host_proc_note + _cd_note + _db_note)', '                ), _host_proc_note + _cd_note)'),
    ('D4 dbnote-not-in-scripts', EX, '        _db_note = _no_db_server_note(output, kwargs.get("run_context"))      # 11-05: a script',
     '        _db_note = ""  # 11-05: a script'),
    ('D5 dbnote-ignores-labs', EX, '            _labs = bool(labs_configured(context))', '            _labs = False'),
    ('D6 dbnote-none-says-unconfigured', EX, '    if context is None:\n        # review r19', '    if False:\n        # review r19'),
    ('D7 dbnote-any-line', EX, 'IGNORECASE | re.MULTILINE)', 'IGNORECASE | re.DOTALL)'),
    ('D8 skill-runner-drops-context', 'src/ghost_agent/tools/registry.py', '                                    run_context=context,     # r19', '                                    # r19'),
    ('D10 dbnote-any-reason', EX, 'failed: "\n    r"(?:No such file or directory|Connection refused|Cannot assign requested address)"', 'failed"\n    r"(?:)"'),
    ('D11 dbnote-no-pytest-prefix', EX, '_DB_LINE_PREFIX = (r"^(?:E\\s+)?', '_DB_LINE_PREFIX = (r"^'),
    ('D12 dbnote-raise-says-unconfigured', EX, '            _labs = None\n', '            _labs = False\n'),
    ('D13 dbnote-no-mysql-tcp', EX, '|\\(?(?:ERROR )?200[23]', '|\\(?(?:ERROR )?200[23]XX'),
    ('D14 dbnote-mysql-word-required', EX, "Can't connect to (?:MySQL )?server on '", "Can't connect to MySQL server on '"),
    ('D9 dbnote-any-host', EX, '_LOOPBACK = r"(?:localhost|127\\.0\\.0\\.1|::1)"', '_LOOPBACK = r"[^\\"\'\\s]+"'),
    ('UD P3 refusal said per call', LB, '    if why and host not in _KEY_REFUSAL_SAID:', '    if why:'),
    ('UD P4 refused host reads as no answer', LB, '    why = config_refusal(context)\n    if why:\n        # not "did not answer"', '    why = ""\n    if why:\n        # not "did not answer"'),
    ('UD P6 list answer not counted', LB, '            report["answered"] = report.get("answered", 0) + 1       # ANY answered call ends an outage (r3 m3)', '            pass'),
    ('UD P8 partial pass not a try', LR, '            if rep.get("no_answer") and not rep.get("answered"):\n                continue', '            if rep.get("no_answer"):\n                continue'),
    ('UD P9 no answer counts as a try', LR, '            if rep.get("no_answer") and not rep.get("answered"):\n                continue', '            if False:\n                continue'),
    ('UD P11 key rule not is_global', LB, '        if any(ip.version == n.version and ip in n for n in _KEY_NETS):', '        if not ip.is_global:'),
    ('UD P12 scheme case', LB, '    low = h.lower()', '    low = h'),
    ('UD P13 any scheme', LB, '_KEY_URL_RE = re.compile(r"^https?://(?P<host>', '_KEY_URL_RE = re.compile(r"^[a-z]+://(?P<host>'),
    ('UD P15 key roundtrip', LB, '        if str(ip) != literal or (ip.version == 6) != host.startswith("["):\n            return f"{host!r} is not a plain IP address"\n', ''),
    ('UD P16 dotted name ok', LB, '_BARE_HOST_RE = re.compile(r"^(?!https?$)[a-z][a-z0-9-]*$")', '_BARE_HOST_RE = re.compile(r"^(?!https?$)[a-z][a-z0-9.-]*$")'),
    ('UD X12 untracked ignores closed rows', LB, '    known = {str(r["lab_id"]) for r in (inv.rows(owner) if inv else [])}', '    known = {str(r["lab_id"]) for r in rows}'),
    ('UD A32 owner_only retried', LB, 'body.get("error") in ("owner_active", "owner_only"):', 'body.get("error") in ("owner_active",):'),
    ('UD A33 owner_active retried', LB, 'body.get("error") in ("owner_active", "owner_only"):', 'body.get("error") in ("owner_only",):'),
    ('UD A34 terminal keeps the tag', LB, '            inv.clear_leaf(lid)\n', '            pass\n'),
    ('UD FA11 create silent on a member not ready', LB, '    late = members_not_ready(d)\n    if late:', '    late = members_not_ready(d)\n    if False:'),
    ('UD FA13 missing ready = not ready', LB, '        if isinstance(m, dict) and "ready" in m and m.get("ready") is not True:', '        if isinstance(m, dict) and m.get("ready") is not True:'),
    ('UD FA14 partial create keeps the hold', LB, 'in ((OutcomeStatus.OK, OutcomeStatus.PARTIAL) if action == "create"', 'in ((OutcomeStatus.OK,) if action == "create"'),
    ('UD FB20 exec carries no tag', LB, '    if _tag:\n        body["tag"] = _tag                      # §4UD F-B', '    if False:\n        body["tag"] = _tag                      # §4UD F-B'),
    ('UD FB22 leafless autopilot cancels', LB, '    if origin == "autopilot" and not tag:', '    if False:'),
    ('UD FB23 cancelled exec reads as a result', LB, '    if r.get("cancelled") or r.get("timed_out"):\n        # §4UD F-B', '    if r.get("timed_out"):\n        # §4UD F-B'),
    ('UD FC30 tree abs path', LB, '                if raw.startswith("/") or "\\x00" in raw or "\\\\" in raw or not parts or ".." in parts:', '                if "\\x00" in raw or "\\\\" in raw or not parts or ".." in parts:'),
    ('UD FC31 tree dotdot', LB, '                if raw.startswith("/") or "\\x00" in raw or "\\\\" in raw or not parts or ".." in parts:', '                if raw.startswith("/") or "\\x00" in raw or "\\\\" in raw or not parts:'),
    ('UD FC32 tree links skipped not refused', LB, '                if not (ti.isdir() or ti.isfile()):\n                    return [], [], f"entry {_q(raw)} is a link or a device"', '                if not (ti.isdir() or ti.isfile()):\n                    continue'),
    ('UD FC34 tree into a file', LB, '    if os.path.lexists(dst) and (dst.is_symlink() or not dst.is_dir()):', '    if False:'),
    ('UD FC35 folder put unbounded', LB, '                n_files += 1\n                if max_entries is not None and n_files > max_entries:', '                n_files += 1\n                if False:'),
    ('UD3 N4 folders not counted', LB, '                    n_files += 1\n                    if max_entries is not None and n_files > max_entries:', '                    if max_entries is not None and n_files > max_entries:'),
    ('UD FC38 folder kind ignored', LB, '        if st == 400 and isinstance(body, dict) and body.get("kind") == "directory":', '        if False:'),
    ('UD R1 restart keeps a foreign leaf tag', R, '    return old if old and old == str(_leaf_meta().get("leaf") or "") else None', '    return old or None'),
    ('UD R2 leaf stop any origin', R, '            if row.get("origin") != "autopilot":\n                continue        # a leaf\'s run', "            if False:\n                continue        # a leaf's run"),
    ('UD R3 kept job refusal empty', R, '    if detail.get("refused"):\n        return _kept_job_refusal(row)\n    if detail.get("gone"):', '    if detail.get("gone"):'),
    ('UD R4 old server note any age', R, 'and row.get("lifetime") == "server" and age > OLD_CONV_SERVER_S):', 'and row.get("lifetime") == "server"):'),
    ('UD W1 no-tor hold ignores tor', 'src/ghost_agent/core/lab_waits.py', '        return waited and (code != "labs_no_tor" or str(health.get("tor", "ok") or "") == "ok")', '        return waited'),
    ('UD W2 not-ready no cooldown', 'src/ghost_agent/core/lab_waits.py', '        return waited and (code != "labs_no_tor" or str(health.get("tor", "ok") or "") == "ok")', '        return (code != "labs_no_tor" or str(health.get("tor", "ok") or "") == "ok")'),
    ('UD I1 set_expiry closed rows', 'src/ghost_agent/memory/lab_inventory.py', 'UPDATE labs SET expires_hard = ? WHERE lab_id = ? AND destroyed_at IS NULL', 'UPDATE labs SET expires_hard = ? WHERE lab_id = ?'),
    ('UD A1 leaf deadline not set', CL, '    _tk_dl = _ldc.set(min(_mine, _outer) if _outer > 0 else _mine)', '    _tk_dl = _ldc.set(_outer)'),
    ('UD A1b outer ignored', CL, '    _tk_dl = _ldc.set(min(_mine, _outer) if _outer > 0 else _mine)', '    _tk_dl = _ldc.set(_mine)'),
    ('UD A2a sweep touches running leaf', LR, 'if leaf not in running}', '}'),
    ('UD A2b running never added', CL, '    RUNNING_LEAVES.add(str(leaf_id))', '    pass'),
    ('UD A2c running never cleared', CL, '                if not _end_pending:\n                    RUNNING_LEAVES.discard(str(leaf_id))', '                pass'),
    ('UD A2d loop drops leaf sweep', R, '        await sweep_ended_leaves(context)\n', '        pass\n'),
    ('UD A2e refusal not said', LB, '            pretty_log("Labs", f"leaf lab {lid} NOT destroyed: the labs host refused ({_code}) — retried by the "\n                               "leaf sweep", level="WARNING", icon=Icons.WARN)', '            pass'),
    ('UD A3 no env', R, '        post["env"] = {"PORT": str(port)}', '        pass'),
    ('UD A3b claims started', R, '    return _out(f"[run {name} launched — {head}]\\n"', '    return _out(f"[run {name} started — {head}]\\n"'),
    ('UD A4 server timeout', R, '    wrapped = (f"bash -c {shlex.quote(command)}" if lifetime == "server" else', '    wrapped = (f"bash -c {shlex.quote(command)}" if False else'),
    ('UD A4b job no timeout', R, '    wrapped = (f"bash -c {shlex.quote(command)}" if lifetime == "server" else', '    wrapped = (f"bash -c {shlex.quote(command)}" if True else'),
    ('UD A5a everyone is owner', R, '    return origin == "owner"', '    return True'),
    ('UD A5d all view probes', R, '        return _status_all(_reg) if _all else', "        return await _status(context, _reg, 'conversation:other', origin, None) if _all else"),
    ('UD A6 url not read', R, '            if same and p.get("url"):', '            if False:'),
    ('UD A7 no app line', LB, '    if isinstance(d.get("app"), dict):\n        lines.append', '    if False:\n        lines.append'),
    ('UD A8 app not lineage', LB, 'LINEAGE_ACTIONS = frozenset((KEPT_ACTIONS - READ_ACTIONS) | {"destroy"})', 'LINEAGE_ACTIONS = frozenset((KEPT_ACTIONS - READ_ACTIONS - {"app"}) | {"destroy"})'),
    ('UD A8b agent hand list', AG, '_LAB_APP_ACTIONS = _LAB_LINEAGE_ACTIONS', '_LAB_APP_ACTIONS = frozenset({"keep", "deploy", "rehearse", "switch", "rollback", "restore", "destroy"})'),
    ('UD A9a past limit never', R, '        return bool(row.get("backend") == "labs_job" and row.get("ended_at") is None and exp', '        return bool(False and exp'),
    ('UD A9b keep keeps expiry', LB, '        _no_expiry(context, lab_id)                # §4UD F10: a kept machine has none', '        pass'),
    ('UD A9c expiry not recorded', LB, '                          expires_hard=_hard_expiry(d))', '                          expires_hard=None)'),
    ('UD A9d note not rendered', 'src/ghost_agent/core/prompts.py', '            if r.get("note"):', '            if False:'),
    ('UD A9e run expiry meta', R, '    _meta["expires_at"] = _run_end', '    pass'),
    ('UD A10a gone marker', R, '            return "lost", "its lab was destroyed", {"gone": "lab"}', '            return "lost", "its lab was destroyed", {}'),
    ('UD A10b cut note', R, '    cut = _lab_log_cut_note(row, detail) if (detail.get("output_truncated") and len(got) <= n) else ""', '    cut = ""'),
    ('UD A11 run logs open', R, '    if _why:\n        return _refuse(f"{row[\'name\']!r} is a kept app (production) — {_why}", "run_not_allowed")', '    if False:\n        return _refuse(f"{row[\'name\']!r} is a kept app (production) — {_why}", "run_not_allowed")'),
    ('UD A11b rule opens logs', LB, 'AUTOPILOT_KEPT_READS = frozenset({"status", "url"})', 'AUTOPILOT_KEPT_READS = frozenset({"status", "url", "logs"})'),
    ('UD A12 features ignored', LB, '    if "drop_data_guard" in await host_features(context, origin):', '    if False:'),
    ('UD A12b always trust', LB, '    if "drop_data_guard" in await host_features(context, origin):', '    if True:'),
    ('UD A14a busy fails', 'src/ghost_agent/core/lab_waits.py', '    "lab_busy": "labs_not_ready",\n', ''),
    ('UD A14b needed dropped', LB, '                     needed_mb=int(getattr(out, "needed_mb", 0) or 0))', '                     needed_mb=0)'),
    ('UD A15 no date', LB, '    return time.strftime("%H:%M" if same_day else "%Y-%m-%d %H:%M", t)', '    return time.strftime("%H:%M", t)'),
    ('UD A16 junk ttl', LB, '        return None, f"ttl_idle_s must be a whole number of seconds (like 7200), not {str(v)[:40]!r}"', '        return 0, ""'),
    ('UD A16b clamp note', LB, '    return f" (the idle TTL asked for is above this lab\'s maximum: capped at {c} s)" if c else ""', '    return ""'),
    ('UD A16c loopback hint', LB, '    if body.get("loopback_only"):\n        return (f"\\nA process', '    if False:\n        return (f"\\nA process'),
    ('UD A17 f2 mark', LB, '"page: this is what had arrived]" if body.get("truncated") else ""),', '"page: this is what had arrived]" if False else ""),'),
    ('UD A-sweep gone', R, 'OLD_CONV_SERVER_S = 24 * 3600', 'OLD_CONV_SERVER_S = 24 * 3600\nCONV_SERVER_MAX_AGE_S = 1'),
    ('UD A-old note project', R, '    if not (str(row.get("owner") or "").startswith("conversation:") and row.get("backend") == "services"', '    if not (row.get("backend") == "services"'),
    ('UD A-old note lab run', R, 'startswith("conversation:") and row.get("backend") == "services"\n', 'startswith("conversation:")\n'),
    ('UD A-restart drops own', R, '    return old if old and old == str(_leaf_meta().get("leaf") or "") else None', '    return None'),
    ('UD A-orphan any origin', LR, '            if (leaf and row.get("lifetime") != "app" and row.get("origin") == "autopilot"', '            if (leaf and row.get("lifetime") != "app"'),
    ('UD A-outage every tick', LB, '    if now_silent and not _LEAF_OUTAGE["on"]:', '    if now_silent:'),
    ('UD A18 no cap', LR, 'if _TRIES.get(leaf, 0) < SWEEP_MAX_TRIES}', '}'),
    ('UD A18b tries not dropped', LR, '            _TRIES.pop(leaf, None)                     # the leaf closed: its count goes with it', '            pass'),
    ('UD A18c cap not said', LR, '            if _TRIES[leaf] == SWEEP_MAX_TRIES:', '            if False:'),
    ('UD A4 seen keeps running', CL, '                        RUNNING_LEAVES.discard(str(_lid))      # the end has run: the sweep may look at it now', '                        pass'),
    ('UD A7 past limit any backend', R, '        return bool(row.get("backend") == "labs_job" and row.get("ended_at") is None and exp', '        return bool(row.get("ended_at") is None and exp'),
    ('UD A8 stop ended by id', R, '    if row["ended_at"] is not None:\n        return _refuse(f"{run_id} ({row[\'name\']}){whose} is not running', '    if False:\n        return _refuse(f"{run_id} ({row[\'name\']}){whose} is not running'),
    ('UD A9 already-kept expiry', LB, '            note = await _record_app_run(context, str(det.get("origin") or origin), det)\n            _no_expiry(context, lab_id)', '            note = await _record_app_run(context, str(det.get("origin") or origin), det)'),
    ('UD A12 not-checked branch', R, '    if st != 200:\n        return _out(f"[run {name} started — {head}]\\n"\n                    f"Its port {port} was NOT checked', '    if st != 200 and st != 403:\n        return _out(f"[run {name} started — {head}]\\n"\n                    f"Its port {port} was NOT checked'),
    ('UD A14 logs by run id', R, '    if _all or (action == "stop" and _RUN_ID_RE.match(_jn)):', '    if _all or (action in ("stop", "logs") and _RUN_ID_RE.match(_jn)):'),
    ('UD A34 leftovers note', R, '            if _exp and not r.get("leftovers"):', '            if _exp:'),
    ('UD A39 follow redirects', LB, 'trust_env=False, timeout=timeout, follow_redirects=False) as client:', 'trust_env=False, timeout=timeout, follow_redirects=True) as client:'),
    ('UD A-key digits name', LB, '_BARE_HOST_RE = re.compile(r"^(?!https?$)[a-z][a-z0-9-]*$")', '_BARE_HOST_RE = re.compile(r"^(?!https?$)[a-z0-9][a-z0-9-]*$")'),
    ('UD A-loopback in flux', LB, '                               world_changed=not body.get("loopback_only"))', '                               world_changed=True)'),
    ('UD A-403 job empty ok', R, '            return "unknown", "its lab is kept (production): its jobs are read by the owner", {"refused": "owner_only"}', '            return "exited", "", {"exit_code": 0, "output_tail": ""}'),
    ('UD A-refused collect live', R, '        state, note, detail = await _probed(context, reg, row, origin)\n        if detail.get("refused"):\n            return _kept_job_refusal(row)', '        state, note, detail = await _probed(context, reg, row, origin)'),
    ('UD A-refused collect ended', R, '        _s, _n, detail = await _probe(context, row, origin)\n        if detail.get("refused"):\n            return _kept_job_refusal(row)', '        _s, _n, detail = await _probe(context, row, origin)'),
    ('UD A-quiet timer', LR, '    if quiet and not (done["runs"] or done["labs"]):\n        return done', '    if False:\n        return done'),
    ('UD X12 known empty', LB, '    known = {str(r["lab_id"]) for r in (inv.rows(owner) if inv else [])}', '    known = set()'),
    ('UD A32 any 403 terminal', LB, '        elif isinstance(body, dict) and body.get("error") in ("owner_active", "owner_only"):', '        elif isinstance(body, dict) and (body.get("error") == "owner_active" or st == 403):'),
    ('UD m1 always warning', R, 'logging.INFO if (_LEAF_OUTAGE["on"] or config_refusal(context)) else logging.WARNING,', 'logging.WARNING,'),
    ('UD M3 exemption back', AG, '            return bool(m and plid and m.group(1) == plid) or pact in _LAB_APP_ACTIONS', '            return bool(m and plid and m.group(1) == plid) or (pact in _LAB_APP_ACTIONS and not a.get("where"))'),
    ('UD FD1 any suffix', LB, '    if not err and dst.suffix.lower() != ".png":', '    if False:'),
    ('UD FD2 folder ok', LB, '        err = f"{shown_path(context, dst)} is a folder — name the .png file to write"', '        err = ""'),
    ('UD FD3 not a png written', LB, '    if not png.startswith(_PNG_MAGIC) or len(png) > SHOT_MAX_BYTES:', '    if len(png) > SHOT_MAX_BYTES:'),
    ('UD FD4 any size', LB, '    if not png.startswith(_PNG_MAGIC) or len(png) > SHOT_MAX_BYTES:', '    if not png.startswith(_PNG_MAGIC):'),
    ('UD FD5 no look line', LB, '        + f").\\nYou have NOT seen this picture.', '        + f").\\nDone.'),
    ('UD FD6 status none unsaid', LB, '        + (f"the page answered HTTP {page}" if page is not None else', '        + (f"the page answered HTTP {page}" if True else'),
    ('UD FD7 screenshot is a read', LB, 'READ_ACTIONS = frozenset({"templates", "list", "status", "logs", "http", "spec", "backups"})', 'READ_ACTIONS = frozenset({"templates", "list", "status", "logs", "http", "spec", "backups", "screenshot"})'),
    ('UD FD8 no write path check', LB, '    dst, err = _sandbox_path(context, a["sandbox_path"], write=True)\n    if not err and dst.is_dir():\n        err = f"{shown_path(context, dst)} is a folder — name the .png', '    dst, err = Path(context.sandbox_dir) / str(a["sandbox_path"] or "x.png"), ""\n    if not err and dst.is_dir():\n        err = f"{shown_path(context, dst)} is a folder — name the .png'),
    ('UD FD9 member not required', LB, '_MEMBER_ACTIONS = frozenset({"exec", "put", "get", "member", "logs", "http", "screenshot"})', '_MEMBER_ACTIONS = frozenset({"exec", "put", "get", "member", "logs", "http"})'),
    ('UD FD10 port default', LB, '    port = _int(a["port"], 80, 1, 65535)\n    st, body = await _call(context, "POST", f"/v1/labs/{lab_id}/screenshot"', '    port = _int(a["port"], 8080, 1, 65535)\n    st, body = await _call(context, "POST", f"/v1/labs/{lab_id}/screenshot"'),
    ('UD2 M1 entries unchecked', LB, '            why = _tree_place_refusal(context, dst, rel, is_dir)', '            why = ""'),
    ('UD2 M1 git allowed', LB, '    if any(_fold(part) == ".git" for part in p.relative_to(root).parts):', '    if False:'),
    ('UD2 M1 git exact case', LB, '    if any(_fold(part) == ".git" for part in p.relative_to(root).parts):', '    if any(part == ".git" for part in p.relative_to(root).parts):'),
    ('UD2 M1 released allowed', LB, '    return str(rb) if rb else ""', '    return ""'),
    ('UD2 M1 write without the check', LB, '    why = write_refusal(context, p)\n    if why:\n        raise WriteRefused(why)\n    from .file_system import write_bytes_nofollow_rel', '    from .file_system import write_bytes_nofollow_rel'),
    ('UD2 M1 mkdir without the check', LB, '    why = write_refusal(context, p)\n    if why:\n        raise WriteRefused(why)\n    root = Path(context.sandbox_dir).resolve()\n    flags =', '    root = Path(context.sandbox_dir).resolve()\n    flags ='),
    ('UD2 M1 final path unchecked', LB, '    why = write_refusal(context, final)\n    if why:\n        return why\n    walk = dst', '    walk = dst'),
    ('UD2 M2 refused host asked for labs', LB, '    if config_refusal(context):\n        return []          # a labs host the key rule refuses', '    if False:\n        return []          # a labs host the key rule refuses'),
    ('UD2 M3 strikes forget CANCELLED', 'src/ghost_agent/core/strikes.py', '(?:exit=|TIMED OUT|no exit code|CANCELLED)', '(?:exit=|TIMED OUT|no exit code)'),
    ('UD2 M3 stored CANCELLED not failed', 'src/ghost_agent/distill/outcome_heuristics.py', 'if " TIMED OUT" in first or " CANCELLED" in first:', 'if " TIMED OUT" in first:'),
    ('UD2 m1 cancel always says killed', LB, '_CANCEL_OUTCOME.get(str(x.get("outcome") or ""), _CANCEL_OUTCOME[""])', '_CANCEL_OUTCOME["killed"]'),
    ('UD2 m1 unknown outcome = killed', LB, '                   "": "asked to end; not confirmed yet"}', '                   "": "killed"}'),
    ('UD2 m1 stuck cancel is OK', LB, 'status=OutcomeStatus.FAILED if stuck and not killed else OutcomeStatus.OK,', 'status=OutcomeStatus.OK,'),
    ('UD2 m1 cancelled exec hides not killed', LB, '        if r.get("killed") is False:\n            fate = (f"NOT KILLED', '        if False:\n            fate = (f"NOT KILLED'),
    ('UD3 N2 not killed also says ended', LB, 'ends it).")\n        else:', 'ends it).") + f" {word}: the command was ended {why}."\n        else:'),
    ('UD3 N3 cut gzip accepted', LB, '    if not d.eof or d.unused_data.strip(b"\\0"):', '    if False:'),
    ('UD3 N3 second gzip ignored', LB, '    if not d.eof or d.unused_data.strip(b"\\0"):', '    if not d.eof:'),
    ('UD3 N5 checks on the loop', LB, '    files, dirs, err, refused = await asyncio.to_thread(unpack_and_check)', '    files, dirs, err, refused = unpack_and_check()'),
    ('UD3 r3 agent zero padding refused', LB, 'd.unused_data.strip(b"\\0"):', 'd.unused_data:'),
    ('UD3 m11d userinfo shown', LB, '.split("/", 1)[0].rsplit("@", 1)[-1]', '.split("/", 1)[0]'),
    ('UD2 m3 case collision allowed', LB, '                    if spelled.setdefault(_pk, _sp) != _sp:', '                    if False:'),
    ('UD2 m3 duplicate allowed', LB, '                if kinds.get(key) == "file" or (kind == "file" and key in kinds):', '                if False:'),
    ('UD2 m3 entry under a file', LB, '                    if kinds.get("/".join(_fold(p) for p in parts[:i])) == "file":', '                    if False:'),
    ('UD2 m3 file over entries', LB, '                if kind == "file" and any(k.startswith(key + "/") for k in kinds):', '                if False:'),
    ('UD2 m3 file in the way removed', LB, '        if os.path.lexists(walk) and (walk.is_symlink() or not walk.is_dir()):', '        if False:'),
    ('UD2 m3 dir through a link', LB, '        if final.is_symlink() and is_dir:', '        if False:'),
    ('UD2 m3 file over a folder', LB, '        if not is_dir and final.is_dir() and not final.is_symlink():', '        if False:'),
    ('UD2 m3 folder over a file', LB, '        if is_dir and not final.is_dir():', '        if False:'),
    ('UD2 m3 empty dirs not made', LB, '        for rel in dirs:\n            _ws_mkdir(context, dst.joinpath(*rel.split("/")))', '        for rel in []:\n            _ws_mkdir(context, dst.joinpath(*rel.split("/")))'),
    ('UD2 m3 put without dirs', LB, 'TREE_MAX_ENTRIES, True)', 'TREE_MAX_ENTRIES, False)'),
    ('UD2 m3 count not from disk', LB, '            done += 1', '            done = len(files)'),
    ('UD2 m5 names unescaped', LB, '    return "\'" + t[:cap].encode("unicode_escape").decode("ascii").replace("\'", "\\\\\'") + ("…\'" if len(t) > cap else "\'")', '    return "\'" + t[:cap] + ("…\'" if len(t) > cap else "\'")'),
    ('UD2 m5 cancel cmd raw', LB, "_q(x.get('cmd') or '', 120)", "str(x.get('cmd') or '')[:120]"),
    ('UD2 m5 place path raw', LB, '            return (f"{_q(shown_path(context, walk), 200)} is a file (or a link)', '            return (f"{shown_path(context, walk)} is a file (or a link)'),
    ('UD2 m7 not checked reads not ready', LB, '"host checked it." if m.get("ready") is False else', '"host checked it." if True else'),
    ('UD2 m8 unbounded inflate', LB, '        out = d.decompress(data, limit + 1)', '        out = d.decompress(data)'),
    ('UD2 m9 answers again unasked', LB, '    if not now_silent and not answered:\n        _LEAF_OUTAGE["on"] = False\n        return', '    if False:\n        _LEAF_OUTAGE["on"] = False\n        return'),
    ('UD2 m9 empty pass claims an answer', LR, '            leaf_outage(False, answered=False)', '            leaf_outage(False)'),
    ('UD2 m11 userinfo accepted', LB, '_KEY_URL_RE = re.compile(r"^https?://(?P<host>', '_KEY_URL_RE = re.compile(r"^https?://(?:[^/@]*@)?(?P<host>'),
    ('UD2 m11 scheme word as a name', LB, '_BARE_HOST_RE = re.compile(r"^(?!https?$)[a-z][a-z0-9-]*$")', '_BARE_HOST_RE = re.compile(r"^[a-z][a-z0-9-]*$")'),
    ('UD2 m11 link-local back', LB, '"100.64.0.0/10", "::1/128", "fc00::/7"))', '"100.64.0.0/10", "::1/128", "fc00::/7", "169.254.0.0/16", "fe80::/10"))'),
    ('UD2 m11 any port number', LB, '    if m.group("port") is not None and not 0 < int(m.group("port")) < 65536:', '    if False:'),
    ('UD2 m6 cancel waits for its exec', AG, '    if act == "cancel" and pact == "exec":\n        return False', '    if False:\n        return False'),
    ('UD2 m6 screenshot keeps the stale listing', LB, 'WORKSPACE_WRITE_ACTIONS = frozenset({"get", "screenshot"})', 'WORKSPACE_WRITE_ACTIONS = frozenset({"get"})'),
    ('UD2 m6 loop reads get only', AG, '.strip().lower() in _LAB_WS_WRITERS) or \\', '.strip().lower() == "get") or \\'),
    ('UD2 A10 screenshot to a script name', LB, '    if not err and dst.suffix.lower() != ".png":', '    if not err and dst.suffix.lower() not in (".png", ".py", ".sh"):'),
    ('UD2 A11 lenient base64', LB, 'png = base64.b64decode(str(body.get("png_b64") or ""), validate=True)', 'png = base64.b64decode(str(body.get("png_b64") or ""))'),
    ('UD2 A24 get any 404 is an older host', LB, 'params={"member": a["member"], "path": a["path"]})\n    body = body if isinstance(body, dict) else {}\n    if st == 404 and "error" not in body:', 'params={"member": a["member"], "path": a["path"]})\n    body = body if isinstance(body, dict) else {}\n    if st == 404:'),
    ('UD2 A24 put any 404 is an older host', LB, '"archive_b64": base64.b64encode(data).decode()})\n    body = body if isinstance(body, dict) else {}\n    if st == 404 and "error" not in body:', '"archive_b64": base64.b64encode(data).decode()})\n    body = body if isinstance(body, dict) else {}\n    if st == 404:'),
    ('UD2 A25 any kind is a folder', LB, 'body.get("kind") == "directory":', 'body.get("kind"):'),
    ('UD2 A4 backslash entry accepted', LB, 'if raw.startswith("/") or "\\x00" in raw or "\\\\" in raw or not parts or ".." in parts:', 'if raw.startswith("/") or not parts or ".." in parts:'),
]
PINS = ["tests/test_4sx_r11_review.py", "tests/test_4sx_r14_review.py", "tests/test_4sx_r15_review.py", "tests/test_4sx_r17_review.py", "tests/test_execute_no_db_server_note.py", "tests/test_tools_audit_fixes.py", "tests/test_4sx_r12_review.py", "tests/test_4sx_r13_review.py", "tests/test_4sx_b_lab_tool.py", "tests/test_4sx_c_run_tool.py",
        "tests/test_4sx_d5_review.py", "tests/test_4sx_d4_autopilot_labs.py", "tests/test_4sx_r16_battery_pins.py",
        "tests/test_4ud_labs.py"]


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
