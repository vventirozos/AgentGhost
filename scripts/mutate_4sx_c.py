#!/usr/bin/env python3
"""§4SX Phase C battery — the `run` tool (tools/runs.py, memory/run_registry.py), jobs start_detached /
started_via, the jobs-tool skip, the wake hint and the operator probe route.
Coverage-selected, parallel (scripts/battery_cov.py); the real src/ is never mutated.
No-op control must SURVIVE, known-bad control must DIE.    python3 scripts/mutate_4sx_c.py [-jN] [all|check|rerun ID …]
"""
import os, subprocess, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = 'src/ghost_agent/tools/runs.py'
RR = 'src/ghost_agent/memory/run_registry.py'
J = 'src/ghost_agent/sandbox/jobs.py'
D = 'src/ghost_agent/tools/delegate.py'
MN = 'src/ghost_agent/main.py'
RT = 'src/ghost_agent/api/routes.py'
MUT = [
    ('CONTROL known-bad', R, '    if refusal:\n        return None, "", "runs are the owner\'s', '    if False:\n        return None, "", "runs are the owner\'s'),
    ('CONTROL no-op', R, 'def _cap(text: str) -> str:', 'def _cap(text: str) -> str:  # no-op'),
    # who / owner / input
    ('W1 refusal-ignored', R, '    if refusal:\n        return _refuse(refusal, "run_not_allowed")', '    if False:\n        return _refuse(refusal, "run_not_allowed")'),
    ('W2 no-owner-allowed', R, "    if not owner:\n        # review: a job's wake turn", "    if False:\n        # review: a job's wake turn"),
    ('W3 name-unchecked', R, '    if not _NAME_RE.match(name):\n        return _refuse("`name` must match', '    if False:\n        return _refuse("`name` must match'),
    ('W4 probe-stops-owner', R, '    if origin == "probe" and row["origin"] != "probe" and action in ("stop", "restart"):', '    if False:'),
    ('W5 bad-action', R, '    if action not in ACTIONS:\n        return _refuse(f"unknown action', '    if False:\n        return _refuse(f"unknown action'),
    ('W6 no-registry-runs', R, '    if reg is None:\n        return _out("RUN ERROR: the run registry is unavailable', '    if False:\n        return _out("RUN ERROR: the run registry is unavailable'),
    ('K1 conv-key-bare', R, '        return "c-" + hashlib.sha1(ident.encode("utf-8")).hexdigest()[:12]', '        return ident'),
    ('K2 project-key-unchecked', R, '        return ident if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", ident) else None', '        return ident'),
    ('S1 where-defaulted', R, '    if not where:\n        return _refuse("`where` is required', '    if not where:\n        where = "workspace"\n    if False:\n        return _refuse("`where` is required'),
    ('S2 app-allowed', R, '    if lifetime == "app":\n        return _refuse("an `app` run is made by', '    if False:\n        return _refuse("an `app` run is made by'),
    ('S3 dup-name-started', R, '        if state in ("running", "unknown"):\n            return _refuse(f"a run named', '        if False:\n            return _refuse(f"a run named'),
    ('S4 unknown-not-dup', R, '        if state in ("running", "unknown"):\n            return _refuse(f"a run named', '        if state in ("running",):\n            return _refuse(f"a run named'),
    ('S5 lab-server-portless', R, '    if lifetime == "server" and port is None:\n        return _refuse("a `server` on a lab', '    if False:\n        return _refuse("a `server` on a lab'),
    ('S6 port-range-off', R, '    if port is not None and not 1 <= port <= 65535:', '    if False:'),
    ('S7 lab-where-loose', R, '_LAB_WHERE_RE = re.compile(r"^(lab_[0-9a-f]{8})/([a-z][a-z0-9-]{0,30})$")', '_LAB_WHERE_RE = re.compile(r"^(lab_[0-9a-f]{8})/(.+)$")'),
    # workspace server
    ('V1 foreign-key-replaced', R, '    if before is not None and await asyncio.to_thread(svc._entry_state, before) is not False:', '    if False:'),
    ('V2 legacy-key-replaced', R, '    if name in ents:\n        # review r1 minor 4', '    if False:\n        # review r1 minor 4'),
    ('V3 token-unchanged-ok', R, '              and (before is None or row.get("token") != before.get("token"))', '              and True'),
    ('V4 dead-start-ok', R, '              and await asyncio.to_thread(svc._entry_state, row) is not False)', '              and True)'),
    ('V5 unrecorded-left', R, '            return text, row, ("undone" if await _svc_undo(svc, key) else "kept"), True', '            return text, row, "kept", True'),
    ('V6 undo-unverified', R, '        return e is None or await asyncio.to_thread(svc._entry_state, e) is False', '        return True'),
    ('V7 stop-asked-ignored', R, '        if _stop_asked():\n            return _NOT_STARTED, None, None, False', '        if False:\n            return _NOT_STARTED, None, None, False'),
    # (V8 inner-shield removal is EQUIVALENT: every start runs inside `_locked`'s shielded task — Q2 pins that)
    # workspace job
    ('J1 dup-recorded', R, '        if entry is None or entry.get("duplicate"):\n            return entry, reason, None', '        if entry is None:\n            return entry, reason, None'),
    ('J2 unrecorded-not-cancelled', R, '                await asyncio.to_thread(jobs.cancel, entry["id"])', '                pass'),
    ('J3 job-error-raises', R, '        except Exception as e:  # noqa: BLE001 — e.g. a missing project folder (review r1 minor 7)\n            return None, f"{type(e).__name__}: {e}", None', '        except ZeroDivisionError as e:\n            return None, f"{type(e).__name__}: {e}", None'),
    ('J4 dup-is-ok', R, '    if entry.get("duplicate"):\n        # (no id', '    if False:\n        # (no id'),
    ('J5 project-wd-dropped', R, '    wd = f"/workspace/projects/{proj}" if proj else None', '    wd = None'),
    # lab start
    ('L1 ttl-wrap-dropped', R, '               f"timeout --foreground -k 30 {LAB_JOB_TTL_S} bash -c {shlex.quote(command)}")', '               command)'),
    ('L2 wrap-unquoted', R, '               f"timeout --foreground -k 30 {LAB_JOB_TTL_S} bash -c {shlex.quote(command)}")', '               f"timeout --foreground -k 30 {LAB_JOB_TTL_S} bash -c {command}")'),
    ('L11 own-process-group', R, '               f"timeout --foreground -k 30 {LAB_JOB_TTL_S} bash -c {shlex.quote(command)}")', '               f"timeout -k 30 {LAB_JOB_TTL_S} bash -c {shlex.quote(command)}")'),
    ('L3 probe-on-owner-lab', R, '    if origin == "probe" and str(det.get("origin") or "") != "probe":', '    if False:'),
    ('L4 other-owner-unnamed', R, '    other = f" (this lab belongs to {theirs})" if theirs and theirs != owner else ""', '    other = ""'),
    ('L5 unconfigured-tries', R, '    if not labs_configured(context):\n        return _refuse("no labs host is configured', '    if False:\n        return _refuse("no labs host is configured'),
    ('L6 not-detached', R, '    post: Dict[str, Any] = {"member": member, "cmd": wrapped, "detach": True}', '    post: Dict[str, Any] = {"member": member, "cmd": wrapped}'),
    ('L7 unrecorded-not-killed', R, '        if rid is not None:\n            return "ok", "", job_id, rid\n        try:', '        if True:\n            return "ok", "", job_id, rid\n        try:'),
    ('L8 kill-unconfirmed-claimed', R, '            killed = dst < 300 and isinstance(dbody, dict) and dbody.get("state") != "running"', '            killed = True'),
    ('L9 no-answer-is-failed', R, '    if kind == "no_answer":\n        return _out(f"RUN ERROR (no_answer): the labs host did not answer ({info}) — the run may have started; "\n                    f"check with `lab status {lab_id}` before starting it again.", OutcomeStatus.UNRESOLVED,', '    if kind == "no_answer":\n        return _out(f"RUN ERROR (no_answer): the labs host did not answer ({info}) — the run may have started; "\n                    f"check with `lab status {lab_id}` before starting it again.", OutcomeStatus.FAILED,'),
    ('L10 no-id-ok', R, '        if not job_id:\n            return "no_id", "", None, None', '        if False:\n            return "no_id", "", None, None'),
    # probes
    ('P1 token-mismatch-running', R, '        if row["backend_token"] and str(e.get("token") or "") != row["backend_token"]:', '        if False:'),
    ('P2 svc-unknown-is-exited', R, '        if alive is None:\n            return "unknown", "the sandbox did not answer the probe"', '        if False:\n            return "unknown", "the sandbox did not answer the probe"'),
    ('P3 cancelled-is-exited', R, '"cancelled": "stopped", "lost": "lost"}.get(st, "unknown")', '"cancelled": "exited", "lost": "lost"}.get(st, "unknown")'),
    ('P4 unknown-job-unknown', R, '        if st == 404 and body.get("error") == "unknown_job":\n            return "lost"', '        if False:\n            return "lost"'),
    ('P5 killed-is-exited', R, '        if body.get("killed"):\n            return "stopped", "killed", detail', '        if False:\n            return "stopped", "killed", detail'),
    ('P6 unknown-recorded', R, '    if state != "unknown":\n        reg.set_state(row["run_id"], state, note)', '    if True:\n        reg.set_state(row["run_id"], state, note)'),
    ('P7 destroyed-lab-unknown', R, '            return "lost", "its lab was destroyed", {"gone": "lab"}', '            return "unknown", "its lab was destroyed", {"gone": "lab"}'),
    # logs / collect / url / status
    ('G1 replaced-logs-shown', R, '        if state == "replaced" or (state == "lost" and row["ended_at"] is None):', '        if False:'),
    ('G2 ended-reprobed', R, '        state = row["state"]                       # r2 minor 4', '        state = state                       # r2 minor 4'),
    ('G3 collect-exit-parse-off', R, '            if _m:\n                detail = {**detail, "exit_code": int(_m.group(1))}', '            if False:\n                detail = {**detail, "exit_code": int(_m.group(1))}'),
    ('G4 collect-running-result', R, '    if state in ("running", "unknown"):\n        return _out(f"[run {row[\'name\']} — {state}', '    if False:\n        return _out(f"[run {row[\'name\']} — {state}'),
    ('G5 nonzero-ok', R, '    ok = state == "exited" and code == 0', '    ok = state == "exited"'),
    ('G6 collect-unknown-ended', R, '        if _s == "unknown":\n            return _out(f"[run {row[\'name\']} {row[\'state\']}] could not read', '        if False:\n            return _out(f"[run {row[\'name\']} {row[\'state\']}] could not read'),
    ('G7 status-ended-probed', R, '        if r["ended_at"] is not None:\n            out.append(_line(r, r["state"], r["state_note"], {}))\n            continue', '        if False:\n            out.append(_line(r, r["state"], r["state_note"], {}))\n            continue'),
    ('G8 collect-server', R, '    if row["lifetime"] != "job":\n        return _refuse(f"{row[\'name\']!r} is a {row[\'lifetime\']} — `collect`', '    if False:\n        return _refuse(f"{row[\'name\']!r} is a {row[\'lifetime\']} — `collect`'),
    ('G9 portless-url', R, '    if not row["port"]:\n        return _out(f"{row[\'name\']}: a portless server', '    if False:\n        return _out(f"{row[\'name\']}: a portless server'),
    # stop
    ('T1 replaced-stopped', R, '    if state == "replaced":\n        reg.set_state(row["run_id"], "replaced", note)', '    if False:\n        reg.set_state(row["run_id"], "replaced", note)'),
    ('T2 ended-stopped-again', R, '    if state in _ENDED:\n        reg.set_state(row["run_id"], state, note)\n        return _out(f"[run {row[\'name\']}] already {state}"', '    if False:\n        reg.set_state(row["run_id"], state, note)\n        return _out(f"[run {row[\'name\']}] already {state}"'),
    ('T3 cancel-false-ended-fail', R, '            if st2 in _ENDED:', '            if False:'),
    ('T4 cancel-false-running-ok', R, '            if st2 == "running":', '            if False:'),
    ('T5 lab-running-after-stop-ok', R, '        if body.get("state") == "running":\n            return _out(f"RUN ERROR: {row[\'name\']!r} is still running after the stop.", OutcomeStatus.FAILED,', '        if False:\n            return _out(f"RUN ERROR: {row[\'name\']!r} is still running after the stop.", OutcomeStatus.FAILED,'),
    ('T6 ws-running-after-stop-ok', R, '    if state in ("running",):\n        return _out(f"RUN ERROR: {row[\'name\']!r} is still running after the stop.\\n', '    if False:\n        return _out(f"RUN ERROR: {row[\'name\']!r} is still running after the stop.\\n'),
    ('T7 unknown-after-stop-ok', R, '    if state == "unknown":\n        return _out(f"[run {row[\'name\']}] stop sent', '    if False:\n        return _out(f"[run {row[\'name\']}] stop sent'),
    ('T8 lost-note-kept', R, '"stopped by run" if state == "lost" else note)', 'note)'),
    ('T9 lab-destroyed-fail', R, '            reg.set_state(row["run_id"], "lost", "its lab was destroyed")', '            pass'),
    # restart
    ('X1 job-restarted', R, '    if row["lifetime"] == "job":\n        return _refuse("a job is not restarted', '    if False:\n        return _refuse("a job is not restarted'),
    ('X2 survived-is-failed', R, '            if not new_token and alive is True:', '            if False:'),
    ('X3 unverified-is-failed', R, '            if alive is None:\n                return "unverified", text, None', '            if False:\n                return "unverified", text, None'),
    ('X4 ended-row-rewritten', R, '            live_row = row["ended_at"] is None          # r4 minor 3', '            live_row = True          # r4 minor 3'),
    ('X5 old-token-is-new', R, '            new_token = bool(e and e.get("token") and str(e.get("token")) != row["backend_token"])', '            new_token = bool(e and e.get("token"))'),
    ('X6 old-row-not-closed', R, '                if live_row:\n                    reg.set_state(row["run_id"], "stopped", f"restarted as {rid}")', '                if False:\n                    reg.set_state(row["run_id"], "stopped", f"restarted as {rid}")'),
    ('X7 crashed-not-restartable', R, '        if last is not None and last["state"] in ("exited", "failed") and last["lifetime"] == "server":', '        if False:'),
    ('X8 replaced-restarted', R, '        if state in ("replaced", "lost"):\n            reg.set_state(row["run_id"], state, note)\n            return _refuse(f"{row[\'name\']!r} is {state} ({note}) — not restarting', '        if False:\n            reg.set_state(row["run_id"], state, note)\n            return _refuse(f"{row[\'name\']!r} is {state} ({note}) — not restarting'),
    ('X9 lab-was-stopped-lie', R, '        was = bool(getattr(stopped, "world_changed", False))', '        was = True'),
    ('X10 restart-unrecorded-left', R, '                    gone = await _svc_undo(svc, row["backend_ref"])', '                    gone = False'),
    # lock
    ('Q1 lock-stop-ignored', R, '            if _stop_asked():\n                coro.close()', '            if False:\n                coro.close()'),
    ('Q2 lock-not-shielded', R, '    return await asyncio.shield(t)', '    return await t'),
    ('Q3 lock-error-raises', R, '            except Exception as e:  # noqa: BLE001\n                return _out(f"RUN ERROR: {type(e).__name__}: {e}", OutcomeStatus.FAILED, changed=None,\n                            code="run_internal")', '            except ZeroDivisionError as e:\n                return _out(f"RUN ERROR: {type(e).__name__}: {e}", OutcomeStatus.FAILED, changed=None,\n                            code="run_internal")'),
    # registry
    ('R1 ended-not-stamped', RR, '                if state in ENDED:\n                    c.execute("UPDATE runs SET state=?, state_note=?, last_probe_at=?, "', '                if False:\n                    c.execute("UPDATE runs SET state=?, state_note=?, last_probe_at=?, "'),
    ('R2 end-restamped', RR, '                              "ended_at=COALESCE(ended_at, ?) WHERE run_id=?"', '                              "ended_at=? WHERE run_id=?"'),
    ('R3 live-includes-ended', RR, '            if r["name"] == name and r["state"] not in ENDED:', '            if r["name"] == name:'),
    ('R4 rows-all-owners', RR, '            if owner is not None:\n                where.append("owner = ?")\n                a.append(owner)', '            if False:\n                where.append("owner = ?")\n                a.append(owner)'),
    ('R5 latest-is-first', RR, '        return rows[-1] if rows else None', '        return rows[0] if rows else None'),
    ('R6 prune-live', RR, '                c.execute("DELETE FROM runs WHERE ended_at IS NOT NULL AND ended_at < ?", (now - PRUNE_AFTER_S,))', '                c.execute("DELETE FROM runs WHERE started_at < ?", (now - 1,))'),
    ('R7 live-only-ignored', RR, '            if live_only:\n                where.append("ended_at IS NULL")', '            if False:\n                where.append("ended_at IS NULL")'),
    # reconcile + boot line
    ('RC1 labs-probed', R, '        if r["backend"] not in ("services", "jobs"):\n            out["labs"] += 1\n            continue', '        if False:\n            out["labs"] += 1\n            continue'),
    ('RC2 probe-error-aborts', R, '        except Exception as e:  # noqa: BLE001 — review: one raising probe must not end the whole pass', '        except ZeroDivisionError as e:'),
    ('RC3 refs-after-pass', R, '    live_refs = {(r["backend"], r["backend_ref"]) for r in rows + reg.rows(live_only=True)}', '    live_refs = {(r["backend"], r["backend_ref"]) for r in reg.rows(live_only=True)}'),
    ('RC3b refs-all-rows', R, '    live_refs = {(r["backend"], r["backend_ref"]) for r in rows + reg.rows(live_only=True)}', '    live_refs = {(r["backend"], r["backend_ref"]) for r in reg.rows()}'),
    ('RC4 untracked-all', R, '                if k and ("services", k) not in live_refs:', '                if k:'),
    ('RC5 orphan-any-job', R, '                if (e.get("started_via") == "run" and e.get("state") == "running"', '                if (e.get("state") == "running"'),
    ('RC6 orphan-ended-job', R, '                if (e.get("started_via") == "run" and e.get("state") == "running"', '                if (e.get("started_via") == "run"'),
    ('RC7 orphan-tracked', R, '                        and ("jobs", str(e.get("id"))) not in live_refs):', '                        and True):'),
    ('RC8 unknown-uncounted', R, '        out["probed"] += 1\n        if state == "unknown":\n            out["unknown"] += 1\n        elif state in _ENDED:', '        out["probed"] += 1\n        if False:\n            out["unknown"] += 1\n        elif state in _ENDED:'),
    ('RC9 ended-unreported', R, '        elif state in _ENDED:\n            out["ended"][r["run_id"]]', '        elif False:\n            out["ended"][r["run_id"]]'),
    ('RC10 boot-never-warns', R, '        warn = bool(rep["ended"] or rep["orphan_run_jobs"] or rep.get("error"))', '        warn = False'),
    ('RC11 boot-raises', R, '    except Exception as e:  # noqa: BLE001 — a background boot task never raises', '    except ZeroDivisionError as e:'),
    ('RC12 boot-unwired', MN, '        from .tools.runs import start_runs_boot_line\n        start_runs_boot_line(context)', '        from .tools.runs import start_runs_boot_line'),
    ('RC13 task-unheld', R, '    _BOOT_TASKS.add(t)\n    t.add_done_callback(_BOOT_TASKS.discard)', '    t.add_done_callback(_BOOT_TASKS.discard)'),
    # jobs / delegate / main / routes
    ('B1 started-via-not-written', J, '            if started_via:\n                # §4SX C', '            if False:\n                # §4SX C'),
    ('B2 dup-not-flagged', J, '            return {**existing, "duplicate": True}, ""', '            return existing, ""'),
    ('B3 cap-ignored', J, '            if len(live) >= max_concurrent_jobs():\n                return None, (f"{len(live)}/{max_concurrent_jobs()} background jobs already running — "\n                              "stop one first")', '            if False:\n                return None, (f"{len(live)}/{max_concurrent_jobs()} background jobs already running — "\n                              "stop one first")'),
    ('B4 promote-none-left-running', J, '            if entry is None:                      # the cap filled in between\n                self._kill_pgroup(pid)', '            if entry is None:                      # the cap filled in between\n                pass'),
    ('B5 crash-leaves-process', J, '        except BaseException:\n            # same guard as run(): no live process without a row\n            pid = self._read_pid(jid) or launcher_pid\n            if pid:', '        except BaseException:\n            # same guard as run(): no live process without a row\n            pid = self._read_pid(jid) or launcher_pid\n            if False:'),
    ('B6 no-ttl', J, '        self._write_script(jid, cmd, job_ttl_s())\n        launcher_pid = None\n        try:\n            launcher_pid = self._launch(jid, {"workdir": wd})', '        self._write_script(jid, cmd, 10 ** 9)\n        launcher_pid = None\n        try:\n            launcher_pid = self._launch(jid, {"workdir": wd})'),
    ('B7 delegate-adopts-run', D, '        if entry.get("started_via"):\n            continue        # §4SX C', '        if False:\n            continue        # §4SX C'),
    ('B8 hint-always-jobs', MN, '    if (entry or {}).get("started_via") == "run":\n        return f"run(action=\'collect\', name=\'{str((entry or {}).get(\'label\')', '    if False:\n        return f"run(action=\'collect\', name=\'{str((entry or {}).get(\'label\')'),
    ('B9 resume-hint-jobs', MN, '_line + ". Read it with " + _collect_hint(entry, context) + ".",', '_line + ". Read it with jobs(action=\'collect\', job_id=…).",'),
]
PINS = ["tests/test_4sx_c_run_tool.py", "tests/test_4sx_c_run_battery.py", "tests/test_sandbox_job_promotion.py", "tests/test_sandbox_services.py", "tests/test_4ud_labs.py"]


def _tests():
    pat = r"tools.runs\|tool_run\|run_registry\|start_detached\|started_via\|_sync_sandbox_jobs\|_collect_hint\|run_probe\|sandbox.jobs\|get_job_supervisor\|_reap_sandbox_jobs"
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
