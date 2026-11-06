#!/usr/bin/env python3
"""§4SX B0 battery — project focus per conversation (request scope, focus rows, console routes, leaves).
Coverage-selected, parallel (scripts/battery_cov.py); the real src/ is never mutated.
No-op control must SURVIVE, known-bad control must DIE.    python3 scripts/mutate_4sx_b0.py [-jN] [all|rerun ID …]
"""
import os, subprocess, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = 'src/ghost_agent/core/agent.py'
P = 'src/ghost_agent/tools/projects.py'
S = 'src/ghost_agent/memory/projects.py'
R = 'src/ghost_agent/core/request_scope.py'
RT = 'src/ghost_agent/api/projects_routes.py'
CL = 'src/ghost_agent/core/coding_loop.py'
MUT = [
    ('CONTROL known-bad', P, '                store.set_focus(conv, project_id)', '                pass'),
    ('CONTROL no-op', P, 'def begin_request_focus(context) -> Optional[str]:', 'def begin_request_focus(context) -> Optional[str]:  # no-op'),
    # request_scope
    ('R1 scope-any-context', R, '    if context is not None and s.context is not context:\n        return None', '    if False:\n        return None'),
    ('R2 slack-prefixed', R, '    if sid.startswith("slack:"):\n        return sid', '    if False:\n        return sid'),
    # GhostContext properties
    ('G1 getter-ignores-scope', A, '            return s.project_id\n        return self.__dict__.get("_project_fallback")', '            pass\n        return self.__dict__.get("_project_fallback")'),
    ('G2 setter-to-fallback', A, '            s.project_id = value or None\n        else:', '            self.__dict__["_project_fallback"] = value or None\n        else:'),
    ('G3 copy-drops-value', A, '        new.__dict__["_project_fallback"] = self.current_project_id', '        pass'),
    ('G4 conv-getter-ignores-scope', A, '            return s.conversation_key\n        return self.__dict__.get("_conversation_fallback", "")', '            pass\n        return self.__dict__.get("_conversation_fallback", "")'),
    # scope builder
    ('S1 member-persists', A, '            return RequestScope(origin=ORIGIN_MEMBER, context=self.context)', '            return RequestScope(conversation_key=conversation_key_for_session(sid), origin=ORIGIN_MEMBER, persist=True, context=self.context)'),
    ('S3 probe-persists', A, '            return RequestScope(conversation_key=key, origin=ORIGIN_PROBE, context=self.context)', '            return RequestScope(conversation_key=key, origin=ORIGIN_PROBE, persist=persist, context=self.context)'),
    ('S4 unattended-off', A, '        if str(req_id).startswith(_UNOWNED_RID_PREFIXES) or turn_origin_value != "user":', '        if False:'),
    ('S5 passed-scope-ignored', A, '        if scope is not None:\n            scope.context = self.context\n            return scope', '        if False:\n            scope.context = self.context\n            return scope'),
    ('S6 no-session-persists', A, '        persist = bool(key)', '        persist = True'),
    ('S7 origin-check-off', A, 'str(req_id).startswith(_UNOWNED_RID_PREFIXES) or turn_origin_value != "user"', 'str(req_id).startswith(_UNOWNED_RID_PREFIXES)'),
    # handle_chat wiring
    ('H1 scope-not-installed', A, '            _scope_token = _scope_install(_req_scope)', '            _scope_token = None'),
    ('H2 scope-not-reset', A, '                _scope_reset(_scope_token)', '                pass'),
    ('H3 focus-not-loaded', A, '                    begin_request_focus(self.context)\n', '                    pass\n'),
    ('H4 conv-ctxvar-unset', A, '            _ckc.set(_req_scope.conversation_key)', '            pass'),
    ('F1 fingerprint-ignores-session', A, '            if _ck.startswith(("session:", "slack:")):', '            if False:'),
    ('D1 drain-scope-lost', A, '            _sc_tok = _sc_inst(_stream_scope) if _stream_scope is not None else None', '            _sc_tok = None'),
    ('D2 drain-scope-not-reset', A, '                if _sc_tok is not None:\n                    _sc_reset(_sc_tok)', '                if False:\n                    _sc_reset(_sc_tok)'),
    ('N1 hint-not-shown', A, '                            if _pnh:\n', '                            if False:\n'),
    ('N2 hint-for-members', A, '                        if not requester_is_member() and not reply_is_public():\n                            from ..tools.projects import project_name_hint', '                        if True:\n                            from ..tools.projects import project_name_hint'),
    # projects tool
    ('P1 writer-ignores-persist', P, '    if sc is None or not sc.persist or not sc.conversation_key:', '    if sc is None or not sc.conversation_key:'),
    ('P3 exit-keeps-row', P, '                store.clear_focus(conv)', '                pass'),
    ('P4 close-not-stamped', P, '            context.last_closed_project = (getattr(context, "conversation_key", None), prev)', '            pass'),
    ('P5 hint-names-current', P, '        if not pid or pid == cur or str(p.get("status") or "").upper() == "ARCHIVED":', '        if not pid:'),
    ('P6 hint-short-titles', P, '                or (len(title) >= 5 and re.search(', '                or (len(title) >= 1 and re.search('),
    ('P6b hint-4-char-titles', P, '                or (len(title) >= 5 and re.search(', '                or (len(title) >= 4 and re.search('),
    ('P7 hint-no-word-bound', P, 're.search(r"\\b" + re.escape(title) + r"\\b", msg)', 're.search(re.escape(title), msg)'),
    ('P8 forget-keeps-rows', P, '            store.clear_focus_for_project(pid)\n        except Exception:  # noqa: BLE001\n            logger.debug("focus clear skipped"', '            pass\n        except Exception:  # noqa: BLE001\n            logger.debug("focus clear skipped"'),
    ('B1 request-only-reads-row', P, '    if sc.persist and sc.conversation_key:', '    if sc.conversation_key:'),
    ('B2 no-hydrate', P, '        if pid or saved:\n            _hydrate_scratchpad(context, pid)', '        if False:\n            _hydrate_scratchpad(context, pid)'),
    ('B3 no-snapshot', P, '        saved = _snapshot_scratchpad(context, live) if live else True', '        saved = True'),
    ('B4 clear-without-snapshot', P, '        if pid or saved:', '        if True:'),
    ('B5 always-hydrate', P, '    if sp is not None and (live or None) != (pid or None):', '    if sp is not None:'),
    ('B6 non-str-pid', P, '        sc.project_id = pid if isinstance(pid, str) and pid else None', '        sc.project_id = pid'),
    ('B7 ctx-not-written', P, '        context.current_project_id = pid\n        context.conversation_key = sc.conversation_key', '        pass'),
    # r1: emptied snapshot, member hide, deleted dir, HTTP session focus, CLI
    ('E1 empty-not-recorded', P, '            if last and ((last[0].get("payload") or {}).get("keys") or {}):\n                store.log_event(project_id, None, "scratchpad_snapshot", {"keys": {}})', '            pass'),
    ('E2 empty-every-time', P, '            if last and ((last[0].get("payload") or {}).get("keys") or {}):\n                store.log_event', '            if True:\n                store.log_event'),
    ('L1 legacy-kept', P, '            scratchpad.delete(k)', '            pass'),
    # store
    # r1: M1 (archived row dropped on read) is now the behaviour — the archive clears rows at write time
    ('M1b archived-dropped-on-read', S, '            if row["status"] is None:\n                conn.execute', '            if row["status"] is None or str(row["status"]).upper() == "ARCHIVED":\n                conn.execute'),
    ('M2 gone-row-kept', S, '            if row["status"] is None:\n                conn.execute', '            if False:\n                conn.execute'),
    ('M3 no-refresh', S, '            conn.execute("UPDATE conversation_focus SET last_used_at = ? WHERE conversation_key = ?",\n                         (_now(), key))', '            pass'),
    ('M4 unknown-project', S, '            if conn.execute("SELECT 1 FROM projects WHERE id = ?", (pid,)).fetchone() is None:\n                return False', '            if False:\n                return False'),
    ('M5 no-prune', S, '            conn.execute("DELETE FROM conversation_focus WHERE last_used_at < ?",\n                         (now - self.FOCUS_IDLE_PRUNE_S,))', '            pass'),
    ('M6 prune-wrong-way', S, '"DELETE FROM conversation_focus WHERE last_used_at < ?"', '"DELETE FROM conversation_focus WHERE last_used_at > ?"'),
    ('M7 project-clear-noop', S, '            cur = conn.execute("DELETE FROM conversation_focus WHERE project_id = ?", (pid,))', '            cur = conn.execute("SELECT 1 WHERE 0")'),
    ('M8 archive-hook-off', S, '            if status_norm == ProjectStatus.ARCHIVED.value:\n                # §4SX B0', '            if False:\n                # §4SX B0'),
    ('M9 clear-noop', S, '            conn.execute("DELETE FROM conversation_focus WHERE conversation_key = ?", (key,))\n            conn.commit()\n\n    def clear_focus_for_project', '            conn.commit()\n\n    def clear_focus_for_project'),
    ('M10 set-overwrites-other', S, '"project_id = excluded.project_id, bound_at = excluded.bound_at, "', '"project_id = project_id, bound_at = excluded.bound_at, "'),
    # routes
    ('RT1 session-optional', RT, '    if required and not key:', '    if False:'),
    ('RT2 body-ignored', RT, '                sid = body.get("session_id")', '                sid = None'),
    ('RT3 list-current-none', RT, '        "current": store.get_focus(key) if key else None,', '        "current": None,'),
    ('RT4 switch-no-focus', RT, '    if not store.set_focus(key, pid):\n        raise HTTPException(404, "project not found")\n    return {"switched_to"', '    if False:\n        raise HTTPException(404, "project not found")\n    return {"switched_to"'),
    ('U1 upload-ignores-session', 'src/ghost_agent/api/routes.py', '    _explicit_pid = (_qp.get("project_id") if _qp else None) or _session_focus_pid(agent, request)', '    _explicit_pid = _qp.get("project_id") if _qp else None'),
    ('U2 download-ignores-session', 'src/ghost_agent/api/routes.py', '        pid = (_qp.get("project_id") if _qp else None) or _session_focus_pid(agent, request)', '        pid = _qp.get("project_id") if _qp else None'),
    ('U3 header-ignored', 'src/ghost_agent/api/routes.py', '    sid = (_qp.get("session_id") if _qp else None) or _hdrs.get("X-Ghost-Session")', '    sid = _qp.get("session_id") if _qp else None'),
    ('CLI1 switch-without-session', 'interface/externals/cli/ghost', '''        return self._json("POST", f"/api/projects/{quote(pid, safe='')}/switch",
                          body={"session_id": session_id})''', '''        return self._json("POST", f"/api/projects/{quote(pid, safe='')}/switch")'''),
    ('CLI2 list-without-session', 'interface/externals/cli/ghost', '                          params={"session_id": session_id} if session_id else None)\n', '                          params=None)\n'),
    ('CLI3 use-passes-no-session', 'interface/externals/cli/ghost', '            status, data = call(pid, self.session_id)', '            status, data = call(pid, "")'),
    ('DL1 proxy-drops-session', 'interface/server.py', '                         **({"X-Ghost-Session": _sid} if _sid else {})},', '                         },'),
    ('DL2 cli-download-no-session', 'interface/externals/cli/ghost', '            status, size = self.api.download(name, dest, self.session_id)', '            status, size = self.api.download(name, dest)'),
    ('SL1 slack-upload-no-session', 'interface/externals/slack_bot/main.py', '                         **({"X-Ghost-Session": session_id} if session_id else {})},', '                         },'),
    ('SL2 slack-history-no-session', 'interface/externals/slack_bot/main.py', '                    filename = await upload_file_to_agent(f, uploader=msg.get("user"),\n                                                          session_id=session_id)', '                    filename = await upload_file_to_agent(f, uploader=msg.get("user"))'),
    ('H5 failed-build-no-scope', A, '                _scope_token = _scope_install(_RS(context=self.context))', '                _scope_token = None'),
    ('SL3 slack-image-no-session', 'interface/externals/slack_bot/main.py', '                                 "X-Ghost-Session": _slack_session_id(thread_ts, requester)},', '                                 },'),
    ('SL4 handler-session-by-fetch-ts', 'interface/externals/slack_bot/main.py', '        current_text=user_text, session_id=_slack_session_id(thread_ts, event.get("user")))\n    if not messages:', '        current_text=user_text, session_id=_slack_session_id(fetch_ts, event.get("user")))\n    if not messages:'),
    ('WJ1 blob-cache-raw-key', 'interface/static/app.js', '    return `${sid}|${rawSrc}`;', '    return rawSrc;'),
    ('WJ2 download-header-dropped', 'interface/static/app.js', "                    if (_sid) headers.set('X-Ghost-Session', String(_sid));", "                    if (false) headers.set('X-Ghost-Session', String(_sid));"),
    ('CW1 clockwork-no-session', 'interface/externals/clockwork_ghost/client.py', '            headers["X-Ghost-Session"] = str(self.session_id)', '            pass'),
    ('H6 double-install', A, '            if _scope_token is None:                # r3: never a second install over the first', '            if True:'),
    ('WJ3 inflight-raw-key', 'interface/static/app.js', '    _authedBlobInflight.set(key, p);\n    const done = () => { _authedBlobInflight.delete(key); };', '    _authedBlobInflight.set(rawSrc, p);\n    const done = () => { _authedBlobInflight.delete(rawSrc); };'),
    # leaves
    ('C1 leaf-scope-not-passed', CL, '                          scope=leaf_scope),', '                          scope=None),'),
    ('C2 leaf-check-reads-iso', CL, '    if _want and leaf_scope.project_id != _want:', '    if _want and getattr(iso, "current_project_id", None) != _want:'),
]


PINS = ["tests/test_project_conversation_scope.py", "tests/test_4sx_b0_wiring.py", "tests/test_projects_api.py",
        "tests/test_4fh_leaf_pin_survives_reconcile.py", "tests/test_scratchpad_scope_fixes.py",
        "tests/test_4nw_webos_defects.py", "tests/test_requester_role.py", "tests/test_grounded_file_verify.py",
        "tests/test_4pe_files_projects_review.py", "tests/test_projects_tool.py", "tests/test_projects_context_switch.py",
        "tests/test_upload_project_scope.py", "tests/test_ghost_cli.py",
        "tests/test_interface_proxy_auth.py", "tests/test_4kw_slack_review.py",
        "tests/test_interface_workspace.py", "tests/test_member_download_owner_4qg.py",
        "tests/test_4ml_interfaces.py", "tests/test_interface_review_4pr.py"]


def _tests():
    pat = r"current_project_id\|conversation_key\|request_scope\|focus\|handle_chat\|projects_router"
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
