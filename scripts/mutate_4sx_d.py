#!/usr/bin/env python3
"""§4SX Phase D2 battery — the project teardown (tools/projects.py _project_inventory/_teardown_*, the delete preview
digest, delete/archive/status) and the console DELETE/PATCH routes. Coverage-selected, parallel (scripts/battery_cov.py);
the real src/ is never mutated. No-op must SURVIVE, known-bad must DIE.
    python3 scripts/mutate_4sx_d.py [-jN] [all|check|rerun ID …]
"""
import os, subprocess, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = 'src/ghost_agent/tools/projects.py'
PR = 'src/ghost_agent/api/projects_routes.py'
MUT = [
    ('CONTROL known-bad', P, '            if _td["live_left"]:\n                return _err(f"NOT deleted: still running', '            if False:\n                return _err(f"NOT deleted: still running'),
    ('CONTROL no-op', P, 'def _teardown_summary(rep: Dict[str, list]) -> str:', 'def _teardown_summary(rep: Dict[str, list]) -> str:  # no-op'),
    # inventory
    ('I1 inv-runs-all', P, '        out["runs"] = reg.rows(owner, live_only=True) if reg is not None else []', '        out["runs"] = []'),
    ('I2 inv-labs-none', P, '        out["labs"] = [dict(r) for r in (inv.rows(owner, live_only=True) if inv is not None else [])]', '        out["labs"] = []'),
    ('I3 inv-jobs-any-project', P, '                           if str(e.get("project_id") or "") == str(project_id)\n                           and e.get("state") == "running"', '                           if True\n                           and e.get("state") == "running"'),
    ('I4 digest-no-labs', P, '    ids += [f"lab:{r.get(\'lab_id\')}" for r in inv.get("labs") or []]', '    pass'),
    ('I5 digest-no-runs', P, '    ids += [f"run:{r.get(\'run_id\')}" for r in inv.get("runs") or []]', '    pass'),
    ('I6 text-no-runs', P, '        bits.append("runs " + ", ".join(', '        bits.append("" + ", ".join('),
    # preview
    ('C1 grew-ignored-token', P, '    _grew = _grew_since_preview(context, act, rid, plan, host_labs)\n    if _grew:', '    _grew = _grew_since_preview(context, act, rid, plan, host_labs)\n    if False:'),
    ('C2 grew-ignored-earlier', P, '        if _plan0 is not None and not _grew_since_preview(context, act, rid, _plan0, host_labs):', '        if _plan0 is not None:'),
    ('C3 preview-no-digest', P, '            _plan_new["inventory"] = _inventory_ids(_inv)', '            pass'),
    ('C4 preview-no-text', P, '            _also = f" It would also stop/destroy: {_txt}." if _txt else ""', '            _also = ""'),
    # (C5 grew-any-act is EQUIVALENT: only a delete preview stores `inventory` in its plan)
    # runs step
    ('R1 archive-stops-app', P, '            if not hard and row["lifetime"] == "app":', '            if False:'),
    ('R2 run-fail-not-left', P, '                rep["failed"].append(f"{label}: {str(out)[:200]}")\n                rep["live_left"].append(label)', '                rep["failed"].append(f"{label}: {str(out)[:200]}")'),
    ('R3 sandbox-down-blocks', P, '            elif getattr(out, "reason_code", None) == "sandbox_unavailable":', '            elif False:'),
    ('R4 lost-not-recorded', P, '                reg.set_state(row["run_id"], "lost", "the sandbox was not running at project teardown")', '                pass'),
    ('R5 run-raise-escapes', P, '            except Exception as e:  # noqa: BLE001\n                rep["failed"].append(f"{label}: {type(e).__name__}: {e}")', '            except ZeroDivisionError as e:\n                rep["failed"].append(f"{label}: {type(e).__name__}: {e}")'),
    ('R6 origin-dropped', P, '                out = await _runs._stop(context, reg, row, origin)', '                out = await _runs._stop(context, reg, row, "autopilot")'),
    # services step
    ('S1 svc-not-stopped', P, '        n = await asyncio.to_thread(_stop_project_services, context, project_id, hard)', '        n = 0'),
    ('S2 svc-survivor-ignored', P, '                alive = sup is not None and await asyncio.to_thread(sup._entry_state, e) is not False', '                alive = False'),
    # jobs step
    ('J1 jobs-not-cancelled', P, '            ok, msg = await asyncio.to_thread(jsup.cancel, e["id"])', '            ok, msg = True, ""'),
    ('J2 jobs-any-project', P, '            if str(e.get("project_id") or "") != str(project_id) or e.get("state") != "running":', '            if e.get("state") != "running":'),
    ('J3 job-fail-not-left', P, '                rep["failed"].append(f"job {e[\'id\']}: {msg}")\n                rep["live_left"].append(f"job {e[\'id\']}")', '                rep["failed"].append(f"job {e[\'id\']}: {msg}")'),
    # labs step
    ('L1 labs-skipped', P, '    await _teardown_labs(context, project_id, owner, hard=hard, origin=origin, rep=rep)', '    pass'),
    ('L2 unconfigured-not-left', P, '            rep["live_left"].append(f"lab {r[\'lab_id\']} (no labs host is configured to destroy it)")', '            pass'),
    ('L3 down-not-left', P, '            rep["live_left"].append(f"lab {r[\'lab_id\']} (the labs host did not answer: {e})")', '            pass'),
    ('L4 others-labs-destroyed', P, '                if isinstance(x, dict) and str(x.get("owner") or "") == owner]', '                if isinstance(x, dict)]'),
    ('L5 archive-destroys-kept', P, '        if kept and not hard:', '        if False:'),
    ('L6 no-include-kept', 'src/ghost_agent/tools/labs.py', '    params = {"include_kept": "1"}', '    params = {}'),
    ('L7 drop-always', 'src/ghost_agent/tools/labs.py', '    if others == []:\n        params["drop_data"] = "1"\n        return params, ""', '    if True:\n        params["drop_data"] = "1"\n        return params, ""'),
    ('L8 drop-never', 'src/ghost_agent/tools/labs.py', '    if others == []:\n        params["drop_data"] = "1"\n        return params, ""', '    if False:\n        params["drop_data"] = "1"\n        return params, ""'),
    ('L9 sibling-is-self', 'src/ghost_agent/tools/labs.py', '                   and str(x.get("lab_id")) != lab_id and str(x.get("lifetime") or "") == "kept"', '                   and str(x.get("lifetime") or "") == "kept"'),
    ('L10 sibling-any-lineage', 'src/ghost_agent/tools/labs.py', '                   and lineage and str(x.get("lineage") or "") == lineage]', '                   and lineage]'),
    ('L11 lab-fail-not-left', P, '            rep["failed"].append(f"lab {lid}: HTTP {st} {b.get(\'error\') or \'\'}".rstrip())\n            rep["live_left"].append(f"lab {lid}")\n    for r in local:', '            rep["failed"].append(f"lab {lid}: HTTP {st} {b.get(\'error\') or \'\'}".rstrip())\n    for r in local:'),
    ('L12 any-404-ok', P, '        if st == 200 or (st == 404 and b.get("error") == "unknown_lab"):', '        if st in (200, 404):'),
    ('L13 destroy-not-recorded', P, '                inv.record_destroy(lid, by=f"project teardown {project_id}")', '                pass'),
    ('L14 lab-origin-dropped', P, '            st, b = await _call(context, "DELETE", f"/v1/labs/{lid}", origin=origin, params=params, timeout=120)', '            st, b = await _call(context, "DELETE", f"/v1/labs/{lid}", origin="autopilot", params=params, timeout=120)'),
    ('L15 list-non200-ok', P, '        if st != 200:\n            raise LabsNoAnswer(f"HTTP {st}")', '        if False:\n            raise LabsNoAnswer(f"HTTP {st}")'),
    # tool branches
    ('T1 delete-no-teardown', P, '            _td = await _teardown_project(context, rid, hard=True, origin=_teardown_origin(context))', '            _td = {"live_left": [], "stopped": [], "kept_up": [], "failed": [], "notes": []}'),
    ('T2 archive-hard', P, '            _td = await _teardown_project(context, rid, hard=False, origin=_teardown_origin(context))', '            _td = await _teardown_project(context, rid, hard=True, origin=_teardown_origin(context))'),
    ('T3 archive-no-kept-note', P, '                                   "whether to stop them." if _td["kept_up"] else "")', '                                   "whether to stop them." if False else "")'),
    ('T4 status-no-inventory', P, '                            "inventory": _inventory_text(_project_inventory(context, rid), hard=None) or "nothing running",', '                            "inventory": "nothing running",'),
    ('T5 origin-owner-always', P, '        return "autopilot" if refusal or not origin else origin', '        return "owner"'),
    # routes
    ('X1 route-probe-allowed', PR, '    if origin == "probe" and not (proj.get("metadata") or {}).get("probe_created"):\n        raise HTTPException(403, "a probe deletes or archives only a project it created")\n    from ..tools.projects import _teardown_project, _teardown_summary', '    if False:\n        raise HTTPException(403, "a probe deletes or archives only a project it created")\n    from ..tools.projects import _teardown_project, _teardown_summary'),
    ('X2 route-live-left-ignored', PR, '    if hard and td["live_left"]:', '    if False:'),
    ('X3 route-origin-autopilot', PR, '    td = await _teardown_project(ctx, pid, hard=hard, origin=origin)', '    td = await _teardown_project(ctx, pid, hard=hard, origin="autopilot")'),
    ('X4 patch-no-teardown', PR, '        td = await _teardown_project(_context(request), pid, hard=False, origin=origin)', '        td = {"stopped": [], "kept_up": [], "failed": [], "live_left": [], "notes": []}'),
    ('X5 route-no-episodes', PR, '        await asyncio.to_thread(_forget_project_episodes, ctx, pid, _title)', '        pass'),
    # review r2
    ('N1 everyone-not-updated', P, '            _destroyed.add(lid)\n', '            pass\n'),
    ('N2 data-claim-always', P, '                what = (f" with its data ({len(gone)} volume(s), final backup taken)" if gone', '                what = (f" with its data ({len(gone)} volume(s), final backup taken)" if True'),
    ('N3 tool-probe-allowed', P, '    if origin == "probe" and not (proj.get("metadata") or {}).get("probe_created"):\n        return f"a probe may', '    if False:\n        return f"a probe may'),
    ('N4 tool-unattended-allowed', P, '    if origin in ("unattended", "leaf", "member"):', '    if origin in ("member",):'),
    ('N5 delete-refusal-unchecked', P, '            _tr = _teardown_refusal(context, store.get_project(rid) or {}, "delete")\n            if _tr:', '            _tr = _teardown_refusal(context, store.get_project(rid) or {}, "delete")\n            if False:'),
    ('N6 archive-refusal-unchecked', P, '            _tr = _teardown_refusal(context, store.get_project(rid) or {}, "archive")\n            if _tr:', '            _tr = _teardown_refusal(context, store.get_project(rid) or {}, "archive")\n            if False:'),
    ('N7 digest-no-kept', P, '    ids += [f"kept:{r.get(\'lab_id\')}" for r in inv.get("labs") or [] if r.get("lifetime") == "kept"]', '    pass'),
    ('N8 host-labs-not-merged', P, '    if host_labs:\n        by_id = {', '    if False:\n        by_id = {'),
    ('N9 host-only-lab-dropped', P, '                out["labs"].append(row)\n', ''),
    ('N10 text-no-kept', P, '                         + (f", KEPT, lineage {r.get(\'lineage\') or \'?\'}" if r.get("lifetime") == "kept" else "")', '                         + ("")'),
    ('N11 prefetch-not-passed', P, '                _gate = _lifecycle_confirmation(store, act, rid, token, context, _hl)', '                _gate = _lifecycle_confirmation(store, act, rid, token, context)'),
    ('N12 ended-job-blocks', P, '            if now is None or (now.get("state") or "") != "running":', '            if False:'),
    ('N13 any-failed-cancel-ok', P, '            if now is None or (now.get("state") or "") != "running":', '            if True:'),
    ('N14 cancel-raise-escapes', P, '            except Exception as ex:  # noqa: BLE001 — one job', '            except ZeroDivisionError as ex:  # noqa: BLE001 — one job'),
    ('N15 svc-state-raise-gone', P, '                alive = True                                   # could not tell', '                alive = False                                   # could not tell'),
    ('N16 labs-raise-not-live', P, '            for r in inv.rows(owner, live_only=True, strict=True):\n                rep["live_left"].append(f"lab {r[\'lab_id\']}")', '            for r in []:\n                rep["live_left"].append(f"lab {r[\'lab_id\']}")'),
    ('N17 lab-delete-generic-escapes', P, 'params=params, timeout=120)\n        except Exception as e:  # noqa: BLE001', 'params=params, timeout=120)\n        except (LabsUnreachable, LabsNoAnswer) as e:'),
    ('N18 patch-no-report', PR, '        out = dict(out, teardown=_teardown_summary(td), kept_up=td["kept_up"], live_left=td["live_left"])', '        pass'),
    ('N19 refusal-text', P, 'listed as stopped/destroyed above already is.', 'Nothing else was removed.'),
    # review r3
    ('M1 preview-not-recorded', P, '                inv.record_create(lid, owner=owner, origin=str(h.get("origin") or ""),', '                (lambda *a, **k: None)(lid, owner=owner, origin=str(h.get("origin") or ""),'),
    ('M2 grew-any-id', P, '                  - set(plan.get("inventory") or []) if i.startswith(("kept:", "app:", "sched:")))', '                  - set(plan.get("inventory") or []))'),
    ('M3 digest-no-app', P, '    ids += [f"app:{r.get(\'run_id\')}" for r in inv.get("runs") or [] if r.get("lifetime") == "app"]', '    pass'),
    ('M4 registry-unread-none', P, '            rep["live_left"].append("runs (the run registry could not be read)")', '            pass'),
    ('M5 registry-lenient', P, '            live_runs = reg.rows(owner, live_only=True, strict=True) if reg is not None else []', '            live_runs = reg.rows(owner, live_only=True) if reg is not None else []'),
    ('M6 replaced-blocks', P, '            elif getattr(out, "reason_code", None) == "run_replaced":', '            elif False:'),
    ('M7 leftovers-ok', P, '        if st == 200 and b.get("leftovers"):\n            # the host destroyed it', '        if False:\n            # the host destroyed it'),
    ('M8 patch-teardown-first', PR, '        archiving = True\n    else:\n        archiving = False', '        archiving = True\n        from ..tools.projects import _teardown_project\n        await _teardown_project(_context(request), pid, hard=False, origin=origin)\n    else:\n        archiving = False'),
    ('M9 update-archived-unchecked', P, '                        return _err(_tr)                 # review r3 m6', '                        pass                 # review r3 m6'),
    ('M10 rr-strict-swallowed', 'src/ghost_agent/memory/run_registry.py', '            if strict:\n                raise', '            if False:\n                raise'),
    ('M11 li-strict-swallowed', 'src/ghost_agent/memory/lab_inventory.py', '            if strict:\n                raise', '            if False:\n                raise'),
    # review r4
    ('Q1 unlisted-assumed-gone', P, '        # not listed: destroyed earlier, or destroyed with leftovers the host still holds — ask it (review r3 #1)\n        try:', '        if inv is not None:\n            inv.record_destroy(lid, by="gone")\n        continue\n        try:'),
    ('Q2 unlisted-leftovers-ok', P, '        if st == 200 and b.get("leftovers"):\n            rep["failed"].append(f"lab {lid}: leftovers {str(b.get(\'leftovers\'))[:160]} — the operator must "', '        if False:\n            rep["failed"].append(f"lab {lid}: leftovers {str(b.get(\'leftovers\'))[:160]} — the operator must "'),
    ('Q3 other-host-deleted', P, '        if r.get("host_id") and r["host_id"] != host_now:', '        if False:'),
    ('Q4 summary-no-live-left', P, '    if rep.get("live_left"):\n        parts.append("STILL RUNNING', '    if False:\n        parts.append("STILL RUNNING'),
    ('Q5 archive-note-no-live', P, '                                   "teardown) — tell the owner." if _td["live_left"] else "")})', '                                   "teardown) — tell the owner." if False else "")})'),
    ('Q6 silent-host-covers', P, '    if host_labs is None and _labs_host_configured(context) and _project_inventory(context, rid)["labs"]:', '    if False:'),
    ('Q7 soft-delete-silent', PR, '    if _left is not None:\n        return JSONResponse(_left, status_code=200)', '    if False:\n        return JSONResponse(_left, status_code=200)'),
    ('Q8 unlisted-error-gone', P, '            rep["failed"].append(f"lab {lid}: HTTP {st} {b.get(\'error\') or \'\'}".rstrip())\n            rep["live_left"].append(f"lab {lid}")\n\n\ndef _teardown_summary', '            rep["failed"].append(f"lab {lid}: HTTP {st} {b.get(\'error\') or \'\'}".rstrip())\n\n\ndef _teardown_summary'),
    # review r5
    ('Z1 unknown-host-ok', P, '        if r.get("host_id") and r["host_id"] != host_now:', '        if r.get("host_id") and host_now and r["host_id"] != host_now:'),
    ('Z2 preview-row-no-host', P, '                                  template=str(h.get("template") or ""), label="recorded at a delete preview",\n                                  host_id=hid)', '                                  template=str(h.get("template") or ""), label="recorded at a delete preview",\n                                  host_id="")'),
    ('Z3 grew-before-yes', P, '    why = _confirm_allowed(plan)\n    if why:', '    _g0 = _grew_since_preview(context, act, rid, plan, host_labs)\n    if _g0:\n        _drop_plan(token)\n        return _err("NOT deleted: gained " + ", ".join(_g0))\n    why = _confirm_allowed(plan)\n    if why:'),
    ('Z4 unopened-registry-empty', P, '            if reg is None and getattr(context, "memory_dir", None):\n                raise RuntimeError("cannot be opened")', '            if False:\n                raise RuntimeError("cannot be opened")'),
    ('Z5 unopened-inventory-empty', P, '    if inv is None and getattr(context, "memory_dir", None):\n        raise RuntimeError("the lab inventory cannot be opened")', '    if False:\n        raise RuntimeError("the lab inventory cannot be opened")'),
    ('Z6 fallback-no-inv-silent', P, '            if inv is None:\n                raise RuntimeError("no lab inventory")', '            if inv is None:\n                return rep'),
    ('Z7 tool-leftovers-recorded', 'src/ghost_agent/tools/labs.py', '        if left:\n            # the host could not remove everything', '        if False:\n            # the host could not remove everything'),
    # review r6
    ('Y1 list-retires-leftovers', 'src/ghost_agent/tools/labs.py', '                if not r.get("destroyed_at") and r.get("leftovers"):\n                    # destroyed, but', '                if False:\n                    # destroyed, but'),
    ('Y2 retry-stops-at-get', 'src/ghost_agent/tools/labs.py', '        if (st == 404 and _row0 and _row0.get("leftovers") and not _row0.get("destroyed_at")', '        if (False and _row0 and _row0.get("leftovers") and not _row0.get("destroyed_at")'),
    ('Y3 retry-leftovers-cleared', 'src/ghost_agent/tools/labs.py', '            if st2 == 200 and left2:', '            if False:'),
    ('Y4 tool-no-mark', 'src/ghost_agent/tools/labs.py', '            if inv is not None:\n                inv.mark_leftovers(lab_id, ", ".join(map(str, left)))', '            pass'),
    ('Y5 teardown-no-mark', P, '                inv.mark_leftovers(lid, ", ".join(map(str, b.get("leftovers"))))   # r5: lab list keeps it', '                pass'),
    ('Y6 no-migration', 'src/ghost_agent/memory/lab_inventory.py', '            if "leftovers" not in cols:', '            if False:'),
    ('Y7 retry-probe-any-owner', 'src/ghost_agent/tools/labs.py', '                and (origin != "probe" or _row0.get("owner") == owner)):', '                and True):'),
    ('Y8 retry-other-host', 'src/ghost_agent/tools/labs.py', '                if _row0["host_id"] != _hnow:', '                if False:'),
]
PINS = ["tests/test_4sx_d2_inventory.py", "tests/test_4sx_c_run_battery.py", "tests/test_4sx_b_lab_tool.py", "tests/test_project_delete_and_guard.py", "tests/test_project_delete_ungated.py", "tests/test_4ud_labs.py"]


def _tests():
    pat = r"manage_projects\|_teardown\|projects_routes\|_lifecycle_confirmation\|delete_project"
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
