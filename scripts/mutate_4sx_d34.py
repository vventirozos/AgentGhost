#!/usr/bin/env python3
"""§4SX Phase D3+D4 battery — kept machines (tools/labs.py keep/destroy/tar, tools/runs.py app runs + down sweep),
the autopilot lab holds (core/lab_waits.py, core/project_advancer.py, core/coding_loop.py) and project-owned
schedules (tools/tasks.py, tools/projects.py teardown). Coverage-selected, parallel (scripts/battery_cov.py); the
real src/ is never mutated. No-op must SURVIVE, known-bad must DIE.
    python3 scripts/mutate_4sx_d34.py [-jN] [all|check|rerun ID …]
"""
import os, subprocess, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L = 'src/ghost_agent/tools/labs.py'
R = 'src/ghost_agent/tools/runs.py'
W = 'src/ghost_agent/core/lab_waits.py'
A = 'src/ghost_agent/core/project_advancer.py'
C = 'src/ghost_agent/core/coding_loop.py'
P = 'src/ghost_agent/tools/projects.py'
T = 'src/ghost_agent/tools/tasks.py'
G = 'src/ghost_agent/tools/registry.py'
MUT = [
    ('CONTROL known-bad', L, '        if action in KEPT_ACTIONS:                 # production: refused before any host call', '        if False:'),
    ('CONTROL no-op', W, 'def holds_of(project: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:', 'def holds_of(project: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:  # no-op'),
    # leaf / autopilot lab access
    ('L1 no-owner-refusal', L, '            why = await _autopilot_owner_refusal(context, lab_id, owner)\n            if why:', '            why = ""\n            if why:'),
    ('L2 all-owners-ok', L, '        if action == "list" and _truthy(all_owners):\n            return ToolOutcome("LAB ERROR (not_allowed): unattended', '        if False:\n            return ToolOutcome("LAB ERROR (not_allowed): unattended'),
    ('L3 owner-404-unknown', L, '        theirs = str((row or {}).get("owner") or "")\n        if not theirs:', '        theirs = ""\n        if not theirs:'),
    ('L4 no-wait-note', L, '    if origin == "autopilot":\n        _note_lab_wait(out, action)', '    if False:\n        _note_lab_wait(out, action)'),
    ('L5 kept-probe-anywhere', L, '        return "" if _probe_project(context) else ("a probe uses', '        return "" if True else ("a probe uses'),
    ('L6 secret-values-ok', L, '        if "secret_values" in p:', '        if False:'),
    ('L7 probe-keep-any-owner', L, '        if origin == "probe" and str(det.get("owner") or "") != f"project:{_cur}":', '        if False:'),
    ('L8 drop-data-always', L, '            if others == []:\n                params["drop_data"] = "1"', '            if True:\n                params["drop_data"] = "1"'),
    ('L9 tar-blocking-open', L, 'os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),', 'os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),'),
    ('L10 leaf-filter-gone', L, '            if r.get("origin") == "autopilot" and str(r.get("leaf") or "") == str(leaf_id)]', '            if r.get("origin") == "autopilot"]'),
    ('L11 destroy-kept-unattended', L, '            if _kept_refusal(origin, context):\n                return ToolOutcome(f"LAB ERROR (not_allowed): {lab_id} is KEPT', '            if False:\n                return ToolOutcome(f"LAB ERROR (not_allowed): {lab_id} is KEPT'),
    ('L12 no-empty-owner-refusal', L, '        if not owner and action in ("create", "list"):', '        if False:'),
    ('L13 no-leaf-tag', L, '    _tag = leaf_tag(_LEAF.get()) if origin == "autopilot" else ""', '    _tag = ""'),
    ('L14 untracked-ignored', L, '                rows.append({"lab_id": str(x["lab_id"]), "untracked": True})', '                pass'),
    ('L15 untracked-any-tag', L, '                    and str(x.get("purpose") or "").endswith(tag)):', '                    ):'),
    ('L16 untracked-kept', L, '                    and str(x.get("lifetime") or "disposable") != "kept"', '                    and True'),
    ('L17 untracked-any-origin', L, '                    and str(x.get("origin") or "") == "autopilot"           # review r5 #3: never the owner\'s copy', ''),
    ('L18 recipe-keeps-tag', L, '                           + _cap(json.dumps(_recipe_without_leaf(body), indent=1), "spec"),', '                           + _cap(json.dumps(body, indent=1), "spec"),'),
    ('L20 leaf-ttl-not-dropped', L, '        body.pop("ttl_idle_s", None)\n        st, d = await _call(', '        st, d = await _call('),
    ('L21 untracked-404-not-gone', L, '            if st in (200, 204) or left or (st == 404 and isinstance(body, dict)', '            if st in (200, 204) or left or (False and isinstance(body, dict)'),
    ('L19 tag-past-cut', L, '        body["purpose"] = f"{str(body[\'purpose\'])[:100]} {_tag}".strip()', '        body["purpose"] = f"{str(body[\'purpose\'])} {_tag}".strip()'),
    # app runs / down sweep
    ('R1 staged-not-down', R, '    if state in ("crashloop", "staged") or detail.get("app_state") == "restarting":', '    if state in ("crashloop",) or detail.get("app_state") == "restarting":'),
    ('R2 owner-stop-down', R, '    return state == "app_stopped" and bool(detail.get("last_error")) and not detail.get("stopped_by_owner")', '    return state == "app_stopped" and bool(detail.get("last_error"))'),
    ('R3 one-poll-notice', R, '            if polls >= 2 and not m.get("down_notified"):', '            if polls >= 1 and not m.get("down_notified"):'),
    ('R4 probe-rows-page', R, '    rows = [r for r in reg.rows(live_only=True) if r["backend"] == "labs_app" and r.get("origin") != "probe"]', '    rows = [r for r in reg.rows(live_only=True) if r["backend"] == "labs_app"]'),
    ('R5 no-adopt', R, '    await _adopt_app_rows(context, reg)\n', '    pass\n'),
    # lab_waits
    # (W1 any-ok-clears is EQUIVALENT since r3 #3: note_outcome returns early on any action but create)
    ('W2 no-needed-mb', W, '    if not isinstance(health, dict) or health.get("ok") is not True:\n        return False', '    needed_mb = 0\n    if not isinstance(health, dict) or health.get("ok") is not True:\n        return False'),
    ('W3 down-ignores-ok', W, '    if not isinstance(health, dict) or health.get("ok") is not True:', '    if not isinstance(health, dict):'),
    ('W4 capacity-no-slot', W, '        return (int(health["labs"]) < int(health["max_labs"]) and free_pool >= need', '        return (free_pool >= need'),
    ('W5 unconfigured-keeps', W, '                unconfigured = True           # no labs host any more: nothing to wait for', '                pass'),
    ('W6 stale-kept', W, '    stale = [k for k in holds if k not in pending]', '    stale = []'),
    ('W7 no-answer-not-wait', W, '    "lab_no_answer": "labs_no_answer",', ''),
    ('W8 no-cooldown', W, '            return (time.time() - float(held_at or 0) >= NO_ANSWER_COOLDOWN_S', '            return (True'),
    ('W9 fixed-disk', W, 'int(health["disk_free_mb"]) >= int(health.get("disk_floor_mb") or ROOM_DISK_MB) + DISK_MARGIN_MB)', 'int(health["disk_free_mb"]) >= 2048)'),
    ('W10 note-any-action', W, '    if not build_key or action != "create":', '    if not build_key:'),
    ('W11 held-at-dropped', W, '                                          held_at=float(v.get("at") or 0))]', '                                          held_at=0.0)]'),
    # advancer
    ('A1 rounds-unbounded', A, '        if rounds_of(store, project_id, nxt.id) >= MAX_ROUNDS:', '        if False:'),
    ('A2 hold-after-release', A, '        set_hold(store, project_id, nxt.id, _lw, str(cres.summary or ""), int(getattr(cres, "lab_needed_mb", 0) or 0))\n        try:\n            store.release_claim(nxt.id, failure_reason=f"code_executor: {cres.summary}"[:300])\n        except Exception:  # noqa: BLE001\n            logger.debug("lab-wait leaf not reset", exc_info=True)', '        try:\n            store.release_claim(nxt.id, failure_reason=f"code_executor: {cres.summary}"[:300])\n        except Exception:  # noqa: BLE001\n            logger.debug("lab-wait leaf not reset", exc_info=True)\n        set_hold(store, project_id, nxt.id, _lw, str(cres.summary or ""), int(getattr(cres, "lab_needed_mb", 0) or 0))'),
    ('A3 no-lab-wait-branch', A, '            pass\n    if _lw:', '            pass\n    if False:'),
    ('A13 other-failure-keeps-rounds', A, '    if not _lw and not cres.ok:\n        try:', '    if False:\n        try:'),
    ('A4 owner-honours-hold', A, '        if _lab_holds and not owner_requested:', '        if _lab_holds:'),
    ('A5 held-not-excluded', A, '        _ex = set(exclude_ids or ()) | _lab_held', '        _ex = set(exclude_ids or ())'),
    ('A6 claim-keeps-hold', A, '    if nxt.id in _lab_holds:\n        from .lab_waits import clear_holds', '    if False:\n        from .lab_waits import clear_holds'),
    ('A7 wait-scored', A, '    if str(res.summary or "").startswith(LAB_WAIT_PREFIX):\n        return res', '    if False:\n        return res'),
    ('A10 gave-up-as-wait', A, 'return AdvanceResult(True, nxt.id, "blocked", f"{LAB_GAVE_UP_PREFIX} [{_lw}]: {_why}", None)', 'return AdvanceResult(True, nxt.id, "blocked", f"{LAB_WAIT_PREFIX} [{_lw}] gave up: {_why}", None)'),
    ('A11 gave-up-no-stop', A, '            stop_reason = "lab_gave_up"', '            stop_reason = "needs_user"'),
    ('A8 no-rounds-reset', A, '            clear_rounds(store, project_id, nxt.id)        # §4SX D4: it got its lab', '            pass        # §4SX D4: it got its lab'),
    ('A9 batch-no-stop', A, '        if res.task_id and res.summary.startswith(LAB_WAIT_PREFIX):', '        if False:'),
    ('A12 gave-up-keeps-rounds', A, '            clear_rounds(store, project_id, nxt.id)     # review r5 #1', '            pass     # review r5 #1'),
    # coding loop
    ('C1 build-key-per-attempt', C, '    _tk = (BUILD_KEY.set(str(leaf_id).rsplit("-a", 1)[0]), LEAF_ID.set(str(leaf_id)))', '    _tk = (BUILD_KEY.set(str(leaf_id)), LEAF_ID.set(str(leaf_id)))'),
    ('C2 no-take-seen', C, '    _lw, _need = take_seen(leaf_base)', '    _lw, _need = "", 0'),
    # schedules / teardown
    ('S1 refused-delete-drops', P, '            if not hard or rep["live_left"]:\n                # review r2 #4', '            if not hard:\n                # review r2 #4'),
    ('S2 inv-no-schedules', P, '        out["schedules"] = project_schedules(project_id)          # §4SX D4', '        pass'),
    ('S3 digest-no-sched', P, '    ids += [f"sched:{e.get(\'id\')}" for e in inv.get("schedules") or []]', '    pass'),
    ('S4 teardown-order', P, '    mine = sorted(mine, key=lambda x: (str(x.get("lifetime") or "") == "kept", bool(x.get("app_mounted"))))', '    pass'),
    ('S5 sched-any-owner', T, '            for k, v in tasks.items() if isinstance(v, dict) and v.get("owner") == owner]', '            for k, v in tasks.items() if isinstance(v, dict)]'),
    ('S6 no-ctx-project', T, '    if action in ("create", "watch") and str(project_owned).strip().lower() in ("true", "1", "yes"):', '    if False:'),
    ('S7 model-project-id', G, '_ctx_project_id=getattr(context, "current_project_id", None)),', '_ctx_project_id=kwargs.get("_ctx_project_id")),'),
    ('S8 holds-not-system', P, '                         "lab_waits", "lab_waits_rounds")', '                         )'),
]
PINS = ["tests/test_4sx_d3_kept.py", "tests/test_4sx_d4_autopilot_labs.py", "tests/test_4sx_b_lab_tool.py",
        "tests/test_4sx_c_run_battery.py"]


def _tests():
    pat = r"lab_waits\|tool_lab\|tools.labs\|tools.runs\|project_advancer\|run_leaf_turn\|manage_tasks\|_teardown\|project_schedules"
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
