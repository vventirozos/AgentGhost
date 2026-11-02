#!/usr/bin/env python3
"""§4SU Phase 2 battery — a task's declared kind (core/task_kind.py) routes it (advancer, close gate, store, tool, web).
Coverage-selected, parallel (scripts/battery_cov.py); the real src/ is never mutated.
No-op control must SURVIVE, known-bad control must DIE.    python3 scripts/mutate_4su2.py [-jN] [all|rerun ID …]
"""
import os, subprocess, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TK = 'src/ghost_agent/core/task_kind.py'
TP = 'src/ghost_agent/tools/projects.py'
PL = 'src/ghost_agent/core/planning.py'
RT = 'src/ghost_agent/api/projects_routes.py'
PA = 'src/ghost_agent/core/project_advancer.py'
EG = 'src/ghost_agent/core/experiment_gate.py'
TC = 'src/ghost_agent/core/task_close.py'
MS = 'src/ghost_agent/memory/projects.py'
MUT = [
    ('CONTROL known-bad', TK, '        raise ValueError(f"unknown task kind', '        return ""\n        raise ValueError(f"unknown task kind'),
    ('CONTROL no-op', TK, 'def with_deliverable(description: str, deliverable: Optional[str]) -> str:', 'def with_deliverable(description: str, deliverable: Optional[str]) -> str:  # no-op'),
    ('V1 alias-lost', TK, '"experiment": "model_run", ', ''),
    ('V2 none-is-unknown', TK, '    if not v or v in ("none", "auto", "default", "-"):', '    if not v:'),
    ('V3 deliverable-twice', TK, '    if not rel or declared_deliverable(desc):', '    if not rel:'),
    ('V4 owner-routed-research', TK, '"model_run": "model_run", "owner": "needs_user"}', '"model_run": "model_run", "owner": "research"}'),
    ('S1 store-add-drops-kind', MS, ' estimated_cost, 0.0, depth, position, kind, now, now),', ' estimated_cost, 0.0, depth, position, "", now, now),'),
    ('S2 store-update-unnormalised', MS, '            fields = {**fields, "kind": normalize_kind(fields["kind"])}   # §4SU Phase 2', '            pass'),
    ('S3 no-migration', MS, '                ("kind", "TEXT NOT NULL DEFAULT \'\'"),', ''),
    ('P1 not-hydrated', PL, '                kind=str(row.get("kind") or ""),', ''),
    ('P2 not-rendered', PL, '        _k = f"<{node.kind}> " if getattr(node, "kind", "") else ""', '        _k = ""'),
    ('P3 decompose-drops-kinds', PL, 'kind=kinds[i] if i < len(kinds) else "")   # §4SU Phase 2', 'kind="")   # §4SU Phase 2'),
    ('T1 objects-not-split', TP, '        subtasks, _subtask_kinds = _split_declared_subtasks(subtasks)', '        _subtask_kinds = {}'),
    ('T2 deliverable-dropped', TP, '            desc = with_deliverable(desc, item.get("deliverable"))', '            pass'),
    ('T3 json-objects-not-read', TP, '    if isinstance(raw, str) and re.match(r"\\s*\\[\\s*\\{", raw):', '    if False:'),
    ('T4 task-add-no-kind', TP, '                kind=task_kind,', ''),
    ('T5 update-ignores-kind', TP, '                if _task_kind_given:\n                    extras["kind"] = task_kind', '                if False:\n                    extras["kind"] = task_kind'),
    ('T6 none-not-a-clear', TP, '        _task_kind_given = bool(str(task_kind or "").strip())', '        _task_kind_given = bool(task_kind and str(task_kind).strip().lower() != "none")'),
    ('T7 bad-kind-swallowed', TP, '    except ValueError as _ke:\n        return _err(str(_ke))', '    except ValueError as _ke:\n        _subtask_kinds = {}'),
    ('A1 declared-ignored', PA, '    if _declared not in ROUTE_LABEL:\n        _declared = ""', '    if True:\n        _declared = ""'),
    ('A2 no-gate-floor', PA, '        _whole = classification != "needs_user" and needs_user_gate(nxt.description)', '        _whole = False'),
    ('A3 declared-routed-by-words', PA, '    if not _declared or (_declared == "research" and declared_deliverable(nxt.description)):', '    if True:'),
    ('A4 declared-research-file-not-built', PA, '    if not _declared or (_declared == "research" and declared_deliverable(nxt.description)):', '    if not _declared:'),
    ('A5 coding-skips-declared-run', PA, '    if (classification == "model_run"\n', '    if ((classification == "model_run" and _proj_kind != "CODING")\n'),
    ('A6 declared-settled', PA, '    _settled = (classification if _declared\n', '    _settled = (classification if False\n'),
    ('A7 introspective-declared-jams', PA, '    if (classification == "needs_user" and _declared != "owner"\n', '    if (classification == "needs_user" and not _declared\n'),
    ('A8 owner-exempted', PA, '    if (classification == "needs_user" and _declared != "owner"\n', '    if (classification == "needs_user"\n'),
    ('A9 introspective-build-rebuilt', PA, '            classification = _route_project_work(      # r9: a promised file still routes', '            classification = ROUTE_LABEL[_declared] if _declared else _route_project_work(      # r9'),
    ('A10 kind-removes-hold', PA, '            or (_proj_kind != "CODING" and classification in ("research", "coding")', '            or (not _declared and _proj_kind != "CODING" and classification in ("research", "coding")'),
    ('A11 no-gate-form-floor', PA, 'not owner_requested and (_whole or gate_form(nxt.description)):', 'not owner_requested and (_whole):'),
    ('A13 owner-run-held-by-form', PA, 'elif classification != "needs_user" and not owner_requested and (_whole', 'elif classification != "needs_user" and (_whole'),
    ('A14 form-hold-is-a-park', PA, '            _why = (f"a gate word — the owner\'s run decides: "\n                    f"{task_topic(nxt.description)[:80]} — {_HELD_MARK}")', '            _why = "flagged for human review"'),
    ('A12 introspective-run-released', PA, '        if _declared == "model_run":\n            # §4SU Phase 2: a declared model run stays one', '        if False:\n            # §4SU Phase 2: a declared model run stays one'),
    ('E1 model-run-by-words', EG, '        by_words = (kind == "model_run"\n                    or is_model_experiment(', '        by_words = (False\n                    or is_model_experiment('),
    ('E2 kind-removes-gate', EG, '        by_words = (kind == "model_run"\n                    or is_model_experiment(', '        by_words = (kind == "model_run"\n                    or not kind and is_model_experiment('),
    ('T8 string-objects-not-read', TP, '        if isinstance(item, str) and item.strip().startswith("{") and item.strip().endswith("}"):', '        if False:'),
    ('T10 bare-object-text', TP, '        if _is_subtask_object(_one):\n            raw = [_one]', '        if False:\n            raw = [_one]'),
    ('T11 shapeless-object-kept-text', TP, '"kind", "task_kind", "deliverable"))\n', '))\n'),
    ('T12 brace-text-parsed', TP, '    return isinstance(obj, dict) and any(', '    return isinstance(obj, dict) or any('),
    ('A15 introspective-whole-to-brief', PA, '        if _whole and (owner_requested or not is_self_referential(nxt.description)):', '        if _whole:'),
    ('T9 broken-list-accepted', TP, '            raise ValueError("`subtasks` looks like a JSON list of objects but does not parse — "', '            return value, {}\n            raise ValueError("`subtasks` looks like a JSON list of objects but does not parse — "'),
    ('C1 close-ignores-kind', TC, '                                            kind=str(getattr(node, "kind", "") or "")),', '                                            kind=""),'),
    ('B1 briefing-no-kind', 'src/ghost_agent/core/prompts.py', '                _kd = f"<{n.kind}> " if getattr(n, "kind", "") else ""', '                _kd = ""'),
    ('W1 web-create-no-kind', RT, '            kind=body.kind,\n        )', '        )'),
]


PINS = ["tests/test_4su_phase2.py", "tests/test_introspective_needs_user.py", "tests/test_4st_experiment_gate.py"]


def _tests():
    pat = r"kind\|advance_once\|check_close\|task_decompose\|add_task"
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
