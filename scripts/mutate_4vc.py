#!/usr/bin/env python3
r"""§4VC battery — the task-tree scrubs are gone from both reply-cleanup sites (finalize, internal turn): a task
list the owner asked for reaches the owner. Each mutant puts one removed scrub back, at either site, or drops a
label strip that must stay. A no-op control must SURVIVE, a known-bad control must DIE.
      python3 scripts/mutate_4vc.py [-jN] [all|rerun M1 …|check]
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AG = "src/ghost_agent/core/agent.py"
F_LABEL = """        final_ai_content = re.sub(r'(?m)^\\s*(?:FOCUS TASK|ACTIVE STRATEGY & PLAN|PLAN|THOUGHT):\\s*', '', final_ai_content)"""
U_LABEL = """    ui_content = re.sub(r'(?m)^\\s*(?:FOCUS TASK|ACTIVE STRATEGY & PLAN|PLAN|THOUGHT):\\s*', '', ui_content)"""
F_EMOJI = """        final_ai_content = re.sub(r'(?m)^\\s*(?:🔄|🟢|⏳|✅|❌|🛑|➖)\\s*\\[.*?\\].*?\\n?', '', final_ai_content)"""
F_TASKID = """        final_ai_content = re.sub(r'(?m)^\\s*(?:\\[)?task_\\d+(?:\\])?\\s*\\n?', '', final_ai_content)"""
F_STATUS = """        final_ai_content = re.sub(r'(?m)^.*\\((?:IN_PROGRESS|READY|PENDING|DONE|FAILED|BLOCKED)\\)\\s*$\\n?', '', final_ai_content)"""
U_BRACKET = """    ui_content = re.sub(r'(?m)^\\s*\\[(?:task_\\d+|IN_PROGRESS|READY|PENDING|DONE|FAILED|BLOCKED)\\b[^\\]]*\\].*?\\n?', '', ui_content)"""
U_TASKID = """    ui_content = re.sub(r'(?m)^\\s*(?:\\[)?task_\\d+(?:\\])?\\s*\\n?', '', ui_content)"""
U_EMOJI = """    ui_content = re.sub(r'(?m)^\\s*(?:🔄|🟢|⏳|✅|❌|🛑|➖)\\s*\\[.*?\\].*?\\n?', '', ui_content)"""
MUT = [
    ("CONTROL no-op", AG, F_LABEL, F_LABEL + "  # no-op"),
    ("CONTROL known-bad", AG, F_LABEL, F_LABEL + "\n        final_ai_content = ''"),
    ("F1 finalize-emoji-line", AG, F_LABEL, F_EMOJI + "\n" + F_LABEL),
    ("F2 finalize-task-id-line", AG, F_LABEL, F_TASKID + "\n" + F_LABEL),
    ("F3 finalize-status-lines", AG, F_LABEL, F_STATUS + "\n" + F_LABEL),
    ("F4 finalize-label-strip-gone", AG, F_LABEL, "        pass"),
    ("U1 turn-bracket-status-line", AG, U_LABEL, U_BRACKET + "\n" + U_LABEL),
    ("U2 turn-task-id-line", AG, U_LABEL, U_TASKID + "\n" + U_LABEL),
    ("U3 turn-emoji-line", AG, U_LABEL, U_EMOJI + "\n" + U_LABEL),
    ("U4 turn-compiled-tree-regex", AG, U_LABEL, U_LABEL + "\n    ui_content = _TREE_RE.sub('', ui_content)"),
    ("U5 turn-helper-not-called", AG, "                ui_content = _scrub_turn_reply(ui_content)\n",
     "                pass\n"),
    ("U6 turn-label-strip-gone", AG, U_LABEL, "    pass"),
]
#: whether the turn calls its cleanup at all is read from the source (coverage maps executed lines only)
FORCE: dict = {"U5 turn-helper-not-called": ["tests/test_4vc_task_lines_reach_the_owner.py::test_the_two_sites_call_only_listed_cleaners"]}
ONLY: dict = {}
PINS = ["tests/test_4vc_task_lines_reach_the_owner.py", "tests/test_4ec_finalize_pins.py",
        "tests/test_finalize_stream_r1_fixes.py", "tests/test_finalize_stream_r2_fixes.py",
        "tests/test_finalize_stream_r3_fixes.py", "tests/test_finalize_stream_pins.py",
        "tests/test_finalize_extraction.py"]

def main():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    args = sys.argv[1:] or ["all"]
    workers = 6
    if args[0].startswith("-j"):
        workers = int(args[0][2:]); args = args[1:] or ["all"]
    if args[0] == "rerun":
        want = set(args[1:]); batch = [m for m in MUT if m[0].split(" ")[0] in want or m[0].startswith("CONTROL")]
    else:
        batch = MUT
    srcs = sorted({m[1] for m in MUT})
    originals = {f: open(os.path.join(ROOT, f)).read() for f in srcs}
    bad = [lab for lab, f, old, new in MUT if originals[f].count(old) != 1]
    if bad:
        raise SystemExit(f"anchors not unique/absent: {bad}")
    if len({m[0] for m in MUT}) != len(MUT) or any(lab not in {m[0] for m in MUT} for lab in list(FORCE) + list(ONLY)):
        raise SystemExit("duplicate mutant label, or a FORCE entry names no mutant")
    missing = [p for p in PINS if not os.path.exists(os.path.join(ROOT, p))]
    if missing:
        raise SystemExit(f"test files not found: {missing}")
    if args[0] == "check":
        print(f"anchors ok: {len(MUT)} mutants in {len(srcs)} files; {len(PINS)} test files"); return
    import battery_cov as BC
    cov = BC.build_map(srcs, PINS, log=lambda m: print(m, flush=True))
    sel = {lab: BC.tests_for(cov, f, old, originals[f]) for lab, f, old, new in batch}
    union = sorted({i for v in sel.values() if v for i in v})

    def select(label, f, old):
        if label.startswith("CONTROL no-op"):
            return union, f"cov {len(union)} tests (all covering)"
        if label in ONLY:
            return list(ONLY[label]), f"only {len(ONLY[label])} file(s)"
        ids = sel.get(label)
        if label in FORCE:
            ids = sorted(set(ids or []) | set(FORCE[label]))
        if ids:
            return ids, f"cov {len(ids)} tests"
        return list(PINS), f"fallback {len(PINS)} files"
    print(f"{len(PINS)} test files; {len(batch)} mutants; {len(union)} covering tests; {workers} trees", flush=True)
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
