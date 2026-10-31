#!/usr/bin/env python3
"""§4SH/§4SI round-1 battery — fresh-eye review fixes (2026-10-30): parts close, empty finish through the loop, evaluate-as-interact readers, redactor, stray-JS regions.
In place, backup + restore + byte compare; one mutant per run; py_compile first; the related test set;
no-op control must SURVIVE, a known-bad control must DIE.    python3 scripts/mutate_4si_r1.py [all|rerun M1 M2 …]
"""
import os, shutil, subprocess, sys, signal, filecmp, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/Users/vasilis/Data/AI/.agent.venv/bin/python"
FS = "src/ghost_agent/tools/file_system.py"; PJ = "src/ghost_agent/tools/projects.py"; MM = "src/ghost_agent/tools/memory.py"
PV = "src/ghost_agent/utils/provenance.py"; ST = "src/ghost_agent/core/strikes.py"; WS = "src/ghost_agent/tools/workspace.py"
WT = "src/ghost_agent/tools/workspace_track.py"; WM = "src/ghost_agent/workspace/model.py"
SV = "src/ghost_agent/tools/sandbox_services.py"; SS = "src/ghost_agent/sandbox/services.py"

FSY = "src/ghost_agent/tools/file_system.py"; AG = "src/ghost_agent/core/agent.py"; TS = "src/ghost_agent/core/task_state.py"
RD = "src/ghost_agent/distill/redact.py"; BS = "src/ghost_agent/tools/browser_support.py"
TC = "src/ghost_agent/core/turn_state_check.py"; OH = "src/ghost_agent/distill/outcome_heuristics.py"
def _tests():
    pat = r"file_system\|redact\|task_state\|untested_pages\|browser_support\|outcome_heuristics\|turn_state_check\|document_ends_here\|_dispatch_and_process_tool_batch\|html_stray\|settle_open_parts"
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py"))), capture_output=True, text=True).stdout.split()
    return sorted({os.path.relpath(t, ROOT) for t in out} | set(PINS))
MUT = [
    ("CONTROL no-op", RD, "def _secret_shaped(value: str) -> bool:", "def _secret_shaped(value: str) -> bool:  # no-op"),
    ("CONTROL known-bad", FSY, 'return "</html>" if opened <= len(', 'return "" if opened <= len('),
    ("M1 redact-shape", RD, "if _CODE_VALUE_RE.match(v) and not _secret_shaped(v):", "if _CODE_VALUE_RE.match(v):"),
    ("M2 redact-head-digits", RD, 'r"^(?:[A-Za-z_$][A-Za-z_$.]*[(\\[]"', 'r"^(?:[A-Za-z_$][\\w$.]*[(\\[]"'),
    ("M3 loop-finish-off", AG, '                    _finishing = t_args.get("content") == "" and (', '                    _finishing = False and ('),
    ("M4 loop-finish-no-append", AG,
     '                    _finishing = t_args.get("content") == "" and (\n                        _app_flag is True or str(_app_flag).strip().lower() in ("true", "1", "yes"))',
     '                    _finishing = t_args.get("content") == ""'),
    ("M5 loop-finish-str-flag", AG, '_app_flag is True or str(_app_flag).strip().lower() in ("true", "1", "yes"))', '_app_flag is True)'),
    ("M6 untested-finish-skip", TS, 'if body.lstrip().startswith("SUCCESS: Finished "):', 'if False:'),
    ("M7 untested-effective", TS, "op, acts = browser_effective_call(a)", 'acts = [x for x in (a.get("actions") or []) if isinstance(x, dict)]'),
    ("M8 tsc-effective", TC, "op = browser_effective_call(args)[0]", 'op = str(args.get("operation") or "").lower()'),
    ("M9 oh-effective", OH, '            if _eop == "interact":\n                op = "interact"', '            if False:\n                op = "interact"'),
    ("M10 agent-extract-effective", AG, "op, _acts = browser_effective_call(args)",
     'op = str(args.get("operation") or "").strip().lower(); _acts = [a for a in (args.get("actions") or []) if isinstance(a, dict)]'),
    ("M11 effective-off", BS, "    if op in _INTERACT_ONLY_ACTIONS:\n        ran = ", "    if False:\n        ran = "),
    ("M12 checked-phrase", FSY, 'if ext in _CHECKED_EXTS or (ext in _NODE_EXTS and _find_node()):\n        return "was checked whole"',
     'if True:\n        return "was checked whole"'),
    ("M13 settle-skeleton", FSY, 'if not finding and not re.search(r"</html\\s*>", text, re.IGNORECASE):',
     'if not finding and not document_ends_here(name, text):'),
    ("M14 edit-close-prev", FSY, 'if not document_ends_here(filename, prev_content) else "")', 'if True else "")'),
    ("M15 finish-whitespace", FSY, 'if (operation == "write" and target_path and content == "" and not final_content',
     'if (operation == "write" and target_path and not str(final_content or "").strip()'),
    ("M16 finish-needs-no-more", FSY, '            and _flag(kwargs.get("append"))):     # r1',
     '            and _flag(kwargs.get("append")) and not _flag(kwargs.get("more_parts"))):     # r1'),
    ("M17 region-custom-el", FSY, 'r"<(script|style|pre|code|textarea|template|xmp)(?![\\w-])[^<>]*>',
     'r"<(script|style|pre|code|textarea|template|xmp)\\b[^<>]*>'),
    ("M18 region-no-tags", FSY, '    r"|<[A-Za-z][\\w:.-]*(?:', '    r"|<NOTAG[A-Za-z][\\w:.-]*(?:'),
    ("M19 script-balance", FSY, 'return "</html>" if opened <= len(', 'return "</html>" if True or opened <= len('),
]
PINS = ["tests/test_4sh_parts_close.py"]
def run_tests(tests):
    env = dict(os.environ, PYTHONPATH="src", GHOST_API_KEY=open(os.path.expanduser("~/Data/AI/.ghost_api_key")).read().strip()); env.pop("FORCE_COLOR", None)
    try:
        r = subprocess.run([PY, "-m", "pytest", *tests, "-q", "--no-header", "-p", "no:cacheprovider", "-x", "-n", "8", "--dist", "loadfile"], capture_output=True, text=True, env=env, cwd=ROOT, timeout=1500)
    except subprocess.TimeoutExpired:
        return 1, "TIMEOUT (a mutant wedged the run — counted as killed)"
    return r.returncode, (r.stdout.strip().splitlines() or ["?"])[-1]
def _fast_tests(mutated_file, pins):
    """Tier 1: the round's pin files plus every test that imports the mutated
    module (by its module name or basename) — the tests that CAN kill a
    mutant in that file. A survivor is re-judged on the full selection."""
    mod = os.path.splitext(os.path.basename(mutated_file))[0]
    if mod == "agent":                       # "agent" names nearly every test: pins first, survivors go full
        return sorted(set(pins))
    pat = mod + r"\b"
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py"))), capture_output=True, text=True).stdout.split()
    return sorted({os.path.relpath(x, ROOT) for x in out} | set(pins))

PINS = ["tests/test_4pg_autopilot_review.py", "tests/test_4pf_open_items.py", "tests/test_4pe_files_projects_review.py"]

def main():
    args = sys.argv[1:] or ["all"]
    if args[0] == "rerun":
        want = set(args[1:]); batch = [m for m in MUT if m[0].split(" ")[0] in want or m[0].startswith("CONTROL")]
    else:
        batch = MUT
    full = _tests(); files = sorted({m[1] for m in batch}); backups = {f: f + ".4si.bak" for f in files}
    for f, b in backups.items(): shutil.copy2(os.path.join(ROOT, f), os.path.join(ROOT, b))
    def restore(*_):
        for f, b in backups.items(): shutil.copy2(os.path.join(ROOT, b), os.path.join(ROOT, f))
        subprocess.run(["find", os.path.join(ROOT, "src"), "-name", "__pycache__", "-exec", "rm", "-rf", "{}", "+"], capture_output=True)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP): signal.signal(sig, lambda *a: (restore(), sys.exit(130)))
    killed = survived = 0
    print(f"{len(full)} test files in the full selection; {len(batch)} mutants; two-tier (fast set, then the full selection for survivors)", flush=True)
    try:
        for label, f, old, new in batch:
            restore(); path = os.path.join(ROOT, f); src = open(path).read()
            if src.count(old) != 1: print(f"SKIP   {label}: anchor count {src.count(old)}", flush=True); continue
            open(path, "w").write(src.replace(old, new))
            if subprocess.run([PY, "-m", "py_compile", path], capture_output=True).returncode != 0: print(f"NOCOMP {label}", flush=True); continue
            fast = _fast_tests(f, PINS) if not label.startswith("CONTROL no-op") else full
            rc, tail = run_tests(fast); tier = f"fast {len(fast)} files"
            if rc == 0 and fast is not full:
                rc, tail = run_tests(full); tier += " → full"
            status = "SURVIVED" if rc == 0 else "KILLED"; killed += rc != 0; survived += rc == 0
            print(f"{status:8} {label} — [{tier}] {tail}", flush=True)
    finally:
        restore(); ok = all(filecmp.cmp(os.path.join(ROOT, f), os.path.join(ROOT, b), shallow=False) for f, b in backups.items())
        for b in backups.values(): os.remove(os.path.join(ROOT, b))
        print(f"RESTORED identical={ok} killed={killed} survived={survived}", flush=True)
if __name__ == "__main__": main()
