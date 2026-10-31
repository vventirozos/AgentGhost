#!/usr/bin/env python3
"""§4SH/§4SI round-2 battery — fresh-eye review fixes (2026-10-30): parts close, empty finish through the loop, evaluate-as-interact readers, redactor, stray-JS regions.
In place, backup + restore + byte compare; one mutant per run; py_compile first; the related test set;
no-op control must SURVIVE, a known-bad control must DIE.    python3 scripts/mutate_4si_r2.py [all|rerun M1 M2 …]
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
WC = "src/ghost_agent/core/write_continuation.py"; CL = "src/ghost_agent/core/coding_loop.py"
def _tests():
    pat = r"file_system\|redact\|task_state\|untested_pages\|browser_support\|write_continuation\|continue_cut_write\|document_ends_here\|_dispatch_and_process_tool_batch\|html_stray\|settle_open_parts\|_run_streamed_report_writes\|coding_loop\|isolation\|_rebuild_available_tools"
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py"))), capture_output=True, text=True).stdout.split()
    return sorted({os.path.relpath(t, ROOT) for t in out} | set(PINS))
MUT = [
    ("CONTROL no-op", RD, "def _is_code_value(v: str) -> bool:", "def _is_code_value(v: str) -> bool:  # no-op"),
    ("CONTROL known-bad", FSY, "        if not close:\n            return True", "        if not close:\n            return False"),
    ("N1 unclosed-comment-skip", FSY, '        if m.group(0).startswith("<!--"):', '        if False:'),
    ("N2 cut-finding-script", FSY, '    if html_unclosed_script(t):\n        return "its last <script>', '    if False:\n        return "its last <script>'),
    ("N3 settle-uses-finding", FSY, "            if not finding:\n                finding = html_cut_finding(text)",
     '            if not finding and not re.search(r"</html\\s*>", text, re.IGNORECASE):\n                finding = "it ends without </html>, so it looks cut short"'),
    ("N4 finish-note-finding", FSY, '    _cut = html_cut_finding(text) if is_html else ""',
     '    _cut = ("it does not end with </html>" if is_html and not document_ends_here(filename, text) else "")'),
    ("N5 ends-unclosed", FSY, 'if not re.search(r"</html\\s*>\\s*\\Z", t, re.IGNORECASE) or html_unclosed_script(t):',
     'if not re.search(r"</html\\s*>\\s*\\Z", t, re.IGNORECASE):'),
    ("N6 alias-action", BS, 'a.get("operation") or a.get("op") or a.get("action") or ""', 'a.get("operation") or a.get("op") or ""'),
    ("N7 alias-steps", BS, 'acts = a.get("actions") or a.get("steps") or a.get("sequence")', 'acts = a.get("actions")'),
    ("N8 alias-link", TS, 'url = str(a.get("url") or a.get("link") or a.get("href") or next(', 'url = str(a.get("url") or next('),
    ("N9 drain-finish", AG, 'not str(args.get("content") or "").strip() and not _fin):', 'not str(args.get("content") or "").strip()):'),
    ("N10 guard-whitespace", AG, '_finishing = isinstance(t_args.get("content"), str) and (', '_finishing = t_args.get("content") == "" and ('),
    ("N11 cont-done-by-end", WC, "                done = bool(document_ends_here(path, await asyncio.to_thread(", "                done = False and bool(document_ends_here(path, await asyncio.to_thread("),
    ("N12 cont-page-notes", WC, '"the file is complete." + (cap_note or "") + page)', '"the file is complete." + (cap_note or ""))'),
    ("N13 redact-roots", RD, "return bool(m and m.group(1) in _CODE_ROOTS)", "return bool(m)"),
    ("N14 redact-prose-exempt", RD, "    if _is_code_value(m.group(2)):                  # §4SH r2", "    if False:                  # §4SH r2"),
    ("N15 spot rebuild-allowlist", AG, '_allow = getattr(self.context, "_subagent_allowed_tools", None)', '_allow = None'),
    ("N16 spot leaf-owner-isolated", CL, "owner_memory_isolated = True", "owner_memory_isolated = False"),
]
PINS = ["tests/test_4sh_parts_close.py", "tests/test_4si_containment_pins.py"]
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
