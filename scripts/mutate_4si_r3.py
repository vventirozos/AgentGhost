#!/usr/bin/env python3
"""§4SH/§4SI round-3 battery — fresh-eye review fixes (2026-10-30): parts close, empty finish through the loop, evaluate-as-interact readers, redactor, stray-JS regions.
In place, backup + restore + byte compare; one mutant per run; py_compile first; the related test set;
no-op control must SURVIVE, a known-bad control must DIE.    python3 scripts/mutate_4si_r3.py [all|rerun M1 M2 …]
"""
import os, shutil, subprocess, sys, signal, filecmp, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/Users/vasilis/Data/AI/.agent.venv/bin/python"
FS = "src/ghost_agent/tools/file_system.py"; PJ = "src/ghost_agent/tools/projects.py"; MM = "src/ghost_agent/tools/memory.py"
PV = "src/ghost_agent/utils/provenance.py"; ST = "src/ghost_agent/core/strikes.py"; WS = "src/ghost_agent/tools/workspace.py"
WT = "src/ghost_agent/tools/workspace_track.py"; WM = "src/ghost_agent/workspace/model.py"
SV = "src/ghost_agent/tools/sandbox_services.py"; SS = "src/ghost_agent/sandbox/services.py"

FSY = "src/ghost_agent/tools/file_system.py"; AG = "src/ghost_agent/core/agent.py"; TS = "src/ghost_agent/core/task_state.py"
RD = "src/ghost_agent/distill/redact.py"; WC = "src/ghost_agent/core/write_continuation.py"; RS = "src/ghost_agent/core/reply_smoothing.py"
def _tests():
    pat = r"file_system\|redact\|task_state\|write_continuation\|continue_cut_write\|document_ends_here\|html_unclosed_script\|_dispatch_and_process_tool_batch\|settle_open_parts\|reply_smoothing\|_parse_assistant_tool_calls\|finish_parts_file"
    out = subprocess.run(["grep", "-l", pat] + sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py"))), capture_output=True, text=True).stdout.split()
    return sorted({os.path.relpath(t, ROOT) for t in out} | set(PINS))
MUT = [
    ("CONTROL no-op", RD, "def _names_itself(name_part: str, v: str) -> bool:", "def _names_itself(name_part: str, v: str) -> bool:  # no-op"),
    ("CONTROL known-bad", FSY, '            return name == "script"                  # a raw-text element runs to the end', '            return False'),
    ("P1 rawtext-only-script", FSY, '        if name not in _RAWTEXT_TAGS or (xml and', '        if name != "script" or (xml and'),
    ("P2 quote-anywhere", FSY, '_TAG_STOP_RE = re.compile(r">|=[ \\t\\r\\n\\f]*([\\"\'])")', '_TAG_STOP_RE = re.compile(r">|([\\"\'])")'),
    ("P3 abrupt-comment", FSY, '            if t.startswith("<!-->", i):\n                pos = i + 5\n                continue\n', ''),
    ("P4 abrupt-comment-dash", FSY, '            if t.startswith("<!--->", i):\n                pos = i + 6\n                continue\n', ''),
    ("P5 eof-in-tag", FSY, '            return False                             # the file ends inside a tag: no script opened', '            return True'),
    ("P6 xml-selfclose", FSY, ' or (xml and t[m.end():end - 1].rstrip().endswith("/")):', ':'),
    ("P7 closer-boundary", FSY, 'r"</" + re.escape(name) + r"[\\s/>]"', 'r"</" + re.escape(name)'),
    ("P8 cdata", FSY, '        if t.startswith("<![CDATA[", i):', '        if False:'),
    ("P9 cont-in-loop-done", WC, '                    if not done:\n                        # §4SH r3', '                    if False:\n                        # §4SH r3'),
    ("P10 redact-semicolon", RD, '    if ";" in v or "=" in v:', '    if False:'),
    ("P11 names-itself", RD, "    if _is_code_value(m.group(2)) or _names_itself(m.group(1), m.group(2)):", "    if _is_code_value(m.group(2)):"),
    ("P12 names-itself-eq", RD, '    if v.startswith("="):\n        return True', '    if False:\n        return True'),
    ("P13 fallback-flags", AG, '                                            args_dict[_flag_k] = True', '                                            pass'),
    ("P14 edit-deferred", FSY, '+ (_PARTS_PENDING_NOTE if (_open and _syn) else _syn))', '+ _syn)'),
    ("P15 edit-feedback-open", FSY, '_syn = await _syntax_feedback(path, filename, in_progress=_open)', '_syn = await _syntax_feedback(path, filename)'),
    ("P16 svg-phrase", FSY, '        if text is None or document_ends_here(filename, text):\n            return "was checked: it ends', '        if True:\n            return "was checked: it ends'),
    ("P17 refusal-info", FSY, """            return (f"NOTE: '{target_path}' has no part pending""", """            return ToolOutcome.rejected(f"NOTE: '{target_path}' has no part pending"""),
    ("P18 tool-fin-hint", FSY, '"do not re-send the file." if _flag(kwargs.get("append")) else "")', '"do not re-send the file." if False else "")'),
    ("P19 block-fin-hint", AG, '"string) with append=true; do not re-send the file"\n                                             if _app_flag is True', '"string) with append=true; do not re-send the file"\n                                             if False'),
    ("P20 footer-phrase", RS, '                            "was left open for more")', '                            )'),
]
PINS = ["tests/test_4sh_parts_close.py", "tests/test_4ec_parser_dialects.py", "tests/test_4oi_open_items.py", "tests/test_4rb_reasoning_effort.py"]
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
