#!/usr/bin/env python3
"""§4SH/§4SI round-4 battery — fresh-eye review fixes (2026-10-30): parts close, empty finish through the loop, evaluate-as-interact readers, redactor, stray-JS regions.
In place, backup + restore + byte compare; one mutant per run; py_compile first; the related test set;
no-op control must SURVIVE, a known-bad control must DIE.    python3 scripts/mutate_4si_r4.py [all|rerun M1 M2 …]
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
    ("CONTROL known-bad", FSY, '            return name                              # a raw-text element runs to the end', '            return ""'),
    ("Q1 lowercased-index", FSY, 're.IGNORECASE | re.ASCII).search(t, pos)', 're.IGNORECASE | re.ASCII).search(t.lower(), pos)'),
    ("Q2 foreign-off", FSY, '            foreign += not closes_itself', '            foreign += 0'),
    ("Q3 foreign-selfclose", FSY, '            foreign += not closes_itself', '            foreign += 1'),
    ("Q4 foreign-end", FSY, '                foreign -= 1', '                foreign -= 0'),
    ("Q5 bang-comment-end", FSY, '_COMMENT_END_RE = re.compile(r"--!?>")', '_COMMENT_END_RE = re.compile(r"-->")'),
    ("Q6 abrupt-comment-dash", FSY, '            if t.startswith("<!--->", i):\n                pos = i + 6\n                continue\n', ''),
    ("Q7 eof-in-tag", FSY, '            return "tag"                             # the file ends inside a tag', '            return ""'),
    ("Q8 cdata-foreign-only", FSY, '        if foreign and t.startswith("<![CDATA[", i):', '        if t.startswith("<![CDATA[", i):'),
    ("Q9 closer-vt", FSY, 'r"[\\t\\n\\f\\r />]", re.IGNORECASE', 'r"[\\s/>]", re.IGNORECASE'),
    ("Q10 ends-state", FSY, 're.IGNORECASE) or html_tail_state(t, xml=ext == "xhtml"):', 're.IGNORECASE) or html_unclosed_script(t, xml=ext == "xhtml"):'),
    ("Q11 cut-other-states", FSY, '    if state:                                    # r4', '    if False:                                    # r4'),
    ("Q12 nested-svg", FSY, '                and len(re.findall(r"</svg\\s*>", t, re.IGNORECASE)) >= _open_svg else "")', '                and True else "")'),
    ("Q13 missing-file-finish", FSY, '        # r4: a finish on a missing file — the MANDATORY refusal below advised this very call\n        return ToolOutcome', '        # r4: a finish on a missing file — the MANDATORY refusal below advised this very call\n        if False: return ToolOutcome'),
    ("Q14 edit-open-note", FSY, '    elif _open:\n        _text += OPEN_PART_NOTE', '    elif False:\n        _text += OPEN_PART_NOTE'),
    ("Q15 edit-feedback-open", FSY, '_syn = await _syntax_feedback(path, filename, in_progress=_open)', '_syn = await _syntax_feedback(path, filename)'),
    ("Q16 untested-empty-finish", TS, '        if op == "write" and a.get("content") == "" and (', '        if False and ('),
    ("Q17 cont-precheck", WC, '        try:\n            done = bool(document_ends_here(path, await asyncio.to_thread(\n                hp.read_text, encoding="utf-8", errors="replace")))\n        except OSError:\n            pass\n        ended_before',
     '        try:\n            done = False and bool(document_ends_here(path, await asyncio.to_thread(\n                hp.read_text, encoding="utf-8", errors="replace")))\n        except OSError:\n            pass\n        ended_before'),
    ("Q18 cont-wording", WC, '"the file is complete." if not ended_before else', '"the file is complete." if True else'),
    ("Q19 redact-lead", RD, '    lead = "=" if v.startswith("=") else ""', '    lead = ""'),
    ("Q20 redact-comparison", RD, '    if not rest:                                 #', '    if False:                                 #'),
    ("Q21 redact-type-first", RD, '    if _TYPE_WORD_RE.match(v):                   # r4', '    if False:                   # r4'),
    ("Q22 redact-attr", RD, 'm = _CODE_VALUE_RE.match(v) or _CODE_ATTR_RE.match(v)', 'm = _CODE_VALUE_RE.match(v)'),
    ("Q23 redact-secretstr", RD, 'string|number|any|SecretStr)', 'string|number|any)'),
    ("Q24 parse-tail-cut", AG, '_body = block_content[:_ftail.start()] if _ftail else block_content', '_body = block_content'),
    ("Q25 parse-flag-zone", AG, ', _fzone, re.IGNORECASE):', ', block_content, re.IGNORECASE):'),
    ("Q26 parse-close-quote", AG, '                                        if _cend:\n', '                                        if False:\n'),
    ("Q27 parse-yes", AG, r'''r'(?:"|\')\s*:\s*(?:true|1|"(?:true|yes|1)")(?![\w"])', _fzone''', r'''r'(?:"|\')\s*:\s*(?:true|"true")', _fzone'''),
    ("Q28 footer-anchor", RS, 'r"\\n{0,4}^⚠️ `[^`\\n]+` [^\\n]*?written in parts and (?:"', 'r"\\n{0,4}⚠️ [^\\n]* written in parts and (?:"'),
]
PINS = ["tests/test_4sh_parts_close.py", "tests/test_4ec_parser_dialects.py", "tests/test_4oi_open_items.py",
        "tests/test_4rb_reasoning_effort.py", "tests/test_4ob_write_continuation.py"]
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
