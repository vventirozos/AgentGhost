#!/usr/bin/env python3
"""§4GW battery — the written-source audit pack (2026-09-14).

    python3 scripts/mutate_4gw.py <copy-root>
"""
import os
import shutil
import subprocess
import sys
import time

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
AG = os.path.join(ROOT, "src/ghost_agent/core/agent.py")
VF = os.path.join(ROOT, "src/ghost_agent/core/verifier.py")
FILES = [AG, VF]
TESTS = [
    "tests/test_4gw_written_source_audit.py",
    "tests/test_verifier_evidence_window.py",
    "tests/test_grounded_file_verify.py",
]

MUTANTS = [
    ("CONTROL no-op comment", AG,
     "def _written_sources_for_audit(", "# ctl\ndef _written_sources_for_audit("),

    ("M1 the pre-fix shape: the pack is never built", AG,
     "                if _wrote:\n"
     "                    code_text = (_wrote + \"\\n\\n# --- command this turn ran \"\n"
     "                                 \"---\\n\" + code_text)\n",
     ""),

    ("M2 the block REPLACES the command instead of augmenting", AG,
     "                    code_text = (_wrote + \"\\n\\n# --- command this turn ran \"\n"
     "                                 \"---\\n\" + code_text)",
     "                    code_text = _wrote"),

    ("M3 a written path may escape the sandbox root", AG,
     "            if not str(target).startswith(str(root) + os.sep):\n"
     "                continue\n",
     ""),

    ("M4 a file's own text may forge a header", AG,
     "        shown = _AUDIT_MARKER_RE.sub(\n"
     "            \"# -- \", _head_and_tail(body, take, missing=max(0, size - take)))",
     "        shown = _head_and_tail(body, take, missing=max(0, size - take))"),

    ("M5 truncation is silent", AG,
     "        if take < size:\n            head += (f\"; only {take} of them",
     "        if False:\n            head += (f\"; only {take} of them"),

    ("M6 the block's own budget is ignored", AG,
     "    room = (budget - len(files) * _AUDIT_HEADER_RESERVE) // len(files)",
     "    room = 10 ** 6"),

    ("M7 an empty file counts as delivery", AG,
     "        if not body.strip():\n            continue\n",
     ""),

    ("M8 a binary deliverable is packed as source", AG,
     "        if not rel or not _AUDIT_SOURCE_EXT_RE.search(rel):\n"
     "            continue\n",
     "        if not rel:\n            continue\n"),

    ("M9 a broken binding takes the verdict down with it", AG,
     "                    _wrote = \"\"\n                if _wrote:",
     "                    raise\n                if _wrote:"),

    ("M10 every file is cut to noise instead of dropping one", AG,
     "    while files and ((budget - len(files) * _AUDIT_HEADER_RESERVE)\n"
     "                     // len(files)) < _AUDIT_SOURCE_MIN_SHARE:\n"
     "        files.pop()\n",
     ""),

    ("M11 a short file's remainder is thrown away", AG,
     "    bonus = spare // over if over else 0",
     "    bonus = 0"),

    ("M12 the budget stops depending on the command", AG,
     "    return max(0, min(_AUDIT_SOURCE_BUDGET,\n"
     "                      _AUDIT_SLOT_CAP - max(len(code_text or \"\"),\n"
     "                                            _AUDIT_COMMAND_FLOOR)))",
     "    return _AUDIT_SOURCE_BUDGET"),

    ("M13 the gate stops sizing the budget at all", AG,
     "                        self._scoped_sandbox_for(project_id),\n"
     "                        budget=_audit_source_budget(code_text))",
     "                        self._scoped_sandbox_for(project_id))"),

    ("M14 the prompt cannot read the evidence", VF,
     "  EXCEPTION \u2014 the source was delivered as a FILE, not as a message:",
     "  EXCEPTION \u2014 nothing to see here:"),

    ("M15 the exception no longer covers the short-answer case", VF,
     " This binds hardest when the user asked for a short answer or asked "
     "not to be shown the code: that is an explicit constraint under check "
     "1, and re-pasting the file would violate it.", ""),

    ("M16 the elision notice is decided AFTER defanging", AG,
     "        if take < size:", "        if len(shown) < size:"),

    ("M17 the exception fires without its evidence", VF,
     " \u26a0 THIS EXCEPTION NEEDS THAT BLOCK. If the CODE section contains "
     "no `# --- file this turn wrote:` header, it does not apply at all "
     "\u2014 you have no evidence the source was written anywhere, a RESPONSE "
     "that merely says it created a file is not a file, and the fence rule "
     "above stands unchanged.", ""),

    ("M18 a head-only cut — the end of the file is thrown away", AG,
     "    return body[:head_n] + marker + body[-tail_n:]",
     "    return body[:take]"),

    ("M19 the seam between the two ends is unmarked", AG,
     "    marker = _AUDIT_ELISION.format(n=missing)",
     "    marker = \"\""),

    ("M20 an elided excerpt can ground a MISSING finding again", VF,
     "AN ELIDED BLOCK IS EVIDENCE OF WHAT IS THERE, NEVER OF WHAT IS NOT.",
     "An elided block is a block."),

    ("KNOWN-BAD control: the pack is always empty", AG,
     "    if not tools_run or not host_dir:\n        return \"\"",
     "    return \"\"\n    if not tools_run or not host_dir:\n        return \"\""),
]


def main():
    saved = {f: f + ".pristine" for f in FILES}
    for f, p in saved.items():
        shutil.copy2(f, p)
    env = dict(os.environ, GHOST_API_KEY="x",
               PYTHONPATH=os.path.join(ROOT, "src"))
    env.pop("FORCE_COLOR", None)
    try:
        for n, (label, target, old, new) in enumerate(MUTANTS):
            for f, p in saved.items():
                shutil.copy2(p, f)
            src = open(saved[target]).read()
            if src.count(old) != 1:
                print(f"[{n:2d}] ANCHOR-MISS ({src.count(old)}x) {label}", flush=True)
                continue
            open(target, "w").write(src.replace(old, new, 1))
            t0 = time.time()
            p = subprocess.run(
                [sys.executable, "-m", "pytest", *TESTS, "-x", "-q",
                 "-p", "no:randomly", "--timeout=300"],
                cwd=ROOT, env=env, capture_output=True, text=True)
            verdict = "SURVIVED" if p.returncode == 0 else "KILLED  "
            tail = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
            print(f"[{n:2d}] {verdict} {label}  ({time.time()-t0:.0f}s) {tail[:70]}",
                  flush=True)
    finally:
        for f, p in saved.items():
            shutil.copy2(p, f)
            os.unlink(p)
    print("=== 4GW BATTERY DONE", flush=True)


if __name__ == "__main__":
    main()
