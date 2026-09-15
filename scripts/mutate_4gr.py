#!/usr/bin/env python3
"""§4GR mutation battery — the absence-issue grammar (2026-09-14).

    python3 scripts/mutate_4gr.py <copy-root>
"""
import os
import shutil
import subprocess
import sys
import time

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
V = os.path.join(ROOT, "src/ghost_agent/core/verifier.py")
O = os.path.join(ROOT, "src/ghost_agent/core/objection.py")
FILES = [V, O]
TESTS = [
    "tests/test_4gr_absence_grammar.py",
    "tests/test_escalation_discipline.py",
    "tests/test_verify_self_consistency.py",
    "tests/test_verify_outcome_logging.py",
    "tests/test_objection_nonassertive.py",
    "tests/test_objection_atom_canon.py",
]

MUTANTS = [
    ("CONTROL verifier no-op", V,
     "def _truncation_min_severity() -> float:",
     "# ctl\ndef _truncation_min_severity() -> float:"),
    ("CONTROL objection no-op", O,
     "_ABSENCE_RE = re.compile(",
     "# ctl\n_ABSENCE_RE = re.compile("),

    ("M1 the adverb slot is gone (verifier)", V,
     '_NEG_ADV = r"(?:(?!only\\b)\\w+ly\\s+)?"',
     '_NEG_ADV = r""'),
    ("M2 the adverb slot is gone (objection)", O,
     '_NEG_ADV = r"(?:(?!only\\b)\\w+ly\\s+)?"',
     '_NEG_ADV = r""'),
    ("M3 'only' is admitted as an adverb (verifier)", V,
     '_NEG_ADV = r"(?:(?!only\\b)\\w+ly\\s+)?"',
     '_NEG_ADV = r"(?:\\w+ly\\s+)?"'),
    ("M4 'only' is admitted as an adverb (objection)", O,
     '_NEG_ADV = r"(?:(?!only\\b)\\w+ly\\s+)?"',
     '_NEG_ADV = r"(?:\\w+ly\\s+)?"'),
    ("M5 the plural subject is dropped again (verifier)", V,
     'r"|do(?:es)? not " + _NEG_ADV + r"appear in|doesn\'?t " + _NEG_ADV + r"appear in"',
     'r"|does not appear in|doesn\'t appear in"'),
    ("M6 the plural subject is dropped again (objection)", O,
     'r"|do(?:es)? not " + _NEG_ADV + r"(?:appear|include|contain|mention|state|"',
     'r"|does not " + _NEG_ADV + r"(?:appear|include|contain|mention|state|"'),
    ("M7 confirmed/verifiable leave the verb list", V,
     'r"listed in|stated in|supported by|corroborated by|confirmed by|"\n'
     '    r"verifiable in|reflected in|included in|"',
     'r"listed in|stated in|supported by|corroborated by|"\n'
     '    r"reflected in|included in|"'),
    ("M8 the adverb slot swallows a clause", V,
     '_NEG_ADV = r"(?:(?!only\\b)\\w+ly\\s+)?"',
     '_NEG_ADV = r"(?:[^.;,\\n]{0,40}?)?"'),

    ("KNOWN-BAD control: every issue is an absence complaint", V,
     "_ABSENCE_ISSUE_RE = re.compile(",
     "_ABSENCE_ISSUE_RE = re.compile(r\".\", re.I) if True else re.compile("),
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
    print("=== 4GR BATTERY DONE", flush=True)


if __name__ == "__main__":
    main()
