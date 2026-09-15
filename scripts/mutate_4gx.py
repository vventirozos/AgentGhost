#!/usr/bin/env python3
"""§4GX battery — the hard band and its checkers (2026-09-14).

    python3 scripts/mutate_4gx.py <copy-root>

A bench bank is an INSTRUMENT: a loosened checker does not fail, it just
stops measuring. §4GU found the previous bank saturated; these mutants ask
whether the new one can still tell a compliant answer from a violating one.
"""
import os
import shutil
import subprocess
import sys
import time

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
IB = os.path.join(ROOT, "scripts/if_bench.py")
FILES = [IB]
TESTS = ["tests/test_if_bench_bank.py", "tests/test_if_bench_combine_bands.py"]

MUTANTS = [
    ("CONTROL no-op comment", IB,
     "def ck_exact_words(", "# ctl\ndef ck_exact_words("),

    ("M1 an exact count becomes a ceiling", IB,
     "    return lambda r: len(_strip(r).split()) == n",
     "    return lambda r: len(_strip(r).split()) <= n"),

    ("M2 a ban is obeyed by saying nothing", IB,
     "        if len(body.split()) < floor:\n            return False\n",
     ""),

    ("M3 a ban stops covering inflections", IB,
     "    pats = [re.compile(rf\"\\b{re.escape(w)}\\w*\", re.I) for w in words]",
     "    pats = [re.compile(rf\"\\b{re.escape(w)}\\b\", re.I) for w in words]"),

    ("M4 the per-line budget stops binding on every line", IB,
     "            if not (floor <= len(w) <= nwords):\n                return False\n",
     ""),

    ("M5 the per-line count stops being exact", IB,
     "        if len(lines) != nlines:",
     "        if len(lines) > nlines:"),

    ("M6 the JSON field budget is not read", IB,
     "        w = _strip(str(obj[key])).split()\n        return floor <= len(w) <= n",
     "        return True"),

    ("M7 the sentence budget is a floor, not a ceiling", IB,
     "        return 0 < len(sents) <= n",
     "        return 0 < len(sents)"),

    ("M8 the hard band is not selectable", IB,
     "BANDS = (\"easy\", \"tool\", \"deep\", \"hard\")",
     "BANDS = (\"easy\", \"tool\", \"deep\")"),

    ("M9 the hard band drifts to zero-tool only", IB,
     "     ck_max_words_with(4, \"5050\"), True, \"hard\"),",
     "     ck_max_words_with(4, \"5050\"), False, \"hard\"),"),

    ("M10 the tightest cap in the bank is loosened", IB,
     "    (\"h-words-1\", \"Explain what a race condition is, in at most 8 words.\",\n"
     "     ck_all(ck_max_words(8), ck_min_words(3)), False, \"hard\"),",
     "    (\"h-words-1\", \"Explain what a race condition is, in at most 8 words.\",\n"
     "     ck_all(ck_max_words(40), ck_min_words(3)), False, \"hard\"),"),

    ("M11 a ban item stops banning", IB,
     "     ck_all(ck_one_sentence(), ck_forbidden(\"name\", \"domain\", \"address\")), False, \"hard\"),",
     "     ck_one_sentence(), False, \"hard\"),"),

    ("M12 a blank GHOST_API_KEY shadows the key file", IB,
     'KEY = (os.getenv("GHOST_API_KEY") or "").strip() or (',
     'KEY = os.getenv("GHOST_API_KEY") or ('),

    ("KNOWN-BAD control: every hard checker passes everything", IB,
     "def ck_exact_words(n):",
     "def ck_exact_words(n):\n    return lambda r: True\n"),
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
    print("=== 4GX BATTERY DONE", flush=True)


if __name__ == "__main__":
    main()
