#!/usr/bin/env python3
"""§4GT battery — the shared UI scrub pattern (2026-09-14).

    python3 scripts/mutate_4gt.py <copy-root>
"""
import os
import shutil
import subprocess
import sys
import time

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
AG = os.path.join(ROOT, "src/ghost_agent/core/agent.py")
FILES = [AG]
TESTS = [
    "tests/test_parser_flood_and_leak_fix.py",
    "tests/test_reply_toolcall_scrub.py",
    "tests/test_streaming_scrub_behavioral.py",
]

MUTANTS = [
    ("CONTROL no-op comment", AG,
     "_UI_SCRUB_RE = re.compile(", "# ctl\n_UI_SCRUB_RE = re.compile("),

    ("M1 the backtick guard is gone again", AG,
     "    r'(?<!`)<(tool_call|tool|function)\\b[^>]*>.*?(?:</\\1\\b[^>]*>|\\Z)',\n"
     "    flags=re.DOTALL | re.IGNORECASE,\n)",
     "    r'<(tool_call|tool|function)\\b[^>]*>.*?(?:</\\1\\b[^>]*>|\\Z)',\n"
     "    flags=re.DOTALL | re.IGNORECASE,\n)"),

    ("M2 back to `$` — the trailing newline escapes", AG,
     "r'(?<!`)<(tool_call|tool|function)\\b[^>]*>.*?(?:</\\1\\b[^>]*>|\\Z)',",
     "r'(?<!`)<(tool_call|tool|function)\\b[^>]*>.*?(?:</\\1\\b[^>]*>|$)',"),

    ("M3 the backreference is dropped", AG,
     "r'(?<!`)<(tool_call|tool|function)\\b[^>]*>.*?(?:</\\1\\b[^>]*>|\\Z)',",
     "r'(?<!`)<(tool_call|tool|function)\\b[^>]*>.*?(?:</[a-z_]+\\b[^>]*>|\\Z)',"),

    ("M4 `function` leaves the alternation", AG,
     "r'(?<!`)<(tool_call|tool|function)\\b[^>]*>.*?(?:</\\1\\b[^>]*>|\\Z)',",
     "r'(?<!`)<(tool_call|tool)\\b[^>]*>.*?(?:</\\1\\b[^>]*>|\\Z)',"),

    ("M5 the mid-flow site rebuilds its own literal", AG,
     "                _scrubbed = _UI_SCRUB_RE.sub('', ui_content)",
     "                _scrubbed = re.sub(\n"
     "                    r'<(tool_call|tool|function)\\b[^>]*>.*?(?:</\\1\\b[^>]*>|\\Z)',\n"
     "                    '', ui_content, flags=re.DOTALL | re.IGNORECASE)"),

    ("KNOWN-BAD control: the scrub matches everything", AG,
     "_UI_SCRUB_RE = re.compile(",
     "_UI_SCRUB_RE = re.compile(r'.', re.DOTALL) if True else re.compile("),
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
    print("=== 4GT BATTERY DONE", flush=True)


if __name__ == "__main__":
    main()
