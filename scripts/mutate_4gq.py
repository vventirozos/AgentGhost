#!/usr/bin/env python3
"""§4GQ round-8 mutation battery (2026-09-14).

Each mutant re-introduces one defect round 8 found INSIDE round 7's fixes.
Whole-file, run against the whole test set — the harness never names its own
killer. A no-op control per file must SURVIVE; a known-bad must be KILLED.

    python3 scripts/mutate_4gq.py <copy-root>
"""
import os
import shutil
import subprocess
import sys
import time

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
SVC = os.path.join(ROOT, "src/ghost_agent/sandbox/services.py")
CL = os.path.join(ROOT, "src/ghost_agent/core/coding_loop.py")
DREAM = os.path.join(ROOT, "src/ghost_agent/core/dream.py")
FILES = [SVC, CL, DREAM]
TESTS = [
    "tests/test_4gq_round8.py",
    "tests/test_4gk_round4.py",
    "tests/test_4fg_coding_loop.py",
    "tests/test_service_port_leases.py",
    "tests/test_dream_replay_validate.py",
]

MUTANTS = [
    ("CONTROL services no-op", SVC,
     "    def _kill_service(self, entry, others=()) -> bool:",
     "    # ctl\n    def _kill_service(self, entry, others=()) -> bool:"),
    ("CONTROL coding_loop no-op", CL,
     "def snapshot_workspace(root: Optional[Path]) -> Dict[str, str]:",
     "# ctl\ndef snapshot_workspace(root: Optional[Path]) -> Dict[str, str]:"),
    ("CONTROL dream no-op", DREAM,
     "def _snapshot_mocks(sandbox_path: Path) -> dict:",
     "# ctl\ndef _snapshot_mocks(sandbox_path: Path) -> dict:"),

    ("M1 the survival verdict depends on the caller again", SVC,
     "        self._last_kill_survived = False\n        pid = entry.get(\"pid\")",
     "        pid = entry.get(\"pid\")"),

    ("M2 a file the workspace snapshot cannot read is silent again", CL,
     "                    out[_SNAPSHOT_INCOMPLETE] = (\n"
     "                        f\"{type(_fexc).__name__} reading {fn!r}: {_fexc}\")\n"
     "                    continue",
     "                    continue"),

    ("M3 a file the self-play snapshot cannot read is silent again", DREAM,
     "                    snap[_SNAPSHOT_INCOMPLETE] = (\n"
     "                        f\"{type(_fexc).__name__} reading {name!r}: {_fexc}\")",
     "                    pass"),

    ("M4 the reading is never retried before the work is", CL,
     "        if snapshot_incomplete(after):\n            after = snapshot_workspace(ws)",
     "        if False:\n            after = snapshot_workspace(ws)"),

    ("M5 a partial reading ships as a complete one", CL,
     "                    + (\" · reading INCOMPLETE\"\n"
     "                       if snapshot_incomplete(before, after) else \"\"))",
     "                    )"),

    ("M6 every reading is announced as partial", CL,
     "            if snapshot_incomplete(before, after):\n                detail_parts.append(",
     "            if True:\n                detail_parts.append("),

    ("KNOWN-BAD control: the workspace snapshot sees nothing", CL,
     "    out: Dict[str, str] = {}\n    if root is None or not root.exists():",
     "    out: Dict[str, str] = {}\n    return out\n    if root is None or not root.exists():"),
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
    print("=== 4GQ BATTERY DONE", flush=True)


if __name__ == "__main__":
    main()
