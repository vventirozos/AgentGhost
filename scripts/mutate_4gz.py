#!/usr/bin/env python3
"""§4GZ battery — review round 9, on §4GV/§4GW's own fixes (2026-09-15).

    python3 scripts/mutate_4gz.py <copy-root>
"""
import os
import shutil
import subprocess
import sys
import time

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
AG = os.path.join(ROOT, "src/ghost_agent/core/agent.py")
CF = os.path.join(ROOT, "tests/conftest.py")
FILES = [AG, CF]
TESTS = [
    "tests/test_4gw_written_source_audit.py",
    "tests/test_4gv_detached_job_reaper.py",
    "tests/test_verifier_evidence_window.py",
]

MUTANTS = [
    ("CONTROL no-op comment", AG,
     "def _read_bounded(", "# ctl\ndef _read_bounded("),

    ("M1 the pre-round shape: the whole file is read", AG,
     "        if size <= 2 * limit:\n"
     "            raw = fh.read()\n"
     "        else:\n"
     "            head = fh.read(limit)\n"
     "            fh.seek(-limit, os.SEEK_END)\n"
     "            raw = head + fh.read(limit)",
     "        raw = fh.read()"),

    ("M2 a bounded read reports the buffer as the file", AG,
     "    return raw.decode(\"utf-8\", \"replace\"), size",
     "    text = raw.decode(\"utf-8\", \"replace\")\n    return text, len(text)"),

    ("M3 the elided count is measured against the buffer", AG,
     "            \"# -- \", _head_and_tail(body, take, missing=max(0, size - take)))",
     "            \"# -- \", _head_and_tail(body, take))"),

    # (The mutant `if take < size:` -> `if take < len(body):` is PROVABLY
    #  EQUIVALENT on this call site — a bounded read is always 2*budget while
    #  `take` is at most budget, so the two predicates never disagree — and a
    #  battery that reports an equivalent mutant as SURVIVED is noise. The
    #  contract is pinned directly instead, by
    #  `test_head_and_tail_reports_the_gap_it_is_TOLD_about`.)

    ("M5 only the head of a big file survives", AG,
     "            head = fh.read(limit)\n"
     "            fh.seek(-limit, os.SEEK_END)\n"
     "            raw = head + fh.read(limit)",
     "            raw = fh.read(2 * limit)"),

    ("M6 one shared registry again — the cross-worker kill switch", CF,
     "JOB_REGISTRY = (Path(tempfile.gettempdir())\n"
     "                / f\"ghost-test-detached-jobs-{os.getpid()}.jsonl\")",
     "JOB_REGISTRY = (Path(tempfile.gettempdir())\n"
     "                / \"ghost-test-detached-jobs.jsonl\")"),

    ("M7 a live sibling's registry is swept too", CF,
     "        if path == JOB_REGISTRY or _owner_is_alive(path):\n            continue",
     "        if path == JOB_REGISTRY:\n            continue"),

    ("M8 the owner question is answered by the clock", CF,
     "    pid = _owner_pid(path)\n    if not pid:\n        return False",
     "    return (time.time() - os.path.getmtime(path)) < 600\n"
     "    pid = _owner_pid(path)\n    if not pid:\n        return False"),

    # (The mutant "teardown sweeps abandoned registries too" SURVIVED, and
    #  it was right: a dead owner's rows are fair game whenever they are
    #  noticed, and the owner-alive guard is what keeps a live sibling safe —
    #  the timing never was. Adopted into the fixture instead of pinned
    #  against. The defect it LOOKED like is M7, which is killed.)
    ("M10 the start sweep no longer looks for inherited registries", CF,
     "    before = reap_abandoned_registries() + reap_detached_jobs(JOB_REGISTRY)",
     "    before = reap_detached_jobs(JOB_REGISTRY)"),

    ("KNOWN-BAD control: the audit pack reads nothing", AG,
     "    size = path.stat().st_size",
     "    return \"\", 0\n    size = path.stat().st_size"),
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
    print("=== 4GZ BATTERY DONE", flush=True)


if __name__ == "__main__":
    main()
