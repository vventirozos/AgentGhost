#!/usr/bin/env python3
"""§4GV battery — the detached-job reaper (2026-09-14).

    python3 scripts/mutate_4gv.py <copy-root>

Never run against the working tree: the mutants make the suite's own cleanup
lie, and one of them is "the reaper kills on the pid alone".
"""
import os
import shutil
import subprocess
import sys
import time

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
CF = os.path.join(ROOT, "tests/conftest.py")
SJ = os.path.join(ROOT, "tests/test_sandbox_job_promotion.py")
T4GV = os.path.join(ROOT, "tests/test_4gv_detached_job_reaper.py")
FILES = [CF, SJ, T4GV]
TESTS = [
    "tests/test_4gv_detached_job_reaper.py",
    "tests/test_sandbox_job_promotion.py",
]

MUTANTS = [
    ("CONTROL no-op comment", CF,
     "def reap_detached_jobs(", "# ctl\ndef reap_detached_jobs("),

    ("M1 kill on the pid alone — no identity check", CF,
     "        if not _same_process(row.get(\"argv\"), _live_command(pid)):\n"
     "            continue                       # gone, or someone else's pid now\n",
     ""),

    ("M2 the recorded argv is compared with its quotes still on", CF,
     "        return \" \".join(str(t).replace(\"'\", \"\").replace('\"', \"\").split())",
     "        return \" \".join(str(t).split())"),

    # §4GZ made the registry per-session-process, so "a per-run registry"
    # is now the DESIGN; what has to hold instead is that a later run can
    # still FIND an earlier one's file.
    ("M3 last run's registries are unreachable — the glob matches nothing", CF,
     'JOB_REGISTRY_GLOB = "ghost-test-detached-jobs-*.jsonl"',
     'JOB_REGISTRY_GLOB = "no-such-registry-*.jsonl"'),

    ("M4 the shim records nothing", SJ,
     "    \"_reg = os.environ.get('GHOST_TEST_JOB_REGISTRY')\\n\"",
     "    \"_reg = None\\n\""),

    ("M5 the registry is never cleared — next run re-kills stale rows", CF,
     "    try:\n"
     "        registry.unlink()\n"
     "    except OSError:\n"
     "        pass\n"
     "    return killed",
     "    return killed"),

    ("M6 only the group leader is signalled", CF,
     "            if os.getpgid(pid) == pgid:    # still its own group leader\n"
     "                os.killpg(pgid, signal.SIGKILL)\n"
     "            else:\n"
     "                os.kill(pid, signal.SIGKILL)",
     "            os.kill(pid, signal.SIGKILL)"),

    ("M7 a corrupt registry takes the session down with it", CF,
     "    except (OSError, ValueError):\n        return 0",
     "    except OSError:\n        return 0"),

    ("M8 the shim is never pointed at the registry", CF,
     "    os.environ[\"GHOST_TEST_JOB_REGISTRY\"] = str(JOB_REGISTRY)\n"
     "    before = reap_abandoned_registries()",
     "    before = reap_abandoned_registries()"),

    ("M9 sweep at teardown only — an inherited stray lives forever", CF,
     "    before = reap_abandoned_registries() + reap_detached_jobs(JOB_REGISTRY)\n"
     "    if before:",
     "    before = 0\n"
     "    if before:"),

    ("M10 this file's own spawns are invisible to the session reaper", T4GV,
     "        with open(JOB_REGISTRY, \"a\") as fh:",
     "        with open(os.devnull, \"a\") as fh:"),

    ("M11 the helper registers its pid only AFTER its own assertion", T4GV,
     "    _session_sweep_row(spawned, [\"sh\", \"-c\", inner])\n",
     ""),

    ("KNOWN-BAD control: the reaper reaps nothing", CF,
     "    registry = Path(registry or JOB_REGISTRY)",
     "    return 0\n    registry = Path(registry or JOB_REGISTRY)"),
]


def main():
    saved = {f: f + ".pristine" for f in FILES}
    for f, p in saved.items():
        shutil.copy2(f, p)
    env = dict(os.environ, GHOST_API_KEY="x",
               PYTHONPATH=os.path.join(ROOT, "src"))
    env.pop("FORCE_COLOR", None)
    env.pop("GHOST_TEST_JOB_REGISTRY", None)
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
    print("=== 4GV BATTERY DONE", flush=True)


if __name__ == "__main__":
    main()
