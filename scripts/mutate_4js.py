#!/usr/bin/env python3
"""§4JS battery — the cancellation-safe wait_for, the bounded watchdog stop,
the planner's DONE plan as a forced final (2026-09-22).

    python3 scripts/mutate_4js.py <copy-root>
"""
import os
import shutil
import subprocess
import sys
import time

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
AIO = os.path.join(ROOT, "src/ghost_agent/utils/aio.py")
LLM = os.path.join(ROOT, "src/ghost_agent/core/llm.py")
MAIN = os.path.join(ROOT, "src/ghost_agent/main.py")
AG = os.path.join(ROOT, "src/ghost_agent/core/agent.py")
FILES = [AIO, LLM, MAIN, AG]
TESTS = [
    "tests/test_4js_cancel_safe_wait_for.py",
    "tests/test_4ji_request_echo_narration.py",
    "tests/test_reply_language_guard_2026_09_22.py",
    "tests/test_planner_termination.py",
]

MUTANTS = [
    ("CONTROL no-op comment", AIO, "async def wait_for(", "# ctl\nasync def wait_for("),

    ("M1 the chunk reader goes back to the stdlib wait_for", LLM,
     "chunk = await _wait_for_cancel_safe(chunk_iter.__anext__(), timeout=_timeout)",
     "chunk = await asyncio.wait_for(chunk_iter.__anext__(), timeout=_timeout)"),

    ("M2 the helper regains the stdlib's done() shortcut", AIO,
     "    except asyncio.CancelledError:\n        fut.cancel()\n        raise\n",
     "    except asyncio.CancelledError:\n        if fut.done():\n            return fut.result()\n        fut.cancel()\n        raise\n"),

    ("M3 the inner is not cancelled on timeout", AIO,
     "    fut.cancel()\n    # Let the inner cancellation land.",
     "    # Let the inner cancellation land."),

    ("M4 a timeout returns None instead of raising", AIO,
     "    raise asyncio.TimeoutError()", "    return None"),

    ("M5 the shutdown stop is unbounded again", MAIN,
     "        _done, pending = await asyncio.wait({bio}, timeout=grace)",
     "        _done, pending = await asyncio.wait({bio}, timeout=None)"),

    ("M6 the shutdown stop forgets to cancel", MAIN,
     "    grace = _BIO_SHUTDOWN_GRACE_S if grace_s is None else float(grace_s)\n    bio.cancel()\n",
     "    grace = _BIO_SHUTDOWN_GRACE_S if grace_s is None else float(grace_s)\n"),

    ("M7 the straggler line loses the frame", MAIN,
     '            "see utils/aio.py)", grace, _task_where(bio))',
     '            "see utils/aio.py)", grace, "?")'),

    ("M8 the planner sets force_stop again (the pre-§4JS world)", AG,
     "                            if _plan_signals_done:\n"
     '                                pretty_log("Finalizing", "Agent signaled completion — final generation, tools off", icon=Icons.OK)\n'
     "                                force_final_response = True\n",
     "                            if _plan_signals_done:\n"
     '                                pretty_log("Finalizing", "Agent signaled completion — final generation, tools off", icon=Icons.OK)\n'
     "                                force_final_response = True\n"
     "                                force_stop = True\n"),

    ("M9 the lifespan awaits the watchdog bare again", MAIN,
     "        await _stop_biological_watchdog(context.biological_task)\n",
     "        bio = context.biological_task\n        if bio is not None:\n            bio.cancel()\n"
     "            try:\n                await bio\n            except asyncio.CancelledError:\n                pass\n"),

    # (an M10 "timeout=None path via stdlib wait_for" mutant was EQUIVALENT — 3.10's
    #  wait_for(aw, None) is `await aw` — and was deleted, per R2)

    ("KNOWN-BAD control: the helper never returns the result", AIO,
     "    if fut in done:\n        return fut.result()\n",
     "    if fut in done:\n        return None\n"),
]


def main():
    saved = {f: f + ".pristine" for f in FILES}
    for f, p in saved.items():
        shutil.copy2(f, p)
    env = dict(os.environ, GHOST_API_KEY="x", PYTHONPATH=os.path.join(ROOT, "src"))
    env.pop("FORCE_COLOR", None)
    killed = survived = 0
    try:
        for n, (label, target, old, new) in enumerate(MUTANTS):
            for f, p in saved.items():
                shutil.copy2(p, f)
            src = open(saved[target]).read()
            if src.count(old) != 1:
                print(f"[{n:2d}] ANCHOR-MISS ({src.count(old)}x) {label}", flush=True)
                continue
            open(target, "w").write(src.replace(old, new, 1))
            for d, _, _fs in os.walk(os.path.join(ROOT, "src")):
                if d.endswith("__pycache__"):
                    shutil.rmtree(d, ignore_errors=True)
            c = subprocess.run([sys.executable, "-m", "py_compile", target], capture_output=True, text=True)
            if c.returncode != 0:
                print(f"[{n:2d}] NO-COMPILE {label}: {c.stderr.strip()[-120:]}", flush=True)
                continue
            t0 = time.time()
            p = subprocess.run(
                [sys.executable, "-m", "pytest", *TESTS, "-x", "-q", "-p", "no:randomly", "--timeout=40"],
                cwd=ROOT, env=env, capture_output=True, text=True)
            verdict = "SURVIVED" if p.returncode == 0 else "KILLED  "
            if not label.startswith("CONTROL"):
                killed += p.returncode != 0
                survived += p.returncode == 0
            tail = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
            print(f"[{n:2d}] {verdict} {label}  ({time.time()-t0:.0f}s) {tail[:70]}", flush=True)
    finally:
        for f, p in saved.items():
            shutil.copy2(p, f)
            os.unlink(p)
    print(f"=== 4JS BATTERY DONE: {killed} killed / {survived} survived (known-bad control counted)", flush=True)


if __name__ == "__main__":
    main()
