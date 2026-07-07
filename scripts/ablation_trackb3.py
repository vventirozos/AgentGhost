"""Track B3 — do the PURE-IDLE learning loops add value? (IMPROVEMENTS.md #4)

Track B proved the cross-session MEMORY substrate earns its keep. B3 isolates the
IDLE-ONLY loops — dream/self-play, idle reflection critique, skills-auto
graduation — which Track B could not reach (they only fire in long idle windows).

This is the harness the ledger describes. It exploits the 2026-07-07 flags:
  --bio-time-scale N   compress hours-long idle windows into minutes/seconds
  --bio-deterministic  fire the probabilistic idle phases every eligible tick

Protocol (per repeat):
  * TREATMENT boots with idle loops ACCELERATED (--bio-time-scale + deterministic)
    and memory ON. We seed K tasks, then idle-sleep through M accelerated epochs
    so the idle loops run, then run the probe suite.
  * CONTROL boots with --bio-time-scale 1 (so the idle loops never fire within the
    short run window) and memory ON — isolating the IDLE loops specifically, not
    memory (which Track B already validated). Same seed + probe.
  * Compare: (a) probe pass rate via McNemar on matched outcomes, AND
    (b) learning artifacts produced during idle — playbook lessons by `source`
    (dream / self_play / reflection), GraduatedSkillStore count, proposed macros.

The multi-hour LIVE run needs an operator session (a real llama-server + time
even under acceleration). This script is the runnable apparatus; it does not
fabricate results.

Run:
    PYTHONPATH=src python scripts/ablation_trackb3.py --repeats 3 \
        --base-port 8046 --time-scale 60 --idle-epochs 6 --report-dir <out>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ablation_eval as AE          # noqa: E402
from ablation_trackb2 import _post  # reuse the proven HTTP helper
from trackb_tasks import load_trackb_pairs          # SeedProbe(seed, probe, validator)


def _b3_report(records, artifacts, meta) -> str:
    """Self-contained B3 report (trackb2's _build_report expects different meta
    keys — reusing it crashed the first run at the formatting step)."""
    L = [f"# Track B3 — idle-loop adjudication", ""]
    L.append(f"harness meta: {meta}")
    # Probe outcomes per arm.
    from collections import defaultdict
    by_arm = defaultdict(lambda: [0, 0])
    for r in records:
        by_arm[r["arm"]][0] += 1
        by_arm[r["arm"]][1] += 1 if r.get("passed") else 0
    L.append("\n## Probe outcomes")
    for arm, (n, p) in sorted(by_arm.items()):
        L.append(f"- {arm}: {p}/{n} passed")
    # Learning artifacts produced DURING idle (the primary B3 signal).
    L.append("\n## Idle-loop learning artifacts (treatment vs control)")
    for rep in artifacts:
        L.append(f"### repeat {rep.get('repeat')}")
        for arm in ("treatment", "control"):
            a = rep.get(arm, {})
            lbs = a.get("lessons_by_source", {})
            L.append(f"- {arm}: lessons_by_source={lbs} "
                     f"graduated_skills={a.get('graduated_skills', 0)} "
                     f"proposed_macros={a.get('proposed_macros', 0)}")
    return "\n".join(L) + "\n"

COMMON = ["--upstream-url", "http://127.0.0.1:8088", "--api-key", "",
          "--no-mandatory-tor"]


def _treatment_flags(time_scale: float) -> List[str]:
    # Idle loops accelerated + deterministic so they actually fire in the run.
    return ["--bio-time-scale", str(time_scale), "--bio-deterministic",
            "--enable-metacog"]


def _control_flags() -> List[str]:
    # Production idle timings → the idle loops never reach their windows within
    # the short run, isolating the idle loops (memory stays ON in both arms).
    return ["--bio-time-scale", "1"]


def _boot_arm(name, flags, port, boot_timeout, logdir):
    gh = Path(tempfile.mkdtemp(prefix=f"ghost-trackb3-{name}-"))
    proc, lf = AE._boot(["--port", str(port)] + COMMON + flags, gh, logdir / f"{name}.log")
    url = f"http://127.0.0.1:{port}"
    ok = AE._wait_ready(url, boot_timeout)
    AE._log(f"  {'ready' if ok else '!! NOT READY'}: {name} on {url} (pid {proc.pid})")
    return {"proc": proc, "lf": lf, "url": url, "home": gh,
            "container": AE._container_name(gh / "sandbox"), "port": port, "ready": ok}


def _teardown(arm):
    try:
        AE._teardown(arm["proc"], arm["lf"], arm["container"])
    except Exception as e:
        AE._log(f"  teardown error: {e}")


def _learning_artifacts(home: Path) -> Dict[str, Any]:
    """Count what the idle loops produced under this arm's GHOST_HOME."""
    out = {"lessons_by_source": {}, "graduated_skills": 0, "proposed_macros": 0}
    pb = home / "system" / "memory" / "skills_playbook.json"
    try:
        if pb.exists():
            for lesson in json.loads(pb.read_text()):
                src = (lesson.get("source") or "unknown")
                out["lessons_by_source"][src] = out["lessons_by_source"].get(src, 0) + 1
    except Exception:
        pass
    grad = home / "system" / "memory" / "graduated_skills.json"
    try:
        if grad.exists():
            out["graduated_skills"] = len(json.loads(grad.read_text()) or [])
    except Exception:
        pass
    macros = home / "system" / "memory" / "composed_skills.json"
    try:
        if macros.exists():
            data = json.loads(macros.read_text()) or []
            out["proposed_macros"] = sum(
                1 for m in data if (m.get("status") == "proposed"))
    except Exception:
        pass
    return out


async def _seed(url, items, model, timeout):
    for it in items:
        await _post(url, [{"role": "user", "content": it.seed}], model, timeout)


async def _probe(url, items, model, timeout, arm, rep):
    recs = []
    for it in items:
        out, dur = await _post(url, [{"role": "user", "content": it.probe}], model, timeout)
        ok, _ = it.validator(out)
        recs.append({"item_id": it.pair_id, "repeat": rep, "arm": arm,
                     "passed": bool(ok), "duration_s": dur, "out": out[:200]})
    return recs


async def _run_arm(name, flags, args, items, rep, epoch_sleep):
    logdir = Path(args.report_dir); logdir.mkdir(parents=True, exist_ok=True)
    arm = _boot_arm(name, flags, args.base_port, args.boot_timeout, logdir)
    if not arm["ready"]:
        _teardown(arm)
        return [], {"error": "boot failed"}
    try:
        await _seed(arm["url"], items, args.model, args.timeout)
        # Idle through M accelerated epochs so the idle loops fire (treatment)
        # or provably do not (control at scale 1). No requests during the idle.
        for _ in range(args.idle_epochs):
            await asyncio.sleep(epoch_sleep)
        recs = await _probe(arm["url"], items, args.model, args.timeout, name, rep)
        artifacts = _learning_artifacts(arm["home"])
        return recs, artifacts
    finally:
        _teardown(arm)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--base-port", type=int, default=8046)
    ap.add_argument("--time-scale", type=float, default=60.0,
                    help="treatment --bio-time-scale (compresses idle windows)")
    ap.add_argument("--idle-epochs", type=int, default=6,
                    help="how many accelerated idle epochs to sleep through")
    ap.add_argument("--epoch-sleep", type=float, default=70.0,
                    help="seconds per idle epoch (must exceed the scaled window; "
                         "the watchdog ticks every 60s)")
    ap.add_argument("--model", default=os.getenv("GHOST_MODEL", "qwen-3.6-35b-a3"))
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--boot-timeout", type=float, default=300.0)
    ap.add_argument("--report-dir", required=True)
    args = ap.parse_args()

    items = load_trackb_pairs()
    all_records: List[Dict[str, Any]] = []
    all_artifacts: List[Dict[str, Any]] = []

    async def _driver():
        for rep in range(args.repeats):
            AE._log(f"=== B3 repeat {rep+1}/{args.repeats} ===")
            # Sequential arms: the host has room for only ONE throwaway agent
            # plus prod (Track-A OOM lesson).
            t_recs, t_art = await _run_arm(
                "treatment", _treatment_flags(args.time_scale), args, items, rep, args.epoch_sleep)
            c_recs, c_art = await _run_arm(
                "control", _control_flags(), args, items, rep, args.epoch_sleep)
            all_records.extend(t_recs); all_records.extend(c_recs)
            all_artifacts.append({"repeat": rep, "treatment": t_art, "control": c_art})

    asyncio.run(_driver())

    report = _b3_report(all_records, all_artifacts, {"harness": "trackb3",
                                                     "time_scale": args.time_scale,
                                                     "idle_epochs": args.idle_epochs})
    out = Path(args.report_dir)
    (out / "trackb3_report.md").write_text(report)
    (out / "trackb3_artifacts.json").write_text(json.dumps(all_artifacts, indent=2))
    (out / "trackb3_records.json").write_text(json.dumps(all_records, indent=2))
    AE._log(f"B3 report written to {out}")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
