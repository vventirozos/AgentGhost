"""Track B4 — idle-loop OUTCOME adjudication on the grounded battery (§4D).

Extends trackb3: same 3 arms (treatment/frontier, treatment_uniform, control),
but the probes are execution-grounded DOING tasks (`trackb4_tasks.py`) instead
of fact recalls, plus:

  * a SEEDING phase (identical in every arm, before the idle window): easy
    tasks feed strong clusters; hard tasks produce real failures in the
    pre-registered WEAK_CLUSTERS — reflection material, auto-memories, and the
    cluster variance frontier selection needs to have signal;
  * `--smart-memory 0.9` in EVERY arm — B3's arms never passed it, and the
    smart-memory consolidator is the only writer of the `type:"auto"`
    fragments dream's entropy gate counts, so dream was unsatisfiable by
    construction;
  * per-probe MEDIATION capture — did any playbook lesson actually surface
    (retrieval-credit counters bumped) during the probe turn? An outcomes-null
    is uninterpretable without this;
  * a task-STRATIFIED sign-flip test next to the exact McNemar (repeats within
    a task are correlated; treating (task, repeat) as independent over-narrows,
    the same flaw §4B flags for the Wilson CIs);
  * `--pilot` mode: one control-configured agent, `--pilot-repeats` passes over
    the full candidate pool, emits `b4_battery.json` keeping tasks that are
    neither all-pass nor all-fail (the implementable version of the
    [0.3, 0.7] calibration band — with 3 binary samples the band is
    "1 or 2 of 3").

Fixture seeds vary per repeat (base+rep) so a memorised answer can't carry
across repeats, while staying IDENTICAL across arms within a repeat (matched
pairs need matched fixtures).

Run (operator, prod stopped — see PROJECT_JOURNAL.md §2):
    PYTHONPATH=src python scripts/ablation_trackb4.py --pilot \
        --report-dir ablation_out/b4-pilot
    PYTHONPATH=src python scripts/ablation_trackb4.py --repeats 3 \
        --battery-file ablation_out/b4-pilot/b4_battery.json \
        --report-dir ablation_out/b4-<date>
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ablation_eval as AE                      # noqa: E402
from ablation_trackb2 import _post              # noqa: E402
from ablation_trackb3 import (                  # noqa: E402
    _learning_artifacts, _mcnemar_exact,
)
from trackb4_tasks import (                     # noqa: E402
    DEFAULT_SEED, WEAK_CLUSTERS, B4Task, load_b4_battery, load_b4_seeding,
)

COMMON = ["--upstream-url", "http://127.0.0.1:8088", "--api-key", "",
          "--no-mandatory-tor",
          # §4D: every arm consolidates — dream's entropy gate counts
          # type:"auto" fragments that only the smart-memory task writes.
          "--smart-memory", "0.9"]


def _treatment_flags(time_scale: float) -> List[str]:
    return ["--bio-time-scale", str(time_scale), "--bio-deterministic",
            "--enable-metacog"]


def _treatment_uniform_flags(time_scale: float) -> List[str]:
    return _treatment_flags(time_scale) + ["--no-frontier-selfplay"]


def _control_flags() -> List[str]:
    return ["--bio-time-scale", "1"]


def _boot_arm(name, flags, port, boot_timeout, logdir):
    gh = Path(tempfile.mkdtemp(prefix=f"ghost-trackb4-{name}-"))
    proc, lf = AE._boot(["--port", str(port)] + COMMON + flags, gh,
                        logdir / f"{name}.log")
    url = f"http://127.0.0.1:{port}"
    ok = AE._wait_ready(url, boot_timeout)
    AE._log(f"  {'ready' if ok else '!! NOT READY'}: {name} on {url} (pid {proc.pid})")
    return {"proc": proc, "lf": lf, "url": url, "home": gh,
            "container": AE._container_name(gh / "sandbox"),
            "log": logdir / f"{name}.log", "port": port, "ready": ok}


def _teardown(arm):
    try:
        AE._teardown(arm["proc"], arm["lf"], arm["container"])
    except Exception as e:
        AE._log(f"  teardown error: {e}")


# ── grounded fixture/artifact plumbing ──────────────────────────────────────

def _place_fixtures(sandbox: Path, task: B4Task, seed: int) -> Dict[str, str]:
    """Write the task's fixtures into the arm's sandbox and REMOVE any stale
    artifact from a previous pass (a leftover artifact would false-pass)."""
    fixtures = task.fixtures(seed)
    for rel, content in fixtures.items():
        p = sandbox / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    try:
        (sandbox / task.artifact).unlink()
    except FileNotFoundError:
        pass
    return fixtures


def _read_artifact(sandbox: Path, task: B4Task) -> Optional[str]:
    p = sandbox / task.artifact
    try:
        return p.read_text(errors="replace") if p.is_file() else None
    except Exception:
        return None


# ── mediation + idle-loop instrumentation ───────────────────────────────────

def _playbook_retrievals(home: Path) -> Dict[str, int]:
    """{stable lesson key: retrievals} — content-hashed keys survive lessons
    being appended mid-run (index positions do not)."""
    pb = home / "system" / "memory" / "skills_playbook.json"
    out: Dict[str, int] = {}
    try:
        if pb.exists():
            for lesson in (json.loads(pb.read_text()) or []):
                key = hashlib.md5("|".join(
                    str(lesson.get(k, "")) for k in
                    ("task", "mistake", "solution", "source")
                ).encode()).hexdigest()[:16]
                out[key] = int(lesson.get("retrievals", 0) or 0)
    except Exception:
        pass
    return out


def _mediation(before: Dict[str, int], after: Dict[str, int]) -> int:
    """How many lessons SURFACED (retrieval credit bumped) between snapshots."""
    return sum(1 for k, n in after.items() if n > before.get(k, n))


def _log_counts(log_path: Path) -> Dict[str, int]:
    """Instrumentation the pretty-stream already carries — no new agent code."""
    try:
        text = log_path.read_text(errors="replace")
    except Exception:
        return {}
    return {
        "auto_memory_stores": text.count("Auto Memory Store"),
        "dream_entropy_skips": text.count("Not enough entropy to dream"),
        "bus_hydrations": text.count("Hydrated context for"),
        "smart_memory_requeues": text.count("re-queued smart_memory"),
    }


def _failed_trajectories(home: Path) -> int:
    n = 0
    tdir = home / "system" / "trajectories"
    try:
        files = list(tdir.glob("*/*.jsonl"))
    except Exception:
        return 0
    for f in files:
        try:
            for line in f.read_text(errors="replace").splitlines():
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                outcome = str(d.get("final_outcome") or d.get("outcome") or "")
                if outcome.upper() == "FAILED":
                    n += 1
        except Exception:
            continue
    return n


def _frontier_clusters(home: Path) -> Dict[str, int]:
    fp = home / "system" / "memory" / "self_play_frontier.json"
    try:
        data = json.loads(fp.read_text()) if fp.exists() else {}
        return {name: int(c.get("runs", 0) or 0)
                for name, c in (data.get("clusters") or {}).items()}
    except Exception:
        return {}


# ── stats ────────────────────────────────────────────────────────────────────

def _stratified_sign_flip(records: List[dict], arm_a: str = "treatment",
                          arm_b: str = "control", n_perm: int = 10000,
                          seed: int = 0) -> Dict[str, Any]:
    """Task-stratified permutation (sign-flip) test on per-task mean outcome
    differences. Respects within-task correlation across repeats — the
    (task, repeat)-independence assumption behind raw McNemar/Wilson is the
    §4B-flagged flaw."""
    per: Dict[str, Dict[int, Dict[str, int]]] = defaultdict(dict)
    for r in records:
        per[r["task_id"]].setdefault(r["repeat"], {})[r["arm"]] = (
            1 if r.get("passed") else 0)
    deltas: List[float] = []
    for task in sorted(per):
        ds = [reps[arm_a] - reps[arm_b] for reps in per[task].values()
              if arm_a in reps and arm_b in reps]
        if ds:
            deltas.append(sum(ds) / len(ds))
    if not deltas:
        return {"mean_delta": 0.0, "p": 1.0, "n_tasks": 0}
    obs = sum(deltas) / len(deltas)
    if all(d == 0 for d in deltas):
        return {"mean_delta": 0.0, "p": 1.0, "n_tasks": len(deltas)}
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        s = sum(d * rng.choice((-1.0, 1.0)) for d in deltas) / len(deltas)
        if abs(s) >= abs(obs) - 1e-12:
            hits += 1
    return {"mean_delta": obs, "p": hits / n_perm, "n_tasks": len(deltas)}


def _mcnemar_cells(records, arm_a="treatment", arm_b="control"):
    outcome: Dict[str, Dict[tuple, bool]] = {}
    for r in records:
        outcome.setdefault(r["arm"], {})[(r["task_id"], r["repeat"])] = bool(
            r.get("passed"))
    a, bmap = outcome.get(arm_a, {}), outcome.get(arm_b, {})
    keys = sorted(set(a) & set(bmap))
    b = sum(1 for k in keys if a[k] and not bmap[k])
    c = sum(1 for k in keys if bmap[k] and not a[k])
    both = sum(1 for k in keys if a[k] and bmap[k])
    neither = sum(1 for k in keys if not a[k] and not bmap[k])
    return len(keys), both, neither, b, c


# ── phases ───────────────────────────────────────────────────────────────────

async def _wait_arm_quiet(arm, requests_sent: int, grace: float = 900.0,
                          hold: float = 1800.0):
    """A client-side probe timeout does NOT stop the agent's in-flight turn:
    the agent keeps working it (and may write the artifact minutes later),
    while the NEXT request queues behind the turn-serialization semaphore and
    burns its own budget waiting — the pilot's timeout-bleed cascade
    (2026-07-09: conc_worker_sum overran and took web_table_sum +
    web_pdf_links down with it, 0/3 each without ever being measured).
    Wait until the arm's log shows every sent request finished, or the grace
    expires (then log and proceed — a wedged arm shouldn't hang the run)."""
    import time as _t
    deadline = _t.time() + grace
    n = -1
    while _t.time() < deadline:
        try:
            # case-insensitive: the pretty-stream renders the END marker as
            # lowercase "request finished" while the source spells it title-
            # case; counting only one form made this wait ALWAYS burn the
            # full grace (re-pilot #2, 2026-07-09 — +240s dead time per task)
            n = arm["log"].read_text(errors="replace").lower().count("request finished")
        except Exception:
            return
        if n >= requests_sent:
            return
        await asyncio.sleep(5)
    # Proceeding into a busy arm re-creates the cascade (re-pilot #2:
    # a wedged multi-turn loop ate 4 queued probes despite the old 240s
    # grace-then-proceed). Past the grace, keep waiting to a hard ceiling;
    # only a truly wedged arm (> ceiling) is worth abandoning, and every
    # record after that is suspect anyway.
    AE._log(f"  !! arm busy past {grace:.0f}s grace "
            f"(finished {n}/{requests_sent}) — holding up to the ceiling")
    ceiling = _t.time() + hold
    while _t.time() < ceiling:
        try:
            n = arm["log"].read_text(errors="replace").lower().count("request finished")
        except Exception:
            return
        if n >= requests_sent:
            AE._log(f"  arm quiet again (finished {n}/{requests_sent})")
            return
        await asyncio.sleep(min(10.0, max(0.05, hold / 10)))
    AE._log(f"  !! arm STILL busy after the {hold:.0f}s hold ceiling — "
            f"proceeding; treat subsequent records as suspect")


async def _drive_task(arm, task: B4Task, seed: int, model: str,
                      timeout: float, home: Path) -> dict:
    sandbox = home / "sandbox"
    fixtures = _place_fixtures(sandbox, task, seed)
    before = _playbook_retrievals(home)
    out, dur = await _post(arm["url"],
                           [{"role": "user", "content": task.prompt()}],
                           model, timeout)
    artifact = _read_artifact(sandbox, task)
    ok, why = task.verify(artifact or "", fixtures)
    return {"task_id": task.task_id, "cluster": task.cluster,
            "ring": task.ring, "role": task.role,
            "passed": bool(ok), "why": ("" if ok else why)[:160],
            "artifact_found": artifact is not None,
            "mediated_lessons": _mediation(before, _playbook_retrievals(home)),
            "duration_s": dur, "out": (out or "")[:160]}


async def _run_arm(name, flags, args, battery, seeding, rep):
    logdir = Path(args.report_dir)
    logdir.mkdir(parents=True, exist_ok=True)
    arm = _boot_arm(name, flags, args.base_port, args.boot_timeout, logdir)
    if not arm["ready"]:
        _teardown(arm)
        return [], {"error": "boot failed"}
    home = arm["home"]
    seed = DEFAULT_SEED + rep
    records: List[dict] = []
    sent = 0
    try:
        # Phase S — identical seeding in every arm.
        for t in seeding:
            rec = await _drive_task(arm, t, seed, args.model, args.timeout, home)
            sent += 1
            await _wait_arm_quiet(arm, sent)
            rec.update({"arm": name, "repeat": rep, "phase": "seed"})
            records.append(rec)
        instrument_s = {
            "failed_trajectories_after_seed": _failed_trajectories(home),
            "frontier_clusters_after_seed": _frontier_clusters(home),
            **_log_counts(arm["log"]),
        }
        # Phase I — the idle window the loops fire in (or provably don't).
        for _ in range(args.idle_epochs):
            await asyncio.sleep(args.epoch_sleep)
        # Phase P — the calibrated battery.
        for t in battery:
            rec = await _drive_task(arm, t, seed, args.model, args.timeout, home)
            sent += 1
            await _wait_arm_quiet(arm, sent)
            rec.update({"arm": name, "repeat": rep, "phase": "probe"})
            records.append(rec)
        probes = [r for r in records if r["phase"] == "probe"]
        artifacts = {
            **_learning_artifacts(home),
            "seed_instrumentation": instrument_s,
            "log_counts_final": _log_counts(arm["log"]),
            "mediation_rate": (
                sum(1 for r in probes if r["mediated_lessons"] > 0)
                / max(1, len(probes))),
        }
        return records, artifacts
    finally:
        _teardown(arm)


# ── report ───────────────────────────────────────────────────────────────────

def _b4_report(records, artifacts, meta) -> str:
    L = ["# Track B4 — grounded outcome battery", "", f"harness meta: {meta}"]
    probes = [r for r in records if r.get("phase") == "probe"]

    by_arm = defaultdict(lambda: [0, 0])
    for r in probes:
        by_arm[r["arm"]][0] += 1
        by_arm[r["arm"]][1] += 1 if r["passed"] else 0
    L.append("\n## Probe outcomes (grounded verify)")
    for arm, (n, p) in sorted(by_arm.items()):
        L.append(f"- {arm}: {p}/{n} passed ({(p / n if n else 0):.0%})")

    L.append("\n## Treatment vs control")
    pairs, both, neither, b, c = _mcnemar_cells(probes)
    L.append(f"- matched pairs: {pairs} (both pass={both}, both fail={neither}); "
             f"discordant b={b} (treatment-only win), c={c} (control-only win)")
    L.append(f"- exact McNemar two-sided p = {_mcnemar_exact(b, c):.4f}")
    strat = _stratified_sign_flip(probes)
    L.append(f"- task-STRATIFIED sign-flip: mean per-task delta = "
             f"{strat['mean_delta']:+.3f} over {strat['n_tasks']} tasks, "
             f"p = {strat['p']:.4f}  (the primary test — repeats within a "
             f"task are correlated)")

    L.append("\n## Mediation (did lessons actually surface during probes?)")
    for rep_art in artifacts:
        for arm in ("treatment", "treatment_uniform", "control"):
            if arm in rep_art:
                mr = rep_art[arm].get("mediation_rate")
                if mr is not None:
                    L.append(f"- repeat {rep_art['repeat']} {arm}: "
                             f"mediation_rate={mr:.0%}")
    L.append("- pre-registered reading: outcomes-null + mediation≈0 → fix "
             "retrieval routing; outcomes-null + mediation healthy → idle "
             "output doesn't transfer at this scale.")

    L.append("\n## #27b — frontier vs uniform on the WEAK clusters "
             f"{list(WEAK_CLUSTERS)}")
    weak = [r for r in probes if r["cluster"] in WEAK_CLUSTERS]
    for arm in ("treatment", "treatment_uniform"):
        sub = [r for r in weak if r["arm"] == arm]
        if sub:
            p = sum(1 for r in sub if r["passed"])
            L.append(f"- {arm}: weak-cluster probes {p}/{len(sub)} "
                     f"({p / len(sub):.0%})")
    for rep_art in artifacts:
        for arm in ("treatment", "treatment_uniform"):
            if arm in rep_art:
                lbs = rep_art[arm].get("lessons_by_source", {})
                L.append(f"- repeat {rep_art['repeat']} {arm}: "
                         f"lessons_by_source={lbs}")
    L.append("- pre-registered rule (§4D item 6): KEEP frontier iff self-play "
             "yield ≥ uniform in ≥2/3 repeats AND weak-cluster delta ≥ 0; "
             "else flip default to uniform. PRM stays either way.")

    L.append("\n## Dream / idle-loop instrumentation")
    for rep_art in artifacts:
        for arm in ("treatment", "treatment_uniform", "control"):
            if arm in rep_art:
                a = rep_art[arm]
                si = a.get("seed_instrumentation", {})
                lf = a.get("log_counts_final", {})
                L.append(
                    f"- repeat {rep_art['repeat']} {arm}: "
                    f"auto_memories(seed)={si.get('auto_memory_stores', '?')} "
                    f"failed_traj(seed)={si.get('failed_trajectories_after_seed', '?')} "
                    f"dream_skips(final)={lf.get('dream_entropy_skips', '?')} "
                    f"hydrations(final)={lf.get('bus_hydrations', '?')}")
    L.append("- gate reading: skips>0 with auto_memories≥3 = NEW BUG; "
             "auto_memories<3 = seeding still starves the gate.")
    return "\n".join(L) + "\n"


# ── pilot ────────────────────────────────────────────────────────────────────

async def _pilot(args, battery, seeding) -> int:
    """One control-configured agent, N passes over the full candidate pool.
    Emits b4_battery.json with the survivors (neither all-pass nor all-fail)."""
    logdir = Path(args.report_dir)
    logdir.mkdir(parents=True, exist_ok=True)
    arm = _boot_arm("pilot", _control_flags(), args.base_port,
                    args.boot_timeout, logdir)
    if not arm["ready"]:
        _teardown(arm)
        return 1
    per_task: Dict[str, List[dict]] = defaultdict(list)
    sent = 0
    try:
        for rep in range(args.pilot_repeats):
            AE._log(f"=== pilot pass {rep + 1}/{args.pilot_repeats} ===")
            for t in battery:
                rec = await _drive_task(arm, t, DEFAULT_SEED + rep,
                                        args.model, args.timeout, arm["home"])
                sent += 1
                await _wait_arm_quiet(arm, sent)
                per_task[t.task_id].append(rec)
    finally:
        _teardown(arm)
    lines = ["# B4 pilot — candidate difficulty", ""]
    survivors: List[str] = []
    for tid in sorted(per_task):
        recs = per_task[tid]
        passes = sum(1 for r in recs if r["passed"])
        n = len(recs)
        keep = 0 < passes < n
        if keep:
            survivors.append(tid)
        mean_dur = sum(r["duration_s"] for r in recs) / max(1, n)
        lines.append(f"- {tid}: {passes}/{n} passed, ~{mean_dur:.0f}s "
                     f"{'KEEP' if keep else 'DROP (ceiling/floor)'}"
                     + ("" if recs[-1]["passed"] else f"  last_fail: {recs[-1]['why']}"))
    lines.append(f"\nsurvivors: {len(survivors)}/{len(per_task)} → b4_battery.json")
    lines.append("(operator may hand-tune: a 3/3 task can stay if it probes a "
                 "cluster nothing else covers — see per-task rows above)")
    report = "\n".join(lines) + "\n"
    (logdir / "b4_pilot_report.md").write_text(report)
    (logdir / "b4_pilot_records.json").write_text(
        json.dumps(per_task, indent=2, default=str))
    (logdir / "b4_battery.json").write_text(json.dumps(survivors, indent=2))
    print(report)
    return 0


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--base-port", type=int, default=8046)
    ap.add_argument("--time-scale", type=float, default=60.0)
    ap.add_argument("--idle-epochs", type=int, default=8)
    ap.add_argument("--epoch-sleep", type=float, default=70.0)
    ap.add_argument("--model", default=os.getenv("GHOST_MODEL", "qwen-3.6-35b-a3"))
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--boot-timeout", type=float, default=300.0)
    ap.add_argument("--no-frontier-arm", action="store_true")
    ap.add_argument("--pilot", action="store_true",
                    help="calibration mode: one control agent × --pilot-repeats "
                         "passes over ALL candidates; emits b4_battery.json")
    ap.add_argument("--pilot-repeats", type=int, default=3)
    ap.add_argument("--battery-file", default=None,
                    help="JSON list of task_ids (from the pilot) to probe with; "
                         "default = the full candidate pool")
    ap.add_argument("--report-dir", required=True)
    args = ap.parse_args()

    candidates = load_b4_battery()
    seeding = load_b4_seeding()

    if args.pilot:
        # --battery-file also filters PILOT candidates, so a re-pilot can
        # measure only new/re-raced tasks instead of repeating the pool.
        if args.battery_file:
            keep = set(json.loads(Path(args.battery_file).read_text()))
            candidates = [t for t in candidates if t.task_id in keep]
            AE._log(f"pilot subset: {len(candidates)} tasks from {args.battery_file}")
        return asyncio.run(_pilot(args, candidates, seeding))

    battery = candidates
    if args.battery_file:
        keep = set(json.loads(Path(args.battery_file).read_text()))
        battery = [t for t in candidates if t.task_id in keep]
        AE._log(f"battery: {len(battery)}/{len(candidates)} tasks from "
                f"{args.battery_file}")

    all_records: List[dict] = []
    all_artifacts: List[dict] = []

    async def _driver():
        for rep in range(args.repeats):
            AE._log(f"=== B4 repeat {rep + 1}/{args.repeats} ===")
            t_recs, t_art = await _run_arm(
                "treatment", _treatment_flags(args.time_scale), args,
                battery, seeding, rep)
            arts = {"repeat": rep, "treatment": t_art}
            if not args.no_frontier_arm:
                u_recs, u_art = await _run_arm(
                    "treatment_uniform", _treatment_uniform_flags(args.time_scale),
                    args, battery, seeding, rep)
                all_records.extend(u_recs)
                arts["treatment_uniform"] = u_art
            c_recs, c_art = await _run_arm(
                "control", _control_flags(), args, battery, seeding, rep)
            all_records.extend(t_recs)
            all_records.extend(c_recs)
            arts["control"] = c_art
            all_artifacts.append(arts)

    asyncio.run(_driver())

    report = _b4_report(all_records, all_artifacts,
                        {"harness": "trackb4", "time_scale": args.time_scale,
                         "idle_epochs": args.idle_epochs,
                         "battery_size": len(battery)})
    out = Path(args.report_dir)
    (out / "trackb4_report.md").write_text(report)
    (out / "trackb4_artifacts.json").write_text(json.dumps(all_artifacts, indent=2))
    (out / "trackb4_records.json").write_text(json.dumps(all_records, indent=2))
    AE._log(f"B4 report written to {out}")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
