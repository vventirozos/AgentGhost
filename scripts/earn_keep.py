"""Earn-your-keep — the standing self-measuring / self-pruning harness.

Every cognitive subsystem must prove it EARNS ITS KEEP or get pruned. This is
the orchestration layer over the existing ablation machinery:

  * `run`  — one on-demand measurement. Boots the leave-one-out config matrix
    (`full` + one `full_no_<x>` per subsystem + `thin`) via the paired driver
    (scripts/ablation_paired.py), fires the hard battery at every arm
    back-to-back (paired → shared-upstream load cancels), and APPENDS the raw
    per-(arm, repeat, task) pass/fail to a durable ledger. Then it re-attributes
    over the WHOLE ledger and AUTO-PRUNES any subsystem that has sustained a
    "doesn't help + costs latency" verdict.
  * `report` — aggregate the ledger → per-subsystem marginal contribution
    (Δ pass-rate vs `full`, bootstrap CI, latency cost, run count) → a ranked
    keep/prune table. `--apply` runs the same auto-prune without a fresh run.

The subsystem↔arm mapping + the prod-apply of a prune live in
`core.prune_overrides` (the single source of truth, shared with main.py).

Run (operator, PROD STOPPED — the paired matrix needs 2+ throwaway agents +
the 35B, which don't fit alongside prod; see PROJECT_JOURNAL.md §2):
    PYTHONPATH=src GHOST_HOME=<live-home> python scripts/earn_keep.py run \
        --track A --repeats 3
    PYTHONPATH=src GHOST_HOME=<live-home> python scripts/earn_keep.py report
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import random
import socket
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "src"))

from ghost_agent.core import prune_overrides as PO  # noqa: E402

AP = None  # scripts/ablation_paired, imported lazily in cmd_run (heavy)

# Track A measures PER-TURN cognition (metacog / deep-reason / verifier / …),
# which fires on the task turns — NOT the biological IDLE watchdog. But the LOO
# matrix boots ~9 arms that SHARE one llama-server slot (-np 1), and each arm's
# own idle loops (dream / REM / self-play / reflection / autoadvance) fire
# between the driver's measured tasks, contending on the single GPU. That load
# is per-arm ASYMMETRIC, which the paired design does NOT cancel (it cancels
# GLOBAL upstream load, not each arm's own background churn) — so it poisons the
# very latency/outcome signal we measure, and slows the run to a crawl. A tiny
# bio-time-scale pushes every idle-window bound out to ~days, so no idle phase
# fires during a run. Applied UNIFORMLY to every arm → cannot bias the paired
# comparison. (Track B, which MEASURES the idle loops, will do the opposite.)
IDLE_QUIESCE_FLAGS = ["--bio-time-scale", "0.001"]


def apply_track_a_quiesce(ap_module) -> List[str]:
    """Append the idle-suppression flags to the paired driver's COMMON (shared
    by every arm's boot). Idempotent; returns the resulting COMMON."""
    if "--bio-time-scale" not in ap_module.COMMON:
        ap_module.COMMON = list(ap_module.COMMON) + IDLE_QUIESCE_FLAGS
    return ap_module.COMMON

# ── pre-registered sustained-verdict rule (do not move post-hoc) ─────────────
MIN_RUNS = 3          # ≥3 separate runs before any auto-prune
MIN_PAIRS = 60        # ≥60 pooled matched pairs
DELTA_UPPER = 0.02    # 90% CI upper bound on Δ-help must be < +2pp to prune
CI_Z_PERCENTILES = (5.0, 95.0)   # 90% bootstrap interval
BOOTSTRAP_N = 2000

EK_ROOT = REPO_ROOT / "ablation_out" / "earn_keep"
LEDGER = EK_ROOT / "results.jsonl"


# ── run directory: progress log + checkpoint + manifest (resumable) ──────────

class Progress:
    """Tees the whole-process narrative to a line-buffered `progress.log` AND
    stdout, so `tail -f progress.log` mirrors what the operator sees live. Also
    installed over ablation_eval._log during a run, so the boot/teardown chatter
    lands in the same unified log."""
    def __init__(self, path: Path):
        self.path = path
        self.f = open(path, "a", buffering=1)

    def __call__(self, msg: str) -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        try:
            self.f.write(line + "\n")
        except Exception:
            pass

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


def _run_dir(run_id: str) -> Path:
    return EK_ROOT / f"run-{run_id}"


def read_manifest(run_id: str) -> Dict[str, Any]:
    p = _run_dir(run_id) / "manifest.json"
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        return {}


def write_manifest(run_id: str, manifest: Dict[str, Any]) -> None:
    d = _run_dir(run_id)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, d / "manifest.json")


def find_resumable(track: Optional[str] = None) -> Optional[str]:
    """The most recent run whose manifest status != 'complete' (optionally
    filtered by track). None → nothing to resume."""
    cands = []
    if not EK_ROOT.exists():
        return None
    for d in EK_ROOT.glob("run-*"):
        m = read_manifest(d.name[len("run-"):])
        if m and m.get("status") != "complete" and (track is None or m.get("track") == track):
            cands.append((m.get("created_at", ""), d.name[len("run-"):]))
    cands.sort()
    return cands[-1][1] if cands else None


def load_done_groups(checkpoint: Path, arm_names) -> set:
    """{(repeat, task_id)} for groups where EVERY arm has a checkpointed cell —
    the resume skip-set. Group-atomic writes mean a partially-run group (killed
    mid-group) has no complete arm-set and is redone."""
    seen: Dict[tuple, set] = defaultdict(set)
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
                seen[(c["repeat"], c["task_id"])].add(c["config"])
            except Exception:
                continue
    need = set(arm_names)
    return {k for k, cfgs in seen.items() if need <= cfgs}


def append_group(checkpoint: Path, cells: List[Dict[str, Any]]) -> None:
    """Write all arm-cells for ONE (repeat, task) group atomically-ish: a single
    write of N newline-joined records, flushed. A kill before this leaves the
    group un-checkpointed (redone on resume); a kill after leaves it complete."""
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("a") as f:
        f.write("".join(json.dumps(c) + "\n" for c in cells))
        f.flush()
        os.fsync(f.fileno())


def fold_checkpoint(checkpoint: Path, run_ts: str, track: str) -> List[Dict[str, Any]]:
    """Checkpoint cells → ledger records (stamped with run_ts+track), deduped
    keep-LAST per (config, repeat, task_id) so a redone group can't double-count."""
    latest: Dict[tuple, Dict[str, Any]] = {}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except Exception:
                continue
            latest[(c["config"], c["repeat"], c["task_id"])] = c
    return [{"run_ts": run_ts, "track": track, "config": c["config"],
             "repeat": c["repeat"], "task_id": c["task_id"],
             "cluster": c.get("cluster", "?"), "passed": bool(c["passed"]),
             "duration_s": float(c.get("duration_s", 0.0))}
            for c in latest.values()]


# ── ledger I/O ──────────────────────────────────────────────────────────────

def append_records(ledger: Path, recs: List[Dict[str, Any]]) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def load_ledger(ledger: Path) -> List[Dict[str, Any]]:
    if not ledger.exists():
        return []
    out = []
    for line in ledger.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue          # tolerate a torn tail line
    return out


# ── attribution (PURE — the testable core) ──────────────────────────────────

def _matched_pairs(records, ref_cfg, arm_cfg) -> List[Tuple[bool, bool]]:
    """[(ref_passed, arm_passed)] over cells (run_ts, repeat, task_id) present
    in BOTH configs. The matched key cancels task difficulty + shared load."""
    ref = {(r["run_ts"], r["repeat"], r["task_id"]): bool(r["passed"])
           for r in records if r["config"] == ref_cfg}
    arm = {(r["run_ts"], r["repeat"], r["task_id"]): bool(r["passed"])
           for r in records if r["config"] == arm_cfg}
    return [(ref[k], arm[k]) for k in ref if k in arm]


def _bootstrap_diff_ci(pairs, seed: int = 12345) -> Tuple[float, float, float]:
    """Δ = p_ref - p_arm (positive ⇒ the subsystem HELPS), with a percentile
    bootstrap CI. Deterministic given a seed (for tests)."""
    n = len(pairs)
    if n == 0:
        return 0.0, 0.0, 0.0
    diffs_ref = [1 if r else 0 for r, _ in pairs]
    diffs_arm = [1 if a else 0 for _, a in pairs]
    point = (sum(diffs_ref) - sum(diffs_arm)) / n
    rng = random.Random(seed)
    boot = []
    idx = range(n)
    for _ in range(BOOTSTRAP_N):
        s = [rng.randrange(n) for _ in idx]
        d = sum(diffs_ref[i] - diffs_arm[i] for i in s) / n
        boot.append(d)
    boot.sort()
    lo = boot[int(CI_Z_PERCENTILES[0] / 100 * (BOOTSTRAP_N - 1))]
    hi = boot[int(CI_Z_PERCENTILES[1] / 100 * (BOOTSTRAP_N - 1))]
    return point, lo, hi


def _mean_dur(records, cfg) -> float:
    ds = [float(r.get("duration_s", 0.0)) for r in records if r["config"] == cfg]
    return sum(ds) / len(ds) if ds else 0.0


def attribute(records: List[Dict[str, Any]], ref_cfg: str = "full") -> Dict[str, Dict[str, Any]]:
    """Per-subsystem marginal contribution over the whole ledger. For each
    subsystem: Δ-help (ref minus arm), bootstrap CI, latency cost (ref mean dur
    minus arm mean dur; positive ⇒ the subsystem adds latency), pooled-pair and
    distinct-run counts, and the verdict."""
    n_runs_total = len({r["run_ts"] for r in records})
    out: Dict[str, Dict[str, Any]] = {}
    for name, spec in PO.SUBSYSTEMS.items():
        arm = spec["arm"]
        pairs = _matched_pairs(records, ref_cfg, arm)
        runs = len({r["run_ts"] for r in records if r["config"] == arm}
                   & {r["run_ts"] for r in records if r["config"] == ref_cfg})
        delta, lo, hi = _bootstrap_diff_ci(pairs)
        lat = _mean_dur(records, ref_cfg) - _mean_dur(records, arm)
        stats = {
            "arm": arm, "delta_help": delta, "ci_lo": lo, "ci_hi": hi,
            "latency_cost_s": lat, "n_pairs": len(pairs), "n_runs": runs,
            "protected": spec["protected"], "costs": spec["costs"],
            "track": spec["track"],
        }
        stats["verdict"] = verdict(name, stats)
        out[name] = stats
    out["_meta"] = {"n_runs_total": n_runs_total, "ref": ref_cfg}
    return out


def verdict(name: str, stats: Dict[str, Any]) -> str:
    """The pre-registered rule. 'protected'/'keep_free' are never auto-pruned;
    'prune' requires sustained evidence AND a real cost."""
    if stats["protected"] or name in PO.PROTECTED:
        return "protected"
    if not stats["costs"]:
        return "keep_free"        # free subsystem: harmless to keep even if useless
    if stats["n_runs"] < MIN_RUNS or stats["n_pairs"] < MIN_PAIRS:
        return "insufficient"
    if stats["delta_help"] <= 0.0 and stats["ci_hi"] < DELTA_UPPER:
        return "prune"            # doesn't help (CI rules out a >2pp benefit) + costs
    return "keep"


# ── auto-prune ──────────────────────────────────────────────────────────────

def auto_prune(attribution: Dict[str, Dict[str, Any]], ghost_home,
               now_iso: str, dry_run: bool = False) -> List[str]:
    """Prune every subsystem whose verdict is 'prune' and that isn't already
    pruned. Protected/unknown are refused by prune_overrides. Returns names
    newly pruned. Loud."""
    already = PO.load_pruned(ghost_home)
    newly = []
    for name, stats in attribution.items():
        if name == "_meta" or stats.get("verdict") != "prune" or name in already:
            continue
        evidence = {k: stats[k] for k in
                    ("delta_help", "ci_lo", "ci_hi", "latency_cost_s",
                     "n_pairs", "n_runs")}
        print(f"⚖️  AUTO-PRUNE: '{name}' does not earn its keep "
              f"(Δ={stats['delta_help']:+.3f}, 90% CI [{stats['ci_lo']:+.3f}, "
              f"{stats['ci_hi']:+.3f}], +{stats['latency_cost_s']:.1f}s latency, "
              f"{stats['n_pairs']} pairs / {stats['n_runs']} runs)"
              + ("  [dry-run]" if dry_run else ""))
        if not dry_run:
            PO.record_prune(ghost_home, name, evidence, now_iso)
        newly.append(name)
    return newly


# ── report rendering ────────────────────────────────────────────────────────

def render_report(attribution: Dict[str, Dict[str, Any]]) -> str:
    meta = attribution.get("_meta", {})
    rows = [(n, s) for n, s in attribution.items() if n != "_meta"]
    rows.sort(key=lambda kv: kv[1]["delta_help"])   # worst (most prunable) first
    lines = [f"Earn-your-keep attribution  (ref={meta.get('ref','full')}, "
             f"{meta.get('n_runs_total',0)} runs in ledger)",
             f"{'subsystem':<18}{'track':<6}{'Δ-help':>8}{'90% CI':>18}"
             f"{'latency':>9}{'pairs':>7}{'runs':>6}  verdict",
             "-" * 82]
    for name, s in rows:
        ci = f"[{s['ci_lo']:+.3f},{s['ci_hi']:+.3f}]"
        lines.append(
            f"{name:<18}{s['track']:<6}{s['delta_help']:>+8.3f}{ci:>18}"
            f"{s['latency_cost_s']:>+8.1f}s{s['n_pairs']:>7}{s['n_runs']:>6}"
            f"  {s['verdict']}")
    return "\n".join(lines)


# ── live run ────────────────────────────────────────────────────────────────

def _prod_running(prod_port: int = 8000) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        return s.connect_ex(("127.0.0.1", prod_port)) == 0
    finally:
        s.close()


async def _measure_resumable(live, arm_names, suite, repeats, model, timeout,
                             seed, checkpoint: Path, prog: Progress,
                             manifest: Dict[str, Any], run_id: str) -> None:
    """Fire every (repeat, task) GROUP — all arms back-to-back (paired, so shared
    upstream load cancels) — skipping groups already in the checkpoint. Each
    completed group is checkpointed atomically; a kill mid-group redoes only that
    group on resume. Per-cell progress is logged live regardless."""
    # Deterministic group order (stable across resume): tasks shuffled per repeat
    # by the run seed. Skip-set decides what's already done, so order only affects
    # where a resume picks up, never correctness.
    rng = random.Random(seed)
    groups = []
    for rep in range(repeats):
        order = list(suite)
        rng.shuffle(order)
        for task in order:
            groups.append((rep, task))
    done = load_done_groups(checkpoint, arm_names)
    total = len(groups)
    prog(f"groups: {total} total, {len(done)} already done (resuming)"
         if done else f"groups: {total} total")
    for gi, (rep, task) in enumerate(groups):
        tid = getattr(task, "task_id", "?")
        if (rep, tid) in done:
            continue
        arm_rng = random.Random((seed, rep, tid).__hash__())
        cfg_order = list(arm_names)
        arm_rng.shuffle(cfg_order)      # randomize who-goes-first within the pair
        cells = []
        prog(f"group {gi + 1}/{total}  repeat={rep} task={tid}")
        for name in cfg_order:
            passed, dur, reason = await AP._run_one(live[name]["url"], model, timeout, task)
            cells.append({"config": name, "repeat": rep, "task_id": tid,
                          "cluster": getattr(task, "cluster", "?") or "?",
                          "passed": bool(passed), "duration_s": float(dur)})
            prog(f"    {name:<24} {'PASS' if passed else 'fail'}  "
                 f"{dur:5.0f}s" + ("" if passed else f"  ({reason[:60]})"))
        append_group(checkpoint, cells)     # group is now durably complete
        manifest["groups_done"] = manifest.get("groups_done", 0) + 1
        manifest["updated_at"] = datetime.datetime.now().isoformat()
        write_manifest(run_id, manifest)


def cmd_run(args) -> int:
    global AP
    import ablation_paired as AP
    from ablation_hard_tasks import load_hard_suite

    if args.track != "A":
        print("Track B (idle/cross-session) is Phase 2 — not yet wired. Use --track A.")
        return 2
    ghost_home = os.environ.get("GHOST_HOME")
    if not ghost_home:
        print("!! GHOST_HOME must be set (the live home receives any prune).")
        return 2

    # Preflight BEFORE any run-dir/manifest is created, so a refused run leaves
    # no resumable stub. Prod must be down — the paired matrix needs 2+ throwaway
    # agents + the 35B, which won't fit alongside prod (RAM).
    if _prod_running() and not args.allow_prod:
        print("!! Prod appears UP on :8000 — the paired matrix won't fit "
              "alongside it (RAM). Stop prod (PROJECT_JOURNAL.md §2), or pass "
              "--allow-prod if the box has headroom.")
        return 2

    names = list(AP.CONFIG_FLAGS.keys())    # full LOO + thin
    suite = load_hard_suite()

    # Track A: quiesce the idle loops so only the driver's tasks hit the shared
    # GPU (clean, uncontended, uniform-across-arms measurement).
    if args.track == "A":
        apply_track_a_quiesce(AP)

    # ── resolve the run id: fresh, or resume an incomplete one ──────────────
    if args.resume or args.run_id:
        run_id = args.run_id or find_resumable(track=args.track)
        if not run_id:
            print("!! nothing to resume (no incomplete run found). Start fresh "
                  "without --resume.")
            return 2
        manifest = read_manifest(run_id)
        if not manifest:
            print(f"!! run {run_id!r} has no manifest — cannot resume.")
            return 2
        if manifest.get("status") == "complete":
            print(f"!! run {run_id} is already complete. Use `report`.")
            return 0
    else:
        incomplete = find_resumable(track=args.track)
        if incomplete and not args.force_new:
            print(f"!! an incomplete run exists: {incomplete}. Resume it with "
                  f"`--resume`, or start a fresh one with `--force-new`.")
            return 2
        run_id = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        manifest = {"run_id": run_id, "track": args.track, "arms": names,
                    "repeats": args.repeats, "seed": args.seed,
                    "model": args.model, "timeout": args.timeout,
                    "total_groups": args.repeats * len(suite), "groups_done": 0,
                    "status": "running",
                    "created_at": datetime.datetime.now().isoformat()}

    rundir = _run_dir(run_id)
    rundir.mkdir(parents=True, exist_ok=True)
    write_manifest(run_id, manifest)
    checkpoint = rundir / "checkpoint.jsonl"
    prog = Progress(rundir / "progress.log")

    # One unified process log: route ablation_eval's boot/teardown chatter into
    # progress.log too.
    AP.AE._log = prog

    prog("═" * 70)
    prog(f"earn-keep run {run_id}  track={args.track}  "
         f"{'RESUME' if (args.resume or args.run_id) else 'fresh'}")
    prog(f"  arms ({len(names)}): {', '.join(names)}")
    prog(f"  battery: {len(suite)} tasks × {args.repeats} repeats  "
         f"seed={args.seed}")
    if args.track == "A":
        prog("  idle loops QUIESCED (--bio-time-scale 0.001) — only measured "
             "tasks hit the shared GPU (no dream/self-play/reflection contention)")
    prog("  MONITOR:")
    prog(f"    whole process : tail -f {rundir / 'progress.log'}")
    prog(f"    an agent      : tail -f {rundir}/<arm>.log   "
         f"(e.g. full.log, thin.log)")
    prog("═" * 70)

    prog(f"booting {len(names)} arms …")
    live = AP._boot_all(names, args.base_port, args.boot_timeout, rundir)
    try:
        asyncio.run(_measure_resumable(
            live, names, suite, args.repeats, args.model, args.timeout,
            args.seed, checkpoint, prog, manifest, run_id))
    except KeyboardInterrupt:
        prog("!! interrupted — progress is checkpointed. Resume with `--resume`.")
        AP._teardown_all(live)
        prog.close()
        return 130
    finally:
        AP._teardown_all(live)

    # ── completion: fold checkpoint → durable ledger, then attribute/prune ──
    done_now = load_done_groups(checkpoint, names)
    if len(done_now) < manifest["total_groups"]:
        prog(f"!! run ended with {len(done_now)}/{manifest['total_groups']} "
             f"groups done — NOT folded into the ledger. Resume with `--resume`.")
        prog.close()
        return 1
    recs = fold_checkpoint(checkpoint, run_id, args.track)
    append_records(LEDGER, recs)
    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.datetime.now().isoformat()
    write_manifest(run_id, manifest)
    prog(f"run {run_id} COMPLETE — folded {len(recs)} records → {LEDGER}")

    attribution = attribute(load_ledger(LEDGER))
    prog("\n" + render_report(attribution) + "\n")
    if args.no_prune:
        prog("(--no-prune: auto-prune skipped)")
    else:
        pruned = auto_prune(attribution, ghost_home, datetime.datetime.now().isoformat())
        prog(f"auto-pruned this run: {pruned or 'none'}  "
             f"(applies on prod's next restart)")
    prog.close()
    return 0


def cmd_report(args) -> int:
    attribution = attribute(load_ledger(LEDGER))
    print(render_report(attribution))
    if args.apply:
        ghost_home = os.environ.get("GHOST_HOME")
        if not ghost_home:
            print("\n!! GHOST_HOME must be set to --apply prunes.")
            return 2
        now_iso = datetime.datetime.now().isoformat()
        pruned = auto_prune(attribution, ghost_home, now_iso, dry_run=args.dry_run)
        print(f"\n{'would prune' if args.dry_run else 'pruned'}: {pruned or 'none'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Earn-your-keep self-measuring harness")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="one on-demand LOO measurement (prod stopped)")
    r.add_argument("--track", default="A", choices=["A", "B"])
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--base-port", type=int, default=8010)
    r.add_argument("--model", default=os.getenv("GHOST_MODEL", "qwen-3.6-35b-a3"))
    r.add_argument("--timeout", type=float, default=300.0)
    r.add_argument("--boot-timeout", type=float, default=300.0)
    r.add_argument("--seed", type=int, default=1234)
    r.add_argument("--no-prune", action="store_true", help="measure only, don't auto-prune")
    r.add_argument("--allow-prod", action="store_true", help="run even if prod is up")
    r.add_argument("--resume", action="store_true",
                   help="resume the latest incomplete run (skips done groups)")
    r.add_argument("--run-id", default="",
                   help="resume a specific run id (e.g. 20260722T031500)")
    r.add_argument("--force-new", action="store_true",
                   help="start a fresh run even if an incomplete one exists")

    rep = sub.add_parser("report", help="attribution table from the ledger")
    rep.add_argument("--apply", action="store_true", help="auto-prune sustained losers")
    rep.add_argument("--dry-run", action="store_true", help="with --apply: show, don't write")

    args = ap.parse_args()
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "report":
        return cmd_report(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
