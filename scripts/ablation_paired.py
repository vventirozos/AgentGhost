"""Paired (time-matched) ablation driver — the rigorous full-vs-thin probe.

WHY THIS EXISTS
---------------
`ablation_eval.py auto` boots one config, scores it to completion, tears it
down, then does the next. That is a SEQUENTIAL design: config A is measured at
a different wall-clock moment than config B, so any drift in shared-upstream
load (e.g. the production agent on :8000 doing idle reflection/dream work)
becomes a confound that is indistinguishable from a real config effect.

This driver removes that confound with a PAIRED design:

  * every config is booted at once, each on its own port + fresh GHOST_HOME;
  * for each (repeat, task) we fire the SAME task at every config back-to-back,
    in randomized order, within a few seconds of each other.

Because the matched calls hit the shared upstream within seconds, whatever load
the upstream is under at that instant is (statistically) the SAME for every
config. The paired comparison (McNemar on the matched pass/fail pairs) cancels
that shared load out. Latency is still reported but is secondary — pass/fail is
the load-robust signal.

Cross-session caveat still applies: every config runs on a FRESH, EMPTY
GHOST_HOME, so memory/selfhood/workspace/reflection start blank and cannot help
on single-shot tasks. This probe measures the IN-SESSION layers only.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ablation_eval as AE  # noqa: E402  (reuse runner/lifecycle/stats)


# Config flag sets (mirror scripts/ablation_configs.json, minus the per-config port).
CONFIG_FLAGS: Dict[str, List[str]] = {
    "full": ["--enable-metacog", "--deep-reason", "--enable-preflight-guard",
             "--postmortem", "--smart-memory", "0.9", "--autoadvance-idle"],
    "full_no_metacog": ["--deep-reason", "--enable-preflight-guard",
                        "--postmortem", "--smart-memory", "0.9", "--autoadvance-idle"],
    "full_no_deepreason": ["--enable-metacog", "--enable-preflight-guard",
                          "--postmortem", "--smart-memory", "0.9", "--autoadvance-idle"],
    "full_no_preflight": ["--enable-metacog", "--deep-reason", "--no-enable-preflight-guard",
                         "--postmortem", "--smart-memory", "0.9", "--autoadvance-idle"],
    "thin": ["--no-self-model", "--no-workspace-model", "--no-reflection",
             "--no-enable-preflight-guard"],
}

COMMON = ["--upstream-url", "http://127.0.0.1:8088", "--api-key", "",
          "--no-mandatory-tor"]


def _port_free(port: int) -> bool:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _boot_all(names: List[str], base_port: int, boot_timeout: float, logdir: Path):
    """Boot every config SEQUENTIALLY (boot -> wait ready -> next). Sequential
    boot avoids the simultaneous-bind race and halves peak boot-time memory.
    Returns dict name -> {proc, lf, url, home, container, port}."""
    live: Dict[str, Any] = {}
    for i, name in enumerate(names):
        port = base_port + i
        if not _port_free(port):
            AE._log(f"  !! port :{port} is BUSY before booting '{name}' — aborting boot")
            raise SystemExit(f"port {port} busy")
        gh = Path(tempfile.mkdtemp(prefix=f"ghost-pair-{name}-"))
        sandbox = gh / "sandbox"
        flags = ["--port", str(port)] + COMMON + CONFIG_FLAGS[name]
        proc, lf = AE._boot(flags, gh, logdir / f"{name}.log")
        base_url = f"http://127.0.0.1:{port}"
        container = AE._container_name(sandbox)
        live[name] = {"proc": proc, "lf": lf, "url": base_url,
                      "home": gh, "container": container, "port": port}
        AE._log(f"  booting '{name}' on :{port} (pid {proc.pid}) GHOST_HOME={gh}")
        # gate on readiness before booting the next config
        if AE._wait_ready(base_url, boot_timeout):
            AE._log(f"  ready: '{name}' on {base_url}")
        else:
            AE._log(f"  !! '{name}' never became ready on {base_url}")
            if proc.poll() is not None:
                AE._log(f"  !! '{name}' process EXITED (code {proc.returncode}) — check its log")
    return live


def _teardown_all(live: Dict[str, Any]) -> None:
    for name, d in live.items():
        AE._log(f"  tearing down '{name}' (:{d['port']})")
        AE._teardown(d["proc"], d["lf"], d["container"])


async def _run_one(url: str, model: str, timeout: float, task) -> Tuple[bool, float, str]:
    runner = AE._make_http_runner(url, "", model, timeout)
    try:
        out = await runner(task, None)
    except Exception as e:
        return False, float(timeout), f"runner error: {type(e).__name__}: {e}"
    output = out.get("output", "")
    dur = float(out.get("duration_s", 0.0))
    try:
        from ghost_agent.eval.tasks import ChallengeTemplateTask as _CTT
        # For a template task the ONLY valid pass signal is the sandbox
        # verdict — hand validate() the whole runner dict so a missing
        # `passed` key is scored unverified/fail, not passed on any non-empty
        # text (mirrors eval/suite.py). Passing the bare string here made
        # every template task false-pass, and scored them opposite to the
        # sequential ablation driver.
        validate_input = out if isinstance(task, _CTT) else output
        passed, reason = task.validate(validate_input, None)
    except Exception as e:
        passed, reason = False, f"validator raised: {type(e).__name__}: {e}"
    return bool(passed), dur, "" if passed else str(reason)


async def _measure(live: Dict[str, Any], suite_tasks, repeats: int, model: str,
                   timeout: float, seed: int, base_rep: int = 0,
                   total_repeats: int = 0,
                   prior: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    names = list(live.keys())
    records: List[Dict[str, Any]] = list(prior or [])
    new: List[Dict[str, Any]] = []
    for rep in range(repeats):
        grep = base_rep + rep
        order = list(suite_tasks)
        rng.shuffle(order)
        for task in order:
            cfg_order = names[:]
            rng.shuffle(cfg_order)  # randomize who-goes-first within the matched pair
            for name in cfg_order:
                d = live[name]
                passed, dur, reason = await _run_one(d["url"], model, timeout, task)
                rec = {
                    "config": name, "repeat": grep,
                    "task_id": getattr(task, "task_id", "?"),
                    "cluster": getattr(task, "cluster", "?") or "?",
                    "passed": passed, "duration_s": dur, "reason": reason,
                }
                new.append(rec)
                records.append(rec)
        per = {n: sum(1 for r in new if r["repeat"] == grep and r["config"] == n and r["passed"])
               for n in names}
        AE._log(f"  repeat {grep+1}/{total_repeats or repeats} done "
                + " | ".join(f"{n}:{per[n]}/{len(suite_tasks)}" for n in names))
        # checkpoint cumulative records after every repeat
        _CKPT.write_text(json.dumps(records))
    return new


# ---- paired statistics -----------------------------------------------------

def _mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value over discordant pairs (b vs c)."""
    n = b + c
    if n == 0:
        return 1.0
    from math import comb
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def _paired_diff_ci(pairs: List[Tuple[bool, bool]], z: float = 1.96) -> Tuple[float, float, float, int, int]:
    """pairs = [(ref_passed, cfg_passed)]. Returns (diff, lo, hi, b, c) where
    diff = p_ref - p_cfg (positive => ref better), via the Wald paired interval."""
    n = len(pairs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0, 0
    b = sum(1 for r, c in pairs if r and not c)   # ref pass, cfg fail
    c = sum(1 for r, c in pairs if (not r) and c) # ref fail, cfg pass
    diff = (b - c) / n
    var = (b + c - (b - c) ** 2 / n) / (n * n)
    se = math.sqrt(max(var, 0.0))
    return diff, diff - z * se, diff + z * se, b, c


def _within_task_variance(records: List[Dict[str, Any]], cfg: str) -> float:
    """Mean per-task variance of pass/fail across repeats for one config.
    ~0 => the model is effectively deterministic on this suite (repeats add
    little; effective n is the #tasks, not #trials)."""
    by_task: Dict[str, List[int]] = {}
    for r in records:
        if r["config"] != cfg:
            continue
        by_task.setdefault(r["task_id"], []).append(1 if r["passed"] else 0)
    vs = []
    for tid, xs in by_task.items():
        if len(xs) < 2:
            continue
        m = sum(xs) / len(xs)
        vs.append(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return sum(vs) / len(vs) if vs else 0.0


def _build_report(records: List[Dict[str, Any]], reference: str, meta: Dict[str, Any]) -> str:
    names = sorted({r["config"] for r in records})
    L: List[str] = []
    L.append("# Paired ablation report — does the cognitive layer help (in-session)?\n")
    L.append(f"_Generated {datetime.datetime.now().isoformat(timespec='seconds')}_  ")
    L.append(f"_Design: PAIRED / time-matched. Reference: **{reference}**. "
             f"Suite: {meta.get('suite')} ({meta.get('n_tasks')} tasks) × "
             f"{meta.get('repeats')} repeats = {meta.get('n_pairs')} matched pairs/config._\n")

    # per-config headline
    L.append("## Per-config success (Wilson 95% CI)\n")
    L.append("| config | success | 95% CI | mean lat/task | within-task variance |")
    L.append("|---|---|---|---|---|")
    stats = {}
    for n in [reference] + [x for x in names if x != reference]:
        recs = [r for r in records if r["config"] == n]
        k = sum(1 for r in recs if r["passed"]); tot = len(recs)
        p, lo, hi = AE.wilson(k, tot)
        lat = sum(r["duration_s"] for r in recs) / max(tot, 1)
        wv = _within_task_variance(records, n)
        stats[n] = (k, tot, p)
        L.append(f"| `{n}` | {k}/{tot} ({p*100:.0f}%) | {lo*100:.0f}%–{hi*100:.0f}% | "
                 f"{lat:.0f}s | {wv:.3f} |")
    L.append("")

    # paired comparisons vs reference
    L.append(f"## Paired comparison vs `{reference}` (McNemar exact)\n")
    L.append("`diff = p_ref − p_cfg` (positive ⇒ reference better). `b` = ref-pass/cfg-fail, "
             "`c` = ref-fail/cfg-pass. Only discordant pairs (b,c) carry signal.\n")
    L.append("| config | diff (ref−cfg) | 95% CI | b | c | McNemar p | verdict |")
    L.append("|---|---|---|---|---|---|---|")
    for n in names:
        if n == reference:
            continue
        # match pairs by (repeat, task_id)
        ref_by = {(r["repeat"], r["task_id"]): r["passed"]
                  for r in records if r["config"] == reference}
        cfg_by = {(r["repeat"], r["task_id"]): r["passed"]
                  for r in records if r["config"] == n}
        keys = sorted(set(ref_by) & set(cfg_by))
        pairs = [(ref_by[k], cfg_by[k]) for k in keys]
        diff, lo, hi, b, c = _paired_diff_ci(pairs)
        p = _mcnemar_exact(b, c)
        if p < 0.05 and diff > 0:
            verdict = "✅ removing it HURTS → layer earns its place (KEEP)"
        elif p < 0.05 and diff < 0:
            verdict = "⛔ removing it HELPS → layer is net-negative (CUT)"
        else:
            verdict = "➖ no significant difference (in-session)"
        L.append(f"| `{n}` | {diff*100:+.1f}% | [{lo*100:+.1f}%, {hi*100:+.1f}%] | "
                 f"{b} | {c} | {p:.3f} | {verdict} |")
    L.append("")

    # per-task pass rates (reference vs thin if present)
    L.append("## Per-task pass rate (where any effect lives)\n")
    tasks = sorted({r["task_id"] for r in records})
    L.append("| task | cluster | " + " | ".join(f"`{n}`" for n in names) + " |")
    L.append("|---|---|" + "|".join(["---"] * len(names)) + "|")
    for tid in tasks:
        cluster = next((r["cluster"] for r in records if r["task_id"] == tid), "?")
        cells = []
        for n in names:
            recs = [r for r in records if r["config"] == n and r["task_id"] == tid]
            k = sum(1 for r in recs if r["passed"]); tot = len(recs)
            cells.append(f"{k}/{tot}")
        L.append(f"| {tid} | {cluster} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## How to read this\n")
    L.append("- **Headline** = `thin` row in the paired table: that is the TOTAL in-session "
             "lift of the whole cognitive stack over the stripped baseline.\n"
             "- A per-layer row that is significant & positive ⇒ that single layer carries "
             "weight on this suite. Non-significant ⇒ no measurable in-session effect "
             "(for cross-session layers that is EXPECTED on a fresh GHOST_HOME — Track B).\n"
             "- If `within-task variance ≈ 0`, the model is near-deterministic here: the "
             "effective sample size is the number of TASKS, not trials — treat per-task "
             "results as the real evidence and don't over-trust tight trial-based CIs.")
    return "\n".join(L) + "\n"


_CKPT = Path("/tmp/ablation_paired_ckpt.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="full,thin",
                    help="comma list from: " + ",".join(CONFIG_FLAGS))
    ap.add_argument("--reference", default="full")
    ap.add_argument("--suite", default="default",
                    choices=("ablation", "default", "post_learning", "hard"))
    ap.add_argument("--only-tasks", default="",
                    help="comma list of task_id substrings to keep (post-calibration subset)")
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--base-port", type=int, default=8010)
    ap.add_argument("--model", default=os.getenv("GHOST_MODEL", "qwen-3.6-35b-a3"))
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--boot-timeout", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--reboot-every", type=int, default=0,
                    help="reboot all agents every N repeats (fresh GHOST_HOME) to "
                         "cap RSS growth / avoid OOM on long runs; 0 = never")
    ap.add_argument("--report-dir", required=True)
    args = ap.parse_args()

    names = [c.strip() for c in args.configs.split(",") if c.strip()]
    for n in names:
        if n not in CONFIG_FLAGS:
            raise SystemExit(f"unknown config {n!r}; known: {list(CONFIG_FLAGS)}")
    if args.reference not in names:
        raise SystemExit(f"reference {args.reference!r} must be in --configs")

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    logdir = report_dir / "agent-logs"
    logdir.mkdir(exist_ok=True)
    global _CKPT
    _CKPT = report_dir / "checkpoint.json"

    if args.suite == "hard":
        from ablation_hard_tasks import load_hard_suite
        suite_tasks = load_hard_suite()
    else:
        suite_tasks = AE._load_suite(args.suite)
    if args.only_tasks:
        keep = [s.strip() for s in args.only_tasks.split(",") if s.strip()]
        suite_tasks = [t for t in suite_tasks
                       if any(k in getattr(t, "task_id", "") for k in keep)]
    n_pairs = args.repeats * len(suite_tasks)
    AE._log(f"PAIRED ablation: configs={names} suite={args.suite} "
            f"tasks={len(suite_tasks)} repeats={args.repeats} "
            f"=> {n_pairs} matched pairs/config")

    # preflight: upstream reachable, docker up, ports free
    probs = AE._preflight("http://127.0.0.1:8088", f"http://127.0.0.1:{args.base_port}")
    probs = [p for p in probs if "ALREADY listening" not in p]  # we boot several ports ourselves
    if probs:
        for p in probs:
            AE._log(f"  PREFLIGHT PROBLEM: {p}")
        return 2

    reboot_every = args.reboot_every if args.reboot_every > 0 else args.repeats
    records: List[Dict[str, Any]] = []
    done = 0
    burst = 0
    while done < args.repeats:
        n_rep = min(reboot_every, args.repeats - done)
        AE._log(f"--- burst {burst} : repeats {done+1}..{done+n_rep} of {args.repeats} "
                f"(fresh agents) ---")
        live = _boot_all(names, args.base_port, args.boot_timeout, logdir)
        try:
            new = asyncio.run(_measure(live, suite_tasks, n_rep, args.model,
                                       args.timeout, args.seed + burst,
                                       base_rep=done, total_repeats=args.repeats,
                                       prior=records))
            records.extend(new)
        finally:
            _teardown_all(live)
        done += n_rep
        burst += 1

    meta = {"suite": args.suite, "n_tasks": len(suite_tasks),
            "repeats": args.repeats, "n_pairs": n_pairs, "reference": args.reference}
    (report_dir / "records.json").write_text(json.dumps(
        {"meta": meta, "records": records}, indent=2))
    report = _build_report(records, args.reference, meta)
    (report_dir / "REPORT.md").write_text(report)
    AE._log(f"REPORT -> {report_dir / 'REPORT.md'}")
    print("\n" + report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
