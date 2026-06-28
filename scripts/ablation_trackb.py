"""Track B (B1) — cross-session RETENTION measurement.

Question Track A cannot answer: does the agent's persistent memory actually make
it better on LATER tasks? Track A runs each config on a fresh empty GHOST_HOME,
so the cross-session layers start blank and cannot help — that is by design.

B1 measures passive recall (vector / profile / episodic / graph), which is
written SYNCHRONOUSLY during a turn, so no idle-consolidation is needed.

Design (see trackb_tasks.py): each item is a (SEED, PROBE) pair sent as two
INDEPENDENT requests (no shared chat history).

  TREATMENT agent — memory ON, persistent across the two calls. Gets SEED, then
                    (after a short consolidation delay) PROBE.
  CONTROL  agent — booted with --no-memory. Gets PROBE only. Cannot know the
                    seeded fact through any channel.

Both agents are up at once and the two PROBE calls run back-to-back, so shared
upstream load is matched. McNemar's exact test on the matched (treatment-probe,
control-probe) pass/fail pairs is the verdict. Agents are re-booted every repeat
to bound RSS growth (the OOM that bit Track A).

Run:
    PYTHONPATH=src python scripts/ablation_trackb.py --repeats 5 \
        --base-port 8040 --delay 8 --report-dir <out>
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ablation_eval as AE          # noqa: E402  (lifecycle + stats)
import ablation_paired as AP        # noqa: E402  (_mcnemar_exact, _paired_diff_ci)
from trackb_tasks import load_trackb_pairs   # noqa: E402

COMMON = ["--upstream-url", "http://127.0.0.1:8088", "--api-key", "",
          "--no-mandatory-tor"]
# B1 isolates ONE variable: persistent memory. Treatment = defaults (all memory
# tiers ON); control = --no-memory (every persistent store off). We deliberately
# do NOT add --deep-reason/--postmortem here — they add latency without affecting
# fact recall and would muddy the isolation (they're Track A territory).
TREATMENT_FLAGS: List[str] = []
# the clean control: every persistent store off
CONTROL_FLAGS = ["--no-memory"]

_CKPT = Path("/tmp/ablation_trackb_ckpt.json")


async def _post(url: str, content: str, model: str, timeout: float):
    """POST a single-user-message chat completion; return (output, duration_s)."""
    import httpx
    payload = {"model": model, "stream": False,
               "messages": [{"role": "user", "content": content}]}
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(f"{url.rstrip('/')}/v1/chat/completions",
                                  json=payload)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        return f"__error__: {type(e).__name__}: {e}", time.monotonic() - t0
    msg = (data.get("choices", [{}])[0].get("message", {}).get("content", "")
           or data.get("message", {}).get("content", ""))
    return str(msg or ""), time.monotonic() - t0


def _boot_arm(name: str, flags: List[str], port: int, boot_timeout: float, logdir: Path):
    gh = Path(tempfile.mkdtemp(prefix=f"ghost-trackb-{name}-"))
    sandbox = gh / "sandbox"
    proc, lf = AE._boot(["--port", str(port)] + COMMON + flags, gh, logdir / f"{name}.log")
    url = f"http://127.0.0.1:{port}"
    ok = AE._wait_ready(url, boot_timeout)
    AE._log(f"  {'ready' if ok else '!! NOT READY'}: {name} on {url} (pid {proc.pid})")
    return {"proc": proc, "lf": lf, "url": url, "home": gh,
            "container": AE._container_name(sandbox), "port": port, "ready": ok}


async def _run_repeat(rep: int, pairs, treat, ctrl, model, timeout, delay,
                      total, rng) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    order = list(pairs)
    rng.shuffle(order)
    for p in order:
        # TREATMENT: seed (separate conversation), consolidation pause, then probe.
        seed_out, _ = await _post(treat["url"], p.seed, model, timeout)
        await asyncio.sleep(delay)
        t_probe, t_dur = await _post(treat["url"], p.probe, model, timeout)
        # CONTROL: probe only, no seed ever delivered.
        c_probe, c_dur = await _post(ctrl["url"], p.probe, model, timeout)
        t_pass, _ = p.validator(t_probe)
        c_pass, _ = p.validator(c_probe)
        recs.append({"pair_id": p.pair_id, "kind": p.kind, "repeat": rep,
                     "arm": "treatment", "passed": bool(t_pass),
                     "duration_s": t_dur, "out": t_probe[:160]})
        recs.append({"pair_id": p.pair_id, "kind": p.kind, "repeat": rep,
                     "arm": "control", "passed": bool(c_pass),
                     "duration_s": c_dur, "out": c_probe[:160]})
    tp = sum(1 for r in recs if r["arm"] == "treatment" and r["passed"])
    cp = sum(1 for r in recs if r["arm"] == "control" and r["passed"])
    AE._log(f"  repeat {rep+1}/{total}: treatment {tp}/{len(pairs)} recalled | "
            f"control {cp}/{len(pairs)} (expected ~0)")
    return recs


def _build_report(records, meta) -> str:
    pairs = sorted({r["pair_id"] for r in records})
    L: List[str] = []
    L.append("# Track B (B1) — cross-session retention: does persistent memory help later tasks?\n")
    L.append(f"_Generated {datetime.datetime.now().isoformat(timespec='seconds')}_  ")
    L.append(f"_seed→probe pairs: {meta['n_pairs']} × {meta['repeats']} repeats. "
             f"TREATMENT = memory ON & persistent; CONTROL = --no-memory (never saw the seed)._\n")

    for arm in ("treatment", "control"):
        recs = [r for r in records if r["arm"] == arm]
        k = sum(1 for r in recs if r["passed"]); n = len(recs)
        p, lo, hi = AE.wilson(k, n)
        L.append(f"- **{arm}**: {k}/{n} recalled ({p*100:.0f}%, 95% CI {lo*100:.0f}–{hi*100:.0f}%)")
    L.append("")

    # paired McNemar: does treatment recall what control cannot?
    keys = sorted({(r["repeat"], r["pair_id"]) for r in records})
    tre = {(r["repeat"], r["pair_id"]): r["passed"] for r in records if r["arm"] == "treatment"}
    con = {(r["repeat"], r["pair_id"]): r["passed"] for r in records if r["arm"] == "control"}
    ppairs = [(tre.get(k, False), con.get(k, False)) for k in keys]
    diff, dlo, dhi, b, c = AP._paired_diff_ci(ppairs)
    pval = AP._mcnemar_exact(b, c)
    if pval < 0.05 and diff > 0:
        verdict = "✅ MEMORY HELPS — treatment recalls facts the control cannot (cross-session value confirmed)"
    elif pval < 0.05 and diff < 0:
        verdict = "⛔ control beats treatment (something is badly wrong)"
    else:
        verdict = "➖ NO measurable retention benefit — treatment recalls no better than a memoryless agent"
    L.append(f"\n## Paired verdict (McNemar exact)\n")
    L.append(f"treatment−control recall diff = **{diff*100:+.0f}%** "
             f"[{dlo*100:+.0f}%, {dhi*100:+.0f}%], b(treat✓/ctrl✗)={b}, "
             f"c(treat✗/ctrl✓)={c}, p={pval:.4f}\n\n**{verdict}**\n")

    # per-pair breakdown
    L.append("## Per-pair recall (treatment / control)\n")
    L.append("| pair | tier | treatment | control |\n|---|---|---|---|")
    for pid in pairs:
        kind = next((r["kind"] for r in records if r["pair_id"] == pid), "?")
        tr = [r for r in records if r["pair_id"] == pid and r["arm"] == "treatment"]
        co = [r for r in records if r["pair_id"] == pid and r["arm"] == "control"]
        tk = sum(1 for r in tr if r["passed"]); ck = sum(1 for r in co if r["passed"])
        L.append(f"| {pid} | {kind} | {tk}/{len(tr)} | {ck}/{len(co)} |")
    L.append("")
    L.append("## How to read this\n")
    L.append("- **treatment ≫ control** ⇒ persistent memory genuinely carries facts "
             "across separate requests and changes later answers — the core cross-session "
             "claim holds.\n"
             "- **treatment ≈ control (both low)** ⇒ the agent fails to recall facts it was "
             "explicitly told to remember: the memory machinery is not delivering in-the-loop "
             "value, regardless of how elaborate the stores are.\n"
             "- This is B1 (passive recall). The ACTIVE learning loops (reflection, skills, "
             "dream) write only during idle phases and need separate triggering — that is B2.")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--base-port", type=int, default=8040)
    ap.add_argument("--delay", type=float, default=8.0,
                    help="seconds between SEED and PROBE (memory consolidation window)")
    ap.add_argument("--model", default=os.getenv("GHOST_MODEL", "qwen-3.6-35b-a3"))
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--boot-timeout", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=20260628)
    ap.add_argument("--report-dir", required=True)
    args = ap.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    logdir = report_dir / "agent-logs"
    logdir.mkdir(exist_ok=True)
    global _CKPT
    _CKPT = report_dir / "checkpoint.json"

    pairs = load_trackb_pairs()
    AE._log(f"Track B (B1): {len(pairs)} seed→probe pairs × {args.repeats} repeats")

    probs = [p for p in AE._preflight("http://127.0.0.1:8088",
                                      f"http://127.0.0.1:{args.base_port}")
             if "ALREADY listening" not in p]
    if probs:
        for p in probs:
            AE._log(f"  PREFLIGHT PROBLEM: {p}")
        return 2

    rng = random.Random(args.seed)
    records: List[Dict[str, Any]] = []
    for rep in range(args.repeats):
        AE._log(f"--- repeat {rep+1}/{args.repeats}: booting fresh treatment + control ---")
        treat = _boot_arm("treatment", TREATMENT_FLAGS, args.base_port,
                          args.boot_timeout, logdir)
        ctrl = _boot_arm("control", CONTROL_FLAGS, args.base_port + 1,
                         args.boot_timeout, logdir)
        try:
            recs = asyncio.run(_run_repeat(rep, pairs, treat, ctrl, args.model,
                                           args.timeout, args.delay, args.repeats, rng))
            records.extend(recs)
            _CKPT.write_text(json.dumps(records))
        finally:
            AE._teardown(treat["proc"], treat["lf"], treat["container"])
            AE._teardown(ctrl["proc"], ctrl["lf"], ctrl["container"])

    meta = {"n_pairs": len(pairs), "repeats": args.repeats}
    (report_dir / "records.json").write_text(json.dumps({"meta": meta, "records": records}, indent=2))
    report = _build_report(records, meta)
    (report_dir / "REPORT.md").write_text(report)
    AE._log(f"REPORT -> {report_dir / 'REPORT.md'}")
    print("\n" + report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
