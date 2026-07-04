"""Track B2 — does the agent LEARN a rule from corrective feedback and apply it
in a LATER, SEPARATE session? (cross-session autonomous learning)

See trackb2_tasks.py for the protocol. TREATMENT (memory ON) gets task →
correction → (consolidation delay) → probe across three independent requests;
CONTROL (--no-memory) gets only the probe. McNemar on the matched probe outcomes.

Run:
    PYTHONPATH=src python scripts/ablation_trackb2.py --repeats 3 \
        --base-port 8044 --delay 25 --report-dir <out>
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
import ablation_eval as AE          # noqa: E402
import ablation_paired as AP        # noqa: E402
from trackb2_tasks import load_trackb2_items   # noqa: E402

COMMON = ["--upstream-url", "http://127.0.0.1:8088", "--api-key", "",
          "--no-mandatory-tor"]
# Treatment needs the learning loops that act on a correction: reflection (inline
# on user-correction) + metacog. Memory is on by default.
TREATMENT_FLAGS = ["--enable-metacog"]
CONTROL_FLAGS = ["--no-memory"]

_CKPT = Path("/tmp/ablation_trackb2_ckpt.json")


async def _post(url: str, messages: List[Dict[str, str]], model: str, timeout: float):
    import httpx
    payload = {"model": model, "stream": False, "messages": messages}
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(f"{url.rstrip('/')}/v1/chat/completions", json=payload)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        return f"__error__: {type(e).__name__}: {e}", time.monotonic() - t0
    msg = (data.get("choices", [{}])[0].get("message", {}).get("content", "")
           or data.get("message", {}).get("content", ""))
    return str(msg or ""), time.monotonic() - t0


def _boot_arm(name, flags, port, boot_timeout, logdir):
    gh = Path(tempfile.mkdtemp(prefix=f"ghost-trackb2-{name}-"))
    proc, lf = AE._boot(["--port", str(port)] + COMMON + flags, gh, logdir / f"{name}.log")
    url = f"http://127.0.0.1:{port}"
    ok = AE._wait_ready(url, boot_timeout)
    AE._log(f"  {'ready' if ok else '!! NOT READY'}: {name} on {url} (pid {proc.pid})")
    return {"proc": proc, "lf": lf, "url": url, "home": gh,
            "container": AE._container_name(gh / "sandbox"), "port": port, "ready": ok}


async def _run_treatment(rep, items, url, model, timeout, delay):
    recs: List[Dict[str, Any]] = []
    for it in items:
        # task -> capture answer -> correction (prior answer in history so the
        # correction-promotion fingerprint matches) -> consolidation pause -> probe.
        a1, _ = await _post(url, [{"role": "user", "content": it.task}], model, timeout)
        corr_msgs = [
            {"role": "user", "content": it.task},
            {"role": "assistant", "content": a1},
            {"role": "user", "content": it.correction},
        ]
        await _post(url, corr_msgs, model, timeout)
        await asyncio.sleep(delay)   # let inline reflect_one + memory writes land
        probe, dur = await _post(url, [{"role": "user", "content": it.probe}], model, timeout)
        ok, _ = it.validator(probe)
        recs.append({"item_id": it.item_id, "repeat": rep, "arm": "treatment",
                     "passed": bool(ok), "duration_s": dur, "out": probe[:200]})
    return recs


async def _run_control(rep, items, url, model, timeout):
    recs: List[Dict[str, Any]] = []
    for it in items:
        probe, dur = await _post(url, [{"role": "user", "content": it.probe}], model, timeout)
        ok, _ = it.validator(probe)
        recs.append({"item_id": it.item_id, "repeat": rep, "arm": "control",
                     "passed": bool(ok), "duration_s": dur, "out": probe[:200]})
    return recs


def _build_report(records, meta) -> str:
    items = sorted({r["item_id"] for r in records})
    L: List[str] = []
    L.append("# Track B2 — autonomous learning from correction (cross-session)\n")
    L.append(f"_Generated {datetime.datetime.now().isoformat(timespec='seconds')}_  ")
    L.append(f"_{meta['n_items']} correction items × {meta['repeats']} repeats. "
             f"TREATMENT learns a rule via correction then is probed in a separate "
             f"session; CONTROL (--no-memory) gets the probe only._\n")
    for arm in ("treatment", "control"):
        recs = [r for r in records if r["arm"] == arm]
        k = sum(1 for r in recs if r["passed"]); n = len(recs)
        p, lo, hi = AE.wilson(k, n)
        L.append(f"- **{arm}**: {k}/{n} applied the rule ({p*100:.0f}%, 95% CI {lo*100:.0f}–{hi*100:.0f}%)")
    keys = sorted({(r["repeat"], r["item_id"]) for r in records})
    tre = {(r["repeat"], r["item_id"]): r["passed"] for r in records if r["arm"] == "treatment"}
    con = {(r["repeat"], r["item_id"]): r["passed"] for r in records if r["arm"] == "control"}
    pp = [(tre.get(k, False), con.get(k, False)) for k in keys]
    diff, dlo, dhi, b, c = AP._paired_diff_ci(pp)
    pval = AP._mcnemar_exact(b, c)
    if pval < 0.05 and diff > 0:
        verdict = "✅ LEARNS FROM CORRECTION — a rule taught once persists & is applied in a later separate session"
    elif pval < 0.05 and diff < 0:
        verdict = "⛔ control beats treatment (broken)"
    else:
        verdict = "➖ NO measurable learning — corrected rule does not carry to a later session better than a memoryless agent"
    L.append(f"\n## Paired verdict (McNemar exact)\n")
    L.append(f"treatment−control diff = **{diff*100:+.0f}%** [{dlo*100:+.0f}%, {dhi*100:+.0f}%], "
             f"b={b}, c={c}, p={pval:.4f}\n\n**{verdict}**\n")
    L.append("## Per-item (treatment / control)\n")
    L.append("| item | treatment | control |\n|---|---|---|")
    for iid in items:
        tr = [r for r in records if r["item_id"] == iid and r["arm"] == "treatment"]
        co = [r for r in records if r["item_id"] == iid and r["arm"] == "control"]
        L.append(f"| {iid} | {sum(1 for r in tr if r['passed'])}/{len(tr)} | "
                 f"{sum(1 for r in co if r['passed'])}/{len(co)} |")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--base-port", type=int, default=8044)
    ap.add_argument("--delay", type=float, default=25.0,
                    help="seconds after the correction before probing (reflect_one is async)")
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

    items = load_trackb2_items()
    AE._log(f"Track B2: {len(items)} correction items × {args.repeats} repeats")
    probs = [p for p in AE._preflight("http://127.0.0.1:8088", f"http://127.0.0.1:{args.base_port}")
             if "ALREADY listening" not in p]
    # Arms run sequentially on base_port (one alive at a time), so that port
    # must be free at start. Bind-test it — the default preflight filters out
    # "ALREADY listening", which let a stale listener silently break an arm
    # (see the Track B1 port bug: control booted, hit EADDRINUSE, shut down,
    # and its dead probes fabricated a bogus 0/N).
    import socket as _socket
    _s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    try:
        _s.bind(("0.0.0.0", args.base_port))
    except OSError:
        probs.append(f"arm port {args.base_port} already in use — free it or pick another --base-port")
    finally:
        _s.close()
    if probs:
        for p in probs:
            AE._log(f"  PREFLIGHT PROBLEM: {p}")
        return 2

    rng = random.Random(args.seed)
    records: List[Dict[str, Any]] = []
    for rep in range(args.repeats):
        order = list(items)
        rng.shuffle(order)
        # Arms run SEQUENTIALLY (one throwaway agent alive at a time) — the host
        # baseline runs at ~94% RAM with the 35B server, so two throwaway agents
        # + prod OOM'd the driver. Pairing is by (repeat, item_id), so running
        # the arms one after another doesn't affect the McNemar statistics.
        AE._log(f"--- repeat {rep+1}/{args.repeats}: TREATMENT arm ---")
        treat = _boot_arm("treatment", TREATMENT_FLAGS, args.base_port, args.boot_timeout, logdir)
        if not treat.get("ready"):
            AE._log("  ABORT: treatment arm not ready — not scoring (a dead arm "
                    "fabricates results). See agent-logs/treatment.log.")
            AE._teardown(treat["proc"], treat["lf"], treat["container"])
            return 3
        try:
            t_recs = asyncio.run(_run_treatment(rep, order, treat["url"], args.model,
                                                args.timeout, args.delay))
            records.extend(t_recs)
            _CKPT.write_text(json.dumps(records))
        finally:
            AE._teardown(treat["proc"], treat["lf"], treat["container"])
        tp = sum(1 for r in t_recs if r["passed"])
        AE._log(f"  treatment {tp}/{len(items)} applied learned rule")

        AE._log(f"--- repeat {rep+1}/{args.repeats}: CONTROL arm ---")
        ctrl = _boot_arm("control", CONTROL_FLAGS, args.base_port, args.boot_timeout, logdir)
        if not ctrl.get("ready"):
            AE._log("  ABORT: control arm not ready — not scoring. See agent-logs/control.log.")
            AE._teardown(ctrl["proc"], ctrl["lf"], ctrl["container"])
            return 3
        try:
            c_recs = asyncio.run(_run_control(rep, order, ctrl["url"], args.model, args.timeout))
            records.extend(c_recs)
            _CKPT.write_text(json.dumps(records))
        finally:
            AE._teardown(ctrl["proc"], ctrl["lf"], ctrl["container"])
        cp = sum(1 for r in c_recs if r["passed"])
        AE._log(f"  control {cp}/{len(items)}")

    meta = {"n_items": len(items), "repeats": args.repeats}
    (report_dir / "records.json").write_text(json.dumps({"meta": meta, "records": records}, indent=2))
    report = _build_report(records, meta)
    (report_dir / "REPORT.md").write_text(report)
    AE._log(f"REPORT -> {report_dir / 'REPORT.md'}")
    print("\n" + report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
