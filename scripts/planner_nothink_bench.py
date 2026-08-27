#!/usr/bin/env python3
"""Does disabling the planner's <think> prelude fix truncation without
breaking the plan? — a PAIRED replay of real recorded planner calls.

WHY A REPLAY AND NOT A LIVE ARM (2026-08-11). Real user traffic is scarce
(§4AQ), so a live A/B on the planner would need
months. The recordings hold every planner payload the agent has actually
issued, so the same prompts can be re-run under both conditions today. Paired
on the identical input, which is what makes a 30-item sample worth reading —
the §4V lesson that a delta inside the unpaired CI is only resolvable when
each item is its own control.

THE TWO ARMS, on identical payloads:
  baseline  — exactly as production sends it
  no_think  — `/no_think` appended to the user turn + `chat_template_kwargs:
              {"enable_thinking": false}`, the same two-switch pair
              `tools/vision.py` already applies to `verify_ui`

WHAT IT MEASURES, mechanically — no judge, nothing that needs an opinion:
  * truncated      finish_reason == "length"  (the defect being fixed)
  * parsed         extract_json_from_text returns a non-empty dict
  * has_tree       a `tree_update` object came back
  * focus_valid    `next_action_id` names a node that EXISTS in that tree.
                   ⚠ This is the plan-QUALITY guard and the reason a judge is
                   not needed for the first pass: a plan whose focus points at
                   a task that does not exist is broken on its own terms, no
                   taste required. If no-think degrades planning, this is
                   where it shows up first.
  * tokens/latency the cost side

Usage:
    PYTHONPATH=src python -u scripts/planner_nothink_bench.py --smoke 2
    PYTHONPATH=src python -u scripts/planner_nothink_bench.py --n 30
    PYTHONPATH=src python -u scripts/planner_nothink_bench.py --report-only
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import urllib.request  # noqa: E402

from ghost_agent.core.agent import extract_json_from_text  # noqa: E402
from ghost_agent.eval.runprogress import RunProgress  # noqa: E402

PLANNER_SYS_HEAD = "### Task\nDecompose the user's request"
ARMS = ("baseline", "no_think")


def _home() -> Path:
    return Path(os.getenv("GHOST_HOME") or (Path.home() / "ghost_llamacpp"))


def load_planner_payloads(limit_days: int = 0):
    """Every recorded planner call, newest partition last.

    Identified by CALL SITE (json_object + the decompose system prompt), not
    by max_tokens — the cap is exactly what this bench changes, so keying on
    it would silently drop the post-raise recordings.
    """
    out = []
    seen = set()
    for f in sorted(glob.glob(str(_home() / "system" / "llm_recordings"
                                  / "2026-*.jsonl")))[-limit_days:] or \
            sorted(glob.glob(str(_home() / "system" / "llm_recordings"
                                 / "2026-*.jsonl"))):
        for line in open(f, errors="replace"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            pay = r.get("payload") or {}
            if (pay.get("response_format") or {}).get("type") != "json_object":
                continue
            msgs = pay.get("messages") or []
            if not any(m.get("role") == "system"
                       and str(m.get("content") or "").startswith(
                           PLANNER_SYS_HEAD) for m in msgs):
                continue
            ch = ((r.get("response") or {}).get("choices") or [{}])[0]
            key = hashlib.sha256(
                json.dumps(msgs, sort_keys=True).encode()).hexdigest()[:16]
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "key": key,
                "messages": msgs,
                "orig_finish": ch.get("finish_reason"),
            })
    return out


def stratified(payloads, n: int):
    """Half from calls that TRUNCATED, half from calls that completed.

    The truncated half is where the fix must work; the completed half is where
    it must not do harm. Sampling only the first would measure a rescue and
    call it an improvement.
    """
    cut = [p for p in payloads if p["orig_finish"] == "length"]
    ok = [p for p in payloads if p["orig_finish"] != "length"]
    half = max(1, n // 2)
    # Deterministic: newest-first within each stratum, no RNG (scripts here
    # must reproduce exactly on a re-run).
    return list(reversed(cut))[:half] + list(reversed(ok))[:n - half]


def build_request(messages, arm: str, max_tokens: int):
    msgs = json.loads(json.dumps(messages))
    body = {
        "model": os.getenv("GHOST_BENCH_MODEL", "default"),
        "messages": msgs,
        "temperature": 0.0,
        "top_p": 0.1,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if arm == "no_think":
        for m in reversed(msgs):
            if m.get("role") == "user":
                m["content"] = str(m.get("content") or "") + "\n\n/no_think"
                break
        # Hard switch beside the soft one — vision.py's comment records that
        # the soft switch alone is not reliable on a thinking model.
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


def call(url: str, body: dict, timeout: float):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode())
    return data, time.time() - t0


def _nodes(n):
    if not isinstance(n, dict):
        return []
    out = [str(n.get("id") or "")]
    for c in (n.get("children") or []):
        out.extend(_nodes(c))
    return [x for x in out if x]


def score(data: dict, dt: float) -> dict:
    ch = ((data.get("choices") or [{}])[0])
    msg = ch.get("message") or {}
    content = str(msg.get("content") or "")
    reasoning = str(msg.get("reasoning_content") or "")
    parsed = extract_json_from_text(content)
    tree = parsed.get("tree_update") if isinstance(parsed, dict) else None
    ids = _nodes(tree if isinstance(tree, dict) else {})
    focus = str((parsed or {}).get("next_action_id") or "")
    return {
        "truncated": ch.get("finish_reason") == "length",
        "parsed": bool(parsed),
        "has_thought": bool((parsed or {}).get("thought")),
        "has_tree": bool(ids),
        "n_nodes": len(ids),
        # Only meaningful when a focus was named AND a tree came back;
        # None keeps "not applicable" out of the failure count.
        "focus_valid": (focus in ids) if (focus and ids) else None,
        "completion_tokens": (data.get("usage") or {}).get("completion_tokens"),
        "reasoning_chars": len(reasoning),
        "content_chars": len(content),
        "latency_s": round(dt, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=os.getenv(
        "GHOST_BENCH_URL", "http://127.0.0.1:8088/v1/chat/completions"))
    ap.add_argument("--n", type=int, default=30, help="payloads (×2 arms)")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--smoke", type=int, default=0,
                    help="run N payloads and print both arms raw — proves the "
                         "arms are DISTINGUISHABLE before a long run")
    ap.add_argument("--out", default="ablation_out/planner_nothink.jsonl")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    payloads = load_planner_payloads()
    # BOUNDED: the tool prints its own denominator. Never derive one.
    print(f"planner payloads on disk: {len(payloads)} "
          f"(truncated {sum(p['orig_finish'] == 'length' for p in payloads)})",
          flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # RESUMABLE: completed (key, arm) pairs are skipped on a re-run.
    done = {}
    if out_path.exists():
        for line in open(out_path, errors="replace"):
            try:
                r = json.loads(line)
                done[(r["key"], r["arm"])] = r
            except Exception:
                continue
    print(f"cached results: {len(done)}", flush=True)

    if args.report_only:
        return report(list(done.values()))

    sample = stratified(payloads, args.smoke or args.n)
    todo = [(p, a) for p in sample for a in ARMS if (p["key"], a) not in done]
    print(f"to run: {len(todo)} calls "
          f"({len(sample)} payloads × {len(ARMS)} arms)", flush=True)

    prog = RunProgress(str(out_path.with_suffix(".progress.json")),
                       total=len(todo), label="planner_nothink")
    results = list(done.values())
    with open(out_path, "a") as fh:
        for i, (p, arm) in enumerate(todo, 1):
            body = build_request(p["messages"], arm, args.max_tokens)
            try:
                data, dt = call(args.url, body, args.timeout)
                row = {"key": p["key"], "arm": arm,
                       "orig_finish": p["orig_finish"], **score(data, dt)}
            except Exception as e:                      # noqa: BLE001
                row = {"key": p["key"], "arm": arm,
                       "orig_finish": p["orig_finish"],
                       "error": f"{type(e).__name__}: {e}"}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            results.append(row)
            prog.tick(extra={"arm": arm})
            r = prog.rate_per_min()
            print(f"[{i}/{len(todo)}] {arm:<9} {p['key'][:8]} "
                  f"trunc={row.get('truncated')} parsed={row.get('parsed')} "
                  f"nodes={row.get('n_nodes')} {row.get('latency_s')}s"
                  + (f"  ~{r:.1f}/min" if r else ""), flush=True)
            if args.smoke:
                print("    " + json.dumps(
                    {k: v for k, v in row.items()
                     if k not in ("key", "arm")})[:300], flush=True)
    return report(results)


def report(rows) -> int:
    rows = [r for r in rows if not r.get("error")]
    if not rows:
        print("no usable rows")
        return 2
    print(f"\n{'':<12}{'n':>4}{'truncated':>11}{'parsed':>9}"
          f"{'has_tree':>10}{'focus_ok':>10}{'tok':>7}{'sec':>7}")
    for arm in ARMS:
        a = [r for r in rows if r["arm"] == arm]
        if not a:
            continue
        fv = [r["focus_valid"] for r in a if r.get("focus_valid") is not None]
        tk = [r["completion_tokens"] for r in a if r.get("completion_tokens")]
        lt = [r["latency_s"] for r in a if r.get("latency_s")]
        print(f"{arm:<12}{len(a):>4}"
              f"{sum(r['truncated'] for r in a) / len(a):>10.0%}"
              f"{sum(r['parsed'] for r in a) / len(a):>9.0%}"
              f"{sum(r['has_tree'] for r in a) / len(a):>10.0%}"
              f"{(sum(fv) / len(fv) if fv else float('nan')):>10.0%}"
              f"{(sum(tk) // len(tk) if tk else 0):>7}"
              f"{(sum(lt) / len(lt) if lt else 0):>7.0f}")
    # PAIRED view — the only one worth acting on at this sample size.
    by = {}
    for r in rows:
        by.setdefault(r["key"], {})[r["arm"]] = r
    both = [v for v in by.values() if len(v) == len(ARMS)]
    if both:
        fixed = sum(1 for v in both
                    if v["baseline"]["truncated"] and not v["no_think"]["truncated"])
        broke = sum(1 for v in both
                    if not v["baseline"]["truncated"] and v["no_think"]["truncated"])
        pgain = sum(1 for v in both
                    if not v["baseline"]["parsed"] and v["no_think"]["parsed"])
        ploss = sum(1 for v in both
                    if v["baseline"]["parsed"] and not v["no_think"]["parsed"])
        fgain = sum(1 for v in both if v["no_think"].get("focus_valid") is True
                    and v["baseline"].get("focus_valid") is False)
        floss = sum(1 for v in both if v["baseline"].get("focus_valid") is True
                    and v["no_think"].get("focus_valid") is False)
        print(f"\nPAIRED on {len(both)} payloads (no_think vs baseline):")
        print(f"  truncation   fixed {fixed}   broke {broke}")
        print(f"  parse        gained {pgain}   lost {ploss}")
        print(f"  focus_valid  gained {fgain}   lost {floss}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
