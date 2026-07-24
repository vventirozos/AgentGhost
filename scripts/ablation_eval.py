#!/usr/bin/env python3
"""Ablation harness — does the cognitive layer have a net-plus impact?

Two ways to use it:

  * `auto`  — FULLY AUTOMATED. Boots each config (isolated GHOST_HOME + sandbox,
              test port), scores it N times, tears it down, then writes a report.
              You run one command and read the report when it finishes.

                python scripts/ablation_eval.py auto --repeats 5

  * `run` / `compare` — manual mode: you boot the agent yourself per config,
              `run` scores whatever is live, `compare` builds the table. Useful
              for one-off checks against an already-running agent.

Flags are boot-time; `auto` manages the whole agent lifecycle for you. Each
config gets a FRESH, EMPTY GHOST_HOME so the cross-session layers start blank —
which means this suite measures the IN-SESSION layers (metacog, deep-reason,
pre-flight guard, swarm). The cross-session layers need the retention protocol
(scripts/ABLATION.md, Track B). See scripts/ablation_configs.json for the matrix.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for ablation_tasks

from ghost_agent.eval.suite import EvalSuite          # noqa: E402
from ghost_agent.eval.tasks import load_default_suite  # noqa: E402


# --------------------------------------------------------------------------
# Paths / suites
# --------------------------------------------------------------------------

def _default_report_dir() -> Path:
    base = Path(os.getenv("GHOST_HOME", str(Path.home() / "ghost_llamacpp")))
    d = base / "system" / "eval" / "ablation"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_suite(name: str):
    if name == "ablation":
        from ablation_tasks import load_ablation_suite
        return load_ablation_suite()
    if name == "behavioral":
        # Execution-grounded tasks: verified against the isolated agent's real
        # sandbox / memory / DB, so a "tooluse" verdict can't be faked by text.
        from ghost_agent.eval import load_behavioral_suite
        return load_behavioral_suite()
    if name == "hard":
        from ablation_hard_tasks import load_hard_suite
        return load_hard_suite()
    if name == "default":
        return [t for t in load_default_suite() if t.category != "regression"]
    if name == "post_learning":
        from ghost_agent.eval.tasks import load_post_learning_suite
        return load_post_learning_suite()
    raise SystemExit(f"unknown suite {name!r}")


def _log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# --------------------------------------------------------------------------
# HTTP runner + scoring core
# --------------------------------------------------------------------------

def _make_http_runner(base_url: str, api_key: str, model: str, timeout: float):
    async def _runner(task, _ctx) -> Dict[str, Any]:
        import httpx
        payload = {"model": model, "stream": False,
                   "messages": [{"role": "user", "content": task.prompt}]}
        headers = {"X-Ghost-Key": api_key} if api_key else {}
        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(f"{base_url.rstrip('/')}/v1/chat/completions",
                                  json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
        dt = time.monotonic() - t0
        content = (data.get("choices", [{}])[0].get("message", {}).get("content", "")
                   or data.get("message", {}).get("content", ""))
        return {"output": str(content or ""), "duration_s": dt, "steps": 1}
    return _runner


async def _score_config(base_url: str, api_key: str, model: str, suite_name: str,
                        repeats: int, timeout: float,
                        ghost_home: Optional[str] = None) -> List[Dict[str, Any]]:
    suite_tasks = _load_suite(suite_name)
    suite = EvalSuite(f"ablation:{suite_name}", suite_tasks)
    if suite_name == "behavioral":
        # Grounded runner + a context pointed at THIS agent's GHOST_HOME (so it
        # verifies the isolated agent's own sandbox / trajectory, not the
        # driver's). The DB is external (shared), so it works per-config too.
        from ghost_agent.eval import EvalContext, agent_behavioral_runner
        gh = Path(ghost_home) if ghost_home else None
        ctx = EvalContext(
            base_url=base_url, model=model, api_key=api_key,
            ghost_home=gh, sandbox_dir=(gh / "sandbox" if gh else None),
            default_db=os.getenv("GHOST_DEFAULT_DB") or "postgresql://ghost@127.0.0.1:5432/agent",
            timeout_s=timeout)
        runner = agent_behavioral_runner()
    else:
        ctx = None
        runner = _make_http_runner(base_url, api_key, model, timeout)
    records: List[Dict[str, Any]] = []
    for rep in range(repeats):
        result = await suite.run(runner, ctx=ctx, per_task_timeout_s=timeout)
        passed = sum(1 for r in result.results if r.passed)
        tot = len(result.results)
        mean_lat = sum(r.duration_s for r in result.results) / max(tot, 1)
        _log(f"    repeat {rep+1}/{repeats}: {passed}/{tot} passed "
             f"(mean {mean_lat:.0f}s/task)")
        for r in result.results:
            records.append({
                "task_id": r.task_id, "cluster": r.cluster or "?",
                "repeat": rep, "passed": bool(r.passed),
                "duration_s": float(r.duration_s),
                "failure_reason": r.failure_reason,
            })
    return records


def _write_result(out_dir: Path, name: str, base_url: str, suite: str,
                  repeats: int, records: List[Dict[str, Any]],
                  flags: Optional[List[str]] = None, note: str = "") -> Path:
    n_tasks = len({r["task_id"] for r in records}) or 0
    out = out_dir / f"{name}.json"
    out.write_text(json.dumps({
        "config": name, "base_url": base_url, "suite": suite, "repeats": repeats,
        "n_tasks": n_tasks, "flags": flags or [], "note": note, "records": records,
    }, indent=2))
    return out


# --------------------------------------------------------------------------
# Statistics (stdlib only)
# --------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def diff_ci(k1: int, n1: int, k2: int, n2: int, z: float = 1.96) -> Tuple[float, float, float]:
    if n1 == 0 or n2 == 0:
        return 0.0, 0.0, 0.0
    p1, p2 = k1 / n1, k2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    d = p1 - p2
    return d, d - z * se, d + z * se


def mean_ci(xs: List[float], z: float = 1.96) -> Tuple[float, float, float]:
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    m = sum(xs) / n
    if n == 1:
        return m, m, m
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    se = math.sqrt(var / n)
    return m, m - z * se, m + z * se


def _stats(cfg: Dict[str, Any]) -> Dict[str, Any]:
    recs = cfg.get("records", [])
    k = sum(1 for r in recs if r["passed"])
    n = len(recs)
    p, lo, hi = wilson(k, n)
    lat, _, _ = mean_ci([r["duration_s"] for r in recs]) if recs else (0.0, 0, 0)
    return {"k": k, "n": n, "p": p, "lo": lo, "hi": hi, "lat": lat,
            "note": cfg.get("note", ""), "flags": cfg.get("flags", [])}


# --------------------------------------------------------------------------
# Report builder (markdown)
# --------------------------------------------------------------------------

def _build_report(configs: Dict[str, Dict[str, Any]], reference: str) -> str:
    if reference not in configs:
        # fall back to the highest-n config as reference
        reference = max(configs, key=lambda c: len(configs[c].get("records", [])))
    rstat = _stats(configs[reference])
    lines: List[str] = []
    lines.append("# Ablation report — Track A (in-session layers)\n")
    lines.append(f"_Generated {datetime.datetime.now().isoformat(timespec='seconds')}_  ")
    lines.append(f"_Reference config: **{reference}**_\n")
    lines.append("Each config is the FULL flag set minus one layer. Verdict is "
                 "relative to the reference, with a 95% CI on the difference.\n")

    lines.append("| config | success | 95% CI | lat/task | Δ success vs ref | verdict |")
    lines.append("|---|---|---|---|---|---|")
    order = [reference] + sorted([c for c in configs if c != reference],
                                 key=lambda c: _stats(configs[c])["p"], reverse=True)
    for name in order:
        s = _stats(configs[name])
        succ = f"{s['k']}/{s['n']} ({s['p']:.0%})" if s['n'] else "BOOT FAILED"
        ci = f"{s['lo']:.0%}–{s['hi']:.0%}" if s['n'] else "—"
        lat = f"{s['lat']:.0f}s" if s['n'] else "—"
        if name == reference:
            delta, verdict = "—", "reference"
        elif s['n'] == 0:
            delta, verdict = "—", "⚠ boot failed"
        else:
            d, dlo, dhi = diff_ci(s['k'], s['n'], rstat['k'], rstat['n'])
            delta = f"{d:+.0%} [{dlo:+.0%},{dhi:+.0%}]"
            if dlo > 0:
                verdict = "🔺 BETTER (layer is net-negative → CUT)"
            elif dhi < 0:
                verdict = "✅ WORSE (layer earns its place → KEEP)"
            else:
                verdict = "➖ indistinguishable (candidate to CUT)"
            lat_delta = s['lat'] - rstat['lat']
            verdict += f" · {lat_delta:+.0f}s/task"
        lines.append(f"| `{name}` | {succ} | {ci} | {lat} | {delta} | {verdict} |")

    lines.append("\n## How to read this\n")
    lines.append("- **WORSE than ref** → removing that layer hurt outcomes → the layer "
                 "earns its place. **KEEP.**")
    lines.append("- **indistinguishable** → removing it changed nothing measurable "
                 "*in-session*. **Candidate to CUT** — but confirm any cross-session "
                 "layer (memory/selfhood/reflection) via Track B before deleting.")
    lines.append("- **BETTER than ref** → removing it *improved* outcomes → the layer "
                 "is net-negative. **CUT.**")
    lines.append("- A negative latency Δ on an indistinguishable config means the "
                 "ablation is also cheaper — extra reason to cut.")
    lines.append("- A wide CI / 'indistinguishable' on few repeats means "
                 "**underpowered, not 'no effect'** — raise `--repeats` or grow the suite.\n")

    # per-cluster breakdown for the reference vs each config (where the effect lives)
    lines.append("## Per-cluster success (where any effect shows up)\n")
    clusters = sorted({r["cluster"] for c in configs.values() for r in c.get("records", [])})
    header = "| config | " + " | ".join(clusters) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(clusters) + 1))
    for name in order:
        recs = configs[name].get("records", [])
        cells = []
        for cl in clusters:
            cr = [r for r in recs if r["cluster"] == cl]
            if cr:
                cells.append(f"{sum(1 for r in cr if r['passed'])}/{len(cr)}")
            else:
                cells.append("—")
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")

    lines.append("\n## Caveat (do not skip)\n")
    lines.append("Cross-session layers (memory/RRF, selfhood, reflection, dream, "
                 "skills_auto, cross-project map) run on a FRESH empty GHOST_HOME here, "
                 "so they **cannot** help on single-shot tasks. If `full_no_selfmodel` / "
                 "`full_no_memory` / `full_no_reflection` look indistinguishable from "
                 "`full`, that is **expected**, not proof they're useless. Adjudicate "
                 "those with the retention protocol (Track B).")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Agent lifecycle (auto mode)
# --------------------------------------------------------------------------

def _container_name(sandbox_dir: Path) -> str:
    h = hashlib.md5(str(sandbox_dir.absolute()).encode()).hexdigest()[:8]
    return f"ghost-agent-sandbox-{h}"


def _ping(base_url: str, timeout: float = 4.0) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/version", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _wait_ready(base_url: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _ping(base_url):
            return True
        time.sleep(3)
    return False


def _wait_down(base_url: str, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _ping(base_url):
            return True
        time.sleep(2)
    return False


def _preflight(upstream_url: str, base_url: str) -> List[str]:
    problems = []
    # docker daemon
    try:
        r = subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=20)
        if r.returncode != 0:
            problems.append("docker daemon not reachable (`docker info` failed)")
    except Exception:
        problems.append("docker CLI not found / not running")
    # upstream LLM
    up_ok = False
    try:
        with urllib.request.urlopen(upstream_url.rstrip("/"), timeout=6):
            up_ok = True
    except urllib.error.HTTPError:
        up_ok = True  # connected, server answered (even 404)
    except Exception:
        up_ok = False
    if not up_ok:
        problems.append(f"upstream LLM not reachable at {upstream_url}")
    # test port must be free
    if _ping(base_url, timeout=3):
        problems.append(f"something is ALREADY listening at {base_url} — pick a free "
                        f"--port or stop it")
    return problems


def _boot(flags: List[str], ghost_home: Path, logpath: Path, extra_env: dict = None):
    env = dict(os.environ)
    env["GHOST_HOME"] = str(ghost_home)
    env["FORCE_COLOR"] = "0"
    env.setdefault("GHOST_API_KEY", "")
    # Per-arm env (e.g. an ablation arm that flips an env-gated module toggle
    # like GHOST_HYPOTHESIS_GROUNDING). Backward-compatible: default no-op.
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    cmd = [sys.executable, "-m", "src.ghost_agent.main"] + flags
    lf = open(logpath, "wb")
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env,
                            stdout=lf, stderr=lf, start_new_session=True)
    return proc, lf


def _teardown(proc, lf, container_name: str) -> None:
    if proc is not None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=25)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                proc.wait(timeout=10)
            except Exception:
                pass
    if lf is not None:
        try:
            lf.close()
        except Exception:
            pass
    # discard the throwaway container
    try:
        subprocess.run(["docker", "rm", "-f", container_name],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    except Exception:
        pass


def cmd_auto(args) -> int:
    recipe = json.loads((Path(__file__).with_name("ablation_configs.json")).read_text())
    reference = recipe.get("reference", "full")
    all_configs = recipe["configs"]
    if args.only:
        wanted = set(args.only.split(","))
        configs = [c for c in all_configs if c["name"] in wanted]
        if reference not in wanted and reference in {c["name"] for c in all_configs}:
            # always include the reference so deltas are computable
            configs = [c for c in all_configs if c["name"] == reference] + configs
    else:
        configs = all_configs

    base_url = f"http://127.0.0.1:{args.port}"
    report_dir = Path(args.report_dir) if args.report_dir else _default_report_dir()
    report_dir.mkdir(parents=True, exist_ok=True)

    _log(f"preflight checks (upstream={args.upstream_url}, test={base_url}) ...")
    problems = _preflight(args.upstream_url, base_url)
    if _ping("http://127.0.0.1:8000", timeout=3):
        _log("  NOTE: a production agent appears to be running on :8000. It shares the "
             "upstream LLM, which will slow & skew timings. Consider stopping it.")
    if problems:
        for p in problems:
            _log(f"  PREFLIGHT FAIL: {p}")
        return 2
    _log("  preflight OK")

    suite_tasks = _load_suite(args.suite)
    n_tasks = len(suite_tasks)
    est_min = len(configs) * (2.5 + args.repeats * n_tasks * 35 / 60)
    _log(f"plan: {len(configs)} configs × {args.repeats} repeats × {n_tasks} tasks "
         f"(~{est_min:.0f} min rough ETA). report -> {report_dir}")

    base_flags = ["--port", str(args.port), "--upstream-url", args.upstream_url,
                  "--api-key", "", "--no-mandatory-tor", "--model", args.model]

    collected: Dict[str, Dict[str, Any]] = {}
    for cfg in configs:
        name = cfg["name"]
        cfg_flags = [tok for f in cfg.get("flags", []) for tok in str(f).split()]
        flags = base_flags + cfg_flags
        tmp = Path(tempfile.mkdtemp(prefix=f"ghost-abl-{name}-"))
        sandbox_dir = tmp / "sandbox"
        cname = _container_name(sandbox_dir)
        logpath = report_dir / f"{name}.boot.log"
        proc = lf = None
        _log(f"=== config '{name}' === GHOST_HOME={tmp}")
        _log(f"    flags: {' '.join(cfg_flags) or '(none)'}")
        try:
            proc, lf = _boot(flags, tmp, logpath)
            _log(f"    booting (pid {proc.pid}); waiting up to {args.boot_timeout}s for ready ...")
            if not _wait_ready(base_url, args.boot_timeout):
                tail = ""
                try:
                    tail = "\n".join(Path(logpath).read_text(errors="ignore").splitlines()[-12:])
                except Exception:
                    pass
                _log(f"    BOOT FAILED for '{name}'. last log lines:\n{tail}")
                _write_result(report_dir, name, base_url, args.suite, args.repeats,
                              [], flags=cfg_flags, note="boot failed")
                collected[name] = {"config": name, "records": [], "flags": cfg_flags,
                                   "note": "boot failed"}
                continue
            _log(f"    ready. scoring {args.repeats} repeats ...")
            records = asyncio.run(_score_config(base_url, "", args.model, args.suite,
                                                args.repeats, args.timeout,
                                                ghost_home=str(tmp)))
            _write_result(report_dir, name, base_url, args.suite, args.repeats,
                          records, flags=cfg_flags, note=cfg.get("note", ""))
            collected[name] = {"config": name, "records": records,
                               "flags": cfg_flags, "note": cfg.get("note", "")}
            k = sum(1 for r in records if r["passed"])
            _log(f"    done '{name}': {k}/{len(records)} passed "
                 f"({(k/max(len(records),1)):.0%})")
        except KeyboardInterrupt:
            _log("interrupted — tearing down current agent and stopping.")
            _teardown(proc, lf, cname)
            shutil.rmtree(tmp, ignore_errors=True)
            break
        except Exception as e:
            _log(f"    ERROR on '{name}': {type(e).__name__}: {e}")
            collected[name] = {"config": name, "records": [], "flags": cfg_flags,
                               "note": f"error: {e}"}
        finally:
            _teardown(proc, lf, cname)
            shutil.rmtree(tmp, ignore_errors=True)
            _wait_down(base_url, timeout=30)

    # Build the report from everything collected (plus any prior JSONs in report_dir).
    for f in report_dir.glob("*.json"):
        try:
            d = json.loads(f.read_text())
            collected.setdefault(d["config"], d)
        except Exception:
            pass
    report_md = _build_report(collected, reference)
    (report_dir / "REPORT.md").write_text(report_md)
    (report_dir / "report.json").write_text(json.dumps(
        {"reference": reference, "generated": datetime.datetime.now().isoformat(),
         "configs": {n: _stats(c) for n, c in collected.items()}}, indent=2, default=float))
    print("\n" + "=" * 70)
    print(report_md)
    _log(f"REPORT written -> {report_dir/'REPORT.md'}")
    return 0


# --------------------------------------------------------------------------
# Manual run / compare
# --------------------------------------------------------------------------

async def cmd_run(args) -> int:
    if not _ping(args.base_url):
        print(f"!! agent not reachable at {args.base_url}", file=sys.stderr)
        return 2
    n_tasks = len(_load_suite(args.suite))
    print(f"config={args.config}  suite={args.suite}  tasks={n_tasks}  repeats={args.repeats}")
    records = await _score_config(args.base_url, args.api_key, args.model,
                                  args.suite, args.repeats, args.timeout,
                                  ghost_home=os.getenv("GHOST_HOME"))
    out = _write_result(_default_report_dir(), args.config, args.base_url,
                        args.suite, args.repeats, records)
    k = sum(1 for r in records if r["passed"]); n = len(records)
    p, lo, hi = wilson(k, n)
    print(f"\n{args.config}: {k}/{n} = {p:.1%} (95% CI {lo:.1%}–{hi:.1%})  -> {out}")
    return 0


def cmd_compare(args) -> int:
    files = sorted(_default_report_dir().glob("*.json"))
    files = [f for f in files if f.name not in ("report.json",)]
    if not files:
        print("no result files found — run some configs first.", file=sys.stderr)
        return 2
    configs = {}
    for f in files:
        try:
            d = json.loads(f.read_text())
            if "records" in d:
                configs[d["config"]] = d
        except Exception:
            pass
    report_md = _build_report(configs, args.reference)
    (_default_report_dir() / "REPORT.md").write_text(report_md)
    print(report_md)
    return 0


def cmd_matrix(_args) -> int:
    print((Path(__file__).with_name("ablation_configs.json")).read_text())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("auto", help="fully automated: boot+score+teardown every config, then report")
    pa.add_argument("--repeats", type=int, default=5)
    pa.add_argument("--suite", default="ablation", choices=("ablation", "behavioral", "hard", "default", "post_learning"))
    pa.add_argument("--port", type=int, default=8010, help="test port for the throwaway agents")
    pa.add_argument("--upstream-url", default="http://127.0.0.1:8088")
    pa.add_argument("--model", default=os.getenv("GHOST_MODEL", "qwen-3.6-35b-a3"))
    pa.add_argument("--only", default="", help="comma-separated config names to run (subset)")
    pa.add_argument("--boot-timeout", type=float, default=300.0)
    pa.add_argument("--timeout", type=float, default=300.0, help="per-task wall-clock")
    pa.add_argument("--report-dir", default="", help="where to write REPORT.md (default $GHOST_HOME/system/eval/ablation)")

    pr = sub.add_parser("run", help="manual: score an already-running agent N times")
    pr.add_argument("--config", required=True)
    pr.add_argument("--base-url", default="http://127.0.0.1:8000")
    pr.add_argument("--api-key", default=os.getenv("GHOST_API_KEY", ""))
    pr.add_argument("--model", default=os.getenv("GHOST_MODEL", "qwen-3.6-35b-a3"))
    pr.add_argument("--suite", default="ablation", choices=("ablation", "behavioral", "hard", "default", "post_learning"))
    pr.add_argument("--repeats", type=int, default=5)
    pr.add_argument("--timeout", type=float, default=300.0)

    pc = sub.add_parser("compare", help="manual: build the report from collected configs")
    pc.add_argument("--reference", default="full")

    sub.add_parser("matrix", help="print the config/boot recipe")

    args = ap.parse_args()
    if args.cmd == "auto":
        return cmd_auto(args)
    if args.cmd == "run":
        return asyncio.run(cmd_run(args))
    if args.cmd == "compare":
        return cmd_compare(args)
    if args.cmd == "matrix":
        return cmd_matrix(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
