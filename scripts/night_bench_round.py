#!/usr/bin/env python3
"""Round 3 of the verification rounds — the oracle benches, run unattended.

Everything here is a MEASUREMENT (M3): a judgment-bearing subsystem scored
against known truth, with a confidence interval where the tool gives one.
Nothing writes to the live agent's data, nothing promotes a baseline or an
artifact — the morning read is the operator's, this only prepares it.

Stages, strictly sequential (the two LLM-heavy ones must never overlap on
the main model):

  oracle    verify_bench_status — is the recorded verifier number still
            valid? (0 = valid, 1 = stale, 2 = no baseline). Decides `verify`.
  offline   read-only instruments, no LLM: escalation_audit,
            turn_state_replay, claim_binding_ledger_report,
            router_confidence_backtest, label_noise_audit,
            recheck_gepa_incumbent + gepa_live_check (expected: "nothing
            live" — that IS the measurement).
  verify    verify_bench, private tier, critic leg, escalated arm, two-stage
            on, `--cache-mode read` — the SAME topology and env as the
            2026-08-09 baseline, so the numbers compare. Runs only when the
            oracle says stale (or --force-verify). Gated by
            preflight_longrun with a real timed 1-case smoke first.
  if_bench  the banded instruction-following bench through the LIVE agent
            (probe origin — never teaches). Full bank, --repeats 2. Gated by
            preflight_longrun with the rate measured on 2026-09-14.
  summary   SUMMARY.md + status.json: the headline of every stage, tonight's
            verifier balanced accuracy against the baseline CI, per-fault
            catch rates, escalation ledger, route health (so a
            contention-degraded run is not read as a verifier regression),
            IF pass rates per band with McNemar.

Usage:
    # arm it now, run at 23:00, detached from this shell:
    PYTHONPATH=src python scripts/night_bench_round.py --launch --start-at 23:00

    # run in the foreground (or already detached):
    PYTHONPATH=src python scripts/night_bench_round.py [--start-at HH:MM]
        [--out DIR] [--only oracle,offline,verify,if_bench,summary]
        [--skip STAGES] [--force-verify] [--if-repeats 2] [--if-limit N]

    # in the morning:
    cat $GHOST_HOME/system/eval/night_rounds/<date>/SUMMARY.md
    python scripts/runstatus.py $GHOST_HOME/system/eval/night_rounds/<date>/verify_bench/progress.json

Resumable: a stage whose `<stage>.done` marker exists is skipped, so a
re-launch after a kill continues where it stopped (verify_bench itself
resumes through its response cache).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
GHOST_HOME = Path(os.environ.get("GHOST_HOME") or "/Users/vasilis/Data/AI/Data")
KEY_PATH = Path.home() / "Data" / "AI" / ".ghost_api_key"

JUDGE_URL = "http://100.83.184.117:8088"   # critic node (Nova) — the cheap judge
MAIN_URL = "http://127.0.0.1:8088"          # main model — escalation
AGENT_URL = "http://127.0.0.1:8000"

# The verify-related env the launcher exports (bin/start-ghost-agent.sh),
# so the bench process runs the SAME pipeline production does. The claim-
# binding flags are unset there too — their defaults apply in both.
PROD_VERIFY_ENV = {
    "GHOST_VERIFY_MAIN_STAGE_STOP": "1",
    "GHOST_CRITIC_ASYNC": "1",
    "GHOST_CRITIC_NO_THINK": "0",
    "GHOST_PIN_TOOL_SCHEMAS": "1",
    "GHOST_LLM_RECORD": "0",
}

STAGES = ("oracle", "offline", "verify", "if_bench", "summary")


# ── plumbing ─────────────────────────────────────────────────────────────

def _now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Round:
    def __init__(self, out: Path):
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
        self.status_path = out / "status.json"
        self.status = self._load_status()
        self.log = open(out / "round.log", "a", buffering=1)

    def _load_status(self) -> dict:
        try:
            return json.loads(self.status_path.read_text())
        except Exception:
            return {"stages": {}, "started": _now()}

    def save(self) -> None:
        tmp = self.status_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.status, indent=1, default=str))
        tmp.replace(self.status_path)

    def say(self, msg: str) -> None:
        line = f"[{_now()}] {msg}"
        print(line, flush=True)
        self.log.write(line + "\n")

    def done(self, stage: str) -> bool:
        return (self.out / f"{stage}.done").exists()

    def mark(self, stage: str, **fields) -> None:
        self.status["stages"].setdefault(stage, {}).update(fields)
        self.save()
        if fields.get("finished"):
            (self.out / f"{stage}.done").write_text(_now())

    def env(self, extra: dict | None = None) -> dict:
        e = dict(os.environ)
        e["PYTHONPATH"] = str(REPO / "src")
        e["GHOST_HOME"] = str(GHOST_HOME)
        e.pop("FORCE_COLOR", None)
        if KEY_PATH.exists():
            e["GHOST_API_KEY"] = KEY_PATH.read_text().strip()
        e.update(extra or {})
        return e

    def run(self, name: str, argv: list, log_name: str, env_extra: dict | None = None,
            timeout: float | None = None) -> tuple[int, str]:
        """Run one command with `python -u`, tee stdout+stderr to a log,
        return (rc, tail). Never raises: an instrument that cannot run is
        recorded as rc=-1 and the round continues."""
        log_path = self.out / log_name
        self.say(f"{name}: {' '.join(str(a) for a in argv)}")
        t0 = time.time()
        try:
            with open(log_path, "ab") as lf:
                lf.write(f"\n===== {_now()} {name}\n$ {' '.join(map(str, argv))}\n".encode())
                lf.flush()
                p = subprocess.run([PY, "-u", *map(str, argv)], cwd=REPO, env=self.env(env_extra),
                                   stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   timeout=timeout)
                rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = -9
        except Exception as exc:  # noqa: BLE001
            rc = -1
            with open(log_path, "ab") as lf:
                lf.write(f"LAUNCH FAILED: {exc!r}\n".encode())
        dt = time.time() - t0
        tail = ""
        try:
            tail = log_path.read_text(errors="replace")[-4000:]
        except Exception:  # noqa: BLE001
            pass
        self.say(f"{name}: rc={rc} in {dt/60:.1f} min → {log_path.name}")
        return rc, tail


# ── pure parsers (pinned in tests/test_night_bench_round.py) ─────────────

def parse_oracle(tail: str, rc: int) -> dict:
    """verify_bench_status output → verdict, drifted component names, pool size."""
    m = re.search(r"pool now: (\d+) cases", tail)
    verdict = {0: "VALID", 1: "STALE", 2: "NO_BASELINE"}.get(rc, f"rc={rc}")
    names = re.findall(r"^\s+• (\S+)\s+—", tail, re.M)
    return {"verdict": verdict, "drifted": names, "drifted_components": len(names),
            "pool_cases": int(m.group(1)) if m else None}


def parse_smoke_live_calls(tail: str):
    """(hits, live_calls) from a verify_bench run's tail. LIVE calls = cache
    misses + writes; with no cache traffic at all, the route line ("cheap
    leg: N calls") is the witness. A replayed smoke (hits only) reads 0 —
    and 0 must refuse the launch: its timing is not a measurement."""
    cm = re.search(r"cache: (\d+) hits / (\d+) misses(?: / (\d+) writes)?", tail)
    hits = int(cm.group(1)) if cm else None
    live = (int(cm.group(2)) + int(cm.group(3) or 0)) if cm else None
    rl = re.search(r"cheap leg: (\d+) calls", tail)
    if live == 0 and hits == 0 and rl:
        live = int(rl.group(1))
    return hits, live


# ── stages ───────────────────────────────────────────────────────────────

def stage_oracle(r: Round) -> None:
    argv = ["scripts/verify_bench_status.py", "--tier", "private", "--base-url", JUDGE_URL,
            "--main-base-url", MAIN_URL, "--leg", "critic"]
    rc, tail = r.run("oracle", argv, "oracle.log", PROD_VERIFY_ENV)
    o = parse_oracle(tail, rc)
    r.mark("oracle", rc=rc, finished=_now(), **o)
    r.say(f"oracle verdict: {o['verdict']} ({o['drifted_components']} drifted: {o['drifted']}, pool={o['pool_cases']})")


def stage_offline(r: Round) -> None:
    jobs = [
        ("escalation_audit", ["scripts/escalation_audit.py", "--limit", "40"], {}),
        ("escalation_audit_json", ["scripts/escalation_audit.py", "--json"], {}),
        ("turn_state_replay", ["scripts/turn_state_replay.py", "--brief", "--days", "30"], {}),
        ("claim_binding_ledger", ["scripts/claim_binding_ledger_report.py", "--days", "7"], {}),
        ("router_backtest", ["scripts/router_confidence_backtest.py"], {}),
        ("router_backtest_json", ["scripts/router_confidence_backtest.py", "--json"], {}),
        ("label_noise_audit", ["scripts/label_noise_audit.py"], {}),
        ("gepa_recheck", ["scripts/recheck_gepa_incumbent.py"], {}),
        ("gepa_live_check", ["scripts/gepa_live_check.py"], {}),
    ]
    results = {}
    for name, argv, extra in jobs:
        rc, tail = r.run(name, argv, f"offline_{name}.log", extra, timeout=1800)
        results[name] = {"rc": rc, "log": f"offline_{name}.log"}
        if name.endswith("_json"):
            # keep the machine-readable body next to the log
            body = tail
            try:
                body = (r.out / f"offline_{name}.log").read_text(errors="replace")
                body = body[body.rfind("\n$ ") + 1:]
                body = body[body.find("\n") + 1:]
                (r.out / f"{name}.json").write_text(body)
            except Exception:  # noqa: BLE001
                pass
    r.mark("offline", results=results, finished=_now())


def _preflight(r: Round, name: str, observable: str, total: int, resumable: str,
               smoke: str, rate: float, rate_source: str) -> bool:
    argv = ["scripts/preflight_longrun.py", "--name", name, "--observable", observable,
            "--total-from-tool", str(total), "--resumable", resumable, "--smoke", smoke,
            "--measured-rate", f"{rate:.3f}", "--rate-source", rate_source]
    rc, _ = r.run(f"preflight[{name}]", argv, "preflight.log")
    return rc == 0


def stage_verify(r: Round, force: bool, smoke_only: bool = False) -> None:
    oracle = r.status["stages"].get("oracle", {})
    if not force and oracle.get("verdict") == "VALID":
        r.mark("verify", skipped="oracle says the baseline is still valid", finished=_now())
        r.say("verify: skipped — baseline VALID")
        return
    out_dir = r.out / "verify_bench"
    out_dir.mkdir(exist_ok=True)
    progress = out_dir / "progress.json"
    base = ["scripts/verify_bench.py", "--base-url", JUDGE_URL, "--main-base-url", MAIN_URL,
            "--two-stage", "on", "--leg", "critic", "--tier", "private", "--seed", "0",
            "--cache-mode", "read", "--out", str(out_dir), "--progress-file", str(progress)]
    # 1) a real timed smoke: ONE case, `--cache-mode write` so every call is
    #    LIVE (a replayed smoke measured 3 s/case in rehearsal and would have
    #    "cleared" a 2.4 h run with a 4-minute ETA — the derived number the
    #    MEASURED gate exists to forbid). Its responses land in the cache, so
    #    the full `read` run replays them — nothing paid twice.
    smoke_argv = [a if a != "read" else "write" for a in base] + ["--max-cases", "1"]
    t0 = time.time()
    rc, tail = r.run("verify-smoke", smoke_argv, "verify_smoke.log", PROD_VERIFY_ENV, timeout=1800)
    smoke_s = time.time() - t0
    # write mode prints "cache: H hits / M misses / W writes — live judge";
    # LIVE calls = misses + writes (a replayed smoke has hits only). The
    # route line ("cheap leg: N calls") is the second witness.
    hits, misses = parse_smoke_live_calls(tail)
    n_trials = None
    for cand in sorted(out_dir.glob("*/results.json"), key=lambda p: p.stat().st_mtime):
        n_trials = (_load(cand) or {}).get("n_trials")
    pool = oracle.get("pool_cases") or 58
    per_case_min = smoke_s / 60.0
    rate_cases_per_min = 1.0 / per_case_min if per_case_min > 0 else 0.0
    eta_lo, eta_hi = pool * per_case_min * 0.8, pool * per_case_min * 1.5
    r.mark("verify", smoke={"rc": rc, "seconds": round(smoke_s), "trials": n_trials,
                            "cache_hits": hits, "cache_misses": misses,
                            "eta_minutes": [round(eta_lo), round(eta_hi)]})
    r.say(f"verify smoke: 1 case ({n_trials} trials, {misses} live calls) in {smoke_s/60:.1f} min "
          f"→ ETA range {eta_lo/60:.1f}–{eta_hi/60:.1f} h for {pool} cases")
    if rc != 0:
        r.mark("verify", error="smoke failed — not launching the full run", finished=_now())
        return
    if not misses:
        r.mark("verify", error="smoke was NOT live (0 cache misses) — its rate is not a measurement; refusing to launch on it",
               finished=_now())
        return
    # 2) the gate, with what the smoke measured
    ok = _preflight(r, "verifier re-bench (night round)",
                    f"progress-file:{progress}", pool,
                    "verify_bench response cache (--cache-mode read): a kill loses nothing already answered",
                    f"oracle: {oracle.get('drifted_components')} components drifted incl. 3 rendered prompts — "
                    f"outputs differ from the baseline by construction; smoke ran {n_trials or '?'} trials, {misses} live calls",
                    rate_cases_per_min, f"timed 1-case smoke tonight ({smoke_s:.0f}s)")
    if not ok:
        r.mark("verify", error="preflight blocked the launch (see preflight.log)", finished=_now())
        return
    if smoke_only:
        r.say("verify: --verify-smoke-only — cleared for launch, not launching")
        r.mark("verify", smoke_only=True)
        return
    # 3) the run — then RE-PASS while escalations came back `unavailable`.
    #    The main model has ONE slot (`-np 1`); at night a self-play turn
    #    can hold it past the 90 s escalation bound, and the verifier then
    #    lets the cheap verdict stand (`unavailable`) — the trial measured
    #    the raw judge, not the pipeline. An unavailable call leaves no
    #    cache entry, so a `read` pass replays everything that answered and
    #    re-asks exactly those escalations. Measured 2026-09-20 (dead-main
    #    smoke: 16 hits / 4 misses / 0 writes; the re-pass: 20 hits).
    passes = []
    results = None
    for i in range(1, 5):
        rc, tail = r.run(f"verify pass {i}", base, "verify_bench.log", PROD_VERIFY_ENV,
                         timeout=6 * 3600)
        results = None
        for cand in sorted(out_dir.glob("*/results.json"), key=lambda p: p.stat().st_mtime):
            results = cand
        unavailable = None
        if results:
            res = _load(results) or {}
            rh = ((res.get("provenance") or {}).get("escalation") or {}).get("route_health") or {}
            unavailable = rh.get("escalation_unavailable")
        passes.append({"pass": i, "rc": rc, "results": str(results) if results else None,
                       "escalation_unavailable": unavailable})
        r.mark("verify", passes=passes)
        r.say(f"verify pass {i}: rc={rc} escalation_unavailable={unavailable}")
        if rc != 0 or not unavailable:
            break
    r.mark("verify", rc=rc, results=str(results) if results else None,
           escalation_unavailable_final=unavailable, finished=_now())


def _agent_up() -> bool:
    try:
        import urllib.request
        req = urllib.request.Request(AGENT_URL + "/api/health",
                                     headers={"X-Ghost-Key": KEY_PATH.read_text().strip()})
        with urllib.request.urlopen(req, timeout=10) as resp:
            d = json.loads(resp.read())
        return bool(d.get("memory_system_loaded")) and bool(d.get("biological_watchdog_alive"))
    except Exception:  # noqa: BLE001
        return False


def stage_if_bench(r: Round, repeats: int, limit: int) -> None:
    if not _agent_up():
        r.mark("if_bench", error="live agent not healthy on :8000 — skipped", finished=_now())
        r.say("if_bench: agent not healthy — skipped")
        return
    out_dir = r.out / "if_bench"
    out_dir.mkdir(exist_ok=True)
    n_items = 61 if not limit else min(limit, 61)
    total_turns = n_items * repeats * 2
    # rate measured 2026-09-14: 60 turns in 2791 s (§4GJ run) = 1.29 turns/min
    ok = _preflight(r, "if_bench full bank (night round)",
                    "unbuffered", total_turns,
                    "per-turn ledger rows are appended as they land; --offset/--items resume a chunk",
                    "2026-09-14 banded run: control 0.77 vs compiled 0.60 on 15 items — the arms differ",
                    1.29, "2026-09-14T183729Z summary: 60 turns / 2791 s")
    if not ok:
        r.mark("if_bench", error="preflight blocked the launch (see preflight.log)", finished=_now())
        return
    argv = ["scripts/if_bench.py", "--repeats", str(repeats), "--out", str(out_dir)]
    if limit:
        argv += ["--limit", str(limit)]
    rc, _ = r.run("if_bench", argv, "if_bench.log", timeout=7 * 3600)
    summ = sorted(out_dir.glob("*.summary.json"), key=lambda p: p.stat().st_mtime)
    r.mark("if_bench", rc=rc, summary=str(summ[-1]) if summ else None, finished=_now())


# ── summary ──────────────────────────────────────────────────────────────

def _load(p) -> dict | None:
    try:
        return json.loads(Path(p).read_text())
    except Exception:  # noqa: BLE001
        return None


def _grep(log: Path, pattern: str, n: int = 6) -> list:
    try:
        txt = log.read_text(errors="replace")
    except Exception:  # noqa: BLE001
        return []
    return re.findall(pattern, txt, re.M)[:n]


def _balanced(res: dict):
    """§4T balanced = 0.5·mean(non-refute correct) + 0.5·mean(refute correct),
    computed with verify_bench_compare's OWN `_correct`/`_trials` so this
    file cannot drift from the tool the morning read relies on."""
    try:
        sys.path.insert(0, str(REPO / "scripts"))
        import verify_bench_compare as vbc  # noqa: E402
        trials = vbc._trials(res, "two_stage_on")
        keys = list(trials)
        nr, n_nr = vbc._rate(trials, keys, want_non_refute=True)
        rf, n_rf = vbc._rate(trials, keys, want_non_refute=False)
        if nr is None or rf is None:
            return None, n_nr, n_rf
        return round(0.5 * nr + 0.5 * rf, 4), n_nr, n_rf
    except Exception as exc:  # noqa: BLE001
        return f"n/a ({exc!r})", 0, 0


def stage_summary(r: Round) -> None:
    st = r.status["stages"]
    lines = [f"# Night bench round — {r.status.get('started')} → {_now()}", ""]
    # oracle
    o = st.get("oracle", {})
    lines += ["## Verifier bench oracle",
              f"- verdict: **{o.get('verdict')}** ({o.get('drifted_components')} drifted components, pool {o.get('pool_cases')} cases) — `oracle.log`", ""]
    # verify
    v = st.get("verify", {})
    lines += ["## Verifier re-bench (private tier, critic leg, escalated arm, two-stage on)"]
    base = _load(GHOST_HOME / "system/eval/verifier_incumbent_baseline.json") or {}
    if v.get("results"):
        res = _load(v["results"]) or {}
        arm = (res.get("arms") or {}).get("two_stage_on") or {}
        met = arm.get("metrics") or {}
        bal, n_nr, n_rf = _balanced(res)
        b_bal, b_ci = base.get("private_incumbent_balanced"), base.get("ci95")
        lines.append(f"- trials {res.get('n_trials')} / cases {res.get('n_cases')} — `{v['results']}`")
        lines.append(f"- balanced tonight (§4T formula, via verify_bench_compare's helpers): **{bal}** "
                     f"(n={n_nr} non-refute / {n_rf} refute)   |   baseline 2026-08-09: {b_bal} {b_ci}")
        if isinstance(bal, (int, float)) and isinstance(b_ci, list) and len(b_ci) == 2:
            where = ("INSIDE the baseline CI — no detectable change at this resolution" if b_ci[0] <= bal <= b_ci[1]
                     else ("ABOVE the baseline CI" if bal > b_ci[1] else "**BELOW the baseline CI — investigate before anything else**"))
            lines.append(f"- reading (unpaired, ±0.04 resolution): {where}")
        # the PAIRED comparison — the one that can resolve a real change
        old_res = base.get("results_path")
        drifted = (st.get("oracle") or {}).get("drifted") or []
        if old_res and Path(old_res).exists():
            argv = ["scripts/verify_bench_compare.py", old_res, v["results"]]
            for d in drifted:
                argv += ["--expect-differs", d]
            rc, tail = r.run("compare", argv, "compare.log")
            keep = [ln for ln in tail.splitlines() if ln.strip()][-14:]
            lines.append(f"- paired comparison vs baseline (`compare.log`, rc={rc}, expect-differs={drifted}):")
            for ln in keep:
                lines.append(f"      {ln[:150]}")
        pf = met.get("per_fault") or {}
        if pf:
            lines.append("- per-fault catch rate (refuted/judged):")
            for k, d in sorted(pf.items()):
                j = d.get("judged") or 0
                lines.append(f"    - {k}: {d.get('refuted')}/{j}" + (f" = {d.get('refuted', 0)/j:.3f}" if j else ""))
        esc = (res.get("provenance") or {}).get("escalation") or {}
        rh = esc.get("route_health") or {}
        unavail = rh.get("escalation_unavailable")
        lines.append(f"- route health: {rh}")
        lines.append(f"- passes: {[(p['pass'], p['escalation_unavailable']) for p in (v.get('passes') or [])]}"
                     " (pass, escalations the strong model never answered — each re-pass re-asks only those)")
        if unavail:
            lines.append(f"- **{unavail} escalation(s) still UNAVAILABLE after the last pass — those trials measured the "
                         "raw cheap judge. The balanced number is contaminated in BOTH directions; do not compare, "
                         "do not adopt. Re-run `--only verify` (read mode) when the main model is quiet.**")
        elif not rh.get("clean"):
            lines.append("- **route health not clean (cheap leg fell through to main) — those trials did not measure the judge under test.**")
        else:
            lines.append("- route health clean: every escalation was answered by the strong model — the number is the pipeline's.")
        led = met.get("escalation") or met.get("escalation_ledger") or {}
        if led:
            lines.append(f"- escalation ledger: {led}")
        if rh.get("clean"):
            lines.append("- to adopt as the new baseline (operator's call): record `results.json` the way §4T did "
                         "(`system/eval/verifier_incumbent_baseline.json`, provenance + CI). Only a clean run qualifies.")
    else:
        lines.append(f"- {v.get('skipped') or v.get('error') or 'not run'}")
    lines.append("")
    # offline instruments — headline greps
    lines += ["## Offline instruments (read-only)"]
    off = (st.get("offline") or {}).get("results") or {}
    # one headline pattern per instrument, taken from what each prints
    HEADLINES = {
        "escalation_audit": r"^ESCALATION AUDIT — .*$|^.*→ OVERTURNED.*$",
        "turn_state_replay": r"^\d+ real user turns;.*$|^\d+ fire\(s\) on a passed.*$",
        "claim_binding_ledger": r"^claim-binding ledger: .*$|^DISAGREEMENTS the incumbent decided: .*$|^escalation ledger, same window: .*$",
        "router_backtest": r"^\s*corpus: .*$|^\s*spread across .*$|^\s*VERDICT: .*$",
        "label_noise_audit": r"^\s*VERIFIER-decided\s*:.*$|^\s*noise-FREE .*$|^\s*replicates where the kill list DIFFERS.*$|^\s*mean share of the victim set decided by noise.*$",
        "gepa_recheck": r"^.*no live artifact.*$|^.*VERDICT.*$",
        "gepa_live_check": r"^(?:KEEP|REVERT|CONFOUNDED|INSUFFICIENT).*$|^.*NO LIVE ARTIFACT.*$",
    }
    for name, pat in HEADLINES.items():
        d = off.get(name) or {}
        log = r.out / f"offline_{name}.log"
        head = _grep(log, pat, 5)
        if name == "escalation_audit":
            n_over = len(_grep(log, r"→ OVERTURNED", 10_000))
            head = head[:1] + [f"{n_over} OVERTURNED card(s) printed — each carries a 'VERDICT (fill in)' line: an operator or judge pass scores them (`escalation_audit_json.json` for the judge)"]
        lines.append(f"- **{name}** rc={d.get('rc')} — `{log.name}`")
        for h in head:
            lines.append(f"    - {h.strip()[:170]}")
    lines.append("")
    # if_bench
    ib = st.get("if_bench", {})
    lines += ["## Instruction-following bench (live agent, probe origin)"]
    if ib.get("summary"):
        s = _load(ib["summary"]) or {}
        lines.append(f"- items {s.get('items')} × repeats {s.get('repeats')} → pairs {s.get('pairs')}, {round((s.get('seconds') or 0)/60)} min, errors {s.get('errors')}")
        lines.append(f"- pass rate: {s.get('pass_rate')}")
        lines.append(f"- by band: {json.dumps(s.get('by_band'))[:600]}")
        lines.append(f"- McNemar: {json.dumps(s.get('mcnemar'))[:400]}")
        lines.append(f"- narration beats: {s.get('narration')}  tool-syntax leaks: {s.get('tool_syntax_leak')}")
    else:
        lines.append(f"- {ib.get('error') or 'not run'}")
    lines.append("")
    lines += ["## Not run tonight (by design)",
              "- `dream_replay_validate.py` — forks sandboxes for up to 8 h to certify an engine that is gated OFF in production (`GHOST_DREAM_REPLAY`); run it the day the engine is turned on.",
              "- the sandbox PROMOTION path — needs a >600 s command through `execute`; a daytime probe.", ""]
    (r.out / "SUMMARY.md").write_text("\n".join(lines))
    r.mark("summary", finished=_now())
    r.say(f"summary written → {r.out / 'SUMMARY.md'}")


# ── main ─────────────────────────────────────────────────────────────────

def _sleep_until(hhmm: str, say) -> None:
    h, m = (int(x) for x in hhmm.split(":"))
    now = _dt.datetime.now()
    target = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if target <= now:
        target += _dt.timedelta(days=1)
    say(f"armed — sleeping until {target:%Y-%m-%d %H:%M} ({(target-now).total_seconds()/3600:.1f} h)")
    while _dt.datetime.now() < target:
        time.sleep(min(60, max(1, (target - _dt.datetime.now()).total_seconds())))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="")
    ap.add_argument("--start-at", default="", help="HH:MM local; sleep until then")
    ap.add_argument("--only", default="", help="comma-separated stages")
    ap.add_argument("--skip", default="", help="comma-separated stages")
    ap.add_argument("--force-verify", action="store_true", help="re-bench even if the oracle says VALID")
    ap.add_argument("--verify-smoke-only", action="store_true",
                    help="run the verify stage's timed smoke + preflight, then stop (a rehearsal)")
    ap.add_argument("--if-repeats", type=int, default=2)
    ap.add_argument("--if-limit", type=int, default=0)
    ap.add_argument("--launch", action="store_true",
                    help="re-exec this command detached (own session, nohup-safe) and return")
    args = ap.parse_args()

    out = Path(args.out) if args.out else (GHOST_HOME / "system" / "eval" / "night_rounds" /
                                           _dt.datetime.now().strftime("%Y-%m-%d"))
    out.mkdir(parents=True, exist_ok=True)

    if args.launch:
        argv = [a for a in sys.argv[1:] if a != "--launch"]
        if not args.out:
            argv += ["--out", str(out)]
        log = open(out / "launcher.log", "ab")
        p = subprocess.Popen([PY, "-u", str(Path(__file__).resolve()), *argv], cwd=REPO,
                             env={**os.environ, "PYTHONPATH": str(REPO / "src"), "GHOST_HOME": str(GHOST_HOME)},
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
        print(f"launched pid={p.pid} (own session) → {out}\n"
              f"  progress: cat {out}/status.json ; tail -f {out}/round.log\n"
              f"  verify:   python scripts/runstatus.py {out}/verify_bench/progress.json\n"
              f"  morning:  cat {out}/SUMMARY.md")
        return 0

    r = Round(out)
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    bad = (only | skip) - set(STAGES)
    if bad:
        r.say(f"unknown stage(s) {sorted(bad)}; valid: {STAGES}")
        return 2
    if args.start_at:
        _sleep_until(args.start_at, r.say)
    r.say(f"round start; out={out}")

    def want(s: str) -> bool:
        if only and s not in only:
            return False
        if s in skip:
            return False
        if r.done(s) and s != "summary":
            r.say(f"{s}: already done (marker present) — skipped")
            return False
        return True

    if want("oracle"):
        stage_oracle(r)
    if want("offline"):
        stage_offline(r)
    if want("verify"):
        stage_verify(r, args.force_verify, args.verify_smoke_only)
    if want("if_bench"):
        stage_if_bench(r, args.if_repeats, args.if_limit)
    if want("summary"):
        stage_summary(r)
    r.say("round finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
