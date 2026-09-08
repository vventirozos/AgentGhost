#!/usr/bin/env python3
"""§4FG — coding-leaf bench: spec executor vs agentic edit-test loop.

Two scratch CODING projects through the live API — one with
`metadata.executor = "agentic"`, one default (spec) — receive the SAME small,
verifiable leaves; each leaf is driven with `POST /api/projects/{pid}/advance`
(one leaf per call, the real executor, the real gates), and its task status
(DONE / FAILED) plus wall time is recorded. Paired per leaf → exact McNemar.
Both projects are hard-deleted at the end (their workspaces too).

Runs on the live agent's main slot; sequential; each leaf may take minutes.
Replicate ≥3× — agentic evals vary run to run even at temperature 0
(arXiv 2602.07150). Default: 6 leaves × 2 executors × 1 repeat as a pilot.

usage: leaf_bench.py [--suite small|app] [--repeats 1] [--limit N] [--keep] [--out DIR]

§4FI: `--suite app` is the DECIDING bench — the §4EI shape: one single-file
Flask app (`app.py`) grown over six dependent leaves, each adding routes and
tests to the same file, so the spec executor must re-emit an ever larger
file whole (the shape that hit the think ceiling and the `files`-boundary
slips) while the loop edits in place. Leaves depend on each other, so run
whole repeats (one invocation per repeat keeps a kill cheap). Progress:
`<ledger>.progress.json` (RunProgress; read with scripts/runstatus.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from ghost_agent.eval.runprogress import RunProgress           # noqa: E402

GHOST_HOME = Path(os.environ.get("GHOST_HOME", "/Users/vasilis/Data/AI/Data"))
AGENT = os.getenv("GHOST_AGENT_URL", "http://127.0.0.1:8000")
KEY = os.getenv("GHOST_API_KEY") or (Path.home() / "Data/AI/.ghost_api_key").read_text().strip()

LEAVES = [
    ("fib", "Create fib.py with an iterative fib(n) returning the nth Fibonacci number (fib(0)=0, fib(1)=1), "
            "and tests/test_fib.py with pytest tests for n=0,1,2,10. Run the tests."),
    ("wc", "Create wc.py: a CLI that prints the number of lines in the file given as argv[1]. "
           "Add tests/test_wc.py that writes a 3-line temp file and asserts the output is 3 (use subprocess). Run the tests."),
    ("slug", "Create utils/slug.py with slugify(text) that lowercases, replaces spaces and non-alphanumerics with single "
             "hyphens, and strips leading/trailing hyphens. Add tests/test_slug.py covering 'Hello World!' -> 'hello-world' "
             "and double spaces. Run the tests."),
    ("health", "Create app.py with a Flask app exposing GET /health returning JSON {\"status\": \"ok\"}. "
               "Add tests/test_app.py using app.test_client() to assert 200 and the body. Install flask with pip if missing. Run the tests."),
    ("j2c", "Create j2c.py that converts a JSON array of flat objects (argv[1]) to CSV on stdout with a header row from the "
            "union of keys (sorted). Add tests/test_j2c.py with a two-row example. Run the tests."),
    ("stats", "Create stats.py with mean(xs) and median(xs) (median of an even list is the average of the middle two). "
              "Add tests/test_stats.py with three cases each. Run the tests."),
]


_ONE_FILE = ("All application code stays in the single file app.py — do not create other modules; "
             "tests live in tests/test_app.py. ")
APP_LEAVES = [
    ("skel", "Create app.py: a Flask app holding an in-memory list of expense records "
             "(fields: id, date 'YYYY-MM-DD', category, amount, note). GET /health returns {\"status\": \"ok\"}; "
             "GET /api/expenses returns the list; POST /api/expenses creates one (amount must be a number > 0, "
             "date must match YYYY-MM-DD; otherwise 400) and returns 201 with the record. "
             "Create tests/test_app.py using app.test_client(): health, empty list, create-then-list, invalid amount -> 400. "
             "Install flask with pip if missing. " + _ONE_FILE + "Run the tests."),
    ("crud", "Extend app.py: PUT /api/expenses/<id> updates any of date/category/amount/note with the same validation "
             "and returns the record; DELETE /api/expenses/<id> returns 204; an unknown id returns 404 for both. "
             "Add tests for update, delete, and 404 to tests/test_app.py. " + _ONE_FILE + "Run the tests."),
    ("page", "Extend app.py: GET / returns an inline HTML page (a Python string rendered with render_template_string) "
             "with a form (date, category, amount, note) that POSTs to /api/expenses via fetch, a table inside "
             "<section data-section=\"expenses\"> filled by fetching /api/expenses, and a total inside "
             "<section data-section=\"summary\">. Add tests: GET / is 200 text/html and contains both data-section "
             "markers and the string '/api/expenses'. " + _ONE_FILE + "Run the tests."),
    ("summary", "Extend app.py: GET /api/summary returns {\"count\": n, \"total\": sum, \"by_category\": {category: sum}}; "
                "an optional ?month=YYYY-MM query filters BOTH /api/expenses and /api/summary by date prefix. "
                "Add tests with three records over two months and two categories. " + _ONE_FILE + "Run the tests."),
    ("csv", "Extend app.py: GET /api/expenses.csv returns text/csv with header id,date,category,amount,note "
            "(honouring ?month=); POST /api/import with a text/csv body (same header) creates the rows and returns "
            "{\"imported\": n}, or 400 naming the first malformed row. Add a round-trip test (create two, export, "
            "delete both, import the export, list shows two). " + _ONE_FILE + "Run the tests."),
    ("persist", "Extend app.py: add create_app(storage_path=None) that loads records from a JSON file at storage_path "
                "on creation and saves after every mutation; keep module-level app = create_app() so existing tests "
                "still pass. Add tests using tmp_path: create two records through one app, build a second app on the "
                "same path, assert both records are listed. " + _ONE_FILE + "Run the tests."),
]
SUITES = {"small": LEAVES, "app": APP_LEAVES}
GOALS = {"small": "leaf bench: small verifiable python leaves",
         "app": "leaf bench: one single-file Flask expense app grown over dependent leaves"}


def _leaf_set(suite: str, limit: int = 0, leaves_csv: str = ""):
    """The leaves one invocation runs, in order. Unknown suite → ValueError."""
    if suite not in SUITES:
        raise ValueError(f"unknown suite {suite!r}; choose from {sorted(SUITES)}")
    leaves = list(SUITES[suite])
    if limit:
        leaves = leaves[:limit]
    if leaves_csv:
        want = {x.strip() for x in leaves_csv.split(",") if x.strip()}
        leaves = [l for l in leaves if l[0] in want]
    return leaves


def _req(method, path, body=None, timeout=900):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{AGENT}{path}", data=data, method=method,
                                 headers={"Content-Type": "application/json", "X-Ghost-Key": KEY})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n))


def run_arm(kind: str, leaves, stamp: str, ledger, goal: str = GOALS["small"], progress=None, rep: int = 0):
    # explicit on BOTH arms: since §4FL the default routes growing leaves to
    # the loop by shape, so the control arm must pin the spec executor
    meta = {"executor": "agentic"} if kind == "agentic" else {"executor": "spec"}
    proj = _req("POST", "/api/projects", {"title": f"leafbench-{kind}-{stamp}", "kind": "CODING",
                                          "goal": goal, "metadata": meta})
    pid = proj["id"]
    results = {}
    try:
        for lid, desc in leaves:
            t = _req("POST", f"/api/projects/{pid}/tasks", {"description": desc})
            tid = t["id"]
            t0 = time.time()
            try:
                adv = _req("POST", f"/api/projects/{pid}/advance", {})
            except Exception as e:  # noqa: BLE001
                adv = {"ok": False, "summary": f"advance error: {e}"}
            dt = round(time.time() - t0, 1)
            task = _req("GET", f"/api/projects/{pid}/tasks").get("tasks", [])
            status = next((x.get("status") for x in task if x.get("id") == tid), "?")
            done = str(status).upper() == "DONE"
            results[lid] = done
            rec = {"kind": kind, "rep": rep, "leaf": lid, "pid": pid, "task_id": tid, "status": status,
                   "done": done, "seconds": dt, "summary": str(adv.get("summary"))[:300],
                   "classification": adv.get("classification")}
            ledger.write(json.dumps(rec) + "\n"); ledger.flush()
            if progress is not None:
                progress.tick(extra={"kind": kind, "leaf": lid, "status": status})
            print(f"[{kind:<7} {lid:<7}] {status:<8} {dt:6.1f}s  {str(adv.get('summary'))[:80]!r}", flush=True)
    finally:
        pass
    return pid, results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="small", choices=sorted(SUITES))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--leaves", default="", help="comma-separated leaf ids to run (chunked runs)")
    ap.add_argument("--keep", action="store_true", help="do not delete the scratch projects")
    ap.add_argument("--out", default=str(GHOST_HOME / "system" / "eval" / "leaf_bench"))
    args = ap.parse_args(argv)
    leaves = _leaf_set(args.suite, args.limit, args.leaves)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    ledger_path = out / f"{stamp}.jsonl"
    done = {"spec": 0, "agentic": 0}
    b = c = 0
    pids = []
    t_start = time.time()
    total = len(leaves) * 2 * args.repeats
    # BOUNDED: the denominator is printed by the tool.
    print(f"suite={args.suite} leaves={len(leaves)} × 2 arms × {args.repeats} repeats = {total} leaf runs",
          flush=True)
    progress = RunProgress(ledger_path.with_suffix(".progress.json"), total,
                           label=f"leaf_bench {args.suite} {stamp}")
    with ledger_path.open("w") as ledger:
        for rep in range(args.repeats):
            res = {}
            for kind in ("spec", "agentic"):
                pid, r = run_arm(kind, leaves, f"{stamp}-r{rep}", ledger,
                                 goal=GOALS[args.suite], progress=progress, rep=rep)
                pids.append(pid); res[kind] = r
                done[kind] += sum(1 for v in r.values() if v)
            for lid, _ in leaves:
                s, a = res["spec"].get(lid), res["agentic"].get(lid)
                if s and not a: b += 1
                if a and not s: c += 1
    n = len(leaves) * args.repeats
    progress.finish(f"pairs={n}")
    summary = {"suite": args.suite, "leaves": len(leaves), "repeats": args.repeats, "pairs": n,
               "done_rate": {k: v / n for k, v in done.items()} if n else {},
               "mcnemar": {"b_spec_only": b, "c_agentic_only": c, "p": mcnemar_exact(b, c)},
               "seconds": round(time.time() - t_start), "ledger": str(ledger_path), "projects": pids}
    (out / f"{stamp}.summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    if not args.keep:
        for pid in pids:
            try:
                _req("DELETE", f"/api/projects/{pid}?hard=true")
            except Exception as e:  # noqa: BLE001
                print(f"delete {pid} failed: {e}")


if __name__ == "__main__":
    sys.exit(main())
