#!/usr/bin/env python3
"""§4FE — tool-selection accuracy: full advertised set vs the head diet.

Replays the mined tool-choice fixtures (`$GHOST_HOME/system/optim/
tool_choice_fixtures.jsonl*`) as single-turn prompts against the live
llama-server: the live SYSTEM_PROMPT + the fixture's user request, once with
every static schema advertised, once with the diet (core + tool_catalog).
Temperature 0, first tool the model picks vs the recorded first choice.

⚠ Proxy, stated plainly: the recorded payloads (history, hydration) were
archived with GHOST_LLM_RECORD, so this replays the REQUEST TEXT, not the
recorded context. It measures whether the request alone still routes to the
same tool under the diet — the dominant signal, not the whole one.

Scoring under the diet: correct if the model picks the recorded tool, or —
when the recorded tool is hidden — picks `tool_catalog` (the designed path).
Paired per fixture → exact McNemar. Sequential on the main slot (politeness).

usage: tool_head_diet_bench.py [--limit N|0=all] [--seed S] [--out DIR] [--resume LEDGER]
                               [--pair A,B]   heads: full | diet | full-legacy-workspace

§4FK (instrument review): the UNIT OF ANALYSIS is the distinct request, not
the fixture row — the corpus holds 587 rows but only 264 distinct requests
(one request 40×), and 142 repeated requests carry conflicting recorded
first tools. `--unit request` (default) collapses rows by request text with
the majority recorded tool among PASSED turns as the truth (all turns if
none passed); `--unit row` is the pre-§4FK behaviour, kept for comparison.
The summary is stratified by recorded outcome (passed/failed turns) and by
origin (bench/user_request), the corpus status is named (`.notready` = a
parked mine whose supply gates were NOT met), and seed/limit/unit are
recorded. Resume refuses a different pair, and a resume that selects
nothing to run is an error, not a finished bench.

§4FJ: `--pair full-legacy-workspace,full` measures ONE description change —
the `workspace` tool as advertised before §4FJ (kept verbatim below as a fixed
head) against the live head — on the same fixtures, paired. Only `diet` gets
the tool_catalog credit; every other head is scored on the exact pick.

§4FI: the full run is ~90 min, so it is RESUMABLE (`--resume <ledger.jsonl>`
skips the fixture ids already in that ledger and appends to it) and
OBSERVABLE (`<ledger>.progress.json` under the RunProgress contract — read
it with `scripts/runstatus.py`, never derive progress from the log).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.request
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("GHOST_HOME", "/Users/vasilis/Data/AI/Data")

from ghost_agent.core.prompts import SYSTEM_PROMPT            # noqa: E402
from ghost_agent.eval.runprogress import RunProgress           # noqa: E402
from ghost_agent.tools import registry as R                    # noqa: E402

UPSTREAM = os.getenv("GHOST_UPSTREAM", "http://127.0.0.1:8088/v1/chat/completions")
GHOST_HOME = Path(os.environ["GHOST_HOME"])


# The `workspace` description as advertised up to 2026-09-07 (§4FJ replaced it):
# on the 587-fixture replay the full head picked `workspace` 98 times with the
# truth being `workspace` once — 60 of those were file_system requests.
WORKSPACE_DESCRIPTION_BEFORE_4FJ = (
    "READ-ONLY view of the user's WORKSPACE state — what's outside of you (files, "
    "scheduled-task outcomes, research artifacts you've pulled, commands you ran). This is "
    "the world-model counterpart to introspect (which reads your selfhood). Use this when "
    "the user asks 'what changed since yesterday?', 'what did my scheduled task do?', 'have "
    "I already pulled this URL?', 'show me what you've been doing in my project'. Distinct "
    "from: introspect (your own selfhood), file_system (one-shot reads of the filesystem), "
    "recall (vector search over ingested docs). Actions: 'summary' (default; stats + "
    "narrative + recent changes + recent tasks/research); 'stats' (counts); 'files' (the "
    "watchlist); 'changes' (diff tracked files against last-seen snapshot); 'tasks' (recent "
    "scheduled-task outcomes); 'research' (URLs you've already pulled); 'commands' "
    "(significant command outcomes); 'narrative' (the running workspace summary); 'recent' "
    "(the activity log, mixed kinds); 'search' (keyword search over the activity log — pass "
    "'query')."
)

HEADS = ("full", "diet", "full-legacy-workspace")


def _head(name):
    """The tool list advertised under head ``name``."""
    if name == "full":
        return list(R.TOOL_DEFINITIONS)
    if name == "diet":
        return R.apply_tool_head_diet(list(R.TOOL_DEFINITIONS))
    if name == "full-legacy-workspace":
        out = []
        for t in R.TOOL_DEFINITIONS:
            if (t.get("function") or {}).get("name") == "workspace":
                fn = dict(t["function"]); fn["description"] = WORKSPACE_DESCRIPTION_BEFORE_4FJ
                t = dict(t); t["function"] = fn
            out.append(t)
        return out
    raise ValueError(f"unknown head {name!r}; choose from {HEADS}")


def _names(tools):
    return {(t.get("function") or {}).get("name") for t in tools}


def _ok(head, picked, truth, advertised=None):
    """Scoring: exact pick; a head that advertises `tool_catalog` earns the
    designed credit when the truth is NOT among the tools it advertised
    (derived from the head actually sent, not from a name set — §4FK M6:
    `vision_analysis` sits in TOOL_HEAD_CORE but is appended only by the
    live builder, so it is absent from the bench diet head)."""
    if picked == truth:
        return True
    names = advertised if advertised is not None else _names(_head(head))
    return "tool_catalog" in names and truth not in names and picked == "tool_catalog"


def _units(rows, unit):
    """Collapse fixture rows into analysis units. `row`: one unit per row.
    `request`: one unit per distinct request text; truth = the majority
    recorded first tool among PASSED turns (all turns when none passed);
    the unit carries the merged rows' labels/origins for the strata and
    `n_rows` for the record."""
    if unit == "row":
        out = []
        for r in rows:
            out.append({"fixture_id": r.get("fixture_id"), "user_request": r.get("user_request"),
                        "truth": _truth(r), "n_rows": 1,
                        "labels": [r.get("label")], "origins": [r.get("origin")]})
        return out
    by = {}
    order = []
    for r in rows:
        key = str(r.get("user_request") or "").strip()
        if key not in by:
            by[key] = []; order.append(key)
        by[key].append(r)
    out = []
    for key in order:
        rs = by[key]
        passed = [r for r in rs if str(r.get("label")) in ("1", "1.0", "True", "true")
                  or str(r.get("outcome")) == "passed"]
        pool = passed or rs
        votes = {}
        for r in pool:
            t = _truth(r)
            if t:
                votes[t] = votes.get(t, 0) + 1
        if not votes:
            continue
        truth = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        out.append({"fixture_id": "req:" + hashlib_sha(key), "user_request": key, "truth": truth,
                    "n_rows": len(rs), "labels": [r.get("label") for r in rs],
                    "origins": [r.get("origin") for r in rs]})
    return out


def hashlib_sha(text):
    import hashlib
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:16]


def _stratum_summary(recs, A, B):
    """Paired numbers for one stratum of ledger rows."""
    n = len(recs)
    if not n:
        return {"n": 0}
    b = sum(1 for r in recs if r[f"ok_{A}"] and not r[f"ok_{B}"])
    c = sum(1 for r in recs if r[f"ok_{B}"] and not r[f"ok_{A}"])
    return {"n": n, f"acc_{A}": sum(bool(r[f"ok_{A}"]) for r in recs) / n,
            f"acc_{B}": sum(bool(r[f"ok_{B}"]) for r in recs) / n,
            f"b_{A}_only": b, f"c_{B}_only": c, "mcnemar_p": mcnemar_exact(b, c)}


def _strata(recs, A, B):
    """By recorded outcome and by origin. A request unit counts as 'passed'
    when ANY merged turn passed, and as 'user_request' when ANY merged turn
    came from a real user."""
    def has(r, key, vals):
        return any(str(x) in vals for x in (r.get(key) or []))
    return {
        "passed_turns": _stratum_summary([r for r in recs if has(r, "labels", ("1", "1.0", "True", "true"))], A, B),
        "failed_turns_only": _stratum_summary([r for r in recs if not has(r, "labels", ("1", "1.0", "True", "true"))], A, B),
        "origin_user_request": _stratum_summary([r for r in recs if has(r, "origins", ("user_request",))], A, B),
        "origin_bench_only": _stratum_summary([r for r in recs if not has(r, "origins", ("user_request",))], A, B),
    }


def _fixtures():
    for cand in ("tool_choice_fixtures.jsonl", "tool_choice_fixtures.jsonl.notready"):
        p = GHOST_HOME / "system" / "optim" / cand
        if p.exists():
            rows = []
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return rows, p
    return [], None


def _truth(fx):
    ch = fx.get("chosen_tools")
    if isinstance(ch, str):
        try:
            ch = json.loads(ch.replace("'", '"'))
        except Exception:
            return None
    if not isinstance(ch, list) or not ch or not isinstance(ch[0], dict):
        return None
    return ch[0].get("name")


def _call(messages, tools, timeout=180.0):
    body = {"messages": messages, "tools": tools, "tool_choice": "auto",
            "temperature": 0.0, "max_tokens": 512, "stream": False}
    req = urllib.request.Request(UPSTREAM, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    msg = (d.get("choices") or [{}])[0].get("message") or {}
    calls = msg.get("tool_calls") or []
    if calls:
        return (calls[0].get("function") or {}).get("name")
    # XML fallback — the model may emit <tool_call> text for an unadvertised tool
    content = str(msg.get("content") or "")
    i = content.find("<tool_call>")
    if i >= 0:
        j = content.find("{", i)
        k = content.find("</tool_call>", i)
        if j < 0:
            return "<unparsed-xml>"
        if k < 0:
            k = len(content)          # truncated at max_tokens: no closing tag (§4FK M-7)
        try:
            obj = json.loads(content[j:k].strip())
            return obj.get("name")
        except Exception:
            return "<unparsed-xml>"
    return None


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * p)


def _ledger_arms(path):
    """The pair the first row of ``path`` was written for (None = legacy)."""
    p = Path(path)
    if not p.exists():
        return None
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        return list(r["arms"]) if isinstance(r.get("arms"), list) else None
    return None


def _ledger_rows(path, arms=("full", "diet"), only=None):
    """Rows of ``path`` scored for ``arms`` (first row per fixture id),
    restricted to the fixture ids in ``only`` when given."""
    a, b_ = arms
    out, seen = [], set()
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        fid = r.get("fixture_id")
        if fid is None or fid in seen or f"ok_{a}" not in r or f"ok_{b_}" not in r:
            continue
        if only is not None and fid not in only:
            continue
        seen.add(fid)
        out.append(r)
    return out


def _load_ledger(path, arms=("full", "diet"), only=None):
    """Rows already paid for in ``path`` → (done fixture ids, hits, b, c).
    Recomputed from the rows themselves (never from a stored summary) so a
    resumed run's totals cover every row in the ledger. Missing file → empty.
    Rows carry ``ok_<A>`` / ``ok_<B>`` for the pair ``arms``; a row written
    for a different pair does not count (and is not skipped on resume)."""
    a, b_ = arms
    done, hits, b, c = set(), {a: 0, b_: 0}, 0, 0
    for r in _ledger_rows(path, arms, only):
        done.add(r["fixture_id"])
        ok_a, ok_b = bool(r.get(f"ok_{a}")), bool(r.get(f"ok_{b_}"))
        hits[a] += ok_a; hits[b_] += ok_b
        if ok_a and not ok_b: b += 1
        if ok_b and not ok_a: c += 1
    return done, hits, b, c


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60, help="0 = every fixture")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(GHOST_HOME / "system" / "eval" / "tool_head_diet"))
    ap.add_argument("--resume", default="", help="ledger .jsonl to continue (skips its fixture ids)")
    ap.add_argument("--pair", default="full,diet", help="two heads A,B from " + "|".join(HEADS))
    ap.add_argument("--unit", default="request", choices=("request", "row"),
                    help="analysis unit: distinct request (default) or fixture row (pre-§4FK)")
    args = ap.parse_args(argv)
    arms = tuple(x.strip() for x in args.pair.split(","))
    if len(arms) != 2 or arms[0] == arms[1] or any(x not in HEADS for x in arms):
        print(f"--pair must name two different heads from {HEADS}"); return 2
    A, B = arms
    rows, src = _fixtures()
    if not rows:
        print("no fixtures found"); return 2
    n_raw = len(rows)
    source_status = ("parked mine — supply gates NOT met (.notready)" if str(src).endswith(".notready")
                     else "promoted fixture pool")
    if str(src).endswith(".notready"):
        print(f"⚠ corpus is {source_status}: {src}", flush=True)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = [r for r in rows if _truth(r) and r.get("user_request")]
    rows = _units(rows, args.unit)
    n_distinct = len(rows)
    if args.limit > 0:
        rows = rows[: args.limit]
    selected_ids = {r["fixture_id"] for r in rows}
    head_a, head_b = _head(A), _head(B)
    names_a, names_b = _names(head_a), _names(head_b)
    system = SYSTEM_PROMPT.replace("{{PROFILE}}", "")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        ledger = Path(args.resume)
        stamp = ledger.stem
        prev_arms = _ledger_arms(ledger)
        if prev_arms and prev_arms != list(arms):
            print(f"refusing to resume {ledger}: it was written for pair {prev_arms}, not {list(arms)}")
            return 2
        done_ids, hits, b, c = _load_ledger(ledger, arms, only=selected_ids)
        skipped = [r for r in rows if r.get("fixture_id") in done_ids]
        rows = [r for r in rows if r.get("fixture_id") not in done_ids]
        print(f"resuming {ledger}: {len(done_ids)} rows of this selection already in the ledger, "
              f"{len(rows)} to run", flush=True)
        if not rows:
            # §4FK C2: a resume that selects nothing must not publish a
            # "finished" progress file over a partial ledger (the default
            # --limit 60 applied before the done-set was removed did exactly
            # that). Say so and stop; the caller wanted --limit 0.
            print("nothing to run for this selection — pass --limit 0 (or the original --limit/--seed) "
                  "to resume the run that wrote this ledger")
            return 2
    else:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        ledger = out_dir / f"{stamp}.jsonl"
        hits = {A: 0, B: 0}
        b = c = 0     # b: A right & B wrong; c: A wrong & B right
    if not rows and not args.resume:
        # a bench that cannot run must not report a summary (§4U)
        print(f"no runnable fixtures in {src} (rows without a recorded first tool are skipped)")
        return 2
    n_total = len(rows) + (len(done_ids) if args.resume else 0)
    # BOUNDED: the denominator is printed by the tool, from the tool's own filter.
    print(f"pair: {A} vs {B}; unit={args.unit}; {len(rows)} to run, {n_total} total in this selection "
          f"({n_distinct} units of {n_raw} rows in {src})", flush=True)
    progress = RunProgress(ledger.with_suffix(".progress.json"), n_total,
                           label=f"tool_head_diet {stamp}")
    progress.done = n_total - len(rows)
    t0 = time.time()
    with ledger.open("a" if args.resume else "w") as f:
        for i, fx in enumerate(rows, 1):
            truth = fx["truth"]
            msgs = [{"role": "system", "content": system},
                    {"role": "user", "content": str(fx["user_request"])[:6000]}]
            try:
                p_a = _call(msgs, head_a)
                p_b = _call(msgs, head_b)
            except Exception as e:  # noqa: BLE001
                print(f"[{i}] transport error: {e}"); continue
            ok_a = _ok(A, p_a, truth, names_a)
            ok_b = _ok(B, p_b, truth, names_b)
            hits[A] += ok_a; hits[B] += ok_b
            if ok_a and not ok_b: b += 1
            if ok_b and not ok_a: c += 1
            rec = {"i": i + (n_total - len(rows)), "fixture_id": fx.get("fixture_id"), "truth": truth,
                   "arms": [A, B], "unit": args.unit, "n_rows": fx.get("n_rows", 1),
                   "labels": fx.get("labels"), "origins": fx.get("origins"),
                   f"picked_{A}": p_a, f"picked_{B}": p_b, f"ok_{A}": ok_a, f"ok_{B}": ok_b,
                   "truth_hidden": truth not in names_b}
            f.write(json.dumps(rec) + "\n"); f.flush()
            progress.tick(extra={"hits": dict(hits), f"b_{A}_only": b, f"c_{B}_only": c})
            print(f"[{i}/{len(rows)}] truth={truth:<24} {A}={str(p_a):<24} {B}={str(p_b):<24} "
                  f"{'=' if ok_a == ok_b else (A.upper() + '>' if ok_a else B.upper() + '>')}", flush=True)
    done_ids, hits, b, c = _load_ledger(ledger, arms, only=selected_ids)
    n = len(done_ids)                           # every row of THIS selection in the ledger
    progress.finish(f"n={n}")
    recs = _ledger_rows(ledger, arms, only=selected_ids)
    summary = {"n": n, "unit": args.unit, "pair": [A, B], "source": str(src),
               "source_status": source_status, "seed": args.seed, "limit": args.limit,
               "distinct_units": n_distinct, "raw_rows": n_raw,
               f"acc_{A}": hits[A] / n if n else None,
               f"acc_{B}": hits[B] / n if n else None, f"b_{A}_only": b, f"c_{B}_only": c,
               "mcnemar_p": mcnemar_exact(b, c), "strata": _strata(recs, A, B),
               "seconds": round(time.time() - t0),
               "ledger": str(ledger), "advertised": {A: sorted(names_a), B: sorted(names_b)}}
    (out_dir / f"{stamp}.summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
