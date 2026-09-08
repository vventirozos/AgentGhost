#!/usr/bin/env python3
"""Combine one or more leaf_bench ledgers (one per repeat, or chunked) into
one paired summary — the deciding read for §4FG/§4FI.

usage: leaf_bench_combine.py LEDGER.jsonl [LEDGER2.jsonl ...]
A pair is (ledger, rep, leaf) with BOTH executors present. Exact McNemar on
DONE outcomes; per-executor DONE rate and mean seconds; a per-leaf table;
the disagreements. Rows without a `rep` field (single-repeat ledgers written
before §4FI) count as rep 0 of their ledger.
"""
import json
import sys
from collections import defaultdict
from math import comb


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n))


def combine(paths):
    by = defaultdict(dict)
    for li, p in enumerate(paths):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by[(li, int(r.get("rep", 0) or 0), r["leaf"])][r["kind"]] = r
    kinds = ("spec", "agentic")
    pairs = {k: v for k, v in by.items() if all(x in v for x in kinds)}
    n = len(pairs)
    done = {x: sum(1 for v in pairs.values() if v[x]["done"]) for x in kinds}
    secs = {x: (round(sum(float(v[x]["seconds"] or 0) for v in pairs.values()) / n, 1) if n else None)
            for x in kinds}
    b = sum(1 for v in pairs.values() if v["spec"]["done"] and not v["agentic"]["done"])
    c = sum(1 for v in pairs.values() if v["agentic"]["done"] and not v["spec"]["done"])
    per_leaf = defaultdict(lambda: {"n": 0, "spec_done": 0, "agentic_done": 0, "spec_s": 0.0, "agentic_s": 0.0})
    for (_, _, leaf), v in pairs.items():
        t = per_leaf[leaf]
        t["n"] += 1
        t["spec_done"] += int(bool(v["spec"]["done"]))
        t["agentic_done"] += int(bool(v["agentic"]["done"]))
        t["spec_s"] += float(v["spec"]["seconds"] or 0)
        t["agentic_s"] += float(v["agentic"]["seconds"] or 0)
    for t in per_leaf.values():
        t["spec_s"] = round(t["spec_s"] / t["n"], 1)
        t["agentic_s"] = round(t["agentic_s"] / t["n"], 1)
    disagreements = [{"ledger": k[0], "rep": k[1], "leaf": k[2],
                      "spec": v["spec"]["status"], "agentic": v["agentic"]["status"],
                      "spec_summary": str(v["spec"].get("summary"))[:80],
                      "agentic_summary": str(v["agentic"].get("summary"))[:80]}
                     for k, v in sorted(pairs.items())
                     if bool(v["spec"]["done"]) != bool(v["agentic"]["done"])]
    return {"ledgers": len(paths), "pairs": n,
            "done_rate": {x: (done[x] / n if n else None) for x in kinds}, "done": done,
            "mean_seconds": secs,
            "mcnemar": {"spec_only": b, "agentic_only": c, "p": mcnemar_exact(b, c)},
            "per_leaf": dict(per_leaf), "disagreements": disagreements}


def main(paths):
    print(json.dumps(combine(paths), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
